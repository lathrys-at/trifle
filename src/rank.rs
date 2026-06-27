//! Ranking — IDF-weighted bit-sliced overlap counting (the candidate generator *is* the
//! ranking) and the pluggable [`Ranker`] for an optional custom reorder.
//!
//! trifle is a **fuzzy lexical overlap engine, not a relevance engine**. It ranks by
//! IDF-weighted token overlap, computed in one pass: read each selected token's posting (no
//! decode — an owned roaring posting *is* the bitmap) and accumulate it into a **bit-sliced
//! counter**, weighted by the token's rarity. Each id's running count is held in binary
//! across bitmap "bit planes"; adding `w` copies of a posting injects a carry at each set bit
//! of `w` and ripples it up (XOR = sum bit, AND = carry), so the accumulation stays
//! `O(k·log k)` bitmap ops (popcount(w) ≤ 2 ripples per token for weights in 1..=4),
//! independent of posting size. A high→low bucket walk hydrates provenance bucket-by-bucket
//! and early-stops once `limit` results lock.
//!
//! **The weights** are a per-query, df-anchored 4-tier scheme (weights `{1,2,3,4}`): for the
//! query's survivors, the most-common (least discriminative) gram gets weight 1 and rarer
//! grams get more, spaced in absolute df-doublings (`log2(df_max/df_i)`). Because IDF *gaps*
//! are `N`-independent (`log(N/df_i) − log(N/df_j) = log(df_j/df_i)`), this needs no corpus
//! size, nothing stored, and nothing precomputed — just each survivor posting's cardinality
//! (its df, by the monotonic-id contract — read straight off the in-hand bitmap, not re-fetched)
//! and one knob `D` (df-doublings per weight step;
//! [`weight_step`](crate::SearchOpts::weight_step)).

use crate::hash::FxHashMap;
use std::rc::Rc;

use roaring::RoaringBitmap;
use rusqlite::Connection;
use rusqlite::types::Value;

use crate::error::Result;
use crate::instrument::trace_debug;
use crate::model::{Key, KeyShape};
use crate::store::Namespace;

/// One ranked segment that survived candidate generation: its provenance (the caller
/// key and the segment label), the internal doc id it dedups on, the segment id used for
/// hydration, its IDF-weighted score and raw overlap count, and (once hydrated) its text.
pub(crate) struct Survivor {
    pub key: Key,
    pub label: String,
    /// Internal document id — the dedup unit (one survivor per document).
    pub doc: u32,
    pub seg_id: u32,
    /// The IDF-weighted overlap score (the bit-sliced bucket value) — the ordering key.
    pub score: u32,
    /// The raw count of distinct selected tokens shared (unweighted); the `min_shared`
    /// floor is enforced against this, and [`Candidate::overlap`] returns it.
    pub overlap: u32,
    /// The matched segment's text — empty until [`hydrate_text`](crate::Index) fills it
    /// (every indexed field is stored, so a survivor's text is always populated before
    /// ranking).
    pub text: String,
    /// The segment's gram length (`|d|`), kept for a custom [`Ranker`] — `0` until hydrated
    /// alongside the text.
    pub len: u32,
}

/// Add `w` copies of `posting` into the bit-sliced planes `acc` (BSI weighted
/// accumulation): inject a carry at each set bit of `w` and ripple it up
/// (XOR = sum bit, AND = carry). Cost is `popcount(w)` ripples — for `w` in `1..=4` that
/// is ≤ 2, vs 1 for an unweighted add.
fn add_weighted(acc: &mut Vec<RoaringBitmap>, posting: &RoaringBitmap, w: u32) {
    let mut bit = 0u32;
    while (w >> bit) != 0 {
        if (w >> bit) & 1 == 1 {
            // Inject 2^bit · posting at plane `bit` and ripple the carry upward. `bit` can
            // start past the current top plane, so grow the planes up to `level` (not just by
            // one) before indexing.
            let mut carry = posting.clone();
            let mut level = bit as usize;
            while !carry.is_empty() {
                while acc.len() <= level {
                    acc.push(RoaringBitmap::new());
                }
                let new_carry = &acc[level] & &carry; // carry-out = already-set AND incoming
                acc[level] ^= &carry; // sum bit at this plane
                carry = new_carry;
                level += 1;
            }
        }
        bit += 1;
    }
}

/// IDF-weighted bit-sliced overlap counting. Returns the count "bit planes" `acc` where a
/// given id's weighted score is `Σ_b 2^b · [id ∈ acc[b]]`. Each posting contributes its
/// IDF-tier weight (`weights[i]`), accumulated via [`add_weighted`], so the per-id weighted
/// overlap for *every* id is computed in `O(k·log k)` bitmap ops — `O(containers)`,
/// independent of posting size. `weights` is parallel to `bitmaps`.
fn weighted_overlap(bitmaps: &[&RoaringBitmap], weights: &[u32]) -> Vec<RoaringBitmap> {
    let mut acc: Vec<RoaringBitmap> = Vec::new();
    for (b, &w) in bitmaps.iter().zip(weights) {
        add_weighted(&mut acc, b, w);
    }
    acc
}

/// The per-query, df-anchored IDF tier weight `{1,2,3,4}` for each posting, from the
/// survivors' df (posting cardinality). The most-common survivor (`df_max`) gets weight 1;
/// rarer grams get more, spaced in df-doublings: `1 + min(3, round(log2(df_max/df_i) / D))`.
/// `D` (> 0) is the df-doublings per weight step; `weights[i]` is parallel to `bitmaps`.
/// `N`-free by construction — IDF *gaps* don't depend on corpus size.
fn tier_weights(bitmaps: &[&RoaringBitmap], d: f64) -> Vec<u32> {
    let d = if d > 0.0 { d } else { 1.0 };
    let df_max = bitmaps.iter().map(|b| b.len()).max().unwrap_or(1).max(1) as f64;
    bitmaps
        .iter()
        .map(|b| {
            let df = b.len().max(1) as f64;
            // df ≤ df_max ⇒ ratio ≥ 1 ⇒ steps ≥ 0; cap at 3 so weight ∈ 1..=4.
            let steps = ((df_max / df).log2() / d).round().max(0.0) as u32;
            1 + steps.min(3)
        })
        .collect()
}

/// The ids whose bit-sliced overlap count is exactly `c`: AND the planes `c` has
/// set, then subtract every plane it has clear — an id survives iff its plane
/// membership is exactly `c`'s bit pattern. `c == 0` selects nothing.
fn count_eq(acc: &[RoaringBitmap], c: u32) -> RoaringBitmap {
    if c == 0 {
        return RoaringBitmap::new();
    }
    // A count whose highest set bit is beyond the planes can't exist.
    if 32 - c.leading_zeros() > acc.len() as u32 {
        return RoaringBitmap::new();
    }
    let set: Vec<usize> = (0..acc.len()).filter(|&b| (c >> b) & 1 == 1).collect();
    let Some((&first, rest)) = set.split_first() else {
        return RoaringBitmap::new();
    };
    let mut out = acc[first].clone();
    for &b in rest {
        out &= &acc[b];
    }
    for (b, plane) in acc.iter().enumerate() {
        if (c >> b) & 1 == 0 {
            out -= plane;
        }
    }
    out
}

/// A compiled Tier-2 structured filter: a `WHERE` fragment with `?` placeholders and its
/// bound params (from [`Filter::compile`](crate::Filter)). [`overlap_search`] applies it
/// **scoped to each bucket's candidate doc ids** (`WHERE id IN rarray(...) AND (<fragment>)`),
/// so the filter's cost is bounded by the hydrated pool, never an O(N) scan of the whole
/// corpus (audit T5 / I12).
pub(crate) struct CompiledFilter<'a> {
    pub where_sql: &'a str,
    pub params: &'a [Value],
}

/// The subset of `cand` (candidate doc ids) that pass `filter`, found with one
/// `SELECT id FROM doc WHERE id IN rarray(?1) AND (<filter>)` — the scope restriction binds as
/// `?1`, so the filter's own bare `?` placeholders number from `?2` (SQLite assigns a bare `?`
/// one past the largest number already seen). O(candidates), not O(corpus).
fn filter_pass(
    conn: &Connection,
    ns: &Namespace,
    filter: &CompiledFilter<'_>,
    cand: &[u32],
) -> Result<RoaringBitmap> {
    let mut out = RoaringBitmap::new();
    if cand.is_empty() {
        return Ok(out);
    }
    let arr: Rc<Vec<Value>> = Rc::new(cand.iter().map(|&i| Value::Integer(i as i64)).collect());
    let sql = format!(
        "SELECT id FROM {doc} WHERE id IN rarray(?1) AND ({where_sql})",
        doc = ns.doc(),
        where_sql = filter.where_sql,
    );
    let mut stmt = conn.prepare_cached(&sql)?;
    let mut binds: Vec<&dyn rusqlite::ToSql> = Vec::with_capacity(1 + filter.params.len());
    binds.push(&arr);
    for p in filter.params {
        binds.push(p);
    }
    let mut rows = stmt.query(binds.as_slice())?;
    while let Some(r) = rows.next()? {
        out.insert(r.get::<_, i64>(0)? as u32);
    }
    Ok(out)
}

/// The knobs [`overlap_search`] reads, bundled so candidate generation takes a short argument
/// list (the data — `conn`/`ns`/`present` — stays separate). All fields are batch-constant, so
/// the caller builds this once and reuses it across a batch's queries.
#[derive(Clone, Copy)]
pub(crate) struct OverlapParams<'a> {
    /// The candidate pool depth (`limit`, possibly [`Effort`](crate::Effort)-deepened).
    pub limit: usize,
    /// The raw (unweighted) overlap floor `m`.
    pub min_shared: u32,
    /// `D` — df-doublings per IDF weight step ([`weight_step`](crate::SearchOpts::weight_step)).
    pub weight_step: f64,
    /// The caller key's stored shape, for provenance hydration.
    pub key_shape: KeyShape,
    /// The compiled Tier-2 filter, applied scoped to candidate ids (`None` = unfiltered).
    pub filter: Option<&'a CompiledFilter<'a>>,
    /// The optional provenance scope predicate.
    pub scope: Option<&'a crate::ScopeFn<'a>>,
}

/// Generate and rank candidates by IDF-weighted overlap, hydrating only as deep as the
/// top-`limit` needs.
///
/// Weights each selected token by rarity ([`tier_weights`], knob `D = weight_step`),
/// accumulates the weighted bit-sliced counter, then walks the weighted-score buckets
/// high → low. Each bucket hydrates its ids' provenance in one batched read, applies the
/// Tier-2 filter and scope predicate, and records the best segment per doc (highest weighted
/// score; lowest id as the tie-break). Once `limit` docs lock at a score, no lower bucket can
/// displace them, so the walk stops — bounding hydration to the high-score head. An id in a
/// posting with no `seg` row (a monotonic-id segment deleted since the last fold) does not
/// hydrate, so it never ranks.
///
/// The `min_shared` floor stays a **raw** token count: since every weight ≥ 1, a candidate's
/// weighted score is ≥ its raw overlap, so the walk can stop at weighted score `floor` and
/// still need only a per-candidate raw-count check to drop high-weight/low-overlap ids.
pub(crate) fn overlap_search(
    conn: &Connection,
    ns: &Namespace,
    present: &[(&str, &RoaringBitmap)],
    params: &OverlapParams<'_>,
) -> Result<Vec<Survivor>> {
    let &OverlapParams {
        limit,
        min_shared,
        weight_step,
        key_shape,
        filter,
        scope,
    } = params;
    let bitmaps: Vec<&RoaringBitmap> = present.iter().map(|(_, b)| *b).collect();
    if bitmaps.is_empty() || limit == 0 {
        return Ok(Vec::new());
    }
    // The per-query overlap floor: a candidate must share at least `min(m, |present|)`
    // selected tokens (raw, unweighted). Basing it on the postings actually present (not the
    // raw selection) means appended absent tokens never inflate it, so a query whose only
    // present token is a single n-gram still ranks at overlap 1.
    let floor = (min_shared as usize).min(bitmaps.len()).max(1) as u32;
    let weights = tier_weights(&bitmaps, weight_step);
    let max_score = weights.iter().sum::<u32>();
    // Raw overlap of `id` — how many present postings contain it (the unweighted count the
    // `min_shared` floor is enforced against).
    let raw_overlap = |id: u32| -> u32 { bitmaps.iter().filter(|b| b.contains(id)).count() as u32 };

    // The `Σ kept-posting cardinality` instrumentation: computed only when
    // the `tracing` feature is on (the macro does not evaluate its arguments
    // otherwise), so the hot path pays nothing for it by default.
    trace_debug!(
        postings = bitmaps.len(),
        sum_cardinality = bitmaps.iter().map(|b| b.len()).sum::<u64>(),
        floor,
        max_score,
        "trifle: weighted overlap candidate generation"
    );

    let acc = weighted_overlap(&bitmaps, &weights);

    // internal doc id -> its best (highest weighted score, lowest-seg-id) segment.
    let mut best: FxHashMap<u32, Survivor> = FxHashMap::default();
    // Memoized Tier-2 verdict per candidate doc id. The filter is evaluated lazily, scoped to
    // the candidate ids each bucket hydrates (one small `WHERE id IN rarray(...)` per bucket),
    // so its total cost is bounded by the hydrated pool — not the corpus (audit T5 / I12).
    let mut filter_memo: FxHashMap<u32, bool> = FxHashMap::default();
    for c in (floor..=max_score).rev() {
        let bucket = count_eq(&acc, c);
        if !bucket.is_empty() {
            let ids: Vec<u32> = bucket.iter().collect();
            let provenance = hydrate_provenance(conn, ns, &ids, key_shape)?;
            // Resolve the Tier-2 filter for this bucket's not-yet-classified candidate docs in
            // one scoped query, caching the verdict so a doc with multiple segments (or a doc
            // seen in a later bucket) is never re-queried.
            if let Some(cf) = filter {
                let unseen: Vec<u32> = provenance
                    .values()
                    .map(|(doc, _, _)| *doc)
                    .filter(|d| !filter_memo.contains_key(d))
                    .collect::<std::collections::BTreeSet<u32>>()
                    .into_iter()
                    .collect();
                if !unseen.is_empty() {
                    let passing = filter_pass(conn, ns, cf, &unseen)?;
                    for d in unseen {
                        filter_memo.insert(d, passing.contains(d));
                    }
                }
            }
            // `ids` is ascending (bitmap order), so the first segment recorded for a
            // doc is its lowest id at this — its highest — score.
            for id in ids {
                let Some((doc, key, label)) = provenance.get(&id) else {
                    continue; // posting id with no live segment row — skip
                };
                // A high-weight rare gram alone can reach score ≥ floor with raw overlap
                // below the floor; enforce the raw `min_shared` floor per candidate.
                let overlap = raw_overlap(id);
                if overlap < floor {
                    continue;
                }
                // Tier-2 filter: keep only docs passing the structured filter (verdict memoized
                // above; a filtered search records a verdict for every candidate doc).
                if filter.is_some() && !filter_memo.get(doc).copied().unwrap_or(false) {
                    continue;
                }
                if let Some(scope) = scope {
                    if !scope(key, label) {
                        continue;
                    }
                }
                best.entry(*doc).or_insert_with(|| Survivor {
                    key: key.clone(),
                    label: label.clone(),
                    doc: *doc,
                    seg_id: id,
                    score: c,
                    overlap,
                    text: String::new(),
                    len: 0,
                });
            }
        }
        if best.len() >= limit {
            break;
        }
    }

    let mut survivors: Vec<Survivor> = best.into_values().collect();
    // Weighted score descending, then internal doc id ascending — stable, deterministic.
    survivors.sort_by(|a, b| b.score.cmp(&a.score).then(a.doc.cmp(&b.doc)));
    survivors.truncate(limit);
    trace_debug!(
        survivors = survivors.len(),
        "trifle: weighted overlap survivors locked"
    );
    Ok(survivors)
}

/// Provenance `(internal doc id, caller key, label)` per segment id, for a set of ids,
/// in one batched `WHERE s.id IN rarray(?1)` read joining `seg` to `doc` — no temp btree,
/// one prepared statement.
fn hydrate_provenance(
    conn: &Connection,
    ns: &Namespace,
    ids: &[u32],
    key_shape: KeyShape,
) -> Result<FxHashMap<u32, (u32, Key, String)>> {
    let mut out = FxHashMap::with_capacity_and_hasher(ids.len(), Default::default());
    if ids.is_empty() {
        return Ok(out);
    }
    let arr: Rc<Vec<Value>> = Rc::new(ids.iter().map(|&i| Value::Integer(i as i64)).collect());
    let sql = format!(
        "SELECT s.id, s.doc_id, s.label, d.key FROM {seg} s \
         JOIN {doc} d ON d.id = s.doc_id WHERE s.id IN rarray(?1)",
        seg = ns.seg(),
        doc = ns.doc(),
    );
    let mut stmt = conn.prepare_cached(&sql)?;
    let mut rows = stmt.query(rusqlite::params![arr])?;
    while let Some(r) = rows.next()? {
        let seg_id: i64 = r.get(0)?;
        let doc: i64 = r.get(1)?;
        let label: String = r.get(2)?;
        let kv: Value = r.get(3)?;
        out.insert(
            seg_id as u32,
            (doc as u32, Key::from_value(key_shape, kv)?, label),
        );
    }
    Ok(out)
}

/// Optionally reorders the IDF-weighted-overlap survivors into a different final order.
///
/// trifle ranks by IDF-weighted lexical overlap in the counter itself, so the **default**
/// [`OverlapRanker`] just preserves that order — there is no built-in relevance/BM25 tier.
/// A custom ranker can reorder for a domain need (proximity, true term-frequency,
/// exact-substring promotion, …) over the segment text each candidate carries. The signals
/// it can read are public: [`Candidate::overlap`] / [`Candidate::score`],
/// [`Candidate::matched_terms`] (each matched token with its `df`), [`Candidate::seg_len`],
/// and [`QueryContext::n_segments`] / [`QueryContext::avgdl`]. Over-fetch a deeper pool with
/// [`Effort`](crate::Effort) so the reorder has candidates to pull up.
///
/// **The unit is the segment.** Candidate generation reduces each document to its single
/// best-matching segment before the ranker runs, so a ranker sees one candidate per document
/// and *cannot* fuse a document's multiple segments — do cross-segment fusion *above* trifle
/// by aggregating results across your own keys.
pub trait Ranker: Send + Sync {
    /// Return the candidates in final result order (best first). May drop candidates
    /// by omitting them; trifle truncates the result to the search limit.
    fn rank(&self, candidates: &Candidates<'_>, query: &QueryContext<'_>) -> Vec<Ranked>;
}

/// The default ranker: preserve the IDF-weighted-overlap order the bit-sliced counter
/// already produced. Reads nothing — the candidates arrive weighted-score-descending and
/// this is the identity over them, so it is effectively free.
#[derive(Clone, Copy, Debug, Default)]
pub struct OverlapRanker;

impl Ranker for OverlapRanker {
    fn rank(&self, candidates: &Candidates<'_>, _query: &QueryContext<'_>) -> Vec<Ranked> {
        (0..candidates.len())
            .map(|candidate| Ranked { candidate })
            .collect()
    }
}

/// Context about the query, passed to a [`Ranker`].
pub struct QueryContext<'a> {
    /// The original query text.
    pub query: &'a str,
    /// The selected token strings, in scan order.
    pub selected: &'a [String],
    /// The match floor `m` in effect for this query.
    pub min_shared: u32,
    /// Total live segments `N`. Not used by the default overlap ranking (IDF *gaps* are
    /// `N`-independent); provided for a custom [`Ranker`] that wants a corpus-relative score.
    pub n_segments: u64,
    /// Mean segment gram length (`avgdl`). Not used by the default overlap ranking; provided
    /// for a custom [`Ranker`] doing length normalization. `0.0` on an empty corpus.
    pub avgdl: f64,
}

/// The overlap-counted survivors handed to a [`Ranker`], with their text already
/// hydrated (only `limit` of them — the fast-path contract).
pub struct Candidates<'a> {
    survivors: &'a [Survivor],
    /// The query's selected tokens that have a posting, paired with it — used for
    /// [`Candidate::matched_tokens`].
    present: &'a [(&'a str, &'a RoaringBitmap)],
}

impl<'a> Candidates<'a> {
    pub(crate) fn new(
        survivors: &'a [Survivor],
        present: &'a [(&'a str, &'a RoaringBitmap)],
    ) -> Self {
        Candidates { survivors, present }
    }

    /// The number of candidates.
    pub fn len(&self) -> usize {
        self.survivors.len()
    }

    /// Whether there are no candidates.
    pub fn is_empty(&self) -> bool {
        self.survivors.is_empty()
    }

    /// The candidate at `index`, if any.
    pub fn get(&self, index: usize) -> Option<Candidate<'_>> {
        self.survivors.get(index).map(|s| Candidate {
            s,
            present: self.present,
            index,
        })
    }

    /// Iterate the candidates in weighted-overlap order.
    pub fn iter(&self) -> impl Iterator<Item = Candidate<'_>> {
        (0..self.len()).map(move |i| self.get(i).expect("index in range"))
    }

    /// The query's selected tokens that have a posting, each paired with its document
    /// frequency `df` (the number of segments containing it) — the rarity inputs a custom
    /// [`Ranker`] needs, without any store read (the postings are already in hand).
    pub fn present_terms(&self) -> impl Iterator<Item = (&str, u64)> {
        self.present.iter().map(|(t, bm)| (*t, bm.len()))
    }
}

/// One candidate exposed to a [`Ranker`].
pub struct Candidate<'a> {
    s: &'a Survivor,
    present: &'a [(&'a str, &'a RoaringBitmap)],
    index: usize,
}

impl Candidate<'_> {
    /// This candidate's index within the [`Candidates`] set (what [`Ranked`] refers to).
    pub fn index(&self) -> usize {
        self.index
    }
    /// The caller's document key.
    pub fn key(&self) -> &Key {
        &self.s.key
    }
    /// The matched segment's label (the text field name).
    pub fn label(&self) -> &str {
        &self.s.label
    }
    /// How many selected tokens this candidate shares (the **raw**, unweighted count).
    pub fn overlap(&self) -> u32 {
        self.s.overlap
    }
    /// This candidate's IDF-weighted overlap score — the value trifle ranks by (the
    /// bit-sliced bucket it locked in). Weighted by per-query gram rarity; see the module
    /// docs for the tiering.
    pub fn score(&self) -> u32 {
        self.s.score
    }
    /// The matched segment's text. Every indexed field is stored, so this is always the
    /// segment's full text.
    pub fn text(&self) -> &str {
        &self.s.text
    }
    /// The matched segment's gram length `|d|` (token count with repetition). Not used by the
    /// default overlap ranking; provided for a custom [`Ranker`] doing length normalization.
    /// `0` only for a sub-n-gram segment that produced no tokens.
    pub fn seg_len(&self) -> u32 {
        self.s.len
    }
    /// Which selected tokens this candidate's segment actually contains.
    pub fn matched_tokens(&self) -> Vec<&str> {
        self.present
            .iter()
            .filter(|(_, bm)| bm.contains(self.s.seg_id))
            .map(|(t, _)| *t)
            .collect()
    }
    /// Which selected tokens this candidate's segment contains, each paired with its document
    /// frequency `df` (segments containing it) — the signals a custom [`Ranker`] needs to
    /// compute its own rarity-weighted score (combine with [`QueryContext::n_segments`] /
    /// [`seg_len`](Self::seg_len) / [`QueryContext::avgdl`] as wanted).
    pub fn matched_terms(&self) -> impl Iterator<Item = (&str, u64)> {
        self.present
            .iter()
            .filter(|(_, bm)| bm.contains(self.s.seg_id))
            .map(|(t, bm)| (*t, bm.len()))
    }
}

/// A [`Ranker`]'s placement of one candidate: its index within the [`Candidates`]
/// set. The position of a `Ranked` in the returned `Vec` is its result rank.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ranked {
    /// The index of the chosen candidate within [`Candidates`].
    pub candidate: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bm(ids: &[u32]) -> RoaringBitmap {
        ids.iter().copied().collect()
    }

    /// Unweighted accumulation (every weight 1) — exercises the counter + `count_eq` the same
    /// way the old `bitsliced_overlap` did, so the plane/boundary invariants stay covered.
    fn unweighted(bitmaps: &[&RoaringBitmap]) -> Vec<RoaringBitmap> {
        let weights = vec![1u32; bitmaps.len()];
        weighted_overlap(bitmaps, &weights)
    }

    #[test]
    fn bitsliced_counts_match_naive_accumulation() {
        // ids: 1 in three postings, 2 in two, 3 in one.
        let a = bm(&[1, 2, 3]);
        let b = bm(&[1, 2]);
        let c = bm(&[1]);
        let acc = unweighted(&[&a, &b, &c]);
        assert_eq!(count_eq(&acc, 3).iter().collect::<Vec<_>>(), [1]);
        assert_eq!(count_eq(&acc, 2).iter().collect::<Vec<_>>(), [2]);
        assert_eq!(count_eq(&acc, 1).iter().collect::<Vec<_>>(), [3]);
        assert!(count_eq(&acc, 4).is_empty());
        assert!(count_eq(&acc, 0).is_empty());
    }

    #[test]
    fn bitsliced_handles_counts_above_a_single_plane() {
        // Five postings all containing id 7 -> count 5 (binary 101, three planes).
        let posts: Vec<RoaringBitmap> = (0..5).map(|_| bm(&[7])).collect();
        let refs: Vec<&RoaringBitmap> = posts.iter().collect();
        let acc = unweighted(&refs);
        assert_eq!(count_eq(&acc, 5).iter().collect::<Vec<_>>(), [7]);
        for c in [1, 2, 3, 4, 6] {
            assert!(count_eq(&acc, c).is_empty(), "count {c} should be empty");
        }
    }

    #[test]
    fn count_eq_is_exact_membership_not_at_least() {
        let a = bm(&[1, 2]);
        let b = bm(&[2, 3]);
        let acc = unweighted(&[&a, &b]);
        // id 2 has count 2; ids 1,3 have count 1.
        assert_eq!(count_eq(&acc, 2).iter().collect::<Vec<_>>(), [2]);
        assert_eq!(count_eq(&acc, 1).iter().collect::<Vec<_>>(), [1, 3]);
    }

    #[test]
    fn count_eq_exact_across_a_plane_boundary() {
        // Counts 8 (0b1000), 7 (0b0111), 4 (0b100) straddle the 3->4 plane growth.
        // A high-count id must be subtracted out of every lower-count query.
        let mut posts: Vec<RoaringBitmap> = Vec::new();
        for k in 0..8 {
            // id 100 in all 8; id 200 in 7; id 300 in 4.
            let mut p = bm(&[100]);
            if k < 7 {
                p.insert(200);
            }
            if k < 4 {
                p.insert(300);
            }
            posts.push(p);
        }
        let refs: Vec<&RoaringBitmap> = posts.iter().collect();
        let acc = unweighted(&refs);
        assert_eq!(count_eq(&acc, 8).iter().collect::<Vec<_>>(), [100]);
        assert_eq!(count_eq(&acc, 7).iter().collect::<Vec<_>>(), [200]);
        assert_eq!(count_eq(&acc, 4).iter().collect::<Vec<_>>(), [300]);
        // The exactness that matters: count-4 must NOT contain the count-7/8 ids.
        for c in [1, 2, 3, 5, 6] {
            assert!(count_eq(&acc, c).is_empty(), "no id has count {c}");
        }
    }

    #[test]
    fn count_eq_plane_guard_rejects_counts_beyond_the_planes() {
        // 7 postings sharing id 1 -> count 7, three planes (0b111).
        let posts: Vec<RoaringBitmap> = (0..7).map(|_| bm(&[1])).collect();
        let refs: Vec<&RoaringBitmap> = posts.iter().collect();
        let acc = unweighted(&refs);
        assert_eq!(acc.len(), 3);
        assert_eq!(count_eq(&acc, 7).iter().collect::<Vec<_>>(), [1]);
        // 8 needs a 4th plane that doesn't exist -> the guard returns empty (no panic).
        assert!(count_eq(&acc, 8).is_empty());
        assert!(count_eq(&acc, 15).is_empty());
    }

    #[test]
    fn count_eq_accepts_a_power_of_two_when_the_plane_exists() {
        // 8 postings sharing id 1 -> count 8 (0b1000), exactly four planes.
        let posts: Vec<RoaringBitmap> = (0..8).map(|_| bm(&[1])).collect();
        let refs: Vec<&RoaringBitmap> = posts.iter().collect();
        let acc = unweighted(&refs);
        assert_eq!(acc.len(), 4);
        assert_eq!(count_eq(&acc, 8).iter().collect::<Vec<_>>(), [1]);
    }

    #[test]
    fn bitsliced_overlap_degenerate_inputs() {
        // No postings -> no planes.
        assert!(unweighted(&[]).is_empty());
        // A single posting -> count 1 for each of its ids, nothing at count 2.
        let single = bm(&[5, 9]);
        let acc = unweighted(&[&single]);
        assert_eq!(count_eq(&acc, 1).iter().collect::<Vec<_>>(), [5, 9]);
        assert!(count_eq(&acc, 2).is_empty());
        // An empty posting contributes nothing and pushes no phantom plane.
        let empty = RoaringBitmap::new();
        assert!(unweighted(&[&empty]).is_empty());
    }

    #[test]
    fn weighted_add_scales_the_count() {
        // One posting added with weight 3 gives every id count 3; a second weight-2 posting
        // bumps the shared id to 5.
        let a = bm(&[1, 2]);
        let b = bm(&[2]);
        let acc = weighted_overlap(&[&a, &b], &[3, 2]);
        assert_eq!(count_eq(&acc, 3).iter().collect::<Vec<_>>(), [1]); // only in `a` (w=3)
        assert_eq!(count_eq(&acc, 5).iter().collect::<Vec<_>>(), [2]); // in both: 3 + 2
        assert!(count_eq(&acc, 2).is_empty());
    }

    #[test]
    fn weighted_overlap_matches_repeated_unweighted_adds() {
        // add_weighted(w) must equal adding the same posting w times unweighted.
        let p = bm(&[1, 4, 9]);
        let q = bm(&[4]);
        let weighted = weighted_overlap(&[&p, &q], &[4, 1]);
        let repeated = unweighted(&[&p, &p, &p, &p, &q]);
        for c in 1..=5u32 {
            assert_eq!(
                count_eq(&weighted, c).iter().collect::<Vec<_>>(),
                count_eq(&repeated, c).iter().collect::<Vec<_>>(),
                "weighted vs repeated disagree at count {c}"
            );
        }
    }

    #[test]
    fn tier_weights_anchor_at_df_max_and_cap_at_four() {
        // df_max anchor (commonest -> weight 1); each df-doubling adds a step (D=1); cap at 4.
        // dfs: 16 (df_max -> 1), 8 (1 doubling -> 2), 4 (2 -> 3), 1 (4 doublings -> capped 4).
        let common = bm(&(0..16).collect::<Vec<_>>());
        let mid = bm(&(0..8).collect::<Vec<_>>());
        let rarer = bm(&(0..4).collect::<Vec<_>>());
        let rarest = bm(&[0]);
        let w = tier_weights(&[&common, &mid, &rarer, &rarest], 1.0);
        assert_eq!(w, vec![1, 2, 3, 4]);
    }

    #[test]
    fn tier_weights_collapse_when_the_band_is_within_one_doubling() {
        // A compressed band (all within one df-doubling of df_max) is all tier 1 — no
        // manufactured spread.
        let a = bm(&(0..10).collect::<Vec<_>>());
        let b = bm(&(0..9).collect::<Vec<_>>());
        let c = bm(&(0..8).collect::<Vec<_>>());
        assert_eq!(tier_weights(&[&a, &b, &c], 1.0), vec![1, 1, 1]);
    }

    #[test]
    fn tier_weights_step_widens_with_d() {
        // A larger D needs more doublings per step: with D=2, an 8x-rarer gram (3 doublings)
        // is only round(3/2)=2 steps -> weight 3 (vs weight 4 at D=1).
        let common = bm(&(0..8).collect::<Vec<_>>());
        let rarer = bm(&[0]); // 8x rarer => 3 doublings
        assert_eq!(tier_weights(&[&common, &rarer], 1.0), vec![1, 4]);
        assert_eq!(tier_weights(&[&common, &rarer], 2.0), vec![1, 3]);
    }
}
