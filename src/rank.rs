//! Ranking — bit-sliced overlap counting (fixed candidate generation) and the
//! pluggable [`Ranker`] that orders the survivors.
//!
//! Candidate generation is fixed and fast: read each selected token's posting (no
//! decode — an owned roaring posting *is* the bitmap) and count overlap with a
//! **bit-sliced counter**. Each id's count is held in binary across bitmap "bit
//! planes"; adding a posting is a ripple-carry binary add at the bitmap level
//! (XOR = sum bit, AND = carry), so the whole accumulation is `O(k·log k)` bitmap
//! ops — independent of posting size. A high→low bucket walk hydrates provenance
//! bucket-by-bucket and early-stops once `limit` results lock.

use std::collections::HashMap;
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
/// hydration, its overlap count, and (once hydrated) its text.
pub(crate) struct Survivor {
    pub key: Key,
    pub label: String,
    /// Internal document id — the dedup unit (one survivor per document).
    pub doc: u32,
    pub seg_id: u32,
    pub overlap: u32,
    /// The matched segment's text — empty until [`hydrate_text`](crate::Index) fills it
    /// (every indexed field is stored, so a survivor's text is always populated before
    /// ranking).
    pub text: String,
    /// The segment's gram length (`|d|`) for BM25+ length normalization — `0` until
    /// hydrated alongside the text.
    pub len: u32,
}

/// Bit-sliced overlap counting. Returns the count "bit planes" `acc` where a given
/// id's overlap count is `Σ_b 2^b · [id ∈ acc[b]]`. Adds each input bitmap with a
/// ripple-carry across the planes, so the per-id overlap for *every* id is computed
/// in `O(k·log k)` bitmap ops — `O(containers)`, independent of posting size.
fn bitsliced_overlap(bitmaps: &[&RoaringBitmap]) -> Vec<RoaringBitmap> {
    let mut acc: Vec<RoaringBitmap> = Vec::new();
    for &b in bitmaps {
        let mut carry = b.clone();
        let mut level = 0usize;
        while !carry.is_empty() {
            if level == acc.len() {
                acc.push(RoaringBitmap::new());
            }
            let new_carry = &acc[level] & &carry; // carry-out = already-set AND incoming
            acc[level] ^= &carry; // sum bit at this plane
            carry = new_carry;
            level += 1;
        }
    }
    acc
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

/// Generate and rank candidates by overlap, hydrating only as deep as the top-`limit`
/// needs.
///
/// Walks the overlap-count buckets high → low; each bucket hydrates its ids'
/// provenance in one batched read, applies the scope predicate, and records the best
/// segment per doc (highest count; lowest id as the tie-break). Once `limit` docs are
/// locked at a count, no lower bucket can displace them, so the walk stops — bounding
/// hydration to the high-overlap head. An id present in a posting but with no `seg`
/// row (a monotonic-id segment deleted since the last fold) simply does not hydrate,
/// so it never ranks.
#[allow(clippy::too_many_arguments)]
pub(crate) fn overlap_search(
    conn: &Connection,
    ns: &Namespace,
    present: &[(&str, &RoaringBitmap)],
    limit: usize,
    min_shared: u32,
    key_shape: KeyShape,
    filter_docs: Option<&RoaringBitmap>,
    scope: Option<&crate::ScopeFn<'_>>,
) -> Result<Vec<Survivor>> {
    let bitmaps: Vec<&RoaringBitmap> = present.iter().map(|(_, b)| *b).collect();
    if bitmaps.is_empty() || limit == 0 {
        return Ok(Vec::new());
    }
    // The per-query overlap floor: a candidate must share at least `min(m, |present|)`
    // selected tokens. Basing it on the postings actually present (not the raw
    // selection) means appended absent tokens never inflate it, so a query whose only
    // present token is a single n-gram still ranks at overlap 1.
    let floor = (min_shared as usize).min(bitmaps.len()).max(1) as u32;
    let max_count = bitmaps.len() as u32;

    // The `Σ kept-posting cardinality` instrumentation: computed only when
    // the `tracing` feature is on (the macro does not evaluate its arguments
    // otherwise), so the hot path pays nothing for it by default.
    trace_debug!(
        postings = bitmaps.len(),
        sum_cardinality = bitmaps.iter().map(|b| b.len()).sum::<u64>(),
        floor,
        "trifle: overlap candidate generation"
    );

    let acc = bitsliced_overlap(&bitmaps);

    // internal doc id -> its best (highest-count, lowest-seg-id) segment.
    let mut best: HashMap<u32, Survivor> = HashMap::new();
    for c in (floor..=max_count).rev() {
        let bucket = count_eq(&acc, c);
        if !bucket.is_empty() {
            let ids: Vec<u32> = bucket.iter().collect();
            let provenance = hydrate_provenance(conn, ns, &ids, key_shape)?;
            // `ids` is ascending (bitmap order), so the first segment recorded for a
            // doc is its lowest id at this — its highest — count.
            for id in ids {
                let Some((doc, key, label)) = provenance.get(&id) else {
                    continue; // posting id with no live segment row — skip
                };
                // Tier-2 filter: keep only docs passing the structured filter.
                if let Some(fd) = filter_docs {
                    if !fd.contains(*doc) {
                        continue;
                    }
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
                    overlap: c,
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
    // Overlap descending, then internal doc id ascending — stable, deterministic.
    survivors.sort_by(|a, b| b.overlap.cmp(&a.overlap).then(a.doc.cmp(&b.doc)));
    survivors.truncate(limit);
    trace_debug!(
        survivors = survivors.len(),
        "trifle: overlap survivors locked"
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
) -> Result<HashMap<u32, (u32, Key, String)>> {
    let mut out = HashMap::with_capacity(ids.len());
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

/// Reorders the overlap-counted candidates into the final result order.
///
/// The default [`OverlapRanker`] orders by overlap (free — it reads only counts).
/// A richer ranker spends more for quality — the built-in [`Bm25Ranker`] (idf + BM25+
/// length normalization over terms), or a custom one doing proximity, true term-frequency,
/// or exact-substring promotion — over the segment text each candidate carries. The inputs a
/// BM25-style custom ranker needs are public: [`Candidate::matched_terms`] (each matched
/// token with its `df`), [`Candidate::seg_len`] (`|d|`), and [`QueryContext::n_segments`] /
/// [`QueryContext::avgdl`].
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

/// The default ranker when [`Effort`](crate::Effort) reranking is **off**: order by
/// overlap count. Reads only the counts the bit-sliced pass already produced, so it is
/// effectively free — the candidates arrive overlap-descending and this preserves it.
#[derive(Clone, Copy, Debug, Default)]
pub struct OverlapRanker;

impl Ranker for OverlapRanker {
    fn rank(&self, candidates: &Candidates<'_>, _query: &QueryContext<'_>) -> Vec<Ranked> {
        (0..candidates.len())
            .map(|candidate| Ranked { candidate })
            .collect()
    }
}

/// BM25 idf for a token present in `df` of `n` segments, clamped at zero. For `df <= n`
/// the value is already non-negative; the clamp only guards the `df > n` case a desynced
/// posting could produce, so a match can never score *negative*.
///
/// Public so a custom [`Ranker`] can weight by the same idf the built-in [`Bm25Ranker`]
/// uses, fed from [`Candidate::matched_terms`] and [`QueryContext::n_segments`].
pub fn idf(df: u64, n: u64) -> f64 {
    let (df, n) = (df as f64, n as f64);
    (1.0 + (n - df + 0.5) / (df + 0.5)).ln().max(0.0)
}

/// The precision-tier reranker run over an over-fetched pool when
/// [`Effort`](crate::Effort) reranking is on (the default): **real BM25+** over the index's
/// **terms** (the n-grams), with the segment as the unit ("document"):
///
/// `score(seg) = Σ_{t ∈ query ∩ seg} idf(t) · [ (k1+1)·tf / (tf + k1·(1 − b + b·|d|/avgdl)) + δ ]`
///
/// - **idf(t)** is the BM25 idf from the term's `df` (segments containing it) over `N`
///   segments — rare shared terms count far more than common ones.
/// - **length normalization** uses the segment's gram length `|d|` against `avgdl` (the
///   online mean segment length), so a long segment doesn't win on incidental overlap. The
///   `δ` lower bound is BM25+'s fix for BM25 over-penalizing long documents.
/// - **tf is binary** (`tf = 1` for a present term): the bit-sliced index records presence,
///   not per-term counts, and for short n-gram segments a gram's tf is almost always 1, so
///   a presence-frequency BM25+ is the faithful-and-cheap form. (An application that needs
///   true tf can recompute it from each candidate's segment text in a custom [`Ranker`];
///   cross-segment fusion is not a ranker concern — the ranker sees one best segment per
///   document — so fuse above trifle.)
///
/// The ad-hoc literal/substring tier of the previous scorer is **gone** — verifying exact
/// substrings is the frontend's job (highlighting/annotation), not the ranker's. Reads only
/// posting cardinalities and the (already-hydrated) segment length — no extra store reads.
#[derive(Clone, Copy, Debug, Default)]
pub struct Bm25Ranker;

impl Ranker for Bm25Ranker {
    fn rank(&self, candidates: &Candidates<'_>, query: &QueryContext<'_>) -> Vec<Ranked> {
        // BM25 saturation (k1) and length-normalization (b) parameters; BM25+ lower bound δ.
        const K1: f64 = 1.2;
        const B: f64 = 0.75;
        const DELTA: f64 = 1.0;
        let n = query.n_segments.max(1);
        let avgdl = query.avgdl;
        // idf is invariant across candidates for a given present term — compute it once
        // (audit I19), not per (candidate × term).
        let idfs: Vec<f64> = candidates
            .present
            .iter()
            .map(|(_, bm)| idf(bm.len(), n))
            .collect();

        let mut scored: Vec<(usize, f64, u32)> = (0..candidates.len())
            .map(|i| {
                let s = &candidates.survivors[i];
                // Length-normalized tf component (binary tf = 1), shared by all of this
                // segment's present terms. avgdl == 0 (empty corpus) → no length norm.
                let dl = (s.len as f64).max(1.0);
                let norm = if avgdl > 0.0 {
                    1.0 - B + B * dl / avgdl
                } else {
                    1.0
                };
                let tf_component = (K1 + 1.0) / (1.0 + K1 * norm) + DELTA;
                let mut idf_sum = 0.0;
                for (j, (_, bm)) in candidates.present.iter().enumerate() {
                    if bm.contains(s.seg_id) {
                        idf_sum += idfs[j];
                    }
                }
                (i, idf_sum * tf_component, s.overlap)
            })
            .collect();

        // Best score first; tie-break by raw overlap, then original (overlap) order.
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(b.2.cmp(&a.2))
                .then(a.0.cmp(&b.0))
        });
        scored
            .into_iter()
            .map(|(candidate, _, _)| Ranked { candidate })
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
    /// Total live segments `N` (the BM25 corpus size), for idf (`idf(t) ∝ ln(N/df(t))`).
    pub n_segments: u64,
    /// Mean segment gram length (`avgdl`), for BM25+ length normalization. `0.0` only on an
    /// empty corpus (the ranker then skips length normalization).
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

    /// Iterate the candidates in overlap order.
    pub fn iter(&self) -> impl Iterator<Item = Candidate<'_>> {
        (0..self.len()).map(move |i| self.get(i).expect("index in range"))
    }

    /// The query's selected tokens that have a posting, each paired with its document
    /// frequency `df` (the number of segments containing it). These are the idf inputs a
    /// custom [`Ranker`] needs — the same the built-in [`Bm25Ranker`] uses — without any
    /// store read (the postings are already in hand).
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
    /// How many selected tokens this candidate shares.
    pub fn overlap(&self) -> u32 {
        self.s.overlap
    }
    /// The matched segment's text. Every indexed field is stored, so this is always the
    /// segment's full text.
    pub fn text(&self) -> &str {
        &self.s.text
    }
    /// The matched segment's gram length `|d|` (token count with repetition) — the BM25+
    /// length-normalization input, the same value the built-in [`Bm25Ranker`] uses. `0` only
    /// for a sub-n-gram segment that produced no tokens.
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
    /// frequency `df` (segments containing it) — everything a custom [`Ranker`] needs to
    /// compute an idf-weighted score, exactly as the built-in [`Bm25Ranker`] does (combine
    /// with [`QueryContext::n_segments`] for idf and [`seg_len`](Self::seg_len) /
    /// [`QueryContext::avgdl`] for length normalization).
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

    #[test]
    fn bitsliced_counts_match_naive_accumulation() {
        // ids: 1 in three postings, 2 in two, 3 in one.
        let a = bm(&[1, 2, 3]);
        let b = bm(&[1, 2]);
        let c = bm(&[1]);
        let acc = bitsliced_overlap(&[&a, &b, &c]);
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
        let acc = bitsliced_overlap(&refs);
        assert_eq!(count_eq(&acc, 5).iter().collect::<Vec<_>>(), [7]);
        for c in [1, 2, 3, 4, 6] {
            assert!(count_eq(&acc, c).is_empty(), "count {c} should be empty");
        }
    }

    #[test]
    fn count_eq_is_exact_membership_not_at_least() {
        let a = bm(&[1, 2]);
        let b = bm(&[2, 3]);
        let acc = bitsliced_overlap(&[&a, &b]);
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
        let acc = bitsliced_overlap(&refs);
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
        let acc = bitsliced_overlap(&refs);
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
        let acc = bitsliced_overlap(&refs);
        assert_eq!(acc.len(), 4);
        assert_eq!(count_eq(&acc, 8).iter().collect::<Vec<_>>(), [1]);
    }

    #[test]
    fn bitsliced_overlap_degenerate_inputs() {
        // No postings -> no planes.
        assert!(bitsliced_overlap(&[]).is_empty());
        // A single posting -> count 1 for each of its ids, nothing at count 2.
        let single = bm(&[5, 9]);
        let acc = bitsliced_overlap(&[&single]);
        assert_eq!(count_eq(&acc, 1).iter().collect::<Vec<_>>(), [5, 9]);
        assert!(count_eq(&acc, 2).is_empty());
        // An empty posting contributes nothing and pushes no phantom plane.
        let empty = RoaringBitmap::new();
        assert!(bitsliced_overlap(&[&empty]).is_empty());
    }
}
