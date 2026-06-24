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
use crate::store::Namespace;

/// One ranked segment that survived candidate generation: its provenance, the id
/// used internally for hydration, its overlap count, and (once hydrated) its text.
pub(crate) struct Survivor {
    pub doc_id: i64,
    pub source: String,
    pub ref_: String,
    pub seg_id: u32,
    pub overlap: u32,
    pub text: Option<String>,
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
pub(crate) fn overlap_search(
    conn: &Connection,
    ns: &Namespace,
    present: &[(&str, &RoaringBitmap)],
    limit: usize,
    min_shared: u32,
    scope: Option<&crate::ScopeFn>,
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

    // The `Σ kept-posting cardinality` instrumentation (§10.2): computed only when
    // the `tracing` feature is on (the macro does not evaluate its arguments
    // otherwise), so the hot path pays nothing for it by default.
    trace_debug!(
        postings = bitmaps.len(),
        sum_cardinality = bitmaps.iter().map(|b| b.len()).sum::<u64>(),
        floor,
        "trifle: overlap candidate generation"
    );

    let acc = bitsliced_overlap(&bitmaps);

    // doc_id -> its best (highest-count, lowest-id) segment.
    let mut best: HashMap<i64, Survivor> = HashMap::new();
    for c in (floor..=max_count).rev() {
        let bucket = count_eq(&acc, c);
        if !bucket.is_empty() {
            let ids: Vec<u32> = bucket.iter().collect();
            let provenance = hydrate_provenance(conn, ns, &ids)?;
            // `ids` is ascending (bitmap order), so the first segment recorded for a
            // doc is its lowest id at this — its highest — count.
            for id in ids {
                let Some((doc_id, source, ref_)) = provenance.get(&id) else {
                    continue; // posting id with no live segment row — skip
                };
                if let Some(scope) = scope {
                    if !scope(*doc_id, source, ref_) {
                        continue;
                    }
                }
                best.entry(*doc_id).or_insert_with(|| Survivor {
                    doc_id: *doc_id,
                    source: source.clone(),
                    ref_: ref_.clone(),
                    seg_id: id,
                    overlap: c,
                    text: None,
                });
            }
        }
        if best.len() >= limit {
            break;
        }
    }

    let mut survivors: Vec<Survivor> = best.into_values().collect();
    // Overlap descending, then doc_id ascending — a stable, deterministic order.
    survivors.sort_by(|a, b| b.overlap.cmp(&a.overlap).then(a.doc_id.cmp(&b.doc_id)));
    survivors.truncate(limit);
    trace_debug!(
        survivors = survivors.len(),
        "trifle: overlap survivors locked"
    );
    Ok(survivors)
}

/// Provenance `(doc_id, source, ref)` per segment id, for a set of ids, in one
/// batched `WHERE id IN rarray(?1)` read — no temp btree, one prepared statement.
fn hydrate_provenance(
    conn: &Connection,
    ns: &Namespace,
    ids: &[u32],
) -> Result<HashMap<u32, (i64, String, String)>> {
    let mut out = HashMap::with_capacity(ids.len());
    if ids.is_empty() {
        return Ok(out);
    }
    let arr: Rc<Vec<Value>> = Rc::new(ids.iter().map(|&i| Value::Integer(i as i64)).collect());
    let sql = format!(
        "SELECT id, doc_id, source, ref FROM {} WHERE id IN rarray(?1)",
        ns.seg()
    );
    let mut stmt = conn.prepare_cached(&sql)?;
    let mut rows = stmt.query(rusqlite::params![arr])?;
    while let Some(r) = rows.next()? {
        let id: i64 = r.get(0)?;
        out.insert(id as u32, (r.get(1)?, r.get(2)?, r.get(3)?));
    }
    Ok(out)
}

/// Reorders the overlap-counted candidates into the final result order.
///
/// The default [`OverlapRanker`] orders by overlap (free — it reads only counts).
/// A richer ranker spends more for quality — literal-verification (promoting exact
/// substring hits, recovering a precision tier without BM25), proximity, or
/// idf-weighting — over the segment text each candidate carries.
pub trait Ranker: Send + Sync {
    /// Return the candidates in final result order (best first). May drop candidates
    /// by omitting them; trifle truncates the result to the search limit.
    fn rank(&self, candidates: &Candidates<'_>, query: &QueryContext<'_>) -> Vec<Ranked>;
}

/// The default ranker: order by overlap count. Reads only the counts the bit-sliced
/// pass already produced, so it is effectively free — the candidates arrive in
/// overlap-descending order, and this preserves it.
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
    /// The caller's document id.
    pub fn doc_id(&self) -> i64 {
        self.s.doc_id
    }
    /// The provenance `source` label.
    pub fn source(&self) -> &str {
        &self.s.source
    }
    /// The provenance `ref` label.
    pub fn ref_(&self) -> &str {
        &self.s.ref_
    }
    /// How many selected tokens this candidate shares.
    pub fn overlap(&self) -> u32 {
        self.s.overlap
    }
    /// The matched segment's text, if available (absent only in contentless mode
    /// when the resolver returned `None`).
    pub fn text(&self) -> Option<&str> {
        self.s.text.as_deref()
    }
    /// Which selected tokens this candidate's segment actually contains.
    pub fn matched_tokens(&self) -> Vec<&str> {
        self.present
            .iter()
            .filter(|(_, bm)| bm.contains(self.s.seg_id))
            .map(|(t, _)| *t)
            .collect()
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
