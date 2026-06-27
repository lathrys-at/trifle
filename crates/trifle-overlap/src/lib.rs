//! `trifle-overlap` — the pure IDF-weighted bit-sliced overlap engine.
//!
//! This is trifle's crown jewel, extracted into a crate that depends on **`roaring` only**.
//! It takes a set of postings (one [`RoaringBitmap`] per selected query token) and streams the
//! candidate ids in **IDF-weighted overlap order** (highest weighted score first). It knows
//! nothing of SQL, segments, keys, text, or provenance — its entire surface is
//! `Vec<RoaringBitmap> → impl Iterator<Item = Scored>`.
//!
//! # The model
//!
//! Each id's *weighted overlap score* is `Σ_i weight_i · [id ∈ posting_i]`, where `weight_i`
//! is the posting's IDF tier weight (rarer grams weigh more; see [`tier_weights`]). The score
//! is accumulated in a **bit-sliced counter** (counts held across bitmap "bit planes"; adding a
//! posting `w` times is a ripple-carry binary add), so building the counter takes `O(k·log k)`
//! bitmap *operations* — the operation **count** is independent of posting cardinality. Wall-clock
//! is *sublinear* in cardinality (in the sparse/array-container regime each op's cost scales with
//! the representation) and genuinely **flat** in the dense bitmap-container regime (fixed-width
//! ops); either way it pulls away from a naive per-id counter as postings densify. A high→low
//! walk over the score buckets then streams candidates lazily: pulling the top-`k` only
//! materializes the high-score head.
//!
//! # Owned, not borrowed
//!
//! [`Counter`] **owns** the postings it is built from (they are moved in). This is what lets a
//! caller embed it in a larger lazy stream with no self-referential lifetime: the [`Walk`]
//! cursor is a plain owned value, and [`Counter::advance`] takes `&self` + `&mut Walk`, so the
//! embedding struct never holds a borrow of itself.
//!
//! ```
//! use trifle_overlap::{Counter, Scored};
//! use roaring::RoaringBitmap;
//!
//! let a: RoaringBitmap = [1u32, 2, 3].into_iter().collect();
//! let b: RoaringBitmap = [2u32, 3].into_iter().collect();
//! let c: RoaringBitmap = [3u32].into_iter().collect();
//! // id 3 is in all three postings, id 2 in two, id 1 in one.
//! let counter = Counter::build(vec![a, b, c], 1.0, 1);
//! let top: Vec<Scored> = counter.stream().collect();
//! assert_eq!(top[0].id, 3);          // highest weighted overlap first
//! assert_eq!(top[0].overlap, 3);     // raw count of postings containing it
//! ```

use roaring::RoaringBitmap;

/// One scored candidate. `id` is an opaque posting id (a segment id, to trifle). `Copy`, 12
/// bytes, no allocation, no provenance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Scored {
    /// The candidate id (opaque to the engine).
    pub id: u32,
    /// The IDF-weighted bit-sliced overlap score — the ordering key.
    pub score: u32,
    /// The raw count of postings containing `id` (the `min_shared` floor is enforced on this).
    pub overlap: u32,
}

/// The IDF-weighted bit-sliced overlap counter over an owned set of postings.
///
/// Build with [`Counter::build`] (default df-anchored tier weights) or
/// [`Counter::build_weighted`] (caller-supplied weights). Stream candidates with
/// [`Counter::stream`] (owning iterator) or drive a [`Walk`] cursor via [`Counter::advance`]
/// (for embedding without a self-referential lifetime).
pub struct Counter {
    /// The postings, owned (moved in). Retained for the raw-overlap check and for the caller's
    /// `matched_terms`-style introspection ([`Counter::postings`]).
    postings: Vec<RoaringBitmap>,
    /// The weighted bit-sliced count: `acc[b]` holds bit `b` of every id's weighted score.
    planes: Vec<RoaringBitmap>,
    /// The per-posting IDF tier weights actually used (each clamped to `>= 1`).
    weights: Vec<u32>,
    /// The raw-overlap floor: a candidate must share at least this many postings.
    floor: u32,
    /// `Σ weights` — the maximum achievable weighted score.
    max_score: u32,
    /// `reachable[c]` = some subset of postings sums (by weight) to exactly `c`. Lets the walk
    /// skip weighted scores no candidate can hold (subset-sums of `{weights}`), cutting
    /// `count_eq` calls. Indexed `0..=max_score`.
    reachable: Vec<bool>,
    /// True iff every weight is exactly 1 (a common collapsed-tier query). Then weighted score
    /// **equals** raw overlap for every id, so the walk can take `overlap = score` directly and
    /// skip the per-id `raw_overlap` `contains` probes — the walk's dominant cost.
    all_weight_one: bool,
}

impl Counter {
    /// Build from owned `postings`, weighting each by per-query df rarity (df = posting
    /// cardinality, by trifle's monotonic-id contract; knob `D = weight_step`, see
    /// [`tier_weights`]). `min_shared` is the raw-overlap floor.
    ///
    /// Builds in `O(k·log k)` bitmap *operations* (the op count is cardinality-independent;
    /// wall-clock is sublinear in cardinality, flat in the dense bitmap-container regime).
    pub fn build(postings: Vec<RoaringBitmap>, weight_step: f64, min_shared: u32) -> Self {
        let cards: Vec<u64> = postings.iter().map(RoaringBitmap::len).collect();
        let weights = tier_weights(&cards, weight_step);
        Self::build_weighted(postings, weights, min_shared)
    }

    /// Build with explicit per-posting `weights` (the escape hatch for a caller-supplied idf,
    /// e.g. BM25-ish or class-normalized). `weights` is parallel to `postings`.
    ///
    /// Every weight is **clamped to `>= 1`**: the floor + early-stop rely on
    /// *weighted ≥ raw*, so a weight of 0 would let a raw-qualifying id have a weighted score
    /// below the floor and be skipped — a *missing* result on a deterministic query. The clamp
    /// makes that unrepresentable.
    ///
    /// # Panics
    ///
    /// Panics if `weights.len() != postings.len()`.
    pub fn build_weighted(
        postings: Vec<RoaringBitmap>,
        mut weights: Vec<u32>,
        min_shared: u32,
    ) -> Self {
        assert_eq!(
            postings.len(),
            weights.len(),
            "weights must be parallel to postings"
        );
        for w in &mut weights {
            *w = (*w).max(1); // correctness: weighted >= raw (floor + early-stop)
        }
        let k = postings.len();
        let floor = (min_shared as usize).min(k).max(1) as u32;

        let mut planes: Vec<RoaringBitmap> = Vec::new();
        for (p, &w) in postings.iter().zip(&weights) {
            add_weighted(&mut planes, p, w);
        }
        let max_score: u32 = weights.iter().sum();

        // Subset-sum reachability over the weights: reachable[0] = true, then for each weight w,
        // reachable[c] |= reachable[c - w]. O(k · max_score), max_score ≤ 4k.
        let mut reachable = vec![false; max_score as usize + 1];
        reachable[0] = true;
        let mut hi = 0u32;
        for &w in &weights {
            let mut c = hi;
            loop {
                if reachable[c as usize] {
                    reachable[(c + w) as usize] = true;
                }
                if c == 0 {
                    break;
                }
                c -= 1;
            }
            hi += w;
        }

        let all_weight_one = weights.iter().all(|&w| w == 1);
        Counter {
            postings,
            planes,
            weights,
            floor,
            max_score,
            reachable,
            all_weight_one,
        }
    }

    /// A fresh best-first walk cursor over this counter. The **only** way to construct a
    /// [`Walk`] (its fields are private), so an embedding crate can build one to drive
    /// [`advance`](Self::advance).
    pub fn walk(&self) -> Walk {
        Walk {
            next_c: self.max_score as i64,
            cur_score: 0,
            bucket: Vec::new(),
            pos: 0,
            scratch: RoaringBitmap::new(),
        }
    }

    /// Produce the next candidate (weighted-score descending, id ascending within a score),
    /// each meeting the raw `min_shared` floor. `&self` + `&mut Walk` — no borrow is stored
    /// next to the counter, so an embedding struct is not self-referential.
    ///
    /// **Lazy:** a consumer that stops calling `advance` never materializes the lower score
    /// buckets — the early-stop the streaming design relies on.
    pub fn advance(&self, w: &mut Walk) -> Option<Scored> {
        loop {
            // Serve the rest of the current bucket, skipping ids below the raw floor (a
            // high-weight rare gram alone can reach a qualifying weighted score with raw
            // overlap below the floor).
            while w.pos < w.bucket.len() {
                let id = w.bucket[w.pos];
                w.pos += 1;
                // When all weights are 1, weighted score == raw overlap, and the walk only
                // visits c >= floor, so every id in the bucket clears the floor — no `contains`
                // probes needed. Otherwise fall back to the exact raw-overlap count.
                let overlap = if self.all_weight_one {
                    w.cur_score
                } else {
                    self.raw_overlap(id)
                };
                if overlap >= self.floor {
                    return Some(Scored {
                        id,
                        score: w.cur_score,
                        overlap,
                    });
                }
            }
            // Current bucket drained — find the next reachable, non-empty lower score bucket.
            let mut refilled = false;
            while w.next_c >= self.floor as i64 {
                let c = w.next_c as u32;
                w.next_c -= 1;
                if !self.reachable[c as usize] {
                    continue; // no subset of weights sums to c — count_eq would be empty
                }
                count_eq_into(&mut w.scratch, &self.planes, c);
                if !w.scratch.is_empty() {
                    w.cur_score = c;
                    w.pos = 0;
                    w.bucket.clear(); // reuse the Vec allocation
                    w.bucket.extend(w.scratch.iter()); // ascending (bitmap order)
                    refilled = true;
                    break;
                }
            }
            if !refilled {
                return None;
            }
        }
    }

    /// Consume the counter into an owning iterator (best-first). Ergonomic for isolated
    /// benchmarking / a power-user caller who does not need to retain the counter.
    pub fn stream(self) -> CounterIter {
        let walk = self.walk();
        CounterIter {
            counter: self,
            walk,
        }
    }

    /// The postings the counter owns (parallel to [`weights`](Self::weights)) — lets an
    /// embedding caller expose `matched_terms`-style introspection without retaining a separate
    /// copy.
    pub fn postings(&self) -> &[RoaringBitmap] {
        &self.postings
    }

    /// The per-posting IDF tier weights used (each `>= 1`).
    pub fn weights(&self) -> &[u32] {
        &self.weights
    }

    /// The raw-overlap floor in effect.
    pub fn floor(&self) -> u32 {
        self.floor
    }

    /// Raw overlap of `id`: how many postings contain it. `O(k)` `contains` probes (k small by
    /// selection), paid only per *yielded* id.
    fn raw_overlap(&self, id: u32) -> u32 {
        self.postings.iter().filter(|p| p.contains(id)).count() as u32
    }
}

/// A plain, owned, `'static` walk cursor. The consumer owns both the [`Counter`] and the
/// `Walk`; neither borrows the other. Construct via [`Counter::walk`] (fields are private).
pub struct Walk {
    /// The next weighted score to consider (descending); `< floor` means the scan is done.
    next_c: i64,
    /// The weighted score of the bucket currently being drained.
    cur_score: u32,
    /// The ids of `cur_score`'s bucket (ascending), being drained. Reused across buckets (the
    /// `Vec` allocation is kept, only `clear`ed) to avoid per-bucket allocation.
    bucket: Vec<u32>,
    /// The drain position within `bucket`.
    pos: usize,
    /// Reused output bitmap for `count_eq_into`, so the walk allocates its working bitmap once
    /// rather than once per bucket.
    scratch: RoaringBitmap,
}

/// The owning iterator returned by [`Counter::stream`].
pub struct CounterIter {
    counter: Counter,
    walk: Walk,
}

impl Iterator for CounterIter {
    type Item = Scored;
    fn next(&mut self) -> Option<Scored> {
        self.counter.advance(&mut self.walk)
    }
}

/// The per-posting df-anchored IDF tier weight `{1,2,3,4}` from each posting's cardinality
/// (`df`). The most-common posting (max df) gets weight 1; rarer grams get more, spaced in
/// df-doublings: `1 + min(3, round(log2(df_max / df_i) / D))`. `D > 0` is the df-doublings per
/// weight step (`<= 0` is treated as `1.0`). **`N`-free** — IDF *gaps* don't depend on corpus
/// size. `weights[i]` is parallel to `cardinalities`.
pub fn tier_weights(cardinalities: &[u64], weight_step: f64) -> Vec<u32> {
    let d = if weight_step > 0.0 { weight_step } else { 1.0 };
    let df_max = cardinalities.iter().copied().max().unwrap_or(1).max(1) as f64;
    cardinalities
        .iter()
        .map(|&card| {
            let df = card.max(1) as f64;
            let steps = ((df_max / df).log2() / d).round().max(0.0) as u32;
            1 + steps.min(3)
        })
        .collect()
}

/// Add `w` copies of `posting` into the bit-sliced planes `acc` (weighted accumulation): inject
/// a carry at each set bit of `w` and ripple it up (XOR = sum bit, AND = carry). Cost is
/// `popcount(w)` ripples — for `w ∈ 1..=4` that is `≤ 2`.
///
/// The first ripple level XORs the **borrowed** `posting` straight into the plane and derives the
/// carry-out from the pre-XOR intersection — so the hot weight-1 path costs one AND + one
/// XOR-assign and **no clone of `posting`** (the old code cloned it into a mutable carry). The
/// carry that propagates to higher planes is the intersection, which is typically far smaller
/// than the posting (often empty), so only it is materialized.
fn add_weighted(acc: &mut Vec<RoaringBitmap>, posting: &RoaringBitmap, w: u32) {
    if posting.is_empty() {
        return; // contributes nothing at any weight
    }
    let mut bit = 0u32;
    while (w >> bit) != 0 {
        if (w >> bit) & 1 == 1 {
            let start = bit as usize;
            while acc.len() <= start {
                acc.push(RoaringBitmap::new());
            }
            // First level: carry-out = already-set AND incoming, then XOR the borrowed posting in.
            // (Inlining the old loop's first iteration, but XOR-ing `&posting` instead of a clone.)
            let mut carry = &acc[start] & posting;
            acc[start] ^= posting;
            let mut level = start + 1;
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

/// Write into `out` (reusing its allocation) the ids whose bit-sliced count is exactly `c`:
/// intersect the planes `c` has set, then subtract every plane it has clear — an id survives iff
/// its plane membership is exactly `c`'s bit pattern. `c == 0` selects nothing.
///
/// Two cheap wins over the naive form: the intersection starts from the **smallest-cardinality**
/// set-bit plane (so the working set shrinks fastest), and both the AND and ANDNOT loops
/// **early-exit** the moment the result empties (a common case deep in the walk).
fn count_eq_into(out: &mut RoaringBitmap, acc: &[RoaringBitmap], c: u32) {
    // A count of 0, or one whose highest set bit is beyond the planes, cannot exist.
    if c == 0 || 32 - c.leading_zeros() > acc.len() as u32 {
        out.clear();
        return;
    }
    // Base = the smallest-cardinality set-bit plane.
    let mut base: Option<usize> = None;
    for (b, plane) in acc.iter().enumerate() {
        if (c >> b) & 1 == 1 && base.is_none_or(|cur| plane.len() < acc[cur].len()) {
            base = Some(b);
        }
    }
    let Some(base) = base else {
        out.clear();
        return;
    };
    out.clone_from(&acc[base]);
    for (b, plane) in acc.iter().enumerate() {
        if b != base && (c >> b) & 1 == 1 {
            *out &= plane;
            if out.is_empty() {
                return;
            }
        }
    }
    for (b, plane) in acc.iter().enumerate() {
        if (c >> b) & 1 == 0 {
            *out -= plane;
            if out.is_empty() {
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bm(ids: &[u32]) -> RoaringBitmap {
        ids.iter().copied().collect()
    }

    fn all(counter: &Counter) -> Vec<Scored> {
        let mut w = counter.walk();
        let mut out = Vec::new();
        while let Some(s) = counter.advance(&mut w) {
            out.push(s);
        }
        out
    }

    #[test]
    fn streams_best_first_with_raw_overlap() {
        let a = bm(&[1, 2, 3]);
        let b = bm(&[2, 3]);
        let c = bm(&[3]);
        // equal cardinalities here would all be weight 1; use min_shared 1.
        let counter = Counter::build(vec![a, b, c], 1.0, 1);
        let got = all(&counter);
        // id 3 (overlap 3) > id 2 (overlap 2) > id 1 (overlap 1).
        assert_eq!(got.iter().map(|s| s.id).collect::<Vec<_>>(), [3, 2, 1]);
        assert_eq!(got[0].overlap, 3);
        assert_eq!(got[1].overlap, 2);
        assert_eq!(got[2].overlap, 1);
        // scores are weakly descending.
        assert!(got.windows(2).all(|p| p[0].score >= p[1].score));
    }

    #[test]
    fn min_shared_floor_excludes_low_overlap() {
        let a = bm(&[1, 2]);
        let b = bm(&[2, 3]);
        // floor 2: only id 2 (in both) qualifies.
        let counter = Counter::build(vec![a, b], 1.0, 2);
        let got = all(&counter);
        assert_eq!(got.iter().map(|s| s.id).collect::<Vec<_>>(), [2]);
    }

    #[test]
    fn weighting_promotes_rarer_grams() {
        // common: df 16 -> weight 1; rare: df 1 -> 4 doublings -> capped weight 4.
        let common = bm(&(0..16).collect::<Vec<_>>());
        let rare = bm(&[5]);
        let w = tier_weights(&[common.len(), rare.len()], 1.0);
        assert_eq!(w, vec![1, 4]);
        // id 5 is in both (raw overlap 2, weighted 1+4=5); ids 0..16\{5} are in `common` only
        // (raw 1, weighted 1). With min_shared 1, id 5 ranks first by weighted score.
        let counter = Counter::build(vec![common, rare], 1.0, 1);
        let got = all(&counter);
        assert_eq!(got[0].id, 5);
        assert_eq!(got[0].score, 5);
        assert_eq!(got[0].overlap, 2);
    }

    #[test]
    fn weight_clamp_keeps_zero_weighted_results() {
        // A 0 weight, if not clamped, would drop id 9 (raw overlap 1) below the weighted floor.
        let a = bm(&[9]);
        let counter = Counter::build_weighted(vec![a], vec![0], 1);
        let got = all(&counter);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].id, 9);
        assert_eq!(got[0].overlap, 1);
        assert_eq!(got[0].score, 1); // clamped to weight 1
    }

    #[test]
    fn empty_and_degenerate_inputs() {
        assert!(all(&Counter::build(vec![], 1.0, 2)).is_empty());
        let counter = Counter::build(vec![RoaringBitmap::new()], 1.0, 1);
        assert!(all(&counter).is_empty());
    }

    #[test]
    fn lazy_walk_matches_full_drain_prefix() {
        // Pulling the top 3 yields exactly the prefix of the full drain.
        let posts: Vec<RoaringBitmap> = (0..6).map(|i| bm(&(0..=(i * 2)).collect::<Vec<_>>())).collect();
        let counter = Counter::build(posts, 1.0, 1);
        let full = all(&counter);
        let counter2 = {
            let posts: Vec<RoaringBitmap> =
                (0..6).map(|i| bm(&(0..=(i * 2)).collect::<Vec<_>>())).collect();
            Counter::build(posts, 1.0, 1)
        };
        let mut w = counter2.walk();
        let mut top3 = Vec::new();
        for _ in 0..3 {
            if let Some(s) = counter2.advance(&mut w) {
                top3.push(s);
            }
        }
        assert_eq!(top3, full[..3]);
    }

    #[test]
    fn bitsliced_count_eq_is_exact_membership() {
        // Reuse the BSI math through the public build: counts must be exact, not "at least".
        let a = bm(&[1, 2]);
        let b = bm(&[2, 3]);
        let counter = Counter::build(vec![a, b], 1.0, 1);
        let got = all(&counter);
        let by_id: std::collections::HashMap<u32, u32> =
            got.iter().map(|s| (s.id, s.overlap)).collect();
        assert_eq!(by_id[&2], 2);
        assert_eq!(by_id[&1], 1);
        assert_eq!(by_id[&3], 1);
    }

    #[test]
    fn high_count_across_plane_boundary() {
        // 8 postings all containing id 100 -> weighted 8 (0b1000), four planes. Exercises the
        // plane-boundary exactness of count_eq via the walk.
        let posts: Vec<RoaringBitmap> = (0..8).map(|_| bm(&[100])).collect();
        let counter = Counter::build(posts, 1.0, 1);
        let got = all(&counter);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].id, 100);
        assert_eq!(got[0].score, 8);
        assert_eq!(got[0].overlap, 8);
    }
}
