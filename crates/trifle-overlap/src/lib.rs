//! `trifle-overlap` — the IDF-weighted bit-sliced overlap engine, on the **CRoaring** backend.
//!
//! Takes a set of postings (one bitmap per selected query token) and streams candidate ids in
//! **IDF-weighted overlap order** (highest weighted score first). It knows nothing of SQL,
//! segments, keys, text, or provenance — its surface is `postings → impl Iterator<Item=Scored>`.
//!
//! # Backend: CRoaring + zero-copy views
//!
//! The engine uses the `croaring` crate (SIMD CRoaring). Postings can be folded in **zero-copy**
//! from their stored bytes via [`Counter::build_from_blobs`]: croaring constructs a transient
//! `BitmapView` over the bytes (no allocation, no container copy) — and the `roaring`-crate's
//! portable serialization is **byte-identical** to croaring's, so trifle's existing stored blobs
//! are viewed directly with no format migration.
//!
//! # Owns-postings + all-weight-1 fast path (one counter, no second build)
//!
//! The engine builds a single **weighted** bit-sliced counter (the score). Raw overlap is taken
//! **directly as the score** when all tier weights are 1 — the common rarest-first case: the
//! weighted counter then *is* the count, the floor is met by every walked bucket, and a zero-copy
//! view build retains nothing. For the mixed-weight minority it retains the **owned postings** and
//! reads overlap via `contains` — measured **~2.5× cheaper** than building a second (unweighted)
//! bit-sliced counter for the overlap. Either way the retained state is owned (`Vec<Bitmap>`), so
//! [`Counter`] is `'static` — no self-referential lifetime when embedded in a stream.
//!
//! # Flatness
//!
//! Building the counter is `O(k·log k)` bitmap *operations* (the op count is cardinality-
//! independent); wall-clock is sublinear in cardinality (sparse) and flat in the dense
//! bitmap-container regime. A high→low walk over the score buckets streams candidates lazily.
//!
//! ```
//! use trifle_overlap::{Counter, Scored};
//! use croaring::Bitmap;
//!
//! let a = Bitmap::of(&[1, 2, 3]);
//! let b = Bitmap::of(&[2, 3]);
//! let c = Bitmap::of(&[3]);
//! let counter = Counter::build(&[a, b, c], 1.0, 1);
//! let top: Vec<Scored> = counter.stream().collect();
//! assert_eq!(top[0].id, 3);       // in all three postings
//! assert_eq!(top[0].overlap, 3);
//! ```

use croaring::{Bitmap, BitmapView, Portable};

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

/// An operand that can be folded into the bit-sliced counter without being materialized — either
/// an owned `&Bitmap` or a zero-copy `&BitmapView`. This is what lets the build be zero-copy over
/// stored bytes yet reuse one fold path for owned inputs.
trait Operand {
    fn is_empty(&self) -> bool;
    /// `self & plane` — the carry-out (owned).
    fn and_plane(&self, plane: &Bitmap) -> Bitmap;
    /// `plane ^= self` — the sum bit, in place against the borrowed operand (no copy of self).
    fn xor_into(&self, plane: &mut Bitmap);
}

impl Operand for &Bitmap {
    fn is_empty(&self) -> bool {
        self.cardinality() == 0
    }
    fn and_plane(&self, plane: &Bitmap) -> Bitmap {
        *self & plane
    }
    fn xor_into(&self, plane: &mut Bitmap) {
        *plane ^= *self;
    }
}

impl Operand for &BitmapView<'_> {
    fn is_empty(&self) -> bool {
        self.cardinality() == 0 // BitmapView has no inherent is_empty; cardinality via Deref
    }
    fn and_plane(&self, plane: &Bitmap) -> Bitmap {
        *self & plane // &BitmapView & &Bitmap -> Bitmap (commuted AND; no copy of the view)
    }
    fn xor_into(&self, plane: &mut Bitmap) {
        *plane ^= *self; // Bitmap ^= &BitmapView
    }
}

/// The IDF-weighted bit-sliced overlap counter over a set of postings (CRoaring backend).
///
/// Build with [`Counter::build`] / [`Counter::build_weighted`] (owned bitmaps) or
/// [`Counter::build_from_blobs`] / [`Counter::build_weighted_from_blobs`] (zero-copy over stored
/// portable bytes). Stream with [`Counter::stream`] or drive a [`Walk`] via [`Counter::advance`].
pub struct Counter {
    /// Weighted bit-sliced count: `weighted[b]` holds bit `b` of every id's weighted score.
    weighted: Vec<Bitmap>,
    /// The owned postings, retained ONLY for mixed-weight queries (raw overlap via `contains`).
    /// **Empty** when all weights are 1 — then weighted score *is* the overlap, so nothing is
    /// retained (and a zero-copy view build keeps no copy at all). Owned `Bitmap`s ⇒ `'static`.
    postings: Vec<Bitmap>,
    /// The raw-overlap floor: a candidate must share at least this many postings.
    floor: u32,
    /// `Σ weights` — the maximum achievable weighted score.
    max_score: u32,
    /// `reachable[c]` = some subset of weights sums to exactly `c`; lets the walk skip
    /// impossible weighted scores. Indexed `0..=max_score`.
    reachable: Vec<bool>,
    /// True iff every weight is 1: then weighted score == raw overlap, so the walk takes
    /// `overlap = score` and the unweighted counter is not built.
    all_weight_one: bool,
}

impl Counter {
    /// Build from owned `bitmaps`, weighting each by per-query df rarity (df = cardinality; knob
    /// `D = weight_step`, see [`tier_weights`]). `min_shared` is the raw-overlap floor.
    pub fn build(bitmaps: &[Bitmap], weight_step: f64, min_shared: u32) -> Self {
        let cards: Vec<u64> = bitmaps.iter().map(Bitmap::cardinality).collect();
        let weights = tier_weights(&cards, weight_step);
        Self::build_weighted(bitmaps, weights, min_shared)
    }

    /// Build from owned `bitmaps` with explicit per-posting `weights` (parallel to `bitmaps`).
    /// Weights are clamped to `>= 1` (the floor + early-stop rely on *weighted ≥ raw*).
    ///
    /// # Panics
    /// Panics if `weights.len() != bitmaps.len()`.
    pub fn build_weighted(bitmaps: &[Bitmap], weights: Vec<u32>, min_shared: u32) -> Self {
        assert_eq!(bitmaps.len(), weights.len(), "weights parallel to bitmaps");
        let Plan { weighted, all_weight_one, max_score, reachable, floor } = {
            let refs: Vec<&Bitmap> = bitmaps.iter().collect();
            plan(&refs, weights, min_shared)
        };
        // Retain owned postings only for mixed-weight queries (raw overlap via `contains`); the
        // all-weight-1 case takes overlap = score and keeps nothing.
        let postings = if all_weight_one { Vec::new() } else { bitmaps.to_vec() };
        Counter { weighted, postings, floor, max_score, reachable, all_weight_one }
    }

    /// Build **zero-copy** from stored portable posting bytes: each blob is viewed in place
    /// (`BitmapView`, no allocation/copy) and folded transiently. The roaring-crate and croaring
    /// portable formats are byte-identical, so trifle's existing blobs work unchanged.
    pub fn build_from_blobs(blobs: &[&[u8]], weight_step: f64, min_shared: u32) -> Self {
        // Cardinalities (= df) come from a cheap view pass (O(containers), no copy); the weighted
        // build (and any owned retention for mixed weights) happens in `build_weighted_from_blobs`.
        let cards: Vec<u64> = blobs
            .iter()
            // SAFETY: `b` outlives the transient view used only to read cardinality here.
            .map(|b| unsafe { BitmapView::deserialize::<Portable>(b) }.cardinality())
            .collect();
        let weights = tier_weights(&cards, weight_step);
        Self::build_weighted_from_blobs(blobs, weights, min_shared)
    }

    /// Zero-copy build with explicit `weights` (parallel to `blobs`).
    ///
    /// # Panics
    /// Panics if `weights.len() != blobs.len()`.
    pub fn build_weighted_from_blobs(blobs: &[&[u8]], weights: Vec<u32>, min_shared: u32) -> Self {
        assert_eq!(blobs.len(), weights.len(), "weights parallel to blobs");
        let Plan { weighted, all_weight_one, max_score, reachable, floor } = {
            // SAFETY: each `blob` outlives its `view` (both live only within this block);
            // portable layout needs no alignment.
            let views: Vec<BitmapView<'_>> = blobs
                .iter()
                .map(|b| unsafe { BitmapView::deserialize::<Portable>(b) })
                .collect();
            let refs: Vec<&BitmapView<'_>> = views.iter().collect();
            plan(&refs, weights, min_shared)
        };
        // all-weight-1: the build was zero-copy (views only) and needs no postings. Mixed-weight:
        // materialize owned postings for `contains` — owned+contains is ~2.5x cheaper than a
        // second (unweighted) bit-sliced build (measured), at the cost of zero-copy for this
        // minority case.
        let postings = if all_weight_one {
            Vec::new()
        } else {
            blobs
                .iter()
                .map(|b| Bitmap::try_deserialize::<Portable>(b).expect("portable posting blob"))
                .collect()
        };
        Counter { weighted, postings, floor, max_score, reachable, all_weight_one }
    }

    /// A fresh best-first walk cursor over this counter (the only way to construct a [`Walk`]).
    pub fn walk(&self) -> Walk {
        Walk {
            next_c: self.max_score as i64,
            cur_score: 0,
            bucket: Vec::new(),
            pos: 0,
            scratch: Bitmap::new(),
        }
    }

    /// Produce the next candidate (weighted-score descending, id ascending within a score), each
    /// meeting the raw `min_shared` floor. `&self` + `&mut Walk` — no self-reference.
    pub fn advance(&self, w: &mut Walk) -> Option<Scored> {
        loop {
            while w.pos < w.bucket.len() {
                let id = w.bucket[w.pos];
                w.pos += 1;
                // all-weight-1: score == raw overlap, and the walk only visits c >= floor, so
                // every id clears the floor — no probing. Otherwise read the unweighted counter.
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
            let mut refilled = false;
            while w.next_c >= self.floor as i64 {
                let c = w.next_c as u32;
                w.next_c -= 1;
                if !self.reachable[c as usize] {
                    continue;
                }
                count_eq_into(&mut w.scratch, &self.weighted, c);
                if !w.scratch.is_empty() {
                    w.cur_score = c;
                    w.pos = 0;
                    w.bucket.clear();
                    w.bucket.extend(w.scratch.iter());
                    refilled = true;
                    break;
                }
            }
            if !refilled {
                return None;
            }
        }
    }

    /// Consume the counter into an owning iterator (best-first).
    pub fn stream(self) -> CounterIter {
        let walk = self.walk();
        CounterIter {
            counter: self,
            walk,
        }
    }

    /// The raw-overlap floor in effect.
    pub fn floor(&self) -> u32 {
        self.floor
    }

    /// Raw overlap of `id`: how many retained postings contain it (`O(k)` SIMD `contains`,
    /// k small by selection, paid only per *yielded* id). Only called for mixed-weight counters;
    /// the all-weight-1 path takes `overlap = score` and retains no postings. Measured ~2.5x
    /// cheaper than building a second (unweighted) bit-sliced counter for the overlap.
    fn raw_overlap(&self, id: u32) -> u32 {
        self.postings.iter().filter(|p| p.contains(id)).count() as u32
    }
}

/// A plain, owned, `'static` walk cursor. The consumer owns both the [`Counter`] and the `Walk`;
/// neither borrows the other. Construct via [`Counter::walk`].
pub struct Walk {
    /// The next weighted score to consider (descending); `< floor` ends the scan.
    next_c: i64,
    /// The weighted score of the bucket currently being drained.
    cur_score: u32,
    /// The ids of `cur_score`'s bucket (ascending). Reused across buckets (only `clear`ed).
    bucket: Vec<u32>,
    /// The drain position within `bucket`.
    pos: usize,
    /// Reused `count_eq_into` output, so the walk allocates its working bitmap once.
    scratch: Bitmap,
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
/// df-doublings: `1 + min(3, round(log2(df_max / df_i) / D))`. `N`-free. Parallel to
/// `cardinalities`.
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

/// The weighted bit-sliced planes + walk metadata. Returned by [`plan`]; the owned postings (for
/// `raw_overlap`) are decided by the caller (retained for mixed weights, dropped for all-1).
struct Plan {
    weighted: Vec<Bitmap>,
    all_weight_one: bool,
    max_score: u32,
    reachable: Vec<bool>,
    floor: u32,
}

/// Build the weighted bit-sliced planes + walk metadata from the operands. Generic over
/// `&Bitmap` / `&BitmapView`, so the build is zero-copy over views yet reusable for owned inputs.
fn plan<O: Operand + Copy>(ops: &[O], mut weights: Vec<u32>, min_shared: u32) -> Plan {
    for w in &mut weights {
        *w = (*w).max(1); // weighted >= raw (floor + early-stop soundness)
    }
    let all_weight_one = weights.iter().all(|&w| w == 1);

    let mut weighted: Vec<Bitmap> = Vec::new();
    for (op, &w) in ops.iter().zip(&weights) {
        add_weighted(&mut weighted, *op, w);
    }

    let floor = (min_shared as usize).min(ops.len()).max(1) as u32;
    let max_score: u32 = weights.iter().sum();

    // Subset-sum reachability over the weights (0/1 knapsack, descending to avoid reuse).
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

    Plan {
        weighted,
        all_weight_one,
        max_score,
        reachable,
        floor,
    }
}

/// Add `w` copies of `op` into the bit-sliced planes `acc`: inject a carry at each set bit of `w`
/// and ripple it up (XOR = sum bit, AND = carry). The first ripple level folds the borrowed
/// operand directly (no copy of the posting); the carry that propagates is the intersection,
/// typically far smaller (often empty).
fn add_weighted<O: Operand>(acc: &mut Vec<Bitmap>, op: O, w: u32) {
    if op.is_empty() {
        return;
    }
    let mut bit = 0u32;
    while (w >> bit) != 0 {
        if (w >> bit) & 1 == 1 {
            let start = bit as usize;
            while acc.len() <= start {
                acc.push(Bitmap::new());
            }
            let mut carry = op.and_plane(&acc[start]);
            op.xor_into(&mut acc[start]);
            let mut level = start + 1;
            while !carry.is_empty() {
                while acc.len() <= level {
                    acc.push(Bitmap::new());
                }
                let new_carry = &acc[level] & &carry;
                acc[level] ^= &carry;
                carry = new_carry;
                level += 1;
            }
        }
        bit += 1;
    }
}

/// Write into `out` (reusing its allocation via `clone_from`) the ids whose bit-sliced count is
/// exactly `c`: intersect the planes `c` has set (smallest-cardinality first), then subtract
/// every plane it has clear. Both loops early-exit the moment the result empties.
fn count_eq_into(out: &mut Bitmap, acc: &[Bitmap], c: u32) {
    if c == 0 || 32 - c.leading_zeros() > acc.len() as u32 {
        out.clear();
        return;
    }
    let mut base: Option<usize> = None;
    for (b, plane) in acc.iter().enumerate() {
        if (c >> b) & 1 == 1 && base.is_none_or(|cur| plane.cardinality() < acc[cur].cardinality())
        {
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

    fn all(counter: &Counter) -> Vec<Scored> {
        let mut w = counter.walk();
        let mut out = Vec::new();
        while let Some(s) = counter.advance(&mut w) {
            out.push(s);
        }
        out
    }

    fn portable(bits: &[u32]) -> Vec<u8> {
        Bitmap::of(bits).serialize::<Portable>()
    }

    #[test]
    fn streams_best_first_with_raw_overlap() {
        let counter = Counter::build(
            &[Bitmap::of(&[1, 2, 3]), Bitmap::of(&[2, 3]), Bitmap::of(&[3])],
            1.0,
            1,
        );
        let got = all(&counter);
        assert_eq!(got.iter().map(|s| s.id).collect::<Vec<_>>(), [3, 2, 1]);
        assert_eq!(got[0].overlap, 3);
        assert!(got.windows(2).all(|p| p[0].score >= p[1].score));
    }

    #[test]
    fn min_shared_floor_excludes_low_overlap() {
        let counter = Counter::build(&[Bitmap::of(&[1, 2]), Bitmap::of(&[2, 3])], 1.0, 2);
        assert_eq!(all(&counter).iter().map(|s| s.id).collect::<Vec<_>>(), [2]);
    }

    #[test]
    fn weighting_promotes_rarer_grams_via_owned_postings() {
        // common df 16 -> weight 1; rare df 1 -> weight 4. Mixed weights -> owned postings retained.
        let common = Bitmap::of(&(0..16).collect::<Vec<_>>());
        let rare = Bitmap::of(&[5]);
        assert_eq!(tier_weights(&[common.cardinality(), rare.cardinality()], 1.0), vec![1, 4]);
        let counter = Counter::build(&[common, rare], 1.0, 1);
        assert!(!counter.all_weight_one);
        assert_eq!(counter.postings.len(), 2, "mixed weights retain owned postings");
        let got = all(&counter);
        assert_eq!(got[0].id, 5);
        assert_eq!(got[0].score, 5); // 1 + 4
        assert_eq!(got[0].overlap, 2); // raw overlap via contains over retained postings
    }

    #[test]
    fn weight_clamp_keeps_zero_weighted_results() {
        let counter = Counter::build_weighted(&[Bitmap::of(&[9])], vec![0], 1);
        let got = all(&counter);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].score, 1); // clamped to 1
        assert_eq!(got[0].overlap, 1);
    }

    #[test]
    fn zero_copy_blobs_match_owned() {
        // Build from portable blobs (zero-copy views) must equal building from owned bitmaps.
        let a = portable(&[1, 2, 3, 70_000]);
        let b = portable(&[2, 3, 70_000]);
        let c = portable(&[3]);
        let from_blobs = Counter::build_from_blobs(&[&a, &b, &c], 1.0, 1);
        let owned = Counter::build(
            &[
                Bitmap::of(&[1, 2, 3, 70_000]),
                Bitmap::of(&[2, 3, 70_000]),
                Bitmap::of(&[3]),
            ],
            1.0,
            1,
        );
        assert_eq!(all(&from_blobs), all(&owned));
    }

    #[test]
    fn empty_and_degenerate_inputs() {
        assert!(all(&Counter::build(&[], 1.0, 2)).is_empty());
        assert!(all(&Counter::build(&[Bitmap::new()], 1.0, 1)).is_empty());
    }

    #[test]
    fn high_count_across_plane_boundary() {
        let posts: Vec<Bitmap> = (0..8).map(|_| Bitmap::of(&[100])).collect();
        let counter = Counter::build(&posts, 1.0, 1);
        let got = all(&counter);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].score, 8);
        assert_eq!(got[0].overlap, 8);
    }

    #[test]
    fn lazy_walk_matches_full_drain_prefix() {
        let posts: Vec<Bitmap> = (0..6).map(|i| Bitmap::of(&(0..=(i * 2)).collect::<Vec<_>>())).collect();
        let counter = Counter::build(&posts, 1.0, 1);
        let full = all(&counter);
        let mut w = counter.walk();
        let top3: Vec<Scored> = (0..3).filter_map(|_| counter.advance(&mut w)).collect();
        assert_eq!(top3, full[..3]);
    }
}
