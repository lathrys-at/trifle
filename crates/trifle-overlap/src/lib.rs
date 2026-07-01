//! `trifle-overlap` — the IDF-weighted bit-sliced overlap engine, on the **CRoaring** backend.
//!
//! Takes a set of postings (one bitmap per selected query token) and streams candidate ids in
//! **IDF-weighted overlap order** (highest weighted score first). It knows nothing of SQL,
//! segments, keys, text, or provenance — its surface is `postings → impl Iterator<Item=Scored>`.
//!
//! # CRoaring backend + zero-copy views
//!
//! The engine uses the `croaring` crate (SIMD CRoaring). Postings can be folded in **zero-copy**
//! from their stored bytes via [`Counter::build_from_blobs`]: croaring constructs a transient
//! `BitmapView` over the standard portable bytes (no allocation, no container copy).
//!
//! # Owns-postings + all-weight-1 fast path (one counter, no second build)
//!
//! The engine builds a single **weighted** bit-sliced counter (the score). Raw overlap is taken
//! **directly as the score** when all tier weights are 1 — the common rarest-first case: the
//! weighted counter then *is* the count, the floor is met by every walked bucket, and a zero-copy
//! view build retains nothing. For the mixed-weight minority it retains the **owned postings** and
//! reads overlap via `contains` — cheaper than building a second (unweighted) bit-sliced counter
//! for the overlap. Either way the retained state is owned (`Vec<Bitmap>`), so
//! [`Counter`] is `'static` — no self-referential lifetime when embedded in a stream.
//!
//! # Weight 0 and the raw-overlap floor (v0.4)
//!
//! Explicit weights may be **0** (a common gram whose logit-idf energy quantizes to nothing). Such
//! a gram contributes no energy to the bit-sliced planes, so a candidate matching *only* weight-0
//! grams has weighted score 0 and is **not** surfaced by the walk — it is a *count-only* candidate
//! recovered by the search layer (which carries the per-order / floored metadata the engine does
//! not). Because weight 0 breaks the old `weighted ≥ raw` invariant (a candidate can have raw
//! overlap ≥ floor yet weighted score < floor), the walk's **scan** lower bound is decoupled from
//! the raw-overlap **gate**: when any weight is 0 the walk scans every positive bucket down to 1
//! and checks `raw_overlap ≥ floor` per id; otherwise (`min weight ≥ 1`, so `weighted ≥ raw`) it
//! keeps scanning only down to `floor`. The all-weight-1 fast path is unchanged.
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
    /// The walk's bucket-scan lower bound, decoupled from [`floor`](Self::floor) for v0.4 weight-0
    /// support: `1` when any weight is 0 (then `weighted < raw` is possible, so every positive
    /// bucket down to 1 must be scanned and gated per-id on raw overlap), else `floor` (then
    /// `weighted ≥ raw`, so buckets below `floor` hold only sub-floor candidates and are skipped).
    scan_floor: u32,
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
    ///
    /// Weights may be **0** (v0.4): a weight-0 gram adds no energy, so a candidate matching only
    /// weight-0 grams is not surfaced by the walk (it is a count-only candidate for the search
    /// layer to recover). The raw-overlap floor is still honored via the decoupled `scan_floor`
    /// (see the module docs). The old `≥ 1` clamp is gone.
    ///
    /// # Panics
    /// Panics if `weights.len() != bitmaps.len()`.
    pub fn build_weighted(bitmaps: &[Bitmap], weights: Vec<u32>, min_shared: u32) -> Self {
        assert_eq!(bitmaps.len(), weights.len(), "weights parallel to bitmaps");
        let Plan {
            weighted,
            all_weight_one,
            max_score,
            reachable,
            floor,
            scan_floor,
        } = {
            let refs: Vec<&Bitmap> = bitmaps.iter().collect();
            plan(&refs, weights, min_shared)
        };
        // Retain owned postings only for mixed-weight queries (raw overlap via `contains`); the
        // all-weight-1 case takes overlap = score and keeps nothing. A weight-0 gram makes the
        // query mixed (not all-weight-1), so postings are retained and `raw_overlap` works.
        let postings = if all_weight_one {
            Vec::new()
        } else {
            bitmaps.to_vec()
        };
        Counter {
            weighted,
            postings,
            floor,
            scan_floor,
            max_score,
            reachable,
            all_weight_one,
        }
    }

    /// Build **zero-copy** from stored portable posting bytes: each blob is viewed in place
    /// (`BitmapView`, no allocation/copy) and folded transiently.
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
        let Plan {
            weighted,
            all_weight_one,
            max_score,
            reachable,
            floor,
            scan_floor,
        } = {
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
        // materialize owned postings for `contains` — cheaper than a second (unweighted)
        // bit-sliced build, at the cost of zero-copy for this minority case.
        let postings = if all_weight_one {
            Vec::new()
        } else {
            blobs
                .iter()
                .map(|b| Bitmap::try_deserialize::<Portable>(b).expect("portable posting blob"))
                .collect()
        };
        Counter {
            weighted,
            postings,
            floor,
            scan_floor,
            max_score,
            reachable,
            all_weight_one,
        }
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
            // Scan down to `scan_floor` (= 1 when a weight-0 gram is present, else `floor`). The
            // per-id `overlap >= floor` gate above still enforces the raw-overlap floor; decoupling
            // the scan bound lets a positive-energy candidate whose weighted score fell below
            // `floor` (energy from a weight-1 gram, the rest weight-0) still be reached.
            while w.next_c >= self.scan_floor as i64 {
                let c = w.next_c as u32;
                w.next_c -= 1;
                if !self.reachable[c as usize] {
                    continue;
                }
                count_eq_into(&mut w.scratch, &self.weighted, c);
                if !w.scratch.is_empty() {
                    w.cur_score = c;
                    w.pos = 0;
                    fill_bucket(&mut w.bucket, &w.scratch);
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
    /// the all-weight-1 path takes `overlap = score` and retains no postings. Cheaper than
    /// building a second (unweighted) bit-sliced counter for the overlap.
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

/// Materialize `src`'s ids (ascending) into `bucket`, reusing its allocation. Uses croaring's
/// bulk cursor `read_many` — for a large bucket this fills the `Vec` far faster than per-id
/// iteration; neutral for small buckets. Walk cost is a small fraction of a shallow top-k query,
/// so this is a deep-pull / large-result win.
fn fill_bucket(bucket: &mut Vec<u32>, src: &Bitmap) {
    let card = src.cardinality() as usize;
    bucket.clear();
    bucket.resize(card, 0);
    let mut cursor = src.cursor();
    let mut filled = 0;
    while filled < card {
        let n = cursor.read_many(&mut bucket[filled..]);
        if n == 0 {
            break; // exhausted (defensive; one read fills a card-sized buffer)
        }
        filled += n;
    }
    bucket.truncate(filled);
}

/// The weighted bit-sliced planes + walk metadata. Returned by [`plan`]; the owned postings (for
/// `raw_overlap`) are decided by the caller (retained for mixed weights, dropped for all-1).
struct Plan {
    weighted: Vec<Bitmap>,
    all_weight_one: bool,
    max_score: u32,
    reachable: Vec<bool>,
    floor: u32,
    scan_floor: u32,
}

/// Build the weighted bit-sliced planes + walk metadata from the operands. Generic over
/// `&Bitmap` / `&BitmapView`, so the build is zero-copy over views yet reusable for owned inputs.
///
/// v0.4: weights are used as given (no `≥ 1` clamp). A weight-0 gram adds no energy; the walk's
/// `scan_floor` compensates so the raw-overlap floor stays sound (see the module docs).
fn plan<O: Operand + Copy>(ops: &[O], weights: Vec<u32>, min_shared: u32) -> Plan {
    let all_weight_one = !weights.is_empty() && weights.iter().all(|&w| w == 1);
    let has_zero_weight = weights.contains(&0);

    let mut weighted: Vec<Bitmap> = Vec::new();
    for (op, &w) in ops.iter().zip(&weights) {
        add_weighted(&mut weighted, *op, w);
    }

    let floor = (min_shared as usize).min(ops.len()).max(1) as u32;
    // A weight-0 gram breaks `weighted ≥ raw`, so scan every positive bucket (down to 1) and gate
    // per-id on raw overlap; otherwise buckets below `floor` hold only sub-floor candidates.
    let scan_floor = if has_zero_weight { 1 } else { floor };
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
        scan_floor,
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
            &[
                Bitmap::of(&[1, 2, 3]),
                Bitmap::of(&[2, 3]),
                Bitmap::of(&[3]),
            ],
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
        assert_eq!(
            tier_weights(&[common.cardinality(), rare.cardinality()], 1.0),
            vec![1, 4]
        );
        let counter = Counter::build(&[common, rare], 1.0, 1);
        assert!(!counter.all_weight_one);
        assert_eq!(
            counter.postings.len(),
            2,
            "mixed weights retain owned postings"
        );
        let got = all(&counter);
        assert_eq!(got[0].id, 5);
        assert_eq!(got[0].score, 5); // 1 + 4
        assert_eq!(got[0].overlap, 2); // raw overlap via contains over retained postings
    }

    #[test]
    fn zero_weight_yields_nothing_from_the_bit_sliced_walk() {
        // v0.4: the old `≥ 1` clamp is removed, so a weight-0 gram contributes no energy — the
        // bit-sliced walk yields nothing for a candidate that matches only weight-0 grams
        // (max_score 0, no positive bucket). Recovering such *count-only* candidates is the search
        // layer's job (it carries the per-order / floored metadata the engine does not).
        let counter = Counter::build_weighted(&[Bitmap::of(&[9])], vec![0], 1);
        assert_eq!(counter.max_score, 0);
        assert!(all(&counter).is_empty());
    }

    #[test]
    fn zero_weight_gram_does_not_hide_a_positive_energy_candidate() {
        // A candidate matching one weight-1 gram + one weight-0 gram has weighted energy 1 but raw
        // overlap 2. With min_shared = 2 the old `scan down to floor` would skip bucket 1 and lose
        // it; the decoupled `scan_floor = 1` (weight-0 present) reaches it, and the per-id raw gate
        // (overlap 2 ≥ 2) admits it. A candidate matching only the weight-1 gram (overlap 1 < 2) is
        // correctly excluded.
        let w1 = Bitmap::of(&[5, 6]); // weight 1: ids 5,6
        let w0 = Bitmap::of(&[5]); // weight 0: id 5
        let counter = Counter::build_weighted(&[w1, w0], vec![1, 0], 2);
        let got = all(&counter);
        assert_eq!(got.len(), 1, "only id 5 clears the raw-overlap floor of 2");
        assert_eq!(got[0].id, 5);
        assert_eq!(got[0].score, 1, "energy is the single weight-1 gram");
        assert_eq!(
            got[0].overlap, 2,
            "raw overlap counts the weight-0 gram too"
        );
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
    fn plane_count_is_cardinality_independent_flatness() {
        // Flatness (the property M3's reshape must preserve): the bit-sliced op count is fixed by
        // the WEIGHTS — plane count = bits(Σ wq), reachability spans 0..=max_score — never by
        // posting cardinality. Two multi-bucket counters with identical weights but vastly
        // different cardinalities must have identical plane count, max_score, and reachability, so
        // the walk does the same number of bitmap ops per bucket regardless of posting size.
        let weights = vec![3u32, 5u32]; // buckets {3, 5, 8} — multi-bucket, not single
        let small = [Bitmap::of(&[1]), Bitmap::of(&[1, 2])];
        let huge = [
            (0..100_000u32).collect::<Bitmap>(),
            (0..200_000u32).collect::<Bitmap>(),
        ];
        let c_small = Counter::build_weighted(&small, weights.clone(), 1);
        let c_huge = Counter::build_weighted(&huge, weights.clone(), 1);
        assert_eq!(
            c_small.weighted.len(),
            c_huge.weighted.len(),
            "plane count is cardinality-independent"
        );
        assert_eq!(c_small.max_score, c_huge.max_score);
        assert_eq!(c_small.reachable.len(), c_huge.reachable.len());
    }

    #[test]
    fn lazy_walk_matches_full_drain_prefix() {
        let posts: Vec<Bitmap> = (0..6)
            .map(|i| Bitmap::of(&(0..=(i * 2)).collect::<Vec<_>>()))
            .collect();
        let counter = Counter::build(&posts, 1.0, 1);
        let full = all(&counter);
        let mut w = counter.walk();
        let top3: Vec<Scored> = (0..3).filter_map(|_| counter.advance(&mut w)).collect();
        assert_eq!(top3, full[..3]);
    }
}
