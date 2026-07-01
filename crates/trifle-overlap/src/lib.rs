//! `trifle-overlap` — the weighted bit-sliced overlap engine, on the **CRoaring** backend.
//!
//! Takes a set of postings (one bitmap per selected query token) with explicit non-negative
//! integer weights, and streams candidate ids in **weighted-overlap (energy) order** — highest
//! weighted score first. It knows nothing of SQL, segments, keys, text, provenance, or weighting
//! models — its surface is `(postings, weights) → impl Iterator<Item = Scored>`; the consumer
//! owns the weighting model (trifle feeds `Δ`-quantized logit-idf energies).
//!
//! # Energy-only (v0.5): the consumer owns the raw-overlap gate
//!
//! The engine builds and walks the weighted bit-sliced counter only; it retains **no** copy of
//! the postings and reports **no** raw overlap. The consumer — which already holds the postings
//! and typically needs a per-id pass of its own (trifle fuses raw overlap with its count credit
//! in one `contains` sweep) — applies the raw-overlap floor to the yielded ids. Pre-v0.5 the
//! engine deep-cloned every posting and paid a second `contains` sweep per candidate to gate
//! internally; both are gone. What remains of `min_shared` inside the engine is the **scan
//! bound** ([`Counter::floor`]): when every weight is `≥ 1`, `weighted ≥ raw`, so score buckets
//! below the floor can only hold sub-floor candidates and are never scanned.
//!
//! # Weight 0 (v0.4)
//!
//! Weights may be **0** (a common gram whose quantized energy is nothing). Such a gram adds no
//! energy to the planes, so a candidate matching *only* weight-0 grams has weighted score 0 and
//! is never surfaced by the walk — recovering those *count-only* candidates is the consumer's
//! job. A weight-0 gram also breaks `weighted ≥ raw`, so the walk's scan bound drops to 1 (every
//! positive bucket is scanned); the consumer's raw-overlap gate stays sound either way.
//!
//! # Flatness
//!
//! Building the counter is `O(k·log k)` bitmap *operations* for `k` postings (the op count is
//! cardinality-independent); wall-clock is sublinear in cardinality (sparse) and flat in the
//! dense bitmap-container regime. More precisely, the build is `O(Σw·bits(Σw))` bitmap ops and
//! the walk visits at most `Σw` buckets of `bits(Σw)` plane ops each — so the honest bound is
//! parameterized by the **weight sum**, and the `O(k·log k)` reading holds because the consumer
//! bounds each weight by a constant ceiling (trifle: `⌊E_max/Δ⌉`, a `~log log N` quantity). See
//! *Weight preconditions* on [`Counter::build_weighted`].
//!
//! ```
//! use trifle_overlap::{Counter, Scored};
//! use croaring::Bitmap;
//!
//! let a = Bitmap::of(&[1, 2, 3]);
//! let b = Bitmap::of(&[2, 3]);
//! let c = Bitmap::of(&[3]);
//! let counter = Counter::build_weighted(&[a, b, c], vec![1, 1, 1], 1);
//! let top: Vec<Scored> = counter.stream().collect();
//! assert_eq!(top[0].id, 3);      // in all three postings
//! assert_eq!(top[0].score, 3);
//! ```

use croaring::Bitmap;

/// One scored candidate. `id` is an opaque posting id (a segment id, to trifle). `Copy`, 8
/// bytes, no allocation, no provenance. Raw overlap is not reported (v0.5) — the consumer, which
/// holds the postings, computes it where needed (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Scored {
    /// The candidate id (opaque to the engine).
    pub id: u32,
    /// The weighted bit-sliced overlap score — the ordering key.
    pub score: u32,
}

/// The weighted bit-sliced overlap counter over a set of postings (CRoaring backend).
///
/// Build with [`Counter::build_weighted`]; stream with [`Counter::stream`] or drive a [`Walk`]
/// via [`Counter::advance`]. The counter owns only its bit-sliced planes (never the postings), so
/// it is `'static` — no self-referential lifetime when embedded in a stream.
pub struct Counter {
    /// Weighted bit-sliced count: `weighted[b]` holds bit `b` of every id's weighted score.
    weighted: Vec<Bitmap>,
    /// The clamped raw-overlap floor (see [`floor`](Self::floor)) — **advisory** (v0.5): the
    /// consumer gates yielded ids on it; the engine only derives `scan_floor` from it.
    floor: u32,
    /// The walk's bucket-scan lower bound: `1` when any weight is 0 (then `weighted < raw` is
    /// possible, so every positive bucket must be scanned), else `floor` (then `weighted ≥ raw`,
    /// so buckets below `floor` hold only candidates the consumer's raw gate would drop anyway).
    scan_floor: u32,
    /// `Σ weights` — the maximum achievable weighted score.
    max_score: u32,
    /// `reachable[c]` = some subset of weights sums to exactly `c`; lets the walk skip
    /// impossible weighted scores. Indexed `0..=max_score`.
    reachable: Vec<bool>,
}

impl Counter {
    /// Build from `bitmaps` with explicit per-posting `weights` (parallel to `bitmaps`).
    /// `min_shared` is the consumer's raw-overlap floor — the engine does **not** enforce it
    /// (v0.5, see the module docs); it derives the walk's scan bound from it and reports the
    /// clamped value back via [`floor`](Self::floor).
    ///
    /// Weights may be **0** (v0.4): a weight-0 gram adds no energy, so a candidate matching only
    /// weight-0 grams is not surfaced by the walk (it is a count-only candidate for the consumer
    /// to recover). The old `≥ 1` clamp is gone.
    ///
    /// # Weight preconditions
    ///
    /// Work and memory scale with the weight **sum** `Σw`: the walk may visit up to `Σw` score
    /// buckets and the reachability table holds `Σw + 1` entries. Callers must bound each weight
    /// by a small ceiling of their own (trifle: `⌊E_max/Δ⌉`) — the engine checks only that `Σw`
    /// does not overflow `u32`.
    ///
    /// # Panics
    /// Panics if `weights.len() != bitmaps.len()`, or if the weight sum overflows `u32`.
    pub fn build_weighted(bitmaps: &[Bitmap], weights: Vec<u32>, min_shared: u32) -> Self {
        assert_eq!(bitmaps.len(), weights.len(), "weights parallel to bitmaps");
        plan(bitmaps, weights, min_shared)
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

    /// Produce the next candidate (weighted-score descending, id ascending within a score).
    /// `&self` + `&mut Walk` — no self-reference. The consumer applies its raw-overlap gate to
    /// the yielded ids (v0.5): with every weight `≥ 1` the scan bound already excludes buckets
    /// that can only hold sub-floor candidates, but a yielded id may still have raw overlap
    /// below the floor (its energy concentrated in few, heavy grams).
    pub fn advance(&self, w: &mut Walk) -> Option<Scored> {
        loop {
            if w.pos < w.bucket.len() {
                let id = w.bucket[w.pos];
                w.pos += 1;
                return Some(Scored {
                    id,
                    score: w.cur_score,
                });
            }
            // Scan down to `scan_floor` (= 1 when a weight-0 gram is present, else `floor`).
            let mut refilled = false;
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

    /// The clamped raw-overlap floor (`min_shared` clamped to `[1, #postings]`). **Advisory**
    /// (v0.5): the engine does not gate on it — the consumer applies `raw_overlap ≥ floor()` to
    /// the ids the walk yields (and to any count-only candidates it recovers itself).
    pub fn floor(&self) -> u32 {
        self.floor
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

/// Build the weighted bit-sliced planes + walk metadata from the postings.
///
/// v0.4: weights are used as given (no `≥ 1` clamp). A weight-0 gram adds no energy; the walk's
/// `scan_floor` compensates (see the module docs). v0.5: the engine retains no postings and the
/// raw-overlap gate belongs to the consumer, so this returns the [`Counter`] directly.
fn plan(bitmaps: &[Bitmap], weights: Vec<u32>, min_shared: u32) -> Counter {
    let has_zero_weight = weights.contains(&0);
    // The checked sum is the documented weight precondition: `Σw` sizes the reachability table
    // and bounds the walk, so an overflowing sum is a caller error, never a silent wrap. Checked
    // first, before any plane work.
    let max_score: u32 = weights
        .iter()
        .try_fold(0u32, |acc, &w| acc.checked_add(w))
        .expect("posting weight sum overflows u32 — bound each weight by a small ceiling");

    let mut weighted: Vec<Bitmap> = Vec::new();
    for (bm, &w) in bitmaps.iter().zip(&weights) {
        add_weighted(&mut weighted, bm, w);
    }

    let floor = (min_shared as usize).min(bitmaps.len()).max(1) as u32;
    // A weight-0 gram breaks `weighted ≥ raw`, so scan every positive bucket (down to 1);
    // otherwise buckets below `floor` hold only candidates the consumer's raw gate would drop.
    let scan_floor = if has_zero_weight { 1 } else { floor };

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

    Counter {
        weighted,
        floor,
        scan_floor,
        max_score,
        reachable,
    }
}

/// Add `w` copies of `bm` into the bit-sliced planes `acc`: inject a carry at each set bit of `w`
/// and ripple it up (XOR = sum bit, AND = carry). The first ripple level folds the borrowed
/// operand directly (no copy of the posting); the carry that propagates is the intersection,
/// typically far smaller (often empty).
fn add_weighted(acc: &mut Vec<Bitmap>, bm: &Bitmap, w: u32) {
    if bm.cardinality() == 0 {
        return;
    }
    let mut bit = 0u32;
    while (w >> bit) != 0 {
        if (w >> bit) & 1 == 1 {
            let start = bit as usize;
            while acc.len() <= start {
                acc.push(Bitmap::new());
            }
            let mut carry = bm & &acc[start];
            acc[start] ^= bm;
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

    /// All-weight-1 build (the unweighted-overlap case: score == raw overlap).
    fn build_unit(bitmaps: &[Bitmap], min_shared: u32) -> Counter {
        let weights = vec![1u32; bitmaps.len()];
        Counter::build_weighted(bitmaps, weights, min_shared)
    }

    #[test]
    fn streams_best_first() {
        let counter = build_unit(
            &[
                Bitmap::of(&[1, 2, 3]),
                Bitmap::of(&[2, 3]),
                Bitmap::of(&[3]),
            ],
            1,
        );
        let got = all(&counter);
        assert_eq!(got.iter().map(|s| s.id).collect::<Vec<_>>(), [3, 2, 1]);
        assert_eq!(got[0].score, 3);
        assert!(got.windows(2).all(|p| p[0].score >= p[1].score));
    }

    #[test]
    fn all_unit_weights_scan_floor_skips_sub_floor_buckets() {
        // With every weight ≥ 1, `weighted ≥ raw`, so buckets below the floor hold only ids the
        // consumer's raw gate would drop — the scan bound excludes them without a per-id check.
        let counter = build_unit(&[Bitmap::of(&[1, 2]), Bitmap::of(&[2, 3])], 2);
        assert_eq!(all(&counter).iter().map(|s| s.id).collect::<Vec<_>>(), [2]);
    }

    #[test]
    fn mixed_weights_can_yield_a_sub_floor_id_for_the_consumer_to_gate() {
        // v0.5 contract: the engine does NOT enforce the raw-overlap floor. Id 9 matches only the
        // weight-3 posting (raw overlap 1 < floor 2) but its energy bucket (3) is ≥ the scan
        // bound, so it IS yielded — the consumer's raw gate drops it downstream.
        let heavy = Bitmap::of(&[9, 10]);
        let light = Bitmap::of(&[10]);
        let counter = Counter::build_weighted(&[heavy, light], vec![3, 1], 2);
        assert_eq!(counter.floor(), 2, "the advisory floor is reported back");
        let got = all(&counter);
        assert_eq!(
            got.iter().map(|s| (s.id, s.score)).collect::<Vec<_>>(),
            [(10, 4), (9, 3)],
            "both ids yielded best-first; gating id 9 (raw 1 < 2) is the consumer's job"
        );
    }

    #[test]
    fn weighting_promotes_rarer_grams() {
        // Explicit weights (the consumer owns the weighting model): the rare gram carries 4.
        let common = Bitmap::of(&(0..16).collect::<Vec<_>>());
        let rare = Bitmap::of(&[5]);
        let counter = Counter::build_weighted(&[common, rare], vec![1, 4], 1);
        let got = all(&counter);
        assert_eq!(got[0].id, 5);
        assert_eq!(got[0].score, 5); // 1 + 4
    }

    #[test]
    fn zero_weight_yields_nothing_from_the_bit_sliced_walk() {
        // v0.4: the old `≥ 1` clamp is removed, so a weight-0 gram contributes no energy — the
        // bit-sliced walk yields nothing for a candidate that matches only weight-0 grams
        // (max_score 0, no positive bucket). Recovering such *count-only* candidates is the
        // consumer's job (it carries the per-order / floored metadata the engine does not).
        let counter = Counter::build_weighted(&[Bitmap::of(&[9])], vec![0], 1);
        assert_eq!(counter.max_score, 0);
        assert!(all(&counter).is_empty());
    }

    #[test]
    fn zero_weight_drops_the_scan_bound_so_low_energy_ids_are_reachable() {
        // A candidate matching one weight-1 gram + one weight-0 gram has weighted energy 1 but
        // raw overlap 2. With min_shared = 2, a `scan down to floor` bound would skip bucket 1
        // and lose it; the decoupled `scan_floor = 1` (weight-0 present) reaches it. The walk
        // yields BOTH bucket-1 ids (5 and 6) — applying the raw gate (id 5: raw 2 ≥ 2 kept,
        // id 6: raw 1 < 2 dropped) is the consumer's job (v0.5).
        let w1 = Bitmap::of(&[5, 6]); // weight 1: ids 5,6
        let w0 = Bitmap::of(&[5]); // weight 0: id 5
        let counter = Counter::build_weighted(&[w1, w0], vec![1, 0], 2);
        let got = all(&counter);
        assert_eq!(
            got.iter().map(|s| (s.id, s.score)).collect::<Vec<_>>(),
            [(5, 1), (6, 1)],
            "the low-energy bucket is reachable; raw gating is downstream"
        );
    }

    #[test]
    fn empty_and_degenerate_inputs() {
        assert!(all(&build_unit(&[], 2)).is_empty());
        assert!(all(&build_unit(&[Bitmap::new()], 1)).is_empty());
    }

    #[test]
    fn high_count_across_plane_boundary() {
        let posts: Vec<Bitmap> = (0..8).map(|_| Bitmap::of(&[100])).collect();
        let counter = build_unit(&posts, 1);
        let got = all(&counter);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].score, 8);
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
        let counter = build_unit(&posts, 1);
        let full = all(&counter);
        let mut w = counter.walk();
        let top3: Vec<Scored> = (0..3).filter_map(|_| counter.advance(&mut w)).collect();
        assert_eq!(top3, full[..3]);
    }

    #[test]
    #[should_panic(expected = "weight sum overflows u32")]
    fn overflowing_weight_sum_panics_rather_than_wrapping() {
        // The documented weight precondition: Σw sizes the reachability table and bounds the
        // walk, so an overflowing sum must fail loudly (a silent wrap would corrupt scoring).
        let bitmaps = [Bitmap::of(&[1]), Bitmap::of(&[2])];
        let _ = Counter::build_weighted(&bitmaps, vec![u32::MAX, 1], 1);
    }
}
