//! Per-script-class document-frequency statistics for class-normalized rarity (§6).
//!
//! The pruner ranks query tokens rarest-first, but raw DF is not comparable across
//! `(script, gram-size)` classes — shorter grams and denser scripts (CJK's thousands of
//! base chars vs Latin's 26) live in different frequency regimes. So rarity is ranked on
//! `z = (ln DF − μ_class) / σ_class`, with `(μ, σ)` maintained **per script class** by
//! [`Welford`]'s online algorithm in **log space** (where the Zipfian DF distribution is
//! well-behaved and the variance numerically small).
//!
//! `z` is a Gaussian *proxy* for percentile — exact in the bulk, drifting in the heavy
//! rare tail the pruner selects from. It is *better-comparable* than raw DF, not
//! calibrated truth; whether it helps mixed-script recall is an empirical question (the
//! benchmark eval), not an assumption. Degenerate classes fall back to raw DF.

use std::collections::HashMap;

/// Classes with fewer than this many sampled terms fall back to raw DF — too sparse to
/// normalize. Treated as "not enough data," **not** a Gaussian-validity threshold: Zipf's
/// slow σ-convergence makes small classes *less* trustworthy than the normal-dist `n≈30`
/// intuition, so lean higher. A sweep constant, expected recall-insensitive over a wide
/// range — verify, don't over-tune.
pub(crate) const N_NORMALIZE_FLOOR: u64 = 50;

/// Floor on σ so a very tight class cannot produce explosive `z`.
const SIGMA_FLOOR: f64 = 1e-2;

/// Online mean/variance over a stream (Welford), supporting sample removal (for DF
/// decrements on delete). `m2` is the sum of squared deviations; variance is
/// `m2 / (n − 1)`.
#[derive(Clone, Copy, Default)]
pub(crate) struct Welford {
    n: u64,
    mean: f64,
    m2: f64,
}

impl Welford {
    /// Add one sample.
    pub(crate) fn add(&mut self, x: f64) {
        self.n += 1;
        let d = x - self.mean;
        self.mean += d / self.n as f64;
        self.m2 += d * (x - self.mean);
    }

    /// Remove one previously-added sample (the inverse of [`add`](Self::add)). `m2` is
    /// clamped at zero against floating-point drift.
    pub(crate) fn remove(&mut self, x: f64) {
        if self.n <= 1 {
            *self = Welford::default();
            return;
        }
        let n = self.n as f64;
        let mean_prev = (n * self.mean - x) / (n - 1.0);
        self.m2 = (self.m2 - (x - self.mean) * (x - mean_prev)).max(0.0);
        self.mean = mean_prev;
        self.n -= 1;
    }

    /// `(n, mean, σ_eff)`. `σ_eff = ∞` when undefined (`n < 2`) so a caller falls back to
    /// raw DF; otherwise `max(σ, SIGMA_FLOOR)`.
    fn snapshot(&self) -> (u64, f64, f64) {
        if self.n < 2 {
            return (self.n, self.mean, f64::INFINITY);
        }
        let var = self.m2 / (self.n as f64 - 1.0);
        (self.n, self.mean, var.sqrt().max(SIGMA_FLOOR))
    }
}

/// The live per-class accumulators (one [`Welford`] per script-tag byte). Each distinct
/// term contributes one sample `ln(df)` to its class. Recomputed from the df column on
/// open/rebuild and maintained incrementally on writes; never persisted (the df column
/// is the single source of truth → no drift). Untouched by `compact` (folds leave df
/// invariant).
pub(crate) struct ClassStats {
    by_class: Vec<Welford>, // length 256, indexed by the script-tag byte
}

impl ClassStats {
    pub(crate) fn new() -> Self {
        ClassStats {
            by_class: vec![Welford::default(); 256],
        }
    }

    /// Add a term's `ln(df)` to its class (recompute path; `df > 0`).
    pub(crate) fn add_sample(&mut self, class: u8, df: i64) {
        if df > 0 {
            self.by_class[class as usize].add((df as f64).ln());
        }
    }

    /// Update a class for a term whose df moved `old_df → new_df` (a write or delete).
    /// `old_df == 0` is an intern/resurrect; `new_df == 0` is a term leaving the index.
    pub(crate) fn update(&mut self, class: u8, old_df: i64, new_df: i64) {
        let w = &mut self.by_class[class as usize];
        if old_df > 0 {
            w.remove((old_df as f64).ln());
        }
        if new_df > 0 {
            w.add((new_df as f64).ln());
        }
    }

    /// Snapshot the given classes' stats for one query (then lock-free).
    pub(crate) fn snapshot_for(&self, classes: impl IntoIterator<Item = u8>) -> ClassSnap {
        let mut by_class = HashMap::new();
        for c in classes {
            by_class
                .entry(c)
                .or_insert_with(|| self.by_class[c as usize].snapshot());
        }
        ClassSnap { by_class }
    }
}

/// A per-query, immutable snapshot of the class stats the query's tokens touch.
pub(crate) struct ClassSnap {
    by_class: HashMap<u8, (u64, f64, f64)>, // class -> (n, mean, σ_eff)
}

impl ClassSnap {
    /// An empty snapshot — every [`rarity`](Self::rarity) falls back to raw DF. Used by
    /// the `select` unit tests.
    #[cfg(test)]
    pub(crate) fn empty() -> Self {
        ClassSnap {
            by_class: HashMap::new(),
        }
    }

    /// The class-normalized rarity of a term (lower = rarer, so the pruner sorts
    /// ascending). Returns the `z`-score for a well-populated class, else falls back to
    /// raw DF (`df` as `f64`) — keeping the rarest-first ordering well-defined either way.
    pub(crate) fn rarity(&self, df: i64, class: u8) -> f64 {
        if let Some(&(n, mean, sigma)) = self.by_class.get(&class) {
            if n >= N_NORMALIZE_FLOOR && sigma.is_finite() {
                return ((df.max(1) as f64).ln() - mean) / sigma;
            }
        }
        df as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_matches_batch_mean_and_variance() {
        let xs = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mut w = Welford::default();
        for &x in &xs {
            w.add(x);
        }
        let (n, mean, sigma) = w.snapshot();
        assert_eq!(n, 5);
        assert!((mean - 3.0).abs() < 1e-12);
        // Sample variance of 1..5 is 2.5 -> sigma ~ 1.5811.
        assert!((sigma - 2.5f64.sqrt()).abs() < 1e-9);
    }

    #[test]
    fn remove_is_the_inverse_of_add() {
        let mut w = Welford::default();
        for &x in &[2.0, 4.0, 6.0, 8.0] {
            w.add(x);
        }
        let before = w.snapshot();
        w.add(100.0);
        w.remove(100.0);
        let after = w.snapshot();
        assert_eq!(before.0, after.0);
        assert!((before.1 - after.1).abs() < 1e-9);
        assert!((before.2 - after.2).abs() < 1e-9);
    }

    #[test]
    fn degenerate_class_falls_back_to_raw_df() {
        // n < 2 -> sigma infinite -> raw df.
        let mut stats = ClassStats::new();
        stats.add_sample(1, 10); // one sample only
        let snap = stats.snapshot_for([1u8]);
        assert_eq!(snap.rarity(10, 1), 10.0);
        // Empty snapshot also falls back.
        assert_eq!(ClassSnap::empty().rarity(7, 1), 7.0);
    }

    #[test]
    fn rare_token_scores_below_common_token_in_a_populated_class() {
        // Populate a class past the floor with a spread of dfs.
        let mut stats = ClassStats::new();
        for df in 1..=100 {
            stats.add_sample(1, df);
        }
        let snap = stats.snapshot_for([1u8]);
        // A df-1 token is rarer (lower rarity) than a df-90 token.
        assert!(snap.rarity(1, 1) < snap.rarity(90, 1));
    }
}
