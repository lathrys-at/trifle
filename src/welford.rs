//! Per-script-class document-frequency statistics for class-normalized rarity.
//!
//! The pruner ranks query tokens rarest-first, but raw DF is not comparable across
//! `(script, order)` classes — shorter grams and denser scripts (CJK's thousands of
//! base chars vs Latin's 26) live in different frequency regimes. So rarity is ranked on
//! `z = (ln DF − μ_class) / σ_class`, with `(μ, σ)` maintained **per `(script, order)` class**
//! by [`Welford`]'s online algorithm in **log space** (where the Zipfian DF distribution is
//! well-behaved and the variance numerically small). v0.4/M5 splits the class key from `script`
//! to `(script, order)` because the [`DefaultTokenizer`](crate::tokenize::DefaultTokenizer) now
//! emits both a primary and a one-shorter secondary order per script (derivation §5/§8), and a
//! trigram and a bigram of the same script occupy different df regimes.
//!
//! `z` is a Gaussian *proxy* for percentile — exact in the bulk, drifting in the heavy
//! rare tail the pruner selects from. It is *better-comparable* than raw DF, not
//! calibrated truth; whether it helps mixed-script recall is an empirical question (the
//! benchmark eval), not an assumption. Degenerate classes fall back to raw DF.

use crate::hash::FxHashMap;

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

/// The composite selection-class key: a `(script, order)` pair (v0.4/M5, derivation §5/§8).
/// A Latin trigram (`order = 3`) and a Latin bigram (`order = 2`) share a script byte but live
/// in different document-frequency regimes — the [`DefaultTokenizer`](crate::tokenize::DefaultTokenizer)
/// now emits both — so rarity is normalized within `(script, order)`, not `script` alone.
/// `order ∈ 1..=3`; the flat index is `class·4 + order` (length `256·4`).
#[inline]
fn class_index(class: u8, order: u8) -> usize {
    (class as usize) * 4 + (order as usize).min(3)
}

/// Number of `(script, order)` accumulator slots: 256 scripts × 4 order slots (`order 0..=3`,
/// where `0` is an unused defensive slot — a real gram has `order ≥ 1`).
const N_CLASS_SLOTS: usize = 256 * 4;

/// The live per-`(script, order)` accumulators (one [`Welford`] each). Each distinct term
/// contributes one sample `ln(df)` to its `(script, order)` class. Recomputed from the df
/// column on open/rebuild and maintained incrementally on writes; never persisted (the df
/// column is the single source of truth → no drift). Untouched by `compact` (folds leave df
/// invariant).
pub(crate) struct ClassStats {
    by_class: Vec<Welford>, // length 256·4, indexed by `class_index(class, order)`
}

impl ClassStats {
    pub(crate) fn new() -> Self {
        ClassStats {
            by_class: vec![Welford::default(); N_CLASS_SLOTS],
        }
    }

    /// Add a term's `ln(df)` to its `(script, order)` class (recompute path; `df > 0`).
    pub(crate) fn add_sample(&mut self, class: u8, order: u8, df: i64) {
        if df > 0 {
            self.by_class[class_index(class, order)].add((df as f64).ln());
        }
    }

    /// Update a `(script, order)` class for a term whose df moved `old_df → new_df` (a write or
    /// delete). `old_df == 0` is an intern/resurrect; `new_df == 0` is a term leaving the index.
    pub(crate) fn update(&mut self, class: u8, order: u8, old_df: i64, new_df: i64) {
        let w = &mut self.by_class[class_index(class, order)];
        if old_df > 0 {
            w.remove((old_df as f64).ln());
        }
        if new_df > 0 {
            w.add((new_df as f64).ln());
        }
    }

    /// Snapshot the given `(script, order)` classes' stats for one query (then lock-free).
    pub(crate) fn snapshot_for(&self, classes: impl IntoIterator<Item = (u8, u8)>) -> ClassSnap {
        let mut by_class = FxHashMap::default();
        for (c, o) in classes {
            by_class
                .entry((c, o))
                .or_insert_with(|| self.by_class[class_index(c, o)].snapshot());
        }
        ClassSnap { by_class }
    }
}

/// A per-query, immutable snapshot of the `(script, order)`-class stats the query's tokens touch.
pub(crate) struct ClassSnap {
    by_class: FxHashMap<(u8, u8), (u64, f64, f64)>, // (class, order) -> (n, mean, σ_eff)
}

impl ClassSnap {
    /// An empty snapshot — every [`rarity`](Self::rarity) falls back to raw DF. Used by
    /// the `select` unit tests.
    #[cfg(test)]
    pub(crate) fn empty() -> Self {
        ClassSnap {
            by_class: FxHashMap::default(),
        }
    }

    /// The class-normalized rarity of a term (lower = rarer, so the pruner sorts
    /// ascending). Returns the `z`-score for a well-populated `(script, order)` class, else falls
    /// back to raw DF (`df` as `f64`) — keeping the rarest-first ordering well-defined either way.
    pub(crate) fn rarity(&self, df: i64, class: u8, order: u8) -> f64 {
        if let Some(&(n, mean, sigma)) = self.by_class.get(&(class, order)) {
            if n >= N_NORMALIZE_FLOOR && sigma.is_finite() {
                return ((df.max(1) as f64).ln() - mean) / sigma;
            }
        }
        df as f64
    }

    /// Pool the query's **present** `(script, order)` classes into one `(mean_log_df, std_log_df)`
    /// over `ln(df)`, by sample count `n` (the derived work-budget's representative-gram statistic,
    /// derivation §5/§7 — consumed by `search::derived_budget`). Only classes with
    /// a **defined** variance (`n ≥ 2`, finite `σ_eff`) contribute; a class too sparse to normalize
    /// is skipped. Returns `None` when no present class qualifies (`Σn < 2`) — the caller then falls
    /// back to an unbounded budget (recall-safe). The pool is the standard parallel-Welford
    /// combination [Chan1979]: `μ = Σ nᵢμᵢ / Σ nᵢ`, and combined `M2 = Σ[(nᵢ−1)σᵢ² + nᵢ(μᵢ−μ)²]`,
    /// so `std = √(M2/(Σnᵢ−1))`.
    pub(crate) fn pooled_log_df(&self) -> Option<(f64, f64)> {
        let mut n_total: u64 = 0;
        let mut mean_acc = 0.0; // Σ nᵢ·μᵢ
        for &(n, mean, sigma) in self.by_class.values() {
            if n >= 2 && sigma.is_finite() && mean.is_finite() {
                n_total += n;
                mean_acc += n as f64 * mean;
            }
        }
        if n_total < 2 {
            return None;
        }
        let pooled_mean = mean_acc / n_total as f64;
        let mut m2 = 0.0; // Σ [ (nᵢ−1)·σᵢ² + nᵢ·(μᵢ−μ)² ]
        for &(n, mean, sigma) in self.by_class.values() {
            if n >= 2 && sigma.is_finite() && mean.is_finite() {
                let var_i = sigma * sigma;
                m2 += (n as f64 - 1.0) * var_i + n as f64 * (mean - pooled_mean).powi(2);
            }
        }
        let pooled_var = (m2 / (n_total as f64 - 1.0)).max(0.0);
        let std = pooled_var.sqrt();
        if pooled_mean.is_finite() && std.is_finite() {
            Some((pooled_mean, std))
        } else {
            None
        }
    }

    /// The **vocabulary size** `V` (distinct in-class terms) of a `(script, order)` class — the
    /// Welford sample count. `0` if the class is absent from this snapshot. The v0.4/M5
    /// rank-view fusion uses `ln V` as the per-`(script, order)` vocabulary-complexity proxy for
    /// the fusion gap `ΔH = ln V_primary − ln V_secondary` (derivation §8): a richer (larger-vocab)
    /// primary order earns more fusion weight. `ln V` is the maximum entropy of a `V`-symbol
    /// alphabet — a directional heuristic, not a conditional entropy (§8/§11).
    pub(crate) fn vocab(&self, class: u8, order: u8) -> u64 {
        self.by_class.get(&(class, order)).map_or(0, |&(n, _, _)| n)
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
        stats.add_sample(1, 3, 10); // one sample only, class (1, order 3)
        let snap = stats.snapshot_for([(1u8, 3u8)]);
        assert_eq!(snap.rarity(10, 1, 3), 10.0);
        // Empty snapshot also falls back.
        assert_eq!(ClassSnap::empty().rarity(7, 1, 3), 7.0);
    }

    #[test]
    fn rare_token_scores_below_common_token_in_a_populated_class() {
        // Populate a class past the floor with a spread of dfs.
        let mut stats = ClassStats::new();
        for df in 1..=100 {
            stats.add_sample(1, 3, df);
        }
        let snap = stats.snapshot_for([(1u8, 3u8)]);
        // A df-1 token is rarer (lower rarity) than a df-90 token.
        assert!(snap.rarity(1, 1, 3) < snap.rarity(90, 1, 3));
    }

    #[test]
    fn same_script_different_order_are_distinct_classes() {
        // v0.4/M5 (script, order) re-key: a Latin trigram (order 3) and a Latin bigram (order 2)
        // share a script byte but accumulate into separate classes with separate df regimes.
        let mut stats = ClassStats::new();
        for df in 1..=80 {
            stats.add_sample(1, 3, df); // trigrams: dfs 1..=80
        }
        for df in 100..=300 {
            stats.add_sample(1, 2, df); // bigrams: dfs 100..=300 (a denser regime)
        }
        let snap = stats.snapshot_for([(1u8, 3u8), (1u8, 2u8)]);
        assert_eq!(snap.vocab(1, 3), 80);
        assert_eq!(snap.vocab(1, 2), 201);
        // A df-150 bigram is common-for-its-order; a df-40 trigram is mid-for-its-order. Each is
        // normalized within its own (script, order) regime, not against the other.
        let _ = snap.rarity(150, 1, 2);
        let _ = snap.rarity(40, 1, 3);
        assert_eq!(snap.vocab(9, 3), 0, "an absent class has vocab 0");
    }

    #[test]
    fn pooled_log_df_matches_a_single_class_welford() {
        // One populated class: the pool is exactly that class's (mean, σ) over ln(df).
        let mut stats = ClassStats::new();
        for df in 1..=100 {
            stats.add_sample(1, 3, df);
        }
        let snap = stats.snapshot_for([(1u8, 3u8)]);
        let (mean, std) = snap.pooled_log_df().expect("a populated class pools");
        // Compare against a direct Welford over the same ln(df) samples.
        let mut w = Welford::default();
        for df in 1..=100 {
            w.add((df as f64).ln());
        }
        let (_, wmean, wsigma) = w.snapshot();
        assert!((mean - wmean).abs() < 1e-9, "pooled mean == class mean");
        assert!((std - wsigma).abs() < 1e-9, "pooled std == class σ");
    }

    #[test]
    fn pooled_log_df_combines_two_classes_by_sample_count() {
        // Two classes pool into the combined mean/variance of all ln(df) samples (parallel Welford).
        let mut stats = ClassStats::new();
        for df in 1..=80 {
            stats.add_sample(1, 3, df);
        }
        for df in 100..=300 {
            stats.add_sample(2, 3, df);
        }
        let snap = stats.snapshot_for([(1u8, 3u8), (2u8, 3u8)]);
        let (mean, std) = snap.pooled_log_df().expect("two classes pool");
        // Oracle: one Welford over the union of both classes' samples.
        let mut w = Welford::default();
        for df in 1..=80 {
            w.add((df as f64).ln());
        }
        for df in 100..=300 {
            w.add((df as f64).ln());
        }
        let (_, wmean, wsigma) = w.snapshot();
        assert!(
            (mean - wmean).abs() < 1e-9,
            "pooled mean matches the union Welford"
        );
        assert!(
            (std - wsigma).abs() < 1e-9,
            "pooled std matches the union Welford"
        );
    }

    #[test]
    fn pooled_log_df_is_none_for_sparse_or_empty_snapshots() {
        // Empty snapshot → None.
        assert_eq!(ClassSnap::empty().pooled_log_df(), None);
        // A single-sample class (n < 2, σ_eff = ∞) does not qualify → None.
        let mut stats = ClassStats::new();
        stats.add_sample(1, 3, 10);
        let snap = stats.snapshot_for([(1u8, 3u8)]);
        assert_eq!(snap.pooled_log_df(), None);
    }
}
