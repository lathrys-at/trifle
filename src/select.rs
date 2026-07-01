//! Selection — the rarest-first token pruner with a confidence-bounded stop (query-time).
//!
//! A query's full token set is the wrong thing to scan: common tokens have O(corpus) postings and
//! little discrimination. Selection keeps a rarest-first prefix. Rarest-first is the whole point —
//! a low-frequency token is both cheapest to scan and most discriminating, perfectly correlated.
//!
//! v0.4 (§5) replaces v0.3's fixed `t_max` count cutoff with a **confidence-bounded stop** over a
//! work budget. The structure, rarest-first:
//!
//! 1. **Unconditional minimum** (kept even past the budget and the stop):
//!    - the **typo floor** `F = m + d` — the rarest `F` present grams (budget-aware: grams with
//!      `df > C` are skipped so the floor cannot blow the work budget). This is trifle's typo /
//!      partial-match tolerance — the corroboration reserve (§5) that survives a lost gram. (The
//!      derivation itself has no count floor — it leans on the §8 secondary rank-view for that
//!      robustness — so `F` and its tuned `d` are tracked legacy: deviation #8 in the §12
//!      deviations table, to be dissolved or derived once the benchmark harness gates it.)
//!    - the **per-class floor** (§5/§8) — the rarest in-corpus (`df > 0`, `df ≤ C`) gram of every
//!      present `(script, order)` class, so per-script representation is an invariant for every
//!      present class with an in-budget gram (dual-order: a script's primary and secondary orders
//!      are distinct classes with distinct seats).
//! 2. **Skip-and-continue budget + Cantelli stop** over the rest, rarest-first: a gram that would
//!    breach the work budget `C` is **skipped** and scanning continues (the class-normalized order
//!    is *not* df-monotone, §5), and collection **stops** once the running identification evidence
//!    clears the target with margin (§5):
//!
//!    `Σ_{g∈P} r·max(0,E_g) − c·σ_match ≥ ln(N/k)`, with the comonotone per-word-block variance
//!    `σ_match² = r(1−r)·Σ_blocks (Σ_{g∈block} max(0,E_g))²` (a block = one query word).
//!
//!    Floored (junk-suspect) grams carry no identification power, so they are **excluded** from the
//!    stop's running mean and variance (they are still admitted and still cost their postings). The
//!    stop is gated to fire only when it can be recall-safe, `r ≥ c²/(B+c²)` with `B` the number of
//!    independent query-word blocks collected (§5/§11) — otherwise collection runs to the budget.
//! 3. **Rescue** (§7/§12): if the budget skipped every present gram, keep the single rarest one
//!    (one posting walk) so a query is never starved to empty when it has an in-corpus gram.
//!
//! Because the unconditional minimum is now `df ≤ C`-bounded and the budget cutoff skips rather
//! than breaks, `Σdf` over the selected set is bounded by `≈ (F + #classes + 1)·C` — O(C) for a
//! fixed, small number of script classes — so the candidate union the engine walks is finally
//! bounded by the work budget even for short / common queries (which were O(N) through M3) —
//! *except* the §7/§12 rescue (step 3 below), which may walk one over-budget posting when **every**
//! present gram is `df > C` (a pure-common query); `Σdf` is then O(N), the recall-floor that beats
//! returning an empty result. v0.4/M5 note: this single-posting rescue carve-out is per *view*, so a
//! **starved** query that falls to the secondary (`search.rs::plan_views`) can hit it on BOTH the
//! primary and the secondary view — at most **two** O(N) postings (and the unigram secondary's
//! postings are the denser of the two). Still bounded (≤ 2), behind the per-script starved gate, and
//! a ratified recall-vs-flatness tradeoff (the secondary only runs when a script is starved).
//!
//! Selection derives only from this query's own per-gram inputs and the shared snapshot it is
//! handed (the per-class stats, `N`, `σ`, the energy/floor — all read once per batch) — never a
//! batch aggregate — so `search_batch([…, q, …])` ranks `q` identically to `search(q)`
//! (**batch == serial**). The stop is now part of that guarantee: it is a pure function of this
//! query's grams plus the shared snapshot.
//!
//! "Rarest" is **class-normalized**: grams are ranked by [`ClassSnap::rarity`] (a `z`-score within
//! the gram's script class, falling back to raw df for a degenerate class), so a CJK bigram and a
//! Latin trigram — different df regimes — compare on the same footing. Within a single script (the
//! common per-query case) this reduces to rarest-by-df, with a true-df-ascending tie-break (§5).

use crate::hash::{FxHashMap, FxHashSet};
use crate::welford::ClassSnap;

/// One query gram, with everything the pruner + the §5 stop need: its document frequency, its
/// `(script, order)` selection class, the comonotone-block (query-word) id, its logit-idf energy
/// `E_g` (raw; clamped to `max(0,·)` at use), and whether it is **floored** (query-side
/// `df ≤ df_min`, a substitution-artifact suspect — §4). All N-anchored quantities (`energy`,
/// `floored`) are computed by the caller from the shared per-batch snapshot, so [`select`] stays a
/// pure function of these inputs (batch == serial).
#[derive(Clone, Debug)]
pub(crate) struct GramRow<Tk> {
    /// The gram token (returned in the selection on the present/absent rules below).
    pub token: Tk,
    /// Live document frequency. `> 0` is present; `≤ 0` is absent (empty posting).
    pub df: i64,
    /// The script class byte (the Welford class id) — the gram's strong script.
    pub class: u8,
    /// The gram's order `n` (codepoint count): a CJK bigram is `2`, else `3`, one less for a
    /// secondary-view gram. Part of the `(script, order)` per-class-floor key (§5/§8).
    pub order: u8,
    /// The comonotone stopping-block id — this gram's query word (§5). Grams sharing a word are one
    /// Bernoulli block in the stop's variance.
    pub word: u32,
    /// The logit-idf energy `E_g` (derivation §2/§4), raw — `select` clamps it to `max(0,·)` when
    /// folding it into the stop's mean / variance, mirroring the accumulator.
    pub energy: f64,
    /// Query-side floored flag (`df ≤ df_min`, §4): a substitution-artifact suspect. Floored grams
    /// are excluded from the stop's running mean + variance (no identification power, §5/§9) but are
    /// still admitted and still counted toward the work budget.
    pub floored: bool,
}

/// The behavior + performance knobs the pruner reads (from [`SearchOpts`](crate::SearchOpts) and
/// the index config), resolved once per batch.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SelectParams {
    /// `m` — the match floor (the strictness dial).
    pub min_shared: u32,
    /// `d` — per-typo token damage; the typo floor is `F = m + d`.
    pub typo_damage: u32,
    /// `C` — the work budget: a cap on the cumulative document frequency (`Σdf`, the posting-list
    /// scan cost) of the selected grams (derivation §5/§7). The unconditional minimum keeps only
    /// `df ≤ C` grams; beyond it, a gram that would breach `C` is **skipped** (scanning continues).
    /// `None` = no cap.
    pub df_budget: Option<u64>,
    /// `c` — the Cantelli stopping margin (a distribution-free bound, not a z-score; §5).
    pub c_margin: f64,
    /// `k` — the stop's target candidate-pool size; the stop aims for `ln(N/k)` nats (§5).
    pub k_target: u64,
    /// `N` — the snapshot's segment count, for the stop target `ln(N/k)` (§5). Read once per batch.
    pub n_segments: u64,
    /// `σ`/`r` — the query-side reliability driving the stop's mean `Σ r·max(0,E)` and variance
    /// `r(1−r)·…` (§3/§5). Query-side this is uniform across grams; doc-side (M5) is per-gram.
    pub sigma: f64,
}

/// The running state of the §5 confidence-bounded stop: the cumulative work `Σdf`, the
/// reliability-weighted energy mean `Σ r·max(0,E)`, the comonotone per-word-block variance
/// `r(1−r)·Σ_blocks (Σe)²`, and the per-block running energy sums it is built from.
///
/// A floored gram (or a non-floored gram whose clamped energy is `0`, i.e. a common gram carrying
/// no identification power) contributes to `sum_df` but **not** to the mean / variance / block set —
/// so it never lets the stop fire early, and never inflates the block count `B` used in the
/// recall-safety gate.
struct StopAcc {
    /// `Σdf` of all admitted grams (the work-budget cost; floored and zero-energy grams included).
    sum_df: u64,
    /// `Σ_{g∈P, e_g>0} r·e_g`, `e_g = max(0, E_g)` — the stop's mean evidence (§5).
    sum_e: f64,
    /// `r(1−r)·Σ_blocks (Σ_{g∈block} e_g)²` — the comonotone per-word-block variance (§5).
    sum_var: f64,
    /// Per query-word running `Σ e_g` over admitted positive-energy non-floored grams — the partial
    /// block sums the variance is accumulated from incrementally.
    block_sum: FxHashMap<u32, f64>,
}

impl StopAcc {
    fn new() -> Self {
        StopAcc {
            sum_df: 0,
            sum_e: 0.0,
            sum_var: 0.0,
            block_sum: FxHashMap::default(),
        }
    }

    /// Fold one admitted gram into the running stop state. `sum_df` always grows (every admitted
    /// posting costs its df); the mean / variance / block set grow only for a **non-floored** gram
    /// with **positive** clamped energy (§5/§9 — floored and zero-energy grams carry no
    /// identification power). `r` is the gram's reliability (query-side: the uniform `σ`).
    fn admit(&mut self, df: i64, energy: f64, floored: bool, word: u32, r: f64) {
        self.sum_df = self.sum_df.saturating_add(df.max(0) as u64);
        if floored {
            return;
        }
        let e = energy.max(0.0);
        if e <= 0.0 {
            return;
        }
        // Comonotone block variance, accumulated incrementally: adding `e` to a block whose running
        // sum is `s` raises (Σe)² by `(s+e)² − s² = 2·s·e + e²` (§5/§12). `φ = 1` (one query word
        // is one comonotone unit), so distinct words are independent and their variances sum.
        let s = self.block_sum.get(&word).copied().unwrap_or(0.0);
        self.sum_e += r * e;
        self.sum_var += r * (1.0 - r) * (2.0 * s * e + e * e);
        self.block_sum.insert(word, s + e);
    }

    /// Whether the §5 stop has fired: `Σ r·e − c·σ_match ≥ ln(N/k)`, gated on the recall-safety
    /// condition `r ≥ c²/(B+c²)` (with `B` the number of independent query-word blocks collected,
    /// §5/§11). The gate matters when the target is non-positive (a tiny corpus, `N < k`, where the
    /// bare inequality would fire trivially) and doc-side (low `r`); when the target is positive the
    /// gate is implied, since the inequality holding against a positive target already requires
    /// `Σ r·e − c·σ_match > 0`, which **implies** `r > c²/(B+c²)` (with equality at `B = 1`, by
    /// Cauchy–Schwarz; for `B ≥ 2` unequal blocks `Σ r·e − c·σ_match > 0` is strictly stronger, so
    /// it still implies the gate).
    ///
    /// The gate itself is §5's *equal-block* bound: for `B ≥ 2` unequal blocks against a non-positive
    /// target (a tiny corpus, `N < k`) it can pass while `Σ r·e − c·σ_match` is still negative — a
    /// bounded, recall-safe residual (§5/§11), since the always-kept unconditional minimum is never
    /// reduced and a sub-`k`-segment corpus has a tiny union anyway.
    fn fired(&self, c: f64, target: f64, r: f64) -> bool {
        let b = self.block_sum.len() as f64;
        if b <= 0.0 {
            return false;
        }
        if r < (c * c) / (b + c * c) {
            return false;
        }
        self.sum_e - c * self.sum_var.sqrt() >= target
    }
}

/// Select the grams to scan, from each distinct query gram with its inputs ([`GramRow`]).
///
/// Returns the selection in scan order: kept **present** (`df > 0`) grams rarest-first (by the §5
/// rules above), then **absent** (`df ≤ 0`) grams in token order (charged nothing — an empty
/// posting costs nothing to scan and never reaches the overlap floor; kept for span location at
/// hydrate). Deterministic run to run.
pub(crate) fn select<Tk: Clone + Ord>(
    rows: &[GramRow<Tk>],
    params: SelectParams,
    classes: &ClassSnap,
) -> Vec<Tk> {
    // Present grams, rarest-first: class-normalized rarity, then true-df ascending (the §5 tie-break
    // that resolves z-score ties and keeps the cheapest first in the floored tail), then token (a
    // deterministic final tie-break so the hash-set dedup upstream cannot leak nondeterminism).
    // Decorate-sort-undecorate: the rarity key (a `ln` per gram) is computed once per gram, not
    // per comparison — `O(n)` transcendentals instead of `O(n log n)`.
    let mut decorated: Vec<(f64, &GramRow<Tk>)> = rows
        .iter()
        .filter(|r| r.df > 0)
        .map(|r| (classes.rarity(r.df, r.class, r.order), r))
        .collect();
    decorated.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.df.cmp(&b.1.df))
            .then_with(|| a.1.token.cmp(&b.1.token))
    });
    let present: Vec<&GramRow<Tk>> = decorated.into_iter().map(|(_, r)| r).collect();

    let n = present.len();
    let cmax: u64 = params.df_budget.unwrap_or(u64::MAX);
    let r = params.sigma;
    let c = params.c_margin;
    let k = params.k_target.max(1);
    let nseg = params.n_segments.max(1);
    let target = (nseg as f64 / k as f64).ln();

    // The typo floor `F = m + d`, clamped to the match floor and to what's present. With `t_max`
    // removed (v0.4/M6), count is bounded only by the query's finite gram set and work by the
    // budget `C`; selection is `F + per-class floors + rarest-first until (Cantelli stop ⊓ C)`.
    let f = params.min_shared as usize + params.typo_damage as usize;
    let floor = f.max(params.min_shared as usize).min(n);

    let mut kept = vec![false; n];
    let mut n_kept = 0usize;
    let mut acc = StopAcc::new();

    // (1a) Per-class floor (§5/§8): the rarest in-budget (`df ≤ C`) gram of every present
    // `(script, order)` class — guarantees one walked posting per representable class. A class whose
    // only grams are `df > C` is excluded by the work budget (a small, bounded recall cost, §5).
    let mut seated: FxHashSet<(u8, u8)> = FxHashSet::default();
    for i in 0..n {
        let row = present[i];
        if row.df as u64 > cmax {
            continue;
        }
        if seated.insert((row.class, row.order)) && !kept[i] {
            kept[i] = true;
            n_kept += 1;
            acc.admit(row.df, row.energy, row.floored, row.word, r);
        }
    }

    // (1b) Typo floor `F` (§5 corroboration reserve): the rarest `F` present grams that fit the
    // budget (`df ≤ C`; an over-budget gram is skipped so the floor cannot blow the budget — the
    // O(C) bound). Kept unconditionally, even past the stop.
    let mut f_count = 0usize;
    for i in 0..n {
        if f_count >= floor {
            break;
        }
        let row = present[i];
        if row.df as u64 > cmax {
            continue;
        }
        f_count += 1;
        if !kept[i] {
            kept[i] = true;
            n_kept += 1;
            acc.admit(row.df, row.energy, row.floored, row.word, r);
        }
    }

    // (2) Skip-and-continue budget + Cantelli stop over the rest, rarest-first. Skip the greedy
    // entirely if the unconditional minimum already cleared the target.
    if !acc.fired(c, target, r) {
        for i in 0..n {
            if kept[i] {
                continue;
            }
            let row = present[i];
            if acc.sum_df.saturating_add(row.df.max(0) as u64) > cmax {
                continue; // skip-and-continue: over budget, keep scanning (z-order is not df-monotone)
            }
            kept[i] = true;
            n_kept += 1;
            acc.admit(row.df, row.energy, row.floored, row.word, r);
            if acc.fired(c, target, r) {
                break; // enough class-fair identification evidence
            }
        }
    }

    // Kept present grams, rarest-first.
    let mut out: Vec<Tk> = Vec::with_capacity(n_kept + 1);
    for i in 0..n {
        if kept[i] {
            out.push(present[i].token.clone());
        }
    }
    // Rescue (§7/§12): a non-empty present set must yield at least one gram — if the budget skipped
    // every present gram (all `df > C`), keep the single rarest (one posting walk).
    if out.is_empty() && n > 0 {
        out.push(present[0].token.clone());
    }

    // Append the absent (df ≤ 0) grams in token order, charged nothing.
    let mut absent: Vec<Tk> = rows
        .iter()
        .filter(|r| r.df <= 0)
        .map(|r| r.token.clone())
        .collect();
    absent.sort();
    out.extend(absent);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::welford::{ClassSnap, ClassStats};

    /// A high-`σ`, lenient default: no stop firing on these tiny fixtures unless the target is
    /// cleared, a generous (unbounded) budget. `min_shared` is the knob under test.
    fn params(min_shared: u32) -> SelectParams {
        SelectParams {
            min_shared,
            typo_damage: 4,
            df_budget: None,
            c_margin: 2.0,
            k_target: 128,
            // N huge so the stop target ln(N/k) is large and never trivially clears on these tiny
            // fixtures — these tests exercise the floor / budget / ordering, not the stop.
            n_segments: 1_000_000,
            sigma: 0.9,
        }
    }

    /// `(name, df)` pairs as single-class single-order [`GramRow`]s (class 0, order 3). With an
    /// empty [`ClassSnap`] the pruner ranks by raw df — these exercise that path.
    fn rows(pairs: &[(&str, i64)]) -> Vec<GramRow<String>> {
        pairs
            .iter()
            .map(|(t, d)| GramRow {
                token: t.to_string(),
                df: *d,
                class: 0,
                order: 3,
                word: 0,
                energy: 0.0,
                floored: false,
            })
            .collect()
    }

    /// The raw-df (empty) class snapshot these tests rank against.
    fn snap() -> ClassSnap {
        ClassSnap::empty()
    }

    #[test]
    fn keeps_the_typo_floor_rarest_present_grams() {
        // 8 present, m=2, d=4 -> F=6. A budget that admits the 6 rarest (df ≤ 6) but skips the two
        // commonest (df 7, 8) isolates the floor: keep exactly the 6 rarest, rarest-first.
        let t = rows(&[
            ("a", 1),
            ("b", 2),
            ("c", 3),
            ("d", 4),
            ("e", 5),
            ("f", 6),
            ("g", 7),
            ("h", 8),
        ]);
        let p = SelectParams {
            df_budget: Some(6),
            ..params(2)
        };
        assert_eq!(select(&t, p, &snap()), ["a", "b", "c", "d", "e", "f"]);
    }

    #[test]
    fn floor_caps_at_present_count() {
        let t = rows(&[("a", 1), ("b", 2)]); // only 2 present, F would be 6
        assert_eq!(select(&t, params(2), &snap()), ["a", "b"]);
    }

    #[test]
    fn floor_clamps_when_m_exceeds_present_count() {
        // m=5 -> F=9, but only 2 grams present: keep both, not 9.
        let t = rows(&[("a", 1), ("b", 2)]);
        assert_eq!(select(&t, params(5), &snap()).len(), 2);
    }

    #[test]
    fn extreme_min_shared_does_not_overflow() {
        // A pathological `min_shared` near u32::MAX must not panic on the floor add (debug overflow
        // check); the floor just clamps to what is present.
        let t = rows(&[("a", 1), ("b", 2), ("c", 3)]);
        let kept = select(&t, params(u32::MAX), &snap());
        assert_eq!(kept, ["a", "b", "c"]); // clamped to present.len()
    }

    #[test]
    fn skip_and_continue_keeps_cheaper_later_grams() {
        // Budget cutoff is skip-and-continue, not break: an over-budget gram is skipped and a later,
        // cheaper gram is still admitted. m=1,d=4 -> F=5 (unconditional). Past F, with budget 100, a
        // df=1000 gram is skipped but a following df=2 gram fits. (Stop target huge -> never fires.)
        let t = rows(&[
            ("a", 1),
            ("b", 1),
            ("c", 1),
            ("d", 1),
            ("e", 1),      // F=5 floor (Σdf 5)
            ("big", 1000), // would breach budget -> skipped
            ("z", 2),      // still fits -> admitted
        ]);
        let p = SelectParams {
            df_budget: Some(100),
            ..params(1)
        };
        let kept = select(&t, p, &snap());
        assert!(
            kept.contains(&"z".to_string()),
            "cheaper later gram admitted"
        );
        assert!(
            !kept.contains(&"big".to_string()),
            "over-budget gram skipped, not a break"
        );
    }

    #[test]
    fn typo_floor_is_budget_aware_skips_over_budget_grams() {
        // The unconditional typo floor itself skips df>C grams, so it cannot blow the budget (the
        // O(C) bound). m=1,d=1 -> F=2. Budget 5: the two df=1000 grams are skipped while filling F;
        // the two df=2 grams seat the floor instead.
        let t = rows(&[("p", 1000), ("q", 1000), ("r", 2), ("s", 2), ("t", 2)]);
        let p = SelectParams {
            min_shared: 1,
            typo_damage: 1,
            df_budget: Some(5),
            ..params(1)
        };
        let kept = select(&t, p, &snap());
        assert!(
            !kept.contains(&"p".to_string()) && !kept.contains(&"q".to_string()),
            "over-budget grams are skipped even by the unconditional floor"
        );
        assert!(
            kept.contains(&"r".to_string()),
            "cheap grams seat the floor"
        );
    }

    #[test]
    fn rescue_keeps_the_rarest_when_budget_skips_everything() {
        // Every present gram is over budget: the rescue keeps the single rarest (one posting walk),
        // so a query with an in-corpus gram is never starved to empty (§7/§12).
        let t = rows(&[("a", 500), ("b", 600), ("c", 700)]);
        let p = SelectParams {
            df_budget: Some(10),
            ..params(2)
        };
        let kept = select(&t, p, &snap());
        assert_eq!(
            kept,
            ["a"],
            "rarest present gram rescued despite over-budget"
        );
    }

    #[test]
    fn absent_grams_kept_uncharged_and_after_present() {
        let t = rows(&[("z", 0), ("a", 5), ("y", 0), ("b", 3)]);
        let kept = select(&t, params(2), &snap());
        // present rarest-first: b(3), a(5); then absent in token order: y, z.
        assert_eq!(kept, ["b", "a", "y", "z"]);
    }

    #[test]
    fn df_ties_break_on_true_df_then_token() {
        // Equal df, supplied out of token order; the kept prefix is token-ascending.
        let t = rows(&[("c", 5), ("a", 5), ("b", 5)]);
        let kept = select(
            &t,
            SelectParams {
                typo_damage: 0,
                ..params(3)
            },
            &snap(),
        );
        assert_eq!(kept, ["a", "b", "c"]);
    }

    #[test]
    fn batch_equals_serial_is_a_pure_function_of_inputs() {
        // Identical inputs -> identical output (incl. the stop), regardless of any external state.
        let t = rows(&[("a", 1), ("b", 2), ("c", 9)]);
        assert_eq!(
            select(&t, params(2), &snap()),
            select(&t, params(2), &snap())
        );
    }

    #[test]
    fn class_normalized_rarity_reorders_across_classes() {
        // Two classes with different df regimes: class 1 ("dense", high dfs) and class 2 ("sparse",
        // low dfs). Populate both past the normalize floor.
        let mut stats = ClassStats::new();
        for df in 50..=200 {
            stats.add_sample(1, 3, df);
        }
        for df in 1..=60 {
            stats.add_sample(2, 3, df);
        }
        let cs = stats.snapshot_for([(1u8, 3u8), (2u8, 3u8)]);
        // A df-40 gram in the dense class is rarer-for-its-kind than a df-40 gram in the sparse
        // class, so it sorts first even though raw df ties.
        let t = vec![
            GramRow {
                token: "sparse40".to_string(),
                df: 40,
                class: 2,
                order: 3,
                word: 0,
                energy: 0.0,
                floored: false,
            },
            GramRow {
                token: "dense40".to_string(),
                df: 40,
                class: 1,
                order: 3,
                word: 0,
                energy: 0.0,
                floored: false,
            },
        ];
        let kept = select(&t, params(1), &cs);
        assert_eq!(kept, ["dense40", "sparse40"]);
    }

    // ----- the §5 confidence-bounded stop --------------------------------------------------------

    /// Build present rows in one word (block 0) with explicit energies, single class/order.
    fn energy_rows(specs: &[(&str, i64, f64)]) -> Vec<GramRow<String>> {
        specs
            .iter()
            .map(|(t, df, e)| GramRow {
                token: t.to_string(),
                df: *df,
                class: 0,
                order: 3,
                word: 0,
                energy: *e,
                floored: false,
            })
            .collect()
    }

    #[test]
    fn stop_halts_collection_once_evidence_clears_the_target() {
        // High-energy rare grams clear ln(N/k) quickly, so collection stops short of all grams even
        // though the budget is generous. m=1,d=0 -> F=1 (minimal floor) so the stop, not the floor,
        // governs. N=1000,k=128 -> target ≈ ln(7.8) ≈ 2.05; σ=0.9,c=2.
        let t = energy_rows(&[
            ("g0", 3, 6.0),
            ("g1", 3, 6.0),
            ("g2", 3, 6.0),
            ("g3", 3, 6.0),
            ("g4", 3, 6.0),
            ("g5", 3, 6.0),
        ]);
        let p = SelectParams {
            min_shared: 1,
            typo_damage: 0,
            n_segments: 1000,
            k_target: 128,
            ..params(1)
        };
        let kept = select(&t, p, &snap());
        // Each gram in its own evidence: with one block (word 0), one or two high-energy grams clear
        // the target — far fewer than all six.
        assert!(
            kept.len() < 6,
            "the stop halts before consuming every gram ({} kept)",
            kept.len()
        );
        assert!(!kept.is_empty(), "but keeps at least the floor");
    }

    #[test]
    fn recall_safety_gate_blocks_an_early_fire_at_low_r() {
        // The gate r ≥ c²/(B+c²): at c=2, B=1 the threshold is 0.8. A low r=0.5 must NOT let the
        // stop fire (it would be recall-unsafe), so collection runs to the budget / ceiling.
        let t = energy_rows(&[
            ("g0", 3, 8.0),
            ("g1", 3, 8.0),
            ("g2", 3, 8.0),
            ("g3", 3, 8.0),
        ]);
        let low_r = SelectParams {
            min_shared: 1,
            typo_damage: 0,
            n_segments: 1000,
            k_target: 128,
            sigma: 0.5, // below the 0.8 gate at B=1
            ..params(1)
        };
        let kept_low = select(&t, low_r, &snap());
        assert_eq!(kept_low.len(), 4, "low r cannot fire the stop -> keeps all");

        // The same query at the default high r=0.9 (above the gate) fires the stop and keeps fewer.
        let high_r = SelectParams {
            sigma: 0.9,
            ..low_r
        };
        let kept_high = select(&t, high_r, &snap());
        assert!(
            kept_high.len() < kept_low.len(),
            "high r (above the gate) fires the stop earlier ({} < {})",
            kept_high.len(),
            kept_low.len()
        );
    }

    #[test]
    fn block_variance_widens_with_more_words_delaying_the_stop() {
        // Same grams, same energies, same count — but spread across distinct query words (blocks)
        // the comonotone variance is SMALLER (independent blocks) than packed into one word, so the
        // multi-word query's stop fires at least as early: more confident. Conversely packing all
        // grams into one comonotone block maximizes σ_match and delays the stop. Assert the
        // one-block query keeps >= the multi-block query (never fewer).
        let one_block: Vec<GramRow<String>> = (0..6)
            .map(|i| GramRow {
                token: format!("g{i}"),
                df: 3,
                class: 0,
                order: 3,
                word: 0, // all one word -> one comonotone block (max variance)
                energy: 3.0,
                floored: false,
            })
            .collect();
        let multi_block: Vec<GramRow<String>> = (0..6)
            .map(|i| GramRow {
                token: format!("g{i}"),
                df: 3,
                class: 0,
                order: 3,
                word: i as u32, // each its own word -> independent blocks (min variance)
                energy: 3.0,
                floored: false,
            })
            .collect();
        let p = SelectParams {
            min_shared: 1,
            typo_damage: 0,
            n_segments: 1000,
            k_target: 128,
            ..params(1)
        };
        let one = select(&one_block, p, &snap()).len();
        let multi = select(&multi_block, p, &snap()).len();
        assert!(
            one >= multi,
            "one comonotone block (higher variance) keeps >= the multi-block query: {one} >= {multi}"
        );
    }

    #[test]
    fn floored_grams_excluded_from_the_stop_but_still_kept() {
        // Floored grams carry no identification power: they never fire the stop (excluded from the
        // mean/variance), so an all-floored query collects to the floor/budget rather than stopping
        // on phantom evidence — yet the floored grams are still admitted (they cost their postings).
        let t: Vec<GramRow<String>> = (0..5)
            .map(|i| GramRow {
                token: format!("j{i}"),
                df: 2,
                class: 0,
                order: 3,
                word: 0,
                energy: 6.9, // E_max-level, but floored -> excluded
                floored: true,
            })
            .collect();
        let p = SelectParams {
            min_shared: 1,
            typo_damage: 0,
            n_segments: 1000,
            k_target: 128,
            ..params(1)
        };
        let kept = select(&t, p, &snap());
        assert_eq!(
            kept.len(),
            5,
            "all-floored query is not stopped early by phantom evidence"
        );
    }

    #[test]
    fn tiny_corpus_keeps_the_floor_despite_negative_target() {
        // N < k -> ln(N/k) < 0 -> the bare inequality would fire trivially. The floor F is still
        // kept (typo tolerance), and the gate (σ=0.9 ≥ 0.8 at B=1) lets the stop cap collection at
        // the minimum rather than over-collecting. The key property: the floor F survives.
        let t = energy_rows(&[
            ("g0", 1, 5.0),
            ("g1", 1, 5.0),
            ("g2", 1, 5.0),
            ("g3", 1, 5.0),
            ("g4", 1, 5.0),
            ("g5", 1, 5.0),
        ]);
        let p = SelectParams {
            min_shared: 2,
            typo_damage: 4, // F = 6
            n_segments: 3,  // tiny: target ln(3/128) < 0
            k_target: 128,
            ..params(2)
        };
        let kept = select(&t, p, &snap());
        assert!(
            kept.len() >= 6,
            "the typo floor F=6 survives a negative target"
        );
    }

    #[test]
    fn per_class_floor_seats_every_present_class() {
        // Two scripts: a "dense" majority class (1) with several grams, and a minority class (2)
        // with a single rare gram. A tight budget + the stop would otherwise spend everything on the
        // majority; the per-class floor must still seat the minority class's gram.
        let mut t = vec![GramRow {
            token: "minority".to_string(),
            df: 5,
            class: 2,
            order: 3,
            word: 1,
            energy: 4.0,
            floored: false,
        }];
        for i in 0..6 {
            t.push(GramRow {
                token: format!("maj{i}"),
                df: 4,
                class: 1,
                order: 3,
                word: 0,
                energy: 5.0,
                floored: false,
            });
        }
        let p = SelectParams {
            min_shared: 1,
            typo_damage: 0,
            df_budget: Some(50),
            n_segments: 1000,
            k_target: 128,
            ..params(1)
        };
        let kept = select(&t, p, &snap());
        assert!(
            kept.contains(&"minority".to_string()),
            "the per-class floor seats the minority script's gram even under a tight stop/budget"
        );
    }

    #[test]
    fn stopacc_b_zero_never_fires() {
        // Non-floored grams whose clamped energy is 0 (common grams) form no positive-energy block,
        // so B stays 0 and the stop CANNOT fire — even against a wildly negative target. (Distinct
        // from the all-floored path, which excludes via the floored flag.)
        let mut acc = StopAcc::new();
        for _ in 0..5 {
            acc.admit(3, -1.0, false, 0, 0.9); // E < 0 -> clamps to 0 -> excluded
        }
        assert_eq!(acc.block_sum.len(), 0, "no positive-energy block formed");
        assert!(
            !acc.fired(2.0, -100.0, 0.9),
            "B=0 must never fire even against a wildly negative target"
        );
    }

    #[test]
    fn gate_boundary_is_inclusive_at_b_one() {
        // The recall-safety gate r ≥ c²/(B+c²). At c=2, B=1 the threshold is exactly 0.8. One block,
        // energy 10: against a zero target the stop fires iff r ≥ 0.8 (the gate is inclusive).
        let mk = |r: f64| {
            let mut acc = StopAcc::new();
            acc.admit(3, 10.0, false, 0, r);
            acc
        };
        assert!(
            !mk(0.79).fired(2.0, 0.0, 0.79),
            "r below the 0.8 gate cannot fire"
        );
        assert!(
            mk(0.80).fired(2.0, 0.0, 0.80),
            "r exactly at the 0.8 gate clears a zero target (inclusive)"
        );
        assert!(mk(0.81).fired(2.0, 0.0, 0.81), "r above the gate fires");
    }

    #[test]
    fn floored_and_zero_energy_keep_count_but_form_no_block() {
        // The two exclusion paths (floored, and clamped-energy 0) are independent and both keep B at
        // 0 while still charging the work budget (sum_df).
        let mut acc = StopAcc::new();
        acc.admit(7, 6.9, true, 0, 0.9); // floored -> excluded from mean/var, sum_df += 7
        acc.admit(4, 0.0, false, 1, 0.9); // e == 0 -> excluded, sum_df += 4
        assert_eq!(acc.sum_df, 11, "both still cost their postings");
        assert_eq!(acc.block_sum.len(), 0, "neither forms an evidence block");
        assert_eq!(acc.sum_e, 0.0);
        assert_eq!(acc.sum_var, 0.0);
    }

    #[test]
    fn single_gram_query_keeps_the_one_gram() {
        let t = energy_rows(&[("only", 4, 6.0)]);
        let p = SelectParams {
            min_shared: 1,
            typo_damage: 0,
            n_segments: 1000,
            k_target: 128,
            ..params(1)
        };
        assert_eq!(select(&t, p, &snap()), ["only"], "the lone gram is kept");
    }

    #[test]
    fn greedy_skipped_when_minimum_already_clears_target() {
        // F=2 high-energy grams clear a small target, so the remaining 4 must NOT be collected.
        let t = energy_rows(&[
            ("a", 3, 9.0),
            ("b", 3, 9.0),
            ("c", 3, 9.0),
            ("d", 3, 9.0),
            ("e", 3, 9.0),
            ("f", 3, 9.0),
        ]);
        let p = SelectParams {
            min_shared: 1,
            typo_damage: 1, // F = 2
            n_segments: 1000,
            k_target: 128,
            ..params(1)
        };
        let kept = select(&t, p, &snap());
        assert_eq!(
            kept.len(),
            2,
            "greedy is skipped once the floor already fired the stop ({kept:?})"
        );
    }

    #[test]
    fn per_class_floor_drops_an_over_budget_only_class() {
        // A class whose only present gram is df>C is excluded by the work budget — NOT rescued into a
        // floor seat (the bounded recall cost of §5), while an in-budget class is seated.
        let mut t = energy_rows(&[("maj", 3, 5.0)]); // class 0, df=3, in budget
        t.push(GramRow {
            token: "expensive".to_string(),
            df: 1000, // > C
            class: 2,
            order: 3,
            word: 1,
            energy: 6.0,
            floored: false,
        });
        let p = SelectParams {
            min_shared: 1,
            typo_damage: 0,
            df_budget: Some(50),
            n_segments: 1000,
            k_target: 128,
            ..params(1)
        };
        let kept = select(&t, p, &snap());
        assert!(kept.contains(&"maj".to_string()), "in-budget class seated");
        assert!(
            !kept.contains(&"expensive".to_string()),
            "a class whose only gram is df>C is dropped, not rescued into a floor seat"
        );
    }
}
