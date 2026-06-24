//! Selection — the cost-budget pruner (query-time).
//!
//! A query's full token set is the wrong thing to scan: common tokens have
//! O(corpus) postings and little discrimination. Selection keeps a rarest-first
//! prefix. Rarest-first is the whole point — a low-frequency token is *both*
//! cheapest to scan and most discriminating, perfectly correlated.
//!
//! Selection derives only from this query's own token document frequencies — never
//! a batch aggregate or the corpus size — so `search_batch([…, q, …])` ranks `q`
//! identically to `search(q)` (batch == serial).

/// The behavior + performance knobs the pruner reads (assembled from
/// [`SearchOpts`](crate::SearchOpts) and [`Advanced`](crate::Advanced)).
#[derive(Clone, Copy, Debug)]
pub(crate) struct SelectParams {
    /// `m` — the match floor (the strictness dial).
    pub min_shared: u32,
    /// `B` — the breadth budget, in cost units. `0` keeps exactly the typo floor.
    pub breadth: u64,
    /// `d` — per-typo token damage; the typo floor is `F = m + d`.
    pub typo_damage: u32,
    /// `k_max` — the absolute ceiling on kept tokens.
    pub k_max: usize,
    /// `α` — per-kept-list cost coefficient.
    pub alpha: f64,
    /// `β` — per-document-frequency cost coefficient.
    pub beta: f64,
}

impl SelectParams {
    /// The two-term scan cost of keeping `n_kept` tokens whose frequencies sum to
    /// `sum_df`: `α·n_kept + β·Σdf`.
    fn cost(&self, n_kept: usize, sum_df: i64) -> f64 {
        self.alpha * n_kept as f64 + self.beta * sum_df.max(0) as f64
    }
}

/// Select the tokens to scan, from each distinct query token paired with its live
/// document frequency.
///
/// Keeps the present (df > 0) tokens rarest-first up to the typo floor `F = m + d`,
/// then admits more until the cumulative [`cost`](SelectParams::cost) reaches the
/// breadth budget `B`, never exceeding `k_max`. Every absent (df = 0) token is then
/// appended uncharged — with a live frequency column a df-0 token has a provably
/// empty posting, so it costs nothing to scan and never reaches the overlap floor
/// (which the ranker derives from the postings actually present).
///
/// `tokens` must be the *distinct* query tokens (deduplicated). The returned
/// selection is in scan order: kept present tokens rarest-first, then absent tokens
/// in token order — deterministic run to run.
pub(crate) fn select<Tk: Clone + Ord>(tokens: &[(Tk, i64)], params: SelectParams) -> Vec<Tk> {
    // Present tokens rarest-first; token as a deterministic tie-break.
    let mut present: Vec<&(Tk, i64)> = tokens.iter().filter(|(_, df)| *df > 0).collect();
    present.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));

    // The typo floor, clamped to the match floor and to what's present; the ceiling
    // can never undercut the floor. Summed in usize so a caller-supplied `min_shared`
    // near `u32::MAX` cannot overflow the add.
    let f = params.min_shared as usize + params.typo_damage as usize;
    let floor = f.max(params.min_shared as usize).min(present.len());
    let ceiling = params.k_max.max(floor);

    let mut kept: Vec<Tk> = Vec::with_capacity(ceiling);
    let mut sum_df: i64 = 0;
    for (tok, df) in present {
        kept.push(tok.clone());
        sum_df += df;
        if kept.len() >= ceiling {
            break; // absolute ceiling
        }
        // Past the floor, stop once the kept lists' scan cost reaches the budget.
        if kept.len() >= floor && params.cost(kept.len(), sum_df) >= params.breadth as f64 {
            break;
        }
    }

    // Append the absent tokens (deterministic token order), charged nothing.
    let mut absent: Vec<Tk> = tokens
        .iter()
        .filter(|(_, df)| *df <= 0)
        .map(|(tok, _)| tok.clone())
        .collect();
    absent.sort();
    kept.extend(absent);
    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(min_shared: u32, breadth: u64) -> SelectParams {
        SelectParams {
            min_shared,
            breadth,
            typo_damage: 4,
            k_max: 12,
            alpha: 0.0,
            beta: 1.0,
        }
    }

    /// Helper: `(name, df)` pairs with single-char string tokens.
    fn toks(pairs: &[(&str, i64)]) -> Vec<(String, i64)> {
        pairs.iter().map(|(t, d)| (t.to_string(), *d)).collect()
    }

    #[test]
    fn default_keeps_exactly_the_typo_floor() {
        // 8 present tokens, m=2 -> F=6; B=0 keeps exactly 6 (rarest first).
        let t = toks(&[
            ("a", 1),
            ("b", 2),
            ("c", 3),
            ("d", 4),
            ("e", 5),
            ("f", 6),
            ("g", 7),
            ("h", 8),
        ]);
        let kept = select(&t, params(2, 0));
        assert_eq!(kept, ["a", "b", "c", "d", "e", "f"]);
    }

    #[test]
    fn extreme_min_shared_does_not_overflow() {
        // A pathological `min_shared` near u32::MAX must not panic on the floor add
        // (debug overflow check); the floor just clamps to what is present.
        let t = toks(&[("a", 1), ("b", 2), ("c", 3)]);
        let kept = select(&t, params(u32::MAX, 0));
        assert_eq!(kept, ["a", "b", "c"]); // clamped to present.len()
    }

    #[test]
    fn cost_budget_admits_breadth_past_the_floor_then_stops() {
        // df sums: after 6 rarest (1..6) Σ=21; budget 30 admits more until Σ>=30.
        let t = toks(&[
            ("a", 1),
            ("b", 2),
            ("c", 3),
            ("d", 4),
            ("e", 5),
            ("f", 6),
            ("g", 7),
            ("h", 8),
        ]);
        // floor 6 (Σ=21<30), +g -> Σ=28<30, +h -> Σ=36>=30 stop. 8 kept.
        let kept = select(&t, params(2, 30));
        assert_eq!(kept, ["a", "b", "c", "d", "e", "f", "g", "h"]);
    }

    #[test]
    fn floor_caps_at_present_count() {
        let t = toks(&[("a", 1), ("b", 2)]); // only 2 present, F would be 6
        let kept = select(&t, params(2, 0));
        assert_eq!(kept, ["a", "b"]);
    }

    #[test]
    fn k_max_caps_breadth_under_a_large_budget() {
        let t: Vec<(String, i64)> = (0..20).map(|i| (format!("{i:02}"), i + 1)).collect();
        let kept = select(
            &t,
            SelectParams {
                k_max: 12,
                ..params(2, u64::MAX)
            },
        );
        assert_eq!(kept.len(), 12);
    }

    #[test]
    fn k_max_never_undercuts_the_typo_floor() {
        // A misconfigured k_max below F must not drop below the floor.
        let t = toks(&[
            ("a", 1),
            ("b", 2),
            ("c", 3),
            ("d", 4),
            ("e", 5),
            ("f", 6),
            ("g", 7),
        ]);
        let kept = select(
            &t,
            SelectParams {
                k_max: 3,
                ..params(2, 0)
            },
        );
        assert_eq!(kept.len(), 6, "floor F=6 wins over k_max=3");
    }

    #[test]
    fn absent_tokens_kept_uncharged_and_after_present() {
        let t = toks(&[("z", 0), ("a", 5), ("y", 0), ("b", 3)]);
        let kept = select(&t, params(2, 0));
        // present rarest-first: b(3), a(5); then absent in token order: y, z.
        assert_eq!(kept, ["b", "a", "y", "z"]);
    }

    #[test]
    fn batch_equals_serial_is_a_pure_function_of_this_querys_dfs() {
        // Identical inputs -> identical output, regardless of any external state.
        let t = toks(&[("a", 1), ("b", 2), ("c", 9)]);
        assert_eq!(select(&t, params(2, 0)), select(&t, params(2, 0)));
    }

    #[test]
    fn budget_boundary_is_inclusive_at_exactly_b() {
        // 7 present, m=2 -> F=6. After the 6 rarest (df 1..6), Σdf = 21.
        let t = toks(&[
            ("a", 1),
            ("b", 2),
            ("c", 3),
            ("d", 4),
            ("e", 5),
            ("f", 6),
            ("g", 7),
        ]);
        // cost(6, 21) = 21 >= B -> stop at the floor. B=21 keeps exactly 6.
        assert_eq!(select(&t, params(2, 21)).len(), 6);
        // B=22 admits one more (cost 21 < 22 at the floor).
        assert_eq!(select(&t, params(2, 22)).len(), 7);
    }

    #[test]
    fn df_ties_break_on_the_token_deterministically() {
        // Equal DF, supplied out of token order; the kept prefix is token-ascending.
        let t = toks(&[("c", 5), ("a", 5), ("b", 5)]);
        let kept = select(
            &t,
            SelectParams {
                typo_damage: 0,
                ..params(3, 0)
            },
        );
        assert_eq!(kept, ["a", "b", "c"]);
    }

    #[test]
    fn non_default_alpha_beta_enters_the_cost() {
        // alpha=1,beta=1, 8 present, m=2 -> F=6. After 6 (df 1..6): cost = 6 + 21 = 27.
        let t = toks(&[
            ("a", 1),
            ("b", 2),
            ("c", 3),
            ("d", 4),
            ("e", 5),
            ("f", 6),
            ("g", 7),
            ("h", 8),
        ]);
        let p = |b: u64| SelectParams {
            alpha: 1.0,
            beta: 1.0,
            ..params(2, b)
        };
        assert_eq!(
            select(&t, p(27)).len(),
            6,
            "cost 27 >= 27 stops at the floor"
        );
        assert_eq!(select(&t, p(28)).len(), 7, "cost 27 < 28 admits one more");
    }

    #[test]
    fn floor_clamps_when_m_exceeds_present_count() {
        // m=5 -> F=9, but only 2 tokens present: keep both, not 9.
        let t = toks(&[("a", 1), ("b", 2)]);
        assert_eq!(select(&t, params(5, 0)).len(), 2);
    }

    #[test]
    fn breadth_grows_the_kept_set_monotonically() {
        let t = toks(&[
            ("a", 1),
            ("b", 2),
            ("c", 3),
            ("d", 4),
            ("e", 5),
            ("f", 6),
            ("g", 7),
            ("h", 8),
        ]);
        let mut prev = 0usize;
        for b in [0u64, 22, 28, 36, u64::MAX] {
            let n = select(&t, params(2, b)).len();
            assert!(n >= prev, "kept count must not shrink as B grows ({b})");
            prev = n;
        }
    }
}
