//! Selection — the rarest-first token pruner (query-time).
//!
//! A query's full token set is the wrong thing to scan: common tokens have
//! O(corpus) postings and little discrimination. Selection keeps a rarest-first
//! prefix. Rarest-first is the whole point — a low-frequency token is both
//! cheapest to scan and most discriminating, perfectly correlated.
//!
//! The prefix runs from the typo floor `F = m + d` up to `t_max` tokens. Its scan
//! cost is the sum of the kept tokens' document frequencies (`Σdf`) — the number of
//! posting rows read — so `t_max` is the cap on rows scanned, the latency proxy.
//!
//! Selection derives only from this query's own token document frequencies (and the
//! per-class statistics snapshot it is handed) — never a batch aggregate or the corpus
//! size — so `search_batch([…, q, …])` ranks `q` identically to `search(q)`
//! (batch == serial).
//!
//! "Rarest" is **class-normalized**: tokens are ranked by [`ClassSnap::rarity`] (a
//! `z`-score within the token's script class, falling back to raw DF for a degenerate
//! class), so a CJK bigram and a Latin trigram — different DF regimes — compare on the
//! same footing. Within a single script (the common per-query case) this reduces to the
//! old rarest-by-DF order.

use crate::welford::ClassSnap;

/// The behavior + performance knobs the pruner reads (from [`SearchOpts`](crate::SearchOpts)).
#[derive(Clone, Copy, Debug)]
pub(crate) struct SelectParams {
    /// `m` — the match floor (the strictness dial).
    pub min_shared: u32,
    /// `d` — per-typo token damage; the typo floor is `F = m + d`.
    pub typo_damage: u32,
    /// `t_max` — the number of rarest tokens to keep (never fewer than the typo floor).
    pub t_max: usize,
    /// `B` — an optional cap on the cumulative document frequency (`Σdf`) of the kept present
    /// tokens. The scan cost is `Σdf`, so this caps *work* directly (rather than count, as
    /// `t_max` does). The typo floor is always kept even if it exceeds `B`; beyond the floor,
    /// rarest-first tokens are kept while `Σdf` stays within budget. `None` = no cap.
    pub df_budget: Option<u64>,
}

/// Select the tokens to scan, from each distinct query token paired with its live
/// document frequency.
///
/// Keeps the present (df > 0) tokens rarest-first, from the typo floor `F = m + d`
/// up to `t_max` (never fewer than `F`, never more than what's present). Every absent
/// (df = 0) token is then appended uncharged — with a live frequency column a df-0
/// token has a provably empty posting, so it costs nothing to scan and never reaches
/// the overlap floor (which the ranker derives from the postings actually present).
///
/// `tokens` are the *distinct* query tokens (deduplicated) as `(token, df, class)`
/// triples — `class` is the token's script-tag byte. `classes` is the per-class stats
/// snapshot used to rank rarity. The returned selection is in scan order: kept present
/// tokens rarest-first (by class-normalized rarity, token tie-break), then absent tokens
/// in token order — deterministic run to run.
pub(crate) fn select<Tk: Clone + Ord>(
    tokens: &[(Tk, i64, u8)],
    params: SelectParams,
    classes: &ClassSnap,
) -> Vec<Tk> {
    // Present tokens rarest-first by class-normalized rarity; token as a deterministic
    // tie-break (and a stable secondary when two rarities compare equal).
    let mut present: Vec<&(Tk, i64, u8)> = tokens.iter().filter(|(_, df, _)| *df > 0).collect();
    present.sort_by(|a, b| {
        classes
            .rarity(a.1, a.2)
            .partial_cmp(&classes.rarity(b.1, b.2))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });

    // The typo floor, clamped to the match floor and to what's present; `t_max` can
    // never undercut it. Summed in usize so a caller-supplied `min_shared` near
    // `u32::MAX` cannot overflow the add.
    let f = params.min_shared as usize + params.typo_damage as usize;
    let floor = f.max(params.min_shared as usize).min(present.len());
    let keep = params.t_max.max(floor).min(present.len());

    // Keep the rarest-first prefix up to `keep` (the count cap), but once past the floor stop
    // as soon as the cumulative df (`Σdf` — the scan cost) would exceed the `df_budget`. The
    // floor is kept unconditionally so a typo'd query never loses its tolerance to the cap.
    let mut kept: Vec<Tk> = Vec::with_capacity(keep);
    let mut cum_df: u64 = 0;
    for (i, (tok, df, _)) in present.iter().enumerate() {
        if i >= keep {
            break;
        }
        let df = (*df).max(0) as u64;
        if i >= floor {
            if let Some(budget) = params.df_budget {
                if cum_df + df > budget {
                    break;
                }
            }
        }
        kept.push(tok.clone());
        cum_df += df;
    }

    // Append the absent tokens (deterministic token order), charged nothing.
    let mut absent: Vec<Tk> = tokens
        .iter()
        .filter(|(_, df, _)| *df <= 0)
        .map(|(tok, _, _)| tok.clone())
        .collect();
    absent.sort();
    kept.extend(absent);
    kept
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::welford::ClassSnap;

    fn params(min_shared: u32, t_max: usize) -> SelectParams {
        SelectParams {
            min_shared,
            typo_damage: 4,
            t_max,
            df_budget: None,
        }
    }

    /// Helper: `(name, df)` pairs as `(token, df, class)` triples, all class 0. With an
    /// empty [`ClassSnap`] (below) the pruner ranks by raw DF — these exercise that path.
    fn toks(pairs: &[(&str, i64)]) -> Vec<(String, i64, u8)> {
        pairs
            .iter()
            .map(|(t, d)| (t.to_string(), *d, 0u8))
            .collect()
    }

    /// The raw-DF (empty) class snapshot these tests rank against.
    fn snap() -> ClassSnap {
        ClassSnap::empty()
    }

    #[test]
    fn keeps_the_t_max_rarest_present_tokens() {
        // 8 present, m=2 -> F=6. t_max=6 keeps the 6 rarest (rarest-first).
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
        assert_eq!(
            select(&t, params(2, 6), &snap()),
            ["a", "b", "c", "d", "e", "f"]
        );
    }

    #[test]
    fn larger_t_max_admits_more_rarest_first() {
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
        // t_max=8 keeps all eight, still rarest-first.
        assert_eq!(
            select(&t, params(2, 8), &snap()),
            ["a", "b", "c", "d", "e", "f", "g", "h"]
        );
    }

    #[test]
    fn t_max_below_the_typo_floor_keeps_the_floor() {
        // A misconfigured t_max below F=6 must not drop below the floor.
        let t = toks(&[
            ("a", 1),
            ("b", 2),
            ("c", 3),
            ("d", 4),
            ("e", 5),
            ("f", 6),
            ("g", 7),
        ]);
        assert_eq!(
            select(&t, params(2, 3), &snap()).len(),
            6,
            "floor F=6 wins over t_max=3"
        );
    }

    #[test]
    fn t_max_caps_under_many_present_tokens() {
        let t: Vec<(String, i64, u8)> = (0..20).map(|i| (format!("{i:02}"), i + 1, 0u8)).collect();
        assert_eq!(select(&t, params(2, 12), &snap()).len(), 12);
    }

    #[test]
    fn floor_caps_at_present_count() {
        let t = toks(&[("a", 1), ("b", 2)]); // only 2 present, F would be 6
        assert_eq!(select(&t, params(2, 12), &snap()), ["a", "b"]);
    }

    #[test]
    fn floor_clamps_when_m_exceeds_present_count() {
        // m=5 -> F=9, but only 2 tokens present: keep both, not 9.
        let t = toks(&[("a", 1), ("b", 2)]);
        assert_eq!(select(&t, params(5, 12), &snap()).len(), 2);
    }

    #[test]
    fn extreme_min_shared_does_not_overflow() {
        // A pathological `min_shared` near u32::MAX must not panic on the floor add
        // (debug overflow check); the floor just clamps to what is present.
        let t = toks(&[("a", 1), ("b", 2), ("c", 3)]);
        let kept = select(&t, params(u32::MAX, 12), &snap());
        assert_eq!(kept, ["a", "b", "c"]); // clamped to present.len()
    }

    #[test]
    fn df_budget_caps_sigma_df_but_never_drops_below_the_floor() {
        // Rarest-first dfs 1,2,3,4,5,6,7,8; m=2 → floor F=6. With typo_damage 0 the floor is m=2.
        let t = toks(&[("a", 1), ("b", 2), ("c", 3), ("d", 4), ("e", 5), ("f", 6)]);
        let p = |budget: Option<u64>| SelectParams {
            min_shared: 2,
            typo_damage: 0,
            t_max: 12,
            df_budget: budget,
        };
        // No budget: all six present tokens kept (Σdf = 21).
        assert_eq!(select(&t, p(None), &snap()).len(), 6);
        // Budget 6 keeps the floor (a,b = Σdf 3) then c (Σdf 6); d would push Σdf to 10 > 6.
        assert_eq!(select(&t, p(Some(6)), &snap()), ["a", "b", "c"]);
        // A budget below the floor's own Σdf still keeps the whole floor (typo tolerance wins).
        let floor3 = select(
            &t,
            SelectParams {
                min_shared: 3,
                typo_damage: 0,
                t_max: 12,
                df_budget: Some(1),
            },
            &snap(),
        );
        assert_eq!(
            floor3,
            ["a", "b", "c"],
            "the floor (m=3) is kept despite the tiny budget"
        );
    }

    #[test]
    fn absent_tokens_kept_uncharged_and_after_present() {
        let t = toks(&[("z", 0), ("a", 5), ("y", 0), ("b", 3)]);
        let kept = select(&t, params(2, 12), &snap());
        // present rarest-first: b(3), a(5); then absent in token order: y, z.
        assert_eq!(kept, ["b", "a", "y", "z"]);
    }

    #[test]
    fn batch_equals_serial_is_a_pure_function_of_this_querys_dfs() {
        // Identical inputs -> identical output, regardless of any external state.
        let t = toks(&[("a", 1), ("b", 2), ("c", 9)]);
        assert_eq!(
            select(&t, params(2, 12), &snap()),
            select(&t, params(2, 12), &snap())
        );
    }

    #[test]
    fn df_ties_break_on_the_token_deterministically() {
        // Equal DF, supplied out of token order; the kept prefix is token-ascending.
        let t = toks(&[("c", 5), ("a", 5), ("b", 5)]);
        let kept = select(
            &t,
            SelectParams {
                typo_damage: 0,
                ..params(3, 12)
            },
            &snap(),
        );
        assert_eq!(kept, ["a", "b", "c"]);
    }

    #[test]
    fn t_max_grows_the_kept_set_monotonically() {
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
        for tm in [0usize, 6, 7, 8, 100] {
            let n = select(&t, params(2, tm), &snap()).len();
            assert!(
                n >= prev,
                "kept count must not shrink as t_max grows ({tm})"
            );
            prev = n;
        }
    }

    #[test]
    fn class_normalized_rarity_reorders_across_classes() {
        use crate::welford::ClassStats;
        // Two classes with different DF regimes: class 1 ("dense", high DFs) and class 2
        // ("sparse", low DFs). Populate both past the normalize floor.
        let mut stats = ClassStats::new();
        for df in 50..=200 {
            stats.add_sample(1, df);
        }
        for df in 1..=60 {
            stats.add_sample(2, df);
        }
        let cs = stats.snapshot_for([1u8, 2u8]);
        // A df-40 token in the dense class is rarer-for-its-kind than a df-40 token in
        // the sparse class, so it sorts first even though raw DF ties.
        let t = vec![
            ("sparse40".to_string(), 40i64, 2u8),
            ("dense40".to_string(), 40i64, 1u8),
        ];
        let kept = select(&t, params(1, 2), &cs);
        assert_eq!(kept, ["dense40", "sparse40"]);
    }
}
