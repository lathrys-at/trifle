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
//! Selection derives only from this query's own token document frequencies — never
//! a batch aggregate or the corpus size — so `search_batch([…, q, …])` ranks `q`
//! identically to `search(q)` (batch == serial).

/// The behavior + performance knobs the pruner reads (from [`SearchOpts`](crate::SearchOpts)).
#[derive(Clone, Copy, Debug)]
pub(crate) struct SelectParams {
    /// `m` — the match floor (the strictness dial).
    pub min_shared: u32,
    /// `d` — per-typo token damage; the typo floor is `F = m + d`.
    pub typo_damage: u32,
    /// `t_max` — the number of rarest tokens to keep (never fewer than the typo floor).
    pub t_max: usize,
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
/// `tokens` must be the *distinct* query tokens (deduplicated). The returned
/// selection is in scan order: kept present tokens rarest-first, then absent tokens
/// in token order — deterministic run to run.
pub(crate) fn select<Tk: Clone + Ord>(tokens: &[(Tk, i64)], params: SelectParams) -> Vec<Tk> {
    // Present tokens rarest-first; token as a deterministic tie-break.
    let mut present: Vec<&(Tk, i64)> = tokens.iter().filter(|(_, df)| *df > 0).collect();
    present.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));

    // The typo floor, clamped to the match floor and to what's present; `t_max` can
    // never undercut it. Summed in usize so a caller-supplied `min_shared` near
    // `u32::MAX` cannot overflow the add.
    let f = params.min_shared as usize + params.typo_damage as usize;
    let floor = f.max(params.min_shared as usize).min(present.len());
    let keep = params.t_max.max(floor).min(present.len());

    let mut kept: Vec<Tk> = present
        .into_iter()
        .take(keep)
        .map(|(tok, _)| tok.clone())
        .collect();

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

    fn params(min_shared: u32, t_max: usize) -> SelectParams {
        SelectParams {
            min_shared,
            typo_damage: 4,
            t_max,
        }
    }

    /// Helper: `(name, df)` pairs with single-char string tokens.
    fn toks(pairs: &[(&str, i64)]) -> Vec<(String, i64)> {
        pairs.iter().map(|(t, d)| (t.to_string(), *d)).collect()
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
        assert_eq!(select(&t, params(2, 6)), ["a", "b", "c", "d", "e", "f"]);
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
            select(&t, params(2, 8)),
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
            select(&t, params(2, 3)).len(),
            6,
            "floor F=6 wins over t_max=3"
        );
    }

    #[test]
    fn t_max_caps_under_many_present_tokens() {
        let t: Vec<(String, i64)> = (0..20).map(|i| (format!("{i:02}"), i + 1)).collect();
        assert_eq!(select(&t, params(2, 12)).len(), 12);
    }

    #[test]
    fn floor_caps_at_present_count() {
        let t = toks(&[("a", 1), ("b", 2)]); // only 2 present, F would be 6
        assert_eq!(select(&t, params(2, 12)), ["a", "b"]);
    }

    #[test]
    fn floor_clamps_when_m_exceeds_present_count() {
        // m=5 -> F=9, but only 2 tokens present: keep both, not 9.
        let t = toks(&[("a", 1), ("b", 2)]);
        assert_eq!(select(&t, params(5, 12)).len(), 2);
    }

    #[test]
    fn extreme_min_shared_does_not_overflow() {
        // A pathological `min_shared` near u32::MAX must not panic on the floor add
        // (debug overflow check); the floor just clamps to what is present.
        let t = toks(&[("a", 1), ("b", 2), ("c", 3)]);
        let kept = select(&t, params(u32::MAX, 12));
        assert_eq!(kept, ["a", "b", "c"]); // clamped to present.len()
    }

    #[test]
    fn absent_tokens_kept_uncharged_and_after_present() {
        let t = toks(&[("z", 0), ("a", 5), ("y", 0), ("b", 3)]);
        let kept = select(&t, params(2, 12));
        // present rarest-first: b(3), a(5); then absent in token order: y, z.
        assert_eq!(kept, ["b", "a", "y", "z"]);
    }

    #[test]
    fn batch_equals_serial_is_a_pure_function_of_this_querys_dfs() {
        // Identical inputs -> identical output, regardless of any external state.
        let t = toks(&[("a", 1), ("b", 2), ("c", 9)]);
        assert_eq!(select(&t, params(2, 12)), select(&t, params(2, 12)));
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
            let n = select(&t, params(2, tm)).len();
            assert!(
                n >= prev,
                "kept count must not shrink as t_max grows ({tm})"
            );
            prev = n;
        }
    }
}
