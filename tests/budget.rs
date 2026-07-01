//! v0.4/M4 (derivation §5/§7): the work budget `C` is finally **realized**. The df≤C-bounded
//! unconditional minimum (typo floor + per-class floor) plus the skip-and-continue budget bound the
//! selected `Σdf` — and hence the candidate union the engine walks — by `≈(F+#classes+1)·C`, O(C),
//! even for a short / common query that was O(N) through M3.
//!
//! Through M3 the bug was that the budget *didn't work even when set*: the unconditional typo-floor
//! prefix kept the `F` rarest grams regardless of df, so when a query had fewer than `F` rare grams
//! the floor reached into the `df≈N` common grams and `Σdf` scaled with `N`. M4 makes the floor
//! df≤C-aware, so "set `C` ⟹ work bounded by `C`" is now true — *except* the §7/§12 rescue, which
//! may walk one over-budget posting when **every** present gram is `df > C` (a pure-common query);
//! `Σdf` is then O(N), the recall-floor that beats an empty result (see
//! `rescue_walks_one_over_budget_posting`).

mod common;
use common::*;
use trifle::SearchOpts;

/// A corpus of `n` docs: every doc contains the common word "common" (so its trigrams have `df = n`),
/// and the first `rare_docs` of them additionally contain the rare word "zqxwv" (its trigrams have
/// `df = rare_docs`). The query "zqxwv common" then mixes 3 rare grams with 4 `df≈N` common grams.
fn corpus(n: i64, rare_docs: i64) -> Harness {
    let h = Harness::new();
    {
        let mut w = h.index.writer().unwrap();
        for d in 1..=n {
            if d <= rare_docs {
                w.upsert(d, &[("f", "zqxwv common")]).unwrap();
            } else {
                let body = format!("common filler{d}");
                w.upsert(d, &[("f", body.as_str())]).unwrap();
            }
        }
        w.commit().unwrap();
    }
    h
}

/// `Σdf` over the selected present grams of `query` (the posting-scan cost the budget bounds), read
/// off a candidate stream's `present_terms`.
fn selected_sigma_df(h: &Harness, query: &str, opts: &SearchOpts<'_>) -> u64 {
    let reader = h.index.reader().unwrap();
    let stream = reader.candidates(query, opts).unwrap();
    stream.present_terms().map(|(_, df)| df).sum()
}

#[test]
fn budget_bounds_selected_sigma_df_independent_of_n() {
    let c = 20u64;
    let opts = SearchOpts::new().df_budget(c).min_shared(2);
    let small = corpus(200, 3);
    let large = corpus(400, 3);

    let sdf_small = selected_sigma_df(&small, "zqxwv common", &opts);
    let sdf_large = selected_sigma_df(&large, "zqxwv common", &opts);

    // With C set, only the in-budget rare grams (df=3 each) are selected; the df≈N "common" grams are
    // skipped. `(F+#classes+1)·C` is a loose ceiling — here Σdf is just the 3 rare grams (9).
    assert!(
        sdf_small <= 6 * c,
        "Σdf is bounded by ~C with a budget (got {sdf_small}, ceiling {})",
        6 * c
    );
    assert_eq!(
        sdf_small, sdf_large,
        "Σdf does not scale with N under a fixed budget ({sdf_small} vs {sdf_large})"
    );
}

#[test]
fn without_a_budget_common_grams_make_sigma_df_scale_with_n() {
    // The mechanism, shown by its absence: with the budget EXPLICITLY unbounded the df≈N common
    // grams ride the typo floor into the selection (only 3 rare grams exist, fewer than F=6), so Σdf
    // scales with N — exactly the M3 behavior the budget bounds. v0.4/M6: `None` now DERIVES a
    // budget, so a caller wanting the old unbounded behavior must bind an explicit huge `df_budget`.
    let opts = SearchOpts::new().min_shared(2).df_budget(u64::MAX); // explicitly unbounded
    let small = corpus(200, 3);
    let large = corpus(400, 3);

    let sdf_small = selected_sigma_df(&small, "zqxwv common", &opts);
    let sdf_large = selected_sigma_df(&large, "zqxwv common", &opts);

    assert!(
        sdf_large > sdf_small,
        "without a budget Σdf grows with N ({sdf_small} -> {sdf_large})"
    );
    assert!(
        sdf_large >= 300,
        "the df≈N common grams dominate the unbounded Σdf (got {sdf_large})"
    );
}

#[test]
fn the_candidate_union_is_bounded_under_a_budget() {
    // The selected Σdf bounds the candidate union the engine walks: under a budget the union is the
    // few rare docs, not all N. Count distinct candidate keys (propagating errors).
    let c = 20u64;
    let opts = SearchOpts::new().df_budget(c).min_shared(2);
    let h = corpus(300, 3);
    let reader = h.index.reader().unwrap();
    let stream = reader.candidates("zqxwv common", &opts).unwrap();
    let mut n_cands = 0usize;
    for cand in stream {
        cand.unwrap();
        n_cands += 1;
    }
    assert!(
        n_cands <= 3,
        "the candidate union is bounded by the budget, not O(N) (got {n_cands} of 300)"
    );
}

#[test]
fn rescue_walks_one_over_budget_posting_the_oc_exception() {
    // The documented O(C) carve-out: a PURE-common query (every present gram is df>C) triggers the
    // §7/§12 rescue, which keeps the single rarest gram regardless of the budget — so Σdf is O(N)
    // here, the recall-floor that beats returning empty. (CODE IS CORRECT; this pins the exception.)
    let h = corpus(200, 0); // all docs are common-only; query "common" has only df≈200 grams
    let opts = SearchOpts::new().df_budget(5).min_shared(1);
    let sigma_df = selected_sigma_df(&h, "common", &opts);
    assert!(
        sigma_df > 5,
        "the rescue admits one over-budget posting; Σdf={sigma_df} exceeds the budget C=5"
    );
    // Recall is preserved — the query is not starved to empty.
    let hits = h.search_opts("common", &opts, 5).unwrap();
    assert!(!hits.is_empty(), "the rescue keeps the query non-empty");
}

#[test]
fn batch_equals_serial_with_the_stop_active_and_under_a_budget() {
    // batch == serial must hold WITH the §5 stop active and under a budget — the stop is a pure
    // function of each query's grams + the shared snapshot, never a batch aggregate.
    let h = Harness::new();
    {
        let mut w = h.index.writer().unwrap();
        for d in 1..=300i64 {
            let body = if d <= 5 {
                format!("zqxwvy plingorf wibblethorpe common filler{d}")
            } else {
                format!("common filler{d}")
            };
            w.upsert(d, &[("f", body.as_str())]).unwrap();
        }
        w.commit().unwrap();
    }
    let queries = [
        "zqxwvy plingorf wibblethorpe", // rare, multi-word -> stop fires
        "common",                       // common-only
        "zqxwvy common plingorf",       // mix
        "wibblethorpe filler3",         // rare + a filler
    ];
    for opts in [
        SearchOpts::new(),                             // stop on at default k/c
        SearchOpts::new().df_budget(30).min_shared(2), // tight budget
        SearchOpts::new().k_target(8).c_margin(2.0),   // small k -> larger target -> later stop
    ] {
        let reader = h.index.reader().unwrap();
        let batched = reader.matches_batch(&queries, &opts, 20).unwrap();
        for (i, q) in queries.iter().enumerate() {
            let serial = h.index.reader().unwrap().matches(q, &opts, 20).unwrap();
            assert_eq!(
                ids(&serial),
                ids(&batched[i]),
                "batch == serial must hold with the stop active for {q:?}"
            );
        }
    }
}

#[test]
fn the_budget_still_finds_the_rare_docs() {
    // The bounded selection preserves recall for the in-budget grams: the rare docs are still found
    // (the 3 rare grams meet the m=2 floor); only the df≈N common grams are dropped (their docs
    // weren't relevant to "zqxwv" anyway — the accepted, bounded recall cost of the budget, §5).
    let c = 20u64;
    let opts = SearchOpts::new().df_budget(c).min_shared(2);
    let h = corpus(200, 3);
    let hits = h.search_opts("zqxwv common", &opts, 10).unwrap();
    let found = ids(&hits);
    for d in 1..=3 {
        assert!(
            found.contains(&d),
            "rare doc {d} is found under the budget: {found:?}"
        );
    }
}
