//! v0.4/M6 (C1, Open decision B): the **recall-guard** — the HARD GATE on the derived work budget.
//!
//! `SearchOpts::df_budget = None` now *derives* the budget `C = (1/σ)·ln(N/k)·d̄/ln(N/d̄)` with
//! `d̄ = exp(mean_lndf + Z·std_lndf)`, `Z = 2` (`search::derived_budget`). Deriving `C` must not
//! **regress recall** versus an unbounded budget: for a ladder of `N` and a query suite over planted
//! markers, `recall(derived) ≥ recall(unbounded)`. The recall-floor is structural — a query whose
//! discriminating grams are all pruned by a tight `C` falls to the §7/§12 single-gram rescue (whose
//! floor clamps to 1), so the relevant docs are still retrieved — which is why `Z = 2` holds here.
//!
//! If a future change to `Z` (or the guards) drops a relevant doc, this test fails; raise `Z` until
//! it does not.

mod common;
use common::*;

use std::collections::BTreeSet;
use trifle::SearchOpts;

const RARE_R: i64 = 10; // docs 1..=RARE_R carry the rare marker
const MID_STEP: i64 = 50; // every MID_STEP-th doc carries the mid-frequency marker
const VOCAB: usize = 60; // distinct mid-frequency Latin "filler" words

/// A distinct 3-letter Latin word (one trigram) for vocabulary index `i` (`i < 26·26`). Ends in a
/// fixed letter so every entry is exactly three codepoints ⇒ a single Latin **trigram** class gram.
fn vocab_word(i: usize) -> String {
    let a = b'a';
    let c0 = (a + (i / 26 % 26) as u8) as char;
    let c1 = (a + (i % 26) as u8) as char;
    format!("{c0}{c1}x")
}

/// A per-doc **unique** 6-letter Latin word (the low-df tail that pulls the Latin-trigram class's
/// `d̄` down into a bounded, biting range, well below `N`). Base-26 of `d` over six letters.
fn unique_word(d: i64) -> String {
    let d = d as usize;
    let a = b'a';
    let l = |p: usize| (a + (d / p % 26) as u8) as char;
    format!("q{}{}{}{}{}", l(1), l(26), l(676), l(17_576), l(456_976))
}

/// A representative small-doc corpus of `n` docs. Each doc mixes a few high-df boilerplate words
/// with three mid-frequency `VOCAB` words (so the **Latin-trigram** class has a realistic Zipfian
/// df spread, `d̄ ∈ (marker df, boilerplate df)` ⇒ a bounded, *biting* derived `C`), plus two planted
/// distinctive markers at different rarities:
///  - RARE  "zynthquor plarnexis" in docs `1..=RARE_R`               (df = RARE_R)
///  - MID   "brimwexil dovcortan" in docs with `d % MID_STEP == 0`   (df ≈ n/MID_STEP)
fn corpus(n: i64) -> Harness {
    let h = Harness::new();
    {
        let mut w = h.index.writer().unwrap();
        for d in 1..=n {
            let du = d as usize;
            let mut body = format!(
                "record entry document field {} {} {} {}",
                unique_word(d),
                vocab_word(du % VOCAB),
                vocab_word((du * 7) % VOCAB),
                vocab_word((du * 13) % VOCAB),
            );
            if d <= RARE_R {
                body.push_str(" zynthquor plarnexis");
            }
            if d % MID_STEP == 0 {
                body.push_str(" brimwexil dovcortan");
            }
            w.upsert(d, &[("body", body.as_str())]).unwrap();
        }
        w.commit().unwrap();
    }
    h
}

/// Recall of `query` under `opts`: fraction of the known-`relevant` docs retrieved in the top-`limit`.
fn recall(
    h: &Harness,
    query: &str,
    opts: &SearchOpts<'_>,
    relevant: &BTreeSet<i64>,
    limit: usize,
) -> f64 {
    let hits = h.search_opts(query, opts, limit).unwrap();
    let got: BTreeSet<i64> = ids(&hits).into_iter().collect();
    let found = relevant.iter().filter(|d| got.contains(d)).count();
    found as f64 / relevant.len() as f64
}

/// `Σdf` over the selected present grams of `query` (the posting-scan cost the budget bounds).
fn selected_sigma_df(h: &Harness, query: &str, opts: &SearchOpts<'_>) -> u64 {
    let reader = h.index.reader().unwrap();
    reader
        .candidates(query, opts)
        .unwrap()
        .present_terms()
        .map(|(_, df)| df)
        .sum()
}

#[test]
fn derived_budget_does_not_regress_recall_vs_unbounded() {
    // The N ladder is > k = 128 so the derived budget is actually active (below k it is None).
    for &n in &[1_000i64, 5_000] {
        let h = corpus(n);
        let derived = SearchOpts::new(); // df_budget = None ⇒ corpus-derived C (the default)
        let unbounded = SearchOpts::new().df_budget(u64::MAX); // explicit unbounded baseline

        // The derived budget must be genuinely ACTIVE here (not silently unbounded), else the recall
        // comparisons below are vacuous: on an all-common query the df≈N boilerplate grams blow any
        // reasonable C, so the derived selection scans strictly less Σdf than the unbounded one.
        assert!(
            selected_sigma_df(&h, "record entry document field", &derived)
                < selected_sigma_df(&h, "record entry document field", &unbounded),
            "N={n}: the derived budget must actively prune common grams (non-vacuous gate)"
        );

        let rare: BTreeSet<i64> = (1..=RARE_R).collect();
        let mid: BTreeSet<i64> = (1..=n).filter(|d| d % MID_STEP == 0).collect();
        let limit = (mid.len().max(rare.len()) + 50).max(64);

        for (q, rel) in [
            ("zynthquor plarnexis", &rare),
            ("brimwexil dovcortan", &mid),
        ] {
            let rd = recall(&h, q, &derived, rel, limit);
            let ru = recall(&h, q, &unbounded, rel, limit);
            assert!(
                ru > 0.0,
                "N={n} q={q:?}: the unbounded baseline must find the relevant docs (test premise)"
            );
            // No regression — and, in fact, the derived default keeps FULL recall on these
            // distinctive-marker queries (the pruned marker grams fall to the §7/§12 rescue).
            assert!(
                rd + 1e-9 >= ru,
                "N={n} q={q:?}: derived-C recall {rd} regressed below unbounded {ru} — raise Z"
            );
            assert!(
                (rd - 1.0).abs() < 1e-9,
                "N={n} q={q:?}: derived-C recall {rd} < 1.0 (a relevant doc was dropped — raise Z)"
            );
        }
    }
}
