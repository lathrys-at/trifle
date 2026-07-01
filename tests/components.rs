//! v0.4/M6 (C4 / derivation §10): the `Candidate` score components —
//! [`energy`](trifle::Candidate::energy) / [`count`](trifle::Candidate::count) /
//! [`length`](trifle::Candidate::length) / [`nat_score`](trifle::Candidate::nat_score).
//!
//! The components are the fusion-ready, cross-query-comparable magnitude a downstream consumer
//! (e.g. ruffle) reads. For a single-view (clean, not-starved) query the `nat_score` is the same
//! float as `corrected_score`; for a fused query it stays nat-scale while `corrected_score` becomes
//! the RRF rank key (that no-cross-view-sum property is unit-tested in `search::fusion_tests`).

mod common;
use common::*;

use trifle::SearchOpts;

/// A clean single-view corpus: `n` docs, the first `relevant` of them carrying a distinctive
/// two-word phrase (mid-frequency trigrams, non-floored), the rest filler.
fn corpus(n: i64, relevant: i64) -> Harness {
    let h = Harness::new();
    {
        let mut w = h.index.writer().unwrap();
        for d in 1..=n {
            if d <= relevant {
                w.upsert(d, &[("body", "vexcorp widgetron assembly")])
                    .unwrap();
            } else {
                let body = format!("common ordinary filler text {d}");
                w.upsert(d, &[("body", body.as_str())]).unwrap();
            }
        }
        w.commit().unwrap();
    }
    h
}

#[test]
fn single_view_components_are_consistent_and_nat_scale() {
    // N = 30 (< k = 128 ⇒ the derived budget is unbounded here) and 10 relevant docs. A SINGLE-word
    // query ("vexcorp", 5 trigrams) keeps all its grams (F = m + d = 5 ≥ 5), so the stop prunes none
    // of the primary energy and the query stays a clean SINGLE rank-view — never the starved bigram
    // fallback (a two-word query here would trip `collected_energy_far_below` and RRF-fuse). Its
    // grams have df = 10 > df_min = √30 ≈ 5.5, so they are NON-floored: positive energy + credit.
    let h = corpus(30, 10);
    let reader = h.index.reader().unwrap();
    let stream = reader
        .candidates("vexcorp", &SearchOpts::new().min_shared(1))
        .unwrap();
    let cands: Vec<_> = stream.map(|c| c.unwrap()).collect();
    assert!(!cands.is_empty(), "the clean query returns candidates");

    const DELTA: f64 = 0.5; // DEFAULT_DELTA
    for c in &cands {
        // The accessor identity: nat_score is exactly energy + count − length.
        assert!(
            (c.nat_score() - (c.energy() + c.count() - c.length())).abs() < 1e-9,
            "nat_score == energy + count − length ({} vs {}+{}-{})",
            c.nat_score(),
            c.energy(),
            c.count(),
            c.length()
        );
        // Single-view: nat_score == corrected_score (both are the one §6/§7 float from score_union).
        assert!(
            (c.nat_score() - c.corrected_score()).abs() < 1e-9,
            "single-view nat_score == corrected_score ({} vs {})",
            c.nat_score(),
            c.corrected_score()
        );
        // Hand fixture: the energy component is exactly the integer bit-sliced energy E_acc times Δ.
        assert!(
            (c.energy() - c.score() as f64 * DELTA).abs() < 1e-9,
            "energy() == score()·Δ ({} vs {}·{DELTA})",
            c.energy(),
            c.score()
        );
        // Components are individually well-formed (energy/count/length all ≥ 0 here).
        assert!(c.energy() >= 0.0 && c.count() >= 0.0 && c.length() >= 0.0);
    }

    // The components are actually populated on the clean path: a non-floored match carries positive
    // energy AND a positive count credit (μ on a non-floored gram), on a nat scale.
    assert!(
        cands.iter().any(|c| c.energy() > 0.0),
        "a non-floored rare-gram candidate has positive nat-scale energy"
    );
    assert!(
        cands.iter().any(|c| c.count() > 0.0),
        "a non-floored match earns a positive count credit"
    );
}
