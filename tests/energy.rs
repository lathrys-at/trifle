//! End-to-end guards for the v0.4 logit-idf energy weighting (derivation §2/§4/§7), exercised
//! through the public `matches`/`matches_batch` API. These are the panel's adversarial cases
//! turned into regression tests: `batch == serial`, the no-vanish recall floor (a `df = N` gram and
//! all-zero-weight tiny corpora still retrieve every match via the engine's `≥ 1` clamp), the
//! all-common count-only degradation, end-to-end rarity ranking, and the degenerate-knob fallback /
//! coarse-`Δ` guard reachable from `SearchOpts`.

mod common;
use common::*;
use trifle::SearchOpts;

/// Load `(id, text)` docs under label `"f"` in one writer batch (faster than per-doc commits).
fn load(h: &Harness, docs: &[(i64, &str)]) {
    let mut w = h.index.writer().unwrap();
    for (id, text) in docs {
        w.upsert(*id, &[("f", *text)]).unwrap();
    }
    w.commit().unwrap();
}

#[test]
fn batch_equals_serial_ranking() {
    // The per-query energy weights derive only from this query's tokens + the shared snapshot, so a
    // query ranks identically run alone vs. mid-batch.
    let h = Harness::new();
    load_fixture(&h);
    let q = "quick brown";
    let serial = ids(&h.search(q, 10).unwrap());
    assert!(!serial.is_empty(), "the probe query must hit something");
    let batch = h
        .search_batch(&["lazy dog", q, "five wizards"], 10)
        .unwrap();
    assert_eq!(
        ids(&batch[1]),
        serial,
        "q ranks identically serial vs. mid-batch (batch == serial)"
    );
}

#[test]
fn ubiquitous_gram_does_not_drop_documents() {
    // "alpha" trigrams sit in every segment (df = N) → energy −∞ → weight 0 → the engine's ≥ 1
    // clamp keeps them count-only, so no document vanishes.
    let h = Harness::new();
    load(
        &h,
        &[
            (1, "alpha one"),
            (2, "alpha two"),
            (3, "alpha three"),
            (4, "alpha four"),
            (5, "alpha five"),
        ],
    );
    let hits = h
        .search_opts("alpha", &SearchOpts::new().min_shared(1), 10)
        .unwrap();
    assert_eq!(hits.len(), 5, "a df = N gram still retrieves every segment");
}

#[test]
fn tiny_corpora_retrieve_every_match() {
    // For N = 1..=4 every energy is ≤ 0 (floor near corpus size) → all weights quantize to 0 →
    // engine ≥ 1 clamp → overlap-only ranking. Every matching doc must still come back.
    for n in 1..=4i64 {
        let h = Harness::new();
        let docs: Vec<(i64, &str)> = (1..=n).map(|i| (i, "quick brown fox")).collect();
        load(&h, &docs);
        let hits = h.search("quick brown fox", 10).unwrap();
        assert_eq!(
            hits.len() as i64,
            n,
            "N={n}: all-zero energy weights still retrieve every doc"
        );
    }
}

#[test]
fn all_common_query_degrades_to_count_only_not_empty() {
    // Every query gram is ubiquitous (df = N) → no rarity discrimination → the query degrades to a
    // count-and-length ranking (plane count floored at 1), never empty or a crash.
    let h = Harness::new();
    let docs: Vec<(i64, &str)> = (1..=6)
        .map(|i| (i, "the common shared words here"))
        .collect();
    load(&h, &docs);
    let hits = h.search("common shared words", 10).unwrap();
    assert_eq!(
        hits.len(),
        6,
        "all-common query degrades to count-only, never empty"
    );
}

#[test]
fn rare_gram_outranks_common_only_match() {
    // On a corpus large enough that a rare gram (df = 1, floored) quantizes well above a common gram
    // (df ≈ N → weight 1), the document holding the rare gram must rank first — energy promotes
    // rarity end to end.
    let h = Harness::new();
    let mut docs: Vec<(i64, &str)> = (2..=30).map(|i| (i, "common filler words")).collect();
    docs.push((1, "common qzjwx words"));
    load(&h, &docs);
    let hits = h
        .search_opts("qzjwx common", &SearchOpts::new().min_shared(1), 10)
        .unwrap();
    assert_eq!(
        hits[0].key.as_i64(),
        Some(1),
        "energy promotes the rare-gram match to the top"
    );
}

#[test]
fn degenerate_knobs_fall_back_to_defaults() {
    // ν/κ/Δ are reachable via the public SearchOpts builders; degenerate values (out-of-domain,
    // NaN, +∞) are sanitized to their defaults — no panic from the debug guards, and identical
    // ranking to the default search.
    let h = Harness::new();
    load_fixture(&h);
    let q = "quick brown";
    let baseline = ids(&h.search(q, 10).unwrap());
    assert!(!baseline.is_empty(), "the probe query must hit something");
    let degenerate = [
        SearchOpts::new().nu(0.0),
        SearchOpts::new().nu(-1.0),
        SearchOpts::new().nu(f64::NAN),
        SearchOpts::new().kappa(f64::NAN),
        SearchOpts::new().delta(f64::INFINITY),
    ];
    for opts in &degenerate {
        let hits = h.search_opts(q, opts, 10).unwrap();
        assert_eq!(
            ids(&hits),
            baseline,
            "a degenerate knob falls back to the default (same ranking)"
        );
    }
}

#[cfg(debug_assertions)]
#[test]
fn coarse_delta_trips_the_quantization_guard_in_debug() {
    // The §7 Δ < 2·E_floored guard is reachable from the public API: a deliberately coarse Δ on a
    // corpus large enough for the floor regime trips the debug_assert (debug builds only — it is
    // compiled out of release). Δ = 100 is a finite, positive value, so sanitization keeps it.
    let h = Harness::new();
    let docs: Vec<(i64, &str)> = (1..=40)
        .map(|i| (i, "assorted filler words and padding here"))
        .collect();
    load(&h, &docs);
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = h.search_opts("assorted filler", &SearchOpts::new().delta(100.0), 10);
    }));
    assert!(res.is_err(), "coarse Δ = 100 trips the §7 guard at N = 40");
}
