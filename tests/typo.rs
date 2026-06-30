//! Typo and partial-match tolerance — the reason trifle exists. Each edit-type test applies a
//! genuine single edit and asserts the target is still found.

mod common;
use common::*;
use trifle::SearchOpts;

/// Index `target` (plus distractors) and report whether `query` finds it.
fn finds(target: &str, query: &str) -> bool {
    let h = Harness::new();
    h.put(1, "f", target);
    h.put(2, "f", "completely unrelated content about sailing ships");
    h.put(
        3,
        "f",
        "another different sentence regarding mountain trails",
    );
    let hits = h.search(query, 5).unwrap();
    hit(&hits, 1)
}

#[test]
fn tolerates_a_substitution() {
    assert!(finds(
        "photosynthesis chlorophyll",
        "photosynthysis chlorophyll"
    ));
}

#[test]
fn tolerates_an_insertion() {
    assert!(finds(
        "photosynthesis chlorophyll",
        "photosynthesis chlorophylll"
    ));
}

#[test]
fn tolerates_a_deletion() {
    assert!(finds("parliamentary procedure", "parliamentry procedure"));
}

#[test]
fn tolerates_a_transposition() {
    assert!(finds("dopamine serotonin", "dopamine seortonin"));
}

#[test]
fn partial_query_matches_a_longer_document() {
    assert!(finds(
        "the international space station orbits the earth",
        "international space"
    ));
}

#[test]
fn noise_floor_rejects_below_min_shared() {
    let h = Harness::new();
    // Query "wxyz" has trigrams {wxy, xyz}.
    h.put(1, "f", "wxy"); // shares only {wxy} -> overlap 1
    h.put(2, "f", "wxyz"); // shares {wxy, xyz} -> overlap 2
    let hits = h.search("wxyz", 5).unwrap();
    assert!(!hit(&hits, 1), "one shared trigram is below the m=2 floor");
    assert!(hit(&hits, 2), "two shared trigrams meets the floor");
}

#[test]
fn min_shared_controls_strictness_when_both_trigrams_are_present() {
    let h = Harness::new();
    h.put(1, "f", "wxy"); // shares {wxy}
    h.put(2, "f", "xyz"); // shares {xyz}
    let strict = h
        .search_opts("wxyz", &SearchOpts::new().min_shared(2), 5)
        .unwrap();
    let lenient = h
        .search_opts("wxyz", &SearchOpts::new().min_shared(1), 5)
        .unwrap();
    assert!(strict.is_empty(), "m=2 needs two shared; each doc has one");
    assert!(hit(&lenient, 1) && hit(&lenient, 2), "m=1 admits both");
}

#[test]
fn wider_t_max_never_loses_a_narrow_hit() {
    let h = Harness::new();
    load_fixture(&h);
    let q = "quick brown";
    let narrow = h.search_opts(q, &SearchOpts::new().t_max(6), 10).unwrap();
    let wide = h.search_opts(q, &SearchOpts::new().t_max(12), 10).unwrap();
    assert!(
        !narrow.is_empty(),
        "narrow must hit something for this to be meaningful"
    );
    let wide_ids = ids(&wide);
    for d in ids(&narrow) {
        assert!(
            wide_ids.contains(&d),
            "wider t_max dropped doc {d} that narrow found"
        );
    }
}

#[test]
fn two_char_words_match_via_the_bigram_fallback() {
    // v0.4/M5 (derivation §8): a doc whose every word is exactly 2 chars now indexes BIGRAMS (the
    // secondary order — a 2-char Latin run is too short for a trigram), and a query of the same short
    // words reaches the bigram (secondary) rank-view via the structural fallback, so it now MATCHES.
    // v0.3/M4 (trigram-only) indexed nothing and returned empty here — this is a deliberate,
    // user-visible recall improvement. Index and query still agree (both emit the same dual-order
    // grams), so the state stays consistent.
    let h = Harness::new();
    h.put(1, "f", "ab cd ef");
    assert!(
        hit(&h.search("ab cd ef", 10).unwrap(), 1),
        "all-short-word doc now matches via the bigram structural fallback (§8)"
    );
    // A normal doc in the same index is unaffected.
    h.put(2, "f", "quick brown");
    assert!(hit(&h.search("quick brown", 10).unwrap(), 2));
}

#[test]
fn ranking_prefers_the_higher_overlap_document() {
    let h = Harness::new();
    h.put(1, "f", "quick brown fox"); // many shared trigrams
    h.put(2, "f", "quick"); // fewer
    let hits = h.search("quick brown fox", 5).unwrap();
    assert_eq!(
        hits[0].key.as_i64(),
        Some(1),
        "the fuller overlap ranks first"
    );
}
