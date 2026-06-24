//! Typo and partial-match tolerance — the reason trifle exists. Each edit-type
//! test applies a genuine single edit and asserts the target is still found.

mod common;
use common::*;
use trifle::SearchOpts;

/// Index `target` (plus distractors) and report whether `query` finds it.
fn finds(target: &str, query: &str) -> bool {
    let h = Harness::new();
    h.put(1, "field", "f", target);
    h.put(
        2,
        "field",
        "f",
        "completely unrelated content about sailing ships",
    );
    h.put(
        3,
        "field",
        "f",
        "another different sentence regarding mountain trails",
    );
    let hits = h.index.search(query, SearchOpts::new(5)).unwrap();
    hit(&hits, 1)
}

#[test]
fn tolerates_a_substitution() {
    // photosynthesis -> photosynthYsis (e -> y)
    assert!(finds(
        "photosynthesis chlorophyll",
        "photosynthysis chlorophyll"
    ));
}

#[test]
fn tolerates_an_insertion() {
    // chlorophyll -> chlorophylLl (extra l)
    assert!(finds(
        "photosynthesis chlorophyll",
        "photosynthesis chlorophylll"
    ));
}

#[test]
fn tolerates_a_deletion() {
    // parliamentary -> parliamentry (dropped 'a')
    assert!(finds("parliamentary procedure", "parliamentry procedure"));
}

#[test]
fn tolerates_a_transposition() {
    // serotonin -> seROtonin -> seORtonin (transpose 'ro')
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
    h.put(1, "field", "f", "wxy"); // shares only {wxy} -> overlap 1
    h.put(2, "field", "f", "wxyz"); // shares {wxy, xyz} -> overlap 2
    let hits = h.index.search("wxyz", SearchOpts::new(5)).unwrap();
    assert!(!hit(&hits, 1), "one shared trigram is below the m=2 floor");
    assert!(hit(&hits, 2), "two shared trigrams meets the floor");
}

#[test]
fn min_shared_controls_strictness_when_both_trigrams_are_present() {
    let h = Harness::new();
    // Both query trigrams exist corpus-wide (so the floor is not clamped below m),
    // but each doc shares only one of them.
    h.put(1, "field", "f", "wxy"); // shares {wxy}
    h.put(2, "field", "f", "xyz"); // shares {xyz}
    // Query "wxyz" -> {wxy, xyz}; each doc overlaps exactly 1.
    let strict = h
        .index
        .search("wxyz", SearchOpts::new(5).min_shared(2))
        .unwrap();
    let lenient = h
        .index
        .search("wxyz", SearchOpts::new(5).min_shared(1))
        .unwrap();
    assert!(strict.is_empty(), "m=2 needs two shared; each doc has one");
    assert!(hit(&lenient, 1) && hit(&lenient, 2), "m=1 admits both");
}

#[test]
fn breadth_never_loses_a_hit_and_helps_two_edit_targets() {
    let h = Harness::new();
    h.put(1, "field", "f", "neurotransmitter dopamine serotonin");
    h.put(
        2,
        "field",
        "f",
        "unrelated text about gardening tools and soil",
    );
    let q = "neurotransmiter dopamin"; // two typos
    let narrow = h.index.search(q, SearchOpts::new(5)).unwrap();
    let wide = h.index.search(q, SearchOpts::new(5).breadth(1000)).unwrap();
    if hit(&narrow, 1) {
        assert!(hit(&wide, 1), "breadth must not drop a hit narrow found");
    }
    assert!(
        hit(&wide, 1),
        "the two-typo target is findable with breadth on"
    );
}

#[test]
fn ranking_prefers_the_higher_overlap_document() {
    let h = Harness::new();
    h.put(1, "field", "f", "quick brown fox"); // many shared trigrams
    h.put(2, "field", "f", "quick"); // fewer
    let hits = h
        .index
        .search("quick brown fox", SearchOpts::new(5))
        .unwrap();
    assert_eq!(hits[0].doc_id, 1, "the fuller overlap ranks first");
}
