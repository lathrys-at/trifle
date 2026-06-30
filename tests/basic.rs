//! Round-trip behaviors: provenance, text, spans, multi-segment docs, limits, dedup.

mod common;
use common::*;
use trifle::SearchOpts;

#[test]
fn exact_query_finds_the_document() {
    let h = Harness::new();
    load_fixture(&h);
    let hits = h.search("quick brown fox", 10).unwrap();
    assert!(hit(&hits, 1), "doc 1 contains the phrase");
}

#[test]
fn whitespace_only_and_empty_queries_are_graceful() {
    // v0.4/M4: whitespace breaks the gram window, so an all-whitespace or empty query produces no
    // grams — it must return no matches (and never panic), while real queries still work.
    let h = Harness::new();
    load_fixture(&h);
    for q in ["", " ", "   \t\n", "\u{00A0}"] {
        assert!(
            h.search(q, 10).unwrap().is_empty(),
            "empty/whitespace query {q:?} yields no matches, no panic"
        );
    }
    assert!(
        !h.search("quick", 10).unwrap().is_empty(),
        "a real query still works"
    );
}

#[test]
fn match_carries_provenance_and_text() {
    let h = Harness::new();
    h.put(42, "page-3.png", "the treaty was signed in vienna");
    let hits = h.search("treaty signed", 10).unwrap();
    let m = hits
        .iter()
        .find(|m| m.key.as_i64() == Some(42))
        .expect("found");
    assert_eq!(m.label, "page-3.png");
    assert_eq!(m.text, "the treaty was signed in vienna");
}

#[test]
fn span_indexes_the_matched_region_of_the_text() {
    let h = Harness::new();
    h.put(1, "f", "alpha bravo charlie delta");
    let hits = h.search("charlie", 5).unwrap();
    let m = &hits[0];
    let (lo, hi) = m.span.expect("a span for a clean ascii match");
    let text = m.text.as_str();
    assert!(lo < hi && hi <= text.len());
    let region = &text[lo..hi];
    assert!(region.contains("charlie"), "span region was {region:?}");
}

#[test]
fn one_match_per_doc_even_with_many_segments() {
    let h = Harness::new();
    // Two segments under one key: both mention "quartz".
    h.upsert(
        7,
        &[
            ("front", "the quartz crystal"),
            ("back", "quartz is a mineral"),
        ],
    );
    let hits = h.search("quartz crystal", 10).unwrap();
    assert_eq!(
        hits.iter().filter(|m| m.key.as_i64() == Some(7)).count(),
        1,
        "a doc appears at most once (deduped by key)"
    );
}

#[test]
fn distinct_segments_are_searchable_and_labeled() {
    let h = Harness::new();
    h.upsert(
        1,
        &[
            ("front", "mitochondria powerhouse cell"),
            ("scan", "ribosome protein synthesis"),
        ],
    );
    let a = h.search("mitochondria", 5).unwrap();
    let b = h.search("ribosome", 5).unwrap();
    assert_eq!(a[0].label, "front");
    assert_eq!(b[0].label, "scan");
}

#[test]
fn upsert_replaces_a_segment_in_place() {
    let h = Harness::new();
    h.put(1, "body", "original alpha content");
    h.put(1, "body", "revised bravo content");
    assert!(h.search("original alpha", 5).unwrap().is_empty());
    let hits = h.search("revised bravo", 5).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].text, "revised bravo content");
}

#[test]
fn remove_and_remove_segment() {
    let h = Harness::new();
    h.upsert(1, &[("a", "alpha bravo"), ("b", "charlie delta")]);
    h.remove_segment(1, "a");
    assert!(
        h.search("alpha bravo", 5).unwrap().is_empty(),
        "segment a gone"
    );
    assert!(
        hit(&h.search("charlie delta", 5).unwrap(), 1),
        "segment b intact"
    );
    h.remove(1);
    assert!(
        h.search("charlie delta", 5).unwrap().is_empty(),
        "whole doc gone"
    );
}

#[test]
fn limit_caps_the_result_count() {
    let h = Harness::new();
    load_fixture(&h);
    let hits = h.search("quick", 2).unwrap();
    assert!(hits.len() <= 2);
}

#[test]
fn no_match_returns_empty_not_error() {
    let h = Harness::new();
    load_fixture(&h);
    let hits = h.search("xylophone zeppelin wombat", 10).unwrap();
    assert!(hits.is_empty());
}

#[test]
fn sub_trigram_query_yields_nothing_without_error() {
    let h = Harness::new();
    load_fixture(&h);
    assert!(h.search("hi", 10).unwrap().is_empty());
    assert!(h.search("", 10).unwrap().is_empty());
}

#[test]
fn empty_index_search_is_empty() {
    let h = Harness::new();
    assert!(h.search("anything at all", 10).unwrap().is_empty());
}

#[test]
fn three_char_query_matches_via_single_trigram() {
    let h = Harness::new();
    h.put(1, "f", "the cat sat");
    // A 3-char query is exactly one trigram; m=1 lets it rank.
    let hits = h
        .search_opts("cat", &SearchOpts::new().min_shared(1), 5)
        .unwrap();
    assert!(hit(&hits, 1));
}
