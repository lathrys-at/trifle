//! Round-trip behaviors: provenance, text, spans, multi-segment docs, limits.

mod common;
use common::*;
use trifle::SearchOpts;

#[test]
fn exact_query_finds_the_document() {
    let h = Harness::new();
    load_fixture(&h);
    let hits = h.search("quick brown fox", SearchOpts::new(10)).unwrap();
    assert!(hit(&hits, 1), "doc 1 contains the phrase");
}

#[test]
fn match_carries_provenance_and_text() {
    let h = Harness::new();
    h.put(42, "ocr", "page-3.png", "the treaty was signed in vienna");
    let hits = h.search("treaty signed", SearchOpts::new(10)).unwrap();
    let m = hits
        .iter()
        .find(|m| m.key.as_i64() == Some(42))
        .expect("found");
    // The segment label (the v0.1 `ref`) is returned; `source` no longer exists.
    assert_eq!(m.label, "page-3.png");
    assert_eq!(m.text.as_deref(), Some("the treaty was signed in vienna"));
}

#[test]
fn span_indexes_the_matched_region_of_the_text() {
    let h = Harness::new();
    h.put(1, "field", "f", "alpha bravo charlie delta");
    let hits = h.search("charlie", SearchOpts::new(5)).unwrap();
    let m = &hits[0];
    let (lo, hi) = m.span.expect("a span for a clean ascii match");
    let text = m.text.as_deref().unwrap();
    // The span must be valid byte offsets bracketing the matched word.
    assert!(lo < hi && hi <= text.len());
    let region = &text[lo..hi];
    assert!(region.contains("charlie"), "span region was {region:?}");
}

#[test]
fn one_match_per_doc_even_with_many_segments() {
    let h = Harness::new();
    // Two segments under one document: both mention "quartz".
    let mut w = h.index.writer().unwrap();
    w.insert(
        7,
        &[
            ("front", "the quartz crystal"),
            ("back", "quartz is a mineral"),
        ],
    )
    .unwrap();
    w.commit().unwrap();
    drop(w);
    let hits = h.search("quartz crystal", SearchOpts::new(10)).unwrap();
    assert_eq!(
        hits.iter().filter(|m| m.key.as_i64() == Some(7)).count(),
        1,
        "a doc appears at most once"
    );
}

#[test]
fn distinct_segments_are_searchable_and_labeled() {
    let h = Harness::new();
    let mut w = h.index.writer().unwrap();
    w.insert(
        1,
        &[
            ("front", "mitochondria powerhouse cell"),
            ("scan", "ribosome protein synthesis"),
        ],
    )
    .unwrap();
    w.commit().unwrap();
    drop(w);
    let a = h.search("mitochondria", SearchOpts::new(5)).unwrap();
    let b = h.search("ribosome", SearchOpts::new(5)).unwrap();
    assert_eq!(a[0].label, "front");
    assert_eq!(b[0].label, "scan");
}

#[test]
fn limit_caps_the_result_count() {
    let h = Harness::new();
    load_fixture(&h);
    // "quick" appears in several docs; ask for only 2.
    let hits = h.search("quick", SearchOpts::new(2)).unwrap();
    assert!(hits.len() <= 2);
}

#[test]
fn no_match_returns_empty_not_error() {
    let h = Harness::new();
    load_fixture(&h);
    let hits = h
        .search("xylophone zeppelin wombat", SearchOpts::new(10))
        .unwrap();
    assert!(hits.is_empty());
}

#[test]
fn sub_trigram_query_yields_nothing_without_error() {
    let h = Harness::new();
    load_fixture(&h);
    assert!(h.search("hi", SearchOpts::new(10)).unwrap().is_empty());
    assert!(h.search("", SearchOpts::new(10)).unwrap().is_empty());
}

#[test]
fn empty_index_search_is_empty() {
    let h = Harness::new();
    assert!(
        h.search("anything at all", SearchOpts::new(10))
            .unwrap()
            .is_empty()
    );
}

#[test]
fn three_char_query_matches_via_single_trigram() {
    let h = Harness::new();
    h.put(1, "field", "f", "the cat sat");
    // A 3-char query is exactly one trigram; the floor drops to 1 so it ranks.
    let hits = h.search("cat", SearchOpts::new(5)).unwrap();
    assert!(hit(&hits, 1));
}
