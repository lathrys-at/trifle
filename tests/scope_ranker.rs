//! The scope predicate, custom rankers, and batch == serial parity.

mod common;
use common::*;
use trifle::SearchOpts;
use trifle::rank::{Candidates, QueryContext, Ranked, Ranker};

// ----- scope predicate --------------------------------------------------------

#[test]
fn scope_filters_by_doc_id() {
    let h = Harness::new();
    load_fixture(&h);
    let even = |doc_id: i64, _s: &str, _r: &str| doc_id % 2 == 0;
    let hits = h
        .index
        .search("quick", SearchOpts::new(10).scope(&even))
        .unwrap();
    assert!(!hits.is_empty());
    assert!(hits.iter().all(|m| m.doc_id % 2 == 0));
}

#[test]
fn scope_filters_by_source() {
    let h = Harness::new();
    h.put(1, "field", "f", "diplomatic negotiations concluded");
    h.put(1, "ocr", "scan", "diplomatic negotiations concluded");
    h.put(2, "field", "f", "diplomatic negotiations resumed");
    let only_ocr = |_d: i64, source: &str, _r: &str| source == "ocr";
    let hits = h
        .index
        .search(
            "diplomatic negotiations",
            SearchOpts::new(10).scope(&only_ocr),
        )
        .unwrap();
    assert!(hits.iter().all(|m| m.source == "ocr"));
    assert!(hit(&hits, 1));
}

#[test]
fn scope_walk_fills_limit_with_passing_docs() {
    let h = Harness::new();
    // Many docs match; the scope rejects the lowest doc ids (highest-overlap-first
    // walk must keep descending until `limit` passing docs lock).
    for doc in 1..=6 {
        h.put(doc, "field", "f", "common shared searchable phrase content");
    }
    let keep_high = |doc_id: i64, _s: &str, _r: &str| doc_id >= 4;
    let hits = h
        .index
        .search("searchable phrase", SearchOpts::new(3).scope(&keep_high))
        .unwrap();
    assert_eq!(hits.len(), 3, "limit filled despite rejecting docs 1-3");
    assert!(hits.iter().all(|m| m.doc_id >= 4));
}

// ----- custom rankers ---------------------------------------------------------

/// Reverses the default (overlap) order — an unambiguous reorder.
struct Reversed;
impl Ranker for Reversed {
    fn rank(&self, candidates: &Candidates<'_>, _q: &QueryContext<'_>) -> Vec<Ranked> {
        (0..candidates.len())
            .rev()
            .map(|candidate| Ranked { candidate })
            .collect()
    }
}

#[test]
fn custom_ranker_controls_order() {
    let h = Harness::new();
    h.put(1, "field", "f", "quick brown fox runs");
    h.put(2, "field", "f", "quick"); // lower overlap -> default-last
    let default = h
        .index
        .search("quick brown fox", SearchOpts::new(10))
        .unwrap();
    let reversed = h
        .index
        .search("quick brown fox", SearchOpts::new(10).ranker(&Reversed))
        .unwrap();
    assert_eq!(
        ids(&reversed),
        ids(&default).into_iter().rev().collect::<Vec<_>>()
    );
}

/// A literal-verify precision tier: keep only candidates whose text contains the
/// raw query as a substring.
struct LiteralOnly;
impl Ranker for LiteralOnly {
    fn rank(&self, candidates: &Candidates<'_>, q: &QueryContext<'_>) -> Vec<Ranked> {
        candidates
            .iter()
            .filter(|c| c.text().is_some_and(|t| t.contains(q.query)))
            .map(|c| Ranked {
                candidate: c.index(),
            })
            .collect()
    }
}

#[test]
fn ranker_can_read_text_and_drop_candidates() {
    let h = Harness::new();
    h.put(1, "field", "f", "the brown fox is quick"); // no contiguous "quick brown"
    h.put(2, "field", "f", "a quick brown hare appears"); // has "quick brown"
    let hits = h
        .index
        .search("quick brown", SearchOpts::new(10).ranker(&LiteralOnly))
        .unwrap();
    assert!(hit(&hits, 2));
    assert!(!hit(&hits, 1), "doc 1 lacks the contiguous phrase");
}

/// Asserts the [`Candidate`] invariants from inside a ranker, then passes through.
struct InvariantChecker;
impl Ranker for InvariantChecker {
    fn rank(&self, candidates: &Candidates<'_>, _q: &QueryContext<'_>) -> Vec<Ranked> {
        for c in candidates.iter() {
            // matched_tokens is exactly the set counted by overlap.
            assert_eq!(c.matched_tokens().len() as u32, c.overlap());
            assert!(c.overlap() >= 1);
        }
        (0..candidates.len())
            .map(|candidate| Ranked { candidate })
            .collect()
    }
}

#[test]
fn matched_tokens_equals_overlap_count() {
    let h = Harness::new();
    load_fixture(&h);
    let hits = h
        .index
        .search(
            "quick brown fox",
            SearchOpts::new(10).ranker(&InvariantChecker),
        )
        .unwrap();
    assert!(!hits.is_empty());
}

// ----- batch == serial --------------------------------------------------------

#[test]
fn search_batch_matches_serial_search_exactly() {
    let h = Harness::new();
    load_fixture(&h);
    let queries = [
        "quick brown",
        "lazy dog",
        "wizards jump",
        "nonexistent zzqqx",
    ];
    let batched = h.index.search_batch(&queries, SearchOpts::new(10)).unwrap();
    for (i, q) in queries.iter().enumerate() {
        let serial = h.index.search(q, SearchOpts::new(10)).unwrap();
        assert_eq!(
            ids(&batched[i]),
            ids(&serial),
            "batch and serial must rank {q:?} identically"
        );
    }
}

#[test]
fn empty_batch_returns_empty() {
    let h = Harness::new();
    load_fixture(&h);
    assert!(
        h.index
            .search_batch(&[], SearchOpts::new(10))
            .unwrap()
            .is_empty()
    );
}
