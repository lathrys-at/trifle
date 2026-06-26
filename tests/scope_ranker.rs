//! The scope predicate, custom rankers, and batch == serial parity.

mod common;
use common::*;
use std::collections::HashSet;
use trifle::rank::{Candidates, QueryContext, Ranked, Ranker};
use trifle::tokenize::{DefaultTokenizer, Tokenizer};
use trifle::{Key, SearchOpts};

// ----- scope predicate --------------------------------------------------------

#[test]
fn scope_filters_by_key() {
    let h = Harness::new();
    load_fixture(&h);
    let even = |key: &Key, _label: &str| key.as_i64().is_some_and(|d| d % 2 == 0);
    let hits = h
        .index
        .reader()
        .unwrap()
        .search("quick", SearchOpts::new(10).scope(&even))
        .unwrap();
    assert!(!hits.is_empty());
    assert!(hits.iter().all(|m| m.key.as_i64().unwrap() % 2 == 0));
}

#[test]
fn scope_filters_by_label() {
    let h = Harness::new();
    // Two segments of doc 1 under different labels; the scope keeps only one label.
    h.put(1, "field", "f", "diplomatic negotiations concluded");
    h.put(1, "ocr", "scan", "diplomatic negotiations concluded");
    h.put(2, "field", "f", "diplomatic negotiations resumed");
    let only_scan = |_k: &Key, label: &str| label == "scan";
    let hits = h
        .index
        .reader()
        .unwrap()
        .search(
            "diplomatic negotiations",
            SearchOpts::new(10).scope(&only_scan),
        )
        .unwrap();
    assert!(hits.iter().all(|m| m.label == "scan"));
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
    let keep_high = |key: &Key, _label: &str| key.as_i64().is_some_and(|d| d >= 4);
    let hits = h
        .index
        .reader()
        .unwrap()
        .search("searchable phrase", SearchOpts::new(3).scope(&keep_high))
        .unwrap();
    assert_eq!(hits.len(), 3, "limit filled despite rejecting docs 1-3");
    assert!(hits.iter().all(|m| m.key.as_i64().unwrap() >= 4));
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
    let default = h.search("quick brown fox", SearchOpts::new(10)).unwrap();
    let reversed = h
        .index
        .reader()
        .unwrap()
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
        .reader()
        .unwrap()
        .search("quick brown", SearchOpts::new(10).ranker(&LiteralOnly))
        .unwrap();
    assert!(hit(&hits, 2));
    assert!(!hit(&hits, 1), "doc 1 lacks the contiguous phrase");
}

/// Asserts the [`Candidate`] invariants from inside a ranker, then passes through.
/// Crucially it re-derives the matched tokens from the segment's *text* (an
/// independent source from the posting-derived overlap), so the check is not
/// circular.
struct InvariantChecker;
impl Ranker for InvariantChecker {
    fn rank(&self, candidates: &Candidates<'_>, q: &QueryContext<'_>) -> Vec<Ranked> {
        let tok = DefaultTokenizer::new();
        for c in candidates.iter() {
            let matched = c.matched_tokens();
            // The count agrees with the bit-sliced overlap.
            assert_eq!(matched.len() as u32, c.overlap());
            assert!(c.overlap() >= 1);
            // Each matched token is genuinely a token of THIS segment's text (recounted
            // independently of the postings) and is one of the query's selected tokens.
            let seg_tokens: HashSet<String> = tok
                .tokenize(c.text().unwrap())
                .map(|g| g.to_string())
                .collect();
            for mt in &matched {
                assert!(
                    seg_tokens.contains(*mt),
                    "matched token {mt:?} absent from segment"
                );
                assert!(
                    q.selected.iter().any(|s| s == mt),
                    "matched token {mt:?} not selected"
                );
            }
        }
        (0..candidates.len())
            .map(|candidate| Ranked { candidate })
            .collect()
    }
}

#[test]
fn matched_tokens_are_real_segment_tokens_counting_overlap() {
    let h = Harness::new();
    load_fixture(&h);
    let hits = h
        .index
        .reader()
        .unwrap()
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
        "quick brown", // same query twice — must not bleed into itself
        "quick lazy",  // overlapping vocabulary with the others (shared reads)
        "nonexistent zzqqx",
    ];
    let batched = h.search_batch(&queries, SearchOpts::new(10)).unwrap();
    for (i, q) in queries.iter().enumerate() {
        let serial = h.search(q, SearchOpts::new(10)).unwrap();
        // Full Match equality (provenance, span, text, order) — not just doc ids — so
        // a batch hydration that picked a different segment would be caught.
        assert_eq!(
            batched[i], serial,
            "batch and serial must be identical for {q:?}"
        );
    }
}

#[test]
fn empty_batch_returns_empty() {
    let h = Harness::new();
    load_fixture(&h);
    assert!(h.search_batch(&[], SearchOpts::new(10)).unwrap().is_empty());
}
