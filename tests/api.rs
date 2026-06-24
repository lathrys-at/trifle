//! Public-API contracts: option edges, thread-safety, and the scope-predicate
//! guarantees the spec states (candidates-only, descending overlap, early-stop).

mod common;
use common::*;

use std::sync::Mutex;

use trifle::store::{Shared, Sidecar};
use trifle::tokenize::TrigramTokenizer;
use trifle::{Index, SearchOpts};

#[test]
fn limit_zero_returns_empty() {
    let h = Harness::new();
    load_fixture(&h);
    assert!(
        h.index
            .search("quick brown fox", SearchOpts::new(0))
            .unwrap()
            .is_empty()
    );
}

#[test]
fn a_huge_limit_does_not_panic_or_overallocate() {
    let h = Harness::new();
    load_fixture(&h);
    let hits = h
        .index
        .search("quick", SearchOpts::new(usize::MAX))
        .unwrap();
    assert!(!hits.is_empty() && hits.len() <= FIXTURE.len());
}

#[test]
fn min_shared_zero_behaves_as_one() {
    let h = Harness::new();
    h.put(1, "field", "f", "wxy"); // a lone trigram shared with "wxyz"
    let zero = h
        .index
        .search("wxyz", SearchOpts::new(5).min_shared(0))
        .unwrap();
    let one = h
        .index
        .search("wxyz", SearchOpts::new(5).min_shared(1))
        .unwrap();
    assert_eq!(ids(&zero), ids(&one));
    assert!(
        hit(&zero, 1),
        "m=0 is clamped to 1, admitting the single-trigram hit"
    );
}

#[test]
fn scope_can_borrow_local_state() {
    // A scope predicate that closes over a stack-local set must compile and work
    // (the ScopeFn lifetime exists precisely so this is not forced to be 'static).
    let h = Harness::new();
    load_fixture(&h);
    let allowed: std::collections::HashSet<i64> = [1, 4].into_iter().collect();
    let pred = |doc_id: i64, _s: &str, _r: &str| allowed.contains(&doc_id);
    let hits = h
        .index
        .search("the lazy dog", SearchOpts::new(10).scope(&pred))
        .unwrap();
    assert!(!hits.is_empty());
    assert!(hits.iter().all(|m| allowed.contains(&m.doc_id)));
}

fn _assert_send_sync<T: Send + Sync>() {}

#[test]
fn index_monomorphizations_are_send_and_sync() {
    // Shared `&self` across threads is a documented contract; pin it at compile time
    // for both backends (the contentless `Box<dyn TextResolver>` field lives in the
    // same `Index<_, Sidecar>` type, so this covers it too).
    _assert_send_sync::<Index<TrigramTokenizer, Sidecar>>();
    _assert_send_sync::<Index<TrigramTokenizer, Shared>>();
}

#[test]
fn a_span_always_implies_text() {
    let h = Harness::new();
    load_fixture(&h);
    for q in ["quick brown fox", "lazy dog", "wizards jump"] {
        for m in h.index.search(q, SearchOpts::new(10)).unwrap() {
            if m.span.is_some() {
                assert!(m.text.is_some(), "a span requires text to index into");
            }
            // And a returned span is always sliceable without panic.
            if let (Some((lo, hi)), Some(text)) = (m.span, m.text.as_deref()) {
                assert!(text.is_char_boundary(lo) && text.is_char_boundary(hi));
                let _ = &text[lo..hi];
            }
        }
    }
}

#[test]
fn scope_is_never_called_over_non_candidates() {
    let h = Harness::new();
    // Docs 1-3 share the query's vocabulary; docs 50-52 share nothing.
    h.put(1, "field", "f", "alpha beta gamma");
    h.put(2, "field", "f", "alpha beta delta");
    h.put(3, "field", "f", "alpha beta epsilon");
    for doc in 50..=52 {
        h.put(doc, "field", "f", "zzz qqq wholly unrelated vocabulary");
    }
    let seen: Mutex<Vec<i64>> = Mutex::new(Vec::new());
    {
        let record = |doc_id: i64, _s: &str, _r: &str| {
            seen.lock().unwrap().push(doc_id);
            true
        };
        let _ = h
            .index
            .search("alpha beta", SearchOpts::new(10).scope(&record))
            .unwrap();
    }
    let seen = seen.into_inner().unwrap();
    assert!(!seen.is_empty(), "candidates were scoped");
    assert!(
        seen.iter().all(|d| (1..=3).contains(d)),
        "scope ran only over candidates, never the unrelated corpus: {seen:?}"
    );
}

#[test]
fn scope_invocations_are_bounded_by_early_stop() {
    let h = Harness::new();
    // A small high-overlap bucket and a large low-overlap one. With a small limit,
    // the walk must lock on the high bucket and never scope the low one.
    for doc in 1..=3 {
        h.put(doc, "field", "f", "quick brown fox");
    }
    for doc in 10..=24 {
        h.put(doc, "field", "f", "quick brown");
    }
    let count: Mutex<usize> = Mutex::new(0);
    let hits = {
        let counting = |_d: i64, _s: &str, _r: &str| {
            *count.lock().unwrap() += 1;
            true
        };
        h.index
            .search("quick brown fox", SearchOpts::new(2).scope(&counting))
            .unwrap()
    };
    assert_eq!(ids(&hits), [1, 2]);
    let n = count.into_inner().unwrap();
    // Only the high-overlap bucket (3 docs) is scoped; the 15 low-overlap docs aren't.
    assert!(
        n <= 3,
        "early-stop must not scope the lower bucket (scoped {n})"
    );
}
