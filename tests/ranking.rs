//! The ranking contract that the bit-sliced walk must honor: deterministic
//! tie-breaks, best-segment-per-doc, and early-stop that never drops a top result.

mod common;
use common::*;
use trifle::SearchOpts;

#[test]
fn ties_keep_the_lowest_doc_ids_in_ascending_order() {
    let h = Harness::new();
    // Six identical docs => identical overlap; the only thing distinguishing them is
    // the deterministic tie-break (doc_id ascending), and the limit truncation.
    for doc in 1..=6 {
        h.put(doc, "field", "f", "wxyza klmno pqrst");
    }
    let hits = h
        .index
        .search("wxyza klmno pqrst", SearchOpts::new(3))
        .unwrap();
    assert_eq!(ids(&hits), [1, 2, 3], "lowest doc_ids, ascending");
}

#[test]
fn best_segment_per_doc_is_the_highest_overlap_one() {
    let h = Harness::new();
    // One doc, two segments under different sources (so both survive the upsert).
    // Source "rich" shares more query trigrams than source "poor".
    h.put(1, "rich", "r", "wxyza klmno pqrst");
    h.put(1, "poor", "p", "wxyza");
    h.put(2, "field", "d", "wxyza"); // distractor so the result isn't trivial
    let hits = h
        .index
        .search("wxyza klmno pqrst", SearchOpts::new(10))
        .unwrap();
    let doc1 = hits.iter().find(|m| m.doc_id == 1).expect("doc 1 present");
    assert_eq!(
        doc1.source, "rich",
        "the higher-overlap segment represents the doc"
    );
    assert_eq!(doc1.ref_, "r");
}

#[test]
fn best_segment_tie_breaks_on_the_lowest_id() {
    let h = Harness::new();
    // Two segments of equal overlap; the first-inserted (lower id) must win.
    h.put(1, "first", "a", "wxyza klmno"); // inserted first -> lower id
    h.put(1, "second", "b", "wxyza klmno"); // identical overlap, higher id
    let hits = h.index.search("wxyza klmno", SearchOpts::new(10)).unwrap();
    assert_eq!(
        hits[0].source, "first",
        "lowest-id segment wins an overlap tie"
    );
}

#[test]
fn early_stop_keeps_the_higher_overlap_bucket() {
    let h = Harness::new();
    // Docs 1,2 contain "fox" (a rare distinguisher) and so out-overlap docs 3-5,
    // which still clear the floor on the shared quick/brown trigrams.
    h.put(1, "field", "f", "quick brown fox");
    h.put(2, "field", "f", "quick brown fox");
    for doc in 3..=5 {
        h.put(doc, "field", "f", "quick brown");
    }
    // limit=2 must lock the two higher-overlap docs and never admit the lower bucket.
    let top = h
        .index
        .search("quick brown fox", SearchOpts::new(2))
        .unwrap();
    assert_eq!(ids(&top), [1, 2], "the cut is by overlap, not by doc order");
    // With room, the lower bucket IS reachable — proving the exclusion above was the
    // limit, not a lost candidate — and the higher bucket still ranks first.
    let all = h
        .index
        .search("quick brown fox", SearchOpts::new(10))
        .unwrap();
    assert_eq!(&ids(&all)[..2], [1, 2]);
    assert!(hit(&all, 3) && hit(&all, 4) && hit(&all, 5));
}
