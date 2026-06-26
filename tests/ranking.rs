//! The ranking contract that the bit-sliced walk must honor: deterministic
//! tie-breaks, best-segment-per-doc, and early-stop that never drops a top result.

mod common;
use common::*;
use trifle::{Effort, SearchOpts};

#[test]
fn ties_keep_the_lowest_doc_ids_in_ascending_order() {
    let h = Harness::new();
    // Six identical docs => identical overlap; the only thing distinguishing them is
    // the deterministic tie-break (doc ascending), and the limit truncation.
    for doc in 1..=6 {
        h.put(doc, "field", "f", "wxyza klmno pqrst");
    }
    let hits = h.search("wxyza klmno pqrst", SearchOpts::new(3)).unwrap();
    assert_eq!(ids(&hits), [1, 2, 3], "lowest doc ids, ascending");
}

#[test]
fn best_segment_per_doc_is_the_highest_overlap_one() {
    let h = Harness::new();
    // One doc, two segments under different labels. Label "r" shares more query trigrams
    // than label "p".
    h.put(1, "rich", "r", "wxyza klmno pqrst");
    h.put(1, "poor", "p", "wxyza");
    h.put(2, "field", "d", "wxyza"); // distractor so the result isn't trivial
    let hits = h.search("wxyza klmno pqrst", SearchOpts::new(10)).unwrap();
    let doc1 = hits
        .iter()
        .find(|m| m.key.as_i64() == Some(1))
        .expect("doc 1 present");
    assert_eq!(
        doc1.label, "r",
        "the higher-overlap segment represents the doc"
    );
}

#[test]
fn best_segment_tie_breaks_on_the_lowest_id() {
    let h = Harness::new();
    // Two segments of equal overlap; the first-inserted (lower id) must win.
    h.put(1, "first", "a", "wxyza klmno"); // inserted first -> lower id
    h.put(1, "second", "b", "wxyza klmno"); // identical overlap, higher id
    let hits = h.search("wxyza klmno", SearchOpts::new(10)).unwrap();
    assert_eq!(hits[0].label, "a", "lowest-id segment wins an overlap tie");
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
    let top = h.search("quick brown fox", SearchOpts::new(2)).unwrap();
    assert_eq!(ids(&top), [1, 2], "the cut is by overlap, not by doc order");
    // With room, the lower bucket IS reachable — proving the exclusion above was the
    // limit, not a lost candidate — and the higher bucket still ranks first.
    let all = h.search("quick brown fox", SearchOpts::new(10)).unwrap();
    assert_eq!(&ids(&all)[..2], [1, 2]);
    assert!(hit(&all, 3) && hit(&all, 4) && hit(&all, 5));
}

#[test]
fn overlap_engine_does_not_promote_a_short_verbatim_match() {
    // trifle is a lexical-overlap engine, not a relevance engine: with equal overlap AND
    // equal gram rarity, a short verbatim match is NOT promoted over equal-overlap distractors
    // (there is no length-normalizing relevance tier — a caller wanting that supplies a custom
    // Ranker). This pins that contract.
    let h = Harness::new();
    let answer = 100;
    // Distractors inserted FIRST (lower internal ids → win the overlap tie-break); they share
    // the query's two grams ("xqz", "qzv") split across a long doc. The short verbatim answer
    // goes last.
    for doc in 1..=12 {
        h.put(
            doc,
            "field",
            "d",
            "xqz aaaa bbbb cccc dddd eeee ffff gggg qzv",
        );
    }
    h.put(answer, "field", "a", "xqzv");

    // Both share the query's two grams (equal overlap), and the grams are equally common
    // (every doc has them) so weighting is uniform → the tie falls to insertion order.
    let hits = h.search("xqzv", SearchOpts::new(1)).unwrap();
    assert_eq!(
        hits[0].key.as_i64(),
        Some(1),
        "equal overlap + equal rarity → insertion order, not relevance promotion"
    );

    // A deep over-fetch pool alone changes nothing without a custom ranker.
    let deep = h
        .search("xqzv", SearchOpts::new(1).rerank(Effort::High))
        .unwrap();
    assert_eq!(
        deep[0].key.as_i64(),
        Some(1),
        "over-fetch alone does not reorder — there is no built-in reranker"
    );
}
