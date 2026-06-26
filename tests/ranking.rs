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
fn effort_over_fetch_lets_the_reranker_recover_a_buried_answer() {
    // The over-fetch + BM25 rerank + truncate path only fires when the pool is deeper
    // than `limit` (pool = max(limit, round(c·√(limit·N)))). On a small corpus the
    // default levels floor to `limit`, so force a deep pool with Effort::Custom and
    // confirm the reranker pulls in an answer that raw overlap order buried past `limit`.
    let h = Harness::new();
    let answer = 100;
    // Insert the distractors FIRST (so they take the lower *internal* doc ids — the
    // overlap tie-break is the internal id = insertion order), then the answer last so it
    // is buried in raw overlap order. The distractors share the same two trigrams ("xqz",
    // "qzv") but split apart in a long doc: equal overlap, yet no literal "xqzv" and a
    // long length — both BM25 signals (length normalization + literal boost) favor the
    // short verbatim answer.
    for doc in 1..=12 {
        h.put(
            doc,
            "field",
            "d",
            "xqz aaaa bbbb cccc dddd eeee ffff gggg qzv",
        );
    }
    // The answer: a short doc that is a verbatim match for the query word.
    h.put(answer, "field", "a", "xqzv");

    // Overlap-only (pool == limit): the equal-overlap distractors win the tie on the
    // lower internal id, so the answer is not the top-1.
    let overlap = h
        .search("xqzv", SearchOpts::new(1).rerank(Effort::None))
        .unwrap();
    assert_eq!(
        overlap[0].key.as_i64(),
        Some(1),
        "overlap order buries the answer behind lower-id ties"
    );

    // Deep pool + BM25 rerank: the short verbatim answer scores highest and surfaces.
    let reranked = h
        .search("xqzv", SearchOpts::new(1).rerank(Effort::Custom(5.0)))
        .unwrap();
    assert_eq!(
        reranked[0].key.as_i64(),
        Some(answer),
        "the reranker recovers the buried answer from the over-fetched pool"
    );
}
