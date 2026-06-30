//! v0.4/M5 rank-view RRF fusion, end-to-end (derivation §8): the secondary view improves recall
//! under starvation, the starved gate / structural fallback fire only when needed, and the fused
//! ordering is `batch == serial` (a pure function of the query's grams + the shared snapshot).

mod common;
use common::*;

use trifle::SearchOpts;

/// A starved query (its result fuses two rank-views) must rank IDENTICALLY whether run alone
/// (`matches`) or inside a batch (`matches_batch`) — the rank-views / starved gate / ΔH / RRF are
/// pure functions of THIS query's grams + the shared snapshot, never a batch aggregate.
#[test]
fn fused_query_batch_equals_serial() {
    let h = Harness::new();
    h.put(1, "f", "漢字研究所");
    h.put(2, "f", "言語処理");
    h.put(3, "f", "quick brown fox");

    // "漢" (1-char CJK, structural fallback → unigram view) and "日本" (2-char, corroboratively
    // starved → bigram + unigram fused) are both starved; "quick brown" is clean (primary-only).
    let queries = ["漢", "quick brown", "字研"];
    let serial: Vec<Vec<i64>> = queries
        .iter()
        .map(|q| ids(&h.search(q, 10).unwrap()))
        .collect();
    let batched = h
        .index
        .reader()
        .unwrap()
        .matches_batch(&queries, &SearchOpts::new(), 10)
        .unwrap();
    let batched_ids: Vec<Vec<i64>> = batched.iter().map(|ms| ids(ms)).collect();
    assert_eq!(
        serial, batched_ids,
        "a fused (starved) query ranks identically serial and in a batch"
    );
}

/// The secondary (bigram) view rescues recall when a typo destroys most of a short word's trigrams.
/// "café" → trigrams caf, afé; a substitution in the middle can kill both trigrams while the edge
/// bigrams survive, and the bigram view then still finds the doc.
#[test]
fn corrupted_short_word_recovered_via_the_bigram_view() {
    let h = Harness::new();
    h.put(1, "f", "café"); // grams: trigrams caf, afé; bigrams ca, af, fé
    h.put(2, "f", "zzz unrelated text"); // control
    // "cafz": trigrams caf, afz (afz absent). One in-corpus trigram (caf) ⇒ corroboratively starved
    // ⇒ the bigram view (ca, af) also runs, corroborating the match to doc 1.
    let hits = h.search("cafz", 10).unwrap();
    assert!(
        hit(&hits, 1),
        "a corrupted short word still finds the doc via the fused bigram view"
    );
    assert!(!hit(&hits, 2), "an unrelated doc is not matched");
}

/// A clean, gram-rich single-script query stays primary-only (it pays nothing for the secondary
/// view): a doc sharing only sub-grams must not leak in.
#[test]
fn clean_query_does_not_activate_the_secondary_view() {
    let h = Harness::new();
    load_fixture(&h);
    // "quickly" is rich in in-corpus trigrams (qui, uic, ...), so it is NOT starved → primary-only.
    // Doc 8 ("the five boxing wizards jump quickly") matches; the result is the trigram-overlap set,
    // unaffected by any bigram coincidence.
    let hits = h.search("quickly", 10).unwrap();
    assert!(
        hit(&hits, 8),
        "the trigram-rich clean query matches via the primary view"
    );
}

/// The fused candidate stream's `matched_terms` reports grams from BOTH rank-views for a starved
/// query (a fused candidate may have matched in either view).
#[test]
fn fused_stream_matched_terms_span_both_views() {
    let h = Harness::new();
    h.put(1, "f", "日本語");
    let reader = h.index.reader().unwrap();
    let opts = SearchOpts::new();
    let mut stream = reader.candidates("日本", &opts).unwrap();
    let first = stream.next().expect("a candidate").unwrap();
    let matched: Vec<String> = stream
        .matched_terms(&first)
        .map(|(t, _)| t.to_string())
        .collect();
    // The 2-char query is starved, so both the bigram (日本) and unigram (日, 本) views contribute;
    // the doc contains all of them.
    assert!(
        matched.iter().any(|t| t == "日本"),
        "the primary bigram is among the matched terms: {matched:?}"
    );
    assert!(
        matched.iter().any(|t| t.chars().count() == 1),
        "a secondary unigram is also among the matched terms: {matched:?}"
    );
}
