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

// ===== per-script secondary gate + the starved-gate narrowing (panel regression guards) =========

/// §8/§12 PER-SCRIPT gate: the secondary view pools ONLY the starved scripts' grams — "a rich
/// script with a primary gram omits its secondary." Latin ("quick brown") is rich (many in-corpus
/// trigrams ⇒ not starved); only the lone CJK 日 is starved (1-char ⇒ no bigram). So the secondary
/// view holds ONLY the CJK unigram 日, never Latin bigrams. Doc 2 ("quack squawk") shares Latin
/// bigrams with quick/brown but no trigram and no CJK — a whole-query gate would leak it in.
#[test]
fn per_script_secondary_gate_does_not_leak_a_rich_script() {
    let h = Harness::new();
    h.put(1, "f", "quick brown 日本語");
    h.put(2, "f", "quack squawk");
    let hits = h.search("quick brown 日", 10).unwrap();
    assert!(hit(&hits, 1), "the genuine dual-script doc matches");
    assert!(
        !hit(&hits, 2),
        "a rich script's bigram coincidences must NOT leak via the secondary view: {:?}",
        ids(&hits)
    );
    // Dropping the lone CJK char: a clean Latin-only query is primary-only and also excludes doc 2.
    assert!(!hit(&h.search("quick brown", 10).unwrap(), 2));
}

/// The narrowing of §12's literal while-loop: a query that PRODUCED primary grams that are all
/// ABSENT (df = 0) stays a no-match — it does not fuzzily fall to the bigram layer. "zqj" produces
/// the trigram "zqj" (absent), and its bigrams "zq"/"qj" both occur in doc 1, but the secondary view
/// never forms (no present primary to corroborate), so doc 1 does not leak in.
#[test]
fn all_absent_primary_does_not_leak_via_incidental_bigram() {
    let h = Harness::new();
    h.put(1, "f", "azqb xqjy");
    h.put(2, "f", "totally different words here");
    let hits = h.search("zqj", 10).unwrap();
    assert!(
        !hit(&hits, 1),
        "all-absent primary must not leak via bigram: {:?}",
        ids(&hits)
    );
    assert!(hits.is_empty());
}

/// One surviving in-corpus primary trigram makes the query corroboratively starved (< ν present),
/// so the bigram secondary view runs and corroborates the match. "zqb" → trigram zqb (present in
/// "azqb xyz") ⇒ starved ⇒ bigrams zq, qb corroborate.
#[test]
fn one_surviving_primary_trigram_enables_bigram_corroboration() {
    let h = Harness::new();
    h.put(1, "f", "azqb xyz");
    h.put(2, "f", "unrelated control text");
    assert!(hit(&h.search("zqb", 10).unwrap(), 1));
}

/// A query too short to produce any primary gram reaches the structural bigram fallback (§8).
#[test]
fn too_short_query_uses_the_structural_bigram_fallback() {
    let h = Harness::new();
    h.put(1, "f", "go team");
    assert!(hit(&h.search("go", 10).unwrap(), 1));
}

/// v0.5 regression (post-v0.4 review §1.2): an INTERIOR digit bigram must not trip a spurious
/// secondary view on a clean query. In `ab12cd`, every primary trigram carries a Latin letter
/// (class Latin), but the secondary bigram `12` has no strong script (class `Common`) — and
/// `Common` never produces a primary-order gram in mixed text, so the pre-v0.5 per-class
/// structural trigger (`produced == 0`) marked it starved and formed a secondary view for a fully
/// corroborated query. Doc 2 shares only the sub-gram bigrams `12`/`cd` (no trigram), so it can
/// surface only through that spurious secondary view.
#[test]
fn interior_digit_bigram_does_not_activate_the_secondary_view() {
    let h = Harness::new();
    h.put(1, "f", "ab12cd marker");
    h.put(2, "f", "ee12ff ggcdhh"); // shares the bigrams "12" and "cd", but no trigram
    h.put(3, "f", "unrelated control text");
    let hits = h.search("ab12cd", 10).unwrap();
    assert!(
        hit(&hits, 1),
        "the genuine doc matches via primary trigrams"
    );
    assert!(
        !hit(&hits, 2),
        "a digit-bigram coincidence must not leak into a clean digit-bearing query: {:?}",
        ids(&hits)
    );
}

/// The counterpart guard: a STANDALONE digit word (its word produced no primary gram) is still
/// structurally starved, so pure- and mixed-query digit words keep their secondary-view signal
/// (the v0.5 `Common` rule is word-granular, not a blanket exclusion).
#[test]
fn standalone_digit_word_still_reaches_the_secondary_view() {
    let h = Harness::new();
    h.put(1, "f", "xx 12 yy");
    h.put(2, "f", "totally unrelated");
    assert!(
        hit(&h.search("12", 10).unwrap(), 1),
        "a pure digit query still matches via the structural fallback"
    );
}
