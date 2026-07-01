//! Normalization, casefolding, non-ASCII text, and class-normalized multi-script rarity.

mod common;
use common::*;

use trifle::store::Sidecar;
use trifle::tokenize::{DefaultTokenizer, Normalization};
use trifle::{Config, Index, Schema, SearchOpts};

fn index_with(tok: DefaultTokenizer) -> (Index<DefaultTokenizer>, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let store = Sidecar::open(dir.path().join("t.db")).unwrap();
    let idx = Index::open(store, tok, Schema::flat(), Config::default()).unwrap();
    (idx, dir)
}

#[test]
fn nfc_default_matches_composed_and_decomposed_forms() {
    let h = Harness::new();
    // Stored decomposed (e + combining acute), queried composed (é): canonical equivalence
    // means they share trigrams under NFC.
    h.put(1, "f", "cafe\u{301} terrasse ambiance");
    let hits = h.search("caf\u{e9} terrasse", 5).unwrap();
    assert!(hit(&hits, 1));
}

#[test]
fn casefolding_is_on_by_default() {
    let h = Harness::new();
    h.put(1, "f", "MACEDONIA THESSALONIKI");
    assert!(hit(&h.search("macedonia", 5).unwrap(), 1));
}

#[test]
fn strip_marks_makes_search_accent_insensitive() {
    let (idx, _dir) = index_with(
        DefaultTokenizer::builder()
            .normalization(Normalization::NfdStripMarks)
            .build(),
    );
    {
        let mut w = idx.writer().unwrap();
        w.upsert(1, &[("f", "café résumé naïve")]).unwrap();
        w.commit().unwrap();
    }
    let hits = idx
        .reader()
        .unwrap()
        .matches("cafe resume", &SearchOpts::new(), 5)
        .unwrap();
    assert!(hit(&hits, 1));
}

#[test]
fn nfc_default_keeps_distinct_accents_apart() {
    let h = Harness::new();
    h.put(1, "f", "résumé");
    h.put(2, "f", "presume assume consume");
    let hits = h.search("resume", 10).unwrap();
    assert!(
        !hit(&hits, 1),
        "résumé shares only 'sum' with 'resume' — below the floor"
    );
    assert!(hit(&hits, 2), "the unaccented distractors do match");
}

#[test]
fn non_latin_scripts_are_searchable() {
    let h = Harness::new();
    h.put(1, "f", "Москва столица России");
    h.put(2, "f", "東京都は日本の首都です");
    assert!(hit(&h.search("Москва столица", 5).unwrap(), 1));
    assert!(hit(&h.search("東京都は日本", 5).unwrap(), 2));
}

#[test]
fn mixed_script_query_finds_each_script() {
    // A document mixing Latin + CJK; class-aware selection compares grams across the two df
    // regimes fairly, so a query in either script still finds it.
    let h = Harness::new();
    h.put(1, "f", "東京 tokyo metropolis 日本 japan");
    assert!(hit(&h.search("tokyo metropolis", 5).unwrap(), 1));
    assert!(hit(&h.search("東京 日本", 5).unwrap(), 1));
}

#[test]
fn emoji_and_wide_chars_do_not_break_indexing() {
    let h = Harness::new();
    h.put(1, "f", "deploy 🚀 to production 🎉 now");
    assert!(hit(&h.search("production", 5).unwrap(), 1));
}

#[test]
fn majority_script_cannot_bury_the_minority_under_stop_and_budget() {
    // v0.4/M4 §5/§8 per-class floor, end-to-end: a Latin-majority corpus where the Latin query grams
    // are themselves rare enough to satisfy the stop, plus one CJK doc. The per-class floor must
    // still seat the minority Han class so its doc is found — representation is an invariant, not a
    // tendency, even when the majority would clear the stop on its own.
    let h = Harness::new();
    {
        let mut w = h.index.writer().unwrap();
        for d in 1..=40i64 {
            // distinctive-ish Latin so the query's Latin grams aren't all df≈N
            let body = format!("alphagamma betadelta epsilonzeta token{d}");
            w.upsert(d, &[("f", body.as_str())]).unwrap();
        }
        w.upsert(99, &[("f", "alphagamma 東京日本語")]).unwrap();
        w.commit().unwrap();
    }
    // Tight budget + a small k (earlier stop). The Han grams (df=1) are the rarest in their class.
    let opts = SearchOpts::new().df_budget(12).min_shared(1).k_target(64);
    let hits = h
        .index
        .reader()
        .unwrap()
        .matches("alphagamma 東京日本語", &opts, 10)
        .unwrap();
    assert!(
        hit(&hits, 99),
        "the minority Han class is seated by the per-class floor; doc 99 found: {:?}",
        ids(&hits)
    );
}

// ===== v0.4/M5 dual-order rank-views (derivation §8) =============================================

#[test]
fn cjk_one_char_query_reaches_the_unigram_fallback() {
    // A CJK run's primary order is the bigram; the secondary is the unigram. A 1-char Han query is
    // too short to produce a bigram, so it falls back to the unigram (secondary) rank-view — the
    // structural fallback (§8) — and matches a doc containing that morpheme. (v0.3/M4 returned
    // empty: a lone Han char produced no bigram.)
    let h = Harness::new();
    h.put(1, "f", "漢字研究");
    h.put(2, "f", "言語学");
    let hits = h.search("漢", 10).unwrap();
    assert!(
        hit(&hits, 1),
        "a 1-char CJK query matches via the unigram structural fallback"
    );
    assert!(!hit(&hits, 2), "an unrelated CJK doc is not matched");
}

#[test]
fn cjk_bigram_query_matches_via_the_primary_then_unigram_views() {
    // A 2-char Han query produces one bigram (the primary). With a single in-corpus primary gram it
    // is corroboratively starved, so the unigram secondary view also runs and the two are fused —
    // and the query matches the doc that contains the bigram.
    let h = Harness::new();
    h.put(1, "f", "日本語の文法");
    h.put(2, "f", "中国語の文字");
    let hits = h.search("日本", 10).unwrap();
    assert!(hit(&hits, 1), "the 日本 bigram matches doc 1");
    assert!(!hit(&hits, 2), "doc 2 (no 日本) is not matched");
}

#[test]
fn mixed_script_query_pools_the_primary_view() {
    // A clean mixed-script query pools every script's PRIMARY order into one view (Latin trigrams ∪
    // CJK bigrams — disjoint, so pooled, not double-counted; §8). A doc containing both a Latin and a
    // CJK match outranks docs matching only one script.
    let h = Harness::new();
    h.put(1, "f", "tokyo 東京 metro"); // both Latin "tokyo"/"metro" and Han 東京
    h.put(2, "f", "tokyo airport bus"); // Latin only
    h.put(3, "f", "京都 大阪 名古屋"); // Han only, unrelated
    let hits = h.search("tokyo metro 東京", 10).unwrap();
    assert!(hit(&hits, 1), "the doc matching both scripts is found");
    assert_eq!(
        hits[0].key.as_i64(),
        Some(1),
        "pooling both scripts' primary grams ranks the dual-script match first"
    );
}

#[test]
fn clean_single_script_query_is_primary_only_and_pays_nothing_for_the_secondary() {
    // The §9 requirement: a clean (not starved) single-script query runs the PRIMARY view ALONE — it
    // pays nothing for §8. Proof: a doc that shares only BIGRAMS with the query (no trigram) must NOT
    // match, because the secondary (bigram) view never forms. "quick brown" has many in-corpus
    // trigrams ⇒ not starved ⇒ primary-only; "quack" shares the bigrams "qu"/"ck" but no trigram, so
    // it stays unmatched.
    let h = Harness::new();
    h.put(1, "f", "quick brown fox");
    h.put(2, "f", "quack"); // shares bigrams qu, ck with "quick" but NO trigram
    let hits = h.search("quick brown", 10).unwrap();
    assert!(hit(&hits, 1), "the trigram match is found");
    assert!(
        !hit(&hits, 2),
        "a clean query stays primary-only: a bigram-only coincidence does not match"
    );
}
