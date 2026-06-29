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
