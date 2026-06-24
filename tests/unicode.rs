//! Normalization, casefolding, and non-ASCII text.

mod common;
use common::*;

use trifle::store::Sidecar;
use trifle::tokenize::{Normalization, TrigramTokenizer};
use trifle::{Config, Index, SearchOpts};

fn index_with(tok: TrigramTokenizer) -> (Index<TrigramTokenizer, Sidecar>, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let backend = Sidecar::open(dir.path().join("t.db")).unwrap();
    let idx = Index::open(backend, tok, Config::default()).unwrap();
    (idx, dir)
}

#[test]
fn nfc_default_matches_composed_and_decomposed_forms() {
    let h = Harness::new();
    // Stored decomposed (e + combining acute), queried composed (é) — canonical
    // equivalence means they share trigrams under NFC.
    h.put(1, "field", "f", "cafe\u{301} terrasse ambiance");
    let hits = h
        .index
        .search("caf\u{e9} terrasse", SearchOpts::new(5))
        .unwrap();
    assert!(hit(&hits, 1));
}

#[test]
fn casefolding_is_on_by_default() {
    let h = Harness::new();
    h.put(1, "field", "f", "MACEDONIA THESSALONIKI");
    assert!(hit(
        &h.index.search("macedonia", SearchOpts::new(5)).unwrap(),
        1
    ));
}

#[test]
fn strip_marks_makes_search_accent_insensitive() {
    let (idx, _dir) = index_with(
        TrigramTokenizer::builder()
            .normalization(Normalization::NfdStripMarks)
            .build(),
    );
    idx.insert(1, "field", &[("f", "café résumé naïve")])
        .unwrap();
    // An accent-free query finds the accented text.
    assert!(hit(
        &idx.search("cafe resume", SearchOpts::new(5)).unwrap(),
        1
    ));
}

#[test]
fn nfc_default_keeps_distinct_accents_apart() {
    let h = Harness::new();
    // Under NFC, "résumé" and "resume" share only the trigram "sum". The distractors
    // make the query's other trigrams (res/esu/ume) present corpus-wide, so the floor
    // is a true 2 — and the accented doc, sharing only "sum", falls below it.
    h.put(1, "field", "f", "résumé");
    h.put(2, "field", "f", "presume assume consume");
    let hits = h.index.search("resume", SearchOpts::new(10)).unwrap();
    assert!(
        !hit(&hits, 1),
        "résumé shares only 'sum' with 'resume' — below the floor"
    );
    assert!(hit(&hits, 2), "the unaccented distractors do match");
}

#[test]
fn non_latin_scripts_are_searchable() {
    let h = Harness::new();
    h.put(1, "field", "f", "Москва столица России");
    h.put(2, "field", "f", "東京都は日本の首都です");
    assert!(hit(
        &h.index
            .search("Москва столица", SearchOpts::new(5))
            .unwrap(),
        1
    ));
    assert!(hit(
        &h.index.search("東京都は日本", SearchOpts::new(5)).unwrap(),
        2
    ));
}

#[test]
fn emoji_and_wide_chars_do_not_break_indexing() {
    let h = Harness::new();
    h.put(1, "field", "f", "deploy 🚀 to production 🎉 now");
    // Should not panic; the ascii words remain findable.
    assert!(hit(
        &h.index.search("production", SearchOpts::new(5)).unwrap(),
        1
    ));
}
