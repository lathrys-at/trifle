//! Drift detection: the cache is dropped (not migrated) on a version/tokenizer
//! mismatch, and warm when nothing changed.

mod common;
use common::*;
use std::path::Path;
use trifle::store::Sidecar;
use trifle::tokenize::{DefaultTokenizer, Normalization};
use trifle::{Config, Document, Index, Schema, SearchOpts};

fn open_default(path: &Path, data_version: u64) -> Index<DefaultTokenizer, Sidecar> {
    let backend = Sidecar::open(path).unwrap();
    Index::open(
        backend,
        DefaultTokenizer::new(),
        Schema::flat(),
        Config::new(data_version),
    )
    .unwrap()
}

/// Search a raw index handle (limit 5) via a reader lease.
fn finds(idx: &Index<DefaultTokenizer, Sidecar>, query: &str) -> bool {
    let hits = idx
        .reader()
        .unwrap()
        .search(query, SearchOpts::new(5))
        .unwrap();
    hits.iter().any(|m| m.key.as_i64() == Some(1))
}

fn is_empty(idx: &Index<DefaultTokenizer, Sidecar>, query: &str) -> bool {
    idx.reader()
        .unwrap()
        .search(query, SearchOpts::new(5))
        .unwrap()
        .is_empty()
}

fn documents() -> impl Iterator<Item = Document> {
    FIXTURE
        .iter()
        .map(|(doc, text)| Document::new(*doc, vec![("body".to_string(), (*text).to_string())]))
}

#[test]
fn reopen_with_same_versions_is_warm() {
    let h = Harness::with_config(Config::new(7));
    load_fixture(&h);
    let path = h.db_path();
    drop(h.index);
    let idx = open_default(&path, 7);
    assert!(finds(&idx, "quick brown fox"));
    drop(h.dir);
}

#[test]
fn bumping_data_version_empties_the_cache() {
    let h = Harness::with_config(Config::new(1));
    load_fixture(&h);
    let path = h.db_path();
    drop(h.index);

    let bumped = open_default(&path, 2);
    assert!(
        is_empty(&bumped, "quick brown fox"),
        "a data_version bump drops the cache to empty"
    );
    assert_eq!(bumped.stats().unwrap().segments, 0);

    // The caller repopulates via rebuild; afterwards a same-version reopen is warm.
    bumped.rebuild(documents()).unwrap();
    assert!(finds(&bumped, "quick brown fox"));
    drop(bumped);
    let warm = open_default(&path, 2);
    assert!(finds(&warm, "quick brown fox"));
    drop(h.dir);
}

#[test]
fn changing_the_tokenizer_empties_the_cache() {
    let h = Harness::new();
    load_fixture(&h);
    let path = h.db_path();
    drop(h.index);

    // Reopen with a behaviorally-different tokenizer (different fingerprint).
    let backend = Sidecar::open(&path).unwrap();
    let tok = DefaultTokenizer::builder()
        .normalization(Normalization::NfdStripMarks)
        .build();
    let idx = Index::open(backend, tok, Schema::flat(), Config::default()).unwrap();
    assert!(
        is_empty(&idx, "quick brown fox"),
        "a tokenizer change drops the cache (postings are keyed by the old tokenizer)"
    );
    drop(h.dir);
}

#[test]
fn schema_version_is_stamped_and_observable() {
    let h = Harness::new();
    let s = h.index.stats().unwrap();
    assert_eq!(s.schema_version, 2);
    assert_eq!(s.data_version, 0);
}

#[test]
fn changing_the_schema_empties_the_cache() {
    // A schema with the same tables but different *semantics* (a declared field instead
    // of the flat default) has a different fingerprint, so it drops the cache.
    let h = Harness::new();
    load_fixture(&h);
    let path = h.db_path();
    drop(h.index);

    let backend = Sidecar::open(&path).unwrap();
    let schema = Schema::chunked().text("body").build().unwrap();
    let idx = Index::open(backend, DefaultTokenizer::new(), schema, Config::default()).unwrap();
    assert!(
        is_empty(&idx, "quick brown fox"),
        "a schema-fingerprint change drops the cache"
    );
    drop(h.dir);
}

#[test]
fn stats_reflects_the_configured_data_version() {
    let h = Harness::with_config(Config::new(99));
    assert_eq!(h.index.stats().unwrap().data_version, 99);
}

#[test]
fn reverting_the_data_version_does_not_resurrect_data() {
    let h = Harness::with_config(Config::new(1));
    load_fixture(&h);
    let path = h.db_path();
    drop(h.index);

    // Bump to 2: cache emptied. Reopen at 1 again: the stamp is now 2, so 1 != 2 is
    // still drift — the data stays gone. No "undo by reverting the token".
    drop(open_default(&path, 2));
    let back = open_default(&path, 1);
    assert!(
        is_empty(&back, "quick brown fox"),
        "reverting the data_version cannot bring the dropped cache back"
    );
    drop(h.dir);
}

#[test]
fn a_partial_stamp_is_treated_as_drift() {
    let h = Harness::new();
    load_fixture(&h);
    let path = h.db_path();
    drop(h.index);

    // Simulate a crash mid-stamp: delete one of the version rows.
    let raw = trifle::rusqlite::Connection::open(&path).unwrap();
    raw.execute("DELETE FROM meta WHERE key = 'data_version'", [])
        .unwrap();
    drop(raw);

    let reopened = open_default(&path, 0);
    assert!(
        is_empty(&reopened, "quick brown fox"),
        "a half-written stamp must rebuild, never half-trust"
    );
    drop(h.dir);
}

#[test]
fn a_seg_id_desync_resets_on_open() {
    let h = Harness::new();
    h.put(1, "field", "f", "indexed before the corruption");
    let path = h.db_path();
    drop(h.index);

    // Break the monotonic-id invariant: push next_id at or below max(seg.id).
    let raw = trifle::rusqlite::Connection::open(&path).unwrap();
    raw.execute("UPDATE meta SET value = '1' WHERE key = 'next_id'", [])
        .unwrap();
    drop(raw);

    let reopened = open_default(&path, 0);
    assert_eq!(
        reopened.stats().unwrap().segments,
        0,
        "a seg<->next_id desync drops the cache at open"
    );
    drop(h.dir);
}
