//! Drift detection: the cache is dropped (not migrated) on a version/tokenizer/schema mismatch,
//! and warm when nothing changed.

mod common;
use common::*;
use std::path::Path;
use trifle::store::Sidecar;
use trifle::tokenize::{DefaultTokenizer, Normalization};
use trifle::{Config, Index, Schema, SearchOpts};

fn open_default(path: &Path, data_version: u64) -> Index<DefaultTokenizer> {
    let store = Sidecar::open(path).unwrap();
    Index::open(
        store,
        DefaultTokenizer::new(),
        Schema::flat(),
        Config::new(data_version),
    )
    .unwrap()
}

fn finds(idx: &Index<DefaultTokenizer>, query: &str) -> bool {
    let hits = idx
        .reader()
        .unwrap()
        .matches(query, &SearchOpts::new(), 5)
        .unwrap();
    hits.iter().any(|m| m.key.as_i64() == Some(1))
}

fn is_empty(idx: &Index<DefaultTokenizer>, query: &str) -> bool {
    idx.reader()
        .unwrap()
        .matches(query, &SearchOpts::new(), 5)
        .unwrap()
        .is_empty()
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

    bumped.rebuild(fixture_docs()).unwrap();
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

    let store = Sidecar::open(&path).unwrap();
    let tok = DefaultTokenizer::builder()
        .normalization(Normalization::NfdStripMarks)
        .build();
    let idx = Index::open(store, tok, Schema::flat(), Config::default()).unwrap();
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
    assert_eq!(s.schema_version, 4);
    assert_eq!(s.data_version, 0);
}

#[test]
fn changing_the_schema_empties_the_cache() {
    // The flat default vs a declared field: same tables, different *semantics* → different
    // fingerprint → drops the cache.
    let h = Harness::new();
    load_fixture(&h);
    let path = h.db_path();
    drop(h.index);

    let store = Sidecar::open(&path).unwrap();
    let schema = Schema::chunked().text("body").build().unwrap();
    let idx = Index::open(store, DefaultTokenizer::new(), schema, Config::default()).unwrap();
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
    h.put(1, "f", "indexed before the corruption");
    let path = h.db_path();
    drop(h.index);

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
