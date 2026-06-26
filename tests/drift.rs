//! Drift detection: the cache is dropped (not migrated) on a version/tokenizer
//! mismatch, and warm when nothing changed.

mod common;
use common::*;
use std::path::Path;
use trifle::store::Sidecar;
use trifle::tokenize::{Normalization, TrigramTokenizer};
use trifle::{Config, Index, SearchOpts};

fn open_default(path: &Path, data_version: u64) -> Index<TrigramTokenizer, Sidecar> {
    let backend = Sidecar::open(path).unwrap();
    Index::open(backend, TrigramTokenizer::new(), Config::new(data_version)).unwrap()
}

fn segments() -> impl Iterator<Item = trifle::Segment> {
    FIXTURE
        .iter()
        .map(|(doc, text)| trifle::Segment::new(*doc, "field", "body", *text))
}

#[test]
fn reopen_with_same_versions_is_warm() {
    let h = Harness::with_config(Config::new(7));
    load_fixture(&h);
    let path = h.db_path();
    drop(h.index);
    let idx = open_default(&path, 7);
    assert!(hit(
        &idx.search("quick brown fox", SearchOpts::new(5)).unwrap(),
        1
    ));
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
        bumped
            .search("quick brown fox", SearchOpts::new(5))
            .unwrap()
            .is_empty(),
        "a data_version bump drops the cache to empty"
    );
    assert_eq!(bumped.stats().unwrap().segments, 0);

    // The caller repopulates via rebuild; afterwards a same-version reopen is warm.
    bumped.rebuild(segments()).unwrap();
    assert!(hit(
        &bumped
            .search("quick brown fox", SearchOpts::new(5))
            .unwrap(),
        1
    ));
    drop(bumped);
    let warm = open_default(&path, 2);
    assert!(hit(
        &warm.search("quick brown fox", SearchOpts::new(5)).unwrap(),
        1
    ));
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
    let tok = TrigramTokenizer::builder()
        .normalization(Normalization::NfdStripMarks)
        .build();
    let idx = Index::open(backend, tok, Config::default()).unwrap();
    assert!(
        idx.search("quick brown fox", SearchOpts::new(5))
            .unwrap()
            .is_empty(),
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
        back.search("quick brown fox", SearchOpts::new(5))
            .unwrap()
            .is_empty(),
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

    // Simulate a crash mid-stamp: delete one of the three version rows.
    let raw = trifle::rusqlite::Connection::open(&path).unwrap();
    raw.execute("DELETE FROM meta WHERE key = 'data_version'", [])
        .unwrap();
    drop(raw);

    let reopened = open_default(&path, 0);
    assert!(
        reopened
            .search("quick brown fox", SearchOpts::new(5))
            .unwrap()
            .is_empty(),
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
