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
    assert_eq!(s.schema_version, 1);
    assert_eq!(s.data_version, 0);
}
