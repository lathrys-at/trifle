//! Drift detection: the cache is dropped (not migrated) on a version/tokenizer/schema mismatch,
//! and warm when nothing changed.

mod common;
use common::*;
use std::path::Path;
use trifle::store::Sidecar;
use trifle::tokenize::{DefaultTokenizer, Normalization, Tokenizer};
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

/// The distinct gram count of `text` under the live default tokenizer (the derivation §0/§6
/// `L_d`, the same quantity `seg.len` stores).
fn distinct_gram_count(text: &str) -> i64 {
    use std::collections::HashSet;
    DefaultTokenizer::new()
        .tokenize(text)
        .collect::<HashSet<_>>()
        .len() as i64
}

/// Read a meta counter as an `i64` straight from the file (independent of the code under test).
fn raw_meta_i64(path: &Path, key: &str) -> i64 {
    let raw = trifle::rusqlite::Connection::open(path).unwrap();
    raw.query_row("SELECT value FROM meta WHERE key = ?1", [key], |r| {
        r.get::<_, String>(0)
    })
    .unwrap()
    .parse()
    .unwrap()
}

/// Read a segment's stored `len` column straight from the file.
fn raw_seg_len(path: &Path, key: i64) -> i64 {
    let raw = trifle::rusqlite::Connection::open(path).unwrap();
    raw.query_row("SELECT len FROM seg WHERE key = ?1", [key], |r| r.get(0))
        .unwrap()
}

/// `avgdl` on the index's current snapshot, via a throwaway candidate stream (the only public
/// surface that exposes it). The query is irrelevant — `avgdl` is corpus-wide.
fn avgdl_of(idx: &Index<DefaultTokenizer>) -> f64 {
    let reader = idx.reader().unwrap();
    let opts = SearchOpts::new();
    let stream = reader.candidates("zzz", &opts).unwrap();
    stream.avgdl()
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
fn a_stale_tokenizer_fingerprint_resets_the_cache() {
    // v0.4/M4 bumped the DefaultTokenizer fingerprint (whitespace now breaks gram windows), so a
    // pre-M4 cache must drop+rebuild on open — never migrate. Forge a stale stored fingerprint and
    // confirm the reset path (the same mechanism the layout-byte bump triggers in production).
    let h = Harness::new();
    load_fixture(&h);
    let path = h.db_path();
    drop(h.index);

    let raw = trifle::rusqlite::Connection::open(&path).unwrap();
    raw.execute(
        "UPDATE meta SET value = '424242' WHERE key = 'tokenizer_fingerprint'",
        [],
    )
    .unwrap();
    drop(raw);

    let reopened = open_default(&path, 0);
    assert!(
        is_empty(&reopened, "quick brown fox"),
        "a tokenizer-fingerprint mismatch drops the cache (reset, never migrate)"
    );
    assert_eq!(reopened.stats().unwrap().segments, 0);
    // The reset re-stamps the live fingerprint, so a second reopen is warm after a rebuild.
    reopened.rebuild(fixture_docs()).unwrap();
    assert!(finds(&reopened, "quick brown fox"));
    drop(open_default(&path, 0));
    drop(h.dir);
}

#[test]
fn schema_version_is_stamped_and_observable() {
    let h = Harness::new();
    let s = h.index.stats().unwrap();
    assert_eq!(s.schema_version, 5);
    assert_eq!(s.data_version, 0);
}

#[test]
fn an_old_schema_version_stamp_resets_the_cache() {
    // trifle is a rebuildable cache: an on-disk `schema_version` from an earlier format must drop
    // (reset) the cache on open, never migrate it. (v0.4 bumped the version: `seg.len` now stores
    // the distinct gram count, so a pre-v0.4 cache must be rebuilt.)
    let h = Harness::new();
    load_fixture(&h);
    let path = h.db_path();
    drop(h.index);

    // Forge a stale on-disk schema version directly in the file.
    let raw = trifle::rusqlite::Connection::open(&path).unwrap();
    raw.execute(
        "UPDATE meta SET value = '4' WHERE key = 'schema_version'",
        [],
    )
    .unwrap();
    drop(raw);

    let reopened = open_default(&path, 0);
    assert!(
        is_empty(&reopened, "quick brown fox"),
        "a schema_version mismatch drops the cache (reset, never migrate)"
    );
    assert_eq!(reopened.stats().unwrap().segments, 0);
    // The reset re-stamps the current version.
    assert_eq!(reopened.stats().unwrap().schema_version, 5);
    drop(h.dir);
}

#[test]
fn stored_seg_len_is_the_distinct_gram_count() {
    // A text with repeated grams (so distinct ≠ with-repetition) locks the v0.4 `seg.len`
    // redefinition: the *incremental upsert* path must store `L_d`, the distinct gram count
    // (derivation §0/§6).
    let h = Harness::new();
    let text = "the the the quick brown fox";
    h.put(1, "body", text);
    let path = h.db_path();

    assert_eq!(
        raw_seg_len(&path, 1),
        distinct_gram_count(text),
        "seg.len must store the DISTINCT gram count L_d (§0/§6), not the with-repetition count"
    );
    drop(h.dir);
}

#[test]
fn rebuild_stores_distinct_seg_len_and_agrees_with_upsert() {
    // The `rebuild()` path is a separate code path from incremental upsert (and M5 rewrites it),
    // so pin it to the same definition: a rebuilt segment's stored `seg.len` is the distinct gram
    // count `L_d`, and the rolling `seg_len_sum` agrees with it. (The upsert path is pinned by
    // `stored_seg_len_is_the_distinct_gram_count`, so matching `L_d` here means the two agree.)
    let text = "the the the quick brown fox";
    let h = Harness::new();
    h.index
        .rebuild(std::iter::once(trifle::Document::new(
            1,
            vec![("body".to_string(), text.to_string())],
        )))
        .unwrap();
    let path = h.db_path();

    let distinct = distinct_gram_count(text);
    assert_eq!(
        raw_seg_len(&path, 1),
        distinct,
        "rebuild stores the distinct seg.len (L_d)"
    );
    assert_eq!(
        raw_meta_i64(&path, "seg_len_sum"),
        distinct,
        "seg_len_sum agrees with the single rebuilt segment's distinct gram count"
    );
    drop(h.dir);
}

#[test]
fn upsert_and_rebuild_agree_on_n_and_avgdl() {
    // The same corpus built two ways (incremental upsert vs `rebuild()`) must yield identical N
    // and avgdl — the highest-value guard against the two paths diverging on "which op ran last".
    // The 1-char segment produces zero grams and must still count toward N (its L_d is 0).
    let corpus: &[(i64, &str)] = &[
        (1, "the quick brown fox"),
        (2, "a"), // 1 char → no trigram → distinct gram count 0
        (3, "the the the lazy dog"),
    ];

    let inc = Harness::new();
    {
        let mut w = inc.index.writer().unwrap();
        for (k, t) in corpus {
            w.upsert(*k, &[("body", *t)]).unwrap();
        }
        w.commit().unwrap();
    }

    let reb = Harness::new();
    reb.index
        .rebuild(
            corpus.iter().map(|(k, t)| {
                trifle::Document::new(*k, vec![("body".to_string(), (*t).to_string())])
            }),
        )
        .unwrap();

    assert_eq!(
        inc.index.stats().unwrap().segments,
        3,
        "the zero-gram segment still counts toward N"
    );
    assert_eq!(
        inc.index.stats().unwrap().segments,
        reb.index.stats().unwrap().segments,
        "upsert and rebuild agree on N"
    );
    let a_inc = avgdl_of(&inc.index);
    let a_reb = avgdl_of(&reb.index);
    assert_eq!(a_inc, a_reb, "upsert and rebuild agree on avgdl");
    assert!(a_inc > 0.0, "avgdl is positive for a non-empty corpus");
    drop(inc.dir);
    drop(reb.dir);
}

#[test]
fn empty_corpus_avgdl_is_zero_no_div_by_zero() {
    let h = Harness::new();
    assert_eq!(h.index.stats().unwrap().segments, 0);
    assert_eq!(
        avgdl_of(&h.index),
        0.0,
        "an empty corpus reports avgdl 0.0 (the seg_count > 0 guard avoids a div-by-zero)"
    );
    drop(h.dir);
}

#[test]
fn drop_then_readd_restores_avgdl() {
    // remove(key) then re-upsert the identical text must restore avgdl exactly (the rolling
    // seg_len_sum / seg_count counters back out and re-add the same amounts).
    let h = Harness::new();
    load_fixture(&h);
    let before = avgdl_of(&h.index);

    h.remove(1);
    h.put(1, "body", FIXTURE[0].1);
    let after = avgdl_of(&h.index);

    assert_eq!(before, after, "remove + re-upsert restores avgdl exactly");
    drop(h.dir);
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
