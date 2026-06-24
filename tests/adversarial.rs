//! Hostile inputs and concurrency. trifle must never panic or corrupt on adversarial
//! queries, and must serve concurrent reads — including across a rebuild swap.

mod common;
use common::*;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

use trifle::store::Sidecar;
use trifle::tokenize::TrigramTokenizer;
use trifle::{Config, Index, SearchOpts, Segment};

/// Queries crafted to break a naive implementation (SQL injection, quoting, control
/// bytes, bidi/zero-width). None may panic or error.
const HOSTILE: &[&str] = &[
    "'; DROP TABLE seg; --",
    "\" OR \"1\"=\"1",
    "%_\\%",
    "robert'); DROP TABLE term;--",
    "\0\0\0 null bytes",
    "\u{202e}\u{200b} bidi and zero width",
    "((((((((((unbalanced",
    "SELECT * FROM sqlite_master",
    "\n\t\r whitespace soup \t\n",
];

#[test]
fn hostile_queries_never_panic_or_error() {
    let h = Harness::new();
    load_fixture(&h);
    for q in HOSTILE {
        let r = h.index.search(q, SearchOpts::new(10));
        assert!(r.is_ok(), "query {q:?} errored: {:?}", r.err());
    }
    // Pathological lengths: a wall of emoji and a 12k-char query. Selection caps the
    // kept tokens, so the work stays bounded.
    assert!(
        h.index
            .search(&"🚀".repeat(500), SearchOpts::new(10))
            .is_ok()
    );
    assert!(
        h.index
            .search(&"lorem ipsum ".repeat(1000), SearchOpts::new(10))
            .is_ok()
    );
}

#[test]
fn an_injection_query_cannot_alter_the_store() {
    let h = Harness::new();
    h.put(1, "field", "f", "survivor document content");
    let _ = h.index.search(
        "'; DROP TABLE seg; DELETE FROM term; --",
        SearchOpts::new(10),
    );
    // The store is intact and still answers.
    assert!(hit(
        &h.index
            .search("survivor document", SearchOpts::new(10))
            .unwrap(),
        1
    ));
    assert_eq!(h.index.stats().unwrap().segments, 1);
}

#[test]
fn hostile_text_is_indexed_and_searchable_verbatim() {
    let h = Harness::new();
    // Provenance and text with quotes/semicolons must round-trip untouched.
    h.put(
        1,
        "field's",
        "a\"b;c",
        "value with 'quotes' and ; semicolons",
    );
    let m = &h
        .index
        .search("quotes semicolons", SearchOpts::new(5))
        .unwrap()[0];
    assert_eq!(m.source, "field's");
    assert_eq!(m.ref_, "a\"b;c");
}

/// Open an `Arc`-shareable index without the harness (so the temp dir outlives the
/// threads via the returned guard).
fn shared_index() -> (Arc<Index<TrigramTokenizer, Sidecar>>, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let backend = Sidecar::open(dir.path().join("t.db")).unwrap();
    let idx = Index::open(backend, TrigramTokenizer::new(), Config::default()).unwrap();
    (Arc::new(idx), dir)
}

#[test]
fn concurrent_reads_run_alongside_writes() {
    let (idx, _dir) = shared_index();
    for doc in 1..=20 {
        idx.insert(
            doc,
            "field",
            &[("f", "quick brown fox shared phrase content")],
        )
        .unwrap();
    }
    let stop = Arc::new(AtomicBool::new(false));
    let readers: Vec<_> = (0..6)
        .map(|_| {
            let idx = Arc::clone(&idx);
            let stop = Arc::clone(&stop);
            thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    let hits = idx.search("quick brown fox", SearchOpts::new(10)).unwrap();
                    assert!(!hits.is_empty());
                    // No phantom: every returned doc is one actually inserted (1..=120),
                    // so concurrent writes never corrupt a read into a bogus doc id.
                    assert!(hits.iter().all(|m| (1..=120).contains(&m.doc_id)));
                }
            })
        })
        .collect();
    // Writer churns while the readers run.
    for doc in 21..=120 {
        idx.insert(
            doc,
            "field",
            &[("f", "quick brown fox shared phrase content")],
        )
        .unwrap();
    }
    stop.store(true, Ordering::Relaxed);
    for r in readers {
        r.join().unwrap();
    }
}

#[test]
fn reads_continue_across_a_rebuild_swap() {
    let (idx, _dir) = shared_index();
    // A token present in BOTH the old and the new corpus, so a correct read always
    // finds it (complete-old or complete-new, never partial).
    for doc in 1..=10 {
        idx.insert(
            doc,
            "field",
            &[("f", "persistent token across the rebuild boundary")],
        )
        .unwrap();
    }
    let stop = Arc::new(AtomicBool::new(false));
    let readers: Vec<_> = (0..4)
        .map(|_| {
            let idx = Arc::clone(&idx);
            let stop = Arc::clone(&stop);
            thread::spawn(move || {
                let mut seen = 0u64;
                while !stop.load(Ordering::Relaxed) {
                    // A read must never error across the swap: the rename keeps the same
                    // table names/shape so rusqlite re-prepares transparently, and
                    // read_retry absorbs any transient SQLITE_SCHEMA/BUSY. And because
                    // the token is in BOTH corpora, every consistent read finds it —
                    // the swap is complete-old or complete-new, never partial.
                    let hits = idx.search("persistent token", SearchOpts::new(10)).unwrap();
                    assert!(
                        !hits.is_empty(),
                        "the always-present token vanished mid-swap"
                    );
                    seen += 1;
                }
                seen
            })
        })
        .collect();
    // Rebuild many times to widen the window the readers race against the swap.
    for _ in 0..40 {
        let corpus: Vec<Segment> = (1..=10)
            .map(|doc| {
                Segment::new(
                    doc,
                    "field",
                    "body",
                    "persistent token across the rebuild boundary",
                )
            })
            .collect();
        idx.rebuild(corpus).unwrap();
    }
    stop.store(true, Ordering::Relaxed);
    for r in readers {
        assert!(r.join().unwrap() > 0);
    }
}
