//! Hostile inputs and concurrency. trifle must never panic or corrupt on adversarial
//! queries, and must serve concurrent reads — including across a rebuild swap.

mod common;
use common::*;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;

use trifle::store::Sidecar;
use trifle::tokenize::DefaultTokenizer;
use trifle::{Config, Document, Index, Schema, SearchOpts};

/// Insert one `(label, text)` segment under `doc`, committed.
fn put_one(idx: &Index<DefaultTokenizer, Sidecar>, doc: i64, label: &str, text: &str) {
    let mut w = idx.writer().unwrap();
    w.insert(doc, &[(label, text)]).unwrap();
    w.commit().unwrap();
}

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
        let r = h.search(q, SearchOpts::new(10));
        assert!(r.is_ok(), "query {q:?} errored: {:?}", r.err());
    }
    // Pathological lengths: a wall of emoji and a 12k-char query. Selection caps the kept
    // tokens, so the work stays bounded.
    assert!(h.search(&"🚀".repeat(500), SearchOpts::new(10)).is_ok());
    assert!(
        h.search(&"lorem ipsum ".repeat(1000), SearchOpts::new(10))
            .is_ok()
    );
}

#[test]
fn an_injection_query_cannot_alter_the_store() {
    let h = Harness::new();
    h.put(1, "field", "f", "survivor document content");
    let _ = h.search(
        "'; DROP TABLE seg; DELETE FROM term; --",
        SearchOpts::new(10),
    );
    // The store is intact and still answers.
    assert!(hit(
        &h.search("survivor document", SearchOpts::new(10)).unwrap(),
        1
    ));
    assert_eq!(h.index.stats().unwrap().segments, 1);
}

#[test]
fn hostile_text_is_indexed_and_searchable_verbatim() {
    let h = Harness::new();
    // A label and text with quotes/semicolons must round-trip untouched.
    h.put(
        1,
        "field's",
        "a\"b;c",
        "value with 'quotes' and ; semicolons",
    );
    let m = &h.search("quotes semicolons", SearchOpts::new(5)).unwrap()[0];
    assert_eq!(m.label, "a\"b;c");
}

/// Open an `Arc`-shareable index without the harness (so the temp dir outlives the threads
/// via the returned guard).
fn shared_index() -> (Arc<Index<DefaultTokenizer, Sidecar>>, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let backend = Sidecar::open(dir.path().join("t.db")).unwrap();
    let idx = Index::open(
        backend,
        DefaultTokenizer::new(),
        Schema::flat(),
        Config::default(),
    )
    .unwrap();
    (Arc::new(idx), dir)
}

#[test]
fn concurrent_reads_run_alongside_writes() {
    let (idx, _dir) = shared_index();
    for doc in 1..=20 {
        put_one(&idx, doc, "f", "quick brown fox shared phrase content");
    }
    let stop = Arc::new(AtomicBool::new(false));
    let readers: Vec<_> = (0..6)
        .map(|_| {
            let idx = Arc::clone(&idx);
            let stop = Arc::clone(&stop);
            thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    let hits = idx
                        .reader()
                        .unwrap()
                        .search("quick brown fox", SearchOpts::new(10))
                        .unwrap();
                    assert!(!hits.is_empty());
                    // No phantom: every returned doc is one actually inserted (1..=120), so
                    // concurrent writes never corrupt a read into a bogus key.
                    assert!(
                        hits.iter()
                            .all(|m| (1..=120).contains(&m.key.as_i64().unwrap()))
                    );
                }
            })
        })
        .collect();
    // Writer churns while the readers run.
    for doc in 21..=120 {
        put_one(&idx, doc, "f", "quick brown fox shared phrase content");
    }
    stop.store(true, Ordering::Relaxed);
    for r in readers {
        r.join().unwrap();
    }
}

#[test]
fn reads_continue_across_a_rebuild_swap() {
    let (idx, _dir) = shared_index();
    // A token present in BOTH the old and the new corpus, so a correct read always finds it
    // (complete-old or complete-new, never partial).
    for doc in 1..=10 {
        put_one(
            &idx,
            doc,
            "f",
            "persistent token across the rebuild boundary",
        );
    }
    let stop = Arc::new(AtomicBool::new(false));
    let readers: Vec<_> = (0..4)
        .map(|_| {
            let idx = Arc::clone(&idx);
            let stop = Arc::clone(&stop);
            thread::spawn(move || {
                let mut seen = 0u64;
                while !stop.load(Ordering::Relaxed) {
                    // A read racing the swap may surface a retryable Error::Busy (a dictionary
                    // generation skew) — the library does NOT sleep to retry internally; the
                    // caller does (search_retrying, on a fresh reader). Because the token is in
                    // BOTH corpora, every consistent read finds it — the swap is complete-old or
                    // complete-new, never partial.
                    let hits = search_retrying(&idx, "persistent token", 10);
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
        let corpus: Vec<Document> = (1..=10)
            .map(|doc| {
                Document::new(
                    doc,
                    vec![(
                        "body".to_string(),
                        "persistent token across the rebuild boundary".to_string(),
                    )],
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

// T6 / OD1 (no library sleeps): a search racing a concurrent rebuild's dictionary reload may
// hit a generation skew. The library must NOT sleep/spin to retry internally — it surfaces a
// retryable Error::Busy and the caller backs off. This probe asserts the contract under churn:
// every search either (a) succeeds with the always-present token, or (b) returns the retryable
// Error::Busy — NEVER a non-retryable error and NEVER a wrong/empty result; and a caller-side
// retry (search_retrying) always makes progress.
//
// #[ignore]: rebuild churn is slow; run on demand.
#[test]
#[ignore = "slow: rebuild churn over a larger vocabulary"]
fn concurrent_search_under_rebuild_churn_surfaces_only_retryable_busy() {
    let (idx, _dir) = shared_index();
    // A larger vocabulary: a per-doc unique gram across 2k docs, plus a token present in every
    // generation so a correct read always finds it (complete-old or complete-new, never partial).
    let corpus = || -> Vec<Document> {
        (0..2000)
            .map(|d| {
                Document::new(
                    d,
                    vec![(
                        "body".to_string(),
                        format!("persistent token document number {d} uniquegram{d} filler text"),
                    )],
                )
            })
            .collect()
    };
    idx.rebuild(corpus()).unwrap();

    let stop = Arc::new(AtomicBool::new(false));
    let busy = Arc::new(AtomicU64::new(0)); // observed for visibility; Busy is allowed, not a failure
    let readers: Vec<_> = (0..4)
        .map(|_| {
            let idx = Arc::clone(&idx);
            let stop = Arc::clone(&stop);
            let busy = Arc::clone(&busy);
            thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    // Raw single attempt, classified to prove the contract.
                    match idx
                        .reader()
                        .and_then(|r| r.search("persistent token", SearchOpts::new(10)))
                    {
                        Ok(hits) => assert!(
                            !hits.is_empty(),
                            "the always-present token vanished mid-swap (wrong result, not Busy)"
                        ),
                        Err(trifle::Error::Busy(_)) => {
                            busy.fetch_add(1, Ordering::Relaxed);
                            // The caller owns backoff: a fresh-reader retry must succeed.
                            assert!(!search_retrying(&idx, "persistent token", 10).is_empty());
                        }
                        Err(e) => {
                            panic!("non-retryable error under rebuild churn (must be Busy): {e:?}")
                        }
                    }
                }
            })
        })
        .collect();
    for _ in 0..30 {
        idx.rebuild(corpus()).unwrap();
    }
    stop.store(true, Ordering::Relaxed);
    for r in readers {
        r.join().unwrap();
    }
    // Busy is permitted (the no-sleep contract); we only require the run completed with no
    // non-retryable error and no wrong result. Report the count for visibility.
    eprintln!(
        "rebuild-churn probe: {} retryable Busy events (all recovered)",
        busy.load(Ordering::Relaxed)
    );
}
