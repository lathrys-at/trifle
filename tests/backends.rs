//! The `Shared` backend, namespacing several indexes into one file, and the
//! stored-text / fwd-based-delete paths (every indexed field is stored in v0.2).

mod common;
use common::*;

use std::path::Path;

use trifle::rusqlite::{Connection, OpenFlags};
use trifle::store::{Backend, Namespace, Shared};
use trifle::tokenize::{DefaultTokenizer, Tokenizer};
use trifle::{Config, Index, Match, Schema, SearchOpts};

// ----- lease helpers (these tests hold raw `Index` handles, not the Harness) ---

/// Insert one `(label, text)` segment under `key`, committed.
fn insert<T: Tokenizer, B: Backend>(idx: &Index<T, B>, key: i64, label: &str, text: &str) {
    let mut w = idx.writer().unwrap();
    w.insert(key, &[(label, text)]).unwrap();
    w.commit().unwrap();
}

/// Search via a fresh reader lease.
fn search<T: Tokenizer, B: Backend>(idx: &Index<T, B>, q: &str, limit: usize) -> Vec<Match> {
    idx.reader()
        .unwrap()
        .search(q, SearchOpts::new(limit))
        .unwrap()
}

/// Remove a whole document, committed.
fn remove<T: Tokenizer, B: Backend>(idx: &Index<T, B>, key: i64) {
    let mut w = idx.writer().unwrap();
    w.remove(key).unwrap();
    w.commit().unwrap();
}

/// Open a caller-owned WAL database and wrap it in a `Shared` backend under the given
/// namespace (flat schema, default config).
fn shared(path: &Path, ns: Namespace) -> Index<DefaultTokenizer, Shared> {
    let write = Connection::open(path).unwrap();
    write.pragma_update(None, "journal_mode", "WAL").unwrap();
    let read_path = path.to_path_buf();
    let backend = Shared::new(ns, write, move || {
        Connection::open_with_flags(
            &read_path,
            OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
    })
    .unwrap();
    Index::open(
        backend,
        DefaultTokenizer::new(),
        Schema::flat(),
        Config::default(),
    )
    .unwrap()
}

#[test]
fn shared_backend_round_trips() {
    let dir = tempfile::tempdir().unwrap();
    let idx = shared(&dir.path().join("host.db"), Namespace::default());
    insert(&idx, 1, "f", "embedded in the host database");
    assert!(hit(&search(&idx, "embedded host", 5), 1));
}

#[test]
fn two_namespaced_indexes_share_one_file_independently() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("host.db");
    let a = shared(&path, Namespace::prefixed("trifle_a_").unwrap());
    let b = shared(&path, Namespace::prefixed("trifle_b_").unwrap());
    // Disjoint vocabularies, so a cross-index hit can only come from a leak.
    insert(&a, 1, "f", "apple apricot avocado almond");
    insert(&b, 1, "f", "banana blueberry blackberry boysenberry");
    // Each index sees only its own data.
    assert!(hit(&search(&a, "apple apricot", 5), 1));
    assert!(search(&a, "banana blueberry", 5).is_empty());
    assert!(hit(&search(&b, "banana blueberry", 5), 1));
    assert!(search(&b, "apple apricot", 5).is_empty());
}

#[test]
fn namespaces_do_not_collide_on_table_names() {
    let a: Vec<String> = Namespace::prefixed("trifle_a_")
        .unwrap()
        .table_names()
        .map(str::to_string)
        .collect();
    let b: Vec<String> = Namespace::prefixed("trifle_b_")
        .unwrap()
        .table_names()
        .map(str::to_string)
        .collect();
    for name in &a {
        assert!(!b.contains(name), "{name} appears in both namespaces");
    }
}

// ----- stored text + fwd-based delete -----------------------------------------

/// A plain stored Sidecar index (flat schema) for the delete / fwd / corruption paths.
fn stored_index() -> (Index, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let idx = Index::open_at(&dir.path().join("t.db"), Schema::flat(), Config::default()).unwrap();
    (idx, dir)
}

#[test]
fn delete_uses_the_stored_token_set() {
    // Delete reads the stored `fwd` term-id set (never the text/tokenizer), so the segment
    // becomes unfindable.
    let (idx, _dir) = stored_index();
    insert(&idx, 7, "body", "ephemeral transient disposable content");
    remove(&idx, 7);
    assert!(search(&idx, "ephemeral transient", 5).is_empty());
}

#[test]
fn handles_an_empty_token_set() {
    // A sub-trigram segment ("hi") yields no tokens; fwd stores a zero-token set. Insert
    // and remove must not panic and must touch no postings.
    let (idx, _dir) = stored_index();
    insert(&idx, 1, "r", "hi");
    assert_eq!(idx.stats().unwrap().segments, 1);
    remove(&idx, 1); // reads the empty fwd token set
    assert_eq!(idx.stats().unwrap().segments, 0);
}

#[test]
fn round_trips_weird_byte_tokens() {
    // Emoji/multibyte grams must intern into fwd (as term-ids) and survive delete.
    let (idx, _dir) = stored_index();
    insert(&idx, 1, "r", "🚀🎉😀 distinctive payload");
    assert!(hit(&search(&idx, "distinctive payload", 5), 1));
    remove(&idx, 1); // reads the emoji-gram fwd term-ids — must not panic
    assert!(search(&idx, "distinctive payload", 5).is_empty());
}

#[test]
fn corrupt_fwd_blob_surfaces_an_error_not_a_panic() {
    let (idx, dir) = stored_index();
    insert(&idx, 7, "r", "soon to be corrupted");
    // Corrupt the stored term-id set via a separate connection (bytes that are not a valid
    // roaring bitmap — the `fwd` blob is a roaring posting of term-ids).
    let raw = Connection::open(dir.path().join("t.db")).unwrap();
    raw.execute(
        "UPDATE fwd SET tokens = ?1",
        [vec![0xFFu8, 0xFF, 0xFF, 0xFF]],
    )
    .unwrap();
    drop(raw);
    // remove() must consult fwd and surface the corruption, not panic. A corrupt roaring
    // blob surfaces as the posting-codec error. The writer drops (rolls back) on error.
    let mut w = idx.writer().unwrap();
    assert!(matches!(w.remove(7), Err(trifle::Error::Posting(_))));
}

#[test]
fn shared_snapshot_returns_the_stored_text() {
    // Shared mode stores a snapshot; Match.text comes from it.
    let dir = tempfile::tempdir().unwrap();
    let idx = shared(&dir.path().join("host.db"), Namespace::default());
    insert(&idx, 1, "r", "snapshot text in the host file");
    let hits = search(&idx, "snapshot text", 5);
    assert_eq!(hits[0].text, "snapshot text in the host file");
}
