//! The `Shared` backend, namespacing several indexes into one file, and the
//! contentless text-resolver mode.

mod common;
use common::*;

use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

use trifle::rusqlite::{Connection, OpenFlags};
use trifle::store::{Namespace, Shared, TextResolver};
use trifle::tokenize::TrigramTokenizer;
use trifle::{Config, Index, Result, SearchOpts};

/// Open a caller-owned WAL database and wrap it in a `Shared` backend under the
/// given namespace.
fn shared(path: &Path, ns: Namespace) -> Index<TrigramTokenizer, Shared> {
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
    Index::open(backend, TrigramTokenizer::new(), Config::default()).unwrap()
}

#[test]
fn shared_backend_round_trips() {
    let dir = tempfile::tempdir().unwrap();
    let idx = shared(&dir.path().join("host.db"), Namespace::default());
    idx.insert(1, "field", &[("f", "embedded in the host database")])
        .unwrap();
    let hits = idx.search("embedded host", SearchOpts::new(5)).unwrap();
    assert!(hit(&hits, 1));
}

#[test]
fn two_namespaced_indexes_share_one_file_independently() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("host.db");
    let a = shared(&path, Namespace::prefixed("trifle_a_").unwrap());
    let b = shared(&path, Namespace::prefixed("trifle_b_").unwrap());
    // Disjoint vocabularies, so a cross-index hit can only come from a leak.
    a.insert(1, "field", &[("f", "apple apricot avocado almond")])
        .unwrap();
    b.insert(
        1,
        "field",
        &[("f", "banana blueberry blackberry boysenberry")],
    )
    .unwrap();
    // Each index sees only its own data.
    assert!(hit(
        &a.search("apple apricot", SearchOpts::new(5)).unwrap(),
        1
    ));
    assert!(
        a.search("banana blueberry", SearchOpts::new(5))
            .unwrap()
            .is_empty()
    );
    assert!(hit(
        &b.search("banana blueberry", SearchOpts::new(5)).unwrap(),
        1
    ));
    assert!(
        b.search("apple apricot", SearchOpts::new(5))
            .unwrap()
            .is_empty()
    );
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

// ----- contentless mode -------------------------------------------------------

/// A resolver that returns a deterministic, *distinct* string per doc — so a hit's
/// `text` proves hydration went through the resolver, not a stored snapshot.
struct TagResolver {
    /// Docs whose text is "unavailable" (resolver returns `None`).
    missing: Mutex<HashMap<i64, ()>>,
}

impl TextResolver for TagResolver {
    fn resolve(&self, segs: &[(i64, &str, &str)]) -> Result<Vec<Option<String>>> {
        let missing = self.missing.lock().unwrap();
        Ok(segs
            .iter()
            .map(|(doc, source, ref_)| {
                if missing.contains_key(doc) {
                    None
                } else {
                    Some(format!("resolved[{doc}/{source}/{ref_}]"))
                }
            })
            .collect())
    }
}

fn contentless_index(resolver: TagResolver) -> (Index, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let config = Config::default().with_external_content(Box::new(resolver));
    let idx = Index::open_at(&dir.path().join("t.db"), config).unwrap();
    (idx, dir)
}

#[test]
fn contentless_hydrates_text_through_the_resolver() {
    let (idx, _dir) = contentless_index(TagResolver {
        missing: Mutex::new(HashMap::new()),
    });
    idx.insert(1, "field", &[("body", "findable distinctive token alpha")])
        .unwrap();
    let hits = idx
        .search("findable distinctive", SearchOpts::new(5))
        .unwrap();
    assert!(hit(&hits, 1));
    // The text is the resolver's output, not the indexed string (proves no snapshot).
    assert_eq!(hits[0].text.as_deref(), Some("resolved[1/field/body]"));
}

#[test]
fn contentless_match_survives_a_none_from_the_resolver() {
    let mut missing = HashMap::new();
    missing.insert(1, ());
    let (idx, _dir) = contentless_index(TagResolver {
        missing: Mutex::new(missing),
    });
    idx.insert(
        1,
        "field",
        &[("body", "still indexable even if unresolvable")],
    )
    .unwrap();
    let hits = idx.search("still indexable", SearchOpts::new(5)).unwrap();
    assert!(hit(&hits, 1), "match present even when text is unavailable");
    assert_eq!(hits[0].text, None);
    assert_eq!(hits[0].span, None, "no text -> no span");
}

#[test]
fn contentless_delete_uses_the_stored_token_set_not_the_resolver() {
    // The resolver returns None for everything, so deletion cannot rely on it; the
    // stored `fwd` token set is what makes the delete correct.
    let mut missing = HashMap::new();
    missing.insert(7, ());
    let (idx, _dir) = contentless_index(TagResolver {
        missing: Mutex::new(missing),
    });
    idx.insert(
        7,
        "field",
        &[("body", "ephemeral transient disposable content")],
    )
    .unwrap();
    idx.remove(7).unwrap();
    assert!(
        idx.search("ephemeral transient", SearchOpts::new(5))
            .unwrap()
            .is_empty()
    );
}

#[test]
fn contentless_upsert_replaces() {
    let (idx, _dir) = contentless_index(TagResolver {
        missing: Mutex::new(HashMap::new()),
    });
    idx.insert(1, "field", &[("body", "original superseded wording")])
        .unwrap();
    idx.insert(1, "field", &[("body", "replacement updated phrasing")])
        .unwrap();
    assert!(
        idx.search("original superseded", SearchOpts::new(5))
            .unwrap()
            .is_empty()
    );
    assert!(hit(
        &idx.search("replacement updated", SearchOpts::new(5))
            .unwrap(),
        1
    ));
}

#[test]
fn contentless_handles_an_empty_token_set() {
    // A sub-trigram segment ("hi") yields no tokens; fwd stores a zero-token set.
    // Insert and remove must not panic and must touch no postings.
    let (idx, _dir) = contentless_index(TagResolver {
        missing: Mutex::new(HashMap::new()),
    });
    idx.insert(1, "field", &[("r", "hi")]).unwrap();
    assert_eq!(idx.stats().unwrap().segments, 1);
    idx.remove(1).unwrap(); // reads the empty fwd token set
    assert_eq!(idx.stats().unwrap().segments, 0);
}

#[test]
fn contentless_round_trips_weird_byte_tokens() {
    // Emoji/multibyte trigrams must encode into fwd and decode back on delete.
    let (idx, _dir) = contentless_index(TagResolver {
        missing: Mutex::new(HashMap::new()),
    });
    idx.insert(1, "field", &[("r", "🚀🎉😀 distinctive payload")])
        .unwrap();
    assert!(hit(
        &idx.search("distinctive payload", SearchOpts::new(5))
            .unwrap(),
        1
    ));
    idx.remove(1).unwrap(); // decodes the emoji-trigram fwd blob — must not panic
    assert!(
        idx.search("distinctive payload", SearchOpts::new(5))
            .unwrap()
            .is_empty()
    );
}

#[test]
fn contentless_corrupt_fwd_blob_surfaces_an_error_not_a_panic() {
    let (idx, dir) = contentless_index(TagResolver {
        missing: Mutex::new(HashMap::new()),
    });
    idx.insert(7, "field", &[("r", "soon to be corrupted")])
        .unwrap();
    // Corrupt the stored term-id set via a separate connection (bytes that are not a
    // valid roaring bitmap — the `fwd` blob is now a roaring posting of term-ids).
    let raw = Connection::open(dir.path().join("t.db")).unwrap();
    raw.execute(
        "UPDATE fwd SET tokens = ?1",
        [vec![0xFFu8, 0xFF, 0xFF, 0xFF]],
    )
    .unwrap();
    drop(raw);
    // remove() must consult fwd and surface the corruption, not panic. A corrupt
    // roaring blob surfaces as the posting-codec error.
    assert!(matches!(idx.remove(7), Err(trifle::Error::Posting(_))));
}

#[test]
fn shared_snapshot_returns_the_stored_text() {
    // Shared mode with no resolver stores a snapshot; Match.text comes from it.
    let dir = tempfile::tempdir().unwrap();
    let idx = shared(&dir.path().join("host.db"), Namespace::default());
    idx.insert(1, "field", &[("r", "snapshot text in the host file")])
        .unwrap();
    let hits = idx.search("snapshot text", SearchOpts::new(5)).unwrap();
    assert_eq!(
        hits[0].text.as_deref(),
        Some("snapshot text in the host file")
    );
}
