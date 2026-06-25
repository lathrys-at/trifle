//! Shared integration-test harness: tiny fixtures, opened against a real temp
//! sidecar file. Kept minimal on purpose — a handful of rows is enough to assert
//! an invariant; nothing here needs a large corpus.
#![allow(dead_code)]

use tempfile::TempDir;
use trifle::store::Sidecar;
use trifle::tokenize::TrigramTokenizer;
use trifle::{Config, Index, Match};

/// A temp sidecar index plus the directory that backs it (kept alive alongside the
/// index — dropping it deletes the files).
pub struct Harness {
    pub index: Index<TrigramTokenizer, Sidecar>,
    pub dir: TempDir,
}

impl Harness {
    /// A fresh empty index in a fresh temp directory.
    pub fn new() -> Harness {
        Harness::with_config(Config::default())
    }

    pub fn with_config(config: Config) -> Harness {
        let dir = tempfile::tempdir().unwrap();
        let index = Index::open_at(&dir.path().join("trifle.db"), config).unwrap();
        Harness { index, dir }
    }

    /// The backing database path (for reopening the same file).
    pub fn db_path(&self) -> std::path::PathBuf {
        self.dir.path().join("trifle.db")
    }

    /// Insert a single `(doc, source, ref, text)` segment.
    pub fn put(&self, doc_id: i64, source: &str, ref_: &str, text: &str) {
        self.index.insert(doc_id, source, &[(ref_, text)]).unwrap();
    }
}

impl Default for Harness {
    fn default() -> Self {
        Harness::new()
    }
}

/// The matched doc ids, in result order.
pub fn ids(matches: &[Match]) -> Vec<i64> {
    matches.iter().map(|m| m.doc_id).collect()
}

/// Whether any match has the given doc id.
pub fn hit(matches: &[Match], doc_id: i64) -> bool {
    matches.iter().any(|m| m.doc_id == doc_id)
}

/// A short, real-English fixture corpus (small docs — trifle's regime). Indexed
/// under `("field", "text")`.
pub const FIXTURE: &[(i64, &str)] = &[
    (1, "the quick brown fox jumps over the lazy dog"),
    (2, "a quick brown hare leaps across the meadow"),
    (3, "lorem ipsum dolor sit amet consectetur"),
    (4, "the lazy dog sleeps in the warm afternoon sun"),
    (5, "pack my box with five dozen liquor jugs"),
    (6, "sphinx of black quartz judge my vow"),
    (7, "how vexingly quick daft zebras jump"),
    (8, "the five boxing wizards jump quickly"),
];

/// Load [`FIXTURE`] into an index.
pub fn load_fixture(h: &Harness) {
    for (doc_id, text) in FIXTURE {
        h.put(*doc_id, "field", "body", text);
    }
}

/// Apply a single-character substitution at byte `pos` (ASCII fixtures only).
pub fn substitute(s: &str, pos: usize, c: char) -> String {
    let mut chars: Vec<char> = s.chars().collect();
    chars[pos] = c;
    chars.into_iter().collect()
}
