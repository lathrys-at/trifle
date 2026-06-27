//! Shared integration-test harness: tiny fixtures, opened against a real temp
//! sidecar file. Kept minimal on purpose — a handful of rows is enough to assert
//! an invariant; nothing here needs a large corpus.
#![allow(dead_code)]

use tempfile::TempDir;
use trifle::store::Sidecar;
use trifle::tokenize::DefaultTokenizer;
use trifle::{Config, Index, Match, Result, Schema, SearchOpts};

/// A temp sidecar index plus the directory that backs it (kept alive alongside the
/// index — dropping it deletes the files).
pub struct Harness {
    pub index: Index<DefaultTokenizer, Sidecar>,
    pub dir: TempDir,
}

impl Harness {
    /// A fresh empty index in a fresh temp directory (flat schema, default config).
    pub fn new() -> Harness {
        Harness::with_config(Config::default())
    }

    pub fn with_config(config: Config) -> Harness {
        Harness::with_schema(Schema::flat(), config)
    }

    /// A fresh index with a caller-supplied schema.
    pub fn with_schema(schema: Schema, config: Config) -> Harness {
        let dir = tempfile::tempdir().unwrap();
        let index = Index::open_at(&dir.path().join("trifle.db"), schema, config).unwrap();
        Harness { index, dir }
    }

    /// The backing database path (for reopening the same file).
    pub fn db_path(&self) -> std::path::PathBuf {
        self.dir.path().join("trifle.db")
    }

    /// Search via a fresh reader lease (convenience for the search-only tests).
    pub fn search(&self, query: &str, opts: SearchOpts) -> Result<Vec<Match>> {
        self.index.reader()?.search(query, opts)
    }

    /// Batch-search via a fresh reader lease.
    pub fn search_batch(&self, queries: &[&str], opts: SearchOpts) -> Result<Vec<Vec<Match>>> {
        self.index.reader()?.search_batch(queries, opts)
    }

    /// Insert a single segment, keyed by `doc_id` under label `ref_`. The v0.1 `source`
    /// (the per-(doc,source) replace group) has no v0.2 analogue, so it is ignored;
    /// callers keep their 4-argument call sites. Uses `upsert`, so a repeated
    /// `(doc, label)` replaces rather than erroring.
    pub fn put(&self, doc_id: i64, _source: &str, ref_: &str, text: &str) {
        let mut w = self.index.writer().unwrap();
        w.upsert(doc_id, &[(ref_, text)]).unwrap();
        w.commit().unwrap();
    }

    /// Insert one doc with multiple `(label, text)` segments in one committed batch
    /// (errors if any `(doc, label)` already exists).
    pub fn insert(&self, doc_id: i64, segments: &[(&str, &str)]) {
        let mut w = self.index.writer().unwrap();
        w.insert(doc_id, segments).unwrap();
        w.commit().unwrap();
    }

    /// Remove a whole document (all its labels) in one committed write.
    pub fn remove(&self, doc_id: i64) {
        let mut w = self.index.writer().unwrap();
        w.remove(doc_id).unwrap();
        w.commit().unwrap();
    }

    /// Remove a single `(doc, label)` segment in one committed write.
    pub fn remove_segment(&self, doc_id: i64, label: &str) {
        let mut w = self.index.writer().unwrap();
        w.remove_segment(doc_id, label).unwrap();
        w.commit().unwrap();
    }
}

impl Default for Harness {
    fn default() -> Self {
        Harness::new()
    }
}

/// Caller-side retry for the library's no-sleeps contract: a search racing a concurrent
/// `rebuild` can surface a retryable [`trifle::Error::Busy`] (a dictionary-generation skew)
/// instead of the library blocking the caller's thread to retry. The application owns the
/// backoff — so the test backs off briefly and retries on a **fresh** reader. A sleep in a
/// *test* is fine; only library code must never sleep. Panics on any non-retryable error or if
/// `Busy` never clears.
pub fn search_retrying(
    index: &Index<DefaultTokenizer, Sidecar>,
    query: &str,
    limit: usize,
) -> Vec<Match> {
    for _ in 0..2000 {
        match index
            .reader()
            .and_then(|r| r.search(query, SearchOpts::new(limit)))
        {
            Ok(hits) => return hits,
            Err(trifle::Error::Busy(_)) => {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            Err(e) => panic!("unexpected non-retryable error from search: {e:?}"),
        }
    }
    panic!("search never cleared Error::Busy within the caller retry budget");
}

/// The matched doc keys (as integers), in result order.
pub fn ids(matches: &[Match]) -> Vec<i64> {
    matches.iter().map(|m| m.key.as_i64().unwrap()).collect()
}

/// Whether any match has the given doc key.
pub fn hit(matches: &[Match], doc_id: i64) -> bool {
    matches.iter().any(|m| m.key.as_i64() == Some(doc_id))
}

/// A short, real-English fixture corpus (small docs — trifle's regime). Indexed
/// under label `"body"`.
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

/// Load [`FIXTURE`] into an index in one writer batch.
pub fn load_fixture(h: &Harness) {
    let mut w = h.index.writer().unwrap();
    for (doc_id, text) in FIXTURE {
        w.upsert(*doc_id, &[("body", *text)]).unwrap();
    }
    w.commit().unwrap();
}

/// [`FIXTURE`] as `Document`s (one `"body"` segment each) for [`Index::rebuild`].
pub fn fixture_docs() -> impl Iterator<Item = trifle::Document> {
    FIXTURE.iter().map(|(doc, text)| {
        trifle::Document::new(*doc, vec![("body".to_string(), (*text).to_string())])
    })
}

/// Apply a single-character substitution at byte `pos` (ASCII fixtures only).
pub fn substitute(s: &str, pos: usize, c: char) -> String {
    let mut chars: Vec<char> = s.chars().collect();
    chars[pos] = c;
    chars.into_iter().collect()
}
