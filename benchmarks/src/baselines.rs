//! The footrace field (§10.1): trifle vs in-process SQLite baselines on the *same*
//! corpus and queries, so the comparison isolates the matching strategy from the
//! store. trifle and both baselines link the same bundled SQLite.
//!
//! - **FTS5-trigram (BM25)** — the in-DB cousin and the quality baseline. A trigram
//!   FTS5 table with `ORDER BY rank`.
//! - **`LIKE '%…%'` scan** — the naive substring floor.
//!
//! External engines from the §10 matrix (pg_trgm, Tantivy+Levenshtein, fzf/nucleo,
//! fst/SymSpell) are out-of-process and live in a separate driver; this module is
//! the embedded subset that shares trifle's store.

use std::path::Path;

use rusqlite::Connection;
use trifle::store::Sidecar;
use trifle::tokenize::TrigramTokenizer;
use trifle::{Config, Index, SearchOpts};

use crate::corpus::Corpus;

/// A search engine under test: build once from a corpus, then answer queries.
///
/// `search` answers one query; `search_many` answers a set. The default
/// `search_many` loops `search`, so a per-query-stateless engine gets batching
/// for free; trifle overrides it to share posting/frequency reads across the set
/// (its `search_batch`), which is the whole point of the `--batched` axis.
pub trait Engine {
    fn name(&self) -> &'static str;
    /// Return the matched document ids, best-first, capped at `k`.
    fn search(&self, query: &str, k: usize) -> Vec<i64>;
    /// Answer a batch of queries, one id list per query in order.
    fn search_many(&self, queries: &[&str], k: usize) -> Vec<Vec<i64>> {
        queries.iter().map(|q| self.search(q, k)).collect()
    }
}

/// Search-strictness knobs for trifle (`m`, `B`; §10.3). `None` leaves the engine
/// default. Baselines have no analogue and ignore these.
#[derive(Clone, Copy, Default)]
pub struct Tuning {
    pub min_shared: Option<u32>,
    pub breadth: Option<u64>,
}

/// trifle itself.
pub struct Trifle {
    index: Index<TrigramTokenizer, Sidecar>,
    tuning: Tuning,
    // Held so the temp file outlives the index.
    _dir: tempfile::TempDir,
}

impl Trifle {
    pub fn build(corpus: &Corpus, tuning: Tuning) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let index = Index::open_at(&dir.path().join("trifle.db"), Config::default()).unwrap();
        let segs = corpus
            .docs
            .iter()
            .map(|d| trifle::Segment::new(d.id, "field", "body", d.text.clone()));
        index.insert_batch(segs).unwrap();
        index.compact().unwrap(); // steady-state read shape (folded bases)
        Trifle {
            index,
            tuning,
            _dir: dir,
        }
    }

    fn opts(&self, k: usize) -> SearchOpts<'static> {
        let mut o = SearchOpts::new(k);
        if let Some(m) = self.tuning.min_shared {
            o = o.min_shared(m);
        }
        if let Some(b) = self.tuning.breadth {
            o = o.breadth(b);
        }
        o
    }
}

impl Engine for Trifle {
    fn name(&self) -> &'static str {
        "trifle"
    }
    fn search(&self, query: &str, k: usize) -> Vec<i64> {
        self.index
            .search(query, self.opts(k))
            .unwrap()
            .into_iter()
            .map(|m| m.doc_id)
            .collect()
    }
    fn search_many(&self, queries: &[&str], k: usize) -> Vec<Vec<i64>> {
        self.index
            .search_batch(queries, self.opts(k))
            .unwrap()
            .into_iter()
            .map(|ms| ms.into_iter().map(|m| m.doc_id).collect())
            .collect()
    }
}

/// FTS5 trigram index with BM25 ranking — the in-DB BM25 baseline.
pub struct Fts5 {
    conn: Connection,
    _dir: tempfile::TempDir,
}

impl Fts5 {
    /// `None` if the linked SQLite lacks FTS5 + the trigram tokenizer (the bundled
    /// build always has both, so this is a defensive probe).
    pub fn build(corpus: &Corpus) -> Option<Self> {
        let dir = tempfile::tempdir().unwrap();
        let conn = Connection::open(dir.path().join("fts5.db")).ok()?;
        conn.execute_batch("CREATE VIRTUAL TABLE docs USING fts5(body, tokenize='trigram');")
            .ok()?;
        let tx = conn.unchecked_transaction().ok()?;
        {
            let mut stmt = tx
                .prepare("INSERT INTO docs(rowid, body) VALUES(?1, ?2)")
                .ok()?;
            for d in &corpus.docs {
                stmt.execute(rusqlite::params![d.id, d.text]).ok()?;
            }
        }
        tx.commit().ok()?;
        conn.execute("INSERT INTO docs(docs) VALUES('optimize')", [])
            .ok()?;
        Some(Fts5 { conn, _dir: dir })
    }
}

impl Engine for Fts5 {
    fn name(&self) -> &'static str {
        "fts5-trigram-bm25"
    }
    fn search(&self, query: &str, k: usize) -> Vec<i64> {
        // Quote the query as one FTS5 string literal (double internal quotes) — the
        // only injection-safe way to feed arbitrary user text to MATCH.
        let quoted = format!("\"{}\"", query.replace('"', "\"\""));
        let mut stmt = match self
            .conn
            .prepare_cached("SELECT rowid FROM docs WHERE docs MATCH ?1 ORDER BY rank LIMIT ?2")
        {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };
        let rows = stmt.query_map(rusqlite::params![quoted, k as i64], |r| r.get::<_, i64>(0));
        match rows {
            Ok(it) => it.filter_map(Result::ok).collect(),
            // A pathological MATCH string can be rejected by FTS5; that's a no-match.
            Err(_) => Vec::new(),
        }
    }
}

/// `LIKE '%…%'` substring scan — the naive floor.
pub struct Like {
    conn: Connection,
    _dir: tempfile::TempDir,
}

impl Like {
    pub fn build(corpus: &Corpus) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let conn = Connection::open(dir.path().join("like.db")).unwrap();
        conn.execute_batch("CREATE TABLE docs(id INTEGER PRIMARY KEY, body TEXT NOT NULL);")
            .unwrap();
        let tx = conn.unchecked_transaction().unwrap();
        {
            let mut stmt = tx
                .prepare("INSERT INTO docs(id, body) VALUES(?1, ?2)")
                .unwrap();
            for d in &corpus.docs {
                stmt.execute(rusqlite::params![d.id, d.text]).unwrap();
            }
        }
        tx.commit().unwrap();
        Like { conn, _dir: dir }
    }
}

impl Engine for Like {
    fn name(&self) -> &'static str {
        "like-scan"
    }
    fn search(&self, query: &str, k: usize) -> Vec<i64> {
        // Bound the LIKE pattern as a parameter; `\` escapes the LIKE metacharacters.
        let pat = format!(
            "%{}%",
            query
                .replace('\\', "\\\\")
                .replace('%', "\\%")
                .replace('_', "\\_")
        );
        let mut stmt = self
            .conn
            .prepare_cached("SELECT id FROM docs WHERE body LIKE ?1 ESCAPE '\\' LIMIT ?2")
            .unwrap();
        stmt.query_map(rusqlite::params![pat, k as i64], |r| r.get::<_, i64>(0))
            .map(|it| it.filter_map(Result::ok).collect())
            .unwrap_or_default()
    }
}

/// Path helper for engines that want their own file (unused by the temp-dir engines
/// above, kept for a future on-disk-size measurement).
#[allow(dead_code)]
pub fn sibling(dir: &Path, name: &str) -> std::path::PathBuf {
    dir.join(name)
}
