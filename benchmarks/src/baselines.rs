//! The baseline field: trifle vs in-process SQLite baselines on the *same* corpus and
//! queries, so the comparison isolates the matching strategy from the store. trifle and
//! both baselines link the same bundled SQLite.
//!
//! - **FTS5-trigram (BM25)** — the in-DB cousin. One trigram table with `ORDER BY rank`,
//!   queried either as a phrase (latency) or as an OR-bag of trigrams (fuzzy/relevance);
//!   see [`MatchMode`].
//! - **FTS5-word (BM25)** — word-level (`unicode61`) BM25, the *canonical* relevance
//!   baseline (real BM25 over words).
//! - **`LIKE '%…%'` scan** — the naive substring floor.
//!
//! Out-of-process engines (pg_trgm, Tantivy+Levenshtein, fzf/nucleo, fst/SymSpell) live
//! in a separate driver; this module is the embedded subset that shares trifle's store.

use std::path::Path;

use rusqlite::Connection;
use trifle::store::Sidecar;
use trifle::tokenize::TrigramTokenizer;
use trifle::{Config, Effort, Index, SearchOpts};

use crate::corpus::Corpus;

/// A search engine under test: build once from a corpus, then answer queries.
///
/// `search` answers one query; `search_batch` answers a set. The default
/// `search_batch` loops `search`, so a per-query-stateless engine gets batching
/// for free; trifle overrides it to share posting/frequency reads across the set
/// (its `search_batch`), which is the whole point of the `--batched` axis.
pub trait Engine {
    fn name(&self) -> &'static str;
    /// Return the matched document ids, best-first, capped at `k`.
    fn search(&self, query: &str, k: usize) -> Vec<i64>;
    /// Answer a batch of queries, one id list per query in order.
    fn search_batch(&self, queries: &[&str], k: usize) -> Vec<Vec<i64>> {
        queries.iter().map(|q| self.search(q, k)).collect()
    }
}

/// Search-strictness knobs for trifle (`m`, `t_max`) plus the rerank [`Effort`].
/// `None` leaves the engine default (Effort defaults to Medium). Baselines have no
/// analogue and ignore these.
#[derive(Clone, Copy, Default)]
pub struct Tuning {
    pub min_shared: Option<u32>,
    pub t_max: Option<usize>,
    pub effort: Option<Effort>,
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
        // Bulk-load via rebuild (roaring accumulation, folded bases): the steady-state
        // read shape, and far more memory-efficient than insert_batch at million-doc N.
        index.rebuild(segs).unwrap();
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
        if let Some(t) = self.tuning.t_max {
            o = o.t_max(t);
        }
        if let Some(e) = self.tuning.effort {
            o = o.rerank(e);
        }
        o
    }

    /// Search the top-`pool` overlap candidates, reranked by the BM25 precision tier,
    /// returning doc ids best-first. Exact pool control for the `ranksweep` calibration:
    /// `Effort::None` disables the √(kN) auto-pool (so pool == `limit`), while the
    /// explicit ranker still reranks. recall@k for any `k <= pool` is `id in result[..k]`.
    pub fn search_pool(&self, query: &str, pool: usize) -> Vec<i64> {
        let ranker = trifle::rank::Bm25Ranker;
        let mut o = SearchOpts::new(pool).ranker(&ranker).rerank(Effort::None);
        if let Some(m) = self.tuning.min_shared {
            o = o.min_shared(m);
        }
        if let Some(t) = self.tuning.t_max {
            o = o.t_max(t);
        }
        self.index
            .search(query, o)
            .unwrap()
            .into_iter()
            .map(|m| m.doc_id)
            .collect()
    }

    /// Like [`search_pool`](Self::search_pool) but with an explicit selection cap `t_max`,
    /// so the rarest `t_max` query trigrams are kept — selection is the swept variable
    /// while the pool is held generous. Backs the t_max knee sweep (`tools/tmax_knee.py`).
    pub fn search_pool_tmax(&self, query: &str, pool: usize, t_max: usize) -> Vec<i64> {
        let ranker = trifle::rank::Bm25Ranker;
        let mut o = SearchOpts::new(pool)
            .ranker(&ranker)
            .rerank(Effort::None)
            .t_max(t_max);
        if let Some(m) = self.tuning.min_shared {
            o = o.min_shared(m);
        }
        self.index
            .search(query, o)
            .unwrap()
            .into_iter()
            .map(|m| m.doc_id)
            .collect()
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
    fn search_batch(&self, queries: &[&str], k: usize) -> Vec<Vec<i64>> {
        self.index
            .search_batch(queries, self.opts(k))
            .unwrap()
            .into_iter()
            .map(|ms| ms.into_iter().map(|m| m.doc_id).collect())
            .collect()
    }
}

/// How an [`Fts5`] query is turned into a `MATCH` expression.
#[derive(Clone, Copy)]
pub enum MatchMode {
    /// The whole query as one quoted phrase — its trigrams must appear **adjacent**.
    /// The latency baseline. A typo splits the trigram run, so phrase mode scores ~0 on
    /// typo'd/paraphrased queries: it is NOT a fuzzy baseline — the recall evals use
    /// [`TrigramOr`](MatchMode::TrigramOr) instead.
    Phrase,
    /// An **OR-bag of the query's trigrams**, bm25-ranked. Overlapping trigrams from a
    /// typo'd or paraphrased query still match, so this is the fair fuzzy *and*
    /// relevance baseline (reported as "FTS5 trigram-MATCH"). Same trigram table.
    TrigramOr,
}

/// Character trigrams of `s` the way FTS5's `tokenize='trigram'` sees them: lowercased,
/// a sliding 3-codepoint window over the *whole* string (spaces and punctuation are
/// ordinary characters), deduplicated. Fewer than 3 codepoints → none. (FTS5 folds case
/// with its own Unicode rule and does no NFC/NFD; this is close enough for ASCII and the
/// residual accent-fold asymmetry is a real "matching semantics" difference, not a bug.)
fn trigrams(s: &str) -> Vec<String> {
    let chars: Vec<char> = s.to_lowercase().chars().collect();
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for w in chars.windows(3) {
        let t: String = w.iter().collect();
        if seen.insert(t.clone()) {
            out.push(t);
        }
    }
    out
}

/// Build the OR-bag `MATCH` expression `"t1" OR "t2" OR …` from the query's trigrams
/// (each quoted, internal quotes doubled). `None` if the query has no trigrams.
fn trigram_or_match(query: &str) -> Option<String> {
    let tris = trigrams(query);
    if tris.is_empty() {
        return None;
    }
    let mut m = String::new();
    for (i, t) in tris.iter().enumerate() {
        if i > 0 {
            m.push_str(" OR ");
        }
        m.push('"');
        m.push_str(&t.replace('"', "\"\""));
        m.push('"');
    }
    Some(m)
}

/// Run a `docs MATCH ?1 ORDER BY rank LIMIT ?2` over an FTS5 `docs(body)` table.
fn fts5_match(conn: &Connection, match_str: &str, k: usize) -> Vec<i64> {
    let mut stmt = match conn
        .prepare_cached("SELECT rowid FROM docs WHERE docs MATCH ?1 ORDER BY rank LIMIT ?2")
    {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };
    match stmt.query_map(rusqlite::params![match_str, k as i64], |r| {
        r.get::<_, i64>(0)
    }) {
        Ok(it) => it.filter_map(Result::ok).collect(),
        // A pathological MATCH string can be rejected by FTS5; that's a no-match.
        Err(_) => Vec::new(),
    }
}

/// Insert every doc into an FTS5 `docs(body)` table under its real id as `rowid`, then
/// `'optimize'`. Returns `None` on any failure (e.g. tokenizer unavailable).
fn fts5_load(conn: &Connection, corpus: &Corpus) -> Option<()> {
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
    Some(())
}

/// FTS5 **trigram** index with BM25 ranking. The same table serves the latency benchmark
/// ([`Phrase`](MatchMode::Phrase)) and the fuzzy/relevance evals
/// ([`TrigramOr`](MatchMode::TrigramOr)); the mode only changes how the query becomes a
/// `MATCH`, so latency keeps its phrase behavior unchanged.
pub struct Fts5 {
    conn: Connection,
    mode: MatchMode,
    _dir: tempfile::TempDir,
}

impl Fts5 {
    /// `None` if the linked SQLite lacks FTS5 + the trigram tokenizer (the bundled
    /// build always has both, so this is a defensive probe).
    pub fn build(corpus: &Corpus, mode: MatchMode) -> Option<Self> {
        let dir = tempfile::tempdir().unwrap();
        let conn = Connection::open(dir.path().join("fts5.db")).ok()?;
        conn.execute_batch("CREATE VIRTUAL TABLE docs USING fts5(body, tokenize='trigram');")
            .ok()?;
        fts5_load(&conn, corpus)?;
        Some(Fts5 {
            conn,
            mode,
            _dir: dir,
        })
    }
}

impl Engine for Fts5 {
    fn name(&self) -> &'static str {
        "fts5-trigram-bm25"
    }
    fn search(&self, query: &str, k: usize) -> Vec<i64> {
        let match_str = match self.mode {
            // One quoted phrase literal (internal quotes doubled).
            MatchMode::Phrase => format!("\"{}\"", query.replace('"', "\"\"")),
            MatchMode::TrigramOr => match trigram_or_match(query) {
                Some(m) => m,
                None => return Vec::new(),
            },
        };
        fts5_match(&self.conn, &match_str, k)
    }
}

/// FTS5 **word-level** (`unicode61`) index with BM25 — the *canonical* BM25 baseline for
/// the relevance eval (real BM25 over words, not trigrams). The trigram-bm25 engine is
/// the same-tokenization cousin; reporting both keeps neither standing in for the other.
pub struct Fts5Word {
    conn: Connection,
    _dir: tempfile::TempDir,
}

impl Fts5Word {
    pub fn build(corpus: &Corpus) -> Option<Self> {
        let dir = tempfile::tempdir().unwrap();
        let conn = Connection::open(dir.path().join("fts5w.db")).ok()?;
        conn.execute_batch("CREATE VIRTUAL TABLE docs USING fts5(body, tokenize='unicode61');")
            .ok()?;
        fts5_load(&conn, corpus)?;
        Some(Fts5Word { conn, _dir: dir })
    }
}

impl Engine for Fts5Word {
    fn name(&self) -> &'static str {
        "fts5-word-bm25"
    }
    fn search(&self, query: &str, k: usize) -> Vec<i64> {
        // OR the query's words (alphanumeric-folded, lowercased) so a multi-word query
        // ranks by bm25 over any matching word — the standard lexical-recall baseline.
        // Bare terms would be implicit-AND (too strict for recall); join with OR.
        let mut m = String::new();
        for w in query.split_whitespace() {
            let w: String = w
                .chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>();
            if w.is_empty() {
                continue;
            }
            if !m.is_empty() {
                m.push_str(" OR ");
            }
            m.push('"');
            m.push_str(&w.to_lowercase());
            m.push('"');
        }
        if m.is_empty() {
            return Vec::new();
        }
        fts5_match(&self.conn, &m, k)
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
