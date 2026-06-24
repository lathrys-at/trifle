//! `trifle` — embedded, typo-tolerant trigram fuzzy search backed by SQLite.
//!
//! trifle indexes short text **segments** and answers typo/partial-tolerant
//! queries, returning a ranked list of matches each carrying *where* it matched. It
//! owns a single SQLite store holding the segment text, provenance, and an owned
//! **roaring inverted index** (a base+delta roaring posting per token); it ranks by
//! shared-rare-token overlap, counted bit-sliced. It is a **derived, rebuildable
//! cache** over a caller-owned source of truth: it never touches the caller's data
//! store.
//!
//! It targets a specific regime — **large corpora of small documents**
//! (≲ 1–2 KB per segment), read-often / write-infrequent. The ranking omits length
//! normalization, which is sound only for small documents.
//!
//! # Quick start
//!
//! ```
//! use trifle::{Config, Index, SearchOpts};
//!
//! # fn main() -> trifle::Result<()> {
//! let dir = tempfile::tempdir().unwrap();
//! let index = Index::open_at(&dir.path().join("trifle.db"), Config::default())?;
//!
//! index.insert(1, "field", &[("front", "the quick brown fox")])?;
//! index.insert(2, "field", &[("front", "the quack brown ox")])?;
//!
//! // A typo'd query still matches.
//! let hits = index.search("quikc brown", SearchOpts::new(10))?;
//! assert!(hits.iter().any(|m| m.doc_id == 1));
//! # Ok(())
//! # }
//! ```
//!
//! # The document model
//!
//! A segment is `(doc_id, source, ref, text)`. `source` and `ref` are two
//! caller-defined, opaque provenance labels (intended: `source` a category like
//! `"field"` or `"ocr"`, `ref` a sub-location like a field name) returned on a
//! match. [`insert`](Index::insert) replaces all segments under a `(doc_id, source)`
//! pair; [`remove`](Index::remove) removes all segments of a `doc_id`.
//!
//! # What trifle leaves to the caller
//!
//! Embeddings/semantic search, fusion (RRF) with other signals, an exact precision
//! tier beyond a custom [`Ranker`](rank::Ranker), sub-trigram (`<3`-char) query
//! handling, and deciding *when* the cache is stale relative to the source of truth.

#![forbid(unsafe_op_in_unsafe_fn)]

mod instrument;

pub mod error;
pub mod rank;
pub mod store;
pub mod tokenize;

mod postings;
mod schema;
mod select;

use std::borrow::Borrow;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::Path;
use std::time::Duration;

use roaring::RoaringBitmap;
use rusqlite::Connection;
use rusqlite::types::Value;

/// Re-export of the `rusqlite` version trifle is built against, so consumers
/// implementing a custom [`store::Backend`] (or constructing a [`store::Shared`])
/// use exactly the right `Connection` type.
pub use rusqlite;

pub use error::{Error, Result};
use rank::{Candidates, OverlapRanker, QueryContext, Ranker, Survivor, overlap_search};
use schema::SCHEMA_VERSION;
use select::{SelectParams, select};
use store::{Backend, Namespace, Sidecar, TextResolver};
use tokenize::{Tokenizer, TrigramTokenizer};

/// Default match floor `m` — shared rare tokens required for a hit.
const DEFAULT_MIN_SHARED: u32 = 2;
/// Default breadth budget `B` — `0` keeps exactly the typo floor.
const DEFAULT_BREADTH: u64 = 0;
/// Per-typo token damage `d`; the typo floor is `F = m + d`.
const TYPO_DAMAGE: u32 = 4;
/// Default absolute ceiling `k_max` on selected tokens.
const DEFAULT_K_MAX: usize = 12;
/// How many times a read retries a transient `SQLITE_BUSY`/`LOCKED`/`SCHEMA`.
const RETRY_MAX: usize = 5;

/// Advanced, rarely-touched tuning. The defaults are a pure `Σdf` selection cost
/// and a fixed safety ceiling; reach for these only when calibrating against a
/// benchmark.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Advanced {
    /// `α` — per-kept-token selection cost coefficient.
    pub alpha: f64,
    /// `β` — per-document-frequency selection cost coefficient.
    pub beta: f64,
    /// `k_max` — the absolute ceiling on selected tokens.
    pub k_max: usize,
}

impl Default for Advanced {
    fn default() -> Self {
        Advanced {
            alpha: 0.0,
            beta: 1.0,
            k_max: DEFAULT_K_MAX,
        }
    }
}

/// Index configuration.
///
/// The common case is `Config::default()` (or [`Config::new`] with a drift token).
/// The tokenizer and backend are *type* parameters supplied at [`open`](Index::open),
/// not config fields.
#[derive(Default)]
pub struct Config {
    /// The caller's drift/epoch token. Bumping it on reopen invalidates the cache
    /// (drops it to empty), so the next [`rebuild`](Index::rebuild) repopulates.
    pub data_version: u64,
    /// Advanced selection tuning (rarely touched).
    pub advanced: Advanced,
    /// A [`TextResolver`] to run **contentless** (store no text snapshot, fetch text
    /// from the caller's source). `None` (default) stores a self-contained snapshot.
    /// See the [crate](crate) docs and [`TextResolver`] for the contract.
    pub external_content: Option<Box<dyn TextResolver>>,
}

impl Config {
    /// A configuration with the given drift token and otherwise default settings.
    pub fn new(data_version: u64) -> Self {
        Config {
            data_version,
            ..Config::default()
        }
    }

    /// Set the advanced tuning.
    pub fn with_advanced(mut self, advanced: Advanced) -> Self {
        self.advanced = advanced;
        self
    }

    /// Run contentless against the given resolver instead of storing a text snapshot.
    pub fn with_external_content(mut self, resolver: Box<dyn TextResolver>) -> Self {
        self.external_content = Some(resolver);
        self
    }
}

/// A segment to index: `(doc_id, source, ref, text)`. Used by
/// [`insert_batch`](Index::insert_batch) and [`rebuild`](Index::rebuild).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Segment {
    /// The caller's document id.
    pub doc_id: i64,
    /// Provenance category (e.g. `"field"`, `"ocr"`).
    pub source: String,
    /// Provenance sub-location (e.g. a field name, a filename).
    pub ref_: String,
    /// The segment text (stored raw; the tokenizer normalizes internally).
    pub text: String,
}

impl Segment {
    /// Construct a segment.
    pub fn new(
        doc_id: i64,
        source: impl Into<String>,
        ref_: impl Into<String>,
        text: impl Into<String>,
    ) -> Self {
        Segment {
            doc_id,
            source: source.into(),
            ref_: ref_.into(),
            text: text.into(),
        }
    }
}

/// One ranked match. Rank is conveyed by position in the returned `Vec<Match>`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Match {
    /// The caller's document id.
    pub doc_id: i64,
    /// The matched segment's provenance category.
    pub source: String,
    /// The matched segment's provenance sub-location.
    pub ref_: String,
    /// The `[first, last)` UTF-8 byte span of the matched region within
    /// [`text`](Self::text). `None` when no span could be located (a custom
    /// tokenizer without span support, or text unavailable in contentless mode).
    pub span: Option<(usize, usize)>,
    /// The whole matched segment, original form. `None` only in contentless mode
    /// when the resolver returned no text.
    pub text: Option<String>,
}

/// A scope/exclusion predicate over a candidate's provenance:
/// `(doc_id, source, ref) -> keep`. Used for [`SearchOpts::scope`].
pub type ScopeFn = dyn Fn(i64, &str, &str) -> bool;

/// Per-search options.
///
/// Construct with [`SearchOpts::new`] and the builder setters, or the public fields.
/// The one knob most callers reach for is [`min_shared`](SearchOpts::min_shared)
/// (`m`, the strictness dial); [`breadth`](SearchOpts::breadth) (`B`) is the
/// orthogonal recall axis.
pub struct SearchOpts<'a> {
    /// Maximum number of matches to return (top-k).
    pub limit: usize,
    /// `m` — the match floor (shared rare tokens for a hit). `None` → `2`.
    pub min_shared: Option<u32>,
    /// `B` — the breadth budget in selection cost units. `None` → `0`.
    pub breadth: Option<u64>,
    /// A per-query [`Ranker`](rank::Ranker). `None` → the built-in
    /// [`OverlapRanker`](rank::OverlapRanker).
    pub ranker: Option<&'a dyn Ranker>,
    /// A membership predicate `(doc_id, source, ref) -> keep` evaluated over
    /// candidates in overlap order — never over the corpus. The walk continues until
    /// `limit` predicate-passing results lock, so scoping needs no over-fetch. The
    /// predicate must not call back into this index's writer.
    pub scope: Option<&'a ScopeFn>,
}

impl<'a> SearchOpts<'a> {
    /// Options for the given result limit, everything else default.
    pub fn new(limit: usize) -> Self {
        SearchOpts {
            limit,
            min_shared: None,
            breadth: None,
            ranker: None,
            scope: None,
        }
    }

    /// Set the match floor `m`.
    pub fn min_shared(mut self, m: u32) -> Self {
        self.min_shared = Some(m);
        self
    }

    /// Set the breadth budget `B`.
    pub fn breadth(mut self, b: u64) -> Self {
        self.breadth = Some(b);
        self
    }

    /// Set a per-query ranker.
    pub fn ranker(mut self, ranker: &'a dyn Ranker) -> Self {
        self.ranker = Some(ranker);
        self
    }

    /// Set a scope predicate.
    pub fn scope(mut self, scope: &'a ScopeFn) -> Self {
        self.scope = Some(scope);
        self
    }
}

impl Default for SearchOpts<'_> {
    fn default() -> Self {
        SearchOpts::new(10)
    }
}

/// A read-only snapshot of the index's observable state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stats {
    /// Number of indexed segments (`N`).
    pub segments: u64,
    /// Number of distinct tokens with a live (non-zero) frequency.
    pub terms: u64,
    /// Pending delta rows — the signal for *when* to [`compact`](Index::compact).
    pub delta_backlog: u64,
    /// On-disk size of the backing database file, in bytes (in `Shared` mode this is
    /// the caller's whole file, not trifle's tables alone).
    pub disk_bytes: u64,
    /// The caller drift token currently stamped.
    pub data_version: u64,
    /// The tokenizer fingerprint currently stamped.
    pub tokenizer_fingerprint: u64,
    /// trifle's on-disk schema version.
    pub schema_version: u32,
}

/// What a [`compact`](Index::compact) reclaimed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CompactStats {
    /// Tokens whose pending delta was folded into the base.
    pub tokens_folded: u64,
    /// Stale ids purged from base postings.
    pub ids_purged: u64,
    /// Tokens dropped entirely (posting emptied or frequency fell to zero).
    pub terms_dropped: u64,
    /// Delta-blob bytes the fold cleared.
    pub bytes_reclaimed: u64,
}

/// Where segment text comes from.
enum Content {
    /// Default: a text snapshot is stored in `seg.txt`.
    Snapshot,
    /// Contentless: no snapshot; text resolved through the caller's [`TextResolver`],
    /// and per-segment token sets stored in `fwd` so deletion needs no text.
    External(Box<dyn TextResolver>),
}

/// An embedded fuzzy-search index over text segments.
///
/// Generic over the [`Tokenizer`](tokenize::Tokenizer) (monomorphized — it is on the
/// hot path) and the storage [`Backend`](store::Backend). Both default, so the common
/// case is just `Index`. Open with [`open_at`](Index::open_at) (an owned sidecar
/// file) or [`open`](Index::open) (any backend).
///
/// All methods are synchronous and thread-safe (`&self`): a single internal writer
/// is serialized, reads run on a pooled connection concurrently with the writer
/// under WAL. A caller on an async runtime dispatches calls to a blocking pool.
pub struct Index<T: Tokenizer = TrigramTokenizer, B: Backend = Sidecar> {
    backend: B,
    tokenizer: T,
    advanced: Advanced,
    data_version: u64,
    content: Content,
}

impl Index<TrigramTokenizer, Sidecar> {
    /// Open (creating if absent) an index at `path` with the default trigram
    /// tokenizer and an owned [`Sidecar`](store::Sidecar) file. The common case.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened or the store cannot be
    /// initialized.
    pub fn open_at(path: &Path, config: Config) -> Result<Self> {
        let backend = Sidecar::open(path)?;
        Index::open(backend, TrigramTokenizer::new(), config)
    }
}

impl<T: Tokenizer, B: Backend> Index<T, B> {
    /// Open an index over a given backend and tokenizer.
    ///
    /// On open, the store is created if absent and checked against three version
    /// stamps (schema, tokenizer fingerprint, caller `data_version`); any mismatch —
    /// or a detected `seg`↔posting desync — drops the cache to empty (no migrations).
    /// After such a reset, [`rebuild`](Self::rebuild) repopulates it.
    ///
    /// # Errors
    ///
    /// Returns an error if the store cannot be initialized.
    pub fn open(backend: B, tokenizer: T, config: Config) -> Result<Self> {
        let content = match config.external_content {
            Some(resolver) => Content::External(resolver),
            None => Content::Snapshot,
        };
        let index = Index {
            backend,
            tokenizer,
            advanced: config.advanced,
            data_version: config.data_version,
            content,
        };
        index.init()?;
        Ok(index)
    }

    fn is_contentless(&self) -> bool {
        matches!(self.content, Content::External(_))
    }

    /// Create tables and reconcile drift, all in one transaction.
    fn init(&self) -> Result<()> {
        let mut guard = self.backend.write()?;
        let ns = self.backend.namespace();
        let tx = guard.transaction()?;
        schema::drop_shadows(&tx, ns)?;
        schema::create_tables(&tx, ns)?;

        let stamps = schema::read_stamps(&tx, ns)?;
        let fingerprint = self.tokenizer.fingerprint();
        let drift = stamps.schema_version != Some(SCHEMA_VERSION)
            || stamps.fingerprint != Some(fingerprint)
            || stamps.data_version != Some(self.data_version);
        let desync = !drift && self.desync(&tx, ns)?;

        if drift || desync {
            schema::reset(&tx, ns)?;
            schema::set_next_id(&tx, ns, 1)?;
            schema::write_stamps(&tx, ns, self.data_version, fingerprint)?;
        }
        tx.commit()?;
        Ok(())
    }

    /// A cheap consistency probe: the monotonic-id invariant `max(seg.id) < next_id`.
    fn desync(&self, conn: &Connection, ns: &Namespace) -> Result<bool> {
        let max_id: Option<i64> =
            conn.query_row(&format!("SELECT max(id) FROM {}", ns.seg()), [], |r| {
                r.get(0)
            })?;
        let next: i64 = schema::meta_get(conn, ns, schema::KEY_NEXT_ID)?
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);
        Ok(max_id.is_some_and(|m| m >= next))
    }

    // ----- writes -------------------------------------------------------------

    /// Replace all segments under `(doc_id, source)` with the given `(ref, text)`
    /// pairs, in one transaction.
    ///
    /// # Errors
    ///
    /// Returns an error if the store write fails.
    pub fn insert(&self, doc_id: i64, source: &str, segments: &[(&str, &str)]) -> Result<()> {
        let mut guard = self.backend.write()?;
        let ns = self.backend.namespace();
        let tx = guard.transaction()?;
        let mut changes = TokenChanges::default();
        self.replace_group(&tx, ns, doc_id, source, segments, &mut changes)?;
        changes.apply(&tx, ns)?;
        tx.commit()?;
        Ok(())
    }

    /// Insert many segments in one transaction. Segments are grouped by
    /// `(doc_id, source)`, and each group replaces any existing segments under that
    /// pair (the same semantics as [`insert`](Self::insert)).
    ///
    /// # Errors
    ///
    /// Returns an error if the store write fails.
    pub fn insert_batch(&self, batch: impl IntoIterator<Item = Segment>) -> Result<()> {
        // Group by (doc_id, source); a group's segments all become that pair's content.
        let mut groups: HashMap<(i64, String), Vec<(String, String)>> = HashMap::new();
        for seg in batch {
            groups
                .entry((seg.doc_id, seg.source))
                .or_default()
                .push((seg.ref_, seg.text));
        }
        if groups.is_empty() {
            return Ok(());
        }
        let mut guard = self.backend.write()?;
        let ns = self.backend.namespace();
        let tx = guard.transaction()?;
        let mut changes = TokenChanges::default();
        for ((doc_id, source), refs_text) in &groups {
            let pairs: Vec<(&str, &str)> = refs_text
                .iter()
                .map(|(r, t)| (r.as_str(), t.as_str()))
                .collect();
            self.replace_group(&tx, ns, *doc_id, source, &pairs, &mut changes)?;
        }
        changes.apply(&tx, ns)?;
        tx.commit()?;
        Ok(())
    }

    /// Remove all segments of `doc_id` (every source), in one transaction.
    ///
    /// # Errors
    ///
    /// Returns an error if the store write fails.
    pub fn remove(&self, doc_id: i64) -> Result<()> {
        let mut guard = self.backend.write()?;
        let ns = self.backend.namespace();
        let tx = guard.transaction()?;

        let old = self.old_segments(&tx, ns, doc_id, None)?;
        let mut changes = TokenChanges::default();
        for (id, tokens) in old {
            changes.remove(id, tokens);
        }
        tx.execute(
            &format!("DELETE FROM {} WHERE doc_id = ?1", ns.seg()),
            rusqlite::params![doc_id],
        )?;
        if self.is_contentless() {
            self.delete_fwd_for_doc(&tx, ns, doc_id)?;
        }
        changes.apply(&tx, ns)?;
        tx.commit()?;
        Ok(())
    }

    /// Delete-then-insert one `(doc_id, source)` group, accumulating its token
    /// changes into `changes` (applied once per write batch).
    fn replace_group(
        &self,
        conn: &Connection,
        ns: &Namespace,
        doc_id: i64,
        source: &str,
        segments: &[(&str, &str)],
        changes: &mut TokenChanges,
    ) -> Result<()> {
        // Removals: the old segments' ids and token sets.
        let old = self.old_segments(conn, ns, doc_id, Some(source))?;
        for (id, tokens) in old {
            changes.remove(id, tokens);
        }
        conn.execute(
            &format!("DELETE FROM {} WHERE doc_id = ?1 AND source = ?2", ns.seg()),
            rusqlite::params![doc_id, source],
        )?;
        if self.is_contentless() {
            conn.execute(
                &format!(
                    "DELETE FROM {} WHERE id IN (SELECT id FROM {} WHERE doc_id = ?1 AND source = ?2)",
                    ns.fwd(),
                    ns.seg()
                ),
                rusqlite::params![doc_id, source],
            )?;
        }

        if segments.is_empty() {
            return Ok(());
        }
        // Additions: fresh monotonic ids for the new segments.
        let first_id = schema::alloc_ids(conn, ns, segments.len() as u64)?;
        let contentless = self.is_contentless();
        let mut seg_ins = conn.prepare_cached(&format!(
            "INSERT INTO {}(id, doc_id, source, ref, txt) VALUES(?1, ?2, ?3, ?4, ?5)",
            ns.seg()
        ))?;
        let mut fwd_ins = if contentless {
            Some(conn.prepare_cached(&format!(
                "INSERT INTO {}(id, tokens) VALUES(?1, ?2)",
                ns.fwd()
            ))?)
        } else {
            None
        };
        for (i, (ref_, text)) in segments.iter().enumerate() {
            let id = first_id + i as i64;
            let stored_txt = if contentless { None } else { Some(*text) };
            seg_ins.execute(rusqlite::params![id, doc_id, source, ref_, stored_txt])?;
            let tokens = self.distinct_tokens(text);
            if let Some(stmt) = fwd_ins.as_mut() {
                stmt.execute(rusqlite::params![id, encode_tokens(&tokens)])?;
            }
            changes.add(id as u32, tokens);
        }
        Ok(())
    }

    /// The `(id, token set)` of every existing segment under `doc_id` (optionally one
    /// `source`). In snapshot mode the token set is re-derived from the stored text;
    /// in contentless mode it comes from the stored `fwd` token set (the source text
    /// is typically already gone on a delete).
    fn old_segments(
        &self,
        conn: &Connection,
        ns: &Namespace,
        doc_id: i64,
        source: Option<&str>,
    ) -> Result<Vec<(u32, Vec<String>)>> {
        let (sql, params): (String, Vec<Value>) = match source {
            Some(s) => (
                format!(
                    "SELECT id, txt FROM {} WHERE doc_id = ?1 AND source = ?2",
                    ns.seg()
                ),
                vec![Value::Integer(doc_id), Value::Text(s.to_string())],
            ),
            None => (
                format!("SELECT id, txt FROM {} WHERE doc_id = ?1", ns.seg()),
                vec![Value::Integer(doc_id)],
            ),
        };
        let mut rows_out: Vec<(u32, Option<String>)> = Vec::new();
        {
            let mut stmt = conn.prepare_cached(&sql)?;
            let mut rows = stmt.query(rusqlite::params_from_iter(params))?;
            while let Some(r) = rows.next()? {
                rows_out.push((r.get::<_, i64>(0)? as u32, r.get::<_, Option<String>>(1)?));
            }
        }
        if self.is_contentless() {
            let ids: Vec<u32> = rows_out.iter().map(|(id, _)| *id).collect();
            let fwd = self.read_fwd(conn, ns, &ids)?;
            Ok(ids
                .into_iter()
                .map(|id| (id, fwd.get(&id).cloned().unwrap_or_default()))
                .collect())
        } else {
            Ok(rows_out
                .into_iter()
                .map(|(id, txt)| (id, self.distinct_tokens(txt.as_deref().unwrap_or(""))))
                .collect())
        }
    }

    /// The distinct token strings of `text`, deduplicated via the token type (no
    /// allocation per duplicate window — only the distinct set is stringified).
    fn distinct_tokens(&self, text: &str) -> Vec<String> {
        let distinct: HashSet<T::Token> = self.tokenizer.tokenize(text).collect();
        distinct
            .into_iter()
            .map(|t| t.borrow().to_string())
            .collect()
    }

    // ----- maintenance --------------------------------------------------------

    /// Fold pending deltas into bases, drop emptied tokens, and prune zero-frequency
    /// terms. Heavier for common tokens; call on a schedule or when idle. Does not
    /// shrink the file (run `VACUUM` yourself if you own it); it bounds delta growth
    /// and posting fragmentation.
    ///
    /// # Errors
    ///
    /// Returns an error if the store write fails.
    pub fn compact(&self) -> Result<CompactStats> {
        let mut guard = self.backend.write()?;
        let ns = self.backend.namespace();
        let tx = guard.transaction()?;
        let stats = postings::fold(&tx, ns)?;
        tx.commit()?;
        Ok(CompactStats {
            tokens_folded: stats.tokens_folded,
            ids_purged: stats.ids_purged,
            terms_dropped: stats.terms_dropped,
            bytes_reclaimed: stats.bytes_reclaimed,
        })
    }

    /// Fully reindex from `corpus` via an atomic shadow swap: build into shadow
    /// tables, then drop-and-rename them in one transaction so a reader sees
    /// complete-old or complete-new, never partial. Reassigns dense ids (reclaiming a
    /// grown monotonic id space) and stamps the current versions.
    ///
    /// Required after a tokenizer change or a `data_version` bump (both empty the
    /// cache on open), and useful to reclaim space.
    ///
    /// # Errors
    ///
    /// Returns an error if the store write fails; on failure the live index is left
    /// intact.
    pub fn rebuild(&self, corpus: impl IntoIterator<Item = Segment>) -> Result<()> {
        let mut guard = self.backend.write()?;
        let ns = self.backend.namespace();
        let contentless = self.is_contentless();

        let tx = guard.transaction()?;
        schema::create_shadows(&tx, ns)?;

        // Accumulate the inverted index in memory while streaming segment rows to the
        // shadow tables (O(chunk) memory for the text; postings are roaring-compact).
        let mut inverted: HashMap<String, RoaringBitmap> = HashMap::new();
        let mut next_id: i64 = 1;
        {
            let mut seg_ins = tx.prepare_cached(&format!(
                "INSERT INTO {}(id, doc_id, source, ref, txt) VALUES(?1, ?2, ?3, ?4, ?5)",
                ns.seg_shadow()
            ))?;
            let mut fwd_ins = if contentless {
                Some(tx.prepare_cached(&format!(
                    "INSERT INTO {}(id, tokens) VALUES(?1, ?2)",
                    ns.fwd_shadow()
                ))?)
            } else {
                None
            };
            for seg in corpus {
                let id = next_id;
                next_id += 1;
                let stored_txt = if contentless {
                    None
                } else {
                    Some(seg.text.as_str())
                };
                seg_ins.execute(rusqlite::params![
                    id, seg.doc_id, seg.source, seg.ref_, stored_txt
                ])?;
                let tokens = self.distinct_tokens(&seg.text);
                if let Some(stmt) = fwd_ins.as_mut() {
                    stmt.execute(rusqlite::params![id, encode_tokens(&tokens)])?;
                }
                for tok in tokens {
                    inverted.entry(tok).or_default().insert(id as u32);
                }
            }
        }

        postings::write_base_postings(
            &tx,
            ns.post_shadow(),
            ns.term_shadow(),
            inverted.iter().map(|(t, bm)| (t.as_str(), bm)),
        )?;

        schema::swap_shadows(&tx, ns)?;
        schema::set_next_id(&tx, ns, next_id)?;
        schema::write_stamps(&tx, ns, self.data_version, self.tokenizer.fingerprint())?;
        tx.commit()?;
        Ok(())
    }

    /// A read-only snapshot of observable state.
    ///
    /// # Errors
    ///
    /// Returns an error if the store read fails.
    pub fn stats(&self) -> Result<Stats> {
        let ns = self.backend.namespace();
        let conn = self.backend.read()?;
        let segments: u64 =
            conn.query_row(&format!("SELECT count(*) FROM {}", ns.seg()), [], |r| {
                r.get::<_, i64>(0)
            })? as u64;
        let page_count: i64 = conn.query_row("PRAGMA page_count", [], |r| r.get(0))?;
        let page_size: i64 = conn.query_row("PRAGMA page_size", [], |r| r.get(0))?;
        let stamps = schema::read_stamps(&conn, ns)?;
        Ok(Stats {
            segments,
            terms: postings::term_count(&conn, ns)?,
            delta_backlog: postings::delta_backlog(&conn, ns)?,
            disk_bytes: (page_count * page_size).max(0) as u64,
            data_version: stamps.data_version.unwrap_or(self.data_version),
            tokenizer_fingerprint: stamps
                .fingerprint
                .unwrap_or_else(|| self.tokenizer.fingerprint()),
            schema_version: stamps.schema_version.unwrap_or(SCHEMA_VERSION),
        })
    }

    // ----- reads --------------------------------------------------------------

    /// Search for `query`, returning up to `opts.limit` ranked matches.
    ///
    /// # Errors
    ///
    /// Returns an error if the store read fails.
    pub fn search(&self, query: &str, opts: SearchOpts<'_>) -> Result<Vec<Match>> {
        Ok(self
            .search_batch(&[query], opts)?
            .into_iter()
            .next()
            .unwrap_or_default())
    }

    /// Search a batch of queries, one result list per query in order.
    ///
    /// Posting and frequency reads are shared across the batch, but each query's
    /// selection and ranking derive only from its own token frequencies, so
    /// `search_batch([…, q, …])` ranks `q` identically to `search(q)` (batch ==
    /// serial).
    ///
    /// # Errors
    ///
    /// Returns an error if the store read fails.
    pub fn search_batch(&self, queries: &[&str], opts: SearchOpts<'_>) -> Result<Vec<Vec<Match>>> {
        if queries.is_empty() {
            return Ok(Vec::new());
        }
        let min_shared = opts.min_shared.unwrap_or(DEFAULT_MIN_SHARED).max(1);
        let breadth = opts.breadth.unwrap_or(DEFAULT_BREADTH);
        let sel_params = SelectParams {
            min_shared,
            breadth,
            typo_damage: TYPO_DAMAGE,
            k_max: self.advanced.k_max,
            alpha: self.advanced.alpha,
            beta: self.advanced.beta,
        };
        let default_ranker = OverlapRanker;
        let ranker: &dyn Ranker = opts.ranker.unwrap_or(&default_ranker);

        // Distinct tokens per query (deduplicated), computed once outside the read.
        let query_tokens: Vec<Vec<String>> =
            queries.iter().map(|q| self.distinct_tokens(q)).collect();

        self.read_retry(|conn, ns| {
            // One batched frequency read over every distinct token in the batch.
            let all_tokens: BTreeSet<&str> = query_tokens
                .iter()
                .flat_map(|q| q.iter().map(String::as_str))
                .collect();
            let all_vec: Vec<&str> = all_tokens.into_iter().collect();
            let dfs = postings::read_dfs(conn, ns, &all_vec)?;

            // Per-query selection, then one batched posting read over every selected
            // token in the batch.
            let selected_per: Vec<Vec<String>> = query_tokens
                .iter()
                .map(|q| {
                    let pairs: Vec<(String, i64)> = q
                        .iter()
                        .map(|t| (t.clone(), dfs.get(t.as_str()).copied().unwrap_or(0)))
                        .collect();
                    select(&pairs, sel_params)
                })
                .collect();
            let sel_all: BTreeSet<&str> = selected_per
                .iter()
                .flat_map(|s| s.iter().map(String::as_str))
                .collect();
            let sel_vec: Vec<&str> = sel_all.into_iter().collect();
            let postings_map = postings::effective_postings(conn, ns, &sel_vec)?;

            let mut out = Vec::with_capacity(queries.len());
            for (qi, query) in queries.iter().enumerate() {
                let selected = &selected_per[qi];
                let present: Vec<(&str, &RoaringBitmap)> = selected
                    .iter()
                    .filter_map(|t| postings_map.get(t.as_str()).map(|bm| (t.as_str(), bm)))
                    .collect();

                let mut survivors =
                    overlap_search(conn, ns, &present, opts.limit, min_shared, opts.scope)?;
                self.hydrate_text(conn, ns, &mut survivors)?;

                out.push(self.rank_to_matches(
                    &survivors, &present, selected, query, min_shared, ranker, opts.limit,
                ));
            }
            Ok(out)
        })
    }

    /// Run the ranker over the survivors and build the result matches.
    #[allow(clippy::too_many_arguments)]
    fn rank_to_matches(
        &self,
        survivors: &[Survivor],
        present: &[(&str, &RoaringBitmap)],
        selected: &[String],
        query: &str,
        min_shared: u32,
        ranker: &dyn Ranker,
        limit: usize,
    ) -> Vec<Match> {
        let candidates = Candidates::new(survivors, present);
        let qctx = QueryContext {
            query,
            selected,
            min_shared,
        };
        let ranked = ranker.rank(&candidates, &qctx);
        let sel_refs: Vec<&str> = selected.iter().map(String::as_str).collect();
        let mut matches = Vec::with_capacity(ranked.len().min(limit));
        for r in ranked.into_iter().take(limit) {
            let Some(s) = survivors.get(r.candidate) else {
                continue;
            };
            let span = s
                .text
                .as_deref()
                .and_then(|t| self.tokenizer.span(t, &sel_refs));
            matches.push(Match {
                doc_id: s.doc_id,
                source: s.source.clone(),
                ref_: s.ref_.clone(),
                span,
                text: s.text.clone(),
            });
        }
        matches
    }

    /// Hydrate each survivor's text: from `seg.txt` (snapshot) or the resolver
    /// (contentless), in one batched read.
    fn hydrate_text(
        &self,
        conn: &Connection,
        ns: &Namespace,
        survivors: &mut [Survivor],
    ) -> Result<()> {
        if survivors.is_empty() {
            return Ok(());
        }
        match &self.content {
            Content::Snapshot => {
                let ids: Vec<u32> = survivors.iter().map(|s| s.seg_id).collect();
                let arr: std::rc::Rc<Vec<Value>> =
                    std::rc::Rc::new(ids.iter().map(|&i| Value::Integer(i as i64)).collect());
                let sql = format!("SELECT id, txt FROM {} WHERE id IN rarray(?1)", ns.seg());
                let mut texts: HashMap<u32, Option<String>> = HashMap::with_capacity(ids.len());
                {
                    let mut stmt = conn.prepare_cached(&sql)?;
                    let mut rows = stmt.query(rusqlite::params![arr])?;
                    while let Some(r) = rows.next()? {
                        texts.insert(r.get::<_, i64>(0)? as u32, r.get::<_, Option<String>>(1)?);
                    }
                }
                for s in survivors {
                    s.text = texts.get(&s.seg_id).cloned().flatten();
                }
            }
            Content::External(resolver) => {
                let segs: Vec<(i64, &str, &str)> = survivors
                    .iter()
                    .map(|s| (s.doc_id, s.source.as_str(), s.ref_.as_str()))
                    .collect();
                let texts = resolver.resolve(&segs)?;
                for (s, text) in survivors.iter_mut().zip(texts) {
                    s.text = text;
                }
            }
        }
        Ok(())
    }

    /// Read the stored `fwd` token sets for a set of segment ids (contentless mode).
    fn read_fwd(
        &self,
        conn: &Connection,
        ns: &Namespace,
        ids: &[u32],
    ) -> Result<HashMap<u32, Vec<String>>> {
        let mut out = HashMap::with_capacity(ids.len());
        if ids.is_empty() {
            return Ok(out);
        }
        let arr: std::rc::Rc<Vec<Value>> =
            std::rc::Rc::new(ids.iter().map(|&i| Value::Integer(i as i64)).collect());
        let sql = format!("SELECT id, tokens FROM {} WHERE id IN rarray(?1)", ns.fwd());
        let mut stmt = conn.prepare_cached(&sql)?;
        let mut rows = stmt.query(rusqlite::params![arr])?;
        while let Some(r) = rows.next()? {
            let id: i64 = r.get(0)?;
            let blob: Vec<u8> = r.get(1)?;
            out.insert(id as u32, decode_tokens(&blob)?);
        }
        Ok(out)
    }

    /// Delete `fwd` rows for every segment of a doc (contentless delete).
    fn delete_fwd_for_doc(&self, conn: &Connection, ns: &Namespace, doc_id: i64) -> Result<()> {
        conn.execute(
            &format!(
                "DELETE FROM {} WHERE id IN (SELECT id FROM {} WHERE doc_id = ?1)",
                ns.fwd(),
                ns.seg()
            ),
            rusqlite::params![doc_id],
        )?;
        Ok(())
    }

    /// Run a read closure on a pooled connection, retrying a transient
    /// busy/locked/schema-change fault a few times before surfacing it.
    fn read_retry<R>(&self, mut f: impl FnMut(&Connection, &Namespace) -> Result<R>) -> Result<R> {
        let ns = self.backend.namespace();
        let mut attempt = 0;
        loop {
            let conn = self.backend.read()?;
            match f(&conn, ns) {
                Err(Error::Sqlite(e)) if attempt < RETRY_MAX && is_retryable(&e) => {
                    drop(conn);
                    attempt += 1;
                    std::thread::sleep(Duration::from_millis(10 * attempt as u64));
                }
                other => return other,
            }
        }
    }
}

/// Accumulated per-token id changes for one write batch, applied in a single
/// [`postings::apply_writes`] pass.
#[derive(Default)]
struct TokenChanges {
    map: HashMap<String, (Vec<u32>, Vec<u32>)>,
}

impl TokenChanges {
    fn add(&mut self, id: u32, tokens: Vec<String>) {
        for t in tokens {
            self.map.entry(t).or_default().0.push(id);
        }
    }

    fn remove(&mut self, id: u32, tokens: Vec<String>) {
        for t in tokens {
            self.map.entry(t).or_default().1.push(id);
        }
    }

    fn apply(&self, conn: &Connection, ns: &Namespace) -> Result<()> {
        let writes: Vec<postings::TermWrite<'_>> = self
            .map
            .iter()
            .map(|(term, (add, remove))| postings::TermWrite { term, add, remove })
            .collect();
        postings::apply_writes(conn, ns, &writes)
    }
}

/// Whether a SQLite error is a transient fault worth retrying on the read path.
fn is_retryable(e: &rusqlite::Error) -> bool {
    matches!(
        e,
        rusqlite::Error::SqliteFailure(
            rusqlite::ffi::Error {
                code: rusqlite::ErrorCode::DatabaseBusy
                    | rusqlite::ErrorCode::DatabaseLocked
                    | rusqlite::ErrorCode::SchemaChanged,
                ..
            },
            _,
        )
    )
}

/// Encode a segment's distinct token set as the contentless-mode `fwd` blob:
/// `u32 count`, then per token `u32 byte-length + bytes`.
fn encode_tokens(tokens: &[String]) -> Vec<u8> {
    let cap = 4 + tokens.iter().map(|t| 4 + t.len()).sum::<usize>();
    let mut buf = Vec::with_capacity(cap);
    buf.extend_from_slice(&(tokens.len() as u32).to_le_bytes());
    for t in tokens {
        buf.extend_from_slice(&(t.len() as u32).to_le_bytes());
        buf.extend_from_slice(t.as_bytes());
    }
    buf
}

/// Decode a `fwd` blob produced by [`encode_tokens`].
fn decode_tokens(blob: &[u8]) -> Result<Vec<String>> {
    let mut pos = 0usize;
    let read_u32 = |blob: &[u8], pos: &mut usize| -> Result<u32> {
        let end = *pos + 4;
        let slice = blob
            .get(*pos..end)
            .ok_or_else(|| Error::corrupt("truncated fwd token blob"))?;
        *pos = end;
        Ok(u32::from_le_bytes(slice.try_into().expect("4 bytes")))
    };
    let count = read_u32(blob, &mut pos)? as usize;
    // Bound the preallocation by what the remaining bytes could possibly hold (each
    // token costs at least its 4-byte length prefix), so a corrupt `count` prefix
    // cannot drive a multi-gigabyte allocation before the per-token reads error out.
    let max_possible = blob.len().saturating_sub(pos) / 4;
    let mut tokens = Vec::with_capacity(count.min(max_possible));
    for _ in 0..count {
        let len = read_u32(blob, &mut pos)? as usize;
        let end = pos + len;
        let bytes = blob
            .get(pos..end)
            .ok_or_else(|| Error::corrupt("truncated fwd token blob"))?;
        tokens.push(
            std::str::from_utf8(bytes)
                .map_err(|_| Error::corrupt("invalid utf-8 in fwd token blob"))?
                .to_string(),
        );
        pos = end;
    }
    Ok(tokens)
}
