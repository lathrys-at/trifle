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
//! use trifle::{Config, Index, Schema, SearchOpts};
//!
//! # fn main() -> trifle::Result<()> {
//! let dir = tempfile::tempdir().unwrap();
//! // Declare a schema (here: an integer key + any text field, stored), then open.
//! let index = Index::open_at(&dir.path().join("trifle.db"), Schema::flat(), Config::default())?;
//!
//! // Writes go through a short-lived writer lease; commit makes them durable + visible.
//! let mut w = index.writer()?;
//! w.insert(1, &[("front", "the quick brown fox")])?;
//! w.insert(2, &[("front", "the quack brown ox")])?;
//! w.commit()?;
//! drop(w);
//!
//! // Reads go through a reader lease. A typo'd query still matches.
//! let hits = index.reader()?.search("quikc brown", SearchOpts::new(10))?;
//! assert!(hits.iter().any(|m| m.key.as_i64() == Some(1)));
//! # Ok(())
//! # }
//! ```
//!
//! # The data model
//!
//! A [`Schema`] declares a **key** (its shape — `Integer`/`Text`/`Blob` — is the one
//! declared type, because the key is what trifle compares) plus named **text fields**.
//! A [`Document`] is a [`Key`] and a set of named segments (`label → text`); a match
//! comes back as a [`Match`] carrying the document key, the matched segment's label, the
//! matched byte span, and the segment's text (every indexed text field is stored).
//!
//! Reads and writes go through **leases** — [`Index::writer`] (the exclusive
//! single-writer lock, with commit-and-continue) and [`Index::reader`] /
//! [`Index::session`] — so coordination is structural, not conventional.
//!
//! # Ingest & maintenance
//!
//! Writes are cheap (a small delta append, `O(tokens)`) and **instantly visible** — a
//! committed write is searchable by the very next [`reader`](Index::reader). This
//! supports both write-infrequent and continual-drip ingest; "write-infrequent" is about
//! *fold amortization*, not write capability. Fold pending deltas into the base postings
//! with [`compact`](Index::compact) on a cadence driven by
//! [`Stats::delta_backlog`](Stats); [`rebuild`](Index::rebuild) fully reindexes via an
//! atomic shadow swap (required after a tokenizer or `data_version`/schema change, which
//! drop the cache on open). A sustained *high-rate* ingest regime would want an
//! LSM-structured posting store — a documented future path, not part of v0.2.
//!
//! # What trifle leaves to the caller
//!
//! Embeddings/semantic search, fusion (RRF) with other signals, an exact precision
//! tier beyond a custom [`Ranker`], sub-trigram (`<3`-char) query
//! handling, and deciding *when* the cache is stale relative to the source of truth.

#![forbid(unsafe_op_in_unsafe_fn)]

mod instrument;

pub mod error;
pub mod rank;
pub mod store;
pub mod tokenize;

mod dict;
mod model;
mod postings;
mod schema;
mod select;
mod term;
mod welford;

use std::borrow::Borrow;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::Path;
use std::time::Duration;

use roaring::RoaringBitmap;
use rusqlite::types::Value;
use rusqlite::{Connection, OptionalExtension};

/// Re-export of the `rusqlite` version trifle is built against, so consumers
/// implementing a custom [`store::Backend`] (or constructing a [`store::Shared`])
/// use exactly the right `Connection` type.
pub use rusqlite;

use dict::{Dictionary, TermId};
pub use error::{Error, Result};
pub use model::{CmpOp, Document, Filter, FilterType, Key, KeyShape, Match, Schema, SchemaBuilder};
use rank::{Bm25Ranker, Candidates, OverlapRanker, QueryContext, Ranker, Survivor, overlap_search};
use schema::SCHEMA_VERSION;
use select::{SelectParams, select};
use store::{Backend, Namespace, Sidecar};
pub use term::{IntoTerm, Term};
use tokenize::{DefaultTokenizer, Tokenizer};
use welford::ClassSnap;

/// Default match floor `m` — shared rare tokens required for a hit.
const DEFAULT_MIN_SHARED: u32 = 2;
/// Per-typo token damage `d`; the typo floor is `F = m + d`.
const TYPO_DAMAGE: u32 = 4;
/// Default `t_max` — the number of rarest query tokens selection keeps.
const DEFAULT_T_MAX: usize = 12;
/// How many times a read retries a transient `SQLITE_BUSY`/`LOCKED`/`SCHEMA`.
const RETRY_MAX: usize = 5;

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
}

impl Config {
    /// A configuration with the given drift token and otherwise default settings.
    pub fn new(data_version: u64) -> Self {
        Config { data_version }
    }
}

// The data model — `Document`, `Match`, `Key`, `Schema`, … — lives in `model` and is
// re-exported at the crate root.

/// A scope/exclusion predicate over a candidate's provenance:
/// `(key, label) -> keep`. Used for [`SearchOpts::scope`].
///
/// The `'a` lifetime lets the predicate borrow local state (e.g. an allow-set on the
/// stack); a `dyn Fn` alias without it would force `'static` and reject such closures.
pub type ScopeFn<'a> = dyn Fn(&Key, &str) -> bool + 'a;

/// How hard the ranker tries: how deep a candidate pool to over-fetch and rerank.
///
/// Candidate generation by bit-sliced overlap is cheap but orders only coarsely (by
/// shared-token count); the precision tier — idf weighting, length normalization, and
/// literal verification (the [`Bm25Ranker`]) — reorders a *pool* of
/// the top candidates. The pool must be deeper than `limit` to recover a relevant
/// document that overlap alone ranked past `limit`, and empirically the depth needed for
/// a given recall scales as **`c·√(limit · N)`** (N = indexed segments) — a power law in
/// corpus size, not a constant. Each level fixes `c` to hit a fraction of the
/// deep-pool recall **ceiling** (calibrated on MS MARCO; see `benchmarks/`):
///
/// | level | `c` | ≈ recall vs ceiling | cost |
/// |-------|-----|---------------------|------|
/// | [`None`](Effort::None)     | 0    | overlap order only | cheapest (pool = `limit`) |
/// | [`Low`](Effort::Low)       | 0.03 | ~50% | |
/// | [`Medium`](Effort::Medium) | 0.05 | ~90% | **the default** |
/// | [`High`](Effort::High)     | 0.10 | ~95% | |
/// | [`Max`](Effort::Max)       | 0.45 | ~99% (saturation tail) | deepest |
///
/// Deeper pool → more hydration + scoring, so latency grows ~linearly with the pool
/// while recall grows logarithmically. The pool is always at least `limit`.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum Effort {
    /// No reranking: return the bit-sliced overlap order (pool = `limit`). Cheapest.
    None,
    /// ~50% of the recall ceiling (`c = 0.03`).
    Low,
    /// ~90% of the recall ceiling (`c = 0.05`). The default.
    Medium,
    /// ~95% of the recall ceiling (`c = 0.10`).
    High,
    /// ~99% of the recall ceiling (`c = 0.45`) — the flat saturation tail; large pool
    /// for the last few points.
    Max,
    /// An explicit coefficient `c` in `pool = max(limit, round(c·√(limit·N)))`. `0`
    /// disables reranking, as [`None`](Effort::None).
    Custom(f64),
}

impl Default for Effort {
    /// [`Medium`](Effort::Medium) — reranking on, ~90% of the recall ceiling.
    fn default() -> Self {
        Effort::Medium
    }
}

impl Effort {
    /// The pool coefficient `c`.
    fn coeff(self) -> f64 {
        match self {
            Effort::None => 0.0,
            Effort::Low => 0.03,
            Effort::Medium => 0.05,
            Effort::High => 0.10,
            Effort::Max => 0.45,
            Effort::Custom(c) => c.max(0.0),
        }
    }

    /// Whether reranking is active (a precision tier runs over an over-fetched pool).
    fn reranks(self) -> bool {
        self.coeff() > 0.0
    }

    /// The rerank pool depth for a result `limit` over `n_segments` indexed segments:
    /// `max(limit, round(c·√(limit·n_segments)))`.
    fn pool(self, limit: usize, n_segments: u64) -> usize {
        let c = self.coeff();
        if c == 0.0 || limit == 0 {
            return limit;
        }
        let p = (c * ((limit as f64) * (n_segments as f64)).sqrt()).round();
        limit.max(p as usize)
    }
}

/// Per-search options.
///
/// Construct with [`SearchOpts::new`] and the builder setters, then set any of the public
/// fields on the returned value. Most searches only set [`min_shared`](SearchOpts::min_shared)
/// (`m`, the strictness dial); [`t_max`](SearchOpts::t_max) is the recall axis (how many
/// rarest query tokens selection keeps). `#[non_exhaustive]`: more knobs may be added, so
/// build from [`new`](SearchOpts::new) rather than a struct literal.
#[non_exhaustive]
pub struct SearchOpts<'a> {
    /// Maximum number of matches to return (top-k).
    pub limit: usize,
    /// `m` — the match floor (shared rare tokens for a hit). `None` → `2`.
    pub min_shared: Option<u32>,
    /// `t_max` — the number of rarest query tokens selection keeps, above the typo
    /// floor `F = m + d`. `None` → `12`. This is the selection breadth knob: more tokens
    /// means more candidates and more posting rows scanned. A `t_max` below `F` is
    /// raised to `F`.
    pub t_max: Option<usize>,
    /// A per-query [`Ranker`]. `None` → the built-in [`OverlapRanker`] (when reranking is
    /// off) or [`Bm25Ranker`] (the [`Effort`] precision tier).
    pub ranker: Option<&'a dyn Ranker>,
    /// A membership predicate `(key, label) -> keep` evaluated over candidates in
    /// overlap order — never over the corpus. The walk continues until `limit`
    /// predicate-passing results lock, so scoping needs no over-fetch. The predicate must
    /// not call back into this index's writer.
    pub scope: Option<&'a ScopeFn<'a>>,
    /// How hard to rerank — the over-fetch [`Effort`]. Defaults to
    /// [`Medium`](Effort::Medium). [`None`](Effort::None) disables reranking and returns
    /// plain overlap order (cheapest). When `ranker` is also set, `effort` still chooses
    /// the pool depth but that custom ranker scores it.
    pub effort: Effort,
    /// An optional [`Filter`] over the schema's **filterable** fields (Tier 2). Applied as
    /// a doc-id intersection during candidate generation — it prunes before rerank and
    /// hydration, but does not save the overlap work itself.
    pub filter: Option<&'a Filter>,
}

impl<'a> SearchOpts<'a> {
    /// Options for the given result limit, everything else default (reranking at
    /// [`Effort::Medium`]).
    pub fn new(limit: usize) -> Self {
        SearchOpts {
            limit,
            min_shared: None,
            t_max: None,
            ranker: None,
            scope: None,
            effort: Effort::default(),
            filter: None,
        }
    }

    /// Set a [`Filter`] over the schema's filterable fields.
    pub fn filter(mut self, filter: &'a Filter) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Set the match floor `m`.
    pub fn min_shared(mut self, m: u32) -> Self {
        self.min_shared = Some(m);
        self
    }

    /// Set `t_max` — the number of rarest query tokens selection keeps.
    pub fn t_max(mut self, t_max: usize) -> Self {
        self.t_max = Some(t_max);
        self
    }

    /// Set a per-query ranker.
    pub fn ranker(mut self, ranker: &'a dyn Ranker) -> Self {
        self.ranker = Some(ranker);
        self
    }

    /// Set the rerank [`Effort`] (pool depth + whether the precision tier runs).
    pub fn rerank(mut self, effort: Effort) -> Self {
        self.effort = effort;
        self
    }

    /// Set a scope predicate.
    pub fn scope(mut self, scope: &'a ScopeFn<'a>) -> Self {
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
    /// The schema fingerprint currently stamped — the semantic identity of the declared
    /// data model (folded into the drift check).
    pub schema_fingerprint: u64,
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

/// An embedded fuzzy-search index over text segments.
///
/// Generic over the [`Tokenizer`] (monomorphized — it is on the
/// hot path) and the storage [`Backend`]. Both default, so the common
/// case is just `Index`. Open with [`open_at`](Index::open_at) (an owned sidecar
/// file) or [`open`](Index::open) (any backend).
///
/// All methods are synchronous and thread-safe (`&self`): a single internal writer
/// is serialized, reads run on a pooled connection concurrently with the writer
/// under WAL. A caller on an async runtime dispatches calls to a blocking pool.
pub struct Index<T: Tokenizer = DefaultTokenizer, B: Backend = Sidecar> {
    backend: B,
    tokenizer: T,
    data_version: u64,
    /// The declared data model (key shape, text fields, filterable columns).
    schema: Schema,
    /// The in-memory faulting term dictionary (gram → `u32` id), shared by the writer
    /// and the read pool. Hydrated on open; rebuilt under the swap on `rebuild`.
    dict: Dictionary,
}

impl Index<DefaultTokenizer, Sidecar> {
    /// Open (creating if absent) an index at `path` with the given [`Schema`], the
    /// default trigram tokenizer, and an owned [`Sidecar`] file. The common case.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened or the store cannot be
    /// initialized.
    pub fn open_at(path: &Path, schema: Schema, config: Config) -> Result<Self> {
        let backend = Sidecar::open(path)?;
        Index::open(backend, DefaultTokenizer::new(), schema, config)
    }
}

impl<T: Tokenizer, B: Backend> Index<T, B> {
    /// Open an index over a given backend and tokenizer.
    ///
    /// On open, the store is created if absent and checked against three version
    /// stamps (schema, tokenizer fingerprint, caller `data_version`); any mismatch —
    /// or a broken id-allocation invariant (a segment id at or past the allocator's
    /// high-water mark) — drops the cache to empty (no migrations). After such a reset,
    /// [`rebuild`](Self::rebuild) repopulates it.
    ///
    /// # Errors
    ///
    /// Returns an error if the store cannot be initialized.
    pub fn open(backend: B, tokenizer: T, schema: Schema, config: Config) -> Result<Self> {
        let index = Index {
            backend,
            tokenizer,
            data_version: config.data_version,
            schema,
            dict: Dictionary::empty(),
        };
        index.init()?;
        // Hydrate the in-memory dictionary from the (possibly just-reset) `dict` table.
        // Done after `init` commits, so a drift reset hydrates an empty map.
        {
            let conn = index.backend.read()?;
            index.dict.load(&conn, index.backend.namespace())?;
        }
        Ok(index)
    }

    /// Create tables and reconcile drift, all in one transaction.
    fn init(&self) -> Result<()> {
        let mut guard = self.backend.write()?;
        let ns = self.backend.namespace();
        let tx = guard.transaction()?;
        schema::drop_shadows(&tx, ns)?;
        schema::create_tables(&tx, ns, &self.schema)?;

        let stamps = schema::read_stamps(&tx, ns)?;
        let fingerprint = self.tokenizer.fingerprint();
        let schema_fp = self.schema.fingerprint();
        let drift = stamps.schema_version != Some(SCHEMA_VERSION)
            || stamps.fingerprint != Some(fingerprint)
            || stamps.data_version != Some(self.data_version)
            || stamps.schema_fingerprint != Some(schema_fp);
        let desync = !drift && self.desync(&tx, ns)?;

        if drift || desync {
            schema::reset(&tx, ns, &self.schema)?;
            schema::set_next_id(&tx, ns, 1)?;
            // A reset empties the dictionary and so reassigns the (now empty) id space;
            // bump the generation so any concurrent reader detects the change.
            schema::bump_dict_generation(&tx, ns)?;
            schema::write_stamps(&tx, ns, self.data_version, fingerprint, schema_fp)?;
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

    // ----- leases -------------------------------------------------------------

    /// Acquire the exclusive [`Writer`] lease: holds the single-writer lock for the
    /// lease's lifetime and opens a transaction. The six write methods live on the
    /// returned guard. Keep it **short-lived** — acquire → batch → [`commit`](Writer::commit)
    /// → drop; never `.await` unrelated work while holding it. **Dropping without
    /// committing rolls back the uncommitted tail** (so a partial batch never persists).
    ///
    /// # Errors
    ///
    /// Returns an error if the write transaction cannot begin.
    pub fn writer(&self) -> Result<Writer<'_, T, B>> {
        Writer::begin(self)
    }

    /// Acquire a [`Reader`] lease for issuing searches. Each search runs under its own
    /// consistent WAL snapshot; acquire a fresh reader to see newer writes.
    ///
    /// # Errors
    ///
    /// Returns an error if the lease cannot be set up.
    pub fn reader(&self) -> Result<Reader<'_, T, B>> {
        Ok(Reader { index: self })
    }

    /// Acquire a [`SearchSession`] — like a [`Reader`] but holding a **warm** pooled
    /// connection across calls, the right shape for an as-you-type burst (no per-search
    /// checkout). Drop and re-acquire to see newer writes.
    ///
    /// # Errors
    ///
    /// Returns an error if a read connection cannot be acquired.
    pub fn session(&self) -> Result<SearchSession<'_, T, B>> {
        let conn = self.backend.read()?;
        Ok(SearchSession { index: self, conn })
    }

    // ----- write internals (used by the `Writer` lease) -----------------------

    /// Find the internal doc id for `key`, creating a fresh `doc` row if absent and
    /// `create` is set (otherwise `None`).
    fn doc_id_for(
        &self,
        conn: &Connection,
        ns: &Namespace,
        key: &Key,
        create: bool,
    ) -> Result<Option<i64>> {
        // The key variant must match the schema's declared shape; otherwise SQLite affinity
        // silently coerces it (e.g. an Integer key into a Text column comes back as a string
        // key), a caller footgun. Debug-asserted (a caller contract, not a recoverable input
        // error).
        debug_assert!(
            matches!(
                (key, self.schema.key_shape()),
                (Key::Integer(_), KeyShape::Integer)
                    | (Key::Text(_), KeyShape::Text)
                    | (Key::Blob(_), KeyShape::Blob)
            ),
            "key variant {key:?} does not match the schema's declared key shape {:?}",
            self.schema.key_shape()
        );
        let found: Option<i64> = conn
            .query_row(
                &format!("SELECT id FROM {} WHERE key = ?1", ns.doc()),
                rusqlite::params![key.to_value()],
                |r| r.get(0),
            )
            .optional()?;
        if let Some(id) = found {
            return Ok(Some(id));
        }
        if !create {
            return Ok(None);
        }
        conn.execute(
            &format!("INSERT INTO {}(key) VALUES(?1)", ns.doc()),
            rusqlite::params![key.to_value()],
        )?;
        Ok(Some(conn.last_insert_rowid()))
    }

    /// The `(seg_id, interned term-id set)` of every segment under an internal doc id,
    /// read from the `fwd` index (delete needs neither the text nor the tokenizer).
    fn doc_segments(
        &self,
        conn: &Connection,
        ns: &Namespace,
        doc: i64,
    ) -> Result<Vec<(u32, Vec<TermId>)>> {
        let ids: Vec<u32> = {
            let mut stmt =
                conn.prepare_cached(&format!("SELECT id FROM {} WHERE doc_id = ?1", ns.seg()))?;
            let mut rows = stmt.query(rusqlite::params![doc])?;
            let mut v = Vec::new();
            while let Some(r) = rows.next()? {
                v.push(r.get::<_, i64>(0)? as u32);
            }
            v
        };
        let fwd = self.read_fwd(conn, ns, &ids)?;
        Ok(ids
            .into_iter()
            .map(|id| (id, fwd.get(&id).cloned().unwrap_or_default()))
            .collect())
    }

    /// Delete the `seg` and `fwd` rows of an internal doc id (not the `doc` row).
    fn delete_doc_rows(&self, conn: &Connection, ns: &Namespace, doc: i64) -> Result<()> {
        conn.execute(
            &format!(
                "DELETE FROM {} WHERE id IN (SELECT id FROM {} WHERE doc_id = ?1)",
                ns.fwd(),
                ns.seg()
            ),
            rusqlite::params![doc],
        )?;
        conn.execute(
            &format!("DELETE FROM {} WHERE doc_id = ?1", ns.seg()),
            rusqlite::params![doc],
        )?;
        Ok(())
    }

    /// Tokenize `text`, dedupe to its distinct grams, and resolve each to a term-id via
    /// `assign` — the single lowering shared by the incremental write path
    /// ([`write_segment`](Self::write_segment), `assign` = stage intern) and
    /// [`rebuild`](Self::rebuild) (`assign` = a dense local allocator). Owning the
    /// tokenize + 3-codepoint ceiling check in one place keeps the two paths from drifting
    /// (audit I4).
    fn distinct_term_ids(
        &self,
        text: &str,
        mut assign: impl FnMut(Term) -> Result<TermId>,
    ) -> Result<Vec<TermId>> {
        let distinct: HashSet<T::Token> = self.tokenizer.tokenize(text).collect();
        let mut ids: Vec<TermId> = Vec::with_capacity(distinct.len());
        for tok in &distinct {
            let term = tok.term().ok_or_else(|| {
                Error::InvalidInput(format!(
                    "gram {:?} exceeds the 3-codepoint term-encoding ceiling",
                    tok.borrow()
                ))
            })?;
            ids.push(assign(term)?);
        }
        Ok(ids)
    }

    /// Write one segment `(doc, label) = text`, interning its grams through `stage` and
    /// storing the term-id set in `fwd` (so delete needs no text). If a segment with this
    /// label already exists: with `replace`, it is dropped first (its removals accumulated
    /// into `changes`); without `replace`, this errors.
    #[allow(clippy::too_many_arguments)]
    fn write_segment(
        &self,
        conn: &Connection,
        ns: &Namespace,
        doc: i64,
        label: &str,
        text: &str,
        changes: &mut TokenChanges,
        stage: &mut dict::InternStage<'_>,
        replace: bool,
    ) -> Result<()> {
        if !self.schema.accepts_label(label) {
            return Err(Error::InvalidInput(format!(
                "label {label:?} is not a field of the schema"
            )));
        }
        if let Some(seg_id) = self.find_segment(conn, ns, doc, label)? {
            if !replace {
                return Err(Error::InvalidInput(format!(
                    "a segment with label {label:?} already exists for this key"
                )));
            }
            self.drop_segment(conn, ns, seg_id, changes)?;
        }
        let id = schema::alloc_ids(conn, ns, 1)?;
        // The segment's gram length (with repetition) for BM25+ length normalization.
        let seg_len = self.tokenizer.tokenize(text).count() as i64;
        conn.execute(
            &format!(
                "INSERT INTO {}(id, doc_id, label, txt, len) VALUES(?1, ?2, ?3, ?4, ?5)",
                ns.seg()
            ),
            rusqlite::params![id, doc, label, text, seg_len],
        )?;
        // Intern straight from the tokens (`token.term()`), not via stringified grams — the
        // blanket `IntoTerm` packs each token with no per-token `String` allocation.
        let ids = self.distinct_term_ids(text, |term| stage.intern_term(term, conn, ns))?;
        let bm: RoaringBitmap = ids.iter().copied().collect();
        conn.execute(
            &format!("INSERT INTO {}(id, tokens) VALUES(?1, ?2)", ns.fwd()),
            rusqlite::params![id, postings::serialize(&bm)?],
        )?;
        changes.add(id as u32, &ids);
        schema::bump_seg_stats(conn, ns, 1, seg_len)?;
        Ok(())
    }

    /// The segment id for `(doc, label)`, if it exists.
    fn find_segment(
        &self,
        conn: &Connection,
        ns: &Namespace,
        doc: i64,
        label: &str,
    ) -> Result<Option<i64>> {
        Ok(conn
            .query_row(
                &format!(
                    "SELECT id FROM {} WHERE doc_id = ?1 AND label = ?2",
                    ns.seg()
                ),
                rusqlite::params![doc, label],
                |r| r.get(0),
            )
            .optional()?)
    }

    /// Drop one segment by id: accumulate its term-id removals into `changes`, then
    /// delete its `seg` and `fwd` rows.
    fn drop_segment(
        &self,
        conn: &Connection,
        ns: &Namespace,
        seg_id: i64,
        changes: &mut TokenChanges,
    ) -> Result<()> {
        let fwd = self.read_fwd(conn, ns, &[seg_id as u32])?;
        if let Some(ids) = fwd.get(&(seg_id as u32)) {
            changes.remove(seg_id as u32, ids);
        }
        // Subtract this segment's gram length from the rolling BM25 stats before deleting.
        let seg_len: i64 = conn
            .query_row(
                &format!("SELECT len FROM {} WHERE id = ?1", ns.seg()),
                rusqlite::params![seg_id],
                |r| r.get(0),
            )
            .optional()?
            .unwrap_or(0);
        conn.execute(
            &format!("DELETE FROM {} WHERE id = ?1", ns.fwd()),
            rusqlite::params![seg_id],
        )?;
        conn.execute(
            &format!("DELETE FROM {} WHERE id = ?1", ns.seg()),
            rusqlite::params![seg_id],
        )?;
        schema::bump_seg_stats(conn, ns, -1, -seg_len)?;
        Ok(())
    }

    /// Remove the segment `(doc, label)` if present (its removals into `changes`); a
    /// no-op if absent. The `doc` row is left intact (it may have other segments).
    fn remove_one_segment(
        &self,
        conn: &Connection,
        ns: &Namespace,
        doc: i64,
        label: &str,
        changes: &mut TokenChanges,
    ) -> Result<()> {
        if let Some(seg_id) = self.find_segment(conn, ns, doc, label)? {
            self.drop_segment(conn, ns, seg_id, changes)?;
        }
        Ok(())
    }

    /// Set a document's filterable column values from `fields` (Tier 2); names not
    /// declared filterable are skipped. `doc_table` is the live or shadow `doc` table.
    fn set_doc_fields(
        &self,
        conn: &Connection,
        doc_table: &str,
        doc_id: i64,
        fields: &[(String, Value)],
    ) -> Result<()> {
        let mut sets = Vec::new();
        let mut params: Vec<Value> = Vec::new();
        for (name, value) in fields {
            if let Ok(col) = self.schema.filter_column(name) {
                sets.push(format!("\"{col}\" = ?")); // double-quoted: keyword-safe column
                params.push(value.clone());
            }
        }
        if sets.is_empty() {
            return Ok(());
        }
        params.push(Value::Integer(doc_id));
        let sql = format!("UPDATE {doc_table} SET {} WHERE id = ?", sets.join(", "));
        conn.execute(&sql, rusqlite::params_from_iter(params))?;
        Ok(())
    }

    /// The internal doc ids matching a structured [`Filter`] over the schema's filterable
    /// columns (Tier 2). Field names are validated against the schema (the injection
    /// guard); a value-less / unsatisfiable filter yields an empty set.
    fn filter_docs(
        &self,
        conn: &Connection,
        ns: &Namespace,
        filter: &Filter,
    ) -> Result<RoaringBitmap> {
        let (where_sql, params) = filter.compile(&self.schema)?;
        let sql = format!("SELECT id FROM {} WHERE {}", ns.doc(), where_sql);
        let mut stmt = conn.prepare(&sql)?;
        let mut rows = stmt.query(rusqlite::params_from_iter(params))?;
        let mut bm = RoaringBitmap::new();
        while let Some(r) = rows.next()? {
            bm.insert(r.get::<_, i64>(0)? as u32);
        }
        Ok(bm)
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
    pub fn rebuild(&self, corpus: impl IntoIterator<Item = Document>) -> Result<()> {
        let mut guard = self.backend.write()?;
        let ns = self.backend.namespace();
        let tx = guard.transaction()?;
        schema::create_shadows(&tx, ns, &self.schema)?;

        // Accumulate the inverted index in memory while streaming doc/seg rows to the
        // shadow tables. Three id spaces are reassigned dense (doc, segment, term); the
        // inverted index is keyed by segment id and the dictionary by the gram's packed
        // u128 (stored as the 16-byte big-endian blob).
        let mut local: HashMap<u128, TermId> = HashMap::new();
        let mut next_term: u64 = 1;
        let mut inverted: HashMap<TermId, RoaringBitmap> = HashMap::new();
        let mut next_doc: i64 = 1;
        let mut next_seg: i64 = 1;
        let mut total_seg_len: i64 = 0;
        {
            let mut doc_ins = tx.prepare_cached(&format!(
                "INSERT INTO {}(id, key) VALUES(?1, ?2)",
                ns.doc_shadow()
            ))?;
            let mut seg_ins = tx.prepare_cached(&format!(
                "INSERT INTO {}(id, doc_id, label, txt, len) VALUES(?1, ?2, ?3, ?4, ?5)",
                ns.seg_shadow()
            ))?;
            let mut fwd_ins = tx.prepare_cached(&format!(
                "INSERT INTO {}(id, tokens) VALUES(?1, ?2)",
                ns.fwd_shadow()
            ))?;
            for doc in corpus {
                let doc_id = next_doc;
                next_doc += 1;
                doc_ins.execute(rusqlite::params![doc_id, doc.key.to_value()])?;
                if !doc.payload.is_empty() {
                    self.set_doc_fields(&tx, ns.doc_shadow(), doc_id, &doc.payload)?;
                }
                for (label, text) in &doc.segments {
                    if !self.schema.accepts_label(label) {
                        return Err(Error::InvalidInput(format!(
                            "label {label:?} is not a field of the schema"
                        )));
                    }
                    let seg_id = next_seg;
                    next_seg += 1;
                    let seg_len = self.tokenizer.tokenize(text).count() as i64;
                    total_seg_len += seg_len;
                    seg_ins.execute(rusqlite::params![
                        seg_id,
                        doc_id,
                        label,
                        text.as_str(),
                        seg_len
                    ])?;
                    // Same lowering as the incremental path, but assigning dense ids from a
                    // rebuild-local allocator (see `distinct_term_ids` / audit I4).
                    let ids = self.distinct_term_ids(text, |term| {
                        let gkey = term.0;
                        if let Some(&t) = local.get(&gkey) {
                            return Ok(t);
                        }
                        if next_term == 0 || next_term > u32::MAX as u64 {
                            return Err(Error::corrupt("term id space exhausted"));
                        }
                        let t = next_term as TermId;
                        next_term += 1;
                        local.insert(gkey, t);
                        Ok(t)
                    })?;
                    for &tid in &ids {
                        inverted.entry(tid).or_default().insert(seg_id as u32);
                    }
                    let bm: RoaringBitmap = ids.iter().copied().collect();
                    fwd_ins.execute(rusqlite::params![seg_id, postings::serialize(&bm)?])?;
                }
            }
        }

        // Persist the rebuilt dictionary (id -> gram) into the shadow table.
        {
            let mut dict_ins = tx.prepare_cached(&format!(
                "INSERT INTO {}(id, gram) VALUES(?1, ?2)",
                ns.dict_shadow()
            ))?;
            for (gkey, id) in &local {
                dict_ins.execute(rusqlite::params![*id as i64, gkey.to_be_bytes().as_slice()])?;
            }
        }

        postings::write_base_postings(
            &tx,
            ns.post_shadow(),
            ns.term_shadow(),
            inverted.iter().map(|(id, bm)| (*id, bm)),
        )?;

        schema::swap_shadows(&tx, ns, &self.schema)?;
        schema::set_next_id(&tx, ns, next_seg)?;
        // The rolling BM25 stats for the freshly-built corpus (segments = next_seg - 1).
        schema::set_seg_stats(&tx, ns, next_seg - 1, total_seg_len)?;
        // Reassigning the term-id space bumps the generation so a concurrent reader
        // detects the change and re-resolves against the new snapshot.
        schema::bump_dict_generation(&tx, ns)?;
        schema::write_stamps(
            &tx,
            ns,
            self.data_version,
            self.tokenizer.fingerprint(),
            self.schema.fingerprint(),
        )?;
        tx.commit()?;
        // Re-hydrate the in-memory dictionary from the swapped-in tables, still under the
        // held write lease — so no reader splices the old map onto the new snapshot.
        self.dict.load(&guard, ns)?;
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
        // One pinned snapshot so the reported fields are mutually consistent (audit RA-2),
        // and segment count is the O(1) rolling meta counter, not an O(N) `count(*)` (I3).
        let tx = conn.unchecked_transaction()?;
        let (seg_count, _) = schema::read_seg_stats(&tx, ns)?;
        let segments = seg_count.max(0) as u64;
        let page_count: i64 = tx.query_row("PRAGMA page_count", [], |r| r.get(0))?;
        let page_size: i64 = tx.query_row("PRAGMA page_size", [], |r| r.get(0))?;
        let stamps = schema::read_stamps(&tx, ns)?;
        Ok(Stats {
            segments,
            terms: postings::term_count(&tx, ns)?,
            delta_backlog: postings::delta_backlog(&tx, ns)?,
            disk_bytes: (page_count * page_size).max(0) as u64,
            data_version: stamps.data_version.unwrap_or(self.data_version),
            tokenizer_fingerprint: stamps
                .fingerprint
                .unwrap_or_else(|| self.tokenizer.fingerprint()),
            schema_version: stamps.schema_version.unwrap_or(SCHEMA_VERSION),
            schema_fingerprint: stamps
                .schema_fingerprint
                .unwrap_or_else(|| self.schema.fingerprint()),
        })
    }

    // ----- reads (implementation; the public read surface is `Reader`) --------

    /// Search a batch of queries, one result list per query in order. Invoked by a
    /// [`Reader`]; not public on `Index` (reads go through a [`reader`](Self::reader)
    /// lease). Each query's selection and ranking derive only from its own token
    /// frequencies, so a batch ranks each query identically to a singleton search
    /// (batch == serial).
    fn search_batch(&self, queries: &[&str], opts: &SearchOpts<'_>) -> Result<Vec<Vec<Match>>> {
        if queries.is_empty() {
            return Ok(Vec::new());
        }
        let (query_tokens, all_grams) = Self::query_grams(queries, |q| self.distinct_tokens(q));
        // Per-search checkout + snapshot, then the shared search body.
        self.search_read(&all_grams, |conn, ns, resolved, class_snap| {
            self.run_search(conn, ns, resolved, class_snap, queries, &query_tokens, opts)
        })
    }

    /// The distinct tokens per query and the batch-wide distinct gram set (resolution
    /// input). Factored so the [`SearchSession`] reuses it.
    fn query_grams(
        queries: &[&str],
        tokenize: impl Fn(&str) -> Vec<String>,
    ) -> (Vec<Vec<String>>, Vec<String>) {
        let query_tokens: Vec<Vec<String>> = queries.iter().map(|q| tokenize(q)).collect();
        let all_grams: Vec<String> = query_tokens
            .iter()
            .flat_map(|q| q.iter().cloned())
            .collect::<BTreeSet<String>>()
            .into_iter()
            .collect();
        (query_tokens, all_grams)
    }

    /// The per-snapshot search body: given the resolved gram→id map and class snapshot,
    /// read dfs/postings, select, generate candidates, apply the filter, rank, hydrate.
    /// Shared by the [`Reader`] (per-search checkout) and the warm [`SearchSession`]
    /// (held connection).
    #[allow(clippy::too_many_arguments)]
    fn run_search(
        &self,
        conn: &Connection,
        ns: &Namespace,
        resolved: &HashMap<String, TermId>,
        class_snap: &ClassSnap,
        queries: &[&str],
        query_tokens: &[Vec<String>],
        opts: &SearchOpts<'_>,
    ) -> Result<Vec<Vec<Match>>> {
        let min_shared = opts.min_shared.unwrap_or(DEFAULT_MIN_SHARED).max(1);
        let sel_params = SelectParams {
            min_shared,
            typo_damage: TYPO_DAMAGE,
            t_max: opts.t_max.unwrap_or(DEFAULT_T_MAX),
        };
        // The reranker: an explicit `opts.ranker` wins; otherwise the precision tier
        // ([`Bm25Ranker`]) when reranking is active, else plain overlap order.
        let overlap = OverlapRanker;
        let bm25 = Bm25Ranker;
        let default_ranker: &dyn Ranker = if opts.effort.reranks() {
            &bm25
        } else {
            &overlap
        };
        let ranker: &dyn Ranker = opts.ranker.unwrap_or(default_ranker);

        // One batched frequency read over every resolved term-id in the batch.
        let all_ids: Vec<TermId> = resolved
            .values()
            .copied()
            .collect::<BTreeSet<TermId>>()
            .into_iter()
            .collect();
        let dfs = postings::read_dfs(conn, ns, &all_ids)?;
        // A gram's df: 0 if it resolved to no id (absent token) or its id has no live df
        // row — exactly the existing df-0 behavior.
        let df_of = |t: &str| -> i64 {
            resolved
                .get(t)
                .and_then(|id| dfs.get(id))
                .copied()
                .unwrap_or(0)
        };

        let selected_per: Vec<Vec<String>> = query_tokens
            .iter()
            .map(|q| {
                let triples: Vec<(String, i64, u8)> = q
                    .iter()
                    .map(|t| {
                        let class = term::encode_term(t).map(|tm| tm.class()).unwrap_or(0);
                        (t.clone(), df_of(t), class)
                    })
                    .collect();
                select(&triples, sel_params, class_snap)
            })
            .collect();
        let sel_ids: Vec<TermId> = selected_per
            .iter()
            .flat_map(|s| s.iter())
            .filter_map(|t| resolved.get(t.as_str()).copied())
            .collect::<BTreeSet<TermId>>()
            .into_iter()
            .collect();
        let postings_map = postings::effective_postings(conn, ns, &sel_ids)?;

        // Corpus size + average segment length from the O(1) rolling meta counters (audit
        // I3), read under this search's snapshot. `avgdl` feeds BM25+ length normalization.
        let (seg_count, seg_len_sum) = schema::read_seg_stats(conn, ns)?;
        let n_segments = seg_count.max(0) as u64;
        let avgdl = if seg_count > 0 {
            seg_len_sum as f64 / seg_count as f64
        } else {
            0.0
        };
        let pool = opts.effort.pool(opts.limit, n_segments);

        // Tier-2 filter: the doc ids matching the structured filter (same per batch).
        let filter_docs = match opts.filter {
            Some(f) => Some(self.filter_docs(conn, ns, f)?),
            None => None,
        };

        let mut out = Vec::with_capacity(queries.len());
        for (qi, query) in queries.iter().enumerate() {
            let selected = &selected_per[qi];
            let present: Vec<(&str, &RoaringBitmap)> = selected
                .iter()
                .filter_map(|t| {
                    resolved
                        .get(t.as_str())
                        .and_then(|id| postings_map.get(id))
                        .map(|bm| (t.as_str(), bm))
                })
                .collect();

            let mut survivors = overlap_search(
                conn,
                ns,
                &present,
                pool,
                min_shared,
                self.schema.key_shape(),
                filter_docs.as_ref(),
                opts.scope,
            )?;
            // Every indexed field is stored, so a match always carries its text and the
            // reranker always sees it (no opt-out that could leave BM25 textless).
            self.hydrate_text(conn, ns, &mut survivors)?;

            out.push(self.rank_to_matches(
                &survivors, &present, selected, query, min_shared, ranker, opts.limit, n_segments,
                avgdl,
            ));
        }
        Ok(out)
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
        n_segments: u64,
        avgdl: f64,
    ) -> Vec<Match> {
        let candidates = Candidates::new(survivors, present);
        let qctx = QueryContext {
            query,
            selected,
            min_shared,
            n_segments,
            avgdl,
        };
        let ranked = ranker.rank(&candidates, &qctx);
        let sel_refs: Vec<&str> = selected.iter().map(String::as_str).collect();
        let mut matches = Vec::with_capacity(ranked.len().min(limit));
        for r in ranked.into_iter().take(limit) {
            let Some(s) = survivors.get(r.candidate) else {
                continue;
            };
            let span = self.tokenizer.span(&s.text, &sel_refs);
            matches.push(Match {
                key: s.key.clone(),
                label: s.label.clone(),
                span,
                text: s.text.clone(),
            });
        }
        matches
    }

    /// Hydrate each survivor's text from `seg.txt` in one batched read (`WHERE id IN
    /// rarray`). Every indexed field is stored, so every survivor gets its segment text.
    fn hydrate_text(
        &self,
        conn: &Connection,
        ns: &Namespace,
        survivors: &mut [Survivor],
    ) -> Result<()> {
        if survivors.is_empty() {
            return Ok(());
        }
        let ids: Vec<u32> = survivors.iter().map(|s| s.seg_id).collect();
        let arr: std::rc::Rc<Vec<Value>> =
            std::rc::Rc::new(ids.iter().map(|&i| Value::Integer(i as i64)).collect());
        let sql = format!(
            "SELECT id, txt, len FROM {} WHERE id IN rarray(?1)",
            ns.seg()
        );
        let mut texts: HashMap<u32, (String, u32)> = HashMap::with_capacity(ids.len());
        {
            let mut stmt = conn.prepare_cached(&sql)?;
            let mut rows = stmt.query(rusqlite::params![arr])?;
            while let Some(r) = rows.next()? {
                let id = r.get::<_, i64>(0)? as u32;
                let txt = r.get::<_, String>(1)?;
                let len = r.get::<_, i64>(2)?.max(0) as u32;
                texts.insert(id, (txt, len));
            }
        }
        for s in survivors.iter_mut() {
            if let Some((t, len)) = texts.remove(&s.seg_id) {
                s.text = t;
                s.len = len;
            }
        }
        Ok(())
    }

    /// Read the stored `fwd` term-id sets for a set of segment ids (every segment has
    /// one, so delete needs neither the text nor the tokenizer).
    fn read_fwd(
        &self,
        conn: &Connection,
        ns: &Namespace,
        ids: &[u32],
    ) -> Result<HashMap<u32, Vec<TermId>>> {
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
            out.insert(id as u32, postings::deserialize(&blob)?.iter().collect());
        }
        Ok(out)
    }

    /// Run a search read on a pooled connection under one WAL snapshot, resolving the
    /// query grams to term-ids and guarding against a concurrent rebuild.
    ///
    /// The whole read runs inside one DEFERRED transaction so every statement (token
    /// dfs, postings, segment count, hydration) sees a single snapshot; without it they
    /// could straddle a concurrent id-reassigning `rebuild` commit and splice postings
    /// from the old snapshot onto seg rows from the new one. The transaction is read-only
    /// and never committed; dropping it just releases the snapshot.
    ///
    /// Because the term dictionary is in memory (out of the SQL snapshot), the grams are
    /// resolved and the dictionary generation captured atomically, then compared to the
    /// snapshot's stored `dict_generation`. A mismatch means a rebuild/reset reassigned
    /// ids relative to this snapshot — the read retries on a fresh snapshot, where the
    /// generations agree. Transient busy/locked/schema-change faults are retried too.
    fn search_read<R>(
        &self,
        all_grams: &[String],
        f: impl FnMut(&Connection, &Namespace, &HashMap<String, TermId>, &ClassSnap) -> Result<R>,
    ) -> Result<R> {
        let conn = self.backend.read()?;
        self.search_read_on(&conn, all_grams, f)
    }

    /// The retry/snapshot/generation-guard loop on a **given** connection — used both by
    /// [`search_read`](Self::search_read) (a fresh pooled checkout per call) and by the
    /// warm [`SearchSession`] (a held connection reused across keystrokes). Retries open a
    /// fresh snapshot on the same connection.
    fn search_read_on<R>(
        &self,
        conn: &Connection,
        all_grams: &[String],
        mut f: impl FnMut(&Connection, &Namespace, &HashMap<String, TermId>, &ClassSnap) -> Result<R>,
    ) -> Result<R> {
        let ns = self.backend.namespace();
        let gram_refs: Vec<&str> = all_grams.iter().map(String::as_str).collect();
        let mut attempt = 0;
        loop {
            let tx = match conn.unchecked_transaction() {
                Ok(tx) => tx,
                Err(e) => return Err(Error::from(e)),
            };
            // Resolve the grams in memory + capture the generation and per-class stats
            // snapshot atomically, then read the snapshot's generation to compare.
            let (resolved, gen_mem, class_snap) = self.dict.resolve_batch(&gram_refs);
            let gen_snap = match schema::dict_generation(&tx, ns) {
                Ok(g) => g,
                Err(e) => {
                    let retry =
                        attempt < RETRY_MAX && matches!(&e, Error::Sqlite(se) if is_retryable(se));
                    drop(tx);
                    if retry {
                        attempt += 1;
                        std::thread::sleep(Duration::from_millis(10 * attempt as u64));
                        continue;
                    }
                    return Err(e);
                }
            };
            if gen_snap != gen_mem {
                // A rebuild/reset reassigned ids between the in-memory capture and this
                // snapshot — retry on a fresh snapshot on the same connection.
                drop(tx);
                if attempt >= RETRY_MAX {
                    // The store is consistent; only the in-memory dictionary raced a
                    // concurrent rebuild's id-reassignment. This is transient — a fresh
                    // reader resolves against the settled generation — so surface it as
                    // retryable, NOT Corrupt (which would wrongly imply an unrepairable store).
                    return Err(Error::busy(
                        "dictionary generation skew did not settle across retries; retry on a fresh reader",
                    ));
                }
                attempt += 1;
                std::thread::sleep(Duration::from_millis(10 * attempt as u64));
                continue;
            }
            match f(&tx, ns, &resolved, &class_snap) {
                Err(Error::Sqlite(e)) if attempt < RETRY_MAX && is_retryable(&e) => {
                    drop(tx);
                    attempt += 1;
                    std::thread::sleep(Duration::from_millis(10 * attempt as u64));
                }
                other => return other,
            }
        }
    }
}

/// Whether `segments` names any label more than once — a caller-contract violation for the
/// insert/upsert methods (a document holds each label at most once). Used only by a
/// `debug_assert`; `O(n²)` over a tiny slice, debug-only.
fn has_duplicate_label(segments: &[(&str, &str)]) -> bool {
    segments
        .iter()
        .enumerate()
        .any(|(i, (label, _))| segments[..i].iter().any(|(prev, _)| prev == label))
}

/// The exclusive write lease (§8): holding it **is** holding the single-writer lock, so
/// the naive-uncoordinated-write bug is inexpressible — there is no top-level write
/// method to misuse. One transaction is open for the lease; [`commit`](Writer::commit)
/// decouples durability from the lease (commit-and-continue), and dropping without
/// committing rolls the uncommitted tail back.
///
/// The six write methods are `batch ≡ atomic fold of single`: `insert`/`insert_segment`
/// error on a `(key, label)` collision (and create the document if absent); `upsert`/
/// `upsert_segment` replace without error and keep a key's other (unnamed) segments;
/// `remove`/`remove_segment` drop a document or one of its segments.
pub struct Writer<'a, T: Tokenizer = DefaultTokenizer, B: Backend = Sidecar> {
    guard: B::WriteGuard<'a>,
    index: &'a Index<T, B>,
    /// The interning session, accumulated across the open transaction; merged into the
    /// shared dictionary only on [`commit`](Self::commit), discarded on rollback.
    stage: Option<dict::InternStage<'a>>,
    /// `(term-id, old_df, new_df)` deltas to fold into the class stats on commit.
    pending_df: Vec<(TermId, i64, i64)>,
    /// Whether uncommitted work exists since the last `BEGIN`.
    dirty: bool,
    /// Whether a transaction is currently open. `false` only if a re-`BEGIN` after a
    /// `commit()` failed — the writer is then poisoned and every method returns an error
    /// rather than silently running in autocommit (which would lose atomicity).
    txn_open: bool,
}

impl<'a, T: Tokenizer, B: Backend> Writer<'a, T, B> {
    /// Acquire the writer lock and open the first transaction.
    fn begin(index: &'a Index<T, B>) -> Result<Self> {
        let guard = index.backend.write()?;
        guard.execute_batch("BEGIN IMMEDIATE")?;
        Ok(Writer {
            guard,
            index,
            stage: Some(index.dict.stage()),
            pending_df: Vec::new(),
            dirty: false,
            txn_open: true,
        })
    }

    /// Run one write method's body inside a `SAVEPOINT`, so a mid-call error rolls back
    /// *all* of that call's effects — both the SQL (`ROLLBACK TO`) and the in-memory intern
    /// staging + pending df deltas — leaving the store exactly as before the call. Without
    /// this, a caller that catches the error and later `commit()`s would persist a torn,
    /// internally-inconsistent partial write (orphan segments, negative df).
    fn atomic<R>(&mut self, body: impl FnOnce(&mut Self) -> Result<R>) -> Result<R> {
        if !self.txn_open {
            return Err(Error::corrupt(
                "writer transaction is not open (a prior commit failed to re-begin); re-acquire the writer",
            ));
        }
        let stage_mark = self.stage.as_ref().expect("writer stage present").mark();
        let df_mark = self.pending_df.len();
        self.guard.execute_batch("SAVEPOINT trifle_w")?;
        match body(self) {
            Ok(r) => {
                self.guard.execute_batch("RELEASE trifle_w")?;
                Ok(r)
            }
            Err(e) => {
                // Undo the SQL and the in-memory staging this call accumulated.
                let _ = self
                    .guard
                    .execute_batch("ROLLBACK TO trifle_w; RELEASE trifle_w");
                if let Some(stage) = self.stage.as_mut() {
                    stage.rollback_to(stage_mark);
                }
                self.pending_df.truncate(df_mark);
                Err(e)
            }
        }
    }

    /// Add the given `(label, text)` segments to the document keyed `key` (creating it if
    /// absent). Errors if any `(key, label)` already exists.
    ///
    /// The labels in `segments` must be distinct (a document holds each label at most once);
    /// passing a duplicate label in one call is a contract violation asserted in debug
    /// builds.
    pub fn insert(&mut self, key: impl Into<Key>, segments: &[(&str, &str)]) -> Result<()> {
        let key = key.into();
        self.atomic(|w| w.write_doc(key, segments, false))
    }

    /// Add a single segment `(key, label) = text` (creating the document if absent).
    /// Errors if `(key, label)` already exists.
    pub fn insert_segment(&mut self, key: impl Into<Key>, label: &str, text: &str) -> Result<()> {
        let key = key.into();
        self.atomic(|w| w.write_doc(key, &[(label, text)], false))
    }

    /// Insert-or-replace the given `(label, text)` segments of the document keyed `key`
    /// (creating it if absent). Never errors on collision; a key's other (unnamed)
    /// segments are left intact.
    ///
    /// The labels in `segments` must be distinct (a document holds each label at most once);
    /// passing a duplicate label in one call is a contract violation asserted in debug
    /// builds.
    pub fn upsert(&mut self, key: impl Into<Key>, segments: &[(&str, &str)]) -> Result<()> {
        let key = key.into();
        self.atomic(|w| w.write_doc(key, segments, true))
    }

    /// Insert-or-replace a single segment `(key, label) = text`.
    pub fn upsert_segment(&mut self, key: impl Into<Key>, label: &str, text: &str) -> Result<()> {
        let key = key.into();
        self.atomic(|w| w.write_doc(key, &[(label, text)], true))
    }

    /// Insert a whole [`Document`] — its segments **and** its filterable payload — in one
    /// atomic call (errors on a `(key, label)` collision). This is the incremental twin of
    /// the [`Document`] form [`rebuild`](Index::rebuild) consumes, so a document with
    /// filterable columns has a single home on the write path (audit I8).
    pub fn insert_document(&mut self, doc: Document) -> Result<()> {
        self.write_document(doc, false)
    }

    /// Insert-or-replace a whole [`Document`] (segments + filterable payload) in one atomic
    /// call; never errors on collision, and keeps the key's other (unnamed) segments.
    pub fn upsert_document(&mut self, doc: Document) -> Result<()> {
        self.write_document(doc, true)
    }

    /// Shared body of [`insert_document`](Self::insert_document) /
    /// [`upsert_document`](Self::upsert_document): write the segments, then the payload, all
    /// under one savepoint.
    fn write_document(&mut self, doc: Document, replace: bool) -> Result<()> {
        self.atomic(|w| {
            let segs: Vec<(&str, &str)> = doc
                .segments
                .iter()
                .map(|(l, t)| (l.as_str(), t.as_str()))
                .collect();
            w.write_doc(doc.key.clone(), &segs, replace)?;
            if !doc.payload.is_empty() {
                let ns = w.index.backend.namespace();
                let conn: &Connection = &w.guard;
                let did = w
                    .index
                    .doc_id_for(conn, ns, &doc.key, true)?
                    .expect("create=true always yields a doc id");
                w.index.set_doc_fields(conn, ns.doc(), did, &doc.payload)?;
            }
            Ok(())
        })
    }

    /// The shared body of the four insert/upsert methods.
    fn write_doc(&mut self, key: Key, segments: &[(&str, &str)], replace: bool) -> Result<()> {
        let ns = self.index.backend.namespace();
        let conn: &Connection = &self.guard;
        let stage = self.stage.as_mut().expect("writer stage present");
        let mut changes = TokenChanges::default();
        let doc = self
            .index
            .doc_id_for(conn, ns, &key, true)?
            .expect("create=true always yields a doc id");
        // A document holds each label at most once, so the caller must pass distinct labels
        // in one call. A repeat would insert a segment and then replace it in the *same*
        // transaction — putting one seg id in both the add and remove sets of a term and
        // drifting `df` (caught downstream by the disjointness `debug_assert`). Defend it
        // cheaply at the boundary with a clearer message; it is a caller contract, not a
        // recoverable input error, so it is debug-only.
        debug_assert!(
            !has_duplicate_label(segments),
            "write_doc requires distinct labels within one call (a document holds each label once)"
        );
        for (label, text) in segments {
            self.index
                .write_segment(conn, ns, doc, label, text, &mut changes, stage, replace)?;
        }
        let df = changes.apply(conn, ns)?;
        self.pending_df.extend(df);
        self.dirty = true;
        Ok(())
    }

    /// Remove the document keyed `key` and all its segments. A nonexistent key is a no-op.
    pub fn remove(&mut self, key: impl Into<Key>) -> Result<()> {
        let key = key.into();
        self.atomic(|w| {
            let ns = w.index.backend.namespace();
            let conn: &Connection = &w.guard;
            let mut changes = TokenChanges::default();
            if let Some(doc) = w.index.doc_id_for(conn, ns, &key, false)? {
                for (id, tokens) in w.index.doc_segments(conn, ns, doc)? {
                    changes.remove(id, &tokens);
                }
                // Subtract every removed segment's gram length + count from the BM25 stats.
                let (seg_n, seg_len): (i64, i64) = conn.query_row(
                    &format!(
                        "SELECT count(*), coalesce(sum(len), 0) FROM {} WHERE doc_id = ?1",
                        ns.seg()
                    ),
                    rusqlite::params![doc],
                    |r| Ok((r.get(0)?, r.get(1)?)),
                )?;
                w.index.delete_doc_rows(conn, ns, doc)?;
                conn.execute(
                    &format!("DELETE FROM {} WHERE id = ?1", ns.doc()),
                    rusqlite::params![doc],
                )?;
                schema::bump_seg_stats(conn, ns, -seg_n, -seg_len)?;
            }
            let df = changes.apply(conn, ns)?;
            w.pending_df.extend(df);
            w.dirty = true;
            Ok(())
        })
    }

    /// Set the document's filterable field values (Tier 2), creating the document if
    /// absent. Undeclared field names are ignored. Independent of the segment methods.
    pub fn set_fields(&mut self, key: impl Into<Key>, fields: &[(&str, Value)]) -> Result<()> {
        let key = key.into();
        self.atomic(|w| {
            let ns = w.index.backend.namespace();
            let conn: &Connection = &w.guard;
            let doc = w
                .index
                .doc_id_for(conn, ns, &key, true)?
                .expect("create=true always yields a doc id");
            let owned: Vec<(String, Value)> = fields
                .iter()
                .map(|(n, v)| (n.to_string(), v.clone()))
                .collect();
            w.index.set_doc_fields(conn, ns.doc(), doc, &owned)?;
            w.dirty = true;
            Ok(())
        })
    }

    /// Remove the segment `(key, label)`. A nonexistent key or label is a no-op; the
    /// document's other segments are left intact.
    pub fn remove_segment(&mut self, key: impl Into<Key>, label: &str) -> Result<()> {
        let key = key.into();
        self.atomic(|w| {
            let ns = w.index.backend.namespace();
            let conn: &Connection = &w.guard;
            let mut changes = TokenChanges::default();
            if let Some(doc) = w.index.doc_id_for(conn, ns, &key, false)? {
                w.index
                    .remove_one_segment(conn, ns, doc, label, &mut changes)?;
            }
            let df = changes.apply(conn, ns)?;
            w.pending_df.extend(df);
            w.dirty = true;
            Ok(())
        })
    }

    /// Commit the open transaction and continue under a fresh one (commit-and-continue),
    /// keeping the lease. Only here do this batch's interned grams and class-stat changes
    /// enter the shared in-memory state (after the durable commit), so a rolled-back tail
    /// leaves no orphan id.
    ///
    /// # Errors
    ///
    /// Returns an error if the commit or the re-`BEGIN` fails.
    pub fn commit(&mut self) -> Result<()> {
        self.guard.execute_batch("COMMIT")?;
        // The COMMIT succeeded; the txn is now closed. If the re-`BEGIN` below fails we must
        // not leave the writer silently in autocommit, so mark it closed up front and only
        // re-open on success.
        self.txn_open = false;
        // Order matters: merge the staged interns first (advancing the shared high-water
        // mark), then snapshot a fresh stage off the advanced mark.
        if let Some(old) = self.stage.take() {
            old.commit();
        }
        let df = std::mem::take(&mut self.pending_df);
        self.index.dict.apply_df_changes(&df);
        self.stage = Some(self.index.dict.stage());
        self.dirty = false;
        self.guard.execute_batch("BEGIN IMMEDIATE")?;
        self.txn_open = true;
        Ok(())
    }
}

impl<T: Tokenizer, B: Backend> Drop for Writer<'_, T, B> {
    fn drop(&mut self) {
        // Roll back the uncommitted tail; the staged interns + df are dropped unmerged,
        // so a write that was never committed leaves no trace (in-memory or on disk). Skip
        // if a failed re-begin already left no open transaction.
        if self.txn_open {
            let _ = self.guard.execute_batch("ROLLBACK");
        }
    }
}

/// A read lease (§8): the surface for searches. Each search runs under its own consistent
/// WAL snapshot; acquire a fresh reader to observe newer writes.
pub struct Reader<'a, T: Tokenizer = DefaultTokenizer, B: Backend = Sidecar> {
    index: &'a Index<T, B>,
}

impl<T: Tokenizer, B: Backend> Reader<'_, T, B> {
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

    /// Search a batch of queries, one result list per query in order (`batch == serial`).
    ///
    /// # Errors
    ///
    /// Returns an error if the store read fails.
    pub fn search_batch(&self, queries: &[&str], opts: SearchOpts<'_>) -> Result<Vec<Vec<Match>>> {
        self.index.search_batch(queries, &opts)
    }
}

/// A warm search lease (§3): it holds a pooled read connection across searches, so an
/// as-you-type burst reuses one connection instead of checking out per keystroke. Each
/// search still runs under its own consistent snapshot on the held connection.
///
/// Warming layers (§3): **Layer 3** — debounce (~30–50 ms) + cancel the superseded
/// in-flight search — is the largest felt win and is a **caller** concern (drop the
/// session/future to cancel). **Layer 1** (a per-session posting/DF cache keyed on the
/// index data-version) and **Layer 2** (an incremental count vector) are documented
/// follow-ups; this type is their home — it already owns the warm connection they cache
/// against.
pub struct SearchSession<'a, T: Tokenizer = DefaultTokenizer, B: Backend = Sidecar> {
    index: &'a Index<T, B>,
    conn: B::ReadGuard<'a>,
}

impl<T: Tokenizer, B: Backend> SearchSession<'_, T, B> {
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

    /// Search a batch of queries on the warm connection (`batch == serial`).
    ///
    /// # Errors
    ///
    /// Returns an error if the store read fails.
    pub fn search_batch(&self, queries: &[&str], opts: SearchOpts<'_>) -> Result<Vec<Vec<Match>>> {
        if queries.is_empty() {
            return Ok(Vec::new());
        }
        let (query_tokens, all_grams) =
            Index::<T, B>::query_grams(queries, |q| self.index.distinct_tokens(q));
        self.index
            .search_read_on(&self.conn, &all_grams, |conn, ns, resolved, class_snap| {
                self.index.run_search(
                    conn,
                    ns,
                    resolved,
                    class_snap,
                    queries,
                    &query_tokens,
                    &opts,
                )
            })
    }
}

/// Accumulated per-term-id segment changes for one write batch, applied in a single
/// [`postings::apply_writes`] pass. Keyed by the interned `u32` term-id; the values are
/// the segment ids added and removed for that term.
#[derive(Default)]
struct TokenChanges {
    map: HashMap<TermId, (Vec<u32>, Vec<u32>)>,
}

impl TokenChanges {
    fn add(&mut self, seg_id: u32, ids: &[TermId]) {
        for &t in ids {
            self.map.entry(t).or_default().0.push(seg_id);
        }
    }

    fn remove(&mut self, seg_id: u32, ids: &[TermId]) {
        for &t in ids {
            self.map.entry(t).or_default().1.push(seg_id);
        }
    }

    /// Apply the accumulated changes, returning the per-term `(id, old_df, new_df)` for
    /// the caller to fold into the class statistics.
    fn apply(&self, conn: &Connection, ns: &Namespace) -> Result<Vec<(TermId, i64, i64)>> {
        let writes: Vec<postings::TermWrite<'_>> = self
            .map
            .iter()
            .map(|(id, (add, remove))| postings::TermWrite {
                id: *id,
                add,
                remove,
            })
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

#[cfg(test)]
mod effort_tests {
    use super::Effort;

    #[test]
    fn pool_floor_none_and_growth() {
        // No over-fetch for None / Custom(0): pool == limit.
        assert_eq!(Effort::None.pool(10, 1_000_000), 10);
        assert_eq!(Effort::Custom(0.0).pool(10, 1_000_000), 10);
        // limit == 0 stays 0.
        assert_eq!(Effort::Max.pool(0, 1_000_000), 0);
        // Small N stays at the floor: 0.05·√(10·100) = 1.58 < limit 10.
        assert_eq!(Effort::Medium.pool(10, 100), 10);
        // Large N follows c·√(k·N): k=10, N=1e6, √(kN)=3162.28.
        assert_eq!(Effort::Medium.pool(10, 1_000_000), 158); // 0.05·3162 = 158.1
        assert_eq!(Effort::High.pool(10, 1_000_000), 316); // 0.10·3162 = 316.2
        assert_eq!(Effort::Max.pool(10, 1_000_000), 1423); // 0.45·3162 = 1423.0
        // reranks() reflects whether c > 0.
        assert!(!Effort::None.reranks());
        assert!(Effort::Medium.reranks());
        assert!(!Effort::Custom(0.0).reranks());
    }
}
