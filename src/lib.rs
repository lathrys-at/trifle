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

pub use error::{Error, Result};
pub use model::{Document, Key, KeyShape, Match, Schema, SchemaBuilder, StorageMode};
use dict::{Dictionary, TermId};
use rank::{Bm25Ranker, Candidates, OverlapRanker, QueryContext, Ranker, Survivor, overlap_search};
use schema::SCHEMA_VERSION;
use select::{SelectParams, select};
use store::{Backend, Namespace, Sidecar, TextResolver};
use tokenize::{Tokenizer, TrigramTokenizer};
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
    /// A [`TextResolver`] to run **contentless** (store no text snapshot, fetch text
    /// from the caller's source). `None` (default) stores a self-contained snapshot.
    /// See the [crate] docs and [`TextResolver`] for the contract.
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

    /// Run contentless against the given resolver instead of storing a text snapshot.
    pub fn with_external_content(mut self, resolver: Box<dyn TextResolver>) -> Self {
        self.external_content = Some(resolver);
        self
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

/// The hydration cost ladder: how much of a match to materialize, as an explicit
/// ordered choice (not independent booleans), so a deeper join is never paid by accident
/// in a hot loop.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Default)]
#[non_exhaustive]
pub enum Hydration {
    /// `(key, label)` only — no segment text is fetched ([`Match::text`] stays `None`).
    Coordinates,
    /// Also fetch the matched segment's text (a [`Stored`](StorageMode::Stored) field's
    /// `seg.txt`, or the resolver for a `Resolver` field). The default.
    #[default]
    SegmentText,
    /// Also fetch the document payload. Payload columns are a later phase; today this
    /// behaves as [`SegmentText`](Hydration::SegmentText).
    DocumentPayload,
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
    /// How much of a match to materialize — the [`Hydration`] cost ladder. Defaults to
    /// [`SegmentText`](Hydration::SegmentText).
    pub hydration: Hydration,
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
            hydration: Hydration::default(),
        }
    }

    /// Set the [`Hydration`] level (how much of each match to materialize).
    pub fn hydrate(mut self, hydration: Hydration) -> Self {
        self.hydration = hydration;
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
pub struct Index<T: Tokenizer = TrigramTokenizer, B: Backend = Sidecar> {
    backend: B,
    tokenizer: T,
    data_version: u64,
    /// The declared data model (key shape, text fields, per-field storage modes).
    schema: Schema,
    /// The caller's text resolver, for `Resolver`-mode fields (`None` if unused).
    resolver: Option<Box<dyn TextResolver>>,
    /// The in-memory faulting term dictionary (gram → `u32` id), shared by the writer
    /// and the read pool. Hydrated on open; rebuilt under the swap on `rebuild`.
    dict: Dictionary,
}

impl Index<TrigramTokenizer, Sidecar> {
    /// Open (creating if absent) an index at `path` with the given [`Schema`], the
    /// default trigram tokenizer, and an owned [`Sidecar`] file. The common case.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened or the store cannot be
    /// initialized.
    pub fn open_at(path: &Path, schema: Schema, config: Config) -> Result<Self> {
        let backend = Sidecar::open(path)?;
        Index::open(backend, TrigramTokenizer::new(), schema, config)
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
        // A `Resolver`-mode field needs a resolver; reject the mismatch up front.
        if schema.needs_resolver() && config.external_content.is_none() {
            return Err(Error::InvalidInput(
                "schema has a Resolver-mode field but no TextResolver was configured".into(),
            ));
        }
        let index = Index {
            backend,
            tokenizer,
            data_version: config.data_version,
            schema,
            resolver: config.external_content,
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
        let key_ty = self.schema.key_shape().sql_type();
        let tx = guard.transaction()?;
        schema::drop_shadows(&tx, ns)?;
        schema::create_tables(&tx, ns, key_ty)?;

        let stamps = schema::read_stamps(&tx, ns)?;
        let fingerprint = self.tokenizer.fingerprint();
        let schema_fp = self.schema.fingerprint();
        let drift = stamps.schema_version != Some(SCHEMA_VERSION)
            || stamps.fingerprint != Some(fingerprint)
            || stamps.data_version != Some(self.data_version)
            || stamps.schema_fingerprint != Some(schema_fp);
        let desync = !drift && self.desync(&tx, ns)?;

        if drift || desync {
            schema::reset(&tx, ns, key_ty)?;
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

    /// Write one segment `(doc, label) = text`, interning its grams through `stage` and
    /// storing the term-id set in `fwd` (so delete needs no text). `seg.txt` is stored
    /// only for a `Stored` field. If a segment with this label already exists: with
    /// `replace`, it is dropped first (its removals accumulated into `changes`); without
    /// `replace`, this errors.
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
        let mode = self.schema.storage_for(label).ok_or_else(|| {
            Error::InvalidInput(format!("label {label:?} is not a field of the schema"))
        })?;
        if let Some(seg_id) = self.find_segment(conn, ns, doc, label)? {
            if !replace {
                return Err(Error::InvalidInput(format!(
                    "a segment with label {label:?} already exists for this key"
                )));
            }
            self.drop_segment(conn, ns, seg_id, changes)?;
        }
        let id = schema::alloc_ids(conn, ns, 1)?;
        let stored_txt = if mode == StorageMode::Stored {
            Some(text)
        } else {
            None
        };
        conn.execute(
            &format!(
                "INSERT INTO {}(id, doc_id, label, txt) VALUES(?1, ?2, ?3, ?4)",
                ns.seg()
            ),
            rusqlite::params![id, doc, label, stored_txt],
        )?;
        let tokens = self.distinct_tokens(text);
        let mut ids: Vec<TermId> = Vec::with_capacity(tokens.len());
        for t in &tokens {
            ids.push(stage.intern(t, conn, ns)?);
        }
        let bm: RoaringBitmap = ids.iter().copied().collect();
        conn.execute(
            &format!("INSERT INTO {}(id, tokens) VALUES(?1, ?2)", ns.fwd()),
            rusqlite::params![id, postings::serialize(&bm)?],
        )?;
        changes.add(id as u32, &ids);
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
                &format!("SELECT id FROM {} WHERE doc_id = ?1 AND label = ?2", ns.seg()),
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
        conn.execute(
            &format!("DELETE FROM {} WHERE id = ?1", ns.fwd()),
            rusqlite::params![seg_id],
        )?;
        conn.execute(
            &format!("DELETE FROM {} WHERE id = ?1", ns.seg()),
            rusqlite::params![seg_id],
        )?;
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
        let key_ty = self.schema.key_shape().sql_type();

        let tx = guard.transaction()?;
        schema::create_shadows(&tx, ns, key_ty)?;

        // Accumulate the inverted index in memory while streaming doc/seg rows to the
        // shadow tables. Three id spaces are reassigned dense (doc, segment, term); the
        // inverted index is keyed by segment id and the dictionary by the gram's packed
        // u128 (stored as the 16-byte big-endian blob).
        let mut local: HashMap<u128, TermId> = HashMap::new();
        let mut next_term: u64 = 1;
        let mut inverted: HashMap<TermId, RoaringBitmap> = HashMap::new();
        let mut next_doc: i64 = 1;
        let mut next_seg: i64 = 1;
        {
            let mut doc_ins = tx.prepare_cached(&format!(
                "INSERT INTO {}(id, key) VALUES(?1, ?2)",
                ns.doc_shadow()
            ))?;
            let mut seg_ins = tx.prepare_cached(&format!(
                "INSERT INTO {}(id, doc_id, label, txt) VALUES(?1, ?2, ?3, ?4)",
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
                for (label, text) in &doc.segments {
                    let mode = self.schema.storage_for(label).ok_or_else(|| {
                        Error::InvalidInput(format!("label {label:?} is not a field of the schema"))
                    })?;
                    let seg_id = next_seg;
                    next_seg += 1;
                    let stored_txt = if mode == StorageMode::Stored {
                        Some(text.as_str())
                    } else {
                        None
                    };
                    seg_ins.execute(rusqlite::params![seg_id, doc_id, label, stored_txt])?;
                    let tokens = self.distinct_tokens(text);
                    let mut ids: Vec<TermId> = Vec::with_capacity(tokens.len());
                    for tok in tokens {
                        let gkey = term::encode_term(&tok)
                            .ok_or_else(|| {
                                Error::InvalidInput(format!(
                                    "gram {tok:?} exceeds the 3-codepoint term-encoding ceiling"
                                ))
                            })?
                            .0;
                        let tid = match local.get(&gkey) {
                            Some(&t) => t,
                            None => {
                                if next_term == 0 || next_term > u32::MAX as u64 {
                                    return Err(Error::corrupt("term id space exhausted"));
                                }
                                let t = next_term as TermId;
                                next_term += 1;
                                local.insert(gkey, t);
                                t
                            }
                        };
                        ids.push(tid);
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

        schema::swap_shadows(&tx, ns)?;
        schema::set_next_id(&tx, ns, next_seg)?;
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

        // Distinct tokens per query (deduplicated), computed once outside the read.
        let query_tokens: Vec<Vec<String>> =
            queries.iter().map(|q| self.distinct_tokens(q)).collect();
        // Every distinct gram across the batch — resolved once to term-ids inside the
        // read (under the dictionary's generation guard).
        let all_grams: Vec<&str> = query_tokens
            .iter()
            .flat_map(|q| q.iter().map(String::as_str))
            .collect::<BTreeSet<&str>>()
            .into_iter()
            .collect();

        self.search_read(&all_grams, |conn, ns, resolved, class_snap| {
            // One batched frequency read over every resolved term-id in the batch.
            let all_ids: Vec<TermId> = resolved
                .values()
                .copied()
                .collect::<BTreeSet<TermId>>()
                .into_iter()
                .collect();
            let dfs = postings::read_dfs(conn, ns, &all_ids)?;
            // A gram's df: 0 if it resolved to no id (absent token) or its id has no
            // live df row — exactly the existing df-0 behavior.
            let df_of = |t: &str| -> i64 {
                resolved
                    .get(t)
                    .and_then(|id| dfs.get(id))
                    .copied()
                    .unwrap_or(0)
            };

            // Per-query selection, then one batched posting read over every selected
            // token in the batch.
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

            // Total live segments — the `N` a length/idf reranker normalizes by. One
            // count for the whole batch (cheap; the default overlap ranker ignores it).
            let n_segments: u64 =
                conn.query_row(&format!("SELECT count(*) FROM {}", ns.seg()), [], |r| {
                    r.get::<_, i64>(0)
                })? as u64;
            // A reranking ranker needs more than `limit` candidates to reorder (the
            // target of a "ranking gap" miss sits past `limit` by raw overlap); fetch a
            // pool, rank it, then truncate to `limit`. Default pool == limit (no
            // over-fetch), so the overlap ranker pays nothing.
            let pool = opts.effort.pool(opts.limit, n_segments);

            let mut out = Vec::with_capacity(queries.len());
            for (qi, query) in queries.iter().enumerate() {
                let selected = &selected_per[qi];
                // Map each selected token back to its posting via its resolved id.
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
                    opts.scope,
                )?;
                if opts.hydration != Hydration::Coordinates {
                    self.hydrate_text(conn, ns, &mut survivors)?;
                }

                out.push(self.rank_to_matches(
                    &survivors, &present, selected, query, min_shared, ranker, opts.limit,
                    n_segments,
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
        n_segments: u64,
    ) -> Vec<Match> {
        let candidates = Candidates::new(survivors, present);
        let qctx = QueryContext {
            query,
            selected,
            min_shared,
            n_segments,
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
                key: s.key.clone(),
                label: s.label.clone(),
                span,
                text: s.text.clone(),
            });
        }
        matches
    }

    /// Hydrate each survivor's text according to its field's [`StorageMode`]: `Stored`
    /// from `seg.txt` (one batched read), `Resolver` from the caller's resolver (one
    /// batched callback), `CoordinatesOnly` left `None`.
    fn hydrate_text(
        &self,
        conn: &Connection,
        ns: &Namespace,
        survivors: &mut [Survivor],
    ) -> Result<()> {
        if survivors.is_empty() {
            return Ok(());
        }
        // Stored text — `seg.txt` is NULL for any non-`Stored` field, so this batched read
        // naturally yields `None` for them.
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
        // Assign per storage mode; collect `Resolver`-mode survivors for one callback.
        let mut resolver_idx: Vec<usize> = Vec::new();
        for (i, s) in survivors.iter_mut().enumerate() {
            match self.schema.storage_for(&s.label) {
                Some(StorageMode::Stored) => s.text = texts.get(&s.seg_id).cloned().flatten(),
                Some(StorageMode::Resolver) => resolver_idx.push(i),
                _ => s.text = None, // CoordinatesOnly (or a label no longer in the schema)
            }
        }
        if !resolver_idx.is_empty() {
            if let Some(resolver) = &self.resolver {
                let got = {
                    let segs: Vec<(&Key, &str)> = resolver_idx
                        .iter()
                        .map(|&i| (&survivors[i].key, survivors[i].label.as_str()))
                        .collect();
                    resolver.resolve(&segs)?
                };
                for (slot, text) in resolver_idx.iter().zip(got) {
                    survivors[*slot].text = text;
                }
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
        all_grams: &[&str],
        mut f: impl FnMut(&Connection, &Namespace, &HashMap<String, TermId>, &ClassSnap) -> Result<R>,
    ) -> Result<R> {
        let ns = self.backend.namespace();
        let mut attempt = 0;
        loop {
            let conn = self.backend.read()?;
            let tx = match conn.unchecked_transaction() {
                Ok(tx) => tx,
                Err(e) => return Err(Error::from(e)),
            };
            // Resolve the grams in memory + capture the generation and per-class stats
            // snapshot atomically, then read the snapshot's generation to compare.
            let (resolved, gen_mem, class_snap) = self.dict.resolve_batch(all_grams);
            let gen_snap = match schema::dict_generation(&tx, ns) {
                Ok(g) => g,
                Err(e) => {
                    let retry =
                        attempt < RETRY_MAX && matches!(&e, Error::Sqlite(se) if is_retryable(se));
                    drop(tx);
                    drop(conn);
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
                // snapshot — retry on a fresh snapshot.
                drop(tx);
                drop(conn);
                if attempt >= RETRY_MAX {
                    return Err(Error::corrupt(
                        "dictionary generation skew did not settle across retries",
                    ));
                }
                attempt += 1;
                std::thread::sleep(Duration::from_millis(10 * attempt as u64));
                continue;
            }
            match f(&tx, ns, &resolved, &class_snap) {
                Err(Error::Sqlite(e)) if attempt < RETRY_MAX && is_retryable(&e) => {
                    drop(tx);
                    drop(conn);
                    attempt += 1;
                    std::thread::sleep(Duration::from_millis(10 * attempt as u64));
                }
                other => return other,
            }
        }
    }
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
pub struct Writer<'a, T: Tokenizer = TrigramTokenizer, B: Backend = Sidecar> {
    guard: B::WriteGuard<'a>,
    index: &'a Index<T, B>,
    /// The interning session, accumulated across the open transaction; merged into the
    /// shared dictionary only on [`commit`](Self::commit), discarded on rollback.
    stage: Option<dict::InternStage<'a>>,
    /// `(term-id, old_df, new_df)` deltas to fold into the class stats on commit.
    pending_df: Vec<(TermId, i64, i64)>,
    /// Whether uncommitted work exists since the last `BEGIN`.
    dirty: bool,
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
        })
    }

    /// Add the given `(label, text)` segments to the document keyed `key` (creating it if
    /// absent). Errors if any `(key, label)` already exists.
    pub fn insert(&mut self, key: impl Into<Key>, segments: &[(&str, &str)]) -> Result<()> {
        self.write_doc(key.into(), segments, false)
    }

    /// Add a single segment `(key, label) = text` (creating the document if absent).
    /// Errors if `(key, label)` already exists.
    pub fn insert_segment(&mut self, key: impl Into<Key>, label: &str, text: &str) -> Result<()> {
        self.write_doc(key.into(), &[(label, text)], false)
    }

    /// Insert-or-replace the given `(label, text)` segments of the document keyed `key`
    /// (creating it if absent). Never errors on collision; a key's other (unnamed)
    /// segments are left intact.
    pub fn upsert(&mut self, key: impl Into<Key>, segments: &[(&str, &str)]) -> Result<()> {
        self.write_doc(key.into(), segments, true)
    }

    /// Insert-or-replace a single segment `(key, label) = text`.
    pub fn upsert_segment(&mut self, key: impl Into<Key>, label: &str, text: &str) -> Result<()> {
        self.write_doc(key.into(), &[(label, text)], true)
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
        let ns = self.index.backend.namespace();
        let conn: &Connection = &self.guard;
        let mut changes = TokenChanges::default();
        if let Some(doc) = self.index.doc_id_for(conn, ns, &key, false)? {
            for (id, tokens) in self.index.doc_segments(conn, ns, doc)? {
                changes.remove(id, &tokens);
            }
            self.index.delete_doc_rows(conn, ns, doc)?;
            conn.execute(
                &format!("DELETE FROM {} WHERE id = ?1", ns.doc()),
                rusqlite::params![doc],
            )?;
        }
        let df = changes.apply(conn, ns)?;
        self.pending_df.extend(df);
        self.dirty = true;
        Ok(())
    }

    /// Remove the segment `(key, label)`. A nonexistent key or label is a no-op; the
    /// document's other segments are left intact.
    pub fn remove_segment(&mut self, key: impl Into<Key>, label: &str) -> Result<()> {
        let key = key.into();
        let ns = self.index.backend.namespace();
        let conn: &Connection = &self.guard;
        let mut changes = TokenChanges::default();
        if let Some(doc) = self.index.doc_id_for(conn, ns, &key, false)? {
            self.index
                .remove_one_segment(conn, ns, doc, label, &mut changes)?;
        }
        let df = changes.apply(conn, ns)?;
        self.pending_df.extend(df);
        self.dirty = true;
        Ok(())
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
        Ok(())
    }
}

impl<T: Tokenizer, B: Backend> Drop for Writer<'_, T, B> {
    fn drop(&mut self) {
        // Roll back the uncommitted tail; the staged interns + df are dropped unmerged,
        // so a write that was never committed leaves no trace (in-memory or on disk).
        let _ = self.guard.execute_batch("ROLLBACK");
    }
}

/// A read lease (§8): the surface for searches. Each search runs under its own consistent
/// WAL snapshot; acquire a fresh reader to observe newer writes.
pub struct Reader<'a, T: Tokenizer = TrigramTokenizer, B: Backend = Sidecar> {
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
