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
//! (≲ 1–2 KB per segment), read-often / write-infrequent. It is a **fuzzy lexical overlap
//! engine, not a relevance engine**: it ranks by IDF-weighted token overlap (rarer shared
//! grams weigh more), computed in the counter itself — there is no BM25/relevance tier. A
//! caller wanting one supplies a custom [`Ranker`].
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
//! committed write is searchable by the very next [`reader`](Index::reader) acquired after
//! [`commit`](Writer::commit) returns. (A reader on *another* thread that opens its snapshot
//! during the sub-millisecond window between the SQL commit and the in-memory dictionary
//! merge may briefly miss a *brand-new* n-gram's first occurrence — missing, never wrong, and
//! self-healing on the next reader.) This
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
mod search;
mod select;
mod term;
mod welford;

use std::borrow::Borrow;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

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
use rank::Ranker;
use schema::SCHEMA_VERSION;
use store::{Backend, Namespace, Sidecar};
pub use term::{IntoTerm, Term};
use tokenize::{DefaultTokenizer, Tokenizer};

/// Default match floor `m` — shared rare tokens required for a hit.
const DEFAULT_MIN_SHARED: u32 = 2;
/// Per-typo token damage `d`; the typo floor is `F = m + d`.
const TYPO_DAMAGE: u32 = 4;
/// Default `t_max` — the number of rarest query tokens selection keeps.
const DEFAULT_T_MAX: usize = 12;
/// Default `D` — df-doublings per IDF weight step in the overlap counter.
const DEFAULT_WEIGHT_STEP: f64 = 1.0;
/// Buckets in the per-query band-spread histogram (the [`Stats`] weight-step hint).
const HINT_BUCKETS: usize = 13;
/// Width of each band-spread histogram bucket, in df-doublings (`log2` units).
const HINT_BUCKET_WIDTH: f64 = 0.5;
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

/// How deep a candidate pool to over-fetch for a **custom** [`Ranker`] to reorder.
///
/// trifle's default ranking is the IDF-weighted overlap order the counter produces, and it
/// is exact at `pool = limit` (the bucket walk early-stops once the top-`limit` lock). So the
/// default is [`None`](Effort::None) — no over-fetch. A *custom*
/// [`ranker`](SearchOpts::ranker) that reorders survivors may need a deeper pool to pull up a
/// document the weighted order placed past `limit`; the levels set that depth as
/// `pool = max(limit, round(c·√(limit · N)))` (N = indexed segments) — a power-law-in-corpus-size
/// over-fetch:
///
/// | level | `c` | pool depth |
/// |-------|-----|------------|
/// | [`None`](Effort::None)     | 0    | `limit` (no over-fetch) — **the default** |
/// | [`Low`](Effort::Low)       | 0.03 | shallow |
/// | [`Medium`](Effort::Medium) | 0.05 | |
/// | [`High`](Effort::High)     | 0.10 | |
/// | [`Max`](Effort::Max)       | 0.45 | deepest |
///
/// Deeper pool → more hydration + custom-ranker scoring, so latency grows ~linearly with the
/// pool. **Without a custom `ranker` a deeper pool changes nothing** (the weighted-overlap
/// order is already the final order), so leave it at `None`. The pool is always at least `limit`.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum Effort {
    /// No over-fetch: pool = `limit`. Exact for the default weighted-overlap order. The default.
    None,
    /// A shallow over-fetch pool for a custom ranker (`c = 0.03`).
    Low,
    /// A moderate over-fetch pool for a custom ranker (`c = 0.05`).
    Medium,
    /// A deep over-fetch pool for a custom ranker (`c = 0.10`).
    High,
    /// The deepest over-fetch pool for a custom ranker (`c = 0.45`).
    Max,
    /// An explicit coefficient `c` in `pool = max(limit, round(c·√(limit·N)))`. `0`
    /// means no over-fetch, as [`None`](Effort::None).
    Custom(f64),
}

impl Default for Effort {
    /// [`None`](Effort::None) — no over-fetch; the weighted-overlap order is exact at
    /// `pool = limit`.
    fn default() -> Self {
        Effort::None
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

    /// The over-fetch pool depth for a result `limit` over `n_segments` indexed segments:
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
    /// A per-query [`Ranker`] to reorder the IDF-weighted-overlap survivors. `None` → the
    /// built-in [`OverlapRanker`](rank::OverlapRanker), which preserves trifle's weighted-overlap order (there is
    /// no built-in relevance/BM25 ranker).
    pub ranker: Option<&'a dyn Ranker>,
    /// A membership predicate `(key, label) -> keep` evaluated over candidates in
    /// weighted-overlap order — never over the corpus. The walk continues until `limit`
    /// predicate-passing results lock, so scoping needs no over-fetch. The predicate must
    /// not call back into this index's writer.
    pub scope: Option<&'a ScopeFn<'a>>,
    /// The over-fetch [`Effort`] — how deep a candidate pool to fetch for a custom `ranker`
    /// to reorder. Defaults to [`None`](Effort::None) (pool = `limit`), which is exact for the
    /// default weighted-overlap order; raise it only when a custom `ranker` needs deeper
    /// candidates to pull up.
    pub effort: Effort,
    /// `D` — df-doublings per IDF weight step in the overlap counter (the lone rarity-weighting
    /// knob). `1.0` (the default) means each weight level is one more halving of df relative to
    /// the query's commonest survivor. Larger `D` widens the steps (more grams share a tier).
    /// `N`-invariant, so it does not go stale on insert. [`Stats::weight_step_hint`] suggests a
    /// corpus-fitted value.
    pub weight_step: f64,
    /// An optional [`Filter`] over the schema's **filterable** fields (Tier 2). Applied as
    /// a doc-id intersection during candidate generation — it prunes before the ranker and
    /// hydration, but does not save the overlap work itself.
    pub filter: Option<&'a Filter>,
}

impl<'a> SearchOpts<'a> {
    /// Options for the given result limit, everything else default (weighted-overlap order,
    /// no over-fetch, weight step `D = 1.0`).
    pub fn new(limit: usize) -> Self {
        SearchOpts {
            limit,
            min_shared: None,
            t_max: None,
            ranker: None,
            scope: None,
            effort: Effort::default(),
            weight_step: DEFAULT_WEIGHT_STEP,
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

    /// Set the over-fetch [`Effort`] — the candidate-pool depth for a custom [`ranker`](Self::ranker).
    pub fn rerank(mut self, effort: Effort) -> Self {
        self.effort = effort;
        self
    }

    /// Set `D`, the df-doublings per IDF weight step ([`weight_step`](Self::weight_step)).
    pub fn weight_step(mut self, d: f64) -> Self {
        self.weight_step = d;
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
///
/// Not `Eq` because [`weight_step_hint`](Stats::weight_step_hint) carries floats.
#[derive(Clone, Copy, Debug, PartialEq)]
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
    /// A corpus-derived suggestion for [`SearchOpts::weight_step`] `D`, accumulated from the
    /// band-spreads of the searches run since this index was opened. `None` until at least one
    /// search has run. See [`WeightStepHint`].
    pub weight_step_hint: Option<WeightStepHint>,
}

/// A suggested [`SearchOpts::weight_step`] `D`, with the band-spread distribution it came from
/// so a caller can judge whether a single `D` fits the corpus.
///
/// Built from the per-query band-spreads (`log2(df_max/df_min)`) observed since the index was
/// opened. `suggested ≈ median / 3` (so a median-width band spans ~3 tier steps → uses
/// weights 1–4). The `iqr` is the confidence signal: a tight IQR means one `D` fits; a wide
/// or bimodal one means the corpus has multiple query regimes and no single `D` is ideal.
/// Spreads are bucketed at `0.5`-doubling granularity, so these are bucket-midpoint estimates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WeightStepHint {
    /// The suggested `D` (`max(0.5, median_spread / 3)`).
    pub suggested: f64,
    /// Median per-query band-spread, in df-doublings.
    pub median_spread: f64,
    /// The interquartile range `(Q1, Q3)` of band-spreads, in df-doublings — the spread of
    /// the spreads (the confidence signal).
    pub iqr: (f64, f64),
    /// How many searches contributed (the sample size behind the suggestion).
    pub samples: u64,
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
    /// Set if [`rebuild`](Self::rebuild)'s in-memory dictionary reload failed *after* its
    /// SQL swap had already committed: the on-disk term space is the new generation but the
    /// in-memory map still reflects the old one, so its ids would mis-route reads and writes.
    /// Every lease then fails closed until the caller reopens (or a later `rebuild` succeeds
    /// and clears it) — far better than silently serving from a stale dictionary (audit C2-FB-C1).
    poisoned: AtomicBool,
    /// In-memory per-query band-spread histogram: each search adds one sample,
    /// `log2(df_max/df_min)` over its present postings, bucketed in `HINT_BUCKET_WIDTH`
    /// df-doublings. [`stats`](Self::stats) derives a suggested [`weight_step`](SearchOpts::weight_step)
    /// `D` from it. Telemetry only — process-local, never persisted, reset by reopening.
    band_spread_hist: [AtomicU64; HINT_BUCKETS],
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
            poisoned: AtomicBool::new(false),
            band_spread_hist: std::array::from_fn(|_| AtomicU64::new(0)),
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
            // A reset empties the corpus, so any band-spread samples no longer describe it
            // (audit T15). `init` runs only at `open` on a freshly-zeroed histogram, so this is
            // currently a no-op — but keeping the clear local to the reset path stops the
            // invariant from silently depending on construction order.
            self.reset_band_spread_hist();
        }
        tx.commit()?;
        Ok(())
    }

    /// Fail closed if a prior `rebuild` left the in-memory dictionary stale (see
    /// [`poisoned`](Self::poisoned)). Called at every lease/maintenance entry point so a
    /// poisoned index never serves a search or accepts a write against a mis-routed map.
    fn check_poisoned(&self) -> Result<()> {
        if self.poisoned.load(Ordering::Acquire) {
            return Err(Error::corrupt(
                "index poisoned: an in-memory dictionary reload failed after a committed \
                 rebuild; reopen the index (its on-disk state is intact) to recover",
            ));
        }
        Ok(())
    }

    /// Record one query's band-spread (`log2(df_max/df_min)` over its present postings) into
    /// the in-memory histogram that backs the [`Stats`] weight-step hint. A no-op for a query
    /// with no present postings. Cheap: one `log2` + one atomic increment per query.
    fn observe_band_spread(&self, present: &[(&str, &RoaringBitmap)]) {
        let (mut lo, mut hi) = (u64::MAX, 0u64);
        for (_, bm) in present {
            let df = bm.len();
            if df > 0 {
                lo = lo.min(df);
                hi = hi.max(df);
            }
        }
        if hi == 0 {
            return; // no present postings — nothing to sample
        }
        let spread = (hi as f64 / lo.max(1) as f64).log2();
        let bucket = ((spread / HINT_BUCKET_WIDTH) as usize).min(HINT_BUCKETS - 1);
        self.band_spread_hist[bucket].fetch_add(1, Ordering::Relaxed);
    }

    /// Zero the in-memory band-spread histogram. Called on a successful [`rebuild`](Self::rebuild)
    /// and on a drift/desync reset: both can shift the corpus df distribution, so pre-change
    /// samples would bias the [`weight_step_hint`](Self::weight_step_hint) (audit T15). A
    /// [`compact`](Self::compact) deliberately does **not** call this — a fold leaves every df
    /// unchanged, so its samples stay valid.
    fn reset_band_spread_hist(&self) {
        for b in &self.band_spread_hist {
            b.store(0, Ordering::Relaxed);
        }
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
        self.check_poisoned()?;
        Writer::begin(self)
    }

    /// Acquire a [`Reader`] lease for issuing searches. Each search runs under its own
    /// consistent WAL snapshot; acquire a fresh reader to see newer writes.
    ///
    /// # Errors
    ///
    /// Returns an error if the lease cannot be set up.
    pub fn reader(&self) -> Result<Reader<'_, T, B>> {
        self.check_poisoned()?;
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
        self.check_poisoned()?;
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

    /// Tokenize `text` once, returning both its distinct resolved term-ids (each assigned
    /// via `assign`) and its total gram count **with repetition** — the stored segment
    /// length `|d|` (kept for a custom [`Ranker`]). This is the single lowering
    /// shared by the incremental write path
    /// ([`write_segment`](Self::write_segment), `assign` = stage intern) and
    /// [`rebuild`](Self::rebuild) (`assign` = a dense local allocator). Owning the
    /// tokenize + 3-codepoint ceiling check in one place keeps the two paths from drifting
    /// (audit I4); returning the length here keeps the segment tokenized exactly **once**
    /// per write/rebuild (audit C2-F1 — was a second `tokenize().count()` pass).
    fn distinct_term_ids(
        &self,
        text: &str,
        mut assign: impl FnMut(Term) -> Result<TermId>,
    ) -> Result<(Vec<TermId>, i64)> {
        let mut distinct: HashSet<T::Token> = HashSet::new();
        let mut seg_len: i64 = 0;
        for tok in self.tokenizer.tokenize(text) {
            seg_len += 1;
            distinct.insert(tok);
        }
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
        Ok((ids, seg_len))
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
        // Tokenize once: intern straight from the tokens (`token.term()`), not via stringified
        // grams — the blanket `IntoTerm` packs each token with no per-token `String` allocation —
        // and take the segment's gram length (with repetition; a stored signal for a custom
        // ranker) from the same pass (audit C2-F1).
        let (ids, seg_len) =
            self.distinct_term_ids(text, |term| stage.intern_term(term, conn, ns))?;
        conn.execute(
            &format!(
                "INSERT INTO {}(id, doc_id, label, txt, len) VALUES(?1, ?2, ?3, ?4, ?5)",
                ns.seg()
            ),
            rusqlite::params![id, doc, label, text, seg_len],
        )?;
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
        // Subtract this segment's gram length from the rolling segment-length stats before deleting.
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
    /// no-op if absent. If that was the document's **last** segment, the now-empty `doc`
    /// row is dropped too, so `remove_segment`-to-empty converges with [`remove`](Writer::remove)
    /// (whole-doc) and a logically-deleted document leaves no orphan row whose filterable
    /// payload a later insert under the same key would silently inherit (audit C2-RA-1).
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
            let remaining: i64 = conn.query_row(
                &format!("SELECT count(*) FROM {} WHERE doc_id = ?1", ns.seg()),
                rusqlite::params![doc],
                |r| r.get(0),
            )?;
            if remaining == 0 {
                conn.execute(
                    &format!("DELETE FROM {} WHERE id = ?1", ns.doc()),
                    rusqlite::params![doc],
                )?;
            }
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
        self.check_poisoned()?;
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
        // The shadow `doc` schema is known here, so fold the filterable columns straight into
        // the INSERT column list and bind their values inline — one write per document instead
        // of a bare `doc` INSERT plus a separate per-payload `UPDATE` (`set_doc_fields`) on the
        // heaviest path (audit T1 / I-N1). Columns are in declaration order; an absent payload
        // field binds NULL (the column's implicit default), so the result matches the old
        // insert-then-update path exactly.
        let filt_cols: Vec<&str> = self
            .schema
            .filterable_columns()
            .iter()
            .map(|(n, _)| n.as_str())
            .collect();
        let doc_ins_sql = {
            let mut cols = String::from("id, key");
            let mut binds = String::from("?1, ?2");
            for (i, col) in filt_cols.iter().enumerate() {
                cols.push_str(&format!(", \"{col}\"")); // double-quoted: keyword-safe column
                binds.push_str(&format!(", ?{}", i + 3));
            }
            format!("INSERT INTO {}({cols}) VALUES({binds})", ns.doc_shadow())
        };
        {
            let mut doc_ins = tx.prepare_cached(&doc_ins_sql)?;
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
                let mut doc_params: Vec<Value> = Vec::with_capacity(2 + filt_cols.len());
                doc_params.push(Value::Integer(doc_id));
                doc_params.push(doc.key.to_value());
                for &col in &filt_cols {
                    let v = doc
                        .payload
                        .iter()
                        .find(|(n, _)| n.as_str() == col)
                        .map(|(_, v)| v.clone())
                        .unwrap_or(Value::Null);
                    doc_params.push(v);
                }
                doc_ins.execute(rusqlite::params_from_iter(doc_params))?;
                for (label, text) in &doc.segments {
                    if !self.schema.accepts_label(label) {
                        return Err(Error::InvalidInput(format!(
                            "label {label:?} is not a field of the schema"
                        )));
                    }
                    let seg_id = next_seg;
                    next_seg += 1;
                    // Same lowering as the incremental path (one tokenize pass yields both the
                    // term-ids and the gram-count length — audit I4/C2-F1), but assigning dense
                    // ids from a rebuild-local allocator.
                    let (ids, seg_len) = self.distinct_term_ids(text, |term| {
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
                    total_seg_len += seg_len;
                    seg_ins.execute(rusqlite::params![
                        seg_id,
                        doc_id,
                        label,
                        text.as_str(),
                        seg_len
                    ])?;
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
        // The rolling segment-length stats for the freshly-built corpus (segments = next_seg - 1).
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
        // held write lease — so no reader splices the old map onto the new snapshot. The SQL
        // swap is already durable; if this reload fails we must NOT keep serving from the now
        // stale in-memory map (its ids point at the OLD term space — they would silently
        // mis-route reads and writes), so poison the index: every lease fails closed until the
        // caller reopens (the on-disk state is the consistent new generation) (audit C2-FB-C1).
        if let Err(e) = self.dict.load(&guard, ns) {
            self.poisoned.store(true, Ordering::Release);
            return Err(e);
        }
        // A successful rebuild rebuilds the dictionary from scratch, so it also clears any
        // poison a previous failed reload left behind — `rebuild` is a recovery path.
        self.poisoned.store(false, Ordering::Release);
        // The rebuilt corpus can have a wholly different df distribution; drop the accumulated
        // band-spread samples so the weight-step hint reflects only the new corpus (audit T15).
        self.reset_band_spread_hist();
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
            weight_step_hint: self.weight_step_hint(),
        })
    }

    /// Derive a [`WeightStepHint`] from the in-memory band-spread histogram (the searches run
    /// since open). `None` until at least one search has contributed.
    fn weight_step_hint(&self) -> Option<WeightStepHint> {
        let hist: [u64; HINT_BUCKETS] =
            std::array::from_fn(|b| self.band_spread_hist[b].load(Ordering::Relaxed));
        let total: u64 = hist.iter().sum();
        if total == 0 {
            return None;
        }
        // Nearest-rank quantile over the bucketed spreads, reported at the bucket midpoint.
        let quantile = |p: f64| -> f64 {
            let target = (p * total as f64).ceil().max(1.0) as u64;
            let mut cum = 0u64;
            for (b, &count) in hist.iter().enumerate() {
                cum += count;
                if cum >= target {
                    return b as f64 * HINT_BUCKET_WIDTH + HINT_BUCKET_WIDTH / 2.0;
                }
            }
            (HINT_BUCKETS - 1) as f64 * HINT_BUCKET_WIDTH + HINT_BUCKET_WIDTH / 2.0
        };
        let median_spread = quantile(0.5);
        // A median band spanning ~3 tier steps uses weights 1–4, so D ≈ median / 3; floor it
        // at a sane sharpest practical value so the suggestion is always a usable `D > 0`.
        let suggested = (median_spread / 3.0).max(0.5);
        Some(WeightStepHint {
            suggested,
            median_spread,
            iqr: (quantile(0.25), quantile(0.75)),
            samples: total,
        })
    }

    // ----- reads (implementation; the public read surface is `Reader`) --------

    /// Search a batch of queries, one result list per query in order. Invoked by a
    /// [`Reader`]; not public on `Index` (reads go through a [`reader`](Self::reader)
    /// lease). Checks out a fresh pooled connection, then runs the shared body.
    fn search_batch(&self, queries: &[&str], opts: &SearchOpts<'_>) -> Result<Vec<Vec<Match>>> {
        let conn = self.backend.read()?;
        self.search_batch_on(&conn, queries, opts)
    }

    /// The shared read body on a **given** connection — used by the [`Reader`] (a fresh
    /// per-search checkout, via [`search_batch`](Self::search_batch)) and by the warm
    /// [`SearchSession`] (its held connection). Tokenizes, then runs the
    /// snapshot/generation-guarded pipeline ([`search`]). Each query's selection and ranking
    /// derive only from its own token frequencies, so a batch ranks each query identically to
    /// a singleton search (batch == serial).
    fn search_batch_on(
        &self,
        conn: &Connection,
        queries: &[&str],
        opts: &SearchOpts<'_>,
    ) -> Result<Vec<Vec<Match>>> {
        if queries.is_empty() {
            return Ok(Vec::new());
        }
        let (query_tokens, all_grams) = search::query_grams(queries, |q| self.distinct_tokens(q));
        search::search_read_on(self, conn, &all_grams, |conn, ns, resolved, class_snap| {
            search::SearchCtx::new(self, conn, ns, resolved, class_snap, opts)
                .run_search(queries, &query_tokens)
        })
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
    /// Whether a transaction is currently open. `false` once the writer is stranded — a
    /// re-`BEGIN` after a `commit()` failed, or a savepoint rollback faulted — so every
    /// method returns an error rather than silently running in autocommit (losing atomicity).
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
            return Err(Error::writer_stranded(
                "the writer is no longer usable (a prior commit or rollback could not maintain its transaction); re-acquire the writer",
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
                // Undo the SQL and the in-memory staging this call accumulated, so the store
                // is left exactly as before the call. If the savepoint undo itself faults (a
                // connection-level failure), escalate to aborting the whole transaction and
                // poison the writer — that prevents a later `commit()` from persisting a torn
                // savepoint, at the cost of the uncommitted tail (within the lease contract).
                if self
                    .guard
                    .execute_batch("ROLLBACK TO trifle_w; RELEASE trifle_w")
                    .is_err()
                {
                    let _ = self.guard.execute_batch("ROLLBACK");
                    self.txn_open = false;
                }
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
                // Subtract every removed segment's gram length + count from the rolling stats.
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
            Ok(())
        })
    }

    /// Set the document's filterable field values (Tier 2). The document **must already
    /// exist** (have at least one segment); calling `set_fields` on a key with no segments
    /// returns [`Error::InvalidInput`] rather than creating a payload-only "ghost" `doc` row
    /// that no search can ever return — a typo'd key can't silently accrete rows (audit T3 /
    /// A4). Undeclared field names are ignored. Independent of the segment methods otherwise.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidInput`] if `key` names no existing document, or a store error if
    /// the write fails.
    pub fn set_fields(&mut self, key: impl Into<Key>, fields: &[(&str, Value)]) -> Result<()> {
        let key = key.into();
        self.atomic(|w| {
            let ns = w.index.backend.namespace();
            let conn: &Connection = &w.guard;
            // create=false: a document is created only by inserting segments, and emptying one
            // reaps its row (C2-RA-1), so "no doc row" == "no such document". Refuse rather than
            // stage payload onto a row search can never surface.
            let Some(doc) = w.index.doc_id_for(conn, ns, &key, false)? else {
                return Err(Error::InvalidInput(format!(
                    "set_fields on key {key:?} which has no segments; insert the document's \
                     segments before setting its filterable fields"
                )));
            };
            let owned: Vec<(String, Value)> = fields
                .iter()
                .map(|(n, v)| (n.to_string(), v.clone()))
                .collect();
            w.index.set_doc_fields(conn, ns.doc(), doc, &owned)?;
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
    /// The two failure points need different caller responses, so they surface as different
    /// errors:
    /// - the **`COMMIT` itself fails** ([`Error::Sqlite`]/[`Error::Busy`]): the batch was
    ///   **not** made durable (it rolls back); retry it on a fresh writer.
    /// - the **`COMMIT` succeeds but the follow-on `BEGIN` fails**
    ///   ([`Error::WriterStranded`]): the batch **is** durable — do not retry it; the writer
    ///   is unusable, so drop it and acquire a fresh one to continue.
    pub fn commit(&mut self) -> Result<()> {
        // A failure here rolls the batch back (Drop's ROLLBACK / connection close); the
        // transaction is still open, so the writer remains usable for a retry.
        self.guard.execute_batch("COMMIT")?;
        // The COMMIT succeeded and is durable; the txn is now closed. If the re-`BEGIN` below
        // fails we must not leave the writer silently in autocommit, so mark it closed up
        // front and only re-open on success.
        self.txn_open = false;
        // Order matters: merge the staged interns first (advancing the shared high-water
        // mark), then snapshot a fresh stage off the advanced mark.
        if let Some(old) = self.stage.take() {
            old.commit();
        }
        let df = std::mem::take(&mut self.pending_df);
        self.index.dict.apply_df_changes(&df);
        self.stage = Some(self.index.dict.stage());
        // The batch is already durable; a re-`BEGIN` failure only strands this writer — say so
        // distinctly so the caller re-acquires instead of re-applying the committed batch.
        if let Err(e) = self.guard.execute_batch("BEGIN IMMEDIATE") {
            return Err(Error::writer_stranded(format!(
                "the batch committed durably, but the writer could not begin a new transaction \
                 (do not retry the committed batch; acquire a fresh writer to continue): {e}"
            )));
        }
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
        self.index.search_batch_on(&self.conn, queries, &opts)
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
        // None / Custom(0) never over-fetch; the pool stays at `limit`.
        assert_eq!(Effort::None.pool(10, 1_000_000), 10);
        assert_eq!(Effort::Custom(0.0).pool(10, 1_000_000), 10);
    }
}

#[cfg(test)]
mod poison_tests {
    use super::*;

    /// A poisoned index (a post-commit dict reload failed — audit C2-FB-C1) must fail every
    /// lease/maintenance entry point closed, and a later successful `rebuild` must clear it.
    #[test]
    fn poison_fails_leases_closed_and_rebuild_recovers() {
        let dir = tempfile::tempdir().unwrap();
        let idx =
            Index::open_at(&dir.path().join("t.db"), Schema::flat(), Config::default()).unwrap();
        assert!(idx.reader().is_ok(), "healthy index opens a reader");

        // Simulate the reload failure that poisons the index.
        idx.poisoned.store(true, Ordering::Release);
        assert!(matches!(idx.reader().err(), Some(Error::Corrupt(_))));
        assert!(matches!(idx.writer().err(), Some(Error::Corrupt(_))));
        assert!(matches!(idx.session().err(), Some(Error::Corrupt(_))));
        assert!(matches!(idx.compact().err(), Some(Error::Corrupt(_))));

        // A successful rebuild rebuilds the in-memory dictionary, so it is a recovery path.
        idx.rebuild(std::iter::empty()).unwrap();
        assert!(idx.reader().is_ok(), "rebuild cleared the poison");
        assert!(idx.writer().is_ok());
    }

    /// A stranded writer (a `commit()` that committed durably but could not re-`BEGIN`, or a
    /// savepoint-rollback fault) fails its methods with [`Error::WriterStranded`] — store
    /// intact, re-acquire — not [`Error::Corrupt`] (which would wrongly imply a rebuild).
    #[test]
    fn a_stranded_writer_reports_writer_stranded() {
        let dir = tempfile::tempdir().unwrap();
        let idx =
            Index::open_at(&dir.path().join("t.db"), Schema::flat(), Config::default()).unwrap();
        let mut w = idx.writer().unwrap();
        // Reproduce exactly what a COMMIT-succeeded-but-reBEGIN-failed leaves: the open txn
        // closed (so nothing is open to roll back) and the writer marked stranded.
        w.guard.execute_batch("COMMIT").unwrap();
        w.txn_open = false;
        assert!(matches!(
            w.insert(1, &[("body", "x")]),
            Err(Error::WriterStranded(_))
        ));
        assert!(matches!(w.remove(1), Err(Error::WriterStranded(_))));
        // A fresh writer is unaffected — the store was never harmed.
        drop(w);
        assert!(idx.writer().is_ok());
    }
}
