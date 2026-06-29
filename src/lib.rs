//! `trifle` — embedded, typo-tolerant trigram fuzzy search backed by SQLite.
//!
//! trifle indexes short text **segments** and answers typo/partial-tolerant queries,
//! returning matches each carrying *where* it matched. It owns a single SQLite store holding the
//! segment text, the caller key, and an owned **inverted index** (a base+delta CRoaring posting
//! per token); it ranks by **IDF-weighted, class-normalized shared-rare-token overlap**, counted
//! bit-sliced in the [`trifle_overlap`] engine. It is a **derived, rebuildable cache** over a
//! caller-owned source of truth: it never touches the caller's data store.
//!
//! It targets a specific regime — **large corpora of small documents** (≲ 1–2 KB per segment),
//! read-often / write-infrequent. It is a **fuzzy lexical overlap engine, not a relevance
//! engine**: rarer shared grams weigh more, and rarity is **class-normalized across scripts** so
//! a CJK bigram and a Latin trigram compare fairly. There is no BM25/relevance tier.
//!
//! # Quick start
//!
//! ```
//! use trifle::{Config, Index, Schema, SearchOpts};
//!
//! # fn main() -> trifle::Result<()> {
//! let dir = tempfile::tempdir().unwrap();
//! let index = Index::open_at(&dir.path().join("trifle.db"), Schema::flat(), Config::default())?;
//!
//! // Writes go through a short-lived writer lease; commit makes them durable + visible.
//! let mut w = index.writer()?;
//! w.upsert(1, &[("front", "the quick brown fox")])?;
//! w.upsert(2, &[("front", "the quack brown ox")])?;
//! w.commit()?;
//! drop(w);
//!
//! // Reads go through a reader lease. A typo'd query still matches.
//! let hits = index.reader()?.matches("quikc brown", &SearchOpts::new(), 10)?;
//! assert!(hits.iter().any(|m| m.key.as_i64() == Some(1)));
//! # Ok(())
//! # }
//! ```
//!
//! # The data model
//!
//! A [`Schema`] declares a **key** (its shape — `Integer`/`Text`/`Blob`) plus named **text
//! fields**. A [`Document`] is a [`Key`] and a set of named segments (`label → text`); a match
//! comes back as a [`Match`] carrying the document key, the matched segment's label, the matched
//! byte span, and the segment's text. There is no `doc` table: the key lives directly on each
//! segment row.
//!
//! # The read surface
//!
//! [`Reader::matches`] is the eager safe default (top-`limit` hydrated matches). [`Reader::candidates`]
//! is the lazy **[`CandidateStream`]** spine — a best-first cursor of provenance-only
//! [`Candidate`]s the caller composes rerank / pagination / fusion on top of, hydrating only
//! what it keeps. Filtering is the opt-in [`SqlFilter`] over the caller's *live* data.
//!
//! # Ingest & maintenance
//!
//! Writes are cheap (a small delta append) and instantly visible after [`commit`](Writer::commit).
//! Fold pending deltas with [`compact`](Index::compact) on a cadence driven by
//! [`Stats::delta_backlog`]; [`rebuild`](Index::rebuild) fully reindexes via an atomic shadow swap
//! (required after a tokenizer or `data_version`/schema change, which drop the cache on open).

#![forbid(unsafe_op_in_unsafe_fn)]

mod instrument;

pub mod error;
pub mod store;
pub mod tokenize;

mod dict;
mod filter;
mod hash;
mod model;
mod postings;
mod schema;
mod search;
mod select;
mod term;
mod welford;

use crate::hash::{FxHashMap, FxHashSet};
use std::borrow::Borrow;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use croaring::Bitmap;
use rusqlite::types::Value;
use rusqlite::{Connection, OptionalExtension};

/// Re-export of the `rusqlite` version trifle is built against, so a caller binding
/// [`SqlFilter`] params uses exactly the right `ToSql`/`Value` types.
pub use rusqlite;

use dict::{Dictionary, TermId};
pub use error::{Error, Result};
pub use filter::SqlFilter;
pub use model::{Document, Key, KeyShape, Match, Schema, SchemaBuilder};
use schema::SCHEMA_VERSION;
pub use search::{Candidate, CandidateStream};
use store::{Namespace, Sidecar};
pub use term::{IntoTerm, Term};
use tokenize::{DefaultTokenizer, Tokenizer};

/// Default match floor `m` — shared rare tokens required for a hit.
pub(crate) const DEFAULT_MIN_SHARED: u32 = 2;
/// Per-typo token damage `d`; the typo floor is `F = m + d`.
pub(crate) const TYPO_DAMAGE: u32 = 4;
/// Default `t_max` — the number of rarest query tokens selection keeps.
pub(crate) const DEFAULT_T_MAX: usize = 12;
/// Default `D` — df-doublings per IDF weight step in the overlap counter.
const DEFAULT_WEIGHT_STEP: f64 = 1.0;
/// Buckets in the per-query band-spread histogram (the [`Stats`] weight-step hint). 33 × 0.5 =
/// 16.5 df-doublings of range (a df ratio up to ~92 000:1).
const HINT_BUCKETS: usize = 33;
/// Width of each band-spread histogram bucket, in df-doublings (`log2` units).
const HINT_BUCKET_WIDTH: f64 = 0.5;

/// Index configuration.
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

/// Per-search options. Construct with [`SearchOpts::new`] and the builder setters.
///
/// `limit` is **not** here — it is a terminal-op argument ([`Reader::matches`]), because the
/// [`candidates`](Reader::candidates) stream is lazy/unbounded (the caller pulls depth via
/// `take`). `#[non_exhaustive]`: build from [`new`](SearchOpts::new), not a struct literal.
#[non_exhaustive]
pub struct SearchOpts<'a> {
    /// `m` — the match floor (shared rare tokens for a hit). `None` → `2`.
    pub min_shared: Option<u32>,
    /// `t_max` — the number of rarest query tokens selection keeps, above the typo floor
    /// `F = m + d`. `None` → `12`. The selection breadth (recall/cost) knob.
    pub t_max: Option<usize>,
    /// `B` — an optional cap on the cumulative document frequency (`Σdf`) of the selected
    /// tokens, which is what candidate generation scans — so this bounds *work* directly
    /// (`t_max` bounds count). Rarest-first tokens are kept while `Σdf` stays within budget; the
    /// typo floor is always kept. `None` (the default) = no cap.
    pub df_budget: Option<u64>,
    /// `D` — df-doublings per IDF weight step in the overlap counter. `1.0` (the default) means
    /// each weight level is one more halving of df relative to the query's commonest survivor.
    /// `N`-invariant. [`Stats::weight_step_hint`] suggests a corpus-fitted value.
    pub weight_step: f64,
    /// An opt-in raw-SQL [`SqlFilter`] over the caller's live data, folded into per-bucket
    /// provenance. `None` = unfiltered.
    pub filter: Option<SqlFilter<'a>>,
}

impl<'a> SearchOpts<'a> {
    /// Default options (weighted-overlap order, weight step `D = 1.0`, no filter).
    pub fn new() -> Self {
        SearchOpts {
            min_shared: None,
            t_max: None,
            df_budget: None,
            weight_step: DEFAULT_WEIGHT_STEP,
            filter: None,
        }
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

    /// Set `df_budget` `B` — cap the cumulative `Σdf` of the selected tokens (bounds scan work).
    pub fn df_budget(mut self, budget: u64) -> Self {
        self.df_budget = Some(budget);
        self
    }

    /// Set `D`, the df-doublings per IDF weight step.
    pub fn weight_step(mut self, d: f64) -> Self {
        self.weight_step = d;
        self
    }

    /// Set an opt-in [`SqlFilter`].
    pub fn filter(mut self, filter: SqlFilter<'a>) -> Self {
        self.filter = Some(filter);
        self
    }
}

impl Default for SearchOpts<'_> {
    fn default() -> Self {
        SearchOpts::new()
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
    /// On-disk size of the backing database file, in bytes.
    pub disk_bytes: u64,
    /// The caller drift token currently stamped.
    pub data_version: u64,
    /// The tokenizer fingerprint currently stamped.
    pub tokenizer_fingerprint: u64,
    /// trifle's on-disk schema version.
    pub schema_version: u32,
    /// The schema fingerprint currently stamped.
    pub schema_fingerprint: u64,
    /// A corpus-derived suggestion for [`SearchOpts::weight_step`] `D`, accumulated from the
    /// band-spreads of the searches run since this index was opened. `None` until at least one
    /// search has run. See [`WeightStepHint`].
    pub weight_step_hint: Option<WeightStepHint>,
}

/// A suggested [`SearchOpts::weight_step`] `D`, with the band-spread distribution it came from.
///
/// Built from the per-query band-spreads (`log2(df_max/df_min)`) observed since the index was
/// opened. `suggested ≈ median / 3`. The `iqr` is the confidence signal: a tight IQR means one
/// `D` fits; a wide one means the corpus has multiple query regimes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WeightStepHint {
    /// The suggested `D` (`max(0.5, median_spread / 3)`).
    pub suggested: f64,
    /// Median per-query band-spread, in df-doublings.
    pub median_spread: f64,
    /// The interquartile range `(Q1, Q3)` of band-spreads, in df-doublings.
    pub iqr: (f64, f64),
    /// How many searches contributed.
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
/// Generic over the [`Tokenizer`] (monomorphized — it is on the hot path), defaulted so the
/// common case is just `Index`. Open with [`open_at`](Index::open_at) (the default tokenizer + an
/// owned sidecar file) or [`open`](Index::open) (a custom tokenizer / namespace).
///
/// All methods are synchronous and thread-safe (`&self`): a single internal writer is serialized,
/// reads run on a pooled connection concurrently with the writer under WAL. An async caller
/// dispatches to a blocking pool.
pub struct Index<T: Tokenizer = DefaultTokenizer> {
    pub(crate) store: Sidecar,
    pub(crate) tokenizer: T,
    data_version: u64,
    /// The declared data model (key shape, text fields).
    pub(crate) schema: Schema,
    /// The in-memory faulting term dictionary (gram → `u32` id) + per-class df statistics,
    /// shared by the writer and the read pool. Hydrated on open; rebuilt under the swap on
    /// `rebuild`.
    pub(crate) dict: Dictionary,
    /// Set if [`rebuild`](Self::rebuild)'s in-memory dictionary reload failed *after* its SQL
    /// swap committed: every lease then fails closed until the caller reopens (or a later
    /// `rebuild` succeeds), rather than serving from a stale, mis-routed map.
    poisoned: AtomicBool,
    /// In-memory per-query band-spread histogram: each search adds one sample
    /// `log2(df_max/df_min)` over its present postings. [`stats`](Self::stats) derives a suggested
    /// [`weight_step`](SearchOpts::weight_step) from it. Process-local; reset by reopening.
    band_spread_hist: [AtomicU64; HINT_BUCKETS],
}

impl Index<DefaultTokenizer> {
    /// Open (creating if absent) an index at `path` with the given [`Schema`], the default
    /// script-segmenting tokenizer, and an owned [`Sidecar`] file. The common case.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened or the store cannot be initialized.
    pub fn open_at(path: &Path, schema: Schema, config: Config) -> Result<Self> {
        let store = Sidecar::open(path)?;
        Index::open(store, DefaultTokenizer::new(), schema, config)
    }
}

impl<T: Tokenizer> Index<T> {
    /// Open an index over a given [`Sidecar`] store and tokenizer.
    ///
    /// On open, the store is created if absent and checked against version stamps (schema,
    /// tokenizer fingerprint, caller `data_version`); any mismatch — or a broken id-allocation
    /// invariant — drops the cache to empty (no migrations). After such a reset,
    /// [`rebuild`](Self::rebuild) repopulates it.
    ///
    /// # Errors
    ///
    /// Returns an error if the store cannot be initialized.
    pub fn open(store: Sidecar, tokenizer: T, schema: Schema, config: Config) -> Result<Self> {
        let index = Index {
            store,
            tokenizer,
            data_version: config.data_version,
            schema,
            dict: Dictionary::empty(),
            poisoned: AtomicBool::new(false),
            band_spread_hist: std::array::from_fn(|_| AtomicU64::new(0)),
        };
        index.init()?;
        {
            let conn = index.store.read()?;
            index.dict.load(&conn, index.store.namespace())?;
        }
        Ok(index)
    }

    /// Create tables and reconcile drift, all in one transaction.
    fn init(&self) -> Result<()> {
        let mut guard = self.store.write()?;
        let ns = self.store.namespace();
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
            // A reset reassigns the (now empty) id space; bump the generation so any concurrent
            // reader detects the change.
            schema::bump_dict_generation(&tx, ns)?;
            schema::write_stamps(&tx, ns, self.data_version, fingerprint, schema_fp)?;
            self.reset_band_spread_hist();
        }
        tx.commit()?;
        Ok(())
    }

    /// Fail closed if a prior `rebuild` left the in-memory dictionary stale. Called at every
    /// lease/maintenance entry point so a poisoned index never serves a search or accepts a write
    /// against a mis-routed map.
    pub(crate) fn check_poisoned(&self) -> Result<()> {
        if self.poisoned.load(Ordering::Acquire) {
            return Err(Error::corrupt(
                "index poisoned: an in-memory dictionary reload failed after a committed \
                 rebuild; reopen the index (its on-disk state is intact) to recover",
            ));
        }
        Ok(())
    }

    /// Record one query's band-spread (`log2(df_max/df_min)` over its present postings) into the
    /// in-memory histogram backing the [`Stats`] weight-step hint. Only a query with **≥ 2**
    /// present postings is sampled (a band needs two endpoints). Cheap: one `log2` + one atomic add.
    pub(crate) fn observe_band_spread(&self, present_dfs: &[u64]) {
        let (mut lo, mut hi, mut n) = (u64::MAX, 0u64, 0u32);
        for &df in present_dfs {
            if df > 0 {
                lo = lo.min(df);
                hi = hi.max(df);
                n += 1;
            }
        }
        if n < 2 {
            return;
        }
        let spread = (hi as f64 / lo.max(1) as f64).log2();
        let bucket = ((spread / HINT_BUCKET_WIDTH) as usize).min(HINT_BUCKETS - 1);
        self.band_spread_hist[bucket].fetch_add(1, Ordering::Relaxed);
    }

    /// Zero the band-spread histogram. Called on a successful [`rebuild`](Self::rebuild) and on a
    /// drift/desync reset (both can shift the corpus df distribution). A [`compact`](Self::compact)
    /// leaves every df unchanged, so it does **not** reset.
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

    /// Acquire the exclusive [`Writer`] lease: holds the single-writer lock for the lease's
    /// lifetime and opens a transaction. Keep it **short-lived** — acquire → batch →
    /// [`commit`](Writer::commit) → drop. Dropping without committing rolls back the uncommitted
    /// tail.
    ///
    /// # Errors
    ///
    /// Returns an error if the write transaction cannot begin.
    pub fn writer(&self) -> Result<Writer<'_, T>> {
        self.check_poisoned()?;
        Writer::begin(self)
    }

    /// Acquire a [`Reader`] lease. The snapshot is **per search/stream**: each
    /// [`matches`](Reader::matches) checks out its own pooled connection and opens its own
    /// consistent WAL snapshot, and each [`candidates`](Reader::candidates) stream holds its own
    /// for its lifetime. So a held reader's next search observes writes committed since its
    /// previous one. For one snapshot across related queries, use a single
    /// [`matches_batch`](Reader::matches_batch).
    ///
    /// # Errors
    ///
    /// Returns an error if the lease cannot be set up.
    pub fn reader(&self) -> Result<Reader<'_, T>> {
        self.check_poisoned()?;
        Ok(Reader { index: self })
    }

    // ----- write internals (used by the `Writer` lease) -----------------------

    /// Tokenize `text` once, returning both its distinct resolved term-ids (each assigned via
    /// `assign`) and its total gram count with repetition — the stored segment length `|d|`.
    fn distinct_term_ids(
        &self,
        text: &str,
        mut assign: impl FnMut(Term) -> Result<TermId>,
    ) -> Result<(Vec<TermId>, i64)> {
        let mut distinct: FxHashSet<T::Token> = FxHashSet::default();
        let mut seg_len: i64 = 0;
        for tok in self.tokenizer.tokenize(text) {
            seg_len += 1;
            distinct.insert(tok);
        }
        let mut ids: Vec<TermId> = Vec::with_capacity(distinct.len());
        for tok in &distinct {
            let term = tok.term().ok_or_else(|| {
                Error::InvalidInput(format!(
                    "gram {:?} is not encodable as a term (over the 3-codepoint ceiling, or it \
                     contains U+0000)",
                    tok.borrow()
                ))
            })?;
            ids.push(assign(term)?);
        }
        Ok((ids, seg_len))
    }

    /// The segment id for `(key, label)`, if it exists.
    fn find_segment(
        &self,
        conn: &Connection,
        ns: &Namespace,
        key: &Key,
        label: &str,
    ) -> Result<Option<i64>> {
        Ok(conn
            .query_row(
                &format!("SELECT id FROM {} WHERE key = ?1 AND label = ?2", ns.seg()),
                rusqlite::params![key.to_value(), label],
                |r| r.get(0),
            )
            .optional()?)
    }

    /// Write one segment `(key, label) = text`, interning its grams through `stage` and storing
    /// the term-id set in `fwd` (so delete needs no text). If a segment with this `(key, label)`
    /// already exists: with `replace` it is dropped first (its removals into `changes`); without
    /// `replace`, this errors.
    #[allow(clippy::too_many_arguments)]
    fn write_segment(
        &self,
        conn: &Connection,
        ns: &Namespace,
        key: &Key,
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
        if let Some(seg_id) = self.find_segment(conn, ns, key, label)? {
            if !replace {
                return Err(Error::InvalidInput(format!(
                    "a segment with label {label:?} already exists for this key"
                )));
            }
            self.drop_segment(conn, ns, seg_id, changes)?;
        }
        let id = schema::alloc_ids(conn, ns, 1)?;
        let (ids, seg_len) =
            self.distinct_term_ids(text, |term| stage.intern_term(term, conn, ns))?;
        conn.execute(
            &format!(
                "INSERT INTO {}(id, key, label, txt, len) VALUES(?1, ?2, ?3, ?4, ?5)",
                ns.seg()
            ),
            rusqlite::params![id, key.to_value(), label, text, seg_len],
        )?;
        let bm: Bitmap = ids.iter().copied().collect();
        conn.execute(
            &format!("INSERT INTO {}(id, tokens) VALUES(?1, ?2)", ns.fwd()),
            rusqlite::params![id, postings::serialize(&bm)],
        )?;
        changes.add(id as u32, &ids);
        schema::bump_seg_stats(conn, ns, 1, seg_len)?;
        Ok(())
    }

    /// Drop one segment by id: accumulate its term-id removals into `changes`, then delete its
    /// `seg` and `fwd` rows and back out its length from the rolling stats.
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

    /// The distinct tokens of `text`, deduplicated via the token type. Returns the tokens
    /// themselves (the read path resolves and selects in term-space, stringifying only the
    /// tokens that reach the result).
    pub(crate) fn distinct_tokens(&self, text: &str) -> Vec<T::Token> {
        let distinct: FxHashSet<T::Token> = self.tokenizer.tokenize(text).collect();
        distinct.into_iter().collect()
    }

    /// Read the stored `fwd` term-id sets for a set of segment ids.
    fn read_fwd(
        &self,
        conn: &Connection,
        ns: &Namespace,
        ids: &[u32],
    ) -> Result<FxHashMap<u32, Vec<TermId>>> {
        let mut out = FxHashMap::with_capacity_and_hasher(ids.len(), Default::default());
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

    // ----- maintenance --------------------------------------------------------

    /// Fold pending deltas into bases, drop emptied tokens, and prune zero-frequency terms.
    /// Heavier for common tokens; call on a schedule or when idle. Does not shrink the file.
    ///
    /// # Errors
    ///
    /// Returns an error if the store write fails.
    pub fn compact(&self) -> Result<CompactStats> {
        self.check_poisoned()?;
        let mut guard = self.store.write()?;
        let ns = self.store.namespace();
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

    /// Fully reindex from `corpus` via an atomic shadow swap: build into shadow tables, then
    /// drop-and-rename them in one transaction so a reader sees complete-old or complete-new.
    /// Reassigns dense ids and stamps the current versions.
    ///
    /// Required after a tokenizer change or a `data_version` bump (both empty the cache on open),
    /// and useful to reclaim space.
    ///
    /// # Errors
    ///
    /// Returns an error if the store write fails; on failure the live index is left intact.
    pub fn rebuild(&self, corpus: impl IntoIterator<Item = Document>) -> Result<()> {
        let mut guard = self.store.write()?;
        let ns = self.store.namespace();
        let tx = guard.transaction()?;
        schema::create_shadows(&tx, ns, &self.schema)?;

        // Accumulate the inverted index in memory while streaming seg rows to the shadow tables.
        // Two id spaces are reassigned dense (segment, term); the inverted index is keyed by seg
        // id and the dictionary by the gram's packed u128 (stored 16-byte big-endian).
        let mut local: FxHashMap<u128, TermId> = FxHashMap::default();
        let mut next_term: u64 = 1;
        let mut inverted: FxHashMap<TermId, Bitmap> = FxHashMap::default();
        let mut next_seg: i64 = 1;
        let mut total_seg_len: i64 = 0;
        {
            let mut seg_ins = tx.prepare_cached(&format!(
                "INSERT INTO {}(id, key, label, txt, len) VALUES(?1, ?2, ?3, ?4, ?5)",
                ns.seg_shadow()
            ))?;
            let mut fwd_ins = tx.prepare_cached(&format!(
                "INSERT INTO {}(id, tokens) VALUES(?1, ?2)",
                ns.fwd_shadow()
            ))?;
            let mut seen_keys: FxHashSet<Key> = FxHashSet::default();
            for doc in corpus {
                // A segment-less document materializes no rows (no ghost is possible without a
                // doc table) — skip it.
                if doc.segments.is_empty() {
                    continue;
                }
                // The key variant must match the schema's declared shape (the same caller contract
                // the incremental path debug-asserts): otherwise SQLite affinity coercion can make
                // two Rust-distinct keys collide. Debug-only; release behavior unchanged.
                debug_assert!(
                    matches!(
                        (&doc.key, self.schema.key_shape()),
                        (Key::Integer(_), KeyShape::Integer)
                            | (Key::Text(_), KeyShape::Text)
                            | (Key::Blob(_), KeyShape::Blob)
                    ),
                    "rebuild key variant {:?} does not match the declared key shape {:?}",
                    doc.key,
                    self.schema.key_shape()
                );
                // Each key appears once in the rebuild corpus; fail fast on a duplicate with a
                // clear message rather than the opaque late UNIQUE(key,label) the swap would raise.
                if !seen_keys.insert(doc.key.clone()) {
                    return Err(Error::InvalidInput(format!(
                        "duplicate key {:?} in the rebuild corpus; each document must have a \
                         unique key",
                        doc.key
                    )));
                }
                for (label, text) in &doc.segments {
                    if !self.schema.accepts_label(label) {
                        return Err(Error::InvalidInput(format!(
                            "label {label:?} is not a field of the schema"
                        )));
                    }
                    let seg_id = next_seg;
                    next_seg += 1;
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
                        doc.key.to_value(),
                        label,
                        text.as_str(),
                        seg_len
                    ])?;
                    for &tid in &ids {
                        inverted.entry(tid).or_default().add(seg_id as u32);
                    }
                    let bm: Bitmap = ids.iter().copied().collect();
                    fwd_ins.execute(rusqlite::params![seg_id, postings::serialize(&bm)])?;
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
        schema::set_seg_stats(&tx, ns, next_seg - 1, total_seg_len)?;
        schema::bump_dict_generation(&tx, ns)?;
        schema::write_stamps(
            &tx,
            ns,
            self.data_version,
            self.tokenizer.fingerprint(),
            self.schema.fingerprint(),
        )?;
        tx.commit()?;
        // Re-hydrate the in-memory dictionary from the swapped-in tables, still under the write
        // lease. If this reload fails we must NOT keep serving from the stale map (its ids point
        // at the OLD term space), so poison the index: every lease fails closed until reopen.
        if let Err(e) = self.dict.load(&guard, ns) {
            self.poisoned.store(true, Ordering::Release);
            return Err(e);
        }
        self.poisoned.store(false, Ordering::Release);
        self.reset_band_spread_hist();
        Ok(())
    }

    /// A read-only snapshot of observable state.
    ///
    /// # Errors
    ///
    /// Returns an error if the store read fails.
    pub fn stats(&self) -> Result<Stats> {
        let ns = self.store.namespace();
        let conn = self.store.read()?;
        // One pinned snapshot so the reported fields are mutually consistent; segment count is the
        // O(1) rolling meta counter, not an O(N) `count(*)`.
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

    /// Derive a [`WeightStepHint`] from the band-spread histogram (the searches run since open).
    /// `None` until at least one search has contributed.
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
        let suggested = (median_spread / 3.0).max(0.5);
        Some(WeightStepHint {
            suggested,
            median_spread,
            iqr: (quantile(0.25), quantile(0.75)),
            samples: total,
        })
    }
}

/// Whether `segments` names any label more than once — a caller-contract violation (a document
/// holds each label at most once). Used only by a `debug_assert`; `O(n²)` over a tiny slice.
fn has_duplicate_label(segments: &[(&str, &str)]) -> bool {
    segments
        .iter()
        .enumerate()
        .any(|(i, (label, _))| segments[..i].iter().any(|(prev, _)| prev == label))
}

/// The exclusive write lease: holding it **is** holding the single-writer lock. One transaction
/// is open for the lease; [`commit`](Writer::commit) decouples durability from the lease
/// (commit-and-continue), and dropping without committing rolls the uncommitted tail back.
///
/// Three write methods: `upsert` (create-or-replace named segments, other labels intact),
/// `remove` (drop a key and all its segments), `remove_segment` (drop one).
pub struct Writer<'a, T: Tokenizer = DefaultTokenizer> {
    guard: std::sync::MutexGuard<'a, Connection>,
    index: &'a Index<T>,
    /// The interning session, accumulated across the open transaction; merged into the shared
    /// dictionary only on [`commit`](Self::commit), discarded on rollback.
    stage: Option<dict::InternStage<'a>>,
    /// `(term-id, old_df, new_df)` deltas to fold into the per-class stats on commit.
    pending_df: Vec<(TermId, i64, i64)>,
    /// Whether a transaction is currently open. `false` once the writer is stranded.
    txn_open: bool,
}

impl<'a, T: Tokenizer> Writer<'a, T> {
    /// Acquire the writer lock and open the first transaction.
    fn begin(index: &'a Index<T>) -> Result<Self> {
        let guard = index.store.write()?;
        guard.execute_batch("BEGIN IMMEDIATE")?;
        Ok(Writer {
            guard,
            index,
            stage: Some(index.dict.stage()),
            pending_df: Vec::new(),
            txn_open: true,
        })
    }

    /// Run one write method's body inside a `SAVEPOINT`, so a mid-call error rolls back *all* of
    /// that call's effects — both the SQL (`ROLLBACK TO`) and the in-memory intern staging +
    /// pending df deltas — leaving the store exactly as before the call.
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

    /// Insert-or-replace the given `(label, text)` segments of the document keyed `key` (creating
    /// it if absent). Never errors on collision; a key's other (unnamed) segments are left intact.
    ///
    /// The labels in `segments` must be distinct (a document holds each label at most once);
    /// a duplicate label in one call is a contract violation asserted in debug builds.
    pub fn upsert(&mut self, key: impl Into<Key>, segments: &[(&str, &str)]) -> Result<()> {
        let key = key.into();
        self.atomic(|w| w.write_doc(&key, segments, true))
    }

    /// The shared body of [`upsert`](Self::upsert).
    fn write_doc(&mut self, key: &Key, segments: &[(&str, &str)], replace: bool) -> Result<()> {
        // Writing zero segments is the fold of zero segment-writes — a no-op (never a key-only
        // row; there is no doc table for one to live in).
        if segments.is_empty() {
            return Ok(());
        }
        let ns = self.index.store.namespace();
        let conn: &Connection = &self.guard;
        let stage = self.stage.as_mut().expect("writer stage present");
        let mut changes = TokenChanges::default();
        debug_assert!(
            !has_duplicate_label(segments),
            "upsert requires distinct labels within one call (a document holds each label once)"
        );
        for (label, text) in segments {
            self.index
                .write_segment(conn, ns, key, label, text, &mut changes, stage, replace)?;
        }
        let df = changes.apply(conn, ns)?;
        self.pending_df.extend(df);
        Ok(())
    }

    /// Remove the document keyed `key` and all its segments. A nonexistent key is a no-op.
    pub fn remove(&mut self, key: impl Into<Key>) -> Result<()> {
        let key = key.into();
        self.atomic(|w| {
            let ns = w.index.store.namespace();
            let conn: &Connection = &w.guard;
            let mut changes = TokenChanges::default();
            // All segment ids under this key.
            let ids: Vec<u32> = {
                let mut stmt =
                    conn.prepare_cached(&format!("SELECT id FROM {} WHERE key = ?1", ns.seg()))?;
                let mut rows = stmt.query(rusqlite::params![key.to_value()])?;
                let mut v = Vec::new();
                while let Some(r) = rows.next()? {
                    v.push(r.get::<_, i64>(0)? as u32);
                }
                v
            };
            if !ids.is_empty() {
                let fwd = w.index.read_fwd(conn, ns, &ids)?;
                for id in &ids {
                    if let Some(tokens) = fwd.get(id) {
                        changes.remove(*id, tokens);
                    }
                }
                let (seg_n, seg_len): (i64, i64) = conn.query_row(
                    &format!(
                        "SELECT count(*), coalesce(sum(len), 0) FROM {} WHERE key = ?1",
                        ns.seg()
                    ),
                    rusqlite::params![key.to_value()],
                    |r| Ok((r.get(0)?, r.get(1)?)),
                )?;
                let arr: std::rc::Rc<Vec<Value>> =
                    std::rc::Rc::new(ids.iter().map(|&i| Value::Integer(i as i64)).collect());
                conn.execute(
                    &format!("DELETE FROM {} WHERE id IN rarray(?1)", ns.fwd()),
                    rusqlite::params![arr],
                )?;
                conn.execute(
                    &format!("DELETE FROM {} WHERE key = ?1", ns.seg()),
                    rusqlite::params![key.to_value()],
                )?;
                schema::bump_seg_stats(conn, ns, -seg_n, -seg_len)?;
            }
            let df = changes.apply(conn, ns)?;
            w.pending_df.extend(df);
            Ok(())
        })
    }

    /// Remove the segment `(key, label)`. A nonexistent key or label is a no-op; the document's
    /// other segments are left intact.
    pub fn remove_segment(&mut self, key: impl Into<Key>, label: &str) -> Result<()> {
        let key = key.into();
        self.atomic(|w| {
            let ns = w.index.store.namespace();
            let conn: &Connection = &w.guard;
            let mut changes = TokenChanges::default();
            if let Some(seg_id) = w.index.find_segment(conn, ns, &key, label)? {
                w.index.drop_segment(conn, ns, seg_id, &mut changes)?;
            }
            let df = changes.apply(conn, ns)?;
            w.pending_df.extend(df);
            Ok(())
        })
    }

    /// Commit the open transaction and continue under a fresh one (commit-and-continue), keeping
    /// the lease. Only here do this batch's interned grams and class-stat changes enter the shared
    /// in-memory state (after the durable commit), so a rolled-back tail leaves no orphan id.
    ///
    /// # Errors
    ///
    /// - the **`COMMIT` itself fails**: the batch was **not** made durable (it rolls back); retry
    ///   it on a fresh writer.
    /// - the **`COMMIT` succeeds but the follow-on `BEGIN` fails**
    ///   ([`Error::WriterStranded`]): the batch **is** durable — do not retry it; drop this writer
    ///   and acquire a fresh one.
    pub fn commit(&mut self) -> Result<()> {
        self.guard.execute_batch("COMMIT")?;
        self.txn_open = false;
        // Merge the staged interns first (advancing the shared high-water mark), then fold the df
        // changes into the per-class stats, then snapshot a fresh stage off the advanced mark.
        if let Some(old) = self.stage.take() {
            old.commit();
        }
        let df = std::mem::take(&mut self.pending_df);
        self.index.dict.apply_df_changes(&df);
        self.stage = Some(self.index.dict.stage());
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

impl<T: Tokenizer> Drop for Writer<'_, T> {
    fn drop(&mut self) {
        if self.txn_open {
            let _ = self.guard.execute_batch("ROLLBACK");
        }
    }
}

/// A read lease: the surface for searches. The snapshot is **per search/stream**, not per lease.
pub struct Reader<'a, T: Tokenizer = DefaultTokenizer> {
    index: &'a Index<T>,
}

impl<T: Tokenizer> Reader<'_, T> {
    /// The eager safe default: up to `limit` ranked matches, text + span hydrated, in
    /// weighted-overlap order.
    ///
    /// # Errors
    ///
    /// Returns an error if the store read fails (a transient fault surfaces as
    /// [`Error::Busy`] — retry on a fresh reader).
    pub fn matches(&self, query: &str, opts: &SearchOpts<'_>, limit: usize) -> Result<Vec<Match>> {
        Ok(self
            .matches_batch(&[query], opts, limit)?
            .into_iter()
            .next()
            .unwrap_or_default())
    }

    /// A batch of queries on one shared snapshot, one result list per query in order
    /// (`batch == serial`).
    ///
    /// # Errors
    ///
    /// Returns an error if the store read fails.
    pub fn matches_batch(
        &self,
        queries: &[&str],
        opts: &SearchOpts<'_>,
        limit: usize,
    ) -> Result<Vec<Vec<Match>>> {
        search::matches_batch(self.index, queries, opts, limit)
    }

    /// The lazy streaming spine: a snapshot-pinned, best-first [`CandidateStream`] of
    /// provenance-only [`Candidate`]s. Compose rerank / pagination / fusion on top, then hydrate
    /// only what you keep. Keep the stream short-lived (it pins a WAL snapshot).
    ///
    /// # Errors
    ///
    /// Returns an error if the snapshot cannot be opened.
    pub fn candidates<'s>(
        &'s self,
        query: &str,
        opts: &SearchOpts<'s>,
    ) -> Result<CandidateStream<'s, T>> {
        search::candidates(self.index, query, opts)
    }
}

/// Accumulated per-term-id segment changes for one write batch, applied in a single
/// [`postings::apply_writes`] pass.
#[derive(Default)]
struct TokenChanges {
    map: FxHashMap<TermId, (Vec<u32>, Vec<u32>)>,
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

    /// Apply the accumulated changes, returning the per-term `(id, old_df, new_df)` for the
    /// caller to fold into the per-class statistics.
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

#[cfg(test)]
mod poison_tests {
    use super::*;

    /// A poisoned index (a post-commit dict reload failed) must fail every lease/maintenance
    /// entry point closed, and a later successful `rebuild` must clear it.
    #[test]
    fn poison_fails_leases_closed_and_rebuild_recovers() {
        let dir = tempfile::tempdir().unwrap();
        let idx =
            Index::open_at(&dir.path().join("t.db"), Schema::flat(), Config::default()).unwrap();
        assert!(idx.reader().is_ok(), "healthy index opens a reader");

        idx.poisoned.store(true, Ordering::Release);
        assert!(matches!(idx.reader().err(), Some(Error::Corrupt(_))));
        assert!(matches!(idx.writer().err(), Some(Error::Corrupt(_))));
        assert!(matches!(idx.compact().err(), Some(Error::Corrupt(_))));

        idx.rebuild(std::iter::empty()).unwrap();
        assert!(idx.reader().is_ok(), "rebuild cleared the poison");
        assert!(idx.writer().is_ok());
    }

    /// A held `Reader` must fail its searches closed with `Corrupt` after a mid-lease poison
    /// (the search path re-checks poison on every call).
    #[test]
    fn a_held_lease_search_fails_closed_after_a_mid_lease_poison() {
        let dir = tempfile::tempdir().unwrap();
        let idx =
            Index::open_at(&dir.path().join("t.db"), Schema::flat(), Config::default()).unwrap();
        {
            let mut w = idx.writer().unwrap();
            w.upsert(1, &[("body", "alpha bravo charlie")]).unwrap();
            w.commit().unwrap();
        }
        let reader = idx.reader().unwrap();
        assert!(
            !reader
                .matches("alpha bravo", &SearchOpts::new(), 10)
                .unwrap()
                .is_empty()
        );

        idx.poisoned.store(true, Ordering::Release);

        assert!(matches!(
            reader.matches("alpha bravo", &SearchOpts::new(), 10),
            Err(Error::Corrupt(_))
        ));
    }

    /// A stranded writer fails its methods with [`Error::WriterStranded`] — store intact.
    #[test]
    fn a_stranded_writer_reports_writer_stranded() {
        let dir = tempfile::tempdir().unwrap();
        let idx =
            Index::open_at(&dir.path().join("t.db"), Schema::flat(), Config::default()).unwrap();
        let mut w = idx.writer().unwrap();
        w.guard.execute_batch("COMMIT").unwrap();
        w.txn_open = false;
        assert!(matches!(
            w.upsert(1, &[("body", "x")]),
            Err(Error::WriterStranded(_))
        ));
        assert!(matches!(w.remove(1), Err(Error::WriterStranded(_))));
        drop(w);
        assert!(idx.writer().is_ok());
    }
}
