//! The query pipeline (the spine): resolve → select → engine candidate-gen → provenance/filter
//! → hydrate.
//!
//! [`Index`](crate::Index) exposes the lifecycle and the lease types; the read path lives here.
//! The IDF-weighted bit-sliced overlap counting itself lives in the [`trifle_overlap`] engine
//! ([`Counter`]); this module wires storage to it: it loads the selected tokens' croaring
//! postings, hands them to the engine, then walks the engine's best-first scored ids,
//! batch-hydrating provenance (and applying the opt-in [`SqlFilter`](crate::SqlFilter)) per chunk,
//! deduping one candidate per key.
//!
//! Two front doors share this pipeline:
//! - [`CandidateStream`] — the lazy, snapshot-pinned spine: a best-first cursor of
//!   provenance-only [`Candidate`]s the caller composes rerank / pagination on top of, with a
//!   terminal batched [`hydrate`](CandidateStream::hydrate).
//! - [`matches_batch`] — the eager safe default: top-`limit` [`Match`]es per query, all queries
//!   sharing one snapshot.
//!
//! `batch == serial`: every per-query input (selection, df's, weights, filter) derives only from
//! that query's own tokens and the shared snapshot, so a query in a batch ranks identically to
//! the same query run alone.

use std::borrow::Borrow;
use std::collections::{BTreeSet, VecDeque};
use std::rc::Rc;

use croaring::Bitmap;
use rusqlite::Connection;
use rusqlite::types::Value;
use trifle_overlap::{Counter, Walk};

use crate::dict::TermId;
use crate::filter::SqlFilter;
use crate::hash::{FxHashMap, FxHashSet};
use crate::instrument::trace_debug;
use crate::model::{Key, KeyShape, Match};
use crate::select::{SelectParams, select};
use crate::store::{Namespace, ReadConn};
use crate::term::Term;
use crate::tokenize::Tokenizer;
use crate::{
    DEFAULT_MIN_SHARED, DEFAULT_T_MAX, Error, Index, IntoTerm, Result, SearchOpts, TYPO_DAMAGE,
    postings, schema,
};

/// How many engine candidates to pull per provenance/filter round-trip.
const CHUNK: usize = 64;

/// A scored, provenance-only candidate (no text — see [`CandidateStream::hydrate`]).
///
/// `seg_id` is snapshot-specific (a [`rebuild`](crate::Index::rebuild) reassigns it), so do not
/// carry a `Candidate` across streams/snapshots.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Candidate {
    key: Key,
    label: String,
    seg_id: u32,
    score: u32,
    overlap: u32,
}

impl Candidate {
    /// The caller's document key.
    pub fn key(&self) -> &Key {
        &self.key
    }
    /// The matched segment's label (the text field name).
    pub fn label(&self) -> &str {
        &self.label
    }
    /// The IDF-weighted overlap score — the value trifle ranks by.
    pub fn score(&self) -> u32 {
        self.score
    }
    /// How many selected tokens this candidate shares (the raw, unweighted count).
    pub fn overlap(&self) -> u32 {
        self.overlap
    }
}

/// The distinct tokens per query and the batch-wide distinct **term** set (the resolution
/// input). The read path stays in term-space: it resolves from each token's
/// [`term()`](crate::IntoTerm::term) (no `Token → String → re-encode`). A token wider than the
/// encoding ceiling has no term and rides along as an absent token (df 0).
fn query_terms<Tk: IntoTerm>(
    queries: &[&str],
    tokenize: impl Fn(&str) -> Vec<Tk>,
) -> (Vec<Vec<Tk>>, Vec<Term>) {
    let query_tokens: Vec<Vec<Tk>> = queries.iter().map(|q| tokenize(q)).collect();
    let all_terms: Vec<Term> = query_tokens
        .iter()
        .flat_map(|q| q.iter().filter_map(|t| t.term()))
        .collect::<BTreeSet<Term>>()
        .into_iter()
        .collect();
    (query_tokens, all_terms)
}

/// One query, planned against a snapshot: the engine [`Counter`] plus the present (df > 0)
/// selected tokens (kept for `present_terms`/`matched_terms`) and the full selected-token
/// strings (for span location at hydrate).
struct QueryPlan {
    counter: Counter,
    present_tokens: Vec<String>,
    present_postings: Vec<Bitmap>,
    present_dfs: Vec<u64>,
    selected_strings: Vec<String>,
}

/// Resolve, select (class-aware rarest-first), load postings, and build the engine [`Counter`]
/// for every query — all against the open snapshot `conn` (a tx must already be open). Verifies
/// the dictionary generation against the snapshot (a concurrent id-reassigning rebuild → retryable
/// [`Error::Busy`]). One plan per query, in order; `batch == serial` (selection/df/weights derive
/// only from each query's own tokens + the shared snapshot).
fn prepare<T: Tokenizer>(
    index: &Index<T>,
    conn: &Connection,
    ns: &Namespace,
    queries: &[&str],
    opts: &SearchOpts<'_>,
) -> Result<Vec<QueryPlan>> {
    let (query_tokens, all_terms) = query_terms(queries, |q| index.distinct_tokens(q));

    // Resolve terms in memory + capture the dict generation atomically, then read the snapshot's
    // stored generation (the tx pins the WAL snapshot) to compare. A skew means a concurrent
    // rebuild/reset reassigned term-ids vs this snapshot — surface as retryable Busy (the store
    // is the consistent new generation; the caller retries on a fresh reader). No internal retry.
    let (resolved, gen_mem, class_snap) = index.dict.resolve_terms(&all_terms);
    let gen_snap = schema::dict_generation(conn, ns)?;
    if gen_snap != gen_mem {
        return Err(Error::busy(
            "dictionary generation skew: a concurrent rebuild reassigned term-ids; retry on a \
             fresh reader",
        ));
    }

    let min_shared = opts.min_shared.unwrap_or(DEFAULT_MIN_SHARED).max(1);
    let sel_params = SelectParams {
        min_shared,
        typo_damage: TYPO_DAMAGE,
        t_max: opts.t_max.unwrap_or(DEFAULT_T_MAX),
        df_budget: opts.df_budget,
    };

    // One batched df read over every resolved term-id in the batch.
    let all_ids: Vec<TermId> = resolved
        .values()
        .copied()
        .collect::<BTreeSet<TermId>>()
        .into_iter()
        .collect();
    let dfs = postings::read_dfs(conn, ns, &all_ids)?;
    // A token's (id, df), resolving straight from its packed term — None if it has no term or is
    // absent from the corpus (df 0).
    let resolve = |tok: &T::Token| -> Option<(TermId, i64)> {
        let id = *resolved.get(&tok.term()?.0)?;
        Some((id, dfs.get(&id).copied().unwrap_or(0)))
    };

    // Per-query selection (class-normalized rarest-first; token tie-break). Multi-script
    // awareness lives here, via the per-class stats snapshot.
    let selected_per: Vec<Vec<T::Token>> = query_tokens
        .iter()
        .map(|q| {
            let triples: Vec<(T::Token, i64, u8)> = q
                .iter()
                .map(|tok| {
                    let class = tok.term().map(|t| t.class()).unwrap_or(0);
                    let df = resolve(tok).map_or(0, |(_, df)| df);
                    (tok.clone(), df, class)
                })
                .collect();
            select(&triples, sel_params, &class_snap)
        })
        .collect();

    // One effective-postings read over the union of all queries' selected ids.
    let sel_ids: Vec<TermId> = selected_per
        .iter()
        .flat_map(|s| s.iter())
        .filter_map(|tok| resolve(tok).map(|(id, _)| id))
        .collect::<BTreeSet<TermId>>()
        .into_iter()
        .collect();
    let postings_map = postings::effective_postings(conn, ns, &sel_ids)?;

    let mut plans = Vec::with_capacity(queries.len());
    for selected in &selected_per {
        let mut selected_strings = Vec::with_capacity(selected.len());
        let mut present_tokens = Vec::new();
        let mut present_postings = Vec::new();
        let mut present_dfs = Vec::new();
        for tok in selected {
            let s = tok.borrow().to_string();
            if let Some(bm) = resolve(tok).and_then(|(id, _)| postings_map.get(&id)) {
                present_tokens.push(s.clone());
                present_dfs.push(bm.cardinality());
                present_postings.push(bm.clone());
            }
            selected_strings.push(s);
        }
        // Telemetry for the weight-step hint (the band-spread of this query's present postings).
        index.observe_band_spread(&present_dfs);
        // The `Σ kept-posting cardinality` work-done probe — only evaluated under the `tracing`
        // feature (the macro does not evaluate its args otherwise), so the hot path pays nothing
        // by default. The benchmark profile pass reads this event.
        trace_debug!(
            postings = present_postings.len(),
            sum_cardinality = present_dfs.iter().sum::<u64>(),
            "trifle: weighted overlap candidate generation"
        );
        let counter = Counter::build(&present_postings, opts.weight_step, min_shared);
        plans.push(QueryPlan {
            counter,
            present_tokens,
            present_postings,
            present_dfs,
            selected_strings,
        });
    }
    Ok(plans)
}

/// The fixed provenance context for a search: the snapshot connection, the namespace, the key
/// shape, and the optional filter. Bundled so the per-chunk driver takes a short argument list.
struct Provenance<'c> {
    conn: &'c Connection,
    ns: &'c Namespace,
    key_shape: KeyShape,
    filter: Option<&'c SqlFilter<'c>>,
}

impl Provenance<'_> {
    /// One batched provenance(+filter) query over a chunk's seg ids: `(key, label)` per id that
    /// exists and passes the filter. Fragment textually first, the candidate-scope param last
    /// (`?{N+1}`), so the caller's `?1..?N` (numbered or anonymous) never collide with the scope.
    fn lookup(&self, seg_ids: &[u32]) -> Result<FxHashMap<u32, (Key, String)>> {
        let mut out = FxHashMap::with_capacity_and_hasher(seg_ids.len(), Default::default());
        if seg_ids.is_empty() {
            return Ok(out);
        }
        let arr: Rc<Vec<Value>> =
            Rc::new(seg_ids.iter().map(|&i| Value::Integer(i as i64)).collect());
        let n = self.filter.map_or(0, |f| f.params.len());
        let sql = match self.filter {
            Some(f) => format!(
                "SELECT id, key, label FROM {seg} WHERE ({frag}) AND id IN rarray(?{scope})",
                seg = self.ns.seg(),
                frag = f.fragment,
                scope = n + 1,
            ),
            None => format!(
                "SELECT id, key, label FROM {seg} WHERE id IN rarray(?1)",
                seg = self.ns.seg()
            ),
        };
        let mut binds: Vec<&dyn rusqlite::ToSql> = Vec::with_capacity(n + 1);
        if let Some(f) = self.filter {
            binds.extend_from_slice(f.params); // ?1..?N
        }
        binds.push(&arr); // ?{N+1}

        let mut stmt = self.conn.prepare_cached(&sql)?;
        let mut rows = stmt.query(binds.as_slice())?;
        while let Some(r) = rows.next()? {
            let id: i64 = r.get(0)?;
            let kv: Value = r.get(1)?;
            let label: String = r.get(2)?;
            out.insert(id as u32, (Key::from_value(self.key_shape, kv)?, label));
        }
        Ok(out)
    }
}

/// Pull up to one engine chunk of best-first scored ids, run one provenance(+filter) query over
/// them, dedup by key (first — i.e. highest-score — segment per key wins), and queue the
/// survivors in score order. Returns `true` once the engine walk is exhausted.
fn pull_chunk(
    prov: &Provenance<'_>,
    counter: &Counter,
    walk: &mut Walk,
    seen: &mut FxHashSet<Key>,
    out: &mut VecDeque<Candidate>,
) -> Result<bool> {
    let mut scored = Vec::with_capacity(CHUNK);
    let mut done = false;
    while scored.len() < CHUNK {
        match counter.advance(walk) {
            Some(s) => scored.push(s),
            None => {
                done = true;
                break;
            }
        }
    }
    if scored.is_empty() {
        return Ok(done);
    }
    let seg_ids: Vec<u32> = scored.iter().map(|s| s.id).collect();
    let found = prov.lookup(&seg_ids)?;
    for s in scored {
        if let Some((key, label)) = found.get(&s.id) {
            if seen.insert(key.clone()) {
                out.push_back(Candidate {
                    key: key.clone(),
                    label: label.clone(),
                    seg_id: s.id,
                    score: s.score,
                    overlap: s.overlap,
                });
            }
        }
    }
    Ok(done)
}

/// Hydrate text + span for exactly `kept` in ONE batched `WHERE id IN rarray(?1)` read.
fn hydrate_matches<T: Tokenizer>(
    conn: &Connection,
    ns: &Namespace,
    tokenizer: &T,
    selected: &[String],
    kept: &[Candidate],
) -> Result<Vec<Match>> {
    if kept.is_empty() {
        return Ok(Vec::new());
    }
    let arr: Rc<Vec<Value>> = Rc::new(
        kept.iter()
            .map(|c| Value::Integer(c.seg_id as i64))
            .collect(),
    );
    let sql = format!("SELECT id, txt FROM {} WHERE id IN rarray(?1)", ns.seg());
    let mut txt: FxHashMap<u32, String> =
        FxHashMap::with_capacity_and_hasher(kept.len(), Default::default());
    {
        let mut stmt = conn.prepare_cached(&sql)?;
        let mut rows = stmt.query(rusqlite::params![arr])?;
        while let Some(r) = rows.next()? {
            let id: i64 = r.get(0)?;
            let t: String = r.get(1)?;
            txt.insert(id as u32, t);
        }
    }
    let sel_refs: Vec<&str> = selected.iter().map(String::as_str).collect();
    Ok(kept
        .iter()
        .map(|c| {
            let text = txt.get(&c.seg_id).cloned().unwrap_or_default();
            let span = tokenizer.span(&text, &sel_refs);
            Match {
                key: c.key.clone(),
                label: c.label.clone(),
                span,
                text,
            }
        })
        .collect())
}

/// Eager: top-`limit` matches per query, all queries sharing one snapshot. The safe default
/// front door (`matches`/`matches_batch`). Drains each plan's walk only as deep as the top-`limit`
/// needs, then hydrates exactly those rows.
pub(crate) fn matches_batch<T: Tokenizer>(
    index: &Index<T>,
    queries: &[&str],
    opts: &SearchOpts<'_>,
    limit: usize,
) -> Result<Vec<Vec<Match>>> {
    index.check_poisoned()?;
    if queries.is_empty() {
        return Ok(Vec::new());
    }
    let ns = index.store.namespace();
    let conn = index.store.read()?;
    // One pinned snapshot for the whole batch (RAII rollback on drop).
    let tx = conn.unchecked_transaction()?;
    let plans = prepare(index, &tx, ns, queries, opts)?;
    let prov = Provenance {
        conn: &tx,
        ns,
        key_shape: index.schema.key_shape(),
        filter: opts.filter.as_ref(),
    };

    let mut out = Vec::with_capacity(queries.len());
    for plan in &plans {
        let mut walk = plan.counter.walk();
        let mut seen: FxHashSet<Key> = FxHashSet::default();
        let mut ready: VecDeque<Candidate> = VecDeque::new();
        let mut kept: Vec<Candidate> = Vec::with_capacity(limit);
        let mut done = false;
        while kept.len() < limit {
            if let Some(c) = ready.pop_front() {
                kept.push(c);
                continue;
            }
            if done {
                break;
            }
            done = pull_chunk(&prov, &plan.counter, &mut walk, &mut seen, &mut ready)?;
        }
        out.push(hydrate_matches(
            &tx,
            ns,
            &index.tokenizer,
            &plan.selected_strings,
            &kept,
        )?);
    }
    Ok(out)
}

/// Open the lazy candidate stream for `query`. The stream owns a pooled connection with a pinned
/// read transaction (manual `BEGIN`/`ROLLBACK`, never a stored `Transaction` — so it has no
/// self-referential lifetime) and the engine [`Counter`].
pub(crate) fn candidates<'a, T: Tokenizer>(
    index: &'a Index<T>,
    query: &str,
    opts: &SearchOpts<'a>,
) -> Result<CandidateStream<'a, T>> {
    index.check_poisoned()?;
    let ns = index.store.namespace();
    let conn = index.store.read()?;
    conn.execute_batch("BEGIN DEFERRED")?; // pin a snapshot for the stream's life
    // prepare may fail (Busy on generation skew); release the snapshot if so.
    let plan = match prepare(index, &conn, ns, &[query], opts) {
        Ok(mut plans) => plans.pop().expect("one plan for one query"),
        Err(e) => {
            let _ = conn.execute_batch("ROLLBACK");
            return Err(e);
        }
    };
    // n_segments / avgdl from this snapshot's rolling counters (a corpus-relative custom score
    // must not cross a snapshot boundary).
    let (seg_count, seg_len_sum) = schema::read_seg_stats(&conn, ns)?;
    let n_segments = seg_count.max(0) as u64;
    let avgdl = if seg_count > 0 {
        seg_len_sum as f64 / seg_count as f64
    } else {
        0.0
    };
    let walk = plan.counter.walk();
    Ok(CandidateStream {
        index,
        conn,
        plan,
        walk,
        filter: opts.filter,
        ready: VecDeque::new(),
        seen: FxHashSet::default(),
        n_segments,
        avgdl,
        done: false,
        errored: false,
    })
}

/// A lazy, snapshot-pinned, best-first candidate cursor (the architectural spine). Owns a pooled
/// connection with a pinned read transaction **and** the engine [`Counter`]; drives the
/// bit-sliced walk, batch-hydrates provenance (+ applies the [`SqlFilter`](crate::SqlFilter)) per
/// chunk, dedups to one candidate per key, best-first. **Fuses on the first error** (a caller
/// never gets a deceptively-complete prefix after a transient `Busy`).
///
/// A live stream pins its WAL snapshot — keep it short-lived; do not park it. Drop releases the
/// snapshot.
pub struct CandidateStream<'a, T: Tokenizer> {
    index: &'a Index<T>,
    conn: ReadConn<'a>,
    plan: QueryPlan,
    walk: Walk,
    filter: Option<SqlFilter<'a>>,
    ready: VecDeque<Candidate>,
    seen: FxHashSet<Key>,
    n_segments: u64,
    avgdl: f64,
    done: bool,
    errored: bool,
}

impl<T: Tokenizer> CandidateStream<'_, T> {
    /// Total live segments `N`, from **this search's** snapshot (not `stats()`).
    pub fn n_segments(&self) -> u64 {
        self.n_segments
    }
    /// Mean segment gram length (`avgdl`) on this snapshot. `0.0` on an empty corpus.
    pub fn avgdl(&self) -> f64 {
        self.avgdl
    }
    /// The selected tokens that have a posting, each with its document frequency `df` (no SQL —
    /// the postings are already in hand).
    pub fn present_terms(&self) -> impl Iterator<Item = (&str, u64)> {
        self.plan
            .present_tokens
            .iter()
            .zip(&self.plan.present_dfs)
            .map(|(t, df)| (t.as_str(), *df))
    }
    /// Which selected tokens this candidate's segment actually contains, each with its `df` (no
    /// SQL). The inputs an IDF-sum-style custom reranker needs.
    pub fn matched_terms<'c>(&'c self, c: &Candidate) -> impl Iterator<Item = (&'c str, u64)> + 'c {
        let seg_id = c.seg_id;
        self.plan
            .present_tokens
            .iter()
            .zip(&self.plan.present_postings)
            .zip(&self.plan.present_dfs)
            .filter(move |((_, bm), _)| bm.contains(seg_id))
            .map(|((t, _), df)| (t.as_str(), *df))
    }

    /// Hydrate text + span for exactly `kept` in ONE batched read (the terminal step). A
    /// pull-many/keep-few caller hydrates only what it kept. Pass candidates from **this** stream
    /// (seg ids are snapshot-specific).
    pub fn hydrate(&self, kept: &[Candidate]) -> Result<Vec<Match>> {
        hydrate_matches(
            &self.conn,
            self.index.store.namespace(),
            &self.index.tokenizer,
            &self.plan.selected_strings,
            kept,
        )
    }

    /// Error-propagating collector: take up to `limit` candidates and hydrate them (no silent
    /// truncation — a mid-stream `Err` propagates).
    pub fn collect_matches(mut self, limit: usize) -> Result<Vec<Match>> {
        let mut kept = Vec::with_capacity(limit);
        while kept.len() < limit {
            match self.next() {
                Some(Ok(c)) => kept.push(c),
                Some(Err(e)) => return Err(e),
                None => break,
            }
        }
        self.hydrate(&kept)
    }
}

impl<T: Tokenizer> Iterator for CandidateStream<'_, T> {
    type Item = Result<Candidate>;
    /// Best-first, deduped-per-key, filtered. Fuses on the first `Err`.
    fn next(&mut self) -> Option<Result<Candidate>> {
        loop {
            if let Some(c) = self.ready.pop_front() {
                return Some(Ok(c));
            }
            if self.done || self.errored {
                return None;
            }
            let prov = Provenance {
                conn: &self.conn,
                ns: self.index.store.namespace(),
                key_shape: self.index.schema.key_shape(),
                filter: self.filter.as_ref(),
            };
            match pull_chunk(
                &prov,
                &self.plan.counter,
                &mut self.walk,
                &mut self.seen,
                &mut self.ready,
            ) {
                Ok(done) => self.done = done,
                Err(e) => {
                    self.errored = true;
                    return Some(Err(e));
                }
            }
        }
    }
}

impl<T: Tokenizer> Drop for CandidateStream<'_, T> {
    fn drop(&mut self) {
        // Release the pinned snapshot. Best-effort; the pool also rolls back any open tx on
        // check-in, so a missed ROLLBACK here cannot leak a snapshot to the next checkout.
        let _ = self.conn.execute_batch("ROLLBACK");
    }
}
