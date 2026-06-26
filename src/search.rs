//! The query pipeline (the spine): resolve → select → candidate-gen + rank → hydrate.
//!
//! [`Index`](crate::Index) exposes the lifecycle and the lease types; the read path lives
//! here. A search runs against one consistent WAL snapshot, which — together with the index
//! and the [`SearchOpts`] — is bundled into a [`SearchCtx`] so the pipeline stages read their
//! inputs from the context instead of threading a long argument list. The
//! retry/snapshot/generation-guard loop ([`search_read_on`]) wraps the body so a concurrent
//! id-reassigning [`rebuild`](crate::Index::rebuild) is observed atomically (old-or-new, never
//! spliced).
//!
//! `batch == serial`: every per-query input (selection, df's, weights, filter) derives only
//! from that query's own tokens and the shared snapshot, so a query in a batch ranks
//! identically to the same query run alone.

use std::collections::{BTreeSet, HashMap};
use std::time::Duration;

use roaring::RoaringBitmap;
use rusqlite::Connection;
use rusqlite::types::Value;

use crate::dict::TermId;
use crate::rank::{
    Candidates, CompiledFilter, OverlapParams, OverlapRanker, QueryContext, Ranker, Survivor,
    overlap_search,
};
use crate::select::{SelectParams, select};
use crate::store::{Backend, Namespace};
use crate::tokenize::Tokenizer;
use crate::welford::ClassSnap;
use crate::{
    DEFAULT_MIN_SHARED, DEFAULT_T_MAX, Error, Index, Match, RETRY_MAX, Result, SearchOpts,
    TYPO_DAMAGE, is_retryable, postings, schema, term,
};

/// The distinct tokens per query and the batch-wide distinct gram set (the resolution input).
/// Factored so the [`Reader`](crate::Reader) and the warm [`SearchSession`](crate::SearchSession)
/// share one tokenize pass shape.
pub(crate) fn query_grams(
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

/// One search's context: the index plus the single consistent snapshot the search runs
/// against (connection, namespace, resolved gram→id map, per-class stats snapshot) and the
/// [`SearchOpts`]. Bundling these is what lets each pipeline stage take `&self` instead of a
/// long, repeated argument list.
pub(crate) struct SearchCtx<'a, T: Tokenizer, B: Backend> {
    index: &'a Index<T, B>,
    conn: &'a Connection,
    ns: &'a Namespace,
    resolved: &'a HashMap<String, TermId>,
    class_snap: &'a ClassSnap,
    opts: &'a SearchOpts<'a>,
}

impl<'a, T: Tokenizer, B: Backend> SearchCtx<'a, T, B> {
    pub(crate) fn new(
        index: &'a Index<T, B>,
        conn: &'a Connection,
        ns: &'a Namespace,
        resolved: &'a HashMap<String, TermId>,
        class_snap: &'a ClassSnap,
        opts: &'a SearchOpts<'a>,
    ) -> Self {
        SearchCtx {
            index,
            conn,
            ns,
            resolved,
            class_snap,
            opts,
        }
    }

    /// The per-snapshot search body: read df's/postings, select, generate candidates, apply
    /// the filter, rank, hydrate. One result list per query, in order.
    pub(crate) fn run_search(
        &self,
        queries: &[&str],
        query_tokens: &[Vec<String>],
    ) -> Result<Vec<Vec<Match>>> {
        let opts = self.opts;
        let min_shared = opts.min_shared.unwrap_or(DEFAULT_MIN_SHARED).max(1);
        let sel_params = SelectParams {
            min_shared,
            typo_damage: TYPO_DAMAGE,
            t_max: opts.t_max.unwrap_or(DEFAULT_T_MAX),
        };
        // Ranking is the IDF-weighted overlap order the counter produces; the default
        // [`OverlapRanker`] preserves it. An explicit `opts.ranker` may reorder (over an
        // [`Effort`](crate::Effort)-deepened pool). There is no built-in relevance/BM25 tier.
        let overlap = OverlapRanker;
        let ranker: &dyn Ranker = opts.ranker.unwrap_or(&overlap);

        // One batched frequency read over every resolved term-id in the batch.
        let all_ids: Vec<TermId> = self
            .resolved
            .values()
            .copied()
            .collect::<BTreeSet<TermId>>()
            .into_iter()
            .collect();
        let dfs = postings::read_dfs(self.conn, self.ns, &all_ids)?;
        // A gram's df: 0 if it resolved to no id (absent token) or its id has no live df
        // row — exactly the existing df-0 behavior.
        let df_of = |t: &str| -> i64 {
            self.resolved
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
                select(&triples, sel_params, self.class_snap)
            })
            .collect();
        let sel_ids: Vec<TermId> = selected_per
            .iter()
            .flat_map(|s| s.iter())
            .filter_map(|t| self.resolved.get(t.as_str()).copied())
            .collect::<BTreeSet<TermId>>()
            .into_iter()
            .collect();
        let postings_map = postings::effective_postings(self.conn, self.ns, &sel_ids)?;

        // Corpus size + average segment length from the O(1) rolling meta counters (audit
        // I3), read under this search's snapshot. Not used by the default overlap ranking
        // (it is df-only); surfaced to a custom [`Ranker`] via [`QueryContext`].
        let (seg_count, seg_len_sum) = schema::read_seg_stats(self.conn, self.ns)?;
        let n_segments = seg_count.max(0) as u64;
        let avgdl = if seg_count > 0 {
            seg_len_sum as f64 / seg_count as f64
        } else {
            0.0
        };
        let pool = opts.effort.pool(opts.limit, n_segments);

        // Tier-2 filter, compiled once for the whole batch. It is applied **scoped to each
        // query's candidate ids** inside `overlap_search` (a small `WHERE id IN rarray(...)` per
        // bucket), so its cost is bounded by the pool, never an O(N) scan of the corpus (audit
        // T5 / I12). Compiling here rather than per query keeps batch == serial — every query in
        // the batch sees the identical predicate.
        let compiled_filter = match opts.filter {
            Some(f) => Some(f.compile(&self.index.schema)?),
            None => None,
        };
        let filter = compiled_filter
            .as_ref()
            .map(|(where_sql, params)| CompiledFilter { where_sql, params });

        // The candidate-generation knobs are all batch-constant — build once, reuse per query.
        let overlap_params = OverlapParams {
            limit: pool,
            min_shared,
            weight_step: opts.weight_step,
            key_shape: self.index.schema.key_shape(),
            filter: filter.as_ref(),
            scope: opts.scope,
        };

        let mut out = Vec::with_capacity(queries.len());
        for (qi, &query) in queries.iter().enumerate() {
            let selected = &selected_per[qi];
            let present: Vec<(&str, &RoaringBitmap)> = selected
                .iter()
                .filter_map(|t| {
                    self.resolved
                        .get(t.as_str())
                        .and_then(|id| postings_map.get(id))
                        .map(|bm| (t.as_str(), bm))
                })
                .collect();
            // Telemetry for the weight-step hint (df-only; no corpus read).
            self.index.observe_band_spread(&present);

            let mut survivors = overlap_search(self.conn, self.ns, &present, &overlap_params)?;
            // Every indexed field is stored, so a match always carries its text and any
            // custom ranker always sees it.
            self.hydrate_text(&mut survivors)?;

            let qctx = QueryContext {
                query,
                selected,
                min_shared,
                n_segments,
                avgdl,
            };
            out.push(self.rank_to_matches(&survivors, &present, &qctx, ranker));
        }
        Ok(out)
    }

    /// Run the ranker over the survivors and build the result matches (limit from
    /// [`SearchOpts`]).
    fn rank_to_matches(
        &self,
        survivors: &[Survivor],
        present: &[(&str, &RoaringBitmap)],
        qctx: &QueryContext<'_>,
        ranker: &dyn Ranker,
    ) -> Vec<Match> {
        let limit = self.opts.limit;
        let candidates = Candidates::new(survivors, present);
        let ranked = ranker.rank(&candidates, qctx);
        let sel_refs: Vec<&str> = qctx.selected.iter().map(String::as_str).collect();
        let mut matches = Vec::with_capacity(ranked.len().min(limit));
        for r in ranked.into_iter().take(limit) {
            let Some(s) = survivors.get(r.candidate) else {
                continue;
            };
            let span = self.index.tokenizer.span(&s.text, &sel_refs);
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
    fn hydrate_text(&self, survivors: &mut [Survivor]) -> Result<()> {
        if survivors.is_empty() {
            return Ok(());
        }
        let ids: Vec<u32> = survivors.iter().map(|s| s.seg_id).collect();
        let arr: std::rc::Rc<Vec<Value>> =
            std::rc::Rc::new(ids.iter().map(|&i| Value::Integer(i as i64)).collect());
        let sql = format!(
            "SELECT id, txt, len FROM {} WHERE id IN rarray(?1)",
            self.ns.seg()
        );
        let mut texts: HashMap<u32, (String, u32)> = HashMap::with_capacity(ids.len());
        {
            let mut stmt = self.conn.prepare_cached(&sql)?;
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
}

/// The retry/snapshot/generation-guard loop on a **given** connection — used by the
/// [`Reader`](crate::Reader) (a fresh pooled checkout per call) and by the warm
/// [`SearchSession`](crate::SearchSession) (a held connection reused across keystrokes).
/// Retries open a fresh snapshot on the same connection.
///
/// The whole read runs inside one DEFERRED transaction so every statement (token df's,
/// postings, segment count, hydration) sees a single snapshot; without it they could straddle
/// a concurrent id-reassigning `rebuild` commit and splice postings from the old snapshot onto
/// seg rows from the new one. The transaction is read-only and never committed; dropping it
/// just releases the snapshot.
///
/// Because the term dictionary is in memory (out of the SQL snapshot), the grams are resolved
/// and the dictionary generation captured atomically, then compared to the snapshot's stored
/// `dict_generation`. A mismatch means a rebuild/reset reassigned ids relative to this
/// snapshot — the read retries on a fresh snapshot, where the generations agree. Transient
/// busy/locked/schema-change faults are retried too.
pub(crate) fn search_read_on<T: Tokenizer, B: Backend, R>(
    index: &Index<T, B>,
    conn: &Connection,
    all_grams: &[String],
    mut f: impl FnMut(&Connection, &Namespace, &HashMap<String, TermId>, &ClassSnap) -> Result<R>,
) -> Result<R> {
    let ns = index.backend.namespace();
    let gram_refs: Vec<&str> = all_grams.iter().map(String::as_str).collect();
    let mut attempt = 0;
    loop {
        let tx = match conn.unchecked_transaction() {
            Ok(tx) => tx,
            Err(e) => return Err(Error::from(e)),
        };
        // Resolve the grams in memory + capture the generation and per-class stats
        // snapshot atomically, then read the snapshot's generation to compare.
        let (resolved, gen_mem, class_snap) = index.dict.resolve_batch(&gram_refs);
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
