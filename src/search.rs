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

use std::borrow::Borrow;
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
    DEFAULT_MIN_SHARED, DEFAULT_T_MAX, Error, Index, IntoTerm, Match, RETRY_MAX, Result,
    SearchOpts, TYPO_DAMAGE, Term, is_retryable, postings, schema,
};

/// The distinct tokens per query and the batch-wide distinct **term** set (the resolution
/// input). Factored so the [`Reader`](crate::Reader) and the warm
/// [`SearchSession`](crate::SearchSession) share one tokenize pass shape. The read path stays
/// in term-space: it resolves from each token's [`term()`](crate::IntoTerm::term) (no
/// `Token → String → re-encode`), so the per-query token vectors carry the tokens themselves
/// and only the selected ones are later stringified (audit T2 / I10). A token wider than the
/// encoding ceiling has no term and is dropped from the resolution set — it resolves to df 0
/// and rides along as an absent token.
pub(crate) fn query_terms<Tk: IntoTerm>(
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

/// One search's context: the index plus the single consistent snapshot the search runs
/// against (connection, namespace, resolved gram→id map, per-class stats snapshot) and the
/// [`SearchOpts`]. Bundling these is what lets each pipeline stage take `&self` instead of a
/// long, repeated argument list.
pub(crate) struct SearchCtx<'a, T: Tokenizer, B: Backend> {
    index: &'a Index<T, B>,
    conn: &'a Connection,
    ns: &'a Namespace,
    /// The query's distinct terms resolved to ids, keyed by the packed term `u128`.
    resolved: &'a HashMap<u128, TermId>,
    class_snap: &'a ClassSnap,
    opts: &'a SearchOpts<'a>,
}

impl<'a, T: Tokenizer, B: Backend> SearchCtx<'a, T, B> {
    pub(crate) fn new(
        index: &'a Index<T, B>,
        conn: &'a Connection,
        ns: &'a Namespace,
        resolved: &'a HashMap<u128, TermId>,
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
        query_tokens: &[Vec<T::Token>],
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
        // A token's `(id, df)`, resolving straight from its packed term — `None` if it has no
        // term (over the encoding ceiling) or its term is absent from the corpus (→ df 0).
        let resolve = |tok: &T::Token| -> Option<(TermId, i64)> {
            let id = *self.resolved.get(&tok.term()?.0)?;
            Some((id, dfs.get(&id).copied().unwrap_or(0)))
        };

        // Selection runs in term-space; its tie-break is the token's own `Ord` (the
        // `Tokenizer::Token` contract), identical to the previous string tie-break for the
        // built-in tokenizers (their `Ord` delegates to `str`).
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
                select(&triples, sel_params, self.class_snap)
            })
            .collect();
        let sel_ids: Vec<TermId> = selected_per
            .iter()
            .flat_map(|s| s.iter())
            .filter_map(|tok| resolve(tok).map(|(id, _)| id))
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
            // The selected tokens that have a posting, paired with it via the token's `&str`
            // view (no allocation — `Borrow<str>`).
            let present: Vec<(&str, &RoaringBitmap)> = selected
                .iter()
                .filter_map(|tok| {
                    resolve(tok)
                        .and_then(|(id, _)| postings_map.get(&id))
                        .map(|bm| (tok.borrow(), bm))
                })
                .collect();
            // Telemetry for the weight-step hint (df-only; no corpus read).
            self.index.observe_band_spread(&present);

            let mut survivors = overlap_search(self.conn, self.ns, &present, &overlap_params)?;
            // Every indexed field is stored, so a match always carries its text and any
            // custom ranker always sees it.
            self.hydrate_text(&mut survivors)?;

            // Stringify only the selected tokens — the public ranker API
            // ([`QueryContext::selected`]) and `span` take strings; nothing else on the read
            // path allocates per token (audit T2 / I10).
            let selected_strings: Vec<String> = selected
                .iter()
                .map(|tok| tok.borrow().to_string())
                .collect();
            let qctx = QueryContext {
                query,
                selected: &selected_strings,
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
    all_terms: &[Term],
    mut f: impl FnMut(&Connection, &Namespace, &HashMap<u128, TermId>, &ClassSnap) -> Result<R>,
) -> Result<R> {
    let ns = index.backend.namespace();
    let mut attempt = 0;
    loop {
        let tx = match conn.unchecked_transaction() {
            Ok(tx) => tx,
            Err(e) => return Err(Error::from(e)),
        };
        // Resolve the terms in memory + capture the generation and per-class stats
        // snapshot atomically, then read the snapshot's generation to compare.
        let (resolved, gen_mem, class_snap) = index.dict.resolve_terms(all_terms);
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
