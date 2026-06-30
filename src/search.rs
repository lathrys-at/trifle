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
use trifle_overlap::{Counter, Scored, Walk};

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
    DEFAULT_DELTA, DEFAULT_KAPPA, DEFAULT_MIN_SHARED, DEFAULT_NU, DEFAULT_T_MAX, Error, Index,
    IntoTerm, Result, SearchOpts, TYPO_DAMAGE, postings, schema,
};

/// How many engine candidates to pull per provenance/filter round-trip.
const CHUNK: usize = 64;

/// The realized floored-gram energy `E_floored` (nats) below which the §7 quantization guard
/// `Δ < 2·E_floored` is treated as inapplicable rather than violated. On a handful-of-segments
/// corpus `E_floored` shrinks below this and the guard cannot hold; that regime is recall-safe via
/// the engine's `≥ 1` weight clamp, so the debug-time guard is skipped there to avoid tripping on
/// legitimately tiny corpora. The guard still fires for a genuinely too-coarse `Δ` once the corpus
/// is large enough for `E_floored` to clear this threshold. See [`prepare`].
const GUARD_MIN_E_FLOORED: f64 = 0.5;

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
/// strings (for span location at hydrate). `n_segments` (`N`) and `avgdl` (`L̄`) are the
/// snapshot-wide corpus stats the N-anchored v0.4 scoring (energy/floor/stop/null) needs; read
/// once per batch so they are identical for every query (the shared snapshot), preserving
/// `batch == serial`.
struct QueryPlan {
    counter: Counter,
    present_tokens: Vec<String>,
    present_postings: Vec<Bitmap>,
    present_dfs: Vec<u64>,
    /// Per present gram (parallel to `present_postings`): whether it is **floored** — query-side
    /// `df ≤ df_min` (derivation §4), i.e. a substitution-artifact suspect. A floored gram carries
    /// `E_max` energy on the bit-sliced side but earns **no** count credit (§9); that withholding,
    /// not the floor, is what restores junk-below-real ordering. (M4 also reads this to exclude
    /// floored grams from the Cantelli stop; M2 uses it only to gate the credit.)
    present_floored: Vec<bool>,
    /// Per present gram (parallel to `present_postings`): its order `n` (gram codepoint count — 2
    /// for a CJK bigram, else 3, fewer for a short structural-fallback gram). Keys the per-order
    /// count credit (§7/§12 `Σ_n μ_n·popcount_n`). Query-side `σ` is uniform so every order maps to
    /// the same `μ` in M2; the bucketing is kept structural so M5's per-order doc-side
    /// `ρ = σ(1−ε)^n` drops in by repopulating [`mu_by_order`](QueryPlan::mu_by_order).
    present_orders: Vec<u8>,
    /// The §9-capped count credit `μ_n = min(max(0, logit σ), cap)`, keyed by the gram order `n`
    /// present among the **non-floored** grams. Looked up by `present_orders` when scoring credit.
    mu_by_order: FxHashMap<u8, f64>,
    /// The energy quantization step `Δ` (nats) this plan was built with — reconverts the integer
    /// bit-sliced energy back to nats for the float ranking key (derivation §7).
    delta: f64,
    selected_strings: Vec<String>,
    n_segments: u64,
    avgdl: f64,
}

impl QueryPlan {
    /// The derivation §3/§9 count credit for candidate `id`: `μ` times the popcount of matched
    /// **non-floored** grams, bucketed by gram order (§7/§12 `Σ_n μ_n·popcount_n`). Floored
    /// (junk-suspect) grams earn nothing (§9) — the policy that restores the junk-below-real
    /// ordering the floor's `E_max` energy alone would invert. `O(k)` per candidate (`k ≤ t_max`),
    /// so the post-pass stays bounded (flatness).
    fn count_credit(&self, id: u32) -> f64 {
        popcount_credit(
            id,
            &self.present_postings,
            &self.present_floored,
            &self.present_orders,
            &self.mu_by_order,
        )
    }

    /// The M2 float ranking key: the integer bit-sliced energy reconverted to nats plus the count
    /// credit (`E_acc·Δ + Σ_n μ_n·popcount_n`, derivation §7/§12). The §6 length null is M3, so it
    /// is not subtracted here. Top-k truncation must use this key, never the bare integer
    /// [`Scored::score`].
    fn float_score(&self, scored: &Scored) -> f64 {
        scored.score as f64 * self.delta + self.count_credit(scored.id)
    }
}

// --- v0.4 logit-idf energy weighting (derivation §2, §4, §7) ------------------------------------
//
// Replaces v0.3's `N`-free 4-tier df-rarity weight with the Robertson–Spärck-Jones log-odds
// (logit-idf) of each gram's Jeffreys-smoothed, contamination-floored segment-frequency, quantized
// to a small non-negative integer for the bit-sliced counter. All quantities are `N`-anchored, so
// they are computed here (the [`trifle_overlap`] engine is `N`-free) and fed to the engine as
// explicit per-posting weights. M1 is the **query-side** channel only — the contamination floor
// `df_min` always applies; the doc-side channel (`ε > 0`, no floor) and the count credit `μ` are
// later milestones, so the M1 score stays an integer bit-sliced energy sum.

/// Contamination floor `df_min = N^((ν−1)/ν)` (derivation §4): the query-side segment-frequency
/// below which a gram is treated as a possible substitution artifact, capping its energy at
/// `E_max`. A degenerate `ν ≤ 0` yields no floor (`df_min = 0`, so `df_eff = df`).
fn df_min(n: f64, nu: f64) -> f64 {
    if nu <= 0.0 {
        return 0.0;
    }
    n.powf((nu - 1.0) / nu)
}

/// Single-gram energy ceiling `E_max = (1/ν)·ln N` (derivation §4): no single gram can identify a
/// segment alone, so at least `ν` matched grams must agree. Bounds a *single* gram's quantized
/// weight bit-width by `⌈log2(E_max/Δ + 1)⌉` (§7) — not the accumulator's plane count, which the
/// engine sizes to `bits(Σ wq)`. Also the upper bound on a floored gram's realized energy
/// (`E_floored ≤ E_max`). `0` for `N ≤ 1` (no discrimination possible).
fn e_max(n: f64, nu: f64) -> f64 {
    if n <= 1.0 || nu <= 0.0 {
        return 0.0;
    }
    n.ln() / nu
}

/// Per-gram query-side energy with a **precomputed** contamination floor — the hot-path form, so a
/// per-query loop never recomputes the `df_min` `powf` (it is `N`/`ν`-constant across the batch).
/// `E_g = ln((N − df_eff − κ)/(df_eff + κ))`, the RSJ logit-idf of the Jeffreys-smoothed, floored
/// estimate (derivation §2/§4), with `df_eff = max(df, df_min)`. Negative for `p_eff > 0.5`
/// (clamped to `0` at quantization); returns `−∞` for a gram present in (nearly) every segment,
/// which [`quantize_energy`] maps to weight `0`.
fn energy_with_floor(df: f64, df_min: f64, n: f64, kappa: f64) -> f64 {
    let df_eff = df.max(df_min);
    let num = n - df_eff - kappa;
    let den = df_eff + kappa;
    if num <= 0.0 || den <= 0.0 {
        return f64::NEG_INFINITY; // p_eff ≈ 1; logit → −∞, clamped to weight 0 at use
    }
    (num / den).ln()
}

/// Per-gram query-side energy (derivation §2/§4) — the convenience form that derives the floor
/// `df_min(N, ν)` itself; see [`energy_with_floor`] for the hoisted hot-path variant.
fn energy(df: f64, n: f64, nu: f64, kappa: f64) -> f64 {
    energy_with_floor(df, df_min(n, nu), n, kappa)
}

/// Realized floored-gram energy `E_floored = ln((N − df_min − κ)/(df_min + κ)) ≤ E_max` — the
/// energy every floored (`df ≤ df_min`) gram carries (derivation §4/§7). The §7 quantization guard
/// `Δ < 2·E_floored` keeps `round(E_floored/Δ) ≥ 1`, so a floored gram never quantizes to `0`. For
/// a tiny corpus this is small or negative — see [`prepare`]'s guard.
fn e_floored(n: f64, nu: f64, kappa: f64) -> f64 {
    // Routes through `energy` (computing `df_min` for both the `df` argument and the floor) so the
    // convenience wrapper stays exercised; called once per batch, so the extra `powf` is free.
    energy(df_min(n, nu), n, nu, kappa)
}

/// Quantize an energy to the bit-sliced weight `wq = max(0, round(E/Δ))` (derivation §7). A
/// non-finite or non-positive energy maps to `0`; the engine then clamps every weight to `≥ 1`
/// (`trifle_overlap`'s `plan`), so a `0`-energy gram still contributes count-only rather than
/// vanishing. `delta` is assumed `> 0` (resolved positive in [`prepare`]).
fn quantize_energy(e: f64, delta: f64) -> u32 {
    let q = (e / delta).round();
    if q.is_finite() && q > 0.0 {
        q as u32
    } else {
        0
    }
}

// --- v0.4 count credit μ + the §9 concentration cap (derivation §3, §7, §9) --------------------
//
// The absent-real-gram evidence regroups into a flat per-match bonus μ on every matched, non-
// floored gram (§3): each contributes the full RSJ weight `E_g + μ`. μ is the policy that orders a
// real (non-floored) match above a junk (floored, `E_max`) one (§9), since the floor alone does not
// (a floored gram sits at the energy ceiling). All of this is query-side and N-anchored, so it
// lives here (the engine is N-free) as a post-pass over the retained postings, NOT a hot-loop add.

/// The query-side count credit `μ = max(0, logit σ)` (derivation §3/§9) — nats per matched,
/// non-floored gram. `σ` is sanitized to `(0,1)` at [`Index::open`](crate::Index::open), so
/// `logit σ = ln(σ/(1−σ))` is finite here; an unreliable `σ ≤ ½` yields `μ = 0` (the `max(0,·)`
/// clamp), a recall-safe no-op (a recall stage never penalizes matches).
fn count_credit(sigma: f64) -> f64 {
    (sigma / (1.0 - sigma)).ln().max(0.0)
}

/// The §9 **concentration cap** on the count credit. Returns `Some(cap)` when the pruned set's
/// energies are *concentrated* — a single dominant rare gram (positive top energy `E_top`) amid
/// **≥ 2** query-relative commons (a gram with `E < ½·E_top`) — else `None` (μ uncapped). An
/// **all-common** query is deliberately left **uncapped** (it degrades to count-and-length ranking
/// rather than having its credit zeroed, §7/§9) — by *either* guard, which together cover both
/// all-common sub-cases: a *ubiquitous* all-common set (every `p ≥ ½` ⇒ every `E ≤ 0`) trips the
/// `E_top ≤ 0` branch, while a *mid-rarity* all-common set (every `p < ½` ⇒ comparable positive
/// energies, none below `½·E_top`) trips the `commons.len() < 2` guard. Both are load-bearing:
/// without them an all-common query would spuriously cap (to 0 in the ubiquitous case).
///
/// `cap = max(0, (E_top − Σ_common max(0,E)) / (#common − 1))`. The hard floor at 0 (reached when
/// the commons collectively outweigh the dominant gram) is the M2 baseline; §9's smoother shrink
/// toward the cap is a deferred tuning refinement.
///
/// **Interpretation note (auditable):** floored grams are **not** excluded from `E_top`. This is
/// the literal §12 reading — `concentrated(P)`/`concentration_cap(P)` range over all of `P` — and
/// matches §9's framing of the cap as a *query-structure* property: a dominant gram (junk-suspect
/// or not) should not be out-credited by commons. The floored *exclusion* governs only which grams
/// earn credit (above) and the M4 stop, not the cap's `E_top`.
///
/// **Consequence flagged for the design owner (behavior NOT changed in M2).** When a *floored* gram
/// is the dominant `E_top` (it sits at `E_max`) and a *real* mid-rare gram is co-present below it,
/// that high floored `E_top` *loosens* the cap, so the cap no longer protects the real
/// discriminating gram from commons-credit — a floored-*excluded* `E_top` would instead clamp
/// tighter and protect it (R1's numeric: off-topic commons doc `8.09 >` on-topic rare doc `6.20`;
/// floored-excluded would tie at `5.25`). This is the literal §12 reading, KEPT for M2 and
/// recall-safe (§9: "a precision distortion the reranker undoes"). Whether the cap should key off
/// only the *real* (non-floored) discriminating grams is a deferred §9/§12 **derivation-text**
/// question for the design owner; M2 does not change the behavior either way.
fn concentration_cap(energies: &[f64]) -> Option<f64> {
    let e_top = energies.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if e_top <= 0.0 {
        return None; // no dominant rare gram (all-common / empty) — μ survives (§7/§9)
    }
    let half = 0.5 * e_top;
    let commons: Vec<f64> = energies.iter().copied().filter(|&e| e < half).collect();
    if commons.len() < 2 {
        return None;
    }
    let sum_commons: f64 = commons.iter().map(|&e| e.max(0.0)).sum();
    let cap = (e_top - sum_commons) / (commons.len() as f64 - 1.0);
    Some(cap.max(0.0))
}

/// `μ`-weighted popcount of `id`'s matched **non-floored** grams, bucketed by gram order
/// (derivation §7/§12 `Σ_n μ_n·popcount_n`). A floored gram is skipped (no credit, §9); a
/// non-floored gram contributes its order's capped `μ_n`. Free function so it is unit-testable
/// without a full [`QueryPlan`]. `O(k)` over the present grams (`k ≤ t_max`).
fn popcount_credit(
    id: u32,
    postings: &[Bitmap],
    floored: &[bool],
    orders: &[u8],
    mu_by_order: &FxHashMap<u8, f64>,
) -> f64 {
    postings
        .iter()
        .enumerate()
        .filter(|&(i, bm)| !floored[i] && bm.contains(id))
        .map(|(i, _)| mu_by_order.get(&orders[i]).copied().unwrap_or(0.0))
        .sum()
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
    // v0.4 energy-weighting knobs (derivation §4/§7), resolved once for the whole batch so every
    // query sees the same `ν/κ/Δ` (batch == serial). All three are sanitized to their domains and
    // fall back to the defaults on a degenerate value (recall-safe), because they are reachable via
    // the public `SearchOpts` builders and feed `powf`/`ln`/divisions plus the debug guards below:
    // `ν ≥ 1` (corroboration depth; `E_max = (1/ν)·ln N` is sensible only for `ν ≥ 1`), `κ ≥ 0`,
    // and a finite `Δ > 0`. The `.is_finite()` checks also reject `NaN` and `+∞` (the latter would
    // slip a bare `d > 0.0`). Note work scales as `~1/Δ`: the engine's plane count, `max_score`,
    // and reachability array all grow with `E_max/Δ`, so a pathologically tiny `Δ` is a
    // memory/`u32`-overflow hazard — the default `0.5` keeps this bounded; no hard lower clamp.
    let nu = opts.nu.unwrap_or(DEFAULT_NU);
    let nu = if nu.is_finite() && nu >= 1.0 {
        nu
    } else {
        DEFAULT_NU
    };
    let kappa = opts.kappa.unwrap_or(DEFAULT_KAPPA);
    let kappa = if kappa.is_finite() && kappa >= 0.0 {
        kappa
    } else {
        DEFAULT_KAPPA
    };
    let delta = {
        let d = opts.delta.unwrap_or(DEFAULT_DELTA);
        if d.is_finite() && d > 0.0 {
            d
        } else {
            DEFAULT_DELTA
        }
    };
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

    // Snapshot-wide corpus stats (N, L̄) for the N-anchored scoring path (energy/floor/stop/null).
    // Read once for the whole batch from this snapshot's rolling counters, so every query sees the
    // same N/avgdl (batch == serial). `matches_batch` ignores these; `CandidateStream` exposes them.
    let (seg_count, seg_len_sum) = schema::read_seg_stats(conn, ns)?;
    let n_segments = seg_count.max(0) as u64;
    let avgdl = if seg_count > 0 {
        seg_len_sum as f64 / seg_count as f64
    } else {
        0.0
    };

    // §7/§12 quantization guard: `Δ < 2·E_floored` keeps `round(E_floored/Δ) ≥ 1`, so a floored
    // gram never quantizes to `0` and drops out of the bit-sliced union. It is satisfiable only
    // once the corpus is large enough for `E_floored` to clear `Δ/2`; on a handful-of-segments
    // corpus `E_floored` shrinks (negative for `N ≲ 4` at the defaults) and the guard cannot hold.
    // That regime is recall-safe regardless — the engine clamps every weight to `≥ 1`, so a
    // floored gram still never vanishes; only its rarity ordering against other floored grams
    // collapses, which a tiny corpus does not need. So the `debug_assert` fires only where
    // `E_floored` is a meaningful positive energy (`GUARD_MIN_E_FLOORED`), catching a genuinely
    // too-coarse `Δ` on a real corpus without tripping the small-N fixtures. Compiled out of
    // release, so it never panics there.
    let e_floored_nats = e_floored(n_segments as f64, nu, kappa);
    debug_assert!(
        e_floored_nats < GUARD_MIN_E_FLOORED || delta < 2.0 * e_floored_nats,
        "Δ ({delta}) too coarse vs realized floored energy E_floored ({e_floored_nats}): floored \
         grams quantize below 1 (N={n_segments}, ν={nu}, κ={kappa})"
    );
    // Flatness ceiling (derivation §4/§7): every gram's energy is `≤ E_max` (the floored grams sit
    // at `E_floored ≤ E_max`, the rest below), so every quantized weight is `≤ ⌊E_max/Δ⌉`. This
    // bound depends on `N` only through `E_max/Δ ~ (ln N)/(ν·Δ)` — i.e. the per-gram weight needs
    // `~log log N` planes — and never on the posting cardinalities, so the engine's op count stays
    // cardinality-independent (the flatness property). Asserted per query below.
    let wq_ceiling = quantize_energy(e_max(n_segments as f64, nu), delta);
    // The contamination floor is `N`/`ν`-constant, so compute it once for the whole batch and
    // thread it into the per-gram energy (no `powf` per gram); reinforces `batch == serial`.
    let df_min_batch = df_min(n_segments as f64, nu);
    // The count credit `μ = max(0, logit σ)` (derivation §3/§9). `σ` is the index-level corpus
    // constant (sanitized at open), so this is one value for the whole batch — read once, never a
    // per-batch aggregate (batch == serial). The per-query §9 concentration cap is applied below.
    let mu = count_credit(index.sigma);

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
        // v0.4 §2/§4/§7: raw logit-idf energies, computed here since the engine is `N`-free. Reused
        // for both the quantized bit-sliced weights (replacing v0.3's `N`-free df-rarity tiers) and
        // the §9 concentration cap below. `present_dfs[i]` is the cardinality of
        // `present_postings[i]`, so everything stays parallel to the postings; the batch-cached
        // `df_min_batch` keeps the floor `powf` out of this per-gram map.
        let energies: Vec<f64> = present_dfs
            .iter()
            .map(|&df| energy_with_floor(df as f64, df_min_batch, n_segments as f64, kappa))
            .collect();
        let weights: Vec<u32> = energies
            .iter()
            .map(|&e| quantize_energy(e, delta))
            .collect();
        debug_assert!(
            weights.iter().all(|&w| w <= wq_ceiling),
            "energy weight exceeds the cardinality-independent E_max ceiling ⌊E_max/Δ⌉={wq_ceiling}: \
             flatness bound violated"
        );
        // v0.4 M2 (§3/§4/§9): per-gram floored flag (query-side `df ≤ df_min`) + gram order
        // (codepoint count), then the §9-capped per-order count credit. The cap ranges over P's
        // energies (floored grams included in `E_top`, §9/§12 — see `concentration_cap`); `σ` is
        // index-level and uniform across orders query-side, so every present non-floored order maps
        // to the same capped `μ`. The per-order bucketing is kept structural for M5's doc-side
        // `ρ = σ(1−ε)^n`. All of this is a pure function of THIS query's grams + the shared
        // (σ, N, ν, κ, Δ) snapshot ⇒ `batch == serial`.
        let present_floored: Vec<bool> = present_dfs
            .iter()
            .map(|&df| (df as f64) <= df_min_batch)
            .collect();
        let present_orders: Vec<u8> = present_tokens
            .iter()
            .map(|t| t.chars().count() as u8)
            .collect();
        let mu_capped = match concentration_cap(&energies) {
            Some(cap) => mu.min(cap),
            None => mu,
        };
        let mut mu_by_order: FxHashMap<u8, f64> = FxHashMap::default();
        for (&order, &floored) in present_orders.iter().zip(&present_floored) {
            if !floored {
                mu_by_order.insert(order, mu_capped);
            }
        }
        // Perf note (M1): unlike v0.3's `N`-free tiers, these are *absolute* energies, so a selected
        // rare gram does not quantize to 1 — the engine's all-weight-1 fast path (overlap = score,
        // no posting retention, no per-candidate `contains`) therefore stops firing for mixed/uniform
        // queries, costing a bounded constant factor (k posting clones + k `contains` per yielded
        // candidate, k ≤ t_max). Flatness still holds (`O(k)`, k bounded by selection); recovering
        // the fast path is deferred to M3's walk reshape, not patched here.
        let counter = Counter::build_weighted(&present_postings, weights, min_shared);
        plans.push(QueryPlan {
            counter,
            present_tokens,
            present_postings,
            present_dfs,
            present_floored,
            present_orders,
            mu_by_order,
            delta,
            selected_strings,
            n_segments,
            avgdl,
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

/// Eager top-`limit` on the **M2 float credit key** (`E_acc·Δ + Σ_n μ_n·popcount_n`, §7/§12).
///
/// Because the float key adds a non-negative count credit, float order differs from the engine's
/// integer best-first order *across* score buckets, so a correct float top-k cannot simply re-sort
/// the integer top-k (a candidate with a lower integer energy but more non-floored matches can
/// belong in the top-k). Under the engine's retained `≥ 1` weight clamp every match contributes to
/// the bit-sliced sum, so the walk's candidate set *is* the full popcount ∪ bit-sliced union; M2
/// drains it whole (bounded by `Σdf ≤ (1+#classes)·C`, derivation §7 — "cheap enough to run over
/// the full candidate set first"), keeps the **max-float** candidate per key, then sorts by the
/// float key and truncates.
///
/// **M2/M3 boundary.** This is the *eager* path only. The bounded-over-sample early-stop (which
/// would avoid full-draining a shallow top-k) and unifying the lazy [`CandidateStream`] onto this
/// same float key are M3's G2 reshape; until then the stream still ranks by the integer score (see
/// its docs). The per-key dedup keeps the highest-**float** segment; the sort tiebreak (integer
/// score desc, then seg id asc) is deterministic so `batch == serial` and the thrash oracle stay
/// reproducible.
fn drain_top_k(prov: &Provenance<'_>, plan: &QueryPlan, limit: usize) -> Result<Vec<Candidate>> {
    let mut walk = plan.counter.walk();
    // Max-float candidate per key, accumulated over the whole bounded union.
    let mut best: FxHashMap<Key, (Candidate, f64)> = FxHashMap::default();
    loop {
        let mut scored = Vec::with_capacity(CHUNK);
        let mut done = false;
        while scored.len() < CHUNK {
            match plan.counter.advance(&mut walk) {
                Some(s) => scored.push(s),
                None => {
                    done = true;
                    break;
                }
            }
        }
        if !scored.is_empty() {
            let seg_ids: Vec<u32> = scored.iter().map(|s| s.id).collect();
            let found = prov.lookup(&seg_ids)?;
            for s in &scored {
                if let Some((key, label)) = found.get(&s.id) {
                    let f = plan.float_score(s);
                    let cand = Candidate {
                        key: key.clone(),
                        label: label.clone(),
                        seg_id: s.id,
                        score: s.score,
                        overlap: s.overlap,
                    };
                    best.entry(key.clone())
                        .and_modify(|slot| {
                            if f > slot.1 {
                                *slot = (cand.clone(), f);
                            }
                        })
                        .or_insert((cand, f));
                }
            }
        }
        if done {
            break;
        }
    }
    let mut ranked: Vec<(Candidate, f64)> = best.into_values().collect();
    ranked.sort_by(|a, b| {
        b.1.total_cmp(&a.1)
            .then_with(|| b.0.score.cmp(&a.0.score))
            .then_with(|| a.0.seg_id.cmp(&b.0.seg_id))
    });
    ranked.truncate(limit);
    Ok(ranked.into_iter().map(|(c, _)| c).collect())
}

/// Eager: top-`limit` matches per query, all queries sharing one snapshot. The safe default
/// front door (`matches`/`matches_batch`). Drains each plan's bounded candidate union, ranks by
/// the M2 float credit key (see [`drain_top_k`]), then hydrates exactly the kept rows.
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
        let kept = drain_top_k(&prov, plan, limit)?;
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
    let walk = plan.counter.walk();
    // N / avgdl live on the plan (computed once in `prepare` from this snapshot's rolling
    // counters); the accessors read them from there. A corpus-relative custom score must not
    // cross a snapshot boundary.
    Ok(CandidateStream {
        index,
        conn,
        plan,
        walk,
        filter: opts.filter,
        ready: VecDeque::new(),
        seen: FxHashSet::default(),
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
/// **M2 interim — ranking asymmetry.** This stream orders candidates by the engine's **integer**
/// bit-sliced energy score (best-first), *not* by the M2 float count-credit key that
/// [`matches`](crate::Reader::matches)/[`matches_batch`](crate::Reader::matches_batch) now rank by
/// (`E_acc·Δ + Σ_n μ_n·popcount_n`, §7/§12). So [`collect_matches`](Self::collect_matches), which
/// takes a best-first prefix, can order a tie-band differently from the eager `matches`. M3's G2
/// reshape unifies them — the stream will accumulate the bounded union and rank by the float key
/// plus the §6 length null. Until then, the candidate's [`score`](Candidate::score) is the integer
/// energy; compose your own float rerank over [`matched_terms`](Self::matched_terms) if you need
/// the credit key from the stream.
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
    done: bool,
    errored: bool,
}

impl<T: Tokenizer> CandidateStream<'_, T> {
    /// Total live segments `N`, from **this search's** snapshot (not `stats()`).
    pub fn n_segments(&self) -> u64 {
        self.plan.n_segments
    }
    /// Mean segment gram length (`avgdl`) on this snapshot. `0.0` on an empty corpus.
    pub fn avgdl(&self) -> f64 {
        self.plan.avgdl
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

#[cfg(test)]
mod energy_tests {
    //! Numerical fixtures for the v0.4 logit-idf energy weighting (derivation §2, §4, §7): the
    //! energy/floor/ceiling values by hand, the `E_floored ≤ E_max` relation, energy monotonicity
    //! and the floor-boundary plateau, the single-gram weight bit-width bound, and the
    //! `Δ < 2·E_floored` guard including its small-N edge.
    use super::{
        DEFAULT_DELTA, DEFAULT_KAPPA, DEFAULT_NU, GUARD_MIN_E_FLOORED, df_min, e_floored, e_max,
        energy, energy_with_floor, quantize_energy,
    };

    const NU: f64 = DEFAULT_NU; // 2.0
    const KAPPA: f64 = DEFAULT_KAPPA; // 0.5
    const DELTA: f64 = DEFAULT_DELTA; // 0.5

    fn approx(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-6, "expected ≈ {b}, got {a}");
    }

    #[test]
    fn df_min_and_e_max_match_derivation() {
        // ν = 2 ⇒ df_min = √N, E_max = ½·ln N.
        approx(df_min(10_000.0, 2.0), 100.0);
        approx(e_max(10_000.0, 2.0), 10_000f64.ln() / 2.0); // ≈ 4.60517
        approx(df_min(1_000.0, 2.0), 1_000f64.sqrt()); // ≈ 31.6228
        approx(e_max(1_000.0, 2.0), 1_000f64.ln() / 2.0); // ≈ 3.45388
        // A degenerate ν disables the floor; E_max collapses on a trivial corpus.
        approx(df_min(1_000.0, 0.0), 0.0);
        approx(e_max(1.0, 2.0), 0.0);
    }

    #[test]
    fn energy_matches_logit_idf_by_hand() {
        let n = 10_000.0_f64;
        // df = 1000 (> df_min = 100): E = ln((N − df − κ)/(df + κ)).
        approx(
            energy(1_000.0, n, NU, KAPPA),
            ((n - 1_000.0 - 0.5) / (1_000.0 + 0.5)).ln(),
        );
        // A gram at or below df_min is floored to df_min, so it carries E_floored exactly.
        approx(energy(100.0, n, NU, KAPPA), e_floored(n, NU, KAPPA)); // df == df_min boundary
        approx(energy(10.0, n, NU, KAPPA), e_floored(n, NU, KAPPA)); // df < df_min
    }

    #[test]
    fn energy_is_nonpositive_for_common_grams() {
        let n = 1_000.0;
        assert!(energy(500.0, n, NU, KAPPA) <= 1e-9); // p ≈ 0.5 ⇒ logit ≈ 0
        assert!(energy(900.0, n, NU, KAPPA) < 0.0); // p > 0.5 ⇒ negative
        assert_eq!(energy(1_000.0, n, NU, KAPPA), f64::NEG_INFINITY); // in every segment ⇒ −∞
    }

    #[test]
    fn e_floored_is_positive_and_at_or_below_e_max_on_a_real_corpus() {
        for &n in &[
            50.0,
            100.0,
            500.0,
            1_000.0,
            5_000.0,
            10_000.0,
            100_000.0,
            1_000_000.0,
        ] {
            let ef = e_floored(n, NU, KAPPA);
            let em = e_max(n, NU);
            assert!(
                ef > 0.0,
                "E_floored positive on a real corpus (N={n}): {ef}"
            );
            assert!(ef <= em + 1e-9, "E_floored {ef} ≤ E_max {em} at N={n}");
        }
    }

    #[test]
    fn energy_wrapper_matches_explicit_floor() {
        // The hot-path `energy_with_floor` (precomputed floor) agrees with the convenience `energy`.
        let n = 100_000.0;
        let dm = df_min(n, NU);
        for &df in &[1.0, 50.0, 316.0, 1_000.0, 50_000.0, 99_999.0] {
            approx(
                energy(df, n, NU, KAPPA),
                energy_with_floor(df, dm, n, KAPPA),
            );
        }
    }

    #[test]
    fn energy_is_monotone_decreasing_in_df_above_the_floor() {
        let n = 100_000.0;
        let dm = df_min(n, NU); // ≈ 316.2
        let mut prev = f64::INFINITY;
        for &df in &[400.0, 1_000.0, 5_000.0, 20_000.0, 49_000.0] {
            assert!(df > dm, "df {df} above the floor {dm}");
            let e = energy(df, n, NU, KAPPA);
            assert!(
                e < prev,
                "energy strictly decreases as df grows: {e} < {prev}"
            );
            prev = e;
        }
    }

    #[test]
    fn floor_boundary_is_a_plateau_then_strictly_below() {
        let n = 10_000.0; // df_min = 100
        let dm = df_min(n, NU);
        // At and below the floor, energy plateaus at E_floored.
        approx(energy(dm, n, NU, KAPPA), e_floored(n, NU, KAPPA));
        approx(energy(dm * 0.5, n, NU, KAPPA), e_floored(n, NU, KAPPA));
        approx(energy(1.0, n, NU, KAPPA), e_floored(n, NU, KAPPA));
        // Strictly above the floor, energy drops strictly below E_floored.
        assert!(energy(dm + 1.0, n, NU, KAPPA) < e_floored(n, NU, KAPPA));
    }

    #[test]
    fn quantize_clamps_and_rounds() {
        // wq = max(0, round(E/Δ)).
        assert_eq!(quantize_energy(2.1917, 0.5), 4); // round(4.3834)
        assert_eq!(quantize_energy(0.24, 0.5), 0); // round(0.48) → 0
        assert_eq!(quantize_energy(0.25, 0.5), 1); // round(0.5) → 1
        assert_eq!(quantize_energy(0.0, 0.5), 0);
        assert_eq!(quantize_energy(-3.0, 0.5), 0); // negative ⇒ 0
        assert_eq!(quantize_energy(f64::NEG_INFINITY, 0.5), 0);
    }

    #[test]
    fn single_gram_weight_bit_width_bound_holds() {
        // §7: a SINGLE gram's quantized weight fits in ⌈log2(E_max/Δ + 1)⌉ bits — this bounds the
        // per-gram weight, NOT the accumulator's plane count (the engine sizes that to bits(Σ wq) =
        // max_score). The realized max single weight is the floored-gram weight (E_floored ≤ E_max),
        // so its bit width respects the bound and grows only as ~log2((ln N)/(νΔ)) — never with the
        // posting cardinalities.
        for &n in &[1_000.0, 10_000.0, 1_000_000.0] {
            let bound = ((e_max(n, NU) / DELTA) + 1.0).log2().ceil() as u32;
            let wq_floored = quantize_energy(e_floored(n, NU, KAPPA), DELTA);
            let wq_max = quantize_energy(e_max(n, NU), DELTA);
            assert!(
                wq_floored <= wq_max,
                "E_floored ≤ E_max ⇒ weight ≤ ceiling (N={n})"
            );
            let bits = 32 - wq_floored.leading_zeros();
            assert!(
                bits <= bound,
                "floored weight {wq_floored} ({bits} bits) ≤ bound {bound}"
            );
        }
    }

    #[test]
    fn quantization_guard_holds_on_a_real_corpus() {
        // §7 guard Δ < 2·E_floored at the defaults, and the floored gram keeps a weight ≥ 1.
        let n = 1_000.0;
        let ef = e_floored(n, NU, KAPPA);
        assert!(ef >= GUARD_MIN_E_FLOORED, "guard is active on this corpus");
        assert!(DELTA < 2.0 * ef, "Δ < 2·E_floored");
        assert!(
            quantize_energy(ef, DELTA) >= 1,
            "floored gram never quantizes to 0"
        );
    }

    #[test]
    fn quantization_guard_degrades_safely_on_a_tiny_corpus() {
        // On N = 4 the floored energy is negative, so the guard cannot hold and is intentionally
        // skipped (E_floored < GUARD_MIN_E_FLOORED). The floored weight quantizes to 0; recall is
        // preserved not here but by the engine's ≥ 1 weight clamp.
        let n = 4.0;
        let ef = e_floored(n, NU, KAPPA);
        assert!(ef < 0.0, "E_floored negative at N=4: {ef}");
        assert!(
            ef < GUARD_MIN_E_FLOORED,
            "guard correctly skipped (tiny corpus)"
        );
        assert!(
            DELTA >= 2.0 * ef,
            "guard cannot hold here — expected, recall-safe via clamp"
        );
        assert_eq!(quantize_energy(ef, DELTA), 0);
    }

    #[test]
    fn energy_helpers_never_panic_on_degenerate_inputs() {
        for &n in &[0.0, 1.0, 2.0, 3.0, 5.0] {
            let _ = df_min(n, NU);
            let _ = e_max(n, NU);
            let _ = e_floored(n, NU, KAPPA);
            let e = energy(1.0, n, NU, KAPPA);
            let _ = quantize_energy(e, DELTA);
        }
    }
}

#[cfg(test)]
mod credit_tests {
    //! Numerical fixtures for the v0.4 count credit `μ` and the §9 concentration cap (derivation
    //! §3/§7/§9/§12): `μ = max(0, logit σ)`; the floor↔μ crossover near `df ≈ 9000`; the cap's
    //! all-common pass-through, its concentrated cap, and its hard floor at 0; the floored-energy
    //! `E_top` interpretation; and the per-order non-floored popcount credit.
    use super::{concentration_cap, count_credit, e_floored, e_max, energy, popcount_credit};
    use crate::DEFAULT_SIGMA;
    use crate::hash::FxHashMap;
    use croaring::Bitmap;

    const NU: f64 = 2.0;
    const KAPPA: f64 = 0.5;

    fn approx(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-6, "expected ≈ {b}, got {a}");
    }

    #[test]
    fn mu_is_logit_sigma_clamped_at_zero() {
        approx(count_credit(0.9), 9.0_f64.ln()); // logit 0.9 = ln 9 ≈ 2.1972
        approx(count_credit(0.5), 0.0); // logit ½ = 0
        assert_eq!(count_credit(0.3), 0.0); // σ < ½ ⇒ negative logit ⇒ clamped to 0 (recall-safe)
        assert!(count_credit(DEFAULT_SIGMA) > 2.0);
    }

    #[test]
    fn floor_mu_crossover_is_near_df_9000() {
        // N=1e6, σ=0.9, ν=2: a single floored junk gram (E_max ≈ E_floored, NO credit) crosses a
        // single non-floored real gram (energy(df) + μ) near df ≈ 9000 (derivation §9). Asserted on
        // raw energies; bit-sliced quantization shifts the realized crossover by ≤ one Δ-bucket.
        let n = 1_000_000.0_f64;
        let mu = count_credit(0.9);
        let junk = e_floored(n, NU, KAPPA); // the junk gram's no-credit energy (≈ E_max)
        approx(e_max(n, NU), 0.5 * n.ln());
        let real = |df: f64| energy(df, n, NU, KAPPA) + mu;
        assert!(
            real(5_000.0) > junk,
            "a df=5000 real gram (E+μ) out-scores floored junk (E_max)"
        );
        assert!(
            real(15_000.0) < junk,
            "a df=15000 real gram loses to floored junk"
        );
        assert!(
            (real(9_000.0) - junk).abs() < 0.1,
            "the ordering flips near df ≈ 9000"
        );
    }

    #[test]
    fn junk_only_does_not_outrank_a_real_gram() {
        // The §9 policy at the score level: one non-floored real gram (df below the crossover) + μ
        // beats one floored junk gram (E_max, no credit). The floor alone would invert this.
        let n = 1_000_000.0_f64;
        let mu = count_credit(0.9);
        let junk = e_floored(n, NU, KAPPA);
        let real = energy(2_000.0, n, NU, KAPPA) + mu; // df=2000 > df_min=1000 ⇒ non-floored
        assert!(real > junk, "real-gram match out-scores junk-only");
    }

    #[test]
    fn cap_passes_through_an_all_common_set() {
        // All energies ≤ 0 (no dominant rare gram) ⇒ not concentrated ⇒ μ survives uncapped.
        assert_eq!(concentration_cap(&[-1.0, -0.5, -2.0]), None);
        // Comparable positive energies with no member below ½·E_top ⇒ no commons ⇒ uncapped.
        assert_eq!(concentration_cap(&[2.0, 2.0, 2.0]), None);
        // Degenerate inputs never panic / never spuriously cap.
        assert_eq!(concentration_cap(&[]), None);
        assert_eq!(
            concentration_cap(&[f64::NEG_INFINITY, f64::NEG_INFINITY]),
            None
        );
    }

    #[test]
    fn cap_limits_a_concentrated_set() {
        // One dominant gram (E_top=5) amid 2 commons (0.5, 0.3 < 2.5): cap = (5 − 0.8)/(2−1) = 4.2.
        let cap = concentration_cap(&[5.0, 0.5, 0.3]).expect("concentrated");
        approx(cap, 4.2);
        // Only 1 common (3.0 ≥ ½·5) ⇒ no dominant-amid-commons ⇒ not concentrated.
        assert_eq!(concentration_cap(&[5.0, 0.5, 3.0]), None);
    }

    #[test]
    fn cap_floors_at_zero_when_commons_outweigh() {
        // The commons collectively outweigh the dominant gram ⇒ the hard floor discards the credit.
        let cap = concentration_cap(&[1.0, 0.4, 0.4, 0.4]).expect("concentrated");
        approx(cap, 0.0); // (1.0 − 1.2)/3 < 0 ⇒ max(0, ·) = 0
    }

    #[test]
    fn cap_includes_floored_energy_in_e_top() {
        // Interpretation guard (§9/§12): a floored gram's E_max-level energy IS eligible to be the
        // dominant E_top — it is not excluded from the cap. E_top=6.9 with 2 commons at 0 ⇒
        // cap = (6.9 − 0)/(2−1) = 6.9, so the high floored energy drove the cap (it was not skipped).
        let cap = concentration_cap(&[6.9, 0.0, 0.0]).expect("floored gram is a valid dominant");
        approx(cap, 6.9);
        // With 3 commons the divisor grows: cap = 6.9/(3−1) = 3.45.
        approx(
            concentration_cap(&[6.9, 0.0, 0.0, 0.0]).expect("concentrated"),
            3.45,
        );
    }

    #[test]
    fn cap_boundary_e_equals_half_is_not_common() {
        // §9 uses a STRICT `E < ½·E_top`: a gram exactly at half is NOT a common, so two grams at
        // the boundary leave 0 commons ⇒ uncapped.
        assert_eq!(concentration_cap(&[4.0, 2.0, 2.0]), None);
        // One nudged just below half ⇒ still only 1 common ⇒ uncapped (needs ≥ 2).
        assert_eq!(concentration_cap(&[4.0, 2.0, 1.999]), None);
        // Two strictly below half ⇒ concentrated.
        assert!(concentration_cap(&[4.0, 1.999, 1.999]).is_some());
    }

    #[test]
    fn cap_neg_infinity_commons_contribute_zero_no_nan() {
        // df = N ubiquitous grams carry −∞ energy: they count as commons (`< ½·E_top`) but
        // contribute `max(0,−∞) = 0` to the sum, so the cap stays finite (no NaN).
        let cap =
            concentration_cap(&[5.0, f64::NEG_INFINITY, f64::NEG_INFINITY]).expect("2 commons");
        approx(cap, 5.0); // (5 − 0)/(2 − 1)
        assert!(cap.is_finite());
        // More ubiquitous commons shrink the cap (the chance-match guard), still finite.
        let cap =
            concentration_cap(&[5.0, f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY])
                .expect("3 commons");
        approx(cap, 2.5); // 5/(3 − 1)
        assert!(cap.is_finite());
    }

    #[test]
    fn cap_floored_e_top_with_co_present_real_gram() {
        // R1's interpretation pin (§9/§12): a FLOORED gram pins E_top = E_max (≈6.9) while a real
        // mid-rare gram (4.0) is co-present above the commons (0.5, 0.5). Under the literal §12
        // reading the real gram is NOT a common (4.0 ≥ ½·6.9 = 3.45), so only the two 0.5 commons
        // count: cap = (6.9 − (0.5+0.5))/(2 − 1) = 5.9 — the high floored E_top loosens the cap.
        let cap = concentration_cap(&[6.9, 4.0, 0.5, 0.5]).expect("concentrated");
        approx(cap, 5.9);
        // A floored-EXCLUDED reading (not what M2 does) would instead take E_top = 4.0, making the
        // 0.5s its commons and the cap a tighter (4.0 − 1.0)/1 = 3.0 — pinned here for contrast so a
        // future derivation-text change to exclude floored grams is a visible, deliberate edit.
        approx(
            concentration_cap(&[4.0, 0.5, 0.5]).expect("concentrated"),
            3.0,
        );
    }

    #[test]
    fn popcount_credit_sums_non_floored_matched_by_order() {
        // Three present grams: a non-floored order-3 (μ=2.0), a FLOORED order-3 (no credit), and a
        // non-floored order-2 (μ=1.5).
        let postings = vec![Bitmap::of(&[7]), Bitmap::of(&[7]), Bitmap::of(&[7, 9])];
        let floored = vec![false, true, false];
        let orders = vec![3u8, 3u8, 2u8];
        let mut mu_by_order = FxHashMap::default();
        mu_by_order.insert(3u8, 2.0);
        mu_by_order.insert(2u8, 1.5);
        // id 7 is in all three, but the floored one earns nothing: 2.0 + 1.5 = 3.5.
        approx(
            popcount_credit(7, &postings, &floored, &orders, &mu_by_order),
            3.5,
        );
        // id 9 is only in the order-2 posting: 1.5.
        approx(
            popcount_credit(9, &postings, &floored, &orders, &mu_by_order),
            1.5,
        );
        // id 1 is in nothing: 0.
        approx(
            popcount_credit(1, &postings, &floored, &orders, &mu_by_order),
            0.0,
        );
    }
}
