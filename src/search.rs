//! The query pipeline (the spine): resolve → select → engine candidate-gen → provenance/filter
//! → hydrate.
//!
//! [`Index`](crate::Index) exposes the lifecycle and the lease types; the read path lives here.
//! The IDF-weighted bit-sliced overlap counting itself lives in the [`trifle_overlap`] engine
//! ([`Counter`]); this module wires storage to it: it loads the selected tokens' croaring
//! postings, hands them to the engine, then walks the engine's best-first scored ids,
//! batch-hydrating provenance (and applying the opt-in [`SqlFilter`](crate::SqlFilter)) per chunk,
//! folding results at the caller's retrieval granularity ([`Collapse`], v0.5: per **segment** by
//! default, per-key best on [`Collapse::Key`]).
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
use trifle_overlap::{Counter, Scored};

use crate::dict::TermId;
use crate::filter::SqlFilter;
use crate::hash::{FxHashMap, FxHashSet};
use crate::instrument::trace_debug;
use crate::model::{Key, KeyShape, Match};
use crate::select::{GramRow, SelectParams, select};
use crate::store::{Namespace, ReadConn};
use crate::term::Term;
use crate::tokenize::Tokenizer;
use crate::welford::ClassSnap;
use crate::{
    Collapse, DEFAULT_C_MARGIN, DEFAULT_DELTA, DEFAULT_K_TARGET, DEFAULT_KAPPA, DEFAULT_MIN_SHARED,
    DEFAULT_NU, Error, Index, IntoTerm, Result, SearchOpts, TYPO_DAMAGE, postings, schema,
};

/// The retrieval-granularity fold key (v0.5): what one result *is*. [`Collapse::None`] keys
/// results by segment id (every matching segment is its own result); [`Collapse::Key`] folds to
/// the per-key best segment (the entity-style collapse — `limit` then counts keys).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum ResultKey {
    Seg(u32),
    Key(Key),
}

/// The fold key for one candidate under the search's [`Collapse`] granularity.
fn result_key(collapse: Collapse, key: &Key, seg_id: u32) -> ResultKey {
    match collapse {
        Collapse::None => ResultKey::Seg(seg_id),
        Collapse::Key => ResultKey::Key(key.clone()),
    }
}

/// How many engine candidates to pull per provenance/filter round-trip.
const CHUNK: usize = 64;

/// The reciprocal-rank-fusion rank constant `k_RRF` (derivation §8): `RRF(seg) = Σ_v w_v /
/// (k_RRF + rank_v)`. The standard Cormack value — a flatter (larger) constant trusts more of
/// each view's ranked tail; `60` balances the head against the tail. A pure shape constant, so
/// it is an internal `const`, not a [`SearchOpts`](crate::SearchOpts) knob.
const K_RRF: f64 = 60.0;

/// The slope of the linear `ΔH → primary-view weight` map (derivation §8): `w_primary =
/// clamp(0.5 + RRF_GAMMA·ΔH, 0.1, 0.9)`, with `ΔH = ln V_primary − ln V_secondary` the
/// per-`(script, order)` vocabulary-complexity gap (richer primary ⇒ more primary weight). A
/// fixed shape choice (§8 calls the map "a fixed shape choice; a linear map suffices"); equal
/// weights when `ΔH` is unavailable.
const RRF_GAMMA: f64 = 0.1;

/// The §5/§12 starvation energy ratio `ρ`: a present script is **starved** (so the secondary
/// rank-view is brought in) when its pruned/collected primary energy `Σ max(0,E)` falls below
/// `ρ ·` its full primary energy — the budget cut usable signal (`collected_energy_far_below`,
/// derivation §12). Recall-safe: the secondary view only *adds* a fused robustness layer.
const STARVED_ENERGY_RATIO: f64 = 0.5;

/// The realized floored-gram energy `E_floored` (nats) below which the §7 quantization guard
/// `Δ < 2·E_floored` is treated as inapplicable rather than violated. On a handful-of-segments
/// corpus `E_floored` shrinks below this and the guard cannot hold; that regime is recall-safe via
/// the §7 **count-only union** ([`score_union`]) — a floored gram that quantizes to weight 0 no
/// longer vanishes (v0.4 removed the engine's `≥ 1` clamp), it is recovered as a count-only
/// candidate — so the debug-time guard is skipped there to avoid tripping on legitimately tiny
/// corpora. The guard still fires for a genuinely too-coarse `Δ` once the corpus is large enough
/// for `E_floored` to clear this threshold. See [`prepare`].
const GUARD_MIN_E_FLOORED: f64 = 0.5;

/// The marginal-probability threshold `P_LINEAR` separating the §6 length null's two regimes. A
/// gram with `p_g = df_g/N < P_LINEAR` is **rare**: its presence rate `π_g(L) = 1−(1−p_g)^(L/L̄)`
/// is within `(L/L̄)·p_g` to first order, so its null contribution folds into the separable,
/// precomputed-once `K_rare = Σ p_g·weight(g)` (derivation §6/§12). A gram with `p_g ≥ P_LINEAR` is
/// **common** and gets the exact saturating term per candidate.
///
/// Recall direction (derivation §6): the linear form *over*-estimates `π_g` (`(L/L̄)·p ≥ π_g`), so
/// it over-debits — but only for rare grams, where the gap is `O((L·p/L̄)²)` and tiny (`< 0.5%` of
/// the gram's weight at `p < 0.02`, `L/L̄ ≤ 5`, trifle's small-segment regime). The genuinely
/// common grams — where an un-saturated linear debit would exceed `1` and wrongly bury a long
/// relevant segment — get the **exact** saturating `π_g`, which never over-debits. So a *lower*
/// `P_LINEAR` is strictly safer (more grams exact) at an `O(k)` per-candidate cost; `0.02` is the
/// conservative default. It is a pure shape constant (no corpus dependence), so it is an internal
/// `const`, not a [`SearchOpts`](crate::SearchOpts) knob.
const P_LINEAR: f64 = 0.02;

/// A scored, provenance-only candidate (no text — see [`CandidateStream::hydrate`]).
///
/// `seg_id` is snapshot-specific (a [`rebuild`](crate::Index::rebuild) reassigns it), so do not
/// carry a `Candidate` across streams/snapshots.
///
/// Not `Eq` (the [`corrected_score`](Self::corrected_score) ranking key is `f64`).
#[derive(Clone, Debug, PartialEq)]
pub struct Candidate {
    key: Key,
    label: String,
    seg_id: u32,
    score: u32,
    overlap: u32,
    corrected_score: f64,
    /// The §10 score components (all in **nats**), from the candidate's **governing** rank-view (its
    /// best-ranked / retained view — never a cross-view sum; see [`energy`](Self::energy)). Stored at
    /// score-time so a downstream fusion consumer reads a stable, cross-query-comparable magnitude
    /// ([`nat_score`](Self::nat_score)) without recomputation.
    energy: f64,
    count: f64,
    length: f64,
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
    /// The integer bit-sliced **energy** component `E_acc` (derivation §7) — the quantized
    /// logit-idf overlap sum, in `Δ` units. A *component* of the score, **not** the ranking key
    /// (see [`corrected_score`](Self::corrected_score)). Since v0.4 removed the engine's `≥ 1`
    /// weight clamp, a candidate matching only weight-0 (common) grams has `score() == 0` while
    /// `overlap() > 0`, so `score() ≥ overlap()` no longer holds — `score` is energy, not a count.
    pub fn score(&self) -> u32 {
        self.score
    }
    /// How many selected tokens this candidate shares (the raw, unweighted count).
    pub fn overlap(&self) -> u32 {
        self.overlap
    }
    /// The **ranking key** trifle sorts by — an `f64` whose meaning depends on the query's path:
    /// - **single-view (clean / not-starved) query:** the §6/§7 length-corrected score
    ///   `E_acc·Δ + Σ_n μ_n·popcount_n − null` (energy in nats + count credit − the saturating
    ///   length null); it can be negative (the null dominates).
    /// - **fused (starved, multi-view) query:** the reciprocal-rank-fusion score
    ///   `Σ_v w_v/(k_RRF + rank_v)` (derivation §8), a small positive value (≈ `0.01`–`0.03` at the
    ///   default `k_RRF = 60`) on a different scale from the single-view float.
    ///
    /// So it is the right key to sort *this query's* candidates by, but it is **not comparable
    /// across queries** of differing cleanliness (a fused query's scores are RRF-scale, a clean
    /// query's are nat-scale). Use this — not [`score`](Self::score) — for a custom rerank that
    /// wants trifle's own per-query ordering value.
    pub fn corrected_score(&self) -> f64 {
        self.corrected_score
    }

    /// The matched grams' logit-idf **energy** `E_acc·Δ`, in **nats**, from this candidate's
    /// **governing** rank-view (the best-ranked / retained view — never a cross-view sum, which
    /// would double-count the same text at two granularities; see [`corrected_score`] on the fused
    /// path). A standalone §10 component usable on its own; a downstream fusion consumer combines
    /// it with [`count`](Self::count)/[`length`](Self::length) via [`nat_score`](Self::nat_score).
    ///
    /// [`corrected_score`]: Self::corrected_score
    pub fn energy(&self) -> f64 {
        self.energy
    }

    /// The §3/§9 **count credit** `Σ μ` (nats) over this candidate's matched **non-floored** grams,
    /// from its governing rank-view. A standalone §10 component (see [`energy`](Self::energy)).
    pub fn count(&self) -> f64 {
        self.count
    }

    /// The §6 saturating **length null** (nats) debited from this candidate, from its governing
    /// rank-view. A standalone §10 component (see [`energy`](Self::energy)).
    pub fn length(&self) -> f64 {
        self.length
    }

    /// trifle's length-corrected **relevance magnitude** in nats: `energy + count − length` (§10).
    /// Unlike [`corrected_score`](Self::corrected_score) — which is the within-query RRF rank key on
    /// a fused query and flips scale by query cleanliness — this is **cross-query comparable** by
    /// construction, the fusion-ready absolute-goodness input for a downstream consumer. For a
    /// single-view (clean) query it *equals* `corrected_score()`; for a fused query it is the stable
    /// nat-scale magnitude of the governing view while `corrected_score()` remains the RRF rank key.
    pub fn nat_score(&self) -> f64 {
        self.energy + self.count - self.length
    }
}

/// A query's `(script, order)` class sets: `actual` (its own gram classes — the derived budget's
/// pool) and `with_siblings` (plus the ±1 sibling orders, for the ΔH vocabulary lookups).
type QueryClassSets = (FxHashSet<(u8, u8)>, FxHashSet<(u8, u8)>);

/// The distinct **word-tagged** tokens per query and the batch-wide distinct **term** set (the
/// resolution input). Each token carries its §5 word id (the comonotone stopping-block id the
/// Cantelli stop needs). The read path stays in term-space: it resolves from each token's
/// [`term()`](crate::IntoTerm::term) (no `Token → String → re-encode`). A token wider than the
/// encoding ceiling has no term and rides along as an absent token (df 0).
fn query_terms<Tk: IntoTerm>(
    queries: &[&str],
    tokenize: impl Fn(&str) -> Vec<(Tk, u32)>,
) -> (Vec<Vec<(Tk, u32)>>, Vec<Term>) {
    let query_tokens: Vec<Vec<(Tk, u32)>> = queries.iter().map(|q| tokenize(q)).collect();
    let all_terms: Vec<Term> = query_tokens
        .iter()
        .flat_map(|q| q.iter().filter_map(|(t, _)| t.term()))
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
    /// Per present gram (parallel to `present_postings`): its quantized bit-sliced energy weight
    /// `wq = max(0, round(E_g/Δ))` (derivation §7). Used to identify the **weight-0** postings whose
    /// union recovers count-only candidates (the §7 union), and as the energy term of the §6 null's
    /// per-gram `weight(g)`.
    present_weights: Vec<u32>,
    /// The §9-capped count credit `μ_n = min(max(0, logit σ), cap)`, keyed by the gram order `n`
    /// present among the **non-floored** grams. Looked up by `present_orders` when scoring credit.
    mu_by_order: FxHashMap<u8, f64>,
    /// The energy quantization step `Δ` (nats) this plan was built with — reconverts the integer
    /// bit-sliced energy back to nats for the float ranking key (derivation §7).
    delta: f64,
    /// The §6 length-null slope over the **rare** grams (`p_g < P_LINEAR`): `K_rare = Σ p_g·weight(g)`
    /// with `weight(g) = wq·Δ + (μ if non-floored)`. Precomputed once per query (the rare grams'
    /// `π_g` is linear, hence separable), so the per-candidate null is `(L_d/L̄)·K_rare` plus the
    /// few `null_commons` saturating terms. Pure function of this query's grams + the snapshot
    /// (`batch == serial`).
    k_rare: f64,
    /// The §6 length-null's **common** grams (`p_g ≥ P_LINEAR`), each as `(p_g, weight(g))`. Each
    /// needs the exact per-candidate saturating term `(1−(1−p_g)^(L_d/L̄))·weight(g)` (the linear
    /// form would over-debit a long segment). Few — rarest-first selection keeps `#commons ≤ k`.
    null_commons: Vec<(f64, f64)>,
    /// The maximum count credit any candidate can earn — `Σ μ_order` over the **non-floored** present
    /// grams (matching all of them). The eager over-sample early-stop's upper bound on an un-yielded
    /// candidate's float is `c·Δ + cred_max` (the null, ≥ 0, only lowers scores, so dropping it keeps
    /// the bound valid — derivation §7).
    cred_max: f64,
    selected_strings: Vec<String>,
    /// Mean segment gram length `L̄` (`avgdl`) on this snapshot — the §6 length null's denominator.
    /// (`N` lives on the wrapping [`PlannedQuery`], which the stream's accessors read.)
    avgdl: f64,
}

impl QueryPlan {
    /// Raw overlap **and** the derivation §3/§9 count credit for `id` in a **single** fused pass
    /// over the present postings (one `contains` per posting — v0.5's one per-candidate sweep,
    /// now that the engine neither retains postings nor gates on raw overlap). The credit is `μ`
    /// times the popcount of matched **non-floored** grams, bucketed by gram order (§7/§12
    /// `Σ_n μ_n·popcount_n`); floored (junk-suspect) grams earn nothing (§9) — the policy that
    /// restores the junk-below-real ordering the floor's `E_max` energy alone would invert.
    /// `O(k)` per candidate (`k` = the selected present grams), so the post-pass stays bounded
    /// (flatness). Used by the walk (which gates `overlap ≥ floor` before any SQL) and by the §7
    /// count-only recovery.
    fn overlap_and_credit(&self, id: u32) -> (u32, f64) {
        overlap_and_credit(
            id,
            &self.present_postings,
            &self.present_floored,
            &self.present_orders,
            &self.mu_by_order,
        )
    }

    /// The §6 length null for a candidate of relative length `rel_len = L_d/L̄`: the separable
    /// rare-gram slope `rel_len·K_rare` plus the saturating common-gram terms (derivation §6/§12).
    /// Subtracted from `E_acc·Δ + credit` to form the [`corrected_score`](Candidate::corrected_score).
    fn length_null(&self, rel_len: f64) -> f64 {
        length_null(rel_len, self.k_rare, &self.null_commons)
    }

    /// The §6/§7 length-corrected ranking key `E_acc·Δ + credit − null` **together with its §10
    /// components** `(corrected, energy, count, length)`, all in nats — so [`score_union`] stores the
    /// components on the [`Candidate`] (C4) without recomputing. `e_acc` is the integer bit-sliced
    /// energy, `credit` the §3/§9 count credit, `rel_len = L_d/L̄`; `corrected = energy + count −
    /// length`.
    fn corrected_parts(&self, e_acc: u32, credit: f64, rel_len: f64) -> (f64, f64, f64, f64) {
        let energy = e_acc as f64 * self.delta;
        let length = self.length_null(rel_len);
        (energy + credit - length, energy, credit, length)
    }
}

/// One query planned into its rank-views (derivation §8) — the reciprocal-rank-fusion unit.
///
/// `views` holds 1 [`QueryPlan`] (PRIMARY-only, the clean-query path — the same CODE PATH as
/// M1–M4: `score_union` with the eager early-stop, the candidate set + recall unchanged) or 2
/// (PRIMARY + SECONDARY, a starved query). Note the single-view *code* path is unchanged, but a
/// clean query's corrected-score *magnitude* shifts slightly vs a trigram-only (pre-M5) index:
/// `avgdl`/`seg.len` now count dual-order grams, so the §6 length null's `L_d/L̄` re-bases — a
/// recall-safe precision-level consequence of dual-order indexing, not a ranking-set change. Each
/// view is scored independently by the full [`score_union`] pipeline; with 2 views the per-view
/// rankings are RRF-fused (`view_weights`, `K_RRF`, `missing="omit"`). `fused_selected` is the union
/// of every view's selected token strings,
/// for span location at hydrate (a fused candidate may have matched in either view). `n_segments` /
/// `avgdl` are the shared snapshot stats (also on each view's plan, but held here so the accessors
/// work even when `views` is empty — a query with no in-corpus gram in any view).
struct PlannedQuery {
    views: Vec<QueryPlan>,
    view_weights: Vec<f64>,
    fused_selected: Vec<String>,
    n_segments: u64,
    avgdl: f64,
}

/// The §6 saturating length null `(L_d/L̄)·K_rare + Σ_commons (1−(1−p_g)^(L_d/L̄))·weight(g)`, a free
/// function so it is unit-testable without a [`QueryPlan`]. `rel_len = L_d/L̄ ≥ 0`; each `(p, w)` is
/// a common gram's marginal `p_g ∈ [0,1]` and its `weight(g)`. `p = 1` (a ubiquitous gram) gives
/// `(1−0^rel_len) = 1`, debiting the full weight — exactly cancelling that gram's credit.
fn length_null(rel_len: f64, k_rare: f64, commons: &[(f64, f64)]) -> f64 {
    let mut null = rel_len * k_rare;
    for &(p, w) in commons {
        null += (1.0 - (1.0 - p).powf(rel_len)) * w;
    }
    null
}

/// Relative length `L_d/L̄` for a candidate, from its fetched distinct-gram count and `avgdl`.
/// `0` when `avgdl ≤ 0` (an empty corpus, which has no candidates anyway) or the length is missing.
fn rel_len(len: Option<i64>, avgdl: f64) -> f64 {
    if avgdl <= 0.0 {
        return 0.0;
    }
    (len.unwrap_or(0).max(0) as f64) / avgdl
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
/// `E_g = ln((N − df_eff + κ)/(df_eff + κ))`, the RSJ log-odds of the **exact** Beta(κ,κ)
/// posterior-mean presence rate `p̂ = (df_eff + κ)/(N + 2κ)` (Jeffreys smoothing at `κ = ½`,
/// derivation §2/§4), with `df_eff = max(df, df_min)`. Negative for `p̂ > 0.5` (clamped to `0` at
/// quantization) and **finite for every `df ∈ [0, N]`** — at `df = N` it is `ln(κ/(N + κ))`, so
/// the pre-v0.5 `−∞` special case for a ubiquitous gram is dissolved (that was an artifact of the
/// unnormalized `(N − df − κ)` numerator, undefined at `df ≥ N − κ`). The guard below is a pure
/// out-of-contract backstop (`df > N + κ`, or `κ = 0` on a degenerate input), unreachable at the
/// sanitized defaults; it degrades to weight `0` via [`quantize_energy`], recall-safe.
fn energy_with_floor(df: f64, df_min: f64, n: f64, kappa: f64) -> f64 {
    let df_eff = df.max(df_min);
    let num = n - df_eff + kappa;
    let den = df_eff + kappa;
    if num <= 0.0 || den <= 0.0 {
        return f64::NEG_INFINITY; // out-of-contract backstop; weight 0 at use
    }
    (num / den).ln()
}

/// Per-gram query-side energy (derivation §2/§4) — the convenience form that derives the floor
/// `df_min(N, ν)` itself; see [`energy_with_floor`] for the hoisted hot-path variant.
fn energy(df: f64, n: f64, nu: f64, kappa: f64) -> f64 {
    energy_with_floor(df, df_min(n, nu), n, kappa)
}

/// Realized floored-gram energy `E_floored = ln((N − df_min + κ)/(df_min + κ)) ≤ E_max` — the
/// energy every floored (`df ≤ df_min`) gram carries (derivation §4/§7). The §7 quantization guard
/// `Δ < 2·E_floored` keeps `round(E_floored/Δ) ≥ 1`, so a floored gram never quantizes to `0`. For
/// a tiny corpus this is small or negative — see [`prepare`]'s guard.
fn e_floored(n: f64, nu: f64, kappa: f64) -> f64 {
    // Routes through `energy` (computing `df_min` for both the `df` argument and the floor) so the
    // convenience wrapper stays exercised; called once per batch, so the extra `powf` is free.
    energy(df_min(n, nu), n, nu, kappa)
}

/// Quantize an energy to the bit-sliced weight `wq = max(0, round(E/Δ))` (derivation §7). A
/// non-finite or non-positive energy maps to `0`. A weight-0 gram adds **no** energy to the
/// bit-sliced planes (v0.4 removed the engine's `≥ 1` clamp); a candidate matching only weight-0
/// grams is recovered as a *count-only* candidate by the §7 union in [`score_union`] rather than
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

/// The derived work budget `C` (derivation §5/§7) — the Lagrangian dual of the §5 confidence-bounded
/// stop, and the default `df_budget` (v0.4/M6). The stop targets `ln(N/k)` nats of identification
/// evidence; `df_budget` is that same requirement in **work-space** (postings scanned). Scanning
/// `df` postings buys `≈ ln(N/df)` nats, so the work to collect the target is
///
/// ```text
/// C = (1/r) · ln(N/k) · d̄ / ln(N/d̄),   d̄ = exp(mean_lndf + Z·std_lndf),  Z = 2
/// ```
///
/// with `r = σ` the reliability, `k = k_target`, and `d̄` a representative surviving-gram df at the
/// `Z = 2` (≈ P98) percentile of the query's present classes' `ln df` (pooled by
/// [`ClassSnap::pooled_log_df`](crate::welford::ClassSnap)). `Z > 0` makes `d̄` a *commoner*
/// percentile ⇒ a larger, **recall-safer** `C`; the in-tree recall-guard (`tests/recall_budget.rs`)
/// is the gate on this shape constant.
///
/// **Recall-safe guards — every degeneracy returns `None` (unbounded), never a too-tight `C`:**
/// stats unavailable (every present class `n < 2`), `N ≤ k`, a non-finite / non-positive `r`,
/// `d̄ ∉ [1, N−1]`, `ln(N/d̄) ≤ 0`, `ln(N/k) ≤ 0`, or a non-finite / overflowing result.
fn derived_budget(n_segments: u64, k_target: u64, r: f64, class_snap: &ClassSnap) -> Option<u64> {
    /// The `d̄` percentile shape constant `Z` (derivation §5/§7, ratified decision A): `Z = 2`
    /// (≈ P98 of `ln df`) — recall-safe with margin, still capping the common-gram tail so `C` bites.
    const Z: f64 = 2.0;

    // A corpus no larger than the target pool has no meaningful pruning to do.
    if n_segments <= k_target {
        return None;
    }
    if !r.is_finite() || r <= 0.0 {
        return None;
    }
    let (mean_log_df, std_log_df) = class_snap.pooled_log_df()?;
    if !mean_log_df.is_finite() || !std_log_df.is_finite() {
        return None;
    }
    let n = n_segments as f64;
    let k = k_target.max(1) as f64;
    let d_bar = (mean_log_df + Z * std_log_df).exp();
    // d̄ must be a valid df strictly inside `(1, N)` so `ln(N/d̄) > 0`.
    if !d_bar.is_finite() || d_bar < 1.0 || d_bar > n - 1.0 {
        return None;
    }
    let ln_n_dbar = (n / d_bar).ln();
    let ln_n_k = (n / k).ln();
    if ln_n_dbar <= 0.0 || ln_n_k <= 0.0 {
        return None;
    }
    let c = (1.0 / r) * ln_n_k * d_bar / ln_n_dbar;
    if !c.is_finite() || c < 0.0 || c >= u64::MAX as f64 {
        return None;
    }
    Some(c.round() as u64)
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
/// `cap = max(0, (E_top − Σ_common max(0,E)) / (#common − 1 + 1{anchor floored}))`. The hard floor
/// at 0 (reached when the commons collectively outweigh the dominant gram) is the baseline; §9's
/// smoother shrink toward the cap is a deferred tuning refinement.
///
/// **The denominator tracks the anchor's credit (v0.5 — resolves handoff flag #2).** The cap is
/// derived from `score(on-topic) ≥ score(commons-doc)`. A **non-floored** dominant gram itself
/// earns `μ`, so `E_top + μ ≥ Σ E_c + #c·μ` gives the `#c − 1` denominator. A **floored** anchor —
/// the common case: `df_min = √N` floors genuinely-rare real grams — earns **no** credit (§9
/// withholding), the inequality is `E_top ≥ Σ E_c + #c·μ`, and the tight cap divides by `#c`. The
/// pre-v0.5 formula used `#c − 1` unconditionally, over-crediting a commons-only doc by exactly
/// one `μ` past the floored-anchor doc — precisely the flag-#2 inversion (off-topic 8.09 vs
/// on-topic 6.20). No ambiguity at the argmax: a floored gram carries `E_floored`, strictly above
/// every non-floored energy, so ties never straddle the floored/non-floored boundary.
///
/// **Floored grams are NOT excluded from `E_top` — the literal §12 reading (v0.4/M6, resolved;
/// unchanged in v0.5).** The cap ranges over *all* of `P`. This is deliberate: on a large corpus
/// `df_min = √N` is a low bar, so a *genuinely rare, real* discriminating gram is itself floored
/// (sits at `E_max`) — the common and valuable "find the doc with this rare term" query. Keeping
/// it in `E_top` lets it anchor the cap, tightening μ so a commons-only doc cannot out-credit the
/// on-topic doc (the strong case). Excluding floored grams (considered and rejected in M6) would
/// instead *disable* the cap for exactly those queries — no non-floored dominant remains, so μ
/// goes uncapped and the commons win. The denominator correction above is what makes the retained
/// floored anchor actually *protect* its doc rather than merely bound the loss. The floored
/// *exclusion* still governs which grams earn count credit (`overlap_and_credit`) and the M4 stop —
/// just not the cap's `E_top`.
fn concentration_cap(energies: &[f64], floored: &[bool]) -> Option<f64> {
    debug_assert_eq!(energies.len(), floored.len());
    let mut top: Option<usize> = None;
    for (i, &e) in energies.iter().enumerate() {
        if top.is_none_or(|t| e > energies[t]) {
            top = Some(i);
        }
    }
    let top = top?;
    let e_top = energies[top];
    if e_top <= 0.0 {
        return None; // no dominant rare gram (all-common / empty) — μ survives (§7/§9)
    }
    let half = 0.5 * e_top;
    let commons: Vec<f64> = energies.iter().copied().filter(|&e| e < half).collect();
    if commons.len() < 2 {
        return None;
    }
    let sum_commons: f64 = commons.iter().map(|&e| e.max(0.0)).sum();
    let denom = commons.len() - 1 + usize::from(floored[top]);
    let cap = (e_top - sum_commons) / denom as f64;
    Some(cap.max(0.0))
}

/// The fused per-candidate pass (v0.5): raw overlap over ALL matched grams, and the `μ`-weighted
/// popcount credit of the matched **non-floored** grams, bucketed by gram order (derivation
/// §7/§12 `Σ_n μ_n·popcount_n`), in one `contains` sweep. A floored gram counts toward overlap
/// but earns no credit (§9); a non-floored gram contributes its order's capped `μ_n`. Free
/// function so it is unit-testable without a full [`QueryPlan`]. `O(k)` over the present grams.
fn overlap_and_credit(
    id: u32,
    postings: &[Bitmap],
    floored: &[bool],
    orders: &[u8],
    mu_by_order: &FxHashMap<u8, f64>,
) -> (u32, f64) {
    let mut overlap = 0u32;
    let mut credit = 0.0;
    for (i, bm) in postings.iter().enumerate() {
        if bm.contains(id) {
            overlap += 1;
            if !floored[i] {
                credit += mu_by_order.get(&orders[i]).copied().unwrap_or(0.0);
            }
        }
    }
    (overlap, credit)
}

/// Resolve, select (class-aware rarest-first), load postings, and build the engine [`Counter`]
/// for every query — all against the open snapshot `conn` (a tx must already be open). Verifies
/// the dictionary generation against the snapshot (a concurrent id-reassigning rebuild → retryable
/// [`Error::Busy`]). One [`PlannedQuery`] (1–2 rank-views) per query, in order; `batch == serial`
/// (selection/df/weights/rank-views/ΔH derive only from each query's own tokens + the shared
/// snapshot).
fn prepare<T: Tokenizer>(
    index: &Index<T>,
    conn: &Connection,
    ns: &Namespace,
    queries: &[&str],
    opts: &SearchOpts<'_>,
) -> Result<Vec<PlannedQuery>> {
    let (query_tokens, all_terms) = query_terms(queries, |q| index.distinct_tokens_tagged(q));

    // Per-query class sets (v0.5, the batch==serial repair): `actual` is the query's own
    // `(script, order)` classes — the pool the derived budget's `d̄` reflects, since it must
    // represent the postings THIS query will scan, never a co-batched query's regime.
    // `with_siblings` adds the ±1 sibling orders so the rank-view ΔH vocabulary lookups
    // (`view_weights_from_dh`) resolve identically solo or batched (a structural-fallback query
    // produces only one order but weighs both). The batch snapshot below is the union — a
    // superset is harmless because every entry is a global corpus statistic; only *pool
    // membership* was ever a leak.
    let q_classes: Vec<QueryClassSets> = query_tokens
        .iter()
        .map(|q| {
            let mut actual: FxHashSet<(u8, u8)> = FxHashSet::default();
            let mut with_siblings: FxHashSet<(u8, u8)> = FxHashSet::default();
            for (tok, _) in q {
                if let Some(t) = tok.term() {
                    let (c, o) = (t.class(), t.order());
                    actual.insert((c, o));
                    with_siblings.insert((c, o));
                    if o >= 2 {
                        with_siblings.insert((c, o - 1));
                    }
                    if o < 3 {
                        with_siblings.insert((c, o + 1));
                    }
                }
            }
            (actual, with_siblings)
        })
        .collect();
    let batch_classes: FxHashSet<(u8, u8)> = q_classes
        .iter()
        .flat_map(|(_, ws)| ws.iter().copied())
        .collect();

    // Resolve terms in memory + capture the dict generation atomically, then read the snapshot's
    // stored generation (the tx pins the WAL snapshot) to compare. A skew means a concurrent
    // rebuild/reset reassigned term-ids vs this snapshot — surface as retryable Busy (the store
    // is the consistent new generation; the caller retries on a fresh reader). No internal retry.
    let (resolved, gen_mem, class_snap) = index
        .dict
        .resolve_terms(&all_terms, batch_classes.iter().copied());
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
    let nu = opts.tuning.nu.unwrap_or(DEFAULT_NU);
    let nu = if nu.is_finite() && nu >= 1.0 {
        nu
    } else {
        DEFAULT_NU
    };
    let kappa = opts.tuning.kappa.unwrap_or(DEFAULT_KAPPA);
    let kappa = if kappa.is_finite() && kappa >= 0.0 {
        kappa
    } else {
        DEFAULT_KAPPA
    };
    let delta = {
        let d = opts.tuning.delta.unwrap_or(DEFAULT_DELTA);
        if d.is_finite() && d > 0.0 {
            d
        } else {
            DEFAULT_DELTA
        }
    };
    // v0.4 M4 §5 Cantelli-stop knobs, resolved once for the batch (batch == serial). `c ≥ 0`
    // (a Cantelli margin), `k ≥ 1` (the stop target `ln(N/k)`), both falling back on a degenerate
    // value (recall-safe). `σ` is the index-level corpus constant, sanitized at open.
    let c_margin = {
        let c = opts.tuning.c_margin.unwrap_or(DEFAULT_C_MARGIN);
        if c.is_finite() && c >= 0.0 {
            c
        } else {
            DEFAULT_C_MARGIN
        }
    };
    let k_target = opts.tuning.k_target.unwrap_or(DEFAULT_K_TARGET).max(1);

    // Snapshot-wide corpus stats (N, L̄) for the N-anchored scoring path (energy/floor/stop/null).
    // Read once for the whole batch from this snapshot's rolling counters, so every query sees the
    // same N/avgdl (batch == serial). Hoisted ABOVE selection because the M4 stop's energy/floored
    // inputs (per gram, fed to `select`) are N-anchored. `matches_batch` ignores avgdl;
    // `CandidateStream` exposes it.
    let (seg_count, seg_len_sum) = schema::read_seg_stats(conn, ns)?;
    let n_segments = seg_count.max(0) as u64;
    let avgdl = if seg_count > 0 {
        seg_len_sum as f64 / seg_count as f64
    } else {
        0.0
    };
    // The contamination floor `df_min` is `N`/`ν`-constant, so compute it once for the whole batch
    // and thread it into the per-gram energy/floored flag (no `powf` per gram); reinforces
    // `batch == serial`.
    let df_min_batch = df_min(n_segments as f64, nu);

    // The knob-derived selection parameters shared by every query; the work budget `C` is filled
    // in **per query** below (v0.5) — it pools the query's own class stats, so it cannot be a
    // batch constant without breaking `batch == serial`.
    let base_sel = SelectParams {
        min_shared,
        typo_damage: TYPO_DAMAGE,
        df_budget: None,
        c_margin,
        k_target,
        n_segments,
        sigma: index.sigma,
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

    // === v0.4/M5 rank-views (derivation §8) =======================================================
    // Partition each query's DUAL-ORDER grams into a PRIMARY rank-view (every script's primary
    // order — Latin trigram, CJK bigram) and a SECONDARY one (every script's one-shorter order). A
    // query runs PRIMARY-ONLY unless a present script is STARVED — too few/too weak primary grams,
    // or none at all (a query too short to produce the primary, the structural fallback) — in which
    // case the secondary view also runs and the two are RRF-fused (§8/§12). The single-view path is
    // exactly the M1–M4 pipeline, so clean queries pay nothing for §8.
    //
    // Everything class-derived is **per query** (v0.5): the query's own snapshot view (`subset`),
    // the derived work budget `C` (§5/§7 — the corpus-derived Lagrangian dual of the §5 stop,
    // pooling ONLY this query's classes; a caller-supplied `df_budget` overrides it, else `None`/
    // unbounded on a degenerate corpus), and selection/starvation/ΔH — all pure functions of THIS
    // query's grams + the shared per-batch snapshot ⇒ `batch == serial`.
    let planned_sel: Vec<ViewSel<T::Token>> = query_tokens
        .iter()
        .zip(&q_classes)
        .map(|(q, (actual, with_siblings))| {
            let q_snap = class_snap.subset(with_siblings, actual);
            let derived_c = derived_budget(n_segments, k_target, index.sigma, &q_snap);
            let sel_params = SelectParams {
                df_budget: opts.df_budget.or(derived_c),
                ..base_sel
            };
            plan_views(
                q,
                &resolve,
                &index.tokenizer,
                &q_snap,
                sel_params,
                df_min_batch,
                n_segments,
                kappa,
                nu,
            )
        })
        .collect();

    // One effective-postings read over the union of every query's every view's selected ids.
    let sel_ids: Vec<TermId> = planned_sel
        .iter()
        .flat_map(|vs| vs.views.iter().flatten())
        .filter_map(|tok| resolve(tok).map(|(id, _)| id))
        .collect::<BTreeSet<TermId>>()
        .into_iter()
        .collect();
    let postings_map = postings::effective_postings(conn, ns, &sel_ids)?;

    // §7/§12 quantization guard: `Δ < 2·E_floored` keeps `round(E_floored/Δ) ≥ 1`, so a floored
    // gram never quantizes to `0` and drops out of the bit-sliced union. It is satisfiable only
    // once the corpus is large enough for `E_floored` to clear `Δ/2`; on a handful-of-segments
    // corpus `E_floored` shrinks (negative for `N ≲ 4` at the defaults) and the guard cannot hold.
    // That regime is recall-safe regardless — the §7 count-only union (`score_union`) recovers a
    // floored gram that quantizes to weight 0 (v0.4 dropped the engine clamp), so it still never
    // vanishes; only its rarity ordering against other floored grams collapses, which a tiny corpus
    // does not need. So the `debug_assert` fires only where
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
    // cardinality-independent (the flatness property). Asserted per view below.
    let wq_ceiling = quantize_energy(e_max(n_segments as f64, nu), delta);
    // The count credit `μ = max(0, logit σ)` (derivation §3/§9). `σ` is the index-level corpus
    // constant (sanitized at open), so this is one value for the whole batch — read once, never a
    // per-batch aggregate (batch == serial). The per-query §9 concentration cap is applied below.
    let mu = count_credit(index.sigma);
    let batch = BatchConsts {
        df_min_batch,
        n_segments,
        avgdl,
        kappa,
        delta,
        mu,
        wq_ceiling,
        min_shared,
    };

    // Build one `QueryPlan` per rank-view, wrapped in a `PlannedQuery` (the RRF fusion unit).
    let mut plans = Vec::with_capacity(queries.len());
    for vs in &planned_sel {
        let mut view_plans = Vec::with_capacity(vs.views.len());
        let mut fused_selected: Vec<String> = Vec::new();
        let mut seen: FxHashSet<String> = FxHashSet::default();
        for selected in &vs.views {
            let plan = build_view_plan::<T>(selected, &postings_map, &resolve, &batch);
            for s in &plan.selected_strings {
                if seen.insert(s.clone()) {
                    fused_selected.push(s.clone());
                }
            }
            view_plans.push(plan);
        }
        plans.push(PlannedQuery {
            views: view_plans,
            view_weights: vs.weights.clone(),
            fused_selected,
            n_segments,
            avgdl,
        });
    }
    Ok(plans)
}

/// The per-batch scoring constants shared by every rank-view's [`QueryPlan`] (read once per batch
/// from the snapshot, so they are identical for every query ⇒ `batch == serial`).
#[derive(Clone, Copy)]
struct BatchConsts {
    df_min_batch: f64,
    n_segments: u64,
    avgdl: f64,
    kappa: f64,
    delta: f64,
    mu: f64,
    wq_ceiling: u32,
    min_shared: u32,
}

/// One query's selected tokens per rank-view (1 = primary-only, 2 = primary + secondary), plus the
/// parallel RRF fusion weights (derivation §8). An empty `views` is a query with no in-corpus gram
/// in any view (it scores empty).
struct ViewSel<Tk> {
    views: Vec<Vec<Tk>>,
    weights: Vec<f64>,
}

/// Partition one query's dual-order grams into rank-views and select each (derivation §8/§12).
///
/// Returns 1 view (primary-only) for a clean query — every present script has enough in-corpus
/// primary grams to corroborate — or 2 (primary + secondary) when any present script is **starved**:
/// fewer than `ν` in-corpus primary grams (a too-short/sparse query), or the budget pruned most of
/// its primary energy (`collected_energy_far_below`, §5/§12). The §12 fallback drops a view that
/// has no in-corpus gram (`PRIMARY → SECONDARY → empty`), so a too-short query (a 2-char Latin / a
/// 1-char CJK run, which produce no primary gram) runs the secondary order alone. Pure function of
/// the query's grams + the shared snapshot ⇒ `batch == serial`.
#[allow(clippy::too_many_arguments)]
fn plan_views<T: Tokenizer>(
    q: &[(T::Token, u32)],
    resolve: &impl Fn(&T::Token) -> Option<(TermId, i64)>,
    tokenizer: &T,
    class_snap: &ClassSnap,
    sel_params: SelectParams,
    df_min_batch: f64,
    n_segments: u64,
    kappa: f64,
    nu: f64,
) -> ViewSel<T::Token> {
    let mut primary_rows: Vec<GramRow<T::Token>> = Vec::new();
    let mut secondary_rows: Vec<GramRow<T::Token>> = Vec::new();
    let mut present_scripts: FxHashSet<u8> = FxHashSet::default();
    // Per-script starvation inputs over the PRIMARY order: how many primary grams the query
    // *produced* (df irrelevant — the too-short test), how many are *in corpus* (df > 0 — the
    // corroboration test), and their full energy (vs the collected energy after pruning, below).
    let mut produced_primary_count: FxHashMap<u8, u32> = FxHashMap::default();
    let mut primary_present_count: FxHashMap<u8, u32> = FxHashMap::default();
    let mut full_primary_e: FxHashMap<u8, f64> = FxHashMap::default();
    // Word-granular coverage for the `Common` structural rule (v0.5, see `starved_scripts`):
    // which query words produced ≥ 1 primary-order gram, and which words own a `Common`-class
    // gram. A `Common` gram is usually an *interior* fragment of a strong-script run (digits
    // inherit the run they sit in — `"12"` inside `ab12cd`), whose word IS primary-covered.
    let mut word_has_primary: FxHashSet<u32> = FxHashSet::default();
    let mut common_words: FxHashSet<u32> = FxHashSet::default();
    for (tok, word) in q {
        // An unencodable token (over the 3-codepoint ceiling / NUL) has no term and rides along
        // absent (df 0); `Common` is the `script_of`-consistent class for "no strong script".
        let class = tok
            .term()
            .map(|t| t.class())
            .unwrap_or(crate::term::COMMON_CLASS);
        let df = resolve(tok).map_or(0, |(_, df)| df);
        let order = tok.borrow().chars().count() as u8;
        let energy = energy_with_floor(df.max(0) as f64, df_min_batch, n_segments as f64, kappa);
        let floored = (df.max(0) as f64) <= df_min_batch;
        present_scripts.insert(class);
        if class == crate::term::COMMON_CLASS {
            common_words.insert(*word);
        }
        // A gram is SECONDARY iff its order is one shorter than its script's primary order (the
        // tokenizer owns the per-script policy; a single-order tokenizer returns `u8::MAX`, so no
        // gram is ever secondary and the secondary view never forms).
        let po = tokenizer.primary_order(class);
        let is_secondary = po != u8::MAX && order != 0 && order + 1 == po;
        let row = GramRow {
            token: tok.clone(),
            df,
            class,
            order,
            word: *word,
            energy,
            floored,
        };
        if is_secondary {
            secondary_rows.push(row);
        } else {
            *produced_primary_count.entry(class).or_insert(0) += 1;
            word_has_primary.insert(*word);
            if df > 0 {
                *primary_present_count.entry(class).or_insert(0) += 1;
                *full_primary_e.entry(class).or_insert(0.0) += energy.max(0.0);
            }
            primary_rows.push(row);
        }
    }

    let primary_selected = select(&primary_rows, sel_params, class_snap);

    // Collected primary energy over the SELECTED present primary grams (the §12 reserve diagnostic).
    let sel_primary: FxHashSet<&T::Token> = primary_selected.iter().collect();
    let mut collected_primary_e: FxHashMap<u8, f64> = FxHashMap::default();
    for r in &primary_rows {
        if r.df > 0 && sel_primary.contains(&r.token) {
            *collected_primary_e.entry(r.class).or_insert(0.0) += r.energy.max(0.0);
        }
    }
    // starved(s): bring in the secondary view when EITHER
    //   • STRUCTURAL — the script produced no primary gram at all (the query is too short to make
    //     one: a 2-char Latin / 1-char CJK run), so the shorter order is the only signal. For the
    //     `Common` class the trigger is **word-granular** (v0.5): `Common` grams are usually
    //     interior fragments of a strong-script run (digits inherit the run — the `"12"` bigram
    //     inside `ab12cd` classes `Common` while every primary trigram classes Latin), so "the
    //     class produced no primary gram" is spuriously true for almost any digit-bearing clean
    //     query. `Common` is structurally starved only when some `Common` gram's *word* produced
    //     no primary gram at all (a standalone digit/symbol word like the `12` in `hello 12`,
    //     whose content genuinely has no primary-order signal) — an interior fragment whose word
    //     is primary-covered never trips a spurious secondary view; OR
    //   • CORROBORATIVE — the script has ≥1 in-corpus primary gram but it is too thin to stand
    //     alone (`< ν` distinct in-corpus primary grams, or the budget pruned most of the primary
    //     energy — `collected_energy_far_below`, §5/§12), so the shorter order corroborates.
    // It deliberately does NOT fire for a query that PRODUCED primary grams that are all absent
    // (df = 0) — a query with no in-corpus primary overlap stays a no-match rather than fuzzily
    // falling to the far-less-selective bigram layer (preserving the no-match precision contract; a
    // narrowing of §12's literal while-loop, which would fall PRIMARY→SECONDARY on any all-absent
    // primary view). The §8 structural fallback (too-short queries) and §5 corroboration (weak but
    // present primary) are both preserved.
    // The PER-SCRIPT starved set (derivation §8/§12): the secondary view pools the one-shorter
    // order of each STARVED script ONLY — "a rich script with a primary gram omits its secondary."
    // Gating the whole query on one bool would leak a rich script's bigram coincidences into a
    // mixed-script query (recall-safe via RRF missing=omit, but under a tight `limit` a leaked
    // candidate could evict a genuine minority-script match), so the set is computed per script.
    let starved_scripts: FxHashSet<u8> = present_scripts
        .iter()
        .copied()
        .filter(|&s| {
            let produced = produced_primary_count.get(&s).copied().unwrap_or(0);
            let present = primary_present_count.get(&s).copied().unwrap_or(0);
            let fe = full_primary_e.get(&s).copied().unwrap_or(0.0);
            let ce = collected_primary_e.get(&s).copied().unwrap_or(0.0);
            let structural = if s == crate::term::COMMON_CLASS {
                // Word-granular (see above): an interior digit fragment is primary-covered.
                common_words.iter().any(|w| !word_has_primary.contains(w))
            } else {
                produced == 0
            };
            let corroborative = present >= 1
                && ((present as f64) < nu || (fe > 0.0 && ce < STARVED_ENERGY_RATIO * fe));
            structural || corroborative
        })
        .collect();
    drop(sel_primary); // release the borrow of `primary_selected` before it is moved below

    let primary_present = primary_rows.iter().any(|r| r.df > 0);
    let mut views: Vec<Vec<T::Token>> = Vec::new();
    if starved_scripts.is_empty() {
        views.push(primary_selected);
    } else {
        // §12 rank_views = [PRIMARY, SECONDARY], with the PRIMARY → SECONDARY → empty fallback: a
        // view with no in-corpus gram is dropped (so a too-short query runs the secondary alone).
        // The secondary pools ONLY the starved scripts' grams — a rich (non-starved) script's
        // secondary order is omitted, so its bigram coincidences never enter a mixed-script query.
        let secondary_rows: Vec<GramRow<T::Token>> = secondary_rows
            .into_iter()
            .filter(|r| starved_scripts.contains(&r.class))
            .collect();
        let secondary_present = secondary_rows.iter().any(|r| r.df > 0);
        if primary_present {
            views.push(primary_selected);
        }
        if secondary_present {
            views.push(select(&secondary_rows, sel_params, class_snap));
        }
    }

    let weights = if views.len() == 2 {
        // ΔH over the STARVED scripts only — those are the ones contributing to the secondary view.
        view_weights_from_dh(&starved_scripts, tokenizer, class_snap)
    } else {
        vec![1.0; views.len()]
    };
    ViewSel { views, weights }
}

/// The RRF fusion weights `[w_primary, w_secondary]` from the per-`(script, order)` vocabulary-
/// complexity gap `ΔH` (derivation §8). `ΔH(s) = ln V_primary − ln V_secondary` with `V` the
/// distinct-gram count of class `(s, order)` (the `ln V` max-entropy proxy), averaged over the
/// (starved) scripts that have both orders populated; a richer primary inventory (`ΔH > 0`, the
/// normal case) earns the primary view more weight via the fixed linear map `w_primary =
/// clamp(0.5 + RRF_GAMMA·ΔH, 0.1, 0.9)`. Equal weights when `ΔH` is unavailable (no script has both
/// vocabularies populated). Pure function of the given scripts + the shared snapshot. Passed the
/// **starved** scripts (those contributing the secondary view), per §8's per-script combination.
fn view_weights_from_dh<T: Tokenizer>(
    scripts: &FxHashSet<u8>,
    tokenizer: &T,
    snap: &ClassSnap,
) -> Vec<f64> {
    let mut sum = 0.0;
    let mut n = 0u32;
    for &s in scripts {
        let po = tokenizer.primary_order(s);
        if po == u8::MAX || po < 1 {
            continue;
        }
        let vp = snap.vocab(s, po);
        let vs = snap.vocab(s, po - 1);
        if vp > 0 && vs > 0 {
            sum += (vp as f64).ln() - (vs as f64).ln();
            n += 1;
        }
    }
    let w_primary = if n > 0 {
        (0.5 + RRF_GAMMA * (sum / n as f64)).clamp(0.1, 0.9)
    } else {
        0.5
    };
    vec![w_primary, 1.0 - w_primary]
}

/// Build one rank-view's [`QueryPlan`] from its `selected` tokens: load the present postings, compute
/// the M1 logit-idf energy planes, the M2 floored flags + §9-capped per-order count credit, and the
/// M3 length-null split — the whole M1–M4 scoring pipeline, unchanged, per view (derivation §2–§9).
fn build_view_plan<T: Tokenizer>(
    selected: &[T::Token],
    postings_map: &FxHashMap<TermId, Bitmap>,
    resolve: &impl Fn(&T::Token) -> Option<(TermId, i64)>,
    batch: &BatchConsts,
) -> QueryPlan {
    let BatchConsts {
        df_min_batch,
        n_segments,
        avgdl,
        kappa,
        delta,
        mu,
        wq_ceiling,
        min_shared,
    } = *batch;
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
    // (codepoint count), then the §9-capped per-order count credit. The cap ranges over ALL of P's
    // energies (the literal §12 reading — a floored `E_max` rare gram anchors `E_top`, protecting
    // it; see `concentration_cap`); `σ` is index-level and uniform across orders query-side, so
    // every present non-floored order maps to the same capped `μ`. The per-order bucketing is kept
    // structural for a future doc-side `ρ = σ(1−ε)^n`. All of this is a pure function of THIS
    // view's grams + the shared (σ, N, ν, κ, Δ) snapshot ⇒ `batch == serial`.
    let present_floored: Vec<bool> = present_dfs
        .iter()
        .map(|&df| (df as f64) <= df_min_batch)
        .collect();
    let present_orders: Vec<u8> = present_tokens
        .iter()
        .map(|t| t.chars().count() as u8)
        .collect();
    let mu_capped = match concentration_cap(&energies, &present_floored) {
        Some(cap) => mu.min(cap),
        None => mu,
    };
    let mut mu_by_order: FxHashMap<u8, f64> = FxHashMap::default();
    for (&order, &floored) in present_orders.iter().zip(&present_floored) {
        if !floored {
            mu_by_order.insert(order, mu_capped);
        }
    }
    // v0.4 M3 (§6/§7): precompute the length-null split and the early-stop ceiling, all pure
    // functions of THIS view's grams + the (σ, N, ν, κ, Δ) snapshot (⇒ batch == serial). The
    // per-gram null `weight(g) = wq·Δ + (μ if non-floored)` mirrors the accumulator term for
    // term (quantized energy, capped μ on non-floored grams only). Rare grams (p < P_LINEAR)
    // fold into the separable `K_rare`; commons (p ≥ P_LINEAR) keep their saturating term.
    let n_f = (n_segments as f64).max(1.0);
    let mut k_rare = 0.0;
    let mut null_commons: Vec<(f64, f64)> = Vec::new();
    let mut cred_max = 0.0;
    for i in 0..present_postings.len() {
        let mu_g = if present_floored[i] {
            0.0
        } else {
            mu_by_order.get(&present_orders[i]).copied().unwrap_or(0.0)
        };
        cred_max += mu_g;
        let weight_g = weights[i] as f64 * delta + mu_g;
        let p_g = ((present_dfs[i] as f64) / n_f).clamp(0.0, 1.0);
        if p_g < P_LINEAR {
            k_rare += p_g * weight_g;
        } else {
            null_commons.push((p_g, weight_g));
        }
    }
    // `present_weights` retains the quantized energy weights (the engine takes `weights` by
    // value): they identify the weight-0 postings whose union recovers count-only candidates
    // (§7) and feed the null's `weight(g)` above.
    let present_weights = weights.clone();
    let counter = Counter::build_weighted(&present_postings, weights, min_shared);
    QueryPlan {
        counter,
        present_tokens,
        present_postings,
        present_dfs,
        present_floored,
        present_orders,
        present_weights,
        mu_by_order,
        delta,
        k_rare,
        null_commons,
        cred_max,
        selected_strings,
        avgdl,
    }
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
    /// One batched provenance(+filter) query over a chunk's seg ids: `(key, label, len)` per id
    /// that exists and passes the filter. `len` is the segment's distinct-gram count `L_d`
    /// (derivation §0/§6), folded into this same read so the §6 length null needs no second
    /// round-trip. Fragment textually first, the candidate-scope param last (`?{N+1}`), so the
    /// caller's `?1..?N` (numbered or anonymous) never collide with the scope.
    fn lookup(&self, seg_ids: &[u32]) -> Result<FxHashMap<u32, (Key, String, i64)>> {
        let mut out = FxHashMap::with_capacity_and_hasher(seg_ids.len(), Default::default());
        if seg_ids.is_empty() {
            return Ok(out);
        }
        let arr: Rc<Vec<Value>> =
            Rc::new(seg_ids.iter().map(|&i| Value::Integer(i as i64)).collect());
        let n = self.filter.map_or(0, |f| f.params.len());
        let sql = match self.filter {
            Some(f) => format!(
                "SELECT id, key, label, len FROM {seg} WHERE ({frag}) AND id IN rarray(?{scope})",
                seg = self.ns.seg(),
                frag = f.fragment,
                scope = n + 1,
            ),
            None => format!(
                "SELECT id, key, label, len FROM {seg} WHERE id IN rarray(?1)",
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
            let len: i64 = r.get(3)?;
            out.insert(
                id as u32,
                (Key::from_value(self.key_shape, kv)?, label, len),
            );
        }
        Ok(out)
    }
}

/// Fold one candidate into the per-result best-float map, keeping the **max-`corrected_score`**
/// candidate per [`ResultKey`] (derivation §7's float top-k). Under the default per-segment
/// granularity the fold is trivial (each segment is scored once per view); under
/// [`Collapse::Key`] it keeps the best segment per caller key. `corrected` is precomputed
/// (cheap, no allocation), so the [`Candidate`] — which clones `key`/`label` — is materialized
/// **only on an insert or a strict win**, never for a loser (the engine-review (4) cost note).
/// Returns whether the candidate was accepted (inserted or strictly won) — the eager path feeds
/// accepted events to its [`TopK`] tracker. Deterministic only via the later sort tiebreak; this
/// map itself is order-free.
#[allow(clippy::too_many_arguments)]
fn upsert_best(
    best: &mut FxHashMap<ResultKey, (Candidate, f64)>,
    rk: &ResultKey,
    key: &Key,
    label: &str,
    seg_id: u32,
    score: u32,
    overlap: u32,
    corrected: f64,
    energy: f64,
    count: f64,
    length: f64,
) -> bool {
    let build = || Candidate {
        key: key.clone(),
        label: label.to_string(),
        seg_id,
        score,
        overlap,
        corrected_score: corrected,
        energy,
        count,
        length,
    };
    match best.get_mut(rk) {
        Some(slot) => {
            if corrected > slot.1 {
                *slot = (build(), corrected);
                true
            } else {
                false
            }
        }
        None => {
            best.insert(rk.clone(), (build(), corrected));
            true
        }
    }
}

/// An `f64` with a total order (`f64::total_cmp`), for the [`TopK`] heap.
#[derive(Clone, Copy, PartialEq)]
struct TotalF64(f64);
impl Eq for TotalF64 {}
impl PartialOrd for TotalF64 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for TotalF64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// Incremental top-`k` tracker over per-result best floats — the eager early-stop's `kth_best`
/// ceiling (derivation §7). v0.5: replaces the per-qualifying-chunk `O(n)` `kth_largest`
/// recompute (worst `O(union²/CHUNK)` on a deep-`k` query) with amortized `O(log n)` per event
/// and an amortized-`O(1)` [`kth_best`](Self::kth_best) read; no per-check allocation.
///
/// Result keys ([`ResultKey`] — a segment id, or a caller key under [`Collapse::Key`]) are
/// interned to dense `u32` slots (one clone per distinct result). The scheme is a
/// **lazy min-heap over the current top set**: `in_top` holds the top-`k` slots; the heap holds
/// every `(value, slot)` event accepted for a top slot, and an entry is stale iff its slot left
/// the top set or its slot's value moved past it — both detected (and popped) on peek. Soundness
/// leans on the caller's invariant that a result's value only ever **increases** (the per-result
/// best is max-folded), so a slot's older entries are always strictly below its current value.
struct TopK {
    k: usize,
    /// result key → dense slot id.
    slots: FxHashMap<ResultKey, u32>,
    /// Current value per slot (the staleness oracle).
    vals: Vec<f64>,
    /// The slots currently in the top-`k` set (`len == min(k, #keys)`).
    in_top: FxHashSet<u32>,
    /// Min-heap of `(value, slot)` events for top slots; lazily invalidated.
    heap: std::collections::BinaryHeap<std::cmp::Reverse<(TotalF64, u32)>>,
}

impl TopK {
    fn new(k: usize) -> Self {
        TopK {
            k,
            slots: FxHashMap::default(),
            vals: Vec::new(),
            in_top: FxHashSet::default(),
            heap: std::collections::BinaryHeap::new(),
        }
    }

    /// Fold one accepted per-result event (an insert, or a strictly-improved best) into the
    /// tracker.
    fn observe(&mut self, rk: &ResultKey, v: f64) {
        let slot = match self.slots.get(rk) {
            Some(&s) => {
                debug_assert!(v > self.vals[s as usize], "per-result bests only increase");
                self.vals[s as usize] = v;
                s
            }
            None => {
                let s = self.vals.len() as u32;
                self.slots.insert(rk.clone(), s);
                self.vals.push(v);
                s
            }
        };
        if self.in_top.contains(&slot) {
            // An improved top member: push the new live entry (the old one is now stale).
            self.heap.push(std::cmp::Reverse((TotalF64(v), slot)));
        } else if self.in_top.len() < self.k {
            self.in_top.insert(slot);
            self.heap.push(std::cmp::Reverse((TotalF64(v), slot)));
        } else if self.peek_valid().is_some_and(|(min_v, _)| v > min_v) {
            // Promote: displace the top set's current minimum (its live entry is the peek).
            let (_, min_slot) = self.peek_valid().expect("just checked");
            self.heap.pop();
            self.in_top.remove(&min_slot);
            self.in_top.insert(slot);
            self.heap.push(std::cmp::Reverse((TotalF64(v), slot)));
        }
        // else: v ≤ the k-th best — the top set is unchanged; the event needs no entry (the key
        // re-enters consideration only via a future, larger event).
    }

    /// The current `k`-th largest per-result best, or `None` while fewer than `k` are tracked.
    fn kth_best(&mut self) -> Option<f64> {
        if self.in_top.len() < self.k {
            return None;
        }
        self.peek_valid().map(|(v, _)| v)
    }

    /// The heap's minimum **live** entry (popping stale ones): the smallest value in the top set.
    fn peek_valid(&mut self) -> Option<(f64, u32)> {
        while let Some(&std::cmp::Reverse((TotalF64(v), slot))) = self.heap.peek() {
            let live = self.in_top.contains(&slot) && self.vals[slot as usize] == v;
            if live {
                return Some((v, slot));
            }
            self.heap.pop();
        }
        None
    }
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
                // The §10 score + components (v0.5): the cross-query-comparable nat-scale
                // magnitude, from the candidate's governing rank-view.
                score: c.nat_score(),
                energy: c.energy(),
                count: c.count(),
                length: c.length(),
            }
        })
        .collect())
}

/// The §6/§7/§12 **float post-pass over the bounded candidate union** — the single scoring core
/// behind both front doors (the G2 reshape). Walks the engine best-first by integer energy,
/// recovers count-only candidates, subtracts the §6 length null, folds candidates at the search's
/// retrieval granularity (`collapse`: per **segment** by default; per-key best under
/// [`Collapse::Key`], keeping the max [`corrected_score`](Candidate::corrected_score)), and
/// returns them sorted best-first. `limit = Some(k)` is the **eager** path (over-sample
/// early-stop, truncated to `k`); `limit = None` is the **lazy** path (the full sorted union).
/// With the same plan, `Some(k)` yields exactly the `k`-prefix of `None` — so
/// `collect_matches(k) == matches(k)` (derivation §7: top-k strictly *after* the floats).
///
/// **The candidate set is invariant** (the M3 spine): it is `{seg : raw_overlap ≥ floor}`,
/// unchanged from M2 — M3 only rescores and reshapes, never shrinks it, so recall is preserved by
/// construction. The set is recovered in two parts (derivation §7's `keys(E_acc) ∪ keys(cred_acc)`):
/// - **the walk** yields every seg with positive bit-sliced energy, best-first by energy `c`
///   (v0.5: the engine is energy-only — this layer applies the `raw_overlap ≥ floor` gate in the
///   same fused `contains` sweep that computes the §3/§9 credit, before any provenance SQL); its
///   `E_acc = s.score`.
/// - **count-only recovery** (`U_zero`): a seg matching *only* weight-0 (common) grams has
///   `E_acc = 0`, so the walk never surfaces it. `U_zero = ⋃ {posting : weight = 0}` minus the
///   segs the walk already saw; each survivor with `raw_overlap ≥ floor` is a count-only candidate
///   (`E_acc = 0`). Enumerated **only when the walk fully exhausted** (so `seen` is complete ⇒ no
///   aliasing) — and **skipped after an early-stop** (their float `≤ cred_max ≤ bound < kth_best`,
///   provably excluded; see below). Under `Δ < 2·E_floored` a floored gram keeps weight ≥ 1, so a
///   weight-0 gram is always non-floored — a count-only candidate always carries some credit.
///
/// **The eager over-sample early-stop** (derivation §7): after each chunk, with `≥ limit` distinct
/// keys collected, peek the next candidate's energy `c_next`. Any un-yielded walk candidate has
/// energy `≤ c_next`, so its float `≤ c_next·Δ + cred_max` (the null `≥ 0` is dropped from the
/// bound — valid, it only lowers scores). Stop once that bound `< kth_best_float`. The cheap gate
/// `bound < max_float_seen` (since `kth_best ≤ max_float_seen`) guards the `O(n)` `kth_largest`,
/// so the **single-bucket degenerate case** (all grams one weight ⇒ every candidate shares `c` ⇒
/// `bound ≥ max_float_seen`) never pays it and drains the bucket fully — correct, there is no sound
/// integer within-bucket discriminator.
///
/// The sort tiebreak (corrected desc → integer energy desc → seg id asc) is a deterministic total
/// order, so `batch == serial` and the thrash oracle stay reproducible.
fn score_union(
    prov: &Provenance<'_>,
    plan: &QueryPlan,
    limit: Option<usize>,
    collapse: Collapse,
) -> Result<Vec<Candidate>> {
    if limit == Some(0) {
        return Ok(Vec::new());
    }
    let counter = &plan.counter;
    let floor = counter.floor();
    let delta = plan.delta;
    let cred_max = plan.cred_max;
    let avgdl = plan.avgdl;

    let mut walk = counter.walk();
    // Max-corrected candidate per result key (segment, or caller key under Collapse::Key),
    // accumulated over the union.
    let mut best: FxHashMap<ResultKey, (Candidate, f64)> = FxHashMap::default();
    // The eager path's incremental `kth_best` tracker (fed by accepted `upsert_best` events).
    let mut topk = limit.map(TopK::new);
    // Seg ids the walk yielded — completes only when the walk exhausts; gates the U_zero pass so a
    // seg matching both weight-0 and positive grams is not double-recovered.
    let mut seen_segs: FxHashSet<u32> = FxHashSet::default();
    // Upper bound on `kth_best` (kth_best ≤ max distinct float ≤ max float seen); cheap stop gate.
    let mut max_float_seen = f64::NEG_INFINITY;
    // One-item lookahead carried across chunks: the peeked next candidate, kept for the next chunk
    // when the early-stop did not fire.
    let mut pending: Option<Scored> = None;
    let mut walk_done = false;
    let mut early_stopped = false;

    loop {
        let mut scored = Vec::with_capacity(CHUNK);
        if let Some(s) = pending.take() {
            scored.push(s);
        }
        while scored.len() < CHUNK {
            match counter.advance(&mut walk) {
                Some(s) => scored.push(s),
                None => {
                    walk_done = true;
                    break;
                }
            }
        }
        if !scored.is_empty() {
            for s in &scored {
                seen_segs.insert(s.id);
            }
            // Fused per-id pass (v0.5): raw overlap + count credit in ONE `contains` sweep, and
            // the raw-overlap floor gate — the engine yields energy-ranked ids only (it retains
            // no postings and does not gate). Gated BEFORE the provenance SQL, so a sub-floor id
            // never costs a round-trip. (A gated-out id stays in `seen_segs`: it has positive
            // energy, so the count-only recovery must not resurrect it either — and its raw
            // overlap fails the same floor there anyway.)
            let gated: Vec<(Scored, u32, f64)> = scored
                .iter()
                .map(|&s| {
                    let (overlap, credit) = plan.overlap_and_credit(s.id);
                    (s, overlap, credit)
                })
                .filter(|&(_, overlap, _)| overlap >= floor)
                .collect();
            let seg_ids: Vec<u32> = gated.iter().map(|&(s, _, _)| s.id).collect();
            let found = prov.lookup(&seg_ids)?;
            for &(s, overlap, credit) in &gated {
                if let Some((key, label, len)) = found.get(&s.id) {
                    let (corrected, energy, count, length) =
                        plan.corrected_parts(s.score, credit, rel_len(Some(*len), avgdl));
                    if corrected > max_float_seen {
                        max_float_seen = corrected;
                    }
                    let rk = result_key(collapse, key, s.id);
                    if upsert_best(
                        &mut best, &rk, key, label, s.id, s.score, overlap, corrected, energy,
                        count, length,
                    ) {
                        if let Some(t) = topk.as_mut() {
                            t.observe(&rk, corrected);
                        }
                    }
                }
            }
        }
        if walk_done {
            break;
        }
        // Eager over-sample early-stop: peek one candidate and test the ceiling.
        if let Some(lim) = limit {
            if best.len() >= lim {
                match counter.advance(&mut walk) {
                    Some(s) => {
                        let bound = s.score as f64 * delta + cred_max;
                        // Cheap gate first (kth_best ≤ max_float_seen): only then consult the
                        // incremental tracker (v0.5 — replaces the per-qualifying-chunk `O(n)`
                        // `kth_largest` recompute, worst `O(union²/CHUNK)` on a deep-k query,
                        // with an amortized-`O(1)` read; no per-check allocation).
                        if bound < max_float_seen {
                            if let Some(kth) = topk.as_mut().and_then(TopK::kth_best) {
                                if bound < kth {
                                    early_stopped = true;
                                    break;
                                }
                            }
                        }
                        pending = Some(s);
                    }
                    None => {
                        walk_done = true;
                        break;
                    }
                }
            }
        }
    }

    // Count-only recovery (derivation §7) — only on a full drain, never after an early-stop.
    if walk_done && !early_stopped {
        let mut u_zero = Bitmap::new();
        for (i, &w) in plan.present_weights.iter().enumerate() {
            if w == 0 {
                u_zero.or_inplace(&plan.present_postings[i]);
            }
        }
        let cand_ids: Vec<u32> = u_zero.iter().filter(|id| !seen_segs.contains(id)).collect();
        for chunk in cand_ids.chunks(CHUNK) {
            // raw_overlap + credit in one fused pass; gate on the floor before any SQL.
            let keep: Vec<(u32, u32, f64)> = chunk
                .iter()
                .map(|&id| {
                    let (overlap, credit) = plan.overlap_and_credit(id);
                    (id, overlap, credit)
                })
                .filter(|&(_, overlap, _)| overlap >= floor)
                .collect();
            if keep.is_empty() {
                continue;
            }
            let seg_ids: Vec<u32> = keep.iter().map(|&(id, _, _)| id).collect();
            let found = prov.lookup(&seg_ids)?;
            for (id, overlap, credit) in keep {
                if let Some((key, label, len)) = found.get(&id) {
                    // E_acc = 0 (matched only weight-0 grams).
                    let (corrected, energy, count, length) =
                        plan.corrected_parts(0, credit, rel_len(Some(*len), avgdl));
                    let rk = result_key(collapse, key, id);
                    upsert_best(
                        &mut best, &rk, key, label, id, 0, overlap, corrected, energy, count,
                        length,
                    );
                }
            }
        }
    }

    let mut ranked: Vec<(Candidate, f64)> = best.into_values().collect();
    ranked.sort_by(|a, b| {
        b.1.total_cmp(&a.1)
            .then_with(|| b.0.score.cmp(&a.0.score))
            .then_with(|| a.0.seg_id.cmp(&b.0.seg_id))
    });
    if let Some(lim) = limit {
        ranked.truncate(lim);
    }
    Ok(ranked.into_iter().map(|(c, _)| c).collect())
}

/// Score a [`PlannedQuery`]'s rank-views (derivation §8/§12). The **single-view** (clean / not
/// starved) path is exactly [`score_union`] with the eager over-sample early-stop — clean queries
/// pay nothing for §8. A **two-view** (starved) query scores each view's *full* bounded union
/// (`limit = None`, since a rank requires the whole ordering) and RRF-fuses them ([`rrf_fuse`]).
/// `limit = Some(k)` truncates the final fused order to `k`; `None` returns the full fused order.
fn score_planned(
    prov: &Provenance<'_>,
    planned: &PlannedQuery,
    limit: Option<usize>,
    collapse: Collapse,
) -> Result<Vec<Candidate>> {
    match planned.views.as_slice() {
        [] => Ok(Vec::new()),
        [single] => score_union(prov, single, limit, collapse),
        views => {
            // Each view's full ranked union (rank = position, so the whole order is needed).
            let mut ranked_views = Vec::with_capacity(views.len());
            for v in views {
                ranked_views.push(score_union(prov, v, None, collapse)?);
            }
            Ok(rrf_fuse(
                &ranked_views,
                &planned.view_weights,
                limit,
                collapse,
            ))
        }
    }
}

/// Reciprocal-rank-fuse the per-view ranked candidate lists (derivation §8). `RRF(seg) = Σ_v w_v /
/// (k_RRF + rank_v)`, 1-based `rank_v` = the candidate's position in view `v`'s corrected-float
/// order, with **`missing = "omit"`**: a result absent from a view contributes nothing from that
/// view (it is *not* given a worst-rank penalty), so a seg surfaced by only one view keeps just
/// that view's contribution. RRF reads RANKS, not summed energy, so a contiguous match that ranks
/// well in both the trigram and bigram views is *not* additively over-credited by its sub-grams
/// (the §8 robustness pooling would break). Folds one candidate per [`ResultKey`] (the best-ranked
/// view's) at the search's granularity — per segment by default, per caller key under
/// [`Collapse::Key`] — reports the fused score as
/// [`corrected_score`](Candidate::corrected_score), and sorts best-first with a deterministic
/// tiebreak (fused desc → integer energy desc → seg id asc) ⇒ `batch == serial`.
fn rrf_fuse(
    views: &[Vec<Candidate>],
    weights: &[f64],
    limit: Option<usize>,
    collapse: Collapse,
) -> Vec<Candidate> {
    // result key -> (best-ranked candidate, fused RRF score, that best 1-based rank).
    let mut acc: FxHashMap<ResultKey, (Candidate, f64, usize)> = FxHashMap::default();
    for (vi, view) in views.iter().enumerate() {
        let w = weights.get(vi).copied().unwrap_or(1.0);
        for (i, c) in view.iter().enumerate() {
            let rank = i + 1; // 1-based
            let contrib = w / (K_RRF + rank as f64);
            let rk = result_key(collapse, c.key(), c.seg_id);
            match acc.get_mut(&rk) {
                Some(slot) => {
                    slot.1 += contrib;
                    if rank < slot.2 {
                        slot.0 = c.clone();
                        slot.2 = rank;
                    }
                }
                None => {
                    acc.insert(rk, (c.clone(), contrib, rank));
                }
            }
        }
    }
    let mut ranked: Vec<(Candidate, f64)> = acc
        .into_values()
        .map(|(mut c, score, _)| {
            // The fused path's ranking key is the RRF score; surface it as the corrected score so
            // `corrected_score()` stays "the value trifle sorted by" (the single-view path keeps the
            // §6/§7 float).
            c.corrected_score = score;
            (c, score)
        })
        .collect();
    ranked.sort_by(|a, b| {
        b.1.total_cmp(&a.1)
            .then_with(|| b.0.score.cmp(&a.0.score))
            .then_with(|| a.0.seg_id.cmp(&b.0.seg_id))
    });
    if let Some(l) = limit {
        ranked.truncate(l);
    }
    ranked.into_iter().map(|(c, _)| c).collect()
}

/// Eager: top-`limit` matches per query, all queries sharing one snapshot. The safe default
/// front door (`matches`/`matches_batch`). Scores each query's rank-views by the §6/§7 corrected
/// float (single view: the over-sample early-stop; two views: RRF fusion, see [`score_planned`]),
/// then hydrates exactly the kept rows.
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
    for planned in &plans {
        let kept = score_planned(&prov, planned, Some(limit), opts.collapse)?;
        out.push(hydrate_matches(
            &tx,
            ns,
            &index.tokenizer,
            &planned.fused_selected,
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
    let planned = match prepare(index, &conn, ns, &[query], opts) {
        Ok(mut plans) => plans.pop().expect("one planned query for one query"),
        Err(e) => {
            let _ = conn.execute_batch("ROLLBACK");
            return Err(e);
        }
    };
    // N / avgdl live on the planned query (computed once in `prepare` from this snapshot's rolling
    // counters); the accessors read them from there. A corpus-relative custom score must not
    // cross a snapshot boundary.
    Ok(CandidateStream {
        index,
        conn,
        planned,
        filter: opts.filter,
        collapse: opts.collapse,
        ready: VecDeque::new(),
        started: false,
        errored: false,
    })
}

/// A lazy, snapshot-pinned candidate cursor (the architectural spine). Owns a pooled connection
/// with a pinned read transaction **and** the engine [`Counter`]. **Fuses on the first error** (a
/// caller never gets a deceptively-complete prefix after a transient `Busy`).
///
/// **Ordering (the M3 G2 unification).** The stream ranks by the §6/§7 corrected float
/// ([`corrected_score`](Candidate::corrected_score)), *identically* to the eager
/// [`matches`](crate::Reader::matches)/[`matches_batch`](crate::Reader::matches_batch) front door —
/// so [`collect_matches(k)`](Self::collect_matches) returns exactly the same matches as
/// `matches(k)`. Because the corrected score is a post-pass float over the **whole** bounded
/// candidate union (count credit + length null + count-only recovery), the cursor is *not*
/// incrementally best-first: the first [`next`](Iterator::next) scores and sorts the full union
/// (caching it), then each call pops the next best. This drops v0.3's incremental best-first lazy
/// contract (per the G2 reshape) — the union is `O(C)`-bounded by selection, so a single pass is
/// affordable, and top-k-after-the-floats (derivation §7) is not expressible incrementally on the
/// integer buckets.
///
/// A live stream pins its WAL snapshot — keep it short-lived; do not park it. Drop releases the
/// snapshot.
pub struct CandidateStream<'a, T: Tokenizer> {
    index: &'a Index<T>,
    conn: ReadConn<'a>,
    planned: PlannedQuery,
    filter: Option<SqlFilter<'a>>,
    collapse: Collapse,
    /// The cached full sorted (and, for a starved query, RRF-fused) union, computed on the first
    /// [`next`](Iterator::next) and drained front-to-back thereafter.
    ready: VecDeque<Candidate>,
    /// Whether the one-shot union scoring has run (so an exhausted stream does not re-score).
    started: bool,
    errored: bool,
}

impl<T: Tokenizer> CandidateStream<'_, T> {
    /// Total live segments `N`, from **this search's** snapshot (not `stats()`).
    pub fn n_segments(&self) -> u64 {
        self.planned.n_segments
    }
    /// Mean **distinct-gram** segment length `L̄` on this snapshot — the §6 length null's
    /// denominator. `0.0` on an empty corpus. (v0.5: renamed from `avgdl`, a BM25-era name; the
    /// derivation's quantity is `L̄`, the mean distinct-gram count.)
    pub fn mean_segment_grams(&self) -> f64 {
        self.planned.avgdl
    }
    /// The selected tokens that have a posting, each with its document frequency `df` (no SQL —
    /// the postings are already in hand). Unions every rank-view (derivation §8); a token present
    /// in more than one view is reported once (its first view's `df`, which is identical across
    /// views — the same snapshot posting).
    pub fn present_terms(&self) -> impl Iterator<Item = (&str, u64)> {
        let mut seen: FxHashSet<&str> = FxHashSet::default();
        self.planned
            .views
            .iter()
            .flat_map(|p| p.present_tokens.iter().zip(&p.present_dfs))
            .filter_map(move |(t, df)| seen.insert(t.as_str()).then_some((t.as_str(), *df)))
    }
    /// Which selected tokens this candidate's segment actually contains, each with its `df` (no
    /// SQL). The inputs an IDF-sum-style custom reranker needs. Scans every rank-view's postings
    /// (a fused candidate may have matched in either view), deduped per token string.
    pub fn matched_terms<'c>(&'c self, c: &Candidate) -> impl Iterator<Item = (&'c str, u64)> + 'c {
        let seg_id = c.seg_id;
        let mut seen: FxHashSet<&str> = FxHashSet::default();
        self.planned
            .views
            .iter()
            .flat_map(|p| {
                p.present_tokens
                    .iter()
                    .zip(&p.present_postings)
                    .zip(&p.present_dfs)
            })
            .filter(move |((_, bm), _)| bm.contains(seg_id))
            .filter_map(move |((t, _), df)| seen.insert(t.as_str()).then_some((t.as_str(), *df)))
    }

    /// Hydrate text + span for exactly `kept` in ONE batched read (the terminal step). A
    /// pull-many/keep-few caller hydrates only what it kept. Pass candidates from **this** stream
    /// (seg ids are snapshot-specific).
    pub fn hydrate(&self, kept: &[Candidate]) -> Result<Vec<Match>> {
        hydrate_matches(
            &self.conn,
            self.index.store.namespace(),
            &self.index.tokenizer,
            &self.planned.fused_selected,
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
    /// Corrected-float order, folded at the stream's [`Collapse`] granularity, filtered. The
    /// first call scores and sorts the whole
    /// bounded union (caching it); each call then pops the next best. Fuses on the first `Err`.
    fn next(&mut self) -> Option<Result<Candidate>> {
        if self.errored {
            return None;
        }
        if !self.started {
            self.started = true;
            // Score the full union once (`limit = None`): count credit + §6 length null +
            // count-only recovery, sorted best-first — and, for a starved query, RRF-fused across
            // the rank-views (derivation §8). Borrows of `self` end with this block.
            let scored = {
                let prov = Provenance {
                    conn: &self.conn,
                    ns: self.index.store.namespace(),
                    key_shape: self.index.schema.key_shape(),
                    filter: self.filter.as_ref(),
                };
                score_planned(&prov, &self.planned, None, self.collapse)
            };
            match scored {
                Ok(v) => self.ready = VecDeque::from(v),
                Err(e) => {
                    self.errored = true;
                    return Some(Err(e));
                }
            }
        }
        self.ready.pop_front().map(Ok)
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
    fn energy_matches_the_exact_jeffreys_posterior_log_odds_by_hand() {
        let n = 10_000.0_f64;
        // df = 1000 (> df_min = 100): E = ln((N − df + κ)/(df + κ)) — the log-odds of the exact
        // Beta(κ,κ) posterior mean p̂ = (df + κ)/(N + 2κ) (v0.5; the pre-v0.5 −κ numerator was
        // the unnormalized approximation, undefined at df ≥ N − κ).
        approx(
            energy(1_000.0, n, NU, KAPPA),
            ((n - 1_000.0 + 0.5) / (1_000.0 + 0.5)).ln(),
        );
        // A gram at or below df_min is floored to df_min, so it carries E_floored exactly.
        approx(energy(100.0, n, NU, KAPPA), e_floored(n, NU, KAPPA)); // df == df_min boundary
        approx(energy(10.0, n, NU, KAPPA), e_floored(n, NU, KAPPA)); // df < df_min
    }

    #[test]
    fn energy_is_nonpositive_for_common_grams_and_finite_at_df_equals_n() {
        let n = 1_000.0;
        assert!(energy(500.0, n, NU, KAPPA) <= 1e-9); // p ≈ 0.5 ⇒ logit ≈ 0
        assert!(energy(900.0, n, NU, KAPPA) < 0.0); // p > 0.5 ⇒ negative
        // A ubiquitous gram (df = N) has FINITE energy under the exact posterior form (v0.5):
        // E = ln(κ/(N + κ)) — deeply negative, quantized to weight 0, but no −∞ special case.
        let ubiquitous = energy(1_000.0, n, NU, KAPPA);
        assert!(ubiquitous.is_finite(), "df = N is finite: {ubiquitous}");
        approx(ubiquitous, (0.5 / (n + 0.5)).ln());
        assert_eq!(super::quantize_energy(ubiquitous, 0.5), 0);
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
        // On N = 4 the floored energy is non-positive (exactly 0 under the exact posterior form:
        // df_min = 2 ⇒ ln((4 − 2 + κ)/(2 + κ)) = ln 1), so the guard cannot hold and is
        // intentionally skipped (E_floored < GUARD_MIN_E_FLOORED). The floored weight quantizes
        // to 0; recall is preserved not here but by the §7 count-only union (`score_union`)
        // recovering the seg.
        let n = 4.0;
        let ef = e_floored(n, NU, KAPPA);
        assert!(ef <= 0.0, "E_floored non-positive at N=4: {ef}");
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
    //! all-common pass-through, its concentrated cap, and its hard floor at 0; the literal-§12
    //! floored-included `E_top` reading; and the per-order non-floored popcount credit.
    use super::{concentration_cap, count_credit, e_floored, e_max, energy, overlap_and_credit};
    use crate::DEFAULT_SIGMA;
    use crate::hash::FxHashMap;
    use croaring::Bitmap;

    const NU: f64 = 2.0;
    const KAPPA: f64 = 0.5;

    fn approx(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-6, "expected ≈ {b}, got {a}");
    }

    /// Thin alias for [`concentration_cap`] with an all-non-floored flag vector, keeping the
    /// (majority) non-floored-anchor fixtures concise.
    fn cap(energies: &[f64]) -> Option<f64> {
        concentration_cap(energies, &vec![false; energies.len()])
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
        assert_eq!(cap(&[-1.0, -0.5, -2.0]), None);
        // Comparable positive energies with no member below ½·E_top ⇒ no commons ⇒ uncapped.
        assert_eq!(cap(&[2.0, 2.0, 2.0]), None);
        // Degenerate inputs never panic / never spuriously cap.
        assert_eq!(cap(&[]), None);
        assert_eq!(cap(&[f64::NEG_INFINITY, f64::NEG_INFINITY]), None);
    }

    #[test]
    fn cap_limits_a_concentrated_set() {
        // One dominant gram (E_top=5) amid 2 commons (0.5, 0.3 < 2.5): cap = (5 − 0.8)/(2−1) = 4.2.
        approx(cap(&[5.0, 0.5, 0.3]).expect("concentrated"), 4.2);
        // Only 1 common (3.0 ≥ ½·5) ⇒ no dominant-amid-commons ⇒ not concentrated.
        assert_eq!(cap(&[5.0, 0.5, 3.0]), None);
    }

    #[test]
    fn cap_floors_at_zero_when_commons_outweigh() {
        // The commons collectively outweigh the dominant gram ⇒ the hard floor discards the credit.
        approx(cap(&[1.0, 0.4, 0.4, 0.4]).expect("concentrated"), 0.0); // (1.0 − 1.2)/3 < 0 ⇒ 0
    }

    #[test]
    fn cap_boundary_e_equals_half_is_not_common() {
        // §9 uses a STRICT `E < ½·E_top`: a gram exactly at half is NOT a common, so two grams at
        // the boundary leave 0 commons ⇒ uncapped.
        assert_eq!(cap(&[4.0, 2.0, 2.0]), None);
        // One nudged just below half ⇒ still only 1 common ⇒ uncapped (needs ≥ 2).
        assert_eq!(cap(&[4.0, 2.0, 1.999]), None);
        // Two strictly below half ⇒ concentrated.
        assert!(cap(&[4.0, 1.999, 1.999]).is_some());
    }

    #[test]
    fn cap_neg_infinity_commons_contribute_zero_no_nan() {
        // Extreme-negative energies (a degenerate/backstop input; in-contract v0.5 energies are
        // finite even at df = N) count as commons (`< ½·E_top`) but contribute `max(0,−∞) = 0`
        // to the sum, so the cap stays finite (no NaN).
        let c = cap(&[5.0, f64::NEG_INFINITY, f64::NEG_INFINITY]).expect("2 commons");
        approx(c, 5.0); // (5 − 0)/(2 − 1)
        assert!(c.is_finite());
        // More ubiquitous commons shrink the cap (the chance-match guard), still finite.
        let c = cap(&[5.0, f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY])
            .expect("3 commons");
        approx(c, 2.5); // 5/(3 − 1)
        assert!(c.is_finite());
    }

    #[test]
    fn cap_floored_e_top_anchors_and_protects_the_rare_gram() {
        // Literal §12 (§9/§12, v0.4/M6, resolved): a genuinely-rare, real discriminating gram is
        // itself FLOORED (df ≤ √N, so it sits at E_max ≈ 6.9) — the common "find the doc with this
        // rare term" query. Because floored grams are NOT excluded, that 6.9 anchors `E_top`; and
        // because a floored anchor earns NO credit itself (§9 withholding), the tight inequality
        // is `E_top ≥ Σ E_c + #c·μ`, so the denominator is `#c` (v0.5 — resolves flag #2): with
        // two commons at 0.5, cap = (6.9 − 1.0)/2 = 2.95. At that cap, a two-commons-only doc
        // scores Σ E_c + 2μ ≤ 6.9 = the on-topic doc's E_top — protected, not merely bounded.
        // (Excluding the floored gram — considered and rejected — would drop `E_top` to 0.5,
        // leave no dominant, and DISABLE the cap for exactly this query, letting the commons win.)
        approx(
            concentration_cap(&[6.9, 0.5, 0.5], &[true, false, false])
                .expect("the floored rare gram anchors E_top"),
            2.95,
        );
    }

    #[test]
    fn cap_denominator_resolves_the_flag_2_inversion() {
        // The recorded M2-panel inversion (handoff flag #2): an off-topic commons doc scored 8.09
        // vs the on-topic floored-anchor doc's 6.20. Mechanism: the pre-v0.5 `#c − 1` denominator
        // assumes the anchor ALSO earns μ; a floored anchor earns none, so a commons doc could
        // out-credit it by exactly one μ. With the `#c` denominator, the commons doc's best score
        // Σ E_c + #c·cap = E_top never exceeds the anchored doc's.
        let (e_top, commons) = (6.2_f64, [0.9_f64, 0.8, 0.7]);
        let energies = [e_top, commons[0], commons[1], commons[2]];
        let floored = [true, false, false, false];
        let sum_c: f64 = commons.iter().sum();
        let cap_new = concentration_cap(&energies, &floored).expect("concentrated");
        approx(cap_new, (e_top - sum_c) / 3.0); // #c = 3 (anchor floored)
        let commons_doc = sum_c + 3.0 * cap_new;
        assert!(
            commons_doc <= e_top + 1e-9,
            "commons-only doc ({commons_doc}) can no longer out-credit the on-topic doc ({e_top})"
        );
        // The old denominator (#c − 1) permitted exactly the inversion:
        let cap_old = (e_top - sum_c) / 2.0;
        assert!(
            sum_c + 3.0 * cap_old > e_top,
            "the pre-v0.5 cap admitted the inversion this test pins the fix for"
        );
        // A NON-floored anchor keeps the #c − 1 denominator (it earns μ itself).
        let cap_unfloored = concentration_cap(&energies, &[false; 4]).expect("concentrated");
        approx(cap_unfloored, (e_top - sum_c) / 2.0);
    }

    #[test]
    fn overlap_and_credit_fuses_raw_count_with_non_floored_per_order_credit() {
        // Three present grams: a non-floored order-3 (μ=2.0), a FLOORED order-3 (no credit), and a
        // non-floored order-2 (μ=1.5). One fused pass yields BOTH the raw overlap (all matched,
        // floored included) and the §3/§9 credit (non-floored only, per-order μ).
        let postings = vec![Bitmap::of(&[7]), Bitmap::of(&[7]), Bitmap::of(&[7, 9])];
        let floored = vec![false, true, false];
        let orders = vec![3u8, 3u8, 2u8];
        let mut mu_by_order = FxHashMap::default();
        mu_by_order.insert(3u8, 2.0);
        mu_by_order.insert(2u8, 1.5);
        // id 7 is in all three (overlap 3), but the floored one earns nothing: 2.0 + 1.5 = 3.5.
        let (overlap, credit) = overlap_and_credit(7, &postings, &floored, &orders, &mu_by_order);
        assert_eq!(overlap, 3, "raw overlap counts the floored gram too");
        approx(credit, 3.5);
        // id 9 is only in the order-2 posting: overlap 1, credit 1.5.
        let (overlap, credit) = overlap_and_credit(9, &postings, &floored, &orders, &mu_by_order);
        assert_eq!(overlap, 1);
        approx(credit, 1.5);
        // id 1 is in nothing.
        let (overlap, credit) = overlap_and_credit(1, &postings, &floored, &orders, &mu_by_order);
        assert_eq!(overlap, 0);
        approx(credit, 0.0);
    }
}

#[cfg(test)]
mod budget_tests {
    //! v0.4/M6 §5/§7 derived work budget `C = (1/r)·ln(N/k)·d̄/ln(N/d̄)`: the closed form against a
    //! reference computation, the recall-safe guards (all → `None`/unbounded), and monotonicity in `d̄`.
    use super::derived_budget;
    use crate::welford::{ClassSnap, ClassStats};

    /// A snapshot whose single Latin-trigram class carries the given `ln(df)` samples.
    fn snap_with(dfs: impl IntoIterator<Item = i64>) -> ClassSnap {
        let mut stats = ClassStats::new();
        for df in dfs {
            stats.add_sample(1, 3, df);
        }
        stats.snapshot_for([(1u8, 3u8)])
    }

    #[test]
    fn formula_matches_the_closed_form() {
        let snap = snap_with(1..=1000);
        let (mean, std) = snap.pooled_log_df().unwrap();
        let d_bar = (mean + 2.0 * std).exp();
        let (n, k, r) = (100_000.0_f64, 128.0_f64, 0.9_f64);
        let want = ((1.0 / r) * (n / k).ln() * d_bar / (n / d_bar).ln()).round();
        let got = derived_budget(100_000, 128, 0.9, &snap).expect("a healthy corpus derives C");
        assert!(
            (got as f64 - want).abs() < 1.0,
            "derived C matches the closed form: got {got}, want {want}"
        );
    }

    #[test]
    fn guards_fall_back_to_unbounded() {
        let snap = snap_with(1..=1000);
        // N ≤ k → None (no pruning to do).
        assert_eq!(derived_budget(100, 128, 0.9, &snap), None);
        assert_eq!(derived_budget(128, 128, 0.9, &snap), None);
        // A non-finite / non-positive reliability → None.
        assert_eq!(derived_budget(100_000, 128, f64::NAN, &snap), None);
        assert_eq!(derived_budget(100_000, 128, 0.0, &snap), None);
        assert_eq!(derived_budget(100_000, 128, -0.5, &snap), None);
        // A snapshot too sparse to pool stats → None.
        assert_eq!(derived_budget(100_000, 128, 0.9, &ClassSnap::empty()), None);
    }

    #[test]
    fn d_bar_at_or_above_n_falls_back_to_unbounded() {
        // When the representative gram df d̄ reaches N−1 (a ~ubiquitous corpus), ln(N/d̄) → 0 and the
        // budget would blow up: the guard returns None (unbounded), the recall-safe outcome.
        let snap = snap_with(1..=1000);
        let (mean, std) = snap.pooled_log_df().unwrap();
        let d_bar = (mean + 2.0 * std).exp();
        // Pick N just below d̄ so d̄ > N−1.
        let n = (d_bar as u64).max(2);
        assert_eq!(derived_budget(n, 1, 0.9, &snap), None);
    }

    #[test]
    fn c_is_monotone_increasing_in_d_bar() {
        // Larger d̄ (a commoner representative gram) ⇒ larger C. Two tight, well-separated classes.
        let low = snap_with(5..=15); // d̄ ≈ 20
        let high = snap_with(990..=1000); // d̄ ≈ 1000
        let (n, k, r) = (1_000_000u64, 128u64, 0.9);
        let (m_lo, s_lo) = low.pooled_log_df().unwrap();
        let (m_hi, s_hi) = high.pooled_log_df().unwrap();
        assert!(
            (m_hi + 2.0 * s_hi).exp() > (m_lo + 2.0 * s_lo).exp(),
            "the high-df class has the larger d̄ (test premise)"
        );
        let c_lo = derived_budget(n, k, r, &low).unwrap();
        let c_hi = derived_budget(n, k, r, &high).unwrap();
        assert!(c_hi > c_lo, "larger d̄ ⇒ larger C ({c_hi} > {c_lo})");
    }
}

#[cfg(test)]
mod null_tests {
    //! Numerical fixtures for the v0.4 M3 §6 saturating length null: the `rel_len = 0` no-op, the
    //! separable rare-gram slope, the commons' saturating `π_g` (incl. `p = 1` full debit), the
    //! rare linear over-debit direction, and length monotonicity.
    use super::length_null;

    fn approx(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-6, "expected ≈ {b}, got {a}");
    }

    #[test]
    fn null_is_zero_at_zero_relative_length() {
        // rel_len = 0 ⇒ a zero-length draw is present nowhere: π_g(0) = 1−(1−p)^0 = 0 for every
        // gram and the rare slope rel_len·K_rare vanishes ⇒ no debit, even for a ubiquitous p = 1.
        approx(length_null(0.0, 5.0, &[(0.5, 2.0), (1.0, 3.0)]), 0.0);
    }

    #[test]
    fn rare_grams_are_a_separable_linear_slope() {
        // No commons ⇒ the null is exactly rel_len·K_rare (the precomputed-once rare slope, §6/§12).
        approx(length_null(1.0, 3.0, &[]), 3.0);
        approx(length_null(2.5, 3.0, &[]), 7.5);
    }

    #[test]
    fn commons_saturate_and_p_one_debits_the_full_weight() {
        // A ubiquitous gram (p = 1) is present in every draw: π = 1−0^rel = 1 (rel > 0), full weight.
        approx(length_null(2.0, 0.0, &[(1.0, 4.0)]), 4.0);
        // p = 0.5, rel = 2: π = 1 − 0.5² = 0.75 ⇒ 0.75·4 = 3.0 (saturating; the linear 2·0.5·4 = 4
        // would over-debit and exceed the weight — exactly what the saturating form prevents).
        approx(length_null(2.0, 0.0, &[(0.5, 4.0)]), 3.0);
        // p = 0.5, rel = 0.5: π = 1 − √0.5 ⇒ ·4.
        approx(
            length_null(0.5, 0.0, &[(0.5, 4.0)]),
            (1.0 - 0.5_f64.sqrt()) * 4.0,
        );
    }

    #[test]
    fn rare_linear_overdebits_but_only_negligibly() {
        // §6 recall direction: the linear rare form (rel·p)·w ≥ the exact saturating π·w, so K_rare
        // OVER-debits — but the gap is O((rel·p)²), < 0.5% of the weight at p < P_LINEAR = 0.02.
        let (p, w, rel) = (0.01_f64, 3.0_f64, 5.0_f64);
        let linear = rel * p * w; // the K_rare contribution
        let exact = (1.0 - (1.0 - p).powf(rel)) * w; // if treated as a saturating common
        assert!(linear >= exact, "linear over-estimates π (over-debits)");
        assert!(
            linear - exact < 0.005 * w,
            "the over-debit is < 0.5% of the weight in the rare regime"
        );
    }

    #[test]
    fn null_is_monotone_increasing_in_length() {
        // Longer segments have more chance matches ⇒ a strictly larger debit (both the rare slope
        // and the saturating commons grow with rel_len).
        let commons = [(0.3, 1.0), (0.6, 2.0)];
        let mut prev = f64::NEG_INFINITY;
        for &rel in &[0.0, 0.5, 1.0, 2.0, 4.0, 8.0] {
            let null = length_null(rel, 1.5, &commons);
            assert!(null > prev, "null grows with length: {null} > {prev}");
            prev = null;
        }
    }
}

#[cfg(test)]
mod topk_tests {
    //! The v0.5 incremental top-`k` tracker behind the eager early-stop's `kth_best` ceiling:
    //! oracle agreement under inserts and per-result increases (the two event kinds `upsert_best`
    //! emits), promotion/demotion through the top set, and the not-yet-full `None`.
    use super::{ResultKey, TopK};

    /// A segment-granularity result key for the fixtures.
    fn rk(key: i64) -> ResultKey {
        ResultKey::Seg(key as u32)
    }

    /// The `k`-th largest of the current per-result values, by sort (the oracle).
    fn oracle(vals: &[(i64, f64)], k: usize) -> Option<f64> {
        if vals.len() < k {
            return None;
        }
        let mut v: Vec<f64> = vals.iter().map(|&(_, x)| x).collect();
        v.sort_by(|a, b| b.total_cmp(a));
        Some(v[k - 1])
    }

    #[test]
    fn tracks_the_kth_best_under_inserts_and_promotions() {
        let mut t = TopK::new(3);
        let mut state: Vec<(i64, f64)> = Vec::new();
        assert_eq!(t.kth_best(), None, "empty");
        for (key, v) in [(1, 5.0), (2, 3.0)] {
            t.observe(&rk(key), v);
            state.push((key, v));
            assert_eq!(t.kth_best(), None, "fewer than k keys");
        }
        // Fill to k, then stream a mix of new keys (some below the kth, some promoting).
        for (key, v) in [(3, 4.0), (4, 1.0), (5, 6.0), (6, 3.5), (7, 4.5)] {
            t.observe(&rk(key), v);
            state.push((key, v));
            assert_eq!(t.kth_best(), oracle(&state, 3), "after inserting {key}={v}");
        }
    }

    #[test]
    fn per_key_increases_update_in_place_and_can_promote() {
        let mut t = TopK::new(2);
        let mut state: Vec<(i64, f64)> = vec![(1, 5.0), (2, 4.0), (3, 1.0)];
        for &(k, v) in &state {
            t.observe(&rk(k), v);
        }
        assert_eq!(t.kth_best(), oracle(&state, 2)); // {5, 4} → 4
        // A top member improves: kth unchanged (it was the max's peer), stale entry ignored.
        t.observe(&rk(2), 4.5);
        state[1].1 = 4.5;
        assert_eq!(t.kth_best(), oracle(&state, 2)); // {5, 4.5} → 4.5
        // A non-top key improves past the kth: promoted, demoting the previous kth.
        t.observe(&rk(3), 4.8);
        state[2].1 = 4.8;
        assert_eq!(t.kth_best(), oracle(&state, 2)); // {5, 4.8} → 4.8
        // The demoted key re-improves and re-promotes (stale entries from both phases ignored).
        t.observe(&rk(2), 6.0);
        state[1].1 = 6.0;
        assert_eq!(t.kth_best(), oracle(&state, 2)); // {6, 5} → 5
    }

    #[test]
    fn k_of_one_tracks_the_max() {
        let mut t = TopK::new(1);
        t.observe(&rk(1), 2.0);
        assert_eq!(t.kth_best(), Some(2.0));
        t.observe(&rk(2), 1.0); // below — ignored
        assert_eq!(t.kth_best(), Some(2.0));
        t.observe(&rk(2), 9.0); // promotes
        assert_eq!(t.kth_best(), Some(9.0));
        t.observe(&rk(1), 11.0); // the demoted key re-promotes
        assert_eq!(t.kth_best(), Some(11.0));
    }

    #[test]
    fn randomized_stream_agrees_with_the_oracle() {
        // A deterministic pseudo-random stream of inserts + increases across ks.
        for k in [1usize, 2, 5, 8] {
            let mut t = TopK::new(k);
            let mut state: Vec<(i64, f64)> = Vec::new();
            let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
            for i in 0..500 {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let key = (x >> 33) as i64 % 40;
                let bump = ((x >> 11) & 0xFFF) as f64 / 256.0 + 1e-3;
                match state.iter_mut().find(|(k2, _)| *k2 == key) {
                    Some(slot) => {
                        slot.1 += bump; // strictly increases (the upsert_best contract)
                        t.observe(&rk(key), slot.1);
                    }
                    None => {
                        state.push((key, bump));
                        t.observe(&rk(key), bump);
                    }
                }
                assert_eq!(t.kth_best(), oracle(&state, k), "step {i}, k={k}");
            }
        }
    }
}

#[cfg(test)]
mod fusion_tests {
    //! v0.4/M5 rank-view RRF fusion (derivation §8): the reciprocal-rank-fusion rank math, the
    //! `missing="omit"` rule (a seg in only one view is not worst-rank-penalized), the "reads RANKS
    //! not summed energy" property (a contiguous match is not additively tripled by its sub-grams),
    //! the deterministic tiebreak, and the ΔH → view-weight map.
    use super::{Candidate, K_RRF, rrf_fuse, view_weights_from_dh};
    use crate::Collapse;
    use crate::hash::FxHashSet;
    use crate::model::Key;
    use crate::term::encode_term;
    use crate::tokenize::DefaultTokenizer;
    use crate::welford::ClassStats;

    fn cand(key: i64, seg: u32, corrected: f64) -> Candidate {
        Candidate {
            key: Key::Integer(key),
            label: "f".to_string(),
            seg_id: seg,
            score: 1,
            overlap: 2,
            corrected_score: corrected,
            energy: 0.0,
            count: 0.0,
            length: 0.0,
        }
    }

    #[test]
    fn rrf_reads_ranks_not_summed_energy_and_omits_the_absent() {
        // view_a = the trigram (primary) view's corrected-float order; view_b = the bigram
        // (secondary) view's. Key 1 is a CONTIGUOUS match: it ranks #1 in BOTH views (its trigram
        // tops view_a, its sub-bigrams top view_b). Keys 2 and 3 appear in ONE view each.
        let view_a = vec![cand(1, 1, 9.0), cand(3, 3, 5.0)];
        let view_b = vec![cand(1, 1, 8.0), cand(2, 2, 4.0)];
        let weights = vec![0.6, 0.4];
        let fused = rrf_fuse(&[view_a, view_b], &weights, None, Collapse::None);

        let score = |k: i64| {
            fused
                .iter()
                .find(|c| c.key().as_i64() == Some(k))
                .unwrap()
                .corrected_score()
        };
        // Key 1: rank 1 in both → 0.6/(k+1) + 0.4/(k+1) = 1.0/(K_RRF+1) — exactly TWO view
        // contributions, NOT proportional to the THREE grams (trigram + 2 sub-bigrams) that drove
        // those ranks. Pooling would have summed the energies (9.0 + 8.0 = 17), letting one
        // contiguous match dominate; RRF caps it at the rank contribution. This IS "fusion beats
        // pooling: a contiguous trigram match is not additively tripled by its sub-bigrams."
        assert!((score(1) - 1.0 / (K_RRF + 1.0)).abs() < 1e-12);
        // Key 2: present ONLY in view_b at rank 2 → 0.4/(K_RRF+2). The absent view_a contributes
        // NOTHING (missing="omit"), not a worst-rank penalty.
        assert!((score(2) - 0.4 / (K_RRF + 2.0)).abs() < 1e-12);
        // Key 3: present ONLY in view_a at rank 2 → 0.6/(K_RRF+2).
        assert!((score(3) - 0.6 / (K_RRF + 2.0)).abs() < 1e-12);
        // Order by fused score: 1 (1/61) > 3 (0.6/62) > 2 (0.4/62).
        assert_eq!(
            fused
                .iter()
                .map(|c| c.key().as_i64().unwrap())
                .collect::<Vec<_>>(),
            [1, 3, 2]
        );
    }

    #[test]
    fn fused_candidate_reports_governing_view_components_never_a_cross_view_sum() {
        // C4/§10: key 1 is in BOTH views — rank #1 in view_a, rank #2 in view_b — so `rrf_fuse`
        // retains view_a's clone (the best-ranked). Its energy()/count()/length() must be view_a's
        // (5.0/1.5/0.5), NEVER the cross-view sum with view_b's (3.0/9.9/9.9): summing would
        // double-count the same text at two granularities, exactly what RRF exists to prevent.
        let mut a1 = cand(1, 1, 0.0);
        a1.energy = 5.0;
        a1.count = 1.5;
        a1.length = 0.5;
        let mut b1 = cand(1, 1, 0.0);
        b1.energy = 3.0;
        b1.count = 9.9;
        b1.length = 9.9;
        let view_a = vec![a1, cand(9, 9, 0.0)]; // key 1 at rank 1 (retained)
        let view_b = vec![cand(2, 2, 0.0), b1]; // key 1 at rank 2
        let fused = rrf_fuse(&[view_a, view_b], &[0.5, 0.5], None, Collapse::None);
        let one = fused.iter().find(|c| c.key().as_i64() == Some(1)).unwrap();
        assert_eq!(
            one.energy(),
            5.0,
            "governing view's energy, not the 8.0 sum"
        );
        assert_eq!(one.count(), 1.5, "governing view's count");
        assert_eq!(one.length(), 0.5, "governing view's length");
        assert!(
            (one.nat_score() - 6.0).abs() < 1e-12,
            "nat_score = 5.0 + 1.5 − 0.5 (nat-scale, from the governing view)"
        );
        // The fused ranking key stays RRF-scale (small), a different scale from nat_score.
        assert!(
            one.corrected_score() > 0.0 && one.corrected_score() < 0.1,
            "corrected_score is the RRF-scale rank key ({})",
            one.corrected_score()
        );
        assert!(
            one.nat_score() > 1.0,
            "nat_score stays nat-scale for a fused query, not RRF-scale"
        );
    }

    #[test]
    fn rrf_truncates_to_limit_after_fusing() {
        let view_a = vec![cand(1, 1, 9.0), cand(2, 2, 5.0), cand(3, 3, 1.0)];
        let view_b = vec![cand(2, 2, 9.0), cand(1, 1, 5.0)];
        let fused = rrf_fuse(&[view_a, view_b], &[0.5, 0.5], Some(2), Collapse::None);
        assert_eq!(fused.len(), 2, "top-k applied after fusion");
        // Keys 1 and 2 are in both views; key 3 only in view_a low → truncated out.
        assert!(fused.iter().all(|c| c.key().as_i64() != Some(3)));
    }

    #[test]
    fn rrf_tiebreak_is_deterministic() {
        // Two keys with identical fused score (each rank 1 in one view, weights equal): the tiebreak
        // (fused desc → integer energy desc → seg id asc) breaks it by seg id, deterministically.
        let view_a = vec![cand(10, 10, 1.0)];
        let view_b = vec![cand(20, 20, 1.0)];
        let a = rrf_fuse(
            &[view_a.clone(), view_b.clone()],
            &[0.5, 0.5],
            None,
            Collapse::None,
        );
        let b = rrf_fuse(&[view_a, view_b], &[0.5, 0.5], None, Collapse::None);
        assert_eq!(
            a.iter()
                .map(|c| c.key().as_i64().unwrap())
                .collect::<Vec<_>>(),
            b.iter()
                .map(|c| c.key().as_i64().unwrap())
                .collect::<Vec<_>>()
        );
        // seg 10 < seg 20, equal scores ⇒ key 10 first.
        assert_eq!(a[0].key().as_i64(), Some(10));
    }

    #[test]
    fn view_weights_favor_the_richer_primary_inventory() {
        // ΔH = ln V_primary − ln V_secondary: a richer (larger-vocab) primary order earns more
        // fusion weight (derivation §8). Latin primary order is the trigram (3), secondary the
        // bigram (2).
        let latin = encode_term("abc").unwrap().class();
        let tok = DefaultTokenizer::new();
        let mut stats = ClassStats::new();
        for df in 1..=1000 {
            stats.add_sample(latin, 3, df); // V_primary = 1000 distinct trigrams
        }
        for df in 1..=50 {
            stats.add_sample(latin, 2, df); // V_secondary = 50 distinct bigrams
        }
        let snap = stats.snapshot_for([(latin, 3u8), (latin, 2u8)]);
        let mut scripts: FxHashSet<u8> = FxHashSet::default();
        scripts.insert(latin);
        let w = view_weights_from_dh(&scripts, &tok, &snap);
        assert!(
            w[0] > 0.5 && w[0] <= 0.9,
            "richer primary inventory ⇒ more primary weight: {w:?}"
        );
        assert!((w[0] + w[1] - 1.0).abs() < 1e-12, "weights sum to 1");
    }

    #[test]
    fn fuse_granularity_folds_per_segment_or_per_key() {
        // Two SEGMENTS of one KEY, each ranked in one view. Per-segment (the v0.5 default):
        // both survive as distinct results. Per-key (Collapse::Key): they fold to one result —
        // the better-ranked segment — with the fused score accumulated across both.
        let seg_a = cand(1, 10, 9.0); // key 1, segment 10
        let seg_b = cand(1, 11, 5.0); // key 1, segment 11
        let view_a = vec![seg_a.clone()];
        let view_b = vec![seg_b.clone()];
        let per_seg = rrf_fuse(
            &[view_a.clone(), view_b.clone()],
            &[0.5, 0.5],
            None,
            Collapse::None,
        );
        assert_eq!(
            per_seg.len(),
            2,
            "per-segment keeps both segments of the key"
        );
        let per_key = rrf_fuse(&[view_a, view_b], &[0.5, 0.5], None, Collapse::Key);
        assert_eq!(
            per_key.len(),
            1,
            "Collapse::Key folds to one result per key"
        );
        assert_eq!(
            per_key[0].seg_id, 10,
            "the fold retains the best-ranked segment"
        );
    }

    #[test]
    fn view_weights_are_equal_when_dh_is_unavailable() {
        // An empty snapshot (no vocab populated) ⇒ ΔH unavailable ⇒ equal 0.5/0.5 weights.
        let tok = DefaultTokenizer::new();
        let snap = crate::welford::ClassSnap::empty();
        let mut scripts: FxHashSet<u8> = FxHashSet::default();
        scripts.insert(encode_term("abc").unwrap().class());
        let w = view_weights_from_dh(&scripts, &tok, &snap);
        assert_eq!(w, vec![0.5, 0.5]);
    }
}
