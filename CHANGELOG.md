# Changelog

## 0.5.0

The post-v0.4.0 review rollup: two correctness fixes to load-bearing invariants, two mathematical
corrections that make the derivation's claims exact, the v0.4.1 performance items, and a breaking
API cleanup that removes the last v0.3 legacy. Purely a **runtime** release for the store — no
`SCHEMA_VERSION` change and no storage-format change — but the `DefaultTokenizer` fingerprint is
unchanged while the `NgramTokenizer` fingerprint gains a layout-version byte, so **NgramTokenizer
caches drift-reset once** on open (drop + rebuild, never migrate); `DefaultTokenizer` caches are
untouched.

### Correctness

- **`batch == serial` restored for mixed-script batches.** The derived work budget `C` pooled its
  representative df `d̄` over the *batch-union* class snapshot, so a Latin-only query co-batched
  with a CJK query derived a different `C` — and could select and rank differently — than the same
  query alone. `C` (and the rank-view `ΔH` lookups, a second membership-dependent leak) are now
  pure functions of each query's own classes. Multi-query `matches_batch` rankings may move — to
  their documented per-query values.
- **Interior digit/symbol fragments no longer trip a spurious secondary rank-view.** In a clean
  query like `ab12cd`, the `Common`-class bigram `12` marked the `Common` "script" structurally
  starved (it never produces a primary-order gram in mixed text), fusing a bigram view into a
  fully-corroborated query — a digit-bigram coincidence could then evict a genuine match under a
  tight `limit`. The `Common` structural trigger is now **word-granular**: an interior fragment
  whose word has primary coverage is not starvation; a standalone digit word still is.
- **Malformed meta counters fail closed.** A present-but-unparseable `next_id` or
  `dict_generation` is now `Error::Corrupt` (mid-operation) or a desync reset (at open) instead of
  silently defaulting — a defaulted `next_id` would *reuse* segment ids.
- **Duplicate labels within one `upsert` call resolve last-wins** (equivalent to sequential
  upserts). Previously debug-asserted; release builds permanently leaked the superseded segment's
  posting entries (df drift + a dead id per shared token until rebuild).

### Scoring (rank-affecting, both in the recall-safe direction)

- **Exact Jeffreys posterior log-odds energy**: `E_g = ln((N − df + κ)/(df + κ))` — the log-odds
  of the exact Beta(κ,κ) posterior mean — replaces the unnormalized `−κ` numerator, which was
  undefined at `df ≥ N − κ` and needed a `−∞` special case. Finite for every `df ≤ N`; differences
  concentrate in the common tail the `max(0,·)` clamp zeroes anyway.
- **The §9 concentration-cap denominator tracks the anchor's credit**:
  `#common − 1 + 1{anchor floored}`. A floored dominant gram earns no count credit itself, so the
  old `#common − 1` over-credited a commons-only doc by exactly one `μ` past the on-topic doc —
  the recorded 8.09-vs-6.20 inversion (v0.4 handoff flag #2), now resolved. Floored grams remain
  eligible to anchor `E_top` (the v0.4/M6 ruling stands).
- Sparse-class rarity now falls back to a **z-score against the query's pooled `ln df` stats**
  (then `ln df`), never raw df — raw df interleaved incommensurably with other classes' z-scores
  in one sort key.

### Performance (no behavior change)

- The engine seam is fused: `trifle-overlap` is **energy-only** (it no longer retains a deep copy
  of every posting nor runs its own raw-overlap `contains` sweep); `search.rs` computes raw
  overlap + count credit + the floor gate in **one** `contains` pass per candidate, before any
  provenance SQL.
- The eager early-stop's `kth_best` ceiling is an **incremental top-k tracker** (amortized
  `O(log n)` per candidate) instead of a per-chunk `O(n)` re-selection (worst `O(union²/CHUNK)`).
- One `alloc_ids(N)` + one `bump_seg_stats` per `upsert` call (was ~6 meta statements per
  segment); selection sort keys are precomputed (`O(n)` transcendentals, not `O(n log n)`);
  `Key` binds as a borrowed SQL parameter (no `Text`/`Blob` clone per bind).
- `benchmarks/` compiles again: the removed `t_max`/`weight_step` sweep arms are replaced by the
  `df_budget` frontier with a derived-`C` marker (the `Z`-knee evaluation of `docs/v0.4.1-plan.md`).

### Breaking API changes

- **`SearchOpts::weight_step`, `WeightStepHint`, and `Stats::weight_step_hint` are removed.**
  `weight_step` has been scoring-inert since v0.4/M1 (the 4-tier weighting it tuned is gone), so
  the hint suggested a value for a knob that did nothing. (Supersedes the v0.3 "keep the
  band-spread hint" ruling, whose rationale died with the tier scheme.)
- **The derivation knobs move behind `SearchOpts::tuning(Tuning)`** (`nu`, `kappa`, `delta`,
  `k_target`, `c_margin`, builder-style). The front line is `min_shared` / `df_budget` / `filter`.
- **[`Match`] now carries the §10 score + components** — `score` (nat-scale
  `energy + count − length` from the governing rank-view, cross-query comparable), `energy`,
  `count`, `length` — and is `PartialEq`-only (floats) and `#[non_exhaustive]`.
- **`#[non_exhaustive]`** on `Config` (new `with_sigma` builder), `Document`, `Match`, `Stats`,
  `CompactStats` — construct via the constructors, not struct literals.
- **`CandidateStream::avgdl()` → `mean_segment_grams()`** (the derivation's `L̄`; `avgdl` was a
  BM25-era name). `tokenize::WindowPolicy` is no longer public (it had no public constructor or
  setter).
- **`trifle-overlap` 0.5.0**: `Counter::build` and `tier_weights` (the v0.3 4-tier scheme) are
  removed — the consumer owns the weighting model; the zero-copy blob API
  (`build_from_blobs`/`build_weighted_from_blobs`, the crate's only `unsafe`) is deleted (no
  consumer could exist: trifle's effective posting is a three-way merge, never a single stored
  blob); `Scored` drops its `overlap` field and the engine no longer enforces the raw-overlap
  floor (`floor()` is advisory; the consumer gates). The weight sum is checked (panics on `u32`
  overflow) with the weight-ceiling precondition documented.

### Docs

- `docs/derivation.md` updated to the exact energy and cap forms, with a quantified §2
  calibration bound, the `ln V` realization of `ΔH` stated, whitespace-only word breaking stated
  as the operative choice, and a new **"Deviations from §12"** table tracking every known
  doc-vs-code gap (including the legacy typo floor `F = m + d`, kept in v0.5 and gated on the
  revived benchmark harness).
- The `Tokenizer` trait documents the full third-party implementor contract; `casefold` is
  documented as locale-independent lowercasing.


## 0.4.0

A scoring rework that re-derives the engine from probabilistic IR (`docs/derivation.md`). The
v0.3 `N`-free 4-tier df-rarity scheme is replaced by an `N`-anchored logit-idf energy, and
tokenization gains whitespace-broken query words and dual-order grams. Drops and rebuilds the
cache on open (`SCHEMA_VERSION` 4 → 5 and a tokenizer-fingerprint bump). The pre-1.0
[`SearchOpts`] surface is finalized here (`t_max` and the reserved `epsilon` are removed — a
breaking change for callers that set them).

- Per-gram weight is the logit-idf **energy** `E_g = ln((N − df_eff − κ)/(df_eff + κ))` — the
  RSJ log-odds, of which v0.3's surprisal is the rare-gram limit — replacing the `N`-free 4-tier
  `{1,2,3,4}` df scheme (knob `D`, which now only feeds the `weight_step_hint` telemetry). The
  engine counts these as `N`-anchored, `Δ`-quantized bit-sliced energy planes.
- A **count credit** `μ = max(0, logit σ)` is added per matched non-floored gram (the §9
  concentration cap bounds it), under a query-side contamination floor `df_min = N^((ν−1)/ν)`
  and energy ceiling `E_max = (1/ν)·ln N`.
- A saturating **length null** `π_g = 1 − (1 − p_g)^(L/L̄)` is subtracted from each candidate.
  The credit and null are a float post-pass over the candidate union in `search.rs` (count-only
  and floored-only candidates are recovered), with top-`k` taken after the floats.
- Pruning gains a distribution-free **Cantelli confidence-bounded stop** (comonotone per-word-block
  variance) plus a per-class floor and skip-and-continue, realizing the §5/§7 `O(C)` work budget.
- The work budget `C` (`SearchOpts::df_budget`) is now **derived by default** from the corpus:
  `C = (1/σ)·ln(N/k)·d̄/ln(N/d̄)`, `d̄ = exp(mean_lndf + 2·std_lndf)` — the Lagrangian dual of the §5
  stop (§5/§7). `None` now means "derive C" (recall-safe guards fall back to unbounded on a
  degenerate corpus); a caller-supplied value still overrides it. This dissolves the last tuned
  selection constant: **`SearchOpts::t_max` is removed** (count is bounded by the query's finite
  gram set, work by `C`).
- **`SearchOpts::epsilon` is removed** (it was reserved and unconsumed; the doc-side `ε` channel is
  a per-*field* property that returns with the field-aware index milestone, post-0.4).
- [`Candidate`] exposes its §10 score components — [`energy`], [`count`], [`length`] (all nats) and
  [`nat_score`] `= energy + count − length` — from the governing (best-ranked / retained) rank-view,
  never a cross-view sum. `nat_score` is a stable, cross-query-comparable magnitude for a downstream
  fusion consumer; `corrected_score` remains the within-query rank key.
- Query tokenization now breaks gram windows on **whitespace and delimiters** and marks query
  words/scripts/order.
- **Dual-order** tokenization: a primary order plus a richness-gated secondary one shorter
  (Latin trigram + bigram, CJK bigram + unigram), with a per-script `starved` gate and rank-view
  **RRF** fusion (cross-script orders pooled, same-run orders fused). A too-short query (2-char
  Latin / 1-char CJK) is now answerable via the structural bigram/unigram fallback — empty in
  v0.3, so a recall improvement.
- `σ` (query-side reliability/topicality) is an index-level [`Config`] constant (a corpus
  property, §3.3). The 7 scoring knobs — `ν`, `κ`, `Δ`, `σ`, `k`, `c`, `C` — live on
  [`SearchOpts`]/[`Config`].
- Deferred to post-0.4: the per-**field** doc-side `ε` channel (`ρ = σ(1 − ε)^n`), which lands with
  the field-aware index milestone, and a few §5/§9 precision refinements. The **performance-only
  0.4.1** follow-up (selsweep-tuning the derived budget's `Z` shape constant down toward the latency
  knee — a recall-preserving win — plus bounded-top-`k` / walk micro-costs) is planned in
  `docs/v0.4.1-plan.md`. Field-scoped results already work today via a [`SqlFilter`] on `seg.label`
  (see its rustdoc).
- On disk: `SCHEMA_VERSION` 4 → 5 (`seg.len` is now the distinct-gram count) and a bumped
  tokenizer fingerprint (windowing change), so an existing cache **drift-resets** (drop + rebuild,
  never migrate). The CRoaring storage byte-format is unchanged.

## 0.3.0

Revision of trifle based on unreleased 0.2.0 draft code. v0.3.0 strips unneeded API complexity and
focuses the crate on the overlap engine and implements performance improvements.

- The `roaring` crate is dropped for `croaring`. Blobs are the standard CRoaring portable
  format.
- The two-level `doc`+`seg` store is now one `seg` table over `(id, key, label, text, len)`,
  with `seg.id` the posting id.
- The pluggable `Ranker`, the over-fetch pool, and the `Effort` knob are removed.
- Per-script-class Welford rarity (multi-script awareness) and the band-spread `WeightStepHint`
  (`Stats.weight_step_hint`) are both retained from v0.2.0.
- Eager `Reader::matches`/`matches_batch` and streaming `Reader::candidates`.
- Opt-in `SqlFilter { fragment, params }` predicate.
- Write API reduced to `upsert(key, &[(label, text)])`, `remove(key)`, and `remove_segment(key, label)`.
- `Index<T: Tokenizer = DefaultTokenizer>` is generic over the tokenizer only.
- `benchmarks/` is reworked against the streaming API.
