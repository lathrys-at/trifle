# Changelog

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
  the field-aware index milestone, and a few §5/§9 precision refinements (incl. selsweep-tuning the
  derived budget's `Z` shape constant down toward the latency knee, a recall-preserving 0.4.1 win).
  Field-scoped results already work today via a [`SqlFilter`] on `seg.label` (see its rustdoc).
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
