# Tuning trifle: when, why, and how to change a default

trifle's defaults are not the usual "reasonable starting values a maintainer picked." Nearly
every one is **derived** — either from the scoring model (`derivation.md`) or from your corpus's
own statistics at query time — and each is chosen so that when it is wrong, it is wrong in the
*recall-safe* direction (you scan slightly more than necessary, rather than miss a match). That
changes the posture of tuning:

> **The burden of proof is on the change, not the default.** Change a knob only when a
> measurement on *your* corpus and *your* queries shows the default is mis-placed — and keep the
> measurement, because the right value can drift as your corpus grows.

This document covers every public knob: what it means in the model, exactly **when** you would
change it, **which direction** and how far, and **what evaluation justifies** the change. The
knobs fall into three tiers:

| Tier | Knobs | Posture |
|---|---|---|
| Day-one dials | `SearchOpts::min_shared`, `SearchOpts::df_budget` | Reach for these freely; they encode product decisions (strictness, latency ceiling) |
| Model constants | `Config::sigma`; `Tuning::{nu, kappa, delta, k_target, c_margin}` | Change only with a measured evaluation; each moves the scoring model itself |
| Index identity | tokenizer choice + normalization, `Config::data_version`, `Schema` | Not tuning — changing any of these **drift-resets the cache** (drop + rebuild, by design) |

---

## 0. First, build the evaluation you'll reuse

Every recommendation below refers to one of three measurements. Set them up once; all three run
against your real index with no labeling effort:

**R — Recall\@k on planted queries.** Sample a few hundred indexed segments. For each, form
queries a user would plausibly type: a distinctive word or two from the text, plus corrupted
variants (1–2 random character substitutions — the same generator the in-tree benchmark harness
uses). The "right answer" is the source segment's key. Report the fraction of queries whose
source key appears in the top `k` (`k` = your product's result count). This is the recall gate:
**no knob change ships if R drops.**

**W — Work per query.** The posting rows a query actually scans, `Σdf` over its selected grams —
the quantity `df_budget` bounds and the honest latency proxy (it is cardinality-independent of
everything else in the pipeline). Measure it off the public API:

```rust
let reader = index.reader()?;
let stream = reader.candidates(query, &opts)?;
let work: u64 = stream.present_terms().map(|(_, df)| df).sum();
```

**L — Latency.** p50/p99 wall-clock of `matches()` on a realistic query mix. W predicts L well;
measure both when you are about to commit a change.

The `benchmarks/` crate automates R and W as a **frontier sweep** (`selsweep`): recall\@k versus
`Σdf` across a grid of `df_budget` values, with a marker for where the *derived* default budget
lands on that curve. If you tune seriously, run it on a dump of your corpus — most of the
questions below are answered by looking at one plot.

---

## 1. Day-one dials

### `SearchOpts::min_shared` — strictness (`m`, default 2)

**What it is.** The raw-overlap floor: a segment must share at least `m` selected grams with the
query to be a candidate at all. It is clamped to the query's gram count, so a 1-gram query still
works at the default.

**When to change it.**

- **`m = 1`** when single-gram evidence should match — very short queries against short fields
  (codes, initials, single CJK characters) where one shared rare gram is genuinely meaningful.
  Expect more coincidence matches on everything else; prefer scoping it per call site rather
  than globally.
- **`m = 3+`** when your UI shows few results and coincidental two-gram overlaps pollute them —
  typically corpora of very short segments (titles), where two shared grams happen easily.

**Evaluation.** This is a precision/recall product decision, so measure both sides: R (does the
planted source still surface under typos? — raising `m` eats typo tolerance directly, since a
typo destroys up to `n` grams per edit) and a precision spot-check of the top-`k` on your worst
noisy queries. There is no corpus-derived "right" value; there is a right value *for your UI*.

### `SearchOpts::df_budget` — the work budget (`C`, default: derived per query)

**What it is.** A cap on `Σdf` over the selected grams — the posting rows scanned, hence the
latency ceiling. `None` does **not** mean unbounded: it means *derive `C` from the corpus*, per
query, as the budget that just funds the confidence-bounded stop
(`C = (1/σ)·ln(N/k)·d̄/ln(N/d̄)`, with `d̄` a high percentile of the query's own classes' df
distribution — see derivation §5/§7). The derived value is deliberately recall-safe-generous.

**When to change it.**

- **Set it explicitly (lower) when you need a hard latency ceiling.** The derived budget adapts
  to the corpus, which is usually what you want — but "usually" is not an SLO. If p99 latency on
  common-word queries must be bounded, bind the budget that meets it.
- **Raise it (or bind `u64::MAX`) when recall on mixed rare+common queries measurably suffers** —
  the signature is planted queries whose distinctive word is *mid-rarity* (df just above the
  derived `d̄`) failing R while their rare-word variants pass.
- Leave it alone otherwise. It is the single biggest work lever, and the derived default is the
  product of the whole §5 derivation.

**Evaluation.** The frontier sweep is purpose-built for this knob: plot R against W across a
budget grid and find the knee. The derived default should sit at or just right of the knee; if
your corpus's knee is materially left of it, bind the knee value and bank the latency. Re-run
the sweep when the corpus grows an order of magnitude — the knee moves, and (unlike a bound
value) the derived default moves with it. That asymmetry is the argument for leaving it derived
unless you have an SLO.

---

## 2. Model constants — change with evidence only

These move the scoring model. Each has a derivation-fixed meaning, a default that is either
canonical (`kappa`) or conservative (`sigma`, `c_margin`), and a specific measurable failure
that justifies moving it. If none of the failure signatures below describes your measurement,
the knob you want is probably in tier 1.

### `Config::sigma` — reliability / topicality (`σ`, default 0.9)

**What it is.** The probability that a relevant segment actually contains a correct query gram —
a property of *your corpus and what "relevant" means in it*, not of any query. It is the one
constant that genuinely cannot be read off index statistics. It drives three things at once:
the count credit `μ = logit σ` (how much matching *more* of the query is worth, versus matching
*rarer* parts of it), the stop's mean and recall-safety gate, and the derived budget's `1/σ`
factor.

**When to change it.**

- **Raise it (0.95+)** when relevant segments essentially always contain the whole query —
  exact-ish lookup over short structured fields (titles, names, IDs). Signature: R is fine but
  ranking is wrong — segments matching *all* query words rank below segments matching one rarer
  word. A higher σ makes count evidence count more.
- **Lower it (0.7–0.8)** when relevance is topical and partial — a relevant segment often lacks
  some query words (notes, prose paragraphs). Signature: the top of the ranking over-rewards
  full-phrase matches over the better on-topic partial match. Note `σ ≤ 0.5` zeroes the credit
  entirely and (below the stop's gate `σ ≥ c²/(B+c²)`) disables early stopping — both are
  legitimate, deliberate configurations for "count evidence means nothing here," not errors.
- It is index-level (`Config`, not `SearchOpts`) because reliability is a corpus property
  (derivation §3.3). It is runtime-only: changing it does **not** drift-reset.

**Evaluation.** A ranking-quality check, not R: take multi-word planted queries and score
MRR/precision\@1 of the source key at σ ∈ {0.7, 0.8, 0.9, 0.95}. Watch W too — higher σ shrinks
the derived budget (`1/σ`) and fires the stop earlier, so it is also a mild work lever; confirm
R holds wherever you land.

### `Tuning::nu` — corroboration depth (`ν`, default 2)

**What it is.** "At least `ν` grams must agree before a segment is identified": it sets both the
contamination floor `df_min = N^((ν−1)/ν)` (grams rarer than this are treated as possible typo
artifacts: energy capped, no count credit) and the per-gram energy ceiling `E_max = (1/ν)·ln N`.
At the default, `df_min = √N`.

**When to change it.**

- **Lower toward 1** when ultra-rare grams in your corpus are *real and load-bearing* — serial
  numbers, hashes, unique codes — and queries against them are trusted (machine-generated, not
  typed). At `ν = 1` the floor vanishes: every rare gram keeps its full energy and credit.
  Signature: exact-ID queries rank their target below fuzzier matches because the ID's grams
  were floored and denied credit.
- **Raise toward 3** when typo-heavy human queries against a huge corpus produce junk-artifact
  matches in the top-`k` (a floored artifact can still anchor a match). This is rare; the §9
  credit-withholding already handles the common case, so demand the measurement first.
- The right long-term fix for the ID case is a dedicated field on the doc-side channel (no
  floor), which is on the roadmap; `ν` is the blunt instrument available today.

**Evaluation.** Two R variants in tension: R on *clean rare-term* queries (IDs — improves as ν
falls) and precision\@k on *corrupted* queries (junk artifacts — degrades as ν falls). Plot both
against ν ∈ {1, 1.5, 2, 3}; move only if your corpus clearly sits off the default's balance.

### `Tuning::kappa` — Jeffreys pseudocount (`κ`, default 0.5)

**What it is.** The estimation-smoothing constant in the energy
`E = ln((N − df + κ)/(df + κ))`. `κ = 0.5` is the Jeffreys prior — the canonical
uninformative-prior choice, not a tuned value.

**When to change it.** Practically never. Its influence is confined to the rarest tail
(`df ≈ 0–2`) and shrinks as `1/N`. There is no failure signature that `κ` fixes which `ν`
doesn't fix better; it is exposed for completeness and experimentation, and the honest guidance
is: if you find yourself sweeping `κ`, the thing you are looking for is `ν`.

**Evaluation.** None justifies it at realistic `N`. On a toy corpus (`N < 100`) where smoothing
is visible at all, R on rare-gram queries is the check.

### `Tuning::delta` — energy quantization step (`Δ`, default 0.5 nats)

**What it is.** The resolution at which gram energies become integer bit-slice weights. It is a
pure fidelity/work trade: work scales as `~1/Δ` (plane count, bucket count, and the walk all
grow with `E_max/Δ`), while coarser Δ merges energy levels and leaves more ranking to the
tiebreak.

**When to change it.**

- **Raise (0.75–1.0)** if profiling attributes real time to the bit-sliced walk on very large
  `N` (where `E_max = ½ln N` grows and drags the plane count with it). Signature: L improves,
  R unchanged, top-`k` order shifts only among near-tied candidates.
- **Lower (0.25)** only if you observe rank instability among candidates your evaluation says
  should be separated — quantization ties that the float post-pass (credit − null) fails to
  break meaningfully. This is uncommon: the post-pass already operates at full float precision.
- Keep `Δ < 2·E_floored` (≈ `ln N` at the defaults — generous for any real corpus): coarser
  steps quantize floored-gram weights to zero. Debug builds assert this.

**Evaluation.** L (or W is unaffected — this knob is about per-candidate arithmetic, so measure
wall-clock) plus a rank-stability diff: run your query set at both Δ values and count top-`k`
order changes. Accept the coarser Δ when the changes are confined to score-adjacent pairs.

### `Tuning::k_target` — the stop's pool-size target (`k`, default 128)

**What it is.** The candidate-pool size the confidence-bounded stop aims for: selection stops
collecting evidence once it can identify "one of the best `k`" (`ln(N/k)` nats) with margin. It
also feeds the derived budget.

**When to change it.** Match it to your consumption depth when that depth is far from 128:

- **Raise** if you routinely consume deep pools — `matches(_, _, 500)` or a `candidates()`
  stream feeding a reranker that wants 1000. A `k` far below your real depth makes selection
  stop on evidence that distinguishes the top-128 but not the top-1000. Signature: R\@500
  materially below R\@128 on the same queries.
- **Lower (32–64)** for a tight-latency top-5 UI; the stop fires earlier and the derived budget
  shrinks. Confirm R\@5 holds.

**Evaluation.** R at *your* consumption depth versus W, sweeping `k` ∈ {32, 128, 512}. The knob
is honest: it should move roughly with the depth you actually read.

### `Tuning::c_margin` — the Cantelli stopping margin (`c`, default 2)

**What it is.** How much margin the stop demands before it stops collecting grams:
`mean − c·σ ≥ target`, with Cantelli's distribution-free guarantee (`c = 2` bounds the
miss-probability at 1/(1+c²) = 20%). Not a Gaussian z-score. Also sets the recall-safety gate
`σ ≥ c²/(B + c²)` below which the stop never fires and collection runs to the budget.

**When to change it.**

- **Raise (3)** if corrupted-query recall is your product's soul and you can afford the extra
  scan work: the stop demands more margin, so damaged queries collect more corroborating grams
  before stopping.
- **Lower (1–1.5)** to shave work on a corpus where queries are clean and R proves insensitive.
- At `c = 0` the stop fires on the bare mean — the maximally aggressive, least recall-safe end.

**Evaluation.** R on the *corrupted* planted-query variants (the clean ones rarely notice)
against W, sweeping c ∈ {1, 2, 3}. Change it only if the curve is visibly flat (lower) or
visibly still climbing (raise) at 2.

---

## 3. Index identity — not tuning

Changing any of these is a **different index**, and trifle enforces that by drift-reset: on
open, the cache drops to empty and you repopulate with `rebuild()`. That is the designed
behavior (a rebuildable cache never migrates), so plan these as rebuild events, not knob turns.

- **Tokenizer + normalization** (`DefaultTokenizer` vs `NgramTokenizer<N>`; NFC/NFD/
  `NfdStripMarks`/none; `casefold`). The one with real product impact is `NfdStripMarks`
  (accent-insensitive matching: query `cafe` matches stored `café`) — the right default for
  human-typed queries over accented corpora. Note `casefold` is locale-independent
  *lowercasing* (`ß` stays `ß`), not full case folding.
- **`Config::data_version`** — your epoch token. Bump it when *your* interpretation of the
  source data changes (re-chunking, a cleaning pass); it exists so a stale cache can never be
  silently served.
- **`Schema`** — the schema fingerprint folds into the same drift check.

## 4. What is deliberately not a knob

You will find constants in the source that are not exposed: the RRF rank constant, the
starvation energy ratio, the rare/common null threshold `P_LINEAR`, the derived budget's `Z`
percentile, the per-class rarity-normalization floor, the typo floor's damage constant. These
are **shape constants of the derivation** — either pure structure (RRF's 60), or values whose
correctness is an engine-level question rather than a per-corpus one. They are validated (or
retired) against the in-tree benchmark harness, not tuned per deployment. If your evaluation
genuinely implicates one of them, that is a bug report we want — an issue with the frontier
plot attached — not a fork-and-tweak.

## 5. A worked example

A notes app: ~200k segments, human-typed queries, top-10 UI, accents matter.

1. Open with `NfdStripMarks` (accent-insensitive — an identity choice, made up front).
2. Ship the defaults. Collect the planted-query set from real segments in week one.
3. The frontier sweep on a corpus dump shows the recall knee at `Σdf ≈ 30k`, and the derived
   budget landing at ~55k with identical R. p99 is fine → leave `df_budget` derived (it will
   track corpus growth); note the knee for the day p99 isn't fine.
4. Ranking review shows multi-word queries under-rewarding full matches → evaluate
   `sigma` 0.9 vs 0.95 on MRR; 0.95 wins, R holds → set `Config::sigma = 0.95` with the
   measurement linked in a comment.
5. Everything else stays default, and the evaluation stays in CI to re-run quarterly.

That is the intended shape of trifle tuning: one identity decision, one product dial bound to
an SLO if you have one, at most one model constant moved on evidence — and a standing
measurement guarding all of it.
