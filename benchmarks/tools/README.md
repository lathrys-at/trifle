# Rerank-pool calibration

`calibrate_pool.py` measures and fits trifle's **rerank-pool-depth law** `p(k, N)` — how
deep a candidate pool the precision-tier reranker ([`Bm25Ranker`]) must rescore to recover
the relevant document at `recall@k`, as a function of the result cutoff `k` and the index
size `N`. It is the source of truth behind the [`Effort`] ladder's constants
(`None / Low / Medium / High / Max`).

It does two things:

1. **Sweep** — drives the `ranksweep` subcommand of `trifle-benchmarks` to build the
   `recall@k(pool, N)` matrix for a chosen corpus, over a grid of index sizes `N`.
2. **Fit + render** — fits the power law, draws the curves, and prints the constant `c`
   at each recall target, mapping them onto the shipped `Effort` constants.

```
python3 benchmarks/tools/calibrate_pool.py --corpus msmarco --queries 500 --seed 42
```

---

## Why pool depth is a *power law* in N (from Zipf's law)

trifle generates candidates by **overlap** — for each segment, the number of *selected*
query trigrams it contains, counted bit-sliced (independent of posting size, §10.2) — and
orders them coarsely by that count. A precision tier then reorders the top-`pool`. The
question is how deep `pool` must be. The answer falls out of the corpus's statistics.

### 1. Zipf's law on trigrams

Character-trigram document frequencies in natural text are Zipf-distributed: rank trigrams
by frequency, and the `r`-th most common has document frequency `df(r) ∝ r^(−s)` with
`s ≈ 1`. Equivalently, the fraction of the `N` segments containing a trigram `t` is
`φ(t) = df(t)/N`, and the `φ`'s are **heavy-tailed** — a few trigrams are near-universal
(`φ ≈ 1`), the vast majority are rare (`φ ≪ 1`). (trifle's `synthetic` corpus samples real
words under a Zipfian law precisely to reproduce this; real prose has it natively.)

### 2. Selection keeps the *rare* trigrams

The pruner (§6) sorts the query's trigrams rarest-first (lowest `df`) and keeps a prefix:
a rare trigram is both cheapest to scan *and* most discriminating. So the `k_sel` **selected**
trigrams have small `φ`.

### 3. A random distractor's overlap

For a non-relevant segment `d`, model its trigram membership as independent across trigrams
(a mean-field assumption — the "good statistical properties" the Zipf structure provides at
scale). It contains selected trigram `tᵢ` with probability `φᵢ = df(tᵢ)/N`, so its overlap

```
O_d = Σᵢ 1[tᵢ ∈ d]   ~  PoissonBinomial(φ₁ … φ_ksel),   mean μ = Σᵢ φᵢ.
```

### 4. The relevant segment's overlap

The relevant segment `r` shares the query's content, but a *paraphrased* query (real prose)
or a *typo'd* one drops some trigrams, so `r` carries only a fraction: `O_r = ρ·k_sel`,
`ρ ∈ (0,1]`. `O_r` is a fixed, query-dependent number; distractor overlaps are random.

### 5. The relevant segment's overlap-RANK is ∝ N

Its rank by overlap is one plus the number of distractors that beat it:

```
rank(r) = 1 + #{ d : O_d > O_r }   ⇒   E[rank(r)] = 1 + (N−1)·q  ≈  1 + N·q,
```

where `q = P(O_d > O_r)` is the per-distractor tail probability. **`q` is fixed by the
query; the expected overlap-rank of the answer grows linearly with `N`.** This is the whole
engine: a bigger index has proportionally more *accidental* high-overlap distractors burying
the true answer.

### 6. The pool must reach that rank → naively `pool ∝ N`

The reranker can only reorder what is *in* the pool, so recovering `r` needs
`pool ≥ rank(r) ∝ N`. Taken literally that is ruinous (pool scaling with the corpus). It
isn't — two effects knock the exponent below 1, but **never to a logarithm**:

- **(a) The reranker is correlated with relevance.** idf-weighting + literal verification
  rank `r` above most of the `q·N` accidental distractors (they share only *common* trigrams,
  or lack the literal query words), so `r` only needs to be *in* the pool, not near its head.
- **(b) Query heterogeneity (Zipf again).** Across queries `O_r` and the `φᵢ` vary over
  orders of magnitude (the rare-trigram tail), so `q` is itself heavy-tailed. Aggregating
  recall over that distribution, the pool for a fixed recall *fraction* scales as `N^b`,
  `b < 1`.

### 7. The law

```
              ┌ k                      (small N: the answer is already in the top-k)
p*(k, N) = max┤
              └ c · k^a · N^b           (large N: the power-law rise)
```

A **floor at `k`**, then a power-law rise. Empirically (MS MARCO) `a ≈ 0.55, b ≈ 0.41`, i.e.

```
p*(k, N) ≈ max(k, c · √(k · N)).
```

The k-dependence is *weak* (`a ≈ 0.2–0.55` across corpora) — pool depth is driven far more
by `N` than by `k`, because the dominant cost is *inclusion* (`rank ∝ N`) and `k` only sets
the final cutoff. `√(kN)` (a = b = ½) is a clean, serviceable middle; the exact exponents
are corpus-dependent (synthetic tilts N-heavy, `a≈0.2 b≈0.77`) — **same magnitude, but
calibrate per corpus, and never expect a log.**

### 8. The constant `c`, and the separate recall ceiling

`c` is not a fudge — it is exactly `c(θ) = p*_θ / √(k·N)` for a chosen recall target `θ`
(a fraction of the deep-pool recall **ceiling**). The tool measures it as the median of
`p*/√(kN)` across the `(k,N)` grid; a *tight* spread means `√(kN)` genuinely holds there.

The ceiling *itself* also falls with `N` — as accidental high-overlap distractors become
genuinely indistinguishable from `r` by the trigram signal, the reranker can't separate
them (a gentler, ~`recall ∝ const − γ·log N` degradation). That is a property of the
matching signal, *not* of the pool depth; the two are reported separately.

### 9. The `Effort` ladder

The shipped constants pin `c` to recall targets (validated by this tool, see below):

| `Effort` | `c` | target | meaning |
|----------|-----|--------|---------|
| `None`   | 0    | —    | no rerank, `pool = k` |
| `Low`    | 0.03 | p*₅₀ | ~50% of the recall ceiling |
| `Medium` | 0.05 | p*₉₀ | ~90% — **the default** |
| `High`   | 0.10 | p*₉₅ | ~95% |
| `Max`    | 0.30 | p*₉₉ | ~99% (the flat saturation tail) |

---

## What the tool does

**Sweep.** For each `N` in `--docs`, `calibrate_pool.py` invokes

```
cargo run -p trifle-benchmarks --release -- ranksweep --corpus C --docs N --queries Q --seed S
```

`ranksweep` builds the index once, then for each pool depth reranks *exactly* the top-`pool`
overlap candidates (via `Trifle::search_pool`, which pins the pool with `Effort::None` and
the explicit reranker) and prints `recall@k` for every `k ≤ pool`. One pass per pool yields
the whole `k` column. Labels: `synthetic`/`geonames-*` carry a single relevant id
(snippet/name + injected typos), `msmarco` the qrel relevant-set.

**Fit + render.** For each recall target it computes `c = p*/√(kN)` and fits
`log p* = const + a·log k + b·log N` over the rising regime (`p* > 1.3·k`), then writes:

| file | what |
|------|------|
| `matrix.csv` | the raw `N,edits,pool,k,queries,recall` measurements |
| `manifold.png` | recall vs pool, faceted by `k`, one line per `N` |
| `pstar_vs_N.png` | `p*` vs `N` (log-log) per `k` — the floor + power-law rise |
| `collapse.png` | `p*` vs the fitted predictor `k^a·N^b` — the power-law collapse |
| `summary.json` | exponents, `R²`, and `c` (with spread) at every target |

It prints a table of `c` per target and maps the shipped `Effort` constants onto the
nearest calibrated target — so if a future corpus or code change moves the constants, the
drift is visible immediately.

## Usage

```
python3 benchmarks/tools/calibrate_pool.py --corpus <corpus> [options]

  --corpus     synthetic | msmarco | geonames-cities | geonames-all   (required)
  --queries N  queries sampled per index size                         [500]
  --seed N     master seed (corpus + query sampling)                  [42]
  --edits N    typos injected (synthetic / geonames)                  [2]
  --docs       comma-separated index sizes N        [1000,5000,10000,50000,100000,500000,1000000]
  --targets    comma-separated recall-fraction-of-ceiling targets     [0.5,0.9,0.95,0.99]
  --max-pool N deepest rerank pool to sweep (raise past 2048 to push the ceiling at
               very large N)                                          [2048]
  --out DIR    output directory                                       [calibration-<corpus>]
  --reuse-csv  reuse an existing matrix.csv in --out (skip the sweep)
```

Requirements: a release build of `trifle-benchmarks` (the tool builds it on first run), and
Python with `numpy`, `pandas`, `matplotlib`. `msmarco` needs the ~1 GiB passage collection
cached (`cargo run -p trifle-benchmarks --release -- fetch --corpus relevance`).

## Caveats

- **The exponents are corpus-dependent** — `√(kN)` is the right *shape* and magnitude, but
  calibrate `c` on a corpus representative of your data. Synthetic over-weights `N`; real
  prose (MS MARCO) is close to `√(kN)`.
- **Span enough `N`.** A small corpus (e.g. `geonames-cities`, ≤34k) sits in the floor
  regime (`p* ≈ k`) and can't reveal the power law — the tool reports "floor regime" and the
  fit is meaningless. Use `msmarco` or `geonames-all` for real calibration.
- **The measured ceiling is pool-limited** at the deepest swept pool (`--max-pool`, default
  2048). At very large `N` the curves may not have fully flattened by 2048, so the
  `N`-exponent is then a slight *under*-estimate — raise `--max-pool` to push it.
- **Query difficulty is a held-fixed axis.** Harder queries (more paraphrase / fewer shared
  trigrams) push the answer deeper by overlap → a steeper `N`-dependence. `--edits` controls
  it for the synthetic/geonames corpora.

[`Bm25Ranker`]: ../../src/rank.rs
[`Effort`]: ../../src/lib.rs
