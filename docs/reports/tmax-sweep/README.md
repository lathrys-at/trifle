# `t_max` selection-cap characterization

**Conclusion:** ship a **fixed small `t_max` ≈ 12** (already the default) — **no per-length,
per-N, or effort coupling.** The value is real (above-floor `t_max` buys recall, increasingly
at scale), but it saturates at a small fixed cap that does not move with query length or
corpus size. Effort must **not** lower it. This is validated across two opposite corpora
(prose/sparse and structured/dense) and N from 1k to 1M.

---

## What `t_max` is

`t_max` is the cap on how many query trigrams the rarest-first pruner keeps (the renamed
`k_max`). Selection always keeps at least the typo floor `F = m + d = 6`, and at most `t_max`.
More kept trigrams → more candidates clear the overlap floor and enter the rerank pool; but
the trigrams added past the rare anchors are *common* (high-DF), so they cost latency and
inject overlap noise. The question: where does `t_max` belong, and does it need to scale with
anything?

## Method

- **Per-query entry/exit-knee distribution**, not mean-recall-vs-`t_max`. Averaging binary
  recall over queries with *different individual knees* manufactures a flat plateau even when
  every query has a sharp knee — and it manufactured an apparent length law on GeoNames that
  the per-query statistic shows is an artifact (below). `t_enter(q,k)` = smallest `t_max`
  recovering `q`'s relevant doc into top-k; `t_exit(q,k)` = the largest such `t_max` (the
  per-query hump). Never-recovered queries are **right-censored** (reported as recovery rate,
  not dropped).
- **Two pool passes.** *Generous* pool (`2·√(50·N)`, capped) isolates the selection ceiling —
  the relevant doc is in the pool regardless of `t_max`, so only selection matters. *Realistic*
  pool (`Effort::Medium`, `0.05·√(50·N)`) is honest production cost — there `t_max` decides
  whether the doc gets *into* the small pool, so it carries the real value.
- **N-ladder** {1k, 5k, 25k, 125k, 625k}, plus a 1M anchor (single point, validated against a
  2×-pool adequacy check: doubling the pool moved the 1M ceiling by ±0.007).
- **Two corpora, analyzed separately:** MS MARCO (real dev queries + qrels — long, paraphrase,
  sparse) and GeoNames all-countries (name + injected edits — short, structured, dense).
- **`t_max` grid:** dense 6–16, anchors 20, 28. Effect sizes with **bootstrap CIs**, linear
  space (never log-log R²).

## The three questions

### Q-EXIST — does the knee depend on query length? *Real but immaterial.*

Pooled entry-knee-vs-length slope (t_max per trigram), 95% CI:

| | MS MARCO | GeoNames |
|---|---|---|
| k=10 | `+0.010 [+0.005, +0.017]` (span +0.8 `t_max`) | `+0.005 [+0.001, +0.009]` (span +0.2) |
| k=50 | `+0.010 [+0.004, +0.016]` (span +0.8) | `+0.001 [-0.001, +0.004]` — **powered null** |

The slope is positive (the mean is pulled by a tail of long queries), but the **implied knee
movement over the entire length range is < 1 `t_max`** — far below any shippable threshold.
On GeoNames at the ceiling it is a clean null. No length law worth shipping in either regime.

### Q-DRIFT — does the optimum drift with N? *No — the median knee is the floor everywhere.*

Median `t_enter@k=10` by (length bucket × N) is **the floor (6) in every cell**, both corpora,
for every N from 1k to **1M**:

```
MS MARCO    len 10-13  14-18  19-25  26-35  36-50  51+
  N=1k          6      6      6      6      6      6
  N=625k        6      6      6      6      6      6
  N=1M          6      6      6      6      6      6
GeoNames    len 4-6  7-9  10-13  14-18  19-25  26-35  36-50
  N=1k..1M    6    6     6      6      6      6      6     (every cell)
```

The typical query recovers at the floor regardless of length or corpus size. There is no
`t_max(N)` rule — the structural prediction that large N wants *fewer* trigrams shows up only
as a tail effect (the hump, below), not as a moving median.

![Per-query entry-knee, MS MARCO](images/entryknee_msmarco.png)
![Per-query entry-knee, GeoNames](images/entryknee_geonames.png)

*Boxes pinned at the floor with a length-growing upper tail — the median is floor; the tiny
length law lives in the tail.*

### Q-VALUE — is above-floor `t_max` worth its honest cost? *Yes, and it grows with N.*

Recall@50 at the **realistic (production) pool**, floor (`t=6`) vs best `t_max`:

| N | MS MARCO gain (best `t`) @ latency× | GeoNames gain (best `t`) @ latency× |
|---|---|---|
| 1k | +0.016 (11) @3.5× | +0.002 (7) @1.0× |
| 25k | +0.034 (16) @1.9× | +0.058 (9) @1.4× |
| 125k | +0.048 (13) @1.6× | +0.105 (11) @2.9× |
| 625k | +0.038 (14) @1.8× | **+0.173 (12) @2.8×** |
| 1M | +0.067 (10) @1.4× | +0.224 (14) @4.2× (sub-ms) |

Above-floor `t_max` is materially valuable — and the value **grows with N**, dramatically so
for GeoNames (+0.2 recall@50 at scale). The best `t_max` is a small fixed cap (~8–16), latency
cost modest (1–4×; sub-millisecond in absolute terms for GeoNames). This is the number that
decides the design — and it required the realistic pool: at the *generous* pool the relevant
doc is already in the pool, so `t_max` looked marginal (~2 points) at a misleading 12–26×.

![Q-VALUE vs N](images/qvalue_vs_N.png)

## The hump — the regime difference (and why "don't go higher")

Per-query drop-out fraction (recovered into top-k, then demoted by `t_max=28`) **grows with N
on prose, stays tiny on structured names**:

| drop-out @k=10 | N=1k | N=125k | N=625k | N=1M |
|---|---|---|---|---|
| MS MARCO | 0.018 | 0.105 | 0.107 | 0.148 |
| GeoNames | 0.000 | 0.005 | 0.012 | 0.022 |

Prose has many common trigrams to add past the rare anchors, so over-selecting injects
distractor noise that demotes the true doc — a real, N-growing cost. Structured names have few
such collisions. Either way it argues the same direction: **do not raise `t_max` past the
optimum.**

![Recall ceiling vs t_max, MS MARCO](images/ceiling_msmarco.png)
![Recall@1 hump vs t_max, MS MARCO](images/hump_msmarco.png)
![Recall@1 hump vs t_max, GeoNames](images/hump_geonames.png)

## Recovery (censoring)

Recovery rate falls with N (the task gets harder), reported separately from the knee
distribution — never dropped, since dropping biases toward easy queries:

| recall-ever @50 | 1k | 25k | 125k | 625k | 1M |
|---|---|---|---|---|---|
| MS MARCO | 0.992 | 0.966 | 0.904 | 0.802 | 0.770 |
| GeoNames | 0.856 | 0.820 | 0.813 | 0.761 | 0.737 |

## Two-regime summary

| | MS MARCO (prose / sparse) | GeoNames (structured / dense) |
|---|---|---|
| Median entry-knee (all N ≤ 1M) | floor (6) | floor (6) |
| Length law (Q-EXIST) @k=50 | tiny (`+0.010`, span +0.8) | null (`+0.001`) |
| Hump (drop@10) at 1M | 0.148 (large) | 0.022 (tiny) |
| Q-VALUE @ realistic pool, 625k | +0.048 | +0.173 |
| Best fixed `t_max` | ~8–16 | ~7–12 |

The two opposite regimes **converge** on the same shape: a small fixed optimum, no length law,
no N-drift in the median. The early mean-based GeoNames "length law" (`knee ≈ 0.5·len + 2.3`)
was an aggregation artifact — the per-query statistic dissolves it.

## Decision

| question | result | implies |
|---|---|---|
| Q-EXIST | real but < 1 `t_max` over the length range | **no length-scaled `t_max`** |
| Q-DRIFT | median knee = floor at every N incl. 1M | **no `t_max(N)` rule** |
| Q-VALUE | material at a fixed small `t_max` (~8–16), grows with N | **ship a fixed small `t_max`** |
| Effort × `t_max` | lowering to the floor costs up to +0.17 recall (GeoNames 625k) | **effort must not lower it** |

→ **Fixed `t_max = 12`** — the current default. It sits between the regime optima (MS MARCO
~10–16, GeoNames ~7–14), captures GeoNames' large at-scale value nearly fully, stays MS MARCO
within ~1 point, and is below where the prose hump bites. The median query is already recovered
at the floor; `t_max = 12` serves the hard tail and the small production pool without demoting
the median. **No engine change is warranted — Phase 1 already shipped the right value.**

## Reproduce

```bash
# generous pass (ceiling + per-query knees) and realistic pass (Q-VALUE), per corpus:
python3 benchmarks/tools/tmax_knee.py --corpus msmarco \
    --docs 1000,5000,25000,125000,625000 --queries 500 --max-tmax 28 --out OUT-gen
python3 benchmarks/tools/tmax_knee.py --corpus msmarco \
    --docs 1000,5000,25000,125000,625000 --queries 500 --max-tmax 28 \
    --pool-coef 0.05 --pool-floor 50 --out OUT-real
python3 benchmarks/tools/tmax_perquery.py --csv OUT-gen/tmax_raw.csv --out OUT-gen/perquery
# (geonames-all: same with --corpus geonames-all --edits 2)
```

Tooling: `benchmarks/src/main.rs` (`tmaxsweep`), `benchmarks/tools/tmax_knee.py` (sweep driver,
peak-based facets), `benchmarks/tools/tmax_perquery.py` (the per-query knee statistic). N-ladder
{1k, 5k, 25k, 125k, 625k} is the standard for these sweeps.
