# `t_max` selection-cap characterization

`t_max` caps how many query trigrams the rarest-first selector keeps. Selection keeps at
least the typo floor `F = m + d = 6` and at most `t_max`. Raising the cap lets more
candidates clear the overlap floor and enter the rerank pool, but the trigrams added past the
rare anchors are common (high document-frequency): they cost scan latency and inject overlap
noise. The cap therefore trades recall against latency, and three questions decide where it
belongs and whether it must scale with anything:

- does the right cap grow with query length?
- does it drift with corpus size `N`?
- is a cap above the floor worth its latency cost at all?

The sweep answers all three across two opposite corpora — MS MARCO (prose, sparse) and
GeoNames all-countries (structured names, dense) — and `N` from 1k to 1M.

## Method

Recall is measured per query, as a knee, not as mean-recall-vs-`t_max`. Averaging binary
recall over queries whose individual knees differ produces a smooth plateau where each query
in fact has a sharp knee, and an uneven length distribution can turn that artifact into a
spurious length law. The per-query statistics avoid both:

- `t_enter(q, k)` — smallest `t_max` that recovers `q`'s relevant doc into top-k.
- `t_exit(q, k)` — largest `t_max` that still holds it there (the per-query hump).
- Queries never recovered at any `t_max` are right-censored: reported as a recovery rate, not
  dropped, since dropping them biases toward easy queries.

Two pool depths separate the two costs `t_max` controls:

- A **generous** pool (`2·√(50·N)`, capped) isolates the selection ceiling — the relevant doc
  is in the pool regardless of `t_max`, so only selection quality varies.
- A **production** pool (`Effort::Medium`, `0.05·√(50·N)`) is the realistic operating point.
  Here `t_max` also decides whether the doc reaches the small pool at all, so it carries the
  real value.

`N`-ladder {1k, 5k, 25k, 125k, 625k}, plus a 1M anchor (one point, checked against a 2×-pool
adequacy test: doubling the pool moved the 1M ceiling by ±0.007). `t_max` grid dense over
6–16, anchors at 20 and 28. Effect sizes carry bootstrap CIs, in linear space.

## Length dependence

Pooled entry-knee-vs-length slope (`t_max` per trigram), 95% CI:

| | MS MARCO | GeoNames |
|---|---|---|
| k=10 | `+0.010 [+0.005, +0.017]` (span +0.8 `t_max`) | `+0.005 [+0.001, +0.009]` (span +0.2) |
| k=50 | `+0.010 [+0.004, +0.016]` (span +0.8) | `+0.001 [-0.001, +0.004]` — **no effect** |

The slope is positive — a tail of long queries pulls the mean up — but the implied knee
movement across the entire length range is under 1 `t_max`, too small to act on. On GeoNames at
k=50 there is no length effect at all, and the data are ample enough to detect a small one.
Neither regime has a length law.

## Drift with corpus size

Median `t_enter@k=10` by (length bucket × `N`) sits at the floor (6) in every cell, both
corpora, for every `N` from 1k to 1M:

```
MS MARCO    len 10-13  14-18  19-25  26-35  36-50  51+
  N=1k          6      6      6      6      6      6
  N=625k        6      6      6      6      6      6
  N=1M          6      6      6      6      6      6
GeoNames    len 4-6  7-9  10-13  14-18  19-25  26-35  36-50
  N=1k..1M    6    6     6      6      6      6      6     (every cell)
```

The typical query recovers at the floor regardless of length or corpus size. The structural
prediction that large `N` should want *fewer* trigrams surfaces only in the tail (the hump,
below), never as a moving median.

![Per-query entry-knee, MS MARCO](images/entryknee_msmarco.png)
![Per-query entry-knee, GeoNames](images/entryknee_geonames.png)

*Boxes pinned at the floor with an upper tail that grows with query length: the median is the
floor; what little length dependence there is sits in the tail.*

## Value above the floor

Recall@50 at the production pool, floor (`t=6`) vs the best `t_max`:

| N | MS MARCO gain (best `t`) @ latency× | GeoNames gain (best `t`) @ latency× |
|---|---|---|
| 1k | +0.016 (11) @3.5× | +0.002 (7) @1.0× |
| 25k | +0.034 (16) @1.9× | +0.058 (9) @1.4× |
| 125k | +0.048 (13) @1.6× | +0.105 (11) @2.9× |
| 625k | +0.038 (14) @1.8× | **+0.173 (12) @2.8×** |
| 1M | +0.067 (10) @1.4× | +0.224 (14) @4.2× (sub-ms) |

A cap above the floor is the largest single recall effect in either report, and the gain grows
with `N`: on GeoNames it reaches +0.17 recall@50 at 625k and +0.22 at 1M. The best `t_max` itself
does not move — it stays a small fixed cap (~8–16) in both regimes at every `N`. What grows with
the corpus is the recall that cap is worth, not its position. Latency cost is modest (1–4×,
sub-millisecond for GeoNames). The gain shows only at the production pool: at the generous pool
the relevant doc is already in the pool, so `t_max` changes recall by ~2 points at a misleading
12–26× latency. At the smaller production pool, `t_max` decides whether that doc is in the pool
at all.

![Above-floor gain vs N](images/qvalue_vs_N.png)

## The hump

Per-query drop-out fraction — recovered into top-k, then demoted by pushing `t_max` to 28 —
grows with `N` on prose and stays near zero on structured names:

| drop-out @k=10 | N=1k | N=125k | N=625k | N=1M |
|---|---|---|---|---|
| MS MARCO | 0.018 | 0.105 | 0.107 | 0.148 |
| GeoNames | 0.000 | 0.005 | 0.012 | 0.022 |

Prose's trigram frequencies are Zipfian with a thick body of common, high-`DF` trigrams;
raising `t_max` past the rare anchors reaches into that body and injects accidental overlap
that demotes the true doc — a cost that grows with `N`. Structured names have a sparser body
and few such collisions. Both regimes point the same way: a cap above the optimum only loses
recall.

The hump also gives independent support to the pool-depth model. The
[pool-depth report](../pool-law) argues from Zipf's law that prose's thick body of common
trigrams should bury relevant documents deeper as `N` grows; the hump is that burial, measured
here by a different statistic (per-query drop-out) on a separate sweep. The two-regime split
therefore shows up in two unrelated measurements, not just one fit.

![Recall ceiling vs t_max, MS MARCO](images/ceiling_msmarco.png)
![Recall ceiling vs t_max, GeoNames](images/ceiling_geonames.png)
![Recall@1 hump vs t_max, MS MARCO](images/hump_msmarco.png)
![Recall@1 hump vs t_max, GeoNames](images/hump_geonames.png)

## Recovery

Recovery rate falls with `N` as the task gets harder, reported separately from the knee
distribution:

| recall-ever @50 | 1k | 25k | 125k | 625k | 1M |
|---|---|---|---|---|---|
| MS MARCO | 0.992 | 0.966 | 0.904 | 0.802 | 0.770 |
| GeoNames | 0.856 | 0.820 | 0.813 | 0.761 | 0.737 |

GeoNames starts *below* MS MARCO at small `N` (0.856 vs 0.992 at 1k) — the dense corpus
recovers worse where the task should be easiest. The cause is query construction, not the
index. GeoNames queries are short names with two injected edits, and two edits corrupt a far
larger *fraction* of a short name's trigrams than of a long passage: a 6-character name carries
~4 trigrams, of which two edits can damage most, whereas the same two edits touch a handful of
a passage's dozens. For a baseline ~14% of GeoNames queries the surviving rare-trigram signature
is too degraded to recover at any `t_max` (or it now resolves to a different real name), so they
are unrecoverable independent of `N`. MS MARCO's queries are real paraphrases, almost all
recoverable at 1k. The gap is a property of edit injection on short strings.

## Two regimes

| | MS MARCO (prose / sparse) | GeoNames (structured / dense) |
|---|---|---|
| Median entry-knee (all N ≤ 1M) | floor (6) | floor (6) |
| Length law @k=50 | tiny (`+0.010`, span +0.8) | null (`+0.001`) |
| Hump (drop@10) at 1M | 0.148 (large) | 0.022 (tiny) |
| Above-floor gain @ production pool, 625k | +0.048 | +0.173 |
| Best fixed `t_max` | ~8–16 | ~7–12 |

The two opposite regimes converge on one shape: a small fixed optimum, no length law, and no
drift in the median. The length dependence that a mean-recall statistic reports is an
aggregation artifact; the per-query knee removes it.

## Conclusion

`t_max` is a high-value parameter. Its above-floor gain is the largest single recall effect in
either report and grows with `N`, up to +0.22 recall@50 at 1M on GeoNames. The best cap does not
move with `N`: it stays a small fixed value (~8–16) in both regimes at every corpus size. What
changes with the corpus is how much recall the cap is worth.

Two views hide this. The median query recovers at the floor in every cell, and at the generous
diagnostic pool the relevant doc is already in the pool, so `t_max` changes recall by only ~2
points; both make the cap look almost irrelevant. It is not — the gain shows at the smaller
production pool, where the cap decides whether the relevant doc is in the pool at all.

| question | finding | implication |
|---|---|---|
| length | real but < 1 `t_max` across the length range | no length-scaled `t_max` |
| drift with `N` | median knee = floor at every `N` to 1M | no `t_max(N)` rule |
| value | large, and grows with `N` (up to +0.22 at 1M, dense) | a fixed cap captures it without per-`N` tuning |
| effort coupling | dropping to the floor costs up to +0.17–0.22 recall (GeoNames) | effort must not lower it |

A fixed `t_max = 12` captures it. Twelve sits at or above the regime optima at every `N` (MS
MARCO ~10–16, GeoNames ~7–14): it takes the large dense-regime gain nearly in full, stays within
~1 recall point of the MS MARCO optimum, and falls below where the prose hump bites — with no
per-length or per-`N` rule. The largest gains are in the dense regime — short, structured records
rather than long prose, which is the kind of corpus trifle is for. Cutting the cap back
toward the floor at a low `Effort` gives up to +0.17–0.22 of recall back, so `Effort` must leave
`t_max` alone.

## Reproduce

```bash
# generous pass (ceiling + per-query knees) and realistic pass (above-floor value), per corpus:
python3 benchmarks/tools/tmax_knee.py --corpus msmarco \
    --docs 1000,5000,25000,125000,625000 --queries 500 --max-tmax 28 --out OUT-gen
python3 benchmarks/tools/tmax_knee.py --corpus msmarco \
    --docs 1000,5000,25000,125000,625000 --queries 500 --max-tmax 28 \
    --pool-coef 0.05 --pool-floor 50 --out OUT-real
python3 benchmarks/tools/tmax_perquery.py --csv OUT-gen/tmax_raw.csv --out OUT-gen/perquery
# (geonames-all: same with --corpus geonames-all --edits 2)
```

Tooling: `benchmarks/src/main.rs` (`tmaxsweep`), `benchmarks/tools/tmax_knee.py` (sweep
driver, peak-based facets), `benchmarks/tools/tmax_perquery.py` (the per-query knee
statistic). `N`-ladder {1k, 5k, 25k, 125k, 625k} is the standard for these sweeps.
