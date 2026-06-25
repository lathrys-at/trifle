# Rerank pool-law: `p* ≈ c·√(k·N)`

**Conclusion:** keep the `√(k·N)` form for the `Effort` rerank-pool knob. It is **calibrated
for and load-bearing on the prose/sparse regime** (MS MARCO), where the pool depth genuinely
scales with `√(k·N)`. For structured/dense corpora (GeoNames) the pool law is actually
`p* ∝ k` — **N-independent** — so `√(k·N)` *over-provisions*, which is safe (more pool than
needed costs latency, never recall). The shipped `Effort` constants are validated.

This is the companion to [`../tmax-sweep`](../tmax-sweep): same five-point N-ladder, same two
regimes. Where `t_max` is a *knee you find* (a fixed small cap), the pool is a *budget you
spend* (a monotone, saturating approach curve).

---

## What the pool is, and why the mean-based statistic is right here

`p` is the rerank pool depth — how many overlap candidates the precision tier rescores. Recall
rises with `p` and **saturates**: a deeper pool never hurts, it just stops helping once the
relevant doc is already in the pool. `p*(k, N)` is the smallest pool reaching a fraction of
that ceiling. Because the curve is monotone-saturating (not the non-monotone hump of `t_max`),
the per-query-knee machinery is unnecessary — mean-recall-vs-pool is the correct statistic; a
mean of a saturating curve has a well-defined `p*`.

## Method

- `ranksweep` measures recall@k vs pool depth, per N, per corpus (one index build, swept over a
  log-spaced pool grid to `--max-pool 4096`).
- `p*(k, N, target)` = smallest pool reaching `target ×` the deep-pool ceiling, for
  target ∈ {50, 90, 95, 99}%.
- Fit `p* = c · k^a · N^b` over the rising regime; `c = median(p*/√(kN))` with p10..p90 spread.
- N-ladder {1k, 5k, 25k, 125k, 625k}; k ∈ {1, 5, 10, 20, 50, 100}.
- Two corpora analyzed separately. Linear-space error; the symmetric `√(kN)` (a=b=½) is the
  deliberate non-over-parameterization the anti-correlated exponent noise calls for.

## MS MARCO (prose / sparse): `p* ∝ √(k·N)`

| target | c (med) | c (p10..p90) | fit `k^a·N^b` | R² |
|---|---|---|---|---|
| 50% | 0.040 | 0.008..0.198 | k^0.99 · N^0.00 | 0.999 |
| 90% | 0.051 | 0.039..0.192 | k^0.67 · N^0.32 | 0.924 |
| 95% | 0.122 | 0.088..0.206 | k^0.52 · N^0.44 | 0.923 |
| 99% | 0.481 | 0.267..0.751 | k^0.34 · N^0.53 | 0.931 |

The **N-exponent grows with the recall target** (0.00 → 0.32 → 0.44 → 0.53): at a low target you
just need ~`k` candidates (N-independent), but the harder-to-recover docs at high recall sit
deeper in the overlap order as N grows — more distractors accidentally out-overlap them. Around
the 90–99% targets `a` and `b` trade off near ½ each, i.e. `√(k·N)`. `c` rises from 0.04 to 0.48.
**This is the regime the shipped Effort `√(kN)` is calibrated for, and where it is load-bearing.**

![MS MARCO p* vs N](images/msmarco_pstar_vs_N.png)
![MS MARCO collapse](images/msmarco_collapse.png)

## GeoNames (structured / dense): `p* ∝ k`, N-independent

| target | c (med) | fit `k^a·N^b` | R² |
|---|---|---|---|
| 50% | 0.047 | k^1.02 · N^0.00 | 1.000 |
| 90% | 0.047 | k^1.02 · N^0.00 | 1.000 |
| 95% | 0.024 | k^0.98 · N^0.01 | 0.997 |
| 99% | 0.061 | k^0.88 · N^0.21 | 0.918 |

The exponent on N is **~0 through the 95% target** — the pool needed is just proportional to `k`,
**independent of corpus size**, only picking up a mild `N^0.21` at the 99% tail. The relevant
name is a near-exact match for the (typo'd) query — it shares the query's rare trigrams, so it
sits at the *top* of the overlap order regardless of N; you only need a pool deep enough to hold
the `k` results, not to dig past `√(kN)` accidental distractors. (The per-target `c` is small and
a touch noisy because recall saturates at a very shallow pool, which compresses the targets.)

![GeoNames p* vs N](images/geonames_pstar_vs_N.png)
![GeoNames collapse](images/geonames_collapse.png)

## The two-regime divergence

| | MS MARCO (prose / sparse) | GeoNames (structured / dense) |
|---|---|---|
| Pool law | `p* ∝ √(k·N)` (N-dependent) | `p* ∝ k` (N-independent until the 99% tail) |
| N-exponent (95%) | 0.44 | 0.01 |
| Why | more N → more accidental high-overlap distractors bury the answer | the answer is a near-exact match — top of the overlap order, N-invariant |

`√(k·N)` is the law of the **harder** (prose) regime. Using it universally **over-provisions**
the pool for easy structured corpora — a safe trade (extra latency, no recall lost). This is the
mirror image of the `t_max` finding: there the two regimes *converged* on a null (no length law);
here they *diverge* on the N-dependence of the pool — and the universal knob is correctly sized
to the harder one.

## `Effort` ladder — validation

Shipped constants in `p = max(limit, round(c·√(limit·N)))`, mapped to the MS MARCO calibration
(the regime the law is load-bearing on):

| Effort | shipped c | ≈ recall ceiling reached (prose) |
|---|---|---|
| Low | 0.03 | ~50% |
| Medium (default) | 0.05 | ~90% |
| High | 0.10 | ~95% |
| Max | 0.30 | ~95–97% |

The constants are order-right and map sensibly to recall targets. One honest caveat: **Max
(0.30) reaches ~95–97%, not a strict 99%** — the measured 99% constant is `c ≈ 0.48` (and noisy,
spread 0.27..0.75), above Max's 0.30. If a true 99%-of-ceiling tier were wanted, `c ≈ 0.45–0.5`;
but the 99% tail is where the fit is weakest and the marginal pool is most expensive, so Max at
~0.30 is a defensible stopping point. On GeoNames every Effort level over-provisions (the law is
N-independent there), so the ladder is safe across regimes.

![MS MARCO manifold](images/msmarco_manifold.png)

## Reproduce

```bash
python3 benchmarks/tools/calibrate_pool.py --corpus msmarco \
    --docs 1000,5000,25000,125000,625000 --queries 500 --max-pool 4096 --out OUT
# geonames-all: same with --corpus geonames-all --edits 2
```

Tooling: `benchmarks/src/main.rs` (`ranksweep`), `benchmarks/tools/calibrate_pool.py` (sweep
driver + `p* = c·k^a·N^b` fit + manifold / `p*`-vs-N / collapse figures). N-ladder {1k, 5k, 25k,
125k, 625k} (the standard for these sweeps; see [`../tmax-sweep`](../tmax-sweep)).
