# Rerank pool-depth characterization

`p` is the rerank pool depth: how many overlap candidates the precision tier rescores. Recall
rises with `p` and saturates — a deeper pool never lowers recall, it only stops helping once
the relevant doc is already pooled. `p*(k, N, target)` is the smallest pool reaching a given
fraction of the saturated ceiling. The question is how `p*` scales with the result count `k`
and the corpus size `N`, and therefore what shape the `Effort` knob that sets the pool should
take.

Because the recall-vs-pool curve is monotone and saturating, its mean over queries has a
well-defined knee. The per-query-knee statistic the [`t_max` sweep](../tmax-sweep) needs (for
a non-monotone hump) is unnecessary here; mean recall vs pool is the statistic. Companion to
[`../tmax-sweep`](../tmax-sweep): same `N`-ladder, same two corpora.

## Zipf's law and corpus scaling

Trigram document-frequencies are Zipfian. Rank a corpus's distinct trigrams by how many
documents contain each; the `r`-th most common has document-frequency

```
DF(r) ≈ C · N / r^s,   with s ≈ 1
```

— a few trigrams occur in nearly every document, and a long tail occur in only a handful. Two
properties of this distribution set the pool law.

First, **document-frequency scales with `N`.** The distinct-trigram vocabulary grows only
sub-linearly with the corpus (Heaps' law, `vocabulary ∝ N^β`, `β < 1`), so new documents
mostly re-use trigrams already seen and each trigram's `DF` grows roughly in proportion to `N`.
A trigram's corpus *fraction* `f = DF/N` is therefore a property of the language, fixed as the
index grows. Rare-first selection keeps the query's lowest-`f` trigrams: cheapest to scan, most
discriminating.

Second, **accidental overlap is governed by those fractions.** Candidate generation scores a
document by overlap — how many of the `t` selected query trigrams it contains. A *relevant*
document overlaps because it is what the query is looking for; a *distractor* overlaps by
accident. For a random document the accidental overlap is approximately Poisson with mean

```
μ = Σ_{i=1..t} f_i
```

the sum of the selected trigrams' fractions (variance ≈ `μ`, the matches being near-independent
and each `f_i` small). `μ` is set by the query and the language, not by `N`.

The pool depth `p*` is the relevant document's rank in this overlap order — the count of
distractors scoring at least as high. Where the relevant document falls relative to `μ` is what
decides whether that rank grows with `N`:

- **Exact match (dense corpora).** A relevant name carries the query's *rare* trigrams, so its
  overlap sits far above `μ`, out in the upper tail. The number of distractors able to
  reproduce that rare-trigram signature is about `N · ∏ f_i` over those trigrams — a product of
  small fractions, negligible and not growing with `N` (each `f_i` is fixed). The pool need only
  be deep enough to hold the `k` results: `p* ∝ k`, independent of corpus size.
- **Paraphrase (sparse corpora).** A relevant passage re-uses only a few of the query's exact
  trigrams, so its overlap is modest and sits inside the bulk of the accidental distribution,
  not above it. The distractors that match or beat that overlap are a roughly fixed fraction of
  the corpus, so their count rises with `N`, burying the relevant document deeper as the index
  grows. The pool must grow with `N` to keep it in.

How fast? An independence model — documents i.i.d., the relevant one fixed at overlap `θ`,
distractors counted above `θ` — gives `p* ≈ N · P(S ≥ θ)`, linear in `N`. Real text is bursty:
trigrams co-occur within documents rather than landing independently, which cuts the effective
number of independent distractors and pulls the growth below that line. The fitted exponent on
`N` (next sections) is sub-linear, climbs with the recall target, and passes through ≈ ½ across
the 90–99% band production targets. The symmetric `√(k·N)` (`a = b = ½`) is the form adopted:
the geometric mean of the pool's two pressures — a floor of `k` results to return, and the
`N`-driven burial above it — and the parsimonious centre of fitted exponents that are
individually noisy and anti-correlated.

## Method

- `ranksweep` measures recall@k vs pool depth, per `N`, per corpus — one index build per `N`,
  swept over a log-spaced pool grid to `--max-pool 4096`.
- `p*(k, N, target)` = smallest pool reaching `target ×` the deep-pool ceiling, for
  target ∈ {50, 90, 95, 99}%.
- Fit `p* = c · k^a · N^b` over the rising regime; `c = median(p*/√(kN))`, with p10..p90
  spread.
- `N`-ladder {1k, 5k, 25k, 125k, 625k}; k ∈ {1, 5, 10, 20, 50, 100}.
- Errors measured in linear space; both the free `k^a·N^b` fit and the symmetric `√(kN)`
  (`a = b = ½`) are reported. The rationale for the ½ exponent is above.

## MS MARCO (prose, sparse): `p* ∝ √(k·N)`

| target | c (med) | c (p10..p90) | fit `k^a·N^b` | R² |
|---|---|---|---|---|
| 50% | 0.040 | 0.008..0.198 | k^0.99 · N^0.00 | 0.999 |
| 90% | 0.051 | 0.039..0.192 | k^0.67 · N^0.32 | 0.924 |
| 95% | 0.122 | 0.088..0.206 | k^0.52 · N^0.44 | 0.923 |
| 99% | 0.481 | 0.267..0.751 | k^0.34 · N^0.53 | 0.931 |

The `N`-exponent grows with the recall target (0.00 → 0.32 → 0.44 → 0.53). At a low target the
pool need only hold ~`k` candidates, `N`-independent; the harder-to-recover docs at high
recall sit deeper in the overlap order as `N` grows, because more distractors accidentally
out-overlap them. Around the 90–99% targets `a` and `b` settle near ½ each — that is `√(k·N)`.
`c` rises from 0.04 to 0.48 across the targets.

![MS MARCO p* vs N](images/msmarco_pstar_vs_N.png)
![MS MARCO collapse](images/msmarco_collapse.png)

## GeoNames (structured names, dense): `p* ∝ k`, `N`-independent

| target | c (med) | fit `k^a·N^b` | R² |
|---|---|---|---|
| 50% | 0.047 | k^1.02 · N^0.00 | 1.000 |
| 90% | 0.047 | k^1.02 · N^0.00 | 1.000 |
| 95% | 0.024 | k^0.98 · N^0.01 | 0.997 |
| 99% | 0.061 | k^0.88 · N^0.21 | 0.918 |

The `N`-exponent is ~0 through the 95% target: the pool need only scale with `k`, independent
of corpus size, picking up a mild `N^0.21` only at the 99% tail. A relevant name is a
near-exact match for the (typo'd) query and shares its rare trigrams, so it sits at the top of
the overlap order regardless of `N`; the pool need only be deep enough to hold the `k`
results, not to dig past accidental distractors. Each per-target `c` is small and a little
noisy because recall saturates at a very shallow pool, which compresses the targets together.

![GeoNames p* vs N](images/geonames_pstar_vs_N.png)
![GeoNames collapse](images/geonames_collapse.png)

## Two regimes

| | MS MARCO (prose / sparse) | GeoNames (structured / dense) |
|---|---|---|
| Pool law | `p* ∝ √(k·N)` (N-dependent) | `p* ∝ k` (N-independent until the 99% tail) |
| N-exponent (95%) | 0.44 | 0.01 |
| Why | more N → more accidental high-overlap distractors bury the answer | the answer is near-exact: top of the overlap order, N-invariant |

`√(k·N)` is the law of the harder, prose regime. On a dense structured corpus the same form
over-provisions the pool — a safe error, costing latency but never recall. Where the
[`t_max`](../tmax-sweep) sweep found the two regimes converging on a null, the pool law has
them diverging on `N`-dependence; in both cases a universal knob sized to the harder regime is
correct.

## Effort ladder

Shipped constants in `p = max(limit, round(c·√(limit·N)))`, against the MS MARCO calibration —
the regime where the `√(kN)` form is load-bearing:

| Effort | shipped c | ≈ recall ceiling reached (prose) |
|---|---|---|
| Low | 0.03 | ~50% |
| Medium (default) | 0.05 | ~90% |
| High | 0.10 | ~95% |
| Max | 0.30 | ~95–97% |

The constants are order-right and map sensibly onto recall targets. One caveat: Max (0.30)
reaches ~95–97% of the ceiling, not a strict 99% — the measured 99% constant is `c ≈ 0.48`
(and noisy, spread 0.27..0.75), above Max's 0.30. A true 99%-of-ceiling tier would want
`c ≈ 0.45–0.5`, but the 99% tail is where the fit is weakest and the marginal pool most
expensive, so stopping Max at 0.30 is defensible. On GeoNames every level over-provisions, so
the ladder is safe across both regimes.

![MS MARCO manifold](images/msmarco_manifold.png)
![GeoNames manifold](images/geonames_manifold.png)

## Conclusion

The pool depth needed for a recall target scales as `√(k·N)` on prose/sparse corpora and as
`k` alone on structured/dense ones. A single `√(k·N)` knob is therefore correctly shaped: it
is doing real work where the law is `√(k·N)`, and over-provisions safely where the law is `k`.
The shipped `Effort` constants — Low 0.03, Medium 0.05, High 0.10, Max 0.30 — track the
50/90/95/95–97% recall targets and need no change, with the one documented limit that Max
falls short of a strict 99%.

## Reproduce

```bash
python3 benchmarks/tools/calibrate_pool.py --corpus msmarco \
    --docs 1000,5000,25000,125000,625000 --queries 500 --max-pool 4096 --out OUT
# geonames-all: same with --corpus geonames-all --edits 2
```

Tooling: `benchmarks/src/main.rs` (`ranksweep`), `benchmarks/tools/calibrate_pool.py` (sweep
driver + `p* = c·k^a·N^b` fit + manifold / `p*`-vs-N / collapse figures). `N`-ladder {1k, 5k,
25k, 125k, 625k} is the standard for these sweeps; see [`../tmax-sweep`](../tmax-sweep).
