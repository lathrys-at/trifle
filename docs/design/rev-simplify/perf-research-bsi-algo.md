# BSI overlap engine — algorithmic optimization research

Scope: the **counting/enumeration math** of `trifle-overlap` (the single-BSI candidate
generator). Lane excludes SIMD/vectorization (systems agent) and selection policy (`select.rs`).
This is analysis + complexity, validated with throwaway probes under `/tmp/bsi-probe` (path-dep on
the crate; not committed).

## TL;DR — the result is mostly negative, and that is the finding

**The engine is build-bound, and the build is already near-optimal.** Every optimization in this
lane (ordered enumeration, floor-prefiltering, the raw-overlap shortcut) targets the *walk*, which
the probes show is **0–23 % of per-query engine time (usually < 10 %)** and structurally bounded by
`max_score ≤ 4k ≤ ~48` buckets. The one lever on the dominant **build** cost — fusing the
half-adder's AND+XOR into one container pass — is real (~1.3–1.5×) but **not expressible over
`roaring` 0.11's public API** without a fork/upstream PR, and overlaps the SIMD lane. There is **no
scalar, pure-Rust-on-stable algorithmic build speedup available in this lane.** The high-leverage
knobs live in *other* lanes: selection (Σdf of selected postings) and SIMD/croaring.

## Cost model (measured, not assumed)

Build folds `k` postings into bit-sliced planes; each posting of weight `w∈{1,2,3,4}` costs
`popcount(w) ≤ 2` ripple injections, each injection ≈ **1 full-width AND (carry) + 1 full-width XOR
(sum)** at its plane, plus carry ripples over the (small) intersections. So:

- **Op count** is cardinality-independent (`O(k·log k)` ops). **Wall-clock is `Ω(Σ set bits)`** —
  the XOR must touch every set bit of every posting at least once to count it. The BSI achieves
  this bound: build ≈ linear in `Σ` selected cardinality (probe: 6k→96k set bits scaled build
  57µs→1565µs).
- **`max_score ≤ Σweights ≤ 4k ≤ ~48`** ⇒ `P = ⌈log2(max_score+1)⌉ ≤ 6` planes, and **≤ ~48 score
  buckets total**. The walk is therefore *structurally* a small constant of `count_eq` calls.

Probe data (`k=12`, top-10, floor=2):

| regime | build µs | walk µs | walk % | count_eq calls | empty buckets | raw probes |
|---|---|---|---|---|---|---|
| planted head, card 200 | 57 | 11 | 16 % | 1 | 0 | 120 |
| planted head, card 3200 | 1565 | 1.7 | 0.1 % | 1 | 0 | 120 |
| sparse rare-gram card 20 | 22 | 1.2 | 5 % | 9 | 0 | 120 |
| sparse, high-overlap card 60/head 40 | 19.5 | 5.9 | 23 % | 3 | 0 | 120 |
| **deep pull / full drain** (37 results) | — | — | — | **11** | 0 | — |

Two facts kill the walk-side optimizations:

1. **`raw_overlap` is already free.** It is paid only per *yielded* id (`k` `contains` probes),
   so top-10 = 120 probes ≈ **1.25 µs total** (< 2 % of build, → 0 % as cardinality grows).
2. **The walk is bounded.** Even a *full drain* costs ~11 `count_eq` calls / 33 plane ops, because
   there are only ~48 buckets. Empty reachable-bucket probes are rare (0–3) and cheap (high
   buckets have tiny planes).

Weight collapse is common: in the sparse rarest-first regime the selected postings have a narrow
df range, so `tier_weights` returns **all-1 weights** in most probed cases (`w={1}`); a `{1,2,3}`
or `{1,2,3,4}` spread needs a ≥4× df gap among the selected (rare) tokens.

---

## The five questions

### Q1 — Ordered MSB→LSB enumeration to replace per-bucket `count_eq`

**Mechanism.** Instead of calling `count_eq(c)` per bucket (each re-ANDs the set-bit planes and
ANDNOTs the clear-bit planes from scratch), partition the candidate set down a binary trie over
score bits MSB→LSB: at node `(set, b)` split `ones = set ∩ plane_b`, `zeros = set − plane_b`,
recurse `ones` (higher) before `zeros` (lower); leaves at `b<0` are exact-score groups in
descending order. Each id is touched once per level ⇒ **`P·|candidates|` element-work for a full
enumeration**, vs `count_eq`'s `Σ_buckets P·|plane|`; and it never probes an empty bucket
(only non-empty branches are pushed). Prefix work (the high-plane intersections) is shared across
all buckets in a subtree.

**Correctness (the "subtle" part, worked out — it *is* correct).**
- *Raw floor* stays a per-yielded-id `raw_overlap ≥ floor` check; weighted-score grouping is
  unchanged, so the rare-gram-singleton case (weighted ≥ floor, raw < floor) is still rejected
  exactly as today. Subtree prune: drop `(set,b)` when `acc + (2^{b+1}−1) < floor`.
- *Dedup + determinism*: leaves are roaring bitmaps iterated ascending — identical within-score id
  order to today's `bucket.iter()`, so the trifle-layer dedup-by-key tie resolution is unchanged.
- *Early-stop*: stop popping once `limit` is filled; **defer `zeros = set − plane_b` lazily**
  (compute only when the lower branch is actually popped) so an unvisited low subtree pays no
  ANDNOT. This preserves the laziness `advance` relies on.

**Win in trifle's regime.** Targets the walk (0–23 %, usually <10 %) which is already bounded by
~48 buckets and ~11 `count_eq` even at full drain. Expected **total-query win < 5 %, typically
< 1 %.** The advantage (prefix sharing, no empty probes, `P·|cand|` vs per-bucket) only matters
when there are *many large* buckets — i.e. a **common token in the selection** (large planes) **and
a deep pull**, neither of which is the design's operating point.

**Effort M** (stack/lazy-thunk management, the floor/dedup/early-stop interactions).
**Risk M** (re-derives a load-bearing, well-tested path).
**Verdict: not-worth-it.** The deferral in the proposal was correct. Revisit *only* if a profiler
shows large-plane + deep-pull queries dominating — **benchmark-gated, low priority.**

### Q2 — Floor-prefiltering (restrict the BSI to raw-overlap ≥ floor before/while building)

**Mechanism / cost.** The "≥ floor" eligible set is itself a threshold-count — the very problem the
BSI solves. Computing it needs either an *unweighted* BSI build (~2k full-width ops) or a saturating
seen-`f` counter (`floor=2`: `seen2 |= seen1 & p; seen1 |= p` ⇒ ~3k ops), **then** `k` ANDs to
restrict the postings, **then** the weighted build over the smaller postings. That **adds** ~3–4k
full-width ops to shrink a walk that is already < 23 % of cost. Since **build dominates**, adding
build work to save walk work is **net-negative**. And `floor=1` queries (the typo-floor /
short-query common case) get *nothing* — every candidate is eligible.

**Effort M, Risk M (a second counting pass with its own carry logic).
Verdict: not-worth-it.**

### Q3 — Fused half-adder (one container pass producing both sum and carry)

**Mechanism.** `add_weighted`'s hot first level does `carry = &acc[s] & posting` then
`acc[s] ^= posting` — **two** passes over the same containers. A fused half-adder walks the
container/word pair **once**, emitting `sum = a^b` and `carry = a&b` together. This is the *only*
idea in this lane that hits the **dominant build cost**: ~**1.3–1.5× on build → ~1.3–1.5× on total
query**.

**Expressibility — blocked.** `roaring` 0.11 exposes only whole-bitmap `&`/`^` (each a full pass)
and a per-*element* iterator (far slower); its container `Vec`/word arrays are **private**. A true
fused AND+XOR therefore **requires forking `roaring`** or an **upstream PR** (a `bitand_bitxor`
fused op, or word-level container access), and post-XOR the carry is destroyed so it can't be
recovered cheaply in a second pass. The croaring opt-in backend (separate, accepted) is the other
route. This also substantially overlaps the **SIMD/systems lane** (a fused word loop is naturally
vectorized).

**Effort L (fork/upstream), Risk M.
Verdict: worth-effort but BLOCKED in pure-Rust-stable default** — route via an upstream `roaring`
fused-op request or croaring/SIMD. **Benchmark to confirm the 1.5×** before investing. The most
*valuable direction*, but not actionable in this lane today.

### Q4 — Reducing the number of bitmap ops in build

**Is ripple-carry op-optimal? Yes.** Counting `k` bitmaps is `Θ(k)` whole-bitmap ops; ripple-carry
achieves it (amortized total carry ops ≤ k). A **Wallace / carry-save tree** has the **same
single-thread total work** — it only shortens the *critical path* (depth `log k`), which buys
nothing for a scalar serial implementation. (It *does* expose ILP/SIMD parallelism — see the
cross-lane handoff; that is the systems agent's win, not a scalar one.) And build is `Ω(Σ set
bits)`, which the BSI already meets — **there is no asymptotic build speedup.**

**Sub-idea A — uniform-weight raw-overlap shortcut (do-now, but ~0 gain).** When weights collapse
to a single value `w` (the *common* case — probes show `w={1}`), `raw_overlap = score / w` exactly,
so the per-yielded-id `contains` probes can be dropped and the walk floor tightened from `floor` to
`w·floor` (skip more low buckets). **Correct and cheap (effort S),** but `raw_overlap` is already
< 2 % of cost, so the measurable win is ~nil. Keep as an optional micro-cleanup only.

**Sub-idea B — separate unweighted-count from weighted-score (the "dual-BSI").** Building a second
unweighted BSI to read raw overlap in `O(log k)` is *strictly more* build work for a raw-overlap
that is already free; probes show even deep pulls don't make `contains` expensive (`k` small).
**Not worth it** (matches the proposal's §11 conclusion).

**Sub-idea C — weight remap `{1,2,3,4}→{1,2,4,8}` (semantic, recall-gated).** Only weight 3 has
`popcount > 1` (two injections); all powers of two are single-injection. If a fraction `f` of
selected postings are weight-3, build does `k(1+f)` injections; remapping removes the `f` surplus
(~20 % build *iff* weight-3 is common — but it usually isn't: weights collapse). This is a
**ranking-semantics change** (rare-gram gaps double, 8× vs 4×), so it is **gated on the fuzzy-recall
eval**, not a free perf win. **Verdict: not-worth-it for perf**; consider only if recall data
independently favors it.

### Q5 — Lazy/partial build (defer low planes until needed) — REFUTED

Build is a binary counter assembled by **ripple-carry, strictly bottom-up**: plane `b` is written
by carries out of plane `b−1`. You cannot produce a high plane (an MSB of the score) without having
propagated the full carry chain from the LSBs. Probe `planes.rs` shows count `8 = 0b1000` sets
plane 3 only after the 8th posting, once carries ripple through planes 0–2 — the MSB is *unreachable*
without the low planes. Top-k needs the high bits ⇒ needs the full carry chain ⇒ **full precision is
required up front. Refuted.** (An MSB-only *approximate* top-k would break exact scores + the exact
raw floor; trifle needs both.) Note: plane count is *already* minimal — grown on demand to exactly
`max_score`'s width — so there is nothing further to defer.

---

## Ranked recommendations

| # | optimization | hits | win in trifle's regime | effort | risk | verdict |
|---|---|---|---|---|---|---|
| 1 | **(cross-lane) cap Σdf of selected postings** — build ∝ Σ selected cardinality | build | high (the real lever) | — | — | **surface to `select.rs` lane** |
| 2 | **Fused half-adder AND+XOR (Q3)** | build | ~1.3–1.5× build → total | L | M | **worth-effort but BLOCKED** (upstream `roaring` PR / croaring / SIMD) — *benchmark first* |
| 3 | **CSA/Wallace tree restructure (Q4)** | build | 0 scalar; enables SIMD/ILP | M | M | **hand to SIMD agent** (not a scalar win) |
| 4 | Uniform-weight raw-overlap shortcut + walk-floor tighten (Q4-A) | walk | ~0 measurable (<2 %) | S | low | **do-now only as cleanup** |
| 5 | Weight remap `{1,2,3,4}→{1,2,4,8}` (Q4-C) | build | ≤20 % *iff* weight-3 common (usually rare) | S | med (semantic) | **not-worth-it for perf; recall-gated** |
| 6 | Ordered MSB→LSB enumeration (Q1) | walk | <5 %, usually <1 % | M | M | **not-worth-it** (revisit iff large-plane+deep-pull profile) |
| 7 | Floor-prefiltering (Q2) | walk | net-negative (adds build) | M | M | **not-worth-it** |
| 8 | Lazy/partial build (Q5) | build | — | — | — | **not-possible (refuted)** |

## Cross-lane handoffs (out of this lane, but where the wins actually are)

- **Selection (`select.rs`)**: build is `Ω(Σ` selected cardinality`)`. Capping `Σdf` (tighter
  `t_max`, or a per-posting df ceiling) cuts the dominant cost more than any engine micro-opt.
- **SIMD/systems**: the fused half-adder (Q3) and a CSA/Wallace reduction tree (Q4) are the same
  build cost restructured for vectorization/ILP — these are the systems agent's territory and are
  the only path to a real build speedup.
- **Upstream `roaring`**: a fused `bitand_bitxor` (or container word-access) would unblock Q3 in the
  pure-Rust default.

## Needs a benchmark to decide

- Q3's ~1.5× build claim (croaring/fused prototype vs current) before any fork/upstream investment.
- Q1/Q2 *only if* a profiler ever shows a large-plane (common-token-in-selection) + deep-pull
  workload — not the design's operating point.
- Q4-C weight remap: fuzzy-recall eval (`geonames-cities` + mixed-script), per
  `tmax-pool-sweep-methodology` — it is a capability/ranking change, not a free win.
