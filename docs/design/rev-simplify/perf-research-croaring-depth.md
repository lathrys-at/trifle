# BSI overlap engine — croaring-depth performance research

Scope: croaring-specific engine internals now reachable since `trifle-overlap` moved to the
**croaring 2.6 / croaring-sys 4.7.1 (CRoaring 4.7.1 C API)** dual-BSI backend. Companion to
`perf-findings.md`, `perf-research-bsi-algo.md`, `perf-research-bsi-systems.md` (the prior
pure-Rust lanes) and `crates/croaring-bsi-bench/`. All numbers from throwaway probes under
`/tmp/cro-depth` (croaring `=2.6.0`, `--release`, LTO, median of hundreds of iters); they are
*relative* signals on this machine, not SLAs. The operating point is unchanged: rarest-first
selection → **small/sparse array-container postings, k ≈ 6–12, small top-k**, build-bound.

## Bottom line

The croaring board surfaces **one clearly actionable win and one cheap cleanup**; the headline
build lever (the fused half-adder) is **confirmed in magnitude but definitively blocked** even via
croaring-sys FFI.

1. **Drop the dual-BSI** (do-now): for mixed-weight queries it adds **+132–156 %** to build, while
   raw-overlap via `contains` on retained owned postings is **~free**. End-to-end from blobs,
   owned+contains is **0.39–0.47× (≈2.2–2.5× faster)** than the current zero-copy-view + dual-BSI —
   even at `limit=8192` it is still 0.65×. Removes code; keeps `'static`; keeps the all-weight-1
   zero-copy / free-overlap path untouched.
2. **`read_many` for bucket materialization** (do-now, cheap): **1.0× small buckets, 1.79× @200,
   4.55× @1000** vs `extend(scratch.iter())`. Real only for large-bucket / deep-pull; neutral for
   shallow top-k. Effort S.
3. **Fused half-adder**: ceiling **confirmed at 1.30–1.64× build**, but **blocked** — no CRoaring
   primitive and croaring-sys exposes containers only as opaque pointers. Upstream-FR only.

## Q1 — Fused half-adder (the key question): confirmed, blocked

`add_weighted`'s hot first level does **two** passes over the same container pair —
`carry = op & acc[start]` then `acc[start] ^= op`. A fused half-adder walks them once, emitting
`sum = a^b` and `carry = a&b` together.

**Magnitude — confirmed.** Probe (sparse, build vs an XOR-only lower bound that skips the carry
AND entirely):

| regime | build | xor-only (lb) | ceiling |
|---|---:|---:|---:|
| sparse k=8 card=2k | 176 µs | 135 µs | **1.30×** |
| sparse k=12 card=2k | 373 µs | 285 µs | **1.31×** |
| sparse k=8 card=8k | 815 µs | 497 µs | **1.64×** |
| sparse k=12 card=500 | 101 µs | 74 µs | **1.36×** |

The carry is *tiny* (`carry_bits / xor_bits ≈ 0.03`) yet the AND still costs a full sorted-array
merge pass regardless of output size — **that** pass (not the carry data) is what a fused op
removes. The carry ripple beyond plane 0 is negligible (~2 %). This concretely ratifies the prior
~1.3–1.5× estimate on croaring.

**Reachability — definitively blocked.** CRoaring 4.7.1 has **no** `and_xor`/half-adder primitive
(grep of `roaring.h`: none). Lazy ops (`roaring_bitmap_lazy_*`) exist only for **OR and XOR, not
AND**, and don't fuse AND+XOR; `lazy_xor_inplace`/`fast_xor` give **1.00×** in the sparse regime
(probe2) and can't be used in the build anyway (the carry AND needs the running plane-0 between
XORs). croaring-sys binds containers only as opaque `*mut *mut c_void` (+ keys/typecodes) — **no
container-level ops** (`array_container_xor` et al. live in CRoaring's internal headers, outside
the public amalgamation). A custom half-adder would mean reimplementing the container layer
(array/bitset/run → 9 type-pair cases, SIMD) over raw pointers against an unstable internal ABI —
**strictly worse than forking pure-Rust `roaring`**. Only unblock: an upstream CRoaring
`roaring_bitmap_and_xor` FR. **Verdict: worth-effort-in-principle, not actionable.**

## Q2 — CSA / Wallace / divide-and-conquer tree: not-worth-it at trifle's k

New nuance vs the prior lanes (which said "same op count, no win" — true only for dense
fixed-width words): the linear ripple is actually **~O(k²·card)** because plane-0 (the parity)
*grows* as sparse-disjoint postings accumulate (probe3: k=8→174 µs, k=12→377 µs, k=16→680 µs,
k=24→1399 µs — quadratic; plane0 grows 15k→46k). A balanced **divide-and-conquer tree of BSI
adders** keeps merges small → ~O(k·card·log k) and is **correct** (verified).

But the **crossover is k≈12–16**, above trifle's operating point:

| k | 4 | 8 | 12 | 16 | 24 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| tree/linear (sparse card=2k) | 0.99× | **0.89×** | 1.06× | 1.24× | 1.44× | 1.67× |

At k≤10 (trifle's regime) the tree is **wash-to-slightly-worse**, and in the **high-overlap
regime it is 2× WORSE** (0.49–0.53×) because there the linear plane-0 stays small (head cancels in
XOR) so linear is already cheap. **Verdict: not-worth-it** for trifle; revisit only if selection
ever keeps large-k prefixes — and even then it regresses overlap-heavy queries.

## Q3 — `read_many` cursor for bucket materialization: do-now (cheap)

`bucket.extend(scratch.iter())` vs `scratch.cursor().read_many(&mut buf)`:

| bucket card | 10 | 50 | 200 | 1000 |
|---|---:|---:|---:|---:|
| speedup | 1.00× | 1.00× | **1.79×** | **4.55×** |

The walk is <10 % of a query and the top bucket is usually small (shallow top-k) → ~0 total win in
the common case; a real win for large buckets (a common token in the selection) and deep pulls.
**Verdict: do-now (effort S)** — resize the reused `bucket` Vec to `scratch.cardinality()` and
`read_many` into it; free where it matters, neutral otherwise.

## Q4 — Dual-BSI cost vs alternatives: DROP it (the actionable win)

The unweighted counter, built only when weights differ, is a whole second ~O(k²·card) build:

| query | weighted-only | +dual-BSI | retain+contains (limit=32) |
|---|---:|---:|---:|
| k=8 card=2k | 504 µs | **+148 %** | **+(-1) %** |
| k=12 card=2k | 1128 µs | **+132 %** | +(-0) % |
| k=8 card=500 | 109 µs | **+156 %** | +1 % |

Full A/B from stored blobs (the real choice), mixed-weight queries:

| query | A: view + dual-BSI | B: owned + contains | B/A |
|---|---:|---:|---:|
| k=8 card=2k | 1188 µs | 477 µs | **0.40×** |
| k=12 card=2k | 2529 µs | 1123 µs | **0.44×** |
| k=8 card=4k | 2496 µs | 1089 µs | **0.44×** |

The zero-copy view's deserialize saving is *dwarfed* by the second-counter build. Owned `Bitmap`s
are `'static`, so retaining them keeps `Counter` `'static`; only zero-copy is lost, and only for
the **mixed-weight minority** (all-weight-1 — the common rarest-first case — keeps views +
`overlap = score`, unchanged). `contains` stays cheaper across the whole practical range: even
`limit=8192` is 0.65×; the crossover where the dual-BSI amortizes is `limit` in the tens of
thousands (full-corpus scan), not a top-k.

**Verdict: do-now / worth-effort.** Replace the dual-BSI: for mixed weights, build the weighted
counter from owned postings, retain them, answer `raw_overlap` with k `contains` per yielded id.
~2.2–2.5× faster on mixed-weight queries **and removes code** (the `unweighted` planes + their
subset-sum). This matches the prior algo lane's Sub-idea B prediction, which the shipped engine
did not follow.

## Q5 — other croaring APIs: not-worth-it

- `and_cardinality` — count without materializing; but the build needs the carry *bitmap*, not its
  size. N/A.
- `fast_or`/`fast_xor`/`or_many` — heap pairwise reduction = the Q2 tree; no win at small k.
- `run_optimize` on planes — planes are transient, sparse symmetric-difference (not run-structured);
  it would spend CPU finding no runs (matches systems §3). No.
- `frozen_view` vs portable view — frozen format is **not** byte-identical to roaring-portable, so
  it would force a storage migration; the portable view is already zero-copy. No extra win.
- `flip` — irrelevant to overlap counting.

## What needs an in-repo benchmark to decide

- **Dual-BSI drop (Q4):** high-confidence per probe, but size the aggregate impact with the
  `latency` eval over **mixed-script / wide-df** queries (how often weights are actually mixed, and
  the real `limit`/k there). The per-query win is large and robust; this only sets the headline %.
- **read_many (Q3):** confirm real bucket sizes in `latency`/`profile` traces — only large buckets
  benefit.
- **Fused half-adder:** nothing to bench in-repo; the magnitude is confirmed and the blocker is an
  upstream-API fact. File the CRoaring `and_xor` FR if it's ever to be unblocked.
