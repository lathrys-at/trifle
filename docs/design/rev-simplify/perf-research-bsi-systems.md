# BSI engine — representation & systems performance research

Scope: representation- and systems-level optimizations for `trifle-overlap` (the owns-postings
single-BSI `Counter`/`Walk` over `roaring` 0.11). **Not** the counting math (carry-save tree,
popcount, dual-BSI) — that is the other agent's lane. Companion to the croaring A/B in
`crates/croaring-bsi-bench/`.

**The operating point** (established, not relitigated): rarest-first selection feeds the engine
**small, sparse postings** (array containers), `k ≤ ~12`, a **small top-k** pull. Build dominates;
deserialize is a minor share in this regime. The default engine stays **pure-Rust / `roaring`-only**
(isolation ethos); croaring (C-FFI, `^2.6`, MSRV-1.85) is an **accepted opt-in** SIMD backend,
already benchmarked at ~1.2× on plane math and ~1.5× on zero-copy load.

All numbers below are from throwaway probes (`/tmp/bsi-probe`, `roaring` 0.11.4, `--release`,
median of thousands of iters, clone baselines subtracted). They are *relative* signals on this
machine, not absolute SLAs.

---

## TL;DR — the honest headline

The systems lane has **limited upside in trifle's regime**. The build cost at the real operating
point is **sparse array-container merges** (`&` for the carry, `^=` for the sum), and those are
already at the floor pure Rust can reach:

- **There is no pure-Rust SIMD win.** Dense-container word ops are already memory-bandwidth-bound
  and LLVM autovectorizes the scalar loop to *exactly* `wide::u64x4` speed. The only ops with SIMD
  headroom are the **sparse array-container merges**, and beating roaring there means
  reimplementing vectorized set intersection — which is precisely what **croaring already is**.
  croaring is the only realistic SIMD path; it is opt-in by design.
- **Scratch/arena reuse is real but small** — a 2× on the *tiniest* queries (saving ~2 µs) that
  decays to ~1.0× at the realistic `card≈2000`, because the bitmap ops (not allocation) dominate
  once postings have any size.
- **Rayon pays only across a *large* batch** (≥ ~16 queries; ~2.4–3.5×). It does not pay
  within a single query (unit too small, and the ripple-carry accumulation is sequential) and
  not for as-you-type (batch size 1).
- **mmap is already enabled** (1 GiB); the remaining zero-copy win is croaring-view-only and minor
  in the sparse regime.

The biggest *certain* wins in this lane are organizational, not algorithmic: (a) the croaring
opt-in backend (already decided), (b) rayon across `search_batch` when batches are large, (c)
modest allocator-pressure relief from a `Scratch` for as-you-type bursts.

---

## 1. Pure-Rust SIMD for the plane AND / XOR / ANDNOT

**Finding: no pure-Rust win over `roaring` 0.11. croaring is the only realistic SIMD path.**

Two regimes, measured:

**Dense bitmap container (1024 × u64 words).** The plane op is a flat word-wise XOR/AND.

| variant | 1024-word XOR (op only) |
|---|---|
| scalar `for` loop (autovectorized) | 0.124 µs |
| `wide::u64x4` (explicit SIMD) | 0.124 µs |
| `roaring` crate dense-container `^=` | 0.125 µs |

All three are **identical**. LLVM already vectorizes the scalar loop to SSE2/AVX; the op is
bandwidth-bound, so `wide` adds nothing and roaring's own container op is already optimal. A
pure-Rust SIMD layer over the *dense* planes is dead on arrival.

**Sparse array containers (the real operating point).** This is where build time actually goes:

| sparse posting card | `&` (new) | `^=` (op) |
|---|---|---|
| 200 | 0.96 µs | 1.54 µs |
| 2 000 | 4.0 µs | 5.1 µs |
| 8 000 | 14.1 µs | 17.5 µs |

At `card≈2000`, `k` of these merges ≈ the entire ~18 µs build+walk. They are sorted-u16 merges
(galloping / binary-search intersection) — branchy, *not* autovectorized. SIMD genuinely helps
here, via shuffle-based vectorized intersection... which is exactly **croaring's** array-container
implementation, and the source of its measured ~1.2× on plane math. Replicating it in pure Rust
means `std::arch` SSE4.2/AVX2 intrinsics (unsafe, arch-gated, `is_x86_feature_detected!` dispatch,
no aarch64 path without NEON work) — i.e. **forking roaring's container internals**. That is a
large, unsafe, perpetually-maintained effort to recover what the croaring opt-in already gives.

`portable_simd`/`std::simd` would make the *dense* path ergonomic but (a) it's **nightly-only**
(off the table at MSRV 1.85) and (b) the dense path has **no headroom anyway**.

**Verdict: not-worth-it** for a pure-Rust SIMD layer. Keep the croaring opt-in backend as the
SIMD answer (already decided/benchmarked). Keeps-pure-Rust default: yes (by not changing it).

---

## 2. Batch scratch / arena reuse

**Finding: real but small; a `Scratch` is worth it for allocator-pressure relief in as-you-type
bursts, not for per-query wall-clock at the realistic operating point.**

Allocation churn per query (counting global allocator; `build` + top-10 walk):

| k | card | allocations | bytes |
|---|---|---|---|
| 4 | 200 | 22 | ~6.9 KB |
| 8 | 500 | 42 | ~33 KB |
| 8 | 2 000 | 42 | ~123 KB |
| 12 | 2 000 | 59 | ~176 KB |

Churn sources: the `planes` Vec backbone + each plane `RoaringBitmap` (build), the `reachable`
`Vec<bool>`, then per walked bucket a `count_eq` output bitmap + the `bucket.iter().collect()`
`Vec<u32>` + count_eq's internal `set: Vec<usize>`. Tens of allocations, tens–hundreds of KB.

Win from a reused `Scratch` (clear-and-reuse the planes backbone, `reachable` buffer, and bucket
`Vec` across sequential queries), clean timing, **per query**:

| k | card | fresh | reuse | win |
|---|---|---|---|---|
| 4 | 200 | 4.00 µs | 1.98 µs | **2.02×** |
| 8 | 500 | 6.39 µs | 5.66 µs | 1.13× |
| 8 | 2 000 | 18.0 µs | 17.9 µs | 1.01× |
| 12 | 2 000 | 26.9 µs | 26.8 µs | 1.00× |

The win is large *only* where queries are already cheapest (tiny postings), and evaporates by
`card≈2000` because the array-container merges (§1) dominate, not allocation. Note this reuse did
**not** recycle the per-bucket `count_eq` output bitmap (hard with roaring's value-returning ops;
`clone_from` into a retained bitmap could recover container allocations) — so deep-pull workloads
that walk many buckets would benefit more than this shallow top-10 measurement shows.

The genuine case for a `Scratch`: **as-you-type / `search_batch`** issues many queries
back-to-back; recycling tens of allocations × hundreds of KB per keystroke reduces allocator
pressure and p99 jitter even when median wall-clock barely moves. It also composes with rayon
(§4) as a per-worker scratch.

**Verdict: worth-effort** (effort **S–M**, pure-Rust **yes**, risk **low**). Add an optional
`Counter::build_in(&mut Scratch, …)` / `advance` that borrows reusable buffers; keep the
allocating `build` as the default ergonomic path. Do **not** oversell it as the headline win.

---

## 3. Memory layout / container behavior (run_optimize, hints)

**Finding: not-worth-it.** trifle's postings are sets of segment ids that *contain a given gram* —
they are sparse and **not run-structured** (monotonic ids ≠ contiguous within a posting), so they
sit in **array containers**, which is the right representation. `run_optimize` only pays for long
contiguous runs; it would spend CPU to find none and can only hurt. The transient **planes** can
become denser in a high-overlap head, but they are rebuilt per query and never serialized, so
container-type tuning on them has no persistent payoff. No cache-layout change beats letting
roaring pick the container. (One micro-nit, folded into §2: `count_eq`'s `set: Vec<usize>` of ≤ ~6
indices should be a stack array, not a heap Vec.)

---

## 4. Parallelism (rayon)

**Finding: pays only across a large batch; fits the sync `&self` model because the parallel part
is pure CPU on owned postings.**

Batch of B queries, serial vs `rayon::par_iter` over `build`+walk (k=8, card=2000):

| B | serial | rayon | speedup |
|---|---|---|---|
| 1 | 18.7 µs | 18.7 µs | 1.00× |
| 4 | 74.8 µs | 70.4 µs | 1.06× |
| 16 | 347 µs | 143 µs | **2.42×** |
| 64 | 1394 µs | 394 µs | **3.54×** |

Below B≈16, thread-dispatch overhead eats the gain. **Within a single query** there is nothing to
parallelize profitably: a single build+walk is ~18–27 µs at the operating point (smaller than
dispatch latency), and the ripple-carry accumulation into shared planes is **inherently
sequential** (a parallel tree-reduction of the k postings is possible but each op is sub-µs — not
worth the join overhead).

Fit with the model: the engine's `build`+`advance` operate on an **owned `Vec<RoaringBitmap>`**
with no `Connection` and no `&self` of `Index`. So `matches_batch` can load all queries' postings
on the shared snapshot/tx (serial, cheap — load is not the bottleneck), then run the **pure**
builds across a rayon pool. This keeps the synchronous, no-async contract intact and never touches
SQLite off-thread.

**Verdict: worth-effort** for `search_batch` *only* (effort **M**, pure-Rust **yes** — `rayon` is
pure Rust, MSRV-fine, but adds a dep to the public crate; gate it behind a feature or apply only in
`matches_batch`). Risk **low–med** (dep surface; determinism preserved since `collect` keeps order).
**Not-worth-it** within a query or for as-you-type (B=1).

---

## 5. Posting-load I/O tie-in (mmap + zero-copy)

**Finding: mostly already done; the remaining win is croaring-view-only and minor here.**

`PRAGMA mmap_size` is **already 1 GiB** on every connection (`src/store/sidecar.rs:16,98`), so
posting blobs are served from the mapped file. The residual copy is `roaring`'s
`deserialize_from`, which parses bytes into an **owned** `RoaringBitmap` (the
`deserialize_cost` example quantifies this share; it grows with cardinality, so it is smallest in
the sparse regime trifle operates in). Two levers:

- **Pure-Rust, free-ish:** read the blob column via `row.get_ref()?.as_blob()?` (a borrowed
  `&[u8]` into the row, valid for the row's lifetime) and `deserialize_from` that slice — avoids an
  intermediate `Vec<u8>` copy of the blob, but **not** the parse-into-owned step. Small.
- **croaring-only, structural:** `BitmapView::deserialize::<Portable>(&blob)` views the stored
  bytes **in place** (no parse, no copy) and feeds them straight into the BSI add. trifle's stored
  format is **byte-identical** to croaring portable, so there is no migration. Measured ~1.5× on
  the *load* portion — but load is a minor share in the sparse regime, so the end-to-end effect is
  modest unless a **common (dense) token sneaks into the selection**, which is exactly when it pays.

**Verdict: worth-effort only bundled with the croaring opt-in backend** (effort **M**, pure-Rust
**no** for the zero-copy view; the borrowed-slice tweak is **yes**, effort **S**, but tiny win).
Standalone, **not-worth-it**.

---

## 6. Alternative bitmap libs / representations

**Finding: not-worth-it.** `RoaringTreemap` is for `u64` keys; trifle's ids are `u32`, so it only
adds an indirection. A raw dense bitset over the id universe would be catastrophic — the universe
is N segments (up to millions) while postings are sparse, the case roaring exists for. Other Rust
roaring impls are less mature than `roaring` 0.11 and would not change the §1 conclusion (the
sparse-merge SIMD gap is fundamental, not a library-quality artifact). croaring remains the single
worthwhile alternative and is already the opt-in backend.

---

## What needs a benchmark to decide (vs. settled here)

- **Settled by probe:** no pure-Rust SIMD win (§1); rayon thresholds (§4); dense-op autovec parity
  (§1); scratch-reuse magnitude at shallow pull (§2).
- **Needs a real-corpus bench to size:** the scratch-reuse win on **deep-pull** workloads (recycle
  the `count_eq` output via `clone_from`) — the shallow top-10 probe understates it. Decide with the
  `latency`/`profile` evals on a realistic as-you-type trace.
- **Needs the end-to-end harness:** the croaring zero-copy-view end-to-end win (§5) once a common
  token is in the selection — the isolated A/B shows the load delta; only the full pipeline shows
  whether it moves p99.
- **Policy, not bench:** whether `rayon` may enter the *public* crate's dep graph, or stay behind a
  feature / `benchmarks`-only.
