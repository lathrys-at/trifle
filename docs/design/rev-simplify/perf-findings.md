# BSI overlap engine — performance findings & roadmap

Consolidates the foreground optimization work + four background research lanes
(`perf-research-bsi-algo.md`, `perf-research-bsi-systems.md`, `perf-research-croaring-depth.md`,
`perf-research-pipeline.md`) + the croaring A/B (`crates/croaring-bsi-bench`). Branch
`feat/lean-trifle-v0.3`. **Round 2 (croaring board) synthesis is at the bottom.**

## Bottom line

The overlap engine is **build-bound, and the build is already near-optimal in pure Rust.**
Both research lanes (algorithmic and systems) independently converge on a negative: there is
**no scalar pure-Rust-stable speedup left in the counting math or the bitmap representation**
beyond constant factors. In trifle's real regime (rarest-first → small/sparse postings, small
top-k) the engine is already fast — the lean end-to-end is **~90–320µs to 200k docs**. The
remaining levers are constant-factor (croaring SIMD) or **cross-layer** (selection), not in the
BSI counting itself.

Profiling split (probes): build ≈ 80–100% of a shallow top-k query; the walk is 0–23%
(usually <10%) and bounded by `max_score ≤ 4k ≤ ~48` buckets; `raw_overlap` <2%.

## Landed (foreground, committed, tests green + clippy clean)

1. **`add_weighted` clone-avoidance** — XOR the borrowed posting into the plane; derive the
   carry from the pre-XOR intersection. Removes a per-posting allocation/copy on the build hot
   path. Modest, allocation-reducing.
2. **all-weight-1 fast path** — when every tier weight is 1 (common collapsed-tier query),
   weighted score == raw overlap and the walk only visits buckets ≥ floor, so the per-id
   `raw_overlap` `contains` probes are redundant. ~4× on a full-drain of large postings;
   **~0 in the real small-posting/top-k regime** (raw_overlap is already <2% there) — so it is
   cleanup + a deep-pull/pagination win, not a headline.
3. **`count_eq_into`** — reuse a scratch bitmap + the bucket `Vec`; intersect the smallest
   set-bit plane first; early-exit both loops on empty. Correct, allocation-lighter; minor
   (count_eq was never the bottleneck).

4. **Moved the engine to the CRoaring backend** (croaring `^2.6`, MSRV 1.85). `trifle-overlap`
   and `trifle-lean` now use croaring (SIMD). The engine is **dual-BSI** (weighted + a small
   unweighted counter built only when weights differ), so it retains only owned planes — no
   posting/view retention — which makes it `'static` AND enables a **zero-copy build**
   (`Counter::build_from_blobs`) that views stored portable bytes in place (the roaring-crate and
   croaring portable formats are byte-identical → no migration). `raw_overlap` is now `O(log k)`
   plane membership (mixed weights) or free (all-weight-1). All prior opts ported.

These captured the genuine low-hanging fruit + the croaring move. Further engine-internal wins
are decision- or benchmark-gated (below).

## The real remaining levers (ranked)

| # | lever | where | expected | gate | verdict |
|---|-------|-------|---------:|------|---------|
| 1 | **Cap Σdf of selected postings** (skip/limit postings once cumulative df exceeds a budget) | `select.rs` (NOT the engine) | build ∝ Σ selected cardinality — the dominant cost | **recall eval** (dropping a posting trades recall for speed) | worth-effort; the single biggest build lever, but a recall/speed tradeoff — gate on the fuzzy-recall eval |
| 2 | **croaring opt-in SIMD backend** | engine backend | ~1.2× plane math (measured); ~1.5× end-to-end load w/ zero-copy view | decided (permissive `^2.6`, MSRV 1.85) | do-now: ratify the opt-in; it's the ONLY SIMD path (no pure-Rust SIMD win exists) |
| 3 | **Fused half-adder** (sum+carry in one container pass) | engine build | ~1.3–1.5× build → total | **blocked in pure Rust** (roaring exposes no fused op / word access) → croaring or upstream roaring PR; **benchmark the 1.5% first** | worth-effort, not actionable in the pure-Rust default; prototype via croaring to confirm before any fork |
| 4 | **Caller-scheduled concurrent reads** (runtime-agnostic; trifle owns NO threads, presumes NO runtime) | caller fans out `matches` on *their* threads/executor over trifle's `Send+Sync` read pool | the same 2.4×@16 / 3.5×@64 — realized by the caller's tool (`std::thread::scope`, their own rayon, tokio `spawn_blocking`), not trifle's | none for the core; trifle just guarantees `Send+Sync` + adequate pool sizing | do-now: guarantee `Reader: Send+Sync`, document the pattern, optionally a dep-free work-unit helper |
| 5 | **Batch `Scratch`/`build_in` reuse** | engine API | 2× at tiny k/card, decays to ~1× once bitmap ops dominate | bench (as-you-type p99 / allocator pressure) | worth-effort for as-you-type allocator-pressure/p99; not the big lever |

### Not worth it (refuted/measured negative)
- Pure-Rust SIMD (`wide`/`std::arch`): dense word-ops are memory-bandwidth-bound → scalar
  autovec ≈ `wide` ≈ roaring's own op (measured equal). croaring is the only SIMD path.
- Ordered MSB→LSB enumeration: correct (floor/dedup/early-stop work out), but the walk is <10%
  and bounded — the deferral was right. Revisit only on a large-plane + deep-pull profile.
- Floor-prefiltering: adds a counting pass to shrink the already-cheap walk; net-negative.
- Lazy/partial build: **refuted** — ripple-carry is strictly bottom-up; full precision needed up front.
- Weight remap {1,2,3,4}→{1,2,4,8}: ~≤20% build *iff* weight-3 is common (usually weights
  collapse to {1}); it's a semantic ranking change — recall-gated, not a perf call.
- `run_optimize` / container hints / alt bitmap libs (RoaringTreemap, raw bitset): wrong shape
  for trifle's postings.

## Decisions (ren)
1. **Batch parallelism = runtime-agnostic; rayon REJECTED.** A library must not own threads or
   presume a runtime. Parallelism = the caller fans out concurrent `matches` (each on its own
   pooled connection) on *their* threads/executor; trifle stays `Send+Sync` with the read pool.
   trifle exposes independent **work units** the caller schedules (closures, dep-free); any
   `Future`-returning sugar is feature-gated so non-async users pull in nothing. The
   shared-snapshot serial `matches_batch` stays the dep-free convenience.
2. **Move to croaring: DONE/decided.** croaring is the engine backend (SIMD + zero-copy Portable
   views over the existing roaring-format blobs, no migration). MSRV 1.85, permissive `^2.6`.
3. **Fused half-adder** — open: worth a croaring/SIMD prototype to confirm ~1.5× build before any
   upstream roaring PR. (Re-evaluate with the next agent round on the clean croaring board.)
4. **Selection Σdf cap** — open: the biggest build lever, but a recall/speed tradeoff in the
   selection layer + recall eval, not the engine.

## Honest summary
The BSI engine's low-hanging fruit is captured. It is build-bound and near the pure-Rust ceiling;
in the real regime it's already sub-ms. The meaningful further gains are (a) cross-layer
(selection Σdf cap, recall-gated), (b) the croaring opt-in (decided, ~1.2×), and (c) batch rayon
(dep-policy-gated). Everything else in the counting/representation math is measured dead-ends.

---

# Round 2 (croaring board) — synthesis

After moving to croaring, two fresh lanes (`perf-research-croaring-depth.md`,
`perf-research-pipeline.md`) re-evaluated. Outcomes:

### Landed (committed)
- **Dropped the dual-BSI** (CroaringDepth, measured): the unweighted counter cost +132–156% build
  for mixed-weight queries; `raw_overlap` via `contains` over retained owned postings is ~free →
  owned+contains is **~2.2–2.5× faster** than view+dual-BSI. Engine is now single weighted BSI +
  all-weight-1 fast path (zero-copy view build, retain nothing) / owned-postings+contains for the
  mixed minority. Faster AND less code. The earlier dual-BSI rec never measured the 2nd build.

### Do-now / cheap
- **`read_many` for bucket materialization** (CroaringDepth): croaring cursor bulk-read into the
  walk's bucket `Vec` (1.79×@200, 4.55×@1000 bucket sizes). ~0 for shallow top-k, real for deep
  pulls. Cheap S — queued as a follow-up.
- **`term.df` as the single rarity source** (Pipeline): weights from `term.df`, no per-query
  cardinality pass — enabler for the Σdf cap and zero-copy load.

### Worth-effort, gated
- **Σdf selection cap** (Pipeline, the biggest cross-layer lever): adaptive budget B, add
  rarest-first until cumulative Σdf would exceed B (keep ≥ floor). **p99 build 2.6×, mean 1.8×,
  for ~−1.5% recall@10** at B≈0.1·N. Tail-tamer (Σdf right-skewed even in a rarest-first set).
  Lives in `select.rs`; **recall-gated** → needs a `dfsweep` eval (mirror tmaxsweep). Budget
  scales with N → `df_budget` knob.
- **Zero-copy base load** (Pipeline): `PRAGMA mmap_size` + `row.get_ref().as_blob()` → engine
  `build_from_blobs`, but only for **base-only** terms (a delta'd posting `(base∪added)\removed`
  isn't a lazy view). Needs a **mixed-operand build** (fold `&Bitmap` + `&BitmapView` together;
  the `Operand` trait already abstracts this). Integration-time.
- **Searcher snapshot model** — see below. **Scratch/arena reuse**, **batch df-read sharing** —
  as-you-type / serial-batch only.

### Not actionable / dead-ends
- **Fused half-adder**: confirmed ~1.3–1.64× build, but **blocked** — no `and_xor` in CRoaring
  4.7.1, croaring-sys binds containers opaquely. Only unblock = an **upstream CRoaring FR**.
- **CSA/Wallace tree**: not worth it at k≤12 (crossover ~16; *2× worse* in high-overlap).
- **frozen_view**: not byte-identical to portable → would force a storage migration; stick with
  portable views.

# Searcher snapshot model (tantivy-style, replaces independent-snapshot work-units)

`Index → Reader → Searcher`. The **`Searcher` owns one snapshot** (a pooled connection with an
open read tx + the dict-generation captured at creation) until drop — so *all* its queries (serial
or parallel) are mutually consistent on that one snapshot.

The key enabler: **the dominant cost (BSI build+walk) is pure CPU on loaded postings — zero SQL.**
So parallelism needs no parallel DB connections (which SQLite fights); it needs one snapshot for
the cheap I/O and parallelism only on the CPU stage:

1. On the Searcher's single snapshot (serial, cheap): resolve → select → **load** the selected
   posting blobs per query.
2. `match_units` returns **pure-CPU work units** (`FnOnce() -> Result<Vec<Scored>> + Send`, each
   owning its loaded postings/blobs) — build+walk, no connection, no per-unit snapshot. The caller
   schedules them on *their* threads/executor (runtime-agnostic; no library-owned threads).
3. Provenance/filter/text hydration: back on the Searcher's snapshot (serial, batched).

Result: tantivy-style "Searcher owns a snapshot until drop" consistency **and** runtime-agnostic
parallelism on the part that actually costs — **without** needing `sqlite3_snapshot` (unavailable/
compile-gated anyway). Caveats: a live Searcher pins its WAL snapshot (keep short-lived); a
concurrent id-reassigning `rebuild` makes its generation stale → searches surface retryable
`Error::Busy` (drop + re-acquire). The `CandidateStream` becomes "a cursor on the Searcher's
snapshot," unifying serial/stream/parallel under one consistency model. Prototype-able in
`trifle-lean` (the parallel stage is CPU-only, so the spike's single connection suffices).
