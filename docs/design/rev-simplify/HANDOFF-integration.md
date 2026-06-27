# trifle rev v0.3 — integration handoff

**Read this first.** This is the entry point for the *integration* session that turns the
proven spike on `feat/lean-trifle-v0.3` into real `trifle`. Written 2026-06-27 at HEAD
`7023503`.

---

## ✅ Integration landed (update, 2026-06-27)

The root `trifle` crate **has been transformed to rev-v0.3** in this worktree (not yet committed).
What's done and green:

- **croaring everywhere** — `postings.rs` and the engine both use `croaring::Bitmap`; the
  `roaring` crate is dropped. Byte-identical portable blobs ⇒ no migration. `effective_postings`
  feeds `Counter::build` directly (no serialize-bridge).
- **Flattened `doc`→`seg`** (key on `seg`, no `doc` table); `schema.rs` `SCHEMA_VERSION = 4`.
- **Store trimmed**: `Backend` trait + `Shared` deleted; `Index<T>` holds a concrete `Sidecar`;
  pool gains the check-in `ROLLBACK` guard.
- **Engine wired**: `trifle-overlap` is a real dep; `rank.rs` deleted (its math is the engine).
- **New read surface**: `Reader::matches`/`matches_batch` (eager) + `Reader::candidates` →
  `CandidateStream` (lazy spine, provenance-only `Candidate`, batched `hydrate`, fuse-on-Err,
  `present_terms`/`matched_terms`/`n_segments`/`avgdl`/`collect_matches`).
- **`SqlFilter`** (fragment-first `?{N+1}` binding) replaces the whole `Filter`/`CmpOp`/`FilterType`
  grammar + `scope`.
- **3-method write API** (`upsert`/`remove`/`remove_segment`); payload + ghost-row machinery gone.
- **KEPT (ren overruled the deletion):** `welford.rs` + per-class multi-script rarity, and the
  band-spread telemetry + `WeightStepHint`.
- **Tests rewritten** (`basic`/`typo`/`unicode`/`drift`/`lifecycle`/`filter`/`stream`/`thrash`);
  obsolete files deleted. **All green:** lib 92 + integration 77 (incl. the 400-case proptest) +
  5 doctests; `clippy --workspace --all-targets -D warnings` and rustdoc `-D warnings` clean.
- **LOC:** public `trifle` `src/` 8,052 → **6,383** (~1.26×; ~1.34× had the two kept features been
  cut as the PROPOSAL planned). Tests 3,045 → 1,524 (~2× collapse).

**Deferred (tracked):** (1) **`benchmarks/` is excluded** from the workspace — its `Effort`
rerank-pool sweep / `Shared` / `Ranker` harness needs a rework against the streaming API (§3, §6.8).
(2) **`CLAUDE.md`** still describes the old architecture — update it. (3) Post-integration perf
levers (Σdf cap + `dfsweep`, zero-copy base load), §6.8.

The rest of this document is the original pre-integration plan, kept for the rationale trail.

---

---

## 0. TL;DR

- A 3-agent adversarial design produced the ratified rev-v0.3 design (`PROPOSAL.md`), then a
  viability spike proved the shape end-to-end and a multi-round perf investigation optimized the
  overlap engine and moved it to **CRoaring**.
- **What exists:** two spike crates — `crates/trifle-overlap` (the pure overlap engine, croaring)
  and `crates/trifle-lean` (storage + streaming candidate + raw-SQL filter slice) — plus a
  quarantined A/B bench `crates/croaring-bsi-bench`. All green, clippy-clean `-D warnings`,
  benchmarked, pushed.
- **What's left:** integrate into the root `trifle` crate (read pool, blob posting store,
  dict-gen guard, drift/rebuild, real `NgramTokenizer`, `SqlFilter`, Searcher snapshot model),
  then the post-integration levers (Σdf cap + `dfsweep` eval, zero-copy base load).
- **The honest headline:** this is a **~1.5× whole-crate simplification** (NOT the 2–3× originally
  asked) and the engine is **build-bound and near its pure-Rust ceiling**; the croaring move buys
  ~1.2× on plane math. Do not re-promise 2–3×. See `perf-findings.md`.

## 1. Orientation — read in this order
1. `PROPOSAL.md` — the ratified rev-v0.3 design (3 layers, the 7 resolved forks, the deletion
   list, the invariant table). **The spec.**
2. `perf-findings.md` — the consolidated performance state, what's landed, the remaining levers
   ranked, the Searcher snapshot model, and the per-decision verdicts.
3. This file — the integration plan + gotchas.
4. (Reference, optional) `proposal-{core,storage,filter}.md`, `critique-*.md`,
   `perf-research-*.md` — the full design/critique/research trail.
5. Memory (loaded automatically via `MEMORY.md`): `rev-v0.3-simplification-plan`,
   `no-library-owned-threads`, `no-sleeps-in-library-code`, `fxhash-not-siphash`.

## 2. Build / test / bench
```bash
cargo test -p trifle-overlap -p trifle-lean
cargo clippy -p trifle-overlap -p trifle-lean --all-targets -- -D warnings
cargo run -p trifle-overlap --example flatness --release
cargo run -p trifle-lean --example query_latency --release
# quarantined A/B (own [workspace], excluded from main):
cargo run --manifest-path crates/croaring-bsi-bench/Cargo.toml --release
```
The root `trifle` crate is **unchanged** (still roaring, still the old API) and builds as before.

## 3. The target architecture (3 layers) — from PROPOSAL.md
1. **`trifle-overlap`** — pure IDF-weighted bit-sliced overlap engine. Streams scored candidate
   ids; zero SQL/provenance/String. (Spike has this, on croaring.)
2. **`trifle`** (public) — lean SQLite storage feeding the engine; **streaming candidate API**
   as the spine + eager `matches()` as the safe default; the opt-in raw-SQL filter.
3. **The raw-SQL filter** — one `SqlFilter { fragment, params }` over the caller's *live* data
   (`key IN rarray(?)` or a co-located `ATTACH` join). **No trifle-stored filter columns**
   (churn/staleness — see decisions).

## 4. What the spike contains (and how it deviates from real trifle)

### `crates/trifle-overlap` — the engine (production-shaped, croaring)
The crown jewel, essentially integration-ready. Design:
- **CRoaring backend** (`croaring = "2.6"`). `Counter` is a single **weighted** bit-sliced
  counter + an **all-weight-1 fast path** (when all tier weights are 1 — the common rarest-first
  case — `overlap = score`, no probing, and the build can be **zero-copy** from views, retaining
  nothing). Mixed-weight (minority) retains **owned postings** and reads overlap via SIMD
  `contains` (measured ~2.5× cheaper than the dual-BSI it replaced). Counter is `'static` (owned
  state) → no self-referential lifetime when embedded.
- **Zero-copy build:** `Counter::build_from_blobs(&[&[u8]], …)` views stored **portable** bytes
  in place via `BitmapView` — and the `roaring`-crate and croaring portable formats are
  **byte-identical**, so trifle's existing stored blobs work with **no migration**.
- Ported opts: clone-avoiding `add_weighted` (via the `Operand` trait over `&Bitmap`/`&BitmapView`),
  `count_eq_into` scratch reuse, reachable-bucket skip, weight clamp ≥1, `read_many` bucket fill.
- API: `build`, `build_weighted`, `build_from_blobs`, `build_weighted_from_blobs`, `walk`,
  `advance(&self, &mut Walk)`, `stream`, `tier_weights`. `Scored { id, score, overlap }`.
- **Integration action:** make the root crate depend on this. **DECIDED (ren): croaring
  everywhere** — the storage posting layer (`postings.rs`) moves to croaring `Bitmap` too and the
  `roaring` crate is **dropped from trifle's deps**. The croaring portable format is byte-identical
  to the old roaring blobs, so this is still **no-migration**; and `effective_postings` then yields
  croaring `Bitmap`s fed straight into `Counter::build` with **no serialize-bridge**. (The earlier
  "keep roaring storage, view it zero-copy in the engine" recommendation is superseded.)

### `crates/trifle-lean` — the storage+stream+filter slice (a SPIKE, not production)
Proves the shape end-to-end; **rewrite against the real store on integration.** Proven:
- Flattened single `seg` table (no `doc` table → no-ghost invariant trivially true).
- `CandidateStream` owns the connection + `Counter` with **no self-referential lifetime** (manual
  `BEGIN`/`ROLLBACK`, never a stored `Transaction`) — the key lifetime proof.
- Provenance-only `Candidate` + batched `hydrate(&[Candidate])`; eager `matches()`.
- Raw-SQL `Filter` folded into the per-chunk provenance query as
  `WHERE (<fragment>) AND id IN rarray(?{N+1})` — **fragment first, scope param last** — so both
  numbered `?1..?N` and anonymous `?` bind with no collision (the audit-F2 fix; 3 tests cover it).
- Stream **fuses on first `Err`** (no silent truncation).
- **Deviations from real trifle (must change on integration):** postings kept **in memory** as
  croaring `Bitmap`s (real: base+delta **croaring** blobs in SQLite, fed via `build`/`build_from_blobs`); a
  single `Mutex<Connection>` (real: the read pool); **append-only** writes (real: replace-on-write
  + delta/compact); a **minimal trigram tokenizer** (real: `NgramTokenizer`); no dict-gen guard,
  no drift/rebuild, no span.

### `crates/croaring-bsi-bench` — quarantined A/B (keep as reference)
Standalone (`[workspace]`, excluded). roaring-vs-croaring + view-vs-deserialize harness. The
empirical justification for the croaring move. Useful when prototyping the fused half-adder.

## 5. Locked decisions (do NOT relitigate)
- **MSRV 1.85.** Low floor = max reach (MSRV is transitive; Rust has no LTS). The consumer
  (ren's shrike/Anki) is on rustc **1.92**, below croaring 2.7's **1.95** cliff — so croaring is
  pinned at **`^2.6`** (permissive caret; 2.6 has no MSRV floor; 2.7 gives trifle nothing — same
  bundled CRoaring 4.7.1). Test both ends in CI.
- **croaring is the bitmap library everywhere — storage AND engine** (SIMD ~1.2× + zero-copy
  views). The `roaring` crate is **dropped from trifle's deps**; `postings.rs` stores/reads
  croaring portable blobs (byte-identical to the old roaring blobs → no migration). Accepts a C
  build dep. Optional pure-Rust-`roaring` fallback feature is *available if wanted* for no-C reach
  (wasm, cross-compile) — not built; decide later.
- **No library-owned threads / no presumed runtime.** Parallelism is **runtime-agnostic**: hand
  the caller dep-free work units; feature-gate any async/futures. rayon REJECTED. See memory
  `no-library-owned-threads`.
- **No trifle-stored filter columns.** Filter the caller's live data (churn/staleness). Raw-SQL
  fragment over `key IN rarray(?)` / co-located `ATTACH` join.
- **Keep `welford.rs` + per-class multi-script rarity** (ren): `ClassStats`/`ClassSnap` and the
  class-aware rarest-first selection are a **key feature**, NOT deleted. This overrules PROPOSAL
  deletion #4 / §11 risk #2. The only v0.3 module deletion is `rank.rs` (engine → `trifle-overlap`).
- **Keep the band-spread telemetry + `WeightStepHint`** (ren): the per-query `log2(df_max/df_min)`
  histogram + `Stats::weight_step_hint` tune the weighted-overlap `weight_step` and stay. This
  overrules PROPOSAL deletion #6 / §5's "drop weight_step_hint".
- **Flatten `doc`→`seg`** (no-ghost dissolves). **No sleeps** — surface retryable `Error::Busy`.
- **Engine is build-bound and ~optimal in pure Rust.** Honest LOC ceiling **~1.5×** whole-crate
  (NOT 2–3×). Don't re-promise 2–3× without the (capability-cutting) tokenizer trim.

## 6. The integration plan (suggested phasing)
Each phase ships green under the load-bearing lint bar (`-D warnings`, clippy, rustdoc).
1. **Land `trifle-overlap` as a real workspace member** the root crate depends on. Add `croaring`
   to the root deps (permissive `^2.6`) and **drop `roaring`** — croaring everywhere (DECIDED).
2. **Rewrite the storage core** to the rev-v0.3 design: flatten `doc`→`seg`; postings as base+delta
   **croaring** blobs; `effective_postings` yields croaring `Bitmap`s fed straight into
   `Counter::build` (no serialize-bridge). Keep the existing `postings.rs`/`schema.rs`/`store`
   machinery's *shape* (the three-way write split, fold, shadow swap) — only the bitmap type
   changes from `roaring::RoaringBitmap` to `croaring::Bitmap`.
3. **Wire the read path:** selection (rarest-first by `term.df`) → engine → per-bucket
   provenance+filter on the snapshot → dedup-by-key → batched hydrate. Preserve the **dict-gen
   guard** (`search_read_on`) and **batch==serial**.
4. **The `SqlFilter`** (delete the old `Filter`/`CmpOp`/`FilterType` grammar + `scope`): fragment
   -first `?{N+1}` binding; the `key IN rarray(?)` default + optional `ATTACH` hook for co-located
   joins.
5. **Collapse the write API** to ~3 methods (`upsert`/`remove`/`remove_segment`); drop payload +
   the ghost-row machinery (dissolved by the flatten).
6. **The Searcher snapshot model** (§7) + the runtime-agnostic `match_units` work-units.
7. **Maintenance:** `compact`/`rebuild`/`stats` per PROPOSAL; the drift-reset stamps incl. the
   schema fingerprint (payload/columns removed from its inputs).
8. **Post-integration perf levers** (need real corpora/eval): the **Σdf selection cap** + a
   `dfsweep` recall eval (mirror `tmaxsweep`); **zero-copy base load** end-to-end.

## 7. Post-integration design notes (carried from the perf rounds)
- **Searcher snapshot model (tantivy-style):** `Index → Reader → Searcher`; the **Searcher owns
  one snapshot** (pooled conn + open read tx + captured dict-gen) until drop, so all its queries
  are mutually consistent. Parallelism is runtime-agnostic and needs **no `sqlite3_snapshot`**:
  the dominant cost (BSI build) is pure CPU on loaded postings, so the Searcher does the cheap
  SQL serially on its one snapshot, then hands out **pure-CPU work units** (own their loaded
  postings; `FnOnce() -> Result<…> + Send`) the caller schedules on their threads. Hydration back
  on the snapshot. A live Searcher pins its WAL snapshot (keep short-lived); a concurrent
  `rebuild` makes its generation stale → `Error::Busy` (drop + re-acquire). Details in
  `perf-findings.md`.
- **Σdf selection cap** (biggest remaining lever, recall-gated): add rarest-first postings until
  cumulative Σdf exceeds budget B (keep ≥ floor). **~2.6× p99 build for ~−1.5% recall@10** at
  B≈0.1·N. Lives in selection; expose a `df_budget` knob (scales with N); gate on a `dfsweep` eval.
- **Zero-copy base load:** `PRAGMA mmap_size` + `row.get_ref().as_blob()` → `build_from_blobs`,
  but only for **base-only** terms (no pending delta). Needs a **mixed-operand build** (fold
  `&Bitmap` deltas + `&BitmapView` bases together — the `Operand` trait already abstracts this).
- **Fused half-adder** (~1.3–1.64× build): **ren's planned upstream future work** — fork
  CRoaring + croaring-rs to add a fused in-place `xor_and`. Not in scope for integration.

## 8. Invariants to preserve (from BRIEF.md)
1. **batch == serial** (per-query inputs only). 2. **no-ghost** (dissolved by the flatten — keep
it dissolved). 3. **dict-generation guard** (read racing a rebuild → `Error::Busy`).
4. **monotonic id + atomic shadow swap** for rebuild. 5. **drift-reset** (schema/tokenizer/
data_version → drop cache, never migrate). 6. **no sleeps / `Error::Busy`** (busy_timeout=0).
7. **flatness** — engine op-count is cardinality-independent (wall-clock sublinear/dense-flat;
NOT literally "independent of posting size"). 8. **single tokenizer** on index+query.

## 9. Open questions for the integration session
- ~~**Storage blobs: roaring vs croaring?**~~ **RESOLVED (ren): croaring everywhere.** `postings.rs`
  stores/reads croaring portable blobs and the `roaring` crate is dropped. Byte-identical portable
  format ⇒ no migration; `effective_postings` feeds the engine with no serialize-bridge.
- **Pure-Rust fallback feature?** A `roaring`-backed engine behind a feature flag for no-C reach
  (wasm/cross-compile) — worth it for "usable by many," or YAGNI? (Roaring impl is in git history.)
- **`ATTACH` for co-located filter joins** — verify rusqlite 0.37 + the read pool wiring (apply
  ATTACH per pooled conn at creation).
- **`dfsweep` eval harness** — build alongside the Σdf cap (mirror the `tmaxsweep` methodology in
  memory `tmax-pool-sweep-methodology`).

## 10. File map
- Design: `docs/design/rev-simplify/PROPOSAL.md` (spec), `BRIEF.md` (invariants/constraints).
- Perf: `perf-findings.md` (consolidated), `perf-research-{bsi-algo,bsi-systems,croaring-depth,pipeline}.md`.
- Trail: `proposal-{core,storage,filter}.md`, `critique-{core,storage,filter}.md`.
- Code: `crates/trifle-overlap/` (engine), `crates/trifle-lean/` (slice),
  `crates/croaring-bsi-bench/` (quarantined A/B). Root `trifle` crate: untouched (the migration target).
