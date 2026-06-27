# Pipeline / cross-layer / integration performance research

Lane: the **cross-layer** levers — selection→engine, store→engine load, batch/parallelism
shape, scratch reuse — that the two engine-internal lanes (`perf-research-bsi-algo.md`,
`perf-research-bsi-systems.md`) explicitly handed off as "where the wins actually are." Branch
`feat/lean-trifle-v0.3`. Established facts honored, not relitigated: the engine is **build-bound
and near the pure-Rust ceiling**; **build ∝ Σ selected posting cardinality**; deserialize is a
minor share in the sparse regime; parallelism is **runtime-agnostic, no owned threads, no
presumed runtime** (decision #1); croaring is the decided backend (zero-copy `build_from_blobs`
already exists, portable bytes are byte-identical → no migration).

Numbers below are from a throwaway probe (`/tmp/sdf-probe`, synthetic 200k-doc Zipfian word
corpus, per-word trigrams, croaring `Counter::build`, `--release`, 2000 queries/edit-count).
Relative signals on this machine, not SLAs — but the *shape* is the finding.

## Ranked recommendations

| # | lever | layer | win in trifle's regime | effort | risk | verdict |
|---|-------|-------|-----------------------:|--------|------|---------|
| 1 | **Selection Σdf cap** (rarest-first, stop at a df budget) | `select` | **2.6× p99 build / 1.8× mean** @ −1.5% recall@10 | S–M | low (recall-gated) | **do-now** — the single biggest cross-layer lever, and it's a *tail tamer* |
| 2 | **`term.df` as the single rarity source** (drives selection + tier-weights + the load decoupling) | `select`/engine wiring | enabler; removes a re-cardinality pass | S | very low | **do-now** — unlocks #1 and #3 cleanly |
| 3 | **Zero-copy base load** (mmap blob → `BitmapView` → mixed-operand build) | store→engine | parse-skip; material **only on dense postings** | M | med | **worth-effort**, bundled w/ croaring; base-only (compacted) terms |
| 4 | **Runtime-agnostic work units** (`match_units` → `Vec<impl FnOnce()->Result + Send>`) | API | the caller's 2.4×@16 / 3.5×@64; trifle owns no threads | M | low | **do-now** (API design); honors decision #1 |
| 5 | **Batch df-read sharing** (one `read_dfs` for a serial `matches_batch`) | store | k_batch round-trips → 1 | S | low | **worth-effort** (serial-batch only) |
| 6 | **`Scratch`/`build_in` reuse** | engine API | 2× at tiny postings → ~1× by card≈2000 | S–M | low | **worth-effort** for as-you-type p99/allocator; not a median lever |

---

## 1. Selection Σdf cap — the biggest build lever (do-now, recall-gated)

**Mechanism.** Selection is already rarest-first (ascending df). Build cost is `Σ` of the kept
postings' cardinalities. So *cap the cumulative Σdf*: walk the rarest-first list adding postings
until the next one would push cumulative Σdf past a budget `B`, but **always keep at least the
typo floor `F = m + d`** postings (correctness floor — never starve a query below its floor).
This is a **df-budget**, strictly better than lowering `t_max`: `t_max` drops a *fixed count*
(hurting a rare-token query that needs all its tokens), whereas the df-budget **adapts** — it
keeps *all* tokens for a cheap query and trims only the expensive tail.

**Why it works: Σdf is heavily right-skewed even inside a rarest-first selection.** Measured: the
single largest kept posting is **~22–23% of Σdf**, the largest three are **~53–55%**. So dropping
the 1–3 least-rare (largest, most expensive, *least* discriminating) kept postings roughly halves
build cost while keeping ~9–11 of 12 discriminating tokens.

**It is a tail-tamer, not a median shift.** Median Σdf is ~3.8k (200k docs) but p90 is ~31k and
max ~726k — build is linear in Σdf, so cost is dominated by a small fraction of common-token
queries. A budget set near p90 leaves typical queries *untouched* (avg kept 11.6/12) and trims
only the expensive tail. Probe (k=1 edit, budget B=20k ≈ 0.1·N):

```
build latency µs   full → capped(B=20k)
  p50    118  →  73    (1.6×)
  p90    554  → 300    (1.8×)
  p99   1298  → 497    (2.6×)   <-- the headline
  max   2382  → 911    (2.6×)
recall@10  0.971 → 0.956   (Δ −0.015)
```

**The recall/speed curve (k=1 edit; pick B on it via the eval):**

| budget B (Σdf≤) | avg kept | mean build | speedup | recall@10 | Δrecall |
|---:|---:|---:|---:|---:|---:|
| ∞ (full t_max=12) | 12.0 | 221 µs | 1.00× | 0.971 | — |
| 50 000 | 11.8 | 151 µs | 1.46× | 0.969 | −0.002 |
| 20 000 | 11.6 | 121 µs | 1.83× | 0.956 | −0.015 |
| 10 000 | 11.2 | 94 µs | 2.36× | 0.923 | −0.048 |
| 5 000 | 10.5 | 72 µs | 3.07× | 0.880 | −0.090 |
| 2 000 | 9.0 | 51 µs | 4.34× | 0.789 | −0.181 |

k=2 edits is the same shape (slightly steeper recall cost). **The knee is ~B≈20k–50k for
N=200k**: 1.5–1.9× mean / 2.6× p99 for ≤1.5% recall. Below ~10k recall erodes fast.

**Setting the budget.** df scales with N, so `B` must scale with corpus size — express it as a
**fraction of N** (the probe's sweet spot is `B ≈ 0.1·N`), or auto-derive from a percentile of
the live `term.df` column. Default conservatively (≈0.1–0.25·N, ~0% recall loss) and expose it as
a `SearchOpts` knob (`df_budget: Option<u64>`), peer to `t_max`.

**Recall gate.** Add a `dfsweep` eval (mirror the existing `tmaxsweep`/`tools/tmax_knee.py`
machinery): recall@k + p50/p99 latency vs `df_budget`, on the `fuzzy` (GeoNames name+edit) and a
mixed-script set, per `tmax-pool-sweep-methodology`. Ship the budget only after the eval confirms
the knee on a *real* corpus (the synthetic skew is representative but not authoritative).

**Composition.** The cap bounds *how much dense posting* enters the build; zero-copy load (§3)
makes the dense bytes you do load parse-free. They are complementary and both peak on the same
common-token queries.

---

## 2. `term.df` as the single rarity source (do-now enabler)

Today selection reads `term.df` (via `read_dfs`) while `rank::tier_weights` re-derives rarity
from each loaded posting's `cardinality()`. Under the monotonic-id contract **`term.df` ==
|effective posting|** exactly (it's maintained live; a removed id decrements both df and the
posting). So:

- Compute **tier weights from `term.df`**, not from materialized bitmaps. Selection and weighting
  then share one `read_dfs`, and the Σdf cap (§1) is computed from the same df map with **zero
  postings loaded** — selection decides what to load *before* loading it.
- This **decouples weights from the blob fold**, which is exactly what the zero-copy load (§3)
  needs: weights are known up front, so the fold can stream heterogeneous operands (views +
  owned) into the planes without a cardinality pre-pass.

Near-zero risk, and it's the clean seam that makes #1 and #3 fall out.

---

## 3. End-to-end SQLite-blob zero-copy load (worth-effort, bundled w/ croaring)

**The target.** `PRAGMA mmap_size` (already 1 GiB) + the stored portable blob fed straight to
`Counter::build_from_blobs` (croaring `BitmapView` over the bytes — no parse, no owned-Bitmap
allocation). The parse-into-owned is the real cost; the byte format is already identical.

**Where it applies — the honest analysis.** The effective posting is `(base ∪ added) \ removed`.
- `post.base` is **one contiguous blob → viewable**.
- `delta.added` / `delta.removed` are separate blobs, and a set-difference can't be expressed as a
  lazy view → a delta'd term must be **materialized owned** (`base | added - removed`, today's
  path). So **only the base is view-able, and only when the term has no pending delta.**
- But trifle is **write-infrequent / read-often and `compact()` folds deltas into bases.** In the
  compacted steady state (the read regime trifle is tuned for) *most/all* selected terms are
  **base-only → fully view-able.** `delta_backlog` already signals when to compact; biasing a
  read-heavy phase to run post-compact maximizes the zero-copy-eligible fraction.

**Two rusqlite constraints (these shape the realizable form):**
1. `row.get_ref(col)?.as_blob()? → &[u8]` is valid **only for the current row step** — you cannot
   collect borrowed slices across a multi-row `rarray` query and then build from all of them. So
   either (a) **per-term point read** (`SELECT base WHERE id=?`, view, `add_weighted` into shared
   planes, drop the view, next term — k≤12 PK lookups), or (b) **arena memcpy**: copy each base's
   raw bytes into one reused `Vec<u8>` with offsets, then view the arena slices. (b) is one flat
   memcpy (cheap) vs the roaring parse (the actual cost), and the arena reuses across queries
   (ties to §6). Recommend (a) for simplicity — weights already come from `term.df` (§2), so no
   cardinality pre-pass is needed and each view is used the instant it's read.
2. The view borrows the mmap page / row; it must be folded and dropped **inside the read tx** —
   fine, the fold is synchronous and immediate.

**Engine change required.** `fold<O: Operand>` is monomorphic over a *single* operand type, so it
can't mix `&Bitmap` (delta'd, owned) and `&BitmapView` (base-only) in one build. Add a
**mixed-operand build** fed precomputed weights (from §2) and an iterator yielding either operand
(a small `enum Operand { Owned(Bitmap), View(BitmapView<'_>) }` folded uniformly, or a two-phase
`add` loop into shared planes followed by reachability finalize). Small, local to `trifle-overlap`.

**Verdict.** Worth-effort, bundled with croaring (already decided). The win is **parse-skip,
material only on dense postings** (sparse parse is already cheap — systems §5); the pure-sparse
regime sees little. That is precisely the common-token case the Σdf cap also targets, so the two
compose. Apply to base-only terms; fall back to owned merge for delta'd terms. Gate the
end-to-end p99 effect on the full `latency`/`profile` harness once a dense token is in selection.

---

## 4. Runtime-agnostic batch parallelism API (do-now design)

Honors decision #1: **no rayon, no owned threads, no presumed runtime.** trifle hands the caller
**independent, dependency-free work units**; the caller schedules them on *their* threads/executor.

```rust
impl<T: Tokenizer> Index<T> {
    /// One independent, `Send` work unit per query. Each unit, when called, checks out its own
    /// pooled read connection, opens its own DEFERRED snapshot, runs the dict-generation guard,
    /// and returns that query's top-`limit` matches (a `Busy` on guard skew, like `matches`).
    /// Schedule on any executor: `std::thread::scope`, your own rayon, tokio `spawn_blocking`.
    pub fn match_units<'q>(
        &self, queries: &'q [&'q str], opts: &SearchOpts<'_>, limit: usize,
    ) -> Vec<impl FnOnce() -> Result<Vec<Match>> + Send + use<'_, 'q, T>>;

    /// `'static` variant for a detached/global executor: captures an `Arc<Index>` clone + owned
    /// query, so units outlive the call site.
    pub fn match_units_owned(
        self: &Arc<Self>, queries: Vec<String>, opts: OwnedSearchOpts, limit: usize,
    ) -> Vec<impl FnOnce() -> Result<Vec<Match>> + Send + 'static>;
}
```

**Each unit is just `matches` on a fresh reader** — so per-query selection/df/weights/filter
derive only from that query's tokens (batch == serial *ranking* holds by construction), and the
generation guard per unit is identical to a single `matches`.

**The one design subtlety — snapshot semantics.** The serial `matches_batch` shares **one**
DEFERRED snapshot across all queries (proposal §3.5). A WAL snapshot is per-connection, and
`Connection` can't be shared live across threads, so **independent units necessarily take
independent snapshots.** Each unit is internally consistent; under write-infrequent operation they
are almost always the *same* snapshot, but it is not guaranteed. **Document the trade:** serial
`matches_batch` = single cross-query snapshot (the dep-free convenience, kept); `match_units` =
per-query-consistent + independent + parallelizable. Ranking is identical either way.

**Constraints to document:** (a) `SqlFilter`'s `&[&dyn ToSql]` must be `Send` for a unit to be
`Send` — require `Send` params on the parallel path (or owned params in `match_units_owned`); (b)
size the read pool ≥ the caller's parallelism width (each live unit holds one pooled conn for its
duration); (c) `Future`-returning sugar stays **feature-gated** — default ship is the work units +
`std::thread::scope` / `spawn_blocking` recipes, no runtime dep.

---

## 5. Batch df-read sharing (serial `matches_batch`, worth-effort)

A serial `matches_batch` (shared snapshot) can batch the `read_dfs` of **all** queries' distinct
tokens into **one** `WHERE id IN rarray(?1)` query, then each query selects from the shared df
map. df is a property of the shared snapshot, identical for every query, so this **preserves batch
== serial** exactly. Collapses `k_batch` df round-trips → 1. Small, clean, serial-batch-only (the
parallel `match_units` path uses separate conns, so it can't share). Pairs with §2 (df is already
the single source).

---

## 6. Scratch / `build_in` reuse (worth-effort for as-you-type)

Systems §2 measured the magnitude: **2× at tiny k/card → ~1× by card≈2000** (bitmap ops dominate
once postings have size). So this is an **as-you-type p99 / allocator-pressure** lever (a 1–2 char
prefix yields tiny postings — exactly the allocation-dominated regime), **not** a median win.

What to recycle across sequential queries, in one `Scratch`: the engine's plane `Vec`s
(`weighted`/`unweighted`) + `reachable` + the walk's `bucket`; the load **arena** (§3); the
provenance/hydrate `rarray` `Vec<Value>` + result maps.

**API tension with F4.** `Counter` owns its planes so it is `'static` and embeds in the streaming
`candidates()` cursor. A scratch-*borrowing* counter is not `'static`. Resolution: the eager
`matches` / `matches_batch` / as-you-type loop builds → drains top-k → discards, and needs no
`'static` — so add `Counter::build_in(&mut Scratch, …)` for *that* path, and keep the owning
`build` for the streaming path. Composes as a per-worker `Scratch` in §4.

---

## What still needs a real-corpus benchmark (vs. settled here)

- **§1 budget knee on a real corpus** — the synthetic skew is representative; the `dfsweep` on
  `fuzzy`/mixed-script is the authoritative gate before shipping the default `df_budget`.
- **§3 end-to-end p99** — the parse-skip win only shows in the full pipeline when a dense token is
  in selection; the isolated A/B already shows the load delta.
- **§4** — policy, not bench: the `Send` bound on `SqlFilter` params and the default pool width.
- **§6** — the deep-pull / as-you-type p99 win (recycle `count_eq` output via `clone_from`) on a
  realistic keystroke trace.
