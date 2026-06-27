# Proposal — Core: the pure overlap engine and a streaming public crate

Author lens: **the pure overlap engine as an isolated inner crate, optimized for maximum
speed.** This proposal is complete (all three layers, the filter, the deletion list, the perf
argument), but the deepest thinking is on the engine boundary and the candidate stream.

Everything below is verified against the current `feat/rev-v0.2` source (8,052 LOC `src/`).

---

## 1. Target layout — three layers, one-way dependencies

```
workspace/
  crates/trifle-overlap/      LAYER 1  pure engine.  deps: roaring ONLY.
  trifle/  (root package)     LAYER 2  storage + overlap wiring + LAYER 3 filter.
                                       deps: rusqlite, roaring, trifle-overlap, unicode-*.
  benchmarks/                          deps: trifle + trifle-overlap (benches the engine
                                       directly, in isolation, on synthetic bitmaps).
```

Dependency direction is strictly `trifle → trifle-overlap`. The engine knows nothing of
`rusqlite`, `Key`, `String`, `Schema`, provenance, hydration, or tokenizers. It is `#![no_std]`
+ `alloc` (roaring needs `alloc`; nothing needs `std`), which *structurally forbids* it from
reaching into SQL or I/O — the isolation the brief wants is enforced by the dependency graph,
not by discipline.

`trifle` modules after the refactor:

| module | role | change |
|---|---|---|
| `lib.rs` | `Index`, `Config`, `SearchOpts`, `Stats`, leases, write API, `rebuild`, `compact` | telemetry/Effort/`SearchSession`/write-proliferation cut; doc/seg flattened |
| `model.rs` | `Key`, `KeyShape`, `Schema`/builder, `Match`, `ColumnType` | **`Filter`/`CmpOp`/`FilterType` grammar deleted**; payload → typed columns only |
| `search.rs` | pipeline glue + the **`Stream`** candidate type | drives the engine `Walk`, batches hydration/filter, dedups |
| `hydrate.rs` | provenance + text + span + raw-SQL `filter_pass` (was half of `rank.rs`) | **`Ranker`/`Candidates`/`Candidate`/`QueryContext`/`Ranked` deleted** |
| `select.rs` | rarest-by-df prefix | class-normalization (welford) deleted; rarity fn pluggable |
| `dict.rs` | faulting term dict | `ClassStats`/`class_of`/`apply_df_changes` deleted |
| `postings.rs` | roaring base+delta index | unchanged (core) |
| `schema.rs` | DDL, stamps, ids, shadow swap | one fewer table (`doc` folded into `seg`) |
| `store/{mod,pool,sidecar}.rs` | Sidecar backend + writer mutex/read pool | **`Shared` deleted** |
| `term.rs`, `tokenize.rs`, `error.rs`, `hash.rs`, `instrument.rs` | term packing, tokenizers, errors | unchanged |
| ~~`welford.rs`~~ | — | **deleted** |

---

## 2. LAYER 1 — `trifle-overlap`: the pure engine (the crown jewel)

### What crosses the boundary

Only `&roaring::RoaringBitmap`, `u32`, `u64` (df), `f64` (one knob), and a tiny POD result:

```rust
/// One scored candidate. `id` is the posting id (an opaque u32 to the engine — a
/// segment id to trifle). `score` is the IDF-weighted bit-sliced bucket value (the
/// ordering key); `overlap` is the raw count of postings the id appears in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Candidate { pub id: u32, pub score: u32, pub overlap: u32 }

/// The lone rarity knob: `D` df-doublings per IDF weight step. `<= 0` ⇒ 1.0.
#[derive(Clone, Copy, Debug)]
pub struct Weighting { pub weight_step: f64 }
```

`df` per posting **is `posting.len()`** — by the monotonic-id contract an effective posting's
cardinality equals its document frequency, and today's `tier_weights` already reads `b.len()`.
So the engine derives rarity from the bitmaps themselves; nothing extra crosses. (A caller who
wants class-normalized or BM25-ish idf supplies explicit weights — see `with_weights`.)

### The streaming candidate API

```rust
/// The default df-anchored tier weights {1,2,3,4}: commonest posting (max df) → 1,
/// rarer → more, spaced in df-doublings. `N`-free (IDF *gaps* don't depend on corpus
/// size). Parallel to `postings`.
pub fn tier_weights(postings: &[&RoaringBitmap], w: Weighting) -> Vec<u32>;

/// The bit-sliced overlap walker: owns the count planes, streams candidates
/// score-descending / id-ascending, lazily.
pub struct Walk { /* weighted planes, unweighted planes, cursor, floor */ }

impl Walk {
    /// Build the weighted + unweighted bit-sliced counters from `postings`, each
    /// posting `i` weighted by `weights[i]`. O(k·log k) bitmap ops, **independent of
    /// posting cardinality** (the flatness property). `min_shared` is the raw-overlap
    /// floor (clamped to `postings.len()`, min 1).
    pub fn build(postings: &[&RoaringBitmap], weights: &[u32], min_shared: u32) -> Self;

    /// Reuse plane allocations across a batch (pops/returns buffers from `scratch`).
    pub fn build_in(scratch: &mut Scratch, postings: &[&RoaringBitmap],
                    weights: &[u32], min_shared: u32) -> Self;
}

/// Lazy: pulling N candidates evaluates only the high-score head — the lower buckets are
/// never materialized. Self-contained (no posting retention — see §2 "raw overlap").
impl Iterator for Walk {
    type Item = Candidate;
    fn next(&mut self) -> Option<Candidate>;
}

/// Reusable plane buffers for a batch loop.
pub struct Scratch { /* Vec<Vec<RoaringBitmap>> free lists */ }
impl Scratch { pub fn new() -> Self; }
```

Ergonomic wrapper for isolated benchmarking / a power-user caller:

```rust
pub fn candidates<'a>(postings: &'a [&'a RoaringBitmap], w: Weighting, min_shared: u32)
    -> impl Iterator<Item = Candidate> + 'a
{
    let weights = tier_weights(postings, w);
    Walk::build(postings, &weights, min_shared)   // Walk: 'static, owns its planes
}
```

### Why zero-alloc / zero-SQL / streaming

- **Zero SQL / zero String**: the engine sees `&[&RoaringBitmap]` and yields `Candidate`
  (three `u32`s). It cannot allocate a `String` or touch a `Connection` — it cannot *name*
  those types (`no_std`, single dep).
- **Zero per-candidate alloc**: `Walk` holds the plane stacks (≤ ~6 weighted + ~4 unweighted
  bitmaps for k ≤ 12) and one reused `Vec<u32>` for the current bucket's ids. `next()` pops
  from that buffer, refilling by `count_eq` only when the buffer drains. No per-candidate heap
  traffic.
- **Streaming**: candidates are produced bucket-by-bucket, high→low. Pulling the top-10 walks
  only the top few buckets. "All planes" (≤ ~10 bitmaps) are always built — that is cheap and
  bounded; what streams is the *bucket/candidate enumeration*, which early-stops the instant the
  consumer stops pulling.

### Raw overlap without retaining the postings — the dual BSI

Today `raw_overlap(id)` re-probes every posting (`bitmaps.iter().filter(|b| b.contains(id))`),
which forces the walker to **retain a borrow of the postings** → a self-referential struct when
embedded. Instead the engine builds a **second, unweighted bit-sliced counter** in the same
pass (each posting added with weight 1). An id's raw overlap is then
`Σ_b 2^b·[id ∈ uplane[b]]` — `O(log k)` `contains` on ≤ 4 planes, no posting borrow. This makes
`Walk` a fully owned, `'static`, self-contained `Iterator` — which is what lets `trifle` embed
it in `Stream` with **no lifetime gymnastics and no `ouroboros`** (forbidden: single dep). The
floor (`overlap >= min_shared`) is applied inside the engine using this counter.

(Alternative kept in reserve: single weighted BSI + retained postings + probe. It saves the
second plane stack at the cost of a borrow; a benchmark decides. Flagged as risk #2.)

### Preserving and *sharpening* flatness

- **Build** is unchanged in asymptotics: `add_weighted` injects `popcount(w) ≤ 2` ripples per
  posting (weights ∈ 1..=4), so the accumulation is `O(k·log k)` roaring ops, each sublinear in
  cardinality — independent of posting size. The dual BSI doubles the constant, still `O(k·log k)`.
- **Bucket walk, sharpened**: today's walk evaluates `count_eq` for *every* integer
  `c ∈ floor..=max_score`. Most are unreachable (scores are subset-sums of `{1,2,3,4}` weights).
  Precompute a `reachable: BitVec[0..=max_score]` by DP over `weights` (`O(k·max_score)`, ≤ 48
  entries) and skip unreachable `c` → fewer `count_eq` clones. Combined with the existing
  leading-zeros plane guard, a typical small-`limit` walk touches only a handful of buckets.
- **Allocation reuse across a batch**: `Scratch` recycles the plane `Vec`s and the bucket
  buffer, so a 1,000-query batch allocates the plane structure ~once, not per query.
- **Deferred (risk #8)**: a single MSB→LSB plane sweep that emits ids in descending-value order
  without per-bucket `count_eq` (classic bit-sliced top-k). Bigger win, but rejection-aware
  ordered enumeration is subtle; keep the proven bucket walk as the baseline and treat this as a
  benchmarked follow-up.

### Where rarity-weighting and selection live (the two cuts I own)

- **Tier-weighting lives *inside* the engine**, deriving df from `posting.len()`, knob `D`.
  Rationale: it is intrinsic to "IDF-weighted overlap," needs no corpus size, and reads off the
  in-hand bitmap. The `with_weights` escape hatch lets a caller inject class-normalized/BM25 idf
  without the engine knowing anything about scripts or corpora.
- **Class-normalized rarity (`welford.rs`) is CUT** from the default. It is unproven ("an
  empirical question," per its own doc); cutting reverts to rarest-by-raw-df (the within-script
  common case). A caller who needs it supplies weights. (Risk #6.)
- **Selection lives *outside* the engine, in `trifle`** (`select.rs`). It must run on
  term-table df *before* postings are fetched — the whole point is to *not* materialize the huge
  common-token posting. That is an IO/storage decision, not an overlap-math one. The engine only
  ever receives the already-selected postings. This is the clean cut: selection = storage layer,
  weighting = engine.

---

## 3. LAYER 2 — `trifle`: storage + overlap, with streaming output

### The public read surface

```rust
impl Reader<'_, T, B> {
    /// The 95% path: top-k hydrated matches in IDF-weighted overlap order.
    pub fn search(&self, q: &str, opts: SearchOpts) -> Result<Vec<Match>>;
    pub fn search_batch(&self, qs: &[&str], opts: SearchOpts) -> Result<Vec<Vec<Match>>>;

    /// The candidate stream: a snapshot-pinned, lazy, best-first cursor the caller
    /// composes ranking / fusion / pagination on top of.
    pub fn stream<'r>(&'r self, q: &str, opts: &SearchOpts) -> Result<Stream<'r>>;
}
```

### `Match` — carries every rerank signal so the trait machinery can die

```rust
pub struct Match {
    pub key: Key,
    pub label: String,
    pub span: Option<(usize, usize)>,
    pub text: String,
    pub score: u32,    // IDF-weighted overlap (the ordering key)
    pub overlap: u32,  // raw shared-token count
    pub seg_len: u32,  // gram length, for length-normalized rerank
}
```

A caller reranks by sorting the returned `Vec<Match>` on `score`/`overlap`/`seg_len`/`text`;
corpus-relative signals (`N`, `avgdl`) come from `stats()`. This replaces the entire
`Ranker`/`Candidates`/`Candidate`/`QueryContext`/`Ranked`/`Effort` apparatus.

### The candidate-stream type

```rust
/// A lazy, snapshot-pinned candidate cursor. Holds one pooled read connection with a
/// read transaction open for its lifetime (one consistent WAL snapshot across all
/// pulls). Drives the engine `Walk`, batch-hydrates provenance + applies the raw-SQL
/// filter + hydrates text per chunk, dedups to one result per key, best-first.
pub struct Stream<'r> {
    guard: ReadConn<'r>,           // owns the conn; Drop runs ROLLBACK (no pool poisoning)
    walk: trifle_overlap::Walk,    // 'static — owns its planes (see dual-BSI, §2)
    postings: Vec<RoaringBitmap>,  // selected postings, kept for matched_terms exposure
    present: Vec<(String, u32)>,   // selected token ⇄ posting index (rerank signal)
    ready: VecDeque<Match>,        // hydrated, deduped, filtered — ready to yield
    seen: FxHashSet<Key>,          // dedup: one result per key
    filter: Option<(String, Vec<Value>)>,
    chunk: usize,                  // engine ids pulled per refill (e.g. 64)
}

impl Iterator for Stream<'_> {
    type Item = Result<Match>;     // best-first, deduped, filtered, hydrated
    fn next(&mut self) -> Option<Result<Match>>;
}

impl Stream<'_> {
    /// Pull up to `n` more matches (the "next best K" primitive).
    pub fn take_n(&mut self, n: usize) -> Result<Vec<Match>>;
}
```

`next()` pops from `ready`; when empty it **refills a chunk**: pull up to `chunk` ids from
`walk`, map seg→key/label in one batched `WHERE id IN rarray(?1)` read, dedup new keys, apply
the raw-SQL filter to the chunk's keys in one query, hydrate text/len for survivors in one read,
push to `ready`. So the surface is a one-at-a-time `Iterator` (composes with `.filter`/`.take`/
fusion) while the SQL stays **batched per chunk**.

`search(q, opts)` is just `stream(q, opts)?.take_n(opts.limit)` with the standard
limit/early-stop. The old `Effort` over-fetch is gone: a reranking caller pulls a deeper pool
explicitly — `stream(q, opts)?.take_n(200)?`, rerank, truncate.

### How a caller hydrates / ranks / filters / fuses on top

```rust
// rerank a deep pool:
let mut s = idx.reader()?.stream("quikc brown", &SearchOpts::new(10))?;
let pool = s.take_n(200)?;                       // deeper than limit, explicitly
let top = my_rerank(pool).into_iter().take(10);  // order on Match.score/overlap/text

// label / arbitrary scope = caller-side filter on the stream (no `scope` closure needed):
let scan: Vec<Match> = idx.reader()?.stream(q, &opts)?
    .filter_map(Result::ok).filter(|m| m.label == "scan").take(10).collect();

// fusion across two indexes = pull both streams and merge (RRF) caller-side.

// raw ids, no SQL at all (power user):
use trifle::overlap::{candidates, Weighting};
for c in candidates(&my_bitmaps, Weighting { weight_step: 1.0 }, 2) { /* c.id, c.score */ }
```

### Snapshot / generation guard (invariant #3, preserved)

`stream()` opens the read tx (manual `BEGIN` on the held `ReadConn`), resolves terms in memory,
captures the dict generation, and compares to the snapshot's stored `dict_generation` — a skew
surfaces as retryable `Error::Busy`, exactly as `search_read_on` does today. `Stream`'s `Drop`
runs `ROLLBACK` before the `ReadConn` returns to the pool, so a tx-bearing connection is never
recycled. No `busy_timeout`, no internal retry (invariant #6).

### `batch == serial` (invariant #1, preserved)

Each query builds its own `Walk` from its own selected postings (cardinality = df) and its own
`min_shared`; `tier_weights` is per-query; selection derives from this query's df's. No batch
aggregate enters any per-query input. A batch shares one snapshot but ranks each query
identically to a singleton on that snapshot — `tests/scope_ranker.rs::search_batch_matches_serial`
holds unchanged.

---

## 4. LAYER 3 — the opt-in raw-SQL filter

Replace `Filter`/`FilterType`/`CmpOp`/`Filter::compile`/`filter_column`/`CompiledFilter`/the
`scope` closure with **one** opt-in raw fragment:

```rust
#[non_exhaustive]
pub struct SearchOpts<'a> {
    pub limit: usize,
    pub min_shared: Option<u32>,
    pub t_max: Option<usize>,
    pub weight_step: f64,
    /// Opt-in raw, parameterized SQL predicate over the candidate id set.
    pub filter: Option<SqlFilter<'a>>,
}

pub struct SqlFilter<'a> { pub sql: &'a str, pub params: &'a [Value] }

impl<'a> SearchOpts<'a> {
    pub fn filter(self, sql: &'a str, params: &'a [Value]) -> Self;
}
```

**Binding.** Per chunk, exactly the mechanism `filter_pass` already uses:

```sql
SELECT id FROM seg WHERE id IN rarray(?1) AND (<sql>)
```

`?1` is the candidate-id rarray; the fragment uses **anonymous `?`** placeholders, which SQLite
numbers from `?2` (one past the largest seen) — so a caller never collides with `?1` (today's
audit-F2 behavior, now the *only* path, not a special case). Cost is bounded by the chunk's
candidate ids, never an O(N) scan.

**What the schema must declare.** Typed columns to filter against. Keep a slimmed payload:

```rust
pub enum ColumnType { Int, Real, Text }   // affinity only; Timestamp = Int (epoch) by convention
Schema::chunked().text("body").column("lang", ColumnType::Text).column("created", ColumnType::Int)
```

Columns materialize on the (flattened) `seg` row and are written with the segment. The *grammar*
(`Cmp`/`In`/`Between`/`IsNull`/`Like`/`And`/`Or`) is gone; the caller writes
`"lang = ? AND created >= ?"` directly.

**Safety.** The fragment is a **trusted compile-time constant** (the caller's own code over the
caller's own columns); data binds through `params`, never formatted in. We drop `filter_column`
ident-validation because there is no longer a grammar that could be fed untrusted field names —
the fragment is trusted by contract, the same contract `Filter::Sql` already carries today. The
candidate-scoping `?1` bounds blast radius to the pool. (Risk: the fragment can name other
tables in `Shared`-style files — but `Shared` is also being cut, so trifle owns the file and
there are no other tables to reach.)

**Scope closure → gone.** Key-scope is a filter fragment (`key IN (...)`); label/arbitrary scope
is a caller-side `.filter()` on the `Match` stream (it has `key` + `label`). The `ScopeFn` type
and the in-walk scope application vanish.

---

## 5. Deletion list with LOC accounting

Measured against current `src/` (8,052 LOC). "Cut" = deleted; "moved" = relocated to the inner
crate (stays in the workspace but leaves the public crate).

| item | est. LOC | risk | invariant touched |
|---|---:|---|---|
| `welford.rs` + `ClassStats`/`ClassSnap` threading (dict/select/search/term) | **−500** | med — may dent mixed-script recall (#6) | none (selection reverts to raw-df) |
| Band-spread telemetry: `WeightStepHint`, `observe_band_spread`, `weight_step_hint`, `reset_band_spread_hist`, `band_spread_hist`, `HINT_*` | **−170** | low — advisory only | none |
| `Effort` enum + `pool()` + `effort_tests` | **−130** | low — replaced by `take_n` | none |
| `Ranker`/`OverlapRanker`/`Candidates`/`Candidate`/`QueryContext`/`Ranked` + plumbing | **−280** | low — signals now on `Match` | none |
| `Filter`/`CmpOp`/`FilterType` grammar + `compile`/`build`/`filter_column`/`CompiledFilter`/`filter_memo` (raw filter adds back ~40) | **−210** | med — callers rewrite filters as SQL | none (raw filter is strict superset) |
| `scope` closure (`ScopeFn`, field, walk application, setters/tests) | **−40** | low — caller-side `.filter()` | none |
| `Shared` backend (`shared.rs`) + re-export/docs | **−70** | med — drops co-location use case | none |
| Write-method proliferation → `upsert(key, segs, cols)` / `remove` / `remove_segment`; drop `*_segment`/`*_document`/`set_fields` split | **−150** | med — API churn | #2 simplified |
| **Flatten `doc`→`seg`** (one table; key inline): removes `doc_id_for`/`doc_segments`/`delete_doc_rows`/`remove_one_segment`/the doc/seg JOIN/most no-ghost code | **−350** | **high** — semantics change (§7, risk #4) | **#2 largely dissolves**; #4/#5 simpler |
| `SearchSession` (warm conn folded into `Stream`) | **−40** | low | none |
| **Cut subtotal (public crate)** | **≈ −1,940** | | |
| Engine **moved** to `trifle-overlap` (`add_weighted`/`weighted_overlap`/`tier_weights`/`count_eq`/walk + its tests) | ~−370 public, +~370 inner | low | #1, #7 preserved |

**Net.** Public `src/` ≈ 8,052 − 1,940 − 370(moved) ≈ **5,740**, plus the inner crate ≈ **370**
→ workspace ≈ **6,110**, i.e. **~1.32× smaller whole-crate** — but that number is dominated by
`tokenize.rs` (1,171 LOC, deliberately kept) and the `postings`/`store` codec (core). On the
**control-plane logic the brief actually targets** (`lib`+`model`+`rank`+`schema`+`search`+
`select`+`dict`+`welford`+`store/mod`+`shared` = 5,548 today), the refactor lands ≈ **2,810**
(public) + 370 (engine) → **~1.75×**, approaching 2× on the targeted surface.

**Honest landing of the 2–3× claim (risk #3):** a *faithful* simplification reaches ~1.3–1.4×
whole-crate / ~1.75× on the control plane. Hitting 2–3× whole-crate **requires** also cutting
`DefaultTokenizer`'s multi-script segmentation (≈ −500, keeping only fixed-`N` `NgramTokenizer`)
and is a **product decision** (drop mixed-script support), not a pure-simplification one. I
recommend the high-confidence core cuts (engine isolation, filter collapse, telemetry/Ranker/
welford/Shared/Effort) and flag the tokenizer + the doc/seg flatten as the two levers for more.

---

## 6. Performance argument (mechanism, not vibes)

1. **No SQL inside candidate generation.** Today `overlap_search` issues a `hydrate_provenance`
   JOIN *and* a `filter_pass` query *per non-empty bucket*, interleaved with the walk. The engine
   refactor removes all SQL from generation; provenance + filter + text batch **once per chunk**.
   Fewer `prepare_cached` executions and fewer SQLite round-trips on the hot path.
2. **Hydrate only what is pulled.** The lazy `Stream` hydrates provenance/text for the chunks the
   caller actually consumes; a `limit`-10 search never hydrates a deep tail. Today provenance is
   hydrated for *every* id of *every* walked bucket, including ids dropped by dedup.
3. **Isolation enables real optimization.** The engine over synthetic `RoaringBitmap`s can be
   profiled and tuned (plane layout, reachable-bucket skipping, roaring's SIMD AND/XOR/ANDNOT)
   with zero SQLite in the loop — impossible today because the walk is entangled with the store.
   `benchmarks/` can target it directly.
4. **Zero-alloc streaming core.** `Walk` reuses one bucket `Vec<u32>` + the plane stacks; `Scratch`
   recycles them across a batch. Per-candidate cost is `O(log k)` plane probes — no heap traffic.
   Today each bucket clones a plane (`count_eq`), builds a `Vec`, and inserts owned `Survivor`s
   (with `String`s) into a `FxHashMap`.
5. **Strings only for results.** The engine is pure `u32`. Provenance/text `String`s materialize
   only for yielded `Match`es (≤ what the caller pulls), not for every candidate id.
6. **Caller-controlled depth replaces speculative over-fetch.** `take_n` hydrates exactly the
   requested depth; no `Effort` heuristic over-hydrating "just in case."
7. **Flatness preserved + sharpened.** Build stays `O(k·log k)` bitmap ops independent of posting
   cardinality; reachable-bucket skipping cuts `count_eq` calls from `O(max_score)` to
   `O(#reachable scores)`.
8. **One-table provenance.** Flattening `doc`→`seg` turns the per-chunk provenance read from a
   `seg ⋈ doc` JOIN into a single-table `SELECT key,label,txt,len FROM seg WHERE id IN rarray`.

---

## 7. The doc→seg flatten (the one structural bet)

Today a `doc` row (key + payload) owns N `seg` rows (label,text). The brief flags this as a prime
simplification. I propose **one table**: `seg(id PK, key, label, txt, len, <columns…>)`, key
indexed for dedup/replace/delete. Consequences:

- **Invariant #2 (no-ghost doc rows) largely dissolves**: a key exists iff it has ≥1 `seg` row;
  there is no separate doc row to ghost. Columns live on the seg row, written with the segment —
  a column-only write to a nonexistent `(key,label)` simply has nothing to attach to and errors.
- **Dedup moves fully to the consumer** (`Stream.seen: FxHashSet<Key>`), keeping the
  highest-score, lowest-id segment per key — matching today's "best segment per doc."
- **Filter granularity becomes per-segment** (the `WHERE` is over `seg`), which is *more*
  flexible (you can filter the OCR segment differently from the field segment), but it **changes
  semantics** for a caller who filtered per-document. This is the highest-risk cut (risk #4) and
  the main thing the adversarial round should pressure-test.

---

## 8. Open risks / least-sure-of

1. **Streaming snapshot lifetime.** Holding a read tx open across `take_n` via manual
   `BEGIN`/`ROLLBACK` on a pooled `ReadConn` is correct only if `Drop` *always* rolls back; a
   panic between pulls that bypassed `Drop` would return a tx-bearing conn to the pool. Mitigation:
   `Drop` + possibly a checkout-time "is this conn mid-tx?" guard. Avoids self-reference, but subtle.
2. **Dual-BSI vs retain-and-probe.** The second (unweighted) plane stack doubles build constant +
   memory to make `Walk` self-contained. Needs a benchmark to confirm it beats retaining postings
   and probing; the single-BSI+borrow fallback is ready if not.
3. **2–3× claim** — addressed in §5: ~1.3× whole-crate honestly, ~1.75× on the targeted control
   plane; 3× needs cutting the tokenizer (a feature decision). I will not pretend otherwise.
4. **doc→seg flatten changes filter granularity and multi-segment semantics** (§7). The biggest
   behavioral bet. `Stream.seen` also grows with results pulled (bounded by pulled count).
5. **Filter still doesn't save overlap work.** A highly selective filter over a broad query can
   make `Stream` walk most of the candidate set before `take_n` fills — unchanged from today, but
   the streaming surface makes the worst case more visible/abusable.
6. **Cutting class-normalized rarity** may regress mixed-script recall; the welford doc itself
   calls its value "an empirical question." Reversible via `with_weights`; keep the benchmark eval.
7. **`with_weights` can violate `weight ≥ 1`**, which the floor + early-stop rely on (weighted ≥
   raw). Must document "weights ≥ 1" or clamp in the engine.
8. **MSB→LSB ordered enumeration** (the bigger flatness win) is deferred and unproven against
   the rejection-aware streaming consumer; the reachable-bucket-skip is the low-risk win I commit to.
