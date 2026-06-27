# trifle rev v0.3 — radical simplification proposal

**Status:** equilibrium reached by a 3-agent adversarial design process (proposals +
cross-review in this directory: `proposal-{core,storage,filter}.md`,
`critique-{core,storage,filter}.md`). This document is the synthesized result.

**Baseline:** `feat/rev-v0.2`, 8,052 src LOC across 18 modules.

---

## 0. Executive summary

trifle is refactored into **three clean layers** with one-way dependencies:

1. **`trifle-overlap`** — a pure, `roaring`-only inner crate that *is* the IDF-weighted
   bit-sliced overlap engine. Zero SQL, zero provenance, zero `String`. Streams scored
   candidate ids. The crown jewel, now optimizable and benchmarkable in isolation.
2. **`trifle`** — the public crate: a lean SQLite storage engine that feeds postings into the
   engine and exposes a **lazy stream of IDF-weighted overlap candidates** as the architectural
   spine, with an eager `matches()` convenience as the safe default front door.
3. **An opt-in raw-SQL filter** — one parameterized fragment over the caller's *live* data
   (`key IN rarray(?)` on any backend, or a co-located join), replacing the entire
   `Filter`/`CmpOp`/`FilterType` grammar, the `scope` closure, and all trifle-stored filter
   columns.

**The headline number, stated honestly:** itemized deletions alone remove ~2,155 LOC →
**~5,900 public src (~1.37× whole-crate)**; the *estimated* structural reshaping that the
`Index<T,B>`→`Index<T>` generic-removal enables across every signature takes it to
**~5,200–5,400 src (~1.5×)**. The test suite *itself* collapses **~1.8×** (its payload/ghost
permutation tests), but **src+tests combined is ~1.5×** — tests shrink *less* than src, so adding
them to the denominator does not lift the whole-crate ratio. On the **control-plane logic the
brief actually targets** (everything but `tokenize.rs` and the roaring/SQLite codec) the refactor
lands **~1.75–2×**. The literal whole-crate **2×** is reachable only by additionally taking an
**optional, benchmark-gated tokenizer trim** (a *capability* cut, not a free simplification).
**3× is not achievable** without dismantling the `postings`/`schema`/`store` bedrock the
invariants forbid — we do not promise it.

**The most important design reversal** the adversarial round produced: trifle should store **no
filter attribute columns at all**. The attributes callers filter on (`due`, `reps`, `deck`,
`tags`) are *high-churn*, and a write-infrequent derived cache that mirrors them serves silently
**stale** filter results — a wrong-results bug that trifle's drift-reset (shape-only) cannot
catch. Filtering the caller's live data is staleness-free by construction. (This flipped the
initial 2-of-3 "keep typed columns" position to a 3-of-3 "no columns".)

---

## 1. Architecture — three layers, one-way dependencies

```
crates/trifle-overlap/   LAYER 1  pure engine.   deps: roaring ONLY  (#![no_std] + alloc)
trifle/  (root package)  LAYER 2  storage + overlap wiring + LAYER 3 raw-SQL filter
                                  deps: rusqlite (bundled+array), roaring, trifle-overlap, unicode-*
benchmarks/              publish=false; benches trifle-overlap directly on synthetic bitmaps
```

Dependency direction is strictly `trifle → trifle-overlap`. The engine cannot *name*
`rusqlite`, `Key`, `Schema`, `String`-provenance, or a `Connection` — the isolation the brief
wants is enforced by the dependency graph (`no_std` + a single dep), not by discipline.

`trifle` module end-state (see §8 for LOC):

| module | role | change |
|---|---|---|
| `lib.rs` | `Index<T>`, `Config`, `SearchOpts`, `Stats`, leases, write API, `rebuild`, `compact` | telemetry / `Effort` / `SearchSession` / write-proliferation / `scope` / ranker plumbing cut; `doc`→`seg` flattened; **`B: Backend` generic dropped** |
| `model.rs` | `Key`, `KeyShape`, `Schema`/builder, `Match`, `Candidate`, `Document` | **`Filter`/`CmpOp`/`FilterType` grammar + payload deleted** |
| `search.rs` | pipeline glue + the `CandidateStream` | drives the engine, batches provenance/filter/hydrate per bucket, dedups, snapshot/generation guard |
| `select.rs` | rarest-by-raw-df prefix | class-normalization (welford) deleted |
| `dict.rs` | faulting term dict | `ClassStats`/`ClassSnap` threading deleted |
| `postings.rs` | roaring base+delta index | unchanged (bedrock) |
| `schema.rs` | DDL, drift stamps, ids, shadow swap | one fewer table (`doc` folded into `seg`); filter columns gone |
| `store/{mod,pool,sidecar}.rs` | Sidecar store + writer mutex / read pool + **optional ATTACH hook** | **`Shared` deleted**; pool gains check-in rollback guard |
| `tokenize.rs`, `term.rs`, `error.rs`, `hash.rs`, `instrument.rs` | tokenizer, term packing, errors | unchanged (bedrock) |
| ~~`welford.rs`~~, ~~`rank.rs`~~ | — | **deleted**; `rank.rs`'s pure engine moves to `trifle-overlap`, its SQL glue folds into `search.rs` |

---

## 2. Layer 1 — `trifle-overlap`: the pure overlap engine

Everything the brief names the crown jewel — `add_weighted`, `weighted_overlap`,
`tier_weights`, `count_eq`, the high→low bucket walk — moves here, behind a small surface.

### What crosses the boundary

Only `&RoaringBitmap` / owned `RoaringBitmap`, `u32`/`u64`/`f64`, and a POD result:

```rust
/// One scored candidate. `id` is an opaque posting id (a segment id to trifle). `Copy`, 12 B.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Scored { pub id: u32, pub score: u32, pub overlap: u32 }
```

### The engine owns the postings (resolves the embedding-lifetime problem)

The decisive engine-design decision (fork F4): the `Counter` **owns** (moves in) the selected
postings, so it is `'static` and embeds in the Layer-2 stream with **no self-referential
lifetime, no `ouroboros`, and no second plane stack**. `effective_postings` already returns
owned bitmaps, so trifle *moves* the selected ones in — no copy.

```rust
pub struct Counter {
    postings: Vec<RoaringBitmap>,  // owned — moved in by the caller
    planes:   Vec<RoaringBitmap>,  // the weighted bit-sliced count (the only build allocation)
    weights:  Vec<u32>,
    floor:    u32,                 // raw min_shared, clamped to postings.len().max(1)
    max_score:u32,
}

impl Counter {
    /// Weight each posting by per-query df rarity (df = posting cardinality, by the monotonic-id
    /// contract; knob `D = weight_step`), then accumulate the weighted bit-sliced counter.
    /// `O(k·log k)` bitmap ops; the op COUNT is cardinality-independent (wall-clock sublinear,
    /// flat in the dense bitmap-container regime — validated by the spike, not "independent of size").
    pub fn build(postings: Vec<RoaringBitmap>, weight_step: f64, min_shared: u32) -> Self;

    /// Build with explicit per-posting weights (escape hatch for a caller-supplied idf, e.g.
    /// BM25-ish). Weights are CLAMPED to ≥ 1 internally (see below).
    pub fn build_weighted(postings: Vec<RoaringBitmap>, weights: Vec<u32>, min_shared: u32) -> Self;

    /// A fresh best-first walk cursor over this counter. The only way to construct a `Walk`
    /// (its fields are private), so `trifle` can build one to drive `advance`.
    pub fn walk(&self) -> Walk;

    /// Advance a plain owned cursor: `&self` + `&mut Walk` ⇒ no borrow stored next to the
    /// Counter, so the embedding struct is not self-referential.
    pub fn advance(&self, w: &mut Walk) -> Option<Scored>;

    pub fn postings(&self) -> &[RoaringBitmap];  // serves trifle's matched_terms (no separate retention)
    pub fn weights(&self)  -> &[u32];

    /// Ergonomic owning iterator for the isolated bench / power-user caller.
    pub fn stream(self) -> impl Iterator<Item = Scored>;
}

/// A plain, owned, `'static` walk cursor. The consumer owns both the Counter and the Walk;
/// neither borrows the other. Constructed via [`Counter::walk`] (fields are private).
pub struct Walk { c: u32, bucket: Vec<u32>, pos: usize }

/// Default df-anchored tier weights {1,2,3,4}: commonest posting (max df) → 1, rarer → more,
/// spaced in df-doublings. `N`-free (IDF *gaps* don't depend on corpus size).
pub fn tier_weights(cardinalities: &[u64], weight_step: f64) -> Vec<u32>;
```

**Raw overlap** of a *yielded* id is a `contains` scan over the owned postings (`O(k)`, k ≤
`t_max` ≤ ~12) — paid only per yielded id, so a shallow top-10 pull never builds machinery for a
tail it won't yield. (The dual-BSI alternative — a second unweighted plane stack giving
`O(log k)` per-id raw overlap — is kept as a **benchmarked alternative** for deep-pull workloads
behind the same owned `Counter`; it is not the default because it pays a full second build up
front. See §11.)

**Mandatory weight clamp (correctness, was a latent bug):** every weight is clamped to `≥ 1`
inside the engine (`weight.max(1)`). The floor + early-stop rely on *weighted ≥ raw*; a weight
of 0 would let an id with raw overlap ≥ `min_shared` have weighted score < `min_shared`, so the
walk would terminate **before yielding a valid result** — a *missing* result on a deterministic
query (wrong, not merely "missing-then-self-healing"). Documenting "use weights ≥ 1" is
insufficient; the engine enforces it.

### Flatness — preserved and sharpened (claim refined by the v0.3 spike)

- **Build** stays `O(k·log k)` roaring ops (`popcount(w) ≤ 2` ripples per posting for
  `w ∈ 1..=4`). The spike (`crates/trifle-overlap/examples/flatness.rs`) showed the precise
  invariant: the **operation count** is cardinality-independent, so wall-clock is *sublinear* in
  the sparse/array-container regime (and *flat* in the dense bitmap-container regime), pulling
  away from a naive per-id counter as postings densify (up to ~16× sparse, ~160× dense). It is
  **not** literally "independent of posting size" — the old `rank.rs` comment overclaimed.
- **Bucket walk** skips unreachable scores: precompute `reachable[floor..=max_score]` by DP over
  `weights` (≤ 48 entries) so `count_eq` runs only for achievable subset-sums of `{1,2,3,4}`,
  not every integer — fewer plane clones.
- **Isolation enables real optimization:** plane layout, SIMD popcount, alternate counters, and
  the dual-BSI experiment are all local changes benchmarkable against synthetic bitmaps with zero
  SQLite in the loop — impossible today (the walk is entangled with the store).

---

## 3. Layer 2 — `trifle`: storage + streaming overlap output

### 3.1 The `doc`→`seg` flatten (fork F2, unanimous)

With payload/columns gone (§4), the `doc(id, key)` row carries nothing but the key. Fold `key`
directly onto `seg` and delete the `doc` table:

```sql
CREATE TABLE seg(
  id    INTEGER PRIMARY KEY,        -- the roaring posting id (monotonic; invariant #4)
  key   <keyty> NOT NULL,           -- the caller key (formerly doc.key)
  label TEXT NOT NULL, txt TEXT NOT NULL, len INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX seg_by_key ON seg(key);                      -- remove(key) / upsert lookup
CREATE UNIQUE INDEX seg_by_key_label ON seg(key, label);  -- (key,label) uniqueness
-- fwd, dict, term, post, delta: unchanged.
```

Consequences, all simplifying:

- **Invariant #2 (no-ghost) dissolves by construction.** There are *no* doc rows, so a
  payload/segment-less document cannot materialize a ghost row a later insert inherits. The
  entire guard apparatus (`set_doc_fields`, the empty-segs branch, `set_fields`'s create=false
  refusal, `remove_one_segment`'s reap, rebuild's payload-only rejection) deletes.
- **Provenance is a single-table point read:** `SELECT id, key, label FROM seg WHERE id IN
  rarray(?1)` replaces the `seg ⋈ doc` join, and the filter folds into it (§4).
- **Dedup keys on `Key`** (an `FxHashMap<Key, …>` over the small survivor set). Cost is hashing a
  `Text`/`Blob` key per survivor — survivors ≤ pulled depth (tens), negligible at trifle's small-
  document scale.
- **`rebuild`** stops reassigning a dense `doc` id space (only segment + term ids); duplicate
  `(key,label)` is caught by the unique index (keep the early `seen` check for a clear error).

The flatten is clean *because* there are no filter columns (F1). If low-churn typed columns ever
return as an opt-in (§4, out of v0.3 scope), they must live in a **doc-level side table**
`attr(key PRIMARY KEY, …)`, never on `seg` — putting columns on `seg` makes them per-segment, and
dedup-by-key would then make a filter's verdict depend on which segment won the score tie (a
nondeterminism bug the cross-review caught).

### 3.2 Drop the `B: Backend` generic; two lease shapes

`Shared` is cut (§4), so the only backend is `Sidecar`. Collapse `Index<T, B>` → `Index<T>` and
drop the `B::WriteGuard`/`B::ReadGuard` associated types threaded through every type — a
substantial readability win against the "radically simpler" mandate. (`T: Tokenizer` stays: it
is on the hot path and must monomorphize.) A thin internal `Store` seam remains for testing.

Leases (fork F6 + Core's concurrency-flaw fix): the warm connection belongs on the **stream**,
not on a merged reader, or concurrent searches on one reader serialize and `matches()` collides
with a live stream (tx-within-tx).

```rust
impl<T: Tokenizer> Index<T> {
    pub fn writer(&self) -> Result<Writer<'_, T>>;   // exclusive write lease — unchanged contract
    pub fn reader(&self) -> Result<Reader<'_, T>>;   // thin; each search/stream checks out its own pooled conn
}
```

Concurrency is preserved: each `matches()` checks out and returns a pooled conn; each live
stream holds its own pooled conn for its lifetime, so N concurrent streams use N pooled conns
(the pool already self-bounds read width). An optional warm `session()` for as-you-type bursts
can be added later; it is not needed for correctness.

### 3.3 The eager headline + the streaming spine (fork F6)

```rust
/// Provenance only — NO text (hydration is the opt-in terminal step, §3.4).
pub struct Candidate { key: Key, label: String, seg_id: u32, score: u32, overlap: u32 }
impl Candidate {
    pub fn key(&self) -> &Key;  pub fn label(&self) -> &str;
    pub fn score(&self) -> u32; pub fn overlap(&self) -> u32;   // weighted / raw
}

/// Query-shaping knobs only. `limit` is deliberately NOT here — it is a terminal-op argument
/// (`matches`/`matches_batch`/`collect_matches`), because the `candidates()` stream is
/// lazy/unbounded (the caller pulls depth via `take`). This is also what resolves the
/// limit-in-opts vs `collect_matches(limit)` mismatch.
#[non_exhaustive]
pub struct SearchOpts<'a> {
    pub min_shared:  Option<u32>,        // m
    pub t_max:       Option<usize>,      // selection breadth
    pub weight_step: f64,                // D
    pub filter:      Option<SqlFilter<'a>>,
}

impl<T: Tokenizer> Reader<'_, T> {
    /// THE SAFE DEFAULT (headline). Top-`limit` matches, text+span hydrated, weighted-overlap
    /// order. Drains its snapshot immediately; errors propagate; no per-item Result, no parked tx.
    pub fn matches(&self, query: &str, opts: &SearchOpts<'_>, limit: usize) -> Result<Vec<Match>>;
    pub fn matches_batch(&self, queries: &[&str], opts: &SearchOpts<'_>, limit: usize)
        -> Result<Vec<Vec<Match>>>;

    /// THE ARCHITECTURAL SPINE (opt-in power tool the brief mandates). A lazy, snapshot-pinned,
    /// best-first candidate cursor callers compose rerank / fusion / pagination on top of.
    pub fn candidates<'r>(&'r self, query: &str, opts: &SearchOpts<'_>)
        -> Result<CandidateStream<'r>>;
}

impl<'r> Iterator for CandidateStream<'r> {
    type Item = Result<Candidate>;   // best-first, deduped-per-key, filtered; FUSES on first Err
    fn next(&mut self) -> Option<Result<Candidate>>;
}
impl<'r> CandidateStream<'r> {
    pub fn n_segments(&self) -> u64;   // N, from THIS search's snapshot (not stats())
    pub fn avgdl(&self) -> f64;        // mean seg gram length, same snapshot
    pub fn present_terms(&self)  -> impl Iterator<Item = (&str, u64)>;            // selected token + df
    pub fn matched_terms(&self, c: &Candidate) -> impl Iterator<Item = (&str, u64)>;  // no SQL — postings in hand
    /// Terminal batched hydrate: one `WHERE id IN rarray(?1)` over EXACTLY the kept candidates.
    pub fn hydrate(&self, kept: &[Candidate]) -> Result<Vec<Match>>;
    /// Error-propagating collector (no silent truncation; see §6).
    pub fn collect_matches(self, limit: usize) -> Result<Vec<Match>>;
}
```

`matches()` is `candidates()` + take + `hydrate` internally, hydrating only `limit` rows.

### 3.4 Choose-then-hydrate (fork F5, unanimous)

`Candidate` is provenance-only; **text hydration is a separate batched terminal step**. This
strictly dominates the alternatives for the streaming use case (a deep-pool rerank):

```rust
let mut s = reader.candidates("quikc brown", &opts)?;
let pool: Vec<Candidate> = s.by_ref().take(200).collect::<Result<Vec<_>>>()?;  // provenance only; propagates Busy
let top10 = my_rerank(&pool, &s);          // reorder on score/overlap/matched_terms — NO text read
let hits  = s.hydrate(&top10[..10])?;      // ONE batched read, 10 rows — not 200
```

Pull-200-keep-10 hydrates 10 `txt` blobs in one query (per-chunk eager hydration would read 200;
per-item lazy `.text()` would be 200 round-trips). Everything the old `Ranker` / `scope` /
`Effort` / `QueryContext` apparatus did is now ordinary code over this stream:

- **scope** → filter the stream on `c.key()`/`c.label()` (handling each `Result` so a `Busy`
  still surfaces — do not `filter_map(Result::ok)`); the lazy walk keeps descending until `limit`
  pass, so scoping needs no over-fetch, exactly as before.
- **custom ranker** → pull a pool, reorder on `score`/`overlap`/`matched_terms`/`n_segments`/
  `avgdl`, hydrate the winners. (`matched_terms` carries each matched token's `df`, so an
  IDF-sum-style reranker — `tests/cycle2.rs::IdfSum` — keeps its inputs; this is flaw C, fixed.)
- **`Effort` over-fetch** → `take(200)`; the caller controls depth directly. Deleted, not ported.
- **precision tier** → `hits.retain(|m| m.text.contains("quick brown"))` after hydrate.

### 3.5 Snapshot/generation guard + batch==serial (invariants #1, #3 — preserved)

`candidates()` / `matches()` open one DEFERRED tx on the checked-out conn (one consistent WAL
snapshot), resolve terms in memory capturing `dict.generation`, then read the snapshot's stored
`dict_generation`; a skew ⇒ retryable `Error::Busy` (a concurrent id-reassigning `rebuild`,
invariant #3); re-check `poisoned` (FINDING-B). No `busy_timeout`, no internal retry (invariant
#6). Corpus signals (`n_segments`/`avgdl`) are read **inside this tx** (flaw D fix), so a
corpus-relative custom score never crosses a snapshot boundary. `batch == serial` holds: every
per-query input (selection, df, weights, filter) derives only from that query's tokens + the
shared snapshot; `matches_batch` shares the tx and the df read but selects per query.

---

## 4. Layer 3 — the opt-in raw-SQL filter (fork F1, reversed to no columns)

Delete the entire grammar (`Filter`/`FilterType`/`CmpOp`/`Cmp`/`In`/`Between`/`IsNull`/`Like`/
`Sql`/`And`/`Or`, `compile`/`build`, `CompiledFilter`/`filter_pass`/`filter_memo`), the `scope`
closure, **and** all trifle-stored filter columns/payload. Replace with one opt-in fragment over
the caller's **live** data:

```rust
pub struct SqlFilter<'a> {
    /// A TRUSTED-CONSTANT predicate fragment. The `seg` columns (id, key, label, txt, len) are in
    /// scope; co-located caller tables are reachable via subquery (Sidecar + ATTACH). Bind data
    /// through `params` only — never format values in.
    pub fragment: &'a str,
    pub params:   &'a [&'a dyn rusqlite::ToSql],   // e.g. an Rc<Vec<Value>> for `key IN rarray(?)`
}
// SearchOpts carries `filter` (full struct in §3.3); `limit` is a terminal-op arg, not a field.
```

### Why no stored columns (the design reversal)

trifle is a **derived, write-infrequent cache**. The attributes callers filter on (`due`,
`reps`, `deck`, `tags`) are **high-churn** and change *decoupled from text* (moving an Anki card
changes `deck`, not `front`/`back`). If trifle mirrored them, the caller's natural "text changed"
re-index trigger would miss the change, and trifle would silently serve **stale, wrong filter
results** — with no drift-reset to catch it (drift is schema-*shape* only). Filtering the
caller's live data is **staleness-free by construction**. Two modes, both zero-declaration:

- **Universal (any backend, incl. Sidecar):** the caller computes an allowed-key set in their own
  source of truth and binds it — `fragment = "key IN rarray(?)"`, `params = [&key_array]`. This
  subsumes the old `scope` predicate *and* arbitrary structured filters (run the structured query
  against your own DB, pass the keys). The key array is cacheable across an as-you-type session.
- **Co-located join (Sidecar + an optional `ATTACH` hook):** the caller's tables are attached to
  trifle's read connections, so the fragment joins directly and pushes the filter down with no
  key-array marshaling: `"key IN (SELECT note_id FROM src.cards WHERE deck = ? AND due <= ?)"`.

### Binding — fragment first, scope param last (fork F3, the one decisive fix)

The candidate id-set binds as the **last** positional parameter, with the **fragment textually
first**:

```sql
SELECT id FROM seg WHERE (<fragment>) AND id IN rarray(?{N+1})   -- N = params.len()
```

This is the one place the cross-review found all prior designs wrong (including the current audit-
F2 behavior). With the fragment first and the scope param at `?{N+1}`, **both** numbered
`?1..?N` *and* anonymous `?` placeholders bind correctly and **no caller can collide** with the
scope param — the caller pastes exactly the SQL they'd run standalone (including positional reuse
like `deck = ?1 OR backup = ?1`). `N+1` is trifle-computed from `params.len()`, never caller
input, so it is not an injection vector. (Putting the rarray *first*, or forcing anonymous-only
`?` from `?2`, re-ships the audit-F2 footgun — and Storage's headline `key IN rarray(?)` idiom is
exactly where a caller's natural `?1` would have broken.)

The filter folds into the per-bucket provenance SELECT (one query for provenance **and**
filtering), evaluated scoped to the bucket's candidate ids, so cost is `O(candidates pulled)`,
never `O(corpus)`. The walk early-stops at `limit` passing docs (scoping/filtering needs no
over-fetch). Caveat (unchanged, stated honestly): the filter prunes before hydrate/rank but does
not save the candidate-generation overlap work — that would need a partitioned index.

### Safety contract

The fragment is a **trusted compile-time constant** (the caller's own code, like a prepared-
statement string); data binds only through `params`; trifle never interpolates caller values into
SQL. The only identifiers trifle interpolates are the validated `Namespace` table names
(unchanged). There are no trifle-owned filter columns to validate.

### `Shared` → cut; co-location via `ATTACH` (fork F1 residual)

`Shared` is cut: it forces the `B: Backend` generic onto every public type (a real verbosity
tax), and its unique value — a staleness-free *pushed-down* co-located join — is recovered far
more cheaply by an **optional `ATTACH` on the Sidecar read-connection factory** (run once per
pooled read connection at creation). This keeps the efficient join-filter path while collapsing
to a concrete store. (Minority view: Filter preferred keeping `Shared` outright as the ~66-LOC
join path; the ATTACH hook meets that requirement with less surface, and a hard single-file
co-location requirement is the one documented reason to reintroduce a Shared-like backend later.)

---

## 5. Write path, compact, rebuild, stats

### Eight write methods → three (no payload)

```rust
impl<T: Tokenizer> Writer<'_, T> {
    pub fn upsert(&mut self, key: impl Into<Key>, segments: &[(&str, &str)]) -> Result<()>;  // create-or-replace; other labels intact
    pub fn remove(&mut self, key: impl Into<Key>) -> Result<()>;                              // drop key + all segments; absent = no-op
    pub fn remove_segment(&mut self, key: impl Into<Key>, label: &str) -> Result<()>;         // drop one segment; absent = no-op
    pub fn commit(&mut self) -> Result<()>;                                                   // commit-and-continue — unchanged
}
```

Gone: `insert`/`insert_segment`/`upsert_segment` (sugar `upsert(k, &[(l,t)])` covers; a derived
cache is re-derived and upserted, so error-on-collision guarded a moot bug class),
`insert_document`/`upsert_document`/`set_fields`/`set_doc_fields` (payload — deleted).
`Document` loses `payload`/`with_payload`; it stays `{ key, segments }` as the `rebuild` corpus
item. The `Writer` keeps its full atomicity machinery (`SAVEPOINT`/`atomic`, commit-and-continue,
`WriterStranded`, `TokenChanges` → `apply_writes` df fold) — those preserve invariants #2/#4 and
the no-sleeps contract and are **not** cut.

- `compact()` — **unchanged** (`postings::fold`; bounds delta backlog).
- `rebuild(corpus)` — unchanged mechanism (shadow swap, monotonic ids, generation bump,
  invariants #4/#5), minus the payload columns and the dense doc-id space (§3.1).
- `stats()` — drop `weight_step_hint` + the whole band-spread histogram; keep `segments`,
  `terms`, `delta_backlog`, `disk_bytes`, the four drift stamps.

---

## 6. Safety contracts (the cross-review's correctness fixes)

| id | contract | why |
|---|---|---|
| weight clamp | engine clamps every weight to `≥ 1` | weight 0 breaks weighted≥raw ⇒ valid result missing (wrong, not self-healing) |
| pool check-in rollback | the read pool runs a defensive `ROLLBACK` / `if !is_autocommit()` on **check-in** of a read conn (not only `Stream::Drop`) | `mem::forget` / `panic=abort` / double-panic bypass `Drop`, leaking a tx-bearing conn → next checkout inherits an open snapshot |
| stream fuses on first `Err` | after the first `Err`, the stream returns `None` thereafter | otherwise `filter_map(Result::ok)` yields a deceptively-complete prefix on a mid-stream transient `Busy` |
| eager methods return `Result<Vec>` | `matches`/`matches_batch`/`collect_matches` propagate errors | no silent candidate drop; the safe default never hides a `Busy` |
| same-snapshot corpus signals | `n_segments`/`avgdl` read inside the search tx, not `stats()` | a corpus-relative custom score must not cross a snapshot boundary |
| `hydrate` snapshot affinity | `hydrate(&[Candidate])` is `&self` on the stream; document "don't pass candidates across streams" (seg ids are snapshot-specific; a `rebuild` reassigns them) | low-severity cross-snapshot footgun |
| stream "drain promptly" | a live stream pins a WAL snapshot — trifle's, **and under ATTACH the caller's** too; document don't-park | parked streams block WAL checkpoint truncation on both DBs |

---

## 7. Performance argument (mechanism, not vibes)

1. **No SQL inside candidate generation.** Provenance + filter + text batch *once per bucket* (or
   once per kept set for text); the engine issues zero SQL. Fewer `prepare_cached` executions and
   round-trips on the hot path than today's per-bucket `hydrate_provenance` JOIN + separate
   `filter_pass`.
2. **Hydrate only what's kept.** Provenance-only `Candidate` + batched `hydrate(&kept)`: a
   pull-1000-keep-10 rerank reads 10 `txt` blobs, not 1000 (today every survivor is text-hydrated
   before ranking).
3. **One filter query, not two.** The fragment folds into the provenance SELECT; the separate
   `filter_pass` round-trip and `filter_memo` HashMap are deleted.
4. **One fewer table, one fewer join.** Provenance is a single-table point read on `seg` (no
   `seg ⋈ doc`); writes touch one fewer table.
5. **Lighter selection.** Removing class-`z` rarity drops a per-token `ln` + `ClassSnap` lookup +
   `f64 partial_cmp` per query token, replaced by an integer-df sort; no per-query `ClassSnap`
   allocation, no band-spread atomic increment per search.
6. **Caller-controlled depth replaces speculative over-fetch.** No `Effort` `c·√(limit·N)`
   heuristic over-hydrating "just in case"; the consumer pulls exactly to its need.
7. **Isolation enables engine optimization.** `&[RoaringBitmap] → Iterator<Scored>` is profilable
   with zero SQLite — plane layout, SIMD popcount, reachable-bucket skipping, the dual-BSI
   experiment — none possible today.
8. **Flatness preserved + sharpened** (reachable-bucket skipping; `O(k·log k)` build independent
   of cardinality).

---

## 8. Deletion list with honest LOC accounting

Against current `src/` (8,052 LOC). "cut" = deleted; "moved" = relocated to `trifle-overlap`.

| # | item | ~LOC | risk | invariant |
|---|------|-----:|------|-----------|
| 1 | `Filter` grammar (`FilterType`/`CmpOp`/`Filter`+builders+`compile`/`build`) + `CompiledFilter`/`filter_pass`/`filter_memo` + filterable cols/indexes + compile glue | 400 | med — callers rewrite as raw SQL (strict superset) | none (Tier-2 only) |
| 2 | payload + write-method collapse (`insert_document`/`upsert_document`/`write_document`/`set_fields`/`set_doc_fields`, `Document.payload`, rebuild payload binding, `insert`/`*_segment` variants) + **ghost-row guards** | 250 | med — API churn | **#2 dissolves** |
| 3 | drop the `doc` table (`doc_id_for`, the `seg⋈doc` join, doc DDL/indexes/shadow/swap, dense doc-id reassignment) | 130 | high→**low** (clean once columns gone) | **#2 trivial**, #4 ok |
| 4 | `welford.rs` + `ClassStats`/`ClassSnap` threading (dict/select/search) | 320 | med — mixed-script recall *may* regress; within-script identical; doc admits empirical | #1 holds (select stays per-query) |
| 5 | `Ranker`/`OverlapRanker`/`Candidates`/`Candidate`/`QueryContext`/`Ranked` + `rank_to_matches` glue | 250 | low — signals move to stream (`matched_terms`/`n_segments`/`avgdl`) | none |
| 6 | band-spread telemetry (`WeightStepHint`, `band_spread_hist`, `observe`/`reset`/`weight_step_hint`, `HINT_*`, quantile, `Stats` field) | 170 | low — advisory | none |
| 7 | `Effort` enum + `coeff`/`pool` + `rerank` setter + tests | 120 | low — laziness supersedes | none |
| 8 | `scope` closure (`ScopeFn`, field/setter, walk application) | 35 | low — `stream.filter()` | #1 ok |
| 9 | `Shared` backend + `B: Backend` generic threading + re-exports | 130 | med — co-location via ATTACH instead | none |
| 10 | `SearchSession` (warm conn folds into the stream) | 50 | low | none |
| — | engine **moved** to `trifle-overlap` (`add_weighted`/`weighted_overlap`/`tier_weights`/`count_eq`/walk + tests) | ~300 moved | low | #1, #7 preserved |
| | **public-crate src removed** | **≈ 1,855 cut + ~300 moved** | | |

**End state (faithful, no tokenizer trim).** The itemized rows above remove **1,855 cut + ~300
moved = ~2,155 LOC** → public `trifle` ≈ **5,900 src (~1.37× whole-crate) from deletions alone**.
The `Index<T,B>`→`Index<T>` generic removal threads through *every* signature and enables leaner
`lib.rs`/`search.rs`/`model.rs` rewrites on retained code — an **estimated** further **~500–700
LOC** (not itemized above because it is reshaping, not deletion, and so is less certain than the
deletions), landing public `trifle` ≈ **5,200–5,400 src (~1.5×)**, plus `trifle-overlap` ≈ **300**
(moved, net wash). The **test suite collapses ~1.8× in isolation** (3,045 → ~1,700; `cycle2.rs`'s
~560 LOC of payload/ghost permutations → ~3 focused tests), so **src+tests combined is ~1.5×**
(11,097 → ~6,900–7,600) — tests shrink less than src, so adding them does not raise the whole-crate
ratio. On the **control-plane** surface the brief targets (`lib`+`model`+`rank`+`schema`+`search`+
`select`+`dict`+`welford`+`store/mod`, ~5,500 today) the refactor lands ≈ **2,800–3,100 → ~1.75–2×**.

**Landing the 2–3× claim — the honest headline:**
> A faithful simplification (no capability loss) is **~1.37× whole-crate from deletions alone,
> ~1.5× with the estimated structural reshaping**; the **test suite itself collapses ~1.8×**
> (src+tests combined ~1.5×); **~1.75–2× on the targeted control-plane**. The literal whole-crate **2×** is
> reachable only with the **optional, benchmark-gated tokenizer trim** (fix to Trigram + one
> normalization form; drop Bigram/Quadgram, NFD, accent-strip) — another ~500–700 LOC for
> `tokenize.rs` 1,171 → ~500 → ~1.9–2.0× — which is a **capability cut** (accent-fold recall,
> alternate n-gram sizes, multi-script normalization), gated on the fuzzy-recall eval. **3× is not
> achievable** without dismantling `postings`/`schema`/`store` bedrock the invariants forbid.

---

## 9. Resolved design forks (the equilibrium)

| fork | decision | rationale | dissent |
|---|---|---|---|
| F1 filter storage | **No trifle-stored columns**; filter caller's live data via `key IN rarray(?)` / co-located join | attribute churn + the staleness flaw make stored columns a silent wrong-results bug | low-churn typed columns a possible out-of-scope future (doc-level side table only) |
| F1 `Shared` | **Cut**; co-location via optional `ATTACH` hook | drops the `B` generic tax; ATTACH preserves the join path | Filter preferred keeping `Shared` (~66 LOC) |
| F2 flatten | **Flatten `doc`→`seg`** | no-ghost #2 dissolves by construction; single-table provenance | — (unanimous after F1) |
| F3 binding | **Fragment first, scope param `?{N+1}` last** | both numbered & anonymous `?` bind, no collision; kills the audit-F2 footgun | — (unanimous) |
| F4 engine lifetime | **Counter owns postings, single-BSI**; raw overlap via `contains` | `'static`, no ouroboros, no 2× memory; shallow-pull optimal | dual-BSI kept as benchmarked alt for deep pulls |
| F5 hydration | **Provenance-only stream + batched `hydrate()`** | strictly dominates eager-per-chunk and per-item lazy | — (unanimous) |
| F6 headline | **Eager `matches()` default; streaming spine secondary** | streaming's footguns (WAL pin, per-item Result) make eager the safe front door | — (unanimous) |
| F7 target | **~1.37–1.5× faithful whole-crate / ~1.75–2× control-plane / ~2× with gated tokenizer trim; never 3×** | invariants forbid cutting bedrock | — (unanimous) |

---

## 10. Invariant preservation check

| # | invariant | status |
|---|---|---|
| 1 | batch == serial | preserved (per-query selection/df/weights/filter; shared snapshot only) |
| 2 | no-ghost doc rows | **dissolved** (no doc rows exist) |
| 3 | dict-generation guard | preserved (same `search_read_on` logic in the stream constructor) |
| 4 | monotonic id + shadow swap | preserved (rebuild reassigns seg+term ids; unique index on `(key,label)`) |
| 5 | drift-reset | preserved (schema fingerprint drops payload/columns from its inputs) |
| 6 | no sleeps / `Error::Busy` | preserved (no `busy_timeout`, no internal retry; stream surfaces `Busy`) |
| 7 | flatness | preserved + sharpened (reachable-bucket skip; engine benchable in isolation). Spike refined the claim: op-count is cardinality-independent; wall-clock sublinear (sparse) / flat (dense), not "independent of size" |
| 8 | single tokenizer | preserved (unchanged) |

---

## 11. Open risks / benchmark-gated decisions

1. **Tokenizer trim (the path to literal 2×).** A *capability* cut (accent-fold / multi-script /
   alt n-gram recall). Gate on the fuzzy-recall eval (`geonames-cities` + a mixed-script set),
   per the `tmax-pool-sweep-methodology` discipline. Do **not** cut it blind for a headline number.
2. **Cutting class-normalized rarity (welford).** Reverts selection to rarest-by-raw-df
   (within-script identical; the doc itself calls cross-script value "an empirical question").
   Cut and re-validate against the eval; it re-enters cleanly (now isolated in `select`/`dict`,
   not smeared across three modules) as an explicit-weights path if the data warrants.
3. **dual-BSI vs owns-postings (engine).** Owns-postings is the default (shallow-pull optimal);
   if a deep-pull profile shows per-id `contains` dominating, the dual-BSI second plane stack is
   the drop-in alternative behind the same `'static` Counter. Benchmark-decidable.
4. **Raw filter is untyped & version-coupled.** A malformed fragment is a runtime error and it
   couples callers to trifle's column names (`key`, `seg`). Acceptable for a derived-cache power
   feature (the old `Filter::Sql` already carried this); now it is the only filter, so the median
   caller has fewer guardrails — mitigated by the `key IN rarray(?)` default being simple and safe.
5. **Universal key-set marshaling.** A large allowed-key array is marshaled per search (mitigated:
   cacheable across an as-you-type session; the ATTACH join avoids it entirely).

---

## 12. Phased implementation plan

1. **Extract `trifle-overlap`.** Move the pure functions; add the owned `Counter` + `Walk` +
   weight clamp + reachable-bucket skip; criterion bench on synthetic bitmaps. (Behavior-neutral.)
2. **Delete the unproven/advisory subsystems.** `welford.rs` + class threading, band-spread
   telemetry, `Effort`, `Ranker` machinery, `scope`. Selection reverts to raw-df. Run the recall
   eval to confirm no regression before/after.
3. **Flatten `doc`→`seg`; drop payload + the 5 payload/segment write methods; drop the `B`
   generic + `Shared`.** Rewrite `schema.rs` DDL; collapse the write API to 3 methods.
4. **Replace the filter grammar with `SqlFilter`** (fragment-first `?{N+1}` binding) + the
   optional `ATTACH` hook. Delete `model.rs`'s grammar.
5. **Build the `CandidateStream`** (owns conn + Counter; provenance-only; `hydrate`/`collect_matches`;
   fuse-on-Err) and the eager `matches()`/`matches_batch()` on top; add the pool check-in rollback.
6. **Rewrite the test suite** around the new surface (the `cycle2.rs` payload/ghost permutations
   collapse to a few focused tests; `tests/thrash.rs` proptest oracle updated to the 3-method
   write API).

Each phase is independently shippable and leaves the crate green under the load-bearing lint bar
(`-D warnings`, `clippy::all`, rustdoc).
