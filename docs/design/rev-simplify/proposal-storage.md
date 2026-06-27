# Proposal — storage core + streaming candidate API (rev v0.2 → v0.3)

**Agent: Storage.** Lens: the public crate as a lean SQLite storage engine that feeds a
pure overlap engine and emits a *streaming candidate iterator* callers build on. This
proposal is complete (all three layers, the filter, the write path, perf, risks) but goes
deepest on the storage surface, the read/write API, and how the stream is produced lazily
while honoring the snapshot/generation guard and `batch == serial`.

The headline idea: **make laziness the organizing principle.** A search returns a lazy,
score-descending stream of provenance-carrying candidates. Doing so *deletes* three whole
subsystems for free — `Effort` over-fetch (the caller now controls depth directly), the
`Ranker` trait machinery (the caller reorders the stream), and the `scope` predicate (the
caller `.filter()`s it) — because every one of them existed only to compensate for an
*eager, fixed-`limit`, baked-in-ranking* result list.

---

## 1. Target layout — three layers

```
trifle-overlap/           # Layer 1: the crown jewel, pure & isolated
  src/lib.rs              #   Counter, Scored, the bit-sliced walk. dep: roaring ONLY.

trifle/  (the public crate)
  src/lib.rs              # Layer 2: Index<T,B>, Config, SearchOpts, Stats, Writer, Reader
  src/search.rs           #   stream production: resolve→select→Counter→hydrate, guarded
  src/postings.rs         #   owned roaring inverted index (base∪added\removed) — unchanged
  src/schema.rs           #   DDL, drift stamps, shadow swap, monotonic ids — trimmed
  src/dict.rs             #   faulting term dict (gram u128→u32) — class plumbing removed
  src/select.rs           #   rarest-first by raw df — class-normalization removed
  src/store/{mod,pool,sidecar,shared}.rs   # Backend trait + pool — kept (see §4, §6)
  src/tokenize.rs         #   NgramTokenizer — owned by the Tokenizer agent
  src/term.rs src/error.rs src/hash.rs src/instrument.rs   # unchanged
  src/model.rs            # Key/KeyShape/Document/Match/Schema — Filter grammar deleted
  # DELETED: src/welford.rs, src/rank.rs (split into trifle-overlap + folded into search.rs)
```

**Dependency direction (strict, acyclic):** `trifle-overlap` depends on `roaring` only and
knows *nothing* of SQL, provenance, text, or `Index`. `trifle` depends on `trifle-overlap`
and wires postings (read from SQLite) into `Counter`. The overlap engine can be fuzzed,
benchmarked, and optimized (SIMD popcount, plane layout, alternate counters) with a handful
of synthetic `RoaringBitmap`s and zero database. That isolation is the point.

---

## 2. Layer 1 — the pure overlap engine (`trifle-overlap`)

Everything the brief names the crown jewel — `add_weighted`, `weighted_overlap`,
`tier_weights`, `count_eq`, the high→low bucket walk — moves here verbatim, behind one type.
**Zero SQL, zero provenance, zero hydration, zero `String`.** The only boundary types are
`u32` ids and `RoaringBitmap` references.

```rust
use roaring::RoaringBitmap;

/// One scored candidate: a posting id and its two overlap measures. `Copy`, 12 bytes,
/// no allocation, no provenance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Scored {
    pub id: u32,       // trifle's segment id; opaque here
    pub score: u32,    // IDF-weighted overlap (the ordering key)
    pub overlap: u32,  // raw count of postings containing `id`
}

/// The IDF-weighted bit-sliced overlap counter over a set of postings.
pub struct Counter<'a> {
    postings: &'a [&'a RoaringBitmap],   // borrowed for the raw-overlap check; no copy
    planes:   Vec<RoaringBitmap>,        // weighted count, bit-sliced (the only allocation)
    weights:  Vec<u32>,
    min_overlap: u32,
    max_score:   u32,
}

impl<'a> Counter<'a> {
    /// Weight each posting by per-query df rarity (df = posting cardinality, by the
    /// monotonic-id contract; knob `D = weight_step`) via `tier_weights`, then accumulate
    /// the weighted bit-sliced counter. `O(k·log k)` bitmap ops, **independent of posting
    /// size** (the flatness claim). `min_shared` is the raw-overlap floor.
    pub fn build(postings: &'a [&'a RoaringBitmap], weight_step: f64, min_shared: u32) -> Self;

    /// The IDF tier weights used (parallel to `postings`) — for benches/tests.
    pub fn weights(&self) -> &[u32];

    /// Stream candidates in weighted-score-descending order (ties by ascending id), each
    /// meeting the raw `min_shared` floor. **Lazy:** each step peels the next non-empty
    /// score bucket with one `count_eq`, then walks that bucket's ids; a consumer that
    /// stops pulling at `k` never computes a lower bucket.
    pub fn stream(&self) -> impl Iterator<Item = Scored> + '_;
}
```

**Why this is the right boundary.** The current `overlap_search` interleaves the pure
counter with `hydrate_provenance` (SQL), `filter_pass` (SQL), the `scope` closure, and
per-doc dedup — none of which belong to "count overlap." Pulling those *up* into Layer 2
leaves a counter whose entire surface is `&[&RoaringBitmap] → impl Iterator<Item=Scored>`.

**Flatness, preserved and now provable in isolation.** `build` is `weighted_overlap`:
`popcount(w) ≤ 2` ripples per posting for `w ∈ 1..=4`, each ripple an `O(containers)`
bitmap XOR/AND ⇒ `O(k·log k)`. `stream` runs `count_eq` (`O(planes)=O(log k)`) per
non-empty bucket, ≤ `Σweights ≤ 4k` buckets. Raw overlap per *yielded* id is `k` `contains`
calls. All three are independent of posting cardinality. A criterion bench over synthetic
bitmaps of fixed `k` and varying cardinality is now a 30-line file in the sub-crate.

**Laziness is load-bearing.** The eager `overlap_search` materialized the whole top-`limit`
set and needed `Effort` to over-fetch for a reranker. A lazy `stream` lets the consumer pull
exactly as deep as its rerank/filter/scope needs — `Effort` is *deleted*, not ported.

---

## 3. Layer 2 — public storage + overlap, streaming output

### 3.1 Leases: two, not three

Collapse `Reader` + `SearchSession` into a single **`Reader`** that holds a warm pooled
connection for its lifetime (the strictly-better `SearchSession` behavior: no per-search
checkout). Keep **`Writer`** (the exclusive write lease) exactly as designed. So: one read
lease, one write lease.

```rust
impl<T: Tokenizer, B: Backend> Index<T, B> {
    pub fn writer(&self) -> Result<Writer<'_, T, B>>;   // unchanged contract
    pub fn reader(&self) -> Result<Reader<'_, T, B>>;   // now holds a warm read connection
}
```

A `Reader` owns one pooled connection; one live `CandidateStream` per reader at a time
(SQLite: one transaction per connection). Concurrent searches acquire multiple readers
(multiple pooled connections) — the existing pool already self-bounds to live read width.

### 3.2 `search` returns a stream; `matches` is the convenience

```rust
pub struct Candidate {            // provenance only — text is NOT here (opt-in, §3.3)
    key: Key, label: String, seg_id: u32, score: u32, overlap: u32,
}
impl Candidate {
    pub fn key(&self) -> &Key;
    pub fn label(&self) -> &str;
    pub fn score(&self) -> u32;     // IDF-weighted overlap
    pub fn overlap(&self) -> u32;   // raw shared-token count
}

impl<T: Tokenizer, B: Backend> Reader<'_, T, B> {
    /// The lazy candidate stream for `query`: IDF-weighted overlap candidates,
    /// score-descending, **deduped to one per document** (best segment), provenance
    /// hydrated per bucket on this search's snapshot. The building block.
    pub fn search<'r>(&'r self, query: &str, opts: &SearchOpts<'_>)
        -> Result<CandidateStream<'r>>;

    /// Convenience: top-`limit` matches with text+span hydrated, in weighted-overlap order
    /// (what most callers want). `= search(q,opts)?.by_ref().take(limit).collect()? then hydrate`.
    pub fn matches(&self, query: &str, opts: &SearchOpts<'_>, limit: usize)
        -> Result<Vec<Match>>;

    /// `matches` for a batch under one shared snapshot + one shared term/df read; each
    /// query's result is identical to calling `matches` alone (batch == serial).
    pub fn matches_batch(&self, queries: &[&str], opts: &SearchOpts<'_>, limit: usize)
        -> Result<Vec<Vec<Match>>>;
}

pub struct CandidateStream<'r> { /* owns the snapshot tx; see §3.4 */ }

impl<'r> Iterator for CandidateStream<'r> {
    type Item = Result<Candidate>;        // each step may run a batched provenance/filter query
    fn next(&mut self) -> Option<Result<Candidate>>;
}
impl<'r> CandidateStream<'r> {
    pub fn n_segments(&self) -> u64;                       // N, for a corpus-relative custom score
    pub fn avgdl(&self) -> f64;                            // mean seg gram length
    pub fn present_terms(&self) -> impl Iterator<Item=(&str,u64)>;          // selected token + df
    pub fn matched_terms(&self, c:&Candidate) -> impl Iterator<Item=(&str,u64)>;  // no SQL
    /// Hydrate text + span (vs the query) of the given candidates in ONE batched read on
    /// this stream's snapshot — the terminal step that builds `Match`es.
    pub fn hydrate(&self, cands: &[Candidate]) -> Result<Vec<Match>>;
}
```

### 3.3 How a caller hydrates / ranks / filters — all *above* trifle

The `Item = Result<Candidate>` shape composes with std iterator combinators. Everything the
old `Ranker`/`scope`/`Effort` did becomes ordinary code over the stream:

```rust
let mut s = reader.search("quikc brown", &opts)?;

// scope (old SearchOpts::scope):  filter, the lazy walk keeps descending until k pass
let kept: Vec<Candidate> = s.by_ref()
    .filter_map(Result::ok)
    .filter(|c| allow.contains(c.key()))      // membership predicate
    .take(10)
    .collect();

// rank (old custom Ranker): hydrate the kept set, then reorder however you like
let mut hits = s.hydrate(&kept)?;
hits.sort_by(|a,b| my_score(b).total_cmp(&my_score(a)));   // RRF, BM25, proximity, …

// precision tier (old LiteralOnly Ranker):  filter on hydrated text
hits.retain(|m| m.text.contains("quick brown"));
```

`matched_terms(&c)` (token+df, no SQL — the postings are already in hand) and
`n_segments`/`avgdl` give a custom scorer the same signals the old `QueryContext`/`Candidate`
exposed. **Text hydration is opt-in and batched:** `Candidate` is provenance-only; `hydrate`
runs one `WHERE id IN rarray(?1)` over exactly the candidates the caller chose to keep — so a
caller that pulls 1000 candidates but keeps 10 hydrates 10 rows, not 1000. The default
`matches`/`matches_batch` do this for you (weighted-overlap order, `limit` rows).

### 3.4 Producing the stream lazily under the snapshot/generation guard (inv. #1, #3)

`CandidateStream<'r>` owns an `unchecked_transaction()` borrowing the reader's warm
connection — one consistent WAL snapshot for the stream's whole life. Construction:

1. tokenize `query`; resolve its distinct terms in-memory, **capturing `dict.generation`**;
   open the tx (its first statement pins the WAL snapshot); read the snapshot's
   `dict_generation`. **Skew ⇒ `Error::Busy`** (a concurrent id-reassigning `rebuild` —
   inv. #3), caller retries on a fresh reader. Re-check `poisoned` here too (FINDING-B).
2. one batched `read_dfs` over the query's term-ids; `select` rarest-first (raw df, §5);
   one batched `effective_postings` over the selected ids ⇒ the `present` postings.
3. `Counter::build(&present_bitmaps, opts.weight_step, m)`; store the pure `stream()`.

Pulling (`next`): drive the pure `Scored` stream; **group equal-score runs** (the stream is
score-descending, so a bucket is a maximal equal-`score` run) and, per run, run **one**
`SELECT id, key, label FROM seg WHERE id IN rarray(?1) [AND (<filter>)]` (provenance **and**
the opt-in filter folded into the same query — §4). Dedup by `key` (keep the first, i.e.
highest-score lowest-id, segment per document — the old per-doc dedup, now keyed on the
caller key since the `doc` table is gone, §5), yield deduped `Candidate`s. The walk stops
when the consumer stops pulling — so `take(k)` reproduces the old "stop once `limit` docs
lock" early-exit *for free*, and `batch == serial` holds because every per-query input
(selection, df, weights, filter) derives only from that query's tokens + the shared snapshot
(inv. #1 — `matches_batch` shares the tx and the resolved/df reads but selects per query).

**Lifetime caveat (documented):** a live stream pins its connection's tx (and WAL snapshot).
Pull promptly; don't park a stream across long pauses (it blocks WAL checkpoint truncation).
The `matches`/`matches_batch` convenience drains immediately, so the common path never parks.

---

## 4. Layer 3 — the opt-in raw-SQL middle-tier filter

Delete the entire grammar (`Filter`/`FilterType`/`CmpOp`/`Cmp`/`In`/`Between`/`IsNull`/
`Like`/`Sql`/`And`/`Or`, `compile`/`build`, `CompiledFilter`/`filter_pass`/`filter_memo`)
**and** the `filterable` schema columns / payload. Replace with one opt-in raw fragment:

```rust
pub struct SqlFilter<'a> {
    /// A TRUSTED-CONSTANT predicate fragment over `seg` (columns `id`, `key`, `label`,
    /// `txt`, `len` in scope; co-located caller tables reachable via subquery). Use
    /// anonymous `?` placeholders (renumbered after the internal candidate-scope param).
    pub fragment: &'a str,
    /// Bound parameters in `?` order (data only — including an `Rc<Vec<Value>>` for a
    /// `key IN rarray(?)` set).
    pub params: &'a [&'a dyn rusqlite::ToSql],
}
// SearchOpts gains:  pub filter: Option<SqlFilter<'a>>
```

**Binding & cost.** The filter is *folded into the per-bucket provenance SELECT*:
`SELECT id, key, label FROM seg WHERE id IN rarray(?1) AND (<fragment>)`. Candidate seg-ids
bind as `?1`; the fragment's anonymous `?` renumber from `?2`. This is **one query for
provenance *and* filtering** (the old design ran a separate `filter_pass` + a `filter_memo`
HashMap — both deleted). Cost is `O(candidates hydrated)`, bounded by pull depth, never
`O(corpus)`.

**What must the caller declare? Nothing.** trifle is a *derived cache over caller-owned
data*, so filter attributes live in the caller's source of truth, joined on the key. Two
modes, no schema declaration either way:

- **Universal (any backend, incl. `Sidecar`):** the caller computes an allowed-key set in
  *their own* store and passes it: `fragment = "key IN rarray(?)"`, `params = [&key_array]`.
  This subsumes both the old `scope` predicate (membership) and structured filters (run the
  structured query against your own DB, pass the resulting keys).
- **Co-located join (`Shared`, or `Sidecar` + `ATTACH`):** the caller's tables are in the
  same SQLite file, so the fragment joins directly:
  `"key IN (SELECT note_id FROM cards WHERE deck = ? AND due <= ?)"`.

**Injection/safety.** The fragment is a trusted constant (the caller's own code, like a
prepared-statement string); *data* binds only through `params`. trifle never interpolates
caller values into SQL. There is no `filterable`-column validation to maintain because there
are no trifle-owned filter columns — the only identifiers trifle interpolates are the
validated table names (`Namespace`), unchanged.

This is also the argument to **keep `Shared`** (the brief flagged it for cutting): the
co-located join is the filter's most powerful mode and *needs* co-location. `Shared` is ~66
LOC and the per-connection machinery (`Store`/pool) is already shared with `Sidecar`, so
keeping it costs almost nothing and preserves the filter's killer feature. (If the team still
wants it gone, the universal key-set mode keeps filtering fully functional on `Sidecar`.)

---

## 5. Write path + storage core after simplification

### 5.1 Drop the `doc` table — collapse to a single keyed `seg` table

With payload gone, `doc(id, key)` carries nothing but the key. **Fold `key` directly onto
`seg`** and delete the `doc` table, its `doc_by_key` index, and `doc_id_for`:

```sql
CREATE TABLE seg(
  id INTEGER PRIMARY KEY,           -- the roaring posting id (monotonic, inv. #4)
  key <keyty> NOT NULL,             -- the caller key, formerly doc.key
  label TEXT NOT NULL, txt TEXT NOT NULL, len INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX seg_by_key ON seg(key);                       -- remove(key), upsert lookup
CREATE UNIQUE INDEX seg_by_key_label ON seg(key, label);   -- (key,label) uniqueness
-- fwd(id, tokens), dict(id, gram), term(id, df), post(id, base), delta(id, added, removed): unchanged
```

Consequences, all simplifying:

- **No-ghost invariant (#2) becomes trivial.** There are *no doc rows*, so a payload/
  segment-less document cannot materialize a ghost row a later insert inherits. The entire
  guard apparatus — `set_doc_fields`, the `write_document` empty-segs branch, `set_fields`'s
  create=false refusal, `remove_one_segment`'s "reap the empty doc row," rebuild's
  payload-only rejection — all delete. The invariant is satisfied by construction.
- **Provenance hydration loses its join.** `SELECT id, key, label FROM seg WHERE id IN
  rarray(?1)` replaces the `seg ⋈ doc` join — one table, faster, and the filter folds into it.
- **Dedup keys on `Key`** instead of an internal `u32` doc id (a `FxHashMap<Key,…>` over the
  small survivor set). Cost: hashing `Key` (a `Text`/`Blob` key may be a few dozen bytes) for
  ≤ pulled-depth survivors — negligible at trifle's scale; see Risks.
- **`rebuild`** stops reassigning a dense `doc` id space (only segment + term ids); the doc
  INSERT becomes `INSERT INTO seg_shadow(id, key, label, txt, len)`. Duplicate `(key,label)`
  caught by the unique index (keep the early `seen` check for a clear error).

### 5.2 Eight write methods → three

```rust
impl<T: Tokenizer, B: Backend> Writer<'_, T, B> {
    /// Insert-or-replace the given (label,text) segments under `key`; other labels intact.
    pub fn upsert(&mut self, key: impl Into<Key>, segments: &[(&str, &str)]) -> Result<()>;
    /// Remove `key` and all its segments. Absent key = no-op.
    pub fn remove(&mut self, key: impl Into<Key>) -> Result<()>;
    /// Remove the single segment (key,label). Absent = no-op; siblings intact.
    pub fn remove_segment(&mut self, key: impl Into<Key>, label: &str) -> Result<()>;
    pub fn commit(&mut self) -> Result<()>;   // commit-and-continue — unchanged
}
```

`insert`/`insert_segment`/`upsert_segment` (sugar; `upsert(k,&[(l,t)])` covers them),
`insert_document`/`upsert_document`/`set_fields` (payload — gone) all delete. A derived cache
is *re-derived and upserted*; the error-on-collision variants guarded a bug class that
`upsert` idempotency makes moot. The `Writer` keeps its full atomicity machinery
(`SAVEPOINT`/`atomic`, commit-and-continue, `WriterStranded`, the `TokenChanges` →
`apply_writes` df fold) — those preserve inv. #2/#4 and the no-sleeps contract and are *not*
cut. `Document` loses its `payload` field and `with_payload`; it stays `{key, segments}` for
`rebuild` and is the corpus item type.

### 5.3 compact / rebuild / stats

- `compact()` — **unchanged** (`postings::fold`; bounds delta backlog; inv. nothing).
- `rebuild(corpus)` — unchanged mechanism (shadow swap, monotonic ids, generation bump —
  inv. #4/#5), minus the payload columns and dense doc-id space (§5.1).
- `stats()` — drop `weight_step_hint`; keep `segments`, `terms`, `delta_backlog`,
  `disk_bytes`, the four drift stamps. `WeightStepHint`, the band-spread histogram, and all
  its plumbing delete (§6).

---

## 6. Deletion list with LOC accounting

Current: **8,052 LOC src** (≈ 6,900 logic + ≈ 1,150 inline tests). Estimates are net of the
reshaping each cut forces.

| # | Cut | ~LOC | Risk | Invariant touched |
|---|-----|-----:|------|-------------------|
| 1 | **`Ranker` trait + `Candidates`/`Candidate`/`QueryContext`/`Ranked`/`OverlapRanker`** (rank.rs) + `rank_to_matches` glue (search.rs) | **250** | Callers must reorder the stream themselves; we provide `matched_terms`/`n_segments`/`avgdl` so no signal is lost | none |
| 2 | **`Filter` grammar** (`FilterType`/`CmpOp`/`Filter` enum+builders+`compile`/`build`, model.rs) + `CompiledFilter`/`filter_pass`/`filter_memo` (rank.rs) + `filterable` cols/indexes (schema.rs) + compile call (search.rs) | **400** | Loses typed/validated filters; raw fragment is untyped & is the injection surface (trusted-constant contract) | none (was Tier-2 only) |
| 3 | **Payload + write-method collapse**: `insert_document`/`upsert_document`/`write_document`/`set_fields`/`set_doc_fields`, `Document.payload`/`with_payload`, rebuild payload binding, ghost-row guards, `insert`/`*_segment` variants | **220** | Smaller write API; "error on collision" gone | **#2 simplified** (no ghosts possible) |
| 4 | **Drop the `doc` table**: `doc_id_for`, `doc`/`seg` join, doc DDL/indexes/shadow/swap, dense doc-id reassignment | **120** | Dedup now hashes `Key` not a `u32`; key repeats per segment on `seg` | **#2 trivial**, #4 unaffected |
| 5 | **`welford.rs` + class-normalized rarity** + `ClassStats`/`ClassSnap` threading (dict/select/search) | **270** | Cross-script mixed-query recall *may* regress; within-script identical; doc admits it's empirical | #1 still holds (select stays per-query) |
| 6 | **Band-spread telemetry**: `WeightStepHint`, `band_spread_hist`, `observe_band_spread`, `reset_band_spread_hist`, `weight_step_hint`, `HINT_*`, the quantile code, `Stats` field | **160** | Loses the corpus-fitted `D` suggestion (advisory only) | none |
| 7 | **`Effort` over-fetch**: enum + `coeff`/`pool` + `rerank` setter + field + tests | **120** | None — laziness supersedes it (caller pulls to depth) | none |
| 8 | **`scope` predicate**: `ScopeFn`, field/setter, application in walk | **30** | Becomes `stream.filter(…)` | inv. #1 preserved |
| 9 | **`Reader`/`SearchSession` collapse** + `search_batch`/`search_batch_on` split | **60** | One live stream per reader (acquire more for concurrency) | none |
| | **Total removed** | **≈ 1,630** | | |

End state per file (logic, excl. moved): rank.rs 720 → ~30 (split out); model.rs 789 → ~400;
welford.rs 201 → 0; lib.rs 1869 → ~1,050; dict.rs 291 → ~225; select.rs 280 → ~190;
schema.rs 559 → ~480; search.rs 338 → ~330 (gains stream machinery). New `trifle-overlap` ≈
280 (engine + its tests, *moved* from rank.rs, not new code). **Net trifle src ≈ 5,900
(−27%), or ~1.4× smaller.**

**Landing the 2–3× claim — honestly.** My surfaces (query pipeline knobs, filter, ranker,
payload, telemetry, class-norm, leases) are ~40% of the non-tokenizer logic and I cut ~1,630
of them. The remaining ~3,000 LOC core — `tokenize.rs` (1,171), `postings.rs` (623), the
`store` (731), `term`/`error`/`hash` (461) — is largely irreducible *without* sacrificing an
invariant or the tokenizer's capability. **A true 2–3× is reached only in combination with
the Tokenizer agent's cuts** (fix to trigram + one normalization ⇒ `tokenize.rs` ~1,171 →
~450, another ~700). Combined: ~2,300 removed ⇒ ~5,750 ⇒ **~1.7–2×**. I recommend targeting
**~1.8×** and treating the 3-layer restructure + streaming output as the primary win;
chasing 3× into `postings`/`store` would break inv. #2/#3/#4. (See Risks.)

---

## 7. Performance argument (mechanism, not just smaller)

1. **The overlap engine optimizes in isolation.** Once `Counter` is `&[&RoaringBitmap] →
   Iterator<Scored>` with no SQL in scope, plane layout, SIMD popcount, a fused raw+weighted
   counter, and alternate bucket strategies are all local changes benchmarkable against
   synthetic bitmaps. Today every such experiment drags the whole SQLite pipeline along.
2. **One SQL query per bucket, not two.** Folding the filter into the provenance SELECT
   removes the separate `filter_pass` round-trip and the `filter_memo` HashMap per filtered
   search — fewer prepared-statement executions and allocations on the hydration path.
3. **Hydration shrinks to what's kept.** Text was hydrated for every survivor before ranking;
   now `Candidate` is provenance-only and `hydrate` touches *only the candidates the caller
   keeps*. A pull-1000-keep-10 rerank reads 10 `txt` blobs, not 1000.
4. **Laziness bounds work to demand.** No `Effort` over-fetch heuristic
   (`c·√(limit·N)`) that guesses a pool depth; the consumer pulls exactly to its rerank/scope
   need and the bucket walk + per-bucket SQL stop there.
5. **One fewer table + one fewer join.** Provenance is a single-table point lookup on `seg`
   (no `seg ⋈ doc`), and writes touch one fewer table (no `doc` insert/lookup/reap).
6. **Per-token allocation already eliminated stays so** (term-space resolution, `Borrow<str>`
   present-token pairing). Removing the class-`z` rarity computation drops a per-query
   `ln`/lookup per token from selection.
7. **Less per-search state.** No band-spread atomic increment per query; no `ClassSnap`
   snapshot allocation per search; `SearchOpts` shrinks from 9 fields to 4.

---

## 8. Open risks / unsure

- **2–3× is not reachable on the storage core alone.** Stated plainly in §6. If the team
  reads "2–3×" as a hard gate on *this* layer, that's a conflict — my honest max here is
  ~1.4×, ~1.8× with the tokenizer cuts. Over-cutting `postings`/`store`/`schema` to force 3×
  would dismantle the very invariants the brief says I must not break.
- **Dedup-by-`Key`.** Hashing/comparing `Key` (esp. `Text`/`Blob`) per survivor replaces a
  `u32` dedup. Fine for ≤ pulled-depth survivors at trifle's "small documents" scale, but a
  pathological pull-everything-with-long-text-keys consumer pays more than before. Mitigation:
  if it bites, keep a `doc`-id-free internal `seg.id`→key cache; I judge it unnecessary.
- **Cutting class-normalized rarity (welford).** Defensible (within-script identical; doc
  calls it empirical), but the mixed-script recall eval is *unrun*. If the benchmark later
  shows a real win, it re-enters cleanly — but now in the isolated `select`/dict layer, not
  smeared across three modules. I recommend cutting and re-validating, per
  `tmax-pool-sweep-methodology` (empirical, not assumed).
- **Stream holds a snapshot/tx open.** A caller that parks a `CandidateStream` across long
  pauses pins a WAL snapshot and blocks checkpoint truncation (WAL growth). Documented + the
  convenience paths drain immediately, but it's a new caller footgun the eager API lacked.
- **`Iterator<Item = Result<Candidate>>` ergonomics.** Idiomatic for fallible streams and
  composes (`filter_map(Result::ok)`, `collect::<Result<Vec<_>>>()`), but a caller who
  ignores the `Result` silently drops candidates on a transient `Busy`. A `try_fold`/
  `collect_matches` helper mitigates; worth bikeshedding.
- **Keeping `Shared` vs the brief's lean.** I argue *keep* it (enables the join-filter, ~66
  LOC). That's a deliberate reversal of the brief's cut candidate; if the adversarial round
  prefers the cut, the universal `key IN rarray(?)` mode keeps filtering whole on `Sidecar`.
- **Raw-SQL filter is untyped & version-coupled.** A malformed fragment is a runtime error,
  and it couples callers to trifle's column names (`key`, `seg`). That was already true of the
  old `Filter::Sql` escape hatch; now it's the *only* filter. Acceptable for a derived-cache
  power feature, but it lowers the guardrails for the median caller.
