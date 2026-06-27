# Proposal — `trifle` rev-0.3 (lens: raw-SQL filter + LOC accounting)

Design agent **Filter**. Complete proposal covering all three layers, with the deepest
work on the raw-SQL middle-tier filter and a ruthless, module-by-module LOC accounting.
All claims verified against `feat/rev-v0.2` source (8,052 src LOC; 3,045 root-test LOC).

---

## 1. Target crate / module layout (the 3 layers)

A **virtual workspace** with three members; the dependency arrow points only downward.

```
trifle-overlap/        # Layer 1 — pure engine.   deps: roaring                (no rusqlite, no std::collections beyond Vec)
  src/lib.rs           #   Scored, tier_weights, OverlapWalk (the BSI counter + bucket walk)

trifle/                # Layer 2 — storage + overlap, streaming output.   deps: trifle-overlap, rusqlite, roaring
  src/lib.rs           #   Index<T,B>, Config, SearchOpts, Stats, Writer/Reader leases, open/rebuild/compact/stats
  src/tokenize.rs      #   NgramTokenizer (+ normalization)            [bedrock]
  src/model.rs         #   Key/KeyShape, Affinity, Document, Match, Schema/SchemaBuilder, Filter
  src/search.rs        #   pipeline glue + snapshot/generation guard + the candidate stream
  src/filter.rs        #   NEW (small): Filter, filter_pass — the raw-SQL middle tier
  src/postings.rs      #   owned roaring index                          [bedrock]
  src/schema.rs        #   DDL / drift / shadow swap / id alloc         [bedrock]
  src/store/{mod,pool,sidecar}.rs   # Sidecar backend + pool            [bedrock, Shared dropped]
  src/dict.rs          #   faulting term dict (ClassStats removed)
  src/select.rs        #   rarest-by-raw-df selection (class-norm removed)
  src/{term,error,hash,instrument}.rs                                   [bedrock]

benchmarks/            # publish=false; can bench trifle-overlap in *complete isolation*
```

**Why this split.** The engine becomes a crate with a single non-`std`-collections
dependency (`roaring`), so it is fuzz/criterion-benchmarkable with synthetic bitmaps and
*cannot* accidentally grow an SQL or `String` dependency — the type system enforces
"zero SQL, zero provenance, zero hydration, zero `String`." Everything provenance-shaped
(keys, labels, text, filtering) lives in `trifle` and is composed *over* the engine's
candidate stream.

---

## 2. Layer 1 — the pure overlap engine (`trifle-overlap`)

Lifted verbatim from today's `rank.rs`: `add_weighted`, `weighted_overlap`, `tier_weights`,
`count_eq` (all already pure — they take `&[&RoaringBitmap]` and return bitmaps). The only
new code turns the eager `overlap_search` bucket loop into a **lazy iterator**.

```rust
use roaring::RoaringBitmap;

/// One scored candidate. Ids + counts only — no provenance, no text. `Copy`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Scored {
    pub id: u32,       // the posting id (trifle's segment id; opaque here)
    pub score: u32,    // IDF-weighted bit-sliced bucket value (the ordering key)
    pub overlap: u32,  // raw distinct-token overlap (the min_shared floor is on this)
}

/// Per-posting df-anchored tier weight {1,2,3,4} from posting cardinalities.
/// `N`-free: IDF *gaps* don't depend on corpus size. (unchanged logic)
pub fn tier_weights(cardinalities: &[u64], weight_step: f64) -> Vec<u32>;

/// The high→low weighted-overlap walk as a lazy iterator. Borrows the postings the
/// storage layer already loaded; allocates only the BSI planes (a handful of bitmaps)
/// once. Yields `Scored` descending by score, ascending by id within a bucket.
pub struct OverlapWalk<'a> { /* planes: Vec<RoaringBitmap>, present: &'a[&'a RoaringBitmap],
                               c: u32 (cursor), floor: u32, bucket: roaring::IntoIter, ... */ }

pub fn overlap_walk<'a>(
    postings: &'a [&'a RoaringBitmap],
    weights: &'a [u32],
    min_shared: u32,
) -> OverlapWalk<'a>;

impl Iterator for OverlapWalk<'_> {
    type Item = Scored;          // one id at a time, bucket by bucket; raw overlap < floor skipped
    fn next(&mut self) -> Option<Scored>;
}
```

**Zero-alloc / zero-SQL / flatness.** Building the planes is `Σ popcount(w_i)` ripple-adds
(≤ 2 per posting for weights 1..=4) of `O(containers)` bitmap ops — independent of posting
*size* (the flatness claim, preserved exactly). Iteration yields `Copy` `Scored` values: no
per-item `String`/`Vec`. No `Connection`, no provenance. The caller (`trifle`) decides when
to stop (`.take(limit)`) — early-stop is just dropping the iterator, so the planes past the
locked head are never walked.

---

## 3. Layer 2 — the public storage + overlap API (`trifle`)

The engine streams ids; `trifle` adds provenance, the raw-SQL filter, and text hydration —
each a *thin* stage the caller can also bypass.

```rust
/// A scored, provenance-hydrated candidate. Text is hydrated lazily (see `candidates`).
pub struct Candidate {
    pub key: Key,
    pub label: String,
    pub score: u32,
    pub overlap: u32,
    pub text: String,             // empty until hydrated
    pub span: Option<(usize, usize)>,
}

#[non_exhaustive]
pub struct SearchOpts<'a> {
    pub limit: usize,
    pub min_shared: Option<u32>,  // m
    pub t_max: Option<usize>,     // selection breadth
    pub weight_step: f64,         // D
    pub filter: Option<Filter<'a>>,   // the raw-SQL middle tier (§4)
}

impl<T: Tokenizer, B: Backend> Reader<'_, T, B> {
    /// Convenience: the top-`limit` matches in weighted-overlap order. = candidates+take+collect.
    pub fn search(&self, query: &str, opts: SearchOpts<'_>) -> Result<Vec<Match>>;
    pub fn search_batch(&self, queries: &[&str], opts: SearchOpts<'_>) -> Result<Vec<Vec<Match>>>;

    /// The streaming spine callers build on "cleanly and flexibly": a lazy iterator of
    /// scored, filtered, provenance-hydrated candidates in weighted-overlap order. Text is
    /// hydrated only for items the caller actually pulls (`.text()`), so a caller that ranks
    /// on score/overlap alone pays no text cost. The iterator holds the read snapshot open
    /// for its lifetime — drain or drop it promptly.
    pub fn candidates<'a>(&'a self, query: &'a str, opts: &'a SearchOpts<'a>)
        -> Result<CandidateStream<'a, T, B>>;   // Iterator<Item = Result<Candidate>>
}
```

**How a caller hydrates / ranks / filters on top.** Ranking, fusion (RRF), an exact
precision tier, length-normalized scoring — all become caller code over `candidates(...)`:
read `score`/`overlap`/`text`, reorder, `take(k)`. The `Ranker` trait, `Effort` over-fetch,
`Candidates`/`Candidate`/`QueryContext`/`Ranked` all disappear (a custom ranker was only
ever "consume candidates and reorder" — which is now literally what the iterator is for).
Filtering is the raw-SQL tier below.

---

## 4. The raw-SQL middle-tier filter (my primary lens)

### 4.1 What replaces what

Deleted wholesale: `Filter` enum (`Cmp`/`In`/`Between`/`IsNull`/`Like`/`Sql`/`And`/`Or`),
`CmpOp`, `Filter::compile`/`build`, `FilterType` (collapsed to `Affinity`), and the
`scope` closure predicate. One type remains:

```rust
/// A raw, parameterized SQL boolean expression over the declared attribute columns of the
/// `doc` row. The *only* filter mechanism — opt-in; absent ⇒ zero filter cost.
#[derive(Clone, Copy, Debug)]
pub struct Filter<'a> {
    where_sql: &'a str,
    params:    &'a [Value],
}

impl<'a> Filter<'a> {
    /// `where_sql` is spliced into `… WHERE id IN rarray(?{N+1}) AND (<where_sql>)`, where
    /// N = `params.len()`. Reference your params as `?1 .. ?N`.
    pub fn new(where_sql: &'a str, params: &'a [Value]) -> Self;
}
```

Use:

```rust
let f = Filter::new("deck = ?1 AND created >= ?2",
                    &[Value::Integer(7), Value::Integer(epoch)]);
reader.search("quikc brown", SearchOpts::new(10).filter(f))?;
```

### 4.2 What the schema must still declare

**Verdict: filterable columns survive as caller-declared columns** — slimmed to a name +
affinity, no grammar. The caller declares them so the DDL creates and indexes them and the
fragment can reference them:

```rust
Schema::builder()
    .key("note_id", KeyShape::Integer)
    .text("front")
    .attr("deck",    Affinity::Integer)   // was .filterable(name, FilterType)
    .attr("created", Affinity::Integer)
    .build()?

pub enum Affinity { Integer, Real, Text, Blob }   // FilterType minus the `Timestamp` sugar
```

`Affinity` materializes each attribute as a real, single-column-indexed `doc` column (the
existing `doc_filt_cols`/`doc_filt_indexes` DDL machinery in `schema.rs`, unchanged). The
`Timestamp` variant is dropped: it was *literally* an INTEGER column plus a doc note; callers
store epoch-ints (numeric order) or ISO-8601 text (lexicographic order) and say so in their
own fragment. Affinity is load-bearing for *range* filters (`created >= ?` must compare
numerically), which is the one thing the bare column-name approach can't guarantee.

Attributes are written on `Document { key, segments, attrs }` (rename of `payload`) and via
one `set_attrs(key, &[(&str, Value)])`.

### 4.3 The binding contract and the F2 placeholder fix

Today (`rank.rs::filter_pass`) the candidate scope binds as **`?1`** and the caller's
fragment is forced to use *anonymous* `?` (which SQLite renumbers from `?2`); a numbered
`?1` collides and fails with a parameter-count error (audit F2). This is a sharp,
type-invisible footgun.

**Fix: bind the candidate id set as the *last* positional parameter, computed from the
caller's count.** The caller numbers naturally from `?1`:

```rust
fn filter_pass(conn: &Connection, ns: &Namespace, f: &Filter<'_>, cand: &[u32])
    -> Result<RoaringBitmap>
{
    let arr: Rc<Vec<Value>> = Rc::new(cand.iter().map(|&i| Value::Integer(i as i64)).collect());
    let ids_pos = f.params.len() + 1;                       // bind rarray one past the caller's
    let sql = format!(
        "SELECT id FROM {doc} WHERE id IN rarray(?{ids_pos}) AND ({frag})",
        doc = ns.doc(), frag = f.where_sql,
    );
    let mut stmt = conn.prepare_cached(&sql)?;
    let mut binds: Vec<&dyn rusqlite::ToSql> = f.params.iter().map(|v| v as _).collect();
    binds.push(&arr);                                       // position N+1
    // … query, collect ids …
}
```

No named/positional mixing, no SQLite renumber surprise, the caller writes exactly the SQL
they'd write standalone. (`?N+1` is one past the caller's range; the only way to break it is
to write `?{N+1}` yourself — documented and absurd.)

### 4.4 Injection / trust contract

One path, one rule — *simpler and more honest* than today's two-tier "validated grammar (safe)
+ `Sql` escape hatch (unsafe)" split that lulled callers into trusting the grammar:

- `where_sql` is **trusted, not sandboxed** — spliced verbatim into a `SELECT … FROM doc`.
  It is the injection surface: it must be a developer-authored constant/template, never built
  from untrusted input. Data flows through `params` (bound), never formatted in.
- A subquery in the fragment *can* reference other tables (not sandboxed) — same caveat as
  today's `Filter::Sql`, now the single documented contract.
- Attribute **column names** are validated identifiers at schema build (`validate_ident`), so
  the columns that exist are injection-safe. An undeclared column referenced in the fragment is
  a **runtime SQL error** (the caller's own trusted SQL), not a silent injection — acceptable
  and documented. (We drop today's per-field `filter_column` validation; it only ever guarded
  the grammar, which is gone.)

### 4.5 Inside the bucket walk vs. outside — and the perf consequence

**Decision: the filter stays *inside* the candidate walk (Layer 2 wiring), the pure engine
stays filter-free.** As the `CandidateStream` pulls each engine bucket, it hydrates that
bucket's provenance and runs `filter_pass` scoped to the bucket's ids (`WHERE id IN
rarray(...) AND (frag)`, memoized per doc — exactly today's structure), keeping only passing
docs and continuing the walk until `limit` passing docs lock. The engine never sees the
filter; `candidates()` takes it as a parameter.

Concrete perf argument for a **selective filter over a large corpus** (N≈1M, limit=10,
selectivity s):

| | filter **inside** (this design) | filter **outside** (caller filters the key-stream) |
|---|---|---|
| SQL round-trips | a few small indexed `IN rarray` lookups over the high-score head | ~`10/s` candidate pulls; per-candidate attribute lookup against caller tables |
| early-stop | preserved — walk stops at 10 *passing* docs | **lost** — trifle can't early-stop on a predicate it doesn't know |
| text hydration | only the ≤ `limit` survivors | over-hydrate filtered-out docs, or a 2nd round-trip |
| O(N) scan | never (bounded by pool depth) | never, but far more work pulled |

Pushing the filter outside loses early-stop and forces over-hydration precisely when the
filter is selective — the case that most needs pruning. The pure engine's `Scored` ids are
*internal* and carry no attributes, so an outside filter must first hydrate (key →
attributes) anyway. Keeping it inside is strictly better and still *opt-in*. Advanced callers
who *want* to filter outside still can: consume `candidates(...)` with no `filter` and filter
the hydrated stream themselves — the streaming API makes both possible.

Caveat (unchanged, stated honestly): the in-walk filter prunes before hydrate/rank but does
**not** save the candidate-generation overlap work — the engine still scans the selected
postings (that would need a partitioned index).

### 4.6 The no-ghost invariant under attributes (honest note)

The brief says "if payload is removed, this simplifies — say so explicitly." I *keep*
attributes (the mandated filter needs columns in trifle's db), so the no-ghost rule does
**not** vanish — but it collapses from today's sprawling C2-WP1/C2-RA-1/F1/T3/C3-WP1 matrix
to **two one-line guards**:

1. `set_attrs(key, …)` requires an existing (segment-bearing) doc, else `InvalidInput` (the
   surviving T3 guard).
2. A `Document` with `attrs` but no `segments` is rejected on insert/rebuild (no payload-only
   ghost row).

Because attributes live only on the `doc` row, and a `doc` row exists iff the doc has ≥1
segment (the `remove_one_segment`/`remove` reaping already enforced), there is no
payload-only write path to create a ghost. The huge test surface (cycle2.rs, 648 LOC, almost
entirely payload/ghost permutations) shrinks to ~3 focused tests.

---

## 5. Write API (filter-coupled; cross-cutting)

Collapse the 9 write methods to **4**, folding attributes in:

```rust
w.insert(key, &[(label, text)])   -> Result<()>   // create-or-error; replaces the key's segment set
w.upsert(key, &[(label, text)])   -> Result<()>   // create-or-replace; idempotent
w.set_attrs(key, &[(&str, Value)])-> Result<()>   // attrs on an existing doc (T3 guard)
w.remove(key)                     -> Result<()>    // drop doc + segments + attrs
```

Gone: `insert_segment`/`upsert_segment`/`remove_segment` (per-segment sugar a caller
expresses by passing the full set), `insert_document`/`upsert_document` (attrs now ride
`insert`/`upsert` via the `Document` form `rebuild` already consumes), `set_fields`
(renamed `set_attrs`). `Document.attrs` carries attributes on the bulk/rebuild path.

---

## 6. Deletion list with LOC accounting

Measured against the current per-module counts (`wc -l src`). Each row: estimated LOC
removed (including inline `#[cfg(test)]`), the risk, and the invariant touched. **L** = my
filter lens; **X** = cross-cutting (from the brief's cut-candidate list, included so the
total is real).

| # | Cut | Module(s) | ~LOC | Lens | Risk | Invariant |
|---|-----|-----------|-----:|:---:|------|-----------|
| 1 | `Filter` enum + `CmpOp` + `compile`/`build` + grammar test | model.rs | 290 | L | low — replaced by `Filter::new` | none (batch==serial preserved; filter still per-query) |
| 2 | `FilterType` → `Affinity` (drop `Timestamp`) | model.rs | 10 | L | low — epoch-int/ISO documented | schema fingerprint (encode `Affinity`) |
| 3 | `set_doc_fields`/`set_fields`/`insert_document`/`upsert_document`/`write_document` → `set_attrs` | lib.rs | 150 | L | med — API churn | no-ghost (slims to 2 guards, §4.6) |
| 4 | payload plumbing in `rebuild` (col-list build, bind loop, payload-only reject) | lib.rs | 40 | L | low | no-ghost via rebuild (C3-WP1) |
| 5 | `ScopeFn` + `scope` field/setter + `scope` in walk | lib.rs, rank.rs | 35 | L | low — folds into `Filter`/caller | none |
| 6 | `CompiledFilter`/`filter_pass`/`filter_memo`/filter-in-`OverlapParams` → slim `filter.rs` | rank.rs | 55 (net) | L | low | filter-scoped-to-candidates (T5/I12) kept |
| 7 | filter compile + wiring | search.rs | 12 | L | low | batch==serial (compile→borrow `Filter`) |
| 8 | filter/payload/ghost tests (cycle2 ~560, lifecycle ~40, api scope ~60, scope_ranker scope ~55) | tests | 715 | L | low | the cut tests' invariants (now 2 guards) |
| 9 | `welford.rs` whole + `ClassStats`/`ClassSnap` threading | welford.rs, dict.rs, search.rs | 320 | X | **med** — reverts to rarest-by-raw-df (within-script common case; mixed-script recall unproven) | selection (still per-query ⇒ batch==serial) |
| 10 | class-normalization in `select` + its tests | select.rs | 160 | X | med — see #9 | batch==serial preserved |
| 11 | band-spread hist + `WeightStepHint` + `HINT_*` + `observe`/`reset`/`weight_step_hint` | lib.rs | 200 | X | low — advisory telemetry | none |
| 12 | `Effort` enum + impl + tests | lib.rs | 115 | X | low — only fed a custom ranker | none |
| 13 | `Ranker` trait + `Candidates`/`Candidate`/`QueryContext`/`Ranked` + `OverlapRanker` (→ caller ranks over the stream; small `Candidate` struct re-added) | rank.rs, search.rs | 190 (net) | X | med — callers re-implement reorder | none |
| 14 | per-segment write methods (`insert_segment`/`upsert_segment`/`remove_segment`) | lib.rs | 120 | X | low — sugar | none |
| 15 | `SearchSession` (warm lease; warming layers unbuilt) | lib.rs | 60 | X | low | none |
| 16 | `Shared` backend + its tests/refs | store/shared.rs, tests | 86 | X | med — co-location callers lose it | namespace isolation unaffected |
| 17 | engine extraction housekeeping (move pure fns to `trifle-overlap`; net wash in workspace, but removes rank↔store coupling) | rank.rs | 0 | X | low | flatness preserved (code unchanged) |

**Totals.** Src removed ≈ **1,750**; tests removed ≈ **900** (cycle2 dominates). Of these,
my filter lens accounts for **~590 src + ~640 test ≈ 1,230 LOC**.

### 6.1 The 2–3× claim — the honest number

Projected **public-crate** end state (per module, attrs kept per the mandate):

| module | now | → | note |
|---|---:|---:|---|
| lib.rs | 1869 | ~720 | Effort/band-spread/payload-methods/seg-methods/SearchSession/scope/ranker out |
| tokenize.rs | 1171 | ~1171 | bedrock (trim is a separate lever — see below) |
| model.rs | 789 | ~430 | Filter grammar out |
| rank.rs | 720 | ~180 | engine → `trifle-overlap`; only filter_pass/hydrate/Candidate stay |
| postings.rs | 623 | ~620 | bedrock |
| schema.rs | 559 | ~545 | attrs DDL kept |
| store/* | 553 | ~470 | Shared out |
| search.rs | 338 | ~250 | class_snap/filter-compile out |
| dict.rs | 291 | ~210 | ClassStats out |
| select.rs | 280 | ~120 | class-norm out |
| welford.rs | 201 | 0 | deleted |
| term/error/hash/instrument | 480 | ~470 | bedrock |
| **public trifle src** | **8052** | **≈ 5,186** | **1.55×** |
| + `trifle-overlap` (new) | — | ~300 | isolated engine |

**Frank verdict:** with the mandated raw-SQL filter (which *requires* keeping the attribute
columns, costing ~250 LOC vs dropping them) and the storage core protected by the invariants,
the achievable reduction is **~1.55× on public-crate src** and **~1.85× counting the test
collapse** (root tests 3,045 → ~2,150). The headline **2–3× is not reachable without cutting
into bedrock** the brief protects (`postings`/`schema`/`store`/`tokenize`). Two levers push
toward 2×, each with a real cost, and I recommend exactly one:

- **Tokenizer trim** (Trigram-only, NFC + optional casefold, drop Bigram/Quadgram aliases +
  NFD + accent-strip): tokenize.rs ~1171 → ~650. Gets the crate to **~4,660 (1.73×)**.
  *Cost:* loses accent-fold recall and the alternate n-gram sizes — a real feature/recall
  regression; **recommend only if benchmarks show the dropped normalizers don't move recall.**
- **Drop trifle-side attributes entirely (Option B)**: removes #2–#4 retained machinery +
  the attrs DDL (another ~250), reaching **~1.9×** — but it *deletes the mandated filter*
  (no columns ⇒ no `WHERE id IN rarray AND frag`; the caller must filter against their own
  store, losing prune-before-hydrate and early-stop, §4.5). **Not recommended** — it trades
  the brief's headline feature for ~5% more shrink.

So: **recommended end state ≈ 1.55–1.7× src (4,700–5,200 LOC) + ~1.85× with tests**, landing
the low end of "2×" only with a benchmark-gated tokenizer trim. I would rather report this
than fabricate a 2–3× the invariants forbid.

---

## 7. Performance argument (faster, not just smaller)

1. **Fewer allocations on the hot path.** The engine yields `Copy` `Scored` (no per-candidate
   `String`/`Vec`); `Survivor`/`Candidate`/`Ranked`/`QueryContext` allocation churn is gone.
   Text is hydrated *lazily* (only items the caller pulls), so a score-only consumer pays no
   text-hydration `String` cost at all — today every survivor is text-hydrated unconditionally
   before the ranker.
2. **Fewer SQL round-trips for selective filters** (§4.5): early-stop preserved + filter
   scoped to candidate ids + single batched provenance/text reads — the in-walk filter beats
   any outside-the-stream approach on the selective-over-large-corpus case.
3. **The F2 fix** removes a class of caller-side query rewrites and the failure mode they
   caused.
4. **Isolation enables optimization.** `trifle-overlap` with only `roaring` is independently
   criterion-benchmarkable on synthetic postings (the brief's "crown jewel, optimizable in
   isolation") — SIMD popcount, plane-layout, container-iteration tuning can be measured
   without SQLite noise. Flatness (`O(k·log k)`, posting-size-independent) is preserved
   verbatim.
5. **Less branch/indirection in selection** (raw-df rarity drops the per-token `ClassSnap`
   lookup + `partial_cmp` on f64 z-scores → an integer-df sort) and **no per-query
   class-snapshot** read-lock traffic in `dict`.

---

## 8. Open risks / things I'm unsure about (for the adversarial round)

1. **The candidate iterator holds a read transaction open for its lifetime.** It pins a WAL
   snapshot (the generation guard needs one consistent snapshot, `search.rs::search_read_on`).
   A caller that holds a lazy `CandidateStream` while doing slow per-item work keeps the
   snapshot alive (delays WAL checkpoint). `search()`/`search_batch()` drain immediately and
   are unaffected; the streaming API needs a clear "drain promptly" contract. Self-referential
   lifetime (stream owns the pooled conn + the `unchecked_transaction`) is awkward in Rust —
   may need a small owning wrapper or `ouroboros`-style care, or yielding *batches* instead of
   single items.
2. **Removing class-normalized rarity (#9/#10)** is a recall bet. The doc itself calls its
   value "an empirical question." Mixed-script corpora (CJK bigram vs Latin trigram in one
   query) are where it could matter — must be benchmark-gated before deletion, not assumed.
3. **Affinity vs untyped columns.** I keep a 4-variant `Affinity` for range-filter
   correctness; an alternative is fully untyped `doc` columns (even simpler) at the cost of
   numeric-range surprises. I judged the affinity worth ~30 LOC; reviewers may disagree.
4. **The raw fragment is unsandboxed.** Same trust contract as today's `Filter::Sql`, but now
   it's the *only* path — a careless caller who formats values into the fragment has an
   injection. Mitigation is documentation + the `params` ergonomics; there is no
   compile-time guard. Acceptable for an embedded library, but it is a real sharp edge.
5. **The 2–3× target is not met** (§6.1) without a tokenizer trim that risks recall. If the
   target is hard, the conflict is between it and (a) the mandated filter keeping attrs and
   (b) the protected storage core. Flagging it loudly so the lead can re-scope rather than
   discover it at implementation time.
6. **Dropping the `Ranker` trait** pushes reorder/precision-tier logic onto every caller.
   Good for the "thin things composed over the stream" mandate, but a downstream that had a
   custom ranker now writes more code. The streaming API makes it *possible* and arguably
   cleaner; whether it's *ergonomic* enough is worth a reviewer's eye.
