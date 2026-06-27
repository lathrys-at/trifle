# Design brief — radical simplification of `trifle` (rev v0.2 → v0.3)

You are one of three design agents proposing how to make `trifle` **radically simpler**
(target: **2–3× less code**) and **maximally fast/efficient**, starting from the current
state of the `feat/rev-v0.2` branch (the user calls it "rev-2.0"; same branch). Read the
real code — this brief orients you, but verify every claim against the source.

## The mandate (non-negotiable shape of the target design)

The user wants the crate refactored into **three clean layers**:

1. **A pure overlap engine** — an internal crate (`trifle-overlap` or similar) *or* a
   strictly-isolated module that is **purely** the IDF-weighted bit-sliced overlap engine.
   It takes postings (roaring bitmaps) + per-token rarity and **streams** scored candidates
   `(id, weighted_score, raw_overlap)` as an iterator. **Zero SQL, zero provenance, zero
   hydration, zero `String`.** It must be benchmarkable and optimizable in complete isolation.
   This is the crown jewel (today in `src/rank.rs`: `add_weighted`, `weighted_overlap`,
   `tier_weights`, `count_eq`, the bucket walk). The "flatness" property — candidate
   generation is `O(k·log k)` bitmap ops, **independent of posting size** — must be preserved.

2. **The public crate = storage engine + overlap, with streaming/iterator output.** The
   public crate is *just* (a) the SQLite-backed storage engine (owned roaring inverted index:
   postings + dict + schema + store/backends) and (b) the overlap engine wired on top of it,
   exposing a **streaming/iterator of IDF-weighted overlap candidates** that callers build on
   top of "cleanly and flexibly." Hydration, ranking, fusion, relevance — all become thin
   things a caller composes over the candidate stream, **not** baked-in machinery.

3. **An opt-in raw-SQL middle-tier filter.** Replace the entire structured-filter
   "hodgepodge" — `Filter`/`FilterType`/`CmpOp` (Cmp/In/Between/IsNull/Like/Sql/And/Or in
   `src/model.rs`), the `scope` closure predicate, and the `filterable` schema columns — with
   **one opt-in raw parameterized SQL filter** applied to the candidate id set
   (`WHERE id IN rarray(?) AND (<caller fragment>)`). No more semi-approximations of SQL; just
   the raw query, opt-in. Decide what (if anything) the caller must declare for it to work.

The user's words: *"an internal crate or module that is purely the overlap engine and the
public crate itself to be just the storage engine and overlap with streaming/iterator output
of IDF-weighted overlap candidates that can be built on top of cleanly and flexibly (with the
opt-in SQL query middle-tier filter available, no more hodgepodge of semi-approximation of SQL
filters, just the raw query opt-in)."*

## What `trifle` is (so you cut the right things)

Embedded, typo-/partial-tolerant **trigram fuzzy search** over SQLite, tuned for **large
corpora of small documents** (≲1–2 KB/segment), **read-often / write-infrequent**. It is a
**derived, rebuildable cache** over a caller-owned source of truth — never migrates, drops the
cache on drift. It is *lexical* fuzzy overlap, **not** semantic, **not** a relevance/BM25
engine. It ranks by IDF-weighted trigram overlap computed in the bit-sliced counter itself
(per-query df-anchored 4-tier weights `{1,2,3,4}`).

## Current architecture (8,052 LOC src; verify against code)

| module | LOC | responsibility | simplification pressure |
|---|---|---|---|
| `src/lib.rs` | 1869 | `Index<T,B>`, `Config`, `SearchOpts`, `Effort`, `Stats`, `WeightStepHint`, `CompactStats`, leases (`Writer`/`Reader`/`SearchSession`), write internals, `rebuild`, `compact`, `stats`, band-spread telemetry | huge; many knobs/telemetry are cut candidates |
| `src/tokenize.rs` | 1171 | `NgramTokenizer` (Trigram/Bigram/Quadgram), NFC/NFD/accent-strip/casefold normalization, inline `Ngram` tokens, span location | core; large but mostly load-bearing |
| `src/model.rs` | 789 | `Key`,`KeyShape`,`FilterType`,`CmpOp`,`Filter`,`Document`,`Match`,`Schema`,`SchemaBuilder` | the **Filter grammar** is the prime cut target |
| `src/rank.rs` | 720 | the overlap engine + `Ranker` trait + `Candidates`/`Candidate`/`QueryContext`/`Ranked` + `CompiledFilter`/`filter_pass` + provenance hydration | **split**: pure engine → inner crate; trait machinery likely cut |
| `src/postings.rs` | 623 | owned roaring index (base ∪ added \ removed), (de)serialize, `apply_writes`, `fold`, `read_dfs`, `effective_postings`, `write_base_postings` | core storage |
| `src/schema.rs` | 559 | DDL from validated `Namespace`, version stamps, drift reset, shadow swap, id allocation, seg-stat meta counters | core; shrinks if filterable cols go |
| `src/store/{mod,pool,sidecar,shared}.rs` | 731 | `Backend` trait, `Namespace`, `validate_ident`, Sidecar + Shared backends, writer mutex + read pool | `Shared` is a cut candidate |
| `src/search.rs` | 338 | pipeline glue: `query_terms`, `SearchCtx`, `run_search`, `hydrate_text`, `search_read_on` (snapshot/generation guard) | restructures around the stream |
| `src/dict.rs` | 291 | in-memory faulting term dict (gram u128→u32), `InternStage`, `resolve_terms`, generation | core; ClassStats threading is a cut candidate |
| `src/select.rs` | 280 | rarest-first selection (class-normalized) | class-normalization is a cut candidate |
| `src/welford.rs` | 201 | per-script-class DF stats for class-normalized rarity | **prime cut**: doc admits its value is "an empirical question" |
| `src/term.rs` | 196 | `Term`/`IntoTerm` (packed u128 gram) | core |
| `src/error.rs`,`hash.rs`,`instrument.rs` | 284 | errors, FxHash, tracing macros | keep |

### Known cut candidates (quantify the savings; argue each)
- `welford.rs` + `ClassStats`/`ClassSnap` threading through dict/select/search — class-normalized
  rarity; unproven. Cutting reverts selection to rarest-by-raw-DF (the within-script common case).
- Band-spread histogram + `WeightStepHint` + `observe_band_spread`/`weight_step_hint`/
  `reset_band_spread_hist` + `band_spread_hist` field (~150 LOC in `lib.rs`) — advisory telemetry.
- `Effort` over-fetch enum (~100 LOC + tests) — only exists to over-fetch for a custom `Ranker`.
- `Ranker` trait + `Candidates`/`Candidate`/`QueryContext`/`Ranked` (~250 LOC) — if output is a
  candidate stream, callers rank above trifle.
- `Filter` grammar + `compile` + `CompiledFilter`/`filter_pass`/`filter_memo` + `filterable`
  columns (~400+ LOC across model/rank/schema) — replaced by raw-SQL opt-in.
- `scope` closure predicate — fold into candidate filtering or drop.
- `Shared` backend — "use only for a hard co-location requirement."
- The 8 write methods (`insert`/`insert_segment`/`upsert`/`upsert_segment`/`insert_document`/
  `upsert_document`/`set_fields`/`remove`/`remove_segment`) + payload/two-level doc→segment model
  — collapses substantially if filterable payload is removed.
- `SearchSession` vs `Reader` — two read leases; the warm-cache layers are unbuilt "follow-ups."

## Invariants you must NOT break (cite these when you critique)

1. **`batch == serial`** — every per-query input (selection, df's, weights, filter) derives only
   from that query's own tokens + the shared snapshot, never a batch aggregate. Tested in
   `tests/scope_ranker.rs`.
2. **No-ghost doc rows** — a payload/segment-less document must never materialize a doc row a
   later insert under the same key would inherit (audits F1/T3/C2-WP1/C2-RA-1). If payload is
   removed, this simplifies — say so explicitly.
3. **Dict-generation guard** — a read racing a concurrent id-reassigning `rebuild` must observe
   old-or-new atomically; a skew surfaces as retryable `Error::Busy` (`search.rs:search_read_on`).
4. **Monotonic id allocation + atomic shadow swap** for `rebuild` (`schema.rs`).
5. **Derived-cache / drift-reset** — schema/tokenizer/data_version drift drops the cache, never
   migrates.
6. **No sleeps in library code** — never block the caller to retry; surface `Error::Busy`,
   `busy_timeout=0`. The caller owns backoff.
7. **The flatness claim** — candidate generation cost is independent of posting size.
8. **Single tokenizer** runs on indexed text, postings, and queries → "index agrees with query"
   by construction.

## Constraints
- MSRV **1.85**, edition **2024**. Cargo workspace (root `trifle` + `benchmarks`).
- Lint bar is load-bearing: `-D warnings` everywhere incl. rustdoc; `clippy::all`,
  `unsafe_op_in_unsafe_fn` forbidden. Broken intra-doc links fail CI.
- Only feature is `tracing` (off by default).
- Single dep: `rusqlite` (bundled SQLite + `array` for `rarray`).
- Use `FxHash`, never SipHash, on hot paths (existing `src/hash.rs`).
- This is a **clean redesign proposal**, not a migration plan — `trifle` is a derived cache,
  versions drop rather than migrate, so backward source-compat is NOT required. Breaking the
  public API freely is allowed and expected. Optimize for the end state.

## Your deliverable

Write your full proposal to the file path given in your task prompt. It must contain:
- **Target crate/module layout** (the 3 layers): names, dependency direction, what each owns.
- **The pure overlap engine's API** — concrete Rust signatures for the streaming/iterator
  candidate output; the data types crossing the boundary; why it's zero-alloc/zero-SQL.
- **The public storage+overlap API** — concrete `Index`/reader/writer/search signatures; the
  candidate-stream type; how a caller hydrates/ranks/filters on top.
- **The raw-SQL middle-tier filter design** — exact API, how it binds to candidate ids, the
  injection/safety story, what the schema must still declare (if anything).
- **The deletion list with LOC accounting** — what's cut, est. LOC removed, the risk of each cut,
  and which invariant (if any) it touches. Land the 2–3× claim with numbers.
- **The performance argument** — concretely why this is faster (fewer allocations, fewer SQL
  round-trips, isolation enabling optimization, etc.), not just smaller.
- **Open risks / things you're unsure about** — be honest; these feed the adversarial round.

Be concrete and opinionated. Full autonomy — do not ask questions. Produce a complete proposal.
