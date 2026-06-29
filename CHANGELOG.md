# Changelog

## 0.3.0 — `rev-v0.3` (lean)

The lean revision, shipped on top of the (unreleased) 0.2.0 draft below. It strips the parts of
0.2.0 whose complexity didn't earn its keep and commits to a single spine: **IDF-weighted overlap
*is* the ranking, and provenance streams best-first before any text is touched.** Another **hard
cache reset** — the on-disk `SCHEMA_VERSION` bumps to 4, so an existing index drops its cache on
open and must be `rebuild()`-ed (data-loss-free for the *cache*; the caller's source of truth is
untouched).

### Bitmaps — CRoaring everywhere
- The `roaring` crate is dropped for **`croaring`** (the CRoaring SIMD bindings), used for **both**
  the storage posting layer and the overlap engine. Blobs are the standard CRoaring portable
  format — byte-identical, so there is no migration beyond the schema-version reset.

### Storage — flattened, no `doc` table
- The two-level `doc`+`seg` store collapses to **one `seg` table** carrying the key directly:
  `(id, key, label, text, len)`, with `seg.id` the posting id. A key with no segments can no longer
  materialize a ghost row, and provenance is a single-table point read.

### Ranking — the weighted-overlap order is final (no rerank tier)
- The pluggable `Ranker`, the over-fetch pool, and the `Effort` knob are **removed** (`src/rank.rs`
  is gone). The IDF-weighted, class-normalized overlap order computed in the bit-sliced counter is
  the result; a caller wanting a domain reorder composes it over the candidate stream rather than an
  in-engine tier. `weight_step` (`D`) stays as the rarity-spacing knob.
- **Kept from the 0.2.0 draft** (proposed for deletion, deliberately retained): the per-script-class
  Welford rarity (multi-script awareness) and the band-spread `WeightStepHint`
  (`Stats.weight_step_hint`).

### Read API — a provenance-first candidate stream
- Alongside `Reader::matches`/`matches_batch` (eager, top-`limit` hydrated), **`Reader::candidates`
  → `CandidateStream`**: a best-first cursor of provenance-only `Candidate`s that fuses on the first
  error. Compose rerank / pagination / fusion on top, then `hydrate` only what you keep — text and
  span are read in one batched pass for exactly those candidates.
- **`df_budget`** — a new `SearchOpts` knob capping selection by a `Σdf` work budget: an adaptive,
  tail-bounding alternative to the fixed `t_max` count.

### Filtering — raw SQL over the caller's live data
- The structured `Filter` grammar (`Cmp`/`In`/`Between`/`Like`/…), `Schema::filterable` columns,
  `FilterType`, and the stored filter/payload columns are **removed** — trifle-stored filter columns
  go stale against a derived cache. In their place, **`SqlFilter { fragment, params }`**: an opt-in
  raw parameterized predicate folded into the per-chunk provenance query against the caller's *live*
  data (universal mode `key IN rarray(?)`; a co-located join via `ATTACH`).

### Write API — segments only
- The writer is three methods: `upsert(key, &[(label, text)])`, `remove(key)`, and
  `remove_segment(key, label)`. The payload-carrying `insert` / `*_document` variants are gone with
  the payload columns.

### Type surface
- `Index<T: Tokenizer = DefaultTokenizer>` is generic over the tokenizer only, over the concrete
  `Sidecar` store; the `Backend` trait and `Shared` wrapper are removed.

### Tooling
- `benchmarks/` is reworked against the streaming API (commands `latency`/`relevance`/`fuzzy`/
  `selsweep`/`dsweep`/`overlap`/`ingest`/`profile`), excluded from the workspace, with a `selsweep`
  selection-cost-frontier plotter at `benchmarks/scripts/plot_selsweep.py`.

## 0.2.0 — `rev-v0.2` (unreleased)

A ground-up rework. **Breaking across the board, and a hard cache reset:** the on-disk
`SCHEMA_VERSION` bumps to 3, so an existing index drops its cache on open and must be
`rebuild()`-ed. trifle is a derived, rebuildable cache, so this is data-loss-free for the
*cache* (the caller's source of truth is untouched).

### Term model & storage
- **Term interning.** Postings (`post`/`delta`/`term`-df) and the forward index are now
  keyed by an interned `u32` term-id behind a faulting dictionary (reader `resolve`,
  writer `intern`), not gram text — narrower B-trees, smaller index.
- **Forward index as term-ids.** Every segment stores its `u32` term-id set, so **delete
  needs neither the text nor the tokenizer**.
- **`u128` term encoding.** A gram (≤3 codepoints) + script tag packs big-endian into a
  `u128`; the script byte is the most-significant byte (script-contiguous order).
- **Script-segmenting default tokenizer.** `DefaultTokenizer` is now what trifle ships:
  it splits text into maximal same-script runs and tokenizes each with a
  script-appropriate window (CJK bigrams, else trigrams), emitting no cross-script grams.
  `NgramTokenizer<N>` (aliases `TrigramTokenizer = NgramTokenizer<3>` /
  `BigramTokenizer = NgramTokenizer<2>`) is the plain fixed-width tokenizer for
  single-script corpora. Both yield the inline `Ngram<N>` token (`Gram = Ngram<3>`), and
  tokenization is **online** — `tokenize()` streams windows off the Unicode-normalization
  adaptors with no intermediate `Vec`. The old `Ngram<N, CAP>` value type is gone.
- **`IntoTerm`.** Blanket-implemented for every `Borrow<str>` and required of a
  tokenizer's `Token`, so a token packs into its interned `Term` directly; the write path
  interns via `token.term()`, with no per-token `String` allocation.
- **Class-normalized pruning.** The rarest-first pruner ranks by a per-script-class
  Welford z-score (log space), falling back to raw DF for sparse classes.

### Data model & API (breaking)
- **Runtime `Schema`** replaces the fixed `(doc_id, source, ref, text)` model: a declared
  **key** (shape `Integer`/`Text`/`Blob`) plus named **text fields**. Documents are
  `key → named segments` over a two-level `doc`+`seg` store.
- **All indexed text is stored and surfaced.** Per-field storage modes, the `TextResolver`
  / contentless mode, and the search-time `Hydration` ladder were **removed**: every
  indexed text field is stored, always returned on a match, and always available to a custom
  ranker (so one can never run textless). Filterable **payload** columns are stored separately
  for filtering only — never ranked or returned. `Match.text` is a `String` (no longer
  `Option`).
- **`Segment`/`Match{doc_id,source,ref_}` → `Document` / `Match{key,label,span,text}`**;
  `Key` is `Integer`/`Text`/`Blob`. The scope predicate is `(&Key, &str label) -> bool`.
- **Lease-based access.** `Index` shrinks to lifecycle (`open`/`rebuild`/`compact`/
  `stats`/`reader`/`writer`/`session`). Writes go through a `Writer` lease (the exclusive
  single-writer lock; `insert`/`upsert`/`remove` × whole/segment, plus
  `insert_document`/`upsert_document` for a whole `Document` with payload; `commit()` is
  commit-and-continue; drop rolls back). **Each write method is atomic** — its body runs in
  a `SAVEPOINT`, so a mid-call error (and a caught-error-then-`commit()`) leaves the store
  exactly as before the call. The labels in one call must be distinct (debug-asserted).
  Reads go through a `Reader`; **each search runs under a consistent snapshot, and a single
  `search_batch` shares one** (a fresh reader observes newer writes). `SearchSession` holds
  a warm connection for as-you-type.
- **`Index::open` now takes a `Schema`**: `Index::open_at(path, schema, config)` /
  `Index::open(backend, tokenizer, schema, config)`.
- **Drift** now also covers a **schema fingerprint** (alongside schema version, tokenizer
  fingerprint, and `data_version`).

### Ranking — IDF-weighted lexical overlap (no BM25, no relevance tier)
- trifle is a **fuzzy lexical overlap engine, not a relevance engine**. Ranking is
  **IDF-weighted token overlap computed in the bit-sliced counter itself** — there is no
  separate BM25/relevance rerank tier. Each selected gram is weighted by rarity with a
  per-query, df-anchored **4-tier scheme** (weights `{1,2,3,4}`): the query's commonest
  survivor gets weight 1, rarer grams more, spaced in df-doublings
  `1 + min(3, round(log2(df_max/df_i) / D))`. This is `N`-free (IDF *gaps* don't depend on
  corpus size), stores nothing, and reuses the survivor df's already fetched for pruning;
  weighted accumulation is BSI arithmetic (popcount(w) ≤ 2 ripples/gram). The knob `D`
  (df-doublings per weight step, default `1.0`) is on `SearchOpts::weight_step`.
- The earlier BM25+ reranker (idf + length normalization + `δ`) and the word-tokenized
  substring/literal tier before it are **both gone**. The default `Ranker` is `OverlapRanker`,
  which preserves the weighted-overlap order; `Effort` now only sets the over-fetch pool depth
  for a *custom* `Ranker` (default `Effort::None` — the weighted order is exact at `pool = limit`).
- The pluggable `Ranker` stays as an extension point: a custom ranker can score over each
  candidate's segment text + signals (`Candidate::overlap`/`score`/`matched_terms` (per-term
  `df`)/`seg_len`, `QueryContext::n_segments`/`avgdl`). The segment is the ranking unit, so
  **cross-segment fusion happens above trifle** (aggregate across your keys), not in a `Ranker`.
  (`seg.len`/`avgdl` are still stored — as signals for such a ranker — but the default ranking
  does not use them.)

### Filtering ladder
- **Tier 2 — filterable columns.** `Schema::filterable(name, FilterType)` materializes
  indexed `doc` columns; a structured `Filter` compiles to a validated, parameterized
  `WHERE` applied during candidate generation. Grammar: `Cmp` (`= <> < <= > >=`), `In`,
  `Between`, `IsNull`, `Like` (with a documented leading-wildcard scan cost), and `And`/
  `Or`. `FilterType::Timestamp` is sugar for an epoch-`INTEGER` datetime column (ISO-8601
  users declare `Text`) — sortable encoding means the plain comparisons just work.
- **Escape hatch — `Filter::sql(fragment, params)`:** a raw parameterized SQL predicate
  fenced to the filterable `doc` columns (full SQLite expression language, but untyped, a
  trusted fragment, and coupled to column names — advanced/may-break).
- **Tier 3 — post-filter predicate** (`SearchOpts::scope`) is retained.

### New dependency
- `unicode-script` (MIT OR Apache-2.0).

### Deferred to follow-ups (not in this revision)
- **Parallel rebuild** (§2) — `rebuild()` is single-threaded.
- **Tier 1 partition** (§7.5) — the partition-keyed `(partition_id, term_id)` postings +
  per-partition DF/Welford; the one hot-path-invasive filtering tier.
- **Async acquisition** (§8 Option 3) — sync core only.
- **Layer-1/2 search-warming caches** (§3) — `SearchSession` holds the warm connection;
  the posting/DF cache and incremental count vector are seams.
- **Mixed-script recall eval** validating the Welford z-score vs raw DF.
