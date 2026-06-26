# Changelog

## 0.2.0 — `rev-v0.2` (unreleased)

A ground-up rework. **Breaking across the board, and a hard cache reset:** the on-disk
`SCHEMA_VERSION` bumps to 2, so an existing index drops its cache on open and must be
`rebuild()`-ed. trifle is a derived, rebuildable cache, so this is data-loss-free for the
*cache* (the caller's source of truth is untouched).

### Term model & storage
- **Term interning.** Postings (`post`/`delta`/`term`-df) and the forward index are now
  keyed by an interned `u32` term-id behind a faulting dictionary (reader `resolve`,
  writer `intern`), not gram text — narrower B-trees, smaller index.
- **Forward index as term-ids.** Every segment stores its `u32` term-id set, so **delete
  needs neither the text nor the tokenizer** regardless of storage mode.
- **`u128` term encoding.** A gram (≤3 codepoints) + script tag packs big-endian into a
  `u128`; the script byte is the most-significant byte (script-contiguous order).
- **Script-segmented tokenizer.** New `ScriptTokenizer` splits text into maximal
  same-script runs and tokenizes each with a script-appropriate window (CJK bigrams, else
  trigrams), emitting no cross-script grams. The default `TrigramTokenizer` is unchanged.
- **Class-normalized pruning.** The rarest-first pruner ranks by a per-script-class
  Welford z-score (log space), falling back to raw DF for sparse classes.

### Data model & API (breaking)
- **Runtime `Schema`** replaces the fixed `(doc_id, source, ref, text)` model: a declared
  **key** (shape `Integer`/`Text`/`Blob`) plus named **text fields**, each with a
  per-field `StorageMode` (`Stored` / `Resolver` / `CoordinatesOnly`). Documents are
  `key → named segments` over a two-level `doc`+`seg` store.
- **`Segment`/`Match{doc_id,source,ref_}` → `Document` / `Match{key,label,span,text}`**;
  `Key` is `Integer`/`Text`/`Blob`. `TextResolver` and the scope predicate move to
  `(&Key, &str label)`.
- **Lease-based access.** `Index` shrinks to lifecycle (`open`/`rebuild`/`compact`/
  `stats`/`reader`/`writer`/`session`). Writes go through a `Writer` lease (the exclusive
  single-writer lock; six methods `insert`/`upsert`/`remove` × whole/segment;
  `commit()` is commit-and-continue; drop rolls back). Reads go through a `Reader`;
  `SearchSession` holds a warm connection for as-you-type. `SearchOpts` gains a
  `Hydration` cost ladder.
- **`Index::open` now takes a `Schema`**: `Index::open_at(path, schema, config)` /
  `Index::open(backend, tokenizer, schema, config)`.
- **Drift** now also covers a **schema fingerprint** (alongside schema version, tokenizer
  fingerprint, and `data_version`).

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
- **Integration-test migration** to the v0.2 API (the library, its unit tests, and the
  doctests are green; the `tests/*.rs` binaries still target the v0.1 API).
