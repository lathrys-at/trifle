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
  indexed text field is stored, always returned on a match, and always surfaced to the
  reranker (so the reranker can never run textless). Filterable **payload** columns are
  stored separately for filtering only — never reranked or returned. `Match.text` is a
  `String` (no longer `Option`).
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

### Ranking — real BM25+
- The default precision-tier reranker is now **BM25+ over the index's terms** (the
  n-grams), with the segment as the unit: `idf` from `df` over `N` segments, length
  normalization against the **online mean segment length** (`avgdl`, maintained from
  rolling `seg_count`/`seg_len_sum` meta counters in the write transaction), and the BM25+
  `δ` lower bound. The previous word-tokenized, ad-hoc substring/literal-verification tier
  is **gone** (frontends annotate exact substrings). An application needing true
  term-frequency can recompute it from each candidate's segment text in a custom `Ranker`
  (`Candidate::matched_terms`/`seg_len` expose the BM25 inputs); the segment is the ranking
  unit, so **cross-segment fusion happens above trifle** (aggregate results across your keys),
  not in a `Ranker`.

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
