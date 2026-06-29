# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What trifle is

`trifle` is an embedded, typo-/partial-tolerant **trigram fuzzy search** library backed by
SQLite, tuned for **large corpora of small documents** (≲ 1–2 KB/segment), read-often /
write-infrequent. It is a **derived, rebuildable cache** over a caller-owned source of truth — it
never touches the caller's data store, and any version/tokenizer/schema drift drops the cache
rather than migrating it.

It is *lexical* fuzzy search, deliberately **not** semantic search and **not a relevance engine**.
It ranks by **IDF-weighted, class-normalized trigram overlap** computed in a bit-sliced counter:
rarer shared grams weigh more (a per-query, df-anchored 4-tier scheme, weights `{1,2,3,4}`, knob
`D`), and rarity is normalized per script class so a CJK bigram and a Latin trigram compare fairly.
There is no BM25/relevance tier; a caller wanting a domain reorder composes it over the candidate
stream. The design assumes small segments throughout.

Bitmaps are **CRoaring** (the `croaring` crate, SIMD) everywhere — both the storage posting layer
and the overlap engine.

## Commands

A Cargo workspace: the root package `trifle` plus the inner engine crate `trifle-overlap`. MSRV is
**1.85**, edition **2024**.

```bash
cargo build                                              # build the library
cargo test                                              # lib + integration + doc tests
cargo test --workspace                                  # also the trifle-overlap engine crate
cargo test --test typo                                  # one integration-test binary (tests/typo.rs)
cargo test --test thrash some_case_name                 # one test by name within a binary

cargo fmt --all                                         # CI gate: `cargo fmt --all --check`
cargo clippy --workspace --all-targets -- -D warnings   # CI also runs with RUSTFLAGS="-D warnings"
cargo doc --no-deps --all-features                      # CI gate with RUSTDOCFLAGS="-D warnings"
cargo deny check licenses bans sources                  # license/advisory gate (needs cargo-deny)
```

Notes that bite:

- **Lint bar is load-bearing.** The workspace denies `clippy::all` and `unsafe_op_in_unsafe_fn`,
  and CI sets `-D warnings` everywhere (build, clippy, **and rustdoc**). A warning is a CI failure;
  a doc-comment with a broken intra-doc link fails the `doc` job.
- The only Cargo feature is `tracing` (off by default); enable hot-path spans with
  `--features tracing`. With it off, the instrumentation macros expand to nothing.
- `benchmarks/` is **excluded from the workspace** pending a rework against the streaming candidate
  API; it does not build today and is not part of any gate.

## Architecture

The rationale for non-obvious choices lives in the module-level doc comments (`//!`) — start with
the module named for the stage you are touching. The query pipeline below is the spine.

### The model (read `src/lib.rs` first)

A [`Schema`] declares a **key** (its shape — `Integer`/`Text`/`Blob`) plus named **text fields**. A
[`Document`] is a `key` plus a set of named segments (`label → text`); a **segment** is one row of
the `seg` table — `(id, key, label, text, len)`, where `seg.id` is the posting id. A `Match` carries
the key, the matched segment's label, the matched byte span, and the segment text. There is no
`doc` table — the key lives directly on each segment row, so a key with no segments cannot
materialize a ghost row, and provenance is a single-table point read.

`Index<T: Tokenizer = DefaultTokenizer>` is generic over the tokenizer only (a *type* parameter,
not a trait object — it is on the hot path and must monomorphize). The store is the concrete
`Sidecar`. Reads/writes go through short-lived **leases**: `index.writer()` (the exclusive
single-writer lock, commit-and-continue) and `index.reader()`.

The write API is three methods on the writer: `upsert(key, &[(label, text)])` (create-or-replace
the named segments, other labels intact), `remove(key)`, and `remove_segment(key, label)`.

### The query pipeline

A search flows through these stages, each its own module:

- **Tokenize** (`src/tokenize.rs`) — one `Tokenizer` runs on indexed text, postings, *and* queries,
  so "index agrees with query" holds by construction. `DefaultTokenizer` splits text into
  maximal same-script runs and windows each (CJK bigrams, else trigrams); `NgramTokenizer<N>`
  (`TrigramTokenizer`/`BigramTokenizer`) is the fixed-width tokenizer. Normalization (NFC default,
  NFD, accent-stripping, casefold) is the tokenizer's job.
- **Select** (`src/select.rs`, with `src/welford.rs`) — keeps a **rarest-first** prefix of the
  query's tokens, from the typo floor `F = m + d` up to `t_max`. Rarity is **class-normalized**: a
  `z`-score within the token's script class (per-class mean/variance maintained in log space by
  `welford.rs`), falling back to raw df for a sparse class — so multi-script queries rank fairly.
  Derives only from *this query's* token document-frequencies, so `matches_batch([…,q,…])` ranks
  `q` identically to `matches(q)` (**batch == serial**).
- **Candidate generation** (`crates/trifle-overlap`, the inner engine crate) — the selected tokens'
  CRoaring postings are handed to a `Counter`, which counts IDF-weighted overlap in a **bit-sliced
  counter** (counts held across bitmap "bit planes"; adding a weighted posting is a ripple-carry
  add), `O(k·log k)` bitmap ops — the op count is cardinality-independent ("flatness"). A high→low
  bucket walk streams scored candidate ids best-first.
- **Provenance + filter + hydrate** (`src/search.rs`) — `search.rs` drives the engine walk in
  chunks, batch-reads each chunk's `(key, label)` from `seg` (folding in the opt-in `SqlFilter`),
  dedups one candidate per key, and finally hydrates text + span for exactly the kept candidates in
  one `WHERE id IN rarray(?1)` read.

`src/search.rs` exposes both front doors: `Reader::matches`/`matches_batch` (the eager safe default,
top-`limit` hydrated matches) and `Reader::candidates` → `CandidateStream` (the lazy spine: a
best-first cursor of provenance-only `Candidate`s that fuses on the first error; compose
rerank/pagination/fusion on top, then `hydrate` only what you keep).

### Filtering

`SqlFilter { fragment, params }` is an opt-in raw-SQL predicate over the caller's **live** data
(no trifle-stored filter columns — those would go stale). It folds into the per-chunk provenance
query as `WHERE (<fragment>) AND id IN rarray(?{N+1})` — fragment textually first, the
candidate-scope param last — so both numbered `?1..?N` and anonymous `?` in the fragment bind with
no collision. The universal mode is `key IN rarray(?)` (bind your own allowed-key set); a co-located
join is reachable via an `ATTACH` on the read connections.

### Storage (`src/postings.rs`, `src/schema.rs`, `src/store/`)

- **Owned inverted index** (`src/postings.rs`) — each token maps to a `(base ∪ added) \ removed`
  CRoaring posting. The three-way **write-frequency split is deliberate**: every write touches only
  the small `term.df` + `delta` rows; the big `post.base` is rewritten *only* by `compact()`'s fold
  or a `rebuild()`. A read needs no freshness gate — the delta is committed in the same transaction
  as the segment, and an effective posting feeds the engine directly. Blobs are the standard
  CRoaring portable format.
- **Schema** (`src/schema.rs`) — all table names come from a validated `Namespace` (no SQL injection
  surface in the interpolated DDL). Version stamps — schema version, tokenizer fingerprint, caller
  `data_version`, schema fingerprint — gate drift; a mismatch (or a broken id-allocation invariant)
  **resets the cache, never migrates**. Monotonic id allocation + the atomic shadow-table swap that
  `rebuild()` uses live here. The term dictionary (`src/dict.rs`) interns grams to `u32` ids and
  holds the per-class statistics; reads resolve in term-space and capture the dictionary generation
  so a concurrent `rebuild` is detected.
- **Store** (`src/store/`) — `Sidecar` owns its own SQLite file: WAL, `mmap`, one mutexed write
  connection plus a pool of read-only connections (`src/store/pool.rs`) that run concurrently with
  the writer under WAL. The pool rolls back any open transaction on check-in. Co-location inside a
  caller-owned database is available via an `ATTACH` on the read-connection factory.
- **Concurrency / threading** — the API is fully **synchronous and `&self`-thread-safe**; no async
  runtime is imposed. One writer is serialized; reads run on the pool under WAL. An async caller
  dispatches to a blocking pool. The library **never blocks the calling thread to retry**
  (`busy_timeout` is 0): a transient `SQLITE_BUSY`/`LOCKED`/`SCHEMA` fault — and a read racing a
  concurrent `rebuild`'s id-reassignment — surfaces immediately as the **retryable `Error::Busy`**
  (mapped at the `From<rusqlite::Error>` boundary), and the caller owns the backoff (retry on a
  fresh reader, or re-submit a write batch).

### Maintenance & lifecycle

`compact()` folds deltas into bases (bounds delta growth; doesn't shrink the file).
`rebuild(corpus)` fully reindexes via an atomic shadow swap (required after a tokenizer or
`data_version`/schema change, all of which empty the cache on open). `stats()` reports
`delta_backlog` — the signal for *when* to compact — plus segment/term counts and a corpus-fitted
`weight_step` hint accumulated from the per-query band-spreads.

### Cross-cutting

- `src/error.rs` — `Error`/`Result`; variants separate the failure classes a caller handles
  differently: a transient store fault (`Error::Busy`), fixable caller input, an impossible
  internal-invariant violation (`Error::Corrupt`), and a stranded writer handle
  (`Error::WriterStranded`). `#[non_exhaustive]`.
- `src/instrument.rs` — `trace_debug!` etc. compile to nothing unless `--features tracing`.
- **Tests** are integration-style binaries under `tests/`, one concern each (`basic`, `typo`,
  `unicode`, `drift`, `lifecycle`, `filter`, `stream`), plus `tests/thrash.rs` — a **proptest**
  oracle that thrashes randomized op sequences against a reference model. `tests/common/mod.rs`
  holds shared fixtures.

### Invariants to preserve

`batch == serial` (per-query selection only); no ghost rows (dissolved by the key-on-`seg`
flatten); the dictionary-generation guard (a read racing a `rebuild` → `Error::Busy`); monotonic
ids + the atomic shadow swap for `rebuild`; drift-reset (drop the cache, never migrate); no sleeps
(surface `Error::Busy`); flatness (engine op count is cardinality-independent); a single tokenizer
on index and query.

[`Schema`]: src/model.rs
[`Document`]: src/model.rs
