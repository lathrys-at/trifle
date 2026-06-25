# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What trifle is

`trifle` is an embedded, typo-/partial-tolerant **trigram fuzzy search** library backed by
SQLite, tuned for **large corpora of small documents** (≲ 1–2 KB/segment), read-often /
write-infrequent. It is a **derived, rebuildable cache** over a caller-owned source of
truth — it never touches the caller's data store, and any version/tokenizer drift drops the
cache rather than migrating it.

It is *lexical* fuzzy search, deliberately **not** semantic search and not a long-document
relevance engine. Candidates come from trigram overlap; the default reranker adds a
BM25-shaped score (idf, length normalization, literal verification) on top, but the design
assumes small segments throughout.

## Commands

This is a Cargo workspace: the root package `trifle` plus one unpublished member,
`benchmarks` (`trifle-benchmarks`). MSRV is **1.85**, edition **2024**.

```bash
cargo build                                              # build the library
cargo test                                               # root-package tests (lib + integration + doc)
cargo test --workspace                                   # also builds/tests the benchmarks crate
cargo test --test typo                                   # one integration-test binary (tests/typo.rs)
cargo test --test thrash some_case_name                  # one test by name within a binary
cargo test --doc                                         # doc-tests only

cargo fmt --all                                          # CI gate: `cargo fmt --all --check`
cargo clippy --all-targets --all-features -- -D warnings # CI also runs with RUSTFLAGS="-D warnings"
cargo doc --no-deps --all-features                       # CI gate with RUSTDOCFLAGS="-D warnings"
cargo deny check licenses bans sources                   # license/advisory gate (needs cargo-deny)
```

Notes that bite:

- **Lint bar is load-bearing.** The workspace denies `clippy::all` and `unsafe_op_in_unsafe_fn`,
  and CI sets `-D warnings` everywhere (build, clippy, **and rustdoc**). A warning is a CI
  failure; a doc-comment with a broken intra-doc link fails the `doc` job.
- `cargo test` from the root tests the `trifle` package only — the benchmark crate needs
  `--workspace` or `-p trifle-benchmarks`. CI's test job runs `cargo test --verbose` (root only),
  plus a separate MSRV-1.85 build job and a `cargo-deny` job.
- The only Cargo feature is `tracing` (off by default); enable hot-path spans with
  `--features tracing`. With it off, the instrumentation macros expand to nothing.

### Benchmarks (`benchmarks/`)

A separate `publish = false` crate; it adds no dependency to what downstream users compile.
Always run `--release` (debug numbers are meaningless). Corpora download on demand,
hash-verified, into the gitignored repo-root `.cache/bench/` — fetch on a networked machine
first, then run offline.

```bash
cargo run -p trifle-benchmarks --release -- fetch --corpus synthetic   # warm the cache first
cargo run -p trifle-benchmarks --release -- latency --docs 100000 --seed 42
cargo run -p trifle-benchmarks --release -- relevance --docs 100000    # MS MARCO recall@k vs BM25
cargo run -p trifle-benchmarks --release -- fuzzy --corpus geonames-cities  # name+edit recall
cargo run -p trifle-benchmarks --release -- profile --docs 1000000     # the work-done curve
cargo run -p trifle-benchmarks --release -- help
```

See `benchmarks/README.md` for the evals, corpora, and `tools/calibrate_pool.py` (the
rerank-pool calibration).

## Architecture

The rationale for non-obvious choices lives in the module-level doc comments (`//!`) — start
with the module named for the stage you are touching. The query pipeline below is the spine.

### The model (read `src/lib.rs` first)

A **segment** is `(doc_id, source, ref, text)`. `source`/`ref` are two opaque provenance
labels returned on a match. `insert(doc_id, source, …)` replaces *all* segments under that
`(doc_id, source)` pair; `remove(doc_id)` drops every segment of a doc. A `Match` carries the
provenance, the matched byte span, and (unless contentless) the segment text.

`Index<T: Tokenizer = TrigramTokenizer, B: Backend = Sidecar>` is generic over **two** type
parameters, both defaulted (so the common type is just `Index`). The tokenizer is a *type*
parameter, not a trait object, because it is on the hot path and must monomorphize.

### The query pipeline

A search flows through four stages, each its own module — this is the spine of the codebase:

1. **Tokenize** (`src/tokenize.rs`) — one `Tokenizer` runs on indexed text, postings, *and*
   queries, so "index agrees with query" holds by construction. The built-in `NgramTokenizer`
   (`Trigram`/`Bigram`/`Quadgram` aliases) slides an N-codepoint window over a normalized form
   and emits zero-allocation inline `Ngram` tokens. Normalization (NFC default, NFD,
   accent-stripping, casefold) is the tokenizer's job.
2. **Select** (`src/select.rs`) — keeps a **rarest-first** prefix of the query's tokens, from
   the typo floor `F = m + d` up to `t_max` (rare = both cheapest to scan and most
   discriminating; the kept tokens' `Σdf` is the rows scanned). Derives only from *this
   query's* token document-frequencies, never a batch aggregate — so
   `search_batch([…,q,…])` ranks `q` identically to `search(q)` (**batch == serial**, an
   invariant tested in `tests/scope_ranker.rs`).
3. **Candidate generation + rank** (`src/rank.rs`) — reads each selected token's roaring
   posting (no decode: an owned roaring posting *is* the bitmap) and counts overlap with a
   **bit-sliced counter** (counts held across bitmap "bit planes"; adding a posting is a
   ripple-carry binary add), making candidate generation `O(k·log k)` bitmap ops —
   **independent of posting size** (the "flatness" claim). A high→low bucket walk hydrates
   provenance and early-stops once `limit` results lock. A pluggable `Ranker`
   (default `OverlapRanker`) then orders survivors; a custom one can add a precision tier.
4. **Hydrate** — survivors' text comes from the stored snapshot or, in contentless mode, the
   caller's `TextResolver`, in one batched `WHERE id IN rarray(?1)` read.

### Storage (`src/postings.rs`, `src/schema.rs`, `src/store/`)

- **Owned roaring inverted index.** Each token maps to a `base ∪ added \ removed` roaring
  posting. The three-way **write-frequency split is deliberate**: every write touches only the
  small `term.df` + `delta` rows; the big `post.base` is rewritten *only* by `compact()`'s fold
  or a `rebuild()`. A read needs no freshness gate — the delta is committed in the same
  transaction as the segment.
- **Schema** (`src/schema.rs`) — all table names come from a validated `Namespace` (no SQL
  injection surface in the interpolated DDL). Three version stamps — schema version, tokenizer
  fingerprint, caller `data_version` — gate drift; a mismatch (or a broken id-allocation
  invariant) **resets the cache, never migrates**. Monotonic id allocation + the atomic
  shadow-table swap that `rebuild()` uses live here.
- **Backends** (`src/store/`, behind the `Backend` trait):
  - `Sidecar` (default) — trifle owns its own SQLite file: WAL, `mmap`, one mutexed write
    connection plus a pool of read-only connections that run concurrently with the writer.
  - `Shared` (opt-in) — trifle's tables live *namespaced* inside a database the caller owns and
    supplies connections to; use only for a hard co-location requirement.
  - `src/store/pool.rs` holds the machinery both share (writer `Mutex` + on-demand read pool).
- **Concurrency / threading** — the API is fully **synchronous and `&self`**-thread-safe; no
  async runtime is imposed. One writer is serialized; reads run on the pool under WAL.
  An async caller dispatches to a blocking pool. Transient `SQLITE_BUSY`/`LOCKED`/`SCHEMA`
  faults are retried internally (`read_retry`) before surfacing as `Error::Sqlite`.

### Maintenance & lifecycle

`compact()` folds deltas into bases (bounds delta growth; doesn't shrink the file).
`rebuild(corpus)` fully reindexes via an atomic shadow swap (required after a tokenizer change
or `data_version` bump, both of which empty the cache on open). `stats()` reports
`delta_backlog` — the signal for *when* to compact.

### Cross-cutting

- `src/error.rs` — `Error`/`Result`; variants separate the three failure classes (transient
  store fault vs. fixable caller input vs. impossible internal-invariant violation). `#[non_exhaustive]`.
- `src/instrument.rs` — `trace_debug!` etc. compile to nothing unless `--features tracing`.
- **Tests** are integration-style binaries under `tests/`, one concern each (`typo`, `unicode`,
  `drift`, `lifecycle`, `ranking`, `scope_ranker`, `backends`, `adversarial`, `api`, `basic`),
  plus `tests/thrash.rs` — a **proptest** oracle that thrashes randomized op sequences against
  a reference model. `tests/common/mod.rs` holds shared fixtures.
