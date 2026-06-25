<div align="center">
  <img src="docs/assets/trifle-logo.svg" alt="trifle" width="160" />
</div>

# trifle

[![CI](https://github.com/lathrys-at/trifle/actions/workflows/test.yaml/badge.svg)](https://github.com/lathrys-at/trifle/actions/workflows/test.yaml)
[![Coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/lathrys-at/trifle/badges/coverage.json)](https://github.com/lathrys-at/trifle/actions/workflows/coverage.yaml)
[![crates.io](https://img.shields.io/crates/v/trifle.svg)](https://crates.io/crates/trifle)
[![docs.rs](https://img.shields.io/docsrs/trifle)](https://docs.rs/trifle)
[![MSRV](https://img.shields.io/crates/msrv/trifle)](https://crates.io/crates/trifle)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE-MIT)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE-APACHE)

Trifle is an embedded, typo-tolerant fuzzy search engine for Rust, backed by SQLite
and tuned for large corpora of mostly small document segments with read-often and
write-infrequent characteristics. Trifle uses roaring bitmaps internally for fast and
space-efficient storage of n-gram token posting lists.

Trifle indexes text segments and returns ranked matches for quries. The backing store is
a rebuildable cache over a caller-owned full-text source. Trifle's storage can be built
as a separate sidecar or inlined within your existing SQLite database as a set of
namespaced tables.

Trifle performs lexical matching only, and is tuned for mostly small-document retrieval
by default but longer documents may be indexed and search-ranked using whatever custom
storage and ranking schemes you provide.

## Quick start

```toml
[dependencies]
trifle = "0.1"
```

```rust
use std::path::Path;
use trifle::{Config, Index, SearchOpts};

fn main() -> trifle::Result<()> {
    let index = Index::open_at(Path::new("search.db"), Config::default())?;

    // A segment is (doc_id, source, ref, text); source and ref are opaque
    // provenance labels returned on a match.
    index.insert(1, "field", &[("title", "the quick brown fox")])?;
    index.insert(2, "field", &[("title", "the quack brown ox")])?;

    // A misspelled query still matches.
    let hits = index.search("quikc brown", SearchOpts::new(10))?;
    assert_eq!(hits[0].doc_id, 1);
    Ok(())
}
```

## Data model

A segment is `(doc_id, source, ref, text)`. The fields are caller-assigned and play
distinct operational roles:

- `doc_id` — the unit of retrieval. A search returns at most one match per `doc_id`, its
  best-matching segment, and `limit` counts `doc_id`s. Make `doc_id` whatever you want back
  as a single result.
- `source` — the unit of write. `insert(doc_id, source, …)` replaces every segment under a
  `(doc_id, source)` pair; `remove_source(doc_id, source)` deletes that pair; `remove(doc_id)`
  deletes every source of a doc.
- `ref` — a free-form label on one segment, returned on a match so you know which segment
  matched. It is metadata, not a key: there is no replace or delete by `ref`.
- `text` — stored raw; the tokenizer normalizes it for matching.

These roles cover two distinct patterns:

- **Provenance** — one logical document per `doc_id`, with a segment per place its text comes
  from: `source` the category (`"ocr"`, `"caption"`, `"field"`), `ref` the sub-location (a
  filename or field name). A search returns the document and its best-matching segment.
- **Chunking a large document** — if one best passage per document is enough, keep `doc_id`
  the document, put the chunks under a single `source`, and use `ref` for each chunk's
  position; a match returns the best chunk, with its text and (when locatable) a byte span.
  To retrieve several passages from the same document at once, give each chunk its own
  `doc_id` (results are deduplicated per `doc_id`) and record the parent in `source` or `ref`.

## Features

- **Typo / partial tolerance** via trigram overlap; strictness (`min_shared`) and recall
  (`t_max`, the rarest query tokens kept) dials.
- **Configurable normalization** — NFC (default), NFD, accent-insensitive
  (`NfdStripMarks`), or none. Unicode casefolding is on by default.
- **Reranking** — bit-sliced posting list overlap generates candidates; the default
  `Effort::Medium` reranks a pool of ~`c·√(k·N)` with a BM25-shaped tier (idf, length
  normalization, literal verification). Tune via `SearchOpts::rerank(Effort)` (`None`
  through `Max`), or supply a custom `Ranker`.
- **Scoped search** — a provenance predicate evaluated over candidates only.

## Usage

Trifle assumes a read-often, write-infrequent workload over small documents, a single
writer, and that it is a derived cache over a source of truth you own. It never writes to
your data and does not know when your source changes — deciding when a segment is stale and
repopulating it is your responsibility. The API is synchronous and `&self`-thread-safe:
writes serialize through one connection, reads run concurrently under WAL, and an async
caller dispatches to a blocking pool.

### Sidecar mode (default)

Trifle owns its own SQLite file — WAL, `mmap`, one write connection and a pool of read
connections — and manages the pragmas and write serialization itself. You open it and use
it; nothing else is required:

```rust
let index = Index::open_at(Path::new("search.db"), Config::default())?;
```

### Shared mode

Trifle's tables live namespaced inside a database you own and supply connections to. Use it
only for a hard co-location requirement. You give it the write connection and a factory for
read-only connections, and you take on three guarantees Trifle cannot enforce across a file
it does not own:

- single-writer serialization across the **whole** database, not just Trifle's tables;
- a WAL and pragma setup compatible with concurrent reads;
- never holding an open transaction across a `rebuild()` — its shadow-table swap must commit.

```rust
let backend = Shared::new(
    Namespace::prefixed("trifle_")?,   // tables become trifle_seg, trifle_post, …
    write_conn,                        // the one connection writes serialize through
    || open_readonly_connection(),     // read-only factory: || -> rusqlite::Result<Connection>
)?;
let index = Index::open(backend, TrigramTokenizer::new(), Config::default())?;
```

### Maintenance

Writes are cheap because they append to a delta; bounding that growth is an explicit step,
not automatic:

- **Compact.** `compact()` folds deltas into bases. Call it when `stats().delta_backlog`
  grows — that is the signal. It bounds delta growth but does not shrink the file.
- **Rebuild.** `rebuild(corpus)` reindexes from scratch via an atomic shadow swap. It is
  required after a tokenizer change or a bumped `Config::data_version`, both of which empty
  the cache on open.
- **Drift.** On open, a schema, tokenizer-fingerprint, or `data_version` mismatch (or
  detected corruption) drops the cache to empty rather than migrating it. Treat an empty
  index after such a change as expected and repopulate with `rebuild` from your source of
  truth.

## Comparison

How Trifle compares to other fuzzy- and substring-search tools, across the axes that matter
when choosing one:

| | embedded (no server)? | updates | scales to 1M+ small docs? | provenance | matching semantics | storage |
|---|---|---|---|---|---|---|
| **[trifle](https://github.com/lathrys-at/trifle)** | yes | incremental (base + delta) | yes | source / ref | trigram overlap + BM25-shaped rerank | disk (SQLite) |
| **[SQLite FTS5](https://www.sqlite.org/fts5.html#the_trigram_tokenizer)** | yes | incremental | yes | rowid | trigram substring (`MATCH` / `LIKE`) | disk (SQLite) |
| **[pg_trgm](https://www.postgresql.org/docs/current/pgtrgm.html)** | no (server) | incremental (GIN / GiST) | yes | table rows | trigram similarity | disk (server) |
| **[Tantivy](https://github.com/quickwit-oss/tantivy)** | yes | incremental (segments) | yes | stored fields | Levenshtein automaton (≤ 2 edits) | disk (segments) |
| **[fzf](https://github.com/junegunn/fzf) / [nucleo](https://github.com/helix-editor/nucleo) / [fuzzy-matcher](https://github.com/skim-rs/fuzzy-matcher)** | yes | rebuild on startup | RAM-bound | none | subsequence | RAM |
| **[fst](https://github.com/BurntSushi/fst) / [SymSpell](https://github.com/wolfgarbe/SymSpell) / [strsim](https://github.com/rapidfuzz/strsim-rs)** | yes | rebuild (immutable) | yes | key / term | edit-distance / delete-neighborhood | mmap / RAM |

The in-memory matchers (fzf, nucleo, fuzzy-matcher) are faster when the corpus stays
RAM-resident and is rebuilt each run, but they keep no durable index. pg_trgm fits when you
already run Postgres; Tantivy is the fuller embedded library — field schemas, stored
documents, edit-distance term queries — when you want Lucene-shaped search. FTS5 and Trifle
both live in a SQLite file: FTS5 matches substrings against its trigram index through
`MATCH`/`LIKE`, while Trifle generates candidates by trigram overlap and reranks them with a
BM25-shaped scorer.

## Non-goals

Embeddings and semantic search; fusion (e.g. RRF) with other signals; an exact precision
tier beyond a custom `Ranker`; sub-trigram (< 3-char) queries; and deciding when the cache
is stale relative to your source of truth.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for
inclusion in trifle by you, as defined in the Apache-2.0 license, shall be dual
licensed as above, without any additional terms or conditions.
