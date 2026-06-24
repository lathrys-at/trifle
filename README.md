# trifle

[![CI](https://github.com/lathrys-at/trifle/actions/workflows/test.yaml/badge.svg)](https://github.com/lathrys-at/trifle/actions/workflows/test.yaml)
[![crates.io](https://img.shields.io/crates/v/trifle.svg)](https://crates.io/crates/trifle)

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

A segment is `(doc_id, source, ref, text)`:

- `doc_id` — your document identifier.
- `source`, `ref` — opaque provenance labels returned on a match (e.g. `source` a
  category like `"ocr"` / `"caption"`, `ref` a field name or filename). trifle indexes
  `(doc_id, source)`, so per-category replace and delete are cheap.
- `text` — stored raw; the tokenizer normalizes for matching.

A document may have many segments. `insert` replaces all segments under a `(doc_id,
source)` pair; `remove_source` deletes that pair; `remove` deletes a whole `doc_id`.

## Features

- **Typo / partial tolerance** via trigram overlap; strictness (`min_shared`) and recall
  (`breadth`) dials.
- **Configurable normalization** — NFC (default), NFD, accent-insensitive
  (`NfdStripMarks`), or none. Unicode casefolding is on by default.
- **Reranking** — bit-sliced posting list overlap generates candidates; the default
  `Effort::Medium` reranks a pool of ~`c·√(k·N)` with a BM25-shaped tier (idf, length
  normalization, literal verification). Tune via `SearchOpts::rerank(Effort)` (`None`
  through `Max`), or supply a custom `Ranker`.
- **Scoped search** — a provenance predicate evaluated over candidates only.
- **Concurrency** — one internal writer plus a pool of read-only connections under WAL,
  or configurable connection handling in shared-mode. Fully Synchronous API, with no
  runtime imposed.
- **Maintenance** — `compact()` folds deltas into bases; `rebuild()` reindexes via an
  atomic shadow swap; `stats()` reports `delta_backlog` to decide when to compact.
- **Rebuildable cache** — a tokenizer change or a bumped `data_version` drops the cache;
  repopulate with `rebuild`.

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
