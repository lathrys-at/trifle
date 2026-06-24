# trifle

[![CI](https://github.com/lathrys-at/trifle/actions/workflows/test.yaml/badge.svg)](https://github.com/lathrys-at/trifle/actions/workflows/test.yaml)
[![crates.io](https://img.shields.io/crates/v/trifle.svg)](https://crates.io/crates/trifle)

Embedded, typo- and partial-tolerant trigram search for Rust, backed by SQLite. It is
built to stay fast over large corpora of **small documents** with a read-often,
write-infrequent shape.

trifle indexes short text segments and answers typo- and partial-tolerant queries,
returning a ranked list of matches that each carry *where* they matched. It owns a
single SQLite store holding the segment text, its provenance, and a roaring inverted
index (a base+delta posting list per token), and ranks candidates by shared-rare-token
overlap, counted bit-sliced. The store is a derived, rebuildable cache over a
caller-owned source of truth: trifle never writes to your data store.

## What it's for

Durable, embedded, incrementally updatable fuzzy search at corpus scale, with provenance
on every match. trifle is tuned for **small documents** — short segments of roughly
≤ 1–2 KB — and does lexical matching, not semantic search or long-document relevance
ranking. For semantic retrieval, an in-memory subsequence filter (fzf/nucleo), or a full
search server (Tantivy, Elasticsearch), use those instead.

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

    // A segment is (doc_id, source, ref, text). `source`/`ref` are opaque provenance
    // labels returned on a match so you know where it landed.
    index.insert(1, "field", &[("title", "the quick brown fox")])?;
    index.insert(2, "field", &[("title", "the quack brown ox")])?;

    // Typo-tolerant: a misspelled query still matches.
    let hits = index.search("quikc brown", SearchOpts::new(10))?;
    assert_eq!(hits[0].doc_id, 1);
    Ok(())
}
```

## The model

A **segment** is `(doc_id, source, ref, text)`:

- `doc_id` — your document identifier.
- `source`, `ref` — two opaque provenance labels (intended: `source` a category like
  `"field"` / `"ocr"` / `"caption"`, `ref` a sub-location like a field name or a
  filename). trifle returns them on a match and indexes `(doc_id, source)` so
  per-category replace/delete is cheap.
- `text` — stored raw; the tokenizer normalizes internally for matching.

A document may have many segments. `insert` replaces all segments under a
`(doc_id, source)` pair; `remove` removes all segments of a `doc_id`.

## Features

- **Typo / partial tolerance** via trigram overlap, with a single strictness dial
  (`min_shared`) and an orthogonal recall dial (`breadth`).
- **Configurable normalization** — NFC (default), NFD, accent-insensitive
  (`NfdStripMarks`), or none; Unicode casefolding on by default.
- **Reranking by default** — bit-sliced overlap generates candidates; the default
  `Effort::Medium` reranks a pool of about `c·√(k·N)` of them with a BM25-shaped precision
  tier (idf weighting, length normalization, literal verification). Tune the
  recall/latency trade-off with `SearchOpts::rerank(Effort)` — from `None` (overlap only,
  lowest latency) through `Max` — or supply a custom `Ranker`. The pool-depth law and the
  `Effort` constants are derived and calibrated in `benchmarks/tools/`.
- **Scoped search** — a membership predicate over provenance, evaluated only over
  candidates (never the corpus).
- **Concurrency** — a single internal writer plus a pooled set of read-only
  connections that run concurrently under WAL. Synchronous API; no runtime imposed.
- **Maintenance** — `compact()` folds deltas into bases; `rebuild()` reindexes via an
  atomic shadow swap; `stats()` reports what you need to decide when to compact.
- **Rebuildable cache** — a tokenizer change or a bumped `data_version` drops the
  cache (no migrations); you repopulate with `rebuild`.

## What it leaves to you

Embeddings / semantic search; fusion (e.g. RRF) with other signals; an exact
precision tier beyond a custom `Ranker`; sub-trigram (`< 3`-char) query handling; and
deciding *when* the cache is stale relative to your source of truth.

## Status

`0.1` — the API may still move before `1.0`, the tokenizer, ranker, and storage-backend
traits especially.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for
inclusion in trifle by you, as defined in the Apache-2.0 license, shall be dual
licensed as above, without any additional terms or conditions.
