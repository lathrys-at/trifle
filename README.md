# trifle

[![CI](https://github.com/lathrys-at/trifle/actions/workflows/test.yaml/badge.svg)](https://github.com/lathrys-at/trifle/actions/workflows/test.yaml)
[![crates.io](https://img.shields.io/crates/v/trifle.svg)](https://crates.io/crates/trifle)

Pretty good lexical/fuzzy search for Rust. Embedded, typo- and partial-tolerant
trigram search backed by SQLite, built to stay fast over large corpora of **small
documents** with a read-often / write-infrequent shape.

trifle indexes short text **segments** and answers typo/partial-tolerant queries,
returning a ranked list of matches each carrying *where* it matched. It owns a
single SQLite store holding the segment text, provenance, and an owned **roaring
inverted index** (a base+delta posting per token); it ranks by shared-rare-token
overlap, counted bit-sliced. It is a **derived, rebuildable cache** over a
caller-owned source of truth — it never touches your data store.

> **Honest about the regime.** This is *lexical* fuzzy search, not BM25-grade
> ranking and not semantic search. It omits length normalization, which is sound
> only for small documents (≲ 1–2 KB per segment). It does not (yet) ship a speed
> superlative — that's a claim to earn with a benchmark, not assert.

## What it's for

The underserved cell trifle aims at: **durable + embedded + incrementally-updatable
+ corpus-scale fuzzy search, with provenance, fast enough to feel instant.** If you
want an in-memory subsequence filter (fzf/nucleo) or a full search server (Tantivy,
Elasticsearch) or semantic retrieval, those are different tools.

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
- **Pluggable ranking** — the default orders by overlap; a custom `Ranker` can add a
  precision tier (literal verification, proximity, idf-weighting) over the text each
  candidate carries.
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

`0.1` — the API (tokenizer, ranker, and storage-backend traits especially) may still
move before `1.0`. See [the design specification](docs/design.md) for the architecture
and the rationale behind non-obvious choices.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for
inclusion in trifle by you, as defined in the Apache-2.0 license, shall be dual
licensed as above, without any additional terms or conditions.
