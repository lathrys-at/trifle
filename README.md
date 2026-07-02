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
write-infrequent characteristics. Trifle uses CRoaring bitmaps internally for fast and
space-efficient storage of n-gram token posting lists.

Trifle indexes text segments and returns ranked matches for queries. The backing store is
a rebuildable cache over a caller-owned full-text source: an owned SQLite sidecar file that
trifle manages itself.

Trifle performs lexical matching only, and is tuned for mostly small-document retrieval
by default but longer documents may be indexed and search-ranked using whatever custom
storage and ranking schemes you provide.

## Quick start

```toml
[dependencies]
trifle = "0.5"
```

```rust
use std::path::Path;
use trifle::{Config, Index, Schema, SearchOpts};

fn main() -> trifle::Result<()> {
    // A flat schema: an integer key plus one text field (any segment label, stored).
    let index = Index::open_at(Path::new("search.db"), Schema::flat(), Config::default())?;

    // Writes go through a single-writer lease; commit() makes them durable.
    let mut w = index.writer()?;
    w.upsert(1, &[("title", "the quick brown fox")])?;
    w.upsert(2, &[("title", "the quack brown ox")])?;
    w.commit()?;

    // Reads go through a reader lease; each search runs under a consistent snapshot
    // (a single matches_batch shares one). A misspelled query still matches.
    let hits = index.reader()?.matches("quikc brown", &SearchOpts::new(), 10)?;
    assert_eq!(hits[0].key.as_i64(), Some(1));
    Ok(())
}
```

## Data model

A `Schema` declares the shape of your data; trifle generates its tables from it. A document
has a **key** and one or more named **segments**:

- **key** — the lifecycle handle, of a declared shape (`Integer`, `Text`, or `Blob`): the unit
  of dedup, replace, and delete. A `Match` carries the `key`, the matched segment's `label`,
  its `text`, a byte `span` (when locatable), and its `score` with the components it
  decomposes into.
- **segment** — a `(label, text)` pair under a key. `label` is a free-form name returned on
  a match so you know which segment matched; a document holds each label at most once. Every
  indexed text field is **stored** and returned on a match. The segment is the ranking **and
  retrieval** unit: by default a search returns every matching segment (a key may appear once
  per matching segment, each self-describing via its `label` and `score`), and `limit` counts
  segments. For entity-style result lists, `SearchOpts::collapse(Collapse::Key)` folds to one
  result per key — its best-scoring segment — with `limit` counting keys. Retrieval
  granularity is a search-time option, deliberately decoupled from the key's lifecycle role.
- `Schema::flat()` is the simplest shape: an integer key and one default text field that
  accepts any label. `Schema::chunked()` / the builder declare named text fields.

This covers two common patterns:

- **Provenance** — one document per key, a segment per place its text comes from (label the
  sub-location: `"ocr"`, `"title"`, a filename). A search returns the document and its
  best-matching segment.
- **Chunking a large document** — keep the document as the key and put each passage under its
  own label. A search returns every matching passage (each with its text and, when locatable,
  a byte span); collapse to `Collapse::Key` when you want just the document's best passage.

## Features

- **Typo / partial tolerance** via n-gram overlap. Strictness is `min_shared` (shared rare
  grams required for a hit); the work/recall dial is `df_budget`, which by default is
  **derived from your corpus** rather than tuned (see [Tuning](#tuning-the-defaults)).
- **Mixed-script aware** — the default `DefaultTokenizer` splits text into same-script runs
  and windows each at a script-appropriate order (CJK bigrams, else trigrams) plus a
  one-shorter secondary order, so no gram straddles a script boundary and short or corrupted
  queries keep a robustness layer. Gram rarity is **class-normalized per `(script, order)`**,
  so a rare CJK bigram and a rare Latin trigram compete fairly for selection.
  `NgramTokenizer<N>` (`TrigramTokenizer` / `BigramTokenizer`) is the plain fixed-width
  tokenizer for single-script corpora.
- **Configurable normalization** — NFC (default), NFD, accent-insensitive
  (`NfdStripMarks`), or none. Lowercasing is on by default.
- **Ranking — logit-idf energy overlap.** Each matched gram is worth its RSJ log-odds
  ("energy": rarer ⇒ more), counted bit-sliced inside the engine; a count credit rewards
  matching more of the query and a saturating length null cancels the long-segment chance-match
  bias. Scores are **nats** — log-odds-shaped, cross-query comparable in the clean limit — and
  every `Match` carries its `score` and the `energy`/`count`/`length` components it decomposes
  into. Short queries (even 2 characters) are answerable: a query too short or too damaged for
  full-width grams falls to the shorter order and the two views are rank-fused internally.
  This is a fuzzy lexical overlap engine, **not** a relevance engine — there is no BM25 tier.
  For a domain-specific reorder, pull a candidate pool from `reader.candidates(...)`, reorder it
  yourself (the stream exposes each candidate's score components and matched terms with their
  df), and hydrate the winners.
- **Bounded work per query.** Selection prunes the query to its rarest useful grams under a
  corpus-derived work budget with a confidence-bounded stop, so a query's posting-scan cost is
  bounded by the budget — not by corpus size — even for short, common-word queries.
- **Filtering** — pass a `SqlFilter` (a trusted-constant SQL predicate fragment plus bound
  params) to cut the candidate set against your **live** data — `key IN rarray(?)` with your own
  allowed-key set, or a co-located join via `ATTACH`. trifle stores no filter columns of its own
  (they would go stale), so filtering is staleness-free by construction.

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
let index = Index::open_at(Path::new("search.db"), Schema::flat(), Config::default())?;
```

A custom tokenizer or table namespace goes through `Index::open(Sidecar::open(path)?, tokenizer,
schema, config)`. To co-locate trifle's tables inside a database you own, `ATTACH` it to the
sidecar's read connections and reference it from a `SqlFilter` fragment.

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

## Tuning the defaults

You should rarely need to. Trifle's defaults are not tuned constants — nearly every one is
*derived*, either from the scoring model (`docs/derivation.md`) or from your corpus's own
statistics at query time, and each is chosen to fail in the recall-safe direction. The two
dials worth knowing on day one:

- `SearchOpts::min_shared` — strictness: how many shared rare grams a hit requires (default 2).
- `SearchOpts::df_budget` — the work budget: how many posting rows a query may scan. The
  default is derived from your corpus per query; override it to pin a latency ceiling.
- `SearchOpts::collapse` — result granularity: every matching segment (default), or one
  result per key (`Collapse::Key`).

Everything else (`Config::sigma` and the `Tuning` group: `nu`, `kappa`, `delta`, `k_target`,
`c_margin`) changes the scoring model itself, and should move only on the strength of a
measured evaluation on your corpus. **[`docs/tuning.md`](docs/tuning.md)** covers every knob:
what it means, exactly when and why you would change it, and which evaluation justifies each
change.

## Comparison

How Trifle compares to other fuzzy- and substring-search tools, across the axes that matter
when choosing one:

| | embedded (no server)? | updates | scales to 1M+ small docs? | provenance | matching semantics | storage |
|---|---|---|---|---|---|---|
| **[trifle](https://github.com/lathrys-at/trifle)** | yes | incremental (base + delta) | yes | key / label | logit-idf-weighted trigram overlap | disk (SQLite) |
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
`MATCH`/`LIKE`, while Trifle ranks by rarity-weighted gram overlap (rarer shared grams weigh
more) — a fuzzy lexical engine, not a relevance engine.

## Non-goals

Embeddings and semantic search; fusing trifle's results with *other engines'* signals (its
internal rank-view fusion is not a general fusion framework — compose external fusion over the
candidate stream); an exact precision tier beyond what you compose over the candidate stream;
and deciding when the cache is stale relative to your source of truth.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for
inclusion in trifle by you, as defined in the Apache-2.0 license, shall be dual
licensed as above, without any additional terms or conditions.
