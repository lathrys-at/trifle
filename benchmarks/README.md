# trifle benchmarks

The rerunnable harness behind trifle's performance numbers. Three evals, each separate:

- **`latency` / throughput** — no labels; realistic corpus and query commonness.
- **`relevance`** — MS MARCO real dev queries + qrels, vs BM25. Real queries share no
  guaranteed substring with their answer.
- **`fuzzy`** — entity name + injected edits over GeoNames; tests typo tolerance (corrupted
  target name, want the target).

> **Not published.** `trifle-benchmarks` is a workspace member with `publish = false`,
> excluded from the `trifle` package, so it adds no dependency to what downstream users
> compile.

## Baselines

All engines run in-process on the same task, corpus, and queries, linking the same bundled
SQLite, so the comparison isolates matching strategy from store.

### Engines

| Engine | What it is | Role |
|--------|-----------|------|
| **trifle** | this crate | the subject |
| **fts5-trigram-bm25** | a trigram FTS5 table, `ORDER BY rank` (BM25) | latency baseline (phrase MATCH); fuzzy baseline (trigram OR-bag) |
| **fts5-word-bm25** | a `unicode61` FTS5 table, BM25 | canonical BM25 baseline for `relevance` |
| **like-scan** | `LIKE '%…%'` over a plain table | the naive substring floor |

Each baseline is matched to its eval. `relevance` compares against word-level BM25 and the
trigram cousin. `fuzzy` compares against FTS5 trigram-MATCH plus the LIKE floor, never
bm25-phrase: an exact-term bm25 scores ~0 on a typo, so a "win" against it is meaningless.
`latency` uses phrase-MATCH FTS5.

### Capability matrix

Latency is one axis. The matrix records the others, including where trifle is not the best
choice:

| | durable? | embedded / no server? | incremental update vs rebuild? | corpus-scale (100k+ small docs)? | provenance? | matching semantics | footprint |
|---|---|---|---|---|---|---|---|
| **trifle** | ✅ disk (SQLite) | ✅ | ✅ incremental (base+delta) | ✅ | ✅ (source/ref) | trigram overlap | disk |
| FTS5-trigram | ✅ | ✅ | ✅ incremental | ✅ | rowid only | trigram + BM25 | disk |
| pg_trgm | ✅ | ❌ server | ✅ | ✅ | table cols | trigram similarity | disk |
| Tantivy + Levenshtein | ✅ | ✅ | ✅ (segments) | ✅ | fields | Levenshtein automaton | disk |
| fzf / nucleo / fuzzy-matcher | ❌ | ✅ | rebuild-on-startup | ⚠️ RAM-bound | — | subsequence | RAM |
| fst / SymSpell / strsim | ⚠️ immutable | ✅ | rebuild-to-update | ✅ | key-oriented | delete-neighborhood / edit-distance | RAM/disk |

The in-memory subsequence filters are faster than trifle when RAM-resident and rebuilt on
startup. The out-of-process engines (pg_trgm, Tantivy, fzf, fst, …) are not wired into this
harness; run them in their own drivers on the same corpus and queries.

## Corpora

| corpus / command | What | Why |
|------------|------|-----|
| `synthetic` *(latency/profile, default)* | real English words (dwyl wordlist) sampled with a Zipfian law, 6–20 words/doc | trigram document-frequencies look like real text. A tiny vocabulary would collapse every trigram onto near-every document — a degenerate dense-posting regime. |
| `msmarco` *(latency/profile)* | a deterministic subsample of MS MARCO passages | real prose, real co-occurrence; the strongest single fit for latency. |
| `relevance` *(command)* | MS MARCO passages built answers + distractors: every qrel-relevant passage for the sampled real dev queries, plus `--docs` random distractors | guarantees the known answer is indexed, so recall@k measures *ranking it over the distractors* for a real paraphrased query — not whether the answer happened to fall in a random subsample. |
| `geonames-cities` *(fuzzy, default)* / `geonames-all` *(fuzzy)* | GeoNames place names (cities > 15k pop, ~34k; or the full gazetteer, ~12M) | short, structured, low-paraphrase names — the regime where name+edit injection is faithful, with natural near-match distractors (many similar names). |

Assets download on demand, hash-verified where the source is immutable, into the gitignored
`.cache/bench/<corpus>/` (namespaced per corpus so files never collide). Bytes are never
committed — only the manifests in `sources/`. GeoNames dumps regenerate roughly daily and
are intentionally unpinned. See [`ASSETS.md`](ASSETS.md) for licenses (MS MARCO is
non-commercial research only; GeoNames is CC BY 4.0).

## Queries

- `latency`/`profile` — in-corpus document snippets (2–5 words, 0–2 typos), no labels.
- `relevance` — real MS MARCO dev queries, labeled by qrels. They paraphrase an information
  need (no guaranteed substring with the answer). Self-derived snippets are a known-item
  smoke test against a zero-by-construction baseline, so they are not used.
- `fuzzy` — entity name + exactly *k* edits (transposition, substitution, deletion,
  insertion, weighted toward adjacent-key typos), labeled by the entity. Reported 1- vs
  2-edit, with a trigram-survival column and a near-distractor density.

Both recall evals tag each miss selection / below-floor / ranking — whether the fix lives in
the pruner (`m`, `B`) or the `Ranker`.

## Reproducibility

A single master `--seed` drives corpus sampling and query generation (independent streams),
so a seed reproduces a run byte for byte. Change it to resample corpus and queries together;
keep it fixed to compare a code change against a baseline. The size knobs (`--docs`,
`--queries`, `--k`, `--repeat`) trace how cost scales with corpus size.

## Running

If the build environment is offline, `fetch` the assets on a connected machine first.
(`relevance` needs the ~1 GiB MS MARCO collection; `fuzzy --corpus geonames-cities` is a few
MB.)

```bash
# 1. warm the cache (downloads + verifies where the source is immutable)
cargo run -p trifle-benchmarks --release -- fetch --corpus relevance       # collection + queries + qrels (~1 GiB)
cargo run -p trifle-benchmarks --release -- fetch --corpus geonames-cities  # ~3 MB

# 2. latency + throughput (serial; add --batched, or --concurrent 8)
cargo run -p trifle-benchmarks --release -- latency --docs 100000 --queries 5000 --seed 42

# 3. relevance: set-recall@k on real dev queries+qrels, vs word/trigram BM25
cargo run -p trifle-benchmarks --release -- relevance --docs 100000 --queries 5000

# 4. fuzzy: name+edit recall (1- and 2-edit), vs FTS5 trigram-MATCH + the LIKE floor
cargo run -p trifle-benchmarks --release -- fuzzy --corpus geonames-cities --queries 5000
#    --corpus geonames-all   for the corpus-scale fuzzy run

# 5. the work-done curve: Σ(kept-posting cardinality)
cargo run -p trifle-benchmarks --release -- profile --docs 1000000
```

Run `… -- help` for the full option list. Always build `--release` — debug numbers are
meaningless.

### Scaling sweep

trifle targets flat latency as the corpus grows. Sweep to confirm:

```bash
for n in 10000 50000 100000 500000 1000000; do
  cargo run -p trifle-benchmarks --release -- latency --docs "$n" --seed 42
done
```

A flat p50/p99 confirms the property; a degrading curve means an assumption broke. `profile`
tags each query with Σ(kept-posting cardinality); correlate its p99 with latency p99. If the
tail tracks it, the residual is big-bitset AND/XOR cost (expected); if not, check hydration
or the predicate.

## Caveats

These are not your users' queries. Prefer relative signal (trifle vs BM25; typo vs no-typo;
tail vs median) over absolute numbers, and re-run on your own corpus.

- **`relevance` understates recall.** MS MARCO dev qrels are sparse (~1 judged passage per
  query), so set-recall@k against a single label is a narrow slice — read it as "did the one
  known answer land in the top k." Both engines score against the identical in-corpus qrels
  and identical k (queries with no in-corpus answer are dropped for every engine); the
  `scored-queries` count is the denominator.
- **`fuzzy` does not transfer to prose.** Entity-name fuzzy is a favorable regime (short,
  structured, low-paraphrase); it validates the fuzzy machinery, not retrieval over
  paraphrase-heavy prose — that is `relevance`'s job. Watch the near-distractor density: if
  low, no confusables were sampled and the numbers are inflated.
