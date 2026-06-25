# trifle benchmarks

Performance and recall evaluation harness for Trifle's performance numbers. There are three
separate evals:

- **`latency` / throughput** — no labels; realistic corpus and query commonness.
- **`relevance`** — MS MARCO real dev queries + qrels, vs BM25. Real queries share no
  guaranteed substring with their answer.
- **`fuzzy`** — GeoNames entity names with injected edits to test typo tolerance.

All engines run in-process on the same task, corpus, and queries, linking the same bundled
SQLite, so the comparison isolates matching strategy from store.

## Engines

| Engine | Description |
|--------|-----------|
| **trifle** | this crate |
| **fts5-trigram-bm25** | trigram FTS5 table, `ORDER BY rank`, BM25 |
| **fts5-word-bm25** | `unicode61` FTS5 table, BM25 |
| **like-scan** | `LIKE '%…%'` over a plain table of all segments |

The `latency` harness uses phrase-MATCH FTS5, `relevance` compares against word-level BM25
and the trigram cousin, and `fuzzy` compares against FTS5 trigram-MATCH plus the LIKE floor.

## Corpora

| Corpus / Command | Description |
|------------|------|-----|
| `synthetic` *(latency/profile, default)* | real English words (dwyl wordlist) sampled with a Zipfian law, 6–20 words/doc |
| `msmarco` *(latency/profile)* | a deterministic subsample of MS MARCO passages for real prose with real n-gram co-occurrence |
| `relevance` *(command)* | MS MARCO passages built answers + distractors: every qrel-relevant passage for the sampled real dev queries, plus `--docs` count filling the remaining cap with random distractors; guarantees the known answer is indexed, so recall@k measures *ranking it over the distractors* for a real paraphrased query |
| `geonames-cities` *(fuzzy, default)* / `geonames-all` *(fuzzy)* | GeoNames place names (cities > 15k pop, ~34k; or the full gazetteer, ~12M) for short, structured text segments with natural near-match distractors |

Assets are download on first use and hash-verified where the source is immutable, and stored in
`.cache/bench/<corpus>/`. GeoNames dumps regenerate roughly daily and are intentionally unpinned.
See [`ASSETS.md`](ASSETS.md) for licenses.

## Evals

- `latency`/`profile` — in-corpus document snippets (2–5 words, 0–2 typos), no labels.
- `relevance` — real MS MARCO dev queries, labeled by qrels. They paraphrase an information
  need (no guaranteed substring with the answer).
- `fuzzy` — entity name + exactly *k* edits (transposition, substitution, deletion,
  insertion, weighted toward adjacent-key typos), labeled by the entity. Reported 1- vs
  2-edit, with a trigram-survival column and a near-distractor density.

Both recall evals tag each miss selection / below-floor / ranking — whether the fix lives in
the pruner (`m`, `B`) or the `Ranker`.

## Reproducibility

A single master `--seed` drives corpus sampling and query generation so a seed reproduces a
run byte for byte.

## Running

If the build environment is offline, `fetch` the assets on a connected machine first.
(`relevance` needs the ~1 GiB MS MARCO collection; `fuzzy --corpus geonames-cities` is a few
MB.)

```bash
# 1. warm the cache (downloads + verifies where the source is immutable)
cargo run -p trifle-benchmarks --release -- fetch --corpus relevance
cargo run -p trifle-benchmarks --release -- fetch --corpus geonames-cities

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

Trifle targets flat latency as the corpus grows. Sweep to confirm:

```bash
for n in 10000 50000 100000 500000 1000000; do
  cargo run -p trifle-benchmarks --release -- latency --docs "$n" --seed 42
done
```

### Latency JSON, plots & profiling

The `latency` command also reports **in-corpus recall@k** for the sampled queries (each
query is a snippet of a real document; that document is the relevant answer), and can:

- **measure several efforts from one index build** — `--effort-sweep low,medium,high`;
- **emit machine-readable JSON** — `--format json` writes one object carrying, per
  (engine, effort), the p50/p90/p99/max latency, throughput, recall@k, **and the raw
  per-query samples** (so a post-processor can recompute anything without re-running);
- **profile a run** — `--instrument xctrace` (Instruments' Time Profiler, macOS) or
  `--instrument samply` (cross-platform) re-execs the run under a sampling profiler and
  writes a trace artifact. A hook for *where* the time goes, separate from the JSON's
  *how much*.

`tools/latency_plot.py` drives all of this: it sweeps the corpus-size ladder, persists the
raw JSON, and renders the grouped p50/p90/p99 chart (recall@k + `*`max annotated) plus the
throughput-vs-`N` plot. See [`tools/README.md`](tools/README.md).

```bash
cargo run -p trifle-benchmarks --release -- fetch --corpus msmarco   # ~1 GiB, once
python3 benchmarks/tools/latency_plot.py --queries 100 --seed 42     # sweep + plots
python3 benchmarks/tools/latency_plot.py --reuse-raw                 # re-plot, no re-run
```

## Caveats

These are not your users' queries. Prefer relative signal (trifle vs BM25; typo vs no-typo;
tail vs median) over absolute numbers, and re-run on your own corpus before choosing
search parameters (effort, sampling depth, etc.) in your application.

- **`relevance` understates recall.** MS MARCO dev qrels are sparse (~1 judged passage per
  query), so set-recall@k against a single label is a narrow slice — read it as "did the one
  known answer land in the top k." Both engines score against the identical in-corpus qrels
  and identical k (queries with no in-corpus answer are dropped for every engine); the
  `scored-queries` count is the denominator.
- **`fuzzy` does not transfer to prose.** Entity-name fuzzy is a favorable regime (short,
  structured, low-paraphrase); it validates the fuzzy machinery, not retrieval over
  paraphrase-heavy prose which is `relevance`'s job. Watch the near-distractor density: if
  low, no confusables were sampled and the numbers are inflated.
