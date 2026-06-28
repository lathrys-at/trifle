# trifle benchmarks

Performance and recall evaluation harness for trifle. It drives trifle's streaming read API
(`index.reader()?.matches(query, &SearchOpts, limit)`) against in-process SQLite baselines on
the *same* corpus and queries, so every comparison isolates the matching strategy from the
store. The weighted-overlap order **is** the ranking — there is no rerank pool, hence no effort
knob.

There are seven benchmarks plus two utilities:

- **`latency`** — search latency + throughput on **clean** queries (no typos). Corpus
  `synthetic`/`msmarco` (in-corpus snippets) or `geonames-*` (exact entity names, the
  short-structured-segment scaling regime). An in-corpus self-recall figure rides along as a
  sanity check; since the queries are clean, every engine has well-defined recall and all are
  scored.
- **`relevance`** — recall + latency on MS MARCO real dev queries + qrels, vs BM25.
- **`fuzzy`** — recall + latency on GeoNames entity names with injected edits (typo tolerance).
- **`selsweep`** — the selection-cost frontier: recall@k vs Σdf (and vs p99 latency) for both
  selection arms, `t_max` and `df_budget` — to pick the better recall-per-unit-work knob.
- **`dsweep`** — recall/MRR/nDCG@{1,5,10,50} vs the IDF `weight_step` `D`, plus a check of the
  corpus `WeightStepHint` against the recall-optimal `D`.
- **`overlap`** — the engine in isolation: `trifle-overlap` build/walk on synthetic CRoaring
  bitmaps (no SQLite). The build-cost-vs-cardinality flatness curve, the all-weight-1 fast path,
  and shallow top-k vs a full deep-pull walk.
- **`ingest`** — write-path throughput: incremental `upsert`/`commit` docs/s, `rebuild` docs/s,
  and `compact()` cost.
- **`profile`** — the work-done curve: the Σ(kept-posting cardinality) distribution.
- **`fetch`** — warm the pinned-corpus cache before an offline run.

All engines run in-process, linking the same bundled SQLite trifle links.

> The benchmark crate is **excluded** from the trifle workspace (it is unpublished and adds no
> dependency to what downstream users compile). Build and run it via its own manifest:
>
> ```bash
> cargo build   --manifest-path benchmarks/Cargo.toml
> cargo clippy  --manifest-path benchmarks/Cargo.toml --all-targets -- -D warnings
> cargo fmt     --manifest-path benchmarks/Cargo.toml
> cargo run     --manifest-path benchmarks/Cargo.toml --release -- <command> [options]
> ```

## Engines

| Engine | `--filter` id | Description |
|--------|---------------|-------------|
| **trifle** | `trifle` | this crate |
| **FTS5 trigram BM25** | `fts5-trigram-bm25` | trigram FTS5 table, `ORDER BY rank`, BM25 |
| **FTS5 word BM25** | `fts5-word-bm25` | `unicode61` FTS5 table, BM25 |
| **LIKE scan** | `like-scan` | `LIKE '%…%'` over a plain table of all segments |

`--filter <id>` skips an engine (repeatable). `latency` uses **phrase**-MATCH FTS5 (a fair
*speed* baseline); `relevance` compares against word-level BM25 + the trigram cousin + the LIKE
floor; `fuzzy` compares against FTS5 trigram-MATCH (**OR-bag**) + the LIKE floor. (Phrase-MATCH
scores ~0 on typo'd/paraphrased queries by construction, so the recall evals never use it.)
trifle is the subject of `fuzzy` and cannot be filtered out.

## Recall methodology — one depth-50 pull

`relevance`, `fuzzy`, and `selsweep` pull every query **once** to a fixed depth `K_MAX = 50`
(the SQLite baselines run at `LIMIT 50`), *timed*. Every reported `recall@k`, `MRR@k`, and
`nDCG@k` for `k ∈ {1, 5, 10, 50}` is then a slice of that single ranked list — no per-k
re-search, so the latency and the quality numbers describe the same work. `recall@k` is
set-recall (mean over scored queries of `|top-k ∩ relevant| / |relevant|`); the `scored-queries`
count (queries with ≥1 in-corpus relevant id) is printed as the denominator.

## Corpora

| Corpus | Used by | Description |
|--------|---------|-------------|
| `synthetic` | latency/profile/selsweep (default latency/profile) | real English words (dwyl wordlist; small offline fallback vocab if no network) sampled with a Zipfian law, 6–20 words/doc |
| `msmarco` | latency/profile | a deterministic subsample of MS MARCO passages — real prose with real n-gram co-occurrence |
| `msmarco-relevance` | relevance/selsweep | every qrel-relevant passage for the sampled real dev queries, plus `--docs` random distractors; the known answer is always indexed, so recall@k measures *ranking it over the distractors* for a real paraphrased query |
| `geonames-cities` / `geonames-all` | fuzzy/latency/selsweep | GeoNames place names (cities > 15k pop, ~34k; or the full gazetteer, ~12M) — short, structured segments with natural near-match distractors |

For `latency`, the `geonames-*` corpora are queried with **exact entity names (no edits)** — a
distinct short-structured-segment scaling regime — while `synthetic`/`msmarco` use in-corpus
snippets.

Assets download on first use and cache into `.cache/bench/<corpus>/`. An immutable source pins a
`sha256` in `benchmarks/sources/` and is hash-verified; the **GeoNames** dumps are intentionally
**unpinned** (empty `sha256`) — they regenerate ~daily, so a pinned hash would go stale every day,
and the dump is used as-is without content verification (the fetch prints the computed hash for
reference). A pinned source whose hash *does* drift errors but keeps the download, so you re-pin
with the printed hash without re-fetching. See [`ASSETS.md`](ASSETS.md) for licenses.

## Reproducibility

A single master `--seed` drives corpus sampling and query generation, so a seed reproduces a
run byte for byte. The size knobs (`--docs`, `--queries`) trace how cost scales with corpus
size.

## Search tuning (trifle only)

`--min-shared <m>` (match floor), `--t-max <T>` (rarest tokens kept by selection), and
`--weight-step <D>` (df-doublings per IDF weight step) set trifle's strictness; each omitted
flag leaves the engine default (`m=2`, `t_max=12`, `D=1.0`). Baselines have no analogue and
ignore them.

## Running

If the build environment is offline, `fetch` the assets on a connected machine first.
(`relevance`/`selsweep --corpus msmarco-relevance` need the ~1 GiB MS MARCO collection;
`fuzzy --corpus geonames-cities` is a few MB. `synthetic` needs no network beyond the one-time
wordlist, and degrades to a small built-in vocab if offline.)

```bash
# 0. warm the cache (downloads + verifies where the source is immutable)
cargo run --manifest-path benchmarks/Cargo.toml --release -- fetch --corpus relevance
cargo run --manifest-path benchmarks/Cargo.toml --release -- fetch --corpus geonames-cities

# 1. latency + throughput (speed only; add --batched, or --concurrent 8)
cargo run --manifest-path benchmarks/Cargo.toml --release -- latency --docs 100000 --queries 5000 --seed 42
#    short-structured-segment scaling regime (exact entity names, no typos):
cargo run --manifest-path benchmarks/Cargo.toml --release -- latency --corpus geonames-all --docs 625000

# 2. relevance: recall/MRR/nDCG@{1,5,10,50} + latency on real dev queries+qrels, vs word/trigram BM25
cargo run --manifest-path benchmarks/Cargo.toml --release -- relevance --docs 100000 --queries 5000

# 3. fuzzy: name+edit recall/MRR/nDCG (random 0-2 typos per query), vs FTS5 trigram + LIKE
cargo run --manifest-path benchmarks/Cargo.toml --release -- fuzzy --corpus geonames-cities --queries 5000

# 4. selsweep: the selection-cost frontier (both arms), CSV to stdout
cargo run --manifest-path benchmarks/Cargo.toml --release -- selsweep --corpus geonames-all --docs 125000 > frontier.csv

# 5. the work-done curve: Σ(kept-posting cardinality)
cargo run --manifest-path benchmarks/Cargo.toml --release -- profile --docs 1000000
```

Run `… -- help` for the full option list. Always build `--release` — debug numbers are
meaningless.

### Latency output, `--batched`, `--concurrent`, and JSON

`latency` reports per engine, in serial mode, **p50 / p90 / p95 / p99 / max** latency plus
throughput, and an in-corpus self-recall@k (every engine — the queries are clean, so recall is
well-defined for all). `--batched` times the whole query set as
one `matches_batch` call (shares posting/frequency reads). `--concurrent T` runs trifle across
`T` worker threads, each opening its own pooled `index.reader()` behind a start gate (the
read-pool caller-fanout axis), and reports aggregate throughput plus the per-query p99 across
all workers.

`--format json` (serial mode only) emits one machine-readable object carrying, per engine, the
p50/p90/**p95**/p99/max + mean latency, throughput, recall, **and the raw per-query ns samples**
(so a post-processor can recompute any statistic without re-running). `relevance` and `fuzzy`
also take `--format json` (the full recall/MRR/nDCG + latency record per engine; `fuzzy` reports
the realized edit-mix of its single batch). `--instrument xctrace|samply` re-execs a `latency`
run under a sampling profiler
and writes a trace artifact — a hook for *where* the time goes, separate from the JSON's *how
much*.

### Scaling / frontier sweeps

`selsweep` takes `--docs` as a single `N` **or a comma-separated ladder** swept in one run (each
`N` rebuilds the corpus; all rows land in one CSV, pivoted apart by the `N` column). The canonical
scaling ladder is the **geometric ×5**: `{1000, 5000, 25000, 125000, 625000}`.

```bash
cargo run --manifest-path benchmarks/Cargo.toml --release -- selsweep \
  --corpus geonames-all --docs 1000,5000,25000,125000,625000 --seed 42 > frontier.csv
```

(The old external `for n in …; do … >> frontier.csv; done` loop still works — the plotter skips the
repeated per-run header lines either way.)

`selsweep` CSV (and `--format json`) columns are
`arm,knob,N,k,recall,sigma_df_p50,sigma_df_p99,lat_p50_us,lat_p99_us`, with one row per
`(arm, knob, k)`. `arm` is `t_max` or `df_budget`; `knob` is the swept value (a `t_max` count,
or a `Σdf` budget). The Σdf and latency columns are per-`(arm, knob)` aggregates captured under
the work-done collector, constant across the four `k` rows; `recall` varies by `k`. Plot
`recall@k` against `sigma_df_p50` (and against `lat_p99_us`) with both arms overlaid for the
selection-cost frontier. The same ladder applies to `latency`/`profile` if you want the
flat-latency-as-N-grows confirmation.

`scripts/plot_selsweep.py` draws the frontier straight from that CSV (needs matplotlib). Three
`--mode`s, all keyed off the `N` column so a ladder file plots in one shot:

- **`facet`** (default) — one panel row per `N`, `recall@k` against Σdf and p99 latency, the
  `t_max` and `df_budget` arms overlaid. The better knob is the curve up-and-left.
- **`overlay`** — every `N` on shared axes (color = `N`, linestyle = arm); shows how the frontier
  shifts as the corpus grows.
- **`knee`** — the scaling analysis: per `N`, the cheapest `df_budget` reaching `--knee-frac` (0.98)
  of its own max recall (the knee), then **optimal df_budget vs `N`** with the `budget*/N` fraction
  and a log-log fit. A slope ≈ 1 means the optimal budget is a constant slice of `N`. The per-`N`
  knee table prints to stderr.

```bash
python3 benchmarks/scripts/plot_selsweep.py frontier.csv                  # facet -> selsweep.png
python3 benchmarks/scripts/plot_selsweep.py frontier.csv --mode overlay
python3 benchmarks/scripts/plot_selsweep.py frontier.csv --mode knee      # optimal df_budget vs N
```

## Caveats

These are not your users' queries. Prefer relative signal (trifle vs BM25; typo vs no-typo;
tail vs median) over absolute numbers, and re-run on your own corpus before choosing search
parameters in your application.

- **`relevance` understates recall.** MS MARCO dev qrels are sparse (~1 judged passage per
  query), so set-recall@k against a single label is a narrow slice — read it as "did the one
  known answer land in the top k." Every engine scores against the identical in-corpus qrels and
  identical k (queries with no in-corpus answer are dropped for every engine); the
  `scored-queries` count is the denominator.
- **`fuzzy` does not transfer to prose.** Entity-name fuzzy is a favorable regime (short,
  structured, low-paraphrase); it validates the fuzzy machinery, not retrieval over
  paraphrase-heavy prose, which is `relevance`'s job. Watch the near-distractor density: if low,
  no confusables were sampled and the numbers are inflated.
