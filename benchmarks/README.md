# trifle benchmarks (design §10)

The rerunnable harness behind trifle's performance claims. Two distinct benchmarks
live here, because conflating them measures the wrong thing (§10):

- **latency / throughput** — no labels needed; realistic corpus + realistic query
  commonness. *How fast?*
- **quality / recall** — needs relevance judgments. *How good, vs BM25?*

A speed superlative is **earned, not asserted**: the README's headline claim ships
only once this harness backs it, with a link here.

> **Not published.** `trifle-benchmarks` is a workspace member with `publish = false`.
> It is excluded from the `trifle` crate package, so it adds **no dependency** to what
> downstream users compile.

## What it measures, and against what

trifle is a *lexical* engine; the honest claim is "BM25-ish lexical recall,
typo-tolerant, much faster" — and ownership of an underserved cell: **durable +
embedded + incrementally-updatable + corpus-scale fuzzy, with provenance, fast enough
to feel instant.** So the comparison is a footrace *and* a matrix.

### The footrace (`latency`, `quality`)

Same task, same corpus, same queries, in-process, so the comparison isolates the
matching strategy from the store — all three link the *same* bundled SQLite:

| Engine | What it is | Role |
|--------|-----------|------|
| **trifle** | this crate | the subject |
| **fts5-trigram-bm25** | a trigram FTS5 table, `ORDER BY rank` (BM25) | the in-DB cousin; the **quality baseline** |
| **like-scan** | `LIKE '%…%'` over a plain table | the naive substring floor |

### The matrix (read this before reading the race)

A latency table omits the **hidden axes**, and footracing the wrong category is a
category error. Fill every row honestly — *including the cells where trifle loses*:

| | durable? | embedded / no server? | incremental update vs rebuild? | corpus-scale (100k+ small docs)? | provenance? | matching semantics | footprint |
|---|---|---|---|---|---|---|---|
| **trifle** | ✅ disk (SQLite) | ✅ | ✅ incremental (base+delta) | ✅ | ✅ (source/ref) | trigram overlap | disk |
| FTS5-trigram | ✅ | ✅ | ✅ incremental | ✅ | rowid only | trigram + BM25 | disk |
| pg_trgm | ✅ | ❌ server | ✅ | ✅ | table cols | trigram similarity | disk |
| Tantivy + Levenshtein | ✅ | ✅ | ✅ (segments) | ✅ | fields | Levenshtein automaton | disk |
| fzf / nucleo / fuzzy-matcher | ❌ | ✅ | rebuild-on-startup | ⚠️ RAM-bound | — | subsequence | RAM |
| fst / SymSpell / strsim | ⚠️ immutable | ✅ | rebuild-to-update | ✅ | key-oriented | delete-neighborhood / edit-distance | RAM/disk |

The in-memory subsequence filters **will out-latency trifle on their turf** (RAM-resident,
rebuild-on-startup). That is the point of the matrix: the real claim is the *cell*, not
a raw-latency superlative. The out-of-process engines (pg_trgm, Tantivy, fzf, fst, …)
are not wired into this harness — run them in their own drivers on the same corpus and
queries, and fill the table.

## Corpora — and why no single one suffices (§10.4)

| `--corpus` | What | Why |
|------------|------|-----|
| `synthetic` *(default)* | real English words (dwyl wordlist) sampled with a **Zipfian** frequency law, 6–20 words/doc | trigram document-frequencies look like real text — common trigrams in many docs, rare ones in few. A tiny vocabulary collapses every trigram onto near-every document and measures a **degenerate dense-posting regime** — the wrong thing. |
| `msmarco` | a deterministic subsample of MS MARCO passages | real prose, real vocabulary and co-occurrence; the strongest single fit for latency *and* the BM25 quality baseline. |

Both download on demand, hash-verified, into the gitignored repo-root `.cache/bench/`.
Their bytes are **never committed** — only the pinned manifests in `sources/`. See
[`ASSETS.md`](ASSETS.md) for licenses (MS MARCO is non-commercial research only).

## Queries — in-corpus snippets ± typos (§10.5)

The realistic query is an **in-corpus document snippet**, optionally with injected
single-character typos — which sidesteps "where do realistic queries come from": the
snippet's vocabulary and co-occurrence are exactly the corpus's, and the source
document is a **free ground-truth label**.

- `latency`/`profile` — snippets of 2–5 words, 0–2 typos, no labels needed.
- `quality` — snippets of 3–6 words with exactly *k* edits, labeled by source doc;
  edits are the four operations (transposition, substitution, deletion, insertion),
  weighted toward realistic typos (adjacent-key, transpositions).

## Reproducibility

Everything is driven by a single master **`--seed`**: it seeds corpus sampling *and*
query generation (the two derive independent streams from it), so **same seed → same
run, byte for byte.** Change the seed to resample the corpus and queries together;
keep it fixed to compare a code change against a baseline. The size knobs — `--docs`
(the index size), `--queries`, `--k`, `--repeat` — let you trace the scaling sweep.

## Running it

The build machine here has **no network**; fetch on a connected machine first, then
run offline against the warm cache.

```bash
# 1. (network machine) warm the cache — downloads + hash-verifies the corpus assets
cargo run -p trifle-benchmarks --release -- fetch --corpus synthetic
cargo run -p trifle-benchmarks --release -- fetch --corpus msmarco   # ~1 GiB, non-commercial license

# 2. the footrace: latency + throughput, serial
cargo run -p trifle-benchmarks --release -- latency --docs 100000 --queries 5000 --seed 42

# the batched axis (one search_batch call shares posting/frequency reads)
cargo run -p trifle-benchmarks --release -- latency --docs 100000 --batched

# the read-pool parallelism axis (trifle only)
cargo run -p trifle-benchmarks --release -- latency --docs 500000 --concurrent 8

# 3. quality: recall@k vs BM25, swept over {0,1,2} edits
cargo run -p trifle-benchmarks --release -- quality --corpus msmarco --docs 100000

# 4. the work-done curve: Σ(kept-posting cardinality), the §10.2 flatness instrument
cargo run -p trifle-benchmarks --release -- profile --docs 1000000
```

Run `… -- help` for the full option list. **Always build `--release`** — debug numbers
are meaningless.

### The scaling sweep is the architectural claim (§10.2)

trifle's central promise is **flatness**: bit-sliced overlap is posting-size-independent
and DF reads are PK seeks, so latency should stay near-flat as the corpus grows. That is
a *curve* — sweep it:

```bash
for n in 10000 50000 100000 500000 1000000; do
  cargo run -p trifle-benchmarks --release -- latency --docs "$n" --seed 42
done
```

A flat p50/p99 curve earns the claim; a degrading one says an assumption broke. The
`profile` command tags each query with **Σ(kept-posting cardinality)** — the quantity
whose growth *would* break flatness. Correlate its p99 with the latency p99: if the tail
tracks this curve, the residual is big-bitset AND/XOR cost (a small constant, expected);
if not, look at hydration or the predicate.

## Caveats (§10.6)

Nobody's queries are *your users'* queries. MS MARCO's distribution is a proxy for
"natural search"; typo injection a proxy for "autocomplete". Weight the **relative**
signal (trifle vs BM25; with-typos vs without; tail vs median) over absolute numbers,
and re-run on your own corpus before trusting any of it.
