# trifle benchmarks (design §10)

The rerunnable harness behind trifle's performance claims. Three evals live here,
because conflating distinct questions measures the wrong thing (§10):

- **`latency` / throughput** — no labels; realistic corpus + query commonness. *How fast?*
- **`relevance`** — MS MARCO **real dev queries + qrels** (§10.4). *How good is recall on
  paraphrased queries, vs BM25?* Real queries share no guaranteed substring with their
  answer, so this is the honest relevance ceiling.
- **`fuzzy`** — **entity name + injected edits** over GeoNames (§10.5). *Does the
  pruner + overlap + tokenizer pipeline survive typos?* — on the task that construction
  faithfully models (you type a corrupted target name, find the target).

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

### The engines, and which eval each baselines

Same task, same corpus, same queries, in-process, so the comparison isolates the
matching strategy from the store — every engine links the *same* bundled SQLite:

| Engine | What it is | Role |
|--------|-----------|------|
| **trifle** | this crate | the subject |
| **fts5-trigram-bm25** | a trigram FTS5 table, `ORDER BY rank` (BM25) | latency cousin (phrase MATCH); **fuzzy** baseline (OR-bag of the query's trigrams — a *fair* fuzzy rival, where a bm25-phrase would score ~0 on typos) |
| **fts5-word-bm25** | a `unicode61` FTS5 table, BM25 | the **canonical BM25** baseline for `relevance` (real BM25 over words) |
| **like-scan** | `LIKE '%…%'` over a plain table | the naive substring floor |

The baseline is **deliberately matched to the eval.** `relevance` (real paraphrased
queries) compares against BM25 — word-level canonical *and* the trigram cousin. `fuzzy`
(typo'd names) compares against FTS5 *fuzzy* (trigram-MATCH) + the LIKE floor — **never
bm25 on a typo eval**, where an exact-term bm25 scores zero by construction and the "win"
is empty. `latency` keeps the phrase-MATCH FTS5 unchanged.

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

| corpus / command | What | Why |
|------------|------|-----|
| `synthetic` *(latency/profile, default)* | real English words (dwyl wordlist) sampled with a **Zipfian** law, 6–20 words/doc | trigram document-frequencies look like real text. A tiny vocabulary would collapse every trigram onto near-every document — a **degenerate dense-posting regime**, the wrong thing. |
| `msmarco` *(latency/profile)* | a deterministic subsample of MS MARCO passages | real prose, real co-occurrence; the strongest single fit for latency. |
| `relevance` *(command)* | MS MARCO passages built **answers + distractors**: every qrel-relevant passage for the sampled real dev queries, plus `--docs` random distractors | guarantees the known answer is indexed, so recall@k measures *ranking it over the distractors* for a real paraphrased query — not whether the answer happened to fall in a random subsample. |
| `geonames-cities` *(fuzzy, default)* / `geonames-all` *(fuzzy)* | GeoNames place names (cities > 15k pop, ~34k; or the full gazetteer, ~12M) | short, structured, low-paraphrase names — the regime where name+edit injection is *faithful*, with natural near-match distractors (many similar names). |

Assets download on demand, hash-verified where the source is immutable, into the
gitignored repo-root `.cache/bench/<corpus>/` (each corpus namespaced so identically
named files never collide). Their bytes are **never committed** — only the manifests in
`sources/`. GeoNames dumps regenerate ~daily, so they are intentionally *unpinned*. See
[`ASSETS.md`](ASSETS.md) for licenses (MS MARCO is non-commercial research only; GeoNames
is CC BY 4.0).

## Queries — matched to the eval (§10.5)

- `latency`/`profile` — **in-corpus document snippets** (2–5 words, 0–2 typos), no
  labels: their vocabulary/co-occurrence are exactly the corpus's, which is all a latency
  measurement needs.
- `relevance` — **real MS MARCO dev queries**, labeled by qrels. These *paraphrase* an
  information need (no guaranteed substring with the answer) — the honest relevance test.
  (Self-derived snippets — the old `quality` eval — were a known-item smoke test against a
  zero-by-construction baseline; that drift is what this harness fixes.)
- `fuzzy` — an **entity name + exactly *k* edits** (the four operations — transposition,
  substitution, deletion, insertion — weighted toward realistic adjacent-key typos),
  labeled by the entity. On *names* this is faithful ("type the target, maybe wrong"),
  reported 1- vs 2-edit, with a trigram-survival column and a near-distractor density so a
  trivially-easy run is visible.

Both recall evals tag each miss **selection / below-floor / ranking** (§4) — whether the
fix lives in the pruner / `m` / `B`, or in the `Ranker`.

## Reproducibility

Everything is driven by a single master **`--seed`**: it seeds corpus sampling *and*
query generation (the two derive independent streams from it), so **same seed → same
run, byte for byte.** Change the seed to resample the corpus and queries together;
keep it fixed to compare a code change against a baseline. The size knobs — `--docs`
(the index size), `--queries`, `--k`, `--repeat` — let you trace the scaling sweep.

## Running it

If your build environment is offline, `fetch` the assets on a connected machine first,
then run against the warm cache. (`relevance` needs the ~1 GiB MS MARCO passage
collection; `fuzzy --corpus geonames-cities` is only a few MB.)

```bash
# 1. warm the cache (downloads + verifies where the source is immutable)
cargo run -p trifle-benchmarks --release -- fetch --corpus relevance       # collection + queries + qrels (~1 GiB)
cargo run -p trifle-benchmarks --release -- fetch --corpus geonames-cities  # ~3 MB

# 2. the footrace: latency + throughput (serial; add --batched, or --concurrent 8)
cargo run -p trifle-benchmarks --release -- latency --docs 100000 --queries 5000 --seed 42

# 3. relevance: set-recall@k on real dev queries+qrels, vs word/trigram BM25
cargo run -p trifle-benchmarks --release -- relevance --docs 100000 --queries 5000

# 4. fuzzy: name+edit recall (1- and 2-edit), vs FTS5 trigram-MATCH + the LIKE floor
cargo run -p trifle-benchmarks --release -- fuzzy --corpus geonames-cities --queries 5000
#    --corpus geonames-all   for the corpus-scale fuzzy run

# 5. the work-done curve: Σ(kept-posting cardinality), the §10.2 flatness instrument
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

Nobody's queries are *your users'* queries. Weight the **relative** signal (trifle vs
BM25; with-typos vs without; tail vs median) over absolute numbers, and re-run on your
own corpus before trusting any of it. Two specifics:

- **`relevance` understates recall.** MS MARCO dev qrels are **sparse** (~1 judged passage
  per query), so set-recall@k against a single label is a narrow slice — read it as "did
  the one known answer land in the top k," not full relevance recall. Both engines score
  against the *identical* in-corpus qrels and *identical* k (the harness drops any query
  with no in-corpus answer for every engine alike); the reported `scored-queries` count is
  the real denominator.
- **`fuzzy` does not transfer to prose.** Entity-name fuzzy is a **favorable** regime
  (short, structured, low-paraphrase). Strong recall there validates the fuzzy *machinery*
  on its home turf; it says nothing about fuzzy retrieval over paraphrase-heavy prose —
  that harder question is `relevance`'s job. Watch the **near-distractor density**: if it
  is low, no confusables were sampled and the numbers are inflated.
