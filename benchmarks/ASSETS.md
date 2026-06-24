# Benchmark corpus attribution

The harness builds its corpora from third-party sources. **None of their bytes are
committed to this repository** — they download on demand into the gitignored
repo-root `.cache/bench/`, hash-verified against the pinned manifests in
[`sources/`](sources/). Only those manifests (url + sha256 + license) are tracked.

| Asset | Used for | License | Pinned in |
|-------|----------|---------|-----------|
| **dwyl/english-words** `words_alpha.txt` | the synthetic-from-wordlist corpus (latency default; a real English vocabulary sampled Zipfian so character-trigram document frequencies look like real text) | The Unlicense (public domain) | [`sources/words_alpha.json`](sources/words_alpha.json) — pinned to commit `8179fe6…`, sha256 verified |
| **MS MARCO passage collection** | the `latency`/`profile` `--corpus msmarco` subsample, and the passage text for the `relevance` eval | MS MARCO Non-Commercial Research License | [`sources/msmarco.json`](sources/msmarco.json) — sha256 pinned (~1 GiB) |
| **MS MARCO dev queries** `queries.dev.tsv` | the real query strings for the `relevance` eval | MS MARCO Non-Commercial Research License | [`sources/msmarco-queries.json`](sources/msmarco-queries.json) — sha256 pinned |
| **MS MARCO dev-small qrels** `qrels.dev.small.tsv` | the relevance judgments (ground truth) for the `relevance` eval | MS MARCO Non-Commercial Research License | [`sources/msmarco-qrels.json`](sources/msmarco-qrels.json) — sha256 pinned |
| **GeoNames** `cities15000` / `allCountries` | the entity corpora for the `fuzzy` (name+edit) eval | CC BY 4.0 | [`sources/geonames-cities15000.json`](sources/geonames-cities15000.json), [`sources/geonames-all.json`](sources/geonames-all.json) — sha256 pinned (frozen snapshot) |

## Sources

- **dwyl/english-words** — a public-domain word list. The harness lowercases, keeps
  ASCII words of length 3–15, and samples them Zipfian. Pinned to an immutable commit
  so the vocabulary (and therefore the trigram-DF distribution) is reproducible.
- **MS MARCO** — non-commercial research license; not redistributable, hence
  fetch-on-demand and never committed. Review the license before use. The three
  artifacts (collection, `queries.dev.tsv`, `qrels.dev.small.tsv`) are immutable: the
  collection, queries, and qrels manifests are all sha256-pinned. The `relevance` eval
  reuses the same cached `collection.tsv` as the `msmarco` latency corpus.
- **GeoNames** — geographical names under **CC BY 4.0** (attribution: © GeoNames,
  <https://www.geonames.org/>). Pinned to a frozen snapshot. The dumps regenerate roughly
  daily, so on an upstream refresh strict verification fails until the manifest sha256 is
  re-pinned (delete the cached zip and re-fetch). The harness reads col 1 (geonameid, the
  doc id) and col 2 (name).

## Offline fallback

If the wordlist is not cached and the network is unavailable, the synthetic corpus falls
back to a small built-in vocabulary (`FALLBACK_VOCAB` in `src/corpus.rs`) and warns on
stderr. Trigram realism is reduced (a small vocabulary makes postings near-dense), so
fallback runs are for smoke-testing only, not publishable numbers.
