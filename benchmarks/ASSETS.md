# Benchmark corpus attribution

The harness builds its corpora from third-party sources. **None of their bytes are
committed to this repository** — they download on demand into the gitignored
repo-root `.cache/bench/`, hash-verified against the pinned manifests in
[`sources/`](sources/). Only those manifests (url + sha256 + license) are tracked.

| Asset | Used for | License | Pinned in |
|-------|----------|---------|-----------|
| **dwyl/english-words** `words_alpha.txt` | the synthetic-from-wordlist corpus (the default; a real English vocabulary sampled with a Zipfian frequency law, so character-trigram document frequencies look like real text) | The Unlicense (public domain) | [`sources/words_alpha.json`](sources/words_alpha.json) — pinned to commit `8179fe6…`, sha256 verified |
| **MS MARCO passage collection** | the `--corpus msmarco` real-document corpus (a deterministic subsample of real passages) | MS MARCO Non-Commercial Research License | [`sources/msmarco.json`](sources/msmarco.json) — sha256 left empty until first fetch |

## Notes on each source

- **dwyl/english-words** — a public-domain word list. The harness lowercases, keeps
  ASCII words of length 3–15, and samples them Zipfian. Pinned to an immutable commit
  so the vocabulary (and therefore the trigram-DF distribution) is reproducible.
- **MS MARCO** — released by Microsoft for **non-commercial research**. Review the
  license before use; it is *not* redistributable here, which is the other reason the
  archive is fetch-on-demand and never committed. The manifest's `sha256` is empty
  until the first fetch on a network machine prints the computed hash to paste back in
  (after which every run verifies against it). The ~1 GiB archive is downloaded and
  `tar`-extracted into the cache; the harness streams `collection.tsv` and subsamples.

## The offline fallback

If the wordlist is not cached and the network is unavailable, the synthetic corpus
falls back to a small built-in vocabulary (`FALLBACK_VOCAB` in `src/corpus.rs`) and
**says so loudly on stderr** — trigram realism is reduced (a small vocabulary makes
postings near-dense), so fallback runs are for smoke-testing the harness, not for
numbers you would publish.
