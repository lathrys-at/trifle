# Changelog

## 0.3.0

Revision of trifle based on unreleased 0.2.0 draft code. v0.3.0 strips unneeded API complexity and
focuses the crate on the overlap engine and implements performance improvements.

- The `roaring` crate is dropped for `croaring`. Blobs are the standard CRoaring portable
  format.
- The two-level `doc`+`seg` store is now one `seg` table over `(id, key, label, text, len)`,
  with `seg.id` the posting id.
- The pluggable `Ranker`, the over-fetch pool, and the `Effort` knob are removed.
- Per-script-class Welford rarity (multi-script awareness) and the band-spread `WeightStepHint`
  (`Stats.weight_step_hint`) are both retained from v0.2.0.
- Eager `Reader::matches`/`matches_batch` and streaming `Reader::candidates`.
- Opt-in `SqlFilter { fragment, params }` predicate.
- Write API reduced to `upsert(key, &[(label, text)])`, `remove(key)`, and `remove_segment(key, label)`.
- `Index<T: Tokenizer = DefaultTokenizer>` is generic over the tokenizer only.
- `benchmarks/` is reworked against the streaming API.
