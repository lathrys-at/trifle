# Critique — Core, round 2 (adversarial cross-review)

Reviewed `proposal-storage.md` and `proposal-filter.md` in full against the source. Below:
my firm position per fork, the new correctness/invariant/perf flaws I found in the other two,
and the places their critique changes my own proposal. I concede F3, F4, F5 outright — each
makes my design strictly better — and push back hard on F1 (Shared) and F6 (headline shape),
where I found concrete flaws.

---

## Per-fork positions

### F1 — attribute storage + Shared. **Position: typed columns (optional) + always-available `key IN rarray`; CUT Shared.** Agree with Filter on columns; reject Storage on both halves.

Storage's philosophy ("trifle is a derived cache, so attributes belong in the caller's source
of truth") is *defused by its own mechanism*: trifle-owned attribute columns are **also
derived** — they ride `Document.attrs`, are rebuilt from the corpus on `rebuild`, and dropped
on drift. Storing them in trifle does not violate the derived-cache model; it *is* the
derived-cache model. So the philosophical objection collapses.

And Storage's `key IN rarray(?)` universal mode has two concrete perf flaws for the headline
case (selective filter, large corpus) that its proposal doesn't price:

1. **The key set is computed over the whole corpus, not the candidate pool.** "All note_ids in
   deck 7" can be 10⁵–10⁶ keys. The caller must enumerate that entire matching *population*
   (a round-trip to a possibly-remote source-of-truth — the exact thing the Sidecar exists to
   avoid), then bind it as a giant `rarray`. Trifle-owned columns make the filter
   `WHERE id IN rarray(candidates) AND deck = ?` — **O(pool)**, never O(matching-population).
2. **That giant rarray is re-bound per bucket.** Storage folds the filter into the per-bucket
   provenance SELECT (its §3.4). So the 10⁵-key `rarray` is rebuilt and re-bound on *every*
   score bucket the walk touches, not once. A trifle-owned `AND deck = ?` binds one integer.

Can both be supported? **Yes, for free** — and this is the synthesis: typed columns are
*optional*; the raw fragment works with or without them. A caller who declares nothing can
still pass `filter = ("key IN rarray(?1)", [keyset])` (Storage's mode, available always). A
caller who declares `attr("deck", Int)` gets trifle-side prune-before-hydrate with no
round-trip. Decl is the opt-in for the fast path; zero-decl always works.

**Cut Shared.** Storage keeps it for the co-located join (`key IN (SELECT … FROM cards …)`).
But the join's *only* unique niche over optional columns is high-churn attributes you don't
want to mirror (e.g. `due <= now()`). That niche costs: the cross-file single-writer
serialization burden Shared documents, **and** it is the sole reason the raw fragment can reach
tables outside trifle (an injection blast-radius the Sidecar-only world doesn't have). For a
"radical simplification" I cut Shared and tell the churn-niche caller to precompute the key set.
Dropping Shared also lets me say the fragment's scope is *trifle's own tables only* — a real
security simplification both Storage and Filter miss.

### F2 — doc→seg flatten. **Position: FLATTEN. Agree with Storage, reject Filter.** Per-segment filtering is not a regression.

Filter keeps two-level so attrs are per-document. But flatten handles the per-document filter
*correctly and more expressively*:

- For per-doc attrs, `upsert(key, segs, attrs)` writes the same attr to all of a key's seg rows
  in one atomic call, so `WHERE deck = 7` over candidate seg-ids yields identical results to a
  per-doc filter — the doc surfaces iff its matching segment carries `deck=7`, which it does.
- Dedup-by-key runs *after* the filter, so a multi-segment doc gets a consistent verdict.
- Per-segment is **strictly more expressive** (filter the OCR layer differently from the field
  layer) — a capability gain. Filter's two-level *cannot* express it.

Worked counter-example proving flatten is more correct: doc with front(deck=7)/back(deck=8),
query matches both, filter `deck=8`. Flatten: front fails, back passes, dedup keeps back —
correct. Two-level (deck per-doc): can't represent it at all.

**Concession to Filter:** denormalized per-seg attrs mean write-amplification (N copies per
doc) and a documented "write attrs consistently across a key's segments for per-doc semantics"
contract. Negligible at trifle's "small documents / few segments" scale, but real — flatten
trades two-level's *structural* per-doc consistency for flexibility + the dissolution of
invariant #2. For a simplification mandate that trade is correct.

### F3 — placeholder binding. **CONCEDE to Filter.** Bind rarray LAST as `?{N+1}`; caller numbers `?1..?N`.

Filter is right and it's a clean win. Anonymous-`?`-renumber-from-`?2` (mine + Storage, the
current audit-F2 behavior) is a type-invisible footgun **and** it forbids positional reuse
(`deck = ?1 OR backup = ?1` is impossible with anonymous `?`). Binding the candidate rarray at
`?{params.len()+1}` lets the caller write exactly the SQL they'd write standalone, with reuse.
No extra `prepare_cached` fragmentation (the fragment text already varies the cache key). I drop
my `?1`-first design.

### F4 — engine lifetime. **CONCEDE my dual-BSI. Adopt: single-BSI `Counter` that OWNS (moves in) the postings + a separate plain `Walk` cursor.** Beats all three prior positions.

The fork note is the key: the stream retains the postings Vec anyway (for `matched_terms`), so
Storage's/Filter's *borrowing* `Counter<'a>`/`OverlapWalk<'a>` makes the Layer-2 stream
self-referential → needs `ouroboros` (forbidden, single dep) or `unsafe`. My dual-BSI dodged
that by not retaining postings — but at ~1.67× plane memory.

The correct fix neither of us proposed: **move the postings into the engine.** `Counter` owns
`Vec<RoaringBitmap>`, builds *one* weighted BSI, computes raw overlap by probing its owned
postings, and exposes `postings()`/`weights()` so the stream serves `matched_terms` from the
Counter (the stream no longer separately retains postings). Single BSI, no self-reference, no
double memory, postings stored exactly once.

The remaining self-reference (a borrowing iterator over `&Counter` stored next to the Counter)
is killed by splitting **immutable engine state** (`Counter`) from a **plain owned cursor**
(`Walk { c, bucket: Vec<u32>, pos }`): the stream owns both, neither borrows the other, and
calls `counter.advance(&mut walk) -> Option<Scored>`. The ergonomic `impl Iterator` wrapper
(for the isolated bench) bundles them via a borrow; the embeddable form keeps them separate.

```rust
pub struct Counter { planes: Vec<RoaringBitmap>, postings: Vec<RoaringBitmap>, weights: Vec<u32>, floor: u32, max_score: u32 }
impl Counter {
    pub fn build(postings: Vec<RoaringBitmap>, weight_step: f64, min_shared: u32) -> Self;
    pub fn advance(&self, w: &mut Walk) -> Option<Scored>;   // &self + &mut cursor: no self-ref
    pub fn postings(&self) -> &[RoaringBitmap];              // serves matched_terms
    pub fn weights(&self) -> &[u32];
    pub fn stream(&self) -> impl Iterator<Item = Scored> + '_;  // bench/ergonomic wrapper
}
pub struct Walk { c: u32, bucket: Vec<u32>, pos: usize }    // 'static, owned by the consumer
```

**Memory, quantified.** Weighted planes ≈ `ceil(log2(Σweights))` ≤ ~6 (k≤12 ⇒ max_score≤48);
dual-BSI adds an unweighted stack of ≤ ~4 → 10 vs 6 planes (~1.67×). Each plane is a roaring
bitmap over the candidate union U; for a broad query U can approach corpus size (dense U of 1M
≈ ~128 KB/plane), so dual-BSI costs ~500 KB extra *per concurrent query*. Owning-postings
single-BSI avoids that **and** the separate postings retention. This is my biggest revision and
the cross-review earned it.

### F5 — hydration shape. **CONCEDE to Storage.** Provenance-only `Candidate` + explicit batched `hydrate(&[Candidate]) -> Vec<Match>`.

My per-chunk eager `Match` over-hydrates text for the rerank case (pull 1000, rerank on
score/overlap, keep 10 → I'd read 1000 `txt` blobs). Storage's split is strictly better:
`Candidate` carries provenance + score + overlap (one batched provenance read per bucket, no
text); `hydrate(&kept)` does one batched `txt` read for exactly the kept set → pull-1000-keep-10
reads 10 blobs. Filter's per-item lazy `.text()` is worse (per-item SQL, or it secretly
re-batches → back to eager). I adopt Storage's shape. Provenance per bucket is unavoidable
(needed for key-dedup and the filter); text is the deferrable part, and Storage defers exactly
it.

### F6 — snapshot safety + headline. **Position: EAGER `search()` is the HEADLINE; streaming is the advanced secondary. Reject Storage's `search → stream` flip.**

Storage makes the snapshot-pinning, `Result`-per-item stream the headline `search()`. That is
the wrong default on safety grounds, and Storage itself names the worse of the two footguns:
**silent truncation.** `Iterator<Item = Result<Candidate>>` consumed via the idiomatic
`.filter_map(Result::ok)` will, on a mid-stream transient `Busy`, *silently drop the erroring
item and every item after it* — the caller believes they received the full result set. A
transient lock turning into "fewer results, no error" is insidious.

My earlier pool-poisoning worry is the *lesser* risk and I downgrade it: Rust runs `Drop` on
panic-unwind, so the rollback fires; only `mem::forget`/abort bypass it, and even then a pooled
conn left mid-tx fails its *next* `BEGIN` with an error (not corruption). Defensive
`ROLLBACK`-on-checkout closes it entirely.

So: **headline = eager `search(q,opts) -> Result<Vec<Match>>` / `search_batch(...)`** — it
propagates errors correctly, drains the snapshot immediately (no WAL-checkpoint pinning), and is
what the median caller wants. The streaming `candidates()` is the opt-in power surface (rerank,
fusion, pagination) with a loud "handle every `Result`; drain promptly" contract and a
`collect_matches()` terminal that propagates the first error instead of swallowing it. This is
also where I align with Filter (eager headline, stream secondary) against Storage.

### F7 — the 2–3× target. **Position: do NOT promise 2–3×. Promise the restructure (~1.4× whole / ~1.8× control-plane). Hold the tokenizer trim as a benchmark-gated, reversible lever.**

All three of us land ~1.3–1.55× whole-crate honestly, ~1.75–1.85× on the control plane, and
agree 2–3× needs cutting into `tokenize.rs`. The tokenizer trim (drop multi-script
`DefaultTokenizer` / NFD / accent-strip / alt n-gram sizes) is a **recall-affecting feature
cut**, not a simplification, and the user's own `tmax-pool-sweep-methodology` memory demands
empirical gating. Recommend telling the user: ship the 3-layer restructure (the real wins —
engine isolation, streaming, filter collapse, welford/Effort/Ranker/telemetry deletion) for
~1.8× on the targeted logic; then run the fuzzy-recall eval (geonames-cities + a mixed-script
set) and decide the tokenizer trim *against data*. Reaching 2× by trimming normalization before
measuring recall would be trading the brief's "do not break capability" for a headline number.

---

## New correctness / invariant / perf flaws found

**In Storage:**

1. **Lease collapse is a concurrency regression (§3.1).** Today `Reader` checks out a *fresh
   pooled connection per search*, so one `Reader` fires many concurrent searches. Storage's
   merged `Reader` "holds a warm pooled connection for its lifetime" with "one live
   `CandidateStream` per reader at a time" — so a single reader now **serializes** its searches,
   and the convenience `matches()` (which opens a tx on that warm conn) **conflicts with a live
   stream on the same reader** ("cannot start a transaction within a transaction"). The warm
   conn belongs on the *stream*, not the reader: keep `Reader` as a per-search/per-stream
   checkout so concurrent streams each hold their own pooled conn. Storage collapsed the two
   leases the wrong way.
2. **Universal-mode filter re-binds a corpus-sized key set per bucket** (F1, flaw #2) — not a
   one-time cost as its §4 implies.
3. **`key IN rarray(precomputed)` forces a source-of-truth round-trip** precisely when the data
   is remote/expensive — the scenario the Sidecar exists for (F1, flaw #1).

**In Filter:**

4. **`Candidate` API is self-contradictory.** §3 declares `pub text: String` ("empty until
   hydrated") while §6/§4 describe lazy hydration via a `.text()` method. A public field and a
   lazy accessor are mutually exclusive: a caller reading `cand.text` gets `""` with no
   interception, silently. Storage's provenance-only `Candidate` + explicit `hydrate()` is the
   correct shape (another reason I concede F5 toward Storage, not Filter).
5. **Keeping two-level adds a `seg ⋈ doc` join to every per-bucket provenance read** on the hot
   path (its §4.5 filter folds into a SELECT that must join to reach `doc` columns and the key),
   where flatten makes it a single-table point read. A self-inflicted perf cost of the F2
   choice.
6. **`matched_terms`/`present_terms` signals are dropped.** Filter's `Candidate` exposes
   `score`/`overlap`/`text` but not the matched-token+df signals the old `Candidate` gave a
   custom ranker. With the `Ranker` trait gone, a reranking caller that needs per-matched-token
   df (for a custom rarity score) has no path. Storage preserves these on the stream
   (`matched_terms`, `n_segments`, `avgdl`); I should too. (Minor; a signal-completeness gap.)

---

## Concessions that change my proposal

- **F3:** bind candidate rarray last as `?{N+1}`; drop my anonymous-`?` renumber.
- **F4:** drop dual-BSI; adopt single-BSI `Counter` that **owns** the postings + a separate
  `Walk` cursor (no self-ref, no 2× memory). My strongest revision.
- **F5:** stream yields provenance-only `Candidate`; add explicit batched
  `hydrate(&[Candidate]) -> Vec<Match>`; eager `search()` does both for the top-`limit`.
- **F6:** make eager `search()` the headline; streaming `candidates()` secondary with a
  `collect_matches()` error-propagating terminal.
- **F1 (partial):** add the zero-declaration `key IN rarray` mode alongside optional typed
  columns (toward Storage's philosophy) — but still cut Shared and keep typed columns as the
  fast path.
- **Signal completeness (from Filter flaw #6 / Storage's stream):** expose
  `matched_terms`/`n_segments`/`avgdl` on the candidate stream so no custom-rank signal is lost.
