# trifle — next tasks & review-cycle playbook

Handoff notes for the work queued after the v0.2 rework (branch `feat/rev-v0.2`). Part 1 is
the task backlog — each item is unblocked and implementable today, with a concrete approach,
the files it touches, and a done-when. Part 2 is the full recipe for running an adversarial
review cycle (the process that produced this list).

Line numbers are **as of commit `f97a774`** and will drift — grep the named function if they
don't match. The design rationale for anything below lives in the module `//!` docs and in the
design spec at `~/Desktop/trifle-design-addendum.md` (+ `trifle-filter-grammar-notes.md`);
section numbers like “§7.5” refer to the addendum.

## Status — completed in the follow-up cycle (branch `feat/rev-v0.2`)

Eight tasks landed after the doc was written, each its own commit with a regression test and
the full gate green (fmt, `cargo test`, clippy `-D warnings`, rustdoc `-D warnings`, workspace;
`cargo-deny` not run locally but no deps changed). **Not yet adversarially reviewed** — that's
the next step.

| Task | Commit | Summary |
|------|--------|---------|
| **T1** | `c81c2b3` | rebuild folds filterable columns into the `doc` INSERT (one write/doc) |
| **T15** | `7e0d73a` | band-spread histogram reset on rebuild / drift reset (not compact) |
| **T5** | `c270db9` | Tier-2 filter applied scoped to candidate ids (last O(N)/search path gone) |
| **T3** | `5bcb0b1` | `set_fields` requires an existing document (option **a**; no ghost rows) |
| **T4** | `5051ece` | CLAUDE.md error-taxonomy synced to four classes |
| **T7** | `0a2deee` | query pipeline extracted to `src/search.rs` behind `SearchCtx`; allows removed |
| **T2** | `feeddda` | read path resolves in term-space (no Token→String→u128→String round-trip) |
| **T6** | `852c2cb` | reader retry across a rebuild reload is time-bounded, not a fixed count |

Still open: **T8–T12** (larger features) and **T13–T14** (ranking benchmark / autotune). One
`#[allow(clippy::too_many_arguments)]` remains on the write-path `write_segment` — deferred to
T7's optional step 4 ("move write helpers onto `Writer`").

## What changed since this doc was first written

A few things landed after the initial backlog, so read the tasks with this context:

- **Ranking is now IDF-weighted lexical overlap, not BM25.** The BM25+ reranker (`Bm25Ranker`)
  and the `rank::idf` helper are **gone**; trifle ranks by rarity-weighted token overlap
  computed in the bit-sliced counter itself (per-query, df-anchored 4-tier `{1,2,3,4}`; knob
  `SearchOpts::weight_step` `D`, default `1.0`). The default `Ranker` is `OverlapRanker`
  (preserves that order); a custom `Ranker` is the only way to rerank, over an `Effort`
  over-fetch pool whose **default is now `Effort::None`** (the weighted order is exact at
  `pool = limit`). `seg.len`/`seg_len_sum`/`avgdl` are still stored but are now *extension
  signals* (for a custom ranker), unused by the default ranking. New telemetry:
  `Stats::weight_step_hint` suggests a corpus-fitted `D` (median band-spread / 3, with an IQR
  confidence signal) from an in-memory per-query band-spread histogram. This redesign is the
  origin of the new tasks **T13–T15** below.
- **The three deferred `Writer` minors are done** (was a "minors" note): the dead
  `Writer::dirty` field is removed; `commit()` now distinguishes a lost batch (retry) from a
  durable-but-stranded writer via the new **`Error::WriterStranded`** (which the `atomic()`
  poison path also returns instead of `Corrupt`); and `atomic()` escalates a faulted savepoint
  rollback to a full `ROLLBACK` + poison. So `Error` now has **four** caller-facing classes,
  not three.
- **The big deferred items (I10/I12/I-N1/I22 → T1/T2/T5/T7, plus C2-FB-C2 → T6) are now done**
  — see the status table above. Line numbers in the task entries below are as of `f97a774` and
  have drifted; grep the named function.
- **Not yet adversarially reviewed:** the ranking redesign (the weighted counter, the
  band-spread hint, the `Effort`-default change) is tested and green but has **not** been
  through a review cycle. Point reviewers there first next cycle.

---

## Part 1 — task backlog

Ordered roughly by value-to-effort. Each task is self-contained; do them in any order unless a
dependency is noted. Every task is **done** only when the full gate (Part 2 §6) is green and
its named test exists.

### Small, self-contained changes

#### ✅ T1 (DONE) — Fold filterable columns into the rebuild doc INSERT (was I-N1)
- **Why.** `rebuild()` writes each document's `doc` row, then issues a *separate* `UPDATE` per
  document that has payload. For a payload-bearing corpus that doubles the per-doc write count
  and re-compiles SQL per row — O(docs) extra writes on the heaviest path.
- **Where.** `src/lib.rs` `rebuild()` ~`:980` (`doc_ins.execute(...)`) then `:982`
  (`self.set_doc_fields(&tx, ns.doc_shadow(), ...)`); the shadow `doc` column list is built in
  `src/schema.rs` (`doc_filt_cols`).
- **Approach.** Build the shadow `doc` INSERT with the filterable columns already in its column
  list and bind their values inline (the shadow schema is known at rebuild time). One cached
  prepared statement, one write per document. Drop the per-doc `set_doc_fields` call in the
  rebuild loop. Keep the incremental path's `set_doc_fields` as-is.
- **Validation.** Existing `tests/cycle2.rs` / `tests/lifecycle.rs` rebuild-with-payload cases
  already assert correctness; add an assertion that a rebuilt payload doc is filterable. No
  behavior change — purely fewer statements.
- **Done-when.** Rebuild of a payload corpus issues one write per doc; gate green.

#### ✅ T2 (DONE) — Resolve the read path in term-space, not string-space (was I10)
- **Why.** The query path round-trips `Token → String → u128 → String`: `distinct_tokens`
  stringifies each query gram, then `resolve_batch` re-encodes the string back to the `u128`
  term key and stringifies again for the map key. The write path already interns straight from
  `token.term()`; the read path abandons that win. Per-query constant (bounded by query length,
  not corpus size), so it’s a cleanliness/latency win, not a scaling fix.
- **Where.** `src/lib.rs` `distinct_tokens` `:903`; `src/dict.rs` `resolve_batch` `:138`,
  `resolve_key` `:125`.
- **Approach.** Thread `Term`/`TermId` through selection instead of `String`. Resolve each
  distinct query token to `(TermId, class)` **once** via `token.term()`, keep the term-keyed
  ids through `read_dfs`/`effective_postings`, and stringify only the selected tokens that
  actually reach `QueryContext::selected` / `Candidate::matched_terms`. Preserve the
  cross-cutting decision that `select.rs`/`rank.rs` tie-breaks remain deterministic.
- **Watch.** `batch == serial` (`tests/scope_ranker.rs`) and the selection tie-break must not
  change. This is the invariant most at risk from re-keying selection.
- **Done-when.** No `to_string()` on the hot read path except for the tokens surfaced to the
  ranker; `scope_ranker.rs` + `thrash.rs` green.

#### ✅ T3 (DONE) — Decide the `set_fields`-on-empty-key contract (was A4 ghost doc)
- **Why.** `set_fields(key, …)` on a key with no segments creates a payload-only `doc` row that
  no search can ever return (search needs segments) — an invisible “ghost”. Not a correctness
  leak (the C2-RA-1 fix reaps *delete*-orphaned rows; this is the *create* path), but a
  footgun: a typo'd key silently accretes rows.
- **Where.** `src/lib.rs` `set_fields` `:1715` → `doc_id_for(..., create=true)` `:602`.
- **Approach.** Pick one and document it: **(a)** make `set_fields` require an existing
  document (`create=false`; return `Error::InvalidInput` if the key has no segments) — the
  stricter, less surprising option; or **(b)** keep create-on-set but document the payload-only
  state explicitly and add a `stats()` counter for segment-less docs so it's observable. Prefer
  (a) unless a caller genuinely needs to stage payload before segments.
- **Validation.** `tests/cycle2.rs` or `tests/api.rs`: a `set_fields` on a fresh key either
  errors (a) or creates an observable-but-unsearchable row (b); and confirm it does **not**
  interact with the C2-RA-1 reaping path.
- **Done-when.** Behavior is intentional, tested, and documented on `set_fields`.

#### ✅ T4 (DONE) — Sync `CLAUDE.md`'s error-taxonomy line
- **Why.** `CLAUDE.md` says `src/error.rs` “variants separate the **three** failure classes”;
  there are now four (`Error::WriterStranded` was added — store-fine/handle-dead). The
  `error.rs` module rustdoc is already correct; only `CLAUDE.md` drifted.
- **Where.** `CLAUDE.md`, the `src/error.rs` bullet under “Cross-cutting”.
- **Approach.** One-line edit: “…separate the failure classes a caller handles differently
  (transient store fault; fixable caller input; impossible internal-invariant violation; a
  stranded writer handle).” *(CLAUDE.md is project-governance — confirm with the owner before
  editing if you're an agent.)*
- **Done-when.** `CLAUDE.md` matches `error.rs`.

### Performance / robustness

#### ✅ T5 (DONE) — Scope the Tier-2 filter to candidate ids (was I12)
- **Why.** A broad structured filter (e.g. `lang = 'en'` over most of a 1M-doc corpus)
  materializes a `RoaringBitmap` of **every** matching `doc` id on **every** search — an O(N)
  row scan + bitmap build, even though only the small candidate pool is ever consulted. It is
  the only O(N)-per-search path left, and it fires precisely when the filter is non-selective.
- **Where.** `src/lib.rs` `filter_docs` `:884`, called once per batch at `:1255`; the keep-set
  is consulted in `src/rank.rs` `overlap_search` (`filter_docs.contains(*doc)`).
- **Approach.** Don't materialize the whole matching universe. After candidate generation,
  intersect against the filter scoped to the candidate ids:
  `SELECT id FROM doc WHERE id IN rarray(?candidates) AND <filter>`. This bounds the work by
  pool size, not corpus size, and **preserves early-stop + `batch == serial`** (the candidate
  set is per-query). Use `prepare_cached`. Note the per-query-vs-per-batch nuance: if a batch
  shares many candidates, a single scoped query over the union is also valid.
- **Validation.** `tests/` (filter cases) must still pass; add a test that a non-selective
  filter returns the same results as today (correctness is unchanged — this is a cost fix).
  Optionally add a `benchmarks/` latency case with a broad filter to show the flattening.
- **Watch.** Early-stop must still fill `limit` with passing docs; `scope_ranker.rs` parity.
- **Done-when.** No O(N) `SELECT id FROM doc WHERE <filter>` on the search path; results
  identical; gate green.

#### ✅ T6 (DONE) — Coordinate readers with an in-progress rebuild reload (was C2-FB-C2)
- **Why.** During a concurrent `rebuild()`, a reader tolerates dictionary-generation skew for a
  **fixed** budget (`RETRY_MAX = 5`, ~150 ms), but the skew window equals the whole
  `dict.load` table scan, which **grows with vocabulary**. At a large-enough vocab, normal
  concurrent searches spuriously fail with `Error::Busy` on every rebuild. (Could not be
  triggered ≤ 8k docs / ~50k grams; bites only at the top of the documented multilingual
  scale.) Distinct from the now-fixed permanent-skew case (C2-FB-C1 poison).
- **Where.** `src/lib.rs` retry loop `search_read` `:1414`, `RETRY_MAX` `:131`, skew message
  `:1468`; `src/dict.rs` `load` scan.
- **Approach.** Replace the fixed retry count with coordination. Options, cheapest first:
  **(a)** bound retries by elapsed time with backoff rather than a fixed count, so a slow
  reload still settles; **(b)** have the reader briefly wait on a reload-in-progress signal
  (e.g. an `RwLock`/`Condvar` the writer holds across the swap+reload) instead of spinning a
  budget; **(c)** make `dict.load` swap an `Arc<DictInner>` so readers see old-or-new
  atomically with no skew window (the cleanest, but a larger change to `Dictionary`). Pick (a)
  for a quick mitigation; (c) is the durable fix and dovetails with T2.
- **Validation.** Extend the concurrency probe (the team's `zz_cycle2_concurrency` pattern: N
  readers vs repeated `rebuild()`, assert every successful search returns the queried doc else
  a retryable error, count busy events) with a large vocabulary, and assert the busy count
  stays at/near zero. Keep it `#[ignore]` if it's slow.
- **Done-when.** Concurrent search under rebuild churn does not spuriously `Busy` at the
  documented scale; coherence probe green.

### Architecture

#### ✅ T7 (DONE) — Extract the search pipeline; give the leases real bodies (was I22)
- **Why.** `src/lib.rs` is ~2.0k lines and the read/write logic lives as inherent `Index`
  methods while `Writer`/`Reader`/`SearchSession` are thin shells — the lease types don't own
  the work the §8 model says they should, and the search pipeline is hand-wired across several
  `#[allow(clippy::too_many_arguments)]` functions. (The ranking redesign added more to
  `run_search`: it now also threads `weight_step` and updates the band-spread histogram.)
- **Where.** `src/lib.rs` `run_search` `:1179`, `rank_to_matches` `:1299`, `hydrate_text`
  `:1339`, `search_read`/`search_read_on` `:1414`/`:1427`; the `#[allow(clippy::too_many_arguments)]`
  sites in `src/lib.rs` plus `overlap_search` in `src/rank.rs`.
- **Approach (incremental, keep each step green).** (1) Introduce a `SearchCtx` struct that
  bundles the args currently threaded through the `too_many_arguments` functions (conn, ns,
  schema, opts incl. `weight_step`, dict snapshot), removing the allows. (2) Move
  `run_search`/`rank_to_matches`/`hydrate_text`/`search_read*` into a new `src/search.rs`
  module operating on `SearchCtx`.
  (3) Have `Reader`/`SearchSession` call into `search.rs` (collapsing the verbatim glue between
  `Reader::search` and `SearchSession::search`). Optionally (4) move the write helpers onto
  `Writer`. Do **not** change behavior — this is a pure refactor; the test suite is the
  safety net.
- **Validation.** No new tests required; the whole existing suite (esp. `thrash.rs`,
  `scope_ranker.rs`, `api.rs`) must stay green at every commit. Clippy with the
  `too_many_arguments` allows **removed**.
- **Done-when.** `lib.rs` is the lifecycle + types; the pipeline lives in `search.rs`; no
  `too_many_arguments` allows; gate green.

### Ranking follow-ups (from the IDF-weighting redesign)

#### T13 — Benchmark whether the IDF weighting earns its keep
- **Why.** The weighting adds ~2 bit-planes and widens the bucket range to `0..4k` (sparser,
  more buckets — the sparse-result case walks more empty buckets). The design's own test is:
  measure **buckets walked to a stable top-k, weighted vs unweighted** — “if it doesn't move,
  the weighting isn't earning its 2 planes.” This was deferred past the build (the user asked
  to ship it first), so it's now a *measure-after* task.
- **Where.** `benchmarks/` (`latency`, `profile`, `relevance`); the bucket walk in
  `src/rank.rs` `overlap_search`. Unweighting for the A/B is free: `SearchOpts::weight_step`
  with a huge `D` collapses every gram to tier 1.
- **Approach.** Add benchmark cases that report, weighted vs unweighted: (1) buckets cracked to
  a stable top-k; (2) p99 latency delta (the ≤2 ripples + extra planes); (3) recall@k vs the
  BM25 baseline on MS MARCO (the `relevance` eval now compares *weighted overlap* to BM25, a
  different question than before). Also plot `log2(df_max/df_i)` across real queries (the spec's
  "measure the band-spread distribution"). If recall/precision doesn't move, reconsider the
  default.
- **Done-when.** A benchmark quantifies the weighting's recall benefit and its latency/plane
  cost, and the no-payoff case is visible in the numbers.

#### T14 — Autotune `D` from the band-spread hint
- **Why.** `Stats::weight_step_hint` *suggests* a `D` (median band-spread / 3, with an IQR
  confidence signal), but the caller must read it and feed it back via `SearchOpts::weight_step`
  manually. Closing that loop — and sharpening the estimate — is the natural next step (the user
  flagged "autotuning D is a different task").
- **Where.** `src/lib.rs` `weight_step_hint()` (the histogram + nearest-rank quantile) and
  `SearchOpts::weight_step`; the histogram is `Index::band_spread_hist`.
- **Approach.** (a) Optionally let the index apply the suggested `D` when the caller doesn't set
  one (a config flag), guarded by a warmup-sample minimum and by the IQR (if the spread of
  spreads is wide/bimodal, *don't* auto-apply a single `D` — the corpus has multiple query
  regimes; surface that instead). (b) Sharpen the estimate beyond the current 0.5-doubling
  bucket midpoints — finer buckets or a streaming-quantile sketch (P² / t-digest). Note the
  histogram is process-local and warms from zero on reopen.
- **Done-when.** A caller can opt into a corpus-fitted `D` without manually plumbing `stats()`
  back into `SearchOpts`, and the multi-regime case is handled honestly (no false single-`D`).

#### ✅ T15 (DONE) — Reset the band-spread histogram on rebuild / drift reset
- **Why.** The weight-step hint accumulates over the index's whole lifetime, but `rebuild()` and
  a `data_version`/schema drift reset can change the df distribution substantially — pre-change
  band-spread samples then bias the suggested `D`.
- **Where.** `src/lib.rs` `rebuild()` (after the swap) and the drift-reset path in `init()`;
  `Index::band_spread_hist`.
- **Approach.** Zero the 13 histogram atomics on a successful `rebuild` and on a drift reset, so
  the hint reflects only the current corpus. Cheap (13 stores). `compact()` must **not** reset
  it (a fold doesn't change df). Tiny test: hint is `None` again immediately after a rebuild.
- **Done-when.** After `rebuild`/reset, `weight_step_hint` reflects only post-change searches.

### Larger features (design exists in the addendum)

These are unblocked — the design is written — but each is a substantial change. Sequence after
the above unless a consumer needs one sooner.

#### T8 — Parallel rebuild (§2)
- `rebuild()` is single-threaded. The addendum’s design: id-disjoint contiguous shards →
  partition-by-term flat merge → single-threaded durable write. Big throughput win on the
  heaviest op. Keep the atomic shadow-swap + post-commit reload (and the C2-FB-C1 poison)
  intact. Validate with `thrash.rs` + a byte-identical-rebuild assertion (already in
  `tests/lifecycle.rs`).

#### T9 — Tier-1 partition (§7.5)
- Partition-keyed `(partition_id, term_id)` postings + per-partition DF/Welford; the one
  hot-path-invasive filtering tier. `src/partition.rs` (parallel to the gram dict, same fault
  split), `pkey(pid,tid) = (pid as u64)<<32 | tid`, `reader_scoped`/`writer_scoped` leases.
  This is the largest item; read §7.5 in full first.

#### T10 — Async acquisition adapter (§8 option 3)
- Keep the sync core; add async **acquisition** of the same lease handles (the `from_guard`
  ctor seam already exists). Do not re-color the synchronous API.

#### T11 — Search-warming caches, layers 1–2 (§3)
- `SearchSession` holds a warm connection but no posting/DF cache. Add the Layer-1
  posting/DF cache keyed by `(partition, term_id)`, flushed when the reader's snapshot
  `data_version` changes. Layer-2 (incremental count vector) is a further step. The seams are
  noted on `SearchSession`.

#### T12 — Mixed-script recall eval (§6)
- The class-normalized (Welford z-score) pruner ships guarded but unvalidated against raw DF on
  a mixed-script corpus. Add a mixed-script eval asset under `benchmarks/` and an A/B
  (the config switch to force raw-DF exists) to confirm the z-score helps. Belongs in the
  `benchmarks` crate, not the library tests.

---

## Part 2 — how to run a review cycle

trifle is hardened by **adversarial team review**: parallel reviewer agents try to break the
code, then a two-step validation gate kills everything that can't survive a peer's refutation
plus a red test. Run a cycle whenever a meaningful change lands and before a release. Cycles
repeat until **quiescent** (a cycle that finds no new correctness work).

### 0. Prerequisites
- A clean working tree on the branch under review; `gh`/git available.
- The `team-review` skill (it carries the portable method; this section is the trifle-specific
  operating procedure).
- The design spec to hand reviewers: `~/Desktop/trifle-design-addendum.md` and
  `~/Desktop/trifle-filter-grammar-notes.md`.

### 1. Pre-flight (before spawning anything)
1. **Pin the commit.** Record the exact SHA; every reviewer reviews *that* SHA and every repro
   must fail/pass against it. `git rev-parse HEAD`.
2. **Green baseline.** Run the full gate (§6) and record the result. A repro is only meaningful
   if the suite was green first. Note anything you did *not* run (e.g. `cargo-deny` if not
   installed; networked benchmark corpora).
3. **Read the spec yourself** and note the invariants reviewers judge against: `batch == serial`;
   scope is candidates-only + early-stop-bounded; monotonic-id no-false-positive; rebuild is
   atomic + reclaims ids + result-stable; drift drops-not-migrates; compact clears-backlog +
   result-invariant; span ⇒ text + char-boundary; the dictionary generation guard (H1 benign /
   H2 retryable); seg-stats counters reconcile with the `seg` table. **Ranking** is now
   IDF-weighted lexical overlap (no BM25): weights are a per-query df-anchored 4-tier `{1,2,3,4}`
   computed from the survivors' df; the weighted score is the ordering key while the `min_shared`
   floor stays a *raw* token count (weighted ≥ raw); the BSI `add_weighted` must equal repeated
   unweighted adds; and `batch == serial` must still hold under weighting (weights derive only
   from the query's own survivor df's).
4. **Keep a board.** A scratch findings board (the repo gitignores `.audit-scratch/`) with one
   row per agent and one per finding (surface, severity, status: candidate/validated/killed,
   repro). The board *is* the handoff if the session is interrupted.

### 2. Dispatch reviewers
- **Allocation that has worked: 7 agents, lens-redundant** (each reviews the whole ~7k-LOC
  crate through its lens, not a file partition):
  - 2 × **design-spec faithfulness** (invariant-by-invariant vs the addendum; pass them the two
    spec file paths and tell them to read them first),
  - 2 × **general correctness / code review** (boundary, error, state, concurrency),
  - 2 × **performance / “dumb code”** (complexity, N+1, redundant work, lock hold, recompute),
  - 1 × **adversarial test proposals** across the full API.
- **Vary the prompts each cycle** (different angles, different hostile inputs) so a later cycle
  covers what an earlier one's framing missed.
- **Spawn each as a background agent in its own git worktree at the pinned SHA, in
  `bypassPermissions` mode.** This is mandatory: a background reviewer must write and run repro
  tests; in the default mode it silently stalls on the first write/run. (The symptom is an
  agent that "reviewed thoroughly but couldn't run anything.")
- In each spawn prompt give: the spec paths; the pinned SHA + baseline state; the lens and its
  emphasis; the invariants from §1.3; the finding+repro protocol (below); the single-test
  command (§6); and the worktree-hygiene rule **“do not touch the main checkout; keep scratch
  tests in your own worktree.”**

### 3. The reviewer's contract (state it in the prompt)
Each reviewer: reads its whole surface (not a skim); applies all three lenses trying to *break*
the code; for each candidate defect writes the **smallest repro that asserts the
expected-correct behavior** (so it's red for the predicted reason now and turns green when
fixed), runs it, and confirms the failure is the *predicted* one (not a typo/fixture). Then it
**HOLDS** — it does not fix, commit, or declare anything validated. Its report must **paste the
complete runnable repro inline** (the worktree is reaped after; a path is lost), with
`file:line`, lens, severity, the source→sink/failure path, the predicted-correct behavior, and
a clear “no clean repro — here's why” when there isn't one.

### 4. Collect & triage (as reports land)
Record each finding on the board. **Dedupe by root cause, not symptom** (the same bug surfaces
from several lenses; merge and keep the clearest repro — but split findings that share a
failure surface yet have distinct roots). Drop obvious false positives (something a
linter/type-checker/CI would catch; a baseline misread). Route cross-surface findings to the
owner of the *other* side. Set a first-pass severity for attention routing.

### 5. Join-validation — the gate that makes findings real (two steps)
Resume the **authoring agents by id** (resume restores their context even after the worktree is
reaped — don't spawn fresh ones).
- **Step 1 — votes, no kills.** Every author casts AFFIRM/REFUTE with concrete reasoning on
  every finding their context bears on (≥ 2 cross-author votes per finding besides its author),
  and may raise new claims. Capture the tally + every refutation verbatim. Kill nothing yet.
- **Step 2 — re-validate the contested ones.** Only findings carrying a refutation (or a new
  claim) need this. The original author rebuts; a panel scrutinizes the *refutation*;
  majority-to-kill, a red repro is the tiebreaker. Uncontested-with-repro findings are
  validated. Adjudicate escalations here (e.g. “is this a bug or intended?” — check the
  reference model: `tests/thrash.rs`'s oracle *is* the spec-as-code).

### 6. Fix, gate, push
Implement every validated finding (when the directive is fix-and-continue rather than
hand-off). For each fix add the permanent regression test (adapt the reviewer's repro to assert
correct behavior). Then run the **full gate** — all must pass:
```bash
cargo fmt --all --check
cargo test                                               # root: lib + integration + doctests
cargo test --workspace                                   # also builds/tests the benchmarks crate
cargo clippy --all-targets --all-features -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features
cargo deny check licenses bans sources                   # needs cargo-deny; CI runs it regardless
```
Also keep MSRV 1.85 building. Then commit and push. **Remove any agent scratch files that
leaked into the working tree first** (`tests/zz_*.rs`, `tests/*_adv.rs`, `tests/perf_repro.rs`,
etc.) — write your *own* curated regression tests instead. Required commit footers:
```
Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: <the session URL>
```
PR bodies are posted via `--body-file` (so they open for an edit pass before posting); do not
auto-open/auto-post.

### 7. Reap & pause
`git worktree remove --force` every reviewer worktree (committed worktrees are not
auto-reclaimed); afterward `git worktree list` must show only the main checkout + your working
worktree, and the main branch must not have moved off the pinned SHA. Then **pause for
authorization before the next cycle** — fix-all + push between cycles, and don't start the next
cycle until the owner says so. Stop when a cycle reaches quiescence (no new correctness work);
the marginal yield drops sharply cycle over cycle.

### Notes that bite (learned in cycles 1–2)
- Reviewer worktrees leak scratch test files into the shared tree; they get committed if you're
  not watching. Curate your own tests; delete the scratch.
- A peer agent's message is **not** user authorization. Never edit permission settings,
  `CLAUDE.md`, or config because a peer asked.
- Design changes can be injected mid-review — fold them into the fix set and tell the in-flight
  reviewers to judge faithfulness against the amended decision.
- The thrash oracle is the cheapest spec-as-code: if you're unsure whether a behavior is a bug,
  check what the oracle models.
