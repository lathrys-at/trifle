# Critique (Storage) — adversarial cross-review, round 2

Reviewing `proposal-core.md` and `proposal-filter.md` hostilely; defending/revising
`proposal-storage.md`. Skipping the settled cuts (welford, Effort, Ranker, telemetry, scope)
and the honest-LOC reality — all three agree. Spending effort only on the seven forks and on
correctness/invariant flaws the others (and I) missed.

---

## New correctness / invariant flaws found

### A. Typed filter columns introduce a staleness surface that text caching does not (Core §4/§7, Filter §4.2)

Both Core and Filter store the caller's filter attributes as trifle-owned columns
(`ColumnType`/`Affinity`), written at index time via `upsert`/`set_attrs`. Neither flags the
consequence: **a cached attribute can silently drift from the caller's source of truth, and
trifle's drift-reset only detects schema-*shape* changes, never stale values.**

The usual rebuttal — "text is cached too, so attrs are no worse" — is wrong, and the
asymmetry is the whole point. A caller's re-index trigger is *"text changed."* Attributes
routinely change **decoupled from text**: moving an Anki card to another deck changes `deck`
but not `front`/`back`. The caller's natural trigger misses it, so the cached `deck` goes
stale and `Filter::new("deck = ?", …)` **silently returns wrong results** — no error, no
drift-reset, no signal. Text caching has no equivalent blind spot (its trigger is complete).

Core makes this worse: it cuts `Shared`, so the filter fragment *cannot* reach the caller's
live tables — there is **no staleness-free filter path at all** in Core's design. Filter has
the same exposure with no escape hatch either.

This is the core argument for my F1 position: the zero-declaration `key IN rarray(?)` / co-located
join filter is evaluated against the source of truth *at query time* and is **staleness-free by
construction.** Typed columns should be an opt-in for callers who accept the maintenance + drift
risk in exchange for pushed-down Sidecar filtering — not the mandated, only mechanism.

### B. `OverlapWalk<'a>` / `Counter<'a>` that *borrows* the postings is self-referential when embedded (Filter §2; my own original §2)

The team lead caught this in my proposal; it applies equally to **Filter**. Filter's
`OverlapWalk<'a>` borrows `present: &'a [&'a RoaringBitmap]`. To embed it, Filter's
`CandidateStream` must own that postings `Vec` (it has nowhere else to live for a per-query
search), and the walk borrowing a `Vec` the same struct owns is an ouroboros — unconstructable
in safe Rust without `ouroboros` (forbidden: single dep) or `unsafe`. Filter's risk #1 flags the
*conn/tx* self-reference but **not** the postings self-reference. Core is the only one who
addressed it (dual-BSI). See F4 for the clean fix that beats both.

### C. Core's `Match`-only output drops the matched-terms-with-df signal a custom reranker needs (Core §3)

Core deletes `Candidate`/`QueryContext` and returns `Match { …, score, overlap, seg_len }`.
But `tests/cycle2.rs::IdfSum` (the canonical custom reranker) scores from **per-candidate
matched tokens each paired with its df**. From a `Match`, that is unrecoverable: re-tokenizing
the text recovers the tokens, but the **df** requires reading the `term` table — a query Core
exposes nowhere. So Core's design silently regresses the documented custom-ranker extension
point. My `CandidateStream::matched_terms(&c) -> (token, df)` (no SQL — postings in hand)
preserves it. Filter has the same gap (its `Candidate` carries no matched-terms accessor).

### D. Core reads `n_segments`/`avgdl` from `stats()` — a *different* snapshot than the search (Core §3)

Core says corpus-relative rerank signals "come from `stats()`." `stats()` opens its own tx
(its own WAL snapshot). A reranker combining `Match.score` (search snapshot) with
`stats().n_segments` (a later snapshot) crosses a snapshot boundary — today `search.rs` reads
`read_seg_stats` *inside* the search tx for exactly this consistency. Minor, but it breaks the
single-snapshot guarantee for any corpus-relative custom score. My `stream.n_segments()/avgdl()`
serve them from the search snapshot.

### E. Core's flatten + typed-columns-on-`seg` = per-segment attribute ambiguity (Core §7)

Core flattens `doc→seg` **and** keeps typed columns, so columns land on the `seg` row → they
become **per-segment**. A chunked doc (front+back) stores `deck` twice; `set_attrs(key, deck=8)`
must rewrite every seg row; and — the real hazard — if two segments of one key hold *different*
column values (now possible), a per-segment `WHERE seg.deck = 7` matches one segment and
dedup-by-key then keeps whichever segment won the score tie. **Filter results become a function
of which segment dedup picked.** Core flags "granularity changes" but not this nondeterminism.
This is why flatten and typed-columns don't compose; see F2.

---

## The seven forks

### F1 — filter storage + Shared. *Synthesis; concede Shared cut, hold key-set default.*

The "you can't early-stop on a predicate you don't store" criticism of my design is **false**:
my filter is *also* in-walk — `key IN rarray(?2)` (or a join) folds into the per-bucket
provenance `SELECT`, so the walk early-stops at `limit` passing docs exactly as theirs does. The
real difference is *where the predicate's data lives*, and on that axis my flaw A wins: query-time
evaluation is staleness-free; typed columns are not.

**Both modes coexist** (the lead's question — yes): the raw fragment is one mechanism; whether it
references a bound key-set rarray, a co-located subquery, or trifle-stored columns is just what
the caller writes. So: **default = zero-declaration key-set/join (blessed, staleness-free, any
backend); typed columns = documented opt-in** (Core/Filter's mechanism) for Sidecar callers who
want pushed-down filtering and accept staleness + write-maintenance.

**Shared: I concede the cut** — not because key-set makes it unnecessary for *perf* (the
co-located join is the only staleness-free *pushed-down* path), but because keeping it forces the
`B: Backend` generic onto every public type (`Index<T,B>`, `Writer/Reader<…,T,B>`), a real
verbosity tax against the simplification mandate. Collapsing to a concrete store is the win.
Recover co-location via an **optional `ATTACH` on the Sidecar read-connection factory** (far
lighter than the whole `Shared` backend) for the niche that needs staleness-free pushed-down joins.

### F2 — doc→seg flatten. *Hold flatten; refine column placement.*

Flatten is right and strongly motivated: invariant #2 (no-ghost) **dissolves entirely** when
there are no doc rows. But flatten and typed-columns-on-seg don't compose (flaw E). Resolution:
the **default design has no columns** (flatten is clean, dedup-on-key, zero ghost machinery). The
**opt-in columns live in a doc-level side table** `attr(key PRIMARY KEY, …)` joined only when
declared — preserving doc-level semantics (no per-segment ambiguity) and keeping the common path
join-free. This beats Core (per-segment hazard) *and* Filter (always-on doc table + the full
no-ghost guard set even for column-free callers).

**Dedup-on-Key cost** (hashing Text/Blob keys per survivor) is a non-issue: survivors ≤ pulled
depth (tens), and a key hash is microseconds. The pathological case (pull-everything, long keys)
is bounded by pulled count. Defended.

### F3 — placeholder binding. *Concede to Filter.*

Filter's "bind rarray **last** as `?{N+1}`, caller numbers naturally from `?1`" is strictly
better and removes the audit-F2 footgun my anonymous-`?` design inherited. `N+1` is
trifle-computed (from `params.len()`), never caller input → no injection. **Adopt it.** It's also
the only one of the three that lets a caller paste the exact SQL they'd run standalone.

### F4 — engine self-containment. *Concede my borrow was unsound; adopt engine-OWNS-postings (not dual-BSI).*

My `Counter<'a>` borrow is self-referential when embedded — conceded (flaw B; Filter shares it).
The fix is **not** Core's dual-BSI. Have the engine **own** the postings: `Counter::build(postings:
Vec<RoaringBitmap>, …)` — `effective_postings` already returns owned bitmaps, so trifle *moves*
the selected ones in (no copy). `Counter` is then `'static` (owns planes **and** postings);
`raw_overlap(id)` is a `contains` scan over its own postings (O(k), k≤12); `CandidateStream` owns
the `Counter` + a parallel `tokens: Vec<String>`, and `matched_terms` zips `tokens` with
`counter.postings()` by index. **No borrow, no ouroboros, no second counter.**

Why this beats Core's dual-BSI: dual-BSI pays a *full second plane build* over all k postings to
make raw-overlap a plane probe — wasted in the common **shallow-pull** case (build the whole
second counter, probe a handful of ids). Owns-postings defers the cost to *per-yielded-id*
(`contains`), so a top-10 search never pays for a tail it doesn't yield. Dual-BSI only wins on
very deep pulls — keep it as a benchmarked alternative behind the same `'static` owned `Counter`,
not the default.

### F5 — candidate / hydration shape. *Hold: choose-then-hydrate (batched) strictly dominates.*

Decisive argument vs Core's per-chunk eager hydration: the **reason to use the stream over
`matches()` is a deep-pool rerank.** Pull 200, keep 10. Core hydrates text for all 200 (every
chunk's survivors) → 190 wasted text reads. Mine pulls 200 provenance-only `Candidate`s, reranks
on `score`/`overlap`/`matched_terms` (no text), then `hydrate(&chosen_10)` → **10 reads, one
batched query.** When the reranker *does* need text, the caller hydrates the full pulled set in
one batched call — identical cost to Core, never worse. So choose-then-hydrate is a strict
generalization: never over-hydrates, always batched.

Against Filter's lazy per-item `.text()`: unless Filter secretly batches, each `.text()` is a
separate SQL round-trip → **N queries for N kept items** vs my one batched `WHERE id IN rarray`.
Filter's "pay no text cost if you rank on score alone" is true, but its hydration path loses
batching. My design gets both: no text cost until you choose, *and* one read when you do.

### F6 — snapshot safety + headline. *Concede eager headline; endorse a robust pool guard.*

**Pool poisoning** (Core's risk): make it impossible structurally — at read-connection
**check-in/checkout**, assert `conn.is_autocommit()` (O(1)); a connection that returns mid-tx
(Drop bypassed via `mem::forget`/double-panic) is rolled back or discarded, never recycled
mid-tx. Cheap, total. Endorse and make firm (the current `pool.rs` lacks it).

**Silent-drop on `Busy`** (my risk): the construction-time generation-skew `Busy` already
surfaces from `search()`/`stream()` *before* iteration (good). Mid-stream transient `Busy` (a
per-bucket query) is the live hazard: a caller doing `.filter_map(Result::ok)` masks it. Fix by
making the **convenience methods return `Result<Vec<_>>`** (propagate, no silent drop) and the raw
`Iterator<Item=Result<Candidate>>` the documented advanced path with a `try_collect` helper.

**Headline:** I concede — given the WAL-pinning + per-item-`Result` footguns, the **ergonomic
front door is eager `matches()`/`matches_batch()`** (drains immediately, no parked snapshot, no
per-item `Result`), with the streaming `CandidateStream` as the **architectural spine** the brief
mandates and the first-class building block for rerank/pagination/fusion. "Headline" splits:
spine = stream (honors the brief); default method most callers type = `matches()` (safe). Both
exist; neither hidden.

### F7 — the 2–3× target. *Consensus.*

Tell the user: commit the faithful restructure (~1.3–1.4× whole-crate, ~1.75× on the targeted
control plane, ~1.85× counting the test collapse) as the deliverable. Treat the **tokenizer trim**
(Trigram-only + NFC/casefold; drop Bigram/Quadgram/NFD/accent-strip) as an **optional,
benchmark-gated lever** that pushes toward ~1.7–2× *iff* the recall eval shows the dropped
normalizers don't move recall — it is a *capability cut*, not a simplification, so don't promise
2–3× on its back and don't cut it blind. Do **not** chase 3× into `postings`/`schema`/`store`/
`tokenize` bedrock; that trades the invariants for line count.

---

## Concessions that change my proposal

1. **F3:** adopt Filter's rarray-binds-last (`?{N+1}`) — drop my anonymous-`?` (the footgun).
2. **F4:** my borrowing `Counter<'a>` was unsound; switch to engine-**owns**-postings (`'static`).
3. **F6:** eager `matches()` is the ergonomic headline; streaming is the spine. Add the
   `is_autocommit()` pool guard and make convenience methods return `Result<Vec<_>>`.
4. **F1:** concede the `Shared` cut (for the `B`-generic collapse); recover co-location via an
   optional `ATTACH` hook. Hold the key-set default; demote typed columns to opt-in.
5. **F2:** hold flatten, but opt-in columns go to a doc-level side table (not on `seg`).

## Where I hold against both

- **F5:** choose-then-hydrate (batched) over Core's eager-per-chunk and Filter's per-item `.text()`.
- **F1 default:** zero-declaration, staleness-free key-set/join filter is the blessed path; typed
  columns are the opt-in, not the mandate (flaw A).
- **F2 default:** column-free flatten — the no-ghost invariant should *dissolve*, not survive as
  guards (against Filter keeping the doc table for all callers).
