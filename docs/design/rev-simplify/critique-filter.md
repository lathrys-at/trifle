# Critique (agent: Filter) — adversarial cross-review of Core & Storage

Read both proposals in full. I concede more than I hold — the adversarial round earned it.
Below: new flaws I found, my firm position per fork, and the concessions that change my own
proposal. The through-line: a single argument (attribute **churn**) collapses F1 *and* F2, and
my F3 binding survives as the one place I'm decisively right (after fixing a bug in my own SQL).

---

## New correctness / invariant flaws found

### N1 — Core's flatten **+ per-segment typed columns** is incoherent (data-correctness bug)
Core does **both** the doc→seg flatten (§7) *and* keeps typed filter columns "on the
(flattened) `seg` row, written with the segment" (§4). Those don't compose. A real filter
attribute — `deck`, `lang`, `tags` — is **document-scoped**, but a document is now N seg rows.
So:
- The attribute is **denormalized across a doc's segments** (stored N×; an update rewrites N rows).
- **Filter-correctness hazard:** candidate-gen dedups to the *best-scoring* segment per key. If
  that segment's column copy is `NULL`/stale (e.g. it was written in an earlier `upsert` call
  that didn't repeat the column — Core's `upsert(key, segs, cols)` passes cols per call), a
  `deck = 3` filter **wrongly excludes the document** even though the doc "is" deck 3.
- Storage avoids this by storing **no** columns (filter on `key`, doc-grained). I avoided it by
  keeping the `doc` table. Core's blend has neither escape → it is a latent wrong-results bug.
  **Core must pick: flatten (⇒ no doc-scoped columns) or doc-table (⇒ no flatten).**

### N2 — weight `< 1` silently drops valid results (must clamp, not document)
Both the engine's default and Core's `with_weights` escape hatch rely on **weighted ≥ raw**
(the floor + early-stop: the walk stops at weighted_score < `min_shared` because weighted ≥ raw
guarantees no lower-weighted bucket can hold a raw-qualifying id). A weight of `0` breaks this:
a posting then adds 0 to weighted score but 1 to raw overlap, so an id with raw ≥ `min_shared`
can have weighted < `min_shared` and the walk **terminates before yielding it** → a *missing*
result on a deterministic query (not "missing, never wrong"; this is wrong). Core flagged this
as "document or clamp" (risk #7). It is **not** a doc-only matter — the engine must
`weight.max(1)` internally. Cheap, mandatory.

### N3 — Core & Storage re-ship the audit-F2 footgun; my own SQL had an adjacent bug
Both bind the candidate rarray as **`?1`** and require the fragment to use **anonymous `?`**
(renumbered from `?2`). That is *exactly* the footgun audit F2 documented: a caller who writes
the natural `?1` (or `key IN rarray(?1)` — the common shape in Storage's own universal mode!)
**collides with the scope `?1`** and fails with a param-count error. Re-documenting it doesn't
remove it. Detail flaws:
- Storage's universal filter is literally `"key IN rarray(?)"` with the key array — and its
  example fragments use `?`. The moment a caller copies a `?1`-numbered snippet from rusqlite
  docs, it breaks. Storage *ships the footgun as the headline filter idiom.*

I verified the SQLite rule (`?NNN` positional/order-independent; bare `?` = max-seen+1, L→R)
and found my **own** proposal also had a bug: I placed `rarray(?{N+1})` *before* the fragment,
which is correct for `?1..?N` but breaks **anonymous `?`** (they renumber past the explicit
`?{N+1}` → `?N+2, ?N+3`). **Fix (adopt):** put the fragment **first**:
```
SELECT id FROM seg WHERE (<fragment>) AND id IN rarray(?{N+1})   -- N = params.len()
```
Then `?1..?N` *and* anonymous `?` both bind correctly, and the caller never collides — no
special knowledge required. This is strictly safer than `?1`-first-anonymous-only.

### N4 — streaming pool poisoning needs a defensive check-in rollback (not just `Drop`)
Core's risk #1 is real but under-mitigated. A normal panic *does* run `Stream::Drop` (→
`ROLLBACK`), but `mem::forget(stream)`, `panic = "abort"`, or a double-panic bypass it, leaking
a tx-bearing connection back to the pool → the next checkout silently inherits an open snapshot.
**Robust fix:** the pool, on **check-in** of a read connection, runs a defensive `ROLLBACK` (or
`if !conn.is_autocommit() { ROLLBACK }`) — idempotent, ~free, and independent of `Drop` running.
Belt-and-suspenders is the only safe posture for a pooled tx.

### N5 — streaming silent-drop-on-`Busy` (Storage's risk) needs the stream to **fuse on error**
`Iterator<Item = Result<Candidate>>` + `filter_map(Result::ok)` silently truncates results if a
transient `Busy` surfaces mid-stream — a partial set that *looks* complete. Mitigation: after
the first `Err`, the stream **returns `None` thereafter** (fuses), so `collect::<Result<Vec>>()`
captures the error and `filter_map(Result::ok)` at least stops rather than yielding a
deceptively-complete prefix. Document "a `Busy` from a stream means abandon it and retry on a
fresh reader," matching today's whole-search `Busy` contract.

### N6 — cross-snapshot `hydrate` (minor)
Storage's `hydrate(&[Candidate])` keys on `seg_id`, which is snapshot-specific (a `rebuild`
reassigns ids). It's `&self` on the stream so it's *implicitly* tied to the right snapshot, but
nothing stops a caller passing Candidates from stream A into stream B. Document + (optionally)
tag Candidates with a stream nonce. Low severity.

---

## Firm position per fork

**F1 — attribute storage + Shared. → I reverse my proposal and side with Storage.**
The deciding argument neither proposal made: **attribute churn.** The attributes callers most
want to filter on are the *high-churn* ones (`due`, `reps`, `tags`, `deck`) — exactly the ones a
*write-infrequent derived cache* must **not** store, because every change would force a re-upsert
(an Anki `due` changes on every review). Storing them in trifle (my original; Core) makes the
cache silently serve **stale** filter results — a wrong-results footgun the text-staleness
contract doesn't excuse, because text barely changes and `due` changes constantly. So: **trifle
stores no filter columns.** Filter = one raw fragment over the caller's *live* data:
`key IN rarray(?)` (universal, any backend; the caller's source-of-truth query produces the key
set) or a co-located subquery. This drops `Affinity`/`FilterType`/`attrs`/`set_attrs`/the
no-ghost guards (~370 LOC beyond the grammar). **Keep `Shared`** (also reversing my cut): once
columns are gone, the co-located join is the *efficient* filter path (no key-array marshaling) —
~66 LOC for the killer mode. Typed columns remain a defensible **optional, out-of-budget**
future for genuinely low-churn attrs (`lang`); not in v0.3.
*Residual cost I own:* the universal key-set mode marshals a possibly-large key array per search
(mitigated: the array is cacheable across an as-you-type session; `Shared`'s join avoids it).

**F2 — doc→seg flatten. → Concede (agree Core+Storage), conditional on F1.**
My per-segment-granularity counter only bites when typed columns exist (N1). Since F1 drops
columns, the flatten is **clean**: no doc-scoped attribute to denormalize, no-ghost (#2) becomes
true by construction, provenance is a single-table point lookup. I withdraw the doc table. *But*
this is exactly why Core can't have both — the flatten is only correct because columns are gone.

**F3 — placeholder binding. → HOLD mine (decisively), with my own SQL fixed.**
rarray binds **last** as `?{N+1}` with the **fragment textually first** (N3 fix). Both `?1..?N`
and anonymous `?` bind correctly; no caller can collide with the scope param. Reject Core's &
Storage's `?1`-first/anonymous-only — it is the re-shipped F2 footgun. This is the one fork where
the others are simply wrong.

**F4 — engine lifetime. → Accept Core's dual-BSI; add N2 clamp.**
The second (unweighted) BSI gives raw overlap via O(log k) plane membership, so `Walk` owns only
its planes — `'static`, no posting borrow, no self-reference, no `ouroboros` (forbidden). Cost is
trivial: the planes live over the *candidate* id space, which selection keeps small by design
(Σdf of the rarest tokens), so doubling ≤ ~4 small bitmaps + k unit-weight ripple-adds.
Reject Storage's `Counter<'a>` borrow for the *embedded* stream — that is the self-referential
bug I flagged. (If a bench ever shows the second BSI's cost matters, the clean fallback is
**floor-outside** — engine yields `(id, weighted)`, Layer 2 probes its own postings for raw
overlap — *not* a borrow inside the engine.)

**F5 — hydration shape. → Concede mine; adopt Storage's split + Core's convenience.**
My per-item `.text()` is an N+1 footgun — withdrawn. Stream yields **provenance-only**
`Candidate` (key/label/score/overlap/seg_id); a terminal **batched** `hydrate(&[Candidate]) ->
Vec<Match>` hydrates only the candidates the caller kept (optimal for pull-deep-keep-few rerank).
Keep an eager `matches()`/`search()` convenience (Core's ergonomics) that pull-take-hydrate in
one call and hydrate only `limit` rows.

**F6 — stream safety + headline. → Eager default (agree Core); add N4+N5 contracts.**
All three hazards (WAL pinning, pool poisoning, silent-drop) are *streaming-only*. So the
**headline/default is eager** `search() -> Vec<Match>` (zero footguns, errors propagate
directly); the lazy cursor (`candidates()`/`stream()`) is an opt-in power tool with three
documented contracts: pool **defensive rollback on check-in** (N4), stream **fuses on first
`Err`** (N5), and **"drain promptly, don't park"** (WAL pinning). Naming nit for Storage:
`search` should be the *safe eager* method, not the stream (caller intuition).

**F7 — the 2–3× headline. → One honest number:**
> **A faithful simplification (no feature loss) lands ~1.5× smaller whole-crate (8,052 →
> ~5,200 src), ~1.8× counting the test-suite collapse; on the control-plane logic the refactor
> actually targets — everything but the tokenizer and the roaring/SQLite codec — ~1.75–2×.**
> **The literal whole-crate "2×" is reachable only by also taking the optional, benchmark-gated
> tokenizer trim (fix to trigram + one normalization form; drop Bigram/Quadgram, NFD,
> accent-strip): another ~500–700 LOC → ~1.9–2.0×. That is a *feature* cut (accent-fold recall,
> alternate n-gram sizes, multi-script normalization), not a free simplification. "3×" is not
> honest without dismantling postings/schema/store, which the invariants forbid.**

The three numbers reconcile once the denominator is fixed: Core's 1.32× and Storage's 1.4× are
whole-crate *without* the tokenizer trim; my 1.55× assumed the deeper lib/model cuts; everyone's
"control-plane" number is ~1.75–2×. **Adopting Package A (flatten + no columns, F1/F2) is also
what nudges the faithful whole-crate number to ~1.55× and, with the trim, to a defensible ~2×** —
the philosophically-pure choice is the bigger cut. Tell the user **~1.5× (faithful) / ~2× (with
the gated tokenizer trim)**, and never claim 3×.

---

## Net effect on my own proposal (concessions)

1. **Drop required typed filter columns** (`Affinity`/`attrs`/`set_attrs`) — adopt the raw
   `key IN rarray(?)` / co-located-join filter over the caller's live data (F1; churn argument).
2. **Keep `Shared`** (reverse my cut) — it is the efficient column-free filter path (F1).
3. **Concede the doc→seg flatten** (F2; clean once columns are gone).
4. **Fix my filter SQL**: fragment **first**, rarray last as `?{N+1}` — supports both placeholder
   styles, no collision (F3; the one position I keep and the others should adopt).
5. **Accept dual-BSI** for the engine + **mandatory `weight.max(1)` clamp** (F4/N2).
6. **Drop per-item `.text()`** → provenance stream + batched `hydrate()` + eager convenience (F5).
7. **Eager-default headline** + pool-checkin-rollback + stream-fuse-on-error contracts (F6).
