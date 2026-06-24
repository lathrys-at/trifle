# trifle — crate specification

> Pretty good lexical/fuzzy search. Not bm25-grade ranking; built to stay fast over large
> corpora of small documents. (The speed *superlative* is a claim to earn with a benchmark,
> not assert — §10.)

A self-contained specification for `trifle`, an embedded, in-process trigram fuzzy-search
crate backed by SQLite. This document is normative for the crate's public surface and
internal architecture. It assumes nothing about any particular host application.

> **Design lineage (read this first if you knew the earlier draft).** This revision removes
> FTS5 entirely. trifle now owns a uniform **roaring inverted index** — one roaring posting
> per token, base + delta, for *every* token — rather than an FTS5 trigram index accelerated
> by a roaring tier for the rare tokens only. The earlier FTS5-based design and why this
> supersedes it are recorded in §12. The short version: the hardest, most dangerous part of
> the FTS5 design (a hand-rolled `unsafe` custom-tokenizer FFI bridge) existed *solely* to
> reconcile FTS5's tokenizer with ours — when the riskiest piece of a design is glue to an
> inherited dependency, the dependency is the anchor, not the foundation. Removing FTS5 also
> deletes the materialization cutoff `C`, its grace band `g`, the dual read path, and the
> debounced DF snapshot. What it *adds* is owning posting maintenance for the common tokens
> FTS5 maintained for free (§7.2) — the price, paid at `compact()` time, not on the hot path.

---

## 1. Overview

trifle indexes short text segments and answers typo/partial-tolerant queries, returning a
ranked list of matches each carrying *where* it matched. It owns a single SQLite store
holding the segment text, provenance, and an owned **roaring inverted index** (a base+delta
roaring posting per token); it ranks by shared-rare-token overlap, counted bit-sliced. It is
a **derived, rebuildable cache** over a caller-owned source of truth: it never touches the
caller's data store.

**Load-bearing assumption — documents are small** (target ≲ 1–2 KB per segment). This, not
corpus size, is what the design rests on: the ranking omits length normalization, which is
sound only for small documents (§7.4). Corpus size is bounded only by SQLite; growth costs
rebuild and fold time, not correctness.

**In scope:** tokenization, rare-token selection, overlap ranking, incremental posting
maintenance, deletion, concurrent reads during writes, full rebuild.

**Out of scope (the caller's, above trifle's boundary):** embeddings/semantic search; fusion
(RRF) with other signals; an exact/literal precision tier beyond what a custom ranker
provides; sub-trigram (`<3`-char) query handling; deciding *when* the cache is stale relative
to the source of truth.

---

## 2. The document model

trifle indexes **segments**. A segment is `(doc_id, source, ref, text)`:

- `doc_id: i64` — the caller's document identifier.
- `source: &str`, `ref: &str` — two **caller-defined, opaque** provenance labels (intended:
  `source` a category — `"field"`, `"ocr"`, `"caption"` — and `ref` a sub-location — a field
  name, a filename). trifle treats them as strings and returns them on a match so the caller
  knows *where* it landed; it keeps a `(doc_id, source)` index so per-category replace/delete
  is cheap.
- `text: &str` — the segment text. Stored **raw**; the tokenizer normalizes internally for
  matching (§5), and `Match.text` returns this original form with `span` indexing it.

A document may have many segments. Upsert replaces all segments under a `(doc_id, source)`
pair; delete removes all segments of a `doc_id`.

---

## 3. The minimal parameter surface

The central API claim — **the caller supplies at most a tokenizer; one optional strictness
dial covers ordinary tuning; everything else is derived or fixed** — gets *stronger* with the
FTS5 removal: the whole performance class around the materialization cutoff (`C`, the grace
band `g`, the cutoff fraction `p`, the posting downsample ceiling, the refresh debounce) is
gone, because there is no longer a fast-tier/slow-tier boundary to manage.

Every parameter is one of three kinds, and only one is legitimately user-facing:

| Kind | Question it answers | In the API? |
|---|---|---|
| **Behavior** | *which* results come back | Yes — the only real tuning |
| **Performance** | *how fast* / how much memory (same results) | No — derive or fix |
| **Strategy** | *which algorithm* (pluggable) | Yes — as a trait |

A second axis — **when** a parameter binds — tells the caller what is cheap to change:
index-time (fixed at build; changing needs a rebuild), query-time (a per-search option, free),
maintenance (recomputed in upkeep, never user-set).

### 3.1 The full classification

| Parameter | Kind | Time | Default | Exposure |
|---|---|---|---|---|
| **tokenizer** | Strategy | index | `TrigramTokenizer` | **exposed** (type param) |
| **ranker** | Strategy | query | `OverlapRanker` | **exposed** (trait) |
| **backend** | Strategy | index | `store::Sidecar` | **exposed** (type param, §4.1) |
| **`m`** (match floor) | Behavior | query | `2` | **exposed** — the one strictness dial |
| **`B`** (breadth budget) | Behavior | query | `0` | exposed (optional, power users) |
| `F` (typo floor) | Behavior | query | **derived** `= m + d` | derived, not free (§6.2) |
| `d` (per-typo damage) | — | — | `4` | constant inside `F` |
| `α, β` (cost coefficients) | Performance | query | `α=0, β=1` | calibrated / default pure `Σdf` |
| `k_max` (selection cap) | Performance | query | `12` | fixed safety cap |
| `data_version` (drift token) | — | index | caller-supplied | **exposed** |

*Removed by the FTS5 pivot:* `C` (materialization cutoff), `g` (cutoff grace band), `p`
(cutoff fraction), the posting downsample ceiling, the refresh debounce. They existed to
manage the boundary between a roaring tier and the live FTS5 read; there is no boundary now.

### 3.2 In practice

Zero-config works (default tokenizer + sidecar). The one knob you reach for is **`m`**
(strictness per match; `2` is a strong default); **`B`** is the orthogonal breadth axis
(default `0`). **`F` is not free** — it is `m + d` (§6.2). Tokenizer/backend are index-time;
ranker/`m`/`B` are query-time. The Performance rows are internal: `α/β` default to pure `Σdf`
and can be self-calibrated rather than guessed; `k_max` is a fixed valve.

---

## 4. Public API

Illustrative signatures (shapes, not a frozen contract).

```rust
// Generic over the tokenizer (monomorphized — hot path, §5) and the storage backend (§4.1).
// Both default, so the common case is just `Index`.
pub struct Index<T: Tokenizer = TrigramTokenizer, B: store::Backend = store::Sidecar> {
    /* owns the backend (connections) + config; not exposed */
}

pub struct Config {
    pub data_version: u64,               // caller's drift/epoch token (§8.4)
    pub advanced:     Advanced,          // α/β, k_max overrides — rarely touched
}
// The tokenizer is a *type* parameter (its value, usually zero-sized, supplied at open), not a
// Config field — it must monomorphize and is fixed for the index's life. The ranker is NOT
// here: ranking is query-time (§3), so it lives only in SearchOpts.

pub struct SearchOpts<'a> {
    pub limit:      usize,                       // top-k
    pub min_shared: Option<u32>,                 // m; default 2
    pub breadth:    Option<u64>,                 // B; default 0
    pub ranker:     Option<&'a dyn Ranker>,      // per-query ranker; None → built-in OverlapRanker
    // scope/exclusion: a membership PREDICATE (§7.4) over the provenance trifle already has.
    pub scope:      Option<&'a dyn Fn(i64, &str, &str) -> bool>,
}

pub struct Match {
    pub doc_id: i64,
    pub source: String,
    pub ref_:   String,
    pub span:   Option<(usize, usize)>,  // [first, last) UTF-8 bytes within the matched segment
    pub text:   Option<String>,          // the whole matched segment, original form (§5, §7.1)
    // rank is conveyed by position in the returned Vec<Match>
}

impl<T: Tokenizer, B: store::Backend> Index<T, B> {
    pub fn open(backend: B, tokenizer: T, config: Config) -> Result<Self>;

    // writes (single-writer; §8.3)
    pub fn insert(&self, doc_id: i64, source: &str, segments: &[(&str, &str)]) -> Result<()>;
    pub fn insert_batch(&self, batch: impl IntoIterator<Item = Segment>) -> Result<()>;
    pub fn remove(&self, doc_id: i64) -> Result<()>;

    // reads
    pub fn search(&self, query: &str, opts: SearchOpts<'_>) -> Result<Vec<Match>>;
    pub fn search_batch(&self, queries: &[&str], opts: SearchOpts<'_>) -> Result<Vec<Vec<Match>>>;

    // maintenance (§8)
    pub fn compact(&self) -> Result<CompactStats>;     // fold deltas into bases, drop emptied terms, reclaim
    pub fn rebuild(&self, corpus: impl IntoIterator<Item = Segment>) -> Result<()>;

    // observability (read-only): one call
    pub fn stats(&self) -> Stats;        // N; term count; total delta backlog; on-disk sizes; staleness
}

// Convenience for the common case (default trigram tokenizer + an owned sidecar file):
impl Index<TrigramTokenizer, store::Sidecar> {
    pub fn open_at(path: &Path, config: Config) -> Result<Self>;
}
```

Note the maintenance surface lost `refresh()`: with DF maintained as a live cardinality column
(§7.2) there is no debounced snapshot to force, and folding deltas is `compact()`.

### 4.1 Storage backend — `trifle::store`

trifle's SQLite surface is contained, so *where connections come from* and *how its tables
are named* are abstracted behind `store::Backend`. With FTS5 gone, that surface is just plain
tables and BLOBs — no virtual tables — which makes the backend genuinely portable.

```rust
pub mod store {
    pub trait Backend: Send + Sync {
        type WriteGuard<'a>: DerefMut<Target = rusqlite::Connection> where Self: 'a;
        type ReadGuard<'a>:  Deref<Target = rusqlite::Connection>    where Self: 'a;
        fn write(&self) -> Result<Self::WriteGuard<'_>>;   // exclusive writer
        fn read(&self)  -> Result<Self::ReadGuard<'_>>;    // a pooled read-only connection
        fn namespace(&self) -> &Namespace;

        // INVARIANT: every connection write()/read() ever hands out must have had this run on
        // it once, before first use. trifle uses it to set pragmas (Sidecar mode) and register
        // the `carray`/`rarray` vtab used for batched provenance hydration (§7.4). With FTS5
        // gone this no longer registers a tokenizer — the connection setup is ordinary.
        fn init_conn(&self, conn: &rusqlite::Connection) -> Result<()>;
    }

    pub struct Sidecar { /* owns a file: write Mutex + read pool; runs init_conn on open */ }
    pub struct Shared  { /* delegates connections; runs init_conn on each before handing back */ }
}
```

- **`store::Sidecar` (default).** trifle opens its own file, owns the write `Mutex` + read
  pool, sets WAL / `mmap` / pragmas. Full encapsulation; the caller passes a path; namespace
  empty. Recommended.
- **`store::Shared` (opt-in).** trifle's tables live, **namespaced**, inside a database the
  caller owns; the caller supplies connection access. Only for a hard co-location requirement.
  The caller takes on single-writer serialization across the whole file, compatible
  WAL/pragma setup, and not holding a transaction across trifle's rebuild swap (§8.5). Drift
  and rebuild touch only trifle's namespaced tables.

#### Namespacing (`store::Namespace`)

A **validated, enumerable** value the caller customizes — not a bare prefix string:

```rust
pub struct Namespace { /* validated */ }
impl Namespace {
    pub fn prefixed(prefix: &str) -> Result<Self>;        // default "trifle_"; identifier-safe, not sqlite_*
    pub fn explicit(map: TableMap) -> Result<Self>;       // name each table explicitly
    pub fn table_names(&self) -> impl Iterator<Item = &str>; // for collision-check / caller migrations
}
```

A welcome simplification from dropping FTS5: there are **no virtual-table backing tables** to
reserve (FTS5 expanded one `idx` into five) — trifle's tables are all plain, so `table_names()`
enumerates exactly what gets created, plus the two rebuild-shadow tables.

#### Text storage — snapshot (default) vs contentless resolver

By default trifle stores a **snapshot** of each segment's text in `seg.txt` (§7.1). That copy
is what makes the index **self-contained** (queries work with the source absent) and deletion
**correct by construction** (delete re-tokenizes bytes trifle wrote itself). It is the right
default for a reusable cache, and the correct choice whenever the source is foreign, remote, or
expensive to reach — there the copy is the cache's whole point, not waste.

But in **`Shared`** mode the text is *already in the same file*, so the snapshot is pure
duplication. For that case trifle can run **contentless** — store no `seg.txt` and fetch text
through a caller-supplied resolver:

```rust
pub trait TextResolver: Send + Sync {
    /// Text for each requested segment, in order. `None` = unavailable at the source.
    fn resolve(&self, segs: &[(i64, &str, &str)]) -> Result<Vec<Option<String>>>;
}
// on Config: external_content: Option<Box<dyn TextResolver>>  // None = snapshot (default)
```

trifle calls the resolver where it would have read `seg.txt`: hydrating survivors (for
`Match.text` and a ranker's literal-verify). That use is **drift-tolerant** — a `None` just
yields `Match.text = None`, a stale string only mis-highlights a span.

**The sharp edge is deletion, and the spec is explicit about it because the obvious
implementation is wrong.** `remove(doc)` must know each segment's token set to pull its id from
the right postings (§7.2). Snapshot mode re-tokenizes the stored text; a resolver returns the
*current* source text — which on a delete is typically **already gone**, so the resolver returns
`None` exactly when trifle needs it most. Contentless mode therefore needs one of:

- **A caller ordering protocol (resolver-only):** the caller updates/removes trifle *before*
  mutating or deleting the source row, so the resolver can always return the **indexed** text
  when trifle asks. This is the cache-invalidation-precedes-source-write discipline; honored, it
  delivers the full space win. Violated, it silently leaks stale posting entries no fold cleans
  — so it is a real correctness contract on the caller, not boilerplate.
- **A stored per-segment token set (`fwd`, §7.1):** trifle keeps each segment's tokens, so
  delete reads them and needs no text. Self-contained, no protocol — but for trigrams a token
  set is **roughly text-sized**, so it partly eats the space win that motivated going
  contentless. (See §12: contentless is a genuine win mainly under the protocol; the token-set
  variant trades a text copy for a similar-sized forward index.)

So contentless is an **advanced `Shared`-mode opt-in with a contract**, not a transparent
optimization; snapshot stays the default everywhere.

---

## 5. Tokenizer (Strategy trait, index-time)

```rust
pub trait Tokenizer: Send + Sync {
    /// The token representation. Bounds let `HashMap<Token, _>` be probed by `&str`
    /// (so a query token keys the same bucket as one read from the store).
    type Token: Borrow<str> + Hash + Eq + Ord + Clone;
    fn tokenize<'a>(&'a self, text: &'a str) -> impl Iterator<Item = Self::Token> + 'a;

    /// A stable hash of this tokenizer's BEHAVIOR (n-gram size, normalization, casefold).
    /// Stamped into the index (§8.4); a change forces a rebuild, because the postings are
    /// keyed by whatever this tokenizer produced at build time. A caller shipping a custom
    /// tokenizer owns bumping this on any behavioral change, including a bug-fix patch.
    fn fingerprint(&self) -> u64;
}
```

`Index` is **generic** over the tokenizer, not `dyn` — it is on the hot path (called per
window), so it monomorphizes. **With FTS5 removed, the tokenizer is now plain Rust applied to
text → tokens; there is no FTS5 fold to reconcile, no custom-tokenizer FFI, no per-connection
tokenizer registration, no index-time offset-map subsystem.** The single-tokenizer invariant
becomes trivially true — there is exactly one tokenizer, used for indexed text, postings, and
queries. This deletion is the core dividend of the pivot (§12).

**Built-in default — `TrigramTokenizer`:** `type Token = Trigram`. Slides a window of 3
Unicode scalar values over a normalized form of the text; a `<3`-char string yields none.
**Normalization is the tokenizer's job and is configurable** (not a hard property of trifle):

```rust
TrigramTokenizer::new()                       // default: NFC + Unicode lowercase
TrigramTokenizer::builder()
    .normalization(Normalization::Nfd)        // or NfdStripMarks (accent-insensitive), Nfc, None
    .casefold(false)                          // skip lowercasing
    .assume_normalized(true)                  // caller guarantees input already in this form
    .build()
```

- **NFC vs NFD is a real choice.** NFC (default) is safe and compact; **NFD / NfdStripMarks**
  (decompose, drop combining marks) is often *better* for fuzzy search — it exposes base
  letters as separate scalars, so a query `cafe` shares trigrams with a stored `café`. The
  default is NFC for least surprise; reach for NFD/strip when accent tolerance matters.
- **`casefold(false)` / `assume_normalized(true)`** skip the passes when the caller guarantees
  inputs are already in form. **The one invariant:** whatever normalization is chosen applies
  identically to indexed text and queries (one tokenizer, both sides), so `assume_normalized`
  is sound only if the guarantee holds for *both* writes and searches.

**`Trigram` / `Ngram` — the zero-allocation token.** Parameterized by **byte capacity**
(generic on *stable* Rust — const-generic array lengths — sidestepping the `N·4` const-expr
problem):

```rust
pub struct Ngram<const CAP: usize> { buf: [u8; CAP], len: u8 }   // inline; Copy
pub type Bigram   = Ngram<8>;    // ≤ 2 code points
pub type Trigram  = Ngram<12>;   // ≤ 3 code points
pub type Quadgram = Ngram<16>;   // ≤ 4 code points
```

`Hash/Eq/Ord` delegate to the `str` slice (via `Borrow<str>`). Aliases are "Trigram & friends";
another width is one `type` alias. Optional nightly feature `ngram-const-exprs` adds the
code-point-count spelling `Ngram<3>` as a *distinct* type (requires nightly, `generic_const_exprs`
is incomplete — most callers leave it off).

**Spans and the offset map (now survivor-only).** `Match.span` indexes the *raw* stored text.
When normalization changes byte layout (length-changing folds, decomposition), mapping a
matched region back to raw bytes needs a normalized→raw offset map — but this now runs only
when **hydrating a survivor** (≤ `limit` of them) or inside a custom ranker's literal-verify,
not as an index-time callback over every token. For identity-ish normalization (ASCII +
NFC+lowercase, the common case) it is trivial. The FTS5 design forced this into an `unsafe`
tokenizer callback on every token; here it is an ordinary Rust helper invoked rarely.

---

## 6. Selection — the cost-budget pruner (query-time)

A query's full token set is the wrong thing to scan (common tokens have O(corpus) postings and
little discrimination). Selection keeps a rarest-first prefix.

### 6.1 The algorithm

```
present = query tokens with df>0, sorted by DF ascending (term tie-break)     -- rarest first
floor   = max(F, m), clamped to |present|
walk present accumulating cost = α·|kept| + β·Σdf:
  - stop at |kept| == max(k_max, floor)            -- absolute ceiling
  - else stop once |kept| >= floor AND cost >= B   -- budget binds only past the floor
append every df=0 token, kept but CHARGED NOTHING
```

- **Rarest-first is the whole point:** a low-DF token is *both* cheapest to scan and most
  discriminating — perfectly correlated.
- **`B=0` reduces to a fixed floor-sized cut**; raising `B` admits breadth in cost units
  (the binding constraint is latency, so sweeping `B` against recall traces the knee).
- **DF is read from the live cardinality column (§7.2), always fresh.** So a `df=0` token is a
  genuinely-absent one — an empty posting, free either way. (The FTS5 design read DF from a
  *lagging* snapshot, which created a "written-since-snapshot, looks absent but isn't" case
  needing special care; a live cardinality removes that subtlety entirely.)
- **Corpus size `N` is not used.** Selection derives only from this query's own token DFs,
  giving **batch == serial**: `search_batch([…,q,…])` ranks `q` identically to `search(q)`.

### 6.2 Why `F` is derived, not a knob

`F` keeps enough tokens that a typo'd query still clears the match floor `m`. A single edit
corrupts the `d ≈ 3–4` tokens spanning that character, so `F = m + d` leaves `m` intact; with
`m=2, d=4`, `F=6` — a one-typo margin. `F` is a function of `m` plus a structural constant,
computed not exposed. *Caveat:* `d` should grow with query length (more characters → more
possible edits); a constant `d` is an as-you-type approximation, documented as such.

---

## 7. Storage and ranking internals

### 7.1 Schema (SQLite, FTS5-free)

```sql
-- (Table names unprefixed here; Sidecar uses them as-is, Shared applies the Namespace prefix.)

-- key/value: schema_version, caller data_version, tokenizer_fingerprint (all drift, §8.4),
-- and rolling counters (N, etc.).
CREATE TABLE meta(key TEXT PRIMARY KEY, value);

-- one row per indexed segment. `id` is the dense integer that appears in the roaring postings.
-- `txt` holds raw text (Match.text source, delete-time re-tokenize, ranker literal-verify) in
-- the default SNAPSHOT mode; it is NULL/absent in CONTENTLESS mode (§4.1), where text comes
-- from the caller's resolver and deletion uses `fwd` below.
CREATE TABLE seg(
  id     INTEGER PRIMARY KEY,    -- the id used in postings (see "id space", §7.2)
  doc_id INTEGER NOT NULL,
  source TEXT    NOT NULL,
  ref    TEXT    NOT NULL,
  txt    TEXT                    -- NOT NULL in snapshot mode; NULL in contentless mode
);
CREATE INDEX seg_doc ON seg(doc_id, source);   -- replace-per-(doc,source), delete-by-doc

-- CONTENTLESS-mode-only forward index: each segment's token set, so delete can pull the
-- segment's id from the right postings WITHOUT text (which is gone at the source on a delete).
-- Absent entirely in snapshot mode (there, re-tokenizing `seg.txt` is the forward index).
-- Also absent in the resolver-only/ordering-protocol variant (§4.1, §12). For trigrams this
-- is roughly text-sized — the cost that partly offsets contentless's space win.
CREATE TABLE fwd(id INTEGER PRIMARY KEY, tokens BLOB NOT NULL);   -- contentless + no-protocol only

-- the DF index: effective cardinality per token, maintained transactionally. the pruner reads
-- DF from HERE (a small-int PK seek) WITHOUT touching the posting blobs.
CREATE TABLE term(term TEXT PRIMARY KEY, df INTEGER NOT NULL);

-- the owned roaring inverted index: one BASE posting per token, written only by fold/rebuild.
CREATE TABLE post(term TEXT PRIMARY KEY, base BLOB NOT NULL);

-- per-token delta, written on EVERY write (small blobs), kept separate from `post` so a write
-- never rewrites the big base. effective posting = (base ∪ added) \ removed.
CREATE TABLE delta(term TEXT PRIMARY KEY, added BLOB NOT NULL, removed BLOB NOT NULL);

-- transient shadow tables for the atomic rebuild swap: seg_shadow, term_shadow,
-- post_shadow, delta_shadow.
```

Three-way write-frequency split is deliberate: a write touches only the *small* rows
(`term.df`, `delta`), never the big `post.base`; the base is rewritten only by `compact()`
(fold) or `rebuild()`. The pruner reads only `term.df`; a query then reads `post.base` +
`delta` for the *kept* (rare) tokens only.

### 7.2 The owned roaring inverted index

Every token has a roaring posting maintained as **base + delta** (per-token added/removed
roaring bitmaps), `O(touched tokens)` per write — the same scheme the FTS5 design used for the
*rare* tier, now applied uniformly because there is no FTS5 to hold the common tokens.

- **Reconstruction:** a token resolves to `(base ∪ added) \ removed`, always fresh (the delta
  is written in the same transaction as `seg`). The result is a `RoaringBitmap` ready for the
  counter — **no decode step.** (This is the key read-path win over FTS5: an FTS5 doclist had
  to be varint-decoded into a bitmap before the counter could use it; an owned roaring posting
  *is* the bitmap, a near-zero-copy deserialize from the mmap'd BLOB.)
- **DF is a maintained cardinality**, not a recounted snapshot: `term.df` is the effective
  cardinality `|(base ∪ added) \ removed|`, updated transactionally on each write. So there is
  **no debounced DF refresh** and DF is always current.
- **The id space (a fork the pivot unlocks).** Because trifle now owns id assignment (FTS5 did
  before, and reused ids), it can choose:
  - **Monotonic ids (primary).** Never reuse a freed id. Deletion is then trivially safe — a
    stale id lingering in a posting until the next fold is harmless because the id is never
    reassigned to a different segment, so there is no "is this the old doc or a new one"
    ambiguity and **no REMOVE-before-ADD ordering discipline is needed.** Cost: the id space
    grows over deletes (roaring stays compact over sparse ranges; `rebuild()` reclaims dense
    ids). Good fit for write-infrequent + periodic rebuild.
  - **Reused ids (alternative).** Dense ids, no growth, but deletion must order REMOVE before
    ADD per `(term, id)` and delete-before-insert for last-writer-wins under reuse — the
    discipline the FTS5 design carried because FTS5 reused rowids.
  Either way, deletion needs each segment's **token set** (which postings to pull its id from).
  The source of that set depends on the text-storage mode (§4.1): in **snapshot** mode the
  writer reads the segment's `seg.txt` and re-tokenizes it (the `seg` row *is* the forward
  index); in **contentless** mode the text is gone at the source on a delete, so the token set
  comes from either the resolver under the ordering protocol (re-tokenize the still-present
  indexed text) or the stored `fwd` table (§7.1) — never a live resolver fetch of
  already-deleted text.
- **Fold (`compact()`) — and the honest cost of owning all tokens.** Folding merges each
  token's delta into its base and drops tokens whose effective posting emptied. For the *rare*
  tokens this is cheap (small bases). For **common** tokens it is not: a high-DF token's base
  is a large bitset, and folding rewrites it — work FTS5 absorbed in its own write path and
  trifle now owns. This is the price of the pivot. It is bounded and *off the hot path*
  (folding is scheduled `compact()`, the query path reads base+delta fresh regardless), and
  the write-infrequent assumption keeps churn — and therefore fold cost — low. The benchmark
  (§10) is what confirms the trade is net-favorable; do not assume it, measure it.

### 7.3 (storage size)

With FTS5 gone there is no separate inverted index *and* doclist storage. Size is: the raw
`txt` (unavoidable, same as before), the roaring postings, and small DF/provenance rows. For
the **common** tokens that dominate index size, a roaring posting (bitset ≈ N/8 bytes, or
run-containers far smaller when ids cluster — which they do under topical/deck structure) is
plausibly **~3–4× smaller** than the delta-varint doclist FTS5 stored even with `detail=none`.
So the on-disk footprint should shrink relative to the FTS5 design — *plausibly*; this is
first-principles reasoning about container sizes, and the benchmark is the arbiter.

### 7.4 Ranking — bit-sliced overlap (default) and the Ranker trait

Candidate generation is fixed and fast; final ordering is pluggable.

**Candidate generation (fixed):** read each kept token's posting (`(base ∪ added) \ removed`,
no decode), then count overlap with a **bit-sliced counter** — each id's count held in binary
across bitmap "bit planes," adding a posting is a ripple-carry binary add at the bitmap level
(XOR = sum bit, AND = carry). The whole accumulation is **O(k·log k) bitmap ops, independent
of posting size**. A high→low bucket walk **hydrates and filters bucket-by-bucket** and
**early-stops** once `limit` *accepted* results lock. The posting scan is **id-only**; per
bucket, survivors' provenance `(doc_id, source, ref, txt)` is hydrated in **one** batched
`seg` lookup via the in-memory `carray`/`rarray` vtab (`WHERE id IN …`) — *not* an `id IN (…)`
literal MATCH that builds an ephemeral temp btree faulting through SQLite's page-cache mutex
and serializing parallel reads.

**No downsample.** The FTS5 design needed a `2^18` positional downsample because a huge live
doclist read was O(posting size) to decode. Owned roaring postings feed the counter directly,
and the counter is posting-size-independent, so **the downsample is gone** — the cut is by
overlap, full stop, with no positional touch. (A very common token is a bigger bitset to
AND/XOR, but that is SIMD-fast bitwise bigness, a small constant — and the pruner drops common
tokens from queries anyway.)

**Scope and exclusion are a predicate, not a materialized set.** `SearchOpts.scope` is an
`Fn(doc_id, source, ref) -> bool`, called only over *candidates* in descending-overlap order
during the bucket walk — never over the corpus. It receives the provenance already hydrated
for that bucket (filter on origin for free) but **not** the segment text (hydrating text is the
expensive step the deferred path avoids; text filtering is a ranker concern). The walk
**continues until `limit` predicate-passing results lock**, so scoping is correct without
over-fetch-then-drop. (The predicate runs inside the read: `Send + Sync`, must not call back
into the writer; cheap in-memory predicates preferred when the scope is hot.)

The per-query overlap floor is `min(m, |selected|)`.

**Why overlap-count and not length-normalized scoring (cost is architectural, not arithmetic).**
Length normalization is a divide — cheap. What is expensive is putting it in the **primary**
key: integer overlap is what the bit-sliced counter produces and what the walk can early-stop
on; a continuous length-blended score forces scoring *every* candidate, sorting them, and
hydrating per-candidate length for all of them. At uniform small doc sizes the factor ≈ 1 for
everyone, so keeping it out of the primary key costs almost nothing. Where wanted, it is cheap
as a **within-bucket tiebreak** or a **survivor reranker** (a custom `Ranker`, or a
literal/exact tier) — using the stored `txt` length, not any maintained doc-length column.

**The Ranker trait (Strategy, query-time):**

```rust
pub trait Ranker: Send + Sync {
    /// Reorder the overlap-counted candidates. The default reranks only the
    /// top-k-by-overlap survivors; richer rankers may request fuller access.
    fn rank(&self, candidates: Candidates<'_>, query: &QueryContext) -> Vec<Ranked>;
}
```

`Candidates` exposes per candidate: overlap count, which selected tokens matched, and **lazy**
access to the segment `txt`/span. Default `OverlapRanker` orders by overlap (free — reads only
counts). A richer ranker spends more for quality: literal-verification (promoting exact
substring hits — how you recover a precision tier without bm25), proximity, or idf-weighting,
over the `txt` the candidate carries. To preserve the fast path, the default contract is
**rerank the survivors** (hydrates only `limit`); full-candidate-set access is an explicit,
expensive opt-in.

### 7.5 Concurrency

One **write connection** (`Mutex<Connection>`, WAL, `synchronous=NORMAL`, `busy_timeout`,
`mmap_size`) plus a **read-only pool** (`READ_ONLY | NO_MUTEX`, opened on demand, self-bounding
to the caller's parallelism). WAL lets pooled reads run concurrently with the single writer.
For parallel-read throughput: process-wide `memstatus` off (per-page allocs don't serialize on
the static-mem mutex) and `mmap` serving pages straight from the mapped file (bypassing the
shared page-cache mutex) — which is why the read path keeps no temp btrees (provenance via
`carray`, scope a predicate). **Dropping FTS5 simplifies this:** no virtual table means no
`SQLITE_SCHEMA` cookie churn on the rebuild swap (the FTS5 vtable had to be reconstructed on a
fresh pool read after a swap), so the schema-change retry the FTS5 design needed is gone.

### 7.6 Threading model — synchronous, no async

No method is or needs to be `async`: SQLite is synchronous and in-process; reads are CPU plus
`mmap` page-faults, writes synchronous commits. A caller on an async runtime dispatches calls
to a blocking pool (`spawn_blocking`) — its integration, so trifle imposes no runtime
dependency. There is also no longer a debounced background refresh to fire (DF is live, §7.2),
so trifle spawns no threads of its own; `compact()` is caller-scheduled.

---

## 8. Maintenance and lifecycle

### 8.1 Inline maintenance (no debounced refresh)

Each write maintains its postings inline: re-tokenize the added/removed text, update each
touched token's `delta` and `term.df` transactionally. There is **no debounced snapshot** — DF
is live and the delta is fresh on read — so unlike the FTS5 design there is no refresh window,
burst threshold, or background timer. The only deferred work is folding deltas into bases,
which is explicit:

### 8.2 Exposed maintenance operations

- **`compact()`** — fold each `delta` into its `base`, drop tokens whose effective posting
  emptied, and reclaim space (VACUUM-class). Bounds delta growth and posting fragmentation.
  Heavier for common tokens (§7.2); call on a schedule or when idle. Returns `CompactStats`
  (tokens folded, ids purged, bytes reclaimed).
- **`rebuild(corpus)`** — full reindex via the **atomic shadow swap** (§8.5). Required on a
  tokenizer change, on schema-version or `data_version` drift, on corruption, and useful to
  reclaim a grown monotonic id space (§7.2).
- **`stats()`** — read-only, the single observability call: `N`; total token count;
  total delta backlog (the signal for *when* to `compact`); on-disk sizes (so you can watch
  the footprint); last-write / staleness; `data_version`.

### 8.3 Write discipline

trifle is **single-writer**: the write connection is internal and serialized; a second
concurrent writer is a `SQLITE_BUSY` bug. A caller with an async write pipeline serializes into
trifle's writer; surfacing any "the index lags the source of truth" advisory is the caller's
(trifle exposes freshness facts via `stats()`, not a staleness policy).

### 8.4 Drift and the version token

The cache rebuilds when any of three `meta` stamps disagree with the present: **`data_version`**
(opaque, caller-supplied), **`tokenizer_fingerprint`** (`Tokenizer::fingerprint()` — the
postings are keyed by the tokenizer's build-time behavior, so any change, *including a patch*,
must rebuild), and **`SCHEMA_VERSION`** (trifle's on-disk format). Any mismatch (or an internal
`seg`↔posting desync detected at open) drops everything and rebuilds — **no migrations**;
trifle is a rebuildable cache.

### 8.5 Rebuild — the atomic shadow swap

Build the new index into `*_shadow` tables off the live ones (the write lock is never held
across a streaming pull), then swap in one short transaction: drop the live tables, rename the
shadows in, recreate indexes, stamp the new versions **last**. A reader sees **complete-old or
complete-new, never partial** — no empty-recall window. A crash before the swap discards the
shadow and leaves the old index intact → it rebuilds on next open. (Simpler than the FTS5
design: no `optimize` pass and no virtual-table reconstruction.)

---

## 9. The boundary — what trifle returns vs what the caller owns

trifle returns a **ranked `Vec<Match>`** (`doc_id`, `source`, `ref`, byte `span`, the whole
matched segment `text`; rank = position). Deliberately above the boundary:

- **A precision tier beyond overlap** — a custom `Ranker` (§7.4), or caller post-processing.
  trifle hands back the segment `text` so the caller can literal-verify (a pure-Rust
  `contains`/`memmem` over the stored raw text now — no FTS5 round-trip), proximity-score, or
  otherwise refine without a second read.
- **Fusion** with other signals (semantic, etc.) — the caller's; trifle supplies rank order.
- **Sub-trigram (`<3`-char) queries** — yield no tokens; the caller falls back (host-specific).
- **Staleness policy** relative to the source of truth — the caller's (§8.3–8.4).

Keeping these out is what makes trifle a reusable lexical component rather than a host-shaped
one.

---

## 10. Benchmarking, tuning, and validation

Two distinct benchmarks hide under one word — latency/throughput (no labels needed; realistic
corpus + realistic query commonness) and quality/recall (needs relevance judgments) — and
conflating them is how you measure the wrong thing.

### 10.1 What to measure, and against what

For **quality**, the baseline is **BM25, not dense retrieval.** trifle is a *lexical* engine;
its claim is "BM25-ish lexical recall, typo-tolerant, much faster." Measure how close trifle's
recall@k gets to BM25's, how much faster, and how much extra recall typo-tolerance buys.
Comparing to an embedding model is a category error.

**The comparison set is a footrace and a matrix** (specific projects illustrative — survey the
current ecosystem; the *categories* are stable):

- **The fair footrace** — same task, same corpus/queries, end-to-end: **FTS5-trigram (bm25)**
  and **pg_trgm** (the in-DB cousins), **Tantivy + Levenshtein** (durable, embedded, does
  more), and a **`LIKE '%…%'` scan** as the naive floor. Report the **hidden axes** a latency
  table omits: durability, footprint *kind* (disk vs RAM), incremental-update cost, matching
  semantics. (Note: FTS5-trigram is now an *external* baseline, not an internal dependency —
  the pivot makes it a thing to beat, not a thing to bridge to.)
- **The matrix, not the race** — in-memory subsequence filters (fzf, nucleo, fuzzy-matcher —
  they will out-latency trifle on their RAM-resident, rebuild-on-startup, subsequence task) and
  immutable fuzzy primitives (fst, SymSpell, strsim — key-oriented, rebuild-to-update). These
  draw the **category boundary**; footracing them is a category error.

**Matrix axes:** durable? · embedded / no server? · incremental update vs rebuild? · corpus-scale
(100k+ small docs)? · doc-oriented with provenance? · matching semantics (overlap / Levenshtein
/ subsequence / delete-neighborhood)? · footprint (disk vs RAM). Fill every row honestly,
**including the cells where trifle loses** (raw latency to in-memory filters; features to
Tantivy) — that is what makes the real claim credible.

**The real claim** is not "fastest fuzzy search" (an in-memory filter beats it on its turf) but
**ownership of an underserved cell:** durable + embedded + incrementally-updatable +
corpus-scale fuzzy, with provenance, fast enough to feel instant. A speed *superlative* is a
claim to **earn, not assert** — it ships in the README only once the footrace backs it, with a
link to a rerunnable benchmark.

### 10.2 The latency harness and its instrumentation

Run a fixed query set (serial *and* concurrent — the read pool's parallelism is a distinct
axis) reporting p50/p90/p99/max and throughput.

**The scaling sweep is the architectural claim.** trifle's central promise is **flatness** —
bit-sliced overlap is posting-size-independent, DF reads are PK seeks — so latency should stay
near-flat as the corpus grows. That is a *curve*; sweep 10k / 50k / 100k / 500k / 1M. A flat
curve earns the claim; a degrading one says the assumption broke (and the profile below usually
says where).

**Instrumentation — and how the pivot changes it.** The FTS5 design's dominant tail driver was
the live-doclist decode (the `C` bimodality: bitmap tier vs live read). **That path is gone** —
there is one read path now, and postings feed the counter without decode. So the tail should
*flatten substantially*; the residual variance is big-bitset AND/XOR cost plus hydration depth.
Tag each query with **Σ(kept-posting cardinality)** (and container-bytes-touched if you can) and
correlate with latency: if the p99 queries are the high-cardinality ones, the residual tail is
the big-bitset cost (a small constant, expected); if not, look at hydration, the predicate, or
FFI/marshalling. There is no longer a `C`/sidecar/tail tradeoff to instrument — the cutoff knob
that produced it is gone (§3).

### 10.3 Tuning the dials

Tune `m`, `B` against the **adversarial** quality eval — exact-match recall is floor-guaranteed,
so a clean-query eval tunes nothing. It must contain **corrupted targets** (1–2 char edits —
typo recall is what `F=m+d` exists for) and **near-match distractors** (shared sub-word — what
`B` exists for). Sweeping `B` traces the recall/latency knee; `m` trades precision against
recall. Report 1-edit and 2-edit recall separately (`F` is tuned for ≈1 typo). Note there is no
`C` to sweep for the latency/footprint tradeoff anymore — that tradeoff left with the cutoff.

### 10.4 Corpora — and why no single one suffices

- **MS MARCO passage (subsample ~100k)** — strongest single fit for latency *and* the BM25
  baseline: real Bing queries (realistic commonness), short passages, relevance labels.
  Subsample to match the deployment's segment-length distribution.
- **Entity corpus + typo injection (GeoNames / Wikidata labels)** — the fuzzy eval MS MARCO
  can't give (real queries are mostly well-spelled): real vocabulary, **labels generate
  themselves** (§10.5).
- **Synthetic deck-structured corpus (with *real* text in the clusters)** — keep it; the only
  instrument that models **deck/sub-vocabulary structure** and the **scope-predicate** path,
  which both real corpora are flat on. Fix the original by sampling *real* sentences into the
  clusters (authentic trigram distributions within each cluster) rather than discarding it.
- **Real shared decks (AnkiWeb), if the host is Anki** — highest deployment fidelity; a final
  transfer check, even with hand-written queries.

### 10.5 The typo-injection harness (free-label fuzzy eval)

Pick a known doc/entity, inject `k` single-character edits to form the **query**; the source is
the ground-truth answer (free label). Specifics: the four edits — substitution, insertion,
deletion, transposition — weighted toward realistic typos (transpositions, adjacent-key
substitutions); control `k` and report 1- vs 2-edit recall; keep the post-edit query `≥
MIN_NGRAM`; index **near-match distractors** so the eval isn't trivially solved.

### 10.6 Caveats

Nobody's queries are *your users'* queries (logs are the hardest artifact — privacy). MS MARCO's
distribution is a proxy for "natural search"; typo injection a proxy for "autocomplete." Weight
the **relative** signal (trifle vs BM25; with-typos vs without; tail vs median; and now,
*owned-index vs the FTS5 version* — §12) over absolute numbers.

---

## 11. Naming, licensing, publication

`trifle` — trigram + "a trivial, lightweight thing." The modesty lives in the name; the README
sells the regime (small documents, honest-not-bm25 ranking) without a speed claim until §10
backs it. Reserve the name with a `0.0.1` placeholder; `cargo publish --dry-run` first.
Stabilize the API only once a second real consumer has exercised the tokenizer, ranker, and
backend traits — those are the seams most likely to move.

**Licensing** (not legal advice; tooling output is the source of truth). Ship **dual `MIT OR
Apache-2.0`** — the ecosystem norm (MIT simple + GPLv2-compatible; Apache-2.0 adds a patent
grant; `OR` lets a downstream pick). Dependencies don't block it: rusqlite, libsqlite3-sys,
roaring, unicode-normalization and the rest are permissive, and bundled SQLite is public domain
— so nothing forces copyleft. The only thing that would is a copyleft transitive dep, so
**verify, don't assume**: `cargo deny check licenses` against a permissive allowlist in CI from
day one. The one non-obvious obligation is the **Unicode data license** (`unicode-normalization`
embeds UCD tables — permissive, attribution to preserve, easy to miss because the *crate* is
MIT/Apache while the *data* isn't). Attribution attaches at **binary**-distribution time (the
app embedding trifle, not the crate on crates.io) — generate an aggregated notice with
`cargo-about`. Set `license = "MIT OR Apache-2.0"` + both `LICENSE-*` files **at the `0.0.1`
reservation, before external contributions** — relicensing later needs every contributor's
consent. (Removing the FTS5 custom-tokenizer FFI also removes any question about distributing
hand-rolled bindings to SQLite's C interface — a minor licensing simplification, since SQLite
is public domain anyway.)

---

## 12. Alternatives considered and open forks

**The FTS5-based design (superseded).** The prior revision used an **FTS5 trigram index as the
base store and posting source**, with a roaring base+delta tier maintained *only* for tokens
below a materialization cutoff `C`; common tokens fell to a **live FTS5 doclist read**. It
required: a **custom FTS5 tokenizer registered via hand-rolled `unsafe` FFI** (`fts5_api` /
`xCreate`/`xTokenize`) to make FTS5's tokenization agree with trifle's — with an index-time
**offset-map** subsystem and **per-connection registration**; a **relative cutoff `C = p·N`**
and a derived **grace band `g`** to anti-thrash the materialize/de-materialize boundary; a
**debounced DF snapshot** (since recounting FTS5 doclists per query is expensive); and a
**dual read path**. It was rejected because the single hardest, most dangerous piece of it (the
`unsafe` tokenizer bridge) existed *solely* as glue to reconcile an inherited dependency, and
because the design was reached by incrementally layering trigram/roaring machinery onto an FTS5
substring engine that was the original *naive* choice — a textbook local minimum. The owned
roaring index deletes the bridge, `C`, `g`, the dual path, the downsample, and the debounced
refresh, in exchange for owning common-token posting maintenance (§7.2). *This doc preserves the
FTS5 approach as the documented alternative; the full prior draft is archived separately.*

**Open fork — postings in SQLite BLOBs (primary) vs an owned mmap file.** The primary design
keeps postings as BLOBs in SQLite (one file, fully transactional, simplest). But once FTS5 is
gone the postings are just a token→blob map, and SQLite is a lot of machinery (planner, btree)
for `get`/`put`. A coherent alternative: keep **text + provenance in SQLite** (where rows,
joins, and transactions are genuinely wanted) and put the **postings in an owned append-only
memory-mapped file** with an offset index — a posting read becomes a raw `mmap` offset, zero
SQL, full layout control, plausibly faster and smaller. The cost is owning a tiny storage format
(durability/fsync discipline, the offset index, crash-consistency, compaction) instead of
leaning on SQLite's. The §10 spike is what decides; this is recorded, not committed.

**Open fork — text storage: snapshot (default) vs contentless resolver vs stored token-set
(§4.1).** trifle copies each segment's text into `seg.txt` so the index is self-contained and
deletion is correct by construction. In `Shared` mode that copy is pure duplication (the text
is in the same file), motivating a **contentless** mode that references the caller's text via a
resolver. The catch is deletion: a resolver returns *current* source text, but on a delete the
source text is gone, so contentless deletion needs either a **caller ordering protocol**
(invalidate trifle before the source write — real space win, sharp correctness contract) or a
**stored per-segment token set** (self-contained, but ~text-sized for trigrams, so it largely
*offsets* the space win it was meant to deliver). So the honest accounting: contentless pays off
mainly under the protocol; the token-set variant mostly relocates the bytes rather than
eliminating them. Snapshot stays the default (self-contained, no contract); contentless is the
`Shared` specialization. There may be a representation that serves both the per-token postings
and the per-segment token-set from shared data (they are transposes) — a spike question, not a
commitment.
data the wrong direction: `rarray` *windows Rust-resident data into SQL*, whereas trifle needs
*SQLite-resident postings in Rust* for the bit-sliced counter. Roaring intersection is not
expressible in SQL, so SQLite would iterate ids per-tuple — exactly the per-row cost bit-sliced
counting exists to avoid — effectively unwinding roaring back into a row stream. And the
"marshalling" it would avoid is already a free `mmap`-pointer dereference plus a near-zero-copy
roaring deserialize (same process, same mapped bytes). The genuinely useful move in that
direction is *less* SQLite (the owned-mmap fork above), not a vtab over it.

**Low-priority — a roaring-merge SQL *function* extension, for writes.** Registering
`roaring_or`/`roaring_merge` SQL functions so a delta-fold runs inside an `UPDATE` (rather than
round-tripping postings to Rust) is genuinely apt and a real pattern — but it optimizes
`compact()`, which is scheduled and off the hot path, while the per-write delta is already
`O(touched)` and tiny. A nice-to-have that trades `unsafe` extension code for a maintenance-path
speedup few callers will feel; filed under possible, not architectural.

**Not reconsidered (fundamental, not inherited).** Trigrams, rarest-first pruning, and
bit-sliced overlap are intrinsic to durable incremental fuzzy search and justified on their own
terms; SQLite-as-store earns its place (durability, mmap, co-location). FTS5 was the one piece
that was incidental to having *started* with a full-text substring engine — which is exactly
why it was the right thing to cut.

---

## Appendix A — defaults at a glance

| Symbol | Meaning | Default |
|---|---|---|
| `m` | match floor (shared rare tokens for a hit) | `2` |
| `B` | breadth budget (selection cost units) | `0` |
| `F` | typo floor (derived) | `m + d = 6` |
| `d` | per-typo token damage | `4` |
| `k_max` | selection cap | `12` |
| `α, β` | cost coefficients (`cost = α·|kept| + β·Σdf`) | `0, 1` |
| `MIN_NGRAM` | tokenizer window (Unicode scalars) | `3` |

*Gone with the FTS5 pivot:* `C` (materialization cutoff), `g` (grace band), `p` (cutoff
fraction), the posting downsample ceiling, the refresh-debounce constants. The performance
surface shrank with the architecture.
