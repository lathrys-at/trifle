//! `trifle-lean` — a viability spike for the rev v0.3 storage + streaming-overlap shape.
//!
//! This is **not** production trifle. It is the smallest end-to-end slice that proves the
//! shape of PROPOSAL.md Layers 2 & 3 is viable and benchmarkable:
//!
//! - **Flattened single `seg` table** (`id, key, label, txt`) — no `doc` table, so the
//!   no-ghost invariant is trivially true (§3.1).
//! - The storage layer feeds owned roaring postings into [`trifle_overlap::Counter`] and exposes
//!   a **lazy [`CandidateStream`]** that owns the connection + the counter with **no
//!   self-referential lifetime** — achieved by issuing `BEGIN`/`ROLLBACK` manually on the owned
//!   connection and *never storing a `Transaction` object* (the trap that would otherwise force
//!   `ouroboros`). This is the concrete proof of fork F4 / §3.2.
//! - **Provenance-only [`Candidate`] + batched [`CandidateStream::hydrate`]** (§3.4): the stream
//!   yields candidates without text; the caller hydrates only the ones it keeps, in one batched
//!   `WHERE id IN rarray(?)` read.
//! - The **opt-in raw-SQL filter** (§4): one `Filter { fragment, params }` folded into the
//!   per-chunk provenance query as `WHERE (<fragment>) AND id IN rarray(?{N+1})` — **fragment
//!   first, scope param last** (fork F3), so both numbered `?1..?N` and anonymous `?` bind with
//!   no collision.
//!
//! Deliberately out of spike scope (bedrock, already proven in trifle / mechanical): posting
//! (de)serialization to SQLite (postings are kept in memory here), the read-connection pool,
//! the dict-generation guard, drift/rebuild, replace-on-write, Unicode normalization, spans.
//! The dict + postings live in memory exactly as the real design intends; only the *novel*
//! storage→stream→filter wiring is exercised against SQLite.

use std::collections::{HashMap, HashSet, VecDeque};
use std::rc::Rc;
use std::sync::Mutex;

use roaring::RoaringBitmap;
use rusqlite::types::Value;
use rusqlite::{Connection, ToSql};
use trifle_overlap::{Counter, Scored};

/// Result alias for the spike.
pub type Result<T> = rusqlite::Result<T>;

/// A hydrated match (key + provenance + score + text). Rank is conveyed by position.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Match {
    pub key: i64,
    pub label: String,
    pub score: u32,
    pub overlap: u32,
    pub text: String,
}

/// A scored, provenance-only candidate (no text — see [`CandidateStream::hydrate`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Candidate {
    pub key: i64,
    pub label: String,
    pub seg_id: u32,
    pub score: u32,
    pub overlap: u32,
}

/// The opt-in raw-SQL filter: a trusted-constant fragment over the `seg` columns (`key`,
/// `label`, `txt`) — or, in a real deployment, co-located caller tables via `ATTACH` — plus its
/// bound params. Folded into the provenance query as `WHERE (<fragment>) AND id IN
/// rarray(?{N+1})`. Reference your params as `?1..?N` (numbered, reusable) or anonymous `?`.
pub struct Filter<'a> {
    pub fragment: &'a str,
    pub params: &'a [&'a dyn ToSql],
}

/// Per-search knobs. `limit` is a terminal-op argument (on [`LeanIndex::matches`]), not here —
/// the [`CandidateStream`] is lazy/unbounded.
#[derive(Clone, Copy, Debug)]
pub struct SearchOpts {
    /// `m` — the raw shared-trigram floor.
    pub min_shared: u32,
    /// How many rarest query trigrams selection keeps (the recall/cost knob).
    pub t_max: usize,
    /// `D` — df-doublings per IDF weight step.
    pub weight_step: f64,
}

impl Default for SearchOpts {
    fn default() -> Self {
        SearchOpts {
            min_shared: 2,
            t_max: 12,
            weight_step: 1.0,
        }
    }
}

/// The lean index: a flattened `seg` table in SQLite + an in-memory trigram dictionary and
/// roaring inverted index.
pub struct LeanIndex {
    conn: Mutex<Connection>,
    /// trigram → term id.
    dict: Mutex<HashMap<String, u32>>,
    /// term id → the seg ids containing that trigram (the owned roaring inverted index).
    postings: Mutex<HashMap<u32, RoaringBitmap>>,
    next_seg: Mutex<u32>,
}

impl LeanIndex {
    /// Open an in-memory index (spike convenience).
    pub fn open_in_memory() -> Result<Self> {
        Self::from_conn(Connection::open_in_memory()?)
    }

    fn from_conn(conn: Connection) -> Result<Self> {
        rusqlite::vtab::array::load_module(&conn)?; // enables `rarray`
        conn.execute_batch(
            "CREATE TABLE seg(
                 id    INTEGER PRIMARY KEY,
                 key   INTEGER NOT NULL,
                 label TEXT NOT NULL,
                 txt   TEXT NOT NULL
             );
             CREATE INDEX seg_by_key ON seg(key);",
        )?;
        Ok(LeanIndex {
            conn: Mutex::new(conn),
            dict: Mutex::new(HashMap::new()),
            postings: Mutex::new(HashMap::new()),
            next_seg: Mutex::new(1),
        })
    }

    /// Index one `(key, label) = text` segment. Append-only in the spike (replace-on-write is
    /// out of scope — it is mechanical, not novel).
    pub fn insert(&self, key: i64, label: &str, text: &str) -> Result<()> {
        let seg_id = {
            let mut n = self.next_seg.lock().unwrap();
            let id = *n;
            *n += 1;
            id
        };
        {
            let conn = self.conn.lock().unwrap();
            conn.execute(
                "INSERT INTO seg(id, key, label, txt) VALUES(?1, ?2, ?3, ?4)",
                rusqlite::params![seg_id as i64, key, label, text],
            )?;
        }
        let mut dict = self.dict.lock().unwrap();
        let mut postings = self.postings.lock().unwrap();
        for gram in distinct_trigrams(text) {
            let next = dict.len() as u32 + 1;
            let tid = *dict.entry(gram).or_insert(next);
            postings.entry(tid).or_default().insert(seg_id);
        }
        Ok(())
    }

    /// Resolve a query to its selected (rarest-first) postings, owned for the engine.
    fn select_postings(&self, query: &str, t_max: usize) -> Vec<RoaringBitmap> {
        let dict = self.dict.lock().unwrap();
        let postings = self.postings.lock().unwrap();
        // distinct query trigrams → (term id, df) for the ones present in the corpus.
        let mut present: Vec<(u32, u64)> = distinct_trigrams(query)
            .into_iter()
            .filter_map(|g| dict.get(&g).copied())
            .filter_map(|tid| postings.get(&tid).map(|bm| (tid, bm.len())))
            .collect();
        // rarest-first (smallest df), deterministic tie-break by term id.
        present.sort_unstable_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));
        present.truncate(t_max.max(1));
        present
            .into_iter()
            .map(|(tid, _)| postings.get(&tid).cloned().unwrap_or_default())
            .collect()
    }

    /// Open a lazy candidate stream for `query`. The stream owns a pinned read transaction on
    /// the connection (manual `BEGIN`; `ROLLBACK` on drop) and the engine [`Counter`] — with no
    /// self-referential lifetime, because it never stores a `Transaction` object.
    pub fn candidates(&self, query: &str, opts: SearchOpts) -> Result<CandidateStream<'_>> {
        let postings = self.select_postings(query, opts.t_max);
        let counter = Counter::build(postings, opts.weight_step, opts.min_shared);
        let guard = self.conn.lock().unwrap();
        guard.execute_batch("BEGIN DEFERRED")?; // pin a snapshot for the stream's life
        let walk = counter.walk();
        Ok(CandidateStream {
            guard,
            counter,
            walk,
            ready: VecDeque::new(),
            seen: HashSet::new(),
            done: false,
            errored: false,
        })
    }

    /// Eager top-`limit` matches (the safe default front door): pull candidates, then hydrate
    /// text for exactly the top `limit` in one batched read.
    pub fn matches(
        &self,
        query: &str,
        opts: SearchOpts,
        limit: usize,
        filter: Option<&Filter<'_>>,
    ) -> Result<Vec<Match>> {
        let mut stream = self.candidates(query, opts)?;
        let mut kept: Vec<Candidate> = Vec::with_capacity(limit);
        while kept.len() < limit {
            match stream.next_filtered(filter) {
                Some(Ok(c)) => kept.push(c),
                Some(Err(e)) => return Err(e),
                None => break,
            }
        }
        stream.hydrate(&kept)
    }
}

/// A lazy, snapshot-pinned candidate cursor. Owns the connection guard **and** the engine
/// [`Counter`]; drives the bit-sliced walk, batch-hydrates provenance (+ applies the raw-SQL
/// filter) per chunk, dedups to one candidate per key, best-first.
///
/// No self-referential lifetime: the `Counter` owns its postings (moved in), and the snapshot
/// is held via a manual `BEGIN`/`ROLLBACK` on `guard` rather than a stored `Transaction`.
pub struct CandidateStream<'a> {
    guard: std::sync::MutexGuard<'a, Connection>,
    counter: Counter,
    walk: trifle_overlap::Walk,
    ready: VecDeque<Candidate>,
    seen: HashSet<i64>,
    done: bool,
    errored: bool,
}

/// How many engine candidates to pull per provenance/filter round-trip.
const CHUNK: usize = 64;

impl CandidateStream<'_> {
    /// Pull the next candidate, applying `filter` (if any). Fuses on the first error so a
    /// caller never gets a deceptively-complete prefix after a transient failure (§6).
    pub fn next_filtered(&mut self, filter: Option<&Filter<'_>>) -> Option<Result<Candidate>> {
        loop {
            if let Some(c) = self.ready.pop_front() {
                return Some(Ok(c));
            }
            if self.done || self.errored {
                return None;
            }
            if let Err(e) = self.refill(filter) {
                self.errored = true;
                return Some(Err(e));
            }
        }
    }

    /// Refill `ready` with one chunk: pull up to [`CHUNK`] scored ids from the engine, run one
    /// provenance(+filter) query over them, dedup by key, queue the survivors in score order.
    fn refill(&mut self, filter: Option<&Filter<'_>>) -> Result<()> {
        let mut scored: Vec<Scored> = Vec::with_capacity(CHUNK);
        while scored.len() < CHUNK {
            match self.counter.advance(&mut self.walk) {
                Some(s) => scored.push(s),
                None => {
                    self.done = true;
                    break;
                }
            }
        }
        if scored.is_empty() {
            return Ok(());
        }
        let seg_ids: Vec<u32> = scored.iter().map(|s| s.id).collect();
        let prov = self.provenance(&seg_ids, filter)?;
        for s in scored {
            if let Some((key, label)) = prov.get(&s.id) {
                if self.seen.insert(*key) {
                    self.ready.push_back(Candidate {
                        key: *key,
                        label: label.clone(),
                        seg_id: s.id,
                        score: s.score,
                        overlap: s.overlap,
                    });
                }
            }
        }
        Ok(())
    }

    /// One batched provenance + filter query over a chunk's seg ids. Fragment first, candidate
    /// scope param last (`?{N+1}`), so the caller's `?1..?N` never collide with the scope param.
    fn provenance(
        &self,
        seg_ids: &[u32],
        filter: Option<&Filter<'_>>,
    ) -> Result<HashMap<u32, (i64, String)>> {
        let arr: Rc<Vec<Value>> = Rc::new(seg_ids.iter().map(|&i| Value::Integer(i as i64)).collect());
        let n = filter.map_or(0, |f| f.params.len());
        let sql = match filter {
            Some(f) => format!(
                "SELECT id, key, label FROM seg WHERE ({frag}) AND id IN rarray(?{scope})",
                frag = f.fragment,
                scope = n + 1,
            ),
            None => "SELECT id, key, label FROM seg WHERE id IN rarray(?1)".to_string(),
        };
        let mut binds: Vec<&dyn ToSql> = Vec::with_capacity(n + 1);
        if let Some(f) = filter {
            binds.extend_from_slice(f.params); // ?1..?N
        }
        binds.push(&arr); // ?{N+1}

        let mut stmt = self.guard.prepare_cached(&sql)?;
        let mut out = HashMap::new();
        let mut rows = stmt.query(binds.as_slice())?;
        while let Some(r) = rows.next()? {
            let id: i64 = r.get(0)?;
            let key: i64 = r.get(1)?;
            let label: String = r.get(2)?;
            out.insert(id as u32, (key, label));
        }
        Ok(out)
    }

    /// Hydrate text for exactly the given candidates in ONE batched `WHERE id IN rarray(?1)`
    /// read (the terminal step that builds `Match`es). A pull-many/keep-few caller hydrates only
    /// what it kept.
    pub fn hydrate(&self, kept: &[Candidate]) -> Result<Vec<Match>> {
        if kept.is_empty() {
            return Ok(Vec::new());
        }
        let arr: Rc<Vec<Value>> =
            Rc::new(kept.iter().map(|c| Value::Integer(c.seg_id as i64)).collect());
        let mut stmt = self
            .guard
            .prepare_cached("SELECT id, txt FROM seg WHERE id IN rarray(?1)")?;
        let mut txt: HashMap<u32, String> = HashMap::new();
        let mut rows = stmt.query(rusqlite::params![arr])?;
        while let Some(r) = rows.next()? {
            let id: i64 = r.get(0)?;
            let t: String = r.get(1)?;
            txt.insert(id as u32, t);
        }
        Ok(kept
            .iter()
            .map(|c| Match {
                key: c.key,
                label: c.label.clone(),
                score: c.score,
                overlap: c.overlap,
                text: txt.get(&c.seg_id).cloned().unwrap_or_default(),
            })
            .collect())
    }
}

impl Iterator for CandidateStream<'_> {
    type Item = Result<Candidate>;
    /// Unfiltered iteration (compose `filter`/`take` on top). For a filtered stream use
    /// [`CandidateStream::next_filtered`].
    fn next(&mut self) -> Option<Result<Candidate>> {
        self.next_filtered(None)
    }
}

impl Drop for CandidateStream<'_> {
    fn drop(&mut self) {
        // Release the pinned snapshot. Best-effort: if it fails the connection is dropped/reset
        // by the pool in a real deployment (here the Mutex just unlocks).
        let _ = self.guard.execute_batch("ROLLBACK");
    }
}

/// Distinct lowercased character trigrams of `text` (a sliding 3-codepoint window). Strings
/// shorter than 3 codepoints yield a single whole-string gram. Minimal spike tokenizer — the
/// real `NgramTokenizer` (normalization, script segmentation) is bedrock, out of scope here.
fn distinct_trigrams(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.to_lowercase().chars().collect();
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    if chars.is_empty() {
        return out;
    }
    if chars.len() < 3 {
        out.push(chars.iter().collect());
        return out;
    }
    for w in chars.windows(3) {
        let g: String = w.iter().collect();
        if seen.insert(g.clone()) {
            out.push(g);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> LeanIndex {
        let idx = LeanIndex::open_in_memory().unwrap();
        idx.insert(1, "front", "the quick brown fox").unwrap();
        idx.insert(2, "front", "the quack brown ox").unwrap();
        idx.insert(3, "front", "lazy dog sleeping").unwrap();
        idx.insert(4, "front", "quick silver fox").unwrap();
        idx
    }

    #[test]
    fn typo_query_ranks_relevant_docs() {
        let idx = fixture();
        let hits = idx
            .matches("quikc brown", SearchOpts::default(), 10, None)
            .unwrap();
        let keys: Vec<i64> = hits.iter().map(|m| m.key).collect();
        // docs 1 and 2 share "brown" + most of "quick/quack"; doc 3 shares nothing.
        assert!(keys.contains(&1), "got {keys:?}");
        assert!(keys.contains(&2), "got {keys:?}");
        assert!(!keys.contains(&3), "doc 3 shares no trigrams: {keys:?}");
        // each result carries its hydrated text and a score.
        assert!(hits.iter().all(|m| !m.text.is_empty() && m.score >= m.overlap));
    }

    #[test]
    fn dedup_one_result_per_key() {
        let idx = LeanIndex::open_in_memory().unwrap();
        idx.insert(7, "front", "alpha bravo").unwrap();
        idx.insert(7, "back", "alpha charlie").unwrap(); // same key, 2 segments
        let hits = idx
            .matches("alpha", SearchOpts::default(), 10, None)
            .unwrap();
        assert_eq!(hits.iter().filter(|m| m.key == 7).count(), 1, "deduped by key");
    }

    #[test]
    fn raw_sql_filter_numbered_placeholder_and_key_set() {
        // The `key IN rarray(?1)` universal filter mode (caller's own allowed-key set), with a
        // numbered placeholder reused implicitly. Restrict to keys {2,4}.
        let idx = fixture();
        let allowed: Rc<Vec<Value>> = Rc::new(vec![Value::Integer(2), Value::Integer(4)]);
        let params: Vec<&dyn ToSql> = vec![&allowed];
        let filter = Filter {
            fragment: "key IN rarray(?1)",
            params: &params,
        };
        let hits = idx
            .matches("quick brown", SearchOpts::default(), 10, Some(&filter))
            .unwrap();
        let keys: Vec<i64> = hits.iter().map(|m| m.key).collect();
        assert!(keys.iter().all(|k| *k == 2 || *k == 4), "filtered to {{2,4}}: {keys:?}");
        assert!(!keys.contains(&1), "doc 1 excluded by filter: {keys:?}");
    }

    #[test]
    fn raw_sql_filter_anonymous_placeholder_no_collision() {
        // An anonymous `?` fragment must bind correctly alongside the scope param at `?{N+1}`
        // (fragment textually first). This is the F3 audit-footgun fix.
        let idx = fixture();
        let pat = "%fox%";
        let params: Vec<&dyn ToSql> = vec![&pat];
        let filter = Filter {
            fragment: "txt LIKE ?",
            params: &params,
        };
        let hits = idx
            .matches("quick", SearchOpts::default(), 10, Some(&filter))
            .unwrap();
        let keys: Vec<i64> = hits.iter().map(|m| m.key).collect();
        // "quick silver fox" (4) and "the quick brown fox" (1) contain "fox"; both share "quick".
        assert!(keys.contains(&4), "got {keys:?}");
        assert!(keys.iter().all(|k| *k == 1 || *k == 4), "only fox docs: {keys:?}");
    }

    #[test]
    fn numbered_placeholder_reuse_does_not_collide_with_scope() {
        // `?1` used TWICE in the fragment — impossible under the old anonymous-only scheme —
        // must bind to the caller's param, not the scope rarray.
        let idx = fixture();
        let kv = 4i64;
        let params: Vec<&dyn ToSql> = vec![&kv];
        let filter = Filter {
            fragment: "key = ?1 OR key = ?1",
            params: &params,
        };
        let hits = idx
            .matches("quick fox", SearchOpts::default(), 10, Some(&filter))
            .unwrap();
        assert!(hits.iter().all(|m| m.key == 4), "reused ?1 bound to 4: {hits:?}");
    }

    #[test]
    fn streaming_choose_then_hydrate() {
        // Pull provenance-only candidates lazily, keep a subset, hydrate only those.
        let idx = fixture();
        let mut stream = idx.candidates("quick brown fox", SearchOpts::default()).unwrap();
        let mut pool: Vec<Candidate> = Vec::new();
        for item in stream.by_ref().take(10) {
            pool.push(item.unwrap()); // propagate errors, never filter_map(Result::ok)
        }
        assert!(!pool.is_empty());
        // candidates carry no text (provenance only).
        let keep: Vec<Candidate> = pool.into_iter().take(2).collect();
        let hits = stream.hydrate(&keep).unwrap();
        assert_eq!(hits.len(), keep.len());
        assert!(hits.iter().all(|m| !m.text.is_empty()));
    }
}
