//! Query generation.
//!
//! - **Latency** ([`perf_queries`]): clean in-corpus document snippets (no typos — latency is
//!   a pure speed measurement). No labels — the snippet's vocabulary/co-occurrence are exactly
//!   the corpus's, which sidesteps "where do realistic queries come from".
//! - **Fuzzy/typo recall** ([`fuzzy_queries`]): an **entity name + injected edits**,
//!   labeled by the entity. On an entity corpus this construction is faithful — the user
//!   types a corrupted target name and wants the target — unlike the same trick on prose
//!   passages, where it degenerates into a known-item smoke test.
//!
//! (The MS MARCO **relevance** eval uses real dev queries + qrels — those aren't
//! generated here; see [`corpus::msmarco_relevance`](crate::corpus::msmarco_relevance).)

use crate::corpus::{Corpus, Entity};
use crate::rng::Rng;

/// A generated latency query: a clean in-corpus snippet plus the id of the document it was
/// drawn from.
///
/// The latency *timing* needs no label, but the snippet's source doc is the natural relevant
/// answer for an in-corpus recall@k readout: a clean snippet should retrieve its own document.
/// The `latency` command reports recall@k over `target` so the speed numbers carry a quality
/// figure alongside (see `cmd_latency`).
pub struct Query {
    pub text: String,
    /// The id of the corpus document this snippet came from — the relevant id for the
    /// latency command's in-corpus recall@k.
    pub target: i64,
}

/// A fuzzy-eval query: the corrupted text, the target entity id (the label), and the
/// clean target name (kept for the trigram-survival / near-distractor diagnostics).
pub struct FuzzyQuery {
    pub text: String,
    pub target: i64,
    pub clean: String,
    /// The number of typos actually injected into this query (a random draw from the requested
    /// `[lo, hi]` range), kept for the run's reported edit-mix.
    pub edits: usize,
}

/// Take a contiguous run of `len` words starting at a random offset in `text`.
fn snippet(text: &str, len: usize, rng: &mut Rng) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() <= len {
        return words.join(" ");
    }
    let start = rng.below(words.len() - len + 1);
    words[start..start + len].join(" ")
}

/// The four single-character edit operations, weighted toward realistic
/// typos: transpositions and substitutions dominate, insertions/deletions are
/// rarer. Keeps the result non-empty.
fn inject_one_edit(s: &str, rng: &mut Rng) -> String {
    let mut chars: Vec<char> = s.chars().collect();
    if chars.is_empty() {
        return s.to_string();
    }
    let kind = rng.below(10);
    match kind {
        // transposition (40%): swap two adjacent chars
        0..=3 if chars.len() >= 2 => {
            let i = rng.below(chars.len() - 1);
            chars.swap(i, i + 1);
        }
        // substitution (40%): replace one char with a nearby letter
        4..=7 => {
            let i = rng.below(chars.len());
            chars[i] = adjacent_key(chars[i], rng);
        }
        // deletion (10%)
        8 if chars.len() >= 2 => {
            chars.remove(rng.below(chars.len()));
        }
        // insertion (10%)
        _ => {
            let i = rng.below(chars.len() + 1);
            chars.insert(i.min(chars.len()), adjacent_key('e', rng));
        }
    }
    chars.into_iter().collect()
}

/// A keyboard-adjacent (or at least plausible) substitution for `c`.
fn adjacent_key(c: char, rng: &mut Rng) -> char {
    const ROWS: &[&[u8]] = &[b"qwertyuiop", b"asdfghjkl", b"zxcvbnm"];
    let lc = c.to_ascii_lowercase();
    for row in ROWS {
        if let Some(pos) = row.iter().position(|&b| b == lc as u8) {
            // A same-row neighbor: step left or right, clamped off the ends so an edge
            // key still moves — and never maps a key to itself.
            let n = row.len();
            let step: isize = if rng.chance(0.5) { 1 } else { -1 };
            let mut np = (pos as isize + step).clamp(0, n as isize - 1) as usize;
            if np == pos {
                np = if pos == 0 { 1 } else { pos - 1 };
            }
            return row[np] as char;
        }
    }
    // non-letter: bump to a random lowercase letter
    (b'a' + rng.below(26) as u8) as char
}

/// Apply `k` edits, never letting the query fall below the trigram floor.
fn corrupt(mut s: String, k: usize, rng: &mut Rng) -> String {
    for _ in 0..k {
        if s.chars().count() < 4 {
            break;
        }
        s = inject_one_edit(&s, rng);
    }
    s
}

/// Generate `n` clean perf queries: in-corpus snippets of varied length (2–5 words), **no
/// typos** — latency is a pure speed measurement. No labels needed beyond the source doc id.
pub fn perf_queries(corpus: &Corpus, n: usize, seed: u64) -> Vec<Query> {
    let mut rng = Rng::new(seed ^ 0xBEEF);
    (0..n)
        .filter_map(|_| {
            let doc = &corpus.docs[rng.below(corpus.docs.len())];
            let len = rng.range(2, 5);
            let snip = snippet(&doc.text, len, &mut rng);
            if snip.chars().count() < 3 {
                return None;
            }
            Some(Query {
                text: snip,
                target: doc.id,
            })
        })
        .collect()
}

/// Labeled snippet queries for the `ranksweep` pool-depth study: each is a 3–6 word
/// snippet of a random corpus doc with `edits` typos, labeled by that doc's id (the
/// known relevant answer). The typos give the true doc only *partial* trigram overlap,
/// so it ranks at depth among distractors — the regime the rerank pool must reach.
pub fn labeled_snippets(corpus: &Corpus, n: usize, edits: usize, seed: u64) -> Vec<(String, i64)> {
    let mut rng = Rng::new(seed ^ 0x5A5A_1234 ^ ((edits as u64) << 40));
    (0..n)
        .filter_map(|_| {
            let doc = &corpus.docs[rng.below(corpus.docs.len())];
            let len = rng.range(3, 6);
            let snip = snippet(&doc.text, len, &mut rng);
            if snip.chars().count() < 4 {
                return None;
            }
            let text = corrupt(snip, edits, &mut rng);
            if text.chars().count() < 3 {
                return None;
            }
            Some((text, doc.id))
        })
        .collect()
}

/// Generate one fuzzy query per target entity: the entity name with a **random** number of
/// single-character typos drawn uniformly from the inclusive range `[lo, hi]` (so a single
/// batch carries a realistic mix of typo counts), labeled by the entity id. Deterministic per
/// `(seed, lo, hi)`. Names too short to form a trigram query after corruption are skipped (a
/// 1–2 char name can't). `lo == hi` pins every query to that exact count.
pub fn fuzzy_queries(targets: &[Entity], lo: usize, hi: usize, seed: u64) -> Vec<FuzzyQuery> {
    let mut rng = Rng::new(seed ^ 0x00C0_FFEE_u64 ^ ((lo as u64) << 32) ^ ((hi as u64) << 48));
    targets
        .iter()
        .filter_map(|t| {
            if t.name.chars().count() < 4 {
                return None;
            }
            let edits = rng.range(lo, hi); // a random typo count per query, uniform in [lo, hi]
            let text = corrupt(t.name.clone(), edits, &mut rng);
            if text.chars().count() < 3 {
                return None;
            }
            Some(FuzzyQuery {
                text,
                target: t.id,
                clean: t.name.clone(),
                edits,
            })
        })
        .collect()
}
