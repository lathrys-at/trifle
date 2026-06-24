//! Query generation from the corpus itself.
//!
//! Per the design: the realistic query is an **in-corpus document snippet**, with
//! or without injected typos. That sidesteps the "where do realistic queries come
//! from" problem entirely — the snippet's vocabulary and co-occurrence are exactly
//! the corpus's. The perf harness doesn't need labels (it measures latency); the
//! quality harness keeps the source document id as the free ground-truth label.

use crate::corpus::Corpus;
use crate::rng::Rng;

/// A generated query, with the document it was drawn from (its relevant answer).
pub struct Query {
    pub text: String,
    pub source_doc: i64,
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

/// The four single-character edit operations (§10.5), weighted toward realistic
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

/// Generate `n` perf queries: snippets of varied length (2–5 words) with 0–2 typos.
/// Latency only — no labels needed.
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
            let edits = match rng.below(3) {
                0 => 0,
                1 => 1,
                _ => 2,
            };
            Some(Query {
                text: corrupt(snip, edits, &mut rng),
                source_doc: doc.id,
                edits,
            })
        })
        .collect()
}

/// Generate `n` quality queries with exactly `edits` typos, labeled by source doc,
/// for recall@k against a baseline.
pub fn quality_queries(corpus: &Corpus, n: usize, edits: usize, seed: u64) -> Vec<Query> {
    let mut rng = Rng::new(seed ^ 0x00C0_FFEE_u64);
    (0..n)
        .filter_map(|_| {
            let doc = &corpus.docs[rng.below(corpus.docs.len())];
            let len = rng.range(3, 6);
            let snip = snippet(&doc.text, len, &mut rng);
            if snip.chars().count() < 4 {
                return None;
            }
            Some(Query {
                text: corrupt(snip, edits, &mut rng),
                source_doc: doc.id,
                edits,
            })
        })
        .collect()
}
