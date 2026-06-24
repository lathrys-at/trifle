//! Tokenization: the one strategy a caller may supply, and the built-in n-gram
//! tokenizer trifle ships.
//!
//! A [`Tokenizer`] turns text into a stream of tokens. The same tokenizer runs on
//! indexed text, on the postings it maintains, and on queries — there is exactly
//! one, so "the index agrees with the query" is true by construction. It is a
//! *type* parameter of [`Index`](crate::Index), not a trait object: it sits on the
//! hot path (called per window of every indexed and queried string), so it
//! monomorphizes.
//!
//! The built-in [`NgramTokenizer`] (aliased [`TrigramTokenizer`], [`BigramTokenizer`],
//! [`QuadgramTokenizer`]) slides an `N`-code-point window over a normalized form of
//! the text, emitting a zero-allocation inline [`Ngram`] per window. Normalization
//! is the tokenizer's job and is [configurable](TrigramTokenizer::builder).

use std::borrow::Borrow;

use unicode_normalization::UnicodeNormalization;
use unicode_normalization::char::is_combining_mark;

/// Turns text into tokens, for both indexing and querying.
///
/// The associated [`Token`](Tokenizer::Token) type must borrow as `&str` so that a
/// token produced from a query keys the same posting bucket as one produced while
/// indexing. It is `Ord` so selection can break document-frequency ties
/// deterministically, and `Clone` (typically `Copy`) so the small inline tokens
/// move freely on the hot path.
pub trait Tokenizer: Send + Sync {
    /// The token representation. The `Borrow<str>` bound lets a `HashMap<Token, _>`
    /// be probed by `&str`; `Hash + Eq` make it a map key; `Ord` gives selection a
    /// stable tie-break.
    type Token: Borrow<str> + std::hash::Hash + Eq + Ord + Clone;

    /// Tokenize `text` into a stream of tokens. May yield duplicates (a repeated
    /// n-gram appears once per occurrence); trifle deduplicates per segment where a
    /// set is required.
    fn tokenize<'a>(&'a self, text: &'a str) -> impl Iterator<Item = Self::Token> + 'a;

    /// A stable hash of this tokenizer's *behavior* (window size, normalization,
    /// casefolding). It is stamped into the index and compared on open; a change
    /// forces a [`rebuild`](crate::Index::rebuild), because the postings are keyed
    /// by whatever this tokenizer produced at build time.
    ///
    /// A caller shipping a custom tokenizer owns changing this on *any* behavioral
    /// change, including a bug-fix patch that alters the token stream — otherwise a
    /// stale index would be served against new query tokenization.
    fn fingerprint(&self) -> u64;

    /// Locate the byte span `[first, last)` within `text` (raw, original form) that
    /// covers the first through last region producing one of `tokens`, for
    /// [`Match.span`](crate::Match::span).
    ///
    /// The default returns `None` — a custom tokenizer that cannot cheaply map
    /// tokens back to raw bytes simply yields no span, and the caller can locate
    /// matches in [`Match.text`](crate::Match::text) itself. The built-in n-gram
    /// tokenizer overrides this. Called only for the (≤ `limit`) survivors, so a
    /// non-trivial implementation is affordable.
    fn span(&self, text: &str, tokens: &[&str]) -> Option<(usize, usize)> {
        let _ = (text, tokens);
        None
    }
}

/// An inline, `Copy`, heap-free n-gram of up to `CAP` UTF-8 bytes.
///
/// The hot path materializes one per window of every indexed and queried string,
/// so a heap `String` per window would dominate; `Ngram` lives on the stack and
/// keys the posting maps directly. `Hash`/`Eq`/`Ord` delegate to the `str` slice,
/// which is what makes the [`Borrow<str>`] impl sound — a `HashMap<Ngram, _>` can
/// be probed with a plain `&str`.
///
/// `CAP` is a byte capacity (sidestepping the `N·4` const-expr problem of spelling
/// the *code-point* count on stable Rust). Pick `CAP ≥ 4·N` so every `N`-code-point
/// window fits; the provided aliases ([`Trigram`] = `Ngram<12>`, etc.) already do.
/// `CAP` must be `≤ 255` (the length is stored in a `u8`).
#[derive(Clone, Copy)]
pub struct Ngram<const CAP: usize> {
    /// UTF-8 bytes, left-aligned; only `buf[..len]` is meaningful.
    buf: [u8; CAP],
    /// Number of meaningful bytes in `buf`. `CAP ≤ 255` keeps this exact.
    len: u8,
}

/// A 2-code-point n-gram (up to 8 UTF-8 bytes).
pub type Bigram = Ngram<8>;
/// A 3-code-point n-gram (up to 12 UTF-8 bytes) — the default token.
pub type Trigram = Ngram<12>;
/// A 4-code-point n-gram (up to 16 UTF-8 bytes).
pub type Quadgram = Ngram<16>;

impl<const CAP: usize> Ngram<CAP> {
    /// The n-gram as a string slice.
    #[inline]
    pub fn as_str(&self) -> &str {
        // SAFETY: `buf[..len]` is only ever written from valid UTF-8 — `from_chars`
        // encodes `char`s and `try_from_str` copies the bytes of an existing `&str`
        // after a length check — so the slice is always valid UTF-8.
        unsafe { std::str::from_utf8_unchecked(&self.buf[..self.len as usize]) }
    }

    /// Build from a slice of code points, encoding them inline. Returns `None` if
    /// the encoded bytes would exceed `CAP` (so a too-wide window is dropped rather
    /// than truncated mid-code-point); with `CAP ≥ 4·N` this never happens.
    #[inline]
    fn from_chars(chars: &[char]) -> Option<Self> {
        // `len` is a `u8`, so `CAP` must fit one — enforced at compile time per
        // monomorphization, turning the doc's "CAP must be ≤ 255" into a hard error.
        const { assert!(CAP <= u8::MAX as usize, "Ngram CAP must be <= 255") };
        let mut buf = [0u8; CAP];
        let mut len = 0usize;
        for &c in chars {
            let need = c.len_utf8();
            if len + need > CAP {
                return None;
            }
            len += c.encode_utf8(&mut buf[len..]).len();
        }
        Some(Self {
            buf,
            len: len as u8,
        })
    }

    /// Build from a string slice already known to be one n-gram. `None` if it is
    /// longer than `CAP` bytes.
    fn try_from_str(s: &str) -> Option<Self> {
        const { assert!(CAP <= u8::MAX as usize, "Ngram CAP must be <= 255") };
        let bytes = s.as_bytes();
        if bytes.len() > CAP {
            return None;
        }
        let mut buf = [0u8; CAP];
        buf[..bytes.len()].copy_from_slice(bytes);
        Some(Self {
            buf,
            len: bytes.len() as u8,
        })
    }
}

impl<const CAP: usize> std::ops::Deref for Ngram<CAP> {
    type Target = str;
    #[inline]
    fn deref(&self) -> &str {
        self.as_str()
    }
}

impl<const CAP: usize> AsRef<str> for Ngram<CAP> {
    #[inline]
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl<const CAP: usize> Borrow<str> for Ngram<CAP> {
    #[inline]
    fn borrow(&self) -> &str {
        self.as_str()
    }
}

impl<const CAP: usize> PartialEq for Ngram<CAP> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_str() == other.as_str()
    }
}
impl<const CAP: usize> Eq for Ngram<CAP> {}

impl<const CAP: usize> std::hash::Hash for Ngram<CAP> {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Delegate to the `str` hash so an `Ngram` and an equal `&str` (probed via
        // `Borrow`) land in the same bucket.
        self.as_str().hash(state);
    }
}

impl<const CAP: usize> PartialOrd for Ngram<CAP> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<const CAP: usize> Ord for Ngram<CAP> {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.as_str().cmp(other.as_str())
    }
}

impl<const CAP: usize> std::fmt::Debug for Ngram<CAP> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(self.as_str(), f)
    }
}

impl<const CAP: usize> std::fmt::Display for Ngram<CAP> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl<const CAP: usize> rusqlite::ToSql for Ngram<CAP> {
    fn to_sql(&self) -> rusqlite::Result<rusqlite::types::ToSqlOutput<'_>> {
        Ok(rusqlite::types::ToSqlOutput::from(self.as_str()))
    }
}

impl<const CAP: usize> rusqlite::types::FromSql for Ngram<CAP> {
    fn column_result(value: rusqlite::types::ValueRef<'_>) -> rusqlite::types::FromSqlResult<Self> {
        let s = value.as_str()?;
        Ngram::try_from_str(s).ok_or(rusqlite::types::FromSqlError::InvalidType)
    }
}

/// How the tokenizer normalizes text before windowing it.
///
/// Whatever is chosen applies identically to indexed text and queries (one
/// tokenizer, both sides) — the single normalization invariant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Normalization {
    /// NFC (canonical composition). Safe and compact; the default.
    #[default]
    Nfc,
    /// NFD (canonical decomposition).
    Nfd,
    /// NFD then drop combining marks — accent-insensitive. A query `cafe` then
    /// shares trigrams with a stored `café`. Often the better choice for fuzzy
    /// search when accent tolerance matters.
    NfdStripMarks,
    /// No normalization. Choose only when the caller guarantees a canonical form on
    /// both writes and queries (see [`assume_normalized`](TrigramTokenizerBuilder::assume_normalized)).
    None,
}

impl Normalization {
    /// A stable discriminant for the fingerprint.
    fn code(self) -> u8 {
        match self {
            Normalization::Nfc => 1,
            Normalization::Nfd => 2,
            Normalization::NfdStripMarks => 3,
            Normalization::None => 4,
        }
    }
}

/// The built-in tokenizer: an `N`-code-point sliding window over a normalized,
/// optionally case-folded form of the text, emitting an inline [`Ngram<CAP>`] per
/// window. A string shorter than `N` code points yields no tokens.
///
/// Use the aliases — [`TrigramTokenizer`] is the default. Construct with
/// [`new`](Self::new) for the defaults (NFC + Unicode lowercase) or
/// [`builder`](Self::builder) to change normalization / casefolding.
///
/// ```
/// use trifle::tokenize::{Tokenizer, TrigramTokenizer};
///
/// let tok = TrigramTokenizer::new();
/// let grams: Vec<String> = tok.tokenize("Café").map(|g| g.to_string()).collect();
/// assert_eq!(grams, ["caf", "afé"]);
/// ```
#[derive(Clone, Debug)]
pub struct NgramTokenizer<const N: usize, const CAP: usize> {
    normalization: Normalization,
    casefold: bool,
    assume_normalized: bool,
}

/// 3-code-point window tokenizer (the default), emitting [`Trigram`]s.
pub type TrigramTokenizer = NgramTokenizer<3, 12>;
/// 2-code-point window tokenizer, emitting [`Bigram`]s.
pub type BigramTokenizer = NgramTokenizer<2, 8>;
/// 4-code-point window tokenizer, emitting [`Quadgram`]s.
pub type QuadgramTokenizer = NgramTokenizer<4, 16>;

impl<const N: usize, const CAP: usize> Default for NgramTokenizer<N, CAP> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize, const CAP: usize> NgramTokenizer<N, CAP> {
    /// A tokenizer with the default behavior: NFC normalization + Unicode
    /// lowercasing.
    pub fn new() -> Self {
        Self {
            normalization: Normalization::Nfc,
            casefold: true,
            assume_normalized: false,
        }
    }

    /// Start a [`builder`](NgramTokenizerBuilder) to customize normalization and
    /// casefolding.
    pub fn builder() -> NgramTokenizerBuilder<N, CAP> {
        NgramTokenizerBuilder { inner: Self::new() }
    }

    /// The number of code points per window.
    pub const fn window(&self) -> usize {
        N
    }

    /// Normalize and (if enabled) case-fold `text` into the code points the window
    /// slides over. One allocation per text, not per window.
    fn prepare(&self, text: &str) -> Vec<char> {
        // Normalize. `assume_normalized` trusts the caller and skips the transform;
        // otherwise apply the chosen form (the quick-check inside `nfc`/`nfd` makes
        // already-normal text near-free).
        let normalized: Vec<char> = if self.assume_normalized {
            match self.normalization {
                // Stripping marks is not a normalized form the input could already
                // be in, so honor it even under `assume_normalized`.
                Normalization::NfdStripMarks => {
                    text.chars().filter(|c| !is_combining_mark(*c)).collect()
                }
                _ => text.chars().collect(),
            }
        } else {
            match self.normalization {
                Normalization::Nfc => text.nfc().collect(),
                Normalization::Nfd => text.nfd().collect(),
                Normalization::NfdStripMarks => {
                    text.nfd().filter(|c| !is_combining_mark(*c)).collect()
                }
                Normalization::None => text.chars().collect(),
            }
        };
        if self.casefold {
            normalized
                .into_iter()
                .flat_map(|c| c.to_lowercase())
                .collect()
        } else {
            normalized
        }
    }
}

impl<const N: usize, const CAP: usize> Tokenizer for NgramTokenizer<N, CAP> {
    type Token = Ngram<CAP>;

    fn tokenize<'a>(&'a self, text: &'a str) -> impl Iterator<Item = Self::Token> + 'a {
        NgramWindows::<N, CAP> {
            chars: self.prepare(text),
            pos: 0,
        }
    }

    fn fingerprint(&self) -> u64 {
        // A stable FNV-1a over a canonical encoding of the behavior — deterministic
        // across Rust versions and machines, unlike the std default hasher, so the
        // stamp only changes when behavior actually changes. `N` and `CAP` are hashed
        // at full width (not narrowed to a byte), so two widths never collide.
        let mut bytes = Vec::with_capacity(2 + 16 + 3);
        bytes.extend_from_slice(b"ng");
        bytes.extend_from_slice(&(N as u64).to_le_bytes());
        bytes.extend_from_slice(&(CAP as u64).to_le_bytes());
        bytes.push(self.normalization.code());
        bytes.push(self.casefold as u8);
        bytes.push(self.assume_normalized as u8);
        fnv1a_64(&bytes)
    }

    fn span(&self, text: &str, tokens: &[&str]) -> Option<(usize, usize)> {
        // Build the normalized + case-folded char stream WITH the raw byte offset
        // each emitted char came from, then window it like `tokenize` does and map a
        // matched window back to raw bytes.
        //
        // Normalization runs per *combining-sequence cluster* (a starter plus its
        // trailing combining marks), not per char: NFC composition and NFD canonical
        // reordering both act within a combining sequence, so a per-cluster pass
        // produces exactly the same tokens as `prepare`'s whole-string pass — fixing
        // the spans that a per-char pass would lose on decomposed or non-canonically
        // ordered input. Every char a cluster emits is mapped to the cluster's whole
        // raw byte range, so the returned offsets are always raw char boundaries.
        let chars: Vec<(usize, char)> = text.char_indices().collect();
        let mut norm: Vec<char> = Vec::with_capacity(chars.len());
        let mut start: Vec<usize> = Vec::with_capacity(chars.len());
        let mut end: Vec<usize> = Vec::with_capacity(chars.len());
        let mut i = 0;
        while i < chars.len() {
            let cluster_start = chars[i].0;
            let mut j = i + 1;
            while j < chars.len() && is_combining_mark(chars[j].1) {
                j += 1;
            }
            let cluster_end = chars.get(j).map_or(text.len(), |(off, _)| *off);
            self.normalize_cluster(&text[cluster_start..cluster_end], |nc| {
                norm.push(nc);
                start.push(cluster_start);
                end.push(cluster_end);
            });
            i = j;
        }
        if norm.len() < N {
            return None;
        }
        let mut first: Option<usize> = None;
        let mut last = 0usize;
        for w in 0..=norm.len() - N {
            if let Some(g) = Ngram::<CAP>::from_chars(&norm[w..w + N]) {
                if tokens.contains(&g.as_str()) {
                    first.get_or_insert(start[w]);
                    last = end[w + N - 1];
                }
            }
        }
        first.map(|f| (f, last))
    }
}

impl<const N: usize, const CAP: usize> NgramTokenizer<N, CAP> {
    /// Apply this tokenizer's normalization + casefolding to one combining-sequence
    /// cluster, emitting the resulting char(s). Mirrors [`prepare`](Self::prepare)
    /// exactly (same normalization form, same casefold, same `assume_normalized`
    /// skip) but on a cluster substring, so [`span`](Self::span) re-derives the same
    /// token stream while tracking raw offsets.
    fn normalize_cluster(&self, cluster: &str, mut push: impl FnMut(char)) {
        let normalized: Vec<char> = if self.assume_normalized {
            match self.normalization {
                Normalization::NfdStripMarks => {
                    cluster.chars().filter(|c| !is_combining_mark(*c)).collect()
                }
                _ => cluster.chars().collect(),
            }
        } else {
            match self.normalization {
                Normalization::Nfc => cluster.nfc().collect(),
                Normalization::Nfd => cluster.nfd().collect(),
                Normalization::NfdStripMarks => {
                    cluster.nfd().filter(|c| !is_combining_mark(*c)).collect()
                }
                Normalization::None => cluster.chars().collect(),
            }
        };
        for c in normalized {
            if self.casefold {
                for lc in c.to_lowercase() {
                    push(lc);
                }
            } else {
                push(c);
            }
        }
    }
}

/// Sliding-window iterator: yields one [`Ngram<CAP>`] per `N`-code-point window of
/// the prepared char buffer. Owns the buffer, so it satisfies the `Tokenizer`
/// lifetime without borrowing the source text.
struct NgramWindows<const N: usize, const CAP: usize> {
    chars: Vec<char>,
    pos: usize,
}

impl<const N: usize, const CAP: usize> Iterator for NgramWindows<N, CAP> {
    type Item = Ngram<CAP>;

    fn next(&mut self) -> Option<Self::Item> {
        // A window of fewer-than-N code points yields nothing.
        while self.pos + N <= self.chars.len() {
            let window = &self.chars[self.pos..self.pos + N];
            self.pos += 1;
            // A window too wide for `CAP` is skipped (only possible when a caller
            // aliases with `CAP < 4·N`); the provided aliases never hit this.
            if let Some(g) = Ngram::<CAP>::from_chars(window) {
                return Some(g);
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = (self.chars.len() + 1).saturating_sub(self.pos + N);
        (0, Some(remaining))
    }
}

/// The builder for [`NgramTokenizer`].
///
/// ```
/// use trifle::tokenize::{Normalization, TrigramTokenizer};
///
/// // Accent-insensitive, still lowercased.
/// let tok = TrigramTokenizer::builder()
///     .normalization(Normalization::NfdStripMarks)
///     .build();
/// ```
#[derive(Clone, Debug)]
pub struct NgramTokenizerBuilder<const N: usize, const CAP: usize> {
    inner: NgramTokenizer<N, CAP>,
}

impl<const N: usize, const CAP: usize> NgramTokenizerBuilder<N, CAP> {
    /// Set the normalization form (default [`Normalization::Nfc`]).
    pub fn normalization(mut self, normalization: Normalization) -> Self {
        self.inner.normalization = normalization;
        self
    }

    /// Enable or disable Unicode lowercasing (default `true`). Disabling skips the
    /// casefold pass when the caller guarantees inputs are already folded.
    pub fn casefold(mut self, casefold: bool) -> Self {
        self.inner.casefold = casefold;
        self
    }

    /// Trust that input is already in the chosen normalization form and skip the
    /// normalization pass (default `false`). Sound only if the guarantee holds for
    /// *both* writes and queries.
    pub fn assume_normalized(mut self, assume_normalized: bool) -> Self {
        self.inner.assume_normalized = assume_normalized;
        self
    }

    /// Finish building the tokenizer.
    pub fn build(self) -> NgramTokenizer<N, CAP> {
        self.inner
    }
}

/// FNV-1a (64-bit). A tiny, dependency-free, version-stable hash for the tokenizer
/// fingerprint — the value is stamped on disk, so it must not drift with the
/// standard library's default hasher.
fn fnv1a_64(bytes: &[u8]) -> u64 {
    const OFFSET: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x00000100000001b3;
    let mut hash = OFFSET;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grams<T: Tokenizer>(tok: &T, text: &str) -> Vec<String> {
        tok.tokenize(text).map(|g| g.borrow().to_string()).collect()
    }

    #[test]
    fn trigram_windows_lowercased_nfc() {
        let tok = TrigramTokenizer::new();
        assert_eq!(grams(&tok, "Hello"), ["hel", "ell", "llo"]);
    }

    #[test]
    fn shorter_than_window_yields_nothing() {
        let tok = TrigramTokenizer::new();
        assert!(grams(&tok, "ab").is_empty());
        assert!(grams(&tok, "").is_empty());
        assert_eq!(grams(&tok, "abc"), ["abc"]);
    }

    #[test]
    fn bigram_and_quadgram_aliases() {
        assert_eq!(grams(&BigramTokenizer::new(), "abcd"), ["ab", "bc", "cd"]);
        assert_eq!(grams(&QuadgramTokenizer::new(), "abcde"), ["abcd", "bcde"]);
    }

    #[test]
    fn nfc_and_nfd_query_share_grams_under_strip_marks() {
        let tok = TrigramTokenizer::builder()
            .normalization(Normalization::NfdStripMarks)
            .build();
        // "café" (composed) and "cafe" should share the same trigram stream.
        assert_eq!(grams(&tok, "café"), grams(&tok, "cafe"));
        assert_eq!(grams(&tok, "café"), ["caf", "afe"]);
    }

    #[test]
    fn nfc_default_keeps_accent_distinct() {
        let tok = TrigramTokenizer::new();
        assert_ne!(grams(&tok, "café"), grams(&tok, "cafe"));
    }

    #[test]
    fn casefold_can_be_disabled() {
        let tok = TrigramTokenizer::builder().casefold(false).build();
        assert_eq!(grams(&tok, "ABC"), ["ABC"]);
    }

    #[test]
    fn nfc_composed_and_decomposed_tokenize_identically() {
        let tok = TrigramTokenizer::new();
        let composed = "caf\u{e9}"; // café
        let decomposed = "cafe\u{301}"; // cafe + combining acute
        assert_eq!(grams(&tok, composed), grams(&tok, decomposed));
    }

    #[test]
    fn ngram_borrow_probes_hashmap_by_str() {
        use std::collections::HashMap;
        let mut m: HashMap<Trigram, i32> = HashMap::new();
        m.insert(Trigram::try_from_str("abc").unwrap(), 7);
        assert_eq!(m.get("abc"), Some(&7));
    }

    #[test]
    fn ngram_ord_matches_str_ord() {
        let a = Trigram::try_from_str("abc").unwrap();
        let b = Trigram::try_from_str("abd").unwrap();
        assert!(a < b);
        assert_eq!(a.cmp(&b), "abc".cmp("abd"));
    }

    #[test]
    fn fingerprint_changes_with_behavior_and_is_stable() {
        let a = TrigramTokenizer::new().fingerprint();
        let b = TrigramTokenizer::new().fingerprint();
        assert_eq!(a, b, "same behavior -> same fingerprint");

        let nfd = TrigramTokenizer::builder()
            .normalization(Normalization::Nfd)
            .build()
            .fingerprint();
        assert_ne!(a, nfd, "normalization change -> different fingerprint");

        let no_fold = TrigramTokenizer::builder()
            .casefold(false)
            .build()
            .fingerprint();
        assert_ne!(a, no_fold, "casefold change -> different fingerprint");

        // Window size is part of behavior too.
        assert_ne!(a, BigramTokenizer::new().fingerprint());
    }

    #[test]
    fn assume_normalized_skips_transform_but_still_casefolds() {
        let tok = TrigramTokenizer::builder().assume_normalized(true).build();
        assert_eq!(grams(&tok, "ABC"), ["abc"]);
    }

    #[test]
    fn ngram_roundtrips_through_sql_value() {
        use rusqlite::types::{FromSql, ToSql, ValueRef};
        let g = Trigram::try_from_str("xyz").unwrap();
        let out = g.to_sql().unwrap();
        // Re-read the bound text back into an Ngram.
        let parsed = Trigram::column_result(ValueRef::Text(b"xyz")).unwrap();
        assert_eq!(parsed, g);
        // The bound value is the trigram text.
        match out {
            rusqlite::types::ToSqlOutput::Borrowed(ValueRef::Text(t)) => assert_eq!(t, b"xyz"),
            other => panic!("unexpected ToSql output: {other:?}"),
        }
    }
}
