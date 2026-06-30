//! Tokenization: the one strategy a caller may supply, the script-aware tokenizer trifle
//! defaults to, and the plain fixed-width n-gram tokenizers for single-script corpora.
//!
//! A [`Tokenizer`] turns text into a stream of grams. The same tokenizer runs on indexed
//! text, on the postings it maintains, and on queries — there is exactly one, so "the
//! index agrees with the query" is true by construction. It is a *type* parameter of
//! [`Index`](crate::Index), not a trait object: it sits on the hot path (called per window
//! of every indexed and queried string), so it monomorphizes.
//!
//! Tokenization is **online**: [`tokenize`](Tokenizer::tokenize) returns a lazy iterator
//! that pulls normalized code points from streaming Unicode-normalization adaptors and
//! slides a window over them — it never materializes an intermediate `Vec`.
//!
//! - [`DefaultTokenizer`] splits text into maximal same-script runs and slides a
//!   script-appropriate window over each, emitting a variable-length [`Gram`] (a CJK run
//!   takes bigrams, everything else trigrams). A mixed-script document therefore indexes
//!   without manufacturing cross-script grams at the seams.
//! - [`NgramTokenizer<N>`](NgramTokenizer) (aliased [`TrigramTokenizer`] /
//!   [`BigramTokenizer`]) is the plain fixed-`N` sliding window — pick it when a corpus is
//!   single-script and the script segmentation buys nothing.
//!
//! Normalization is the tokenizer's job and is configurable on either tokenizer.

use std::borrow::Borrow;
use std::str::Chars;

use unicode_normalization::char::is_combining_mark;
use unicode_normalization::{Decompositions, Recompositions, UnicodeNormalization};
use unicode_script::{Script, UnicodeScript};

use crate::term::IntoTerm;

/// Turns text into grams, for both indexing and querying.
///
/// The associated [`Token`](Tokenizer::Token) type is bound to [`IntoTerm`] — which is
/// itself `Borrow<str>` — so a token produced from a query keys the same posting bucket as
/// one produced while indexing, *and* every token can be packed into the interned
/// [`Term`](crate::Term) the dictionary keys on (a structural guarantee, not a
/// convention). It is `Ord` so selection can break document-frequency ties
/// deterministically, and `Clone` (typically `Copy`) so the small inline tokens move
/// freely on the hot path.
pub trait Tokenizer: Send + Sync {
    /// The token representation. [`IntoTerm`] gives it both the `Borrow<str>` needed to
    /// probe a `HashMap<Token, _>` by `&str` and the [`term`](IntoTerm::term) packing the
    /// interning dictionary needs; `Hash + Eq` make it a map key; `Ord` gives selection a
    /// stable tie-break.
    ///
    /// `Ord` **must be a total order consistent with `Eq`** (and, for a custom token, ideally
    /// with its `Borrow<str>` view): selection deduplicates tokens through a hash set, whose
    /// iteration order is nondeterministic, and recovers a deterministic result *only* because
    /// it then sorts by `(rarity, Token)`. An `Ord` that returns `Equal` for two `Eq`-distinct
    /// tokens would let that nondeterministic order leak into results. The built-in tokens
    /// satisfy this (their `Ord` delegates to `str`).
    type Token: IntoTerm + std::hash::Hash + Eq + Ord + Clone;

    /// Tokenize `text` into a lazy stream of grams. May yield duplicates (a repeated gram
    /// appears once per occurrence); trifle deduplicates per segment where a set is
    /// required.
    fn tokenize<'a>(&'a self, text: &'a str) -> impl Iterator<Item = Self::Token> + 'a;

    /// Query-side word tagging: like [`tokenize`](Tokenizer::tokenize), but each gram is paired
    /// with its **word id** — the comonotone *stopping block* the pruner's confidence-bounded stop
    /// groups co-failing grams into (derivation §5). Word ids increment at each word boundary;
    /// grams within one word share an id, and (for a tokenizer that breaks its windows on those
    /// boundaries) no gram straddles two words.
    ///
    /// The default assigns every gram to block `0` (one block) — always recall-safe, because
    /// merging blocks only *raises* the stop's variance (§5), never lowers it. A tokenizer that
    /// knows its word boundaries (e.g. [`DefaultTokenizer`], which breaks on whitespace) overrides
    /// this to report them, tightening the bound. Used only on the **query** side; indexing uses
    /// [`tokenize`](Tokenizer::tokenize) (the grams agree because both apply the same windowing).
    fn tokenize_words<'a>(
        &'a self,
        text: &'a str,
    ) -> impl Iterator<Item = (Self::Token, u32)> + 'a {
        self.tokenize(text).map(|t| (t, 0))
    }

    /// The **primary** gram order (codepoint window width) this tokenizer uses for a given
    /// script-class byte, or `u8::MAX` for a **single-order** tokenizer that has no shorter
    /// secondary order — the default.
    ///
    /// v0.4/M5 (derivation §8) scores each script run at its primary order plus, when a query
    /// is starved, a **secondary** one order shorter (Latin trigram + bigram, CJK bigram +
    /// unigram), reciprocal-rank-fused. The search path classifies a query gram `(script,
    /// order)` into the primary rank-view (`order == primary_order(script)`) or the secondary
    /// (`order == primary_order(script) − 1`). A single-order tokenizer returns `u8::MAX`, so
    /// no gram is ever classified secondary and the secondary view never forms — it pays
    /// nothing for the §8 machinery. [`DefaultTokenizer`] overrides this with its per-script
    /// window policy; the fixed-width [`NgramTokenizer`] keeps the default (it emits exactly
    /// one order and never a shorter fallback).
    fn primary_order(&self, script: u8) -> u8 {
        let _ = script;
        u8::MAX
    }

    /// A stable hash of this tokenizer's *behavior* (window policy, normalization,
    /// casefolding). It is stamped into the index and compared on open; a change forces a
    /// [`rebuild`](crate::Index::rebuild), because the postings are keyed by whatever this
    /// tokenizer produced at build time.
    ///
    /// A caller shipping a custom tokenizer owns changing this on *any* behavioral
    /// change, including a bug-fix patch that alters the token stream — otherwise a stale
    /// index would be served against new query tokenization.
    fn fingerprint(&self) -> u64;

    /// Locate the byte span `[first, last)` within `text` (raw, original form) that covers
    /// the first through last region producing one of `tokens`, for
    /// [`Match.span`](crate::Match::span).
    ///
    /// The default returns `None` — a custom tokenizer that cannot cheaply map tokens back
    /// to raw bytes simply yields no span, and the caller can locate matches in
    /// [`Match.text`](crate::Match::text) itself. The built-in tokenizers override this.
    /// Called only for the (≤ `limit`) survivors, so a non-trivial implementation is
    /// affordable.
    fn span(&self, text: &str, tokens: &[&str]) -> Option<(usize, usize)> {
        let _ = (text, tokens);
        None
    }
}

/// The byte capacity of an [`Ngram`]: three 4-byte code points. The term-encoding ceiling
/// is 3 code points, so every window a built-in tokenizer emits fits.
const GRAM_CAP: usize = 12;

/// An inline, `Copy`, heap-free gram of up to `N` code points (`N ≤ 3`, the term-encoding
/// ceiling). The const parameter is the window identity: [`Ngram<2>`] and [`Ngram<3>`] are
/// distinct types, so a bigram tokenizer and a trigram tokenizer cannot be confused.
///
/// The hot path materializes one per window of every indexed and queried string, so a heap
/// `String` per window would dominate; `Ngram` lives on the stack and keys the posting maps
/// directly. The bytes are stored UTF-8-encoded, so `Hash`/`Eq`/`Ord`/`Borrow<str>` all
/// delegate to the `str` slice — which is what makes a `HashMap<Ngram, _>` probe-able by a
/// plain `&str` and gives it [`IntoTerm`] for free.
///
/// A run-length CJK bigram and a Latin trigram both come back from [`DefaultTokenizer`] as
/// the variable-length alias [`Gram`] = `Ngram<3>` (a `Gram` holding two code points is a
/// bigram); the fixed-`N` [`NgramTokenizer<N>`](NgramTokenizer) yields exactly-`N` grams.
#[derive(Clone, Copy)]
pub struct Ngram<const N: usize> {
    /// UTF-8 bytes, left-aligned; only `buf[..len]` is meaningful.
    buf: [u8; GRAM_CAP],
    /// Number of meaningful bytes in `buf` (`≤ GRAM_CAP`, well within a `u8`).
    len: u8,
}

/// A variable-length gram of up to three code points — the token [`DefaultTokenizer`]
/// emits (a CJK bigram and a Latin trigram are both `Gram`s).
pub type Gram = Ngram<3>;

impl<const N: usize> Ngram<N> {
    /// The gram as a string slice.
    #[inline]
    pub fn as_str(&self) -> &str {
        // SAFETY: `buf[..len]` is only ever written from valid UTF-8 — `from_chars`
        // encodes `char`s and `try_from_str` copies the bytes of an existing `&str` after
        // a length check — so the slice is always valid UTF-8.
        unsafe { std::str::from_utf8_unchecked(&self.buf[..self.len as usize]) }
    }

    /// Build from a slice of code points, encoding them inline. Returns `None` if the
    /// encoded bytes would exceed [`GRAM_CAP`] (so a too-wide window is dropped rather than
    /// truncated mid-code-point); a window of ≤3 code points always fits.
    #[inline]
    fn from_chars(chars: &[char]) -> Option<Self> {
        let mut buf = [0u8; GRAM_CAP];
        let mut len = 0usize;
        for &c in chars {
            let need = c.len_utf8();
            if len + need > GRAM_CAP {
                return None;
            }
            len += c.encode_utf8(&mut buf[len..]).len();
        }
        Some(Self {
            buf,
            len: len as u8,
        })
    }

    /// Build from a string slice already known to be one gram. `None` if it is longer than
    /// [`GRAM_CAP`] bytes.
    #[cfg(test)]
    fn try_from_str(s: &str) -> Option<Self> {
        let bytes = s.as_bytes();
        if bytes.len() > GRAM_CAP {
            return None;
        }
        let mut buf = [0u8; GRAM_CAP];
        buf[..bytes.len()].copy_from_slice(bytes);
        Some(Self {
            buf,
            len: bytes.len() as u8,
        })
    }
}

impl<const N: usize> std::ops::Deref for Ngram<N> {
    type Target = str;
    #[inline]
    fn deref(&self) -> &str {
        self.as_str()
    }
}

impl<const N: usize> AsRef<str> for Ngram<N> {
    #[inline]
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl<const N: usize> Borrow<str> for Ngram<N> {
    #[inline]
    fn borrow(&self) -> &str {
        self.as_str()
    }
}

impl<const N: usize> PartialEq for Ngram<N> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_str() == other.as_str()
    }
}
impl<const N: usize> Eq for Ngram<N> {}

impl<const N: usize> std::hash::Hash for Ngram<N> {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Delegate to the `str` hash so an `Ngram` and an equal `&str` (probed via
        // `Borrow`) land in the same bucket.
        self.as_str().hash(state);
    }
}

impl<const N: usize> PartialOrd for Ngram<N> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<const N: usize> Ord for Ngram<N> {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.as_str().cmp(other.as_str())
    }
}

impl<const N: usize> std::fmt::Debug for Ngram<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(self.as_str(), f)
    }
}

impl<const N: usize> std::fmt::Display for Ngram<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// How a tokenizer normalizes text before windowing it.
///
/// Whatever is chosen applies identically to indexed text and queries (one tokenizer, both
/// sides) — the single normalization invariant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Normalization {
    /// NFC (canonical composition). Safe and compact; the default.
    #[default]
    Nfc,
    /// NFD (canonical decomposition).
    Nfd,
    /// NFD then drop combining marks — accent-insensitive. A query `cafe` then shares grams
    /// with a stored `café`. Often the better choice for fuzzy search when accent tolerance
    /// matters.
    NfdStripMarks,
    /// No normalization. Choose only when the caller guarantees a canonical form on both
    /// writes and queries (see [`assume_normalized`](NgramTokenizerBuilder::assume_normalized)).
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

/// The normalization configuration shared by both built-in tokenizers. Owns the online
/// normalized-char source and the per-cluster re-derivation that [`Tokenizer::span`] uses.
#[derive(Clone, Debug)]
struct Norm {
    normalization: Normalization,
    casefold: bool,
    assume_normalized: bool,
}

impl Default for Norm {
    fn default() -> Self {
        Norm {
            normalization: Normalization::Nfc,
            casefold: true,
            assume_normalized: false,
        }
    }
}

impl Norm {
    /// The online, allocation-free normalized + case-folded code-point stream for `text`.
    fn prepared<'a>(&self, text: &'a str) -> PreparedChars<'a> {
        PreparedChars::new(text, self)
    }

    /// Three fingerprint bytes: normalization form, casefold, assume-normalized.
    fn fingerprint_bytes(&self) -> [u8; 3] {
        [
            self.normalization.code(),
            self.casefold as u8,
            self.assume_normalized as u8,
        ]
    }

    /// Apply this normalization + casefolding to one combining-sequence cluster, emitting
    /// the resulting char(s) one at a time. Mirrors [`PreparedChars`] exactly (same form,
    /// same casefold, same `assume_normalized` skip) but on a cluster substring, so
    /// [`Norm::for_each_normalized_char`] re-derives the same token stream while tracking
    /// raw byte offsets.
    fn normalize_cluster(&self, cluster: &str, push: &mut dyn FnMut(char)) {
        let casefold = self.casefold;
        let mut emit = |c: char| {
            if casefold {
                for lc in c.to_lowercase() {
                    push(lc);
                }
            } else {
                push(c);
            }
        };
        if self.assume_normalized {
            match self.normalization {
                Normalization::NfdStripMarks => cluster
                    .chars()
                    .filter(|c| !is_combining_mark(*c))
                    .for_each(&mut emit),
                _ => cluster.chars().for_each(&mut emit),
            }
        } else {
            match self.normalization {
                Normalization::Nfc => cluster.nfc().for_each(&mut emit),
                Normalization::Nfd => cluster.nfd().for_each(&mut emit),
                Normalization::NfdStripMarks => cluster
                    .nfd()
                    .filter(|c| !is_combining_mark(*c))
                    .for_each(&mut emit),
                Normalization::None => cluster.chars().for_each(&mut emit),
            }
        }
    }

    /// Call `f(ch, cluster_start, cluster_end)` for each normalized code point, where the
    /// `[start, end)` is the raw byte range of the cluster that produced it. Normalization
    /// runs per *cluster* — a starter plus its trailing combining marks and conjoining
    /// Hangul jamo — which produces the same code points as the whole-string streaming pass
    /// (NFC composition and NFD canonical reordering both act within such a unit), so the
    /// spans line up with the tokens on decomposed, reordered, or jamo-composed input.
    /// Every code point a cluster emits maps to the cluster's whole raw range, so offsets
    /// are always raw char boundaries.
    fn for_each_normalized_char(&self, text: &str, mut f: impl FnMut(char, usize, usize)) {
        let mut it = text.char_indices().peekable();
        while let Some((cluster_start, _)) = it.next() {
            while it
                .peek()
                .is_some_and(|&(_, c)| is_combining_mark(c) || is_conjoining_jamo(c))
            {
                it.next();
            }
            let cluster_end = it.peek().map_or(text.len(), |&(off, _)| off);
            self.normalize_cluster(&text[cluster_start..cluster_end], &mut |ch| {
                f(ch, cluster_start, cluster_end)
            });
        }
    }
}

/// The online normalized-char source: a streaming Unicode-normalization adaptor plus an
/// optional case-fold expansion, yielding one code point at a time with no buffering of the
/// whole text.
struct PreparedChars<'a> {
    kind: NormKind<'a>,
    /// Drop combining marks after normalization (the `NfdStripMarks` tail).
    strip_marks: bool,
    casefold: bool,
    /// The in-flight lowercase expansion of the current source char (`to_lowercase` may
    /// yield up to three code points, e.g. `İ` → `i̇`).
    pending: Option<std::char::ToLowercase>,
}

/// The streaming normalization adaptor backing [`PreparedChars`] — one variant per form
/// (`assume_normalized` collapses to [`NormKind::Raw`]). All yield `char`.
enum NormKind<'a> {
    Nfc(Recompositions<Chars<'a>>),
    Nfd(Decompositions<Chars<'a>>),
    Raw(Chars<'a>),
}

impl<'a> PreparedChars<'a> {
    fn new(text: &'a str, norm: &Norm) -> Self {
        let (kind, strip_marks) = if norm.assume_normalized {
            // Trust the caller's form; only mark-stripping is still honored (it is not a
            // normalized form the input could already be in).
            match norm.normalization {
                Normalization::NfdStripMarks => (NormKind::Raw(text.chars()), true),
                _ => (NormKind::Raw(text.chars()), false),
            }
        } else {
            match norm.normalization {
                Normalization::Nfc => (NormKind::Nfc(text.nfc()), false),
                Normalization::Nfd => (NormKind::Nfd(text.nfd()), false),
                Normalization::NfdStripMarks => (NormKind::Nfd(text.nfd()), true),
                Normalization::None => (NormKind::Raw(text.chars()), false),
            }
        };
        PreparedChars {
            kind,
            strip_marks,
            casefold: norm.casefold,
            pending: None,
        }
    }

    /// The next normalized (mark-stripped) char before casefolding.
    fn raw_next(&mut self) -> Option<char> {
        loop {
            let c = match &mut self.kind {
                NormKind::Nfc(it) => it.next(),
                NormKind::Nfd(it) => it.next(),
                NormKind::Raw(it) => it.next(),
            }?;
            if self.strip_marks && is_combining_mark(c) {
                continue;
            }
            return Some(c);
        }
    }
}

impl Iterator for PreparedChars<'_> {
    type Item = char;

    fn next(&mut self) -> Option<char> {
        loop {
            if let Some(p) = &mut self.pending {
                if let Some(c) = p.next() {
                    return Some(c);
                }
                self.pending = None;
            }
            let c = self.raw_next()?;
            if self.casefold {
                let mut lc = c.to_lowercase();
                // `to_lowercase` yields at least one char; return the first, stash the rest.
                if let Some(first) = lc.next() {
                    self.pending = Some(lc);
                    return Some(first);
                }
            } else {
                return Some(c);
            }
        }
    }
}

/// Conjoining Hangul Vowel/Trailing jamo (U+1161..=U+1175, U+11A8..=U+11C2). Like combining
/// marks these attach to a preceding character — NFC composes a leading jamo plus these into
/// one syllable — so [`Norm::for_each_normalized_char`] keeps them in the same cluster;
/// splitting the composition would lose the token for that region.
fn is_conjoining_jamo(c: char) -> bool {
    matches!(c, '\u{1161}'..='\u{1175}' | '\u{11A8}'..='\u{11C2}')
}

/// A plain fixed-width n-gram tokenizer: an `N`-code-point sliding window over a normalized,
/// optionally case-folded form of the text, with no script segmentation. Pick it when a
/// corpus is single-script; otherwise prefer [`DefaultTokenizer`].
///
/// `N` must be in `1..=3` (the term-encoding ceiling) — `NgramTokenizer<4>::new()` is a
/// compile error. Use the aliases [`TrigramTokenizer`] (`N = 3`) and [`BigramTokenizer`]
/// (`N = 2`).
///
/// ```
/// use trifle::tokenize::{Tokenizer, TrigramTokenizer};
///
/// let tok = TrigramTokenizer::new();
/// let grams: Vec<String> = tok.tokenize("Hello").map(|g| g.to_string()).collect();
/// assert_eq!(grams, ["hel", "ell", "llo"]);
/// ```
#[derive(Clone, Debug)]
pub struct NgramTokenizer<const N: usize> {
    norm: Norm,
}

/// A 3-code-point sliding-window tokenizer (no script segmentation).
pub type TrigramTokenizer = NgramTokenizer<3>;
/// A 2-code-point sliding-window tokenizer (no script segmentation).
pub type BigramTokenizer = NgramTokenizer<2>;

impl<const N: usize> Default for NgramTokenizer<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> NgramTokenizer<N> {
    /// A tokenizer with the default behavior: NFC normalization + Unicode lowercasing.
    pub fn new() -> Self {
        const {
            assert!(
                N >= 1 && N <= 3,
                "NgramTokenizer supports N in 1..=3 (the 3-code-point term-encoding ceiling)"
            )
        }
        NgramTokenizer {
            norm: Norm::default(),
        }
    }

    /// Start a [builder](NgramTokenizerBuilder) to customize normalization/casefolding.
    pub fn builder() -> NgramTokenizerBuilder<N> {
        NgramTokenizerBuilder { inner: Self::new() }
    }

    /// The number of code points per window.
    pub const fn window(&self) -> usize {
        N
    }
}

impl<const N: usize> Tokenizer for NgramTokenizer<N> {
    type Token = Ngram<N>;

    fn tokenize<'a>(&'a self, text: &'a str) -> impl Iterator<Item = Self::Token> + 'a {
        NgramWindows::<N> {
            chars: self.norm.prepared(text),
            win: ['\0'; N],
            filled: 0,
        }
    }

    fn fingerprint(&self) -> u64 {
        // FNV-1a over a canonical encoding of the behavior: a tag, the window width, and the
        // normalization bytes. `N` is hashed at full width so two widths never collide.
        let mut bytes = Vec::with_capacity(2 + 8 + 3);
        bytes.extend_from_slice(b"ng");
        bytes.extend_from_slice(&(N as u64).to_le_bytes());
        bytes.extend_from_slice(&self.norm.fingerprint_bytes());
        fnv1a_64(&bytes)
    }

    fn span(&self, text: &str, tokens: &[&str]) -> Option<(usize, usize)> {
        let mut first: Option<usize> = None;
        let mut last = 0usize;
        let mut win_ch = ['\0'; N];
        let mut win_start = [0usize; N];
        let mut win_end = [0usize; N];
        let mut filled = 0usize;
        self.norm.for_each_normalized_char(text, |ch, cs, ce| {
            if N > 1 {
                win_ch.copy_within(1..N, 0);
                win_start.copy_within(1..N, 0);
                win_end.copy_within(1..N, 0);
            }
            win_ch[N - 1] = ch;
            win_start[N - 1] = cs;
            win_end[N - 1] = ce;
            filled += 1;
            if filled >= N {
                if let Some(g) = Ngram::<N>::from_chars(&win_ch) {
                    if tokens.contains(&g.as_str()) {
                        first.get_or_insert(win_start[0]);
                        last = win_end[N - 1];
                    }
                }
            }
        });
        first.map(|f| (f, last))
    }
}

/// Lazy fixed-`N` sliding window over a [`PreparedChars`] stream. Owns the source, so it
/// satisfies the [`Tokenizer`] lifetime without a separate buffer.
struct NgramWindows<'a, const N: usize> {
    chars: PreparedChars<'a>,
    /// The last up-to-`N` code points (a small ring shifted left as it fills).
    win: [char; N],
    /// How many of `win`'s slots are populated (saturates at `N`).
    filled: usize,
}

impl<const N: usize> Iterator for NgramWindows<'_, N> {
    type Item = Ngram<N>;

    fn next(&mut self) -> Option<Ngram<N>> {
        loop {
            let c = self.chars.next()?;
            if self.filled < N {
                self.win[self.filled] = c;
                self.filled += 1;
            } else {
                self.win.copy_within(1..N, 0);
                self.win[N - 1] = c;
            }
            if self.filled == N {
                // For N ≤ 3 a window is ≤ 12 bytes, so `from_chars` always succeeds.
                if let Some(g) = Ngram::<N>::from_chars(&self.win) {
                    return Some(g);
                }
            }
        }
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
pub struct NgramTokenizerBuilder<const N: usize> {
    inner: NgramTokenizer<N>,
}

impl<const N: usize> NgramTokenizerBuilder<N> {
    /// Set the normalization form (default [`Normalization::Nfc`]).
    pub fn normalization(mut self, normalization: Normalization) -> Self {
        self.inner.norm.normalization = normalization;
        self
    }

    /// Enable or disable Unicode lowercasing (default `true`).
    pub fn casefold(mut self, casefold: bool) -> Self {
        self.inner.norm.casefold = casefold;
        self
    }

    /// Trust that input is already in the chosen normalization form and skip the
    /// normalization pass (default `false`). Sound only if the guarantee holds for *both*
    /// writes and queries.
    pub fn assume_normalized(mut self, assume_normalized: bool) -> Self {
        self.inner.norm.assume_normalized = assume_normalized;
        self
    }

    /// Finish building the tokenizer.
    pub fn build(self) -> NgramTokenizer<N> {
        self.inner
    }
}

/// The tokenizer trifle ships and defaults to: it splits a normalized, optionally
/// case-folded form of the text into maximal same-script runs and slides a script-appropriate
/// window over each run, emitting an inline [`Gram`] per window.
///
/// `Common`/`Inherited` code points (digits, punctuation, combining marks) are transparent
/// — they inherit the current run's script rather than breaking it, so no run boundary falls
/// in the middle of a word; a leading run of only `Common` code points forms its own
/// `Common`-class run. The default [`WindowPolicy`] uses bigrams for the dense CJK scripts
/// (Han / Hiragana / Katakana / Hangul) and trigrams elsewhere.
///
/// v0.4/M5 (derivation §8): each run is windowed at **two** orders — its primary order *and* a
/// secondary one shorter (Latin trigram + bigram, CJK bigram + unigram) — so the tokenizer emits
/// dual-order grams. The shorter order doubles as the structural fallback for a run too short to
/// produce the primary (a 2-char Latin run yields its bigram; a 1-char CJK run its unigram).
///
/// Construct with [`new`](Self::new) for the defaults (NFC + Unicode lowercase) or
/// [`builder`](Self::builder) to change normalization / casefolding.
///
/// ```
/// use trifle::tokenize::{DefaultTokenizer, Tokenizer};
///
/// let tok = DefaultTokenizer::new();
/// // "ab漢字" → Latin run "ab" (a bigram, too short for a trigram) + Han run "漢字" (bigram + unigrams).
/// let grams: Vec<String> = tok.tokenize("ab漢字").map(|g| g.to_string()).collect();
/// assert_eq!(grams, ["ab", "漢", "漢字", "字"]);
/// ```
#[derive(Clone, Debug)]
pub struct DefaultTokenizer {
    norm: Norm,
    policy: WindowPolicy,
}

/// Per-script-class window sizes for [`DefaultTokenizer`], indexed by the script-class byte
/// (`unicode_script::Script as u8`, the same byte the term encoding stores). The default is
/// trigrams everywhere except the dense CJK scripts, which take bigrams.
#[derive(Clone, Debug)]
pub struct WindowPolicy {
    sizes: [u8; 256],
}

impl Default for WindowPolicy {
    fn default() -> Self {
        let mut sizes = [3u8; 256];
        for s in [
            Script::Han,
            Script::Hiragana,
            Script::Katakana,
            Script::Hangul,
        ] {
            sizes[s as usize] = 2;
        }
        WindowPolicy { sizes }
    }
}

impl Default for DefaultTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl DefaultTokenizer {
    /// A tokenizer with the default behavior: NFC + Unicode lowercasing and the default
    /// (CJK-bigram, else-trigram) window policy.
    pub fn new() -> Self {
        DefaultTokenizer {
            norm: Norm::default(),
            policy: WindowPolicy::default(),
        }
    }

    /// Start a [builder](DefaultTokenizerBuilder) to customize normalization/casefolding.
    pub fn builder() -> DefaultTokenizerBuilder {
        DefaultTokenizerBuilder { inner: Self::new() }
    }
}

impl Tokenizer for DefaultTokenizer {
    type Token = Gram;

    fn tokenize<'a>(&'a self, text: &'a str) -> impl Iterator<Item = Self::Token> + 'a {
        GramTokens {
            chars: self.norm.prepared(text),
            sizes: self.policy.sizes,
            win: ['\0'; 3],
            win_len: 0,
            class: None,
            n: 0,
            word: 0,
            pending: None,
        }
    }

    fn tokenize_words<'a>(
        &'a self,
        text: &'a str,
    ) -> impl Iterator<Item = (Self::Token, u32)> + 'a {
        GramWords(GramTokens {
            chars: self.norm.prepared(text),
            sizes: self.policy.sizes,
            win: ['\0'; 3],
            win_len: 0,
            class: None,
            n: 0,
            word: 0,
            pending: None,
        })
    }

    fn primary_order(&self, script: u8) -> u8 {
        // The DefaultTokenizer's per-script primary gram order (CJK bigrams, else trigrams) —
        // the SECONDARY order the v0.4/M5 rank-views fuse over is one shorter (derivation §8).
        // The search path reads this to classify a query gram (script, order) into the primary
        // or secondary rank-view without hard-coding the CJK list. (`NgramTokenizer` keeps the
        // trait default `u8::MAX` — single-order, no rank-views.)
        self.policy.sizes[script as usize]
    }

    fn fingerprint(&self) -> u64 {
        // FNV-1a over a canonical encoding of the behavior: a tag, the normalization bytes,
        // the window policy, and a layout-version byte — so a change to segmentation/window
        // sizing forces a rebuild.
        let mut bytes = Vec::with_capacity(3 + 3 + 256 + 1);
        bytes.extend_from_slice(b"scr");
        bytes.extend_from_slice(&self.norm.fingerprint_bytes());
        bytes.extend_from_slice(&self.policy.sizes);
        // Layout version. Bumped 1 → 2 in v0.4/M4 (whitespace now breaks gram windows) and
        // 2 → 3 in v0.4/M5: each script run is now windowed at BOTH its primary order and a
        // secondary one-shorter order (Latin trigram + bigram, CJK bigram + unigram), so the
        // tokenizer emits dual-order grams on both the index and query sides (derivation §8). A
        // pre-M5 cache therefore drift-resets (drop + rebuild) on open — the expected "drop,
        // never migrate" path, not a migration; the on-disk FORMAT is unchanged (a bigram and a
        // trigram are already distinct `Term`s/postings), so only this tokenizer-behavior byte
        // moves, not `SCHEMA_VERSION`. The change is in `GramTokens`, not a config byte, so it
        // must be stamped here explicitly.
        bytes.push(3);
        fnv1a_64(&bytes)
    }

    fn span(&self, text: &str, tokens: &[&str]) -> Option<(usize, usize)> {
        let sizes = self.policy.sizes;
        let mut first: Option<usize> = None;
        let mut last = 0usize;
        let mut win_ch = ['\0'; 3];
        let mut win_start = [0usize; 3];
        let mut win_end = [0usize; 3];
        let mut win_len = 0usize;
        let mut class: Option<u8> = None;
        let mut n = 0usize;
        self.norm.for_each_normalized_char(text, |ch, cs, ce| {
            if is_break(ch) {
                // Mirror `GramTokens`: whitespace breaks the window (no gram straddles it) and
                // resets the class, so span re-derivation tracks the same grams the tokenizer emits.
                class = None;
                n = 0;
                win_len = 0;
                return;
            }
            let class_c = script_class_of(ch, class);
            if Some(class_c) != class {
                class = Some(class_c);
                n = (sizes[class_c as usize] as usize).min(3);
                win_len = 0;
            }
            if n == 0 {
                return;
            }
            if win_len < n {
                win_ch[win_len] = ch;
                win_start[win_len] = cs;
                win_end[win_len] = ce;
                win_len += 1;
            } else {
                win_ch.copy_within(1..n, 0);
                win_start.copy_within(1..n, 0);
                win_end.copy_within(1..n, 0);
                win_ch[n - 1] = ch;
                win_start[n - 1] = cs;
                win_end[n - 1] = ce;
            }
            // Mirror `GramTokens`: check BOTH the primary (full `n`-wide) window and the
            // secondary (last `n − 1` code points) window, so a span can be located for either
            // order's gram (derivation §8 dual-order).
            let mut consider = |lo_idx: usize, hi_idx: usize| {
                if let Some(g) = Gram::from_chars(&win_ch[lo_idx..=hi_idx]) {
                    if tokens.contains(&g.as_str()) {
                        first.get_or_insert(win_start[lo_idx]);
                        last = last.max(win_end[hi_idx]);
                    }
                }
            };
            if win_len == n {
                consider(0, n - 1);
            }
            let ns = n.saturating_sub(1);
            if ns >= 1 && win_len >= ns {
                consider(win_len - ns, win_len - 1);
            }
        });
        first.map(|f| (f, last))
    }
}

/// A code point that **breaks** a gram window — and, on the query side, marks a word boundary (the
/// §5 comonotone *stopping-block* boundary). v0.4/M4 breaks on Unicode whitespace only.
///
/// Whitespace-only is the recall-safe subset of the derivation's "whitespace + delimiter
/// punctuation" (§5/§8). The asymmetry is decisive: *under*-splitting (merging two query words into
/// one block) only **raises** the stop's variance, so it is conservative, whereas *over*-splitting
/// a real word (the apostrophe in `don't`, the dots in `U.S.A`) under-counts the variance and fires
/// the stop early — a recall loss. So inter-word delimiter-*punctuation* breaking is a deferred
/// §5/§8 precision refinement; intra-word punctuation is word-internal regardless (§11). Digits and
/// combining marks stay transparent (they inherit the current run), unchanged from v0.3.
///
/// Applied identically on the index side ([`tokenize`](Tokenizer::tokenize)) and the query side
/// ([`tokenize_words`](Tokenizer::tokenize_words)) so the stored and queried grams agree (the
/// single-tokenizer invariant). A break resets the window — so no gram spans it — and the dropped
/// whitespace is not itself part of any gram.
#[inline]
fn is_break(c: char) -> bool {
    c.is_whitespace()
}

/// The script class of `ch` for run segmentation: its strong script as `Script as u8`, or —
/// for transparent `Common`/`Inherited` code points — the current run's class (or `Common`
/// when no run has started yet).
#[inline]
fn script_class_of(ch: char, current: Option<u8>) -> u8 {
    match ch.script() {
        Script::Common | Script::Inherited => current.unwrap_or(Script::Common as u8),
        s => s as u8,
    }
}

/// Lazy script-segmented window over a [`PreparedChars`] stream. Maintains a sliding window
/// that resets at every change of script class and whose width is that class's policy; a
/// leading `Common` run uses the `Common` width. Owns the source (no separate buffer).
///
/// v0.4/M5 (derivation §8): each run is windowed at BOTH its primary order `n` and a
/// **secondary** one shorter (`n − 1`) — Latin trigram + bigram, CJK bigram + unigram — so one
/// advance can yield two grams. The second (the secondary) is buffered in
/// [`pending`](Self::pending) and emitted on the next call.
struct GramTokens<'a> {
    chars: PreparedChars<'a>,
    /// The window policy, copied in (a `[u8; 256]` is `Copy`).
    sizes: [u8; 256],
    /// The current run's last up-to-`n` code points.
    win: [char; 3],
    win_len: usize,
    /// The current run's script class, or `None` before the first code point.
    class: Option<u8>,
    /// The current run's primary window width (`1..=3`), `0` before the first code point.
    n: usize,
    /// The current word id (the §5 comonotone *stopping-block* id), incremented at each
    /// [`is_break`] boundary. Surfaced only through [`next_tagged`](GramTokens::next_tagged) /
    /// [`tokenize_words`](DefaultTokenizer::tokenize_words); the bare [`Iterator`] drops it. Starts
    /// at `0`; leading/consecutive breaks may leave gaps, which is fine — only the *partition* of
    /// grams into words matters to the stop.
    word: u32,
    /// The buffered **secondary**-order gram produced alongside a primary gram in one advance,
    /// returned on the next call (with the word id it was produced under). Drained before any new
    /// code point is read, so it always belongs to the position just processed and never straddles
    /// a word/run boundary.
    pending: Option<(Gram, u32)>,
}

impl GramTokens<'_> {
    /// The next gram paired with its word id (the §5 stopping-block id). Whitespace (and any other
    /// [`is_break`] code point) resets the window — so no gram straddles it — and bumps the word id.
    /// A break also resets the script class to `None`, so the text after a break starts a fresh run
    /// exactly like the start of the string (the leading-`Common` rule reapplies).
    ///
    /// Each run is windowed at its primary order `n` AND the secondary order `n − 1` (derivation
    /// §8). A single advance can therefore produce two grams — the **primary** (the full `n`-wide
    /// window) and the **secondary** (the last `n − 1` code points). The primary is returned first
    /// and the secondary buffered in [`pending`](Self::pending) for the next call. Sliding the
    /// secondary window across the whole run recovers every shorter gram (e.g. `"abcd"` →
    /// `ab, bc, cd` for the bigrams and `abc, bcd` for the trigrams), the structural fallback a
    /// too-short run (a 2-char Latin word, a 1-char CJK run) relies on.
    fn next_tagged(&mut self) -> Option<(Gram, u32)> {
        // Drain a buffered secondary gram before reading any new code point — it belongs to the
        // position already processed, so it is emitted before any boundary reset.
        if let Some(p) = self.pending.take() {
            return Some(p);
        }
        loop {
            let c = self.chars.next()?;
            if is_break(c) {
                // Word/window boundary: drop the break char, reset the window + class, advance the
                // word id. `saturating_add` so a pathologically long query cannot wrap the counter.
                self.class = None;
                self.n = 0;
                self.win_len = 0;
                self.word = self.word.saturating_add(1);
                continue;
            }
            let class_c = script_class_of(c, self.class);
            if Some(class_c) != self.class {
                // Run break: start a fresh window of the new class's width.
                self.class = Some(class_c);
                self.n = (self.sizes[class_c as usize] as usize).min(3);
                self.win_len = 0;
            }
            if self.n == 0 {
                continue;
            }
            if self.win_len < self.n {
                self.win[self.win_len] = c;
                self.win_len += 1;
            } else {
                self.win.copy_within(1..self.n, 0);
                self.win[self.n - 1] = c;
            }
            // Primary gram: the full `n`-wide window once it has filled.
            let primary = (self.win_len == self.n)
                .then(|| Gram::from_chars(&self.win[..self.n]))
                .flatten();
            // Secondary gram: the last `n − 1` code points (the one-shorter order), once that many
            // are present. `n == 1` (a unigram-primary run) has no shorter order, so no secondary.
            let ns = self.n - 1;
            let secondary = (ns >= 1 && self.win_len >= ns)
                .then(|| Gram::from_chars(&self.win[self.win_len - ns..self.win_len]))
                .flatten();
            match (primary, secondary) {
                (Some(p), sec) => {
                    self.pending = sec.map(|s| (s, self.word));
                    return Some((p, self.word));
                }
                (None, Some(s)) => return Some((s, self.word)),
                (None, None) => continue,
            }
        }
    }
}

impl Iterator for GramTokens<'_> {
    type Item = Gram;

    fn next(&mut self) -> Option<Gram> {
        self.next_tagged().map(|(g, _)| g)
    }
}

/// The query-side word-tagged adaptor over [`GramTokens`] — yields each gram with its §5
/// stopping-block (word) id. Indexing uses the bare [`GramTokens`] iterator (the word id is dropped
/// but the *windowing* — whitespace breaking — is identical, so stored grams agree with queried).
struct GramWords<'a>(GramTokens<'a>);

impl Iterator for GramWords<'_> {
    type Item = (Gram, u32);

    fn next(&mut self) -> Option<(Gram, u32)> {
        self.0.next_tagged()
    }
}

/// Builder for [`DefaultTokenizer`].
///
/// ```
/// use trifle::tokenize::{DefaultTokenizer, Normalization};
///
/// // Accent-insensitive, still lowercased.
/// let tok = DefaultTokenizer::builder()
///     .normalization(Normalization::NfdStripMarks)
///     .build();
/// ```
#[derive(Clone, Debug)]
pub struct DefaultTokenizerBuilder {
    inner: DefaultTokenizer,
}

impl DefaultTokenizerBuilder {
    /// Set the normalization form (default [`Normalization::Nfc`]).
    pub fn normalization(mut self, normalization: Normalization) -> Self {
        self.inner.norm.normalization = normalization;
        self
    }

    /// Enable or disable Unicode lowercasing (default `true`).
    pub fn casefold(mut self, casefold: bool) -> Self {
        self.inner.norm.casefold = casefold;
        self
    }

    /// Trust that input is already in the chosen normalization form (default `false`).
    /// Sound only if the guarantee holds for *both* writes and queries.
    pub fn assume_normalized(mut self, assume_normalized: bool) -> Self {
        self.inner.norm.assume_normalized = assume_normalized;
        self
    }

    /// Finish building the tokenizer.
    pub fn build(self) -> DefaultTokenizer {
        self.inner
    }
}

/// FNV-1a (64-bit). A tiny, dependency-free, version-stable hash for the tokenizer and
/// schema fingerprints — the values are stamped on disk, so they must not drift with the
/// standard library's default hasher.
pub(crate) fn fnv1a_64(bytes: &[u8]) -> u64 {
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

    // ----- NgramTokenizer (fixed width) ------------------------------------------

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
    fn bigram_alias() {
        assert_eq!(grams(&BigramTokenizer::new(), "abcd"), ["ab", "bc", "cd"]);
    }

    #[test]
    fn ngram_does_not_script_segment() {
        // The plain n-gram tokenizer windows straight across a script boundary (unlike
        // DefaultTokenizer); a CJK + Latin trigram straddles by design.
        let tok = TrigramTokenizer::new();
        assert_eq!(grams(&tok, "ab漢"), ["ab漢"]);
    }

    #[test]
    fn fingerprint_distinguishes_width() {
        assert_ne!(
            TrigramTokenizer::new().fingerprint(),
            BigramTokenizer::new().fingerprint()
        );
    }

    // ----- shared normalization --------------------------------------------------

    #[test]
    fn nfc_and_nfd_query_share_grams_under_strip_marks() {
        let tok = TrigramTokenizer::builder()
            .normalization(Normalization::NfdStripMarks)
            .build();
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
    fn length_changing_casefold_is_handled() {
        // 'İ'.to_lowercase() == ['i', '\u{307}'] — one source char folds to two; the online
        // casefold expansion must thread both into the window stream.
        let tok = TrigramTokenizer::new();
        assert_eq!(grams(&tok, "İst"), ["i\u{307}s", "\u{307}st"]);
    }

    #[test]
    fn strip_marks_collapses_dotted_capital_i_with_plain_forms() {
        let tok = TrigramTokenizer::builder()
            .normalization(Normalization::NfdStripMarks)
            .build();
        assert_eq!(grams(&tok, "İab"), grams(&tok, "Iab"));
        assert_eq!(grams(&tok, "İab"), grams(&tok, "iab"));
        assert_eq!(grams(&tok, "İab"), ["iab"]);
    }

    #[test]
    fn assume_normalized_matches_the_normalizing_path_on_conforming_input() {
        let normalizing = TrigramTokenizer::new();
        let assuming = TrigramTokenizer::builder().assume_normalized(true).build();
        let nfc_input = "caf\u{e9} r\u{e9}sum\u{e9}"; // precomposed, already NFC
        assert_eq!(grams(&normalizing, nfc_input), grams(&assuming, nfc_input));
    }

    #[test]
    fn assume_normalized_skips_transform_but_still_casefolds() {
        let tok = TrigramTokenizer::builder().assume_normalized(true).build();
        assert_eq!(grams(&tok, "ABC"), ["abc"]);
    }

    #[test]
    fn fingerprint_changes_with_behavior_and_is_stable() {
        let a = DefaultTokenizer::new().fingerprint();
        assert_eq!(a, DefaultTokenizer::new().fingerprint());
        assert_ne!(
            a,
            DefaultTokenizer::builder()
                .normalization(Normalization::Nfd)
                .build()
                .fingerprint()
        );
        assert_ne!(
            a,
            DefaultTokenizer::builder()
                .casefold(false)
                .build()
                .fingerprint()
        );
        assert_ne!(
            a,
            DefaultTokenizer::builder()
                .assume_normalized(true)
                .build()
                .fingerprint()
        );
        // The script tokenizer and the plain trigram tokenizer must not collide.
        assert_ne!(a, TrigramTokenizer::new().fingerprint());
    }

    // ----- Ngram value type ------------------------------------------------------

    #[test]
    fn gram_borrow_probes_hashmap_by_str() {
        use std::collections::HashMap;
        let mut m: HashMap<Gram, i32> = HashMap::new();
        m.insert(Gram::try_from_str("abc").unwrap(), 7);
        assert_eq!(m.get("abc"), Some(&7));
    }

    #[test]
    fn gram_ord_matches_str_ord() {
        let a = Gram::try_from_str("abc").unwrap();
        let b = Gram::try_from_str("abd").unwrap();
        assert!(a < b);
        assert_eq!(a.cmp(&b), "abc".cmp("abd"));
    }

    #[test]
    fn gram_term_round_trips_through_into_term() {
        // The blanket IntoTerm packs a token into a Term via its UTF-8 form; a Gram and the
        // equal &str must pack to the same Term (so write-path interning by token agrees
        // with query-path interning by string).
        let g = Gram::try_from_str("abc").unwrap();
        assert!(g.term().is_some());
        assert_eq!(g.term(), "abc".term());
    }

    #[test]
    fn from_chars_rejects_a_window_wider_than_cap() {
        // Three 4-byte code points = 12 bytes: exactly fills the buffer; a fourth overflows.
        assert!(Gram::from_chars(&['🚀', '🎉', '😀']).is_some());
        assert!(Gram::from_chars(&['🚀', '🎉', '😀', '🔥']).is_none());
    }

    // ----- DefaultTokenizer (script segmentation) --------------------------------

    #[test]
    fn segments_by_script_with_no_cross_script_grams() {
        let tok = DefaultTokenizer::new();
        let g = grams(&tok, "hello漢字");
        assert!(g.contains(&"hel".to_string()));
        assert!(g.contains(&"llo".to_string()));
        assert!(g.contains(&"漢字".to_string()));
        // v0.4/M5 dual-order: the Han run also yields its unigrams (漢, 字) and the Latin run its
        // bigrams. The invariant under test is that no gram STRADDLES the script seam — every gram
        // is purely Latin or purely Han, never a mix.
        assert!(g.iter().all(|t| {
            let has_han = t.chars().any(|c| matches!(c, '漢' | '字'));
            let has_latin = t.chars().any(|c| c.is_ascii_alphabetic());
            !(has_han && has_latin)
        }));
    }

    #[test]
    fn common_codepoints_inherit_the_run_script() {
        let tok = DefaultTokenizer::new();
        // An interior digit is Common: it inherits the Latin run rather than splitting it. v0.4/M5
        // dual-order: the Latin run yields its trigram AND its bigrams (no run boundary at the digit).
        assert_eq!(grams(&tok, "a1b"), ["a1", "a1b", "1b"]);
    }

    #[test]
    fn leading_common_run_is_its_own_class() {
        let tok = DefaultTokenizer::new();
        // "12漢字": "12" is a leading Common run — now a Common bigram "12" (v0.4/M5 dual-order) —
        // then the Han run yields its bigram 漢字 AND its unigrams 漢, 字. No gram straddles the
        // Common→Han seam (no "2漢").
        assert_eq!(grams(&tok, "12漢字"), ["12", "漢", "漢字", "字"]);
    }

    #[test]
    fn uses_bigrams_for_cjk() {
        let tok = DefaultTokenizer::new();
        // CJK primary order is the bigram; v0.4/M5 also emits the unigram secondary (derivation §8).
        assert_eq!(grams(&tok, "漢字漢"), ["漢", "漢字", "字", "字漢", "漢"]);
    }

    // ----- v0.4/M4: whitespace breaks gram windows + word tagging -----------------

    #[test]
    fn whitespace_breaks_gram_windows_no_gram_straddles_a_word() {
        let tok = DefaultTokenizer::new();
        // "quick brown" used to be one run yielding cross-word grams ("k b", " br", …). Now the
        // space breaks the window: per-word trigrams only, and no gram contains whitespace.
        let g = grams(&tok, "quick brown");
        // v0.4/M5 dual-order: per-word trigrams AND bigrams; still no gram contains whitespace.
        assert_eq!(
            g,
            [
                "qu", "qui", "ui", "uic", "ic", "ick", "ck", "br", "bro", "ro", "row", "ow", "own",
                "wn"
            ]
        );
        assert!(
            g.iter().all(|t| !t.contains(' ')),
            "no gram straddles the space: {g:?}"
        );
    }

    #[test]
    fn multiple_and_leading_whitespace_just_break() {
        let tok = DefaultTokenizer::new();
        // Runs of whitespace, tabs/newlines, and leading space all break without manufacturing grams.
        // v0.4/M5 dual-order: each word yields its trigram AND bigrams.
        assert_eq!(
            grams(&tok, "  abc\tdef\n"),
            ["ab", "abc", "bc", "de", "def", "ef"]
        );
    }

    #[test]
    fn break_resets_class_like_string_start() {
        let tok = DefaultTokenizer::new();
        // After a break the next run starts fresh: "ab 123" → "ab" is now a Latin bigram (v0.4/M5),
        // then "123" is a leading-Common run → the Common trigram "123" and its bigrams 12, 23 (the
        // digit does not inherit a stale Latin run across the break).
        assert_eq!(grams(&tok, "ab 123"), ["ab", "12", "123", "23"]);
        // Within a word an interior digit still inherits the run (unchanged): "a1b" yields its
        // trigram and bigrams; "cd" yields its bigram. No gram crosses the break.
        assert_eq!(grams(&tok, "a1b cd"), ["a1", "a1b", "1b", "cd"]);
    }

    #[test]
    fn tokenize_words_tags_one_id_per_query_word() {
        let tok = DefaultTokenizer::new();
        let tagged: Vec<(String, u32)> = tok
            .tokenize_words("quick brown")
            .map(|(g, w)| (g.to_string(), w))
            .collect();
        // Word 0 = "quick" grams; the space bumps to word 1 = "brown" grams. Grams within a word
        // share an id; no gram crosses the boundary. v0.4/M5 dual-order: trigrams AND bigrams.
        assert_eq!(
            tagged,
            [
                ("qu".to_string(), 0),
                ("qui".to_string(), 0),
                ("ui".to_string(), 0),
                ("uic".to_string(), 0),
                ("ic".to_string(), 0),
                ("ick".to_string(), 0),
                ("ck".to_string(), 0),
                ("br".to_string(), 1),
                ("bro".to_string(), 1),
                ("ro".to_string(), 1),
                ("row".to_string(), 1),
                ("ow".to_string(), 1),
                ("own".to_string(), 1),
                ("wn".to_string(), 1),
            ]
        );
    }

    #[test]
    fn tokenize_words_default_is_one_block_for_plain_ngram() {
        // The plain n-gram tokenizer keeps the trait default: every gram in block 0 (recall-safe),
        // and it does NOT break on whitespace (its deliberate straddling is unchanged).
        let tok = TrigramTokenizer::new();
        let tagged: Vec<(String, u32)> = tok
            .tokenize_words("a b")
            .map(|(g, w)| (g.to_string(), w))
            .collect();
        assert_eq!(tagged, [("a b".to_string(), 0)]);
    }

    #[test]
    fn index_equals_query_grams_under_adversarial_whitespace() {
        // The single-tokenizer invariant: the index path (`tokenize`) and the query path
        // (`tokenize_words`) MUST emit the same grams — only the word tags differ — across the full
        // zoo of whitespace (NBSP, em-space, VT/FF/NEL, CR/LF, tabs, runs, leading/trailing), short
        // words, mixed script, and intra-word punctuation. And no emitted gram may contain a break.
        let tok = DefaultTokenizer::new();
        let cases = [
            "",
            " ",
            "   ",
            "\t\n\r",
            "\u{00A0}",                         // NBSP only
            "a",                                // 1-char Latin
            "ab",                               // 2-char Latin (no trigram)
            "abc",                              // exactly one trigram
            "  abc\tdef\n",                     // leading + tab + trailing
            "quick\u{00A0}brown",               // NBSP between words
            "a\u{2003}b\u{2003}ccc",            // em-space separated, short/long mix
            "café 漢字 test",                   // mixed script with spaces + combining-capable
            "漢字\t漢",                         // CJK split by tab
            "a1b cd2",                          // interior digits + a 2-char word
            "\u{000B}\u{000C}word\u{0085}next", // VT, FF, NEL
            "don't stop",                       // apostrophe is NOT whitespace -> intra-word
        ];
        for text in cases {
            let bare = grams(&tok, text);
            let tagged: Vec<String> = tok
                .tokenize_words(text)
                .map(|(g, _)| g.to_string())
                .collect();
            assert_eq!(
                bare, tagged,
                "index vs query grams must agree (single-tokenizer invariant) for {text:?}"
            );
            assert!(
                bare.iter().all(|g| !g.chars().any(|c| c.is_whitespace())),
                "no gram may contain whitespace for {text:?}: {bare:?}"
            );
        }
    }

    #[test]
    fn word_ids_partition_grams_by_word_no_cross_word_gram() {
        // Word ids are constant within a word and non-decreasing across words; three words yield
        // three distinct ids and no gram straddles a boundary.
        let tok = DefaultTokenizer::new();
        let tagged: Vec<(String, u32)> = tok
            .tokenize_words("quick brown foxes")
            .map(|(g, w)| (g.to_string(), w))
            .collect();
        let mut last = 0u32;
        let mut words = std::collections::BTreeSet::new();
        for (g, w) in &tagged {
            assert!(*w >= last, "word ids are non-decreasing: {tagged:?}");
            last = *w;
            words.insert(*w);
            assert!(
                !g.chars().any(|c| c.is_whitespace()),
                "no cross-word gram: {g:?}"
            );
        }
        assert_eq!(words.len(), 3, "three distinct words tagged: {tagged:?}");
    }

    #[test]
    fn span_never_returns_a_cross_word_gram() {
        // Span re-derivation applies the same break, so it locates per-word grams and never a
        // space-straddling one.
        let tok = DefaultTokenizer::new();
        let span = tok.span("quick brown", &["uic"]).expect("span for uic");
        assert_eq!(&"quick brown"[span.0..span.1], "uic");
        assert_eq!(
            tok.span("quick brown", &["k b"]),
            None,
            "a space-straddling gram is never located"
        );
        assert_eq!(
            tok.span("quick brown", &["ck "]),
            None,
            "a trailing-space gram is never located"
        );
    }

    // ----- span -------------------------------------------------------------------

    #[test]
    fn span_brackets_first_through_last_occurrence() {
        let tok = TrigramTokenizer::new();
        assert_eq!(tok.span("abcZZZZZZabc", &["abc"]), Some((0, 12)));
        assert_eq!(tok.span("zz abc zz", &["abc"]), Some((3, 6)));
        assert_eq!(tok.span("abcdef", &["zzz"]), None);
    }

    #[test]
    fn span_recovers_decomposed_input_under_nfc() {
        let tok = DefaultTokenizer::new();
        let stored = "Xcafe\u{301}Y"; // X c a f e ́ Y
        // The composed trigram "afé" is present in the dual-order stream (alongside bigrams); pick
        // it by value rather than position (v0.4/M5 interleaves bigrams, so it is no longer nth(2)).
        let token: String = tok
            .tokenize(stored)
            .map(|g| g.to_string())
            .find(|g| g == "af\u{e9}")
            .expect("the composed trigram afé is emitted"); // afé (composed)
        assert_eq!(token, "af\u{e9}"); // afé (composed)
        let span = tok
            .span(stored, &[token.as_str()])
            .expect("a span for the match");
        assert!(stored.is_char_boundary(span.0) && stored.is_char_boundary(span.1));
        assert!(span.0 < span.1);
    }

    #[test]
    fn span_recovers_noncanonically_ordered_marks_under_nfd() {
        let tok = DefaultTokenizer::builder()
            .normalization(Normalization::Nfd)
            .build();
        let stored = "Xq\u{0301}\u{0323}Y"; // q + acute(ccc 230) + dot-below(ccc 220)
        let tokens: Vec<String> = tok.tokenize(stored).map(|g| g.to_string()).collect();
        assert!(tokens.iter().any(|t| t.starts_with('q')));
        let qtok = tokens.iter().find(|t| t.starts_with('q')).unwrap();
        assert!(tok.span(stored, &[qtok.as_str()]).is_some());
    }

    #[test]
    fn span_recovers_conjoining_hangul_jamo_under_nfc() {
        // Two syllables, each stored as conjoining jamo ᄀ ᅡ ᆨ; NFC composes each into "각",
        // giving "각각" — a Hangul bigram (one syllable alone is too short). A per-cluster
        // pass that split at each jamo (they are starters, not combining marks) would never
        // re-derive the composed syllables and would return None.
        let tok = DefaultTokenizer::new();
        let stored = "\u{1100}\u{1161}\u{11A8}\u{1100}\u{1161}\u{11A8}"; // "각각"
        let tokens: Vec<String> = tok.tokenize(stored).map(|g| g.to_string()).collect();
        // v0.4/M5: the Hangul (CJK) run yields its bigram 각각 AND its unigrams 각, 각.
        assert_eq!(tokens, ["각", "각각", "각"], "{tokens:?}");
        let refs: Vec<&str> = tokens.iter().map(String::as_str).collect();
        let (lo, hi) = tok.span(stored, &refs).expect("a span for the match");
        assert!(stored.is_char_boundary(lo) && stored.is_char_boundary(hi));
        assert_eq!((lo, hi), (0, stored.len()));
    }

    #[test]
    fn span_offsets_are_always_char_boundaries_across_modes() {
        let inputs = ["İstanbul café", "straße fluß", "ab🚀cd🎉ef", "Δοκιμή test"];
        for norm in [
            Normalization::Nfc,
            Normalization::Nfd,
            Normalization::NfdStripMarks,
            Normalization::None,
        ] {
            let tok = DefaultTokenizer::builder().normalization(norm).build();
            for input in inputs {
                let tokens: Vec<String> = tok.tokenize(input).map(|g| g.to_string()).collect();
                let refs: Vec<&str> = tokens.iter().map(String::as_str).collect();
                if let Some((lo, hi)) = tok.span(input, &refs) {
                    assert!(
                        input.is_char_boundary(lo) && input.is_char_boundary(hi),
                        "span ({lo},{hi}) not on char boundaries for {input:?} / {norm:?}"
                    );
                    assert!(lo < hi && hi <= input.len());
                    let _ = &input[lo..hi];
                }
            }
        }
    }
}
