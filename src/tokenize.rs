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
use unicode_script::{Script, UnicodeScript};

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
    /// both writes and queries (see [`assume_normalized`](NgramTokenizerBuilder::assume_normalized)).
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
        prepare_chars(
            text,
            self.normalization,
            self.casefold,
            self.assume_normalized,
        )
    }
}

/// Normalize and (if enabled) case-fold `text` into the code points a window slides
/// over — the shared front of both built-in tokenizers. `assume_normalized` trusts the
/// caller and skips the transform; otherwise apply the chosen form (the quick-check
/// inside `nfc`/`nfd` makes already-normal text near-free).
fn prepare_chars(
    text: &str,
    normalization: Normalization,
    casefold: bool,
    assume_normalized: bool,
) -> Vec<char> {
    let normalized: Vec<char> = if assume_normalized {
        match normalization {
            // Stripping marks is not a normalized form the input could already be in, so
            // honor it even under `assume_normalized`.
            Normalization::NfdStripMarks => {
                text.chars().filter(|c| !is_combining_mark(*c)).collect()
            }
            _ => text.chars().collect(),
        }
    } else {
        match normalization {
            Normalization::Nfc => text.nfc().collect(),
            Normalization::Nfd => text.nfd().collect(),
            Normalization::NfdStripMarks => text.nfd().filter(|c| !is_combining_mark(*c)).collect(),
            Normalization::None => text.chars().collect(),
        }
    };
    if casefold {
        normalized
            .into_iter()
            .flat_map(|c| c.to_lowercase())
            .collect()
    } else {
        normalized
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
        // Re-derive the normalized + case-folded token stream and map a matched window
        // back to raw bytes, **streaming** — O(N) stack state, zero heap allocation.
        //
        // Normalization runs per *cluster* — a starter plus its trailing combining marks
        // and conjoining Hangul jamo — not per char: NFC composition and NFD canonical
        // reordering both act within such a unit, so a per-cluster pass produces exactly
        // the same tokens as `prepare`'s whole-string pass — fixing the spans a per-char
        // pass would lose on decomposed, non-canonically-ordered, or jamo-composed input.
        // Every char a cluster emits is mapped to the cluster's whole raw byte range, so
        // the returned offsets are always raw char boundaries.
        //
        // The window holds the last N emitted chars and their raw ranges; a long
        // document never materializes an O(text) buffer (span runs for the ≤ limit
        // survivors, but a survivor can itself be large).
        let mut first: Option<usize> = None;
        let mut last = 0usize;
        {
            let mut win_ch = ['\0'; N];
            let mut win_start = [0usize; N];
            let mut win_end = [0usize; N];
            let mut filled = 0usize;
            let mut consider = |ch: char, cluster_start: usize, cluster_end: usize| {
                if N > 1 {
                    win_ch.copy_within(1..N, 0);
                    win_start.copy_within(1..N, 0);
                    win_end.copy_within(1..N, 0);
                }
                win_ch[N - 1] = ch;
                win_start[N - 1] = cluster_start;
                win_end[N - 1] = cluster_end;
                filled += 1;
                if filled >= N {
                    if let Some(g) = Ngram::<CAP>::from_chars(&win_ch) {
                        if tokens.contains(&g.as_str()) {
                            first.get_or_insert(win_start[0]);
                            last = win_end[N - 1];
                        }
                    }
                }
            };

            // Walk the raw text in clusters without materializing it: a peekable
            // `char_indices` finds each cluster's `[start, end)` byte range, and the
            // cluster substring is normalized straight into the window.
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
                    consider(ch, cluster_start, cluster_end)
                });
            }
        }
        first.map(|f| (f, last))
    }
}

/// Conjoining Hangul Vowel/Trailing jamo (U+1161..=U+1175, U+11A8..=U+11C2). Like
/// combining marks these attach to a preceding character — NFC composes a leading jamo
/// plus these into one syllable — so [`span`](NgramTokenizer::span) keeps them in the
/// same cluster; splitting the composition would lose the token for that region.
fn is_conjoining_jamo(c: char) -> bool {
    matches!(c, '\u{1161}'..='\u{1175}' | '\u{11A8}'..='\u{11C2}')
}

impl<const N: usize, const CAP: usize> NgramTokenizer<N, CAP> {
    /// Apply this tokenizer's normalization + casefolding to one combining-sequence
    /// cluster, emitting the resulting char(s) one at a time (no intermediate
    /// buffer). Mirrors [`prepare`](Self::prepare) exactly (same normalization form,
    /// same casefold, same `assume_normalized` skip) but on a cluster substring, so
    /// [`span`](Self::span) re-derives the same token stream while tracking offsets.
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

/// A script-segmented n-gram tokenizer (§6): it splits text into maximal same-script
/// runs and tokenizes each run with a script-appropriate window size, so a mixed-script
/// document indexes correctly without manufacturing cross-script grams at the seams.
///
/// `Common`/`Inherited` codepoints (digits, punctuation, combining marks) are
/// transparent — they extend the current run rather than breaking it. The default
/// [`WindowPolicy`] uses bigrams for the dense CJK scripts (Han / Hiragana / Katakana /
/// Hangul) and trigrams elsewhere. Its [`Token`](Tokenizer::Token) is a [`Trigram`]
/// (which also holds bigrams). Grams are capped at 3 codepoints (the term-encoding
/// storage ceiling). [`span`](Tokenizer::span) currently returns `None`; the
/// default-trigram tokenizer keeps its span support.
///
/// ```
/// use trifle::tokenize::{ScriptTokenizer, Tokenizer};
///
/// let tok = ScriptTokenizer::new();
/// // "ab漢字" → Latin trigram-window run "ab" (too short, no gram) + Han bigram "漢字".
/// let grams: Vec<String> = tok.tokenize("ab漢字").map(|g| g.to_string()).collect();
/// assert_eq!(grams, ["漢字"]);
/// ```
#[derive(Clone, Debug)]
pub struct ScriptTokenizer {
    normalization: Normalization,
    casefold: bool,
    assume_normalized: bool,
    policy: WindowPolicy,
}

/// Per-script-class window sizes for [`ScriptTokenizer`], indexed by the script-tag byte.
/// The default is trigrams everywhere except the dense CJK scripts, which take bigrams.
#[derive(Clone, Debug)]
pub struct WindowPolicy {
    sizes: [u8; 256],
}

impl Default for WindowPolicy {
    fn default() -> Self {
        let mut sizes = [3u8; 256];
        for tag in [
            crate::term::TAG_HAN,
            crate::term::TAG_HIRAGANA,
            crate::term::TAG_KATAKANA,
            crate::term::TAG_HANGUL,
        ] {
            sizes[tag as usize] = 2;
        }
        WindowPolicy { sizes }
    }
}

impl Default for ScriptTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl ScriptTokenizer {
    /// A script tokenizer with the default behavior: NFC + Unicode lowercasing and the
    /// default (CJK-bigram, else-trigram) window policy.
    pub fn new() -> Self {
        ScriptTokenizer {
            normalization: Normalization::Nfc,
            casefold: true,
            assume_normalized: false,
            policy: WindowPolicy::default(),
        }
    }

    /// Start a [builder](ScriptTokenizerBuilder) to customize normalization/casefolding.
    pub fn builder() -> ScriptTokenizerBuilder {
        ScriptTokenizerBuilder { inner: Self::new() }
    }

    /// Emit the windows of one same-script run into `out`. The window size is the run's
    /// script-class policy; a `Common`-only run uses the common-class size.
    fn emit_run(&self, out: &mut Vec<Trigram>, run: &[char], run_script: Option<Script>) {
        let tag = match run_script {
            Some(s) => crate::term::script_tag(s),
            None => crate::term::TAG_COMMON,
        };
        let n = self.policy.sizes[tag as usize] as usize;
        if n == 0 || run.len() < n {
            return;
        }
        for w in run.windows(n) {
            if let Some(g) = Trigram::from_chars(w) {
                out.push(g);
            }
        }
    }
}

impl Tokenizer for ScriptTokenizer {
    type Token = Trigram;

    fn tokenize<'a>(&'a self, text: &'a str) -> impl Iterator<Item = Self::Token> + 'a {
        let chars = prepare_chars(
            text,
            self.normalization,
            self.casefold,
            self.assume_normalized,
        );
        let mut out: Vec<Trigram> = Vec::new();
        // Walk the chars, breaking a run only at a change of *strong* script;
        // Common/Inherited codepoints are transparent and stay in the current run.
        let mut start = 0usize;
        let mut run_script: Option<Script> = None;
        for (i, &c) in chars.iter().enumerate() {
            match c.script() {
                Script::Common | Script::Inherited => {}
                s => match run_script {
                    None => run_script = Some(s),
                    Some(cur) if cur == s => {}
                    Some(_) => {
                        self.emit_run(&mut out, &chars[start..i], run_script);
                        start = i;
                        run_script = Some(s);
                    }
                },
            }
        }
        self.emit_run(&mut out, &chars[start..], run_script);
        out.into_iter()
    }

    fn fingerprint(&self) -> u64 {
        // FNV-1a over a canonical encoding of the behavior: a tag, normalization,
        // casefold, assume_normalized, the window policy, and a layout-version byte — so a
        // change to segmentation/window sizing forces a rebuild.
        let mut bytes = Vec::with_capacity(3 + 3 + 256 + 1);
        bytes.extend_from_slice(b"scr");
        bytes.push(self.normalization.code());
        bytes.push(self.casefold as u8);
        bytes.push(self.assume_normalized as u8);
        bytes.extend_from_slice(&self.policy.sizes);
        bytes.push(1); // encoding/layout version
        fnv1a_64(&bytes)
    }
}

/// Builder for [`ScriptTokenizer`].
#[derive(Clone, Debug)]
pub struct ScriptTokenizerBuilder {
    inner: ScriptTokenizer,
}

impl ScriptTokenizerBuilder {
    /// Set the normalization form (default [`Normalization::Nfc`]).
    pub fn normalization(mut self, normalization: Normalization) -> Self {
        self.inner.normalization = normalization;
        self
    }

    /// Enable or disable Unicode lowercasing (default `true`).
    pub fn casefold(mut self, casefold: bool) -> Self {
        self.inner.casefold = casefold;
        self
    }

    /// Trust that input is already in the chosen normalization form (default `false`).
    /// Sound only if the guarantee holds for *both* writes and queries.
    pub fn assume_normalized(mut self, assume_normalized: bool) -> Self {
        self.inner.assume_normalized = assume_normalized;
        self
    }

    /// Finish building the tokenizer.
    pub fn build(self) -> ScriptTokenizer {
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

        // `assume_normalized` changes the token stream on non-normalized input, so it
        // must change the fingerprint.
        let assume = TrigramTokenizer::builder()
            .assume_normalized(true)
            .build()
            .fingerprint();
        assert_ne!(
            a, assume,
            "assume_normalized change -> different fingerprint"
        );
    }

    #[test]
    fn fingerprint_distinguishes_cap_at_equal_window() {
        // Same window N=3, different byte capacity: must not collide (the old `as u8`
        // encoding made CAP=12 and CAP=268 hash equal — but CAP=268 is now a compile
        // error, so compare two *legal* widths that differ only in CAP).
        let trigram = NgramTokenizer::<3, 12>::new().fingerprint();
        let wide = NgramTokenizer::<3, 16>::new().fingerprint();
        assert_ne!(trigram, wide, "CAP is part of the fingerprint");
    }

    #[test]
    fn assume_normalized_matches_the_normalizing_path_on_conforming_input() {
        // On input already in NFC, the skip must produce the SAME tokens as the
        // normalizing path — proving it is a sound shortcut, not a behavior change.
        let normalizing = TrigramTokenizer::new();
        let assuming = TrigramTokenizer::builder().assume_normalized(true).build();
        let nfc_input = "caf\u{e9} r\u{e9}sum\u{e9}"; // precomposed, already NFC
        assert_eq!(grams(&normalizing, nfc_input), grams(&assuming, nfc_input));
    }

    #[test]
    fn length_changing_casefold_is_handled() {
        // 'İ'.to_lowercase() == ['i', '\u{307}'] — one source char folds to two.
        let tok = TrigramTokenizer::new();
        let grams = grams(&tok, "İst");
        // The fold expands, so the trigram stream starts at the folded form.
        assert_eq!(grams, ["i\u{307}s", "\u{307}st"]);
    }

    #[test]
    fn strip_marks_collapses_dotted_capital_i_with_plain_forms() {
        let tok = TrigramTokenizer::builder()
            .normalization(Normalization::NfdStripMarks)
            .build();
        // İ -> NFD [I, ̇ ] -> strip -> I -> fold -> i. So İ/I/i all collapse.
        assert_eq!(grams(&tok, "İab"), grams(&tok, "Iab"));
        assert_eq!(grams(&tok, "İab"), grams(&tok, "iab"));
        assert_eq!(grams(&tok, "İab"), ["iab"]);
    }

    #[test]
    fn from_chars_rejects_a_window_wider_than_cap() {
        // Three 4-byte code points = 12 bytes: fits Ngram<12>, not Ngram<9>.
        let wide = ['🚀', '🎉', '😀'];
        assert!(Ngram::<12>::from_chars(&wide).is_some());
        assert!(Ngram::<9>::from_chars(&wide).is_none());
    }

    #[test]
    fn undersized_cap_alias_silently_drops_only_the_too_wide_window() {
        // Ngram<9> can't hold a 3-emoji window (12 bytes); it is dropped, the rest kept.
        let tok = NgramTokenizer::<3, 9>::new();
        // "🚀🎉😀ab": windows [🚀🎉😀](dropped), [🎉😀a], [😀ab].
        assert_eq!(grams(&tok, "🚀🎉😀ab"), ["🎉😀a", "😀ab"]);
    }

    #[test]
    fn span_brackets_first_through_last_occurrence() {
        let tok = TrigramTokenizer::new();
        // "abc" occurs at byte 0 and byte 9; the span runs first-start..last-end.
        assert_eq!(tok.span("abcZZZZZZabc", &["abc"]), Some((0, 12)));
        // Single occurrence is tight around the word.
        assert_eq!(tok.span("zz abc zz", &["abc"]), Some((3, 6)));
        // No matching token -> no span.
        assert_eq!(tok.span("abcdef", &["zzz"]), None);
    }

    #[test]
    fn span_recovers_decomposed_input_under_nfc() {
        // Stored decomposed; the index composes the token "afé". The cluster-based
        // span must still bracket it (the old per-char span returned None here).
        let tok = TrigramTokenizer::new();
        let stored = "Xcafe\u{301}Y"; // X c a f e ́ Y
        let token: String = tok.tokenize(stored).map(|g| g.to_string()).nth(2).unwrap();
        assert_eq!(token, "af\u{e9}"); // afé (composed)
        let span = tok
            .span(stored, &[token.as_str()])
            .expect("a span for the match");
        assert!(stored.is_char_boundary(span.0) && stored.is_char_boundary(span.1));
        assert!(span.0 < span.1);
    }

    #[test]
    fn span_recovers_noncanonically_ordered_marks_under_nfd() {
        let tok = TrigramTokenizer::builder()
            .normalization(Normalization::Nfd)
            .build();
        // q + acute(ccc 230) + dot-below(ccc 220) — given in non-canonical order.
        let stored = "Xq\u{0301}\u{0323}Y";
        // The tokenizer canonically reorders to q, dot-below, acute.
        let tokens: Vec<String> = tok.tokenize(stored).map(|g| g.to_string()).collect();
        assert!(tokens.iter().any(|t| t.starts_with('q')));
        // Whichever token contains 'q', span must bracket it (per-char would miss it).
        let qtok = tokens.iter().find(|t| t.starts_with('q')).unwrap();
        assert!(tok.span(stored, &[qtok.as_str()]).is_some());
    }

    #[test]
    fn span_recovers_conjoining_hangul_jamo_under_nfc() {
        // Stored as conjoining jamo ᄀ ᅡ ᆨ (U+1100 U+1161 U+11A8); NFC composes them into
        // the single syllable "각". A per-cluster pass that split at each jamo (they are
        // starters, not combining marks) would never re-derive the composed token and
        // would return None — so this is the Hangul regression for the cluster walker.
        let tok = TrigramTokenizer::new();
        let stored = "\u{1100}\u{1161}\u{11A8}xy"; // composes to "각xy"
        let tokens: Vec<String> = tok.tokenize(stored).map(|g| g.to_string()).collect();
        assert!(
            tokens.iter().any(|t| t.starts_with('\u{ac01}')),
            "{tokens:?}"
        );
        let refs: Vec<&str> = tokens.iter().map(String::as_str).collect();
        let (lo, hi) = tok
            .span(stored, &refs)
            .expect("a span for the composed match");
        assert!(stored.is_char_boundary(lo) && stored.is_char_boundary(hi));
        // The match brackets the whole jamo run through "xy".
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
            let tok = TrigramTokenizer::builder().normalization(norm).build();
            for input in inputs {
                let tokens: Vec<String> = tok.tokenize(input).map(|g| g.to_string()).collect();
                let refs: Vec<&str> = tokens.iter().map(String::as_str).collect();
                if let Some((lo, hi)) = tok.span(input, &refs) {
                    assert!(
                        input.is_char_boundary(lo) && input.is_char_boundary(hi),
                        "span ({lo},{hi}) not on char boundaries for {input:?} / {norm:?}"
                    );
                    assert!(lo < hi && hi <= input.len());
                    // A caller WILL slice this — it must not panic.
                    let _ = &input[lo..hi];
                }
            }
        }
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

    #[test]
    fn script_tokenizer_segments_by_script_with_no_cross_script_grams() {
        let tok = ScriptTokenizer::new();
        let g = grams(&tok, "hello漢字");
        // Latin run -> trigrams; Han run -> one bigram.
        assert!(g.contains(&"hel".to_string()));
        assert!(g.contains(&"llo".to_string()));
        assert!(g.contains(&"漢字".to_string()));
        // No gram straddles the script boundary.
        assert!(g.iter().all(|t| !t.contains('漢') || t == "漢字"));
    }

    #[test]
    fn script_tokenizer_common_codepoints_are_transparent() {
        let tok = ScriptTokenizer::new();
        // A digit is Common: it stays in the Latin run rather than splitting it.
        assert_eq!(grams(&tok, "a1b"), ["a1b"]);
    }

    #[test]
    fn script_tokenizer_uses_bigrams_for_cjk() {
        let tok = ScriptTokenizer::new();
        assert_eq!(grams(&tok, "漢字漢"), ["漢字", "字漢"]);
    }

    #[test]
    fn script_tokenizer_fingerprint_differs_from_ngram_and_is_stable() {
        assert_eq!(
            ScriptTokenizer::new().fingerprint(),
            ScriptTokenizer::new().fingerprint()
        );
        assert_ne!(
            ScriptTokenizer::new().fingerprint(),
            TrigramTokenizer::new().fingerprint()
        );
    }
}
