//! The term model: a gram (up to 3 UTF-32 codepoints) plus a script tag, packed
//! big-endian into a `u128`.
//!
//! A [`Term`] is the identity the interning dictionary keys on. UTF-32 (fixed 4-byte
//! slots) over UTF-8 gives fixed-offset decode and unambiguous zero-fill — an all-zero
//! slot means "no codepoint" (U+0000 is normalized out). The script byte is the
//! most-significant byte, so terms sort script-contiguously *as `u128` values*; stored
//! as the `u128`'s big-endian bytes, the on-disk BLOB's memcmp order equals the value
//! order.
//!
//! **Storage-format ceiling:** UTF-32 makes "max 3 codepoints" a *storage* ceiling, not
//! just a tokenizer fact — [`encode_term`] returns `None` for a wider gram (callers
//! error, never silently drop). Raising it later needs a wider key + a rebuild.
//!
//! **Reserved bytes:** the low 24 bits are reserved and stay zero — load-bearing for a
//! future `WHERE term BETWEEN pack(s,0,0,0) AND pack(s,MAX,MAX,MAX)` script-range scan,
//! which is clean only while they are zero in both bounds. Using them would break that.
//!
//! The script byte is **derived from the gram's codepoints** ([`script_of`]) and never
//! set independently, so the same gram always yields the same term (identity cannot
//! fragment). The script-tag byte assignments below are part of on-disk identity:
//! renumbering them, or changing the layout, is a `SCHEMA_VERSION` change + rebuild.

use unicode_script::{Script, UnicodeScript};

/// A packed term: `[ script:8 | c0:32 | c1:32 | c2:32 | reserved:24=0 ]` (MSB→LSB).
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub(crate) struct Term(pub(crate) u128);

/// Pack three codepoints (zero = absent) and a script tag into the term layout.
#[inline]
pub(crate) fn pack(c0: u32, c1: u32, c2: u32, script: u8) -> u128 {
    ((script as u128) << 120)
        | ((c0 as u128) << 88)
        | ((c1 as u128) << 56)
        | ((c2 as u128) << 24) // low 24 bits reserved = 0
}

/// Reverse [`pack`] (debug / lossless decode). Explicit masks document that the script
/// bits are dropped from the codepoint slots.
#[allow(dead_code)]
#[inline]
pub(crate) fn unpack(t: u128) -> (u32, u32, u32, u8) {
    (
        ((t >> 88) & 0xFFFF_FFFF) as u32,
        ((t >> 56) & 0xFFFF_FFFF) as u32,
        ((t >> 24) & 0xFFFF_FFFF) as u32,
        (t >> 120) as u8,
    )
}

impl Term {
    /// The script class byte (the Welford class id) — the most-significant byte.
    #[inline]
    pub(crate) fn class(self) -> u8 {
        (self.0 >> 120) as u8
    }

    /// Reverse the encoding back to the gram string. Lossless for fitting grams; for
    /// debug/tooling only (no `TEXT` column is stored).
    #[allow(dead_code)]
    pub(crate) fn decode(self) -> String {
        let (a, b, c, _) = unpack(self.0);
        [a, b, c]
            .into_iter()
            .filter(|&x| x != 0)
            .filter_map(char::from_u32)
            .collect()
    }
}

/// Encode a gram into a [`Term`], or `None` if it exceeds the 3-codepoint storage
/// ceiling. The script tag is derived from the codepoints, so the same gram always maps
/// to the same term.
pub(crate) fn encode_term(gram: &str) -> Option<Term> {
    let mut cps = [0u32; 3];
    for (n, ch) in gram.chars().enumerate() {
        if n == 3 {
            return None; // storage-format ceiling
        }
        cps[n] = ch as u32; // U+0000 normalized out, so 0 always means "no codepoint"
    }
    let script = script_of(gram);
    Some(Term(pack(cps[0], cps[1], cps[2], script)))
}

/// The script tag for a gram: the script of its first *strong*-script codepoint
/// (`Common`/`Inherited` are transparent; an all-common gram is [`TAG_COMMON`]).
/// Scripts trifle distinguishes get a stable byte; everything else collapses to
/// [`TAG_OTHER`] — sound because two scripts collide only if their codepoints also
/// match (they do not), so terms never collide; it merely widens one Welford class.
pub(crate) fn script_of(gram: &str) -> u8 {
    for ch in gram.chars() {
        match ch.script() {
            Script::Common | Script::Inherited => continue,
            s => return script_tag(s),
        }
    }
    TAG_COMMON
}

// Stable script-tag bytes — part of on-disk term identity (never renumber without a
// schema-version bump + rebuild). `OTHER` is the catch-all for everything else. The CJK
// tags are `pub(crate)` so the script-segmented tokenizer's default window policy can
// pick them out (they take bigrams, not trigrams).
pub(crate) const TAG_COMMON: u8 = 0;
const TAG_LATIN: u8 = 1;
const TAG_CYRILLIC: u8 = 2;
const TAG_GREEK: u8 = 3;
pub(crate) const TAG_HAN: u8 = 4;
pub(crate) const TAG_HIRAGANA: u8 = 5;
pub(crate) const TAG_KATAKANA: u8 = 6;
pub(crate) const TAG_HANGUL: u8 = 7;
const TAG_ARABIC: u8 = 8;
const TAG_HEBREW: u8 = 9;
const TAG_DEVANAGARI: u8 = 10;
const TAG_THAI: u8 = 11;
const TAG_OTHER: u8 = 255;

/// The script-tag byte for a strong [`Script`] (the window-policy / Welford-class index).
pub(crate) fn script_tag(s: Script) -> u8 {
    match s {
        Script::Latin => TAG_LATIN,
        Script::Cyrillic => TAG_CYRILLIC,
        Script::Greek => TAG_GREEK,
        Script::Han => TAG_HAN,
        Script::Hiragana => TAG_HIRAGANA,
        Script::Katakana => TAG_KATAKANA,
        Script::Hangul => TAG_HANGUL,
        Script::Arabic => TAG_ARABIC,
        Script::Hebrew => TAG_HEBREW,
        Script::Devanagari => TAG_DEVANAGARI,
        Script::Thai => TAG_THAI,
        _ => TAG_OTHER,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_round_trips() {
        let t = pack('a' as u32, 'b' as u32, 'c' as u32, TAG_LATIN);
        assert_eq!(unpack(t), ('a' as u32, 'b' as u32, 'c' as u32, TAG_LATIN));
        // Low 24 bits (reserved) are zero.
        assert_eq!(t & 0xFF_FFFF, 0);
    }

    #[test]
    fn encode_decode_round_trips_for_fitting_grams() {
        for g in ["abc", "ab", "a", "déf"] {
            let term = encode_term(g).unwrap();
            assert_eq!(term.decode(), g);
        }
    }

    #[test]
    fn rejects_grams_over_three_codepoints() {
        assert!(encode_term("abcd").is_none());
        // Three multi-byte codepoints still fit (the ceiling is codepoints, not bytes).
        assert!(encode_term("日本語").is_some());
        assert!(encode_term("日本語学").is_none());
    }

    #[test]
    fn script_tag_is_derived_and_stable() {
        assert_eq!(encode_term("abc").unwrap().class(), TAG_LATIN);
        assert_eq!(encode_term("日本").unwrap().class(), TAG_HAN);
        // Digits / punctuation are Common.
        assert_eq!(encode_term("123").unwrap().class(), TAG_COMMON);
        // A leading common codepoint is transparent: the strong script wins.
        assert_eq!(encode_term("1ab").unwrap().class(), TAG_LATIN);
    }

    #[test]
    fn terms_sort_script_contiguously() {
        // Han (tag 4) sorts after Latin (tag 1) as a u128 value.
        let latin = encode_term("abc").unwrap();
        let han = encode_term("日本").unwrap();
        assert!(latin < han);
    }
}
