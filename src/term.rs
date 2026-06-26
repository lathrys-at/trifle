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

/// A packed term: a gram (≤3 codepoints) plus a script tag, `[ script:8 | c0:32 | c1:32
/// | c2:32 | reserved:24=0 ]` (MSB→LSB). Produced by [`IntoTerm::term`]; opaque to
/// callers (the encoding is an internal detail).
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct Term(pub(crate) u128);

/// Pack three codepoints (zero = absent) and a script tag into the term layout.
#[inline]
pub(crate) fn pack(c0: u32, c1: u32, c2: u32, script: u8) -> u128 {
    ((script as u128) << 120) | ((c0 as u128) << 88) | ((c1 as u128) << 56) | ((c2 as u128) << 24) // low 24 bits reserved = 0
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
    /// The script class byte (the Welford class id) — the most-significant byte, equal
    /// to the gram's strong script as `unicode_script::Script as u8`.
    #[inline]
    pub fn class(self) -> u8 {
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

/// The script class byte for a gram: the script of its first *strong*-script codepoint
/// as `unicode_script::Script as u8` (`Common`/`Inherited` are transparent; an all-common
/// gram is `Script::Common`). The `Script` enum is `#[repr(u8)]` with explicit, distinct
/// discriminants for every script, so this is complete (no hand table) and is the same
/// byte the Welford pruner uses as its class id.
///
/// **On-disk identity:** this byte is stored in the term encoding, so it ties identity to
/// `unicode_script`'s discriminants. They are explicit (intentional), and the crate is
/// pinned; a version that renumbered scripts would need a rebuild (bump the tokenizer
/// fingerprint).
pub(crate) fn script_of(gram: &str) -> u8 {
    for ch in gram.chars() {
        match ch.script() {
            Script::Common | Script::Inherited => continue,
            s => return s as u8,
        }
    }
    Script::Common as u8
}

/// Conversion of a gram-like value into its packed [`Term`]. Blanket-implemented for
/// every `Borrow<str>`, so a string-backed tokenizer token gets it for free; the default
/// derives the script and packs the gram's code points. The
/// [`Tokenizer`](crate::tokenize::Tokenizer) token type is bound to this, so interning a
/// token into a term goes through the token itself rather than back through a `&str`.
pub trait IntoTerm: std::borrow::Borrow<str> {
    /// The packed term for this gram, or `None` if it exceeds the 3-codepoint ceiling.
    fn term(&self) -> Option<Term> {
        encode_term(self.borrow())
    }
}

impl<T: std::borrow::Borrow<str>> IntoTerm for T {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_round_trips() {
        let latin = Script::Latin as u8;
        let t = pack('a' as u32, 'b' as u32, 'c' as u32, latin);
        assert_eq!(unpack(t), ('a' as u32, 'b' as u32, 'c' as u32, latin));
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
    fn script_class_is_derived_and_stable() {
        assert_eq!(encode_term("abc").unwrap().class(), Script::Latin as u8);
        assert_eq!(encode_term("日本").unwrap().class(), Script::Han as u8);
        // Digits / punctuation are Common.
        assert_eq!(encode_term("123").unwrap().class(), Script::Common as u8);
        // A leading common codepoint is transparent: the strong script wins.
        assert_eq!(encode_term("1ab").unwrap().class(), Script::Latin as u8);
    }

    #[test]
    fn terms_sort_script_contiguously() {
        // The script byte is the MSB, so same-script grams order by codepoints and a
        // script change dominates the ordering (which direction depends on the script
        // discriminants — we assert the structure, not a specific cross-script order).
        let ab = encode_term("abc").unwrap();
        let abd = encode_term("abd").unwrap();
        assert!(ab < abd, "same script -> ordered by codepoints");
        let han = encode_term("日本").unwrap();
        assert_ne!(
            ab.class(),
            han.class(),
            "different scripts -> different MSB"
        );
    }
}
