//! A small, fast **FxHash** (rotate-xor-multiply) hasher and the [`FxHashMap`]/[`FxHashSet`]
//! aliases trifle uses in place of std's SipHash.
//!
//! trifle is a derived, rebuildable local cache with **no adversarial-input threat model**, so
//! SipHash's DoS-resistance is pure overhead on the hottest path — hashing `u128` term keys,
//! `u32` ids, and short gram strings millions of times per search/build. FxHash (the
//! `rustc-hash` algorithm) is far cheaper for these key types. Implementing it inline, rather
//! than taking a `rustc-hash` dependency, keeps trifle's curated, minimal dependency set.
//!
//! Non-cryptographic and not DoS-resistant by design — never feed it untrusted keys where a
//! collision flood would matter (trifle's keys are internal ids and tokenizer-normalized grams).

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};

/// A [`HashMap`] keyed with [`FxHasher`].
pub(crate) type FxHashMap<K, V> = HashMap<K, V, BuildHasherDefault<FxHasher>>;
/// A [`HashSet`] keyed with [`FxHasher`].
pub(crate) type FxHashSet<K> = HashSet<K, BuildHasherDefault<FxHasher>>;

/// The FxHash multiplier (a 64-bit odd constant) and per-word rotation.
const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;
const ROTATE: u32 = 5;

/// A fast, non-cryptographic FxHash hasher (rotate-xor-multiply). The integer `write_*` methods
/// are specialized for the key classes trifle hashes on the hot path (`u32` ids, `u64`, `u128`
/// term keys); the generic [`write`](Hasher::write) byte path serves `&str` / token keys.
#[derive(Default)]
pub(crate) struct FxHasher {
    hash: u64,
}

impl FxHasher {
    #[inline]
    fn add(&mut self, word: u64) {
        self.hash = (self.hash.rotate_left(ROTATE) ^ word).wrapping_mul(SEED);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, mut bytes: &[u8]) {
        while let Some(chunk) = bytes.first_chunk::<8>() {
            self.add(u64::from_le_bytes(*chunk));
            bytes = &bytes[8..];
        }
        if let Some(chunk) = bytes.first_chunk::<4>() {
            self.add(u32::from_le_bytes(*chunk) as u64);
            bytes = &bytes[4..];
        }
        if let Some(chunk) = bytes.first_chunk::<2>() {
            self.add(u16::from_le_bytes(*chunk) as u64);
            bytes = &bytes[2..];
        }
        if let Some(&b) = bytes.first() {
            self.add(b as u64);
        }
    }
    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.add(i as u64);
    }
    #[inline]
    fn write_u16(&mut self, i: u16) {
        self.add(i as u64);
    }
    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.add(i as u64);
    }
    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }
    #[inline]
    fn write_u128(&mut self, i: u128) {
        self.add(i as u64);
        self.add((i >> 64) as u64);
    }
    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }
    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::Hash;

    fn hash_of<T: Hash>(v: &T) -> u64 {
        let mut h = FxHasher::default();
        v.hash(&mut h);
        h.finish()
    }

    #[test]
    fn deterministic_and_distinguishes_keys() {
        // Same key → same hash; different keys → (almost surely) different.
        assert_eq!(hash_of(&123u128), hash_of(&123u128));
        assert_ne!(hash_of(&1u128), hash_of(&2u128));
        assert_ne!(hash_of(&1u32), hash_of(&2u32));
        assert_ne!(hash_of(&"abc"), hash_of(&"abd"));
    }

    #[test]
    fn map_round_trips_each_key_class() {
        let mut m: FxHashMap<u128, &str> = FxHashMap::default();
        m.insert(1, "a");
        m.insert(u128::MAX, "max");
        assert_eq!(m.get(&1), Some(&"a"));
        assert_eq!(m.get(&u128::MAX), Some(&"max"));
        assert_eq!(m.get(&2), None);

        // A set probed by &str via Borrow must find an owned String inserted under the same text.
        let mut s: FxHashSet<String> = FxHashSet::default();
        s.insert("gram".to_string());
        assert!(s.contains("gram"));
        assert!(!s.contains("nope"));
    }
}
