//! A tiny deterministic PRNG (xorshift64*). No `rand` dependency, and a fixed seed
//! makes every corpus and query set byte-reproducible run to run.

pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        // Avoid the all-zero state, which xorshift cannot escape.
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in `[0, n)`.
    #[inline]
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// Uniform in `[lo, hi]`.
    #[inline]
    pub fn range(&mut self, lo: usize, hi: usize) -> usize {
        lo + self.below(hi - lo + 1)
    }

    /// `true` with probability `p`.
    #[inline]
    pub fn chance(&mut self, p: f64) -> bool {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64) < p
    }
}
