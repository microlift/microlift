#[derive(Clone)]
pub struct Rng(pub u64);

impl Rng {
    /// Create a generator from `seed`.
    pub fn new(seed: u64) -> Self {
        Rng(seed)
    }

    /// Next 64-bit output.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform value in `0..n`. Panics if `n == 0`.
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// A coin flip.
    pub fn chance(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }
}
