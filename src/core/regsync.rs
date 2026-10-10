#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegSlot {
    Gpr(u8),
    Flag(u8),
    /// Vector register, or its low half where split.
    Vec(u8),
    /// High half where split.
    VecHi(u8),
    /// Segment bases, `SP`, thread pointer, etc.
    Extra(u8),
}
