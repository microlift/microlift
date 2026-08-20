#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegSlot {
    /// A general-purpose register.
    Gpr(u8),
    /// A condition flag.
    Flag(u8),
    /// A vector register, or its low half where the two are separate variables.
    Vec(u8),
    /// The high half of a vector register, where that is a separate variable.
    VecHi(u8),
    /// Anything else the frontend syncs: segment bases, `SP`, a thread pointer.
    Extra(u8),
}
