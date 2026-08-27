//! Alignment carried by the type rather than by a padding field.

/// A `T` the compiler is told to place on a 16-byte boundary.
///
/// For a `#[repr(C)]` context field whose alignment lifted code depends on -
/// x86-64's XMM save area, where an aligned load is worth twenty-two
/// instructions per region entry. A hand-placed `u64` of padding buys the same
/// thing until someone adds or removes a field above it and silently takes it
/// away again; this cannot be taken away, because the alignment is a property
/// of the field's type and the compiler re-pads around it.
///
/// [`Deref`](std::ops::Deref) makes the wrapper invisible at every use that
/// reaches *into* the value (`ctx.xmm[3]`); one that wants the value itself
/// dereferences explicitly (`*ctx.xmm`).
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Align16<T>(pub T);

impl<T> std::ops::Deref for Align16<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T> std::ops::DerefMut for Align16<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

impl<T> From<T> for Align16<T> {
    fn from(value: T) -> Self {
        Self(value)
    }
}
