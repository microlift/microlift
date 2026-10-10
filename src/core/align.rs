//! Alignment carried by the type rather than by a padding field.

/// A `T` on a 16-byte boundary, so aligned loads of the XMM save area stay valid
/// however fields above it move. `Deref` makes it transparent (`ctx.xmm[3]`).
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
