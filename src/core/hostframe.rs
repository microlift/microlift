/// How deep guest calls may nest as host frames before falling back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostFrameConfig {
    /// The most host frames guest calls may occupy at once.
    ///
    /// Each is a real stack frame of a compiled region, so this is a bound on
    /// host stack use and wants setting against the thread's stack size rather
    /// than against how deep the guest is expected to go. Exceeding it is not an
    /// error and not a fault path: the call takes the linked exit it would have
    /// taken with this mechanism off.
    pub max_depth: u32,
}

impl HostFrameConfig {
    /// Nest at most `max_depth` guest calls as host frames.
    ///
    /// # Panics
    ///
    /// If `max_depth` is zero, which would make every call take the fallback and
    /// is a configuration mistake rather than a way to switch the feature off -
    /// `None` is that.
    pub fn new(max_depth: u32) -> Self {
        assert!(max_depth > 0, "a host-frame depth of zero disables nothing");
        Self { max_depth }
    }
}

/// Where a guest's `RegisterContext` keeps the host-frame depth.
///
/// Returned by [`GuestArch::host_frame_offsets`](crate::core::GuestArch::host_frame_offsets);
/// a guest that has not made room for it returns `None` and never nests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostFrameOffsets {
    /// Byte offset of the `u32` counting how many guest calls are currently live
    /// as host frames.
    ///
    /// Incremented before a nested call and restored after it, by the region
    /// making the call - so it is correct however that call ends, including the
    /// unwinding path. A host initialises it to zero and never touches it again.
    pub depth: i32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "disables nothing")]
    fn a_zero_depth_is_refused_rather_than_taken_as_off() {
        HostFrameConfig::new(0);
    }
}
