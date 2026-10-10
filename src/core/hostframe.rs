/// How deep guest calls may nest as host frames before falling back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostFrameConfig {
    /// Bounds host stack use; set against the thread's stack size. Exceeding it falls back to the linked exit.
    pub max_depth: u32,
}

impl HostFrameConfig {
    /// # Panics
    ///
    /// If `max_depth` is zero (use `None` to disable).
    pub fn new(max_depth: u32) -> Self {
        assert!(max_depth > 0, "a host-frame depth of zero disables nothing");
        Self { max_depth }
    }
}

/// Context offset of the host-frame depth ([`GuestArch::host_frame_offsets`](crate::core::GuestArch::host_frame_offsets)); `None` means never nests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostFrameOffsets {
    /// Offset of the `u32` live-frame count; the region maintains it, the host only zero-initialises it.
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
