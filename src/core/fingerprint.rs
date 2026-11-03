//! Hashes that name what compiled code depends on, stable across runs and builds.

use std::hash::{Hash, Hasher};

use crate::core::LifterConfig;

/// FNV-1a over exactly the bytes written: unlike `DefaultHasher`, its output is fixed
/// across Rust releases, so a fingerprint can be stored.
pub struct StableHasher(u64);

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

impl StableHasher {
    pub fn new() -> Self {
        Self(FNV_OFFSET)
    }

    /// Length-prefixed, so adjacent strings cannot run together.
    pub fn str(&mut self, s: &str) {
        self.write_u64(s.len() as u64);
        self.write(s.as_bytes());
    }
}

impl Default for StableHasher {
    fn default() -> Self {
        Self::new()
    }
}

impl Hasher for StableHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 ^= u64::from(byte);
            self.0 = self.0.wrapping_mul(FNV_PRIME);
        }
    }

    // Fixed width and byte order, so a fingerprint means the same on every host.
    fn write_usize(&mut self, n: usize) {
        self.write_u64(n as u64);
    }

    fn write_isize(&mut self, n: isize) {
        self.write_u64(n as u64);
    }

    fn write_u16(&mut self, n: u16) {
        self.write(&n.to_le_bytes());
    }

    fn write_u32(&mut self, n: u32) {
        self.write(&n.to_le_bytes());
    }

    fn write_u64(&mut self, n: u64) {
        self.write(&n.to_le_bytes());
    }

    fn write_u128(&mut self, n: u128) {
        self.write(&n.to_le_bytes());
    }
}

impl LifterConfig {
    /// A hash of every setting that changes emitted code.
    ///
    /// Excludes what varies per region or per run and never reaches the code as a
    /// constant: segments, imports and the registry (recorded per region as
    /// [`LiftedCode::import_decisions`](crate::core::LiftedCode::import_decisions)),
    /// block ranges, resume points and [`guest_slide`](Self::guest_slide). `execution_mode`
    /// selects nothing.
    pub fn codegen_fingerprint(&self) -> u64 {
        let Self {
            execution_mode: _,
            address_mode,
            guest_slide: _,
            segments: _,
            interpreter_fallback,
            resumption,
            analyze_jump_tables,
            imports: _,
            import_registry: _,
            indirect_call_bridge,
            use_vector_xmm,
            specialization,
            atomic128,
            total_store_order,
            generated_nan,
            analysis_mode,
            exit_trampoline_address,
            outline_helpers,
            emit_source_locs,
            lazy_register_sync,
            region_tail_exit,
            page_size,
            block_ranges: _,
            resume_points: _,
            eliminate_dead_flags,
            strict_resume,
            block_link,
            inline_link_probes,
            link_cells,
            call_return_stack,
            host_call_frames,
            optimizations,
        } = self;
        let mut hasher = StableHasher::new();
        address_mode.hash(&mut hasher);
        interpreter_fallback.hash(&mut hasher);
        resumption.hash(&mut hasher);
        analyze_jump_tables.hash(&mut hasher);
        if let Some(bridge) = indirect_call_bridge {
            hasher.str(bridge.name);
            bridge.effects.hash(&mut hasher);
            matches!(bridge.target, crate::core::bridge::BridgeTarget::Slot(_)).hash(&mut hasher);
        }
        indirect_call_bridge.is_some().hash(&mut hasher);
        use_vector_xmm.hash(&mut hasher);
        specialization.hash(&mut hasher);
        atomic128.hash(&mut hasher);
        total_store_order.hash(&mut hasher);
        generated_nan.hash(&mut hasher);
        analysis_mode.hash(&mut hasher);
        exit_trampoline_address.hash(&mut hasher);
        outline_helpers.hash(&mut hasher);
        emit_source_locs.hash(&mut hasher);
        lazy_register_sync.hash(&mut hasher);
        region_tail_exit.hash(&mut hasher);
        page_size.hash(&mut hasher);
        eliminate_dead_flags.hash(&mut hasher);
        strict_resume.hash(&mut hasher);
        block_link.hash(&mut hasher);
        inline_link_probes.hash(&mut hasher);
        link_cells.hash(&mut hasher);
        call_return_stack.hash(&mut hasher);
        host_call_frames.hash(&mut hasher);
        optimizations.hash(&mut hasher);
        hasher.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::link::BlockLinkConfig;
    use crate::core::{AddressMode, Atomic128Mode};

    #[test]
    fn codegen_settings_change_the_fingerprint() {
        let base = LifterConfig::default();
        let fingerprint = base.codegen_fingerprint();
        let changed = [
            LifterConfig {
                address_mode: AddressMode::Relocated,
                ..LifterConfig::default()
            },
            LifterConfig {
                atomic128: Atomic128Mode::Interpret,
                ..LifterConfig::default()
            },
            LifterConfig {
                total_store_order: true,
                ..LifterConfig::default()
            },
            LifterConfig {
                block_link: Some(BlockLinkConfig::new(1024)),
                ..LifterConfig::default()
            },
            LifterConfig {
                block_link: Some(BlockLinkConfig::new(1024).with_ways(2)),
                ..LifterConfig::default()
            },
        ];
        for config in changed {
            assert_ne!(config.codegen_fingerprint(), fingerprint, "{config:?}");
        }
    }

    #[test]
    fn per_run_inputs_leave_the_fingerprint_alone() {
        let mut config = LifterConfig {
            guest_slide: 0x4000,
            resume_points: vec![0x1000],
            ..LifterConfig::default()
        };
        config.register_slow_import(0x2000);
        assert_eq!(
            config.codegen_fingerprint(),
            LifterConfig::default().codegen_fingerprint()
        );
    }

    #[test]
    fn the_hasher_is_fnv_1a() {
        let mut hasher = StableHasher::new();
        hasher.write(b"a");
        assert_eq!(hasher.finish(), 0xaf63_dc4c_8601_ec8c);
    }
}
