//! x86-64 guest register context: the flat `#[repr(C)]` struct passed to lifted code.
//!
//! Embeds [`ControlHeader`] (reached via `Deref`/`DerefMut`), shared by every guest.

/// `MXCSR` after processor reset: exceptions masked, round-nearest, FZ/DAZ off.
/// The [`Default`] for [`RegisterContext::mxcsr`]; zero would unmask every SSE exception.
pub const MXCSR_RESET: u32 = 0x0000_1F80;

/// Defined `MXCSR` bits; writing reserved bits 31:16 raises `#GP(0)`, see
/// [`crate::arch::x86_64::Lifter::handle_ldmxcsr`].
pub const MXCSR_MASK: u32 = 0x0000_FFFF;

/// x87 control word after `FNINIT`/reset: exceptions masked, 64-bit precision, round-nearest.
pub const X87_CW_RESET: u16 = 0x037F;

/// `CS` in 64-bit user mode on Darwin (`USER64_CS`).
pub const CS_USER64_DARWIN: u16 = 0x2B;
/// `SS` in 64-bit user mode on Darwin (`USER_DS`).
pub const SS_USER64_DARWIN: u16 = 0x23;
/// `CS` in 64-bit user mode on Linux (`__USER_CS`, RPL 3).
pub const CS_USER64_LINUX: u16 = 0x33;
/// `SS` in 64-bit user mode on Linux (`__USER_DS`, RPL 3).
pub const SS_USER64_LINUX: u16 = 0x2B;

use crate::core::align::Align16;
use crate::core::exit::ControlHeader;

/// Host routines lifted code calls inline; contract: [`crate::core::callback`].
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct X86_64Callbacks {
    /// Services one `SYSCALL`; `rip` is the return address.
    pub handle_syscall: unsafe extern "C" fn(ctx: *mut RegisterContext),
    /// Interprets `num_instructions` unlifted instructions from `rip`.
    pub interpreter_fallback:
        unsafe extern "C" fn(ctx: *mut RegisterContext, num_instructions: u64),
}

/// Aborts: running on past an unserviced syscall would corrupt the guest.
unsafe extern "C" fn default_syscall_handler(_ctx: *mut RegisterContext) {
    panic!("no syscall callback registered");
}

/// Aborts, like [`default_syscall_handler`].
unsafe extern "C" fn default_interpreter_fallback(
    _ctx: *mut RegisterContext,
    _num_instructions: u64,
) {
    panic!("no interpreter-fallback callback registered");
}

impl Default for X86_64Callbacks {
    fn default() -> Self {
        Self {
            handle_syscall: default_syscall_handler,
            interpreter_fallback: default_interpreter_fallback,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RegisterContext {
    pub rax: u64,
    pub rcx: u64,
    pub rdx: u64,
    pub rbx: u64,
    pub rsp: u64,
    pub rbp: u64,
    pub rsi: u64,
    pub rdi: u64,
    pub r8: u64,
    pub r9: u64,
    pub r10: u64,
    pub r11: u64,
    pub r12: u64,
    pub r13: u64,
    pub r14: u64,
    pub r15: u64,
    pub rip: u64,
    pub callbacks: X86_64Callbacks,
    /// The host's block-link table for this thread, or null. Written only by the host; see [`crate::core::link`].
    pub link_table: *mut crate::core::link::BlockLinkEntry,
    pub cf: u8,
    /// **A byte whose parity is PF**, not PF itself; hosts must use [`Self::pf`] / [`Self::set_pf`].
    pub pf_src: u8,
    /// **A byte whose bit 4 is AF** (`lhs ^ rhs ^ res`), not AF itself. See [`Self::af`], [`Self::pf_src`].
    pub af_src: u8,
    pub zf: u8,
    pub sf: u8,
    pub of: u8,
    pub df: u8,
    pub control: ControlHeader,
    /// XMM0-XMM15 as [lo_64, hi_64]; 16-byte aligned for single-access moves.
    pub xmm: Align16<[[u64; 2]; 16]>,
    pub fs_base: u64,
    pub gs_base: u64,

    /// x87 registers as `[mantissa, sign_exp]` in FXSAVE `_st[i]` layout (explicit integer bit at 63).
    /// `ST(i)` is slot `(fpu_top + i) & 7`.
    pub fpr: [[u64; 2]; 8],
    /// Top-of-stack pointer (0-7).
    pub fpu_top: u16,
    /// Condition codes C0-C3 at bits 8, 9, 10, 14; `TOP` is materialised from `fpu_top` on `FNSTSW`.
    pub fpu_sw: u16,
    pub fpu_cw: u16,
    /// Abridged (`FXSAVE`) tag word: bit `i` set when physical register `i` is valid.
    /// The full 2-bit tag word of `FNSTENV`/`FLDENV` is derived from it.
    pub fpu_tag: u16,

    pub mxcsr: u32,

    pub cs: u16,
    pub ss: u16,
    pub ds: u16,
    pub es: u16,
    /// `FS` selector; independent of [`fs_base`](Self::fs_base).
    pub fs: u16,
    pub gs: u16,

    /// Next link row for the exited region ([`BlockLinkEntry`](crate::core::link::BlockLinkEntry)
    /// address in [`link_table`](Self::link_table)), or zero for "service this exit yourself".
    pub link_hit: u64,
    /// Base of the call-return ring ([`CallRetEntry`](crate::core::callret::CallRetEntry)), or null (prediction off).
    /// Must be [`CallRetConfig::bytes`] long, 8-byte aligned and zeroed.
    ///
    /// [`CallRetConfig::bytes`]: crate::core::callret::CallRetConfig::bytes
    pub callret_base: *mut crate::core::callret::CallRetEntry,
    /// Byte offset of the top ring frame; any value is valid, so push/pop need no bounds check.
    pub callret_off: u64,
    /// Live guest calls as host frames ([`crate::core::hostframe`]); maintained by lifted code, host zeroes it once.
    pub host_frame_depth: u32,
    /// Padding so the struct size is independent of what follows `host_frame_depth`.
    pub _pad_host_frame: u32,
}

impl Default for RegisterContext {
    /// Processor post-reset state: zero except `mxcsr` and `fpu_cw` ([`MXCSR_RESET`], [`X87_CW_RESET`]).
    /// `cs`/`ss` default to Darwin's values; Linux/Windows hosts should overwrite them
    /// ([`CS_USER64_LINUX`], [`SS_USER64_LINUX`]).
    fn default() -> Self {
        Self {
            rax: 0,
            rcx: 0,
            rdx: 0,
            rbx: 0,
            rsp: 0,
            rbp: 0,
            rsi: 0,
            rdi: 0,
            r8: 0,
            r9: 0,
            r10: 0,
            r11: 0,
            r12: 0,
            r13: 0,
            r14: 0,
            r15: 0,
            rip: 0,
            callbacks: X86_64Callbacks::default(),
            link_table: std::ptr::null_mut(),
            cf: 0,
            // Odd parity: the flag is 0, and RFLAGS after reset has PF clear.
            pf_src: 1,
            af_src: 0,
            zf: 0,
            sf: 0,
            of: 0,
            df: 0,
            control: ControlHeader::ZERO,
            xmm: Align16([[0; 2]; 16]),
            fs_base: 0,
            gs_base: 0,
            fpr: [[0; 2]; 8],
            fpu_top: 0,
            fpu_sw: 0,
            fpu_cw: X87_CW_RESET,
            fpu_tag: 0,
            mxcsr: MXCSR_RESET,
            cs: CS_USER64_DARWIN,
            ss: SS_USER64_DARWIN,
            ds: 0,
            es: 0,
            fs: 0,
            gs: 0,
            link_hit: 0,
            callret_base: std::ptr::null_mut(),
            callret_off: 0,
            host_frame_depth: 0,
            _pad_host_frame: 0,
        }
    }
}

impl RegisterContext {
    pub fn pf(&self) -> bool {
        self.pf_src.count_ones().is_multiple_of(2)
    }

    pub fn set_pf(&mut self, pf: bool) {
        self.pf_src = u8::from(!pf);
    }

    pub fn af(&self) -> bool {
        self.af_src & 0x10 != 0
    }

    pub fn set_af(&mut self, af: bool) {
        self.af_src = u8::from(af) << 4;
    }

    pub fn new() -> Self {
        Self::default()
    }
}

/// Reads control fields as if declared inline (`ctx.exit_reason`).
impl std::ops::Deref for RegisterContext {
    type Target = ControlHeader;

    fn deref(&self) -> &ControlHeader {
        &self.control
    }
}

impl std::ops::DerefMut for RegisterContext {
    fn deref_mut(&mut self) -> &mut ControlHeader {
        &mut self.control
    }
}

/// Converts an 80-bit x87 value to `f64` bits; exact for zero and in-range normals
/// (round-trips with [`f64_bits_to_f80`]), else truncates and flushes denormals.
#[inline]
pub fn f80_to_f64_bits(mantissa: u64, sign_exp: u16) -> u64 {
    let sign = ((sign_exp >> 15) & 1) as u64;
    if mantissa == 0 {
        // +=0 (also the encoding produced for flushed/true zeros).
        return sign << 63;
    }
    let exp80 = (sign_exp & 0x7FFF) as u64;
    // f80 bias 16383, f64 bias 1023 -> exp64 = exp80 - 15360.
    let exp64 = exp80.wrapping_sub(15360) & 0x7FF;
    // Drop the explicit integer bit; the f64 fraction is the next 52 bits.
    let frac = (mantissa >> 11) & 0x000F_FFFF_FFFF_FFFF;
    (sign << 63) | (exp64 << 52) | frac
}

/// Converts `f64` bits to `(mantissa, sign_exp)`, the layout [`f80_to_f64_bits`] consumes.
#[inline]
pub fn f64_bits_to_f80(bits: u64) -> (u64, u16) {
    let sign = (bits >> 63) & 1;
    let exp64 = (bits >> 52) & 0x7FF;
    let frac = bits & 0x000F_FFFF_FFFF_FFFF;
    if exp64 == 0 && frac == 0 {
        return (0, (sign as u16) << 15);
    }
    // Explicit integer bit set + fraction shifted up by 11.
    let mantissa = (1u64 << 63) | (frac << 11);
    let exp80 = if exp64 == 0x7FF {
        0x7FFF // Inf / NaN
    } else {
        exp64 + 15360
    };
    (mantissa, ((sign as u16) << 15) | exp80 as u16)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `xmm`'s alignment saves 22 instructions per region entry; this checks the padding before it still adds up.
    #[test]
    fn the_vector_save_area_stays_sixteen_byte_aligned() {
        assert_eq!(std::mem::offset_of!(RegisterContext, xmm) % 16, 0);
    }

    /// Link and call-return fields are pointer-aligned, as the probe's notrap aligned accesses assume.
    #[test]
    fn the_link_fields_are_aligned_for_the_probe() {
        assert_eq!(std::mem::offset_of!(RegisterContext, link_table) % 8, 0);
        assert_eq!(std::mem::offset_of!(RegisterContext, link_hit) % 8, 0);
        assert_eq!(std::mem::offset_of!(RegisterContext, callret_base) % 8, 0);
        assert_eq!(std::mem::offset_of!(RegisterContext, callret_off) % 8, 0);
    }

    /// Offsets at or below `xmm` may be hardcoded by hosts: append fields, never insert.
    #[test]
    fn appending_a_field_does_not_move_the_ones_below_it() {
        assert_eq!(std::mem::offset_of!(RegisterContext, link_table), 152);
        assert_eq!(std::mem::offset_of!(RegisterContext, xmm), 192);
        assert_eq!(std::mem::offset_of!(RegisterContext, link_hit), 616);
    }

    /// A fresh context links nothing: null table makes the probe a return, zero `link_hit` is inert.
    #[test]
    fn a_default_context_is_unlinked() {
        let ctx = RegisterContext::new();
        assert!(ctx.link_table.is_null());
        assert_eq!(ctx.link_hit, 0);
        assert!(ctx.callret_base.is_null());
        assert_eq!(ctx.host_frame_depth, 0);
    }
}
