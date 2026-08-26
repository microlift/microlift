//! x86-64 guest register context (the FFI struct passed to lifted code).
//!
//! A flat `#[repr(C)]` struct, except for the architecture-neutral control
//! fields, which are the embedded [`ControlHeader`] every guest shares.
//! `Deref`/`DerefMut` forward to it, so `ctx.exit_reason` and
//! `ctx.image_base = base` work as if those fields were declared here.

/// The value `MXCSR` holds after a processor reset / `FXRSTOR` of a fresh
/// state: all six exception flags clear, all six exception masks set, round to
/// nearest-even, FZ and DAZ off. Used as the [`Default`] for
/// [`RegisterContext::mxcsr`] so that a freshly constructed context is one the
/// hardware could actually be in - a zeroed `MXCSR` is not, since it unmasks
/// every SSE exception.
pub const MXCSR_RESET: u32 = 0x0000_1F80;

/// The `MXCSR` bits that are defined; bits 31:16 are reserved and writing a
/// non-zero value to any of them raises `#GP(0)`. See
/// [`crate::arch::x86_64::Lifter::handle_ldmxcsr`] for how that is handled.
pub const MXCSR_MASK: u32 = 0x0000_FFFF;

/// The x87 control word after `FNINIT` / a processor reset: all six exception
/// masks set, extended (64-bit) precision, round to nearest-even.
pub const X87_CW_RESET: u16 = 0x037F;

/// `CS` in 64-bit user mode on Darwin (`USER64_CS`).
pub const CS_USER64_DARWIN: u16 = 0x2B;
/// `SS` in 64-bit user mode on Darwin (`USER_DS`).
pub const SS_USER64_DARWIN: u16 = 0x23;
/// `CS` in 64-bit user mode on Linux (`__USER_CS`, RPL 3).
pub const CS_USER64_LINUX: u16 = 0x33;
/// `SS` in 64-bit user mode on Linux (`__USER_DS`, RPL 3).
pub const SS_USER64_LINUX: u16 = 0x2B;

use crate::core::exit::ControlHeader;

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
    /// A syscall handler intended for the RuntimeAssisted execution mode.
    pub syscall_handler: Option<unsafe extern "C" fn(ctx: *mut RegisterContext)>,
    /// The host's block-link table for the thread running this context, or null
    /// when it has none. See [`crate::core::link`].
    ///
    /// Read by lifted code at every exit that continues at a guest address, and
    /// written by nothing but the host.
    pub link_table: *mut crate::core::link::BlockLinkEntry,
    // Flags as individual bytes for simpler IR load/store
    pub cf: u8,
    /// **A byte whose parity is PF**, not PF itself. See [`Self::pf`].
    ///
    /// PF is defined as the even parity of an operation's low result byte, so
    /// the result is already a valid stored form and lifted code keeps it
    /// without folding: the fold happens at the few instructions that read PF.
    /// A host reading or writing the flag must go through [`Self::pf`] /
    /// [`Self::set_pf`].
    pub pf_src: u8,
    /// **A byte whose bit 4 is AF**, not AF itself - `lhs ^ rhs ^ res` for the
    /// operation that last wrote it. See [`Self::af`] and [`Self::pf_src`].
    pub af_src: u8,
    pub zf: u8,
    pub sf: u8,
    pub of: u8,
    pub df: u8,
    pub control: ControlHeader,
    /// Padding, so that [`xmm`](Self::xmm) begins at a 16-byte-aligned
    /// offset.
    pub _align_xmm: u64,
    /// XMM0–XMM15, each stored as [lo_64, hi_64].
    pub xmm: [[u64; 2]; 16],
    pub fs_base: u64,
    pub gs_base: u64,

    /// x87 data registers in true 80-bit extended precision, one physical register
    /// per 16-byte slot laid out as `[mantissa, sign_exp]`: `[0]` is the 64-bit
    /// significand (explicit integer bit at 63), `[1]` the sign bit (15) and 15-bit
    /// biased exponent in the low 16 bits.
    ///
    /// This matches the FXSAVE `_st[i]` layout, so a host can copy each register
    /// to/from the native x87 save area with no format conversion. `ST(i)` maps to
    /// physical slot `(fpu_top + i) & 7`.
    pub fpr: [[u64; 2]; 8],
    /// Top-of-stack pointer (0-7). `ST(0)` lives in `fpr[fpu_top]`.
    pub fpu_top: u16,
    /// x87 status-word condition-code bits (C0-C3 in their architectural bit
    /// positions: C0=8, C1=9, C2=10, C3=14). The `TOP` field (bits 11-13) is
    /// materialised from `fpu_top` when `FNSTSW` reads the full status word.
    pub fpu_sw: u16,
    /// x87 control word (used by `FLDCW`/`FNSTCW`).
    pub fpu_cw: u16,
    /// x87 tag word, in the abridged (`FXSAVE`) encoding: bit `i` is set when
    /// physical register `i` holds a value and clear when it is empty. The full
    /// two-bits-per-register architectural tag word that `FNSTENV`/`FLDENV`
    /// exchange with memory is expanded from / collapsed into this on the fly.
    pub fpu_tag: u16,

    /// SSE control and status register, read by `STMXCSR` and written by `LDMXCSR`.
    pub mxcsr: u32,

    pub cs: u16,
    pub ss: u16,
    pub ds: u16,
    pub es: u16,
    /// `FS` selector. Distinct from [`fs_base`](Self::fs_base); a real 64-bit
    /// user process reports 0 here even with a non-zero base.
    pub fs: u16,
    /// `GS` selector. Distinct from [`gs_base`](Self::gs_base).
    pub gs: u16,

    /// Where the region that just exited says to go next: the address of a
    /// [`BlockLinkEntry`](crate::core::link::BlockLinkEntry) in
    /// [`link_table`](Self::link_table), or zero for "no link - service this exit
    /// yourself".
    pub link_hit: u64,
    /// Base of this thread's call-return ring
    /// ([`CallRetEntry`](crate::core::callret::CallRetEntry)), or null.
    ///
    /// Null disables return prediction for the thread with no special case, the
    /// same way a null [`link_table`](Self::link_table) disables linking. The
    /// ring must be [`CallRetConfig::bytes`] long, 8-byte aligned, and zeroed -
    /// zero is the "empty frame" marker, so a fresh allocation is already valid.
    ///
    /// [`CallRetConfig::bytes`]: crate::core::callret::CallRetConfig::bytes
    pub callret_base: *mut crate::core::callret::CallRetEntry,
    /// Byte offset of the top frame within the ring.
    ///
    /// Any value is valid, including zero: the ring has no invalid state, which
    /// is what lets the emitted push and pop carry no bounds check. See
    /// [`crate::core::callret`].
    pub callret_off: u64,
    /// How many guest calls are currently live as host frames
    /// ([`crate::core::hostframe`]).
    ///
    /// Maintained entirely by lifted code - incremented before a nested call and
    /// restored after it, so it is right however that call ends. A host sets it
    /// to zero once and never touches it again.
    pub host_frame_depth: u32,
    /// Padding, so the struct's size does not depend on what follows
    /// `host_frame_depth`. Never read.
    pub _pad_host_frame: u32,
}

impl Default for RegisterContext {
    /// A context in the processor's post-reset state: everything zero except
    /// `mxcsr` and `fpu_cw` ([`MXCSR_RESET`], [`X87_CW_RESET`]). Zero in those two
    /// would mean "every floating-point exception unmasked", which no real thread
    /// starts in.
    ///
    /// `cs`/`ss` are the same kind of exception - a 64-bit thread with `CS = 0` is
    /// not a state that exists - but their value is a property of the *guest OS*,
    /// not the processor. The default is Darwin's; a host running a Linux or
    /// Windows guest should overwrite them ([`CS_USER64_LINUX`],
    /// [`SS_USER64_LINUX`]). `ds`/`es` and the `fs`/`gs` selectors are 0, which is
    /// what a real 64-bit user-mode process reports.
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
            syscall_handler: None,
            link_table: std::ptr::null_mut(),
            cf: 0,
            // An *odd*-parity byte, because the flag it stands for is 0 and a
            // byte of zero would mean PF=1. `RFLAGS` after a reset is `0x2`,
            // which has PF clear.
            pf_src: 1,
            af_src: 0,
            zf: 0,
            sf: 0,
            of: 0,
            df: 0,
            control: ControlHeader::ZERO,
            _align_xmm: 0,
            xmm: [[0; 2]; 16],
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
    /// The parity flag, folded out of [`pf_src`](Self::pf_src).
    pub fn pf(&self) -> bool {
        self.pf_src.count_ones().is_multiple_of(2)
    }

    /// Set the parity flag, choosing a byte of the right parity to stand for it.
    pub fn set_pf(&mut self, pf: bool) {
        self.pf_src = u8::from(!pf);
    }

    /// The auxiliary-carry flag, which is bit 4 of [`af_src`](Self::af_src).
    pub fn af(&self) -> bool {
        self.af_src & 0x10 != 0
    }

    /// Set the auxiliary-carry flag.
    pub fn set_af(&mut self, af: bool) {
        self.af_src = u8::from(af) << 4;
    }

    pub fn new() -> Self {
        Self::default()
    }
}

/// The control fields read as if they were still declared inline, which is what
/// keeps `ctx.exit_reason` / `ctx.image_base` working at every host and test
/// call site after they moved into [`ControlHeader`].
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

/// Convert an 80-bit x87 extended-precision value to the raw bits of an `f64`.
///
/// For host runtimes that need to view a guest x87 register as an `f64`. Exact
/// for zero and for normals in the `f64` exponent range (and round-trips with
/// [`f64_bits_to_f80`]), but it truncates the significand and flushes denormals
/// to zero rather than rounding.
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

/// Convert the raw bits of an `f64` to an 80-bit x87 value, returning
/// `(mantissa, sign_exp)` in the layout [`f80_to_f64_bits`] consumes.
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

    /// `xmm`'s alignment is worth twenty-two instructions per region entry and
    /// is a property of every field declared before it. This is the assertion
    /// that says so, rather than the comment on `_align_xmm` alone.
    #[test]
    fn the_vector_save_area_stays_sixteen_byte_aligned() {
        assert_eq!(std::mem::offset_of!(RegisterContext, xmm) % 16, 0);
    }

    /// The link fields are pointer-sized and pointer-aligned, which is what the
    /// probe's `trusted` (notrap, aligned) loads and stores assume. The same
    /// holds for the call-return ring's, for the same reason.
    #[test]
    fn the_link_fields_are_aligned_for_the_probe() {
        assert_eq!(std::mem::offset_of!(RegisterContext, link_table) % 8, 0);
        assert_eq!(std::mem::offset_of!(RegisterContext, link_hit) % 8, 0);
        assert_eq!(std::mem::offset_of!(RegisterContext, callret_base) % 8, 0);
        assert_eq!(std::mem::offset_of!(RegisterContext, callret_off) % 8, 0);
    }

    /// `link_hit`'s doc comment tells the next person to append. This is what
    /// catches them if they insert instead: every offset at or below `xmm` is
    /// one a host may have hardcoded, and the alignment test above would not
    /// notice a field slipped in among the flags.
    #[test]
    fn appending_a_field_does_not_move_the_ones_below_it() {
        assert_eq!(std::mem::offset_of!(RegisterContext, link_table), 144);
        assert_eq!(std::mem::offset_of!(RegisterContext, xmm), 192);
        assert_eq!(std::mem::offset_of!(RegisterContext, link_hit), 616);
    }

    /// A fresh context links nothing and claims nothing: a null table is what
    /// turns the probe into a return, and a zero `link_hit` is what makes a host
    /// that never looks at it behave as it did before linking existed.
    #[test]
    fn a_default_context_is_unlinked() {
        let ctx = RegisterContext::new();
        assert!(ctx.link_table.is_null());
        assert_eq!(ctx.link_hit, 0);
        assert!(ctx.callret_base.is_null());
        assert_eq!(ctx.host_frame_depth, 0);
    }
}
