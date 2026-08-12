//! x86-64 guest register context (the FFI struct passed to lifted code).
//!
//! Kept as a flat `#[repr(C)]` struct so existing host/test code that reads
//! fields by name (`ctx.rax`, `ctx.rip`, `ctx.exit_reason`, ) is unaffected.
//! The architecture-neutral control fields (`exit_reason`, `exit_code`,
//! `imported_function_va`, `image_base`) mirror [`crate::core::exit::ControlHeader`]
//! conceptually but stay inline to preserve the established layout.

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
    pub image_base: u64,
    /// A syscall handler intended for the RuntimeAssisted execution mode.
    pub syscall_handler: Option<unsafe extern "C" fn(ctx: *mut RegisterContext)>,
    /// FIXME:
    /// # **DO NOT USE!!!**
    /// Resume function pointer for AOT block dispatch (**not implemented yet** - this is a placeholder)
    pub resume_function: Option<unsafe extern "C" fn(ctx: *mut RegisterContext)>,
    // Flags as individual bytes for simpler IR load/store
    pub cf: u8,
    pub pf: u8,
    pub af: u8,
    pub zf: u8,
    pub sf: u8,
    pub of: u8,
    /// Direction flag (DF, RFLAGS bit 10). Controls whether string operations
    /// advance their pointers upward (`DF=0`) or downward (`DF=1`).
    pub df: u8,
    pub exit_reason: u32,
    pub exit_code: u32,
    /// On `ImportedFunctionCall` exits, the virtual address of the imported
    /// function that was called. `rip` points to the instruction *after* the
    /// `CALL` - the resume point once the host has handled the call.
    pub imported_function_va: u64,
    /// XMM0–XMM15, each stored as [lo_64, hi_64].
    pub xmm: [[u64; 2]; 16],
    /// FS segment base address. In x86-64 user mode this is set via
    /// `arch_prctl(ARCH_SET_FS)` or `wrfsbase` and used for thread-local
    /// storage. Reads/writes through `fs:` prefixed addresses add this base.
    pub fs_base: u64,
    /// GS segment base address. On Windows this points to the TEB; on Linux
    /// it is often used for kernel->user communication via `swapgs`.
    /// Reads/writes through `gs:` prefixed addresses add this base.
    pub gs_base: u64,

    /// x87 data registers in true 80-bit extended precision, one physical
    /// register per 16-byte slot laid out as `[mantissa, sign_exp]`:
    ///  - `[0]` = the 64-bit significand (explicit integer bit at position 63);
    ///  - `[1]` = the sign bit (15) and 15-bit biased exponent in the low 16
    ///    bits (upper 48 bits unused/zero).
    ///
    /// This matches the FXSAVE `_st[i]` layout (10 bytes used, 16-byte slot), so
    /// a host runtime can copy each register to/from the native x87 save area
    /// without any format conversion. The architectural `ST(i)` register maps to
    /// the physical slot `(fpu_top + i) & 7`; pushes decrement `fpu_top` and
    /// pops increment it.
    pub fpr: [[u64; 2]; 8],
    /// Top-of-stack pointer (0-7). `ST(0)` lives in `fpr[fpu_top]`.
    pub fpu_top: u16,
    /// x87 status-word condition-code bits (C0-C3 in their architectural bit
    /// positions: C0=8, C1=9, C2=10, C3=14). The `TOP` field (bits 11-13) is
    /// materialised from `fpu_top` when `FNSTSW` reads the full status word.
    pub fpu_sw: u16,
    /// x87 control word (used by `FLDCW`/`FNSTCW`).
    pub fpu_cw: u16,
    /// x87 tag word, in the *abridged* (`FXSAVE`) encoding: bit `i` is set when
    /// physical register `i` holds a value and clear when it is empty. The full
    /// two-bits-per-register architectural tag word that `FNSTENV`/`FLDENV`
    /// exchange with memory is expanded from / collapsed into this on the fly.
    pub fpu_tag: u16,

    /// SSE control and status register, read by `STMXCSR` and written by
    /// `LDMXCSR`.
    ///
    /// The field is architecturally faithful *as storage* - whatever `LDMXCSR`
    /// writes, `STMXCSR` reads back - but the lifter does not currently let it
    /// influence SSE arithmetic: rounding is always round-to-nearest-even
    /// (`RC = 00`), FZ/DAZ are not applied to denormals, and the six exception
    /// flags (bits 5:0) are never set by a lifted SSE instruction. Code that
    /// saves, modifies and restores `MXCSR` therefore behaves correctly; code
    /// that changes the rounding mode and expects arithmetic to follow it, or
    /// that polls the sticky exception flags, does not.
    pub mxcsr: u32,

    /// The six segment *selectors*, as read by `MOV r/m16, Sreg` and
    /// `PUSH FS`/`PUSH GS`.
    ///
    /// These are **selectors, not bases**: the address a `fs:`/`gs:`-prefixed
    /// access forms comes from [`fs_base`](Self::fs_base) /
    /// [`gs_base`](Self::gs_base), which are entirely independent fields here.
    /// That mirrors 64-bit hardware, where `WRFSBASE`/`arch_prctl` move a base
    /// without touching the selector - but it also means the reverse does *not*
    /// hold: writing `fs` or `gs` (`MOV Sreg, r/m16`, `POP FS`) stores the
    /// selector and leaves the base alone, where hardware would reload the base
    /// from the descriptor table. A guest that changes its TLS by loading a
    /// selector is therefore not modelled; one that reads its selectors for a
    /// diagnostic record - which is what these exist for - is.
    ///
    /// Like `MXCSR`, they are faithful storage rather than a control input:
    /// nothing in the lifter interprets them, and a segment override other than
    /// `FS`/`GS` is ignored when forming an address (all four have base 0 in
    /// long mode).
    ///
    /// Written by the lifter only through the segment-register semantics; the
    /// host is expected to seed them, since their correct values are a property
    /// of the guest OS rather than of the code being lifted
    /// ([`CS_USER64_DARWIN`] / [`CS_USER64_LINUX`] and friends).
    pub cs: u16,
    pub ss: u16,
    pub ds: u16,
    pub es: u16,
    /// `FS` selector. Distinct from [`fs_base`](Self::fs_base); a real 64-bit
    /// user process reports 0 here even with a non-zero base.
    pub fs: u16,
    /// `GS` selector. Distinct from [`gs_base`](Self::gs_base).
    pub gs: u16,
}

impl Default for RegisterContext {
    /// A context in the processor's post-reset state: everything zero except
    /// `mxcsr` and `fpu_cw`, whose reset values ([`MXCSR_RESET`],
    /// [`X87_CW_RESET`]) are not zero. Zero in those two fields would mean
    /// "every floating-point exception unmasked", which no real thread starts
    /// out in and which would make `STMXCSR`/`FNSTCW` report an impossible
    /// state.
    ///
    /// `cs`/`ss` are the same kind of exception, for the same reason - a
    /// 64-bit thread with `CS = 0` is not a state that exists - but unlike the
    /// FP fields their value is a property of the *guest OS*, not of the
    /// processor. The default is Darwin's ([`CS_USER64_DARWIN`],
    /// [`SS_USER64_DARWIN`]); a host running a Linux or Windows guest should
    /// overwrite them ([`CS_USER64_LINUX`], [`SS_USER64_LINUX`]). `ds`/`es` and
    /// the `fs`/`gs` *selectors* are 0, which is what a real 64-bit user-mode
    /// process reports.
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
            image_base: 0,
            syscall_handler: None,
            resume_function: None,
            cf: 0,
            pf: 0,
            af: 0,
            zf: 0,
            sf: 0,
            of: 0,
            df: 0,
            exit_reason: 0,
            exit_code: 0,
            imported_function_va: 0,
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
        }
    }
}

impl RegisterContext {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Convert an 80-bit x87 extended-precision value to the raw bits of an `f64`.
///
/// A convenience round-tripper for host runtimes that need to view a guest x87
/// register as an `f64`. It is exact for zero and for normal values within
/// the `f64` exponent range (and round-trips with [`f64_bits_to_f80`]), but it
/// truncates the significand and flushes denormals to zero rather than
/// rounding.
///
/// It is therefore *not* identical to the narrowing the lifter emits for
/// `FST m32/m64`, which uses the precise softfloat path
/// ([`crate::arch::x86_64::Lifter::f80_to_f64`]) with round-to-nearest-even and
/// full subnormal handling. Use this only for interop/inspection, not to
/// reproduce lifted `FST` results bit-for-bit.
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

/// Convert the raw bits of an `f64` to an 80-bit x87 extended-precision value,
/// returning `(mantissa, sign_exp)` in the same layout [`f80_to_f64_bits`]
/// consumes.
///
/// `f64` is a strict subset of the 80-bit format, so this widening is **exact**
/// for every finite `f64` except denormals, which are flushed to zero. It is
/// used at lift time by `FLDZ`/`FLD1`/`FLDPI` to encode their (normal, in-range)
/// constants, where it agrees with the lifter's general `f64_to_f80` path.
#[inline]
pub fn f64_bits_to_f80(bits: u64) -> (u64, u16) {
    let sign = (bits >> 63) & 1;
    let exp64 = (bits >> 52) & 0x7FF;
    let frac = bits & 0x000F_FFFF_FFFF_FFFF;
    if exp64 == 0 && frac == 0 {
        // ±0 (denormals are flushed to zero, matching the lifter).
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
