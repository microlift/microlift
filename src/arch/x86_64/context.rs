//! x86-64 guest register context (the FFI struct passed to lifted code).
//!
//! Kept as a flat `#[repr(C)]` struct so existing host/test code that reads
//! fields by name (`ctx.rax`, `ctx.rip`, `ctx.exit_reason`, …) is unaffected.
//! The architecture-neutral control fields (`exit_reason`, `exit_code`,
//! `imported_function_va`, `image_base`) mirror [`crate::core::exit::ControlHeader`]
//! conceptually but stay inline to preserve the established layout.

#[repr(C)]
#[derive(Default, Debug, Clone, Copy)]
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

    /// x87 data registers, each stored as the raw bits of
    /// an `f64`. The architectural `ST(i)` register maps to the physical slot
    /// `(fpu_top + i) & 7`; pushes decrement `fpu_top` and pops increment it.
    ///
    /// The lifter models the x87 stack in `f64` precision rather than the true
    /// 80-bit extended format. `FLD`/`FSTP` of an 80-bit `m80` value convert to
    /// and from `f64`, so the extra mantissa/exponent range of `long double` is
    /// not preserved for normal values (and denormals/NaN payloads are
    /// approximated).
    pub fpr: [u64; 8],
    /// Top-of-stack pointer (0-7). `ST(0)` lives in `fpr[fpu_top]`.
    pub fpu_top: u16,
    /// x87 status-word condition-code bits (C0-C3 in their architectural bit
    /// positions: C0=8, C1=9, C2=10, C3=14). The `TOP` field (bits 11-13) is
    /// materialised from `fpu_top` when `FNSTSW` reads the full status word.
    pub fpu_sw: u16,
    /// x87 control word (used by `FLDCW`/`FNSTCW`).
    pub fpu_cw: u16,
    /// x87 tag word (2 bits per physical register). Maintained loosely; kept for
    /// `FNSTSW`/debugging completeness.
    pub fpu_tag: u16,
}

impl RegisterContext {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Convert an 80-bit x87 extended-precision value to the raw bits of an `f64`.
/// This is the pure-Rust twin of the IR emitted by the lifter's `FLD m80`
/// ([`crate::arch::x86_64::Lifter::fpu_load_f80`]); the two are kept bit-for-bit
/// identical so a host runtime that materialises guest x87 state links against
/// exactly the same conversion the lifted code uses. Exact for zero and normal
/// values within the `f64` exponent range; denormals flush to zero and
/// out-of-range exponents wrap (the extra range/precision of 80-bit
/// `long double` is not representable in `f64`).
#[inline]
pub fn f80_to_f64_bits(mantissa: u64, sign_exp: u16) -> u64 {
    let sign = ((sign_exp >> 15) & 1) as u64;
    if mantissa == 0 {
        // ±0 (also the encoding produced for flushed/true zeros).
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
/// Pure-Rust twin of the lifter's `FSTP m80`
/// ([`crate::arch::x86_64::Lifter::fpu_store_f80`]); kept bit-for-bit identical.
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
