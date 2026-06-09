//! Direct-native-call trampoline used by [`LifterConfig::direct_native_call`].
//!
//! On x86-64 hosts, lifted code can call out to an unknown / imported function
//! "natively" — instead of taking a VM exit, the lifter emits a call to the
//! trampoline below.  The trampoline:
//!
//! 1. Saves host SysV callee-saved state on the host stack.
//! 2. Switches `rsp` to the guest's `rsp` (with a small save area carved out).
//! 3. Loads the full guest GPR + XMM state from [`RegisterContext`] into the
//!    real CPU registers.
//! 4. Calls the target.  The callee runs against the real CPU state and uses
//!    the guest stack — exactly as if it had been called directly by guest
//!    code.
//! 5. Writes the post-call register state back into [`RegisterContext`], so
//!    the lifted code that resumes after the trampoline observes the call's
//!    side effects (return value in RAX, etc.).
//! 6. Restores host state and returns.
//!
//! The trampoline is only emitted when the host architecture is x86-64; the
//! [`LifterConfig::direct_native_call`] setter rejects the request on other
//! architectures.
//!
//! ABI assumptions:
//!
//! - Callee follows SysV AMD64.  Caller-saved registers (RAX, RCX, RDX, RSI,
//!   RDI, R8–R11, XMM0–XMM15) are considered clobbered by the call.  We store
//!   their post-call values back into the context anyway, because that's what
//!   real hardware would observe.  `R11` is sacrificed as the writeback ctx
//!   pointer; its post-call value in the context is undefined (consistent
//!   with ABI semantics).
//! - The trampoline does *not* preserve flags.  SysV-callable functions are
//!   allowed to clobber RFLAGS arithmetic bits, so the guest must already
//!   treat them as undefined after the call.

use crate::context::RegisterContext;

/// The Cranelift-side function pointer value used to import the trampoline.
///
/// Returns the raw address as an `i64` ready to be embedded as a Cranelift
/// `iconst` operand.
#[cfg(target_arch = "x86_64")]
pub fn direct_native_call_trampoline_addr() -> i64 {
    (microlift_direct_native_call_trampoline as *const ()) as i64
}

/// Stub used on non-x86-64 hosts so the rest of the crate can reference the
/// symbol without `#[cfg]` everywhere.  It is never actually called because
/// `LifterConfig::set_direct_native_call(true)` is rejected on those hosts.
#[cfg(not(target_arch = "x86_64"))]
pub fn direct_native_call_trampoline_addr() -> i64 {
    0
}

/// Whether the current host supports the direct-native-call trampoline.
pub const HOST_SUPPORTS_DIRECT_NATIVE_CALL: bool = cfg!(target_arch = "x86_64");

#[cfg(target_arch = "x86_64")]
#[unsafe(naked)]
pub unsafe extern "sysv64" fn microlift_direct_native_call_trampoline(
    _ctx: *mut RegisterContext,
    _target: u64,
) {
    core::arch::naked_asm!(
        // SysV: rdi = ctx, rsi = target.
        //
        // Step 1: save host SysV callee-saved registers on the host stack.
        "push rbx",
        "push rbp",
        "push r12",
        "push r13",
        "push r14",
        "push r15",

        // Step 2: load guest RSP and carve out a 16-byte-aligned save area on
        // it.  We allocate 64 bytes and round down to a 16-byte boundary so
        // that the `call` below leaves the callee with the SysV-required
        // entry alignment (`rsp % 16 == 8` at function entry).
        //
        //   [r10 +  0] : target address (for the indirect `call`)
        //   [r10 +  8] : ctx pointer  (recovered after the call)
        //   [r10 + 16] : host RSP     (recovered after the call)
        //   [r10 + 24] : original guest RSP (written back after the call)
        "mov r10, [rdi + {rsp_off}]",
        "mov rax, r10",
        "sub r10, 64",
        "and r10, -16",
        "mov [r10 +  0], rsi",
        "mov [r10 +  8], rdi",
        "mov [r10 + 16], rsp",
        "mov [r10 + 24], rax",

        // Step 3: load guest XMM state.  `rdi` still holds ctx.
        "movups xmm0,  [rdi + {xmm_off} +   0]",
        "movups xmm1,  [rdi + {xmm_off} +  16]",
        "movups xmm2,  [rdi + {xmm_off} +  32]",
        "movups xmm3,  [rdi + {xmm_off} +  48]",
        "movups xmm4,  [rdi + {xmm_off} +  64]",
        "movups xmm5,  [rdi + {xmm_off} +  80]",
        "movups xmm6,  [rdi + {xmm_off} +  96]",
        "movups xmm7,  [rdi + {xmm_off} + 112]",
        "movups xmm8,  [rdi + {xmm_off} + 128]",
        "movups xmm9,  [rdi + {xmm_off} + 144]",
        "movups xmm10, [rdi + {xmm_off} + 160]",
        "movups xmm11, [rdi + {xmm_off} + 176]",
        "movups xmm12, [rdi + {xmm_off} + 192]",
        "movups xmm13, [rdi + {xmm_off} + 208]",
        "movups xmm14, [rdi + {xmm_off} + 224]",
        "movups xmm15, [rdi + {xmm_off} + 240]",

        // Step 4: switch to the guest stack.
        "mov rsp, r10",

        // Step 5: load guest GPRs.  `rdi` is loaded last because it is also
        // the source pointer.
        "mov rax, [rdi + {rax_off}]",
        "mov rcx, [rdi + {rcx_off}]",
        "mov rdx, [rdi + {rdx_off}]",
        "mov rbx, [rdi + {rbx_off}]",
        "mov rbp, [rdi + {rbp_off}]",
        "mov rsi, [rdi + {rsi_off}]",
        "mov r8,  [rdi + {r8_off}]",
        "mov r9,  [rdi + {r9_off}]",
        "mov r10, [rdi + {r10_off}]",
        "mov r11, [rdi + {r11_off}]",
        "mov r12, [rdi + {r12_off}]",
        "mov r13, [rdi + {r13_off}]",
        "mov r14, [rdi + {r14_off}]",
        "mov r15, [rdi + {r15_off}]",
        "mov rdi, [rdi + {rdi_off}]",

        // Step 6: call the target.  `call qword ptr [rsp]` reads the target
        // from the save area we built before the rsp switch.
        "call qword ptr [rsp]",

        // Step 7: recover ctx pointer.  We sacrifice R11 — its value after a
        // SysV call is undefined from the caller's point of view anyway.
        "mov r11, [rsp + 8]",

        // Step 8: write back guest GPRs (skip R11, which is undefined).
        "mov [r11 + {rax_off}], rax",
        "mov [r11 + {rcx_off}], rcx",
        "mov [r11 + {rdx_off}], rdx",
        "mov [r11 + {rbx_off}], rbx",
        "mov [r11 + {rbp_off}], rbp",
        "mov [r11 + {rsi_off}], rsi",
        "mov [r11 + {rdi_off}], rdi",
        "mov [r11 + {r8_off}],  r8",
        "mov [r11 + {r9_off}],  r9",
        "mov [r11 + {r10_off}], r10",
        "mov [r11 + {r12_off}], r12",
        "mov [r11 + {r13_off}], r13",
        "mov [r11 + {r14_off}], r14",
        "mov [r11 + {r15_off}], r15",

        // Step 9: write back the original guest rsp captured before the
        // switch (callee balanced its own frame, so the value is unchanged
        // from what we saved).
        "mov rax, [rsp + 24]",
        "mov [r11 + {rsp_off}], rax",

        // Step 10: write back guest XMM state.
        "movups [r11 + {xmm_off} +   0], xmm0",
        "movups [r11 + {xmm_off} +  16], xmm1",
        "movups [r11 + {xmm_off} +  32], xmm2",
        "movups [r11 + {xmm_off} +  48], xmm3",
        "movups [r11 + {xmm_off} +  64], xmm4",
        "movups [r11 + {xmm_off} +  80], xmm5",
        "movups [r11 + {xmm_off} +  96], xmm6",
        "movups [r11 + {xmm_off} + 112], xmm7",
        "movups [r11 + {xmm_off} + 128], xmm8",
        "movups [r11 + {xmm_off} + 144], xmm9",
        "movups [r11 + {xmm_off} + 160], xmm10",
        "movups [r11 + {xmm_off} + 176], xmm11",
        "movups [r11 + {xmm_off} + 192], xmm12",
        "movups [r11 + {xmm_off} + 208], xmm13",
        "movups [r11 + {xmm_off} + 224], xmm14",
        "movups [r11 + {xmm_off} + 240], xmm15",

        // Step 11: restore host rsp.
        "mov rsp, [rsp + 16]",

        // Step 12: restore host callee-saved and return.
        "pop r15",
        "pop r14",
        "pop r13",
        "pop r12",
        "pop rbp",
        "pop rbx",
        "ret",

        rax_off = const std::mem::offset_of!(RegisterContext, rax),
        rcx_off = const std::mem::offset_of!(RegisterContext, rcx),
        rdx_off = const std::mem::offset_of!(RegisterContext, rdx),
        rbx_off = const std::mem::offset_of!(RegisterContext, rbx),
        rsp_off = const std::mem::offset_of!(RegisterContext, rsp),
        rbp_off = const std::mem::offset_of!(RegisterContext, rbp),
        rsi_off = const std::mem::offset_of!(RegisterContext, rsi),
        rdi_off = const std::mem::offset_of!(RegisterContext, rdi),
        r8_off  = const std::mem::offset_of!(RegisterContext, r8),
        r9_off  = const std::mem::offset_of!(RegisterContext, r9),
        r10_off = const std::mem::offset_of!(RegisterContext, r10),
        r11_off = const std::mem::offset_of!(RegisterContext, r11),
        r12_off = const std::mem::offset_of!(RegisterContext, r12),
        r13_off = const std::mem::offset_of!(RegisterContext, r13),
        r14_off = const std::mem::offset_of!(RegisterContext, r14),
        r15_off = const std::mem::offset_of!(RegisterContext, r15),
        xmm_off = const std::mem::offset_of!(RegisterContext, xmm),
    )
}
