//! Direct-native-call trampoline used by [`crate::lifter::LifterConfig::direct_native_call`].
//!
//! On x86-64 hosts, lifted code can call out to an unknown / imported function
//! "natively" — instead of taking a VM exit, the lifter emits a call to the
//! trampoline below.  The trampoline:
//!
//! 1. Saves host SysV callee-saved state on the host stack.
//! 2. Switches `rsp` to the guest's `rsp` (with a small save area carved out).
//! 3. Loads the full guest GPR + XMM state from [`crate::context::RegisterContext`]
//!    into the real CPU registers.
//! 4. Calls the target.  The callee runs against the real CPU state and uses
//!    the guest stack — exactly as if it had been called directly by guest
//!    code.
//! 5. Writes the post-call register state back into [`crate::context::RegisterContext`].
//! 6. Restores host state and returns.
//!
//! The trampoline is emitted via [`core::arch::global_asm!`] with a `start`
//! and `end` label so that its raw machine bytes can be embedded into both
//! JIT memory and AOT object files via
//! [`cranelift_module::Module::define_function_bytes`].  That is what makes
//! `direct_native_call` usable for AOT — the trampoline travels with the
//! produced object instead of being resolved at runtime.
//!
//! ABI assumptions:
//!
//! - Callee follows SysV AMD64.  Caller-saved registers (RAX, RCX, RDX, RSI,
//!   RDI, R8–R11, XMM0–XMM15) are considered clobbered by the call.  We still
//!   store their post-call values back into the context because that mirrors
//!   what real hardware would leave there.  `R11` is sacrificed as the
//!   writeback ctx pointer; its post-call value in the context is undefined
//!   (consistent with ABI semantics).
//! - The trampoline does *not* preserve flags.  SysV-callable functions are
//!   allowed to clobber RFLAGS arithmetic bits, so the guest must already
//!   treat them as undefined after the call.

use crate::context::RegisterContext;
use cranelift_codegen::ir::{AbiParam, Function, Signature, types};
use cranelift_codegen::isa::CallConv;
use cranelift_module::{FuncId, Linkage, Module, ModuleError, ModuleReloc};

/// Symbol the trampoline is declared under in any [`Module`].  AOT outputs
/// will contain a `Local` (private) symbol with this name pointing at the
/// embedded trampoline bytes.
pub const TRAMPOLINE_SYMBOL: &str = "microlift_direct_native_call_trampoline";

/// Whether the current host supports the direct-native-call trampoline.
pub const HOST_SUPPORTS_DIRECT_NATIVE_CALL: bool = cfg!(target_arch = "x86_64");

/// SysV signature `fn(*mut RegisterContext, target_va: u64)` returning void.
pub fn trampoline_signature() -> Signature {
    let mut sig = Signature::new(CallConv::SystemV);
    sig.params.push(AbiParam::new(types::I64)); // ctx pointer
    sig.params.push(AbiParam::new(types::I64)); // target VA
    sig
}

#[cfg(target_arch = "x86_64")]
unsafe extern "sysv64" {
    fn microlift_direct_native_call_trampoline(ctx: *mut RegisterContext, target: u64);
    static microlift_direct_native_call_trampoline_end: u8;
}

#[cfg(target_arch = "x86_64")]
core::arch::global_asm!(
    ".p2align 4",
    ".globl {sym_start}",
    ".hidden {sym_start}",
    "{sym_start}:",

    // SysV: rdi = ctx, rsi = target.
    //
    // Step 1: save host SysV callee-saved registers on the host stack.
    "push rbx",
    "push rbp",
    "push r12",
    "push r13",
    "push r14",
    "push r15",

    // Step 2: load guest RSP and carve out a 16-byte-aligned save area on it.
    // We allocate 64 bytes and round down to a 16-byte boundary so that the
    // `call` below leaves the callee with the SysV-required entry alignment
    // (`rsp % 16 == 8` at function entry).
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

    // Step 5: load guest GPRs.  `rdi` is loaded last because it is also the
    // source pointer.
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

    // Step 6: call the target.  `call qword ptr [rsp]` reads the target from
    // the save area we built before the rsp switch.
    "call qword ptr [rsp]",

    // Step 7: recover ctx pointer.  R11's post-call value is ABI-undefined so
    // we sacrifice it as the writeback pointer.
    "mov r11, [rsp + 8]",

    // Step 8: write back guest GPRs (R11 is skipped — undefined post-call).
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

    // Step 9: write back the original guest rsp (callee balanced its frame).
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

    // End label, used to compute the trampoline's byte length.
    ".globl {sym_end}",
    ".hidden {sym_end}",
    "{sym_end}:",

    sym_start = sym microlift_direct_native_call_trampoline,
    sym_end = sym microlift_direct_native_call_trampoline_end,
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
);

/// Raw machine bytes of the trampoline.  Position-independent x86-64
/// instructions — safe to embed into any text section as-is.
#[cfg(target_arch = "x86_64")]
pub fn trampoline_bytes() -> &'static [u8] {
    let start = microlift_direct_native_call_trampoline as *const () as usize;
    let end = unsafe { &microlift_direct_native_call_trampoline_end as *const u8 as usize };
    debug_assert!(end > start, "trampoline end must follow start");
    unsafe { core::slice::from_raw_parts(start as *const u8, end - start) }
}

/// Stub for non-x86-64 hosts.  The lifter never asks for these bytes because
/// `LifterConfig::set_direct_native_call(true)` is rejected up front.
#[cfg(not(target_arch = "x86_64"))]
pub fn trampoline_bytes() -> &'static [u8] {
    &[]
}

/// Idempotently declare *and* define the trampoline as a `Local` function in
/// the given module, returning its `FuncId`.
///
/// Safe to call multiple times on the same module — repeated `define`
/// attempts are detected via [`ModuleError::DuplicateDefinition`] and
/// silently ignored.
pub fn ensure_trampoline_defined<M: Module>(module: &mut M) -> crate::Result<FuncId> {
    let id = module
        .declare_function(TRAMPOLINE_SYMBOL, Linkage::Local, &trampoline_signature())
        .map_err(|e| crate::MicroliftError::CraneliftError(e.to_string()))?;

    let bytes = trampoline_bytes();
    if bytes.is_empty() {
        return Err(crate::MicroliftError::UnsupportedInstruction {
            mnemonic: "direct_native_call trampoline is x86-64 only".to_string(),
        });
    }

    let relocs: &[ModuleReloc] = &[];
    match module.define_function_bytes(id, 16, bytes, relocs) {
        Ok(()) => {}
        Err(ModuleError::DuplicateDefinition(_)) => {
            // Already defined in a previous call - that's fine.
        }
        Err(e) => return Err(crate::MicroliftError::CraneliftError(e.to_string())),
    }
    Ok(id)
}

/// Declare the trampoline in `func` and return its `FuncRef`, ensuring the
/// trampoline is defined in `module` first.
pub fn declare_trampoline_in_func<M: Module>(
    module: &mut M,
    func: &mut Function,
) -> crate::Result<cranelift_codegen::ir::FuncRef> {
    let id = ensure_trampoline_defined(module)?;
    Ok(module.declare_func_in_func(id, func))
}
