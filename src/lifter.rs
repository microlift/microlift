use crate::{
    MicroliftError,
    context::{ExitReason, MemorySegment, RegisterContext, TransferKind},
    function_traits::{DynamicSymbol, FunctionArgumentType},
    os::get_host_calling_conv,
};
use cranelift::prelude::*;
use cranelift_codegen::ir::SigRef;
use cranelift_codegen::isa::CallConv;
use cranelift_frontend::{FunctionBuilder, Switch, Variable};
use iced_x86::{Decoder, DecoderOptions, Instruction, Mnemonic, OpKind, Register};
use std::collections::{HashMap, HashSet};

/// Mapping of x86 registers to Cranelift variables.
pub struct RegisterState {
    pub gprs: HashMap<Register, Variable>,
    pub flags: HashMap<X86Flag, Variable>,
    pub context_ptr: Variable,
    pub image_base: Variable,
    /// XMM0–XMM15 low 64 bits (index 0–15).
    /// Populated only when [`LifterConfig::use_vector_xmm`] is `false`.
    pub xmms_lo: HashMap<u8, Variable>,
    /// XMM0–XMM15 high 64 bits (index 0–15).
    /// Populated only when [`LifterConfig::use_vector_xmm`] is `false`.
    pub xmms_hi: HashMap<u8, Variable>,
    /// XMM0–XMM15 as a single native `I64X2` SSA variable per register.
    /// Populated only when [`LifterConfig::use_vector_xmm`] is `true`.
    pub xmms_vec: HashMap<u8, Variable>,
    /// Mirror of [`LifterConfig::use_vector_xmm`] - selects between the
    /// scalar-pair representation and the native vector representation.
    pub use_vector_xmm: bool,
    /// FS segment base address SSA variable.
    pub fs_base: Variable,
    /// GS segment base address SSA variable.
    pub gs_base: Variable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum X86Flag {
    Carry,
    Parity,
    Auxiliary,
    Zero,
    Sign,
    Overflow,
}

impl RegisterState {
    pub fn new(builder: &mut FunctionBuilder, use_vector_xmm: bool) -> Self {
        let mut gprs = HashMap::new();

        // Define all 64-bit GPRs
        let qword_gprs = [
            Register::RAX,
            Register::RCX,
            Register::RDX,
            Register::RBX,
            Register::RSP,
            Register::RBP,
            Register::RSI,
            Register::RDI,
            Register::R8,
            Register::R9,
            Register::R10,
            Register::R11,
            Register::R12,
            Register::R13,
            Register::R14,
            Register::R15,
        ];

        for &reg in &qword_gprs {
            let var = builder.declare_var(types::I64);
            gprs.insert(reg, var);
        }

        let mut flags = HashMap::new();
        let flag_types = [
            X86Flag::Carry,
            X86Flag::Parity,
            X86Flag::Auxiliary,
            X86Flag::Zero,
            X86Flag::Sign,
            X86Flag::Overflow,
        ];

        for &flag in &flag_types {
            let var = builder.declare_var(types::I8);
            flags.insert(flag, var);
        }

        let context_ptr = builder.declare_var(types::I64); // Assume 64-bit pointer
        let image_base = builder.declare_var(types::I64);

        let mut xmms_lo = HashMap::new();
        let mut xmms_hi = HashMap::new();
        let mut xmms_vec = HashMap::new();
        if use_vector_xmm {
            for i in 0u8..16 {
                xmms_vec.insert(i, builder.declare_var(types::I64X2));
            }
        } else {
            for i in 0u8..16 {
                xmms_lo.insert(i, builder.declare_var(types::I64));
                xmms_hi.insert(i, builder.declare_var(types::I64));
            }
        }

        let fs_base = builder.declare_var(types::I64);
        let gs_base = builder.declare_var(types::I64);

        Self {
            gprs,
            flags,
            context_ptr,
            image_base,
            xmms_lo,
            xmms_hi,
            xmms_vec,
            use_vector_xmm,
            fs_base,
            gs_base,
        }
    }

    pub fn get_gpr_offset(reg: Register) -> crate::Result<i32> {
        match reg {
            Register::RAX => Ok(std::mem::offset_of!(RegisterContext, rax) as i32),
            Register::RCX => Ok(std::mem::offset_of!(RegisterContext, rcx) as i32),
            Register::RDX => Ok(std::mem::offset_of!(RegisterContext, rdx) as i32),
            Register::RBX => Ok(std::mem::offset_of!(RegisterContext, rbx) as i32),
            Register::RSP => Ok(std::mem::offset_of!(RegisterContext, rsp) as i32),
            Register::RBP => Ok(std::mem::offset_of!(RegisterContext, rbp) as i32),
            Register::RSI => Ok(std::mem::offset_of!(RegisterContext, rsi) as i32),
            Register::RDI => Ok(std::mem::offset_of!(RegisterContext, rdi) as i32),
            Register::R8 => Ok(std::mem::offset_of!(RegisterContext, r8) as i32),
            Register::R9 => Ok(std::mem::offset_of!(RegisterContext, r9) as i32),
            Register::R10 => Ok(std::mem::offset_of!(RegisterContext, r10) as i32),
            Register::R11 => Ok(std::mem::offset_of!(RegisterContext, r11) as i32),
            Register::R12 => Ok(std::mem::offset_of!(RegisterContext, r12) as i32),
            Register::R13 => Ok(std::mem::offset_of!(RegisterContext, r13) as i32),
            Register::R14 => Ok(std::mem::offset_of!(RegisterContext, r14) as i32),
            Register::R15 => Ok(std::mem::offset_of!(RegisterContext, r15) as i32),
            _ => Err(MicroliftError::UnsupportedRegister(reg)),
        }
    }

    pub fn get_flag_offset(flag: X86Flag) -> crate::Result<i32> {
        match flag {
            X86Flag::Carry => Ok(std::mem::offset_of!(RegisterContext, cf) as i32),
            X86Flag::Parity => Ok(std::mem::offset_of!(RegisterContext, pf) as i32),
            X86Flag::Auxiliary => Ok(std::mem::offset_of!(RegisterContext, af) as i32),
            X86Flag::Zero => Ok(std::mem::offset_of!(RegisterContext, zf) as i32),
            X86Flag::Sign => Ok(std::mem::offset_of!(RegisterContext, sf) as i32),
            X86Flag::Overflow => Ok(std::mem::offset_of!(RegisterContext, of) as i32),
        }
    }

    /// Returns the byte offset of the low (hi=false) or high (hi=true) 64-bit
    /// half of XMM register `idx` (0–15) within `RegisterContext`.
    pub fn get_xmm_offset(idx: u8, hi: bool) -> i32 {
        let base = std::mem::offset_of!(RegisterContext, xmm) as i32;
        base + (idx as i32) * 16 + if hi { 8 } else { 0 }
    }

    /// Byte offset of the whole 128-bit XMM register `idx` (0–15) within
    /// `RegisterContext`. Used for native `I64X2` loads/stores when
    /// [`LifterConfig::use_vector_xmm`] is enabled.
    pub fn get_xmm_vec_offset(idx: u8) -> i32 {
        let base = std::mem::offset_of!(RegisterContext, xmm) as i32;
        base + (idx as i32) * 16
    }

    pub fn get_fs_base_offset() -> i32 {
        std::mem::offset_of!(RegisterContext, fs_base) as i32
    }

    pub fn get_gs_base_offset() -> i32 {
        std::mem::offset_of!(RegisterContext, gs_base) as i32
    }
}

/// Convert a `FunctionArgumentType` to the corresponding Cranelift IR type.
/// Returns `None` for `None` (void) and `VariadicList` (stop marker).
pub(crate) fn fat_to_cl_type(ty: FunctionArgumentType) -> Option<types::Type> {
    match ty {
        FunctionArgumentType::None | FunctionArgumentType::VariadicList => None,
        FunctionArgumentType::SignedByte | FunctionArgumentType::UnsignedByte => Some(types::I8),
        FunctionArgumentType::SignedHalfword | FunctionArgumentType::UnsignedHalfword => {
            Some(types::I16)
        }
        FunctionArgumentType::SignedWord | FunctionArgumentType::UnsignedWord => Some(types::I32),
        FunctionArgumentType::SignedDoubleword
        | FunctionArgumentType::UnsignedDoubleword
        | FunctionArgumentType::Pointer => Some(types::I64),
        FunctionArgumentType::SinglePrecisionFp => Some(types::F32),
        FunctionArgumentType::DoublePrecisionFp => Some(types::F64),
    }
}

/// Build a Cranelift `Signature` from a `DynamicSymbol`'s `arg_types` slice.
///
/// `arg_types[0]` is the return type; `arg_types[1..]` are parameter types.
/// Stops at the first `VariadicList` entry.
pub(crate) fn build_import_sig(
    arg_types: &[FunctionArgumentType],
    call_conv: CallConv,
) -> Signature {
    let mut sig = Signature::new(call_conv);
    if let Some(ret_ty) = arg_types.first().copied().and_then(fat_to_cl_type) {
        sig.returns.push(AbiParam::new(ret_ty));
    }
    for &ty in arg_types.iter().skip(1) {
        if ty == FunctionArgumentType::VariadicList {
            break;
        }
        if let Some(param_ty) = fat_to_cl_type(ty) {
            sig.params.push(AbiParam::new(param_ty));
        }
    }
    sig
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExecutionMode {
    #[default]
    Jit,
    RuntimeAssisted,
}

#[derive(Debug, Clone)]
pub struct LifterConfig {
    pub execution_mode: ExecutionMode,
    pub syscall_handler_call_conv: CallConv,
    pub segments: Vec<MemorySegment>,
    pub interpreter_fallback: bool,
    pub emit_resume_blocks: bool,
    /// Registered import addresses.
    ///
    /// - Key: virtual address of the imported function (e.g. a PLT stub or
    ///   thunk address) as it appears in the code being lifted.
    /// - Value: `Some(sym)` -> attempt a fast inline call using SysV AMD64
    ///   argument passing (skipped for variadic functions, which fall back to
    ///   a slow VM exit). `None` -> always emit an `ImportedFunctionCall` exit.
    ///
    /// Use [`register_fast_import`] / [`register_slow_import`] to populate.
    pub imports: HashMap<u64, Option<DynamicSymbol>>,
    /// When `true`, XMM registers are stored as native `I64X2` SSA
    /// variables and whole-register SSE handlers (movaps / xorps /
    /// andps / orps / andnps and their `pd`/integer aliases) take a
    /// vector fast path, lowering directly to host XMM registers on
    /// x86-64 JIT targets.
    ///
    /// When `false` (the default) XMM state is stored as scalar lo/hi
    /// 64-bit pairs and all handlers operate on the pair. This avoids
    /// relying on Cranelift's vector lowering and is required for AOT
    /// targets that lack SIMD support (e.g. RISC-V64).
    pub use_vector_xmm: bool,
    /// When `true`, calls that would otherwise produce a VM exit
    /// (`ImportedFunctionCall` for slow imports, `UnknownJump`/`DirectCall`
    /// for direct out-of-region calls, `IndirectCall` for unregistered
    /// indirect calls) are instead routed through an x86-64 trampoline
    /// (see [`crate::trampoline`]). The trampoline syncs the virtual CPU
    /// state into the real CPU and calls the target natively.
    ///
    /// Only meaningful on x86-64 hosts. Use
    /// [`LifterConfig::set_direct_native_call`] to enable; it returns an
    /// error on non-x86-64 hosts.
    pub direct_native_call: bool,
}

impl Default for LifterConfig {
    fn default() -> Self {
        Self {
            execution_mode: ExecutionMode::Jit,
            syscall_handler_call_conv: CallConv::SystemV,
            segments: Vec::new(),
            interpreter_fallback: false,
            emit_resume_blocks: false,
            imports: HashMap::new(),
            use_vector_xmm: false,
            direct_native_call: false,
        }
    }
}

impl LifterConfig {
    /// Register an import with full fast-call metadata.
    ///
    /// For non-variadic functions the lifter will emit an inline `call_indirect`
    /// using the SysV AMD64 ABI, reading arguments from the appropriate x86
    /// registers (RDI/RSI/RDX/RCX/R8/R9 for integers, XMM0–7 for floats) and
    /// writing the return value into RAX or XMM0.
    ///
    /// Variadic functions (`is_ellipsis_function = true`) are automatically
    /// demoted to slow calls.
    pub fn register_fast_import(&mut self, va: u64, symbol: DynamicSymbol) {
        self.imports.insert(va, Some(symbol));
    }

    /// Register an import address with no fast-call metadata.
    ///
    /// Every call to this address will produce an `ImportedFunctionCall` VM exit
    /// regardless of whether resume blocks are enabled.
    pub fn register_slow_import(&mut self, va: u64) {
        self.imports.insert(va, None);
    }

    /// Enable or disable [`Self::direct_native_call`].
    ///
    /// Returns an error when called with `true` on a host that is not x86-64,
    /// because the trampoline that implements the feature is x86-64 only.
    pub fn set_direct_native_call(&mut self, enabled: bool) -> crate::Result<()> {
        if enabled && !crate::trampoline::HOST_SUPPORTS_DIRECT_NATIVE_CALL {
            return Err(crate::MicroliftError::UnsupportedInstruction {
                mnemonic: "direct_native_call requires an x86-64 host".to_string(),
            });
        }
        self.direct_native_call = enabled;
        Ok(())
    }
}

pub struct Lifter<'a> {
    pub builder: FunctionBuilder<'a>,
    pub state: RegisterState,
    pub config: LifterConfig,
    pub blocks: HashMap<u64, Block>,
    pub sealed_blocks: HashSet<Block>,
    /// The `ip` value passed to `lift()` - i.e. the original VA of the first
    /// instruction in the lifted region.  RIP offsets are relative to this.
    pub lift_base: u64,
    /// One past the last byte of the lifted region (`lift_base + bytes.len()`).
    /// Used to detect out-of-region call targets.
    pub lift_end: u64,
    /// Tracks `instruction.next_ip()` for the instruction currently being
    /// lifted.  This is the value RIP would hold after that instruction
    /// retires, and is what we materialise on `Register::RIP` reads and store
    /// into `RegisterContext::rip` on VM exit.
    pub current_ip: u64,
    /// Blocks that have been filled with instructions during the lift loop.
    pub visited_blocks: HashSet<Block>,
    /// Tracks if the current block has been terminated (jmp, ret, etc.)
    pub block_terminated: bool,
    /// Designated address for controlled VM exit via RET.
    pub exit_trampoline_address: u64,
    pub syscall_handler_sig: Option<SigRef>,
    /// Dirty-bit tracking: GPRs that have been written since the last sync.
    pub dirty_gprs: HashSet<Register>,
    /// Dirty-bit tracking: flags that have been written since the last sync.
    pub dirty_flags: HashSet<X86Flag>,
    /// Dirty-bit tracking: XMM registers (by index 0–15) written since last sync.
    pub dirty_xmms: HashSet<u8>,
    /// Dirty-bit tracking: FS base written since last sync.
    pub dirty_fs_base: bool,
    /// Dirty-bit tracking: GS base written since last sync.
    pub dirty_gs_base: bool,
    /// Pre-built fast-call data for registered imports.
    ///
    /// Key = import VA.  Value = (signature ref, raw fn pointer as i64, arg
    /// types cloned from the `DynamicSymbol`). Only populated for non-variadic
    /// `Some(sym)` entries in `config.imports`.
    pub fast_call_data: HashMap<u64, (SigRef, i64, Vec<FunctionArgumentType>)>,
    /// All registered import virtual addresses (both fast and slow).
    pub import_vas: HashSet<u64>,
    /// Records what kind of x86 transfer targets each Cranelift block.
    /// Used by `finish()` to emit the correct `TransferKind` into `exit_code`
    /// when stubbing out unvisited (out-of-region) blocks.
    pub block_transfer_kinds: HashMap<Block, TransferKind>,
    /// Cranelift signature ref for the direct-native-call trampoline.
    /// Populated when `config.direct_native_call` is true.
    /// Signature: `fn(ctx: i64, target: i64)`.
    pub direct_native_call_sig: Option<SigRef>,
    /// Raw address of the direct-native-call trampoline embedded as an
    /// `i64`. Populated when `config.direct_native_call` is true.
    pub direct_native_call_addr: i64,
}

impl<'a> Lifter<'a> {
    /// Create a new `Lifter`.
    ///
    /// RIP is reconciled dynamically at runtime as `image_base + (current_ip - lift_base)`,
    /// where `image_base` is loaded from the `RegisterContext`. This allows the
    /// lifted code to remain correct even if the program image is relocated by ASLR.
    pub fn new(builder: FunctionBuilder<'a>) -> Self {
        Self::with_config(builder, LifterConfig::default())
    }

    pub fn with_config(builder: FunctionBuilder<'a>, config: LifterConfig) -> Self {
        let mut builder = builder;
        let state = RegisterState::new(&mut builder, config.use_vector_xmm);
        let syscall_handler_sig = match config.execution_mode {
            ExecutionMode::Jit => None,
            ExecutionMode::RuntimeAssisted => {
                let mut sig = Signature::new(config.syscall_handler_call_conv);
                sig.params.push(AbiParam::new(types::I64));
                Some(builder.import_signature(sig))
            }
        };

        // Build fast-call data for every registered non-variadic import.
        let host_cc = get_host_calling_conv();
        let mut fast_call_data: HashMap<u64, (SigRef, i64, Vec<FunctionArgumentType>)> =
            HashMap::new();
        let mut import_vas: HashSet<u64> = HashSet::new();

        for (&va, opt_sym) in &config.imports {
            import_vas.insert(va);
            if let Some(sym) = opt_sym {
                if !sym.is_ellipsis_function {
                    let sig = build_import_sig(&sym.arg_types, host_cc);
                    let sig_ref = builder.import_signature(sig);
                    fast_call_data.insert(va, (sig_ref, sym.value as i64, sym.arg_types.clone()));
                }
            }
        }

        let (direct_native_call_sig, direct_native_call_addr) = if config.direct_native_call {
            let mut sig = Signature::new(CallConv::SystemV);
            sig.params.push(AbiParam::new(types::I64)); // ctx
            sig.params.push(AbiParam::new(types::I64)); // target
            (
                Some(builder.import_signature(sig)),
                crate::trampoline::direct_native_call_trampoline_addr(),
            )
        } else {
            (None, 0)
        };

        Self {
            builder,
            state,
            config,
            blocks: HashMap::new(),
            sealed_blocks: HashSet::new(),
            lift_base: 0,
            lift_end: 0,
            current_ip: 0,
            visited_blocks: HashSet::new(),
            block_terminated: false,
            exit_trampoline_address: 0xFFFF_FFFF_FFFF_FFFF, // Default to a safe high address
            syscall_handler_sig,
            dirty_gprs: HashSet::new(),
            dirty_flags: HashSet::new(),
            dirty_xmms: HashSet::new(),
            dirty_fs_base: false,
            dirty_gs_base: false,
            fast_call_data,
            import_vas,
            block_transfer_kinds: HashMap::new(),
            direct_native_call_sig,
            direct_native_call_addr,
        }
    }

    pub fn get_block(&mut self, offset: u64) -> Block {
        if let Some(&block) = self.blocks.get(&offset) {
            block
        } else {
            let block = self.builder.create_block();
            self.blocks.insert(offset, block);
            block
        }
    }

    /// Like `get_block`, but also records what x86 transfer kind reaches the
    /// block. The first caller wins - a block may be targeted by multiple
    /// edges (e.g. both a JMP and a conditional branch), but the first
    /// registration is sufficient for `finish()` to emit a meaningful exit.
    pub fn get_block_with_kind(&mut self, offset: u64, kind: TransferKind) -> Block {
        let block = self.get_block(offset);
        self.block_transfer_kinds.entry(block).or_insert(kind);
        block
    }

    /// Emit a store of `code` into `RegisterContext::exit_code`.
    /// Call this after any `sync_to_context*` that needs a specific exit code.
    pub fn write_exit_code(&mut self, code: u32) {
        let ctx_ptr = self.builder.use_var(self.state.context_ptr);
        let val = self.builder.ins().iconst(types::I32, code as i64);
        let offset = std::mem::offset_of!(RegisterContext, exit_code) as i32;
        self.builder
            .ins()
            .store(MemFlags::trusted(), val, ctx_ptr, offset);
    }

    pub fn lift(&mut self, bytes: &[u8], ip: u64) -> crate::Result<()> {
        let mut decoder = Decoder::with_ip(64, bytes, ip, DecoderOptions::NONE);
        let mut instructions = Vec::new();
        let mut leaders = HashSet::from([ip]);

        while decoder.can_decode() {
            let mut instruction = Instruction::default();
            decoder.decode_out(&mut instruction);

            if instruction.is_invalid() {
                // Treat as block terminator - next IP may still be a
                // valid branch target from elsewhere in the function.
                leaders.insert(instruction.next_ip());
                instructions.push(instruction); // keep it, phase 2 will emit a fault stub
                continue; // keep scanning past it
            }

            match instruction.mnemonic() {
                Mnemonic::Jmp
                | Mnemonic::Je
                | Mnemonic::Jne
                | Mnemonic::Jb
                | Mnemonic::Jbe
                | Mnemonic::Ja
                | Mnemonic::Jae
                | Mnemonic::Jg
                | Mnemonic::Jge
                | Mnemonic::Jl
                | Mnemonic::Jle
                | Mnemonic::Js
                | Mnemonic::Jns
                | Mnemonic::Jo
                | Mnemonic::Jno
                | Mnemonic::Jp
                | Mnemonic::Jnp
                | Mnemonic::Loop => {
                    leaders.insert(instruction.near_branch_target());
                }
                Mnemonic::Call => {
                    // Only direct (near-branch) calls have a statically known
                    // target. Indirect calls (register/memory) return 0 from
                    // near_branch_target() which we don't want as a leader.
                    if matches!(
                        instruction.op0_kind(),
                        OpKind::NearBranch16 | OpKind::NearBranch32 | OpKind::NearBranch64
                    ) {
                        let target = instruction.near_branch_target();
                        if self.import_vas.contains(&target) {
                            // Import call: don't add the import address as a
                            // block (we intercept it inline). Instead, add the
                            // resume point (next_ip) so it appears in the
                            // entry dispatch switch and slow-call exits can
                            // resume here without needing emit_resume_blocks.
                            leaders.insert(instruction.next_ip());
                        } else if target >= ip && target < ip + bytes.len() as u64 {
                            // In-region call: add the callee as a block leader.
                            leaders.insert(target);
                        } else {
                            // Out-of-region direct call: the callee is outside
                            // the lifted bytes so it will be handled as a
                            // DirectCall VM exit. Add the resume point (next_ip)
                            // so the entry dispatch switch can route back here
                            // after the runtime handles the call - without this
                            // the switch has no entry for the resume RIP and
                            // falls through to start_block, restarting the
                            // function from the top (infinite loop).
                            leaders.insert(instruction.next_ip());
                        }
                    }
                }
                _ => {}
            }

            instructions.push(instruction);
        }

        // Record the base IP for RIP-offset arithmetic.
        self.lift_base = ip;
        self.lift_end = ip + bytes.len() as u64;

        // Create an entry block that receives function parameters
        let entry_block = self.builder.create_block();
        self.builder
            .append_block_params_for_function_params(entry_block);
        self.builder.switch_to_block(entry_block);

        // Use the first function parameter as the context pointer
        let ctx_ptr_val = self.builder.block_params(entry_block)[0];
        self.builder.def_var(self.state.context_ptr, ctx_ptr_val);

        // Load state from context *before* dispatching. Logic in blocks will use
        // the SSA variables defined here.
        self.load_state_from_context()?;

        if self.config.emit_resume_blocks {
            for instruction in &instructions {
                leaders.insert(instruction.ip());
                leaders.insert(instruction.next_ip());
            }
        }

        // Populate blocks map for all leaders
        for &leader in &leaders {
            self.get_block(leader);
        }

        // Resumable Dispatch:
        // Instead of always jumping to the 'ip' passed to lift, we look at the
        // current RIP in the context. If it matches one of our blocks, we jump there.
        // This is what enables resumable AOT.
        let rip_val = self.builder.ins().load(
            types::I64,
            MemFlags::trusted(),
            ctx_ptr_val,
            std::mem::offset_of!(RegisterContext, rip) as i32,
        );
        let image_base_val = self.builder.use_var(self.state.image_base);

        // Normalize RIP to a region-relative offset
        let relative_rip = self.builder.ins().isub(rip_val, image_base_val);

        let mut switch = Switch::new();
        for (&offset, &block) in &self.blocks {
            let relative_offset = offset.wrapping_sub(self.lift_base);
            switch.set_entry(relative_offset.into(), block);
        }

        let start_block = self.get_block(ip);
        switch.emit(&mut self.builder, relative_rip, start_block);

        // We don't switch_to_block here yet, we'll do it in the loop
        self.visited_blocks.insert(start_block);
        self.block_terminated = true; // Force switch to first block in loop

        for instruction in instructions {
            let current_ip_dec = instruction.ip();

            // Check if this IP is a jump target
            if let Some(&block) = self.blocks.get(&current_ip_dec) {
                if Some(block) != self.builder.current_block() || self.block_terminated {
                    if !self.block_terminated {
                        self.builder.ins().jump(block, &[]);
                    }
                    self.builder.switch_to_block(block);
                    self.visited_blocks.insert(block);
                    self.block_terminated = false;
                }
            }

            if self.block_terminated {
                // Dead code after a terminator - decode and skip.
                let mut instruction = Instruction::default();
                decoder.decode_out(&mut instruction);
                continue;
            }

            if instruction.is_invalid() {
                // Sync RIP to point AT the invalid instruction so the host
                // sees the correct faulting address.
                self.current_ip = instruction.ip(); // not next_ip - we didn't execute it
                self.sync_to_context(None, ExitReason::InvalidInstruction)?;
                self.builder.ins().return_(&[]);
                self.block_terminated = true;
                continue;
            }

            // Advance current_ip to next_ip *before* dispatching so that any
            // read of Register::RIP inside the handler sees the correct value
            // (RIP always points past the current instruction on real hardware).
            self.current_ip = instruction.next_ip();

            self.lift_instruction(&instruction)?;
        }

        // Handle fallthrough at the end of the lifted region.
        if !self.block_terminated {
            self.sync_to_context(None, ExitReason::None)?;
            self.builder.ins().return_(&[]);
            self.block_terminated = true;
        }

        self.finish()
    }

    /// Fill any blocks that were created (e.g. as jump targets) but never
    /// visited by the decoder. These represent "VM Exits" to addresses
    /// outside the currently lifted region.
    pub fn finish(&mut self) -> crate::Result<()> {
        let mut processed_blocks = HashSet::new();
        let all_offsets: Vec<u64> = self.blocks.keys().cloned().collect();

        for offset in all_offsets {
            let block = self.blocks[&offset];
            if !self.visited_blocks.contains(&block) && !processed_blocks.contains(&block) {
                self.builder.switch_to_block(block);
                // Sync to context at this target IP, exiting with UnknownJump.
                // `current_ip` is set to the block's address so that
                // `flush_rip_and_reason` materialises it as the target RIP.
                self.current_ip = offset;
                self.sync_to_context(None, ExitReason::UnknownJump)?;
                // Write the transfer kind so the runtime knows how control
                // arrived here (direct JMP, conditional branch, RET, etc.).
                let kind = self
                    .block_transfer_kinds
                    .get(&block)
                    .copied()
                    .unwrap_or(TransferKind::Unknown);
                self.write_exit_code(kind as u32);
                self.builder.ins().return_(&[]);
                processed_blocks.insert(block);
            }
        }

        // Final Sealing Pass: Seal all blocks currently known to the builder.
        // This includes all internal targets, exit blocks, and fallthroughs.
        self.builder.seal_all_blocks();

        Ok(())
    }

    fn lift_instruction(&mut self, instruction: &Instruction) -> crate::Result<()> {
        self.handle_instruction(instruction)
    }

    /// Full sync - writes all GPRs and flags back to the `RegisterContext`.
    /// Used on context reload paths and in unvisited exit stubs where we have
    /// no accurate dirty-set information.
    pub fn sync_to_context(&mut self, rip: Option<Value>, reason: ExitReason) -> crate::Result<()> {
        let ctx_ptr = self.builder.use_var(self.state.context_ptr);

        for (&reg, &var) in &self.state.gprs {
            let val = self.builder.use_var(var);
            let offset = RegisterState::get_gpr_offset(reg)?;
            self.builder
                .ins()
                .store(MemFlags::trusted(), val, ctx_ptr, offset);
        }

        for (&flag, &var) in &self.state.flags {
            let val = self.builder.use_var(var);
            let offset = RegisterState::get_flag_offset(flag)?;
            self.builder
                .ins()
                .store(MemFlags::trusted(), val, ctx_ptr, offset);
        }

        if self.state.use_vector_xmm {
            for i in 0u8..16 {
                let var = *self.state.xmms_vec.get(&i).unwrap();
                let val = self.builder.use_var(var);
                self.builder.ins().store(
                    MemFlags::new().with_notrap(),
                    val,
                    ctx_ptr,
                    RegisterState::get_xmm_vec_offset(i),
                );
            }
        } else {
            for i in 0u8..16 {
                let lo_var = *self.state.xmms_lo.get(&i).unwrap();
                let hi_var = *self.state.xmms_hi.get(&i).unwrap();
                let lo_val = self.builder.use_var(lo_var);
                let hi_val = self.builder.use_var(hi_var);
                self.builder.ins().store(
                    MemFlags::trusted(),
                    lo_val,
                    ctx_ptr,
                    RegisterState::get_xmm_offset(i, false),
                );
                self.builder.ins().store(
                    MemFlags::trusted(),
                    hi_val,
                    ctx_ptr,
                    RegisterState::get_xmm_offset(i, true),
                );
            }
        }

        let fs_base_val = self.builder.use_var(self.state.fs_base);
        self.builder.ins().store(
            MemFlags::trusted(),
            fs_base_val,
            ctx_ptr,
            RegisterState::get_fs_base_offset(),
        );
        let gs_base_val = self.builder.use_var(self.state.gs_base);
        self.builder.ins().store(
            MemFlags::trusted(),
            gs_base_val,
            ctx_ptr,
            RegisterState::get_gs_base_offset(),
        );

        self.flush_rip_and_reason(rip, reason)?;
        Ok(())
    }

    /// Selective sync - only writes GPRs and flags that have been written since
    /// the last `load_state_from_context` call. RIP and exit_reason are always
    /// written. This is the preferred sync to use on normal VM-exit paths.
    pub fn sync_to_context_selective(
        &mut self,
        rip: Option<Value>,
        reason: ExitReason,
    ) -> crate::Result<()> {
        let ctx_ptr = self.builder.use_var(self.state.context_ptr);

        // Clone the dirty sets so we can iterate without borrowing self.
        let dirty_gprs: Vec<Register> = self.dirty_gprs.iter().cloned().collect();
        let dirty_flags: Vec<X86Flag> = self.dirty_flags.iter().cloned().collect();

        for reg in dirty_gprs {
            if let Some(&var) = self.state.gprs.get(&reg) {
                let val = self.builder.use_var(var);
                let offset = RegisterState::get_gpr_offset(reg)?;
                self.builder
                    .ins()
                    .store(MemFlags::trusted(), val, ctx_ptr, offset);
            }
        }

        for flag in dirty_flags {
            if let Some(&var) = self.state.flags.get(&flag) {
                let val = self.builder.use_var(var);
                let offset = RegisterState::get_flag_offset(flag)?;
                self.builder
                    .ins()
                    .store(MemFlags::trusted(), val, ctx_ptr, offset);
            }
        }

        let dirty_xmms: Vec<u8> = self.dirty_xmms.iter().cloned().collect();
        if self.state.use_vector_xmm {
            for idx in dirty_xmms {
                let var = *self.state.xmms_vec.get(&idx).unwrap();
                let val = self.builder.use_var(var);
                self.builder.ins().store(
                    MemFlags::new().with_notrap(),
                    val,
                    ctx_ptr,
                    RegisterState::get_xmm_vec_offset(idx),
                );
            }
        } else {
            for idx in dirty_xmms {
                let lo_var = *self.state.xmms_lo.get(&idx).unwrap();
                let hi_var = *self.state.xmms_hi.get(&idx).unwrap();
                let lo_val = self.builder.use_var(lo_var);
                let hi_val = self.builder.use_var(hi_var);
                self.builder.ins().store(
                    MemFlags::trusted(),
                    lo_val,
                    ctx_ptr,
                    RegisterState::get_xmm_offset(idx, false),
                );
                self.builder.ins().store(
                    MemFlags::trusted(),
                    hi_val,
                    ctx_ptr,
                    RegisterState::get_xmm_offset(idx, true),
                );
            }
        }

        if self.dirty_fs_base {
            let fs_base_val = self.builder.use_var(self.state.fs_base);
            self.builder.ins().store(
                MemFlags::trusted(),
                fs_base_val,
                ctx_ptr,
                RegisterState::get_fs_base_offset(),
            );
        }
        if self.dirty_gs_base {
            let gs_base_val = self.builder.use_var(self.state.gs_base);
            self.builder.ins().store(
                MemFlags::trusted(),
                gs_base_val,
                ctx_ptr,
                RegisterState::get_gs_base_offset(),
            );
        }

        self.flush_rip_and_reason(rip, reason)?;
        Ok(())
    }

    /// Shared helper: write RIP and exit_reason to the context. Always called
    /// by both `sync_to_context` and `sync_to_context_selective`.
    fn flush_rip_and_reason(
        &mut self,
        rip: Option<Value>,
        reason: ExitReason,
    ) -> crate::Result<()> {
        let ctx_ptr = self.builder.use_var(self.state.context_ptr);

        let rip_val = if let Some(r) = rip {
            r
        } else {
            // Reconcile statically from current_ip
            let image_base = self.builder.use_var(self.state.image_base);
            let offset = self.current_ip.wrapping_sub(self.lift_base);
            self.builder.ins().iadd_imm(image_base, offset as i64)
        };

        let rip_offset = std::mem::offset_of!(crate::context::RegisterContext, rip) as i32;
        self.builder
            .ins()
            .store(MemFlags::trusted(), rip_val, ctx_ptr, rip_offset);

        // Store exit reason
        let reason_val = self.builder.ins().iconst(types::I32, reason as i64);
        let reason_offset =
            std::mem::offset_of!(crate::context::RegisterContext, exit_reason) as i32;
        self.builder
            .ins()
            .store(MemFlags::trusted(), reason_val, ctx_ptr, reason_offset);

        Ok(())
    }

    pub fn load_state_from_context(&mut self) -> crate::Result<()> {
        let ctx_ptr = self.builder.use_var(self.state.context_ptr);

        // RIP is not kept as an SSA variable. We materialize it from current_ip
        // when syncing state back to the context.
        for (&reg, &var) in &self.state.gprs {
            let offset = RegisterState::get_gpr_offset(reg)?;
            let val = self
                .builder
                .ins()
                .load(types::I64, MemFlags::trusted(), ctx_ptr, offset);
            self.builder.def_var(var, val);
        }

        for (&flag, &var) in &self.state.flags {
            let offset = RegisterState::get_flag_offset(flag)?;
            let val = self
                .builder
                .ins()
                .load(types::I8, MemFlags::trusted(), ctx_ptr, offset);
            self.builder.def_var(var, val);
        }

        let image_base_offset = std::mem::offset_of!(RegisterContext, image_base) as i32;
        let image_base_val =
            self.builder
                .ins()
                .load(types::I64, MemFlags::trusted(), ctx_ptr, image_base_offset);
        self.builder.def_var(self.state.image_base, image_base_val);

        if self.state.use_vector_xmm {
            for i in 0u8..16 {
                let offset = RegisterState::get_xmm_vec_offset(i);
                let val = self.builder.ins().load(
                    types::I64X2,
                    MemFlags::new().with_notrap(),
                    ctx_ptr,
                    offset,
                );
                let var = *self.state.xmms_vec.get(&i).unwrap();
                self.builder.def_var(var, val);
            }
        } else {
            for i in 0u8..16 {
                let lo_offset = RegisterState::get_xmm_offset(i, false);
                let hi_offset = RegisterState::get_xmm_offset(i, true);
                let lo_val =
                    self.builder
                        .ins()
                        .load(types::I64, MemFlags::trusted(), ctx_ptr, lo_offset);
                let hi_val =
                    self.builder
                        .ins()
                        .load(types::I64, MemFlags::trusted(), ctx_ptr, hi_offset);
                let lo_var = *self.state.xmms_lo.get(&i).unwrap();
                let hi_var = *self.state.xmms_hi.get(&i).unwrap();
                self.builder.def_var(lo_var, lo_val);
                self.builder.def_var(hi_var, hi_val);
            }
        }

        let fs_base_val = self.builder.ins().load(
            types::I64,
            MemFlags::trusted(),
            ctx_ptr,
            RegisterState::get_fs_base_offset(),
        );
        self.builder.def_var(self.state.fs_base, fs_base_val);

        let gs_base_val = self.builder.ins().load(
            types::I64,
            MemFlags::trusted(),
            ctx_ptr,
            RegisterState::get_gs_base_offset(),
        );
        self.builder.def_var(self.state.gs_base, gs_base_val);

        // After a full reload, the dirty sets are cleared - everything in the
        // context is now in sync with our SSA variables.
        self.dirty_gprs.clear();
        self.dirty_flags.clear();
        self.dirty_xmms.clear();
        self.dirty_fs_base = false;
        self.dirty_gs_base = false;

        Ok(())
    }
}
