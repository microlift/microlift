//! Guest calling-convention descriptors for fast-import calls.
//!
//! Turning a guest `CALL` into an inline Cranelift call requires knowing **where
//! the guest put the arguments** - a property of the ABI the *guest binary* was
//! compiled for, not of the host. A Windows guest passes its first integer
//! argument in RCX; a SysV guest uses RDI.
//!
//! [`CallAbi`] is that description, supplied per registered import, so one
//! lifted region can mix conventions. x86-64 presets live in
//! [`crate::arch::x86_64::abi`].
//!
//! Two independent conventions are in play: the **guest** ABI ([`CallAbi`]'s
//! register/stack lists) says where to *read* argument values from, while the
//! **host** ABI ([`CallAbi::host_call_conv`]) is the Cranelift [`CallConv`] of
//! the native callee, i.e. where Cranelift places them when it emits the call.

use crate::function_traits::FunctionArgumentType;
use cranelift_codegen::isa::CallConv;

/// A guest general-purpose register, in the guest frontend's own numbering.
///
/// For x86-64 the payload is `iced_x86::Register as u32`; construct these with
/// the `From<Register>` impl in [`crate::arch::x86_64::abi`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GuestReg(pub u32);

/// How an ABI passes arguments that do not fit in registers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StackArgs {
    /// Bytes the caller reserves below the first stack argument, measured from
    /// the guest stack pointer at the call site. 0 for SysV AMD64, 32 for the
    /// Microsoft x64 "shadow space".
    pub reserved_bytes: u32,
    /// Size of one stack argument slot in bytes (8 on both x86-64 ABIs).
    pub slot_bytes: u32,
}

/// Where one argument lives at the moment the guest executes its `CALL`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArgLocation {
    /// A guest general-purpose register.
    IntReg(GuestReg),
    /// A guest vector register, by index (XMM0 = 0 on x86-64).
    FpReg(u8),
    /// Memory at `guest_sp + offset`. The offset is relative to the stack
    /// pointer *before* the call pushes a return address, which is exactly the
    /// state the lifter is in when it emits the inline call.
    Stack(u32),
}

/// A guest argument-passing convention for fast-import calls.
///
/// Build one with a preset (e.g. [`CallAbi::sysv_amd64`], [`CallAbi::win64`])
/// or by filling in the fields directly for an ABI Microlift does not ship.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallAbi {
    /// Human-readable name, used in diagnostics only.
    pub name: &'static str,
    /// Integer/pointer argument registers, in argument order.
    pub int_arg_regs: Vec<GuestReg>,
    /// Floating-point argument registers (vector-register indices), in order.
    pub fp_arg_regs: Vec<u8>,
    /// When `true`, integer and FP arguments share one positional slot list:
    /// the Nth argument uses the Nth register of its own class and consumes a
    /// slot for the other class too (Microsoft x64). When `false`, the two
    /// classes are counted independently (SysV AMD64).
    pub unified_arg_slots: bool,
    /// Stack passing for arguments that run out of registers, or `None` if any
    /// such argument should demote the import to a slow VM exit instead.
    pub stack_args: Option<StackArgs>,
    /// Register receiving an integer/pointer return value.
    pub int_ret_reg: GuestReg,
    /// Vector register index receiving a floating-point return value.
    pub fp_ret_reg: u8,
    /// Calling convention of the *host* function being called. `None` uses
    /// [`crate::os::get_host_calling_conv`].
    pub host_call_conv: Option<CallConv>,
}

impl CallAbi {
    /// Compute where each argument of `arg_types` is passed.
    ///
    /// `arg_types[0]` is the return type and is skipped; the walk stops at the
    /// first `VariadicList`/`None` marker, matching
    /// [`build_import_sig`](crate::core::build_import_sig).
    ///
    /// `None` when an argument cannot be placed - its class's registers are
    /// exhausted and this ABI has no `stack_args` rule - which callers treat as
    /// "demote to the slow path".
    pub fn place_args(
        &self,
        arg_types: &[FunctionArgumentType],
    ) -> Option<Vec<(FunctionArgumentType, ArgLocation)>> {
        let mut placed = Vec::new();
        let mut int_idx = 0usize;
        let mut fp_idx = 0usize;
        let mut stack_off = self.stack_args.map_or(0, |s| s.reserved_bytes);

        for &ty in arg_types.iter().skip(1) {
            if matches!(
                ty,
                FunctionArgumentType::VariadicList | FunctionArgumentType::None
            ) {
                break;
            }
            let is_fp = matches!(
                ty,
                FunctionArgumentType::SinglePrecisionFp | FunctionArgumentType::DoublePrecisionFp
            );
            // With unified slots the argument's position (= how many arguments
            // have been placed so far) selects the register; otherwise each
            // class has its own counter.
            let reg_idx = if self.unified_arg_slots {
                placed.len()
            } else if is_fp {
                fp_idx
            } else {
                int_idx
            };
            let in_reg = if is_fp {
                self.fp_arg_regs
                    .get(reg_idx)
                    .copied()
                    .map(ArgLocation::FpReg)
            } else {
                self.int_arg_regs
                    .get(reg_idx)
                    .copied()
                    .map(ArgLocation::IntReg)
            };

            let loc = match in_reg {
                Some(loc) => {
                    if is_fp {
                        fp_idx += 1;
                    } else {
                        int_idx += 1;
                    }
                    loc
                }
                None => {
                    let stack = self.stack_args?;
                    let loc = ArgLocation::Stack(stack_off);
                    stack_off += stack.slot_bytes;
                    loc
                }
            };
            placed.push((ty, loc));
        }
        Some(placed)
    }

    /// Where the return value of type `ret_ty` is left by the callee, or `None`
    /// for a void return.
    pub fn return_location(&self, ret_ty: FunctionArgumentType) -> Option<ArgLocation> {
        match ret_ty {
            FunctionArgumentType::None | FunctionArgumentType::VariadicList => None,
            FunctionArgumentType::SinglePrecisionFp | FunctionArgumentType::DoublePrecisionFp => {
                Some(ArgLocation::FpReg(self.fp_ret_reg))
            }
            _ => Some(ArgLocation::IntReg(self.int_ret_reg)),
        }
    }

    /// Whether every argument of `arg_types` can be marshaled inline under this
    /// ABI. Imports that fail this check are demoted to slow VM exits.
    pub fn can_fast_call(&self, arg_types: &[FunctionArgumentType]) -> bool {
        self.place_args(arg_types).is_some()
    }

    /// Builder helper: enable (or replace) stack argument passing.
    pub fn with_stack_args(mut self, stack_args: StackArgs) -> Self {
        self.stack_args = Some(stack_args);
        self
    }

    /// Builder helper: pin the calling convention of the host callee instead of
    /// defaulting to the host's own.
    pub fn with_host_call_conv(mut self, call_conv: CallConv) -> Self {
        self.host_call_conv = Some(call_conv);
        self
    }
}
