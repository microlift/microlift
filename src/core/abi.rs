//! Guest calling-convention descriptors for fast-import calls.
//!
//! [`CallAbi`] says where the guest put arguments (guest ABI); `host_call_conv`
//! is the native callee's convention. Presets: [`crate::arch::x86_64::abi`].

use crate::function_traits::FunctionArgumentType;
use cranelift_codegen::isa::CallConv;

/// A guest GPR in the frontend's numbering (x86-64: `iced_x86::Register as u32`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GuestReg(pub u32);

/// How an ABI passes arguments that do not fit in registers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StackArgs {
    /// Bytes reserved below the first stack argument (SysV 0, Win64 shadow space 32).
    pub reserved_bytes: u32,
    /// Bytes per stack slot.
    pub slot_bytes: u32,
}

/// Where one argument lives at the moment the guest executes its `CALL`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArgLocation {
    IntReg(GuestReg),
    /// Vector register index (XMM0 = 0).
    FpReg(u8),
    /// `guest_sp + offset`, taken before the call pushes a return address.
    Stack(u32),
}

/// A guest argument-passing convention (presets: [`CallAbi::sysv_amd64`], [`CallAbi::win64`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallAbi {
    /// Diagnostics only.
    pub name: &'static str,
    /// Integer/pointer argument registers, in order.
    pub int_arg_regs: Vec<GuestReg>,
    /// FP argument registers (vector indices), in order.
    pub fp_arg_regs: Vec<u8>,
    /// Win64-style: the Nth argument uses the Nth register of its class. `false`: SysV, classes counted separately.
    pub unified_arg_slots: bool,
    /// `None`: an argument that runs out of registers demotes the import to slow.
    pub stack_args: Option<StackArgs>,
    pub int_ret_reg: GuestReg,
    pub fp_ret_reg: u8,
    /// Host callee's convention; `None` uses [`crate::os::get_host_calling_conv`].
    pub host_call_conv: Option<CallConv>,
}

impl CallAbi {
    /// Where each argument is passed; `arg_types[0]` (return type) is skipped and the
    /// walk stops at the first `VariadicList`/`None`.
    ///
    /// `None` when registers run out and there is no `stack_args` rule.
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

    /// Where the callee leaves the return value; `None` for void.
    pub fn return_location(&self, ret_ty: FunctionArgumentType) -> Option<ArgLocation> {
        match ret_ty {
            FunctionArgumentType::None | FunctionArgumentType::VariadicList => None,
            FunctionArgumentType::SinglePrecisionFp | FunctionArgumentType::DoublePrecisionFp => {
                Some(ArgLocation::FpReg(self.fp_ret_reg))
            }
            _ => Some(ArgLocation::IntReg(self.int_ret_reg)),
        }
    }

    /// Whether every argument can be marshaled inline (else the import is demoted to slow).
    pub fn can_fast_call(&self, arg_types: &[FunctionArgumentType]) -> bool {
        self.place_args(arg_types).is_some()
    }

    pub fn with_stack_args(mut self, stack_args: StackArgs) -> Self {
        self.stack_args = Some(stack_args);
        self
    }

    pub fn with_host_call_conv(mut self, call_conv: CallConv) -> Self {
        self.host_call_conv = Some(call_conv);
        self
    }
}
