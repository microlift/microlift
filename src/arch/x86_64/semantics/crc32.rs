//! SSE4.2 `CRC32` - the CRC-32C (Castagnoli) accumulator.
//!
//! `CRC32 r32/r64, r/m8/16/32/64` folds its source into the 32-bit running
//! remainder in the destination's low half, using the *reflected* CRC-32C
//! polynomial. No flags are touched.
//!
//! Cranelift exposes no host instruction for it, and the usual 256-entry table
//! would have to live in guest memory this crate does not own. What is left is
//! the bit-serial loop, made cheap enough to unroll by:
//!
//! > XORing a whole `k`-bit chunk of data into the low bits of the remainder and
//! > then shifting `k` times is the same as consuming the data one bit at a time,
//! > for any `k` up to the register width (32).
//!
//! So `m8` is 8 shift steps and `m32` is 32; `m64` is two 32-bit folds. At four
//! IR instructions a step the 64-bit form is ~260, so [`Crc32Kernel`] outlines
//! each width into a shared helper, as the x87 softfloat kernels do.

use crate::MicroliftError;
use crate::lifter::Lifter;
use cranelift::prelude::*;
use cranelift_codegen::isa::CallConv;
use cranelift_frontend::FunctionBuilder;
use iced_x86::Instruction;

/// CRC-32C in reflected (LSB-first) form: `0x1EDC6F41` bit-reversed.
const POLY: i64 = 0x82F6_3B78;

/// Fold `bits` bits of `data` into the 32-bit remainder `crc`.
///
/// `crc` is an `I32`; `data` is an `I64` whose low `bits` bits are the operand.
/// The result is an `I32`. `bits` must be 8, 16, 32 or 64.
pub fn emit_crc32c(b: &mut FunctionBuilder, crc: Value, data: Value, bits: u32) -> Value {
    debug_assert!(matches!(bits, 8 | 16 | 32 | 64));
    let low = b.ins().ireduce(types::I32, data);
    let mut acc = fold(b, crc, low, bits.min(32));
    if bits == 64 {
        // The second half is an ordinary 32-bit fold of the same remainder -
        // see the chunking identity in the module note.
        let shifted = b.ins().ushr_imm_u(data, 32);
        let high = b.ins().ireduce(types::I32, shifted);
        acc = fold(b, acc, high, 32);
    }
    acc
}

/// One chunk: XOR the low `bits` of `data` into `crc`, then run `bits`
/// remainder steps.
fn fold(b: &mut FunctionBuilder, crc: Value, data: Value, bits: u32) -> Value {
    let masked = if bits == 32 {
        data
    } else {
        b.ins().band_imm_u(data, ((1u64 << bits) - 1) as i64)
    };
    let mut acc = b.ins().bxor(crc, masked);
    for _ in 0..bits {
        // acc = (acc >> 1) ^ (POLY & -(acc & 1))
        let bit = b.ins().band_imm_u(acc, 1);
        // 0 or 1 negated is 0 or all-ones: the branchless "was the outgoing bit
        // set" mask, which is cheaper than a select on every backend.
        let mask = b.ins().ineg(bit);
        let poly = b.ins().band_imm_u(mask, POLY);
        let shifted = b.ins().ushr_imm_u(acc, 1);
        acc = b.ins().bxor(shifted, poly);
    }
    acc
}

/// One outlinable `CRC32` expansion, named by its *source* operand width.
///
/// The destination width does not appear here: `CRC32 r64, r/m8` and
/// `CRC32 r32, r/m8` fold the same 8 bits into the same 32-bit remainder and
/// differ only in how the caller writes the result back.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Crc32Kernel {
    Byte,
    Word,
    Dword,
    Qword,
}

/// Every kernel, for callers that want to pre-emit or enumerate them.
pub const CRC32_KERNELS: [Crc32Kernel; 4] = [
    Crc32Kernel::Byte,
    Crc32Kernel::Word,
    Crc32Kernel::Dword,
    Crc32Kernel::Qword,
];

impl Crc32Kernel {
    /// The kernel for a source operand of `size` bytes.
    pub fn for_size(size: usize) -> Option<Crc32Kernel> {
        Some(match size {
            1 => Crc32Kernel::Byte,
            2 => Crc32Kernel::Word,
            4 => Crc32Kernel::Dword,
            8 => Crc32Kernel::Qword,
            _ => return None,
        })
    }

    /// Source width in bits.
    pub fn bits(self) -> u32 {
        match self {
            Crc32Kernel::Byte => 8,
            Crc32Kernel::Word => 16,
            Crc32Kernel::Dword => 32,
            Crc32Kernel::Qword => 64,
        }
    }

    /// The symbol the helper function is declared under. Namespaced and stable
    /// for the same reason the x87 kernels' are: two AOT-lifted regions linked
    /// together must share one copy.
    pub fn symbol(self) -> &'static str {
        match self {
            Crc32Kernel::Byte => "__microlift_crc32c_u8",
            Crc32Kernel::Word => "__microlift_crc32c_u16",
            Crc32Kernel::Dword => "__microlift_crc32c_u32",
            Crc32Kernel::Qword => "__microlift_crc32c_u64",
        }
    }

    /// `(crc: I32, data: I64) -> I32`. `call_conv` should match the calling
    /// function's, so a call needs no convention bridging.
    pub fn signature(self, call_conv: CallConv) -> Signature {
        let mut sig = Signature::new(call_conv);
        sig.params.push(AbiParam::new(types::I32));
        sig.params.push(AbiParam::new(types::I64));
        sig.returns.push(AbiParam::new(types::I32));
        sig
    }

    /// Emit the kernel's body into `b`, whose function must already have been
    /// given [`signature`](Self::signature).
    pub fn build_body(self, b: &mut FunctionBuilder) {
        let entry = b.create_block();
        b.append_block_params_for_function_params(entry);
        b.switch_to_block(entry);
        b.seal_block(entry);
        let p: Vec<Value> = b.block_params(entry).to_vec();
        let r = emit_crc32c(b, p[0], p[1], self.bits());
        b.ins().return_(&[r]);
    }
}

impl<'a> Lifter<'a> {
    /// `CRC32 r32/r64, r/m8/16/32/64`.
    ///
    /// The remainder lives in the destination's low 32 bits whatever the
    /// destination width, so the `r64` forms zero the upper half explicitly rather
    /// than through a 32-bit write. No flag is touched.
    pub(super) fn handle_crc32(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let src_size = self.operand_size(instruction, 1)?;
        let kernel = Crc32Kernel::for_size(src_size).ok_or_else(|| {
            MicroliftError::UnsupportedInstruction {
                mnemonic: format!("crc32-src-{}", src_size),
            }
        })?;

        let src = self.read_operand(instruction, 1, src_size)?;
        let src64 = if self.builder.func.dfg.value_type(src) == types::I64 {
            src
        } else {
            self.builder.ins().uextend(types::I64, src)
        };

        let dst_reg = self.operand_register(instruction, 0);
        let dst = self.get_reg_val(dst_reg)?;
        let crc = self.narrow_to(types::I32, dst);

        let result = self.crc32_kernel(kernel, crc, src64);

        if dst_reg.size() == 8 {
            let wide = self.builder.ins().uextend(types::I64, result);
            self.set_reg_val(dst_reg, wide)
        } else {
            self.set_reg_val(dst_reg, result)
        }
    }

    /// Emit one `CRC32` fold, as a call to the shared helper or - when
    /// outlining is off - inline at this site. Both paths run the identical
    /// [`emit_crc32c`] builder.
    fn crc32_kernel(&mut self, kernel: Crc32Kernel, crc: Value, data: Value) -> Value {
        if !self.outline_helpers() {
            return emit_crc32c(&mut self.builder, crc, data, kernel.bits());
        }

        let call_conv = self.builder.func.signature.call_conv;
        let func_ref = self.outlined_helper(kernel.symbol(), kernel.signature(call_conv), |b| {
            kernel.build_body(b)
        });
        let call = self.builder.ins().call(func_ref, &[crc, data]);
        self.builder.inst_results(call)[0]
    }
}
