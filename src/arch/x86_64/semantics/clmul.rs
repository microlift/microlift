//! `PCLMULQDQ`: carry-less 64x64 -> 128 multiply, one outlined shift-and-xor kernel.
//!
//! Branch-free over the multiplier bits, so constant-time.

use crate::lifter::Lifter;
use cranelift::prelude::*;
use cranelift_codegen::isa::CallConv;
use cranelift_frontend::FunctionBuilder;
use iced_x86::Instruction;

/// Helper symbol; stable so AOT regions linked together share one copy.
pub const CLMUL_SYMBOL: &str = "__microlift_clmul64";

/// `(a, b) -> (lo, hi)`, all `I64`.
pub fn clmul_signature(call_conv: CallConv) -> Signature {
    let mut sig = Signature::new(call_conv);
    sig.params.push(AbiParam::new(types::I64));
    sig.params.push(AbiParam::new(types::I64));
    sig.returns.push(AbiParam::new(types::I64));
    sig.returns.push(AbiParam::new(types::I64));
    sig
}

pub fn emit_clmul(b: &mut FunctionBuilder, a: Value, x: Value) -> (Value, Value) {
    let mut lo = b.ins().iconst(types::I64, 0);
    let mut hi = lo;
    for bit in 0..64i64 {
        let at_top = b.ins().ishl_imm_u(x, 63 - bit);
        let mask = b.ins().sshr_imm_u(at_top, 63);
        let low_part = b.ins().ishl_imm_u(a, bit);
        let low_part = b.ins().band(low_part, mask);
        lo = b.ins().bxor(lo, low_part);
        if bit > 0 {
            let high_part = b.ins().ushr_imm_u(a, 64 - bit);
            let high_part = b.ins().band(high_part, mask);
            hi = b.ins().bxor(hi, high_part);
        }
    }
    (lo, hi)
}

pub fn build_clmul_body(b: &mut FunctionBuilder) {
    let entry = b.create_block();
    b.append_block_params_for_function_params(entry);
    b.switch_to_block(entry);
    b.seal_block(entry);
    let p: Vec<Value> = b.block_params(entry).to_vec();
    let (lo, hi) = emit_clmul(b, p[0], p[1]);
    b.ins().return_(&[lo, hi]);
}

impl<'a> Lifter<'a> {
    /// `PCLMULQDQ xmm1, xmm2/m128, imm8`: imm bit 0 picks the `xmm1` quadword, bit 4 the source's.
    pub(super) fn handle_pclmulqdq(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let dst = self.xmm_operand_index(instruction, 0, "pclmulqdq")?;
        let (src_lo, src_hi) = self.read_xmm_src(instruction, 1)?;
        let imm = instruction.immediate8();
        let a = if imm & 0x01 == 0 {
            self.get_xmm_lo(dst)
        } else {
            self.get_xmm_hi(dst)
        };
        let x = if imm & 0x10 == 0 { src_lo } else { src_hi };
        let (lo, hi) = self.clmul_kernel(a, x);
        self.set_xmm_lo(dst, lo);
        self.set_xmm_hi(dst, hi);
        Ok(())
    }

    /// Emit one multiply as a call to the shared helper, or inline when outlining is off.
    fn clmul_kernel(&mut self, a: Value, x: Value) -> (Value, Value) {
        if !self.outline_helpers() {
            return emit_clmul(&mut self.builder, a, x);
        }
        let call_conv = self.builder.func.signature.call_conv;
        let func_ref =
            self.outlined_helper(CLMUL_SYMBOL, clmul_signature(call_conv), build_clmul_body);
        let call = self.builder.ins().call(func_ref, &[a, x]);
        let results = self.builder.inst_results(call);
        (results[0], results[1])
    }
}
