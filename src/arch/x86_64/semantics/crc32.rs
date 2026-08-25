//! SSE4.2 `CRC32` - the CRC-32C (Castagnoli) accumulator.
//!
//! `CRC32 r32/r64, r/m8/16/32/64` folds its source into the 32-bit running
//! remainder in the destination's low half, using the reflected CRC-32C
//! polynomial. No flags are touched.
//!
//! Cranelift exposes no host instruction for it, so this is the ordinary
//! byte-at-a-time table algorithm: `crc = (crc >> 8) ^ T[(crc ^ byte) & 0xFF]`,
//! four steps for a `m32` and eight for a `m64`. [`CRC32C_TABLE`] is computed by
//! `const fn` at our build time and emitted into the artifact as a read-only
//! [`DataObject`](crate::core::DataObject) - guest memory this crate does not
//! own is not involved, and neither is any run-time initialisation.
//!
//! At six IR instructions a byte the 64-bit form is ~50, so [`Crc32Kernel`]
//! still outlines each width into a shared helper, as the x87 softfloat kernels
//! do. The table's base is a parameter of that helper rather than a symbol it
//! names itself, so the relocation stays in the region the backend renumbers.

use crate::MicroliftError;
use crate::lifter::Lifter;
use cranelift::prelude::*;
use cranelift_codegen::isa::CallConv;
use cranelift_frontend::FunctionBuilder;
use iced_x86::Instruction;

/// CRC-32C in reflected (LSB-first) form: `0x1EDC6F41` bit-reversed.
const POLY: u32 = 0x82F6_3B78;

pub const CRC32C_TABLE_SYMBOL: &str = "__microlift_crc32c_table";

pub const CRC32C_TABLE: [u32; 256] = build_table();

const fn build_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut acc = i as u32;
        let mut step = 0;
        while step < 8 {
            acc = (acc >> 1) ^ (POLY & (0u32.wrapping_sub(acc & 1)));
            step += 1;
        }
        table[i] = acc;
        i += 1;
    }
    table
}

pub fn table_bytes() -> Vec<u8> {
    CRC32C_TABLE
        .iter()
        .flat_map(|entry| entry.to_le_bytes())
        .collect()
}

/// Fold `bits` bits of `data` into the 32-bit remainder `crc`, reading the
/// table at `table`.
///
/// `crc` is an `I32`; `data` is an `I64` whose low `bits` bits are the operand.
/// The result is an `I32`. `bits` must be 8, 16, 32 or 64.
pub fn emit_crc32c(
    b: &mut FunctionBuilder,
    crc: Value,
    data: Value,
    bits: u32,
    table: Value,
    flags: MemFlagsData,
) -> Value {
    debug_assert!(matches!(bits, 8 | 16 | 32 | 64));
    let mut acc = crc;
    for byte in 0..bits / 8 {
        let shifted = if byte == 0 {
            data
        } else {
            b.ins().ushr_imm_u(data, i64::from(byte) * 8)
        };
        let low = b.ins().ireduce(types::I32, shifted);
        acc = fold_byte(b, acc, low, table, flags);
    }
    acc
}

/// One table step. Only the low byte of `data` is read.
fn fold_byte(
    b: &mut FunctionBuilder,
    crc: Value,
    data: Value,
    table: Value,
    flags: MemFlagsData,
) -> Value {
    let mixed = b.ins().bxor(crc, data);
    let index = b.ins().band_imm_u(mixed, 0xff);
    let offset = b.ins().uextend(types::I64, index);
    let offset = b.ins().ishl_imm_u(offset, 2);
    let address = b.ins().iadd(table, offset);
    let entry = b.ins().load(types::I32, flags, address, 0);
    let shifted = b.ins().ushr_imm_u(crc, 8);
    b.ins().bxor(shifted, entry)
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

    /// `(crc: I32, data: I64, table: I64) -> I32`. `call_conv` should match the
    /// calling function's, so a call needs no convention bridging.
    pub fn signature(self, call_conv: CallConv) -> Signature {
        let mut sig = Signature::new(call_conv);
        sig.params.push(AbiParam::new(types::I32));
        sig.params.push(AbiParam::new(types::I64));
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
        let flags = crate::core::alias::region_flags(
            b.func,
            crate::core::alias::CONST_DATA_ID,
            "constant tables",
        )
        .with_readonly();
        let r = emit_crc32c(b, p[0], p[1], self.bits(), p[2], flags);
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
    /// [`emit_crc32c`] builder over the same emitted table.
    fn crc32_kernel(&mut self, kernel: Crc32Kernel, crc: Value, data: Value) -> Value {
        let table = self.data_constant(CRC32C_TABLE_SYMBOL, 4, table_bytes);

        if !self.outline_helpers() {
            let flags = self.const_data_flags;
            return emit_crc32c(&mut self.builder, crc, data, kernel.bits(), table, flags);
        }

        let call_conv = self.builder.func.signature.call_conv;
        let func_ref = self.outlined_helper(kernel.symbol(), kernel.signature(call_conv), |b| {
            kernel.build_body(b)
        });
        let call = self.builder.ins().call(func_ref, &[crc, data, table]);
        self.builder.inst_results(call)[0]
    }
}
