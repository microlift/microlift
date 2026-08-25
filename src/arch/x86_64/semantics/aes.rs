//! AES-NI - `AESENC`, `AESENCLAST`, `AESDEC`, `AESDECLAST`, `AESIMC` and
//! `AESKEYGENASSIST`.
//!
//! Five of the six are one AES round in some form; the sixth folds the S-box
//! into a key-schedule step. All of them are built out of `SubBytes`,
//! `ShiftRows` and `MixColumns`, and all three are table work: the S-box, its
//! inverse, and `xtime` (multiply by `x` in GF(2^8)) are computed by `const fn`
//! at our build time and emitted into the artifact as one read-only
//! [`DataObject`](crate::core::DataObject), the way
//! [`CRC32`](super::crc32)'s table is. Nothing is derived at run time and no
//! guest memory is involved.
//!
//! A round is then 16 S-box loads and, where `MixColumns` runs, 16 `xtime`
//! loads over bytes unpacked from the two halves of the state: 233 IR
//! instructions for an `AESENC` body, against 1621 for the bitsliced circuit
//! this replaces, which folded a GF(2^8) inverse chain on every invocation. The
//! object for one `AESENC` region goes from 7104 bytes to 2224, tables and all.
//! Each instruction is still one [`AesKernel`], outlined like the x87 and CRC32
//! kernels, since a region usually holds ten of the calls.
//!
//! This is a table-driven AES, so it is not constant-time: the load
//! addresses depend on the state, where the guest's own AES-NI would not. A
//! guest whose key material must survive a cache-timing attacker cannot get
//! that property from a lift that touches memory at data-dependent addresses.

use crate::lifter::Lifter;
use cranelift::prelude::*;
use cranelift_codegen::isa::CallConv;
use cranelift_frontend::FunctionBuilder;
use iced_x86::Instruction;

/// The sixteen bytes of a 128-bit state, each an `I64` holding `0..=255`.
///
/// Byte `4c + r` is row `r` of column `c`, the order AES numbers the state in
/// and the order the little-endian halves of an XMM register already hold.
type State = [Value; 16];

/// Symbol the tables are emitted under. Namespaced and stable for the same
/// reason the kernels' symbols are: two AOT-lifted regions linked together must
/// share one copy.
pub const AES_TABLE_SYMBOL: &str = "__microlift_aes_tables";

/// Byte offsets of the three tables within that blob.
const SBOX_OFFSET: i32 = 0;
const INV_SBOX_OFFSET: i32 = 256;
const XTIME_OFFSET: i32 = 512;

/// Total size of the blob, and the alignment it is emitted with.
pub const AES_TABLE_LEN: usize = 768;

/// `a * b` in GF(2^8) modulo `x^8 + x^4 + x^3 + x + 1`.
const fn gf_mul(a: u8, b: u8) -> u8 {
    let mut acc = 0u8;
    let mut x = a;
    let mut bits = b;
    let mut step = 0;
    while step < 8 {
        if bits & 1 != 0 {
            acc ^= x;
        }
        let high = x & 0x80;
        x <<= 1;
        if high != 0 {
            x ^= 0x1B;
        }
        bits >>= 1;
        step += 1;
    }
    acc
}

/// The multiplicative inverse, with `0` mapping to `0`. Found by search: this
/// runs once at compile time, so the addition chain the bitsliced version needed
/// buys nothing here.
const fn gf_inverse(a: u8) -> u8 {
    if a == 0 {
        return 0;
    }
    let mut candidate = 1u16;
    while candidate < 256 {
        if gf_mul(a, candidate as u8) == 1 {
            return candidate as u8;
        }
        candidate += 1;
    }
    0
}

/// The AES S-box: the inverse, then the affine map
/// `s ^ rotl(s,1) ^ rotl(s,2) ^ rotl(s,3) ^ rotl(s,4) ^ 0x63`.
const fn sbox_table() -> [u8; 256] {
    let mut table = [0u8; 256];
    let mut i = 0;
    while i < 256 {
        let s = gf_inverse(i as u8);
        table[i] = s ^ s.rotate_left(1) ^ s.rotate_left(2) ^ s.rotate_left(3) ^ s.rotate_left(4)
            ^ 0x63;
        i += 1;
    }
    table
}

/// The S-box inverted by scanning it, rather than by writing the inverse affine
/// map out a second time.
const fn inv_sbox_table(sbox: &[u8; 256]) -> [u8; 256] {
    let mut table = [0u8; 256];
    let mut i = 0;
    while i < 256 {
        table[sbox[i] as usize] = i as u8;
        i += 1;
    }
    table
}

/// Multiply by `x`: a shift, with the reduction folded back in.
const fn xtime_table() -> [u8; 256] {
    let mut table = [0u8; 256];
    let mut i = 0;
    while i < 256 {
        table[i] = gf_mul(i as u8, 2);
        i += 1;
    }
    table
}

pub const AES_SBOX: [u8; 256] = sbox_table();
pub const AES_INV_SBOX: [u8; 256] = inv_sbox_table(&AES_SBOX);
pub const AES_XTIME: [u8; 256] = xtime_table();

/// The three tables as the one blob that goes into the artifact, in
/// [`SBOX_OFFSET`] / [`INV_SBOX_OFFSET`] / [`XTIME_OFFSET`] order.
pub fn table_bytes() -> Vec<u8> {
    let mut bytes = Vec::with_capacity(AES_TABLE_LEN);
    bytes.extend_from_slice(&AES_SBOX);
    bytes.extend_from_slice(&AES_INV_SBOX);
    bytes.extend_from_slice(&AES_XTIME);
    bytes
}

/// Where the tables are and how to read them, threaded through the round so no
/// layer has to name the blob itself.
#[derive(Copy, Clone)]
pub struct Tables {
    /// Base address of the blob.
    pub base: Value,
    /// Memflags for a load out of it.
    pub flags: MemFlagsData,
}

impl Tables {
    /// `table[index]`, zero-extended. `index` must already be masked to a byte,
    /// which every caller here gets from an unpack or a previous lookup.
    fn get(self, b: &mut FunctionBuilder, table: i32, index: Value) -> Value {
        let address = b.ins().iadd(self.base, index);
        b.ins().uload8(types::I64, self.flags, address, table)
    }
}

fn xor(b: &mut FunctionBuilder, x: Value, y: Value) -> Value {
    b.ins().bxor(x, y)
}

/// Unpack a `(lo, hi)` state into its sixteen bytes.
fn split_state(b: &mut FunctionBuilder, lo: Value, hi: Value) -> State {
    core::array::from_fn(|j| {
        let half = if j < 8 { lo } else { hi };
        let shifted = match (j % 8) * 8 {
            0 => half,
            n => b.ins().ushr_imm_u(half, n as i64),
        };
        b.ins().band_imm_u(shifted, 0xFF)
    })
}

/// Pack sixteen bytes back into a `(lo, hi)` state.
fn pack_state(b: &mut FunctionBuilder, s: &State) -> (Value, Value) {
    let mut half = |bytes: &[Value]| {
        let mut acc = bytes[0];
        for (i, byte) in bytes.iter().enumerate().skip(1) {
            let placed = b.ins().ishl_imm_u(*byte, (i * 8) as i64);
            acc = b.ins().bor(acc, placed);
        }
        acc
    };
    (half(&s[..8]), half(&s[8..]))
}

/// `ShiftRows`, or its inverse: row `r` rotates left by `r` columns. A pure
/// renumbering of the bytes - it emits nothing.
fn shift_rows(s: &State, inverse: bool) -> State {
    core::array::from_fn(|j| {
        let (row, column) = (j % 4, j / 4);
        let source = if inverse {
            (column + 4 - row) % 4
        } else {
            (column + row) % 4
        };
        s[4 * source + row]
    })
}

fn sub_bytes(b: &mut FunctionBuilder, s: &State, tables: Tables, inverse: bool) -> State {
    let table = if inverse { INV_SBOX_OFFSET } else { SBOX_OFFSET };
    core::array::from_fn(|j| tables.get(b, table, s[j]))
}

/// `MixColumns`: `d_r = xtime(a_r ^ a_r+1) ^ a_r+1 ^ a_r+2 ^ a_r+3`, which is
/// the `{2,3,1,1}` column with the two doublings shared.
fn mix_columns(b: &mut FunctionBuilder, s: &State, tables: Tables) -> State {
    core::array::from_fn(|j| {
        let (row, column) = (j % 4, j / 4);
        let a = |r: usize| s[4 * column + (row + r) % 4];
        let pair = xor(b, a(0), a(1));
        let doubled = tables.get(b, XTIME_OFFSET, pair);
        let rest = xor(b, a(1), a(2));
        let rest = xor(b, rest, a(3));
        xor(b, doubled, rest)
    })
}

/// `InvMixColumns` is `MixColumns` over a pre-mixed state: adding
/// `xtime^2(a_r ^ a_r+2)` to each byte turns the `{2,3,1,1}` column into the
/// `{14,11,13,9}` one.
fn inv_mix_columns(b: &mut FunctionBuilder, s: &State, tables: Tables) -> State {
    let pre: State = core::array::from_fn(|j| {
        let (row, column) = (j % 4, j / 4);
        let across = s[4 * column + (row + 2) % 4];
        let diff = xor(b, s[j], across);
        let doubled = tables.get(b, XTIME_OFFSET, diff);
        let quad = tables.get(b, XTIME_OFFSET, doubled);
        xor(b, s[j], quad)
    });
    mix_columns(b, &pre, tables)
}

/// One AES round as `AESENC`/`AESDEC` define it: the state transform, then the
/// round key. `last` drops the (Inv)MixColumns step.
pub fn emit_aes_round(
    b: &mut FunctionBuilder,
    state: (Value, Value),
    key: (Value, Value),
    tables: Tables,
    decrypt: bool,
    last: bool,
) -> (Value, Value) {
    let s = split_state(b, state.0, state.1);
    let s = shift_rows(&s, decrypt);
    let s = sub_bytes(b, &s, tables, decrypt);
    let s = match (last, decrypt) {
        (true, _) => s,
        (false, false) => mix_columns(b, &s, tables),
        (false, true) => inv_mix_columns(b, &s, tables),
    };
    let (lo, hi) = pack_state(b, &s);
    (b.ins().bxor(lo, key.0), b.ins().bxor(hi, key.1))
}

/// `AESIMC`.
pub fn emit_inv_mix_columns(
    b: &mut FunctionBuilder,
    src: (Value, Value),
    tables: Tables,
) -> (Value, Value) {
    let s = split_state(b, src.0, src.1);
    let s = inv_mix_columns(b, &s, tables);
    pack_state(b, &s)
}

/// One half of an `AESKEYGENASSIST` result: the substituted word stays in the
/// low dword, and its `RotWord` with `RCON` added goes in the high one.
fn keygen_half(b: &mut FunctionBuilder, subbed: Value, rcon: Value) -> Value {
    let word = b.ins().ushr_imm_u(subbed, 32);
    let right = b.ins().ushr_imm_u(word, 8);
    let left = b.ins().ishl_imm_u(word, 24);
    let left = b.ins().band_imm_u(left, 0xFFFF_FFFF);
    let rotated = b.ins().bor(right, left);
    let keyed = b.ins().bxor(rotated, rcon);
    let high = b.ins().ishl_imm_u(keyed, 32);
    b.ins().bor(word, high)
}

/// `AESKEYGENASSIST`: `SubWord` of dwords 1 and 3, each also rotated a byte and
/// XORed with `RCON` into the dword above it.
pub fn emit_keygen_assist(
    b: &mut FunctionBuilder,
    src: (Value, Value),
    rcon: Value,
    tables: Tables,
) -> (Value, Value) {
    // The S-box runs over all sixteen bytes: only two dwords are wanted, but
    // the unpack costs the same either way and the rest simply goes unused.
    let s = split_state(b, src.0, src.1);
    let s = sub_bytes(b, &s, tables, false);
    let (lo, hi) = pack_state(b, &s);

    let rcon = b.ins().band_imm_u(rcon, 0xFF);
    (keygen_half(b, lo, rcon), keygen_half(b, hi, rcon))
}

/// One outlinable AES expansion. The key-schedule round constant is a parameter
/// rather than part of the kernel, so all 256 `imm8`s share one body.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum AesKernel {
    Enc,
    EncLast,
    Dec,
    DecLast,
    InvMixColumns,
    KeygenAssist,
}

/// Every kernel, for callers that want to pre-emit or enumerate them.
pub const AES_KERNELS: [AesKernel; 6] = [
    AesKernel::Enc,
    AesKernel::EncLast,
    AesKernel::Dec,
    AesKernel::DecLast,
    AesKernel::InvMixColumns,
    AesKernel::KeygenAssist,
];

impl AesKernel {
    /// The symbol the helper is declared under, namespaced and stable so two
    /// AOT-lifted regions linked together share one copy.
    pub fn symbol(self) -> &'static str {
        match self {
            AesKernel::Enc => "__microlift_aesenc",
            AesKernel::EncLast => "__microlift_aesenclast",
            AesKernel::Dec => "__microlift_aesdec",
            AesKernel::DecLast => "__microlift_aesdeclast",
            AesKernel::InvMixColumns => "__microlift_aesimc",
            AesKernel::KeygenAssist => "__microlift_aeskeygenassist",
        }
    }

    /// Number of `I64` parameters: state, the round key or `RCON`, and the
    /// table base last.
    fn arity(self) -> usize {
        let operands = match self {
            AesKernel::InvMixColumns => 2,
            AesKernel::KeygenAssist => 3,
            _ => 4,
        };
        operands + 1
    }

    /// `(state_lo, state_hi, ..., tables) -> (lo, hi)`. `call_conv` should match
    /// the calling function's, so a call needs no convention bridging.
    pub fn signature(self, call_conv: CallConv) -> Signature {
        let mut sig = Signature::new(call_conv);
        for _ in 0..self.arity() {
            sig.params.push(AbiParam::new(types::I64));
        }
        sig.returns.push(AbiParam::new(types::I64));
        sig.returns.push(AbiParam::new(types::I64));
        sig
    }

    /// Emit the operation over `args`, whose length is [`arity`](Self::arity)
    /// and whose last element is the table base.
    pub fn emit(self, b: &mut FunctionBuilder, args: &[Value], flags: MemFlagsData) -> (Value, Value) {
        debug_assert_eq!(args.len(), self.arity());
        let tables = Tables {
            base: args[self.arity() - 1],
            flags,
        };
        let state = (args[0], args[1]);
        match self {
            AesKernel::Enc => emit_aes_round(b, state, (args[2], args[3]), tables, false, false),
            AesKernel::EncLast => emit_aes_round(b, state, (args[2], args[3]), tables, false, true),
            AesKernel::Dec => emit_aes_round(b, state, (args[2], args[3]), tables, true, false),
            AesKernel::DecLast => emit_aes_round(b, state, (args[2], args[3]), tables, true, true),
            AesKernel::InvMixColumns => emit_inv_mix_columns(b, state, tables),
            AesKernel::KeygenAssist => emit_keygen_assist(b, state, args[2], tables),
        }
    }

    /// Emit the kernel's body into `b`, whose function must already have been
    /// given [`signature`](Self::signature).
    pub fn build_body(self, b: &mut FunctionBuilder) {
        let entry = b.create_block();
        b.append_block_params_for_function_params(entry);
        b.switch_to_block(entry);
        b.seal_block(entry);
        let params: Vec<Value> = b.block_params(entry).to_vec();
        let flags = crate::core::alias::region_flags(
            b.func,
            crate::core::alias::CONST_DATA_ID,
            "constant tables",
        )
        .with_readonly();
        let (lo, hi) = self.emit(b, &params, flags);
        b.ins().return_(&[lo, hi]);
    }
}

impl<'a> Lifter<'a> {
    /// `AESENC` / `AESENCLAST` / `AESDEC` / `AESDECLAST xmm1, xmm2/m128`.
    ///
    /// The state is the destination register, the round key the source; the
    /// result replaces the destination.
    pub(super) fn handle_aes_round(
        &mut self,
        instruction: &Instruction,
        kernel: AesKernel,
    ) -> crate::Result<()> {
        let dst = self.xmm_operand_index(instruction, 0, kernel.symbol())?;
        let (key_lo, key_hi) = self.read_xmm_src(instruction, 1)?;
        let lo = self.get_xmm_lo(dst);
        let hi = self.get_xmm_hi(dst);
        let (res_lo, res_hi) = self.aes_kernel(kernel, &[lo, hi, key_lo, key_hi]);
        self.set_xmm_lo(dst, res_lo);
        self.set_xmm_hi(dst, res_hi);
        Ok(())
    }

    /// `AESIMC xmm1, xmm2/m128` - the key-schedule half of decryption, where the
    /// source is the operand rather than the destination.
    pub(super) fn handle_aesimc(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let kernel = AesKernel::InvMixColumns;
        let dst = self.xmm_operand_index(instruction, 0, kernel.symbol())?;
        let (lo, hi) = self.read_xmm_src(instruction, 1)?;
        let (res_lo, res_hi) = self.aes_kernel(kernel, &[lo, hi]);
        self.set_xmm_lo(dst, res_lo);
        self.set_xmm_hi(dst, res_hi);
        Ok(())
    }

    /// `AESKEYGENASSIST xmm1, xmm2/m128, imm8`.
    pub(super) fn handle_aeskeygenassist(
        &mut self,
        instruction: &Instruction,
    ) -> crate::Result<()> {
        let kernel = AesKernel::KeygenAssist;
        let dst = self.xmm_operand_index(instruction, 0, kernel.symbol())?;
        let (lo, hi) = self.read_xmm_src(instruction, 1)?;
        let rcon = instruction.immediate8() as i64;
        let rcon = self.builder.ins().iconst(types::I64, rcon);
        let (res_lo, res_hi) = self.aes_kernel(kernel, &[lo, hi, rcon]);
        self.set_xmm_lo(dst, res_lo);
        self.set_xmm_hi(dst, res_hi);
        Ok(())
    }

    /// Emit one AES kernel, as a call to the shared helper or - when outlining
    /// is off - inline at this site. Both paths run the identical builder over
    /// the same emitted tables, whose base is the kernel's last argument.
    fn aes_kernel(&mut self, kernel: AesKernel, operands: &[Value]) -> (Value, Value) {
        let base = self.data_constant(AES_TABLE_SYMBOL, 16, table_bytes);
        let mut args = operands.to_vec();
        args.push(base);

        if !self.outline_helpers() {
            let flags = self.const_data_flags;
            return kernel.emit(&mut self.builder, &args, flags);
        }

        let call_conv = self.builder.func.signature.call_conv;
        let func_ref = self.outlined_helper(kernel.symbol(), kernel.signature(call_conv), |b| {
            kernel.build_body(b)
        });
        let call = self.builder.ins().call(func_ref, &args);
        let results = self.builder.inst_results(call);
        (results[0], results[1])
    }
}
