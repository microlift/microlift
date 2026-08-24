//! AES-NI - `AESENC`, `AESENCLAST`, `AESDEC`, `AESDECLAST`, `AESIMC` and
//! `AESKEYGENASSIST`.
//!
//! Five of the six are one AES round in some form; the sixth folds the S-box
//! into a key-schedule step. The pieces they are built from - `SubBytes`,
//! `ShiftRows`, `MixColumns` - all want a 256-entry S-box table, and, as with
//! [`CRC32`](super::crc32), there is no guest memory this crate owns to put one
//! in. So the S-box is *computed*: the multiplicative inverse in GF(2^8) by an
//! Itoh-Tsujii addition chain, then the affine map.
//!
//! ## Bitslicing
//!
//! Doing that per byte would mean sixteen inverses per round. Instead the state
//! is transposed into **eight planes**: plane `k` is a 16-bit value whose bit
//! `j` is bit `k` of state byte `j`. One plane-level circuit then does all
//! sixteen bytes at once, and every layer of the round becomes cheap:
//!
//! - `SubBytes` is the inverse chain plus the affine map, ~4 field multiplies
//!   for the whole state rather than 16.
//! - `ShiftRows` is a bit permutation *inside* each plane: state byte `4c + r`
//!   takes byte `4c + r + 4r (mod 16)`, i.e. a 16-bit rotate of the bits whose
//!   index is `r` mod 4.
//! - `xtime` is a fixed shuffle of plane indices - six XORs, no shifts.
//! - `MixColumns` needs the other three bytes of each column, which is a rotate
//!   *within* each nibble of a plane.
//!
//! The transposes at the two ends are the movmask multiply and its inverse.
//!
//! Each instruction is one [`AesKernel`], outlined like the x87 and CRC32
//! kernels: a round body is ~1600 IR instructions (~6.5 KiB of x86-64), and the
//! same region usually contains ten of the calls.

use crate::lifter::Lifter;
use cranelift::prelude::*;
use cranelift_codegen::isa::CallConv;
use cranelift_frontend::FunctionBuilder;
use iced_x86::Instruction;

/// Eight bit-planes of a 128-bit state, 16 significant bits each.
type Planes = [Value; 8];

/// Live bits of a plane: one per state byte.
const PLANE: i64 = 0xFFFF;
/// Bit 0 of every byte.
const BYTE_LSBS: i64 = 0x0101_0101_0101_0101u64 as i64;
/// Multiplier that collects bit 0 of each byte into the top byte - `MOVMSKB`
/// without a vector unit.
const COLLECT: i64 = 0x0102_0408_1020_4080u64 as i64;
/// Its inverse's mask: bit `i` of byte `i`, selecting one broadcast copy per byte.
const SELECT: i64 = 0x8040_2010_0804_0201u64 as i64;
/// Per-byte "is non-zero" bump: any of the [`SELECT`] bits carries into bit 7.
const CARRY: i64 = 0x7F7F_7F7F_7F7F_7F7Fu64 as i64;

/// The affine map's constant, `S(x) = A x^-1 + 0x63`.
const AFFINE_CONST: u8 = 0x63;
/// The inverse map's, `A^-1 (y + 0x63) = A^-1 y + 0x05`.
const INV_AFFINE_CONST: u8 = 0x05;

fn shl(b: &mut FunctionBuilder, v: Value, n: i64) -> Value {
    if n == 0 { v } else { b.ins().ishl_imm_u(v, n) }
}

fn shr(b: &mut FunctionBuilder, v: Value, n: i64) -> Value {
    if n == 0 { v } else { b.ins().ushr_imm_u(v, n) }
}

/// Bit `k` of each of the eight bytes of `x`, packed into bits 0..8.
fn collect_bits(b: &mut FunctionBuilder, x: Value, k: i64) -> Value {
    let shifted = shr(b, x, k);
    let masked = b.ins().band_imm_u(shifted, BYTE_LSBS);
    let m = b.ins().iconst(types::I64, COLLECT);
    let spread = b.ins().imul(masked, m);
    b.ins().ushr_imm_u(spread, 56)
}

/// The inverse of [`collect_bits`]: bit `j` of `p` becomes bit 0 of byte `j`.
fn spread_bits(b: &mut FunctionBuilder, p: Value) -> Value {
    let m = b.ins().iconst(types::I64, BYTE_LSBS);
    let broadcast = b.ins().imul(p, m);
    let picked = b.ins().band_imm_u(broadcast, SELECT);
    let bumped = b.ins().iadd_imm_s(picked, CARRY);
    let bit = b.ins().ushr_imm_u(bumped, 7);
    b.ins().band_imm_u(bit, BYTE_LSBS)
}

/// Transpose a `(lo, hi)` state into planes.
fn to_planes(b: &mut FunctionBuilder, lo: Value, hi: Value) -> Planes {
    core::array::from_fn(|k| {
        let l = collect_bits(b, lo, k as i64);
        let h = collect_bits(b, hi, k as i64);
        let h = b.ins().ishl_imm_u(h, 8);
        b.ins().bor(l, h)
    })
}

/// Transpose planes back into a `(lo, hi)` state.
fn from_planes(b: &mut FunctionBuilder, p: &Planes) -> (Value, Value) {
    let mut lo = b.ins().iconst(types::I64, 0);
    let mut hi = b.ins().iconst(types::I64, 0);
    for (k, plane) in p.iter().enumerate() {
        let half_lo = b.ins().band_imm_u(*plane, 0xFF);
        let bits_lo = spread_bits(b, half_lo);
        let bits_lo = shl(b, bits_lo, k as i64);
        lo = b.ins().bor(lo, bits_lo);

        let half_hi = b.ins().ushr_imm_u(*plane, 8);
        let bits_hi = spread_bits(b, half_hi);
        let bits_hi = shl(b, bits_hi, k as i64);
        hi = b.ins().bor(hi, bits_hi);
    }
    (lo, hi)
}

/// Rotate the 16 bits of a plane right by `n`.
fn rotate_plane(b: &mut FunctionBuilder, v: Value, n: i64) -> Value {
    let n = n.rem_euclid(16);
    if n == 0 {
        return v;
    }
    let right = b.ins().ushr_imm_u(v, n);
    let left = b.ins().ishl_imm_u(v, 16 - n);
    let left = b.ins().band_imm_u(left, PLANE);
    b.ins().bor(right, left)
}

/// `ShiftRows`, or its inverse: row `r` of the state rotates by `r` columns,
/// which in plane bits is a rotate by `4r` of the bits at index `r` mod 4.
fn shift_rows(b: &mut FunctionBuilder, p: &Planes, inverse: bool) -> Planes {
    core::array::from_fn(|k| {
        let mut acc = b.ins().band_imm_u(p[k], 0x1111);
        for row in 1..4i64 {
            let group = b.ins().band_imm_u(p[k], 0x1111 << row);
            let n = if inverse { -4 * row } else { 4 * row };
            let rotated = rotate_plane(b, group, n);
            acc = b.ins().bor(acc, rotated);
        }
        acc
    })
}

/// Rotate the four rows of every column by `n`: state byte `4c + r` takes the
/// byte at `4c + (r + n) mod 4`, a rotate inside each nibble of the plane.
fn rotate_rows(b: &mut FunctionBuilder, v: Value, n: u32) -> Value {
    let n = n % 4;
    if n == 0 {
        return v;
    }
    let low: i64 = (0..4).fold(0i64, |m, g| m | (((1i64 << (4 - n)) - 1) << (4 * g)));
    let high = PLANE & !low;
    let right = b.ins().ushr_imm_u(v, n as i64);
    let right = b.ins().band_imm_u(right, low);
    let left = b.ins().ishl_imm_u(v, (4 - n) as i64);
    let left = b.ins().band_imm_u(left, high);
    b.ins().bor(right, left)
}

/// Multiply every byte by `x` in GF(2^8): a shift of the plane indices, with the
/// reduction by `x^8 = x^4 + x^3 + x + 1` folded back into planes 1, 3 and 4.
fn xtime(b: &mut FunctionBuilder, a: &Planes) -> Planes {
    [
        a[7],
        b.ins().bxor(a[0], a[7]),
        a[1],
        b.ins().bxor(a[2], a[7]),
        b.ins().bxor(a[3], a[7]),
        a[4],
        a[5],
        a[6],
    ]
}

fn xor_planes(b: &mut FunctionBuilder, x: &Planes, y: &Planes) -> Planes {
    core::array::from_fn(|k| b.ins().bxor(x[k], y[k]))
}

/// Reduce a 15-term carry-less product modulo the AES polynomial: a term at
/// `x^t`, `t >= 8`, folds back into `t-8`, `t-7`, `t-5` and `t-4`.
fn reduce(b: &mut FunctionBuilder, mut prod: [Option<Value>; 15]) -> Planes {
    for t in (8..15).rev() {
        if let Some(v) = prod[t].take() {
            for offset in [8, 7, 5, 4] {
                let i = t - offset;
                prod[i] = Some(match prod[i] {
                    None => v,
                    Some(u) => b.ins().bxor(u, v),
                });
            }
        }
    }
    core::array::from_fn(|k| match prod[k] {
        Some(v) => v,
        None => b.ins().iconst(types::I64, 0),
    })
}

/// Bytewise GF(2^8) multiply of two bitsliced operands.
fn gf_mul(b: &mut FunctionBuilder, x: &Planes, y: &Planes) -> Planes {
    let mut prod: [Option<Value>; 15] = [None; 15];
    for i in 0..8 {
        for j in 0..8 {
            let term = b.ins().band(x[i], y[j]);
            prod[i + j] = Some(match prod[i + j] {
                None => term,
                Some(v) => b.ins().bxor(v, term),
            });
        }
    }
    reduce(b, prod)
}

/// Bytewise GF(2^8) squaring - linear, so just the spread and the reduction.
fn gf_square(b: &mut FunctionBuilder, x: &Planes) -> Planes {
    let mut prod: [Option<Value>; 15] = [None; 15];
    for i in 0..8 {
        prod[2 * i] = Some(x[i]);
    }
    reduce(b, prod)
}

fn gf_square_n(b: &mut FunctionBuilder, x: &Planes, n: usize) -> Planes {
    (0..n).fold(*x, |acc, _| gf_square(b, &acc))
}

/// `x^254`, the multiplicative inverse in GF(2^8) with `0` mapping to `0`, via
/// the Itoh-Tsujii chain `x^3 -> x^7 -> x^63 -> x^127`. Four multiplies.
fn gf_inverse(b: &mut FunctionBuilder, x: &Planes) -> Planes {
    let x2 = gf_square(b, x);
    let x3 = gf_mul(b, &x2, x);
    let x3sq = gf_square(b, &x3);
    let x7 = gf_mul(b, &x3sq, x);
    let x56 = gf_square_n(b, &x7, 3);
    let x63 = gf_mul(b, &x56, &x7);
    let x126 = gf_square(b, &x63);
    let x127 = gf_mul(b, &x126, x);
    gf_square(b, &x127)
}

/// An affine map over the bits of each byte: `b_i = XOR over taps of
/// a_(i+tap) ^ c_i`, indices mod 8.
fn affine(b: &mut FunctionBuilder, a: &Planes, constant: u8, taps: &[usize]) -> Planes {
    core::array::from_fn(|i| {
        let mut acc = a[(i + taps[0]) % 8];
        for tap in &taps[1..] {
            acc = b.ins().bxor(acc, a[(i + tap) % 8]);
        }
        if constant & (1 << i) != 0 {
            acc = b.ins().bxor_imm_u(acc, PLANE);
        }
        acc
    })
}

fn sub_bytes(b: &mut FunctionBuilder, p: &Planes) -> Planes {
    let inv = gf_inverse(b, p);
    affine(b, &inv, AFFINE_CONST, &[0, 4, 5, 6, 7])
}

/// The inverse affine map has only three taps, and `x^-1` is its own inverse.
fn inv_sub_bytes(b: &mut FunctionBuilder, p: &Planes) -> Planes {
    let undone = affine(b, p, INV_AFFINE_CONST, &[2, 5, 7]);
    gf_inverse(b, &undone)
}

/// `MixColumns`: `d_r = s_r ^ (s_0^s_1^s_2^s_3) ^ xtime(s_r ^ s_r+1)`.
fn mix_columns(b: &mut FunctionBuilder, p: &Planes) -> Planes {
    let r1: Planes = core::array::from_fn(|k| rotate_rows(b, p[k], 1));
    let r2: Planes = core::array::from_fn(|k| rotate_rows(b, p[k], 2));
    let r3: Planes = core::array::from_fn(|k| rotate_rows(b, p[k], 3));
    let sum = xor_planes(b, &r1, &r2);
    let sum = xor_planes(b, &sum, &r3);
    let sum = xor_planes(b, &sum, p);
    let pair = xor_planes(b, p, &r1);
    let doubled = xtime(b, &pair);
    let acc = xor_planes(b, p, &sum);
    xor_planes(b, &acc, &doubled)
}

/// `InvMixColumns` is `MixColumns` over a pre-mixed state: adding
/// `xtime^2(s_r ^ s_r+2)` to each byte turns the `{2,3,1,1}` column into the
/// `{14,11,13,9}` one.
fn inv_mix_columns(b: &mut FunctionBuilder, p: &Planes) -> Planes {
    let r2: Planes = core::array::from_fn(|k| rotate_rows(b, p[k], 2));
    let diff = xor_planes(b, p, &r2);
    let quad = xtime(b, &diff);
    let quad = xtime(b, &quad);
    let pre = xor_planes(b, p, &quad);
    mix_columns(b, &pre)
}

/// One AES round as `AESENC`/`AESDEC` define it: the state transform, then the
/// round key. `last` drops the (Inv)MixColumns step.
pub fn emit_aes_round(
    b: &mut FunctionBuilder,
    state: (Value, Value),
    key: (Value, Value),
    decrypt: bool,
    last: bool,
) -> (Value, Value) {
    let p = to_planes(b, state.0, state.1);
    let p = shift_rows(b, &p, decrypt);
    let p = if decrypt {
        inv_sub_bytes(b, &p)
    } else {
        sub_bytes(b, &p)
    };
    let p = match (last, decrypt) {
        (true, _) => p,
        (false, false) => mix_columns(b, &p),
        (false, true) => inv_mix_columns(b, &p),
    };
    let (lo, hi) = from_planes(b, &p);
    (b.ins().bxor(lo, key.0), b.ins().bxor(hi, key.1))
}

/// `AESIMC`.
pub fn emit_inv_mix_columns(b: &mut FunctionBuilder, src: (Value, Value)) -> (Value, Value) {
    let p = to_planes(b, src.0, src.1);
    let p = inv_mix_columns(b, &p);
    from_planes(b, &p)
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
) -> (Value, Value) {
    // The S-box runs over all sixteen bytes: only two dwords are wanted, but
    // the transpose costs the same either way and the rest simply goes unused.
    let p = to_planes(b, src.0, src.1);
    let subbed = sub_bytes(b, &p);
    let (lo, hi) = from_planes(b, &subbed);

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

    /// Number of `I64` parameters: state, plus the round key or `RCON`.
    fn arity(self) -> usize {
        match self {
            AesKernel::InvMixColumns => 2,
            AesKernel::KeygenAssist => 3,
            _ => 4,
        }
    }

    /// `(state_lo, state_hi, ...) -> (lo, hi)`. `call_conv` should match the
    /// calling function's, so a call needs no convention bridging.
    pub fn signature(self, call_conv: CallConv) -> Signature {
        let mut sig = Signature::new(call_conv);
        for _ in 0..self.arity() {
            sig.params.push(AbiParam::new(types::I64));
        }
        sig.returns.push(AbiParam::new(types::I64));
        sig.returns.push(AbiParam::new(types::I64));
        sig
    }

    /// Emit the operation over `args`, whose length is [`arity`](Self::arity).
    pub fn emit(self, b: &mut FunctionBuilder, args: &[Value]) -> (Value, Value) {
        debug_assert_eq!(args.len(), self.arity());
        let state = (args[0], args[1]);
        match self {
            AesKernel::Enc => emit_aes_round(b, state, (args[2], args[3]), false, false),
            AesKernel::EncLast => emit_aes_round(b, state, (args[2], args[3]), false, true),
            AesKernel::Dec => emit_aes_round(b, state, (args[2], args[3]), true, false),
            AesKernel::DecLast => emit_aes_round(b, state, (args[2], args[3]), true, true),
            AesKernel::InvMixColumns => emit_inv_mix_columns(b, state),
            AesKernel::KeygenAssist => emit_keygen_assist(b, state, args[2]),
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
        let (lo, hi) = self.emit(b, &params);
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
    /// is off - inline at this site. Both paths run the identical builder.
    fn aes_kernel(&mut self, kernel: AesKernel, args: &[Value]) -> (Value, Value) {
        if !self.outline_helpers() {
            return kernel.emit(&mut self.builder, args);
        }

        let call_conv = self.builder.func.signature.call_conv;
        let func_ref = self.outlined_helper(kernel.symbol(), kernel.signature(call_conv), |b| {
            kernel.build_body(b)
        });
        let call = self.builder.ins().call(func_ref, args);
        let results = self.builder.inst_results(call);
        (results[0], results[1])
    }
}
