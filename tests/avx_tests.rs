//! VEX-128 integer ops. Each lifted VEX instruction must equal its architectural definition, the
//! legacy SSE form with a non-destructive first source (`dst = src1; op dst, rest..`). The legacy
//! code is found by name from iced, independently of the lifter's table.

use iced_x86::{Code, Encoder, Instruction, OpCodeOperandKind, OpKind, Register};
use microlift::arch::x86_64::flag_effects::flag_effects;
use microlift::core::flags::FlagEffects;
use microlift::testkit::oracle::{CpuOracle, CpuState, MicroliftOracle, OracleOutcome, gpr};
use microlift::testkit::prelude::*;
use microlift::testkit::rng::Rng;

const CODE_VA: u64 = 0x1000;
const CASES_PER_CODE: usize = 24;

const VEX_CODES: &[Code] = &[
    Code::VEX_Vmovdqa_xmm_xmmm128,
    Code::VEX_Vmovdqa_xmmm128_xmm,
    Code::VEX_Vmovdqu_xmm_xmmm128,
    Code::VEX_Vmovdqu_xmmm128_xmm,
    Code::VEX_Vmovaps_xmm_xmmm128,
    Code::VEX_Vmovaps_xmmm128_xmm,
    Code::VEX_Vmovups_xmm_xmmm128,
    Code::VEX_Vmovups_xmmm128_xmm,
    Code::VEX_Vmovd_xmm_rm32,
    Code::VEX_Vmovd_rm32_xmm,
    Code::VEX_Vmovq_xmm_rm64,
    Code::VEX_Vmovq_rm64_xmm,
    Code::VEX_Vmovq_xmm_xmmm64,
    Code::VEX_Vmovq_xmmm64_xmm,
    Code::VEX_Vpxor_xmm_xmm_xmmm128,
    Code::VEX_Vpor_xmm_xmm_xmmm128,
    Code::VEX_Vpand_xmm_xmm_xmmm128,
    Code::VEX_Vpandn_xmm_xmm_xmmm128,
    Code::VEX_Vxorps_xmm_xmm_xmmm128,
    Code::VEX_Vpaddb_xmm_xmm_xmmm128,
    Code::VEX_Vpaddw_xmm_xmm_xmmm128,
    Code::VEX_Vpaddd_xmm_xmm_xmmm128,
    Code::VEX_Vpaddq_xmm_xmm_xmmm128,
    Code::VEX_Vpsubb_xmm_xmm_xmmm128,
    Code::VEX_Vpsubw_xmm_xmm_xmmm128,
    Code::VEX_Vpsubd_xmm_xmm_xmmm128,
    Code::VEX_Vpsubq_xmm_xmm_xmmm128,
    Code::VEX_Vpmuludq_xmm_xmm_xmmm128,
    Code::VEX_Vpclmulqdq_xmm_xmm_xmmm128_imm8,
    Code::VEX_Vpsrlw_xmm_xmm_xmmm128,
    Code::VEX_Vpsrld_xmm_xmm_xmmm128,
    Code::VEX_Vpsrlq_xmm_xmm_xmmm128,
    Code::VEX_Vpsllw_xmm_xmm_xmmm128,
    Code::VEX_Vpslld_xmm_xmm_xmmm128,
    Code::VEX_Vpsllq_xmm_xmm_xmmm128,
    Code::VEX_Vpsraw_xmm_xmm_xmmm128,
    Code::VEX_Vpsrad_xmm_xmm_xmmm128,
    Code::VEX_Vpsrlw_xmm_xmm_imm8,
    Code::VEX_Vpsrld_xmm_xmm_imm8,
    Code::VEX_Vpsrlq_xmm_xmm_imm8,
    Code::VEX_Vpsllw_xmm_xmm_imm8,
    Code::VEX_Vpslld_xmm_xmm_imm8,
    Code::VEX_Vpsllq_xmm_xmm_imm8,
    Code::VEX_Vpsraw_xmm_xmm_imm8,
    Code::VEX_Vpsrad_xmm_xmm_imm8,
    Code::VEX_Vpsrldq_xmm_xmm_imm8,
    Code::VEX_Vpslldq_xmm_xmm_imm8,
    Code::VEX_Vpshufb_xmm_xmm_xmmm128,
    Code::VEX_Vpshufd_xmm_xmmm128_imm8,
    Code::VEX_Vpalignr_xmm_xmm_xmmm128_imm8,
    Code::VEX_Vpunpcklbw_xmm_xmm_xmmm128,
    Code::VEX_Vpunpcklwd_xmm_xmm_xmmm128,
    Code::VEX_Vpunpckldq_xmm_xmm_xmmm128,
    Code::VEX_Vpunpcklqdq_xmm_xmm_xmmm128,
    Code::VEX_Vpunpckhbw_xmm_xmm_xmmm128,
    Code::VEX_Vpunpckhwd_xmm_xmm_xmmm128,
    Code::VEX_Vpunpckhdq_xmm_xmm_xmmm128,
    Code::VEX_Vpunpckhqdq_xmm_xmm_xmmm128,
    Code::VEX_Vpcmpeqb_xmm_xmm_xmmm128,
    Code::VEX_Vpcmpeqw_xmm_xmm_xmmm128,
    Code::VEX_Vpcmpeqd_xmm_xmm_xmmm128,
    Code::VEX_Vpcmpeqq_xmm_xmm_xmmm128,
    Code::VEX_Vpcmpgtb_xmm_xmm_xmmm128,
    Code::VEX_Vpcmpgtw_xmm_xmm_xmmm128,
    Code::VEX_Vpcmpgtd_xmm_xmm_xmmm128,
    Code::VEX_Vpcmpgtq_xmm_xmm_xmmm128,
    Code::VEX_Vpinsrb_xmm_xmm_r32m8_imm8,
    Code::VEX_Vpinsrw_xmm_xmm_r32m16_imm8,
    Code::VEX_Vpinsrd_xmm_xmm_rm32_imm8,
    Code::VEX_Vpinsrq_xmm_xmm_rm64_imm8,
    Code::VEX_Vpextrb_r32m8_xmm_imm8,
    Code::VEX_Vpextrw_r32_xmm_imm8,
    Code::VEX_Vpextrw_r32m16_xmm_imm8,
    Code::VEX_Vpextrd_rm32_xmm_imm8,
    Code::VEX_Vpextrq_rm64_xmm_imm8,
    Code::VEX_Vaesenc_xmm_xmm_xmmm128,
    Code::VEX_Vaesenclast_xmm_xmm_xmmm128,
    Code::VEX_Vaesdec_xmm_xmm_xmmm128,
    Code::VEX_Vaesdeclast_xmm_xmm_xmmm128,
    Code::VEX_Vaesimc_xmm_xmmm128,
    Code::VEX_Vaeskeygenassist_xmm_xmmm128_imm8,
];

/// Small pools so operands alias often; XMM9/XMM15 and R9 exercise the extended encodings.
const XMM_POOL: [Register; 5] = [
    Register::XMM0,
    Register::XMM1,
    Register::XMM2,
    Register::XMM9,
    Register::XMM15,
];
const GPR32_POOL: [Register; 3] = [Register::EAX, Register::ECX, Register::R9D];
const GPR64_POOL: [Register; 3] = [Register::RAX, Register::RCX, Register::R9];
const MEM_BASE: Register = Register::RSI;

#[repr(C, align(16))]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Buffer([u8; 32]);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Operand {
    Reg(Register),
    Mem,
    Imm(u8),
}

struct Case {
    vex: Vec<Operand>,
    init: CpuState,
    buffer: Buffer,
}

fn legacy_counterpart(vex: Code) -> Code {
    let name = format!("{vex:?}");
    let rest = name.strip_prefix("VEX_V").expect("a VEX code");
    let mut tokens: Vec<String> = rest.split('_').map(str::to_owned).collect();
    tokens[0] = capitalize(&tokens[0]);
    let same = tokens.join("_");
    tokens.remove(2);
    let merged = tokens.join("_");
    find_code(&same)
        .or_else(|| find_code(&merged))
        .unwrap_or_else(|| panic!("no legacy counterpart for {name}"))
}

fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    let first = chars.next().expect("non-empty").to_ascii_uppercase();
    std::iter::once(first).chain(chars).collect()
}

fn find_code(name: &str) -> Option<Code> {
    Code::values().find(|c| format!("{c:?}") == name)
}

fn operand_count(code: Code) -> usize {
    code.op_code().op_count() as usize
}

fn merges(vex: Code) -> bool {
    operand_count(vex) == operand_count(legacy_counterpart(vex)) + 1
}

fn random_operand(kind: OpCodeOperandKind, rng: &mut Rng) -> Operand {
    use OpCodeOperandKind as K;
    let pick = |pool: &[Register], rng: &mut Rng| pool[rng.below(pool.len())];
    let maybe_mem = |reg: Operand, rng: &mut Rng| {
        if rng.below(4) == 0 { Operand::Mem } else { reg }
    };
    match kind {
        K::xmm_reg | K::xmm_vvvv | K::xmm_rm => Operand::Reg(pick(&XMM_POOL, rng)),
        K::xmm_or_mem => maybe_mem(Operand::Reg(pick(&XMM_POOL, rng)), rng),
        K::r32_reg => Operand::Reg(pick(&GPR32_POOL, rng)),
        K::r32_or_mem => maybe_mem(Operand::Reg(pick(&GPR32_POOL, rng)), rng),
        K::r64_or_mem => maybe_mem(Operand::Reg(pick(&GPR64_POOL, rng)), rng),
        K::imm8 => Operand::Imm(rng.next_u64() as u8),
        other => panic!("operand kind {other:?} has no generator"),
    }
}

fn random_case(code: Code, rng: &mut Rng) -> Case {
    let kinds = code.op_code().op_kinds()[..operand_count(code)].to_vec();
    let vex = kinds.iter().map(|&k| random_operand(k, rng)).collect();
    let mut init = CpuState::zeroed();
    for xmm in init.xmm.iter_mut() {
        *xmm = [rng.next_u64(), rng.next_u64()];
    }
    for slot in [gpr::RAX, gpr::RCX, 9] {
        init.gpr[slot] = rng.next_u64();
    }
    let mut buffer = Buffer([0; 32]);
    for byte in buffer.0.iter_mut() {
        *byte = rng.next_u64() as u8;
    }
    Case { vex, init, buffer }
}

fn instruction(code: Code, operands: &[Operand]) -> Instruction {
    let mut i = Instruction::default();
    i.set_code(code);
    for (n, &op) in operands.iter().enumerate() {
        let n = n as u32;
        match op {
            Operand::Reg(r) => {
                i.set_op_kind(n, OpKind::Register);
                i.set_op_register(n, r);
            }
            Operand::Mem => {
                i.set_op_kind(n, OpKind::Memory);
                i.set_memory_base(MEM_BASE);
            }
            Operand::Imm(v) => {
                i.set_op_kind(n, OpKind::Immediate8);
                i.set_immediate8(v);
            }
        }
    }
    i
}

fn assemble(instructions: &[Instruction]) -> Vec<u8> {
    let mut encoder = Encoder::new(64);
    let mut ip = CODE_VA;
    for i in instructions {
        let len = encoder
            .encode(i, ip)
            .unwrap_or_else(|e| panic!("cannot encode {:?}: {e}", i.code()));
        ip += len as u64;
    }
    let mut bytes = encoder.take_buffer();
    bytes.push(0xC3);
    bytes
}

fn xmm_index(r: Register) -> usize {
    r.number()
}

/// The legacy sequence `dst = src1; op dst, rest..`. A later source that names `dst` reads a spare
/// register seeded with the same value, so the copy does not clobber it.
fn legacy_program(code: Code, case: &mut Case) -> Vec<Instruction> {
    let legacy = legacy_counterpart(code);
    if !merges(code) {
        return vec![instruction(legacy, &case.vex)];
    }
    let (Operand::Reg(dst), Operand::Reg(src1)) = (case.vex[0], case.vex[1]) else {
        panic!("{code:?}: a merging form starts with two registers");
    };
    let mut rest: Vec<Operand> = case.vex[2..].to_vec();
    let mut program = Vec::new();
    if dst != src1 {
        let spare = spare_xmm(&case.vex);
        for op in rest.iter_mut() {
            if *op == Operand::Reg(dst) {
                case.init.xmm[xmm_index(spare)] = case.init.xmm[xmm_index(dst)];
                *op = Operand::Reg(spare);
            }
        }
        program.push(instruction(
            Code::Movdqa_xmm_xmmm128,
            &[Operand::Reg(dst), Operand::Reg(src1)],
        ));
    }
    let operands: Vec<Operand> = std::iter::once(Operand::Reg(dst)).chain(rest).collect();
    program.push(instruction(legacy, &operands));
    program
}

fn spare_xmm(used: &[Operand]) -> Register {
    [Register::XMM5, Register::XMM6, Register::XMM7]
        .into_iter()
        .find(|r| !used.contains(&Operand::Reg(*r)))
        .expect("a spare register outside the pool")
}

/// Final state and buffer; `rsi` points at a fresh copy of `buffer`.
fn run(
    oracle: &mut impl CpuOracle,
    code: &[u8],
    init: &CpuState,
    buffer: Buffer,
) -> (CpuState, Buffer) {
    let mut memory = Box::new(buffer);
    let mut init = *init;
    init.gpr[gpr::RSI] = memory.0.as_mut_ptr() as u64;
    let outcome = oracle.run(code, CODE_VA, &init, &[], 1 << 16);
    let OracleOutcome::Halted(mut state) = outcome else {
        panic!("{} did not halt cleanly: {outcome:?}", oracle.name());
    };
    state.gpr[gpr::RSI] = 0;
    state.gpr[gpr::RSP] = 0;
    state.rip = 0;
    (state, *memory)
}

fn microlift(vector_xmm: bool) -> MicroliftOracle {
    MicroliftOracle {
        vector_xmm,
        ..MicroliftOracle::new(Opt::Speed)
    }
}

fn for_every_case(mut check: impl FnMut(Code, &[u8], &[u8], &CpuState, Buffer)) {
    let mut rng = Rng::new(0x5EED_0A5C);
    for &code in VEX_CODES {
        for _ in 0..CASES_PER_CODE {
            let mut case = random_case(code, &mut rng);
            let vex_bytes = assemble(&[instruction(code, &case.vex)]);
            let legacy_bytes = assemble(&legacy_program(code, &mut case));
            check(code, &vex_bytes, &legacy_bytes, &case.init, case.buffer);
        }
    }
}

#[test]
fn every_lowered_form_matches_its_legacy_definition() {
    for vector_xmm in [false, true] {
        for_every_case(|code, vex, legacy, init, buffer| {
            let got = run(&mut microlift(vector_xmm), vex, init, buffer);
            let want = run(&mut microlift(vector_xmm), legacy, init, buffer);
            assert_eq!(
                got, want,
                "{code:?} (vector_xmm={vector_xmm})\n  vex    = {vex:02x?}\n  legacy = {legacy:02x?}"
            );
        });
    }
}

#[cfg(feature = "unicorn-oracle")]
#[test]
fn every_lowered_form_matches_unicorn_on_the_legacy_definition() {
    use microlift::testkit::oracle::unicorn::UnicornOracle;
    for_every_case(|code, vex, legacy, init, buffer| {
        let got = run(&mut microlift(true), vex, init, buffer);
        let want = run(&mut UnicornOracle::new(), legacy, init, buffer);
        assert_eq!(
            got, want,
            "{code:?}\n  vex    = {vex:02x?}\n  legacy = {legacy:02x?}"
        );
    });
}

#[test]
fn every_vex_code_under_test_is_a_128_bit_form() {
    for &code in VEX_CODES {
        let name = format!("{code:?}");
        assert!(!name.contains("ymm"), "{name} is not VEX-128");
        legacy_counterpart(code);
    }
}

/// `vpsubd xmm1, xmm0, xmm1`: the destination is also the second source.
#[test]
fn destination_aliasing_the_second_source_reads_it_before_the_copy() {
    let code = [0xC5, 0xF9, 0xFA, 0xC9, 0xC3];
    for vector_xmm in [false, true] {
        let out = LiftTest::new(&code)
            .vector_xmm(vector_xmm)
            .reg(|c| {
                c.xmm[0] = [100 | (200 << 32), 300 | (400 << 32)];
                c.xmm[1] = [1 | (2 << 32), 3 | (4 << 32)];
            })
            .run();
        assert_eq!(out.ctx.xmm[1], [99 | (198 << 32), 297 | (396 << 32)]);
        assert_eq!(out.ctx.xmm[0], [100 | (200 << 32), 300 | (400 << 32)]);
    }
}

fn clmul_reference(a: u64, b: u64) -> u128 {
    (0..64)
        .filter(|bit| b >> bit & 1 == 1)
        .fold(0u128, |acc, bit| acc ^ ((a as u128) << bit))
}

/// `vpclmulqdq xmm2, xmm0, xmm1, imm` for every quadword selector.
#[test]
fn vpclmulqdq_selects_quadwords_and_multiplies_without_carries() {
    let a = [0x8000_0000_0000_0001u64, 0xFFFF_FFFF_FFFF_FFFF];
    let b = [0xDEAD_BEEF_0BAD_F00Du64, 0x8765_4321_1234_5678];
    for imm in [0x00u8, 0x01, 0x10, 0x11] {
        let code = [0xC4, 0xE3, 0x79, 0x44, 0xD1, imm, 0xC3];
        let expected = clmul_reference(a[(imm & 1) as usize], b[(imm >> 4) as usize]);
        for vector_xmm in [false, true] {
            let out = LiftTest::new(&code)
                .vector_xmm(vector_xmm)
                .reg(|c| {
                    c.xmm[0] = a;
                    c.xmm[1] = b;
                })
                .run();
            assert_eq!(
                out.ctx.xmm[2],
                [expected as u64, (expected >> 64) as u64],
                "imm={imm:#x}"
            );
        }
    }
}

/// Legacy `pclmulqdq xmm0, xmm1, imm` over a seeded sweep (Unicorn cross-checks it under `unicorn-oracle`).
#[test]
fn pclmulqdq_matches_a_reference_multiply() {
    let mut rng = Rng::new(0xC1_0C);
    for _ in 0..64 {
        let a = [rng.next_u64(), rng.next_u64()];
        let b = [rng.next_u64(), rng.next_u64()];
        let imm = [0x00u8, 0x01, 0x10, 0x11][rng.below(4)];
        let code = [0x66, 0x0F, 0x3A, 0x44, 0xC1, imm, 0xC3];
        let expected = clmul_reference(a[(imm & 1) as usize], b[(imm >> 4) as usize]);
        let out = LiftTest::new(&code)
            .reg(|c| {
                c.xmm[0] = a;
                c.xmm[1] = b;
            })
            .run();
        assert_eq!(out.ctx.xmm[0], [expected as u64, (expected >> 64) as u64]);
    }
}

#[test]
fn vzeroall_clears_every_xmm_and_vzeroupper_keeps_them() {
    let seeded = |c: &mut RegisterContext| {
        for (i, xmm) in c.xmm.iter_mut().enumerate() {
            *xmm = [i as u64 + 1, !(i as u64)];
        }
    };
    for vector_xmm in [false, true] {
        let zeroall = LiftTest::new(&[0xC5, 0xFC, 0x77, 0xC3])
            .vector_xmm(vector_xmm)
            .reg(seeded)
            .run();
        assert!(zeroall.ctx.xmm.iter().all(|x| *x == [0, 0]));

        let zeroupper = LiftTest::new(&[0xC5, 0xF8, 0x77, 0xC3])
            .vector_xmm(vector_xmm)
            .reg(seeded)
            .run();
        for (i, xmm) in zeroupper.ctx.xmm.iter().enumerate() {
            assert_eq!(*xmm, [i as u64 + 1, !(i as u64)]);
        }
    }
}

/// `vpaddd ymm0, ymm1, ymm2` has no lowering and must not run as its 128-bit form.
#[test]
fn vex_256_is_not_lowered() {
    let code = [0xC5, 0xF5, 0xFE, 0xC2, 0xC3];
    assert!(LiftTest::new(&code).try_run().is_err());
}

fn decode(bytes: &[u8]) -> Instruction {
    iced_x86::Decoder::with_ip(64, bytes, 0, iced_x86::DecoderOptions::NONE).decode()
}

#[test]
fn vex_forms_inherit_the_flag_effects_of_their_legacy_form() {
    for bytes in [
        &[0xC5, 0xF9, 0xFE, 0xC2][..],
        &[0xC4, 0xE3, 0x79, 0x44, 0xD1, 0x00][..],
        &[0xC5, 0xF8, 0x77][..],
        &[0xC5, 0xFC, 0x77][..],
    ] {
        let i = decode(bytes);
        assert_eq!(
            flag_effects(&i),
            Some(FlagEffects::TRANSPARENT),
            "{:?}",
            i.code()
        );
    }
}
