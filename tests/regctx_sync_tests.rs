use cranelift::codegen::ir::Opcode;
use cranelift::codegen::isa::CallConv;
use cranelift_codegen::settings::{self, Configurable};
use microlift::arch::x86_64::X86_64;
use microlift::backend::ArtifactBackend;
use microlift::core::{LifterConfig, lift_to_code};
use microlift::testkit::prelude::*;
use microlift::testkit::{DEFAULT_LIFT_BASE, EXIT_TRAMPOLINE};

const TRIPLES: [&str; 2] = ["x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu"];

fn config(lazy: bool) -> LifterConfig {
    LifterConfig {
        lazy_register_sync: lazy,
        exit_trampoline_address: Some(EXIT_TRAMPOLINE),
        ..Default::default()
    }
}

fn flags(opt: &str) -> settings::Flags {
    let mut b = settings::builder();
    b.set("use_colocated_libcalls", "false").unwrap();
    b.set("is_pic", "false").unwrap();
    b.set("opt_level", opt).unwrap();
    settings::Flags::new(b)
}

fn code_len(bytes: &[u8], lazy: bool, opt: &str, triple: &str) -> usize {
    let mut code = lift_to_code::<X86_64>(bytes, DEFAULT_LIFT_BASE, config(lazy)).expect("lift");
    code.func.signature.call_conv = CallConv::SystemV;
    ArtifactBackend::compile_lifted_to_raw(code, Some(triple), Some(flags(opt)))
        .expect("compile")
        .len()
}

/// Count of instructions with `op` in a lift of `bytes`.
fn op_count(bytes: &[u8], lazy: bool, op: Opcode) -> usize {
    let code = lift_to_code::<X86_64>(bytes, DEFAULT_LIFT_BASE, config(lazy)).expect("lift");
    code.func
        .layout
        .blocks()
        .flat_map(|b| code.func.layout.block_insts(b))
        .filter(|&i| code.func.dfg.insts[i].opcode() == op)
        .count()
}

/// `mov rax, 42 ; ret` - touches one GPR and no flags.
const MOV_RET: [u8; 8] = [0x48, 0xC7, 0xC0, 0x2A, 0x00, 0x00, 0x00, 0xC3];

/// `mov rcx, 3 ; dec rcx ; jnz -3 ; ret`.
const LOOP_DEC: [u8; 12] = [
    0x48, 0xC7, 0xC1, 0x03, 0x00, 0x00, 0x00, // mov rcx, 3
    0x48, 0xFF, 0xC9, // dec rcx
    0x75, 0xFB, // jnz -5
];

fn writes_every_gpr() -> Vec<u8> {
    let mut out = Vec::new();
    // xor r, r per GPR (REX.W 31 /r, modrm = 0xC0 | r<<3 | r).
    for r in 0u8..16 {
        if r == 4 {
            continue; // RSP
        }
        let rex = 0x48 | ((r >> 3) * 0b101);
        out.extend_from_slice(&[rex, 0x31, 0xC0 | ((r & 7) << 3) | (r & 7)]);
    }
    out.push(0xC3);
    out
}

#[test]
fn the_prune_changes_no_architectural_state() {
    let loop_dec = {
        let mut v = LOOP_DEC.to_vec();
        v.push(0xC3);
        v
    };
    let cases: [(&str, Vec<u8>); 3] = [
        ("mov/ret", MOV_RET.to_vec()),
        ("loop", loop_dec),
        ("every gpr", writes_every_gpr()),
    ];

    for (name, bytes) in cases {
        for lazy in [false, true] {
            let out = LiftTest::new(&bytes)
                .with_config(|c| c.lazy_register_sync = lazy)
                .reg(|c| {
                    c.rax = 0x1111;
                    c.rbx = 0x2222;
                    c.rcx = 0x3333;
                    c.r15 = 0x4444;
                })
                .run();
            let eager = LiftTest::new(&bytes)
                .with_config(|c| c.lazy_register_sync = false)
                .reg(|c| {
                    c.rax = 0x1111;
                    c.rbx = 0x2222;
                    c.rcx = 0x3333;
                    c.r15 = 0x4444;
                })
                .run();
            assert_eq!(out.ctx.rax, eager.ctx.rax, "{name}: rax (lazy={lazy})");
            assert_eq!(out.ctx.rbx, eager.ctx.rbx, "{name}: rbx (lazy={lazy})");
            assert_eq!(out.ctx.rcx, eager.ctx.rcx, "{name}: rcx (lazy={lazy})");
            assert_eq!(out.ctx.r15, eager.ctx.r15, "{name}: r15 (lazy={lazy})");
            assert_eq!(out.ctx.zf, eager.ctx.zf, "{name}: zf (lazy={lazy})");
        }
    }
}

#[test]
fn the_loop_carried_write_is_still_stored() {
    let mut bytes = LOOP_DEC.to_vec();
    bytes.push(0xC3);
    let out = LiftTest::new(&bytes).reg(|c| c.rcx = 0xDEAD).run();
    assert_eq!(out.ctx.rcx, 0, "the loop must publish its final RCX");
}

#[test]
fn an_untouched_register_is_neither_stored_nor_loaded() {
    let lean_stores = op_count(&MOV_RET, true, Opcode::Store);
    let eager_stores = op_count(&MOV_RET, false, Opcode::Store);
    assert!(
        lean_stores * 4 < eager_stores,
        "a region touching one GPR should shed most of its {eager_stores} stores, kept {lean_stores}"
    );

    let lean_loads = op_count(&MOV_RET, true, Opcode::Load);
    let eager_loads = op_count(&MOV_RET, false, Opcode::Load);
    assert!(
        lean_loads * 4 < eager_loads,
        "and most of its {eager_loads} entry loads, kept {lean_loads}"
    );
}

fn stores_to_field(ir: &str, field: &str) -> usize {
    let decl = ir
        .lines()
        .map(str::trim)
        .find(|l| l.ends_with(&format!("\"{field}\"")) && l.starts_with("region"))
        .unwrap_or_else(|| panic!("no alias region declared for {field}:\n{ir}"));
    let region = decl.split_whitespace().next().expect("region name");
    // Lines carry a srcloc prefix, so match `store` and the region as words.
    ir.lines()
        .filter(|l| {
            let mut words = l.split_whitespace();
            // Opcode prints typed (`store.i64`) or bare.
            let is_store = words.any(|w| w == "store" || w.starts_with("store."));
            is_store && l.split_whitespace().any(|w| w == region)
        })
        .count()
}

#[test]
fn a_written_register_keeps_its_store_and_an_unwritten_one_does_not() {
    let ir = {
        let code = lift_to_code::<X86_64>(&MOV_RET, DEFAULT_LIFT_BASE, config(true)).expect("lift");
        code.func.display().to_string()
    };
    assert!(
        ir.contains("\"rax\"") && ir.contains("\"rbx\""),
        "alias regions should be named after their fields:\n{ir}"
    );
    // Writes RAX (and RSP via the pop) only.
    assert!(
        stores_to_field(&ir, "rax") > 0,
        "RAX is written, so its store must survive:\n{ir}"
    );
    assert_eq!(
        stores_to_field(&ir, "rbx"),
        0,
        "RBX is never written, so no store of it should remain:\n{ir}"
    );
    assert_eq!(
        stores_to_field(&ir, "xmm3"),
        0,
        "nor of any XMM the region never touched:\n{ir}"
    );
}

#[test]
fn the_prune_is_selective_not_merely_enthusiastic() {
    let many = writes_every_gpr();
    let many_lean = op_count(&many, true, Opcode::Store);
    let one_lean = op_count(&MOV_RET, true, Opcode::Store);
    assert!(
        many_lean > one_lean * 3,
        "fifteen written GPRs must keep far more stores than one: \
         {many_lean} vs {one_lean}"
    );

    // Prune still drops the sixteen XMMs and two segment bases.
    let many_eager = op_count(&many, false, Opcode::Store);
    assert!(
        many_lean < many_eager,
        "even a GPR-heavy region has untouched vector state to shed: \
         {many_lean} vs {many_eager}"
    );
}

#[test]
fn the_prune_shrinks_the_compiled_code() {
    for triple in TRIPLES {
        let lean = code_len(&MOV_RET, true, "none", triple);
        let eager = code_len(&MOV_RET, false, "none", triple);
        assert!(
            lean < eager,
            "{triple}: lazy sync should shrink .text ({lean} vs {eager})"
        );
    }
}

#[test]
fn alias_regions_let_cranelift_reach_the_same_answer() {
    for triple in TRIPLES {
        let none = code_len(&MOV_RET, false, "none", triple);
        let speed = code_len(&MOV_RET, false, "speed", triple);
        assert!(
            speed * 2 < none,
            "{triple}: per-field alias regions should let `speed` shed the \
             context traffic ({speed} vs {none})"
        );
    }
}

/// At `speed` the prune is never worse than the alias regions, and sometimes better: deleting a sync store can leave dead block parameters that alias analysis cannot remove.
/// If the prune ever came out larger, something it deleted was paid for twice.
#[test]
fn the_prune_is_never_worse_than_the_alias_regions() {
    for triple in TRIPLES {
        let lean = code_len(&MOV_RET, true, "speed", triple);
        let eager = code_len(&MOV_RET, false, "speed", triple);
        assert!(
            lean <= eager,
            "{triple}: the prune came out larger than leaving it to the alias \
             regions at `speed` ({lean} vs {eager})"
        );
    }
}

#[test]
fn two_lifts_of_the_same_bytes_compile_to_the_same_bytes() {
    for triple in TRIPLES {
        let sizes: Vec<usize> = (0..8)
            .map(|_| code_len(&MOV_RET, true, "none", triple))
            .collect();
        assert!(
            sizes.windows(2).all(|w| w[0] == w[1]),
            "{triple}: lifting is not deterministic: {sizes:?}"
        );
    }
}

#[test]
fn the_default_is_on() {
    assert!(LifterConfig::default().lazy_register_sync);
}

// ---------------------------------------------------------------------------
// Dead block parameters (`microlift::core::dce`)
// ---------------------------------------------------------------------------

/// Block parameters across every block but the entry.
fn block_params(bytes: &[u8], lazy: bool) -> usize {
    let code = lift_to_code::<X86_64>(bytes, DEFAULT_LIFT_BASE, config(lazy)).expect("lift");
    let entry = code.func.layout.entry_block();
    code.func
        .layout
        .blocks()
        .filter(|b| Some(*b) != entry)
        .map(|b| code.func.dfg.block_params(b).len())
        .sum()
}

/// A region's `RET` dispatch has an in-region arm back, so blocks get a parameter per register.
/// The prune deletes their readers; the pass must then drop the parameters. Checked on IR, since regalloc hides the cost.
#[test]
fn the_prune_takes_the_block_parameters_with_it() {
    let lean = block_params(&MOV_RET, true);
    let eager = block_params(&MOV_RET, false);
    assert!(
        lean < eager,
        "deleting the sync stores should leave block parameters with no reader, \
         and those should go too ({lean} vs {eager})"
    );
}

/// Control: a parameter something reads must survive, or the pass would be a miscompile.
#[test]
fn a_parameter_something_reads_is_kept() {
    // Loop so the counter register is a parameter the body reads: mov rcx, 3 ; top: dec rcx ; jnz top ; ret
    #[rustfmt::skip]
    let loop_bytes: &[u8] = &[
        0x48, 0xC7, 0xC1, 0x03, 0x00, 0x00, 0x00, // mov rcx, 3
        0x48, 0xFF, 0xC9,                         // dec rcx
        0x75, 0xFB,                               // jnz -5
        0xC3,                                     // ret
    ];
    assert!(
        block_params(loop_bytes, true) > 0,
        "a loop-carried register must still be a block parameter"
    );

    // And the program still computes the right value.
    let out = LiftTest::new(loop_bytes).run();
    assert_eq!(out.ctx.rcx, 0);
}

/// A parameter feeding only itself around a loop is dead; only the fixpoint sees it (a one-pass scan keeps it because the back-edge names it).
#[test]
fn a_parameter_that_only_feeds_itself_is_not_kept_alive_by_that() {
    #[rustfmt::skip]
    let loop_bytes: &[u8] = &[
        0x48, 0xC7, 0xC1, 0x03, 0x00, 0x00, 0x00, // mov rcx, 3
        0x48, 0xFF, 0xC9,                         // dec rcx
        0x75, 0xFB,                               // jnz -5
        0xC3,                                     // ret
    ];
    let lean = block_params(loop_bytes, true);
    let eager = block_params(loop_bytes, false);
    assert!(
        lean < eager,
        "the loop's untouched registers are passed around it and read by nothing; \
         a fixpoint should see that ({lean} vs {eager})"
    );
}
