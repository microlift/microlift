use microlift::arch::x86_64::X86_64;
use microlift::core::{LifterConfig, lift_to_function};

// mov rax, [rdi] ; mov rcx, [rdi+8] ; mov [rsi], rax ; mov [rsi+8], rcx ; ret
const CODE: [u8; 15] = [
    0x48, 0x8B, 0x07, 0x48, 0x8B, 0x4F, 0x08, 0x48, 0x89, 0x06, 0x48, 0x89, 0x4E, 0x08, 0xC3,
];

fn fences(tso: bool) -> usize {
    let config = LifterConfig {
        total_store_order: tso,
        outline_helpers: false,
        ..LifterConfig::default()
    };
    let func = lift_to_function::<X86_64>(&CODE, 0x1000, config).unwrap();
    func.display().to_string().matches("fence").count()
}

#[test]
fn tso_fences_ordered_guest_accesses() {
    assert_eq!(fences(false), 0);
    // Before each of the two loads after the first, and before each store.
    assert!(fences(true) >= 3, "only {} fences", fences(true));
}

mod across_blocks {
    use cranelift::prelude::*;
    use cranelift_codegen::ir::{Function, Opcode, UserFuncName};
    use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
    use microlift::core::{alias, tso};

    /// A diamond whose arms either both store or leave a load pending, then a load.
    /// Returns how many fences the join block's load got.
    fn fences_before_join_load(second_arm_stores: bool) -> usize {
        let mut func = Function::with_name_signature(
            UserFuncName::user(0, 0),
            Signature::new(isa::CallConv::SystemV),
        );
        func.signature.params.push(AbiParam::new(types::I64));
        let guest = alias::guest_memory_flags(&mut func);
        let mut fctx = FunctionBuilderContext::new();
        let mut b = FunctionBuilder::new(&mut func, &mut fctx);
        let [entry, left, right, join] = [(); 4].map(|_| b.create_block());
        b.append_block_params_for_function_params(entry);
        b.switch_to_block(entry);
        let p = b.block_params(entry)[0];
        let v = b.ins().load(types::I64, guest, p, 0);
        b.ins().brif(v, left, &[], right, &[]);
        b.switch_to_block(left);
        b.ins().store(guest, v, p, 8);
        b.ins().jump(join, &[]);
        b.switch_to_block(right);
        if second_arm_stores {
            b.ins().store(guest, v, p, 16);
        }
        b.ins().jump(join, &[]);
        b.switch_to_block(join);
        b.ins().load(types::I64, guest, p, 24);
        b.ins().return_(&[]);
        b.seal_all_blocks();
        b.finalize(isa::TargetFrontendConfig {
            default_call_conv: isa::CallConv::SystemV,
            pointer_width: target_lexicon::PointerWidth::U64,
            page_size_align_log2: 12,
        });

        tso::insert_fences(&mut func, guest.alias_region());
        func.layout
            .block_insts(join)
            .filter(|&i| func.dfg.insts[i].opcode() == Opcode::Fence)
            .count()
    }

    #[test]
    fn a_load_after_stores_on_every_path_is_not_fenced() {
        assert_eq!(fences_before_join_load(true), 0);
    }

    #[test]
    fn a_load_reachable_from_an_unordered_load_is_fenced() {
        assert_eq!(fences_before_join_load(false), 1);
    }
}


