//! [`LifterConfig::live_immediates`]: a JIT repoints a call by storing a new `MOV r64, imm64`
//! immediate, and the translation follows the store without being lifted again.

use microlift::arch::x86_64::X86_64;
use microlift::core::{AddressMode, LifterConfig, lift_to_code};
use microlift::testkit::prelude::*;

const BASE: u64 = 0x1000;

/// `mov r11, imm64 ; ret`. The immediate is bytes 2..10.
fn mov_r11_then_ret(imm: u64) -> Vec<u8> {
    let mut bytes = vec![0x49, 0xBB];
    bytes.extend_from_slice(&imm.to_le_bytes());
    bytes.push(0xC3);
    bytes
}

#[test]
fn a_live_immediate_is_left_out_of_covered() {
    let config = LifterConfig {
        live_immediates: true,
        ..LifterConfig::default()
    };

    let code = lift_to_code::<X86_64>(&mov_r11_then_ret(0x1122), BASE, config).unwrap();

    assert_eq!(code.covered, vec![BASE..BASE + 2, BASE + 10..BASE + 11]);
}

#[test]
fn without_the_option_the_immediate_is_covered() {
    let code =
        lift_to_code::<X86_64>(&mov_r11_then_ret(0x1122), BASE, LifterConfig::default()).unwrap();

    assert_eq!(code.covered, vec![BASE..BASE + 11]);
}

#[test]
fn a_store_to_a_live_immediate_takes_effect_without_a_new_lift() {
    let mut bytes = mov_r11_then_ret(0x1111);
    // Lifted where the bytes really are, so the load reads this buffer.
    let at = bytes.as_ptr() as u64;
    let first = LiftTest::new(&bytes)
        .at(at)
        .image_base(at)
        .with_config(|c| {
            c.address_mode = AddressMode::Direct;
            c.live_immediates = true;
        })
        .no_unicorn()
        .run();
    assert_eq!(first.ctx.r11, 0x1111);

    bytes[2..10].copy_from_slice(&0x2222u64.to_le_bytes());
    let initial_rsp = first.initial_rsp;
    let second = first.resume_with(|ctx| {
        ctx.rip = at;
        ctx.rsp = initial_rsp;
        ctx.r11 = 0;
    });

    assert_eq!(second.ctx.r11, 0x2222);
}
