use iced_x86::{Decoder, DecoderOptions};
use microlift::context::MemorySegment;
use microlift::vsa::{ValueSet, ValueSetAnalyzer};

#[test]
fn test_vsa_rodata_load() {
    // 0x48, 0x8B, 0x05, 0x00, 0x00, 0x00, 0x00: mov rax, [rip + 0]
    let bytes = [0x48, 0x8B, 0x05, 0x00, 0x00, 0x00, 0x00];
    let ip = 0x1000;
    let mut decoder = Decoder::with_ip(64, &bytes, ip, DecoderOptions::NONE);
    let mut instructions = Vec::new();
    while decoder.can_decode() {
        instructions.push(decoder.decode());
    }

    // Define a rodata segment at the target address (0x1007)
    let target_addr = 0x1007;
    let segment_data = vec![0x2A, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]; // 42 in 64-bit LE
    let segment = MemorySegment::new(target_addr, segment_data, false);

    let mut analyzer = ValueSetAnalyzer::with_segments(vec![segment]);
    analyzer.analyze(&instructions);

    let state = analyzer.value_sets.get(&ip).unwrap();
    let inst = instructions[0];

    // Evaluate the move
    let out_state = analyzer.transfer(state, &inst);
    let rax_vs = out_state.get(iced_x86::Register::RAX);

    assert_eq!(rax_vs, ValueSet::Constant(42));
}

#[test]
fn test_vsa_data_load_unpredictable() {
    // mov rax, [0x2000]
    let bytes = [0x48, 0x8B, 0x04, 0x25, 0x00, 0x20, 0x00, 0x00];
    let ip = 0x1000;
    let mut decoder = Decoder::with_ip(64, &bytes, ip, DecoderOptions::NONE);
    let mut instructions = Vec::new();
    while decoder.can_decode() {
        instructions.push(decoder.decode());
    }

    // Define a writable segment at 0x2000
    let segment = MemorySegment::new(0x2000, vec![0; 64], true);

    let mut analyzer = ValueSetAnalyzer::with_segments(vec![segment]);
    analyzer.analyze(&instructions);

    let state = analyzer.value_sets.get(&ip).unwrap();
    let out_state = analyzer.transfer(state, &instructions[0]);
    let rax_vs = out_state.get(iced_x86::Register::RAX);

    // Should be Top because it's writable
    assert_eq!(rax_vs, ValueSet::Top);
}

#[test]
fn test_vsa_indirect_jump_resolution() {
    // mov eax, [0x2000] (0x2000 maps to 0x3000 in rodata)
    // jmp rax
    let bytes = [
        0x48, 0x8B, 0x04, 0x25, 0x00, 0x20, 0x00, 0x00, // mov rax, [0x2000]
        0xFF, 0xE0, // jmp rax
    ];
    let ip = 0x1000;
    let mut decoder = Decoder::with_ip(64, &bytes, ip, DecoderOptions::NONE);
    let mut instructions = Vec::new();
    while decoder.can_decode() {
        instructions.push(decoder.decode());
    }

    let segment_data = vec![0x00, 0x30, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]; // 0x3000
    let segment = MemorySegment::new(0x2000, segment_data, false);

    let mut analyzer = ValueSetAnalyzer::with_segments(vec![segment]);
    analyzer.analyze(&instructions);

    let jmp_inst = instructions[1];
    let targets = analyzer.indirect_jump_targets(&jmp_inst);

    assert!(targets.is_some());
    assert_eq!(targets.unwrap(), vec![0x3000]);
}
