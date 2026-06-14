//! Compile a real C shared library, extract a function from
//! the ELF using the linker map file, and lift + execute it through microlift
//! with slow-import handling for `strlen` and `strcmp`.
//!
//! The C function (`process_string`) calls `strlen` to measure a string,
//! duplicates it when length > 1, advances the first byte by one, then returns
//! `strcmp(original, mutated)`.  Microlift intercepts both library calls as
//! `ImportedFunctionCall` VM exits; the test handles each exit in a loop,
//! calling host-side pure-Rust replacements.

use cranelift::prelude::*;
use cranelift_codegen::ir::{Function, UserFuncName};
use microlift::backend::CraneliftBackend;
use microlift::context::{ExitReason, RegisterContext};
use microlift::lifter::{Lifter, LifterConfig};
use microlift::os::get_host_calling_conv;
use object::{Object, ObjectSection, ObjectSymbol};
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;

const C_SOURCE: &str = r#"
#include <string.h>

/* Called with a string `s`.
 *   len = strlen(s)
 *   if len <= 1: return 0
 *   copy = duplicate of s
 *   copy[0] = s[0] + 1 (shift the first character up by one)
 *   return strcmp(s, copy)
 */
int process_string(const char *s) {
    int len = (int)strlen(s);
    if (len <= 1) return 0;

    char copy[256];
    int i;
    for (i = 0; i < len + 1; i++) {
        copy[i] = s[i];
    }
    copy[0] = (char)(s[0] + 1);

    return strcmp(s, copy);
}
"#;

unsafe extern "C" fn host_strlen(s: *const u8) -> u64 {
    let mut n = 0u64;
    while unsafe { *s.add(n as usize) } != 0 {
        n += 1;
    }
    n
}

unsafe extern "C" fn host_strcmp(s1: *const u8, s2: *const u8) -> i32 {
    let mut i = 0;
    loop {
        let c1 = unsafe { *s1.add(i) };
        let c2 = unsafe { *s2.add(i) };
        if c1 != c2 {
            return (c1 as i32) - (c2 as i32);
        }
        if c1 == 0 {
            return 0;
        }
        i += 1;
    }
}

/// Directory inside the source tree used for compiled test artifacts.
/// Listed in `.gitignore`.
fn artifact_dir() -> PathBuf {
    // CARGO_MANIFEST_DIR points to the crate root at test time.
    let root = std::env::var("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."));
    root.join("test-artifacts").join("elf_import_test")
}

/// Compile the C source to a shared library with a GNU ld map file.
/// Returns `(elf_bytes, map_file_contents)`.
fn compile_shared_lib() -> (Vec<u8>, String) {
    let dir = artifact_dir();
    std::fs::create_dir_all(&dir).expect("failed to create artifact dir");

    let c_path = dir.join("process_string.c");
    let so_path = dir.join("libprocess_string.so");
    let map_path = dir.join("process_string.map");

    std::fs::write(&c_path, C_SOURCE).expect("failed to write C source");

    let map_arg = format!("-Wl,-Map={}", map_path.display());
    let status = Command::new("gcc")
        .args([
            "-shared",
            "-fPIC",
            "-O0",
            "-fno-stack-protector",
            "-fno-builtin", // force calls to strlen/strcmp rather than builtins
            "-fcf-protection=none", // standard .plt layout (no .plt.sec / IBT)
            "-D_FORTIFY_SOURCE=0",
            c_path.to_str().unwrap(),
            "-o",
            so_path.to_str().unwrap(),
            &map_arg,
        ])
        .status()
        .expect("failed to spawn gcc - is gcc in PATH?");
    assert!(status.success(), "gcc compilation failed");

    let elf_bytes = std::fs::read(&so_path).expect("failed to read compiled .so");
    let map = std::fs::read_to_string(&map_path).expect("failed to read map file");
    (elf_bytes, map)
}

/// Scan a GNU ld map file for the virtual address of `sym_name`.
///
/// The relevant line looks like:
/// ```text
///                 0x0000000000001119                process_string
/// ```
fn parse_map_symbol_va(map: &str, sym_name: &str) -> Option<u64> {
    for line in map.lines() {
        let trimmed = line.trim();
        if !trimmed.starts_with("0x") {
            continue;
        }
        let mut parts = trimmed.splitn(2, |c: char| c.is_ascii_whitespace());
        let addr_str = parts.next()?;
        let name = parts.next()?.trim();
        if name == sym_name {
            let hex = addr_str.trim_start_matches("0x");
            return u64::from_str_radix(hex, 16).ok();
        }
    }
    None
}

/// Look up `sym_name` in the ELF symbol tables and return its size in bytes.
///
/// Checks `.dynsym` first (exported symbols), then `.symtab`.
fn elf_symbol_size(elf_bytes: &[u8], sym_name: &str) -> Option<u64> {
    let file = object::File::parse(elf_bytes).ok()?;
    for sym in file.dynamic_symbols() {
        if sym.name() == Ok(sym_name) && sym.size() > 0 {
            return Some(sym.size());
        }
    }
    for sym in file.symbols() {
        if sym.name() == Ok(sym_name) && sym.size() > 0 {
            return Some(sym.size());
        }
    }
    None
}

/// Return the file offset inside `elf_bytes` that corresponds to virtual
/// address `va` in the ELF.
fn elf_va_to_file_offset(elf_bytes: &[u8], va: u64) -> Option<usize> {
    let file = object::File::parse(elf_bytes).ok()?;
    for section in file.sections() {
        let start = section.address();
        let size = section.size();
        if va >= start && va < start + size {
            let (file_off, _) = section.file_range()?;
            return Some(file_off as usize + (va - start) as usize);
        }
    }
    None
}

/// Find the PLT stub virtual address for the import named `import_name`.
///
/// Reads `.rela.plt` to obtain the relocation order, looks up each
/// relocation's symbol in `.dynsym`, and computes the stub address from
/// the `.plt` section base (with the standard x86-64 layout where entry 0
/// is the resolver and entries 1..n follow in `.rela.plt` order, each 16 B).
fn find_plt_entry_va(elf_bytes: &[u8], import_name: &str) -> Option<u64> {
    let file = object::File::parse(elf_bytes).ok()?;

    // Build index -> name map from .dynsym
    let dynsym: HashMap<usize, String> = file
        .dynamic_symbols()
        .filter_map(|sym| sym.name().ok().map(|n| (sym.index().0, n.to_string())))
        .collect();

    // Locate .rela.plt
    let rela_section = file.section_by_name(".rela.plt")?;
    let rela_data = rela_section.data().ok()?;

    // Find the PLT section.  With -fcf-protection=none there is no .plt.sec;
    // calls go directly to .plt, with entry 0 being the resolver stub and
    // entries 1..n corresponding to .rela.plt[0..n-1].
    let (plt_base, first_entry_offset): (u64, u64) =
        if let Some(sec) = file.section_by_name(".plt.sec") {
            // IBT variant: entries start at index 0, no resolver prefix
            (sec.address(), 0)
        } else {
            let plt = file.section_by_name(".plt")?;
            (plt.address(), 16) // skip the 16-byte resolver stub
        };

    // Parse Elf64_Rela entries: r_offset (8 B), r_info (8 B), r_addend (8 B)
    const RELA64_SIZE: usize = 24;
    for (idx, chunk) in rela_data.chunks_exact(RELA64_SIZE).enumerate() {
        let r_info = u64::from_le_bytes(chunk[8..16].try_into().unwrap());
        let sym_idx = (r_info >> 32) as usize;

        if dynsym.get(&sym_idx).map(|n| n.as_str()) == Some(import_name) {
            return Some(plt_base + first_entry_offset + (idx as u64) * 16);
        }
    }
    None
}

#[test]
fn test_elf_process_string() {
    if !(cfg!(target_os = "linux") && cfg!(target_arch = "x86_64")) {
        return;
    }

    let (elf_bytes, map) = compile_shared_lib();

    let func_va = parse_map_symbol_va(&map, "process_string")
        .expect("process_string not found in linker map file");

    let func_size = elf_symbol_size(&elf_bytes, "process_string")
        .expect("process_string symbol not found in ELF");

    let func_file_offset = elf_va_to_file_offset(&elf_bytes, func_va)
        .expect("failed to map function VA to ELF file offset");
    let func_bytes = &elf_bytes[func_file_offset..func_file_offset + func_size as usize];

    let strlen_plt_va =
        find_plt_entry_va(&elf_bytes, "strlen").expect("strlen PLT entry not found");
    let strcmp_plt_va =
        find_plt_entry_va(&elf_bytes, "strcmp").expect("strcmp PLT entry not found");

    let mut config = LifterConfig::default();
    config.register_slow_import(strlen_plt_va);
    config.register_slow_import(strcmp_plt_va);

    let mut func = Function::with_name_signature(
        UserFuncName::user(0, 0),
        Signature::new(get_host_calling_conv()),
    );
    func.signature.params.push(AbiParam::new(types::I64));

    let mut func_ctx = cranelift_frontend::FunctionBuilderContext::new();
    let builder = cranelift_frontend::FunctionBuilder::new(&mut func, &mut func_ctx);
    let mut lifter = Lifter::with_config(builder, config);
    lifter.lift(func_bytes, func_va).expect("lifting failed");
    drop(lifter);

    let mut flag_builder = settings::builder();
    flag_builder.set("opt_level", "speed").unwrap();
    let flags = settings::Flags::new(flag_builder);
    let mut backend = CraneliftBackend::with_flags(flags).expect("backend init failed");
    let code_ptr = backend
        .compile(func.clone())
        .expect("JIT compilation failed");
    let f: fn(*mut RegisterContext) = unsafe { std::mem::transmute(code_ptr) };

    let mut stack = vec![0u64; 256];
    let stack_top = stack.as_mut_ptr() as u64 + (stack.len() as u64) * 8;
    let initial_rsp = stack_top - 8;
    // Pre-push the exit trampoline so the first RET exits cleanly.
    unsafe { *(initial_rsp as *mut u64) = 0xFFFF_FFFF_FFFF_FFFF };

    let test_string = b"hello\0";
    let string_ptr = test_string.as_ptr() as u64;

    let mut ctx = RegisterContext::default();
    ctx.rsp = initial_rsp;
    ctx.rdi = string_ptr; // first argument: const char *s
    ctx.image_base = func_va;
    ctx.rip = func_va;

    eprintln!("strlen_plt_va={strlen_plt_va:#x}, strcmp_plt_va={strcmp_plt_va:#x}");
    eprintln!("func_va={func_va:#x}, func_size={func_size}");
    loop {
        f(&mut ctx);
        eprintln!(
            "exit: reason={}, rip={:#x}, imported_va={:#x}",
            ctx.exit_reason, ctx.rip, ctx.imported_function_va
        );

        if ctx.exit_reason == ExitReason::Return as u32 {
            break;
        }

        assert_eq!(
            ctx.exit_reason,
            ExitReason::ImportedFunctionCall as u32,
            "unexpected exit reason: {} (rip={:#x}, imported_va={:#x})",
            ctx.exit_reason,
            ctx.rip,
            ctx.imported_function_va
        );

        let va = ctx.imported_function_va;
        if va == strlen_plt_va {
            // strlen(rdi) -> rax
            let s = ctx.rdi as *const u8;
            ctx.rax = unsafe { host_strlen(s) };
        } else if va == strcmp_plt_va {
            // strcmp(rdi, rsi) -> rax  (sign-extend i32 result to u64)
            let s1 = ctx.rdi as *const u8;
            let s2 = ctx.rsi as *const u8;
            ctx.rax = unsafe { host_strcmp(s1, s2) } as i64 as u64;
        } else {
            panic!("unexpected ImportedFunctionCall VA: {va:#x}");
        }
    }

    // "hello" -> strlen=5 > 1 -> copy="iello\0" -> strcmp("hello","iello") < 0
    let result = ctx.rax as i32;
    assert!(
        result < 0,
        "expected strcmp(\"hello\", \"iello\") < 0, got {result}"
    );

    // Sanity-check: short string branch (len <= 1) returns 0 without any import.
    let short_string = b"x\0";
    ctx = RegisterContext::default();
    ctx.rsp = initial_rsp;
    unsafe { *(initial_rsp as *mut u64) = 0xFFFF_FFFF_FFFF_FFFF };
    ctx.rdi = short_string.as_ptr() as u64;
    ctx.image_base = func_va;
    ctx.rip = func_va;

    // This call needs strlen but len==1 so it returns early (no strcmp call).
    loop {
        f(&mut ctx);
        if ctx.exit_reason == ExitReason::Return as u32 {
            break;
        }
        assert_eq!(ctx.exit_reason, ExitReason::ImportedFunctionCall as u32);
        let va = ctx.imported_function_va;
        assert_eq!(va, strlen_plt_va, "only strlen expected for short string");
        ctx.rax = unsafe { host_strlen(ctx.rdi as *const u8) };
    }

    assert_eq!(ctx.rax as i32, 0, "process_string(\"x\") should return 0");
}
