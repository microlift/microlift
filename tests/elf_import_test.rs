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

const PROCESS_STRING_SOURCE: &str = r#"
unsigned long strlen(const char *s);
int strcmp(const char *a, const char *b);

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

const MEMCPY_SOURCE: &str = r#"
void *memcpy(void *dst, const void *src, unsigned long n);

/* Copy n bytes src->dst with memcpy, then fold the copied bytes (read back
 * through memcpy's returned pointer, which must equal dst) into a 64-bit
 * FNV-1a hash. Exercises the ImportedFunctionCall argument passing
 * (rdi=dst, rsi=src, rdx=n), memcpy's effect on caller memory, its return
 * value in rax, and callee-saved-register preservation across the slow call. */
unsigned long memcpy_and_checksum(unsigned char *dst, const unsigned char *src,
                                  unsigned long n) {
    unsigned char *ret = memcpy(dst, src, n);
    unsigned long h = 14695981039346656037UL; /* FNV-1a 64-bit offset basis */
    for (unsigned long i = 0; i < n; i++) {
        h = (h ^ (unsigned long)ret[i]) * 1099511628211UL; /* FNV prime */
    }
    return h;
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

unsafe extern "C" fn host_memcpy(dst: *mut u8, src: *const u8, n: u64) -> *mut u8 {
    unsafe { std::ptr::copy_nonoverlapping(src, dst, n as usize) };
    dst
}

fn fnv1a_64(data: &[u8]) -> u64 {
    let mut h: u64 = 14695981039346656037;
    for &b in data {
        h = (h ^ b as u64).wrapping_mul(1099511628211);
    }
    h
}

fn artifact_dir() -> PathBuf {
    let root = std::env::var("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."));
    root.join("test-artifacts").join("elf_import_test")
}

fn command_works(cmd: &str) -> bool {
    Command::new(cmd)
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn x86_64_elf_compiler() -> Option<String> {
    if let Ok(cc) = std::env::var("MICROLIFT_X86_64_ELF_CC") {
        return command_works(&cc).then_some(cc);
    }
    let mut candidates: Vec<&str> = Vec::new();
    if cfg!(target_os = "linux") && cfg!(target_arch = "x86_64") {
        candidates.extend(["gcc", "cc"]);
    }
    candidates.extend(["x86_64-unknown-linux-gnu-gcc", "x86_64-linux-gnu-gcc"]);
    candidates
        .into_iter()
        .find(|c| command_works(c))
        .map(String::from)
}

fn compile_x86_64_elf(source: &str, basename: &str) -> Option<Vec<u8>> {
    let cc = x86_64_elf_compiler()?;

    let dir = artifact_dir();
    std::fs::create_dir_all(&dir).expect("failed to create artifact dir");
    let c_path = dir.join(format!("{basename}.c"));
    let so_path = dir.join(format!("lib{basename}.so"));
    std::fs::write(&c_path, source).expect("failed to write C source");

    let status = Command::new(&cc)
        .args([
            "-shared",
            "-fPIC",
            "-O0",
            "-fno-stack-protector",
            "-fno-builtin",        // force real calls to strlen/strcmp/memcpy
            "-fcf-protection=none", // standard .plt layout (no .plt.sec / IBT)
            "-D_FORTIFY_SOURCE=0",
            "-Wl,-z,lazy",
            c_path.to_str().unwrap(),
            "-o",
            so_path.to_str().unwrap(),
        ])
        .status()
        .unwrap_or_else(|e| panic!("failed to spawn {cc}: {e}"));
    assert!(status.success(), "{cc} failed to compile {basename}");

    Some(std::fs::read(&so_path).expect("failed to read compiled .so"))
}

fn elf_symbol_va_size(elf_bytes: &[u8], sym_name: &str) -> Option<(u64, u64)> {
    let file = object::File::parse(elf_bytes).ok()?;
    for sym in file.dynamic_symbols().chain(file.symbols()) {
        if sym.name() == Ok(sym_name) && sym.size() > 0 {
            return Some((sym.address(), sym.size()));
        }
    }
    None
}

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

fn find_plt_entry_va(elf_bytes: &[u8], import_name: &str) -> Option<u64> {
    let file = object::File::parse(elf_bytes).ok()?;

    let dynsym: HashMap<usize, String> = file
        .dynamic_symbols()
        .filter_map(|sym| sym.name().ok().map(|n| (sym.index().0, n.to_string())))
        .collect();

    let rela_section = file.section_by_name(".rela.plt")?;
    let rela_data = rela_section.data().ok()?;

    let (plt_base, first_entry_offset): (u64, u64) =
        if let Some(sec) = file.section_by_name(".plt.sec") {
            (sec.address(), 0)
        } else {
            let plt = file.section_by_name(".plt")?;
            (plt.address(), 16)
        };

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

const EXIT_TRAMPOLINE: u64 = 0xFFFF_FFFF_FFFF_FFFF;

fn fresh_stack(stack: &mut [u64]) -> u64 {
    let top = stack.as_mut_ptr() as u64 + (stack.len() as u64) * 8;
    let rsp = top - 8;
    unsafe { *(rsp as *mut u64) = EXIT_TRAMPOLINE };
    rsp
}

fn run_guest_with_slow_imports(
    func_bytes: &[u8],
    func_va: u64,
    slow_imports: &[u64],
    mut ctx: RegisterContext,
    mut service: impl FnMut(u64, &mut RegisterContext),
) -> RegisterContext {
    let mut config = LifterConfig::default();
    for &va in slow_imports {
        config.register_slow_import(va);
    }

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
    let mut backend =
        CraneliftBackend::with_flags(settings::Flags::new(flag_builder)).expect("backend init");
    let code_ptr = backend.compile(func).expect("JIT compilation failed");
    let f: fn(*mut RegisterContext) = unsafe { std::mem::transmute(code_ptr) };

    loop {
        f(&mut ctx);
        if ctx.exit_reason == ExitReason::Return as u32 {
            break;
        }
        assert_eq!(
            ctx.exit_reason,
            ExitReason::ImportedFunctionCall as u32,
            "unexpected exit reason {} (rip={:#x}, imported_va={:#x})",
            ctx.exit_reason,
            ctx.rip,
            ctx.imported_function_va,
        );
        let va = ctx.imported_function_va;
        service(va, &mut ctx);
    }
    ctx
}

#[test]
fn test_elf_process_string() {
    let Some(elf) = compile_x86_64_elf(PROCESS_STRING_SOURCE, "process_string") else {
        eprintln!("skipping test_elf_process_string: no x86-64 ELF compiler on PATH");
        return;
    };

    let (func_va, func_size) =
        elf_symbol_va_size(&elf, "process_string").expect("process_string symbol");
    let off = elf_va_to_file_offset(&elf, func_va).expect("VA->file offset");
    let func_bytes = &elf[off..off + func_size as usize];

    let strlen_plt = find_plt_entry_va(&elf, "strlen").expect("strlen PLT entry");
    let strcmp_plt = find_plt_entry_va(&elf, "strcmp").expect("strcmp PLT entry");
    let imports = [strlen_plt, strcmp_plt];

    let service = |va: u64, ctx: &mut RegisterContext| {
        if va == strlen_plt {
            ctx.rax = unsafe { host_strlen(ctx.rdi as *const u8) };
        } else if va == strcmp_plt {
            ctx.rax =
                unsafe { host_strcmp(ctx.rdi as *const u8, ctx.rsi as *const u8) } as i64 as u64;
        } else {
            panic!("unexpected ImportedFunctionCall VA: {va:#x}");
        }
    };

    // "hello" -> len 5 -> copy "iello" -> strcmp("hello","iello") < 0
    let test_string = b"hello\0";
    let mut stack = vec![0u64; 256];
    let rsp = fresh_stack(&mut stack);
    let mut ctx = RegisterContext::default();
    ctx.rsp = rsp;
    ctx.rdi = test_string.as_ptr() as u64;
    ctx.image_base = func_va;
    ctx.rip = func_va;
    let out = run_guest_with_slow_imports(func_bytes, func_va, &imports, ctx, service);
    assert!(
        (out.rax as i32) < 0,
        "expected strcmp(\"hello\",\"iello\") < 0, got {}",
        out.rax as i32
    );
    let _ = &stack;

    // "x" -> len 1 -> early return 0, only strlen called.
    let short = b"x\0";
    let mut stack2 = vec![0u64; 256];
    let rsp2 = fresh_stack(&mut stack2);
    let mut ctx2 = RegisterContext::default();
    ctx2.rsp = rsp2;
    ctx2.rdi = short.as_ptr() as u64;
    ctx2.image_base = func_va;
    ctx2.rip = func_va;
    let out2 = run_guest_with_slow_imports(func_bytes, func_va, &imports, ctx2, |va, ctx| {
        assert_eq!(va, strlen_plt, "only strlen expected for the short string");
        ctx.rax = unsafe { host_strlen(ctx.rdi as *const u8) };
    });
    assert_eq!(out2.rax as i32, 0, "process_string(\"x\") should return 0");
    let _ = &stack2;
}

#[test]
fn test_elf_memcpy_semantics() {
    let Some(elf) = compile_x86_64_elf(MEMCPY_SOURCE, "memcpy_checksum") else {
        eprintln!("skipping test_elf_memcpy_semantics: no x86-64 ELF compiler on PATH");
        return;
    };

    let (func_va, func_size) =
        elf_symbol_va_size(&elf, "memcpy_and_checksum").expect("memcpy_and_checksum symbol");
    let off = elf_va_to_file_offset(&elf, func_va).expect("VA->file offset");
    let func_bytes = &elf[off..off + func_size as usize];

    let memcpy_plt = find_plt_entry_va(&elf, "memcpy").expect("memcpy PLT entry");

    let data: [u8; 11] = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xAA, 0xBB];
    let mut dst = [0u8; 11];
    assert_ne!(dst, data, "destination must start different from source");

    let mut stack = vec![0u64; 256];
    let rsp = fresh_stack(&mut stack);

    let mut ctx = RegisterContext::default();
    ctx.rsp = rsp;
    ctx.rdi = dst.as_mut_ptr() as u64; // dst
    ctx.rsi = data.as_ptr() as u64; // src
    ctx.rdx = data.len() as u64; // n
    ctx.image_base = func_va;
    ctx.rip = func_va;

    let mut memcpy_calls = 0usize;
    let mut seen_args: Option<(u64, u64, u64)> = None;
    let out = run_guest_with_slow_imports(func_bytes, func_va, &[memcpy_plt], ctx, |va, ctx| {
        assert_eq!(va, memcpy_plt, "only memcpy should be imported");
        memcpy_calls += 1;
        seen_args = Some((ctx.rdi, ctx.rsi, ctx.rdx));
        ctx.rax = unsafe { host_memcpy(ctx.rdi as *mut u8, ctx.rsi as *const u8, ctx.rdx) } as u64;
    });

    assert_eq!(memcpy_calls, 1, "memcpy must be called exactly once");
    assert_eq!(
        seen_args,
        Some((dst.as_ptr() as u64, data.as_ptr() as u64, data.len() as u64)),
        "memcpy received the wrong (rdi, rsi, rdx) arguments"
    );

    assert_eq!(dst, data, "memcpy did not copy the source bytes into dst");

    assert_eq!(
        out.rax,
        fnv1a_64(&data),
        "checksum mismatch: return value / callee-saved state not preserved across the call"
    );

    let _ = &stack;
}
