//! Real-compiler ELF functions driven through the testkit's import-trampoline
//! system.
//!
//! Each case compiles a small C function to an x86-64 shared object with a real
//! toolchain, lifts the function's bytes straight out of the `.text`, and traps
//! its PLT entries back to host implementations of `strlen`/`strcmp`/`memcpy`
//! through a [`ImportTable`]. That table is engine-neutral: the *same* handler
//! services Microlift's `ImportedFunctionCall` VM exits and Unicorn's stub
//! hooks, so under `--features unicorn-oracle` every case additionally asserts
//! the two engines end in the same architectural state.
//!
//! Run: `cargo test --features testkit --test elf_import_test`
//!      `cargo test --features unicorn-oracle --test elf_import_test`
//!
//! Skipped (with a note) when no x86-64 ELF compiler is on `PATH`; set
//! `MICROLIFT_X86_64_ELF_CC` to point at one.

#![cfg(feature = "testkit")]

use microlift::testkit::oracle::*;
use microlift::testkit::trampoline::*;
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

/// Instruction budget for the emulated (Unicorn) side; the JIT ignores it.
const BUDGET: u64 = 1 << 20;

/// Both engines run on one shared [`HostStack`], so `rsp` itself is comparable.
#[cfg(feature = "unicorn-oracle")]
const DIFF: DiffOptions = DiffOptions {
    compare_rsp: true,
    compare_rip: false,
};

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
            "-fno-builtin",         // force real calls to strlen/strcmp/memcpy
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

/// A lifted function: its bytes, the VA they were loaded at, and the PLT entries
/// it calls.
struct ElfFunction {
    bytes: Vec<u8>,
    va: u64,
}

fn load_function(source: &str, basename: &str, symbol: &str) -> Option<(ElfFunction, Vec<u8>)> {
    let elf = compile_x86_64_elf(source, basename)?;
    let (va, size) = elf_symbol_va_size(&elf, symbol).expect("function symbol");
    let off = elf_va_to_file_offset(&elf, va).expect("VA->file offset");
    let bytes = elf[off..off + size as usize].to_vec();
    Some((ElfFunction { bytes, va }, elf))
}

/// Run `f` on the Microlift JIT with `imports` trapped to `handler`, asserting a
/// clean halt.
fn run_microlift(
    f: &ElfFunction,
    init: &CpuState,
    imports: &[u64],
    handler: &mut dyn FnMut(u64, &mut CpuState),
) -> CpuState {
    let mut table = ImportTable::new(imports.iter().copied(), handler);
    let out = MicroliftOracle::default().run_with_imports(
        &f.bytes,
        f.va,
        init,
        &[],
        BUDGET,
        &mut table,
    );
    assert!(
        out.is_halted(),
        "microlift did not reach a clean halt: {out:?}"
    );
    *out.state()
}

/// Same program under Unicorn, through the stub-page trampolines.
#[cfg(feature = "unicorn-oracle")]
fn run_unicorn(
    f: &ElfFunction,
    init: &CpuState,
    imports: &[u64],
    handler: &mut dyn FnMut(u64, &mut CpuState),
) -> CpuState {
    use microlift::testkit::oracle::unicorn::UnicornOracle;

    let mut table = ImportTable::new(imports.iter().copied(), handler);
    let out =
        UnicornOracle::new().run_with_imports(&f.bytes, f.va, init, &[], BUDGET, &mut table);
    assert!(
        out.is_halted(),
        "unicorn did not reach a clean halt: {out:?}"
    );
    *out.state()
}

#[test]
fn test_elf_process_string() {
    let Some((func, elf)) = load_function(PROCESS_STRING_SOURCE, "process_string", "process_string")
    else {
        eprintln!("skipping test_elf_process_string: no x86-64 ELF compiler on PATH");
        return;
    };

    let strlen_plt = find_plt_entry_va(&elf, "strlen").expect("strlen PLT entry");
    let strcmp_plt = find_plt_entry_va(&elf, "strcmp").expect("strcmp PLT entry");
    let imports = [strlen_plt, strcmp_plt];

    // The handler is a pure function of the guest state, so both engines can be
    // handed a freshly built copy of it.
    let make_handler = || {
        move |va: u64, st: &mut CpuState| {
            if va == strlen_plt {
                st.gpr[gpr::RAX] = unsafe { host_strlen(st.gpr[gpr::RDI] as *const u8) };
            } else if va == strcmp_plt {
                st.gpr[gpr::RAX] = unsafe {
                    host_strcmp(st.gpr[gpr::RDI] as *const u8, st.gpr[gpr::RSI] as *const u8)
                } as i64 as u64;
            } else {
                panic!("unexpected imported call VA: {va:#x}");
            }
        }
    };

    // "hello" -> len 5 -> copy "iello" -> strcmp("hello","iello") < 0.
    let test_string = b"hello\0";
    let stack = HostStack::new(1024);
    let mut init = CpuState::zeroed();
    init.gpr[gpr::RSP] = stack.rsp();
    init.gpr[gpr::RDI] = test_string.as_ptr() as u64;

    let ml = run_microlift(&func, &init, &imports, &mut make_handler());
    assert!(
        (ml.gpr[gpr::RAX] as i32) < 0,
        "expected strcmp(\"hello\",\"iello\") < 0, got {}",
        ml.gpr[gpr::RAX] as i32
    );

    #[cfg(feature = "unicorn-oracle")]
    {
        let uc = run_unicorn(&func, &init, &imports, &mut make_handler());
        let diff = diff_states(&ml, &uc, DIFF);
        assert!(diff.is_empty(), "process_string diverged: {diff:?}");
    }

    // "x" -> len 1 -> early return 0; only strlen is called, on both engines.
    let short = b"x\0";
    let stack2 = HostStack::new(1024);
    let mut init2 = CpuState::zeroed();
    init2.gpr[gpr::RSP] = stack2.rsp();
    init2.gpr[gpr::RDI] = short.as_ptr() as u64;

    let mut only_strlen = |va: u64, st: &mut CpuState| {
        assert_eq!(va, strlen_plt, "only strlen expected for the short string");
        st.gpr[gpr::RAX] = unsafe { host_strlen(st.gpr[gpr::RDI] as *const u8) };
    };
    let ml2 = run_microlift(&func, &init2, &imports, &mut only_strlen);
    assert_eq!(
        ml2.gpr[gpr::RAX] as i32,
        0,
        "process_string(\"x\") should return 0"
    );

    #[cfg(feature = "unicorn-oracle")]
    {
        let uc2 = run_unicorn(&func, &init2, &imports, &mut only_strlen);
        let diff = diff_states(&ml2, &uc2, DIFF);
        assert!(diff.is_empty(), "process_string(\"x\") diverged: {diff:?}");
    }
}

#[test]
fn test_elf_memcpy_semantics() {
    let Some((func, elf)) = load_function(MEMCPY_SOURCE, "memcpy_checksum", "memcpy_and_checksum")
    else {
        eprintln!("skipping test_elf_memcpy_semantics: no x86-64 ELF compiler on PATH");
        return;
    };

    let memcpy_plt = find_plt_entry_va(&elf, "memcpy").expect("memcpy PLT entry");

    let data: [u8; 11] = [
        0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xAA, 0xBB,
    ];
    let mut dst = [0u8; 11];
    assert_ne!(dst, data, "destination must start different from source");

    let stack = HostStack::new(1024);
    let mut init = CpuState::zeroed();
    init.gpr[gpr::RSP] = stack.rsp();
    init.gpr[gpr::RDI] = dst.as_mut_ptr() as u64; // dst
    init.gpr[gpr::RSI] = data.as_ptr() as u64; // src
    init.gpr[gpr::RDX] = data.len() as u64; // n

    let expected_args = (
        dst.as_ptr() as u64,
        data.as_ptr() as u64,
        data.len() as u64,
    );

    // Per-engine bookkeeping: each run records its own calls, so "memcpy is
    // called exactly once" is asserted for each engine independently.
    let mut ml_calls: Vec<(u64, u64, u64)> = Vec::new();
    let handler = |va: u64, st: &mut CpuState| {
        assert_eq!(va, memcpy_plt, "only memcpy should be imported");
        st.gpr[gpr::RAX] = unsafe {
            host_memcpy(
                st.gpr[gpr::RDI] as *mut u8,
                st.gpr[gpr::RSI] as *const u8,
                st.gpr[gpr::RDX],
            )
        } as u64;
    };

    let ml = {
        let mut recording = |va: u64, st: &mut CpuState| {
            ml_calls.push((st.gpr[gpr::RDI], st.gpr[gpr::RSI], st.gpr[gpr::RDX]));
            handler(va, st);
        };
        run_microlift(&func, &init, &[memcpy_plt], &mut recording)
    };

    assert_eq!(ml_calls.len(), 1, "memcpy must be called exactly once");
    assert_eq!(
        ml_calls[0], expected_args,
        "memcpy received the wrong (rdi, rsi, rdx) arguments"
    );
    assert_eq!(dst, data, "memcpy did not copy the source bytes into dst");
    assert_eq!(
        ml.gpr[gpr::RAX],
        fnv1a_64(&data),
        "checksum mismatch: return value / callee-saved state not preserved across the call"
    );

    #[cfg(feature = "unicorn-oracle")]
    {
        // Clear the destination so the emulated run has to re-establish it
        // through its own trampoline rather than reading Microlift's leftovers.
        dst = [0u8; 11];

        let mut uc_calls: Vec<(u64, u64, u64)> = Vec::new();
        let uc = {
            let mut recording = |va: u64, st: &mut CpuState| {
                uc_calls.push((st.gpr[gpr::RDI], st.gpr[gpr::RSI], st.gpr[gpr::RDX]));
                handler(va, st);
            };
            run_unicorn(&func, &init, &[memcpy_plt], &mut recording)
        };

        assert_eq!(uc_calls, ml_calls, "engines disagree on the import calls");
        assert_eq!(dst, data, "unicorn run did not repopulate dst");

        let diff = diff_states(&ml, &uc, DIFF);
        assert!(diff.is_empty(), "memcpy_and_checksum diverged: {diff:?}");
    }
}

/// The one-line form: [`differential_with_imports`] runs both engines from one
/// seed with one handler factory and reports their state difference directly.
#[cfg(feature = "unicorn-oracle")]
#[test]
fn test_elf_process_string_differential_helper() {
    let Some((func, elf)) = load_function(PROCESS_STRING_SOURCE, "process_string", "process_string")
    else {
        eprintln!("skipping differential helper test: no x86-64 ELF compiler on PATH");
        return;
    };

    let strlen_plt = find_plt_entry_va(&elf, "strlen").expect("strlen PLT entry");
    let strcmp_plt = find_plt_entry_va(&elf, "strcmp").expect("strcmp PLT entry");

    let test_string = b"microlift\0";
    let stack = HostStack::new(1024);
    let mut init = CpuState::zeroed();
    init.gpr[gpr::RSP] = stack.rsp();
    init.gpr[gpr::RDI] = test_string.as_ptr() as u64;

    let diff = differential_with_imports(
        &func.bytes,
        func.va,
        &init,
        BUDGET,
        DIFF,
        &[strlen_plt, strcmp_plt],
        || {
            move |va: u64, st: &mut CpuState| {
                if va == strlen_plt {
                    st.gpr[gpr::RAX] = unsafe { host_strlen(st.gpr[gpr::RDI] as *const u8) };
                } else {
                    st.gpr[gpr::RAX] = unsafe {
                        host_strcmp(st.gpr[gpr::RDI] as *const u8, st.gpr[gpr::RSI] as *const u8)
                    } as i64 as u64;
                }
            }
        },
    )
    .expect("both engines should reach a clean halt");

    assert!(diff.is_empty(), "microlift vs unicorn diverged: {diff:?}");
}
