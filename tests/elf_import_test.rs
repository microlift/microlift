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
use object::{Object, ObjectSection, ObjectSegment, ObjectSymbol};
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
    compile_x86_64_elf_with(source, basename, &["-O0"])
}

/// As [`compile_x86_64_elf`], with extra compiler/linker flags appended (opt
/// level, load address, ...).
fn compile_x86_64_elf_with(source: &str, basename: &str, extra: &[&str]) -> Option<Vec<u8>> {
    compile_x86_64_elf_multi(&[("main", source)], basename, extra)
}

/// Serializes the compile-and-read below.
///
/// `basename` names a file in a directory shared by the whole binary, and two
/// tests here pass the same one (`process_string`). Run in parallel, one test's
/// `gcc` truncates the `.so` while the other is reading it, and the reader gets
/// a valid-but-wrong or half-written image - which surfaced as an intermittent
/// `expect("function symbol")`, the symbol genuinely absent from the bytes that
/// were read. Unique per-call paths would work too, at the cost of leaving a
/// fresh artifact behind on every run; the compile is not on any hot path, so
/// serializing is the cheaper answer.
static COMPILE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Compile several translation units into one shared object. Link order is the
/// given order, which is what decides `.text` layout - the only way to pin a
/// specific function at the front of the image, since GCC emits callees before
/// their callers within a single unit.
fn compile_x86_64_elf_multi(
    sources: &[(&str, &str)],
    basename: &str,
    extra: &[&str],
) -> Option<Vec<u8>> {
    let cc = x86_64_elf_compiler()?;
    let _guard = COMPILE_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let dir = artifact_dir();
    std::fs::create_dir_all(&dir).expect("failed to create artifact dir");
    let so_path = dir.join(format!("lib{basename}.so"));

    let mut c_paths = Vec::new();
    for (unit, src) in sources {
        let c_path = dir.join(format!("{basename}_{unit}.c"));
        std::fs::write(&c_path, src).expect("failed to write C source");
        c_paths.push(c_path);
    }

    let mut cmd = Command::new(&cc);
    cmd.args([
        "-shared",
        "-fPIC",
        "-fno-stack-protector",
        "-fno-builtin",         // force real calls to strlen/strcmp/memcpy
        "-fcf-protection=none", // standard .plt layout (no .plt.sec / IBT)
        "-D_FORTIFY_SOURCE=0",
        "-Wl,-z,lazy",
    ])
    .args(extra)
    .args(c_paths.iter().map(|p| p.to_str().unwrap()))
    .args(["-o", so_path.to_str().unwrap()]);

    let status = cmd
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
    let out =
        MicroliftOracle::default().run_with_imports(&f.bytes, f.va, init, &[], BUDGET, &mut table);
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
    let out = UnicornOracle::new().run_with_imports(&f.bytes, f.va, init, &[], BUDGET, &mut table);
    assert!(
        out.is_halted(),
        "unicorn did not reach a clean halt: {out:?}"
    );
    *out.state()
}

#[test]
fn test_elf_process_string() {
    let Some((func, elf)) =
        load_function(PROCESS_STRING_SOURCE, "process_string", "process_string")
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

    let expected_args = (dst.as_ptr() as u64, data.as_ptr() as u64, data.len() as u64);

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
    let Some((func, elf)) =
        load_function(PROCESS_STRING_SOURCE, "process_string", "process_string")
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

// The guest is a whole loaded ELF image rather than a single function. It is
// linked at a high, mappable base and identity-mapped into the host address
// space, so guest VA == host VA holds for its code, its `.rodata` strings and
// its data - which is what lets `puts` be serviced by simply reading the host
// pointer the guest passed.
//
// The program checks its own `rsp` across every call shape Microlift has to get
// right (uninlined guest leaf calls, nested and recursive guest->guest calls,
// imported calls, and imported calls made from inside a guest callee holding a
// large frame), reports each violation through `puts`, and folds the failures
// into its return value. A passing run is `rax == 0` with no `puts` output.

/// The self-test compiles to a fixed path and maps its image at a fixed VA, so
/// only one of these tests may be in flight at a time - `cargo test` runs the
/// cases in a file concurrently. Poisoning is ignored: a panicking test has
/// already dropped its mapping, and cascading failures only hide the real one.
static SELFTEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn selftest_guard() -> std::sync::MutexGuard<'static, ()> {
    SELFTEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

const SELFTEST_LOAD_BASE: u64 = 0x3_0000_0000;

const SELFTEST_PRELUDE: &str = r#"
int puts(const char *s);
unsigned long strlen(const char *s);
void *memcpy(void *dst, const void *src, unsigned long n);

/* Read the live stack pointer. `volatile` keeps GCC from moving it across the
 * calls whose stack discipline we are measuring. */
static inline unsigned long read_sp(void) {
    unsigned long sp;
    __asm__ volatile("movq %%rsp, %0" : "=r"(sp));
    return sp;
}

/* Failure bits, OR-ed into the return value. */
#define F_LEAF_SP        0x0001
#define F_LEAF_VALUE     0x0002
#define F_NESTED_SP      0x0004
#define F_NESTED_VALUE   0x0008
#define F_IMPORT_SP      0x0010
#define F_IMPORT_VALUE   0x0020
#define F_MIXED_SP       0x0040
#define F_MIXED_VALUE    0x0080
#define F_RECURSE_SP     0x0100
#define F_RECURSE_VALUE  0x0200
#define F_FRAME_SP       0x0400
#define F_FRAME_VALUE    0x0800
#define F_SAVED_REGS     0x1000
#define F_ENTRY_SP       0x2000

#define FNV_BASIS 14695981039346656037UL
#define FNV_PRIME 1099511628211UL

/* Sentinels returned instead of a real result when a callee catches its own
 * stack-pointer violation, so the caller's value check trips too. */
#define BAD_LONG  (-999999L)
#define BAD_ULONG (~0UL)

long ml_leaf_add(long a, long b);
long ml_nested3(long x);
unsigned long ml_calls_import(const char *s);
unsigned long ml_mixed(const char *s);
long ml_recurse(long n);
unsigned long ml_big_frame(const unsigned char *src, unsigned long n);
int ml_import_preserves_regs(const char *s);

"#;

/// The entry translation unit. Linked first so it owns the lowest `.text`
/// addresses: the lifted region starts at this function.
const SELFTEST_ENTRY: &str = r#"/* ENTRY POINT. Lives in its own translation unit, linked first, so it owns the
 * lowest `.text` address of the program: the lifted region starts here and runs
 * to the end of `.text`, which is what puts every guest callee in region.
 *
 *   rdi = NUL-terminated string, rsi = blob, rdx = blob length.
 * Returns 0 when every check passed, otherwise the OR of the failure bits. */
int stack_discipline_main(const char *s, const unsigned char *blob, unsigned long n) {
    unsigned long entry_sp = read_sp();
    unsigned long sp;
    int fails = 0;

    /* 1. A forcefully uninlined guest leaf call. */
    sp = read_sp();
    if (ml_leaf_add(3, 4) != 7) {
        puts("FAIL: ml_leaf_add(3,4) did not return 7");
        fails |= F_LEAF_VALUE;
    }
    if (read_sp() != sp) {
        puts("FAIL: rsp not restored across the guest leaf call");
        fails |= F_LEAF_SP;
    }

    /* 2. Three levels of guest->guest calls, each checking its own frame. */
    sp = read_sp();
    if (ml_nested3(10) != 16) {
        puts("FAIL: ml_nested3(10) did not return 16");
        fails |= F_NESTED_VALUE;
    }
    if (read_sp() != sp) {
        puts("FAIL: rsp not restored across the nested guest calls");
        fails |= F_NESTED_SP;
    }

    /* 3. A direct imported (external) call from the entry frame. */
    sp = read_sp();
    if (strlen(s) != 9) {
        puts("FAIL: strlen(s) did not return 9");
        fails |= F_IMPORT_VALUE;
    }
    if (read_sp() != sp) {
        puts("FAIL: rsp not restored across the imported call");
        fails |= F_IMPORT_SP;
    }

    /* 4. Guest call whose callee itself calls imports. */
    sp = read_sp();
    if (ml_mixed(s) != 9 * 2 + 8) {
        puts("FAIL: ml_mixed did not return the expected length mix");
        fails |= F_MIXED_VALUE;
    }
    if (read_sp() != sp) {
        puts("FAIL: rsp not restored across the mixed guest/import call");
        fails |= F_MIXED_SP;
    }

    /* 5. Recursion: eight frames pushed and popped. */
    sp = read_sp();
    if (ml_recurse(8) != 36) {
        puts("FAIL: ml_recurse(8) did not return 36");
        fails |= F_RECURSE_VALUE;
    }
    if (read_sp() != sp) {
        puts("FAIL: rsp not restored across the recursive guest calls");
        fails |= F_RECURSE_SP;
    }

    /* 6. An imported call made under a large live stack frame. */
    {
        unsigned long h = FNV_BASIS;
        unsigned long i;
        for (i = 0; i < n; i++) {
            h = (h ^ (unsigned long)blob[i]) * FNV_PRIME;
        }
        sp = read_sp();
        if (ml_big_frame(blob, n) != h) {
            puts("FAIL: ml_big_frame checksum mismatch");
            fails |= F_FRAME_VALUE;
        }
        if (read_sp() != sp) {
            puts("FAIL: rsp not restored across the large-frame callee");
            fails |= F_FRAME_SP;
        }
    }

    /* 7. Callee-saved registers must survive an imported call. */
    if (ml_import_preserves_regs(s) != 0) {
        fails |= F_SAVED_REGS;
    }

    /* 8. ...and the entry frame must be exactly where it started. */
    if (read_sp() != entry_sp) {
        puts("FAIL: rsp drifted over the whole test body");
        fails |= F_ENTRY_SP;
    }

    return fails;
}

"#;

/// The callees, linked after the entry.
const SELFTEST_HELPERS: &str = r#"__attribute__((noinline)) long ml_leaf_add(long a, long b) {
    return a + b;
}

__attribute__((noinline)) long ml_nested1(long x) {
    unsigned long sp = read_sp();
    long r = ml_leaf_add(x, 1);
    if (read_sp() != sp) {
        puts("FAIL: rsp not restored inside ml_nested1");
        return BAD_LONG;
    }
    return r;
}

__attribute__((noinline)) long ml_nested2(long x) {
    unsigned long sp = read_sp();
    long r = ml_nested1(x) + 2;
    if (read_sp() != sp) {
        puts("FAIL: rsp not restored inside ml_nested2");
        return BAD_LONG;
    }
    return r;
}

__attribute__((noinline)) long ml_nested3(long x) {
    unsigned long sp = read_sp();
    long r = ml_nested2(x) + 3;
    if (read_sp() != sp) {
        puts("FAIL: rsp not restored inside ml_nested3");
        return BAD_LONG;
    }
    return r;
}

__attribute__((noinline)) unsigned long ml_calls_import(const char *s) {
    unsigned long sp = read_sp();
    unsigned long len = strlen(s);
    if (read_sp() != sp) {
        puts("FAIL: rsp not restored inside ml_calls_import");
        return BAD_ULONG;
    }
    return len;
}

/* Guest -> guest -> import, twice, with a guest leaf call in between. */
__attribute__((noinline)) unsigned long ml_mixed(const char *s) {
    unsigned long sp = read_sp();
    unsigned long a = ml_calls_import(s);
    long mid = ml_leaf_add(3, 5);
    unsigned long b = ml_calls_import(s);
    if (read_sp() != sp) {
        puts("FAIL: rsp not restored inside ml_mixed");
        return BAD_ULONG;
    }
    return a + b + (unsigned long)mid;
}

__attribute__((noinline)) long ml_recurse(long n) {
    unsigned long sp;
    long r;
    if (n <= 0) {
        return 0;
    }
    sp = read_sp();
    r = ml_recurse(n - 1) + n;
    if (read_sp() != sp) {
        puts("FAIL: rsp not restored inside ml_recurse");
        return BAD_LONG;
    }
    return r;
}

/* A 1 KiB frame that stays live across an imported `memcpy`. */
__attribute__((noinline)) unsigned long ml_big_frame(const unsigned char *src,
                                                     unsigned long n) {
    unsigned char buf[1024];
    unsigned long sp, h, i;

    if (n > sizeof(buf)) {
        puts("FAIL: ml_big_frame called with an oversized length");
        return BAD_ULONG;
    }
    sp = read_sp();
    memcpy(buf, src, n);
    if (read_sp() != sp) {
        puts("FAIL: rsp not restored across memcpy under a large frame");
        return BAD_ULONG;
    }
    h = FNV_BASIS;
    for (i = 0; i < n; i++) {
        h = (h ^ (unsigned long)buf[i]) * FNV_PRIME;
    }
    return h;
}

/* Load sentinels into the callee-saved GPRs, call an import, and verify the
 * import path gave them all back. Written as one asm block so the values are
 * provably in registers rather than spilled by the compiler. */
__attribute__((noinline)) int ml_import_preserves_regs(const char *s) {
    unsigned long bad = 0;
    __asm__ volatile(
        "movabsq $0x1111111111111111, %%r12\n\t"
        "movabsq $0x2222222222222222, %%r13\n\t"
        "movabsq $0x3333333333333333, %%r14\n\t"
        "movabsq $0x4444444444444444, %%r15\n\t"
        "movq %[str], %%rdi\n\t"
        "call strlen@PLT\n\t"
        "movabsq $0x1111111111111111, %%rax\n\t"
        "xorq %%rax, %%r12\n\t"
        "movabsq $0x2222222222222222, %%rax\n\t"
        "xorq %%rax, %%r13\n\t"
        "movabsq $0x3333333333333333, %%rax\n\t"
        "xorq %%rax, %%r14\n\t"
        "movabsq $0x4444444444444444, %%rax\n\t"
        "xorq %%rax, %%r15\n\t"
        "orq %%r12, %[bad]\n\t"
        "orq %%r13, %[bad]\n\t"
        "orq %%r14, %[bad]\n\t"
        "orq %%r15, %[bad]\n\t"
        : [bad] "+m"(bad)
        : [str] "r"(s)
        : "rax", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12",
          "r13", "r14", "r15", "cc", "memory");
    if (bad != 0) {
        puts("FAIL: callee-saved registers clobbered across an imported call");
        return 1;
    }
    return 0;
}
"#;

/// A whole ELF image mapped so that guest VA == host VA, plus the pieces of it
/// the test drives: the entry function's byte range and its PLT entries.
struct LoadedImage {
    /// Keeps the identity mapping alive; dropping it unmaps the guest.
    _arena: IdentityArena,
    entry: ElfFunction,
    puts_plt: u64,
    strlen_plt: u64,
    memcpy_plt: u64,
}

fn page_align_down(v: u64) -> u64 {
    v & !0xFFF
}

fn page_align_up(v: u64) -> u64 {
    (v + 0xFFF) & !0xFFF
}

/// Compile the self-test, map every `PT_LOAD` at its link-time address, and
/// return the lifted region: the entry function through the end of `.text`, so
/// every guest->guest callee is inside the lifted bytes.
fn load_stack_discipline_image() -> Option<LoadedImage> {
    let entry_unit = format!("{SELFTEST_PRELUDE}{SELFTEST_ENTRY}");
    let helpers_unit = format!("{SELFTEST_PRELUDE}{SELFTEST_HELPERS}");
    let elf = compile_x86_64_elf_multi(
        &[
            ("entry", entry_unit.as_str()),
            ("helpers", helpers_unit.as_str()),
        ],
        "stack_discipline",
        &[
            "-O2",
            // Keep every call a real call: no tail-call jumps to the PLT (those
            // would leave the region as a jump rather than an import call).
            "-fno-optimize-sibling-calls",
            // Guest->guest calls must be *direct*. Default visibility in a
            // shared object makes them interposable, so GCC routes even
            // intra-image calls through the PLT - which would leave the lifted
            // region on every call.
            "-fvisibility=hidden",
            "-fno-semantic-interposition",
            &format!("-Wl,-Ttext-segment={SELFTEST_LOAD_BASE:#x}"),
            // Compact the image so the identity mapping is a few pages, not the
            // default 2 MiB-aligned segments.
            "-Wl,-z,max-page-size=0x1000",
        ],
    )?;

    let file = object::File::parse(&*elf).expect("parse self-test ELF");

    let mut lo = u64::MAX;
    let mut hi = 0u64;
    for seg in file.segments() {
        if seg.size() == 0 {
            continue;
        }
        lo = lo.min(page_align_down(seg.address()));
        hi = hi.max(page_align_up(seg.address() + seg.size()));
    }
    assert!(lo < hi, "self-test ELF has no loadable segments");
    assert!(
        lo >= SELFTEST_LOAD_BASE,
        "self-test image landed at {lo:#x}, below the requested base \
         {SELFTEST_LOAD_BASE:#x}; the -Ttext-segment flag did not take"
    );

    let arena = IdentityArena::map(lo, (hi - lo) as usize)?;
    let mut arena = arena;
    for seg in file.segments() {
        let data = seg.data().expect("segment data");
        if data.is_empty() {
            continue;
        }
        arena.write_at((seg.address() - lo) as usize, data);
    }

    let (entry_va, _) =
        elf_symbol_va_size(&elf, "stack_discipline_main").expect("stack_discipline_main symbol");

    // The lifted region has to contain every guest callee, and the entry must
    // be its first byte - so the entry function must be the lowest-addressed
    // one. GCC emits in definition order, which the C source relies on.
    for name in [
        "ml_leaf_add",
        "ml_nested1",
        "ml_nested2",
        "ml_nested3",
        "ml_calls_import",
        "ml_mixed",
        "ml_recurse",
        "ml_big_frame",
        "ml_import_preserves_regs",
    ] {
        let (va, _) = elf_symbol_va_size(&elf, name).unwrap_or_else(|| panic!("{name} symbol"));
        assert!(
            va > entry_va,
            "{name} at {va:#x} precedes the entry at {entry_va:#x}; the lifted \
             region must start at the entry function"
        );
    }

    let text = file.section_by_name(".text").expect(".text section");
    let text_end = text.address() + text.size();
    assert!(text_end > entry_va);

    let off = elf_va_to_file_offset(&elf, entry_va).expect("VA->file offset");
    let bytes = elf[off..off + (text_end - entry_va) as usize].to_vec();

    Some(LoadedImage {
        _arena: arena,
        entry: ElfFunction {
            bytes,
            va: entry_va,
        },
        puts_plt: find_plt_entry_va(&elf, "puts").expect("puts PLT entry"),
        strlen_plt: find_plt_entry_va(&elf, "strlen").expect("strlen PLT entry"),
        memcpy_plt: find_plt_entry_va(&elf, "memcpy").expect("memcpy PLT entry"),
    })
}

/// Read a NUL-terminated guest string. Sound because the image is identity
/// mapped: the guest pointer *is* a host pointer.
fn read_guest_cstr(p: u64) -> String {
    assert_ne!(p, 0, "guest passed a NULL string to puts");
    let mut out = Vec::new();
    let mut q = p as *const u8;
    loop {
        let b = unsafe { *q };
        if b == 0 {
            break;
        }
        out.push(b);
        q = unsafe { q.add(1) };
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Callee-saved sentinels seeded into the guest: the ABI says the guest must
/// give every one of them back untouched.
const CALLEE_SAVED: [(usize, u64); 6] = [
    (gpr::RBX, 0xBBBB_0000_0000_BBBB),
    (gpr::RBP, 0xB9B9_0000_0000_B9B9),
    (12, 0x1212_0000_0000_1212),
    (13, 0x1313_0000_0000_1313),
    (14, 0x1414_0000_0000_1414),
    (15, 0x1515_0000_0000_1515),
];

/// One execution of the self-test: returns the guest's grade (`rax`), whatever
/// it printed, and the final state.
struct SelfTestRun {
    grade: i32,
    messages: Vec<String>,
    state: CpuState,
}

/// `strlen_bias` is added to every `strlen` result: 0 services the imports
/// faithfully, anything else lies to the guest so its own checks must fire.
fn run_stack_discipline(
    img: &LoadedImage,
    init: &CpuState,
    strlen_bias: u64,
    engine: Engine,
) -> SelfTestRun {
    let imports = [img.puts_plt, img.strlen_plt, img.memcpy_plt];
    let mut messages: Vec<String> = Vec::new();

    let state = {
        let mut handler = |va: u64, st: &mut CpuState| {
            if va == img.puts_plt {
                messages.push(read_guest_cstr(st.gpr[gpr::RDI]));
                st.gpr[gpr::RAX] = 1; // puts returns a non-negative value
            } else if va == img.strlen_plt {
                st.gpr[gpr::RAX] =
                    unsafe { host_strlen(st.gpr[gpr::RDI] as *const u8) } + strlen_bias;
            } else if va == img.memcpy_plt {
                st.gpr[gpr::RAX] = unsafe {
                    host_memcpy(
                        st.gpr[gpr::RDI] as *mut u8,
                        st.gpr[gpr::RSI] as *const u8,
                        st.gpr[gpr::RDX],
                    )
                } as u64;
            } else {
                panic!("unexpected imported call VA: {va:#x}");
            }
        };
        match engine {
            Engine::Microlift => run_microlift(&img.entry, init, &imports, &mut handler),
            #[cfg(feature = "unicorn-oracle")]
            Engine::Unicorn => run_unicorn(&img.entry, init, &imports, &mut handler),
        }
    };

    SelfTestRun {
        grade: state.gpr[gpr::RAX] as u32 as i32,
        messages,
        state,
    }
}

#[derive(Clone, Copy)]
enum Engine {
    Microlift,
    #[cfg(feature = "unicorn-oracle")]
    Unicorn,
}

/// Seed the guest entry: `rdi` = a 9-byte string, `rsi`/`rdx` = a blob, plus the
/// callee-saved sentinels and the shared stack.
fn selftest_init(stack: &HostStack, s: &[u8], blob: &[u8], img: &LoadedImage) -> CpuState {
    let mut init = CpuState::zeroed();
    init.gpr[gpr::RSP] = stack.rsp();
    init.rip = img.entry.va;
    init.gpr[gpr::RDI] = s.as_ptr() as u64;
    init.gpr[gpr::RSI] = blob.as_ptr() as u64;
    init.gpr[gpr::RDX] = blob.len() as u64;
    for (i, v) in CALLEE_SAVED {
        init.gpr[i] = v;
    }
    init
}

fn assert_callee_saved_intact(state: &CpuState, who: &str) {
    for (i, v) in CALLEE_SAVED {
        assert_eq!(
            state.gpr[i], v,
            "{who}: callee-saved gpr[{i}] was not restored by the guest"
        );
    }
}

/// The guest grades its own stack discipline across every call shape, and must
/// come back clean.
#[test]
fn test_guest_stack_discipline_selftest() {
    // Held until the image below is dropped (guards the fixed mapping).
    let _serialize = selftest_guard();

    let Some(img) = load_stack_discipline_image() else {
        eprintln!("skipping stack-discipline self-test: no x86-64 ELF compiler on PATH");
        return;
    };

    let s = b"microlift\0"; // strlen == 9
    let blob: [u8; 11] = [
        0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xAA, 0xBB,
    ];

    let stack = HostStack::new(4096);
    let init = selftest_init(&stack, s, &blob, &img);

    let ml = run_stack_discipline(&img, &init, 0, Engine::Microlift);
    assert!(
        ml.messages.is_empty() && ml.grade == 0,
        "guest self-test failed under microlift (grade {:#06x}):\n  {}",
        ml.grade,
        ml.messages.join("\n  ")
    );
    assert_callee_saved_intact(&ml.state, "microlift");
    // The `ret` into the exit trampoline pops the return address and nothing
    // else: the guest balanced its own stack exactly.
    assert_eq!(
        ml.state.gpr[gpr::RSP],
        stack.rsp() + 8,
        "microlift: guest did not return with a balanced stack"
    );

    #[cfg(feature = "unicorn-oracle")]
    {
        let uc = run_stack_discipline(&img, &init, 0, Engine::Unicorn);
        assert!(
            uc.messages.is_empty() && uc.grade == 0,
            "guest self-test failed under unicorn (grade {:#06x}):\n  {}",
            uc.grade,
            uc.messages.join("\n  ")
        );
        assert_callee_saved_intact(&uc.state, "unicorn");
        let diff = diff_states(&ml.state, &uc.state, DIFF);
        assert!(
            diff.is_empty(),
            "stack-discipline self-test diverged: {diff:?}"
        );
    }
}

/// Proves the self-test can actually fail, and that its reporting path works:
/// lie about `strlen` and the guest must notice, print, and grade itself down -
/// identically on both engines.
#[test]
fn test_guest_stack_discipline_detects_bad_import() {
    // Held until the image below is dropped (guards the fixed mapping).
    let _serialize = selftest_guard();

    let Some(img) = load_stack_discipline_image() else {
        eprintln!("skipping stack-discipline failure test: no x86-64 ELF compiler on PATH");
        return;
    };

    let s = b"microlift\0";
    let blob: [u8; 11] = [
        0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xAA, 0xBB,
    ];

    let stack = HostStack::new(4096);
    let init = selftest_init(&stack, s, &blob, &img);

    // `strlen` now returns one too many: the direct import check (bit 0x20) and
    // the mixed guest/import check (bit 0x80) must both trip - and only those,
    // since every stack-pointer check is unaffected by a wrong return value.
    const F_IMPORT_VALUE: i32 = 0x0020;
    const F_MIXED_VALUE: i32 = 0x0080;

    let ml = run_stack_discipline(&img, &init, 1, Engine::Microlift);
    assert_eq!(
        ml.grade,
        F_IMPORT_VALUE | F_MIXED_VALUE,
        "unexpected grade from the poisoned run; messages: {:?}",
        ml.messages
    );
    assert_eq!(
        ml.messages,
        vec![
            "FAIL: strlen(s) did not return 9".to_string(),
            "FAIL: ml_mixed did not return the expected length mix".to_string(),
        ],
        "guest did not report the failures it graded itself down for"
    );
    assert_callee_saved_intact(&ml.state, "microlift");

    #[cfg(feature = "unicorn-oracle")]
    {
        let uc = run_stack_discipline(&img, &init, 1, Engine::Unicorn);
        assert_eq!(
            uc.messages, ml.messages,
            "engines disagree on the guest's puts output"
        );
        let diff = diff_states(&ml.state, &uc.state, DIFF);
        assert!(diff.is_empty(), "poisoned self-test diverged: {diff:?}");
    }
}
