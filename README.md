# Microlift

Microlift is a crate for lifting x86-64 and AArch64 machine code into [Cranelift](https://cranelift.dev) IR, then using Cranelift, JIT or AOT compiles it for the host. Its primary purpose is to be the engine of a recompilation-based usermode hypervisor, and doubles as a general binary lifter and analysis backend.

Microlift's main user is [Rosemary](https://github.com/microlift/rosemary), a high-level emulator and compatibility layer combo that runs macOS applications across architectures and operating systems. It builds its `emulated` backend on top of Microlift, using it to run x86-64 macOS apps on AArch64 macOS, Linux, and FreeBSD.

## Status

| Guest | State |
| --- | --- |
| x86-64 | Mature - Good baseline of what usermode programs utilize, SSE through SSE4.2, AES-NI, atomics, and 80-bit x87 FPU written as pure IR. |
| AArch64 | Early - Only trivial programs will run, many instructions are still missing. |

Host code generation goes through Cranelift, so any Cranelift backend can be a target. The test suite compiles against x86-64, AArch64 and RISC-V64.

## Getting started

Microlift builds against our [fork of Wasmtime/Cranelift](https://github.com/microlift/wasmtime), which has experimental improvements that we aim to upstream if in-scope for Cranelift. The primary difference between our fork and upstream Cranelift is a wider register ABI for register pinning in block linking, and a `spin_hint` opcode, which emits to `PAUSE` on x86-64, and `YIELD` on AArch64.

The Nix flake is the recommended way to run Microlift's test suite. Using it on a system with Nix installed is simple:

```sh
nix develop --command cargo build
nix develop --command cargo test --release --features testkit
```

Lifting and running code with Microlift is easy. The below x86-64 code is lifted to a standard Cranelift function with Microlift, JIT-compiled with Cranelift, and called with a `RegisterContext`:

```rust
use microlift::arch::x86_64::X86_64;
use microlift::backend::CraneliftBackend;
use microlift::context::{ExitReason, RegisterContext};
use microlift::core::{LifterConfig, lift_to_code};

let code = [0x48, 0xC7, 0xC0, 0x2A, 0, 0, 0, 0xC3];
let base = 0x1000;

let lifted = lift_to_code::<X86_64>(&code, base, LifterConfig::default())?;
let mut backend = CraneliftBackend::new()?;
let entry = backend.compile_lifted(lifted)?;
let entry: extern "C" fn(*mut RegisterContext) = unsafe { std::mem::transmute(entry) };

let mut stack = vec![0u64; 64];
let mut ctx = RegisterContext::default();
ctx.image_base = base;
ctx.rsp = stack.as_mut_ptr() as u64 + 63 * 8;

entry(&mut ctx);
assert_eq!(ExitReason::try_from(ctx.exit_reason), Ok(ExitReason::UnknownJump));
assert_eq!(ctx.rax, 42);
```

A lifted region is a function taking one `*mut RegisterContext`. It runs the guest code against that context and reports why it stopped in `ctx.exit_reason`. Here the `ret` leaves the region, so the host sees an `UnknownJump` to the return address; the backend and its code must outlive the call. `CraneliftBackend` JIT-compiles in-process, and `ArtifactBackend` produces object code for any target triple.

## Testing

The `testkit` feature wraps the above in a common module that supplies the stack, the exit sentinel, and optional Unicorn cross-checking.

```rust
use microlift::testkit::prelude::*;

let out = LiftTest::new(&[0x48, 0xC7, 0xC0, 0x2A, 0, 0, 0, 0xC3]).run();
assert_eq!(out.ctx.rax, 42);
```

`LiftTest` takes its starting state and configuration through builder methods (`.at(ip)`, `.ctx(..)`, and so on). `.run()` panics if lifting fails, while `.try_run()` returns the error. Register every new test target in `Cargo.toml` with `required-features = ["testkit"]`, and run the suite with `cargo test --features testkit`. Adding `--features unicorn-oracle` makes every `run()` also execute the program on Unicorn and compare the resulting state.

## Features

* Address-keyed lifting. Decoding follows control flow into an address-keyed map, so overlapping and sub-encoded instructions lift correctly.
* CPU state as SSA. Registers and flags are Cranelift variables, not memory, so unused flag math disappears at lowering.
* Dead flag elimination and lazy register sync to optimize flag computation and register context loads/stores.
* Imports: guest calls to known addresses become inline native calls (fast), and VM exits (slow), or bridged calls (slated for removal).
* Syscall and interpreter callbacks that let the code carry on inline instead of exiting.
* Block linking, with link cells, pinned registers, a call-return stack and host call frames, to keep execution inside generated code.
* Host runtime with a page-indexed code cache, invalidation, and per-thread link tables.
* Source maps from host code back to guest addresses, for faults and debugger integration.
* Target specialization that picks cheaper lowerings per host without changing a single result bit.
* Static analysis: linear-sweep CFG, value-set analysis for indirect targets, and Clang jump-table recovery.
* Analysis mode that produces inspection-only IR.

## Accuracy

Microlift is accuracy-first. An instruction that is not implemented accurately must have its divergence documented, and the divergence is expected to be fixed eventually.

It is differentially tested against [Unicorn](https://www.unicorn-engine.org) (`--features unicorn-oracle`), with exemptions only for Microlift-specific features and documented inaccuracies in Unicorn itself. The suite has found a number of bugs in Unicorn and the QEMU TCG version it uses.

Optimizations are held to the same standard. A target specialization may not change a result bit. The exceptions are explicit opt-ins named `Unsafe*`.

## Performance

* Flags are SSA. Dead flag math is removed by Cranelift in optimization phases, which is why emitted IR looks far messier than the source assembly and compiles into much cleaner machine code. When reading IR, it helps to run Cranelift's optimization passes first.
* Register-context load/store is removed by the lifter's own optimizations (`lazy_register_sync`, on by default) and at `opt_level = speed` or above, by per-field alias regions Cranelift's egraph can see through.
* Vectors. With `use_vector_xmm`, XMM state is native `I64X2` and SSE instructions become Cranelift vector ops. x86 NaN behavior is still emulated for exactness, which costs extra compares on hosts with different NaN rules.
* Linking. Block linking is the biggest optimization: it lets lifted code resolve its own successors instead of returning to a dispatch loop, which can achieve performance improvements of ~20x depending on the guest.

## Execution modes

`ExecutionMode::Jit` is the default. `RuntimeAssisted` assumes the lifter is gone at run time and a custom runtime exists, so helpers become relocatable imports and exits. It is slower and needs more VM entry dispatch points.

## Known limitations

* MXCSR is not honored. Rounding mode, FZ/DAZ and sticky flags are stored faithfully but do not affect lifted arithmetic. Do not use Microlift for scientific work, and expect adversarial guests to detect it.
* x87 is software. Cranelift has no 80-bit float, so the FPU is pure IR, apfloat-verified. Transcendentals share an outlined kernel that is faithful to about 1 ulp, not bit-exact. `FPATAN`, `F2XM1`, `FYL2X` and `FYL2XP1` are unimplemented.
* NaN emulation overhead. Hosts whose NaN rules differ from x86 (such as AArch64) pay extra comparisons in vectorized floating point.
* AES-NI is not constant time. Do not rely on guest AES-NI for timing-attack resistance.
* `PAUSE` is dropped unless a `Specialization` is used with a target with the spin-loop hint, so spin loops can burn CPU and on AArch64 thrash TSO fences.
* `atomic128`: the default `Emulated` mode is not atomic. Use `Native` (x86-64 `lock cmpxchg16b`, or `caspal` with LSE on AArch64) or `Interpret`. `Emulated` is slated for removal due to its inherit unsafety.
* Segments: writing `FS`/`GS` stores the selector only, with no descriptor reload.
