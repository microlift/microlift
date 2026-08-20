# Microlift

Microlift is a crate for lifting x86_64 and AArch64 to Cranelift IR, with the goal of providing a simple and efficient way to run foreign machine code on different architectures in JIT and AOT recompilation scenarios. It is primarily intended to be used as the basis for a recompilation based usermode hypervisor.

Due to its ability to also be used as a general purpose binary lifter, it has also been used for various other projects despite its primary goal as a hypervisor base, such as in the Microreverse binary analysis framework.

## Features

* Most common instructions are implemented
* VSA for more efficient code generation and dead store elimination
* Hints intended for use by high-level emulators such as information about entire memory segments, which assist in VSA and code analysis
* Support for x86_64 and AArch64

## Accuracy

Microlift takes an accuracy-first approach to all instructions. If an instruction is not implemented accurately, for any reason, it is required that the divergence is documented, and that the divergence eventually is fixed.

It is differentially tested against Unicorn, with the only exemption to differential testing being Microlift-specific features and documented inaccuracies in the oracle we are testing against. Thanks to this approach, differential testing has found a dozen or so bugs in Unicorn (and consequently the QEMU TCG version Unicorn uses) thanks to its vast test suite. We plan on working directly with Unicorn and QEMU's developers to fix these bugs rather than let them sit around as trophies.

## Performance

Microlift integrates various techniques to maximize performance - the primary feature that gains the most performance is the elimination of unused CPU flags. CPU flags are represented as SSA variables, which allows them to be eliminated if they are set but unused by an instruction. This optimization is performed by Cranelift at lowering time. Due to this design choice, Cranelift IR emitted by Microlift may appear far messier than the source at first, but then compile into much cleaner assembly. If you wish to look at the Cranelift IR directly, it may help to run a dead store/load pass on it to eliminate unused CPU flags. Optimization passes from Cranelift should not affect the semantic equivalence of the code, although it will definitely mutate its structure.

Vectorized instructions such as the ones in SSE are emitted as native vectorized instructions via Cranelift's 128-bit integer support. By default, this will still be slower because Microlift opts for semantic accuracy first and performance second. This means we have to accurately emulate NaN behavior by doing manual calculations.

Some instructions do not have peephole optimizations necessary for efficient emulation of some x86-64 semantics. We plan to address this by directly contributing CLIF passes to Cranelift from our fork where not domain-specific, and by adding our own optimizations at lift time.

Additionally, compile with `opt_level = "speed"` (or `speed_and_size`) if you care about the register-context traffic: Cranelift's alias analysis runs inside the egraph pass, which only runs above `opt_level = none`, and the lifter tags every context field with its own alias region so that analysis can see the exit stores writing back exactly what the entry loads read.

The lifter also drops that traffic itself, unconditionally and at any opt level, via `LifterConfig::lazy_register_sync` (on by default) - a region never stores back a register it did not write.

## Notes for non-JIT users

Microlift is primarily designed for JIT recompilation, however, it also has support for a RuntimeAssisted execution mode. In RuntimeAssisted, helper code is adjusted to assume that the `lifter` is gone, and that instead a custom runtime will be present specific to your architecture and platform. RuntimeAssisted code may be slower or lower quality than JIT code due to the extra VM exits and necessity for more VM entry dispatch points.

## Warnings and known limitations

We want to make sure all known issues are documented, as they are usually symptoms of correctness bugs rather than something inherit to recompilation. These are the currently known semantic accuracy and performance issues:

* `MXCSR` is not respected yet - do not use Microlift in scientific applications, and expect it to be detected by adversial guest programs if they are looking for quirks.
* x87 floating points are slow and software-based, even on x86 CPUs due to a lack of support for native 80-bit floats in Cranelift. This will be addressed by implementing custom CLIF opcodes in our fork of Cranelift, though it is unlikely this work will ever be merged upstream as 80-bit floats is not in any WASM spec, and even on x86-64 there are few programs that utilize them. In particular, only `libstdc++` appears to use these features, and it's only for randomness distribution.
* Even with native vectorization support, x86 NaN behavior is emulated on platforms that do not have its semantics such as AArch64. This can cause overhead in programs using heavy floating point vectorization due to the additional comparisons for NaN emulation. This will eventually be togglable if eliding it is not required for program correctness, however, any optimizations that change semantics are potentially unsafe, and this is one of them.
* `PAUSE` is only emitted under the `SpinLoopHint` specialization, and is discarded otherwise - a tight loop relying on the CPU to listen to its hint will appear to use tons of processing power. Not required for semantic accuracy, but can cause performance issues across various programs depending on what the loop does (ie if it uses atomics a bunch it can thrash fences required for TSO emulation on AArch64). Enable it by lifting under a `Specialization` naming your target; there is no portable lowering to fall back on, since only the target knows what its spin-loop hint is.
