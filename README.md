# Microlift

Microlift is a crate for lifting x86_64 and aarch64 to Cranelift IR, with the goal of providing a simple and efficient way to run foreign machine code on different architectures in JIT and AOT recompilation scenarios. It is primarily intended to be used as the basis for a recompilation based usermode hypervisor.

Due to its ability to also be used as a general purpose binary lifter, it has also been used for various other projects despite its primary goal as a hypervisor base, such as in the Microreverse binary analysis framework.

## Features

* Most common instructions are implemented
* VSA for more efficient code generation and dead store elimination
* Hints intended for use by high-level emulators such as information about entire memory segments, which assist in VSA and code analysis
* Support for x86_64 and aarch64

## Performance

Microlift integrates various techniques to maximize performance - the primary feature that gains the most performance is the elimination of unused CPU flags. CPU flags are represented as SSA variables, which allows them to be eliminated if they are set but unused by an instruction. This optimization is performed by Cranelift at lowering time. Due to this design choice, Cranelift IR emitted by Microlift may appear far messier than the source at first, but then compile into much cleaner assembly. If you wish to look at the Cranelift IR directly, it may help to run a dead store/load pass on it to eliminate unused CPU flags. Optimization passes from Cranelift should not affect the semantic equivalence of the code, although it will definitely mutate its structure.

## Notes for non-JIT users

Microlift is primarily designed for JIT recompilation, however, it also has support for a RuntimeAssisted execution mode. In RuntimeAssisted, helper code is adjusted to assume that the `lifter` is gone, and that instead a custom runtime will be present specific to your architecture and platform. RuntimeAssisted code may be slower or lower quality than JIT code due to the extra VM exits and necessity for more VM entry dispatch points.
