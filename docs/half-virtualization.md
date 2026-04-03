# Half-Virtualization in Microlift

Half-virtualization is a hybrid software protection technique that interleaves native machine code with virtualized execution segments. In Microlift, we leverage **Cranelift IR** as the virtualization backend to achieve near-native performance for these protected regions, offering a significant speed advantage over traditional bytecode interpreters while maintaining a high level of abstraction.

---

## 🏗️ Architecture Overview

The system operates on a "trap-and-emulate" model, where the transition between native and virtual states is handled by a high-level runtime bridge.

### 1. VM Entry (The Trap)
The transition from native x86-64 to Microlift IR is triggered by an entry stub.

*   **Mechanism**: A `ud2` (Undefined Instruction) is inserted at the entry point of the protected code.
*   **Trap Handling**: When the CPU encounters `ud2`, it generates an **Illegal Instruction** trap (`SIGILL`).
*   **Signal Dispatcher**: A custom runtime handler intercepts the signal. It extracts the current CPU context (Registers, EFLAGS, IP) and maps the faulting RIP to a pre-compiled Cranelift block.

### 2. The Context Bridge
The **Context Bridge** is the data structure that facilitates the transfer of state between the host CPU and the Virtual Machine.

*   **State Extraction**: The signal handler converts the architecture-specific context (e.g., `ucontext_t`) into a portable `RegisterState` format.
*   **Initial Values**: The Lifter initializes Cranelift variables from this bridge at the start of the virtualized block.
*   **Memory Access**: Memory operations within the IR access the host process's address space directly, ensuring transparency for pointers and stack-relative operations.

### 3. Execution & Optimization Control
A critical challenge in JIT-based virtualization is preventing the compiler from removing "redundant" state updates that are semantically required for the host CPU.

#### 🛡️ Optimization Barriers
To ensure that all registers and flags are correctly restored during a **VM Exit**, we implement specialized IR barriers:
- **Materialization**: At the end of a virtual block, the IR must explicitly define all GPRs and EFLAGS as "live" or used.
- **Volatile Sinks**: We use opaque function calls or specialized return signatures that treat the entire CPU state as an output, preventing Cranelift from optimizing away flag calculations (like Zero or Sign flags) that might be needed by the subsequent native code.

### 4. VM Exit & Resumption
The "Exit" returns control to the native host environment.

1.  **State Flush**: The lifted block writes its internal `Variable` states back to the Context Bridge.
2.  **RIP Modification**: The instruction pointer in the signal context is updated to point to the resumption address (the instruction immediately following the `ud2`).
3.  **State Restore**: The OS signal return mechanism restores the modified registers and flags from the context to the actual hardware.
4.  **Native Resumption**: The CPU continues execution in the native environment with the semantically updated state.

---

## 🎯 Implementation Goals

| Feature | Description | Status |
| :--- | :--- | :--- |
| **UD2 Entry** | Support `ud2` as the primary VM entry mechanism via signal handlers. | 📝 Planned |
| **Flag Modeling** | Precise lifting of x86 bitwise flags (ZF, SF, CF, OF, PF, AF). | ✅ Partial |
| **Cross-Arch State** | Ability to lift x86-64 code and run it on AArch64 hosts. | 🚀 Researching |
| **Liveness Sync** | Automatic synchronization of modified registers back to host state. | 📝 Planned |

---

## 🔬 Technical Workflow: Case Study
If we virtualize `add rax, rbx`:

1.  **Host**: Hits `ud2`.
2.  **Microlift**: Captures `rax=10, rbx=20`.
3.  **IR**: Executes `v1 = iadd rax, rbx` and calculates new `ZF`.
4.  **Exit**: Stores `rax=30, ZF=0` into the `ucontext_t`.
5.  **Host**: Resumes at next IP with `rax=30`.