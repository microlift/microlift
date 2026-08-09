#!/usr/bin/env python3
"""Re-derive the flag-free mnemonic list in `arch/x86_64/flag_effects.rs`.

Prints the `matches!` body for `is_flag_free`. Run from the repository root
after touching `src/arch/x86_64/semantics/`, and diff the result against what
that function contains:

    python3 tools/derive-flag-free.py

An instruction qualifies when its lift provably (a) never touches a condition
flag and (b) never syncs guest state to the host. Both are required: a handler
that syncs publishes every flag, so a flag write elided before it would be
published stale.

The taint propagates through *any mention* of another function's name, not just
`self.f(` call syntax. That is deliberate over-approximation - `MOVSB` reaches
`string_advance`, which reads DF, only by passing `Self::emit_movs` as a value,
and a call-shaped search declares it flag-free. Which is a miscompile.

The output is an allowlist, so erring towards a shorter list costs speed and
erring towards a longer one costs correctness. When in doubt, leave it out.
"""

import glob
import re
import sys

SEMANTICS = "src/arch/x86_64/semantics"

# A function mentioning any of these reads or publishes flag state.
TAINT = (
    "set_flag(",
    "state.flags",
    "flag_needed(",
    "sync_to_context",
    "write_exit_reason",
    "write_exit_code",
    "resumes_at",
    "handle_unimplemented",
)

FN = r"\n    (?:pub(?:\(super\))?\s+)?fn (\w+)\s*[(<]"


def function_bodies() -> dict[str, str]:
    """Every `fn` defined at impl level under `semantics/`, by name."""
    bodies: dict[str, str] = {}
    for path in sorted(glob.glob(f"{SEMANTICS}/*.rs")):
        src = open(path).read()
        for m in re.finditer(FN, src):
            start = m.end()
            nxt = re.search(FN, src[start:])
            end = start + (nxt.start() if nxt else len(src) - start)
            bodies[m.group(1)] = src[start:end]
    return bodies


def tainted(bodies: dict[str, str]) -> dict[str, bool]:
    """Which functions touch flags or sync, directly or transitively."""
    taint = {n: any(t in b for t in TAINT) for n, b in bodies.items()}
    names = set(bodies)
    refs = {
        n: {w for w in re.findall(r"\b\w+\b", b) if w in names and w != n}
        for n, b in bodies.items()
    }
    changed = True
    while changed:
        changed = False
        for n in bodies:
            if not taint[n] and any(taint[c] for c in refs[n]):
                taint[n] = True
                changed = True
    return taint


def dispatch() -> dict[str, str]:
    """Mnemonic -> handler, from `handle_instruction`'s match."""
    src = open(f"{SEMANTICS}/mod.rs").read()
    i = src.index("pub fn handle_instruction")
    j = src.index("\n    }\n", src.index("match instruction.mnemonic()", i))
    body = src[i:j]
    arms = re.findall(
        r"((?:\s*(?:\|\s*)?Mnemonic::\w+(?:\s*//[^\n]*\n)?)+)\s*=>\s*(?:\{)?\s*self\.(\w+)",
        body,
    )
    handlers: dict[str, str] = {}
    for names, handler in arms:
        for mnemonic in re.findall(r"Mnemonic::(\w+)", names):
            handlers[mnemonic] = handler
    return handlers


def main() -> int:
    bodies = function_bodies()
    if not bodies:
        print(f"no functions found under {SEMANTICS}/ - wrong directory?", file=sys.stderr)
        return 1
    taint = tainted(bodies)
    handlers = dispatch()
    clean = sorted(m for m, h in handlers.items() if not taint.get(h, True))
    print(f"// {len(clean)} mnemonics", file=sys.stderr)
    print(" | ".join(f"Mnemonic::{m}" for m in clean))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
