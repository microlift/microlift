use crate::lifter::Lifter;
use iced_x86::Instruction;

impl<'a> Lifter<'a> {
    /// `SYSCALL`: inline host callback; the published PC is the return address (as `RCX` would be).
    pub(super) fn handle_syscall(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let rip_after = self.runtime_addr(instruction.next_ip());
        self.emit_syscall_callback(Some(rip_after))
    }
}
