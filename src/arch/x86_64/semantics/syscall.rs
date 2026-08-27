use crate::lifter::Lifter;
use iced_x86::Instruction;

impl<'a> Lifter<'a> {
    /// `SYSCALL` calls the host's syscall callback inline; see
    /// [`crate::core::callback`]. The PC published to it is the return address,
    /// matching the `RCX` a real `SYSCALL` would leave.
    pub(super) fn handle_syscall(&mut self, instruction: &Instruction) -> crate::Result<()> {
        let rip_after = self.runtime_addr(instruction.next_ip());
        self.emit_syscall_callback(Some(rip_after))
    }
}
