#[repr(C)]
#[derive(Default, Debug, Clone, Copy)]
pub struct RegisterContext {
    pub rax: u64,
    pub rcx: u64,
    pub rdx: u64,
    pub rbx: u64,
    pub rsp: u64,
    pub rbp: u64,
    pub rsi: u64,
    pub rdi: u64,
    pub r8: u64,
    pub r9: u64,
    pub r10: u64,
    pub r11: u64,
    pub r12: u64,
    pub r13: u64,
    pub r14: u64,
    pub r15: u64,
    pub rip: u64,
    pub image_base: u64,
    /// A syscall handler intended for the RuntimeAssisted execution mode.
    pub syscall_handler: Option<unsafe extern "C" fn(ctx: *mut RegisterContext)>,
    /// FIXME:
    /// # **DO NOT USE!!!**
    /// Resume function pointer for AOT block dispatch (**not implemented yet** - this is a placeholder)
    pub resume_function: Option<unsafe extern "C" fn(ctx: *mut RegisterContext)>,
    // Flags as individual bytes for simpler IR load/store
    pub cf: u8,
    pub pf: u8,
    pub af: u8,
    pub zf: u8,
    pub sf: u8,
    pub of: u8,
    pub exit_reason: u32,
    pub exit_code: u32,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExitReason {
    None = 0,
    Return = 1,
    Syscall = 2,
    UnknownJump = 3,
    UnimplementedInstruction = 4,
}

impl RegisterContext {
    pub fn new() -> Self {
        Self::default()
    }
}

#[derive(Debug, Clone)]
pub struct MemorySegment {
    pub address: u64,
    pub data: Vec<u8>,
    pub writable: bool,
}

impl MemorySegment {
    pub fn new(address: u64, data: Vec<u8>, writable: bool) -> Self {
        Self {
            address,
            data,
            writable,
        }
    }

    pub fn contains(&self, addr: u64) -> bool {
        addr >= self.address && addr < self.address + self.data.len() as u64
    }

    pub fn read(&self, addr: u64, size: usize) -> Option<u64> {
        if !self.contains(addr) || !self.contains(addr + size as u64 - 1) {
            return None;
        }

        let offset = (addr - self.address) as usize;
        let bytes = &self.data[offset..offset + size];

        let mut val = 0u64;
        for (i, &b) in bytes.iter().enumerate() {
            val |= (b as u64) << (i * 8);
        }
        Some(val)
    }
}
