//! Executable memory shared by every region the JIT compiles.
//!
//! A JIT module per region gave each one pages of its own: on a 16 KiB page host, a
//! region of a few hundred bytes of code held 32 KiB or more, and a large program
//! lifts hundreds of thousands. Here regions are packed one after another into large
//! chunks, so the code costs its own size.
//!
//! Code is appended to pages other threads are executing, so a chunk can never be
//! made writable as a whole. On Darwin a `MAP_JIT` chunk is writable only on the
//! thread that opens a write window (`pthread_jit_write_protect_np`); elsewhere the
//! chunk is mapped twice, once executable and once writable. Nothing is ever freed:
//! a translation may run for as long as the process does.

use std::sync::Mutex;

/// Address space reserved per chunk; pages cost nothing until they are written.
const CHUNK: usize = 64 << 20;

/// One reservation being filled.
struct Chunk {
    /// Where the code runs.
    exec: *mut u8,
    /// Where it is written: `exec` itself where the platform toggles protection.
    write: *mut u8,
    len: usize,
    used: usize,
}

// SAFETY: the pointers name mappings that are never unmapped; allocation is serialised
// by the arena's lock and each range is written by the one thread it was handed to.
unsafe impl Send for Chunk {}

/// A bump allocator over chunks, one for code and one for data lifted code writes.
pub struct CodeArena {
    code: Mutex<Option<Chunk>>,
    data: Mutex<Option<Chunk>>,
}

/// Executable bytes handed out by [`CodeArena::code`], writable through
/// [`CodeArena::write`].
#[derive(Clone, Copy, Debug)]
pub struct CodeSpan {
    pub exec: *mut u8,
    write: *mut u8,
    pub len: usize,
}

impl CodeArena {
    pub const fn new() -> Self {
        Self {
            code: Mutex::new(None),
            data: Mutex::new(None),
        }
    }

    /// The process's arena.
    pub fn global() -> &'static CodeArena {
        static ARENA: CodeArena = CodeArena::new();
        &ARENA
    }

    /// `len` executable bytes aligned to `align`, not yet written.
    pub fn code(&self, len: usize, align: usize) -> std::io::Result<CodeSpan> {
        let (exec, write) = bump(&self.code, len, align, map_code)?;
        Ok(CodeSpan { exec, write, len })
    }

    /// `len` zeroed, writable bytes aligned to `align`.
    pub fn data(&self, len: usize, align: usize) -> std::io::Result<*mut u8> {
        Ok(bump(&self.data, len, align, map_data)?.0)
    }

    /// Writes `span` with `fill`, then makes what was written visible to instruction
    /// fetch on every thread.
    pub fn write<R>(&self, span: CodeSpan, fill: impl FnOnce(&mut [u8]) -> R) -> R {
        let result = {
            let _window = WriteWindow::open();
            // SAFETY: `span` came from this arena, which handed its bytes to one caller.
            let bytes = unsafe { std::slice::from_raw_parts_mut(span.write, span.len) };
            fill(bytes)
        };
        // SAFETY: the range just written, which is mapped executable.
        unsafe {
            wasmtime_internal_jit_icache_coherence::clear_cache(span.exec as *const _, span.len)
                .expect("clearing the instruction cache");
        }
        wasmtime_internal_jit_icache_coherence::pipeline_flush_mt().expect("flushing the pipeline");
        result
    }
}

impl Default for CodeArena {
    fn default() -> Self {
        Self::new()
    }
}

fn bump(
    chunk: &Mutex<Option<Chunk>>,
    len: usize,
    align: usize,
    map: fn(usize) -> std::io::Result<Chunk>,
) -> std::io::Result<(*mut u8, *mut u8)> {
    let mut chunk = chunk.lock().unwrap_or_else(|e| e.into_inner());
    let fits = |c: &Chunk| c.used.next_multiple_of(align) + len <= c.len;
    if !chunk.as_ref().is_some_and(fits) {
        // The rest of a full chunk is abandoned; it was never touched, so it costs
        // address space only.
        *chunk = Some(map(CHUNK.max(len.next_multiple_of(page_size())))?);
    }
    let chunk = chunk.as_mut().expect("just mapped");
    let at = chunk.used.next_multiple_of(align);
    chunk.used = at + len;
    // SAFETY: `at + len` is within the chunk, checked above.
    Ok(unsafe { (chunk.exec.add(at), chunk.write.add(at)) })
}

fn page_size() -> usize {
    // SAFETY: no preconditions.
    unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
}

fn mapped(ptr: *mut libc::c_void) -> std::io::Result<*mut u8> {
    match ptr == libc::MAP_FAILED {
        true => Err(std::io::Error::last_os_error()),
        false => Ok(ptr.cast()),
    }
}

fn map_data(len: usize) -> std::io::Result<Chunk> {
    // SAFETY: a fresh anonymous mapping.
    let ptr = mapped(unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            len,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANON,
            -1,
            0,
        )
    })?;
    Ok(Chunk {
        exec: ptr,
        write: ptr,
        len,
        used: 0,
    })
}

#[cfg(target_os = "macos")]
fn map_code(len: usize) -> std::io::Result<Chunk> {
    // SAFETY: a fresh anonymous mapping.
    let ptr = mapped(unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            len,
            libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
            libc::MAP_PRIVATE | libc::MAP_ANON | libc::MAP_JIT,
            -1,
            0,
        )
    })?;
    Ok(Chunk {
        exec: ptr,
        write: ptr,
        len,
        used: 0,
    })
}

/// Maps a shared memory object twice: executable, and writable elsewhere.
#[cfg(not(target_os = "macos"))]
fn map_code(len: usize) -> std::io::Result<Chunk> {
    let Some(fd) = memory_object() else {
        return map_writable_code(len);
    };
    let map = |prot| {
        // SAFETY: a fresh shared mapping of the object.
        mapped(unsafe { libc::mmap(std::ptr::null_mut(), len, prot, libc::MAP_SHARED, fd, 0) })
    };
    // SAFETY: `fd` is the object just created and owned here.
    let sized = unsafe { libc::ftruncate(fd, len as libc::off_t) } == 0;
    let chunk = match sized {
        false => Err(std::io::Error::last_os_error()),
        true => map(libc::PROT_READ | libc::PROT_EXEC).and_then(|exec| {
            Ok(Chunk {
                exec,
                write: map(libc::PROT_READ | libc::PROT_WRITE)?,
                len,
                used: 0,
            })
        }),
    };
    // SAFETY: the mappings keep the object alive; the descriptor is no longer needed.
    unsafe { libc::close(fd) };
    chunk.or_else(|_| map_writable_code(len))
}

#[cfg(any(target_os = "linux", target_os = "android", target_os = "freebsd"))]
fn memory_object() -> Option<libc::c_int> {
    // SAFETY: a NUL-terminated name and valid flags.
    let fd = unsafe { libc::memfd_create(c"microlift-code".as_ptr(), libc::MFD_CLOEXEC) };
    (fd >= 0).then_some(fd)
}

#[cfg(not(any(
    target_os = "macos",
    target_os = "linux",
    target_os = "android",
    target_os = "freebsd"
)))]
fn memory_object() -> Option<libc::c_int> {
    None
}

/// Where no memory object can be had: one mapping, writable and executable at once.
#[cfg(not(target_os = "macos"))]
fn map_writable_code(len: usize) -> std::io::Result<Chunk> {
    // SAFETY: a fresh anonymous mapping.
    let ptr = mapped(unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            len,
            libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
            libc::MAP_PRIVATE | libc::MAP_ANON,
            -1,
            0,
        )
    })?;
    Ok(Chunk {
        exec: ptr,
        write: ptr,
        len,
        used: 0,
    })
}

/// Makes `MAP_JIT` memory writable, and not executable, on this thread until dropped.
struct WriteWindow;

impl WriteWindow {
    fn open() -> Self {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        // SAFETY: affects only this thread's view of `MAP_JIT` memory, restored on drop.
        unsafe {
            pthread_jit_write_protect_np(0)
        };
        WriteWindow
    }
}

impl Drop for WriteWindow {
    fn drop(&mut self) {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        // SAFETY: as in `open`.
        unsafe {
            pthread_jit_write_protect_np(1)
        };
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
unsafe extern "C" {
    fn pthread_jit_write_protect_np(enabled: libc::c_int);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn code_written_through_the_arena_runs() {
        let arena = CodeArena::new();
        #[cfg(target_arch = "aarch64")]
        let code: &[u8] = &[0x40, 0x05, 0x80, 0x52, 0xc0, 0x03, 0x5f, 0xd6]; // mov w0, #42; ret
        #[cfg(target_arch = "x86_64")]
        let code: &[u8] = &[0xb8, 42, 0, 0, 0, 0xc3]; // mov eax, 42; ret
        #[cfg(target_arch = "riscv64")]
        let code: &[u8] = &[0x13, 0x05, 0xa0, 0x02, 0x67, 0x80, 0x00, 0x00]; // li a0, 42; ret
        for _ in 0..3 {
            let span = arena.code(code.len(), 16).unwrap();
            arena.write(span, |bytes| bytes.copy_from_slice(code));
            // SAFETY: the bytes just written are a complete function of this type.
            let f: extern "C" fn() -> u32 = unsafe { std::mem::transmute(span.exec) };
            assert_eq!(f(), 42);
        }
    }

    #[test]
    fn allocations_are_packed_and_aligned() {
        let arena = CodeArena::new();
        let a = arena.code(10, 16).unwrap().exec as usize;
        let b = arena.code(10, 16).unwrap().exec as usize;
        assert_eq!(b - a, 16);
        let d = arena.data(64, 32).unwrap() as usize;
        assert_eq!(d % 32, 0);
    }
}
