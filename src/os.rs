use cranelift_codegen::isa::CallConv;

/// Returns the host calling convention for the current platform.
///
/// Technically, `CallConv::Fast` works for the current Microlift implementation on all
/// platforms, but this is not guaranteed, so it's better to be safe.
pub fn get_host_calling_conv() -> CallConv {
    if cfg!(target_os = "windows") {
        CallConv::WindowsFastcall
    } else if (cfg!(target_os = "macos") || cfg!(target_os = "ios"))
        && cfg!(target_arch = "aarch64")
    {
        // Apple has a custom calling convention for aarch64. Technically, it's
        // a variant of SystemV, but it's different enough that we need to
        // use a different calling convention.
        CallConv::AppleAarch64
    } else {
        // On all other platforms, we assume SystemV calling convention.
        CallConv::SystemV
    }
}
