use cranelift_codegen::isa::CallConv;

/// Host calling convention. `CallConv::Fast` happens to work everywhere but is not guaranteed.
pub fn get_host_calling_conv() -> CallConv {
    if cfg!(target_os = "windows") {
        CallConv::WindowsFastcall
    } else if (cfg!(target_os = "macos") || cfg!(target_os = "ios"))
        && cfg!(target_arch = "aarch64")
    {
        // Apple's aarch64 variant of SystemV differs enough to need its own convention.
        CallConv::AppleAarch64
    } else {
        CallConv::SystemV
    }
}
