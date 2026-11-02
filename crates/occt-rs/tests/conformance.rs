//! The active backend against the kernel specification.

#[test]
fn capi_kernel_conforms() {
    occt_conformance::run_all::<occt_kernel_capi::CapiKernel>();
}
