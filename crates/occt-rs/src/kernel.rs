//! Selects the kernel. This is the one module that names a backend.

use occt_kernel::{Kernel, Result, Topology};

#[cfg(not(feature = "capi"))]
compile_error!("occt-rs needs a kernel: enable the `capi` feature");

#[cfg(feature = "capi")]
pub(crate) type Active = occt_kernel_capi::CapiKernel;

/// What the active kernel calls a shape.
pub(crate) type RawShape = <Active as Topology>::Shape;

/// Attribution text that whoever distributes this software must reproduce, as the
/// active kernel's licences define. Empty if none is required.
///
/// # Examples
/// ```
/// println!("{}", occt::notice());
/// ```
#[must_use]
pub const fn notice() -> &'static str {
    Active::NOTICE
}

/// The version of the implementation behind the active kernel.
///
/// # Errors
/// [`Error::Unavailable`](crate::Error::Unavailable) if the kernel is not usable.
///
/// # Examples
/// ```
/// # fn main() -> occt::Result<()> {
/// assert!(!occt::kernel_version()?.is_empty());
/// # Ok(())
/// # }
/// ```
pub fn kernel_version() -> Result<String> {
    Active::version()
}

/// Checks that the kernel can run now.
///
/// Native kernels are always ready. On WebAssembly the kernel may load separately, and
/// until it has, this and every modelling operation report the error below. Calling
/// this is optional; it just lets you find out before starting work.
///
/// # Errors
/// [`Error::Unavailable`](crate::Error::Unavailable) if the kernel is not loaded yet or
/// has an incompatible version.
///
/// # Examples
/// ```
/// # fn main() -> occt::Result<()> {
/// occt::ready()?;
/// # Ok(())
/// # }
/// ```
pub fn ready() -> Result<()> {
    Active::ready()
}
