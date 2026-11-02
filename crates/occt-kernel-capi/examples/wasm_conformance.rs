//! Runs the conformance suite in the browser or Node target, driven by
//! `wasm_conformance.mjs`. wasm32 panics abort, so the driver calls one check at a time and
//! treats a trap as that check's failure. Natively it runs them all.

#[cfg(target_arch = "wasm32")]
mod web {
    use occt_kernel_capi::CapiKernel;
    use wasm_bindgen::prelude::wasm_bindgen;

    /// How many checks there are.
    #[wasm_bindgen]
    pub fn check_count() -> usize {
        occt_conformance::checks::<CapiKernel>().len()
    }

    /// The name of a check.
    #[wasm_bindgen]
    pub fn check_name(index: usize) -> String {
        occt_conformance::checks::<CapiKernel>()[index].0.to_owned()
    }

    /// Runs one check.
    #[wasm_bindgen]
    pub fn run_check(index: usize) {
        (occt_conformance::checks::<CapiKernel>()[index].1)();
    }
}

fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    occt_conformance::run_all::<occt_kernel_capi::CapiKernel>();
}
