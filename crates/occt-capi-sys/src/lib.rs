//! Raw bindings to the occt_capi module, the one replaceable C ABI over OCCT.
//!
//! Only `occt-kernel-capi` should use this crate. The ABI itself is documented in
//! `capi/src/internal.hpp`. All pointers cross as `usize`, because on the web they are
//! offsets into the module's own linear memory rather than Rust pointers.
//!
//! # Finding the library at run time
//! When this crate builds the module under `OUT_DIR`, Cargo puts it on the dynamic
//! loader path for `cargo run` and `cargo test`. A directory from `OCCT_CAPI_LIB_DIR`
//! outside the target directory isn't, so set `LD_LIBRARY_PATH` (Linux),
//! `DYLD_FALLBACK_LIBRARY_PATH` (macOS) or `PATH` (Windows) yourself. To ship, copy
//! the library from `DEP_OCCT_CAPI_LIB_DIR` next to the binary.
//!
//! # Web
//! On wasm32 the functions call `globalThis.occtCapi`, the instance of the Emscripten
//! module. Calling one while it's missing is a JS error, so check
//! [`module_status`] first.

/// The ABI this crate declares; compare with [`occt_abi_version`].
pub const ABI_VERSION: u32 = 3;

macro_rules! capi {
    ($( $js:tt fn $name:ident($($arg:ident: $ty:ty),* $(,)?) $(-> $ret:ty)?; )*) => {
        #[cfg(not(target_arch = "wasm32"))]
        #[allow(missing_docs)]
        mod raw {
            unsafe extern "C" {
                $( pub fn $name($($arg: $ty),*) $(-> $ret)?; )*
            }
        }

        #[cfg(target_arch = "wasm32")]
        #[allow(missing_docs)]
        mod raw {
            use wasm_bindgen::prelude::wasm_bindgen;

            #[wasm_bindgen]
            extern "C" {
                $(
                    #[wasm_bindgen(js_namespace = occtCapi, js_name = $js)]
                    pub unsafe fn $name($($arg: $ty),*) $(-> $ret)?;
                )*
            }
        }

        #[allow(unused_imports)]
        pub use raw::*;

        /// The module's exports, as named by Emscripten.
        #[allow(dead_code)]
        pub const EXPORTS: &[&str] = &[$($js),*];
    };
}

mod exchange;
mod features;
mod geometry;
mod inspection;
mod operations;
mod surfacing;

pub use exchange::*;
pub use features::*;
pub use geometry::*;
pub use inspection::*;
pub use operations::*;
pub use surfacing::*;

capi! {
    "_occt_abi_version" fn occt_abi_version() -> u32;
    "_occt_library_version_ptr" fn occt_library_version_ptr() -> usize;
    "_occt_library_version_len" fn occt_library_version_len() -> usize;
    "_occt_last_error_ptr" fn occt_last_error_ptr() -> usize;
    "_occt_last_error_len" fn occt_last_error_len() -> usize;

    "_occt_alloc" fn occt_alloc(len: usize) -> usize;
    "_occt_dealloc" fn occt_dealloc(ptr: usize);

    "_occt_buffer_ptr" fn occt_buffer_ptr(h: u32) -> usize;
    "_occt_buffer_len" fn occt_buffer_len(h: u32) -> usize;
    "_occt_buffer_free" fn occt_buffer_free(h: u32);

    "_occt_shape_free" fn occt_shape_free(h: u32);
    "_occt_shape_clone" fn occt_shape_clone(h: u32) -> u32;
    "_occt_shape_kind" fn occt_shape_kind(h: u32) -> i32;
    "_occt_shape_is_same" fn occt_shape_is_same(a: u32, b: u32) -> i32;
    "_occt_shape_hash" fn occt_shape_hash(h: u32) -> u32;
    "_occt_shape_is_valid" fn occt_shape_is_valid(h: u32) -> i32;
    "_occt_shape_subshapes" fn occt_shape_subshapes(h: u32, kind: i32) -> u32;

    "_occt_make_box" fn occt_make_box(x: f64, y: f64, z: f64, dx: f64, dy: f64, dz: f64) -> u32;
    "_occt_make_cylinder" fn occt_make_cylinder(
        px: f64, py: f64, pz: f64, dx: f64, dy: f64, dz: f64, radius: f64, height: f64,
    ) -> u32;
    "_occt_make_cone" fn occt_make_cone(
        px: f64, py: f64, pz: f64, dx: f64, dy: f64, dz: f64,
        radius1: f64, radius2: f64, height: f64,
    ) -> u32;
    "_occt_make_sphere" fn occt_make_sphere(x: f64, y: f64, z: f64, radius: f64) -> u32;
    "_occt_make_polygon_face" fn occt_make_polygon_face(points: usize, count: usize) -> u32;

    "_occt_extrude" fn occt_extrude(h: u32, dx: f64, dy: f64, dz: f64) -> u32;
    "_occt_revolve" fn occt_revolve(
        h: u32, px: f64, py: f64, pz: f64, dx: f64, dy: f64, dz: f64, angle: f64,
    ) -> u32;

    "_occt_fuse" fn occt_fuse(a: u32, b: u32) -> u32;
    "_occt_cut" fn occt_cut(a: u32, b: u32) -> u32;
    "_occt_common" fn occt_common(a: u32, b: u32) -> u32;

    "_occt_fillet" fn occt_fillet(h: u32, edges: usize, count: usize, radius: f64) -> u32;
    "_occt_chamfer" fn occt_chamfer(h: u32, edges: usize, count: usize, distance: f64) -> u32;

    "_occt_translate" fn occt_translate(h: u32, x: f64, y: f64, z: f64) -> u32;
    "_occt_rotate" fn occt_rotate(
        h: u32, px: f64, py: f64, pz: f64, dx: f64, dy: f64, dz: f64, angle: f64,
    ) -> u32;
    "_occt_scale" fn occt_scale(h: u32, x: f64, y: f64, z: f64, factor: f64) -> u32;

    "_occt_volume" fn occt_volume(h: u32) -> f64;
    "_occt_area" fn occt_area(h: u32) -> f64;
    "_occt_length" fn occt_length(h: u32) -> f64;
    "_occt_center_of_mass" fn occt_center_of_mass(h: u32) -> u32;
    "_occt_bounding_box" fn occt_bounding_box(h: u32) -> u32;
    "_occt_vertex_point" fn occt_vertex_point(h: u32) -> u32;
    "_occt_face_surface_kind" fn occt_face_surface_kind(h: u32) -> i32;
    "_occt_edge_curve_kind" fn occt_edge_curve_kind(h: u32) -> i32;

    "_occt_mesh" fn occt_mesh(
        h: u32, linear_deflection: f64, angular_deflection: f64, relative: i32,
    ) -> u32;

    "_occt_step_write" fn occt_step_write(h: u32) -> u32;
    "_occt_step_read" fn occt_step_read(data: usize, len: usize) -> u32;
}

/// Whether the module can be called now. Native code is linked, so it always can.
///
/// # Errors
/// On the web, if `globalThis.occtCapi` is missing or isn't the module.
#[cfg(not(target_arch = "wasm32"))]
pub fn module_status() -> Result<(), String> {
    Ok(())
}

/// Copies `len` bytes out of module memory.
///
/// # Safety
/// `ptr..ptr + len` must be readable module memory. `ptr` may be anything if `len` is 0.
#[cfg(not(target_arch = "wasm32"))]
#[must_use]
pub unsafe fn copy_out(ptr: usize, len: usize) -> Vec<u8> {
    if len == 0 {
        return Vec::new();
    }
    // SAFETY: the caller guarantees the range is readable and `len` is nonzero; the
    // module doesn't mutate it during this call.
    unsafe { std::slice::from_raw_parts(ptr as *const u8, len) }.to_vec()
}

/// Calls `f(ptr, len)` with `input` readable in module memory for the duration.
/// The range has no particular alignment.
///
/// # Errors
/// If the module can't allocate room for the input.
#[cfg(not(target_arch = "wasm32"))]
pub fn with_input<R>(input: &[u8], f: impl FnOnce(usize, usize) -> R) -> Result<R, String> {
    Ok(f(input.as_ptr() as usize, input.len()))
}

/// Whether the module can be called now.
///
/// # Errors
/// If `globalThis.occtCapi` is missing or isn't the module.
#[cfg(target_arch = "wasm32")]
pub fn module_status() -> Result<(), String> {
    web::status()
}

/// Copies `len` bytes out of module memory.
///
/// # Safety
/// `ptr..ptr + len` must be readable module memory. `ptr` may be anything if `len` is 0.
#[cfg(target_arch = "wasm32")]
#[must_use]
pub unsafe fn copy_out(ptr: usize, len: usize) -> Vec<u8> {
    web::copy_out(ptr, len)
}

/// Calls `f(ptr, len)` with `input` readable in module memory for the duration.
/// The range has no particular alignment.
///
/// # Errors
/// If the module can't allocate room for the input.
#[cfg(target_arch = "wasm32")]
pub fn with_input<R>(input: &[u8], f: impl FnOnce(usize, usize) -> R) -> Result<R, String> {
    web::with_input(input, f)
}

#[cfg(target_arch = "wasm32")]
mod web {
    use js_sys::{Reflect, Uint8Array, global};
    use wasm_bindgen::{JsCast, JsValue};

    const NOT_LOADED: &str = "the occt_capi module is not loaded: \
                              set globalThis.occtCapi = await createOcctCapi() first";

    fn module() -> Option<JsValue> {
        Reflect::get(&global(), &JsValue::from_str("occtCapi"))
            .ok()
            .filter(|m| m.is_object())
    }

    pub(super) fn status() -> Result<(), String> {
        let module = module().ok_or(NOT_LOADED)?;
        let probe = Reflect::get(&module, &JsValue::from_str("_occt_abi_version"))
            .map_err(|_| NOT_LOADED.to_owned())?;
        if probe.is_function() {
            Ok(())
        } else {
            Err("globalThis.occtCapi is not the occt_capi module".to_owned())
        }
    }

    // Fetched per call: growing the memory replaces the module's view of it.
    fn heap() -> Uint8Array {
        module()
            .and_then(|m| Reflect::get(&m, &JsValue::from_str("HEAPU8")).ok())
            .and_then(|h| h.dyn_into().ok())
            .expect("the occt_capi module exports HEAPU8")
    }

    fn range(heap: &Uint8Array, ptr: usize, len: usize) -> Uint8Array {
        let start = u32::try_from(ptr).expect("wasm32 pointers fit in u32");
        let end = u32::try_from(ptr + len).expect("wasm32 ranges fit in u32");
        heap.subarray(start, end)
    }

    pub(super) fn copy_out(ptr: usize, len: usize) -> Vec<u8> {
        if len == 0 {
            return Vec::new();
        }
        range(&heap(), ptr, len).to_vec()
    }

    struct Allocation(usize);

    impl Drop for Allocation {
        fn drop(&mut self) {
            // SAFETY: the pointer came from occt_alloc and is freed only here.
            unsafe { super::occt_dealloc(self.0) };
        }
    }

    pub(super) fn with_input<R>(
        input: &[u8],
        f: impl FnOnce(usize, usize) -> R,
    ) -> Result<R, String> {
        // SAFETY: occt_alloc takes any length and reports failure with 0.
        let ptr = unsafe { super::occt_alloc(input.len()) };
        if ptr == 0 {
            return Err(format!(
                "the module could not allocate {} bytes",
                input.len()
            ));
        }
        let allocation = Allocation(ptr);
        if !input.is_empty() {
            range(&heap(), allocation.0, input.len()).copy_from(input);
        }
        Ok(f(allocation.0, input.len()))
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    #[test]
    fn declarations_match_the_module_exports() {
        let mut declared: Vec<String> = [
            super::EXPORTS,
            super::exchange::EXPORTS,
            super::features::EXPORTS,
            super::geometry::EXPORTS,
            super::inspection::EXPORTS,
            super::operations::EXPORTS,
            super::surfacing::EXPORTS,
        ]
        .concat()
        .iter()
        .map(|&s| s.to_owned())
        .collect();
        declared.sort();

        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("capi/src");
        let mut defined = Vec::new();
        for entry in fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().is_some_and(|e| e == "cpp") {
                for line in fs::read_to_string(path).unwrap().lines() {
                    if line.starts_with("API ") {
                        let name = line.split('(').next().unwrap_or_default();
                        let name = name.rsplit([' ', '*']).next().unwrap_or_default();
                        defined.push(format!("_{name}"));
                    }
                }
            }
        }
        defined.sort();
        assert_eq!(declared, defined, "src/ declarations and capi/src differ");
    }
}
