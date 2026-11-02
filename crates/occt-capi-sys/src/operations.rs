//! Raw declarations of the operations domain, matching `capi/src/operations.cpp`.

capi! {
    "_occt_boolean_general" fn occt_boolean_general(op: u32, input: usize, len: usize, fuzzy: f64, flags: u32, cancel: usize) -> u32;
    "_occt_split" fn occt_split(input: usize, len: usize, fuzzy: f64, flags: u32, cancel: usize) -> u32;
    "_occt_section" fn occt_section(a: u32, b: u32) -> u32;
    "_occt_section_plane" fn occt_section_plane(h: u32, ox: f64, oy: f64, oz: f64, nx: f64, ny: f64, nz: f64) -> u32;
    "_occt_fillet_edges" fn occt_fillet_edges(h: u32, input: usize, len: usize) -> u32;
    "_occt_chamfer_edges" fn occt_chamfer_edges(h: u32, input: usize, len: usize) -> u32;
}
