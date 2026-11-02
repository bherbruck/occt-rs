//! Raw declarations of the surfacing domain, matching `capi/src/surfacing.cpp`.

capi! {
    "_occt_measure_continuity" fn occt_measure_continuity(a: u32, b: u32, edge: u32) -> u32;
    "_occt_blend_faces" fn occt_blend_faces(data: usize, len: usize) -> u32;
    "_occt_extend_face" fn occt_extend_face(h: u32, length: f64, sides: u32) -> u32;
}
