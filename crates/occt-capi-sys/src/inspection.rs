//! Raw declarations of the inspection domain, matching `capi/src/inspection.cpp`.

capi! {
    "_occt_mass_properties" fn occt_mass_properties(h: u32) -> u32;
    "_occt_distance" fn occt_distance(a: u32, b: u32) -> u32;
    "_occt_locate_point" fn occt_locate_point(h: u32, x: f64, y: f64, z: f64, tolerance: f64) -> i32;
    "_occt_ray_cast" fn occt_ray_cast(
        h: u32, ox: f64, oy: f64, oz: f64, dx: f64, dy: f64, dz: f64, max_distance: f64,
    ) -> u32;
    "_occt_ancestors" fn occt_ancestors(root: u32, sub: u32, kind: i32) -> u32;
    "_occt_edge_convexity" fn occt_edge_convexity(root: u32, e: u32) -> i32;
    "_occt_face_wires" fn occt_face_wires(h: u32) -> u32;
    "_occt_wire_edges" fn occt_wire_edges(h: u32) -> u32;
    "_occt_tessellate" fn occt_tessellate(
        h: u32, linear_deflection: f64, angular_deflection: f64, relative: i32, cancel: usize,
    ) -> u32;
    "_occt_hidden_lines" fn occt_hidden_lines(h: u32, view: usize, len: usize) -> u32;
}
