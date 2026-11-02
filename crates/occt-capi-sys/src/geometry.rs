//! Raw declarations of the geometry domain, matching `capi/src/geometry.cpp`.

capi! {
    "_occt_make_vertex" fn occt_make_vertex(x: f64, y: f64, z: f64) -> u32;
    "_occt_make_edge" fn occt_make_edge(data: usize, len: usize) -> u32;
    "_occt_make_wire" fn occt_make_wire(edges: usize, count: usize) -> u32;
    "_occt_make_planar_face" fn occt_make_planar_face(outer: u32, holes: usize, count: usize) -> u32;
    "_occt_make_surface_face" fn occt_make_surface_face(data: usize, len: usize) -> u32;
    "_occt_fill_boundary" fn occt_fill_boundary(data: usize, len: usize) -> u32;
    "_occt_sew" fn occt_sew(shapes: usize, count: usize, tolerance: f64) -> u32;
    "_occt_make_solid" fn occt_make_solid(h: u32) -> u32;
    "_occt_make_compound" fn occt_make_compound(shapes: usize, count: usize) -> u32;
    "_occt_make_torus" fn occt_make_torus(data: usize, len: usize) -> u32;
    "_occt_interpolate_curve" fn occt_interpolate_curve(points: usize, count: usize, closed: i32) -> u32;
    "_occt_fit_curve" fn occt_fit_curve(points: usize, count: usize, tolerance: f64) -> u32;
    "_occt_make_helix" fn occt_make_helix(
        px: f64, py: f64, pz: f64, dx: f64, dy: f64, dz: f64,
        radius: f64, pitch: f64, height: f64, left_handed: i32,
    ) -> u32;
    "_occt_offset_wire" fn occt_offset_wire(h: u32, distance: f64) -> u32;

    "_occt_edge_geometry" fn occt_edge_geometry(h: u32) -> u32;
    "_occt_edge_point" fn occt_edge_point(h: u32, t: f64) -> u32;
    "_occt_face_geometry" fn occt_face_geometry(h: u32) -> u32;
    "_occt_face_point" fn occt_face_point(h: u32, u: f64, v: f64) -> u32;
    "_occt_project_onto_face" fn occt_project_onto_face(h: u32, x: f64, y: f64, z: f64) -> u32;
    "_occt_project_onto_edge" fn occt_project_onto_edge(h: u32, x: f64, y: f64, z: f64) -> u32;
    "_occt_is_closed" fn occt_is_closed(h: u32) -> i32;
    "_occt_tolerance" fn occt_tolerance(h: u32) -> f64;
}
