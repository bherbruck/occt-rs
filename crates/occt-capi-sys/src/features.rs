//! Raw declarations of the features domain, matching `capi/src/features.cpp`.

capi! {
    "_occt_extrude_modeled" fn occt_extrude_modeled(h: u32, dx: f64, dy: f64, dz: f64) -> u32;
    "_occt_revolve_modeled" fn occt_revolve_modeled(
        h: u32, px: f64, py: f64, pz: f64, dx: f64, dy: f64, dz: f64, angle: f64,
    ) -> u32;
    "_occt_extrude_tapered" fn occt_extrude_tapered(
        h: u32, dx: f64, dy: f64, dz: f64, taper: f64,
    ) -> u32;
    "_occt_sweep" fn occt_sweep(
        profile: u32, path: u32, orientation: i32, has_binormal: i32,
        bx: f64, by: f64, bz: f64, solid: i32,
    ) -> u32;
    "_occt_loft" fn occt_loft(
        sections: usize, count: usize, solid: i32, ruled: i32, tolerance: f64,
    ) -> u32;

    "_occt_shell" fn occt_shell(h: u32, faces: usize, count: usize, thickness: f64) -> u32;
    "_occt_offset" fn occt_offset(h: u32, distance: f64) -> u32;
    "_occt_thicken" fn occt_thicken(h: u32, thickness: f64) -> u32;
    "_occt_draft" fn occt_draft(
        h: u32, faces: usize, count: usize,
        pull_x: f64, pull_y: f64, pull_z: f64,
        px: f64, py: f64, pz: f64, nx: f64, ny: f64, nz: f64, angle: f64,
    ) -> u32;
    "_occt_remove_faces" fn occt_remove_faces(h: u32, faces: usize, count: usize) -> u32;
    "_occt_unify" fn occt_unify(h: u32) -> u32;
    "_occt_heal" fn occt_heal(h: u32, tolerance: f64) -> u32;
    "_occt_transformed" fn occt_transformed(h: u32, rows: usize, len: usize) -> u32;
    "_occt_placed" fn occt_placed(h: u32, rows: usize, len: usize) -> u32;
    "_occt_reversed" fn occt_reversed(h: u32) -> u32;
    "_occt_shape_shares_geometry" fn occt_shape_shares_geometry(a: u32, b: u32) -> i32;
}
