//! Behaviour checks any `occt-kernel` implementation must pass.
//!
//! This crate is the executable specification of the kernel contract. Every check
//! is generic over [`Kernel`], is written only against the contract, and compares
//! the kernel with an answer derived on paper (usually a closed-form volume), so a
//! future kernel passes by being correct, not by resembling OCCT.
//!
//! Each public function checks one behaviour and panics with a message naming what
//! went wrong. [`run_all`] runs them all and reports every failure together;
//! [`checks`] lists them for a harness that wants to run them its own way.
//!
//! ```no_run
//! # fn demo<K: occt_kernel::Kernel>() {
//! occt_conformance::run_all::<K>();
//! # }
//! ```
//!
//! # Tolerances
//! The tolerances live in one private module and are derived from what a modelling
//! kernel can promise, not from what any one kernel happens to deliver. See the
//! comments there before loosening one.

#![forbid(unsafe_code)]

pub mod advanced_blends_and_booleans;
mod blends;
mod booleans;
pub mod construction_and_geometry;
pub mod exchange;
pub mod features_and_sweeps;
pub mod inspection;
mod meshing;
mod primitives;
mod step;
mod support;
pub mod surfacing;
mod sweeps;
mod topology;
mod transforms;

use std::panic;

use occt_kernel::Kernel;

pub use blends::*;
pub use booleans::*;
pub use meshing::*;
pub use primitives::*;
pub use step::*;
pub use sweeps::*;
pub use topology::*;
pub use transforms::*;

macro_rules! check_list {
    ($($name:ident),* $(,)?) => {
        /// Every check as a `(name, function)` pair, in a stable order.
        #[must_use]
        pub fn checks<K: Kernel>() -> Vec<(&'static str, fn())> {
            let mut checks: Vec<(&'static str, fn())> =
                vec![$((stringify!($name), $name::<K> as fn())),*];
            checks.extend(construction_and_geometry::checks::<K>());
            checks.extend(features_and_sweeps::checks::<K>());
            checks.extend(advanced_blends_and_booleans::checks::<K>());
            checks.extend(inspection::checks::<K>());
            checks.extend(exchange::checks::<K>());
            checks.extend(surfacing::checks::<K>());
            checks
        }
    };
}

check_list![
    kernel_identifies_itself,
    box_measures,
    cylinder_measures,
    cone_measures,
    frustum_measures,
    sphere_measures,
    polygon_face_measures,
    non_planar_polygon_is_rejected,
    box_topology_counts,
    box_subshapes_are_distinct,
    box_topology_is_shared,
    box_vertices_are_its_corners,
    box_edges_are_lines_with_expected_lengths,
    total_length_counts_each_edge_once,
    a_shape_contains_itself,
    surface_and_curve_kinds,
    identity_is_an_equivalence_with_consistent_hash,
    traversal_is_stable,
    separately_built_shapes_are_not_same,
    fuse_cut_common_of_overlapping_boxes,
    cut_cylinder_through_box,
    cut_cavity_keeps_both_surfaces,
    common_of_sphere_and_box_is_a_hemisphere,
    fuse_of_touching_boxes_removes_the_shared_face,
    common_of_disjoint_shapes_has_no_volume,
    booleans_leave_operands_untouched,
    extrude_face_along_oblique_vector,
    revolve_rectangle_full_turn,
    revolve_rectangle_partial_turn,
    fillet_single_edge,
    chamfer_single_edge,
    fillet_all_edges_of_a_cube,
    impossible_fillet_is_rejected,
    translate_moves_without_changing_size,
    rotate_quarter_turn_about_offset_axis,
    scale_about_a_point,
    transforms_make_independent_shapes,
    mesh_is_well_formed,
    mesh_box_is_exact,
    mesh_curved_solid_is_within_deflection,
    mesh_is_watertight,
    mesh_winds_outward,
    finer_deflection_gives_more_triangles,
    step_round_trip_keeps_measures,
    step_round_trip_keeps_placement,
    step_output_is_part_21,
    unparseable_step_is_rejected,
];

/// Runs every check, then panics listing the ones that failed.
///
/// A failing check does not stop the others, so one run shows everything a kernel
/// still has to fix. Each check's own panic message goes to the panic hook as usual.
///
/// # Panics
/// If any check fails.
pub fn run_all<K: Kernel>() {
    let failed: Vec<_> = checks::<K>()
        .into_iter()
        .filter(|(_, check)| panic::catch_unwind(*check).is_err())
        .map(|(name, _)| name)
        .collect();
    assert!(
        failed.is_empty(),
        "kernel {} failed {} conformance check(s): {}",
        K::NAME,
        failed.len(),
        failed.join(", ")
    );
}
