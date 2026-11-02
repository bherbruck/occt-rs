//! Checks for [`BooleanOps`] and [`AdvancedBlends`].
//!
//! Expected answers are set arithmetic and closed-form geometry. History is checked by
//! what a subshape's successors measure, so a kernel may rebuild topology as it likes.

use std::f64::consts::{FRAC_PI_2, PI};

use occt_kernel::{
    AdvancedBlends, Angle, Axis, BooleanOp, BooleanOps, BooleanOptions, CancelFlag, ChamferSize,
    Dir3, Error, History, Kernel, Plane, Point3, RadiusLaw, Result, ShapeKind, Vec3,
};

use crate::support::{
    BLEND, MEASURE, area_is, assert_valid, ball_volume, cuboid, must, near, parts, volume_is,
};

/// A bound on what these checks need from a kernel.
pub trait Operations: Kernel + BooleanOps + AdvancedBlends {}
impl<K: Kernel + BooleanOps + AdvancedBlends> Operations for K {}

macro_rules! check_list {
    ($($name:ident),* $(,)?) => {
        /// Every check of this domain as a `(name, function)` pair, in a stable order.
        #[must_use]
        pub fn checks<K: Operations>() -> Vec<(&'static str, fn())> {
            vec![$((stringify!($name), $name::<K> as fn())),*]
        }
    };
}

check_list![
    cut_many_arguments_by_many_tools,
    cut_by_overlapping_tools_counts_the_overlap_once,
    fuse_of_a_set_of_shapes,
    common_between_sets,
    common_of_disjoint_sets_has_no_volume,
    fuzzy_value_closes_a_small_gap,
    unify_merges_coplanar_faces,
    unified_history_points_into_the_result,
    parallel_gives_the_same_answer,
    boolean_with_flag_already_set_is_cancelled,
    split_with_flag_already_set_is_cancelled,
    boolean_leaves_inputs_untouched,
    history_of_a_cut_follows_faces,
    history_of_a_cut_by_an_enclosed_tool,
    history_of_a_fuse_removes_the_shared_faces,
    section_of_two_shapes,
    section_of_disjoint_shapes_is_empty,
    section_plane_of_a_box_is_its_perimeter,
    section_plane_oblique_through_a_cube_is_a_hexagon,
    section_plane_through_a_ball_is_a_circle,
    section_plane_missing_the_shape_is_empty,
    split_by_a_face_keeps_both_sides,
    split_by_crossing_faces_gives_four_pieces,
    split_by_a_solid_keeps_inside_and_outside,
    split_history_follows_divided_faces,
    fillet_constant_radius_sweep,
    fillet_linear_radius_removes_the_integral,
    fillet_linear_radius_runs_from_the_first_vertex,
    fillet_through_points_passes_through_them,
    fillet_with_different_radius_per_edge,
    fillet_of_a_closed_edge,
    fillet_history_generates_the_rounding,
    chamfer_equal_sweep,
    chamfer_two_distances_measures_on_the_reference_face,
    chamfer_distance_angle_sweep,
    chamfer_two_edges_sharing_a_corner,
    chamfer_three_edges_sharing_a_corner,
    chamfer_with_different_size_per_edge,
    chamfer_history_generates_the_bevel,
    foreign_edges_and_faces_are_rejected,
    wrong_kinds_are_rejected,
    asymmetric_chamfer_needs_a_reference_face,
    too_large_blends_are_rejected,
];

const NO_OPTIONS: BooleanOptions = BooleanOptions {
    fuzzy: None,
    unify: false,
    parallel: false,
    cancel: None,
};

#[track_caller]
fn block<K: Kernel>(corner: [f64; 3], size: [f64; 3]) -> K::Shape {
    must(
        K::make_box(Point3::from(corner), Vec3::from(size)),
        "make_box",
    )
}

fn contains<K: Kernel>(shapes: &[K::Shape], shape: &K::Shape) -> bool {
    shapes.iter().any(|s| K::is_same(s, shape))
}

/// The entry for `shape` in a history list.
fn entry<'a, K: Kernel>(list: &'a [(K::Shape, Vec<K::Shape>)], shape: &K::Shape) -> &'a [K::Shape] {
    list.iter()
        .find(|(input, _)| K::is_same(input, shape))
        .map_or(&[], |(_, output)| output.as_slice())
}

/// The total length of a shape's edges, or zero if it has none.
#[track_caller]
fn total_length<K: Kernel>(shape: &K::Shape) -> f64 {
    parts::<K>(shape, ShapeKind::Edge)
        .iter()
        .map(|e| must(K::length(e), "edge length"))
        .sum()
}

#[track_caller]
fn total_area<K: Kernel>(shapes: &[K::Shape]) -> f64 {
    shapes.iter().map(|s| must(K::area(s), "area")).sum()
}

/// The face whose centre lies at `center`.
#[track_caller]
fn face_at<K: Kernel>(shape: &K::Shape, center: [f64; 3]) -> K::Shape {
    let center = Point3::from(center);
    parts::<K>(shape, ShapeKind::Face)
        .into_iter()
        .find(|f| must(K::center_of_mass(f), "centre").distance(center) < 1e-6)
        .unwrap_or_else(|| panic!("no face centred at {center:?}"))
}

/// The edge between two points given as its ends, in either order.
#[track_caller]
fn edge_between<K: Kernel>(shape: &K::Shape, a: [f64; 3], b: [f64; 3]) -> K::Shape {
    let middle = Point3::new(
        (a[0] + b[0]) / 2.0,
        (a[1] + b[1]) / 2.0,
        (a[2] + b[2]) / 2.0,
    );
    parts::<K>(shape, ShapeKind::Edge)
        .into_iter()
        .find(|e| must(K::center_of_mass(e), "centre").distance(middle) < 1e-6)
        .unwrap_or_else(|| panic!("no edge centred at {middle:?}"))
}

/// The ends of an edge, the first as the kernel lists its vertices.
#[track_caller]
fn ends<K: Kernel>(edge: &K::Shape) -> (Point3, Point3) {
    let vertices = parts::<K>(edge, ShapeKind::Vertex);
    assert_eq!(vertices.len(), 2, "an open edge has two vertices");
    (
        must(K::vertex_point(&vertices[0]), "vertex"),
        must(K::vertex_point(&vertices[1]), "vertex"),
    )
}

/// The radius of a rounded straight edge of a square bar of side `side`, a fraction along
/// the edge from `start` to `end`. A square section with one corner rounded has perimeter
/// `4 side - (2 - PI/2) r`, so the section's length gives `r`.
#[track_caller]
fn radius_along<K: Kernel + BooleanOps>(
    shape: &K::Shape,
    side: f64,
    (start, end): (Point3, Point3),
    fraction: f64,
) -> f64 {
    let along = end - start;
    let normal = must(Dir3::new(along).map_err(Error::from), "direction");
    let cut = must(
        K::section_plane(shape, Plane::new(start + along * fraction, normal)),
        "section",
    );
    (4.0 * side - total_length::<K>(&cut)) / (2.0 - FRAC_PI_2)
}

fn modeled_volume<K: Kernel>(result: Result<occt_kernel::Modeled<K::Shape>>, what: &str) -> f64 {
    let modeled = must(result, what);
    assert_valid::<K>(&modeled.shape, what);
    must(K::volume(&modeled.shape), what)
}

// --- booleans ---------------------------------------------------------------------

/// Two separate bars, each cut by two separate slots of known overlap with them.
///
/// # Panics
/// If the volume differs.
pub fn cut_many_arguments_by_many_tools<K: Operations>() {
    for gap in [1.0, 2.0, 5.0] {
        let a = block::<K>([0.0, 0.0, 0.0], [10.0, 2.0, 2.0]);
        let b = block::<K>([0.0, 2.0 + gap, 0.0], [10.0, 2.0, 2.0]);
        let y_span = 4.0 + gap + 2.0;
        let s1 = block::<K>([2.0, -1.0, -1.0], [1.0, y_span, 4.0]);
        let s2 = block::<K>([6.0, -1.0, -1.0], [2.0, y_span, 4.0]);
        let cut = must(
            K::boolean(BooleanOp::Cut, &[&a, &b], &[&s1, &s2], &NO_OPTIONS),
            "cut",
        );
        assert_valid::<K>(&cut.shape, "cut");
        volume_is::<K>(&cut.shape, 2.0 * 40.0 - 2.0 * 12.0, "cut");
        assert_eq!(
            parts::<K>(&cut.shape, ShapeKind::Solid).len(),
            6,
            "each bar falls into three pieces"
        );
    }
}

/// Overlapping tools remove their union.
///
/// # Panics
/// If the volume differs.
pub fn cut_by_overlapping_tools_counts_the_overlap_once<K: Operations>() {
    let a = cuboid::<K>(10.0, 10.0, 10.0);
    let t1 = block::<K>([-1.0, -1.0, -1.0], [3.0, 12.0, 12.0]);
    let t2 = block::<K>([1.0, -1.0, -1.0], [3.0, 12.0, 12.0]);
    let cut = must(
        K::boolean(BooleanOp::Cut, &[&a], &[&t1, &t2], &NO_OPTIONS),
        "cut",
    );
    volume_is::<K>(&cut.shape, 6.0 * 100.0, "cut by overlapping tools");

    // Overlapping arguments count as their union as well.
    let b = block::<K>([5.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
    let tool = block::<K>([-1.0, -1.0, -1.0], [2.0, 12.0, 12.0]);
    let cut = must(
        K::boolean(BooleanOp::Cut, &[&a, &b], &[&tool], &NO_OPTIONS),
        "cut",
    );
    assert_valid::<K>(&cut.shape, "cut of overlapping arguments");
    volume_is::<K>(&cut.shape, 14.0 * 100.0, "cut of overlapping arguments");
}

/// A row of overlapping boxes unites, counting each overlap once.
///
/// # Panics
/// If the volume or solid count differs.
pub fn fuse_of_a_set_of_shapes<K: Operations>() {
    for n in [3_u32, 5, 8] {
        let boxes: Vec<_> = (0..n)
            .map(|i| block::<K>([f64::from(i) * 1.5, 0.0, 0.0], [2.0, 2.0, 2.0]))
            .collect();
        let refs: Vec<_> = boxes.iter().collect();
        let fused = must(
            K::boolean(BooleanOp::Fuse, &refs[..1], &refs[1..], &NO_OPTIONS),
            "fuse",
        );
        assert_valid::<K>(&fused.shape, "fused row");
        let length = f64::from(n - 1) * 1.5 + 2.0;
        volume_is::<K>(&fused.shape, length * 4.0, "fused row");
        assert_eq!(
            parts::<K>(&fused.shape, ShapeKind::Solid).len(),
            1,
            "one bar"
        );
    }
}

/// Each side of a common counts as the union of its shapes: two separate arguments with
/// one tool, then one argument with two overlapping tools.
///
/// # Panics
/// If the volume differs.
pub fn common_between_sets<K: Operations>() {
    let a1 = block::<K>([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]);
    let a2 = block::<K>([4.0, 0.0, 0.0], [2.0, 2.0, 2.0]);
    let tool = block::<K>([1.0, 0.0, 0.0], [4.0, 1.0, 2.0]);
    let common = must(
        K::boolean(BooleanOp::Common, &[&a1, &a2], &[&tool], &NO_OPTIONS),
        "common",
    );
    volume_is::<K>(
        &common.shape,
        2.0 + 2.0,
        "common of two arguments with a tool",
    );

    let a = cuboid::<K>(2.0, 2.0, 2.0);
    let t1 = block::<K>([1.0, 0.0, 0.0], [2.0, 2.0, 2.0]);
    let t2 = block::<K>([0.0, 1.0, 0.0], [2.0, 2.0, 2.0]);
    let common = must(
        K::boolean(BooleanOp::Common, &[&a], &[&t1, &t2], &NO_OPTIONS),
        "common",
    );
    // All of the argument except the unit-square column that neither tool reaches.
    volume_is::<K>(&common.shape, 6.0, "common of an argument with two tools");
}

/// Disjoint sets share nothing, and that isn't a failure of a general boolean.
///
/// # Panics
/// If it fails or has volume.
pub fn common_of_disjoint_sets_has_no_volume<K: Operations>() {
    let a = cuboid::<K>(1.0, 1.0, 1.0);
    let b = block::<K>([5.0, 5.0, 5.0], [1.0, 1.0, 1.0]);
    let common = must(
        K::boolean(BooleanOp::Common, &[&a], &[&b], &NO_OPTIONS),
        "common",
    );
    assert!(must(K::volume(&common.shape), "volume").abs() <= MEASURE);
    assert!(parts::<K>(&common.shape, ShapeKind::Solid).is_empty());
}

/// Boxes a thousandth apart stay separate, unless the fuzzy value says the gap is nothing.
///
/// # Panics
/// If the solid counts differ.
pub fn fuzzy_value_closes_a_small_gap<K: Operations>() {
    let gap = 1e-3;
    let a = cuboid::<K>(1.0, 1.0, 1.0);
    let b = block::<K>([1.0 + gap, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let plain = must(
        K::boolean(BooleanOp::Fuse, &[&a], &[&b], &NO_OPTIONS),
        "fuse",
    );
    assert_eq!(
        parts::<K>(&plain.shape, ShapeKind::Solid).len(),
        2,
        "a real gap"
    );
    let options = BooleanOptions {
        fuzzy: Some(2.0 * gap),
        ..NO_OPTIONS
    };
    let closed = must(
        K::boolean(BooleanOp::Fuse, &[&a], &[&b], &options),
        "fuzzy fuse",
    );
    assert_eq!(
        parts::<K>(&closed.shape, ShapeKind::Solid).len(),
        1,
        "gap closed"
    );
    near(
        must(K::volume(&closed.shape), "volume"),
        2.0,
        2.0 * gap,
        "fuzzy fuse volume",
    );
}

/// Fusing touching boxes with unification leaves the six faces of one box.
///
/// # Panics
/// If the face count differs.
pub fn unify_merges_coplanar_faces<K: Operations>() {
    let a = cuboid::<K>(1.0, 1.0, 1.0);
    let b = block::<K>([1.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let options = BooleanOptions {
        unify: true,
        ..NO_OPTIONS
    };
    let fused = must(K::boolean(BooleanOp::Fuse, &[&a], &[&b], &options), "fuse");
    assert_valid::<K>(&fused.shape, "unified fuse");
    volume_is::<K>(&fused.shape, 2.0, "unified fuse");
    area_is::<K>(&fused.shape, 10.0, "unified fuse");
    assert_eq!(
        parts::<K>(&fused.shape, ShapeKind::Face).len(),
        6,
        "a plain box"
    );
}

/// Unification rebuilds faces, and the history still names faces that are in the result.
///
/// # Panics
/// If history names a face outside the result.
pub fn unified_history_points_into_the_result<K: Operations>() {
    let a = cuboid::<K>(1.0, 1.0, 1.0);
    let b = block::<K>([1.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let options = BooleanOptions {
        unify: true,
        ..NO_OPTIONS
    };
    let fused = must(K::boolean(BooleanOp::Fuse, &[&a], &[&b], &options), "fuse");
    let result = parts::<K>(&fused.shape, ShapeKind::Face);
    for input in [&a, &b] {
        for face in parts::<K>(input, ShapeKind::Face) {
            for output in entry::<K>(&fused.history.modified, &face) {
                assert!(
                    contains::<K>(&result, output),
                    "a successor is in the result"
                );
            }
        }
    }
    // The bottom of the bar has merged, and both of its sources lead there.
    let bottom = face_at::<K>(&fused.shape, [1.0, 0.5, 0.0]);
    for input in [&a, &b] {
        let x = if K::is_same(input, &a) { 0.5 } else { 1.5 };
        let source = face_at::<K>(input, [x, 0.5, 0.0]);
        let successors = entry::<K>(&fused.history.modified, &source);
        assert!(
            contains::<K>(successors, &bottom),
            "merged faces trace to the merged face"
        );
    }
}

/// Asking for threads changes nothing about the answer.
///
/// # Panics
/// If a measure differs.
pub fn parallel_gives_the_same_answer<K: Operations>() {
    let a = cuboid::<K>(10.0, 10.0, 10.0);
    let tools: Vec<_> = (0..4)
        .map(|i| {
            let axis = Axis::new(Point3::new(2.0 * f64::from(i) + 2.0, 5.0, -1.0), Dir3::Z);
            must(K::make_cylinder(axis, 0.5, 12.0), "make_cylinder")
        })
        .collect();
    let refs: Vec<_> = tools.iter().collect();
    let options = BooleanOptions {
        parallel: true,
        ..NO_OPTIONS
    };
    let cut = must(
        K::boolean(BooleanOp::Cut, &[&a], &refs, &options),
        "parallel cut",
    );
    assert_valid::<K>(&cut.shape, "parallel cut");
    volume_is::<K>(&cut.shape, 1000.0 - 4.0 * PI * 0.25 * 10.0, "parallel cut");
}

/// A flag that is set before the start stops the operation.
///
/// # Panics
/// If the result is anything but the cancelled error.
pub fn boolean_with_flag_already_set_is_cancelled<K: Operations>() {
    let flag = CancelFlag::new();
    flag.cancel();
    let options = BooleanOptions {
        cancel: Some(flag),
        ..NO_OPTIONS
    };
    let a = cuboid::<K>(2.0, 2.0, 2.0);
    let b = block::<K>([1.0, 1.0, 1.0], [2.0, 2.0, 2.0]);
    for op in [BooleanOp::Fuse, BooleanOp::Cut, BooleanOp::Common] {
        assert!(
            matches!(
                K::boolean(op, &[&a], &[&b], &options),
                Err(Error::Cancelled)
            ),
            "{op:?} with a set flag must report cancellation"
        );
    }
}

/// A split stops the same way.
///
/// # Panics
/// If the result is anything but the cancelled error.
pub fn split_with_flag_already_set_is_cancelled<K: Operations>() {
    let flag = CancelFlag::new();
    flag.cancel();
    let options = BooleanOptions {
        cancel: Some(flag),
        ..NO_OPTIONS
    };
    let a = cuboid::<K>(2.0, 2.0, 2.0);
    let b = block::<K>([1.0, 1.0, 1.0], [2.0, 2.0, 2.0]);
    assert!(matches!(
        K::split(&[&a], &[&b], &options),
        Err(Error::Cancelled)
    ));
}

/// None of these operations changes its inputs.
///
/// # Panics
/// If an input changes.
pub fn boolean_leaves_inputs_untouched<K: Operations>() {
    let a = cuboid::<K>(2.0, 2.0, 2.0);
    let b = block::<K>([1.0, 1.0, 1.0], [2.0, 2.0, 2.0]);
    for op in [BooleanOp::Fuse, BooleanOp::Cut, BooleanOp::Common] {
        must(K::boolean(op, &[&a], &[&b], &NO_OPTIONS), "boolean");
    }
    must(K::split(&[&a], &[&b], &NO_OPTIONS), "split");
    must(K::section(&a, &b), "section");
    for input in [&a, &b] {
        volume_is::<K>(input, 8.0, "an input afterwards");
        assert_eq!(parts::<K>(input, ShapeKind::Face).len(), 6);
    }
}

/// A through hole leaves the side faces alone, changes top and bottom, and the tool's side
/// makes the wall. The tool's caps lie outside the block and are gone.
///
/// # Panics
/// If the history says otherwise.
pub fn history_of_a_cut_follows_faces<K: Operations>() {
    let (side, r) = (10.0, 2.0);
    let block = cuboid::<K>(side, side, side);
    let axis = Axis::new(Point3::new(5.0, 5.0, -5.0), Dir3::Z);
    let drill = must(K::make_cylinder(axis, r, 20.0), "make_cylinder");
    let cut = must(
        K::boolean(BooleanOp::Cut, &[&block], &[&drill], &NO_OPTIONS),
        "cut",
    );
    let History {
        modified,
        generated,
        deleted,
    } = &cut.history;
    let result = parts::<K>(&cut.shape, ShapeKind::Face);

    // Untouched: not deleted, and followed either to itself or to a same-sized successor.
    let wall_side = face_at::<K>(&block, [0.0, 5.0, 5.0]);
    assert!(
        !contains::<K>(deleted, &wall_side),
        "an untouched face isn't deleted"
    );
    let after = entry::<K>(modified, &wall_side);
    let followed = if after.is_empty() {
        vec![wall_side.clone()]
    } else {
        after.to_vec()
    };
    assert!(
        followed.iter().all(|f| contains::<K>(&result, f)),
        "it is in the result"
    );
    near(
        total_area::<K>(&followed),
        side * side,
        MEASURE,
        "untouched face area",
    );

    // Cut: modified, and its successor lost the circle.
    for z in [0.0, side] {
        let cap = face_at::<K>(&block, [5.0, 5.0, z]);
        let after = entry::<K>(modified, &cap);
        assert!(!after.is_empty(), "a cut face is reported modified");
        assert!(
            after.iter().all(|f| contains::<K>(&result, f)),
            "its successor is in the result"
        );
        near(
            total_area::<K>(after),
            side * side - PI * r * r,
            MEASURE,
            "cut face area",
        );
    }

    // The tool's side generated the wall of the hole, and its caps are gone.
    let drill_faces = parts::<K>(&drill, ShapeKind::Face);
    let (sides, caps): (Vec<_>, Vec<_>) = drill_faces
        .iter()
        .partition(|f| must(K::area(f), "area") > 2.0 * PI * r * r);
    assert_eq!((sides.len(), caps.len()), (1, 2), "a cylinder's faces");
    let walls: Vec<_> = entry::<K>(generated, sides[0])
        .iter()
        .filter(|s| must(K::kind(s), "kind") == ShapeKind::Face)
        .cloned()
        .collect();
    assert!(!walls.is_empty(), "the tool's side generates the wall");
    assert!(
        walls.iter().all(|f| contains::<K>(&result, f)),
        "the wall is in the result"
    );
    near(
        total_area::<K>(&walls),
        2.0 * PI * r * side,
        MEASURE,
        "wall area",
    );
    for cap in caps {
        assert!(
            contains::<K>(deleted, cap),
            "a tool cap outside the block is deleted"
        );
    }
}

/// A tool entirely inside makes a cavity from all of its surface, undivided.
///
/// # Panics
/// If the history says otherwise.
pub fn history_of_a_cut_by_an_enclosed_tool<K: Operations>() {
    let r = 2.0;
    let block = cuboid::<K>(10.0, 10.0, 10.0);
    let ball = must(K::make_sphere(Point3::new(5.0, 5.0, 5.0), r), "make_sphere");
    let cut = must(
        K::boolean(BooleanOp::Cut, &[&block], &[&ball], &NO_OPTIONS),
        "cut",
    );
    volume_is::<K>(&cut.shape, 1000.0 - ball_volume(r), "cavity");
    let result = parts::<K>(&cut.shape, ShapeKind::Face);
    let faces = parts::<K>(&ball, ShapeKind::Face);
    let walls: Vec<_> = faces
        .iter()
        .flat_map(|f| entry::<K>(&cut.history.generated, f).to_vec())
        .filter(|s| must(K::kind(s), "kind") == ShapeKind::Face)
        .collect();
    assert!(
        walls.iter().all(|f| contains::<K>(&result, f)),
        "walls are in the result"
    );
    near(
        total_area::<K>(&walls),
        4.0 * PI * r * r,
        MEASURE,
        "cavity wall area",
    );
}

/// Fusing touching boxes removes both faces at the joint, in full.
///
/// # Panics
/// If the history says otherwise.
pub fn history_of_a_fuse_removes_the_shared_faces<K: Operations>() {
    let a = cuboid::<K>(1.0, 1.0, 1.0);
    let b = block::<K>([1.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let fused = must(
        K::boolean(BooleanOp::Fuse, &[&a], &[&b], &NO_OPTIONS),
        "fuse",
    );
    for joint in [
        face_at::<K>(&a, [1.0, 0.5, 0.5]),
        face_at::<K>(&b, [1.0, 0.5, 0.5]),
    ] {
        assert!(
            contains::<K>(&fused.history.deleted, &joint),
            "a face inside the union is deleted"
        );
    }
    assert!(
        !contains::<K>(&fused.history.deleted, &face_at::<K>(&a, [0.0, 0.5, 0.5])),
        "an outer face survives"
    );
}

// --- sections ---------------------------------------------------------------------

/// A bar through a block crosses each face it passes in one circle.
///
/// # Panics
/// If the length differs.
pub fn section_of_two_shapes<K: Operations>() {
    for r in [0.5, 1.0, 2.5] {
        let block = cuboid::<K>(10.0, 10.0, 4.0);
        let axis = Axis::new(Point3::new(5.0, 5.0, -3.0), Dir3::Z);
        let bar = must(K::make_cylinder(axis, r, 10.0), "make_cylinder");
        let section = must(K::section(&block, &bar), "section");
        near(
            total_length::<K>(&section),
            2.0 * 2.0 * PI * r,
            MEASURE,
            "two rims",
        );
    }
}

/// Shapes that don't meet have no curves in common.
///
/// # Panics
/// If there are any, or it fails.
pub fn section_of_disjoint_shapes_is_empty<K: Operations>() {
    let a = cuboid::<K>(1.0, 1.0, 1.0);
    let b = block::<K>([5.0, 5.0, 5.0], [1.0, 1.0, 1.0]);
    let section = must(K::section(&a, &b), "section");
    assert!(parts::<K>(&section, ShapeKind::Edge).is_empty());
}

/// A plane at any height through a box crosses its four sides.
///
/// # Panics
/// If the length differs.
pub fn section_plane_of_a_box_is_its_perimeter<K: Operations>() {
    let (x, y, z) = (2.0, 3.0, 4.0);
    let block = cuboid::<K>(x, y, z);
    let cases = [
        (Point3::new(0.0, 0.0, z * 0.5), Dir3::Z, 2.0 * (x + y)),
        (Point3::new(0.0, 0.0, z * 0.1), Dir3::Z, 2.0 * (x + y)),
        (Point3::new(x * 0.5, 0.0, 0.0), Dir3::X, 2.0 * (y + z)),
        (Point3::new(0.0, y * 0.25, 0.0), Dir3::Y, 2.0 * (x + z)),
    ];
    for (origin, normal, want) in cases {
        let section = must(
            K::section_plane(&block, Plane::new(origin, normal)),
            "section",
        );
        near(total_length::<K>(&section), want, MEASURE, "perimeter");
    }
}

/// The plane through a cube's centre, square to its diagonal, cuts a regular hexagon of
/// side `a / sqrt 2`.
///
/// # Panics
/// If the length differs.
pub fn section_plane_oblique_through_a_cube_is_a_hexagon<K: Operations>() {
    let a = 6.0;
    let cube = cuboid::<K>(a, a, a);
    let normal = must(
        Dir3::new(Vec3::new(1.0, 1.0, 1.0)).map_err(Error::from),
        "normal",
    );
    let plane = Plane::new(Point3::new(a / 2.0, a / 2.0, a / 2.0), normal);
    let section = must(K::section_plane(&cube, plane), "section");
    near(
        total_length::<K>(&section),
        3.0 * 2.0_f64.sqrt() * a,
        MEASURE,
        "hexagon",
    );
    assert_eq!(parts::<K>(&section, ShapeKind::Edge).len(), 6, "six sides");
}

/// A plane at height `h` cuts a ball of radius `r` in a circle of radius `sqrt(r^2 - h^2)`.
///
/// # Panics
/// If the length differs.
pub fn section_plane_through_a_ball_is_a_circle<K: Operations>() {
    let r = 5.0;
    let ball = must(K::make_sphere(Point3::ORIGIN, r), "make_sphere");
    for h in [0.5, 1.0, 3.0, 4.0] {
        let plane = Plane::new(Point3::new(0.0, 0.0, h), Dir3::Z);
        let section = must(K::section_plane(&ball, plane), "section");
        near(
            total_length::<K>(&section),
            2.0 * PI * (r * r - h * h).sqrt(),
            MEASURE,
            "circle of latitude",
        );
    }
}

/// A plane that misses the shape has no curves in common with it.
///
/// # Panics
/// If there are any, or it fails.
pub fn section_plane_missing_the_shape_is_empty<K: Operations>() {
    let block = cuboid::<K>(1.0, 1.0, 1.0);
    let plane = Plane::new(Point3::new(0.0, 0.0, 2.0), Dir3::Z);
    let section = must(K::section_plane(&block, plane), "section");
    assert!(parts::<K>(&section, ShapeKind::Edge).is_empty());
}

// --- splitting --------------------------------------------------------------------

/// A flat blade at `x` wider than a box of side `side` in Y and Z.
fn x_blade<K: Kernel>(x: f64, side: f64) -> K::Shape {
    let (lo, hi) = (-side, 2.0 * side);
    must(
        K::make_polygon_face(&[
            Point3::new(x, lo, lo),
            Point3::new(x, hi, lo),
            Point3::new(x, hi, hi),
            Point3::new(x, lo, hi),
        ]),
        "make_polygon_face",
    )
}

/// A blade anywhere through a box gives two pieces of the right sizes.
///
/// # Panics
/// If a piece differs.
pub fn split_by_a_face_keeps_both_sides<K: Operations>() {
    let side = 4.0;
    for x in [0.5, 1.0, 2.0, 3.5] {
        let block = cuboid::<K>(side, side, side);
        let blade = x_blade::<K>(x, side);
        let split = must(K::split(&[&block], &[&blade], &NO_OPTIONS), "split");
        assert_valid::<K>(&split.shape, "split");
        volume_is::<K>(&split.shape, side * side * side, "total of the pieces");
        let solids = parts::<K>(&split.shape, ShapeKind::Solid);
        assert_eq!(solids.len(), 2, "two sides");
        let mut volumes: Vec<f64> = solids
            .iter()
            .map(|s| must(K::volume(s), "volume"))
            .collect();
        volumes.sort_by(f64::total_cmp);
        let smaller = x.min(side - x) * side * side;
        near(volumes[0], smaller, MEASURE, "the smaller piece");
    }
}

/// Two blades crossing inside a box give four pieces.
///
/// # Panics
/// If a count or the total differs.
pub fn split_by_crossing_faces_gives_four_pieces<K: Operations>() {
    let side = 4.0;
    let block = cuboid::<K>(side, side, side);
    let across = x_blade::<K>(1.0, side);
    let along = must(
        K::make_polygon_face(&[
            Point3::new(-4.0, 3.0, -4.0),
            Point3::new(8.0, 3.0, -4.0),
            Point3::new(8.0, 3.0, 8.0),
            Point3::new(-4.0, 3.0, 8.0),
        ]),
        "make_polygon_face",
    );
    let split = must(
        K::split(&[&block], &[&across, &along], &NO_OPTIONS),
        "split",
    );
    assert_eq!(
        parts::<K>(&split.shape, ShapeKind::Solid).len(),
        4,
        "four quarters"
    );
    volume_is::<K>(&split.shape, side * side * side, "total of the pieces");
}

/// A solid tool inside a box divides it into a core and a shell, and doesn't add itself.
///
/// # Panics
/// If a count or the total differs.
pub fn split_by_a_solid_keeps_inside_and_outside<K: Operations>() {
    let big = cuboid::<K>(3.0, 3.0, 3.0);
    let core = block::<K>([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]);
    let split = must(K::split(&[&big], &[&core], &NO_OPTIONS), "split");
    assert_eq!(
        parts::<K>(&split.shape, ShapeKind::Solid).len(),
        2,
        "core and shell"
    );
    volume_is::<K>(&split.shape, 27.0, "total of the pieces");
}

/// A divided face has several successors that add up to it, and an undivided face stays.
///
/// # Panics
/// If the history says otherwise.
pub fn split_history_follows_divided_faces<K: Operations>() {
    let side = 4.0;
    let block = cuboid::<K>(side, side, side);
    let blade = x_blade::<K>(1.0, side);
    let split = must(K::split(&[&block], &[&blade], &NO_OPTIONS), "split");
    let result = parts::<K>(&split.shape, ShapeKind::Face);
    let divided = face_at::<K>(&block, [2.0, 2.0, 0.0]);
    let pieces = entry::<K>(&split.history.modified, &divided);
    assert_eq!(pieces.len(), 2, "a divided face has two pieces");
    assert!(
        pieces.iter().all(|p| contains::<K>(&result, p)),
        "both are in the result"
    );
    near(
        total_area::<K>(pieces),
        side * side,
        MEASURE,
        "the pieces' area",
    );
    let whole = face_at::<K>(&block, [0.0, 2.0, 2.0]);
    assert!(
        !contains::<K>(&split.history.deleted, &whole),
        "an undivided face stays"
    );
}

// --- fillets ----------------------------------------------------------------------

/// What rounding a right-angled edge takes from the cross-section, per square of radius.
const CORNER: f64 = 1.0 - PI / 4.0;

/// One radius removes one prism.
///
/// # Panics
/// If the volume differs.
pub fn fillet_constant_radius_sweep<K: Operations>() {
    let side = 10.0;
    for r in [0.1, 0.5, 1.0, 3.0] {
        let block = cuboid::<K>(side, side, side);
        let edge = edge_between::<K>(&block, [0.0, 0.0, 0.0], [0.0, 0.0, side]);
        let volume = modeled_volume::<K>(
            K::fillet_edges(&block, &[(&edge, RadiusLaw::Constant(r))]),
            "fillet",
        );
        near(
            volume,
            1000.0 - CORNER * r * r * side,
            BLEND * 1e-2,
            "rounded box",
        );
    }
}

/// With radius running evenly from `a` to `b`, the prism removed has volume
/// `CORNER * L * (a^2 + a b + b^2) / 3`.
///
/// The slope stays below 1/20, where the difference between a section square to the edge
/// and one through the rolling ball's contacts is about the slope squared, under the blend tolerance.
///
/// # Panics
/// If the volume differs.
pub fn fillet_linear_radius_removes_the_integral<K: Operations>() {
    let side = 20.0;
    for (a, b) in [(1.0, 1.5), (1.5, 1.0), (0.5, 1.5), (2.0, 2.5)] {
        let block = cuboid::<K>(side, side, side);
        let edge = edge_between::<K>(&block, [0.0, 0.0, 0.0], [0.0, 0.0, side]);
        let law = RadiusLaw::Linear { start: a, end: b };
        let removed = side * side * side
            - modeled_volume::<K>(K::fillet_edges(&block, &[(&edge, law)]), "linear fillet");
        let want = CORNER * side * (a * a + a * b + b * b) / 3.0;
        near(removed, want, BLEND, "removed by a linear fillet");
    }
}

/// The radius starts at the first vertex the kernel lists for the edge, and is even
/// along the way.
///
/// # Panics
/// If the radius at a station differs.
pub fn fillet_linear_radius_runs_from_the_first_vertex<K: Operations>() {
    let side = 20.0;
    let block = cuboid::<K>(side, side, side);
    let corners = [
        ([0.0, 0.0, 0.0], [0.0, 0.0, side]),
        ([0.0, 0.0, 0.0], [side, 0.0, 0.0]),
        ([side, side, 0.0], [side, side, side]),
        ([side, 0.0, side], [side, side, side]),
    ];
    let (a, b) = (1.0, 1.5);
    for (p, q) in corners {
        let edge = edge_between::<K>(&block, p, q);
        let law = RadiusLaw::Linear { start: a, end: b };
        let rounded = must(K::fillet_edges(&block, &[(&edge, law)]), "linear fillet");
        let span = ends::<K>(&edge);
        for t in [0.25, 0.5, 0.75] {
            near(
                radius_along::<K>(&rounded.shape, side, span, t),
                a + t * (b - a),
                BLEND,
                format_args!("radius at {t}"),
            );
        }
    }
}

/// A law through points has each radius at its fraction, whichever way round the edge is.
///
/// # Panics
/// If a radius differs.
pub fn fillet_through_points_passes_through_them<K: Operations>() {
    let side = 20.0;
    let block = cuboid::<K>(side, side, side);
    let edge = edge_between::<K>(&block, [0.0, 0.0, 0.0], [0.0, 0.0, side]);
    let points = vec![(0.0, 1.0), (0.3, 1.5), (0.6, 1.2), (1.0, 1.6)];
    let law = RadiusLaw::Points(points.clone());
    let rounded = must(
        K::fillet_edges(&block, &[(&edge, law)]),
        "fillet through points",
    );
    assert_valid::<K>(&rounded.shape, "fillet through points");
    let span = ends::<K>(&edge);
    for (t, r) in &points[1..3] {
        near(
            radius_along::<K>(&rounded.shape, side, span, *t),
            *r,
            BLEND,
            format_args!("radius at {t}"),
        );
    }
}

/// Parallel edges with separate radii remove separate prisms.
///
/// # Panics
/// If the volume differs.
pub fn fillet_with_different_radius_per_edge<K: Operations>() {
    let side = 10.0;
    let block = cuboid::<K>(side, side, side);
    let radii = [0.5, 1.0, 1.5, 2.0];
    let corners = [[0.0, 0.0], [side, 0.0], [0.0, side], [side, side]];
    let edges: Vec<_> = corners
        .iter()
        .map(|c| edge_between::<K>(&block, [c[0], c[1], 0.0], [c[0], c[1], side]))
        .collect();
    let spec: Vec<_> = edges
        .iter()
        .zip(radii)
        .map(|(e, r)| (e, RadiusLaw::Constant(r)))
        .collect();
    let removed: f64 = radii.iter().map(|r| CORNER * r * r * side).sum();
    let volume = modeled_volume::<K>(K::fillet_edges(&block, &spec), "per-edge fillet");
    near(volume, 1000.0 - removed, BLEND * 1e-2, "per-edge fillet");
}

/// The rim of a cylinder rounds to a known solid of revolution: Pappus gives the volume
/// taken as the spandrel's area times the path of its centroid.
///
/// # Panics
/// If the volume differs.
pub fn fillet_of_a_closed_edge<K: Operations>() {
    let (radius, height, r) = (5.0, 10.0, 1.0);
    let bar = must(
        K::make_cylinder(Axis::new(Point3::ORIGIN, Dir3::Z), radius, height),
        "make_cylinder",
    );
    let rim = parts::<K>(&bar, ShapeKind::Edge)
        .into_iter()
        .find(|e| {
            let length = must(K::length(e), "length");
            let c = must(K::center_of_mass(e), "centre");
            (length - 2.0 * PI * radius).abs() < 1e-6 && (c.z - height).abs() < 1e-6
        })
        .expect("the cylinder has a top rim");
    let rounded = must(
        K::fillet_edges(&bar, &[(&rim, RadiusLaw::Constant(r))]),
        "rim fillet",
    );
    assert_valid::<K>(&rounded.shape, "rim fillet");
    // The spandrel, area (1 - PI/4) r^2, has its centroid r (10 - 3 PI) / (12 - 3 PI) from
    // each of the two sides meeting at the corner.
    let area = CORNER * r * r;
    let inset = r * (10.0 - 3.0 * PI) / (12.0 - 3.0 * PI);
    let removed = area * 2.0 * PI * (radius - inset);
    near(
        must(K::volume(&rounded.shape), "volume"),
        PI * radius * radius * height - removed,
        BLEND,
        "rounded rim",
    );
    assert!(
        K::fillet_edges(
            &bar,
            &[(
                &rim,
                RadiusLaw::Linear {
                    start: 1.0,
                    end: 2.0
                }
            )]
        )
        .is_err(),
        "a radius can't jump where a closed edge meets itself"
    );
}

/// The rounding made from an edge is in the result, and is about as big as it should be;
/// the neighbours shrink.
///
/// # Panics
/// If the history says otherwise.
pub fn fillet_history_generates_the_rounding<K: Operations>() {
    let (side, r) = (10.0, 2.0);
    let block = cuboid::<K>(side, side, side);
    let edge = edge_between::<K>(&block, [0.0, 0.0, 0.0], [0.0, 0.0, side]);
    let rounded = must(
        K::fillet_edges(&block, &[(&edge, RadiusLaw::Constant(r))]),
        "fillet",
    );
    let result = parts::<K>(&rounded.shape, ShapeKind::Face);
    let made = entry::<K>(&rounded.history.generated, &edge);
    assert!(!made.is_empty(), "an edge generates its rounding");
    assert!(
        made.iter().all(|f| contains::<K>(&result, f)),
        "it is in the result"
    );
    near(
        total_area::<K>(made),
        FRAC_PI_2 * r * side,
        BLEND,
        "area of the rounding",
    );
    assert!(
        contains::<K>(&rounded.history.deleted, &edge),
        "the sharp edge is gone"
    );
    for neighbour in [[0.0, 5.0, 5.0], [5.0, 0.0, 5.0]] {
        let face = face_at::<K>(&block, neighbour);
        let after = entry::<K>(&rounded.history.modified, &face);
        assert!(!after.is_empty(), "a neighbouring face is modified");
        near(
            total_area::<K>(after),
            side * (side - r),
            MEASURE,
            "neighbour after rounding",
        );
    }
    let far = edge_between::<K>(&block, [side, side, 0.0], [side, side, side]);
    assert!(
        !contains::<K>(&rounded.history.deleted, &far),
        "a distant edge survives"
    );
}

// --- chamfers ---------------------------------------------------------------------

/// The straight edge along Z at the origin of a cube, with the face at x = 0 next to it.
fn z_edge_and_face<K: Kernel>(block: &K::Shape, side: f64) -> (K::Shape, K::Shape) {
    let edge = edge_between::<K>(block, [0.0, 0.0, 0.0], [0.0, 0.0, side]);
    let face = face_at::<K>(block, [0.0, side / 2.0, side / 2.0]);
    (edge, face)
}

/// An equal bevel removes a triangular prism.
///
/// # Panics
/// If the volume differs.
pub fn chamfer_equal_sweep<K: Operations>() {
    let side = 10.0;
    for d in [0.1, 0.5, 1.0, 4.0] {
        let block = cuboid::<K>(side, side, side);
        let (edge, _) = z_edge_and_face::<K>(&block, side);
        let volume = modeled_volume::<K>(
            K::chamfer_edges(&block, &[(&edge, ChamferSize::Equal(d), None)]),
            "chamfer",
        );
        near(
            volume,
            1000.0 - d * d / 2.0 * side,
            MEASURE,
            "chamfered box",
        );
    }
}

/// With two distances the first runs along the reference face, which loses that strip.
///
/// # Panics
/// If the volume or the reference face's area differs.
pub fn chamfer_two_distances_measures_on_the_reference_face<K: Operations>() {
    let side = 10.0;
    for (d1, d2) in [(1.0, 3.0), (3.0, 1.0), (0.5, 4.0), (2.0, 2.0)] {
        let block = cuboid::<K>(side, side, side);
        let (edge, face) = z_edge_and_face::<K>(&block, side);
        let size = ChamferSize::TwoDistances(d1, d2);
        let bevelled = must(
            K::chamfer_edges(&block, &[(&edge, size, Some(&face))]),
            "chamfer",
        );
        assert_valid::<K>(&bevelled.shape, "chamfer");
        volume_is::<K>(&bevelled.shape, 1000.0 - d1 * d2 / 2.0 * side, "chamfer");
        let after = entry::<K>(&bevelled.history.modified, &face);
        near(
            total_area::<K>(after),
            side * (side - d1),
            MEASURE,
            "reference face",
        );
    }
}

/// A distance and an angle from the reference face reach `d tan(angle)` along the other.
///
/// # Panics
/// If the volume or the reference face's area differs.
pub fn chamfer_distance_angle_sweep<K: Operations>() {
    let (side, d) = (10.0, 1.5);
    for degrees in [15.0, 30.0, 45.0, 60.0, 75.0_f64] {
        let block = cuboid::<K>(side, side, side);
        let (edge, face) = z_edge_and_face::<K>(&block, side);
        let size = ChamferSize::DistanceAngle(d, Angle::degrees(degrees));
        let bevelled = must(
            K::chamfer_edges(&block, &[(&edge, size, Some(&face))]),
            "chamfer",
        );
        assert_valid::<K>(&bevelled.shape, "chamfer");
        let other = d * degrees.to_radians().tan();
        volume_is::<K>(&bevelled.shape, 1000.0 - d * other / 2.0 * side, "chamfer");
        let after = entry::<K>(&bevelled.history.modified, &face);
        near(
            total_area::<K>(after),
            side * (side - d),
            MEASURE,
            "reference face",
        );
    }
}

/// Two bevels meeting at a corner remove the union of two prisms, whose overlap is a
/// pyramid of volume `d^3 / 3`.
///
/// # Panics
/// If the volume differs.
pub fn chamfer_two_edges_sharing_a_corner<K: Operations>() {
    let side = 10.0;
    for d in [0.5, 1.0, 2.0] {
        let block = cuboid::<K>(side, side, side);
        let e1 = edge_between::<K>(&block, [0.0, 0.0, 0.0], [0.0, 0.0, side]);
        let e2 = edge_between::<K>(&block, [0.0, 0.0, 0.0], [side, 0.0, 0.0]);
        let volume = modeled_volume::<K>(
            K::chamfer_edges(
                &block,
                &[
                    (&e1, ChamferSize::Equal(d), None),
                    (&e2, ChamferSize::Equal(d), None),
                ],
            ),
            "corner chamfer",
        );
        let removed = 2.0 * (d * d / 2.0 * side) - d * d * d / 3.0;
        near(volume, 1000.0 - removed, MEASURE, "corner chamfer");
    }
}

/// Three bevels at a corner remove at least the union of their prisms, of volume
/// `3 d^2 L / 2 - 3 d^3 / 4`. How the corner closes is the kernel's choice, and at most it
/// takes the rest of the corner tetrahedron, of volume `d^3 / 6`.
///
/// # Panics
/// If the volume is outside those bounds.
pub fn chamfer_three_edges_sharing_a_corner<K: Operations>() {
    let (side, d) = (10.0, 1.0);
    let block = cuboid::<K>(side, side, side);
    let e1 = edge_between::<K>(&block, [0.0, 0.0, 0.0], [0.0, 0.0, side]);
    let e2 = edge_between::<K>(&block, [0.0, 0.0, 0.0], [side, 0.0, 0.0]);
    let e3 = edge_between::<K>(&block, [0.0, 0.0, 0.0], [0.0, side, 0.0]);
    let volume = modeled_volume::<K>(
        K::chamfer_edges(
            &block,
            &[
                (&e1, ChamferSize::Equal(d), None),
                (&e2, ChamferSize::Equal(d), None),
                (&e3, ChamferSize::Equal(d), None),
            ],
        ),
        "three-edge chamfer",
    );
    let removed = 1000.0 - volume;
    let union = 1.5 * d * d * side - 0.75 * d * d * d;
    let slack = MEASURE * 1000.0;
    assert!(
        removed >= union - slack && removed <= union + d * d * d / 6.0 + slack,
        "removed {removed}, the union is {union}"
    );
}

/// Separate edges take separate sizes of different kinds at once.
///
/// # Panics
/// If the volume differs.
pub fn chamfer_with_different_size_per_edge<K: Operations>() {
    let side = 10.0;
    let block = cuboid::<K>(side, side, side);
    let e1 = edge_between::<K>(&block, [0.0, 0.0, 0.0], [0.0, 0.0, side]);
    let f1 = face_at::<K>(&block, [0.0, 5.0, 5.0]);
    let e2 = edge_between::<K>(&block, [side, side, 0.0], [side, side, side]);
    let f2 = face_at::<K>(&block, [side, 5.0, 5.0]);
    let e3 = edge_between::<K>(&block, [side, 0.0, 0.0], [side, 0.0, side]);
    let volume = modeled_volume::<K>(
        K::chamfer_edges(
            &block,
            &[
                (&e1, ChamferSize::TwoDistances(1.0, 2.0), Some(&f1)),
                (
                    &e2,
                    ChamferSize::DistanceAngle(1.0, Angle::degrees(45.0)),
                    Some(&f2),
                ),
                (&e3, ChamferSize::Equal(0.5), None),
            ],
        ),
        "per-edge chamfer",
    );
    let removed = (1.0 + 0.5 + 0.125) * side;
    near(volume, 1000.0 - removed, MEASURE, "per-edge chamfer");
}

/// The bevel made from an edge is in the result, with the width of the slant.
///
/// # Panics
/// If the history says otherwise.
pub fn chamfer_history_generates_the_bevel<K: Operations>() {
    let (side, d) = (10.0, 2.0);
    let block = cuboid::<K>(side, side, side);
    let (edge, _) = z_edge_and_face::<K>(&block, side);
    let bevelled = must(
        K::chamfer_edges(&block, &[(&edge, ChamferSize::Equal(d), None)]),
        "chamfer",
    );
    let result = parts::<K>(&bevelled.shape, ShapeKind::Face);
    let made = entry::<K>(&bevelled.history.generated, &edge);
    assert!(!made.is_empty(), "an edge generates its bevel");
    assert!(
        made.iter().all(|f| contains::<K>(&result, f)),
        "it is in the result"
    );
    near(
        total_area::<K>(made),
        d * 2.0_f64.sqrt() * side,
        MEASURE,
        "bevel area",
    );
}

// --- rejections -------------------------------------------------------------------

/// Edges and faces that aren't where they should be are errors, not ignored.
///
/// # Panics
/// If one is accepted.
pub fn foreign_edges_and_faces_are_rejected<K: Operations>() {
    let side = 10.0;
    let block = cuboid::<K>(side, side, side);
    let other = cuboid::<K>(side, side, side);
    let (edge, face) = z_edge_and_face::<K>(&block, side);
    let (foreign_edge, foreign_face) = z_edge_and_face::<K>(&other, side);
    let two = ChamferSize::TwoDistances(1.0, 2.0);

    assert!(
        K::fillet_edges(&block, &[(&foreign_edge, RadiusLaw::Constant(1.0))]).is_err(),
        "a fillet on another shape's edge"
    );
    assert!(
        K::chamfer_edges(&block, &[(&foreign_edge, ChamferSize::Equal(1.0), None)]).is_err(),
        "a chamfer on another shape's edge"
    );
    assert!(
        K::chamfer_edges(&block, &[(&edge, two, Some(&foreign_face))]).is_err(),
        "a chamfer referring to another shape's face"
    );
    let far_face = face_at::<K>(&block, [side, side / 2.0, side / 2.0]);
    assert!(
        K::chamfer_edges(&block, &[(&edge, two, Some(&far_face))]).is_err(),
        "a chamfer referring to a face away from its edge"
    );
    assert!(K::chamfer_edges(&block, &[(&edge, two, Some(&face))]).is_ok());
}

/// A face given as an edge and an edge given as a face are errors.
///
/// # Panics
/// If one is accepted.
pub fn wrong_kinds_are_rejected<K: Operations>() {
    let side = 10.0;
    let block = cuboid::<K>(side, side, side);
    let (edge, face) = z_edge_and_face::<K>(&block, side);
    assert!(K::fillet_edges(&block, &[(&face, RadiusLaw::Constant(1.0))]).is_err());
    assert!(K::chamfer_edges(&block, &[(&block, ChamferSize::Equal(1.0), None)]).is_err());
    assert!(
        K::chamfer_edges(
            &block,
            &[(&edge, ChamferSize::TwoDistances(1.0, 2.0), Some(&edge))]
        )
        .is_err()
    );
}

/// Asymmetric sizes mean nothing without a side.
///
/// # Panics
/// If one is accepted.
pub fn asymmetric_chamfer_needs_a_reference_face<K: Operations>() {
    let side = 10.0;
    let block = cuboid::<K>(side, side, side);
    let (edge, _) = z_edge_and_face::<K>(&block, side);
    for size in [
        ChamferSize::TwoDistances(1.0, 2.0),
        ChamferSize::DistanceAngle(1.0, Angle::degrees(30.0)),
    ] {
        assert!(K::chamfer_edges(&block, &[(&edge, size, None)]).is_err());
    }
}

/// Blends bigger than the shape allows are errors, and leave no broken shape behind.
///
/// # Panics
/// If one is accepted.
pub fn too_large_blends_are_rejected<K: Operations>() {
    let side = 2.0;
    let block = cuboid::<K>(side, side, side);
    let (edge, face) = z_edge_and_face::<K>(&block, side);
    assert!(K::fillet_edges(&block, &[(&edge, RadiusLaw::Constant(5.0))]).is_err());
    assert!(K::chamfer_edges(&block, &[(&edge, ChamferSize::Equal(5.0), None)]).is_err());
    assert!(
        K::chamfer_edges(
            &block,
            &[(&edge, ChamferSize::TwoDistances(0.5, 5.0), Some(&face))]
        )
        .is_err()
    );
    // A right angle's bevel can't rise square from the reference face.
    assert!(
        K::chamfer_edges(
            &block,
            &[(
                &edge,
                ChamferSize::DistanceAngle(1.0, Angle::degrees(90.0)),
                Some(&face)
            )]
        )
        .is_err()
    );
}
