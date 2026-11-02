//! Questions about shapes, against closed forms and hand-countable topology.

use std::f64::consts::{PI, TAU};

use occt_kernel::{
    Angle, Axis, CancelFlag, CurveKind, Dir3, DrawingLine, DrawingLineKind, DrawingView,
    EdgeConvexity, Error, Frame, Inspection, Kernel, MassProperties, MeshOptions, Point3,
    PointLocation, ShapeKind, ShapeTessellation, SurfaceKind, Vec3,
};

use crate::support::{MEASURE, POSITION, ball_volume, cuboid, must, near, near_point, parts};

/// Relative tolerance for a tessellation of straight-sided geometry, which it reproduces
/// exactly up to the coincidence distance.
const FLAT_MESH: f64 = MEASURE;

/// Relative tolerance for a measure of a mesh of a curved surface at the deflection
/// `CURVED_DEFLECTION`. A chord within a distance `d` of a surface of size 1 to 10
/// changes an area or volume by about `d` relative to size, so 1e-3 plus a margin.
const CURVED_MESH: f64 = 5e-3;

/// The absolute deflection the curved mesh checks ask for.
const CURVED_DEFLECTION: f64 = 1e-3;

/// The absolute deflection the drawing checks ask for.
const DRAWING_DEFLECTION: f64 = 1e-3;

/// Lengths in a drawing are exact for straight lines, and within what chords give up for
/// curves: for a circle of radius `r` and deflection `d`, about `d / r` relative.
const DRAWING_LENGTH: f64 = 1e-2;

/// Every check as a `(name, function)` pair, in a stable order.
#[must_use]
pub fn checks<K: Kernel + Inspection>() -> Vec<(&'static str, fn())> {
    macro_rules! list {
        ($($name:ident),* $(,)?) => { vec![$((stringify!($name), $name::<K> as fn())),*] };
    }
    list![
        mass_properties_of_boxes,
        mass_properties_of_a_cylinder,
        mass_properties_of_a_sphere,
        inertia_has_the_usual_signs,
        principal_axes_are_eigenvectors,
        mass_properties_of_a_face_and_an_edge,
        mass_properties_reject_a_vertex,
        distance_between_separated_shapes,
        distance_of_touching_shapes_is_zero,
        distance_of_overlapping_shapes_is_zero,
        distance_from_a_vertex_to_a_sphere,
        distance_of_a_shape_inside_a_solid_is_zero,
        locate_in_a_box,
        locate_in_a_sphere,
        locate_needs_a_solid,
        ray_through_a_box_hits_in_order,
        ray_faces_belong_to_the_shape,
        ray_stops_at_max_distance,
        ray_that_misses_hits_nothing,
        ray_starting_inside_hits_once,
        ray_through_a_sphere_at_the_right_distances,
        ray_through_a_plate_with_a_hole,
        box_edges_have_two_faces,
        box_vertices_have_three_faces_and_edges,
        ancestors_reach_every_kind,
        ancestors_reject_a_foreign_subshape,
        box_edges_are_convex,
        l_extrusion_has_one_concave_edge,
        fillet_boundary_is_tangent,
        sheet_edges_are_open,
        cylinder_rims_are_convex,
        convexity_rejects_a_foreign_edge,
        hole_face_has_outer_wire_first,
        box_face_wire_has_four_connected_edges,
        wire_edges_chain_on_curved_faces,
        tessellation_faces_follow_face_order,
        tessellation_edges_follow_edge_order,
        tessellation_polylines_meet_the_meshes,
        tessellation_merges_to_the_volume,
        tessellation_normals_face_out,
        tessellation_uvs_are_plane_coordinates,
        tessellation_of_a_lone_edge,
        tessellation_sets_cancelled_when_flagged,
        tessellation_works_after_cancelling,
        box_from_above_draws_its_rectangle,
        box_from_a_corner_draws_nine_and_three,
        drawing_uses_the_view_frame,
        cylinder_from_the_side_has_outlines,
        cylinder_silhouette_shows_from_every_side,
        fillet_draws_a_smooth_line,
        perspective_enlarges_what_is_near,
    ]
}

fn dir(v: [f64; 3]) -> Dir3 {
    Dir3::new(v).expect("a direction with length")
}

fn frame_of(origin: Point3, z: Dir3, x: Vec3) -> Frame {
    Frame::new(origin, z, x).expect("an X that is not along Z")
}

fn offset_box<K: Kernel>(corner: [f64; 3], size: [f64; 3]) -> K::Shape {
    must(K::make_box(corner.into(), size.into()), "make_box")
}

/// A symmetric tensor check, for `want` of a known magnitude.
#[track_caller]
fn near_tensor(got: [[f64; 3]; 3], want: [[f64; 3]; 3], scale: f64, what: &str) {
    for row in 0..3 {
        for column in 0..3 {
            assert!(
                (got[row][column] - want[row][column]).abs() <= MEASURE * scale,
                "{what}: inertia[{row}][{column}] is {}, want {} (matrices {got:?}, {want:?})",
                got[row][column],
                want[row][column],
            );
        }
    }
}

fn box_tensor(mass: f64, [x, y, z]: [f64; 3]) -> [[f64; 3]; 3] {
    let moment = |a: f64, b: f64| mass * (a * a + b * b) / 12.0;
    [
        [moment(y, z), 0.0, 0.0],
        [0.0, moment(x, z), 0.0],
        [0.0, 0.0, moment(x, y)],
    ]
}

/// A box's inertia is `m (a² + b²) / 12` about each central axis, wherever the box is.
///
/// # Panics
/// If the mass, centre or inertia differs.
pub fn mass_properties_of_boxes<K: Kernel + Inspection>() {
    for (corner, size) in [
        ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
        ([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]),
        ([-5.0, 3.0, 7.0], [4.0, 0.5, 2.5]),
        ([100.0, -40.0, 0.0], [10.0, 20.0, 30.0]),
    ] {
        let shape = offset_box::<K>(corner, size);
        let props = must(K::mass_properties(&shape), "mass_properties");
        let mass = size[0] * size[1] * size[2];
        near(props.mass, mass, MEASURE, "box mass");
        near_point(
            props.center,
            Point3::new(
                corner[0] + size[0] / 2.0,
                corner[1] + size[1] / 2.0,
                corner[2] + size[2] / 2.0,
            ),
            "box centre",
        );
        let want = box_tensor(mass, size);
        let scale = want[0][0].max(want[1][1]).max(want[2][2]);
        near_tensor(props.inertia, want, scale, "box");
    }
}

/// A cylinder's inertia is `m r² / 2` about its axis and `m (3 r² + h²) / 12` across it.
///
/// # Panics
/// If the mass, centre or inertia differs.
pub fn mass_properties_of_a_cylinder<K: Kernel + Inspection>() {
    for (r, h) in [(1.0, 1.0), (2.0, 5.0), (5.0, 0.5)] {
        let shape = must(K::make_cylinder(Axis::Z, r, h), "make_cylinder");
        let props = must(K::mass_properties(&shape), "mass_properties");
        let mass = PI * r * r * h;
        near(props.mass, mass, MEASURE, "cylinder mass");
        near_point(
            props.center,
            Point3::new(0.0, 0.0, h / 2.0),
            "cylinder centre",
        );
        let across = mass * (3.0 * r * r + h * h) / 12.0;
        let around = mass * r * r / 2.0;
        near_tensor(
            props.inertia,
            [[across, 0.0, 0.0], [0.0, across, 0.0], [0.0, 0.0, around]],
            across.max(around),
            "cylinder",
        );
    }
}

/// A ball's inertia is `2 m r² / 5` about every axis, so any orthonormal axes are
/// principal, and all three moments agree.
///
/// # Panics
/// If the mass, moments or axes differ.
pub fn mass_properties_of_a_sphere<K: Kernel + Inspection>() {
    let r = 3.0;
    let shape = must(K::make_sphere(Point3::new(1.0, 2.0, 3.0), r), "make_sphere");
    let props = must(K::mass_properties(&shape), "mass_properties");
    near(props.mass, ball_volume(r), MEASURE, "ball mass");
    let moment = 2.0 * props.mass * r * r / 5.0;
    for m in props.principal_moments {
        near(m, moment, MEASURE, "ball moment");
    }
    assert_orthonormal(&props, "ball");
}

/// A box turned by `a` about Z has the tensor `R I Rᵀ`, whose XY term is
/// `(Ixx - Iyy) sin a cos a` with the usual sign convention.
///
/// # Panics
/// If the tensor differs.
pub fn inertia_has_the_usual_signs<K: Kernel + Inspection>() {
    let size = [4.0, 1.0, 2.0];
    let shape = offset_box::<K>([-2.0, -0.5, -1.0], size);
    for degrees in [30.0, -30.0, 75.0] {
        let turn = Angle::degrees(degrees);
        let turned = must(K::rotate(&shape, Axis::Z, turn), "rotate");
        let props = must(K::mass_properties(&turned), "mass_properties");
        let i = box_tensor(8.0, size);
        let (s, c) = turn.as_radians().sin_cos();
        let (ixx, iyy) = (i[0][0], i[1][1]);
        let want = [
            [ixx * c * c + iyy * s * s, (ixx - iyy) * s * c, 0.0],
            [(ixx - iyy) * s * c, ixx * s * s + iyy * c * c, 0.0],
            [0.0, 0.0, i[2][2]],
        ];
        near_tensor(props.inertia, want, ixx.max(iyy), "turned box");
    }
}

#[track_caller]
fn assert_orthonormal(props: &MassProperties, what: &str) {
    let axes = props.principal_axes;
    for (i, a) in axes.iter().enumerate() {
        near(
            a.length(),
            1.0,
            MEASURE,
            format_args!("{what}: axis {i} length"),
        );
        for b in &axes[i + 1..] {
            assert!(
                a.dot(*b).abs() <= MEASURE,
                "{what}: axes {a:?} and {b:?} aren't perpendicular"
            );
        }
    }
}

/// `I a = m a` for each principal axis `a` and moment `m`, and a turned box's axes run
/// along its sides.
///
/// # Panics
/// If an axis isn't an eigenvector or points elsewhere.
pub fn principal_axes_are_eigenvectors<K: Kernel + Inspection>() {
    let size = [4.0, 1.0, 2.0];
    let shape = offset_box::<K>([-2.0, -0.5, -1.0], size);
    let axis = Axis::new(Point3::ORIGIN, dir([1.0, 2.0, 3.0]));
    let turn = Angle::degrees(40.0);
    let turned = must(K::rotate(&shape, axis, turn), "rotate");
    let props = must(K::mass_properties(&turned), "mass_properties");
    assert_orthonormal(&props, "turned box");

    let i = box_tensor(8.0, size);
    let scale = i[0][0].max(i[1][1]).max(i[2][2]);
    for (a, m) in props.principal_axes.iter().zip(props.principal_moments) {
        let v = [a.x, a.y, a.z];
        for (row, value) in props.inertia.iter().zip(v) {
            let product: f64 = row.iter().zip(v).map(|(x, y)| x * y).sum();
            assert!(
                (product - m * value).abs() <= MEASURE * scale,
                "axis {a:?} with moment {m} isn't an eigenvector of {:?}",
                props.inertia
            );
        }
    }

    // The sides have distinct moments, so each is the axis with its moment, up to sign.
    let side_x = occt_kernel::Transform::rotation(axis, turn).vector(Vec3::new(1.0, 0.0, 0.0));
    let smallest = props
        .principal_moments
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(index, _)| index);
    near(
        props.principal_moments[smallest],
        i[0][0],
        MEASURE,
        "smallest moment, around the longest side",
    );
    near(
        props.principal_axes[smallest].dot(side_x).abs(),
        1.0,
        MEASURE,
        "axis of the smallest moment",
    );
}

/// With no volume, a shape's mass is its area, and with no area its length. A square's
/// second moment across its middle is `m a² / 12`, and a rod's is `m l² / 12`.
///
/// # Panics
/// If a measure differs.
pub fn mass_properties_of_a_face_and_an_edge<K: Kernel + Inspection>() {
    let a = 2.0;
    let square = must(
        K::make_polygon_face(&[
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(a, 0.0, 0.0),
            Point3::new(a, a, 0.0),
            Point3::new(0.0, a, 0.0),
        ]),
        "make_polygon_face",
    );
    let props = must(K::mass_properties(&square), "mass_properties");
    near(props.mass, a * a, MEASURE, "square mass");
    near_point(props.center, Point3::new(1.0, 1.0, 0.0), "square centre");
    let across = props.mass * a * a / 12.0;
    near(props.inertia[0][0], across, MEASURE, "square inertia");
    near(
        props.inertia[2][2],
        2.0 * across,
        MEASURE,
        "square inertia about its normal",
    );

    let plate = cuboid::<K>(3.0, 3.0, 3.0);
    let edge = parts::<K>(&plate, ShapeKind::Edge).swap_remove(0);
    let props = must(K::mass_properties(&edge), "mass_properties");
    near(props.mass, 3.0, MEASURE, "edge mass");
    let moments = props.principal_moments;
    let largest = moments.iter().copied().fold(f64::MIN, f64::max);
    near(largest, props.mass * 9.0 / 12.0, MEASURE, "edge moment");
}

/// A point has nothing to weigh.
///
/// # Panics
/// If a vertex gets mass properties.
pub fn mass_properties_reject_a_vertex<K: Kernel + Inspection>() {
    let vertex = parts::<K>(&cuboid::<K>(1.0, 1.0, 1.0), ShapeKind::Vertex).swap_remove(0);
    assert!(K::mass_properties(&vertex).is_err());
}

/// Boxes along an axis, with closest points on the faces that look at each other.
///
/// # Panics
/// If the distance or a closest point differs.
pub fn distance_between_separated_shapes<K: Kernel + Inspection>() {
    for gap in [0.5, 1.0, 3.0, 25.0] {
        let a = cuboid::<K>(1.0, 1.0, 1.0);
        let b = offset_box::<K>([1.0 + gap, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let d = must(K::distance(&a, &b), "distance");
        assert!((d.value - gap).abs() <= POSITION, "gap {gap}: {d:?}");
        assert!((d.on_a.x - 1.0).abs() <= POSITION, "gap {gap}: {d:?}");
        assert!(
            (d.on_b.x - (1.0 + gap)).abs() <= POSITION,
            "gap {gap}: {d:?}"
        );
        assert!((d.on_a.distance(d.on_b) - gap).abs() <= POSITION);

        let d = must(K::distance(&b, &a), "distance");
        assert!((d.value - gap).abs() <= POSITION, "reversed: {d:?}");
        assert!(
            (d.on_a.x - (1.0 + gap)).abs() <= POSITION,
            "on_a stays on the first: {d:?}"
        );
    }

    // Corner to corner, along the diagonal.
    let a = cuboid::<K>(1.0, 1.0, 1.0);
    let b = offset_box::<K>([2.0, 2.0, 2.0], [1.0, 1.0, 1.0]);
    let d = must(K::distance(&a, &b), "distance");
    assert!((d.value - 3f64.sqrt()).abs() <= POSITION, "{d:?}");
    near_point(d.on_a, Point3::new(1.0, 1.0, 1.0), "corner on the first");
    near_point(d.on_b, Point3::new(2.0, 2.0, 2.0), "corner on the second");
}

/// Boxes sharing a face, an edge or a corner touch, and spheres tangent at one point do.
///
/// # Panics
/// If a distance isn't zero.
pub fn distance_of_touching_shapes_is_zero<K: Kernel + Inspection>() {
    let a = cuboid::<K>(1.0, 1.0, 1.0);
    for corner in [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]] {
        let b = offset_box::<K>(corner, [1.0, 1.0, 1.0]);
        let d = must(K::distance(&a, &b), "distance");
        assert!(d.value.abs() <= POSITION, "touching at {corner:?}: {d:?}");
    }
    let s1 = must(K::make_sphere(Point3::ORIGIN, 1.0), "make_sphere");
    let s2 = must(
        K::make_sphere(Point3::new(3.0, 0.0, 0.0), 2.0),
        "make_sphere",
    );
    let d = must(K::distance(&s1, &s2), "distance");
    assert!(d.value.abs() <= POSITION, "tangent spheres: {d:?}");
    near_point(d.on_a, Point3::new(1.0, 0.0, 0.0), "tangent point");
}

/// A shape overlaps itself and any box crossing it.
///
/// # Panics
/// If a distance isn't zero.
pub fn distance_of_overlapping_shapes_is_zero<K: Kernel + Inspection>() {
    let a = cuboid::<K>(2.0, 2.0, 2.0);
    let b = offset_box::<K>([1.0, 1.0, 1.0], [2.0, 2.0, 2.0]);
    let d = must(K::distance(&a, &b), "distance");
    assert!(d.value.abs() <= POSITION, "overlapping boxes: {d:?}");
    let d = must(K::distance(&a, &a), "distance");
    assert!(d.value.abs() <= POSITION, "a box and itself: {d:?}");
}

/// From a point outside a ball, the nearest surface point is on the line to the centre.
///
/// # Panics
/// If the distance or closest point differs.
pub fn distance_from_a_vertex_to_a_sphere<K: Kernel + Inspection>() {
    let r = 2.0;
    let ball = must(K::make_sphere(Point3::ORIGIN, r), "make_sphere");
    // Any box vertex will do as the point, at its known position.
    let corner = offset_box::<K>([4.0, 4.0, 2.0], [1.0, 1.0, 1.0]);
    for vertex in parts::<K>(&corner, ShapeKind::Vertex) {
        let p = must(K::vertex_point(&vertex), "vertex_point");
        let from_center = p.distance(Point3::ORIGIN);
        let d = must(K::distance(&vertex, &ball), "distance");
        assert!(
            (d.value - (from_center - r)).abs() <= POSITION,
            "from {p:?}: {d:?}"
        );
        near_point(d.on_a, p, "the vertex itself");
        let unit = (p - Point3::ORIGIN) / from_center;
        near_point(d.on_b, Point3::ORIGIN + unit * r, "nearest on the ball");
    }
}

/// A solid is all of its volume, so what lies inside is at distance zero, even though it
/// is far from the surface.
///
/// # Panics
/// If the distance isn't zero.
pub fn distance_of_a_shape_inside_a_solid_is_zero<K: Kernel + Inspection>() {
    let outer = cuboid::<K>(10.0, 10.0, 10.0);
    let inner = offset_box::<K>([4.0, 4.0, 4.0], [1.0, 1.0, 1.0]);
    for (a, b) in [(&outer, &inner), (&inner, &outer)] {
        let d = must(K::distance(a, b), "distance");
        assert!(d.value.abs() <= POSITION, "{d:?}");
        assert!(d.on_a.distance(d.on_b) <= POSITION, "{d:?}");
    }
}

#[track_caller]
fn assert_located<K: Kernel + Inspection>(solid: &K::Shape, point: [f64; 3], want: PointLocation) {
    let got = must(K::locate_point(solid, point.into(), 1e-6), "locate_point");
    assert_eq!(got, want, "at {point:?}");
}

/// The middle is inside, faces, edges and corners are on the boundary, and past a face is
/// outside, including points just beyond the tolerance.
///
/// # Panics
/// If a point is misplaced.
pub fn locate_in_a_box<K: Kernel + Inspection>() {
    use PointLocation::{Inside, OnBoundary, Outside};
    let b = cuboid::<K>(2.0, 2.0, 2.0);
    for (p, want) in [
        ([1.0, 1.0, 1.0], Inside),
        ([0.1, 1.9, 1.0], Inside),
        ([2.0, 1.0, 1.0], OnBoundary),
        ([1.0, 0.0, 1.0], OnBoundary),
        ([2.0, 2.0, 1.0], OnBoundary),
        ([0.0, 0.0, 0.0], OnBoundary),
        ([3.0, 1.0, 1.0], Outside),
        ([-0.001, 1.0, 1.0], Outside),
        ([5.0, 5.0, 5.0], Outside),
    ] {
        assert_located::<K>(&b, p, want);
    }
}

/// A ball's classification is its radius, at any angle.
///
/// # Panics
/// If a point is misplaced.
pub fn locate_in_a_sphere<K: Kernel + Inspection>() {
    use PointLocation::{Inside, OnBoundary, Outside};
    let r = 2.0;
    let ball = must(K::make_sphere(Point3::ORIGIN, r), "make_sphere");
    for i in 0..12 {
        let angle = f64::from(i) * TAU / 12.0 + 0.1;
        let (s, c) = angle.sin_cos();
        let unit = [c * 0.8, s * 0.8, 0.6];
        let at = |radius: f64| unit.map(|v| v * radius);
        assert_located::<K>(&ball, at(r - 0.01), Inside);
        assert_located::<K>(&ball, at(r), OnBoundary);
        assert_located::<K>(&ball, at(r + 0.01), Outside);
    }
}

/// A face has no inside.
///
/// # Panics
/// If a face is accepted.
pub fn locate_needs_a_solid<K: Kernel + Inspection>() {
    let face = parts::<K>(&cuboid::<K>(1.0, 1.0, 1.0), ShapeKind::Face).swap_remove(0);
    assert!(K::locate_point(&face, Point3::ORIGIN, 1e-6).is_err());
}

fn position_of<K: Kernel>(shape: &K::Shape, list: &[K::Shape]) -> Option<usize> {
    list.iter().position(|other| K::is_same(shape, other))
}

/// A ray through two opposite faces, from each direction, at the expected distances.
///
/// # Panics
/// If a hit is missing or out of order.
pub fn ray_through_a_box_hits_in_order<K: Kernel + Inspection>() {
    let b = cuboid::<K>(2.0, 3.0, 4.0);
    for (origin, direction, near_hit, far_hit) in [
        ([1.0, 1.0, -5.0], Dir3::Z, 5.0, 9.0),
        ([-1.0, 1.0, 1.0], Dir3::X, 1.0, 3.0),
        ([1.0, 8.0, 1.0], dir([0.0, -1.0, 0.0]), 5.0, 8.0),
    ] {
        let hits = must(K::ray_cast(&b, origin.into(), direction, None), "ray_cast");
        assert_eq!(hits.len(), 2, "from {origin:?}: {hits:?}");
        assert!((hits[0].distance - near_hit).abs() <= POSITION, "{hits:?}");
        assert!((hits[1].distance - far_hit).abs() <= POSITION, "{hits:?}");
        let start = Point3::from(origin);
        near_point(
            hits[0].point,
            start + direction.vector() * near_hit,
            "first hit",
        );
        near_point(
            hits[1].point,
            start + direction.vector() * far_hit,
            "second hit",
        );
        assert!(
            !K::is_same(&hits[0].face, &hits[1].face),
            "two distinct faces"
        );
    }
}

/// The face hit is one the shape's traversal gives, and the right one: its centre has the
/// hit's coordinate along the ray.
///
/// # Panics
/// If a hit's face isn't the shape's, or isn't where the hit is.
pub fn ray_faces_belong_to_the_shape<K: Kernel + Inspection>() {
    let b = cuboid::<K>(2.0, 2.0, 2.0);
    let faces = parts::<K>(&b, ShapeKind::Face);
    for (origin, direction) in [
        ([1.0, 1.0, -3.0], Dir3::Z),
        ([-3.0, 1.0, 1.0], Dir3::X),
        ([1.0, -3.0, 1.0], Dir3::Y),
    ] {
        let hits = must(K::ray_cast(&b, origin.into(), direction, None), "ray_cast");
        assert_eq!(hits.len(), 2);
        for hit in &hits {
            let index = position_of::<K>(&hit.face, &faces)
                .unwrap_or_else(|| panic!("{:?} isn't a face of the shape", hit.face));
            let center = must(K::center_of_mass(&faces[index]), "center_of_mass");
            let along = (hit.point - center).dot(direction.vector());
            assert!(
                along.abs() <= POSITION,
                "hit {:?} on face at {center:?}",
                hit.point
            );
        }
    }
}

/// The limit keeps what's nearer than it and no more.
///
/// # Panics
/// If the hit count differs.
pub fn ray_stops_at_max_distance<K: Kernel + Inspection>() {
    let b = cuboid::<K>(2.0, 2.0, 2.0);
    let origin = Point3::new(1.0, 1.0, -5.0);
    for (limit, want) in [(1.0, 0), (5.5, 1), (6.9, 1), (7.5, 2), (100.0, 2)] {
        let hits = must(K::ray_cast(&b, origin, Dir3::Z, Some(limit)), "ray_cast");
        assert_eq!(hits.len(), want, "limit {limit}: {hits:?}");
    }
}

/// Rays that pass by, or point away, hit nothing, without that being an error.
///
/// # Panics
/// If a ray hits something or fails.
pub fn ray_that_misses_hits_nothing<K: Kernel + Inspection>() {
    let b = cuboid::<K>(2.0, 2.0, 2.0);
    let away = dir([0.0, 0.0, -1.0]);
    for (origin, direction) in [
        ([5.0, 5.0, -5.0], Dir3::Z),
        ([1.0, 1.0, -5.0], away),
        ([1.0, 1.0, 5.0], Dir3::Z),
    ] {
        let hits = must(K::ray_cast(&b, origin.into(), direction, None), "ray_cast");
        assert!(hits.is_empty(), "from {origin:?}: {hits:?}");
    }
}

/// From inside a closed solid, a ray leaves through one face.
///
/// # Panics
/// If the hits differ.
pub fn ray_starting_inside_hits_once<K: Kernel + Inspection>() {
    let b = cuboid::<K>(2.0, 2.0, 2.0);
    let hits = must(
        K::ray_cast(&b, Point3::new(1.0, 1.0, 1.0), Dir3::Z, None),
        "ray_cast",
    );
    assert_eq!(hits.len(), 1, "{hits:?}");
    assert!((hits[0].distance - 1.0).abs() <= POSITION, "{hits:?}");
}

/// A chord of a ball at offset `h` from the centre starts `sqrt(r² - h²)` before the
/// closest approach and ends as far after, for rays from all around, and on any seam.
///
/// # Panics
/// If the distances differ.
pub fn ray_through_a_sphere_at_the_right_distances<K: Kernel + Inspection>() {
    let r = 3.0;
    let center = Point3::new(1.0, -2.0, 0.5);
    let ball = must(K::make_sphere(center, r), "make_sphere");
    let reach = 10.0;
    for i in 0..16 {
        let angle = f64::from(i) * TAU / 16.0;
        let (s, c) = angle.sin_cos();
        for h in [0.0, 1.0, 2.5] {
            let direction = dir([c, s, 0.0]);
            let sideways = Vec3::new(-s, c, 0.0);
            let origin = center + sideways * h - direction.vector() * reach;
            let hits = must(K::ray_cast(&ball, origin, direction, None), "ray_cast");
            assert_eq!(hits.len(), 2, "angle {angle}, offset {h}: {hits:?}");
            let half = (r * r - h * h).sqrt();
            assert!(
                (hits[0].distance - (reach - half)).abs() <= POSITION,
                "{hits:?}"
            );
            assert!(
                (hits[1].distance - (reach + half)).abs() <= POSITION,
                "{hits:?}"
            );
            assert!(
                (hits[0].point.distance(center) - r).abs() <= POSITION,
                "{hits:?}"
            );
        }
    }
}

fn plate_with_hole<K: Kernel>() -> K::Shape {
    let plate = cuboid::<K>(10.0, 8.0, 2.0);
    let axis = Axis::new(Point3::new(5.0, 4.0, -1.0), Dir3::Z);
    let drill = must(K::make_cylinder(axis, 1.5, 4.0), "make_cylinder");
    must(K::cut(&plate, &drill), "cut")
}

/// Across a holed plate through its bore, a ray crosses the two sides and the bore's
/// wall twice; down the bore's axis, it hits nothing.
///
/// # Panics
/// If the hits differ.
pub fn ray_through_a_plate_with_a_hole<K: Kernel + Inspection>() {
    let plate = plate_with_hole::<K>();
    let hits = must(
        K::ray_cast(&plate, Point3::new(-1.0, 4.0, 1.0), Dir3::X, None),
        "ray_cast",
    );
    let distances: Vec<f64> = hits.iter().map(|h| h.distance).collect();
    let want = [1.0, 4.5, 7.5, 11.0];
    assert_eq!(distances.len(), want.len(), "{distances:?}");
    for (got, want) in distances.iter().zip(want) {
        assert!((got - want).abs() <= POSITION, "{distances:?}");
    }
    assert!(
        distances.windows(2).all(|d| d[0] <= d[1]),
        "nearest first: {distances:?}"
    );
    let down = must(
        K::ray_cast(
            &plate,
            Point3::new(5.0, 4.0, 5.0),
            dir([0.0, 0.0, -1.0]),
            None,
        ),
        "ray_cast",
    );
    assert!(down.is_empty(), "down the bore: {down:?}");
}

/// Each of a box's 12 edges is on two faces, the same ones `subshapes` gives.
///
/// # Panics
/// If the adjacency differs.
pub fn box_edges_have_two_faces<K: Kernel + Inspection>() {
    let b = cuboid::<K>(1.0, 2.0, 3.0);
    let faces = parts::<K>(&b, ShapeKind::Face);
    for edge in parts::<K>(&b, ShapeKind::Edge) {
        let found = must(K::ancestors(&b, &edge, ShapeKind::Face), "ancestors");
        assert_eq!(found.len(), 2, "faces of an edge");
        assert!(!K::is_same(&found[0], &found[1]), "two different faces");
        for face in &found {
            assert!(
                position_of::<K>(face, &faces).is_some(),
                "the box's own face"
            );
            let edges = parts::<K>(face, ShapeKind::Edge);
            assert!(
                position_of::<K>(&edge, &edges).is_some(),
                "the face has the edge"
            );
        }
    }
}

/// Each of a box's 8 vertices is on three faces and three edges.
///
/// # Panics
/// If the adjacency differs.
pub fn box_vertices_have_three_faces_and_edges<K: Kernel + Inspection>() {
    let b = cuboid::<K>(1.0, 2.0, 3.0);
    for vertex in parts::<K>(&b, ShapeKind::Vertex) {
        let faces = must(K::ancestors(&b, &vertex, ShapeKind::Face), "ancestors");
        assert_eq!(faces.len(), 3, "faces of a vertex");
        let edges = must(K::ancestors(&b, &vertex, ShapeKind::Edge), "ancestors");
        assert_eq!(edges.len(), 3, "edges of a vertex");
    }
}

/// A face belongs to one shell and one solid, and asking for ancestors of a lesser kind
/// returns nothing rather than failing.
///
/// # Panics
/// If the adjacency differs.
pub fn ancestors_reach_every_kind<K: Kernel + Inspection>() {
    let b = cuboid::<K>(1.0, 2.0, 3.0);
    let face = parts::<K>(&b, ShapeKind::Face).swap_remove(0);
    let solids = must(K::ancestors(&b, &face, ShapeKind::Solid), "ancestors");
    assert_eq!(solids.len(), 1);
    assert!(K::is_same(&solids[0], &b));
    assert_eq!(
        must(K::ancestors(&b, &face, ShapeKind::Shell), "ancestors").len(),
        1
    );
    assert!(must(K::ancestors(&b, &face, ShapeKind::Edge), "ancestors").is_empty());
}

/// What isn't part of the shape has no place in it.
///
/// # Panics
/// If a foreign subshape is accepted.
pub fn ancestors_reject_a_foreign_subshape<K: Kernel + Inspection>() {
    let b = cuboid::<K>(1.0, 2.0, 3.0);
    let other = cuboid::<K>(1.0, 2.0, 3.0);
    let foreign = parts::<K>(&other, ShapeKind::Edge).swap_remove(0);
    assert!(K::ancestors(&b, &foreign, ShapeKind::Face).is_err());
}

/// Every edge of a box has a material angle of 90 degrees.
///
/// # Panics
/// If an edge differs.
pub fn box_edges_are_convex<K: Kernel + Inspection>() {
    let b = cuboid::<K>(1.0, 2.0, 3.0);
    for edge in parts::<K>(&b, ShapeKind::Edge) {
        assert_eq!(
            must(K::edge_convexity(&b, &edge), "edge_convexity"),
            EdgeConvexity::Convex
        );
    }
}

/// An L profile extruded has one reflex corner, along its inner edge, and nothing else.
///
/// # Panics
/// If the classification differs.
pub fn l_extrusion_has_one_concave_edge<K: Kernel + Inspection>() {
    let profile = must(
        K::make_polygon_face(&[
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
            Point3::new(2.0, 1.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(1.0, 2.0, 0.0),
            Point3::new(0.0, 2.0, 0.0),
        ]),
        "make_polygon_face",
    );
    let solid = must(K::extrude(&profile, Vec3::new(0.0, 0.0, 3.0)), "extrude");
    let mut concave = 0;
    for edge in parts::<K>(&solid, ShapeKind::Edge) {
        let got = must(K::edge_convexity(&solid, &edge), "edge_convexity");
        let extent = must(K::bounding_box(&edge), "bounding_box");
        let inner = (extent.min.x - 1.0).abs() < POSITION
            && (extent.min.y - 1.0).abs() < POSITION
            && (extent.size().z - 3.0).abs() < POSITION;
        if inner {
            concave += 1;
            assert_eq!(
                got,
                EdgeConvexity::Concave,
                "the inner corner at {extent:?}"
            );
        } else {
            assert_eq!(got, EdgeConvexity::Convex, "edge at {extent:?}");
        }
    }
    assert_eq!(concave, 1);
}

/// A fillet meets its neighbours smoothly along its long sides, and sharply at its ends.
///
/// # Panics
/// If the classification differs.
pub fn fillet_boundary_is_tangent<K: Kernel + Inspection>() {
    let b = cuboid::<K>(4.0, 4.0, 4.0);
    let edge = parts::<K>(&b, ShapeKind::Edge).swap_remove(0);
    let rounded = must(K::fillet(&b, &[&edge], 1.0), "fillet");
    let round = parts::<K>(&rounded, ShapeKind::Face)
        .into_iter()
        .find(|f| must(K::surface_kind(f), "surface_kind") == SurfaceKind::Cylinder)
        .expect("a fillet on a straight edge has a cylindrical face");
    let (mut tangent, mut sharp) = (0, 0);
    for e in parts::<K>(&round, ShapeKind::Edge) {
        let got = must(K::edge_convexity(&rounded, &e), "edge_convexity");
        if must(K::curve_kind(&e), "curve_kind") == CurveKind::Line {
            assert_eq!(got, EdgeConvexity::Tangent, "a fillet's long side");
            tangent += 1;
        } else {
            assert_eq!(got, EdgeConvexity::Convex, "a fillet's end");
            sharp += 1;
        }
    }
    assert_eq!((tangent, sharp), (2, 2));
}

/// The edges of a lone face border nothing else.
///
/// # Panics
/// If an edge isn't open.
pub fn sheet_edges_are_open<K: Kernel + Inspection>() {
    let sheet = must(
        K::make_polygon_face(&[
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ]),
        "make_polygon_face",
    );
    for edge in parts::<K>(&sheet, ShapeKind::Edge) {
        assert_eq!(
            must(K::edge_convexity(&sheet, &edge), "edge_convexity"),
            EdgeConvexity::Open
        );
    }
}

/// Where a cylinder's cap meets its side is convex; any edge running along the side, as
/// a seam, has that one surface on both sides.
///
/// # Panics
/// If the classification differs.
pub fn cylinder_rims_are_convex<K: Kernel + Inspection>() {
    let c = must(K::make_cylinder(Axis::Z, 2.0, 5.0), "make_cylinder");
    let mut rims = 0;
    for edge in parts::<K>(&c, ShapeKind::Edge) {
        let got = must(K::edge_convexity(&c, &edge), "edge_convexity");
        if must(K::curve_kind(&edge), "curve_kind") == CurveKind::Circle {
            assert_eq!(got, EdgeConvexity::Convex, "a rim");
            rims += 1;
        } else {
            assert!(
                matches!(got, EdgeConvexity::Open | EdgeConvexity::Tangent),
                "along the side: {got:?}"
            );
        }
    }
    assert_eq!(rims, 2);
}

/// An edge of another shape has no faces in this one.
///
/// # Panics
/// If a foreign edge is accepted.
pub fn convexity_rejects_a_foreign_edge<K: Kernel + Inspection>() {
    let b = cuboid::<K>(1.0, 1.0, 1.0);
    let other = cuboid::<K>(1.0, 1.0, 1.0);
    let foreign = parts::<K>(&other, ShapeKind::Edge).swap_remove(0);
    assert!(K::edge_convexity(&b, &foreign).is_err());
}

/// The top of a plate with a hole has the rectangle first and the circle second.
///
/// # Panics
/// If a wire differs.
pub fn hole_face_has_outer_wire_first<K: Kernel + Inspection>() {
    let plate = plate_with_hole::<K>();
    let top = parts::<K>(&plate, ShapeKind::Face)
        .into_iter()
        .find(|f| {
            let center = must(K::center_of_mass(f), "center_of_mass");
            (center.z - 2.0).abs() < POSITION
        })
        .expect("the plate has a top");
    let wires = must(K::face_wires(&top), "face_wires");
    assert_eq!(wires.len(), 2);
    for wire in &wires {
        assert_eq!(must(K::kind(wire), "kind"), ShapeKind::Wire);
    }
    let outer = must(K::bounding_box(&wires[0]), "bounding_box").size();
    assert!(
        (outer.x - 10.0).abs() < POSITION && (outer.y - 8.0).abs() < POSITION,
        "{outer:?}"
    );
    let hole = must(K::bounding_box(&wires[1]), "bounding_box").size();
    assert!(
        (hole.x - 3.0).abs() < 1e-3 && (hole.y - 3.0).abs() < 1e-3,
        "{hole:?}"
    );
}

/// Consecutive edges share a vertex, and a closed wire's last meets its first.
#[track_caller]
fn assert_chained<K: Kernel + Inspection>(wire: &K::Shape, closed: bool) -> Vec<K::Shape> {
    let edges = must(K::wire_edges(wire), "wire_edges");
    assert!(!edges.is_empty());
    let meets = |a: &K::Shape, b: &K::Shape| {
        let theirs = parts::<K>(b, ShapeKind::Vertex);
        parts::<K>(a, ShapeKind::Vertex)
            .iter()
            .any(|v| position_of::<K>(v, &theirs).is_some())
    };
    for pair in edges.windows(2) {
        assert!(meets(&pair[0], &pair[1]), "consecutive edges don't meet");
    }
    if closed {
        assert!(
            meets(&edges[edges.len() - 1], &edges[0]),
            "a closed wire's ends don't meet"
        );
    }
    edges
}

/// A box's face is bounded by one wire of four edges, one after another, and each edge
/// is the face's own.
///
/// # Panics
/// If the wire differs.
pub fn box_face_wire_has_four_connected_edges<K: Kernel + Inspection>() {
    let b = cuboid::<K>(1.0, 2.0, 3.0);
    for face in parts::<K>(&b, ShapeKind::Face) {
        let wires = must(K::face_wires(&face), "face_wires");
        assert_eq!(wires.len(), 1);
        let edges = assert_chained::<K>(&wires[0], true);
        assert_eq!(edges.len(), 4);
        let own = parts::<K>(&face, ShapeKind::Edge);
        for edge in &edges {
            assert!(
                position_of::<K>(edge, &own).is_some(),
                "the wire's edge is the face's"
            );
        }
    }
}

/// Curved faces have awkward boundaries, where one vertex can start several edges. Order
/// must hold there too, and account for all of the faces' edges.
///
/// # Panics
/// If a wire isn't a chain.
pub fn wire_edges_chain_on_curved_faces<K: Kernel + Inspection>() {
    for shape in [
        must(K::make_cylinder(Axis::Z, 2.0, 5.0), "make_cylinder"),
        must(K::make_cone(Axis::Z, 2.0, 0.0, 5.0), "make_cone"),
        must(K::make_sphere(Point3::ORIGIN, 2.0), "make_sphere"),
    ] {
        for face in parts::<K>(&shape, ShapeKind::Face) {
            let own = parts::<K>(&face, ShapeKind::Edge);
            let mut seen = Vec::new();
            for wire in must(K::face_wires(&face), "face_wires") {
                seen.extend(assert_chained::<K>(&wire, true));
            }
            for edge in &own {
                assert!(
                    position_of::<K>(edge, &seen).is_some(),
                    "a face's edge is missing"
                );
            }
        }
    }
}

fn fine() -> MeshOptions {
    MeshOptions::default()
        .absolute(CURVED_DEFLECTION)
        .angular(Angle::degrees(10.0))
}

fn tessellation<K: Kernel + Inspection>(shape: &K::Shape) -> ShapeTessellation {
    must(K::tessellate(shape, &fine(), None), "tessellate")
}

/// The sample solids of these checks: straight-sided, curved, and with an inner boundary.
fn sample_solids<K: Kernel>() -> Vec<(&'static str, K::Shape, bool)> {
    vec![
        ("box", cuboid::<K>(1.0, 2.0, 3.0), false),
        (
            "cylinder",
            must(K::make_cylinder(Axis::Z, 2.0, 5.0), "make_cylinder"),
            true,
        ),
        (
            "cone",
            must(K::make_cone(Axis::Z, 2.0, 0.0, 5.0), "make_cone"),
            true,
        ),
        (
            "sphere",
            must(K::make_sphere(Point3::ORIGIN, 2.0), "make_sphere"),
            true,
        ),
        ("plate with a hole", plate_with_hole::<K>(), true),
    ]
}

/// `faces[i]` has the area and centre of the `i`th face.
///
/// # Panics
/// If a mesh doesn't match its face.
pub fn tessellation_faces_follow_face_order<K: Kernel + Inspection>() {
    for (name, shape, curved) in sample_solids::<K>() {
        let faces = parts::<K>(&shape, ShapeKind::Face);
        let t = tessellation::<K>(&shape);
        assert_eq!(t.faces.len(), faces.len(), "{name}");
        let tolerance = if curved { CURVED_MESH } else { FLAT_MESH };
        for (face, part) in faces.iter().zip(&t.faces) {
            let area = must(K::area(face), "area");
            near(
                part.mesh.area(),
                area,
                tolerance,
                format_args!("{name}: face area"),
            );
            let mut sum = Vec3::ZERO;
            let mut weight = 0.0;
            for [a, b, c] in part.mesh.triangles() {
                let w = (b - a).cross(c - a).length() / 2.0;
                let offset = |p: Point3| p - Point3::ORIGIN;
                sum = sum + (offset(a) + offset(b) + offset(c)) / 3.0 * w;
                weight += w;
            }
            let centroid = Point3::ORIGIN + sum / weight;
            let center = must(K::center_of_mass(face), "center_of_mass");
            assert!(
                centroid.distance(center) <= tolerance * 10.0,
                "{name}: mesh centroid {centroid:?}, face centre {center:?}"
            );
        }
    }
}

fn polyline_length(points: &[Point3]) -> f64 {
    points.windows(2).map(|p| p[0].distance(p[1])).sum()
}

/// `edges[i]` has the length of the `i`th edge and ends at its vertices.
///
/// # Panics
/// If a polyline doesn't match its edge.
pub fn tessellation_edges_follow_edge_order<K: Kernel + Inspection>() {
    for (name, shape, curved) in sample_solids::<K>() {
        let edges = parts::<K>(&shape, ShapeKind::Edge);
        let t = tessellation::<K>(&shape);
        assert_eq!(t.edges.len(), edges.len(), "{name}");
        let tolerance = if curved { CURVED_MESH } else { FLAT_MESH };
        for (edge, line) in edges.iter().zip(&t.edges) {
            assert!(
                line.points.len() >= 2,
                "{name}: a polyline needs two points"
            );
            let length = must(K::length(edge), "length");
            let got = polyline_length(&line.points);
            assert!(
                (got - length).abs() <= tolerance * length.max(1.0),
                "{name}: polyline {got}, edge {length}"
            );
            let ends: Vec<Point3> = parts::<K>(edge, ShapeKind::Vertex)
                .iter()
                .map(|v| must(K::vertex_point(v), "vertex_point"))
                .collect();
            for end in [line.points[0], line.points[line.points.len() - 1]] {
                assert!(
                    ends.iter().any(|v| v.distance(end) <= POSITION),
                    "{name}: {end:?} isn't one of {ends:?}"
                );
            }
        }
    }
}

/// Each edge's polyline is made of vertices of the mesh of each face it borders, so
/// drawn edges sit exactly on drawn faces.
///
/// # Panics
/// If a polyline point isn't a mesh vertex.
pub fn tessellation_polylines_meet_the_meshes<K: Kernel + Inspection>() {
    for (name, shape, _) in sample_solids::<K>() {
        let faces = parts::<K>(&shape, ShapeKind::Face);
        let edges = parts::<K>(&shape, ShapeKind::Edge);
        let t = tessellation::<K>(&shape);
        for (edge, line) in edges.iter().zip(&t.edges) {
            for face in must(K::ancestors(&shape, edge, ShapeKind::Face), "ancestors") {
                let index = position_of::<K>(&face, &faces).expect("the shape's own face");
                let mesh = &t.faces[index].mesh;
                for p in &line.points {
                    assert!(
                        mesh.positions.iter().any(|q| q.distance(*p) <= POSITION),
                        "{name}: polyline point {p:?} isn't on the face's mesh"
                    );
                }
            }
        }
    }
}

/// The merged faces enclose the solid's volume, so no triangle is missing or inverted.
///
/// # Panics
/// If the volume differs.
pub fn tessellation_merges_to_the_volume<K: Kernel + Inspection>() {
    for (name, shape, curved) in sample_solids::<K>() {
        let merged = tessellation::<K>(&shape).merged();
        let tolerance = if curved { CURVED_MESH } else { FLAT_MESH };
        near(
            merged.volume(),
            must(K::volume(&shape), "volume"),
            tolerance,
            format_args!("{name}: merged volume"),
        );
    }
}

/// Normals are unit length and agree with each triangle's winding, whichever way each
/// face's surface runs.
///
/// # Panics
/// If a normal differs.
pub fn tessellation_normals_face_out<K: Kernel + Inspection>() {
    for (name, shape, _) in sample_solids::<K>() {
        let t = tessellation::<K>(&shape);
        for part in &t.faces {
            let mesh = &part.mesh;
            assert_eq!(mesh.normals.len(), mesh.positions.len());
            for n in &mesh.normals {
                assert!((n.length() - 1.0).abs() <= MEASURE, "{name}: {n:?}");
            }
            for t in &mesh.indices {
                let [a, b, c] = t.map(|i| mesh.positions[i as usize]);
                let facet = (b - a).cross(c - a);
                if facet.length() < 1e-9 {
                    continue;
                }
                let average = t
                    .iter()
                    .fold(Vec3::ZERO, |sum, &i| sum + mesh.normals[i as usize]);
                assert!(
                    facet.dot(average) > 0.0,
                    "{name}: a triangle winds against its normals"
                );
            }
        }
    }
}

/// A plane's parameters are lengths along it, so two vertices are as far apart in
/// parameters as in space. This also shows the parameters line up with their vertices.
///
/// # Panics
/// If the distances differ.
pub fn tessellation_uvs_are_plane_coordinates<K: Kernel + Inspection>() {
    let b = offset_box::<K>([1.0, -2.0, 3.0], [1.0, 2.0, 3.0]);
    for part in &tessellation::<K>(&b).faces {
        assert_eq!(part.uvs.len(), part.mesh.positions.len());
        for (i, (uv_a, a)) in part.uvs.iter().zip(&part.mesh.positions).enumerate() {
            for (uv_b, b) in part.uvs.iter().zip(&part.mesh.positions).skip(i + 1) {
                let in_uv = (uv_a[0] - uv_b[0]).hypot(uv_a[1] - uv_b[1]);
                assert!(
                    (in_uv - a.distance(*b)).abs() <= POSITION,
                    "{uv_a:?} {uv_b:?}"
                );
            }
        }
    }
}

/// An edge alone has no triangles, and one polyline.
///
/// # Panics
/// If the tessellation differs.
pub fn tessellation_of_a_lone_edge<K: Kernel + Inspection>() {
    let b = cuboid::<K>(3.0, 3.0, 3.0);
    let edge = parts::<K>(&b, ShapeKind::Edge).swap_remove(0);
    let t = tessellation::<K>(&edge);
    assert!(t.faces.is_empty());
    assert_eq!(t.edges.len(), 1);
    near(
        polyline_length(&t.edges[0].points),
        3.0,
        MEASURE,
        "edge polyline",
    );

    let c = must(K::make_cylinder(Axis::Z, 2.0, 5.0), "make_cylinder");
    let rim = parts::<K>(&c, ShapeKind::Edge)
        .into_iter()
        .find(|e| must(K::curve_kind(e), "curve_kind") == CurveKind::Circle)
        .expect("a cylinder has rims");
    let t = tessellation::<K>(&rim);
    near(
        polyline_length(&t.edges[0].points),
        TAU * 2.0,
        CURVED_MESH,
        "rim polyline",
    );
}

/// A flag that's already set stops the work, every time and whatever was meshed before.
///
/// # Panics
/// If tessellation carries on.
pub fn tessellation_sets_cancelled_when_flagged<K: Kernel + Inspection>() {
    let flag = CancelFlag::new();
    flag.cancel();
    let b = cuboid::<K>(1.0, 1.0, 1.0);
    for _ in 0..2 {
        assert!(matches!(
            K::tessellate(&b, &fine(), Some(&flag)),
            Err(Error::Cancelled)
        ));
    }
    tessellation::<K>(&b);
    assert!(matches!(
        K::tessellate(&b, &fine(), Some(&flag)),
        Err(Error::Cancelled)
    ));
}

/// Cancelling leaves nothing behind that later work trips over, and an unset flag
/// changes nothing.
///
/// # Panics
/// If later tessellation is wrong.
pub fn tessellation_works_after_cancelling<K: Kernel + Inspection>() {
    let ball = must(K::make_sphere(Point3::ORIGIN, 2.0), "make_sphere");
    let flag = CancelFlag::new();
    flag.cancel();
    assert!(K::tessellate(&ball, &fine(), Some(&flag)).is_err());
    let fresh = CancelFlag::new();
    let t = must(K::tessellate(&ball, &fine(), Some(&fresh)), "tessellate");
    near(
        t.merged().volume(),
        ball_volume(2.0),
        CURVED_MESH,
        "after cancelling",
    );
}

fn drawing<K: Kernel + Inspection>(shape: &K::Shape, frame: Frame) -> Vec<DrawingLine> {
    let view = DrawingView {
        frame,
        perspective: None,
        deflection: DRAWING_DEFLECTION,
    };
    must(K::hidden_lines(shape, &view), "hidden_lines")
}

fn line_length(line: &DrawingLine) -> f64 {
    line.points
        .windows(2)
        .map(|p| (p[0][0] - p[1][0]).hypot(p[0][1] - p[1][1]))
        .sum()
}

fn total_length(lines: &[&DrawingLine]) -> f64 {
    lines.iter().copied().map(line_length).sum()
}

fn extent(lines: &[&DrawingLine]) -> ([f64; 2], [f64; 2]) {
    let mut min = [f64::MAX; 2];
    let mut max = [f64::MIN; 2];
    for point in lines.iter().flat_map(|l| &l.points) {
        for axis in 0..2 {
            min[axis] = min[axis].min(point[axis]);
            max[axis] = max[axis].max(point[axis]);
        }
    }
    (min, max)
}

/// From straight above, a box is its rectangle, drawn by the four sides of its top. Its
/// height goes end-on and its underside is behind the top.
///
/// # Panics
/// If the drawing differs.
pub fn box_from_above_draws_its_rectangle<K: Kernel + Inspection>() {
    for (w, d, h) in [(1.0, 1.0, 1.0), (4.0, 2.0, 1.0)] {
        let b = cuboid::<K>(w, d, h);
        let lines = drawing::<K>(&b, Frame::WORLD);
        let visible: Vec<_> = lines.iter().filter(|l| l.visible).collect();
        assert_eq!(visible.len(), 4, "{lines:?}");
        assert!(
            visible.iter().all(|l| l.kind == DrawingLineKind::Sharp),
            "{lines:?}"
        );
        near(total_length(&visible), 2.0 * (w + d), MEASURE, "perimeter");
        let (min, max) = extent(&visible);
        assert!(
            min[0].abs() <= POSITION && min[1].abs() <= POSITION,
            "{min:?}"
        );
        assert!(
            (max[0] - w).abs() <= POSITION && (max[1] - d).abs() <= POSITION,
            "{max:?}"
        );
    }
}

/// From a corner, all but the three edges of the far corner show, and those are hidden.
/// Their lengths are the projections of the edges.
///
/// # Panics
/// If the drawing differs.
pub fn box_from_a_corner_draws_nine_and_three<K: Kernel + Inspection>() {
    let size = Vec3::new(2.0, 3.0, 4.0);
    let b = cuboid::<K>(size.x, size.y, size.z);
    let z = dir([1.0, -1.0, 1.0]);
    let lines = drawing::<K>(&b, Frame::from_normal(Point3::ORIGIN, z));

    let visible: Vec<_> = lines.iter().filter(|l| l.visible).collect();
    let hidden: Vec<_> = lines.iter().filter(|l| !l.visible).collect();
    assert_eq!((visible.len(), hidden.len()), (9, 3), "{lines:?}");
    assert!(
        lines.iter().all(|l| l.kind == DrawingLineKind::Sharp),
        "{lines:?}"
    );

    // An edge of length l along a unit axis a projects to l sqrt(1 - (a.z)²). Viewed from
    // (+,-,+), the far corner is the one at (0, y, 0), and three edges meet there.
    let projected = |edge: Vec3| {
        let along = edge.dot(z.vector());
        (edge.dot(edge) - along * along).sqrt()
    };
    let x = projected(Vec3::new(size.x, 0.0, 0.0));
    let y = projected(Vec3::new(0.0, size.y, 0.0));
    let up = projected(Vec3::new(0.0, 0.0, size.z));
    near(total_length(&hidden), x + y + up, MEASURE, "hidden length");
    near(
        total_length(&visible),
        3.0 * (x + y + up),
        MEASURE,
        "visible length",
    );
}

/// Coordinates are measured from the frame's origin along its X and Y, where Y is Z
/// cross X, so views from above and from the side each show the expected two sizes.
///
/// # Panics
/// If the drawing is elsewhere.
pub fn drawing_uses_the_view_frame<K: Kernel + Inspection>() {
    let b = offset_box::<K>([10.0, 20.0, 30.0], [1.0, 2.0, 3.0]);

    // Looking down from +Z, with X right: straight subtraction of the origin.
    let origin = Point3::new(10.0, 20.0, 0.0);
    let frame = frame_of(origin, Dir3::Z, Vec3::new(1.0, 0.0, 0.0));
    let lines = drawing::<K>(&b, frame);
    let visible: Vec<_> = lines.iter().filter(|l| l.visible).collect();
    let (min, max) = extent(&visible);
    for (got, want) in [(min[0], 0.0), (min[1], 0.0), (max[0], 1.0), (max[1], 2.0)] {
        assert!(
            (got - want).abs() <= POSITION,
            "from above: {min:?} {max:?}"
        );
    }

    // From +X with world Y as view X, view Y is X cross Y, which is world Z.
    let frame = frame_of(Point3::ORIGIN, Dir3::X, Vec3::new(0.0, 1.0, 0.0));
    let lines = drawing::<K>(&b, frame);
    let visible: Vec<_> = lines.iter().filter(|l| l.visible).collect();
    let (min, max) = extent(&visible);
    for (got, want) in [
        (min[0], 20.0),
        (min[1], 30.0),
        (max[0], 22.0),
        (max[1], 33.0),
    ] {
        assert!(
            (got - want).abs() <= POSITION,
            "from the side: {min:?} {max:?}"
        );
    }
}

/// From the side, a cylinder is a rectangle: the long sides are silhouettes, belonging
/// to no edge, and the short sides are the rims, seen edge on. Nothing runs down the
/// middle, even where a kernel has a seam, as that is no feature of the part. A view at 45
/// degrees to the axes keeps the silhouettes clear of where a seam would naturally be.
///
/// # Panics
/// If the drawing differs.
pub fn cylinder_from_the_side_has_outlines<K: Kernel + Inspection>() {
    let (r, h) = (2.0, 5.0);
    let c = must(K::make_cylinder(Axis::Z, r, h), "make_cylinder");
    let frame = frame_of(
        Point3::ORIGIN,
        dir([1.0, -1.0, 0.0]),
        Vec3::new(1.0, 1.0, 0.0),
    );
    let lines = drawing::<K>(&c, frame);

    let outlines: Vec<_> = lines
        .iter()
        .filter(|l| l.kind == DrawingLineKind::Outline && l.visible)
        .collect();
    assert_eq!(outlines.len(), 2, "{lines:?}");
    near(
        total_length(&outlines),
        2.0 * h,
        DRAWING_LENGTH,
        "silhouette length",
    );
    let (min, max) = extent(&outlines);
    assert!(
        (min[0] + r).abs() <= POSITION && (max[0] - r).abs() <= POSITION,
        "{min:?} {max:?}"
    );

    let visible: Vec<_> = lines.iter().filter(|l| l.visible).collect();
    let (min, max) = extent(&visible);
    assert!((max[1] - min[1] - h).abs() <= POSITION, "{min:?} {max:?}");
    assert!(
        (max[0] - min[0] - 2.0 * r).abs() <= POSITION,
        "{min:?} {max:?}"
    );
    let inside = visible
        .iter()
        .flat_map(|l| &l.points)
        .any(|p| p[0].abs() < r - POSITION && p[1] > POSITION && p[1] < h - POSITION);
    assert!(!inside, "a line down the middle: {lines:?}");
}

/// However the surface happens to be divided, a cylinder's silhouette is drawn on both
/// sides, from any direction around it. Lines are compared by place: a kernel with a seam
/// exactly on the silhouette may call that part of it a model edge.
///
/// # Panics
/// If a side is missing.
pub fn cylinder_silhouette_shows_from_every_side<K: Kernel + Inspection>() {
    let (r, h) = (2.0, 5.0);
    let c = must(K::make_cylinder(Axis::Z, r, h), "make_cylinder");
    for i in 0..8 {
        let (s, c_) = (f64::from(i) * TAU / 8.0).sin_cos();
        let frame = frame_of(Point3::ORIGIN, dir([c_, s, 0.0]), Vec3::new(-s, c_, 0.0));
        let lines = drawing::<K>(&c, frame);
        for side in [-r, r] {
            let length: f64 = lines
                .iter()
                .filter(|l| l.visible)
                .filter(|l| l.points.iter().all(|p| (p[0] - side).abs() <= POSITION))
                .map(line_length)
                .sum();
            near(
                length,
                h,
                DRAWING_LENGTH,
                format_args!("view {i}, side {side}"),
            );
        }
    }
}

/// A rounded edge is two tangent boundaries; a plain box has none.
///
/// # Panics
/// If a smooth line is missing or unwarranted.
pub fn fillet_draws_a_smooth_line<K: Kernel + Inspection>() {
    let z = dir([1.0, -1.0, 1.0]);
    let frame = Frame::from_normal(Point3::ORIGIN, z);
    let b = cuboid::<K>(4.0, 4.0, 4.0);
    assert!(
        drawing::<K>(&b, frame)
            .iter()
            .all(|l| l.kind != DrawingLineKind::Smooth),
        "a box has no smooth edges"
    );
    let vertical = parts::<K>(&b, ShapeKind::Edge)
        .into_iter()
        .filter(|e| {
            let size = must(K::bounding_box(e), "bounding_box").size();
            size.z > 1.0
        })
        .collect::<Vec<_>>();
    let rounded = must(
        K::fillet(&b, &vertical.iter().collect::<Vec<_>>(), 1.0),
        "fillet",
    );
    let lines = drawing::<K>(&rounded, frame);
    let smooth = lines.iter().filter(|l| l.kind == DrawingLineKind::Smooth);
    assert!(smooth.count() > 0, "{lines:?}");
}

/// A point at height `z` projects scaled by `f / (f - z)`. From an eye above the middle
/// of a box, the top hides everything else: the sides lean inward going down, and the
/// bottom comes out smaller, so those eight lines are hidden, inside the top's four.
///
/// # Panics
/// If the extent differs.
pub fn perspective_enlarges_what_is_near<K: Kernel + Inspection>() {
    let (half, h, focus) = (1.0, 2.0, 10.0);
    let b = offset_box::<K>([-half, -half, 0.0], [2.0 * half, 2.0 * half, h]);
    let view = DrawingView {
        frame: Frame::WORLD,
        perspective: Some(focus),
        deflection: DRAWING_DEFLECTION,
    };
    let lines = must(K::hidden_lines(&b, &view), "hidden_lines");
    let visible: Vec<_> = lines.iter().filter(|l| l.visible).collect();
    let hidden: Vec<_> = lines.iter().filter(|l| !l.visible).collect();
    assert_eq!((visible.len(), hidden.len()), (4, 8), "{lines:?}");

    let (min, max) = extent(&visible);
    let top = half * focus / (focus - h);
    assert!((max[0] - top).abs() <= POSITION, "max {max:?}, want {top}");
    assert!(
        (min[1] + top).abs() <= POSITION,
        "min {min:?}, want {}",
        -top
    );

    // The far rectangle is on the plane through the view's origin, so stays true to size.
    let bottom = hidden
        .iter()
        .copied()
        .filter(|l| {
            l.points
                .iter()
                .all(|p| p[0].abs().max(p[1].abs()) <= half + POSITION)
        })
        .collect::<Vec<_>>();
    near(
        total_length(&bottom),
        8.0 * half,
        MEASURE,
        "the far rectangle",
    );
}
