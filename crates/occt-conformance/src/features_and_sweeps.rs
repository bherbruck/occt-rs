//! Checks for `AdvancedSweeps` and `Features`, against prism, frustum and Pappus formulas.
//!
//! The contract has no way to name a wire or a path directly here, so paths and profiles
//! come from what any kernel can do: a point moved along a vector or around an axis traces
//! an edge, and the outline of a face is a closed wire.

use std::f64::consts::{FRAC_PI_2, PI};
use std::fmt::Display;

use occt_kernel::{
    AdvancedSweeps, Angle, Axis, Dir3, Error, Features, Kernel, LoftOptions, Plane, Point3,
    ShapeKind, SweepOptions, SweepOrientation, Transform, Vec3,
};

use crate::support::{
    MEASURE, area_is, assert_valid, ball_volume, cuboid, must, near, near_point, parts, volume_is,
};

/// Everything these checks need of a kernel.
pub trait FeatureKernel: Kernel + AdvancedSweeps + Features {}
impl<K: Kernel + AdvancedSweeps + Features> FeatureKernel for K {}

/// Relative tolerance where a kernel may fit a spline to the true surface, as in a sweep
/// along a curve or a non-uniform scale of a sphere. Surface fits conventionally hold about
/// 1e-4 of model size, which is about 1e-3 of the volume of the shapes used here.
const FITTED: f64 = 1e-3;

// --- building blocks ------------------------------------------------------------------

fn square_at<K: FeatureKernel>(origin: Point3, side: f64) -> K::Shape {
    let (x, y, z) = (origin.x, origin.y, origin.z);
    must(
        K::make_polygon_face(&[
            Point3::new(x, y, z),
            Point3::new(x + side, y, z),
            Point3::new(x + side, y + side, z),
            Point3::new(x, y + side, z),
        ]),
        "make_polygon_face",
    )
}

/// A square of the given side centred on `(0, 0, z)`.
fn centred_square<K: FeatureKernel>(side: f64, z: f64) -> K::Shape {
    square_at::<K>(Point3::new(-side / 2.0, -side / 2.0, z), side)
}

/// A rectangle in the plane `y = 0`, with `x` from `inner` to `outer` and `z` from `-h` to `h`.
fn section<K: FeatureKernel>(inner: f64, outer: f64, h: f64) -> K::Shape {
    must(
        K::make_polygon_face(&[
            Point3::new(inner, 0.0, -h),
            Point3::new(outer, 0.0, -h),
            Point3::new(outer, 0.0, h),
            Point3::new(inner, 0.0, h),
        ]),
        "make_polygon_face",
    )
}

fn outline<K: FeatureKernel>(face: &K::Shape) -> K::Shape {
    parts::<K>(face, ShapeKind::Wire)
        .into_iter()
        .next()
        .expect("a face has an outline")
}

/// The subshape of a kind whose centre of mass is at a point.
fn part_at<K: FeatureKernel>(shape: &K::Shape, kind: ShapeKind, at: Point3) -> K::Shape {
    parts::<K>(shape, kind)
        .into_iter()
        .find(|part| must(K::center_of_mass(part), "center_of_mass").distance(at) < 1e-6)
        .unwrap_or_else(|| panic!("no {kind} at {at:?}"))
}

fn vertex_at<K: FeatureKernel>(at: Point3) -> K::Shape {
    parts::<K>(&square_at::<K>(at, 1.0), ShapeKind::Vertex)
        .into_iter()
        .find(|v| must(K::vertex_point(v), "vertex_point").distance(at) < 1e-6)
        .expect("a square has a vertex at its corner")
}

/// A flat disc around `center` with the given normal.
fn disc<K: FeatureKernel>(center: Point3, normal: Dir3, radius: f64) -> K::Shape {
    let bar = must(
        K::make_cylinder(Axis::new(center, normal), radius, 1.0),
        "make_cylinder",
    );
    part_at::<K>(&bar, ShapeKind::Face, center)
}

/// A straight path edge.
fn line<K: FeatureKernel>(from: Point3, by: Vec3) -> K::Shape {
    must(K::extrude(&vertex_at::<K>(from), by), "extrude of a vertex")
}

/// The arc from `(radius, 0, 0)` around Z, counter-clockwise.
fn arc<K: FeatureKernel>(radius: f64, angle: Angle) -> K::Shape {
    let start = vertex_at::<K>(Point3::new(radius, 0.0, 0.0));
    must(K::revolve(&start, Axis::Z, angle), "revolve of a vertex")
}

fn is_solid<K: FeatureKernel>(shape: &K::Shape, what: impl Display) {
    assert_eq!(
        must(K::kind(shape), &what),
        ShapeKind::Solid,
        "{what}: expected a solid"
    );
}

fn must_fail<T>(result: occt_kernel::Result<T>, what: impl Display) -> Error {
    match result {
        Ok(_) => panic!("{what}: expected an error"),
        Err(e) => e,
    }
}

fn contains<K: FeatureKernel>(shape: &K::Shape, kind: ShapeKind, part: &K::Shape) -> bool {
    parts::<K>(shape, kind).iter().any(|p| K::is_same(p, part))
}

/// A square frustum's volume from its base side, top side and height.
fn frustum(a: f64, b: f64, h: f64) -> f64 {
    h / 3.0 * (a * a + a * b + b * b)
}

/// A 10 x 10 x 2 plate with a round hole of radius 1 through its middle.
fn plate_with_hole<K: FeatureKernel>() -> K::Shape {
    let plate = cuboid::<K>(10.0, 10.0, 2.0);
    let axis = Axis::new(Point3::new(5.0, 5.0, -1.0), Dir3::Z);
    let drill = must(K::make_cylinder(axis, 1.0, 4.0), "make_cylinder");
    must(K::cut(&plate, &drill), "cut")
}

// --- sweeps -----------------------------------------------------------------------------

/// Extruding reports a side face for each profile edge.
///
/// # Panics
/// If the prism or its history is wrong.
pub fn extrude_modeled_maps_edges_to_faces<K: FeatureKernel>() {
    let face = square_at::<K>(Point3::ORIGIN, 2.0);
    let prism = must(
        K::extrude_modeled(&face, Vec3::new(0.0, 0.0, 3.0)),
        "extrude_modeled",
    );
    is_solid::<K>(&prism.shape, "prism");
    assert_valid::<K>(&prism.shape, "prism");
    volume_is::<K>(&prism.shape, 12.0, "prism");
    for edge in parts::<K>(&face, ShapeKind::Edge) {
        let (_, generated) = prism
            .history
            .generated
            .iter()
            .find(|(input, _)| K::is_same(input, &edge))
            .expect("every profile edge generates something");
        assert_eq!(generated.len(), 1, "one side face per edge");
        assert_eq!(must(K::kind(&generated[0]), "kind"), ShapeKind::Face);
        assert!(
            contains::<K>(&prism.shape, ShapeKind::Face, &generated[0]),
            "a generated face belongs to the result"
        );
    }
}

/// Revolving reports the same way, and the volume follows Pappus.
///
/// # Panics
/// If the solid or its history is wrong.
pub fn revolve_modeled_maps_edges_to_faces<K: FeatureKernel>() {
    let (r1, r2, h) = (1.0, 3.0, 1.0);
    let face = section::<K>(r1, r2, h);
    for degrees in [90.0, 200.0, 360.0] {
        let angle = Angle::degrees(degrees);
        let ring = must(K::revolve_modeled(&face, Axis::Z, angle), "revolve_modeled");
        is_solid::<K>(&ring.shape, "revolved section");
        assert_valid::<K>(&ring.shape, "revolved section");
        // The section is 2 by 2h, with its centroid at radius 2.
        volume_is::<K>(
            &ring.shape,
            2.0 * 2.0 * h * 2.0 * angle.as_radians(),
            "ring",
        );
        for edge in parts::<K>(&face, ShapeKind::Edge) {
            assert!(
                ring.history
                    .generated
                    .iter()
                    .any(|(input, out)| K::is_same(input, &edge) && !out.is_empty()),
                "every profile edge generates something at {degrees} degrees"
            );
        }
    }
}

/// A tapered square is a frustum, whichever way the taper goes.
///
/// # Panics
/// If a volume differs.
pub fn tapered_square_is_a_frustum<K: FeatureKernel>() {
    let (a, h) = (10.0, 3.0);
    let face = square_at::<K>(Point3::ORIGIN, a);
    for degrees in [-30.0_f64, -5.0, 1.0, 10.0, 45.0] {
        let taper = Angle::degrees(degrees);
        let solid = must(
            K::extrude_tapered(&face, Vec3::new(0.0, 0.0, h), taper),
            format_args!("taper by {degrees}"),
        );
        is_solid::<K>(&solid, "tapered prism");
        assert_valid::<K>(&solid, "tapered prism");
        let top = a - 2.0 * h * taper.as_radians().tan();
        volume_is::<K>(
            &solid,
            frustum(a, top, h),
            format_args!("taper by {degrees}"),
        );
    }
}

/// No taper is an ordinary prism, and a downward extrusion narrows downward.
///
/// # Panics
/// If a volume or position differs.
pub fn taper_follows_the_extrusion<K: FeatureKernel>() {
    let (a, h) = (10.0, 3.0);
    let face = square_at::<K>(Point3::ORIGIN, a);
    let plain = must(
        K::extrude_tapered(&face, Vec3::new(0.0, 0.0, h), Angle::radians(0.0)),
        "taper by nothing",
    );
    volume_is::<K>(&plain, a * a * h, "taper by nothing");

    let taper = Angle::degrees(20.0);
    let down = must(
        K::extrude_tapered(&face, Vec3::new(0.0, 0.0, -h), taper),
        "downward taper",
    );
    assert_valid::<K>(&down, "downward taper");
    let top = a - 2.0 * h * taper.as_radians().tan();
    volume_is::<K>(&down, frustum(a, top, h), "downward taper");
    assert!(
        must(K::center_of_mass(&down), "center_of_mass").z < 0.0,
        "the extrusion went down"
    );
}

/// A tapered disc is a truncated cone, and the hole of a tapered ring opens as the outside
/// closes in.
///
/// # Panics
/// If a volume differs.
pub fn taper_handles_round_sides_and_holes<K: FeatureKernel>() {
    let (outer, inner, h) = (5.0, 2.0, 2.0);
    let taper = Angle::degrees(15.0);
    let run = h * taper.as_radians().tan();
    let cone = |r1: f64, r2: f64| PI * h / 3.0 * (r1 * r1 + r1 * r2 + r2 * r2);

    let face = disc::<K>(Point3::ORIGIN, Dir3::Z, outer);
    let solid = must(
        K::extrude_tapered(&face, Vec3::new(0.0, 0.0, h), taper),
        "tapered disc",
    );
    assert_valid::<K>(&solid, "tapered disc");
    volume_is::<K>(&solid, cone(outer, outer - run), "tapered disc");

    let ring = must(
        K::cut(
            &must(K::make_cylinder(Axis::Z, outer, h), "make_cylinder"),
            &must(K::make_cylinder(Axis::Z, inner, h), "make_cylinder"),
        ),
        "cut",
    );
    let base = parts::<K>(&ring, ShapeKind::Face)
        .into_iter()
        .find(|f| {
            let c = must(K::center_of_mass(f), "center_of_mass");
            c.distance(Point3::ORIGIN) < 1e-6
        })
        .expect("a tube has a base");
    let solid = must(
        K::extrude_tapered(&base, Vec3::new(0.0, 0.0, h), taper),
        "tapered ring",
    );
    assert_valid::<K>(&solid, "tapered ring");
    volume_is::<K>(
        &solid,
        cone(outer, outer - run) - cone(inner, inner + run),
        "tapered ring",
    );
}

/// A taper that would close the profile before the far end has no solid to return.
///
/// # Panics
/// If it is accepted.
pub fn taper_too_steep_is_rejected<K: FeatureKernel>() {
    let face = square_at::<K>(Point3::ORIGIN, 2.0);
    must_fail(
        K::extrude_tapered(&face, Vec3::new(0.0, 0.0, 10.0), Angle::degrees(45.0)),
        "a taper past the apex",
    );
}

/// A closed profile along a straight line is the extrusion along that line.
///
/// # Panics
/// If a measure differs.
pub fn sweep_along_a_line_equals_an_extrusion<K: FeatureKernel>() {
    let (a, h) = (2.0, 7.0);
    let face = square_at::<K>(Point3::ORIGIN, a);
    let by = Vec3::new(0.0, 0.0, h);
    let path = line::<K>(Point3::ORIGIN, by);
    let area = 2.0 * a * a + 4.0 * a * h;

    for orientation in [
        SweepOrientation::Corrected,
        SweepOrientation::Frenet,
        SweepOrientation::Fixed,
    ] {
        let options = SweepOptions {
            orientation,
            solid: true,
            binormal: None,
        };
        for (what, profile) in [("face", face.clone()), ("wire", outline::<K>(&face))] {
            let swept = must(
                K::sweep(&profile, &path, &options),
                format_args!("{what} sweep, {orientation:?}"),
            );
            is_solid::<K>(&swept.shape, what);
            assert_valid::<K>(&swept.shape, what);
            volume_is::<K>(&swept.shape, a * a * h, what);
            area_is::<K>(&swept.shape, area, what);
        }
    }
}

/// A slanted path keeps the section's relation to the path, or its fixed orientation: in
/// both cases the volume is the area times the height over the section's plane.
///
/// # Panics
/// If a volume differs.
pub fn sweep_along_a_slanted_line<K: FeatureKernel>() {
    let (a, h) = (2.0, 7.0);
    let face = square_at::<K>(Point3::ORIGIN, a);
    let path = line::<K>(Point3::ORIGIN, Vec3::new(4.0, 1.0, h));
    let options = SweepOptions {
        orientation: SweepOrientation::Fixed,
        ..SweepOptions::default()
    };
    let swept = must(K::sweep(&face, &path, &options), "slanted sweep");
    assert_valid::<K>(&swept.shape, "slanted sweep");
    volume_is::<K>(&swept.shape, a * a * h, "slanted sweep");
}

/// By Pappus, a disc moved along an arc across it sweeps its area times the arc length.
/// The face, the closed wire and every way of keeping the section across the path agree.
///
/// # Panics
/// If a volume differs.
pub fn sweep_circle_along_an_arc_obeys_pappus<K: FeatureKernel>() {
    let (bend, radius) = (10.0, 1.0);
    let face = disc::<K>(Point3::new(bend, 0.0, 0.0), Dir3::Y, radius);
    let section_area = PI * radius * radius;
    for degrees in [30.0, 90.0, 180.0, 270.0] {
        let angle = Angle::degrees(degrees);
        let path = arc::<K>(bend, angle);
        let want = section_area * bend * angle.as_radians();
        let settings = [
            (SweepOrientation::Corrected, None),
            (SweepOrientation::Frenet, None),
            (SweepOrientation::Corrected, Some(Dir3::Z)),
        ];
        for (orientation, binormal) in settings {
            let options = SweepOptions {
                orientation,
                binormal,
                solid: true,
            };
            let what = format_args!("{degrees} degree bend, {orientation:?}, {binormal:?}");
            let swept = must(K::sweep(&face, &path, &options), what);
            is_solid::<K>(&swept.shape, &what);
            assert_valid::<K>(&swept.shape, &what);
            near(must(K::volume(&swept.shape), what), want, FITTED, what);

            let wire = outline::<K>(&face);
            let swept = must(K::sweep(&wire, &path, &options), what);
            near(must(K::volume(&swept.shape), what), want, FITTED, what);
        }
    }
}

/// A section that stays parallel to itself along a quarter circle sweeps its area times the
/// distance the path covers across the section's plane, which is the bend radius. Following
/// the path would give the arc length, so this tells the two apart.
///
/// # Panics
/// If the volume differs.
pub fn sweep_with_fixed_orientation_keeps_the_section_parallel<K: FeatureKernel>() {
    let (bend, radius) = (10.0, 1.0);
    let face = disc::<K>(Point3::new(bend, 0.0, 0.0), Dir3::Y, radius);
    let path = arc::<K>(bend, Angle::radians(FRAC_PI_2));
    let options = SweepOptions {
        orientation: SweepOrientation::Fixed,
        ..SweepOptions::default()
    };
    let swept = must(K::sweep(&face, &path, &options), "parallel sweep");
    assert_valid::<K>(&swept.shape, "parallel sweep");
    near(
        must(K::volume(&swept.shape), "parallel sweep"),
        PI * radius * radius * bend,
        FITTED,
        "parallel sweep",
    );
}

/// A closed wire without `solid` makes a tube with open ends; an open profile makes a sheet.
///
/// # Panics
/// If a measure or kind differs.
pub fn sweep_without_solid_makes_a_sheet<K: FeatureKernel>() {
    let (a, h) = (2.0, 7.0);
    let face = square_at::<K>(Point3::ORIGIN, a);
    let path = line::<K>(Point3::ORIGIN, Vec3::new(0.0, 0.0, h));
    let options = SweepOptions::default();

    let tube = must(K::sweep(&outline::<K>(&face), &path, &options), "tube");
    assert_ne!(must(K::kind(&tube.shape), "kind"), ShapeKind::Solid);
    area_is::<K>(&tube.shape, 4.0 * a * h, "tube");

    let edge = parts::<K>(&face, ShapeKind::Edge).remove(0);
    let strip = must(K::sweep(&edge, &path, &options), "strip");
    area_is::<K>(&strip.shape, a * h, "strip");
}

/// A face keeps its holes: a ring along a line is a tube, and along an arc obeys Pappus.
///
/// # Panics
/// If a volume differs.
pub fn sweep_of_a_face_keeps_its_holes<K: FeatureKernel>() {
    let (outer, inner, bend) = (2.0, 1.0, 10.0);
    let start = Point3::new(bend, 0.0, 0.0);
    let axis = Axis::new(start, Dir3::Y);
    let tube = must(
        K::cut(
            &must(K::make_cylinder(axis, outer, 1.0), "make_cylinder"),
            &must(K::make_cylinder(axis, inner, 1.0), "make_cylinder"),
        ),
        "cut",
    );
    let ring = part_at::<K>(&tube, ShapeKind::Face, start);
    let ring_area = PI * (outer * outer - inner * inner);

    let path = line::<K>(start, Vec3::new(0.0, 5.0, 0.0));
    let straight = must(K::sweep(&ring, &path, &SweepOptions::default()), "ring");
    assert_valid::<K>(&straight.shape, "ring along a line");
    near(
        must(K::volume(&straight.shape), "volume"),
        ring_area * 5.0,
        FITTED,
        "ring along a line",
    );

    let quarter = Angle::radians(FRAC_PI_2);
    let path = arc::<K>(bend, quarter);
    let bent = must(K::sweep(&ring, &path, &SweepOptions::default()), "ring");
    assert_valid::<K>(&bent.shape, "ring along an arc");
    near(
        must(K::volume(&bent.shape), "volume"),
        ring_area * bend * FRAC_PI_2,
        FITTED,
        "ring along an arc",
    );

    for edge in parts::<K>(&ring, ShapeKind::Edge) {
        let survivors = bent
            .history
            .generated
            .iter()
            .find(|(input, _)| K::is_same(input, &edge))
            .map_or(0, |(_, out)| {
                out.iter()
                    .filter(|f| contains::<K>(&bent.shape, ShapeKind::Face, f))
                    .count()
            });
        assert!(
            survivors > 0,
            "a ring's edges each make a face of the result"
        );
    }
}

/// A rectangle swept all the way around a square path makes a mitred frame: the area of the
/// square ring it covers, times the rectangle's height.
///
/// # Panics
/// If the volume differs.
pub fn sweep_around_a_closed_path_makes_a_frame<K: FeatureKernel>() {
    let (s, w, h) = (10.0, 1.0, 1.0);
    // The section sits across the path where it starts, at the square's first corner.
    let square = square_at::<K>(Point3::ORIGIN, s);
    let path = outline::<K>(&square);
    let section = must(
        K::make_polygon_face(&[
            Point3::new(0.0, -w, -h),
            Point3::new(0.0, w, -h),
            Point3::new(0.0, w, h),
            Point3::new(0.0, -w, h),
        ]),
        "make_polygon_face",
    );
    let frame = must(
        K::sweep(
            &outline::<K>(&section),
            &path,
            &SweepOptions {
                solid: true,
                ..SweepOptions::default()
            },
        ),
        "frame",
    );
    assert_valid::<K>(&frame.shape, "frame");
    let ring = (s + 2.0 * w).powi(2) - (s - 2.0 * w).powi(2);
    volume_is::<K>(&frame.shape, ring * 2.0 * h, "frame");

    let from_face = must(K::sweep(&section, &path, &SweepOptions::default()), "frame");
    assert_valid::<K>(&from_face.shape, "frame from a face");
    volume_is::<K>(&from_face.shape, ring * 2.0 * h, "frame from a face");
}

/// Each profile edge of a sweep generates faces of the result.
///
/// # Panics
/// If the history is missing.
pub fn sweep_reports_side_faces<K: FeatureKernel>() {
    let face = square_at::<K>(Point3::ORIGIN, 2.0);
    let path = line::<K>(Point3::ORIGIN, Vec3::new(0.0, 0.0, 5.0));
    let swept = must(K::sweep(&face, &path, &SweepOptions::default()), "sweep");
    for edge in parts::<K>(&face, ShapeKind::Edge) {
        let (_, faces) = swept
            .history
            .generated
            .iter()
            .find(|(input, _)| K::is_same(input, &edge))
            .expect("a profile edge generates something");
        assert!(!faces.is_empty());
        for face in faces {
            assert!(
                contains::<K>(&swept.shape, ShapeKind::Face, face),
                "a generated face belongs to the result"
            );
        }
    }
}

/// Inputs that cannot make a sweep are errors.
///
/// # Panics
/// If one is accepted.
pub fn sweep_rejects_unusable_inputs<K: FeatureKernel>() {
    let face = square_at::<K>(Point3::ORIGIN, 2.0);
    let solid = cuboid::<K>(1.0, 1.0, 1.0);
    let path = line::<K>(Point3::ORIGIN, Vec3::new(0.0, 0.0, 5.0));
    let edge = parts::<K>(&face, ShapeKind::Edge).remove(0);
    let plain = SweepOptions::default();
    let solid_options = SweepOptions {
        solid: true,
        ..plain
    };

    must_fail(K::sweep(&solid, &path, &plain), "a solid profile");
    must_fail(K::sweep(&face, &solid, &plain), "a solid path");
    must_fail(K::sweep(&face, &face, &plain), "a face path");
    must_fail(
        K::sweep(&edge, &path, &solid_options),
        "a solid from an open profile",
    );
}

/// Between two squares a loft is a frustum, ruled or smooth.
///
/// # Panics
/// If a volume differs.
pub fn loft_between_squares_is_a_frustum<K: FeatureKernel>() {
    let (a, b, h) = (4.0, 2.0, 3.0);
    let bottom = centred_square::<K>(a, 0.0);
    let top = centred_square::<K>(b, h);
    for ruled in [true, false] {
        let options = LoftOptions {
            ruled,
            ..LoftOptions::default()
        };
        for (what, sections) in [
            ("faces", [bottom.clone(), top.clone()]),
            ("wires", [outline::<K>(&bottom), outline::<K>(&top)]),
        ] {
            let what = format_args!("{what}, ruled {ruled}");
            let refs: Vec<_> = sections.iter().collect();
            let loft = must(K::loft(&refs, &options), what);
            is_solid::<K>(&loft, &what);
            assert_valid::<K>(&loft, &what);
            near(
                must(K::volume(&loft), what),
                frustum(a, b, h),
                MEASURE,
                what,
            );
        }
    }
}

/// Through three sections, a ruled loft is two stacked frustums; a smooth one passes
/// through the same sections but bulges or pinches between them.
///
/// # Panics
/// If a volume differs or the smooth loft misses a section.
pub fn loft_through_several_sections<K: FeatureKernel>() {
    let (a, b, c, h) = (4.0, 2.0, 5.0, 3.0);
    let sections = [
        centred_square::<K>(a, 0.0),
        centred_square::<K>(b, h),
        centred_square::<K>(c, 2.0 * h),
    ];
    let refs: Vec<_> = sections.iter().collect();

    let ruled = must(
        K::loft(
            &refs,
            &LoftOptions {
                ruled: true,
                ..LoftOptions::default()
            },
        ),
        "ruled loft",
    );
    assert_valid::<K>(&ruled, "ruled loft");
    volume_is::<K>(&ruled, frustum(a, b, h) + frustum(b, c, h), "ruled loft");

    let smooth = must(K::loft(&refs, &LoftOptions::default()), "smooth loft");
    assert_valid::<K>(&smooth, "smooth loft");
    let bounds = must(K::bounding_box(&smooth), "bounding_box");
    near(bounds.size().z, 2.0 * h, MEASURE, "smooth loft height");
    let volume = must(K::volume(&smooth), "volume");
    assert!(
        volume > 0.0 && (volume - must(K::volume(&ruled), "volume")).abs() > MEASURE,
        "a smooth loft through a pinch differs from the ruled one, got {volume}"
    );
}

/// A loft from a square to a point is a pyramid, and from circle to circle a cone.
///
/// # Panics
/// If a volume differs.
pub fn loft_to_a_point_and_between_circles<K: FeatureKernel>() {
    let (a, h) = (4.0, 3.0);
    let base = centred_square::<K>(a, 0.0);
    let apex = vertex_at::<K>(Point3::new(0.0, 0.0, h));
    let pyramid = must(
        K::loft(
            &[&base, &apex],
            &LoftOptions {
                ruled: true,
                ..LoftOptions::default()
            },
        ),
        "pyramid",
    );
    assert_valid::<K>(&pyramid, "pyramid");
    volume_is::<K>(&pyramid, a * a * h / 3.0, "pyramid");

    let (r1, r2) = (3.0, 1.0);
    let low = disc::<K>(Point3::ORIGIN, Dir3::Z, r1);
    let high = disc::<K>(Point3::new(0.0, 0.0, h), Dir3::Z, r2);
    let cone = must(K::loft(&[&low, &high], &LoftOptions::default()), "cone");
    assert_valid::<K>(&cone, "cone");
    near(
        must(K::volume(&cone), "cone"),
        PI * h / 3.0 * (r1 * r1 + r1 * r2 + r2 * r2),
        FITTED,
        "cone",
    );
}

/// Without closed ends a loft is a sheet, with the frustum's sides only.
///
/// # Panics
/// If the area or kind differs.
pub fn loft_as_a_sheet_has_no_ends<K: FeatureKernel>() {
    let (a, b, h) = (4.0, 2.0, 3.0);
    let bottom = centred_square::<K>(a, 0.0);
    let top = centred_square::<K>(b, h);
    let sheet = must(
        K::loft(
            &[&bottom, &top],
            &LoftOptions {
                solid: false,
                ruled: true,
                ..LoftOptions::default()
            },
        ),
        "sheet",
    );
    assert_ne!(must(K::kind(&sheet), "kind"), ShapeKind::Solid);
    let slant = (h * h + ((a - b) / 2.0).powi(2)).sqrt();
    area_is::<K>(&sheet, 2.0 * (a + b) * slant, "sheet");
}

/// Inputs that cannot make a loft are errors.
///
/// # Panics
/// If one is accepted.
pub fn loft_rejects_unusable_inputs<K: FeatureKernel>() {
    let square = centred_square::<K>(2.0, 0.0);
    let far = centred_square::<K>(2.0, 3.0);
    let point = vertex_at::<K>(Point3::new(0.0, 0.0, 1.0));
    let solid = cuboid::<K>(1.0, 1.0, 1.0);
    let edge = parts::<K>(&square, ShapeKind::Edge).remove(0);
    let other = parts::<K>(&far, ShapeKind::Edge).remove(0);
    let options = LoftOptions::default();

    must_fail(K::loft(&[], &options), "no sections");
    must_fail(K::loft(&[&square], &options), "one section");
    must_fail(K::loft(&[&solid, &far], &options), "a solid section");
    must_fail(
        K::loft(&[&square, &point, &far], &options),
        "a point inside",
    );
    must_fail(
        K::loft(&[&edge, &other], &options),
        "a solid from open sections",
    );
}

// --- features ---------------------------------------------------------------------------

/// A block hollowed to walls of `t`, open at the top, loses an inner box of the given size.
///
/// # Panics
/// If a volume differs.
pub fn shell_with_an_open_face<K: FeatureKernel>() {
    let (a, b, c) = (10.0, 8.0, 6.0);
    let block = cuboid::<K>(a, b, c);
    let top = part_at::<K>(&block, ShapeKind::Face, Point3::new(a / 2.0, b / 2.0, c));
    for t in [0.1, 1.0, 2.5] {
        let tray = must(K::shell(&block, &[&top], -t), format_args!("walls of {t}"));
        is_solid::<K>(&tray.shape, "tray");
        assert_valid::<K>(&tray.shape, "tray");
        let inner = (a - 2.0 * t) * (b - 2.0 * t) * (c - t);
        volume_is::<K>(&tray.shape, a * b * c - inner, format_args!("walls of {t}"));
    }
}

/// Walls grown outward leave the original block as the cavity.
///
/// # Panics
/// If a volume differs.
pub fn shell_grown_outward<K: FeatureKernel>() {
    let (a, t) = (10.0, 1.0);
    let block = cuboid::<K>(a, a, a);
    let top = part_at::<K>(&block, ShapeKind::Face, Point3::new(a / 2.0, a / 2.0, a));
    let tray = must(K::shell(&block, &[&top], t), "outward walls");
    assert_valid::<K>(&tray.shape, "outward walls");
    let outer = (a + 2.0 * t) * (a + 2.0 * t) * (a + t);
    volume_is::<K>(&tray.shape, outer - a * a * a, "outward walls");
}

/// Two opposite open faces make a tube of square section.
///
/// # Panics
/// If a volume differs.
pub fn shell_with_two_open_faces<K: FeatureKernel>() {
    let (a, t) = (10.0, 1.0);
    let block = cuboid::<K>(a, a, a);
    let top = part_at::<K>(&block, ShapeKind::Face, Point3::new(a / 2.0, a / 2.0, a));
    let bottom = part_at::<K>(&block, ShapeKind::Face, Point3::new(a / 2.0, a / 2.0, 0.0));
    let tube = must(K::shell(&block, &[&top, &bottom], -t), "tube");
    assert_valid::<K>(&tube.shape, "tube");
    let hole = a - 2.0 * t;
    volume_is::<K>(&tube.shape, a * a * a - hole * hole * a, "tube");
}

/// With no open face the cavity is closed inside the walls.
///
/// # Panics
/// If a volume differs.
pub fn shell_with_no_open_faces_has_a_cavity<K: FeatureKernel>() {
    let (a, t) = (10.0, 1.0);
    let block = cuboid::<K>(a, a, a);
    let hollow = must(K::shell(&block, &[], -t), "closed cavity");
    is_solid::<K>(&hollow.shape, "closed cavity");
    assert_valid::<K>(&hollow.shape, "closed cavity");
    volume_is::<K>(
        &hollow.shape,
        a.powi(3) - (a - 2.0 * t).powi(3),
        "closed cavity",
    );
}

/// A round cup: outside minus a shorter, narrower cylinder.
///
/// # Panics
/// If a volume differs.
pub fn shell_of_a_cylinder<K: FeatureKernel>() {
    let (r, h, t) = (5.0, 10.0, 1.0);
    let bar = must(K::make_cylinder(Axis::Z, r, h), "make_cylinder");
    let top = part_at::<K>(&bar, ShapeKind::Face, Point3::new(0.0, 0.0, h));
    let cup = must(K::shell(&bar, &[&top], -t), "cup");
    assert_valid::<K>(&cup.shape, "cup");
    let inner = PI * (r - t) * (r - t) * (h - t);
    volume_is::<K>(&cup.shape, PI * r * r * h - inner, "cup");
}

/// Walls thicker than the solid, foreign faces, and other kinds are errors.
///
/// # Panics
/// If one is accepted.
pub fn shell_rejects_unusable_inputs<K: FeatureKernel>() {
    let block = cuboid::<K>(2.0, 2.0, 2.0);
    let other = cuboid::<K>(2.0, 2.0, 2.0);
    let foreign = parts::<K>(&other, ShapeKind::Face).remove(0);
    let own = parts::<K>(&block, ShapeKind::Face).remove(0);
    let edge = parts::<K>(&block, ShapeKind::Edge).remove(0);

    must_fail(K::shell(&block, &[&own], -5.0), "walls past the middle");
    must_fail(K::shell(&block, &[&foreign], -0.1), "a foreign face");
    must_fail(K::shell(&block, &[&edge], -0.1), "an edge as a face");
    must_fail(K::shell(&own, &[], -0.1), "a face as the solid");
}

/// Every face of a box moves, so each side grows by twice the distance, and a sphere's
/// radius grows by the distance.
///
/// # Panics
/// If a volume differs.
pub fn offset_of_a_box_and_a_sphere<K: FeatureKernel>() {
    let (a, b, c) = (10.0, 8.0, 6.0);
    let block = cuboid::<K>(a, b, c);
    for d in [0.5, 2.0, -0.5, -2.0] {
        let grown = must(K::offset(&block, d), format_args!("offset by {d}"));
        assert_valid::<K>(&grown, "offset box");
        volume_is::<K>(
            &grown,
            (a + 2.0 * d) * (b + 2.0 * d) * (c + 2.0 * d),
            format_args!("offset by {d}"),
        );
    }
    let ball = must(K::make_sphere(Point3::ORIGIN, 5.0), "make_sphere");
    let larger = must(K::offset(&ball, 1.0), "offset sphere");
    assert_valid::<K>(&larger, "offset sphere");
    near(
        must(K::volume(&larger), "volume"),
        ball_volume(6.0),
        FITTED,
        "offset sphere",
    );
}

/// Growing a plate narrows its hole.
///
/// # Panics
/// If a volume differs.
pub fn offset_narrows_a_hole<K: FeatureKernel>() {
    let d = 0.25;
    let grown = must(K::offset(&plate_with_hole::<K>(), d), "offset plate");
    assert_valid::<K>(&grown, "offset plate");
    let (w, t, r) = (10.0 + 2.0 * d, 2.0 + 2.0 * d, 1.0 - d);
    volume_is::<K>(&grown, w * w * t - PI * r * r * t, "offset plate");
}

/// Shrinking past the middle leaves nothing.
///
/// # Panics
/// If it is accepted.
pub fn offset_past_the_middle_is_rejected<K: FeatureKernel>() {
    must_fail(
        K::offset(&cuboid::<K>(2.0, 2.0, 2.0), -2.0),
        "offset past the middle",
    );
}

/// A flat face grows along its normal or against it; a curved sheet makes a wall.
///
/// # Panics
/// If a measure differs.
pub fn thicken_a_face<K: FeatureKernel>() {
    let (a, t) = (4.0, 0.5);
    let face = square_at::<K>(Point3::ORIGIN, a);
    for (signed, z) in [(t, t / 2.0), (-t, -t / 2.0)] {
        let plate = must(
            K::thicken(&face, signed),
            format_args!("thicken by {signed}"),
        );
        is_solid::<K>(&plate, "thick plate");
        assert_valid::<K>(&plate, "thick plate");
        volume_is::<K>(&plate, a * a * t, "thick plate");
        near_point(
            must(K::center_of_mass(&plate), "center_of_mass"),
            Point3::new(a / 2.0, a / 2.0, z),
            format_args!("thicken by {signed}"),
        );
    }

    let (r, h) = (5.0, 3.0);
    let bar = must(K::make_cylinder(Axis::Z, r, h), "make_cylinder");
    let wall = parts::<K>(&bar, ShapeKind::Face)
        .into_iter()
        .find(|f| {
            must(K::center_of_mass(f), "center_of_mass").distance(Point3::new(0.0, 0.0, h)) > 1e-6
                && must(K::center_of_mass(f), "center_of_mass").distance(Point3::ORIGIN) > 1e-6
        })
        .expect("a cylinder has a wall");
    let tube = must(K::thicken(&wall, t), "thick wall");
    assert_valid::<K>(&tube, "thick wall");
    volume_is::<K>(&tube, PI * ((r + t) * (r + t) - r * r) * h, "thick wall");
}

/// A solid or an edge is not a sheet.
///
/// # Panics
/// If one is accepted.
pub fn thicken_rejects_other_kinds<K: FeatureKernel>() {
    let block = cuboid::<K>(1.0, 1.0, 1.0);
    let edge = parts::<K>(&block, ShapeKind::Edge).remove(0);
    must_fail(K::thicken(&block, 0.5), "thicken a solid");
    must_fail(K::thicken(&edge, 0.5), "thicken an edge");
}

/// One side of a block hinged at its bottom removes a wedge.
///
/// # Panics
/// If the volume differs.
pub fn draft_one_face_removes_a_wedge<K: FeatureKernel>() {
    let (a, h) = (10.0, 2.0);
    let block = cuboid::<K>(a, a, h);
    let side = part_at::<K>(&block, ShapeKind::Face, Point3::new(a, a / 2.0, h / 2.0));
    for degrees in [1.0_f64, 10.0, 30.0, -10.0] {
        let angle = Angle::degrees(degrees);
        let drafted = must(
            K::draft(&block, &[&side], Dir3::Z, Plane::XY, angle),
            format_args!("draft by {degrees}"),
        );
        is_solid::<K>(&drafted.shape, "draft");
        assert_valid::<K>(&drafted.shape, "draft");
        // Positive removes material above the hinge, on the side `pull` points to.
        let wedge = h * h * angle.as_radians().tan() / 2.0 * a;
        volume_is::<K>(
            &drafted.shape,
            a * a * h - wedge,
            format_args!("draft by {degrees}"),
        );
    }
}

/// All four sides make a frustum.
///
/// # Panics
/// If the volume differs.
pub fn draft_four_faces_makes_a_frustum<K: FeatureKernel>() {
    let (a, h) = (10.0, 2.0);
    let block = cuboid::<K>(a, a, h);
    let sides = [
        Point3::new(a, a / 2.0, h / 2.0),
        Point3::new(0.0, a / 2.0, h / 2.0),
        Point3::new(a / 2.0, a, h / 2.0),
        Point3::new(a / 2.0, 0.0, h / 2.0),
    ]
    .map(|at| part_at::<K>(&block, ShapeKind::Face, at));
    let angle = Angle::degrees(20.0);
    let refs: Vec<_> = sides.iter().collect();
    let drafted = must(K::draft(&block, &refs, Dir3::Z, Plane::XY, angle), "draft");
    assert_valid::<K>(&drafted.shape, "draft");
    let top = a - 2.0 * h * angle.as_radians().tan();
    volume_is::<K>(&drafted.shape, frustum(a, top, h), "draft");
}

/// Hinged at mid-height, what leaves above the plane comes back below it.
///
/// # Panics
/// If the volume differs.
pub fn draft_about_the_middle_keeps_the_volume<K: FeatureKernel>() {
    let (a, h) = (10.0, 2.0);
    let block = cuboid::<K>(a, a, h);
    let side = part_at::<K>(&block, ShapeKind::Face, Point3::new(a, a / 2.0, h / 2.0));
    let neutral = Plane::new(Point3::new(0.0, 0.0, h / 2.0), Dir3::Z);
    let drafted = must(
        K::draft(&block, &[&side], Dir3::Z, neutral, Angle::degrees(15.0)),
        "draft",
    );
    assert_valid::<K>(&drafted.shape, "draft");
    volume_is::<K>(&drafted.shape, a * a * h, "draft");
}

/// A tilted face is reported as modified.
///
/// # Panics
/// If the history doesn't say so.
pub fn draft_reports_the_modified_face<K: FeatureKernel>() {
    let (a, h) = (10.0, 2.0);
    let block = cuboid::<K>(a, a, h);
    let side = part_at::<K>(&block, ShapeKind::Face, Point3::new(a, a / 2.0, h / 2.0));
    let drafted = must(
        K::draft(&block, &[&side], Dir3::Z, Plane::XY, Angle::degrees(10.0)),
        "draft",
    );
    let (_, out) = drafted
        .history
        .modified
        .iter()
        .find(|(input, _)| K::is_same(input, &side))
        .expect("the tilted face is reported");
    assert!(
        out.iter()
            .all(|f| contains::<K>(&drafted.shape, ShapeKind::Face, f))
            && !out.is_empty(),
        "the tilted face's successor belongs to the result"
    );
}

/// A face the pull direction crosses has no draft; a foreign face isn't part of the solid.
///
/// # Panics
/// If one is accepted.
pub fn draft_rejects_unusable_inputs<K: FeatureKernel>() {
    let (a, h) = (10.0, 2.0);
    let block = cuboid::<K>(a, a, h);
    let top = part_at::<K>(&block, ShapeKind::Face, Point3::new(a / 2.0, a / 2.0, h));
    let other = cuboid::<K>(a, a, h);
    let foreign = parts::<K>(&other, ShapeKind::Face).remove(0);
    let angle = Angle::degrees(5.0);
    must_fail(
        K::draft(&block, &[&top], Dir3::Z, Plane::XY, angle),
        "the top",
    );
    must_fail(
        K::draft(&block, &[&foreign], Dir3::Z, Plane::XY, angle),
        "a foreign face",
    );
    must_fail(K::draft(&block, &[], Dir3::Z, Plane::XY, angle), "no faces");
}

/// Removing a hole's wall gives back the plain plate, and the wall is reported deleted.
///
/// # Panics
/// If a measure differs.
pub fn remove_faces_fills_a_hole<K: FeatureKernel>() {
    let holed = plate_with_hole::<K>();
    let wall = parts::<K>(&holed, ShapeKind::Face)
        .into_iter()
        .find(|f| must(K::surface_kind(f), "surface_kind") == occt_kernel::SurfaceKind::Cylinder)
        .expect("the hole has a wall");
    let filled = must(K::remove_faces(&holed, &[&wall]), "remove_faces");
    is_solid::<K>(&filled.shape, "filled plate");
    assert_valid::<K>(&filled.shape, "filled plate");
    volume_is::<K>(&filled.shape, 200.0, "filled plate");
    assert_eq!(parts::<K>(&filled.shape, ShapeKind::Face).len(), 6);
    assert!(
        filled.history.deleted.iter().any(|f| K::is_same(f, &wall)),
        "the wall is reported deleted"
    );
}

/// Removing the faces of a boss takes the boss off.
///
/// # Panics
/// If a measure differs.
pub fn remove_faces_takes_off_a_boss<K: FeatureKernel>() {
    let base = cuboid::<K>(10.0, 10.0, 2.0);
    let axis = Axis::new(Point3::new(5.0, 5.0, 2.0), Dir3::Z);
    let post = must(K::make_cylinder(axis, 1.0, 3.0), "make_cylinder");
    let part = must(K::fuse(&base, &post), "fuse");
    volume_is::<K>(&part, 200.0 + 3.0 * PI, "boss");
    let cap = part_at::<K>(&part, ShapeKind::Face, Point3::new(5.0, 5.0, 5.0));
    let side = parts::<K>(&part, ShapeKind::Face)
        .into_iter()
        .find(|f| must(K::surface_kind(f), "surface_kind") == occt_kernel::SurfaceKind::Cylinder)
        .expect("the post has a side");
    let flat = must(K::remove_faces(&part, &[&cap, &side]), "remove_faces");
    assert_valid::<K>(&flat.shape, "boss removed");
    volume_is::<K>(&flat.shape, 200.0, "boss removed");
}

/// A foreign face, an empty list, and a face whose neighbours can't meet are errors.
///
/// # Panics
/// If one is accepted.
pub fn remove_faces_rejects_unusable_inputs<K: FeatureKernel>() {
    let block = cuboid::<K>(1.0, 1.0, 1.0);
    let other = cuboid::<K>(1.0, 1.0, 1.0);
    let foreign = parts::<K>(&other, ShapeKind::Face).remove(0);
    let all = parts::<K>(&block, ShapeKind::Face);
    let refs: Vec<_> = all.iter().collect();
    must_fail(K::remove_faces(&block, &[&foreign]), "a foreign face");
    must_fail(K::remove_faces(&block, &[]), "no faces");
    must_fail(K::remove_faces(&block, &refs), "every face");
}

/// Coplanar boxes fuse with split faces; unifying leaves a plain block, and reports where
/// the split faces went.
///
/// # Panics
/// If a count or measure differs.
pub fn unify_merges_coplanar_faces<K: FeatureKernel>() {
    let unit = Vec3::new(1.0, 1.0, 1.0);
    let a = must(K::make_box(Point3::ORIGIN, unit), "make_box");
    let b = must(K::make_box(Point3::new(1.0, 0.0, 0.0), unit), "make_box");
    let fused = must(K::fuse(&a, &b), "fuse");
    let unified = must(K::unify(&fused), "unify");
    assert_valid::<K>(&unified.shape, "unified block");
    assert_eq!(
        parts::<K>(&unified.shape, ShapeKind::Face).len(),
        6,
        "a block's faces"
    );
    assert_eq!(
        parts::<K>(&unified.shape, ShapeKind::Edge).len(),
        12,
        "a block's edges"
    );
    volume_is::<K>(&unified.shape, 2.0, "unified block");
    area_is::<K>(&unified.shape, 10.0, "unified block");

    let mut merged = 0;
    for face in parts::<K>(&fused, ShapeKind::Face) {
        if let Some((_, out)) = unified
            .history
            .modified
            .iter()
            .find(|(input, _)| K::is_same(input, &face))
        {
            merged += 1;
            assert!(
                out.iter()
                    .all(|f| contains::<K>(&unified.shape, ShapeKind::Face, f)),
                "a merged face belongs to the result"
            );
        }
    }
    let before = parts::<K>(&fused, ShapeKind::Face).len();
    if before > 6 {
        assert!(merged >= 2, "the split faces are reported modified");
    }
}

/// Unifying a plain box changes nothing measurable.
///
/// # Panics
/// If a count or measure differs.
pub fn unify_leaves_a_plain_shape_alone<K: FeatureKernel>() {
    let block = cuboid::<K>(1.0, 2.0, 3.0);
    let unified = must(K::unify(&block), "unify");
    assert_eq!(parts::<K>(&unified.shape, ShapeKind::Face).len(), 6);
    volume_is::<K>(&unified.shape, 6.0, "unified box");
}

/// Healing leaves a sound shape measuring the same, and turns an inside-out solid right
/// way out.
///
/// # Panics
/// If a measure differs.
pub fn heal_keeps_sound_shapes_and_fixes_inside_out_ones<K: FeatureKernel>() {
    let holed = plate_with_hole::<K>();
    let healed = must(K::heal(&holed, 1e-4), "heal");
    assert_valid::<K>(&healed, "healed plate");
    volume_is::<K>(&healed, 200.0 - 2.0 * PI, "healed plate");

    let inside_out = must(K::reversed(&holed), "reversed");
    let healed = must(K::heal(&inside_out, 1e-4), "heal");
    assert_valid::<K>(&healed, "healed inside-out plate");
    volume_is::<K>(&healed, 200.0 - 2.0 * PI, "healed inside-out plate");
}

/// Volume scales with the determinant, whatever the matrix, and stays positive.
///
/// # Panics
/// If a measure differs.
pub fn transformed_scales_volume_by_the_determinant<K: FeatureKernel>() {
    let block = cuboid::<K>(1.0, 2.0, 3.0);
    let stretch = Transform::scale(Point3::ORIGIN, Vec3::new(2.0, 3.0, 4.0));
    let stretched = must(K::transformed(&block, &stretch), "non-uniform scale");
    assert_valid::<K>(&stretched, "non-uniform scale");
    volume_is::<K>(&stretched, 6.0 * 24.0, "non-uniform scale");

    let shear = Transform {
        rows: [
            [1.0, 0.5, 0.0, 1.0],
            [0.0, 1.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
        ],
    };
    let sheared = must(K::transformed(&block, &shear), "shear");
    assert_valid::<K>(&sheared, "shear");
    volume_is::<K>(&sheared, 6.0, "shear");

    let flipped_stretch = Transform::scale(Point3::ORIGIN, Vec3::new(-2.0, 1.0, 1.0));
    let flipped = must(
        K::transformed(&block, &flipped_stretch),
        "reflected stretch",
    );
    assert_valid::<K>(&flipped, "reflected stretch");
    volume_is::<K>(&flipped, 12.0, "reflected stretch");
    assert!(must(K::center_of_mass(&flipped), "center_of_mass").x < 0.0);

    let uniform = Transform::scale(Point3::new(1.0, 1.0, 1.0), Vec3::new(2.0, 2.0, 2.0));
    let doubled = must(K::transformed(&block, &uniform), "uniform scale");
    volume_is::<K>(&doubled, 48.0, "uniform scale");
}

/// A ball stretched per axis is an ellipsoid.
///
/// # Panics
/// If the volume differs.
pub fn transformed_makes_a_ball_an_ellipsoid<K: FeatureKernel>() {
    let ball = must(K::make_sphere(Point3::ORIGIN, 1.0), "make_sphere");
    let stretch = Transform::scale(Point3::ORIGIN, Vec3::new(1.0, 2.0, 3.0));
    let ellipsoid = must(K::transformed(&ball, &stretch), "ellipsoid");
    assert_valid::<K>(&ellipsoid, "ellipsoid");
    near(
        must(K::volume(&ellipsoid), "volume"),
        ball_volume(1.0) * 6.0,
        FITTED,
        "ellipsoid",
    );
    let bounds = must(K::bounding_box(&ellipsoid), "bounding_box");
    near(bounds.size().z, 6.0, FITTED, "ellipsoid height");
}

/// A mirrored solid measures the same, on the other side of the plane, with a positive
/// volume, whether straight or curved.
///
/// # Panics
/// If a measure differs.
pub fn mirror_keeps_a_positive_volume<K: FeatureKernel>() {
    let holed = plate_with_hole::<K>();
    let want = 200.0 - 2.0 * PI;
    let plane = Plane::new(Point3::new(-1.0, 0.0, 0.0), Dir3::X);
    let mirrored = must(K::transformed(&holed, &Transform::mirror(plane)), "mirror");
    is_solid::<K>(&mirrored, "mirror");
    assert_valid::<K>(&mirrored, "mirror");
    volume_is::<K>(&mirrored, want, "mirror");
    near_point(
        must(K::center_of_mass(&mirrored), "center_of_mass"),
        Point3::new(-7.0, 5.0, 1.0),
        "mirror",
    );
    assert!(
        !K::is_same(&mirrored, &holed),
        "a transformed shape is an independent copy"
    );
}

/// A matrix that flattens space has no result.
///
/// # Panics
/// If it is accepted.
pub fn transformed_rejects_a_singular_matrix<K: FeatureKernel>() {
    let block = cuboid::<K>(1.0, 1.0, 1.0);
    let flat = Transform::scale(Point3::ORIGIN, Vec3::new(1.0, 1.0, 0.0));
    must_fail(K::transformed(&block, &flat), "a flattening matrix");
}

/// Placing moves a shape rigidly, keeps its topology and doesn't disturb the original.
///
/// # Panics
/// If a measure differs.
pub fn placed_moves_rigidly<K: FeatureKernel>() {
    let block = cuboid::<K>(1.0, 2.0, 3.0);
    let turn = Transform::rotation(Axis::Z, Angle::degrees(90.0));
    let place = Transform::translation(Vec3::new(10.0, 0.0, 0.0)).then_after(turn);
    let instance = must(K::placed(&block, &place), "placed");
    assert_valid::<K>(&instance, "instance");
    volume_is::<K>(&instance, 6.0, "instance");
    near_point(
        must(K::center_of_mass(&instance), "center_of_mass"),
        Point3::new(9.0, 0.5, 1.5),
        "instance",
    );
    assert_eq!(parts::<K>(&instance, ShapeKind::Face).len(), 6);
    near_point(
        must(K::center_of_mass(&block), "center_of_mass"),
        Point3::new(0.5, 1.0, 1.5),
        "original",
    );
}

/// Anything other than rotation and translation is refused.
///
/// # Panics
/// If one is accepted.
pub fn placed_rejects_non_rigid_transforms<K: FeatureKernel>() {
    let block = cuboid::<K>(1.0, 1.0, 1.0);
    let uniform = Transform::scale(Point3::ORIGIN, Vec3::new(2.0, 2.0, 2.0));
    let uneven = Transform::scale(Point3::ORIGIN, Vec3::new(1.0, 2.0, 1.0));
    let shear = Transform {
        rows: [
            [1.0, 0.5, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
    };
    must_fail(K::placed(&block, &uniform), "a uniform scale");
    must_fail(K::placed(&block, &uneven), "a non-uniform scale");
    must_fail(K::placed(&block, &shear), "a shear");
    must_fail(
        K::placed(&block, &Transform::mirror(Plane::XY)),
        "a reflection",
    );
}

/// Reversing keeps identity and kind, and twice gives back the same measures.
///
/// # Panics
/// If they differ.
pub fn reversed_keeps_identity<K: FeatureKernel>() {
    let face = square_at::<K>(Point3::ORIGIN, 2.0);
    let flipped = must(K::reversed(&face), "reversed");
    assert!(K::is_same(&face, &flipped), "reversing keeps identity");
    assert_eq!(must(K::kind(&flipped), "kind"), ShapeKind::Face);
    area_is::<K>(&flipped, 4.0, "reversed face");
    let twice = must(K::reversed(&flipped), "reversed");
    assert!(K::is_same(&face, &twice));

    // A face's normal shows through what thickening does with it.
    let up = must(K::thicken(&face, 1.0), "thicken");
    let down = must(K::thicken(&flipped, 1.0), "thicken");
    let (up, down) = (
        must(K::center_of_mass(&up), "center_of_mass").z,
        must(K::center_of_mass(&down), "center_of_mass").z,
    );
    assert!(
        up * down < 0.0,
        "reversing flips the normal: {up} and {down}"
    );
}

/// Nothing here changes its input.
///
/// # Panics
/// If an input changes.
pub fn inputs_are_left_untouched<K: FeatureKernel>() {
    let block = cuboid::<K>(4.0, 4.0, 4.0);
    let top = part_at::<K>(&block, ShapeKind::Face, Point3::new(2.0, 2.0, 4.0));
    must(K::shell(&block, &[&top], -1.0), "shell");
    must(K::offset(&block, 1.0), "offset");
    must(K::unify(&block), "unify");
    must(K::heal(&block, 1e-4), "heal");
    must(
        K::transformed(&block, &Transform::mirror(Plane::XY)),
        "mirror",
    );
    must(
        K::placed(&block, &Transform::translation(Vec3::new(1.0, 0.0, 0.0))),
        "placed",
    );
    must(K::reversed(&block), "reversed");
    volume_is::<K>(&block, 64.0, "input block");
    assert_eq!(parts::<K>(&block, ShapeKind::Face).len(), 6);
    near_point(
        must(K::center_of_mass(&block), "center_of_mass"),
        Point3::new(2.0, 2.0, 2.0),
        "input block",
    );
}

/// The linear and quadratic terms of a polygon's area as it moves in by `k`, for sides
/// `perimeter` long and corners of `turning`: `A(k) = A - perimeter k + turning k^2`, where
/// a right-angled corner adds 1 going out and a reflex one takes 1 away, and round corners
/// add their total turn. Integrated over a taper's height this is its volume.
fn tapered_volume(area: f64, perimeter: f64, turning: f64, h: f64, taper: Angle) -> f64 {
    let t = taper.as_radians().tan();
    area * h - perimeter * t * h * h / 2.0 + turning * t * t * h * h * h / 3.0
}

/// A concave outline tapers corner by corner, the reflex corner moving the other way, and
/// tangent flat and round sides taper together.
///
/// # Panics
/// If a volume differs.
pub fn taper_handles_concave_and_tangent_outlines<K: FeatureKernel>() {
    let corners = [
        (0.0, 0.0),
        (4.0, 0.0),
        (4.0, 2.0),
        (2.0, 2.0),
        (2.0, 4.0),
        (0.0, 4.0),
    ];
    let ell = must(
        K::make_polygon_face(&corners.map(|(x, y)| Point3::new(x, y, 0.0))),
        "make_polygon_face",
    );
    for degrees in [10.0, -10.0] {
        let taper = Angle::degrees(degrees);
        let solid = must(
            K::extrude_tapered(&ell, Vec3::new(0.0, 0.0, 1.0), taper),
            format_args!("L taper by {degrees}"),
        );
        assert_valid::<K>(&solid, "tapered L");
        // Five convex corners and one reflex: a net turning of 4.
        volume_is::<K>(
            &solid,
            tapered_volume(12.0, 16.0, 4.0, 1.0, taper),
            "tapered L",
        );
    }

    let plate = cuboid::<K>(10.0, 8.0, 1.0);
    let uprights: Vec<_> = parts::<K>(&plate, ShapeKind::Edge)
        .into_iter()
        .filter(|e| {
            let size = must(K::bounding_box(e), "bounding_box").size();
            size.x < 1e-6 && size.y < 1e-6
        })
        .collect();
    let refs: Vec<_> = uprights.iter().collect();
    let r = 2.0;
    let rounded = must(K::fillet(&plate, &refs, r), "fillet");
    let base = parts::<K>(&rounded, ShapeKind::Face)
        .into_iter()
        .find(|f| must(K::center_of_mass(f), "center_of_mass").z.abs() < 1e-6)
        .expect("the plate has a base");
    let area = must(K::area(&base), "area");
    let perimeter = 2.0 * (6.0 + 4.0) + 2.0 * PI * r;
    let (h, taper) = (2.0, Angle::degrees(10.0));
    let solid = must(
        K::extrude_tapered(&base, Vec3::new(0.0, 0.0, h), taper),
        "tapered rounded rectangle",
    );
    assert_valid::<K>(&solid, "tapered rounded rectangle");
    // Round corners shrink with the rest and add a full turn in all.
    volume_is::<K>(
        &solid,
        tapered_volume(area, perimeter, PI, h, taper),
        "tapered rounded rectangle",
    );
}

/// The slope of an oblique extrusion is measured from its direction. Each side stays
/// straight, so a level slice is a rectangle whose width changes linearly: integrate that.
///
/// # Panics
/// If the volume differs.
pub fn taper_along_an_oblique_extrusion<K: FeatureKernel>() {
    let (a, by, taper) = (10.0, Vec3::new(1.0, 0.0, 3.0), Angle::degrees(5.0));
    let solid = must(
        K::extrude_tapered(&square_at::<K>(Point3::ORIGIN, a), by, taper),
        "oblique taper",
    );
    assert_valid::<K>(&solid, "oblique taper");

    let (sin, cos) = taper.as_radians().sin_cos();
    let length = by.length();
    let d = [by.x / length, by.z / length];
    // A side hinged along Y: its normal in the XZ plane, turned from square to the direction
    // toward it by the taper. The two sides differ because the direction leans.
    let lean = |outward: f64| {
        let square = [outward * d[1], -outward * d[0]];
        let n = [square[0] * cos + d[0] * sin, square[1] * cos + d[1] * sin];
        -n[1] / n[0]
    };
    let shrink_x = lean(-1.0) - lean(1.0);
    // A side hinged along X: its normal has no X part, so only the direction's Z part counts.
    let nz = sin / d[1];
    let shrink_y = 2.0 * nz / (1.0 - nz * nz).sqrt();
    let h = by.z;
    let want =
        a * a * h - a * (shrink_x + shrink_y) * h * h / 2.0 + shrink_x * shrink_y * h * h * h / 3.0;
    volume_is::<K>(&solid, want, "oblique taper");
}

/// However thick a solid's walls, what is asked for either happens or is an error, right up
/// to and past the point where the inside vanishes.
///
/// # Panics
/// If a collapsed result is returned.
pub fn offsets_near_collapse_are_exact_or_rejected<K: FeatureKernel>() {
    let block = cuboid::<K>(2.0, 2.0, 2.0);
    let bar = must(K::make_cylinder(Axis::Z, 1.0, 10.0), "make_cylinder");
    type Exact = fn(f64) -> f64;
    let box_volume: Exact = |d| (2.0 - 2.0 * d).powi(3);
    let bar_volume: Exact = |d| PI * (1.0 - d) * (1.0 - d) * (10.0 - 2.0 * d);
    for (shape, volume) in [(&block, box_volume), (&bar, bar_volume)] {
        for d in [0.5_f64, 0.9] {
            let smaller = must(K::offset(shape, -d), format_args!("offset by -{d}"));
            near(
                must(K::volume(&smaller), "volume"),
                volume(d),
                MEASURE,
                "smaller",
            );
        }
        for d in [1.0, 1.0 + 1e-7, 1.2, 1.9, 2.0, 3.0, 50.0] {
            must_fail(K::offset(shape, -d), format_args!("offset by -{d}"));
            must_fail(
                K::shell(shape, &[], -d),
                format_args!("cavity of walls {d}"),
            );
        }
    }
    // Past the axis a cylinder's wall would turn inside out rather than vanish.
    let wall = parts::<K>(&bar, ShapeKind::Face)
        .into_iter()
        .find(|f| must(K::surface_kind(f), "surface_kind") == occt_kernel::SurfaceKind::Cylinder)
        .expect("a cylinder has a wall");
    must_fail(K::thicken(&wall, -1.2), "thicken past the axis");
    let top = part_at::<K>(&bar, ShapeKind::Face, Point3::new(0.0, 0.0, 10.0));
    must_fail(K::shell(&bar, &[&top], -1.2), "walls past the axis");
}

/// A sheet offsets to a parallel sheet of the same kind, on the side its normal points to.
///
/// # Panics
/// If a measure differs.
pub fn offset_of_a_face_is_a_parallel_face<K: FeatureKernel>() {
    let face = square_at::<K>(Point3::ORIGIN, 2.0);
    let moved = must(K::offset(&face, 1.5), "offset face");
    assert_eq!(must(K::kind(&moved), "kind"), ShapeKind::Face);
    area_is::<K>(&moved, 4.0, "offset face");
    near_point(
        must(K::center_of_mass(&moved), "center_of_mass"),
        Point3::new(1.0, 1.0, 1.5),
        "offset face",
    );
}

/// An L's reflex corner shows in a mitred offset: the outline grows by its perimeter and
/// corners, and the height by twice the distance.
///
/// # Panics
/// If the volume differs.
pub fn offset_of_a_concave_prism<K: FeatureKernel>() {
    let corners = [
        (0.0, 0.0),
        (4.0, 0.0),
        (4.0, 2.0),
        (2.0, 2.0),
        (2.0, 4.0),
        (0.0, 4.0),
    ];
    let ell = must(
        K::make_polygon_face(&corners.map(|(x, y)| Point3::new(x, y, 0.0))),
        "make_polygon_face",
    );
    let prism = must(K::extrude(&ell, Vec3::new(0.0, 0.0, 1.0)), "extrude");
    let d = 0.5;
    let grown = must(K::offset(&prism, d), "offset");
    assert_valid::<K>(&grown, "offset L");
    volume_is::<K>(
        &grown,
        (12.0 + 16.0 * d + 4.0 * d * d) * (1.0 + 2.0 * d),
        "offset L",
    );
}

/// A multi-face sheet thickens too: a square tube's walls go wholly to one side.
///
/// # Panics
/// If the volume differs.
pub fn thicken_a_shell<K: FeatureKernel>() {
    let (a, h, t) = (2.0, 3.0, 0.5);
    let wire = outline::<K>(&square_at::<K>(Point3::ORIGIN, a));
    let tube = must(K::extrude(&wire, Vec3::new(0.0, 0.0, h)), "extrude");
    assert_eq!(must(K::kind(&tube), "kind"), ShapeKind::Shell);
    let inside = (a * a - (a - 2.0 * t) * (a - 2.0 * t)) * h;
    let outside = ((a + 2.0 * t) * (a + 2.0 * t) - a * a) * h;
    let mut volumes = [0.0; 2];
    for (signed, volume) in [t, -t].into_iter().zip(&mut volumes) {
        let thick = must(K::thicken(&tube, signed), "thick tube");
        assert_valid::<K>(&thick, "thick tube");
        *volume = must(K::volume(&thick), "volume");
    }
    volumes.sort_by(f64::total_cmp);
    near(volumes[0], inside, MEASURE, "tube thickened inward");
    near(volumes[1], outside, MEASURE, "tube thickened outward");
}

/// A cylinder's wall drafts to a cone, and a hinge away from a face moves all of it.
///
/// # Panics
/// If a volume differs.
pub fn draft_a_round_face_and_about_a_distant_plane<K: FeatureKernel>() {
    let (r, h, angle) = (5.0, 2.0, Angle::degrees(10.0));
    let t = angle.as_radians().tan();
    let bar = must(K::make_cylinder(Axis::Z, r, h), "make_cylinder");
    let wall = parts::<K>(&bar, ShapeKind::Face)
        .into_iter()
        .find(|f| must(K::surface_kind(f), "surface_kind") == occt_kernel::SurfaceKind::Cylinder)
        .expect("a cylinder has a wall");
    let cone = must(K::draft(&bar, &[&wall], Dir3::Z, Plane::XY, angle), "draft");
    assert_valid::<K>(&cone.shape, "drafted cylinder");
    let top = r - h * t;
    volume_is::<K>(
        &cone.shape,
        PI * h / 3.0 * (r * r + r * top + top * top),
        "drafted cylinder",
    );

    let a = 10.0;
    let block = cuboid::<K>(a, a, a);
    let side = part_at::<K>(&block, ShapeKind::Face, Point3::new(a, a / 2.0, a / 2.0));
    let below = Plane::new(Point3::new(0.0, 0.0, -5.0), Dir3::Z);
    let far = must(K::draft(&block, &[&side], Dir3::Z, below, angle), "draft");
    assert_valid::<K>(&far.shape, "draft from below");
    // The side moves in by 5t at the bottom and 15t at the top.
    volume_is::<K>(
        &far.shape,
        a * a * a - a * a * (5.0 + 15.0) * t / 2.0,
        "draft from below",
    );
}

/// Whatever history reports exists: outputs are in the result and what's deleted is not.
///
/// # Panics
/// If the history names anything else.
pub fn history_names_only_what_exists<K: FeatureKernel>() {
    let holed = plate_with_hole::<K>();
    let wall = parts::<K>(&holed, ShapeKind::Face)
        .into_iter()
        .find(|f| must(K::surface_kind(f), "surface_kind") == occt_kernel::SurfaceKind::Cylinder)
        .expect("the hole has a wall");
    let side = part_at::<K>(&holed, ShapeKind::Face, Point3::new(10.0, 5.0, 1.0));
    let top = part_at::<K>(&holed, ShapeKind::Face, Point3::new(5.0, 5.0, 2.0));
    let by = Vec3::new(0.0, 0.0, 5.0);
    let square = square_at::<K>(Point3::ORIGIN, 2.0);
    let path = line::<K>(Point3::ORIGIN, by);
    let frame_path = outline::<K>(&square_at::<K>(Point3::ORIGIN, 10.0));
    let cross = must(
        K::make_polygon_face(&[
            Point3::new(0.0, -1.0, -1.0),
            Point3::new(0.0, 1.0, -1.0),
            Point3::new(0.0, 1.0, 1.0),
            Point3::new(0.0, -1.0, 1.0),
        ]),
        "make_polygon_face",
    );
    let plain = SweepOptions::default();
    let results = [
        ("extrude", must(K::extrude_modeled(&square, by), "extrude")),
        (
            "revolve",
            must(
                K::revolve_modeled(&section::<K>(1.0, 2.0, 1.0), Axis::Z, Angle::FULL_TURN),
                "revolve",
            ),
        ),
        ("sweep", must(K::sweep(&square, &path, &plain), "sweep")),
        (
            "closed sweep",
            must(K::sweep(&cross, &frame_path, &plain), "sweep"),
        ),
        ("shell", must(K::shell(&holed, &[&top], -0.25), "shell")),
        ("cavity", must(K::shell(&holed, &[], -0.25), "shell")),
        (
            "draft",
            must(
                K::draft(&holed, &[&side], Dir3::Z, Plane::XY, Angle::degrees(5.0)),
                "draft",
            ),
        ),
        (
            "remove",
            must(K::remove_faces(&holed, &[&wall]), "remove_faces"),
        ),
        ("unify", must(K::unify(&holed), "unify")),
    ];
    for (what, result) in &results {
        let mut all = Vec::new();
        for kind in [ShapeKind::Face, ShapeKind::Edge, ShapeKind::Vertex] {
            all.extend(parts::<K>(&result.shape, kind));
        }
        let exists = |s: &K::Shape| all.iter().any(|p| K::is_same(p, s));
        for (_, outputs) in result
            .history
            .modified
            .iter()
            .chain(&result.history.generated)
        {
            assert!(
                outputs.iter().all(exists),
                "{what}: history names a shape not in the result"
            );
        }
        assert!(
            !result.history.deleted.iter().any(exists),
            "{what}: history calls a shape of the result deleted"
        );
    }
}

macro_rules! check_list {
    ($($name:ident),* $(,)?) => {
        /// Every check of this module as a `(name, function)` pair, in a stable order.
        #[must_use]
        pub fn checks<K: Kernel + AdvancedSweeps + Features>() -> Vec<(&'static str, fn())> {
            vec![$((stringify!($name), $name::<K> as fn())),*]
        }
    };
}

check_list![
    extrude_modeled_maps_edges_to_faces,
    revolve_modeled_maps_edges_to_faces,
    tapered_square_is_a_frustum,
    taper_follows_the_extrusion,
    taper_handles_round_sides_and_holes,
    taper_handles_concave_and_tangent_outlines,
    taper_along_an_oblique_extrusion,
    taper_too_steep_is_rejected,
    sweep_along_a_line_equals_an_extrusion,
    sweep_along_a_slanted_line,
    sweep_circle_along_an_arc_obeys_pappus,
    sweep_with_fixed_orientation_keeps_the_section_parallel,
    sweep_without_solid_makes_a_sheet,
    sweep_of_a_face_keeps_its_holes,
    sweep_around_a_closed_path_makes_a_frame,
    sweep_reports_side_faces,
    sweep_rejects_unusable_inputs,
    loft_between_squares_is_a_frustum,
    loft_through_several_sections,
    loft_to_a_point_and_between_circles,
    loft_as_a_sheet_has_no_ends,
    loft_rejects_unusable_inputs,
    shell_with_an_open_face,
    shell_grown_outward,
    shell_with_two_open_faces,
    shell_with_no_open_faces_has_a_cavity,
    shell_of_a_cylinder,
    shell_rejects_unusable_inputs,
    offset_of_a_box_and_a_sphere,
    offset_narrows_a_hole,
    offset_past_the_middle_is_rejected,
    offsets_near_collapse_are_exact_or_rejected,
    offset_of_a_face_is_a_parallel_face,
    offset_of_a_concave_prism,
    thicken_a_shell,
    thicken_a_face,
    thicken_rejects_other_kinds,
    draft_one_face_removes_a_wedge,
    draft_four_faces_makes_a_frustum,
    draft_about_the_middle_keeps_the_volume,
    draft_reports_the_modified_face,
    draft_a_round_face_and_about_a_distant_plane,
    draft_rejects_unusable_inputs,
    remove_faces_fills_a_hole,
    remove_faces_takes_off_a_boss,
    remove_faces_rejects_unusable_inputs,
    unify_merges_coplanar_faces,
    unify_leaves_a_plain_shape_alone,
    heal_keeps_sound_shapes_and_fixes_inside_out_ones,
    transformed_scales_volume_by_the_determinant,
    transformed_makes_a_ball_an_ellipsoid,
    mirror_keeps_a_positive_volume,
    transformed_rejects_a_singular_matrix,
    placed_moves_rigidly,
    placed_rejects_non_rigid_transforms,
    reversed_keeps_identity,
    inputs_are_left_untouched,
    history_names_only_what_exists,
];
