//! Joining surfaces, against shapes whose continuity is known without computing it.
//!
//! A sphere continues itself at G2 everywhere, a cube's faces meet at a quarter turn, and a
//! circular fillet is tangent to a plane but changes curvature abruptly by one over its radius.

use std::f64::consts::FRAC_PI_2;

use occt_kernel::{
    Angle, Axis, BlendContinuity, BlendSide, ContinuityReport, Curve, Dir3, FaceSides, FillEdge,
    Frame, Kernel, ParamRange, Point3, ShapeKind, Surface, Vec3,
};

use crate::support::{assert_valid, must, near, parts};

/// A smooth join: tangency within this, in degrees. Constructions fit to within about
/// 1e-3 of model size, which turns the tangent plane by far less than this.
const TANGENT_DEGREES: f64 = 0.5;

/// Every check as a `(name, function)` pair, in a stable order.
#[must_use]
pub fn checks<K: Kernel>() -> Vec<(&'static str, fn())> {
    macro_rules! list {
        ($($name:ident),* $(,)?) => { vec![$((stringify!($name), $name::<K> as fn())),*] };
    }
    list![
        a_sphere_patch_continues_its_neighbour_at_g2,
        a_cube_edge_is_a_quarter_turn,
        a_circular_fillet_is_g1_but_not_g2,
        continuity_needs_an_edge_on_both_faces,
        a_g2_blend_of_planes_leaves_them_with_no_curvature,
        a_g2_blend_matches_a_cylinder_a_g1_blend_does_not,
        a_g2_blend_matches_a_sphere,
        a_g2_blend_matches_freeform_moved_faces,
        a_blend_sews_to_the_faces_it_joins,
        blends_reject_unusable_input,
        a_tangent_fill_follows_a_sphere_better_than_a_free_one,
        a_tangent_fill_that_cannot_be_tangent_fails,
        extending_grows_only_the_chosen_sides,
        extending_a_cylinder_follows_it,
        extending_drops_holes,
        extending_rejects_bad_input,
    ]
}

const fn p(x: f64, y: f64, z: f64) -> Point3 {
    Point3::new(x, y, z)
}

fn segment<K: Kernel>(a: Point3, b: Point3) -> K::Shape {
    let direction = must(Dir3::new(b - a).map_err(Into::into), "direction");
    must(
        K::make_edge(
            &Curve::Line {
                origin: a,
                direction,
            },
            Some(ParamRange {
                start: 0.0,
                end: a.distance(b),
            }),
        ),
        "make_edge",
    )
}

fn closed_wire<K: Kernel>(points: &[Point3]) -> K::Shape {
    let edges: Vec<K::Shape> = (0..points.len())
        .map(|i| segment::<K>(points[i], points[(i + 1) % points.len()]))
        .collect();
    let refs: Vec<&K::Shape> = edges.iter().collect();
    must(K::make_wire(&refs), "make_wire")
}

fn sphere_patch<K: Kernel>(radius: f64, u: (f64, f64), v: (f64, f64)) -> K::Shape {
    surface_patch::<K>(
        Surface::Sphere {
            frame: Frame::WORLD,
            radius,
        },
        u,
        v,
    )
}

fn cylinder_patch<K: Kernel>(radius: f64, u: (f64, f64), v: (f64, f64)) -> K::Shape {
    surface_patch::<K>(
        Surface::Cylinder {
            frame: Frame::WORLD,
            radius,
        },
        u,
        v,
    )
}

fn surface_patch<K: Kernel>(surface: Surface, u: (f64, f64), v: (f64, f64)) -> K::Shape {
    let bounds = (
        ParamRange {
            start: u.0,
            end: u.1,
        },
        ParamRange {
            start: v.0,
            end: v.1,
        },
    );
    must(
        K::make_surface_face(&surface, Some(bounds), None, &[]),
        "make_surface_face",
    )
}

/// The edge of `face` whose middle is closest to `near`.
fn edge_near<K: Kernel>(face: &K::Shape, target: Point3) -> K::Shape {
    parts::<K>(face, ShapeKind::Edge)
        .into_iter()
        .min_by(|a, b| {
            middle::<K>(a)
                .distance(target)
                .total_cmp(&middle::<K>(b).distance(target))
        })
        .expect("a face has edges")
}

fn middle<K: Kernel>(edge: &K::Shape) -> Point3 {
    let geometry = must(K::edge_geometry(edge), "edge_geometry");
    let at = 0.5 * (geometry.range.start + geometry.range.end);
    must(K::edge_point(edge, at), "edge_point").point
}

fn measure<K: Kernel>(a: &K::Shape, b: &K::Shape, edge: &K::Shape) -> ContinuityReport {
    must(K::measure_continuity(a, b, edge), "measure_continuity")
}

fn on_sphere(radius: f64, u: f64, v: f64) -> Point3 {
    p(
        radius * v.cos() * u.cos(),
        radius * v.cos() * u.sin(),
        radius * v.sin(),
    )
}

/// Two flat faces at a right angle with a gap between them, and the edges facing the gap.
fn floor_and_wall<K: Kernel>() -> ([K::Shape; 2], [K::Shape; 2]) {
    let floor = must(
        K::make_polygon_face(&[p(0., -9., 0.), p(9., -9., 0.), p(9., 0., 0.), p(0., 0., 0.)]),
        "floor",
    );
    let wall = must(
        K::make_polygon_face(&[p(0., 4., 4.), p(9., 4., 4.), p(9., 4., 9.), p(0., 4., 9.)]),
        "wall",
    );
    let floor_edge = edge_near::<K>(&floor, p(4.5, 0., 0.));
    let wall_edge = edge_near::<K>(&wall, p(4.5, 4., 4.));
    ([floor, wall], [floor_edge, wall_edge])
}

/// # Panics
/// If neighbouring patches of one sphere don't measure as continuing each other.
pub fn a_sphere_patch_continues_its_neighbour_at_g2<K: Kernel>() {
    let a = sphere_patch::<K>(10.0, (-0.5, 0.0), (-0.4, 0.4));
    let b = sphere_patch::<K>(10.0, (0.0, 0.5), (-0.4, 0.4));
    let edge = edge_near::<K>(&a, on_sphere(10.0, 0.0, 0.0));
    let report = measure::<K>(&a, &b, &edge);
    assert!(report.gap < 1e-6, "gap {}", report.gap);
    assert!(report.angle.as_degrees() < 1e-3, "{:?}", report.angle);
    assert!(report.curvature_gap < 1e-6, "{}", report.curvature_gap);
}

/// # Panics
/// If a cube's edge doesn't measure as a right angle with no curvature.
pub fn a_cube_edge_is_a_quarter_turn<K: Kernel>() {
    let cube = must(
        K::make_box(Point3::ORIGIN, Vec3::new(2.0, 2.0, 2.0)),
        "make_box",
    );
    let faces = parts::<K>(&cube, ShapeKind::Face);
    let edge = parts::<K>(&faces[0], ShapeKind::Edge).remove(0);
    let other = faces[1..]
        .iter()
        .find(|f| {
            parts::<K>(f, ShapeKind::Edge)
                .iter()
                .any(|e| K::is_same(e, &edge))
        })
        .expect("every edge of a cube joins two faces");
    let report = measure::<K>(&faces[0], other, &edge);
    near(report.angle.as_degrees(), 90.0, 1e-6, "cube edge");
    assert!(report.curvature_gap < 1e-9, "{}", report.curvature_gap);
}

/// The reason G2 blends exist: a circular fillet joins a plane smoothly, but curvature jumps
/// from nothing to one over its radius.
///
/// # Panics
/// If the fillet isn't tangent, or the jump isn't one over its radius.
pub fn a_circular_fillet_is_g1_but_not_g2<K: Kernel>() {
    let radius = 2.0;
    let cube = must(
        K::make_box(Point3::ORIGIN, Vec3::new(10.0, 10.0, 10.0)),
        "make_box",
    );
    let edges = parts::<K>(&cube, ShapeKind::Edge);
    let filleted = must(K::fillet(&cube, &[&edges[0]], radius), "fillet");
    let faces = parts::<K>(&filleted, ShapeKind::Face);
    let mut joints = 0;
    for (i, a) in faces.iter().enumerate() {
        let a_is_round = !matches!(
            must(K::face_geometry(a), "face_geometry").surface,
            Surface::Plane { .. }
        );
        if !a_is_round {
            continue;
        }
        for edge in parts::<K>(a, ShapeKind::Edge) {
            for (j, b) in faces.iter().enumerate() {
                if i == j
                    || !parts::<K>(b, ShapeKind::Edge)
                        .iter()
                        .any(|e| K::is_same(e, &edge))
                {
                    continue;
                }
                let report = measure::<K>(a, b, &edge);
                if report.angle.as_degrees() > TANGENT_DEGREES {
                    continue; // The fillet's ends, where it stops against the cube's sides.
                }
                near(report.curvature_gap, 1.0 / radius, 1e-3, "curvature jump");
                joints += 1;
            }
        }
    }
    assert_eq!(joints, 2, "a fillet has two tangent sides");
}

/// # Panics
/// If an edge away from a face is measured anyway, or a wrong kind is accepted.
pub fn continuity_needs_an_edge_on_both_faces<K: Kernel>() {
    let ([floor, wall], [floor_edge, wall_edge]) = floor_and_wall::<K>();
    assert!(K::measure_continuity(&floor, &wall, &floor_edge).is_err());
    assert!(K::measure_continuity(&floor, &wall, &wall_edge).is_err());
    assert!(K::measure_continuity(&floor_edge, &wall, &floor_edge).is_err());
    assert!(K::measure_continuity(&floor, &wall, &floor).is_err());
}

fn blend_of<K: Kernel>(
    a: (&K::Shape, &K::Shape),
    b: (&K::Shape, &K::Shape),
    continuity: BlendContinuity,
) -> K::Shape {
    let face = must(
        K::blend_faces(
            BlendSide {
                edge: a.0,
                face: a.1,
            },
            BlendSide {
                edge: b.0,
                face: b.1,
            },
            continuity,
            1.0,
        ),
        "blend_faces",
    );
    assert_valid::<K>(&face, "blend");
    face
}

/// # Panics
/// If the blend creases or curves where it leaves a flat face.
pub fn a_g2_blend_of_planes_leaves_them_with_no_curvature<K: Kernel>() {
    let ([floor, wall], [floor_edge, wall_edge]) = floor_and_wall::<K>();
    let blend = blend_of::<K>(
        (&floor_edge, &floor),
        (&wall_edge, &wall),
        BlendContinuity::G2,
    );
    for (face, edge) in [(&floor, &floor_edge), (&wall, &wall_edge)] {
        let report = measure::<K>(&blend, face, edge);
        assert!(report.gap < 1e-5, "gap {}", report.gap);
        assert!(report.angle.as_degrees() < 0.01, "{:?}", report.angle);
        assert!(report.curvature_gap < 1e-4, "{}", report.curvature_gap);
    }
}

/// # Panics
/// If a G2 blend doesn't pick up a cylinder's curvature, or a G1 blend does: that would mean
/// the measurement can't tell them apart.
pub fn a_g2_blend_matches_a_cylinder_a_g1_blend_does_not<K: Kernel>() {
    let radius = 5.0;
    let a = cylinder_patch::<K>(radius, (-1.5, -0.5), (0.0, 4.0));
    let b = cylinder_patch::<K>(radius, (0.5, 1.5), (0.0, 4.0));
    let at = |u: f64| p(radius * u.cos(), radius * u.sin(), 2.0);
    let ea = edge_near::<K>(&a, at(-0.5));
    let eb = edge_near::<K>(&b, at(0.5));
    let kappa = 1.0 / radius;
    let g2 = blend_of::<K>((&ea, &a), (&eb, &b), BlendContinuity::G2);
    for (face, edge) in [(&a, &ea), (&b, &eb)] {
        let report = measure::<K>(&g2, face, edge);
        assert!(report.angle.as_degrees() < 0.01, "{:?}", report.angle);
        assert!(
            report.curvature_gap < 0.01 * kappa,
            "G2 left {} against {kappa}",
            report.curvature_gap
        );
    }
    let g1 = blend_of::<K>((&ea, &a), (&eb, &b), BlendContinuity::G1);
    let report = measure::<K>(&g1, &a, &ea);
    assert!(report.angle.as_degrees() < 0.01, "{:?}", report.angle);
    assert!(
        report.curvature_gap > 0.5 * kappa,
        "G1 has no reason to match curvature, yet left {}",
        report.curvature_gap
    );
}

/// # Panics
/// If a blend between two bands of a sphere doesn't match it along edges of unequal length.
pub fn a_g2_blend_matches_a_sphere<K: Kernel>() {
    let radius = 10.0;
    let a = sphere_patch::<K>(radius, (-0.5, 0.5), (-0.9, -0.3));
    let b = sphere_patch::<K>(radius, (-0.5, 0.5), (0.2, 0.7));
    let ea = edge_near::<K>(&a, on_sphere(radius, 0.0, -0.3));
    let eb = edge_near::<K>(&b, on_sphere(radius, 0.0, 0.2));
    let blend = blend_of::<K>((&ea, &a), (&eb, &b), BlendContinuity::G2);
    let kappa = 1.0 / radius;
    for (face, edge) in [(&a, &ea), (&b, &eb)] {
        let report = measure::<K>(&blend, face, edge);
        assert!(report.gap < 1e-4, "gap {}", report.gap);
        assert!(report.angle.as_degrees() < 0.01, "{:?}", report.angle);
        assert!(
            report.curvature_gap < 0.01 * kappa,
            "G2 left {} against {kappa}",
            report.curvature_gap
        );
    }
}

/// A dome, its first copy turned about its corner, so that nothing about the surfaces is
/// analytic or sits where it was built.
fn domes<K: Kernel>() -> ([K::Shape; 2], [K::Shape; 2]) {
    let corners = [p(0., 0., 0.), p(4., 0., 0.), p(4., 4., 0.), p(0., 4., 0.)];
    let edges: Vec<K::Shape> = (0..4)
        .map(|i| segment::<K>(corners[i], corners[(i + 1) % 4]))
        .collect();
    let refs: Vec<&K::Shape> = edges.iter().collect();
    let dome = must(K::fill_boundary(&refs, &[p(2., 2., 1.)]), "dome");
    let along_x = Axis::new(p(4., 0., 0.), Dir3::Y);
    let moved = must(K::translate(&dome, Vec3::new(9., 0., 0.)), "translate");
    let tilted = must(K::rotate(&moved, along_x, Angle::degrees(20.)), "rotate");
    let ea = edge_near::<K>(&dome, p(4., 2., 0.));
    let far = must(K::bounding_box(&tilted), "bounding_box");
    let eb = parts::<K>(&tilted, ShapeKind::Edge)
        .into_iter()
        .min_by(|a, b| {
            let along = |e: &K::Shape| {
                let m = middle::<K>(e);
                (m.x - far.min.x).abs() + (m.y - 2.0).abs()
            };
            along(a).total_cmp(&along(b))
        })
        .expect("a face has edges");
    ([dome, tilted], [ea, eb])
}

/// # Panics
/// If the blend loses curvature on a freeform surface under a placement.
pub fn a_g2_blend_matches_freeform_moved_faces<K: Kernel>() {
    let ([a, b], [ea, eb]) = domes::<K>();
    let blend = blend_of::<K>((&ea, &a), (&eb, &b), BlendContinuity::G2);
    for (face, edge) in [(&a, &ea), (&b, &eb)] {
        let report = measure::<K>(&blend, face, edge);
        assert!(report.gap < 1e-3, "gap {}", report.gap);
        assert!(report.angle.as_degrees() < 0.05, "{:?}", report.angle);
        assert!(report.curvature_gap < 2e-3, "{}", report.curvature_gap);
    }
}

/// # Panics
/// If sewing doesn't merge the patch with both faces into one valid shell.
pub fn a_blend_sews_to_the_faces_it_joins<K: Kernel>() {
    let ([floor, wall], [floor_edge, wall_edge]) = floor_and_wall::<K>();
    let blend = blend_of::<K>(
        (&floor_edge, &floor),
        (&wall_edge, &wall),
        BlendContinuity::G2,
    );
    let shell = must(K::sew(&[&floor, &blend, &wall], 1e-3), "sew");
    assert_valid::<K>(&shell, "sewn");
    assert_eq!(parts::<K>(&shell, ShapeKind::Shell).len(), 1);
    assert_eq!(parts::<K>(&shell, ShapeKind::Face).len(), 3);
    // Four sides each, two shared.
    assert_eq!(parts::<K>(&shell, ShapeKind::Edge).len(), 10);
}

/// # Panics
/// If nonsense makes a patch.
pub fn blends_reject_unusable_input<K: Kernel>() {
    let ([floor, wall], [floor_edge, wall_edge]) = floor_and_wall::<K>();
    let side = |edge, face| BlendSide { edge, face };
    let good = side(&floor_edge, &floor);
    let other = side(&wall_edge, &wall);
    let g2 = BlendContinuity::G2;
    assert!(
        K::blend_faces(good, good, g2, 1.0).is_err(),
        "one edge twice"
    );
    assert!(K::blend_faces(good, other, g2, 0.0).is_err(), "no tension");
    assert!(K::blend_faces(good, other, g2, f64::NAN).is_err());
    assert!(
        K::blend_faces(side(&wall_edge, &floor), other, g2, 1.0).is_err(),
        "off the face"
    );
    assert!(
        K::blend_faces(side(&floor, &floor), other, g2, 1.0).is_err(),
        "not an edge"
    );
    assert!(K::blend_faces(side(&floor_edge, &floor_edge), other, g2, 1.0).is_err());

    // Edges that meet at a corner have nothing between them there.
    let e = parts::<K>(&floor, ShapeKind::Edge);
    let touching = e
        .iter()
        .find(|x| {
            !K::is_same(x, &floor_edge)
                && must(K::edge_point(x, 0.0), "edge_point").point.y.abs() < 1e-9
        })
        .unwrap_or(&e[1]);
    assert!(K::blend_faces(good, side(touching, &floor), g2, 1.0).is_err());
}

/// # Panics
/// If tangent sides don't help: they are what turns a dome into the sphere it fills.
pub fn a_tangent_fill_follows_a_sphere_better_than_a_free_one<K: Kernel>() {
    let radius = 10.0;
    let h = 0.4;
    let w = 0.3;
    let neighbours = [
        sphere_patch::<K>(radius, (-h - w, -h), (-h, h)),
        sphere_patch::<K>(radius, (h, h + w), (-h, h)),
        sphere_patch::<K>(radius, (-h, h), (-h - w, -h)),
        sphere_patch::<K>(radius, (-h, h), (h, h + w)),
    ];
    let targets = [
        on_sphere(radius, -h, 0.0),
        on_sphere(radius, h, 0.0),
        on_sphere(radius, 0.0, -h),
        on_sphere(radius, 0.0, h),
    ];
    let edges: Vec<K::Shape> = neighbours
        .iter()
        .zip(targets)
        .map(|(face, target)| edge_near::<K>(face, target))
        .collect();
    let sides: Vec<_> = edges
        .iter()
        .zip(&neighbours)
        .map(|(edge, face)| FillEdge {
            edge,
            tangent_to: Some(face),
        })
        .collect();
    let tangent = must(K::fill_boundary_tangent(&sides, &[]), "tangent fill");
    assert_valid::<K>(&tangent, "tangent fill");
    let refs: Vec<&K::Shape> = edges.iter().collect();
    let free = must(K::fill_boundary(&refs, &[]), "free fill");
    for ((edge, face), _) in edges.iter().zip(&neighbours).zip(0..) {
        let good = measure::<K>(&tangent, face, edge);
        let loose = measure::<K>(&free, face, edge);
        assert!(good.gap < 1e-3, "gap {}", good.gap);
        assert!(
            good.angle.as_degrees() < TANGENT_DEGREES,
            "{:?}",
            good.angle
        );
        assert!(
            good.angle.as_degrees() < 0.5 * loose.angle.as_degrees(),
            "tangent {:?}, free {:?}",
            good.angle,
            loose.angle
        );
    }
}

/// Straight sides leave a flat face at 45 degrees, so no patch can both follow them and stay
/// tangent to it. A kernel may not hand one back anyway.
///
/// # Panics
/// If such a patch is returned.
pub fn a_tangent_fill_that_cannot_be_tangent_fails<K: Kernel>() {
    let ([floor, wall], [floor_edge, wall_edge]) = floor_and_wall::<K>();
    let start = segment::<K>(p(0., 0., 0.), p(0., 4., 4.));
    let end = segment::<K>(p(9., 0., 0.), p(9., 4., 4.));
    let sides = [
        FillEdge {
            edge: &floor_edge,
            tangent_to: Some(&floor),
        },
        FillEdge {
            edge: &wall_edge,
            tangent_to: Some(&wall),
        },
        FillEdge {
            edge: &start,
            tangent_to: None,
        },
        FillEdge {
            edge: &end,
            tangent_to: None,
        },
    ];
    assert!(K::fill_boundary_tangent(&sides, &[]).is_err());
}

fn only(u_min: bool, u_max: bool, v_min: bool, v_max: bool) -> FaceSides {
    FaceSides {
        u_min,
        u_max,
        v_min,
        v_max,
    }
}

/// # Panics
/// If the area isn't what growing those sides gives.
pub fn extending_grows_only_the_chosen_sides<K: Kernel>() {
    let plate = must(
        K::make_polygon_face(&[p(0., 0., 0.), p(2., 0., 0.), p(2., 3., 0.), p(0., 3., 0.)]),
        "plate",
    );
    for side in [
        only(true, false, false, false),
        only(false, true, false, false),
        only(false, false, true, false),
        only(false, false, false, true),
    ] {
        let bigger = must(K::extend_face(&plate, 1.0, side), "extend_face");
        assert_valid::<K>(&bigger, "extended");
        let area = must(K::area(&bigger), "area");
        // A polygon's parameter axes are the kernel's choice, so either side can be along U.
        assert!(
            (area - 9.0).abs() < 1e-6 || (area - 8.0).abs() < 1e-6,
            "one side grown by 1: {area}"
        );
    }
    let all = must(
        K::extend_face(&plate, 1.0, only(true, true, true, true)),
        "extend_face",
    );
    near(must(K::area(&all), "area"), 4.0 * 5.0, 1e-9, "all sides");
}

/// # Panics
/// If the grown face leaves the cylinder or misses the area of a longer patch.
pub fn extending_a_cylinder_follows_it<K: Kernel>() {
    let radius = 5.0;
    let patch = cylinder_patch::<K>(radius, (0.0, FRAC_PI_2), (0.0, 4.0));
    let bigger = must(
        K::extend_face(&patch, 1.0, only(false, false, false, true)),
        "extend_face",
    );
    assert_valid::<K>(&bigger, "extended");
    near(
        must(K::area(&bigger), "area"),
        FRAC_PI_2 * radius * 5.0,
        1e-6,
        "taller patch",
    );
    let around = must(
        K::extend_face(&patch, 1.0, only(false, true, false, false)),
        "extend_face",
    );
    near(
        must(K::area(&around), "area"),
        (FRAC_PI_2 + 1.0 / radius) * radius * 4.0,
        // The length is "about": turning it into an angle may use the chord, not the arc.
        1e-3,
        "wider patch",
    );
    assert!(matches!(
        must(K::face_geometry(&around), "face_geometry").surface,
        Surface::Cylinder { .. }
    ));
}

/// # Panics
/// If a hole survives, which the contract says it doesn't.
pub fn extending_drops_holes<K: Kernel>() {
    let outer = closed_wire::<K>(&[p(0., 0., 0.), p(4., 0., 0.), p(4., 4., 0.), p(0., 4., 0.)]);
    let inner = closed_wire::<K>(&[p(1., 1., 0.), p(1., 3., 0.), p(3., 3., 0.), p(3., 1., 0.)]);
    let plate = must(K::make_planar_face(&outer, &[&inner]), "plate");
    near(must(K::area(&plate), "area"), 12.0, 1e-9, "plate with hole");
    let bigger = must(
        K::extend_face(&plate, 1.0, only(true, false, false, false)),
        "extend_face",
    );
    near(must(K::area(&bigger), "area"), 20.0, 1e-9, "no hole");
    assert_eq!(parts::<K>(&bigger, ShapeKind::Wire).len(), 1);
}

/// # Panics
/// If nonsense is grown.
pub fn extending_rejects_bad_input<K: Kernel>() {
    let plate = must(
        K::make_polygon_face(&[p(0., 0., 0.), p(2., 0., 0.), p(2., 3., 0.)]),
        "plate",
    );
    let any = only(true, false, false, false);
    assert!(K::extend_face(&plate, 0.0, any).is_err());
    assert!(K::extend_face(&plate, -1.0, any).is_err());
    assert!(K::extend_face(&plate, f64::INFINITY, any).is_err());
    assert!(K::extend_face(&plate, 1.0, FaceSides::default()).is_err());
    let edge = parts::<K>(&plate, ShapeKind::Edge).remove(0);
    assert!(K::extend_face(&edge, 1.0, any).is_err());
}
