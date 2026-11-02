//! Checks for Construction and Geometry.
//!
//! Every expected value comes from a closed form or a direct numeric integral of the
//! definition. Curvature is compared by magnitude where the sign depends on a convention the
//! contract doesn't state, and through a sign-independent property otherwise.

use std::f64::consts::{FRAC_PI_2, PI, TAU};

use occt_kernel::{
    Axis, BSplineCurve, BSplineSurface, Construction, Curve, CurveKind, Dir3, Frame, Geometry,
    Kernel, ParamRange, Point3, ShapeKind, Surface, SurfaceKind, Vec3,
};

use crate::support::{
    MEASURE, POSITION, area_is, assert_valid, ball_volume, must, near, near_point, parts,
    sphere_area, volume_is,
};

/// What these checks need from a kernel.
pub trait Modelling: Kernel + Construction + Geometry {}
impl<K: Kernel + Construction + Geometry> Modelling for K {}

/// How closely a smooth approximation is expected to follow what it approximates, relative
/// to model size: filling and fitting tolerances are conventionally around 1e-3 to 1e-4.
const APPROXIMATE: f64 = 2e-3;

/// Every check as a `(name, function)` pair, in a stable order.
#[must_use]
pub fn checks<K: Modelling>() -> Vec<(&'static str, fn())> {
    macro_rules! list {
        ($($name:ident),* $(,)?) => {
            vec![$((stringify!($name), $name::<K> as fn())),*]
        };
    }
    list![
        vertex_sits_at_its_point,
        line_edges_measure_and_read_back,
        unbounded_or_undescribed_curves_are_rejected,
        circles_and_arcs_measure_and_evaluate,
        range_longer_than_a_turn_is_rejected,
        ellipse_measures_and_curves,
        rational_quarter_circle_is_exact,
        spline_data_round_trips,
        periodic_spline_round_trips,
        spline_outside_its_knots_is_rejected,
        spline_edge_can_use_part_of_the_curve,
        wire_accepts_edges_in_any_order_and_direction,
        wire_rejects_gaps_branches_and_repeats,
        wire_edges_walk_the_chain,
        planar_face_subtracts_holes_either_way_round,
        planar_face_is_a_plane_with_the_wire_counterclockwise,
        planar_face_rejects_bad_wires,
        faces_on_analytic_surfaces_by_bounds,
        whole_bounded_surfaces_make_faces,
        unbounded_surfaces_need_bounds,
        bounds_outside_or_too_wide_are_rejected,
        spline_surface_round_trips_and_measures,
        rational_spline_surface_keeps_its_weights,
        face_on_a_surface_bounded_by_a_wire,
        face_on_a_surface_rejects_a_wire_off_it,
        filling_a_flat_loop_is_flat,
        filling_passes_through_points,
        filling_rejects_open_loops,
        sewing_joins_along_shared_edges,
        sewing_respects_its_tolerance,
        sewing_rejects_what_has_no_faces,
        solid_from_a_shell_is_positive_whatever_the_winding,
        solid_rejects_an_open_shell,
        compound_holds_what_it_is_given,
        torus_measures,
        torus_rejects_bad_radii,
        interpolation_hits_every_point,
        closed_interpolation_is_smooth_where_it_closes,
        interpolation_rejects_coincident_neighbours,
        fit_stays_within_tolerance,
        helix_follows_the_formula,
        helix_handedness,
        helix_along_any_axis,
        offset_of_a_circle_is_a_circle,
        offset_of_a_polygon_grows_and_shrinks,
        offset_ignores_winding,
        offset_past_collapse_fails,
        offset_of_an_open_wire_is_parallel,
        offset_rejects_non_planar_wires,
        box_faces_read_as_outward_planes,
        cylinder_faces_read_back,
        cone_faces_read_back,
        sphere_reads_back_with_singular_poles,
        torus_reads_back,
        projection_onto_faces,
        projection_onto_edges_stops_at_the_ends,
        closedness_of_edges_wires_and_shells,
        tolerance_is_small_and_positive,
        wrong_kinds_are_errors_not_crashes,
    ]
}

fn p(x: f64, y: f64, z: f64) -> Point3 {
    Point3::new(x, y, z)
}

fn segment<K: Modelling>(a: Point3, b: Point3) -> K::Shape {
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

fn circle<K: Modelling>(center: Point3, radius: f64) -> K::Shape {
    must(
        K::make_edge(
            &Curve::Circle {
                frame: Frame::from_normal(center, Dir3::Z),
                radius,
            },
            None,
        ),
        "make_edge",
    )
}

fn wire_of<K: Modelling>(edges: &[K::Shape]) -> K::Shape {
    let refs: Vec<&K::Shape> = edges.iter().collect();
    must(K::make_wire(&refs), "make_wire")
}

/// A closed polygon wire through the points, in order.
fn polygon<K: Modelling>(points: &[Point3]) -> K::Shape {
    let edges: Vec<K::Shape> = (0..points.len())
        .map(|i| segment::<K>(points[i], points[(i + 1) % points.len()]))
        .collect();
    wire_of::<K>(&edges)
}

fn square(center: Point3, side: f64) -> Vec<Point3> {
    let h = side / 2.0;
    vec![
        center + Vec3::new(-h, -h, 0.0),
        center + Vec3::new(h, -h, 0.0),
        center + Vec3::new(h, h, 0.0),
        center + Vec3::new(-h, h, 0.0),
    ]
}

fn reversed<T: Clone>(items: &[T]) -> Vec<T> {
    items.iter().rev().cloned().collect()
}

fn planar_area<K: Modelling>(wire: &K::Shape) -> f64 {
    let face = must(K::make_planar_face(wire, &[]), "make_planar_face");
    must(K::area(&face), "area")
}

/// Sums the planar areas of the wires of a shape.
fn wires_area<K: Modelling>(shape: &K::Shape) -> f64 {
    parts::<K>(shape, ShapeKind::Wire)
        .iter()
        .map(planar_area::<K>)
        .sum()
}

fn spline_of(curve: Curve) -> BSplineCurve {
    match curve {
        Curve::BSpline(spline) => spline,
        other => panic!("expected a spline, got {other:?}"),
    }
}

fn line_range<K: Modelling>(edge: &K::Shape) -> ParamRange {
    must(K::edge_geometry(edge), "edge_geometry").range
}

/// The six faces of a cube of this side at the origin. `flip(i)` reverses one's winding.
fn cube_faces<K: Modelling>(side: f64, flip: impl Fn(usize) -> bool) -> Vec<K::Shape> {
    let s = side;
    let loops = [
        [p(0., 0., 0.), p(0., s, 0.), p(s, s, 0.), p(s, 0., 0.)],
        [p(0., 0., s), p(s, 0., s), p(s, s, s), p(0., s, s)],
        [p(0., 0., 0.), p(s, 0., 0.), p(s, 0., s), p(0., 0., s)],
        [p(0., s, 0.), p(0., s, s), p(s, s, s), p(s, s, 0.)],
        [p(0., 0., 0.), p(0., 0., s), p(0., s, s), p(0., s, 0.)],
        [p(s, 0., 0.), p(s, s, 0.), p(s, s, s), p(s, 0., s)],
    ];
    loops
        .iter()
        .enumerate()
        .map(|(i, corners)| {
            let corners = if flip(i) {
                reversed(corners)
            } else {
                corners.to_vec()
            };
            let wire = polygon::<K>(&corners);
            must(K::make_planar_face(&wire, &[]), "make_planar_face")
        })
        .collect()
}

fn sew<K: Modelling>(shapes: &[K::Shape], tolerance: f64) -> K::Shape {
    let refs: Vec<&K::Shape> = shapes.iter().collect();
    must(K::sew(&refs, tolerance), "sew")
}

/// A vertex reports the point it was built at.
///
/// # Panics
/// If it doesn't.
pub fn vertex_sits_at_its_point<K: Modelling>() {
    for at in [p(0., 0., 0.), p(1., -2., 3.5), p(1e4, 0., -1e4)] {
        let v = must(K::make_vertex(at), "make_vertex");
        assert_eq!(must(K::kind(&v), "kind"), ShapeKind::Vertex);
        near_point(must(K::vertex_point(&v), "vertex_point"), at, "vertex");
    }
}

/// A line reads back as the same line, and evaluates by distance with no curvature.
///
/// # Panics
/// If a value differs.
pub fn line_edges_measure_and_read_back<K: Modelling>() {
    let cases = [
        (p(0., 0., 0.), p(1., 0., 0.)),
        (p(1., 2., 3.), p(4., 6., 3.)),
        (p(-1., -1., -1.), p(1., 1., 1.)),
    ];
    for (a, b) in cases {
        let length = a.distance(b);
        let edge = segment::<K>(a, b);
        assert_eq!(must(K::kind(&edge), "kind"), ShapeKind::Edge);
        assert_eq!(must(K::curve_kind(&edge), "curve_kind"), CurveKind::Line);
        near(must(K::length(&edge), "length"), length, MEASURE, "length");

        let geometry = must(K::edge_geometry(&edge), "edge_geometry");
        let Curve::Line { origin, direction } = geometry.curve else {
            panic!("expected a line, got {:?}", geometry.curve);
        };
        near_point(origin, a, "origin");
        assert!((direction.vector() - (b - a) / length).length() < POSITION);
        assert!((geometry.range.end - length).abs() < POSITION);

        let middle = must(K::edge_point(&edge, length / 2.0), "edge_point");
        near_point(middle.point, a + (b - a) / 2.0, "middle");
        assert!((middle.derivative.length() - 1.0).abs() < POSITION);
        assert!(middle.curvature.abs() < POSITION);
    }
}

#[track_caller]
fn assert_same_frame(got: Frame, want: Frame) {
    near_point(got.origin, want.origin, "frame origin");
    assert!(
        (got.z.vector() - want.z.vector()).length() < POSITION,
        "frame z"
    );
    assert!(
        (got.x.vector() - want.x.vector()).length() < POSITION,
        "frame x"
    );
}

#[track_caller]
fn assert_same_circle(got: &Curve, frame: Frame, radius: f64) {
    let Curve::Circle {
        frame: f,
        radius: r,
    } = got
    else {
        panic!("expected a circle, got {got:?}");
    };
    assert_same_frame(*f, frame);
    near(*r, radius, MEASURE, "circle radius");
}

/// Lines have no natural ends, and `Other` describes nothing.
///
/// # Panics
/// If either is accepted.
pub fn unbounded_or_undescribed_curves_are_rejected<K: Modelling>() {
    let line = Curve::Line {
        origin: Point3::ORIGIN,
        direction: Dir3::X,
    };
    assert!(K::make_edge(&line, None).is_err(), "a line with no range");
    let whole = Some(ParamRange {
        start: 0.0,
        end: 1.0,
    });
    assert!(K::make_edge(&Curve::Other, None).is_err());
    assert!(K::make_edge(&Curve::Other, whole).is_err());
}

/// Circles measure `2 pi r`, a range measures its share, and evaluation gives the exact
/// point, a speed of `r`, and curvature `1 / r`.
///
/// # Panics
/// If a value differs.
pub fn circles_and_arcs_measure_and_evaluate<K: Modelling>() {
    for radius in [0.5, 1.0, 7.0, 100.0] {
        let frame = Frame::from_normal(
            p(1., 2., 3.),
            must(Dir3::new([1., 1., 0.]).map_err(Into::into), "dir"),
        );
        let curve = Curve::Circle { frame, radius };
        let full = must(K::make_edge(&curve, None), "make_edge");
        near(
            must(K::length(&full), "length"),
            TAU * radius,
            MEASURE,
            "full",
        );
        assert!(
            must(K::is_closed(&full), "is_closed"),
            "a full circle closes"
        );

        for (start, end) in [(0.0, FRAC_PI_2), (1.0, 3.0), (-1.0, 1.0)] {
            let arc = must(
                K::make_edge(&curve, Some(ParamRange { start, end })),
                "make_edge",
            );
            near(
                must(K::length(&arc), "length"),
                radius * (end - start),
                MEASURE,
                "arc",
            );
            assert!(!must(K::is_closed(&arc), "is_closed"), "an arc is open");
            let t = (start + end) / 2.0;
            let at = must(K::edge_point(&arc, t), "edge_point");
            let (x, y) = (frame.x.vector(), frame.y().vector());
            let want = frame.origin + x * (radius * t.cos()) + y * (radius * t.sin());
            near_point(at.point, want, "arc point");
            near(at.derivative.length(), radius, MEASURE, "speed");
            near(at.curvature, 1.0 / radius, MEASURE, "curvature");

            let geometry = must(K::edge_geometry(&arc), "edge_geometry");
            assert_same_circle(&geometry.curve, frame, radius);
            // A range is good up to whole turns of a periodic curve.
            let turns = (geometry.range.start - start) / TAU;
            assert!((turns - turns.round()).abs() < POSITION, "range start");
            near(
                geometry.range.end - geometry.range.start,
                end - start,
                MEASURE,
                "span",
            );
            assert!(K::edge_point(&arc, end + 0.5).is_err(), "past the end");
        }
    }
}

/// A range past one turn can't describe an edge, and must not be taken modulo a turn.
///
/// # Panics
/// If it is accepted.
pub fn range_longer_than_a_turn_is_rejected<K: Modelling>() {
    let curve = Curve::Circle {
        frame: Frame::WORLD,
        radius: 1.0,
    };
    let range = ParamRange {
        start: 0.0,
        end: 3.0 * PI,
    };
    assert!(K::make_edge(&curve, Some(range)).is_err());
    let backwards = ParamRange {
        start: 2.0,
        end: 1.0,
    };
    assert!(K::make_edge(&curve, Some(backwards)).is_err());
}

/// An ellipse's perimeter against a numeric integral, and its curvature `a / b^2` at the end
/// of the major axis and `b / a^2` at the end of the minor one.
///
/// # Panics
/// If a value differs.
pub fn ellipse_measures_and_curves<K: Modelling>() {
    let (a, b) = (3.0f64, 1.0f64);
    let curve = Curve::Ellipse {
        frame: Frame::WORLD,
        major_radius: a,
        minor_radius: b,
    };
    let edge = must(K::make_edge(&curve, None), "make_edge");
    // The trapezoid rule on a smooth periodic integrand converges exponentially.
    let n = 2000;
    let perimeter: f64 = (0..n)
        .map(|i| f64::from(i) * TAU / f64::from(n))
        .map(|t| (a * t.sin()).hypot(b * t.cos()) * TAU / f64::from(n))
        .sum();
    // The arc length integrated from the kernel's own derivative must be exact to the
    // sampling. A kernel's general length measure gets more slack: it is a numeric integral
    // too, of a curve with no closed form, and one percent is enough for what it's for.
    let integrated: f64 = (0..n)
        .map(|i| f64::from(i) * TAU / f64::from(n))
        .map(|t| {
            let at = must(K::edge_point(&edge, t), "edge_point");
            at.derivative.length() * TAU / f64::from(n)
        })
        .sum();
    near(integrated, perimeter, MEASURE, "integrated perimeter");
    near(
        must(K::length(&edge), "length"),
        perimeter,
        1e-2,
        "perimeter",
    );
    let at_major = must(K::edge_point(&edge, 0.0), "edge_point");
    near_point(at_major.point, p(a, 0., 0.), "major end");
    near(at_major.curvature, a / (b * b), MEASURE, "major curvature");
    let at_minor = must(K::edge_point(&edge, FRAC_PI_2), "edge_point");
    near_point(at_minor.point, p(0., b, 0.), "minor end");
    near(at_minor.curvature, b / (a * a), MEASURE, "minor curvature");
    assert_eq!(must(K::edge_geometry(&edge), "edge_geometry").curve, curve);
}

/// The standard rational quadratic for a quarter circle: exact only if weights take part.
///
/// # Panics
/// If the curve leaves the circle or the weights are lost.
pub fn rational_quarter_circle_is_exact<K: Modelling>() {
    let r = 2.0;
    let spline = BSplineCurve {
        degree: 2,
        knots: vec![0.0, 1.0],
        multiplicities: vec![3, 3],
        poles: vec![p(r, 0., 0.), p(r, r, 0.), p(0., r, 0.)],
        weights: Some(vec![1.0, 0.5f64.sqrt(), 1.0]),
        periodic: false,
    };
    let edge = must(
        K::make_edge(&Curve::BSpline(spline.clone()), None),
        "make_edge",
    );
    near(
        must(K::length(&edge), "length"),
        FRAC_PI_2 * r,
        MEASURE,
        "length",
    );
    for i in 0..=10 {
        let at = must(K::edge_point(&edge, f64::from(i) / 10.0), "edge_point");
        assert!(
            ((at.point - Point3::ORIGIN).length() - r).abs() < POSITION,
            "{:?} left the circle",
            at.point
        );
        near(at.curvature, 1.0 / r, MEASURE, "curvature");
    }
    let back = spline_of(must(K::edge_geometry(&edge), "edge_geometry").curve);
    assert_eq!(back, spline, "the spline reads back unchanged");
}

/// A spline with interior knots, and one without weights, read back as they went in.
///
/// # Panics
/// If the data changes.
pub fn spline_data_round_trips<K: Modelling>() {
    let spline = BSplineCurve {
        degree: 3,
        knots: vec![0.0, 1.0, 2.5, 4.0],
        multiplicities: vec![4, 1, 2, 4],
        poles: vec![
            p(0., 0., 0.),
            p(1., 2., 0.),
            p(2., 3., 1.),
            p(3., 3., 2.),
            p(4., 2., 2.),
            p(5., 1., 1.),
            p(6., 0., 0.),
        ],
        weights: None,
        periodic: false,
    };
    let edge = must(
        K::make_edge(&Curve::BSpline(spline.clone()), None),
        "make_edge",
    );
    assert_eq!(must(K::curve_kind(&edge), "curve_kind"), CurveKind::Spline);
    let geometry = must(K::edge_geometry(&edge), "edge_geometry");
    assert_eq!(spline_of(geometry.curve), spline);
    assert_eq!((geometry.range.start, geometry.range.end), (0.0, 4.0));
    near_point(
        must(K::edge_point(&edge, 0.0), "edge_point").point,
        p(0., 0., 0.),
        "clamped start",
    );
    near_point(
        must(K::edge_point(&edge, 4.0), "edge_point").point,
        p(6., 0., 0.),
        "clamped end",
    );
}

/// A periodic curve is a closed edge and stays periodic.
///
/// # Panics
/// If it isn't closed or the data changes.
pub fn periodic_spline_round_trips<K: Modelling>() {
    let spline = BSplineCurve {
        degree: 2,
        knots: vec![0.0, 1.0, 2.0, 3.0, 4.0],
        multiplicities: vec![1, 1, 1, 1, 1],
        poles: vec![p(1., 0., 0.), p(0., 1., 0.), p(-1., 0., 0.), p(0., -1., 0.)],
        weights: None,
        periodic: true,
    };
    let edge = must(
        K::make_edge(&Curve::BSpline(spline.clone()), None),
        "make_edge",
    );
    assert!(must(K::is_closed(&edge), "is_closed"));
    assert_eq!(
        spline_of(must(K::edge_geometry(&edge), "edge_geometry").curve),
        spline
    );
}

/// The knots bound a non-periodic curve.
///
/// # Panics
/// If a range outside them is accepted.
pub fn spline_outside_its_knots_is_rejected<K: Modelling>() {
    let spline = Curve::BSpline(BSplineCurve {
        degree: 1,
        knots: vec![0.0, 1.0],
        multiplicities: vec![2, 2],
        poles: vec![p(0., 0., 0.), p(1., 0., 0.)],
        weights: None,
        periodic: false,
    });
    let range = ParamRange {
        start: 0.5,
        end: 2.0,
    };
    assert!(K::make_edge(&spline, Some(range)).is_err());
}

/// A part of a straight spline measures its share, and reports the whole curve as its basis
/// with the part as its range.
///
/// # Panics
/// If a value differs.
pub fn spline_edge_can_use_part_of_the_curve<K: Modelling>() {
    let spline = BSplineCurve {
        degree: 1,
        knots: vec![0.0, 1.0],
        multiplicities: vec![2, 2],
        poles: vec![p(0., 0., 0.), p(4., 0., 0.)],
        weights: None,
        periodic: false,
    };
    let range = ParamRange {
        start: 0.25,
        end: 0.75,
    };
    let edge = must(
        K::make_edge(&Curve::BSpline(spline.clone()), Some(range)),
        "make_edge",
    );
    near(must(K::length(&edge), "length"), 2.0, MEASURE, "length");
    let geometry = must(K::edge_geometry(&edge), "edge_geometry");
    assert_eq!(spline_of(geometry.curve), spline);
    assert_eq!(geometry.range, range);
}

/// A closed square assembled from every order and with edges drawn either way round.
///
/// # Panics
/// If the wire fails or isn't closed.
pub fn wire_accepts_edges_in_any_order_and_direction<K: Modelling>() {
    let c = square(Point3::ORIGIN, 2.0);
    let forward = |i: usize| (c[i], c[(i + 1) % 4]);
    let orders: [[usize; 4]; 5] = [
        [0, 1, 2, 3],
        [3, 2, 1, 0],
        [0, 2, 1, 3],
        [2, 0, 3, 1],
        [1, 3, 0, 2],
    ];
    for order in orders {
        for flips in 0..16 {
            let edges: Vec<K::Shape> = order
                .iter()
                .map(|&i| {
                    let (a, b) = forward(i);
                    if flips & (1 << i) != 0 {
                        segment::<K>(b, a)
                    } else {
                        segment::<K>(a, b)
                    }
                })
                .collect();
            let wire = wire_of::<K>(&edges);
            assert_eq!(must(K::kind(&wire), "kind"), ShapeKind::Wire);
            assert!(must(K::is_closed(&wire), "is_closed"), "{order:?} {flips}");
            near(
                must(K::length(&wire), "length"),
                8.0,
                MEASURE,
                "wire length",
            );
            assert_eq!(parts::<K>(&wire, ShapeKind::Edge).len(), 4);
            near(planar_area::<K>(&wire), 4.0, MEASURE, "wire area");
        }
    }
}

/// Things that aren't one chain.
///
/// # Panics
/// If a wire is made from them.
pub fn wire_rejects_gaps_branches_and_repeats<K: Modelling>() {
    let a = segment::<K>(p(0., 0., 0.), p(1., 0., 0.));
    let far = segment::<K>(p(5., 5., 0.), p(6., 5., 0.));
    assert!(K::make_wire(&[&a, &far]).is_err(), "a gap");
    let b = segment::<K>(p(1., 0., 0.), p(2., 0., 0.));
    let c = segment::<K>(p(1., 0., 0.), p(1., 1., 0.));
    assert!(K::make_wire(&[&a, &b, &c]).is_err(), "a branch");
    assert!(K::make_wire(&[&a, &a]).is_err(), "a repeat");
    assert!(K::make_wire(&[]).is_err(), "nothing");
}

/// The edges of a wire come in order: each ends where the next starts.
///
/// # Panics
/// If they're shuffled.
pub fn wire_edges_walk_the_chain<K: Modelling>() {
    let chain = [
        p(0., 0., 0.),
        p(1., 0., 0.),
        p(1., 1., 0.),
        p(2., 1., 0.),
        p(2., 3., 0.),
    ];
    let segments: Vec<K::Shape> = chain.windows(2).map(|w| segment::<K>(w[0], w[1])).collect();
    let shuffled = [
        segments[2].clone(),
        segments[0].clone(),
        segments[3].clone(),
        segments[1].clone(),
    ];
    let wire = wire_of::<K>(&shuffled);
    let edges = parts::<K>(&wire, ShapeKind::Edge);
    let ends = |edge: &K::Shape| {
        let range = line_range::<K>(edge);
        let a = must(K::edge_point(edge, range.start), "edge_point").point;
        let b = must(K::edge_point(edge, range.end), "edge_point").point;
        [a, b]
    };
    for pair in edges.windows(2) {
        let (first, second) = (ends(&pair[0]), ends(&pair[1]));
        let touches = first
            .iter()
            .any(|a| second.iter().any(|b| a.distance(*b) < POSITION));
        assert!(touches, "neighbours {first:?} and {second:?} don't touch");
    }
}

/// The outer area less the holes', however the wires run.
///
/// # Panics
/// If the area differs or the face is invalid.
pub fn planar_face_subtracts_holes_either_way_round<K: Modelling>() {
    let outer = square(Point3::ORIGIN, 10.0);
    let a = square(p(-2., 0., 0.), 2.0);
    let b = square(p(2., 0., 0.), 1.0);
    for flip_outer in [false, true] {
        for flip_a in [false, true] {
            for flip_b in [false, true] {
                let pick = |points: &Vec<Point3>, flip: bool| {
                    polygon::<K>(&if flip {
                        reversed(points)
                    } else {
                        points.clone()
                    })
                };
                let o = pick(&outer, flip_outer);
                let h1 = pick(&a, flip_a);
                let h2 = pick(&b, flip_b);
                let face = must(K::make_planar_face(&o, &[&h1, &h2]), "make_planar_face");
                assert_valid::<K>(&face, "face with holes");
                area_is::<K>(&face, 100.0 - 4.0 - 1.0, "face with holes");
                assert_eq!(parts::<K>(&face, ShapeKind::Wire).len(), 3);
            }
        }
    }
    let round = must(
        K::make_wire(&[&circle::<K>(p(0., 0., 0.), 3.0)]),
        "make_wire",
    );
    let face = must(
        K::make_planar_face(&polygon::<K>(&outer), &[&round]),
        "make_planar_face",
    );
    area_is::<K>(&face, 100.0 - PI * 9.0, "face with a round hole");
}

/// A planar face's outward normal sees its outer wire run counterclockwise.
///
/// # Panics
/// If the normal is the wrong way or the surface isn't a plane.
pub fn planar_face_is_a_plane_with_the_wire_counterclockwise<K: Modelling>() {
    let corners = square(p(0., 0., 3.), 2.0);
    let up = polygon::<K>(&corners);
    let face = must(K::make_planar_face(&up, &[]), "make_planar_face");
    assert_eq!(
        must(K::surface_kind(&face), "surface_kind"),
        SurfaceKind::Plane
    );
    let geometry = must(K::face_geometry(&face), "face_geometry");
    let u = (geometry.u.start + geometry.u.end) / 2.0;
    let v = (geometry.v.start + geometry.v.end) / 2.0;
    let at = must(K::face_point(&face, u, v), "face_point");
    let normal = at.normal.expect("a plane has a normal").vector();
    assert!(normal.z.abs() > 1.0 - POSITION, "normal {normal:?}");
    near(at.point.z, 3.0, MEASURE, "plane height");
}

/// Wires that bound no planar region, and holes that aren't in one.
///
/// # Panics
/// If a face is made from them.
pub fn planar_face_rejects_bad_wires<K: Modelling>() {
    let open = wire_of::<K>(&[
        segment::<K>(p(0., 0., 0.), p(1., 0., 0.)),
        segment::<K>(p(1., 0., 0.), p(1., 1., 0.)),
    ]);
    assert!(K::make_planar_face(&open, &[]).is_err(), "open");
    let skew = polygon::<K>(&[p(0., 0., 0.), p(1., 0., 0.), p(1., 1., 1.), p(0., 1., 0.)]);
    assert!(K::make_planar_face(&skew, &[]).is_err(), "not planar");
    let bow = polygon::<K>(&[p(0., 0., 0.), p(1., 1., 0.), p(1., 0., 0.), p(0., 1., 0.)]);
    assert!(K::make_planar_face(&bow, &[]).is_err(), "self-crossing");
    let outer = polygon::<K>(&square(Point3::ORIGIN, 4.0));
    let outside = polygon::<K>(&square(p(10., 0., 0.), 1.0));
    assert!(
        K::make_planar_face(&outer, &[&outside]).is_err(),
        "hole outside"
    );
    let crossing = polygon::<K>(&square(p(2., 0., 0.), 1.0));
    assert!(
        K::make_planar_face(&outer, &[&crossing]).is_err(),
        "hole crossing"
    );
    let lifted = polygon::<K>(&square(p(0., 0., 1.), 1.0));
    assert!(
        K::make_planar_face(&outer, &[&lifted]).is_err(),
        "hole off plane"
    );
    let inner_a = polygon::<K>(&square(p(0.25, 0., 0.), 1.0));
    let inner_b = polygon::<K>(&square(p(-0.25, 0., 0.), 1.0));
    assert!(
        K::make_planar_face(&outer, &[&inner_a, &inner_b]).is_err(),
        "overlapping holes"
    );
}

fn range(start: f64, end: f64) -> ParamRange {
    ParamRange { start, end }
}

/// Patches of analytic surfaces have closed-form areas.
///
/// # Panics
/// If an area differs.
pub fn faces_on_analytic_surfaces_by_bounds<K: Modelling>() {
    let frame = Frame::WORLD;
    let cases: Vec<(Surface, ParamRange, ParamRange, f64)> = vec![
        (Surface::Plane { frame }, range(0., 2.), range(-1., 2.), 6.0),
        (
            Surface::Cylinder { frame, radius: 2.0 },
            range(0., FRAC_PI_2),
            range(0., 3.),
            2.0 * FRAC_PI_2 * 3.0,
        ),
        (
            Surface::Sphere { frame, radius: 2.0 },
            range(0., PI),
            range(0., FRAC_PI_2),
            // A quarter of the sphere: half of its upper half.
            sphere_area(2.0) / 4.0,
        ),
        (
            Surface::Torus {
                frame,
                major_radius: 3.0,
                minor_radius: 1.0,
            },
            range(0., PI),
            range(0., TAU),
            2.0 * PI * PI * 3.0,
        ),
    ];
    for (surface, u, v, area) in cases {
        let face = must(
            K::make_surface_face(&surface, Some((u, v)), None, &[]),
            "make_surface_face",
        );
        assert_valid::<K>(&face, "patch");
        area_is::<K>(&face, area, format_args!("patch of {surface:?}"));
        let geometry = must(K::face_geometry(&face), "face_geometry");
        assert_eq!(geometry.surface, surface);
        assert!((geometry.u.start - u.start).abs() < POSITION);
        assert!((geometry.u.end - u.end).abs() < POSITION);
        assert!((geometry.v.start - v.start).abs() < POSITION);
        assert!((geometry.v.end - v.end).abs() < POSITION);
    }
    // A cone frustum: a slant of 2 with half angle 30 degrees widens from 1 by sin(30) * 2.
    let half_angle = PI / 6.0;
    let cone = Surface::Cone {
        frame,
        radius: 1.0,
        half_angle,
    };
    let face = must(
        K::make_surface_face(&cone, Some((range(0., TAU), range(0., 2.))), None, &[]),
        "make_surface_face",
    );
    let (r1, r2) = (1.0, 1.0 + 2.0 * half_angle.sin());
    area_is::<K>(&face, PI * (r1 + r2) * 2.0, "frustum");
}

/// A sphere or torus is bounded in itself, so it needs no bounds to make a whole face.
///
/// # Panics
/// If an area differs.
pub fn whole_bounded_surfaces_make_faces<K: Modelling>() {
    let frame = Frame::WORLD;
    let sphere = Surface::Sphere { frame, radius: 3.0 };
    let face = must(
        K::make_surface_face(&sphere, None, None, &[]),
        "sphere face",
    );
    area_is::<K>(&face, sphere_area(3.0), "whole sphere");
    let torus = Surface::Torus {
        frame,
        major_radius: 4.0,
        minor_radius: 1.0,
    };
    let face = must(K::make_surface_face(&torus, None, None, &[]), "torus face");
    area_is::<K>(&face, 4.0 * PI * PI * 4.0, "whole torus");
}

/// Planes, cylinders and cones go on forever.
///
/// # Panics
/// If a face is made from one with no bounds.
pub fn unbounded_surfaces_need_bounds<K: Modelling>() {
    let frame = Frame::WORLD;
    assert!(K::make_surface_face(&Surface::Plane { frame }, None, None, &[]).is_err());
    let cylinder = Surface::Cylinder { frame, radius: 1.0 };
    assert!(K::make_surface_face(&cylinder, None, None, &[]).is_err());
    assert!(K::make_surface_face(&Surface::Other, None, None, &[]).is_err());
}

/// Bounds that leave the surface or wrap past a period.
///
/// # Panics
/// If they're accepted.
pub fn bounds_outside_or_too_wide_are_rejected<K: Modelling>() {
    let frame = Frame::WORLD;
    let sphere = Surface::Sphere { frame, radius: 1.0 };
    let past_pole = (range(0., 1.), range(0., 3.));
    assert!(K::make_surface_face(&sphere, Some(past_pole), None, &[]).is_err());
    let cylinder = Surface::Cylinder { frame, radius: 1.0 };
    let wrapped = (range(0., 3.0 * PI), range(0., 1.));
    assert!(K::make_surface_face(&cylinder, Some(wrapped), None, &[]).is_err());
    let backwards = (range(1., 0.), range(0., 1.));
    assert!(K::make_surface_face(&cylinder, Some(backwards), None, &[]).is_err());
}

fn flat_patch(side: f64) -> BSplineSurface {
    BSplineSurface {
        degree_u: 1,
        degree_v: 1,
        knots_u: vec![0.0, 1.0],
        multiplicities_u: vec![2, 2],
        knots_v: vec![0.0, 1.0],
        multiplicities_v: vec![2, 2],
        pole_count_u: 2,
        pole_count_v: 2,
        poles: vec![
            p(0., 0., 0.),
            p(0., side, 0.),
            p(side, 0., 0.),
            p(side, side, 0.),
        ],
        weights: None,
        periodic_u: false,
        periodic_v: false,
    }
}

/// A bilinear square is a spline whose poles lay out its corners, U varying slowest.
///
/// # Panics
/// If the data changes or a value differs.
pub fn spline_surface_round_trips_and_measures<K: Modelling>() {
    let surface = Surface::BSpline(flat_patch(3.0));
    let face = must(
        K::make_surface_face(&surface, None, None, &[]),
        "make_surface_face",
    );
    assert_eq!(
        must(K::surface_kind(&face), "surface_kind"),
        SurfaceKind::Spline
    );
    area_is::<K>(&face, 9.0, "patch");
    assert_eq!(
        must(K::face_geometry(&face), "face_geometry").surface,
        surface
    );
    let at = must(K::face_point(&face, 1.0, 0.0), "face_point");
    near_point(at.point, p(3., 0., 0.), "U runs along the slow index");
    let part = must(
        K::make_surface_face(&surface, Some((range(0., 0.5), range(0., 0.5))), None, &[]),
        "make_surface_face",
    );
    area_is::<K>(&part, 9.0 / 4.0, "part of a patch");
}

/// A rational surface: a quarter cylinder as a surface whose U runs a rational arc.
///
/// # Panics
/// If the weights are lost or points leave the cylinder.
pub fn rational_spline_surface_keeps_its_weights<K: Modelling>() {
    let (r, w, h) = (2.0, 0.5f64.sqrt(), 3.0);
    let surface = Surface::BSpline(BSplineSurface {
        degree_u: 2,
        degree_v: 1,
        knots_u: vec![0.0, 1.0],
        multiplicities_u: vec![3, 3],
        knots_v: vec![0.0, 1.0],
        multiplicities_v: vec![2, 2],
        pole_count_u: 3,
        pole_count_v: 2,
        poles: vec![
            p(r, 0., 0.),
            p(r, 0., h),
            p(r, r, 0.),
            p(r, r, h),
            p(0., r, 0.),
            p(0., r, h),
        ],
        weights: Some(vec![1.0, 1.0, w, w, 1.0, 1.0]),
        periodic_u: false,
        periodic_v: false,
    });
    let face = must(
        K::make_surface_face(&surface, None, None, &[]),
        "make_surface_face",
    );
    assert_eq!(
        must(K::face_geometry(&face), "face_geometry").surface,
        surface
    );
    area_is::<K>(&face, FRAC_PI_2 * r * h, "quarter cylinder");
    let at = must(K::face_point(&face, 0.5, 0.5), "face_point");
    let radial = at.point.x.hypot(at.point.y);
    assert!((radial - r).abs() < POSITION, "{radial} off the cylinder");
    // Principal curvatures of a cylinder: zero along its axis, 1 / r around it.
    let mut sizes = at.principal_curvatures.map(f64::abs);
    sizes.sort_by(f64::total_cmp);
    assert!(sizes[0] < POSITION);
    near(sizes[1], 1.0 / r, MEASURE, "curvature");
}

/// Builds the closed loop of a window on a cylinder from arcs and rulings.
fn window<K: Modelling>(radius: f64, from: f64, to: f64, low: f64, high: f64) -> K::Shape {
    let arc = |z: f64, start: f64, end: f64| {
        let frame = Frame::from_normal(p(0., 0., z), Dir3::Z);
        // from_normal's X for +Z is world X, which is the cylinder's seam direction too.
        must(
            K::make_edge(&Curve::Circle { frame, radius }, Some(range(start, end))),
            "arc",
        )
    };
    let corner = |t: f64, z: f64| p(radius * t.cos(), radius * t.sin(), z);
    let edges = [
        arc(low, from, to),
        segment::<K>(corner(to, low), corner(to, high)),
        arc(high, from, to),
        segment::<K>(corner(from, high), corner(from, low)),
    ];
    wire_of::<K>(&edges)
}

/// A window cut from a cylinder by wires has area `r * angle * height`, less its own hole.
///
/// # Panics
/// If an area differs.
pub fn face_on_a_surface_bounded_by_a_wire<K: Modelling>() {
    let r = 2.0;
    let surface = Surface::Cylinder {
        frame: Frame::WORLD,
        radius: r,
    };
    let outer = window::<K>(r, 0.5, 2.5, 0.0, 4.0);
    let face = must(
        K::make_surface_face(&surface, None, Some(&outer), &[]),
        "window",
    );
    assert_valid::<K>(&face, "window");
    area_is::<K>(&face, r * 2.0 * 4.0, "window");
    assert_eq!(
        must(K::surface_kind(&face), "surface_kind"),
        SurfaceKind::Cylinder
    );

    let hole = window::<K>(r, 1.0, 2.0, 1.0, 3.0);
    let holed = must(
        K::make_surface_face(&surface, None, Some(&outer), &[&hole]),
        "window with a hole",
    );
    assert_valid::<K>(&holed, "window with a hole");
    area_is::<K>(&holed, r * 2.0 * 4.0 - r * 1.0 * 2.0, "window with a hole");

    let plane = Surface::Plane {
        frame: Frame::WORLD,
    };
    let wire = polygon::<K>(&square(Point3::ORIGIN, 2.0));
    let flat = must(K::make_surface_face(&plane, None, Some(&wire), &[]), "flat");
    area_is::<K>(&flat, 4.0, "a square on a plane");
}

/// A wire that misses the surface can't bound a face on it.
///
/// # Panics
/// If it is accepted.
pub fn face_on_a_surface_rejects_a_wire_off_it<K: Modelling>() {
    let surface = Surface::Cylinder {
        frame: Frame::WORLD,
        radius: 2.0,
    };
    let off = polygon::<K>(&square(p(0., 0., 1.), 0.5));
    assert!(K::make_surface_face(&surface, None, Some(&off), &[]).is_err());
    let open = wire_of::<K>(&[segment::<K>(p(2., 0., 0.), p(2., 0., 1.))]);
    assert!(K::make_surface_face(&surface, None, Some(&open), &[]).is_err());
}

/// A flat loop's smooth patch is that flat region, within the fit.
///
/// # Panics
/// If the area differs or the face is invalid.
pub fn filling_a_flat_loop_is_flat<K: Modelling>() {
    let c = square(Point3::ORIGIN, 2.0);
    let edges: Vec<K::Shape> = (0..4).map(|i| segment::<K>(c[i], c[(i + 1) % 4])).collect();
    let shuffled = [&edges[2], &edges[0], &edges[3], &edges[1]];
    let face = must(K::fill_boundary(&shuffled, &[]), "fill_boundary");
    assert_valid::<K>(&face, "patch");
    near(
        must(K::area(&face), "area"),
        4.0,
        APPROXIMATE,
        "square patch",
    );
    let bbox = must(K::bounding_box(&face), "bounding_box");
    assert!(bbox.size().z < 0.01, "a flat loop stays flat: {bbox:?}");

    let ring = circle::<K>(Point3::ORIGIN, 3.0);
    let disc = must(K::fill_boundary(&[&ring], &[]), "fill_boundary");
    near(
        must(K::area(&disc), "area"),
        PI * 9.0,
        5.0 * APPROXIMATE,
        "disc",
    );
}

/// With a point to pass through, the patch does, and rises above the flat one's area.
///
/// # Panics
/// If the point is missed.
pub fn filling_passes_through_points<K: Modelling>() {
    let c = square(Point3::ORIGIN, 4.0);
    let edges: Vec<K::Shape> = (0..4).map(|i| segment::<K>(c[i], c[(i + 1) % 4])).collect();
    let refs: Vec<&K::Shape> = edges.iter().collect();
    for height in [0.5, 1.0, -1.0] {
        let target = p(0., 0., height);
        let face = must(K::fill_boundary(&refs, &[target]), "fill_boundary");
        assert_valid::<K>(&face, "dome");
        assert!(must(K::area(&face), "area") > 16.0, "a dome has more area");
        let (.., foot) = must(K::project_onto_face(&face, target), "project_onto_face");
        assert!(
            foot.distance(target) < 0.01,
            "missed by {}",
            foot.distance(target)
        );
    }
}

/// An open chain has nothing to fill, and neither does a gap.
///
/// # Panics
/// If a face is made.
pub fn filling_rejects_open_loops<K: Modelling>() {
    let a = segment::<K>(p(0., 0., 0.), p(1., 0., 0.));
    let b = segment::<K>(p(1., 0., 0.), p(1., 1., 0.));
    assert!(K::fill_boundary(&[&a, &b], &[]).is_err());
    let far = segment::<K>(p(5., 5., 0.), p(6., 5., 0.));
    assert!(K::fill_boundary(&[&a, &far], &[]).is_err());
    assert!(K::fill_boundary(&[], &[]).is_err());
}

/// Two squares sharing an edge become a sheet with seven edges, with the area of both.
///
/// # Panics
/// If the topology or the area differs.
pub fn sewing_joins_along_shared_edges<K: Modelling>() {
    let a = polygon::<K>(&square(p(0., 0., 0.), 1.0));
    let b = polygon::<K>(&square(p(1., 0., 0.), 1.0));
    let fa = must(K::make_planar_face(&a, &[]), "face");
    let fb = must(K::make_planar_face(&b, &[]), "face");
    let sheet = sew::<K>(&[fa, fb], 1e-6);
    assert_eq!(must(K::kind(&sheet), "kind"), ShapeKind::Shell);
    assert!(!must(K::is_closed(&sheet), "is_closed"));
    assert_eq!(parts::<K>(&sheet, ShapeKind::Edge).len(), 7);
    area_is::<K>(&sheet, 2.0, "sheet");

    let single = polygon::<K>(&square(p(0., 0., 0.), 1.0));
    let lone = must(K::make_planar_face(&single, &[]), "face");
    area_is::<K>(&sew::<K>(&[lone], 1e-6), 1.0, "a single face");
}

/// A gap closes if the tolerance is wider than it, and stays open if not.
///
/// # Panics
/// If the tolerance isn't honoured.
pub fn sewing_respects_its_tolerance<K: Modelling>() {
    let gap = 1e-3;
    let a = polygon::<K>(&square(p(0., 0., 0.), 1.0));
    let b = polygon::<K>(&square(p(1.0 + gap, 0., 0.), 1.0));
    let fa = must(K::make_planar_face(&a, &[]), "face");
    let fb = must(K::make_planar_face(&b, &[]), "face");
    let wide = sew::<K>(&[fa.clone(), fb.clone()], gap * 10.0);
    assert_eq!(
        parts::<K>(&wide, ShapeKind::Edge).len(),
        7,
        "wide tolerance"
    );
    let narrow = sew::<K>(&[fa, fb], gap / 10.0);
    assert_eq!(
        parts::<K>(&narrow, ShapeKind::Edge).len(),
        8,
        "narrow tolerance"
    );
}

/// Sewing is for faces.
///
/// # Panics
/// If something else is accepted.
pub fn sewing_rejects_what_has_no_faces<K: Modelling>() {
    let edge = segment::<K>(p(0., 0., 0.), p(1., 0., 0.));
    assert!(K::sew(&[&edge], 1e-6).is_err());
    assert!(K::sew(&[], 1e-6).is_err());
}

/// A cube of any winding encloses its volume positively.
///
/// # Panics
/// If the volume or validity differs.
pub fn solid_from_a_shell_is_positive_whatever_the_winding<K: Modelling>() {
    type Flip = fn(usize) -> bool;
    let patterns: [Flip; 5] = [|_| false, |_| true, |i| i % 2 == 0, |i| i == 0, |i| i >= 3];
    for (index, flip) in patterns.iter().enumerate() {
        let shell = sew::<K>(&cube_faces::<K>(2.0, flip), 1e-6);
        assert_eq!(must(K::kind(&shell), "kind"), ShapeKind::Shell);
        assert!(must(K::is_closed(&shell), "is_closed"), "pattern {index}");
        let solid = must(K::make_solid(&shell), "make_solid");
        assert_eq!(must(K::kind(&solid), "kind"), ShapeKind::Solid);
        assert_valid::<K>(&solid, format_args!("pattern {index}"));
        volume_is::<K>(&solid, 8.0, format_args!("pattern {index}"));
        area_is::<K>(&solid, 24.0, format_args!("pattern {index}"));
    }
}

/// A box lacking its lid holds nothing.
///
/// # Panics
/// If a solid is made.
pub fn solid_rejects_an_open_shell<K: Modelling>() {
    let mut faces = cube_faces::<K>(2.0, |_| false);
    faces.pop();
    let shell = sew::<K>(&faces, 1e-6);
    assert!(!must(K::is_closed(&shell), "is_closed"));
    assert!(K::make_solid(&shell).is_err());
    let edge = segment::<K>(p(0., 0., 0.), p(1., 0., 0.));
    assert!(K::make_solid(&edge).is_err(), "not a shell");
}

/// A compound's measures are the sums of its parts'.
///
/// # Panics
/// If a measure differs.
pub fn compound_holds_what_it_is_given<K: Modelling>() {
    let a = must(K::make_box(Point3::ORIGIN, Vec3::new(1., 1., 1.)), "box");
    let b = must(K::make_sphere(p(5., 0., 0.), 1.0), "sphere");
    let e = segment::<K>(p(0., 9., 0.), p(1., 9., 0.));
    let group = must(K::make_compound(&[&a, &b, &e]), "make_compound");
    assert_eq!(must(K::kind(&group), "kind"), ShapeKind::Compound);
    assert_eq!(parts::<K>(&group, ShapeKind::Solid).len(), 2);
    volume_is::<K>(&group, 1.0 + ball_volume(1.0), "compound");
    let nested = must(K::make_compound(&[&group, &a]), "nested");
    assert_eq!(
        parts::<K>(&nested, ShapeKind::Solid).len(),
        2,
        "seen through nesting"
    );
    let empty = must(K::make_compound(&[]), "empty");
    assert!(parts::<K>(&empty, ShapeKind::Face).is_empty());
}

/// A torus's volume `2 pi^2 R r^2` and area `4 pi^2 R r`.
///
/// # Panics
/// If a measure differs.
pub fn torus_measures<K: Modelling>() {
    let z = must(Dir3::new([1., 2., 3.]).map_err(Into::into), "dir");
    let tilted = Frame::from_normal(p(1., -2., 4.), z);
    for frame in [Frame::WORLD, tilted] {
        for (major, minor) in [(3.0, 1.0), (10.0, 0.5), (2.0, 1.9)] {
            let torus = must(K::make_torus(frame, major, minor), "make_torus");
            assert_valid::<K>(&torus, "torus");
            volume_is::<K>(&torus, 2.0 * PI * PI * major * minor * minor, "torus");
            area_is::<K>(&torus, 4.0 * PI * PI * major * minor, "torus");
            near_point(
                must(K::center_of_mass(&torus), "center_of_mass"),
                frame.origin,
                "torus center",
            );
        }
    }
}

/// Self-intersecting tori aren't solids.
///
/// # Panics
/// If one is built.
pub fn torus_rejects_bad_radii<K: Modelling>() {
    assert!(K::make_torus(Frame::WORLD, 1.0, 2.0).is_err());
    assert!(K::make_torus(Frame::WORLD, 1.0, 1.0).is_err());
    assert!(K::make_torus(Frame::WORLD, 1.0, 0.0).is_err());
}

fn wave() -> Vec<Point3> {
    (0..12)
        .map(|i| {
            let t = f64::from(i) * 0.7;
            p(t, t.sin() * 2.0, (t * 0.5).cos())
        })
        .collect()
}

/// An interpolating curve hits every point, on its way between them.
///
/// # Panics
/// If a point is missed.
pub fn interpolation_hits_every_point<K: Modelling>() {
    let cases: [Vec<Point3>; 3] = [
        wave(),
        vec![p(0., 0., 0.), p(1., 0., 0.)],
        vec![p(0., 0., 0.), p(1., 1., 0.), p(2., 0., 1.)],
    ];
    for points in cases {
        let curve = must(K::interpolate_curve(&points, false), "interpolate_curve");
        let edge = must(K::make_edge(&curve, None), "make_edge");
        let chord: f64 = points.windows(2).map(|w| w[0].distance(w[1])).sum();
        assert!(must(K::length(&edge), "length") >= chord * (1.0 - MEASURE));
        for point in points {
            let (_, foot) = must(K::project_onto_edge(&edge, point), "project_onto_edge");
            assert!(foot.distance(point) < POSITION, "missed {point:?}");
        }
    }
}

/// A closed curve through points on a circle is a closed edge, close to the circle, with no
/// corner where it closes.
///
/// # Panics
/// If it isn't.
pub fn closed_interpolation_is_smooth_where_it_closes<K: Modelling>() {
    let points: Vec<Point3> = (0..16)
        .map(|i| f64::from(i) * TAU / 16.0)
        .map(|t| p(2.0 * t.cos(), 2.0 * t.sin(), 0.0))
        .collect();
    let curve = must(K::interpolate_curve(&points, true), "interpolate_curve");
    let edge = must(K::make_edge(&curve, None), "make_edge");
    assert!(must(K::is_closed(&edge), "is_closed"));
    near(must(K::length(&edge), "length"), TAU * 2.0, 1e-3, "length");
    let range = line_range::<K>(&edge);
    let start = must(K::edge_point(&edge, range.start), "edge_point");
    let end = must(K::edge_point(&edge, range.end), "edge_point");
    near_point(start.point, end.point, "closure");
    let (a, b) = (start.derivative, end.derivative);
    assert!(
        (a - b).length() < 1e-6 * a.length(),
        "a corner where it closes: {a:?} and {b:?}"
    );
    assert!(
        K::interpolate_curve(&points[..2], true).is_err(),
        "two points can't close"
    );
}

/// No curve moves between two copies of one point in no parameter.
///
/// # Panics
/// If it is accepted.
pub fn interpolation_rejects_coincident_neighbours<K: Modelling>() {
    let points = [p(0., 0., 0.), p(0., 0., 0.), p(1., 0., 0.)];
    assert!(K::interpolate_curve(&points, false).is_err());
    assert!(K::interpolate_curve(&[], false).is_err());
    assert!(K::interpolate_curve(&[p(0., 0., 0.)], false).is_err());
}

/// A fit passes within its tolerance of all the data, and uses fewer poles than it was given
/// points when the shape is simple.
///
/// # Panics
/// If a point is too far.
pub fn fit_stays_within_tolerance<K: Modelling>() {
    let points: Vec<Point3> = (0..200)
        .map(|i| f64::from(i) * 0.05)
        .map(|t| p(t, t.sin(), (t * 0.5).cos()))
        .collect();
    for tolerance in [1e-1, 1e-2, 1e-3] {
        let curve = must(K::fit_curve(&points, tolerance), "fit_curve");
        let edge = must(K::make_edge(&curve, None), "make_edge");
        for point in &points {
            let (_, foot) = must(K::project_onto_edge(&edge, *point), "project_onto_edge");
            assert!(
                foot.distance(*point) <= tolerance * (1.0 + 1e-9),
                "{point:?} is {} from the fit at tolerance {tolerance}",
                foot.distance(*point)
            );
        }
        let spline = spline_of(curve);
        assert!(spline.poles.len() < points.len(), "a fit should compress");
    }
    let line = [p(0., 0., 0.), p(1., 0., 0.)];
    assert!(K::fit_curve(&line, 1e-3).is_ok());
    assert!(K::fit_curve(&[p(0., 0., 0.)], 1e-3).is_err());
}

/// Samples a wire's edges, calling back for each point and tangent in order.
fn walk<K: Modelling>(wire: &K::Shape, mut visit: impl FnMut(Point3, Vec3)) {
    for edge in parts::<K>(wire, ShapeKind::Edge) {
        let geometry = must(K::edge_geometry(&edge), "edge_geometry");
        let (a, b) = (geometry.range.start, geometry.range.end);
        for i in 0..=32 {
            let t = a + (b - a) * f64::from(i) / 32.0;
            let at = must(K::edge_point(&edge, t), "edge_point");
            let tangent = if geometry.reversed {
                -at.derivative
            } else {
                at.derivative
            };
            visit(at.point, tangent);
        }
    }
}

/// A helix stays on its cylinder, rises `pitch` per turn, and has the length of its
/// unrolled line.
///
/// # Panics
/// If a point leaves it.
pub fn helix_follows_the_formula<K: Modelling>() {
    for (radius, pitch, height) in [
        (5.0, 2.0, 6.0),
        (1.0, 0.5, 0.25),
        (10.0, 1.0, 7.5),
        (0.5, 3.0, 12.0),
    ] {
        let wire = must(
            K::make_helix(Axis::Z, radius, pitch, height, false),
            "make_helix",
        );
        assert_eq!(must(K::kind(&wire), "kind"), ShapeKind::Wire);
        assert!(!must(K::is_closed(&wire), "is_closed"));
        assert_valid::<K>(&wire, "helix");
        let turns: f64 = height / pitch;
        let length: f64 = (TAU * radius * turns).hypot(height);
        near(
            must(K::length(&wire), "length"),
            length,
            1e-5,
            "helix length",
        );
        let bbox = must(K::bounding_box(&wire), "bounding_box");
        assert!((bbox.min.z).abs() < 1e-5 && (bbox.max.z - height).abs() < 1e-5);
        walk::<K>(&wire, |at, _| {
            let radial = at.x.hypot(at.y);
            assert!((radial - radius).abs() < 1e-5, "{at:?} is off the cylinder");
            // The height fixes the angle, up to whole turns.
            let angle = TAU * at.z / pitch;
            let along = Vec3::new(angle.cos(), angle.sin(), 0.0);
            let unit = Vec3::new(at.x, at.y, 0.0) / radial;
            assert!(
                (unit - along).length() < 1e-5 * (1.0 + radius),
                "{at:?} is off the helix"
            );
        });
    }
}

/// Seen from above, a right-handed helix rising towards you turns counterclockwise, and a
/// left-handed one the other way.
///
/// # Panics
/// If either turns the wrong way.
pub fn helix_handedness<K: Modelling>() {
    for (left_handed, sign) in [(false, 1.0), (true, -1.0)] {
        let wire = must(
            K::make_helix(Axis::Z, 2.0, 1.0, 2.0, left_handed),
            "make_helix",
        );
        walk::<K>(&wire, |at, tangent| {
            let radial = at - Point3::new(0.0, 0.0, at.z);
            let around = radial.cross(tangent).z * sign;
            assert!(around > 0.0, "turns the wrong way at {at:?}");
            assert!(tangent.z > 0.0, "doesn't rise at {at:?}");
        });
    }
}

/// A helix around a tilted, moved axis keeps its distance from that axis.
///
/// # Panics
/// If it doesn't.
pub fn helix_along_any_axis<K: Modelling>() {
    let direction = must(Dir3::new([1., 1., 1.]).map_err(Into::into), "dir");
    let axis = Axis::new(p(3., -1., 2.), direction);
    let wire = must(K::make_helix(axis, 2.0, 1.0, 3.0, false), "make_helix");
    let z = direction.vector();
    walk::<K>(&wire, |at, tangent| {
        let along = at - axis.origin;
        let rise = along.dot(z);
        assert!((-1e-6..=3.0 + 1e-6).contains(&rise), "rise {rise}");
        let off = along - z * rise;
        assert!((off.length() - 2.0).abs() < 1e-5, "{at:?} is off the axis");
        assert!(tangent.dot(z) > 0.0);
    });
}

/// A circle's offset is a concentric circle.
///
/// # Panics
/// If the area differs.
pub fn offset_of_a_circle_is_a_circle<K: Modelling>() {
    let wire = wire_of::<K>(&[circle::<K>(Point3::ORIGIN, 5.0)]);
    for (distance, want) in [(1.0, 6.0), (-1.0, 4.0), (3.0, 8.0)] {
        let result = must(K::offset_wire(&wire, distance), "offset_wire");
        near(
            must(K::length(&result), "length"),
            TAU * want,
            MEASURE,
            "offset circle",
        );
        near(
            wires_area::<K>(&result),
            PI * want * want,
            1e-4,
            "offset circle area",
        );
    }
}

/// Growing a square gives somewhere between a rounded and a mitred outline, whichever a
/// kernel chooses for corners. Shrinking it is exactly a smaller square.
///
/// # Panics
/// If the area differs.
pub fn offset_of_a_polygon_grows_and_shrinks<K: Modelling>() {
    let (side, d) = (10.0f64, 1.0f64);
    let wire = polygon::<K>(&square(Point3::ORIGIN, side));
    let grown = must(K::offset_wire(&wire, d), "offset_wire");
    let area = wires_area::<K>(&grown);
    let rounded = side * side + 4.0 * side * d + PI * d * d;
    let mitred = (side + 2.0 * d) * (side + 2.0 * d);
    assert!(
        area >= rounded * (1.0 - 1e-4) && area <= mitred * (1.0 + 1e-4),
        "grown area {area} outside {rounded}..{mitred}"
    );
    let shrunk = must(K::offset_wire(&wire, -d), "offset_wire");
    let want = (side - 2.0 * d) * (side - 2.0 * d);
    near(wires_area::<K>(&shrunk), want, 1e-4, "shrunk");
    near(
        must(K::length(&shrunk), "length"),
        4.0 * (side - 2.0 * d),
        1e-4,
        "shrunk perimeter",
    );
}

/// Outward doesn't depend on which way the wire runs.
///
/// # Panics
/// If a clockwise wire goes the other way.
pub fn offset_ignores_winding<K: Modelling>() {
    let corners = square(Point3::ORIGIN, 4.0);
    for points in [corners.clone(), reversed(&corners)] {
        let wire = polygon::<K>(&points);
        assert!(wires_area::<K>(&must(K::offset_wire(&wire, 0.5), "grow")) > 16.0);
        assert!(wires_area::<K>(&must(K::offset_wire(&wire, -0.5), "shrink")) < 16.0);
    }
}

/// Past half its width, nothing is left of a square.
///
/// # Panics
/// If a wire comes back with area.
pub fn offset_past_collapse_fails<K: Modelling>() {
    let wire = polygon::<K>(&square(Point3::ORIGIN, 2.0));
    assert!(K::offset_wire(&wire, -5.0).is_err());
}

/// An open wire's offset runs parallel to it, at the distance, and stays open.
///
/// # Panics
/// If it doesn't.
pub fn offset_of_an_open_wire_is_parallel<K: Modelling>() {
    let arc = must(
        K::make_edge(
            &Curve::Circle {
                frame: Frame::WORLD,
                radius: 5.0,
            },
            Some(range(0.0, FRAC_PI_2)),
        ),
        "arc",
    );
    let wire = wire_of::<K>(&[arc]);
    let result = must(K::offset_wire(&wire, 1.0), "offset_wire");
    // Grown or shrunk per the side a kernel takes, but concentric either way.
    let length = must(K::length(&result), "length");
    let (grown, shrunk) = (6.0 * FRAC_PI_2, 4.0 * FRAC_PI_2);
    assert!(
        (length - grown).abs() < 1e-4 || (length - shrunk).abs() < 1e-4,
        "offset arc length {length}"
    );
    for wire in parts::<K>(&result, ShapeKind::Wire) {
        assert!(!must(K::is_closed(&wire), "is_closed"));
    }
}

/// Offsetting needs a plane.
///
/// # Panics
/// If a skew wire is offset.
pub fn offset_rejects_non_planar_wires<K: Modelling>() {
    let skew = polygon::<K>(&[p(0., 0., 0.), p(1., 0., 0.), p(1., 1., 1.), p(0., 1., 0.)]);
    assert!(K::offset_wire(&skew, 0.1).is_err());
    let helix = must(K::make_helix(Axis::Z, 1.0, 1.0, 2.0, false), "make_helix");
    assert!(K::offset_wire(&helix, 0.1).is_err());
}

/// Every face of a box is a plane, with a normal leaving the box.
///
/// # Panics
/// If a normal points in.
pub fn box_faces_read_as_outward_planes<K: Modelling>() {
    let cube = must(K::make_box(p(1., 2., 3.), Vec3::new(2., 2., 2.)), "box");
    let center = p(2., 3., 4.);
    for face in parts::<K>(&cube, ShapeKind::Face) {
        let geometry = must(K::face_geometry(&face), "face_geometry");
        assert!(matches!(geometry.surface, Surface::Plane { .. }));
        let u = (geometry.u.start + geometry.u.end) / 2.0;
        let v = (geometry.v.start + geometry.v.end) / 2.0;
        let at = must(K::face_point(&face, u, v), "face_point");
        let normal = at.normal.expect("a plane has a normal").vector();
        assert!(normal.dot(at.point - center) > 0.9, "normal points in");
        assert!(at.principal_curvatures[0].abs() < POSITION);
        assert!(at.principal_curvatures[1].abs() < POSITION);
    }
}

/// A cylinder's side reads back as one, with an outward normal and one principal curvature
/// of `1 / r`.
///
/// # Panics
/// If a value differs.
pub fn cylinder_faces_read_back<K: Modelling>() {
    let (r, h) = (2.0, 5.0);
    let solid = must(K::make_cylinder(Axis::Z, r, h), "cylinder");
    let side = parts::<K>(&solid, ShapeKind::Face)
        .into_iter()
        .find(|f| must(K::surface_kind(f), "kind") == SurfaceKind::Cylinder)
        .expect("a cylinder has a side");
    let geometry = must(K::face_geometry(&side), "face_geometry");
    let Surface::Cylinder { radius, frame } = geometry.surface else {
        panic!("expected a cylinder, got {:?}", geometry.surface);
    };
    near(radius, r, MEASURE, "radius");
    assert!((frame.z.vector() - Vec3::new(0., 0., 1.)).length() < POSITION);
    near(geometry.v.end - geometry.v.start, h, MEASURE, "height");
    near(geometry.u.end - geometry.u.start, TAU, MEASURE, "turn");

    let at = must(K::face_point(&side, 1.0, 2.0), "face_point");
    let normal = at.normal.expect("regular").vector();
    let out = Vec3::new(at.point.x, at.point.y, 0.0) / r;
    assert!((normal - out).length() < POSITION, "normal {normal:?}");
    assert!((at.point.z - 2.0).abs() < POSITION);
    near(at.du.length(), r, MEASURE, "dU");
    near(at.dv.length(), 1.0, MEASURE, "dV");
    let mut sizes = at.principal_curvatures.map(f64::abs);
    sizes.sort_by(f64::total_cmp);
    assert!(sizes[0] < POSITION);
    near(sizes[1], 1.0 / r, MEASURE, "curvature");
    assert!(
        at.principal_curvatures[0] >= at.principal_curvatures[1],
        "larger first"
    );
}

/// A cone's half angle comes from its radii and height.
///
/// # Panics
/// If a value differs.
pub fn cone_faces_read_back<K: Modelling>() {
    let (r1, r2, h) = (3.0f64, 1.0f64, 4.0f64);
    let solid = must(K::make_cone(Axis::Z, r1, r2, h), "cone");
    let side = parts::<K>(&solid, ShapeKind::Face)
        .into_iter()
        .find(|f| must(K::surface_kind(f), "kind") == SurfaceKind::Cone)
        .expect("a cone has a side");
    let geometry = must(K::face_geometry(&side), "face_geometry");
    let Surface::Cone {
        radius,
        half_angle,
        frame,
    } = geometry.surface
    else {
        panic!("expected a cone, got {:?}", geometry.surface);
    };
    let want = (r2 - r1).atan2(h);
    assert!(
        (half_angle - want).abs() < POSITION,
        "half angle {half_angle}"
    );
    // The contract's radius is at the frame's origin. The surface may be anchored anywhere
    // along the axis, so compare the radius it implies at the base.
    let base =
        radius + half_angle.tan() * (0.0 - (frame.origin - Point3::ORIGIN).dot(frame.z.vector()));
    assert!((base - r1).abs() < 1e-5, "radius at the base {base}");
}

/// A sphere's normal is radial and its curvatures `1 / r`, except at the poles, where the
/// parameters collapse and there's no normal to give.
///
/// # Panics
/// If a value differs.
pub fn sphere_reads_back_with_singular_poles<K: Modelling>() {
    let r = 2.0;
    let ball = must(K::make_sphere(p(1., 1., 1.), r), "sphere");
    let center = p(1., 1., 1.);
    let face = parts::<K>(&ball, ShapeKind::Face).remove(0);
    let geometry = must(K::face_geometry(&face), "face_geometry");
    assert!(
        matches!(geometry.surface, Surface::Sphere { radius, .. } if (radius - r).abs() < POSITION)
    );
    for (u, v) in [(0.0, 0.0), (1.0, 0.5), (4.0, -1.0)] {
        let at = must(K::face_point(&face, u, v), "face_point");
        let out = (at.point - center) / r;
        assert!((at.point.distance(center) - r).abs() < POSITION);
        let normal = at.normal.expect("regular").vector();
        assert!((normal - out).length() < POSITION, "normal {normal:?}");
        near(
            at.principal_curvatures[0].abs(),
            1.0 / r,
            MEASURE,
            "curvature",
        );
        near(
            at.principal_curvatures[1].abs(),
            1.0 / r,
            MEASURE,
            "curvature",
        );
    }
    let pole = must(K::face_point(&face, 0.0, FRAC_PI_2), "face_point");
    assert!(pole.normal.is_none(), "a pole is singular");
    near_point(pole.point, p(1., 1., 1. + r), "pole");
    assert!(K::face_point(&face, 0.0, 2.0).is_err(), "past the pole");
}

/// A torus's outer equator curves `1 / minor` one way and `1 / (major + minor)` the other.
///
/// # Panics
/// If a value differs.
pub fn torus_reads_back<K: Modelling>() {
    let (major, minor) = (3.0, 1.0);
    let torus = must(K::make_torus(Frame::WORLD, major, minor), "make_torus");
    let face = parts::<K>(&torus, ShapeKind::Face).remove(0);
    let geometry = must(K::face_geometry(&face), "face_geometry");
    assert!(matches!(geometry.surface, Surface::Torus { .. }));
    let at = must(K::face_point(&face, 0.0, 0.0), "face_point");
    near_point(at.point, p(major + minor, 0., 0.), "outer equator");
    let normal = at.normal.expect("regular").vector();
    assert!((normal - Vec3::new(1., 0., 0.)).length() < POSITION);
    let mut sizes = at.principal_curvatures.map(f64::abs);
    sizes.sort_by(f64::total_cmp);
    near(sizes[0], 1.0 / (major + minor), MEASURE, "small curvature");
    near(sizes[1], 1.0 / minor, MEASURE, "large curvature");
}

/// Projection lands where geometry says, and evaluating its parameters gets back there.
///
/// # Panics
/// If a value differs.
pub fn projection_onto_faces<K: Modelling>() {
    let wire = polygon::<K>(&square(Point3::ORIGIN, 2.0));
    let plate = must(K::make_planar_face(&wire, &[]), "face");
    let (u, v, foot) = must(K::project_onto_face(&plate, p(0.25, -0.5, 3.0)), "project");
    near_point(foot, p(0.25, -0.5, 0.0), "onto a plane");
    near_point(
        must(K::face_point(&plate, u, v), "face_point").point,
        foot,
        "evaluated",
    );

    // Past the face's boundary, the surface goes on.
    let (.., beyond) = must(K::project_onto_face(&plate, p(9.0, 9.0, 1.0)), "project");
    near_point(beyond, p(9.0, 9.0, 0.0), "past the boundary");

    let r = 2.0;
    let ball = must(K::make_sphere(Point3::ORIGIN, r), "sphere");
    let face = parts::<K>(&ball, ShapeKind::Face).remove(0);
    for target in [
        p(5., 0., 0.),
        p(0., 0.3, 0.2),
        p(-1., -2., 3.),
        p(0., 1., -7.),
    ] {
        let (u, v, foot) = must(K::project_onto_face(&face, target), "project");
        let want =
            Point3::ORIGIN + (target - Point3::ORIGIN) * (r / (target - Point3::ORIGIN).length());
        near_point(foot, want, "onto a sphere");
        near_point(
            must(K::face_point(&face, u, v), "face_point").point,
            foot,
            "evaluated",
        );
    }
    let (.., any) = must(
        K::project_onto_face(&face, Point3::ORIGIN),
        "project from the center",
    );
    assert!((any.distance(Point3::ORIGIN) - r).abs() < POSITION);
}

/// The closest point of an edge can be an end, and is reported with its parameter.
///
/// # Panics
/// If a value differs.
pub fn projection_onto_edges_stops_at_the_ends<K: Modelling>() {
    let edge = segment::<K>(p(0., 0., 0.), p(4., 0., 0.));
    let (t, foot) = must(K::project_onto_edge(&edge, p(1., 2., 0.)), "project");
    assert!((t - 1.0).abs() < POSITION);
    near_point(foot, p(1., 0., 0.), "foot");
    let (_, end) = must(K::project_onto_edge(&edge, p(9., 2., 0.)), "project");
    near_point(end, p(4., 0., 0.), "past the end");
    let (_, start) = must(K::project_onto_edge(&edge, p(-9., 2., 0.)), "project");
    near_point(start, p(0., 0., 0.), "before the start");

    let arc = must(
        K::make_edge(
            &Curve::Circle {
                frame: Frame::WORLD,
                radius: 2.0,
            },
            Some(range(0.0, FRAC_PI_2)),
        ),
        "arc",
    );
    let (_, on) = must(K::project_onto_edge(&arc, p(3., 3., 0.)), "project");
    let s = 2.0 / 2.0f64.sqrt();
    near_point(on, p(s, s, 0.), "onto an arc");
    let (_, stop) = must(K::project_onto_edge(&arc, p(-1., -3., 0.)), "project");
    assert!(
        stop.distance(p(2., 0., 0.)) < POSITION || stop.distance(p(0., 2., 0.)) < POSITION,
        "{stop:?} is not an end"
    );
    let (_, from_center) = must(K::project_onto_edge(&arc, Point3::ORIGIN), "project");
    assert!((from_center.distance(Point3::ORIGIN) - 2.0).abs() < POSITION);
}

/// Closed means no free ends or boundary.
///
/// # Panics
/// If a report is wrong.
pub fn closedness_of_edges_wires_and_shells<K: Modelling>() {
    let line = segment::<K>(p(0., 0., 0.), p(1., 0., 0.));
    assert!(!must(K::is_closed(&line), "line"));
    let ring = circle::<K>(Point3::ORIGIN, 1.0);
    assert!(must(K::is_closed(&ring), "circle"));
    let open = wire_of::<K>(std::slice::from_ref(&line));
    assert!(!must(K::is_closed(&open), "open wire"));
    let closed = polygon::<K>(&square(Point3::ORIGIN, 1.0));
    assert!(must(K::is_closed(&closed), "closed wire"));
    let cube = must(K::make_box(Point3::ORIGIN, Vec3::new(1., 1., 1.)), "box");
    let shell = parts::<K>(&cube, ShapeKind::Shell).remove(0);
    assert!(must(K::is_closed(&shell), "box shell"));
    let face = parts::<K>(&cube, ShapeKind::Face).remove(0);
    let single = sew::<K>(
        &[face.clone(), parts::<K>(&cube, ShapeKind::Face).remove(1)],
        1e-6,
    );
    if must(K::kind(&single), "kind") == ShapeKind::Shell {
        assert!(!must(K::is_closed(&single), "open shell"));
    }
    assert!(K::is_closed(&face).is_err(), "a face is neither");
    assert!(K::is_closed(&cube).is_err(), "a solid is neither");
}

/// Modelling tolerances are positive and well below the model.
///
/// # Panics
/// If not.
pub fn tolerance_is_small_and_positive<K: Modelling>() {
    let cube = must(K::make_box(Point3::ORIGIN, Vec3::new(1., 1., 1.)), "box");
    let t = must(K::tolerance(&cube), "tolerance");
    assert!(t > 0.0 && t < 1e-3, "tolerance {t}");
    let edge = segment::<K>(p(0., 0., 0.), p(1., 0., 0.));
    let t = must(K::tolerance(&edge), "tolerance");
    assert!(t > 0.0 && t < 1e-3, "tolerance {t}");
    let empty = must(K::make_compound(&[]), "empty");
    assert!(must(K::tolerance(&empty), "tolerance") >= 0.0);
}

/// Handing an operation the wrong kind of shape is an error.
///
/// # Panics
/// If one succeeds.
pub fn wrong_kinds_are_errors_not_crashes<K: Modelling>() {
    let cube = must(K::make_box(Point3::ORIGIN, Vec3::new(1., 1., 1.)), "box");
    let face = parts::<K>(&cube, ShapeKind::Face).remove(0);
    let edge = parts::<K>(&cube, ShapeKind::Edge).remove(0);
    assert!(K::make_wire(&[&face]).is_err());
    assert!(K::make_planar_face(&edge, &[]).is_err());
    assert!(K::make_planar_face(&cube, &[]).is_err());
    assert!(K::fill_boundary(&[&face], &[]).is_err());
    assert!(K::make_solid(&cube).is_err());
    assert!(K::offset_wire(&edge, 1.0).is_err());
    assert!(K::edge_geometry(&face).is_err());
    assert!(K::edge_point(&face, 0.0).is_err());
    assert!(K::face_geometry(&edge).is_err());
    assert!(K::face_point(&edge, 0.0, 0.0).is_err());
    assert!(K::project_onto_face(&edge, Point3::ORIGIN).is_err());
    assert!(K::project_onto_edge(&face, Point3::ORIGIN).is_err());
    let plane = Surface::Plane {
        frame: Frame::WORLD,
    };
    assert!(K::make_surface_face(&plane, None, Some(&edge), &[]).is_err());
}
