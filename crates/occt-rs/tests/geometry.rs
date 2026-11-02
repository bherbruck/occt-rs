//! Construction and geometry reading, against the specification and through the public API.

use std::f64::consts::{FRAC_PI_2, PI, TAU};

use occt::prelude::*;
use occt::{
    BSplineCurve, BoundaryFill, Compound, Curve, Frame, Helix, ParamRange, PlanarFace, Sewing,
    Shell, Spline, Surface, SurfaceFace, Torus, Wire,
};
use occt_conformance::construction_and_geometry::checks;
use occt_kernel_capi::CapiKernel;

#[test]
fn the_kernel_passes_every_check() {
    let failed: Vec<_> = checks::<CapiKernel>()
        .into_iter()
        .filter(|(name, check)| {
            let failed = std::panic::catch_unwind(*check).is_err();
            if failed {
                eprintln!("FAILED {name}");
            }
            failed
        })
        .map(|(name, _)| name)
        .collect();
    assert!(failed.is_empty(), "failed checks: {failed:?}");
}

fn invalid<T: std::fmt::Debug>(result: occt::Result<T>) {
    assert!(
        matches!(result, Err(Error::InvalidInput(_))),
        "expected invalid input, got {result:?}"
    );
}

fn near(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-6
}

const NAN: f64 = f64::NAN;
const INF: f64 = f64::INFINITY;

#[test]
fn plain_arguments_are_checked_before_the_kernel() {
    invalid(Vertex::new([NAN, 0.0, 0.0]));
    invalid(Edge::line([0.0; 3], [0.0; 3]));
    invalid(Edge::line([0.0; 3], [INF, 0.0, 0.0]));
    invalid(Edge::circle([0.0; 3], Dir3::Z, 0.0));
    invalid(Edge::circle([0.0; 3], Dir3::Z, -1.0));
    invalid(Edge::circle([0.0; 3], [0.0; 3], 1.0));
    invalid(Edge::circle([NAN; 3], Dir3::Z, 1.0));
    invalid(Edge::arc([0.0; 3], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]));
    invalid(Edge::arc([0.0; 3], [0.0; 3], [2.0, 0.0, 0.0]));
    invalid(Edge::arc([0.0; 3], [NAN; 3], [2.0, 0.0, 0.0]));
    invalid(Edge::ellipse(Frame::WORLD, 1.0, 2.0));
    invalid(Edge::ellipse(Frame::WORLD, 1.0, 0.0));
    invalid(Edge::spline([[0.0; 3]]));
    invalid(Edge::spline([[0.0; 3], [0.0; 3], [1.0; 3]]));
    invalid(Edge::spline([[NAN; 3], [1.0; 3]]));
    invalid(Edge::closed_spline([[0.0; 3], [1.0; 3], [0.0; 3]]));
    invalid(Wire::new(Vec::<Edge>::new()));
    invalid(Wire::polygon([[0.0; 3], [1.0; 3]]));
    invalid(Wire::polyline([[0.0; 3]]));
    invalid(Wire::polyline([[0.0; 3], [0.0; 3]]));
    invalid(Wire::rectangle(0.0, 1.0));
    invalid(Wire::rectangle(1.0, NAN));
    invalid(Sewing::new().build());
    invalid(Sewing::new().tolerance(0.0).build());
    invalid(Torus::new(1.0, 1.0).build());
    invalid(Torus::new(NAN, 1.0).build());
    invalid(Torus::new(2.0, 1.0).at([NAN; 3]).build());
    invalid(Helix::new(0.0, 1.0, 1.0).build());
    invalid(Helix::new(1.0, -1.0, 1.0).build());
    invalid(Helix::new(1.0, 1.0, INF).build());
    invalid(Spline::through(Vec::<[f64; 3]>::new()).build());
    invalid(Spline::fit([[0.0; 3], [0.0; 3]]).build());
    invalid(Spline::fit([[0.0; 3], [1.0; 3]]).tolerance(NAN).build());
    invalid(Face::fill(Vec::<Edge>::new()).build());
}

#[test]
fn curve_data_is_checked_before_the_kernel() {
    let frame = Frame::WORLD;
    let line = Curve::Line {
        origin: Point3::ORIGIN,
        direction: Dir3::X,
    };
    invalid(Edge::from_curve(line.clone()).build());
    invalid(Edge::from_curve(line.clone()).range(1.0..0.0).build());
    invalid(Edge::from_curve(line.clone()).range(NAN..1.0).build());
    invalid(Edge::from_curve(Curve::Other).build());
    invalid(
        Edge::from_curve(Curve::Circle { frame, radius: 1.0 })
            .range(0.0..10.0)
            .build(),
    );
    let good = BSplineCurve {
        degree: 2,
        knots: vec![0.0, 1.0],
        multiplicities: vec![3, 3],
        poles: vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
        ],
        weights: None,
        periodic: false,
    };
    assert!(
        Edge::from_curve(Curve::BSpline(good.clone()))
            .build()
            .is_ok()
    );
    for bad in [
        BSplineCurve {
            degree: 0,
            ..good.clone()
        },
        BSplineCurve {
            knots: vec![1.0, 0.0],
            ..good.clone()
        },
        BSplineCurve {
            knots: vec![0.0, NAN],
            ..good.clone()
        },
        BSplineCurve {
            multiplicities: vec![3],
            ..good.clone()
        },
        BSplineCurve {
            multiplicities: vec![4, 4],
            ..good.clone()
        },
        BSplineCurve {
            poles: good.poles[..2].to_vec(),
            ..good.clone()
        },
        BSplineCurve {
            weights: Some(vec![1.0, 1.0]),
            ..good.clone()
        },
        BSplineCurve {
            weights: Some(vec![1.0, -1.0, 1.0]),
            ..good.clone()
        },
        BSplineCurve {
            weights: Some(vec![1.0, NAN, 1.0]),
            ..good.clone()
        },
    ] {
        invalid(Edge::from_curve(Curve::BSpline(bad)).build());
    }
}

#[test]
fn surface_data_is_checked_before_the_kernel() {
    let frame = Frame::WORLD;
    let square = Wire::rectangle(1.0, 1.0).unwrap();
    invalid(Face::from_surface(Surface::Other).build());
    invalid(Face::from_surface(Surface::Cylinder { frame, radius: 0.0 }).build());
    invalid(
        Face::from_surface(Surface::Cone {
            frame,
            radius: 1.0,
            half_angle: FRAC_PI_2,
        })
        .build(),
    );
    invalid(
        Face::from_surface(Surface::Plane { frame })
            .bounds(1.0..0.0, 0.0..1.0)
            .build(),
    );
    invalid(
        Face::from_surface(Surface::Plane { frame })
            .bounds(0.0..1.0, 0.0..NAN)
            .build(),
    );
    invalid(
        Face::from_surface(Surface::Plane { frame })
            .bounds(0.0..1.0, 0.0..1.0)
            .outer(&square)
            .build(),
    );
    invalid(
        Face::from_surface(Surface::Plane { frame })
            .bounds(0.0..1.0, 0.0..1.0)
            .hole(&square)
            .build(),
    );
}

#[test]
fn evaluation_checks_its_arguments() {
    let edge = Edge::line([0.0; 3], [1.0, 0.0, 0.0]).unwrap();
    invalid(edge.point_at(NAN));
    invalid(edge.project([NAN; 3]));
    assert!(matches!(edge.point_at(5.0), Err(Error::Operation { .. })));
    let cube = Cuboid::new([1.0, 1.0, 1.0]).build().unwrap();
    let face = cube.faces().unwrap().next().unwrap();
    invalid(face.point_at(INF, 0.0));
    invalid(face.project([NAN; 3]));
    invalid(cube.is_closed());
    invalid(face.is_closed());
}

#[test]
fn an_arc_goes_through_its_middle_point() {
    let s = 0.5f64.sqrt();
    type Case = ([f64; 3], [f64; 3], [f64; 3], f64);
    let cases: [Case; 3] = [
        ([1.0, 0.0, 0.0], [s, s, 0.0], [0.0, 1.0, 0.0], FRAC_PI_2),
        ([1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], PI),
        // The long way round.
        (
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            3.0 * FRAC_PI_2,
        ),
    ];
    for (a, m, b, length) in cases {
        let arc = Edge::arc(a, m, b).unwrap();
        assert!(near(arc.length().unwrap(), length), "{a:?} {m:?} {b:?}");
        assert!(arc.start_point().unwrap().distance(a.into()) < 1e-9);
        assert!(arc.end_point().unwrap().distance(b.into()) < 1e-9);
        assert!(arc.project(m).unwrap().point.distance(m.into()) < 1e-9);
    }
}

#[test]
fn wires_walk_in_order_and_edges_report_their_direction() {
    let a = Edge::line([0.0; 3], [1.0, 0.0, 0.0]).unwrap();
    let b = Edge::line([2.0, 0.0, 0.0], [1.0, 0.0, 0.0]).unwrap();
    let c = Edge::line([2.0, 0.0, 0.0], [2.0, 1.0, 0.0]).unwrap();
    let wire = Wire::new([&c, &a, &b]).unwrap();
    let edges: Vec<Edge> = wire.edges().unwrap().collect();
    assert_eq!(edges.len(), 3);
    for pair in edges.windows(2) {
        assert!(
            pair[0]
                .end_point()
                .unwrap()
                .distance(pair[1].start_point().unwrap())
                < 1e-9,
            "{:?} and {:?}",
            pair[0].end_point(),
            pair[1].start_point()
        );
    }
    // Whichever end it starts from, the middle edge runs against its own curve in one case.
    let middle = edges
        .iter()
        .find(|e| {
            near(e.project([1.5, 0.0, 0.0]).unwrap().point.y, 0.0)
                && near(e.length().unwrap(), 1.0)
                && e.bounding_box().unwrap().size().x > 0.5
                && e.bounding_box().unwrap().min.x > 0.5
        })
        .unwrap();
    let ends = (
        middle.start_point().unwrap().x,
        middle.end_point().unwrap().x,
    );
    assert!(ends == (1.0, 2.0) || ends == (2.0, 1.0), "{ends:?}");
}

#[test]
fn builders_take_references_and_values() {
    let outer = Wire::rectangle(4.0, 4.0).unwrap();
    let hole = Wire::rectangle(1.0, 1.0).unwrap();
    let by_ref: PlanarFace = Face::planar(&outer).hole(&hole);
    assert!(near(by_ref.build().unwrap().area().unwrap(), 15.0));
    let by_value = Face::planar(outer.clone()).holes([hole.clone()]);
    assert!(near(by_value.build().unwrap().area().unwrap(), 15.0));
    let many = Face::planar(&outer).holes([&hole]);
    assert!(near(many.build().unwrap().area().unwrap(), 15.0));

    let frame = Frame::WORLD;
    let builder: SurfaceFace = Face::from_surface(Surface::Plane { frame }).outer(&outer);
    assert!(near(builder.build().unwrap().area().unwrap(), 16.0));
    let circle = Edge::circle([0.0; 3], Dir3::Z, 1.0).unwrap();
    let fill: BoundaryFill = Face::fill(vec![circle]);
    assert!(fill.build().is_ok());
}

#[test]
fn a_cube_goes_through_every_level() {
    let s = 2.0;
    let square =
        |z: f64| Wire::polygon([[0.0, 0.0, z], [s, 0.0, z], [s, s, z], [0.0, s, z]]).unwrap();
    let bottom = Face::planar(square(0.0)).build().unwrap();
    let top = Face::planar(square(s)).build().unwrap();
    let mut faces = vec![bottom, top];
    for w in [
        [[0.0, 0.0, 0.0], [s, 0.0, 0.0], [s, 0.0, s], [0.0, 0.0, s]],
        [[s, 0.0, 0.0], [s, s, 0.0], [s, s, s], [s, 0.0, s]],
        [[s, s, 0.0], [0.0, s, 0.0], [0.0, s, s], [s, s, s]],
        [[0.0, s, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, s], [0.0, s, s]],
    ] {
        faces.push(Face::planar(Wire::polygon(w).unwrap()).build().unwrap());
    }
    let shell = Shell::try_from(Sewing::new().add_all(&faces).build().unwrap()).unwrap();
    assert!(shell.is_closed().unwrap());
    assert_eq!(shell.edges().unwrap().count(), 12);
    assert_eq!(shell.vertices().unwrap().count(), 8);
    let solid = Solid::from_shell(&shell).unwrap();
    assert!(solid.is_valid().unwrap());
    assert!(near(solid.volume().unwrap(), 8.0));
    assert!(near(
        Sewing::new()
            .add_all(faces)
            .build_solid()
            .unwrap()
            .volume()
            .unwrap(),
        8.0
    ));

    let group = Compound::new([&*solid, &*shell]).unwrap();
    // The solid's skin is that shell, so the group holds one.
    assert_eq!(group.shells().unwrap().count(), 1);
    assert_eq!(group.solids().unwrap().count(), 1);
    assert_eq!(
        Compound::try_from(Shape::from(group))
            .unwrap()
            .compounds()
            .unwrap()
            .count(),
        1
    );
    assert!(matches!(
        Shell::try_from(&*solid),
        Err(Error::WrongKind { .. })
    ));
    assert!(matches!(
        Wire::try_from(&*solid),
        Err(Error::WrongKind { .. })
    ));
    assert_eq!(
        Compound::new(Vec::<Shape>::new())
            .unwrap()
            .faces()
            .unwrap()
            .count(),
        0
    );
}

#[test]
fn sewing_faces_that_stay_apart_gives_no_solid() {
    let a = Polygon::new([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        .build()
        .unwrap();
    let b = Polygon::new([[5.0, 0.0, 0.0], [6.0, 0.0, 0.0], [5.0, 1.0, 0.0]])
        .build()
        .unwrap();
    assert!(Sewing::new().add(&a).add(&b).build_solid().is_err());
}

#[test]
fn analytic_edges_and_faces_rebuild_from_what_they_read() {
    let arc = Edge::arc([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]).unwrap();
    let geometry = arc.geometry().unwrap();
    let ParamRange { start, end } = geometry.range;
    let again = Edge::from_curve(geometry.curve)
        .range(start..end)
        .build()
        .unwrap();
    assert!(near(again.length().unwrap(), PI));

    let pipe = Cylinder::new(2.0, 3.0).build().unwrap();
    let side = pipe
        .faces()
        .unwrap()
        .find(|f| f.surface_kind().unwrap() == SurfaceKind::Cylinder)
        .unwrap();
    let geometry = side.geometry().unwrap();
    let again = Face::from_surface(geometry.surface)
        .bounds(
            geometry.u.start..geometry.u.end,
            geometry.v.start..geometry.v.end,
        )
        .build()
        .unwrap();
    assert!(near(again.area().unwrap(), TAU * 2.0 * 3.0));
}

#[test]
fn undescribed_geometry_still_evaluates() {
    // A sphere's edges include a pole, which is a point with no curve.
    let ball = Sphere::new(1.0).build().unwrap();
    let pole = ball
        .edges()
        .unwrap()
        .find(|e| {
            e.curve_kind().is_ok_and(|k| k == CurveKind::Other)
                || e.length().is_ok_and(|l| l < 1e-9)
        })
        .expect("a sphere has degenerate edges");
    assert!(pole.geometry().unwrap().curve == Curve::Other);
    let range = pole.geometry().unwrap().range;
    assert!(pole.point_at(range.start).is_ok());
    assert!(pole.project([0.0, 0.0, 0.0]).is_err());
}

#[test]
fn a_swept_spline_has_no_description_but_evaluates() {
    let profile = Edge::spline([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 0.0, 0.0]]).unwrap();
    let sheet = profile.extruded([0.0, 0.0, 2.0]).unwrap();
    let face = sheet.faces().unwrap().next().unwrap();
    let geometry = face.geometry().unwrap();
    assert_eq!(geometry.surface, Surface::Other);
    // Which way V runs along the sweep is the kernel's choice; the two ends span the height.
    let at = face.point_at(geometry.u.start, geometry.v.end).unwrap();
    let other = face.point_at(geometry.u.start, geometry.v.start).unwrap();
    assert!(
        (at.point.z - other.point.z).abs() - 2.0 < 1e-6,
        "{geometry:?}"
    );
    assert!(at.point.x.hypot(at.point.y) < 1e-6);
    assert!(at.normal.is_some());
    assert!((face.project([0.0, 0.0, 1.0]).unwrap().point.z - 1.0).abs() < 1e-6);
    invalid(Face::from_surface(geometry.surface).build());
}

#[test]
fn helix_builder_options() {
    let axis = Axis::new([1.0, 0.0, 0.0].into(), Dir3::X);
    let path = Helix::new(2.0, 1.0, 2.5)
        .along(axis)
        .left_handed()
        .build()
        .unwrap();
    assert_eq!(path.edges().unwrap().count(), 3);
    assert!(path.is_valid().unwrap());
    let size = path.bounding_box().unwrap().size();
    assert!((size.x - 2.5).abs() < 1e-3);
    assert!(
        Helix::new(1.0, 1e-9, 1.0).build().is_err(),
        "too many turns"
    );
}

#[test]
fn torus_builder_options() {
    let torus = Torus::new(3.0, 1.0)
        .along(Axis::new([0.0, 0.0, 1.0].into(), Dir3::X))
        .build()
        .unwrap();
    let size = torus.bounding_box().unwrap().size();
    assert!((size.x - 2.0).abs() < 1e-3);
    assert!((size.y - 8.0).abs() < 1e-3);
    let moved = Torus::new(3.0, 1.0).at([1.0, 2.0, 3.0]).build().unwrap();
    assert!(
        moved
            .center_of_mass()
            .unwrap()
            .distance([1.0, 2.0, 3.0].into())
            < 1e-6
    );
}

#[test]
fn wire_offset_gives_wires() {
    let wire = Wire::circle([0.0; 3], Dir3::Z, 5.0).unwrap();
    let out = wire.offset(1.0).unwrap();
    assert_eq!(out.len(), 1);
    assert!((out[0].length().unwrap() - TAU * 6.0).abs() < 1e-5);
    invalid(wire.offset(0.0));
    invalid(wire.offset(NAN));
    let straight = Wire::polyline([[0.0; 3], [1.0, 0.0, 0.0]]).unwrap();
    assert!(matches!(straight.offset(1.0), Err(Error::Operation { .. })));
}

#[test]
fn a_wire_is_usable_with_existing_operations() {
    let square = Wire::rectangle(2.0, 2.0).unwrap();
    let plate = Face::planar(&square).build().unwrap();
    let block = plate.extruded([0.0, 0.0, 3.0]).unwrap();
    assert!(near(block.volume().unwrap(), 12.0));
}

#[test]
fn tolerances_read_from_real_shapes() {
    let cube = Cuboid::new([1.0, 1.0, 1.0]).build().unwrap();
    assert!(cube.tolerance().unwrap() > 0.0);
    let frame = Frame::WORLD;
    let ellipse = Edge::ellipse(frame, 3.0, 1.0).unwrap();
    assert!(ellipse.tolerance().unwrap() < 1e-5);
    assert!(matches!(ellipse.curve().unwrap(), Curve::Ellipse { .. }));
}
