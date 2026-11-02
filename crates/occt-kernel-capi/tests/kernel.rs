//! Runs the kernel for real against the occt_capi module.

use std::f64::consts::PI;

use occt_kernel::{
    Angle, Axis, Blends, Booleans, CurveKind, Dir3, Error, Kernel, MeshOptions, Point3, Primitives,
    Properties, ShapeKind, StepIo, SurfaceKind, Sweeps, Tessellation, Topology, Transforms, Vec3,
};
use occt_kernel_capi::{CapiKernel as K, CapiShape};

fn unit_box(size: f64) -> CapiShape {
    K::make_box(Point3::ORIGIN, Vec3::new(size, size, size)).unwrap()
}

fn close(a: f64, b: f64, tolerance: f64) {
    assert!(
        (a - b).abs() <= tolerance,
        "{a} is not within {tolerance} of {b}"
    );
}

#[test]
fn reports_ready_and_version() {
    K::ready().unwrap();
    assert!(K::version().unwrap().starts_with("8.0.1"));
    assert!(K::NOTICE.contains("Open CASCADE Technology"));
    assert!(K::NOTICE.contains("https://github.com/Open-Cascade-SAS/OCCT"));
}

#[test]
fn box_measures() {
    let b = K::make_box(Point3::new(1.0, 2.0, 3.0), Vec3::new(2.0, 3.0, 4.0)).unwrap();
    assert_eq!(K::kind(&b).unwrap(), ShapeKind::Solid);
    assert!(K::is_valid(&b).unwrap());
    close(K::volume(&b).unwrap(), 24.0, 1e-9);
    close(K::area(&b).unwrap(), 52.0, 1e-9);
    close(K::length(&b).unwrap(), 36.0, 1e-9);
    let c = K::center_of_mass(&b).unwrap();
    assert!(c.distance(Point3::new(2.0, 3.5, 5.0)) < 1e-9);
    let bb = K::bounding_box(&b).unwrap();
    assert!(bb.min.distance(Point3::new(1.0, 2.0, 3.0)) < 1e-6, "{bb:?}");
    assert!(bb.max.distance(Point3::new(3.0, 5.0, 7.0)) < 1e-6, "{bb:?}");
}

#[test]
fn a_face_encloses_no_volume() {
    let z1 = |x, y| Point3::new(x, y, 1.0);
    let face =
        K::make_polygon_face(&[z1(0.0, 0.0), z1(2.0, 0.0), z1(2.0, 2.0), z1(0.0, 2.0)]).unwrap();
    assert_eq!(K::kind(&face).unwrap(), ShapeKind::Face);
    assert_eq!(K::surface_kind(&face).unwrap(), SurfaceKind::Plane);
    close(K::volume(&face).unwrap(), 0.0, 1e-12);
    close(K::area(&face).unwrap(), 4.0, 1e-9);
    assert!(K::center_of_mass(&face).unwrap().distance(z1(1.0, 1.0)) < 1e-9);

    let prism = K::extrude(&face, Vec3::new(0.0, 0.0, 3.0)).unwrap();
    close(K::volume(&prism).unwrap(), 12.0, 1e-9);
    let ring = K::translate(&face, Vec3::new(5.0, 0.0, -1.0)).unwrap();
    let y_at_origin = Axis::new(Point3::ORIGIN, Dir3::Y);
    let turned = K::revolve(&ring, y_at_origin, Angle::FULL_TURN).unwrap();
    close(K::volume(&turned).unwrap(), PI * (49.0 - 25.0) * 2.0, 1e-6);
}

#[test]
fn cut_cylinder_from_box() {
    let plate = K::make_box(Point3::ORIGIN, Vec3::new(10.0, 10.0, 2.0)).unwrap();
    let axis = Axis::new(Point3::new(5.0, 5.0, -1.0), Dir3::Z);
    let tool = K::make_cylinder(axis, 2.0, 4.0).unwrap();
    let cut = K::cut(&plate, &tool).unwrap();
    assert_eq!(K::kind(&cut).unwrap(), ShapeKind::Solid);
    assert!(K::is_valid(&cut).unwrap());
    close(K::volume(&cut).unwrap(), 200.0 - PI * 8.0, 1e-6);

    let kinds: Vec<_> = K::subshapes(&cut, ShapeKind::Face)
        .unwrap()
        .iter()
        .map(|f| K::surface_kind(f).unwrap())
        .collect();
    assert_eq!(
        kinds
            .iter()
            .filter(|k| **k == SurfaceKind::Cylinder)
            .count(),
        1
    );
    assert_eq!(kinds.len(), 7);
}

#[test]
fn other_booleans_and_primitives() {
    let a = unit_box(2.0);
    let b = K::translate(&a, Vec3::new(1.0, 0.0, 0.0)).unwrap();
    close(K::volume(&K::fuse(&a, &b).unwrap()).unwrap(), 12.0, 1e-9);
    close(K::volume(&K::common(&a, &b).unwrap()).unwrap(), 4.0, 1e-9);

    let s = K::make_sphere(Point3::ORIGIN, 2.0).unwrap();
    close(K::volume(&s).unwrap(), 4.0 / 3.0 * PI * 8.0, 1e-6);
    let cone = K::make_cone(Axis::Z, 2.0, 0.0, 3.0).unwrap();
    close(K::volume(&cone).unwrap(), PI * 4.0, 1e-6);
    let far = K::translate(&a, Vec3::new(100.0, 0.0, 0.0)).unwrap();
    close(
        K::volume(&K::common(&a, &far).unwrap()).unwrap(),
        0.0,
        1e-12,
    );
}

#[test]
fn transforms() {
    let a = unit_box(1.0);
    let moved = K::translate(&a, Vec3::new(10.0, 0.0, 0.0)).unwrap();
    let turned = K::rotate(&moved, Axis::Z, Angle::degrees(90.0)).unwrap();
    let c = K::center_of_mass(&turned).unwrap();
    assert!(c.distance(Point3::new(-0.5, 10.5, 0.5)) < 1e-9, "{c:?}");
    let scaled = K::scale(&a, Point3::ORIGIN, 3.0).unwrap();
    close(K::volume(&scaled).unwrap(), 27.0, 1e-9);
    assert!(!K::is_same(&a, &moved));
}

#[test]
fn identity_and_traversal() {
    let a = unit_box(1.0);
    let clone = a.clone();
    assert!(K::is_same(&a, &clone));
    assert_eq!(K::hash_code(&a), K::hash_code(&clone));
    assert!(!K::is_same(&a, &unit_box(1.0)));

    let counts = [
        (ShapeKind::Solid, 1),
        (ShapeKind::Face, 6),
        (ShapeKind::Edge, 12),
    ];
    for (kind, count) in counts {
        assert_eq!(K::subshapes(&a, kind).unwrap().len(), count, "{kind}");
    }
    let vertices = K::subshapes(&a, ShapeKind::Vertex).unwrap();
    assert_eq!(vertices.len(), 8);
    assert!(K::vertex_point(&vertices[0]).unwrap().x.abs() <= 1.0);
    let edges = K::subshapes(&a, ShapeKind::Edge).unwrap();
    assert_eq!(K::curve_kind(&edges[0]).unwrap(), CurveKind::Line);
    let again = K::subshapes(&a, ShapeKind::Edge).unwrap();
    for (e, f) in edges.iter().zip(&again) {
        assert!(K::is_same(e, f));
    }

    let debug = format!("{a:?}");
    assert!(
        debug.contains("handle") && debug.contains("Solid"),
        "{debug}"
    );
}

#[test]
fn shapes_outlive_the_ones_they_came_from() {
    let edges = {
        let a = unit_box(1.0);
        let clone = a.clone();
        drop(a);
        close(K::volume(&clone).unwrap(), 1.0, 1e-9);
        K::subshapes(&clone, ShapeKind::Edge).unwrap()
    };
    close(K::length(&edges[0]).unwrap(), 1.0, 1e-9);
    drop(edges);

    let a = unit_box(2.0);
    let b = unit_box(1.0);
    let cut = K::cut(&a, &b).unwrap();
    drop(a);
    drop(b);
    close(K::volume(&cut).unwrap(), 7.0, 1e-9);
}

fn operation_message(e: Error, expected_op: &str) -> String {
    match e {
        Error::Operation { op, message } => {
            assert_eq!(op, expected_op);
            assert!(!message.is_empty());
            message
        }
        other => panic!("expected an operation error, got {other:?}"),
    }
}

#[test]
fn failures_become_errors() {
    // Bypasses the caller's validation on purpose.
    let e = K::make_box(Point3::ORIGIN, Vec3::ZERO).unwrap_err();
    let message = operation_message(e, "make_box");
    assert!(message.contains("Standard_"), "{message}");

    operation_message(
        K::make_sphere(Point3::ORIGIN, -1.0).unwrap_err(),
        "make_sphere",
    );
    operation_message(
        K::make_polygon_face(&[Point3::ORIGIN]).unwrap_err(),
        "make_polygon_face",
    );
    operation_message(K::read_step(b"not a step file").unwrap_err(), "read_step");
    operation_message(K::read_step(b"").unwrap_err(), "read_step");

    let solid = unit_box(1.0);
    operation_message(K::fillet(&solid, &[], 0.1).unwrap_err(), "fillet");
    operation_message(K::scale(&solid, Point3::ORIGIN, 0.0).unwrap_err(), "scale");

    // The error is cleared by the next good call.
    close(K::volume(&solid).unwrap(), 1.0, 1e-9);
}

#[test]
fn wrong_kinds_are_reported() {
    let solid = unit_box(1.0);
    let kind_error = |e| {
        assert!(
            matches!(
                e,
                Error::WrongKind {
                    found: ShapeKind::Solid,
                    ..
                }
            ),
            "unexpected error {e:?}"
        );
    };
    kind_error(K::vertex_point(&solid).unwrap_err());
    kind_error(K::surface_kind(&solid).unwrap_err());
    kind_error(K::curve_kind(&solid).unwrap_err());
    kind_error(K::fillet(&solid, &[&solid], 0.1).unwrap_err());
}

#[test]
fn fillet_rounds_an_edge() {
    let a = unit_box(10.0);
    let edges = K::subshapes(&a, ShapeKind::Edge).unwrap();
    let r = 1.0;
    let round = K::fillet(&a, &[&edges[0]], r).unwrap();
    assert!(K::is_valid(&round).unwrap());
    close(
        K::volume(&round).unwrap(),
        1000.0 - (1.0 - PI / 4.0) * r * r * 10.0,
        1e-5,
    );
    let faces = K::subshapes(&round, ShapeKind::Face).unwrap();
    assert_eq!(faces.len(), 7);

    let all: Vec<&CapiShape> = edges.iter().collect();
    let all_round = K::fillet(&a, &all, r).unwrap();
    assert!(K::is_valid(&all_round).unwrap());
    assert!(K::volume(&all_round).unwrap() < 1000.0);

    let e = K::fillet(&a, &[&edges[0]], 100.0).unwrap_err();
    operation_message(e, "fillet");

    let other = unit_box(1.0);
    let foreign = K::subshapes(&other, ShapeKind::Edge).unwrap();
    operation_message(K::fillet(&a, &[&foreign[0]], r).unwrap_err(), "fillet");
}

#[test]
fn chamfer_bevels_an_edge() {
    let a = unit_box(10.0);
    let edges = K::subshapes(&a, ShapeKind::Edge).unwrap();
    let bevelled = K::chamfer(&a, &[&edges[0]], 1.0).unwrap();
    assert!(K::is_valid(&bevelled).unwrap());
    close(K::volume(&bevelled).unwrap(), 1000.0 - 5.0, 1e-6);
}

#[test]
fn mesh_of_a_solid_is_closed_and_outward() {
    let b = unit_box(2.0);
    let mesh = K::mesh(&b, &MeshOptions::default()).unwrap();
    assert_eq!(mesh.positions.len(), mesh.normals.len());
    assert_eq!(mesh.indices.len(), 12);
    close(mesh.volume(), 8.0, 1e-6);
    close(mesh.area(), 24.0, 1e-6);

    let center = Point3::new(1.0, 1.0, 1.0);
    for (p, n) in mesh.positions.iter().zip(&mesh.normals) {
        close(n.length(), 1.0, 1e-5);
        assert!(
            (*p - center).dot(*n) > 0.0,
            "normal {n:?} at {p:?} points inward"
        );
    }

    let plate = K::make_box(Point3::ORIGIN, Vec3::new(10.0, 10.0, 2.0)).unwrap();
    let axis = Axis::new(Point3::new(5.0, 5.0, -1.0), Dir3::Z);
    let hole = K::make_cylinder(axis, 2.0, 4.0).unwrap();
    let part = K::cut(&plate, &hole).unwrap();
    let fine = MeshOptions::default()
        .absolute(0.001)
        .angular(Angle::degrees(5.0));
    let mesh = K::mesh(&part, &fine).unwrap();
    close(mesh.volume(), K::volume(&part).unwrap(), 0.01);

    // Reversed faces: the hole's normals point towards its axis.
    let on_wall = |p: &Point3| ((p.x - 5.0).hypot(p.y - 5.0) - 2.0).abs() < 1e-6;
    let wall: Vec<_> = mesh
        .positions
        .iter()
        .zip(&mesh.normals)
        .filter(|(p, n)| on_wall(p) && n.z.abs() < 1e-6)
        .collect();
    assert!(!wall.is_empty());
    for (p, n) in wall {
        assert!(
            (p.x - 5.0) * n.x + (p.y - 5.0) * n.y < 0.0,
            "normal {n:?} at {p:?}"
        );
    }
}

#[test]
fn step_round_trip() {
    let plate = K::make_box(Point3::ORIGIN, Vec3::new(10.0, 10.0, 2.0)).unwrap();
    let hole = K::make_cylinder(Axis::new(Point3::new(5.0, 5.0, -1.0), Dir3::Z), 2.0, 4.0).unwrap();
    let part = K::cut(&plate, &hole).unwrap();

    let step = K::write_step(&part).unwrap();
    assert!(step.starts_with(b"ISO-10303-21"));
    let back = K::read_step(&step).unwrap();
    assert_eq!(K::kind(&back).unwrap(), ShapeKind::Solid);
    assert!(K::is_valid(&back).unwrap());
    close(K::volume(&back).unwrap(), K::volume(&part).unwrap(), 1e-6);
    assert_eq!(K::subshapes(&back, ShapeKind::Face).unwrap().len(), 7);
}

#[test]
fn stale_handles_are_errors() {
    // Never handed out: handles are small and only count up from 1.
    let stale = 0xfff0_0000;
    // SAFETY: all of these take any handle; that's what this test checks.
    unsafe {
        occt_capi_sys::occt_shape_free(stale);
        occt_capi_sys::occt_shape_free(stale);
        assert_eq!(occt_capi_sys::occt_shape_kind(stale), -1);
        assert!(occt_capi_sys::occt_volume(stale).is_nan());
        assert_eq!(occt_capi_sys::occt_cut(stale, stale), 0);
        occt_capi_sys::occt_buffer_free(u32::MAX);
        assert_eq!(occt_capi_sys::occt_buffer_len(u32::MAX), 0);
    }
}
