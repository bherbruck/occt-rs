//! Geometry behaviour that only arises from what this kernel stores.

use std::f64::consts::{FRAC_PI_2, TAU};

use occt_kernel::{
    Axis, Construction, Curve, Frame, Geometry, ParamRange, Point3, Primitives, Properties,
    ShapeKind, Surface, SurfaceKind, Topology, Transforms,
};
use occt_kernel_capi::CapiKernel as K;

/// Negative scaling turns an analytic surface's placement left-handed, which the contract's
/// frames can't state directly. What's reported must still be self-consistent.
#[test]
fn a_mirrored_cylinder_reads_back_consistently() {
    let original = K::make_cylinder(Axis::Z, 2.0, 5.0).unwrap();
    let mirrored = K::scale(&original, Point3::ORIGIN, -1.0).unwrap();
    assert!((K::volume(&mirrored).unwrap() - 20.0 * std::f64::consts::PI).abs() < 1e-6);
    let side = K::subshapes(&mirrored, ShapeKind::Face)
        .unwrap()
        .into_iter()
        .find(|f| K::surface_kind(f).unwrap() == SurfaceKind::Cylinder)
        .unwrap();
    let geometry = K::face_geometry(&side).unwrap();
    let Surface::Cylinder { frame, radius } = geometry.surface else {
        panic!("{:?}", geometry.surface);
    };
    assert!((radius - 2.0).abs() < 1e-9);
    assert!((geometry.u.end - geometry.u.start - TAU).abs() < 1e-9);

    let u = (geometry.u.start + geometry.u.end) / 2.0 + 0.3;
    let v = (geometry.v.start + geometry.v.end) / 2.0;
    let at = K::face_point(&side, u, v).unwrap();

    // The description, evaluated by hand, is where the face says this point is.
    let (x, y, z) = (frame.x.vector(), frame.y().vector(), frame.z.vector());
    let by_hand = frame.origin + x * (radius * u.cos()) + y * (radius * u.sin()) + z * v;
    assert!(
        at.point.distance(by_hand) < 1e-9,
        "{:?} {by_hand:?}",
        at.point
    );

    // dU agrees with the description too, and the normal still points out of the solid.
    let du_by_hand = y * (radius * u.cos()) - x * (radius * u.sin());
    assert!(
        (at.du - du_by_hand).length() < 1e-9,
        "{:?} {du_by_hand:?}",
        at.du
    );
    let normal = at.normal.unwrap().vector();
    let out = at.point - Point3::new(0.0, 0.0, at.point.z);
    assert!(normal.dot(out) > 0.0, "the normal points in");

    // The natural normal of the description, flipped per `reversed`, is that normal too.
    let natural = du_by_hand.cross(z);
    let natural = if geometry.reversed { -natural } else { natural };
    assert!(
        natural.dot(normal) > 0.0,
        "`reversed` disagrees with the normal"
    );

    let (pu, pv, foot) = K::project_onto_face(&side, at.point + normal).unwrap();
    assert!(foot.distance(at.point) < 1e-6);
    assert!(K::face_point(&side, pu, pv).unwrap().point.distance(foot) < 1e-6);
    assert!((pu - u).abs() < 1e-6, "{pu} {u}");
}

/// A trimmed basis is what shows: the circle, and the part of it the edge uses.
#[test]
fn an_arc_reports_its_circle_and_range() {
    let circle = Curve::Circle {
        frame: Frame::WORLD,
        radius: 2.0,
    };
    let range = ParamRange {
        start: 0.25,
        end: FRAC_PI_2,
    };
    let edge = K::make_edge(&circle, Some(range)).unwrap();
    let moved = K::translate(&edge, [1.0, 2.0, 3.0].into()).unwrap();
    let geometry = K::edge_geometry(&moved).unwrap();
    let Curve::Circle { frame, radius } = geometry.curve else {
        panic!("{:?}", geometry.curve);
    };
    assert!((radius - 2.0).abs() < 1e-9);
    assert!(frame.origin.distance(Point3::new(1.0, 2.0, 3.0)) < 1e-9);
    assert_eq!(geometry.range, range);
}

#[test]
fn a_shape_handle_of_the_wrong_kind_never_crashes_the_module() {
    let cube = K::make_box(Point3::ORIGIN, [1.0, 1.0, 1.0].into()).unwrap();
    assert!(K::make_wire(&[&cube]).is_err());
    assert!(K::make_solid(&cube).is_err());
    assert!(K::is_closed(&cube).is_err());
    assert!(K::tolerance(&cube).is_ok());
}
