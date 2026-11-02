//! Kernel-level behaviour of the operations domain that the public API can't reach,
//! because it rejects these inputs first.

use occt_kernel::{
    AdvancedBlends, BooleanOp, BooleanOps, BooleanOptions, ChamferSize, Point3, Primitives,
    RadiusLaw, ShapeKind, Topology, Vec3,
};
use occt_kernel_capi::CapiKernel as K;

fn cube() -> <K as Topology>::Shape {
    K::make_box(Point3::ORIGIN, Vec3::new(2.0, 2.0, 2.0)).unwrap()
}

#[test]
fn empty_sides_are_errors() {
    let a = cube();
    let options = BooleanOptions::default();
    assert!(K::boolean(BooleanOp::Cut, &[], &[&a], &options).is_err());
    assert!(K::boolean(BooleanOp::Cut, &[&a], &[], &options).is_err());
    assert!(K::split(&[], &[&a], &options).is_err());
    assert!(K::split(&[&a], &[], &options).is_err());
    assert!(K::fillet_edges(&a, &[]).is_err());
    assert!(K::chamfer_edges(&a, &[]).is_err());
}

#[test]
fn laws_out_of_range_are_errors_not_garbage() {
    let a = cube();
    let edge = K::subshapes(&a, ShapeKind::Edge).unwrap().remove(0);
    for law in [
        RadiusLaw::Constant(0.0),
        RadiusLaw::Linear {
            start: 1.0,
            end: -1.0,
        },
        RadiusLaw::Points(vec![]),
        RadiusLaw::Points(vec![(0.0, 1.0)]),
        RadiusLaw::Points(vec![(0.0, 1.0), (1.0, f64::NAN)]),
    ] {
        assert!(
            K::fillet_edges(&a, &[(&edge, law.clone())]).is_err(),
            "{law:?}"
        );
    }
}

#[test]
fn a_failed_blend_reports_the_operation() {
    let a = cube();
    let edge = K::subshapes(&a, ShapeKind::Edge).unwrap().remove(0);
    let error = K::chamfer_edges(&a, &[(&edge, ChamferSize::Equal(100.0), None)]).unwrap_err();
    assert!(error.to_string().contains("chamfer"), "{error}");
}
