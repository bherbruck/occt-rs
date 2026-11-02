//! What the contract can't tell apart: sharing between placed shapes.

use occt_kernel::{Features, Plane, Point3, Primitives, ShapeKind, Topology, Transform, Vec3};
use occt_kernel_capi::CapiKernel as K;

fn block() -> <K as Topology>::Shape {
    K::make_box(Point3::ORIGIN, Vec3::new(1.0, 1.0, 1.0)).unwrap()
}

#[test]
fn placed_shares_its_geometry_with_the_original() {
    let original = block();
    let moved = K::placed(&original, &Transform::translation(Vec3::new(5.0, 0.0, 0.0))).unwrap();
    assert!(K::shares_geometry(&original, &moved).unwrap());
    assert!(
        !K::is_same(&original, &moved),
        "a placed shape is a different instance"
    );

    let faces = |s| K::subshapes(s, ShapeKind::Face).unwrap();
    for (a, b) in faces(&original).iter().zip(&faces(&moved)) {
        assert!(K::shares_geometry(a, b).unwrap(), "faces are shared too");
    }
}

#[test]
fn transformed_copies_rather_than_shares() {
    let original = block();
    let copies = [
        Transform::translation(Vec3::new(5.0, 0.0, 0.0)),
        Transform::mirror(Plane::XY),
        Transform::scale(Point3::ORIGIN, Vec3::new(1.0, 2.0, 3.0)),
    ];
    for transform in &copies {
        let copy = K::transformed(&original, transform).unwrap();
        assert!(!K::shares_geometry(&original, &copy).unwrap());
    }
}

#[test]
fn reversed_shares_and_separate_shapes_do_not() {
    let original = block();
    let flipped = K::reversed(&original).unwrap();
    assert!(K::shares_geometry(&original, &flipped).unwrap());
    assert!(!K::shares_geometry(&original, &block()).unwrap());
}

#[test]
fn the_module_checks_transforms_too() {
    let original = block();
    let mut broken = Transform::IDENTITY;
    broken.rows[0][0] = f64::NAN;
    assert!(K::transformed(&original, &broken).is_err());
    assert!(K::placed(&original, &broken).is_err());
    assert!(
        K::placed(
            &original,
            &Transform::scale(Point3::ORIGIN, Vec3::new(2.0, 2.0, 2.0))
        )
        .is_err()
    );
    assert!(
        K::transformed(
            &original,
            &Transform::scale(Point3::ORIGIN, Vec3::new(0.0, 1.0, 1.0))
        )
        .is_err()
    );
}
