//! Boolean operations, against volume arithmetic.

use std::f64::consts::PI;

use occt_kernel::{Axis, Dir3, Kernel, Point3, ShapeKind, Vec3};

use crate::support::{
    MEASURE, area_is, assert_valid, ball_volume, cuboid, must, parts, sphere_area, volume_is,
};

/// Two 2-unit cubes overlapping in a unit cube: 15 united, 7 left after a cut, 1 shared.
///
/// # Panics
/// If the arithmetic differs.
pub fn fuse_cut_common_of_overlapping_boxes<K: Kernel>() {
    let size = Vec3::new(2.0, 2.0, 2.0);
    let a = must(K::make_box(Point3::ORIGIN, size), "make_box");
    let b = must(K::make_box(Point3::new(1.0, 1.0, 1.0), size), "make_box");

    let fused = must(K::fuse(&a, &b), "fuse");
    assert_valid::<K>(&fused, "fuse");
    volume_is::<K>(&fused, 15.0, "fuse");

    let cut = must(K::cut(&a, &b), "cut");
    assert_valid::<K>(&cut, "cut");
    volume_is::<K>(&cut, 7.0, "cut");

    let common = must(K::common(&a, &b), "common");
    assert_valid::<K>(&common, "common");
    volume_is::<K>(&common, 1.0, "common");
}

/// A cylinder through a plate removes its own cross section from the plate, leaving
/// one solid.
///
/// # Panics
/// If the volume or the solid count differs.
pub fn cut_cylinder_through_box<K: Kernel>() {
    let (w, d, t, r) = (10.0, 8.0, 2.0, 1.5);
    let plate = cuboid::<K>(w, d, t);
    let axis = Axis::new(Point3::new(w / 2.0, d / 2.0, -1.0), Dir3::Z);
    let drill = must(K::make_cylinder(axis, r, t + 2.0), "make_cylinder");
    let holed = must(K::cut(&plate, &drill), "cut");
    assert_valid::<K>(&holed, "plate with a hole");
    volume_is::<K>(&holed, w * d * t - PI * r * r * t, "plate with a hole");
    let area = 2.0 * (w * d + w * t + d * t) - 2.0 * PI * r * r + 2.0 * PI * r * t;
    area_is::<K>(&holed, area, "plate with a hole");
    assert_eq!(
        parts::<K>(&holed, ShapeKind::Solid).len(),
        1,
        "one plate, one solid"
    );
}

/// Cutting a ball out of the middle of a box makes a cavity: the volume drops and the
/// inner surface counts towards the area.
///
/// # Panics
/// If a measure differs.
pub fn cut_cavity_keeps_both_surfaces<K: Kernel>() {
    let (side, r) = (10.0, 2.0);
    let b = cuboid::<K>(side, side, side);
    let ball = must(K::make_sphere(Point3::new(5.0, 5.0, 5.0), r), "make_sphere");
    let hollow = must(K::cut(&b, &ball), "cut");
    assert_valid::<K>(&hollow, "hollow box");
    volume_is::<K>(&hollow, side * side * side - ball_volume(r), "hollow box");
    area_is::<K>(&hollow, 6.0 * side * side + sphere_area(r), "hollow box");
}

/// A box covering exactly the upper half of a ball leaves a hemisphere.
///
/// # Panics
/// If a measure differs.
pub fn common_of_sphere_and_box_is_a_hemisphere<K: Kernel>() {
    let r = 3.0;
    let ball = must(K::make_sphere(Point3::ORIGIN, r), "make_sphere");
    let upper = must(
        K::make_box(Point3::new(-10.0, -10.0, 0.0), Vec3::new(20.0, 20.0, 10.0)),
        "make_box",
    );
    let half = must(K::common(&ball, &upper), "common");
    assert_valid::<K>(&half, "hemisphere");
    volume_is::<K>(&half, ball_volume(r) / 2.0, "hemisphere");
    area_is::<K>(&half, sphere_area(r) / 2.0 + PI * r * r, "hemisphere");
}

/// Fusing unit cubes that share a face gives one solid with the shared face gone:
/// a 2x1x1 block's surface area, however the kernel splits its faces.
///
/// # Panics
/// If a measure differs.
pub fn fuse_of_touching_boxes_removes_the_shared_face<K: Kernel>() {
    let unit = Vec3::new(1.0, 1.0, 1.0);
    let a = must(K::make_box(Point3::ORIGIN, unit), "make_box");
    let b = must(K::make_box(Point3::new(1.0, 0.0, 0.0), unit), "make_box");
    let block = must(K::fuse(&a, &b), "fuse");
    assert_valid::<K>(&block, "block");
    volume_is::<K>(&block, 2.0, "block");
    area_is::<K>(&block, 10.0, "block");
    assert_eq!(
        parts::<K>(&block, ShapeKind::Solid).len(),
        1,
        "touching boxes fuse into one solid"
    );
}

/// Disjoint shapes share nothing. Whether a kernel represents that as an empty shape
/// or as an error is its choice; what it may not do is report a volume.
///
/// # Panics
/// If the common part of disjoint shapes has volume.
pub fn common_of_disjoint_shapes_has_no_volume<K: Kernel>() {
    let unit = Vec3::new(1.0, 1.0, 1.0);
    let a = must(K::make_box(Point3::ORIGIN, unit), "make_box");
    let b = must(K::make_box(Point3::new(5.0, 5.0, 5.0), unit), "make_box");
    if let Ok(none) = K::common(&a, &b) {
        if let Ok(volume) = K::volume(&none) {
            assert!(
                volume.abs() <= MEASURE,
                "disjoint shapes share a volume of {volume}"
            );
        }
    }
}

/// A boolean leaves its operands as they were.
///
/// # Panics
/// If an operand changes.
pub fn booleans_leave_operands_untouched<K: Kernel>() {
    let size = Vec3::new(2.0, 2.0, 2.0);
    let a = must(K::make_box(Point3::ORIGIN, size), "make_box");
    let b = must(K::make_box(Point3::new(1.0, 1.0, 1.0), size), "make_box");
    must(K::fuse(&a, &b), "fuse");
    must(K::cut(&a, &b), "cut");
    must(K::common(&a, &b), "common");
    for (name, operand) in [("first", &a), ("second", &b)] {
        volume_is::<K>(operand, 8.0, format_args!("{name} operand"));
        assert_eq!(
            parts::<K>(operand, ShapeKind::Face).len(),
            6,
            "{name} operand faces"
        );
    }
}
