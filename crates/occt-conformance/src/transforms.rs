//! Transforms.

use occt_kernel::{Aabb, Angle, Axis, Dir3, Kernel, Point3, Vec3};

use crate::support::{MEASURE, assert_valid, cuboid, must, near, near_box, near_point, volume_is};

/// Translation shifts position and bounds and nothing else.
///
/// # Panics
/// If a measure differs.
pub fn translate_moves_without_changing_size<K: Kernel>() {
    let b = cuboid::<K>(1.0, 2.0, 3.0);
    let by = Vec3::new(10.0, -20.0, 30.0);
    let moved = must(K::translate(&b, by), "translate");
    assert_valid::<K>(&moved, "moved box");
    volume_is::<K>(&moved, 6.0, "moved box");
    near_point(
        must(K::center_of_mass(&moved), "com"),
        Point3::new(0.5, 1.0, 1.5) + by,
        "moved centre",
    );
    near_box(
        must(K::bounding_box(&moved), "bbox"),
        Point3::ORIGIN + by,
        Point3::new(1.0, 2.0, 3.0) + by,
        "moved bounds",
    );
}

/// A quarter turn, counter-clockwise seen from +Z, around the vertical line through
/// (5, 0). Relative to that line `(x, y)` goes to `(-y, x)`, so the footprint
/// `[0,1] x [0,2]` goes to `[3,5] x [-5,-4]`.
///
/// # Panics
/// If a measure differs.
pub fn rotate_quarter_turn_about_offset_axis<K: Kernel>() {
    let b = cuboid::<K>(1.0, 2.0, 3.0);
    let axis = Axis::new(Point3::new(5.0, 0.0, 0.0), Dir3::Z);
    let turned = must(K::rotate(&b, axis, Angle::degrees(90.0)), "rotate");
    assert_valid::<K>(&turned, "turned box");
    volume_is::<K>(&turned, 6.0, "turned box");
    near_box(
        must(K::bounding_box(&turned), "bbox"),
        Point3::new(3.0, -5.0, 0.0),
        Point3::new(5.0, -4.0, 3.0),
        "turned bounds",
    );
    near_point(
        must(K::center_of_mass(&turned), "com"),
        Point3::new(4.0, -4.5, 1.5),
        "turned centre",
    );
}

/// Scaling by `f` multiplies lengths by `f`, areas by `f^2` and volumes by `f^3`, and
/// leaves the scaling centre where it is.
///
/// # Panics
/// If a measure differs.
pub fn scale_about_a_point<K: Kernel>() {
    let b = cuboid::<K>(1.0, 2.0, 3.0);
    let center = Point3::new(1.0, 1.0, 1.0);
    let scaled = must(K::scale(&b, center, 2.0), "scale");
    assert_valid::<K>(&scaled, "scaled box");
    volume_is::<K>(&scaled, 48.0, "scaled box");
    near(
        must(K::area(&scaled), "area"),
        88.0,
        MEASURE,
        "scaled box area",
    );
    let want = Aabb {
        min: Point3::new(-1.0, -1.0, -1.0),
        max: Point3::new(1.0, 3.0, 5.0),
    };
    near_box(
        must(K::bounding_box(&scaled), "bbox"),
        want.min,
        want.max,
        "scaled bounds",
    );
}

/// A transform makes a new shape and leaves the original where it was.
///
/// # Panics
/// If the two are confused.
pub fn transforms_make_independent_shapes<K: Kernel>() {
    let b = cuboid::<K>(1.0, 1.0, 1.0);
    let moved = must(K::translate(&b, Vec3::new(5.0, 0.0, 0.0)), "translate");
    assert!(!K::is_same(&b, &moved), "a moved copy is a different shape");
    near_point(
        must(K::center_of_mass(&b), "com"),
        Point3::new(0.5, 0.5, 0.5),
        "the original stays put",
    );
    drop(b);
    volume_is::<K>(&moved, 1.0, "the copy after the original is dropped");
}
