//! STEP exchange.

use std::f64::consts::PI;

use occt_kernel::{Axis, Dir3, Kernel, Point3, ShapeKind, Vec3};

use crate::support::{area_is, assert_valid, cuboid, must, near_point, parts, volume_is};

/// Writing and re-reading keeps the measures and the number of solids.
///
/// # Panics
/// If the round trip changes the shape.
pub fn step_round_trip_keeps_measures<K: Kernel>() {
    let block = cuboid::<K>(1.0, 2.0, 3.0);
    let back = must(
        K::read_step(&must(K::write_step(&block), "write_step")),
        "read_step",
    );
    assert_valid::<K>(&back, "box from STEP");
    volume_is::<K>(&back, 6.0, "box from STEP");
    area_is::<K>(&back, 22.0, "box from STEP");
    assert_eq!(
        parts::<K>(&back, ShapeKind::Solid).len(),
        1,
        "solids of a box from STEP"
    );
    assert_eq!(
        parts::<K>(&back, ShapeKind::Face).len(),
        6,
        "faces of a box from STEP"
    );

    let (width, depth, thick, radius) = (10.0, 8.0, 2.0, 1.5);
    let plate = cuboid::<K>(width, depth, thick);
    let axis = Axis::new(Point3::new(5.0, 4.0, -1.0), Dir3::Z);
    let drill = must(K::make_cylinder(axis, radius, thick + 2.0), "make_cylinder");
    let holed = must(K::cut(&plate, &drill), "cut");
    let back = must(
        K::read_step(&must(K::write_step(&holed), "write_step")),
        "read_step",
    );
    assert_valid::<K>(&back, "plate from STEP");
    volume_is::<K>(
        &back,
        width * depth * thick - PI * radius * radius * thick,
        "plate from STEP",
    );
}

/// A shape comes back where it was, including after a transform.
///
/// # Panics
/// If the round trip moves the shape.
pub fn step_round_trip_keeps_placement<K: Kernel>() {
    let by = Vec3::new(10.0, 20.0, 30.0);
    let moved = must(K::translate(&cuboid::<K>(2.0, 2.0, 2.0), by), "translate");
    let back = must(
        K::read_step(&must(K::write_step(&moved), "write_step")),
        "read_step",
    );
    near_point(
        must(K::center_of_mass(&back), "com"),
        Point3::new(1.0, 1.0, 1.0) + by,
        "centre after a STEP round trip",
    );
}

/// Output is an ISO 10303-21 file, identifiable by its mandatory first token.
///
/// # Panics
/// If the output has no Part 21 header.
pub fn step_output_is_part_21<K: Kernel>() {
    let data = must(K::write_step(&cuboid::<K>(1.0, 1.0, 1.0)), "write_step");
    assert!(
        data.starts_with(b"ISO-10303-21"),
        "STEP output must open with ISO-10303-21"
    );
}

/// Bytes that are not STEP return an error.
///
/// # Panics
/// If the kernel returns a shape, or panics.
pub fn unparseable_step_is_rejected<K: Kernel>() {
    for (name, data) in [
        ("empty input", &b""[..]),
        ("arbitrary text", &b"this is not a STEP file"[..]),
        ("arbitrary bytes", &[0xff, 0x00, 0xfe, 0x01, 0x80, 0x7f][..]),
        (
            "a header and nothing else",
            &b"ISO-10303-21;\nHEADER;\n"[..],
        ),
    ] {
        assert!(
            K::read_step(data).is_err(),
            "read_step of {name} must return Err"
        );
    }
}
