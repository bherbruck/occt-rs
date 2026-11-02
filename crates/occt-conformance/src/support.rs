//! Tolerances and small helpers shared by the checks.

use std::f64::consts::PI;
use std::fmt::Display;

use occt_kernel::{Aabb, Kernel, Point3, Result, ShapeKind};

/// Relative tolerance for a measurement of exactly representable geometry.
///
/// Modelling kernels conventionally treat points closer than about 1e-7 model units
/// as coincident. Shifting a surface of area `A` by that much changes a volume by
/// `1e-7 * A`, which for the shapes here (size 1 to 20, so `A / V` at most about 6)
/// is below 1e-6 of the volume. Asking for better would demand more than the
/// geometry is defined to.
pub(crate) const MEASURE: f64 = 1e-6;

/// Absolute tolerance for a computed position, in model units: ten times the
/// conventional coincidence distance, since a position may combine a few of them.
pub(crate) const POSITION: f64 = 1e-6;

/// Relative tolerance for a blend whose surfaces need not be exact.
///
/// Where three blends meet, kernels generally fit an approximating patch rather than
/// an analytic one, and blend construction tolerances are conventionally around
/// 1e-4 of model size, roughly 1e-3 of the volume of the affected regions.
pub(crate) const BLEND: f64 = 1e-3;

/// How far a bounding box of a curved shape may exceed the true extent, relative to
/// its size. The contract says tight, but curved extents are found numerically, and
/// one part in a thousand keeps a box useful for what bounding boxes are for.
pub(crate) const CURVED_BOX: f64 = 1e-3;

/// Unwraps a kernel result, naming the operation in the panic.
#[track_caller]
pub(crate) fn must<T>(result: Result<T>, what: impl Display) -> T {
    result.unwrap_or_else(|e| panic!("{what}: unexpected error: {e}"))
}

/// Asserts that `got` is within a relative tolerance of `want`.
#[track_caller]
pub(crate) fn near(got: f64, want: f64, relative: f64, what: impl Display) {
    assert!(
        (got - want).abs() <= relative * want.abs(),
        "{what}: got {got}, want {want} (relative tolerance {relative})"
    );
}

/// Asserts that two positions agree within [`POSITION`].
#[track_caller]
pub(crate) fn near_point(got: Point3, want: Point3, what: impl Display) {
    assert!(
        got.distance(want) <= POSITION,
        "{what}: got {got:?}, want {want:?}"
    );
}

/// Asserts that a box of straight-sided geometry matches exactly.
#[track_caller]
pub(crate) fn near_box(got: Aabb, min: Point3, max: Point3, what: impl Display) {
    assert!(
        got.min.distance(min) <= POSITION && got.max.distance(max) <= POSITION,
        "{what}: got {got:?}, want min {min:?} max {max:?}"
    );
}

/// A volume check.
#[track_caller]
pub(crate) fn volume_is<K: Kernel>(shape: &K::Shape, want: f64, what: impl Display) {
    near(
        must(K::volume(shape), &what),
        want,
        MEASURE,
        format_args!("{what}: volume"),
    );
}

/// An area check.
#[track_caller]
pub(crate) fn area_is<K: Kernel>(shape: &K::Shape, want: f64, what: impl Display) {
    near(
        must(K::area(shape), &what),
        want,
        MEASURE,
        format_args!("{what}: area"),
    );
}

/// A validity check.
#[track_caller]
pub(crate) fn assert_valid<K: Kernel>(shape: &K::Shape, what: impl Display) {
    assert!(
        must(K::is_valid(shape), &what),
        "{what}: the kernel reports the shape invalid"
    );
}

/// The subshapes of a kind.
#[track_caller]
pub(crate) fn parts<K: Kernel>(shape: &K::Shape, kind: ShapeKind) -> Vec<K::Shape> {
    must(
        K::subshapes(shape, kind),
        format_args!("subshapes of kind {kind}"),
    )
}

/// A box at the origin.
#[track_caller]
pub(crate) fn cuboid<K: Kernel>(x: f64, y: f64, z: f64) -> K::Shape {
    must(
        K::make_box(Point3::ORIGIN, occt_kernel::Vec3::new(x, y, z)),
        "make_box",
    )
}

/// The volume of a ball.
pub(crate) fn ball_volume(radius: f64) -> f64 {
    4.0 / 3.0 * PI * radius.powi(3)
}

/// The area of a sphere.
pub(crate) fn sphere_area(radius: f64) -> f64 {
    4.0 * PI * radius * radius
}
