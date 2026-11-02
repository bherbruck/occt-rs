//! Fillets and chamfers.
//!
//! A blend on one edge of a box removes a prism of known cross section along that
//! edge: `(1 - PI/4) r^2` for a fillet and `d^2 / 2` for a chamfer.

use std::f64::consts::PI;

use occt_kernel::{Kernel, Point3, ShapeKind};

use crate::support::{BLEND, MEASURE, assert_valid, cuboid, must, near, parts, volume_is};

const SIDES: [f64; 3] = [10.0, 12.0, 14.0];

/// Rounding one edge of a box removes the corner prism.
///
/// # Panics
/// If the volume differs.
pub fn fillet_single_edge<K: Kernel>() {
    let [x, y, z] = SIDES;
    let block = cuboid::<K>(x, y, z);
    let edge = parts::<K>(&block, ShapeKind::Edge).remove(0);
    let length = must(K::length(&edge), "edge length");
    let r = 2.0;
    let rounded = must(K::fillet(&block, &[&edge], r), "fillet");
    assert_valid::<K>(&rounded, "filleted box");
    volume_is::<K>(
        &rounded,
        x * y * z - (1.0 - PI / 4.0) * r * r * length,
        "filleted box",
    );
    assert!(
        must(K::area(&rounded), "area") < must(K::area(&block), "area"),
        "a fillet on a convex edge shortens the surface"
    );
    volume_is::<K>(&block, x * y * z, "the box after filleting a copy");
}

/// Bevelling one edge of a box removes a triangular prism.
///
/// # Panics
/// If the volume differs.
pub fn chamfer_single_edge<K: Kernel>() {
    let [x, y, z] = SIDES;
    let block = cuboid::<K>(x, y, z);
    let edge = parts::<K>(&block, ShapeKind::Edge).remove(0);
    let length = must(K::length(&edge), "edge length");
    let d = 2.0;
    let bevelled = must(K::chamfer(&block, &[&edge], d), "chamfer");
    assert_valid::<K>(&bevelled, "chamfered box");
    volume_is::<K>(&bevelled, x * y * z - d * d / 2.0 * length, "chamfered box");
    assert!(
        must(K::area(&bevelled), "area") < must(K::area(&block), "area"),
        "a chamfer on a convex edge shortens the surface"
    );
}

/// A cube rounded on all edges is the Minkowski sum of a smaller cube and a ball.
/// With inner side `a`, that is `a^3 + 6 a^2 r + 3 PI a r^2 + 4/3 PI r^3`.
///
/// # Panics
/// If the volume differs.
pub fn fillet_all_edges_of_a_cube<K: Kernel>() {
    let (side, r) = (10.0_f64, 2.0_f64);
    let b = cuboid::<K>(side, side, side);
    let edges = parts::<K>(&b, ShapeKind::Edge);
    let refs: Vec<_> = edges.iter().collect();
    let rounded = must(K::fillet(&b, &refs, r), "fillet");
    assert_valid::<K>(&rounded, "rounded cube");
    let a = side - 2.0 * r;
    let want = a.powi(3) + 6.0 * a * a * r + 3.0 * PI * a * r * r + 4.0 / 3.0 * PI * r.powi(3);
    near(
        must(K::volume(&rounded), "volume"),
        want,
        BLEND,
        "rounded cube volume",
    );
    assert!(
        want < side.powi(3) * (1.0 - MEASURE),
        "the formula itself removes material"
    );
}

/// A radius larger than the shape cannot be built, and must say so.
///
/// # Panics
/// If the kernel returns a shape, or panics.
pub fn impossible_fillet_is_rejected<K: Kernel>() {
    let b = must(
        K::make_box(Point3::ORIGIN, [1.0, 1.0, 1.0].into()),
        "make_box",
    );
    let edges = parts::<K>(&b, ShapeKind::Edge);
    let refs: Vec<_> = edges.iter().collect();
    assert!(
        K::fillet(&b, &refs, 10.0).is_err(),
        "radius 10 on a unit cube must return Err"
    );
}
