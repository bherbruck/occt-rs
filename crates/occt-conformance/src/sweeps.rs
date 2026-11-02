//! Sweeps, against the prism formula and Pappus's theorem.

use std::f64::consts::{PI, TAU};

use occt_kernel::{Angle, Axis, Dir3, Kernel, Point3, ShapeKind, Vec3};

use crate::support::{area_is, assert_valid, must, volume_is};

/// A rectangle in the XZ plane, `x` from `inner` to `outer` and `z` from 0 to `height`.
fn rectangle<K: Kernel>(inner: f64, outer: f64, height: f64) -> K::Shape {
    let points = [
        Point3::new(inner, 0.0, 0.0),
        Point3::new(outer, 0.0, 0.0),
        Point3::new(outer, 0.0, height),
        Point3::new(inner, 0.0, height),
    ];
    must(K::make_polygon_face(&points), "make_polygon_face")
}

/// Extruding a face makes a prism: base area times the sweep's height above the base,
/// however oblique the sweep.
///
/// # Panics
/// If a measure differs.
pub fn extrude_face_along_oblique_vector<K: Kernel>() {
    let square = [
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(2.0, 0.0, 0.0),
        Point3::new(2.0, 2.0, 0.0),
        Point3::new(0.0, 2.0, 0.0),
    ];
    let face = must(K::make_polygon_face(&square), "make_polygon_face");

    let straight = must(K::extrude(&face, Vec3::new(0.0, 0.0, 3.0)), "extrude");
    assert_eq!(
        must(K::kind(&straight), "kind"),
        ShapeKind::Solid,
        "an extruded face is a solid"
    );
    assert_valid::<K>(&straight, "prism");
    volume_is::<K>(&straight, 12.0, "prism");
    area_is::<K>(&straight, 2.0 * 4.0 + 4.0 * 2.0 * 3.0, "prism");

    let oblique = must(K::extrude(&face, Vec3::new(5.0, -1.0, 3.0)), "extrude");
    assert_valid::<K>(&oblique, "oblique prism");
    volume_is::<K>(&oblique, 12.0, "oblique prism");
}

/// Revolving a rectangle a full turn around Z makes a tube.
///
/// # Panics
/// If a measure differs.
pub fn revolve_rectangle_full_turn<K: Kernel>() {
    let (r1, r2, h) = (1.0, 3.0, 2.0);
    let tube = must(
        K::revolve(&rectangle::<K>(r1, r2, h), Axis::Z, Angle::FULL_TURN),
        "revolve",
    );
    assert_eq!(
        must(K::kind(&tube), "kind"),
        ShapeKind::Solid,
        "a revolved face is a solid"
    );
    assert_valid::<K>(&tube, "tube");
    volume_is::<K>(&tube, PI * (r2 * r2 - r1 * r1) * h, "tube");
    let area = 2.0 * PI * (r2 * r2 - r1 * r1) + TAU * (r1 + r2) * h;
    area_is::<K>(&tube, area, "tube");
}

/// By Pappus, a partial revolve scales with its angle, and revolving about a shifted
/// axis changes the path of the centroid.
///
/// # Panics
/// If a measure differs.
pub fn revolve_rectangle_partial_turn<K: Kernel>() {
    let (r1, r2, h) = (1.0, 3.0, 2.0);
    let section = rectangle::<K>(r1, r2, h);
    let quarter = must(
        K::revolve(&section, Axis::Z, Angle::degrees(90.0)),
        "revolve",
    );
    assert_valid::<K>(&quarter, "quarter tube");
    volume_is::<K>(&quarter, PI * (r2 * r2 - r1 * r1) * h / 4.0, "quarter tube");

    // Around the line x = -1, the centroid at x = 2 is 3 from the axis. Area is 4.
    let shifted = Axis::new(Point3::new(-1.0, 0.0, 0.0), Dir3::Z);
    let half = must(
        K::revolve(&section, shifted, Angle::degrees(180.0)),
        "revolve",
    );
    assert_valid::<K>(&half, "half tube about a shifted axis");
    volume_is::<K>(&half, 4.0 * PI * 3.0, "half tube about a shifted axis");
}
