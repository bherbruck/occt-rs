//! Direct construction, against closed-form measures.

use std::f64::consts::PI;

use occt_kernel::{Axis, Dir3, Kernel, Point3, ShapeKind, Vec3};

use crate::support::{
    CURVED_BOX, MEASURE, POSITION, area_is, assert_valid, ball_volume, must, near, near_box,
    near_point, sphere_area, volume_is,
};

/// A kernel names itself, reports a version and is ready to run.
///
/// # Panics
/// If it does not.
pub fn kernel_identifies_itself<K: Kernel>() {
    must(K::ready(), "ready");
    assert!(!K::NAME.is_empty(), "the kernel name is empty");
    assert!(
        !must(K::version(), "version").is_empty(),
        "the kernel version is empty"
    );
}

/// A box has the volume, area, centre and bounds its corner and size imply.
///
/// # Panics
/// If any measure differs.
pub fn box_measures<K: Kernel>() {
    let corner = Point3::new(1.0, 2.0, 3.0);
    let b = must(K::make_box(corner, Vec3::new(2.0, 3.0, 4.0)), "make_box");
    assert_eq!(
        must(K::kind(&b), "kind"),
        ShapeKind::Solid,
        "a box is a solid"
    );
    assert_valid::<K>(&b, "box");
    volume_is::<K>(&b, 24.0, "box");
    area_is::<K>(&b, 2.0 * (6.0 + 12.0 + 8.0), "box");
    near_point(
        must(K::center_of_mass(&b), "com"),
        Point3::new(2.0, 3.5, 5.0),
        "box centre",
    );
    near_box(
        must(K::bounding_box(&b), "bbox"),
        corner,
        Point3::new(3.0, 5.0, 7.0),
        "box bounds",
    );
}

/// A cylinder runs from the axis origin along its direction, on any axis.
///
/// # Panics
/// If any measure differs.
pub fn cylinder_measures<K: Kernel>() {
    let origin = Point3::new(1.0, 2.0, 3.0);
    let dir = Dir3::new([1.0, 2.0, 2.0]).expect("nonzero");
    let (r, h) = (2.0, 6.0);
    let c = must(
        K::make_cylinder(Axis::new(origin, dir), r, h),
        "make_cylinder",
    );
    assert_eq!(
        must(K::kind(&c), "kind"),
        ShapeKind::Solid,
        "a cylinder is a solid"
    );
    assert_valid::<K>(&c, "cylinder");
    volume_is::<K>(&c, PI * r * r * h, "cylinder");
    area_is::<K>(&c, 2.0 * PI * r * (r + h), "cylinder");
    let centre = origin + dir.vector() * (h / 2.0);
    near_point(
        must(K::center_of_mass(&c), "com"),
        centre,
        "cylinder centre",
    );
}

/// A cone has `radius1` at the axis origin and `radius2` at `height`; either may be
/// zero. A cone's centroid is a quarter of the way up from its base.
///
/// # Panics
/// If any measure differs.
pub fn cone_measures<K: Kernel>() {
    let (r, h, slant) = (3.0, 4.0, 5.0);
    let volume = PI * r * r * h / 3.0;
    let area = PI * r * (r + slant);

    let base_down = must(K::make_cone(Axis::Z, r, 0.0, h), "make_cone");
    assert_valid::<K>(&base_down, "cone");
    volume_is::<K>(&base_down, volume, "cone");
    area_is::<K>(&base_down, area, "cone");
    near_point(
        must(K::center_of_mass(&base_down), "com"),
        Point3::new(0.0, 0.0, h / 4.0),
        "cone centre: radius1 belongs at the axis origin",
    );

    let base_up = must(K::make_cone(Axis::Z, 0.0, r, h), "make_cone");
    volume_is::<K>(&base_up, volume, "inverted cone");
    near_point(
        must(K::center_of_mass(&base_up), "com"),
        Point3::new(0.0, 0.0, 3.0 * h / 4.0),
        "inverted cone centre: radius2 belongs at the far end",
    );
}

/// A frustum matches the closed forms.
///
/// # Panics
/// If any measure differs.
pub fn frustum_measures<K: Kernel>() {
    let (r1, r2, h) = (3.0_f64, 1.0_f64, 4.0_f64);
    let slant = (h * h + (r1 - r2).powi(2)).sqrt();
    let f = must(K::make_cone(Axis::Z, r1, r2, h), "make_cone");
    assert_valid::<K>(&f, "frustum");
    volume_is::<K>(&f, PI * h / 3.0 * (r1 * r1 + r1 * r2 + r2 * r2), "frustum");
    area_is::<K>(
        &f,
        PI * (r1 + r2) * slant + PI * (r1 * r1 + r2 * r2),
        "frustum",
    );
}

/// A sphere matches the closed forms, centred where asked, with tight bounds.
///
/// # Panics
/// If any measure differs.
pub fn sphere_measures<K: Kernel>() {
    let (center, r) = (Point3::new(-1.0, 5.0, 2.0), 3.0);
    let s = must(K::make_sphere(center, r), "make_sphere");
    assert_eq!(
        must(K::kind(&s), "kind"),
        ShapeKind::Solid,
        "a sphere is a solid"
    );
    assert_valid::<K>(&s, "sphere");
    volume_is::<K>(&s, ball_volume(r), "sphere");
    area_is::<K>(&s, sphere_area(r), "sphere");
    near_point(must(K::center_of_mass(&s), "com"), center, "sphere centre");

    let bounds = must(K::bounding_box(&s), "bbox");
    let size = bounds.size();
    for (axis, extent, low, want_low) in [
        ("x", size.x, bounds.min.x, center.x - r),
        ("y", size.y, bounds.min.y, center.y - r),
        ("z", size.z, bounds.min.z, center.z - r),
    ] {
        assert!(
            low <= want_low + POSITION,
            "sphere bounds exclude the sphere along {axis}"
        );
        near(
            extent,
            2.0 * r,
            CURVED_BOX,
            format_args!("sphere bounds along {axis}"),
        );
    }
}

/// A closed polygon makes a planar face: a surface, enclosing no volume.
///
/// # Panics
/// If any measure differs.
pub fn polygon_face_measures<K: Kernel>() {
    let z = 2.0;
    let triangle = [
        Point3::new(0.0, 0.0, z),
        Point3::new(3.0, 0.0, z),
        Point3::new(0.0, 4.0, z),
    ];
    let f = must(K::make_polygon_face(&triangle), "make_polygon_face");
    assert_eq!(
        must(K::kind(&f), "kind"),
        ShapeKind::Face,
        "a polygon makes a face"
    );
    assert_valid::<K>(&f, "polygon");
    area_is::<K>(&f, 6.0, "3-4-5 triangle");
    let volume = must(K::volume(&f), "polygon volume");
    assert!(
        volume.abs() <= MEASURE,
        "a face encloses no volume, got {volume}"
    );
    near_point(
        must(K::center_of_mass(&f), "com"),
        Point3::new(1.0, 4.0 / 3.0, z),
        "with no volume the centre of mass is the surface's",
    );
}

/// Points that do not lie in one plane cannot bound a planar face.
///
/// # Panics
/// If the kernel makes a face anyway.
pub fn non_planar_polygon_is_rejected<K: Kernel>() {
    let points = [
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(1.0, 1.0, 1.0),
        Point3::new(0.0, 1.0, 0.0),
    ];
    assert!(
        K::make_polygon_face(&points).is_err(),
        "a non-planar polygon must return Err"
    );
}
