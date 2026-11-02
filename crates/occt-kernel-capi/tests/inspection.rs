//! What the module does with parameters and identity, which the generic specification
//! doesn't pin down: this kernel's surfaces have known parameterisations.

use std::f64::consts::{FRAC_PI_2, TAU};

use occt_kernel::{
    Axis, Inspection, MeshOptions, Point3, Primitives, Properties, ShapeKind, Topology, Vec3,
};
use occt_kernel_capi::CapiKernel as K;

#[test]
fn sphere_uvs_are_longitude_and_latitude() {
    let r = 2.0;
    let ball = K::make_sphere(Point3::ORIGIN, r).unwrap();
    let t = K::tessellate(&ball, &MeshOptions::default().absolute(0.05), None).unwrap();
    assert_eq!(t.faces.len(), 1);
    let part = &t.faces[0];
    assert!(part.mesh.positions.len() > 10);
    for (&[u, v], p) in part.uvs.iter().zip(&part.mesh.positions) {
        assert!((-1e-9..=TAU + 1e-9).contains(&u), "u = {u}");
        assert!(
            (-FRAC_PI_2 - 1e-9..=FRAC_PI_2 + 1e-9).contains(&v),
            "v = {v}"
        );
        let want = Point3::new(r * v.cos() * u.cos(), r * v.cos() * u.sin(), r * v.sin());
        assert!(want.distance(*p) < 1e-6, "uv ({u}, {v}): {p:?} vs {want:?}");
    }
}

#[test]
fn cylinder_uvs_are_angle_and_height() {
    let (r, h) = (2.0, 5.0);
    let c = K::make_cylinder(Axis::Z, r, h).unwrap();
    let t = K::tessellate(&c, &MeshOptions::default().absolute(0.05), None).unwrap();
    let faces = K::subshapes(&c, ShapeKind::Face).unwrap();
    let side = faces
        .iter()
        .position(|f| K::surface_kind(f).unwrap() == occt_kernel::SurfaceKind::Cylinder)
        .unwrap();
    for (&[u, v], p) in t.faces[side].uvs.iter().zip(&t.faces[side].mesh.positions) {
        let want = Point3::new(r * u.cos(), r * u.sin(), v);
        assert!(want.distance(*p) < 1e-6, "uv ({u}, {v}): {p:?}");
    }
}

#[test]
fn ray_uv_evaluates_to_the_hit() {
    let r = 2.0;
    let ball = K::make_sphere(Point3::ORIGIN, r).unwrap();
    let direction = occt_kernel::Dir3::new(Vec3::new(-1.0, -0.5, -0.25)).unwrap();
    let origin = Point3::new(5.0, 2.5, 1.25);
    let hits = K::ray_cast(&ball, origin, direction, None).unwrap();
    assert_eq!(hits.len(), 2);
    for hit in &hits {
        let (u, v) = hit.uv;
        let want = Point3::new(r * v.cos() * u.cos(), r * v.cos() * u.sin(), r * v.sin());
        assert!(want.distance(hit.point) < 1e-6, "{hit:?} vs {want:?}");
    }
}

#[test]
fn a_ray_on_the_seam_is_one_crossing() {
    // The seam is at u = 0, along +X. A ray straight into it could hit at u = 0 and at
    // u = 2 pi, which are the same place.
    let ball = K::make_sphere(Point3::ORIGIN, 1.0).unwrap();
    let direction = occt_kernel::Dir3::new(Vec3::new(-1.0, 0.0, 0.0)).unwrap();
    let hits = K::ray_cast(&ball, Point3::new(3.0, 0.0, 0.0), direction, None).unwrap();
    assert_eq!(hits.len(), 2, "{hits:?}");
}

#[test]
fn stale_handles_are_errors_after_a_failed_query() {
    // Errors clear on the next call instead of sticking.
    let b = K::make_box(Point3::ORIGIN, Vec3::new(1.0, 1.0, 1.0)).unwrap();
    let vertex = K::subshapes(&b, ShapeKind::Vertex).unwrap().remove(0);
    assert!(K::mass_properties(&vertex).is_err());
    assert!(K::mass_properties(&b).is_ok());
}
