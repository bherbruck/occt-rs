//! Tessellation.

use std::collections::HashMap;
use std::f64::consts::PI;

use occt_kernel::{Angle, Axis, Dir3, Kernel, Mesh, MeshOptions, Point3};

use crate::support::{MEASURE, POSITION, ball_volume, cuboid, must, near, sphere_area};

/// Coordinates closer than this are one mesh vertex when testing for closure: the
/// position tolerance, since faces mesh separately and duplicate their shared nodes.
const WELD: f64 = POSITION;

/// The closed solids the mesh checks run over, by name.
fn sample_solids<K: Kernel>() -> Vec<(&'static str, K::Shape)> {
    let plate = cuboid::<K>(10.0, 8.0, 2.0);
    let axis = Axis::new(Point3::new(5.0, 4.0, -1.0), Dir3::Z);
    let drill = must(K::make_cylinder(axis, 1.5, 4.0), "make_cylinder");
    vec![
        ("box", cuboid::<K>(1.0, 2.0, 3.0)),
        (
            "cylinder",
            must(K::make_cylinder(Axis::Z, 2.0, 5.0), "make_cylinder"),
        ),
        (
            "cone",
            must(K::make_cone(Axis::Z, 2.0, 0.0, 5.0), "make_cone"),
        ),
        (
            "sphere",
            must(
                K::make_sphere(Point3::new(1.0, 2.0, 3.0), 4.0),
                "make_sphere",
            ),
        ),
        ("plate with a hole", must(K::cut(&plate, &drill), "cut")),
    ]
}

fn fine() -> MeshOptions {
    MeshOptions::default()
        .absolute(0.01)
        .angular(Angle::degrees(10.0))
}

fn welded(p: Point3) -> [i64; 3] {
    // Rounding to a grid, with `+ 0.0` so that -0.0 and 0.0 fall on the same cell.
    #[allow(clippy::cast_possible_truncation)]
    [p.x, p.y, p.z].map(|c| ((c / WELD).round() + 0.0) as i64)
}

/// Directed edges without a matching reverse edge, after welding coincident vertices.
fn open_edges(mesh: &Mesh) -> usize {
    let mut balance: HashMap<([i64; 3], [i64; 3]), i64> = HashMap::new();
    for triangle in mesh.triangles() {
        let [a, b, c] = triangle.map(welded);
        for (from, to) in [(a, b), (b, c), (c, a)] {
            if from != to {
                *balance.entry((from, to)).or_default() += 1;
                *balance.entry((to, from)).or_default() -= 1;
            }
        }
    }
    balance.values().filter(|n| **n != 0).count() / 2
}

/// Normals parallel positions, are unit length, and indices are in bounds.
///
/// # Panics
/// If the mesh is malformed.
pub fn mesh_is_well_formed<K: Kernel>() {
    for (name, shape) in sample_solids::<K>() {
        let mesh = must(K::mesh(&shape, &fine()), name);
        assert!(!mesh.is_empty(), "{name}: mesh has no triangles");
        assert_eq!(
            mesh.normals.len(),
            mesh.positions.len(),
            "{name}: one normal per position"
        );
        let count = u32::try_from(mesh.positions.len()).expect("mesh fits u32 indices");
        assert!(
            mesh.indices.iter().flatten().all(|i| *i < count),
            "{name}: index out of bounds"
        );
        for (i, normal) in mesh.normals.iter().enumerate() {
            near(
                normal.length(),
                1.0,
                MEASURE,
                format_args!("{name}: length of normal {i}"),
            );
        }
    }
}

/// A box has only planar faces, so its mesh is exact whatever the deflection.
///
/// # Panics
/// If the mesh differs from the box.
pub fn mesh_box_is_exact<K: Kernel>() {
    let (x, y, z) = (1.0, 2.0, 3.0);
    let b = cuboid::<K>(x, y, z);
    for options in [
        MeshOptions::default(),
        MeshOptions::default().absolute(0.5),
        fine(),
    ] {
        let mesh = must(K::mesh(&b, &options), "mesh");
        near(
            mesh.volume(),
            x * y * z,
            MEASURE,
            format_args!("box mesh volume, {options:?}"),
        );
        near(
            mesh.area(),
            2.0 * (x * y + y * z + z * x),
            MEASURE,
            "box mesh area",
        );
    }
}

/// If no point of the mesh is further than `d` from the surface of area `A`, the volume
/// between them is at most about `A * d`. A mesh that honours its deflection therefore
/// has a volume within that of the true one. Its triangles also stay within the angular
/// deflection of the surface, so its area is at least `cos(angle)` of the true area.
///
/// # Panics
/// If the mesh is further off than its deflection permits.
pub fn mesh_curved_solid_is_within_deflection<K: Kernel>() {
    let r = 4.0;
    let d = 0.01;
    let center = Point3::new(1.0, 2.0, 3.0);
    let sphere = must(K::make_sphere(center, r), "make_sphere");
    let options = MeshOptions::default().absolute(d);
    let mesh = must(K::mesh(&sphere, &options), "mesh");
    let bound = sphere_area(r) * d;
    let error = (mesh.volume() - ball_volume(r)).abs();
    assert!(
        error <= bound,
        "sphere mesh volume is off by {error}, more than {bound}"
    );
    let error = (mesh.area() - sphere_area(r)).abs();
    let bound = sphere_area(r) * (1.0 - options.angular_deflection.as_radians().cos());
    assert!(
        error <= bound,
        "sphere mesh area is off by {error}, more than {bound}"
    );

    let (rc, h) = (2.0, 5.0);
    let area = 2.0 * PI * rc * (rc + h);
    let cylinder = must(K::make_cylinder(Axis::Z, rc, h), "make_cylinder");
    let mesh = must(K::mesh(&cylinder, &options), "mesh");
    let error = (mesh.volume() - PI * rc * rc * h).abs();
    let bound = area * d;
    assert!(
        error <= bound,
        "cylinder mesh volume is off by {error}, more than {bound}"
    );
}

/// A mesh of a closed solid has no boundary: every edge is walked once each way.
///
/// # Panics
/// If a mesh has open edges.
pub fn mesh_is_watertight<K: Kernel>() {
    for (name, shape) in sample_solids::<K>() {
        let mesh = must(K::mesh(&shape, &MeshOptions::default()), name);
        let open = open_edges(&mesh);
        assert_eq!(
            open,
            0,
            "{name}: mesh has {open} open edges of {}",
            mesh.indices.len() * 3
        );
    }
}

/// Triangles are counter-clockwise from outside: the signed volume is positive, and on
/// a convex solid each triangle faces away from the centre and agrees with its normals.
///
/// # Panics
/// If a triangle winds inward.
pub fn mesh_winds_outward<K: Kernel>() {
    for (name, shape) in sample_solids::<K>() {
        let mesh = must(K::mesh(&shape, &fine()), name);
        assert!(
            mesh.volume() > 0.0,
            "{name}: signed volume {} is not positive",
            mesh.volume()
        );
        for (i, indices) in mesh.indices.iter().enumerate() {
            let [a, b, c] = indices.map(|n| mesh.positions[n as usize]);
            let facing = (b - a).cross(c - a);
            if facing.length() == 0.0 {
                continue;
            }
            for n in indices {
                let along = facing.dot(mesh.normals[*n as usize]);
                assert!(
                    along > 0.0,
                    "{name}: triangle {i} opposes the normal at vertex {n}"
                );
            }
        }
    }

    let center = Point3::new(1.0, 2.0, 3.0);
    let sphere = must(K::make_sphere(center, 4.0), "make_sphere");
    let mesh = must(K::mesh(&sphere, &fine()), "mesh");
    for (i, [a, b, c]) in mesh.triangles().enumerate() {
        let outward = (b - a).cross(c - a).dot(a - center);
        assert!(outward >= 0.0, "sphere: triangle {i} winds inward");
    }
}

/// Asking for a closer fit costs more triangles.
///
/// # Panics
/// If a finer tolerance does not refine the mesh.
pub fn finer_deflection_gives_more_triangles<K: Kernel>() {
    // Separate shapes, and a wide angular limit so that only the distance decides.
    let count = |deflection: f64| {
        let sphere = must(K::make_sphere(Point3::ORIGIN, 10.0), "make_sphere");
        let options = MeshOptions::default()
            .absolute(deflection)
            .angular(Angle::degrees(80.0));
        must(K::mesh(&sphere, &options), "mesh").indices.len()
    };
    let (coarse, fine) = (count(1.0), count(0.01));
    assert!(
        fine > coarse,
        "deflection 0.01 gave {fine} triangles, deflection 1.0 gave {coarse}"
    );
}
