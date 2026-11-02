//! Traversal and identity.

use std::collections::HashSet;

use occt_kernel::{Axis, CurveKind, Kernel, Point3, ShapeKind, SurfaceKind};

use crate::support::{MEASURE, cuboid, must, near, near_point, parts};

/// A box has 6 faces, 12 edges and 8 vertices.
///
/// # Panics
/// If any count differs.
pub fn box_topology_counts<K: Kernel>() {
    let b = cuboid::<K>(1.0, 2.0, 3.0);
    assert_eq!(parts::<K>(&b, ShapeKind::Face).len(), 6, "faces of a box");
    assert_eq!(parts::<K>(&b, ShapeKind::Edge).len(), 12, "edges of a box");
    assert_eq!(
        parts::<K>(&b, ShapeKind::Vertex).len(),
        8,
        "vertices of a box"
    );
}

/// Subshapes have the requested kind and are pairwise distinct.
///
/// # Panics
/// If a subshape has the wrong kind or appears twice.
pub fn box_subshapes_are_distinct<K: Kernel>() {
    let b = cuboid::<K>(1.0, 2.0, 3.0);
    for kind in [ShapeKind::Face, ShapeKind::Edge, ShapeKind::Vertex] {
        let all = parts::<K>(&b, kind);
        for (i, a) in all.iter().enumerate() {
            assert_eq!(
                must(K::kind(a), "kind"),
                kind,
                "subshape {i} has the wrong kind"
            );
            for c in &all[i + 1..] {
                assert!(
                    !K::is_same(a, c),
                    "two distinct {kind} subshapes compare same"
                );
            }
        }
    }
}

/// Neighbours share topology: each edge of a box borders exactly two faces, and each
/// vertex belongs to exactly three faces. Reached from different faces with different
/// orientations, a shared edge is still the same edge with the same hash.
///
/// # Panics
/// If neighbours do not share subshapes.
pub fn box_topology_is_shared<K: Kernel>() {
    let b = cuboid::<K>(1.0, 2.0, 3.0);
    let faces = parts::<K>(&b, ShapeKind::Face);
    for (kind, each) in [(ShapeKind::Edge, 2), (ShapeKind::Vertex, 3)] {
        for part in parts::<K>(&b, kind) {
            let touching: Vec<_> = faces
                .iter()
                .flat_map(|f| parts::<K>(f, kind))
                .filter(|p| K::is_same(p, &part))
                .collect();
            assert_eq!(
                touching.len(),
                each,
                "a box {kind} must belong to {each} faces"
            );
            for t in touching {
                assert_eq!(
                    K::hash_code(&t),
                    K::hash_code(&part),
                    "same {kind}, different hash"
                );
            }
        }
    }
}

/// The vertices of a box are exactly its corners.
///
/// # Panics
/// If a corner is missing.
pub fn box_vertices_are_its_corners<K: Kernel>() {
    let (x, y, z) = (1.0, 2.0, 3.0);
    let points: Vec<_> = parts::<K>(&cuboid::<K>(x, y, z), ShapeKind::Vertex)
        .iter()
        .map(|v| must(K::vertex_point(v), "vertex_point"))
        .collect();
    for cx in [0.0, x] {
        for cy in [0.0, y] {
            for cz in [0.0, z] {
                let corner = Point3::new(cx, cy, cz);
                assert!(
                    points
                        .iter()
                        .any(|p| p.distance(corner) <= crate::support::POSITION),
                    "no vertex at corner {corner:?}; vertices: {points:?}"
                );
            }
        }
    }
}

/// Box edges are straight, and four run along each axis with that axis's length.
///
/// # Panics
/// If a kind or a length differs.
pub fn box_edges_are_lines_with_expected_lengths<K: Kernel>() {
    let (x, y, z) = (1.0, 2.0, 3.0);
    let edges = parts::<K>(&cuboid::<K>(x, y, z), ShapeKind::Edge);
    for want in [x, y, z] {
        let count = edges
            .iter()
            .filter(|e| {
                assert_eq!(
                    must(K::curve_kind(e), "curve_kind"),
                    CurveKind::Line,
                    "box edge"
                );
                (must(K::length(e), "length") - want).abs() <= MEASURE * want
            })
            .count();
        assert_eq!(count, 4, "edges of length {want}");
    }
}

/// `length` of a whole shape is its total edge length, each edge counted once even
/// though it borders two faces.
///
/// # Panics
/// If the total differs.
pub fn total_length_counts_each_edge_once<K: Kernel>() {
    let (x, y, z) = (1.0, 2.0, 3.0);
    let total = must(K::length(&cuboid::<K>(x, y, z)), "length");
    near(
        total,
        4.0 * (x + y + z),
        MEASURE,
        "total edge length of a box",
    );
}

/// Asking a shape for subshapes of its own kind gives the shape itself: a traversal
/// includes its starting point, so callers need no special case for it.
///
/// # Panics
/// If it does not.
pub fn a_shape_contains_itself<K: Kernel>() {
    let solid = cuboid::<K>(1.0, 1.0, 1.0);
    let face = parts::<K>(&solid, ShapeKind::Face).remove(0);
    let edge = parts::<K>(&solid, ShapeKind::Edge).remove(0);
    let vertex = parts::<K>(&solid, ShapeKind::Vertex).remove(0);
    for (kind, shape) in [
        (ShapeKind::Solid, &solid),
        (ShapeKind::Face, &face),
        (ShapeKind::Edge, &edge),
        (ShapeKind::Vertex, &vertex),
    ] {
        let own = parts::<K>(shape, kind);
        assert_eq!(own.len(), 1, "{kind} subshapes of a {kind}");
        assert!(
            K::is_same(&own[0], shape),
            "the {kind} inside a {kind} is that {kind}"
        );
    }
}

/// Faces and edges report the geometry they were built from.
///
/// # Panics
/// If an expected kind is missing.
pub fn surface_and_curve_kinds<K: Kernel>() {
    let surfaces = |s: &K::Shape| -> Vec<SurfaceKind> {
        parts::<K>(s, ShapeKind::Face)
            .iter()
            .map(|f| must(K::surface_kind(f), "surface_kind"))
            .collect()
    };
    let curves = |s: &K::Shape| -> Vec<CurveKind> {
        parts::<K>(s, ShapeKind::Edge)
            .iter()
            .map(|e| must(K::curve_kind(e), "curve_kind"))
            .collect()
    };

    let b = surfaces(&cuboid::<K>(1.0, 1.0, 1.0));
    assert!(
        b.iter().all(|k| *k == SurfaceKind::Plane),
        "box faces are planes: {b:?}"
    );

    let c = must(K::make_cylinder(Axis::Z, 1.0, 2.0), "make_cylinder");
    let kinds = surfaces(&c);
    assert!(
        kinds.contains(&SurfaceKind::Cylinder),
        "a cylinder has a cylindrical face: {kinds:?}"
    );
    assert!(
        kinds.contains(&SurfaceKind::Plane),
        "a cylinder has planar caps: {kinds:?}"
    );
    let edges = curves(&c);
    assert!(
        edges.contains(&CurveKind::Circle),
        "a cylinder has circular edges: {edges:?}"
    );

    let cone = surfaces(&must(K::make_cone(Axis::Z, 2.0, 1.0, 2.0), "make_cone"));
    assert!(
        cone.contains(&SurfaceKind::Cone),
        "a frustum has a conical face: {cone:?}"
    );

    let sphere = surfaces(&must(K::make_sphere(Point3::ORIGIN, 1.0), "make_sphere"));
    assert!(
        sphere.contains(&SurfaceKind::Sphere),
        "a sphere has a spherical face: {sphere:?}"
    );
}

/// `is_same` is reflexive and symmetric, holds across clones, and agrees with the hash.
///
/// # Panics
/// If identity and hash disagree.
pub fn identity_is_an_equivalence_with_consistent_hash<K: Kernel>() {
    let a = cuboid::<K>(1.0, 1.0, 1.0);
    let clone = a.clone();
    assert!(K::is_same(&a, &a), "is_same must be reflexive");
    assert!(
        K::is_same(&a, &clone) && K::is_same(&clone, &a),
        "a clone is the same shape"
    );
    assert_eq!(
        K::hash_code(&a),
        K::hash_code(&clone),
        "a clone has the same hash"
    );

    let clone_of_clone = clone.clone();
    drop(clone);
    assert!(
        K::is_same(&a, &clone_of_clone),
        "dropping a clone leaves the others usable"
    );
    near(
        must(K::volume(&clone_of_clone), "volume"),
        1.0,
        MEASURE,
        "clone of a dropped clone",
    );
}

/// Traversing twice gives the same subshapes in the same order.
///
/// # Panics
/// If the order or identity changes.
pub fn traversal_is_stable<K: Kernel>() {
    let b = cuboid::<K>(1.0, 2.0, 3.0);
    for kind in [ShapeKind::Face, ShapeKind::Edge, ShapeKind::Vertex] {
        let (first, second) = (parts::<K>(&b, kind), parts::<K>(&b, kind));
        assert_eq!(
            first.len(),
            second.len(),
            "{kind} count changed between traversals"
        );
        for (x, y) in first.iter().zip(&second) {
            assert!(K::is_same(x, y), "{kind} order changed between traversals");
            assert_eq!(K::hash_code(x), K::hash_code(y), "{kind} hash changed");
        }
        let hashes: HashSet<u32> = first.iter().map(K::hash_code).collect();
        assert!(
            hashes.len() > 1 || first.len() <= 1,
            "every {kind} hashes alike"
        );
    }
}

/// Identity is about topology, not geometry: equal boxes built twice differ.
///
/// # Panics
/// If they compare same.
pub fn separately_built_shapes_are_not_same<K: Kernel>() {
    let (a, b) = (cuboid::<K>(1.0, 1.0, 1.0), cuboid::<K>(1.0, 1.0, 1.0));
    assert!(
        !K::is_same(&a, &b),
        "two separately built boxes are different shapes"
    );
    near_point(
        must(K::center_of_mass(&a), "com"),
        must(K::center_of_mass(&b), "com"),
        "same geometry",
    );
}
