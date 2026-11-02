//! Triangle mesh data and tessellation options.
//!
//! Plain owned data: once a mesh exists it has no tie to the kernel that made it.

#![forbid(unsafe_code)]

use occt_math::{Angle, Point3, Vec3};

pub mod export;

/// How far a tessellation may stray from the exact surface.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Deflection {
    /// A maximum distance in model units.
    Absolute(f64),
    /// A maximum distance as a fraction of each edge's size.
    Relative(f64),
}

/// Tessellation quality.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshOptions {
    /// Maximum distance between the mesh and the surface.
    pub deflection: Deflection,
    /// Maximum angle between adjacent triangles.
    pub angular_deflection: Angle,
}

impl Default for MeshOptions {
    fn default() -> Self {
        Self {
            deflection: Deflection::Relative(0.01),
            angular_deflection: Angle::degrees(20.0),
        }
    }
}

impl MeshOptions {
    /// Sets an absolute linear deflection.
    #[must_use]
    pub fn absolute(mut self, distance: f64) -> Self {
        self.deflection = Deflection::Absolute(distance);
        self
    }

    /// Sets a relative linear deflection.
    #[must_use]
    pub fn relative(mut self, fraction: f64) -> Self {
        self.deflection = Deflection::Relative(fraction);
        self
    }

    /// Sets the angular deflection.
    #[must_use]
    pub fn angular(mut self, angle: Angle) -> Self {
        self.angular_deflection = angle;
        self
    }
}

/// An indexed triangle mesh with one normal per vertex and counter-clockwise
/// triangles seen from outside the solid.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Mesh {
    /// Vertex positions.
    pub positions: Vec<Point3>,
    /// Unit normals, parallel to `positions`.
    pub normals: Vec<Vec3>,
    /// Triangles as indices into `positions`.
    pub indices: Vec<[u32; 3]>,
}

impl Mesh {
    /// Whether the mesh has no triangles.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Iterates over triangles as corner positions.
    ///
    /// # Panics
    /// If an index is out of bounds, which a kernel-produced mesh never has.
    pub fn triangles(&self) -> impl ExactSizeIterator<Item = [Point3; 3]> + '_ {
        self.indices
            .iter()
            .map(|t| t.map(|i| self.positions[i as usize]))
    }

    /// Total triangle area.
    #[must_use]
    pub fn area(&self) -> f64 {
        self.triangles()
            .map(|[a, b, c]| (b - a).cross(c - a).length() / 2.0)
            .sum()
    }

    /// Enclosed signed volume; positive for a closed outward-facing mesh.
    #[must_use]
    pub fn volume(&self) -> f64 {
        self.triangles()
            .map(|[a, b, c]| {
                let (a, b, c) = (a - Point3::ORIGIN, b - Point3::ORIGIN, c - Point3::ORIGIN);
                a.dot(b.cross(c)) / 6.0
            })
            .sum()
    }
}

/// The triangles of one face.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FaceMesh {
    /// The triangles, with outward winding and normals.
    pub mesh: Mesh,
    /// Surface parameters of each vertex, parallel to `mesh.positions`.
    pub uvs: Vec<[f64; 2]>,
}

/// The polyline of one edge, following the same tessellation as its faces.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EdgePolyline {
    /// Points along the edge.
    pub points: Vec<Point3>,
}

/// A shape's tessellation with its structure kept, for display and picking.
///
/// `faces[i]` belongs to the shape's `i`th distinct face and `edges[i]` to its `i`th
/// distinct edge, in the order the kernel's subshape traversal gives them.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ShapeTessellation {
    /// One mesh per distinct face.
    pub faces: Vec<FaceMesh>,
    /// One polyline per distinct edge.
    pub edges: Vec<EdgePolyline>,
}

impl ShapeTessellation {
    /// Merges every face into a single mesh.
    #[must_use]
    pub fn merged(&self) -> Mesh {
        let mut out = Mesh::default();
        for FaceMesh { mesh, .. } in &self.faces {
            let base = u32::try_from(out.positions.len()).unwrap_or(u32::MAX);
            out.positions.extend_from_slice(&mesh.positions);
            out.normals.extend_from_slice(&mesh.normals);
            out.indices
                .extend(mesh.indices.iter().map(|t| t.map(|i| i + base)));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tetrahedron() -> Mesh {
        Mesh {
            positions: vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(0.0, 1.0, 0.0),
                Point3::new(0.0, 0.0, 1.0),
            ],
            normals: vec![Vec3::ZERO; 4],
            indices: vec![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
        }
    }

    #[test]
    fn volume_and_area() {
        let m = tetrahedron();
        assert!((m.volume() - 1.0 / 6.0).abs() < 1e-12);
        assert!((m.area() - (1.5 + 3f64.sqrt() / 2.0)).abs() < 1e-12);
        assert_eq!(m.triangles().len(), 4);
    }
}
