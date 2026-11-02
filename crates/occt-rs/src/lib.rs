//! Idiomatic solid modelling.
//!
//! Build solids from primitives, combine them with booleans, round them with blends,
//! and get triangles and STEP files out. Geometry is plain Rust values; shapes are
//! cheap handles that clean up after themselves; every fallible operation returns a
//! [`Result`].
//!
//! # Example
//! A plate with rounded corners and a hole through it, meshed and written to STEP:
//!
//! ```
//! use occt::prelude::*;
//! use std::f64::consts::PI;
//!
//! fn main() -> occt::Result<()> {
//!     // A solid, with one corner at the origin.
//!     let (width, depth, thick) = (80.0, 40.0, 6.0);
//!     let plate = Cuboid::new([width, depth, thick]).build()?;
//!
//!     // Select the four vertical edges, then round them.
//!     let mut corners = Vec::new();
//!     for edge in plate.edges()? {
//!         let size = edge.bounding_box()?.size();
//!         if size.x < 1e-3 && size.y < 1e-3 {
//!             corners.push(edge);
//!         }
//!     }
//!     assert_eq!(corners.len(), 4);
//!     let radius = 3.0;
//!     let rounded = plate.fillet(radius).edges(&corners).build()?;
//!
//!     // Drill a hole. Operators on references read like the set operations they are.
//!     let bore = 5.0;
//!     let axis = Axis::new([width / 2.0, depth / 2.0, -1.0].into(), Dir3::Z);
//!     let drill = Cylinder::new(bore, thick + 2.0).along(axis).build()?;
//!     let part = (&rounded - &drill)?;
//!
//!     // Measurements come from the exact geometry.
//!     let corner_loss = 4.0 * (1.0 - PI / 4.0) * radius * radius * thick;
//!     let expected = width * depth * thick - corner_loss - PI * bore * bore * thick;
//!     assert!((part.volume()? - expected).abs() < 1e-2);
//!
//!     // A mesh is plain data, closed and wound outward, ready for a renderer.
//!     let mesh = part.mesh(MeshOptions::default().absolute(0.05))?;
//!     assert!(!mesh.is_empty());
//!     assert!((mesh.volume() - expected).abs() / expected < 1e-2);
//!
//!     // STEP goes to anything that implements `Write`, such as a `File`.
//!     let mut step = Vec::new();
//!     part.write_step(&mut step)?;
//!     let reloaded = Shape::from_step(&step)?;
//!     assert!((reloaded.volume()? - expected).abs() < 1e-2);
//!     Ok(())
//! }
//! ```
//!
//! # Threads
//! [`Shape`] and its refinements are [`Send`] and [`Sync`], so bodies can live in shared
//! documents and be dropped or read from any thread. Kernel calls are safe from any thread
//! but run one at a time, because shapes share topology that operations such as meshing
//! update. Long operations don't make other threads' work fail; it just waits.
//!
//! ```
//! fn assert_send_sync<T: Send + Sync>() {}
//! assert_send_sync::<occt::Shape>();
//! ```
//!
//! # Licensing
//! The active kernel may carry attribution duties for whoever distributes it. See
//! [`notice`].

#![forbid(unsafe_code)]

mod blend;
mod boolean;
mod construction;
mod exchange;
mod feature;
mod history;
mod inspect;
mod kernel;
mod mesh;
mod ops;
pub mod prelude;
mod primitives;
mod shape;
mod step;
mod subshape;
mod surfacing;
mod sweep;
mod transform;
mod validate;

pub use occt_geom::{
    BSplineCurve, BSplineSurface, Curve, CurvePoint, EdgeGeometry, FaceGeometry, ParamRange,
    Surface, SurfacePoint,
};
pub use occt_kernel::{CancelFlag, CurveKind, Error, Result, ShapeKind, SurfaceKind};
pub use occt_math::{
    Aabb, Angle, Axis, Dir3, Frame, Plane, Point3, Transform, Vec3, ZeroDirection,
};
pub use occt_mesh::{Deflection, EdgePolyline, FaceMesh, Mesh, MeshOptions, ShapeTessellation};

pub use blend::{Chamfer, Fillet};
pub use construction::*;
pub use exchange::*;
pub use feature::*;
pub use history::{History, Modeled};
pub use inspect::*;
pub use kernel::{kernel_version, notice, ready};
pub use ops::*;
pub use primitives::{Cone, Cuboid, Cylinder, Polygon, Sphere};
pub use shape::Shape;
pub use subshape::{Edge, Face, Solid, Vertex};
pub use surfacing::*;
pub use sweep::*;

/// Compile-checks the example in the repository's README.
#[cfg(doctest)]
#[doc = include_str!("../../../README.md")]
struct ReadmeDoctests;
