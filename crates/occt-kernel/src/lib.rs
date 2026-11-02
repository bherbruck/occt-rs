//! The kernel contract of occt-rs.
//!
//! Everything above this crate talks to geometry only through these traits, and
//! this crate contains no FFI. A backend (today `occt-kernel-capi`, later possibly
//! a pure Rust implementation) implements them; `occt-conformance` checks that it
//! behaves. The traits are split by domain so one domain can move to a new
//! implementation without touching the others.
//!
//! Functions have no receiver: a kernel is a type, not a value.
//!
//! # Preconditions
//! Callers (the `occt-rs` API layer) validate plain arguments before calling:
//! coordinates are finite, sizes and radii are strictly positive. That keeps
//! validation and its messages identical across kernels. Kernels report
//! everything else, such as a failed boolean, as an [`Error`].

#![forbid(unsafe_code)]

use std::fmt;

pub use occt_geom::{
    BSplineCurve, BSplineSurface, Curve, CurvePoint, EdgeGeometry, FaceGeometry, ParamRange,
    Surface, SurfacePoint,
};
pub use occt_math::{
    Aabb, Angle, Axis, Dir3, Frame, Plane, Point3, Transform, Vec3, ZeroDirection,
};
pub use occt_mesh::{Deflection, EdgePolyline, FaceMesh, Mesh, MeshOptions, ShapeTessellation};

mod cad;
pub use cad::*;

/// The error type of all modelling operations.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// An argument was rejected before reaching the kernel.
    InvalidInput(String),
    /// A shape of a different kind was required.
    WrongKind {
        /// The kind the operation needs.
        expected: ShapeKind,
        /// The kind it got.
        found: ShapeKind,
    },
    /// The kernel could not carry out the operation.
    Operation {
        /// The operation, such as `"fuse"`.
        op: &'static str,
        /// The kernel's explanation.
        message: String,
    },
    /// The operation was cancelled through its [`CancelFlag`].
    Cancelled,
    /// The kernel is not usable: not loaded yet, or an incompatible version.
    Unavailable(String),
    /// Reading or writing a stream failed.
    Io(std::io::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(message) => write!(f, "invalid input: {message}"),
            Self::WrongKind { expected, found } => {
                write!(f, "expected a {expected}, found a {found}")
            }
            Self::Operation { op, message } => write!(f, "{op} failed: {message}"),
            Self::Cancelled => f.write_str("the operation was cancelled"),
            Self::Unavailable(message) => write!(f, "kernel unavailable: {message}"),
            Self::Io(e) => write!(f, "I/O error: {e}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<ZeroDirection> for Error {
    fn from(e: ZeroDirection) -> Self {
        Self::InvalidInput(e.to_string())
    }
}

/// Result of a modelling operation.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// The topological kind of a shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ShapeKind {
    /// A group of arbitrary shapes.
    Compound,
    /// Solids sharing faces.
    CompSolid,
    /// A closed volume.
    Solid,
    /// Connected faces.
    Shell,
    /// A bounded surface.
    Face,
    /// Connected edges.
    Wire,
    /// A bounded curve.
    Edge,
    /// A point.
    Vertex,
}

impl fmt::Display for ShapeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Compound => "compound",
            Self::CompSolid => "compsolid",
            Self::Solid => "solid",
            Self::Shell => "shell",
            Self::Face => "face",
            Self::Wire => "wire",
            Self::Edge => "edge",
            Self::Vertex => "vertex",
        })
    }
}

/// The geometry underlying a face.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SurfaceKind {
    /// A plane.
    Plane,
    /// A cylinder.
    Cylinder,
    /// A cone.
    Cone,
    /// A sphere.
    Sphere,
    /// A torus.
    Torus,
    /// A B-spline or Bézier surface.
    Spline,
    /// Anything else, such as a surface of revolution.
    Other,
}

/// The geometry underlying an edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum CurveKind {
    /// A straight line.
    Line,
    /// A circle or arc.
    Circle,
    /// An ellipse or arc of one.
    Ellipse,
    /// A B-spline or Bézier curve.
    Spline,
    /// Anything else.
    Other,
}

/// Shape storage, identity and traversal. Every other trait builds on this.
pub trait Topology: 'static {
    /// An owned reference to a shape. Dropping it releases the reference.
    /// Clones share the same underlying topology and compare as same. References can
    /// cross threads; the kernel makes concurrent calls safe, serialising them if it
    /// has to.
    type Shape: Clone + fmt::Debug + Send + Sync + 'static;

    /// The kind of a shape.
    fn kind(shape: &Self::Shape) -> Result<ShapeKind>;
    /// Whether two references name the same topology at the same location,
    /// regardless of orientation. An equivalence relation.
    fn is_same(a: &Self::Shape, b: &Self::Shape) -> bool;
    /// A hash consistent with [`Topology::is_same`].
    fn hash_code(shape: &Self::Shape) -> u32;
    /// Whether the shape's topology and geometry pass the kernel's validity check.
    fn is_valid(shape: &Self::Shape) -> Result<bool>;
    /// The distinct subshapes of a kind, in a stable order. Includes `shape` itself
    /// when it is of that kind.
    fn subshapes(shape: &Self::Shape, kind: ShapeKind) -> Result<Vec<Self::Shape>>;
}

/// Direct solid and face construction.
pub trait Primitives: Topology {
    /// A box with one corner at `corner`, extending along +X, +Y, +Z by `size`.
    fn make_box(corner: Point3, size: Vec3) -> Result<Self::Shape>;
    /// A cylinder from `axis.origin` along `axis.direction`.
    fn make_cylinder(axis: Axis, radius: f64, height: f64) -> Result<Self::Shape>;
    /// A cone or frustum with `radius1` at `axis.origin` and `radius2` at the far
    /// end; one radius may be zero.
    fn make_cone(axis: Axis, radius1: f64, radius2: f64, height: f64) -> Result<Self::Shape>;
    /// A sphere.
    fn make_sphere(center: Point3, radius: f64) -> Result<Self::Shape>;
    /// A planar face bounded by a closed polygon through the points.
    fn make_polygon_face(points: &[Point3]) -> Result<Self::Shape>;
}

/// Solids made by moving a lower-dimensional shape.
pub trait Sweeps: Topology {
    /// Sweeps along a straight vector.
    fn extrude(shape: &Self::Shape, by: Vec3) -> Result<Self::Shape>;
    /// Sweeps around an axis.
    fn revolve(shape: &Self::Shape, axis: Axis, angle: Angle) -> Result<Self::Shape>;
}

/// Boolean operations.
pub trait Booleans: Topology {
    /// The union.
    fn fuse(a: &Self::Shape, b: &Self::Shape) -> Result<Self::Shape>;
    /// `a` without `b`.
    fn cut(a: &Self::Shape, b: &Self::Shape) -> Result<Self::Shape>;
    /// The intersection.
    fn common(a: &Self::Shape, b: &Self::Shape) -> Result<Self::Shape>;
}

/// Edge blends.
pub trait Blends: Topology {
    /// Rounds the edges of `shape`.
    fn fillet(shape: &Self::Shape, edges: &[&Self::Shape], radius: f64) -> Result<Self::Shape>;
    /// Bevels the edges of `shape`.
    fn chamfer(shape: &Self::Shape, edges: &[&Self::Shape], distance: f64) -> Result<Self::Shape>;
}

/// Rigid and uniform transforms. Results are independent copies.
pub trait Transforms: Topology {
    /// Moves a shape.
    fn translate(shape: &Self::Shape, by: Vec3) -> Result<Self::Shape>;
    /// Rotates a shape around an axis.
    fn rotate(shape: &Self::Shape, axis: Axis, angle: Angle) -> Result<Self::Shape>;
    /// Scales a shape uniformly about a point.
    fn scale(shape: &Self::Shape, center: Point3, factor: f64) -> Result<Self::Shape>;
}

/// Measurements.
pub trait Properties: Topology {
    /// Enclosed volume.
    fn volume(shape: &Self::Shape) -> Result<f64>;
    /// Surface area.
    fn area(shape: &Self::Shape) -> Result<f64>;
    /// Total edge length.
    fn length(shape: &Self::Shape) -> Result<f64>;
    /// Centre of mass; of the surface when the shape encloses no volume.
    fn center_of_mass(shape: &Self::Shape) -> Result<Point3>;
    /// A tight axis-aligned bounding box.
    fn bounding_box(shape: &Self::Shape) -> Result<Aabb>;
    /// The position of a vertex shape.
    fn vertex_point(vertex: &Self::Shape) -> Result<Point3>;
    /// The geometry under a face shape.
    fn surface_kind(face: &Self::Shape) -> Result<SurfaceKind>;
    /// The geometry under an edge shape.
    fn curve_kind(edge: &Self::Shape) -> Result<CurveKind>;
}

/// Conversion to triangles.
pub trait Tessellation: Topology {
    /// Tessellates a shape.
    fn mesh(shape: &Self::Shape, options: &MeshOptions) -> Result<Mesh>;
}

/// STEP exchange.
pub trait StepIo: Topology {
    /// Serialises a shape as a STEP file.
    fn write_step(shape: &Self::Shape) -> Result<Vec<u8>>;
    /// Parses a STEP file.
    fn read_step(data: &[u8]) -> Result<Self::Shape>;
}

/// A complete kernel: every domain, plus what the kernel must disclose.
pub trait Kernel:
    Primitives
    + Sweeps
    + Booleans
    + Blends
    + Transforms
    + Properties
    + Tessellation
    + StepIo
    + Construction
    + Geometry
    + AdvancedSweeps
    + Features
    + BooleanOps
    + AdvancedBlends
    + Inspection
    + Exchange
    + Surfacing
{
    /// Short kernel name, for diagnostics.
    const NAME: &'static str;

    /// Attribution that distributors of this kernel must reproduce, as defined by
    /// the kernel's licences. Empty if none is required.
    const NOTICE: &'static str;

    /// The version of the underlying implementation.
    fn version() -> Result<String>;

    /// Checks that the kernel can run now. Backends that load at run time report
    /// [`Error::Unavailable`] here and from operations until they are loaded.
    fn ready() -> Result<()>;
}
