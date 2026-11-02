//! The names most modelling code needs, for a glob import.
//!
//! ```
//! use occt::prelude::*;
//!
//! # fn main() -> occt::Result<()> {
//! let ball = Sphere::new(1.0).at([0.0, 0.0, 2.0]).build()?;
//! assert_eq!(ball.kind()?, ShapeKind::Solid);
//! # Ok(())
//! # }
//! ```
//!
//! [`Result`](crate::Result) is deliberately left out so that it does not shadow the
//! standard one. Write `occt::Result`.

pub use crate::{
    Aabb, Angle, Assembly, Axis, BlendContinuity, Boolean, BooleanOp, CancelFlag, Chamfer,
    ChamferSize, Color, Compound, Cone, ContinuityReport, Cuboid, Curve, CurveKind, Cylinder,
    Deflection, Dir3, Distance, Document, DrawingLine, DrawingLineKind, Edge, EdgeChamfer,
    EdgeConvexity, EdgeFillet, Error, Face, Fillet, Frame, Helix, History, Loft, MassProperties,
    Mesh, MeshOptions, Modeled, Plane, Point3, PointLocation, Polygon, RadiusLaw, RayHit, Sewing,
    Shape, ShapeKind, ShapeTessellation, Shell, Solid, Sphere, Spline, Split, StepOptions,
    StepSchema, Surface, SurfaceKind, Sweep, SweepOrientation, Tessellate, Torus, Transform, Vec3,
    Vertex, View, Wire,
};
