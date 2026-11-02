//! Exact curves and surfaces as plain data.
//!
//! These describe geometry going into and out of a kernel. They own no kernel state.

#![forbid(unsafe_code)]

use occt_math::{Dir3, Frame, Point3, Vec3};

/// A closed parameter interval.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParamRange {
    /// Lower bound.
    pub start: f64,
    /// Upper bound.
    pub end: f64,
}

/// A NURBS curve in the usual knot-with-multiplicity form.
#[derive(Clone, Debug, PartialEq)]
pub struct BSplineCurve {
    /// Polynomial degree.
    pub degree: u32,
    /// Distinct knots, strictly increasing.
    pub knots: Vec<f64>,
    /// Multiplicity of each knot, parallel to `knots`.
    pub multiplicities: Vec<u32>,
    /// Control points.
    pub poles: Vec<Point3>,
    /// Weights parallel to `poles`, or `None` for a non-rational curve.
    pub weights: Option<Vec<f64>>,
    /// Whether the curve is periodic.
    pub periodic: bool,
}

/// A NURBS surface.
#[derive(Clone, Debug, PartialEq)]
pub struct BSplineSurface {
    /// Degree in U.
    pub degree_u: u32,
    /// Degree in V.
    pub degree_v: u32,
    /// Distinct U knots.
    pub knots_u: Vec<f64>,
    /// Multiplicity of each U knot.
    pub multiplicities_u: Vec<u32>,
    /// Distinct V knots.
    pub knots_v: Vec<f64>,
    /// Multiplicity of each V knot.
    pub multiplicities_v: Vec<u32>,
    /// Number of poles in U; `poles` holds rows of poles in V order.
    pub pole_count_u: u32,
    /// Number of poles in V.
    pub pole_count_v: u32,
    /// Control points, `pole_count_u * pole_count_v` long, U index varying slowest.
    pub poles: Vec<Point3>,
    /// Weights parallel to `poles`, or `None` for a non-rational surface.
    pub weights: Option<Vec<f64>>,
    /// Periodic in U.
    pub periodic_u: bool,
    /// Periodic in V.
    pub periodic_v: bool,
}

/// An exact 3D curve.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum Curve {
    /// An infinite line, parameterised by distance from `origin`.
    Line {
        /// A point on the line.
        origin: Point3,
        /// Its direction.
        direction: Dir3,
    },
    /// A circle in `frame`'s XY plane, parameterised by angle from `frame.x`.
    Circle {
        /// Centre and orientation.
        frame: Frame,
        /// Radius.
        radius: f64,
    },
    /// An ellipse in `frame`'s XY plane with its major axis along `frame.x`.
    Ellipse {
        /// Centre and orientation.
        frame: Frame,
        /// Major radius.
        major_radius: f64,
        /// Minor radius.
        minor_radius: f64,
    },
    /// A NURBS curve; Bézier curves come back in this form too.
    BSpline(BSplineCurve),
    /// A curve the kernel holds in a form this type doesn't describe. It can be
    /// sampled through evaluation but not rebuilt from this value.
    Other,
}

/// An exact surface.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum Surface {
    /// A plane, `frame`'s XY plane.
    Plane {
        /// Origin and orientation.
        frame: Frame,
    },
    /// A cylinder around `frame.z`.
    Cylinder {
        /// Axis and seam orientation.
        frame: Frame,
        /// Radius.
        radius: f64,
    },
    /// A cone around `frame.z`.
    Cone {
        /// Axis and seam orientation.
        frame: Frame,
        /// Radius at `frame.origin`.
        radius: f64,
        /// Half angle in radians; positive when the radius grows along `frame.z`.
        half_angle: f64,
    },
    /// A sphere.
    Sphere {
        /// Centre and orientation.
        frame: Frame,
        /// Radius.
        radius: f64,
    },
    /// A torus around `frame.z`.
    Torus {
        /// Centre and orientation.
        frame: Frame,
        /// Distance from the axis to the tube centre.
        major_radius: f64,
        /// Tube radius.
        minor_radius: f64,
    },
    /// A NURBS surface; Bézier surfaces come back in this form too.
    BSpline(BSplineSurface),
    /// A surface the kernel holds in a form this type doesn't describe, such as an
    /// offset surface. It can be evaluated but not rebuilt from this value.
    Other,
}

/// The exact geometry under an edge.
#[derive(Clone, Debug, PartialEq)]
pub struct EdgeGeometry {
    /// The underlying curve.
    pub curve: Curve,
    /// The part of the curve the edge uses.
    pub range: ParamRange,
    /// Whether the edge runs against the curve's direction.
    pub reversed: bool,
}

/// The exact geometry under a face.
#[derive(Clone, Debug, PartialEq)]
pub struct FaceGeometry {
    /// The underlying surface.
    pub surface: Surface,
    /// Parameter bounds of the face in U.
    pub u: ParamRange,
    /// Parameter bounds of the face in V.
    pub v: ParamRange,
    /// Whether the face's outward normal is opposite the surface normal.
    pub reversed: bool,
}

/// A point on a curve with its first derivative and curvature.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CurvePoint {
    /// The position.
    pub point: Point3,
    /// First derivative with respect to the parameter.
    pub derivative: Vec3,
    /// Curvature, 1 / radius.
    pub curvature: f64,
}

/// A point on a face with its derivatives and curvature.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfacePoint {
    /// The position.
    pub point: Point3,
    /// The outward normal, accounting for the face's orientation. `None` where the
    /// surface is singular, such as a sphere's pole.
    pub normal: Option<Dir3>,
    /// First derivative in U.
    pub du: Vec3,
    /// First derivative in V.
    pub dv: Vec3,
    /// The two principal curvatures, larger first; negative when the face curves away
    /// from its outward normal, and NaN at a singular point.
    pub principal_curvatures: [f64; 2],
}
