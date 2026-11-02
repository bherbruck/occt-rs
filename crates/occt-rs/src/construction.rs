//! Building topology bottom up from geometry, and reading exact geometry back.
//!
//! Vertices make [`Edge`]s, edges make [`Wire`]s, wires bound [`Face`]s, faces sew into
//! [`Shell`]s, and a closed shell makes a [`Solid`]:
//!
//! ```
//! use occt::{Compound, Wire};
//! use occt::prelude::*;
//!
//! # fn main() -> occt::Result<()> {
//! let outer = Wire::rectangle(10.0, 10.0)?;
//! let hole = Wire::circle([0.0, 0.0, 0.0], Dir3::Z, 2.0)?;
//! let plate = Face::planar(&outer).hole(&hole).build()?;
//! assert!((plate.area()? - (100.0 - std::f64::consts::PI * 4.0)).abs() < 1e-6);
//!
//! let group = Compound::new([&plate])?;
//! assert_eq!(group.faces()?.count(), 1);
//! # Ok(())
//! # }
//! ```

use std::borrow::Borrow;
use std::ops::{Deref, Range};

use occt_geom::{
    BSplineCurve, BSplineSurface, Curve, CurvePoint, EdgeGeometry, FaceGeometry, ParamRange,
    Surface, SurfacePoint,
};
use occt_kernel::{
    Axis, Construction, Dir3, Error, FillEdge, Frame, Geometry, Point3, Properties, Result,
    ShapeKind, Surfacing, Topology, Vec3,
};

use crate::kernel::Active;
use crate::shape::Shape;
use crate::subshape::{Edge, Face, Solid, Vertex};
use crate::validate::{axis, finite, invalid, point, positive, vector};

/// Below this distance, model units, two points are the same point.
const COINCIDENT: f64 = 1e-7;
const FULL_TURN: f64 = std::f64::consts::TAU;

// --- refinements ----------------------------------------------------------------------

/// Declares a refinement of [`Shape`] and every conversion it has, as `subshape.rs` does for
/// the kinds it holds.
macro_rules! refinement {
    ($(#[$doc:meta])* $name:ident, $kind:expr, $iter:ident, $plural:literal) => {
        $(#[$doc])*
        ///
        /// This is a refinement of [`Shape`], in the way [`String`] refines [`str`]: it
        /// [`Deref`]s to `Shape`, so every shape method is available and `&Self` coerces
        /// to `&Shape`. It compares and hashes exactly as the shape it holds. Convert with
        /// [`TryFrom`] (checks the kind) and [`From`] (always succeeds).
        #[derive(Clone, Debug, PartialEq, Eq, Hash)]
        pub struct $name(Shape);

        impl $name {
            /// The kind that every value of this type has.
            pub const KIND: ShapeKind = $kind;

            /// Wraps a shape that is known to have this kind.
            fn known(shape: Shape) -> Self {
                debug_assert!(shape.kind().is_ok_and(|kind| kind == Self::KIND));
                Self(shape)
            }
        }

        impl Shape {
            #[doc = concat!("The distinct ", $plural, " of this shape, in a stable order.")]
            ///
            /// The iterator owns its items and does not borrow the shape. Shapes that
            /// neighbours share appear once. If this shape is itself of that kind, the
            /// iterator yields it.
            ///
            /// # Errors
            /// [`Error::Operation`] if the traversal fails, which is reported before
            /// any iteration starts.
            pub fn $iter(
                &self,
            ) -> Result<impl ExactSizeIterator<Item = $name> + DoubleEndedIterator + use<>> {
                let raw = Active::subshapes(self.raw(), $kind)?;
                Ok(raw.into_iter().map(|raw| $name::known(Shape::from_raw(raw))))
            }
        }

        impl TryFrom<Shape> for $name {
            type Error = Error;

            /// # Errors
            /// [`Error::WrongKind`] if the shape has a different kind.
            fn try_from(shape: Shape) -> Result<Self> {
                match shape.kind()? {
                    found if found == Self::KIND => Ok(Self(shape)),
                    found => Err(Error::WrongKind { expected: Self::KIND, found }),
                }
            }
        }

        impl TryFrom<&Shape> for $name {
            type Error = Error;

            /// # Errors
            /// [`Error::WrongKind`] if the shape has a different kind.
            fn try_from(shape: &Shape) -> Result<Self> {
                Self::try_from(shape.clone())
            }
        }

        impl From<$name> for Shape {
            fn from(value: $name) -> Shape {
                value.0
            }
        }

        impl Deref for $name {
            type Target = Shape;

            fn deref(&self) -> &Shape {
                &self.0
            }
        }

        impl AsRef<Shape> for $name {
            fn as_ref(&self) -> &Shape {
                &self.0
            }
        }

        impl Borrow<Shape> for $name {
            fn borrow(&self) -> &Shape {
                &self.0
            }
        }
    };
}

refinement!(
    /// A connected chain of edges, open or closed: a profile, a path or a face's boundary.
    Wire, ShapeKind::Wire, wires, "wires"
);
refinement!(
    /// Faces joined along shared edges: a sheet, or the skin of a solid when closed.
    Shell, ShapeKind::Shell, shells, "shells"
);
refinement!(
    /// A group of shapes of any kinds, kept together but not joined.
    Compound, ShapeKind::Compound, compounds, "compounds"
);

// --- argument checks ------------------------------------------------------------------

fn typed<T: TryFrom<Shape, Error = Error>>(raw: crate::kernel::RawShape) -> Result<T> {
    Shape::from_raw(raw).try_into()
}

fn direction(what: &str, value: impl Into<Vec3>) -> Result<Dir3> {
    let value = vector(what, value.into())?;
    Dir3::new(value).map_err(|_| invalid(format!("{what} must not be zero")))
}

fn frame(what: &str, value: Frame) -> Result<Frame> {
    point(what, value.origin)?;
    Ok(value)
}

fn points(what: &str, values: Vec<Point3>) -> Result<Vec<Point3>> {
    for value in &values {
        point(what, *value)?;
    }
    Ok(values)
}

/// Requires consecutive points to be distinct, which any curve through them needs.
fn distinct_neighbours(what: &str, values: &[Point3]) -> Result<()> {
    if values.windows(2).any(|w| w[0].distance(w[1]) < COINCIDENT) {
        return Err(invalid(format!("{what} has two coincident neighbours")));
    }
    Ok(())
}

fn param_range(what: &str, range: &Range<f64>) -> Result<ParamRange> {
    finite(what, range.start)?;
    finite(what, range.end)?;
    if range.start >= range.end {
        return Err(invalid(format!(
            "{what} must run upward, got {}..{}",
            range.start, range.end
        )));
    }
    Ok(ParamRange {
        start: range.start,
        end: range.end,
    })
}

/// Checks a knot vector and returns the number of poles it calls for.
fn knots(
    what: &str,
    degree: u32,
    knots: &[f64],
    multiplicities: &[u32],
    periodic: bool,
) -> Result<usize> {
    if degree == 0 {
        return Err(invalid(format!("{what} needs a degree of at least 1")));
    }
    if knots.len() < 2 || knots.len() != multiplicities.len() {
        return Err(invalid(format!(
            "{what} needs at least 2 knots and one multiplicity per knot"
        )));
    }
    if knots.iter().any(|k| !k.is_finite()) || knots.windows(2).any(|w| w[0] >= w[1]) {
        return Err(invalid(format!(
            "{what} needs finite, strictly increasing knots"
        )));
    }
    let last = multiplicities.len() - 1;
    for (i, m) in multiplicities.iter().enumerate() {
        let end = i == 0 || i == last;
        if *m == 0 || *m > degree + u32::from(end && !periodic) {
            return Err(invalid(format!(
                "{what} has an out-of-range multiplicity {m} at knot {i}"
            )));
        }
    }
    let sum: usize = multiplicities.iter().map(|m| *m as usize).sum();
    if periodic {
        if multiplicities[0] != multiplicities[last] {
            return Err(invalid(format!(
                "{what} is periodic, so its first and last multiplicities must match"
            )));
        }
        Ok(sum - multiplicities[last] as usize)
    } else {
        sum.checked_sub(degree as usize + 1)
            .filter(|n| *n >= 2)
            .ok_or_else(|| invalid(format!("{what} has too few knots for its degree")))
    }
}

fn weights(what: &str, values: Option<&[f64]>, poles: usize) -> Result<()> {
    let Some(values) = values else { return Ok(()) };
    if values.len() != poles {
        return Err(invalid(format!(
            "{what} needs one weight per pole, got {} for {poles}",
            values.len()
        )));
    }
    if values.iter().any(|w| !w.is_finite() || *w <= 0.0) {
        return Err(invalid(format!("{what} needs positive, finite weights")));
    }
    Ok(())
}

fn check_bspline_curve(s: &BSplineCurve) -> Result<()> {
    const WHAT: &str = "a spline curve";
    let poles = knots(WHAT, s.degree, &s.knots, &s.multiplicities, s.periodic)?;
    if s.poles.len() != poles {
        return Err(invalid(format!(
            "{WHAT} with these knots needs {poles} poles, got {}",
            s.poles.len()
        )));
    }
    for pole in &s.poles {
        point("a spline pole", *pole)?;
    }
    weights(WHAT, s.weights.as_deref(), poles)
}

fn check_bspline_surface(s: &BSplineSurface) -> Result<()> {
    const WHAT: &str = "a spline surface";
    let u = knots(
        WHAT,
        s.degree_u,
        &s.knots_u,
        &s.multiplicities_u,
        s.periodic_u,
    )?;
    let v = knots(
        WHAT,
        s.degree_v,
        &s.knots_v,
        &s.multiplicities_v,
        s.periodic_v,
    )?;
    if u != s.pole_count_u as usize || v != s.pole_count_v as usize {
        return Err(invalid(format!(
            "{WHAT} with these knots needs {u} by {v} poles, got {} by {}",
            s.pole_count_u, s.pole_count_v
        )));
    }
    if s.poles.len() != u * v {
        return Err(invalid(format!(
            "{WHAT} needs {} poles, got {}",
            u * v,
            s.poles.len()
        )));
    }
    for pole in &s.poles {
        point("a spline pole", *pole)?;
    }
    weights(WHAT, s.weights.as_deref(), u * v)
}

fn check_curve(curve: &Curve) -> Result<()> {
    match curve {
        Curve::Line { origin, .. } => {
            point("a line's origin", *origin)?;
        }
        Curve::Circle { frame: f, radius } => {
            frame("a circle's frame", *f)?;
            positive("a circle's radius", *radius)?;
        }
        Curve::Ellipse {
            frame: f,
            major_radius,
            minor_radius,
        } => {
            frame("an ellipse's frame", *f)?;
            positive("an ellipse's minor radius", *minor_radius)?;
            positive("an ellipse's major radius", *major_radius)?;
            if major_radius < minor_radius {
                return Err(invalid(format!(
                    "an ellipse's major radius {major_radius} is below its minor radius \
                     {minor_radius}"
                )));
            }
        }
        Curve::BSpline(spline) => check_bspline_curve(spline)?,
        _ => {
            return Err(invalid(
                "this curve has no description, so it can't be built",
            ));
        }
    }
    Ok(())
}

fn check_surface(surface: &Surface) -> Result<()> {
    match surface {
        Surface::Plane { frame: f } => {
            frame("a plane's frame", *f)?;
        }
        Surface::Cylinder { frame: f, radius } => {
            frame("a cylinder's frame", *f)?;
            positive("a cylinder's radius", *radius)?;
        }
        Surface::Cone {
            frame: f,
            radius,
            half_angle,
        } => {
            frame("a cone's frame", *f)?;
            if !radius.is_finite() || *radius < 0.0 {
                return Err(invalid(format!(
                    "a cone's radius must be finite and not negative, got {radius}"
                )));
            }
            if !half_angle.is_finite()
                || *half_angle == 0.0
                || half_angle.abs() >= std::f64::consts::FRAC_PI_2
            {
                return Err(invalid(format!(
                    "a cone's half angle must be between 0 and a quarter turn, got {half_angle}"
                )));
            }
        }
        Surface::Sphere { frame: f, radius } => {
            frame("a sphere's frame", *f)?;
            positive("a sphere's radius", *radius)?;
        }
        Surface::Torus {
            frame: f,
            major_radius,
            minor_radius,
        } => {
            frame("a torus's frame", *f)?;
            positive("a torus's minor radius", *minor_radius)?;
            positive("a torus's major radius", *major_radius)?;
        }
        Surface::BSpline(spline) => check_bspline_surface(spline)?,
        _ => {
            return Err(invalid(
                "this surface has no description, so it can't be built",
            ));
        }
    }
    Ok(())
}

/// Requires a range to fit in one turn of a closed analytic curve.
fn check_curve_range(curve: &Curve, range: ParamRange) -> Result<()> {
    let span = range.end - range.start;
    if matches!(curve, Curve::Circle { .. } | Curve::Ellipse { .. })
        && span > FULL_TURN * (1.0 + 1e-12)
    {
        return Err(invalid(format!(
            "the range of a closed curve can't exceed one turn, got a span of {span}"
        )));
    }
    Ok(())
}

// --- vertices -------------------------------------------------------------------------

impl Vertex {
    /// A vertex at a point.
    ///
    /// # Errors
    /// [`Error::InvalidInput`] if the point is not finite; [`Error::Operation`] if the
    /// kernel fails.
    ///
    /// # Examples
    /// ```
    /// use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let v = Vertex::new([1.0, 2.0, 3.0])?;
    /// assert_eq!(v.point()?, Point3::new(1.0, 2.0, 3.0));
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(at: impl Into<Point3>) -> Result<Self> {
        typed(Active::make_vertex(point("a vertex", at.into())?)?)
    }
}

// --- edges ----------------------------------------------------------------------------

/// A builder for an edge along any curve, from [`Edge::from_curve`].
///
/// # Examples
/// ```
/// use occt::prelude::*;
/// use occt::{Curve, Frame};
/// use std::f64::consts::PI;
///
/// # fn main() -> occt::Result<()> {
/// let circle = Curve::Circle { frame: Frame::WORLD, radius: 2.0 };
/// let half = Edge::from_curve(circle).range(0.0..PI).build()?;
/// assert!((half.length()? - 2.0 * PI).abs() < 1e-9);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq)]
#[must_use = "a builder does nothing until `build` is called"]
pub struct EdgeBuilder {
    curve: Curve,
    range: Option<Range<f64>>,
}

impl EdgeBuilder {
    /// Uses only this part of the curve, in the curve's own parameter. Required for a line;
    /// without it, a bounded curve is used whole.
    pub fn range(mut self, range: Range<f64>) -> Self {
        self.range = Some(range);
        self
    }

    /// Builds the edge.
    ///
    /// # Errors
    /// [`Error::InvalidInput`] if the curve's data is not consistent or not finite, or has no
    /// description; if the curve is an unbounded line with no range, or the range is
    /// backwards or longer than one turn of a closed curve. [`Error::Operation`] if the
    /// kernel rejects it, as for a range outside a spline's knots.
    ///
    /// # Examples
    /// ```
    /// use occt::prelude::*;
    /// use occt::{Curve, Frame};
    ///
    /// # fn main() -> occt::Result<()> {
    /// let line = Curve::Line { origin: Point3::ORIGIN, direction: Dir3::X };
    /// assert!(Edge::from_curve(line.clone()).build().is_err());
    /// let edge = Edge::from_curve(line).range(0.0..3.0).build()?;
    /// assert!((edge.length()? - 3.0).abs() < 1e-9);
    /// # Ok(())
    /// # }
    /// ```
    pub fn build(self) -> Result<Edge> {
        check_curve(&self.curve)?;
        let range = match &self.range {
            Some(range) => {
                let range = param_range("an edge's range", range)?;
                check_curve_range(&self.curve, range)?;
                Some(range)
            }
            None if matches!(self.curve, Curve::Line { .. }) => {
                return Err(invalid("a line is unbounded, so its edge needs a range"));
            }
            None => None,
        };
        typed(Active::make_edge(&self.curve, range)?)
    }
}

/// Where a point lands on an edge, from [`Edge::project`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CurveProjection {
    /// The curve parameter of the closest point. It lies in the edge's range.
    pub parameter: f64,
    /// The closest point of the edge.
    pub point: Point3,
}

impl Edge {
    /// A straight edge between two points, the line segment `a` to `b`.
    ///
    /// # Errors
    /// [`Error::InvalidInput`] if a point is not finite or the points coincide.
    ///
    /// # Examples
    /// ```
    /// use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let edge = Edge::line([0.0, 0.0, 0.0], [3.0, 4.0, 0.0])?;
    /// assert!((edge.length()? - 5.0).abs() < 1e-9);
    /// # Ok(())
    /// # }
    /// ```
    #[doc(alias = "segment")]
    pub fn line(a: impl Into<Point3>, b: impl Into<Point3>) -> Result<Self> {
        let (a, b) = (
            point("a line's end", a.into())?,
            point("a line's end", b.into())?,
        );
        let length = a.distance(b);
        if length < COINCIDENT {
            return Err(invalid("a line needs two different points"));
        }
        let line = Curve::Line {
            origin: a,
            direction: Dir3::new(b - a)?,
        };
        Self::from_curve(line).range(0.0..length).build()
    }

    /// A full circle. It starts and ends at the same vertex, and is closed.
    ///
    /// `normal` is anything that converts to a vector, such as a [`Dir3`] or an array. The
    /// circle runs counterclockwise seen with the normal pointing at you.
    ///
    /// # Errors
    /// [`Error::InvalidInput`] if the radius is not finite and positive, the normal is
    /// zero, or something is not finite.
    ///
    /// # Examples
    /// ```
    /// use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let edge = Edge::circle([0.0, 0.0, 0.0], Dir3::Z, 2.0)?;
    /// assert!(edge.is_closed()?);
    /// assert!((edge.length()? - 4.0 * std::f64::consts::PI).abs() < 1e-9);
    /// # Ok(())
    /// # }
    /// ```
    pub fn circle(center: impl Into<Point3>, normal: impl Into<Vec3>, radius: f64) -> Result<Self> {
        let center = point("a circle's center", center.into())?;
        let normal = direction("a circle's normal", normal)?;
        let frame = Frame::from_normal(center, normal);
        Self::from_curve(Curve::Circle { frame, radius }).build()
    }

    /// A circular arc from `start` through `through` to `end`.
    ///
    /// # Errors
    /// [`Error::InvalidInput`] if a point is not finite or the points don't define a
    /// circle, because two coincide or all three lie on one line.
    ///
    /// # Examples
    /// ```
    /// use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let half = Edge::arc([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0])?;
    /// assert!((half.length()? - std::f64::consts::PI).abs() < 1e-9);
    /// assert!(half.start_point()?.distance([1.0, 0.0, 0.0].into()) < 1e-9);
    /// assert!(half.end_point()?.distance([-1.0, 0.0, 0.0].into()) < 1e-9);
    /// # Ok(())
    /// # }
    /// ```
    pub fn arc(
        start: impl Into<Point3>,
        through: impl Into<Point3>,
        end: impl Into<Point3>,
    ) -> Result<Self> {
        let start = point("an arc's start", start.into())?;
        let through = point("an arc's middle", through.into())?;
        let end = point("an arc's end", end.into())?;
        let (u, v) = (through - start, end - start);
        let w = u.cross(v);
        let twice = 2.0 * w.dot(w);
        let scale = u.dot(u).max(v.dot(v));
        // |w|^2 is scale^2 sin^2 of the angle between them; this asks for about 1e-7 rad.
        if twice <= 1e-14 * scale * scale || scale < COINCIDENT * COINCIDENT {
            return Err(invalid("an arc's three points must not lie on one line"));
        }
        let to_center = (v.cross(w) * u.dot(u) + w.cross(u) * v.dot(v)) / twice;
        let center = start + to_center;
        let radius = to_center.length();
        let frame = Frame::new(center, Dir3::new(w)?, start - center)?;
        let (x, y) = (frame.x.vector(), frame.y().vector());
        let to_end = end - center;
        let mut sweep = to_end.dot(y).atan2(to_end.dot(x));
        if sweep <= 0.0 {
            sweep += FULL_TURN;
        }
        Self::from_curve(Curve::Circle { frame, radius })
            .range(0.0..sweep)
            .build()
    }

    /// A full ellipse in `frame`'s XY plane, with its major axis along `frame.x`.
    ///
    /// # Errors
    /// [`Error::InvalidInput`] if a radius is not finite and positive or `major` is below
    /// `minor`.
    ///
    /// # Examples
    /// ```
    /// use occt::prelude::*;
    /// use occt::Frame;
    /// # fn main() -> occt::Result<()> {
    /// let edge = Edge::ellipse(Frame::WORLD, 3.0, 1.0)?;
    /// assert!(edge.is_closed()?);
    /// # Ok(())
    /// # }
    /// ```
    pub fn ellipse(frame: Frame, major: f64, minor: f64) -> Result<Self> {
        Self::from_curve(Curve::Ellipse {
            frame,
            major_radius: major,
            minor_radius: minor,
        })
        .build()
    }

    /// A smooth edge through the points, in order.
    ///
    /// # Errors
    /// See [`Spline::build`].
    ///
    /// # Examples
    /// ```
    /// use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let edge = Edge::spline([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 0.0, 0.0]])?;
    /// assert_eq!(edge.curve_kind()?, CurveKind::Spline);
    /// # Ok(())
    /// # }
    /// ```
    pub fn spline(points: impl IntoIterator<Item: Into<Point3>>) -> Result<Self> {
        Self::from_curve(Spline::through(points).build()?).build()
    }

    /// A smooth closed edge through the points, with no corner where it closes.
    ///
    /// # Errors
    /// See [`Spline::build`].
    ///
    /// # Examples
    /// ```
    /// use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let edge = Edge::closed_spline([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])?;
    /// assert!(edge.is_closed()?);
    /// # Ok(())
    /// # }
    /// ```
    pub fn closed_spline(points: impl IntoIterator<Item: Into<Point3>>) -> Result<Self> {
        Self::from_curve(Spline::through(points).closed().build()?).build()
    }

    /// Starts an edge along any curve, as plain data.
    ///
    /// Any curve read from [`Edge::curve`] other than [`Curve::Other`] can be passed back.
    ///
    /// # Examples
    /// ```
    /// use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let edge = Edge::line([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])?;
    /// let copy = Edge::from_curve(edge.curve()?).range(0.0..0.5).build()?;
    /// assert!((copy.length()? - 0.5).abs() < 1e-9);
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_curve(curve: Curve) -> EdgeBuilder {
        EdgeBuilder { curve, range: None }
    }

    /// The exact geometry under this edge: its curve, the range used and its direction.
    ///
    /// A curve the kernel holds in a form [`Curve`] has no description of comes back as
    /// [`Curve::Other`], and [`point_at`](Edge::point_at) still evaluates it. A range on a
    /// periodic curve may be a whole number of periods from the one the edge was built with.
    ///
    /// # Errors
    /// [`Error::Operation`] if the kernel cannot read it.
    ///
    /// # Examples
    /// ```
    /// use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let edge = Edge::line([0.0, 0.0, 0.0], [2.0, 0.0, 0.0])?;
    /// let geometry = edge.geometry()?;
    /// assert_eq!((geometry.range.start, geometry.range.end), (0.0, 2.0));
    /// assert!(!geometry.reversed);
    /// # Ok(())
    /// # }
    /// ```
    pub fn geometry(&self) -> Result<EdgeGeometry> {
        Active::edge_geometry(self.raw())
    }

    /// The curve under this edge, whole, without the range the edge uses of it.
    ///
    /// # Errors
    /// [`Error::Operation`] if the kernel cannot read it.
    ///
    /// # Examples
    /// ```
    /// use occt::prelude::*;
    /// use occt::Curve;
    /// # fn main() -> occt::Result<()> {
    /// let edge = Edge::circle([0.0, 0.0, 0.0], Dir3::Z, 2.0)?;
    /// assert!(matches!(edge.curve()?, Curve::Circle { radius, .. } if radius == 2.0));
    /// # Ok(())
    /// # }
    /// ```
    pub fn curve(&self) -> Result<Curve> {
        Ok(self.geometry()?.curve)
    }

    /// Evaluates this edge at a curve parameter in its range: position, first derivative
    /// and curvature.
    ///
    /// The parameter runs along the curve whichever way the edge is used. Curvature is NaN
    /// where the first derivative vanishes, because it is then not defined.
    ///
    /// # Errors
    /// [`Error::InvalidInput`] if the parameter is not finite; [`Error::Operation`] if it is
    /// outside the edge's range.
    ///
    /// # Examples
    /// ```
    /// use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let edge = Edge::circle([0.0, 0.0, 0.0], Dir3::Z, 2.0)?;
    /// assert!((edge.point_at(0.0)?.curvature - 0.5).abs() < 1e-9);
    /// # Ok(())
    /// # }
    /// ```
    pub fn point_at(&self, parameter: f64) -> Result<CurvePoint> {
        Active::edge_point(self.raw(), finite("a curve parameter", parameter)?)
    }

    /// Where this edge starts, in the direction it is used.
    ///
    /// # Errors
    /// [`Error::Operation`] if the kernel cannot read it.
    ///
    /// # Examples
    /// ```
    /// use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let edge = Edge::line([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])?;
    /// assert!(edge.start_point()?.distance([1.0, 2.0, 3.0].into()) < 1e-9);
    /// # Ok(())
    /// # }
    /// ```
    pub fn start_point(&self) -> Result<Point3> {
        let geometry = self.geometry()?;
        let at = if geometry.reversed {
            geometry.range.end
        } else {
            geometry.range.start
        };
        Ok(self.point_at(at)?.point)
    }

    /// Where this edge ends, in the direction it is used.
    ///
    /// # Errors
    /// [`Error::Operation`] if the kernel cannot read it.
    ///
    /// # Examples
    /// ```
    /// use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let edge = Edge::line([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])?;
    /// assert!(edge.end_point()?.distance([4.0, 5.0, 6.0].into()) < 1e-9);
    /// # Ok(())
    /// # }
    /// ```
    pub fn end_point(&self) -> Result<Point3> {
        let geometry = self.geometry()?;
        let at = if geometry.reversed {
            geometry.range.start
        } else {
            geometry.range.end
        };
        Ok(self.point_at(at)?.point)
    }

    /// The closest point of this edge to `to`, which may be one of its ends.
    ///
    /// # Errors
    /// [`Error::InvalidInput`] if the point is not finite; [`Error::Operation`] if the kernel
    /// cannot find it, as for a degenerate edge, which has no curve.
    ///
    /// # Examples
    /// ```
    /// use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let edge = Edge::line([0.0, 0.0, 0.0], [2.0, 0.0, 0.0])?;
    /// let foot = edge.project([0.5, 1.0, 0.0])?;
    /// assert!((foot.parameter - 0.5).abs() < 1e-9);
    /// assert!(edge.project([9.0, 1.0, 0.0])?.point.distance([2.0, 0.0, 0.0].into()) < 1e-9);
    /// # Ok(())
    /// # }
    /// ```
    pub fn project(&self, to: impl Into<Point3>) -> Result<CurveProjection> {
        let (parameter, point) =
            Active::project_onto_edge(self.raw(), point("a projected point", to.into())?)?;
        Ok(CurveProjection { parameter, point })
    }
}

// --- curves ---------------------------------------------------------------------------

/// A smooth curve through or near points, as plain data.
///
/// Use [`Spline::through`] when the curve must hit every point, and [`Spline::fit`] for
/// noisy or dense data it only has to stay close to.
///
/// # Examples
/// ```
/// use occt::prelude::*;
/// use occt::{Curve, Spline};
///
/// # fn main() -> occt::Result<()> {
/// let curve = Spline::through([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 0.0, 0.0]]).build()?;
/// assert!(matches!(curve, Curve::BSpline(_)));
/// let edge = Edge::from_curve(curve).build()?;
/// assert!(edge.length()? > 2.0);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq)]
#[must_use = "a builder does nothing until `build` is called"]
pub struct Spline {
    points: Vec<Point3>,
    closed: bool,
}

impl Spline {
    /// A curve through the points, in order.
    pub fn through(points: impl IntoIterator<Item: Into<Point3>>) -> Self {
        Self {
            points: points.into_iter().map(Into::into).collect(),
            closed: false,
        }
    }

    /// A curve staying within a tolerance of the points, in order. It may use far fewer
    /// control points than there are points.
    pub fn fit(points: impl IntoIterator<Item: Into<Point3>>) -> SplineFit {
        SplineFit {
            points: points.into_iter().map(Into::into).collect(),
            tolerance: 1e-3,
        }
    }

    /// Makes the curve loop back to its first point smoothly. The points need not repeat the
    /// first one at the end, but may.
    pub fn closed(mut self) -> Self {
        self.closed = true;
        self
    }

    /// Computes the curve.
    ///
    /// # Errors
    /// [`Error::InvalidInput`] if a point is not finite, there are too few (2 for an open
    /// curve, 3 for a closed one), or two neighbours coincide. [`Error::Operation`] if the
    /// kernel can't find a curve.
    ///
    /// # Examples
    /// ```
    /// use occt::Spline;
    /// # fn main() -> occt::Result<()> {
    /// let square = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]];
    /// assert!(Spline::through(square).closed().build().is_ok());
    /// assert!(Spline::through([[0.0, 0.0, 0.0]]).build().is_err());
    /// # Ok(())
    /// # }
    /// ```
    pub fn build(self) -> Result<Curve> {
        let mut values = points("a spline point", self.points)?;
        if self.closed
            && values.len() > 1
            && values[0].distance(values[values.len() - 1]) < COINCIDENT
        {
            values.pop();
        }
        let minimum = if self.closed { 3 } else { 2 };
        if values.len() < minimum {
            return Err(invalid(format!(
                "this spline needs at least {minimum} points, got {}",
                values.len()
            )));
        }
        distinct_neighbours("a spline's points", &values)?;
        Active::interpolate_curve(&values, self.closed)
    }
}

/// A builder for a curve approximating points, from [`Spline::fit`].
///
/// # Examples
/// ```
/// use occt::Spline;
/// # fn main() -> occt::Result<()> {
/// let wave: Vec<[f64; 3]> = (0..50)
///     .map(|i| f64::from(i) * 0.1)
///     .map(|x| [x, x.sin(), 0.0])
///     .collect();
/// let curve = Spline::fit(wave).tolerance(1e-2).build()?;
/// # let _ = curve;
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq)]
#[must_use = "a builder does nothing until `build` is called"]
pub struct SplineFit {
    points: Vec<Point3>,
    tolerance: f64,
}

impl SplineFit {
    /// How far the curve may pass from any point. The default is 1e-3.
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Computes the curve. Every point is within the tolerance of it.
    ///
    /// # Errors
    /// [`Error::InvalidInput`] if a point or the tolerance is not finite, the tolerance is
    /// not positive, there are fewer than 2 points, or all the points coincide.
    /// [`Error::Operation`] if no curve meets the tolerance.
    ///
    /// # Examples
    /// ```
    /// use occt::Spline;
    /// # fn main() -> occt::Result<()> {
    /// let points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]];
    /// assert!(Spline::fit(points).build().is_ok());
    /// assert!(Spline::fit(points).tolerance(-1.0).build().is_err());
    /// # Ok(())
    /// # }
    /// ```
    pub fn build(self) -> Result<Curve> {
        let tolerance = positive("a fit's tolerance", self.tolerance)?;
        let values = points("a spline point", self.points)?;
        if values.len() < 2 {
            return Err(invalid(format!(
                "a fit needs at least 2 points, got {}",
                values.len()
            )));
        }
        if values.iter().all(|p| p.distance(values[0]) < COINCIDENT) {
            return Err(invalid("a fit's points all coincide"));
        }
        Active::fit_curve(&values, tolerance)
    }
}

// --- wires ----------------------------------------------------------------------------

fn chain(mut values: Vec<Point3>, closed: bool) -> Result<Wire> {
    values = points("a wire's point", values)?;
    if closed && values.len() > 1 && values[0].distance(values[values.len() - 1]) < COINCIDENT {
        values.pop();
    }
    let minimum = if closed { 3 } else { 2 };
    if values.len() < minimum {
        return Err(invalid(format!(
            "this wire needs at least {minimum} points, got {}",
            values.len()
        )));
    }
    let mut edges = values
        .windows(2)
        .map(|w| Edge::line(w[0], w[1]))
        .collect::<Result<Vec<_>>>()?;
    if closed {
        edges.push(Edge::line(values[values.len() - 1], values[0])?);
    }
    Wire::new(edges)
}

impl Wire {
    /// A wire from connected edges, given in any order and either way round.
    ///
    /// The edges of the result run in order along the wire, so [`edges`](Shape::edges)
    /// walks it. Ends that are within tolerance of each other but aren't one vertex get
    /// merged, so the wire's edges may be new ones rather than those passed in.
    ///
    /// # Errors
    /// [`Error::InvalidInput`] if there are no edges. [`Error::Operation`] if they don't form
    /// a single chain, because of a gap or a branch, or if an edge is given twice.
    ///
    /// # Examples
    /// ```
    /// use occt::prelude::*;
    /// use occt::Wire;
    /// # fn main() -> occt::Result<()> {
    /// let a = Edge::line([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])?;
    /// let b = Edge::line([1.0, 0.0, 0.0], [1.0, 1.0, 0.0])?;
    /// let c = Edge::line([0.0, 0.0, 0.0], [1.0, 1.0, 0.0])?;
    /// let wire = Wire::new([&b, &c, &a])?;
    /// assert!(wire.is_closed()?);
    /// assert_eq!(wire.edges()?.count(), 3);
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(edges: impl IntoIterator<Item: Borrow<Edge>>) -> Result<Self> {
        let edges: Vec<_> = edges.into_iter().collect();
        if edges.is_empty() {
            return Err(invalid("a wire needs at least one edge"));
        }
        let raw: Vec<_> = edges.iter().map(|e| e.borrow().raw()).collect();
        typed(Active::make_wire(&raw)?)
    }

    /// A closed wire of straight edges through the points, with one more edge back to the
    /// first. The last point may repeat the first.
    ///
    /// # Errors
    /// [`Error::InvalidInput`] if a point is not finite, there are fewer than 3, or two
    /// neighbours coincide.
    ///
    /// # Examples
    /// ```
    /// use occt::Wire;
    /// # fn main() -> occt::Result<()> {
    /// let triangle = Wire::polygon([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])?;
    /// assert!(triangle.is_closed()?);
    /// # Ok(())
    /// # }
    /// ```
    pub fn polygon(points: impl IntoIterator<Item: Into<Point3>>) -> Result<Self> {
        chain(points.into_iter().map(Into::into).collect(), true)
    }

    /// An open wire of straight edges through the points, in order.
    ///
    /// # Errors
    /// [`Error::InvalidInput`] if a point is not finite, there are fewer than 2, or two
    /// neighbours coincide.
    ///
    /// # Examples
    /// ```
    /// use occt::Wire;
    /// # fn main() -> occt::Result<()> {
    /// let path = Wire::polyline([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])?;
    /// assert!(!path.is_closed()?);
    /// assert!((path.length()? - 2.0).abs() < 1e-9);
    /// # Ok(())
    /// # }
    /// ```
    pub fn polyline(points: impl IntoIterator<Item: Into<Point3>>) -> Result<Self> {
        chain(points.into_iter().map(Into::into).collect(), false)
    }

    /// A closed wire of one circle. See [`Edge::circle`].
    ///
    /// # Errors
    /// See [`Edge::circle`].
    ///
    /// # Examples
    /// ```
    /// use occt::prelude::*;
    /// use occt::Wire;
    /// # fn main() -> occt::Result<()> {
    /// let ring = Wire::circle([0.0, 0.0, 0.0], Dir3::Z, 2.0)?;
    /// assert!(ring.is_closed()?);
    /// # Ok(())
    /// # }
    /// ```
    pub fn circle(center: impl Into<Point3>, normal: impl Into<Vec3>, radius: f64) -> Result<Self> {
        Self::new([Edge::circle(center, normal, radius)?])
    }

    /// A closed rectangle in the XY plane, centred on the origin. Move it with the
    /// transforms on [`Shape`].
    ///
    /// # Errors
    /// [`Error::InvalidInput`] if a side is not finite and positive.
    ///
    /// # Examples
    /// ```
    /// use occt::Wire;
    /// # fn main() -> occt::Result<()> {
    /// let frame = Wire::rectangle(4.0, 2.0)?;
    /// assert!((frame.length()? - 12.0).abs() < 1e-9);
    /// # Ok(())
    /// # }
    /// ```
    pub fn rectangle(width: f64, height: f64) -> Result<Self> {
        let (x, y) = (
            positive("a rectangle's width", width)? / 2.0,
            positive("a rectangle's height", height)? / 2.0,
        );
        Self::polygon([[-x, -y, 0.0], [x, -y, 0.0], [x, y, 0.0], [-x, y, 0.0]])
    }

    /// The total length of the wire's edges.
    ///
    /// # Errors
    /// [`Error::Operation`] if it cannot be computed.
    ///
    /// # Examples
    /// ```
    /// use occt::Wire;
    /// # fn main() -> occt::Result<()> {
    /// let path = Wire::polyline([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])?;
    /// assert!((path.length()? - 5.0).abs() < 1e-9);
    /// # Ok(())
    /// # }
    /// ```
    pub fn length(&self) -> Result<f64> {
        Active::length(self.raw())
    }

    /// The wires parallel to this one at `distance`, within its plane.
    ///
    /// A closed wire grows outward for positive distances and shrinks for negative ones.
    /// Convex corners round off when growing. Shrinking can pinch a shape into several
    /// loops, which is why this returns a list. An open wire offsets to one side, and
    /// the sign chooses which; it needs a plane, so a straight one can't be offset.
    ///
    /// # Errors
    /// [`Error::InvalidInput`] if the distance is zero or not finite. [`Error::Operation`] if
    /// the wire isn't planar or nothing is left at this distance.
    ///
    /// # Examples
    /// ```
    /// use occt::Wire;
    /// # fn main() -> occt::Result<()> {
    /// let square = Wire::rectangle(10.0, 10.0)?;
    /// let inner = square.offset(-1.0)?;
    /// assert_eq!(inner.len(), 1);
    /// assert!((inner[0].length()? - 32.0).abs() < 1e-6);
    /// # Ok(())
    /// # }
    /// ```
    pub fn offset(&self, distance: f64) -> Result<Vec<Self>> {
        if finite("an offset distance", distance)? == 0.0 {
            return Err(invalid("an offset distance must not be zero"));
        }
        let raw = Active::offset_wire(self.raw(), distance)?;
        Ok(Shape::from_raw(raw).wires()?.collect())
    }
}

// --- faces ----------------------------------------------------------------------------

/// A builder for a flat face, from [`Face::planar`].
///
/// # Examples
/// ```
/// use occt::prelude::*;
/// use occt::Wire;
/// # fn main() -> occt::Result<()> {
/// let outer = Wire::rectangle(4.0, 4.0)?;
/// let hole = Wire::rectangle(1.0, 1.0)?;
/// let face = Face::planar(&outer).hole(&hole).build()?;
/// assert!((face.area()? - 15.0).abs() < 1e-9);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug)]
#[must_use = "a builder does nothing until `build` is called"]
pub struct PlanarFace {
    outer: Wire,
    holes: Vec<Wire>,
}

impl PlanarFace {
    /// Adds a hole. Its direction doesn't matter.
    pub fn hole(mut self, hole: impl Borrow<Wire>) -> Self {
        self.holes.push(hole.borrow().clone());
        self
    }

    /// Adds holes. Their directions don't matter.
    pub fn holes(mut self, holes: impl IntoIterator<Item: Borrow<Wire>>) -> Self {
        self.holes
            .extend(holes.into_iter().map(|h| h.borrow().clone()));
        self
    }

    /// Builds the face. Its normal follows the outer wire counterclockwise.
    ///
    /// # Errors
    /// [`Error::Operation`] if a wire is open, crosses itself or is out of the plane, or a
    /// hole isn't inside the outer wire or overlaps another.
    ///
    /// # Examples
    /// ```
    /// use occt::prelude::*;
    /// use occt::Wire;
    /// # fn main() -> occt::Result<()> {
    /// let open = Wire::polyline([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])?;
    /// assert!(Face::planar(&open).build().is_err());
    /// # Ok(())
    /// # }
    /// ```
    pub fn build(self) -> Result<Face> {
        let holes: Vec<_> = self.holes.iter().map(|h| h.raw()).collect();
        typed(Active::make_planar_face(self.outer.raw(), &holes)?)
    }
}

/// A builder for a face on a surface, from [`Face::from_surface`].
///
/// # Examples
/// ```
/// use occt::prelude::*;
/// use occt::{Frame, Surface};
/// use std::f64::consts::PI;
/// # fn main() -> occt::Result<()> {
/// let cylinder = Surface::Cylinder { frame: Frame::WORLD, radius: 2.0 };
/// let quarter = Face::from_surface(cylinder).bounds(0.0..PI / 2.0, 0.0..3.0).build()?;
/// assert!((quarter.area()? - PI * 3.0).abs() < 1e-6);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug)]
#[must_use = "a builder does nothing until `build` is called"]
pub struct SurfaceFace {
    surface: Surface,
    bounds: Option<(Range<f64>, Range<f64>)>,
    outer: Option<Wire>,
    holes: Vec<Wire>,
}

impl SurfaceFace {
    /// Bounds the face by parameter ranges along U and V. Use this or
    /// [`outer`](SurfaceFace::outer), not both.
    pub fn bounds(mut self, u: Range<f64>, v: Range<f64>) -> Self {
        self.bounds = Some((u, v));
        self
    }

    /// Bounds the face by a closed wire lying on the surface.
    pub fn outer(mut self, outer: impl Borrow<Wire>) -> Self {
        self.outer = Some(outer.borrow().clone());
        self
    }

    /// Adds a hole: a closed wire lying on the surface, inside the outer boundary.
    pub fn hole(mut self, hole: impl Borrow<Wire>) -> Self {
        self.holes.push(hole.borrow().clone());
        self
    }

    /// Builds the face.
    ///
    /// With neither bounds nor an outer wire, it covers the whole surface, which needs a
    /// bounded surface such as a sphere.
    ///
    /// # Errors
    /// [`Error::InvalidInput`] if the surface's data is not consistent or not finite or
    /// has no description, both bounds and a wire are given, holes have no outer
    /// wire, or a range runs backwards. [`Error::Operation`] if the bounds fall outside the
    /// surface or are missing on an unbounded one, or a wire is open, or off the
    /// surface, or doesn't make a single valid face.
    ///
    /// # Examples
    /// ```
    /// use occt::prelude::*;
    /// use occt::{Frame, Surface};
    /// # fn main() -> occt::Result<()> {
    /// let ball = Face::from_surface(Surface::Sphere { frame: Frame::WORLD, radius: 1.0 }).build()?;
    /// assert!((ball.area()? - 4.0 * std::f64::consts::PI).abs() < 1e-6);
    /// assert!(Face::from_surface(Surface::Plane { frame: Frame::WORLD }).build().is_err());
    /// # Ok(())
    /// # }
    /// ```
    pub fn build(self) -> Result<Face> {
        check_surface(&self.surface)?;
        if self.bounds.is_some() && self.outer.is_some() {
            return Err(invalid(
                "a face is bounded by parameter ranges or by a wire, not both",
            ));
        }
        if self.outer.is_none() && !self.holes.is_empty() {
            return Err(invalid("a face with holes needs an outer wire"));
        }
        let bounds = match &self.bounds {
            Some((u, v)) => Some((
                param_range("a face's U range", u)?,
                param_range("a face's V range", v)?,
            )),
            None => None,
        };
        let holes: Vec<_> = self.holes.iter().map(|h| h.raw()).collect();
        let outer = self.outer.as_ref().map(|w| w.raw());
        typed(Active::make_surface_face(
            &self.surface,
            bounds,
            outer,
            &holes,
        )?)
    }
}

/// A builder for a smooth patch filling a loop of edges, from [`Face::fill`].
///
/// # Examples
/// ```
/// use occt::prelude::*;
/// # fn main() -> occt::Result<()> {
/// let a = Edge::line([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])?;
/// let b = Edge::line([1.0, 0.0, 0.0], [1.0, 1.0, 0.0])?;
/// let c = Edge::line([1.0, 1.0, 0.0], [0.0, 1.0, 0.0])?;
/// let d = Edge::line([0.0, 1.0, 0.0], [0.0, 0.0, 0.0])?;
/// let patch = Face::fill([&a, &b, &c, &d]).through([[0.5, 0.5, 0.2]]).build()?;
/// assert!(patch.area()? > 1.0);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug)]
#[must_use = "a builder does nothing until `build` is called"]
pub struct BoundaryFill {
    edges: Vec<Edge>,
    tangent: Vec<(Edge, Face)>,
    through: Vec<Point3>,
}

impl BoundaryFill {
    /// A fill whose edges all come from [`tangent_to`](BoundaryFill::tangent_to).
    pub fn new() -> Self {
        BoundaryFill {
            edges: Vec::new(),
            tangent: Vec::new(),
            through: Vec::new(),
        }
    }

    /// Adds a boundary edge of `face` that the patch leaves tangent to, within a fraction of
    /// a degree. The edges beside it have to leave it within the face's tangent plane.
    ///
    /// This stops at tangency. For matching curvature too, use [`Face::blend`].
    ///
    /// # Examples
    /// ```
    /// use occt::BoundaryFill;
    /// use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// // A curved hole in a cylinder of radius 5: four neighbours and the edges facing it.
    /// let cylinder = Surface::Cylinder { frame: Frame::WORLD, radius: 5.0 };
    /// let patch = |u: std::ops::Range<f64>, v: std::ops::Range<f64>| {
    ///     Face::from_surface(cylinder.clone()).bounds(u, v).build()
    /// };
    /// let around: [(Face, [f64; 2]); 4] = [
    ///     (patch(-1.5..-0.5, -2.0..2.0)?, [-0.5, 0.0]),
    ///     (patch(0.5..1.5, -2.0..2.0)?, [0.5, 0.0]),
    ///     (patch(-0.5..0.5, -4.0..-2.0)?, [0.0, -2.0]),
    ///     (patch(-0.5..0.5, 2.0..4.0)?, [0.0, 2.0]),
    /// ];
    /// let mut fill = BoundaryFill::new();
    /// let mut joints = Vec::new();
    /// for (face, [u, v]) in &around {
    ///     let target = [5.0 * u.cos(), 5.0 * u.sin(), *v];
    ///     let mut closest = None;
    ///     for edge in face.edges()? {
    ///         // Each edge is straight or an arc, so its middle is enough to tell them apart.
    ///         let range = edge.geometry()?.range;
    ///         let middle = edge.point_at(0.5 * (range.start + range.end))?.point;
    ///         let distance = middle.distance(target.into());
    ///         if closest.as_ref().is_none_or(|(d, _)| distance < *d) {
    ///             closest = Some((distance, edge));
    ///         }
    ///     }
    ///     let (_, edge) = closest.expect("a face has edges");
    ///     fill = fill.tangent_to(&edge, face);
    ///     joints.push((edge, face));
    /// }
    /// let cover = fill.build()?;
    /// for (edge, face) in joints {
    ///     assert!(cover.continuity_with(face, &edge)?.angle.as_degrees() < 0.5);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn tangent_to(mut self, edge: &Edge, face: &Face) -> Self {
        self.tangent.push((edge.clone(), face.clone()));
        self
    }

    /// Makes the patch pass through these points as well.
    pub fn through(mut self, points: impl IntoIterator<Item: Into<Point3>>) -> Self {
        self.through.extend(points.into_iter().map(Into::into));
        self
    }

    /// Builds the patch. It follows the edges and points closely but generally not exactly:
    /// it is a smooth approximation. Its accuracy shows in its [`tolerance`](Shape::tolerance).
    ///
    /// # Errors
    /// [`Error::InvalidInput`] if a point is not finite or there are no edges.
    /// [`Error::Operation`] if the edges don't form one closed loop, or no valid patch meets
    /// the points.
    ///
    /// # Examples
    /// ```
    /// use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let open = Edge::line([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])?;
    /// assert!(Face::fill([&open]).build().is_err());
    /// # Ok(())
    /// # }
    /// ```
    pub fn build(self) -> Result<Face> {
        if self.edges.is_empty() && self.tangent.is_empty() {
            return Err(invalid("filling needs at least one edge"));
        }
        let through = points("a filling's point", self.through)?;
        if self.tangent.is_empty() {
            let edges: Vec<_> = self.edges.iter().map(|e| e.raw()).collect();
            return typed(Active::fill_boundary(&edges, &through)?);
        }
        let free = self.edges.iter().map(|edge| FillEdge {
            edge: edge.raw(),
            tangent_to: None,
        });
        let tangent = self.tangent.iter().map(|(edge, face)| FillEdge {
            edge: edge.raw(),
            tangent_to: Some(face.raw()),
        });
        let sides: Vec<_> = tangent.chain(free).collect();
        typed(Active::fill_boundary_tangent(&sides, &through)?)
    }
}

impl Default for BoundaryFill {
    fn default() -> Self {
        Self::new()
    }
}

/// Where a point lands on a face's surface, from [`Face::project`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceProjection {
    /// The U parameter of the closest point.
    pub u: f64,
    /// The V parameter of the closest point.
    pub v: f64,
    /// The closest point of the surface.
    pub point: Point3,
}

impl Face {
    /// Starts a flat face bounded by a closed wire. Add holes on the returned builder.
    ///
    /// # Examples
    /// ```
    /// use occt::prelude::*;
    /// use occt::Wire;
    /// # fn main() -> occt::Result<()> {
    /// let outer = Wire::rectangle(3.0, 2.0)?;
    /// assert!((Face::planar(&outer).build()?.area()? - 6.0).abs() < 1e-9);
    /// # Ok(())
    /// # }
    /// ```
    pub fn planar(outer: impl Borrow<Wire>) -> PlanarFace {
        PlanarFace {
            outer: outer.borrow().clone(),
            holes: Vec::new(),
        }
    }

    /// Starts a face on a surface. Bound it on the returned builder.
    ///
    /// Any surface read from [`Face::surface`] other than [`Surface::Other`] can be passed
    /// back.
    ///
    /// # Examples
    /// ```
    /// use occt::prelude::*;
    /// use occt::{Frame, Surface};
    /// # fn main() -> occt::Result<()> {
    /// let plane = Surface::Plane { frame: Frame::WORLD };
    /// let face = Face::from_surface(plane).bounds(0.0..2.0, 0.0..3.0).build()?;
    /// assert!((face.area()? - 6.0).abs() < 1e-9);
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_surface(surface: Surface) -> SurfaceFace {
        SurfaceFace {
            surface,
            bounds: None,
            outer: None,
            holes: Vec::new(),
        }
    }

    /// Starts a smooth patch filling a closed loop of edges, given in any order.
    ///
    /// # Examples
    /// ```
    /// use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let circle = Edge::circle([0.0, 0.0, 0.0], Dir3::Z, 1.0)?;
    /// let disc = Face::fill([&circle]).build()?;
    /// assert!((disc.area()? - std::f64::consts::PI).abs() < 1e-2);
    /// # Ok(())
    /// # }
    /// ```
    pub fn fill(edges: impl IntoIterator<Item: Borrow<Edge>>) -> BoundaryFill {
        BoundaryFill {
            edges: edges.into_iter().map(|e| e.borrow().clone()).collect(),
            tangent: Vec::new(),
            through: Vec::new(),
        }
    }

    /// The exact geometry under this face: its surface, parameter bounds and direction.
    ///
    /// A surface the kernel holds in a form [`Surface`] has no description of comes back as
    /// [`Surface::Other`], and [`point_at`](Face::point_at) still evaluates it.
    ///
    /// # Errors
    /// [`Error::Operation`] if the kernel cannot read it.
    ///
    /// # Examples
    /// ```
    /// use occt::prelude::*;
    /// use occt::Surface;
    /// # fn main() -> occt::Result<()> {
    /// let ball = Sphere::new(2.0).build()?;
    /// let face = ball.faces()?.next().expect("a sphere has a face");
    /// assert!(matches!(face.geometry()?.surface, Surface::Sphere { radius, .. } if radius == 2.0));
    /// # Ok(())
    /// # }
    /// ```
    pub fn geometry(&self) -> Result<FaceGeometry> {
        Active::face_geometry(self.raw())
    }

    /// The surface under this face, whole.
    ///
    /// # Errors
    /// [`Error::Operation`] if the kernel cannot read it.
    ///
    /// # Examples
    /// ```
    /// use occt::prelude::*;
    /// use occt::Surface;
    /// # fn main() -> occt::Result<()> {
    /// let cube = Cuboid::new([1.0, 1.0, 1.0]).build()?;
    /// let face = cube.faces()?.next().expect("a cube has faces");
    /// assert!(matches!(face.surface()?, Surface::Plane { .. }));
    /// # Ok(())
    /// # }
    /// ```
    pub fn surface(&self) -> Result<Surface> {
        Ok(self.geometry()?.surface)
    }

    /// Evaluates this face's surface at parameters: position, outward normal, first
    /// derivatives and principal curvatures.
    ///
    /// Curvature is signed against the outward normal: negative where the face curves away
    /// from it, as on the outside of a ball. It is NaN, and the normal `None`, where the
    /// surface is singular, as at a sphere's pole.
    ///
    /// # Errors
    /// [`Error::InvalidInput`] if a parameter is not finite; [`Error::Operation`] if it is
    /// outside the surface.
    ///
    /// # Examples
    /// ```
    /// use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let ball = Sphere::new(2.0).build()?;
    /// let face = ball.faces()?.next().expect("a sphere has a face");
    /// let at = face.point_at(0.0, 0.0)?;
    /// let normal = at.normal.expect("the equator is regular");
    /// assert!(normal.vector().dot(at.point - Point3::ORIGIN) > 0.0);
    /// assert!((at.principal_curvatures[0] + 0.5).abs() < 1e-9);
    /// # Ok(())
    /// # }
    /// ```
    pub fn point_at(&self, u: f64, v: f64) -> Result<SurfacePoint> {
        Active::face_point(
            self.raw(),
            finite("a surface parameter", u)?,
            finite("a surface parameter", v)?,
        )
    }

    /// The closest point of this face's surface to `to`, and its parameters.
    ///
    /// This looks at the whole surface, past the face's own boundary. Where several points
    /// tie, as from a sphere's centre, it returns one of them.
    ///
    /// # Errors
    /// [`Error::InvalidInput`] if the point is not finite; [`Error::Operation`] if the kernel
    /// cannot find it.
    ///
    /// # Examples
    /// ```
    /// use occt::prelude::*;
    /// use occt::Wire;
    /// # fn main() -> occt::Result<()> {
    /// let square = Wire::rectangle(2.0, 2.0)?;
    /// let face = Face::planar(&square).build()?;
    /// let foot = face.project([0.5, 0.5, 3.0])?;
    /// assert!(foot.point.distance([0.5, 0.5, 0.0].into()) < 1e-9);
    /// # Ok(())
    /// # }
    /// ```
    pub fn project(&self, to: impl Into<Point3>) -> Result<SurfaceProjection> {
        let (u, v, point) =
            Active::project_onto_face(self.raw(), point("a projected point", to.into())?)?;
        Ok(SurfaceProjection { u, v, point })
    }
}

// --- sewing, solids, groups -----------------------------------------------------------

/// Joins faces and shells along boundaries that coincide, into one shape.
///
/// # Examples
/// ```
/// use occt::prelude::*;
/// use occt::{Sewing, Shell};
/// # fn main() -> occt::Result<()> {
/// // Two squares meeting along an edge, as separate faces.
/// let a = Polygon::new([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]).build()?;
/// let b = Polygon::new([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [1.0, 1.0, 0.0]]).build()?;
/// let sheet = Shell::try_from(Sewing::new().add(&a).add(&b).build()?)?;
/// assert_eq!(sheet.faces()?.count(), 2);
/// assert_eq!(sheet.edges()?.count(), 7);
/// assert!(!sheet.is_closed()?);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug)]
#[must_use = "a builder does nothing until `build` is called"]
pub struct Sewing {
    shapes: Vec<Shape>,
    tolerance: f64,
}

impl Default for Sewing {
    fn default() -> Self {
        Self::new()
    }
}

impl Sewing {
    /// Starts with no shapes and a tolerance of 1e-6.
    pub fn new() -> Self {
        Self {
            shapes: Vec::new(),
            tolerance: 1e-6,
        }
    }

    /// How far apart two boundaries can be and still join.
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Adds a face, a shell or anything holding them.
    #[allow(
        clippy::should_implement_trait,
        reason = "a builder step, not arithmetic"
    )]
    pub fn add(mut self, shape: impl AsRef<Shape>) -> Self {
        self.shapes.push(shape.as_ref().clone());
        self
    }

    /// Adds faces, shells or anything holding them.
    pub fn add_all(mut self, shapes: impl IntoIterator<Item: AsRef<Shape>>) -> Self {
        self.shapes
            .extend(shapes.into_iter().map(|s| s.as_ref().clone()));
        self
    }

    /// Sews the shapes together.
    ///
    /// The result is a shell if everything joined into one, a face if there was only one
    /// face, or a compound of those if parts stayed apart. Check
    /// [`is_closed`](Shape::is_closed) on a shell to see if any boundary is left.
    ///
    /// # Errors
    /// [`Error::InvalidInput`] if there are no shapes, or the tolerance is not finite and
    /// positive. [`Error::Operation`] if a shape holds no faces.
    ///
    /// # Examples
    /// ```
    /// use occt::prelude::*;
    /// use occt::Sewing;
    /// # fn main() -> occt::Result<()> {
    /// assert!(Sewing::new().build().is_err());
    /// let edge = Edge::line([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])?;
    /// assert!(Sewing::new().add(&edge).build().is_err());
    /// # Ok(())
    /// # }
    /// ```
    pub fn build(self) -> Result<Shape> {
        let tolerance = positive("a sewing tolerance", self.tolerance)?;
        if self.shapes.is_empty() {
            return Err(invalid("sewing needs at least one shape"));
        }
        let raw: Vec<_> = self.shapes.iter().map(Shape::raw).collect();
        Ok(Shape::from_raw(Active::sew(&raw, tolerance)?))
    }

    /// Sews the shapes, then makes a solid of the result.
    ///
    /// # Errors
    /// As for [`build`](Sewing::build) and [`Solid::from_shell`], and [`Error::WrongKind`] if
    /// the shapes don't join into a single shell.
    ///
    /// # Examples
    /// ```
    /// use occt::prelude::*;
    /// use occt::Sewing;
    /// # fn main() -> occt::Result<()> {
    /// let cube = Cuboid::new([2.0, 2.0, 2.0]).build()?;
    /// let solid = Sewing::new().add_all(cube.faces()?).build_solid()?;
    /// assert!((solid.volume()? - 8.0).abs() < 1e-6);
    /// # Ok(())
    /// # }
    /// ```
    pub fn build_solid(self) -> Result<Solid> {
        Solid::from_shell(Shell::try_from(self.build()?)?)
    }
}

impl Solid {
    /// A solid enclosed by a closed shell, facing outward so its volume is positive, even if
    /// the shell's faces were facing inward or disagreed.
    ///
    /// # Errors
    /// [`Error::Operation`] if the shell isn't closed or encloses no volume.
    ///
    /// # Examples
    /// ```
    /// use occt::prelude::*;
    /// use occt::Shell;
    /// # fn main() -> occt::Result<()> {
    /// let cube = Cuboid::new([2.0, 2.0, 2.0]).build()?;
    /// let skin: Shell = cube.shells()?.next().expect("a box has a shell");
    /// let copy = Solid::from_shell(&skin)?;
    /// assert!((copy.volume()? - 8.0).abs() < 1e-6);
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_shell(shell: impl Borrow<Shell>) -> Result<Self> {
        typed(Active::make_solid(shell.borrow().raw())?)
    }
}

impl Compound {
    /// A group holding the shapes, in any mix of kinds. It may be empty.
    ///
    /// # Errors
    /// [`Error::Operation`] if the kernel fails.
    ///
    /// # Examples
    /// ```
    /// use occt::prelude::*;
    /// use occt::Compound;
    /// # fn main() -> occt::Result<()> {
    /// let a = Cuboid::new([1.0, 1.0, 1.0]).build()?;
    /// let b = Sphere::new(1.0).at([5.0, 0.0, 0.0]).build()?;
    /// let group = Compound::new([&a, &b])?;
    /// assert_eq!(group.solids()?.count(), 2);
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(shapes: impl IntoIterator<Item: AsRef<Shape>>) -> Result<Self> {
        let shapes: Vec<_> = shapes.into_iter().collect();
        let raw: Vec<_> = shapes.iter().map(|s| s.as_ref().raw()).collect();
        typed(Active::make_compound(&raw)?)
    }
}

impl Shape {
    /// Whether this edge, wire or shell closes on itself, leaving no free end or boundary.
    ///
    /// # Errors
    /// [`Error::InvalidInput`] if this is another kind of shape.
    ///
    /// # Examples
    /// ```
    /// use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// assert!(!Edge::line([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])?.is_closed()?);
    /// let cube = Cuboid::new([1.0, 1.0, 1.0]).build()?;
    /// assert!(cube.shells()?.next().expect("a box has a shell").is_closed()?);
    /// assert!(cube.is_closed().is_err());
    /// # Ok(())
    /// # }
    /// ```
    pub fn is_closed(&self) -> Result<bool> {
        match self.kind()? {
            ShapeKind::Edge | ShapeKind::Wire | ShapeKind::Shell => Active::is_closed(self.raw()),
            other => Err(invalid(format!(
                "only an edge, wire or shell can be closed or open, not a {other}"
            ))),
        }
    }

    /// The largest geometric tolerance stored on this shape: how far apart things it treats
    /// as touching may be. It says how exact the shape is, and is 0 if it holds no geometry.
    ///
    /// # Errors
    /// [`Error::Operation`] if it cannot be read.
    ///
    /// # Examples
    /// ```
    /// use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let cube = Cuboid::new([1.0, 1.0, 1.0]).build()?;
    /// assert!(cube.tolerance()? < 1e-5);
    /// # Ok(())
    /// # }
    /// ```
    pub fn tolerance(&self) -> Result<f64> {
        Active::tolerance(self.raw())
    }
}

// --- more primitives ------------------------------------------------------------------

/// A ring with a round cross section. By default it lies flat around the origin.
///
/// # Examples
/// ```
/// use occt::prelude::*;
/// use occt::Torus;
/// use std::f64::consts::PI;
/// # fn main() -> occt::Result<()> {
/// let ring = Torus::new(3.0, 1.0).at([0.0, 0.0, 5.0]).build()?;
/// assert!((ring.volume()? - 2.0 * PI * PI * 3.0).abs() < 1e-4);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
#[must_use = "a builder does nothing until `build` is called"]
pub struct Torus {
    major: f64,
    minor: f64,
    axis: Axis,
}

impl Torus {
    /// A torus with `major` from the axis to the tube's centre, and a tube of radius `minor`.
    pub fn new(major: f64, minor: f64) -> Self {
        Self {
            major,
            minor,
            axis: Axis::Z,
        }
    }

    /// Centres it on a point, keeping its axis direction.
    pub fn at(mut self, center: impl Into<Point3>) -> Self {
        self.axis.origin = center.into();
        self
    }

    /// Sets the axis it goes around, centring it on the axis's origin.
    pub fn along(mut self, axis: Axis) -> Self {
        self.axis = axis;
        self
    }

    /// Builds the torus.
    ///
    /// # Errors
    /// [`Error::InvalidInput`] if a radius is not finite and positive, `major` isn't above
    /// `minor`, or the position is not finite.
    ///
    /// # Examples
    /// ```
    /// use occt::Torus;
    /// # fn main() -> occt::Result<()> {
    /// assert!(Torus::new(1.0, 2.0).build().is_err());
    /// assert!(Torus::new(2.0, 1.0).build().is_ok());
    /// # Ok(())
    /// # }
    /// ```
    pub fn build(self) -> Result<Solid> {
        let major = positive("a torus's major radius", self.major)?;
        let minor = positive("a torus's minor radius", self.minor)?;
        if major <= minor {
            return Err(invalid(format!(
                "a torus's major radius {major} must be above its minor radius {minor}"
            )));
        }
        let axis = axis("a torus's axis", self.axis)?;
        let frame = Frame::from_normal(axis.origin, axis.direction);
        typed(Active::make_torus(frame, major, minor)?)
    }
}

/// A helical path on a cylinder, as a wire to sweep a profile along.
///
/// It rises along its axis from the axis's origin, right-handed unless told otherwise. It
/// starts at `radius` along the X of [`Frame::from_normal`], so on the Z axis, at
/// `[radius, 0, 0]`.
///
/// # Examples
/// ```
/// use occt::prelude::*;
/// use occt::Helix;
/// use std::f64::consts::TAU;
/// # fn main() -> occt::Result<()> {
/// let path = Helix::new(5.0, 2.0, 6.0).build()?;
/// let turn = (TAU * 5.0f64).hypot(2.0);
/// assert!((path.length()? - 3.0 * turn).abs() < 1e-3);
/// assert!(!path.is_closed()?);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
#[must_use = "a builder does nothing until `build` is called"]
pub struct Helix {
    radius: f64,
    pitch: f64,
    height: f64,
    axis: Axis,
    left_handed: bool,
}

impl Helix {
    /// A helix of this radius rising `pitch` per turn, up to `height` along its axis.
    pub fn new(radius: f64, pitch: f64, height: f64) -> Self {
        Self {
            radius,
            pitch,
            height,
            axis: Axis::Z,
            left_handed: false,
        }
    }

    /// Sets the axis to rise along.
    pub fn along(mut self, axis: Axis) -> Self {
        self.axis = axis;
        self
    }

    /// Makes it turn the other way.
    pub fn left_handed(mut self) -> Self {
        self.left_handed = true;
        self
    }

    /// Builds the wire. It has one edge per turn, joined smoothly, and follows the true helix
    /// to about 1e-7.
    ///
    /// # Errors
    /// [`Error::InvalidInput`] if the radius, pitch or height is not finite and positive, or
    /// the axis is not finite. [`Error::Operation`] if it would have an unreasonable number
    /// of turns.
    ///
    /// # Examples
    /// ```
    /// use occt::Helix;
    /// # fn main() -> occt::Result<()> {
    /// assert!(Helix::new(1.0, 0.0, 1.0).build().is_err());
    /// assert_eq!(Helix::new(1.0, 1.0, 2.5).left_handed().build()?.edges()?.count(), 3);
    /// # Ok(())
    /// # }
    /// ```
    pub fn build(self) -> Result<Wire> {
        let radius = positive("a helix's radius", self.radius)?;
        let pitch = positive("a helix's pitch", self.pitch)?;
        let height = positive("a helix's height", self.height)?;
        let axis = axis("a helix's axis", self.axis)?;
        typed(Active::make_helix(
            axis,
            radius,
            pitch,
            height,
            self.left_handed,
        )?)
    }
}
