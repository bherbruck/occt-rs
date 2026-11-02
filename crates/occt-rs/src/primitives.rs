//! Builders for primitive shapes.
//!
//! Builders hold plain values and check them in `build`, so bad input is always an
//! [`Error::InvalidInput`](crate::Error::InvalidInput) and never a panic.

use occt_kernel::{Axis, Point3, Primitives, Result, Vec3};

use crate::kernel::Active;
use crate::shape::Shape;
use crate::subshape::{Face, Solid};
use crate::validate::{axis, invalid, non_negative, point, positive, vector};

/// A rectangular box.
///
/// By default its lowest corner is at the origin. [`at`](Cuboid::at) moves that corner,
/// and [`centered`](Cuboid::centered) puts the box's centre there instead.
///
/// # Examples
/// ```
/// use occt::prelude::*;
///
/// # fn main() -> occt::Result<()> {
/// let plate = Cuboid::new([80.0, 40.0, 6.0]).at([0.0, 0.0, 10.0]).centered().build()?;
/// assert!(plate.center_of_mass()?.distance([0.0, 0.0, 10.0].into()) < 1e-6);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
#[must_use = "a builder does nothing until `build` is called"]
pub struct Cuboid {
    size: Vec3,
    at: Point3,
    centered: bool,
}

impl Cuboid {
    /// A box of the given size along X, Y and Z.
    pub fn new(size: impl Into<Vec3>) -> Self {
        Self {
            size: size.into(),
            at: Point3::ORIGIN,
            centered: false,
        }
    }

    /// Places the box at a point: its lowest corner, or its centre if
    /// [`centered`](Cuboid::centered).
    pub fn at(mut self, point: impl Into<Point3>) -> Self {
        self.at = point.into();
        self
    }

    /// Makes the placement point the box's centre rather than its lowest corner.
    pub fn centered(mut self) -> Self {
        self.centered = true;
        self
    }

    /// Builds the box.
    ///
    /// # Errors
    /// [`Error::InvalidInput`](crate::Error::InvalidInput) if a side is not finite and
    /// positive or the placement is not finite;
    /// [`Error::Operation`](crate::Error::Operation) if the kernel fails.
    pub fn build(self) -> Result<Solid> {
        let size = vector("box size", self.size)?;
        for side in [size.x, size.y, size.z] {
            positive("box side", side)?;
        }
        let at = point("box position", self.at)?;
        let corner = if self.centered { at - size / 2.0 } else { at };
        Active::make_box(corner, size)
            .map(Shape::from_raw)?
            .try_into()
    }
}

/// A round cylinder. By default it stands on the origin and rises along +Z.
///
/// # Examples
/// ```
/// use occt::prelude::*;
/// use std::f64::consts::PI;
///
/// # fn main() -> occt::Result<()> {
/// let pin = Cylinder::new(5.0, 10.0).along(Axis::X).build()?;
/// assert!((pin.volume()? - PI * 250.0).abs() < 1e-3);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
#[must_use = "a builder does nothing until `build` is called"]
pub struct Cylinder {
    radius: f64,
    height: f64,
    axis: Axis,
}

impl Cylinder {
    /// A cylinder of the given radius and height.
    pub fn new(radius: f64, height: f64) -> Self {
        Self {
            radius,
            height,
            axis: Axis::Z,
        }
    }

    /// Starts the cylinder at the axis origin and runs it along the axis direction.
    pub fn along(mut self, axis: Axis) -> Self {
        self.axis = axis;
        self
    }

    /// Builds the cylinder.
    ///
    /// # Errors
    /// [`Error::InvalidInput`](crate::Error::InvalidInput) if the radius or height is not
    /// finite and positive or the axis origin is not finite;
    /// [`Error::Operation`](crate::Error::Operation) if the kernel fails.
    pub fn build(self) -> Result<Solid> {
        let radius = positive("cylinder radius", self.radius)?;
        let height = positive("cylinder height", self.height)?;
        let axis = axis("cylinder axis", self.axis)?;
        Active::make_cylinder(axis, radius, height)
            .map(Shape::from_raw)?
            .try_into()
    }
}

/// A cone or frustum. By default its base is on the origin and it rises along +Z.
///
/// One radius may be zero, for a pointed cone. Equal radii are not a cone: use
/// [`Cylinder`].
///
/// # Examples
/// ```
/// use occt::prelude::*;
/// use std::f64::consts::PI;
///
/// # fn main() -> occt::Result<()> {
/// let spike = Cone::new(3.0, 0.0, 4.0).build()?;
/// assert!((spike.volume()? - 12.0 * PI).abs() < 1e-3);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
#[must_use = "a builder does nothing until `build` is called"]
pub struct Cone {
    base_radius: f64,
    top_radius: f64,
    height: f64,
    axis: Axis,
}

impl Cone {
    /// A cone with the given radius at its base, radius at its top, and height.
    pub fn new(base_radius: f64, top_radius: f64, height: f64) -> Self {
        Self {
            base_radius,
            top_radius,
            height,
            axis: Axis::Z,
        }
    }

    /// Puts the base at the axis origin and runs the cone along the axis direction.
    pub fn along(mut self, axis: Axis) -> Self {
        self.axis = axis;
        self
    }

    /// Builds the cone.
    ///
    /// # Errors
    /// [`Error::InvalidInput`](crate::Error::InvalidInput) if a radius is negative or not
    /// finite, the radii are equal, the height is not finite and positive, or the axis
    /// origin is not finite; [`Error::Operation`](crate::Error::Operation) if the kernel
    /// fails.
    pub fn build(self) -> Result<Solid> {
        let base = non_negative("cone base radius", self.base_radius)?;
        let top = non_negative("cone top radius", self.top_radius)?;
        #[allow(clippy::float_cmp, reason = "only exactly equal radii are a cylinder")]
        if base == top {
            return Err(invalid(format!("cone radii must differ, both are {base}")));
        }
        let height = positive("cone height", self.height)?;
        let axis = axis("cone axis", self.axis)?;
        Active::make_cone(axis, base, top, height)
            .map(Shape::from_raw)?
            .try_into()
    }
}

/// A sphere. By default it is centred on the origin.
///
/// # Examples
/// ```
/// use occt::prelude::*;
/// use std::f64::consts::PI;
///
/// # fn main() -> occt::Result<()> {
/// let ball = Sphere::new(2.0).at([1.0, 0.0, 0.0]).build()?;
/// assert!((ball.area()? - 16.0 * PI).abs() < 1e-3);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
#[must_use = "a builder does nothing until `build` is called"]
pub struct Sphere {
    radius: f64,
    center: Point3,
}

impl Sphere {
    /// A sphere of the given radius.
    pub fn new(radius: f64) -> Self {
        Self {
            radius,
            center: Point3::ORIGIN,
        }
    }

    /// Centres the sphere on a point.
    pub fn at(mut self, center: impl Into<Point3>) -> Self {
        self.center = center.into();
        self
    }

    /// Builds the sphere.
    ///
    /// # Errors
    /// [`Error::InvalidInput`](crate::Error::InvalidInput) if the radius is not finite and
    /// positive or the centre is not finite;
    /// [`Error::Operation`](crate::Error::Operation) if the kernel fails.
    pub fn build(self) -> Result<Solid> {
        let radius = positive("sphere radius", self.radius)?;
        let center = point("sphere centre", self.center)?;
        Active::make_sphere(center, radius)
            .map(Shape::from_raw)?
            .try_into()
    }
}

/// A flat face bounded by a closed polygon. The last point joins back to the first.
///
/// # Examples
/// ```
/// use occt::prelude::*;
///
/// # fn main() -> occt::Result<()> {
/// let triangle = Polygon::new([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]]).build()?;
/// assert!((triangle.area()? - 6.0).abs() < 1e-6);
/// assert_eq!(triangle.surface_kind()?, SurfaceKind::Plane);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq)]
#[must_use = "a builder does nothing until `build` is called"]
pub struct Polygon {
    points: Vec<Point3>,
}

impl Polygon {
    /// A polygon through the given points, in order.
    pub fn new(points: impl IntoIterator<Item: Into<Point3>>) -> Self {
        Self {
            points: points.into_iter().map(Into::into).collect(),
        }
    }

    /// Builds the face.
    ///
    /// # Errors
    /// [`Error::InvalidInput`](crate::Error::InvalidInput) if there are fewer than three
    /// points or a point is not finite;
    /// [`Error::Operation`](crate::Error::Operation) if the kernel cannot make a face, as
    /// when the points do not lie in one plane.
    pub fn build(self) -> Result<Face> {
        if self.points.len() < 3 {
            return Err(invalid(format!(
                "a polygon needs at least 3 points, got {}",
                self.points.len()
            )));
        }
        for p in &self.points {
            point("polygon point", *p)?;
        }
        Active::make_polygon_face(&self.points)
            .map(Shape::from_raw)?
            .try_into()
    }
}
