//! Transforms and sweeps: new shapes made by moving an existing one.

use std::f64::consts::TAU;

use occt_kernel::{Angle, Axis, Point3, Result, Sweeps, Transforms, Vec3};

use crate::kernel::Active;
use crate::shape::Shape;
use crate::validate::{axis, finite, invalid, point, positive, vector};

/// Slack on the full-turn limit, so that `Angle::degrees(360.0)` passes however it rounds.
const FULL_TURN_SLACK: f64 = 1e-9;

impl Shape {
    /// A copy moved by a vector.
    ///
    /// # Errors
    /// [`Error::InvalidInput`](crate::Error::InvalidInput) if the vector is not finite;
    /// [`Error::Operation`](crate::Error::Operation) if the kernel fails.
    ///
    /// # Examples
    /// ```
    /// # use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let moved = Cuboid::new([2.0, 2.0, 2.0]).build()?.translated([10.0, 0.0, 0.0])?;
    /// assert!(moved.center_of_mass()?.distance([11.0, 1.0, 1.0].into()) < 1e-6);
    /// # Ok(())
    /// # }
    /// ```
    pub fn translated(&self, by: impl Into<Vec3>) -> Result<Shape> {
        let by = vector("translation", by.into())?;
        Active::translate(self.raw(), by).map(Shape::from_raw)
    }

    /// A copy rotated around an axis, by the right-hand rule.
    ///
    /// # Errors
    /// [`Error::InvalidInput`](crate::Error::InvalidInput) if the axis origin or the
    /// angle is not finite; [`Error::Operation`](crate::Error::Operation) if the kernel
    /// fails.
    ///
    /// # Examples
    /// ```
    /// # use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let bar = Cuboid::new([4.0, 1.0, 1.0]).build()?;
    /// let turned = bar.rotated(Axis::Z, Angle::degrees(90.0))?;
    /// assert!((turned.bounding_box()?.size().y - 4.0).abs() < 1e-6);
    /// # Ok(())
    /// # }
    /// ```
    pub fn rotated(&self, around: Axis, angle: Angle) -> Result<Shape> {
        let around = axis("rotation axis", around)?;
        finite("rotation angle", angle.as_radians())?;
        Active::rotate(self.raw(), around, angle).map(Shape::from_raw)
    }

    /// A copy scaled uniformly about a point.
    ///
    /// # Errors
    /// [`Error::InvalidInput`](crate::Error::InvalidInput) if the centre is not finite,
    /// or the factor is not finite and positive; [`Error::Operation`](crate::Error::Operation)
    /// if the kernel fails.
    ///
    /// # Examples
    /// ```
    /// # use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let twice = Cuboid::new([1.0, 1.0, 1.0]).build()?.scaled(Point3::ORIGIN, 2.0)?;
    /// assert!((twice.volume()? - 8.0).abs() < 1e-6);
    /// # Ok(())
    /// # }
    /// ```
    pub fn scaled(&self, center: impl Into<Point3>, factor: f64) -> Result<Shape> {
        let center = point("scale centre", center.into())?;
        let factor = positive("scale factor", factor)?;
        Active::scale(self.raw(), center, factor).map(Shape::from_raw)
    }

    /// The shape swept along a straight vector: a face makes a solid.
    ///
    /// # Errors
    /// [`Error::InvalidInput`](crate::Error::InvalidInput) if the vector is zero or not
    /// finite; [`Error::Operation`](crate::Error::Operation) if the kernel fails.
    ///
    /// # Examples
    /// ```
    /// # use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let square = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]];
    /// let prism = Polygon::new(square).build()?.extruded([0.0, 0.0, 3.0])?;
    /// assert!((prism.volume()? - 3.0).abs() < 1e-6);
    /// # Ok(())
    /// # }
    /// ```
    pub fn extruded(&self, by: impl Into<Vec3>) -> Result<Shape> {
        let by = vector("extrusion", by.into())?;
        if by == Vec3::ZERO {
            return Err(invalid("extrusion must not be zero"));
        }
        Active::extrude(self.raw(), by).map(Shape::from_raw)
    }

    /// The shape swept around an axis, by the right-hand rule.
    ///
    /// The angle must be more than zero and at most a full turn. To sweep the other way,
    /// negate the axis direction.
    ///
    /// # Errors
    /// [`Error::InvalidInput`](crate::Error::InvalidInput) if the axis origin is not
    /// finite or the angle is out of range; [`Error::Operation`](crate::Error::Operation)
    /// if the kernel fails, as when the shape crosses the axis.
    ///
    /// # Examples
    /// ```
    /// # use occt::prelude::*;
    /// # use std::f64::consts::PI;
    /// # fn main() -> occt::Result<()> {
    /// let section = [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [3.0, 0.0, 1.0], [1.0, 0.0, 1.0]];
    /// let ring = Polygon::new(section).build()?.revolved(Axis::Z, Angle::FULL_TURN)?;
    /// assert!((ring.volume()? - PI * (9.0 - 1.0)).abs() < 1e-5);
    /// # Ok(())
    /// # }
    /// ```
    pub fn revolved(&self, around: Axis, angle: Angle) -> Result<Shape> {
        let around = axis("revolve axis", around)?;
        let radians = positive("revolve angle", angle.as_radians())?;
        if radians > TAU + FULL_TURN_SLACK {
            return Err(invalid(format!(
                "revolve angle must be at most a full turn, got {radians} radians"
            )));
        }
        Active::revolve(self.raw(), around, angle).map(Shape::from_raw)
    }
}
