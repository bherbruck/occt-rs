//! Boolean operations, as methods and as operators.

use std::ops::{BitAnd, BitOr, Sub};

use occt_kernel::{Booleans, Result};

use crate::kernel::Active;
use crate::shape::Shape;
use crate::subshape::{Edge, Face, Solid, Vertex};

impl Shape {
    /// The union of this shape and another. Also written `&a | &b`.
    ///
    /// The result may be a compound. Use [`solids`](Shape::solids) to see through it.
    ///
    /// # Errors
    /// [`Error::Operation`](crate::Error::Operation) if the kernel cannot compute it.
    ///
    /// # Examples
    /// ```
    /// # use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let a = Cuboid::new([1.0, 1.0, 1.0]).build()?;
    /// let b = Cuboid::new([1.0, 1.0, 1.0]).at([1.0, 0.0, 0.0]).build()?;
    /// assert!((a.fuse(&b)?.volume()? - 2.0).abs() < 1e-6);
    /// # Ok(())
    /// # }
    /// ```
    pub fn fuse(&self, other: &Shape) -> Result<Shape> {
        Active::fuse(self.raw(), other.raw()).map(Shape::from_raw)
    }

    /// This shape without the other. Also written `&a - &b`.
    ///
    /// # Errors
    /// [`Error::Operation`](crate::Error::Operation) if the kernel cannot compute it.
    ///
    /// # Examples
    /// ```
    /// # use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let a = Cuboid::new([2.0, 1.0, 1.0]).build()?;
    /// let b = Cuboid::new([1.0, 1.0, 1.0]).at([1.0, 0.0, 0.0]).build()?;
    /// assert!((a.cut(&b)?.volume()? - 1.0).abs() < 1e-6);
    /// # Ok(())
    /// # }
    /// ```
    pub fn cut(&self, other: &Shape) -> Result<Shape> {
        Active::cut(self.raw(), other.raw()).map(Shape::from_raw)
    }

    /// The part this shape shares with the other. Also written `&a & &b`.
    ///
    /// When the shapes do not meet, the result is either an error or a shape with no
    /// volume, depending on the kernel.
    ///
    /// # Errors
    /// [`Error::Operation`](crate::Error::Operation) if the kernel cannot compute it.
    ///
    /// # Examples
    /// ```
    /// # use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let a = Cuboid::new([2.0, 1.0, 1.0]).build()?;
    /// let b = Cuboid::new([2.0, 1.0, 1.0]).at([1.0, 0.0, 0.0]).build()?;
    /// assert!((a.common(&b)?.volume()? - 1.0).abs() < 1e-6);
    /// # Ok(())
    /// # }
    /// ```
    pub fn common(&self, other: &Shape) -> Result<Shape> {
        Active::common(self.raw(), other.raw()).map(Shape::from_raw)
    }
}

/// Implements the operators for references to a shape type. The right side is anything
/// that views as a shape, so a [`Solid`] and a [`Shape`] combine freely.
macro_rules! boolean_operators {
    ($($lhs:ty),* $(,)?) => {$(
        /// Union: see [`Shape::fuse`].
        impl<R: AsRef<Shape>> BitOr<R> for &$lhs {
            type Output = Result<Shape>;

            fn bitor(self, rhs: R) -> Result<Shape> {
                self.fuse(rhs.as_ref())
            }
        }

        /// Difference: see [`Shape::cut`].
        impl<R: AsRef<Shape>> Sub<R> for &$lhs {
            type Output = Result<Shape>;

            fn sub(self, rhs: R) -> Result<Shape> {
                self.cut(rhs.as_ref())
            }
        }

        /// Intersection: see [`Shape::common`].
        impl<R: AsRef<Shape>> BitAnd<R> for &$lhs {
            type Output = Result<Shape>;

            fn bitand(self, rhs: R) -> Result<Shape> {
                self.common(rhs.as_ref())
            }
        }
    )*};
}

boolean_operators!(Shape, Solid, Face, Edge, Vertex);
