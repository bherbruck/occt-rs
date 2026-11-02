//! Tessellation.

use occt_kernel::{Deflection, Mesh, MeshOptions, Result, Tessellation};

use crate::kernel::Active;
use crate::shape::Shape;
use crate::validate::positive;

impl Shape {
    /// Converts this shape to triangles.
    ///
    /// The [`Mesh`] is plain owned data with no tie to the kernel, so it can go to any
    /// thread. For a closed solid it is watertight, apart from vertices being duplicated
    /// along sharp edges to carry a normal per face, and wound counter-clockwise from
    /// outside.
    ///
    /// # Errors
    /// [`Error::InvalidInput`](crate::Error::InvalidInput) if a deflection is not finite
    /// and positive; [`Error::Operation`](crate::Error::Operation) if the kernel fails.
    ///
    /// # Examples
    /// ```
    /// # use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let cube = Cuboid::new([1.0, 1.0, 1.0]).build()?;
    /// let mesh = cube.mesh(MeshOptions::default())?;
    /// assert_eq!(mesh.indices.len(), 12);
    /// assert!((mesh.volume() - 1.0).abs() < 1e-6);
    ///
    /// assert!(cube.mesh(MeshOptions::default().absolute(-1.0)).is_err());
    /// # Ok(())
    /// # }
    /// ```
    pub fn mesh(&self, options: MeshOptions) -> Result<Mesh> {
        match options.deflection {
            Deflection::Absolute(distance) => positive("absolute deflection", distance)?,
            Deflection::Relative(fraction) => positive("relative deflection", fraction)?,
        };
        positive(
            "angular deflection",
            options.angular_deflection.as_radians(),
        )?;
        Active::mesh(self.raw(), &options)
    }
}
