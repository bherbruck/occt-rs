//! STEP exchange.
//!
//! The kernel works on bytes. Files are the caller's concern, through
//! [`std::fs::File`] or any other [`Read`] or [`Write`], so this works in environments
//! with no file system.

use std::io::{Read, Write};

use occt_kernel::{Result, StepIo};

use crate::kernel::Active;
use crate::shape::Shape;

impl Shape {
    /// Reads a STEP file from a stream, to its end.
    ///
    /// Files are the caller's concern: pass a [`std::fs::File`] to read from disk.
    ///
    /// # Errors
    /// [`Error::Io`](crate::Error::Io) if reading fails;
    /// [`Error::Operation`](crate::Error::Operation) if the data is not a STEP file the
    /// kernel can use.
    ///
    /// # Examples
    /// ```no_run
    /// # fn main() -> occt::Result<()> {
    /// let part = occt::Shape::read_step(std::fs::File::open("part.step")?)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn read_step(mut reader: impl Read) -> Result<Shape> {
        let mut data = Vec::new();
        reader.read_to_end(&mut data)?;
        Self::from_step(&data)
    }

    /// Writes this shape to a stream as a STEP file.
    ///
    /// Files are the caller's concern: pass a [`std::fs::File`] to write to disk, and
    /// wrap nothing, since this makes one large write.
    ///
    /// # Errors
    /// [`Error::Io`](crate::Error::Io) if writing fails;
    /// [`Error::Operation`](crate::Error::Operation) if the kernel cannot serialise the
    /// shape.
    ///
    /// # Examples
    /// ```no_run
    /// # use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let part = Cuboid::new([1.0, 1.0, 1.0]).build()?;
    /// part.write_step(std::fs::File::create("part.step")?)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn write_step(&self, mut writer: impl Write) -> Result<()> {
        writer.write_all(&self.to_step()?)?;
        writer.flush()?;
        Ok(())
    }

    /// Parses a STEP file held in memory.
    ///
    /// # Errors
    /// [`Error::Operation`](crate::Error::Operation) if the data is not a STEP file the
    /// kernel can use.
    ///
    /// # Examples
    /// ```
    /// # use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let cube = Cuboid::new([1.0, 1.0, 1.0]).build()?;
    /// let copy = Shape::from_step(&cube.to_step()?)?;
    /// assert!((copy.volume()? - 1.0).abs() < 1e-6);
    /// assert!(Shape::from_step(b"not a STEP file").is_err());
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_step(data: &[u8]) -> Result<Shape> {
        Active::read_step(data).map(Shape::from_raw)
    }

    /// Serialises this shape as the bytes of a STEP file.
    ///
    /// # Errors
    /// [`Error::Operation`](crate::Error::Operation) if the kernel cannot serialise the
    /// shape.
    pub fn to_step(&self) -> Result<Vec<u8>> {
        Active::write_step(self.raw())
    }
}
