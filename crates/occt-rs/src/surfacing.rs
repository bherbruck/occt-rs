//! Joining surfaces smoothly, and measuring how smoothly they join.
//!
//! [`Face::continuity_with`] answers whether two faces meet at G0, G1 or G2.
//! [`Face::blend`] builds a patch across the gap between two faces, matching their curvature
//! by default. [`Face::extended`] grows a face along its own surface.

use occt_kernel::{BlendSide, FaceSides, Surfacing};

pub use occt_kernel::{BlendContinuity, ContinuityReport};

use crate::Result;
use crate::kernel::Active;
use crate::shape::Shape;
use crate::subshape::{Edge, Face};
use crate::validate::{invalid, positive};

impl Face {
    /// Measures how this face and another meet along an edge lying on both.
    ///
    /// Sampled away from the edge's ends: G1 holds when [`angle`](ContinuityReport::angle)
    /// is small, and G2 when [`curvature_gap`](ContinuityReport::curvature_gap) is also small
    /// next to the faces' curvature.
    ///
    /// # Errors
    /// [`crate::Error::Operation`] if the edge doesn't border both faces.
    ///
    /// # Examples
    /// ```
    /// use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let cube = Cuboid::new([2.0, 2.0, 2.0]).build()?;
    /// let bottom = cube.faces()?.next().expect("a cube has faces");
    /// let edge = bottom.edges()?.next().expect("a face has edges");
    /// let side = cube
    ///     .faces()?
    ///     .find(|f| *f != bottom && f.edges().is_ok_and(|mut e| e.any(|e| e == edge)))
    ///     .expect("every edge of a cube joins two faces");
    /// let report = bottom.continuity_with(&side, &edge)?;
    /// assert!((report.angle.as_degrees() - 90.0).abs() < 1e-6);
    /// # Ok(())
    /// # }
    /// ```
    pub fn continuity_with(&self, other: &Face, edge: &Edge) -> Result<ContinuityReport> {
        Active::measure_continuity(self.raw(), other.raw(), edge.raw())
    }

    /// Starts a patch joining two faces across the gap between an edge on each.
    ///
    /// It matches curvature unless told otherwise, and each edge only has to lie on its face.
    /// The patch stays separate: [`Sewing`](crate::Sewing) joins it to the faces.
    pub fn blend(a: (&Edge, &Face), b: (&Edge, &Face)) -> FaceBlend {
        FaceBlend {
            a: (a.0.clone(), a.1.clone()),
            b: (b.0.clone(), b.1.clone()),
            continuity: BlendContinuity::G2,
            tension: 1.0,
        }
    }

    /// Starts growing this face by about `length` along its own surface.
    ///
    /// The result covers the face's whole parameter rectangle: trims and holes are gone, so
    /// it also untrims. A periodic or bounded surface stops where it runs out.
    pub fn extended(&self, length: f64) -> FaceExtension {
        FaceExtension {
            face: self.clone(),
            length,
            sides: FaceSides::default(),
        }
    }
}

/// A patch being set up, from [`Face::blend`].
#[derive(Clone, Debug)]
#[must_use = "a blend does nothing until built"]
pub struct FaceBlend {
    a: (Edge, Face),
    b: (Edge, Face),
    continuity: BlendContinuity,
    tension: f64,
}

impl FaceBlend {
    /// Sets how closely the patch continues the faces. The default is
    /// [`G2`](BlendContinuity::G2).
    pub fn continuity(mut self, continuity: BlendContinuity) -> Self {
        self.continuity = continuity;
        self
    }

    /// Scales how far the patch follows each face before turning. The default is 1.
    pub fn tension(mut self, tension: f64) -> Self {
        self.tension = tension;
        self
    }

    /// Builds the patch and checks it against both faces.
    ///
    /// # Errors
    /// [`crate::Error::InvalidInput`] if the tension isn't positive.
    /// [`crate::Error::Operation`] if an edge doesn't border its face, the edges touch, or
    /// the patch comes out invalid or not tangent.
    ///
    /// # Examples
    /// ```
    /// use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let floor = Polygon::new([
    ///     [0.0, -9.0, 0.0],
    ///     [9.0, -9.0, 0.0],
    ///     [9.0, 0.0, 0.0],
    ///     [0.0, 0.0, 0.0],
    /// ])
    /// .build()?;
    /// let wall = Polygon::new([
    ///     [0.0, 4.0, 4.0],
    ///     [9.0, 4.0, 4.0],
    ///     [9.0, 4.0, 9.0],
    ///     [0.0, 4.0, 9.0],
    /// ])
    /// .build()?;
    /// // The third edge of each polygon faces the gap for the floor, the first for the wall.
    /// let floor_edge = floor.edges()?.nth(2).expect("a polygon has edges");
    /// let wall_edge = wall.edges()?.next().expect("a polygon has edges");
    /// let blend = Face::blend((&floor_edge, &floor), (&wall_edge, &wall)).build()?;
    /// let joint = blend.continuity_with(&floor, &floor_edge)?;
    /// assert!(joint.angle.as_degrees() < 0.1);
    /// assert!(joint.curvature_gap < 1e-3);
    /// # Ok(())
    /// # }
    /// ```
    pub fn build(self) -> Result<Face> {
        positive("a blend's tension", self.tension)?;
        let a = BlendSide {
            edge: self.a.0.raw(),
            face: self.a.1.raw(),
        };
        let b = BlendSide {
            edge: self.b.0.raw(),
            face: self.b.1.raw(),
        };
        let raw = Active::blend_faces(a, b, self.continuity, self.tension)?;
        Face::try_from(Shape::from_raw(raw))
    }
}

/// A face being grown, from [`Face::extended`].
#[derive(Clone, Debug)]
#[must_use = "an extension does nothing until built"]
pub struct FaceExtension {
    face: Face,
    length: f64,
    sides: FaceSides,
}

impl FaceExtension {
    /// Grows the side at the smallest U.
    pub fn u_min(mut self) -> Self {
        self.sides.u_min = true;
        self
    }
    /// Grows the side at the largest U.
    pub fn u_max(mut self) -> Self {
        self.sides.u_max = true;
        self
    }
    /// Grows the side at the smallest V.
    pub fn v_min(mut self) -> Self {
        self.sides.v_min = true;
        self
    }
    /// Grows the side at the largest V.
    pub fn v_max(mut self) -> Self {
        self.sides.v_max = true;
        self
    }
    /// Grows every side.
    pub fn all_sides(self) -> Self {
        self.u_min().u_max().v_min().v_max()
    }

    /// Builds the grown face.
    ///
    /// # Errors
    /// [`crate::Error::InvalidInput`] if the length isn't positive or no side was chosen.
    /// [`crate::Error::Operation`] if the face can't grow.
    ///
    /// # Examples
    /// ```
    /// use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let plate = Face::planar(&occt::Wire::rectangle(2.0, 3.0)?).build()?;
    /// let bigger = plate.extended(1.0).all_sides().build()?;
    /// assert!((bigger.area()? - 20.0).abs() < 1e-6);
    /// assert!(plate.extended(1.0).build().is_err());
    /// # Ok(())
    /// # }
    /// ```
    pub fn build(self) -> Result<Face> {
        positive("an extension's length", self.length)?;
        if self.sides == FaceSides::default() {
            return Err(invalid("an extension needs at least one side"));
        }
        let raw = Active::extend_face(self.face.raw(), self.length, self.sides)?;
        Face::try_from(Shape::from_raw(raw))
    }
}
