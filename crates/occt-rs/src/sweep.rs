//! Sweeps and lofts: solids made by moving or joining profiles.

use std::f64::consts::{FRAC_PI_2, TAU};

pub use occt_kernel::SweepOrientation;
use occt_kernel::{AdvancedSweeps, LoftOptions, SweepOptions};

use crate::history::Modeled;
use crate::kernel::Active;
use crate::shape::Shape;
use crate::subshape::{Face, Solid};
use crate::validate::{axis, finite, invalid, positive, vector};
use crate::{Angle, Axis, Dir3, Result, Vec3};

/// Slack on the full-turn limit, so that `Angle::degrees(360.0)` passes however it rounds.
const FULL_TURN_SLACK: f64 = 1e-9;

impl Shape {
    /// [`extruded`](Shape::extruded), also reporting which input subshape made what: each
    /// edge of a face generates a side face of the prism.
    ///
    /// # Errors
    /// [`Error::InvalidInput`](crate::Error::InvalidInput) if the vector is zero or not
    /// finite; [`Error::Operation`](crate::Error::Operation) if the kernel fails.
    ///
    /// # Examples
    /// ```
    /// use occt::prelude::*;
    ///
    /// # fn main() -> occt::Result<()> {
    /// let square = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]];
    /// let face = Polygon::new(square).build()?;
    /// let prism = face.extruded_modeled([0.0, 0.0, 3.0])?;
    /// assert!((prism.shape.volume()? - 3.0).abs() < 1e-6);
    ///
    /// let edge = face.edges()?.next().expect("a square has edges");
    /// assert_eq!(prism.history.generated(&edge).len(), 1);
    /// # Ok(())
    /// # }
    /// ```
    pub fn extruded_modeled(&self, by: impl Into<Vec3>) -> Result<Modeled> {
        let by = extrusion(by.into())?;
        Active::extrude_modeled(self.raw(), by).map(Modeled::from_raw)
    }

    /// [`revolved`](Shape::revolved), also reporting which input subshape made what.
    ///
    /// # Errors
    /// [`Error::InvalidInput`](crate::Error::InvalidInput) if the axis origin is not finite
    /// or the angle is not more than zero and at most a full turn;
    /// [`Error::Operation`](crate::Error::Operation) if the kernel fails, as when the shape
    /// crosses the axis.
    ///
    /// # Examples
    /// ```
    /// use occt::prelude::*;
    /// use std::f64::consts::PI;
    ///
    /// # fn main() -> occt::Result<()> {
    /// let section = [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [3.0, 0.0, 1.0], [1.0, 0.0, 1.0]];
    /// let face = Polygon::new(section).build()?;
    /// let ring = face.revolved_modeled(Axis::Z, Angle::FULL_TURN)?;
    /// assert!((ring.shape.volume()? - 8.0 * PI).abs() < 1e-5);
    ///
    /// let edge = face.edges()?.next().expect("a section has edges");
    /// assert!(!ring.history.generated(&edge).is_empty());
    /// # Ok(())
    /// # }
    /// ```
    pub fn revolved_modeled(&self, around: Axis, angle: Angle) -> Result<Modeled> {
        let around = axis("revolve axis", around)?;
        let radians = positive("revolve angle", angle.as_radians())?;
        if radians > TAU + FULL_TURN_SLACK {
            return Err(invalid(format!(
                "revolve angle must be at most a full turn, got {radians} radians"
            )));
        }
        Active::revolve_modeled(self.raw(), around, angle).map(Modeled::from_raw)
    }
}

fn extrusion(by: Vec3) -> Result<Vec3> {
    let by = vector("extrusion", by)?;
    if by == Vec3::ZERO {
        return Err(invalid("extrusion must not be zero"));
    }
    Ok(by)
}

impl Face {
    /// This planar face extruded with sloped sides, as a moulded part needs.
    ///
    /// A positive `taper` removes material toward the far end, a negative one adds it, and
    /// zero is a plain extrusion. The slope is measured from the extrusion direction. Holes
    /// in the face slope the same way relative to the material, so they widen as the
    /// outline narrows. Sides must be flat or round: other profiles are an error.
    ///
    /// # Errors
    /// [`Error::InvalidInput`](crate::Error::InvalidInput) if the vector is zero or not
    /// finite, or the taper is not finite and under a quarter turn either way;
    /// [`Error::Operation`](crate::Error::Operation) if the face is not planar, the vector
    /// lies in its plane, or the taper closes the profile before the far end.
    ///
    /// # Examples
    /// ```
    /// use occt::prelude::*;
    ///
    /// # fn main() -> occt::Result<()> {
    /// let square = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0], [0.0, 10.0, 0.0]];
    /// let taper = Angle::degrees(45.0);
    /// let frustum = Polygon::new(square).build()?.extruded_tapered([0.0, 0.0, 2.0], taper)?;
    ///
    /// // Each side moves in by 2, so the top is 6 across.
    /// let expected = 2.0 / 3.0 * (100.0 + 60.0 + 36.0);
    /// assert!((frustum.volume()? - expected).abs() < 1e-4);
    /// # Ok(())
    /// # }
    /// ```
    pub fn extruded_tapered(&self, by: impl Into<Vec3>, taper: Angle) -> Result<Solid> {
        let by = extrusion(by.into())?;
        let radians = finite("taper", taper.as_radians())?;
        if radians.abs() >= FRAC_PI_2 {
            return Err(invalid(format!(
                "taper must be under a quarter turn, got {radians} radians"
            )));
        }
        Active::extrude_tapered(self.raw(), by, taper)
            .map(Shape::from_raw)?
            .try_into()
    }
}

/// A profile moving along a path, from [`Sweep::profile`] and [`SweepProfile::along`].
///
/// The profile is a wire, an edge or a face, and usually sits at the start of the path,
/// across it. It keeps its position relative to the path as it moves, so the path need not
/// touch it. A face sweeps into a solid with its holes as tunnels. A closed wire makes a
/// tube unless you ask for a [`solid`](Sweep::solid), and an open one makes a sheet.
///
/// A section wider than the path's tightest bend folds through itself. That is not reported
/// as an error, so keep sections narrower than the bend radius.
///
/// # Examples
/// A round bar bent through a quarter turn. By Pappus's theorem its volume is the section
/// area times the distance the section's centre travels.
///
/// ```
/// use occt::Sweep;
/// use occt::prelude::*;
/// use std::f64::consts::{FRAC_PI_2, PI};
///
/// # fn main() -> occt::Result<()> {
/// let (bend, radius) = (10.0, 1.0);
///
/// // The disc at one end of a cylinder gives a section at the start of the path.
/// let bar = Cylinder::new(radius, 1.0).along(Axis::new([bend, 0.0, 0.0].into(), Dir3::Y)).build()?;
/// let disc = bar.faces()?
///     .find(|f| f.center_of_mass().is_ok_and(|c| c.distance([bend, 0.0, 0.0].into()) < 1e-6))
///     .expect("the cylinder has an end there");
///
/// // A point moved around an axis traces an arc, which is the path.
/// let start = Cuboid::new([1.0, 1.0, 1.0]).at([bend, 0.0, 0.0]).build()?
///     .vertices()?
///     .find(|v| v.point().is_ok_and(|p| p.distance([bend, 0.0, 0.0].into()) < 1e-6))
///     .expect("the box has a corner there");
/// let arc = start.revolved(Axis::Z, Angle::degrees(90.0))?;
///
/// let swept = Sweep::profile(&disc).along(&arc).build()?;
/// let expected = PI * radius * radius * bend * FRAC_PI_2;
/// assert!((swept.shape.volume()? - expected).abs() / expected < 1e-3);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug)]
#[must_use = "a builder does nothing until `build` is called"]
pub struct Sweep<'a> {
    profile: &'a Shape,
    path: &'a Shape,
    options: SweepOptions,
}

/// A profile waiting for its path, from [`Sweep::profile`].
#[derive(Clone, Copy, Debug)]
#[must_use = "a sweep needs a path: call `along`"]
pub struct SweepProfile<'a> {
    profile: &'a Shape,
}

impl<'a> SweepProfile<'a> {
    /// Sets the path, a wire or an edge.
    pub fn along(self, path: &'a Shape) -> Sweep<'a> {
        Sweep {
            profile: self.profile,
            path,
            options: SweepOptions::default(),
        }
    }
}

impl<'a> Sweep<'a> {
    /// Starts a sweep of a wire, an edge or a face. Continue with
    /// [`along`](SweepProfile::along).
    pub fn profile(profile: &'a Shape) -> SweepProfile<'a> {
        SweepProfile { profile }
    }

    /// Chooses how the profile turns along the path. The default,
    /// [`Corrected`](SweepOrientation::Corrected), follows the path with as little twist as
    /// it can.
    pub fn orientation(mut self, orientation: SweepOrientation) -> Self {
        self.options.orientation = orientation;
        self
    }

    /// Keeps the profile at a constant angle to a direction, which is what stops the
    /// section of a thread rolling around a helix. This replaces any
    /// [`orientation`](Sweep::orientation).
    pub fn binormal(mut self, direction: Dir3) -> Self {
        self.options.binormal = Some(direction);
        self
    }

    /// Closes the ends of a closed wire's sweep to make a solid. A face always does.
    pub fn solid(mut self) -> Self {
        self.options.solid = true;
        self
    }

    /// Builds the sweep. Its history maps each edge of the profile to the side faces it
    /// made, and a face profile to the two end faces.
    ///
    /// # Errors
    /// [`Error::WrongKind`](crate::Error::WrongKind) if the profile is not a wire, edge or
    /// face, or the path is not a wire or edge;
    /// [`Error::Operation`](crate::Error::Operation) if a solid is asked of an open profile,
    /// or the kernel cannot make the profile follow the path.
    pub fn build(self) -> Result<Modeled> {
        Active::sweep(self.profile.raw(), self.path.raw(), &self.options).map(Modeled::from_raw)
    }
}

/// A shape joining sections in order, from [`Loft::through`].
///
/// Sections are wires, edges or faces, of which only the outline counts. The first and last
/// may be points instead, as a [`Vertex`](crate::Vertex). The result is a smooth solid
/// unless you ask for [`ruled`](Loft::ruled) or a [`sheet`](Loft::sheet).
///
/// # Examples
/// ```
/// use occt::Loft;
/// use occt::prelude::*;
///
/// # fn main() -> occt::Result<()> {
/// let square = |side: f64, z: f64| {
///     Polygon::new([[0.0, 0.0, z], [side, 0.0, z], [side, side, z], [0.0, side, z]]).build()
/// };
/// let (bottom, top) = (square(4.0, 0.0)?, square(2.0, 3.0)?);
///
/// // Between the two squares of sides 4 and 2, 3 apart: a frustum.
/// let frustum = Loft::through([&bottom, &top]).ruled().build()?;
/// assert!((frustum.volume()? - 3.0 / 3.0 * (16.0 + 8.0 + 4.0)).abs() < 1e-4);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug)]
#[must_use = "a builder does nothing until `build` is called"]
pub struct Loft {
    sections: Vec<Shape>,
    options: LoftOptions,
}

impl Loft {
    /// A loft through the sections, in order, as references or owned.
    pub fn through(sections: impl IntoIterator<Item: AsRef<Shape>>) -> Self {
        Self {
            sections: sections.into_iter().map(|s| s.as_ref().clone()).collect(),
            options: LoftOptions::default(),
        }
    }

    /// Joins consecutive sections with straight lines, rather than one smooth surface
    /// through them all.
    pub fn ruled(mut self) -> Self {
        self.options.ruled = true;
        self
    }

    /// Leaves the ends open, which makes a sheet rather than a solid.
    pub fn sheet(mut self) -> Self {
        self.options.solid = false;
        self
    }

    /// How closely fitted surfaces must follow the sections. The default is 1e-6.
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.options.tolerance = tolerance;
        self
    }

    /// Builds the loft.
    ///
    /// # Errors
    /// [`Error::InvalidInput`](crate::Error::InvalidInput) if there are fewer than two
    /// sections or the tolerance is not finite and positive;
    /// [`Error::WrongKind`](crate::Error::WrongKind) if a section is not a wire, edge, face
    /// or vertex; [`Error::Operation`](crate::Error::Operation) if a solid is asked of open
    /// sections, a point is not at an end, or the kernel cannot join the sections.
    pub fn build(self) -> Result<Shape> {
        if self.sections.len() < 2 {
            return Err(invalid(format!(
                "a loft needs at least 2 sections, got {}",
                self.sections.len()
            )));
        }
        positive("loft tolerance", self.options.tolerance)?;
        let raw: Vec<_> = self.sections.iter().map(Shape::raw).collect();
        Active::loft(&raw, &self.options).map(Shape::from_raw)
    }
}
