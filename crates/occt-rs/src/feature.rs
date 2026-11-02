//! Local edits of solids, and transforms beyond the rigid ones.

use std::borrow::Borrow;
use std::collections::HashSet;
use std::f64::consts::FRAC_PI_2;

use occt_kernel::Features;

use crate::history::Modeled;
use crate::kernel::Active;
use crate::shape::Shape;
use crate::subshape::{Face, Solid};
use crate::validate::{finite, invalid, point, positive};
use crate::{Angle, Dir3, Plane, Result, Transform};

/// How far a matrix's columns may be from orthonormal and still count as a rigid motion.
/// Products of a few rotations stay well inside this.
const RIGID_TOLERANCE: f64 = 1e-9;

/// The distinct faces, each checked to belong to the solid.
fn own_faces(solid: &Solid, faces: Vec<Face>, op: &str) -> Result<Vec<Face>> {
    let own: HashSet<Face> = solid.faces()?.collect();
    let mut seen = HashSet::new();
    let mut distinct = Vec::with_capacity(faces.len());
    for face in faces {
        if !own.contains(&face) {
            return Err(invalid(format!(
                "{op} was given a face of a different shape"
            )));
        }
        if seen.insert(face.clone()) {
            distinct.push(face);
        }
    }
    Ok(distinct)
}

fn solid_result(modeled: Modeled) -> Result<Modeled<Solid>> {
    Ok(Modeled {
        shape: modeled.shape.try_into()?,
        history: modeled.history,
    })
}

fn nonzero(what: &str, value: f64) -> Result<f64> {
    if finite(what, value)? == 0.0 {
        return Err(invalid(format!("{what} must not be zero")));
    }
    Ok(value)
}

fn finite_transform(what: &str, transform: Transform) -> Result<Transform> {
    if transform.is_finite() {
        Ok(transform)
    } else {
        Err(invalid(format!("{what} must be finite, got {transform:?}")))
    }
}

/// The determinant of the transform's linear part.
fn determinant(t: &Transform) -> f64 {
    let r = t.rows;
    r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
        - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
        + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0])
}

/// Whether the transform is a rotation followed by a translation.
fn is_rigid(t: &Transform) -> bool {
    let column = |c: usize| [t.rows[0][c], t.rows[1][c], t.rows[2][c]];
    let orthonormal = (0..3).all(|i| {
        (0..3).all(|j| {
            let (a, b) = (column(i), column(j));
            let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
            (dot - if i == j { 1.0 } else { 0.0 }).abs() <= RIGID_TOLERANCE
        })
    });
    orthonormal && determinant(t) > 0.0
}

/// A builder for a hollowed solid, from [`Solid::shelled`].
#[derive(Clone, Debug)]
#[must_use = "a builder does nothing until `build` is called"]
pub struct Hollow<'a> {
    solid: &'a Solid,
    thickness: f64,
    open: Vec<Face>,
    outward: bool,
}

impl Hollow<'_> {
    /// Removes a face, leaving an opening there.
    pub fn open(mut self, face: &Face) -> Self {
        self.open.push(face.clone());
        self
    }

    /// Removes several faces, as references or owned.
    pub fn open_faces(mut self, faces: impl IntoIterator<Item: Borrow<Face>>) -> Self {
        self.open
            .extend(faces.into_iter().map(|f| f.borrow().clone()));
        self
    }

    /// Grows the walls outside the original surface. By default they grow inside it, so
    /// the outer size stays as it was.
    pub fn outward(mut self) -> Self {
        self.outward = true;
        self
    }

    /// Builds the hollow solid. With no open faces it has a closed cavity.
    ///
    /// # Errors
    /// [`Error::InvalidInput`](crate::Error::InvalidInput) if the thickness is not finite
    /// and positive or a face is not part of this solid;
    /// [`Error::Operation`](crate::Error::Operation) if the kernel cannot build the walls,
    /// as when they are too thick for the solid.
    pub fn build(self) -> Result<Modeled<Solid>> {
        let thickness = positive("wall thickness", self.thickness)?;
        let signed = if self.outward { thickness } else { -thickness };
        let open = own_faces(self.solid, self.open, "shell")?;
        let raw: Vec<_> = open.iter().map(|face| face.raw()).collect();
        solid_result(Modeled::from_raw(Active::shell(
            self.solid.raw(),
            &raw,
            signed,
        )?))
    }
}

/// A builder for sloped faces, from [`Solid::drafted`].
#[derive(Clone, Debug)]
#[must_use = "a builder does nothing until `build` is called"]
pub struct Draft<'a> {
    solid: &'a Solid,
    pull: Dir3,
    neutral: Plane,
    angle: Angle,
    faces: Vec<Face>,
}

impl Draft<'_> {
    /// Adds a face to tilt.
    pub fn face(mut self, face: &Face) -> Self {
        self.faces.push(face.clone());
        self
    }

    /// Adds several faces to tilt, as references or owned. Selecting a face twice is the
    /// same as selecting it once.
    pub fn faces(mut self, faces: impl IntoIterator<Item: Borrow<Face>>) -> Self {
        self.faces
            .extend(faces.into_iter().map(|f| f.borrow().clone()));
        self
    }

    /// Builds the drafted solid.
    ///
    /// # Errors
    /// [`Error::InvalidInput`](crate::Error::InvalidInput) if no faces are selected, a face
    /// is not part of this solid, the plane's origin is not finite, or the angle is zero, not
    /// finite or not under a quarter turn either way;
    /// [`Error::Operation`](crate::Error::Operation) if the kernel cannot tilt a face, as
    /// when it is not flat or round, or the pull direction crosses it.
    pub fn build(self) -> Result<Modeled<Solid>> {
        point("draft plane", self.neutral.origin)?;
        let radians = nonzero("draft angle", self.angle.as_radians())?;
        if radians.abs() >= FRAC_PI_2 {
            return Err(invalid(format!(
                "draft angle must be under a quarter turn, got {radians} radians"
            )));
        }
        let faces = own_faces(self.solid, self.faces, "draft")?;
        if faces.is_empty() {
            return Err(invalid("draft needs at least one face"));
        }
        let raw: Vec<_> = faces.iter().map(|face| face.raw()).collect();
        solid_result(Modeled::from_raw(Active::draft(
            self.solid.raw(),
            &raw,
            self.pull,
            self.neutral,
            self.angle,
        )?))
    }
}

impl Solid {
    /// Starts hollowing this solid, leaving walls of a thickness. Choose the
    /// [`open`](Hollow::open) faces on the returned builder.
    ///
    /// # Examples
    /// An open-topped box:
    ///
    /// ```
    /// use occt::prelude::*;
    ///
    /// # fn main() -> occt::Result<()> {
    /// let block = Cuboid::new([10.0, 10.0, 10.0]).build()?;
    /// let top = block.faces()?
    ///     .find(|f| f.center_of_mass().is_ok_and(|c| c.z > 9.0))
    ///     .expect("a box has a top");
    /// let tray = block.shelled(1.0).open(&top).build()?;
    /// assert!((tray.shape.volume()? - (1000.0 - 8.0 * 8.0 * 9.0)).abs() < 1e-3);
    /// # Ok(())
    /// # }
    /// ```
    pub fn shelled(&self, thickness: f64) -> Hollow<'_> {
        Hollow {
            solid: self,
            thickness,
            open: Vec::new(),
            outward: false,
        }
    }

    /// Starts tilting faces so a mould can release along `pull`. Choose them on the
    /// returned builder.
    ///
    /// Each face stays put where it meets the `neutral` plane and turns by `angle` from the
    /// pull direction. A positive angle removes material on the side of the plane that
    /// `pull` points to, and adds it on the other side.
    ///
    /// # Examples
    /// ```
    /// use occt::Plane;
    /// use occt::prelude::*;
    ///
    /// # fn main() -> occt::Result<()> {
    /// let block = Cuboid::new([10.0, 10.0, 2.0]).build()?;
    /// let side = block.faces()?
    ///     .find(|f| f.center_of_mass().is_ok_and(|c| c.x > 9.0))
    ///     .expect("a box has a side");
    /// let drafted = block
    ///     .drafted(Dir3::Z, Plane::XY, Angle::degrees(45.0))
    ///     .face(&side)
    ///     .build()?;
    ///
    /// // The top edge moves in by 2, removing a wedge along the 10 long side.
    /// assert!((drafted.shape.volume()? - (200.0 - 2.0 * 2.0 / 2.0 * 10.0)).abs() < 1e-3);
    /// # Ok(())
    /// # }
    /// ```
    pub fn drafted(&self, pull: Dir3, neutral: Plane, angle: Angle) -> Draft<'_> {
        Draft {
            solid: self,
            pull,
            neutral,
            angle,
            faces: Vec::new(),
        }
    }

    /// This solid without some faces, its neighbours extended to close the gap. It takes
    /// away features such as holes and bosses.
    ///
    /// # Errors
    /// [`Error::InvalidInput`](crate::Error::InvalidInput) if no faces are given or a face
    /// is not part of this solid; [`Error::Operation`](crate::Error::Operation) if the
    /// neighbouring faces cannot close the gap.
    ///
    /// # Examples
    /// Filling a hole in a plate:
    ///
    /// ```
    /// use occt::prelude::*;
    ///
    /// # fn main() -> occt::Result<()> {
    /// let plate = Cuboid::new([10.0, 10.0, 2.0]).build()?;
    /// let axis = Axis::new([5.0, 5.0, -1.0].into(), Dir3::Z);
    /// let drill = Cylinder::new(1.0, 4.0).along(axis).build()?;
    /// let holed = Solid::try_from((&plate - &drill)?)?;
    /// let wall = holed.faces()?
    ///     .find(|f| f.surface_kind().is_ok_and(|kind| kind == SurfaceKind::Cylinder))
    ///     .expect("the hole has a wall");
    ///
    /// let filled = holed.without_faces([&wall])?;
    /// assert!((filled.shape.volume()? - 200.0).abs() < 1e-4);
    /// # Ok(())
    /// # }
    /// ```
    pub fn without_faces(
        &self,
        faces: impl IntoIterator<Item: Borrow<Face>>,
    ) -> Result<Modeled<Solid>> {
        let faces: Vec<Face> = faces.into_iter().map(|f| f.borrow().clone()).collect();
        let faces = own_faces(self, faces, "without_faces")?;
        if faces.is_empty() {
            return Err(invalid("without_faces needs at least one face"));
        }
        let raw: Vec<_> = faces.iter().map(|face| face.raw()).collect();
        solid_result(Modeled::from_raw(Active::remove_faces(self.raw(), &raw)?))
    }
}

impl Shape {
    /// A copy with every face moved along its normal. Positive grows a solid and negative
    /// shrinks it; corners stay sharp.
    ///
    /// # Errors
    /// [`Error::InvalidInput`](crate::Error::InvalidInput) if the distance is zero or not
    /// finite; [`Error::Operation`](crate::Error::Operation) if the kernel fails, as when a
    /// shrink would consume the solid.
    ///
    /// # Examples
    /// ```
    /// use occt::prelude::*;
    ///
    /// # fn main() -> occt::Result<()> {
    /// let block = Cuboid::new([2.0, 2.0, 2.0]).build()?;
    /// assert!((block.offset(1.0)?.volume()? - 64.0).abs() < 1e-4);
    /// # Ok(())
    /// # }
    /// ```
    pub fn offset(&self, distance: f64) -> Result<Shape> {
        let distance = nonzero("offset distance", distance)?;
        Active::offset(self.raw(), distance).map(Shape::from_raw)
    }

    /// A solid made by giving this face or shell a thickness, along its normal when
    /// positive and against it when negative.
    ///
    /// # Errors
    /// [`Error::InvalidInput`](crate::Error::InvalidInput) if the thickness is zero or not
    /// finite; [`Error::WrongKind`](crate::Error::WrongKind) if this is not a face or shell;
    /// [`Error::Operation`](crate::Error::Operation) if the kernel fails.
    ///
    /// # Examples
    /// ```
    /// use occt::prelude::*;
    ///
    /// # fn main() -> occt::Result<()> {
    /// let square = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 2.0, 0.0], [0.0, 2.0, 0.0]];
    /// let plate = Polygon::new(square).build()?.thickened(0.5)?;
    /// assert!((plate.volume()? - 2.0).abs() < 1e-5);
    /// # Ok(())
    /// # }
    /// ```
    pub fn thickened(&self, thickness: f64) -> Result<Solid> {
        let thickness = nonzero("thickness", thickness)?;
        Active::thicken(self.raw(), thickness)
            .map(Shape::from_raw)?
            .try_into()
    }

    /// A copy with faces on one surface merged, and edges on one curve merged. Booleans
    /// leave such seams behind.
    ///
    /// # Errors
    /// [`Error::Operation`](crate::Error::Operation) if the kernel fails.
    ///
    /// # Examples
    /// ```
    /// use occt::prelude::*;
    ///
    /// # fn main() -> occt::Result<()> {
    /// let a = Cuboid::new([1.0, 1.0, 1.0]).build()?;
    /// let b = Cuboid::new([1.0, 1.0, 1.0]).at([1.0, 0.0, 0.0]).build()?;
    /// let block = a.fuse(&b)?.unified()?;
    /// assert_eq!(block.shape.faces()?.len(), 6);
    /// # Ok(())
    /// # }
    /// ```
    pub fn unified(&self) -> Result<Modeled> {
        Active::unify(self.raw()).map(Modeled::from_raw)
    }

    /// A copy with tolerance and topology problems repaired, as imported files often need.
    /// Geometry closer together than `tolerance` is treated as touching.
    ///
    /// # Errors
    /// [`Error::InvalidInput`](crate::Error::InvalidInput) if the tolerance is not finite
    /// and positive; [`Error::Operation`](crate::Error::Operation) if the kernel fails.
    ///
    /// # Examples
    /// ```
    /// use occt::prelude::*;
    ///
    /// # fn main() -> occt::Result<()> {
    /// let healed = Cuboid::new([1.0, 1.0, 1.0]).build()?.healed(1e-4)?;
    /// assert!(healed.is_valid()?);
    /// # Ok(())
    /// # }
    /// ```
    pub fn healed(&self, tolerance: f64) -> Result<Shape> {
        let tolerance = positive("healing tolerance", tolerance)?;
        Active::heal(self.raw(), tolerance).map(Shape::from_raw)
    }

    /// An independent copy under any invertible affine transform, such as a non-uniform
    /// scale. Solids keep a positive volume even under a reflection. Non-uniform transforms
    /// may turn exact curves and surfaces into splines.
    ///
    /// # Errors
    /// [`Error::InvalidInput`](crate::Error::InvalidInput) if the transform is not finite
    /// or flattens space; [`Error::Operation`](crate::Error::Operation) if the kernel fails.
    ///
    /// # Examples
    /// ```
    /// use occt::Transform;
    /// use occt::prelude::*;
    /// use std::f64::consts::PI;
    ///
    /// # fn main() -> occt::Result<()> {
    /// let stretch = Transform::scale(Point3::ORIGIN, Vec3::new(1.0, 2.0, 3.0));
    /// let ellipsoid = Sphere::new(1.0).build()?.transformed(stretch)?;
    /// let exact = 4.0 / 3.0 * PI * 6.0;
    /// assert!((ellipsoid.volume()? - exact).abs() / exact < 1e-3);
    /// # Ok(())
    /// # }
    /// ```
    pub fn transformed(&self, transform: Transform) -> Result<Shape> {
        let transform = finite_transform("transform", transform)?;
        if determinant(&transform).abs() < f64::MIN_POSITIVE.sqrt() {
            return Err(invalid("transform must not flatten space"));
        }
        Active::transformed(self.raw(), &transform).map(Shape::from_raw)
    }

    /// An independent copy reflected through a plane. A solid keeps a positive volume.
    ///
    /// # Errors
    /// [`Error::InvalidInput`](crate::Error::InvalidInput) if the plane's origin is not
    /// finite; [`Error::Operation`](crate::Error::Operation) if the kernel fails.
    ///
    /// # Examples
    /// ```
    /// use occt::Plane;
    /// use occt::prelude::*;
    ///
    /// # fn main() -> occt::Result<()> {
    /// let block = Cuboid::new([1.0, 2.0, 3.0]).at([1.0, 0.0, 0.0]).build()?;
    /// let mirrored = block.mirrored(Plane::YZ)?;
    /// assert!((mirrored.volume()? - 6.0).abs() < 1e-6);
    /// assert!((mirrored.center_of_mass()?.x + 1.5).abs() < 1e-6);
    /// # Ok(())
    /// # }
    /// ```
    pub fn mirrored(&self, plane: Plane) -> Result<Shape> {
        point("mirror plane", plane.origin)?;
        self.transformed(Transform::mirror(plane))
    }

    /// This shape at a new placement, sharing its geometry with the original as the
    /// instances of an assembly do. This is much lighter than a copy, but only a rotation
    /// and translation can be done this way: to scale or reflect, use
    /// [`transformed`](Shape::transformed).
    ///
    /// # Errors
    /// [`Error::InvalidInput`](crate::Error::InvalidInput) if the transform is not finite
    /// or is not a rotation and translation; [`Error::Operation`](crate::Error::Operation)
    /// if the kernel fails.
    ///
    /// # Examples
    /// ```
    /// use occt::Transform;
    /// use occt::prelude::*;
    ///
    /// # fn main() -> occt::Result<()> {
    /// let part = Cuboid::new([1.0, 1.0, 1.0]).build()?;
    /// let instance = part.placed(Transform::translation(Vec3::new(5.0, 0.0, 0.0)))?;
    /// assert!((instance.center_of_mass()?.x - 5.5).abs() < 1e-6);
    ///
    /// let scale = Transform::scale(Point3::ORIGIN, Vec3::new(2.0, 2.0, 2.0));
    /// assert!(part.placed(scale).is_err());
    /// # Ok(())
    /// # }
    /// ```
    pub fn placed(&self, transform: Transform) -> Result<Shape> {
        let transform = finite_transform("placement", transform)?;
        if !is_rigid(&transform) {
            return Err(invalid("a placement must be a rotation and translation"));
        }
        Active::placed(self.raw(), &transform).map(Shape::from_raw)
    }

    /// The same topology with its orientation flipped: a face's normal points the other
    /// way, and a solid turned inside out measures a negative volume. It compares equal to
    /// the original, because `==` ignores orientation.
    ///
    /// # Errors
    /// [`Error::Operation`](crate::Error::Operation) if the kernel fails.
    ///
    /// # Examples
    /// ```
    /// use occt::prelude::*;
    ///
    /// # fn main() -> occt::Result<()> {
    /// let square = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]];
    /// let face = Polygon::new(square).build()?;
    /// let flipped = face.reversed()?;
    /// assert_eq!(flipped, *face);
    ///
    /// // A thickness grows along the normal, so this one grows the other way.
    /// let up = face.thickened(1.0)?.bounding_box()?.center().z;
    /// let down = flipped.thickened(1.0)?.bounding_box()?.center().z;
    /// assert!((up - 0.5).abs() < 1e-6 && (down + 0.5).abs() < 1e-6);
    /// # Ok(())
    /// # }
    /// ```
    pub fn reversed(&self) -> Result<Shape> {
        Active::reversed(self.raw()).map(Shape::from_raw)
    }
}
