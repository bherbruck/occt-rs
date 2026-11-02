//! Questions about shapes that don't change them: measuring, locating, adjacency,
//! tessellation that keeps faces apart, and hidden-line drawings.

use occt_kernel::{
    self as contract, CancelFlag, Deflection, Dir3, DrawingView, Frame, Inspection, MeshOptions,
    Point3, Result, ShapeKind, ShapeTessellation, Topology, Vec3,
};

pub use occt_kernel::{
    Distance, DrawingLine, DrawingLineKind, EdgeConvexity, MassProperties, PointLocation,
};

use crate::Wire;
use crate::kernel::Active;
use crate::shape::Shape;
use crate::subshape::{Edge, Face, Solid, Vertex};
use crate::validate::{invalid, point, positive, vector};

/// A ray meeting a face, from [`Shape::ray_cast`]. The [`face`](occt_kernel::RayHit::face)
/// compares equal to the same face from [`Shape::faces`].
pub type RayHit = contract::RayHit<Face>;

/// How close a point must be to the boundary to count as on it, by default: the distance
/// at which modelling kernels conventionally treat points as coincident.
const DEFAULT_LOCATE_TOLERANCE: f64 = 1e-7;

/// How far a drawing's curves may stray from the exact projection by default, in model
/// units: well under what shows at any usual plotting scale for millimetre models.
const DEFAULT_DRAWING_DEFLECTION: f64 = 0.01;

fn typed<T: TryFrom<Shape, Error = occt_kernel::Error>>(
    raw: impl IntoIterator<Item = crate::kernel::RawShape>,
) -> Result<Vec<T>> {
    raw.into_iter()
        .map(|raw| T::try_from(Shape::from_raw(raw)))
        .collect()
}

/// Kernels report a foreign subshape too, but in their own words and as an operation
/// failure. The caller's mistake should read the same on every kernel.
fn require_part_of(part: &Shape, root: &Shape, op: &str) -> Result<()> {
    let own = Active::subshapes(root.raw(), part.kind()?)?;
    if own.iter().any(|raw| Active::is_same(raw, part.raw())) {
        Ok(())
    } else {
        Err(invalid(format!(
            "{op} was given a {} of a different shape",
            part.kind()?
        )))
    }
}

impl Shape {
    /// The mass, centre and inertia of this shape at unit density.
    ///
    /// The shape counts by its highest dimension: a solid's mass is its volume, a face's
    /// its area and an edge's its length. Multiply [`mass`](MassProperties::mass) and
    /// [`inertia`](MassProperties::inertia) by a density to get physical values. Inertia
    /// is about the centre of mass with the usual signs, so `inertia[0][1]` is `-∫xy dm`.
    ///
    /// # Errors
    /// [`Error::Operation`](crate::Error::Operation) if the shape has nothing to measure,
    /// such as a lone vertex.
    ///
    /// # Examples
    /// ```
    /// # use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let brick = Cuboid::new([1.0, 2.0, 3.0]).build()?;
    /// let props = brick.mass_properties()?;
    /// assert!((props.mass - 6.0).abs() < 1e-6);
    /// assert!(props.center.distance([0.5, 1.0, 1.5].into()) < 1e-6);
    /// // Around X, a box of mass m has m (y² + z²) / 12.
    /// assert!((props.inertia[0][0] - 6.0 * (4.0 + 9.0) / 12.0).abs() < 1e-6);
    /// # Ok(())
    /// # }
    /// ```
    pub fn mass_properties(&self) -> Result<MassProperties> {
        Active::mass_properties(self.raw())
    }

    /// The shortest distance to another shape, with the point on each that realise it.
    ///
    /// Shapes that touch or overlap are at distance zero, as is a shape inside a solid: a
    /// solid is its whole volume here, not only its surface.
    ///
    /// # Errors
    /// [`Error::Operation`](crate::Error::Operation) if it cannot be computed.
    ///
    /// # Examples
    /// ```
    /// # use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let a = Cuboid::new([1.0, 1.0, 1.0]).build()?;
    /// let b = Cuboid::new([1.0, 1.0, 1.0]).at([4.0, 0.0, 0.0]).build()?;
    /// let gap = a.distance_to(&b)?;
    /// assert!((gap.value - 3.0).abs() < 1e-6);
    /// assert!((gap.on_a.x - 1.0).abs() < 1e-6);
    /// assert!((gap.on_b.x - 4.0).abs() < 1e-6);
    /// # Ok(())
    /// # }
    /// ```
    pub fn distance_to(&self, other: &Shape) -> Result<Distance> {
        Active::distance(self.raw(), other.raw())
    }

    /// Every face a ray crosses, nearest first.
    ///
    /// Use it for picking: each hit's face compares equal to the same face from
    /// [`faces`](Shape::faces). The direction needn't have unit length; distances are in
    /// model units regardless. A ray through an edge shared by two faces hits both.
    ///
    /// # Errors
    /// [`Error::InvalidInput`](crate::Error::InvalidInput) if the origin isn't finite or
    /// the direction has no usable length; [`Error::Operation`](crate::Error::Operation)
    /// if the kernel fails. Missing is not an error, but an empty list.
    ///
    /// # Examples
    /// ```
    /// # use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let cube = Cuboid::new([2.0, 2.0, 2.0]).build()?;
    /// let hits = cube.ray_cast([1.0, 1.0, -5.0], [0.0, 0.0, 1.0])?;
    /// assert_eq!(hits.len(), 2);
    /// assert!((hits[0].distance - 5.0).abs() < 1e-6);
    /// assert!((hits[1].distance - 7.0).abs() < 1e-6);
    /// assert!(cube.faces()?.any(|face| face == hits[0].face));
    /// # Ok(())
    /// # }
    /// ```
    pub fn ray_cast(
        &self,
        origin: impl Into<Point3>,
        direction: impl Into<Vec3>,
    ) -> Result<Vec<RayHit>> {
        self.cast(origin.into(), direction.into(), None)
    }

    /// [`ray_cast`](Shape::ray_cast) that ignores what's further than `max_distance`.
    ///
    /// # Errors
    /// As [`ray_cast`](Shape::ray_cast), and
    /// [`Error::InvalidInput`](crate::Error::InvalidInput) if `max_distance` isn't finite
    /// and positive.
    ///
    /// # Examples
    /// ```
    /// # use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let cube = Cuboid::new([2.0, 2.0, 2.0]).build()?;
    /// let hits = cube.ray_cast_within([1.0, 1.0, -5.0], [0.0, 0.0, 1.0], 6.0)?;
    /// assert_eq!(hits.len(), 1);
    /// # Ok(())
    /// # }
    /// ```
    pub fn ray_cast_within(
        &self,
        origin: impl Into<Point3>,
        direction: impl Into<Vec3>,
        max_distance: f64,
    ) -> Result<Vec<RayHit>> {
        let limit = positive("ray length", max_distance)?;
        self.cast(origin.into(), direction.into(), Some(limit))
    }

    fn cast(&self, origin: Point3, direction: Vec3, limit: Option<f64>) -> Result<Vec<RayHit>> {
        let origin = point("ray origin", origin)?;
        let direction = Dir3::new(direction)?;
        Active::ray_cast(self.raw(), origin, direction, limit)?
            .into_iter()
            .map(|hit| {
                Ok(RayHit {
                    face: Face::try_from(Shape::from_raw(hit.face))?,
                    distance: hit.distance,
                    point: hit.point,
                    uv: hit.uv,
                })
            })
            .collect()
    }

    /// The subshapes of `kind` within `root` that contain this shape, in a stable order.
    ///
    /// [`Edge::faces_in`] and its like say the same with typed results; this one reaches
    /// any kind, such as the solid of a face.
    ///
    /// # Errors
    /// [`Error::InvalidInput`](crate::Error::InvalidInput) if this shape isn't part of
    /// `root`; [`Error::Operation`](crate::Error::Operation) if the kernel fails.
    ///
    /// # Examples
    /// ```
    /// # use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let cube = Cuboid::new([1.0, 1.0, 1.0]).build()?;
    /// let face = cube.faces()?.next().expect("a cube has faces");
    /// let owners = face.ancestors_in(&cube, ShapeKind::Solid)?;
    /// assert_eq!(owners.len(), 1);
    /// assert_eq!(owners[0], *cube);
    /// # Ok(())
    /// # }
    /// ```
    pub fn ancestors_in(&self, root: &Shape, kind: ShapeKind) -> Result<Vec<Shape>> {
        require_part_of(self, root, "an adjacency query")?;
        let raw = Active::ancestors(root.raw(), self.raw(), kind)?;
        Ok(raw.into_iter().map(Shape::from_raw).collect())
    }

    /// The edges of this shape, which must be a wire, in the order they connect.
    ///
    /// An edge the wire passes along twice appears twice, as the seam does in the wire
    /// around a cylinder's side.
    ///
    /// # Errors
    /// [`Error::WrongKind`](crate::Error::WrongKind) if this isn't a wire;
    /// [`Error::Operation`](crate::Error::Operation) if its edges don't form one chain.
    ///
    /// # Examples
    /// ```
    /// # use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let square = Polygon::new([
    ///     [0.0, 0.0, 0.0],
    ///     [1.0, 0.0, 0.0],
    ///     [1.0, 1.0, 0.0],
    ///     [0.0, 1.0, 0.0],
    /// ])
    /// .build()?;
    /// let outer = &square.wires()?[0];
    /// assert_eq!(outer.wire_edges()?.len(), 4);
    /// # Ok(())
    /// # }
    /// ```
    pub fn wire_edges(&self) -> Result<Vec<Edge>> {
        typed(Active::wire_edges(self.raw())?)
    }

    /// Starts a tessellation that keeps faces and edges apart, for display and picking.
    ///
    /// In the result, `faces[i]` belongs to the `i`th item of [`faces`](Shape::faces) and
    /// `edges[i]` to the `i`th of [`edges`](Shape::edges), and an edge's polyline runs
    /// through vertices of the meshes on either side. For one [`Mesh`](crate::Mesh)
    /// use [`ShapeTessellation::merged`] or [`Shape::mesh`].
    ///
    /// # Examples
    /// ```
    /// # use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let cube = Cuboid::new([1.0, 1.0, 1.0]).build()?;
    /// let parts = cube.tessellate(MeshOptions::default()).build()?;
    /// assert_eq!(parts.faces.len(), 6);
    /// assert_eq!(parts.edges.len(), 12);
    /// assert!((parts.merged().volume() - 1.0).abs() < 1e-6);
    /// # Ok(())
    /// # }
    /// ```
    pub fn tessellate(&self, options: MeshOptions) -> Tessellate<'_> {
        Tessellate {
            shape: self,
            options,
            cancel: None,
        }
    }

    /// A hidden-line drawing of this shape, as drafting views need.
    ///
    /// Lines come back as polylines in view coordinates, each marked as
    /// [`Sharp`](DrawingLineKind::Sharp), [`Smooth`](DrawingLineKind::Smooth) or
    /// [`Outline`](DrawingLineKind::Outline), and visible or hidden. Edges are split
    /// wherever their visibility changes. The seam of a closed surface is left out, not
    /// being a feature of the part.
    ///
    /// # Errors
    /// [`Error::InvalidInput`](crate::Error::InvalidInput) if the view has a bad value;
    /// [`Error::Operation`](crate::Error::Operation) if the kernel fails.
    ///
    /// # Examples
    /// ```
    /// use occt::prelude::*;
    /// use occt::View;
    ///
    /// # fn main() -> occt::Result<()> {
    /// let cube = Cuboid::new([1.0, 1.0, 1.0]).build()?;
    /// let lines = cube.hidden_lines(View::top())?;
    /// let visible = lines.iter().filter(|line| line.visible).count();
    /// assert_eq!(visible, 4);
    /// # Ok(())
    /// # }
    /// ```
    pub fn hidden_lines(&self, view: View) -> Result<Vec<DrawingLine>> {
        Active::hidden_lines(self.raw(), &view.resolve()?)
    }
}

impl Solid {
    /// Where a point lies relative to this solid. Points within about 1e-7 of the
    /// boundary count as on it; see [`locate_within`](Solid::locate_within).
    ///
    /// # Errors
    /// [`Error::InvalidInput`](crate::Error::InvalidInput) if the point isn't finite;
    /// [`Error::Operation`](crate::Error::Operation) if it can't be classified.
    ///
    /// # Examples
    /// ```
    /// # use occt::prelude::*;
    /// use occt::PointLocation;
    /// # fn main() -> occt::Result<()> {
    /// let cube = Cuboid::new([2.0, 2.0, 2.0]).build()?;
    /// assert_eq!(cube.locate([1.0, 1.0, 1.0])?, PointLocation::Inside);
    /// assert_eq!(cube.locate([2.0, 1.0, 1.0])?, PointLocation::OnBoundary);
    /// assert_eq!(cube.locate([3.0, 1.0, 1.0])?, PointLocation::Outside);
    /// # Ok(())
    /// # }
    /// ```
    pub fn locate(&self, point: impl Into<Point3>) -> Result<PointLocation> {
        self.locate_within(point, DEFAULT_LOCATE_TOLERANCE)
    }

    /// [`locate`](Solid::locate) that counts points within `tolerance` of the boundary as
    /// on it.
    ///
    /// # Errors
    /// As [`locate`](Solid::locate), and
    /// [`Error::InvalidInput`](crate::Error::InvalidInput) if `tolerance` isn't finite
    /// and positive.
    ///
    /// # Examples
    /// ```
    /// # use occt::prelude::*;
    /// use occt::PointLocation;
    /// # fn main() -> occt::Result<()> {
    /// let cube = Cuboid::new([2.0, 2.0, 2.0]).build()?;
    /// let near = [2.0 + 1e-3, 1.0, 1.0];
    /// assert_eq!(cube.locate(near)?, PointLocation::Outside);
    /// assert_eq!(cube.locate_within(near, 1e-2)?, PointLocation::OnBoundary);
    /// # Ok(())
    /// # }
    /// ```
    pub fn locate_within(&self, at: impl Into<Point3>, tolerance: f64) -> Result<PointLocation> {
        let at = point("point to locate", at.into())?;
        let tolerance = positive("tolerance", tolerance)?;
        Active::locate_point(self.raw(), at, tolerance)
    }
}

impl Edge {
    /// The faces of `shape` that this edge borders: two on a closed solid.
    ///
    /// # Errors
    /// [`Error::InvalidInput`](crate::Error::InvalidInput) if this edge isn't part of
    /// `shape`; [`Error::Operation`](crate::Error::Operation) if the kernel fails.
    ///
    /// # Examples
    /// ```
    /// # use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let cube = Cuboid::new([1.0, 1.0, 1.0]).build()?;
    /// for edge in cube.edges()? {
    ///     assert_eq!(edge.faces_in(&cube)?.len(), 2);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn faces_in(&self, shape: &Shape) -> Result<Vec<Face>> {
        require_part_of(self, shape, "an adjacency query")?;
        typed(Active::ancestors(shape.raw(), self.raw(), ShapeKind::Face)?)
    }

    /// How the faces of `shape` meet at this edge. Faces within about 0.06 degrees of
    /// each other count as [`Tangent`](EdgeConvexity::Tangent), since blends are only
    /// tangent that closely.
    ///
    /// # Errors
    /// [`Error::InvalidInput`](crate::Error::InvalidInput) if this edge isn't part of
    /// `shape`. [`Error::Operation`](crate::Error::Operation) if there is no one answer:
    /// the edge has more than two faces, is convex in places and concave in others, or
    /// its faces fold back onto each other.
    ///
    /// # Examples
    /// ```
    /// # use occt::prelude::*;
    /// use occt::EdgeConvexity;
    /// # fn main() -> occt::Result<()> {
    /// let cube = Cuboid::new([1.0, 1.0, 1.0]).build()?;
    /// let edge = cube.edges()?.next().expect("a cube has edges");
    /// assert_eq!(edge.convexity_in(&cube)?, EdgeConvexity::Convex);
    /// # Ok(())
    /// # }
    /// ```
    pub fn convexity_in(&self, shape: &Shape) -> Result<EdgeConvexity> {
        require_part_of(self, shape, "a convexity query")?;
        Active::edge_convexity(shape.raw(), self.raw())
    }
}

impl Vertex {
    /// The faces of `shape` that meet at this vertex: three at a box's corner.
    ///
    /// # Errors
    /// [`Error::InvalidInput`](crate::Error::InvalidInput) if this vertex isn't part of
    /// `shape`; [`Error::Operation`](crate::Error::Operation) if the kernel fails.
    ///
    /// # Examples
    /// ```
    /// # use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let cube = Cuboid::new([1.0, 1.0, 1.0]).build()?;
    /// let corner = cube.vertices()?.next().expect("a cube has corners");
    /// assert_eq!(corner.faces_in(&cube)?.len(), 3);
    /// # Ok(())
    /// # }
    /// ```
    pub fn faces_in(&self, shape: &Shape) -> Result<Vec<Face>> {
        require_part_of(self, shape, "an adjacency query")?;
        typed(Active::ancestors(shape.raw(), self.raw(), ShapeKind::Face)?)
    }

    /// The edges of `shape` that end at this vertex: three at a box's corner.
    ///
    /// # Errors
    /// [`Error::InvalidInput`](crate::Error::InvalidInput) if this vertex isn't part of
    /// `shape`; [`Error::Operation`](crate::Error::Operation) if the kernel fails.
    ///
    /// # Examples
    /// ```
    /// # use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let cube = Cuboid::new([1.0, 1.0, 1.0]).build()?;
    /// let corner = cube.vertices()?.next().expect("a cube has corners");
    /// assert_eq!(corner.edges_in(&cube)?.len(), 3);
    /// # Ok(())
    /// # }
    /// ```
    pub fn edges_in(&self, shape: &Shape) -> Result<Vec<Edge>> {
        require_part_of(self, shape, "an adjacency query")?;
        typed(Active::ancestors(shape.raw(), self.raw(), ShapeKind::Edge)?)
    }
}

impl Face {
    /// The wires bounding this face: the outer boundary first, then one per hole.
    /// Read each with [`Shape::wire_edges`].
    ///
    /// # Errors
    /// [`Error::Operation`](crate::Error::Operation) if the kernel fails.
    ///
    /// # Examples
    /// ```
    /// # use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let cube = Cuboid::new([1.0, 1.0, 1.0]).build()?;
    /// let face = cube.faces()?.next().expect("a cube has faces");
    /// let wires = face.wires()?;
    /// assert_eq!(wires.len(), 1);
    /// assert_eq!(wires[0].kind()?, ShapeKind::Wire);
    /// # Ok(())
    /// # }
    /// ```
    pub fn wires(&self) -> Result<Vec<Wire>> {
        let raw = Active::face_wires(self.raw())?;
        raw.into_iter()
            .map(|w| Wire::try_from(Shape::from_raw(w)))
            .collect()
    }
}

/// A builder for a structured tessellation, from [`Shape::tessellate`].
#[derive(Clone, Debug)]
#[must_use = "a builder does nothing until `build` is called"]
pub struct Tessellate<'a> {
    shape: &'a Shape,
    options: MeshOptions,
    cancel: Option<CancelFlag>,
}

impl Tessellate<'_> {
    /// Stops when `flag` is set from another thread, which ends the build with
    /// [`Error::Cancelled`](crate::Error::Cancelled). The kernel checks it between faces.
    /// A flag already set stops the build on any platform, but one set later only can
    /// where there are threads and shared memory, so not on WebAssembly.
    ///
    /// # Examples
    /// ```
    /// use occt::prelude::*;
    /// use occt::CancelFlag;
    ///
    /// # fn main() -> occt::Result<()> {
    /// let cube = Cuboid::new([1.0, 1.0, 1.0]).build()?;
    /// let flag = CancelFlag::new();
    /// flag.cancel();
    /// let result = cube.tessellate(MeshOptions::default()).cancel_with(&flag).build();
    /// assert!(matches!(result, Err(Error::Cancelled)));
    /// # Ok(())
    /// # }
    /// ```
    pub fn cancel_with(mut self, flag: &CancelFlag) -> Self {
        self.cancel = Some(flag.clone());
        self
    }

    /// Builds the tessellation.
    ///
    /// # Errors
    /// [`Error::InvalidInput`](crate::Error::InvalidInput) if a deflection isn't finite
    /// and positive; [`Error::Cancelled`](crate::Error::Cancelled) if stopped;
    /// [`Error::Operation`](crate::Error::Operation) if the kernel fails on a face.
    pub fn build(self) -> Result<ShapeTessellation> {
        match self.options.deflection {
            Deflection::Absolute(distance) => positive("absolute deflection", distance)?,
            Deflection::Relative(fraction) => positive("relative deflection", fraction)?,
        };
        positive(
            "angular deflection",
            self.options.angular_deflection.as_radians(),
        )?;
        Active::tessellate(self.shape.raw(), &self.options, self.cancel.as_ref())
    }
}

/// Where a hidden-line drawing looks from, for [`Shape::hidden_lines`].
///
/// A view has a parallel projection unless [`perspective`](View::perspective) says
/// otherwise. Its coordinates have X to the right and Y up as the viewer sees them,
/// measured from [`looking_at`](View::looking_at).
///
/// # Examples
/// ```
/// use occt::View;
///
/// let view = View::from_direction([1.0, -2.0, 1.0])
///     .up([0.0, 0.0, 1.0])
///     .looking_at([5.0, 5.0, 5.0])
///     .deflection(0.001);
/// # drop(view);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
#[must_use = "a view does nothing until passed to `hidden_lines`"]
pub struct View {
    toward_viewer: Vec3,
    up: Option<Vec3>,
    target: Point3,
    perspective: Option<f64>,
    deflection: f64,
}

impl View {
    /// A view from `toward_viewer`, the direction from the model to the viewer. Unless
    /// [`up`](View::up) says otherwise, +Z is up, or +Y when looking along Z.
    pub fn from_direction(toward_viewer: impl Into<Vec3>) -> Self {
        Self {
            toward_viewer: toward_viewer.into(),
            up: None,
            target: Point3::ORIGIN,
            perspective: None,
            deflection: DEFAULT_DRAWING_DEFLECTION,
        }
    }

    /// Looking down from +Z: +X to the right and +Y up.
    pub fn top() -> Self {
        Self::from_direction([0.0, 0.0, 1.0]).up([0.0, 1.0, 0.0])
    }

    /// Looking from -Y: +X to the right and +Z up.
    pub fn front() -> Self {
        Self::from_direction([0.0, -1.0, 0.0]).up([0.0, 0.0, 1.0])
    }

    /// Looking from +X: +Y to the right and +Z up.
    pub fn right() -> Self {
        Self::from_direction([1.0, 0.0, 0.0]).up([0.0, 0.0, 1.0])
    }

    /// The usual isometric view, from the front, right and above, with +Z up.
    pub fn isometric() -> Self {
        Self::from_direction([1.0, -1.0, 1.0]).up([0.0, 0.0, 1.0])
    }

    /// Sets which way is up. It need only be roughly so: what counts is its part
    /// perpendicular to the view direction.
    pub fn up(mut self, up: impl Into<Vec3>) -> Self {
        self.up = Some(up.into());
        self
    }

    /// Sets the point that becomes the drawing's origin. A perspective's eye faces it.
    pub fn looking_at(mut self, target: impl Into<Point3>) -> Self {
        self.target = target.into();
        self
    }

    /// Uses a perspective projection, with the eye this far from the
    /// [`looking_at`](View::looking_at) point. Geometry beyond the eye can't be drawn.
    pub fn perspective(mut self, distance: f64) -> Self {
        self.perspective = Some(distance);
        self
    }

    /// Sets how far polylines may stray from the exact projected curves, in model units.
    pub fn deflection(mut self, deflection: f64) -> Self {
        self.deflection = deflection;
        self
    }

    fn resolve(self) -> Result<DrawingView> {
        let z = Dir3::new(self.toward_viewer)?;
        let target = point("view target", self.target)?;
        let up = match self.up {
            Some(up) => vector("view up direction", up)?,
            None if z.vector().z.abs() > 0.999 => Vec3::new(0.0, 1.0, 0.0),
            None => Vec3::new(0.0, 0.0, 1.0),
        };
        // X is what's right of the viewer: up × toward the viewer.
        let frame = Frame::new(target, z, up.cross(z.vector()))
            .map_err(|_| invalid("the view's up direction must not be along the view"))?;
        if let Some(distance) = self.perspective {
            positive("perspective distance", distance)?;
        }
        Ok(DrawingView {
            frame,
            perspective: self.perspective,
            deflection: positive("drawing deflection", self.deflection)?,
        })
    }
}
