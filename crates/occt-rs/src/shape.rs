//! The shape type: identity and measurement.

use std::fmt;
use std::hash::{Hash, Hasher};

use occt_kernel::{Aabb, Point3, Properties, Result, ShapeKind, Topology};

use crate::kernel::{Active, RawShape};

/// A shape: a solid, a face, an edge, a vertex or a group of them.
///
/// `Shape` owns a reference to its topology, so [`Clone`] is cheap and shapes release
/// themselves when dropped. Modelling operations never change a shape: they all return
/// a new one, which is why methods are named like [`translated`](Shape::translated).
///
/// When you know a shape's kind, the refinements [`Solid`](crate::Solid),
/// [`Face`](crate::Face), [`Edge`](crate::Edge) and [`Vertex`](crate::Vertex) add what
/// only makes sense for that kind.
///
/// # Equality
/// `==` and [`Hash`] compare *identity*, not geometry: whether two values refer to the
/// same piece of topology at the same place, whichever way round it is used. A clone
/// equals its original, and an edge reached through either of its two faces equals
/// itself, but two boxes built separately are not equal even when they coincide. To
/// compare geometry, compare measurements.
///
/// ```
/// use occt::prelude::*;
/// use std::collections::HashSet;
///
/// # fn main() -> occt::Result<()> {
/// let a = Cuboid::new([1.0, 1.0, 1.0]).build()?;
/// assert_eq!(a, a.clone());
/// assert_ne!(a, Cuboid::new([1.0, 1.0, 1.0]).build()?);
///
/// // Faces share edges, and a set sees through that.
/// let mut edges = HashSet::new();
/// for face in a.faces()? {
///     edges.extend(face.edges()?);
/// }
/// assert_eq!(edges.len(), 12);
/// # Ok(())
/// # }
/// ```
///
/// # Threads
/// `Shape` is neither [`Send`] nor [`Sync`], because the active kernel's shapes are not.
/// Keep one thread per kernel user, or move a [`Mesh`](crate::Mesh) or STEP bytes
/// between threads. See the [crate documentation](crate#threads).
#[derive(Clone)]
pub struct Shape {
    raw: RawShape,
}

impl Shape {
    pub(crate) fn from_raw(raw: RawShape) -> Self {
        Self { raw }
    }

    pub(crate) fn raw(&self) -> &RawShape {
        &self.raw
    }

    /// The topological kind of this shape.
    ///
    /// Combining solids may produce a compound, so test a result's kind rather than
    /// assuming it, or use [`solids`](Shape::solids), which sees through grouping.
    ///
    /// # Errors
    /// [`Error::Operation`](crate::Error::Operation) if the kernel cannot tell.
    pub fn kind(&self) -> Result<ShapeKind> {
        Active::kind(&self.raw)
    }

    /// Whether the shape's topology and geometry pass the kernel's validity check.
    ///
    /// # Errors
    /// [`Error::Operation`](crate::Error::Operation) if the check itself cannot run.
    pub fn is_valid(&self) -> Result<bool> {
        Active::is_valid(&self.raw)
    }

    /// The enclosed volume. Zero for a shape that encloses nothing, such as a face.
    ///
    /// # Errors
    /// [`Error::Operation`](crate::Error::Operation) if it cannot be computed.
    pub fn volume(&self) -> Result<f64> {
        Active::volume(&self.raw)
    }

    /// The surface area.
    ///
    /// # Errors
    /// [`Error::Operation`](crate::Error::Operation) if it cannot be computed.
    pub fn area(&self) -> Result<f64> {
        Active::area(&self.raw)
    }

    /// The centre of mass, assuming uniform density. For a shape that encloses no volume
    /// this is the centre of its surface.
    ///
    /// # Errors
    /// [`Error::Operation`](crate::Error::Operation) if it cannot be computed.
    pub fn center_of_mass(&self) -> Result<Point3> {
        Active::center_of_mass(&self.raw)
    }

    /// A tight axis-aligned bounding box.
    ///
    /// # Errors
    /// [`Error::Operation`](crate::Error::Operation) if it cannot be computed, as for an
    /// empty shape.
    pub fn bounding_box(&self) -> Result<Aabb> {
        Active::bounding_box(&self.raw)
    }
}

impl fmt::Debug for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Deliberately not the kernel's own Debug: nothing backend-shaped leaks here.
        let mut out = f.debug_struct("Shape");
        match self.kind() {
            Ok(kind) => out.field("kind", &kind),
            Err(_) => out.field("kind", &format_args!("unknown")),
        };
        out.field(
            "id",
            &format_args!("{:#010x}", Active::hash_code(&self.raw)),
        )
        .finish()
    }
}

impl PartialEq for Shape {
    fn eq(&self, other: &Self) -> bool {
        Active::is_same(&self.raw, &other.raw)
    }
}

impl Eq for Shape {}

impl Hash for Shape {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u32(Active::hash_code(&self.raw));
    }
}

impl AsRef<Shape> for Shape {
    fn as_ref(&self) -> &Shape {
        self
    }
}
