//! Typed refinements of [`Shape`], and traversal that produces them.

use std::borrow::Borrow;
use std::ops::Deref;

use occt_kernel::{CurveKind, Error, Point3, Properties, Result, ShapeKind, SurfaceKind, Topology};

use crate::kernel::Active;
use crate::shape::Shape;

/// Declares a refinement of [`Shape`] and every conversion it has, so that no kind can
/// drift from the others.
macro_rules! refinement {
    ($(#[$doc:meta])* $name:ident, $kind:expr, $iter:ident, $plural:literal) => {
        $(#[$doc])*
        ///
        /// This is a refinement of [`Shape`], in the way [`String`] refines [`str`]: it
        /// [`Deref`]s to `Shape`, so every shape method is available and `&Self` coerces
        /// to `&Shape`. It compares and hashes exactly as the shape it holds. Convert with
        /// [`TryFrom`] (checks the kind) and [`From`] (always succeeds).
        #[derive(Clone, Debug, PartialEq, Eq, Hash)]
        pub struct $name(Shape);

        impl $name {
            /// The kind that every value of this type has.
            pub const KIND: ShapeKind = $kind;

            /// Wraps a shape that is known to have this kind.
            fn known(shape: Shape) -> Self {
                debug_assert!(shape.kind().is_ok_and(|kind| kind == Self::KIND));
                Self(shape)
            }
        }

        impl Shape {
            #[doc = concat!("The distinct ", $plural, " of this shape, in a stable order.")]
            ///
            /// The iterator owns its items and does not borrow the shape. Shapes that
            /// neighbours share, such as the edge between two faces, appear once. If this
            /// shape is itself of that kind, the iterator yields it.
            ///
            /// # Errors
            /// [`Error::Operation`] if the traversal fails, which is reported before
            /// any iteration starts.
            pub fn $iter(
                &self,
            ) -> Result<impl ExactSizeIterator<Item = $name> + DoubleEndedIterator + use<>> {
                let raw = Active::subshapes(self.raw(), $kind)?;
                Ok(raw.into_iter().map(|raw| $name::known(Shape::from_raw(raw))))
            }
        }

        impl TryFrom<Shape> for $name {
            type Error = Error;

            /// # Errors
            /// [`Error::WrongKind`] if the shape has a different kind.
            fn try_from(shape: Shape) -> Result<Self> {
                match shape.kind()? {
                    found if found == Self::KIND => Ok(Self(shape)),
                    found => Err(Error::WrongKind { expected: Self::KIND, found }),
                }
            }
        }

        impl TryFrom<&Shape> for $name {
            type Error = Error;

            /// # Errors
            /// [`Error::WrongKind`] if the shape has a different kind.
            fn try_from(shape: &Shape) -> Result<Self> {
                Self::try_from(shape.clone())
            }
        }

        impl From<$name> for Shape {
            fn from(value: $name) -> Shape {
                value.0
            }
        }

        impl Deref for $name {
            type Target = Shape;

            fn deref(&self) -> &Shape {
                &self.0
            }
        }

        impl AsRef<Shape> for $name {
            fn as_ref(&self) -> &Shape {
                &self.0
            }
        }

        impl Borrow<Shape> for $name {
            fn borrow(&self) -> &Shape {
                &self.0
            }
        }
    };
}

refinement!(
    /// A closed volume.
    Solid, ShapeKind::Solid, solids, "solids"
);
refinement!(
    /// A bounded piece of surface.
    Face, ShapeKind::Face, faces, "faces"
);
refinement!(
    /// A bounded piece of curve.
    Edge, ShapeKind::Edge, edges, "edges"
);
refinement!(
    /// A point in the topology, where edges end.
    Vertex, ShapeKind::Vertex, vertices, "vertices"
);

impl Vertex {
    /// The position of this vertex.
    ///
    /// # Errors
    /// [`Error::Operation`] if the kernel cannot read it.
    pub fn point(&self) -> Result<Point3> {
        Active::vertex_point(self.raw())
    }
}

impl Edge {
    /// The length of this edge.
    ///
    /// # Errors
    /// [`Error::Operation`] if it cannot be computed.
    pub fn length(&self) -> Result<f64> {
        Active::length(self.raw())
    }

    /// The kind of curve under this edge.
    ///
    /// # Errors
    /// [`Error::Operation`] if the kernel cannot read it.
    pub fn curve_kind(&self) -> Result<CurveKind> {
        Active::curve_kind(self.raw())
    }
}

impl Face {
    /// The kind of surface under this face.
    ///
    /// # Errors
    /// [`Error::Operation`] if the kernel cannot read it.
    pub fn surface_kind(&self) -> Result<SurfaceKind> {
        Active::surface_kind(self.raw())
    }
}
