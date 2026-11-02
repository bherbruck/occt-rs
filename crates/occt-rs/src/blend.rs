//! Fillets and chamfers.

use std::borrow::Borrow;
use std::collections::HashSet;

use occt_kernel::{Blends, Result};

use crate::kernel::Active;
use crate::shape::Shape;
use crate::subshape::Edge;
use crate::validate::{invalid, positive};

/// Which edges a blend applies to.
#[derive(Clone, Debug)]
enum Selection {
    All,
    Listed(Vec<Edge>),
}

impl Selection {
    fn add(&mut self, edges: impl IntoIterator<Item: Borrow<Edge>>) {
        match self {
            Self::All => {}
            Self::Listed(listed) => listed.extend(edges.into_iter().map(|e| e.borrow().clone())),
        }
    }

    /// The distinct edges to blend, checked against the shape that will be blended.
    fn resolve(self, shape: &Shape, op: &str) -> Result<Vec<Edge>> {
        let own: Vec<Edge> = shape.edges()?.collect();
        let edges = match self {
            Self::All => own,
            Self::Listed(listed) => {
                let own: HashSet<Edge> = own.into_iter().collect();
                if !listed.iter().all(|edge| own.contains(edge)) {
                    return Err(invalid(format!(
                        "{op} was given an edge of a different shape"
                    )));
                }
                let mut seen = HashSet::new();
                listed
                    .into_iter()
                    .filter(|edge| seen.insert(edge.clone()))
                    .collect()
            }
        };
        if edges.is_empty() {
            return Err(invalid(format!("{op} needs at least one edge")));
        }
        Ok(edges)
    }
}

/// Declares a blend builder and the [`Shape`] method that starts one. The two blends
/// differ only in the number they take and the kernel call they end in.
macro_rules! blend {
    (
        $(#[$doc:meta])*
        $builder:ident, $method:ident, $op:literal, $size:ident, $kernel:ident,
        $(#[$method_doc:meta])*
    ) => {
        $(#[$doc])*
        #[derive(Clone, Debug)]
        #[must_use = "a builder does nothing until `build` is called"]
        pub struct $builder<'a> {
            shape: &'a Shape,
            $size: f64,
            selection: Selection,
        }

        impl $builder<'_> {
            /// Selects every edge of the shape.
            pub fn all_edges(mut self) -> Self {
                self.selection = Selection::All;
                self
            }

            /// Adds edges to the selection, as references or owned. Selecting an edge
            /// twice is the same as selecting it once.
            pub fn edges(mut self, edges: impl IntoIterator<Item: Borrow<Edge>>) -> Self {
                self.selection.add(edges);
                self
            }

            /// Builds the blended shape. The original is unchanged.
            ///
            /// # Errors
            /// [`Error::InvalidInput`](crate::Error::InvalidInput) if the size is not finite
            /// and positive, no edges are selected, or an edge does not belong to this
            /// shape; [`Error::Operation`](crate::Error::Operation) if the kernel cannot
            /// build it, as when the size is too large for the shape.
            pub fn build(self) -> Result<Shape> {
                let size = positive(concat!($op, " size"), self.$size)?;
                let edges = self.selection.resolve(self.shape, $op)?;
                let raw: Vec<_> = edges.iter().map(|edge| edge.raw()).collect();
                Active::$kernel(self.shape.raw(), &raw, size).map(Shape::from_raw)
            }
        }

        impl Shape {
            $(#[$method_doc])*
            pub fn $method(&self, $size: f64) -> $builder<'_> {
                $builder { shape: self, $size, selection: Selection::Listed(Vec::new()) }
            }
        }
    };
}

blend!(
    /// A builder for rounded edges, from [`Shape::fillet`].
    Fillet, fillet, "fillet", radius, fillet,
    /// Starts rounding edges to a radius. Choose them on the returned builder.
    ///
    /// # Examples
    /// ```
    /// # use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let cube = Cuboid::new([10.0, 10.0, 10.0]).build()?;
    /// let soft = cube.fillet(1.0).all_edges().build()?;
    /// assert!(soft.volume()? < cube.volume()?);
    ///
    /// let edge = cube.edges()?.next().expect("a cube has edges");
    /// let one = cube.fillet(1.0).edges([&edge]).build()?;
    /// assert!(one.volume()? > soft.volume()?);
    /// # Ok(())
    /// # }
    /// ```
);

blend!(
    /// A builder for bevelled edges, from [`Shape::chamfer`].
    Chamfer, chamfer, "chamfer", distance, chamfer,
    /// Starts bevelling edges by a distance. Choose them on the returned builder.
    ///
    /// # Examples
    /// ```
    /// # use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let cube = Cuboid::new([10.0, 10.0, 10.0]).build()?;
    /// let edge = cube.edges()?.next().expect("a cube has edges");
    /// let cut = cube.chamfer(2.0).edges([&edge]).build()?;
    /// assert!((cut.volume()? - (1000.0 - 2.0 * 2.0 / 2.0 * 10.0)).abs() < 1e-3);
    /// # Ok(())
    /// # }
    /// ```
);
