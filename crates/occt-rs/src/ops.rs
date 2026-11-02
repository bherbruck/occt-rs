//! General booleans, sectioning, splitting and per-edge blends, with history.

use std::borrow::Borrow;
use std::collections::{HashMap, HashSet};

use occt_kernel::{AdvancedBlends, BooleanOps, BooleanOptions, Result};

pub use occt_kernel::{BooleanOp, ChamferSize, RadiusLaw};

use crate::kernel::Active;
use crate::shape::Shape;
use crate::subshape::{Edge, Face};
use crate::validate::{invalid, point, positive};
use crate::{Angle, CancelFlag, Modeled, Plane};

fn owned(shapes: impl IntoIterator<Item: AsRef<Shape>>) -> Vec<Shape> {
    shapes.into_iter().map(|s| s.as_ref().clone()).collect()
}

/// The options that booleans and splits share.
#[derive(Clone, Debug, Default)]
struct Options {
    fuzzy: Option<f64>,
    unify: bool,
    parallel: bool,
    cancel: Option<CancelFlag>,
}

impl Options {
    /// The kernel's form of these options, checked.
    fn to_kernel(&self, op: &str) -> Result<BooleanOptions> {
        if let Some(fuzzy) = self.fuzzy {
            positive(&format!("{op} fuzzy value"), fuzzy)?;
        }
        Ok(BooleanOptions {
            fuzzy: self.fuzzy,
            unify: self.unify,
            parallel: self.parallel,
            cancel: self.cancel.clone(),
        })
    }
}

fn raw_list(shapes: &[Shape]) -> Vec<&crate::kernel::RawShape> {
    shapes.iter().map(Shape::raw).collect()
}

/// A boolean between sets of shapes, with options and [`History`](crate::History).
///
/// Each side counts as the union of its shapes, and a side's shapes may overlap.
///
/// A cut's history reports a tool's surface that becomes a wall of the result under both
/// [`modified`](crate::History::modified) and [`generated`](crate::History::generated).
///
/// For a plain boolean between two shapes, [`Shape::cut`] and its siblings are shorter.
///
/// # Examples
/// Drill a plate with several tools at once, and ask what became of its top face:
/// ```
/// use occt::prelude::*;
/// use occt::{Boolean, CancelFlag};
/// use std::f64::consts::PI;
///
/// # fn main() -> occt::Result<()> {
/// let plate = Cuboid::new([10.0, 10.0, 1.0]).build()?;
/// let drill = |x: f64| {
///     Cylinder::new(1.0, 3.0)
///         .along(Axis::new([x, 5.0, -1.0].into(), Dir3::Z))
///         .build()
/// };
/// let (left, right) = (drill(3.0)?, drill(7.0)?);
/// let flag = CancelFlag::new();
///
/// let drilled = Boolean::cut([&plate])
///     .tools([&left, &right])
///     .fuzzy(1e-6)
///     .unify()
///     .cancel_with(&flag)
///     .build()?;
///
/// assert!((drilled.shape.volume()? - (100.0 - 2.0 * PI)).abs() < 1e-6);
/// let top = plate
///     .faces()?
///     .find(|f| f.center_of_mass().is_ok_and(|c| (c.z - 1.0).abs() < 1e-9))
///     .expect("a plate has a top");
/// assert_eq!(drilled.history.modified(&top).len(), 1);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug)]
#[must_use = "a builder does nothing until `build` is called"]
pub struct Boolean {
    op: BooleanOp,
    arguments: Vec<Shape>,
    tools: Vec<Shape>,
    options: Options,
}

impl Boolean {
    /// Starts a boolean of a kind chosen at run time.
    ///
    /// ```
    /// use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let a = Cuboid::new([2.0, 1.0, 1.0]).build()?;
    /// let b = Cuboid::new([1.0, 1.0, 1.0]).build()?;
    /// let rest = occt::Boolean::new(occt::BooleanOp::Cut, [&a]).tools([&b]).build()?;
    /// assert!((rest.shape.volume()? - 1.0).abs() < 1e-9);
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(op: BooleanOp, arguments: impl IntoIterator<Item: AsRef<Shape>>) -> Self {
        Self {
            op,
            arguments: owned(arguments),
            tools: Vec::new(),
            options: Options::default(),
        }
    }

    /// The union of the arguments and any [`tools`](Self::tools).
    ///
    /// ```
    /// use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let row: Vec<_> = (0..3)
    ///     .map(|i| Cuboid::new([1.0, 1.0, 1.0]).at([f64::from(i), 0.0, 0.0]).build())
    ///     .collect::<occt::Result<_>>()?;
    /// let bar = occt::Boolean::fuse(&row).unify().build()?;
    /// assert_eq!(bar.shape.faces()?.len(), 6);
    /// # Ok(())
    /// # }
    /// ```
    pub fn fuse(arguments: impl IntoIterator<Item: AsRef<Shape>>) -> Self {
        Self::new(BooleanOp::Fuse, arguments)
    }

    /// The arguments without the [`tools`](Self::tools), which are required.
    ///
    /// ```
    /// use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let a = Cuboid::new([2.0, 1.0, 1.0]).build()?;
    /// let b = Cuboid::new([1.0, 1.0, 1.0]).build()?;
    /// let rest = occt::Boolean::cut([&a]).tools([&b]).build()?;
    /// assert!((rest.shape.volume()? - 1.0).abs() < 1e-9);
    /// # Ok(())
    /// # }
    /// ```
    pub fn cut(arguments: impl IntoIterator<Item: AsRef<Shape>>) -> Self {
        Self::new(BooleanOp::Cut, arguments)
    }

    /// The part of the arguments that lies inside the [`tools`](Self::tools), which are
    /// required. Each side counts as the union of its shapes, so a point is in the result
    /// if it is in any argument and in any tool.
    ///
    /// When nothing is shared the result is a shape with no volume, not an error.
    ///
    /// ```
    /// use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let a = Cuboid::new([2.0, 2.0, 2.0]).build()?;
    /// let b = Cuboid::new([1.0, 2.0, 2.0]).at([1.0, 0.0, 0.0]).build()?;
    /// let c = Cuboid::new([2.0, 1.0, 2.0]).at([0.0, 1.0, 0.0]).build()?;
    /// let shared = occt::Boolean::common([&a]).tools([&b, &c]).build()?;
    /// assert!((shared.shape.volume()? - 6.0).abs() < 1e-9);
    /// # Ok(())
    /// # }
    /// ```
    pub fn common(arguments: impl IntoIterator<Item: AsRef<Shape>>) -> Self {
        Self::new(BooleanOp::Common, arguments)
    }

    /// Adds shapes on the other side of the operation.
    ///
    /// ```
    /// use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let a = Cuboid::new([2.0, 1.0, 1.0]).build()?;
    /// let b = Cuboid::new([1.0, 1.0, 1.0]).build()?;
    /// let c = Cuboid::new([1.0, 1.0, 1.0]).at([1.0, 0.0, 0.0]).build()?;
    /// let rest = occt::Boolean::cut([&a]).tools([&b]).tools([&c]).build()?;
    /// assert!(rest.shape.volume()?.abs() < 1e-9);
    /// # Ok(())
    /// # }
    /// ```
    pub fn tools(mut self, tools: impl IntoIterator<Item: AsRef<Shape>>) -> Self {
        self.tools.extend(owned(tools));
        self
    }

    /// Treats geometry closer than `tolerance` as coincident, which rescues near-tangent
    /// and nearly touching inputs. Must be finite and positive.
    ///
    /// ```
    /// use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let a = Cuboid::new([1.0, 1.0, 1.0]).build()?;
    /// let b = Cuboid::new([1.0, 1.0, 1.0]).at([1.001, 0.0, 0.0]).build()?;
    /// let joined = occt::Boolean::fuse([&a, &b]).fuzzy(0.01).build()?;
    /// assert_eq!(joined.shape.solids()?.len(), 1);
    /// # Ok(())
    /// # }
    /// ```
    pub fn fuzzy(mut self, tolerance: f64) -> Self {
        self.options.fuzzy = Some(tolerance);
        self
    }

    /// Merges faces that lie on the same surface, and edges on the same curve, in the
    /// result. The history follows.
    ///
    /// ```
    /// use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let a = Cuboid::new([1.0, 1.0, 1.0]).build()?;
    /// let b = Cuboid::new([1.0, 1.0, 1.0]).at([1.0, 0.0, 0.0]).build()?;
    /// let bar = occt::Boolean::fuse([&a, &b]).unify().build()?;
    /// assert_eq!(bar.shape.faces()?.len(), 6);
    /// # Ok(())
    /// # }
    /// ```
    pub fn unify(mut self) -> Self {
        self.options.unify = true;
        self
    }

    /// Uses several threads where the platform has them.
    ///
    /// ```
    /// use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let a = Cuboid::new([2.0, 1.0, 1.0]).build()?;
    /// let b = Cuboid::new([1.0, 1.0, 1.0]).build()?;
    /// let rest = occt::Boolean::cut([&a]).tools([&b]).parallel().build()?;
    /// assert!((rest.shape.volume()? - 1.0).abs() < 1e-9);
    /// # Ok(())
    /// # }
    /// ```
    pub fn parallel(mut self) -> Self {
        self.options.parallel = true;
        self
    }

    /// Lets `flag` stop the operation from another thread.
    ///
    /// ```
    /// use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let a = Cuboid::new([2.0, 1.0, 1.0]).build()?;
    /// let b = Cuboid::new([1.0, 1.0, 1.0]).build()?;
    /// let flag = occt::CancelFlag::new();
    /// flag.cancel();
    /// let stopped = occt::Boolean::cut([&a]).tools([&b]).cancel_with(&flag).build();
    /// assert!(matches!(stopped, Err(Error::Cancelled)));
    /// # Ok(())
    /// # }
    /// ```
    pub fn cancel_with(mut self, flag: &CancelFlag) -> Self {
        self.options.cancel = Some(flag.clone());
        self
    }

    /// Runs the operation. The inputs are unchanged.
    ///
    /// The result may be a compound; [`Shape::solids`] sees through it.
    ///
    /// # Errors
    /// [`Error::InvalidInput`](crate::Error::InvalidInput) if the fuzzy value isn't finite
    /// and positive or a side is empty; [`Error::Cancelled`](crate::Error::Cancelled) if the
    /// flag stopped it; [`Error::Operation`](crate::Error::Operation) if the kernel can't
    /// compute it or the result fails its sanity checks.
    ///
    /// ```
    /// use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let a = Cuboid::new([2.0, 1.0, 1.0]).build()?;
    /// let b = Cuboid::new([1.0, 1.0, 1.0]).build()?;
    /// let rest = occt::Boolean::cut([&a]).tools([&b]).build()?;
    /// assert!(rest.shape.is_valid()?);
    /// # Ok(())
    /// # }
    /// ```
    pub fn build(self) -> Result<Modeled> {
        let options = self.options.to_kernel("boolean")?;
        let (arguments, tools): (&[Shape], &[Shape]) = match self.op {
            _ if self.arguments.is_empty() => {
                return Err(invalid("a boolean needs at least one argument"));
            }
            BooleanOp::Cut | BooleanOp::Common if self.tools.is_empty() => {
                return Err(invalid("this boolean needs at least one tool"));
            }
            BooleanOp::Fuse if self.tools.is_empty() => {
                if self.arguments.len() < 2 {
                    return Err(invalid("a fuse needs at least two shapes"));
                }
                self.arguments.split_at(1)
            }
            _ => (&self.arguments, &self.tools),
        };
        Active::boolean(self.op, &raw_list(arguments), &raw_list(tools), &options)
            .map(Modeled::from_raw)
    }
}

/// Divides shapes by tools, keeping every piece, with options and
/// [`History`](crate::History).
///
/// [`Shape::split_by`] is the short form for one shape without options. Tools may be solids
/// or faces and only divide: none of a tool's own material joins the result.
///
/// # Examples
/// ```
/// use occt::prelude::*;
/// use occt::Split;
///
/// # fn main() -> occt::Result<()> {
/// let block = Cuboid::new([2.0, 1.0, 1.0]).build()?;
/// let blade = Polygon::new([
///     [1.0, -1.0, -1.0],
///     [1.0, 2.0, -1.0],
///     [1.0, 2.0, 2.0],
///     [1.0, -1.0, 2.0],
/// ])
/// .build()?;
/// let pieces = Split::new([&block]).tools([&blade]).build()?;
/// assert_eq!(pieces.shape.solids()?.len(), 2);
/// assert!((pieces.shape.volume()? - 2.0).abs() < 1e-9);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug)]
#[must_use = "a builder does nothing until `build` is called"]
pub struct Split {
    arguments: Vec<Shape>,
    tools: Vec<Shape>,
    options: Options,
}

impl Split {
    /// Starts splitting these shapes.
    ///
    /// ```
    /// use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let block = Cuboid::new([3.0, 3.0, 3.0]).build()?;
    /// let core = Cuboid::new([1.0, 1.0, 1.0]).at([1.0, 1.0, 1.0]).build()?;
    /// let pieces = occt::Split::new([&block]).tools([&core]).build()?;
    /// assert_eq!(pieces.shape.solids()?.len(), 2);
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(arguments: impl IntoIterator<Item: AsRef<Shape>>) -> Self {
        Self {
            arguments: owned(arguments),
            tools: Vec::new(),
            options: Options::default(),
        }
    }

    /// Adds shapes to divide by.
    ///
    /// ```
    /// use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let block = Cuboid::new([3.0, 3.0, 3.0]).build()?;
    /// let core = Cuboid::new([1.0, 1.0, 1.0]).at([1.0, 1.0, 1.0]).build()?;
    /// let other = Cuboid::new([1.0, 1.0, 1.0]).at([0.0, 0.0, 0.0]).build()?;
    /// let pieces = occt::Split::new([&block]).tools([&core]).tools([&other]).build()?;
    /// assert_eq!(pieces.shape.solids()?.len(), 3);
    /// # Ok(())
    /// # }
    /// ```
    pub fn tools(mut self, tools: impl IntoIterator<Item: AsRef<Shape>>) -> Self {
        self.tools.extend(owned(tools));
        self
    }

    /// Treats geometry closer than `tolerance` as coincident. Must be finite and positive.
    ///
    /// ```
    /// use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let block = Cuboid::new([3.0, 3.0, 3.0]).build()?;
    /// let core = Cuboid::new([1.0, 1.0, 1.0]).at([1.0, 1.0, 1.0]).build()?;
    /// let pieces = occt::Split::new([&block]).tools([&core]).fuzzy(1e-6).build()?;
    /// assert_eq!(pieces.shape.solids()?.len(), 2);
    /// # Ok(())
    /// # }
    /// ```
    pub fn fuzzy(mut self, tolerance: f64) -> Self {
        self.options.fuzzy = Some(tolerance);
        self
    }

    /// Merges faces that lie on the same surface within each piece.
    ///
    /// ```
    /// use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let block = Cuboid::new([3.0, 3.0, 3.0]).build()?;
    /// let core = Cuboid::new([1.0, 1.0, 1.0]).at([1.0, 1.0, 1.0]).build()?;
    /// let pieces = occt::Split::new([&block]).tools([&core]).unify().build()?;
    /// assert_eq!(pieces.shape.solids()?.len(), 2);
    /// # Ok(())
    /// # }
    /// ```
    pub fn unify(mut self) -> Self {
        self.options.unify = true;
        self
    }

    /// Uses several threads where the platform has them.
    ///
    /// ```
    /// use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let block = Cuboid::new([3.0, 3.0, 3.0]).build()?;
    /// let core = Cuboid::new([1.0, 1.0, 1.0]).at([1.0, 1.0, 1.0]).build()?;
    /// let pieces = occt::Split::new([&block]).tools([&core]).parallel().build()?;
    /// assert_eq!(pieces.shape.solids()?.len(), 2);
    /// # Ok(())
    /// # }
    /// ```
    pub fn parallel(mut self) -> Self {
        self.options.parallel = true;
        self
    }

    /// Lets `flag` stop the operation from another thread.
    ///
    /// ```
    /// use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let block = Cuboid::new([3.0, 3.0, 3.0]).build()?;
    /// let core = Cuboid::new([1.0, 1.0, 1.0]).at([1.0, 1.0, 1.0]).build()?;
    /// let flag = occt::CancelFlag::new();
    /// flag.cancel();
    /// let stopped = occt::Split::new([&block]).tools([&core]).cancel_with(&flag).build();
    /// assert!(matches!(stopped, Err(Error::Cancelled)));
    /// # Ok(())
    /// # }
    /// ```
    pub fn cancel_with(mut self, flag: &CancelFlag) -> Self {
        self.options.cancel = Some(flag.clone());
        self
    }

    /// Runs the operation. The result is a compound of the pieces, and the inputs are
    /// unchanged.
    ///
    /// # Errors
    /// [`Error::InvalidInput`](crate::Error::InvalidInput) if the fuzzy value isn't finite
    /// and positive or a side is empty; [`Error::Cancelled`](crate::Error::Cancelled) if the
    /// flag stopped it; [`Error::Operation`](crate::Error::Operation) if the kernel can't
    /// compute it or the pieces fail its sanity checks.
    ///
    /// ```
    /// use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let block = Cuboid::new([3.0, 3.0, 3.0]).build()?;
    /// let core = Cuboid::new([1.0, 1.0, 1.0]).at([1.0, 1.0, 1.0]).build()?;
    /// let pieces = occt::Split::new([&block]).tools([&core]).build()?;
    /// assert!((pieces.shape.volume()? - 27.0).abs() < 1e-9);
    /// # Ok(())
    /// # }
    /// ```
    pub fn build(self) -> Result<Modeled> {
        let options = self.options.to_kernel("split")?;
        if self.arguments.is_empty() {
            return Err(invalid("a split needs at least one shape to divide"));
        }
        if self.tools.is_empty() {
            return Err(invalid("a split needs at least one tool"));
        }
        Active::split(&raw_list(&self.arguments), &raw_list(&self.tools), &options)
            .map(Modeled::from_raw)
    }
}

impl Shape {
    /// The curves where this shape meets another, as a shape of edges. It has no edges when
    /// the two don't meet.
    ///
    /// # Errors
    /// [`Error::Operation`](crate::Error::Operation) if the kernel can't compute it.
    ///
    /// # Examples
    /// ```
    /// use occt::prelude::*;
    /// use std::f64::consts::PI;
    ///
    /// # fn main() -> occt::Result<()> {
    /// let block = Cuboid::new([4.0, 4.0, 4.0]).centered().build()?;
    /// let bar = Cylinder::new(1.0, 6.0)
    ///     .along(Axis::new([0.0, 0.0, 0.0].into(), Dir3::Z))
    ///     .build()?;
    /// let rim: f64 = block
    ///     .section(&bar)?
    ///     .edges()?
    ///     .map(|e| e.length())
    ///     .sum::<occt::Result<f64>>()?;
    /// assert!((rim - 2.0 * PI).abs() < 1e-6);
    /// # Ok(())
    /// # }
    /// ```
    pub fn section(&self, other: &Shape) -> Result<Shape> {
        Active::section(self.raw(), other.raw()).map(Shape::from_raw)
    }

    /// The curves where this shape meets an unbounded plane, as a shape of edges. It has
    /// no edges when the two don't meet.
    ///
    /// # Errors
    /// [`Error::InvalidInput`](crate::Error::InvalidInput) if the plane's origin isn't
    /// finite; [`Error::Operation`](crate::Error::Operation) if the kernel can't compute it.
    ///
    /// # Examples
    /// ```
    /// use occt::prelude::*;
    /// use occt::Plane;
    ///
    /// # fn main() -> occt::Result<()> {
    /// let block = Cuboid::new([2.0, 3.0, 4.0]).build()?;
    /// let waist = block.section_plane(Plane::new([0.0, 0.0, 2.0].into(), Dir3::Z))?;
    /// let perimeter: f64 = waist.edges()?.map(|e| e.length()).sum::<occt::Result<f64>>()?;
    /// assert!((perimeter - 10.0).abs() < 1e-6);
    /// # Ok(())
    /// # }
    /// ```
    pub fn section_plane(&self, plane: Plane) -> Result<Shape> {
        point("section plane origin", plane.origin)?;
        Active::section_plane(self.raw(), plane).map(Shape::from_raw)
    }

    /// Divides this shape by tools, keeping every piece. Use [`Split`] for options.
    ///
    /// # Errors
    /// As for [`Split::build`].
    ///
    /// # Examples
    /// ```
    /// use occt::prelude::*;
    ///
    /// # fn main() -> occt::Result<()> {
    /// let block = Cuboid::new([3.0, 3.0, 3.0]).build()?;
    /// let core = Cuboid::new([1.0, 1.0, 1.0]).at([1.0, 1.0, 1.0]).build()?;
    /// let pieces = block.split_by([&core])?;
    /// assert_eq!(pieces.shape.solids()?.len(), 2);
    /// assert!((pieces.shape.volume()? - 27.0).abs() < 1e-9);
    /// # Ok(())
    /// # }
    /// ```
    pub fn split_by(&self, tools: impl IntoIterator<Item: AsRef<Shape>>) -> Result<Modeled> {
        Split::new([self]).tools(tools).build()
    }
}

/// Rounds edges with per-edge control of the radius, and reports [`History`](crate::History).
///
/// For one radius on many edges without history, [`Shape::fillet`] is shorter. To round
/// edges and bevel others, build one result and find the remaining edges with its history.
///
/// Edges that continue a chosen edge tangentially are rounded with it, because a blend
/// can't stop at a smooth join.
///
/// # Examples
/// ```
/// use occt::prelude::*;
/// use occt::{EdgeFillet, RadiusLaw};
///
/// # fn main() -> occt::Result<()> {
/// let block = Cuboid::new([10.0, 10.0, 10.0]).build()?;
/// let edges: Vec<_> = block.edges()?.collect();
/// let rounded = EdgeFillet::on(&block)
///     .radius(&edges[0], 1.0)
///     .linear(&edges[2], 1.0, 2.0)
///     .law(&edges[4], RadiusLaw::Points(vec![(0.0, 1.0), (0.5, 2.0), (1.0, 1.0)]))
///     .build()?;
/// assert!(rounded.shape.volume()? < block.volume()?);
///
/// // The rounding made from an edge is found through the history.
/// assert!(!rounded.history.generated(&edges[0]).is_empty());
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug)]
#[must_use = "a builder does nothing until `build` is called"]
pub struct EdgeFillet<'a> {
    shape: &'a Shape,
    edges: Vec<(Edge, RadiusLaw)>,
}

impl<'a> EdgeFillet<'a> {
    /// Starts rounding edges of `shape`.
    ///
    /// ```
    /// use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let block = Cuboid::new([10.0, 10.0, 10.0]).build()?;
    /// let edge = block.edges()?.next().expect("a box has edges");
    /// let rounded = occt::EdgeFillet::on(&block).radius(&edge, 1.0).build()?;
    /// assert!(rounded.shape.volume()? < block.volume()?);
    /// # Ok(())
    /// # }
    /// ```
    pub fn on(shape: &'a Shape) -> Self {
        Self {
            shape,
            edges: Vec::new(),
        }
    }

    /// Rounds an edge with a law.
    ///
    /// ```
    /// use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let block = Cuboid::new([10.0, 10.0, 10.0]).build()?;
    /// let edge = block.edges()?.next().expect("a box has edges");
    /// let law = occt::RadiusLaw::Linear { start: 1.0, end: 2.0 };
    /// let rounded = occt::EdgeFillet::on(&block).law(&edge, law).build()?;
    /// assert!(rounded.shape.volume()? < block.volume()?);
    /// # Ok(())
    /// # }
    /// ```
    pub fn law(mut self, edge: impl Borrow<Edge>, law: RadiusLaw) -> Self {
        self.edges.push((edge.borrow().clone(), law));
        self
    }

    /// Rounds an edge with one radius.
    ///
    /// ```
    /// use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let block = Cuboid::new([10.0, 10.0, 10.0]).build()?;
    /// let edge = block.edges()?.next().expect("a box has edges");
    /// let rounded = occt::EdgeFillet::on(&block).radius(&edge, 1.0).build()?;
    /// assert!(rounded.shape.volume()? < block.volume()?);
    /// # Ok(())
    /// # }
    /// ```
    pub fn radius(self, edge: impl Borrow<Edge>, radius: f64) -> Self {
        self.law(edge, RadiusLaw::Constant(radius))
    }

    /// Rounds an edge with a radius changing evenly from `start` at its first vertex, as
    /// [`Shape::vertices`] lists them, to `end` at its last. Very near the two ends the
    /// radius eases in, off by at most about a fifth of a percent of `end - start`.
    ///
    /// ```
    /// use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let block = Cuboid::new([10.0, 10.0, 10.0]).build()?;
    /// let edge = block.edges()?.next().expect("a box has edges");
    /// let rounded = occt::EdgeFillet::on(&block).linear(&edge, 1.0, 2.0).build()?;
    /// assert!(rounded.shape.volume()? < block.volume()?);
    /// # Ok(())
    /// # }
    /// ```
    pub fn linear(self, edge: impl Borrow<Edge>, start: f64, end: f64) -> Self {
        self.law(edge, RadiusLaw::Linear { start, end })
    }

    /// Rounds an edge with a radius passing smoothly through `(fraction, radius)` points,
    /// fractions running from 0 at the edge's first vertex to 1 at its last.
    ///
    /// Fractions must increase. Past the first and last point the radius holds steady.
    ///
    /// ```
    /// use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let block = Cuboid::new([10.0, 10.0, 10.0]).build()?;
    /// let edge = block.edges()?.next().expect("a box has edges");
    /// let points = [(0.0, 1.0), (0.5, 2.0), (1.0, 1.0)];
    /// let rounded = occt::EdgeFillet::on(&block).through(&edge, points).build()?;
    /// assert!(rounded.shape.volume()? < block.volume()?);
    /// # Ok(())
    /// # }
    /// ```
    pub fn through(
        self,
        edge: impl Borrow<Edge>,
        points: impl IntoIterator<Item = (f64, f64)>,
    ) -> Self {
        self.law(edge, RadiusLaw::Points(points.into_iter().collect()))
    }

    /// Builds the rounded shape. The original is unchanged.
    ///
    /// # Errors
    /// [`Error::InvalidInput`](crate::Error::InvalidInput) if no edges were added, an edge
    /// isn't part of the shape, an edge has two different laws, a radius isn't finite and
    /// positive, or a law's fractions don't increase within 0 to 1;
    /// [`Error::Operation`](crate::Error::Operation) if the kernel can't build it, as when a
    /// radius is too large for the shape.
    ///
    /// ```
    /// use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let block = Cuboid::new([10.0, 10.0, 10.0]).build()?;
    /// let edge = block.edges()?.next().expect("a box has edges");
    /// let rounded = occt::EdgeFillet::on(&block).radius(&edge, 1.0).build()?;
    /// assert_eq!(rounded.history.generated(&edge).len(), 1);
    /// # Ok(())
    /// # }
    /// ```
    pub fn build(self) -> Result<Modeled> {
        for (_, law) in &self.edges {
            check_law(law)?;
        }
        let edges = distinct(self.shape, self.edges, "fillet")?;
        let raw: Vec<_> = edges
            .iter()
            .map(|(e, law)| (e.raw(), law.clone()))
            .collect();
        Active::fillet_edges(self.shape.raw(), &raw).map(Modeled::from_raw)
    }
}

fn check_law(law: &RadiusLaw) -> Result<()> {
    match law {
        RadiusLaw::Constant(radius) => {
            positive("fillet radius", *radius)?;
        }
        RadiusLaw::Linear { start, end } => {
            positive("fillet radius", *start)?;
            positive("fillet radius", *end)?;
        }
        RadiusLaw::Points(points) => {
            if points.len() < 2 {
                return Err(invalid("a radius law needs at least two points"));
            }
            let mut last = f64::NEG_INFINITY;
            for &(fraction, radius) in points {
                positive("fillet radius", radius)?;
                if !(0.0..=1.0).contains(&fraction) {
                    return Err(invalid(format!(
                        "a radius law's fractions must be within 0 to 1, got {fraction}"
                    )));
                }
                if fraction <= last {
                    return Err(invalid("a radius law's fractions must increase"));
                }
                last = fraction;
            }
        }
    }
    Ok(())
}

/// Checks that edges belong to the shape, each with one description, and removes repeats.
fn distinct<T: PartialEq>(
    shape: &Shape,
    edges: Vec<(Edge, T)>,
    op: &str,
) -> Result<Vec<(Edge, T)>> {
    if edges.is_empty() {
        return Err(invalid(format!("{op} needs at least one edge")));
    }
    let own: HashSet<Edge> = shape.edges()?.collect();
    let mut seen: HashMap<Edge, usize> = HashMap::new();
    let mut out: Vec<(Edge, T)> = Vec::with_capacity(edges.len());
    for (edge, size) in edges {
        if !own.contains(&edge) {
            return Err(invalid(format!(
                "{op} was given an edge of a different shape"
            )));
        }
        if let Some(&first) = seen.get(&edge) {
            if out[first].1 != size {
                return Err(invalid(format!(
                    "{op} was given two different sizes for one edge"
                )));
            }
            continue;
        }
        seen.insert(edge.clone(), out.len());
        out.push((edge, size));
    }
    Ok(out)
}

/// Bevels edges with per-edge sizes, and reports [`History`](crate::History).
///
/// For one distance on many edges without history, [`Shape::chamfer`] is shorter. To round
/// edges and bevel others, build one result and find the remaining edges with its history.
///
/// # Examples
/// ```
/// use occt::prelude::*;
/// use occt::EdgeChamfer;
///
/// # fn main() -> occt::Result<()> {
/// let block = Cuboid::new([10.0, 10.0, 10.0]).build()?;
/// let edge = block.edges()?.next().expect("a box has edges");
/// let face = block
///     .faces()?
///     .find(|f| f.edges().is_ok_and(|mut e| e.any(|e| e == edge)))
///     .expect("an edge has faces");
///
/// let cut = EdgeChamfer::on(&block)
///     .two_distances(&edge, &face, 1.0, 3.0)
///     .build()?;
/// assert!((cut.shape.volume()? - (1000.0 - 0.5 * 1.0 * 3.0 * 10.0)).abs() < 1e-6);
/// assert!(!cut.history.generated(&edge).is_empty());
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug)]
#[must_use = "a builder does nothing until `build` is called"]
pub struct EdgeChamfer<'a> {
    shape: &'a Shape,
    edges: Vec<(Edge, (ChamferSize, Option<Face>))>,
}

impl<'a> EdgeChamfer<'a> {
    /// Starts bevelling edges of `shape`.
    ///
    /// ```
    /// use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let block = Cuboid::new([10.0, 10.0, 10.0]).build()?;
    /// let edge = block.edges()?.next().expect("a box has edges");
    /// let cut = occt::EdgeChamfer::on(&block).equal(&edge, 1.0).build()?;
    /// assert!((cut.shape.volume()? - 995.0).abs() < 1e-6);
    /// # Ok(())
    /// # }
    /// ```
    pub fn on(shape: &'a Shape) -> Self {
        Self {
            shape,
            edges: Vec::new(),
        }
    }

    /// Bevels an edge by the same distance on both faces.
    ///
    /// ```
    /// use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let block = Cuboid::new([10.0, 10.0, 10.0]).build()?;
    /// let edge = block.edges()?.next().expect("a box has edges");
    /// let cut = occt::EdgeChamfer::on(&block).equal(&edge, 2.0).build()?;
    /// assert!((cut.shape.volume()? - 980.0).abs() < 1e-6);
    /// # Ok(())
    /// # }
    /// ```
    pub fn equal(mut self, edge: impl Borrow<Edge>, distance: f64) -> Self {
        self.edges
            .push((edge.borrow().clone(), (ChamferSize::Equal(distance), None)));
        self
    }

    /// Bevels an edge by `on_face` along `face`, which must be next to the edge, and by
    /// `on_other` along the face opposite.
    ///
    /// ```
    /// use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let block = Cuboid::new([10.0, 10.0, 10.0]).build()?;
    /// let edge = block.edges()?.next().expect("a box has edges");
    /// let face = block
    ///     .faces()?
    ///     .find(|f| f.edges().is_ok_and(|mut e| e.any(|e| e == edge)))
    ///     .expect("an edge has faces");
    /// let cut = occt::EdgeChamfer::on(&block)
    ///     .two_distances(&edge, &face, 1.0, 2.0)
    ///     .build()?;
    /// assert!((cut.shape.volume()? - 990.0).abs() < 1e-6);
    /// # Ok(())
    /// # }
    /// ```
    pub fn two_distances(
        mut self,
        edge: impl Borrow<Edge>,
        face: impl Borrow<Face>,
        on_face: f64,
        on_other: f64,
    ) -> Self {
        self.edges.push((
            edge.borrow().clone(),
            (
                ChamferSize::TwoDistances(on_face, on_other),
                Some(face.borrow().clone()),
            ),
        ));
        self
    }

    /// Bevels an edge by `distance` along `face`, which must be next to the edge, with the
    /// bevel rising from `face` at `angle`.
    ///
    /// ```
    /// use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let block = Cuboid::new([10.0, 10.0, 10.0]).build()?;
    /// let edge = block.edges()?.next().expect("a box has edges");
    /// let face = block
    ///     .faces()?
    ///     .find(|f| f.edges().is_ok_and(|mut e| e.any(|e| e == edge)))
    ///     .expect("an edge has faces");
    /// let cut = occt::EdgeChamfer::on(&block)
    ///     .distance_angle(&edge, &face, 1.0, Angle::degrees(45.0))
    ///     .build()?;
    /// assert!((cut.shape.volume()? - 995.0).abs() < 1e-6);
    /// # Ok(())
    /// # }
    /// ```
    pub fn distance_angle(
        mut self,
        edge: impl Borrow<Edge>,
        face: impl Borrow<Face>,
        distance: f64,
        angle: Angle,
    ) -> Self {
        self.edges.push((
            edge.borrow().clone(),
            (
                ChamferSize::DistanceAngle(distance, angle),
                Some(face.borrow().clone()),
            ),
        ));
        self
    }

    /// Builds the bevelled shape. The original is unchanged.
    ///
    /// # Errors
    /// [`Error::InvalidInput`](crate::Error::InvalidInput) if no edges were added, an edge
    /// or face isn't part of the shape, a face isn't next to its edge, an edge has two
    /// different sizes, a distance isn't finite and positive, or an angle isn't strictly
    /// between 0 and 180 degrees; [`Error::Operation`](crate::Error::Operation) if the
    /// kernel can't build it, as when a distance is too large for the shape.
    ///
    /// ```
    /// use occt::prelude::*;
    /// # fn main() -> occt::Result<()> {
    /// let block = Cuboid::new([10.0, 10.0, 10.0]).build()?;
    /// let edge = block.edges()?.next().expect("a box has edges");
    /// let cut = occt::EdgeChamfer::on(&block).equal(&edge, 1.0).build()?;
    /// assert_eq!(cut.history.generated(&edge).len(), 1);
    /// # Ok(())
    /// # }
    /// ```
    pub fn build(self) -> Result<Modeled> {
        for (edge, (size, face)) in &self.edges {
            check_size(*size)?;
            if let Some(face) = face {
                check_reference(self.shape, edge, face)?;
            }
        }
        let edges = distinct(self.shape, self.edges, "chamfer")?;
        let raw: Vec<_> = edges
            .iter()
            .map(|(edge, (size, face))| (edge.raw(), *size, face.as_ref().map(|f| f.raw())))
            .collect();
        Active::chamfer_edges(self.shape.raw(), &raw).map(Modeled::from_raw)
    }
}

fn check_size(size: ChamferSize) -> Result<()> {
    match size {
        ChamferSize::Equal(distance) => {
            positive("chamfer distance", distance)?;
        }
        ChamferSize::TwoDistances(a, b) => {
            positive("chamfer distance", a)?;
            positive("chamfer distance", b)?;
        }
        ChamferSize::DistanceAngle(distance, angle) => {
            positive("chamfer distance", distance)?;
            let radians = angle.as_radians();
            if !(radians > 0.0 && radians < std::f64::consts::PI) {
                return Err(invalid(format!(
                    "a chamfer angle must be between 0 and 180 degrees, got {radians} radians"
                )));
            }
        }
    }
    Ok(())
}

fn check_reference(shape: &Shape, edge: &Edge, face: &Face) -> Result<()> {
    if !shape.faces()?.any(|own| own == *face) {
        return Err(invalid("chamfer was given a face of a different shape"));
    }
    if !face.edges()?.any(|own| own == *edge) {
        return Err(invalid(
            "a chamfer's reference face must be next to its edge",
        ));
    }
    Ok(())
}
