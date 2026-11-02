//! Assembly documents, saving shapes in the kernel's own format, and IGES.
//!
//! Like [`step`](crate::Shape::from_step), everything here works on bytes or on
//! [`Read`] and [`Write`], so it needs no file system of yours. Geometry is always in
//! millimetres, whatever unit a file was written in.

use std::fmt;
use std::io::{Read, Write};
use std::iter::FusedIterator;

use occt_kernel::{self as contract, CancelFlag, Exchange, Result, StepWriteOptions, Transform};

pub use occt_kernel::StepSchema;

use crate::kernel::{Active, RawShape};
use crate::shape::Shape;
use crate::validate::invalid;

/// The deepest assembly nesting accepted.
const MAX_DEPTH: usize = 256;

/// How far a placement may be from rigid, as the size of the error in its rotation. Double
/// arithmetic through a few nested placements stays well inside this.
const RIGID_TOLERANCE: f64 = 1e-9;

// --- colour ---------------------------------------------------------------------------

/// A display colour with red, green and blue in `0.0..=1.0`, in the sRGB values that
/// colour pickers and files use.
///
/// Values are checked where the colour is used, so a bad one is an
/// [`Error::InvalidInput`](crate::Error::InvalidInput) and not a panic.
///
/// # Examples
/// ```
/// use occt::Color;
///
/// assert_eq!(Color::rgb(1.0, 0.0, 0.0), Color::RED);
/// assert_eq!(Color::from([0.0, 0.0, 1.0]), Color::BLUE);
/// assert_eq!(Color::from_rgb8(255, 255, 255), Color::WHITE);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Color {
    /// Red.
    pub r: f32,
    /// Green.
    pub g: f32,
    /// Blue.
    pub b: f32,
}

impl Color {
    /// Black.
    pub const BLACK: Self = Self::rgb(0.0, 0.0, 0.0);
    /// White.
    pub const WHITE: Self = Self::rgb(1.0, 1.0, 1.0);
    /// Pure red.
    pub const RED: Self = Self::rgb(1.0, 0.0, 0.0);
    /// Pure green.
    pub const GREEN: Self = Self::rgb(0.0, 1.0, 0.0);
    /// Pure blue.
    pub const BLUE: Self = Self::rgb(0.0, 0.0, 1.0);

    /// A colour from its channels in `0.0..=1.0`.
    #[must_use]
    pub const fn rgb(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b }
    }

    /// A colour from 8-bit channels.
    #[must_use]
    pub fn from_rgb8(r: u8, g: u8, b: u8) -> Self {
        let channel = |v: u8| f32::from(v) / 255.0;
        Self::rgb(channel(r), channel(g), channel(b))
    }

    fn is_valid(self) -> bool {
        [self.r, self.g, self.b]
            .iter()
            .all(|c| (0.0..=1.0).contains(c))
    }
}

impl From<[f32; 3]> for Color {
    fn from([r, g, b]: [f32; 3]) -> Self {
        Self::rgb(r, g, b)
    }
}

impl From<(f32, f32, f32)> for Color {
    fn from((r, g, b): (f32, f32, f32)) -> Self {
        Self::rgb(r, g, b)
    }
}

// --- nodes ----------------------------------------------------------------------------

/// What a [`Node`] holds.
#[derive(Clone, Debug)]
pub enum NodeContent {
    /// A part. Nodes that use one part hold the same [`Shape`], so `==` tells you.
    Part(Shape),
    /// An assembly of child nodes.
    Assembly(Vec<Node>),
}

/// A node of a [`Document`]'s product tree: a named, coloured, placed part or assembly.
///
/// A part's shape sits at its own origin and carries no placement, so that all instances
/// of a part can share it. To see where an instance really is, use
/// [`Document::parts`], which accumulates the placements on the way down.
#[derive(Clone, Debug)]
pub struct Node {
    name: Option<String>,
    color: Option<Color>,
    placement: Transform,
    content: NodeContent,
    /// A builder method called with no node to apply to, reported when writing.
    misuse: Option<&'static str>,
}

impl Node {
    fn new(name: String, content: NodeContent) -> Self {
        Self {
            name: Some(name),
            color: None,
            placement: Transform::IDENTITY,
            content,
            misuse: None,
        }
    }

    /// The product or instance name, if there is one.
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// The display colour, if one was given.
    #[must_use]
    pub fn color(&self) -> Option<Color> {
        self.color
    }

    /// Where this node is relative to its parent.
    #[must_use]
    pub fn placement(&self) -> Transform {
        self.placement
    }

    /// What this node holds.
    #[must_use]
    pub fn content(&self) -> &NodeContent {
        &self.content
    }

    /// The part's shape, or `None` for an assembly.
    #[must_use]
    pub fn shape(&self) -> Option<&Shape> {
        match &self.content {
            NodeContent::Part(shape) => Some(shape),
            NodeContent::Assembly(_) => None,
        }
    }

    /// The child nodes. Empty for a part.
    #[must_use]
    pub fn children(&self) -> &[Node] {
        match &self.content {
            NodeContent::Part(_) => &[],
            NodeContent::Assembly(children) => children,
        }
    }

    fn check(&self, depth: usize) -> Result<()> {
        if depth > MAX_DEPTH {
            return Err(invalid(format!(
                "an assembly is nested deeper than {MAX_DEPTH} levels"
            )));
        }
        check_misuse(self.misuse)?;
        let shown = self.name().unwrap_or("an unnamed node");
        if self.name().is_some_and(|name| name.contains('\0')) {
            return Err(invalid(format!("the name {shown:?} contains a NUL")));
        }
        if let Some(color) = self.color
            && !color.is_valid()
        {
            return Err(invalid(format!(
                "the colour of {shown} must have channels in 0..=1, got {color:?}"
            )));
        }
        check_rigid(shown, self.placement)?;
        if let NodeContent::Assembly(children) = &self.content {
            if children.is_empty() {
                return Err(invalid(format!("the assembly {shown} has no children")));
            }
            for child in children {
                child.check(depth + 1)?;
            }
        }
        Ok(())
    }

    fn from_raw(raw: contract::DocumentNode<RawShape>) -> Self {
        Self {
            name: raw.name,
            color: raw.color.map(|c| Color::rgb(c.r, c.g, c.b)),
            placement: raw.placement,
            content: match raw.content {
                contract::NodeContent::Part(shape) => NodeContent::Part(Shape::from_raw(shape)),
                contract::NodeContent::Assembly(children) => {
                    NodeContent::Assembly(children.into_iter().map(Self::from_raw).collect())
                }
            },
            misuse: None,
        }
    }

    fn to_raw(&self) -> contract::DocumentNode<RawShape> {
        contract::DocumentNode {
            name: self.name.clone(),
            color: self.color.map(|c| contract::Color {
                r: c.r,
                g: c.g,
                b: c.b,
            }),
            placement: self.placement,
            content: match &self.content {
                NodeContent::Part(shape) => contract::NodeContent::Part(shape.raw().clone()),
                NodeContent::Assembly(children) => {
                    contract::NodeContent::Assembly(children.iter().map(Self::to_raw).collect())
                }
            },
        }
    }
}

/// Requires a finite rotation and translation, with no scale, shear or mirroring.
fn check_rigid(what: &str, t: Transform) -> Result<()> {
    if !t.is_finite() {
        return Err(invalid(format!("the placement of {what} must be finite")));
    }
    let r = t.rows;
    let column = |i: usize| [r[0][i], r[1][i], r[2][i]];
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let (x, y, z) = (column(0), column(1), column(2));
    let orthonormal = [
        (dot(x, x), 1.0),
        (dot(y, y), 1.0),
        (dot(z, z), 1.0),
        (dot(x, y), 0.0),
        (dot(y, z), 0.0),
        (dot(x, z), 0.0),
    ]
    .iter()
    .all(|(got, want)| (got - want).abs() <= RIGID_TOLERANCE);
    let cross = [
        x[1] * y[2] - x[2] * y[1],
        x[2] * y[0] - x[0] * y[2],
        x[0] * y[1] - x[1] * y[0],
    ];
    if orthonormal && dot(cross, z) > 0.0 {
        Ok(())
    } else {
        Err(invalid(format!(
            "the placement of {what} must be rigid, with no scale, shear or mirroring"
        )))
    }
}

fn check_misuse(misuse: Option<&'static str>) -> Result<()> {
    match misuse {
        Some(method) => Err(invalid(format!(
            "`{method}` applies to the node added before it, and none was"
        ))),
        None => Ok(()),
    }
}

/// Adds the methods that attach to "the node added last" to a builder's node list. A
/// misuse is kept and reported where errors can be, since the builders chain by value.
#[derive(Clone, Debug, Default)]
struct NodeList {
    nodes: Vec<Node>,
    misuse: Option<&'static str>,
}

impl NodeList {
    fn last(&mut self, method: &'static str) -> Option<&mut Node> {
        let last = self.nodes.last_mut();
        if last.is_none() {
            self.misuse.get_or_insert(method);
        }
        last
    }

    fn color(&mut self, color: Color) {
        if let Some(node) = self.last("color") {
            node.color = Some(color);
        }
    }

    fn at(&mut self, placement: Transform) {
        if let Some(node) = self.last("at") {
            node.placement = placement;
        }
    }
}

/// An assembly being built, for [`Document::assembly`] or another `Assembly`.
///
/// [`part`](Assembly::part) and [`assembly`](Assembly::assembly) add a child, and
/// [`color`](Assembly::color) and [`at`](Assembly::at) apply to the child added last. To
/// colour or place the assembly itself, call them on whatever it is added to.
///
/// # Examples
/// ```
/// use occt::prelude::*;
/// use occt::{Assembly, Transform};
///
/// # fn main() -> occt::Result<()> {
/// let wheel = Cylinder::new(30.0, 10.0).build()?;
/// let axle = Assembly::new("axle")
///     .part("left", &wheel)
///     .at(Transform::translation([0.0, 0.0, -50.0].into()))
///     .part("right", &wheel)
///     .at(Transform::translation([0.0, 0.0, 50.0].into()));
/// # let _ = axle;
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug)]
#[must_use = "an assembly does nothing until it's added to a document"]
pub struct Assembly {
    name: String,
    children: NodeList,
}

impl Assembly {
    /// An assembly with no children yet.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            children: NodeList::default(),
        }
    }

    /// Adds an instance of a part. Adding one [`Shape`] several times makes them
    /// instances of one part, which a file stores once. Give them different placements: a
    /// file can't tell apart two instances at the very same place, so they'd share a name
    /// and a colour.
    pub fn part(mut self, name: impl Into<String>, shape: impl AsRef<Shape>) -> Self {
        self.children.nodes.push(Node::new(
            name.into(),
            NodeContent::Part(shape.as_ref().clone()),
        ));
        self
    }

    /// Adds a nested assembly.
    pub fn assembly(mut self, assembly: Assembly) -> Self {
        self.children.nodes.push(assembly.into_node());
        self
    }

    /// Gives the child added last a colour.
    pub fn color(mut self, color: impl Into<Color>) -> Self {
        self.children.color(color.into());
        self
    }

    /// Places the child added last relative to this assembly. The transform must be rigid.
    pub fn at(mut self, placement: Transform) -> Self {
        self.children.at(placement);
        self
    }

    fn into_node(self) -> Node {
        let mut node = Node::new(self.name, NodeContent::Assembly(self.children.nodes));
        node.misuse = self.children.misuse;
        node
    }
}

// --- documents ------------------------------------------------------------------------

/// A product tree of named, coloured, placed parts, as STEP files hold it.
///
/// Read one with [`from_step`](Document::from_step), or build one with
/// [`part`](Document::part) and [`assembly`](Document::assembly), where
/// [`color`](Document::color) and [`at`](Document::at) apply to the node added last.
///
/// # Examples
/// Two instances of one part, written to STEP and read back:
/// ```
/// use occt::prelude::*;
/// use occt::{Assembly, Color, Document, StepOptions, StepSchema, Transform};
///
/// # fn main() -> occt::Result<()> {
/// let bolt = Cuboid::new([1.0, 2.0, 3.0]).build()?;
/// let document = Document::new().assembly(
///     Assembly::new("frame")
///         .part("bolt-a", &bolt)
///         .color(Color::RED)
///         .part("bolt-b", &bolt)
///         .color(Color::RED)
///         .at(Transform::translation([100.0, 0.0, 0.0].into())),
/// );
///
/// let step = document.to_step(StepOptions::new().schema(StepSchema::Ap242))?;
/// let back = Document::from_step(&step)?;
///
/// let parts: Vec<_> = back.parts().collect();
/// assert_eq!(parts[1].path(), ["frame", "bolt-b"]);
/// assert_eq!(parts[0].shape(), parts[1].shape());
/// assert!((parts[1].placement().rows[0][3] - 100.0).abs() < 1e-9);
/// assert!(parts[1].color().is_some());
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, Default)]
#[must_use]
pub struct Document {
    roots: NodeList,
    file_units_per_metre: Option<f64>,
}

impl Document {
    /// An empty document.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a part at the top level. Adding one [`Shape`] several times makes them
    /// instances of one part, which a file stores once.
    pub fn part(mut self, name: impl Into<String>, shape: impl AsRef<Shape>) -> Self {
        self.roots.nodes.push(Node::new(
            name.into(),
            NodeContent::Part(shape.as_ref().clone()),
        ));
        self
    }

    /// Adds an assembly at the top level.
    pub fn assembly(mut self, assembly: Assembly) -> Self {
        self.roots.nodes.push(assembly.into_node());
        self
    }

    /// Gives the node added last a colour. On an assembly, it's the colour of each part in
    /// it that has none of its own, as files can't colour an assembly as such.
    pub fn color(mut self, color: impl Into<Color>) -> Self {
        self.roots.color(color.into());
        self
    }

    /// Places the node added last. The transform must be rigid.
    pub fn at(mut self, placement: Transform) -> Self {
        self.roots.at(placement);
        self
    }

    /// The top-level nodes.
    #[must_use]
    pub fn roots(&self) -> &[Node] {
        &self.roots.nodes
    }

    /// The length units per metre that the file this came from declared, such as `1000.0`
    /// for millimetres, or `None` if it didn't, or this document didn't come from a file.
    ///
    /// This is for information: geometry is in millimetres regardless.
    ///
    /// # Examples
    /// ```
    /// use occt::prelude::*;
    /// use occt::{Document, LengthUnit, StepOptions};
    ///
    /// # fn main() -> occt::Result<()> {
    /// let part = Cuboid::new([1000.0, 1000.0, 1000.0]).build()?;
    /// let document = Document::new().part("cube", &part);
    /// assert_eq!(document.file_units_per_metre(), None);
    ///
    /// let step = document.to_step(StepOptions::new().unit(LengthUnit::Metre))?;
    /// let back = Document::from_step(&step)?;
    /// assert_eq!(back.file_units_per_metre(), Some(1.0));
    /// assert!((back.parts().next().unwrap().shape().volume()? - 1e9).abs() < 1.0);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn file_units_per_metre(&self) -> Option<f64> {
        self.file_units_per_metre
    }

    /// Every part instance in the tree, depth first, with its world placement and the
    /// names on the way to it.
    ///
    /// # Examples
    /// ```
    /// use occt::prelude::*;
    /// use occt::{Assembly, Document, Transform};
    ///
    /// # fn main() -> occt::Result<()> {
    /// let block = Cuboid::new([1.0, 1.0, 1.0]).build()?;
    /// let document = Document::new()
    ///     .assembly(Assembly::new("inner").part("block", &block))
    ///     .at(Transform::translation([5.0, 0.0, 0.0].into()));
    ///
    /// let part = document.parts().next().unwrap();
    /// assert_eq!(part.path(), ["inner", "block"]);
    /// assert_eq!(part.full_name(), "inner/block");
    /// assert_eq!(part.placement().rows[0][3], 5.0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn parts(&self) -> Parts<'_> {
        let mut parts = Vec::new();
        let mut path = Vec::new();
        for node in &self.roots.nodes {
            collect(node, Transform::IDENTITY, None, &mut path, &mut parts);
        }
        Parts(parts.into_iter())
    }

    /// Parses a STEP file held in memory, with its product tree, names and colours.
    ///
    /// Instances of one part in the file hold one [`Shape`]. A face's own colour has no
    /// place in a node, so it's only kept when one colour covers the whole part.
    ///
    /// # Errors
    /// [`Error::Operation`](crate::Error::Operation) if the data is not a STEP file the
    /// kernel can use.
    ///
    /// # Examples
    /// ```
    /// assert!(occt::Document::from_step(b"not a STEP file").is_err());
    /// ```
    pub fn from_step(data: &[u8]) -> Result<Self> {
        Self::parse(data, None)
    }

    /// Like [`from_step`](Document::from_step), and stops when `cancel` is set.
    ///
    /// # Errors
    /// [`Error::Cancelled`](crate::Error::Cancelled) if `cancel` was set in time;
    /// [`Error::Operation`](crate::Error::Operation) if the data is not a STEP file the
    /// kernel can use.
    ///
    /// # Examples
    /// ```
    /// use occt::{CancelFlag, Document, Error};
    /// use occt::prelude::*;
    ///
    /// # fn main() -> occt::Result<()> {
    /// let step = Cuboid::new([1.0, 1.0, 1.0]).build()?.to_step()?;
    /// let cancel = CancelFlag::new();
    /// cancel.cancel();
    /// let result = Document::from_step_cancellable(&step, &cancel);
    /// assert!(matches!(result, Err(Error::Cancelled)));
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_step_cancellable(data: &[u8], cancel: &CancelFlag) -> Result<Self> {
        Self::parse(data, Some(cancel))
    }

    fn parse(data: &[u8], cancel: Option<&CancelFlag>) -> Result<Self> {
        let raw = Active::read_step_document(data, cancel)?;
        Ok(Self {
            roots: NodeList {
                nodes: raw.roots.into_iter().map(Node::from_raw).collect(),
                misuse: None,
            },
            file_units_per_metre: raw.file_units_per_metre,
        })
    }

    /// Serialises this document as the bytes of a STEP file.
    ///
    /// # Errors
    /// [`Error::InvalidInput`](crate::Error::InvalidInput) if the document is empty, or a
    /// node has a bad colour, a placement that's not rigid or a name with a NUL, or an
    /// assembly has no children;
    /// [`Error::Cancelled`](crate::Error::Cancelled) if the options' flag was set in time;
    /// [`Error::Operation`](crate::Error::Operation) if the kernel cannot write it.
    ///
    /// # Examples
    /// ```
    /// use occt::prelude::*;
    /// use occt::{Document, StepOptions};
    ///
    /// # fn main() -> occt::Result<()> {
    /// let part = Cuboid::new([1.0, 1.0, 1.0]).build()?;
    /// let step = Document::new().part("cube", &part).to_step(StepOptions::new())?;
    /// assert!(step.starts_with(b"ISO-10303-21"));
    /// assert!(Document::new().to_step(StepOptions::new()).is_err());
    /// # Ok(())
    /// # }
    /// ```
    pub fn to_step(&self, options: StepOptions) -> Result<Vec<u8>> {
        check_misuse(self.roots.misuse)?;
        if self.roots.nodes.is_empty() {
            return Err(invalid("a document needs at least one part"));
        }
        for node in &self.roots.nodes {
            node.check(1)?;
        }
        let raw = contract::Document {
            roots: self.roots.nodes.iter().map(Node::to_raw).collect(),
            file_units_per_metre: options.unit.map(LengthUnit::per_metre),
        };
        Active::write_step_document(&raw, &options.write)
    }

    /// Writes this document to a stream as a STEP file.
    ///
    /// # Errors
    /// As [`to_step`](Document::to_step), and
    /// [`Error::Io`](crate::Error::Io) if writing fails.
    ///
    /// # Examples
    /// ```no_run
    /// use occt::prelude::*;
    /// use occt::{Document, StepOptions};
    ///
    /// # fn main() -> occt::Result<()> {
    /// let part = Cuboid::new([1.0, 1.0, 1.0]).build()?;
    /// let file = std::fs::File::create("cube.step")?;
    /// Document::new().part("cube", &part).write_step(file, StepOptions::new())?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn write_step(&self, mut writer: impl Write, options: StepOptions) -> Result<()> {
        writer.write_all(&self.to_step(options)?)?;
        writer.flush()?;
        Ok(())
    }
}

/// Reads a STEP file from a stream, to its end, with its product tree, names and colours.
///
/// Files are the caller's concern: pass a [`std::fs::File`] to read from disk.
///
/// # Errors
/// [`Error::Io`](crate::Error::Io) if reading fails;
/// [`Error::Operation`](crate::Error::Operation) if the data is not a STEP file the kernel
/// can use.
///
/// # Examples
/// ```
/// use occt::prelude::*;
/// use occt::{Document, StepOptions};
///
/// # fn main() -> occt::Result<()> {
/// let part = Cuboid::new([1.0, 1.0, 1.0]).build()?;
/// let step = Document::new().part("cube", &part).to_step(StepOptions::new())?;
/// let back = occt::read_step_document(step.as_slice())?;
/// assert_eq!(back.roots()[0].name(), Some("cube"));
/// # Ok(())
/// # }
/// ```
pub fn read_step_document(mut reader: impl Read) -> Result<Document> {
    let mut data = Vec::new();
    reader.read_to_end(&mut data)?;
    Document::from_step(&data)
}

fn collect<'a>(
    node: &'a Node,
    parent: Transform,
    inherited: Option<Color>,
    path: &mut Vec<&'a str>,
    out: &mut Vec<PartInstance<'a>>,
) {
    let placement = parent.then_after(node.placement);
    let color = node.color.or(inherited);
    if let Some(name) = node.name() {
        path.push(name);
    }
    match &node.content {
        NodeContent::Part(shape) => out.push(PartInstance {
            shape,
            placement,
            color,
            path: path.clone(),
        }),
        NodeContent::Assembly(children) => {
            for child in children {
                collect(child, placement, color, path, out);
            }
        }
    }
    if node.name().is_some() {
        path.pop();
    }
}

/// A part where it finally sits in a [`Document`], as [`Document::parts`] yields it.
#[derive(Clone, Debug)]
pub struct PartInstance<'a> {
    shape: &'a Shape,
    placement: Transform,
    color: Option<Color>,
    path: Vec<&'a str>,
}

impl<'a> PartInstance<'a> {
    /// The part's shape, at its own origin. All instances of a part hold the same one.
    #[must_use]
    pub fn shape(&self) -> &'a Shape {
        self.shape
    }

    /// The accumulated placement of this instance in the world.
    #[must_use]
    pub fn placement(&self) -> Transform {
        self.placement
    }

    /// The instance's colour: its own, or that of its nearest coloured ancestor.
    #[must_use]
    pub fn color(&self) -> Option<Color> {
        self.color
    }

    /// The names from the root to this instance, without nodes that have no name.
    #[must_use]
    pub fn path(&self) -> &[&'a str] {
        &self.path
    }

    /// The names from the root to this instance, joined by `/`.
    #[must_use]
    pub fn full_name(&self) -> String {
        self.path.join("/")
    }
}

/// The iterator of [`Document::parts`].
#[derive(Clone, Debug)]
#[must_use = "iterators do nothing until consumed"]
pub struct Parts<'a>(std::vec::IntoIter<PartInstance<'a>>);

impl<'a> Iterator for Parts<'a> {
    type Item = PartInstance<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl DoubleEndedIterator for Parts<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.0.next_back()
    }
}

impl ExactSizeIterator for Parts<'_> {}
impl FusedIterator for Parts<'_> {}

/// A length unit a STEP file can be written in. Geometry in memory is in millimetres, so
/// this only changes the numbers in the file, not what they mean.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum LengthUnit {
    /// Micrometres.
    Micrometre,
    /// Millimetres.
    #[default]
    Millimetre,
    /// Centimetres.
    Centimetre,
    /// Metres.
    Metre,
    /// Kilometres.
    Kilometre,
    /// Microinches.
    Microinch,
    /// Thousandths of an inch.
    Mil,
    /// Inches.
    Inch,
    /// Feet.
    Foot,
    /// Miles.
    Mile,
}

impl LengthUnit {
    /// How many of this unit fit a metre.
    ///
    /// # Examples
    /// ```
    /// assert_eq!(occt::LengthUnit::Millimetre.per_metre(), 1000.0);
    /// ```
    #[must_use]
    pub fn per_metre(self) -> f64 {
        match self {
            Self::Micrometre => 1e6,
            Self::Millimetre => 1e3,
            Self::Centimetre => 100.0,
            Self::Metre => 1.0,
            Self::Kilometre => 1e-3,
            Self::Microinch => 1000.0 / 0.000_025_4,
            Self::Mil => 1000.0 / 0.0254,
            Self::Inch => 1000.0 / 25.4,
            Self::Foot => 1000.0 / 304.8,
            Self::Mile => 1000.0 / 1_609_344.0,
        }
    }
}

/// How to write a STEP document: the schema, the unit and a way to stop.
///
/// # Examples
/// ```
/// use occt::{CancelFlag, LengthUnit, StepOptions, StepSchema};
///
/// let options = StepOptions::new()
///     .schema(StepSchema::Ap242)
///     .unit(LengthUnit::Inch)
///     .cancel(&CancelFlag::new());
/// # let _ = options;
/// ```
#[derive(Clone, Debug, Default)]
#[must_use = "options do nothing until passed to `to_step`"]
pub struct StepOptions {
    write: StepWriteOptions,
    unit: Option<LengthUnit>,
}

impl StepOptions {
    /// AP214, in millimetres.
    pub fn new() -> Self {
        Self::default()
    }

    /// The schema to write. AP214 is the default, and the most widely read. AP203 predates
    /// colours, so not every reader will find them in such a file.
    pub fn schema(mut self, schema: StepSchema) -> Self {
        self.write.schema = schema;
        self
    }

    /// The unit to write lengths in. Millimetres by default.
    pub fn unit(mut self, unit: LengthUnit) -> Self {
        self.unit = Some(unit);
        self
    }

    /// Stops the write when `flag` is set, with
    /// [`Error::Cancelled`](crate::Error::Cancelled).
    pub fn cancel(mut self, flag: &CancelFlag) -> Self {
        self.write.cancel = Some(flag.clone());
        self
    }
}

impl fmt::Display for LengthUnit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Micrometre => "µm",
            Self::Millimetre => "mm",
            Self::Centimetre => "cm",
            Self::Metre => "m",
            Self::Kilometre => "km",
            Self::Microinch => "µin",
            Self::Mil => "mil",
            Self::Inch => "in",
            Self::Foot => "ft",
            Self::Mile => "mi",
        })
    }
}

// --- single shapes --------------------------------------------------------------------

impl Shape {
    /// Serialises this shape in the kernel's own lossless format, for saving your own
    /// files: an application's document, an undo stack, a cache.
    ///
    /// Topology and geometry come back as they were, down to the last vertex. Only the
    /// same kind of kernel can read the bytes, so use
    /// [`to_step`](Shape::to_step) for files that other programs will open. Display meshes
    /// aren't saved.
    ///
    /// # Errors
    /// [`Error::Operation`](crate::Error::Operation) if the kernel cannot serialise the
    /// shape.
    ///
    /// # Examples
    /// ```
    /// use occt::prelude::*;
    ///
    /// # fn main() -> occt::Result<()> {
    /// let part = Cuboid::new([1.5, 2.5, 3.5]).build()?;
    /// let saved = part.to_native()?;
    /// let back = Shape::from_native(&saved)?;
    /// assert_eq!(back.faces()?.len(), 6);
    /// assert!((back.volume()? - part.volume()?).abs() < 1e-12);
    /// # Ok(())
    /// # }
    /// ```
    pub fn to_native(&self) -> Result<Vec<u8>> {
        Active::write_native(self.raw())
    }

    /// Reads bytes made by [`to_native`](Shape::to_native), for loading your own files.
    ///
    /// # Errors
    /// [`Error::Operation`](crate::Error::Operation) if the data isn't a saved shape.
    ///
    /// # Examples
    /// ```
    /// assert!(occt::Shape::from_native(b"not a saved shape").is_err());
    /// ```
    pub fn from_native(data: &[u8]) -> Result<Shape> {
        Active::read_native(data).map(Shape::from_raw)
    }

    /// Reads an IGES file held in memory, in millimetres.
    ///
    /// IGES usually stores a solid as loose faces, so faces are sewn together, and whatever
    /// that closes comes back as a [`Solid`](crate::Solid). Faces that don't close stay as
    /// a shell or a face. Several results come back in a compound.
    ///
    /// On native targets this writes a short-lived file to the system's temporary
    /// directory.
    ///
    /// # Errors
    /// [`Error::Operation`](crate::Error::Operation) if the data is not an IGES file the
    /// kernel can use.
    ///
    /// # Examples
    /// ```
    /// use occt::prelude::*;
    ///
    /// # fn main() -> occt::Result<()> {
    /// let part = Cuboid::new([2.0, 3.0, 4.0]).build()?;
    /// let back = Shape::from_iges(&part.to_iges()?)?;
    /// assert_eq!(back.kind()?, ShapeKind::Solid);
    /// assert!((back.volume()? - 24.0).abs() < 1e-4);
    /// assert!(Shape::from_iges(b"not an IGES file").is_err());
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_iges(data: &[u8]) -> Result<Shape> {
        Active::read_iges(data).map(Shape::from_raw)
    }

    /// Reads an IGES file from a stream, to its end. See [`from_iges`](Shape::from_iges).
    ///
    /// # Errors
    /// [`Error::Io`](crate::Error::Io) if reading fails;
    /// [`Error::Operation`](crate::Error::Operation) if the data is not an IGES file the
    /// kernel can use.
    ///
    /// # Examples
    /// ```no_run
    /// # fn main() -> occt::Result<()> {
    /// let part = occt::Shape::read_iges(std::fs::File::open("part.igs")?)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn read_iges(mut reader: impl Read) -> Result<Shape> {
        let mut data = Vec::new();
        reader.read_to_end(&mut data)?;
        Self::from_iges(&data)
    }

    /// Serialises this shape as the bytes of an IGES file, in millimetres.
    ///
    /// # Errors
    /// [`Error::Operation`](crate::Error::Operation) if the kernel cannot write the shape.
    ///
    /// # Examples
    /// ```
    /// use occt::prelude::*;
    ///
    /// # fn main() -> occt::Result<()> {
    /// let iges = Cuboid::new([1.0, 1.0, 1.0]).build()?.to_iges()?;
    /// assert!(!iges.is_empty());
    /// # Ok(())
    /// # }
    /// ```
    pub fn to_iges(&self) -> Result<Vec<u8>> {
        Active::write_iges(&[self.raw()])
    }
}

/// Serialises shapes together as the bytes of one IGES file, in millimetres.
///
/// # Errors
/// [`Error::InvalidInput`](crate::Error::InvalidInput) if there are no shapes;
/// [`Error::Operation`](crate::Error::Operation) if the kernel cannot write them.
///
/// # Examples
/// ```
/// use occt::prelude::*;
///
/// # fn main() -> occt::Result<()> {
/// let a = Cuboid::new([1.0, 1.0, 1.0]).build()?;
/// let b = Sphere::new(1.0).at([5.0, 0.0, 0.0]).build()?;
/// let back = Shape::from_iges(&occt::write_iges([&a, &b])?)?;
/// assert_eq!(back.solids()?.len(), 2);
/// assert!(occt::write_iges(Vec::<Shape>::new()).is_err());
/// # Ok(())
/// # }
/// ```
pub fn write_iges<I>(shapes: I) -> Result<Vec<u8>>
where
    I: IntoIterator,
    I::Item: AsRef<Shape>,
{
    let shapes: Vec<I::Item> = shapes.into_iter().collect();
    if shapes.is_empty() {
        return Err(invalid("writing IGES needs at least one shape"));
    }
    let raw: Vec<&RawShape> = shapes.iter().map(|shape| shape.as_ref().raw()).collect();
    Active::write_iges(&raw)
}
