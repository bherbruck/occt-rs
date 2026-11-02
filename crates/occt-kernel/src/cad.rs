//! The CAD-level parts of the contract: construction, inspection, features, history
//! and exchange. Preconditions match the crate docs: the API layer validates plain
//! arguments, and kernels report the rest as errors.

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

use occt_geom::{Curve, CurvePoint, EdgeGeometry, FaceGeometry, ParamRange, Surface, SurfacePoint};
use occt_math::{Angle, Axis, Dir3, Frame, Plane, Point3, Transform, Vec3};
use occt_mesh::{MeshOptions, ShapeTessellation};

use crate::{Result, ShapeKind, Topology};

/// Asks a running operation to stop. Clones share one flag.
///
/// Cancellation is cooperative: the kernel checks at its own pace, and the operation
/// then returns [`Error::Cancelled`](crate::Error::Cancelled). It works when the flag is
/// set from another thread, so it has no effect on single-threaded WebAssembly.
#[derive(Clone, Debug, Default)]
pub struct CancelFlag(Arc<AtomicU8>);

impl CancelFlag {
    /// Creates an unset flag.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Requests cancellation.
    pub fn cancel(&self) {
        self.0.store(1, Ordering::SeqCst);
    }

    /// Whether cancellation was requested.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::SeqCst) != 0
    }

    /// The flag's byte, for kernels that poll it from foreign code. It lives as long as
    /// any clone of this flag: 0 means keep going, anything else means stop.
    #[must_use]
    pub fn as_atomic(&self) -> &AtomicU8 {
        &self.0
    }
}

/// What happened to the subshapes of an operation's inputs. This is the raw material of
/// persistent naming: it lets an application follow a face or edge through a rebuild.
#[derive(Clone, Debug)]
pub struct History<S> {
    /// Input subshapes that survive in changed form, with what they became.
    pub modified: Vec<(S, Vec<S>)>,
    /// Input subshapes that gave rise to new subshapes of a different origin, such as
    /// an edge producing a fillet face.
    pub generated: Vec<(S, Vec<S>)>,
    /// Input subshapes with no counterpart in the result.
    pub deleted: Vec<S>,
}

impl<S> Default for History<S> {
    fn default() -> Self {
        Self {
            modified: Vec::new(),
            generated: Vec::new(),
            deleted: Vec::new(),
        }
    }
}

/// An operation's result with its history.
#[derive(Clone, Debug)]
pub struct Modeled<S> {
    /// The resulting shape.
    pub shape: S,
    /// How the input subshapes relate to it.
    pub history: History<S>,
}

// --- construction -----------------------------------------------------------------

/// Building topology from geometry, bottom up.
pub trait Construction: Topology {
    /// A vertex at a point.
    fn make_vertex(point: Point3) -> Result<Self::Shape>;
    /// An edge along a curve; `range` is required for an unbounded curve such as a line.
    fn make_edge(curve: &Curve, range: Option<ParamRange>) -> Result<Self::Shape>;
    /// A wire from connected edges, given in any order.
    fn make_wire(edges: &[&Self::Shape]) -> Result<Self::Shape>;
    /// A planar face from a closed outer wire and hole wires.
    fn make_planar_face(outer: &Self::Shape, holes: &[&Self::Shape]) -> Result<Self::Shape>;
    /// A face on a surface, bounded by wires lying on it, or by the parameter range
    /// when `outer` is `None`.
    fn make_surface_face(
        surface: &Surface,
        bounds: Option<(ParamRange, ParamRange)>,
        outer: Option<&Self::Shape>,
        holes: &[&Self::Shape],
    ) -> Result<Self::Shape>;
    /// A smooth patch filling a closed loop of edges, optionally passing through points.
    fn fill_boundary(edges: &[&Self::Shape], through: &[Point3]) -> Result<Self::Shape>;
    /// Joins faces or shells with coincident boundaries, within `tolerance`.
    fn sew(shapes: &[&Self::Shape], tolerance: f64) -> Result<Self::Shape>;
    /// A solid from a closed shell.
    fn make_solid(shell: &Self::Shape) -> Result<Self::Shape>;
    /// A compound holding the shapes.
    fn make_compound(shapes: &[&Self::Shape]) -> Result<Self::Shape>;
    /// A torus around `frame.z`.
    fn make_torus(frame: Frame, major_radius: f64, minor_radius: f64) -> Result<Self::Shape>;
    /// A curve through the points.
    fn interpolate_curve(points: &[Point3], closed: bool) -> Result<Curve>;
    /// A curve approximating the points within `tolerance`.
    fn fit_curve(points: &[Point3], tolerance: f64) -> Result<Curve>;
    /// A helical wire around `axis`, starting at distance `radius`.
    fn make_helix(
        axis: Axis,
        radius: f64,
        pitch: f64,
        height: f64,
        left_handed: bool,
    ) -> Result<Self::Shape>;
    /// A wire offset within its plane. Positive grows a closed wire outward.
    fn offset_wire(wire: &Self::Shape, distance: f64) -> Result<Self::Shape>;
}

// --- reading geometry --------------------------------------------------------------

/// Reading the exact geometry back out of topology.
pub trait Geometry: Topology {
    /// The curve under an edge.
    fn edge_geometry(edge: &Self::Shape) -> Result<EdgeGeometry>;
    /// Evaluates an edge at a parameter in its range.
    fn edge_point(edge: &Self::Shape, parameter: f64) -> Result<CurvePoint>;
    /// The surface under a face.
    fn face_geometry(face: &Self::Shape) -> Result<FaceGeometry>;
    /// Evaluates a face's surface at parameters.
    fn face_point(face: &Self::Shape, u: f64, v: f64) -> Result<SurfacePoint>;
    /// The closest point of the face's surface to `point`: its parameters and position.
    fn project_onto_face(face: &Self::Shape, point: Point3) -> Result<(f64, f64, Point3)>;
    /// The curve and parameter closest to `point` on an edge.
    fn project_onto_edge(edge: &Self::Shape, point: Point3) -> Result<(f64, Point3)>;
    /// Whether a wire, edge or shell is closed.
    fn is_closed(shape: &Self::Shape) -> Result<bool>;
    /// The largest geometric tolerance stored on the shape.
    fn tolerance(shape: &Self::Shape) -> Result<f64>;
}

// --- sweeps -----------------------------------------------------------------------

/// How a swept profile turns along its path.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum SweepOrientation {
    /// Follows the path while minimising twist.
    #[default]
    Corrected,
    /// Follows the path's Frenet frame.
    Frenet,
    /// Keeps the profile's orientation fixed.
    Fixed,
}

/// Options for [`AdvancedSweeps::sweep`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SweepOptions {
    /// How the profile turns.
    pub orientation: SweepOrientation,
    /// A fixed binormal direction, keeping the profile upright. Needed for threads.
    pub binormal: Option<Dir3>,
    /// Whether to close the ends into a solid where the profile is a closed wire.
    pub solid: bool,
}

/// Options for [`AdvancedSweeps::loft`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LoftOptions {
    /// Close the ends into a solid.
    pub solid: bool,
    /// Straight ruled surfaces between sections instead of smooth ones.
    pub ruled: bool,
    /// Tolerance for fitting the surfaces.
    pub tolerance: f64,
}

impl Default for LoftOptions {
    fn default() -> Self {
        Self {
            solid: true,
            ruled: false,
            tolerance: 1e-6,
        }
    }
}

/// Sweeps beyond straight extrusion and revolution, with history.
pub trait AdvancedSweeps: Topology {
    /// Extrudes with history; `generated` maps profile edges to side faces.
    fn extrude_modeled(shape: &Self::Shape, by: Vec3) -> Result<Modeled<Self::Shape>>;
    /// Revolves with history.
    fn revolve_modeled(
        shape: &Self::Shape,
        axis: Axis,
        angle: Angle,
    ) -> Result<Modeled<Self::Shape>>;
    /// Extrudes a planar face with tapered sides; positive `taper` narrows along `by`.
    fn extrude_tapered(face: &Self::Shape, by: Vec3, taper: Angle) -> Result<Self::Shape>;
    /// Sweeps a wire or face along a path wire.
    fn sweep(
        profile: &Self::Shape,
        path: &Self::Shape,
        options: &SweepOptions,
    ) -> Result<Modeled<Self::Shape>>;
    /// Lofts through two or more wires, in order. Vertices may be used as end points.
    fn loft(sections: &[&Self::Shape], options: &LoftOptions) -> Result<Self::Shape>;
}

// --- features ---------------------------------------------------------------------

/// Local edits of a solid.
pub trait Features: Topology {
    /// Hollows a solid: removes `open_faces` and gives the walls `thickness`, grown
    /// inward when negative and outward when positive.
    fn shell(
        solid: &Self::Shape,
        open_faces: &[&Self::Shape],
        thickness: f64,
    ) -> Result<Modeled<Self::Shape>>;
    /// Offsets every face. Positive grows the shape.
    fn offset(shape: &Self::Shape, distance: f64) -> Result<Self::Shape>;
    /// Turns a face or sheet into a solid by offsetting it.
    fn thicken(sheet: &Self::Shape, thickness: f64) -> Result<Self::Shape>;
    /// Tilts faces by `angle` from `pull`, hinged where they meet `neutral`. A positive
    /// angle removes material on the side `pull` points to.
    fn draft(
        solid: &Self::Shape,
        faces: &[&Self::Shape],
        pull: Dir3,
        neutral: Plane,
        angle: Angle,
    ) -> Result<Modeled<Self::Shape>>;
    /// Removes faces and heals the gap by extending their neighbours.
    fn remove_faces(solid: &Self::Shape, faces: &[&Self::Shape]) -> Result<Modeled<Self::Shape>>;
    /// Merges faces on the same surface and edges on the same curve.
    fn unify(shape: &Self::Shape) -> Result<Modeled<Self::Shape>>;
    /// Repairs tolerance and topology problems, as imported files often need.
    fn heal(shape: &Self::Shape, tolerance: f64) -> Result<Self::Shape>;
    /// Applies any affine transform as an independent copy. Non-rigid transforms may
    /// convert analytic geometry to splines.
    fn transformed(shape: &Self::Shape, transform: &Transform) -> Result<Self::Shape>;
    /// Places a shape by a rigid transform, sharing its geometry, as assembly instances
    /// do. Fails on a transform with scale or shear.
    fn placed(shape: &Self::Shape, transform: &Transform) -> Result<Self::Shape>;
    /// The same topology with the orientation flipped.
    fn reversed(shape: &Self::Shape) -> Result<Self::Shape>;
}

// --- booleans ---------------------------------------------------------------------

/// A boolean operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BooleanOp {
    /// Union.
    Fuse,
    /// `arguments` without `tools`.
    Cut,
    /// Intersection of the union of `arguments` with the union of `tools`.
    Common,
}

/// Options for [`BooleanOps::boolean`].
#[derive(Clone, Debug, Default)]
pub struct BooleanOptions {
    /// Treats geometry closer than this as coincident, which rescues near-tangent cases.
    pub fuzzy: Option<f64>,
    /// Merges faces on the same surface in the result.
    pub unify: bool,
    /// Uses several threads when the platform has them.
    pub parallel: bool,
    /// Stops the operation when set.
    pub cancel: Option<CancelFlag>,
}

/// General booleans, sectioning and splitting, with history.
pub trait BooleanOps: Topology {
    /// A boolean between sets of shapes.
    fn boolean(
        op: BooleanOp,
        arguments: &[&Self::Shape],
        tools: &[&Self::Shape],
        options: &BooleanOptions,
    ) -> Result<Modeled<Self::Shape>>;
    /// The curves where two shapes meet.
    fn section(a: &Self::Shape, b: &Self::Shape) -> Result<Self::Shape>;
    /// The curves where a shape meets a plane.
    fn section_plane(shape: &Self::Shape, plane: Plane) -> Result<Self::Shape>;
    /// Splits shapes by tools, keeping every piece.
    fn split(
        arguments: &[&Self::Shape],
        tools: &[&Self::Shape],
        options: &BooleanOptions,
    ) -> Result<Modeled<Self::Shape>>;
}

// --- blends -----------------------------------------------------------------------

/// How a fillet's radius runs along an edge.
#[derive(Clone, Debug, PartialEq)]
pub enum RadiusLaw {
    /// One radius.
    Constant(f64),
    /// Linear from the edge's first vertex to its last, in traversal order.
    Linear {
        /// Radius at the start.
        start: f64,
        /// Radius at the end.
        end: f64,
    },
    /// Interpolated through `(fraction of the edge in 0..=1, radius)` pairs.
    Points(Vec<(f64, f64)>),
}

/// How a chamfer is sized.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ChamferSize {
    /// The same distance on both faces.
    Equal(f64),
    /// One distance on the reference face and another on the opposite face.
    TwoDistances(f64, f64),
    /// A distance on the reference face and an angle from it.
    DistanceAngle(f64, Angle),
}

/// An edge to chamfer: the edge, its size and its reference face.
pub type ChamferEdge<'a, S> = (&'a S, ChamferSize, Option<&'a S>);

/// Blends with per-edge control and history.
pub trait AdvancedBlends: Topology {
    /// Rounds edges, each with its own law.
    fn fillet_edges(
        shape: &Self::Shape,
        edges: &[(&Self::Shape, RadiusLaw)],
    ) -> Result<Modeled<Self::Shape>>;
    /// Bevels edges. Each carries its size and, for asymmetric sizes, a reference face
    /// adjacent to the edge.
    fn chamfer_edges(
        shape: &Self::Shape,
        edges: &[ChamferEdge<'_, Self::Shape>],
    ) -> Result<Modeled<Self::Shape>>;
}

// --- inspection -------------------------------------------------------------------

/// Mass properties at unit density.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MassProperties {
    /// Volume, or area or length for lower-dimensional shapes.
    pub mass: f64,
    /// Centre of mass.
    pub center: Point3,
    /// Inertia tensor about the centre of mass.
    pub inertia: [[f64; 3]; 3],
    /// Principal moments of inertia.
    pub principal_moments: [f64; 3],
    /// Principal axes, parallel to `principal_moments`.
    pub principal_axes: [Vec3; 3],
}

/// The minimum distance between two shapes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Distance {
    /// The distance; zero if they touch.
    pub value: f64,
    /// The closest point on the first shape.
    pub on_a: Point3,
    /// The closest point on the second shape.
    pub on_b: Point3,
}

/// Where a point lies relative to a solid.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PointLocation {
    /// Strictly inside.
    Inside,
    /// On the boundary.
    OnBoundary,
    /// Strictly outside.
    Outside,
}

/// A ray meeting a face.
#[derive(Clone, Debug)]
pub struct RayHit<S> {
    /// The face that was hit.
    pub face: S,
    /// Distance along the ray.
    pub distance: f64,
    /// The position.
    pub point: Point3,
    /// Surface parameters on the face.
    pub uv: (f64, f64),
}

/// The local shape of an edge between two faces.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum EdgeConvexity {
    /// The material angle is under 180 degrees.
    Convex,
    /// The material angle is over 180 degrees.
    Concave,
    /// The faces meet smoothly.
    Tangent,
    /// The edge borders only one face or is a seam.
    Open,
}

/// What a line of a hidden-line drawing represents.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DrawingLineKind {
    /// A sharp model edge.
    Sharp,
    /// An edge between faces that meet tangentially.
    Smooth,
    /// A silhouette of a curved surface.
    Outline,
}

/// A polyline in view coordinates, with X to the right and Y up.
#[derive(Clone, Debug, PartialEq)]
pub struct DrawingLine {
    /// What it represents.
    pub kind: DrawingLineKind,
    /// Whether it is visible from the view direction.
    pub visible: bool,
    /// Points in view coordinates.
    pub points: Vec<[f64; 2]>,
}

/// The view of a hidden-line drawing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DrawingView {
    /// `z` points from the model toward the viewer; `x` is the view's X.
    pub frame: Frame,
    /// Distance for perspective, or `None` for a parallel projection.
    pub perspective: Option<f64>,
    /// Maximum chord deviation of the returned polylines.
    pub deflection: f64,
}

/// Questions about shapes that don't change them.
pub trait Inspection: Topology {
    /// Mass properties of the shape's highest dimension.
    fn mass_properties(shape: &Self::Shape) -> Result<MassProperties>;
    /// The minimum distance between shapes.
    fn distance(a: &Self::Shape, b: &Self::Shape) -> Result<Distance>;
    /// Classifies a point against a solid.
    fn locate_point(solid: &Self::Shape, point: Point3, tolerance: f64) -> Result<PointLocation>;
    /// Every face a ray crosses, nearest first.
    fn ray_cast(
        shape: &Self::Shape,
        origin: Point3,
        direction: Dir3,
        max_distance: Option<f64>,
    ) -> Result<Vec<RayHit<Self::Shape>>>;
    /// The subshapes of `kind` within `root` that contain `sub`, such as an edge's faces.
    fn ancestors(
        root: &Self::Shape,
        sub: &Self::Shape,
        kind: ShapeKind,
    ) -> Result<Vec<Self::Shape>>;
    /// The local shape of an edge within `root`.
    fn edge_convexity(root: &Self::Shape, edge: &Self::Shape) -> Result<EdgeConvexity>;
    /// A face's wires, outer first.
    fn face_wires(face: &Self::Shape) -> Result<Vec<Self::Shape>>;
    /// A wire's edges in connection order.
    fn wire_edges(wire: &Self::Shape) -> Result<Vec<Self::Shape>>;
    /// Tessellation that keeps faces and edges apart, for display and picking.
    fn tessellate(
        shape: &Self::Shape,
        options: &MeshOptions,
        cancel: Option<&CancelFlag>,
    ) -> Result<ShapeTessellation>;
    /// A hidden-line drawing, as drafting views need.
    fn hidden_lines(shape: &Self::Shape, view: &DrawingView) -> Result<Vec<DrawingLine>>;
}

// --- exchange ---------------------------------------------------------------------

/// An RGB colour with components in `0.0..=1.0`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Color {
    /// Red.
    pub r: f32,
    /// Green.
    pub g: f32,
    /// Blue.
    pub b: f32,
}

/// What a node of an exchange document holds.
#[derive(Clone, Debug)]
pub enum NodeContent<S> {
    /// A part. Nodes that share a part share its shape.
    Part(S),
    /// An assembly of child nodes.
    Assembly(Vec<DocumentNode<S>>),
}

/// A node in a product structure.
#[derive(Clone, Debug)]
pub struct DocumentNode<S> {
    /// The product or instance name.
    pub name: Option<String>,
    /// Display colour, if the file gave one.
    pub color: Option<Color>,
    /// Placement relative to the parent.
    pub placement: Transform,
    /// The part or the children.
    pub content: NodeContent<S>,
}

/// A STEP file's product structure.
#[derive(Clone, Debug)]
pub struct Document<S> {
    /// Top-level nodes.
    pub roots: Vec<DocumentNode<S>>,
    /// Model units per metre as the file declared them, such as 1000 for millimetres.
    /// Geometry is always delivered in millimetres.
    pub file_units_per_metre: Option<f64>,
}

/// The STEP schema to write.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum StepSchema {
    /// AP203: geometry and product structure.
    Ap203,
    /// AP214: adds colours and layers.
    #[default]
    Ap214,
    /// AP242: the current standard.
    Ap242,
}

/// Options for writing a STEP document.
#[derive(Clone, Debug, Default)]
pub struct StepWriteOptions {
    /// The schema.
    pub schema: StepSchema,
    /// Stops the operation when set.
    pub cancel: Option<CancelFlag>,
}

/// File formats beyond plain STEP geometry.
pub trait Exchange: Topology {
    /// Serialises a shape in the kernel's own lossless format, for saving documents.
    /// Only the same kind of kernel can read it back.
    fn write_native(shape: &Self::Shape) -> Result<Vec<u8>>;
    /// Reads the kernel's own format.
    fn read_native(data: &[u8]) -> Result<Self::Shape>;
    /// Reads a STEP file with names, colours and product structure.
    fn read_step_document(
        data: &[u8],
        cancel: Option<&CancelFlag>,
    ) -> Result<Document<Self::Shape>>;
    /// Writes a STEP file with names, colours and product structure.
    fn write_step_document(
        document: &Document<Self::Shape>,
        options: &StepWriteOptions,
    ) -> Result<Vec<u8>>;
    /// Reads an IGES file.
    fn read_iges(data: &[u8]) -> Result<Self::Shape>;
    /// Writes shapes as an IGES file.
    fn write_iges(shapes: &[&Self::Shape]) -> Result<Vec<u8>>;
}

// --- surfacing -----------------------------------------------------------------------

/// One side of a boundary fill: an edge, and optionally the face to leave tangent to.
#[derive(Debug)]
pub struct FillEdge<'a, S> {
    /// The boundary edge.
    pub edge: &'a S,
    /// A face owning this edge that the patch continues smoothly.
    pub tangent_to: Option<&'a S>,
}

/// One side of a blend: an edge and the face it borders.
#[derive(Debug)]
pub struct BlendSide<'a, S> {
    /// Where the blend starts. It only has to lie on the face.
    pub edge: &'a S,
    /// The face the blend continues.
    pub face: &'a S,
}

// By hand, because a derive would require `S: Copy` for what only holds references.
impl<S> Clone for FillEdge<'_, S> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<S> Copy for FillEdge<'_, S> {}
impl<S> Clone for BlendSide<'_, S> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<S> Copy for BlendSide<'_, S> {}

/// How closely a blend continues the faces it joins.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum BlendContinuity {
    /// Tangent planes match along the edges.
    G1,
    /// Tangent planes and curvature match along the edges.
    #[default]
    G2,
}

/// How two faces meet along an edge, the worst over samples away from its ends.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContinuityReport {
    /// The largest distance from the edge to either face, in model units.
    pub gap: f64,
    /// The largest crease across the edge: zero when one face continues the other, a
    /// quarter turn at a cube's edge, and a half turn when one folds back over the other.
    pub angle: Angle,
    /// The largest difference in normal curvature in any sampled direction, in 1 / model
    /// units. Compare it with the curvature of the faces.
    pub curvature_gap: f64,
}

/// Which sides of a face's parameter rectangle to extend.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct FaceSides {
    /// The side at the smallest U.
    pub u_min: bool,
    /// The side at the largest U.
    pub u_max: bool,
    /// The side at the smallest V.
    pub v_min: bool,
    /// The side at the largest V.
    pub v_max: bool,
}

/// Surface work across edges: smooth patches, curvature-matched blends and measuring them.
pub trait Surfacing: Topology {
    /// A smooth patch filling a closed loop of edges, tangent to the faces given. It meets
    /// them approximately, within a fraction of a degree, or fails.
    ///
    /// # Errors
    /// [`Error::Operation`](crate::Error::Operation) if the edges beside a tangent edge make the tangency impossible,
    /// or the input is unusable as for [`Construction::fill_boundary`].
    fn fill_boundary_tangent(
        edges: &[FillEdge<'_, Self::Shape>],
        through: &[Point3],
    ) -> Result<Self::Shape>;

    /// A four-sided patch joining two faces across the gap between two edges. `tension`
    /// scales how far it runs along each face's direction before turning; 1 is usual.
    ///
    /// # Errors
    /// [`Error::Operation`](crate::Error::Operation) if an edge doesn't border its face or the edges touch.
    fn blend_faces(
        a: BlendSide<'_, Self::Shape>,
        b: BlendSide<'_, Self::Shape>,
        continuity: BlendContinuity,
        tension: f64,
    ) -> Result<Self::Shape>;

    /// Measures how two faces meet along an edge lying on both.
    ///
    /// # Errors
    /// [`Error::Operation`](crate::Error::Operation) if the edge doesn't border both faces.
    fn measure_continuity(
        a: &Self::Shape,
        b: &Self::Shape,
        edge: &Self::Shape,
    ) -> Result<ContinuityReport>;

    /// The face on its own surface over its parameter rectangle, grown by about `length`
    /// on the chosen sides. Trims and holes go, so this also untrims a face.
    ///
    /// # Errors
    /// [`Error::Operation`](crate::Error::Operation) if `length` isn't positive, no side is chosen or the face
    /// can't grow.
    fn extend_face(face: &Self::Shape, length: f64, sides: FaceSides) -> Result<Self::Shape>;
}
