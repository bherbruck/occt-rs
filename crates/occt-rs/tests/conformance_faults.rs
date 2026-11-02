//! The conformance suite has to be able to fail. This wraps the real kernel, breaks one
//! behaviour at a time, and checks that the suite notices. Without this, a green run
//! says nothing about whether the checks can see anything.

use std::cell::Cell;
use std::panic;

use occt_conformance::checks;
use occt_kernel::{
    Aabb, AdvancedBlends, AdvancedSweeps, Angle, Axis, BlendContinuity, BlendSide, Blends,
    BooleanOp, BooleanOps, BooleanOptions, Booleans, CancelFlag, ChamferEdge, Construction,
    ContinuityReport, Curve, CurveKind, CurvePoint, Dir3, Distance, Document, DrawingLine,
    DrawingView, EdgeConvexity, EdgeGeometry, Exchange, FaceGeometry, FaceSides, Features,
    FillEdge, Frame, Geometry, Inspection, Kernel, LoftOptions, MassProperties, Mesh, MeshOptions,
    Modeled, ParamRange, Plane, Point3, PointLocation, Primitives, Properties, RadiusLaw, RayHit,
    Result, ShapeKind, ShapeTessellation, StepIo, StepWriteOptions, Surface, SurfaceKind,
    SurfacePoint, Surfacing, SweepOptions, Sweeps, Tessellation, Topology, Transform, Transforms,
    Vec3,
};
use occt_kernel_capi::CapiKernel as Real;

#[derive(Clone, Copy, PartialEq)]
enum Fault {
    None,
    VolumeOffByOnePart,
    AreaOffByOnePart,
    ReversedWinding,
    OpenMesh,
    HashDiffersPerCall,
    EverythingIsSame,
    DuplicateSubshapes,
    FuseIgnoresSecond,
    CutIgnoresSecond,
    FilletDoesNothing,
    TranslateDoesNothing,
    ScaleSquaresFactor,
    RotateBackwards,
    ExtrudeHalfway,
    RevolveFullTurnAlways,
    StepAcceptsGarbage,
    StepLosesPlacement,
    CylinderOffAxis,
    ConeRadiiSwapped,
    PolygonIgnoresWarp,
    BlendIgnoresCurvature,
    ContinuityHidesCreases,
    ContinuityHidesCurvature,
    TangentFillIsFree,
    ExtendGrowsEverySide,
}

thread_local!(static FAULT: Cell<Fault> = const { Cell::new(Fault::None) });

fn fault(which: Fault) -> bool {
    FAULT.get() == which
}

struct Faulty;

impl Topology for Faulty {
    type Shape = <Real as Topology>::Shape;

    fn kind(shape: &Self::Shape) -> Result<ShapeKind> {
        Real::kind(shape)
    }
    fn is_same(a: &Self::Shape, b: &Self::Shape) -> bool {
        fault(Fault::EverythingIsSame) || Real::is_same(a, b)
    }
    fn hash_code(shape: &Self::Shape) -> u32 {
        thread_local!(static NEXT: Cell<u32> = const { Cell::new(0) });
        if fault(Fault::HashDiffersPerCall) {
            NEXT.set(NEXT.get() + 1);
            return NEXT.get();
        }
        Real::hash_code(shape)
    }
    fn is_valid(shape: &Self::Shape) -> Result<bool> {
        Real::is_valid(shape)
    }
    fn subshapes(shape: &Self::Shape, kind: ShapeKind) -> Result<Vec<Self::Shape>> {
        let mut parts = Real::subshapes(shape, kind)?;
        if fault(Fault::DuplicateSubshapes) && !parts.is_empty() {
            parts.push(parts[0].clone());
        }
        Ok(parts)
    }
}

impl Primitives for Faulty {
    fn make_box(corner: Point3, size: Vec3) -> Result<Self::Shape> {
        Real::make_box(corner, size)
    }
    fn make_cylinder(axis: Axis, radius: f64, height: f64) -> Result<Self::Shape> {
        if fault(Fault::CylinderOffAxis) {
            return Real::make_cylinder(Axis::Z, radius, height);
        }
        Real::make_cylinder(axis, radius, height)
    }
    fn make_cone(axis: Axis, radius1: f64, radius2: f64, height: f64) -> Result<Self::Shape> {
        if fault(Fault::ConeRadiiSwapped) {
            return Real::make_cone(axis, radius2, radius1, height);
        }
        Real::make_cone(axis, radius1, radius2, height)
    }
    fn make_sphere(center: Point3, radius: f64) -> Result<Self::Shape> {
        Real::make_sphere(center, radius)
    }
    fn make_polygon_face(points: &[Point3]) -> Result<Self::Shape> {
        if fault(Fault::PolygonIgnoresWarp) {
            let flat: Vec<_> = points
                .iter()
                .map(|p| Point3::new(p.x, p.y, points[0].z))
                .collect();
            return Real::make_polygon_face(&flat);
        }
        Real::make_polygon_face(points)
    }
}

impl Sweeps for Faulty {
    fn extrude(shape: &Self::Shape, by: Vec3) -> Result<Self::Shape> {
        Real::extrude(
            shape,
            if fault(Fault::ExtrudeHalfway) {
                by / 2.0
            } else {
                by
            },
        )
    }
    fn revolve(shape: &Self::Shape, axis: Axis, angle: Angle) -> Result<Self::Shape> {
        let angle = if fault(Fault::RevolveFullTurnAlways) {
            Angle::FULL_TURN
        } else {
            angle
        };
        Real::revolve(shape, axis, angle)
    }
}

impl Booleans for Faulty {
    fn fuse(a: &Self::Shape, b: &Self::Shape) -> Result<Self::Shape> {
        if fault(Fault::FuseIgnoresSecond) {
            return Ok(a.clone());
        }
        Real::fuse(a, b)
    }
    fn cut(a: &Self::Shape, b: &Self::Shape) -> Result<Self::Shape> {
        if fault(Fault::CutIgnoresSecond) {
            return Ok(a.clone());
        }
        Real::cut(a, b)
    }
    fn common(a: &Self::Shape, b: &Self::Shape) -> Result<Self::Shape> {
        Real::common(a, b)
    }
}

impl Blends for Faulty {
    fn fillet(shape: &Self::Shape, edges: &[&Self::Shape], radius: f64) -> Result<Self::Shape> {
        if fault(Fault::FilletDoesNothing) {
            return Ok(shape.clone());
        }
        Real::fillet(shape, edges, radius)
    }
    fn chamfer(shape: &Self::Shape, edges: &[&Self::Shape], distance: f64) -> Result<Self::Shape> {
        Real::chamfer(shape, edges, distance)
    }
}

impl Transforms for Faulty {
    fn translate(shape: &Self::Shape, by: Vec3) -> Result<Self::Shape> {
        if fault(Fault::TranslateDoesNothing) {
            return Ok(shape.clone());
        }
        Real::translate(shape, by)
    }
    fn rotate(shape: &Self::Shape, axis: Axis, angle: Angle) -> Result<Self::Shape> {
        if fault(Fault::RotateBackwards) {
            return Real::rotate(shape, axis, Angle::radians(-angle.as_radians()));
        }
        Real::rotate(shape, axis, angle)
    }
    fn scale(shape: &Self::Shape, center: Point3, factor: f64) -> Result<Self::Shape> {
        let factor = if fault(Fault::ScaleSquaresFactor) {
            factor * factor
        } else {
            factor
        };
        Real::scale(shape, center, factor)
    }
}

impl Properties for Faulty {
    fn volume(shape: &Self::Shape) -> Result<f64> {
        let off = if fault(Fault::VolumeOffByOnePart) {
            1.001
        } else {
            1.0
        };
        Real::volume(shape).map(|v| v * off)
    }
    fn area(shape: &Self::Shape) -> Result<f64> {
        let off = if fault(Fault::AreaOffByOnePart) {
            1.001
        } else {
            1.0
        };
        Real::area(shape).map(|a| a * off)
    }
    fn length(shape: &Self::Shape) -> Result<f64> {
        Real::length(shape)
    }
    fn center_of_mass(shape: &Self::Shape) -> Result<Point3> {
        Real::center_of_mass(shape)
    }
    fn bounding_box(shape: &Self::Shape) -> Result<Aabb> {
        Real::bounding_box(shape)
    }
    fn vertex_point(vertex: &Self::Shape) -> Result<Point3> {
        Real::vertex_point(vertex)
    }
    fn surface_kind(face: &Self::Shape) -> Result<SurfaceKind> {
        Real::surface_kind(face)
    }
    fn curve_kind(edge: &Self::Shape) -> Result<CurveKind> {
        Real::curve_kind(edge)
    }
}

impl Tessellation for Faulty {
    fn mesh(shape: &Self::Shape, options: &MeshOptions) -> Result<Mesh> {
        let mut mesh = Real::mesh(shape, options)?;
        if fault(Fault::ReversedWinding) {
            mesh.indices.iter_mut().for_each(|t| t.swap(1, 2));
        }
        if fault(Fault::OpenMesh) {
            mesh.indices.pop();
        }
        Ok(mesh)
    }
}

impl StepIo for Faulty {
    fn write_step(shape: &Self::Shape) -> Result<Vec<u8>> {
        Real::write_step(shape)
    }
    fn read_step(data: &[u8]) -> Result<Self::Shape> {
        if fault(Fault::StepAcceptsGarbage) {
            return Real::make_box(Point3::ORIGIN, Vec3::new(1.0, 2.0, 3.0));
        }
        let shape = Real::read_step(data)?;
        if fault(Fault::StepLosesPlacement) {
            let bounds = Real::bounding_box(&shape)?;
            return Real::translate(&shape, Point3::ORIGIN - bounds.min);
        }
        Ok(shape)
    }
}

// The CAD-level domains pass through unchanged; their own tests cover them.
impl Construction for Faulty {
    fn make_vertex(point: Point3) -> Result<<Real as Topology>::Shape> {
        <Real as Construction>::make_vertex(point)
    }
    fn make_edge(curve: &Curve, range: Option<ParamRange>) -> Result<<Real as Topology>::Shape> {
        <Real as Construction>::make_edge(curve, range)
    }
    fn make_wire(edges: &[&<Real as Topology>::Shape]) -> Result<<Real as Topology>::Shape> {
        <Real as Construction>::make_wire(edges)
    }
    fn make_planar_face(
        outer: &<Real as Topology>::Shape,
        holes: &[&<Real as Topology>::Shape],
    ) -> Result<<Real as Topology>::Shape> {
        <Real as Construction>::make_planar_face(outer, holes)
    }
    fn make_surface_face(
        surface: &Surface,
        bounds: Option<(ParamRange, ParamRange)>,
        outer: Option<&<Real as Topology>::Shape>,
        holes: &[&<Real as Topology>::Shape],
    ) -> Result<<Real as Topology>::Shape> {
        <Real as Construction>::make_surface_face(surface, bounds, outer, holes)
    }
    fn fill_boundary(
        edges: &[&<Real as Topology>::Shape],
        through: &[Point3],
    ) -> Result<<Real as Topology>::Shape> {
        <Real as Construction>::fill_boundary(edges, through)
    }
    fn sew(
        shapes: &[&<Real as Topology>::Shape],
        tolerance: f64,
    ) -> Result<<Real as Topology>::Shape> {
        <Real as Construction>::sew(shapes, tolerance)
    }
    fn make_solid(shell: &<Real as Topology>::Shape) -> Result<<Real as Topology>::Shape> {
        <Real as Construction>::make_solid(shell)
    }
    fn make_compound(shapes: &[&<Real as Topology>::Shape]) -> Result<<Real as Topology>::Shape> {
        <Real as Construction>::make_compound(shapes)
    }
    fn make_torus(
        frame: Frame,
        major_radius: f64,
        minor_radius: f64,
    ) -> Result<<Real as Topology>::Shape> {
        <Real as Construction>::make_torus(frame, major_radius, minor_radius)
    }
    fn interpolate_curve(points: &[Point3], closed: bool) -> Result<Curve> {
        <Real as Construction>::interpolate_curve(points, closed)
    }
    fn fit_curve(points: &[Point3], tolerance: f64) -> Result<Curve> {
        <Real as Construction>::fit_curve(points, tolerance)
    }
    fn make_helix(
        axis: Axis,
        radius: f64,
        pitch: f64,
        height: f64,
        left_handed: bool,
    ) -> Result<<Real as Topology>::Shape> {
        <Real as Construction>::make_helix(axis, radius, pitch, height, left_handed)
    }
    fn offset_wire(
        wire: &<Real as Topology>::Shape,
        distance: f64,
    ) -> Result<<Real as Topology>::Shape> {
        <Real as Construction>::offset_wire(wire, distance)
    }
}

impl Geometry for Faulty {
    fn edge_geometry(edge: &<Real as Topology>::Shape) -> Result<EdgeGeometry> {
        <Real as Geometry>::edge_geometry(edge)
    }
    fn edge_point(edge: &<Real as Topology>::Shape, parameter: f64) -> Result<CurvePoint> {
        <Real as Geometry>::edge_point(edge, parameter)
    }
    fn face_geometry(face: &<Real as Topology>::Shape) -> Result<FaceGeometry> {
        <Real as Geometry>::face_geometry(face)
    }
    fn face_point(face: &<Real as Topology>::Shape, u: f64, v: f64) -> Result<SurfacePoint> {
        <Real as Geometry>::face_point(face, u, v)
    }
    fn project_onto_face(
        face: &<Real as Topology>::Shape,
        point: Point3,
    ) -> Result<(f64, f64, Point3)> {
        <Real as Geometry>::project_onto_face(face, point)
    }
    fn project_onto_edge(edge: &<Real as Topology>::Shape, point: Point3) -> Result<(f64, Point3)> {
        <Real as Geometry>::project_onto_edge(edge, point)
    }
    fn is_closed(shape: &<Real as Topology>::Shape) -> Result<bool> {
        <Real as Geometry>::is_closed(shape)
    }
    fn tolerance(shape: &<Real as Topology>::Shape) -> Result<f64> {
        <Real as Geometry>::tolerance(shape)
    }
}

impl AdvancedSweeps for Faulty {
    fn extrude_modeled(
        shape: &<Real as Topology>::Shape,
        by: Vec3,
    ) -> Result<Modeled<<Real as Topology>::Shape>> {
        <Real as AdvancedSweeps>::extrude_modeled(shape, by)
    }
    fn revolve_modeled(
        shape: &<Real as Topology>::Shape,
        axis: Axis,
        angle: Angle,
    ) -> Result<Modeled<<Real as Topology>::Shape>> {
        <Real as AdvancedSweeps>::revolve_modeled(shape, axis, angle)
    }
    fn extrude_tapered(
        face: &<Real as Topology>::Shape,
        by: Vec3,
        taper: Angle,
    ) -> Result<<Real as Topology>::Shape> {
        <Real as AdvancedSweeps>::extrude_tapered(face, by, taper)
    }
    fn sweep(
        profile: &<Real as Topology>::Shape,
        path: &<Real as Topology>::Shape,
        options: &SweepOptions,
    ) -> Result<Modeled<<Real as Topology>::Shape>> {
        <Real as AdvancedSweeps>::sweep(profile, path, options)
    }
    fn loft(
        sections: &[&<Real as Topology>::Shape],
        options: &LoftOptions,
    ) -> Result<<Real as Topology>::Shape> {
        <Real as AdvancedSweeps>::loft(sections, options)
    }
}

impl Features for Faulty {
    fn shell(
        solid: &<Real as Topology>::Shape,
        open_faces: &[&<Real as Topology>::Shape],
        thickness: f64,
    ) -> Result<Modeled<<Real as Topology>::Shape>> {
        <Real as Features>::shell(solid, open_faces, thickness)
    }
    fn offset(
        shape: &<Real as Topology>::Shape,
        distance: f64,
    ) -> Result<<Real as Topology>::Shape> {
        <Real as Features>::offset(shape, distance)
    }
    fn thicken(
        sheet: &<Real as Topology>::Shape,
        thickness: f64,
    ) -> Result<<Real as Topology>::Shape> {
        <Real as Features>::thicken(sheet, thickness)
    }
    fn draft(
        solid: &<Real as Topology>::Shape,
        faces: &[&<Real as Topology>::Shape],
        pull: Dir3,
        neutral: Plane,
        angle: Angle,
    ) -> Result<Modeled<<Real as Topology>::Shape>> {
        <Real as Features>::draft(solid, faces, pull, neutral, angle)
    }
    fn remove_faces(
        solid: &<Real as Topology>::Shape,
        faces: &[&<Real as Topology>::Shape],
    ) -> Result<Modeled<<Real as Topology>::Shape>> {
        <Real as Features>::remove_faces(solid, faces)
    }
    fn unify(shape: &<Real as Topology>::Shape) -> Result<Modeled<<Real as Topology>::Shape>> {
        <Real as Features>::unify(shape)
    }
    fn heal(
        shape: &<Real as Topology>::Shape,
        tolerance: f64,
    ) -> Result<<Real as Topology>::Shape> {
        <Real as Features>::heal(shape, tolerance)
    }
    fn transformed(
        shape: &<Real as Topology>::Shape,
        transform: &Transform,
    ) -> Result<<Real as Topology>::Shape> {
        <Real as Features>::transformed(shape, transform)
    }
    fn placed(
        shape: &<Real as Topology>::Shape,
        transform: &Transform,
    ) -> Result<<Real as Topology>::Shape> {
        <Real as Features>::placed(shape, transform)
    }
    fn reversed(shape: &<Real as Topology>::Shape) -> Result<<Real as Topology>::Shape> {
        <Real as Features>::reversed(shape)
    }
}

impl BooleanOps for Faulty {
    fn boolean(
        op: BooleanOp,
        arguments: &[&<Real as Topology>::Shape],
        tools: &[&<Real as Topology>::Shape],
        options: &BooleanOptions,
    ) -> Result<Modeled<<Real as Topology>::Shape>> {
        <Real as BooleanOps>::boolean(op, arguments, tools, options)
    }
    fn section(
        a: &<Real as Topology>::Shape,
        b: &<Real as Topology>::Shape,
    ) -> Result<<Real as Topology>::Shape> {
        <Real as BooleanOps>::section(a, b)
    }
    fn section_plane(
        shape: &<Real as Topology>::Shape,
        plane: Plane,
    ) -> Result<<Real as Topology>::Shape> {
        <Real as BooleanOps>::section_plane(shape, plane)
    }
    fn split(
        arguments: &[&<Real as Topology>::Shape],
        tools: &[&<Real as Topology>::Shape],
        options: &BooleanOptions,
    ) -> Result<Modeled<<Real as Topology>::Shape>> {
        <Real as BooleanOps>::split(arguments, tools, options)
    }
}

impl AdvancedBlends for Faulty {
    fn fillet_edges(
        shape: &<Real as Topology>::Shape,
        edges: &[(&<Real as Topology>::Shape, RadiusLaw)],
    ) -> Result<Modeled<<Real as Topology>::Shape>> {
        <Real as AdvancedBlends>::fillet_edges(shape, edges)
    }
    fn chamfer_edges(
        shape: &<Real as Topology>::Shape,
        edges: &[ChamferEdge<'_, <Real as Topology>::Shape>],
    ) -> Result<Modeled<<Real as Topology>::Shape>> {
        <Real as AdvancedBlends>::chamfer_edges(shape, edges)
    }
}

impl Inspection for Faulty {
    fn mass_properties(shape: &<Real as Topology>::Shape) -> Result<MassProperties> {
        <Real as Inspection>::mass_properties(shape)
    }
    fn distance(a: &<Real as Topology>::Shape, b: &<Real as Topology>::Shape) -> Result<Distance> {
        <Real as Inspection>::distance(a, b)
    }
    fn locate_point(
        solid: &<Real as Topology>::Shape,
        point: Point3,
        tolerance: f64,
    ) -> Result<PointLocation> {
        <Real as Inspection>::locate_point(solid, point, tolerance)
    }
    fn ray_cast(
        shape: &<Real as Topology>::Shape,
        origin: Point3,
        direction: Dir3,
        max_distance: Option<f64>,
    ) -> Result<Vec<RayHit<<Real as Topology>::Shape>>> {
        <Real as Inspection>::ray_cast(shape, origin, direction, max_distance)
    }
    fn ancestors(
        root: &<Real as Topology>::Shape,
        sub: &<Real as Topology>::Shape,
        kind: ShapeKind,
    ) -> Result<Vec<<Real as Topology>::Shape>> {
        <Real as Inspection>::ancestors(root, sub, kind)
    }
    fn edge_convexity(
        root: &<Real as Topology>::Shape,
        edge: &<Real as Topology>::Shape,
    ) -> Result<EdgeConvexity> {
        <Real as Inspection>::edge_convexity(root, edge)
    }
    fn face_wires(face: &<Real as Topology>::Shape) -> Result<Vec<<Real as Topology>::Shape>> {
        <Real as Inspection>::face_wires(face)
    }
    fn wire_edges(wire: &<Real as Topology>::Shape) -> Result<Vec<<Real as Topology>::Shape>> {
        <Real as Inspection>::wire_edges(wire)
    }
    fn tessellate(
        shape: &<Real as Topology>::Shape,
        options: &MeshOptions,
        cancel: Option<&CancelFlag>,
    ) -> Result<ShapeTessellation> {
        <Real as Inspection>::tessellate(shape, options, cancel)
    }
    fn hidden_lines(
        shape: &<Real as Topology>::Shape,
        view: &DrawingView,
    ) -> Result<Vec<DrawingLine>> {
        <Real as Inspection>::hidden_lines(shape, view)
    }
}

impl Exchange for Faulty {
    fn write_native(shape: &<Real as Topology>::Shape) -> Result<Vec<u8>> {
        <Real as Exchange>::write_native(shape)
    }
    fn read_native(data: &[u8]) -> Result<<Real as Topology>::Shape> {
        <Real as Exchange>::read_native(data)
    }
    fn read_step_document(
        data: &[u8],
        cancel: Option<&CancelFlag>,
    ) -> Result<Document<<Real as Topology>::Shape>> {
        <Real as Exchange>::read_step_document(data, cancel)
    }
    fn write_step_document(
        document: &Document<<Real as Topology>::Shape>,
        options: &StepWriteOptions,
    ) -> Result<Vec<u8>> {
        <Real as Exchange>::write_step_document(document, options)
    }
    fn read_iges(data: &[u8]) -> Result<<Real as Topology>::Shape> {
        <Real as Exchange>::read_iges(data)
    }
    fn write_iges(shapes: &[&<Real as Topology>::Shape]) -> Result<Vec<u8>> {
        <Real as Exchange>::write_iges(shapes)
    }
}

impl Surfacing for Faulty {
    fn fill_boundary_tangent(
        edges: &[FillEdge<'_, <Real as Topology>::Shape>],
        through: &[Point3],
    ) -> Result<<Real as Topology>::Shape> {
        if fault(Fault::TangentFillIsFree) {
            let free: Vec<_> = edges
                .iter()
                .map(|side| FillEdge {
                    edge: side.edge,
                    tangent_to: None,
                })
                .collect();
            return <Real as Surfacing>::fill_boundary_tangent(&free, through);
        }
        <Real as Surfacing>::fill_boundary_tangent(edges, through)
    }
    fn blend_faces(
        a: BlendSide<'_, <Real as Topology>::Shape>,
        b: BlendSide<'_, <Real as Topology>::Shape>,
        continuity: BlendContinuity,
        tension: f64,
    ) -> Result<<Real as Topology>::Shape> {
        let continuity = if fault(Fault::BlendIgnoresCurvature) {
            BlendContinuity::G1
        } else {
            continuity
        };
        <Real as Surfacing>::blend_faces(a, b, continuity, tension)
    }
    fn measure_continuity(
        a: &<Real as Topology>::Shape,
        b: &<Real as Topology>::Shape,
        edge: &<Real as Topology>::Shape,
    ) -> Result<ContinuityReport> {
        let mut report = <Real as Surfacing>::measure_continuity(a, b, edge)?;
        if fault(Fault::ContinuityHidesCreases) {
            report.angle = Angle::radians(0.0);
        }
        if fault(Fault::ContinuityHidesCurvature) {
            report.curvature_gap = 0.0;
        }
        Ok(report)
    }
    fn extend_face(
        face: &<Real as Topology>::Shape,
        length: f64,
        sides: FaceSides,
    ) -> Result<<Real as Topology>::Shape> {
        let sides = if fault(Fault::ExtendGrowsEverySide) {
            FaceSides {
                u_min: true,
                u_max: true,
                v_min: true,
                v_max: true,
            }
        } else {
            sides
        };
        <Real as Surfacing>::extend_face(face, length, sides)
    }
}

impl Kernel for Faulty {
    const NAME: &'static str = "faulty";
    const NOTICE: &'static str = "";

    fn version() -> Result<String> {
        Real::version()
    }
    fn ready() -> Result<()> {
        Real::ready()
    }
}

/// The checks that fail with a fault in place.
fn failures(which: Fault) -> Vec<&'static str> {
    FAULT.set(which);
    let failed = checks::<Faulty>()
        .into_iter()
        .filter(|(_, check)| panic::catch_unwind(*check).is_err())
        .map(|(name, _)| name)
        .collect();
    FAULT.set(Fault::None);
    failed
}

#[test]
fn every_fault_is_caught_by_the_expected_check() {
    panic::set_hook(Box::new(|_| {}));
    let expectations: &[(&str, Fault, &str)] = &[
        (
            "volume 0.1% high",
            Fault::VolumeOffByOnePart,
            "box_measures",
        ),
        ("area 0.1% high", Fault::AreaOffByOnePart, "box_measures"),
        (
            "triangles wound backwards",
            Fault::ReversedWinding,
            "mesh_winds_outward",
        ),
        (
            "one triangle missing",
            Fault::OpenMesh,
            "mesh_is_watertight",
        ),
        (
            "hash not stable",
            Fault::HashDiffersPerCall,
            "identity_is_an_equivalence_with_consistent_hash",
        ),
        (
            "is_same always true",
            Fault::EverythingIsSame,
            "box_subshapes_are_distinct",
        ),
        (
            "is_same always true",
            Fault::EverythingIsSame,
            "separately_built_shapes_are_not_same",
        ),
        (
            "subshape listed twice",
            Fault::DuplicateSubshapes,
            "box_topology_counts",
        ),
        (
            "fuse ignores operand",
            Fault::FuseIgnoresSecond,
            "fuse_cut_common_of_overlapping_boxes",
        ),
        (
            "cut ignores operand",
            Fault::CutIgnoresSecond,
            "cut_cylinder_through_box",
        ),
        (
            "fillet does nothing",
            Fault::FilletDoesNothing,
            "fillet_single_edge",
        ),
        (
            "fillet does nothing",
            Fault::FilletDoesNothing,
            "impossible_fillet_is_rejected",
        ),
        (
            "translate does nothing",
            Fault::TranslateDoesNothing,
            "translate_moves_without_changing_size",
        ),
        (
            "scale squares factor",
            Fault::ScaleSquaresFactor,
            "scale_about_a_point",
        ),
        (
            "rotate backwards",
            Fault::RotateBackwards,
            "rotate_quarter_turn_about_offset_axis",
        ),
        (
            "extrude stops halfway",
            Fault::ExtrudeHalfway,
            "extrude_face_along_oblique_vector",
        ),
        (
            "revolve ignores angle",
            Fault::RevolveFullTurnAlways,
            "revolve_rectangle_partial_turn",
        ),
        (
            "STEP accepts garbage",
            Fault::StepAcceptsGarbage,
            "unparseable_step_is_rejected",
        ),
        (
            "STEP moves the shape",
            Fault::StepLosesPlacement,
            "step_round_trip_keeps_placement",
        ),
        (
            "cylinder ignores axis",
            Fault::CylinderOffAxis,
            "cylinder_measures",
        ),
        (
            "cone radii swapped",
            Fault::ConeRadiiSwapped,
            "cone_measures",
        ),
        (
            "polygon flattened",
            Fault::PolygonIgnoresWarp,
            "non_planar_polygon_is_rejected",
        ),
        (
            "G2 blend quietly made G1",
            Fault::BlendIgnoresCurvature,
            "a_g2_blend_matches_a_sphere",
        ),
        (
            "measuring reports no crease",
            Fault::ContinuityHidesCreases,
            "a_cube_edge_is_a_quarter_turn",
        ),
        (
            "measuring reports no curvature jump",
            Fault::ContinuityHidesCurvature,
            "a_circular_fillet_is_g1_but_not_g2",
        ),
        (
            "tangent fill ignores the faces",
            Fault::TangentFillIsFree,
            "a_tangent_fill_follows_a_sphere_better_than_a_free_one",
        ),
        (
            "extending ignores the sides chosen",
            Fault::ExtendGrowsEverySide,
            "extending_grows_only_the_chosen_sides",
        ),
    ];

    assert_eq!(
        failures(Fault::None),
        Vec::<&str>::new(),
        "the wrapper alone must pass"
    );

    let missed: Vec<_> = expectations
        .iter()
        .filter(|(_, which, check)| !failures(*which).contains(check))
        .map(|(what, _, check)| format!("{check} missed: {what}"))
        .collect();
    let _ = panic::take_hook();
    assert!(missed.is_empty(), "{missed:#?}");
}
