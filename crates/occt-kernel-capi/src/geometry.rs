//! The geometry domain of the contract: building topology, and reading exact geometry back.
//!
//! The byte formats are described at the top of `capi/src/geometry.cpp`.

use occt_capi_sys as sys;
use occt_kernel::{
    Axis, BSplineCurve, BSplineSurface, Construction, Curve, CurvePoint, Dir3, EdgeGeometry, Error,
    FaceGeometry, Frame, Geometry, ParamRange, Point3, Result, ShapeKind, Surface, SurfacePoint,
};

use super::wire::{Reader, Writer, handles};
use super::{
    Buffer, CapiShape, ensure_ready, failure, query, require_kind, shape_result, with_input,
};

// --- encoding ------------------------------------------------------------------------

fn write_frame(w: &mut Writer, frame: &Frame) {
    w.point(frame.origin)
        .vec3(frame.z.vector())
        .vec3(frame.x.vector());
}

fn write_curve(w: &mut Writer, curve: &Curve) -> Result<()> {
    match curve {
        Curve::Line { origin, direction } => {
            w.u32(1).point(*origin).vec3(direction.vector());
        }
        Curve::Circle { frame, radius } => {
            w.u32(2);
            write_frame(w, frame);
            w.f64(*radius);
        }
        Curve::Ellipse {
            frame,
            major_radius,
            minor_radius,
        } => {
            w.u32(3);
            write_frame(w, frame);
            w.f64(*major_radius).f64(*minor_radius);
        }
        Curve::BSpline(spline) => write_bspline_curve(w, spline),
        _ => {
            return Err(Error::InvalidInput(
                "this curve has no description, so it can't be built".to_owned(),
            ));
        }
    }
    Ok(())
}

fn write_bspline_curve(w: &mut Writer, s: &BSplineCurve) {
    let flags = u32::from(s.periodic) | (u32::from(s.weights.is_some()) << 1);
    w.u32(4)
        .u32(s.degree)
        .u32(flags)
        .count(s.knots.len())
        .count(s.poles.len());
    for k in &s.knots {
        w.f64(*k);
    }
    for m in &s.multiplicities {
        w.u32(*m);
    }
    for p in &s.poles {
        w.point(*p);
    }
    for weight in s.weights.iter().flatten() {
        w.f64(*weight);
    }
}

fn write_surface(w: &mut Writer, surface: &Surface) -> Result<()> {
    match surface {
        Surface::Plane { frame } => {
            w.u32(1);
            write_frame(w, frame);
        }
        Surface::Cylinder { frame, radius } => {
            w.u32(2);
            write_frame(w, frame);
            w.f64(*radius);
        }
        Surface::Cone {
            frame,
            radius,
            half_angle,
        } => {
            w.u32(3);
            write_frame(w, frame);
            w.f64(*radius).f64(*half_angle);
        }
        Surface::Sphere { frame, radius } => {
            w.u32(4);
            write_frame(w, frame);
            w.f64(*radius);
        }
        Surface::Torus {
            frame,
            major_radius,
            minor_radius,
        } => {
            w.u32(5);
            write_frame(w, frame);
            w.f64(*major_radius).f64(*minor_radius);
        }
        Surface::BSpline(spline) => write_bspline_surface(w, spline),
        _ => {
            return Err(Error::InvalidInput(
                "this surface has no description, so it can't be built".to_owned(),
            ));
        }
    }
    Ok(())
}

fn write_bspline_surface(w: &mut Writer, s: &BSplineSurface) {
    let flags = u32::from(s.periodic_u)
        | (u32::from(s.periodic_v) << 1)
        | (u32::from(s.weights.is_some()) << 2);
    w.u32(6)
        .u32(s.degree_u)
        .u32(s.degree_v)
        .u32(flags)
        .count(s.knots_u.len())
        .count(s.knots_v.len())
        .u32(s.pole_count_u)
        .u32(s.pole_count_v);
    for k in &s.knots_u {
        w.f64(*k);
    }
    for m in &s.multiplicities_u {
        w.u32(*m);
    }
    for k in &s.knots_v {
        w.f64(*k);
    }
    for m in &s.multiplicities_v {
        w.u32(*m);
    }
    for p in &s.poles {
        w.point(*p);
    }
    for weight in s.weights.iter().flatten() {
        w.f64(*weight);
    }
}

// --- decoding ------------------------------------------------------------------------

fn malformed(r: &Reader<'_>) -> Error {
    super::malformed(r.op, "geometry buffer")
}

fn read_dir(r: &mut Reader<'_>) -> Result<Dir3> {
    Dir3::new(r.vec3()?).map_err(|_| malformed(r))
}

fn read_frame(r: &mut Reader<'_>) -> Result<Frame> {
    let origin = r.point()?;
    let z = read_dir(r)?;
    let x = r.vec3()?;
    Frame::new(origin, z, x).map_err(|_| malformed(r))
}

/// Checks a count against the bytes left, so that a corrupt one can't allocate much.
fn read_count(r: &mut Reader<'_>, item_size: usize) -> Result<usize> {
    let n = r.u32()? as usize;
    if n.saturating_mul(item_size) > r.bytes.len() {
        return Err(malformed(r));
    }
    Ok(n)
}

fn read_many<'a, T>(
    r: &mut Reader<'a>,
    n: usize,
    mut item: impl FnMut(&mut Reader<'a>) -> Result<T>,
) -> Result<Vec<T>> {
    (0..n).map(|_| item(r)).collect()
}

fn read_curve(r: &mut Reader<'_>) -> Result<Curve> {
    Ok(match r.u32()? {
        0 => Curve::Other,
        1 => Curve::Line {
            origin: r.point()?,
            direction: read_dir(r)?,
        },
        2 => Curve::Circle {
            frame: read_frame(r)?,
            radius: r.f64()?,
        },
        3 => Curve::Ellipse {
            frame: read_frame(r)?,
            major_radius: r.f64()?,
            minor_radius: r.f64()?,
        },
        4 => {
            let degree = r.u32()?;
            let flags = r.u32()?;
            let knots = read_count(r, 12)?;
            let poles = read_count(r, 24)?;
            let rational = flags & 2 != 0;
            Curve::BSpline(BSplineCurve {
                degree,
                knots: read_many(r, knots, Reader::f64)?,
                multiplicities: read_many(r, knots, Reader::u32)?,
                poles: read_many(r, poles, Reader::point)?,
                weights: rational
                    .then(|| read_many(r, poles, Reader::f64))
                    .transpose()?,
                periodic: flags & 1 != 0,
            })
        }
        _ => return Err(malformed(r)),
    })
}

fn read_surface(r: &mut Reader<'_>) -> Result<Surface> {
    Ok(match r.u32()? {
        0 => Surface::Other,
        1 => Surface::Plane {
            frame: read_frame(r)?,
        },
        2 => Surface::Cylinder {
            frame: read_frame(r)?,
            radius: r.f64()?,
        },
        3 => Surface::Cone {
            frame: read_frame(r)?,
            radius: r.f64()?,
            half_angle: r.f64()?,
        },
        4 => Surface::Sphere {
            frame: read_frame(r)?,
            radius: r.f64()?,
        },
        5 => Surface::Torus {
            frame: read_frame(r)?,
            major_radius: r.f64()?,
            minor_radius: r.f64()?,
        },
        6 => {
            let degree_u = r.u32()?;
            let degree_v = r.u32()?;
            let flags = r.u32()?;
            let knots_u = read_count(r, 12)?;
            let knots_v = read_count(r, 12)?;
            let pole_count_u = r.u32()?;
            let pole_count_v = r.u32()?;
            let poles = (pole_count_u as usize).saturating_mul(pole_count_v as usize);
            if poles.saturating_mul(24) > r.bytes.len() {
                return Err(malformed(r));
            }
            let rational = flags & 4 != 0;
            Surface::BSpline(BSplineSurface {
                degree_u,
                degree_v,
                knots_u: read_many(r, knots_u, Reader::f64)?,
                multiplicities_u: read_many(r, knots_u, Reader::u32)?,
                knots_v: read_many(r, knots_v, Reader::f64)?,
                multiplicities_v: read_many(r, knots_v, Reader::u32)?,
                pole_count_u,
                pole_count_v,
                poles: read_many(r, poles, Reader::point)?,
                weights: rational
                    .then(|| read_many(r, poles, Reader::f64))
                    .transpose()?,
                periodic_u: flags & 1 != 0,
                periodic_v: flags & 2 != 0,
            })
        }
        _ => return Err(malformed(r)),
    })
}

fn points_bytes(points: &[Point3]) -> Vec<u8> {
    let mut w = Writer::default();
    for p in points {
        w.point(*p);
    }
    w.0
}

/// A curve from a module buffer.
fn curve_from(op: &'static str, handle: u32) -> Result<Curve> {
    let buffer = Buffer::new(op, handle)?;
    let bytes = buffer.bytes();
    read_curve(&mut Reader { op, bytes: &bytes })
}

/// The handle of a shape list in module memory, for a call taking `(ptr, count)`.
fn with_shapes(
    op: &'static str,
    shapes: &[&CapiShape],
    call: impl FnOnce(usize, usize) -> u32,
) -> Result<CapiShape> {
    let bytes = handles(shapes);
    let handle = with_input(op, &bytes, |ptr, _| call(ptr, shapes.len()))?;
    shape_result(op, handle)
}

// --- contract -------------------------------------------------------------------------

impl Construction for super::CapiKernel {
    fn make_vertex(point: Point3) -> Result<CapiShape> {
        ensure_ready()?;
        // SAFETY: the module is ready and the call takes only scalars.
        let handle = unsafe { sys::occt_make_vertex(point.x, point.y, point.z) };
        shape_result("make_vertex", handle)
    }

    fn make_edge(curve: &Curve, range: Option<ParamRange>) -> Result<CapiShape> {
        const OP: &str = "make_edge";
        ensure_ready()?;
        let mut w = Writer::default();
        write_curve(&mut w, curve)?;
        let range_or_nan = range.map_or((f64::NAN, f64::NAN), |r| (r.start, r.end));
        w.u32(u32::from(range.is_some()))
            .f64(range_or_nan.0)
            .f64(range_or_nan.1);
        let handle = with_input(OP, &w.0, |ptr, len| {
            // SAFETY: `ptr..ptr + len` holds the input for the duration of the call, and the
            // module has no alignment requirement on it.
            unsafe { sys::occt_make_edge(ptr, len) }
        })?;
        shape_result(OP, handle)
    }

    fn make_wire(edges: &[&CapiShape]) -> Result<CapiShape> {
        for edge in edges {
            require_kind(edge, ShapeKind::Edge)?;
        }
        ensure_ready()?;
        with_shapes("make_wire", edges, |ptr, count| {
            // SAFETY: `ptr` holds `count` handles for the duration of the call, and the
            // module has no alignment requirement on it.
            unsafe { sys::occt_make_wire(ptr, count) }
        })
    }

    fn make_planar_face(outer: &CapiShape, holes: &[&CapiShape]) -> Result<CapiShape> {
        require_kind(outer, ShapeKind::Wire)?;
        for hole in holes {
            require_kind(hole, ShapeKind::Wire)?;
        }
        with_shapes("make_planar_face", holes, |ptr, count| {
            // SAFETY: `ptr` holds `count` handles for the duration of the call, and the
            // module has no alignment requirement on it.
            unsafe { sys::occt_make_planar_face(outer.handle, ptr, count) }
        })
    }

    fn make_surface_face(
        surface: &Surface,
        bounds: Option<(ParamRange, ParamRange)>,
        outer: Option<&CapiShape>,
        holes: &[&CapiShape],
    ) -> Result<CapiShape> {
        const OP: &str = "make_surface_face";
        ensure_ready()?;
        if let Some(outer) = outer {
            require_kind(outer, ShapeKind::Wire)?;
        }
        for hole in holes {
            require_kind(hole, ShapeKind::Wire)?;
        }
        let mut w = Writer::default();
        write_surface(&mut w, surface)?;
        w.u32(u32::from(bounds.is_some()));
        let (u, v) = bounds.unwrap_or((
            ParamRange {
                start: 0.0,
                end: 0.0,
            },
            ParamRange {
                start: 0.0,
                end: 0.0,
            },
        ));
        w.f64(u.start).f64(u.end).f64(v.start).f64(v.end);
        w.u32(outer.map_or(0, |o| o.handle));
        w.count(holes.len()).bytes(&handles(holes));
        let handle = with_input(OP, &w.0, |ptr, len| {
            // SAFETY: `ptr..ptr + len` holds the input for the duration of the call, and the
            // module has no alignment requirement on it.
            unsafe { sys::occt_make_surface_face(ptr, len) }
        })?;
        shape_result(OP, handle)
    }

    fn fill_boundary(edges: &[&CapiShape], through: &[Point3]) -> Result<CapiShape> {
        const OP: &str = "fill_boundary";
        for edge in edges {
            require_kind(edge, ShapeKind::Edge)?;
        }
        ensure_ready()?;
        let w = fill_input(edges, &vec![None; edges.len()], through);
        let handle = with_input(OP, &w.0, |ptr, len| {
            // SAFETY: `ptr..ptr + len` holds the input for the duration of the call, and the
            // module has no alignment requirement on it.
            unsafe { sys::occt_fill_boundary(ptr, len) }
        })?;
        shape_result(OP, handle)
    }

    fn sew(shapes: &[&CapiShape], tolerance: f64) -> Result<CapiShape> {
        ensure_ready()?;
        with_shapes("sew", shapes, |ptr, count| {
            // SAFETY: `ptr` holds `count` handles for the duration of the call, and the
            // module has no alignment requirement on it.
            unsafe { sys::occt_sew(ptr, count, tolerance) }
        })
    }

    fn make_solid(shell: &CapiShape) -> Result<CapiShape> {
        require_kind(shell, ShapeKind::Shell)?;
        // SAFETY: the call takes any handle and reports failure with 0.
        shape_result("make_solid", unsafe { sys::occt_make_solid(shell.handle) })
    }

    fn make_compound(shapes: &[&CapiShape]) -> Result<CapiShape> {
        ensure_ready()?;
        with_shapes("make_compound", shapes, |ptr, count| {
            // SAFETY: `ptr` holds `count` handles for the duration of the call, and the
            // module has no alignment requirement on it.
            unsafe { sys::occt_make_compound(ptr, count) }
        })
    }

    fn make_torus(frame: Frame, major_radius: f64, minor_radius: f64) -> Result<CapiShape> {
        const OP: &str = "make_torus";
        ensure_ready()?;
        let mut w = Writer::default();
        write_frame(&mut w, &frame);
        w.f64(major_radius).f64(minor_radius);
        let handle = with_input(OP, &w.0, |ptr, len| {
            // SAFETY: `ptr..ptr + len` holds the input for the duration of the call, and the
            // module has no alignment requirement on it.
            unsafe { sys::occt_make_torus(ptr, len) }
        })?;
        shape_result(OP, handle)
    }

    fn interpolate_curve(points: &[Point3], closed: bool) -> Result<Curve> {
        const OP: &str = "interpolate_curve";
        ensure_ready()?;
        let bytes = points_bytes(points);
        let handle = with_input(OP, &bytes, |ptr, _| {
            // SAFETY: `ptr` holds points.len() * 3 doubles for the duration of the call, and
            // the module has no alignment requirement on it.
            unsafe { sys::occt_interpolate_curve(ptr, points.len(), i32::from(closed)) }
        })?;
        curve_from(OP, handle)
    }

    fn fit_curve(points: &[Point3], tolerance: f64) -> Result<Curve> {
        const OP: &str = "fit_curve";
        ensure_ready()?;
        let bytes = points_bytes(points);
        let handle = with_input(OP, &bytes, |ptr, _| {
            // SAFETY: `ptr` holds points.len() * 3 doubles for the duration of the call, and
            // the module has no alignment requirement on it.
            unsafe { sys::occt_fit_curve(ptr, points.len(), tolerance) }
        })?;
        curve_from(OP, handle)
    }

    fn make_helix(
        axis: Axis,
        radius: f64,
        pitch: f64,
        height: f64,
        left_handed: bool,
    ) -> Result<CapiShape> {
        ensure_ready()?;
        let (o, d) = (axis.origin, axis.direction.vector());
        // SAFETY: the module is ready and the call takes only scalars.
        let handle = unsafe {
            sys::occt_make_helix(
                o.x,
                o.y,
                o.z,
                d.x,
                d.y,
                d.z,
                radius,
                pitch,
                height,
                i32::from(left_handed),
            )
        };
        shape_result("make_helix", handle)
    }

    fn offset_wire(wire: &CapiShape, distance: f64) -> Result<CapiShape> {
        require_kind(wire, ShapeKind::Wire)?;
        // SAFETY: the call takes any handle and a scalar, and reports failure with 0.
        shape_result("offset_wire", unsafe {
            sys::occt_offset_wire(wire.handle, distance)
        })
    }
}

impl Geometry for super::CapiKernel {
    fn edge_geometry(edge: &CapiShape) -> Result<EdgeGeometry> {
        const OP: &str = "edge_geometry";
        require_kind(edge, ShapeKind::Edge)?;
        // SAFETY: the call takes any handle and reports failure with 0.
        let buffer = Buffer::new(OP, unsafe { sys::occt_edge_geometry(edge.handle) })?;
        let bytes = buffer.bytes();
        let mut r = Reader {
            op: OP,
            bytes: &bytes,
        };
        Ok(EdgeGeometry {
            curve: read_curve(&mut r)?,
            range: ParamRange {
                start: r.f64()?,
                end: r.f64()?,
            },
            reversed: r.u32()? != 0,
        })
    }

    fn edge_point(edge: &CapiShape, parameter: f64) -> Result<CurvePoint> {
        const OP: &str = "edge_point";
        require_kind(edge, ShapeKind::Edge)?;
        // SAFETY: the call takes any handle and a scalar, and reports failure with 0.
        let buffer = Buffer::new(OP, unsafe { sys::occt_edge_point(edge.handle, parameter) })?;
        let bytes = buffer.bytes();
        let mut r = Reader {
            op: OP,
            bytes: &bytes,
        };
        Ok(CurvePoint {
            point: r.point()?,
            derivative: r.vec3()?,
            curvature: r.f64()?,
        })
    }

    fn face_geometry(face: &CapiShape) -> Result<FaceGeometry> {
        const OP: &str = "face_geometry";
        require_kind(face, ShapeKind::Face)?;
        // SAFETY: the call takes any handle and reports failure with 0.
        let buffer = Buffer::new(OP, unsafe { sys::occt_face_geometry(face.handle) })?;
        let bytes = buffer.bytes();
        let mut r = Reader {
            op: OP,
            bytes: &bytes,
        };
        Ok(FaceGeometry {
            surface: read_surface(&mut r)?,
            u: ParamRange {
                start: r.f64()?,
                end: r.f64()?,
            },
            v: ParamRange {
                start: r.f64()?,
                end: r.f64()?,
            },
            reversed: r.u32()? != 0,
        })
    }

    fn face_point(face: &CapiShape, u: f64, v: f64) -> Result<SurfacePoint> {
        const OP: &str = "face_point";
        require_kind(face, ShapeKind::Face)?;
        // SAFETY: the call takes any handle and scalars, and reports failure with 0.
        let buffer = Buffer::new(OP, unsafe { sys::occt_face_point(face.handle, u, v) })?;
        let bytes = buffer.bytes();
        let mut r = Reader {
            op: OP,
            bytes: &bytes,
        };
        let has_normal = r.u32()? != 0;
        let point = r.point()?;
        let normal = r.vec3()?;
        Ok(SurfacePoint {
            point,
            normal: if has_normal {
                Some(Dir3::new(normal).map_err(|_| malformed(&r))?)
            } else {
                None
            },
            du: r.vec3()?,
            dv: r.vec3()?,
            principal_curvatures: [r.f64()?, r.f64()?],
        })
    }

    fn project_onto_face(face: &CapiShape, point: Point3) -> Result<(f64, f64, Point3)> {
        const OP: &str = "project_onto_face";
        require_kind(face, ShapeKind::Face)?;
        // SAFETY: the call takes any handle and scalars, and reports failure with 0.
        let buffer = Buffer::new(OP, unsafe {
            sys::occt_project_onto_face(face.handle, point.x, point.y, point.z)
        })?;
        let bytes = buffer.bytes();
        let mut r = Reader {
            op: OP,
            bytes: &bytes,
        };
        Ok((r.f64()?, r.f64()?, r.point()?))
    }

    fn project_onto_edge(edge: &CapiShape, point: Point3) -> Result<(f64, Point3)> {
        const OP: &str = "project_onto_edge";
        require_kind(edge, ShapeKind::Edge)?;
        // SAFETY: the call takes any handle and scalars, and reports failure with 0.
        let buffer = Buffer::new(OP, unsafe {
            sys::occt_project_onto_edge(edge.handle, point.x, point.y, point.z)
        })?;
        let bytes = buffer.bytes();
        let mut r = Reader {
            op: OP,
            bytes: &bytes,
        };
        Ok((r.f64()?, r.point()?))
    }

    fn is_closed(shape: &CapiShape) -> Result<bool> {
        // SAFETY: the call takes any handle and reports failure with -1.
        match unsafe { sys::occt_is_closed(shape.handle) } {
            1 => Ok(true),
            0 => Ok(false),
            _ => Err(failure("is_closed")),
        }
    }

    fn tolerance(shape: &CapiShape) -> Result<f64> {
        // SAFETY: the call takes any handle and reports failure with NaN.
        query("tolerance", unsafe { sys::occt_tolerance(shape.handle) })
    }
}

/// The input of `occt_fill_boundary`, where a support of 0 means no tangency on that edge.
pub(crate) fn fill_input(
    edges: &[&CapiShape],
    supports: &[Option<&CapiShape>],
    through: &[Point3],
) -> Writer {
    let mut w = Writer::default();
    w.count(edges.len()).bytes(&handles(edges));
    for support in supports {
        w.u32(support.map_or(0, |face| face.handle));
    }
    w.count(through.len()).bytes(&points_bytes(through));
    w
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spline() -> BSplineCurve {
        BSplineCurve {
            degree: 2,
            knots: vec![0.0, 1.0],
            multiplicities: vec![3, 3],
            poles: vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 1.0, 0.0),
                Point3::new(2.0, 0.0, 0.0),
            ],
            weights: Some(vec![1.0, 2.0, 1.0]),
            periodic: false,
        }
    }

    #[test]
    fn a_spline_curve_round_trips_through_the_format() {
        let curve = Curve::BSpline(spline());
        let mut w = Writer::default();
        write_curve(&mut w, &curve).unwrap();
        let mut r = Reader {
            op: "test",
            bytes: &w.0,
        };
        assert_eq!(read_curve(&mut r).unwrap(), curve);
        assert!(r.bytes.is_empty());
    }

    #[test]
    fn a_spline_surface_round_trips_through_the_format() {
        let p = Point3::new;
        let surface = Surface::BSpline(BSplineSurface {
            degree_u: 1,
            degree_v: 1,
            knots_u: vec![0.0, 1.0],
            multiplicities_u: vec![2, 2],
            knots_v: vec![0.0, 2.0],
            multiplicities_v: vec![2, 2],
            pole_count_u: 2,
            pole_count_v: 2,
            poles: vec![
                p(0.0, 0.0, 0.0),
                p(0.0, 1.0, 0.0),
                p(1.0, 0.0, 0.0),
                p(1.0, 1.0, 1.0),
            ],
            weights: Some(vec![1.0, 2.0, 3.0, 4.0]),
            periodic_u: false,
            periodic_v: true,
        });
        let mut w = Writer::default();
        write_surface(&mut w, &surface).unwrap();
        let mut r = Reader {
            op: "test",
            bytes: &w.0,
        };
        assert_eq!(read_surface(&mut r).unwrap(), surface);
        assert!(r.bytes.is_empty());
    }

    #[test]
    fn analytic_curves_and_surfaces_round_trip_through_the_format() {
        let frame = Frame::WORLD;
        for curve in [
            Curve::Line {
                origin: Point3::ORIGIN,
                direction: Dir3::Z,
            },
            Curve::Circle { frame, radius: 2.0 },
            Curve::Ellipse {
                frame,
                major_radius: 3.0,
                minor_radius: 1.0,
            },
        ] {
            let mut w = Writer::default();
            write_curve(&mut w, &curve).unwrap();
            assert_eq!(
                read_curve(&mut Reader {
                    op: "test",
                    bytes: &w.0
                })
                .unwrap(),
                curve
            );
        }
        for surface in [
            Surface::Plane { frame },
            Surface::Cylinder { frame, radius: 2.0 },
            Surface::Cone {
                frame,
                radius: 2.0,
                half_angle: 0.5,
            },
            Surface::Sphere { frame, radius: 2.0 },
            Surface::Torus {
                frame,
                major_radius: 3.0,
                minor_radius: 1.0,
            },
        ] {
            let mut w = Writer::default();
            write_surface(&mut w, &surface).unwrap();
            assert_eq!(
                read_surface(&mut Reader {
                    op: "test",
                    bytes: &w.0
                })
                .unwrap(),
                surface
            );
        }
    }

    #[test]
    fn descriptions_of_nothing_are_rejected_for_building() {
        let mut w = Writer::default();
        assert!(matches!(
            write_curve(&mut w, &Curve::Other),
            Err(Error::InvalidInput(_))
        ));
        assert!(matches!(
            write_surface(&mut w, &Surface::Other),
            Err(Error::InvalidInput(_))
        ));
        assert_eq!(
            read_curve(&mut Reader {
                op: "test",
                bytes: &0u32.to_ne_bytes()
            })
            .unwrap(),
            Curve::Other
        );
    }

    #[test]
    fn corrupt_buffers_are_errors() {
        for tag in [4u32, 6, 99] {
            let mut bytes = tag.to_ne_bytes().to_vec();
            bytes.extend_from_slice(&[0xff; 64]);
            assert!(
                read_curve(&mut Reader {
                    op: "test",
                    bytes: &bytes
                })
                .is_err()
            );
            assert!(
                read_surface(&mut Reader {
                    op: "test",
                    bytes: &bytes
                })
                .is_err()
            );
        }
    }
}
