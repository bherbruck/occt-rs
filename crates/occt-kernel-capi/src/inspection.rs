//! [`Inspection`] through the module's inspection entry points.

use occt_capi_sys as sys;
use occt_kernel::{
    CancelFlag, Deflection, Dir3, Distance, DrawingLine, DrawingLineKind, DrawingView,
    EdgeConvexity, EdgePolyline, Error, FaceMesh, Inspection, MassProperties, Mesh, MeshOptions,
    Point3, PointLocation, RayHit, Result, ShapeKind, ShapeTessellation,
};

use crate::wire::{Reader, Writer, cancel_address, cancelled_or};
use crate::{Buffer, CapiKernel, CapiShape, failure, kind_to_raw, malformed, require_kind, u32s};

/// The shapes named by a buffer of bare handles, in order.
fn shapes_from(buffer: &Buffer) -> Vec<CapiShape> {
    u32s(&buffer.bytes()).map(CapiShape::adopt).collect()
}

/// Counts in a module buffer are trusted as far as what they can claim: a bad one runs out
/// of bytes, which [`Reader`] reports, so capacity comes from what the bytes could hold.
fn capacity(r: &Reader<'_>, count: u32, item_size: usize) -> usize {
    (count as usize).min(r.bytes.len() / item_size)
}

fn decode_face(r: &mut Reader<'_>) -> Result<FaceMesh> {
    const OP: &str = "tessellate";
    let vertices = r.u32()?;
    let triangles = r.u32()?;
    let mut positions = Vec::with_capacity(capacity(r, vertices, 24));
    for _ in 0..vertices {
        positions.push(r.point()?);
    }
    let mut normals = Vec::with_capacity(capacity(r, vertices, 24));
    for _ in 0..vertices {
        normals.push(r.vec3()?);
    }
    let mut uvs = Vec::with_capacity(capacity(r, vertices, 16));
    for _ in 0..vertices {
        uvs.push([r.f64()?, r.f64()?]);
    }
    let mut indices = Vec::with_capacity(capacity(r, triangles, 12));
    for _ in 0..triangles {
        let triangle = [r.u32()?, r.u32()?, r.u32()?];
        if triangle.iter().any(|&i| i >= vertices) {
            return Err(malformed(OP, "triangle"));
        }
        indices.push(triangle);
    }
    Ok(FaceMesh {
        mesh: Mesh {
            positions,
            normals,
            indices,
        },
        uvs,
    })
}

fn decode_tessellation(bytes: &[u8]) -> Result<ShapeTessellation> {
    let mut r = Reader {
        op: "tessellate",
        bytes,
    };
    let face_count = r.u32()?;
    let mut faces = Vec::with_capacity(capacity(&r, face_count, 8));
    for _ in 0..face_count {
        faces.push(decode_face(&mut r)?);
    }
    let edge_count = r.u32()?;
    let mut edges = Vec::with_capacity(capacity(&r, edge_count, 4));
    for _ in 0..edge_count {
        let count = r.u32()?;
        let mut points = Vec::with_capacity(capacity(&r, count, 24));
        for _ in 0..count {
            points.push(r.point()?);
        }
        edges.push(EdgePolyline { points });
    }
    Ok(ShapeTessellation { faces, edges })
}

fn decode_drawing(bytes: &[u8]) -> Result<Vec<DrawingLine>> {
    const OP: &str = "hidden_lines";
    let mut r = Reader { op: OP, bytes };
    let count = r.u32()?;
    let mut lines = Vec::with_capacity(capacity(&r, count, 12));
    for _ in 0..count {
        let kind = match r.u32()? {
            0 => DrawingLineKind::Sharp,
            1 => DrawingLineKind::Smooth,
            2 => DrawingLineKind::Outline,
            _ => return Err(malformed(OP, "line kind")),
        };
        let visible = r.u32()? != 0;
        let points = r.u32()?;
        let mut out = Vec::with_capacity(capacity(&r, points, 16));
        for _ in 0..points {
            out.push([r.f64()?, r.f64()?]);
        }
        lines.push(DrawingLine {
            kind,
            visible,
            points: out,
        });
    }
    Ok(lines)
}

impl Inspection for CapiKernel {
    fn mass_properties(shape: &CapiShape) -> Result<MassProperties> {
        const OP: &str = "mass_properties";
        // SAFETY: the call takes any handle and reports failure with 0.
        let buffer = Buffer::new(OP, unsafe { sys::occt_mass_properties(shape.handle) })?;
        let bytes = buffer.bytes();
        let mut r = Reader {
            op: OP,
            bytes: &bytes,
        };
        let mass = r.f64()?;
        let center = r.point()?;
        let mut inertia = [[0.0; 3]; 3];
        for value in inertia.iter_mut().flatten() {
            *value = r.f64()?;
        }
        let principal_moments = [r.f64()?, r.f64()?, r.f64()?];
        let principal_axes = [r.vec3()?, r.vec3()?, r.vec3()?];
        Ok(MassProperties {
            mass,
            center,
            inertia,
            principal_moments,
            principal_axes,
        })
    }

    fn distance(a: &CapiShape, b: &CapiShape) -> Result<Distance> {
        const OP: &str = "distance";
        // SAFETY: the call takes any handles and reports failure with 0.
        let buffer = Buffer::new(OP, unsafe { sys::occt_distance(a.handle, b.handle) })?;
        let bytes = buffer.bytes();
        let mut r = Reader {
            op: OP,
            bytes: &bytes,
        };
        Ok(Distance {
            value: r.f64()?,
            on_a: r.point()?,
            on_b: r.point()?,
        })
    }

    fn locate_point(solid: &CapiShape, point: Point3, tolerance: f64) -> Result<PointLocation> {
        require_kind(solid, ShapeKind::Solid)?;
        // SAFETY: the call takes any handle and scalars, and reports failure with -1.
        let raw =
            unsafe { sys::occt_locate_point(solid.handle, point.x, point.y, point.z, tolerance) };
        match raw {
            0 => Ok(PointLocation::Inside),
            1 => Ok(PointLocation::OnBoundary),
            2 => Ok(PointLocation::Outside),
            _ => Err(failure("locate_point")),
        }
    }

    fn ray_cast(
        shape: &CapiShape,
        origin: Point3,
        direction: Dir3,
        max_distance: Option<f64>,
    ) -> Result<Vec<RayHit<CapiShape>>> {
        const OP: &str = "ray_cast";
        let d = direction.vector();
        // SAFETY: the call takes any handle and scalars, and reports failure with 0.
        let handle = unsafe {
            sys::occt_ray_cast(
                shape.handle,
                origin.x,
                origin.y,
                origin.z,
                d.x,
                d.y,
                d.z,
                max_distance.unwrap_or(-1.0),
            )
        };
        let buffer = Buffer::new(OP, handle)?;
        let bytes = buffer.bytes();
        let mut r = Reader {
            op: OP,
            bytes: &bytes,
        };
        let count = r.u32()?;
        let mut hits = Vec::with_capacity(capacity(&r, count, 52));
        for _ in 0..count {
            hits.push(RayHit {
                face: r.shape()?,
                distance: r.f64()?,
                point: r.point()?,
                uv: (r.f64()?, r.f64()?),
            });
        }
        Ok(hits)
    }

    fn ancestors(root: &CapiShape, sub: &CapiShape, kind: ShapeKind) -> Result<Vec<CapiShape>> {
        let raw = kind_to_raw(kind)?;
        // SAFETY: the call takes any handles and kind, and reports failure with 0.
        let handle = unsafe { sys::occt_ancestors(root.handle, sub.handle, raw) };
        Ok(shapes_from(&Buffer::new("ancestors", handle)?))
    }

    fn edge_convexity(root: &CapiShape, edge: &CapiShape) -> Result<EdgeConvexity> {
        require_kind(edge, ShapeKind::Edge)?;
        // SAFETY: the call takes any handles and reports failure with -1.
        match unsafe { sys::occt_edge_convexity(root.handle, edge.handle) } {
            0 => Ok(EdgeConvexity::Convex),
            1 => Ok(EdgeConvexity::Concave),
            2 => Ok(EdgeConvexity::Tangent),
            3 => Ok(EdgeConvexity::Open),
            _ => Err(failure("edge_convexity")),
        }
    }

    fn face_wires(face: &CapiShape) -> Result<Vec<CapiShape>> {
        require_kind(face, ShapeKind::Face)?;
        // SAFETY: the call takes any handle and reports failure with 0.
        let handle = unsafe { sys::occt_face_wires(face.handle) };
        Ok(shapes_from(&Buffer::new("face_wires", handle)?))
    }

    fn wire_edges(wire: &CapiShape) -> Result<Vec<CapiShape>> {
        require_kind(wire, ShapeKind::Wire)?;
        // SAFETY: the call takes any handle and reports failure with 0.
        let handle = unsafe { sys::occt_wire_edges(wire.handle) };
        Ok(shapes_from(&Buffer::new("wire_edges", handle)?))
    }

    fn tessellate(
        shape: &CapiShape,
        options: &MeshOptions,
        cancel: Option<&CancelFlag>,
    ) -> Result<ShapeTessellation> {
        const OP: &str = "tessellate";
        // The module can't see the flag on wasm, so this is the check that works there. It
        // also spares a shape that's already meshed, which gives the mesher nothing to stop.
        if cancel.is_some_and(CancelFlag::is_cancelled) {
            return Err(Error::Cancelled);
        }
        let (linear, relative) = match options.deflection {
            Deflection::Absolute(distance) => (distance, 0),
            Deflection::Relative(fraction) => (fraction, 1),
        };
        let angular = options.angular_deflection.as_radians();
        let address = cancel_address(cancel);
        // SAFETY: the call takes any handle and scalars and reports failure with 0. `address`
        // is 0 or the flag's byte, which `cancel` keeps alive throughout the call.
        let handle =
            unsafe { sys::occt_tessellate(shape.handle, linear, angular, relative, address) };
        let buffer = Buffer::new(OP, handle).map_err(|e| cancelled_or(cancel, e))?;
        decode_tessellation(&buffer.bytes())
    }

    fn hidden_lines(shape: &CapiShape, view: &DrawingView) -> Result<Vec<DrawingLine>> {
        const OP: &str = "hidden_lines";
        let mut input = Writer::default();
        input
            .point(view.frame.origin)
            .vec3(view.frame.z.vector())
            .vec3(view.frame.x.vector())
            .f64(view.perspective.unwrap_or(-1.0))
            .f64(view.deflection);
        let handle = crate::with_input(OP, &input.0, |ptr, len| {
            // SAFETY: `ptr..ptr + len` holds the input for the duration of the call, and the
            // module has no alignment requirement on it.
            unsafe { sys::occt_hidden_lines(shape.handle, ptr, len) }
        })?;
        let buffer = Buffer::new(OP, handle)?;
        decode_drawing(&buffer.bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_malformed<T>(result: Result<T>) -> bool {
        matches!(result, Err(Error::Operation { .. }))
    }

    #[test]
    fn truncated_buffers_are_errors() {
        assert!(is_malformed(decode_tessellation(&[])));
        assert!(is_malformed(decode_drawing(&[])));
        // Claims more faces and lines than the bytes could hold.
        let claim = u32::MAX.to_ne_bytes();
        assert!(is_malformed(decode_tessellation(&claim)));
        assert!(is_malformed(decode_drawing(&claim)));
    }

    #[test]
    fn decodes_a_face_and_rejects_a_bad_index() {
        let encode = |index: u32| {
            let mut w = Writer::default();
            w.u32(1).u32(1).u32(1);
            for _ in 0..8 {
                w.f64(0.5);
            }
            w.u32(0).u32(0).u32(index);
            w.u32(0);
            w.0
        };
        let good = decode_tessellation(&encode(0)).unwrap();
        assert_eq!(good.faces[0].uvs, [[0.5, 0.5]]);
        assert!(good.edges.is_empty());
        assert!(is_malformed(decode_tessellation(&encode(1))));
    }

    #[test]
    fn decodes_a_drawing_and_rejects_a_bad_kind() {
        let encode = |kind: u32| {
            let mut w = Writer::default();
            w.u32(1).u32(kind).u32(1).u32(1).f64(1.0).f64(2.0);
            w.0
        };
        let lines = decode_drawing(&encode(2)).unwrap();
        assert_eq!(lines[0].kind, DrawingLineKind::Outline);
        assert!(lines[0].visible);
        assert_eq!(lines[0].points, [[1.0, 2.0]]);
        assert!(is_malformed(decode_drawing(&encode(9))));
    }
}
