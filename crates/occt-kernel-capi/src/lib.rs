//! The [`occt_kernel`] contract implemented with OCCT, through the occt_capi module.
//!
//! Nothing OCCT-specific leaves this crate. Shapes are `Send + Sync`: the module holds one
//! lock over the kernel, so calls from different threads are safe but run one at a time.

use std::cell::Cell;
use std::fmt;

use occt_capi_sys as sys;
use occt_kernel::{
    Aabb, Angle, Axis, Blends, Booleans, CurveKind, Deflection, Error, Kernel, Mesh, MeshOptions,
    Point3, Primitives, Properties, Result, ShapeKind, StepIo, SurfaceKind, Sweeps, Tessellation,
    Topology, Transforms, Vec3,
};

mod exchange;
mod features;
mod geometry;
mod inspection;
mod operations;
mod surfacing;
mod wire;

/// The OCCT-backed kernel.
#[derive(Clone, Copy, Debug, Default)]
pub struct CapiKernel;

/// An owned reference to a shape.
///
/// ```
/// fn assert_send_sync<T: Send + Sync>() {}
/// assert_send_sync::<occt_kernel_capi::CapiShape>();
/// ```
pub struct CapiShape {
    handle: u32,
}

// SAFETY: the handle is a plain integer. Every module call that uses it, including
// release, takes the module's kernel-wide lock, so no two threads touch OCCT at once.
// On wasm32 there is one thread and these impls change nothing in practice.
unsafe impl Send for CapiShape {}
// SAFETY: as for Send; `&CapiShape` only allows module calls, which are serialised.
unsafe impl Sync for CapiShape {}

impl CapiShape {
    fn adopt(handle: u32) -> Self {
        Self { handle }
    }
}

impl Drop for CapiShape {
    fn drop(&mut self) {
        // SAFETY: freeing takes any handle; this one is owned and freed only here.
        unsafe { sys::occt_shape_free(self.handle) };
    }
}

impl Clone for CapiShape {
    fn clone(&self) -> Self {
        // SAFETY: the call takes any handle and reports failure with 0.
        let handle = unsafe { sys::occt_shape_clone(self.handle) };
        assert!(handle != 0, "cloning a shape failed: {}", last_error());
        Self::adopt(handle)
    }
}

impl fmt::Debug for CapiShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut s = f.debug_struct("CapiShape");
        s.field("handle", &self.handle);
        match CapiKernel::kind(self) {
            Ok(kind) => s.field("kind", &kind),
            Err(e) => s.field("kind", &format_args!("<{e}>")),
        };
        s.finish()
    }
}

// --- module state ---------------------------------------------------------------

thread_local! {
    static READY: Cell<bool> = const { Cell::new(false) };
}

/// Guards every call that doesn't already hold a shape, which proves readiness.
fn ensure_ready() -> Result<()> {
    if READY.get() {
        return Ok(());
    }
    sys::module_status().map_err(Error::Unavailable)?;
    // SAFETY: the module is present, and this call takes no arguments.
    let abi = unsafe { sys::occt_abi_version() };
    if abi != sys::ABI_VERSION {
        return Err(Error::Unavailable(format!(
            "the occt_capi module has ABI version {abi}, expected {}",
            sys::ABI_VERSION
        )));
    }
    READY.set(true);
    Ok(())
}

fn last_error() -> String {
    // SAFETY: the pair names the current thread's error string, which lives until the
    // next module call on this thread; nothing runs in between.
    let bytes = unsafe { sys::copy_out(sys::occt_last_error_ptr(), sys::occt_last_error_len()) };
    if bytes.is_empty() {
        "unknown error".to_owned()
    } else {
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

fn failure(op: &'static str) -> Error {
    Error::Operation {
        op,
        message: last_error(),
    }
}

fn shape_result(op: &'static str, handle: u32) -> Result<CapiShape> {
    if handle == 0 {
        Err(failure(op))
    } else {
        Ok(CapiShape::adopt(handle))
    }
}

fn with_input<R>(op: &'static str, input: &[u8], f: impl FnOnce(usize, usize) -> R) -> Result<R> {
    sys::with_input(input, f).map_err(|message| Error::Operation { op, message })
}

/// A buffer handle from the module, freed on drop.
struct Buffer(u32);

impl Buffer {
    fn new(op: &'static str, handle: u32) -> Result<Self> {
        if handle == 0 {
            Err(failure(op))
        } else {
            Ok(Self(handle))
        }
    }

    fn bytes(&self) -> Vec<u8> {
        // SAFETY: while this guard lives the buffer isn't freed or changed, and the pair
        // describes it; an unknown handle would read as length 0.
        unsafe { sys::copy_out(sys::occt_buffer_ptr(self.0), sys::occt_buffer_len(self.0)) }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        // SAFETY: freeing takes any handle; this one is owned and freed only here.
        unsafe { sys::occt_buffer_free(self.0) };
    }
}

// --- decoding ---------------------------------------------------------------------

fn malformed(op: &'static str, what: &str) -> Error {
    Error::Operation {
        op,
        message: format!("the module returned a malformed {what}"),
    }
}

fn f64_at(bytes: &[u8], index: usize) -> f64 {
    let mut raw = [0; 8];
    raw.copy_from_slice(&bytes[index * 8..(index + 1) * 8]);
    f64::from_ne_bytes(raw)
}

fn u32s(bytes: &[u8]) -> impl Iterator<Item = u32> + '_ {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
}

fn triples(bytes: &[u8]) -> impl Iterator<Item = [f64; 3]> + '_ {
    bytes
        .chunks_exact(24)
        .map(|c| [f64_at(c, 0), f64_at(c, 1), f64_at(c, 2)])
}

fn point_from(op: &'static str, buffer: &Buffer) -> Result<Point3> {
    let bytes = buffer.bytes();
    if bytes.len() != 24 {
        return Err(malformed(op, "point"));
    }
    Ok(Point3::new(
        f64_at(&bytes, 0),
        f64_at(&bytes, 1),
        f64_at(&bytes, 2),
    ))
}

/// See `occt_mesh` in `occt_capi.cpp` for the layout.
fn decode_mesh(bytes: &[u8]) -> Option<Mesh> {
    let mut header = u32s(bytes.get(..8)?);
    let vertices = usize::try_from(header.next()?).ok()?;
    let triangles = usize::try_from(header.next()?).ok()?;

    let positions_end = vertices.checked_mul(24)?.checked_add(8)?;
    let normals_end = vertices.checked_mul(24)?.checked_add(positions_end)?;
    let indices_end = triangles.checked_mul(12)?.checked_add(normals_end)?;
    if bytes.len() != indices_end {
        return None;
    }

    let positions: Vec<Point3> = triples(&bytes[8..positions_end])
        .map(|[x, y, z]| Point3::new(x, y, z))
        .collect();
    let normals: Vec<Vec3> = triples(&bytes[positions_end..normals_end])
        .map(|[x, y, z]| Vec3::new(x, y, z))
        .collect();
    let raw_indices: Vec<u32> = u32s(&bytes[normals_end..indices_end]).collect();
    let indices: Vec<[u32; 3]> = raw_indices
        .chunks_exact(3)
        .map(|t| [t[0], t[1], t[2]])
        .collect();
    if raw_indices.iter().any(|&i| i as usize >= vertices) {
        return None;
    }
    Some(Mesh {
        positions,
        normals,
        indices,
    })
}

// --- enums, as ordered in TopAbs_ShapeEnum.hxx, GeomAbs_SurfaceType.hxx and
// GeomAbs_CurveType.hxx of OCCT 8.0.1 ---------------------------------------------

fn kind_from_raw(raw: i32) -> Option<ShapeKind> {
    Some(match raw {
        0 => ShapeKind::Compound,
        1 => ShapeKind::CompSolid,
        2 => ShapeKind::Solid,
        3 => ShapeKind::Shell,
        4 => ShapeKind::Face,
        5 => ShapeKind::Wire,
        6 => ShapeKind::Edge,
        7 => ShapeKind::Vertex,
        _ => return None,
    })
}

fn kind_to_raw(kind: ShapeKind) -> Result<i32> {
    Ok(match kind {
        ShapeKind::Compound => 0,
        ShapeKind::CompSolid => 1,
        ShapeKind::Solid => 2,
        ShapeKind::Shell => 3,
        ShapeKind::Face => 4,
        ShapeKind::Wire => 5,
        ShapeKind::Edge => 6,
        ShapeKind::Vertex => 7,
        other => {
            return Err(Error::InvalidInput(format!(
                "unsupported shape kind {other}"
            )));
        }
    })
}

fn surface_from_raw(raw: i32) -> SurfaceKind {
    match raw {
        0 => SurfaceKind::Plane,
        1 => SurfaceKind::Cylinder,
        2 => SurfaceKind::Cone,
        3 => SurfaceKind::Sphere,
        4 => SurfaceKind::Torus,
        5 | 6 => SurfaceKind::Spline,
        _ => SurfaceKind::Other,
    }
}

fn curve_from_raw(raw: i32) -> CurveKind {
    match raw {
        0 => CurveKind::Line,
        1 => CurveKind::Circle,
        2 => CurveKind::Ellipse,
        5 | 6 => CurveKind::Spline,
        _ => CurveKind::Other,
    }
}

fn require_kind(shape: &CapiShape, expected: ShapeKind) -> Result<()> {
    let found = CapiKernel::kind(shape)?;
    if found == expected {
        Ok(())
    } else {
        Err(Error::WrongKind { expected, found })
    }
}

fn query(op: &'static str, value: f64) -> Result<f64> {
    if value.is_nan() {
        Err(failure(op))
    } else {
        Ok(value)
    }
}

// --- contract -----------------------------------------------------------------------

impl Topology for CapiKernel {
    type Shape = CapiShape;

    fn kind(shape: &CapiShape) -> Result<ShapeKind> {
        // SAFETY: the call takes any handle and reports failure with -1.
        let raw = unsafe { sys::occt_shape_kind(shape.handle) };
        if raw < 0 {
            return Err(failure("kind"));
        }
        kind_from_raw(raw).ok_or_else(|| malformed("kind", "shape kind"))
    }

    fn is_same(a: &CapiShape, b: &CapiShape) -> bool {
        // SAFETY: the call takes any handles and reports failure with -1.
        unsafe { sys::occt_shape_is_same(a.handle, b.handle) == 1 }
    }

    fn hash_code(shape: &CapiShape) -> u32 {
        // SAFETY: the call takes any handle.
        unsafe { sys::occt_shape_hash(shape.handle) }
    }

    fn is_valid(shape: &CapiShape) -> Result<bool> {
        // SAFETY: the call takes any handle and reports failure with -1.
        match unsafe { sys::occt_shape_is_valid(shape.handle) } {
            1 => Ok(true),
            0 => Ok(false),
            _ => Err(failure("is_valid")),
        }
    }

    fn subshapes(shape: &CapiShape, kind: ShapeKind) -> Result<Vec<CapiShape>> {
        let raw = kind_to_raw(kind)?;
        // SAFETY: the call takes any handle and kind, and reports failure with 0.
        let buffer = Buffer::new("subshapes", unsafe {
            sys::occt_shape_subshapes(shape.handle, raw)
        })?;
        Ok(u32s(&buffer.bytes()).map(CapiShape::adopt).collect())
    }
}

impl Primitives for CapiKernel {
    fn make_box(corner: Point3, size: Vec3) -> Result<CapiShape> {
        ensure_ready()?;
        // SAFETY: the module is ready and the call takes only scalars.
        let handle =
            unsafe { sys::occt_make_box(corner.x, corner.y, corner.z, size.x, size.y, size.z) };
        shape_result("make_box", handle)
    }

    fn make_cylinder(axis: Axis, radius: f64, height: f64) -> Result<CapiShape> {
        ensure_ready()?;
        let (o, d) = (axis.origin, axis.direction.vector());
        // SAFETY: the module is ready and the call takes only scalars.
        let handle =
            unsafe { sys::occt_make_cylinder(o.x, o.y, o.z, d.x, d.y, d.z, radius, height) };
        shape_result("make_cylinder", handle)
    }

    fn make_cone(axis: Axis, radius1: f64, radius2: f64, height: f64) -> Result<CapiShape> {
        ensure_ready()?;
        let (o, d) = (axis.origin, axis.direction.vector());
        // SAFETY: the module is ready and the call takes only scalars.
        let handle =
            unsafe { sys::occt_make_cone(o.x, o.y, o.z, d.x, d.y, d.z, radius1, radius2, height) };
        shape_result("make_cone", handle)
    }

    fn make_sphere(center: Point3, radius: f64) -> Result<CapiShape> {
        ensure_ready()?;
        // SAFETY: the module is ready and the call takes only scalars.
        let handle = unsafe { sys::occt_make_sphere(center.x, center.y, center.z, radius) };
        shape_result("make_sphere", handle)
    }

    fn make_polygon_face(points: &[Point3]) -> Result<CapiShape> {
        ensure_ready()?;
        let bytes: Vec<u8> = points
            .iter()
            .flat_map(|p| [p.x, p.y, p.z])
            .flat_map(f64::to_ne_bytes)
            .collect();
        let handle = with_input("make_polygon_face", &bytes, |ptr, _| {
            // SAFETY: `ptr` holds points.len() * 3 doubles, and the module has no
            // alignment requirement on it.
            unsafe { sys::occt_make_polygon_face(ptr, points.len()) }
        })?;
        shape_result("make_polygon_face", handle)
    }
}

impl Sweeps for CapiKernel {
    fn extrude(shape: &CapiShape, by: Vec3) -> Result<CapiShape> {
        // SAFETY: the call takes any handle and scalars.
        let handle = unsafe { sys::occt_extrude(shape.handle, by.x, by.y, by.z) };
        shape_result("extrude", handle)
    }

    fn revolve(shape: &CapiShape, axis: Axis, angle: Angle) -> Result<CapiShape> {
        let (o, d) = (axis.origin, axis.direction.vector());
        // SAFETY: the call takes any handle and scalars.
        let handle = unsafe {
            sys::occt_revolve(
                shape.handle,
                o.x,
                o.y,
                o.z,
                d.x,
                d.y,
                d.z,
                angle.as_radians(),
            )
        };
        shape_result("revolve", handle)
    }
}

impl Booleans for CapiKernel {
    fn fuse(a: &CapiShape, b: &CapiShape) -> Result<CapiShape> {
        // SAFETY: the call takes any handles.
        shape_result("fuse", unsafe { sys::occt_fuse(a.handle, b.handle) })
    }

    fn cut(a: &CapiShape, b: &CapiShape) -> Result<CapiShape> {
        // SAFETY: the call takes any handles.
        shape_result("cut", unsafe { sys::occt_cut(a.handle, b.handle) })
    }

    fn common(a: &CapiShape, b: &CapiShape) -> Result<CapiShape> {
        // SAFETY: the call takes any handles.
        shape_result("common", unsafe { sys::occt_common(a.handle, b.handle) })
    }
}

fn edge_handles(edges: &[&CapiShape]) -> Result<Vec<u8>> {
    let mut bytes = Vec::with_capacity(edges.len() * 4);
    for edge in edges {
        require_kind(edge, ShapeKind::Edge)?;
        bytes.extend_from_slice(&edge.handle.to_ne_bytes());
    }
    Ok(bytes)
}

impl Blends for CapiKernel {
    fn fillet(shape: &CapiShape, edges: &[&CapiShape], radius: f64) -> Result<CapiShape> {
        let bytes = edge_handles(edges)?;
        let handle = with_input("fillet", &bytes, |ptr, _| {
            // SAFETY: `ptr` holds edges.len() handles, and the module has no alignment
            // requirement on it.
            unsafe { sys::occt_fillet(shape.handle, ptr, edges.len(), radius) }
        })?;
        shape_result("fillet", handle)
    }

    fn chamfer(shape: &CapiShape, edges: &[&CapiShape], distance: f64) -> Result<CapiShape> {
        let bytes = edge_handles(edges)?;
        let handle = with_input("chamfer", &bytes, |ptr, _| {
            // SAFETY: `ptr` holds edges.len() handles, and the module has no alignment
            // requirement on it.
            unsafe { sys::occt_chamfer(shape.handle, ptr, edges.len(), distance) }
        })?;
        shape_result("chamfer", handle)
    }
}

impl Transforms for CapiKernel {
    fn translate(shape: &CapiShape, by: Vec3) -> Result<CapiShape> {
        // SAFETY: the call takes any handle and scalars.
        shape_result("translate", unsafe {
            sys::occt_translate(shape.handle, by.x, by.y, by.z)
        })
    }

    fn rotate(shape: &CapiShape, axis: Axis, angle: Angle) -> Result<CapiShape> {
        let (o, d) = (axis.origin, axis.direction.vector());
        // SAFETY: the call takes any handle and scalars.
        let handle = unsafe {
            sys::occt_rotate(
                shape.handle,
                o.x,
                o.y,
                o.z,
                d.x,
                d.y,
                d.z,
                angle.as_radians(),
            )
        };
        shape_result("rotate", handle)
    }

    fn scale(shape: &CapiShape, center: Point3, factor: f64) -> Result<CapiShape> {
        // SAFETY: the call takes any handle and scalars.
        let handle = unsafe { sys::occt_scale(shape.handle, center.x, center.y, center.z, factor) };
        shape_result("scale", handle)
    }
}

impl Properties for CapiKernel {
    fn volume(shape: &CapiShape) -> Result<f64> {
        // SAFETY: the call takes any handle and reports failure with NaN.
        query("volume", unsafe { sys::occt_volume(shape.handle) })
    }

    fn area(shape: &CapiShape) -> Result<f64> {
        // SAFETY: the call takes any handle and reports failure with NaN.
        query("area", unsafe { sys::occt_area(shape.handle) })
    }

    fn length(shape: &CapiShape) -> Result<f64> {
        // SAFETY: the call takes any handle and reports failure with NaN.
        query("length", unsafe { sys::occt_length(shape.handle) })
    }

    fn center_of_mass(shape: &CapiShape) -> Result<Point3> {
        const OP: &str = "center_of_mass";
        // SAFETY: the call takes any handle and reports failure with 0.
        let buffer = Buffer::new(OP, unsafe { sys::occt_center_of_mass(shape.handle) })?;
        point_from(OP, &buffer)
    }

    fn bounding_box(shape: &CapiShape) -> Result<Aabb> {
        const OP: &str = "bounding_box";
        // SAFETY: the call takes any handle and reports failure with 0.
        let buffer = Buffer::new(OP, unsafe { sys::occt_bounding_box(shape.handle) })?;
        let bytes = buffer.bytes();
        if bytes.len() != 48 {
            return Err(malformed(OP, "bounding box"));
        }
        let v = |i| f64_at(&bytes, i);
        Ok(Aabb {
            min: Point3::new(v(0), v(1), v(2)),
            max: Point3::new(v(3), v(4), v(5)),
        })
    }

    fn vertex_point(vertex: &CapiShape) -> Result<Point3> {
        const OP: &str = "vertex_point";
        require_kind(vertex, ShapeKind::Vertex)?;
        // SAFETY: the call takes any handle and reports failure with 0.
        let buffer = Buffer::new(OP, unsafe { sys::occt_vertex_point(vertex.handle) })?;
        point_from(OP, &buffer)
    }

    fn surface_kind(face: &CapiShape) -> Result<SurfaceKind> {
        require_kind(face, ShapeKind::Face)?;
        // SAFETY: the call takes any handle and reports failure with -1.
        let raw = unsafe { sys::occt_face_surface_kind(face.handle) };
        if raw < 0 {
            Err(failure("surface_kind"))
        } else {
            Ok(surface_from_raw(raw))
        }
    }

    fn curve_kind(edge: &CapiShape) -> Result<CurveKind> {
        require_kind(edge, ShapeKind::Edge)?;
        // SAFETY: the call takes any handle and reports failure with -1.
        let raw = unsafe { sys::occt_edge_curve_kind(edge.handle) };
        if raw < 0 {
            Err(failure("curve_kind"))
        } else {
            Ok(curve_from_raw(raw))
        }
    }
}

impl Tessellation for CapiKernel {
    fn mesh(shape: &CapiShape, options: &MeshOptions) -> Result<Mesh> {
        const OP: &str = "mesh";
        let (linear, relative) = match options.deflection {
            Deflection::Absolute(distance) => (distance, 0),
            Deflection::Relative(fraction) => (fraction, 1),
        };
        let angular = options.angular_deflection.as_radians();
        // SAFETY: the call takes any handle and scalars, and reports failure with 0.
        let handle = unsafe { sys::occt_mesh(shape.handle, linear, angular, relative) };
        let buffer = Buffer::new(OP, handle)?;
        decode_mesh(&buffer.bytes()).ok_or_else(|| malformed(OP, "mesh"))
    }
}

impl StepIo for CapiKernel {
    fn write_step(shape: &CapiShape) -> Result<Vec<u8>> {
        // SAFETY: the call takes any handle and reports failure with 0.
        let buffer = Buffer::new("write_step", unsafe { sys::occt_step_write(shape.handle) })?;
        Ok(buffer.bytes())
    }

    fn read_step(data: &[u8]) -> Result<CapiShape> {
        const OP: &str = "read_step";
        ensure_ready()?;
        let handle = with_input(OP, data, |ptr, len| {
            // SAFETY: `ptr..ptr + len` holds the input for the duration of the call.
            unsafe { sys::occt_step_read(ptr, len) }
        })?;
        shape_result(OP, handle)
    }
}

impl Kernel for CapiKernel {
    const NAME: &'static str = "occt";

    const NOTICE: &'static str = "\
This software uses Open CASCADE Technology, licensed under the GNU Lesser General Public \
License version 2.1 with the Open CASCADE exception version 1.0. \
Source code: https://github.com/Open-Cascade-SAS/OCCT";

    fn version() -> Result<String> {
        ensure_ready()?;
        // SAFETY: the module is ready, and the pair names a string constant in the module.
        let bytes = unsafe {
            sys::copy_out(
                sys::occt_library_version_ptr(),
                sys::occt_library_version_len(),
            )
        };
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    fn ready() -> Result<()> {
        ensure_ready()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encoded(vertices: u32, triangles: u32, indices: &[u32]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&vertices.to_ne_bytes());
        out.extend_from_slice(&triangles.to_ne_bytes());
        for i in 0..vertices * 6 {
            out.extend_from_slice(&f64::from(i).to_ne_bytes());
        }
        for i in indices {
            out.extend_from_slice(&i.to_ne_bytes());
        }
        out
    }

    #[test]
    fn decodes_a_mesh() {
        let mesh = decode_mesh(&encoded(3, 1, &[0, 1, 2])).unwrap();
        assert_eq!(mesh.positions[1], Point3::new(3.0, 4.0, 5.0));
        assert_eq!(mesh.normals[0], Vec3::new(9.0, 10.0, 11.0));
        assert_eq!(mesh.indices, [[0, 1, 2]]);
    }

    #[test]
    fn rejects_malformed_meshes() {
        assert!(decode_mesh(&[]).is_none());
        assert!(decode_mesh(&encoded(3, 2, &[0, 1, 2])).is_none());
        assert!(decode_mesh(&encoded(3, 1, &[0, 1, 3])).is_none());
        let mut huge = encoded(0, 0, &[]);
        huge[..4].copy_from_slice(&u32::MAX.to_ne_bytes());
        assert!(decode_mesh(&huge).is_none());
    }
}
