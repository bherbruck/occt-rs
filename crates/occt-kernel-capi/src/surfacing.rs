//! [`Surfacing`] through the module's surfacing entry points.

use occt_capi_sys as sys;
use occt_kernel::{
    Angle, BlendContinuity, BlendSide, ContinuityReport, FaceSides, FillEdge, Point3, Result,
    ShapeKind, Surfacing,
};

use crate::geometry::fill_input;
use crate::wire::{Reader, Writer};
use crate::{Buffer, CapiKernel, CapiShape, ensure_ready, require_kind, shape_result, with_input};

impl Surfacing for CapiKernel {
    fn fill_boundary_tangent(
        edges: &[FillEdge<'_, CapiShape>],
        through: &[Point3],
    ) -> Result<CapiShape> {
        const OP: &str = "fill_boundary_tangent";
        for side in edges {
            require_kind(side.edge, ShapeKind::Edge)?;
            if let Some(face) = side.tangent_to {
                require_kind(face, ShapeKind::Face)?;
            }
        }
        ensure_ready()?;
        let plain: Vec<_> = edges.iter().map(|side| side.edge).collect();
        let supports: Vec<_> = edges.iter().map(|side| side.tangent_to).collect();
        let w = fill_input(&plain, &supports, through);
        let handle = with_input(OP, &w.0, |ptr, len| {
            // SAFETY: `ptr..ptr + len` holds the input for the duration of the call, and the
            // module has no alignment requirement on it.
            unsafe { sys::occt_fill_boundary(ptr, len) }
        })?;
        shape_result(OP, handle)
    }

    fn blend_faces(
        a: BlendSide<'_, CapiShape>,
        b: BlendSide<'_, CapiShape>,
        continuity: BlendContinuity,
        tension: f64,
    ) -> Result<CapiShape> {
        const OP: &str = "blend_faces";
        for side in [&a, &b] {
            require_kind(side.edge, ShapeKind::Edge)?;
            require_kind(side.face, ShapeKind::Face)?;
        }
        ensure_ready()?;
        let mut w = Writer::default();
        w.shape(a.edge).shape(a.face).shape(b.edge).shape(b.face);
        w.u32(match continuity {
            BlendContinuity::G1 => 1,
            BlendContinuity::G2 => 2,
        });
        w.f64(tension);
        let handle = with_input(OP, &w.0, |ptr, len| {
            // SAFETY: `ptr..ptr + len` holds the input for the duration of the call, and the
            // module has no alignment requirement on it.
            unsafe { sys::occt_blend_faces(ptr, len) }
        })?;
        shape_result(OP, handle)
    }

    fn measure_continuity(
        a: &CapiShape,
        b: &CapiShape,
        edge: &CapiShape,
    ) -> Result<ContinuityReport> {
        const OP: &str = "measure_continuity";
        require_kind(a, ShapeKind::Face)?;
        require_kind(b, ShapeKind::Face)?;
        require_kind(edge, ShapeKind::Edge)?;
        // SAFETY: the call takes any handles and reports failure with 0.
        let handle = unsafe { sys::occt_measure_continuity(a.handle, b.handle, edge.handle) };
        let buffer = Buffer::new(OP, handle)?;
        let bytes = buffer.bytes();
        let mut r = Reader {
            op: OP,
            bytes: &bytes,
        };
        Ok(ContinuityReport {
            gap: r.f64()?,
            angle: Angle::radians(r.f64()?),
            curvature_gap: r.f64()?,
        })
    }

    fn extend_face(face: &CapiShape, length: f64, sides: FaceSides) -> Result<CapiShape> {
        require_kind(face, ShapeKind::Face)?;
        let bits = u32::from(sides.u_min)
            | u32::from(sides.u_max) << 1
            | u32::from(sides.v_min) << 2
            | u32::from(sides.v_max) << 3;
        // SAFETY: the call takes any handle and scalars, and reports failure with 0.
        let handle = unsafe { sys::occt_extend_face(face.handle, length, bits) };
        shape_result("extend_face", handle)
    }
}
