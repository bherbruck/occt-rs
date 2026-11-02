//! The features domain of the contract: sweeps with history, lofts and local edits.

use occt_capi_sys as sys;
use occt_kernel::{
    AdvancedSweeps, Angle, Axis, Dir3, Error, Features, LoftOptions, Modeled, Plane, Result,
    ShapeKind, SweepOptions, SweepOrientation, Topology, Transform, Vec3,
};

use crate::wire::{Writer, decode_modeled, handles};
use crate::{Buffer, CapiKernel, CapiShape, failure, require_kind, shape_result, with_input};

fn modeled(op: &'static str, handle: u32) -> Result<Modeled<CapiShape>> {
    decode_modeled(op, &Buffer::new(op, handle)?)
}

fn require_faces(faces: &[&CapiShape]) -> Result<()> {
    faces
        .iter()
        .try_for_each(|face| require_kind(face, ShapeKind::Face))
}

impl CapiKernel {
    /// Whether two shapes are the same stored topology, wherever each is placed.
    ///
    /// This makes [`Features::placed`] testable: nothing in the contract can tell a shared
    /// shape from an identical copy.
    ///
    /// # Errors
    /// [`Error::Operation`] if a shape is stale.
    #[doc(hidden)]
    pub fn shares_geometry(a: &CapiShape, b: &CapiShape) -> Result<bool> {
        // SAFETY: the call takes any handles and reports failure with -1.
        match unsafe { sys::occt_shape_shares_geometry(a.handle, b.handle) } {
            1 => Ok(true),
            0 => Ok(false),
            _ => Err(failure("shares_geometry")),
        }
    }
}

impl AdvancedSweeps for CapiKernel {
    fn extrude_modeled(shape: &CapiShape, by: Vec3) -> Result<Modeled<CapiShape>> {
        // SAFETY: the call takes any handle and scalars, and reports failure with 0.
        let handle = unsafe { sys::occt_extrude_modeled(shape.handle, by.x, by.y, by.z) };
        modeled("extrude_modeled", handle)
    }

    fn revolve_modeled(shape: &CapiShape, axis: Axis, angle: Angle) -> Result<Modeled<CapiShape>> {
        let (o, d) = (axis.origin, axis.direction.vector());
        // SAFETY: the call takes any handle and scalars, and reports failure with 0.
        let handle = unsafe {
            sys::occt_revolve_modeled(
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
        modeled("revolve_modeled", handle)
    }

    fn extrude_tapered(face: &CapiShape, by: Vec3, taper: Angle) -> Result<CapiShape> {
        require_kind(face, ShapeKind::Face)?;
        // SAFETY: the call takes any handle and scalars, and reports failure with 0.
        let handle =
            unsafe { sys::occt_extrude_tapered(face.handle, by.x, by.y, by.z, taper.as_radians()) };
        shape_result("extrude_tapered", handle)
    }

    fn sweep(
        profile: &CapiShape,
        path: &CapiShape,
        options: &SweepOptions,
    ) -> Result<Modeled<CapiShape>> {
        let found = CapiKernel::kind(profile)?;
        if !matches!(found, ShapeKind::Wire | ShapeKind::Edge | ShapeKind::Face) {
            return Err(Error::WrongKind {
                expected: ShapeKind::Wire,
                found,
            });
        }
        let found = CapiKernel::kind(path)?;
        if !matches!(found, ShapeKind::Wire | ShapeKind::Edge) {
            return Err(Error::WrongKind {
                expected: ShapeKind::Wire,
                found,
            });
        }
        let orientation = match options.orientation {
            SweepOrientation::Corrected => 0,
            SweepOrientation::Frenet => 1,
            SweepOrientation::Fixed => 2,
            other => {
                return Err(Error::InvalidInput(format!(
                    "unsupported sweep orientation {other:?}"
                )));
            }
        };
        let (has_binormal, b) = match options.binormal {
            Some(binormal) => (1, binormal.vector()),
            None => (0, Vec3::ZERO),
        };
        // SAFETY: the call takes any handles and scalars, and reports failure with 0.
        let handle = unsafe {
            sys::occt_sweep(
                profile.handle,
                path.handle,
                orientation,
                has_binormal,
                b.x,
                b.y,
                b.z,
                i32::from(options.solid),
            )
        };
        modeled("sweep", handle)
    }

    fn loft(sections: &[&CapiShape], options: &LoftOptions) -> Result<CapiShape> {
        const OP: &str = "loft";
        ensure_loft_kinds(sections)?;
        let bytes = handles(sections);
        let handle = with_input(OP, &bytes, |ptr, _| {
            // SAFETY: `ptr` holds sections.len() handles, with no alignment requirement.
            unsafe {
                sys::occt_loft(
                    ptr,
                    sections.len(),
                    i32::from(options.solid),
                    i32::from(options.ruled),
                    options.tolerance,
                )
            }
        })?;
        shape_result(OP, handle)
    }
}

fn ensure_loft_kinds(sections: &[&CapiShape]) -> Result<()> {
    // An empty list holds no shape to prove readiness, and the module would be called.
    crate::ensure_ready()?;
    for section in sections {
        let found = CapiKernel::kind(section)?;
        if !matches!(
            found,
            ShapeKind::Wire | ShapeKind::Edge | ShapeKind::Face | ShapeKind::Vertex
        ) {
            return Err(Error::WrongKind {
                expected: ShapeKind::Wire,
                found,
            });
        }
    }
    Ok(())
}

impl Features for CapiKernel {
    fn shell(
        solid: &CapiShape,
        open_faces: &[&CapiShape],
        thickness: f64,
    ) -> Result<Modeled<CapiShape>> {
        const OP: &str = "shell";
        require_kind(solid, ShapeKind::Solid)?;
        require_faces(open_faces)?;
        let bytes = handles(open_faces);
        let handle = with_input(OP, &bytes, |ptr, _| {
            // SAFETY: `ptr` holds open_faces.len() handles, with no alignment requirement.
            unsafe { sys::occt_shell(solid.handle, ptr, open_faces.len(), thickness) }
        })?;
        modeled(OP, handle)
    }

    fn offset(shape: &CapiShape, distance: f64) -> Result<CapiShape> {
        // SAFETY: the call takes any handle and scalars, and reports failure with 0.
        shape_result("offset", unsafe {
            sys::occt_offset(shape.handle, distance)
        })
    }

    fn thicken(sheet: &CapiShape, thickness: f64) -> Result<CapiShape> {
        let found = CapiKernel::kind(sheet)?;
        if !matches!(found, ShapeKind::Face | ShapeKind::Shell) {
            return Err(Error::WrongKind {
                expected: ShapeKind::Face,
                found,
            });
        }
        // SAFETY: the call takes any handle and scalars, and reports failure with 0.
        shape_result("thicken", unsafe {
            sys::occt_thicken(sheet.handle, thickness)
        })
    }

    fn draft(
        solid: &CapiShape,
        faces: &[&CapiShape],
        pull: Dir3,
        neutral: Plane,
        angle: Angle,
    ) -> Result<Modeled<CapiShape>> {
        const OP: &str = "draft";
        require_kind(solid, ShapeKind::Solid)?;
        require_faces(faces)?;
        let bytes = handles(faces);
        let (p, o, n) = (pull.vector(), neutral.origin, neutral.normal.vector());
        let handle = with_input(OP, &bytes, |ptr, _| {
            // SAFETY: `ptr` holds faces.len() handles, with no alignment requirement.
            unsafe {
                sys::occt_draft(
                    solid.handle,
                    ptr,
                    faces.len(),
                    p.x,
                    p.y,
                    p.z,
                    o.x,
                    o.y,
                    o.z,
                    n.x,
                    n.y,
                    n.z,
                    angle.as_radians(),
                )
            }
        })?;
        modeled(OP, handle)
    }

    fn remove_faces(solid: &CapiShape, faces: &[&CapiShape]) -> Result<Modeled<CapiShape>> {
        const OP: &str = "remove_faces";
        require_kind(solid, ShapeKind::Solid)?;
        require_faces(faces)?;
        let bytes = handles(faces);
        let handle = with_input(OP, &bytes, |ptr, _| {
            // SAFETY: `ptr` holds faces.len() handles, with no alignment requirement.
            unsafe { sys::occt_remove_faces(solid.handle, ptr, faces.len()) }
        })?;
        modeled(OP, handle)
    }

    fn unify(shape: &CapiShape) -> Result<Modeled<CapiShape>> {
        // SAFETY: the call takes any handle and reports failure with 0.
        modeled("unify", unsafe { sys::occt_unify(shape.handle) })
    }

    fn heal(shape: &CapiShape, tolerance: f64) -> Result<CapiShape> {
        // SAFETY: the call takes any handle and scalars, and reports failure with 0.
        shape_result("heal", unsafe { sys::occt_heal(shape.handle, tolerance) })
    }

    fn transformed(shape: &CapiShape, transform: &Transform) -> Result<CapiShape> {
        const OP: &str = "transformed";
        let mut rows = Writer::default();
        rows.transform(transform);
        let handle = with_input(OP, &rows.0, |ptr, len| {
            // SAFETY: `ptr..ptr + len` holds the rows, with no alignment requirement.
            unsafe { sys::occt_transformed(shape.handle, ptr, len) }
        })?;
        shape_result(OP, handle)
    }

    fn placed(shape: &CapiShape, transform: &Transform) -> Result<CapiShape> {
        const OP: &str = "placed";
        let mut rows = Writer::default();
        rows.transform(transform);
        let handle = with_input(OP, &rows.0, |ptr, len| {
            // SAFETY: `ptr..ptr + len` holds the rows, with no alignment requirement.
            unsafe { sys::occt_placed(shape.handle, ptr, len) }
        })?;
        shape_result(OP, handle)
    }

    fn reversed(shape: &CapiShape) -> Result<CapiShape> {
        // SAFETY: the call takes any handle and reports failure with 0.
        shape_result("reversed", unsafe { sys::occt_reversed(shape.handle) })
    }
}
