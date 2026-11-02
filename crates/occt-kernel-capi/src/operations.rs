//! The operations domain of the contract.

use occt_capi_sys as sys;
use occt_kernel::{
    AdvancedBlends, BooleanOp, BooleanOps, BooleanOptions, ChamferSize, Error, Modeled, Plane,
    RadiusLaw, Result,
};

use crate::wire::{Writer, cancel_address, cancelled_or, decode_modeled};
use crate::{Buffer, CapiKernel, CapiShape, shape_result, with_input};

const UNIFY: u32 = 1;
const PARALLEL: u32 = 2;

/// Two lists of shapes, as the module's general operations read them.
fn argument_lists(arguments: &[&CapiShape], tools: &[&CapiShape]) -> Vec<u8> {
    let mut w = Writer::default();
    for list in [arguments, tools] {
        w.count(list.len());
        for shape in list {
            w.shape(shape);
        }
    }
    w.0
}

fn flags(options: &BooleanOptions) -> u32 {
    let mut flags = 0;
    if options.unify {
        flags |= UNIFY;
    }
    if options.parallel {
        flags |= PARALLEL;
    }
    flags
}

/// Runs a cancellable general operation, which fills a modeled buffer.
///
/// The flag is also checked here, because on the web the module can't read it.
fn general(
    op: &'static str,
    options: &BooleanOptions,
    call: impl FnOnce(f64, u32, usize) -> Result<u32>,
) -> Result<Modeled<CapiShape>> {
    let cancel = options.cancel.as_ref();
    if cancel.is_some_and(occt_kernel::CancelFlag::is_cancelled) {
        return Err(Error::Cancelled);
    }
    let handle = call(
        options.fuzzy.unwrap_or(0.0),
        flags(options),
        cancel_address(cancel),
    )?;
    let buffer = Buffer::new(op, handle).map_err(|e| cancelled_or(cancel, e))?;
    decode_modeled(op, &buffer)
}

impl BooleanOps for CapiKernel {
    fn boolean(
        op: BooleanOp,
        arguments: &[&CapiShape],
        tools: &[&CapiShape],
        options: &BooleanOptions,
    ) -> Result<Modeled<CapiShape>> {
        const NAME: &str = "boolean";
        crate::ensure_ready()?;
        let code = match op {
            BooleanOp::Fuse => 0,
            BooleanOp::Cut => 1,
            BooleanOp::Common => 2,
        };
        let input = argument_lists(arguments, tools);
        general(NAME, options, |fuzzy, flags, cancel| {
            with_input(NAME, &input, |ptr, len| {
                // SAFETY: the range is readable for the call and holds what the entry point
                // decodes, bounds-checked there. `cancel` is 0 or the address of the flag's
                // byte, which `options` keeps alive for the call.
                unsafe { sys::occt_boolean_general(code, ptr, len, fuzzy, flags, cancel) }
            })
        })
    }

    fn section(a: &CapiShape, b: &CapiShape) -> Result<CapiShape> {
        // SAFETY: the call takes any handles and reports failure with 0.
        shape_result("section", unsafe { sys::occt_section(a.handle, b.handle) })
    }

    fn section_plane(shape: &CapiShape, plane: Plane) -> Result<CapiShape> {
        let (o, n) = (plane.origin, plane.normal.vector());
        // SAFETY: the call takes any handle and plain numbers, and reports failure with 0.
        let handle = unsafe { sys::occt_section_plane(shape.handle, o.x, o.y, o.z, n.x, n.y, n.z) };
        shape_result("section_plane", handle)
    }

    fn split(
        arguments: &[&CapiShape],
        tools: &[&CapiShape],
        options: &BooleanOptions,
    ) -> Result<Modeled<CapiShape>> {
        const NAME: &str = "split";
        crate::ensure_ready()?;
        let input = argument_lists(arguments, tools);
        general(NAME, options, |fuzzy, flags, cancel| {
            with_input(NAME, &input, |ptr, len| {
                // SAFETY: as for `boolean`.
                unsafe { sys::occt_split(ptr, len, fuzzy, flags, cancel) }
            })
        })
    }
}

impl AdvancedBlends for CapiKernel {
    fn fillet_edges(
        shape: &CapiShape,
        edges: &[(&CapiShape, RadiusLaw)],
    ) -> Result<Modeled<CapiShape>> {
        const NAME: &str = "fillet_edges";
        let mut w = Writer::default();
        w.count(edges.len());
        for (edge, law) in edges {
            w.shape(edge);
            match law {
                RadiusLaw::Constant(radius) => {
                    w.u32(0).f64(*radius);
                }
                RadiusLaw::Linear { start, end } => {
                    w.u32(1).f64(*start).f64(*end);
                }
                RadiusLaw::Points(points) => {
                    w.u32(2).count(points.len());
                    for (fraction, radius) in points {
                        w.f64(*fraction).f64(*radius);
                    }
                }
            }
        }
        let handle = with_input(NAME, &w.0, |ptr, len| {
            // SAFETY: the range is readable for the call and decoded with bounds checks.
            unsafe { sys::occt_fillet_edges(shape.handle, ptr, len) }
        })?;
        decode_modeled(NAME, &Buffer::new(NAME, handle)?)
    }

    fn chamfer_edges(
        shape: &CapiShape,
        edges: &[(&CapiShape, ChamferSize, Option<&CapiShape>)],
    ) -> Result<Modeled<CapiShape>> {
        const NAME: &str = "chamfer_edges";
        let mut w = Writer::default();
        w.count(edges.len());
        for (edge, size, face) in edges {
            w.shape(edge);
            match face {
                Some(face) => w.shape(face),
                None => w.u32(0),
            };
            let (kind, a, b) = match *size {
                ChamferSize::Equal(d) => (0, d, 0.0),
                ChamferSize::TwoDistances(d1, d2) => (1, d1, d2),
                ChamferSize::DistanceAngle(d, angle) => (2, d, angle.as_radians()),
            };
            w.u32(kind).f64(a).f64(b);
        }
        let handle = with_input(NAME, &w.0, |ptr, len| {
            // SAFETY: the range is readable for the call and decoded with bounds checks.
            unsafe { sys::occt_chamfer_edges(shape.handle, ptr, len) }
        })?;
        decode_modeled(NAME, &Buffer::new(NAME, handle)?)
    }
}
