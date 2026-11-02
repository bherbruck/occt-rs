//! The exchange domain of the contract.

use occt_capi_sys as sys;
use occt_kernel::{
    CancelFlag, Color, Document, DocumentNode, Error, Exchange, NodeContent, Result, StepSchema,
    StepWriteOptions, Transform,
};

use crate::wire::{Reader, Writer, cancel_address, cancelled_or, handles};
use crate::{Buffer, CapiShape, ensure_ready, shape_result, with_input};

/// The deepest product structure the module accepts.
const MAX_DEPTH: usize = 256;

/// Reads what `append_node` in `exchange.cpp` writes.
fn decode_node(r: &mut Reader<'_>, depth: usize) -> Result<DocumentNode<CapiShape>> {
    if depth > MAX_DEPTH + 1 {
        return Err(Error::Operation {
            op: r.op,
            message: "the product structure is too deep".to_owned(),
        });
    }
    let name = r.string()?;
    let color = if r.u32()? == 0 {
        None
    } else {
        // Colours cross as f64 only to keep one number type on the wire.
        #[allow(clippy::cast_possible_truncation)]
        let channel = |value: f64| value as f32;
        Some(Color {
            r: channel(r.f64()?),
            g: channel(r.f64()?),
            b: channel(r.f64()?),
        })
    };
    let mut placement = Transform::IDENTITY;
    for row in &mut placement.rows {
        for cell in row {
            *cell = r.f64()?;
        }
    }
    let content = if r.u32()? == 0 {
        NodeContent::Part(r.shape()?)
    } else {
        let count = r.u32()?;
        let children = (0..count)
            .map(|_| decode_node(r, depth + 1))
            .collect::<Result<_>>()?;
        NodeContent::Assembly(children)
    };
    Ok(DocumentNode {
        name: (!name.is_empty()).then_some(name),
        color,
        placement,
        content,
    })
}

/// Writes what `read_node` in `exchange.cpp` reads.
fn encode_node(w: &mut Writer, node: &DocumentNode<CapiShape>, depth: usize) -> Result<()> {
    if depth > MAX_DEPTH {
        return Err(Error::InvalidInput(format!(
            "the product structure is deeper than {MAX_DEPTH} levels"
        )));
    }
    let name = node.name.as_deref().unwrap_or_default();
    w.count(name.len()).bytes(name.as_bytes());
    match node.color {
        Some(Color { r, g, b }) => {
            w.u32(1)
                .f64(f64::from(r))
                .f64(f64::from(g))
                .f64(f64::from(b));
        }
        None => {
            w.u32(0);
        }
    }
    w.transform(&node.placement);
    match &node.content {
        NodeContent::Part(shape) => {
            w.u32(0).shape(shape);
        }
        NodeContent::Assembly(children) => {
            w.u32(1).count(children.len());
            for child in children {
                encode_node(w, child, depth + 1)?;
            }
        }
    }
    Ok(())
}

impl Exchange for super::CapiKernel {
    fn write_native(shape: &CapiShape) -> Result<Vec<u8>> {
        // SAFETY: the call takes any handle and a flag, and reports failure with 0.
        let handle = unsafe { sys::occt_native_write(shape.handle, 0) };
        Ok(Buffer::new("write_native", handle)?.bytes())
    }

    fn read_native(data: &[u8]) -> Result<CapiShape> {
        const OP: &str = "read_native";
        ensure_ready()?;
        let handle = with_input(OP, data, |ptr, len| {
            // SAFETY: `ptr..ptr + len` holds the input for the duration of the call.
            unsafe { sys::occt_native_read(ptr, len) }
        })?;
        shape_result(OP, handle)
    }

    fn read_step_document(data: &[u8], cancel: Option<&CancelFlag>) -> Result<Document<CapiShape>> {
        if cancel.is_some_and(CancelFlag::is_cancelled) {
            return Err(occt_kernel::Error::Cancelled);
        }
        const OP: &str = "read_step_document";
        ensure_ready()?;
        let flag = cancel_address(cancel);
        let handle = with_input(OP, data, |ptr, len| {
            // SAFETY: `ptr..ptr + len` holds the input for the duration of the call, and
            // `flag` is 0 or the address of the flag's byte, which `cancel` keeps alive.
            unsafe { sys::occt_step_document_read(ptr, len, flag) }
        })?;
        let buffer = Buffer::new(OP, handle).map_err(|e| cancelled_or(cancel, e))?;
        let bytes = buffer.bytes();
        let mut r = Reader {
            op: OP,
            bytes: &bytes,
        };
        let units = r.f64()?;
        let count = r.u32()?;
        let roots = (0..count)
            .map(|_| decode_node(&mut r, 0))
            .collect::<Result<_>>()?;
        Ok(Document {
            roots,
            file_units_per_metre: (!units.is_nan()).then_some(units),
        })
    }

    fn write_step_document(
        document: &Document<CapiShape>,
        options: &StepWriteOptions,
    ) -> Result<Vec<u8>> {
        const OP: &str = "write_step_document";
        ensure_ready()?;
        let schema = match options.schema {
            StepSchema::Ap203 => 0,
            StepSchema::Ap214 => 1,
            StepSchema::Ap242 => 2,
            other => {
                return Err(Error::InvalidInput(format!(
                    "unsupported STEP schema {other:?}"
                )));
            }
        };
        let mut w = Writer::default();
        w.count(document.roots.len());
        for node in &document.roots {
            encode_node(&mut w, node, 1)?;
        }
        let units = document.file_units_per_metre.unwrap_or(f64::NAN);
        let cancel = options.cancel.as_ref();
        if cancel.is_some_and(CancelFlag::is_cancelled) {
            return Err(occt_kernel::Error::Cancelled);
        }
        let flag = cancel_address(cancel);
        let handle = with_input(OP, &w.0, |ptr, len| {
            // SAFETY: `ptr..ptr + len` holds the encoded nodes for the duration of the call,
            // and `flag` is 0 or the address of the flag's byte, which `options` keeps alive.
            unsafe { sys::occt_step_document_write(ptr, len, schema, units, flag) }
        })?;
        let buffer = Buffer::new(OP, handle).map_err(|e| cancelled_or(cancel, e))?;
        Ok(buffer.bytes())
    }

    fn read_iges(data: &[u8]) -> Result<CapiShape> {
        const OP: &str = "read_iges";
        ensure_ready()?;
        let handle = with_input(OP, data, |ptr, len| {
            // SAFETY: `ptr..ptr + len` holds the input for the duration of the call.
            unsafe { sys::occt_iges_read(ptr, len) }
        })?;
        shape_result(OP, handle)
    }

    fn write_iges(shapes: &[&CapiShape]) -> Result<Vec<u8>> {
        const OP: &str = "write_iges";
        ensure_ready()?;
        let bytes = handles(shapes);
        let handle = with_input(OP, &bytes, |ptr, _| {
            // SAFETY: `ptr` holds shapes.len() handles, and the module has no alignment
            // requirement on it.
            unsafe { sys::occt_iges_write(ptr, shapes.len()) }
        })?;
        Ok(Buffer::new(OP, handle)?.bytes())
    }
}
