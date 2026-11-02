//! The byte formats shared with `capi/src/internal.hpp`.

#![allow(dead_code)]

use occt_kernel::{CancelFlag, Error, History, Modeled, Point3, Result, Transform, Vec3};

use crate::{Buffer, CapiShape};

/// Builds an input range for the module. Native byte order; no alignment assumed.
#[derive(Default)]
pub(crate) struct Writer(pub(crate) Vec<u8>);

impl Writer {
    pub(crate) fn f64(&mut self, v: f64) -> &mut Self {
        self.0.extend_from_slice(&v.to_ne_bytes());
        self
    }
    pub(crate) fn u32(&mut self, v: u32) -> &mut Self {
        self.0.extend_from_slice(&v.to_ne_bytes());
        self
    }
    pub(crate) fn count(&mut self, n: usize) -> &mut Self {
        self.u32(u32::try_from(n).expect("input counts fit in u32"))
    }
    pub(crate) fn bytes(&mut self, v: &[u8]) -> &mut Self {
        self.0.extend_from_slice(v);
        self
    }
    pub(crate) fn point(&mut self, p: Point3) -> &mut Self {
        self.f64(p.x).f64(p.y).f64(p.z)
    }
    pub(crate) fn vec3(&mut self, v: Vec3) -> &mut Self {
        self.f64(v.x).f64(v.y).f64(v.z)
    }
    pub(crate) fn shape(&mut self, s: &CapiShape) -> &mut Self {
        self.u32(s.handle)
    }
    pub(crate) fn transform(&mut self, t: &Transform) -> &mut Self {
        for row in t.rows {
            for v in row {
                self.f64(v);
            }
        }
        self
    }
}

/// The handles of shapes as a tightly packed input range.
pub(crate) fn handles(shapes: &[&CapiShape]) -> Vec<u8> {
    let mut w = Writer::default();
    for s in shapes {
        w.shape(s);
    }
    w.0
}

/// Decodes a module buffer. Reading past the end is a malformed-buffer error.
pub(crate) struct Reader<'a> {
    pub(crate) op: &'static str,
    pub(crate) bytes: &'a [u8],
}

impl Reader<'_> {
    fn take<const N: usize>(&mut self) -> Result<[u8; N]> {
        if self.bytes.len() < N {
            return Err(Error::Operation {
                op: self.op,
                message: "the module returned a malformed buffer".to_owned(),
            });
        }
        let (head, tail) = self.bytes.split_at(N);
        self.bytes = tail;
        let mut out = [0; N];
        out.copy_from_slice(head);
        Ok(out)
    }
    pub(crate) fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_ne_bytes(self.take()?))
    }
    pub(crate) fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_ne_bytes(self.take()?))
    }
    pub(crate) fn point(&mut self) -> Result<Point3> {
        Ok(Point3::new(self.f64()?, self.f64()?, self.f64()?))
    }
    pub(crate) fn vec3(&mut self) -> Result<Vec3> {
        Ok(Vec3::new(self.f64()?, self.f64()?, self.f64()?))
    }
    /// Takes ownership of a handle the module issued for this caller.
    pub(crate) fn shape(&mut self) -> Result<CapiShape> {
        Ok(CapiShape::adopt(self.u32()?))
    }
    /// A u32 count, then that many shapes.
    pub(crate) fn shapes(&mut self) -> Result<Vec<CapiShape>> {
        let n = self.u32()?;
        (0..n).map(|_| self.shape()).collect()
    }
    /// A u32 byte length, then UTF-8.
    pub(crate) fn string(&mut self) -> Result<String> {
        let n = self.u32()? as usize;
        if self.bytes.len() < n {
            return Err(Error::Operation {
                op: self.op,
                message: "the module returned a malformed buffer".to_owned(),
            });
        }
        let (head, tail) = self.bytes.split_at(n);
        self.bytes = tail;
        Ok(String::from_utf8_lossy(head).into_owned())
    }
}

fn decode_pairs(r: &mut Reader<'_>) -> Result<Vec<(CapiShape, Vec<CapiShape>)>> {
    let n = r.u32()?;
    (0..n).map(|_| Ok((r.shape()?, r.shapes()?))).collect()
}

/// Decodes `modeled_buffer` from `internal.hpp`, after a nonzero buffer handle.
pub(crate) fn decode_modeled(op: &'static str, buffer: &Buffer) -> Result<Modeled<CapiShape>> {
    let bytes = buffer.bytes();
    let mut r = Reader { op, bytes: &bytes };
    let shape = r.shape()?;
    let modified = decode_pairs(&mut r)?;
    let generated = decode_pairs(&mut r)?;
    let deleted = r.shapes()?;
    Ok(Modeled {
        shape,
        history: History {
            modified,
            generated,
            deleted,
        },
    })
}

/// The address the module polls for a cancel flag, or 0. Module memory on wasm is separate,
/// so a flag can't be shared there; it also couldn't be set while the single thread is busy.
pub(crate) fn cancel_address(cancel: Option<&CancelFlag>) -> usize {
    #[cfg(not(target_arch = "wasm32"))]
    {
        cancel.map_or(0, |c| std::ptr::from_ref(c.as_atomic()) as usize)
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = cancel;
        0
    }
}

/// Turns a failed operation into [`Error::Cancelled`] if its flag was set.
pub(crate) fn cancelled_or(cancel: Option<&CancelFlag>, error: Error) -> Error {
    if cancel.is_some_and(CancelFlag::is_cancelled) {
        Error::Cancelled
    } else {
        error
    }
}
