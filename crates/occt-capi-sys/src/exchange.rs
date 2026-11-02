//! Raw declarations of the exchange domain, matching `capi/src/exchange.cpp`.
// `EXPORTS` is only read by the crate's own test that compares it with the C++.
#![allow(dead_code)]
capi! {
    "_occt_native_write" fn occt_native_write(h: u32, with_triangulation: i32) -> u32;
    "_occt_native_read" fn occt_native_read(data: usize, len: usize) -> u32;
    "_occt_step_document_read" fn occt_step_document_read(
        data: usize, len: usize, cancel: usize,
    ) -> u32;
    "_occt_step_document_write" fn occt_step_document_write(
        nodes: usize, len: usize, schema: i32, units: f64, cancel: usize,
    ) -> u32;
    "_occt_iges_read" fn occt_iges_read(data: usize, len: usize) -> u32;
    "_occt_iges_write" fn occt_iges_write(handles: usize, count: usize) -> u32;
}
