//! Exercises the kernel end to end. On wasm32 it's driven by `wasm_e2e.mjs`, natively
//! it just prints the same report.

use std::f64::consts::PI;

use occt_kernel::{
    Axis, Blends, Booleans, Dir3, Error, Kernel, MeshOptions, Point3, Primitives, Properties,
    ShapeKind, StepIo, Tessellation, Topology, Vec3,
};
use occt_kernel_capi::CapiKernel as K;

fn report() -> Result<String, Error> {
    let version = K::version()?;
    let plate = K::make_box(Point3::ORIGIN, Vec3::new(10.0, 10.0, 2.0))?;
    let axis = Axis::new(Point3::new(5.0, 5.0, -1.0), Dir3::Z);
    let hole = K::make_cylinder(axis, 2.0, 4.0)?;
    let part = K::cut(&plate, &hole)?;
    let volume = K::volume(&part)?;
    let expected = 200.0 - PI * 8.0;
    if (volume - expected).abs() > 1e-6 {
        return Err(Error::InvalidInput(format!(
            "volume {volume}, expected {expected}"
        )));
    }

    let edges = K::subshapes(&plate, ShapeKind::Edge)?;
    let rounded = K::volume(&K::fillet(&plate, &[&edges[0]], 0.5)?)?;
    let mesh = K::mesh(&part, &MeshOptions::default().absolute(0.01))?;
    let step = K::write_step(&part)?;
    let back = K::read_step(&step)?;

    let zero_box = K::make_box(Point3::ORIGIN, Vec3::ZERO);
    let zero_box = match zero_box {
        Err(e @ Error::Operation { .. }) => e.to_string(),
        other => return Err(Error::InvalidInput(format!("zero box gave {other:?}"))),
    };

    // Forces module memory to grow, then checks later calls still see the right bytes.
    let large_input = K::read_step(&vec![b' '; 64 << 20]).is_err();
    let after_growth = K::volume(&K::read_step(&step)?)?;
    if !large_input || (after_growth - expected).abs() > 1e-6 {
        return Err(Error::InvalidInput(format!(
            "after growth: {large_input} {after_growth}"
        )));
    }

    Ok(format!(
        "version={version} part={part:?} volume={volume:.6} filleted={rounded:.6} \
         triangles={} mesh_volume={:.6} step_bytes={} step_volume={:.6} \
         after_growth={after_growth:.6} zero_box=[{zero_box}]",
        mesh.indices.len(),
        mesh.volume(),
        step.len(),
        K::volume(&back)?,
    ))
}

/// What the kernel says before the module is loaded.
fn before_load() -> String {
    format!("{:?}", K::ready())
}

#[cfg(target_arch = "wasm32")]
mod web {
    use wasm_bindgen::prelude::wasm_bindgen;

    /// See [`super::before_load`].
    #[wasm_bindgen]
    pub fn before_load() -> String {
        super::before_load()
    }

    /// See [`super::report`].
    #[wasm_bindgen]
    pub fn report() -> Result<String, String> {
        super::report().map_err(|e| e.to_string())
    }
}

fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    match report() {
        Ok(report) => println!("{}\n{report}", before_load()),
        Err(e) => panic!("{e}"),
    }
}
