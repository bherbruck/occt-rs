//! Shapes cross threads and kernel calls from many threads are safe.

use std::sync::Arc;
use std::thread;

use occt_kernel::{
    Booleans, MeshOptions, Point3, Primitives, Properties, Tessellation, Topology, Vec3,
};
use occt_kernel_capi::CapiKernel as K;

#[test]
fn many_threads_share_and_drop_shapes() {
    let base = Arc::new(K::make_box(Point3::ORIGIN, Vec3::new(10.0, 10.0, 10.0)).unwrap());
    let workers: Vec<_> = (0..8)
        .map(|i| {
            let base = Arc::clone(&base);
            thread::spawn(move || {
                for j in 0..10 {
                    let offset = f64::from(i) * 0.1 + f64::from(j) * 0.01;
                    let tool = K::make_box(
                        Point3::new(offset, offset, offset),
                        Vec3::new(2.0, 2.0, 2.0),
                    )
                    .unwrap();
                    let cut = K::cut(&base, &tool).unwrap();
                    let expected = 1000.0 - 8.0;
                    assert!((K::volume(&cut).unwrap() - expected).abs() < 1e-6);
                    // Meshing writes to shared topology, which is what the lock has to protect.
                    let mesh = K::mesh(&base, &MeshOptions::default()).unwrap();
                    assert!(!mesh.is_empty());
                    let faces = K::subshapes(&cut, occt_kernel::ShapeKind::Face).unwrap();
                    thread::spawn(move || drop(faces)).join().unwrap();
                }
            })
        })
        .collect();
    for worker in workers {
        worker.join().unwrap();
    }
}
