//! The operations domain: the kernel checks, then the public API and its error paths.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use occt::prelude::*;
use occt::{
    Angle, Boolean, BooleanOp, CancelFlag, ChamferSize, EdgeChamfer, EdgeFillet, Plane, RadiusLaw,
    Split,
};
use occt_conformance::advanced_blends_and_booleans::checks;
use occt_kernel_capi::CapiKernel;

fn is_invalid<T: std::fmt::Debug>(result: occt::Result<T>) -> bool {
    matches!(result, Err(Error::InvalidInput(_)))
}

fn cube(side: f64) -> Solid {
    Cuboid::new([side, side, side]).build().unwrap()
}

fn drill(x: f64, y: f64, r: f64) -> Solid {
    Cylinder::new(r, 30.0)
        .along(Axis::new([x, y, -10.0].into(), Dir3::Z))
        .build()
        .unwrap()
}

fn edge_of(shape: &Shape, a: [f64; 3], b: [f64; 3]) -> Edge {
    let middle = Point3::new(
        (a[0] + b[0]) / 2.0,
        (a[1] + b[1]) / 2.0,
        (a[2] + b[2]) / 2.0,
    );
    shape
        .edges()
        .unwrap()
        .find(|e| e.center_of_mass().unwrap().distance(middle) < 1e-6)
        .expect("the edge exists")
}

fn face_of(shape: &Shape, center: [f64; 3]) -> Face {
    let center = Point3::from(center);
    shape
        .faces()
        .unwrap()
        .find(|f| f.center_of_mass().unwrap().distance(center) < 1e-6)
        .expect("the face exists")
}

fn length(shape: &Shape) -> f64 {
    shape.edges().unwrap().map(|e| e.length().unwrap()).sum()
}

/// Every check, each reported by name when it fails.
#[test]
fn capi_kernel_conforms_to_operations() {
    let mut failed = Vec::new();
    for (name, check) in checks::<CapiKernel>() {
        if std::panic::catch_unwind(check).is_err() {
            failed.push(name);
        }
    }
    assert!(failed.is_empty(), "failed: {failed:?}");
}

// --- the builder ----------------------------------------------------------------

#[test]
fn every_boolean_option_composes() {
    let plate = Cuboid::new([20.0, 20.0, 2.0]).build().unwrap();
    let flag = CancelFlag::new();
    let holes = [drill(5.0, 5.0, 1.0), drill(15.0, 5.0, 1.0)];
    let out = Boolean::cut([&plate])
        .tools(&holes)
        .fuzzy(1e-6)
        .unify()
        .parallel()
        .cancel_with(&flag)
        .build()
        .unwrap();
    let want = 800.0 - 2.0 * std::f64::consts::PI * 2.0;
    assert!((out.shape.volume().unwrap() - want).abs() < 1e-6);
    assert!(out.shape.is_valid().unwrap());
}

#[test]
fn boolean_new_picks_the_operation() {
    let a = cube(2.0);
    let b = Cuboid::new([2.0, 2.0, 2.0])
        .at([1.0, 0.0, 0.0])
        .build()
        .unwrap();
    for (op, want) in [
        (BooleanOp::Fuse, 12.0),
        (BooleanOp::Cut, 4.0),
        (BooleanOp::Common, 4.0),
    ] {
        let out = Boolean::new(op, [&a]).tools([&b]).build().unwrap();
        assert!((out.shape.volume().unwrap() - want).abs() < 1e-9, "{op:?}");
    }
}

#[test]
fn fuse_takes_all_its_shapes_as_arguments() {
    let shapes: Vec<Solid> = (0..4)
        .map(|i| {
            Cuboid::new([2.0, 2.0, 2.0])
                .at([f64::from(i), 0.0, 0.0])
                .build()
                .unwrap()
        })
        .collect();
    let fused = Boolean::fuse(&shapes).build().unwrap();
    assert!((fused.shape.volume().unwrap() - 20.0).abs() < 1e-9);
    assert!(is_invalid(Boolean::common(&shapes).build()));
}

#[test]
fn boolean_rejects_bad_arguments() {
    let a = cube(1.0);
    let none: [&Shape; 0] = [];
    assert!(is_invalid(Boolean::cut(none).tools([&a]).build()));
    assert!(is_invalid(Boolean::cut([&a]).build()));
    assert!(is_invalid(Boolean::fuse([&a]).build()));
    assert!(is_invalid(Boolean::common([&a]).build()));
    for fuzzy in [f64::NAN, f64::INFINITY, 0.0, -1.0] {
        assert!(
            is_invalid(Boolean::cut([&a]).tools([&a]).fuzzy(fuzzy).build()),
            "{fuzzy}"
        );
        assert!(
            is_invalid(Split::new([&a]).tools([&a]).fuzzy(fuzzy).build()),
            "{fuzzy}"
        );
    }
    assert!(is_invalid(Split::new(none).tools([&a]).build()));
    assert!(is_invalid(Split::new([&a]).build()));
    assert!(is_invalid(a.split_by(none)));
}

#[test]
fn a_shape_cut_by_itself_leaves_nothing_or_an_error() {
    let a = cube(1.0);
    if let Ok(out) = Boolean::cut([&a]).tools([&a]).build() {
        assert!(out.shape.volume().unwrap().abs() < 1e-9);
    }
}

#[test]
fn mixing_dimensions_is_a_kernel_error_not_a_crash() {
    let a = cube(1.0);
    let edge = a.edges().unwrap().next().unwrap();
    let result = Boolean::fuse([&*a, &*edge]).build();
    let Err(Error::Operation { message, .. }) = result else {
        panic!("{result:?}");
    };
    assert!(message.contains("kinds"), "{message}");
}

// --- cancellation ---------------------------------------------------------------

/// A boolean of many curved tools through a plate, slow enough to catch in the act.
fn big_boolean() -> Boolean {
    let plate = Cuboid::new([64.0, 64.0, 4.0]).build().unwrap();
    let mut balls = Vec::new();
    for i in 0..16 {
        for j in 0..16 {
            balls.push(
                Sphere::new(2.2)
                    .at([4.0 * f64::from(i) + 2.0, 4.0 * f64::from(j) + 2.0, 2.0])
                    .build()
                    .unwrap(),
            );
        }
    }
    Boolean::cut([&plate]).tools(&balls)
}

#[test]
fn a_set_flag_cancels() {
    let flag = CancelFlag::new();
    flag.cancel();
    let a = cube(2.0);
    let b = Cuboid::new([2.0, 2.0, 2.0])
        .at([1.0, 1.0, 1.0])
        .build()
        .unwrap();
    let result = Boolean::cut([&a]).tools([&b]).cancel_with(&flag).build();
    assert!(matches!(result, Err(Error::Cancelled)), "{result:?}");
    let result = Split::new([&a]).tools([&b]).cancel_with(&flag).build();
    assert!(matches!(result, Err(Error::Cancelled)), "{result:?}");
}

/// Set from another thread, a flag stops a running boolean long before it would finish. A
/// clone of the builder without the flag gives the time it would otherwise take.
#[test]
fn a_flag_set_from_another_thread_stops_a_running_boolean() {
    let boolean = big_boolean();
    let begun = Instant::now();
    let finished = boolean
        .clone()
        .build()
        .expect("the uncancelled run succeeds");
    let full = begun.elapsed();
    assert!(finished.shape.is_valid().unwrap());
    assert!(
        full > Duration::from_millis(200),
        "the boolean is too quick to test cancelling: {full:?}"
    );

    let flag = CancelFlag::new();
    let started = Arc::new(AtomicBool::new(false));
    let canceller = {
        let (flag, started) = (flag.clone(), Arc::clone(&started));
        thread::spawn(move || {
            while !started.load(Ordering::SeqCst) {
                thread::yield_now();
            }
            thread::sleep(full / 10);
            flag.cancel();
        })
    };
    let boolean = boolean.cancel_with(&flag);
    let begun = Instant::now();
    started.store(true, Ordering::SeqCst);
    let result = boolean.build();
    let stopped = begun.elapsed();
    canceller.join().unwrap();
    assert!(matches!(result, Err(Error::Cancelled)), "{result:?}");
    println!("a full run takes {full:?}, the cancelled one {stopped:?}");
    assert!(
        stopped < full * 3 / 4,
        "stopped after {stopped:?}, a full run takes {full:?}"
    );

    // The kernel is still fine afterwards.
    let a = cube(2.0);
    let b = Cuboid::new([2.0, 2.0, 2.0])
        .at([1.0, 1.0, 1.0])
        .build()
        .unwrap();
    assert!((a.cut(&b).unwrap().volume().unwrap() - 7.0).abs() < 1e-9);
}

#[test]
fn a_flag_stops_a_parallel_boolean_too() {
    let flag = CancelFlag::new();
    let boolean = big_boolean().parallel().cancel_with(&flag);
    let canceller = {
        let flag = flag.clone();
        thread::spawn(move || {
            thread::sleep(Duration::from_millis(10));
            flag.cancel();
        })
    };
    let result = boolean.build();
    canceller.join().unwrap();
    assert!(matches!(result, Err(Error::Cancelled)), "{result:?}");
}

#[test]
fn an_unset_flag_changes_nothing() {
    let flag = CancelFlag::new();
    let a = cube(2.0);
    let b = Cuboid::new([2.0, 2.0, 2.0])
        .at([1.0, 1.0, 1.0])
        .build()
        .unwrap();
    let out = Boolean::cut([&a])
        .tools([&b])
        .cancel_with(&flag)
        .build()
        .unwrap();
    assert!((out.shape.volume().unwrap() - 7.0).abs() < 1e-9);
}

// --- history ---------------------------------------------------------------------

#[test]
fn history_follows_a_box_cut_by_a_cylinder() {
    let block = cube(10.0);
    let tool = drill(5.0, 5.0, 2.0);
    let out = Boolean::cut([&block]).tools([&tool]).build().unwrap();
    let result: Vec<Face> = out.shape.faces().unwrap().collect();
    let in_result = |s: &Shape| result.iter().any(|f| **f == *s);

    // An untouched face stays traceable.
    let side = face_of(&block, [0.0, 5.0, 5.0]);
    assert!(!out.history.is_deleted(&side));
    let traced = out.history.trace(&side);
    assert_eq!(traced.len(), 1);
    assert!(in_result(&traced[0]));

    // A cut face reports modified, and that's what it traces to.
    let top = face_of(&block, [5.0, 5.0, 10.0]);
    let after = out.history.modified(&top);
    assert_eq!(after.len(), 1);
    assert!(in_result(&after[0]));
    assert!((after[0].area().unwrap() - (100.0 - 4.0 * std::f64::consts::PI)).abs() < 1e-6);
    assert_eq!(out.history.trace(&top), after);

    // The cylinder's side generates the wall of the hole.
    let wall_source = tool
        .faces()
        .unwrap()
        .find(|f| f.surface_kind().unwrap() == SurfaceKind::Cylinder)
        .unwrap();
    let walls: Vec<&Shape> = out
        .history
        .generated(&wall_source)
        .iter()
        .filter(|s| s.kind().unwrap() == ShapeKind::Face)
        .collect();
    assert_eq!(walls.len(), 1);
    assert!(in_result(walls[0]));
    let wall = Face::try_from(walls[0]).unwrap();
    assert_eq!(wall.surface_kind().unwrap(), SurfaceKind::Cylinder);
    assert!((wall.area().unwrap() - 2.0 * std::f64::consts::PI * 2.0 * 10.0).abs() < 1e-6);

    // The tool's caps lie outside, so nothing is left of them.
    for cap in tool
        .faces()
        .unwrap()
        .filter(|f| f.surface_kind().unwrap() == SurfaceKind::Plane)
    {
        assert!(out.history.is_deleted(&cap));
        assert!(out.history.trace(&cap).is_empty());
    }

    // Edges and vertices are covered too: the rim on top is gone, the box's corners stay.
    let corner = block.vertices().unwrap().next().expect("a box has corners");
    assert!(!out.history.is_deleted(&corner));
}

#[test]
fn history_survives_unify() {
    let a = cube(1.0);
    let b = Cuboid::new([1.0, 1.0, 1.0])
        .at([1.0, 0.0, 0.0])
        .build()
        .unwrap();
    let out = Boolean::fuse([&a, &b]).unify().build().unwrap();
    assert_eq!(out.shape.faces().unwrap().len(), 6);
    let top_a = face_of(&a, [0.5, 0.5, 1.0]);
    let top_b = face_of(&b, [1.5, 0.5, 1.0]);
    let top = face_of(&out.shape, [1.0, 0.5, 1.0]);
    assert_eq!(out.history.trace(&top_a), [Shape::from(top.clone())]);
    assert_eq!(out.history.trace(&top_b), [Shape::from(top)]);
}

#[test]
fn inputs_stay_as_they_were() {
    let a = cube(2.0);
    let b = Cuboid::new([2.0, 2.0, 2.0])
        .at([1.0, 1.0, 1.0])
        .build()
        .unwrap();
    let faces: Vec<Face> = a.faces().unwrap().collect();
    Boolean::cut([&a]).tools([&b]).build().unwrap();
    a.split_by([&b]).unwrap();
    assert!((a.volume().unwrap() - 8.0).abs() < 1e-12);
    assert_eq!(a.faces().unwrap().collect::<Vec<_>>(), faces);
    assert!(a.is_valid().unwrap());
}

// --- sections ---------------------------------------------------------------------

#[test]
fn section_plane_of_a_box_mid_plane_is_its_perimeter() {
    let block = Cuboid::new([2.0, 3.0, 4.0]).build().unwrap();
    let cases = [
        (Plane::new([1.0, 0.0, 0.0].into(), Dir3::X), 14.0),
        (Plane::new([0.0, 1.5, 0.0].into(), Dir3::Y), 12.0),
        (Plane::new([0.0, 0.0, 2.0].into(), Dir3::Z), 10.0),
    ];
    for (plane, perimeter) in cases {
        let section = block.section_plane(plane).unwrap();
        assert!((length(&section) - perimeter).abs() < 1e-9, "{plane:?}");
        assert_eq!(section.edges().unwrap().len(), 4);
    }
}

#[test]
fn section_plane_rejects_a_bad_plane() {
    let block = cube(1.0);
    for origin in [f64::NAN, f64::INFINITY] {
        let plane = Plane::new(Point3::new(origin, 0.0, 0.0), Dir3::Z);
        assert!(is_invalid(block.section_plane(plane)));
    }
}

#[test]
fn sections_that_miss_are_empty() {
    let block = cube(1.0);
    let plane = Plane::new([0.0, 0.0, 5.0].into(), Dir3::Z);
    assert_eq!(
        block.section_plane(plane).unwrap().edges().unwrap().len(),
        0
    );
    let far = Cuboid::new([1.0, 1.0, 1.0])
        .at([5.0, 5.0, 5.0])
        .build()
        .unwrap();
    assert_eq!(block.section(&far).unwrap().edges().unwrap().len(), 0);
}

// --- splitting --------------------------------------------------------------------

#[test]
fn split_keeps_every_piece_and_the_total_volume() {
    let block = cube(9.0);
    let blades: Vec<Face> = [3.0, 6.0]
        .iter()
        .map(|&x| {
            Polygon::new([
                [x, -1.0, -1.0],
                [x, 10.0, -1.0],
                [x, 10.0, 10.0],
                [x, -1.0, 10.0],
            ])
            .build()
            .unwrap()
        })
        .collect();
    let split = block.split_by(&blades).unwrap();
    let pieces: Vec<Solid> = split.shape.solids().unwrap().collect();
    assert_eq!(pieces.len(), 3);
    for piece in &pieces {
        assert!((piece.volume().unwrap() - 243.0).abs() < 1e-9);
    }
    assert!((split.shape.volume().unwrap() - 729.0).abs() < 1e-9);
    assert_eq!(split.shape.kind().unwrap(), ShapeKind::Compound);

    let bottom = face_of(&block, [4.5, 4.5, 0.0]);
    assert_eq!(split.history.modified(&bottom).len(), 3);
}

#[test]
fn split_takes_its_options() {
    let block = cube(4.0);
    let blade = Polygon::new([
        [2.0, -1.0, -1.0],
        [2.0, 5.0, -1.0],
        [2.0, 5.0, 5.0],
        [2.0, -1.0, 5.0],
    ])
    .build()
    .unwrap();
    let flag = CancelFlag::new();
    let split = Split::new([&*block])
        .tools([&*blade])
        .fuzzy(1e-7)
        .unify()
        .parallel()
        .cancel_with(&flag)
        .build()
        .unwrap();
    assert_eq!(split.shape.solids().unwrap().len(), 2);
    assert!((split.shape.volume().unwrap() - 64.0).abs() < 1e-9);
}

#[test]
fn a_tool_that_misses_leaves_one_piece() {
    let block = cube(1.0);
    let far = Cuboid::new([1.0, 1.0, 1.0])
        .at([5.0, 0.0, 0.0])
        .build()
        .unwrap();
    let split = block.split_by([&far]).unwrap();
    assert_eq!(split.shape.solids().unwrap().len(), 1);
    assert!((split.shape.volume().unwrap() - 1.0).abs() < 1e-12);
}

// --- fillets ----------------------------------------------------------------------

#[test]
fn edge_fillet_laws_and_history() {
    use std::f64::consts::PI;
    let block = cube(20.0);
    let edge = edge_of(&block, [0.0, 0.0, 0.0], [0.0, 0.0, 20.0]);
    let other = edge_of(&block, [20.0, 20.0, 0.0], [20.0, 20.0, 20.0]);
    let third = edge_of(&block, [20.0, 0.0, 0.0], [20.0, 0.0, 20.0]);
    let out = EdgeFillet::on(&block)
        .radius(&edge, 1.0)
        .linear(&other, 1.0, 1.5)
        .through(&third, [(0.0, 1.0), (0.5, 1.5), (1.0, 1.0)])
        .build()
        .unwrap();
    assert!(out.shape.is_valid().unwrap());
    assert!(out.shape.volume().unwrap() < 8000.0);
    for source in [&edge, &other, &third] {
        let made = out.history.generated(source);
        assert!(!made.is_empty());
        for face in made {
            assert_ne!(face.kind().unwrap(), ShapeKind::Edge);
        }
        assert!(out.history.is_deleted(source));
    }
    let round = Face::try_from(&out.history.generated(&edge)[0]).unwrap();
    assert!((round.area().unwrap() - PI / 2.0 * 20.0).abs() < 1e-3);
    assert!(out.shape.faces().unwrap().any(|f| f == round));
}

#[test]
fn law_can_be_built_from_data() {
    let block = cube(10.0);
    let edge = edge_of(&block, [0.0, 0.0, 0.0], [0.0, 0.0, 10.0]);
    let law: RadiusLaw = RadiusLaw::Constant(1.0);
    let by_law = EdgeFillet::on(&block).law(&edge, law).build().unwrap();
    let by_radius = EdgeFillet::on(&block).radius(&edge, 1.0).build().unwrap();
    assert!((by_law.shape.volume().unwrap() - by_radius.shape.volume().unwrap()).abs() < 1e-9);
    // Owned edges work as well as borrowed ones.
    EdgeFillet::on(&block).radius(edge, 1.0).build().unwrap();
}

#[test]
fn fractions_short_of_the_ends_hold_the_radius() {
    let block = cube(20.0);
    let edge = edge_of(&block, [0.0, 0.0, 0.0], [0.0, 0.0, 20.0]);
    let out = EdgeFillet::on(&block)
        .through(&edge, [(0.25, 1.0), (0.75, 1.5)])
        .build()
        .unwrap();
    assert!(out.shape.is_valid().unwrap());
    let removed = 8000.0 - out.shape.volume().unwrap();
    // 5 long at r = 1, 5 long at r = 1.5, and between them at no more than 1.5 or less than 1.
    let corner = 1.0 - std::f64::consts::PI / 4.0;
    let low = corner * (10.0 + 5.0 * 2.25);
    let high = corner * (5.0 + 15.0 * 2.25);
    assert!(
        removed > low && removed < high,
        "{removed} not in {low}..{high}"
    );
}

#[test]
fn fillet_rejects_bad_laws() {
    let block = cube(10.0);
    let edge = block.edges().unwrap().next().unwrap();
    let on = || EdgeFillet::on(&block);
    for bad in [f64::NAN, f64::INFINITY, 0.0, -1.0] {
        assert!(is_invalid(on().radius(&edge, bad).build()), "{bad}");
        assert!(is_invalid(on().linear(&edge, bad, 1.0).build()), "{bad}");
        assert!(is_invalid(on().linear(&edge, 1.0, bad).build()), "{bad}");
        assert!(
            is_invalid(on().through(&edge, [(0.0, bad), (1.0, 1.0)]).build()),
            "{bad}"
        );
    }
    assert!(is_invalid(on().build()));
    assert!(is_invalid(on().through(&edge, []).build()));
    assert!(is_invalid(on().through(&edge, [(0.0, 1.0)]).build()));
    assert!(is_invalid(
        on().through(&edge, [(0.5, 1.0), (0.5, 2.0)]).build()
    ));
    assert!(is_invalid(
        on().through(&edge, [(0.6, 1.0), (0.5, 2.0)]).build()
    ));
    assert!(is_invalid(
        on().through(&edge, [(-0.1, 1.0), (0.5, 2.0)]).build()
    ));
    assert!(is_invalid(
        on().through(&edge, [(0.5, 1.0), (1.1, 2.0)]).build()
    ));
    assert!(is_invalid(
        on().through(&edge, [(f64::NAN, 1.0), (1.0, 2.0)]).build()
    ));
    assert!(is_invalid(
        on().radius(&edge, 1.0).radius(&edge, 2.0).build()
    ));
    assert!(on().radius(&edge, 1.0).radius(&edge, 1.0).build().is_ok());
}

#[test]
fn fillet_rejects_foreign_edges() {
    let block = cube(10.0);
    let other = cube(10.0);
    let foreign = other.edges().unwrap().next().unwrap();
    assert!(is_invalid(
        EdgeFillet::on(&block).radius(&foreign, 1.0).build()
    ));
}

#[test]
fn a_fillet_too_large_is_an_operation_error() {
    let block = cube(2.0);
    let edge = block.edges().unwrap().next().unwrap();
    let result = EdgeFillet::on(&block).radius(&edge, 50.0).build();
    assert!(matches!(result, Err(Error::Operation { .. })), "{result:?}");
}

/// A rounded rim and the walls next to it join tangentially, so the chain rounds as one,
/// even when only one is named, and naming two of its edges with the same size is fine.
#[test]
fn tangent_chains_round_together() {
    let block = cube(20.0);
    let edge = edge_of(&block, [0.0, 0.0, 0.0], [0.0, 0.0, 20.0]);
    let round = block.fillet(3.0).edges([&edge]).build().unwrap();
    // The top of the rounded box has straight edges and an arc, all tangent in a chain.
    let chain: Vec<Edge> = round
        .edges()
        .unwrap()
        .filter(|e| {
            let c = e.center_of_mass().unwrap();
            let on_top = (e.bounding_box().unwrap().min.z - 20.0).abs() < 1e-6;
            on_top && (c.x < 10.0 || c.y < 10.0)
        })
        .collect();
    assert!(chain.len() >= 3, "{}", chain.len());
    let one = EdgeFillet::on(&round)
        .radius(&chain[0], 1.0)
        .build()
        .unwrap();
    assert!(one.shape.is_valid().unwrap());
    let mut all = EdgeFillet::on(&round);
    for edge in &chain {
        all = all.radius(edge, 1.0);
    }
    let all = all.build().unwrap();
    assert!(all.shape.is_valid().unwrap());
    assert!(
        (one.shape.volume().unwrap() - all.shape.volume().unwrap()).abs() < 1e-3,
        "{} {}",
        one.shape.volume().unwrap(),
        all.shape.volume().unwrap()
    );
}

/// A radius law around a closed edge has to come back to where it started.
#[test]
fn closed_edges_take_laws_that_close() {
    use std::f64::consts::PI;
    let bar = Cylinder::new(5.0, 10.0).build().unwrap();
    let rim = bar
        .edges()
        .unwrap()
        .find(|e| {
            (e.length().unwrap() - 2.0 * PI * 5.0).abs() < 1e-6
                && (e.center_of_mass().unwrap().z - 10.0).abs() < 1e-6
        })
        .unwrap();
    let open = EdgeFillet::on(&bar)
        .through(&rim, [(0.0, 1.0), (1.0, 2.0)])
        .build();
    assert!(matches!(open, Err(Error::Operation { .. })), "{open:?}");
    let closed = EdgeFillet::on(&bar)
        .through(&rim, [(0.0, 1.0), (0.5, 1.2), (1.0, 1.0)])
        .build()
        .unwrap();
    assert!(closed.shape.is_valid().unwrap());
    let volume = closed.shape.volume().unwrap();
    let whole = PI * 25.0 * 10.0;
    assert!(volume < whole - 1.0 && volume > whole - 10.0, "{volume}");
}

// --- chamfers ---------------------------------------------------------------------

/// Edges joined tangentially bevel as one, so they can't take separate sizes.
#[test]
fn chamfers_on_a_tangent_chain_share_one_size() {
    let block = cube(20.0);
    let edge = edge_of(&block, [0.0, 0.0, 0.0], [0.0, 0.0, 20.0]);
    let round = block.fillet(3.0).edges([&edge]).build().unwrap();
    let chain: Vec<Edge> = round
        .edges()
        .unwrap()
        .filter(|e| {
            let c = e.center_of_mass().unwrap();
            (e.bounding_box().unwrap().min.z - 20.0).abs() < 1e-6 && (c.x < 10.0 || c.y < 10.0)
        })
        .collect();
    assert!(chain.len() >= 3, "{}", chain.len());
    let same = EdgeChamfer::on(&round)
        .equal(&chain[0], 1.0)
        .equal(&chain[1], 1.0)
        .build();
    assert!(same.unwrap().shape.is_valid().unwrap());
    let mixed = EdgeChamfer::on(&round)
        .equal(&chain[0], 1.0)
        .equal(&chain[1], 2.0)
        .build();
    assert!(matches!(mixed, Err(Error::Operation { .. })), "{mixed:?}");
}

#[test]
fn edge_chamfer_sizes_and_history() {
    let block = cube(10.0);
    let e1 = edge_of(&block, [0.0, 0.0, 0.0], [0.0, 0.0, 10.0]);
    let f1 = face_of(&block, [0.0, 5.0, 5.0]);
    let e2 = edge_of(&block, [10.0, 10.0, 0.0], [10.0, 10.0, 10.0]);
    let f2 = face_of(&block, [10.0, 5.0, 5.0]);
    let e3 = edge_of(&block, [10.0, 0.0, 0.0], [10.0, 0.0, 10.0]);
    let out = EdgeChamfer::on(&block)
        .two_distances(&e1, &f1, 1.0, 2.0)
        .distance_angle(&e2, &f2, 2.0, Angle::degrees(45.0))
        .equal(&e3, 1.0)
        .build()
        .unwrap();
    let removed = (1.0 + 2.0 + 0.5) * 10.0;
    assert!((out.shape.volume().unwrap() - (1000.0 - removed)).abs() < 1e-6);
    for edge in [&e1, &e2, &e3] {
        assert_eq!(out.history.generated(edge).len(), 1);
    }
    assert!(out.history.modified(&f1).len() == 1);
}

#[test]
fn chamfer_rejects_bad_sizes_and_references() {
    let block = cube(10.0);
    let edge = edge_of(&block, [0.0, 0.0, 0.0], [0.0, 0.0, 10.0]);
    let face = face_of(&block, [0.0, 5.0, 5.0]);
    let away = face_of(&block, [10.0, 5.0, 5.0]);
    let other = cube(10.0);
    let foreign_edge = edge_of(&other, [0.0, 0.0, 0.0], [0.0, 0.0, 10.0]);
    let foreign_face = face_of(&other, [0.0, 5.0, 5.0]);
    let on = || EdgeChamfer::on(&block);
    let a45 = Angle::degrees(45.0);
    for bad in [f64::NAN, f64::INFINITY, 0.0, -1.0] {
        assert!(is_invalid(on().equal(&edge, bad).build()), "{bad}");
        assert!(
            is_invalid(on().two_distances(&edge, &face, bad, 1.0).build()),
            "{bad}"
        );
        assert!(
            is_invalid(on().two_distances(&edge, &face, 1.0, bad).build()),
            "{bad}"
        );
        assert!(
            is_invalid(on().distance_angle(&edge, &face, bad, a45).build()),
            "{bad}"
        );
    }
    for bad in [0.0, -30.0, 180.0, 200.0, f64::NAN, f64::INFINITY] {
        let angle = Angle::degrees(bad);
        assert!(
            is_invalid(on().distance_angle(&edge, &face, 1.0, angle).build()),
            "{bad}"
        );
    }
    assert!(is_invalid(on().build()));
    assert!(is_invalid(on().equal(&foreign_edge, 1.0).build()));
    assert!(is_invalid(
        on().two_distances(&foreign_edge, &face, 1.0, 2.0).build()
    ));
    assert!(is_invalid(
        on().two_distances(&edge, &foreign_face, 1.0, 2.0).build()
    ));
    assert!(is_invalid(
        on().two_distances(&edge, &away, 1.0, 2.0).build()
    ));
    assert!(is_invalid(
        on().distance_angle(&edge, &away, 1.0, a45).build()
    ));
    assert!(is_invalid(on().equal(&edge, 1.0).equal(&edge, 2.0).build()));
    assert!(is_invalid(
        on().equal(&edge, 1.0)
            .two_distances(&edge, &face, 1.0, 1.0)
            .build()
    ));
    assert!(on().equal(&edge, 1.0).equal(&edge, 1.0).build().is_ok());
}

#[test]
fn chamfers_too_large_or_too_steep_are_operation_errors() {
    let block = cube(2.0);
    let edge = edge_of(&block, [0.0, 0.0, 0.0], [0.0, 0.0, 2.0]);
    let face = face_of(&block, [0.0, 1.0, 1.0]);
    let too_large = EdgeChamfer::on(&block).equal(&edge, 50.0).build();
    assert!(
        matches!(too_large, Err(Error::Operation { .. })),
        "{too_large:?}"
    );
    let steep = EdgeChamfer::on(&block)
        .distance_angle(&edge, &face, 1.0, Angle::degrees(120.0))
        .build();
    assert!(matches!(steep, Err(Error::Operation { .. })), "{steep:?}");
}

#[test]
fn chamfer_size_is_public_data() {
    assert_eq!(ChamferSize::Equal(1.0), ChamferSize::Equal(1.0));
}
