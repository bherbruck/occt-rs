//! The inspection domain: the kernel specification against the active backend, and the
//! public API around it, including what it refuses.

use std::collections::HashSet;
use std::f64::consts::PI;
use std::fs::File;
use std::thread;
use std::time::{Duration, Instant};

use occt::prelude::*;
use occt::{CancelFlag, EdgeConvexity, PointLocation, View};
use occt_kernel_capi::CapiKernel;

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn is_invalid<T: std::fmt::Debug>(result: occt::Result<T>) -> bool {
    matches!(result, Err(Error::InvalidInput(_)))
}

fn cube() -> Solid {
    Cuboid::new([2.0, 2.0, 2.0]).build().unwrap()
}

fn l_block() -> Shape {
    Polygon::new([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
    ])
    .build()
    .unwrap()
    .extruded([0.0, 0.0, 3.0])
    .unwrap()
}

#[test]
fn capi_kernel_conforms() {
    let failed: Vec<_> = occt_conformance::inspection::checks::<CapiKernel>()
        .into_iter()
        .filter(|(name, check)| {
            let failed = std::panic::catch_unwind(*check).is_err();
            if failed {
                eprintln!("FAILED: {name}");
            }
            failed
        })
        .map(|(name, _)| name)
        .collect();
    assert!(failed.is_empty(), "failed: {failed:?}");
}

#[test]
fn mass_properties_of_a_turned_cylinder() -> TestResult {
    let (r, h) = (1.5, 4.0);
    let c = Cylinder::new(r, h).along(Axis::X).build()?;
    let props = c.mass_properties()?;
    let m = PI * r * r * h;
    assert!((props.mass - m).abs() < 1e-6 * m);
    assert!(props.center.distance([h / 2.0, 0.0, 0.0].into()) < 1e-6);
    assert!((props.inertia[0][0] - m * r * r / 2.0).abs() < 1e-5);
    assert!((props.inertia[1][1] - m * (3.0 * r * r + h * h) / 12.0).abs() < 1e-5);
    Ok(())
}

#[test]
fn mass_properties_of_a_vertex_is_an_error_not_a_panic() -> TestResult {
    let corner = cube().vertices()?.next().unwrap();
    assert!(matches!(
        corner.mass_properties(),
        Err(Error::Operation { .. })
    ));
    Ok(())
}

#[test]
fn distance_takes_any_refinement() -> TestResult {
    let a = cube();
    let far = Sphere::new(1.0).at([10.0, 1.0, 1.0]).build()?;
    let d = a.distance_to(&far)?;
    assert!((d.value - 7.0).abs() < 1e-6);
    assert!(d.on_a.distance([2.0, 1.0, 1.0].into()) < 1e-6);
    assert!(d.on_b.distance([9.0, 1.0, 1.0].into()) < 1e-6);
    let edge = a.edges()?.next().unwrap();
    assert!(edge.distance_to(&a)?.value.abs() < 1e-6);
    Ok(())
}

#[test]
fn distance_into_a_group_of_solids_is_zero() -> TestResult {
    let far = Cuboid::new([1.0, 1.0, 1.0]).at([50.0, 0.0, 0.0]).build()?;
    let two = Cuboid::new([10.0, 10.0, 10.0]).build()?.fuse(&far)?;
    assert_eq!(two.kind()?, ShapeKind::Compound);
    let inner = Cuboid::new([1.0, 1.0, 1.0]).at([4.0, 4.0, 4.0]).build()?;
    assert!(two.distance_to(&inner)?.value.abs() < 1e-6);
    assert!(inner.distance_to(&two)?.value.abs() < 1e-6);
    Ok(())
}

#[test]
fn locate_rejects_bad_numbers() {
    let c = cube();
    assert!(is_invalid(c.locate([f64::NAN, 0.0, 0.0])));
    assert!(is_invalid(c.locate([f64::INFINITY, 0.0, 0.0])));
    assert!(is_invalid(c.locate_within([0.0, 0.0, 0.0], 0.0)));
    assert!(is_invalid(c.locate_within([0.0, 0.0, 0.0], -1.0)));
    assert!(is_invalid(c.locate_within([0.0, 0.0, 0.0], f64::NAN)));
}

#[test]
fn locate_works_on_a_cavity() -> TestResult {
    let ball = Sphere::new(3.0).at([5.0, 5.0, 5.0]).build()?;
    let hollow = Cuboid::new([10.0, 10.0, 10.0]).build()?.cut(&ball)?;
    let hollow = Solid::try_from(hollow)?;
    assert_eq!(hollow.locate([5.0, 5.0, 5.0])?, PointLocation::Outside);
    assert_eq!(hollow.locate([8.0, 5.0, 5.0])?, PointLocation::OnBoundary);
    assert_eq!(hollow.locate([9.0, 5.0, 5.0])?, PointLocation::Inside);
    Ok(())
}

#[test]
fn ray_cast_rejects_bad_input() {
    let c = cube();
    assert!(is_invalid(
        c.ray_cast([f64::NAN, 0.0, 0.0], [0.0, 0.0, 1.0])
    ));
    assert!(is_invalid(c.ray_cast([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])));
    assert!(is_invalid(
        c.ray_cast([0.0, 0.0, 0.0], [f64::NAN, 0.0, 1.0])
    ));
    assert!(is_invalid(c.ray_cast_within(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        0.0
    )));
    assert!(is_invalid(c.ray_cast_within(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        -1.0
    )));
    assert!(is_invalid(c.ray_cast_within(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        f64::NAN
    )));
}

#[test]
fn ray_cast_ignores_direction_length_and_identifies_faces() -> TestResult {
    let c = cube();
    let hits = c.ray_cast([1.0, 1.0, -5.0], [0.0, 0.0, 100.0])?;
    assert_eq!(hits.len(), 2);
    assert!((hits[0].distance - 5.0).abs() < 1e-6);
    let faces: HashSet<Face> = c.faces()?.collect();
    assert!(hits.iter().all(|hit| faces.contains(&hit.face)));
    let edges = hits[0].face.edges()?.count();
    assert_eq!(edges, 4);
    Ok(())
}

#[test]
fn ray_through_a_shared_edge_hits_both_faces() -> TestResult {
    let c = cube();
    // Through the top edge at x = 2, along the diagonal into the cube.
    let hits = c.ray_cast([3.0, 1.0, 3.0], [-1.0, 0.0, -1.0])?;
    let at_edge = hits
        .iter()
        .filter(|h| (h.distance - 2f64.sqrt()).abs() < 1e-6)
        .count();
    assert_eq!(at_edge, 2, "{hits:?}");
    Ok(())
}

#[test]
fn adjacency_reports_foreign_parts_as_invalid() -> TestResult {
    let (c, other) = (cube(), cube());
    let foreign_edge = other.edges()?.next().unwrap();
    let foreign_vertex = other.vertices()?.next().unwrap();
    assert!(is_invalid(foreign_edge.faces_in(&c)));
    assert!(is_invalid(foreign_edge.convexity_in(&c)));
    assert!(is_invalid(foreign_vertex.faces_in(&c)));
    assert!(is_invalid(foreign_vertex.edges_in(&c)));
    assert!(is_invalid(foreign_edge.ancestors_in(&c, ShapeKind::Face)));
    Ok(())
}

#[test]
fn adjacency_is_typed_and_agrees_with_faces() -> TestResult {
    let c = cube();
    let faces: HashSet<Face> = c.faces()?.collect();
    for edge in c.edges()? {
        let around = edge.faces_in(&c)?;
        assert_eq!(around.len(), 2);
        assert!(around.iter().all(|f| faces.contains(f)));
    }
    for corner in c.vertices()? {
        assert_eq!(corner.faces_in(&c)?.len(), 3);
        assert_eq!(corner.edges_in(&c)?.len(), 3);
    }
    Ok(())
}

#[test]
fn convexity_tells_an_l_block_apart() -> TestResult {
    let block = l_block();
    let mut concave = 0;
    for edge in block.edges()? {
        match edge.convexity_in(&block)? {
            EdgeConvexity::Concave => concave += 1,
            EdgeConvexity::Convex => {}
            other => panic!("unexpected {other:?}"),
        }
    }
    assert_eq!(concave, 1);
    Ok(())
}

#[test]
fn convexity_on_a_sphere_seam_is_open() -> TestResult {
    let ball = Sphere::new(1.0).build()?;
    for edge in ball.edges()? {
        assert_eq!(edge.convexity_in(&ball)?, EdgeConvexity::Open);
    }
    Ok(())
}

#[test]
fn wires_and_their_edges_are_typed() -> TestResult {
    let c = cube();
    let face = c.faces()?.next().unwrap();
    let wires = face.wires()?;
    assert_eq!(wires.len(), 1);
    let edges = wires[0].wire_edges()?;
    assert_eq!(edges.len(), 4);
    let own: HashSet<Edge> = face.edges()?.collect();
    assert!(edges.iter().all(|e| own.contains(e)));
    Ok(())
}

#[test]
fn wire_edges_needs_a_wire() {
    assert!(matches!(
        cube().wire_edges(),
        Err(Error::WrongKind {
            expected: ShapeKind::Wire,
            found: ShapeKind::Solid
        })
    ));
}

#[test]
fn cylinder_side_wire_passes_the_seam_twice() -> TestResult {
    let c = Cylinder::new(1.0, 2.0).build()?;
    let side = c
        .faces()?
        .find(|f| f.surface_kind().is_ok_and(|k| k == SurfaceKind::Cylinder))
        .unwrap();
    let wires = side.wires()?;
    assert_eq!(wires.len(), 1);
    let edges = wires[0].wire_edges()?;
    assert_eq!(edges.len(), 4, "two rims and the seam twice");
    assert_eq!(edges.iter().collect::<HashSet<_>>().len(), 3);
    Ok(())
}

#[test]
fn tessellate_rejects_bad_options() {
    let c = cube();
    for options in [
        MeshOptions::default().absolute(0.0),
        MeshOptions::default().absolute(f64::NAN),
        MeshOptions::default().relative(-1.0),
        MeshOptions::default().angular(Angle::radians(0.0)),
    ] {
        assert!(is_invalid(c.tessellate(options).build()), "{options:?}");
    }
}

#[test]
fn tessellate_lines_up_with_traversal() -> TestResult {
    let block = l_block();
    let parts = block.tessellate(MeshOptions::default()).build()?;
    let faces: Vec<Face> = block.faces()?.collect();
    let edges: Vec<Edge> = block.edges()?.collect();
    assert_eq!(parts.faces.len(), faces.len());
    assert_eq!(parts.edges.len(), edges.len());
    for (face, part) in faces.iter().zip(&parts.faces) {
        assert!((part.mesh.area() - face.area()?).abs() < 1e-6);
    }
    for (edge, line) in edges.iter().zip(&parts.edges) {
        let length: f64 = line.points.windows(2).map(|p| p[0].distance(p[1])).sum();
        assert!((length - edge.length()?).abs() < 1e-6);
    }
    assert!((parts.merged().volume() - block.volume()?).abs() < 1e-6);
    Ok(())
}

#[test]
fn tessellate_handles_shapes_built_from_shared_parts() -> TestResult {
    // A cut shares untouched faces with its operand, which has been meshed coarser.
    let plate = Cuboid::new([10.0, 10.0, 1.0]).build()?;
    plate
        .tessellate(MeshOptions::default().absolute(1.0))
        .build()?;
    let drill = Cylinder::new(1.0, 3.0)
        .along(Axis::new([5.0, 5.0, -1.0].into(), Dir3::Z))
        .build()?;
    let holed = plate.cut(&drill)?;
    let parts = holed
        .tessellate(MeshOptions::default().absolute(0.01))
        .build()?;
    let want = 100.0 - PI;
    assert!((parts.merged().volume() - want).abs() / want < 1e-3);
    // And the operand still answers.
    let again = plate
        .tessellate(MeshOptions::default().absolute(0.01))
        .build()?;
    assert!((again.merged().volume() - 100.0).abs() < 1e-6);
    Ok(())
}

#[test]
fn tessellate_stops_when_cancelled_from_another_thread() -> TestResult {
    // Many faces, since the mesher checks between faces, and each with enough to do that
    // the run takes far longer than the wait below.
    let mut shape: Shape = Sphere::new(10.0).build()?.into();
    for i in 1..16 {
        let ball = Sphere::new(10.0)
            .at([f64::from(i) * 30.0, 0.0, 0.0])
            .build()?;
        shape = shape.fuse(&ball)?;
    }
    let options = MeshOptions::default().absolute(5e-3);
    let flag = CancelFlag::new();
    let setter = flag.clone();
    let canceller = thread::spawn(move || {
        thread::sleep(Duration::from_millis(20));
        setter.cancel();
    });
    let start = Instant::now();
    let result = shape.tessellate(options).cancel_with(&flag).build();
    let stopped_after = start.elapsed();
    canceller.join().unwrap();
    assert!(
        matches!(result, Err(Error::Cancelled)),
        "{:?}",
        result.map(|_| ())
    );

    // A flag that was only read at the end would also say cancelled, so it must also have
    // saved the time. The first run can only have kept a face or two of its meshes.
    let start = Instant::now();
    let full = shape.tessellate(options).build()?;
    let full_run = start.elapsed();
    eprintln!("stopped after {stopped_after:?}, a full run takes {full_run:?}");
    assert!(stopped_after < full_run / 2);
    assert_eq!(full.faces.len(), 16);
    Ok(())
}

#[test]
fn view_rejects_bad_values() {
    let c = cube();
    assert!(is_invalid(
        c.hidden_lines(View::from_direction([0.0, 0.0, 0.0]))
    ));
    assert!(is_invalid(c.hidden_lines(View::from_direction([
        f64::NAN,
        0.0,
        1.0
    ]))));
    assert!(is_invalid(c.hidden_lines(View::top().up([0.0, 0.0, 1.0]))));
    assert!(is_invalid(c.hidden_lines(View::top().up([
        f64::NAN,
        0.0,
        1.0
    ]))));
    assert!(is_invalid(c.hidden_lines(View::top().looking_at([
        f64::NAN,
        0.0,
        0.0
    ]))));
    assert!(is_invalid(c.hidden_lines(View::top().perspective(0.0))));
    assert!(is_invalid(c.hidden_lines(View::top().deflection(-1.0))));
}

fn visible_extent(lines: &[occt::DrawingLine]) -> ([f64; 2], [f64; 2]) {
    let mut min = [f64::MAX; 2];
    let mut max = [f64::MIN; 2];
    for point in lines.iter().filter(|l| l.visible).flat_map(|l| &l.points) {
        for i in 0..2 {
            min[i] = min[i].min(point[i]);
            max[i] = max[i].max(point[i]);
        }
    }
    (min, max)
}

#[test]
fn standard_views_look_where_they_say() -> TestResult {
    let block = Cuboid::new([1.0, 2.0, 3.0]).build()?;
    for (view, want) in [
        (View::top(), [1.0, 2.0]),
        (View::front(), [1.0, 3.0]),
        (View::right(), [2.0, 3.0]),
    ] {
        let (min, max) = visible_extent(&block.hidden_lines(view)?);
        for i in 0..2 {
            assert!(min[i].abs() < 1e-6, "{view:?}: {min:?}");
            assert!((max[i] - want[i]).abs() < 1e-6, "{view:?}: {max:?}");
        }
    }
    let lines = block.hidden_lines(View::isometric())?;
    let visible = lines.iter().filter(|l| l.visible).count();
    assert_eq!((visible, lines.len() - visible), (9, 3));
    Ok(())
}

#[test]
fn view_target_is_the_drawing_origin() -> TestResult {
    let block = Cuboid::new([1.0, 2.0, 3.0])
        .at([10.0, 20.0, 30.0])
        .build()?;
    let (min, _) = visible_extent(&block.hidden_lines(View::top().looking_at([10.0, 20.0, 0.0]))?);
    assert!(min[0].abs() < 1e-6 && min[1].abs() < 1e-6, "{min:?}");
    Ok(())
}

#[test]
fn hidden_lines_leave_the_shape_alone() -> TestResult {
    // The smooth-edge bookkeeping happens on a copy, so identity and measures survive.
    let cube = cube();
    let before: Vec<Face> = cube.faces()?.collect();
    cube.hidden_lines(View::isometric())?;
    let after: Vec<Face> = cube.faces()?.collect();
    assert_eq!(before, after);
    assert!(cube.is_valid()?);
    Ok(())
}

#[test]
fn mesh_exports_write_to_real_files() -> TestResult {
    let ball = Sphere::new(5.0).build()?;
    let mesh = ball
        .tessellate(MeshOptions::default().absolute(0.1))
        .build()?
        .merged();
    let dir = env!("CARGO_TARGET_TMPDIR");

    let stl = format!("{dir}/ball.stl");
    mesh.write_stl_binary(File::create(&stl)?)?;
    assert_eq!(
        std::fs::metadata(&stl)?.len() as usize,
        84 + mesh.indices.len() * 50
    );
    mesh.write_stl_ascii(File::create(format!("{dir}/ball.txt.stl"))?)?;
    mesh.write_obj(File::create(format!("{dir}/ball.obj"))?)?;
    mesh.write_3mf(File::create(format!("{dir}/ball.3mf"))?)?;
    Ok(())
}
