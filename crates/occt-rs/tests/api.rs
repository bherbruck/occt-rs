//! Behaviour of the public API that the kernel specification does not cover: operators,
//! conversions, validation, and the standard traits.

use std::collections::HashSet;
use std::error::Error as _;
use std::f64::consts::PI;
use std::hash::{BuildHasher, RandomState};
use std::io::{self, Cursor, Read, Write};

use occt::prelude::*;

const TOLERANCE: f64 = 1e-6;

type TestResult = occt::Result<()>;

fn near(got: f64, want: f64) {
    assert!(
        (got - want).abs() <= TOLERANCE * want.abs().max(1.0),
        "got {got}, want {want}"
    );
}

fn unit_cube() -> Solid {
    Cuboid::new([1.0, 1.0, 1.0]).build().unwrap()
}

fn offset_cube() -> Solid {
    Cuboid::new([1.0, 1.0, 1.0])
        .at([0.5, 0.0, 0.0])
        .build()
        .unwrap()
}

#[track_caller]
fn assert_invalid<T: std::fmt::Debug>(result: occt::Result<T>) {
    match result {
        Err(Error::InvalidInput(message)) => assert!(!message.is_empty()),
        other => panic!("expected InvalidInput, got {other:?}"),
    }
}

#[test]
fn operators_match_methods() -> TestResult {
    let (a, b) = (unit_cube(), offset_cube());
    near((&a | &b)?.volume()?, 1.5);
    near((&a - &b)?.volume()?, 0.5);
    near((&a & &b)?.volume()?, 0.5);
    near(a.fuse(&b)?.volume()?, 1.5);
    near(a.cut(&b)?.volume()?, 0.5);
    near(a.common(&b)?.volume()?, 0.5);
    Ok(())
}

#[test]
fn operators_mix_shapes_and_refinements() -> TestResult {
    let (solid, other) = (unit_cube(), offset_cube());
    let shape = Shape::from(other.clone());
    near((&solid - &shape)?.volume()?, 0.5);
    near((&shape - &solid)?.volume()?, 0.5);
    near(
        (&shape | &shape.translated([0.5, 0.0, 0.0])?)?.volume()?,
        1.5,
    );
    near((&solid & &other)?.volume()?, 0.5);
    Ok(())
}

#[test]
fn try_from_checks_the_kind() -> TestResult {
    let shape = Shape::from(unit_cube());
    assert_eq!(Solid::try_from(&shape)?.kind()?, ShapeKind::Solid);

    match Face::try_from(shape.clone()) {
        Err(Error::WrongKind { expected, found }) => {
            assert_eq!(expected, ShapeKind::Face);
            assert_eq!(found, ShapeKind::Solid);
        }
        other => panic!("expected WrongKind, got {other:?}"),
    }
    assert!(matches!(
        Edge::try_from(&shape),
        Err(Error::WrongKind { .. })
    ));
    assert!(matches!(
        Vertex::try_from(&shape),
        Err(Error::WrongKind { .. })
    ));

    let face = shape.faces()?.next().expect("a cube has faces");
    let round_trip = Face::try_from(Shape::from(face.clone()))?;
    assert_eq!(round_trip, face);
    Ok(())
}

#[test]
fn refinements_deref_to_shape_methods() -> TestResult {
    let cube = unit_cube();
    let face = cube.faces()?.next().expect("a cube has faces");
    near(face.area()?, 1.0);
    assert_eq!(face.surface_kind()?, SurfaceKind::Plane);
    assert_eq!(face.edges()?.len(), 4);

    let edge = cube.edges()?.next().expect("a cube has edges");
    near(edge.length()?, 1.0);
    assert_eq!(edge.curve_kind()?, CurveKind::Line);
    assert_eq!(edge.vertices()?.len(), 2);

    let vertex = cube.vertices()?.next().expect("a cube has vertices");
    assert!(cube.bounding_box()?.contains(vertex.point()?));

    let coerced: &Shape = &face;
    assert_eq!(coerced.kind()?, ShapeKind::Face);
    Ok(())
}

#[test]
fn iterators_are_exact_double_ended_and_outlive_the_shape() -> TestResult {
    let cube = unit_cube();
    let mut faces = cube.faces()?;
    assert_eq!(faces.len(), 6);
    let (first, last) = (faces.next().unwrap(), faces.next_back().unwrap());
    assert_ne!(first, last);
    assert_eq!(faces.len(), 4);
    assert_eq!(cube.edges()?.len(), 12);
    assert_eq!(cube.vertices()?.len(), 8);
    assert_eq!(cube.solids()?.len(), 1);
    assert_eq!(cube.solids()?.next().unwrap(), cube);

    let edges = unit_cube().edges()?;
    near(edges.map(|e| e.length().unwrap()).sum(), 12.0);
    Ok(())
}

#[test]
fn equality_and_hash_are_identity() -> TestResult {
    let cube = unit_cube();
    let state = RandomState::new();
    assert_eq!(cube, cube.clone());
    assert_eq!(state.hash_one(&cube), state.hash_one(cube.clone()));
    // A refinement hashes as its shape, which is what `Borrow<Shape>` promises.
    assert_eq!(
        state.hash_one(&cube),
        state.hash_one(Shape::from(cube.clone()))
    );
    assert_ne!(cube, unit_cube());

    let mut set = HashSet::new();
    assert!(set.insert(cube.clone()));
    assert!(!set.insert(cube.clone()));
    assert!(set.insert(unit_cube()));
    assert_eq!(set.len(), 2);
    assert!(set.contains(&Shape::from(cube)));

    // Each edge borders two faces but is one edge.
    let cube = unit_cube();
    let mut edges = HashSet::new();
    let mut visits = 0;
    for face in cube.faces()? {
        for edge in face.edges()? {
            visits += 1;
            edges.insert(edge);
        }
    }
    assert_eq!(visits, 24);
    assert_eq!(edges.len(), 12);
    Ok(())
}

#[test]
fn clone_outlives_original() -> TestResult {
    let copy = {
        let cube = unit_cube();
        cube.clone()
    };
    near(copy.volume()?, 1.0);
    assert!(copy.is_valid()?);
    Ok(())
}

#[test]
fn debug_output_names_no_backend() {
    let text = format!("{:?}", unit_cube());
    assert!(text.contains("Solid"), "{text}");
    for word in ["occt_", "Capi", "handle", "TopoDS"] {
        assert!(!text.contains(word), "{text} mentions {word}");
    }
}

#[test]
fn builders_reject_bad_numbers() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0, 0.0] {
        assert_invalid(Cuboid::new([bad, 1.0, 1.0]).build());
        assert_invalid(Cuboid::new([1.0, bad, 1.0]).build());
        assert_invalid(Cuboid::new([1.0, 1.0, bad]).build());
        assert_invalid(Cylinder::new(bad, 1.0).build());
        assert_invalid(Cylinder::new(1.0, bad).build());
        assert_invalid(Cone::new(1.0, 0.5, bad).build());
        assert_invalid(Sphere::new(bad).build());
    }
    for bad in [f64::NAN, f64::INFINITY, -1.0] {
        assert_invalid(Cone::new(bad, 1.0, 1.0).build());
        assert_invalid(Cone::new(1.0, bad, 1.0).build());
    }
    for bad in [f64::NAN, f64::INFINITY] {
        assert_invalid(Cuboid::new([1.0, 1.0, 1.0]).at([bad, 0.0, 0.0]).build());
        assert_invalid(Sphere::new(1.0).at([0.0, bad, 0.0]).build());
        let axis = Axis::new([0.0, 0.0, bad].into(), Dir3::Z);
        assert_invalid(Cylinder::new(1.0, 1.0).along(axis).build());
        assert_invalid(Cone::new(1.0, 0.0, 1.0).along(axis).build());
        assert_invalid(Polygon::new([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [bad, 1.0, 0.0]]).build());
    }
    assert_invalid(Cone::new(1.0, 1.0, 1.0).build());
    assert_invalid(Cone::new(0.0, 0.0, 1.0).build());
    assert_invalid(Polygon::new([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]).build());
    assert_invalid(Polygon::new(Vec::<Point3>::new()).build());
}

#[test]
fn builders_place_shapes_as_documented() -> TestResult {
    let corner = Cuboid::new([2.0, 2.0, 2.0]).at([1.0, 1.0, 1.0]).build()?;
    assert!(corner.center_of_mass()?.distance([2.0, 2.0, 2.0].into()) < TOLERANCE);

    for centered in [
        Cuboid::new([2.0, 2.0, 2.0])
            .at([1.0, 1.0, 1.0])
            .centered()
            .build()?,
        Cuboid::new([2.0, 2.0, 2.0])
            .centered()
            .at([1.0, 1.0, 1.0])
            .build()?,
    ] {
        assert!(centered.center_of_mass()?.distance([1.0, 1.0, 1.0].into()) < TOLERANCE);
    }

    let tall = Cylinder::new(1.0, 4.0).build()?;
    assert!(tall.center_of_mass()?.distance([0.0, 0.0, 2.0].into()) < TOLERANCE);
    near(tall.volume()?, PI * 4.0);

    let sideways = Cone::new(3.0, 0.0, 4.0).along(Axis::X).build()?;
    assert!(sideways.center_of_mass()?.distance([1.0, 0.0, 0.0].into()) < TOLERANCE);

    let points = vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
    ];
    near(Polygon::new(points).build()?.area()?, 0.5);
    Ok(())
}

#[test]
fn non_planar_polygon_is_an_error_not_a_panic() {
    let warped = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 0.0],
    ];
    assert!(Polygon::new(warped).build().is_err());
}

#[test]
fn transforms_reject_bad_numbers() {
    let cube = unit_cube();
    assert_invalid(cube.translated([f64::NAN, 0.0, 0.0]));
    assert_invalid(cube.rotated(
        Axis::new([f64::INFINITY, 0.0, 0.0].into(), Dir3::Z),
        Angle::degrees(1.0),
    ));
    assert_invalid(cube.rotated(Axis::Z, Angle::radians(f64::NAN)));
    assert_invalid(cube.scaled([f64::NAN, 0.0, 0.0], 2.0));
    for factor in [0.0, -2.0, f64::NAN, f64::INFINITY] {
        assert_invalid(cube.scaled(Point3::ORIGIN, factor));
    }
    assert_invalid(cube.extruded(Vec3::ZERO));
    assert_invalid(cube.extruded([f64::NAN, 0.0, 0.0]));
    for angle in [0.0, -1.0, 7.0, f64::NAN] {
        assert_invalid(cube.revolved(Axis::Z, Angle::radians(angle)));
    }
}

#[test]
fn full_turn_in_degrees_is_accepted() -> TestResult {
    let section = [
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
    ];
    let ring = Polygon::new(section)
        .build()?
        .revolved(Axis::Z, Angle::degrees(360.0))?;
    near(ring.volume()?, PI * 3.0);
    Ok(())
}

#[test]
fn blends_validate_size_and_selection() -> TestResult {
    let cube = unit_cube();
    assert_invalid(cube.fillet(0.1).build());
    assert_invalid(cube.chamfer(0.1).build());
    assert_invalid(cube.fillet(0.1).edges(Vec::<Edge>::new()).build());
    for bad in [0.0, -0.1, f64::NAN, f64::INFINITY] {
        assert_invalid(cube.fillet(bad).all_edges().build());
        assert_invalid(cube.chamfer(bad).all_edges().build());
    }
    let foreign = offset_cube().edges()?.next().expect("a cube has edges");
    assert_invalid(cube.fillet(0.1).edges([&foreign]).build());
    assert_invalid(cube.chamfer(0.1).edges([&foreign]).build());
    Ok(())
}

#[test]
fn blends_select_edges_flexibly() -> TestResult {
    let side = 10.0;
    let cube = Cuboid::new([side, side, side]).build()?;
    let edges: Vec<Edge> = cube.edges()?.collect();
    let (d, one) = (1.0, side * side * side - 0.5 * side);

    near(cube.chamfer(d).edges([&edges[0]]).build()?.volume()?, one);
    // By slice, by owned iterator, with a duplicate, and across several calls.
    near(cube.chamfer(d).edges(&edges[..1]).build()?.volume()?, one);
    near(
        cube.chamfer(d)
            .edges(edges[..1].to_vec())
            .build()?
            .volume()?,
        one,
    );
    near(
        cube.chamfer(d)
            .edges([&edges[0], &edges[0]])
            .build()?
            .volume()?,
        one,
    );
    near(
        cube.chamfer(d)
            .edges([&edges[0]])
            .edges([&edges[0]])
            .build()?
            .volume()?,
        one,
    );

    let all = cube.fillet(1.0).all_edges().build()?;
    let again = cube.fillet(1.0).edges(&edges).build()?;
    near(all.volume()?, again.volume()?);
    assert!(all.is_valid()?);
    Ok(())
}

#[test]
fn impossible_blend_is_an_operation_error() {
    let result = unit_cube().fillet(10.0).all_edges().build();
    assert!(matches!(result, Err(Error::Operation { .. })), "{result:?}");
}

#[test]
fn mesh_validates_options() -> TestResult {
    let cube = unit_cube();
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert_invalid(cube.mesh(MeshOptions::default().absolute(bad)));
        assert_invalid(cube.mesh(MeshOptions::default().relative(bad)));
        assert_invalid(cube.mesh(MeshOptions::default().angular(Angle::radians(bad))));
    }
    let mesh = cube.mesh(MeshOptions::default())?;
    near(mesh.volume(), 1.0);
    assert_eq!(mesh.triangles().len(), mesh.indices.len());
    Ok(())
}

#[test]
fn step_streams_round_trip() -> TestResult {
    let cube = unit_cube();
    let mut bytes = Vec::new();
    cube.write_step(&mut bytes)?;
    // Not compared with `to_step`: a STEP header carries a timestamp.
    assert!(bytes.starts_with(b"ISO-10303-21"));
    near(Shape::from_step(&cube.to_step()?)?.volume()?, 1.0);

    near(Shape::read_step(Cursor::new(&bytes))?.volume()?, 1.0);
    near(Shape::read_step(&bytes[..])?.volume()?, 1.0);
    near(Shape::from_step(&bytes)?.volume()?, 1.0);
    assert!(matches!(
        Shape::from_step(b"garbage"),
        Err(Error::Operation { .. })
    ));
    Ok(())
}

struct Broken;

impl Read for Broken {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
        Err(io::Error::other("the stream broke"))
    }
}

impl Write for Broken {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> {
        Err(io::Error::other("the stream broke"))
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[test]
fn stream_failures_are_io_errors() {
    assert!(matches!(Shape::read_step(Broken), Err(Error::Io(_))));
    assert!(matches!(unit_cube().write_step(Broken), Err(Error::Io(_))));
}

#[test]
fn errors_display_and_chain() {
    let wrong = Face::try_from(Shape::from(unit_cube())).unwrap_err();
    assert_eq!(wrong.to_string(), "expected a face, found a solid");
    assert!(wrong.source().is_none());

    let invalid = Sphere::new(-1.0).build().unwrap_err();
    assert_eq!(
        invalid.to_string(),
        "invalid input: sphere radius must be finite and positive, got -1"
    );
    assert!(invalid.source().is_none());

    let operation = Shape::from_step(b"garbage").unwrap_err();
    assert!(!operation.to_string().is_empty());
    assert!(operation.source().is_none());

    let io = Shape::read_step(Broken).unwrap_err();
    assert_eq!(io.to_string(), "I/O error: the stream broke");
    assert_eq!(
        io.source().expect("I/O errors chain").to_string(),
        "the stream broke"
    );
}

#[test]
fn kernel_information_is_available() -> TestResult {
    occt::ready()?;
    assert!(!occt::kernel_version()?.is_empty());
    assert!(!occt::notice().is_empty());
    Ok(())
}

#[test]
fn shapes_derive_common_traits() {
    fn check<T: Clone + std::fmt::Debug + Eq + std::hash::Hash + AsRef<Shape>>() {}
    check::<Shape>();
    check::<Solid>();
    check::<Face>();
    check::<Edge>();
    check::<Vertex>();
}
