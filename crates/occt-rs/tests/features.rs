//! The features domain: the contract's checks, then the public API.

use std::panic;

#[test]
fn capi_kernel_passes_the_feature_checks() {
    let checks = occt_conformance::features_and_sweeps::checks::<occt_kernel_capi::CapiKernel>();
    let failed: Vec<_> = checks
        .iter()
        .filter(|(name, check)| {
            eprintln!("--- {name}");
            panic::catch_unwind(*check).is_err()
        })
        .map(|(name, _)| *name)
        .collect();
    assert!(
        failed.is_empty(),
        "{} of {} checks failed: {}",
        failed.len(),
        checks.len(),
        failed.join(", ")
    );
}

mod api {
    use std::f64::consts::{FRAC_PI_2, PI};

    use occt::prelude::*;
    use occt::{Loft, Plane, Sweep, SweepOrientation, Transform};

    fn square(side: f64, z: f64) -> Face {
        Polygon::new([
            [0.0, 0.0, z],
            [side, 0.0, z],
            [side, side, z],
            [0.0, side, z],
        ])
        .build()
        .unwrap()
    }

    fn near(got: f64, want: f64) {
        assert!(
            (got - want).abs() <= 1e-3 * want.abs(),
            "got {got}, want {want}"
        );
    }

    fn is_invalid_input<T: std::fmt::Debug>(result: occt::Result<T>) {
        match result {
            Err(Error::InvalidInput(_)) => {}
            other => panic!("expected invalid input, got {other:?}"),
        }
    }

    fn is_wrong_kind<T: std::fmt::Debug>(result: occt::Result<T>) {
        match result {
            Err(Error::WrongKind { .. }) => {}
            other => panic!("expected the wrong kind, got {other:?}"),
        }
    }

    fn is_operation<T: std::fmt::Debug>(result: occt::Result<T>) {
        match result {
            Err(Error::Operation { .. }) => {}
            other => panic!("expected a failed operation, got {other:?}"),
        }
    }

    fn top_of(solid: &Solid) -> Face {
        let height = solid.bounding_box().unwrap().max.z;
        solid
            .faces()
            .unwrap()
            .find(|f| (f.center_of_mass().unwrap().z - height).abs() < 1e-6)
            .unwrap()
    }

    fn vertex_at(at: [f64; 3]) -> Vertex {
        square(1.0, at[2])
            .translated([at[0], at[1], 0.0])
            .unwrap()
            .vertices()
            .unwrap()
            .find(|v| v.point().unwrap().distance(at.into()) < 1e-6)
            .unwrap()
    }

    /// A disc across the path at `(bend, 0, 0)`, and a quarter turn around Z from there.
    fn disc_and_arc(bend: f64, radius: f64) -> (Face, Shape) {
        let start = [bend, 0.0, 0.0];
        let bar = Cylinder::new(radius, 1.0)
            .along(Axis::new(start.into(), Dir3::Y))
            .build()
            .unwrap();
        let disc = bar
            .faces()
            .unwrap()
            .find(|f| f.center_of_mass().unwrap().distance(start.into()) < 1e-6)
            .unwrap();
        let arc = vertex_at(start)
            .revolved(Axis::Z, Angle::degrees(90.0))
            .unwrap();
        (disc, arc)
    }

    #[test]
    fn modeled_extrude_and_revolve_trace_their_inputs() {
        let face = square(2.0, 0.0);
        let prism = face.extruded_modeled([0.0, 0.0, 3.0]).unwrap();
        near(prism.shape.volume().unwrap(), 12.0);
        for edge in face.edges().unwrap() {
            assert_eq!(prism.history.generated(&edge).len(), 1);
        }

        is_invalid_input(face.extruded_modeled([0.0, 0.0, 0.0]));
        is_invalid_input(face.extruded_modeled([f64::NAN, 0.0, 1.0]));

        let moved = face.translated([5.0, 0.0, 0.0]).unwrap();
        let ring = moved.revolved_modeled(Axis::Y, Angle::FULL_TURN).unwrap();
        near(ring.shape.volume().unwrap(), 4.0 * 2.0 * PI * 6.0);
        for edge in moved.edges().unwrap() {
            assert!(!ring.history.is_deleted(&edge));
        }

        is_invalid_input(moved.revolved_modeled(Axis::Y, Angle::radians(0.0)));
        is_invalid_input(moved.revolved_modeled(Axis::Y, Angle::radians(-1.0)));
        is_invalid_input(moved.revolved_modeled(Axis::Y, Angle::radians(7.0)));
        is_invalid_input(moved.revolved_modeled(Axis::Y, Angle::radians(f64::NAN)));
        let nowhere = Axis::new(Point3::new(f64::NAN, 0.0, 0.0), Dir3::Z);
        is_invalid_input(moved.revolved_modeled(nowhere, Angle::FULL_TURN));
    }

    #[test]
    fn tapered_extrusion() {
        let face = square(10.0, 0.0);
        let by = [0.0, 0.0, 2.0];
        let frustum = face.extruded_tapered(by, Angle::degrees(45.0)).unwrap();
        near(frustum.volume().unwrap(), 2.0 / 3.0 * (100.0 + 60.0 + 36.0));

        is_invalid_input(face.extruded_tapered([0.0, 0.0, 0.0], Angle::degrees(5.0)));
        is_invalid_input(face.extruded_tapered([0.0, f64::INFINITY, 1.0], Angle::degrees(5.0)));
        is_invalid_input(face.extruded_tapered(by, Angle::radians(f64::NAN)));
        is_invalid_input(face.extruded_tapered(by, Angle::degrees(90.0)));
        is_invalid_input(face.extruded_tapered(by, Angle::degrees(-120.0)));

        // In its own plane, too steep, and on a curved face.
        is_operation(face.extruded_tapered([1.0, 0.0, 0.0], Angle::degrees(5.0)));
        is_operation(face.extruded_tapered([0.0, 0.0, 50.0], Angle::degrees(45.0)));
        let round = Sphere::new(1.0)
            .build()
            .unwrap()
            .faces()
            .unwrap()
            .next()
            .unwrap();
        is_operation(round.extruded_tapered(by, Angle::degrees(5.0)));
    }

    #[test]
    fn sweep_builder_options() {
        let (bend, radius) = (10.0, 1.0);
        let (disc, arc) = disc_and_arc(bend, radius);
        let pappus = PI * radius * radius * bend * FRAC_PI_2;

        let plain = Sweep::profile(&disc).along(&arc).build().unwrap();
        near(plain.shape.volume().unwrap(), pappus);
        assert_eq!(plain.shape.kind().unwrap(), ShapeKind::Solid);

        let frenet = Sweep::profile(&disc)
            .along(&arc)
            .orientation(SweepOrientation::Frenet)
            .build()
            .unwrap();
        near(frenet.shape.volume().unwrap(), pappus);

        let upright = Sweep::profile(&disc)
            .along(&arc)
            .binormal(Dir3::Z)
            .build()
            .unwrap();
        near(upright.shape.volume().unwrap(), pappus);

        let parallel = Sweep::profile(&disc)
            .along(&arc)
            .orientation(SweepOrientation::Fixed)
            .build()
            .unwrap();
        near(
            parallel.shape.volume().unwrap(),
            PI * radius * radius * bend,
        );

        // The disc's rim is a closed edge: a tube, or a solid on request.
        let rim = disc.edges().unwrap().next().unwrap();
        let tube = Sweep::profile(&rim).along(&arc).build().unwrap();
        assert_ne!(tube.shape.kind().unwrap(), ShapeKind::Solid);
        near(
            tube.shape.area().unwrap(),
            2.0 * PI * radius * bend * FRAC_PI_2,
        );
        let bar = Sweep::profile(&rim).along(&arc).solid().build().unwrap();
        near(bar.shape.volume().unwrap(), pappus);

        // The disc's edges and the disc itself are traced to the result.
        assert!(!plain.history.generated(&rim).is_empty());
        assert_eq!(plain.history.generated(&disc).len(), 2);
    }

    #[test]
    fn sweep_rejects_wrong_kinds_and_open_solids() {
        let face = square(2.0, 0.0);
        let block = Cuboid::new([1.0, 1.0, 1.0]).build().unwrap();
        let line = vertex_at([0.0, 0.0, 0.0])
            .extruded([0.0, 0.0, 5.0])
            .unwrap();
        let edge = face.edges().unwrap().next().unwrap();

        is_wrong_kind(Sweep::profile(&block).along(&line).build());
        is_wrong_kind(
            Sweep::profile(&vertex_at([0.0, 0.0, 0.0]))
                .along(&line)
                .build(),
        );
        is_wrong_kind(Sweep::profile(&face).along(&block).build());
        is_wrong_kind(Sweep::profile(&face).along(&face).build());
        is_operation(Sweep::profile(&edge).along(&line).solid().build());
    }

    #[test]
    fn loft_builder_options() {
        let (bottom, top) = (square(4.0, 0.0), square(2.0, 3.0));
        let frustum = 3.0 / 3.0 * (16.0 + 8.0 + 4.0);

        // References, owned values and whatever an iterator yields all work. These squares
        // share a corner rather than an axis, which a frustum's volume doesn't notice.
        let by_ref = Loft::through([&bottom, &top]).ruled().build().unwrap();
        near(by_ref.volume().unwrap(), frustum);
        let owned = Loft::through(vec![bottom.clone(), top.clone()])
            .build()
            .unwrap();
        near(owned.volume().unwrap(), frustum);
        let shapes: Vec<Shape> = vec![bottom.clone().into(), top.clone().into()];
        let from_iter = Loft::through(shapes.iter())
            .tolerance(1e-4)
            .build()
            .unwrap();
        near(from_iter.volume().unwrap(), frustum);

        let sheet = Loft::through([&bottom, &top])
            .ruled()
            .sheet()
            .build()
            .unwrap();
        assert_ne!(sheet.kind().unwrap(), ShapeKind::Solid);
        near(sheet.volume().unwrap().abs() + 1.0, 1.0);

        let apex = vertex_at([2.0, 2.0, 3.0]);
        let pyramid = Loft::through([&*bottom, &*apex]).ruled().build().unwrap();
        near(pyramid.volume().unwrap(), 16.0 * 3.0 / 3.0);

        let none: [&Shape; 0] = [];
        is_invalid_input(Loft::through(none).build());
        is_invalid_input(Loft::through([&bottom]).build());
        is_invalid_input(Loft::through([&bottom, &top]).tolerance(0.0).build());
        is_invalid_input(Loft::through([&bottom, &top]).tolerance(f64::NAN).build());

        let block = Cuboid::new([1.0, 1.0, 1.0]).build().unwrap();
        is_wrong_kind(Loft::through([&*bottom, &*block]).build());
        is_operation(Loft::through([&*bottom, &*apex, &*top]).build());
    }

    #[test]
    fn hollowing() {
        let block = Cuboid::new([10.0, 10.0, 10.0]).build().unwrap();
        let top = top_of(&block);
        let tray = block.shelled(1.0).open(&top).build().unwrap();
        near(tray.shape.volume().unwrap(), 1000.0 - 8.0 * 8.0 * 9.0);
        // The result is typed, so it goes straight on into another edit.
        assert!(tray.shape.shelled(0.1).build().is_ok());

        // Selecting a face twice is the same as once.
        let twice = block
            .shelled(1.0)
            .open(&top)
            .open_faces([&top])
            .build()
            .unwrap();
        near(twice.shape.volume().unwrap(), tray.shape.volume().unwrap());

        let outer = block.shelled(1.0).open(&top).outward().build().unwrap();
        near(outer.shape.volume().unwrap(), 12.0 * 12.0 * 11.0 - 1000.0);

        let closed = block.shelled(1.0).build().unwrap();
        near(closed.shape.volume().unwrap(), 1000.0 - 512.0);

        let other = Cuboid::new([10.0, 10.0, 10.0]).build().unwrap();
        let foreign = top_of(&other);
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            is_invalid_input(block.shelled(bad).open(&top).build());
        }
        is_invalid_input(block.shelled(1.0).open(&foreign).build());
        is_operation(block.shelled(6.0).build());
    }

    #[test]
    fn offset_and_thicken() {
        let block = Cuboid::new([2.0, 2.0, 2.0]).build().unwrap();
        near(block.offset(1.0).unwrap().volume().unwrap(), 64.0);
        near(block.offset(-0.5).unwrap().volume().unwrap(), 1.0);
        is_invalid_input(block.offset(0.0));
        is_invalid_input(block.offset(f64::NAN));
        is_operation(block.offset(-1.0));
        is_operation(block.offset(-7.0));

        let face = square(2.0, 0.0);
        let plate: Solid = face.thickened(0.5).unwrap();
        near(plate.volume().unwrap(), 2.0);
        is_invalid_input(face.thickened(0.0));
        is_invalid_input(face.thickened(f64::NEG_INFINITY));
        is_wrong_kind(block.thickened(1.0));
        is_wrong_kind(face.edges().unwrap().next().unwrap().thickened(1.0));
    }

    #[test]
    fn drafting() {
        let block = Cuboid::new([10.0, 10.0, 2.0]).build().unwrap();
        let side = block
            .faces()
            .unwrap()
            .find(|f| f.center_of_mass().unwrap().x > 9.0)
            .unwrap();
        let angle = Angle::degrees(45.0);
        let wedge = block
            .drafted(Dir3::Z, Plane::XY, angle)
            .face(&side)
            .faces([&side])
            .build()
            .unwrap();
        near(wedge.shape.volume().unwrap(), 200.0 - 20.0);
        let tilted = wedge.history.modified(&side);
        assert_eq!(tilted.len(), 1);
        assert_eq!(wedge.history.trace(&side), tilted);

        let with =
            |pull, plane, angle: Angle| block.drafted(pull, plane, angle).face(&side).build();
        is_invalid_input(block.drafted(Dir3::Z, Plane::XY, angle).build());
        is_invalid_input(with(Dir3::Z, Plane::XY, Angle::radians(0.0)));
        is_invalid_input(with(Dir3::Z, Plane::XY, Angle::radians(f64::NAN)));
        is_invalid_input(with(Dir3::Z, Plane::XY, Angle::degrees(90.0)));
        let adrift = Plane::new(Point3::new(0.0, f64::NAN, 0.0), Dir3::Z);
        is_invalid_input(with(Dir3::Z, adrift, angle));

        let other = Cuboid::new([1.0, 1.0, 1.0]).build().unwrap();
        let foreign = other.faces().unwrap().next().unwrap();
        is_invalid_input(
            block
                .drafted(Dir3::Z, Plane::XY, angle)
                .face(&foreign)
                .build(),
        );
        is_operation(
            block
                .drafted(Dir3::Z, Plane::XY, angle)
                .face(&top_of(&block))
                .build(),
        );
    }

    #[test]
    fn removing_faces() {
        let plate = Cuboid::new([10.0, 10.0, 2.0]).build().unwrap();
        let axis = Axis::new([5.0, 5.0, -1.0].into(), Dir3::Z);
        let drill = Cylinder::new(1.0, 4.0).along(axis).build().unwrap();
        let holed = Solid::try_from((&plate - &drill).unwrap()).unwrap();
        let wall = holed
            .faces()
            .unwrap()
            .find(|f| f.surface_kind().unwrap() == SurfaceKind::Cylinder)
            .unwrap();

        let filled = holed.without_faces([&wall]).unwrap();
        near(filled.shape.volume().unwrap(), 200.0);
        assert!(filled.history.is_deleted(&wall));
        assert!(filled.history.trace(&wall).is_empty());

        let none: [&Face; 0] = [];
        is_invalid_input(holed.without_faces(none));
        let other = Cuboid::new([1.0, 1.0, 1.0]).build().unwrap();
        let foreign = other.faces().unwrap().next().unwrap();
        is_invalid_input(holed.without_faces([&foreign]));

        // Its neighbours can't close the gap a side of the plate leaves behind.
        let side = holed
            .faces()
            .unwrap()
            .find(|f| f.center_of_mass().unwrap().x > 9.0)
            .unwrap();
        is_operation(holed.without_faces([&side]));
    }

    #[test]
    fn unifying_and_healing() {
        let a = Cuboid::new([1.0, 1.0, 1.0]).build().unwrap();
        let b = Cuboid::new([1.0, 1.0, 1.0])
            .at([1.0, 0.0, 0.0])
            .build()
            .unwrap();
        let block = a.fuse(&b).unwrap().unified().unwrap();
        assert_eq!(block.shape.faces().unwrap().len(), 6);
        assert_eq!(block.shape.edges().unwrap().len(), 12);
        near(block.shape.volume().unwrap(), 2.0);

        near(block.shape.healed(1e-4).unwrap().volume().unwrap(), 2.0);
        for bad in [0.0, -1.0, f64::NAN] {
            is_invalid_input(a.healed(bad));
        }
    }

    #[test]
    fn transforms() {
        let block = Cuboid::new([1.0, 2.0, 3.0])
            .at([1.0, 0.0, 0.0])
            .build()
            .unwrap();
        let stretch = Transform::scale(Point3::ORIGIN, Vec3::new(2.0, 3.0, 4.0));
        near(
            block.transformed(stretch).unwrap().volume().unwrap(),
            6.0 * 24.0,
        );

        let mirrored = block.mirrored(Plane::YZ).unwrap();
        near(mirrored.volume().unwrap(), 6.0);
        near(mirrored.center_of_mass().unwrap().x, -1.5);
        assert_ne!(mirrored, *block);

        let flat = Transform::scale(Point3::ORIGIN, Vec3::new(1.0, 0.0, 1.0));
        is_invalid_input(block.transformed(flat));
        let mut broken = Transform::IDENTITY;
        broken.rows[1][2] = f64::NAN;
        is_invalid_input(block.transformed(broken));
        is_invalid_input(block.placed(broken));
        is_invalid_input(block.mirrored(Plane::new(Point3::new(f64::NAN, 0.0, 0.0), Dir3::X)));
    }

    #[test]
    fn placing() {
        let block = Cuboid::new([1.0, 2.0, 3.0]).build().unwrap();
        let by = Transform::translation(Vec3::new(5.0, 0.0, 0.0));
        let instance = block.placed(by).unwrap();
        near(instance.center_of_mass().unwrap().x, 5.5);
        assert_ne!(instance, *block);
        assert_eq!(instance.faces().unwrap().len(), 6);

        // A chain of rotations stays a placement despite rounding.
        let turn = Transform::rotation(
            Axis::new([1.0, 2.0, 3.0].into(), Dir3::new([1.0, 1.0, 1.0]).unwrap()),
            Angle::degrees(7.3),
        );
        let chain = (0..100).fold(Transform::IDENTITY, |t, _| turn.then_after(t));
        near(block.placed(chain).unwrap().volume().unwrap(), 6.0);

        is_invalid_input(block.placed(Transform::scale(Point3::ORIGIN, Vec3::new(2.0, 2.0, 2.0))));
        is_invalid_input(block.placed(Transform::scale(Point3::ORIGIN, Vec3::new(1.0, 1.0, 2.0))));
        is_invalid_input(block.placed(Transform::mirror(Plane::XY)));
    }

    #[test]
    fn reversing() {
        let face = square(1.0, 0.0);
        let flipped = face.reversed().unwrap();
        assert_eq!(flipped, *face);
        near(
            flipped.thickened(1.0).unwrap().center_of_mass().unwrap().z,
            -0.5,
        );
        near(
            flipped
                .reversed()
                .unwrap()
                .thickened(1.0)
                .unwrap()
                .center_of_mass()
                .unwrap()
                .z,
            0.5,
        );

        let solid = Cuboid::new([1.0, 1.0, 1.0]).build().unwrap();
        near(solid.reversed().unwrap().volume().unwrap(), -1.0);
        near(
            solid
                .reversed()
                .unwrap()
                .healed(1e-4)
                .unwrap()
                .volume()
                .unwrap(),
            1.0,
        );
    }
}
