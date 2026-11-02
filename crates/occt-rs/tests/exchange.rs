//! The exchange domain: the kernel checks, and the public API with its error paths.

use std::process::Command;

use occt::prelude::*;
use occt::{
    Assembly, CancelFlag, Color, Document, LengthUnit, NodeContent, StepOptions, StepSchema,
    Transform,
};
use occt_kernel_capi::CapiKernel;

fn check(name: &str) {
    let (_, run) = occt_conformance::exchange::checks::<CapiKernel>()
        .into_iter()
        .find(|(n, _)| *n == name)
        .unwrap_or_else(|| panic!("no check named {name}"));
    run();
}

macro_rules! conformance {
    ($($name:ident),* $(,)?) => {
        $(#[test] fn $name() { check(stringify!($name)); })*

        #[test]
        fn every_check_has_a_test() {
            let listed = [$(stringify!($name)),*];
            for (name, _) in occt_conformance::exchange::checks::<CapiKernel>() {
                assert!(listed.contains(&name), "{name} is not run by this file");
            }
        }
    };
}

conformance![
    native_round_trip_keeps_everything,
    native_round_trip_keeps_placement,
    native_rejects_garbage,
    native_rejects_damage,
    step_document_of_one_part,
    step_document_keeps_a_flat_list_of_roots,
    step_document_two_instances_share_a_part,
    step_document_keeps_colors,
    step_document_colors_instances_of_one_part_apart,
    step_document_nested_placements_accumulate,
    step_document_places_top_level_nodes,
    step_document_schemas,
    step_document_schema_does_not_stick,
    step_document_reports_and_converts_units,
    step_document_reads_a_plain_file,
    step_document_reads_colors_of_a_plain_file,
    step_document_rejects_garbage,
    step_document_rejects_bad_trees,
    step_document_honours_a_set_cancel_flag,
    iges_round_trip_makes_a_solid,
    iges_round_trip_of_curved_solids,
    iges_holds_several_shapes,
    iges_rejects_garbage,
];

fn block() -> Solid {
    Cuboid::new([1.0, 2.0, 3.0]).build().unwrap()
}

fn invalid<T: std::fmt::Debug>(result: occt::Result<T>) {
    let err = result.unwrap_err();
    assert!(matches!(err, Error::InvalidInput(_)), "{err}");
}

fn shift(x: f64, y: f64, z: f64) -> Transform {
    Transform::translation([x, y, z].into())
}

// --- documents, public API -----------------------------------------------------------

#[test]
fn the_builder_round_trips_a_tree() {
    let part = block();
    let other = Sphere::new(2.0).build().unwrap();
    let document = Document::new()
        .part("loose", &other)
        .color(Color::GREEN)
        .assembly(
            Assembly::new("outer")
                .part("first", &part)
                .color(Color::RED)
                .at(shift(10.0, 0.0, 0.0))
                .assembly(
                    Assembly::new("inner")
                        .part("second", &part)
                        .color(Color::RED)
                        .at(shift(0.0, 5.0, 0.0)),
                )
                .at(shift(0.0, 0.0, 7.0)),
        );

    let step = document.to_step(StepOptions::new()).unwrap();
    let back = Document::from_step(&step).unwrap();
    assert_eq!(back.file_units_per_metre(), Some(1000.0));
    assert_eq!(back.roots().len(), 2);
    assert_eq!(back.roots()[0].name(), Some("loose"));

    let parts: Vec<_> = back.parts().collect();
    assert_eq!(parts.len(), 3);
    assert_eq!(back.parts().len(), 3);
    assert_eq!(parts[0].path(), ["loose"]);
    assert_eq!(parts[1].path(), ["outer", "first"]);
    assert_eq!(parts[2].path(), ["outer", "inner", "second"]);
    assert_eq!(parts[2].full_name(), "outer/inner/second");

    assert_eq!(parts[1].shape(), parts[2].shape());
    assert_ne!(parts[0].shape(), parts[1].shape());
    assert_eq!(parts[1].placement().rows[0][3], 10.0);
    let second = parts[2].placement();
    assert_eq!(
        [second.rows[0][3], second.rows[1][3], second.rows[2][3]],
        [0.0, 5.0, 7.0]
    );

    assert_eq!(parts[0].color(), Some(Color::GREEN));
    assert_eq!(parts[1].color(), Some(Color::RED));
    assert_eq!(parts[2].color(), Some(Color::RED));
    assert!(back.parts().next_back().is_some());
}

#[test]
fn nodes_expose_their_content() {
    let part = block();
    let document = Document::new().assembly(Assembly::new("a").part("p", &part));
    let root = &document.roots()[0];
    assert!(root.shape().is_none());
    assert_eq!(root.name(), Some("a"));
    assert_eq!(root.color(), None);
    assert!(matches!(root.content(), NodeContent::Assembly(_)));
    let child = &root.children()[0];
    assert!(child.children().is_empty());
    assert_eq!(child.shape(), Some(&*part));
    assert_eq!(child.placement(), Transform::IDENTITY);
}

#[test]
fn an_assembly_colour_reaches_its_parts() {
    let part = block();
    let document = Document::new()
        .assembly(
            Assembly::new("a")
                .part("p", &part)
                .part("q", &part)
                .color(Color::BLUE)
                .at(shift(5.0, 0.0, 0.0)),
        )
        .color(Color::RED);
    let parts: Vec<_> = document.parts().collect();
    assert_eq!(parts[0].color(), Some(Color::RED));
    assert_eq!(parts[1].color(), Some(Color::BLUE));

    let back = Document::from_step(&document.to_step(StepOptions::new()).unwrap()).unwrap();
    let parts: Vec<_> = back.parts().collect();
    assert_eq!(parts[0].color(), Some(Color::RED));
    assert_eq!(parts[1].color(), Some(Color::BLUE));
    assert_eq!(parts[0].shape(), parts[1].shape());
}

#[test]
fn a_part_can_be_a_face() {
    let face: Face = block().faces().unwrap().next().unwrap();
    let area = face.area().unwrap();
    let document = Document::new().part("sheet", &face);
    let back = Document::from_step(&document.to_step(StepOptions::new()).unwrap()).unwrap();
    let part = back.parts().next().unwrap();
    assert!((part.shape().area().unwrap() - area).abs() < 1e-6);
}

#[test]
fn identical_instances_at_one_place_stay_two() {
    let part = block();
    let document = Document::new().assembly(Assembly::new("a").part("x", &part).part("y", &part));
    let back = Document::from_step(&document.to_step(StepOptions::new()).unwrap()).unwrap();
    assert_eq!(back.parts().count(), 2);
}

#[test]
fn a_repeated_assembly_comes_back_with_both_instances() {
    let part = block();
    let inner = Assembly::new("axle").part("wheel", &part);
    let document = Document::new().assembly(
        Assembly::new("car")
            .assembly(inner.clone())
            .at(shift(1.0, 0.0, 0.0))
            .assembly(inner)
            .at(shift(2.0, 0.0, 0.0)),
    );
    let back = Document::from_step(&document.to_step(StepOptions::new()).unwrap()).unwrap();
    let parts: Vec<_> = back.parts().collect();
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0].placement().rows[0][3], 1.0);
    assert_eq!(parts[1].placement().rows[0][3], 2.0);
    assert_eq!(parts[0].shape(), parts[1].shape());
    // One product in the file: the second instance doesn't repeat the axle.
    let text = String::from_utf8_lossy(&document.to_step(StepOptions::new()).unwrap()).into_owned();
    assert_eq!(text.matches("PRODUCT('axle'").count(), 1, "{text}");
}

#[test]
fn names_survive_awkward_characters() {
    let part = block();
    for name in [
        "with space",
        "it's",
        "a/b",
        "back\\slash",
        "ÅngströmÜñí",
        "部品",
    ] {
        let document = Document::new().assembly(
            Assembly::new("frame")
                .part(name, &part)
                .at(shift(1.0, 0.0, 0.0)),
        );
        let back = Document::from_step(&document.to_step(StepOptions::new()).unwrap()).unwrap();
        assert_eq!(back.parts().next().unwrap().path(), ["frame", name]);
    }
}

#[test]
fn a_part_with_no_name_is_not_named_by_accident() {
    let part = block();
    // Only reading can make a nameless node: file parts with empty names.
    let document = Document::new().part("", &part);
    let back = Document::from_step(&document.to_step(StepOptions::new()).unwrap()).unwrap();
    let root = &back.roots()[0];
    assert_eq!(root.name(), None, "got {:?}", root.name());
}

#[test]
fn read_step_document_reads_a_stream() {
    let document = Document::new().part("p", block());
    let step = document.to_step(StepOptions::new()).unwrap();
    let back = occt::read_step_document(step.as_slice()).unwrap();
    assert_eq!(back.roots()[0].name(), Some("p"));

    struct Broken;
    impl std::io::Read for Broken {
        fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
            Err(std::io::Error::other("broken"))
        }
    }
    assert!(matches!(
        occt::read_step_document(Broken),
        Err(Error::Io(_))
    ));
}

#[test]
fn write_step_writes_to_a_stream() {
    let document = Document::new().part("p", block());
    let mut bytes = Vec::new();
    document.write_step(&mut bytes, StepOptions::new()).unwrap();
    assert!(bytes.starts_with(b"ISO-10303-21"));
}

#[test]
fn units_can_be_chosen_for_writing() {
    let part = Cuboid::new([1000.0, 2000.0, 3000.0]).build().unwrap();
    let document = Document::new().assembly(
        Assembly::new("a")
            .part("p", &part)
            .at(shift(1000.0, 0.0, 0.0))
            .part("q", &part)
            .at(shift(0.0, 2000.0, 0.0)),
    );
    for unit in [
        LengthUnit::Millimetre,
        LengthUnit::Metre,
        LengthUnit::Centimetre,
        LengthUnit::Inch,
        LengthUnit::Foot,
    ] {
        let step = document.to_step(StepOptions::new().unit(unit)).unwrap();
        let back = Document::from_step(&step).unwrap();
        let reported = back.file_units_per_metre().unwrap();
        assert!(
            (reported / unit.per_metre() - 1.0).abs() < 1e-6,
            "{unit}: got {reported}"
        );
        let parts: Vec<_> = back.parts().collect();
        let volume = parts[0].shape().volume().unwrap();
        assert!((volume / 6e9 - 1.0).abs() < 1e-6, "{unit}: volume {volume}");
        assert!(
            (parts[0].placement().rows[0][3] - 1000.0).abs() < 1e-5,
            "{unit}"
        );
        assert!(
            (parts[1].placement().rows[1][3] - 2000.0).abs() < 1e-5,
            "{unit}"
        );
    }
    let metres = document
        .to_step(StepOptions::new().unit(LengthUnit::Metre))
        .unwrap();
    assert!(String::from_utf8_lossy(&metres).contains(".METRE."));
    assert!(!String::from_utf8_lossy(&metres).contains(".MILLI."));
}

#[test]
fn bad_documents_are_invalid_input() {
    let part = block();
    let options = StepOptions::new;
    invalid(Document::new().to_step(options()));
    invalid(
        Document::new()
            .color(Color::RED)
            .part("p", &part)
            .to_step(options()),
    );
    invalid(
        Document::new()
            .at(Transform::IDENTITY)
            .part("p", &part)
            .to_step(options()),
    );
    invalid(
        Document::new()
            .assembly(Assembly::new("a").color(Color::RED).part("p", &part))
            .to_step(options()),
    );
    invalid(
        Document::new()
            .assembly(Assembly::new("empty"))
            .to_step(options()),
    );
    invalid(Document::new().part("nul\0name", &part).to_step(options()));
    for color in [
        Color::rgb(f32::NAN, 0.0, 0.0),
        Color::rgb(1.5, 0.0, 0.0),
        Color::rgb(0.0, -0.1, 0.0),
        Color::rgb(0.0, 0.0, f32::INFINITY),
    ] {
        invalid(
            Document::new()
                .part("p", &part)
                .color(color)
                .to_step(options()),
        );
    }
    let mut nan = Transform::IDENTITY;
    nan.rows[0][3] = f64::NAN;
    let mut zero = Transform::IDENTITY;
    zero.rows[0][0] = 0.0;
    let scale = Transform::scale(Point3::ORIGIN, [2.0, 2.0, 2.0].into());
    let mirror = Transform::scale(Point3::ORIGIN, [-1.0, 1.0, 1.0].into());
    let mut shear = Transform::IDENTITY;
    shear.rows[0][1] = 0.5;
    for placement in [nan, zero, scale, mirror, shear] {
        invalid(
            Document::new()
                .part("p", &part)
                .at(placement)
                .to_step(options()),
        );
    }

    let mut deep = Assembly::new("leaf").part("p", &part);
    for _ in 0..300 {
        deep = Assembly::new("level").assembly(deep);
    }
    invalid(Document::new().assembly(deep).to_step(options()));
}

#[test]
fn deep_trees_within_the_limit_work() {
    let part = block();
    let mut deep = Assembly::new("leaf").part("p", &part);
    for _ in 0..40 {
        deep = Assembly::new("level").assembly(deep);
    }
    let step = Document::new()
        .assembly(deep)
        .to_step(StepOptions::new())
        .unwrap();
    let back = Document::from_step(&step).unwrap();
    assert_eq!(back.parts().next().unwrap().path().len(), 42);
}

#[test]
fn a_part_can_be_a_vertex_or_an_edge() {
    let part = block();
    let vertex: Vertex = part.vertices().unwrap().next().unwrap();
    let edge: Edge = part.edges().unwrap().next().unwrap();
    for (name, shape) in [("point", &*vertex), ("line", &*edge)] {
        let step = Document::new()
            .part(name, shape)
            .to_step(StepOptions::new())
            .unwrap();
        assert_eq!(Document::from_step(&step).unwrap().parts().count(), 1);
    }
}

// --- cancellation ---------------------------------------------------------------------

#[test]
fn cancelling_reads_and_writes() {
    let part = block();
    let document = Document::new().part("p", &part);
    let flag = CancelFlag::new();
    flag.cancel();
    assert!(matches!(
        document.to_step(StepOptions::new().cancel(&flag)),
        Err(Error::Cancelled)
    ));
    let step = document.to_step(StepOptions::new()).unwrap();
    assert!(matches!(
        Document::from_step_cancellable(&step, &flag),
        Err(Error::Cancelled)
    ));
    assert_eq!(
        Document::from_step_cancellable(&step, &CancelFlag::new())
            .unwrap()
            .roots()
            .len(),
        1
    );
}

/// A read in progress stops part way, not only when the flag was set beforehand.
#[test]
fn cancelling_a_read_in_progress() {
    let shapes: Vec<Solid> = (1..200)
        .map(|i| Cuboid::new([1.0, 1.0, f64::from(i)]).build().unwrap())
        .collect();
    let mut document = Document::new();
    for (i, shape) in shapes.iter().enumerate() {
        document = document.part(format!("part {i}"), shape);
    }
    let step = document.to_step(StepOptions::new()).unwrap();
    drop(document);
    drop(shapes);

    let start = std::time::Instant::now();
    assert_eq!(Document::from_step(&step).unwrap().roots().len(), 199);
    let whole = start.elapsed();

    // The transfer is the larger part of a read, so one of these lands inside it.
    let mut stopped = false;
    for eighth in 1..8 {
        let flag = CancelFlag::new();
        let setter = {
            let flag = flag.clone();
            std::thread::spawn(move || {
                std::thread::sleep(whole * eighth / 8);
                flag.cancel();
            })
        };
        let result = Document::from_step_cancellable(&step, &flag);
        setter.join().unwrap();
        match result {
            Err(Error::Cancelled) => stopped = true,
            Ok(document) => assert_eq!(document.roots().len(), 199),
            Err(other) => panic!("{other}"),
        }
    }
    assert!(stopped, "no read stopped after a delay of up to {whole:?}");
}

// --- native ---------------------------------------------------------------------------

#[test]
fn native_round_trip_is_exact() {
    for size in [
        [1.0, 2.0, 3.0],
        [1.0 / 3.0, 7.0 / 11.0, 13.0 / 17.0],
        [1e-3, 1e4, 0.1],
    ] {
        let part = Cuboid::new(size)
            .at([1.0 / 7.0, -3.3, 1e3 / 3.0])
            .build()
            .unwrap();
        let saved = part.to_native().unwrap();
        let back = Shape::from_native(&saved).unwrap();
        assert_eq!(back.kind().unwrap(), ShapeKind::Solid);
        assert_eq!(back.volume().unwrap(), part.volume().unwrap(), "{size:?}");
        assert_eq!(back.area().unwrap(), part.area().unwrap(), "{size:?}");
        assert_eq!(back.faces().unwrap().len(), 6);
        assert_eq!(back.edges().unwrap().len(), 12);
        assert_eq!(back.vertices().unwrap().len(), 8);
        // And saving again gives the same bytes.
        assert_eq!(back.to_native().unwrap(), saved);
    }
}

#[test]
fn native_has_no_mesh_in_it() {
    let part = Sphere::new(5.0).build().unwrap();
    let bare = part.to_native().unwrap().len();
    part.mesh(MeshOptions::default().absolute(0.01)).unwrap();
    assert_eq!(part.to_native().unwrap().len(), bare);
}

#[test]
fn native_garbage_is_an_error() {
    assert!(Shape::from_native(b"").is_err());
    assert!(Shape::from_native(b"OCCTSAV1").is_err());
    assert!(Shape::from_native(&[b'O'; 64]).is_err());
    assert!(Shape::from_native(&[0xff; 4096]).is_err());
}

// --- IGES -----------------------------------------------------------------------------

#[test]
fn iges_of_a_face_comes_back_as_a_face() {
    let face: Face = block().faces().unwrap().next().unwrap();
    let back = Shape::from_iges(&face.to_iges().unwrap()).unwrap();
    assert_eq!(back.kind().unwrap(), ShapeKind::Face);
    assert!((back.area().unwrap() - face.area().unwrap()).abs() < 1e-5);
}

#[test]
fn iges_holds_vertices_and_edges() {
    let part = block();
    let vertex: Vertex = part.vertices().unwrap().next().unwrap();
    let edge: Edge = part.edges().unwrap().next().unwrap();
    let back = Shape::from_iges(&vertex.to_iges().unwrap()).unwrap();
    assert_eq!(back.kind().unwrap(), ShapeKind::Vertex);
    let back = Shape::from_iges(&edge.to_iges().unwrap()).unwrap();
    assert_eq!(back.kind().unwrap(), ShapeKind::Edge);
}

#[test]
fn iges_files_do_not_name_the_user() {
    let iges = block().to_iges().unwrap();
    let text = String::from_utf8_lossy(&iges);
    for variable in ["USER", "LOGNAME", "USERNAME"] {
        if let Ok(name) = std::env::var(variable)
            && name.len() > 2
        {
            assert!(!text.contains(&name), "{name} in {text}");
        }
    }
    assert_eq!(
        Shape::from_iges(&iges).unwrap().kind().unwrap(),
        ShapeKind::Solid
    );
}

#[test]
fn iges_of_a_plate_with_a_hole_closes() {
    let plate = Cuboid::new([10.0, 8.0, 2.0]).build().unwrap();
    let axis = Axis::new([5.0, 4.0, -1.0].into(), Dir3::Z);
    let drill = Cylinder::new(1.5, 4.0).along(axis).build().unwrap();
    let holed = (&plate - &drill).unwrap();
    let back = Shape::from_iges(&holed.to_iges().unwrap()).unwrap();
    assert_eq!(back.kind().unwrap(), ShapeKind::Solid);
    assert!((back.volume().unwrap() / holed.volume().unwrap() - 1.0).abs() < 1e-6);
}

#[test]
fn iges_keeps_a_solid_and_a_lone_face_apart() {
    let solid = block();
    let moved = Cuboid::new([5.0, 5.0, 1.0])
        .at([50.0, 0.0, 0.0])
        .build()
        .unwrap();
    let face: Face = moved.faces().unwrap().next().unwrap();
    let back = Shape::from_iges(&occt::write_iges([&*solid as &Shape, &*face as &Shape]).unwrap())
        .unwrap();
    assert_eq!(back.kind().unwrap(), ShapeKind::Compound);
    assert_eq!(back.solids().unwrap().len(), 1);
    assert_eq!(back.faces().unwrap().len(), 7);
}

#[test]
fn iges_garbage_is_an_error() {
    assert!(Shape::from_iges(b"").is_err());
    assert!(Shape::from_iges(&[0xff; 4096]).is_err());
    assert!(occt::write_iges(Vec::<Shape>::new()).is_err());
    assert!(matches!(
        occt::write_iges(Vec::<Shape>::new()),
        Err(Error::InvalidInput(_))
    ));
}

#[test]
fn iges_temporary_files_are_cleaned_up() {
    let iges = block().to_iges().unwrap();
    let before = temp_files();
    Shape::from_iges(&iges).unwrap();
    assert!(Shape::from_iges(b"garbage").is_err());
    assert_eq!(temp_files(), before);
}

fn temp_files() -> usize {
    std::fs::read_dir(std::env::temp_dir())
        .unwrap()
        .filter(|e| {
            let name = e.as_ref().unwrap().file_name();
            let name = name.to_string_lossy();
            name.starts_with("occt-") && name.ends_with(".igs")
        })
        .count()
}

// --- the environment ------------------------------------------------------------------

/// Exercises everything once, printing nothing itself. Run alone by the test below.
#[test]
#[ignore = "run by no_output_and_no_environment_needed, in a child process"]
fn exercise_everything_quietly() {
    let part = block();
    let document = Document::new().assembly(
        Assembly::new("a")
            .part("p", &part)
            .color(Color::RED)
            .part("q", &part),
    );
    for schema in [StepSchema::Ap203, StepSchema::Ap214, StepSchema::Ap242] {
        let step = document.to_step(StepOptions::new().schema(schema)).unwrap();
        assert_eq!(Document::from_step(&step).unwrap().parts().count(), 2);
    }
    assert!(Document::from_step(b"garbage").is_err());
    let saved = part.to_native().unwrap();
    assert!(Shape::from_native(&saved).is_ok());
    assert!(Shape::from_native(b"garbage").is_err());
    assert!(Shape::from_native(&saved[..saved.len() / 2]).is_err());
    let iges = part.to_iges().unwrap();
    assert_eq!(
        Shape::from_iges(&iges).unwrap().kind().unwrap(),
        ShapeKind::Solid
    );
    assert!(Shape::from_iges(b"garbage").is_err());
}

/// Nothing above writes to stdout, and it all works with no CASROOT or the like set.
#[test]
fn no_output_and_no_environment_needed() {
    let exe = std::env::current_exe().unwrap();
    let mut command = Command::new(exe);
    command
        .args([
            "--exact",
            "exercise_everything_quietly",
            "--ignored",
            "--nocapture",
            "-q",
        ])
        .env_clear();
    for name in [
        "LD_LIBRARY_PATH",
        "DYLD_FALLBACK_LIBRARY_PATH",
        "PATH",
        "TMPDIR",
        "TEMP",
        "TMP",
    ] {
        if let Some(value) = std::env::var_os(name) {
            command.env(name, value);
        }
    }
    let output = command.output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stdout}\n{stderr}");
    // What the test harness itself prints starts with these; the rest isn't its own.
    let foreign: Vec<&str> = stdout
        .lines()
        .filter(|line| !line.trim().is_empty())
        .filter(|line| {
            !(line.starts_with("running 1 test")
                || line.starts_with("test result:")
                || line.trim() == ".")
        })
        .collect();
    assert!(foreign.is_empty(), "printed to stdout: {foreign:#?}");
    assert!(stderr.trim().is_empty(), "printed to stderr: {stderr}");
}
