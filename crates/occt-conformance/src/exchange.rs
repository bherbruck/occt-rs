//! Checks for Exchange: native saves, STEP documents and IGES.
//!
//! Expected answers come from the construction of each input, and every file a check
//! reads that it did not itself write is generated here as plain text.

use std::f64::consts::PI;
use std::fmt::Write as _;

use occt_kernel::{
    Axis, CancelFlag, Color, Dir3, Document, DocumentNode, Error, Exchange, Kernel, NodeContent,
    Point3, ShapeKind, StepSchema, StepWriteOptions, Transform, Vec3,
};

use crate::support::{MEASURE, ball_volume, cuboid, must, near, near_point, parts, volume_is};

/// Colours are 8-bit in practice, so a file may keep them no better than one step of 255.
const COLOR: f32 = 1.0 / 255.0;

/// Native saves are lossless, so what's computed from one is what was computed before,
/// to within a few units of rounding in the computation itself.
const NATIVE: f64 = 1e-14;

/// The text of a Part 21 file's schema declaration, which may span lines.
fn declared_schema(bytes: &[u8]) -> Option<String> {
    let text = String::from_utf8_lossy(bytes);
    let (_, rest) = text.split_once("FILE_SCHEMA")?;
    let (schema, _) = rest.split_once(';')?;
    Some(schema.to_owned())
}

/// Every check of this module as a `(name, function)` pair.
#[must_use]
pub fn checks<K: Kernel + Exchange>() -> Vec<(&'static str, fn())> {
    macro_rules! list {
        ($($name:ident),* $(,)?) => {
            vec![$((stringify!($name), $name::<K> as fn())),*]
        };
    }
    list![
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
    ]
}

/// A node with a name and no colour.
fn node<S>(name: &str, placement: Transform, content: NodeContent<S>) -> DocumentNode<S> {
    DocumentNode {
        name: Some(name.to_owned()),
        color: None,
        placement,
        content,
    }
}

fn document<S>(roots: Vec<DocumentNode<S>>) -> Document<S> {
    Document {
        roots,
        file_units_per_metre: None,
    }
}

#[track_caller]
fn round_trip<K: Exchange>(document: &Document<K::Shape>) -> Document<K::Shape> {
    let bytes = must(
        K::write_step_document(document, &StepWriteOptions::default()),
        "write_step_document",
    );
    must(K::read_step_document(&bytes, None), "read_step_document")
}

fn children<S>(node: &DocumentNode<S>) -> &[DocumentNode<S>] {
    match &node.content {
        NodeContent::Assembly(children) => children,
        NodeContent::Part(_) => panic!("expected an assembly, got a part"),
    }
}

fn shape_of<S>(node: &DocumentNode<S>) -> &S {
    match &node.content {
        NodeContent::Part(shape) => shape,
        NodeContent::Assembly(_) => panic!("expected a part, got an assembly"),
    }
}

#[track_caller]
fn near_color(got: Option<Color>, want: Color, what: &str) {
    let got = got.unwrap_or_else(|| panic!("{what}: the colour was lost"));
    assert!(
        (got.r - want.r).abs() <= COLOR
            && (got.g - want.g).abs() <= COLOR
            && (got.b - want.b).abs() <= COLOR,
        "{what}: got {got:?}, want {want:?}"
    );
}

/// The world position of a part's centre, from its shape and placement.
fn center_at<K: Kernel>(shape: &K::Shape, placement: Transform) -> Point3 {
    placement.point(must(K::center_of_mass(shape), "center_of_mass"))
}

/// A sweep of shapes with an analytic volume, solid count and face count.
fn sample_shapes<K: Kernel>() -> Vec<(&'static str, K::Shape, f64, usize, usize)> {
    let third = 1.0 / 3.0;
    let block = cuboid::<K>(third, 7.0 / 11.0, 13.0 / 17.0);
    let cylinder = must(K::make_cylinder(Axis::Z, 1.5, 2.5), "make_cylinder");
    let ball = must(K::make_sphere(Point3::ORIGIN, 0.7), "make_sphere");
    let plate = cuboid::<K>(10.0, 8.0, 2.0);
    let drill = must(
        K::make_cylinder(Axis::new(Point3::new(5.0, 4.0, -1.0), Dir3::Z), 1.5, 4.0),
        "make_cylinder",
    );
    let holed = must(K::cut(&plate, &drill), "cut");
    vec![
        ("box", block, third * 7.0 / 11.0 * 13.0 / 17.0, 1, 6),
        ("cylinder", cylinder, PI * 1.5 * 1.5 * 2.5, 1, 3),
        ("sphere", ball, ball_volume(0.7), 1, 1),
        (
            "plate with a hole",
            holed,
            10.0 * 8.0 * 2.0 - PI * 1.5 * 1.5 * 2.0,
            1,
            7,
        ),
    ]
}

/// A native save brings back the same measures and the same number of every subshape.
///
/// # Panics
/// If the round trip changes anything.
pub fn native_round_trip_keeps_everything<K: Kernel + Exchange>() {
    for (name, shape, volume, solids, faces) in sample_shapes::<K>() {
        let bytes = must(K::write_native(&shape), name);
        let back = must(K::read_native(&bytes), name);
        assert!(
            must(K::is_valid(&back), name),
            "{name}: invalid after a save"
        );
        near(must(K::volume(&back), name), volume, NATIVE, name);
        near(
            must(K::volume(&back), name),
            must(K::volume(&shape), name),
            NATIVE,
            name,
        );
        assert_eq!(
            must(K::kind(&back), name),
            must(K::kind(&shape), name),
            "{name}"
        );
        assert_eq!(parts::<K>(&back, ShapeKind::Solid).len(), solids, "{name}");
        assert_eq!(parts::<K>(&back, ShapeKind::Face).len(), faces, "{name}");
        for kind in [
            ShapeKind::Shell,
            ShapeKind::Wire,
            ShapeKind::Edge,
            ShapeKind::Vertex,
        ] {
            assert_eq!(
                parts::<K>(&back, kind).len(),
                parts::<K>(&shape, kind).len(),
                "{name}: number of {kind}s"
            );
        }
    }
}

/// A moved shape stays where it was, and shapes that aren't solids survive too.
///
/// # Panics
/// If the shape moves or changes kind.
pub fn native_round_trip_keeps_placement<K: Kernel + Exchange>() {
    let by = Vec3::new(1.0 / 3.0, -2.0 / 7.0, 1e4 / 3.0);
    let moved = must(K::translate(&cuboid::<K>(1.0, 1.0, 1.0), by), "translate");
    let back = must(
        K::read_native(&must(K::write_native(&moved), "write_native")),
        "read_native",
    );
    near_point(
        must(K::center_of_mass(&back), "com"),
        Point3::new(0.5, 0.5, 0.5) + by,
        "centre after a native save",
    );

    let face = parts::<K>(&cuboid::<K>(2.0, 3.0, 4.0), ShapeKind::Face)
        .into_iter()
        .next()
        .expect("a box has faces");
    let back = must(
        K::read_native(&must(K::write_native(&face), "write_native")),
        "read_native",
    );
    assert_eq!(must(K::kind(&back), "kind"), ShapeKind::Face);
    near(
        must(K::area(&back), "area"),
        must(K::area(&face), "area"),
        NATIVE,
        "area of a saved face",
    );
}

/// Bytes that aren't a save are an error, and text from other formats is not a save.
///
/// # Panics
/// If the kernel returns a shape, or panics.
pub fn native_rejects_garbage<K: Kernel + Exchange>() {
    let step = must(K::write_step(&cuboid::<K>(1.0, 1.0, 1.0)), "write_step");
    for (name, data) in [
        ("empty input", &b""[..]),
        ("arbitrary text", &b"this is not a saved shape"[..]),
        ("arbitrary bytes", &[0xff, 0x00, 0xfe, 0x01, 0x80, 0x7f][..]),
        ("a STEP file", &step[..]),
    ] {
        assert!(
            K::read_native(data).is_err(),
            "read_native of {name} must return Err"
        );
    }
}

/// A save cut short or changed anywhere is an error, not a wrong shape or a crash.
///
/// # Panics
/// If a damaged save gives a shape, or the kernel panics.
pub fn native_rejects_damage<K: Kernel + Exchange>() {
    let bytes = must(K::write_native(&cuboid::<K>(1.0, 2.0, 3.0)), "write_native");
    for percent in [1, 5, 10, 25, 50, 75, 90, 95] {
        let cut = bytes.len() * percent / 100;
        assert!(
            K::read_native(&bytes[..cut]).is_err(),
            "reading the first {percent}% of a save must return Err"
        );
    }
    for at in (0..bytes.len()).step_by(bytes.len() / 500 + 1) {
        let mut damaged = bytes.clone();
        damaged[at] ^= 0x41;
        assert!(
            K::read_native(&damaged).is_err(),
            "a save with byte {at} changed must return Err"
        );
    }
}

/// One part comes back as one root part with its name, measure and identity placement.
///
/// # Panics
/// If the document differs.
pub fn step_document_of_one_part<K: Kernel + Exchange>() {
    let block = cuboid::<K>(1.0, 2.0, 3.0);
    let back = round_trip::<K>(&document(vec![node(
        "block",
        Transform::IDENTITY,
        NodeContent::Part(block),
    )]));
    assert_eq!(back.roots.len(), 1, "number of roots");
    let root = &back.roots[0];
    assert_eq!(root.name.as_deref(), Some("block"));
    volume_is::<K>(shape_of(root), 6.0, "part from a document");
    assert_eq!(
        parts::<K>(shape_of(root), ShapeKind::Face).len(),
        6,
        "faces of a part from a document"
    );
    near_point(
        center_at::<K>(shape_of(root), root.placement),
        Point3::new(0.5, 1.0, 1.5),
        "the part's centre",
    );
}

/// Unrelated parts stay separate roots, in order, with their names.
///
/// # Panics
/// If the roots differ.
pub fn step_document_keeps_a_flat_list_of_roots<K: Kernel + Exchange>() {
    let sizes = [1.0, 2.0, 3.0];
    let names = ["small", "medium", "large"];
    let roots = sizes
        .iter()
        .zip(names)
        .map(|(&side, name)| {
            node(
                name,
                Transform::IDENTITY,
                NodeContent::Part(cuboid::<K>(side, side, side)),
            )
        })
        .collect();
    let back = round_trip::<K>(&document(roots));
    assert_eq!(back.roots.len(), 3, "number of roots");
    for ((root, side), name) in back.roots.iter().zip(sizes).zip(names) {
        assert_eq!(root.name.as_deref(), Some(name));
        volume_is::<K>(shape_of(root), side * side * side, name);
    }
}

/// Two instances keep their names, placements and colour, and refer to one part.
///
/// # Panics
/// If any of that is lost.
pub fn step_document_two_instances_share_a_part<K: Kernel + Exchange>() {
    let bolt = cuboid::<K>(1.0, 2.0, 3.0);
    let turned = Transform::rotation(
        Axis::new(Point3::new(10.0, 0.0, 0.0), Dir3::Z),
        occt_kernel::Angle::degrees(90.0),
    );
    let color = Color {
        r: 0.25,
        g: 0.5,
        b: 0.75,
    };
    let left = DocumentNode {
        color: Some(color),
        ..node(
            "left",
            Transform::translation(Vec3::new(100.0, 0.0, 0.0)),
            NodeContent::Part(bolt.clone()),
        )
    };
    let right = DocumentNode {
        color: Some(color),
        ..node("right", turned, NodeContent::Part(bolt))
    };
    let back = round_trip::<K>(&document(vec![node(
        "frame",
        Transform::IDENTITY,
        NodeContent::Assembly(vec![left, right]),
    )]));

    assert_eq!(back.roots.len(), 1, "number of roots");
    assert_eq!(back.roots[0].name.as_deref(), Some("frame"));
    let instances = children(&back.roots[0]);
    assert_eq!(instances.len(), 2, "number of instances");
    assert_eq!(instances[0].name.as_deref(), Some("left"));
    assert_eq!(instances[1].name.as_deref(), Some("right"));
    let (a, b) = (shape_of(&instances[0]), shape_of(&instances[1]));
    assert!(K::is_same(a, b), "the instances must refer to one part");
    volume_is::<K>(a, 6.0, "the shared part");
    near_color(instances[0].color, color, "left");
    near_color(instances[1].color, color, "right");

    let center = Point3::new(0.5, 1.0, 1.5);
    near_point(
        center_at::<K>(a, instances[0].placement),
        Transform::translation(Vec3::new(100.0, 0.0, 0.0)).point(center),
        "left's centre",
    );
    near_point(
        center_at::<K>(b, instances[1].placement),
        turned.point(center),
        "right's centre",
    );
}

/// A sweep of colours on separate parts survives.
///
/// # Panics
/// If a colour is lost or changed.
pub fn step_document_keeps_colors<K: Kernel + Exchange>() {
    let colors = [
        Color {
            r: 1.0,
            g: 0.0,
            b: 0.0,
        },
        Color {
            r: 0.0,
            g: 1.0,
            b: 0.0,
        },
        Color {
            r: 0.0,
            g: 0.0,
            b: 1.0,
        },
        Color {
            r: 1.0,
            g: 1.0,
            b: 1.0,
        },
        Color {
            r: 0.0,
            g: 0.0,
            b: 0.0,
        },
        Color {
            r: 0.1,
            g: 0.2,
            b: 0.3,
        },
        Color {
            r: 0.9,
            g: 0.5,
            b: 0.05,
        },
    ];
    let roots = colors
        .iter()
        .enumerate()
        .map(|(i, &color)| DocumentNode {
            color: Some(color),
            ..node(
                &format!("part {i}"),
                Transform::IDENTITY,
                NodeContent::Part(cuboid::<K>(1.0, 1.0, 1.0 + i as f64)),
            )
        })
        .collect();
    let back = round_trip::<K>(&document(roots));
    assert_eq!(back.roots.len(), colors.len(), "number of roots");
    for (root, &color) in back.roots.iter().zip(&colors) {
        near_color(root.color, color, root.name.as_deref().unwrap_or("?"));
    }
}

/// Instances of one part can have colours of their own, and an uncoloured part has none.
///
/// # Panics
/// If the colours mix.
pub fn step_document_colors_instances_of_one_part_apart<K: Kernel + Exchange>() {
    let bolt = cuboid::<K>(1.0, 1.0, 1.0);
    let plain = cuboid::<K>(2.0, 2.0, 2.0);
    let red = Color {
        r: 1.0,
        g: 0.0,
        b: 0.0,
    };
    let blue = Color {
        r: 0.0,
        g: 0.0,
        b: 1.0,
    };
    let instance = |name: &str, x: f64, color| DocumentNode {
        color: Some(color),
        ..node(
            name,
            Transform::translation(Vec3::new(x, 0.0, 0.0)),
            NodeContent::Part(bolt.clone()),
        )
    };
    let back = round_trip::<K>(&document(vec![node(
        "frame",
        Transform::IDENTITY,
        NodeContent::Assembly(vec![
            instance("red", 0.0, red),
            instance("blue", 10.0, blue),
            node("plain", Transform::IDENTITY, NodeContent::Part(plain)),
        ]),
    )]));
    let instances = children(&back.roots[0]);
    assert!(
        K::is_same(shape_of(&instances[0]), shape_of(&instances[1])),
        "the instances must still refer to one part"
    );
    near_color(instances[0].color, red, "red");
    near_color(instances[1].color, blue, "blue");
    assert!(instances[2].color.is_none(), "the plain part has no colour");
}

/// Placements through three levels accumulate, and a repeated assembly's parts stay one.
///
/// # Panics
/// If a world position or the sharing is wrong.
pub fn step_document_nested_placements_accumulate<K: Kernel + Exchange>() {
    let wheel = cuboid::<K>(2.0, 2.0, 2.0);
    let axle = |name: &str, placement| {
        node(
            name,
            placement,
            NodeContent::Assembly(vec![
                node(
                    "wheel-l",
                    Transform::translation(Vec3::new(0.0, -5.0, 0.0)),
                    NodeContent::Part(wheel.clone()),
                ),
                node(
                    "wheel-r",
                    Transform::translation(Vec3::new(0.0, 5.0, 0.0)),
                    NodeContent::Part(wheel.clone()),
                ),
            ]),
        )
    };
    let front = Transform::translation(Vec3::new(30.0, 0.0, 0.0));
    let rear = Transform::translation(Vec3::new(-30.0, 0.0, 7.0));
    let top = Transform::translation(Vec3::new(0.0, 100.0, 0.0));
    let back = round_trip::<K>(&document(vec![node(
        "car",
        top,
        NodeContent::Assembly(vec![axle("front", front), axle("rear", rear)]),
    )]));

    assert_eq!(back.roots.len(), 1, "number of roots");
    let car = &back.roots[0];
    assert_eq!(car.name.as_deref(), Some("car"));
    let axles = children(car);
    assert_eq!(axles.len(), 2, "number of axles");
    assert_eq!(axles[0].name.as_deref(), Some("front"));
    assert_eq!(axles[1].name.as_deref(), Some("rear"));

    let center = Point3::new(1.0, 1.0, 1.0);
    let mut first = None;
    for (axle, axle_at) in axles.iter().zip([front, rear]) {
        let wheels = children(axle);
        assert_eq!(wheels.len(), 2, "number of wheels");
        for (wheel, y) in wheels.iter().zip([-5.0, 5.0]) {
            let world = car
                .placement
                .then_after(axle.placement)
                .then_after(wheel.placement);
            let want = top
                .then_after(axle_at)
                .then_after(Transform::translation(Vec3::new(0.0, y, 0.0)))
                .point(center);
            near_point(center_at::<K>(shape_of(wheel), world), want, "a wheel");
            let shape = shape_of(wheel);
            let first = first.get_or_insert_with(|| shape.clone());
            assert!(K::is_same(first, shape), "all the wheels are one part");
        }
    }
}

/// A top-level node can be placed, and one part can be used twice at the top level.
///
/// # Panics
/// If the roots differ.
pub fn step_document_places_top_level_nodes<K: Kernel + Exchange>() {
    let block = cuboid::<K>(1.0, 1.0, 1.0);
    let (a_at, b_at) = (
        Transform::translation(Vec3::new(5.0, 0.0, 0.0)),
        Transform::translation(Vec3::new(0.0, 5.0, 0.0)),
    );
    let back = round_trip::<K>(&document(vec![
        node("a", a_at, NodeContent::Part(block.clone())),
        node("b", b_at, NodeContent::Part(block)),
    ]));
    assert_eq!(back.roots.len(), 2, "number of roots");
    assert_eq!(back.roots[0].name.as_deref(), Some("a"));
    assert_eq!(back.roots[1].name.as_deref(), Some("b"));
    let (a, b) = (shape_of(&back.roots[0]), shape_of(&back.roots[1]));
    assert!(K::is_same(a, b), "the roots must refer to one part");
    let center = Point3::new(0.5, 0.5, 0.5);
    near_point(
        center_at::<K>(a, back.roots[0].placement),
        a_at.point(center),
        "a",
    );
    near_point(
        center_at::<K>(b, back.roots[1].placement),
        b_at.point(center),
        "b",
    );
}

/// Each schema writes a file that declares itself and reads back the same tree.
///
/// # Panics
/// If a schema fails, or a file declares the wrong one.
pub fn step_document_schemas<K: Kernel + Exchange>() {
    let block = cuboid::<K>(1.0, 2.0, 3.0);
    let doc = document(vec![node(
        "frame",
        Transform::IDENTITY,
        NodeContent::Assembly(vec![
            node(
                "a",
                Transform::translation(Vec3::new(10.0, 0.0, 0.0)),
                NodeContent::Part(block.clone()),
            ),
            node(
                "b",
                Transform::translation(Vec3::new(20.0, 0.0, 0.0)),
                NodeContent::Part(block),
            ),
        ]),
    )]);
    // The schema names ISO 10303 gives these application protocols.
    for (schema, declared) in [
        (StepSchema::Ap203, "CONFIG_CONTROL_DESIGN"),
        (StepSchema::Ap214, "AUTOMOTIVE_DESIGN"),
        (StepSchema::Ap242, "AP242"),
    ] {
        let options = StepWriteOptions {
            schema,
            cancel: None,
        };
        let bytes = must(
            K::write_step_document(&doc, &options),
            "write_step_document",
        );
        let header = declared_schema(&bytes)
            .unwrap_or_else(|| panic!("{schema:?}: no FILE_SCHEMA in the file"));
        assert!(header.contains(declared), "{schema:?}: {header}");

        let back = must(K::read_step_document(&bytes, None), "read_step_document");
        let instances = children(&back.roots[0]);
        assert_eq!(instances.len(), 2, "{schema:?}: number of instances");
        assert!(
            K::is_same(shape_of(&instances[0]), shape_of(&instances[1])),
            "{schema:?}: the instances must refer to one part"
        );
        volume_is::<K>(shape_of(&instances[0]), 6.0, format_args!("{schema:?}"));
    }
}

/// Asking for one schema leaves the default as it was, for later writes of both kinds.
///
/// # Panics
/// If a later write uses the earlier schema.
pub fn step_document_schema_does_not_stick<K: Kernel + Exchange>() {
    let block = cuboid::<K>(1.0, 1.0, 1.0);
    let schema_of = |bytes: &[u8]| declared_schema(bytes).expect("a FILE_SCHEMA in the file");
    let doc = document(vec![node(
        "block",
        Transform::IDENTITY,
        NodeContent::Part(block.clone()),
    )]);
    let before = schema_of(&must(K::write_step(&block), "write_step"));
    let default_before = must(
        K::write_step_document(&doc, &StepWriteOptions::default()),
        "write_step_document",
    );
    for schema in [StepSchema::Ap203, StepSchema::Ap242] {
        let options = StepWriteOptions {
            schema,
            cancel: None,
        };
        must(
            K::write_step_document(&doc, &options),
            "write_step_document",
        );
    }
    assert_eq!(
        schema_of(&must(K::write_step(&block), "write_step")),
        before
    );
    let default_after = must(
        K::write_step_document(&doc, &StepWriteOptions::default()),
        "write_step_document",
    );
    assert_eq!(schema_of(&default_after), schema_of(&default_before));
}

/// What a hand-written file gives a colour to.
#[derive(Clone, Copy, PartialEq)]
enum Styled {
    Nothing,
    Solid,
    OneFace,
}

/// A complete AP214 file holding one cube of `side` file units, in the given length unit.
/// It's written out by hand, entity by entity, so that no kernel's writer is involved.
fn cube_step(side: f64, unit: &str, styled: Styled) -> Vec<u8> {
    let mut e = String::new();
    let mut id = 0_u32;
    let mut add = |text: String| {
        id += 1;
        writeln!(e, "#{id}={text};").expect("writing to a string");
        id
    };
    let list = |ids: &[u32]| {
        ids.iter()
            .map(|i| format!("#{i}"))
            .collect::<Vec<_>>()
            .join(",")
    };

    // The corner of index i has coordinates (bit 0, bit 1, bit 2) * side.
    let corner = |i: usize| {
        [
            (i & 1) as f64 * side,
            ((i >> 1) & 1) as f64 * side,
            ((i >> 2) & 1) as f64 * side,
        ]
    };
    let mut points = Vec::new();
    let mut vertices = Vec::new();
    for i in 0..8 {
        let [x, y, z] = corner(i);
        let p = add(format!("CARTESIAN_POINT('',({x:?},{y:?},{z:?}))"));
        points.push(p);
        vertices.push(add(format!("VERTEX_POINT('',#{p})")));
    }

    // Faces by corners, counterclockwise from outside, with their outward normals and
    // first direction along their first edge.
    #[rustfmt::skip]
    let faces: [([usize; 4], [f64; 3], [f64; 3]); 6] = [
        ([0, 2, 3, 1], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]),
        ([4, 5, 7, 6], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
        ([0, 1, 5, 4], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]),
        ([2, 6, 7, 3], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
        ([0, 4, 6, 2], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
        ([1, 3, 7, 5], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
    ];

    let mut edges: Vec<((usize, usize), u32)> = Vec::new();
    let mut advanced = Vec::new();
    for (corners, normal, along) in faces {
        let mut oriented = Vec::new();
        for k in 0..4 {
            let (a, b) = (corners[k], corners[(k + 1) % 4]);
            let key = (a.min(b), a.max(b));
            let known = edges.iter().find(|(k, _)| *k == key).map(|(_, id)| *id);
            let edge = if let Some(edge) = known {
                edge
            } else {
                let (from, to) = (corner(key.0), corner(key.1));
                let d = [
                    (to[0] - from[0]) / side,
                    (to[1] - from[1]) / side,
                    (to[2] - from[2]) / side,
                ];
                let dir = add(format!("DIRECTION('',({:?},{:?},{:?}))", d[0], d[1], d[2]));
                let vector = add(format!("VECTOR('',#{dir},1.0)"));
                let line = add(format!("LINE('',#{},#{vector})", points[key.0]));
                let edge = add(format!(
                    "EDGE_CURVE('',#{},#{},#{line},.T.)",
                    vertices[key.0], vertices[key.1]
                ));
                edges.push((key, edge));
                edge
            };
            let same = if a < b { ".T." } else { ".F." };
            oriented.push(add(format!("ORIENTED_EDGE('',*,*,#{edge},{same})")));
        }
        let loop_id = add(format!("EDGE_LOOP('',({}))", list(&oriented)));
        let bound = add(format!("FACE_OUTER_BOUND('',#{loop_id},.T.)"));
        let n = add(format!(
            "DIRECTION('',({:?},{:?},{:?}))",
            normal[0], normal[1], normal[2]
        ));
        let x = add(format!(
            "DIRECTION('',({:?},{:?},{:?}))",
            along[0], along[1], along[2]
        ));
        let frame = add(format!(
            "AXIS2_PLACEMENT_3D('',#{},#{n},#{x})",
            points[corners[0]]
        ));
        let plane = add(format!("PLANE('',#{frame})"));
        advanced.push(add(format!("ADVANCED_FACE('',(#{bound}),#{plane},.T.)")));
    }
    let shell = add(format!("CLOSED_SHELL('',({}))", list(&advanced)));
    let brep = add(format!("MANIFOLD_SOLID_BREP('cube',#{shell})"));

    let origin = add("CARTESIAN_POINT('',(0.,0.,0.))".to_owned());
    let z = add("DIRECTION('',(0.,0.,1.))".to_owned());
    let x = add("DIRECTION('',(1.,0.,0.))".to_owned());
    let frame = add(format!("AXIS2_PLACEMENT_3D('',#{origin},#{z},#{x})"));

    let length = add(unit.to_owned());
    let angle = add("(NAMED_UNIT(*)PLANE_ANGLE_UNIT()SI_UNIT($,.RADIAN.))".to_owned());
    let solid_angle = add("(NAMED_UNIT(*)SI_UNIT($,.STERADIAN.)SOLID_ANGLE_UNIT())".to_owned());
    let uncertainty = add(format!(
        "UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-7),#{length},'distance_accuracy_value','')"
    ));
    let context = add(format!(
        "(GEOMETRIC_REPRESENTATION_CONTEXT(3)\
         GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#{uncertainty}))\
         GLOBAL_UNIT_ASSIGNED_CONTEXT((#{length},#{angle},#{solid_angle}))\
         REPRESENTATION_CONTEXT('',''))"
    ));
    let representation = add(format!(
        "ADVANCED_BREP_SHAPE_REPRESENTATION('',(#{brep},#{frame}),#{context})"
    ));

    if styled != Styled::Nothing {
        let target = if styled == Styled::Solid {
            brep
        } else {
            advanced[0]
        };
        let colour = add("COLOUR_RGB('',0.2,0.4,0.6)".to_owned());
        let fill = add(format!("FILL_AREA_STYLE_COLOUR('',#{colour})"));
        let area = add(format!("FILL_AREA_STYLE('',(#{fill}))"));
        let surface = add(format!("SURFACE_STYLE_FILL_AREA(#{area})"));
        let side_style = add(format!("SURFACE_SIDE_STYLE('',(#{surface}))"));
        let usage = add(format!("SURFACE_STYLE_USAGE(.BOTH.,#{side_style})"));
        let assignment = add(format!("PRESENTATION_STYLE_ASSIGNMENT((#{usage}))"));
        let item = add(format!("STYLED_ITEM('',(#{assignment}),#{target})"));
        add(format!(
            "MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION('',(#{item}),#{context})"
        ));
    }

    let app = add("APPLICATION_CONTEXT('automotive design')".to_owned());
    let protocol = add(format!(
        "APPLICATION_PROTOCOL_DEFINITION('international standard','automotive_design',2000,#{app})"
    ));
    let _ = protocol;
    let product_context = add(format!("PRODUCT_CONTEXT('',#{app},'mechanical')"));
    let product = add(format!(
        "PRODUCT('hand-made cube','hand-made cube','',(#{product_context}))"
    ));
    let formation = add(format!("PRODUCT_DEFINITION_FORMATION('','',#{product})"));
    let definition_context = add(format!(
        "PRODUCT_DEFINITION_CONTEXT('part definition',#{app},'design')"
    ));
    let definition = add(format!(
        "PRODUCT_DEFINITION('design','',#{formation},#{definition_context})"
    ));
    let shape = add(format!("PRODUCT_DEFINITION_SHAPE('','',#{definition})"));
    add(format!(
        "SHAPE_DEFINITION_REPRESENTATION(#{shape},#{representation})"
    ));

    format!(
        "ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION(('hand-made'),'2;1');\n\
         FILE_NAME('cube','2026-01-01T00:00:00',(''),(''),'','','');\n\
         FILE_SCHEMA(('AUTOMOTIVE_DESIGN {{ 1 0 10303 214 1 1 1 1 }}'));\nENDSEC;\nDATA;\n{e}\
         ENDSEC;\nEND-ISO-10303-21;\n"
    )
    .into_bytes()
}

const SI_METRE: &str = "(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT($,.METRE.))";
const SI_MILLIMETRE: &str = "(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.))";
const SI_CENTIMETRE: &str = "(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.CENTI.,.METRE.))";

/// A file in metres, centimetres or millimetres gives millimetres, and reports its unit.
///
/// # Panics
/// If a size is not converted, or the unit is misreported.
pub fn step_document_reports_and_converts_units<K: Kernel + Exchange>() {
    for (unit, side_in_file, per_metre) in [
        (SI_MILLIMETRE, 500.0, 1000.0),
        (SI_CENTIMETRE, 50.0, 100.0),
        (SI_METRE, 0.5, 1.0),
    ] {
        let data = cube_step(side_in_file, unit, Styled::Nothing);
        let back = must(K::read_step_document(&data, None), unit);
        let reported = back
            .file_units_per_metre
            .unwrap_or_else(|| panic!("{unit}: the unit was not reported"));
        near(reported, per_metre, MEASURE, unit);
        assert_eq!(back.roots.len(), 1, "{unit}: number of roots");
        let shape = shape_of(&back.roots[0]);
        volume_is::<K>(shape, 500.0 * 500.0 * 500.0, unit);
        near_point(
            must(K::center_of_mass(shape), unit),
            Point3::new(250.0, 250.0, 250.0),
            unit,
        );
    }
}

/// A file no modelling kernel wrote gives its product name and a solid.
///
/// # Panics
/// If the file is misread.
pub fn step_document_reads_a_plain_file<K: Kernel + Exchange>() {
    let data = cube_step(2.0, SI_MILLIMETRE, Styled::Nothing);
    let back = must(K::read_step_document(&data, None), "read_step_document");
    assert_eq!(back.roots[0].name.as_deref(), Some("hand-made cube"));
    assert!(back.roots[0].color.is_none(), "the file gave no colour");
    let shape = shape_of(&back.roots[0]);
    assert_eq!(parts::<K>(shape, ShapeKind::Solid).len(), 1, "solids");
    assert_eq!(parts::<K>(shape, ShapeKind::Face).len(), 6, "faces");
}

/// A colour on the whole solid is the part's colour. One on a single face is not: nodes
/// have no way to say which face, so the part reports none.
///
/// # Panics
/// If a colour is lost, or spreads from one face to the part.
pub fn step_document_reads_colors_of_a_plain_file<K: Kernel + Exchange>() {
    let whole = must(
        K::read_step_document(&cube_step(2.0, SI_MILLIMETRE, Styled::Solid), None),
        "a file styling the solid",
    );
    near_color(
        whole.roots[0].color,
        Color {
            r: 0.2,
            g: 0.4,
            b: 0.6,
        },
        "the solid's colour",
    );
    let one_face = must(
        K::read_step_document(&cube_step(2.0, SI_MILLIMETRE, Styled::OneFace), None),
        "a file styling one face",
    );
    assert!(
        one_face.roots[0].color.is_none(),
        "one face's colour must not become the part's: {:?}",
        one_face.roots[0].color
    );
}

/// Bytes that are not STEP return an error.
///
/// # Panics
/// If the kernel returns a document, or panics.
pub fn step_document_rejects_garbage<K: Kernel + Exchange>() {
    for (name, data) in [
        ("empty input", &b""[..]),
        ("arbitrary text", &b"this is not a STEP file"[..]),
        ("arbitrary bytes", &[0xff, 0x00, 0xfe, 0x01, 0x80, 0x7f][..]),
        (
            "a header and nothing else",
            &b"ISO-10303-21;\nHEADER;\n"[..],
        ),
        (
            "a file with no data",
            &b"ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION((''),'2;1');\nENDSEC;\nDATA;\nENDSEC;\nEND-ISO-10303-21;\n"[..],
        ),
    ] {
        assert!(
            K::read_step_document(data, None).is_err(),
            "read_step_document of {name} must return Err"
        );
    }
}

/// A tree that a file can't hold is an error, not a file missing something.
///
/// # Panics
/// If the kernel writes one, or panics.
pub fn step_document_rejects_bad_trees<K: Kernel + Exchange>() {
    let options = StepWriteOptions::default();
    let block = cuboid::<K>(1.0, 1.0, 1.0);
    assert!(
        K::write_step_document(&document(Vec::new()), &options).is_err(),
        "a document with no nodes"
    );
    assert!(
        K::write_step_document(
            &document(vec![node(
                "empty",
                Transform::IDENTITY,
                NodeContent::Assembly(Vec::new())
            )]),
            &options
        )
        .is_err(),
        "an assembly with no children"
    );
    for (name, placement) in [
        (
            "a scale",
            Transform::scale(Point3::ORIGIN, Vec3::new(2.0, 2.0, 2.0)),
        ),
        (
            "a mirror",
            Transform::scale(Point3::ORIGIN, Vec3::new(-1.0, 1.0, 1.0)),
        ),
    ] {
        assert!(
            K::write_step_document(
                &document(vec![node("x", placement, NodeContent::Part(block.clone()))]),
                &options
            )
            .is_err(),
            "a placement that is {name}"
        );
    }
}

/// A flag set beforehand stops a read and a write with `Cancelled`, and the same calls
/// work once a fresh flag is used.
///
/// # Panics
/// If an operation ignores the flag, or reports anything else.
pub fn step_document_honours_a_set_cancel_flag<K: Kernel + Exchange>() {
    let doc = document(vec![node(
        "block",
        Transform::IDENTITY,
        NodeContent::Part(cuboid::<K>(1.0, 1.0, 1.0)),
    )]);
    let set = CancelFlag::new();
    set.cancel();
    let cancelled = StepWriteOptions {
        schema: StepSchema::default(),
        cancel: Some(set.clone()),
    };
    assert!(
        matches!(
            K::write_step_document(&doc, &cancelled),
            Err(Error::Cancelled)
        ),
        "a write with a set flag"
    );
    let live = StepWriteOptions {
        schema: StepSchema::default(),
        cancel: Some(CancelFlag::new()),
    };
    let bytes = must(K::write_step_document(&doc, &live), "write_step_document");
    assert!(
        matches!(
            K::read_step_document(&bytes, Some(&set)),
            Err(Error::Cancelled)
        ),
        "a read with a set flag"
    );
    let fresh = CancelFlag::new();
    must(
        K::read_step_document(&bytes, Some(&fresh)),
        "read_step_document with a fresh flag",
    );
}

/// A box goes through IGES and comes back as a solid of the same volume and position.
///
/// # Panics
/// If the kind, volume or position differ.
pub fn iges_round_trip_makes_a_solid<K: Kernel + Exchange>() {
    for (x, y, z) in [(1.0, 2.0, 3.0), (0.5, 40.0, 7.0), (100.0, 100.0, 100.0)] {
        let block = cuboid::<K>(x, y, z);
        let by = Vec3::new(3.0, -4.0, 5.0);
        let moved = must(K::translate(&block, by), "translate");
        let bytes = must(K::write_iges(&[&moved]), "write_iges");
        let back = must(K::read_iges(&bytes), "read_iges");
        assert_eq!(
            must(K::kind(&back), "kind"),
            ShapeKind::Solid,
            "kind of a box"
        );
        assert_eq!(
            parts::<K>(&back, ShapeKind::Face).len(),
            6,
            "faces of a box"
        );
        volume_is::<K>(&back, x * y * z, "box from IGES");
        near_point(
            must(K::center_of_mass(&back), "com"),
            Point3::new(x / 2.0, y / 2.0, z / 2.0) + by,
            "centre of a box from IGES",
        );
    }
}

/// Solids with curved faces close too. Spline conversion is allowed in IGES, so the
/// measure is held to the blend tolerance rather than the exact-geometry one.
///
/// # Panics
/// If a solid doesn't come back.
pub fn iges_round_trip_of_curved_solids<K: Kernel + Exchange>() {
    let cylinder = must(K::make_cylinder(Axis::Z, 2.0, 5.0), "make_cylinder");
    let ball = must(K::make_sphere(Point3::ORIGIN, 3.0), "make_sphere");
    for (name, shape, volume) in [
        ("cylinder", cylinder, PI * 4.0 * 5.0),
        ("sphere", ball, ball_volume(3.0)),
    ] {
        let bytes = must(K::write_iges(&[&shape]), name);
        let back = must(K::read_iges(&bytes), name);
        assert_eq!(must(K::kind(&back), name), ShapeKind::Solid, "{name}");
        near(
            must(K::volume(&back), name),
            volume,
            crate::support::BLEND,
            name,
        );
    }
}

/// Several disjoint shapes stay separate solids with their total volume.
///
/// # Panics
/// If the count or the volume differs.
pub fn iges_holds_several_shapes<K: Kernel + Exchange>() {
    let sides = [1.0_f64, 2.0, 3.0];
    let shapes: Vec<K::Shape> = sides
        .iter()
        .enumerate()
        .map(|(i, &side)| {
            must(
                K::make_box(
                    Point3::new(i as f64 * 10.0, 0.0, 0.0),
                    Vec3::new(side, side, side),
                ),
                "make_box",
            )
        })
        .collect();
    let refs: Vec<&K::Shape> = shapes.iter().collect();
    let bytes = must(K::write_iges(&refs), "write_iges");
    let back = must(K::read_iges(&bytes), "read_iges");
    let solids = parts::<K>(&back, ShapeKind::Solid);
    assert_eq!(solids.len(), sides.len(), "number of solids");
    let total: f64 = sides.iter().map(|s| s * s * s).sum();
    volume_is::<K>(&back, total, "all the solids");
}

/// Bytes that are not IGES return an error, as does asking to write nothing.
///
/// # Panics
/// If the kernel returns a shape, or panics.
pub fn iges_rejects_garbage<K: Kernel + Exchange>() {
    for (name, data) in [
        ("empty input", &b""[..]),
        ("arbitrary text", &b"this is not an IGES file"[..]),
        ("arbitrary bytes", &[0xff, 0x00, 0xfe, 0x01, 0x80, 0x7f][..]),
    ] {
        assert!(
            K::read_iges(data).is_err(),
            "read_iges of {name} must return Err"
        );
    }
    assert!(K::write_iges(&[]).is_err(), "write_iges of nothing");
}
