//! What a CAD application does with a kernel, end to end: a small feature tree rebuilt from
//! sketch to solid, selections that survive the rebuild through history, picking, drawing,
//! saving and exchange.
//!
//! Run it with `cargo run -p occt-rs --example cad_session`.

use std::thread;

use occt::prelude::*;

/// A body in the application's document. A face is referred to by what made it, and history
/// says where it went.
struct Body {
    shape: Shape,
    /// The top face of the base extrusion, followed through later features.
    top: Vec<Shape>,
}

fn nearly(a: f64, b: f64, tolerance: f64) -> bool {
    (a - b).abs() <= tolerance
}

/// Feature 1: sketch a plate outline with a round hole and extrude it.
fn base_plate() -> occt::Result<Body> {
    let outline = Wire::polygon([
        [0.0, 0.0, 0.0],
        [80.0, 0.0, 0.0],
        [80.0, 40.0, 0.0],
        [0.0, 40.0, 0.0],
    ])?;
    let hole = Wire::circle([20.0, 20.0, 0.0], [0.0, 0.0, 1.0], 6.0)?;
    let profile = Face::planar(&outline).hole(&hole).build()?;
    let extruded = profile.extruded_modeled([0.0, 0.0, 10.0])?;

    let top = extruded
        .shape
        .faces()?
        .filter(|f| f.center_of_mass().is_ok_and(|c| nearly(c.z, 10.0, 1e-6)))
        .map(Shape::from)
        .collect();
    Ok(Body {
        shape: extruded.shape,
        top,
    })
}

/// Feature 2: cut a pocket, following the top face through the operation.
fn pocket(body: Body) -> occt::Result<Body> {
    let tool = Cuboid::new([20.0, 20.0, 6.0])
        .at([50.0, 10.0, 5.0])
        .build()?;
    let cut = Boolean::cut([&body.shape]).tools([&tool]).build()?;
    let top = body.top.iter().flat_map(|f| cut.history.trace(f)).collect();
    Ok(Body {
        shape: cut.shape,
        top,
    })
}

/// Feature 3: round the four vertical outer edges, picked by geometry.
fn rounded_corners(body: Body) -> occt::Result<Body> {
    let mut fillet = EdgeFillet::on(&body.shape);
    for edge in body.shape.edges()? {
        let (a, b) = (edge.start_point()?, edge.end_point()?);
        let vertical = nearly(a.x, b.x, 1e-6)
            && nearly(a.y, b.y, 1e-6)
            && nearly((a.z - b.z).abs(), 10.0, 1e-6);
        let corner = (nearly(a.x, 0.0, 1e-6) || nearly(a.x, 80.0, 1e-6))
            && (nearly(a.y, 0.0, 1e-6) || nearly(a.y, 40.0, 1e-6));
        if vertical && corner {
            fillet = fillet.radius(edge, 4.0);
        }
    }
    let rounded = fillet.build()?;
    let top = body
        .top
        .iter()
        .flat_map(|f| rounded.history.trace(f))
        .collect();
    Ok(Body {
        shape: rounded.shape,
        top,
    })
}

fn main() -> occt::Result<()> {
    println!("kernel: {}\n", occt::kernel_version()?);

    // Rebuild the feature tree.
    let body = rounded_corners(pocket(base_plate()?)?)?;
    let props = body.shape.mass_properties()?;
    println!(
        "body: {} faces, volume {:.1}, centre of mass {:?}",
        body.shape.faces()?.len(),
        props.mass,
        props.center
    );
    assert!(body.shape.is_valid()?);
    assert_eq!(body.top.len(), 1, "the selection survives the rebuild");
    println!(
        "top face selection survived: area {:.1}",
        body.top[0].area()?
    );

    // Display: one mesh per face and one polyline per edge, in the order of faces() and edges().
    let display = body
        .shape
        .tessellate(MeshOptions::default().absolute(0.05))
        .build()?;
    let triangles: usize = display.faces.iter().map(|f| f.mesh.indices.len()).sum();
    println!(
        "display: {} face meshes, {triangles} triangles, {} edge polylines",
        display.faces.len(),
        display.edges.len()
    );

    // Picking: a ray from the camera through the cursor.
    let hits = body.shape.ray_cast(
        [40.0, 35.0, 100.0],
        Dir3::new([0.0, 0.0, -1.0]).expect("nonzero"),
    )?;
    println!(
        "pick: {} hits, nearest at z = {:.1}",
        hits.len(),
        hits[0].point.z
    );
    assert!(*hits[0].face == body.top[0]);

    // Drawing: a hidden-line view.
    let lines = body
        .shape
        .hidden_lines(View::isometric().deflection(0.05))?;
    let hidden = lines.iter().filter(|l| !l.visible).count();
    println!("drawing: {} lines, {hidden} hidden", lines.len());

    // A thread: a triangular profile swept along a helix.
    let helix = Helix::new(4.0, 1.5, 6.0).build()?;
    let profile = Wire::polygon([[4.0, 0.0, 0.0], [4.6, 0.0, 0.4], [4.0, 0.0, 0.8]])?;
    let thread_face = Face::planar(&profile).build()?;
    let thread = Sweep::profile(&thread_face)
        .along(&helix)
        .binormal(Dir3::Z)
        .build()?;
    println!(
        "thread: volume {:.2}, valid {}",
        thread.shape.volume()?,
        thread.shape.is_valid()?
    );
    assert!(thread.shape.volume()? > 0.0);

    // Saving your own document: lossless, for the same kernel only.
    let saved = body.shape.to_native()?;
    let reopened = Shape::from_native(&saved)?;
    assert!(nearly(reopened.volume()?, props.mass, 1e-6));
    println!("native save: {} bytes", saved.len());

    // Exchange: an assembly with two instances of one part.
    let bolt = Cylinder::new(3.0, 20.0).build()?;
    let document = Document::new()
        .part("plate", &body.shape)
        .color([0.6, 0.65, 0.7])
        .assembly(
            Assembly::new("fasteners")
                .part("bolt-a", &bolt)
                .at(Transform::translation(Vec3::new(10.0, 10.0, 0.0))),
        );
    let step = document.to_step(StepOptions::new())?;
    let imported = Document::from_step(&step)?;
    println!(
        "STEP: {} bytes, {} part instances",
        step.len(),
        imported.parts().len()
    );
    for part in imported.parts() {
        println!("  {}", part.full_name());
    }

    // Exports for printing.
    let mesh = display.merged();
    let mut stl = Vec::new();
    mesh.write_stl_binary(&mut stl)?;
    println!("STL: {} bytes", stl.len());

    // Bodies can move between threads; heavy work can run off the UI thread and be cancelled.
    let flag = CancelFlag::new();
    let worker = {
        let shape = body.shape.clone();
        let flag = flag.clone();
        thread::spawn(move || {
            let tool = Sphere::new(5.0).at([40.0, 20.0, 10.0]).build()?;
            Boolean::cut([&shape])
                .tools([&tool])
                .cancel_with(&flag)
                .build()
        })
    };
    let dimpled = worker.join().expect("the worker doesn't panic")?;
    println!(
        "worker thread: dimpled volume {:.1}",
        dimpled.shape.volume()?
    );

    println!("\n{}", occt::notice());
    Ok(())
}
