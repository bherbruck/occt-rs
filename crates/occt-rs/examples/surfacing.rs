//! Measures what each way of joining surfaces delivers, on shapes with known answers.

use std::ops::Range;

use occt::prelude::*;

fn patch(surface: &Surface, u: Range<f64>, v: Range<f64>) -> occt::Result<Face> {
    Face::from_surface(surface.clone()).bounds(u, v).build()
}

/// The edge of `face` whose middle is closest to `target`.
fn edge_near(face: &Face, target: [f64; 3]) -> occt::Result<Edge> {
    let target = Point3::from(target);
    let mut best: Option<(f64, Edge)> = None;
    for edge in face.edges()? {
        let range = edge.geometry()?.range;
        let middle = edge.point_at(0.5 * (range.start + range.end))?.point;
        let distance = middle.distance(target);
        if best.as_ref().is_none_or(|(d, _)| distance < *d) {
            best = Some((distance, edge));
        }
    }
    Ok(best.expect("a face has edges").1)
}

fn show(label: &str, report: ContinuityReport, curvature: f64) {
    println!(
        "{label:<34} gap {:.1e}  crease {:>8.4} deg  curvature gap {:.5} ({:>6.1}% of {curvature})",
        report.gap,
        report.angle.as_degrees(),
        report.curvature_gap,
        100.0 * report.curvature_gap / curvature,
    );
}

fn main() -> occt::Result<()> {
    let radius = 5.0;
    let kappa = 1.0 / radius;
    let cylinder = Surface::Cylinder {
        frame: Frame::WORLD,
        radius,
    };
    let at = |u: f64| [radius * u.cos(), radius * u.sin(), 2.0];
    let a = patch(&cylinder, -1.5..-0.5, 0.0..4.0)?;
    let b = patch(&cylinder, 0.5..1.5, 0.0..4.0)?;
    let (ea, eb) = (edge_near(&a, at(-0.5))?, edge_near(&b, at(0.5))?);
    let g1 = Face::blend((&ea, &a), (&eb, &b))
        .continuity(BlendContinuity::G1)
        .build()?;
    show("cylinder, G1 blend", g1.continuity_with(&a, &ea)?, kappa);
    let g2 = Face::blend((&ea, &a), (&eb, &b)).build()?;
    show("cylinder, G2 blend", g2.continuity_with(&a, &ea)?, kappa);

    let radius = 10.0;
    let kappa = 1.0 / radius;
    let sphere = Surface::Sphere {
        frame: Frame::WORLD,
        radius,
    };
    let at = |u: f64, v: f64| {
        [
            radius * v.cos() * u.cos(),
            radius * v.cos() * u.sin(),
            radius * v.sin(),
        ]
    };
    let a = patch(&sphere, -0.5..0.5, -0.9..-0.3)?;
    let b = patch(&sphere, -0.5..0.5, 0.2..0.7)?;
    let (ea, eb) = (edge_near(&a, at(0.0, -0.3))?, edge_near(&b, at(0.0, 0.2))?);
    let g2 = Face::blend((&ea, &a), (&eb, &b)).build()?;
    show("sphere, G2 blend", g2.continuity_with(&a, &ea)?, kappa);
    show(
        "sphere, G2 blend, far side",
        g2.continuity_with(&b, &eb)?,
        kappa,
    );

    let box_ = Cuboid::new([10.0, 10.0, 10.0]).build()?;
    let edge = box_.edges()?.next().expect("a box has edges");
    let rounded = box_.fillet(2.0).edges([&edge]).build()?;
    for round in rounded.faces()? {
        if matches!(round.surface()?, Surface::Plane { .. }) {
            continue;
        }
        for joint in round.edges()? {
            for other in rounded.faces()? {
                if other == round || !other.edges()?.any(|e| e == joint) {
                    continue;
                }
                let report = round.continuity_with(&other, &joint)?;
                if report.angle.as_degrees() < 1.0 {
                    show("circular fillet on a plane", report, 0.5);
                }
            }
        }
    }
    Ok(())
}
