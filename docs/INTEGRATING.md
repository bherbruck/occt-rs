# Integrating occt-rs into a CAD application

`crates/occt-rs/examples/cad_session.rs` runs all of this end to end:

```sh
cargo run -p occt-rs --example cad_session
```

## 1. Add it

```toml
[dependencies]
occt-rs = { path = "../occt-rs/crates/occt-rs" }
```

```rust
use occt::prelude::*;
```

The first build compiles OCCT, about 8 minutes with 8 jobs. Point `OCCT_CAPI_LIB_DIR` at a built
module to skip that.

## 2. Keep it behind your own type

Wrap `occt::Shape` in your body type, the same way you'd wrap any other kernel:

```rust
struct OcctBody(occt::Shape);
```

- `Shape` is `Send + Sync`, and cloning is cheap: clones share topology.
- `Shape` implements `Eq + Hash` by identity, so faces and edges work as `HashMap` keys.
- Kernel calls run one at a time across the process, so run long ones off the UI thread and cancel
  them with `CancelFlag`.

## 3. What each CAD job calls

| Job | API |
|---|---|
| Sketch profile to face | `Wire::polygon`, `Wire::new([Edge::line(..), Edge::arc(..)])`, `Face::planar(&w).hole(&h).build()` |
| Extrude, revolve | `shape.extruded_modeled(v)`, `shape.revolved_modeled(axis, angle)`, `face.extruded_tapered(v, angle)` |
| Sweep, loft | `Sweep::profile(&p).along(&path)`, `Loft::through(sections)` |
| Thread | `Helix::new(r, pitch, h)` then `Sweep` with `.binormal(Dir3::Z)` |
| Join, cut, intersect | `Boolean::cut([&a]).tools([&b]).fuzzy(t).unify().build()`, or `(&a - &b)?` |
| Split, section | `shape.split_by(tools)`, `shape.section_plane(plane)` |
| Fillet, chamfer | `EdgeFillet::on(&s).radius(e, r).linear(e, a, b)`, `EdgeChamfer::on(&s).equal(e, d)` |
| Shell, draft | `solid.shelled(t).open(&face)`, `solid.drafted(pull, neutral, angle).face(&f)` |
| Delete face, offset, thicken | `solid.without_faces(..)`, `shape.offset(d)`, `shape.thickened(t)` |
| Surface modelling | `Face::from_surface`, `Face::fill(edges)`, `Sewing::new().add(..)`, `Loft::through(..).sheet()` |
| Clean up imports | `shape.healed(tol)`, `shape.unified()` |
| Pattern, mirror, instance | `shape.transformed(t)`, `shape.mirrored(plane)`, `shape.placed(t)` |
| Selection and persistent naming | `modeled.history.trace(&face)`, `.modified`, `.generated`, `.is_deleted` |
| Display | `shape.tessellate(options).build()`: `faces[i]` is the i-th of `shape.faces()`, `edges[i]` the i-th of `shape.edges()` |
| Picking | `shape.ray_cast(origin, dir)` returns the face hit |
| Face and edge inspector | `face.surface()`, `face.point_at(u, v)`, `edge.curve()`, `edge.convexity_in(&s)`, `edge.faces_in(&s)` |
| Measure | `shape.mass_properties()`, `shape.distance_to(&other)`, `solid.locate(p)`, `shape.bounding_box()` |
| Drawings | `shape.hidden_lines(View::isometric())` |
| Save your document | `shape.to_native()`, `Shape::from_native(&bytes)` |
| STEP | `Document::from_step`, `document.parts()`, `Document::new().part(..).to_step(..)`, `Shape::read_step` |
| IGES | `Shape::from_iges`, `occt::write_iges(shapes)` |
| Printing | `mesh.write_stl_binary`, `write_obj`, `write_3mf` |

## 4. Following selections through a rebuild

Every operation that changes topology can return `Modeled { shape, history }`. Keep a reference to
what the user picked and carry it through each feature:

```rust
let cut = Boolean::cut([&body]).tools([&tool]).build()?;
let selection: Vec<Shape> = selection.iter().flat_map(|f| cut.history.trace(f)).collect();
```

`trace` returns what the face became, itself if untouched, or nothing if it was deleted.

## 5. Web builds

Rust stays on `wasm32-unknown-unknown`. The kernel is a separate module, and it loads asynchronously:

```js
import createOcctCapi from "./occt_capi.mjs";
globalThis.occtCapi = await createOcctCapi();
```

Until that's done, operations return `Error::Unavailable`, and `occt::ready()` checks without doing
any work. The module is about 24 MB, or 7.2 MB gzipped, so load it lazily. The web build runs on one
thread: a set `CancelFlag` is honoured when an operation starts but can't stop a running one.

## 6. Plugging into SolidCraft

SolidCraft's boundary is `solidcraft-kernel`: functions over `Body`. occt-rs doesn't copy that
contract, but the concepts line up, so an adapter behind a `kernel-occt` cargo feature is mostly
one-liners.

| SolidCraft | occt-rs |
|---|---|
| `Body` | `Shape`, `Solid` |
| `box_solid`, `cylinder`, `sphere`, `torus` | `Cuboid`, `Cylinder`, `Sphere`, `Torus` |
| `extrude`, `extrude_tapered`, `revolve`, `sweep`, `loft` | `extruded_modeled`, `extruded_tapered`, `revolved_modeled`, `Sweep`, `Loft` |
| `boolean(BoolOp, ..)`, `split_by_plane`, `split_body` | `Boolean`, `split_by` |
| `fillet`, `fillet_variable`, `chamfer`, `chamfer_sides` | `EdgeFillet`, `EdgeChamfer` |
| `shell`, `draft`, `offset_faces`, `thicken`, `delete_faces` | `shelled`, `drafted`, `offset`, `thickened`, `without_faces` |
| `sew`, `stitch`, `patch_edges`, `loft_sheet`, `sweep_sheet` | `Sewing`, `Face::fill`, `Loft::sheet`, `Sweep` on a wire |
| `face_surface`, `edge_circle`, `EdgeInfo`, `FaceInfo` | `face.surface()`, `edge.curve()` |
| `measure`, `measure_mass`, `body_distance` | `mass_properties`, `distance_to` |
| `mesh_body`, `FacePaint` | `tessellate`, whose per-face meshes give the face mapping |
| `encode_native_body`, `decode_native_body` | `to_native`, `from_native` |
| `step_import`, `step_export_products`, `iges_import`, `iges_export_bodies` | `Document`, `Shape::from_iges`, `write_iges` |
| `history`, `provenance` | `History` on `Modeled` |

Not covered, so SolidCraft would keep its own code or report the feature as unsupported on this kernel:

- G2 edge fillets on a solid and their corner patches. OCCT's fillet is G1, which `a_circular_fillet_is_g1_but_not_g2` checks. `Face::blend` gives a G2 patch between two edges, but it doesn't trim the solid's faces back or build corners.
- T-splines, which OCCT doesn't have.
- Surface match that deforms an existing surface.
- Emboss on curved faces and text. OCCT has a text builder, but this build leaves FreeType out.
- Prismatic mesh conversion, and SolidCraft's own boolean diagnostics.

### Smooth joins

| Job | Call | What it delivers |
|---|---|---|
| Is this join G0, G1 or G2? | `Face::continuity_with` | Worst gap, crease and curvature difference over 21 points along the edge, not at its ends |
| Patch between two edges | `Face::blend` | G2 by default, or G1. Built directly, then measured, and it fails rather than return a creased patch |
| Patch filling a closed loop | `Face::fill`, `BoundaryFill::tangent_to` | G0 or G1 only. Fails if the side edges make tangency impossible |
| Grow or untrim a face | `Face::extended` | Follows the surface. Trims and holes go |

Filling stops at G1 on purpose. OCCT documents G2 for `BRepOffsetAPI_MakeFilling`, but `GeomAbs_G2` always throws, because `GeomPlate` compares the enum's raw value against 2. Passing `GeomAbs_C1`, whose raw value is 2, does reach the curvature mode, but on a sphere of curvature 0.1 it left curvature errors of 0.35 to 0.58, against 0.002 for G1. `Face::blend` is the G2 route: on the same sphere it leaves about 0.00001.

The limits of `Face::blend`:

- It joins two edges, so a patch closed on four sides still needs `Face::fill` at G1.
- Its side edges are whatever the construction gives, and aren't matched to anything.
- Its edges must not touch, so a blend can't run out to a point.
- A large `tension` can make it cross itself, and only validity is checked, not self-intersection.

The sketch solver and sheet metal live above the kernel and don't change.

Before merging an adapter, these need changing, because SolidCraft forbids OCCT today:

1. The non-negotiables in `AGENTS.md`
2. Rule 2 in `CONTRIBUTING.md`
3. `plan/adr/0001-geometry-kernel.md`
4. `cargo xtask licences` and `deny.toml`, which must also learn about the separately shipped module
5. Native saves are kernel-specific, so a document needs to record which kernel wrote each body

## 7. Limits worth knowing

- **Concurrency:** calls are serialised process-wide, so parallel rebuilds of separate bodies don't
  speed up. `Boolean::parallel()` uses threads inside one operation.
- **Native saves:** corrupted files are caught by a checksum, but a deliberately forged one can still
  crash OCCT. Only open native saves you trust; use STEP for files from others.
- **Cancelling:** STEP parsing and IGES can't be interrupted, only STEP's transfer stage, and meshing
  stops only between faces.
- **Tangent chains:** fillets and chamfers extend along tangent edges, as most CAD systems do.
