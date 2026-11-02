# occt-rs

Idiomatic Rust solid modelling on [Open CASCADE Technology](https://github.com/Open-Cascade-SAS/OCCT)
8.0.1, for Linux, macOS, Windows and WebAssembly.

```rust,no_run
use occt::prelude::*;

fn main() -> occt::Result<()> {
    let plate = Cuboid::new([80.0, 40.0, 6.0]).build()?;
    let hole = Cylinder::new(5.0, 10.0).along(Axis::new([40.0, 20.0, -2.0].into(), Dir3::Z)).build()?;
    let part = (&plate - &hole)?;

    let mesh = part.mesh(MeshOptions::default())?;
    println!("{} triangles, volume {:.1}", mesh.indices.len(), part.volume()?);

    part.write_step(std::fs::File::create("part.step")?)?;
    Ok(())
}
```

It covers what a CAD application needs: curves and NURBS, sketch-to-solid, sweeps and lofts, general
booleans with history for persistent naming, variable fillets, shell, draft, per-face meshes, ray
picking, hidden-line drawings, native saves, and STEP assemblies, IGES, STL, OBJ and 3MF.
[docs/INTEGRATING.md](docs/INTEGRATING.md) shows how to plug it into an application, and
`cargo run -p occt-rs --example cad_session` runs it end to end.

## Licence: read this first

The crates here are MIT OR Apache-2.0. **OCCT itself stays LGPL-2.1 with the Open CASCADE
exception**, and the exception doesn't relicense it. Your application can keep its own terms
as long as the `occt_capi` module ships as a separate, replaceable file. See
[docs/LICENSING.md](docs/LICENSING.md) for the checklist.

## Crates

| Crate | Job | Depends on |
|---|---|---|
| `occt-rs` (`occt`) | The public idiomatic API | `occt-kernel`, one backend |
| `occt-kernel` | The contract: traits, errors, kinds. No FFI | `occt-math`, `occt-mesh` |
| `occt-kernel-capi` | The OCCT backend | `occt-kernel`, `occt-capi-sys` |
| `occt-capi-sys` | Builds the module, raw bindings | OCCT, at build time |
| `occt-conformance` | Behaviour tests any kernel must pass | `occt-kernel` |
| `occt-math`, `occt-geom`, `occt-mesh` | Pure value types: points and transforms, exact curves and surfaces, meshes and file export | nothing OCCT-related |

To change backend, implement the `occt-kernel` traits, pass `occt-conformance`, and change the
one private module in `occt-rs` that names the kernel. The public API has no kernel type in it.

## Build

Needs CMake 3.21+, a C++17 compiler and Rust. The first build downloads OCCT (hash-pinned)
and compiles it, about 8 minutes with 8 jobs.

```sh
cargo test --workspace
```

Environment variables:

| Variable | Effect |
|---|---|
| `OCCT_CAPI_LIB_DIR` | Use an already built module and skip the build |
| `OCCT_SOURCE_DIR` | Use local OCCT sources and skip the download |
| `CMAKE_TOOLCHAIN_FILE` | Cross-compile |
| `OCCT_BUILD_JOBS` | Compile job count. Default: one per 3 GiB of available RAM, up to the core count |

`cargo run` and `cargo test` find the built module themselves. To ship, copy it next to your
binary. Build scripts get its directory as `DEP_OCCT_CAPI_LIB_DIR`. On Linux the loader doesn't
search the binary's folder by default, so also link with an rpath:

```sh
RUSTFLAGS='-C link-arg=-Wl,-rpath,$ORIGIN' cargo build --release
```

Windows finds a DLL next to the executable. On macOS use `@loader_path` as the rpath. Only the
Linux steps have been run so far.

## WebAssembly

The module is about 24 MB, or 7.2 MB gzipped.

Rust stays on `wasm32-unknown-unknown`. OCCT runs as a separate Emscripten module:

```sh
source <emsdk>/emsdk_env.sh
cmake -DOUT_DIR=$PWD/out -DWASM=ON \
  -DTOOLCHAIN=$EMSDK/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake \
  -P crates/occt-capi-sys/capi/build.cmake
```

Load it before calling into Rust:

```js
import createOcctCapi from "./out/lib/occt_capi.mjs";
globalThis.occtCapi = await createOcctCapi();
```

Calls made before that return `Error::Unavailable`.

## Threads

Shapes are `Send + Sync`. Kernel calls are safe from any thread but run one at a time, because
shapes share topology that operations such as meshing update.
