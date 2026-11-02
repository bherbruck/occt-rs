# Licensing

This is an engineering summary, not legal advice. Have counsel review it before
you ship a product that relies on it.

## What each part is licensed under

| Part | Licence |
|---|---|
| All crates in this repository, including the `occt_capi` shim | MIT OR Apache-2.0 |
| OCCT, downloaded at build time and not stored here | LGPL-2.1-only WITH OCCT-exception-1.0 |
| A built `occt_capi` module (shared library or `.wasm`), which contains OCCT | LGPL-2.1-only WITH OCCT-exception-1.0 for the OCCT part |

## What the exception does and does not do

The Open CASCADE exception lets object code that incorporates material from OCCT
*header files* be distributed under terms of your choice, provided the supporting
documentation prominently says the code uses OCCT.

It does **not** relicense OCCT. It does not make OCCT MIT or Apache-2.0, and it does
not remove the LGPL's duties for OCCT's own compiled code. So MIT OR Apache-2.0 is
accurate for this project's code. It is not accurate for a binary bundle as a whole.

## How the design keeps your application under your own terms

Rust code never touches OCCT headers or code. It calls one module through a plain C
ABI, and that module is the same replaceable unit on every platform:

| Platform | Module | Loaded |
|---|---|---|
| Linux | `libocct_capi.so` | dynamic linking |
| macOS | `libocct_capi.dylib` | dynamic linking |
| Windows | `occt_capi.dll` | dynamic linking |
| Browser, Node | `occt_capi.wasm` + `occt_capi.mjs` | separate module, own linear memory |

A user can rebuild the module from modified OCCT sources and swap it in without
touching your application, which is what LGPL-2.1 section 6 is for. Do not link OCCT
statically into your binary or merge the wasm module into your own unless you also
provide what section 6 then requires, such as relinkable object files.

## Checklist when you distribute

1. Ship the `occt_capi` module as a separate, replaceable file.
2. Include `NOTICE` and the files in `crates/occt-capi-sys/licenses/`. Show
   `occt::notice()` in your about box or documentation.
3. Provide the OCCT source and this repository's `capi/` build files for the exact
   version shipped, or a written offer as LGPL-2.1 section 6 describes. Keep your own
   copy: an upstream URL can disappear.
4. If you modified OCCT, mark the changes and ship them under the same licence.
5. Do not forbid users from modifying the module or reverse engineering for debugging
   their modifications.

Whether a separately loaded wasm module counts as a "suitable shared library mechanism"
under section 6(b) has little settled guidance. Have counsel look at web distribution
in particular. Mobile app stores raise their own LGPL questions.

## A future Rust kernel

The `occt-kernel` traits let another kernel replace this backend, but the trait
boundary has no effect on copyright:

- **A port or translation of OCCT source to Rust is a derivative work.** It stays under
  LGPL-2.1, and the exception, which only covers header material in object code, does
  not change that.
- **Only an independent implementation can be MIT OR Apache-2.0.** Write it from public
  literature and specifications, by people who have not worked from OCCT's source.
  Decide on this process before anyone starts, because it can't be applied afterwards.
- `occt-conformance` tests behaviour with analytic answers. That is what such a kernel
  should target, instead of OCCT's code.
