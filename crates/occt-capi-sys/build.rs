//! Provides the occt_capi module: a prebuilt one from `OCCT_CAPI_LIB_DIR`, or one
//! built with `capi/build.cmake`. On wasm32 the module is loaded by JS at run time,
//! so nothing happens here.

use std::env;
use std::io::{self, ErrorKind};
use std::path::PathBuf;
use std::process::{Command, Stdio};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=capi");
    for var in [
        "OCCT_CAPI_LIB_DIR",
        "OCCT_SOURCE_DIR",
        "OCCT_BUILD_JOBS",
        "CMAKE_TOOLCHAIN_FILE",
    ] {
        println!("cargo:rerun-if-env-changed={var}");
    }

    if env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("wasm32") {
        return;
    }

    let lib_dir = match env::var_os("OCCT_CAPI_LIB_DIR") {
        Some(dir) => PathBuf::from(dir),
        None => build_module(),
    };
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=occt_capi");
    println!("cargo:lib_dir={}", lib_dir.display());
}

fn build_module() -> PathBuf {
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("cargo sets OUT_DIR")).join("occt_capi");
    let script =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR"))
            .join("capi")
            .join("build.cmake");

    let mut cmake = Command::new("cmake");
    cmake.arg(format!("-DOUT_DIR={}", out.display()));
    if let Some(dir) = env::var_os("OCCT_SOURCE_DIR") {
        cmake.arg(format!(
            "-DOCCT_SOURCE_DIR={}",
            PathBuf::from(dir).display()
        ));
    }
    if let Some(file) = env::var_os("CMAKE_TOOLCHAIN_FILE") {
        cmake.arg(format!("-DTOOLCHAIN={}", PathBuf::from(file).display()));
    }
    cmake.arg("-P").arg(&script);
    // Cargo parses a build script's stdout, so CMake's output goes to stderr.
    cmake.stdout(Stdio::from(io::stderr()));

    match cmake.status() {
        Ok(status) if status.success() => out.join("lib"),
        Ok(status) => panic!("{} failed with {status}", script.display()),
        Err(e) if e.kind() == ErrorKind::NotFound => panic!(
            "cmake was not found on PATH. occt-capi-sys needs CMake 3.21+ and a C++17 compiler \
             to build the occt_capi module, or set OCCT_CAPI_LIB_DIR to a prebuilt one"
        ),
        Err(e) => panic!("could not run cmake: {e}"),
    }
}
