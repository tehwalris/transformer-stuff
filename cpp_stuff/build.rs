use std::{env, path::PathBuf};

extern crate bindgen;
extern crate cc;

fn main() {
    println!("cargo:rerun-if-changed=src");

    cc::Build::new()
        .cpp(true)
        .file("src/example.cpp")
        .compile("example");

    let bindings = bindgen::Builder::default()
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .header("src/wrapper.h")
        .clang_arg("-x")
        .clang_arg("c++")
        .opaque_type("std::.*")
        .allowlist_type("cml::SimpleLlamaModelLoader")
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
