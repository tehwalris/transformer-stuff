use std::{env, path::PathBuf};

extern crate bindgen;
extern crate cc;

fn main() {
    println!("cargo:rerun-if-changed=src");

    cc::Build::new().file("src/ggml.c").compile("ggml");

    cc::Build::new()
        .cpp(true)
        .file("src/llama.cpp")
        .file("src/loading.cpp")
        .flag("-mavx2")
        .flag("-mfma")
        .flag("-mf16c")
        .compile("cpp_stuff_cc");

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
