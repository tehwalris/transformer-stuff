use std::{env, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=cpp_src");

    let dst = cmake::build("cpp_src");
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-lib=static=cpp_stuff_base");
    println!("cargo:rustc-flags=-l dylib=stdc++");

    let bindings = bindgen::Builder::default()
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .header("cpp_src/wrapper.h")
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
