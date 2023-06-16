use std::{env, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=cpp_src");
    println!("cargo:rerun-if-changed=build.ninja");

    // let dst = cmake::build("cpp_src");
    // println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-search=native=/home/philippe/src/github.com/tehwalris/transformer-stuff/cpp_stuff/build/lib");
    println!("cargo:rustc-link-lib=static=cpp_stuff");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-lib=cublas");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=amdhip64");

    let bindings = bindgen::Builder::default()
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .header("cpp_src/wrapper.h")
        .clang_arg("-x")
        .clang_arg("c++")
        .opaque_type("std::.*")
        .allowlist_type("cml::SimpleLlamaModelLoader")
        .allowlist_type("cml::SimpleTransformerLayer")
        .allowlist_type("llama_hparams")
        .allowlist_function("cml::simple_transformer_layer_delete")
        .allowlist_function("cml::simple_transformer_layer_forward")
        .allowlist_function("cml::simple_transformer_layer_reset")
        .allowlist_function("cml::baseline::create_llama_layer")
        .allowlist_function("cml::baseline::create_llama_final_layer")
        .allowlist_function("cml::cuda::create_llama_layer")
        .allowlist_function("cml::hip::create_llama_layer")
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
