#![feature(exit_status_error)]

use std::{
    env,
    path::{Path, PathBuf},
    process::Command,
};

fn out_relative_path(suffix: impl AsRef<Path>) -> PathBuf {
    PathBuf::from(env::var("OUT_DIR").unwrap()).join(suffix)
}

fn manifest_relative_path(suffix: impl AsRef<Path>) -> PathBuf {
    PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join(suffix)
}

fn escape_ninja_path(p: impl AsRef<Path>) -> String {
    let s = p.as_ref().to_str().unwrap();
    s.replace('$', "$$").replace(' ', "$ ").replace(':', "$:")
}

struct NinjaVariableList(Vec<String>);

impl NinjaVariableList {
    fn new() -> Self {
        Self(vec![])
    }

    fn push_var(&mut self, name: &str, value: &str) {
        self.0.push(format!("{} = {}", name, value));
    }

    fn push_empty_line(&mut self) {
        self.0.push("".to_owned());
    }
}

fn generate_ninja_variables(ninja_builddir: impl AsRef<Path>) -> String {
    let mut lines = NinjaVariableList::new();

    lines.push_var("ninja_required_version", "1.3");
    lines.push_empty_line();

    lines.push_var(
        "root",
        &escape_ninja_path(manifest_relative_path("cpp_src")),
    );
    lines.push_var("builddir", &escape_ninja_path(ninja_builddir));
    lines.push_empty_line();

    if let Some(rocthrust_path) = env::var("ROCTHRUST_PATH").ok() {
        lines.push_var(
            "rocthrust_include_flags",
            &format!(
                "-isystem {}",
                escape_ninja_path(PathBuf::from(rocthrust_path).join("include"))
            ),
        );
        lines.push_empty_line();
    }

    let opt_level = env::var("OPT_LEVEL").unwrap_or_else(|_| "0".to_owned());
    let debug = env::var("DEBUG").unwrap_or_else(|_| "false".to_owned()) != "false";
    let mut shared_compiler_flags = vec![format!("-O{}", opt_level), "-fPIC".to_owned()];
    if debug {
        shared_compiler_flags.push("-g".to_owned());
    }

    lines.push_var(
        "cpu_flags",
        "-march=native -mtune=native -mavx2 -mfma -mf16c",
    );
    lines.push_var("shared_compiler_flags", &shared_compiler_flags.join(" "));
    lines.push_empty_line();

    lines.0.join("\n")
}

fn main() {
    println!("cargo:rerun-if-changed=cpp_src");

    let ninja_builddir = out_relative_path("ninja_builddir");
    let ninja_variables = generate_ninja_variables(&ninja_builddir);
    let ninja_build_base = std::fs::read_to_string("build-base.ninja").unwrap();
    println!("cargo:rerun-if-changed=build-base.ninja");
    std::fs::write(
        out_relative_path("build.ninja"),
        [ninja_variables, ninja_build_base].join("\n") + "\n",
    )
    .unwrap();

    Command::new("ninja")
        .current_dir(&out_relative_path("."))
        .status()
        .unwrap()
        .exit_ok()
        .unwrap();

    println!(
        "cargo:rustc-link-search=native={}",
        ninja_builddir.join("lib").display()
    );
    println!("cargo:rustc-link-lib=static=cpp_stuff");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-lib=cublas");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=amdhip64");

    let bindings = bindgen::Builder::default()
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .header(
            manifest_relative_path("cpp_src/wrapper.h")
                .to_str()
                .unwrap(),
        )
        .clang_arg("-x")
        .clang_arg("c++")
        .opaque_type("std::.*")
        .allowlist_type("cml::SimpleLlamaModelLoader")
        .allowlist_type("cml::SimpleTransformerLayer")
        .allowlist_type("llama_hparams")
        .allowlist_function("cml::simple_transformer_layer_delete")
        .allowlist_function("cml::simple_transformer_layer_forward")
        .allowlist_function("cml::simple_transformer_layer_next_i")
        .allowlist_function("cml::simple_transformer_layer_reset")
        .allowlist_function("cml::baseline::create_llama_layer")
        .allowlist_function("cml::baseline::create_llama_final_layer")
        .allowlist_function("cml::cuda::create_llama_layer")
        .allowlist_function("cml::cuda::create_llama_final_layer")
        .allowlist_function("cml::hip::create_llama_layer")
        .generate()
        .expect("Unable to generate bindings");

    bindings
        .write_to_file(out_relative_path("bindings.rs"))
        .expect("Couldn't write bindings!");
}
