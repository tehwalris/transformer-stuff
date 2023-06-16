fn do_thing(model_path: &str) {
    unsafe {
        let fname_base = std::ffi::CString::new(model_path).unwrap();
        let mut loader = cpp_stuff::cml_SimpleLlamaModelLoader::new(fname_base.as_ptr());
        let cuda_model = cpp_stuff::cml_cuda_create_llama_layer(&mut loader, 0);
        let hip_model = cpp_stuff::cml_hip_create_llama_layer(&mut loader, 0);
        loader.destruct();
        cpp_stuff::cml_delete_simple_transformer_layer(cuda_model);
        cpp_stuff::cml_delete_simple_transformer_layer(hip_model);
    }
}

fn main() {
    // read model path from first argument
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        println!("Usage: toy <model_path>");
        return;
    }
    let model_path = &args[1];

    println!("Hello, world!");
    do_thing(model_path);
    println!("Goodbye, world!")
}
