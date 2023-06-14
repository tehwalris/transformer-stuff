fn do_thing(model_path: &str) {
    unsafe {
        let fname_base = std::ffi::CString::new(model_path).unwrap();
        let mut x = cpp_stuff::cml_SimpleLlamaModelLoader::new(fname_base.as_ptr());
        x.destruct();
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
