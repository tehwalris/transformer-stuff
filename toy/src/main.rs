fn do_thing() {
    unsafe {
        let fname_base = std::ffi::CString::new("test").unwrap();
        let mut x = cpp_stuff::cml_SimpleLlamaModelLoader::new(fname_base.as_ptr());
        x.destruct();
    }
}

fn main() {
    println!("Hello, world!");
    do_thing();
    println!("Goodbye, world!")
}
