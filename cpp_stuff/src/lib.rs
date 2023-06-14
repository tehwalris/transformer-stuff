#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

pub fn do_thing() {
    unsafe {
        let fname_base = std::ffi::CString::new("test").unwrap();
        let mut x = cml_SimpleLlamaModelLoader::new(fname_base.as_ptr());
        x.destruct();
    }
}
