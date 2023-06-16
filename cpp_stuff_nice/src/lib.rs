use cpp_stuff::*;

pub struct SimpleLlamaModelLoader(cml_SimpleLlamaModelLoader);

impl SimpleLlamaModelLoader {
    pub fn new(fname_base: &str) -> Self {
        let fname_base = std::ffi::CString::new(fname_base).unwrap();
        unsafe {
            SimpleLlamaModelLoader(cpp_stuff::cml_SimpleLlamaModelLoader::new(
                fname_base.as_ptr(),
            ))
        }
    }

    pub fn n_hidden(&self) -> u32 {
        unsafe { (*self.0.get_hparams()).n_embd }
    }
}

impl Drop for SimpleLlamaModelLoader {
    fn drop(&mut self) {
        unsafe { self.0.destruct() }
    }
}

pub struct SimpleTransformerLayer(*mut cml_SimpleTransformerLayer);

impl SimpleTransformerLayer {
    pub fn forward(&mut self, hidden_in: &[f32], hidden_out: &mut [f32]) {
        assert!(hidden_in.len() == hidden_out.len());
        unsafe {
            cml_simple_transformer_layer_forward(
                self.0,
                hidden_in.len().try_into().unwrap(),
                hidden_in.as_ptr(),
                hidden_out.as_mut_ptr(),
            )
        }
    }

    pub fn reset(&mut self) {
        unsafe { cml_simple_transformer_layer_reset(self.0) }
    }
}

impl Drop for SimpleTransformerLayer {
    fn drop(&mut self) {
        unsafe { cml_simple_transformer_layer_delete(self.0) }
    }
}

pub mod baseline {
    use super::*;

    pub fn create_llama_layer(
        loader: &mut SimpleLlamaModelLoader,
        layer_index: u32,
    ) -> SimpleTransformerLayer {
        unsafe {
            SimpleTransformerLayer(cml_baseline_create_llama_layer(&mut loader.0, layer_index))
        }
    }
}

pub mod cuda {
    use super::*;

    pub fn create_llama_layer(
        loader: &mut SimpleLlamaModelLoader,
        layer_index: u32,
    ) -> SimpleTransformerLayer {
        unsafe { SimpleTransformerLayer(cml_cuda_create_llama_layer(&mut loader.0, layer_index)) }
    }
}

pub mod hip {
    use super::*;

    pub fn create_llama_layer(
        loader: &mut SimpleLlamaModelLoader,
        layer_index: u32,
    ) -> SimpleTransformerLayer {
        unsafe { SimpleTransformerLayer(cml_hip_create_llama_layer(&mut loader.0, layer_index)) }
    }
}
