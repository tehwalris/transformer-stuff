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

    pub fn n_layers(&self) -> u32 {
        unsafe { (*self.0.get_hparams()).n_layer }
    }

    pub fn n_vocab(&self) -> u32 {
        unsafe { (*self.0.get_hparams()).n_vocab }
    }

    pub fn n_context(&self) -> u32 {
        unsafe { (*self.0.get_hparams()).n_ctx }
    }
}

impl Drop for SimpleLlamaModelLoader {
    fn drop(&mut self) {
        unsafe { self.0.destruct() }
    }
}

pub struct SimpleTransformerLayer(*mut cml_SimpleTransformerLayer);

impl SimpleTransformerLayer {
    pub fn forward(&mut self, hidden_in: &[f32], hidden_out: &mut [f32], path: &[u32]) {
        unsafe {
            cml_simple_transformer_layer_forward(
                self.0,
                hidden_in.len().try_into().unwrap(),
                hidden_in.as_ptr(),
                hidden_out.len().try_into().unwrap(),
                hidden_out.as_mut_ptr(),
                path.len().try_into().unwrap(),
                path.as_ptr(),
            )
        }
    }

    pub fn next_i(&self) -> u32 {
        unsafe { cml_simple_transformer_layer_next_i(self.0) }
    }

    pub fn retain(&mut self, indices: &[u32]) {
        unsafe {
            cml_simple_transformer_layer_retain(
                self.0,
                indices.len().try_into().unwrap(),
                indices.as_ptr(),
            )
        }
    }
}

impl Drop for SimpleTransformerLayer {
    fn drop(&mut self) {
        unsafe { cml_simple_transformer_layer_delete(self.0) }
    }
}

unsafe impl Send for SimpleTransformerLayer {}

pub mod baseline {
    use super::*;

    pub fn create_llama_layer(
        loader: &mut SimpleLlamaModelLoader,
        layer_index: u32,
        n_cache: u32,
    ) -> SimpleTransformerLayer {
        unsafe {
            SimpleTransformerLayer(cml_baseline_create_llama_layer(
                &mut loader.0,
                layer_index,
                n_cache,
            ))
        }
    }

    pub fn create_llama_final_layer(loader: &mut SimpleLlamaModelLoader) -> SimpleTransformerLayer {
        unsafe { SimpleTransformerLayer(cml_baseline_create_llama_final_layer(&mut loader.0)) }
    }
}

pub mod cuda {
    use super::*;

    pub fn create_llama_layer(
        loader: &mut SimpleLlamaModelLoader,
        layer_index: u32,
        n_cache: u32,
    ) -> SimpleTransformerLayer {
        unsafe {
            SimpleTransformerLayer(cml_cuda_create_llama_layer(
                &mut loader.0,
                layer_index,
                n_cache,
            ))
        }
    }

    pub fn create_llama_final_layer(loader: &mut SimpleLlamaModelLoader) -> SimpleTransformerLayer {
        unsafe { SimpleTransformerLayer(cml_cuda_create_llama_final_layer(&mut loader.0)) }
    }
}

pub mod hip {
    use super::*;

    pub fn create_llama_layer(
        loader: &mut SimpleLlamaModelLoader,
        layer_index: u32,
        n_cache: u32,
    ) -> SimpleTransformerLayer {
        unsafe {
            SimpleTransformerLayer(cml_hip_create_llama_layer(
                &mut loader.0,
                layer_index,
                n_cache,
            ))
        }
    }
}
