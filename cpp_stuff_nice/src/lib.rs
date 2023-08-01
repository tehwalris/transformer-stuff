use std::marker::PhantomData;

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

pub struct GPTQMatrix<'a> {
    inner: cml_GPTQMatrix,
    phantom: PhantomData<&'a [u8]>,
}

impl<'a> GPTQMatrix<'a> {
    pub fn new(
        rows: usize,
        cols: usize,
        block_size: usize,
        qweight: &'a [u8],
        qzeros: &'a [u8],
        scales: &'a [u8],
    ) -> Self {
        assert!(block_size % 8 == 0);
        assert!(rows % block_size == 0);
        assert!(cols % block_size == 0);
        assert!(qweight.len() == 4 * cols / 8 * rows);
        assert!(qzeros.len() == 4 * cols / block_size * rows / 8);
        assert!(scales.len() == 2 * cols / block_size * rows);
        GPTQMatrix {
            inner: cml_GPTQMatrix {
                rows: rows.try_into().unwrap(),
                cols: cols.try_into().unwrap(),
                block_size: block_size.try_into().unwrap(),
                qweight: qweight.as_ptr() as *mut u32,
                qzeros: qzeros.as_ptr() as *mut u32,
                scales: scales.as_ptr() as *mut u16,
            },
            phantom: PhantomData,
        }
    }

    pub fn rows(&self) -> usize {
        self.inner.rows.try_into().unwrap()
    }

    pub fn cols(&self) -> usize {
        self.inner.cols.try_into().unwrap()
    }
}

pub struct LlamaGPTQLayerWeights<'a> {
    inner: cml_LlamaGPTQLayerWeights,
    phantom: PhantomData<&'a [u8]>,
}

impl<'a> LlamaGPTQLayerWeights<'a> {
    pub fn new(
        input_layernorm: &'a [u8],
        self_attn_q_proj: GPTQMatrix<'a>,
        self_attn_k_proj: GPTQMatrix<'a>,
        self_attn_v_proj: GPTQMatrix<'a>,
        self_attn_o_proj: GPTQMatrix<'a>,
        post_attention_layernorm: &'a [u8],
        mlp_up_proj: GPTQMatrix<'a>,
        mlp_gate_proj: GPTQMatrix<'a>,
        mlp_down_proj: GPTQMatrix<'a>,
    ) -> Self {
        let hidden_size: usize = self_attn_q_proj.inner.cols.try_into().unwrap();
        assert!(input_layernorm.len() == hidden_size * 2);
        assert!(post_attention_layernorm.len() == hidden_size * 2);

        LlamaGPTQLayerWeights {
            inner: cml_LlamaGPTQLayerWeights {
                input_layernorm: input_layernorm.as_ptr() as *mut u16,
                self_attn_q_proj: self_attn_q_proj.inner,
                self_attn_k_proj: self_attn_k_proj.inner,
                self_attn_v_proj: self_attn_v_proj.inner,
                self_attn_o_proj: self_attn_o_proj.inner,
                post_attention_layernorm: post_attention_layernorm.as_ptr() as *mut u16,
                mlp_up_proj: mlp_up_proj.inner,
                mlp_gate_proj: mlp_gate_proj.inner,
                mlp_down_proj: mlp_down_proj.inner,
            },
            phantom: PhantomData,
        }
    }
}

pub struct LlamaFinalLayerWeights<'a> {
    inner: cml_LlamaFinalLayerWeights,
    phantom: PhantomData<&'a [u8]>,
}

impl<'a> LlamaFinalLayerWeights<'a> {
    pub fn new(norm: &'a [u8], lm_head: &'a [u8]) -> Self {
        assert!(norm.len() % 2 == 0);
        assert!(lm_head.len() % 2 == 0);
        assert!(lm_head.len() % norm.len() == 0);

        LlamaFinalLayerWeights {
            inner: cml_LlamaFinalLayerWeights {
                norm: norm.as_ptr() as *mut u16,
                lm_head: lm_head.as_ptr() as *mut u16,
            },
            phantom: PhantomData,
        }
    }
}

pub type LlamaHyperparams = cml_LlamaHyperparams;

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

    pub fn create_llama_layer_gptq(
        loader_weights: &LlamaGPTQLayerWeights,
        params: LlamaHyperparams,
        n_cache: usize,
    ) -> SimpleTransformerLayer {
        unsafe {
            SimpleTransformerLayer(cml_baseline_create_llama_layer_gptq(
                &loader_weights.inner,
                params,
                n_cache.try_into().unwrap(),
            ))
        }
    }

    pub fn create_llama_final_layer(
        loader_weights: &LlamaFinalLayerWeights,
        params: LlamaHyperparams,
    ) -> SimpleTransformerLayer {
        unsafe {
            SimpleTransformerLayer(cml_baseline_create_llama_final_layer(
                &loader_weights.inner,
                params,
            ))
        }
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
