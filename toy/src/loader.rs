use anyhow::{anyhow, Result};
use cpp_stuff_nice::{GPTQMatrix, LlamaGPTQLayerWeights, LlamaHyperparams};

use safetensors::SafeTensors;

pub struct GPTQLlamaLoader<'a> {
    tensors: SafeTensors<'a>,
    params: LlamaHyperparams,
    gptq_block_size: usize,
}

impl<'a> GPTQLlamaLoader<'a> {
    pub fn new(buffer: &'a [u8], gptq_block_size: usize) -> Result<Self> {
        if gptq_block_size % 8 != 0 {
            return Err(anyhow!(
                "gptq_block_size must be a multiple of 8, got {}",
                gptq_block_size
            ));
        }

        let tensors = SafeTensors::deserialize(&buffer)?;
        let params = Self::params_from_tensors(&tensors)?;
        Ok(Self {
            tensors,
            params,
            gptq_block_size,
        })
    }

    fn params_from_tensors(tensors: &SafeTensors) -> Result<LlamaHyperparams> {
        Err(anyhow::anyhow!("TODO"))
    }

    pub fn params(&self) -> &LlamaHyperparams {
        &self.params
    }

    pub fn load_layer(&self, i_layer: usize) -> Result<LlamaGPTQLayerWeights<'_>> {
        let get_name = |suffix| format!("model.layers.{}.{}", i_layer, suffix);

        let n_hidden: usize = self.params.n_hidden.try_into().unwrap();
        let n_context: usize = self.params.n_context.try_into().unwrap();
        let n_heads: usize = self.params.n_heads.try_into().unwrap();
        let n_ff: usize = self.params.n_ff.try_into().unwrap();

        let input_layernorm =
            self.load_1d_tensor_with_shape(&get_name("input_layernorm"), n_hidden)?;
        let self_attn_q_proj =
            self.load_gptq_matrix_with_shape(&get_name("self_attn_q_proj"), n_hidden, n_hidden)?;
        let self_attn_k_proj =
            self.load_gptq_matrix_with_shape(&get_name("self_attn_k_proj"), n_hidden, n_hidden)?;
        let self_attn_v_proj =
            self.load_gptq_matrix_with_shape(&get_name("self_attn_v_proj"), n_hidden, n_hidden)?;
        let self_attn_o_proj =
            self.load_gptq_matrix_with_shape(&get_name("self_attn_o_proj"), n_hidden, n_hidden)?;
        let post_attention_layernorm =
            self.load_1d_tensor_with_shape(&get_name("post_attention_layernorm"), n_hidden)?;
        let mlp_up_proj =
            self.load_gptq_matrix_with_shape(&get_name("mlp_up_proj"), n_hidden, n_ff)?;
        let mlp_gate_proj =
            self.load_gptq_matrix_with_shape(&get_name("mlp_gate_proj"), n_hidden, n_ff)?;
        let mlp_down_proj =
            self.load_gptq_matrix_with_shape(&get_name("mlp_down_proj"), n_ff, n_hidden)?;

        Ok(LlamaGPTQLayerWeights::new(
            input_layernorm,
            self_attn_q_proj,
            self_attn_k_proj,
            self_attn_v_proj,
            self_attn_o_proj,
            post_attention_layernorm,
            mlp_up_proj,
            mlp_gate_proj,
            mlp_down_proj,
        ))
    }

    pub fn load_gptq_matrix<'b>(&'b self, name_prefix: &str) -> Result<GPTQMatrix<'b>> {
        let (qweight, qweight_shape) = self.load_2d_tensor(&format!("{}.qweight", name_prefix))?;
        let (qzeros, qzeros_shape) = self.load_2d_tensor(&format!("{}.qzeros", name_prefix))?;
        let (scales, scales_shape) = self.load_2d_tensor(&format!("{}.scales", name_prefix))?;

        let rows = qweight_shape.1;
        let cols = qweight_shape.0 * 8;

        if rows % self.gptq_block_size != 0 || cols % self.gptq_block_size != 0 {
            return Err(anyhow!(
                "GPTQ matrix {} has shape ({}, {}), which is not a multiple of gptq_block_size {}",
                name_prefix,
                rows,
                cols,
                self.gptq_block_size,
            ));
        }

        let expected_qweight_shape = (cols / 8, rows);
        let expected_qzeros_shape = (cols / self.gptq_block_size, rows / 8);
        let expected_scales_shape = (cols / 8, rows);

        if qweight_shape != expected_qweight_shape {
            return Err(anyhow!(
                "Expected {}.qweight shape {:?}, got {:?}",
                name_prefix,
                expected_qweight_shape,
                qweight_shape
            ));
        }
        if qzeros_shape != expected_qzeros_shape {
            return Err(anyhow!(
                "Expected {}.qzeros shape {:?}, got {:?}",
                name_prefix,
                expected_qzeros_shape,
                qzeros_shape
            ));
        }
        if scales_shape != expected_scales_shape {
            return Err(anyhow!(
                "Expected {}.scales shape {:?}, got {:?}",
                name_prefix,
                expected_scales_shape,
                scales_shape
            ));
        }

        Ok(GPTQMatrix::new(
            rows,
            cols,
            self.gptq_block_size,
            qweight,
            qzeros,
            scales,
        ))
    }

    fn load_gptq_matrix_with_shape<'b>(
        &'b self,
        name_prefix: &str,
        expected_rows: usize,
        expected_cols: usize,
    ) -> Result<GPTQMatrix<'b>> {
        let matrix = self.load_gptq_matrix(name_prefix)?;
        let actual_rows = matrix.rows();
        let actual_cols = matrix.cols();
        if actual_rows != expected_rows || actual_cols != expected_cols {
            return Err(anyhow!(
                "Expected matrix {} to have shape (rows: {}, cols: {}), got (rows: {}, cols: {})",
                name_prefix,
                expected_rows,
                expected_cols,
                actual_rows,
                actual_cols
            ));
        }
        Ok(matrix)
    }

    fn load_1d_tensor<'b>(&'b self, name: &str) -> Result<(&'b [u8], usize)> {
        let tensor = self.tensors.tensor(&name)?;
        let shape = tensor.shape();
        if shape.len() != 1 {
            return Err(anyhow!("Expected 1D tensor, got {:?}", shape));
        }
        let len = shape[0];
        let data = tensor.data();
        Ok((data, len))
    }

    fn load_1d_tensor_with_shape<'b>(
        &'b self,
        name: &str,
        expected_len: usize,
    ) -> Result<&'b [u8]> {
        let (data, actual_len) = self.load_1d_tensor(name)?;
        if actual_len != expected_len {
            return Err(anyhow!(
                "Expected tensor {} to have length {}, got {}",
                name,
                expected_len,
                actual_len
            ));
        }
        Ok(data)
    }

    fn load_2d_tensor<'b>(&'b self, name: &str) -> Result<(&'b [u8], (usize, usize))> {
        let tensor = self.tensors.tensor(&name)?;
        let shape = tensor.shape();
        if shape.len() != 2 {
            return Err(anyhow!("Expected 2D tensor, got {:?}", shape));
        }
        let cols = shape[0];
        let rows = shape[1];
        let data = tensor.data();
        Ok((data, (rows, cols)))
    }
}
