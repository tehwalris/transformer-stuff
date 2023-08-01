use anyhow::{anyhow, Result};
use cpp_stuff_nice::{GPTQMatrix, LlamaFinalLayerWeights, LlamaGPTQLayerWeights, LlamaHyperparams};

use half::f16;
use safetensors::SafeTensors;

use crate::vocab::VocabEmbeddings;

pub struct GPTQLlamaLoader<'a> {
    tensors: SafeTensors<'a>,
    params: LlamaHyperparams,
}

impl<'a> GPTQLlamaLoader<'a> {
    pub fn new(buffer: &'a [u8], params: LlamaHyperparams) -> Result<Self> {
        if params.gptq_block_size % 8 != 0 {
            return Err(anyhow!(
                "gptq_block_size must be a multiple of 8, got {}",
                params.gptq_block_size
            ));
        }

        /*
        lm_head.weight [32000, 4096] F16
        model.embed_tokens.weight [32000, 4096] F16
        model.norm.weight [4096] F16
        */

        let tensors = SafeTensors::deserialize(&buffer)?;
        Ok(Self { tensors, params })
    }

    pub fn params(&self) -> &LlamaHyperparams {
        &self.params
    }

    pub fn load_layer(&self, i_layer: usize) -> Result<LlamaGPTQLayerWeights<'_>> {
        let get_name = |suffix| format!("model.layers.{}.{}", i_layer, suffix);

        let n_hidden: usize = self.params.n_hidden.try_into().unwrap();
        let n_ff: usize = self.params.n_ff.try_into().unwrap();

        let input_layernorm =
            self.load_1d_tensor_with_shape(&get_name("input_layernorm.weight"), n_hidden)?;
        let self_attn_q_proj =
            self.load_gptq_matrix_with_shape(&get_name("self_attn.q_proj"), n_hidden, n_hidden)?;
        let self_attn_k_proj =
            self.load_gptq_matrix_with_shape(&get_name("self_attn.k_proj"), n_hidden, n_hidden)?;
        let self_attn_v_proj =
            self.load_gptq_matrix_with_shape(&get_name("self_attn.v_proj"), n_hidden, n_hidden)?;
        let self_attn_o_proj =
            self.load_gptq_matrix_with_shape(&get_name("self_attn.o_proj"), n_hidden, n_hidden)?;
        let post_attention_layernorm =
            self.load_1d_tensor_with_shape(&get_name("post_attention_layernorm.weight"), n_hidden)?;
        let mlp_up_proj =
            self.load_gptq_matrix_with_shape(&get_name("mlp.up_proj"), n_ff, n_hidden)?;
        let mlp_gate_proj =
            self.load_gptq_matrix_with_shape(&get_name("mlp.gate_proj"), n_ff, n_hidden)?;
        let mlp_down_proj =
            self.load_gptq_matrix_with_shape(&get_name("mlp.down_proj"), n_hidden, n_ff)?;

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

    pub fn debug_gptq(&self) -> Result<()> {
        let matrix_name = "model.layers.0.mlp.down_proj";
        // let w = self.load_gptq_matrix(matrix_name)?;
        // let (rows, columns) = (w.rows(), w.cols());
        // println!("rows: {}, columns: {}", rows, columns);

        let (qweight, qweight_shape) = self.load_2d_tensor(&format!("{}.qweight", matrix_name))?;
        let (qzeros, qzeros_shape) = self.load_2d_tensor(&format!("{}.qzeros", matrix_name))?;
        let (scales, scales_shape) = self.load_2d_tensor(&format!("{}.scales", matrix_name))?;

        println!("qweight: {:?}", qweight_shape);
        println!("qzeros: {:?}", qzeros_shape);
        println!("scales: {:?}", scales_shape);

        let gptq_block_size: usize = self.params.gptq_block_size.try_into().unwrap();
        let get_weight = |i_row: usize, i_column: usize| -> f32 {
            let i_qweight = (i_row / 8) * qweight_shape.1 + i_column;
            let i_qzeros = (i_row / gptq_block_size) * qzeros_shape.1 + (i_column / 8);
            let i_scales = (i_row / gptq_block_size) * scales_shape.1 + i_column;

            let bytes_qweight: [u8; 4] = [
                qweight[4 * i_qweight + 0],
                qweight[4 * i_qweight + 1],
                qweight[4 * i_qweight + 2],
                qweight[4 * i_qweight + 3],
            ];
            let bytes_qzeros: [u8; 4] = [
                qzeros[4 * i_qzeros + 0],
                qzeros[4 * i_qzeros + 1],
                qzeros[4 * i_qzeros + 2],
                qzeros[4 * i_qzeros + 3],
            ];
            let bytes_scales: [u8; 2] = [scales[2 * i_scales + 0], scales[2 * i_scales + 1]];

            let value_qweight = (i32::from_le_bytes(bytes_qweight) >> ((i_row % 8) * 4)) & 0xF;
            let value_qzeros = (i32::from_le_bytes(bytes_qzeros) >> ((i_column % 8) * 4)) & 0xF;
            let value_scales = f16::from_le_bytes(bytes_scales);

            return (value_qweight - value_qzeros - 1) as f32 * value_scales.to_f32();
        };

        /*
        >>> w = model['layers.0.feed_forward.w2.weight']
        >>> w[0:3, 0:3]
        tensor([[ 0.0027, -0.0145,  0.0083],
                [ 0.0046, -0.0042,  0.0090],
                [ 0.0020,  0.0339, -0.0044]], dtype=torch.bfloat16)
        */

        for i_row in 0..3 {
            for i_column in 0..3 {
                print!("{:8.4} ", get_weight(i_row, i_column));
            }
            println!();
        }

        Ok(())
    }

    pub fn load_final_layer(&self) -> Result<LlamaFinalLayerWeights<'_>> {
        let n_hidden: usize = self.params.n_hidden.try_into().unwrap();
        let n_vocab: usize = self.params.n_vocab.try_into().unwrap();

        let norm = self.load_1d_tensor_with_shape("model.norm.weight", n_hidden)?;
        let (lm_head, lm_head_shape) = self.load_2d_tensor("lm_head.weight")?;
        if lm_head_shape != (n_vocab, n_hidden) {
            return Err(anyhow!(
                "lm_head.weight has shape {:?}, expected ({}, {})",
                lm_head_shape,
                n_vocab,
                n_hidden
            ));
        }

        Ok(LlamaFinalLayerWeights::new(norm, lm_head))
    }

    pub fn load_vocab_embeddings(&self) -> Result<VocabEmbeddings> {
        let tensor = self.tensors.tensor("model.embed_tokens.weight")?;
        Ok(VocabEmbeddings::load_from_safetensors(
            self.params.n_hidden.try_into().unwrap(),
            self.params.n_vocab.try_into().unwrap(),
            tensor,
        ))
    }

    fn load_gptq_matrix<'b>(&'b self, name_prefix: &str) -> Result<GPTQMatrix<'b>> {
        let gptq_block_size = self.params.gptq_block_size.try_into().unwrap();

        let (qweight, qweight_shape) = self.load_2d_tensor(&format!("{}.qweight", name_prefix))?;
        let (qzeros, qzeros_shape) = self.load_2d_tensor(&format!("{}.qzeros", name_prefix))?;
        let (scales, scales_shape) = self.load_2d_tensor(&format!("{}.scales", name_prefix))?;

        let rows = qweight_shape.1;
        let cols = qweight_shape.0 * 8;

        if rows % gptq_block_size != 0 || cols % gptq_block_size != 0 {
            return Err(anyhow!(
                "GPTQ matrix {} has shape ({}, {}), which is not a multiple of gptq_block_size {}",
                name_prefix,
                rows,
                cols,
                gptq_block_size,
            ));
        }

        let expected_qweight_shape = (cols / 8, rows);
        let expected_qzeros_shape = (cols / gptq_block_size, rows / 8);
        let expected_scales_shape = (cols / gptq_block_size, rows);

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
            gptq_block_size,
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
        let data = tensor.data();
        Ok((data, shape[0]))
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
        let data = tensor.data();
        Ok((data, (shape[0], shape[1])))
    }
}
