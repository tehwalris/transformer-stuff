use std::path::{Path, PathBuf};
use std::thread;
use std::{self, fs::File};

use anyhow::{anyhow, Result};
use indicatif::ProgressBar;
use memmap2::MmapOptions;
use rayon::prelude::*;
use tokenizers::Tokenizer;
use tracing::span;

use cpp_stuff_nice::{LlamaHyperparams, SimpleTransformerLayer};

use crate::loader::GPTQLlamaLoader;
use crate::vocab::VocabEmbeddings;

pub enum Backend {
    Baseline,
    Cuda,
    Hip,
}

impl Backend {
    pub fn create_llama_layer_ggml(
        &self,
        loader: &mut cpp_stuff_nice::SimpleLlamaModelLoader,
        i_layer: usize,
        n_cache: usize,
    ) -> SimpleTransformerLayer {
        let i_layer = i_layer.try_into().unwrap();
        let n_cache = n_cache.try_into().unwrap();
        match self {
            Backend::Baseline => {
                panic!("GGML is not supported with the baseline backend")
            }
            Backend::Cuda => {
                panic!("GGML is not supported with the CUDA backend")
            }
            Backend::Hip => cpp_stuff_nice::hip::create_llama_layer(loader, i_layer, n_cache),
        }
    }
}

pub struct Model {
    pub n_hidden: usize,
    pub n_vocab: usize,
    pub n_context: usize,
    pub n_cache: usize,
    pub layers: Vec<SimpleTransformerLayer>,
    pub final_layer: SimpleTransformerLayer,
}

impl Model {
    pub fn load(path: impl AsRef<Path>, layer_backends: &[Backend], show_progress: bool) -> Self {
        let path = path.as_ref().to_str().unwrap().to_owned();

        let loader = cpp_stuff_nice::SimpleLlamaModelLoader::new(&path);
        let n_hidden = usize::try_from(loader.n_hidden()).unwrap();
        let n_layers = usize::try_from(loader.n_layers()).unwrap();
        let n_vocab = usize::try_from(loader.n_vocab()).unwrap();
        let n_context = usize::try_from(loader.n_context()).unwrap();
        drop(loader);

        let progress_bar = if show_progress {
            ProgressBar::new(n_layers.try_into().unwrap())
        } else {
            ProgressBar::hidden()
        };

        assert_eq!(layer_backends.len(), n_layers);

        let path_for_final_layer = path.clone();
        let final_layer_thread = thread::spawn(move || {
            let mut loader = cpp_stuff_nice::SimpleLlamaModelLoader::new(&path_for_final_layer);
            // cpp_stuff_nice::cuda::create_llama_final_layer(&mut loader)
            panic!("No backend supports a GGML final layer")
        });

        let n_cache = n_context * 2; // arbitrary choice
        let layers: Vec<SimpleTransformerLayer> = layer_backends
            .par_iter()
            .enumerate()
            .map(|(i_layer, backend)| {
                let span = span!(tracing::Level::INFO, "load_layer", i_layer);
                let _enter = span.enter();
                let mut loader = cpp_stuff_nice::SimpleLlamaModelLoader::new(&path);
                let layer = backend.create_llama_layer_ggml(&mut loader, i_layer, n_cache);
                progress_bar.inc(1);
                layer
            })
            .collect();

        let final_layer = final_layer_thread.join().unwrap();

        Self {
            n_hidden,
            n_vocab,
            n_context,
            n_cache,
            layers,
            final_layer,
        }
    }

    pub fn load_gptq(
        path: impl AsRef<Path>,
        params: LlamaHyperparams,
        n_cache: usize,
    ) -> Result<(Self, Tokenizer, VocabEmbeddings)> {
        let path = path.as_ref();
        let weights_path = path.join(format!(
            "gptq_model-4bit-{}g.safetensors",
            params.gptq_block_size
        ));
        let tokenizer_path = path.join("tokenizer.json");

        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|err| anyhow!(err))?;

        let weights_buffer = {
            let file = File::open(weights_path)?;
            unsafe { MmapOptions::new().map(&file)? }
        };
        let loader = GPTQLlamaLoader::new(&weights_buffer, params)?;

        let vocab_embeddings = loader.load_vocab_embeddings()?;

        let layers = (0..(params.n_layers as usize))
            .map(|layer_index| {
                let layer_weights = loader.load_layer(layer_index)?;
                Ok(cpp_stuff_nice::cuda::create_llama_layer_gptq(
                    &layer_weights,
                    params,
                    n_cache,
                ))
            })
            .collect::<Result<Vec<_>>>()?;

        let final_layer = {
            let final_layer_weights = loader.load_final_layer()?;
            cpp_stuff_nice::cuda::create_llama_final_layer(&final_layer_weights, params)
        };

        let model = Self {
            n_hidden: params.n_hidden.try_into().unwrap(),
            n_vocab: params.n_vocab.try_into().unwrap(),
            n_context: params.n_context.try_into().unwrap(),
            n_cache: n_cache.try_into().unwrap(),
            layers,
            final_layer,
        };

        Ok((model, tokenizer, vocab_embeddings))
    }

    pub fn predict(&mut self, hidden_in: &[f32], path: &[u32]) -> Vec<f32> {
        assert_eq!(hidden_in.len(), self.n_hidden);
        assert!(path.len() > 0);
        assert!(path.len() <= self.n_context);
        assert!(*path.last().unwrap() == self.next_i());

        let mut hidden_in = hidden_in.to_vec();
        let mut hidden_out = vec![0.0; self.n_hidden];

        for (i_layer, layer) in self.layers.iter_mut().enumerate() {
            hidden_out.fill(0.0);

            {
                let span = span!(tracing::Level::INFO, "layer_forward", i_layer);
                let _enter = span.enter();
                layer.forward(&mut hidden_in, &mut hidden_out, path);
            }

            std::mem::swap(&mut hidden_in, &mut hidden_out);
        }

        let mut final_out = vec![0.0; self.n_vocab];
        {
            let span = span!(tracing::Level::INFO, "final_layer_forward");
            let _enter = span.enter();
            self.final_layer
                .forward(&mut hidden_in, &mut final_out, path);
        }

        final_out
    }

    pub fn next_i(&self) -> u32 {
        self.layers.first().unwrap().next_i()
    }

    pub fn retain(&mut self, indices: &[usize]) {
        let indices: Vec<u32> = indices.iter().map(|&i| i.try_into().unwrap()).collect();
        assert!(indices.len() <= self.n_cache);
        for layer in &mut self.layers {
            layer.retain(&indices);
        }
        self.final_layer.retain(&indices);
    }
}
