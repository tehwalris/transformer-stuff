use std;
use std::path::Path;
use std::thread;

use indicatif::ProgressBar;
use rayon::prelude::*;

use cpp_stuff_nice::SimpleTransformerLayer;
use tracing::span;

pub(crate) enum Backend {
    Baseline,
    Cuda,
    Hip,
}

impl Backend {
    pub(crate) fn create_llama_layer(
        &self,
        loader: &mut cpp_stuff_nice::SimpleLlamaModelLoader,
        i_layer: usize,
        n_cache: usize,
    ) -> SimpleTransformerLayer {
        let i_layer = i_layer.try_into().unwrap();
        let n_cache = n_cache.try_into().unwrap();
        match self {
            Backend::Baseline => {
                cpp_stuff_nice::baseline::create_llama_layer(loader, i_layer, n_cache)
            }
            Backend::Cuda => cpp_stuff_nice::cuda::create_llama_layer(loader, i_layer, n_cache),
            Backend::Hip => cpp_stuff_nice::hip::create_llama_layer(loader, i_layer, n_cache),
        }
    }
}

pub(crate) struct Model {
    pub(crate) n_hidden: usize,
    pub(crate) n_vocab: usize,
    pub(crate) n_context: usize,
    pub(crate) layers: Vec<SimpleTransformerLayer>,
    pub(crate) final_layer: SimpleTransformerLayer,
}

impl Model {
    pub(crate) fn load(
        path: impl AsRef<Path>,
        layer_backends: &[Backend],
        show_progress: bool,
    ) -> Self {
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
            cpp_stuff_nice::cuda::create_llama_final_layer(&mut loader)
        });

        let n_cache = n_context * 2; // arbitrary choice
        let layers: Vec<SimpleTransformerLayer> = layer_backends
            .par_iter()
            .enumerate()
            .map(|(i_layer, backend)| {
                let span = span!(tracing::Level::INFO, "load_layer", i_layer);
                let _enter = span.enter();
                let mut loader = cpp_stuff_nice::SimpleLlamaModelLoader::new(&path);
                let layer = backend.create_llama_layer(&mut loader, i_layer, n_cache);
                progress_bar.inc(1);
                layer
            })
            .collect();

        let final_layer = final_layer_thread.join().unwrap();

        Self {
            n_hidden,
            n_vocab,
            n_context,
            layers,
            final_layer,
        }
    }

    pub(crate) fn predict(&mut self, hidden_in: &[f32], path: &[u32]) -> Vec<f32> {
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

    pub(crate) fn next_i(&self) -> u32 {
        self.layers.first().unwrap().next_i()
    }

    pub(crate) fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
        self.final_layer.reset();
    }
}
