#![feature(iter_array_chunks)]

use std::{
    io::{self, BufRead, Seek, Write},
    path::Path,
    thread,
};

use cpp_stuff_nice::SimpleTransformerLayer;
use ggml::format::TensorLoadInfo;
use half::{f16, slice::HalfFloatSliceExt};
use indicatif::ProgressBar;
use llm_base::{TokenUtf8Buffer, Vocabulary};
use rayon::prelude::*;

enum Backend {
    Baseline,
    Cuda,
    Hip,
}

impl Backend {
    fn create_llama_layer(
        &self,
        loader: &mut cpp_stuff_nice::SimpleLlamaModelLoader,
        i_layer: usize,
    ) -> SimpleTransformerLayer {
        let i_layer = i_layer.try_into().unwrap();
        match self {
            Backend::Baseline => cpp_stuff_nice::baseline::create_llama_layer(loader, i_layer),
            Backend::Cuda => cpp_stuff_nice::cuda::create_llama_layer(loader, i_layer),
            Backend::Hip => cpp_stuff_nice::hip::create_llama_layer(loader, i_layer),
        }
    }
}

struct Model {
    n_hidden: usize,
    n_vocab: usize,
    n_context: usize,
    layers: Vec<SimpleTransformerLayer>,
    final_layer: SimpleTransformerLayer,
}

impl Model {
    fn load(path: impl AsRef<Path>, layer_backends: &[Backend], show_progress: bool) -> Self {
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
            cpp_stuff_nice::baseline::create_llama_final_layer(&mut loader)
        });

        let layers: Vec<SimpleTransformerLayer> = layer_backends
            .par_iter()
            .enumerate()
            .map(|(i_layer, backend)| {
                let mut loader = cpp_stuff_nice::SimpleLlamaModelLoader::new(&path);
                let layer = backend.create_llama_layer(&mut loader, i_layer);
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

    fn predict(&mut self, hidden_in: &[f32]) -> Vec<f32> {
        assert_eq!(hidden_in.len(), self.n_hidden);

        let mut hidden_in = hidden_in.to_vec();
        let mut hidden_out = vec![0.0; self.n_hidden];

        for layer in &mut self.layers {
            hidden_out.fill(0.0);
            layer.forward(&mut hidden_in, &mut hidden_out);
            std::mem::swap(&mut hidden_in, &mut hidden_out);
        }

        let mut final_out = vec![0.0; self.n_vocab];
        self.final_layer.forward(&mut hidden_in, &mut final_out);

        final_out
    }

    fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
        self.final_layer.reset();
    }
}

struct VocabEmbeddings {
    n_hidden: usize,
    n_vocab: usize,
    embeddings: Vec<f32>,
}

impl VocabEmbeddings {
    fn load_from_ggml<R: BufRead + Seek>(
        n_hidden: usize,
        n_vocab: usize,
        info: &TensorLoadInfo,
        reader: &mut R,
    ) -> Self {
        assert_eq!(info.dims(), &[n_hidden, n_vocab]); // This is in GGML order (contiguous index first)
        assert_eq!(info.element_type, ggml::ElementType::F16);

        let embeddings_f16: Vec<f16> = info
            .read_data(reader)
            .unwrap()
            .into_iter()
            .array_chunks()
            .map(f16::from_le_bytes)
            .collect();
        let mut embeddings_f32: Vec<f32> = vec![0.0; n_hidden * n_vocab];
        assert_eq!(embeddings_f16.len(), embeddings_f32.len());
        embeddings_f16.convert_to_f32_slice(&mut embeddings_f32);

        Self {
            n_hidden,
            n_vocab,
            embeddings: embeddings_f32,
        }
    }

    fn get_embedding(&self, i: usize) -> &[f32] {
        assert!(i < self.n_vocab);
        let offset = i * self.n_hidden;
        &self.embeddings[offset..offset + self.n_hidden]
    }
}

fn load_vocab(model_path: &str) -> (Vocabulary, VocabEmbeddings) {
    let mut loader = llm_base::Loader::<llm_llama::Hyperparameters, _>::new(|_| {});
    let mut file = std::io::BufReader::new(std::fs::File::open(model_path).unwrap());
    ggml::format::load(&mut file, &mut loader).unwrap();

    let n_hidden = loader.hyperparameters.n_embd;
    let n_vocab = loader.hyperparameters.n_vocab;

    let vocab_embeddings = VocabEmbeddings::load_from_ggml(
        n_hidden,
        n_vocab,
        loader
            .tensors
            .get(&"tok_embeddings.weight".to_string())
            .unwrap(),
        &mut file,
    );

    (loader.vocabulary, vocab_embeddings)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        println!("Usage: toy <model_path>");
        return;
    }
    let model_path = &args[1];

    let mut layer_backends = vec![];
    for _ in 0..16 {
        layer_backends.push(Backend::Cuda);
    }
    for _ in 0..16 {
        layer_backends.push(Backend::Hip);
    }

    println!("Loading model...");
    let (vocab, vocab_embeddings) = load_vocab(model_path);
    let mut model = Model::load(model_path, &layer_backends, true);

    let input_text = "5 JavaScript edge cases that broke prod\n";
    let input_tokens_ids: Vec<usize> = vocab
        .tokenize(input_text, true)
        .unwrap()
        .into_iter()
        .map(|(_, id)| id.try_into().unwrap())
        .collect();

    let mut last_token_id = input_tokens_ids.first().unwrap().clone();
    let mut print_buffer = TokenUtf8Buffer::new();
    for i_context in 0..model.n_context {
        let input_token_id = if i_context < input_tokens_ids.len() {
            input_tokens_ids[i_context].clone()
        } else {
            last_token_id
        };

        if let Some(s) = print_buffer.push(vocab.token(input_token_id)) {
            print!("{}", s);
            io::stdout().flush().unwrap();
        }

        let hidden_in = vocab_embeddings.get_embedding(input_token_id.try_into().unwrap());
        let final_out = model.predict(&hidden_in);
        let token_id = final_out
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        last_token_id = token_id;
    }
    println!();
}
