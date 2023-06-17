#![feature(iter_array_chunks)]

use std::{
    io::{BufRead, Seek},
    path::Path,
    thread,
};

use cpp_stuff_nice::SimpleTransformerLayer;
use ggml::format::TensorLoadInfo;
use half::f16;
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
    layers: Vec<SimpleTransformerLayer>,
    final_layer: SimpleTransformerLayer,
}

impl Model {
    fn load(path: impl AsRef<Path>, layer_backends: &[Backend], show_progress: bool) -> Self {
        let path = path.as_ref().to_str().unwrap().to_owned();

        let loader = cpp_stuff_nice::SimpleLlamaModelLoader::new(&path);
        let n_hidden = usize::try_from(loader.n_hidden()).unwrap();
        let n_layers = usize::try_from(loader.n_layers()).unwrap();
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
            layers,
            final_layer,
        }
    }
}

struct VocabEmbeddings {
    n_hidden: usize,
    n_vocab: usize,
    embeddings: Vec<f16>,
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

        let embeddings: Vec<f16> = info
            .read_data(reader)
            .unwrap()
            .into_iter()
            .array_chunks()
            .map(f16::from_le_bytes)
            .collect();
        assert_eq!(embeddings.len(), n_hidden * n_vocab);

        Self {
            n_hidden,
            n_vocab,
            embeddings,
        }
    }

    fn get_embedding(&self, i: usize) -> &[f16] {
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
    let model = Model::load(model_path, &layer_backends, true);

    let input_text = "This is a test";
    let input_tokens = vocab.tokenize(input_text, true).unwrap();

    println!("Testing token printing");
    let mut token_print_buffer = TokenUtf8Buffer::new();
    for &(token_bytes, _) in input_tokens.iter() {
        if let Some(valid_str) = token_print_buffer.push(&token_bytes) {
            print!("{}", valid_str);
        }
    }
    println!();
}
