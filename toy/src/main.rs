#![feature(iter_array_chunks)]

use std::io::{BufRead, Seek};

use ggml::format::TensorLoadInfo;
use half::f16;
use llm_base::Vocabulary;
use rand::Rng;
use rand_distr::StandardNormal;

fn do_thing(model_path: &str) {
    let mut rng = rand::thread_rng();

    let mut loader = cpp_stuff_nice::SimpleLlamaModelLoader::new(model_path);
    let n_hidden = usize::try_from(loader.n_hidden()).unwrap();

    let mut baseline_model = cpp_stuff_nice::baseline::create_llama_layer(&mut loader, 0);
    let mut cuda_model = cpp_stuff_nice::cuda::create_llama_layer(&mut loader, 0);
    let mut hip_model = cpp_stuff_nice::hip::create_llama_layer(&mut loader, 0);
    drop(loader);

    let hidden_in: Vec<f32> = (0..n_hidden)
        .map(|_| rng.sample(StandardNormal))
        .collect::<Vec<_>>();
    let mut hidden_out: Vec<f32> = vec![0.0; n_hidden];

    baseline_model.forward(&hidden_in, &mut hidden_out);
    let hidden_out_baseline = hidden_out.clone();

    cuda_model.forward(&hidden_in, &mut hidden_out);
    let hidden_out_cuda = hidden_out.clone();

    hip_model.forward(&hidden_in, &mut hidden_out);
    let hidden_out_hip = hidden_out.clone();

    for i_hidden in [0, 1, n_hidden - 2, n_hidden - 1].into_iter() {
        println!(
            "hidden_out_baseline[{}] = {}",
            i_hidden, hidden_out_baseline[i_hidden]
        );
        println!(
            "hidden_out_cuda[{}] = {}",
            i_hidden, hidden_out_cuda[i_hidden]
        );
        println!(
            "hidden_out_hip[{}] = {}",
            i_hidden, hidden_out_hip[i_hidden]
        );
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

    println!("Loading vocab");
    let (vocab, vocab_embeddings) = load_vocab(model_path);
    println!("Done loading vocab");
    let token_id = 123;
    println!(
        "Embedding of token \"{}\" ({}) starts with {:?}",
        String::from_utf8(vocab.token(token_id).to_vec()).unwrap(),
        token_id,
        &vocab_embeddings.get_embedding(token_id)[..10]
    );

    println!("Doing thing");
    do_thing(model_path);
    println!("Done doing thing")
}
