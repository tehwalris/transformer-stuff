use std::{
    io::{BufRead, Seek},
    path::Path,
};

use ggml::format::TensorLoadInfo;
use half::{f16, slice::HalfFloatSliceExt};
use llm_base::Vocabulary;

pub struct VocabEmbeddings {
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

    pub fn get_embedding(&self, i: usize) -> &[f32] {
        assert!(i < self.n_vocab);
        let offset = i * self.n_hidden;
        &self.embeddings[offset..offset + self.n_hidden]
    }
}

pub fn load_vocab(model_path: impl AsRef<Path>) -> (Vocabulary, VocabEmbeddings) {
    let mut loader = llm_base::Loader::<llm_llama::Hyperparameters, _>::new(|_| {});
    let mut file = std::io::BufReader::new(std::fs::File::open(&model_path).unwrap());
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
