#![feature(iter_array_chunks)]

mod model;
mod ui;
mod vocab;

use std::{
    fs::File,
    io::BufWriter,
    path::PathBuf,
    sync::{Arc, Mutex},
    time::Duration,
};

use llm_base::TokenId;
use tracing_subscriber::prelude::*;
use vocab::VocabEmbeddings;

use crate::{model::Model, vocab::load_vocab};

struct InferenceTree(InferenceTreeNode);

impl InferenceTree {
    fn new(root_token_id: TokenId) -> Self {
        Self(InferenceTreeNode {
            token_id: root_token_id,
            probability: 1.0,
            prediction_id: None,
            children: None,
        })
    }

    fn get_node_mut(&mut self, path: &[TokenId]) -> &mut InferenceTreeNode {
        let mut node = &mut self.0;
        assert!(path[0] == node.token_id);
        for &token_id in &path[1..] {
            node = node.get_child_mut(token_id).unwrap();
        }
        node
    }
}

#[derive(Clone)]
struct InferenceTreeNode {
    token_id: TokenId,
    probability: f32,
    prediction_id: Option<u32>,
    children: Option<Vec<InferenceTreeNode>>,
}

impl InferenceTreeNode {
    fn get_child(&self, token_id: TokenId) -> Option<&InferenceTreeNode> {
        if let Some(children) = &self.children {
            let child = &children[usize::try_from(token_id).unwrap()];
            assert!(child.token_id == token_id);
            Some(child)
        } else {
            None
        }
    }

    fn get_child_mut(&mut self, token_id: TokenId) -> Option<&mut InferenceTreeNode> {
        if let Some(children) = &mut self.children {
            let child = &mut children[usize::try_from(token_id).unwrap()];
            assert!(child.token_id == token_id);
            Some(child)
        } else {
            None
        }
    }
}

fn get_prediction_path<'a>(
    inference_tree: &'a InferenceTree,
    focused_path: &[TokenId],
) -> Option<(Vec<u32>, &'a InferenceTreeNode)> {
    if focused_path.is_empty() {
        return None;
    }
    if focused_path[0] != inference_tree.0.token_id {
        eprint!("WARNING: focused path does not match inference tree root");
        return None;
    }

    let mut prediction_path = vec![];
    let mut node = &inference_tree.0;
    for &token_id in &focused_path[1..] {
        prediction_path.push(node.prediction_id?);
        node = node.get_child(token_id)?;
    }
    Some((prediction_path, node))
}

fn try_predict_next(
    model: &mut Model,
    vocab_embeddings: &VocabEmbeddings,
    inference_tree: &Arc<Mutex<InferenceTree>>,
    focused_path: &[TokenId],
) -> bool {
    let (mut prediction_path, final_node_clone) = {
        let inference_tree = inference_tree.lock().unwrap();
        match get_prediction_path(&inference_tree, &focused_path) {
            Some((prediction_path, final_node)) => (prediction_path, final_node.clone()),
            None => return false,
        }
    };

    if final_node_clone.children.is_some() {
        return false;
    }
    let prediction_id = model.next_i();
    prediction_path.push(prediction_id);

    let hidden_in = vocab_embeddings.get_embedding(final_node_clone.token_id.try_into().unwrap());
    let final_out = model.predict(&hidden_in, &prediction_path);

    let children = final_out
        .iter()
        .enumerate()
        .map(|(i, &probability)| InferenceTreeNode {
            token_id: TokenId::try_from(i).unwrap(),
            probability,
            prediction_id: None,
            children: None,
        })
        .collect();

    {
        let mut inference_tree = inference_tree.lock().unwrap();
        let final_node = inference_tree.get_node_mut(focused_path);

        if final_node.prediction_id.is_some() || final_node.children.is_some() {
            eprintln!("WARNING: duplicate prediction");
            return false;
        }
        final_node.prediction_id = Some(prediction_id);
        final_node.children = Some(children);
    }

    true
}

fn prediction_thread_main(
    model_path: PathBuf,
    inference_tree: Arc<Mutex<InferenceTree>>,
    focused_path: Arc<Mutex<Vec<TokenId>>>,
) {
    let mut layer_backends = vec![];
    for _ in 0..16 {
        layer_backends.push(model::Backend::Cuda);
    }
    for _ in 0..16 {
        layer_backends.push(model::Backend::Hip);
    }

    println!("Loading model...");
    let (vocab, vocab_embeddings) = load_vocab(&model_path);
    let mut model = Model::load(&model_path, &layer_backends, true);
    println!("Done loading model");

    loop {
        let focused_path = focused_path.lock().unwrap().clone();
        let did_predict = try_predict_next(
            &mut model,
            &vocab_embeddings,
            &inference_tree,
            &focused_path,
        );
        if did_predict {
            println!("Predicted path {:?}", focused_path);
        } else {
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        println!("Usage: toy <model_path>");
        return;
    }
    let model_path = PathBuf::from(&args[1]);

    let trace_writer = BufWriter::new(File::create("trace.json").unwrap());
    let (chrome_layer, guard) = tracing_chrome::ChromeLayerBuilder::new()
        .writer(trace_writer)
        .include_args(true)
        .build();
    tracing_subscriber::registry().with(chrome_layer).init();

    let bos_token_id = 1;
    let inference_tree = Arc::new(Mutex::new(InferenceTree::new(bos_token_id)));
    let focused_path = Arc::new(Mutex::new(vec![bos_token_id]));

    {
        let inference_tree = inference_tree.clone();
        let focused_path = focused_path.clone();
        std::thread::spawn(move || {
            prediction_thread_main(model_path, inference_tree, focused_path);
        });
    }

    nannou::app(ui::model).update(ui::update).run();

    guard.flush();

    // let input_text = "5 JavaScript edge cases that broke prod\n";
    // let input_tokens_ids: Vec<usize> = vocab
    //     .tokenize(input_text, true)
    //     .unwrap()
    //     .into_iter()
    //     .map(|(_, id)| id.try_into().unwrap())
    //     .collect();

    // let mut last_token_id = input_tokens_ids.first().unwrap().clone();
    // let mut print_buffer = TokenUtf8Buffer::new();
    // for i_context in 0..model.n_context {
    //     let span = span!(tracing::Level::INFO, "token processing", i_context);
    //     let _enter = span.enter();

    //     if i_context % 10 == 0 {
    //         guard.flush();
    //     }

    //     let input_token_id = if i_context < input_tokens_ids.len() {
    //         input_tokens_ids[i_context].clone()
    //     } else {
    //         last_token_id
    //     };

    //     if let Some(s) = print_buffer.push(vocab.token(input_token_id)) {
    //         print!("{}", s);
    //         io::stdout().flush().unwrap();
    //     }

    //     let hidden_in = vocab_embeddings.get_embedding(input_token_id.try_into().unwrap());
    //     prediction_path.push(model.next_i());
    //     let final_out = model.predict(&hidden_in, &prediction_path);
    //     let token_id = final_out
    //         .iter()
    //         .enumerate()
    //         .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
    //         .unwrap()
    //         .0;
    //     last_token_id = token_id;
    // }
    // println!();

    // guard.flush();
}

#[cfg(test)]
mod tests {
    use rand::Rng;

    const MODEL_PATH: &str = "/home/philippe/Documents/llama/7B/ggml-model-f16.bin"; // HACK

    #[test]
    fn test_cuda_final_layer() {
        let mut loader = cpp_stuff_nice::SimpleLlamaModelLoader::new(MODEL_PATH);
        let n_hidden = usize::try_from(loader.n_hidden()).unwrap();
        let n_vocab = usize::try_from(loader.n_vocab()).unwrap();

        let mut baseline_layer = cpp_stuff_nice::baseline::create_llama_final_layer(&mut loader);
        let mut cuda_layer = cpp_stuff_nice::cuda::create_llama_final_layer(&mut loader);

        let mut rng = rand::thread_rng();

        let mut hidden_in = (0..n_hidden).map(|_| rng.gen()).collect::<Vec<_>>();
        let mut final_out_baseline = vec![0.0; n_vocab];
        let mut final_out_cuda = vec![0.0; n_vocab];

        baseline_layer.forward(
            &hidden_in,
            &mut final_out_baseline,
            &[baseline_layer.next_i()],
        );
        cuda_layer.forward(&hidden_in, &mut final_out_cuda, &[cuda_layer.next_i()]);

        let tolerance = 0.1;
        let mut all_close_enough = true;
        for (i, (a, b)) in final_out_baseline
            .iter()
            .zip(final_out_cuda.iter())
            .enumerate()
        {
            let close_enough = (a - b).abs() < tolerance;
            if !close_enough {
                all_close_enough = false;
                println!("{}: {} != {} (tolerance {})", i, a, b, tolerance)
            }
        }
        assert!(all_close_enough);
    }
}
