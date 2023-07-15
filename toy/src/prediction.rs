use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
    time::Duration,
};

use crate::{
    model::{Backend, Model},
    tree::{InferenceTree, InferenceTreeNode},
    vocab::{load_vocab, VocabEmbeddings},
};
use llm_base::TokenId;

fn get_prediction_path<'a>(
    inference_tree: &'a InferenceTree,
    focused_path: &[TokenId],
) -> Option<(Vec<u32>, &'a InferenceTreeNode)> {
    if focused_path.is_empty() {
        return None;
    }
    if focused_path[0] != inference_tree.root().token_id {
        eprint!("WARNING: focused path does not match inference tree root");
        return None;
    }

    let mut prediction_path = vec![];
    let mut node = inference_tree.root();
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

pub fn prediction_thread_main(
    model_path: PathBuf,
    inference_tree: Arc<Mutex<InferenceTree>>,
    focused_path: Arc<Mutex<Vec<TokenId>>>,
) {
    let mut layer_backends = vec![];
    for _ in 0..16 {
        layer_backends.push(Backend::Cuda);
    }
    for _ in 0..16 {
        layer_backends.push(Backend::Hip);
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
