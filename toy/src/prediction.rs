use std::{
    io::Write,
    path::PathBuf,
    sync::{Arc, Mutex},
    time::Duration,
};

use crate::{
    model::{Backend, Model},
    tree::{InferenceTree, InferenceTreeChildren, InferenceTreeNode},
    vocab::{load_vocab, VocabEmbeddings},
};
use llm_base::{TokenId, TokenUtf8Buffer, Vocabulary};

fn get_prediction_path<'a>(
    inference_tree: &'a InferenceTree,
    focused_path: &[TokenId],
) -> Option<(Vec<u32>, Vec<TokenId>, &'a InferenceTreeNode)> {
    let mut nodes_on_path = inference_tree.try_get_nodes_on_path(&focused_path);

    if nodes_on_path.is_empty() {
        return None;
    }

    if let Some(first_index_without_prediction_id) = nodes_on_path
        .iter()
        .position(|node| node.prediction_id.is_none())
    {
        nodes_on_path.truncate(first_index_without_prediction_id + 1);
        let prediction_path: Vec<u32> = nodes_on_path
            .iter()
            .filter_map(|node| node.prediction_id)
            .collect();
        assert!(prediction_path.len() == first_index_without_prediction_id);
        let target_path = nodes_on_path.iter().map(|node| node.token_id).collect();
        return Some((prediction_path, target_path, nodes_on_path.last().unwrap()));
    }

    if let Some(children) = &nodes_on_path.last().unwrap().children {
        for &child_index in &children.indices_by_interval_size[..10] {
            let child = &children.nodes[child_index];
            if child.prediction_id.is_none() {
                let prediction_path = nodes_on_path
                    .iter()
                    .map(|node| node.prediction_id.unwrap())
                    .collect();
                let mut target_path: Vec<TokenId> =
                    nodes_on_path.iter().map(|node| node.token_id).collect();
                target_path.push(child.token_id);
                return Some((prediction_path, target_path, child));
            }
        }
    }

    None
}

fn children_from_logits(logits: &[f32], vocab: &Vocabulary) -> InferenceTreeChildren {
    let mut probabilities: Vec<f64> = logits
        .iter()
        .map(|&log_probability| (log_probability as f64).exp())
        .collect();
    let sum_of_probabilities: f64 = probabilities.iter().sum();
    for probability in probabilities.iter_mut() {
        *probability /= sum_of_probabilities;
    }

    InferenceTreeChildren::from_nodes(
        probabilities
            .into_iter()
            .enumerate()
            .map(|(i, probability)| InferenceTreeNode {
                token_id: TokenId::try_from(i).unwrap(),
                token: vocab.token(i).to_vec(),
                probability,
                prediction_id: None,
                children: None,
            })
            .collect(),
    )
}

fn try_predict_next(
    model: &mut Model,
    vocab: &Vocabulary,
    vocab_embeddings: &VocabEmbeddings,
    inference_tree: &Arc<Mutex<InferenceTree>>,
    focused_path: &[TokenId],
) -> bool {
    let (mut prediction_path, target_path, target_node_clone) = {
        let inference_tree = inference_tree.lock().unwrap();
        match get_prediction_path(&inference_tree, &focused_path) {
            Some((prediction_path, target_path, target_node)) => {
                (prediction_path, target_path, target_node.clone())
            }
            None => return false,
        }
    };

    let prediction_id = model.next_i();
    prediction_path.push(prediction_id);

    let hidden_in = vocab_embeddings.get_embedding(target_node_clone.token_id.try_into().unwrap());
    let logits = model.predict(&hidden_in, &prediction_path);
    let children = children_from_logits(&logits, vocab);

    {
        let mut inference_tree = inference_tree.lock().unwrap();
        let final_node = inference_tree.get_node_mut(&target_path);

        if final_node.prediction_id.is_some() || final_node.children.is_some() {
            eprintln!("WARNING: duplicate prediction");
            return false;
        }
        final_node.prediction_id = Some(prediction_id);
        final_node.children = Some(children);
    }

    true
}

fn map_prediction_ids(
    node: &mut InferenceTreeNode,
    new_ids_by_old_id: &[Option<u32>],
    all_parents_have_prediction_ids: bool,
) {
    if let Some(prediction_id) = node.prediction_id {
        node.prediction_id = new_ids_by_old_id[usize::try_from(prediction_id).unwrap()];

        // Invariant: Nodes can only have a prediction_id if all of their ancestors have one.
        assert!(node.prediction_id.is_none() || all_parents_have_prediction_ids);

        if let Some(children) = &mut node.children {
            for child in &mut children.nodes {
                map_prediction_ids(child, new_ids_by_old_id, node.prediction_id.is_some());
            }
        }
    }
}

fn clear_most_cache(
    model: &mut Model,
    inference_tree: &mut InferenceTree,
    focused_path: &[TokenId],
) {
    let mut should_retain = vec![false; model.n_cache];

    let focused_nodes = inference_tree.try_get_nodes_on_path(focused_path);
    for &node in &focused_nodes {
        if let Some(prediction_id) = node.prediction_id {
            should_retain[usize::try_from(prediction_id).unwrap()] = true;
        }
    }

    if focused_nodes.len() == focused_path.len() {
        if let Some(children) = &focused_nodes.last().unwrap().children {
            for &child_index in &children.indices_by_interval_size[..10] {
                let child = &children.nodes[child_index];
                if let Some(prediction_id) = child.prediction_id {
                    should_retain[usize::try_from(prediction_id).unwrap()] = true;
                }
            }
        }
    }

    model.retain(
        &should_retain
            .iter()
            .enumerate()
            .filter(|(_, should_retain)| **should_retain)
            .map(|(old_id, _)| old_id)
            .collect::<Vec<_>>(),
    );

    let mut new_ids_by_old_id = vec![None; model.n_cache];
    let mut next_id: u32 = 0;
    for i in 0..model.n_cache {
        if should_retain[i] {
            new_ids_by_old_id[i] = Some(next_id);
            next_id += 1;
        }
    }

    map_prediction_ids(inference_tree.root_mut(), &new_ids_by_old_id, true);
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

    {
        let input_str =
            " After locking up for the night, Jane and I went to the pub. It was a rustic place, but charming. We had a few drinks and caught up with a few friends who stopped by.\nAfter a few drinks.";
        let input_encoding: Vec<_> = vocab
            .tokenize(input_str, true)
            .unwrap()
            .into_iter()
            .map(|(_, id)| id)
            .collect();

        let mut prediction_path = vec![];
        let mut next_token_id = input_encoding[0];
        let mut token_buffer = TokenUtf8Buffer::new();
        for i in 1..model.n_cache {
            if let Some(s) = token_buffer.push(vocab.token(next_token_id.try_into().unwrap())) {
                print!("{}", s);
                std::io::stdout().flush().unwrap();
            }

            prediction_path.push(model.next_i());
            let hidden_in = vocab_embeddings.get_embedding(next_token_id.try_into().unwrap());
            let logits = model.predict(hidden_in, &prediction_path);

            let argmax_logits = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0;

            if i < input_encoding.len() {
                next_token_id = input_encoding[i].try_into().unwrap();
            } else {
                next_token_id = argmax_logits.try_into().unwrap();
            }
        }
    }

    loop {
        let focused_path = focused_path.lock().unwrap().clone();

        if model.next_i() as f32 / model.n_cache as f32 > 0.9 {
            println!("Clearing most cache to free space");
            clear_most_cache(
                &mut model,
                &mut inference_tree.lock().unwrap(),
                &focused_path,
            );
        }

        let did_predict = try_predict_next(
            &mut model,
            &vocab,
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
