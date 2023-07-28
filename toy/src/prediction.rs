use std::{
    arch::x86_64::_MM_FROUND_FLOOR,
    collections::BinaryHeap,
    io::Write,
    path::PathBuf,
    sync::{Arc, Mutex},
};

use crate::{
    model::{Backend, Model},
    tree::{InferenceTree, InferenceTreeChildren, InferenceTreeNode},
    vocab::{load_vocab, VocabEmbeddings},
};

use llm_base::{TokenId, TokenUtf8Buffer, Vocabulary};
use nannou::winit::platform::unix::x11::ffi::ClipByChildren;
use regex_automata::{
    dfa::{regex::Regex, Automaton},
    Input,
};

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

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum PotentialMatch {
    DidMatch,
    CanMatch,
    CanNotMatch,
}

fn potentially_matches(pattern: &Regex, input: &str) -> PotentialMatch {
    let input_bytes = input.as_bytes();
    let dfa = pattern.forward();
    let mut state = dfa.start_state_forward(&Input::new(input_bytes)).unwrap();
    for &byte in input_bytes {
        state = dfa.next_state(state, byte);
        if dfa.is_dead_state(state) {
            return PotentialMatch::CanNotMatch;
        }
    }
    state = dfa.next_eoi_state(state);
    if dfa.is_match_state(state) {
        PotentialMatch::DidMatch
    } else {
        PotentialMatch::CanMatch
    }
}

#[derive(PartialEq, Clone)]
struct ExplorationItem {
    log_probability: f64,
    path: Vec<TokenId>,
    text: String,
    text_tail: Option<TokenUtf8Buffer>,
    approx_words: usize,
    good_tokens: usize,
}

impl ExplorationItem {
    fn sort_key(&self) -> f64 {
        self.log_probability + 5.0 * (self.good_tokens as f64)
    }
}

impl PartialOrd for ExplorationItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.sort_key().partial_cmp(&other.sort_key())
    }
}

impl Eq for ExplorationItem {}

impl Ord for ExplorationItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
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

    let prefix = " Philippe (12:23 PM):\nWhat were they all laughing about?\nKris (12:25 PM):\n";
    assert!(prefix.starts_with(" "));
    let mut explore_regex = r"^".to_string();
    explore_regex.push_str(&regex::escape(prefix));
    for (i, letter) in "cetat".chars().enumerate() {
        if i > 0 {
            explore_regex.push_str(r"[.,;!?\- ()] ?");
        }
        explore_regex.push_str(&format!(r"[{}{}]", letter, letter.to_uppercase()));
        explore_regex.push_str(r"[a-z']*");
    }
    explore_regex.push_str(r"[.!?]$");
    println!("Explore regex: {}", explore_regex);
    let explore_regex = Regex::new(&explore_regex).unwrap();
    assert!(potentially_matches(&explore_regex, prefix) == PotentialMatch::CanMatch);

    println!("Loading model...");
    let (vocab, vocab_embeddings) = load_vocab(&model_path);
    let mut model = Model::load(&model_path, &layer_backends, true);
    println!("Done loading model");

    let tokenized_prefix: Vec<TokenId> = vocab
        .tokenize(&prefix, true)
        .unwrap()
        .into_iter()
        .map(|(_, token_id)| token_id)
        .collect();

    let mut priority_queue = BinaryHeap::new();
    let bos_token_id = 1;
    priority_queue.push(ExplorationItem {
        log_probability: 0.0,
        path: vec![bos_token_id],
        text: "".to_string(),
        text_tail: None,
        approx_words: 0,
        good_tokens: 0,
    });
    let mut found_any_matches = false;
    while let Some(item) = priority_queue.pop() {
        if explore_regex.is_match(&item.text) {
            found_any_matches = true;
            println!("");
            println!(
                "Found match ({}): {}",
                item.log_probability,
                item.text.strip_prefix(prefix).unwrap().trim()
            );
            continue;
        }

        let mut inference_tree = inference_tree.lock().unwrap();

        if model.next_i() as f32 / model.n_cache as f32 > 0.9 {
            println!("Clearing cache to free space");
            model.retain(&[]);
            *inference_tree = InferenceTree::new(bos_token_id);
        }

        if found_any_matches {
            print!(".");
            std::io::stdout().flush().unwrap();
        } else {
            println!(
                "Sort key: {:?}, log probability: {}, depth: {}, has tail: {}\n{}",
                item.sort_key(),
                item.log_probability,
                item.path.len(),
                item.text_tail.is_some(),
                item.text.lines().last().unwrap_or_default()
            );
        }

        let mut node = inference_tree.root_mut();
        assert_eq!(item.path[0], node.token_id);
        let mut prediction_path = vec![];
        for &next_token_id in &item.path[1..] {
            match node.prediction_id {
                Some(prediction_id) => {
                    prediction_path.push(prediction_id);
                }
                None => {
                    let prediction_id = model.next_i();
                    node.prediction_id = Some(prediction_id);
                    prediction_path.push(prediction_id);
                    let hidden_in =
                        vocab_embeddings.get_embedding(node.token_id.try_into().unwrap());
                    let logits = model.predict(&hidden_in, &prediction_path);
                    node.children = Some(children_from_logits(&logits, &vocab));
                }
            }
            node = &mut node.children.as_mut().unwrap().nodes[next_token_id as usize];
        }

        {
            assert!(node.prediction_id.is_none());
            let prediction_id = model.next_i();
            node.prediction_id = Some(prediction_id);
            prediction_path.push(prediction_id);
            let hidden_in = vocab_embeddings.get_embedding(node.token_id.try_into().unwrap());
            let logits = model.predict(&hidden_in, &prediction_path);
            node.children = Some(children_from_logits(&logits, &vocab));
            let children = node.children.as_ref().unwrap();

            let high_probability =
                children.nodes[children.indices_by_interval_size[10]].probability;

            for child in &node.children.as_ref().unwrap().nodes {
                let mut child_item = item.clone();
                child_item.log_probability += child.probability.ln() as f64;
                child_item.path.push(child.token_id);
                if child.probability >= high_probability && child.token.starts_with(" ".as_bytes())
                {
                    child_item.good_tokens += 1;
                }
                let mut tail_buffer = child_item.text_tail.take().unwrap_or_default();
                if let Some(tail) = tail_buffer.push(&child.token) {
                    child_item.text += &tail;
                }
                if tail_buffer != TokenUtf8Buffer::default() {
                    // HACK We don't want to explore paths with tokens that are not valid UTF-8 by themselves, because they usually are not interesting.
                    continue;
                }
                child_item.approx_words = child_item.text.split_whitespace().count();
                if child_item
                    .path
                    .iter()
                    .zip(&tokenized_prefix)
                    .any(|(&a, &b)| a != b)
                {
                    continue;
                }
                if potentially_matches(&explore_regex, &child_item.text)
                    == PotentialMatch::CanNotMatch
                {
                    continue;
                }
                priority_queue.push(child_item);
            }
        }
    }
}
