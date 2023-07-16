use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
    time::Duration,
};

use crate::{
    model::{Backend, Model},
    vocab::{load_vocab, VocabEmbeddings},
};
use llm_base::TokenId;

pub struct InferenceTree(InferenceTreeNode);

impl InferenceTree {
    pub fn new(root_token_id: TokenId) -> Self {
        Self(InferenceTreeNode {
            token_id: root_token_id,
            token: vec![],
            probability: 1.0,
            prediction_id: None,
            children: None,
        })
    }

    pub fn root(&self) -> &InferenceTreeNode {
        &self.0
    }

    pub fn get_nodes_on_path(&self, path: &[TokenId]) -> Vec<&InferenceTreeNode> {
        if path.is_empty() {
            return vec![];
        }
        let mut nodes = vec![];
        let mut node = self.root();
        nodes.push(node);
        for &token_id in &path[1..] {
            node = match node.get_child(token_id) {
                Some(child) => child,
                None => break,
            };
            nodes.push(node);
        }
        assert!(nodes.len() == path.len());
        nodes
    }

    pub fn get_node_mut(&mut self, path: &[TokenId]) -> &mut InferenceTreeNode {
        let mut node = &mut self.0;
        assert!(path[0] == node.token_id);
        for &token_id in &path[1..] {
            node = node.get_child_mut(token_id).unwrap();
        }
        node
    }
}

#[derive(Clone)]
pub struct InferenceTreeNode {
    pub token_id: TokenId,
    pub token: Vec<u8>,
    pub probability: f32,
    pub prediction_id: Option<u32>,
    pub children: Option<InferenceTreeChildren>,
}

impl InferenceTreeNode {
    pub fn get_child(&self, token_id: TokenId) -> Option<&InferenceTreeNode> {
        if let Some(children) = &self.children {
            let child = &children.nodes[usize::try_from(token_id).unwrap()];
            assert!(child.token_id == token_id);
            Some(child)
        } else {
            None
        }
    }

    pub fn get_child_mut(&mut self, token_id: TokenId) -> Option<&mut InferenceTreeNode> {
        if let Some(children) = &mut self.children {
            let child = &mut children.nodes[usize::try_from(token_id).unwrap()];
            assert!(child.token_id == token_id);
            Some(child)
        } else {
            None
        }
    }
}

#[derive(Clone)]
pub struct InferenceTreeChildren {
    pub nodes: Vec<InferenceTreeNode>,
    pub interval_starts: Vec<f32>,
    pub interval_sizes: Vec<f32>,
    pub indices_by_interval_size: Vec<usize>, // largest interval size first
}

impl InferenceTreeChildren {
    pub fn from_nodes(nodes: Vec<InferenceTreeNode>) -> InferenceTreeChildren {
        let interval_sizes: Vec<f32> = nodes.iter().map(|node| node.probability).collect();
        let interval_ends: Vec<f32> = interval_sizes
            .iter()
            .scan(0.0, |end, &size| {
                *end += size;
                Some(*end)
            })
            .collect();
        let interval_starts = interval_ends
            .iter()
            .zip(interval_sizes.iter())
            .map(|(&end, &size)| end - size)
            .collect();
        let indices_by_interval_size = {
            let mut indices: Vec<usize> = (0..nodes.len()).collect();
            indices.sort_by(|&i, &j| interval_sizes[j].partial_cmp(&interval_sizes[i]).unwrap());
            indices
        };
        InferenceTreeChildren {
            nodes,
            interval_starts,
            interval_sizes,
            indices_by_interval_size,
        }
    }
}
