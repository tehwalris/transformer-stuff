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
    pub probability: f32,
    pub prediction_id: Option<u32>,
    pub children: Option<Vec<InferenceTreeNode>>,
}

impl InferenceTreeNode {
    pub fn get_child(&self, token_id: TokenId) -> Option<&InferenceTreeNode> {
        if let Some(children) = &self.children {
            let child = &children[usize::try_from(token_id).unwrap()];
            assert!(child.token_id == token_id);
            Some(child)
        } else {
            None
        }
    }

    pub fn get_child_mut(&mut self, token_id: TokenId) -> Option<&mut InferenceTreeNode> {
        if let Some(children) = &mut self.children {
            let child = &mut children[usize::try_from(token_id).unwrap()];
            assert!(child.token_id == token_id);
            Some(child)
        } else {
            None
        }
    }
}
