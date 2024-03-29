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

    pub fn root_mut(&mut self) -> &mut InferenceTreeNode {
        &mut self.0
    }

    pub fn get_nodes_on_path(&self, path: &[TokenId]) -> Vec<&InferenceTreeNode> {
        if path.is_empty() {
            return vec![];
        }
        let mut nodes = vec![];
        let mut node = self.root();
        assert!(path[0] == node.token_id);
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

    pub fn try_get_nodes_on_path(&self, path: &[TokenId]) -> Vec<&InferenceTreeNode> {
        let mut node = self.root();
        if node.token_id != path[0] {
            return vec![];
        }
        let mut nodes = vec![node];
        for &token_id in &path[1..] {
            node = match node.get_child(token_id) {
                Some(child) => child,
                None => break,
            };
            nodes.push(node);
        }
        assert!(nodes.len() <= path.len());
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
    pub probability: f64,
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
    pub interval_starts: Vec<f64>,
    pub interval_sizes: Vec<f64>,
    pub indices_by_interval_size: Vec<usize>, // largest interval size first
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum NodeSortGroup {
    NoSpaceLowercase,
    NoSpaceUppercase,
    SpaceLowercase,
    SpaceUppercase,
    Other,
}

impl NodeSortGroup {
    fn from_token(token: &[u8]) -> NodeSortGroup {
        if token.is_empty() {
            return NodeSortGroup::Other;
        }
        let first_char = token[0];
        if first_char == b' ' {
            if token.len() == 1 {
                NodeSortGroup::Other
            } else if token[1].is_ascii_lowercase() {
                NodeSortGroup::SpaceLowercase
            } else if token[1].is_ascii_uppercase() {
                NodeSortGroup::SpaceUppercase
            } else {
                NodeSortGroup::Other
            }
        } else {
            if first_char.is_ascii_lowercase() {
                return NodeSortGroup::NoSpaceLowercase;
            } else if first_char.is_ascii_uppercase() {
                return NodeSortGroup::NoSpaceUppercase;
            } else {
                NodeSortGroup::Other
            }
        }
    }
}

impl InferenceTreeChildren {
    pub fn from_nodes(nodes: Vec<InferenceTreeNode>) -> InferenceTreeChildren {
        let sorted_indices = {
            let mut indices: Vec<usize> = (0..nodes.len()).collect();
            indices.sort_by_key(|&i| {
                let token = &nodes[i].token;
                (NodeSortGroup::from_token(token), token)
            });
            indices
        };
        let inverse_sorted_indices: Vec<usize> = {
            let mut indices: Vec<usize> = (0..nodes.len()).collect();
            indices.sort_by_key(|&i| sorted_indices[i]);
            indices
        };

        let sorted_interval_sizes: Vec<f64> = sorted_indices
            .iter()
            .map(|&i| nodes[i].probability)
            .collect();
        let sorted_interval_ends: Vec<f64> = sorted_interval_sizes
            .iter()
            .scan(0.0, |end, &size| {
                *end += size;
                Some(*end)
            })
            .collect();
        let sorted_interval_starts: Vec<f64> = sorted_interval_ends
            .iter()
            .zip(sorted_interval_sizes.iter())
            .map(|(&end, &size)| end - size)
            .collect();

        let interval_starts: Vec<f64> = inverse_sorted_indices
            .iter()
            .map(|&i| sorted_interval_starts[i])
            .collect();
        let interval_sizes: Vec<f64> = inverse_sorted_indices
            .iter()
            .map(|&i| sorted_interval_sizes[i])
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
