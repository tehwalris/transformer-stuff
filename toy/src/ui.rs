use std::sync::{Arc, Mutex};

use llm_base::TokenId;
use nannou::prelude::*;

use crate::tree::InferenceTree;

struct Cursor {
    path: Vec<TokenId>,
    x: f32,
    y: f32,
}

impl Cursor {
    fn new(path: Vec<TokenId>) -> Self {
        assert!(path.len() > 0);
        Self {
            path,
            x: 0.0,
            y: 0.5,
        }
    }

    fn normalize(&mut self, inference_tree: &InferenceTree) {
        // Discover all the nodes from the root to the node at `path`.
        let mut nodes_from_root = vec![];
        {
            let mut node = inference_tree.root();
            nodes_from_root.push(node);
            for &token_id in &self.path[1..] {
                node = match node.get_child(token_id) {
                    Some(child) => child,
                    None => break,
                };
                nodes_from_root.push(node);
            }
        }
        assert!(nodes_from_root.len() == self.path.len());
        assert!(nodes_from_root.len() > 0);

        // Move outward until we are fully inside the node at `path` or at the root.
        while nodes_from_root.len() > 1 && !self.is_contained_in_path() {
            let node = nodes_from_root.pop().unwrap();
            self.path.pop().unwrap();
            let parent = *nodes_from_root.last().unwrap();
            let parent_children = parent.children.as_ref().unwrap();
            let node_probability_offset = parent_children
                .iter()
                .take_while(|child| child.token_id != node.token_id)
                .map(|child| child.probability)
                .sum::<f32>();
            self.y = node_probability_offset + self.y * node.probability;
            self.x = 1.0 - (1.0 - self.x) * node.probability;
        }
        assert!(nodes_from_root.len() == self.path.len());
        assert!(nodes_from_root.len() > 0);

        // Clamp to never be outside the path (will only happen if we are at the root).
        if !self.is_contained_in_path() {
            assert!(nodes_from_root.len() == 1);
            self.x = self.x.clamp(0.0, 1.0);
            self.y = self.y.clamp(0.0, 1.0);
        }
        assert!(self.is_contained_in_path());

        // Move inward as long as we are fully inside the node we're moving into.
        loop {
            let node = *nodes_from_root.last().unwrap();
            let children = match &node.children {
                Some(children) => children,
                None => break,
            };
            let child_end_probabilities = children
                .iter()
                .scan(0.0, |cumulative_probability, child| {
                    *cumulative_probability += child.probability;
                    Some(*cumulative_probability)
                })
                .collect::<Vec<_>>();

            let child_index = children
                .iter()
                .zip(child_end_probabilities.iter())
                .position(|(child, &end_probability)| {
                    let start_probability = end_probability - child.probability;
                    child.probability > 0.0
                        && self.y >= start_probability
                        && self.y <= end_probability
                });
            let child_index = match child_index {
                Some(candidate_child_index) => candidate_child_index,
                None => break,
            };

            let child = &children[child_index];
            let child_end_probability = child_end_probabilities[child_index];
            let child_start_probability = child_end_probability - child.probability;

            if self.x < 1.0 - child.probability {
                break;
            }

            nodes_from_root.push(child);
            self.path.push(child.token_id);
            self.y = ((self.y - child_start_probability) / child.probability).clamp(0.0, 1.0);
            self.x = ((self.x - (1.0 - child.probability)) / child.probability).clamp(0.0, 1.0);
        }
        assert!(nodes_from_root.len() == self.path.len());
        assert!(nodes_from_root.len() > 0);
        assert!(self.is_contained_in_path());
    }

    fn is_contained_in_path(&self) -> bool {
        self.x >= 0.0 && self.x <= 1.0 && self.y >= 0.0 && self.y <= 1.0
    }
}

pub struct UIModel {
    _window: window::Id,
    inference_tree: Arc<Mutex<InferenceTree>>,
    focused_path: Arc<Mutex<Vec<TokenId>>>,
    cursor: Cursor,
}

impl UIModel {
    pub fn new(
        app: &App,
        inference_tree: Arc<Mutex<InferenceTree>>,
        focused_path: Arc<Mutex<Vec<TokenId>>>,
    ) -> Self {
        let cursor = Cursor::new(focused_path.lock().unwrap().clone());
        Self {
            _window: app.new_window().view(view).build().unwrap(),
            inference_tree,
            focused_path,
            cursor,
        }
    }

    pub fn update(&mut self, _app: &App, _update: Update) {
        let inference_tree = self.inference_tree.lock().unwrap();
        self.cursor.normalize(&inference_tree);
        *self.focused_path.lock().unwrap() = self.cursor.path.clone();
    }
}

fn view(app: &App, _model: &UIModel, frame: Frame) {
    let draw = app.draw();
    draw.background().color(PLUM);
    draw.ellipse().color(STEELBLUE);
    draw.to_frame(app, &frame).unwrap();
}
