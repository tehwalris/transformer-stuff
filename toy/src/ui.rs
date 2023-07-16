use std::sync::{Arc, Mutex};

use llm_base::{TokenId, TokenUtf8Buffer};
use nannou::{event::ElementState, prelude::*};
use rayon::string;

use crate::tree::{InferenceTree, InferenceTreeNode};

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
        let mut nodes_from_root = inference_tree.get_nodes_on_path(&self.path);
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
    moving: bool,
    speed: f32,
    steering: Vec2,
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
            moving: false,
            speed: 3.0,
            steering: Vec2::default(),
        }
    }

    pub fn update(&mut self, app: &App, update: Update) {
        let inference_tree = self.inference_tree.lock().unwrap();

        let window_rect = app.window_rect();
        let right_half_width = window_rect.w() / 3.0;

        self.steering = (app.mouse.position()).clamp_length_max(right_half_width);
        let speed = self.steering / right_half_width
            * (1.0 - self.cursor.x)
            * self.speed
            * update.since_last.secs() as f32;
        if self.moving {
            self.cursor.x += speed.x;
            self.cursor.y -= speed.y;
        }
        // HACK prevent zooming in extremely before entering a child
        if self.cursor.x > 0.9999 {
            self.cursor.x = 0.9999;
        }

        self.cursor.normalize(&inference_tree);
        *self.focused_path.lock().unwrap() = self.cursor.path.clone();
    }

    pub fn event(&mut self, _app: &App, event: Event) {
        match event {
            Event::WindowEvent {
                id: _,
                simple: Some(WindowEvent::MousePressed(MouseButton::Left)),
            } => {
                self.moving = !self.moving;
            }
            _ => {}
        }
    }
}

struct DisplayInterval {
    start: f32,
    end: f32,
    depth: usize,
}

fn intervals_from_tree(
    inference_tree: &InferenceTree,
    cursor: &Cursor,
    window_height: f32,
    right_half_width: f32,
) -> Vec<DisplayInterval> {
    let nodes_from_root = inference_tree.get_nodes_on_path(&cursor.path);
    assert!(nodes_from_root.len() > 0);
    let size = right_half_width / (1.0 - cursor.x);
    let mut intervals = Vec::new();
    intervals_from_node(
        nodes_from_root.last().unwrap(),
        window_height,
        size,
        -cursor.y * size,
        cursor.path.len(),
        &mut intervals,
        -0.5 * window_height - 1.0,
        0.5 * window_height + 1.0,
    );
    intervals
}

fn intervals_from_node(
    node: &InferenceTreeNode,
    window_height: f32,
    size: f32,
    start: f32,
    depth: usize,
    output: &mut Vec<DisplayInterval>,
    output_start: f32,
    output_end: f32,
) {
    let end = start + size;
    output.push(DisplayInterval { start, end, depth });

    if size < 5.0 {
        return;
    }

    if let Some(children) = node.children.as_ref() {
        let mut child_start = start;
        for child in children {
            let remaining_space = size - child_start;
            if remaining_space < 1.0 {
                break;
            }
            let child_size = child.probability * size;
            let child_end = child_start + child_size;
            if child_size >= 1.0 && child_end >= output_start && child_start <= output_end {
                intervals_from_node(
                    child,
                    window_height,
                    child_size,
                    child_start,
                    depth + 1,
                    output,
                    output_start,
                    output_end,
                );
            }
            child_start = child_end;
        }
    }
}

fn get_text_at_path(inference_tree: &InferenceTree, path: &[TokenId]) -> String {
    let mut string_parts = vec![];
    let mut print_buffer = TokenUtf8Buffer::new();
    for node in inference_tree.get_nodes_on_path(path) {
        if let Some(string_part) = print_buffer.push(&node.token) {
            string_parts.push(string_part);
        }
    }
    string_parts.join("")
}

fn view(app: &App, model: &UIModel, frame: Frame) {
    let draw = app.draw();
    let window_rect = app.window_rect();
    let right_half_width = window_rect.w() / 3.0;

    draw.background().color(BLACK);

    let inference_tree = model.inference_tree.lock().unwrap();

    let intervals = intervals_from_tree(
        &inference_tree,
        &model.cursor,
        window_rect.h(),
        right_half_width,
    );

    let cursor_text = get_text_at_path(&inference_tree, &model.cursor.path);

    drop(inference_tree);

    for DisplayInterval { start, end, depth } in intervals {
        let size = end - start;
        let rect = Rect::from_x_y_w_h(
            right_half_width - 0.5 * size,
            -0.5 * (start + end),
            size,
            size,
        );
        if let Some(rect) = rect.overlap(window_rect) {
            draw.rect()
                .xy(rect.xy())
                .wh(rect.wh())
                .color(if depth % 2 == 0 { GREEN } else { BLUE });
        }
    }

    // Horizontal and vertical guide lines
    draw.line()
        .start(pt2(-0.5 * window_rect.w(), 0.0))
        .end(pt2(0.5 * window_rect.w(), 0.0))
        .color(WHITE);
    draw.line()
        .start(pt2(0.0, -0.5 * window_rect.h()))
        .end(pt2(0.0, 0.5 * window_rect.h()))
        .color(WHITE);

    // Steering indicator
    draw.line().end(model.steering).color(WHITE);
    draw.ellipse()
        .xy(model.steering)
        .radius(5.0)
        .color(if model.moving { WHITE } else { BLACK });

    let text_rect = Rect::from_wh(0.5 * window_rect.wh()).top_left_of(window_rect);
    draw.text(&cursor_text)
        .xy(text_rect.xy())
        .wh(text_rect.wh())
        .left_justify()
        .align_text_bottom()
        .color(WHITE);

    draw.to_frame(app, &frame).unwrap();
}
