use std::sync::{Arc, Mutex};

use llm_base::{TokenId, TokenUtf8Buffer};
use nannou::prelude::*;

use crate::tree::{InferenceTree, InferenceTreeNode};

struct Cursor {
    path: Vec<TokenId>,
    x: f64,
    y: f64,
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
            let node_index: usize = node.token_id.try_into().unwrap();
            let node_interval_start = parent_children.interval_starts[node_index];
            let node_interval_size = parent_children.interval_sizes[node_index];
            self.y = node_interval_start + self.y * node_interval_size;
            self.x = 1.0 - (1.0 - self.x) * node_interval_size;
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

            let child_index = children
                .interval_starts
                .iter()
                .zip(children.interval_sizes.iter())
                .position(|(&interval_start, &interval_size)| {
                    let interval_end = interval_start + interval_size;
                    interval_size > 0.0 && self.y >= interval_start && self.y <= interval_end
                });
            let child_index = match child_index {
                Some(candidate_child_index) => candidate_child_index,
                None => break,
            };

            let child = &children.nodes[child_index];
            let child_interval_size = children.interval_sizes[child_index];
            let child_interval_start = children.interval_starts[child_index];

            if self.x < 1.0 - child.probability {
                break;
            }

            nodes_from_root.push(child);
            self.path.push(child.token_id);
            self.y = ((self.y - child_interval_start) / child_interval_size).clamp(0.0, 1.0);
            self.x = ((self.x - (1.0 - child_interval_size)) / child_interval_size).clamp(0.0, 1.0);
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
            speed: 5.0,
            steering: Vec2::default(),
        }
    }

    pub fn update(&mut self, app: &App, update: Update) {
        let inference_tree = self.inference_tree.lock().unwrap();

        let window_rect = app.window_rect();
        let right_half_width = window_rect.w() / 3.0;

        self.steering = (app.mouse.position()).clamp_length_max(right_half_width);
        let speed =
            (self.steering / right_half_width * self.speed * update.since_last.secs() as f32)
                .as_f64()
                * (1.0 - self.cursor.x);
        if self.moving {
            self.cursor.x += speed.x;
            self.cursor.y -= speed.y;
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

struct DisplayInterval<'a> {
    start: f32,
    end: f32,
    depth: usize,
    node: &'a InferenceTreeNode,
}

fn intervals_from_tree<'a>(
    inference_tree: &'a InferenceTree,
    cursor: &Cursor,
    window_wh: Vec2,
    right_half_width: f32,
) -> Vec<DisplayInterval<'a>> {
    let nodes_from_root = inference_tree.get_nodes_on_path(&cursor.path);
    assert!(nodes_from_root.len() > 0);
    let size = (right_half_width as f64) / (1.0 - cursor.x);
    let mut intervals = Vec::new();
    intervals_from_parent_nodes(
        &nodes_from_root,
        &cursor.path,
        window_wh,
        size,
        -cursor.y * size,
        cursor.path.len(),
        &mut intervals,
    );
    intervals.reverse();
    intervals_from_node(
        nodes_from_root.last().unwrap(),
        window_wh.y as f64,
        size,
        -(cursor.y as f64) * size,
        cursor.path.len(),
        &mut intervals,
        (-0.5 * window_wh.y - 1.0) as f64,
        (0.5 * window_wh.y + 1.0) as f64,
    );
    intervals
}

fn intervals_from_parent_nodes<'a>(
    nodes: &[&'a InferenceTreeNode],
    path: &[TokenId],
    window_wh: Vec2,
    size: f64,
    start: f64,
    depth: usize,
    output: &mut Vec<DisplayInterval<'a>>,
) {
    if nodes.len() < 2 {
        return;
    }

    let node_index: usize = path[nodes.len() - 1].try_into().unwrap();
    let parent = nodes[nodes.len() - 2];
    let parent_children = parent.children.as_ref().unwrap();

    let parent_size = size / parent_children.interval_sizes[node_index];
    let parent_start = start - parent_children.interval_starts[node_index] * parent_size;
    let parent_depth = depth - 1;

    output.push(DisplayInterval {
        start: parent_start as f32,
        end: (parent_start + parent_size) as f32,
        depth: parent_depth,
        node: parent,
    });

    if (parent_size as f32) < window_wh.x || (parent_size as f32) < window_wh.y {
        intervals_from_parent_nodes(
            &nodes[..nodes.len() - 1],
            &path[..nodes.len() - 1],
            window_wh,
            parent_size,
            parent_start,
            parent_depth,
            output,
        );
    }
}

fn intervals_from_node<'a>(
    node: &'a InferenceTreeNode,
    window_height: f64,
    size: f64,
    start: f64,
    depth: usize,
    output: &mut Vec<DisplayInterval<'a>>,
    output_start: f64,
    output_end: f64,
) {
    let end = start + size;
    output.push(DisplayInterval {
        start: start as f32,
        end: end as f32,
        depth,
        node,
    });

    if size < 1.0 {
        return;
    }

    if let Some(children) = node.children.as_ref() {
        for &child_index in &children.indices_by_interval_size {
            let child_size = children.interval_sizes[child_index] * size;
            if child_size < 1.0 {
                break;
            }
            let child_start = children.interval_starts[child_index] * size + start;
            let child_end = child_start + child_size;
            if child_end >= output_start && child_start <= output_end {
                intervals_from_node(
                    &children.nodes[child_index],
                    window_height,
                    child_size,
                    child_start,
                    depth + 1,
                    output,
                    output_start,
                    output_end,
                );
            }
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
        window_rect.wh(),
        right_half_width,
    );

    let cursor_text = get_text_at_path(&inference_tree, &model.cursor.path);

    for &DisplayInterval {
        start,
        end,
        depth,
        node: _,
    } in &intervals
    {
        let size = end - start;
        let w_clamped = size.min(window_rect.len());
        let rect = Rect::from_x_y_w_h(
            right_half_width - 0.5 * w_clamped,
            -0.5 * (start + end),
            w_clamped,
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

    // Generated text
    let text_rect = Rect::from_wh(0.5 * window_rect.wh()).top_left_of(window_rect);
    draw.text(&cursor_text)
        .xy(text_rect.xy())
        .wh(text_rect.wh())
        .left_justify()
        .align_text_bottom()
        .color(WHITE);

    // Text of possible next tokens
    for interval in intervals {
        if interval.depth != model.cursor.path.len() + 1 {
            continue;
        }
        if interval.end - interval.start < 20.0 {
            continue;
        }
        if let Ok(text) = String::from_utf8(interval.node.token.clone()) {
            draw.text(&text)
                .x_y(
                    right_half_width + 0.5 * (0.5 * window_rect.w() - right_half_width),
                    -0.5 * (interval.start + interval.end),
                )
                .w_h(
                    0.5 * window_rect.w() - right_half_width,
                    interval.end - interval.start,
                )
                .left_justify()
                .color(if text.starts_with(' ') { RED } else { WHITE });
        }
    }

    drop(inference_tree);

    draw.to_frame(app, &frame).unwrap();
}
