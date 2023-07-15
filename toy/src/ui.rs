use nannou::prelude::*;

pub struct UIModel {
    _window: window::Id,
}

pub fn model(app: &App) -> UIModel {
    let _window = app.new_window().view(view).build().unwrap();
    UIModel { _window }
}

pub fn update(_app: &App, _model: &mut UIModel, _update: Update) {}

fn view(app: &App, _model: &UIModel, frame: Frame) {
    let draw = app.draw();
    draw.background().color(PLUM);
    draw.ellipse().color(STEELBLUE);
    draw.to_frame(app, &frame).unwrap();
}
