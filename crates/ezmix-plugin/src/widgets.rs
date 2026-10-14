//! Rotary knob and toggle widgets bound to nih-plug parameters.

use nih_plug::prelude::{Param, ParamSetter};
use nih_plug_egui::egui::{self, pos2, vec2, Color32, Response, Sense, Shape, Stroke, Ui};

pub const ACCENT: Color32 = Color32::from_rgb(232, 160, 64);
pub const TRACK: Color32 = Color32::from_rgb(58, 60, 66);
pub const KNOB_BG: Color32 = Color32::from_rgb(34, 35, 40);

const START: f32 = std::f32::consts::PI * 0.75;
const SWEEP: f32 = std::f32::consts::PI * 1.5;

fn arc_points(center: egui::Pos2, radius: f32, from: f32, to: f32) -> Vec<egui::Pos2> {
    let steps = ((to - from).abs() * 24.0).ceil().max(2.0) as usize;
    (0..=steps)
        .map(|i| {
            let a = from + (to - from) * i as f32 / steps as f32;
            pos2(center.x + radius * a.cos(), center.y + radius * a.sin())
        })
        .collect()
}

/// A rotary knob for any parameter. `reset_to` overrides the parameter's
/// default for double-click resets (presets supply their own defaults).
pub fn knob<P: Param>(ui: &mut Ui, param: &P, setter: &ParamSetter, label: &str, size: f32, reset_to: Option<f32>) -> Response {
    let (rect, mut response) = ui.allocate_exact_size(vec2(size, size + 34.0), Sense::click_and_drag());
    let knob_rect = egui::Rect::from_min_size(rect.min, vec2(size, size));
    let id = response.id;

    if response.drag_started() {
        setter.begin_set_parameter(param);
        ui.memory_mut(|m| m.data.insert_temp(id, param.unmodulated_normalized_value()));
    }
    if response.dragged() {
        let fine = ui.input(|i| i.modifiers.shift);
        let delta = -response.drag_delta().y + response.drag_delta().x;
        let start: f32 = ui.memory(|m| m.data.get_temp(id).unwrap_or(0.5));
        let v = (start + delta / if fine { 900.0 } else { 220.0 }).clamp(0.0, 1.0);
        ui.memory_mut(|m| m.data.insert_temp(id, v));
        setter.set_parameter_normalized(param, v);
        response.mark_changed();
    }
    if response.drag_stopped() {
        setter.end_set_parameter(param);
    }
    if response.double_clicked() {
        setter.begin_set_parameter(param);
        match reset_to {
            Some(v) => setter.set_parameter_normalized(param, v),
            None => setter.set_parameter(param, param.default_plain_value()),
        }
        setter.end_set_parameter(param);
        response.mark_changed();
    }

    let value = param.unmodulated_normalized_value();
    let painter = ui.painter_at(rect.expand(2.0));
    let c = knob_rect.center();
    let r = size * 0.5 - 4.0;
    let hovered = response.hovered() || response.dragged();
    painter.circle_filled(c, r - 6.0, KNOB_BG);
    painter.add(Shape::line(arc_points(c, r, START, START + SWEEP), Stroke::new(4.0_f32, TRACK)));
    if value > 0.001 {
        painter.add(Shape::line(arc_points(c, r, START, START + SWEEP * value), Stroke::new(4.0_f32, ACCENT)));
    }
    let a = START + SWEEP * value;
    painter.line_segment(
        [pos2(c.x + (r - 16.0) * a.cos(), c.y + (r - 16.0) * a.sin()), pos2(c.x + (r - 7.0) * a.cos(), c.y + (r - 7.0) * a.sin())],
        Stroke::new(3.0_f32, if hovered { Color32::WHITE } else { Color32::from_gray(210) }),
    );
    painter.text(
        pos2(c.x, rect.max.y - 24.0),
        egui::Align2::CENTER_CENTER,
        label,
        egui::FontId::proportional(13.0),
        Color32::from_gray(225),
    );
    let text = param.to_string();
    painter.text(pos2(c.x, rect.max.y - 8.0), egui::Align2::CENTER_CENTER, text, egui::FontId::proportional(11.0), Color32::from_gray(150));
    response.on_hover_text("Drag to change, Shift for fine control, double-click to reset")
}

/// An on/off switch stored in a 0..1 float parameter.
pub fn toggle<P: Param>(ui: &mut Ui, param: &P, setter: &ParamSetter, label: &str, width: f32) -> (Response, bool) {
    let on = param.unmodulated_normalized_value() > 0.5;
    let (rect, response) = ui.allocate_exact_size(vec2(width, 72.0), Sense::click());
    let painter = ui.painter_at(rect);
    let pill = egui::Rect::from_center_size(pos2(rect.center().x, rect.min.y + 22.0), vec2(44.0, 22.0));
    painter.rect_filled(pill, 11.0, if on { ACCENT } else { TRACK });
    let knob_x = if on { pill.max.x - 11.0 } else { pill.min.x + 11.0 };
    painter.circle_filled(pos2(knob_x, pill.center().y), 8.0, Color32::from_gray(235));
    painter.text(pos2(rect.center().x, rect.max.y - 22.0), egui::Align2::CENTER_CENTER, label, egui::FontId::proportional(13.0), Color32::from_gray(225));
    let mut toggled = false;
    if response.clicked() {
        setter.begin_set_parameter(param);
        setter.set_parameter_normalized(param, if on { 0.0 } else { 1.0 });
        setter.end_set_parameter(param);
        toggled = true;
    }
    (response, toggled)
}

/// Horizontal peak meter, `level` is linear gain.
pub fn meter(ui: &mut Ui, label: &str, level: f32) {
    let (rect, _) = ui.allocate_exact_size(vec2(120.0, 14.0), Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 3.0, TRACK);
    let db = 20.0 * level.max(1e-6).log10();
    let frac = ((db + 60.0) / 60.0).clamp(0.0, 1.0);
    let color = if db > -0.5 { Color32::from_rgb(230, 70, 60) } else if db > -6.0 { ACCENT } else { Color32::from_rgb(90, 190, 120) };
    let mut fill = rect;
    fill.set_width(rect.width() * frac);
    painter.rect_filled(fill, 3.0, color);
    painter.text(rect.left_center() + vec2(4.0, 0.0), egui::Align2::LEFT_CENTER, label, egui::FontId::proportional(10.0), Color32::WHITE);
}
