//! The simplified interface: a searchable preset browser and the preset's
//! macro knobs.

use std::collections::BTreeMap;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, OnceLock};

use nih_plug::prelude::{Editor, Param, ParamSetter};
use nih_plug_egui::egui::{self, Color32, RichText, Vec2};
use nih_plug_egui::{create_egui_editor, resizable_window::ResizableWindow};

use ezmix_content::captures::{IR_PACK, NAM_PACK};
use ezmix_content::Library;

use crate::loader::{LoadedInfo, Shared};
use crate::widgets::{self, ACCENT};
use crate::{EzParams, PresetSelection, RackSlot, RackState, MAX_SLOTS, NUM_KNOBS};

/// Searchable metadata for one preset.
#[derive(Clone, Debug)]
pub struct IndexEntry {
    pub pack_id: String,
    pub pack_name: String,
    pub path: String,
    pub name: String,
    pub folder: String,
    pub kind: String,
    pub instrument: String,
    pub search: String,
}

/// Built once per process on a background thread.
fn index(shared: &Arc<Shared>) -> Option<Arc<Vec<IndexEntry>>> {
    static INDEX: OnceLock<Mutex<Option<Arc<Vec<IndexEntry>>>>> = OnceLock::new();
    let cell = INDEX.get_or_init(|| {
        let lib = shared.library.clone();
        let cell = Mutex::new(None);
        std::thread::spawn(move || {
            let mut out = Vec::new();
            for pack in &lib.packs {
                for r in &pack.presets {
                    let (kind, instrument) = match lib.load_preset(r) {
                        Ok(p) => (p.tags.kind, p.tags.instrument),
                        Err(_) => (String::new(), String::new()),
                    };
                    let search = format!("{} {} {} {} {}", r.name, r.folder, pack.name, kind, instrument).to_lowercase();
                    out.push(IndexEntry {
                        pack_id: pack.id.clone(),
                        pack_name: pack.name.clone(),
                        path: r.path.clone(),
                        name: r.name.clone(),
                        folder: r.folder.clone(),
                        kind,
                        instrument,
                        search,
                    });
                }
            }
            if let Some(c) = INDEX.get() {
                *c.lock().unwrap_or_else(|e| e.into_inner()) = Some(Arc::new(out));
            }
        });
        cell
    });
    cell.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

/// Normalize free-form instrument tags into a short filter list.
fn instrument_family(tag: &str) -> &'static str {
    let t = tag.to_lowercase();
    let pick = [
        ("vocal", "Vocals"),
        ("drum", "Drums"),
        ("kick", "Drums"),
        ("snare", "Drums"),
        ("tom", "Drums"),
        ("cymbal", "Drums"),
        ("perc", "Drums"),
        ("bass", "Bass"),
        ("guitar", "Guitar"),
        ("piano", "Keys"),
        ("key", "Keys"),
        ("organ", "Keys"),
        ("synth", "Keys"),
        ("string", "Orchestral"),
        ("brass", "Orchestral"),
        ("wind", "Orchestral"),
        ("master", "Mastering"),
        ("mix", "Mastering"),
    ];
    pick.iter().find(|(k, _)| t.contains(k)).map(|(_, v)| *v).unwrap_or("Other")
}

struct UiState {
    search: String,
    pack: Option<String>,
    family: Option<&'static str>,
    restored: bool,
    /// Channel of the tab last scrolled into view.
    shown_active: Option<usize>,
}

/// Where the rack is remembered between sessions.
fn state_file(name: &str) -> Option<std::path::PathBuf> {
    if cfg!(test) {
        return None;
    }
    std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join("Library/Application Support/EZ-er Mix").join(name))
}

fn remember(rack: &RackState) {
    if let Some(f) = state_file("last-rack.json") {
        let _ = std::fs::create_dir_all(f.parent().unwrap_or(&f));
        if let Ok(json) = serde_json::to_string(rack) {
            let _ = std::fs::write(f, json);
        }
    }
}

/// The rack on a first start: TONE3000's "EVH 5150 Iconic Chainsaw" amp
/// capture into an EVH 5150 III cabinet IR, as far as those files are in a
/// capture folder (matched by the end of their path).
const DEFAULT_RACK: [(&str, &str); 2] = [(NAM_PACK, "EVH 5150 Iconic Chainsaw/5150  MAX Chainsaw.nam"), (IR_PACK, "EVH 5150 III.wav")];

pub(crate) fn default_rack(lib: &Library) -> Vec<PresetSelection> {
    DEFAULT_RACK
        .iter()
        .filter_map(|(pack, tail)| {
            let r = lib.pack(pack)?.presets.iter().find(|r| r.path == *tail || r.path.ends_with(&format!("/{tail}")))?;
            Some(PresetSelection { pack_id: r.pack_id.clone(), path: r.path.clone() })
        })
        .collect()
}

/// `EZERMIX_PRESET="<pack id>:<path>[;<pack id>:<path>…]"` overrides the
/// remembered rack; without one, [`DEFAULT_RACK`].
fn startup_rack(lib: &Library) -> Option<RackState> {
    let from_selections = |sels: Vec<PresetSelection>| {
        let slots: Vec<RackSlot> = sels.into_iter().take(MAX_SLOTS).enumerate().map(|(channel, preset)| RackSlot { channel, preset }).collect();
        (!slots.is_empty()).then_some(RackState { slots, active: 0 })
    };
    if let Ok(v) = std::env::var("EZERMIX_PRESET") {
        let sels =
            v.split(';').filter_map(|s| s.split_once(':')).map(|(pack_id, path)| PresetSelection { pack_id: pack_id.into(), path: path.into() }).collect();
        return from_selections(sels);
    }
    if let Some(rack) = state_file("last-rack.json").and_then(|f| std::fs::read_to_string(f).ok()).and_then(|t| serde_json::from_str(&t).ok()) {
        return Some(rack);
    }
    // Single preset remembered by earlier versions.
    let legacy = state_file("last-preset.json").and_then(|f| std::fs::read_to_string(f).ok()).and_then(|t| serde_json::from_str(&t).ok());
    from_selections(legacy.map_or_else(|| default_rack(lib), |sel| vec![sel]))
}

pub fn create(params: Arc<EzParams>, shared: Arc<Shared>) -> Option<Box<dyn Editor>> {
    let egui_state = params.editor_state.clone();
    create_egui_editor(
        params.editor_state.clone(),
        UiState { search: String::new(), pack: None, family: None, restored: false, shown_active: None },
        |ctx, _| {
            let mut v = egui::Visuals::dark();
            v.panel_fill = Color32::from_rgb(24, 25, 29);
            v.selection.bg_fill = ACCENT.linear_multiply(0.6);
            ctx.set_visuals(v);
        },
        move |ctx, setter, state| {
            // Keep meters and loading state fresh.
            ctx.request_repaint_after(std::time::Duration::from_millis(33));
            ResizableWindow::new("ez-er-mix").min_size(Vec2::new(780.0, 480.0)).show(ctx, egui_state.as_ref(), |ui| {
                draw(ui, &params, &shared, setter, state);
            });
        },
    )
}

fn set_param<P: Param>(setter: &ParamSetter, param: &P, value: P::Plain) {
    setter.begin_set_parameter(param);
    setter.set_parameter(param, value);
    setter.end_set_parameter(param);
}

fn rack(params: &EzParams) -> std::sync::MutexGuard<'_, RackState> {
    params.rack.lock().unwrap_or_else(|e| e.into_inner())
}

/// Change the rack, then tell the audio thread and remember it.
fn edit_rack(params: &EzParams, shared: &Shared, publish: bool, f: impl FnOnce(&mut RackState)) {
    let mut rack = rack(params);
    f(&mut rack);
    rack.sanitize();
    if publish {
        shared.publish_order(rack.channels());
    }
    remember(&rack);
}

/// Load a preset into a slot channel, starting from the preset's own knob
/// positions.
fn load_into(channel: usize, sel: &PresetSelection, params: &EzParams, shared: &Shared, setter: &ParamSetter) {
    let lib = &shared.library;
    let Some(r) = lib.find_preset(&sel.pack_id, &sel.path).cloned() else {
        return;
    };
    if let Ok(p) = lib.load_preset(&r) {
        for i in 0..NUM_KNOBS {
            set_param(setter, &params.slots[channel].knobs[i].value, p.knobs.get(i).map(|k| k.value).unwrap_or(0.0));
        }
    }
    shared.request_load(channel, &r.pack_id, &r.path);
}

/// The browser and previous/next load into the active tab.
fn select_preset(pack_id: &str, path: &str, params: &EzParams, shared: &Shared, setter: &ParamSetter) {
    let sel = PresetSelection { pack_id: pack_id.into(), path: path.into() };
    let channel = rack(params).active_slot().channel;
    load_into(channel, &sel, params, shared, setter);
    edit_rack(params, shared, false, |rack| {
        let active = rack.active;
        rack.slots[active].preset = sel;
    });
}

fn add_slot(params: &EzParams, shared: &Shared, setter: &ParamSetter) {
    let Some(channel) = rack(params).free_channel() else {
        return;
    };
    // The channel may hold values from an earlier tab.
    set_param(setter, &params.slots[channel].enabled, true);
    for k in &params.slots[channel].knobs {
        set_param(setter, &k.value, 0.5);
    }
    shared.request_clear(channel);
    edit_rack(params, shared, true, |rack| {
        rack.slots.push(RackSlot { channel, preset: PresetSelection::default() });
        rack.active = rack.slots.len() - 1;
    });
}

fn remove_slot(index: usize, params: &EzParams, shared: &Shared) {
    edit_rack(params, shared, true, |rack| {
        let removed = rack.slots.remove(index);
        shared.request_clear(removed.channel);
        if rack.slots.is_empty() {
            rack.slots.push(RackSlot { channel: removed.channel, preset: PresetSelection::default() });
        }
        if rack.active > index || rack.active >= rack.slots.len() {
            rack.active = rack.active.saturating_sub(1);
        }
    });
}

fn move_slot(from: usize, to: usize, params: &EzParams, shared: &Shared, publish: bool) {
    edit_rack(params, shared, publish, |rack| {
        let active = rack.slots[rack.active].channel;
        let slot = rack.slots.remove(from);
        rack.slots.insert(to.min(rack.slots.len()), slot);
        rack.active = rack.slots.iter().position(|s| s.channel == active).unwrap_or(0);
    });
}

fn slot_title(slot: &RackSlot, info: Option<&LoadedInfo>, idx: Option<&Arc<Vec<IndexEntry>>>) -> String {
    if slot.preset.is_empty() {
        return "Empty".into();
    }
    if let Some(i) = info.filter(|i| i.pack_id == slot.preset.pack_id && i.path == slot.preset.path) {
        return i.name.clone();
    }
    idx.and_then(|e| e.iter().find(|e| e.pack_id == slot.preset.pack_id && e.path == slot.preset.path))
        .map(|e| e.name.clone())
        .unwrap_or_else(|| slot.preset.path.rsplit('/').next().unwrap_or_default().to_string())
}

enum TabAction {
    Activate(usize),
    Toggle(usize),
    Remove(usize),
    Move { from: usize, to: usize, publish: bool },
    Add,
}

/// The rack as a row of tabs: power switch, name and close button each;
/// drag to reorder, right-click for options, "+" adds an effect.
fn tab_bar(ui: &mut egui::Ui, params: &EzParams, shared: &Arc<Shared>, setter: &ParamSetter, idx: Option<&Arc<Vec<IndexEntry>>>, state: &mut UiState) {
    let rack_now = rack(params).clone();
    let n = rack_now.slots.len();
    let mut action = None;
    let tab_h = 32.0;
    let add_w = 36.0;
    // Tabs scroll when they do not fit; "+" stays in view on the right.
    let tabs_w = ui.available_width() - add_w - 4.0;
    let tab_w = (tabs_w / n as f32 - 4.0).clamp(124.0, 190.0);
    let active_channel = rack_now.active_slot().channel;
    let scroll_to = state.shown_active.replace(active_channel) != Some(active_channel);

    ui.with_layout(egui::Layout::left_to_right(egui::Align::Min), |ui| {
        ui.allocate_ui_with_layout(Vec2::new(tabs_w, tab_h), egui::Layout::left_to_right(egui::Align::Min), |ui| {
            egui::ScrollArea::horizontal().id_salt("rack-tabs").auto_shrink([false, true]).show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 4.0;
                    let mut rects = Vec::with_capacity(n);
                    let mut dragging = None;
                    for (i, slot) in rack_now.slots.iter().enumerate() {
                        let ch = slot.channel;
                        let on = params.slots[ch].enabled.value();
                        let active = i == rack_now.active;
                        let info = shared.info(ch);
                        let loading = shared.slots[ch].loading.load(Ordering::Relaxed);
                        let title = slot_title(slot, info.as_ref(), idx);

                        // Keyed by channel so a drag stays on its tab while reordering.
                        let (_, rect) = ui.allocate_space(Vec2::new(tab_w, tab_h));
                        let body = ui.interact(rect, egui::Id::new(("rack-tab", ch)), egui::Sense::click_and_drag());
                        if active && scroll_to {
                            ui.scroll_to_rect(rect, None);
                        }
                        rects.push(rect);
                        let power_rect = egui::Rect::from_center_size(egui::pos2(rect.min.x + 16.0, rect.center().y), Vec2::splat(20.0));
                        let close_rect = egui::Rect::from_center_size(egui::pos2(rect.max.x - 14.0, rect.center().y), Vec2::splat(18.0));
                        let power = ui.interact(power_rect, body.id.with("power"), egui::Sense::click());
                        // Pointer-in-rect rather than `hovered()`: the close button
                        // sits on top of the tab and would steal its hover.
                        let over = ui.rect_contains_pointer(rect);
                        let show_close = (n > 1 || !slot.preset.is_empty()) && (active || over);
                        let close = show_close.then(|| ui.interact(close_rect, body.id.with("close"), egui::Sense::click()));

                        // Paint.
                        let painter = ui.painter_at(rect);
                        let bg = if active {
                            Color32::from_rgb(46, 48, 56)
                        } else if over || body.dragged() {
                            Color32::from_rgb(37, 38, 45)
                        } else {
                            Color32::from_rgb(31, 32, 37)
                        };
                        painter.rect_filled(rect, egui::CornerRadius { nw: 6, ne: 6, sw: 0, se: 0 }, bg);
                        if active {
                            painter.rect_filled(egui::Rect::from_min_max(egui::pos2(rect.min.x, rect.max.y - 2.0), rect.max), 0.0, ACCENT);
                        }
                        let pc = power_rect.center();
                        if on {
                            painter.circle_filled(pc, 6.0, ACCENT);
                        } else {
                            painter.circle_stroke(pc, 5.5, egui::Stroke::new(1.5_f32, Color32::from_gray(if power.hovered() { 200 } else { 120 })));
                        }
                        if power.hovered() {
                            painter.circle_stroke(pc, 9.0, egui::Stroke::new(1.0_f32, Color32::from_gray(90)));
                        }
                        let text_left = rect.min.x + 30.0;
                        let text_right = rect.max.x - if show_close { 28.0 } else { 10.0 } - if loading { 18.0 } else { 0.0 };
                        let color = if slot.preset.is_empty() {
                            Color32::from_gray(130)
                        } else if on {
                            Color32::from_gray(if active { 245 } else { 215 })
                        } else {
                            Color32::from_gray(120)
                        };
                        let galley = egui::WidgetText::from(RichText::new(&title).size(13.0).color(color)).into_galley(
                            ui,
                            Some(egui::TextWrapMode::Truncate),
                            (text_right - text_left).max(10.0),
                            egui::TextStyle::Body,
                        );
                        painter.galley(egui::pos2(text_left, rect.center().y - galley.size().y / 2.0), galley, color);
                        if loading {
                            egui::Spinner::new()
                                .size(12.0)
                                .paint_at(ui, egui::Rect::from_center_size(egui::pos2(text_right + 10.0, rect.center().y), Vec2::splat(12.0)));
                        }
                        if let Some(close) = &close {
                            let c = close_rect.center();
                            if close.hovered() {
                                painter.circle_filled(c, 8.0, Color32::from_gray(70));
                            }
                            let s = 3.5;
                            let stroke = egui::Stroke::new(1.5_f32, Color32::from_gray(if close.hovered() { 240 } else { 150 }));
                            painter.line_segment([c + Vec2::new(-s, -s), c + Vec2::new(s, s)], stroke);
                            painter.line_segment([c + Vec2::new(-s, s), c + Vec2::new(s, -s)], stroke);
                        }

                        // Interact.
                        if power.clicked() {
                            action = Some(TabAction::Toggle(ch));
                        } else if close.as_ref().is_some_and(|c| c.clicked()) {
                            action = Some(TabAction::Remove(i));
                        } else if body.clicked() || body.drag_started() {
                            action = Some(TabAction::Activate(i));
                        }
                        if body.dragged() {
                            dragging = Some(i);
                        }
                        if body.drag_stopped() {
                            action = Some(TabAction::Move { from: i, to: i, publish: true });
                        }
                        power.on_hover_text(if on { "Turn this effect off" } else { "Turn this effect on" });
                        let _ = body.context_menu(|ui| {
                            if ui.button(if on { "Turn off" } else { "Turn on" }).clicked() {
                                action = Some(TabAction::Toggle(ch));
                                ui.close_menu();
                            }
                            if ui.add_enabled(i > 0, egui::Button::new("Move left")).clicked() {
                                action = Some(TabAction::Move { from: i, to: i - 1, publish: true });
                                ui.close_menu();
                            }
                            if ui.add_enabled(i + 1 < n, egui::Button::new("Move right")).clicked() {
                                action = Some(TabAction::Move { from: i, to: i + 1, publish: true });
                                ui.close_menu();
                            }
                            ui.separator();
                            if ui.button("Remove").clicked() {
                                action = Some(TabAction::Remove(i));
                                ui.close_menu();
                            }
                        });
                        body.on_hover_text(format!("Rack slot {} (host parameters \"Slot {} …\")\nDrag to reorder, right-click for options", i + 1, ch + 1));
                    }

                    // Live reordering while a tab is dragged; the audio thread gets
                    // the new order when the drag ends.
                    if let (Some(from), Some(pos)) = (dragging, ui.ctx().pointer_interact_pos()) {
                        if let Some(to) = rects.iter().position(|r| r.x_range().contains(pos.x)) {
                            if to != from {
                                action = Some(TabAction::Move { from, to, publish: false });
                            }
                        }
                    }
                });
            });
        });
        let can_add = rack_now.free_channel().is_some();
        let add = ui.add_enabled(can_add, egui::Button::new(RichText::new("+").size(18.0)).min_size(Vec2::new(add_w, tab_h)).corner_radius(6.0));
        if add.clicked() {
            action = Some(TabAction::Add);
        }
        add.on_hover_text(if can_add { "Add an effect to the rack (effects run left to right)" } else { "The rack is full" });
    });
    // Baseline under the tabs.
    let y = ui.min_rect().max.y;
    ui.painter().hline(ui.min_rect().x_range(), y, egui::Stroke::new(1.0_f32, Color32::from_gray(50)));
    ui.add_space(8.0);

    match action {
        Some(TabAction::Activate(i)) => edit_rack(params, shared, false, |r| r.active = i),
        Some(TabAction::Toggle(ch)) => set_param(setter, &params.slots[ch].enabled, !params.slots[ch].enabled.value()),
        Some(TabAction::Remove(i)) => remove_slot(i, params, shared),
        Some(TabAction::Move { from, to, publish }) => move_slot(from, to, params, shared, publish),
        Some(TabAction::Add) => add_slot(params, shared, setter),
        None => {}
    }
}

fn draw(ui: &mut egui::Ui, params: &EzParams, shared: &Arc<Shared>, setter: &ParamSetter, state: &mut UiState) {
    // A fresh instance with an empty rack starts on the last-used rack.
    if !state.restored {
        state.restored = true;
        let fresh = {
            let r = rack(params);
            r.slots.len() == 1 && r.slots[0].preset.is_empty()
        };
        if fresh {
            if let Some(mut saved) = startup_rack(&shared.library) {
                saved.sanitize();
                for slot in &saved.slots {
                    if !slot.preset.is_empty() {
                        load_into(slot.channel, &slot.preset, params, shared, setter);
                    }
                }
                edit_rack(params, shared, true, |r| *r = saved);
            }
        }
    }
    let rack_now = rack(params).clone();
    let active = rack_now.active_slot().clone();
    let channel = active.channel;
    let current = shared.info(channel).filter(|i| i.pack_id == active.preset.pack_id && i.path == active.preset.path);
    let selection = active.preset.clone();
    let idx = index(shared);

    // Filtered list (also drives previous/next).
    let filtered: Vec<&IndexEntry> = match &idx {
        Some(entries) => {
            let q = state.search.to_lowercase();
            entries
                .iter()
                .filter(|e| state.pack.as_ref().is_none_or(|p| &e.pack_id == p))
                .filter(|e| state.family.is_none_or(|f| instrument_family(&e.instrument) == f))
                .filter(|e| q.split_whitespace().all(|w| e.search.contains(w)))
                .collect()
        }
        None => Vec::new(),
    };
    let pos = filtered.iter().position(|e| e.pack_id == selection.pack_id && e.path == selection.path);

    // Header -----------------------------------------------------------------
    egui::TopBottomPanel::top("header").exact_height(64.0).show_inside(ui, |ui| {
        ui.horizontal_centered(|ui| {
            ui.label(RichText::new("EZ-ER MIX").strong().size(20.0).color(ACCENT));
            ui.add_space(16.0);
            let nav = |ui: &mut egui::Ui, label: &str| ui.add_enabled(!filtered.is_empty(), egui::Button::new(label).min_size(Vec2::new(28.0, 28.0))).clicked();
            if nav(ui, "◀") {
                let i = pos.map(|p| (p + filtered.len() - 1) % filtered.len()).unwrap_or(0);
                select_preset(&filtered[i].pack_id, &filtered[i].path, params, shared, setter);
            }
            let title = if selection.is_empty() { "Empty slot".to_string() } else { slot_title(&active, current.as_ref(), idx.as_ref()) };
            ui.add_sized([280.0, 28.0], egui::Label::new(RichText::new(title).size(17.0)));
            if nav(ui, "▶") {
                let i = pos.map(|p| (p + 1) % filtered.len()).unwrap_or(0);
                select_preset(&filtered[i].pack_id, &filtered[i].path, params, shared, setter);
            }
            if shared.slots[channel].loading.load(Ordering::Relaxed) {
                ui.spinner();
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let bypass = params.bypass.value();
                if ui.selectable_label(bypass, if bypass { "Bypassed" } else { "Bypass" }).clicked() {
                    setter.begin_set_parameter(&params.bypass);
                    setter.set_parameter(&params.bypass, !bypass);
                    setter.end_set_parameter(&params.bypass);
                }
                ui.add_space(12.0);
                ui.allocate_ui(Vec2::new(124.0, 36.0), |ui| {
                    ui.vertical(|ui| {
                        widgets::meter(ui, "IN", shared.in_peak.load());
                        widgets::meter(ui, "OUT", shared.out_peak.load());
                    });
                });
            });
        });
    });

    // Browser ----------------------------------------------------------------
    egui::SidePanel::left("browser").resizable(true).default_width(320.0).min_width(240.0).show_inside(ui, |ui| {
        ui.add_space(6.0);
        ui.add(egui::TextEdit::singleline(&mut state.search).hint_text("Search presets…").desired_width(f32::INFINITY));
        ui.horizontal(|ui| {
            let pack_label = state
                .pack
                .as_ref()
                .and_then(|id| shared.library.pack(id))
                .map(|p| p.name.clone())
                .unwrap_or_else(|| "All packs".into());
            egui::ComboBox::from_id_salt("pack").selected_text(pack_label).width(140.0).show_ui(ui, |ui| {
                ui.selectable_value(&mut state.pack, None, "All packs");
                for p in &shared.library.packs {
                    ui.selectable_value(&mut state.pack, Some(p.id.clone()), &p.name);
                }
            });
            egui::ComboBox::from_id_salt("family").selected_text(state.family.unwrap_or("All instruments")).width(130.0).show_ui(ui, |ui| {
                ui.selectable_value(&mut state.family, None, "All instruments");
                for f in ["Vocals", "Drums", "Bass", "Guitar", "Keys", "Orchestral", "Mastering", "Other"] {
                    ui.selectable_value(&mut state.family, Some(f), f);
                }
            });
        });
        ui.separator();
        if idx.is_none() {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label("Indexing presets…");
            });
            return;
        }
        ui.label(RichText::new(format!("{} presets", filtered.len())).small().weak());
        // Group by pack, then folder.
        let mut tree: BTreeMap<(usize, String), BTreeMap<String, Vec<&IndexEntry>>> = BTreeMap::new();
        for &e in &filtered {
            let order = shared.library.packs.iter().position(|p| p.id == e.pack_id).unwrap_or(usize::MAX);
            tree.entry((order, e.pack_name.clone())).or_default().entry(e.folder.clone()).or_default().push(e);
        }
        let searching = !state.search.is_empty() || state.family.is_some();
        egui::ScrollArea::vertical().auto_shrink([false; 2]).show(ui, |ui| {
            for ((_, pack_name), folders) in &tree {
                let contains_sel = folders.values().flatten().any(|e| e.pack_id == selection.pack_id);
                egui::CollapsingHeader::new(RichText::new(pack_name).strong())
                    .id_salt(("pack", pack_name))
                    .default_open(contains_sel || searching || tree.len() == 1)
                    .show(ui, |ui| {
                        for (folder, entries) in folders {
                            let list = |ui: &mut egui::Ui| {
                                for e in entries {
                                    let selected = e.pack_id == selection.pack_id && e.path == selection.path;
                                    let resp = ui.selectable_label(selected, &e.name);
                                    let resp = if e.kind.is_empty() { resp } else { resp.on_hover_text(format!("{} · {}", e.kind, e.instrument)) };
                                    if resp.clicked() {
                                        select_preset(&e.pack_id, &e.path, params, shared, setter);
                                    }
                                }
                            };
                            if folder.is_empty() {
                                list(ui);
                            } else {
                                let open = searching || entries.iter().any(|e| e.path == selection.path && e.pack_id == selection.pack_id);
                                egui::CollapsingHeader::new(folder).id_salt(("folder", pack_name, folder)).default_open(open).show(ui, list);
                            }
                        }
                    });
            }
        });
    });

    // Controls ---------------------------------------------------------------
    egui::CentralPanel::default().show_inside(ui, |ui| {
        ui.add_space(4.0);
        tab_bar(ui, params, shared, setter, idx.as_ref(), state);
        let error = shared.slots[channel].error.lock().unwrap_or_else(|e| e.into_inner()).clone();
        let Some(info) = current else {
            ui.centered_and_justified(|ui| {
                let msg = if shared.library.packs.is_empty() {
                    "No EZmix preset packs were found.\nExpected them in /Library/Application Support/EZmix or /Users/Shared/Toontrack/EZmix/Presets."
                        .to_string()
                } else if let Some(e) = error.filter(|_| !selection.is_empty()) {
                    e
                } else if !selection.is_empty() {
                    "Loading…".to_string()
                } else {
                    "Pick a preset from the browser to load it into this slot.".to_string()
                };
                ui.label(RichText::new(msg).size(15.0).weak());
            });
            return;
        };
        let knobs = &params.slots[channel].knobs;
        ui.add_space(8.0);
        let pack_name = shared.library.pack(&info.pack_id).map(|p| p.name.clone()).unwrap_or_default();
        let folder = info.path.trim_start_matches("/Presets/").rsplit_once('/').map(|(f, _)| f.to_string()).unwrap_or_default();
        ui.label(RichText::new(format!("{pack_name}{}{folder}", if folder.is_empty() { "" } else { " · " })).weak());
        let tags: Vec<&str> = [info.tags.kind.as_str(), info.tags.instrument.as_str(), info.tags.genre.as_str(), info.tags.producer.as_str()]
            .into_iter()
            .filter(|t| !t.is_empty() && *t != "Generic" && *t != "Any")
            .collect();
        if !tags.is_empty() {
            ui.label(RichText::new(tags.join("  ·  ")).small().color(Color32::from_gray(150)));
        }
        ui.add_space(14.0);

        // Macro knobs, laid out by the preset's knob groups.
        let n = info.knobs.len().min(NUM_KNOBS);
        let mut sections: Vec<(String, Vec<usize>)> = Vec::new();
        for g in &info.groups {
            let members: Vec<usize> = (g.first..=g.last).filter(|&i| i < n).collect();
            if !members.is_empty() {
                sections.push((g.name.clone(), members));
            }
        }
        let grouped: Vec<usize> = sections.iter().flat_map(|(_, m)| m.clone()).collect();
        let loose: Vec<usize> = (0..n).filter(|i| !grouped.contains(i)).collect();
        if !loose.is_empty() {
            sections.insert(0, (String::new(), loose));
        }
        if n == 0 {
            ui.label(RichText::new("This preset has no adjustable controls.").weak());
        }
        ui.horizontal_wrapped(|ui| {
            for (name, members) in &sections {
                ui.vertical(|ui| {
                    ui.label(RichText::new(if name.is_empty() { " " } else { name }).small().color(ACCENT));
                    ui.horizontal_wrapped(|ui| {
                        for &i in members {
                            let k = &info.knobs[i];
                            let p = &knobs[i].value;
                            if k.is_button {
                                let (_, toggled) = widgets::toggle(ui, p, setter, &k.name, 76.0);
                                // Radio behaviour inside a button group.
                                if toggled && k.button_group && p.unmodulated_normalized_value() <= 0.5 {
                                    for (j, other) in info.knobs.iter().enumerate().take(n) {
                                        if j != i && other.is_button && other.button_group {
                                            let op = &knobs[j].value;
                                            setter.begin_set_parameter(op);
                                            setter.set_parameter(op, 0.0);
                                            setter.end_set_parameter(op);
                                        }
                                    }
                                }
                            } else {
                                widgets::knob(ui, p, setter, &k.name, 76.0, Some(k.default));
                            }
                        }
                    });
                });
                ui.add_space(18.0);
            }
        });

        ui.add_space(18.0);
        ui.separator();
        ui.horizontal(|ui| {
            widgets::knob(ui, &params.input, setter, "Input", 64.0, None);
            widgets::knob(ui, &params.output, setter, "Output", 64.0, None);
        });

        // Notes about what could not be reproduced.
        let mut notes = Vec::new();
        if info.report.skipped_neural > 0 {
            notes.push("AI console stage skipped (not supported)".to_string());
        }
        for m in &info.report.missing_models {
            notes.push(format!("Amp model not found: {m}"));
        }
        for e in &info.report.load_errors {
            notes.push(format!("Could not load {e}"));
        }
        if !info.report.unsupported.is_empty() {
            notes.push(format!("Bypassed: {}", info.report.unsupported.join(", ")));
        }
        if !params.slots[channel].enabled.value() {
            notes.push("This effect is turned off".to_string());
        }
        if let Some(e) = error {
            notes.push(e);
        }
        for n in notes {
            ui.label(RichText::new(n).small().color(Color32::from_rgb(220, 170, 90)));
        }
    });
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use nih_plug::prelude::{GuiContext, ParamPtr, PluginApi, PluginState};
    use nih_plug_egui::egui::{pos2, vec2, Event, Id, Modifiers, PointerButton, Pos2, RawInput, Rect};

    use super::*;
    use crate::loader;

    /// Records parameter changes instead of forwarding them to a host.
    #[derive(Default)]
    struct Host(Mutex<Vec<(ParamPtr, f32)>>);

    impl GuiContext for Host {
        fn plugin_api(&self) -> PluginApi {
            PluginApi::Clap
        }
        fn request_resize(&self) -> bool {
            false
        }
        unsafe fn raw_begin_set_parameter(&self, _: ParamPtr) {}
        unsafe fn raw_set_parameter_normalized(&self, param: ParamPtr, normalized: f32) {
            self.0.lock().unwrap().push((param, normalized));
        }
        unsafe fn raw_end_set_parameter(&self, _: ParamPtr) {}
        fn get_state(&self) -> PluginState {
            unimplemented!()
        }
        fn set_state(&self, _: PluginState) {
            unimplemented!()
        }
    }

    struct Harness {
        ctx: egui::Context,
        params: Arc<EzParams>,
        shared: Arc<Shared>,
        host: Host,
        state: UiState,
        events: Vec<Event>,
        time: f64,
        pointer: Pos2,
    }

    impl Harness {
        fn new() -> Self {
            Self {
                ctx: egui::Context::default(),
                params: Arc::new(EzParams::default()),
                shared: loader::spawn(),
                host: Host::default(),
                // Skip restoring the user's last rack.
                state: UiState { search: String::new(), pack: None, family: None, restored: true, shown_active: None },
                events: Vec::new(),
                time: 0.0,
                pointer: Pos2::ZERO,
            }
        }

        fn frame(&mut self) {
            let raw = RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(940.0, 580.0))),
                events: std::mem::take(&mut self.events),
                time: Some(self.time),
                ..Default::default()
            };
            self.time += 1.0 / 60.0;
            let (params, shared, host, state) = (&self.params, &self.shared, &self.host, &mut self.state);
            let _ = self.ctx.run(raw, |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| draw(ui, params, shared, &ParamSetter::new(host), state));
            });
        }

        fn move_to(&mut self, p: Pos2) {
            self.pointer = p;
            self.events.push(Event::PointerMoved(p));
            self.frame();
        }

        fn button(&mut self, pressed: bool, button: PointerButton) {
            self.events.push(Event::PointerButton { pos: self.pointer, button, pressed, modifiers: Modifiers::default() });
            self.frame();
        }

        fn click(&mut self, p: Pos2) {
            self.move_to(p);
            self.button(true, PointerButton::Primary);
            self.button(false, PointerButton::Primary);
            self.frame();
        }

        fn drag(&mut self, from: Pos2, to: Pos2) {
            self.move_to(from);
            self.button(true, PointerButton::Primary);
            for i in 1..=12 {
                self.move_to(from + (to - from) * (i as f32 / 12.0));
            }
            self.button(false, PointerButton::Primary);
            self.frame();
        }

        fn tab(&self, channel: usize) -> Rect {
            self.ctx.read_response(Id::new(("rack-tab", channel))).expect("tab exists").rect
        }

        fn channels(&self) -> Vec<usize> {
            rack(&self.params).channels().collect()
        }

        fn active_channel(&self) -> usize {
            rack(&self.params).active_slot().channel
        }

        fn published(&self) -> Vec<usize> {
            let (order, n) = loader::unpack_order(self.shared.order.load(Ordering::Acquire));
            order[..n].to_vec()
        }

        fn add(&mut self) {
            let r = self.tab(*self.channels().last().unwrap());
            // "+" is pinned to the right edge of the window.
            self.click(pos2(940.0 - 8.0 - 18.0, r.center().y));
        }

        fn sets(&self, param: ParamPtr) -> Vec<f32> {
            self.host.0.lock().unwrap().iter().filter(|(p, _)| *p == param).map(|(_, v)| *v).collect()
        }
    }

    #[test]
    fn tabs_add_select_reorder_toggle_and_close() {
        let mut h = Harness::new();
        h.frame();
        h.frame();
        assert_eq!(h.channels(), [0]);

        // "+" adds an empty slot on a fresh channel and selects it.
        h.add();
        assert_eq!(h.channels(), [0, 1]);
        assert_eq!(h.active_channel(), 1);
        assert_eq!(h.published(), [0, 1]);
        assert_eq!(h.sets(h.params.slots[1].enabled.as_ptr()), [1.0]);
        assert_eq!(h.sets(h.params.slots[1].knobs[0].value.as_ptr()), [0.5]);
        h.add();
        assert_eq!(h.channels(), [0, 1, 2]);
        assert_eq!(h.active_channel(), 2);

        // Clicking a tab selects it.
        let r = h.tab(0);
        h.click(r.center());
        assert_eq!(h.active_channel(), 0);

        // Dragging moves the tab; the audio thread gets the order on release.
        let (from, to) = (h.tab(0).center(), h.tab(2).center());
        h.drag(from, to);
        assert_eq!(h.channels(), [1, 2, 0]);
        assert_eq!(h.active_channel(), 0, "the dragged tab stays selected");
        assert_eq!(h.published(), [1, 2, 0]);

        // The power dot switches a slot off without selecting it.
        let r = h.tab(1);
        h.click(pos2(r.min.x + 16.0, r.center().y));
        assert_eq!(h.sets(h.params.slots[1].enabled.as_ptr()), [1.0, 0.0]);
        assert_eq!(h.active_channel(), 0);

        // Hovering an inactive tab shows its close button steadily.
        let r = h.tab(2);
        let close = pos2(r.max.x - 14.0, r.center().y);
        h.move_to(close);
        for _ in 0..3 {
            h.frame();
            assert!(h.ctx.read_response(Id::new(("rack-tab", 2)).with("close")).is_some_and(|c| c.hovered()));
        }

        // Closing removes the slot and empties its channel.
        let before = h.shared.slots[2].generation.load(Ordering::SeqCst);
        h.click(close);
        assert_eq!(h.channels(), [1, 0]);
        assert_eq!(h.published(), [1, 0]);
        assert_eq!(h.active_channel(), 0);
        assert!(h.shared.slots[2].generation.load(Ordering::SeqCst) > before, "channel was cleared");

        // Closing the last tab leaves one empty tab.
        for _ in 0..2 {
            let ch = h.channels()[0];
            let r = h.tab(ch);
            h.click(r.center());
            h.click(pos2(r.max.x - 14.0, r.center().y));
        }
        assert_eq!(h.channels().len(), 1);
        assert_eq!(h.published().len(), 1);

        // The rack is full at MAX_SLOTS tabs.
        for _ in 0..MAX_SLOTS + 2 {
            h.add();
        }
        assert_eq!(h.channels().len(), MAX_SLOTS);
    }
}
