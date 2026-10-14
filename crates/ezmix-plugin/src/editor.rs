//! The simplified interface: a searchable preset browser and the preset's
//! macro knobs.

use std::collections::BTreeMap;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, OnceLock};

use nih_plug::prelude::{Editor, Param, ParamSetter};
use nih_plug_egui::egui::{self, Color32, RichText, Vec2};
use nih_plug_egui::{create_egui_editor, resizable_window::ResizableWindow};

use crate::loader::Shared;
use crate::widgets::{self, ACCENT};
use crate::{EzParams, PresetSelection, NUM_KNOBS};

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
}

/// Where the last chosen preset is remembered between sessions.
fn last_preset_file() -> Option<std::path::PathBuf> {
    std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join("Library/Application Support/EZ-er Mix/last-preset.json"))
}

fn remember(sel: &PresetSelection) {
    if let Some(f) = last_preset_file() {
        let _ = std::fs::create_dir_all(f.parent().unwrap_or(&f));
        if let Ok(json) = serde_json::to_string(sel) {
            let _ = std::fs::write(f, json);
        }
    }
}

/// `EZERMIX_PRESET="<pack id>:<path>"` overrides the remembered preset.
fn startup_preset() -> Option<PresetSelection> {
    if let Ok(v) = std::env::var("EZERMIX_PRESET") {
        if let Some((pack_id, path)) = v.split_once(':') {
            return Some(PresetSelection { pack_id: pack_id.into(), path: path.into() });
        }
    }
    let text = std::fs::read_to_string(last_preset_file()?).ok()?;
    serde_json::from_str(&text).ok()
}

pub fn create(params: Arc<EzParams>, shared: Arc<Shared>) -> Option<Box<dyn Editor>> {
    let egui_state = params.editor_state.clone();
    create_egui_editor(
        params.editor_state.clone(),
        UiState { search: String::new(), pack: None, family: None, restored: false },
        |ctx, _| {
            let mut v = egui::Visuals::dark();
            v.panel_fill = Color32::from_rgb(24, 25, 29);
            v.selection.bg_fill = ACCENT.linear_multiply(0.6);
            ctx.set_visuals(v);
        },
        move |ctx, setter, state| {
            // Keep meters and loading state fresh.
            ctx.request_repaint_after(std::time::Duration::from_millis(33));
            ResizableWindow::new("ez-er-mix").min_size(Vec2::new(760.0, 460.0)).show(ctx, egui_state.as_ref(), |ui| {
                draw(ui, &params, &shared, setter, state);
            });
        },
    )
}

fn select_preset(entry_pack: &str, entry_path: &str, params: &EzParams, shared: &Arc<Shared>, setter: &ParamSetter) {
    let lib = &shared.library;
    let Some(r) = lib.find_preset(entry_pack, entry_path).cloned() else { return };
    // Push the preset's knob positions to the host parameters first so the
    // new processor starts from them.
    if let Ok(p) = lib.load_preset(&r) {
        for i in 0..NUM_KNOBS {
            let v = p.knobs.get(i).map(|k| k.value).unwrap_or(0.0);
            let param = &params.knobs[i].value;
            setter.begin_set_parameter(param);
            setter.set_parameter(param, v);
            setter.end_set_parameter(param);
        }
    }
    let sel = PresetSelection { pack_id: r.pack_id.clone(), path: r.path.clone() };
    remember(&sel);
    *params.selection.lock().unwrap_or_else(|e| e.into_inner()) = sel;
    shared.request_load(&r.pack_id, &r.path);
}

fn draw(ui: &mut egui::Ui, params: &EzParams, shared: &Arc<Shared>, setter: &ParamSetter, state: &mut UiState) {
    // A fresh instance with nothing selected starts on the last-used preset.
    if !state.restored {
        state.restored = true;
        let empty = params.selection.lock().unwrap_or_else(|e| e.into_inner()).pack_id.is_empty();
        if empty {
            if let Some(sel) = startup_preset() {
                select_preset(&sel.pack_id, &sel.path, params, shared, setter);
            }
        }
    }
    let current = shared.current.lock().unwrap_or_else(|e| e.into_inner()).clone();
    let selection = params.selection.lock().unwrap_or_else(|e| e.into_inner()).clone();
    let idx = index(shared);

    // Filtered list (also drives previous/next).
    let filtered: Vec<IndexEntry> = match &idx {
        Some(entries) => {
            let q = state.search.to_lowercase();
            entries
                .iter()
                .filter(|e| state.pack.as_ref().is_none_or(|p| &e.pack_id == p))
                .filter(|e| state.family.is_none_or(|f| instrument_family(&e.instrument) == f))
                .filter(|e| q.split_whitespace().all(|w| e.search.contains(w)))
                .cloned()
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
            let title = current.as_ref().map(|c| c.name.clone()).unwrap_or_else(|| "No preset loaded".into());
            ui.add_sized([280.0, 28.0], egui::Label::new(RichText::new(title).size(17.0)));
            if nav(ui, "▶") {
                let i = pos.map(|p| (p + 1) % filtered.len()).unwrap_or(0);
                select_preset(&filtered[i].pack_id, &filtered[i].path, params, shared, setter);
            }
            if shared.loading.load(Ordering::Relaxed) {
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
        for e in &filtered {
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
        let Some(info) = current else {
            ui.centered_and_justified(|ui| {
                let msg = if shared.library.packs.is_empty() {
                    "No EZmix preset packs were found.\nExpected them in /Library/Application Support/EZmix or /Users/Shared/Toontrack/EZmix/Presets."
                } else {
                    "Pick a preset from the browser."
                };
                ui.label(RichText::new(msg).size(15.0).weak());
            });
            return;
        };
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
                            let p = &params.knobs[i].value;
                            if k.is_button {
                                let (_, toggled) = widgets::toggle(ui, p, setter, &k.name, 76.0);
                                // Radio behaviour inside a button group.
                                if toggled && k.button_group && p.unmodulated_normalized_value() <= 0.5 {
                                    for (j, other) in info.knobs.iter().enumerate().take(n) {
                                        if j != i && other.is_button && other.button_group {
                                            let op = &params.knobs[j].value;
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
        if !info.report.unsupported.is_empty() {
            notes.push(format!("Bypassed: {}", info.report.unsupported.join(", ")));
        }
        if let Some(e) = shared.error.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
            notes.push(e.clone());
        }
        for n in notes {
            ui.label(RichText::new(n).small().color(Color32::from_rgb(220, 170, 90)));
        }
    });
}
