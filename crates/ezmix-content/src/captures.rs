//! Neural Amp Modeler captures (`.nam`) and cabinet impulse responses
//! (`.wav`) kept in folders on disk. They show up as two extra packs whose
//! preset paths are relative to the capture folder, and load as made-up
//! presets with a few amp or cabinet controls.

use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::library::{Pack, PresetRef};
use crate::preset::{IrState, Knob, KnobGroup, KnobTarget, NamState, Preset, Slot, SlotKind, SourceFormat, Tags};

pub const NAM_PACK: &str = "NAM";
pub const IR_PACK: &str = "IR";

/// Larger `.wav` files are recordings, not impulse responses.
const MAX_IR_BYTES: u64 = 4 << 20;
const MAX_DEPTH: usize = 6;

/// Folders searched for captures: `EZERMIX_CAPTURES` (separated by `:`),
/// then `~/Music/NAM` and `~/Library/Application Support/EZ-er Mix/Captures`.
pub fn dirs() -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = std::env::var_os("EZERMIX_CAPTURES").map(|v| std::env::split_paths(&v).collect()).unwrap_or_default();
    if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
        dirs.push(home.join("Music/NAM"));
        dirs.push(home.join("Library/Application Support/EZ-er Mix/Captures"));
    }
    dirs
}

/// The capture packs found in `dirs`; empty packs are left out.
pub fn scan(dirs: &[PathBuf]) -> Vec<Pack> {
    let mut nam: Vec<PresetRef> = Vec::new();
    let mut ir: Vec<PresetRef> = Vec::new();
    for root in dirs {
        walk(root, 0, &mut |file| {
            let Some(rel) = file.strip_prefix(root).ok().and_then(Path::to_str) else { return };
            let ext = file.extension().and_then(|e| e.to_str()).unwrap_or_default().to_ascii_lowercase();
            let list = match ext.as_str() {
                "nam" => &mut nam,
                "wav" if file.metadata().is_ok_and(|m| m.len() <= MAX_IR_BYTES) => &mut ir,
                _ => return,
            };
            // The first folder wins when two hold the same relative path.
            if !list.iter().any(|r| r.path == rel) {
                list.push(preset_ref(if ext == "nam" { NAM_PACK } else { IR_PACK }, rel, file));
            }
        });
    }
    [(NAM_PACK, "NAM Captures", nam), (IR_PACK, "Cab IRs", ir)]
        .into_iter()
        .filter(|(_, _, presets)| !presets.is_empty())
        .map(|(id, name, mut presets)| {
            presets.sort_by(|a, b| (&a.folder, &a.name).cmp(&(&b.folder, &b.name)));
            Pack { id: id.into(), name: name.into(), file: PathBuf::new(), presets }
        })
        .collect()
}

fn walk(dir: &Path, depth: usize, found: &mut impl FnMut(&Path)) {
    let Ok(read) = std::fs::read_dir(dir) else { return };
    let mut paths: Vec<PathBuf> = read.filter_map(|e| e.ok()).map(|e| e.path()).collect();
    paths.sort();
    for path in paths {
        if path.file_name().is_some_and(|n| n.to_string_lossy().starts_with('.')) {
            continue;
        }
        if path.is_dir() {
            if depth < MAX_DEPTH {
                walk(&path, depth + 1, found);
            }
        } else {
            found(&path);
        }
    }
}

fn preset_ref(pack_id: &str, rel: &str, file: &Path) -> PresetRef {
    let (folder, leaf) = rel.rsplit_once('/').unwrap_or(("", rel));
    let stem = leaf.rsplit_once('.').map_or(leaf, |(s, _)| s);
    PresetRef {
        pack_id: pack_id.into(),
        path: rel.into(),
        name: stem.split_whitespace().collect::<Vec<_>>().join(" "),
        folder: folder.into(),
        file: Some(file.to_path_buf()),
    }
}

/// Build the preset for a capture file.
pub fn load(r: &PresetRef, file: &Path) -> Result<Preset, String> {
    if r.pack_id == NAM_PACK {
        nam_preset(r, file)
    } else {
        Ok(ir_preset(r, file))
    }
}

fn knob(name: &str, param: usize, lo: f32, hi: f32, value: f32) -> Knob {
    Knob { name: name.into(), value, targets: vec![KnobTarget { slot: 0, param, lo, hi, curve: 0.0 }], is_button: false, button_group: false }
}

fn group(name: &str, first: usize, last: usize) -> KnobGroup {
    KnobGroup { name: name.into(), first, last }
}

fn instrument(r: &PresetRef) -> String {
    if r.path.to_lowercase().contains("bass") { "Bass" } else { "Guitar" }.into()
}

fn nam_preset(r: &PresetRef, file: &Path) -> Result<Preset, String> {
    let bytes = std::fs::read(file).map_err(|e| format!("{}: {e}", r.name))?;
    let json: Value = serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", r.name))?;
    // TONE3000 fills unknown fields with "T3K-…" placeholders.
    let meta = |key: &str| {
        json.pointer(&format!("/metadata/{key}")).and_then(Value::as_str).map(str::trim).filter(|v| !v.starts_with("T3K")).unwrap_or_default().to_string()
    };
    let tags = Tags { kind: gear_label(&meta("gear_type")), instrument: instrument(r), genre: meta("tone_type"), producer: meta("modeled_by") };
    let state =
        NamState { has_custom_model: true, custom_model: file.to_string_lossy().into_owned(), gate_active: true, eq_active: true, ..NamState::default() };
    Ok(Preset {
        name: r.name.clone(),
        tags,
        output_gain: 1.0,
        slots: vec![Slot { kind: SlotKind::Nam(state), active: true }],
        // NamState ranges: input -20..+20 dB, gate -100..0 dB, output -40..+40 dB.
        knobs: vec![
            knob("Gain", 0, 0.25, 0.75, 0.5),
            knob("Gate", 3, 0.0, 0.6, 0.0),
            knob("Bass", 6, 0.0, 1.0, 0.5),
            knob("Mid", 7, 0.0, 1.0, 0.5),
            knob("Treble", 5, 0.0, 1.0, 0.5),
            knob("Level", 1, 0.375, 0.625, 0.5),
            knob("Width", 8, 0.0, 1.0, 0.0),
        ],
        groups: vec![group("Amp", 0, 1), group("Tone", 2, 4), group("Output", 5, 6)],
        format: SourceFormat::Capture,
    })
}

fn ir_preset(r: &PresetRef, file: &Path) -> Preset {
    Preset {
        name: r.name.clone(),
        tags: Tags { kind: "Cabinet IR".into(), instrument: instrument(r), ..Tags::default() },
        output_gain: 1.0,
        slots: vec![Slot { kind: SlotKind::CabIr(IrState::new(file.to_path_buf())), active: true }],
        knobs: vec![knob("Level", 0, 0.0, 1.0, 0.5), knob("Low cut", 1, 0.0, 1.0, 0.0), knob("High cut", 2, 0.0, 1.0, 1.0)],
        groups: vec![group("Cabinet", 0, 2)],
        format: SourceFormat::Capture,
    }
}

fn gear_label(gear: &str) -> String {
    match gear {
        "amp" => "Amp".into(),
        "amp_cab" => "Amp + cab".into(),
        "full_rig" => "Full rig".into(),
        "pedal" => "Pedal".into(),
        "pedal_amp" => "Pedal + amp".into(),
        other => {
            let words = other.replace('_', " ");
            let mut chars = words.chars();
            chars.next().map(|c| c.to_uppercase().chain(chars).collect()).unwrap_or_default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scans_capture_folders() {
        let root = std::env::temp_dir().join(format!("ezermix-captures-{}", std::process::id()));
        let amp = root.join("EVH 5150 Iconic Chainsaw");
        std::fs::create_dir_all(&amp).unwrap();
        std::fs::write(amp.join("5150  MAX Chainsaw.nam"), r#"{"metadata":{"gear_type":"amp","tone_type":"T3K-Null","modeled_by":"someone"}}"#).unwrap();
        std::fs::write(root.join("Cab.wav"), b"RIFF").unwrap();
        std::fs::write(root.join("notes.txt"), b"").unwrap();
        std::fs::write(root.join(".hidden.nam"), b"").unwrap();
        let packs = scan(std::slice::from_ref(&root));
        let _ = std::fs::remove_dir_all(&root);

        assert_eq!(packs.iter().map(|p| p.id.as_str()).collect::<Vec<_>>(), [NAM_PACK, IR_PACK]);
        let r = &packs[0].presets[0];
        assert_eq!(
            (r.path.as_str(), r.name.as_str(), r.folder.as_str()),
            ("EVH 5150 Iconic Chainsaw/5150  MAX Chainsaw.nam", "5150 MAX Chainsaw", "EVH 5150 Iconic Chainsaw")
        );
        assert_eq!(packs[1].presets[0].path, "Cab.wav");
    }

    #[test]
    fn capture_presets() {
        let file = std::env::temp_dir().join(format!("ezermix-capture-{}.nam", std::process::id()));
        std::fs::write(&file, r#"{"metadata":{"gear_type":"amp_cab","tone_type":"T3K-Null","modeled_by":"someone"}}"#).unwrap();
        let r = preset_ref(NAM_PACK, "x.nam", &file);
        let p = load(&r, &file).unwrap();
        let _ = std::fs::remove_file(&file);
        assert_eq!((p.tags.kind.as_str(), p.tags.genre.as_str(), p.tags.producer.as_str()), ("Amp + cab", "", "someone"));
        let SlotKind::Nam(state) = &p.slots[0].kind else { panic!("not a NAM slot") };
        assert!(matches!(state.model(), crate::preset::NamModelRef::File(f) if f == file));
        // Knobs start at the capture's neutral settings.
        for k in &p.knobs {
            let mut s = state.clone();
            s.set_index(k.targets[0].param, k.targets[0].eval(k.value));
            assert_eq!(format!("{s:?}"), format!("{state:?}"), "{}", k.name);
        }

        let r = preset_ref(IR_PACK, "Cabs/Bass 8x10.wav", Path::new("/x/Cabs/Bass 8x10.wav"));
        let p = load(&r, Path::new("/x/Cabs/Bass 8x10.wav")).unwrap();
        assert_eq!((p.name.as_str(), p.tags.instrument.as_str(), p.knobs.len()), ("Bass 8x10", "Bass", 3));
    }
}
