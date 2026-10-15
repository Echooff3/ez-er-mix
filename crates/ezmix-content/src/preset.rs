//! Preset model shared by all pack generations.
//!
//! EZmix 3 presets are a brace-structured text document:
//!
//! ```text
//! Preset {
//! PresetInfo { Name "…" TagType "…" … }
//! PresetProcessor {
//! OutputGain 0.65
//! MixerChannel {
//! effect 0 { sidechain "" effect 8 active 1 chunk <b64 len> <raw len>
//! <base64 payload> }
//! …
//! }
//! PresetKnobs { knob 0 { name "…" target 0 Effect basic { ix2 1 ix3 4 range a b c KnobValue v } } … }
//! }
//! }
//! ```
//!
//! Effect type 8 is the multi-channel engine (XML, see [`crate::engine`]),
//! type 9 is a Neural Amp Modeler capture, type 10 is a TFLite "AI console"
//! model (not supported; such slots are bypassed).
//!
//! Older packs store the engine XML directly; they become a single engine
//! slot with one knob per labelled macro fader.

use std::path::{Path, PathBuf};

use crate::engine::{self, EngineState};

#[derive(Debug, Clone, Default)]
pub struct Tags {
    pub kind: String,
    pub instrument: String,
    pub genre: String,
    pub producer: String,
}

#[derive(Debug, Clone)]
pub struct NamState {
    pub has_custom_model: bool,
    pub internal_model: u32,
    pub custom_model: String,
    pub input_level: f32,
    pub output_level: f32,
    pub gate_active: bool,
    pub gate_threshold: f32,
    pub eq_active: bool,
    pub eq_bass: f32,
    pub eq_mid: f32,
    pub eq_treble: f32,
    pub stereo_phase: f32,
}

impl Default for NamState {
    fn default() -> Self {
        Self {
            has_custom_model: false,
            internal_model: 0,
            custom_model: String::new(),
            input_level: 0.5,
            output_level: 0.5,
            gate_active: false,
            gate_threshold: 0.0,
            eq_active: false,
            eq_bass: 0.5,
            eq_mid: 0.5,
            eq_treble: 0.5,
            stereo_phase: 0.0,
        }
    }
}

/// File stems in the `Models` resource for internal model indices 1..=18.
pub const INTERNAL_NAM_MODELS: [&str; 18] = [
    "RandalSatan",
    "DeluxeReverb",
    "Bassman",
    "JCM800",
    "Plexi",
    "Ampeg",
    "VOXAC50",
    "Peavey5150",
    "SovtekMIG50",
    "MesaBoogie",
    "SwedishChainsaw",
    "TubeScreamer",
    "KLON",
    "RatPedal",
    "BossBluesDrive",
    "BluesbreakerPedal",
    "HM2",
    "DarkglassB7K",
];

pub enum NamModelRef<'a> {
    None,
    Internal(&'static str),
    Custom(&'a str),
    /// A `.nam` file on disk (captures from a capture folder).
    File(&'a Path),
}

impl NamState {
    /// Knob parameter index -> normalized value accessors.
    pub fn set_index(&mut self, index: usize, v: f32) {
        match index {
            0 => self.input_level = v,
            1 => self.output_level = v,
            2 => self.gate_active = v > 0.5,
            3 => self.gate_threshold = v,
            4 => self.eq_active = v > 0.5,
            5 => self.eq_treble = v,
            6 => self.eq_bass = v,
            7 => self.eq_mid = v,
            8 => self.stereo_phase = v,
            _ => {}
        }
    }

    pub fn model(&self) -> NamModelRef<'_> {
        if self.has_custom_model || self.internal_model == 20 {
            if self.custom_model.is_empty() || self.custom_model == "NoModel" {
                NamModelRef::None
            } else if self.custom_model.starts_with('/') {
                NamModelRef::File(Path::new(&self.custom_model))
            } else {
                NamModelRef::Custom(&self.custom_model)
            }
        } else {
            match self.internal_model {
                1..=18 => NamModelRef::Internal(INTERNAL_NAM_MODELS[self.internal_model as usize - 1]),
                _ => NamModelRef::None,
            }
        }
    }
}

/// A cabinet impulse response from a `.wav` file; knob values are 0..1.
#[derive(Debug, Clone)]
pub struct IrState {
    pub file: PathBuf,
    /// -12..+12 dB
    pub level: f32,
    /// 20..400 Hz
    pub low_cut: f32,
    /// 2..20 kHz, 1 = off
    pub high_cut: f32,
}

impl IrState {
    pub fn new(file: PathBuf) -> Self {
        Self { file, level: 0.5, low_cut: 0.0, high_cut: 1.0 }
    }

    pub fn set_index(&mut self, index: usize, v: f32) {
        match index {
            0 => self.level = v,
            1 => self.low_cut = v,
            2 => self.high_cut = v,
            _ => {}
        }
    }
}

#[derive(Debug, Clone)]
pub enum SlotKind {
    Engine(Box<EngineState>),
    Nam(NamState),
    CabIr(IrState),
    /// TFLite neural processing (AI feature, intentionally unsupported).
    Neural { model: String },
    Unknown(u32),
}

#[derive(Debug, Clone)]
pub struct Slot {
    pub kind: SlotKind,
    pub active: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct KnobTarget {
    pub slot: usize,
    pub param: usize,
    pub lo: f32,
    pub hi: f32,
    pub curve: f32,
}

impl KnobTarget {
    /// Value written to the target parameter for a knob position in 0..1.
    pub fn eval(&self, x: f32) -> f32 {
        let x = x.clamp(0.0, 1.0);
        let t = x + self.curve * x * (x - 1.0);
        self.lo + t * (self.hi - self.lo)
    }
}

#[derive(Debug, Clone)]
pub struct Knob {
    pub name: String,
    pub value: f32,
    pub targets: Vec<KnobTarget>,
    /// Two-state control rendered as a switch.
    pub is_button: bool,
    /// Buttons sharing a group act as radio buttons.
    pub button_group: bool,
}

#[derive(Debug, Clone)]
pub struct KnobGroup {
    pub name: String,
    pub first: usize,
    pub last: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceFormat {
    Ezmix3Text,
    LegacyXml(u32),
    /// Made up for a capture file (`.nam` or `.wav`).
    Capture,
}

#[derive(Debug, Clone)]
pub struct Preset {
    pub name: String,
    pub tags: Tags,
    pub output_gain: f32,
    pub slots: Vec<Slot>,
    pub knobs: Vec<Knob>,
    pub groups: Vec<KnobGroup>,
    pub format: SourceFormat,
}

impl Preset {
    pub fn uses_neural(&self) -> bool {
        self.slots.iter().any(|s| matches!(s.kind, SlotKind::Neural { .. }))
    }

    pub fn uses_nam(&self) -> bool {
        self.slots.iter().any(|s| matches!(s.kind, SlotKind::Nam(_)))
    }
}

#[derive(Debug)]
pub struct PresetError(pub String);

impl std::fmt::Display for PresetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for PresetError {}

/// Parse a preset payload of any supported generation.
pub fn parse(name_hint: &str, bytes: &[u8]) -> Result<Preset, PresetError> {
    let start = bytes.iter().position(|b| !b.is_ascii_whitespace()).unwrap_or(0);
    let body = &bytes[start..];
    if body.starts_with(b"Preset {") {
        parse_text(name_hint, body)
    } else if body.starts_with(b"<?xml") || body.starts_with(b"<preset") {
        parse_legacy(name_hint, body)
    } else {
        Err(PresetError("unrecognized preset payload".into()))
    }
}

fn parse_legacy(name_hint: &str, bytes: &[u8]) -> Result<Preset, PresetError> {
    let state = engine::parse(bytes).map_err(|e| PresetError(e.to_string()))?;
    let tag = |k: &str| {
        state.tags.iter().find(|(n, _)| n == k).map(|(_, v)| v.clone()).unwrap_or_default()
    };
    let tags = Tags { kind: tag("Type"), instrument: tag("Instrument"), genre: tag("Genre"), producer: tag("Producer") };
    let info_names = state.info_knob_names();
    let mut knobs = Vec::new();
    for (i, xf) in state.xfaders.iter().enumerate() {
        if xf.interpolations.is_empty() {
            continue;
        }
        let name = if !xf.label.is_empty() {
            xf.label.clone()
        } else {
            info_names.get(knobs.len()).cloned().unwrap_or_else(|| format!("Macro {}", i + 1))
        };
        knobs.push(Knob {
            name,
            value: xf.value,
            targets: vec![KnobTarget { slot: 0, param: i, lo: 0.0, hi: 1.0, curve: 0.0 }],
            is_button: false,
            button_group: false,
        });
    }
    Ok(Preset {
        name: name_hint.to_owned(),
        tags,
        output_gain: 1.0,
        slots: vec![Slot { kind: SlotKind::Engine(Box::new(state.clone())), active: true }],
        knobs,
        groups: Vec::new(),
        format: SourceFormat::LegacyXml(state.format_version),
    })
}

/// A node in the brace-structured text format.
#[derive(Debug, Default)]
struct Node {
    head: String,
    children: Vec<Node>,
    /// Payload line following a `chunk` header.
    data: Option<String>,
}

impl Node {
    fn key(&self) -> &str {
        self.head.split_whitespace().next().unwrap_or("")
    }

    fn args(&self) -> Vec<&str> {
        self.head.split_whitespace().skip(1).collect()
    }

    fn child(&self, key: &str) -> Option<&Node> {
        self.children.iter().find(|c| c.key() == key)
    }

    fn value(&self, key: &str) -> Option<String> {
        self.child(key).map(|c| unquote(c.head[key.len()..].trim()))
    }

    fn num(&self, key: &str) -> Option<f32> {
        self.child(key).and_then(|c| c.args().first().and_then(|v| v.parse().ok()))
    }
}

fn unquote(s: &str) -> String {
    s.strip_prefix('"').and_then(|s| s.strip_suffix('"')).unwrap_or(s).to_owned()
}

fn parse_tree(text: &str) -> Vec<Node> {
    let mut lines = text.lines().map(str::trim).peekable();
    fn block<'a>(lines: &mut std::iter::Peekable<impl Iterator<Item = &'a str>>) -> Vec<Node> {
        let mut out = Vec::new();
        while let Some(l) = lines.next() {
            if l == "}" {
                break;
            }
            if l.is_empty() {
                continue;
            }
            if let Some(head) = l.strip_suffix('{') {
                let children = block(lines);
                out.push(Node { head: head.trim().to_owned(), children, data: None });
            } else if l.starts_with("chunk ") {
                let data = lines.next().map(str::to_owned);
                out.push(Node { head: l.to_owned(), children: Vec::new(), data });
            } else {
                out.push(Node { head: l.to_owned(), children: Vec::new(), data: None });
            }
        }
        out
    }
    block(&mut lines)
}

fn parse_text(name_hint: &str, bytes: &[u8]) -> Result<Preset, PresetError> {
    let text = String::from_utf8_lossy(bytes);
    let tree = parse_tree(&text);
    let root = tree.iter().find(|n| n.key() == "Preset").ok_or_else(|| PresetError("missing Preset block".into()))?;
    let info = root.child("PresetInfo");
    let proc = root.child("PresetProcessor").ok_or_else(|| PresetError("missing PresetProcessor".into()))?;

    let tags = info
        .map(|i| Tags {
            kind: i.value("TagType").unwrap_or_default(),
            instrument: i.value("TagInstrument").unwrap_or_default(),
            genre: i.value("TagGenre").unwrap_or_default(),
            producer: i.value("TagProducer").unwrap_or_default(),
        })
        .unwrap_or_default();
    let name = info.and_then(|i| i.value("Name")).unwrap_or_else(|| name_hint.to_owned());

    let mut slots = Vec::new();
    if let Some(mixer) = proc.child("MixerChannel") {
        let mut fx: Vec<(usize, &Node)> = mixer
            .children
            .iter()
            .filter(|c| c.key() == "effect" && !c.children.is_empty())
            .filter_map(|c| c.args().first().and_then(|i| i.parse().ok()).map(|i| (i, c)))
            .collect();
        fx.sort_by_key(|(i, _)| *i);
        for (_, node) in fx {
            slots.push(parse_slot(node)?);
        }
    }

    let mut knobs = Vec::new();
    let mut groups = Vec::new();
    if let Some(pk) = proc.child("PresetKnobs") {
        for c in &pk.children {
            match c.key() {
                "knob" => knobs.push(parse_knob(c)),
                "knobgroup" => {
                    let r = c.child("range").map(|r| r.args()).unwrap_or_default();
                    let first = r.first().and_then(|v| v.parse().ok()).unwrap_or(0);
                    let last = r.get(1).and_then(|v| v.parse().ok()).unwrap_or(first);
                    groups.push(KnobGroup { name: c.value("name").unwrap_or_default(), first, last });
                }
                _ => {}
            }
        }
    }

    Ok(Preset {
        name,
        tags,
        output_gain: proc.num("OutputGain").unwrap_or(1.0),
        slots,
        knobs,
        groups,
        format: SourceFormat::Ezmix3Text,
    })
}

fn chunk_payload(node: &Node) -> Option<Vec<u8>> {
    let chunk = node.child("chunk")?;
    let data = chunk.data.as_deref()?;
    Some(engine::decode_b64(data.trim()))
}

fn parse_slot(node: &Node) -> Result<Slot, PresetError> {
    let kind_id: u32 = node.num("effect").map(|v| v as u32).unwrap_or(0);
    let active = node.num("active").map(|v| v > 0.5).unwrap_or(true);
    let payload = chunk_payload(node).unwrap_or_default();
    let kind = match kind_id {
        8 => SlotKind::Engine(Box::new(engine::parse(&payload).map_err(|e| PresetError(e.to_string()))?)),
        9 => SlotKind::Nam(parse_nam(&String::from_utf8_lossy(&payload))),
        10 => {
            let text = String::from_utf8_lossy(&payload);
            let model = text
                .lines()
                .find_map(|l| l.strip_prefix("ModelName "))
                .map(unquote)
                .unwrap_or_default();
            SlotKind::Neural { model }
        }
        other => SlotKind::Unknown(other),
    };
    Ok(Slot { kind, active })
}

pub fn parse_nam(text: &str) -> NamState {
    let mut s = NamState::default();
    for line in text.lines() {
        let Some((k, v)) = line.trim().split_once(' ') else { continue };
        let f = || v.trim().parse::<f32>().unwrap_or(0.0);
        match k {
            "HasCustomModel" => s.has_custom_model = f() > 0.5,
            "InternalModel" => s.internal_model = f() as u32,
            "CustomModel" => s.custom_model = unquote(v.trim()),
            "InputLevel" => s.input_level = f(),
            "OutputLevel" => s.output_level = f(),
            "NoiseGateActive" => s.gate_active = f() > 0.5,
            "NoiseThreshold" => s.gate_threshold = f(),
            "EQActive" => s.eq_active = f() > 0.5,
            "EQBass" => s.eq_bass = f(),
            "EQMid" => s.eq_mid = f(),
            "EQTreble" => s.eq_treble = f(),
            "StereoPhase" => s.stereo_phase = f(),
            _ => {}
        }
    }
    s
}

fn parse_knob(node: &Node) -> Knob {
    let mut targets = Vec::new();
    let mut value = None;
    for t in node.children.iter().filter(|c| c.key() == "target") {
        let range = t.child("range").map(|r| r.args()).unwrap_or_default();
        let r = |i: usize, d: f32| range.get(i).and_then(|v| v.parse::<f32>().ok()).unwrap_or(d);
        targets.push(KnobTarget {
            slot: t.num("ix2").map(|v| v as usize).unwrap_or(0),
            param: t.num("ix3").map(|v| v as usize).unwrap_or(0),
            lo: r(0, 0.0),
            hi: r(1, 1.0),
            curve: r(2, 0.0),
        });
        if value.is_none() {
            value = t.num("KnobValue");
        }
    }
    let button = node.child("button");
    Knob {
        name: node.value("name").unwrap_or_default(),
        value: value.unwrap_or(0.0),
        targets,
        is_button: button.is_some_and(|b| b.args().first().is_some_and(|v| *v != "0")),
        button_group: button.is_some_and(|b| b.args().contains(&"group")),
    }
}
