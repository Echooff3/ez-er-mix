//! The multi-channel effect "engine" state stored as XML inside presets.
//!
//! Three generations exist:
//! * format 1 (EZmix 1 era packs): IN / AUX / OUT channels, routing stored as
//!   channel attributes plus a single `<aux_send>`.
//! * formats 4 and 6 (EZmix 2 era): IN, AUX1, AUX2, OUT channels, routing in a
//!   `TA_GLOBAL` pseudo-effect.
//! * format 8 (EZmix 3): IN, AUX1, AUX2, OUT, AUX3, AUX4, routing in `TA_GLOBAL`.
//!
//! Every format carries up to nine `<xfader>` macro controls, each
//! interpolating any number of effect parameters.

use base64::Engine as _;

/// Index of the output chain inside [`EngineState::channels`].
pub const CH_IN: usize = 0;
pub const CH_AUX1: usize = 1;
pub const CH_AUX2: usize = 2;
pub const CH_OUT: usize = 3;
pub const CH_AUX3: usize = 4;
pub const CH_AUX4: usize = 5;

/// Order of `-200;k;level` send targets (format 8) and of the
/// `auxsend,<src>_<dst>,…` global parameters.
pub const SEND_NAMES: [&str; 10] = [
    "in_aux1", "in_aux2", "aux1_aux2", "in_aux3", "in_aux4", "aux1_aux3", "aux1_aux4", "aux2_aux3",
    "aux2_aux4", "aux3_aux4",
];

/// `ch,<name>,…` names of the mixable channels in TA_GLOBAL.
pub const MIX_CHANNEL_NAMES: [(&str, usize); 5] =
    [("in", CH_IN), ("aux1", CH_AUX1), ("aux2", CH_AUX2), ("aux3", CH_AUX3), ("aux4", CH_AUX4)];

#[derive(Debug, Clone)]
pub enum ParamValue {
    Num(f32),
    /// A tempo-syncable parameter.
    Sync { unsynced: f32, synced: f32, sync: bool },
    Data(Vec<u8>),
    Str(String),
}

impl ParamValue {
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            ParamValue::Num(v) => Some(*v),
            ParamValue::Sync { unsynced, .. } => Some(*unsynced),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EffectState {
    pub id: String,
    pub power: bool,
    pub params: Vec<(String, ParamValue)>,
}

impl EffectState {
    pub fn param(&self, name: &str) -> Option<&ParamValue> {
        self.params.iter().find(|(n, _)| n == name).map(|(_, v)| v)
    }

    pub fn num(&self, name: &str) -> Option<f32> {
        self.param(name).and_then(ParamValue::as_f32)
    }

    pub fn is_dummy(&self) -> bool {
        self.id == "TA_DUMMY" || self.id.is_empty()
    }
}

#[derive(Debug, Clone, Default)]
pub struct ChannelState {
    pub slots: Vec<EffectState>,
}

#[derive(Debug, Clone, Copy)]
pub struct MixChannel {
    pub volume_db: f32,
    /// -1 (left) .. 1 (right)
    pub pan: f32,
    pub mute: bool,
    pub solo: bool,
    pub phase_invert: bool,
}

impl Default for MixChannel {
    fn default() -> Self {
        Self { volume_db: 0.0, pan: 0.0, mute: false, solo: false, phase_invert: false }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Send {
    pub from: usize,
    pub to: usize,
    pub level_db: f32,
    /// Tap position: the send is taken before slot `split` of the source
    /// chain; values >= the slot count tap the end of the chain.
    pub split: usize,
}

#[derive(Debug, Clone)]
pub struct Routing {
    pub input_db: f32,
    /// Indexed by channel index (CH_*). The OUT entry is unused.
    pub mix: [MixChannel; 6],
    pub sends: Vec<Send>,
    pub output_db: f32,
    pub output_pan: f32,
    pub output_bypass: bool,
    pub multiband: Option<[f32; 3]>,
}

impl Default for Routing {
    fn default() -> Self {
        Self {
            input_db: 0.0,
            mix: [MixChannel::default(); 6],
            sends: Vec::new(),
            output_db: 0.0,
            output_pan: 0.0,
            output_bypass: false,
            multiband: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Target {
    Effect { channel: usize, slot: usize, param: String },
    ChannelVolume(usize),
    Send(usize),
    Unknown(String),
}

#[derive(Debug, Clone)]
pub struct Interpolation {
    pub target: Target,
    pub label: String,
    pub min: f32,
    pub max: f32,
    pub curve: f32,
}

impl Interpolation {
    /// Normalized target position for a fader value in 0..1.
    pub fn eval(&self, x: f32) -> f32 {
        let x = x.clamp(0.0, 1.0);
        let t = if self.curve == 0.0 { x } else { x.powf(10f32.powf(self.curve)) };
        self.min + t * (self.max - self.min)
    }
}

#[derive(Debug, Clone)]
pub struct Xfader {
    pub value: f32,
    pub label: String,
    pub interpolations: Vec<Interpolation>,
}

#[derive(Debug, Clone)]
pub struct EngineState {
    pub format_version: u32,
    pub information: String,
    /// Always six entries, indexed by CH_* constants.
    pub channels: Vec<ChannelState>,
    pub routing: Routing,
    pub xfaders: Vec<Xfader>,
    pub tags: Vec<(String, String)>,
}

impl EngineState {
    pub fn effect(&self, channel: usize, slot: usize) -> Option<&EffectState> {
        self.channels.get(channel)?.slots.get(slot)
    }

    /// The "Shape:" / "Blend:" knob names from the information string used by
    /// older presets.
    pub fn info_knob_names(&self) -> Vec<String> {
        self.information
            .lines()
            .filter_map(|l| l.split_once(':'))
            .filter(|(k, _)| matches!(k.trim(), "Shape" | "Blend"))
            .map(|(_, v)| v.trim().to_owned())
            .collect()
    }
}

#[derive(Debug)]
pub struct ParseError(pub String);

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ParseError {}

fn attr_f32(n: roxmltree::Node, name: &str) -> Option<f32> {
    n.attribute(name).and_then(|v| v.trim().parse().ok())
}

fn attr_bool(n: roxmltree::Node, name: &str) -> Option<bool> {
    n.attribute(name).map(|v| v.trim() != "0")
}

pub fn parse(xml: &[u8]) -> Result<EngineState, ParseError> {
    let text = std::str::from_utf8(xml)
        .map(str::to_owned)
        .unwrap_or_else(|_| xml.iter().map(|&b| b as char).collect());
    let text = text.trim_start_matches('\u{feff}');
    let doc = roxmltree::Document::parse(text).map_err(|e| ParseError(e.to_string()))?;
    let preset = doc
        .descendants()
        .find(|n| n.has_tag_name("preset"))
        .ok_or_else(|| ParseError("no <preset> element".into()))?;
    let format_version: u32 =
        preset.attribute("format_version").and_then(|v| v.parse().ok()).unwrap_or(1);

    let mut state = EngineState {
        format_version,
        information: preset.attribute("information").unwrap_or_default().to_owned(),
        channels: vec![ChannelState::default(); 6],
        routing: Routing::default(),
        xfaders: Vec::new(),
        tags: Vec::new(),
    };

    let channel_order: &[usize] = match format_version {
        1 => &[CH_IN, CH_AUX1, CH_OUT],
        2..=7 => &[CH_IN, CH_AUX1, CH_AUX2, CH_OUT],
        _ => &[CH_IN, CH_AUX1, CH_AUX2, CH_OUT, CH_AUX3, CH_AUX4],
    };

    let mut global: Option<EffectState> = None;
    let mut output: Option<EffectState> = None;

    for child in preset.children().filter(|n| n.is_element()) {
        match child.tag_name().name() {
            "channels" => {
                for (i, ch) in child.children().filter(|n| n.has_tag_name("channel")).enumerate() {
                    let idx = ch
                        .attribute("engine_index")
                        .and_then(|v| v.parse::<usize>().ok())
                        .unwrap_or(i);
                    let Some(&target) = channel_order.get(idx) else { continue };
                    let slots = ch
                        .children()
                        .find(|n| n.has_tag_name("slots"))
                        .map(|s| s.children().filter(|n| n.has_tag_name("effect")).map(parse_effect).collect())
                        .unwrap_or_default();
                    state.channels[target].slots = slots;
                    if format_version == 1 {
                        let m = &mut state.routing.mix[target];
                        m.volume_db = attr_f32(ch, "volume_user_value").unwrap_or(0.0);
                        m.pan = normalize_pan(attr_f32(ch, "pan_user_value").unwrap_or(0.0));
                        m.mute = attr_bool(ch, "muted").unwrap_or(false)
                            || attr_bool(ch, "bypassed").unwrap_or(false);
                        m.solo = attr_bool(ch, "solo").unwrap_or(false);
                        m.phase_invert = attr_bool(ch, "phase_inverted").unwrap_or(false);
                    }
                }
            }
            "aux_send" => {
                let level = attr_f32(child, "user_value").unwrap_or(-1000.0);
                let split = attr_f32(child, "aux_split_index").map(|v| v as usize).unwrap_or(usize::MAX);
                let post = attr_bool(child, "send_post").unwrap_or(false);
                state.routing.sends.push(Send {
                    from: CH_IN,
                    to: CH_AUX1,
                    level_db: level,
                    split: if post { usize::MAX } else { split },
                });
            }
            "global" => global = child.children().find(|n| n.has_tag_name("effect")).map(parse_effect),
            "out" => {
                for e in child.descendants().filter(|n| n.has_tag_name("effect")) {
                    let e = parse_effect(e);
                    if e.id == "TA_OUTPUT" {
                        output = Some(e);
                    }
                }
            }
            "reserved_tags" | "user_tags" => {
                for t in child.children().filter(|n| n.has_tag_name("tag")) {
                    state.tags.push((
                        t.attribute("name").unwrap_or_default().to_owned(),
                        t.attribute("value").unwrap_or_default().to_owned(),
                    ));
                }
            }
            "xfader" | "auxxfader" => {
                if child.tag_name().name() == "auxxfader" && attr_bool(child, "enabled") == Some(false) {
                    continue;
                }
                state.xfaders.push(Xfader {
                    value: attr_f32(child, "value").unwrap_or(0.0),
                    label: child.attribute("label").unwrap_or_default().to_owned(),
                    interpolations: child
                        .children()
                        .filter(|n| n.has_tag_name("interpolation"))
                        .map(|n| Interpolation {
                            target: parse_target(n.attribute("param").or(n.attribute("name")).unwrap_or_default()),
                            label: n.attribute("label").unwrap_or_default().to_owned(),
                            min: attr_f32(n, "min").unwrap_or(0.0),
                            max: attr_f32(n, "max").unwrap_or(1.0),
                            curve: attr_f32(n, "curve").unwrap_or(0.0),
                        })
                        .collect(),
                });
            }
            _ => {}
        }
    }

    if let Some(g) = &global {
        apply_global(&mut state.routing, g);
    }
    if let Some(o) = &output {
        state.routing.output_db = o.num("ch,out,volume").unwrap_or(0.0);
        state.routing.output_pan = normalize_pan(o.num("ch,out,pan").unwrap_or(0.0));
        state.routing.output_bypass = o.num("ch,out,bypass").unwrap_or(0.0) > 0.5;
    }
    Ok(state)
}

fn normalize_pan(p: f32) -> f32 {
    if p.abs() > 1.0 { (p / 100.0).clamp(-1.0, 1.0) } else { p }
}

fn apply_global(r: &mut Routing, g: &EffectState) {
    r.input_db = g.num("in_lev").unwrap_or(0.0);
    for (name, ch) in MIX_CHANNEL_NAMES {
        let m = &mut r.mix[ch];
        m.volume_db = g.num(&format!("ch,{name},volume")).unwrap_or(0.0);
        m.pan = normalize_pan(g.num(&format!("ch,{name},pan")).unwrap_or(0.0));
        m.mute = g.num(&format!("ch,{name},mute")).unwrap_or(0.0) > 0.5;
        m.solo = g.num(&format!("ch,{name},solo")).unwrap_or(0.0) > 0.5;
        m.phase_invert = g.num(&format!("ch,{name},phase")).unwrap_or(0.0) > 0.5;
    }
    for name in SEND_NAMES {
        let Some(level) = g.num(&format!("auxsend,{name},level")) else { continue };
        let (from, to) = send_endpoints(name);
        let split = g.num(&format!("auxsend,{name},split")).unwrap_or(1000.0);
        r.sends.push(Send { from, to, level_db: level, split: split.max(0.0) as usize });
    }
    if g.num("multiband_enable").unwrap_or(0.0) > 0.5 {
        r.multiband = Some([
            g.num("xover,0,freq").unwrap_or(120.0),
            g.num("xover,1,freq").unwrap_or(800.0),
            g.num("xover,2,freq").unwrap_or(5000.0),
        ]);
    }
}

pub fn send_endpoints(name: &str) -> (usize, usize) {
    let ch = |s: &str| match s {
        "in" => CH_IN,
        "aux1" => CH_AUX1,
        "aux2" => CH_AUX2,
        "aux3" => CH_AUX3,
        _ => CH_AUX4,
    };
    let (a, b) = name.split_once('_').unwrap_or(("in", "aux1"));
    (ch(a), ch(b))
}

fn parse_effect(node: roxmltree::Node) -> EffectState {
    let mut power = attr_bool(node, "power").unwrap_or(true) && !attr_bool(node, "bypassed").unwrap_or(false);
    let mut params = Vec::new();
    for p in node.descendants().filter(|n| n.has_tag_name("parameter")) {
        let Some(name) = p.attribute("name") else { continue };
        let value = if let Some(v) = attr_f32(p, "user_value") {
            ParamValue::Num(v)
        } else if let Some(u) = attr_f32(p, "unsynced_user_value") {
            ParamValue::Sync {
                unsynced: u,
                synced: attr_f32(p, "synced_user_value").unwrap_or(0.0),
                sync: attr_f32(p, "sync_user_value").unwrap_or(0.0) > 0.5,
            }
        } else if let Some(d) = p.attribute("data") {
            let cleaned: String = d.chars().filter(|c| !c.is_whitespace()).collect();
            ParamValue::Data(decode_b64(&cleaned))
        } else if let Some(s) = p.attribute("value") {
            ParamValue::Str(s.to_owned())
        } else {
            continue;
        };
        if name == "power" {
            if let Some(v) = value.as_f32() {
                power &= v > 0.5;
            }
            continue;
        }
        params.push((name.to_owned(), value));
    }
    EffectState { id: node.attribute("id").unwrap_or_default().to_owned(), power, params }
}

pub fn decode_b64(s: &str) -> Vec<u8> {
    let trimmed = s.trim_end_matches('=');
    base64::engine::general_purpose::STANDARD_NO_PAD
        .decode(trimmed)
        .or_else(|_| base64::engine::general_purpose::STANDARD.decode(s))
        .unwrap_or_default()
}

pub fn parse_target(s: &str) -> Target {
    // Format 1: slot(<channel>-<slot>)<param>, channelVolume-N, channelAuxSend-N
    if let Some(rest) = s.strip_prefix("slot(") {
        if let Some((nums, param)) = rest.split_once(')') {
            if let Some((c, sl)) = nums.split_once('-') {
                if let (Ok(c), Ok(sl)) = (c.parse::<usize>(), sl.parse::<usize>()) {
                    let channel = [CH_IN, CH_AUX1, CH_OUT].get(c).copied().unwrap_or(CH_IN);
                    return Target::Effect { channel, slot: sl, param: param.to_owned() };
                }
            }
        }
        return Target::Unknown(s.to_owned());
    }
    if let Some(n) = s.strip_prefix("channelVolume-") {
        return match n.parse::<usize>() {
            Ok(c) => Target::ChannelVolume([CH_IN, CH_AUX1, CH_OUT].get(c).copied().unwrap_or(CH_IN)),
            Err(_) => Target::Unknown(s.to_owned()),
        };
    }
    if s.starts_with("channelAuxSend-") {
        return Target::Send(0);
    }
    let f: Vec<&str> = s.split(';').collect();
    match f.as_slice() {
        // Formats 4/6: e;1;<channel>;<slot>;<param>
        ["e", _, c, sl, param] => match (c.parse::<usize>(), sl.parse::<usize>()) {
            (Ok(c), Ok(sl)) => Target::Effect { channel: c, slot: sl, param: (*param).to_owned() },
            _ => Target::Unknown(s.to_owned()),
        },
        // Formats 4/6: c;0;<channel> channel volume
        ["c", _, c] => match c.parse::<usize>() {
            Ok(c) => Target::ChannelVolume(c),
            Err(_) => Target::Unknown(s.to_owned()),
        },
        // Format 8: -200;<send index>;level
        ["-200", k, "level"] => match k.parse::<usize>() {
            Ok(k) if k < SEND_NAMES.len() => Target::Send(k),
            _ => Target::Unknown(s.to_owned()),
        },
        // Format 8: <channel>;<slot>;<param>
        [c, sl, param] => match (c.parse::<usize>(), sl.parse::<usize>()) {
            (Ok(c), Ok(sl)) => Target::Effect { channel: c, slot: sl, param: (*param).to_owned() },
            _ => Target::Unknown(s.to_owned()),
        },
        _ => Target::Unknown(s.to_owned()),
    }
}

/// dB value used by the engine for "off".
pub const MINUS_INF_DB: f32 = -1000.0;

/// Send levels: linear -40..0 dB, the bottom of the range is silence.
pub fn send_db_from_norm(n: f32) -> f32 {
    if n <= 0.0 { MINUS_INF_DB } else { -40.0 + 40.0 * n.min(1.0) }
}

/// Channel volumes: linear -36..+12 dB, the bottom of the range is silence.
pub fn channel_db_from_norm(n: f32) -> f32 {
    if n <= 0.0 { MINUS_INF_DB } else { -36.0 + 48.0 * n.min(1.0) }
}

pub fn db_to_gain(db: f32) -> f32 {
    if db <= -200.0 { 0.0 } else { 10f32.powf(db / 20.0) }
}
