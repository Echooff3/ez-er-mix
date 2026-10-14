//! Normalized (0..1) <-> engine value mapping for every effect parameter.
//!
//! Macro faders store normalized target positions, so moving a knob needs each
//! parameter's range and taper. The table in `data/param_ranges.json` was
//! derived from the installed presets: for parameters that macro faders drive,
//! range and taper were fitted from (fader position, stored value) pairs; the
//! remainder fall back to unit-based defaults.

use std::collections::HashMap;
use std::sync::OnceLock;

use serde::Deserialize;

#[derive(Debug, Clone, Copy, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Taper {
    Lin,
    Log,
    /// Integer positions (switches, selectors).
    Step,
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub struct Range {
    pub kind: Taper,
    pub lo: f32,
    pub hi: f32,
    #[serde(default)]
    pub default: Option<f32>,
    #[serde(default)]
    pub sync_steps: Option<u32>,
}

impl Range {
    pub fn denormalize(&self, n: f32) -> f32 {
        let n = n.clamp(0.0, 1.0);
        match self.kind {
            Taper::Lin => self.lo + n * (self.hi - self.lo),
            Taper::Log if self.lo > 0.0 && self.hi > 0.0 => self.lo * (self.hi / self.lo).powf(n),
            Taper::Log => self.lo + n * (self.hi - self.lo),
            Taper::Step => (self.lo + n * (self.hi - self.lo)).round(),
        }
    }

    pub fn normalize(&self, v: f32) -> f32 {
        let n = match self.kind {
            Taper::Log if self.lo > 0.0 && self.hi > 0.0 && v > 0.0 => (v / self.lo).ln() / (self.hi / self.lo).ln(),
            _ if self.hi != self.lo => (v - self.lo) / (self.hi - self.lo),
            _ => 0.0,
        };
        n.clamp(0.0, 1.0)
    }
}

type Table = HashMap<String, HashMap<String, Range>>;

fn table() -> &'static Table {
    static TABLE: OnceLock<Table> = OnceLock::new();
    TABLE.get_or_init(|| {
        serde_json::from_str(include_str!("../data/param_ranges.json")).expect("valid embedded range table")
    })
}

pub fn lookup(effect_id: &str, param: &str) -> Option<Range> {
    table().get(effect_id)?.get(param).copied()
}

/// Catalog default for a parameter, if known.
pub fn default_value(effect_id: &str, param: &str) -> Option<f32> {
    lookup(effect_id, param)?.default
}

pub fn denormalize(effect_id: &str, param: &str, n: f32) -> f32 {
    match lookup(effect_id, param) {
        Some(r) => r.denormalize(n),
        None => n,
    }
}

/// Every parameter name known for an effect type.
pub fn param_names(effect_id: &str) -> Vec<&'static str> {
    table().get(effect_id).map(|m| m.keys().map(String::as_str).collect()).unwrap_or_default()
}
