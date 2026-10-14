//! EZ-er Mix: plays installed EZmix preset packs with a simplified interface.

use std::sync::{Arc, Mutex};

use nih_plug::prelude::*;
use nih_plug_egui::EguiState;
use serde::{Deserialize, Serialize};

use ezmix_dsp::PresetProcessor;

mod editor;
pub mod loader;
mod widgets;

use loader::{Request, Shared};

pub const NUM_KNOBS: usize = 8;
/// Rack slots. Each has its own power switch and macro parameters.
pub const MAX_SLOTS: usize = 8;

/// A preset reference, saved with the host project. Empty = no preset.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct PresetSelection {
    pub pack_id: String,
    pub path: String,
}

impl PresetSelection {
    pub fn is_empty(&self) -> bool {
        self.pack_id.is_empty()
    }
}

/// One tab of the rack. `channel` picks the slot's host parameters and stays
/// with the effect when tabs are reordered, so automation follows it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RackSlot {
    pub channel: usize,
    #[serde(default)]
    pub preset: PresetSelection,
}

/// The rack in processing order, saved with the host project.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct RackState {
    pub slots: Vec<RackSlot>,
    pub active: usize,
}

impl Default for RackState {
    fn default() -> Self {
        Self { slots: vec![RackSlot { channel: 0, preset: PresetSelection::default() }], active: 0 }
    }
}

impl RackState {
    /// Drops invalid or duplicate channels (e.g. from a hand-edited state).
    pub fn sanitize(&mut self) {
        let mut seen = [false; MAX_SLOTS];
        self.slots.retain(|s| s.channel < MAX_SLOTS && !std::mem::replace(&mut seen[s.channel], true));
        if self.slots.is_empty() {
            *self = Self::default();
        }
        self.active = self.active.min(self.slots.len() - 1);
    }

    pub fn free_channel(&self) -> Option<usize> {
        (0..MAX_SLOTS).find(|c| self.slots.iter().all(|s| s.channel != *c))
    }

    pub fn active_slot(&self) -> &RackSlot {
        &self.slots[self.active.min(self.slots.len() - 1)]
    }

    pub fn channels(&self) -> impl Iterator<Item = usize> + '_ {
        self.slots.iter().map(|s| s.channel)
    }
}

#[derive(Params)]
pub struct EzParams {
    #[persist = "editor-state"]
    editor_state: Arc<EguiState>,

    #[persist = "rack"]
    pub rack: Arc<Mutex<RackState>>,

    #[nested(array, group = "Slot")]
    pub slots: [SlotParams; MAX_SLOTS],

    #[id = "input"]
    pub input: FloatParam,

    #[id = "output"]
    pub output: FloatParam,

    #[id = "bypass"]
    pub bypass: BoolParam,
}

#[derive(Params)]
pub struct SlotParams {
    #[id = "on"]
    pub enabled: BoolParam,

    #[nested(array)]
    pub knobs: [KnobParam; NUM_KNOBS],
}

#[derive(Params)]
pub struct KnobParam {
    #[id = "macro"]
    pub value: FloatParam,
}

impl Default for EzParams {
    fn default() -> Self {
        let gain = |name: &str, min: f32, max: f32| {
            FloatParam::new(
                name,
                util::db_to_gain(0.0),
                FloatRange::Skewed { min: util::db_to_gain(min), max: util::db_to_gain(max), factor: FloatRange::gain_skew_factor(min, max) },
            )
            .with_smoother(SmoothingStyle::Logarithmic(50.0))
            .with_unit(" dB")
            .with_value_to_string(formatters::v2s_f32_gain_to_db(1))
            .with_string_to_value(formatters::s2v_f32_gain_to_db())
        };
        Self {
            editor_state: EguiState::from_size(940, 580),
            rack: Arc::new(Mutex::new(RackState::default())),
            slots: std::array::from_fn(|s| SlotParams {
                enabled: BoolParam::new(format!("Slot {} On", s + 1), true),
                knobs: std::array::from_fn(|i| KnobParam {
                    value: FloatParam::new(format!("Slot {} Macro {}", s + 1, i + 1), 0.5, FloatRange::Linear { min: 0.0, max: 1.0 })
                        .with_smoother(SmoothingStyle::Linear(40.0))
                        .with_unit(" %")
                        .with_value_to_string(formatters::v2s_f32_percentage(1))
                        .with_string_to_value(formatters::s2v_f32_percentage()),
                }),
            }),
            input: gain("Input", -24.0, 24.0),
            output: gain("Output", -36.0, 12.0),
            bypass: BoolParam::new("Bypass", false).make_bypass(),
        }
    }
}

/// Audio-thread state of one rack slot.
struct Slot {
    processor: Option<Box<PresetProcessor>>,
    /// A replacement (or `Some(None)` to empty the slot) that is swapped in
    /// once the current processor has faded out.
    incoming: Option<Option<Box<PresetProcessor>>>,
    generation: u64,
    applied: [f32; NUM_KNOBS],
    /// Crossfade between this slot's input (0) and its output (1).
    wet: f32,
}

impl Slot {
    fn new() -> Self {
        Self { processor: None, incoming: None, generation: 0, applied: [f32::NAN; NUM_KNOBS], wet: 0.0 }
    }
}

pub struct EzErMix {
    params: Arc<EzParams>,
    shared: Arc<Shared>,
    slots: [Slot; MAX_SLOTS],
    /// Rack order in use, and the one waiting for the rack to fade out.
    order: ([usize; MAX_SLOTS], usize),
    order_word: u64,
    sample_rate: f32,
    /// Crossfade for bypass and reordering (0 = dry, 1 = wet).
    wet: f32,
    meter_decay: f32,
}

impl Default for EzErMix {
    fn default() -> Self {
        let shared = loader::spawn();
        let order_word = shared.order.load(std::sync::atomic::Ordering::Acquire);
        Self {
            params: Arc::new(EzParams::default()),
            shared,
            slots: std::array::from_fn(|_| Slot::new()),
            order: loader::unpack_order(order_word),
            order_word,
            sample_rate: 48000.0,
            wet: 1.0,
            meter_decay: 0.999,
        }
    }
}

const SUB_BLOCK: usize = 64;

/// One step of a linear crossfade towards `target`, stopping there.
#[inline]
fn fade(current: f32, target: f32, step: f32) -> f32 {
    if current < target {
        (current + step).min(target)
    } else {
        (current - step).max(target)
    }
}

impl EzErMix {
    fn release(&self, p: Box<PresetProcessor>) {
        // Freed on the loader thread; only dropped here if its queue is full.
        let _ = self.shared.requests.try_send(Request::Drop(p));
    }

    /// Take finished builds from the loader without blocking.
    fn pick_up_pending(&mut self) {
        if !self.shared.has_pending.swap(false, std::sync::atomic::Ordering::AcqRel) {
            return;
        }
        let mut retry = false;
        for ch in 0..MAX_SLOTS {
            let ready = match self.shared.slots[ch].pending.try_lock() {
                Ok(mut pending) => pending.take(),
                Err(_) => {
                    retry = true;
                    None
                }
            };
            let Some(ready) = ready else { continue };
            let slot = &mut self.slots[ch];
            if ready.generation < slot.generation {
                if let Some(p) = ready.processor {
                    self.release(p);
                }
                continue;
            }
            slot.generation = ready.generation;
            if let Some(Some(stale)) = slot.incoming.replace(ready.processor) {
                self.release(stale);
            }
        }
        if retry {
            self.shared.has_pending.store(true, std::sync::atomic::Ordering::Release);
        }
    }

    /// Swap in replacements for slots that have faded out (or are silent).
    fn swap_incoming(&mut self, ch: usize, audible: bool) {
        let slot = &mut self.slots[ch];
        if slot.incoming.is_none() || (audible && slot.wet > 0.0 && slot.processor.is_some()) {
            return;
        }
        let mut next = slot.incoming.take().flatten();
        if let Some(p) = &mut next {
            p.reset();
        }
        slot.applied = [f32::NAN; NUM_KNOBS];
        slot.wet = 0.0;
        let old = std::mem::replace(&mut slot.processor, next);
        if let Some(old) = old {
            self.release(old);
        }
    }

    /// The channels in `order` that currently produce sound.
    fn effective(&self, order: &([usize; MAX_SLOTS], usize)) -> ([usize; MAX_SLOTS], usize) {
        let mut out = [0; MAX_SLOTS];
        let mut n = 0;
        for &ch in &order.0[..order.1] {
            if self.slots[ch].processor.is_some() {
                out[n] = ch;
                n += 1;
            }
        }
        (out, n)
    }
}

impl Plugin for EzErMix {
    const NAME: &'static str = "EZ-er Mix";
    const VENDOR: &'static str = "ez-er-mix";
    const URL: &'static str = "https://github.com/Echooff3/ez-er-mix";
    const EMAIL: &'static str = "john.skibicki.iii@gmail.com";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    // Stereo only: presets are stereo processors, and hosts handle mono
    // tracks by feeding both inputs.
    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] =
        &[AudioIOLayout { main_input_channels: NonZeroU32::new(2), main_output_channels: NonZeroU32::new(2), ..AudioIOLayout::const_default() }];

    const SAMPLE_ACCURATE_AUTOMATION: bool = false;

    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        editor::create(self.params.clone(), self.shared.clone())
    }

    fn initialize(&mut self, _layout: &AudioIOLayout, config: &BufferConfig, _context: &mut impl InitContext<Self>) -> bool {
        self.sample_rate = config.sample_rate;
        *self.shared.sample_rate.lock().unwrap_or_else(|e| e.into_inner()) = config.sample_rate;
        self.meter_decay = 0.25f32.powf(1.0 / (config.sample_rate * 0.15));

        // Not processing now, so finished builds can be installed directly.
        self.pick_up_pending();
        for ch in 0..MAX_SLOTS {
            self.swap_incoming(ch, false);
        }

        // (Re)build the rack for this sample rate. This also runs after the
        // host restores a saved state.
        let rack = {
            let mut rack = self.params.rack.lock().unwrap_or_else(|e| e.into_inner());
            rack.sanitize();
            rack.clone()
        };
        self.shared.publish_order(rack.channels());
        for ch in 0..MAX_SLOTS {
            let wanted = rack.slots.iter().find(|s| s.channel == ch).map(|s| &s.preset).filter(|p| !p.is_empty());
            let slot = &self.slots[ch];
            let latest = self.shared.slots[ch].generation.load(std::sync::atomic::Ordering::SeqCst);
            let info = self.shared.info(ch);
            match wanted {
                Some(sel) => {
                    let current = slot.processor.as_ref().is_some_and(|p| p.sample_rate() == config.sample_rate)
                        && slot.generation == latest
                        && info.is_some_and(|i| i.pack_id == sel.pack_id && i.path == sel.path);
                    if !current {
                        self.shared.request_load(ch, &sel.pack_id, &sel.path);
                    }
                }
                None if slot.processor.is_some() || info.is_some() => self.shared.request_clear(ch),
                None => {}
            }
        }
        true
    }

    fn reset(&mut self) {
        for slot in &mut self.slots {
            if let Some(p) = &mut slot.processor {
                p.reset();
            }
        }
    }

    fn process(&mut self, buffer: &mut Buffer, _aux: &mut AuxiliaryBuffers, context: &mut impl ProcessContext<Self>) -> ProcessStatus {
        let bpm = context.transport().tempo.unwrap_or(120.0) as f32;
        let channels = buffer.as_slice();
        if channels.len() < 2 {
            return ProcessStatus::Normal;
        }
        let (first, second) = channels.split_at_mut(1);
        self.process_stereo(bpm, &mut *first[0], &mut *second[0]);
        ProcessStatus::Normal
    }
}

impl EzErMix {
    /// Runs the rack over one host buffer.
    fn process_stereo(&mut self, bpm: f32, l: &mut [f32], r: &mut [f32]) {
        self.pick_up_pending();
        let n = l.len().min(r.len());
        let mut in_peak = self.shared.in_peak.load();
        let mut out_peak = self.shared.out_peak.load();
        let fade_step = 1.0 / (0.02 * self.sample_rate);

        let mut start = 0;
        while start < n {
            let end = (start + SUB_BLOCK).min(n);
            let len = end - start;

            // A new rack order is adopted directly when it does not change
            // what is heard, otherwise once the rack has faded to dry.
            let word = self.shared.order.load(std::sync::atomic::Ordering::Acquire);
            let mut reordering = false;
            if word != self.order_word {
                let next = loader::unpack_order(word);
                if self.wet <= 0.0 || self.effective(&next) == self.effective(&self.order) {
                    self.order = next;
                    self.order_word = word;
                } else {
                    reordering = true;
                }
            }

            let in_gain = self.params.input.smoothed.next_step(len as u32);
            let out_gain = self.params.output.smoothed.next_step(len as u32);
            let (order, count) = self.order;
            for ch in 0..MAX_SLOTS {
                let audible = self.wet > 0.0 && order[..count].contains(&ch);
                if !audible {
                    self.slots[ch].wet = 0.0;
                }
                self.swap_incoming(ch, audible);
                let slot = &mut self.slots[ch];
                for (i, k) in self.params.slots[ch].knobs.iter().enumerate() {
                    let v = k.value.smoothed.next_step(len as u32);
                    if let Some(p) = &mut slot.processor {
                        if i < p.knobs.len() && (slot.applied[i].is_nan() || (v - slot.applied[i]).abs() > 1e-5) {
                            p.set_knob(i, v);
                            slot.applied[i] = v;
                        }
                    }
                }
            }

            let (bl, br) = (&mut l[start..end], &mut r[start..end]);
            let mut dry = [[0.0f32; SUB_BLOCK]; 2];
            for i in 0..len {
                bl[i] *= in_gain;
                br[i] *= in_gain;
                dry[0][i] = bl[i];
                dry[1][i] = br[i];
                in_peak = in_peak.max(bl[i].abs()).max(br[i].abs());
            }

            let rack_target = if self.params.bypass.value() || reordering { 0.0 } else { 1.0 };
            if rack_target > 0.0 || self.wet > 0.0 {
                let mut slot_dry = [[0.0f32; SUB_BLOCK]; 2];
                for &ch in &order[..count] {
                    let enabled = self.params.slots[ch].enabled.value();
                    let slot = &mut self.slots[ch];
                    let Some(p) = &mut slot.processor else {
                        continue;
                    };
                    let target = if enabled && slot.incoming.is_none() { 1.0 } else { 0.0 };
                    if target == 0.0 && slot.wet <= 0.0 {
                        continue;
                    }
                    slot_dry[0][..len].copy_from_slice(bl);
                    slot_dry[1][..len].copy_from_slice(br);
                    p.process(bpm, bl, br);
                    for i in 0..len {
                        slot.wet = fade(slot.wet, target, fade_step);
                        let w = slot.wet;
                        bl[i] = slot_dry[0][i] * (1.0 - w) + bl[i] * w;
                        br[i] = slot_dry[1][i] * (1.0 - w) + br[i] * w;
                    }
                }
            }

            for i in 0..len {
                self.wet = fade(self.wet, rack_target, fade_step);
                let w = self.wet;
                bl[i] = (dry[0][i] * (1.0 - w) + bl[i] * w) * out_gain;
                br[i] = (dry[1][i] * (1.0 - w) + br[i] * w) * out_gain;
                out_peak = out_peak.max(bl[i].abs()).max(br[i].abs());
            }
            start = end;
        }
        let decay = self.meter_decay.powi(n as i32);
        self.shared.in_peak.store(in_peak * decay);
        self.shared.out_peak.store(out_peak * decay);
    }
}

impl ClapPlugin for EzErMix {
    const CLAP_ID: &'static str = "com.ez-er-mix.ez-er-mix";
    const CLAP_DESCRIPTION: Option<&'static str> = Some("Plays installed EZmix preset packs with a simplified interface");
    const CLAP_MANUAL_URL: Option<&'static str> = None;
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[ClapFeature::AudioEffect, ClapFeature::Stereo, ClapFeature::Mixing, ClapFeature::Mastering];
}

impl Vst3Plugin for EzErMix {
    const VST3_CLASS_ID: [u8; 16] = *b"EzErMixRustPlug1";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] = &[Vst3SubCategory::Fx, Vst3SubCategory::Mastering, Vst3SubCategory::Stereo];
}

#[cfg(test)]
mod tests;

nih_export_clap!(EzErMix);
nih_export_vst3!(EzErMix);
