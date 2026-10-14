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

/// The selected preset, saved with the host project.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct PresetSelection {
    pub pack_id: String,
    pub path: String,
}

#[derive(Params)]
pub struct EzParams {
    #[persist = "editor-state"]
    editor_state: Arc<EguiState>,

    #[persist = "preset"]
    pub selection: Arc<Mutex<PresetSelection>>,

    #[nested(array, group = "Macros")]
    pub knobs: [KnobParam; NUM_KNOBS],

    #[id = "input"]
    pub input: FloatParam,

    #[id = "output"]
    pub output: FloatParam,

    #[id = "bypass"]
    pub bypass: BoolParam,
}

#[derive(Params)]
pub struct KnobParam {
    #[id = "macro"]
    pub value: FloatParam,
}

impl Default for EzParams {
    fn default() -> Self {
        let gain = |name: &str, min: f32, max: f32| {
            FloatParam::new(name, util::db_to_gain(0.0), FloatRange::Skewed { min: util::db_to_gain(min), max: util::db_to_gain(max), factor: FloatRange::gain_skew_factor(min, max) })
                .with_smoother(SmoothingStyle::Logarithmic(50.0))
                .with_unit(" dB")
                .with_value_to_string(formatters::v2s_f32_gain_to_db(1))
                .with_string_to_value(formatters::s2v_f32_gain_to_db())
        };
        Self {
            editor_state: EguiState::from_size(900, 560),
            selection: Arc::new(Mutex::new(PresetSelection::default())),
            knobs: std::array::from_fn(|i| KnobParam {
                value: FloatParam::new(format!("Macro {}", i + 1), 0.5, FloatRange::Linear { min: 0.0, max: 1.0 })
                    .with_smoother(SmoothingStyle::Linear(40.0))
                    .with_unit(" %")
                    .with_value_to_string(formatters::v2s_f32_percentage(1))
                    .with_string_to_value(formatters::s2v_f32_percentage()),
            }),
            input: gain("Input", -24.0, 24.0),
            output: gain("Output", -36.0, 12.0),
            bypass: BoolParam::new("Bypass", false).make_bypass(),
        }
    }
}

pub struct EzErMix {
    params: Arc<EzParams>,
    shared: Arc<Shared>,
    processor: Option<Box<PresetProcessor>>,
    generation: u64,
    applied: [f32; NUM_KNOBS],
    sample_rate: f32,
    /// Crossfade state for preset switches and bypass (0 = dry, 1 = wet).
    wet: f32,
    meter_decay: f32,
}

impl Default for EzErMix {
    fn default() -> Self {
        Self {
            params: Arc::new(EzParams::default()),
            shared: loader::spawn(),
            processor: None,
            generation: 0,
            applied: [f32::NAN; NUM_KNOBS],
            sample_rate: 48000.0,
            wet: 1.0,
            meter_decay: 0.999,
        }
    }
}

const SUB_BLOCK: usize = 64;

impl Plugin for EzErMix {
    const NAME: &'static str = "EZ-er Mix";
    const VENDOR: &'static str = "ez-er-mix";
    const URL: &'static str = "https://github.com/";
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

        // (Re)build the selected preset for this sample rate. This also runs
        // after the host restores a saved state.
        let sel = self.params.selection.lock().unwrap_or_else(|e| e.into_inner()).clone();
        let rebuild = match &self.processor {
            Some(p) => p.sample_rate() != config.sample_rate,
            None => true,
        };
        let current = self.shared.current.lock().unwrap_or_else(|e| e.into_inner()).as_ref().map(|c| (c.pack_id.clone(), c.path.clone()));
        if !sel.pack_id.is_empty() && (rebuild || current != Some((sel.pack_id.clone(), sel.path.clone()))) {
            self.processor = None;
            self.shared.request_load(&sel.pack_id, &sel.path);
        }
        true
    }

    fn reset(&mut self) {
        if let Some(p) = &mut self.processor {
            p.reset();
        }
    }

    fn process(&mut self, buffer: &mut Buffer, _aux: &mut AuxiliaryBuffers, context: &mut impl ProcessContext<Self>) -> ProcessStatus {
        // Pick up a freshly built preset without blocking.
        if let Ok(mut slot) = self.shared.pending.try_lock() {
            if let Some(ready) = slot.take() {
                if ready.generation >= self.generation {
                    self.generation = ready.generation;
                    let mut p = ready.processor;
                    self.applied = [f32::NAN; NUM_KNOBS];
                    p.reset();
                    if let Some(old) = self.processor.replace(p) {
                        let _ = self.shared.requests.try_send(Request::Drop(old));
                    }
                    self.wet = 0.0;
                } else {
                    let _ = self.shared.requests.try_send(Request::Drop(ready.processor));
                }
            }
        }

        let bpm = context.transport().tempo.unwrap_or(120.0) as f32;
        let n = buffer.samples();
        let channels = buffer.as_slice();
        if channels.len() < 2 {
            return ProcessStatus::Normal;
        }
        let (first, second) = channels.split_at_mut(1);
        let (l, r) = (&mut *first[0], &mut *second[0]);
        let mut in_peak = self.shared.in_peak.load();
        let mut out_peak = self.shared.out_peak.load();
        let fade_step = 1.0 / (0.02 * self.sample_rate);

        let mut start = 0;
        while start < n {
            let end = (start + SUB_BLOCK).min(n);
            let len = end - start;
            let in_gain = self.params.input.smoothed.next_step(len as u32);
            let out_gain = self.params.output.smoothed.next_step(len as u32);
            let target = if self.params.bypass.value() || self.processor.is_none() { 0.0 } else { 1.0 };
            for (i, k) in self.params.knobs.iter().enumerate() {
                let v = k.value.smoothed.next_step(len as u32);
                if let Some(p) = &mut self.processor {
                    if i < p.knobs.len() && (self.applied[i].is_nan() || (v - self.applied[i]).abs() > 1e-5) {
                        p.set_knob(i, v);
                        self.applied[i] = v;
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
            match &mut self.processor {
                Some(p) if target > 0.0 || self.wet > 0.0 => p.process(bpm, bl, br),
                _ => {}
            }
            for i in 0..len {
                self.wet = if target > self.wet { (self.wet + fade_step).min(1.0) } else { (self.wet - fade_step).max(0.0) };
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
        ProcessStatus::Normal
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

nih_export_clap!(EzErMix);
nih_export_vst3!(EzErMix);
