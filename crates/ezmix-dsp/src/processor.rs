//! A complete preset: its chain of slots and macro knobs.

use ezmix_content::preset::{Knob, KnobGroup, Preset, SlotKind};
use ezmix_content::Library;

use crate::cab::CabSlot;
use crate::effect::Ctx;
use crate::engine::{EngineGraph, BLOCK};
use crate::nam::NamSlot;

enum Proc {
    Engine(Box<EngineGraph>),
    Nam(Box<NamSlot>),
    Cab(Box<CabSlot>),
    Bypass,
}

/// Notes about parts of a preset that could not be reproduced.
#[derive(Debug, Clone, Default)]
pub struct LoadReport {
    /// AI "console" stages that were skipped.
    pub skipped_neural: usize,
    /// NAM captures whose model file could not be found.
    pub missing_models: Vec<String>,
    /// Model or IR files that were found but could not be used.
    pub load_errors: Vec<String>,
    /// Engine effect IDs without an implementation.
    pub unsupported: Vec<String>,
}

pub struct PresetProcessor {
    pub name: String,
    pub knobs: Vec<Knob>,
    pub groups: Vec<KnobGroup>,
    pub report: LoadReport,
    slots: Vec<(Proc, bool)>,
    output_gain: f32,
    sr: f32,
}

impl PresetProcessor {
    /// Build every processor for `preset` (allocates; call off the audio
    /// thread).
    pub fn new(preset: &Preset, library: &Library, sr: f32) -> Self {
        let mut report = LoadReport::default();
        let slots = preset
            .slots
            .iter()
            .map(|s| {
                let p = match &s.kind {
                    SlotKind::Engine(state) => {
                        let g = EngineGraph::new(state, sr);
                        for u in &g.unsupported {
                            if !report.unsupported.contains(u) {
                                report.unsupported.push(u.clone());
                            }
                        }
                        Proc::Engine(Box::new(g))
                    }
                    SlotKind::Nam(state) => {
                        let json = library.load_nam_model(&state.model());
                        let slot = NamSlot::new(json.as_deref(), state.clone(), sr);
                        if !slot.has_model() {
                            let name =
                                if state.custom_model.is_empty() { format!("internal #{}", state.internal_model) } else { file_name(&state.custom_model) };
                            match &slot.error {
                                Some(e) => report.load_errors.push(format!("{name}: {e}")),
                                None => report.missing_models.push(name),
                            }
                        }
                        Proc::Nam(Box::new(slot))
                    }
                    SlotKind::CabIr(state) => {
                        let ir = library.load_ir(state).map_err(|e| report.load_errors.push(format!("{}: {e}", file_name(&state.file.to_string_lossy())))).ok();
                        Proc::Cab(Box::new(CabSlot::new(ir.as_ref(), state.clone(), sr)))
                    }
                    SlotKind::Neural { .. } => {
                        report.skipped_neural += 1;
                        Proc::Bypass
                    }
                    SlotKind::Unknown(_) => Proc::Bypass,
                };
                (p, s.active)
            })
            .collect();
        Self {
            name: preset.name.clone(),
            knobs: preset.knobs.clone(),
            groups: preset.groups.clone(),
            report,
            slots,
            output_gain: preset.output_gain,
            sr,
        }
    }

    pub fn sample_rate(&self) -> f32 {
        self.sr
    }

    /// Apply a knob position (0..1) to every target. Real-time safe.
    pub fn set_knob(&mut self, index: usize, value: f32) {
        let Some(knob) = self.knobs.get_mut(index) else { return };
        knob.value = value;
        for t in 0..knob.targets.len() {
            let target = self.knobs[index].targets[t];
            let v = target.eval(value);
            match self.slots.get_mut(target.slot) {
                Some((Proc::Engine(g), _)) => g.set_xfader(target.param, v),
                Some((Proc::Nam(n), _)) => n.set_index(target.param, v),
                Some((Proc::Cab(c), _)) => c.set_index(target.param, v),
                _ => {}
            }
        }
    }

    pub fn reset(&mut self) {
        for (p, _) in &mut self.slots {
            match p {
                Proc::Engine(g) => g.reset(),
                Proc::Nam(n) => n.reset(),
                Proc::Cab(c) => c.reset(),
                Proc::Bypass => {}
            }
        }
    }

    pub fn process(&mut self, bpm: f32, l: &mut [f32], r: &mut [f32]) {
        let ctx = Ctx { sr: self.sr, bpm };
        let n = l.len().min(r.len());
        let mut start = 0;
        while start < n {
            let end = (start + BLOCK).min(n);
            let (bl, br) = (&mut l[start..end], &mut r[start..end]);
            for (p, active) in &mut self.slots {
                if !*active {
                    continue;
                }
                match p {
                    Proc::Engine(g) => g.process_block(&ctx, bl, br),
                    Proc::Nam(s) => s.process(bl, br),
                    Proc::Cab(c) => c.process(bl, br),
                    Proc::Bypass => {}
                }
            }
            for (a, b) in bl.iter_mut().zip(br.iter_mut()) {
                *a = sanitize(*a * self.output_gain);
                *b = sanitize(*b * self.output_gain);
            }
            start = end;
        }
    }
}

/// Last component of a model path, for messages.
fn file_name(path: &str) -> String {
    std::path::Path::new(path).file_name().map_or_else(|| path.to_string(), |n| n.to_string_lossy().into_owned())
}

#[inline]
fn sanitize(x: f32) -> f32 {
    if x.is_finite() { x.clamp(-8.0, 8.0) } else { 0.0 }
}
