//! Cabinet impulse responses loaded from `.wav` files, with level and
//! low/high cut.

use ezmix_content::preset::IrState;
use ezmix_content::wav::Wav;

use crate::conv::{resample, Convolver};
use crate::dsp::*;

pub struct CabSlot {
    sr: f32,
    state: IrState,
    conv: Option<[Convolver; 2]>,
    low_cut: StereoBiquad,
    high_cut: StereoBiquad,
}

impl CabSlot {
    pub fn new(ir: Option<&Wav>, state: IrState, sr: f32) -> Self {
        // NAM's IR loader level: -18 dB, scaled so the response does not
        // depend on the sample rate the IR is resampled to.
        let gain = db_to_gain(-18.0) * 48000.0 / sr;
        let conv = ir.map(|w| {
            let build = |ch: &[f32]| {
                let mut ir = resample(ch, w.sample_rate, sr);
                ir.iter_mut().for_each(|v| *v *= gain);
                Convolver::new(&ir)
            };
            let left = &w.channels[0];
            [build(left), build(w.channels.get(1).unwrap_or(left))]
        });
        let mut s = Self { sr, state, conv, low_cut: StereoBiquad::default(), high_cut: StereoBiquad::default() };
        s.update_filters();
        s
    }

    pub fn has_ir(&self) -> bool {
        self.conv.is_some()
    }

    fn update_filters(&mut self) {
        let (sr, s) = (self.sr, &self.state);
        let nyquist = 0.45 * sr;
        self.low_cut.set(if s.low_cut <= 0.0 {
            Coefs::IDENTITY
        } else {
            Coefs::rbj(BiquadKind::HighPass, sr, (20.0 * 20f32.powf(s.low_cut)).min(nyquist), 0.707, 0.0)
        });
        self.high_cut.set(if s.high_cut >= 1.0 {
            Coefs::IDENTITY
        } else {
            Coefs::rbj(BiquadKind::LowPass, sr, (2000.0 * 10f32.powf(s.high_cut)).min(nyquist), 0.707, 0.0)
        });
    }

    pub fn set_index(&mut self, index: usize, v: f32) {
        self.state.set_index(index, v);
        if index > 0 {
            self.update_filters();
        }
    }

    pub fn reset(&mut self) {
        if let Some(c) = &mut self.conv {
            c.iter_mut().for_each(Convolver::reset);
        }
        self.low_cut.reset();
        self.high_cut.reset();
    }

    pub fn process(&mut self, l: &mut [f32], r: &mut [f32]) {
        let Some([cl, cr]) = &mut self.conv else { return };
        let g = db_to_gain((self.state.level - 0.5) * 24.0);
        for (a, b) in l.iter_mut().zip(r.iter_mut()) {
            let (x, y) = self.low_cut.process(cl.process(*a), cr.process(*b));
            let (x, y) = self.high_cut.process(x, y);
            *a = x * g;
            *b = y * g;
        }
    }
}
