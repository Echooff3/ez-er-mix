//! Effects driven by data embedded in presets: cabinet IRs, reverb IRs and
//! matching-EQ profiles.

use super::reverbs::{Fdn, FdnParams};
use crate::conv::{min_phase_fir, resample, Convolver};
use crate::dsp::*;
use crate::effect::{on, Ctx, Effect};

/// Sample rate of the impulse responses and spectra stored in presets.
pub const CONTENT_RATE: f32 = 44100.0;

// ---------------------------------------------------------------------------
// Cabinet IR (TA_CABIR): raw little-endian f32 mono IR.

pub struct CabIr {
    sr: f32,
    raw: Vec<f32>,
    level: f32,
    delay_ms: f32,
    conv: Option<[Convolver; 2]>,
}

impl CabIr {
    pub fn new() -> Self {
        Self { sr: 48000.0, raw: Vec::new(), level: 0.0, delay_ms: 0.0, conv: None }
    }

    fn build(&mut self) {
        if self.raw.is_empty() {
            self.conv = None;
            return;
        }
        // "Delay comp" trims the IR's leading samples.
        let skip = ((self.delay_ms / 1000.0) * CONTENT_RATE) as usize;
        let ir = resample(&self.raw[skip.min(self.raw.len() - 1)..], CONTENT_RATE, self.sr);
        self.conv = Some([Convolver::new(&ir), Convolver::new(&ir)]);
    }
}

impl Effect for CabIr {
    fn prepare(&mut self, sr: f32) {
        self.sr = sr;
        self.build();
    }

    fn set(&mut self, name: &str, v: f32) {
        match name {
            "level" => self.level = v,
            // Changing the trim needs a rebuild; it is only ever set at load.
            "delay" => {
                if (v - self.delay_ms).abs() > 1e-4 {
                    self.delay_ms = v;
                    if !self.raw.is_empty() && self.conv.is_some() {
                        self.build();
                    }
                }
            }
            _ => {}
        }
    }

    fn set_data(&mut self, name: &str, data: &[u8]) {
        if name == "ir" {
            self.raw = data.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
            if self.raw.iter().any(|v| !v.is_finite()) {
                self.raw.clear();
            }
            self.build();
        }
    }

    fn reset(&mut self) {
        if let Some(c) = &mut self.conv {
            c.iter_mut().for_each(Convolver::reset);
        }
    }

    fn process(&mut self, _: &Ctx, l: &mut [f32], r: &mut [f32]) {
        let Some(conv) = &mut self.conv else { return };
        let g = db_to_gain(self.level);
        for (a, b) in l.iter_mut().zip(r.iter_mut()) {
            *a = conv[0].process(*a) * g;
            *b = conv[1].process(*b) * g;
        }
    }
}

// ---------------------------------------------------------------------------
// Reverb IR (TA_REVIR)

/// Header of the compressed reverb IR blobs: channel count, frames, peak.
pub struct IrHeader {
    pub channels: u32,
    pub frames: u32,
    pub peak: f32,
}

pub fn ir_header(data: &[u8]) -> Option<IrHeader> {
    if data.len() < 12 {
        return None;
    }
    let u = |i: usize| u32::from_le_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]);
    let h = IrHeader { channels: u(0), frames: u(4), peak: f32::from_le_bytes([data[8], data[9], data[10], data[11]]) };
    (matches!(h.channels, 1 | 2) && h.frames > 0).then_some(h)
}

pub struct ReverbIr {
    sr: f32,
    channels: Vec<Vec<f32>>,
    header_frames: u32,
    length: f32,
    mix: f32,
    level: f32,
    predelay: f32,
    fake_stereo: bool,
    conv: Vec<Convolver>,
    fallback: Option<Fdn>,
    pre: [DelayLine; 2],
}

impl ReverbIr {
    pub fn new() -> Self {
        Self {
            sr: 48000.0,
            channels: Vec::new(),
            header_frames: 0,
            length: 100.0,
            mix: 100.0,
            level: 0.0,
            predelay: 0.0,
            fake_stereo: false,
            conv: Vec::new(),
            fallback: None,
            pre: [DelayLine::new(1), DelayLine::new(1)],
        }
    }

    fn build(&mut self) {
        self.conv.clear();
        self.fallback = None;
        // Cap very long IRs to keep the partitioned convolution affordable.
        let max_frames = (8.0 * CONTENT_RATE) as usize;
        if !self.channels.is_empty() {
            // Drop the digital silence many long IRs carry at the end.
            let peak = self.channels.iter().flatten().fold(0.0f32, |m, v| m.max(v.abs()));
            let floor = peak * 1e-5;
            let audible = self
                .channels
                .iter()
                .map(|ch| ch.iter().rposition(|v| v.abs() > floor).map_or(1, |i| i + 1))
                .max()
                .unwrap_or(1);
            let keep = |ch: &Vec<f32>| {
                let n = ((audible as f32 * self.length / 100.0) as usize).clamp(1, max_frames.min(ch.len()));
                let mut v = ch[..n].to_vec();
                // Short fade-out where the IR is truncated.
                let fade = (n / 20).min(2048);
                for i in 0..fade {
                    v[n - 1 - i] *= i as f32 / fade as f32;
                }
                resample(&v, CONTENT_RATE, self.sr)
            };
        } else if self.header_frames > 0 {
            // Undecodable IR: approximate with an algorithmic tail of the same
            // length.
            let secs = self.header_frames as f32 / CONTENT_RATE * self.length / 100.0;
            let mut f = Fdn::new(self.sr);
            f.configure(FdnParams { rt60: (secs * 0.8).clamp(0.2, 12.0), hf_ratio: 0.6, damp_hz: 7000.0, size: 1.0, diffusion: 0.7, modulation: 0.2 });
            self.fallback = Some(f);
        }
    }
}

impl Effect for ReverbIr {
    fn prepare(&mut self, sr: f32) {
        self.sr = sr;
        for p in &mut self.pre {
            p.resize((1.0 * sr) as usize);
        }
        self.build();
    }

    fn set(&mut self, name: &str, v: f32) {
        match name {
            // Truncation is applied when the IR is (re)built at load time;
            // rebuilding from the audio thread would allocate.
            "length" => self.length = v,
            "mix" => self.mix = v,
            "level" => self.level = v,
            "predelay" => self.predelay = v,
            "fakestereo" => self.fake_stereo = on(v),
            _ => {}
        }
    }

    fn set_data(&mut self, name: &str, data: &[u8]) {
        if name != "ir" {
            return;
        }
        let Some(h) = ir_header(data) else { return };
        self.header_frames = h.frames;
        self.channels = crate::irdecode::decode(data).unwrap_or_default();
        self.build();
    }

    fn reset(&mut self) {
        self.conv.iter_mut().for_each(Convolver::reset);
        if let Some(f) = &mut self.fallback {
            f.clear();
        }
    }

    fn process(&mut self, _: &Ctx, l: &mut [f32], r: &mut [f32]) {
        let mix = self.mix / 100.0;
        let g = db_to_gain(self.level);
        let pd = (self.predelay * self.sr).max(1.0);
        for i in 0..l.len() {
            let (x, y) = (l[i], r[i]);
            self.pre[0].push(x);
            self.pre[1].push(y);
            let (px, py) = (self.pre[0].read(pd), self.pre[1].read(pd));
            let (wl, wr) = if self.conv.len() == 2 {
                if self.fake_stereo {
                    // Cross-feed so each input excites both IR channels.
                    let m = (px + py) * 0.5;
                    (self.conv[0].process(m), self.conv[1].process(m))
                } else {
                    (self.conv[0].process(px), self.conv[1].process(py))
                }
            } else if let Some(f) = &mut self.fallback {
                f.process(px, py)
            } else {
                (0.0, 0.0)
            };
            // The original applies a fixed 0.08 to the (peak-normalized) wet
            // signal; the fallback tail is already at a sensible level.
            let wet_scale = if self.conv.len() == 2 { 0.08 } else { 1.0 };
            l[i] = (x * (1.0 - mix) + wl * mix * wet_scale) * g;
            r[i] = (y * (1.0 - mix) + wr * mix * wet_scale) * g;
        }
    }
}

// ---------------------------------------------------------------------------
// Matching EQ (TA_TT_MATCHINGEQ)

pub struct MatchingEq {
    sr: f32,
    a: Vec<f32>,
    b: Vec<f32>,
    amount: f32,
    smooth: f32,
    hpf: f32,
    lpf: f32,
    lev_comp: bool,
    conv: Option<[Convolver; 2]>,
    /// Amount the current filter was designed for; later changes scale the
    /// wet signal instead of redesigning on the audio thread.
    built_amount: f32,
}

impl MatchingEq {
    pub fn new() -> Self {
        Self {
            sr: 48000.0,
            a: Vec::new(),
            b: Vec::new(),
            amount: 100.0,
            smooth: 50.0,
            hpf: 20.0,
            lpf: 20000.0,
            lev_comp: false,
            conv: None,
            built_amount: 100.0,
        }
    }

    fn build(&mut self) {
        self.built_amount = self.amount;
        if self.a.len() < 16 || self.a.len() != self.b.len() {
            self.conv = None;
            return;
        }
        let bins = self.a.len();
        let df = CONTENT_RATE / ((bins - 1) * 2) as f32;
        let raw: Vec<f32> = self
            .a
            .iter()
            .zip(&self.b)
            .map(|(&a, &b)| 20.0 * ((b.max(1e-9)) / (a.max(1e-9))).log10())
            .collect();
        // Fractional-octave smoothing: 1/24 octave .. 1 octave.
        let width_oct = lerp(1.0 / 24.0, 1.0, self.smooth / 100.0);
        let half = 2f32.powf(width_oct / 2.0);
        let mut prefix = vec![0.0f64; bins + 1];
        for i in 0..bins {
            prefix[i + 1] = prefix[i] + raw[i] as f64;
        }
        let mut corr = vec![0.0f32; bins];
        for (i, c) in corr.iter_mut().enumerate() {
            let f = (i as f32 * df).max(df);
            let lo = ((f / half / df) as usize).min(bins - 1);
            let hi = ((f * half / df).ceil() as usize).clamp(lo + 1, bins);
            let avg = ((prefix[hi] - prefix[lo]) / (hi - lo) as f64) as f32;
            let in_band = f >= self.hpf && f <= self.lpf;
            *c = if in_band { avg.clamp(-18.0, 18.0) * self.amount / 100.0 } else { 0.0 };
        }
        if self.lev_comp {
            let mean = corr.iter().sum::<f32>() / bins as f32;
            corr.iter_mut().for_each(|c| *c -= mean);
        }
        // Design at the content rate, then resample the FIR.
        let mag: Vec<f32> = corr.iter().map(|&db| db_to_gain(db)).collect();
        let fir = resample(&min_phase_fir(&mag, 2048), CONTENT_RATE, self.sr);
        self.conv = Some([Convolver::new(&fir), Convolver::new(&fir)]);
    }
}

impl Effect for MatchingEq {
    fn prepare(&mut self, sr: f32) {
        self.sr = sr;
        self.build();
    }

    fn set(&mut self, name: &str, v: f32) {
        // Shape parameters take effect when the filter is designed (load time).
        match name {
            "amount" => self.amount = v,
            "smooth" => self.smooth = v,
            "hpf" => self.hpf = v,
            "lpf" => self.lpf = v,
            "lev_comp" => self.lev_comp = on(v),
            _ => {}
        }
    }

    fn set_data(&mut self, name: &str, data: &[u8]) {
        if name != "profiles" {
            return;
        }
        let f: Vec<f32> = data.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        if f.len() < 4 {
            return;
        }
        let bins = (f.len() - 2) / 2;
        self.a = f[2..2 + bins].to_vec();
        self.b = f[2 + bins..2 + 2 * bins].to_vec();
        self.build();
    }

    fn reset(&mut self) {
        if let Some(c) = &mut self.conv {
            c.iter_mut().for_each(Convolver::reset);
        }
    }

    fn process(&mut self, _: &Ctx, l: &mut [f32], r: &mut [f32]) {
        let Some(conv) = &mut self.conv else { return };
        let wet = if self.built_amount > 0.0 { (self.amount / self.built_amount).clamp(0.0, 1.0) } else { 0.0 };
        for (a, b) in l.iter_mut().zip(r.iter_mut()) {
            *a = lerp(*a, conv[0].process(*a), wet);
            *b = lerp(*b, conv[1].process(*b), wet);
        }
    }
}
