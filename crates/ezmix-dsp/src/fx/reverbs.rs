//! Algorithmic reverbs built on an 8-line feedback delay network.

use crate::dsp::*;
use crate::effect::{idx, on, Ctx, Effect, SyncParam, INVERSE_NOTES, PREDELAY_NOTES};

const N: usize = 8;
/// Mutually prime-ish base lengths in milliseconds.
const BASE_MS: [f32; N] = [29.7, 37.1, 41.1, 43.7, 53.3, 59.9, 67.7, 73.1];

/// Normalized 8x8 Hadamard mixing.
#[inline]
fn hadamard(x: &mut [f32; N]) {
    let mut h = 1;
    while h < N {
        let mut i = 0;
        while i < N {
            for j in i..i + h {
                let (a, b) = (x[j], x[j + h]);
                x[j] = a + b;
                x[j + h] = a - b;
            }
            i += 2 * h;
        }
        h *= 2;
    }
    let s = 1.0 / (N as f32).sqrt();
    x.iter_mut().for_each(|v| *v *= s);
}

#[derive(Clone)]
pub struct Fdn {
    sr: f32,
    lines: Vec<DelayLine>,
    len: [f32; N],
    gain: [f32; N],
    damp: [OnePole; N],
    diff: Vec<Allpass>,
    lfo: Lfo,
    mod_depth: f32,
    state: [f32; N],
}

#[derive(Clone, Copy, Debug)]
pub struct FdnParams {
    pub rt60: f32,
    /// Ratio of high-frequency to mid RT60 (1 = no damping).
    pub hf_ratio: f32,
    pub damp_hz: f32,
    pub size: f32,
    pub diffusion: f32,
    pub modulation: f32,
}

impl Fdn {
    pub fn new(sr: f32) -> Self {
        let max = (0.08 * 2.0 * sr) as usize + 64;
        let diff = [4.7f32, 6.3, 8.9, 12.1]
            .iter()
            .map(|ms| Allpass::new((ms * 0.001 * sr * 2.0) as usize + 8, ms * 0.001 * sr, 0.6))
            .collect();
        Self {
            sr,
            lines: (0..N).map(|_| DelayLine::new(max)).collect(),
            len: [1000.0; N],
            gain: [0.5; N],
            damp: Default::default(),
            diff,
            lfo: Lfo::with_freq(0.9),
            mod_depth: 0.0,
            state: [0.0; N],
        }
    }

    pub fn configure(&mut self, p: FdnParams) {
        let sr = self.sr;
        let size = p.size.clamp(0.1, 2.0);
        let rt = p.rt60.max(0.05);
        for i in 0..N {
            self.len[i] = BASE_MS[i] * 0.001 * sr * size;
            // Per-line gain for the requested decay.
            self.gain[i] = 10f32.powf(-3.0 * self.len[i] / (rt * sr));
            // Damping filter: lower cutoff for stronger HF decay.
            let hf = (p.damp_hz * p.hf_ratio.clamp(0.1, 4.0)).clamp(500.0, sr * 0.45);
            self.damp[i].set_lp(sr, hf);
        }
        for (k, a) in self.diff.iter_mut().enumerate() {
            a.g = p.diffusion.clamp(0.0, 0.85) * (1.0 - 0.05 * k as f32);
            a.delay = [4.7f32, 6.3, 8.9, 12.1][k] * 0.001 * sr * size.min(1.5);
        }
        self.mod_depth = p.modulation * 0.0008 * sr;
    }

    pub fn clear(&mut self) {
        self.lines.iter_mut().for_each(DelayLine::clear);
        self.diff.iter_mut().for_each(Allpass::clear);
        self.state = [0.0; N];
    }

    #[inline]
    pub fn process(&mut self, l: f32, r: f32) -> (f32, f32) {
        let mut x = (l + r) * 0.5;
        for a in &mut self.diff {
            x = a.process(x);
        }
        self.lfo.advance(self.sr);
        let mut taps = [0.0f32; N];
        for i in 0..N {
            let m = if self.mod_depth > 0.0 { self.mod_depth * self.lfo.value(Wave::Sine, i as f32 / N as f32) } else { 0.0 };
            taps[i] = self.lines[i].read(self.len[i] + m);
        }
        let mut fb = taps;
        hadamard(&mut fb);
        let side = (l - r) * 0.25;
        for i in 0..N {
            let inj = if i % 2 == 0 { x + side } else { x - side };
            let v = self.damp[i].lp(fb[i] * self.gain[i]) + inj * 0.35;
            self.lines[i].push(v);
            self.state[i] = v;
        }
        let ol = taps[0] - taps[2] + taps[4] - taps[6] + 0.5 * (taps[1] + taps[5]);
        let or = taps[1] - taps[3] + taps[5] - taps[7] + 0.5 * (taps[2] + taps[6]);
        (ol * 0.5, or * 0.5)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ReverbKind {
    Hall,
    Room,
    Plate,
    Spring,
}

pub struct Reverb {
    sr: f32,
    kind: ReverbKind,
    time: f32,
    size: f32,
    diffusion: f32,
    shape: f32,
    high: f32,
    low_hz: f32,
    high_hz: f32,
    damping: f32,
    decay: f32,
    motion: f32,
    predelay: SyncParam,
    wet: f32,
    dry: f32,
    level: f32,
    level_db: f32,
    tension: f32,
    bright: f32,
    fdn: Fdn,
    pre: [DelayLine; 2],
    tone: [StereoBiquad; 2],
    spring_ap: [Vec<Allpass>; 2],
    er: [DelayLine; 2],
    dirty: bool,
}

impl Reverb {
    pub fn new(kind: ReverbKind) -> Self {
        Self {
            sr: 48000.0,
            kind,
            time: if kind == ReverbKind::Spring { 4.0 } else { 2.0 },
            size: 50.0,
            diffusion: 50.0,
            shape: 50.0,
            high: 1.0,
            low_hz: 100.0,
            high_hz: 8000.0,
            damping: 1.0,
            decay: 50.0,
            motion: 20.0,
            predelay: SyncParam::time(0.0, PREDELAY_NOTES),
            wet: 15.0,
            dry: 100.0,
            level: 100.0,
            level_db: 0.0,
            tension: 1.0,
            bright: 50.0,
            fdn: Fdn::new(48000.0),
            pre: [DelayLine::new(1), DelayLine::new(1)],
            tone: Default::default(),
            spring_ap: Default::default(),
            er: [DelayLine::new(1), DelayLine::new(1)],
            dirty: true,
        }
    }

    fn update(&mut self) {
        let sr = self.sr;
        let size = match self.kind {
            ReverbKind::Hall => lerp(0.7, 1.6, self.size / 100.0),
            ReverbKind::Room => lerp(0.25, 0.7, self.size / 100.0),
            ReverbKind::Plate => lerp(0.4, 0.9, self.size / 100.0),
            ReverbKind::Spring => 0.35,
        };
        let (hf_ratio, damp_hz) = match self.kind {
            ReverbKind::Hall => (self.high, 6000.0),
            ReverbKind::Plate => (self.high, 9000.0),
            ReverbKind::Room => (1.0 / self.damping.max(0.1), self.high_hz.min(12000.0)),
            ReverbKind::Spring => (lerp(0.5, 1.5, self.bright / 100.0), 5000.0),
        };
        let rt = match self.kind {
            ReverbKind::Room => self.time * lerp(0.6, 1.4, self.decay / 100.0),
            _ => self.time,
        };
        self.fdn.configure(FdnParams {
            rt60: rt,
            hf_ratio,
            damp_hz,
            size,
            diffusion: match self.kind {
                ReverbKind::Plate => lerp(0.5, 0.8, self.diffusion / 100.0),
                ReverbKind::Spring => 0.3,
                _ => lerp(0.2, 0.75, self.diffusion / 100.0),
            },
            modulation: match self.kind {
                ReverbKind::Room => self.motion / 100.0,
                ReverbKind::Spring => 0.4,
                _ => 0.25,
            },
        });
        let (lo, hi) = match self.kind {
            ReverbKind::Room => (self.low_hz, self.high_hz),
            ReverbKind::Spring => (180.0, lerp(3000.0, 7000.0, self.bright / 100.0)),
            _ => (40.0, 16000.0),
        };
        self.tone[0].set(highpass(sr, lo.max(20.0)));
        self.tone[1].set(if hi < sr * 0.45 { lowpass(sr, hi) } else { Coefs::IDENTITY });
        if self.kind == ReverbKind::Spring {
            // Dispersive allpass chain gives the spring "chirp"; tension
            // shortens the delays.
            let d = 0.0015 * sr / self.tension.clamp(0.3, 4.0);
            for ch in 0..2 {
                for (k, a) in self.spring_ap[ch].iter_mut().enumerate() {
                    a.delay = d * (1.0 + 0.07 * k as f32 + 0.03 * ch as f32);
                    a.g = 0.6;
                }
            }
        }
        self.dirty = false;
    }
}

impl Effect for Reverb {
    fn prepare(&mut self, sr: f32) {
        self.sr = sr;
        self.fdn = Fdn::new(sr);
        for p in &mut self.pre {
            p.resize((2.5 * sr) as usize);
        }
        for e in &mut self.er {
            e.resize((0.1 * sr) as usize);
        }
        if self.kind == ReverbKind::Spring {
            // Longest dispersion delay at minimum tension.
            let max = (0.0015 * sr / 0.3 * 2.2) as usize + 8;
            for ch in 0..2 {
                self.spring_ap[ch] = (0..12).map(|_| Allpass::new(max, 1.0, 0.6)).collect();
            }
        }
        self.update();
    }

    fn set(&mut self, name: &str, v: f32) {
        self.dirty = true;
        match name {
            "time" => self.time = v,
            "size" => self.size = v,
            "diffusion" => self.diffusion = v,
            "shape" => self.shape = v,
            "high" if self.kind == ReverbKind::Room => self.high_hz = v,
            "high" | "colour" => self.high = v,
            "low" => self.low_hz = v,
            "damping" => self.damping = v,
            "decay" => self.decay = v,
            "motion" => self.motion = v,
            "depth" => self.motion = v,
            "predelay" => self.predelay.free = v,
            "predelay-synced" => self.predelay.synced = on(v),
            "predelay-bpm" => self.predelay.index = idx(v),
            "wet" => self.wet = v,
            "dry" => self.dry = v,
            "level" if self.kind == ReverbKind::Spring => self.level_db = v,
            "level" => self.level = v,
            "tension" => self.tension = v,
            "bright" => self.bright = v,
            _ => self.dirty = false,
        }
    }

    fn reset(&mut self) {
        self.fdn.clear();
        self.pre.iter_mut().for_each(DelayLine::clear);
    }

    fn process(&mut self, ctx: &Ctx, l: &mut [f32], r: &mut [f32]) {
        if self.dirty {
            self.update();
        }
        let pd = (self.predelay.value(ctx.bpm) * self.sr).clamp(1.0, 2.4 * self.sr);
        let wet = self.wet / 100.0;
        let (dry, out) = match self.kind {
            ReverbKind::Spring => (self.dry / 100.0, db_to_gain(self.level_db)),
            _ => (1.0 - wet, self.level / 100.0),
        };
        // Shape: early reflection emphasis (0) to smooth late tail (100).
        let er_gain = (1.0 - self.shape / 100.0) * 0.6;
        let er_taps = [(0.0071, 0.8), (0.0113, 0.6), (0.0179, 0.5), (0.0241, 0.4), (0.0317, 0.3)];
        for i in 0..l.len() {
            let (x, y) = (l[i], r[i]);
            self.pre[0].push(x);
            self.pre[1].push(y);
            let (mut px, mut py) = (self.pre[0].read(pd), self.pre[1].read(pd));
            if self.kind == ReverbKind::Spring {
                for a in &mut self.spring_ap[0] {
                    px = a.process(px);
                }
                for a in &mut self.spring_ap[1] {
                    py = a.process(py);
                }
            }
            let (mut rl, mut rr) = self.fdn.process(px, py);
            if er_gain > 0.0 && self.kind != ReverbKind::Spring {
                self.er[0].push(px);
                self.er[1].push(py);
                for (k, &(t, g)) in er_taps.iter().enumerate() {
                    let d = (t * self.sr) as usize;
                    let (a, b) = if k % 2 == 0 { (0, 1) } else { (1, 0) };
                    rl += self.er[a].tap(d) * g * er_gain;
                    rr += self.er[b].tap(d + 7) * g * er_gain;
                }
            }
            let (rl, rr) = self.tone[0].process(rl, rr);
            let (rl, rr) = self.tone[1].process(rl, rr);
            l[i] = (x * dry + rl * wet) * out;
            r[i] = (y * dry + rr * wet) * out;
        }
    }
}

// ---------------------------------------------------------------------------
// Sparse-FIR reverbs: inverse (TA_REVINVERSE) and gated/non-linear
// (TA_AMSRMX16NONLIN)

const TAPS: usize = 96;

pub struct SparseReverb {
    sr: f32,
    inverse: bool,
    time: SyncParam,
    diffusion: f32,
    high_hz: f32,
    low_hz: f32,
    predelay: SyncParam,
    wet: f32,
    level: f32,
    dry_db: f32,
    wet_db: f32,
    level_db: f32,
    decay: f32,
    high_amt: f32,
    line: [DelayLine; 2],
    taps: [(usize, f32, f32); TAPS],
    diff: [Allpass; 2],
    tone: [StereoBiquad; 2],
    built_for: f32,
}

impl SparseReverb {
    pub fn new(inverse: bool) -> Self {
        Self {
            sr: 48000.0,
            inverse,
            time: SyncParam::time(1.0, INVERSE_NOTES),
            diffusion: 50.0,
            high_hz: 8000.0,
            low_hz: 100.0,
            predelay: SyncParam::time(0.0, PREDELAY_NOTES),
            wet: 25.0,
            level: 100.0,
            dry_db: -40.0,
            wet_db: -12.0,
            level_db: 0.0,
            decay: 5.0,
            high_amt: 0.0,
            line: [DelayLine::new(1), DelayLine::new(1)],
            taps: [(1, 0.0, 0.0); TAPS],
            diff: [Allpass::new(1, 1.0, 0.0), Allpass::new(1, 1.0, 0.0)],
            tone: Default::default(),
            built_for: -1.0,
        }
    }

    fn build(&mut self, length_s: f32) {
        let mut rng = Noise::default();
        let len = (length_s * self.sr).max(64.0);
        let mut norm = 0.0;
        for k in 0..TAPS {
            let pos = (k as f32 + 0.5 + 0.4 * rng.next()) / TAPS as f32;
            let env = if self.inverse {
                // Rising envelope ending abruptly.
                (pos * 4.0).exp() / 4f32.exp()
            } else {
                // Gated: dense, slightly decaying, hard cut.
                1.0 - 0.5 * pos
            };
            let gl = env * rng.next().signum();
            let gr = env * rng.next().signum();
            norm += env * env;
            self.taps[k] = (((pos * len) as usize).max(1), gl, gr);
        }
        let n = 1.0 / norm.sqrt().max(1e-6);
        for t in &mut self.taps {
            t.1 *= n;
            t.2 *= n;
        }
        self.built_for = length_s;
    }

    fn update(&mut self) {
        let sr = self.sr;
        let (lo, hi) = if self.inverse {
            (self.low_hz, self.high_hz)
        } else {
            (60.0, 16000.0 / 2f32.powf(self.high_amt.max(0.0) * 0.5))
        };
        self.tone[0].set(highpass(sr, lo.max(20.0)));
        self.tone[1].set(if hi < sr * 0.45 { lowpass(sr, hi) } else { Coefs::IDENTITY });
        let g = if self.inverse { self.diffusion / 100.0 * 0.7 } else { 0.5 };
        for (k, d) in self.diff.iter_mut().enumerate() {
            d.g = g;
            d.delay = (0.0071 + 0.0033 * k as f32) * sr;
        }
    }
}

impl Effect for SparseReverb {
    fn prepare(&mut self, sr: f32) {
        self.sr = sr;
        for l in &mut self.line {
            l.resize((4.5 * sr) as usize);
        }
        self.diff = [Allpass::new((0.02 * sr) as usize, 0.0071 * sr, 0.5), Allpass::new((0.02 * sr) as usize, 0.0104 * sr, 0.5)];
        self.built_for = -1.0;
        self.update();
    }

    fn set(&mut self, name: &str, v: f32) {
        match name {
            "time" => self.time.free = v,
            "time-synced" => self.time.synced = on(v),
            "time-bpm" => self.time.index = idx(v),
            "diffusion" => self.diffusion = v,
            "high" if self.inverse => self.high_hz = v,
            "high" => self.high_amt = v,
            "low" => self.low_hz = v,
            "predelay" => self.predelay.free = v,
            "predelay-synced" => self.predelay.synced = on(v),
            "predelay-bpm" => self.predelay.index = idx(v),
            "wet" if self.inverse => self.wet = v,
            "wet" => self.wet_db = v,
            "dry" => self.dry_db = v,
            "level" if self.inverse => self.level = v,
            "level" => self.level_db = v,
            "decay" => self.decay = v,
            _ => return,
        }
        self.update();
    }

    fn process(&mut self, ctx: &Ctx, l: &mut [f32], r: &mut [f32]) {
        let length = if self.inverse {
            self.time.value(ctx.bpm).clamp(0.05, 4.0)
        } else {
            0.08 + self.decay.clamp(0.0, 9.0) / 9.0 * 0.6
        };
        if (length - self.built_for).abs() > 1e-3 {
            self.build(length);
        }
        let pd = (self.predelay.value(ctx.bpm) * self.sr) as usize;
        let (dry, wet, out) = if self.inverse {
            (1.0 - self.wet / 100.0, self.wet / 100.0, self.level / 100.0)
        } else {
            (db_to_gain(self.dry_db), db_to_gain(self.wet_db), db_to_gain(self.level_db))
        };
        for i in 0..l.len() {
            let (x, y) = (l[i], r[i]);
            self.line[0].push(x);
            self.line[1].push(y);
            let (mut a, mut b) = (0.0, 0.0);
            for &(d, gl, gr) in &self.taps {
                a += self.line[0].tap(d + pd) * gl;
                b += self.line[1].tap(d + pd) * gr;
            }
            let a = self.diff[0].process(a);
            let b = self.diff[1].process(b);
            let (a, b) = self.tone[0].process(a, b);
            let (a, b) = self.tone[1].process(a, b);
            l[i] = (x * dry + a * wet) * out;
            r[i] = (y * dry + b * wet) * out;
        }
    }
}
