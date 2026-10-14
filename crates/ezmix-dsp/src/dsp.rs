//! Shared DSP building blocks.

use std::f32::consts::PI;

pub const TAU: f32 = 2.0 * PI;

#[inline]
pub fn db_to_gain(db: f32) -> f32 {
    if db <= -150.0 { 0.0 } else { 10f32.powf(db * 0.05) }
}

#[inline]
pub fn gain_to_db(g: f32) -> f32 {
    20.0 * g.max(1e-9).log10()
}

#[inline]
pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Map a 0..10 amp-style knob to 0..1.
#[inline]
pub fn knob10(v: f32) -> f32 {
    (v / 10.0).clamp(0.0, 1.0)
}

/// Coefficient for a one-pole smoother reaching ~63% in `seconds`.
#[inline]
pub fn time_coef(seconds: f32, sr: f32) -> f32 {
    if seconds <= 0.0 { 0.0 } else { (-1.0 / (seconds * sr)).exp() }
}

/// Exponential parameter smoother.
#[derive(Clone, Copy, Debug)]
pub struct Smoothed {
    pub current: f32,
    pub target: f32,
    coef: f32,
}

impl Smoothed {
    pub fn new(v: f32) -> Self {
        Self { current: v, target: v, coef: 0.999 }
    }

    pub fn prepare(&mut self, sr: f32, seconds: f32) {
        self.coef = time_coef(seconds, sr);
    }

    pub fn set(&mut self, v: f32) {
        self.target = v;
    }

    pub fn snap(&mut self) {
        self.current = self.target;
    }

    #[inline]
    pub fn next(&mut self) -> f32 {
        self.current = self.target + (self.current - self.target) * self.coef;
        self.current
    }
}

/// Soft saturators.
pub mod shape {
    #[inline]
    pub fn tanh(x: f32) -> f32 {
        // Pade-style approximation, accurate to ~1e-3, clamped.
        let x = x.clamp(-4.97, 4.97);
        let x2 = x * x;
        x * (27.0 + x2) / (27.0 + 9.0 * x2)
    }

    #[inline]
    pub fn soft_clip(x: f32) -> f32 {
        if x > 1.0 {
            2.0 / 3.0
        } else if x < -1.0 {
            -2.0 / 3.0
        } else {
            x - x * x * x / 3.0
        }
    }

    /// Asymmetric triode-like curve; `bias` (-1..1) adds even harmonics.
    #[inline]
    pub fn tube(x: f32, bias: f32) -> f32 {
        let y = tanh(x + bias) - tanh(bias);
        if y > 0.0 { y } else { y * (1.0 - 0.25 * bias.abs()) }
    }

    /// Silicon diode pair (Tube-Screamer style feedback clipping).
    #[inline]
    pub fn diode(x: f32) -> f32 {
        x / (1.0 + x.abs()).powf(0.8)
    }

    #[inline]
    pub fn hard(x: f32, limit: f32) -> f32 {
        x.clamp(-limit, limit)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BiquadKind {
    LowPass,
    HighPass,
    BandPass,
    Notch,
    Peak,
    LowShelf,
    HighShelf,
    AllPass,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Coefs {
    pub b0: f32,
    pub b1: f32,
    pub b2: f32,
    pub a1: f32,
    pub a2: f32,
}

impl Coefs {
    pub const IDENTITY: Coefs = Coefs { b0: 1.0, b1: 0.0, b2: 0.0, a1: 0.0, a2: 0.0 };

    /// RBJ cookbook designs. `gain_db` only matters for peak/shelf.
    pub fn rbj(kind: BiquadKind, sr: f32, freq: f32, q: f32, gain_db: f32) -> Self {
        let f = freq.clamp(5.0, sr * 0.49);
        let w = TAU * f / sr;
        let (sn, cs) = w.sin_cos();
        let q = q.max(0.025);
        let alpha = sn / (2.0 * q);
        let a = 10f32.powf(gain_db / 40.0);
        let (b0, b1, b2, a0, a1, a2) = match kind {
            BiquadKind::LowPass => ((1.0 - cs) / 2.0, 1.0 - cs, (1.0 - cs) / 2.0, 1.0 + alpha, -2.0 * cs, 1.0 - alpha),
            BiquadKind::HighPass => ((1.0 + cs) / 2.0, -(1.0 + cs), (1.0 + cs) / 2.0, 1.0 + alpha, -2.0 * cs, 1.0 - alpha),
            BiquadKind::BandPass => (alpha, 0.0, -alpha, 1.0 + alpha, -2.0 * cs, 1.0 - alpha),
            BiquadKind::Notch => (1.0, -2.0 * cs, 1.0, 1.0 + alpha, -2.0 * cs, 1.0 - alpha),
            BiquadKind::AllPass => (1.0 - alpha, -2.0 * cs, 1.0 + alpha, 1.0 + alpha, -2.0 * cs, 1.0 - alpha),
            BiquadKind::Peak => (1.0 + alpha * a, -2.0 * cs, 1.0 - alpha * a, 1.0 + alpha / a, -2.0 * cs, 1.0 - alpha / a),
            BiquadKind::LowShelf => {
                let s = 2.0 * a.sqrt() * alpha;
                (
                    a * ((a + 1.0) - (a - 1.0) * cs + s),
                    2.0 * a * ((a - 1.0) - (a + 1.0) * cs),
                    a * ((a + 1.0) - (a - 1.0) * cs - s),
                    (a + 1.0) + (a - 1.0) * cs + s,
                    -2.0 * ((a - 1.0) + (a + 1.0) * cs),
                    (a + 1.0) + (a - 1.0) * cs - s,
                )
            }
            BiquadKind::HighShelf => {
                let s = 2.0 * a.sqrt() * alpha;
                (
                    a * ((a + 1.0) + (a - 1.0) * cs + s),
                    -2.0 * a * ((a - 1.0) + (a + 1.0) * cs),
                    a * ((a + 1.0) + (a - 1.0) * cs - s),
                    (a + 1.0) - (a - 1.0) * cs + s,
                    2.0 * ((a - 1.0) - (a + 1.0) * cs),
                    (a + 1.0) - (a - 1.0) * cs - s,
                )
            }
        };
        Coefs { b0: b0 / a0, b1: b1 / a0, b2: b2 / a0, a1: a1 / a0, a2: a2 / a0 }
    }

    /// First-order sections expressed as a biquad.
    pub fn one_pole_lp(sr: f32, freq: f32) -> Self {
        let k = (PI * freq.clamp(5.0, sr * 0.49) / sr).tan();
        let n = 1.0 / (1.0 + k);
        Coefs { b0: k * n, b1: k * n, b2: 0.0, a1: (k - 1.0) * n, a2: 0.0 }
    }

    pub fn one_pole_hp(sr: f32, freq: f32) -> Self {
        let k = (PI * freq.clamp(5.0, sr * 0.49) / sr).tan();
        let n = 1.0 / (1.0 + k);
        Coefs { b0: n, b1: -n, b2: 0.0, a1: (k - 1.0) * n, a2: 0.0 }
    }

    /// First-order shelves (gentle "one-resonance-free" shelves).
    pub fn one_pole_shelf(sr: f32, freq: f32, gain_db: f32, high: bool) -> Self {
        let g = db_to_gain(gain_db);
        let k = (PI * freq.clamp(5.0, sr * 0.49) / sr).tan();
        // H(s) = (s + g w) / (s + w) for low shelf, (g s + w) / (s + w) high.
        let (nb0, nb1) = if high { (g + k, k - g) } else { (1.0 + g * k, g * k - 1.0) };
        let a0 = 1.0 + k;
        Coefs { b0: nb0 / a0, b1: nb1 / a0, b2: 0.0, a1: (k - 1.0) / a0, a2: 0.0 }
    }

    /// Magnitude response at `freq`.
    pub fn magnitude(&self, sr: f32, freq: f32) -> f32 {
        let w = TAU * freq / sr;
        let (s1, c1) = w.sin_cos();
        let (s2, c2) = (2.0 * w).sin_cos();
        let nr = self.b0 + self.b1 * c1 + self.b2 * c2;
        let ni = -self.b1 * s1 - self.b2 * s2;
        let dr = 1.0 + self.a1 * c1 + self.a2 * c2;
        let di = -self.a1 * s1 - self.a2 * s2;
        ((nr * nr + ni * ni) / (dr * dr + di * di).max(1e-20)).sqrt()
    }
}

/// Transposed direct form II biquad.
#[derive(Clone, Copy, Debug, Default)]
pub struct Biquad {
    pub c: Coefs,
    z1: f32,
    z2: f32,
}

impl Biquad {
    pub fn new(c: Coefs) -> Self {
        Self { c, z1: 0.0, z2: 0.0 }
    }

    pub fn identity() -> Self {
        Self::new(Coefs::IDENTITY)
    }

    pub fn set(&mut self, c: Coefs) {
        self.c = c;
    }

    pub fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }

    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        let c = &self.c;
        let y = c.b0 * x + self.z1;
        self.z1 = c.b1 * x - c.a1 * y + self.z2;
        self.z2 = c.b2 * x - c.a2 * y;
        if !y.is_finite() {
            self.reset();
            return 0.0;
        }
        y
    }
}

/// Stereo pair of identical biquads.
#[derive(Clone, Copy, Debug, Default)]
pub struct StereoBiquad {
    pub l: Biquad,
    pub r: Biquad,
}

impl StereoBiquad {
    pub fn set(&mut self, c: Coefs) {
        self.l.set(c);
        self.r.set(c);
    }

    pub fn reset(&mut self) {
        self.l.reset();
        self.r.reset();
    }

    #[inline]
    pub fn process(&mut self, l: f32, r: f32) -> (f32, f32) {
        (self.l.process(l), self.r.process(r))
    }
}

/// Butterworth low/high pass of order 1..=8 built from biquads.
#[derive(Clone, Debug, Default)]
pub struct Butterworth {
    stages: [Biquad; 4],
    n: usize,
}

impl Butterworth {
    pub fn design(&mut self, sr: f32, freq: f32, order: usize, high: bool) {
        let order = order.clamp(1, 8);
        let pairs = order / 2;
        let mut n = 0;
        for k in 0..pairs {
            let theta = PI * (2 * k + 1) as f32 / (2 * order) as f32;
            let q = 1.0 / (2.0 * theta.sin());
            let kind = if high { BiquadKind::HighPass } else { BiquadKind::LowPass };
            self.stages[n].set(Coefs::rbj(kind, sr, freq, q, 0.0));
            n += 1;
        }
        if order % 2 == 1 {
            self.stages[n].set(if high { Coefs::one_pole_hp(sr, freq) } else { Coefs::one_pole_lp(sr, freq) });
            n += 1;
        }
        self.n = n;
    }

    pub fn reset(&mut self) {
        self.stages.iter_mut().for_each(Biquad::reset);
    }

    #[inline]
    pub fn process(&mut self, mut x: f32) -> f32 {
        for s in &mut self.stages[..self.n] {
            x = s.process(x);
        }
        x
    }
}

/// Linkwitz-Riley 4th order crossover producing a low and a high band.
#[derive(Clone, Debug, Default)]
pub struct Crossover {
    lp: [Biquad; 2],
    hp: [Biquad; 2],
}

impl Crossover {
    pub fn set(&mut self, sr: f32, freq: f32) {
        let q = std::f32::consts::FRAC_1_SQRT_2;
        for b in &mut self.lp {
            b.set(Coefs::rbj(BiquadKind::LowPass, sr, freq, q, 0.0));
        }
        for b in &mut self.hp {
            b.set(Coefs::rbj(BiquadKind::HighPass, sr, freq, q, 0.0));
        }
    }

    #[inline]
    pub fn split(&mut self, x: f32) -> (f32, f32) {
        let lo = self.lp[0].process(x);
        let lo = self.lp[1].process(lo);
        let hi = self.hp[0].process(x);
        let hi = self.hp[1].process(hi);
        (lo, hi)
    }
}

/// Splits a signal into four bands with three LR4 crossovers.
#[derive(Clone, Debug, Default)]
pub struct FourBandSplit {
    x: [Crossover; 3],
    // Allpass compensation so that bands sum flat.
    ap_low: [Biquad; 2],
}

impl FourBandSplit {
    pub fn set(&mut self, sr: f32, f: [f32; 3]) {
        let mut f = f;
        f.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        self.x[0].set(sr, f[0]);
        self.x[1].set(sr, f[1]);
        self.x[2].set(sr, f[2]);
        for (b, freq) in self.ap_low.iter_mut().zip([f[1], f[2]]) {
            b.set(Coefs::rbj(BiquadKind::AllPass, sr, freq, std::f32::consts::FRAC_1_SQRT_2, 0.0));
        }
    }

    #[inline]
    pub fn split(&mut self, x: f32) -> [f32; 4] {
        let (lo_mid, hi_all) = self.x[1].split(x);
        let (b0, b1) = self.x[0].split(lo_mid);
        let (b2, b3) = self.x[2].split(hi_all);
        // Phase-align the lower half with the third crossover.
        let b0 = self.ap_low[1].process(b0);
        let b1 = self.ap_low[0].process(b1);
        [b0, b1, b2, b3]
    }
}

/// Zero-delay-feedback state variable filter (Simper/Zavalishin).
#[derive(Clone, Copy, Debug, Default)]
pub struct Svf {
    g: f32,
    k: f32,
    a1: f32,
    a2: f32,
    a3: f32,
    ic1: f32,
    ic2: f32,
}

pub struct SvfOut {
    pub low: f32,
    pub band: f32,
    pub high: f32,
}

impl Svf {
    pub fn set(&mut self, sr: f32, freq: f32, q: f32) {
        self.g = (PI * freq.clamp(5.0, sr * 0.48) / sr).tan();
        self.k = 1.0 / q.max(0.05);
        self.a1 = 1.0 / (1.0 + self.g * (self.g + self.k));
        self.a2 = self.g * self.a1;
        self.a3 = self.g * self.a2;
    }

    pub fn reset(&mut self) {
        self.ic1 = 0.0;
        self.ic2 = 0.0;
    }

    #[inline]
    pub fn process(&mut self, x: f32) -> SvfOut {
        let v3 = x - self.ic2;
        let v1 = self.a1 * self.ic1 + self.a2 * v3;
        let v2 = self.ic2 + self.a2 * self.ic1 + self.a3 * v3;
        self.ic1 = 2.0 * v1 - self.ic1;
        self.ic2 = 2.0 * v2 - self.ic2;
        if !self.ic1.is_finite() || !self.ic2.is_finite() {
            self.reset();
            return SvfOut { low: 0.0, band: 0.0, high: 0.0 };
        }
        SvfOut { low: v2, band: v1, high: x - self.k * v1 - v2 }
    }
}

/// One-pole lowpass/highpass on a single sample stream.
#[derive(Clone, Copy, Debug, Default)]
pub struct OnePole {
    a: f32,
    z: f32,
}

impl OnePole {
    pub fn set_lp(&mut self, sr: f32, freq: f32) {
        self.a = (-TAU * freq.clamp(1.0, sr * 0.49) / sr).exp();
    }

    #[inline]
    pub fn lp(&mut self, x: f32) -> f32 {
        self.z = x + (self.z - x) * self.a;
        self.z
    }

    #[inline]
    pub fn hp(&mut self, x: f32) -> f32 {
        x - self.lp(x)
    }

    pub fn reset(&mut self) {
        self.z = 0.0;
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DcBlock {
    x1: f32,
    y1: f32,
    r: f32,
}

impl DcBlock {
    pub fn prepare(&mut self, sr: f32) {
        self.r = 1.0 - TAU * 15.0 / sr;
    }

    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        let y = x - self.x1 + self.r * self.y1;
        self.x1 = x;
        self.y1 = y;
        y
    }
}

/// Circular delay line with fractional reads.
#[derive(Clone, Debug, Default)]
pub struct DelayLine {
    buf: Vec<f32>,
    pos: usize,
}

impl DelayLine {
    pub fn new(max_samples: usize) -> Self {
        Self { buf: vec![0.0; max_samples.max(4).next_power_of_two()], pos: 0 }
    }

    pub fn resize(&mut self, max_samples: usize) {
        let n = max_samples.max(4).next_power_of_two();
        if n != self.buf.len() {
            self.buf = vec![0.0; n];
            self.pos = 0;
        }
    }

    pub fn len(&self) -> usize {
        self.buf.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    pub fn clear(&mut self) {
        self.buf.iter_mut().for_each(|v| *v = 0.0);
    }

    #[inline]
    pub fn push(&mut self, x: f32) {
        let mask = self.buf.len() - 1;
        self.pos = (self.pos + 1) & mask;
        self.buf[self.pos] = x;
    }

    /// Sample written `d` pushes ago (0 = most recent).
    #[inline]
    pub fn tap(&self, d: usize) -> f32 {
        let mask = self.buf.len() - 1;
        self.buf[(self.pos.wrapping_sub(d.min(mask))) & mask]
    }

    /// Fractional delay with cubic (Hermite) interpolation.
    #[inline]
    pub fn read(&self, d: f32) -> f32 {
        let d = d.clamp(1.0, (self.buf.len() - 3) as f32);
        let i = d.floor() as usize;
        let f = d - i as f32;
        let y0 = self.tap(i - 1);
        let y1 = self.tap(i);
        let y2 = self.tap(i + 1);
        let y3 = self.tap(i + 2);
        let c0 = y1;
        let c1 = 0.5 * (y2 - y0);
        let c2 = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3;
        let c3 = 0.5 * (y3 - y0) + 1.5 * (y1 - y2);
        ((c3 * f + c2) * f + c1) * f + c0
    }

    #[inline]
    pub fn read_linear(&self, d: f32) -> f32 {
        let d = d.clamp(0.0, (self.buf.len() - 2) as f32);
        let i = d.floor() as usize;
        let f = d - i as f32;
        lerp(self.tap(i), self.tap(i + 1), f)
    }
}

/// Schroeder allpass diffuser.
#[derive(Clone, Debug, Default)]
pub struct Allpass {
    line: DelayLine,
    pub delay: f32,
    pub g: f32,
}

impl Allpass {
    pub fn new(max: usize, delay: f32, g: f32) -> Self {
        Self { line: DelayLine::new(max + 4), delay, g }
    }

    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        let d = self.line.read_linear(self.delay);
        let v = x + self.g * d;
        self.line.push(v);
        d - self.g * v
    }

    pub fn clear(&mut self) {
        self.line.clear();
    }
}

/// Peak/RMS envelope follower with separate attack and release.
#[derive(Clone, Copy, Debug, Default)]
pub struct Envelope {
    pub value: f32,
    att: f32,
    rel: f32,
}

impl Envelope {
    pub fn set(&mut self, sr: f32, attack_s: f32, release_s: f32) {
        self.att = time_coef(attack_s.max(1e-5), sr);
        self.rel = time_coef(release_s.max(1e-4), sr);
    }

    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        let c = if x > self.value { self.att } else { self.rel };
        self.value = x + (self.value - x) * c;
        self.value
    }

    pub fn reset(&mut self) {
        self.value = 0.0;
    }
}

/// Static compression curve: gain reduction in dB for an input level in dB.
#[inline]
pub fn compress_db(level_db: f32, threshold_db: f32, ratio: f32, knee_db: f32) -> f32 {
    let ratio = ratio.max(1.0);
    let over = level_db - threshold_db;
    let slope = 1.0 - 1.0 / ratio;
    if knee_db > 0.0 && over.abs() < knee_db * 0.5 {
        let x = over + knee_db * 0.5;
        -slope * x * x / (2.0 * knee_db)
    } else if over > 0.0 {
        -slope * over
    } else {
        0.0
    }
}

/// Simple sine/triangle/square/saw/random LFO.
#[derive(Clone, Copy, Debug, Default)]
pub struct Lfo {
    pub phase: f32,
    pub freq: f32,
    rand_state: u32,
    rand_value: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Wave {
    Sine,
    Triangle,
    Square,
    Saw,
    Random,
}

impl Lfo {
    pub fn with_freq(freq: f32) -> Self {
        Self { freq, ..Default::default() }
    }

    #[inline]
    pub fn advance(&mut self, sr: f32) {
        self.phase += self.freq / sr;
        if self.phase >= 1.0 {
            self.phase -= self.phase.floor();
            self.rand_state = self.rand_state.wrapping_mul(1664525).wrapping_add(1013904223);
            self.rand_value = (self.rand_state >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0;
        }
    }

    /// Bipolar output for the given phase offset (in cycles).
    #[inline]
    pub fn value(&self, wave: Wave, offset: f32) -> f32 {
        let p = (self.phase + offset).fract();
        match wave {
            Wave::Sine => (TAU * p).sin(),
            Wave::Triangle => 1.0 - 4.0 * (p - 0.5).abs(),
            Wave::Square => {
                if p < 0.5 {
                    1.0
                } else {
                    -1.0
                }
            }
            Wave::Saw => 2.0 * p - 1.0,
            Wave::Random => self.rand_value,
        }
    }
}

/// Musical note lengths in quarter notes for the sync tables used by the
/// catalog ("1/4d", "1/8t", "4/1", ...).
pub fn note_quarters(label: &str) -> f32 {
    let (body, mult) = if let Some(b) = label.strip_suffix('d') {
        (b, 1.5)
    } else if let Some(b) = label.strip_suffix('t') {
        (b, 2.0 / 3.0)
    } else {
        (label, 1.0)
    };
    let (n, d) = body.split_once('/').unwrap_or((body, "1"));
    let n: f32 = n.trim().parse().unwrap_or(1.0);
    let d: f32 = d.trim().parse().unwrap_or(4.0);
    4.0 * n / d * mult
}

/// Seconds for a note label at the given tempo.
pub fn note_seconds(label: &str, bpm: f32) -> f32 {
    note_quarters(label) * 60.0 / bpm.max(20.0)
}

/// Deterministic white noise.
#[derive(Clone, Copy, Debug)]
pub struct Noise(u32);

impl Default for Noise {
    fn default() -> Self {
        Noise(0x1234_5678)
    }
}

impl Noise {
    #[inline]
    pub fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        (self.0 as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

/// 2x oversampler using polyphase IIR halfband filters (HIIR-style allpass
/// pairs, 8 coefficients, ~0.05 transition band, >75 dB rejection).
#[derive(Clone, Debug)]
pub struct Oversampler2x {
    up: Halfband,
    down: Halfband,
}

impl Default for Oversampler2x {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug)]
struct Halfband {
    even: Vec<AllpassSection>,
    odd: Vec<AllpassSection>,
}

#[derive(Clone, Copy, Debug, Default)]
struct AllpassSection {
    c: f32,
    x1: f32,
    y1: f32,
}

impl AllpassSection {
    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        let y = (x - self.y1) * self.c + self.x1;
        self.x1 = x;
        self.y1 = y;
        y
    }
}

/// Elliptic halfband allpass coefficients (Valenzuela & Constantinides, as in HIIR).
fn halfband_coefs(n: usize, tbw: f64) -> Vec<f64> {
    use std::f64::consts::PI;
    let k = ((1.0 - 2.0 * tbw) * PI / 4.0).tan().powi(2);
    let kksqrt = (1.0 - k * k).powf(0.25);
    let e = 0.5 * (1.0 - kksqrt) / (1.0 + kksqrt);
    let e4 = e.powi(4);
    let q = e * (1.0 + e4 * (2.0 + e4 * (15.0 + 150.0 * e4)));
    let order = (n * 2 + 1) as f64;
    (0..n)
        .map(|index| {
            let c = (index + 1) as f64;
            let mut num = 0.0;
            let (mut i, mut j) = (0i32, 1.0);
            loop {
                let t = q.powi(i * (i + 1)) * ((2 * i + 1) as f64 * c * PI / order).sin() * j;
                num += t;
                j = -j;
                i += 1;
                if t.abs() < 1e-100 || i > 100 {
                    break;
                }
            }
            num *= q.powf(0.25);
            let mut den = 0.0;
            let (mut i, mut j) = (1i32, -1.0);
            loop {
                let t = q.powi(i * i) * ((2 * i) as f64 * c * PI / order).cos() * j;
                den += t;
                j = -j;
                i += 1;
                if t.abs() < 1e-100 || i > 100 {
                    break;
                }
            }
            den += 0.5;
            let ww = num / den;
            let wwsq = ww * ww;
            let x = ((1.0 - wwsq * k) * (1.0 - wwsq / k)).sqrt() / (1.0 + wwsq);
            (1.0 - x) / (1.0 + x)
        })
        .collect()
}

impl Halfband {
    fn new() -> Self {
        let c = halfband_coefs(8, 0.05);
        let sect = |v: f64| AllpassSection { c: v as f32, x1: 0.0, y1: 0.0 };
        Halfband {
            even: c.iter().step_by(2).map(|&v| sect(v)).collect(),
            odd: c.iter().skip(1).step_by(2).map(|&v| sect(v)).collect(),
        }
    }

    #[inline]
    fn path_even(&mut self, x: f32) -> f32 {
        self.even.iter_mut().fold(x, |v, s| s.process(v))
    }

    #[inline]
    fn path_odd(&mut self, x: f32) -> f32 {
        self.odd.iter_mut().fold(x, |v, s| s.process(v))
    }
}

impl Oversampler2x {
    pub fn new() -> Self {
        Self { up: Halfband::new(), down: Halfband::new() }
    }

    /// Process one input sample through `f` at twice the rate.
    #[inline]
    pub fn process(&mut self, x: f32, mut f: impl FnMut(f32) -> f32) -> f32 {
        let u0 = self.up.path_even(x);
        let u1 = self.up.path_odd(x);
        let y0 = f(u0);
        let y1 = f(u1);
        0.5 * (self.down.path_even(y1) + self.down.path_odd(y0))
    }
}

/// Second order Butterworth helper for quick tone filters.
pub fn lowpass(sr: f32, f: f32) -> Coefs {
    Coefs::rbj(BiquadKind::LowPass, sr, f, std::f32::consts::FRAC_1_SQRT_2, 0.0)
}

pub fn highpass(sr: f32, f: f32) -> Coefs {
    Coefs::rbj(BiquadKind::HighPass, sr, f, std::f32::consts::FRAC_1_SQRT_2, 0.0)
}

pub fn peak(sr: f32, f: f32, q: f32, db: f32) -> Coefs {
    Coefs::rbj(BiquadKind::Peak, sr, f, q, db)
}

pub fn low_shelf(sr: f32, f: f32, db: f32) -> Coefs {
    Coefs::rbj(BiquadKind::LowShelf, sr, f, std::f32::consts::FRAC_1_SQRT_2, db)
}

pub fn high_shelf(sr: f32, f: f32, db: f32) -> Coefs {
    Coefs::rbj(BiquadKind::HighShelf, sr, f, std::f32::consts::FRAC_1_SQRT_2, db)
}

/// Mid/side helpers.
#[inline]
pub fn to_ms(l: f32, r: f32) -> (f32, f32) {
    ((l + r) * 0.5, (l - r) * 0.5)
}

#[inline]
pub fn from_ms(m: f32, s: f32) -> (f32, f32) {
    (m + s, m - s)
}
