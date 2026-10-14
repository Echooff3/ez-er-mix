//! Drive pedals, saturation and lo-fi processors.

use crate::dsp::*;
use crate::effect::{idx, on, Ctx, Effect};

/// Stereo pair of 2x oversamplers.
#[derive(Clone, Default)]
pub struct Os2 {
    pub l: Oversampler2x,
    pub r: Oversampler2x,
}

impl Os2 {
    pub fn new() -> Self {
        Self { l: Oversampler2x::new(), r: Oversampler2x::new() }
    }
}

// ---------------------------------------------------------------------------
// Pedals (TA_DIST, TA_CENTAUR, TA_BIGM, TA_FUZZRACE, TA_METALPLANET)

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PedalKind {
    Screamer,
    Klon,
    Muff,
    Fuzz,
    Metal,
}

pub struct Pedal {
    sr: f32,
    kind: PedalKind,
    drive: f32,
    tone: f32,
    level: f32,
    low: f32,
    mid: f32,
    high: f32,
    pre: [Biquad; 2],
    clip_hp: [Biquad; 2],
    tone_lp: [Biquad; 2],
    tone_hp: [Biquad; 2],
    eq: [StereoBiquad; 3],
    os: Os2,
    dc: [DcBlock; 2],
    dirty: bool,
}

impl Pedal {
    pub fn new(kind: PedalKind) -> Self {
        Self {
            sr: 48000.0,
            kind,
            drive: 6.0,
            tone: 5.0,
            level: 7.0,
            low: 5.0,
            mid: 5.0,
            high: 5.0,
            pre: Default::default(),
            clip_hp: Default::default(),
            tone_lp: Default::default(),
            tone_hp: Default::default(),
            eq: Default::default(),
            os: Os2::new(),
            dc: Default::default(),
            dirty: true,
        }
    }

    fn update(&mut self) {
        let sr = self.sr;
        let t = knob10(self.tone);
        for ch in 0..2 {
            match self.kind {
                PedalKind::Screamer => {
                    self.pre[ch].set(Coefs::one_pole_hp(sr, 30.0));
                    self.clip_hp[ch].set(Coefs::one_pole_hp(sr, 720.0));
                    self.tone_lp[ch].set(Coefs::one_pole_lp(sr, lerp(700.0, 6500.0, t * t)));
                    self.tone_hp[ch].set(Coefs::IDENTITY);
                }
                PedalKind::Klon => {
                    self.pre[ch].set(Coefs::one_pole_hp(sr, 40.0));
                    self.clip_hp[ch].set(Coefs::one_pole_hp(sr, 400.0));
                    self.tone_lp[ch].set(high_shelf(sr, 1500.0, lerp(-8.0, 8.0, t)));
                    self.tone_hp[ch].set(Coefs::IDENTITY);
                }
                PedalKind::Muff => {
                    self.pre[ch].set(Coefs::one_pole_hp(sr, 60.0));
                    self.clip_hp[ch].set(Coefs::one_pole_lp(sr, 5000.0));
                    self.tone_lp[ch].set(Coefs::one_pole_lp(sr, 480.0));
                    self.tone_hp[ch].set(Coefs::one_pole_hp(sr, 1100.0));
                }
                PedalKind::Fuzz => {
                    self.pre[ch].set(Coefs::one_pole_hp(sr, 80.0));
                    self.clip_hp[ch].set(Coefs::one_pole_lp(sr, 7000.0));
                    self.tone_lp[ch].set(Coefs::one_pole_lp(sr, lerp(1500.0, 9000.0, t)));
                    self.tone_hp[ch].set(Coefs::IDENTITY);
                }
                PedalKind::Metal => {
                    self.pre[ch].set(Coefs::one_pole_hp(sr, 120.0));
                    self.clip_hp[ch].set(peak(sr, 1200.0, 0.8, 6.0));
                    self.tone_lp[ch].set(lowpass(sr, 6500.0));
                    self.tone_hp[ch].set(Coefs::IDENTITY);
                }
            }
        }
        if self.kind == PedalKind::Metal {
            self.eq[0].set(low_shelf(sr, 120.0, (self.low - 5.0) * 3.0));
            self.eq[1].set(peak(sr, 800.0, 0.8, (self.mid - 5.0) * 3.0));
            self.eq[2].set(high_shelf(sr, 3500.0, (self.high - 5.0) * 3.0));
        }
        self.dirty = false;
    }
}

impl Effect for Pedal {
    fn prepare(&mut self, sr: f32) {
        self.sr = sr;
        for d in &mut self.dc {
            d.prepare(sr);
        }
        self.dirty = true;
    }

    fn set(&mut self, name: &str, v: f32) {
        match name {
            "drive" => self.drive = v,
            "tone" => self.tone = v,
            "level" => self.level = v,
            "low" => self.low = v,
            "mid" => self.mid = v,
            "high" => self.high = v,
            _ => return,
        }
        self.dirty = true;
    }

    fn process(&mut self, _: &Ctx, l: &mut [f32], r: &mut [f32]) {
        if self.dirty {
            self.update();
        }
        let d = knob10(self.drive);
        let kind = self.kind;
        let gain = match kind {
            PedalKind::Screamer => db_to_gain(6.0 + 40.0 * d),
            PedalKind::Klon => db_to_gain(3.0 + 30.0 * d),
            PedalKind::Muff => db_to_gain(25.0 + 30.0 * d),
            PedalKind::Fuzz => db_to_gain(15.0 + 30.0 * d),
            PedalKind::Metal => db_to_gain(25.0 + 35.0 * d),
        };
        // Level knob: 0 dB around 7 for the screamer family, 8 for fuzzes.
        let unity = if matches!(kind, PedalKind::Muff | PedalKind::Fuzz | PedalKind::Metal) { 8.0 } else { 7.0 };
        // About 2 dB per step around the unity position; 0 is quiet, not mute.
        let out = db_to_gain((self.level - unity) * 2.0);
        let t = knob10(self.tone);
        for ch in 0..2 {
            let s = if ch == 0 { &mut *l } else { &mut *r };
            let os = if ch == 0 { &mut self.os.l } else { &mut self.os.r };
            for x in s.iter_mut() {
                let v = self.pre[ch].process(*x);
                let y = match kind {
                    PedalKind::Screamer => {
                        let h = self.clip_hp[ch].process(v);
                        let c = os.process(h * gain, |u| shape::diode(u) * 0.5);
                        self.tone_lp[ch].process(v + c)
                    }
                    PedalKind::Klon => {
                        let h = self.clip_hp[ch].process(v);
                        let c = os.process(h * gain, |u| shape::tanh(u) * 0.6);
                        // Clean blend fades as gain rises.
                        let clean = v * lerp(1.0, 0.25, d);
                        self.tone_lp[ch].process(clean + c)
                    }
                    PedalKind::Muff => {
                        let c = os.process(v * gain, |u| shape::tanh(shape::tanh(u) * 6.0) * 0.5);
                        let c = self.clip_hp[ch].process(c);
                        let lo = self.tone_lp[ch].process(c);
                        let hi = self.tone_hp[ch].process(c);
                        lerp(lo, hi, t) * 1.4
                    }
                    PedalKind::Fuzz => {
                        let c = os.process(v * gain, |u| {
                            // Germanium-ish asymmetric clipping.
                            if u > 0.0 { shape::tanh(u * 0.8) } else { shape::tanh(u * 1.6) * 0.6 }
                        });
                        let c = self.dc[ch].process(c);
                        self.tone_lp[ch].process(self.clip_hp[ch].process(c)) * 0.6
                    }
                    PedalKind::Metal => {
                        let pre = self.clip_hp[ch].process(v);
                        let c = os.process(pre * gain, |u| shape::hard(shape::tanh(u) * 1.2, 1.0) * 0.5);
                        let mut c = self.tone_lp[ch].process(c);
                        for e in &mut self.eq {
                            c = if ch == 0 { e.l.process(c) } else { e.r.process(c) };
                        }
                        c
                    }
                };
                *x = y * out;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Valve distortion box (TA_CULTURE_VULTURE)

pub struct ValveBox {
    sr: f32,
    drive: f32,
    overdrive: bool,
    dist_type: usize,
    bias: f32,
    filter: usize,
    level: f32,
    outdamp: bool,
    lp: [Biquad; 2],
    dc: [DcBlock; 2],
    os: Os2,
}

impl ValveBox {
    pub fn new() -> Self {
        Self {
            sr: 48000.0,
            drive: 5.0,
            overdrive: false,
            dist_type: 0,
            bias: 3.0,
            filter: 0,
            level: 0.0,
            outdamp: false,
            lp: Default::default(),
            dc: Default::default(),
            os: Os2::new(),
        }
    }

    fn update(&mut self) {
        let f: f32 = match self.filter {
            1 => 7000.0,
            2 => 4000.0,
            _ => 20000.0,
        };
        let f = if self.outdamp { f.min(9000.0) } else { f };
        for b in &mut self.lp {
            b.set(if f < self.sr * 0.45 { lowpass(self.sr, f) } else { Coefs::IDENTITY });
        }
    }
}

impl Effect for ValveBox {
    fn prepare(&mut self, sr: f32) {
        self.sr = sr;
        for d in &mut self.dc {
            d.prepare(sr);
        }
        self.update();
    }

    fn set(&mut self, name: &str, v: f32) {
        match name {
            "drive" => self.drive = v,
            "overdrive" => self.overdrive = on(v),
            "dist_type" => self.dist_type = idx(v),
            "bias" => self.bias = v,
            "filter" => self.filter = idx(v),
            "level" => self.level = v,
            "outdamp" => self.outdamp = on(v),
            _ => return,
        }
        self.update();
    }

    fn process(&mut self, _: &Ctx, l: &mut [f32], r: &mut [f32]) {
        let g = db_to_gain(self.drive + if self.overdrive { 12.0 } else { 0.0 });
        // T: triode (even), P1/P2: pentode (odd, harder), MD/OV: increasingly
        // asymmetric "starved" modes.
        let t = self.dist_type;
        let bias = (self.bias / 11.0) * match t {
            0 => 0.6,
            1 | 2 => 0.2,
            _ => 0.9,
        };
        let hard = match t {
            1 => 1.5,
            2 => 2.2,
            3..=7 => 1.0 + 0.3 * (t - 3) as f32,
            8 | 9 => 3.0,
            _ => 1.0,
        };
        let out = db_to_gain(self.level) / g.sqrt().max(1.0);
        for ch in 0..2 {
            let s = if ch == 0 { &mut *l } else { &mut *r };
            let os = if ch == 0 { &mut self.os.l } else { &mut self.os.r };
            for x in s.iter_mut() {
                let y = os.process(*x * g, |u| shape::tube(u * hard, bias) / hard.sqrt());
                *x = self.lp[ch].process(self.dc[ch].process(y)) * out;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tape (TA_TOONTRACK_TAPESIM, TA_TAPEEMU)

pub struct Tape {
    sr: f32,
    emu: bool,
    drive: f32,
    bias: f32,
    speed_ips: f32,
    noise: f32,
    wow: f32,
    level: f32,
    play: f32,
    over_bias: bool,
    head: [Biquad; 2],
    hf: [Biquad; 2],
    pre: [Biquad; 2],
    de: [Biquad; 2],
    line: [DelayLine; 2],
    lfo: Lfo,
    flutter: Lfo,
    rng: Noise,
}

impl Tape {
    pub fn new(emu: bool) -> Self {
        Self {
            sr: 48000.0,
            emu,
            drive: if emu { 0.0 } else { 10.0 },
            bias: 50.0,
            speed_ips: 15.0,
            noise: -1000.0,
            wow: 0.0,
            level: 0.0,
            play: 0.0,
            over_bias: false,
            head: Default::default(),
            hf: Default::default(),
            pre: Default::default(),
            de: Default::default(),
            line: [DelayLine::new(4096), DelayLine::new(4096)],
            lfo: Lfo::default(),
            flutter: Lfo::default(),
            rng: Noise::default(),
        }
    }

    fn update(&mut self) {
        let sr = self.sr;
        let s = self.speed_ips.clamp(3.75, 30.0);
        // Head bump moves up with speed; HF loss moves down with speed and
        // with bias.
        let bump = 30.0 * s / 7.5;
        let bias = if self.emu { if self.over_bias { 70.0 } else { 50.0 } } else { self.bias };
        let hf = (12000.0 * (s / 15.0).powf(0.7) * lerp(1.4, 0.7, bias / 100.0)).min(sr * 0.45);
        for ch in 0..2 {
            self.head[ch].set(peak(sr, bump, 1.2, 2.5));
            self.hf[ch].set(lowpass(sr, hf));
            // Record pre-emphasis / playback de-emphasis around the saturator.
            self.pre[ch].set(high_shelf(sr, 3000.0, 6.0));
            self.de[ch].set(high_shelf(sr, 3000.0, -6.0));
        }
        self.lfo.freq = 0.6;
        self.flutter.freq = 7.0;
    }
}

impl Effect for Tape {
    fn prepare(&mut self, sr: f32) {
        self.sr = sr;
        self.update();
    }

    fn set(&mut self, name: &str, v: f32) {
        match name {
            "drive" | "rec-level" => self.drive = v,
            "bias" => self.bias = v,
            "speed" => self.speed_ips = v,
            "tape-speed" => self.speed_ips = [15.0, 7.5, 30.0][idx(v).min(2)],
            "tape-bias" => self.over_bias = on(v),
            "noise" => self.noise = v,
            "wow" | "wow-flut" => self.wow = v,
            "level" => self.level = v,
            "play-level" => self.play = v,
            _ => return,
        }
        self.update();
    }

    fn process(&mut self, _: &Ctx, l: &mut [f32], r: &mut [f32]) {
        // Drive in dB relative to a nominal +10 operating level for TAPESIM.
        let drive_db = if self.emu { self.drive + 6.0 } else { self.drive - 4.0 };
        let g = db_to_gain(drive_db);
        let out = db_to_gain(if self.emu { self.play } else { self.level }) / g.max(0.25);
        let noise = db_to_gain(self.noise);
        let wow = self.wow / 100.0;
        let base = 0.002 * self.sr;
        for i in 0..l.len() {
            self.lfo.advance(self.sr);
            self.flutter.advance(self.sr);
            let mod_d = base * (1.0 + wow * (0.6 * self.lfo.value(Wave::Sine, 0.0) + 0.15 * self.flutter.value(Wave::Sine, 0.0)));
            for ch in 0..2 {
                let x = if ch == 0 { l[i] } else { r[i] };
                let e = self.pre[ch].process(x * g);
                let sat = shape::tanh(e * 0.9) / 0.9;
                let mut y = self.de[ch].process(sat);
                y = self.head[ch].process(y);
                y = self.hf[ch].process(y);
                if wow > 0.0 {
                    self.line[ch].push(y);
                    y = self.line[ch].read(mod_d);
                }
                y += self.rng.next() * noise;
                if ch == 0 {
                    l[i] = y * out;
                } else {
                    r[i] = y * out;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Bit crusher (TA_BIT_CRUSH, TA_BIT_CRUSH2)

pub struct BitCrush {
    sr: f32,
    v2: bool,
    bits: f32,
    rate: f32,
    antialias: bool,
    drive: f32,
    phase: f32,
    held: [f32; 2],
    aa: [Butterworth; 2],
}

impl BitCrush {
    pub fn new(v2: bool) -> Self {
        Self {
            sr: 48000.0,
            v2,
            bits: 24.0,
            rate: 22050.0,
            antialias: false,
            drive: 0.0,
            phase: 0.0,
            held: [0.0; 2],
            aa: Default::default(),
        }
    }
}

impl Effect for BitCrush {
    fn prepare(&mut self, sr: f32) {
        self.sr = sr;
        self.set("downsample", self.rate);
    }

    fn set(&mut self, name: &str, v: f32) {
        match name {
            "bit" => {
                let i = idx(v);
                self.bits = if self.v2 {
                    if i >= 23 { 32.0 } else { (i + 2) as f32 }
                } else {
                    [2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0][i.min(6)]
                };
            }
            "downsample" => {
                self.rate = v;
                for f in &mut self.aa {
                    f.design(self.sr, (v * 0.45).min(self.sr * 0.45), 4, false);
                }
            }
            "antialias" => self.antialias = on(v),
            "drive" => self.drive = v,
            _ => {}
        }
    }

    fn process(&mut self, _: &Ctx, l: &mut [f32], r: &mut [f32]) {
        let g = db_to_gain(self.drive * 2.0);
        let q = if self.bits >= 32.0 { 0.0 } else { 2f32.powf(self.bits - 1.0) };
        let step = (self.rate / self.sr).min(1.0);
        for i in 0..l.len() {
            let mut x = [l[i] * g, r[i] * g];
            if self.drive > 0.0 {
                x = [shape::tanh(x[0]), shape::tanh(x[1])];
            }
            if self.antialias {
                x = [self.aa[0].process(x[0]), self.aa[1].process(x[1])];
            }
            self.phase += step;
            if self.phase >= 1.0 {
                self.phase -= 1.0;
                for ch in 0..2 {
                    self.held[ch] = if q > 0.0 { (x[ch] * q).round() / q } else { x[ch] };
                }
            }
            l[i] = self.held[0];
            r[i] = self.held[1];
        }
    }
}

// ---------------------------------------------------------------------------
// Multiband distortion (TA_MULTIBANDDIST)

pub struct MultibandDist {
    sr: f32,
    freqs: [f32; 3],
    drive: [f32; 4],
    shape: [f32; 4],
    bias: [f32; 4],
    level: [f32; 4],
    master_gain: f32,
    master_level: f32,
    parallel: f32,
    split: [FourBandSplit; 2],
    dirty: bool,
}

impl MultibandDist {
    pub fn new() -> Self {
        Self {
            sr: 48000.0,
            freqs: [100.0, 800.0, 5000.0],
            drive: [0.0; 4],
            shape: [20.0; 4],
            bias: [0.0; 4],
            level: [0.0; 4],
            master_gain: 0.0,
            master_level: 0.0,
            parallel: 100.0,
            split: Default::default(),
            dirty: true,
        }
    }
}

impl Effect for MultibandDist {
    fn prepare(&mut self, sr: f32) {
        self.sr = sr;
        self.dirty = true;
    }

    fn set(&mut self, name: &str, v: f32) {
        self.dirty = true;
        match name {
            "freq0" => self.freqs[0] = v,
            "freq1" => self.freqs[1] = v,
            "freq2" => self.freqs[2] = v,
            "master_gain" => self.master_gain = v,
            "master_level" => self.master_level = v,
            "parallel" => self.parallel = v,
            _ => {
                let Some(c) = name.chars().next().and_then(|c| c.to_digit(10)) else { return };
                let i = (c as usize).min(3);
                match &name[1..] {
                    "drive" => self.drive[i] = v,
                    "shape" => self.shape[i] = v,
                    "bias" => self.bias[i] = v,
                    "Level" => self.level[i] = v,
                    _ => {}
                }
            }
        }
    }

    fn process(&mut self, _: &Ctx, l: &mut [f32], r: &mut [f32]) {
        if self.dirty {
            for s in &mut self.split {
                s.set(self.sr, self.freqs);
            }
            self.dirty = false;
        }
        let gin = db_to_gain(self.master_gain);
        let gout = db_to_gain(self.master_level);
        let drive: [f32; 4] = std::array::from_fn(|i| db_to_gain(self.drive[i]));
        let lv: [f32; 4] = std::array::from_fn(|i| db_to_gain(self.level[i]));
        let wet = self.parallel / 100.0;
        for (a, b) in l.iter_mut().zip(r.iter_mut()) {
            let (x, y) = (*a * gin, *b * gin);
            let bl = self.split[0].split(x);
            let br = self.split[1].split(y);
            let (mut ox, mut oy) = (0.0, 0.0);
            for i in 0..4 {
                let k = 1.0 + self.shape[i] / 20.0;
                let bias = self.bias[i] / 100.0;
                let f = |v: f32| shape::tube(v * drive[i] * k, bias) / (k * drive[i].sqrt());
                ox += f(bl[i]) * lv[i];
                oy += f(br[i]) * lv[i];
            }
            *a = lerp(x, ox, wet) * gout;
            *b = lerp(y, oy, wet) * gout;
        }
    }
}

// ---------------------------------------------------------------------------
// Console emulation (TA_CONSOLEEMU)

pub struct Console {
    sr: f32,
    model: usize,
    trim: f32,
    mic_pre: bool,
    mix: f32,
    tolerances: bool,
    tilt: [StereoBiquad; 2],
}

impl Console {
    pub fn new() -> Self {
        Self { sr: 48000.0, model: 0, trim: 0.0, mic_pre: false, mix: 0.0, tolerances: false, tilt: Default::default() }
    }

    fn update(&mut self) {
        // S4000: tight lows / forward highs; N80: thick lows, silky top;
        // T88: round mids.
        let (lo, hi) = match self.model {
            0 => (-0.5, 0.8),
            1 => (1.2, -0.4),
            _ => (0.6, -0.8),
        };
        self.tilt[0].set(low_shelf(self.sr, 80.0, lo));
        self.tilt[1].set(high_shelf(self.sr, 10000.0, hi));
    }
}

impl Effect for Console {
    fn prepare(&mut self, sr: f32) {
        self.sr = sr;
        self.update();
    }

    fn set(&mut self, name: &str, v: f32) {
        match name {
            "console-model" => self.model = idx(v),
            "input-trim" => self.trim = v,
            "mic-pre" => self.mic_pre = on(v),
            "mix-level" => self.mix = v,
            "tolerances" => self.tolerances = on(v),
            _ => return,
        }
        self.update();
    }

    fn process(&mut self, _: &Ctx, l: &mut [f32], r: &mut [f32]) {
        let drive = db_to_gain(self.trim + if self.mic_pre { 8.0 } else { 0.0 });
        let out = db_to_gain(self.mix);
        let even = [0.05, 0.12, 0.08][self.model.min(2)];
        let tol = if self.tolerances { [1.0, 1.04] } else { [1.0, 1.0] };
        for (a, b) in l.iter_mut().zip(r.iter_mut()) {
            let (mut x, mut y) = (*a * drive * tol[0], *b * drive * tol[1]);
            x = shape::tube(x * 0.5, even) * 2.0 / drive.max(1.0);
            y = shape::tube(y * 0.5, even) * 2.0 / drive.max(1.0);
            for t in &mut self.tilt {
                (x, y) = t.process(x, y);
            }
            *a = x * out;
            *b = y * out;
        }
    }
}

// ---------------------------------------------------------------------------
// Preamps: TC Pre (TA_TCPRE), Rhodes preamp (TA_RHODESPRE), Tech21 bass
// driver (TA_T21BASSDRIVE), Alpha-Omega (TA_ALPHAOMEGA)

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PreampKind {
    TcPre,
    Rhodes,
    BassDriver,
    AlphaOmega,
}

pub struct Preamp {
    sr: f32,
    kind: PreampKind,
    model: usize,
    gain: f32,
    bass: f32,
    mid: f32,
    treble: f32,
    presence: f32,
    blend: f32,
    level: f32,
    gaincomp: f32,
    growl: bool,
    bite: bool,
    trem_mode: bool,
    trem_speed: f32,
    trem_intens: f32,
    eq: [StereoBiquad; 4],
    pre_hp: StereoBiquad,
    os: Os2,
    lfo: Lfo,
}

impl Preamp {
    pub fn new(kind: PreampKind) -> Self {
        Self {
            sr: 48000.0,
            kind,
            model: 0,
            gain: 5.0,
            bass: 5.0,
            mid: 5.0,
            treble: 5.0,
            presence: 5.0,
            blend: 8.0,
            level: if kind == PreampKind::AlphaOmega { 5.0 } else { 7.0 },
            gaincomp: if kind == PreampKind::BassDriver { 12.0 } else { 0.0 },
            growl: false,
            bite: false,
            trem_mode: false,
            trem_speed: 5.0,
            trem_intens: 0.0,
            eq: Default::default(),
            pre_hp: StereoBiquad::default(),
            os: Os2::new(),
            lfo: Lfo::default(),
        }
    }

    fn update(&mut self) {
        let sr = self.sr;
        let k = |v: f32, range: f32| (v - 5.0) / 5.0 * range;
        let (bf, mf, tf) = match self.kind {
            PreampKind::TcPre => (100.0, 1000.0, 4000.0),
            PreampKind::Rhodes => (120.0, 800.0, 3000.0),
            PreampKind::BassDriver => (80.0, 500.0, 3000.0),
            PreampKind::AlphaOmega => (100.0, 600.0, 2500.0),
        };
        self.eq[0].set(low_shelf(sr, bf * if self.growl { 1.3 } else { 1.0 }, k(self.bass, 12.0) + if self.growl { 4.0 } else { 0.0 }));
        self.eq[1].set(peak(sr, mf, 0.8, k(self.mid, 12.0)));
        self.eq[2].set(high_shelf(sr, tf, k(self.treble, 12.0) + if self.bite { 4.0 } else { 0.0 }));
        self.eq[3].set(high_shelf(sr, 5000.0, k(self.presence, 9.0)));
        self.pre_hp.set(Coefs::one_pole_hp(sr, if self.kind == PreampKind::TcPre { 40.0 } else { 25.0 }));
        self.lfo.freq = lerp(1.0, 9.0, knob10(self.trem_speed));
    }
}

impl Effect for Preamp {
    fn prepare(&mut self, sr: f32) {
        self.sr = sr;
        self.update();
    }

    fn set(&mut self, name: &str, v: f32) {
        match name {
            "model" => self.model = idx(v),
            "pre_gain" | "drive" => self.gain = v,
            "bass" => self.bass = v,
            "mid" => self.mid = v,
            "treble" => self.treble = v,
            "presence" => self.presence = v,
            "blend" => self.blend = v,
            "level" | "volume" => self.level = v,
            "gaincomp" => self.gaincomp = v,
            "growl" => self.growl = on(v),
            "bite" => self.bite = on(v),
            "mode" => self.trem_mode = on(v),
            "speed" => self.trem_speed = v,
            "intens" => self.trem_intens = v,
            _ => return,
        }
        self.update();
    }

    fn process(&mut self, _: &Ctx, l: &mut [f32], r: &mut [f32]) {
        let drive = match self.kind {
            PreampKind::TcPre => db_to_gain(self.gain * 2.0 + [0.0, 3.0, -2.0][self.model.min(2)]),
            PreampKind::Rhodes => db_to_gain(knob10(self.level) * 12.0 - 6.0),
            PreampKind::BassDriver | PreampKind::AlphaOmega => db_to_gain(knob10(self.gain) * 36.0),
        };
        let blend = match self.kind {
            PreampKind::BassDriver | PreampKind::AlphaOmega => knob10(self.blend),
            _ => 1.0,
        };
        let out = match self.kind {
            PreampKind::TcPre => db_to_gain((self.level - 8.0) * 3.0),
            PreampKind::Rhodes => 1.0,
            _ => db_to_gain((self.level - 5.0) * 3.0 - self.gaincomp + 6.0) / drive.sqrt(),
        };
        let trem = knob10(self.trem_intens);
        let n = l.len();
        for i in 0..n {
            self.lfo.advance(self.sr);
            let m = self.lfo.value(Wave::Sine, 0.0);
            for ch in 0..2 {
                let x = if ch == 0 { l[i] } else { r[i] };
                let os = if ch == 0 { &mut self.os.l } else { &mut self.os.r };
                let pre = if ch == 0 { self.pre_hp.l.process(x) } else { self.pre_hp.r.process(x) };
                let dirty = os.process(pre * drive, |u| shape::tube(u, 0.15));
                let mut y = lerp(x * drive.min(4.0), dirty, blend);
                for e in &mut self.eq {
                    y = if ch == 0 { e.l.process(y) } else { e.r.process(y) };
                }
                if self.kind == PreampKind::Rhodes && trem > 0.0 {
                    // Stereo pan tremolo (mode on) or amplitude tremolo.
                    let side = if ch == 0 { 1.0 } else { -1.0 };
                    let g = if self.trem_mode { 1.0 + trem * m * side } else { 1.0 - trem * 0.5 * (1.0 + m) };
                    y *= g;
                }
                if ch == 0 {
                    l[i] = y * out;
                } else {
                    r[i] = y * out;
                }
            }
        }
    }
}
