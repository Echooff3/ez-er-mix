//! Time-varying effects.

use crate::dsp::*;
use crate::effect::{idx, on, Ctx, Effect, SyncParam, RATE_NOTES, RATE_NOTES_LONG};

/// Two-tap crossfading delay pitch shifter.
#[derive(Clone)]
pub struct PitchShifter {
    line: DelayLine,
    phase: f32,
    window: f32,
}

impl Default for PitchShifter {
    fn default() -> Self {
        Self { line: DelayLine::new(8192), phase: 0.0, window: 2400.0 }
    }
}

impl PitchShifter {
    pub fn prepare(&mut self, sr: f32) {
        self.window = (0.05 * sr).min(4000.0);
        self.line.resize((self.window as usize) * 2 + 8);
    }

    #[inline]
    pub fn process(&mut self, x: f32, ratio: f32) -> f32 {
        self.line.push(x);
        if (ratio - 1.0).abs() < 1e-4 {
            return x;
        }
        self.phase = (self.phase + (1.0 - ratio) / self.window).rem_euclid(1.0);
        let p2 = (self.phase + 0.5).fract();
        let w1 = (std::f32::consts::PI * self.phase).sin().powi(2);
        let w2 = (std::f32::consts::PI * p2).sin().powi(2);
        w1 * self.line.read(1.0 + self.phase * self.window) + w2 * self.line.read(1.0 + p2 * self.window)
    }
}

/// Modulated delay voice shared by the chorus family.
#[derive(Clone)]
struct Voice {
    line: DelayLine,
}

impl Voice {
    fn new() -> Self {
        Self { line: DelayLine::new(1 << 13) }
    }

    #[inline]
    fn run(&mut self, x: f32, delay: f32) -> f32 {
        self.line.push(x);
        self.line.read(delay.max(1.0))
    }
}

// ---------------------------------------------------------------------------
// Chorus (TA_TOONTRACK_CHORUS), String-machine ensemble (TA_SOLINA_MOD),
// Tri-chorus (TA_TTTRICHORUS), Dimension (TA_DIMENSIOND), bass chorus
// (TA_CEB3)

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ChorusKind {
    Basic,
    Solina,
    Tri,
    Dimension,
    BassChorus,
}

pub struct Chorus {
    sr: f32,
    kind: ChorusKind,
    speed: SyncParam,
    depth: f32,
    mix: f32,
    stereo: f32,
    level_db: f32,
    voice_depth: [f32; 3],
    buttons: [bool; 4],
    filter: f32,
    voices: [[Voice; 3]; 2],
    lfo: Lfo,
    lfo2: Lfo,
    split: [Crossover; 2],
}

impl Chorus {
    pub fn new(kind: ChorusKind) -> Self {
        let (speed, depth, mix) = match kind {
            ChorusKind::Basic => (0.5, 50.0, 65.0),
            ChorusKind::Solina => (0.6, 75.0, 100.0),
            ChorusKind::Tri => (5.0, 50.0, 100.0),
            ChorusKind::Dimension => (0.25, 0.0, 100.0),
            ChorusKind::BassChorus => (1.0, 50.0, 50.0),
        };
        Self {
            sr: 48000.0,
            kind,
            speed: SyncParam::rate(speed, RATE_NOTES_LONG),
            depth,
            mix,
            stereo: 50.0,
            level_db: 0.0,
            voice_depth: [50.0; 3],
            buttons: [false; 4],
            filter: 50.0,
            voices: std::array::from_fn(|_| std::array::from_fn(|_| Voice::new())),
            lfo: Lfo::default(),
            lfo2: Lfo::default(),
            split: Default::default(),
        }
    }
}

impl Effect for Chorus {
    fn prepare(&mut self, sr: f32) {
        self.sr = sr;
        self.set("filter", self.filter);
    }

    fn set(&mut self, name: &str, v: f32) {
        match name {
            "speed" => self.speed.free = v,
            "depth" => self.depth = v,
            "level" if self.kind == ChorusKind::Basic => self.mix = v,
            "level" => self.level_db = v,
            "mix" | "effect" => self.mix = v,
            "stereo" => self.stereo = v,
            "ldepth" => self.voice_depth[0] = v,
            "cdepth" => self.voice_depth[1] = v,
            "rdepth" => self.voice_depth[2] = v,
            "p1" => self.buttons[0] = on(v),
            "p2" => self.buttons[1] = on(v),
            "p3" => self.buttons[2] = on(v),
            "p4" => self.buttons[3] = on(v),
            "filter" => {
                self.filter = v;
                for s in &mut self.split {
                    s.set(self.sr, lerp(80.0, 800.0, v / 100.0));
                }
            }
            _ => {}
        }
    }

    fn set_sync(&mut self, name: &str, unsynced: f32, synced: f32, sync: bool) {
        if name == "speed" {
            self.speed.set(unsynced, synced, sync);
        } else {
            self.set(name, unsynced);
        }
    }

    fn process(&mut self, ctx: &Ctx, l: &mut [f32], r: &mut [f32]) {
        let sr = self.sr;
        let ms = sr / 1000.0;
        let out = db_to_gain(self.level_db);
        match self.kind {
            ChorusKind::Basic | ChorusKind::BassChorus => {
                self.lfo.freq = self.speed.value(ctx.bpm);
                let depth = self.depth / 100.0 * 4.0 * ms;
                let base = 8.0 * ms;
                let wet = self.mix / 100.0;
                let st = self.stereo / 100.0 * 0.5;
                for i in 0..l.len() {
                    self.lfo.advance(sr);
                    let ml = self.lfo.value(Wave::Triangle, 0.0);
                    let mr = self.lfo.value(Wave::Triangle, st);
                    let (x, y) = (l[i], r[i]);
                    if self.kind == ChorusKind::BassChorus {
                        let (lo_l, hi_l) = self.split[0].split(x);
                        let (lo_r, hi_r) = self.split[1].split(y);
                        let cl = self.voices[0][0].run(hi_l, base + depth * ml);
                        let cr = self.voices[1][0].run(hi_r, base + depth * mr);
                        l[i] = lo_l + lerp(hi_l, cl, wet);
                        r[i] = lo_r + lerp(hi_r, cr, wet);
                    } else {
                        let cl = self.voices[0][0].run(x, base + depth * ml);
                        let cr = self.voices[1][0].run(y, base + depth * mr);
                        l[i] = x + (cl - x) * wet * 0.7;
                        r[i] = y + (cr - y) * wet * 0.7;
                    }
                }
            }
            ChorusKind::Solina => {
                self.lfo.freq = 0.6;
                self.lfo2.freq = 6.0;
                let d = self.depth / 100.0;
                let wet = self.mix / 100.0;
                for i in 0..l.len() {
                    self.lfo.advance(sr);
                    self.lfo2.advance(sr);
                    let (x, y) = (l[i], r[i]);
                    let mut acc = [0.0f32; 2];
                    for v in 0..3 {
                        let off = v as f32 / 3.0;
                        let m = 0.7 * self.lfo.value(Wave::Sine, off) + 0.3 * self.lfo2.value(Wave::Sine, off);
                        let delay = (6.0 + 3.5 * d * m) * ms;
                        acc[0] += self.voices[0][v].run(x, delay);
                        acc[1] += self.voices[1][v].run(y, delay + 0.3 * ms);
                    }
                    l[i] = lerp(x, acc[0] / 3.0 * 1.2, wet) * out;
                    r[i] = lerp(y, acc[1] / 3.0 * 1.2, wet) * out;
                }
            }
            ChorusKind::Tri => {
                self.lfo.freq = lerp(0.2, 6.0, knob10(self.speed.free));
                let wet = self.mix / 100.0;
                let st = if self.stereo > 0.5 { 1.0 } else { 0.0 };
                for i in 0..l.len() {
                    self.lfo.advance(sr);
                    let x = (l[i] + r[i]) * 0.5;
                    let mut v = [0.0f32; 3];
                    for k in 0..3 {
                        let m = self.lfo.value(Wave::Sine, k as f32 / 3.0);
                        v[k] = self.voices[0][k].run(x, (7.0 + 3.0 * self.voice_depth[k] / 100.0 * m) * ms);
                    }
                    let left = v[0] + 0.7 * v[1] + (1.0 - st) * v[2];
                    let right = v[2] + 0.7 * v[1] + (1.0 - st) * v[0];
                    l[i] = lerp(l[i], left * 0.5, wet) * out;
                    r[i] = lerp(r[i], right * 0.5, wet) * out;
                }
            }
            ChorusKind::Dimension => {
                let mode = self.buttons.iter().enumerate().filter(|(_, b)| **b).map(|(i, _)| i + 1).max();
                let Some(mode) = mode else { return };
                let pressed = self.buttons.iter().filter(|b| **b).count() as f32;
                let (rate, depth) = match mode {
                    1 => (0.25, 0.6),
                    2 => (0.25, 1.0),
                    3 => (0.25, 1.4),
                    _ => (0.5, 1.8),
                };
                self.lfo.freq = rate;
                let depth = depth * (1.0 + 0.2 * (pressed - 1.0)) * ms;
                for i in 0..l.len() {
                    self.lfo.advance(sr);
                    let m = self.lfo.value(Wave::Triangle, 0.0);
                    let x = (l[i] + r[i]) * 0.5;
                    let a = self.voices[0][0].run(x, 7.0 * ms + depth * m);
                    let b = self.voices[1][0].run(x, 7.0 * ms - depth * m);
                    // Cross-mixed in opposite polarity like the original.
                    l[i] = l[i] + 0.5 * a - 0.3 * b;
                    r[i] = r[i] + 0.5 * b - 0.3 * a;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Flanger (TA_FLANGER)

pub struct Flanger {
    sr: f32,
    rate: SyncParam,
    range: f32,
    color: f32,
    intensity: f32,
    voices: [Voice; 2],
    fb: [f32; 2],
    lfo: Lfo,
}

impl Flanger {
    pub fn new() -> Self {
        Self {
            sr: 48000.0,
            rate: SyncParam::rate(0.5, RATE_NOTES_LONG),
            range: 50.0,
            color: 50.0,
            intensity: 50.0,
            voices: [Voice::new(), Voice::new()],
            fb: [0.0; 2],
            lfo: Lfo::default(),
        }
    }
}

impl Effect for Flanger {
    fn prepare(&mut self, sr: f32) {
        self.sr = sr;
    }

    fn set(&mut self, name: &str, v: f32) {
        match name {
            "rate" => self.rate.free = v,
            "range" => self.range = v,
            "color" => self.color = v,
            "intensity" => self.intensity = v,
            _ => {}
        }
    }

    fn set_sync(&mut self, name: &str, unsynced: f32, synced: f32, sync: bool) {
        if name == "rate" {
            self.rate.set(unsynced, synced, sync);
        } else {
            self.set(name, unsynced);
        }
    }

    fn process(&mut self, ctx: &Ctx, l: &mut [f32], r: &mut [f32]) {
        let ms = self.sr / 1000.0;
        self.lfo.freq = self.rate.value(ctx.bpm);
        let max_d = lerp(1.0, 8.0, self.range / 100.0) * ms;
        let fb = self.color / 100.0 * 0.85;
        let mix = self.intensity / 100.0;
        for i in 0..l.len() {
            self.lfo.advance(self.sr);
            for ch in 0..2 {
                let m = 0.5 + 0.5 * self.lfo.value(Wave::Triangle, ch as f32 * 0.25);
                let x = if ch == 0 { l[i] } else { r[i] };
                let d = 0.2 * ms + max_d * m;
                let y = self.voices[ch].run(x + self.fb[ch] * fb, d);
                self.fb[ch] = y;
                let o = x + y * mix;
                if ch == 0 {
                    l[i] = o * 0.7;
                } else {
                    r[i] = o * 0.7;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Phasers (TA_PHASE90, TA_TTSMALLSTONE)

pub struct Phaser {
    sr: f32,
    stages: usize,
    speed: SyncParam,
    resonance: f32,
    level: f32,
    color: bool,
    ap: [[(f32, f32); 6]; 2],
    fb: [f32; 2],
    lfo: Lfo,
}

impl Phaser {
    pub fn new(small_stone: bool) -> Self {
        Self {
            sr: 48000.0,
            stages: if small_stone { 6 } else { 4 },
            speed: SyncParam::rate(1.0, RATE_NOTES_LONG),
            resonance: if small_stone { 0.0 } else { 50.0 },
            level: if small_stone { 80.0 } else { 60.0 },
            color: false,
            ap: [[(0.0, 0.0); 6]; 2],
            fb: [0.0; 2],
            lfo: Lfo::default(),
        }
    }
}

impl Effect for Phaser {
    fn prepare(&mut self, sr: f32) {
        self.sr = sr;
    }

    fn set(&mut self, name: &str, v: f32) {
        match name {
            "speed" => self.speed.free = v,
            "resonance" => self.resonance = v,
            "level" => self.level = v,
            "color" => {
                self.color = on(v);
                self.resonance = if self.color { 60.0 } else { 0.0 };
            }
            _ => {}
        }
    }

    fn set_sync(&mut self, name: &str, unsynced: f32, synced: f32, sync: bool) {
        if name == "speed" {
            self.speed.set(unsynced, synced, sync);
        } else {
            self.set(name, unsynced);
        }
    }

    fn process(&mut self, ctx: &Ctx, l: &mut [f32], r: &mut [f32]) {
        self.lfo.freq = self.speed.value(ctx.bpm);
        let fb = self.resonance / 100.0 * 0.7;
        let mix = self.level / 100.0;
        for i in 0..l.len() {
            self.lfo.advance(self.sr);
            for ch in 0..2 {
                let m = 0.5 + 0.5 * self.lfo.value(Wave::Sine, ch as f32 * 0.1);
                let f = 200.0 * (8.0f32).powf(m);
                let t = (std::f32::consts::PI * f / self.sr).tan();
                let a = (t - 1.0) / (t + 1.0);
                let x = if ch == 0 { l[i] } else { r[i] };
                let mut y = x + self.fb[ch] * fb;
                for s in &mut self.ap[ch][..self.stages] {
                    let out = a * y + s.0;
                    s.0 = y - a * out;
                    y = out;
                }
                self.fb[ch] = y;
                let o = (x + y * mix) / (1.0 + mix * 0.5);
                if ch == 0 {
                    l[i] = o;
                } else {
                    r[i] = o;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tremolo / auto-pan (TA_TREMOLO)

pub struct Tremolo {
    sr: f32,
    opto: bool,
    rate: SyncParam,
    intensity: f32,
    panning: bool,
    lfo: Lfo,
    smooth: f32,
}

impl Tremolo {
    pub fn new() -> Self {
        Self { sr: 48000.0, opto: true, rate: SyncParam::rate(4.0, RATE_NOTES), intensity: 8.0, panning: false, lfo: Lfo::default(), smooth: 1.0 }
    }
}

impl Effect for Tremolo {
    fn prepare(&mut self, sr: f32) {
        self.sr = sr;
    }

    fn set(&mut self, name: &str, v: f32) {
        match name {
            "mode" => self.opto = idx(v) == 0,
            "rate" => self.rate.free = v,
            "intensity" => self.intensity = v,
            "panning" => self.panning = on(v),
            _ => {}
        }
    }

    fn set_sync(&mut self, name: &str, unsynced: f32, synced: f32, sync: bool) {
        if name == "rate" {
            self.rate.set(unsynced, synced, sync);
        } else {
            self.set(name, unsynced);
        }
    }

    fn process(&mut self, ctx: &Ctx, l: &mut [f32], r: &mut [f32]) {
        self.lfo.freq = self.rate.value(ctx.bpm);
        let depth = knob10(self.intensity);
        let c = time_coef(if self.opto { 0.008 } else { 0.0005 }, self.sr);
        for i in 0..l.len() {
            self.lfo.advance(self.sr);
            let raw = self.lfo.value(if self.opto { Wave::Sine } else { Wave::Triangle }, 0.0);
            // Opto cells respond asymmetrically: quick on, slow off.
            let target = if self.opto { raw.signum() * raw.abs().powf(0.6) } else { raw };
            self.smooth = target + (self.smooth - target) * c;
            let m = self.smooth;
            if self.panning {
                let p = 0.5 + 0.5 * m * depth;
                l[i] *= (1.0 - p).sqrt() * std::f32::consts::SQRT_2;
                r[i] *= p.sqrt() * std::f32::consts::SQRT_2;
            } else {
                let g = 1.0 - depth * 0.5 * (1.0 - m);
                l[i] *= g;
                r[i] *= g;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Vibrato (TA_VB2), Hammond scanner vibrato (TA_HAMMONDVIBRATO / 2)

pub struct Vibrato {
    sr: f32,
    hammond: bool,
    speed: SyncParam,
    depth: f32,
    level: f32,
    mode: usize,
    depth_cents: [f32; 3],
    vdry: f32,
    vwet: f32,
    cdry: f32,
    cwet: f32,
    voices: [Voice; 2],
    lfo: Lfo,
}

impl Vibrato {
    pub fn new(hammond: bool) -> Self {
        Self {
            sr: 48000.0,
            hammond,
            speed: SyncParam::rate(if hammond { 6.9 } else { 5.0 }, RATE_NOTES),
            depth: 50.0,
            level: 80.0,
            mode: 1,
            depth_cents: [5.8, 8.1, 13.0],
            vdry: 70.0,
            vwet: 30.0,
            cdry: 84.0,
            cwet: 30.0,
            voices: [Voice::new(), Voice::new()],
            lfo: Lfo::default(),
        }
    }
}

impl Effect for Vibrato {
    fn prepare(&mut self, sr: f32) {
        self.sr = sr;
    }

    fn set(&mut self, name: &str, v: f32) {
        match name {
            "speed" => self.speed.free = v,
            "depth" => self.depth = v,
            "level" => self.level = v,
            "mode" => self.mode = idx(v),
            "depth1" => self.depth_cents[0] = v,
            "depth2" => self.depth_cents[1] = v,
            "depth3" => self.depth_cents[2] = v,
            "vdry" => self.vdry = v,
            "vwet" => self.vwet = v,
            "cdry" => self.cdry = v,
            "cwet" => self.cwet = v,
            _ => {}
        }
    }

    fn set_sync(&mut self, name: &str, unsynced: f32, synced: f32, sync: bool) {
        if name == "speed" {
            self.speed.set(unsynced, synced, sync);
        } else {
            self.set(name, unsynced);
        }
    }

    fn process(&mut self, ctx: &Ctx, l: &mut [f32], r: &mut [f32]) {
        let rate = self.speed.value(ctx.bpm);
        self.lfo.freq = rate;
        let (depth_s, dry, wet) = if self.hammond {
            if self.mode == 0 {
                return;
            }
            // V-n: pure vibrato; C-n: mixed with dry (chorus).
            let n = (self.mode - 1) / 2;
            let chorus = (self.mode - 1) % 2 == 1;
            let cents = self.depth_cents[n.min(2)];
            // Peak delay swing giving the requested pitch deviation.
            let swing = (2f32.powf(cents / 1200.0) - 1.0) / (TAU * rate.max(0.1));
            if chorus {
                (swing, self.cdry / 100.0, self.cwet / 100.0)
            } else {
                (swing, 1.0 - self.vwet / 100.0 * 0.0 - 1.0, 1.0)
            }
        } else {
            let swing = 0.003 * self.depth / 100.0;
            let m = self.level / 100.0;
            (swing, 1.0 - m, m)
        };
        let depth = depth_s * self.sr;
        let base = depth + 2.0;
        for i in 0..l.len() {
            self.lfo.advance(self.sr);
            let m = self.lfo.value(if self.hammond { Wave::Triangle } else { Wave::Sine }, 0.0);
            for ch in 0..2 {
                let x = if ch == 0 { l[i] } else { r[i] };
                let y = self.voices[ch].run(x, base + depth * m);
                let o = x * dry.max(0.0) + y * wet;
                if ch == 0 {
                    l[i] = o;
                } else {
                    r[i] = o;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Auto-wah (TA_AUTOWAH)

pub struct AutoWah {
    sr: f32,
    filter: f32,
    speed: SyncParam,
    depth: f32,
    sensitivity: f32,
    release: f32,
    mode: usize,
    negative: bool,
    level: f32,
    env: Envelope,
    lfo: Lfo,
    svf: [Svf; 2],
}

impl AutoWah {
    pub fn new() -> Self {
        Self {
            sr: 48000.0,
            filter: 0.0,
            speed: SyncParam::rate(1.0, RATE_NOTES_LONG),
            depth: 0.0,
            sensitivity: 0.0,
            release: 1.0,
            mode: 1,
            negative: false,
            level: 0.0,
            env: Envelope::default(),
            lfo: Lfo::default(),
            svf: Default::default(),
        }
    }
}

impl Effect for AutoWah {
    fn prepare(&mut self, sr: f32) {
        self.sr = sr;
        self.env.set(sr, 0.003, self.release);
    }

    fn set(&mut self, name: &str, v: f32) {
        match name {
            "filter" => self.filter = v,
            "speed" => self.speed.free = v,
            "depth" => self.depth = v,
            "sensitivity" => self.sensitivity = v,
            "release" => {
                self.release = v;
                self.env.set(self.sr, 0.003, v);
            }
            "mode" => self.mode = idx(v),
            "polar" => self.negative = idx(v) == 1,
            "level" => self.level = v,
            _ => {}
        }
    }

    fn set_sync(&mut self, name: &str, unsynced: f32, synced: f32, sync: bool) {
        if name == "speed" {
            self.speed.set(unsynced, synced, sync);
        } else {
            self.set(name, unsynced);
        }
    }

    fn process(&mut self, ctx: &Ctx, l: &mut [f32], r: &mut [f32]) {
        self.lfo.freq = self.speed.value(ctx.bpm);
        let base = 300.0 * 2f32.powf(self.filter / 100.0 * 3.0);
        let g = db_to_gain(self.level);
        let sign = if self.negative { -1.0 } else { 1.0 };
        for i in 0..l.len() {
            self.lfo.advance(self.sr);
            let e = self.env.process(l[i].abs().max(r[i].abs()));
            let m = (e * 8.0 * self.sensitivity / 100.0).min(1.0) + (0.5 + 0.5 * self.lfo.value(Wave::Sine, 0.0)) * self.depth / 100.0;
            let f = (base * 2f32.powf(sign * m * 3.0)).clamp(80.0, self.sr * 0.4);
            for ch in 0..2 {
                self.svf[ch].set(self.sr, f, 4.0);
                let x = if ch == 0 { l[i] } else { r[i] };
                let o = self.svf[ch].process(x);
                let y = match self.mode {
                    0 => o.low,
                    1 => o.band * 2.0,
                    _ => o.high,
                };
                if ch == 0 {
                    l[i] = y * g;
                } else {
                    r[i] = y * g;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Analog octave divider (TA_OCTAVER2)

pub struct Octaver {
    sr: f32,
    tone: f32,
    oct1: f32,
    oct2: f32,
    dry: f32,
    state: [(bool, f32, f32, bool); 2],
    env: [Envelope; 2],
    lp: [Biquad; 2],
    pre: [Biquad; 2],
}

impl Octaver {
    pub fn new() -> Self {
        Self {
            sr: 48000.0,
            tone: 0.0,
            oct1: 0.0,
            oct2: -36.0,
            dry: 0.0,
            state: [(false, 1.0, 1.0, false); 2],
            env: Default::default(),
            lp: Default::default(),
            pre: Default::default(),
        }
    }
}

impl Effect for Octaver {
    fn prepare(&mut self, sr: f32) {
        self.sr = sr;
        for e in &mut self.env {
            e.set(sr, 0.002, 0.05);
        }
        for p in &mut self.pre {
            p.set(lowpass(sr, 400.0));
        }
        self.set("tone", self.tone);
    }

    fn set(&mut self, name: &str, v: f32) {
        match name {
            "tone" => {
                self.tone = v;
                for b in &mut self.lp {
                    b.set(lowpass(self.sr, lerp(250.0, 2500.0, v / 100.0)));
                }
            }
            "oct1" => self.oct1 = v,
            "oct2" => self.oct2 = v,
            "dry" => self.dry = v,
            _ => {}
        }
    }

    fn process(&mut self, _: &Ctx, l: &mut [f32], r: &mut [f32]) {
        let (g1, g2, gd) = (db_to_gain(self.oct1), db_to_gain(self.oct2), db_to_gain(self.dry));
        for (ch, s) in [l, r].into_iter().enumerate() {
            for x in s.iter_mut() {
                let p = self.pre[ch].process(*x);
                let e = self.env[ch].process(p.abs());
                let st = &mut self.state[ch];
                // Flip-flop dividers clocked on positive zero crossings.
                let pos = p > 0.0;
                if pos && !st.0 {
                    st.1 = -st.1;
                    if st.1 > 0.0 {
                        st.2 = -st.2;
                    }
                }
                st.0 = pos;
                let sub = self.lp[ch].process((st.1 * g1 + st.2 * g2) * e * 2.0);
                *x = *x * gd + sub;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Polyphonic octave generator (TA_POG2)

pub struct Pog {
    sr: f32,
    dry: f32,
    levels: [f32; 4],
    attack: f32,
    cutoff: f32,
    q: f32,
    detune: f32,
    shifters: [[PitchShifter; 4]; 2],
    svf: [Svf; 2],
    env: [Envelope; 2],
    swell: [f32; 2],
}

impl Pog {
    pub fn new() -> Self {
        Self {
            sr: 48000.0,
            dry: 10.0,
            levels: [0.0; 4],
            attack: 0.0,
            cutoff: 8000.0,
            q: 1.0,
            detune: 0.0,
            shifters: Default::default(),
            svf: Default::default(),
            env: Default::default(),
            swell: [0.0; 2],
        }
    }
}

impl Effect for Pog {
    fn prepare(&mut self, sr: f32) {
        self.sr = sr;
        for s in self.shifters.iter_mut().flatten() {
            s.prepare(sr);
        }
        for e in &mut self.env {
            e.set(sr, 0.001, 0.1);
        }
    }

    fn set(&mut self, name: &str, v: f32) {
        match name {
            "dry" => self.dry = v,
            "m2octave" => self.levels[0] = v,
            "m1octave" => self.levels[1] = v,
            "p1octave" => self.levels[2] = v,
            "p2octave" => self.levels[3] = v,
            "attack" => self.attack = v,
            "cutoff" => self.cutoff = v,
            "q" => self.q = v,
            "detune" => self.detune = v,
            _ => {}
        }
    }

    fn process(&mut self, _: &Ctx, l: &mut [f32], r: &mut [f32]) {
        let det = 2f32.powf(self.detune / 10.0 * 0.15 / 12.0);
        let ratios = [0.25, 0.5 * det, 2.0 / det, 4.0];
        let lv: [f32; 4] = std::array::from_fn(|i| knob10(self.levels[i]));
        let dry = knob10(self.dry);
        let swell_c = time_coef(knob10(self.attack) * 0.5 + 1e-4, self.sr);
        for ch in 0..2 {
            self.svf[ch].set(self.sr, self.cutoff.clamp(100.0, self.sr * 0.45), self.q.max(0.5));
        }
        for (ch, s) in [l, r].into_iter().enumerate() {
            for x in s.iter_mut() {
                let mut wet = 0.0;
                for k in 0..4 {
                    if lv[k] > 0.0 {
                        wet += self.shifters[ch][k].process(*x, ratios[k]) * lv[k];
                    } else {
                        self.shifters[ch][k].process(*x, ratios[k]);
                    }
                }
                let e = self.env[ch].process(x.abs());
                let target = if e > 0.01 { 1.0 } else { 0.0 };
                self.swell[ch] = target + (self.swell[ch] - target) * swell_c;
                let wet = self.svf[ch].process(wet * self.swell[ch]).low;
                *x = *x * dry + wet;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Stereo enhancer (TA_TTSPACER)

pub struct StereoEnhancer {
    sr: f32,
    width: f32,
    gain: f32,
    asymmetry: f32,
    rotation: f32,
    shuffle: f32,
    shuffle_f: Biquad,
}

impl StereoEnhancer {
    pub fn new() -> Self {
        Self { sr: 48000.0, width: 0.0, gain: 0.0, asymmetry: 0.0, rotation: 0.0, shuffle: 0.0, shuffle_f: Biquad::identity() }
    }
}

impl Effect for StereoEnhancer {
    fn prepare(&mut self, sr: f32) {
        self.sr = sr;
        self.shuffle_f.set(lowpass(sr, 700.0));
    }

    fn set(&mut self, name: &str, v: f32) {
        match name {
            "width" => self.width = v,
            "gain" => self.gain = v,
            "asymmetry" => self.asymmetry = v,
            "rotation" => self.rotation = v,
            "shuffle" => self.shuffle = v,
            _ => {}
        }
    }

    fn process(&mut self, _: &Ctx, l: &mut [f32], r: &mut [f32]) {
        // width: -100 (mono) .. 0 (unchanged) .. +100 (2x side)
        let side = (1.0 + self.width / 100.0).max(0.0);
        let g = db_to_gain(self.gain);
        let asym = self.asymmetry / 100.0;
        let rot = self.rotation / 100.0 * std::f32::consts::FRAC_PI_4;
        let (sn, cs) = rot.sin_cos();
        let shuffle = self.shuffle / 100.0;
        for (a, b) in l.iter_mut().zip(r.iter_mut()) {
            let (m, s) = to_ms(*a, *b);
            let low_s = self.shuffle_f.process(s);
            let s = s * side + low_s * shuffle * 1.5;
            let m = m + asym * s;
            let (x, y) = from_ms(m, s);
            let (x, y) = (x * cs - y * sn, x * sn + y * cs);
            *a = x * g;
            *b = y * g;
        }
    }
}
