//! Compressors, limiters, gates and other level processors.

use crate::dsp::*;
use crate::effect::{for_each, idx, on, Ctx, Effect};

/// Gain computer with log-domain attack/release smoothing.
#[derive(Clone, Copy, Debug)]
pub struct CompCore {
    pub threshold: f32,
    pub ratio: f32,
    pub knee: f32,
    att: f32,
    rel: f32,
    /// Current gain reduction in dB (<= 0).
    pub gr: f32,
}

impl Default for CompCore {
    fn default() -> Self {
        Self { threshold: 0.0, ratio: 1.0, knee: 6.0, att: 0.9, rel: 0.999, gr: 0.0 }
    }
}

impl CompCore {
    pub fn times(&mut self, sr: f32, attack: f32, release: f32) {
        self.att = time_coef(attack.max(2e-5), sr);
        self.rel = time_coef(release.max(5e-3), sr);
    }

    /// Returns the gain (linear) for the detector level (linear peak).
    #[inline]
    pub fn gain(&mut self, level: f32) -> f32 {
        db_to_gain(self.gain_db(level))
    }

    #[inline]
    pub fn gain_db(&mut self, level: f32) -> f32 {
        let target = compress_db(gain_to_db(level), self.threshold, self.ratio, self.knee);
        let c = if target < self.gr { self.att } else { self.rel };
        self.gr = target + (self.gr - target) * c;
        self.gr
    }

    /// Same with a per-sample release coefficient (program dependent).
    #[inline]
    pub fn gain_db_rel(&mut self, level: f32, rel: f32) -> f32 {
        let target = compress_db(gain_to_db(level), self.threshold, self.ratio, self.knee);
        let c = if target < self.gr { self.att } else { rel };
        self.gr = target + (self.gr - target) * c;
        self.gr
    }
}

// ---------------------------------------------------------------------------
// Generic compressor (TA_TOONTRACK_COMPR / _COMPRMS / TA_SLKSCOMPRESSOR)

#[derive(Clone, Copy)]
struct CompSettings {
    threshold: f32,
    attack: f32,
    release: f32,
    ratio: f32,
    makeup: f32,
    sidechain: f32,
}

impl Default for CompSettings {
    fn default() -> Self {
        Self { threshold: -30.0, attack: 0.01, release: 1.0, ratio: 1.0, makeup: 0.0, sidechain: 110.0 }
    }
}

impl CompSettings {
    fn set(&mut self, field: &str, v: f32) -> bool {
        match field {
            "threshold" => self.threshold = v,
            "attack" => self.attack = v,
            "release" => self.release = v,
            "ratio" => self.ratio = v,
            "makeup" => self.makeup = v,
            "sidechain" => self.sidechain = v,
            _ => return false,
        }
        true
    }
}

pub struct Compressor {
    sr: f32,
    mid_side: bool,
    auto_makeup: bool,
    a: CompSettings,
    b: CompSettings,
    parallel: f32,
    level: f32,
    core: [CompCore; 2],
    hp: [Biquad; 2],
    dirty: bool,
}

impl Compressor {
    pub fn new(mid_side: bool) -> Self {
        Self {
            sr: 48000.0,
            mid_side,
            auto_makeup: false,
            a: CompSettings::default(),
            b: CompSettings::default(),
            parallel: 100.0,
            level: 0.0,
            core: Default::default(),
            hp: Default::default(),
            dirty: true,
        }
    }

    fn update(&mut self) {
        for (i, s) in [self.a, if self.mid_side { self.b } else { self.a }].iter().enumerate() {
            let c = &mut self.core[i];
            c.threshold = s.threshold;
            // A ratio below 1 in presets means "off".
            c.ratio = s.ratio.max(1.0);
            c.knee = 6.0;
            c.times(self.sr, s.attack, s.release);
            self.hp[i].set(if s.sidechain > 20.0 { highpass(self.sr, s.sidechain) } else { Coefs::IDENTITY });
        }
        self.dirty = false;
    }

    fn makeup(&self, s: &CompSettings) -> f32 {
        let auto = if self.auto_makeup { -compress_db(0.0, s.threshold, s.ratio.max(1.0), 0.0) * 0.5 } else { 0.0 };
        db_to_gain(s.makeup + auto)
    }
}

impl Effect for Compressor {
    fn prepare(&mut self, sr: f32) {
        self.sr = sr;
        self.dirty = true;
    }

    fn set(&mut self, name: &str, v: f32) {
        self.dirty = true;
        match name {
            "parallel" => self.parallel = v,
            "level" => self.level = v,
            "automku" => self.auto_makeup = on(v),
            _ => {
                if let Some(f) = name.strip_prefix("mid_") {
                    self.a.set(f, v);
                } else if let Some(f) = name.strip_prefix("side_") {
                    self.b.set(f, v);
                } else {
                    self.a.set(name, v);
                    self.b.set(name, v);
                }
            }
        }
    }

    fn reset(&mut self) {
        self.core.iter_mut().for_each(|c| c.gr = 0.0);
    }

    fn process(&mut self, _: &Ctx, l: &mut [f32], r: &mut [f32]) {
        if self.dirty {
            self.update();
        }
        let mk = [self.makeup(&self.a), self.makeup(&self.b)];
        let wet = self.parallel / 100.0;
        let out = db_to_gain(self.level);
        for (a, b) in l.iter_mut().zip(r.iter_mut()) {
            let (x, y) = if self.mid_side { to_ms(*a, *b) } else { (*a, *b) };
            let (gx, gy) = if self.mid_side {
                let dx = self.hp[0].process(x).abs();
                let dy = self.hp[1].process(y).abs();
                (self.core[0].gain(dx) * mk[0], self.core[1].gain(dy) * mk[1])
            } else {
                let d = self.hp[0].process(x).abs().max(self.hp[1].process(y).abs());
                let g = self.core[0].gain(d) * mk[0];
                (g, g)
            };
            let (cx, cy) = (x * gx, y * gy);
            let (mut ox, mut oy) = (lerp(x, cx, wet), lerp(y, cy, wet));
            if self.mid_side {
                (ox, oy) = from_ms(ox, oy);
            }
            *a = ox * out;
            *b = oy * out;
        }
    }
}

// ---------------------------------------------------------------------------
// Peak limiter (TA_TOONTRACK_LIMITER / _LIMITERMS)

pub struct Limiter {
    sr: f32,
    mid_side: bool,
    ceiling: f32,
    drive: [f32; 2],
    release: [f32; 2],
    level: f32,
    env: [f32; 2],
    rel_c: [f32; 2],
}

impl Limiter {
    pub fn new(mid_side: bool) -> Self {
        Self {
            sr: 48000.0,
            mid_side,
            ceiling: 0.0,
            drive: [0.0; 2],
            release: [1.0; 2],
            level: 0.0,
            env: [1.0; 2],
            rel_c: [0.9999; 2],
        }
    }
}

impl Effect for Limiter {
    fn prepare(&mut self, sr: f32) {
        self.sr = sr;
        for i in 0..2 {
            self.rel_c[i] = time_coef(self.release[i] * 0.25, sr);
        }
    }

    fn set(&mut self, name: &str, v: f32) {
        match name {
            "ceiling" => self.ceiling = v,
            "threshold" => self.drive = [v.abs(); 2],
            "mid_threshold" => self.drive[0] = v.abs(),
            "side_threshold" => self.drive[1] = v.abs(),
            "release" => self.release = [v; 2],
            "mid_release" => self.release[0] = v,
            "side_release" => self.release[1] = v,
            "level" => self.level = v,
            _ => return,
        }
        self.prepare(self.sr);
    }

    fn process(&mut self, _: &Ctx, l: &mut [f32], r: &mut [f32]) {
        let ceil = db_to_gain(self.ceiling);
        let out = db_to_gain(self.level);
        let d = [db_to_gain(self.drive[0]), db_to_gain(self.drive[1])];
        for (a, b) in l.iter_mut().zip(r.iter_mut()) {
            let (x, y) = if self.mid_side { to_ms(*a, *b) } else { (*a, *b) };
            let (x, y) = (x * d[0], y * d[if self.mid_side { 1 } else { 0 }]);
            let (mut gx, mut gy);
            if self.mid_side {
                gx = self.track(0, x.abs(), ceil);
                gy = self.track(1, y.abs(), ceil);
            } else {
                gx = self.track(0, x.abs().max(y.abs()), ceil);
                gy = gx;
            }
            // Catch whatever the envelope missed.
            let soft = |v: f32| ceil * shape::tanh(v / ceil);
            let (mut ox, mut oy) = (soft(x * gx), soft(y * gy));
            if self.mid_side {
                (ox, oy) = from_ms(ox, oy);
            }
            gx = out;
            gy = out;
            *a = ox * gx;
            *b = oy * gy;
        }
    }
}

impl Limiter {
    #[inline]
    fn track(&mut self, i: usize, peak: f32, ceil: f32) -> f32 {
        let want = if peak > ceil { ceil / peak } else { 1.0 };
        if want < self.env[i] {
            self.env[i] = want;
        } else {
            self.env[i] = want + (self.env[i] - want) * self.rel_c[i];
        }
        self.env[i]
    }
}

// ---------------------------------------------------------------------------
// Mastering limiter (TA_MASTERLIMITER / TA_MASTERLIMITERTHRESHOLD)

pub struct MasterLimiter {
    sr: f32,
    drive: f32,
    attack_ms: f32,
    release: f32,
    knee: f32,
    enhance: f32,
    level: f32,
    env: f32,
    att_c: f32,
    rel_c: f32,
    low: [Biquad; 2],
}

impl MasterLimiter {
    pub fn new() -> Self {
        Self {
            sr: 48000.0,
            drive: 0.0,
            attack_ms: 0.1,
            release: 0.1,
            knee: 0.0,
            enhance: 0.0,
            level: 0.0,
            env: 1.0,
            att_c: 0.0,
            rel_c: 0.999,
            low: Default::default(),
        }
    }

    fn update(&mut self) {
        // Attack is realised as a smoothing of the gain; the limiter itself is
        // zero-latency so the plugin never reports latency changes.
        self.att_c = time_coef(self.attack_ms / 1000.0, self.sr);
        self.rel_c = time_coef(self.release.max(0.01), self.sr);
        for b in &mut self.low {
            b.set(low_shelf(self.sr, 120.0, self.enhance * 0.35));
        }
    }
}

impl Effect for MasterLimiter {
    fn prepare(&mut self, sr: f32) {
        self.sr = sr;
        self.update();
    }

    fn set(&mut self, name: &str, v: f32) {
        match name {
            "gain" => self.drive = v,
            "threshold" => self.drive = -v,
            "attack" => self.attack_ms = v,
            "release" => self.release = v,
            "knee" => self.knee = v,
            "enhance" => self.enhance = v,
            "Level" => self.level = v,
            _ => return,
        }
        self.update();
    }

    fn process(&mut self, _: &Ctx, l: &mut [f32], r: &mut [f32]) {
        let drive = db_to_gain(self.drive);
        let out = db_to_gain(self.level);
        // knee is a negative dB value where soft limiting starts below 0 dBFS.
        let knee = db_to_gain(self.knee.min(0.0));
        let ceil = 0.98;
        for (a, b) in l.iter_mut().zip(r.iter_mut()) {
            let x = self.low[0].process(*a * drive);
            let y = self.low[1].process(*b * drive);
            let peak = x.abs().max(y.abs());
            let thr = ceil * knee.max(0.05);
            let want = if peak > thr { thr / peak } else { 1.0 };
            let c = if want < self.env { self.att_c } else { self.rel_c };
            self.env = want + (self.env - want) * c;
            let g = self.env;
            let clip = |v: f32| {
                // Soft region between the knee and the ceiling.
                let s = v * g;
                let m = s.abs();
                if m <= thr {
                    s
                } else {
                    s.signum() * (thr + (ceil - thr) * shape::tanh((m - thr) / (ceil - thr + 1e-6)))
                }
            };
            *a = clip(x) * out;
            *b = clip(y) * out;
        }
    }
}

// ---------------------------------------------------------------------------
// Multiband compressor (TA_MULTIBANDCOMPR / ...THRESHOLD)

pub struct MultibandComp {
    sr: f32,
    threshold_mode: bool,
    freqs: [f32; 3],
    drive: [f32; 4],
    attack: [f32; 4],
    release: [f32; 4],
    ratio: [f32; 4],
    band_level: [f32; 4],
    enable: [bool; 4],
    master_gain: f32,
    master_level: f32,
    parallel: f32,
    split: [FourBandSplit; 2],
    core: [CompCore; 4],
    dirty: bool,
}

impl MultibandComp {
    pub fn new(threshold_mode: bool) -> Self {
        Self {
            sr: 48000.0,
            threshold_mode,
            freqs: [100.0, 800.0, 5000.0],
            drive: [0.0; 4],
            attack: [0.04; 4],
            release: [2.0; 4],
            ratio: [2.0; 4],
            band_level: [0.0; 4],
            enable: [true; 4],
            master_gain: 0.0,
            master_level: 0.0,
            parallel: 100.0,
            split: Default::default(),
            core: Default::default(),
            dirty: true,
        }
    }
}

impl Effect for MultibandComp {
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
                if let Some(rest) = name.strip_prefix("enable") {
                    if let Ok(i) = rest.parse::<usize>() {
                        if (1..=4).contains(&i) {
                            self.enable[i - 1] = on(v);
                        }
                    }
                    return;
                }
                let Some(c) = name.chars().next().and_then(|c| c.to_digit(10)) else { return };
                let i = (c as usize).min(3);
                match &name[1..] {
                    "drive" => self.drive[i] = v,
                    "thresh" => self.drive[i] = -v,
                    "attack" => self.attack[i] = v,
                    "release" => self.release[i] = v,
                    "ratio" => self.ratio[i] = v,
                    "Level" => self.band_level[i] = v,
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
            for i in 0..4 {
                let c = &mut self.core[i];
                // "Drive" pushes the band into a fixed threshold; the threshold
                // variant exposes the threshold directly.
                c.threshold = if self.threshold_mode { -self.drive[i] } else { -12.0 - self.drive[i] };
                c.ratio = self.ratio[i];
                c.knee = 6.0;
                c.times(self.sr, self.attack[i], self.release[i]);
            }
            self.dirty = false;
        }
        let gin = db_to_gain(self.master_gain);
        let gout = db_to_gain(self.master_level);
        let lv: [f32; 4] = std::array::from_fn(|i| db_to_gain(self.band_level[i]));
        let wet = self.parallel / 100.0;
        for (a, b) in l.iter_mut().zip(r.iter_mut()) {
            let (x, y) = (*a * gin, *b * gin);
            let bl = self.split[0].split(x);
            let br = self.split[1].split(y);
            let (mut ox, mut oy) = (0.0, 0.0);
            for i in 0..4 {
                let g = if self.enable[i] { self.core[i].gain(bl[i].abs().max(br[i].abs())) } else { 1.0 };
                ox += bl[i] * g * lv[i];
                oy += br[i] * g * lv[i];
            }
            *a = lerp(x, ox, wet) * gout;
            *b = lerp(y, oy, wet) * gout;
        }
    }
}

// ---------------------------------------------------------------------------
// FET compressor (TA_1176)

pub struct Fet {
    sr: f32,
    p: [[f32; 5]; 2], // input, output, attack, release, ratio
    midside: bool,
    mix: f32,
    core: [CompCore; 2],
}

impl Fet {
    pub fn new() -> Self {
        Self { sr: 48000.0, p: [[-18.0, 0.0, 4.0, 4.0, 0.0]; 2], midside: false, mix: 100.0, core: Default::default() }
    }

    fn update(&mut self) {
        for i in 0..2 {
            let [_, _, att, rel, ratio] = self.p[i];
            let c = &mut self.core[i];
            // Knob 1 is slowest, 7 fastest (inverted like the hardware).
            let a = 0.0008 * (0.02f32 / 0.8).powf(((att.clamp(1.0, 7.0)) - 1.0) / 6.0);
            let r = 1.1 * (0.05f32 / 1.1).powf(((rel.clamp(1.0, 7.0)) - 1.0) / 6.0);
            c.times(self.sr, a, r);
            let (ratio, knee) = match idx(ratio) {
                0 => (4.0, 6.0),
                1 => (8.0, 4.0),
                2 => (12.0, 3.0),
                3 => (20.0, 2.0),
                _ => (30.0, 10.0),
            };
            c.ratio = ratio;
            c.knee = knee;
            c.threshold = -20.0;
        }
    }
}

impl Effect for Fet {
    fn prepare(&mut self, sr: f32) {
        self.sr = sr;
        self.update();
    }

    fn set(&mut self, name: &str, v: f32) {
        let (i, field) = match name.strip_prefix("side-") {
            Some(f) => (1, f),
            None => (0, name),
        };
        let k = match field {
            "input" => 0,
            "output" => 1,
            "attack" => 2,
            "release" => 3,
            "ratio" => 4,
            "midside" => {
                self.midside = on(v);
                return;
            }
            "mix" => {
                self.mix = v;
                return;
            }
            _ => return,
        };
        self.p[i][k] = v;
        if i == 0 && !self.midside {
            // Keep the side settings in step until mid/side is engaged.
        }
        self.update();
    }

    fn process(&mut self, _: &Ctx, l: &mut [f32], r: &mut [f32]) {
        let ms = self.midside;
        let sets = if ms { [0, 1] } else { [0, 0] };
        let gin = [db_to_gain(self.p[sets[0]][0] + 18.0), db_to_gain(self.p[sets[1]][0] + 18.0)];
        let gout = [db_to_gain(self.p[sets[0]][1]), db_to_gain(self.p[sets[1]][1])];
        let all = [idx(self.p[0][4]) >= 4, idx(self.p[1][4]) >= 4];
        let mix = self.mix / 100.0;
        for (a, b) in l.iter_mut().zip(r.iter_mut()) {
            let (x0, y0) = if ms { to_ms(*a, *b) } else { (*a, *b) };
            let (x, y) = (x0 * gin[0], y0 * gin[1]);
            let (gx, gy) = if ms {
                (self.core[0].gain(x.abs()), self.core[1].gain(y.abs()))
            } else {
                let g = self.core[0].gain(x.abs().max(y.abs()));
                (g, g)
            };
            // FET stage colouration, more aggressive in "all buttons" mode.
            let drive = |v: f32, all: bool| {
                let k = if all { 1.6 } else { 0.8 };
                shape::tube(v * k, 0.05) / k
            };
            let (mut ox, mut oy) = (drive(x * gx, all[0]) * gout[0], drive(y * gy, all[1]) * gout[1]);
            ox = lerp(x0, ox, mix);
            oy = lerp(y0, oy, mix);
            if ms {
                (ox, oy) = from_ms(ox, oy);
            }
            *a = ox;
            *b = oy;
        }
    }
}

// ---------------------------------------------------------------------------
// Opto compressors (TA_LA2A, TA_LA3A)

#[derive(Clone, Copy)]
struct OptoSet {
    gain: f32,
    peak_red: f32,
    limit: bool,
}

pub struct Opto {
    sr: f32,
    la3a: bool,
    set: [OptoSet; 2],
    midside: bool,
    in_lev: f32,
    sidechain: f32,
    parallel: f32,
    out_level: f32,
    harmonics: f32,
    core: [CompCore; 2],
    slow: [f32; 2],
    sc: [Biquad; 2],
    fast_rel: f32,
    slow_rel: f32,
    slow_c: f32,
}

impl Opto {
    pub fn new(la3a: bool) -> Self {
        let s = OptoSet { gain: 20.0, peak_red: if la3a { 20.0 } else { 0.0 }, limit: false };
        Self {
            sr: 48000.0,
            la3a,
            set: [s; 2],
            midside: false,
            in_lev: 0.0,
            sidechain: if la3a { 120.0 } else { 100.0 },
            parallel: 100.0,
            out_level: 0.0,
            harmonics: 0.0,
            core: Default::default(),
            slow: [0.0; 2],
            sc: Default::default(),
            fast_rel: 0.999,
            slow_rel: 0.9999,
            slow_c: 0.9999,
        }
    }

    fn update(&mut self) {
        for i in 0..2 {
            let s = self.set[i];
            let c = &mut self.core[i];
            c.threshold = -s.peak_red * 0.4;
            c.ratio = if s.limit { 12.0 } else { 3.0 };
            c.knee = 10.0;
            c.times(self.sr, if self.la3a { 0.0015 } else { 0.01 }, 0.06);
            // LA-2A "R37" emphasis: the sidechain control tilts detection
            // towards high frequencies at low settings. LA-3A: plain HPF.
            let f = if self.la3a { self.sidechain.max(20.0) } else { lerp(2000.0, 30.0, self.sidechain / 100.0) };
            self.sc[i].set(if self.la3a { highpass(self.sr, f) } else { high_shelf(self.sr, f, 6.0) });
        }
        self.fast_rel = time_coef(0.06, self.sr);
        self.slow_rel = time_coef(1.5, self.sr);
        self.slow_c = time_coef(1.0, self.sr);
    }
}

impl Effect for Opto {
    fn prepare(&mut self, sr: f32) {
        self.sr = sr;
        self.update();
    }

    fn set(&mut self, name: &str, v: f32) {
        let (i, f) = match name.strip_prefix("side-") {
            Some(f) => (1, f),
            None => (0, name),
        };
        match f {
            "gain" => self.set[i].gain = v,
            "peak_red" => self.set[i].peak_red = v,
            "mode" => self.set[i].limit = on(v),
            "midside-mode" => self.midside = on(v),
            "in-lev" => self.in_lev = v,
            "sidechain" => self.sidechain = v,
            "parallel" => self.parallel = v,
            "out_level" => self.out_level = v,
            "harmonics" => self.harmonics = v,
            _ => return,
        }
        if i == 0 && !self.midside && f != "midside-mode" {
            self.set[1] = self.set[0];
        }
        self.update();
    }

    fn process(&mut self, _: &Ctx, l: &mut [f32], r: &mut [f32]) {
        let ms = self.midside;
        let gin = db_to_gain(self.in_lev);
        // Gain knob: ~0.4 dB per step around a reference of 20.
        let makeup = [db_to_gain((self.set[0].gain - 20.0) * 0.4), db_to_gain((self.set[1].gain - 20.0) * 0.4)];
        let out = db_to_gain(self.out_level);
        let wet = self.parallel / 100.0;
        let drive = db_to_gain(self.harmonics) * 0.5;
        for (a, b) in l.iter_mut().zip(r.iter_mut()) {
            let (x, y) = if ms { to_ms(*a * gin, *b * gin) } else { (*a * gin, *b * gin) };
            let mut g = [0.0f32; 2];
            let det = [self.sc[0].process(x).abs(), self.sc[1].process(y).abs()];
            let n = if ms { 2 } else { 1 };
            for i in 0..n {
                let d = if ms { det[i] } else { det[0].max(det[1]) };
                // Program-dependent release: long compression slows recovery.
                self.slow[i] = self.core[i].gr + (self.slow[i] - self.core[i].gr) * self.slow_c;
                let depth = (-self.slow[i] / 10.0).clamp(0.0, 1.0);
                let rel = lerp(self.fast_rel, self.slow_rel, depth);
                g[i] = db_to_gain(self.core[i].gain_db_rel(d, rel));
            }
            if !ms {
                g[1] = g[0];
            }
            let sat = |v: f32| if drive > 0.0 { shape::tube(v * drive, 0.15) / drive } else { v };
            let (mut ox, mut oy) = (sat(x * g[0] * makeup[0]), sat(y * g[1] * makeup[if ms { 1 } else { 0 }]));
            ox = lerp(x, ox, wet);
            oy = lerp(y, oy, wet);
            if ms {
                (ox, oy) = from_ms(ox, oy);
            }
            *a = ox * out;
            *b = oy * out;
        }
    }
}

// ---------------------------------------------------------------------------
// Variable-mu compressor (TA_FAIRCHILD670)

pub struct VariMu {
    sr: f32,
    model: usize,
    gain: [f32; 2],
    thresh: [f32; 2],
    dc: [f32; 2],
    mode: [usize; 2],
    parallel: f32,
    level: f32,
    midside: bool,
    sc_hp: [f32; 2],
    side_level: [f32; 2],
    harmonics: f32,
    core: [CompCore; 2],
    hp: [Biquad; 2],
}

impl VariMu {
    pub fn new() -> Self {
        Self {
            sr: 48000.0,
            model: 0,
            gain: [0.0; 2],
            thresh: [5.0; 2],
            dc: [5.0; 2],
            mode: [5; 2],
            parallel: 100.0,
            level: 0.0,
            midside: false,
            sc_hp: [30.0; 2],
            side_level: [0.0; 2],
            harmonics: -4.0,
            core: Default::default(),
            hp: Default::default(),
        }
    }

    fn update(&mut self) {
        // Time constants 1-6 of the original unit (attack, release).
        const TIMES: [(f32, f32); 6] = [(0.0002, 0.3), (0.0002, 0.8), (0.0004, 2.0), (0.0004, 5.0), (0.0002, 2.0), (0.0002, 0.3)];
        for i in 0..2 {
            let c = &mut self.core[i];
            let (a, r) = TIMES[self.mode[i].min(5)];
            c.times(self.sr, a, r);
            // Threshold knob 0..10 lowers the threshold; the DC bias sets the
            // knee / ratio (higher bias = softer, lower ratio).
            c.threshold = -2.0 - self.thresh[i] * 2.4;
            c.ratio = lerp(8.0, 2.0, (self.dc[i] / 10.0).clamp(0.0, 1.0));
            c.knee = 12.0;
            self.hp[i].set(highpass(self.sr, self.sc_hp[i].max(20.0)));
        }
    }
}

impl Effect for VariMu {
    fn prepare(&mut self, sr: f32) {
        self.sr = sr;
        self.update();
    }

    fn set(&mut self, name: &str, v: f32) {
        match name {
            "model" => self.model = idx(v),
            "lat_gain" => self.gain[0] = v,
            "vert_gain" => self.gain[1] = v,
            "lat_thresh" => self.thresh[0] = v,
            "vert_thresh" => self.thresh[1] = v,
            "lat_dc_thresh" => self.dc[0] = v,
            "vert_dc_thresh" => self.dc[1] = v,
            "lat_mode" => self.mode[0] = idx(v),
            "vert_mode" => self.mode[1] = idx(v),
            "parallel" => self.parallel = v,
            "level" => self.level = v,
            "stereo" => self.midside = on(v),
            "sidechain-mid" => self.sc_hp[0] = v,
            "sidechain-side" => self.sc_hp[1] = v,
            "mid-level" => self.side_level[0] = v,
            "side-level" => self.side_level[1] = v,
            "harmonics" => self.harmonics = v,
            _ => return,
        }
        self.update();
    }

    fn process(&mut self, _: &Ctx, l: &mut [f32], r: &mut [f32]) {
        let ms = self.midside;
        let gin = [db_to_gain(self.gain[0]), db_to_gain(self.gain[1])];
        let lv = [db_to_gain(self.side_level[0]), db_to_gain(self.side_level[1])];
        let out = db_to_gain(self.level);
        let wet = self.parallel / 100.0;
        let drive = db_to_gain(self.harmonics) * [0.8, 1.0, 1.3][self.model.min(2)];
        for (a, b) in l.iter_mut().zip(r.iter_mut()) {
            let (x0, y0) = if ms { to_ms(*a, *b) } else { (*a, *b) };
            let (x, y) = (x0 * gin[0], y0 * gin[1]);
            let gx = self.core[0].gain(self.hp[0].process(x).abs());
            let gy = self.core[1].gain(self.hp[1].process(y).abs());
            let sat = |v: f32| shape::tube(v * drive, 0.2) / drive;
            let (mut ox, mut oy) = (sat(x * gx) * lv[0], sat(y * gy) * lv[1]);
            ox = lerp(x0, ox, wet);
            oy = lerp(y0, oy, wet);
            if ms {
                (ox, oy) = from_ms(ox, oy);
            }
            *a = ox * out;
            *b = oy * out;
        }
    }
}

// ---------------------------------------------------------------------------
// VCA bus compressor (TA_SSLCOMPG) and listen-mic compressor (TA_SSLLMC)

pub struct VcaBus {
    sr: f32,
    gain: f32,
    midside: bool,
    thr: [f32; 2],
    att_ms: [f32; 2],
    ratio: [f32; 2],
    rel: [f32; 2],
    makeup: [f32; 2],
    sidechain: f32,
    parallel: f32,
    output: f32,
    core: [CompCore; 2],
    hp: [Biquad; 2],
}

impl VcaBus {
    pub fn new() -> Self {
        Self {
            sr: 48000.0,
            gain: 0.0,
            midside: false,
            thr: [0.0; 2],
            att_ms: [1.0; 2],
            ratio: [4.0; 2],
            rel: [0.6; 2],
            makeup: [0.0; 2],
            sidechain: 150.0,
            parallel: 100.0,
            output: 0.0,
            core: Default::default(),
            hp: Default::default(),
        }
    }

    fn update(&mut self) {
        for i in 0..2 {
            let c = &mut self.core[i];
            // Knob reads -15..+15; maps onto a -20 dBFS reference.
            c.threshold = self.thr[i] - 20.0;
            c.ratio = self.ratio[i].max(1.0);
            c.knee = 4.0;
            c.times(self.sr, self.att_ms[i] / 1000.0, self.rel[i]);
            self.hp[i].set(if self.sidechain > 20.0 { highpass(self.sr, self.sidechain) } else { Coefs::IDENTITY });
        }
    }
}

impl Effect for VcaBus {
    fn prepare(&mut self, sr: f32) {
        self.sr = sr;
        self.update();
    }

    fn set(&mut self, name: &str, v: f32) {
        let (i, f) = match name.strip_prefix("side-") {
            Some(f) => (1, f),
            None => (0, name),
        };
        match f {
            "gain" => self.gain = v,
            "midside-mode" => self.midside = on(v),
            "threshold" => self.thr[i] = v,
            "attack" => self.att_ms[i] = v,
            "ratio" => self.ratio[i] = v,
            "release" => self.rel[i] = v,
            "makeup" => self.makeup[i] = v,
            "sidechain" => self.sidechain = v,
            "parallel" => self.parallel = v,
            "output-level" => self.output = v,
            _ => return,
        }
        if i == 0 && !self.midside {
            self.thr[1] = self.thr[0];
            self.att_ms[1] = self.att_ms[0];
            self.ratio[1] = self.ratio[0];
            self.rel[1] = self.rel[0];
            self.makeup[1] = self.makeup[0];
        }
        self.update();
    }

    fn process(&mut self, _: &Ctx, l: &mut [f32], r: &mut [f32]) {
        let ms = self.midside;
        let gin = db_to_gain(self.gain);
        let mk = [db_to_gain(self.makeup[0]), db_to_gain(self.makeup[1])];
        let out = db_to_gain(self.output);
        let wet = self.parallel / 100.0;
        for (a, b) in l.iter_mut().zip(r.iter_mut()) {
            let (x, y) = if ms { to_ms(*a * gin, *b * gin) } else { (*a * gin, *b * gin) };
            let dx = self.hp[0].process(x).abs();
            let dy = self.hp[1].process(y).abs();
            let (gx, gy) = if ms {
                (self.core[0].gain(dx), self.core[1].gain(dy))
            } else {
                let g = self.core[0].gain(dx.max(dy));
                (g, g)
            };
            let (mut ox, mut oy) = (lerp(x, x * gx * mk[0], wet), lerp(y, y * gy * mk[if ms { 1 } else { 0 }], wet));
            if ms {
                (ox, oy) = from_ms(ox, oy);
            }
            *a = ox * out;
            *b = oy * out;
        }
    }
}

pub struct ListenMic {
    sr: f32,
    trim: f32,
    threshold: f32,
    drywet: f32,
    comp_in: bool,
    filter_in: bool,
    lf: f32,
    hf: f32,
    core: CompCore,
    f: [StereoBiquad; 2],
}

impl ListenMic {
    pub fn new() -> Self {
        Self {
            sr: 48000.0,
            trim: 0.0,
            threshold: -10.0,
            drywet: 100.0,
            comp_in: true,
            filter_in: false,
            lf: 30.0,
            hf: 15000.0,
            core: CompCore::default(),
            f: Default::default(),
        }
    }
}

impl Effect for ListenMic {
    fn prepare(&mut self, sr: f32) {
        self.sr = sr;
        self.core.times(sr, 0.0005, 0.25);
        self.core.ratio = 30.0;
        self.core.knee = 2.0;
        self.set("threshold", self.threshold);
    }

    fn set(&mut self, name: &str, v: f32) {
        match name {
            "trim" => self.trim = v,
            "threshold" => self.threshold = v,
            "drywet" => self.drywet = v,
            "compressor-in" => self.comp_in = on(v),
            "filter-in" => self.filter_in = on(v),
            "lffreq" => self.lf = v,
            "hffreq" => self.hf = v,
            _ => return,
        }
        self.core.threshold = self.threshold;
        self.f[0].set(if self.filter_in { highpass(self.sr, self.lf) } else { Coefs::IDENTITY });
        self.f[1].set(if self.filter_in && self.hf < self.sr * 0.45 { lowpass(self.sr, self.hf) } else { Coefs::IDENTITY });
    }

    fn process(&mut self, _: &Ctx, l: &mut [f32], r: &mut [f32]) {
        let gin = db_to_gain(self.trim);
        let wet = self.drywet / 100.0;
        let makeup = db_to_gain((-self.threshold * 0.6).max(0.0));
        for (a, b) in l.iter_mut().zip(r.iter_mut()) {
            let (dx, dy) = (*a * gin, *b * gin);
            let (mut x, mut y) = self.f[1].process(dx, dy);
            (x, y) = self.f[0].process(x, y);
            if self.comp_in {
                let g = self.core.gain(x.abs().max(y.abs())) * makeup;
                x = shape::tanh(x * g);
                y = shape::tanh(y * g);
            }
            *a = lerp(dx, x, wet);
            *b = lerp(dy, y, wet);
        }
    }
}

// ---------------------------------------------------------------------------
// dbx 160 (TA_DBX160)

pub struct Dbx160 {
    sr: f32,
    threshold_v: f32,
    ratio: f32,
    out: f32,
    core: CompCore,
}

impl Dbx160 {
    pub fn new() -> Self {
        Self { sr: 48000.0, threshold_v: 1.0, ratio: 4.0, out: 0.0, core: CompCore::default() }
    }

    fn update(&mut self) {
        // Threshold in volts; 0.775 V = 0 dBu which we place at -18 dBFS.
        self.core.threshold = 20.0 * (self.threshold_v.max(0.005) / 0.775).log10() - 18.0;
        self.core.ratio = self.ratio.max(1.0);
        self.core.knee = 1.0;
        // RMS-ish program-dependent timing of the original.
        self.core.times(self.sr, 0.003, 0.12);
    }
}

impl Effect for Dbx160 {
    fn prepare(&mut self, sr: f32) {
        self.sr = sr;
        self.update();
    }

    fn set(&mut self, name: &str, v: f32) {
        match name {
            "threshold" => self.threshold_v = v,
            "ratio" => self.ratio = v,
            "outlevel" => self.out = v,
            _ => return,
        }
        self.update();
    }

    fn process(&mut self, _: &Ctx, l: &mut [f32], r: &mut [f32]) {
        let out = db_to_gain(self.out);
        let core = &mut self.core;
        for_each(l, r, |a, b| {
            let g = core.gain(a.abs().max(b.abs())) * out;
            (a * g, b * g)
        });
    }
}

// ---------------------------------------------------------------------------
// MXR Dyna Comp style (TA_LIMITER143)

pub struct PedalLimiter {
    sr: f32,
    sens: f32,
    attack: f32,
    release: f32,
    level: f32,
    link: bool,
    core: [CompCore; 2],
}

impl PedalLimiter {
    pub fn new() -> Self {
        Self { sr: 48000.0, sens: 50.0, attack: 50.0, release: 50.0, level: 0.0, link: false, core: Default::default() }
    }

    fn update(&mut self) {
        for c in &mut self.core {
            c.threshold = -self.sens * 0.4;
            c.ratio = 8.0;
            c.knee = 6.0;
            c.times(self.sr, lerp(0.0005, 0.03, self.attack / 100.0), lerp(0.05, 1.0, self.release / 100.0));
        }
    }
}

impl Effect for PedalLimiter {
    fn prepare(&mut self, sr: f32) {
        self.sr = sr;
        self.update();
    }

    fn set(&mut self, name: &str, v: f32) {
        match name {
            "sensitivity" => self.sens = v,
            "attack" => self.attack = v,
            "release" => self.release = v,
            "level" => self.level = v,
            "stereo-lnk" => self.link = on(v),
            _ => return,
        }
        self.update();
    }

    fn process(&mut self, _: &Ctx, l: &mut [f32], r: &mut [f32]) {
        let mk = db_to_gain(self.sens * 0.2 + self.level);
        for (a, b) in l.iter_mut().zip(r.iter_mut()) {
            let (ga, gb) = if self.link {
                let g = self.core[0].gain(a.abs().max(b.abs()));
                (g, g)
            } else {
                (self.core[0].gain(a.abs()), self.core[1].gain(b.abs()))
            };
            *a = shape::tanh(*a * ga * mk);
            *b = shape::tanh(*b * gb * mk);
        }
    }
}

// ---------------------------------------------------------------------------
// One-knob "Overloud" compressor/maximizer (TA_TOONTRACK_OVERLOUD)

pub struct OneKnob {
    sr: f32,
    amount: f32,
    core: CompCore,
}

impl OneKnob {
    pub fn new() -> Self {
        Self { sr: 48000.0, amount: 0.0, core: CompCore::default() }
    }
}

impl Effect for OneKnob {
    fn prepare(&mut self, sr: f32) {
        self.sr = sr;
        self.core.times(sr, 0.003, 0.15);
        self.set("Amount", self.amount);
    }

    fn set(&mut self, name: &str, v: f32) {
        if name == "Amount" {
            self.amount = v;
            let a = v / 100.0;
            self.core.threshold = -30.0 * a;
            self.core.ratio = 1.0 + 5.0 * a;
            self.core.knee = 10.0;
        }
    }

    fn process(&mut self, _: &Ctx, l: &mut [f32], r: &mut [f32]) {
        let a = self.amount / 100.0;
        let makeup = db_to_gain(30.0 * a * 0.45);
        let core = &mut self.core;
        for_each(l, r, |x, y| {
            let g = core.gain(x.abs().max(y.abs())) * makeup;
            (shape::tanh(x * g), shape::tanh(y * g))
        });
    }
}

// ---------------------------------------------------------------------------
// Punch / compander boxes (TA_DBX180A, TA_DOLBY361, TA_DOLBY361MOD)

pub struct Compander {
    sr: f32,
    kind: CompanderKind,
    compression: f32,
    dry: f32,
    wet: f32,
    level: f32,
    split: [FourBandSplit; 2],
    core: [CompCore; 4],
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum CompanderKind {
    Punch,
    Dolby,
    DolbyVocal,
}

impl Compander {
    pub fn new(kind: CompanderKind) -> Self {
        Self {
            sr: 48000.0,
            kind,
            compression: 50.0,
            dry: 50.0,
            wet: 50.0,
            level: 0.0,
            split: Default::default(),
            core: Default::default(),
        }
    }

    fn update(&mut self) {
        // Dolby A band edges.
        for s in &mut self.split {
            s.set(self.sr, [80.0, 3000.0, 9000.0]);
        }
        let c = self.compression / 100.0;
        for (i, core) in self.core.iter_mut().enumerate() {
            match self.kind {
                CompanderKind::Punch => {
                    core.threshold = -10.0 - 25.0 * c;
                    core.ratio = 2.0 + 6.0 * c;
                    core.times(self.sr, 0.004, 0.08);
                }
                _ => {
                    // Encoder-style low level boost: heavy compression of quiet
                    // detail, the classic "Dolby trick" brightness.
                    core.threshold = -40.0 + 10.0 * (1.0 - c);
                    core.ratio = 2.0 + 2.0 * c;
                    core.times(self.sr, 0.002 + 0.002 * i as f32, 0.05);
                }
            }
            core.knee = 12.0;
        }
    }
}

impl Effect for Compander {
    fn prepare(&mut self, sr: f32) {
        self.sr = sr;
        self.update();
    }

    fn set(&mut self, name: &str, v: f32) {
        match name {
            "compression" => self.compression = v,
            "dry" => self.dry = v,
            "wet" => self.wet = v,
            "level" => self.level = v,
            _ => return,
        }
        self.update();
    }

    fn process(&mut self, _: &Ctx, l: &mut [f32], r: &mut [f32]) {
        let (dry, wet, out) = (self.dry / 100.0, self.wet / 100.0, db_to_gain(self.level));
        let weights: [f32; 4] = match self.kind {
            CompanderKind::Punch => [1.0, 1.0, 1.0, 1.0],
            CompanderKind::Dolby => [0.3, 0.6, 1.0, 1.0],
            CompanderKind::DolbyVocal => [0.1, 0.8, 1.2, 0.8],
        };
        for (a, b) in l.iter_mut().zip(r.iter_mut()) {
            let (x, y) = (*a, *b);
            let (wx, wy) = if self.kind == CompanderKind::Punch {
                let g = self.core[0].gain(x.abs().max(y.abs()));
                let mk = db_to_gain(self.compression * 0.12);
                (x * g * mk, y * g * mk)
            } else {
                let bl = self.split[0].split(x);
                let br = self.split[1].split(y);
                let (mut ox, mut oy) = (0.0, 0.0);
                for i in 0..4 {
                    let g = self.core[i].gain(bl[i].abs().max(br[i].abs()));
                    let mk = db_to_gain(-self.core[i].threshold * (1.0 - 1.0 / self.core[i].ratio) * 0.5);
                    ox += bl[i] * g * mk * weights[i];
                    oy += br[i] * g * mk * weights[i];
                }
                (ox, oy)
            };
            *a = (x * dry + wx * wet) * out;
            *b = (y * dry + wy * wet) * out;
        }
    }
}

// ---------------------------------------------------------------------------
// De-esser (TA_DEESSER)

pub struct DeEsser {
    sr: f32,
    split_band: bool,
    threshold: f32,
    freq: f32,
    bandpass: bool,
    sc: [Biquad; 2],
    band: [Crossover; 2],
    core: CompCore,
}

impl DeEsser {
    pub fn new() -> Self {
        Self {
            sr: 48000.0,
            split_band: false,
            threshold: 0.0,
            freq: 5000.0,
            bandpass: false,
            sc: Default::default(),
            band: Default::default(),
            core: CompCore::default(),
        }
    }

    fn update(&mut self) {
        let c = if self.bandpass {
            Coefs::rbj(BiquadKind::BandPass, self.sr, self.freq, 1.4, 0.0)
        } else {
            highpass(self.sr, self.freq)
        };
        for f in &mut self.sc {
            f.set(c);
        }
        for b in &mut self.band {
            b.set(self.sr, self.freq);
        }
        self.core.threshold = self.threshold;
        self.core.ratio = 6.0;
        self.core.knee = 6.0;
        self.core.times(self.sr, 0.0005, 0.06);
    }
}

impl Effect for DeEsser {
    fn prepare(&mut self, sr: f32) {
        self.sr = sr;
        self.update();
    }

    fn set(&mut self, name: &str, v: f32) {
        match name {
            "mode" => self.split_band = on(v),
            "threshold" => self.threshold = v,
            "freq" => self.freq = v,
            "sidechain" => self.bandpass = on(v),
            _ => return,
        }
        self.update();
    }

    fn process(&mut self, _: &Ctx, l: &mut [f32], r: &mut [f32]) {
        for (a, b) in l.iter_mut().zip(r.iter_mut()) {
            let d = self.sc[0].process(*a).abs().max(self.sc[1].process(*b).abs());
            let g = self.core.gain(d);
            if self.split_band {
                let (lo_a, hi_a) = self.band[0].split(*a);
                let (lo_b, hi_b) = self.band[1].split(*b);
                *a = lo_a + hi_a * g;
                *b = lo_b + hi_b * g;
            } else {
                *a *= g;
                *b *= g;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Gate / expander with sidechain filters (TA_FREQ_GATE)

pub struct Gate {
    sr: f32,
    gate_thresh: f32,
    exp_thresh: f32,
    attack: f32,
    hold: f32,
    release: f32,
    ratio: f32,
    low_cut: f32,
    high_cut: f32,
    sc: [StereoBiquad; 2],
    env: Envelope,
    gain: f32,
    hold_count: f32,
    att_c: f32,
    rel_c: f32,
}

impl Gate {
    pub fn new() -> Self {
        Self {
            sr: 48000.0,
            gate_thresh: -40.0,
            exp_thresh: -30.0,
            attack: 0.005,
            hold: 0.0,
            release: 0.1,
            ratio: 0.5,
            low_cut: 20.0,
            high_cut: 20000.0,
            sc: Default::default(),
            env: Envelope::default(),
            gain: 1.0,
            hold_count: 0.0,
            att_c: 0.0,
            rel_c: 0.0,
        }
    }

    fn update(&mut self) {
        self.sc[0].set(if self.low_cut > 21.0 { highpass(self.sr, self.low_cut) } else { Coefs::IDENTITY });
        self.sc[1].set(if self.high_cut < 19000.0 { lowpass(self.sr, self.high_cut) } else { Coefs::IDENTITY });
        self.env.set(self.sr, 0.0002, 0.02);
        self.att_c = time_coef(self.attack, self.sr);
        self.rel_c = time_coef(self.release, self.sr);
    }
}

impl Effect for Gate {
    fn prepare(&mut self, sr: f32) {
        self.sr = sr;
        self.update();
    }

    fn set(&mut self, name: &str, v: f32) {
        match name {
            "gate_thresh" => self.gate_thresh = v,
            "exp_thresh" => self.exp_thresh = v,
            "attack" => self.attack = v,
            "hold" => self.hold = v,
            "release" => self.release = v,
            "ratio" => self.ratio = v,
            "low_cut" => self.low_cut = v,
            "high_cut" => self.high_cut = v,
            _ => return,
        }
        self.update();
    }

    fn process(&mut self, _: &Ctx, l: &mut [f32], r: &mut [f32]) {
        let hold_samples = self.hold * self.sr;
        for (a, b) in l.iter_mut().zip(r.iter_mut()) {
            let (x, y) = self.sc[0].process(*a, *b);
            let (x, y) = self.sc[1].process(x, y);
            let lvl = gain_to_db(self.env.process(x.abs().max(y.abs())));
            let target = if lvl < self.gate_thresh {
                0.0
            } else if lvl < self.exp_thresh {
                // Downward expansion; ratio is the slope of the gain below the
                // expander threshold (1 = no expansion).
                let below = self.exp_thresh - lvl;
                db_to_gain(-below * (1.0 / self.ratio.max(0.05) - 1.0).min(10.0))
            } else {
                1.0
            };
            if target >= self.gain {
                self.hold_count = hold_samples;
                self.gain = target + (self.gain - target) * self.att_c;
            } else if self.hold_count > 0.0 {
                self.hold_count -= 1.0;
            } else {
                self.gain = target + (self.gain - target) * self.rel_c;
            }
            *a *= self.gain;
            *b *= self.gain;
        }
    }
}

// ---------------------------------------------------------------------------
// Transient designer (TA_TRANSIENT_DESIGNER)

pub struct TransientShaper {
    sr: f32,
    attack: f32,
    sustain: f32,
    fast: Envelope,
    slow: Envelope,
    sus_fast: Envelope,
    sus_slow: Envelope,
}

impl TransientShaper {
    pub fn new() -> Self {
        Self {
            sr: 48000.0,
            attack: 0.0,
            sustain: 0.0,
            fast: Envelope::default(),
            slow: Envelope::default(),
            sus_fast: Envelope::default(),
            sus_slow: Envelope::default(),
        }
    }
}

impl Effect for TransientShaper {
    fn prepare(&mut self, sr: f32) {
        self.sr = sr;
        self.fast.set(sr, 0.0005, 0.05);
        self.slow.set(sr, 0.02, 0.05);
        self.sus_fast.set(sr, 0.002, 0.03);
        self.sus_slow.set(sr, 0.002, 0.5);
    }

    fn set(&mut self, name: &str, v: f32) {
        match name {
            "attack" => self.attack = v,
            "sustain" => self.sustain = v,
            _ => {}
        }
    }

    fn process(&mut self, _: &Ctx, l: &mut [f32], r: &mut [f32]) {
        for (a, b) in l.iter_mut().zip(r.iter_mut()) {
            let x = a.abs().max(b.abs()) + 1e-9;
            let att = gain_to_db(self.fast.process(x)) - gain_to_db(self.slow.process(x));
            let sus = gain_to_db(self.sus_slow.process(x)) - gain_to_db(self.sus_fast.process(x));
            // Attack / sustain are the maximum gain changes (dB) applied to the
            // transient and sustain portions; the detectors give 0..1 weights.
            let tw = (att / 12.0).clamp(0.0, 1.0);
            let sw = (sus / 12.0).clamp(0.0, 1.0);
            let g_db = (self.attack * tw + self.sustain * sw).clamp(-40.0, 18.0);
            let g = db_to_gain(g_db);
            *a *= g;
            *b *= g;
        }
    }
}

// ---------------------------------------------------------------------------
// Aural exciter (TA_TTEXCITER)

pub struct Exciter {
    sr: f32,
    tune: f32,
    harmonics: f32,
    threshold: f32,
    mix: f32,
    hp: [Butterworth; 2],
    post: [Biquad; 2],
}

impl Exciter {
    pub fn new() -> Self {
        Self {
            sr: 48000.0,
            tune: 3000.0,
            harmonics: 1.0,
            threshold: -6.0,
            mix: -10.0,
            hp: Default::default(),
            post: Default::default(),
        }
    }

    fn update(&mut self) {
        // Older presets store the frequency in kHz.
        let f = if self.tune < 20.0 { self.tune * 1000.0 } else { self.tune }.clamp(500.0, 16000.0);
        for h in &mut self.hp {
            h.design(self.sr, f, 3, true);
        }
        for p in &mut self.post {
            p.set(highpass(self.sr, f * 0.8));
        }
    }
}

impl Effect for Exciter {
    fn prepare(&mut self, sr: f32) {
        self.sr = sr;
        self.update();
    }

    fn set(&mut self, name: &str, v: f32) {
        match name {
            "tune" => self.tune = v,
            "harmonics" => self.harmonics = v,
            "threshold" => self.threshold = v,
            "mix" => self.mix = v,
            _ => return,
        }
        self.update();
    }

    fn process(&mut self, _: &Ctx, l: &mut [f32], r: &mut [f32]) {
        let mix = db_to_gain(self.mix);
        let drive = db_to_gain(-self.threshold) * (1.0 + self.harmonics / 10.0);
        let bias = self.harmonics / 100.0 * 0.5;
        for (ch, s) in [l, r].into_iter().enumerate() {
            for x in s.iter_mut() {
                let h = self.hp[ch].process(*x);
                let e = self.post[ch].process(shape::tube(h * drive, bias) / drive.sqrt());
                *x += e * mix;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Harmonizer / pitch shifter (TA_EVENHARM)

pub struct Harmonizer {
    sr: f32,
    continuous: bool,
    coarse: f32,
    fine: f32,
    cont: f32,
    delay: crate::effect::SyncParam,
    feedback: f32,
    anti: f32,
    mix: f32,
    shifter: [crate::fx::modulation::PitchShifter; 2],
    line: [DelayLine; 2],
    fb: [f32; 2],
}

impl Harmonizer {
    pub fn new() -> Self {
        Self {
            sr: 48000.0,
            continuous: false,
            coarse: 12.0,
            fine: 0.0,
            cont: 0.0,
            delay: crate::effect::SyncParam::time(0.0, crate::effect::DELAY_NOTES),
            feedback: 0.0,
            anti: 0.0,
            mix: 100.0,
            shifter: Default::default(),
            line: [DelayLine::new(1 << 18), DelayLine::new(1 << 18)],
            fb: [0.0; 2],
        }
    }
}

impl Effect for Harmonizer {
    fn prepare(&mut self, sr: f32) {
        self.sr = sr;
        for s in &mut self.shifter {
            s.prepare(sr);
        }
        let n = (sr * 2.5) as usize;
        for l in &mut self.line {
            l.resize(n);
        }
    }

    fn set(&mut self, name: &str, v: f32) {
        match name {
            "pitchmode" => self.continuous = on(v),
            // Selector 0..24 => -12..+12 semitones.
            "pitchcoarse" => self.coarse = v,
            "pitchfine" => self.fine = v,
            "pitchcont" => self.cont = v,
            "delay" => self.delay.free = v,
            "feedback" => self.feedback = v,
            "antifeedback" => self.anti = v,
            "mix" => self.mix = v,
            _ => {}
        }
    }

    fn set_sync(&mut self, name: &str, unsynced: f32, synced: f32, sync: bool) {
        if name == "delay" {
            self.delay.set(unsynced, synced, sync);
        } else {
            self.set(name, unsynced);
        }
    }

    fn process(&mut self, ctx: &Ctx, l: &mut [f32], r: &mut [f32]) {
        // Coarse mode uses semitone steps plus fine cents; continuous mode the
        // free pitch control.
        let semis = if self.continuous { self.cont } else { (self.coarse - 12.0) + self.fine / 100.0 };
        let ratio = 2f32.powf(semis / 12.0);
        let d = (self.delay.value(ctx.bpm) * self.sr).max(1.0);
        let fb = self.feedback / 100.0 * (1.0 - self.anti / 200.0);
        let mix = self.mix / 100.0;
        for (ch, s) in [l, r].into_iter().enumerate() {
            for x in s.iter_mut() {
                let dry = *x;
                let shifted = self.shifter[ch].process(dry + shape::tanh(self.fb[ch] * fb), ratio);
                self.line[ch].push(shifted);
                let wet = self.line[ch].read(d);
                self.fb[ch] = wet;
                *x = lerp(dry, wet, mix);
            }
        }
    }
}
