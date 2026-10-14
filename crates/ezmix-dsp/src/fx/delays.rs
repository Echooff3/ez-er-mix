//! Echo and delay effects.

use crate::dsp::*;
use crate::effect::{idx, on, Ctx, Effect, SyncParam, DELAY_NOTES, SPREAD_NOTES};

const MAX_DELAY_S: f32 = 6.0;

/// Reverse playback helper: reads a window of the past backwards with a
/// smooth envelope (two overlapping grains).
#[derive(Clone, Copy, Default)]
struct Reverser {
    phase: f32,
}

impl Reverser {
    #[inline]
    fn read(&mut self, line: &DelayLine, window: f32) -> f32 {
        let w = window.max(64.0);
        self.phase = (self.phase + 1.0 / w).fract();
        let mut out = 0.0;
        for k in 0..2 {
            let p = (self.phase + 0.5 * k as f32).fract();
            // Going forward in p means reading further back in time.
            let d = 1.0 + 2.0 * p * w;
            let env = (std::f32::consts::PI * p).sin();
            out += line.read_linear(d) * env * env;
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Tape delay (TA_ECHOTAPE) and filter delay (TA_ECHOFILTER)

pub struct TapeDelay {
    sr: f32,
    filter_mode: bool,
    time: SyncParam,
    spread: SyncParam,
    regen: f32,
    wow: f32,
    dist: f32,
    mix: f32,
    dry: f32,
    level: f32,
    flip: bool,
    reverse: bool,
    speed: f32,
    center: f32,
    depth_oct: f32,
    lines: [DelayLine; 2],
    rev: [Reverser; 2],
    lp: [Biquad; 2],
    hp: [Biquad; 2],
    svf: [Svf; 2],
    fb: [f32; 2],
    smooth_d: [f32; 2],
    lfo: Lfo,
    wow_lfo: Lfo,
}

impl TapeDelay {
    pub fn new(filter_mode: bool) -> Self {
        Self {
            sr: 48000.0,
            filter_mode,
            time: SyncParam::time(0.5, DELAY_NOTES),
            spread: SyncParam::time(0.0, SPREAD_NOTES),
            regen: 25.0,
            wow: 0.0,
            dist: 30.0,
            mix: 30.0,
            dry: 100.0,
            level: 0.0,
            flip: false,
            reverse: false,
            speed: 1.0,
            center: 4000.0,
            depth_oct: 1.0,
            lines: [DelayLine::new(1), DelayLine::new(1)],
            rev: Default::default(),
            lp: Default::default(),
            hp: Default::default(),
            svf: Default::default(),
            fb: [0.0; 2],
            smooth_d: [0.0; 2],
            lfo: Lfo::default(),
            wow_lfo: Lfo::default(),
        }
    }
}

impl Effect for TapeDelay {
    fn prepare(&mut self, sr: f32) {
        self.sr = sr;
        for l in &mut self.lines {
            l.resize((MAX_DELAY_S * sr) as usize);
        }
        for b in &mut self.lp {
            b.set(lowpass(sr, if self.filter_mode { 12000.0 } else { 5500.0 }));
        }
        for b in &mut self.hp {
            b.set(Coefs::one_pole_hp(sr, 90.0));
        }
        self.wow_lfo.freq = 0.7;
    }

    fn set(&mut self, name: &str, v: f32) {
        match name {
            "time" => self.time.free = v,
            "spread" => self.spread.free = v,
            "regen" => self.regen = v,
            "wow-flut" => self.wow = v,
            "dist" => self.dist = v,
            "mix" => self.mix = v,
            "dry" => self.dry = v,
            "level" => self.level = v,
            "stereo_flip" => self.flip = on(v),
            "reverse" => self.reverse = on(v),
            "speed" => self.speed = v,
            "center" => self.center = v,
            "depth" => self.depth_oct = v,
            _ => {}
        }
    }

    fn set_sync(&mut self, name: &str, unsynced: f32, synced: f32, sync: bool) {
        match name {
            "time" => self.time.set(unsynced, synced, sync),
            "spread" => self.spread.set(unsynced, synced, sync),
            _ => self.set(name, unsynced),
        }
    }

    fn reset(&mut self) {
        self.lines.iter_mut().for_each(DelayLine::clear);
        self.fb = [0.0; 2];
    }

    fn process(&mut self, ctx: &Ctx, l: &mut [f32], r: &mut [f32]) {
        let sr = self.sr;
        let t = self.time.value(ctx.bpm).clamp(0.001, MAX_DELAY_S - 0.5);
        let sp = self.spread.value(ctx.bpm).clamp(0.0, 1.0);
        let target = [t * sr, (t + sp) * sr];
        if self.smooth_d[0] == 0.0 {
            self.smooth_d = target;
        }
        let glide = time_coef(0.15, sr);
        let fb = (self.regen / 100.0).min(1.05);
        let drive = 1.0 + self.dist / 100.0 * 4.0;
        let (wet, dry, out) = (self.mix / 100.0, self.dry / 100.0, db_to_gain(self.level));
        let wow = self.wow / 100.0 * 0.004 * sr;
        self.lfo.freq = self.speed;
        for i in 0..l.len() {
            self.lfo.advance(sr);
            self.wow_lfo.advance(sr);
            let wm = wow * (self.wow_lfo.value(Wave::Sine, 0.0) + 0.3 * self.wow_lfo.value(Wave::Sine, 0.37));
            let input = [l[i], r[i]];
            let mut taps = [0.0f32; 2];
            for ch in 0..2 {
                self.smooth_d[ch] = target[ch] + (self.smooth_d[ch] - target[ch]) * glide;
                let d = (self.smooth_d[ch] + wm).max(2.0);
                taps[ch] = if self.reverse {
                    self.rev[ch].read(&self.lines[ch], d * 0.5)
                } else {
                    self.lines[ch].read(d)
                };
            }
            for ch in 0..2 {
                // Ping-pong when flipped: each line is fed by the other one and
                // only the left line receives (mono) input.
                let src = if self.flip { taps[1 - ch] } else { taps[ch] };
                let mut f = self.hp[ch].process(src * fb);
                if self.filter_mode {
                    let m = self.lfo.value(Wave::Sine, ch as f32 * 0.25);
                    let fc = (self.center * 2f32.powf(m * self.depth_oct)).clamp(100.0, sr * 0.4);
                    self.svf[ch].set(sr, fc, 2.0);
                    // Unity-peak bandpass (the SVF band output peaks at Q).
                    f = shape::tanh(self.svf[ch].process(f).band * 0.5);
                } else {
                    f = self.lp[ch].process(shape::tanh(f * drive) / drive);
                }
                let feed_in = match (self.flip, ch) {
                    (true, 0) => (input[0] + input[1]) * 0.5,
                    (true, _) => 0.0,
                    (false, _) => input[ch],
                };
                self.lines[ch].push(feed_in + f);
                self.fb[ch] = f;
            }
            l[i] = (input[0] * dry + taps[0] * wet) * out;
            r[i] = (input[1] * dry + taps[1] * wet) * out;
        }
    }
}

// ---------------------------------------------------------------------------
// Digital stereo echo (TA_AMSRMX16ECHO)

pub struct DigitalEcho {
    sr: f32,
    dry: f32,
    wet: f32,
    delay: [SyncParam; 2],
    regen: [f32; 2],
    reverse: bool,
    low: f32,
    high: f32,
    level: f32,
    lines: [DelayLine; 2],
    rev: [Reverser; 2],
    f: [StereoBiquad; 2],
}

impl DigitalEcho {
    pub fn new() -> Self {
        Self {
            sr: 48000.0,
            dry: 0.0,
            wet: -12.0,
            delay: [SyncParam::time(0.5, DELAY_NOTES); 2],
            regen: [3.0; 2],
            reverse: false,
            low: 0.0,
            high: 0.0,
            level: 0.0,
            lines: [DelayLine::new(1), DelayLine::new(1)],
            rev: Default::default(),
            f: Default::default(),
        }
    }

    fn update(&mut self) {
        // 0 = flat; positive values cut progressively.
        let lo = 20.0 * 2f32.powf(self.low.max(0.0));
        let hi = 20000.0 / 2f32.powf(self.high.max(0.0));
        self.f[0].set(if self.low > 0.0 { highpass(self.sr, lo) } else { Coefs::IDENTITY });
        self.f[1].set(if self.high > 0.0 { lowpass(self.sr, hi) } else { Coefs::IDENTITY });
    }
}

impl Effect for DigitalEcho {
    fn prepare(&mut self, sr: f32) {
        self.sr = sr;
        for l in &mut self.lines {
            l.resize((MAX_DELAY_S * sr) as usize);
        }
        self.update();
    }

    fn set(&mut self, name: &str, v: f32) {
        match name {
            "dry" => self.dry = v,
            "wet" => self.wet = v,
            "delay_l" => self.delay[0].free = v,
            "delay_r" => self.delay[1].free = v,
            "dell-synced" => self.delay[0].synced = on(v),
            "delr-synced" => self.delay[1].synced = on(v),
            "dell-bpm" => self.delay[0].index = idx(v),
            "delr-bpm" => self.delay[1].index = idx(v),
            "regen_l" => self.regen[0] = v,
            "regen_r" => self.regen[1] = v,
            "reverse" => self.reverse = on(v),
            "low" => self.low = v,
            "high" => self.high = v,
            "level" => self.level = v,
            _ => return,
        }
        self.update();
    }

    fn process(&mut self, ctx: &Ctx, l: &mut [f32], r: &mut [f32]) {
        let d = [
            (self.delay[0].value(ctx.bpm) * self.sr).clamp(2.0, (MAX_DELAY_S - 0.5) * self.sr),
            (self.delay[1].value(ctx.bpm) * self.sr).clamp(2.0, (MAX_DELAY_S - 0.5) * self.sr),
        ];
        let fb = [self.regen[0] / 10.0, self.regen[1] / 10.0];
        let (dry, wet, out) = (db_to_gain(self.dry), db_to_gain(self.wet), db_to_gain(self.level));
        for i in 0..l.len() {
            let mut taps = [0.0f32; 2];
            for ch in 0..2 {
                taps[ch] = if self.reverse { self.rev[ch].read(&self.lines[ch], d[ch] * 0.5) } else { self.lines[ch].read(d[ch]) };
            }
            let (fl, fr) = self.f[0].process(taps[0], taps[1]);
            let (fl, fr) = self.f[1].process(fl, fr);
            self.lines[0].push(l[i] + fl * fb[0]);
            self.lines[1].push(r[i] + fr * fb[1]);
            l[i] = (l[i] * dry + fl * wet) * out;
            r[i] = (r[i] * dry + fr * wet) * out;
        }
    }
}

// ---------------------------------------------------------------------------
// Tapped delay with modulation (TA_ADASTD1)

pub struct TappedDelay {
    sr: f32,
    mixa: f32,
    mixb: f32,
    level: f32,
    reg_on: bool,
    regen: f32,
    reg_col: f32,
    reg_tap: usize,
    routes: [usize; 6],
    del_fix: f32,
    del_mix: f32,
    sweep: f32,
    modu: f32,
    line: DelayLine,
    lp: Biquad,
    lfo: Lfo,
}

impl TappedDelay {
    pub fn new() -> Self {
        Self {
            sr: 48000.0,
            mixa: 50.0,
            mixb: 50.0,
            level: 0.0,
            reg_on: false,
            regen: 0.0,
            reg_col: 3000.0,
            reg_tap: 2,
            routes: [0, 1, 0, 1, 0, 1],
            del_fix: 3.0,
            del_mix: 20.0,
            sweep: 2.0,
            modu: 0.0,
            line: DelayLine::new(1),
            lp: Biquad::identity(),
            lfo: Lfo::default(),
        }
    }
}

impl Effect for TappedDelay {
    fn prepare(&mut self, sr: f32) {
        self.sr = sr;
        self.line.resize((2.0 * sr) as usize);
        self.lp.set(lowpass(sr, self.reg_col));
    }

    fn set(&mut self, name: &str, v: f32) {
        match name {
            "mixa" => self.mixa = v,
            "mixb" => self.mixb = v,
            "level" => self.level = v,
            "reg_on" => self.reg_on = on(v),
            "regen" => self.regen = v,
            "reg_col" => {
                self.reg_col = v;
                self.lp.set(lowpass(self.sr, v.clamp(200.0, 18000.0)));
            }
            "reg_tap" => self.reg_tap = idx(v),
            "del_fix" => self.del_fix = v,
            "del_mix" => self.del_mix = v,
            "del_sweep" => self.sweep = v,
            "del_mod" => self.modu = v,
            n if n.starts_with("reg") && n.len() == 4 => {
                if let Some(i) = n[3..].parse::<usize>().ok().filter(|i| (1..=6).contains(i)) {
                    self.routes[i - 1] = idx(v);
                }
            }
            _ => {}
        }
    }

    fn process(&mut self, _: &Ctx, l: &mut [f32], r: &mut [f32]) {
        let sr = self.sr;
        // Tap spacing: del_fix (x) multiplies a 25 ms base.
        let spacing = (0.025 * self.del_fix.max(0.2)) * sr;
        self.lfo.freq = self.sweep;
        let depth = self.modu / 100.0 * 0.004 * sr;
        let fb_tap = [1usize, 3, 6][self.reg_tap.min(2)];
        let fb = if self.reg_on { self.regen / 100.0 } else { 0.0 };
        let (ma, mb, dm, out) = (self.mixa / 100.0, self.mixb / 100.0, self.del_mix / 100.0, db_to_gain(self.level));
        for i in 0..l.len() {
            self.lfo.advance(sr);
            let m = self.lfo.value(Wave::Triangle, 0.0) * depth;
            let mut acc = [0.0f32; 2];
            let mut fb_sig = 0.0;
            for k in 0..6 {
                let tap = self.line.read(spacing * (k + 1) as f32 + m + 2.0);
                if k + 1 == fb_tap {
                    fb_sig = tap;
                }
                match self.routes[k] {
                    0 => acc[0] += tap,
                    1 => acc[1] += tap,
                    _ => {}
                }
            }
            let x = (l[i] + r[i]) * 0.5;
            self.line.push(x + self.lp.process(fb_sig) * fb);
            l[i] = (l[i] * ma + acc[0] * dm) * out;
            r[i] = (r[i] * mb + acc[1] * dm) * out;
        }
    }
}

// ---------------------------------------------------------------------------
// Multi-head drum echo (TA_ECHOSON)

pub struct DrumEcho {
    sr: f32,
    in_level: f32,
    mix: f32,
    out_level: f32,
    time: f32,
    swell: f32,
    tone: f32,
    wow: f32,
    mode: usize,
    play: [bool; 4],
    stereo: bool,
    stereo_mode: usize,
    time2: f32,
    swell2: f32,
    mode2: usize,
    play2: [bool; 4],
    lines: [DelayLine; 2],
    tone_f: [Biquad; 2],
    hp: [Biquad; 2],
    lfo: Lfo,
}

impl DrumEcho {
    pub fn new() -> Self {
        Self {
            sr: 48000.0,
            in_level: 0.0,
            mix: 2.0,
            out_level: 0.0,
            time: 0.31,
            swell: 2.0,
            tone: 0.0,
            wow: 0.0,
            mode: 0,
            play: [false, false, false, true],
            stereo: false,
            stereo_mode: 0,
            time2: 1.0,
            swell2: 2.0,
            mode2: 0,
            play2: [false, false, false, true],
            lines: [DelayLine::new(1), DelayLine::new(1)],
            tone_f: Default::default(),
            hp: Default::default(),
            lfo: Lfo::default(),
        }
    }
}

impl Effect for DrumEcho {
    fn prepare(&mut self, sr: f32) {
        self.sr = sr;
        for l in &mut self.lines {
            l.resize((4.0 * sr) as usize);
        }
        for h in &mut self.hp {
            h.set(Coefs::one_pole_hp(sr, 120.0));
        }
        self.set("tone", self.tone);
        self.lfo.freq = 1.3;
    }

    fn set(&mut self, name: &str, v: f32) {
        match name {
            "in_level" => self.in_level = v,
            "mix" => self.mix = v,
            "out_level" => self.out_level = v,
            "time" => self.time = v,
            "swell" => self.swell = v,
            "tone" => {
                self.tone = v;
                for f in &mut self.tone_f {
                    f.set(Coefs::rbj(BiquadKind::HighShelf, self.sr, 2500.0, 0.7, v * 0.9 - 4.0));
                }
            }
            "wow" => self.wow = v,
            "mode" => self.mode = idx(v),
            "stereo-onoff" => self.stereo = on(v),
            "stereo-mode" => self.stereo_mode = idx(v),
            "time-2nd" => self.time2 = v,
            "swell-2nd" => self.swell2 = v,
            "mode-2nd" => self.mode2 = idx(v),
            n => {
                let second = n.ends_with("-2nd");
                let base = n.trim_end_matches("-2nd");
                if let Some(k) = base.strip_prefix("play").and_then(|k| k.parse::<usize>().ok()) {
                    if (1..=4).contains(&k) {
                        if second {
                            self.play2[k - 1] = on(v);
                        } else {
                            self.play[k - 1] = on(v);
                        }
                    }
                }
            }
        }
    }

    fn process(&mut self, _: &Ctx, l: &mut [f32], r: &mut [f32]) {
        let sr = self.sr;
        let gin = db_to_gain(self.in_level);
        let mix = knob10(self.mix);
        let out = db_to_gain(self.out_level);
        let wow = self.wow / 10.0 * 0.002 * sr;
        let units = [
            (self.time, self.swell, self.mode, self.play),
            (self.time * if self.stereo { self.time2 } else { 1.0 }, if self.stereo { self.swell2 } else { self.swell }, if self.stereo { self.mode2 } else { self.mode }, if self.stereo { self.play2 } else { self.play }),
        ];
        for i in 0..l.len() {
            self.lfo.advance(sr);
            let wm = wow * self.lfo.value(Wave::Sine, 0.0);
            let input = [l[i] * gin, r[i] * gin];
            let mut wet = [0.0f32; 2];
            for ch in 0..2 {
                let (time, swell, mode, play) = units[ch];
                let head = (time.clamp(0.02, 0.95) * sr) / 4.0;
                let mut sum = 0.0;
                let mut n = 0.0f32;
                for k in 0..4 {
                    if play[k] {
                        sum += self.lines[ch].read(head * (k + 1) as f32 + wm + 2.0);
                        n += 1.0;
                    }
                }
                let heads = if n > 0.0 { sum / n.sqrt() } else { 0.0 };
                // Echo: single repeats; Rep: feedback; Swell: heavy feedback.
                let fbk = match mode {
                    0 => 0.0,
                    1 => knob10(swell) * 0.9,
                    _ => 0.5 + knob10(swell) * 0.55,
                };
                let shaped = self.tone_f[ch].process(self.hp[ch].process(heads));
                let src = match (self.stereo, self.stereo_mode) {
                    (true, 1) => input[1 - ch],
                    _ => input[ch],
                };
                self.lines[ch].push(src + shape::tanh(shaped * fbk));
                wet[ch] = shaped;
            }
            l[i] = (lerp(input[0], wet[0], mix * 0.5) + wet[0] * mix * 0.5) * out;
            r[i] = (lerp(input[1], wet[1], mix * 0.5) + wet[1] * mix * 0.5) * out;
        }
    }
}
