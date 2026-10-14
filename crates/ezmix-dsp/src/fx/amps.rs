//! Amplifier, cabinet and rotary speaker simulations.

use super::dist::Os2;
use crate::dsp::*;
use crate::effect::{idx, on, Ctx, Effect};

/// Static description of an amp voicing.
#[derive(Clone, Copy, Debug)]
struct Voicing {
    /// Preamp gain range in dB for gain 0..1.
    gain_db: (f32, f32),
    stages: usize,
    bias: f32,
    /// Coupling high-pass between stages.
    couple_hz: f32,
    /// Preamp Miller low-pass.
    pre_lp_hz: f32,
    bass_hz: f32,
    mid_hz: f32,
    treble_hz: f32,
    /// Fixed mid scoop (dB) of the passive tone stack at noon.
    scoop_db: f32,
    /// Power amp drive range (dB) for master 0..1.
    power_db: (f32, f32),
    out_lp_hz: f32,
    bass_amp: bool,
}

const FENDER_TWIN: Voicing = Voicing {
    gain_db: (0.0, 22.0),
    stages: 1,
    bias: 0.05,
    couple_hz: 30.0,
    pre_lp_hz: 12000.0,
    bass_hz: 100.0,
    mid_hz: 500.0,
    treble_hz: 2500.0,
    scoop_db: -6.0,
    power_db: (-6.0, 8.0),
    out_lp_hz: 9000.0,
    bass_amp: false,
};

const fn high_gain(gain_hi: f32, scoop: f32, mid: f32) -> Voicing {
    Voicing {
        gain_db: (12.0, gain_hi),
        stages: 3,
        bias: 0.2,
        couple_hz: 120.0,
        pre_lp_hz: 7000.0,
        bass_hz: 110.0,
        mid_hz: mid,
        treble_hz: 2800.0,
        scoop_db: scoop,
        power_db: (-6.0, 10.0),
        out_lp_hz: 8000.0,
        bass_amp: false,
    }
}

const fn marshall(gain_lo: f32, gain_hi: f32, stages: usize) -> Voicing {
    Voicing {
        gain_db: (gain_lo, gain_hi),
        stages,
        bias: 0.15,
        couple_hz: 80.0,
        pre_lp_hz: 9000.0,
        bass_hz: 120.0,
        mid_hz: 700.0,
        treble_hz: 2500.0,
        scoop_db: -2.0,
        power_db: (-4.0, 12.0),
        out_lp_hz: 8000.0,
        bass_amp: false,
    }
}

const VOX: Voicing = Voicing {
    gain_db: (3.0, 30.0),
    stages: 2,
    bias: 0.3,
    couple_hz: 60.0,
    pre_lp_hz: 10000.0,
    bass_hz: 120.0,
    mid_hz: 900.0,
    treble_hz: 3000.0,
    scoop_db: -1.0,
    power_db: (-2.0, 14.0),
    out_lp_hz: 9000.0,
    bass_amp: false,
};

const TWEED: Voicing = Voicing {
    gain_db: (2.0, 30.0),
    stages: 2,
    bias: 0.35,
    couple_hz: 40.0,
    pre_lp_hz: 8000.0,
    bass_hz: 120.0,
    mid_hz: 800.0,
    treble_hz: 2200.0,
    scoop_db: 0.0,
    power_db: (0.0, 16.0),
    out_lp_hz: 7000.0,
    bass_amp: false,
};

const BASS_TUBE: Voicing = Voicing {
    gain_db: (0.0, 30.0),
    stages: 2,
    bias: 0.15,
    couple_hz: 20.0,
    pre_lp_hz: 9000.0,
    bass_hz: 60.0,
    mid_hz: 500.0,
    treble_hz: 3000.0,
    scoop_db: 0.0,
    power_db: (-4.0, 10.0),
    out_lp_hz: 6000.0,
    bass_amp: true,
};

const LAB_SERIES: Voicing = Voicing {
    gain_db: (0.0, 24.0),
    stages: 1,
    bias: 0.05,
    couple_hz: 40.0,
    pre_lp_hz: 12000.0,
    bass_hz: 100.0,
    mid_hz: 800.0,
    treble_hz: 3000.0,
    scoop_db: 0.0,
    power_db: (-6.0, 6.0),
    out_lp_hz: 10000.0,
    bass_amp: false,
};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AmpKind {
    /// TA_GUITAR_AMP with its `model` selector.
    Guitar,
    SvtClassic,
    TweedDeluxe,
    Mesa50Clean,
    Mesa50Lead,
    LabSeriesClean,
    LabSeriesDist,
    SuperLead,
    BassTube,
    AmpegB15,
}

pub struct Amp {
    sr: f32,
    kind: AmpKind,
    model: usize,
    gain: f32,
    bass: f32,
    mid: f32,
    treble: f32,
    presence: f32,
    master: f32,
    level: f32,
    bright: bool,
    ultra_lo: bool,
    ultra_hi: bool,
    mid_freq: f32,
    blend: f32,
    graphic: [f32; 5],
    compression: f32,
    v: Voicing,
    input_hp: StereoBiquad,
    bright_f: StereoBiquad,
    couple: [StereoBiquad; 3],
    pre_lp: [StereoBiquad; 3],
    stack: [StereoBiquad; 3],
    extra: [StereoBiquad; 7],
    presence_f: StereoBiquad,
    out_lp: StereoBiquad,
    os: Os2,
    sag: Envelope,
    comp: super::dynamics::CompCore,
    dc: [DcBlock; 2],
    dirty: bool,
}

impl Amp {
    pub fn new(kind: AmpKind) -> Self {
        let mut a = Self {
            sr: 48000.0,
            kind,
            model: 1,
            gain: 0.5,
            bass: 0.5,
            mid: 0.5,
            treble: 0.5,
            presence: 0.2,
            master: 0.8,
            level: 0.0,
            bright: false,
            ultra_lo: false,
            ultra_hi: false,
            mid_freq: 800.0,
            blend: 1.0,
            graphic: [0.0; 5],
            compression: 0.0,
            v: FENDER_TWIN,
            input_hp: Default::default(),
            bright_f: Default::default(),
            couple: Default::default(),
            pre_lp: Default::default(),
            stack: Default::default(),
            extra: Default::default(),
            presence_f: Default::default(),
            out_lp: Default::default(),
            os: Os2::new(),
            sag: Envelope::default(),
            comp: Default::default(),
            dc: Default::default(),
            dirty: true,
        };
        if matches!(kind, AmpKind::AmpegB15 | AmpKind::SvtClassic) {
            a.mid_freq = 800.0;
        }
        a
    }

    fn voicing(&self) -> Voicing {
        match self.kind {
            AmpKind::Guitar => match self.model {
                0 => FENDER_TWIN,
                1 => marshall(10.0, 45.0, 3),
                2 => high_gain(60.0, -8.0, 650.0),
                3 => VOX,
                4 => high_gain(62.0, -5.0, 750.0),
                5 => marshall(6.0, 38.0, 2),
                6 => marshall(0.0, 26.0, 2),
                7 => marshall(12.0, 50.0, 3),
                _ => marshall(3.0, 30.0, 2),
            },
            AmpKind::SvtClassic => Voicing { power_db: (-2.0, 12.0), ..BASS_TUBE },
            AmpKind::AmpegB15 => Voicing { gain_db: (0.0, 24.0), stages: 1, ..BASS_TUBE },
            AmpKind::BassTube => BASS_TUBE,
            AmpKind::TweedDeluxe => TWEED,
            AmpKind::Mesa50Clean => Voicing { gain_db: (0.0, 24.0), stages: 2, ..FENDER_TWIN },
            AmpKind::Mesa50Lead => high_gain(55.0, -4.0, 750.0),
            AmpKind::LabSeriesClean => LAB_SERIES,
            AmpKind::LabSeriesDist => Voicing { gain_db: (10.0, 45.0), stages: 3, bias: 0.25, ..LAB_SERIES },
            AmpKind::SuperLead => marshall(4.0, 34.0, 2),
        }
    }

    fn update(&mut self) {
        let sr = self.sr;
        self.v = self.voicing();
        let v = self.v;
        self.input_hp.set(Coefs::one_pole_hp(sr, if v.bass_amp { 15.0 } else { 60.0 }));
        let bright_db = if self.bright { 6.0 * (1.0 - self.gain) } else { 0.0 };
        self.bright_f.set(high_shelf(sr, 2000.0, bright_db));
        for i in 0..3 {
            self.couple[i].set(Coefs::one_pole_hp(sr, v.couple_hz * (1.0 + i as f32 * 0.3)));
            self.pre_lp[i].set(Coefs::one_pole_lp(sr, v.pre_lp_hz));
        }
        let tone = |x: f32, range: f32| (x - 0.5) * 2.0 * range;
        self.stack[0].set(Coefs::rbj(BiquadKind::LowShelf, sr, v.bass_hz, 0.6, tone(self.bass, 10.0) + if self.ultra_lo { 4.0 } else { 0.0 }));
        let mid_hz = if matches!(self.kind, AmpKind::SvtClassic | AmpKind::LabSeriesDist) { self.mid_freq } else { v.mid_hz };
        self.stack[1].set(peak(sr, mid_hz, 0.7, v.scoop_db * (1.0 - self.mid) * 2.0 + tone(self.mid, 6.0)));
        self.stack[2].set(Coefs::rbj(BiquadKind::HighShelf, sr, v.treble_hz, 0.6, tone(self.treble, 10.0) + if self.ultra_hi { 4.0 } else { 0.0 }));
        // Graphic EQ (Mesa: 80/240/750/2200/6600 Hz), ultra-low cut for SVT.
        let gfreqs = [80.0, 240.0, 750.0, 2200.0, 6600.0];
        for i in 0..5 {
            self.extra[i].set(if self.graphic[i].abs() > 1e-3 { peak(sr, gfreqs[i], 1.2, self.graphic[i]) } else { Coefs::IDENTITY });
        }
        self.extra[5].set(if self.ultra_lo && self.kind == AmpKind::SvtClassic { peak(sr, 500.0, 0.8, -6.0) } else { Coefs::IDENTITY });
        self.extra[6].set(Coefs::IDENTITY);
        self.presence_f.set(high_shelf(sr, 3500.0, tone(self.presence, 8.0)));
        self.out_lp.set(lowpass(sr, v.out_lp_hz));
        self.sag.set(sr, 0.01, 0.15);
        self.comp.threshold = -20.0;
        self.comp.ratio = 1.0 + self.compression * 6.0;
        self.comp.knee = 10.0;
        self.comp.times(sr, 0.003, 0.15);
        self.dirty = false;
    }
}

impl Effect for Amp {
    fn prepare(&mut self, sr: f32) {
        self.sr = sr;
        for d in &mut self.dc {
            d.prepare(sr);
        }
        self.dirty = true;
    }

    fn set(&mut self, name: &str, v: f32) {
        self.dirty = true;
        let k = knob10(v);
        match (self.kind, name) {
            (_, "model") => self.model = idx(v),
            (AmpKind::TweedDeluxe, "volume") => self.gain = k,
            (AmpKind::TweedDeluxe, "tone") => self.treble = k,
            (AmpKind::TweedDeluxe, "inp") | (_, "bright") | (_, "input") | (_, "in") => self.bright = on(v),
            (AmpKind::LabSeriesClean | AmpKind::LabSeriesDist, "volume") => self.gain = k,
            (_, "drive") | (_, "gain") => self.gain = k,
            (AmpKind::AmpegB15, "ch1volume") => self.gain = k,
            (AmpKind::AmpegB15, "ch1bass") | (_, "bass") => self.bass = k,
            (AmpKind::AmpegB15, "ch1treble") | (_, "treble") => self.treble = k,
            (AmpKind::AmpegB15, "ch1bright") => self.bright = on(v),
            (AmpKind::AmpegB15, "ch1ultralo") | (_, "ultra_lo") => self.ultra_lo = on(v),
            (AmpKind::AmpegB15, "ch1ultrahi") | (_, "ultra_hi") => self.ultra_hi = on(v),
            (_, "mid") | (_, "midrange") | (_, "mid-low") => self.mid = k,
            (_, "mid-high") => self.presence = k,
            (_, "mid_freq") => self.mid_freq = [220.0, 450.0, 800.0, 1600.0, 3000.0][idx(v).min(4)],
            (_, "midfreq") => self.mid_freq = v.max(100.0),
            (_, "presence") | (_, "shape") => self.presence = k,
            (AmpKind::SuperLead, "master") => self.level = v,
            (_, "volume") | (_, "master") => self.master = k,
            (_, "level") => self.level = v,
            (_, "blend") => self.blend = k,
            (_, "compression") => self.compression = k,
            (_, n) if n.starts_with("band") => {
                if let Ok(i) = n[4..].parse::<usize>() {
                    if i < 5 {
                        self.graphic[i] = v;
                    }
                }
            }
            _ => {}
        }
    }

    fn reset(&mut self) {
        self.sag.reset();
    }

    fn process(&mut self, _: &Ctx, l: &mut [f32], r: &mut [f32]) {
        if self.dirty {
            self.update();
        }
        let v = self.v;
        let g_total = db_to_gain(lerp(v.gain_db.0, v.gain_db.1, self.gain));
        let per_stage = g_total.powf(1.0 / v.stages as f32);
        let power = db_to_gain(lerp(v.power_db.0, v.power_db.1, self.master));
        let out = db_to_gain(self.level) * 0.5;
        let blend = self.blend;
        for i in 0..l.len() {
            let peak_in = l[i].abs().max(r[i].abs());
            let sag = 1.0 - 0.3 * self.sag.process(peak_in * per_stage).min(1.0);
            for ch in 0..2 {
                let x = if ch == 0 { l[i] } else { r[i] };
                let pick = |s: &mut StereoBiquad, v: f32| if ch == 0 { s.l.process(v) } else { s.r.process(v) };
                let mut y = pick(&mut self.input_hp, x);
                y = pick(&mut self.bright_f, y);
                let clean = y;
                let os = if ch == 0 { &mut self.os.l } else { &mut self.os.r };
                let bias = v.bias;
                let stages = v.stages;
                // Preamp stages run at 2x to keep aliasing down at high gain.
                y = os.process(y, |mut u| {
                    for st in 0..stages {
                        u = shape::tube(u * per_stage, bias * (1.0 - 0.3 * st as f32));
                    }
                    u
                });
                for s in 0..stages.min(3) {
                    y = pick(&mut self.couple[s], y);
                    y = pick(&mut self.pre_lp[s], y);
                }
                if blend < 1.0 {
                    y = lerp(clean * per_stage.min(4.0), y, blend);
                }
                for s in 0..3 {
                    y = pick(&mut self.stack[s], y);
                }
                for s in 0..7 {
                    y = pick(&mut self.extra[s], y);
                }
                if self.compression > 0.0 {
                    let g = self.comp.gain(y.abs());
                    y *= g;
                }
                // Push-pull power stage with sag and presence.
                y = pick(&mut self.presence_f, y);
                y = shape::tanh(y * power * sag) ;
                y = self.dc[ch].process(y);
                y = pick(&mut self.out_lp, y);
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
// Bass cabinets (TA_BASS8X10, TA_BASS1x15)

pub struct BassCab {
    sr: f32,
    eight_ten: bool,
    x_pos: bool,
    z_pos: bool,
    tweeter: bool,
    f: [StereoBiquad; 5],
}

impl BassCab {
    pub fn new(eight_ten: bool) -> Self {
        Self { sr: 48000.0, eight_ten, x_pos: false, z_pos: false, tweeter: false, f: Default::default() }
    }

    fn update(&mut self) {
        let sr = self.sr;
        let (res, lp) = if self.eight_ten { (90.0, 4500.0) } else { (65.0, 3000.0) };
        let lp = lp * if self.x_pos { 0.7 } else { 1.0 };
        self.f[0].set(highpass(sr, if self.eight_ten { 45.0 } else { 35.0 }));
        self.f[1].set(peak(sr, res, 1.5, 4.0));
        self.f[2].set(peak(sr, if self.eight_ten { 1500.0 } else { 900.0 }, 2.0, -5.0));
        self.f[3].set(Coefs::rbj(BiquadKind::LowPass, sr, lp, 0.9, 0.0));
        self.f[4].set(if self.tweeter { high_shelf(sr, 5000.0, 8.0) } else { lowpass(sr, lp * 1.5) });
        if self.z_pos {
            self.f[1].set(peak(sr, res, 1.5, 7.0));
        }
    }
}

impl Effect for BassCab {
    fn prepare(&mut self, sr: f32) {
        self.sr = sr;
        self.update();
    }

    fn set(&mut self, name: &str, v: f32) {
        match name {
            "x_pos" => self.x_pos = on(v),
            "z_pos" => self.z_pos = on(v),
            "tweeter" => self.tweeter = on(v),
            _ => return,
        }
        self.update();
    }

    fn process(&mut self, _: &Ctx, l: &mut [f32], r: &mut [f32]) {
        for (a, b) in l.iter_mut().zip(r.iter_mut()) {
            let (mut x, mut y) = (*a, *b);
            for f in &mut self.f {
                (x, y) = f.process(x, y);
            }
            *a = x;
            *b = y;
        }
    }
}

// ---------------------------------------------------------------------------
// Rotary speaker (TA_LESLIE, TA_LESLIE2, TA_LESLIE3)

pub struct Rotary {
    sr: f32,
    speed: usize,
    stop_index: usize,
    drive: f32,
    room: f32,
    lo_hi: f32,
    level: f32,
    bass_rate: (f32, f32),
    horn_rate: (f32, f32),
    bass_accel: (f32, f32),
    horn_accel: (f32, f32),
    spread: f32,
    mono: bool,
    split: [Crossover; 2],
    horn_line: [DelayLine; 2],
    drum_line: [DelayLine; 2],
    horn_phase: f32,
    drum_phase: f32,
    horn_speed: f32,
    drum_speed: f32,
    room_lines: [DelayLine; 2],
}

impl Rotary {
    /// `stop_index` is the selector value meaning "stop" (Leslie3 orders the
    /// positions Slow|Stop|Fast).
    pub fn new(stop_index: usize) -> Self {
        Self {
            sr: 48000.0,
            speed: 1,
            stop_index,
            drive: 10.0,
            room: -30.0,
            lo_hi: 0.0,
            level: 2.0,
            bass_rate: (0.4, 3.8),
            horn_rate: (0.5, 6.7),
            bass_accel: (3.5, 2.0),
            horn_accel: (1.5, 1.5),
            spread: 180.0,
            mono: false,
            split: Default::default(),
            horn_line: [DelayLine::new(1024), DelayLine::new(1024)],
            drum_line: [DelayLine::new(1024), DelayLine::new(1024)],
            horn_phase: 0.0,
            drum_phase: 0.25,
            horn_speed: 0.0,
            drum_speed: 0.0,
            room_lines: [DelayLine::new(8192), DelayLine::new(8192)],
        }
    }

    fn targets(&self) -> (f32, f32) {
        let fast_index = 2;
        if self.speed == self.stop_index {
            (0.0, 0.0)
        } else if self.speed == fast_index {
            (self.horn_rate.1, self.bass_rate.1)
        } else {
            (self.horn_rate.0, self.bass_rate.0)
        }
    }
}

impl Effect for Rotary {
    fn prepare(&mut self, sr: f32) {
        self.sr = sr;
        for s in &mut self.split {
            s.set(sr, 800.0);
        }
        let (h, d) = self.targets();
        self.horn_speed = h;
        self.drum_speed = d;
    }

    fn set(&mut self, name: &str, v: f32) {
        match name {
            "speed" => self.speed = idx(v),
            "drive" => self.drive = v,
            "room" => self.room = v,
            "lo-hi" => self.lo_hi = v,
            "level" => self.level = v,
            "bass-slow" => self.bass_rate.0 = v,
            "bass-fast" => self.bass_rate.1 = v,
            "horn-slow" => self.horn_rate.0 = v,
            "horn-fast" => self.horn_rate.1 = v,
            "bass-accel" => self.bass_accel.0 = v,
            "bass-decel" => self.bass_accel.1 = v,
            "horn-accel" => self.horn_accel.0 = v,
            "horn-decel" => self.horn_accel.1 = v,
            "mic-spread" => self.spread = v,
            "mono" => self.mono = on(v),
            _ => {}
        }
    }

    fn process(&mut self, _: &Ctx, l: &mut [f32], r: &mut [f32]) {
        let sr = self.sr;
        let (ht, dt) = self.targets();
        let ramp = |cur: f32, target: f32, (acc, dec): (f32, f32)| {
            let t = if target > cur { acc } else { dec };
            target + (cur - target) * time_coef(t.max(0.05) / 3.0, sr)
        };
        let drive = db_to_gain(knob10(self.drive) * 12.0);
        let room = db_to_gain(self.room);
        let bal = self.lo_hi / 100.0;
        let (lo_g, hi_g) = (1.0 - bal.max(0.0), 1.0 + bal.min(0.0));
        let out = db_to_gain(self.level);
        let spread = (self.spread / 360.0).clamp(0.0, 0.5);
        let horn_depth = 0.00035 * sr;
        let drum_depth = 0.0002 * sr;
        for i in 0..l.len() {
            self.horn_speed = ramp(self.horn_speed, ht, self.horn_accel);
            self.drum_speed = ramp(self.drum_speed, dt, self.bass_accel);
            self.horn_phase = (self.horn_phase + self.horn_speed / sr).fract();
            self.drum_phase = (self.drum_phase + self.drum_speed / sr).fract();
            let input = if self.mono { (l[i] + r[i]) * 0.5 } else { l[i] };
            let x = shape::tanh(input * drive) / drive.sqrt();
            let (lo, hi) = self.split[0].split(x);
            for ch in 0..2 {
                self.horn_line[ch].push(hi);
                self.drum_line[ch].push(lo);
            }
            let mut outs = [0.0f32; 2];
            for (ch, o) in outs.iter_mut().enumerate() {
                let off = if ch == 0 { 0.0 } else { spread };
                let hp = TAU * (self.horn_phase + off);
                let dp = TAU * (self.drum_phase + off);
                let h = self.horn_line[ch].read(horn_depth * (1.2 + hp.sin())) * (1.0 + 0.45 * hp.cos());
                let d = self.drum_line[ch].read(drum_depth * (1.2 + dp.sin())) * (1.0 + 0.2 * dp.cos());
                *o = h * hi_g + d * lo_g;
            }
            if room > 0.0 {
                for ch in 0..2 {
                    self.room_lines[ch].push(outs[ch]);
                    let refl = self.room_lines[ch].tap((sr * 0.011) as usize + ch * 37)
                        + 0.7 * self.room_lines[1 - ch].tap((sr * 0.023) as usize)
                        + 0.5 * self.room_lines[ch].tap((sr * 0.037) as usize + ch * 53);
                    outs[ch] += refl * room;
                }
            }
            l[i] = outs[0] * out;
            r[i] = outs[1] * out;
        }
    }
}
