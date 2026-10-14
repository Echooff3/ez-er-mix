//! Equalizers and filters.

use crate::dsp::*;
use crate::effect::{for_each, idx, on, Ctx, Effect};

// ---------------------------------------------------------------------------
// Five band parametric EQ (TA_TOONTRACK_EQ2) and its mid/side variant.

#[derive(Clone, Copy)]
struct Band {
    on: bool,
    shelf: bool,
    gain: f32,
    freq: f32,
    q: f32,
}

impl Band {
    fn new(freq: f32) -> Self {
        Self { on: true, shelf: false, gain: 0.0, freq, q: 1.0 }
    }

    fn coefs(&self, sr: f32, shelf_kind: BiquadKind) -> Coefs {
        if !self.on || self.gain.abs() < 1e-3 {
            return Coefs::IDENTITY;
        }
        let kind = if self.shelf { shelf_kind } else { BiquadKind::Peak };
        let q = if self.shelf { (self.q * std::f32::consts::FRAC_1_SQRT_2).clamp(0.3, 2.0) } else { self.q };
        Coefs::rbj(kind, sr, self.freq, q, self.gain)
    }
}

#[derive(Clone, Copy)]
struct FiveBand {
    bands: [Band; 5],
}

impl FiveBand {
    fn new() -> Self {
        Self { bands: [Band::new(100.0), Band::new(500.0), Band::new(1000.0), Band::new(2000.0), Band::new(10000.0)] }
    }

    fn band_index(name: &str) -> Option<usize> {
        Some(match name {
            "low" => 0,
            "mid1" => 1,
            "mid2" => 2,
            "mid3" => 3,
            "high" => 4,
            _ => return None,
        })
    }

    /// `name` is e.g. `low_gain`, `mid2_freq`.
    fn set(&mut self, name: &str, v: f32) -> bool {
        let Some((band, field)) = name.split_once('_') else { return false };
        let Some(i) = Self::band_index(band) else { return false };
        let b = &mut self.bands[i];
        match field {
            "onoff" => b.on = on(v),
            "mode" => b.shelf = on(v),
            "gain" => b.gain = v,
            "freq" => b.freq = v,
            "q" => b.q = v,
            _ => return false,
        }
        true
    }

    fn design(&self, sr: f32, out: &mut [StereoBiquad; 5]) {
        for (i, (b, f)) in self.bands.iter().zip(out.iter_mut()).enumerate() {
            let kind = if i == 0 { BiquadKind::LowShelf } else { BiquadKind::HighShelf };
            f.set(b.coefs(sr, kind));
        }
    }
}

pub struct ParametricEq {
    sr: f32,
    eq: FiveBand,
    filters: [StereoBiquad; 5],
    level: f32,
    dirty: bool,
}

impl ParametricEq {
    pub fn new() -> Self {
        Self { sr: 48000.0, eq: FiveBand::new(), filters: Default::default(), level: 0.0, dirty: true }
    }
}

impl Effect for ParametricEq {
    fn prepare(&mut self, sr: f32) {
        self.sr = sr;
        self.dirty = true;
    }

    fn set(&mut self, name: &str, v: f32) {
        if name == "level" {
            self.level = v;
        } else {
            self.eq.set(name, v);
        }
        self.dirty = true;
    }

    fn reset(&mut self) {
        self.filters.iter_mut().for_each(StereoBiquad::reset);
    }

    fn process(&mut self, _: &Ctx, l: &mut [f32], r: &mut [f32]) {
        if self.dirty {
            self.eq.design(self.sr, &mut self.filters);
            self.dirty = false;
        }
        let g = db_to_gain(self.level);
        let f = &mut self.filters;
        for_each(l, r, |mut a, mut b| {
            for s in f.iter_mut() {
                (a, b) = s.process(a, b);
            }
            (a * g, b * g)
        });
    }
}

pub struct ParametricEqMs {
    sr: f32,
    mid: FiveBand,
    side: FiveBand,
    fm: [StereoBiquad; 5],
    fs: [StereoBiquad; 5],
    level: f32,
    dirty: bool,
}

impl ParametricEqMs {
    pub fn new() -> Self {
        Self {
            sr: 48000.0,
            mid: FiveBand::new(),
            side: FiveBand::new(),
            fm: Default::default(),
            fs: Default::default(),
            level: 0.0,
            dirty: true,
        }
    }
}

impl Effect for ParametricEqMs {
    fn prepare(&mut self, sr: f32) {
        self.sr = sr;
        self.dirty = true;
    }

    fn set(&mut self, name: &str, v: f32) {
        self.dirty = true;
        if name == "level" {
            self.level = v;
        } else if let Some(rest) = name.strip_prefix("m_") {
            self.mid.set(rest, v);
        } else if let Some(rest) = name.strip_prefix("s_") {
            self.side.set(rest, v);
        } else {
            // Shared on/off switches.
            self.mid.set(name, v);
            self.side.set(name, v);
        }
    }

    fn process(&mut self, _: &Ctx, l: &mut [f32], r: &mut [f32]) {
        if self.dirty {
            self.mid.design(self.sr, &mut self.fm);
            self.side.design(self.sr, &mut self.fs);
            self.dirty = false;
        }
        let g = db_to_gain(self.level);
        let (fm, fs) = (&mut self.fm, &mut self.fs);
        for_each(l, r, |a, b| {
            let (mut m, mut s) = to_ms(a, b);
            for f in fm.iter_mut() {
                m = f.l.process(m);
            }
            for f in fs.iter_mut() {
                s = f.l.process(s);
            }
            let (a, b) = from_ms(m, s);
            (a * g, b * g)
        });
    }
}

// ---------------------------------------------------------------------------
// LPF / HPF (TA_TOONTRACK_LPFHPF and _MS)

#[derive(Clone, Copy)]
struct FilterPair {
    lpf_on: bool,
    lpf_freq: f32,
    lpf_order: usize,
    hpf_on: bool,
    hpf_freq: f32,
    hpf_order: usize,
}

impl Default for FilterPair {
    fn default() -> Self {
        Self { lpf_on: true, lpf_freq: 18000.0, lpf_order: 0, hpf_on: true, hpf_freq: 50.0, hpf_order: 0 }
    }
}

#[derive(Default, Clone)]
struct FilterPairState {
    lp: [Butterworth; 2],
    hp: [Butterworth; 2],
    lp_active: bool,
    hp_active: bool,
}

impl FilterPairState {
    fn design(&mut self, sr: f32, p: &FilterPair) {
        // 12/18/24 dB per octave
        let order = |o: usize| 2 + o.min(2);
        self.lp_active = p.lpf_on && p.lpf_freq < sr * 0.45;
        self.hp_active = p.hpf_on && p.hpf_freq > 10.0;
        for f in &mut self.lp {
            f.design(sr, p.lpf_freq, order(p.lpf_order), false);
        }
        for f in &mut self.hp {
            f.design(sr, p.hpf_freq, order(p.hpf_order), true);
        }
    }

    #[inline]
    fn process(&mut self, ch: usize, mut x: f32) -> f32 {
        if self.hp_active {
            x = self.hp[ch].process(x);
        }
        if self.lp_active {
            x = self.lp[ch].process(x);
        }
        x
    }
}

pub struct LpfHpf {
    sr: f32,
    mid_side: bool,
    mid: FilterPair,
    side: FilterPair,
    sm: FilterPairState,
    ss: FilterPairState,
    dirty: bool,
}

impl LpfHpf {
    pub fn new(mid_side: bool) -> Self {
        Self {
            sr: 48000.0,
            mid_side,
            mid: FilterPair::default(),
            side: FilterPair::default(),
            sm: Default::default(),
            ss: Default::default(),
            dirty: true,
        }
    }
}

impl Effect for LpfHpf {
    fn prepare(&mut self, sr: f32) {
        self.sr = sr;
        self.dirty = true;
    }

    fn set(&mut self, name: &str, v: f32) {
        self.dirty = true;
        let (base, target_side) = if let Some(b) = name.strip_suffix("_mid") {
            (b, Some(false))
        } else if let Some(b) = name.strip_suffix("_side") {
            (b, Some(true))
        } else {
            (name, None)
        };
        for (is_side, p) in [(false, &mut self.mid), (true, &mut self.side)] {
            if target_side.is_some_and(|s| s != is_side) {
                continue;
            }
            match base {
                "lpf_onoff" => p.lpf_on = on(v),
                "lpf_freq" => p.lpf_freq = v,
                "lpf_order" => p.lpf_order = idx(v),
                "hpf_onoff" => p.hpf_on = on(v),
                "hpf_freq" => p.hpf_freq = v,
                "hpf_order" => p.hpf_order = idx(v),
                _ => {}
            }
        }
    }

    fn process(&mut self, _: &Ctx, l: &mut [f32], r: &mut [f32]) {
        if self.dirty {
            self.sm.design(self.sr, &self.mid);
            self.ss.design(self.sr, if self.mid_side { &self.side } else { &self.mid });
            self.dirty = false;
        }
        if self.mid_side {
            let (sm, ss) = (&mut self.sm, &mut self.ss);
            for_each(l, r, |a, b| {
                let (m, s) = to_ms(a, b);
                from_ms(sm.process(0, m), ss.process(0, s))
            });
        } else {
            let sm = &mut self.sm;
            for_each(l, r, |a, b| (sm.process(0, a), sm.process(1, b)));
        }
    }
}

// ---------------------------------------------------------------------------
// Extended 10-band EQ (TA_EXTENDED_EQ)

#[derive(Clone, Copy)]
struct ExtBand {
    mode: usize,
    gain: f32,
    freq: f32,
    q: f32,
    slope: usize,
}

impl Default for ExtBand {
    fn default() -> Self {
        Self { mode: 4, gain: 0.0, freq: 1000.0, q: 1.0, slope: 1 }
    }
}

#[derive(Default, Clone)]
struct ExtBandFilter {
    stages: [Biquad; 4],
    n: usize,
}

impl ExtBandFilter {
    fn design(&mut self, sr: f32, b: &ExtBand) {
        // slope: 6,12,18,24,36,48 dB/oct -> filter order 1..8
        let order = [1, 2, 3, 4, 6, 8][b.slope.min(5)];
        let q = b.q.max(0.1);
        self.n = 0;
        let mut push = |c: Coefs| {
            if self.n < 4 {
                self.stages[self.n].set(c);
                self.n += 1;
            }
        };
        match b.mode {
            0 | 1 => {
                let mut bw = Butterworth::default();
                bw.design(sr, b.freq, order, b.mode == 0);
                // Re-derive the stages directly (keeps state local).
                let pairs = order / 2;
                for k in 0..pairs {
                    let theta = std::f32::consts::PI * (2 * k + 1) as f32 / (2 * order) as f32;
                    let mut qq = 1.0 / (2.0 * theta.sin());
                    if k == 0 {
                        qq *= q / std::f32::consts::FRAC_1_SQRT_2;
                    }
                    let kind = if b.mode == 0 { BiquadKind::HighPass } else { BiquadKind::LowPass };
                    push(Coefs::rbj(kind, sr, b.freq, qq, 0.0));
                }
                if order % 2 == 1 {
                    push(if b.mode == 0 { Coefs::one_pole_hp(sr, b.freq) } else { Coefs::one_pole_lp(sr, b.freq) });
                }
            }
            2 => push(Coefs::rbj(BiquadKind::BandPass, sr, b.freq, q, 0.0)),
            3 => push(Coefs::rbj(BiquadKind::Notch, sr, b.freq, q, 0.0)),
            4 => push(Coefs::rbj(BiquadKind::Peak, sr, b.freq, q, b.gain)),
            5 => push(Coefs::rbj(BiquadKind::LowShelf, sr, b.freq, q.min(2.0), b.gain)),
            6 => push(Coefs::rbj(BiquadKind::HighShelf, sr, b.freq, q.min(2.0), b.gain)),
            7 => {
                // Tilt: opposite shelves around the frequency.
                push(Coefs::one_pole_shelf(sr, b.freq, -b.gain * 0.5, false));
                push(Coefs::one_pole_shelf(sr, b.freq, b.gain * 0.5, true));
            }
            8 => push(Coefs::one_pole_shelf(sr, b.freq, b.gain, false)),
            _ => push(Coefs::one_pole_shelf(sr, b.freq, b.gain, true)),
        }
    }

    #[inline]
    fn process(&mut self, mut x: f32) -> f32 {
        for s in &mut self.stages[..self.n] {
            x = s.process(x);
        }
        x
    }
}

pub struct ExtendedEq {
    sr: f32,
    bands: [ExtBand; 10],
    side_bands: [ExtBand; 10],
    midside: [bool; 10],
    bypass: [bool; 10],
    filters: [[ExtBandFilter; 2]; 10],
    level: f32,
    dirty: bool,
}

impl ExtendedEq {
    pub fn new() -> Self {
        Self {
            sr: 48000.0,
            bands: [ExtBand::default(); 10],
            side_bands: [ExtBand::default(); 10],
            midside: [false; 10],
            bypass: [false; 10],
            filters: Default::default(),
            level: 0.0,
            dirty: true,
        }
    }
}

impl Effect for ExtendedEq {
    fn prepare(&mut self, sr: f32) {
        self.sr = sr;
        self.dirty = true;
    }

    fn set(&mut self, name: &str, v: f32) {
        self.dirty = true;
        if name == "master-level" {
            self.level = v;
            return;
        }
        let Some(rest) = name.strip_prefix("band") else { return };
        let Some((num, field)) = rest.split_once('-') else { return };
        let (num, side) = match num.strip_suffix('s') {
            Some(n) => (n, true),
            None => (num, false),
        };
        let Ok(i) = num.parse::<usize>() else { return };
        if i == 0 || i > 10 {
            return;
        }
        let i = i - 1;
        let b = if side { &mut self.side_bands[i] } else { &mut self.bands[i] };
        match field {
            "mode" => b.mode = idx(v),
            "gain" => b.gain = v,
            "freq" => b.freq = v,
            "q" => b.q = v,
            "slope" => b.slope = idx(v),
            "midside" => self.midside[i] = on(v),
            "bypass" => self.bypass[i] = on(v),
            _ => {}
        }
    }

    fn process(&mut self, _: &Ctx, l: &mut [f32], r: &mut [f32]) {
        if self.dirty {
            for i in 0..10 {
                self.filters[i][0].design(self.sr, &self.bands[i]);
                let second = if self.midside[i] { self.side_bands[i] } else { self.bands[i] };
                self.filters[i][1].design(self.sr, &second);
            }
            self.dirty = false;
        }
        let g = db_to_gain(self.level);
        for (a, b) in l.iter_mut().zip(r.iter_mut()) {
            let (mut x, mut y) = (*a, *b);
            for i in 0..10 {
                if self.bypass[i] {
                    continue;
                }
                let f = &mut self.filters[i];
                if self.midside[i] {
                    let (m, s) = to_ms(x, y);
                    (x, y) = from_ms(f[0].process(m), f[1].process(s));
                } else {
                    x = f[0].process(x);
                    y = f[1].process(y);
                }
            }
            *a = x * g;
            *b = y * g;
        }
    }
}

// ---------------------------------------------------------------------------
// Neve 1084 style EQ (TA_EQ84)

pub struct Eq84 {
    sr: f32,
    input: f32,
    high_off: bool,
    high_gain: f32,
    high_freq: usize,
    mid_off: bool,
    mid_gain: f32,
    mid_freq: usize,
    mid_hiq: bool,
    low_off: bool,
    low_gain: f32,
    low_freq: usize,
    hpf_bypass: bool,
    hpf_freq: usize,
    lpf_bypass: bool,
    lpf_freq: usize,
    f: [StereoBiquad; 6],
    dirty: bool,
}

impl Eq84 {
    pub fn new() -> Self {
        Self {
            sr: 48000.0,
            input: 0.0,
            high_off: true,
            high_gain: 0.0,
            high_freq: 0,
            mid_off: true,
            mid_gain: 0.0,
            mid_freq: 0,
            mid_hiq: false,
            low_off: true,
            low_gain: 0.0,
            low_freq: 0,
            hpf_bypass: true,
            hpf_freq: 0,
            lpf_bypass: true,
            lpf_freq: 0,
            f: Default::default(),
            dirty: true,
        }
    }
}

impl Effect for Eq84 {
    fn prepare(&mut self, sr: f32) {
        self.sr = sr;
        self.dirty = true;
    }

    fn set(&mut self, name: &str, v: f32) {
        self.dirty = true;
        match name {
            "input" => self.input = v,
            "high-mute" => self.high_off = on(v),
            "high-gain" => self.high_gain = v,
            "high-freq" => self.high_freq = idx(v),
            "mid-mute" => self.mid_off = on(v),
            "mid-gain" => self.mid_gain = v,
            "mid-freq" => self.mid_freq = idx(v),
            "mid-q" => self.mid_hiq = on(v),
            "low-mute" => self.low_off = on(v),
            "low-gain" => self.low_gain = v,
            "low-freq" => self.low_freq = idx(v),
            "hpf-bypass" => self.hpf_bypass = on(v),
            "hpf-freq" => self.hpf_freq = idx(v),
            "lpf-bypass" => self.lpf_bypass = on(v),
            "lpf-freq" => self.lpf_freq = idx(v),
            _ => {}
        }
    }

    fn process(&mut self, _: &Ctx, l: &mut [f32], r: &mut [f32]) {
        if self.dirty {
            let sr = self.sr;
            let hf = [10000.0, 12000.0, 16000.0][self.high_freq.min(2)];
            let mf = [350.0, 700.0, 1600.0, 3200.0, 4800.0, 7260.0, 10000.0][self.mid_freq.min(6)];
            let lf = [35.0, 60.0, 110.0, 220.0][self.low_freq.min(3)];
            let hpf = [45.0, 70.0, 160.0, 360.0][self.hpf_freq.min(3)];
            let lpf = [6000.0, 8000.0, 10000.0, 14000.0, 18000.0][self.lpf_freq.min(4)];
            let id = Coefs::IDENTITY;
            self.f[0].set(if self.high_off { id } else { Coefs::rbj(BiquadKind::HighShelf, sr, hf, 0.6, self.high_gain) });
            let q = if self.mid_hiq { 2.0 } else { 0.7 };
            self.f[1].set(if self.mid_off { id } else { peak(sr, mf, q, self.mid_gain) });
            self.f[2].set(if self.low_off { id } else { Coefs::rbj(BiquadKind::LowShelf, sr, lf, 0.6, self.low_gain) });
            self.f[3].set(if self.hpf_bypass { id } else { highpass(sr, hpf) });
            self.f[4].set(if self.hpf_bypass { id } else { Coefs::one_pole_hp(sr, hpf) });
            self.f[5].set(if self.lpf_bypass { id } else { lowpass(sr, lpf) });
            self.dirty = false;
        }
        let g = db_to_gain(self.input);
        let f = &mut self.f;
        for_each(l, r, |a, b| {
            // Transformer-coupled class A stage: gentle saturation.
            let (mut a, mut b) = (shape::tanh(a * g * 0.9) / 0.9, shape::tanh(b * g * 0.9) / 0.9);
            for s in f.iter_mut() {
                (a, b) = s.process(a, b);
            }
            (a, b)
        });
    }
}

// ---------------------------------------------------------------------------
// Pultec EQP-1A style program EQ (TA_EQP1A)

pub struct Pultec {
    sr: f32,
    in_lev: f32,
    out_lev: f32,
    low_boost: f32,
    low_atten: f32,
    low_cps: usize,
    bandwidth: f32,
    high_boost: f32,
    high_atten: f32,
    high_kcs: usize,
    atten_sel: usize,
    f: [StereoBiquad; 4],
    dirty: bool,
}

impl Pultec {
    pub fn new() -> Self {
        Self {
            sr: 48000.0,
            in_lev: 0.0,
            out_lev: 0.0,
            low_boost: 0.0,
            low_atten: 0.0,
            low_cps: 1,
            bandwidth: 5.0,
            high_boost: 0.0,
            high_atten: 0.0,
            high_kcs: 4,
            atten_sel: 1,
            f: Default::default(),
            dirty: true,
        }
    }
}

impl Effect for Pultec {
    fn prepare(&mut self, sr: f32) {
        self.sr = sr;
        self.dirty = true;
    }

    fn set(&mut self, name: &str, v: f32) {
        self.dirty = true;
        match name {
            "in-lev" => self.in_lev = v,
            "out-lev" => self.out_lev = v,
            "low-boost" => self.low_boost = v,
            "low-atten" => self.low_atten = v,
            "low-cps" => self.low_cps = idx(v),
            "bandwith" => self.bandwidth = v,
            "high-boost" => self.high_boost = v,
            "high-atten" => self.high_atten = v,
            "high-kcs" => self.high_kcs = idx(v),
            "atten-sel" => self.atten_sel = idx(v),
            _ => {}
        }
    }

    fn process(&mut self, _: &Ctx, l: &mut [f32], r: &mut [f32]) {
        if self.dirty {
            let sr = self.sr;
            let lf = [20.0, 30.0, 60.0, 100.0][self.low_cps.min(3)];
            let hf = [3000.0, 4000.0, 5000.0, 8000.0, 10000.0, 12000.0, 16000.0][self.high_kcs.min(6)];
            let af = [5000.0, 10000.0, 20000.0][self.atten_sel.min(2)];
            // Boost and cut are separate networks; the famous low "trick" comes
            // from the cut sitting slightly above the boost frequency.
            self.f[0].set(Coefs::rbj(BiquadKind::LowShelf, sr, lf * 1.3, 0.5, self.low_boost * 1.35));
            self.f[1].set(Coefs::rbj(BiquadKind::LowShelf, sr, lf * 2.2, 0.6, -self.low_atten * 1.75));
            let q = lerp(2.5, 0.35, self.bandwidth / 10.0);
            self.f[2].set(peak(sr, hf, q, self.high_boost * 1.8));
            self.f[3].set(Coefs::rbj(BiquadKind::HighShelf, sr, af, 0.7, -self.high_atten * 1.6));
            self.dirty = false;
        }
        let gi = db_to_gain(self.in_lev);
        let go = db_to_gain(self.out_lev);
        let f = &mut self.f;
        for_each(l, r, |a, b| {
            let (mut a, mut b) = (a * gi, b * gi);
            for s in f.iter_mut() {
                (a, b) = s.process(a, b);
            }
            // Tube make-up stage.
            (shape::tube(a * 0.7, 0.1) / 0.7 * go, shape::tube(b * 0.7, 0.1) / 0.7 * go)
        });
    }
}

// ---------------------------------------------------------------------------
// API 550A style EQ (TA_EQ550A)

pub struct Api550 {
    sr: f32,
    gain: f32,
    out: f32,
    bands: [(bool, f32, f32); 4],
    low_shelf: bool,
    high_shelf: bool,
    bandpass: bool,
    f: [StereoBiquad; 6],
    dirty: bool,
}

impl Api550 {
    pub fn new() -> Self {
        Self {
            sr: 48000.0,
            gain: 0.0,
            out: 0.0,
            bands: [(true, 150.0, 0.0), (true, 1000.0, 0.0), (true, 2000.0, 0.0), (true, 10000.0, 0.0)],
            low_shelf: true,
            high_shelf: true,
            bandpass: false,
            f: Default::default(),
            dirty: true,
        }
    }

    /// API "proportional Q": wider at small gains.
    fn pq(gain: f32) -> f32 {
        0.4 + gain.abs() / 12.0 * 1.0
    }
}

impl Effect for Api550 {
    fn prepare(&mut self, sr: f32) {
        self.sr = sr;
        self.dirty = true;
    }

    fn set(&mut self, name: &str, v: f32) {
        self.dirty = true;
        let band = |n: &str| match n {
            "low" => Some(0),
            "mid1" => Some(1),
            "mid2" => Some(2),
            "high" => Some(3),
            _ => None,
        };
        match name {
            "gain" => self.gain = v,
            "out-level" => self.out = v,
            "low-mode" => self.low_shelf = on(v),
            "high-mode" => self.high_shelf = on(v),
            "bandpass-power" => self.bandpass = on(v),
            _ => {
                if let Some((b, field)) = name.split_once('-') {
                    if let Some(i) = band(b) {
                        match field {
                            "power" => self.bands[i].0 = on(v),
                            "freq" => self.bands[i].1 = v,
                            "gain" => self.bands[i].2 = v,
                            _ => {}
                        }
                    }
                }
            }
        }
    }

    fn process(&mut self, _: &Ctx, l: &mut [f32], r: &mut [f32]) {
        if self.dirty {
            let sr = self.sr;
            for (i, &(p, f, g)) in self.bands.iter().enumerate() {
                let c = if !p || g.abs() < 1e-3 {
                    Coefs::IDENTITY
                } else if i == 0 && self.low_shelf {
                    Coefs::rbj(BiquadKind::LowShelf, sr, f, 0.6, g)
                } else if i == 3 && self.high_shelf {
                    Coefs::rbj(BiquadKind::HighShelf, sr, f, 0.6, g)
                } else {
                    peak(sr, f, Self::pq(g), g)
                };
                self.f[i].set(c);
            }
            self.f[4].set(if self.bandpass { highpass(sr, 50.0) } else { Coefs::IDENTITY });
            self.f[5].set(if self.bandpass { lowpass(sr, 15000.0) } else { Coefs::IDENTITY });
            self.dirty = false;
        }
        let gi = db_to_gain(self.gain);
        let go = db_to_gain(self.out);
        let f = &mut self.f;
        for_each(l, r, |a, b| {
            let (mut a, mut b) = (a * gi, b * gi);
            for s in f.iter_mut() {
                (a, b) = s.process(a, b);
            }
            // 2520 op-amp: clean until it isn't.
            (shape::tanh(a * 0.5) * 2.0 * go, shape::tanh(b * 0.5) * 2.0 * go)
        });
    }
}

// ---------------------------------------------------------------------------
// Creative filter (TA_TTFILTER)

pub struct CreativeFilter {
    sr: f32,
    freq: f32,
    res: f32,
    poles: usize,
    mode: usize,
    source: usize,
    amount: f32,
    polarity: bool,
    level: f32,
    lfo_speed: f32,
    lfo_wave: usize,
    lfo_sync: usize,
    env_release: f32,
    sequence: usize,
    seq_speed: f32,
    seq_sync: usize,
    svf: [[Svf; 2]; 2],
    lfo: Lfo,
    env: Envelope,
    seq_phase: f32,
}

const SEQUENCES: [[f32; 8]; 10] = [
    [1.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.75, 0.25],
    [0.0, 0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25],
    [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 1.0, 0.5, 1.0, 0.5, 1.0],
    [1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.2, 0.4],
    [0.2, 0.9, 0.4, 1.0, 0.1, 0.7, 0.3, 0.8],
    [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
    [0.5, 1.0, 0.5, 0.0, 0.5, 1.0, 0.5, 0.0],
    [0.0, 0.0, 1.0, 1.0, 0.5, 0.5, 1.0, 0.0],
    [1.0, 0.5, 0.75, 0.25, 1.0, 0.5, 0.0, 0.25],
];

impl CreativeFilter {
    pub fn new() -> Self {
        Self {
            sr: 48000.0,
            freq: 1000.0,
            res: 1.0,
            poles: 1,
            mode: 0,
            source: 0,
            amount: 0.0,
            polarity: false,
            level: 0.0,
            lfo_speed: 1.0,
            lfo_wave: 0,
            lfo_sync: 3,
            env_release: 1.0,
            sequence: 3,
            seq_speed: 1.0,
            seq_sync: 3,
            svf: Default::default(),
            lfo: Lfo::default(),
            env: Envelope::default(),
            seq_phase: 0.0,
        }
    }
}

impl Effect for CreativeFilter {
    fn prepare(&mut self, sr: f32) {
        self.sr = sr;
        self.env.set(sr, 0.005, self.env_release);
    }

    fn set(&mut self, name: &str, v: f32) {
        match name {
            "center_freq" => self.freq = v,
            "resonance" => self.res = v,
            "poles" => self.poles = idx(v),
            "mode" => self.mode = idx(v),
            "modulation_source" => self.source = idx(v),
            "modulation_mnt" => self.amount = v,
            "polarity" => self.polarity = on(v),
            "level" => self.level = v,
            "lfo_speed" => self.lfo_speed = v,
            "lfo_waveform" => self.lfo_wave = idx(v),
            "lfo_sync_soeed" => self.lfo_sync = idx(v),
            "env_release" => {
                self.env_release = v;
                self.env.set(self.sr, 0.005, v);
            }
            "sequence" => self.sequence = idx(v),
            "sequence_speed" => self.seq_speed = v,
            "sequence_sync_speed" => self.seq_sync = idx(v),
            _ => {}
        }
    }

    fn process(&mut self, ctx: &Ctx, l: &mut [f32], r: &mut [f32]) {
        let sr = self.sr;
        let wave = [Wave::Triangle, Wave::Square, Wave::Saw, Wave::Sine, Wave::Random][self.lfo_wave.min(4)];
        let rate_of = |i: usize| 1.0 / note_seconds(crate::effect::RATE_NOTES_LONG[i.min(18)], ctx.bpm);
        self.lfo.freq = match self.source {
            4 => rate_of(self.lfo_sync),
            _ => self.lfo_speed,
        };
        let seq_rate = match self.source {
            5 => rate_of(self.seq_sync),
            _ => 1.0 / self.seq_speed.max(0.01),
        };
        let depth_oct = self.amount / 100.0 * 4.0 * if self.polarity { -1.0 } else { 1.0 };
        let q = (0.5 + self.res.max(0.0) * 0.7).min(20.0);
        let stages = if self.poles >= 2 { 2 } else { 1 };
        let gain = db_to_gain(self.level);
        let seq = SEQUENCES[self.sequence.min(9)];
        for (a, b) in l.iter_mut().zip(r.iter_mut()) {
            let m = match self.source {
                1 | 4 => {
                    self.lfo.advance(sr);
                    self.lfo.value(wave, 0.0) * 0.5 + 0.5
                }
                2 => (self.env.process((*a).abs().max((*b).abs())) * 4.0).min(1.0),
                3 | 5 => {
                    self.seq_phase = (self.seq_phase + seq_rate * 8.0 / sr).fract();
                    seq[(self.seq_phase * 8.0) as usize % 8]
                }
                _ => 0.0,
            };
            let f = (self.freq * 2f32.powf(m * depth_oct)).clamp(20.0, sr * 0.45);
            for ch in 0..2 {
                for s in 0..stages {
                    self.svf[ch][s].set(sr, f, if s == 0 { q } else { std::f32::consts::FRAC_1_SQRT_2 });
                }
            }
            let mut run = |ch: usize, x: f32| {
                let mut y = x;
                for s in 0..stages {
                    let o = self.svf[ch][s].process(y);
                    y = match self.mode {
                        0 => o.low,
                        1 => o.high,
                        2 => o.band,
                        _ => o.low + o.high,
                    };
                }
                y
            };
            *a = run(0, *a) * gain;
            *b = run(1, *b) * gain;
        }
    }
}

// ---------------------------------------------------------------------------
// SRS WOW style enhancer (TA_SRSWOW)

pub struct WowEnhancer {
    sr: f32,
    small: bool,
    bass: f32,
    wow: f32,
    level: f32,
    lp: [Biquad; 2],
    harm: [Biquad; 2],
    side_hp: Biquad,
}

impl WowEnhancer {
    pub fn new() -> Self {
        Self {
            sr: 48000.0,
            small: false,
            bass: 0.0,
            wow: 0.0,
            level: 0.0,
            lp: Default::default(),
            harm: Default::default(),
            side_hp: Biquad::identity(),
        }
    }
}

impl Effect for WowEnhancer {
    fn prepare(&mut self, sr: f32) {
        self.sr = sr;
        self.side_hp.set(highpass(sr, 300.0));
        self.set("bass_mode", if self.small { 1.0 } else { 0.0 });
    }

    fn set(&mut self, name: &str, v: f32) {
        match name {
            "bass_mode" => {
                self.small = on(v);
                let f = if self.small { 150.0 } else { 90.0 };
                for b in &mut self.lp {
                    b.set(lowpass(self.sr, f));
                }
                for b in &mut self.harm {
                    b.set(Coefs::rbj(BiquadKind::BandPass, self.sr, f * 2.0, 1.0, 0.0));
                }
            }
            "tru_bass" => self.bass = v / 100.0,
            "wow" => self.wow = v / 100.0,
            "level" => self.level = v,
            _ => {}
        }
    }

    fn process(&mut self, _: &Ctx, l: &mut [f32], r: &mut [f32]) {
        let g = db_to_gain(self.level);
        for (a, b) in l.iter_mut().zip(r.iter_mut()) {
            let (m, s) = to_ms(*a, *b);
            // Psychoacoustic bass: generate harmonics of the low band.
            let low = self.lp[0].process(m);
            let h = self.harm[0].process(shape::tanh(low * 4.0) - low);
            let m2 = m + self.bass * (low * 0.6 + h * 1.5);
            let s2 = s + self.wow * 1.2 * self.side_hp.process(s);
            let (x, y) = from_ms(m2, s2);
            *a = x * g;
            *b = y * g;
        }
    }
}
