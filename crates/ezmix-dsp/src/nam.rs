//! Neural Amp Modeler (WaveNet) inference and the amp-capture slot.
//!
//! Weight layout and processing order follow the open-source NAM core
//! (`wavenet.cpp`): per layer array a 1x1 "rechannel", then dilated causal
//! convolutions with a condition mix-in, tanh, a 1x1 residual projection and
//! a summed head, then a head projection; finally `head_scale`.

use serde::Deserialize;

use crate::dsp::*;
use ezmix_content::preset::NamState;

#[derive(Deserialize)]
struct NamFile {
    architecture: String,
    config: NamConfig,
    weights: Vec<f32>,
    #[serde(default)]
    sample_rate: Option<f32>,
    #[serde(default)]
    metadata: Option<NamMetadata>,
}

#[derive(Deserialize)]
struct NamMetadata {
    #[serde(default)]
    loudness: Option<f32>,
}

#[derive(Deserialize)]
struct NamConfig {
    layers: Vec<LayerConfig>,
    #[serde(default)]
    head_scale: Option<f32>,
}

#[derive(Deserialize, Clone)]
struct LayerConfig {
    input_size: usize,
    condition_size: usize,
    head_size: usize,
    channels: usize,
    kernel_size: usize,
    dilations: Vec<usize>,
    activation: String,
    gated: bool,
    head_bias: bool,
}

struct Layer {
    dilation: usize,
    /// [k][out][in]
    conv_w: Vec<f32>,
    conv_b: Vec<f32>,
    /// [out] (condition size 1)
    mixin_w: Vec<f32>,
    /// [out][in]
    w1x1: Vec<f32>,
    b1x1: Vec<f32>,
}

struct LayerArray {
    in_size: usize,
    ch: usize,
    head_size: usize,
    k: usize,
    rechannel: Vec<f32>,
    layers: Vec<Layer>,
    head_w: Vec<f32>,
    head_b: Option<Vec<f32>>,
    /// One history buffer per layer input: [ch][buf_len]
    bufs: Vec<Vec<f32>>,
    /// Output of the last layer for the current block: [ch][max_block]
    out: Vec<f32>,
    /// Head accumulator: [ch][max_block]
    head_acc: Vec<f32>,
    /// Head output: [head_size][max_block]
    head_out: Vec<f32>,
    z: Vec<f32>,
}

pub struct WaveNet {
    arrays: Vec<LayerArray>,
    head_scale: f32,
    buf_len: usize,
    history: usize,
    start: usize,
    pub sample_rate: f32,
    pub loudness: Option<f32>,
    cond: Vec<f32>,
}

pub const MAX_BLOCK: usize = 1024;

/// Rational tanh approximation used by NAM's "fast tanh" activation.
#[inline]
fn fast_tanh(x: f32) -> f32 {
    let ax = x.abs();
    let x2 = x * x;
    (x * (2.455_507_5 + 2.455_507_5 * ax + (0.893_229_85 + 0.821_226_7 * ax) * x2))
        / (2.445_066_3 + (2.445_066_3 + x2) * (x + 0.814_642_7 * x * ax).abs())
}

struct Reader<'a> {
    w: &'a [f32],
    pos: usize,
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Result<Vec<f32>, String> {
        if self.pos + n > self.w.len() {
            return Err("weights too short".into());
        }
        let v = self.w[self.pos..self.pos + n].to_vec();
        self.pos += n;
        Ok(v)
    }
}

impl WaveNet {
    pub fn from_json(bytes: &[u8]) -> Result<Self, String> {
        let file: NamFile = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
        if file.architecture != "WaveNet" {
            return Err(format!("unsupported NAM architecture {}", file.architecture));
        }
        let mut r = Reader { w: &file.weights, pos: 0 };
        let mut arrays = Vec::new();
        let mut history = 0;
        for lc in &file.config.layers {
            if lc.gated || lc.condition_size != 1 || !lc.activation.eq_ignore_ascii_case("tanh") {
                return Err("unsupported WaveNet layer configuration".into());
            }
            let (ch, k) = (lc.channels, lc.kernel_size);
            let rechannel = r.take(ch * lc.input_size)?;
            let mut layers = Vec::new();
            for &d in &lc.dilations {
                // Conv1D weights are stored [out][in][k]; transpose to [k][out][in].
                let raw = r.take(ch * ch * k)?;
                let mut conv_w = vec![0.0; ch * ch * k];
                for o in 0..ch {
                    for i in 0..ch {
                        for kk in 0..k {
                            conv_w[(kk * ch + o) * ch + i] = raw[(o * ch + i) * k + kk];
                        }
                    }
                }
                let conv_b = r.take(ch)?;
                let mixin_w = r.take(ch)?;
                let w1x1 = r.take(ch * ch)?;
                let b1x1 = r.take(ch)?;
                history = history.max(d * (k - 1));
                layers.push(Layer { dilation: d, conv_w, conv_b, mixin_w, w1x1, b1x1 });
            }
            let head_w = r.take(lc.head_size * ch)?;
            let head_b = if lc.head_bias { Some(r.take(lc.head_size)?) } else { None };
            arrays.push((lc.clone(), rechannel, layers, head_w, head_b));
        }
        // The trailing weight is the head scale; the config value is only a
        // construction default.
        let head_scale = match r.take(1) {
            Ok(v) => v[0],
            Err(_) => file.config.head_scale.unwrap_or(1.0),
        };
        let buf_len = history + MAX_BLOCK * 8;
        let arrays = arrays
            .into_iter()
            .map(|(lc, rechannel, layers, head_w, head_b)| {
                let ch = lc.channels;
                let n_layers = layers.len();
                LayerArray {
                    in_size: lc.input_size,
                    ch,
                    head_size: lc.head_size,
                    k: lc.kernel_size,
                    rechannel,
                    layers,
                    head_w,
                    head_b,
                    bufs: vec![vec![0.0; ch * buf_len]; n_layers],
                    out: vec![0.0; ch * MAX_BLOCK],
                    head_acc: vec![0.0; ch * MAX_BLOCK],
                    head_out: vec![0.0; lc.head_size * MAX_BLOCK],
                    z: vec![0.0; ch * MAX_BLOCK],
                }
            })
            .collect::<Vec<_>>();
        // Consistency: each array's head feeds the next array's head sum.
        for w in arrays.windows(2) {
            if w[0].head_size != w[1].ch || w[1].in_size != w[0].ch {
                return Err("inconsistent layer array sizes".into());
            }
        }
        Ok(Self {
            arrays,
            head_scale,
            buf_len,
            history,
            start: history,
            sample_rate: file.sample_rate.unwrap_or(48000.0),
            loudness: file.metadata.and_then(|m| m.loudness),
            cond: vec![0.0; MAX_BLOCK],
        })
    }

    pub fn reset(&mut self) {
        for a in &mut self.arrays {
            for b in &mut a.bufs {
                b.iter_mut().for_each(|v| *v = 0.0);
            }
        }
        self.start = self.history;
    }

    /// Process up to MAX_BLOCK samples in place.
    pub fn process(&mut self, io: &mut [f32]) {
        for chunk in io.chunks_mut(MAX_BLOCK) {
            self.process_block(chunk);
        }
    }

    fn rewind_if_needed(&mut self, n: usize) {
        if self.start + n <= self.buf_len {
            return;
        }
        let (h, bl, st) = (self.history, self.buf_len, self.start);
        for a in &mut self.arrays {
            for b in &mut a.bufs {
                for c in 0..a.ch {
                    let row = &mut b[c * bl..(c + 1) * bl];
                    row.copy_within(st - h..st, 0);
                }
            }
        }
        self.start = h;
    }

    fn process_block(&mut self, io: &mut [f32]) {
        let n = io.len();
        self.rewind_if_needed(n);
        let (bl, st) = (self.buf_len, self.start);
        self.cond[..n].copy_from_slice(io);
        let cond = &self.cond[..n];

        let n_arrays = self.arrays.len();
        for ai in 0..n_arrays {
            let (prev, rest) = self.arrays.split_at_mut(ai);
            let a = &mut rest[0];
            let (ch, k) = (a.ch, a.k);
            // Rechannel into the first layer buffer.
            {
                let buf0 = &mut a.bufs[0];
                for o in 0..ch {
                    let dst = &mut buf0[o * bl + st..o * bl + st + n];
                    dst.iter_mut().for_each(|v| *v = 0.0);
                    for i in 0..a.in_size {
                        let w = a.rechannel[o * a.in_size + i];
                        let src: &[f32] = if ai == 0 { cond } else { &prev[ai - 1].out[i * MAX_BLOCK..i * MAX_BLOCK + n] };
                        for (d, s) in dst.iter_mut().zip(src) {
                            *d += w * s;
                        }
                    }
                }
            }
            // Head accumulator starts from the previous array's head output.
            for c in 0..ch {
                let acc = &mut a.head_acc[c * MAX_BLOCK..c * MAX_BLOCK + n];
                if ai == 0 {
                    acc.iter_mut().for_each(|v| *v = 0.0);
                } else {
                    acc.copy_from_slice(&prev[ai - 1].head_out[c * MAX_BLOCK..c * MAX_BLOCK + n]);
                }
            }
            let n_layers = a.layers.len();
            for li in 0..n_layers {
                let layer = &a.layers[li];
                let input = &a.bufs[li];
                // z = conv(input) + bias + mixin * cond
                for o in 0..ch {
                    let z = &mut a.z[o * MAX_BLOCK..o * MAX_BLOCK + n];
                    let b = layer.conv_b[o];
                    let m = layer.mixin_w[o];
                    for (zv, c) in z.iter_mut().zip(cond) {
                        *zv = b + m * c;
                    }
                    for kk in 0..k {
                        let off = layer.dilation * (k - 1 - kk);
                        for i in 0..ch {
                            let w = layer.conv_w[(kk * ch + o) * ch + i];
                            let src = &input[i * bl + st - off..i * bl + st - off + n];
                            for (zv, s) in z.iter_mut().zip(src) {
                                *zv += w * s;
                            }
                        }
                    }
                    for zv in z.iter_mut() {
                        *zv = fast_tanh(*zv);
                    }
                    let acc = &mut a.head_acc[o * MAX_BLOCK..o * MAX_BLOCK + n];
                    for (h, zv) in acc.iter_mut().zip(z.iter()) {
                        *h += zv;
                    }
                }
                // Residual: next = input + 1x1(z) + b
                let last = li + 1 == n_layers;
                for o in 0..ch {
                    let mut tmp = [0.0f32; MAX_BLOCK];
                    let t = &mut tmp[..n];
                    let b = layer.b1x1[o];
                    let res = &a.bufs[li][o * bl + st..o * bl + st + n];
                    for (tv, r) in t.iter_mut().zip(res) {
                        *tv = r + b;
                    }
                    for i in 0..ch {
                        let w = layer.w1x1[o * ch + i];
                        let z = &a.z[i * MAX_BLOCK..i * MAX_BLOCK + n];
                        for (tv, zv) in t.iter_mut().zip(z) {
                            *tv += w * zv;
                        }
                    }
                    if last {
                        a.out[o * MAX_BLOCK..o * MAX_BLOCK + n].copy_from_slice(t);
                    } else {
                        a.bufs[li + 1][o * bl + st..o * bl + st + n].copy_from_slice(t);
                    }
                }
            }
            // Head projection.
            for h in 0..a.head_size {
                let dst = &mut a.head_out[h * MAX_BLOCK..h * MAX_BLOCK + n];
                let b = a.head_b.as_ref().map(|b| b[h]).unwrap_or(0.0);
                dst.iter_mut().for_each(|v| *v = b);
                for c in 0..ch {
                    let w = a.head_w[h * ch + c];
                    let src = &a.head_acc[c * MAX_BLOCK..c * MAX_BLOCK + n];
                    for (d, s) in dst.iter_mut().zip(src) {
                        *d += w * s;
                    }
                }
            }
        }
        let last = &self.arrays[n_arrays - 1];
        for (o, h) in io.iter_mut().zip(&last.head_out[..n]) {
            *o = h * self.head_scale;
        }
        self.start += n;
    }
}

/// Streaming windowed-sinc resampler for a fixed rate ratio.
pub struct Resampler {
    step: f64,
    hist: Vec<f32>,
    written: u64,
    pos: f64,
    cutoff: f32,
}

const HALF: usize = 12;

impl Resampler {
    pub fn new(from: f32, to: f32) -> Self {
        Self {
            step: from as f64 / to as f64,
            hist: vec![0.0; 1 << 14],
            written: 0,
            pos: HALF as f64,
            cutoff: (to / from).min(1.0) * 0.92,
        }
    }

    pub fn push(&mut self, x: f32) {
        let mask = self.hist.len() - 1;
        self.hist[(self.written as usize) & mask] = x;
        self.written += 1;
    }

    /// Emit every output sample whose window is fully available.
    pub fn pull(&mut self, mut emit: impl FnMut(f32)) {
        let mask = self.hist.len() - 1;
        while self.pos + (HALF as f64) < self.written as f64 {
            let base = self.pos.floor();
            let frac = (self.pos - base) as f32;
            let base = base as i64;
            let mut acc = 0.0;
            let mut norm = 0.0;
            for j in -(HALF as i64) + 1..=HALF as i64 {
                let x = (j as f32 - frac) * self.cutoff;
                let sinc = if x.abs() < 1e-6 { 1.0 } else { (std::f32::consts::PI * x).sin() / (std::f32::consts::PI * x) };
                let w = 0.5 + 0.5 * (std::f32::consts::PI * (j as f32 - frac) / HALF as f32).cos();
                let c = sinc * w;
                let idx = base + j;
                if idx >= 0 {
                    acc += self.hist[(idx as usize) & mask] * c;
                }
                norm += c;
            }
            emit(acc / norm.max(1e-6));
            self.pos += self.step;
        }
    }
}

/// The amp-capture slot: input/output gain, noise gate, model, tone stack.
pub struct NamSlot {
    sr: f32,
    model: Option<WaveNet>,
    state: NamState,
    norm_gain: f32,
    up: Option<Resampler>,
    down: Option<Resampler>,
    model_buf: Vec<f32>,
    fifo: std::collections::VecDeque<f32>,
    mono: Vec<f32>,
    tone: [Biquad; 3],
    hp: Biquad,
    gate_env: f32,
    gate_gain: f32,
    doubler: DelayLine,
}

impl NamSlot {
    pub fn new(model_json: Option<&[u8]>, state: NamState, sr: f32) -> Self {
        let model = model_json.and_then(|j| WaveNet::from_json(j).ok());
        // Normalize captures to a common loudness like the NAM plugin does.
        let norm_gain = model.as_ref().and_then(|m| m.loudness).map(|l| db_to_gain(-18.0 - l)).unwrap_or(1.0);
        let mut s = Self {
            sr,
            model,
            state,
            norm_gain,
            up: None,
            down: None,
            model_buf: Vec::with_capacity(8192),
            fifo: std::collections::VecDeque::with_capacity(16384),
            mono: vec![0.0; 8192],
            tone: Default::default(),
            hp: Biquad::identity(),
            gate_env: 0.0,
            gate_gain: 1.0,
            doubler: DelayLine::new((0.05 * sr) as usize + 8),
        };
        s.configure();
        s
    }

    pub fn has_model(&self) -> bool {
        self.model.is_some()
    }

    fn configure(&mut self) {
        let sr = self.sr;
        if let Some(m) = &self.model {
            if (m.sample_rate - sr).abs() > 1.0 {
                self.up = Some(Resampler::new(sr, m.sample_rate));
                self.down = Some(Resampler::new(m.sample_rate, sr));
                // Prime with enough silence to cover both resampler windows.
                self.fifo.clear();
                self.fifo.extend(std::iter::repeat_n(0.0, 4 * HALF + 8));
            }
        }
        self.update_tone();
        self.hp.set(Coefs::one_pole_hp(sr, 8.0));
    }

    fn update_tone(&mut self) {
        let sr = self.sr;
        let s = &self.state;
        if !s.eq_active {
            self.tone.iter_mut().for_each(|b| b.set(Coefs::IDENTITY));
            return;
        }
        // Basic NAM tone stack, knobs 0..10 with 5 = flat.
        let (b, m, t) = (s.eq_bass * 10.0, s.eq_mid * 10.0, s.eq_treble * 10.0);
        self.tone[0].set(Coefs::rbj(BiquadKind::LowShelf, sr, 150.0, 0.707, 4.0 * (b - 5.0)));
        let mid_db = 3.0 * (m - 5.0);
        self.tone[1].set(peak(sr, 425.0, if mid_db < 0.0 { 1.5 } else { 0.7 }, mid_db));
        self.tone[2].set(Coefs::rbj(BiquadKind::HighShelf, sr, 1800.0, 0.707, 2.0 * (t - 5.0)));
    }

    pub fn set_index(&mut self, index: usize, v: f32) {
        self.state.set_index(index, v);
        if (4..=7).contains(&index) {
            self.update_tone();
        }
    }

    pub fn reset(&mut self) {
        if let Some(m) = &mut self.model {
            m.reset();
        }
    }

    pub fn process(&mut self, l: &mut [f32], r: &mut [f32]) {
        let Some(model) = &mut self.model else { return };
        let n = l.len();
        let s = &self.state;
        // Ranges recovered from the original wrapper: input -20..+20 dB,
        // output -40..+40 dB, gate threshold -100..0 dB.
        let in_gain = db_to_gain((s.input_level - 0.5) * 40.0);
        let out_gain = db_to_gain((s.output_level - 0.5) * 80.0) * self.norm_gain;
        let gate_thr = db_to_gain(-100.0 + 100.0 * s.gate_threshold);
        let gate_on = s.gate_active;
        let (att, rel) = (time_coef(0.001, self.sr), time_coef(0.05, self.sr));
        let env_c = time_coef(0.01, self.sr);

        // Mono input with gate detection.
        for i in 0..n {
            let x = (l[i] + r[i]) * 0.5 * in_gain;
            self.gate_env = x.abs() + (self.gate_env - x.abs()) * env_c;
            let target = if !gate_on || self.gate_env >= gate_thr {
                1.0
            } else {
                // 1:10 downward expansion below the threshold.
                (self.gate_env / gate_thr).powf(9.0).clamp(0.0, 1.0)
            };
            let c = if target > self.gate_gain { att } else { rel };
            self.gate_gain = target + (self.gate_gain - target) * c;
            self.mono[i] = x;
            l[i] = self.gate_gain;
        }

        match (&mut self.up, &mut self.down) {
            (Some(up), Some(down)) => {
                self.model_buf.clear();
                for &x in &self.mono[..n] {
                    up.push(x);
                }
                let buf = &mut self.model_buf;
                up.pull(|v| buf.push(v));
                model.process(&mut self.model_buf);
                for &y in &self.model_buf {
                    down.push(y);
                }
                let fifo = &mut self.fifo;
                down.pull(|v| fifo.push_back(v));
                for i in 0..n {
                    self.mono[i] = self.fifo.pop_front().unwrap_or(0.0);
                }
            }
            _ => model.process(&mut self.mono[..n]),
        }

        let dbl = (s.stereo_phase * 0.02 * self.sr).max(0.0);
        for i in 0..n {
            let mut y = self.hp.process(self.mono[i]);
            for t in &mut self.tone {
                y = t.process(y);
            }
            // l[i] carries the gate gain computed above.
            y *= out_gain * l[i];
            self.doubler.push(y);
            l[i] = y;
            r[i] = if dbl > 1.0 { self.doubler.read(dbl) } else { y };
        }
    }
}
