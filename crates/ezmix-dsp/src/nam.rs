//! Neural Amp Modeler (WaveNet) inference and the amp-capture slot.
//!
//! Weight layout and processing order follow the open-source NAM core
//! (`wavenet/model.cpp`): per layer array a 1x1 "rechannel", then per layer
//! a dilated causal convolution plus a condition mix-in, the activation
//! (optionally gated), a 1x1 residual projection and a skip output summed
//! into the head; the head sum goes through a causal convolution, then
//! `head_scale`. This covers the original "A1" captures and TONE3000's "A2"
//! ones (a `SlimmableContainer` of WaveNets with per-layer kernel sizes,
//! LeakyReLU and a 16-tap head), which run at their largest size.

use serde_json::Value;

use crate::dsp::*;
use ezmix_content::preset::NamState;

pub const MAX_BLOCK: usize = 1024;

/// Rational tanh approximation used by NAM's "fast tanh" activation.
#[inline]
fn fast_tanh(x: f32) -> f32 {
    let ax = x.abs();
    let x2 = x * x;
    (x * (2.455_507_5 + 2.455_507_5 * ax + (0.893_229_85 + 0.821_226_7 * ax) * x2))
        / (2.445_066_3 + (2.445_066_3 + x2) * (x + 0.814_642_7 * x * ax).abs())
}

#[derive(Clone, Debug, PartialEq)]
enum Activation {
    Tanh,
    Hardtanh,
    Relu,
    LeakyRelu(f32),
    /// Per-channel negative slopes.
    PRelu(Vec<f32>),
    Sigmoid,
    Silu,
    Hardswish,
    LeakyHardtanh { min_val: f32, max_val: f32, min_slope: f32, max_slope: f32 },
    Softsign,
}

impl Activation {
    fn parse(v: &Value) -> Result<Self, String> {
        let (name, obj) = match v {
            Value::String(s) => (s.as_str(), None),
            Value::Object(_) => (v.get("type").and_then(Value::as_str).ok_or("activation without a type")?, Some(v)),
            _ => return Err("invalid activation".into()),
        };
        let num = |key: &str, default: f32| obj.and_then(|o| o.get(key)).and_then(Value::as_f64).map_or(default, |x| x as f32);
        Ok(match name {
            "Tanh" | "Fasttanh" => Self::Tanh,
            "Hardtanh" => Self::Hardtanh,
            "ReLU" => Self::Relu,
            "LeakyReLU" => Self::LeakyRelu(num("negative_slope", 0.01)),
            "PReLU" => match obj.and_then(|o| o.get("negative_slopes")).and_then(Value::as_array) {
                Some(s) => Self::PRelu(s.iter().map(|x| x.as_f64().unwrap_or(0.01) as f32).collect()),
                None => Self::LeakyRelu(num("negative_slope", 0.01)),
            },
            "Sigmoid" => Self::Sigmoid,
            "SiLU" => Self::Silu,
            "Hardswish" => Self::Hardswish,
            "LeakyHardtanh" | "LeakyHardTanh" => Self::LeakyHardtanh {
                min_val: num("min_val", -1.0),
                max_val: num("max_val", 1.0),
                min_slope: num("min_slope", 0.01),
                max_slope: num("max_slope", 0.01),
            },
            "Softsign" => Self::Softsign,
            other => return Err(format!("unsupported activation {other}")),
        })
    }

    /// Apply in place to one channel's samples.
    fn apply(&self, channel: usize, row: &mut [f32]) {
        let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
        match self {
            Self::Tanh => row.iter_mut().for_each(|x| *x = fast_tanh(*x)),
            Self::Hardtanh => row.iter_mut().for_each(|x| *x = x.clamp(-1.0, 1.0)),
            Self::Relu => row.iter_mut().for_each(|x| *x = x.max(0.0)),
            Self::LeakyRelu(s) => row.iter_mut().for_each(|x| *x = if *x > 0.0 { *x } else { *x * s }),
            Self::PRelu(slopes) => {
                let s = slopes.get(channel).or(slopes.last()).copied().unwrap_or(0.01);
                row.iter_mut().for_each(|x| *x = if *x > 0.0 { *x } else { *x * s });
            }
            Self::Sigmoid => row.iter_mut().for_each(|x| *x = sigmoid(*x)),
            Self::Silu => row.iter_mut().for_each(|x| *x *= sigmoid(*x)),
            Self::Hardswish => row.iter_mut().for_each(|x| *x = *x * (*x + 3.0).clamp(0.0, 6.0) / 6.0),
            Self::LeakyHardtanh { min_val, max_val, min_slope, max_slope } => row.iter_mut().for_each(|x| {
                *x = if *x < *min_val {
                    (*x - min_val) * min_slope + min_val
                } else if *x > *max_val {
                    (*x - max_val) * max_slope + max_val
                } else {
                    *x
                }
            }),
            Self::Softsign => row.iter_mut().for_each(|x| *x /= 1.0 + x.abs()),
        }
    }
}

fn field<'a>(v: &'a Value, key: &str) -> Result<&'a Value, String> {
    v.get(key).filter(|x| !x.is_null()).ok_or_else(|| format!("missing {key}"))
}

fn uint(v: &Value, key: &str) -> Result<usize, String> {
    field(v, key)?.as_u64().map(|x| x as usize).ok_or_else(|| format!("{key} is not a count"))
}

fn flag(v: &Value, key: &str) -> Result<bool, String> {
    field(v, key)?.as_bool().ok_or_else(|| format!("{key} is not a boolean"))
}

/// A per-layer setting given either once for the whole array or as a list.
fn per_layer<T: Clone>(v: Option<&Value>, n: usize, default: Option<T>, parse: impl Fn(&Value) -> Result<T, String>) -> Result<Vec<T>, String> {
    match v.filter(|v| !v.is_null()) {
        Some(Value::Array(items)) if items.len() == n => items.iter().map(parse).collect(),
        Some(Value::Array(_)) => Err("per-layer list does not match the number of layers".into()),
        Some(v) => Ok(vec![parse(v)?; n]),
        None => default.map(|d| vec![d; n]).ok_or_else(|| "missing per-layer setting".into()),
    }
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

    /// Conv1D weights are stored [out][in][k]; returns them as [k][out][in].
    fn conv(&mut self, out: usize, inp: usize, k: usize) -> Result<Vec<f32>, String> {
        let raw = self.take(out * inp * k)?;
        let mut w = vec![0.0; raw.len()];
        for o in 0..out {
            for i in 0..inp {
                for kk in 0..k {
                    w[(kk * out + o) * inp + i] = raw[(o * inp + i) * k + kk];
                }
            }
        }
        Ok(w)
    }
}

struct Layer {
    k: usize,
    dilation: usize,
    /// [k][conv_out][channels]; conv_out is twice the bottleneck when gated.
    conv_w: Vec<f32>,
    conv_b: Vec<f32>,
    /// [conv_out] (condition size 1)
    mixin_w: Vec<f32>,
    act: Activation,
    /// Gating activation applied to the second half of the conv output.
    gate: Option<Activation>,
    /// [channels][bottleneck] and bias; without it the residual is the input.
    layer1x1: Option<(Vec<f32>, Vec<f32>)>,
    /// [head_in][bottleneck] and bias; without it the activations feed the head.
    head1x1: Option<(Vec<f32>, Vec<f32>)>,
}

struct LayerArray {
    in_size: usize,
    ch: usize,
    bottleneck: usize,
    /// Rows summed into the head by every layer.
    head_in: usize,
    head_size: usize,
    head_k: usize,
    head_dilation: usize,
    /// [ch][in_size]
    rechannel: Vec<f32>,
    layers: Vec<Layer>,
    /// [head_k][head_size][head_in]
    head_w: Vec<f32>,
    head_b: Option<Vec<f32>>,
    /// One history buffer per layer input: [ch][buf_len]
    bufs: Vec<Vec<f32>>,
    /// History of the summed head input: [head_in][buf_len]
    head_buf: Vec<f32>,
    /// Output of the last layer for the current block: [ch][MAX_BLOCK]
    out: Vec<f32>,
    /// Head output: [head_size][MAX_BLOCK]
    head_out: Vec<f32>,
    z: Vec<f32>,
}

impl LayerArray {
    fn parse(lc: &Value, r: &mut Reader) -> Result<Self, String> {
        if uint(lc, "condition_size")? != 1 {
            return Err("only mono conditioning is supported".into());
        }
        for key in ["groups_input", "groups_input_mixin"] {
            if lc.get(key).and_then(Value::as_u64).unwrap_or(1) != 1 {
                return Err("grouped convolutions are not supported".into());
            }
        }
        for key in [
            "conv_pre_film",
            "conv_post_film",
            "input_mixin_pre_film",
            "input_mixin_post_film",
            "activation_pre_film",
            "activation_post_film",
            "layer1x1_post_film",
            "head1x1_post_film",
        ] {
            let active = match lc.get(key) {
                None | Some(Value::Null) | Some(Value::Bool(false)) => false,
                Some(v) => v.get("active").and_then(Value::as_bool).unwrap_or(true),
            };
            if active {
                return Err("FiLM conditioning is not supported".into());
            }
        }
        let in_size = uint(lc, "input_size")?;
        let ch = uint(lc, "channels")?;
        let bottleneck = lc.get("bottleneck").and_then(Value::as_u64).map_or(ch, |b| b as usize);
        let layer1x1 = match lc.get("layer1x1").filter(|v| v.is_object()) {
            Some(v) => {
                if v.get("groups").and_then(Value::as_u64).unwrap_or(1) != 1 {
                    return Err("grouped convolutions are not supported".into());
                }
                flag(v, "active")?
            }
            None => true,
        };
        if !layer1x1 && bottleneck != ch {
            return Err("bottleneck must equal channels without layer1x1".into());
        }
        let head1x1 = match lc.get("head1x1").filter(|v| v.is_object()) {
            Some(v) if flag(v, "active")? => {
                if v.get("groups").and_then(Value::as_u64).unwrap_or(1) != 1 {
                    return Err("grouped convolutions are not supported".into());
                }
                Some(uint(v, "out_channels")?)
            }
            _ => None,
        };
        let (head_size, head_k, head_dilation, head_bias) = match lc.get("head").filter(|v| !v.is_null()) {
            Some(h) => {
                (uint(h, "out_channels")?, uint(h, "kernel_size")?, h.get("head_dilation").and_then(Value::as_u64).unwrap_or(1) as usize, flag(h, "bias")?)
            }
            None => (uint(lc, "head_size")?, 1, 1, flag(lc, "head_bias")?),
        };
        let dilations: Vec<usize> = field(lc, "dilations")?
            .as_array()
            .ok_or("dilations is not a list")?
            .iter()
            .map(|d| d.as_u64().map(|d| d as usize).ok_or_else(|| "bad dilation".to_string()))
            .collect::<Result<_, _>>()?;
        let n = dilations.len();
        if n == 0 || head_k == 0 {
            return Err("empty layer array".into());
        }
        let kernels = per_layer(lc.get("kernel_sizes").or(lc.get("kernel_size")), n, None, |v| {
            v.as_u64().map(|k| k as usize).filter(|&k| k > 0).ok_or_else(|| "bad kernel size".into())
        })?;
        let acts = per_layer(lc.get("activation"), n, None, Activation::parse)?;
        let gated = match lc.get("gating_mode").filter(|v| !v.is_null()) {
            Some(g) => per_layer(Some(g), n, None, |v| match v.as_str() {
                Some("none") => Ok(false),
                Some("gated") => Ok(true),
                Some(m) => Err(format!("unsupported gating mode {m}")),
                None => Err("bad gating mode".into()),
            })?,
            None => vec![lc.get("gated").and_then(Value::as_bool).unwrap_or(false); n],
        };
        let gates = per_layer(lc.get("secondary_activation"), n, Some(None), |v| if v.is_null() { Ok(None) } else { Activation::parse(v).map(Some) })?;
        let head_in = head1x1.unwrap_or(bottleneck);

        let rechannel = r.take(ch * in_size)?;
        let mut layers = Vec::with_capacity(n);
        for i in 0..n {
            let gate = gated[i].then(|| gates[i].clone().unwrap_or(Activation::Sigmoid));
            let conv_out = if gate.is_some() { 2 * bottleneck } else { bottleneck };
            let k = kernels[i];
            let conv_w = r.conv(conv_out, ch, k)?;
            let conv_b = r.take(conv_out)?;
            let mixin_w = r.take(conv_out)?;
            let layer1x1 = if layer1x1 { Some((r.take(ch * bottleneck)?, r.take(ch)?)) } else { None };
            let head1x1 = match head1x1 {
                Some(out) => Some((r.take(out * bottleneck)?, r.take(out)?)),
                None => None,
            };
            layers.push(Layer { k, dilation: dilations[i], conv_w, conv_b, mixin_w, act: acts[i].clone(), gate, layer1x1, head1x1 });
        }
        let head_w = r.conv(head_size, head_in, head_k)?;
        let head_b = if head_bias { Some(r.take(head_size)?) } else { None };
        let z_rows = layers.iter().map(|l| l.conv_b.len()).max().unwrap_or(0);
        Ok(Self {
            in_size,
            ch,
            bottleneck,
            head_in,
            head_size,
            head_k,
            head_dilation,
            rechannel,
            layers,
            head_w,
            head_b,
            bufs: Vec::new(),
            head_buf: Vec::new(),
            out: vec![0.0; ch * MAX_BLOCK],
            head_out: vec![0.0; head_size * MAX_BLOCK],
            z: vec![0.0; z_rows * MAX_BLOCK],
        })
    }

    /// Samples of history the convolutions look back.
    fn history(&self) -> usize {
        let layers = self.layers.iter().map(|l| l.dilation * (l.k - 1)).max().unwrap_or(0);
        layers.max(self.head_dilation * (self.head_k - 1))
    }

    fn receptive_field(&self) -> usize {
        self.layers.iter().map(|l| l.dilation * (l.k - 1)).sum::<usize>() + self.head_dilation * (self.head_k - 1)
    }
}

/// Every history row of every array.
fn rows(arrays: &mut [LayerArray], buf_len: usize) -> impl Iterator<Item = &mut [f32]> {
    arrays.iter_mut().flat_map(move |a| a.bufs.iter_mut().flat_map(move |b| b.chunks_mut(buf_len)).chain(a.head_buf.chunks_mut(buf_len)))
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
    /// History of every row after settling on silence, see [`Self::prewarm`].
    settled: Vec<f32>,
}

impl WaveNet {
    pub fn from_json(bytes: &[u8]) -> Result<Self, String> {
        let root: Value = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
        Self::from_value(&root, None, None)
    }

    fn from_value(v: &Value, sample_rate: Option<f32>, loudness: Option<f32>) -> Result<Self, String> {
        let sample_rate = v.get("sample_rate").and_then(Value::as_f64).map(|x| x as f32).or(sample_rate);
        let loudness = v.pointer("/metadata/loudness").and_then(Value::as_f64).map(|x| x as f32).or(loudness);
        let config = field(v, "config")?;
        match v.get("architecture").and_then(Value::as_str).unwrap_or_default() {
            "WaveNet" => Self::build(config, field(v, "weights")?, sample_rate, loudness),
            // The same capture at several sizes; NAM runs the largest by default.
            "SlimmableContainer" => {
                let max_value = |s: &Value| s.get("max_value").and_then(Value::as_f64).unwrap_or(0.0);
                let largest = field(config, "submodels")?
                    .as_array()
                    .and_then(|s| s.iter().max_by(|a, b| max_value(a).total_cmp(&max_value(b))))
                    .ok_or("container without models")?;
                Self::from_value(field(largest, "model")?, sample_rate, loudness)
            }
            other => Err(format!("unsupported NAM architecture {other}")),
        }
    }

    fn build(config: &Value, weights: &Value, sample_rate: Option<f32>, loudness: Option<f32>) -> Result<Self, String> {
        if config.get("condition_dsp").is_some_and(|c| !c.is_null()) {
            return Err("condition DSPs are not supported".into());
        }
        if config.get("head").is_some_and(|h| !h.is_null()) {
            return Err("post-stack heads are not supported".into());
        }
        if config.get("in_channels").and_then(Value::as_u64).unwrap_or(1) != 1 {
            return Err("only mono models are supported".into());
        }
        let weights: Vec<f32> = weights.as_array().ok_or("weights is not a list")?.iter().map(|w| w.as_f64().unwrap_or(0.0) as f32).collect();
        let mut r = Reader { w: &weights, pos: 0 };
        let mut arrays = Vec::new();
        for lc in field(config, "layers")?.as_array().ok_or("layers is not a list")? {
            arrays.push(LayerArray::parse(lc, &mut r)?);
        }
        if arrays.is_empty() {
            return Err("no layer arrays".into());
        }
        // Each array continues the previous one's layer outputs and head.
        for w in arrays.windows(2) {
            if w[1].in_size != w[0].ch || w[1].head_in != w[0].head_size {
                return Err("inconsistent layer array sizes".into());
            }
        }
        if arrays[arrays.len() - 1].head_size != 1 {
            return Err("only mono models are supported".into());
        }
        // The trailing weight is the head scale; the config value is only a
        // construction default.
        let head_scale = match weights.len() - r.pos {
            1 => weights[r.pos],
            0 => config.get("head_scale").and_then(Value::as_f64).unwrap_or(1.0) as f32,
            extra => return Err(format!("{extra} more weights than the configuration uses")),
        };
        let history = arrays.iter().map(LayerArray::history).max().unwrap_or(0);
        let buf_len = history + MAX_BLOCK * 8;
        for a in &mut arrays {
            a.bufs = vec![vec![0.0; a.ch * buf_len]; a.layers.len()];
            a.head_buf = vec![0.0; a.head_in * buf_len];
        }
        let mut model = Self {
            arrays,
            head_scale,
            buf_len,
            history,
            start: history,
            sample_rate: sample_rate.unwrap_or(48000.0),
            loudness,
            cond: vec![0.0; MAX_BLOCK],
            settled: Vec::new(),
        };
        model.prewarm();
        Ok(model)
    }

    /// Biases leave the network in a non-zero state on silence. Like NAM,
    /// run silence through the whole receptive field before use, and keep
    /// that state so `reset` can return to it without recomputing.
    fn prewarm(&mut self) {
        let mut left = 1 + self.arrays.iter().map(LayerArray::receptive_field).sum::<usize>();
        let mut zeros = [0.0f32; MAX_BLOCK];
        while left > 0 {
            let n = left.min(MAX_BLOCK);
            zeros[..n].fill(0.0);
            self.process_block(&mut zeros[..n]);
            left -= n;
        }
        let (h, st) = (self.history, self.start);
        self.settled = rows(&mut self.arrays, self.buf_len).flat_map(|row| row[st - h..st].to_vec()).collect();
        self.reset();
    }

    pub fn reset(&mut self) {
        let h = self.history;
        if h > 0 {
            for (row, settled) in rows(&mut self.arrays, self.buf_len).zip(self.settled.chunks(h)) {
                row[..h].copy_from_slice(settled);
            }
        }
        self.start = h;
    }

    /// Process samples in place.
    pub fn process(&mut self, io: &mut [f32]) {
        for chunk in io.chunks_mut(MAX_BLOCK) {
            self.process_block(chunk);
        }
    }

    fn rewind_if_needed(&mut self, n: usize) {
        if self.start + n <= self.buf_len {
            return;
        }
        let (h, st) = (self.history, self.start);
        for row in rows(&mut self.arrays, self.buf_len) {
            row.copy_within(st - h..st, 0);
        }
        self.start = h;
    }

    fn process_block(&mut self, io: &mut [f32]) {
        let n = io.len();
        self.rewind_if_needed(n);
        let (bl, st) = (self.buf_len, self.start);
        self.cond[..n].copy_from_slice(io);
        let cond = &self.cond[..n];

        for ai in 0..self.arrays.len() {
            let (prev, rest) = self.arrays.split_at_mut(ai);
            let prev = prev.last();
            let a = &mut rest[0];
            let (ch, bn) = (a.ch, a.bottleneck);
            // Rechannel into the first layer buffer.
            for o in 0..ch {
                let dst = &mut a.bufs[0][o * bl + st..][..n];
                dst.fill(0.0);
                for i in 0..a.in_size {
                    let w = a.rechannel[o * a.in_size + i];
                    let src = match prev {
                        None => cond,
                        Some(p) => &p.out[i * MAX_BLOCK..][..n],
                    };
                    for (d, s) in dst.iter_mut().zip(src) {
                        *d += w * s;
                    }
                }
            }
            // The head sum starts from the previous array's head output.
            for c in 0..a.head_in {
                let dst = &mut a.head_buf[c * bl + st..][..n];
                match prev {
                    None => dst.fill(0.0),
                    Some(p) => dst.copy_from_slice(&p.head_out[c * MAX_BLOCK..][..n]),
                }
            }
            let n_layers = a.layers.len();
            for li in 0..n_layers {
                let layer = &a.layers[li];
                let (k, conv_out) = (layer.k, layer.conv_b.len());
                // z = conv(input) + bias + mixin * cond
                let input = &a.bufs[li];
                for o in 0..conv_out {
                    let z = &mut a.z[o * MAX_BLOCK..][..n];
                    let (b, m) = (layer.conv_b[o], layer.mixin_w[o]);
                    for (zv, c) in z.iter_mut().zip(cond) {
                        *zv = b + m * c;
                    }
                    for kk in 0..k {
                        let off = layer.dilation * (k - 1 - kk);
                        for i in 0..ch {
                            let w = layer.conv_w[(kk * conv_out + o) * ch + i];
                            let src = &input[i * bl + st - off..][..n];
                            for (zv, s) in z.iter_mut().zip(src) {
                                *zv += w * s;
                            }
                        }
                    }
                }
                // Activation; gated layers multiply by the activated second half.
                match &layer.gate {
                    None => {
                        for c in 0..bn {
                            layer.act.apply(c, &mut a.z[c * MAX_BLOCK..][..n]);
                        }
                    }
                    Some(gate) => {
                        let (top, bottom) = a.z.split_at_mut(bn * MAX_BLOCK);
                        for c in 0..bn {
                            let (t, g) = (&mut top[c * MAX_BLOCK..][..n], &mut bottom[c * MAX_BLOCK..][..n]);
                            layer.act.apply(c, t);
                            gate.apply(c, g);
                            for (tv, gv) in t.iter_mut().zip(g.iter()) {
                                *tv *= gv;
                            }
                        }
                    }
                }
                // Skip connection into the head sum.
                for h in 0..a.head_in {
                    let dst = &mut a.head_buf[h * bl + st..][..n];
                    match &layer.head1x1 {
                        None => {
                            for (d, zv) in dst.iter_mut().zip(&a.z[h * MAX_BLOCK..][..n]) {
                                *d += zv;
                            }
                        }
                        Some((w, b)) => {
                            dst.iter_mut().for_each(|d| *d += b[h]);
                            for c in 0..bn {
                                let wv = w[h * bn + c];
                                for (d, zv) in dst.iter_mut().zip(&a.z[c * MAX_BLOCK..][..n]) {
                                    *d += wv * zv;
                                }
                            }
                        }
                    }
                }
                // Residual: next = input + 1x1(z) + b
                let last = li + 1 == n_layers;
                for o in 0..ch {
                    let mut tmp = [0.0f32; MAX_BLOCK];
                    let t = &mut tmp[..n];
                    t.copy_from_slice(&a.bufs[li][o * bl + st..][..n]);
                    if let Some((w, b)) = &layer.layer1x1 {
                        t.iter_mut().for_each(|v| *v += b[o]);
                        for c in 0..bn {
                            let wv = w[o * bn + c];
                            for (tv, zv) in t.iter_mut().zip(&a.z[c * MAX_BLOCK..][..n]) {
                                *tv += wv * zv;
                            }
                        }
                    }
                    if last {
                        a.out[o * MAX_BLOCK..][..n].copy_from_slice(t);
                    } else {
                        a.bufs[li + 1][o * bl + st..][..n].copy_from_slice(t);
                    }
                }
            }
            // Head convolution over the summed skip outputs.
            for h in 0..a.head_size {
                let dst = &mut a.head_out[h * MAX_BLOCK..][..n];
                dst.fill(a.head_b.as_ref().map_or(0.0, |b| b[h]));
                for kk in 0..a.head_k {
                    let off = a.head_dilation * (a.head_k - 1 - kk);
                    for c in 0..a.head_in {
                        let w = a.head_w[(kk * a.head_size + h) * a.head_in + c];
                        for (d, s) in dst.iter_mut().zip(&a.head_buf[c * bl + st - off..][..n]) {
                            *d += w * s;
                        }
                    }
                }
            }
        }
        let last = &self.arrays[self.arrays.len() - 1];
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
    /// Why the model file could not be used.
    pub error: Option<String>,
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
        let (model, error) = match model_json.map(WaveNet::from_json) {
            Some(Ok(m)) => (Some(m), None),
            Some(Err(e)) => (None, Some(e)),
            None => (None, None),
        };
        // Normalize captures to a common loudness like the NAM plugin does.
        let norm_gain = model.as_ref().and_then(|m| m.loudness).map(|l| db_to_gain(-18.0 - l)).unwrap_or(1.0);
        let mut s = Self {
            sr,
            model,
            error,
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A small made-up model in a slimmable container that exercises
    /// per-layer kernel sizes, LeakyReLU, a head convolution, gating with a
    /// head 1x1 and no layer 1x1. Expected values (every 25th sample) are
    /// from NAM core's `render` tool with fast tanh enabled.
    const FEATURES: &[u8] = include_bytes!("../testdata/wavenet_features.nam");
    const EXPECTED: [f32; 120] = [
        0.05687504, -0.04661148, 0.004098516, -0.06205193, 0.01019803, 0.03529003, 0.09000033, 0.1516038,
        0.08278467, 0.1181758, 0.02584345, 0.06105244, -0.07422858, -0.01017127, 0.03630203, 0.0487765,
        0.05004364, 0.06400495, 0.08368389, 0.05424159, 0.06663945, 0.0383538, 0.04306647, 0.01142809,
        0.02677728, 0.04601973, 0.04013551, 0.07366034, 0.119962, 0.1688979, 0.02944611, 0.1076618,
        -0.03509207, 0.006516121, -0.08909977, -0.002209742, 0.02735368, 0.04683322, 0.06399722, 0.0704643,
        0.08446581, 0.04141145, 0.06451315, 0.01669325, 0.03781152, 0.01960288, 0.03725229, 0.0399527,
        0.06150908, 0.08062853, 0.07188044, 0.1383064, 0.03081098, 0.08130907, -0.05857395, 0.01492282,
        -0.05910091, 0.01210672, 0.02839299, 0.09186528, 0.1220764, 0.1105521, 0.07941953, 0.04692883,
        0.05763806, 0.01497875, 0.03579593, 0.01912232, 0.03662504, 0.05004181, 0.06330864, 0.08374038,
        0.05739447, 0.1068804, 0.02527879, 0.06105272, -0.07348828, -0.0001691282, 0.009152273, 0.03923513,
        0.03441247, 0.09946585, 0.1522523, 0.06672313, 0.1038269, 0.04072056, 0.04385856, 0.01142786,
        0.03114842, 0.04623751, 0.0487804, 0.0575064, 0.07007423, 0.08824602, 0.04317968, 0.06629919,
        0.02380769, 0.04098432, -0.08815136, -0.00221584, 0.02958625, 0.04951485, 0.08663, 0.1206929,
        0.1651294, 0.02144102, 0.1050263, -0.05862742, 0.008506056, -0.01491277, 0.03640082, 0.03995066,
        0.06444106, 0.06688501, 0.05679027, 0.07705596, 0.04333497, 0.05854122, 0.01671848, 0.04160591,
    ];

    fn input() -> Vec<f32> {
        (0..3000).map(|i| ((i * 37) % 64 - 32) as f32 / 64.0 * if (i / 300) % 2 == 0 { 1.0 } else { 0.25 }).collect()
    }

    fn check(out: &[f32]) {
        for (i, e) in EXPECTED.iter().enumerate() {
            assert!((out[i * 25] - e).abs() < 2e-5, "sample {}: {} vs {e}", i * 25, out[i * 25]);
        }
    }

    #[test]
    fn matches_nam_core() {
        let mut m = WaveNet::from_json(FEATURES).unwrap();
        assert_eq!(m.loudness, Some(-20.0));
        for block in [1024, 333, 64] {
            let mut x = input();
            x.chunks_mut(block).for_each(|c| m.process(c));
            check(&x);
            m.reset();
        }
        // Long silence wraps the history buffers and settles the same way.
        let mut silence = vec![0.0; 20_000];
        m.process(&mut silence);
        let mut x = input();
        m.process(&mut x);
        check(&x);
    }

    #[test]
    fn rejects_what_it_cannot_run() {
        let err = |json: &str| WaveNet::from_json(json.as_bytes()).err().unwrap_or_default();
        assert!(err(r#"{"architecture":"LSTM","config":{},"weights":[]}"#).contains("LSTM"));
        let mut model: Value = serde_json::from_slice(FEATURES).unwrap();
        let full = model.pointer_mut("/config/submodels/1/model").unwrap();
        full["config"]["layers"][0]["conv_post_film"]["active"] = Value::Bool(true);
        assert!(err(&full.to_string()).contains("FiLM"));
        full["config"]["layers"][0]["conv_post_film"]["active"] = Value::Bool(false);
        full["weights"].as_array_mut().unwrap().push(Value::from(0.0));
        assert!(err(&full.to_string()).contains("more weights"));
    }
}
