//! Zero-latency non-uniformly partitioned FFT convolution.
//!
//! The first `DIRECT` taps run as a time-domain FIR. Later IR segments are
//! handled by uniformly partitioned overlap-save stages with growing block
//! sizes; every stage starts at an IR offset no smaller than its block size,
//! so its block latency is hidden and the whole convolver adds no delay.

use std::sync::Arc;

use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

const DIRECT: usize = 64;
/// (block size, IR segment start, IR segment end)
const STAGES: [(usize, usize, usize); 5] =
    [(64, 64, 1024), (256, 1024, 4096), (1024, 4096, 16384), (4096, 16384, 65536), (16384, 65536, usize::MAX)];

struct Stage {
    block: usize,
    start: usize,
    fwd: Arc<dyn RealToComplex<f32>>,
    inv: Arc<dyn ComplexToReal<f32>>,
    /// IR partitions in the frequency domain.
    h: Vec<Vec<Complex<f32>>>,
    /// Frequency-domain delay line of past input frames.
    fdl: Vec<Vec<Complex<f32>>>,
    fdl_pos: usize,
    /// Time-domain frame: previous block followed by current block.
    frame: Vec<f32>,
    fill: usize,
    acc: Vec<Complex<f32>>,
    scratch_t: Vec<f32>,
    scratch_c: Vec<Complex<f32>>,
    fft_scratch: Vec<Complex<f32>>,
}

impl Stage {
    fn new(planner: &mut RealFftPlanner<f32>, block: usize, start: usize, ir: &[f32]) -> Self {
        let n = 2 * block;
        let fwd = planner.plan_fft_forward(n);
        let inv = planner.plan_fft_inverse(n);
        let bins = block + 1;
        let mut h = Vec::new();
        let mut buf = vec![0.0f32; n];
        let scale = 1.0 / n as f32;
        for chunk in ir.chunks(block) {
            buf.iter_mut().for_each(|v| *v = 0.0);
            for (d, s) in buf.iter_mut().zip(chunk) {
                *d = s * scale;
            }
            let mut spec = vec![Complex::new(0.0, 0.0); bins];
            fwd.process(&mut buf, &mut spec).expect("fft sizes match");
            h.push(spec);
        }
        let parts = h.len();
        let scratch_len = fwd.get_scratch_len().max(inv.get_scratch_len());
        Self {
            block,
            start,
            fwd,
            inv,
            h,
            fdl: vec![vec![Complex::new(0.0, 0.0); bins]; parts],
            fdl_pos: 0,
            frame: vec![0.0; n],
            fill: 0,
            acc: vec![Complex::new(0.0, 0.0); bins],
            scratch_t: vec![0.0; n],
            scratch_c: vec![Complex::new(0.0, 0.0); bins],
            fft_scratch: vec![Complex::new(0.0, 0.0); scratch_len],
        }
    }

    /// Feed one sample; when a block completes, add its contribution to the
    /// output ring starting `delay_from_now` samples after the current one.
    #[inline]
    fn push(&mut self, x: f32, out: &mut OutRing) {
        self.frame[self.block + self.fill] = x;
        self.fill += 1;
        if self.fill < self.block {
            return;
        }
        self.fill = 0;
        let parts = self.h.len();
        self.scratch_t.copy_from_slice(&self.frame);
        self.fdl_pos = (self.fdl_pos + parts - 1) % parts;
        let slot = &mut self.fdl[self.fdl_pos];
        self.fwd.process_with_scratch(&mut self.scratch_t, slot, &mut self.fft_scratch).expect("fft sizes match");
        self.acc.iter_mut().for_each(|v| *v = Complex::new(0.0, 0.0));
        for p in 0..parts {
            let x = &self.fdl[(self.fdl_pos + p) % parts];
            for ((a, xv), hv) in self.acc.iter_mut().zip(x).zip(&self.h[p]) {
                *a += xv * hv;
            }
        }
        self.scratch_c.copy_from_slice(&self.acc);
        // DC and Nyquist bins must be real for the inverse real FFT.
        self.scratch_c[0].im = 0.0;
        let last = self.scratch_c.len() - 1;
        self.scratch_c[last].im = 0.0;
        self.inv.process_with_scratch(&mut self.scratch_c, &mut self.scratch_t, &mut self.fft_scratch).expect("fft sizes match");
        // Valid outputs are the second half; the first of them belongs to the
        // oldest sample of the block, i.e. (block - 1) samples ago, shifted by
        // the IR segment start.
        let offset = self.start + 1 - self.block;
        out.add(offset, &self.scratch_t[self.block..]);
        // Slide the frame.
        let (a, b) = self.frame.split_at_mut(self.block);
        a.copy_from_slice(b);
    }
}

struct OutRing {
    buf: Vec<f32>,
    pos: usize,
}

impl OutRing {
    #[inline]
    fn add(&mut self, offset: usize, data: &[f32]) {
        let n = self.buf.len();
        let mut p = (self.pos + offset) % n;
        for &v in data {
            self.buf[p] += v;
            p += 1;
            if p == n {
                p = 0;
            }
        }
    }

    #[inline]
    fn take(&mut self) -> f32 {
        let v = self.buf[self.pos];
        self.buf[self.pos] = 0.0;
        self.pos += 1;
        if self.pos == self.buf.len() {
            self.pos = 0;
        }
        v
    }
}

pub struct Convolver {
    head: Vec<f32>,
    hist: Vec<f32>,
    hist_pos: usize,
    stages: Vec<Stage>,
    out: OutRing,
}

impl Convolver {
    pub fn new(ir: &[f32]) -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let head: Vec<f32> = ir.iter().take(DIRECT).copied().collect();
        let mut stages = Vec::new();
        let mut max_reach = DIRECT;
        for &(block, start, end) in &STAGES {
            if start >= ir.len() {
                break;
            }
            let seg = &ir[start..end.min(ir.len())];
            max_reach = max_reach.max(start + seg.len().div_ceil(block) * block + block);
            stages.push(Stage::new(&mut planner, block, start, seg));
        }
        Self {
            hist: vec![0.0; DIRECT],
            hist_pos: 0,
            head,
            stages,
            out: OutRing { buf: vec![0.0; (max_reach + 1).next_power_of_two()], pos: 0 },
        }
    }

    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        // Time-domain head.
        self.hist_pos = if self.hist_pos == 0 { DIRECT - 1 } else { self.hist_pos - 1 };
        self.hist[self.hist_pos] = x;
        let mut y = 0.0;
        let n = self.head.len();
        for (k, h) in self.head.iter().enumerate().take(n) {
            y += h * self.hist[(self.hist_pos + k) % DIRECT];
        }
        for s in &mut self.stages {
            s.push(x, &mut self.out);
        }
        y + self.out.take()
    }

    pub fn reset(&mut self) {
        self.hist.iter_mut().for_each(|v| *v = 0.0);
        self.out.buf.iter_mut().for_each(|v| *v = 0.0);
        for s in &mut self.stages {
            s.frame.iter_mut().for_each(|v| *v = 0.0);
            s.fill = 0;
            for f in &mut s.fdl {
                f.iter_mut().for_each(|v| *v = Complex::new(0.0, 0.0));
            }
        }
    }
}

/// Windowed-sinc resampling of an impulse response (offline).
pub fn resample(ir: &[f32], from: f32, to: f32) -> Vec<f32> {
    if (from - to).abs() < 1.0 || ir.is_empty() {
        return ir.to_vec();
    }
    let ratio = to / from;
    let out_len = ((ir.len() as f32) * ratio).ceil() as usize;
    let cutoff = ratio.min(1.0);
    let half = 16.0f32;
    let mut out = vec![0.0f32; out_len];
    for (i, o) in out.iter_mut().enumerate() {
        let t = i as f32 / ratio;
        let lo = (t - half / cutoff).floor().max(0.0) as usize;
        let hi = ((t + half / cutoff).ceil() as usize).min(ir.len() - 1);
        let mut acc = 0.0;
        for (j, s) in ir.iter().enumerate().take(hi + 1).skip(lo) {
            let x = (t - j as f32) * cutoff;
            let sinc = if x.abs() < 1e-6 { 1.0 } else { (std::f32::consts::PI * x).sin() / (std::f32::consts::PI * x) };
            let w = 0.5 + 0.5 * (std::f32::consts::PI * x / half).cos();
            if x.abs() < half {
                acc += s * sinc * w;
            }
        }
        *o = acc * cutoff;
    }
    out
}

/// Minimum-phase FIR from a magnitude response sampled on `n/2+1` linear
/// bins (real cepstrum method).
pub fn min_phase_fir(mag: &[f32], taps: usize) -> Vec<f32> {
    let bins = mag.len();
    let n = (bins - 1) * 2;
    let mut planner = RealFftPlanner::<f32>::new();
    let fwd = planner.plan_fft_forward(n);
    let inv = planner.plan_fft_inverse(n);
    let mut spec: Vec<Complex<f32>> = mag.iter().map(|&m| Complex::new(m.max(1e-6).ln(), 0.0)).collect();
    let mut cep = vec![0.0f32; n];
    inv.process(&mut spec, &mut cep).expect("fft sizes match");
    for v in cep.iter_mut() {
        *v /= n as f32;
    }
    // Fold to a causal cepstrum.
    for (i, v) in cep.iter_mut().enumerate() {
        if i == 0 || i == n / 2 {
            continue;
        }
        *v = if i < n / 2 { *v * 2.0 } else { 0.0 };
    }
    let mut c = vec![Complex::new(0.0, 0.0); bins];
    fwd.process(&mut cep, &mut c).expect("fft sizes match");
    let mut ex: Vec<Complex<f32>> = c.iter().map(|z| z.exp()).collect();
    ex[0].im = 0.0;
    ex[bins - 1].im = 0.0;
    let mut h = vec![0.0f32; n];
    inv.process(&mut ex, &mut h).expect("fft sizes match");
    let taps = taps.min(n);
    let fade = taps / 4;
    h.truncate(taps);
    for (i, v) in h.iter_mut().enumerate() {
        *v /= n as f32;
        if i >= taps - fade {
            let t = (taps - i) as f32 / fade as f32;
            *v *= 0.5 - 0.5 * (std::f32::consts::PI * t).cos();
        }
    }
    h
}
