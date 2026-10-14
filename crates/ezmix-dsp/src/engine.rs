//! Renders an [`EngineState`]: up to five mixable channels (IN, AUX1-4) with
//! sends between them, a mix bus and an OUT chain.

use ezmix_content::engine::{
    channel_db_from_norm, send_db_from_norm, send_endpoints, EngineState, ParamValue, Routing, Target, Xfader, CH_AUX1,
    CH_AUX2, CH_AUX3, CH_AUX4, CH_IN, CH_OUT, SEND_NAMES,
};
use ezmix_content::ranges;

use crate::dsp::{db_to_gain, FourBandSplit};
use crate::effect::{Ctx, Effect};
use crate::fx;

pub const BLOCK: usize = 64;

struct Slot {
    id: String,
    fx: Box<dyn Effect>,
    enabled: bool,
}

#[derive(Default)]
struct Chain {
    slots: Vec<Slot>,
    l: Vec<f32>,
    r: Vec<f32>,
    /// Signal arriving from sends (AUX channels).
    in_l: Vec<f32>,
    in_r: Vec<f32>,
    used: bool,
}

impl Chain {
    fn new() -> Self {
        Self { l: vec![0.0; BLOCK], r: vec![0.0; BLOCK], in_l: vec![0.0; BLOCK], in_r: vec![0.0; BLOCK], ..Default::default() }
    }
}

/// Linear gain ramp across a block to avoid zipper noise.
#[derive(Clone, Copy, Default)]
struct Ramp {
    current: f32,
}

impl Ramp {
    #[inline]
    fn gains(&mut self, target: f32, n: usize) -> (f32, f32) {
        let start = self.current;
        self.current = target;
        (start, (target - start) / n.max(1) as f32)
    }
}

pub struct EngineGraph {
    chains: Vec<Chain>,
    routing: Routing,
    xfaders: Vec<Xfader>,
    in_ramp: Ramp,
    mix_ramps: [(Ramp, Ramp); 6],
    send_ramps: Vec<Ramp>,
    out_ramps: (Ramp, Ramp),
    multiband: Option<[FourBandSplit; 2]>,
    mix_l: Vec<f32>,
    mix_r: Vec<f32>,
    pub unsupported: Vec<String>,
}

/// Channels that carry signal into the mix, in processing order.
const MIX_ORDER: [usize; 5] = [CH_IN, CH_AUX1, CH_AUX2, CH_AUX3, CH_AUX4];

impl EngineGraph {
    /// Build all processors (allocates; call off the audio thread).
    pub fn new(state: &EngineState, sr: f32) -> Self {
        let mut unsupported = Vec::new();
        let mut chains: Vec<Chain> = (0..6).map(|_| Chain::new()).collect();
        for (ci, ch) in state.channels.iter().enumerate() {
            for e in &ch.slots {
                let (fx, known) = match fx::create(&e.id) {
                    Some(f) => (f, true),
                    None => (Box::new(crate::effect::Bypass) as Box<dyn Effect>, false),
                };
                if !known && !unsupported.contains(&e.id) {
                    unsupported.push(e.id.clone());
                }
                let mut fx = fx;
                fx.prepare(sr);
                apply_params(fx.as_mut(), &e.id, &e.params);
                let enabled = e.power && !e.is_dummy() && known;
                chains[ci].used |= enabled;
                chains[ci].slots.push(Slot { id: e.id.clone(), fx, enabled });
            }
        }
        let multiband = state.routing.multiband.map(|f| {
            let mut s: [FourBandSplit; 2] = Default::default();
            s.iter_mut().for_each(|x| x.set(sr, f));
            s
        });
        let send_count = state.routing.sends.len();
        let mut g = Self {
            chains,
            routing: state.routing.clone(),
            xfaders: state.xfaders.clone(),
            in_ramp: Ramp::default(),
            mix_ramps: Default::default(),
            send_ramps: vec![Ramp::default(); send_count],
            out_ramps: Default::default(),
            multiband,
            mix_l: vec![0.0; BLOCK],
            mix_r: vec![0.0; BLOCK],
            unsupported,
        };
        g.snap_ramps();
        g
    }

    fn snap_ramps(&mut self) {
        self.in_ramp.current = db_to_gain(self.routing.input_db);
        for (i, s) in self.routing.sends.iter().enumerate() {
            self.send_ramps[i].current = db_to_gain(s.level_db);
        }
        for ch in MIX_ORDER {
            let (gl, gr) = self.mix_gains(ch);
            self.mix_ramps[ch].0.current = gl;
            self.mix_ramps[ch].1.current = gr;
        }
        let (gl, gr) = self.out_gains();
        self.out_ramps.0.current = gl;
        self.out_ramps.1.current = gr;
    }

    pub fn xfader_count(&self) -> usize {
        self.xfaders.len()
    }

    /// Move macro fader `index` to `value` (0..1). Real-time safe.
    pub fn set_xfader(&mut self, index: usize, value: f32) {
        let Some(xf) = self.xfaders.get_mut(index) else { return };
        xf.value = value;
        for k in 0..self.xfaders[index].interpolations.len() {
            let it = &self.xfaders[index].interpolations[k];
            let n = it.eval(value);
            match &it.target {
                Target::Effect { channel, slot, param } => {
                    if let Some(s) = self.chains.get_mut(*channel).and_then(|c| c.slots.get_mut(*slot)) {
                        if param == "power" {
                            s.enabled = n > 0.5 && s.id != "TA_DUMMY";
                        } else {
                            let v = ranges::denormalize(&s.id, param, n);
                            s.fx.set(param, v);
                        }
                    }
                }
                Target::ChannelVolume(ch) => {
                    if let Some(m) = self.routing.mix.get_mut(*ch) {
                        m.volume_db = channel_db_from_norm(n);
                    }
                }
                Target::Send(k) => {
                    let level = send_db_from_norm(n);
                    if self.routing.sends.len() == 1 && *k == 0 {
                        self.routing.sends[0].level_db = level;
                    } else if let Some(name) = SEND_NAMES.get(*k) {
                        let (from, to) = send_endpoints(name);
                        for s in &mut self.routing.sends {
                            if s.from == from && s.to == to {
                                s.level_db = level;
                            }
                        }
                    }
                }
                Target::Unknown(_) => {}
            }
        }
    }

    pub fn reset(&mut self) {
        for c in &mut self.chains {
            for s in &mut c.slots {
                s.fx.reset();
            }
        }
    }

    fn mix_gains(&self, ch: usize) -> (f32, f32) {
        let any_solo = MIX_ORDER.iter().any(|&c| self.routing.mix[c].solo);
        let m = &self.routing.mix[ch];
        if m.mute || (any_solo && !m.solo) {
            return (0.0, 0.0);
        }
        let g = db_to_gain(m.volume_db) * if m.phase_invert { -1.0 } else { 1.0 };
        let p = m.pan.clamp(-1.0, 1.0);
        (g * (1.0 - p).min(1.0), g * (1.0 + p).min(1.0))
    }

    fn out_gains(&self) -> (f32, f32) {
        let g = db_to_gain(self.routing.output_db);
        let p = self.routing.output_pan.clamp(-1.0, 1.0);
        (g * (1.0 - p).min(1.0), g * (1.0 + p).min(1.0))
    }

    /// Process a block of at most [`BLOCK`] frames in place.
    pub fn process_block(&mut self, ctx: &Ctx, l: &mut [f32], r: &mut [f32]) {
        let n = l.len().min(BLOCK);
        for c in &mut self.chains {
            c.in_l[..n].iter_mut().for_each(|v| *v = 0.0);
            c.in_r[..n].iter_mut().for_each(|v| *v = 0.0);
        }
        let (g0, dg) = self.in_ramp.gains(db_to_gain(self.routing.input_db), n);
        if let Some(split) = &mut self.multiband {
            // Bands feed IN, AUX1, AUX2, AUX3.
            let dests = [CH_IN, CH_AUX1, CH_AUX2, CH_AUX3];
            for i in 0..n {
                let g = g0 + dg * i as f32;
                let bl = split[0].split(l[i] * g);
                let br = split[1].split(r[i] * g);
                for (b, &d) in dests.iter().enumerate() {
                    self.chains[d].in_l[i] = bl[b];
                    self.chains[d].in_r[i] = br[b];
                }
            }
        } else {
            let c = &mut self.chains[CH_IN];
            for i in 0..n {
                let g = g0 + dg * i as f32;
                c.in_l[i] = l[i] * g;
                c.in_r[i] = r[i] * g;
            }
        }

        self.mix_l[..n].iter_mut().for_each(|v| *v = 0.0);
        self.mix_r[..n].iter_mut().for_each(|v| *v = 0.0);

        for &ch in &MIX_ORDER {
            let has_input = ch == CH_IN
                || self.multiband.is_some()
                || self.routing.sends.iter().any(|s| s.to == ch && s.level_db > -200.0);
            if !has_input && !self.chains[ch].used {
                continue;
            }
            {
                let c = &mut self.chains[ch];
                c.l[..n].copy_from_slice(&c.in_l[..n]);
                c.r[..n].copy_from_slice(&c.in_r[..n]);
            }
            let nslots = self.chains[ch].slots.len();
            for pos in 0..=nslots {
                // Sends tapped before slot `pos` (or after the chain).
                for si in 0..self.routing.sends.len() {
                    let s = self.routing.sends[si];
                    if s.from != ch || s.to == ch {
                        continue;
                    }
                    let tap = s.split.min(nslots);
                    if tap != pos {
                        continue;
                    }
                    let (sg0, sdg) = self.send_ramps[si].gains(db_to_gain(s.level_db), n);
                    if sg0 == 0.0 && sdg == 0.0 {
                        continue;
                    }
                    let (src, dst) = two_mut(&mut self.chains, ch, s.to);
                    for i in 0..n {
                        let g = sg0 + sdg * i as f32;
                        dst.in_l[i] += src.l[i] * g;
                        dst.in_r[i] += src.r[i] * g;
                    }
                }
                if pos < nslots {
                    let c = &mut self.chains[ch];
                    let slot = &mut c.slots[pos];
                    if slot.enabled {
                        slot.fx.process(ctx, &mut c.l[..n], &mut c.r[..n]);
                    }
                }
            }
            let (gl, gr) = self.mix_gains(ch);
            let (l0, dl) = self.mix_ramps[ch].0.gains(gl, n);
            let (r0, dr) = self.mix_ramps[ch].1.gains(gr, n);
            let c = &self.chains[ch];
            for i in 0..n {
                self.mix_l[i] += c.l[i] * (l0 + dl * i as f32);
                self.mix_r[i] += c.r[i] * (r0 + dr * i as f32);
            }
        }

        // OUT chain.
        let out = &mut self.chains[CH_OUT];
        for slot in &mut out.slots {
            if slot.enabled {
                slot.fx.process(ctx, &mut self.mix_l[..n], &mut self.mix_r[..n]);
            }
        }
        let (gl, gr) = if self.routing.output_bypass { (1.0, 1.0) } else { self.out_gains() };
        let (l0, dl) = self.out_ramps.0.gains(gl, n);
        let (r0, dr) = self.out_ramps.1.gains(gr, n);
        for i in 0..n {
            l[i] = self.mix_l[i] * (l0 + dl * i as f32);
            r[i] = self.mix_r[i] * (r0 + dr * i as f32);
        }
    }
}

fn two_mut(v: &mut [Chain], a: usize, b: usize) -> (&mut Chain, &mut Chain) {
    assert!(a != b);
    if a < b {
        let (x, y) = v.split_at_mut(b);
        (&mut x[a], &mut y[0])
    } else {
        let (x, y) = v.split_at_mut(a);
        (&mut y[0], &mut x[b])
    }
}

/// Push stored preset values into a freshly created effect.
pub fn apply_params(fx: &mut dyn Effect, _id: &str, params: &[(String, ParamValue)]) {
    // Scalars first so data-driven effects build with their final settings.
    for (name, v) in params {
        match v {
            ParamValue::Num(x) => fx.set(name, *x),
            ParamValue::Sync { unsynced, synced, sync } => fx.set_sync(name, *unsynced, *synced, *sync),
            _ => {}
        }
    }
    for (name, v) in params {
        if let ParamValue::Data(d) = v {
            if !d.is_empty() {
                fx.set_data(name, d);
            }
        }
    }
}
