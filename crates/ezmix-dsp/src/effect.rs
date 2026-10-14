//! Effect interface shared by every processor in the engine.

use crate::dsp::note_seconds;

/// Per-block context.
#[derive(Clone, Copy, Debug)]
pub struct Ctx {
    pub sr: f32,
    pub bpm: f32,
}

impl Default for Ctx {
    fn default() -> Self {
        Self { sr: 48_000.0, bpm: 120.0 }
    }
}

/// Stereo, in-place effect. Parameters use the engine's user units (dB, Hz,
/// seconds, percent, 0..10 knobs, selector indices) exactly as stored in
/// presets.
pub trait Effect: Send {
    fn prepare(&mut self, sr: f32);

    fn set(&mut self, name: &str, value: f32);

    /// Tempo-syncable parameter: `synced` is an index into the parameter's
    /// note table.
    fn set_sync(&mut self, name: &str, unsynced: f32, synced: f32, sync: bool) {
        let _ = (synced, sync);
        self.set(name, unsynced);
    }

    fn set_data(&mut self, _name: &str, _data: &[u8]) {}

    fn reset(&mut self) {}

    fn process(&mut self, ctx: &Ctx, l: &mut [f32], r: &mut [f32]);
}

/// Pass-through used for empty slots and unsupported effect types.
pub struct Bypass;

impl Effect for Bypass {
    fn prepare(&mut self, _sr: f32) {}
    fn set(&mut self, _name: &str, _value: f32) {}
    fn process(&mut self, _ctx: &Ctx, _l: &mut [f32], _r: &mut [f32]) {}
}

pub const DELAY_NOTES: &[&str] = &[
    "1/32t", "1/32", "1/32d", "1/16t", "1/16", "1/16d", "1/8t", "1/8", "1/8d", "1/4t", "1/4", "1/4d", "1/2t", "1/2",
    "1/2d", "1/1",
];
pub const RATE_NOTES_LONG: &[&str] = &[
    "4/1", "3/1", "2/1", "1/1", "1/2d", "1/2", "1/2t", "1/4d", "1/4", "1/4t", "1/8d", "1/8", "1/8t", "1/16d", "1/16",
    "1/16t", "1/32d", "1/32", "1/32t",
];
pub const RATE_NOTES: &[&str] = &[
    "1/1", "1/2d", "1/2", "1/2t", "1/4d", "1/4", "1/4t", "1/8d", "1/8", "1/8t", "1/16d", "1/16", "1/16t", "1/32d",
    "1/32", "1/32t",
];
pub const SPREAD_NOTES: &[&str] =
    &["1/64", "1/32t", "1/32", "1/32d", "1/16t", "1/16", "1/16d", "1/8t", "1/8", "1/8d", "1/4t", "1/4"];
pub const PREDELAY_NOTES: &[&str] = &["1/32", "1/32d", "1/16", "1/16d", "1/8", "1/8d", "1/4", "1/4d", "1/2"];
pub const INVERSE_NOTES: &[&str] =
    &["1/16t", "1/16", "1/16d", "1/8t", "1/8", "1/8d", "1/4t", "1/4", "1/4d", "1/2t", "1/2", "1/2d"];

/// A free-running value that can alternatively follow the host tempo.
#[derive(Clone, Copy, Debug)]
pub struct SyncParam {
    pub free: f32,
    pub index: usize,
    pub synced: bool,
    table: &'static [&'static str],
    /// The free value is a rate in Hz rather than a time in seconds.
    rate: bool,
}

impl SyncParam {
    pub fn time(free: f32, table: &'static [&'static str]) -> Self {
        Self { free, index: 0, synced: false, table, rate: false }
    }

    pub fn rate(free: f32, table: &'static [&'static str]) -> Self {
        Self { free, index: 0, synced: false, table, rate: true }
    }

    pub fn set(&mut self, unsynced: f32, synced: f32, sync: bool) {
        self.free = unsynced;
        self.index = synced.max(0.0).round() as usize;
        self.synced = sync;
    }

    /// Seconds for time parameters, Hz for rate parameters.
    pub fn value(&self, bpm: f32) -> f32 {
        if !self.synced {
            return self.free;
        }
        let label = self.table[self.index.min(self.table.len() - 1)];
        let secs = note_seconds(label, bpm);
        if self.rate { 1.0 / secs.max(1e-3) } else { secs }
    }
}

/// Process a stereo buffer sample by sample.
#[inline]
pub fn for_each(l: &mut [f32], r: &mut [f32], mut f: impl FnMut(f32, f32) -> (f32, f32)) {
    for (a, b) in l.iter_mut().zip(r.iter_mut()) {
        let (x, y) = f(*a, *b);
        *a = x;
        *b = y;
    }
}

#[inline]
pub fn on(v: f32) -> bool {
    v > 0.5
}

/// Selector value as an index.
#[inline]
pub fn idx(v: f32) -> usize {
    v.max(0.0).round() as usize
}
