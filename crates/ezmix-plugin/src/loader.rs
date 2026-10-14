//! Background thread that parses presets and builds processors so the audio
//! thread never allocates or frees large structures.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;

use crossbeam_channel::{bounded, Sender};
use ezmix_content::preset::KnobGroup;
use ezmix_content::{Library, Preset, PresetRef};
use ezmix_dsp::{LoadReport, PresetProcessor};

use crate::MAX_SLOTS;

/// One library scan shared by every plugin instance in the process.
pub fn library() -> Arc<Library> {
    static LIB: OnceLock<Arc<Library>> = OnceLock::new();
    LIB.get_or_init(|| Arc::new(Library::scan())).clone()
}

#[derive(Debug, Clone)]
pub struct KnobInfo {
    pub name: String,
    pub default: f32,
    pub is_button: bool,
    pub button_group: bool,
}

/// What the editor shows about the preset loaded into a rack slot.
#[derive(Debug, Clone)]
pub struct LoadedInfo {
    pub pack_id: String,
    pub path: String,
    pub name: String,
    pub tags: ezmix_content::preset::Tags,
    pub knobs: Vec<KnobInfo>,
    pub groups: Vec<KnobGroup>,
    pub report: LoadReport,
}

impl LoadedInfo {
    pub fn from_preset(r: &PresetRef, p: &Preset, report: LoadReport) -> Self {
        Self {
            pack_id: r.pack_id.clone(),
            path: r.path.clone(),
            name: p.name.clone(),
            tags: p.tags.clone(),
            knobs: p.knobs.iter().map(|k| KnobInfo { name: k.name.clone(), default: k.value, is_button: k.is_button, button_group: k.button_group }).collect(),
            groups: p.groups.clone(),
            report,
        }
    }
}

pub enum Request {
    Load { slot: usize, pack_id: String, path: String, sample_rate: f32, generation: u64 },
    Clear { slot: usize, generation: u64 },
    Drop(Box<PresetProcessor>),
}

/// A finished build (or an empty slot) waiting for the audio thread.
pub struct Ready {
    pub processor: Option<Box<PresetProcessor>>,
    pub generation: u64,
}

/// Loader-side state of one rack slot. Slots are identified by a fixed
/// channel index that also selects their host parameters.
#[derive(Default)]
pub struct SlotShared {
    pub info: Mutex<Option<LoadedInfo>>,
    pub error: Mutex<Option<String>>,
    pub loading: AtomicBool,
    /// Generation of the latest request; stale results are discarded.
    pub generation: AtomicU64,
    /// Waiting for the audio thread (which only `try_lock`s).
    pub pending: Mutex<Option<Ready>>,
}

/// State shared between the plugin, its loader thread and the editor.
pub struct Shared {
    pub library: Arc<Library>,
    pub slots: [SlotShared; MAX_SLOTS],
    /// Set whenever any slot has a pending result.
    pub has_pending: AtomicBool,
    /// Processing order of the rack, see [`pack_order`].
    pub order: AtomicU64,
    next_generation: AtomicU64,
    pub requests: Sender<Request>,
    pub sample_rate: Mutex<f32>,
    pub in_peak: atomic_f32::AtomicF32,
    pub out_peak: atomic_f32::AtomicF32,
}

/// Tiny atomic f32 for meters.
pub mod atomic_f32 {
    use std::sync::atomic::{AtomicU32, Ordering};

    #[derive(Default)]
    pub struct AtomicF32(AtomicU32);

    impl AtomicF32 {
        pub fn load(&self) -> f32 {
            f32::from_bits(self.0.load(Ordering::Relaxed))
        }

        pub fn store(&self, v: f32) {
            self.0.store(v.to_bits(), Ordering::Relaxed)
        }
    }
}

/// Packs up to [`MAX_SLOTS`] slot channels into one word so the audio thread
/// can read the rack order without locking: the count in the low four bits,
/// then four bits per channel.
pub fn pack_order(channels: impl IntoIterator<Item = usize>) -> u64 {
    let mut word = 0u64;
    let mut n = 0u64;
    for ch in channels.into_iter().take(MAX_SLOTS) {
        word |= ((ch as u64) & 0xf) << (4 + 4 * n);
        n += 1;
    }
    word | n
}

pub fn unpack_order(word: u64) -> ([usize; MAX_SLOTS], usize) {
    let n = ((word & 0xf) as usize).min(MAX_SLOTS);
    let mut out = [0; MAX_SLOTS];
    for (i, ch) in out.iter_mut().enumerate().take(n) {
        *ch = ((word >> (4 + 4 * i)) & 0xf) as usize % MAX_SLOTS;
    }
    (out, n)
}

impl Shared {
    fn begin_request(&self, slot: usize) -> u64 {
        let generation = self.next_generation.fetch_add(1, Ordering::SeqCst) + 1;
        self.slots[slot].generation.store(generation, Ordering::SeqCst);
        generation
    }

    /// Ask the loader to build a preset into `slot`. Returns immediately.
    pub fn request_load(&self, slot: usize, pack_id: &str, path: &str) {
        let generation = self.begin_request(slot);
        self.slots[slot].loading.store(true, Ordering::Relaxed);
        let sample_rate = *self.sample_rate.lock().unwrap_or_else(|e| e.into_inner());
        let _ = self.requests.send(Request::Load { slot, pack_id: pack_id.into(), path: path.into(), sample_rate, generation });
    }

    /// Empty `slot`; the audio thread fades it out and frees its processor.
    pub fn request_clear(&self, slot: usize) {
        let generation = self.begin_request(slot);
        self.slots[slot].loading.store(false, Ordering::Relaxed);
        let _ = self.requests.send(Request::Clear { slot, generation });
    }

    pub fn publish_order(&self, channels: impl IntoIterator<Item = usize>) {
        self.order.store(pack_order(channels), Ordering::Release);
    }

    pub fn info(&self, slot: usize) -> Option<LoadedInfo> {
        self.slots[slot].info.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    fn is_latest(&self, slot: usize, generation: u64) -> bool {
        self.slots[slot].generation.load(Ordering::SeqCst) == generation
    }

    fn publish(&self, slot: usize, ready: Ready) {
        // Replace any result the audio thread has not picked up yet; the old
        // one is freed here.
        let old = self.slots[slot].pending.lock().unwrap_or_else(|e| e.into_inner()).replace(ready);
        self.has_pending.store(true, Ordering::Release);
        drop(old);
    }
}

/// Spawns the loader thread and returns the shared state.
pub fn spawn() -> Arc<Shared> {
    let (req_tx, req_rx) = bounded::<Request>(64);
    let shared = Arc::new(Shared {
        library: library(),
        slots: Default::default(),
        has_pending: AtomicBool::new(false),
        order: AtomicU64::new(pack_order([0])),
        next_generation: AtomicU64::new(0),
        requests: req_tx,
        sample_rate: Mutex::new(48000.0),
        in_peak: Default::default(),
        out_peak: Default::default(),
    });
    let weak = Arc::downgrade(&shared);
    thread::Builder::new()
        .name("ez-er-mix loader".into())
        .spawn(move || {
            while let Ok(req) = req_rx.recv() {
                let Some(shared) = weak.upgrade() else { break };
                match req {
                    Request::Drop(p) => drop(p),
                    Request::Clear { slot, generation } => {
                        if shared.is_latest(slot, generation) {
                            *shared.slots[slot].info.lock().unwrap_or_else(|e| e.into_inner()) = None;
                            *shared.slots[slot].error.lock().unwrap_or_else(|e| e.into_inner()) = None;
                            shared.publish(slot, Ready { processor: None, generation });
                        }
                    }
                    Request::Load { slot, pack_id, path, sample_rate, generation } => {
                        // Skip requests that were superseded while queued.
                        if !shared.is_latest(slot, generation) {
                            continue;
                        }
                        let lib = &shared.library;
                        let result = lib
                            .find_preset(&pack_id, &path)
                            .cloned()
                            .ok_or_else(|| format!("Preset not found: {pack_id}{path}"))
                            .and_then(|r| lib.load_preset(&r).map(|p| (r, p)));
                        let state = &shared.slots[slot];
                        match result {
                            Ok((r, preset)) => {
                                let processor = Box::new(PresetProcessor::new(&preset, lib, sample_rate));
                                if !shared.is_latest(slot, generation) {
                                    continue;
                                }
                                let info = LoadedInfo::from_preset(&r, &preset, processor.report.clone());
                                *state.info.lock().unwrap_or_else(|e| e.into_inner()) = Some(info);
                                *state.error.lock().unwrap_or_else(|e| e.into_inner()) = None;
                                shared.publish(slot, Ready { processor: Some(processor), generation });
                            }
                            Err(e) => {
                                *state.error.lock().unwrap_or_else(|e| e.into_inner()) = Some(e);
                            }
                        }
                        if shared.is_latest(slot, generation) {
                            state.loading.store(false, Ordering::Relaxed);
                        }
                    }
                }
            }
        })
        .expect("spawn loader thread");
    shared
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn order_round_trips() {
        for order in [vec![], vec![0], vec![3, 0, 7], (0..MAX_SLOTS).rev().collect::<Vec<_>>()] {
            let (out, n) = unpack_order(pack_order(order.iter().copied()));
            assert_eq!(&out[..n], &order[..]);
        }
    }
}
