//! Background thread that parses presets and builds processors so the audio
//! thread never allocates or frees large structures.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;

use crossbeam_channel::{bounded, Sender};
use ezmix_content::preset::KnobGroup;
use ezmix_content::{Library, Preset, PresetRef};
use ezmix_dsp::{LoadReport, PresetProcessor};

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

/// What the editor shows about the active preset.
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
            knobs: p
                .knobs
                .iter()
                .map(|k| KnobInfo { name: k.name.clone(), default: k.value, is_button: k.is_button, button_group: k.button_group })
                .collect(),
            groups: p.groups.clone(),
            report,
        }
    }
}

pub enum Request {
    Load { pack_id: String, path: String, sample_rate: f32, generation: u64 },
    Drop(Box<PresetProcessor>),
}

pub struct Ready {
    pub processor: Box<PresetProcessor>,
    pub generation: u64,
}

/// State shared between the plugin, its loader thread and the editor.
pub struct Shared {
    pub library: Arc<Library>,
    pub current: Mutex<Option<LoadedInfo>>,
    pub error: Mutex<Option<String>>,
    pub loading: AtomicBool,
    /// Incremented for every load request; stale results are discarded.
    pub generation: AtomicU64,
    pub requests: Sender<Request>,
    /// Finished build waiting for the audio thread (which only `try_lock`s).
    pub pending: Mutex<Option<Ready>>,
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

impl Shared {
    /// Ask the loader to build a preset. Returns immediately.
    pub fn request_load(&self, pack_id: &str, path: &str) {
        let generation = self.generation.fetch_add(1, Ordering::SeqCst) + 1;
        self.loading.store(true, Ordering::Relaxed);
        let sample_rate = *self.sample_rate.lock().unwrap_or_else(|e| e.into_inner());
        let _ = self.requests.send(Request::Load { pack_id: pack_id.into(), path: path.into(), sample_rate, generation });
    }
}

/// Spawns the loader thread and returns the shared state.
pub fn spawn() -> Arc<Shared> {
    let (req_tx, req_rx) = bounded::<Request>(64);
    let shared = Arc::new(Shared {
        library: library(),
        current: Mutex::new(None),
        error: Mutex::new(None),
        loading: AtomicBool::new(false),
        generation: AtomicU64::new(0),
        requests: req_tx,
        pending: Mutex::new(None),
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
                    Request::Load { pack_id, path, sample_rate, generation } => {
                        // Skip requests that were superseded while queued.
                        if shared.generation.load(Ordering::SeqCst) != generation {
                            continue;
                        }
                        let lib = &shared.library;
                        let result = lib
                            .find_preset(&pack_id, &path)
                            .cloned()
                            .ok_or_else(|| format!("Preset not found: {pack_id}{path}"))
                            .and_then(|r| lib.load_preset(&r).map(|p| (r, p)));
                        match result {
                            Ok((r, preset)) => {
                                let processor = Box::new(PresetProcessor::new(&preset, lib, sample_rate));
                                let info = LoadedInfo::from_preset(&r, &preset, processor.report.clone());
                                *shared.current.lock().unwrap_or_else(|e| e.into_inner()) = Some(info);
                                *shared.error.lock().unwrap_or_else(|e| e.into_inner()) = None;
                                // Replace any build the audio thread has not
                                // picked up yet; the old one is freed here.
                                let old = shared.pending.lock().unwrap_or_else(|e| e.into_inner()).replace(Ready { processor, generation });
                                drop(old);
                            }
                            Err(e) => {
                                *shared.error.lock().unwrap_or_else(|e| e.into_inner()) = Some(e);
                            }
                        }
                        shared.loading.store(false, Ordering::Relaxed);
                    }
                }
            }
        })
        .expect("spawn loader thread");
    shared
}
