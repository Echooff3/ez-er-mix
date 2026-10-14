//! Discovery of installed preset packs and model resources.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::preset::{self, Preset};
use crate::ttfs::Archive;

const EZMIX3_PREFS: &str = "/Users/Shared/Toontrack/Preferences/com.toontrack.ezmix3.plist";
const DEFAULT_PACK_DIRS: &[&str] = &["/Library/Application Support/EZmix", "/Users/Shared/Toontrack/EZmix/Presets"];
const DEFAULT_RESOURCE_DIR: &str = "/Library/Application Support/Toontrack/EZmix 3";

#[derive(Debug, Clone)]
pub struct PresetRef {
    /// Pack file name, e.g. `EMX3_Core2`. Stable identifier used in saved state.
    pub pack_id: String,
    /// Path inside the pack, e.g. `/Presets/Inserts/Bass/Bass`.
    pub path: String,
    pub name: String,
    /// Folder path below `/Presets`, `""` for top-level presets.
    pub folder: String,
}

#[derive(Debug, Clone)]
pub struct Pack {
    pub id: String,
    pub name: String,
    pub file: PathBuf,
    pub presets: Vec<PresetRef>,
}

impl Pack {
    /// Presets grouped by folder, in folder order.
    pub fn folders(&self) -> BTreeMap<&str, Vec<&PresetRef>> {
        let mut map: BTreeMap<&str, Vec<&PresetRef>> = BTreeMap::new();
        for p in &self.presets {
            map.entry(p.folder.as_str()).or_default().push(p);
        }
        map
    }
}

#[derive(Debug, Clone)]
pub struct Paths {
    pub pack_dirs: Vec<PathBuf>,
    pub resource_dir: PathBuf,
}

impl Paths {
    /// Pack and resource locations from the EZmix 3 preferences, falling back
    /// to the default install locations.
    pub fn detect() -> Self {
        let mut pack_dirs: Vec<PathBuf> = Vec::new();
        let mut resource_dir = PathBuf::from(DEFAULT_RESOURCE_DIR);
        if let Ok(plist::Value::Dictionary(d)) = plist::Value::from_file(EZMIX3_PREFS) {
            if let Some(p) = d.get("PresetPackPath").and_then(|v| v.as_string()) {
                pack_dirs.push(PathBuf::from(p));
            }
            if let Some(p) = d.get("ResourcePath").and_then(|v| v.as_string()) {
                resource_dir = PathBuf::from(p);
            }
        }
        for d in DEFAULT_PACK_DIRS {
            let d = PathBuf::from(d);
            if !pack_dirs.contains(&d) {
                pack_dirs.push(d);
            }
        }
        Self { pack_dirs, resource_dir }
    }
}

#[derive(Debug, Clone)]
pub struct NamModelLocation {
    pub file: PathBuf,
    pub entry: String,
}

pub struct Library {
    pub packs: Vec<Pack>,
    pub paths: Paths,
    /// NAM captures shipped inside packs, keyed by stem.
    pack_nam_models: HashMap<String, NamModelLocation>,
    archives: Mutex<HashMap<PathBuf, Arc<Archive>>>,
}

impl Library {
    pub fn scan() -> Self {
        Self::scan_paths(Paths::detect())
    }

    pub fn scan_paths(paths: Paths) -> Self {
        let mut packs = Vec::new();
        let mut pack_nam_models = HashMap::new();
        let mut archives = HashMap::new();
        for dir in &paths.pack_dirs {
            let Ok(read) = std::fs::read_dir(dir) else { continue };
            let mut files: Vec<PathBuf> = read.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.is_file()).collect();
            files.sort();
            for file in files {
                let Ok(archive) = Archive::open(&file) else { continue };
                let id = file.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                if packs.iter().any(|p: &Pack| p.id == id) {
                    continue;
                }
                let name = archive.get_string("/name").unwrap_or_else(|| id.clone());
                let mut presets = Vec::new();
                for e in archive.entries() {
                    if e.is_dir {
                        continue;
                    }
                    if let Some(rest) = e.path.strip_prefix("/Presets/") {
                        let (folder, leaf) = match rest.rsplit_once('/') {
                            Some((f, l)) => (f.to_owned(), l.to_owned()),
                            None => (String::new(), rest.to_owned()),
                        };
                        presets.push(PresetRef { pack_id: id.clone(), path: e.path.clone(), name: leaf, folder });
                    } else if let Some(model) = e.path.strip_prefix("/NAMModels/") {
                        if let Some(stem) = model.strip_suffix(".nam") {
                            pack_nam_models
                                .entry(stem.to_owned())
                                .or_insert(NamModelLocation { file: file.clone(), entry: e.path.clone() });
                        }
                    }
                }
                presets.sort_by(|a, b| (&a.folder, &a.name).cmp(&(&b.folder, &b.name)));
                if !presets.is_empty() {
                    packs.push(Pack { id, name, file: file.clone(), presets });
                }
                archives.insert(file, Arc::new(archive));
            }
        }
        // Core packs first, then add-ons alphabetically.
        packs.sort_by_key(|p| (!p.id.starts_with("EMX3_Core"), p.name.to_lowercase()));
        Self { packs, paths, pack_nam_models, archives: Mutex::new(archives) }
    }

    pub fn pack(&self, id: &str) -> Option<&Pack> {
        self.packs.iter().find(|p| p.id == id)
    }

    pub fn find_preset(&self, pack_id: &str, path: &str) -> Option<&PresetRef> {
        self.pack(pack_id)?.presets.iter().find(|p| p.path == path)
    }

    fn archive(&self, file: &Path) -> std::io::Result<Arc<Archive>> {
        let mut cache = self.archives.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(a) = cache.get(file) {
            return Ok(a.clone());
        }
        let a = Arc::new(Archive::open(file)?);
        cache.insert(file.to_path_buf(), a.clone());
        Ok(a)
    }

    pub fn load_preset(&self, r: &PresetRef) -> Result<Preset, String> {
        let pack = self.pack(&r.pack_id).ok_or_else(|| format!("pack {} not installed", r.pack_id))?;
        let archive = self.archive(&pack.file).map_err(|e| e.to_string())?;
        let bytes = archive.get(&r.path).ok_or_else(|| format!("{} missing from {}", r.path, r.pack_id))?;
        preset::parse(&r.name, bytes).map_err(|e| format!("{}: {e}", r.name))
    }

    /// Raw `.nam` JSON for a model referenced by a preset.
    pub fn load_nam_model(&self, model: &preset::NamModelRef<'_>) -> Option<Vec<u8>> {
        match model {
            preset::NamModelRef::None => None,
            preset::NamModelRef::Internal(stem) => {
                let models = self.paths.resource_dir.join("Models");
                let archive = self.archive(&models).ok()?;
                archive.get(&format!("/NeuralAmpModeler/Amps/{stem}.nam")).map(<[u8]>::to_vec)
            }
            preset::NamModelRef::Custom(stem) => {
                let loc = self.pack_nam_models.get(*stem)?;
                let archive = self.archive(&loc.file).ok()?;
                archive.get(&loc.entry).map(<[u8]>::to_vec)
            }
        }
    }

    pub fn preset_count(&self) -> usize {
        self.packs.iter().map(|p| p.presets.len()).sum()
    }
}
