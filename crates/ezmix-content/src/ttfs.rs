//! Reader for Toontrack "ttfs" containers (EZmix pack files and the
//! `Models` resource file).
//!
//! Layout (all integers little-endian):
//!
//! ```text
//! 0x00  "ttfs"
//! 0x06  u16  root directory offset minus 0x10
//! 0x10  format tag ("EMX", "EMX2", "EMX3", ...) followed by an opaque
//!       signature block that runs up to the root directory
//! root  directory table
//! ```
//!
//! A directory table is a `u32` entry count followed by entries of
//! `u32 offset, u32 size, NUL-terminated name`. Sub-directories are entries
//! with size 0 whose offset points at another directory table; an offset of
//! `0xFFFF_FFFF` marks an empty placeholder. File payloads are stored in the
//! clear.

use std::collections::HashSet;
use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};

use memmap2::Mmap;

#[derive(Debug, Clone)]
pub struct Entry {
    /// Path inside the container, always starting with `/`.
    pub path: String,
    pub offset: u64,
    pub size: u64,
    pub is_dir: bool,
}

pub struct Archive {
    path: PathBuf,
    map: Mmap,
    entries: Vec<Entry>,
}

const MAX_ENTRIES_PER_DIR: u32 = 200_000;

impl Archive {
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)?;
        // SAFETY: pack files are read-only content installed by the vendor. If
        // another process truncates the file while it is mapped, reads could
        // fault, which is the usual memory-mapping caveat.
        let map = unsafe { Mmap::map(&file)? };
        if map.len() < 0x20 || &map[0..4] != b"ttfs" {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "not a ttfs container"));
        }
        let root = u16::from_le_bytes([map[6], map[7]]) as usize + 0x10;
        let mut entries = Vec::new();
        let mut seen = HashSet::new();
        walk(&map, root, "", &mut entries, &mut seen, 0)?;
        Ok(Self { path, map, entries })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The format tag at offset 0x10, e.g. `EMX3`.
    pub fn format_tag(&self) -> String {
        let tag = &self.map[0x10..0x16];
        let end = tag.iter().position(|&b| b == 0).unwrap_or(tag.len());
        String::from_utf8_lossy(&tag[..end]).into_owned()
    }

    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    pub fn data(&self, entry: &Entry) -> &[u8] {
        let start = entry.offset as usize;
        let end = start.saturating_add(entry.size as usize).min(self.map.len());
        &self.map[start.min(end)..end]
    }

    pub fn find(&self, path: &str) -> Option<&Entry> {
        self.entries.iter().find(|e| e.path == path)
    }

    pub fn get(&self, path: &str) -> Option<&[u8]> {
        self.find(path).filter(|e| !e.is_dir).map(|e| self.data(e))
    }

    pub fn get_string(&self, path: &str) -> Option<String> {
        self.get(path).map(decode_name)
    }
}

fn read_u32(buf: &[u8], at: usize) -> io::Result<u32> {
    buf.get(at..at + 4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "truncated ttfs directory"))
}

fn walk(
    buf: &[u8],
    table: usize,
    prefix: &str,
    out: &mut Vec<Entry>,
    seen: &mut HashSet<usize>,
    depth: usize,
) -> io::Result<()> {
    if depth > 32 || !seen.insert(table) {
        return Ok(());
    }
    let count = read_u32(buf, table)?;
    if count > MAX_ENTRIES_PER_DIR {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "implausible ttfs entry count"));
    }
    let mut p = table + 4;
    for _ in 0..count {
        let offset = read_u32(buf, p)?;
        let size = read_u32(buf, p + 4)?;
        p += 8;
        let name_len = buf[p..]
            .iter()
            .position(|&b| b == 0)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "unterminated name"))?;
        let name = decode_name(&buf[p..p + name_len]);
        p += name_len + 1;
        let path = format!("{prefix}/{name}");

        let is_dir = size == 0 && offset != u32::MAX && looks_like_table(buf, offset as usize);
        out.push(Entry { path: path.clone(), offset: offset as u64, size: size as u64, is_dir });
        if is_dir {
            walk(buf, offset as usize, &path, out, seen, depth + 1)?;
        }
    }
    Ok(())
}

fn looks_like_table(buf: &[u8], at: usize) -> bool {
    match read_u32(buf, at) {
        Ok(n) => n > 0 && n <= MAX_ENTRIES_PER_DIR && at + 4 + 9 <= buf.len(),
        Err(_) => false,
    }
}

/// Names are UTF-8 in current packs; fall back to Latin-1 for older ones.
pub fn decode_name(bytes: &[u8]) -> String {
    match std::str::from_utf8(bytes) {
        Ok(s) => s.to_owned(),
        Err(_) => bytes.iter().map(|&b| b as char).collect(),
    }
}
