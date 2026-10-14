//! Audio engine that renders EZmix presets.

// Effects are constructed with explicit settings; `new()` reads clearer than
// `Default` at the factory call sites.
#![allow(clippy::new_without_default)]

pub mod conv;
pub mod dsp;
pub mod effect;
pub mod engine;
pub mod fx;
pub mod irdecode;
pub mod nam;
pub mod processor;

pub use processor::{LoadReport, PresetProcessor};
