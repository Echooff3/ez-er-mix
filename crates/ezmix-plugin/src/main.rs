use cpal::traits::{DeviceTrait, HostTrait};
use nih_plug::prelude::*;

use ez_er_mix::EzErMix;

/// Standalone entry point. Unless overridden on the command line, connect the
/// system's default input device and run at the output device's native rate,
/// so the app works when launched from Finder.
fn main() {
    let mut args: Vec<String> = std::env::args().collect();
    let given = |args: &[String], long: &str, short: Option<&str>| {
        args.iter().any(|a| a == long || a.starts_with(&format!("{long}=")) || short.is_some_and(|s| a == s))
    };
    let host = cpal::default_host();
    if !given(&args, "--input-device", None) {
        // Built-in microphones next to built-in speakers would feed back, so
        // those must be chosen explicitly with --input-device.
        if let Some(name) = host.default_input_device().and_then(|d| d.name().ok()).filter(|n| !n.contains("Microphone")) {
            args.extend(["--input-device".into(), name]);
        }
    }
    if !given(&args, "--output-device", None) {
        if let Some(name) = host.default_output_device().and_then(|d| d.name().ok()) {
            args.extend(["--output-device".into(), name]);
        }
    }
    if !given(&args, "--sample-rate", Some("-r")) {
        if let Some(rate) = host.default_output_device().and_then(|d| d.default_output_config().ok()).map(|c| c.sample_rate().0) {
            args.extend(["--sample-rate".into(), rate.to_string()]);
        }
    }
    if !given(&args, "--period-size", Some("-p")) {
        args.extend(["--period-size".into(), "256".into()]);
    }
    nih_export_standalone_with_args::<EzErMix, _>(args);
}
