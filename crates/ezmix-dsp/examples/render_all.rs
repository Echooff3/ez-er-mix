//! Renders test audio through every installed preset and reports problems:
//! non-finite output, runaway levels, silence, and CPU cost.
use std::time::Instant;

use ezmix_content::Library;
use ezmix_dsp::PresetProcessor;

fn test_signal(sr: f32, secs: f32) -> (Vec<f32>, Vec<f32>) {
    let n = (sr * secs) as usize;
    let mut l = vec![0.0; n];
    let mut r = vec![0.0; n];
    let mut seed = 1u32;
    for i in 0..n {
        let t = i as f32 / sr;
        // Plucked notes every 0.5 s plus a little noise.
        let note_t = t % 0.5;
        let f = [110.0, 146.8, 196.0, 82.4][(t / 0.5) as usize % 4];
        let env = (-note_t * 6.0).exp();
        let saw = 2.0 * ((t * f).fract()) - 1.0;
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        let noise = (seed >> 9) as f32 / (1u32 << 23) as f32 - 1.0;
        let x = 0.3 * env * saw + 0.01 * noise;
        l[i] = x;
        r[i] = x * 0.9;
    }
    (l, r)
}

fn main() {
    let filter = std::env::args().nth(1);
    let lib = Library::scan();
    let sr = 48000.0;
    let (src_l, src_r) = test_signal(sr, 2.0);
    let audio_secs = 2.0 * 2.0;
    let mut worst: Vec<(f64, String)> = Vec::new();
    let (mut bad, mut total) = (0, 0);
    for pack in &lib.packs {
        for pr in &pack.presets {
            if let Some(f) = &filter {
                if !pr.path.contains(f.as_str()) {
                    continue;
                }
            }
            let Ok(preset) = lib.load_preset(pr) else { continue };
            total += 1;
            let t0 = Instant::now();
            let mut p = PresetProcessor::new(&preset, &lib, sr);
            let build = t0.elapsed().as_secs_f64();
            let t1 = Instant::now();
            let (mut peak, mut energy, mut nonfinite) = (0.0f32, 0.0f64, false);
            let mut pass_peaks = [0.0f32; 2];
            for pass in 0..2 {
                let (mut l, mut r) = (src_l.clone(), src_r.clone());
                for (bi, (bl, br)) in l.chunks_mut(64).zip(r.chunks_mut(64)).enumerate() {
                    if pass == 1 {
                        // Sweep every knob continuously while audio runs.
                        for k in 0..p.knobs.len() {
                            let v = 0.5 + 0.5 * (bi as f32 * 0.004 + k as f32).sin();
                            p.set_knob(k, v);
                        }
                    }
                    p.process(120.0, bl, br);
                }
                for (a, b) in l.iter().zip(&r) {
                    if !a.is_finite() || !b.is_finite() {
                        nonfinite = true;
                    }
                    peak = peak.max(a.abs()).max(b.abs());
                    pass_peaks[pass] = pass_peaks[pass].max(a.abs()).max(b.abs());
                    energy += (*a as f64).powi(2) + (*b as f64).powi(2);
                }
            }
            let cpu = t1.elapsed().as_secs_f64() / audio_secs;
            let rms_db = 10.0 * (energy / (src_l.len() as f64 * 4.0)).max(1e-20).log10();
            let issue = if nonfinite {
                "NONFINITE"
            } else if peak >= 7.9 {
                "CLIPPED"
            } else if rms_db < -70.0 {
                "SILENT"
            } else {
                ""
            };
            let rep = &p.report;
            if !issue.is_empty() || !rep.unsupported.is_empty() || !rep.missing_models.is_empty() || filter.is_some() {
                println!(
                    "{:<9} {:<26} {:<48} rms {:6.1} dB peak {:5.2} {:?} cpu {:5.1}% build {:4.0}ms unsupported {:?} missing {:?} neural {}",
                    issue, pack.id, pr.path, rms_db, peak, pass_peaks, cpu * 100.0, build * 1000.0, rep.unsupported, rep.missing_models, rep.skipped_neural
                );
            }
            if !issue.is_empty() {
                bad += 1;
            }
            worst.push((cpu, format!("{}{}", pack.id, pr.path)));
        }
    }
    worst.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    println!("\n{bad} problem presets out of {total}");
    println!("highest CPU (fraction of real time, single core, 48 kHz):");
    for (c, n) in worst.iter().take(8) {
        println!("  {:5.1}%  {n}", c * 100.0);
    }
    let avg = worst.iter().map(|w| w.0).sum::<f64>() / worst.len().max(1) as f64;
    println!("average {:.2}%", avg * 100.0);
}
