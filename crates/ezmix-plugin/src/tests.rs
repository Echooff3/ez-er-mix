//! Rack behaviour on the audio thread, driven with the installed packs.

use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use super::*;

const SR: f32 = 48000.0;
const PLEXI: (&str, &str) = ("EMX3_Core3", "/Presets/68 Plexi Crunch");
const PLATE: (&str, &str) = ("EMX3_Core2", "/Presets/Ronnie/Overhead with Plate");

fn plugin() -> Option<EzErMix> {
    let mut p = EzErMix::default();
    if p.shared.library.find_preset(PLEXI.0, PLEXI.1).is_none() || p.shared.library.find_preset(PLATE.0, PLATE.1).is_none() {
        eprintln!("test presets not installed; skipping");
        return None;
    }
    p.sample_rate = SR;
    *p.shared.sample_rate.lock().unwrap() = SR;
    // The wrappers normally prime the smoothers.
    p.params.input.smoothed.reset(p.params.input.value());
    p.params.output.smoothed.reset(p.params.output.value());
    for s in &p.params.slots {
        for k in &s.knobs {
            k.value.smoothed.reset(k.value.value());
        }
    }
    Some(p)
}

/// Waits for the loader to finish everything requested so far.
fn settle(p: &EzErMix) {
    let start = Instant::now();
    while p.shared.slots.iter().any(|s| s.loading.load(Ordering::Relaxed)) || !p.shared.requests.is_empty() {
        assert!(start.elapsed() < Duration::from_secs(20), "loader timed out");
        std::thread::sleep(Duration::from_millis(5));
    }
    std::thread::sleep(Duration::from_millis(20));
}

/// Processes `ms` of a 220 Hz sine and returns the output.
fn run(p: &mut EzErMix, ms: f32, phase: &mut f32) -> Vec<f32> {
    let n = (SR * ms / 1000.0) as usize;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n.div_ceil(256) {
        let mut l = [0.0f32; 256];
        for x in l.iter_mut() {
            *x = 0.2 * (*phase).sin();
            *phase += std::f32::consts::TAU * 220.0 / SR;
        }
        let mut r = l;
        p.process_stereo(120.0, &mut l, &mut r);
        out.extend_from_slice(&l);
    }
    out
}

fn names(p: &EzErMix) -> Vec<String> {
    p.order.0[..p.order.1].iter().map(|&c| p.slots[c].processor.as_ref().map(|x| x.name.clone()).unwrap_or_default()).collect()
}

#[test]
fn rack_loads_reorders_swaps_and_removes() {
    let Some(mut p) = plugin() else { return };
    let mut phase = 0.0;

    // Two slots in series.
    p.shared.publish_order([0, 1]);
    p.shared.request_load(0, PLEXI.0, PLEXI.1);
    p.shared.request_load(1, PLATE.0, PLATE.1);
    settle(&p);
    let out = run(&mut p, 200.0, &mut phase);
    assert!(out.iter().all(|x| x.is_finite()));
    assert_eq!(names(&p), ["68 Plexi Crunch", "Overhead with Plate"]);
    assert_eq!(p.wet, 1.0);
    assert!(p.slots[0].wet == 1.0 && p.slots[1].wet == 1.0);
    let rms = |v: &[f32]| (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt();
    assert!(rms(&out[out.len() / 2..]) > 1e-3, "rack is silent");

    // Reordering fades the rack out, swaps, and fades back in.
    p.shared.publish_order([1, 0]);
    run(&mut p, 5.0, &mut phase);
    assert_eq!(names(&p), ["68 Plexi Crunch", "Overhead with Plate"], "order must wait for the fade");
    assert!(p.wet < 1.0);
    run(&mut p, 60.0, &mut phase);
    assert_eq!(names(&p), ["Overhead with Plate", "68 Plexi Crunch"]);
    assert_eq!(p.wet, 1.0);

    // Adding an empty slot changes nothing audible, so it is immediate.
    p.shared.publish_order([1, 0, 2]);
    run(&mut p, 2.0, &mut phase);
    assert_eq!(p.order.1, 3);
    assert_eq!(p.wet, 1.0);

    // Loading into a slot that is playing fades it out before swapping.
    p.shared.request_load(0, PLATE.0, PLATE.1);
    settle(&p);
    run(&mut p, 5.0, &mut phase);
    assert_eq!(p.slots[0].processor.as_ref().unwrap().name, "68 Plexi Crunch");
    run(&mut p, 60.0, &mut phase);
    assert_eq!(p.slots[0].processor.as_ref().unwrap().name, "Overhead with Plate");
    assert_eq!(p.slots[0].wet, 1.0);

    // Removing a slot empties it.
    p.shared.publish_order([1, 2]);
    p.shared.request_clear(0);
    settle(&p);
    let out = run(&mut p, 100.0, &mut phase);
    assert!(out.iter().all(|x| x.is_finite()));
    assert!(p.slots[0].processor.is_none() && p.slots[0].incoming.is_none());
    assert_eq!(names(&p), ["Overhead with Plate", ""]);
    assert_eq!(p.wet, 1.0);

    // A stale build never replaces a newer request.
    p.shared.request_load(2, PLEXI.0, PLEXI.1);
    p.shared.request_clear(2);
    settle(&p);
    run(&mut p, 60.0, &mut phase);
    assert!(p.slots[2].processor.is_none());
    assert!(p.shared.info(2).is_none());
}

/// The default rack: a TONE3000 NAM capture into a cabinet IR, when those
/// files are in a capture folder.
#[test]
fn default_rack_plays_capture_into_cabinet() {
    let Some(mut p) = plugin() else { return };
    let rack = crate::editor::default_rack(&p.shared.library);
    if rack.len() < 2 {
        eprintln!("default captures not installed; skipping");
        return;
    }
    let mut phase = 0.0;
    p.shared.publish_order([0, 1]);
    for (channel, sel) in rack.iter().enumerate() {
        p.shared.request_load(channel, &sel.pack_id, &sel.path);
    }
    settle(&p);
    let out = run(&mut p, 600.0, &mut phase);
    assert_eq!(names(&p), ["5150 MAX Chainsaw", "EVH 5150 III"]);
    for channel in 0..2 {
        let info = p.shared.info(channel).unwrap();
        assert!(info.report.missing_models.is_empty() && info.report.load_errors.is_empty(), "{:?}", info.report);
    }
    assert_eq!(p.shared.info(0).unwrap().knobs.len(), 7);
    let tail = &out[out.len() / 2..];
    let rms_db = 10.0 * (tail.iter().map(|x| x * x).sum::<f32>() / tail.len() as f32).log10();
    assert!(tail.iter().all(|x| x.is_finite()) && (-40.0..0.0).contains(&rms_db), "rms {rms_db} dB");
}

#[test]
fn rack_state_sanitizes() {
    let mut rack = RackState {
        slots: vec![
            RackSlot { channel: 3, preset: PresetSelection::default() },
            RackSlot { channel: 3, preset: PresetSelection::default() },
            RackSlot { channel: 42, preset: PresetSelection::default() },
        ],
        active: 9,
    };
    rack.sanitize();
    assert_eq!(rack.channels().collect::<Vec<_>>(), [3]);
    assert_eq!(rack.active, 0);
    assert_eq!(rack.free_channel(), Some(0));

    let mut empty = RackState { slots: vec![], active: 0 };
    empty.sanitize();
    assert_eq!(empty, RackState::default());

    let json = serde_json::to_string(&RackState::default()).unwrap();
    assert_eq!(serde_json::from_str::<RackState>(&json).unwrap(), RackState::default());
    // Fields missing from older or hand-written states fall back to defaults.
    assert_eq!(serde_json::from_str::<RackState>("{}").unwrap(), RackState::default());
}

#[test]
fn param_ids_are_unique_and_stable() {
    let params = EzParams::default();
    let map = params.param_map();
    let ids: Vec<&str> = map.iter().map(|(id, _, _)| id.as_str()).collect();
    let mut sorted = ids.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted.len(), ids.len());
    assert_eq!(ids.len(), 3 + MAX_SLOTS * (1 + NUM_KNOBS));
    for id in ["on_1", "macro_1_1", "macro_8_8", "on_8", "input", "output", "bypass"] {
        assert!(ids.contains(&id), "missing {id}");
    }
}
