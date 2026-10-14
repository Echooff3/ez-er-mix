use ezmix_content::Library;
use ezmix_dsp::PresetProcessor;

fn main() {
    let needle = std::env::args().nth(1).unwrap();
    let lib = Library::scan();
    let r = lib.packs.iter().flat_map(|p| &p.presets).find(|p| p.path.contains(&needle)).unwrap().clone();
    let preset = lib.load_preset(&r).unwrap();
    let mut p = PresetProcessor::new(&preset, &lib, 48000.0);
    let mut phase = 0.0f32;
    for blk in 0..400 {
        let mut l = vec![0.0f32; 256];
        for x in l.iter_mut() {
            phase = (phase + 110.0 / 48000.0).fract();
            *x = 0.3 * (2.0 * phase - 1.0);
        }
        let mut r2 = l.clone();
        let mut knob_vals = vec![];
        for k in 0..p.knobs.len() {
            let v = (p.knobs[k].value + 0.013).fract();
            p.set_knob(k, v);
            knob_vals.push(v);
        }
        p.process(120.0, &mut l, &mut r2);
        let peak = l.iter().chain(&r2).fold(0.0f32, |a, b| a.max(b.abs()));
        if peak > 1.5 || blk % 50 == 0 {
            println!("blk {blk} knobs {:?} peak {peak}", knob_vals);
        }
    }
}
