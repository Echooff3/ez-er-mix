//! Dumps a preset's structure: `cargo run --example inspect -- "<path substring>"`.
use ezmix_content::engine::ParamValue;
use ezmix_content::preset::SlotKind;
use ezmix_content::Library;

fn main() {
    let needle = std::env::args().nth(1).expect("path substring");
    let lib = Library::scan();
    for pack in &lib.packs {
        for r in pack.presets.iter().filter(|p| p.path.contains(&needle)) {
            let p = lib.load_preset(r).unwrap();
            println!("== {} :: {} (format {:?}) output_gain {}", pack.id, r.path, p.format, p.output_gain);
            for (si, s) in p.slots.iter().enumerate() {
                match &s.kind {
                    SlotKind::Engine(e) => {
                        println!(" slot {si}: engine v{} active {} routing {:?}", e.format_version, s.active, e.routing);
                        for (ci, ch) in e.channels.iter().enumerate() {
                            for (k, fx) in ch.slots.iter().enumerate().filter(|(_, f)| !f.is_dummy()) {
                                let ps: Vec<String> = fx
                                    .params
                                    .iter()
                                    .map(|(n, v)| match v {
                                        ParamValue::Num(x) => format!("{n}={x}"),
                                        ParamValue::Sync { unsynced, synced, sync } => format!("{n}={unsynced}/{synced}/{sync}"),
                                        ParamValue::Data(d) => format!("{n}=<{} bytes>", d.len()),
                                        ParamValue::Str(s) => format!("{n}={s:?}"),
                                    })
                                    .collect();
                                println!("   ch{ci} s{k} {} power={} {}", fx.id, fx.power, ps.join(" "));
                            }
                        }
                        for (i, x) in e.xfaders.iter().enumerate().filter(|(_, x)| !x.interpolations.is_empty()) {
                            println!("   xfader {i} {:?} = {}", x.label, x.value);
                            for it in &x.interpolations {
                                println!("      {:?} {} [{} .. {}] curve {}", it.target, it.label, it.min, it.max, it.curve);
                            }
                        }
                    }
                    SlotKind::Nam(n) => println!(" slot {si}: NAM {:?}", n),
                    SlotKind::Neural { model } => println!(" slot {si}: neural {model}"),
                    SlotKind::Unknown(k) => println!(" slot {si}: unknown type {k}"),
                }
            }
            for (i, k) in p.knobs.iter().enumerate() {
                println!(" knob {i} {:?} = {} button={} targets {:?}", k.name, k.value, k.is_button, k.targets);
            }
        }
    }
}
