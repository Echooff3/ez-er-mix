//! Parses every installed preset and reports what was found.
use ezmix_content::preset::SlotKind;
use ezmix_content::Library;

fn main() {
    let lib = Library::scan();
    println!("pack dirs: {:?}", lib.paths.pack_dirs);
    let (mut ok, mut failed, mut neural, mut nam, mut nam_missing) = (0, 0, 0, 0, 0);
    for pack in &lib.packs {
        println!("{:<28} {:<24} {:>4} presets", pack.id, pack.name, pack.presets.len());
        for r in &pack.presets {
            match lib.load_preset(r) {
                Ok(p) => {
                    ok += 1;
                    if p.uses_neural() {
                        neural += 1;
                    }
                    for s in &p.slots {
                        if let SlotKind::Nam(n) = &s.kind {
                            nam += 1;
                            if lib.load_nam_model(&n.model()).is_none() {
                                nam_missing += 1;
                                eprintln!("  missing NAM model for {} ({:?} {})", p.name, n.internal_model, n.custom_model);
                            }
                        }
                    }
                }
                Err(e) => {
                    failed += 1;
                    eprintln!("  FAILED {}: {e}", r.path);
                }
            }
        }
    }
    println!("{ok} parsed, {failed} failed, {neural} with unsupported neural slots, {nam} NAM slots ({nam_missing} missing models)");
}
