# EZ-er Mix

A Rust re-implementation of the idea behind Toontrack EZmix 3: pick a preset,
turn a few macro knobs. It loads the EZmix preset packs already installed on
this Mac (no copies, no conversion) and renders them with its own DSP engine.
It ships as a **VST3** and **CLAP** plugin plus a **standalone app**.

Like EZmix 3's rack, presets can be stacked: each tab above the knobs is one
effect in a serial chain (left to right). "+" adds a tab, the dot on a tab
turns that effect on or off, × removes it, and tabs can be dragged (or moved
from the right-click menu) to reorder the chain. The browser and the ◀ ▶
buttons load into the selected tab. Up to 8 slots; each has its own host
parameters ("Slot N On", "Slot N Macro 1–8") that stay with the effect when it
is moved. Switching presets and reordering crossfade instead of clicking.

Deliberately left out: the AI features (Mix Assistant / preset space, and the
TFLite "AI console" stages used by 88 presets — those stages are bypassed and
the UI says so). Neural Amp Modeler captures (174 presets) are supported.

Your own Neural Amp Modeler captures and cabinet IRs load too: `.nam` files
(A1 WaveNets and TONE3000's A2 captures) and `.wav` impulse responses in
`~/Music/NAM` or `~/Library/Application Support/EZ-er Mix/Captures` (add
folders with `EZERMIX_CAPTURES=dir1:dir2`) show up as the "NAM Captures" and
"Cab IRs" packs, sub-folders included. Captures get Gain, Gate, Bass, Mid,
Treble, Level and Width; IRs get Level, Low cut and High cut. For an amp head
capture, put a cab IR in the next tab. On a first start the rack is TONE3000's
[EVH 5150 Iconic Chainsaw](https://www.tone3000.com/tones/evh-5150-iconic-chainsaw-45921)
("5150 MAX Chainsaw") into an `EVH 5150 III.wav` cab IR, if those files are
in a capture folder. LSTM models and NAM's FiLM / condition-DSP options are
not supported (the UI says so).

## Build and install

```sh
./scripts/bundle.sh    # target/bundled/EZ-er Mix.{vst3,clap,app}
./scripts/install.sh   # ~/Library/Audio/Plug-Ins/{VST3,CLAP}, ~/Applications
```

Standalone: `EZ-er Mix.app` uses the default output device at its native rate
and the default input device (except built-in microphones, to avoid feedback).
From a terminal: `target/release/ez-er-mix --input-device "<name>"`.
The standalone reopens the last rack. `EZERMIX_PRESET="EMX3_Core3:/Presets/68 Plexi Crunch"`
preselects a preset; separate several with `;` to start with a rack. Captures
use the pack ids `NAM` and `IR` with paths relative to their capture folder,
e.g. `NAM:EVH 5150 Iconic Chainsaw/5150  MAX Chainsaw.nam`.

Tools:

```sh
cargo run --release -p ezmix-content --example scan        # parse every preset
cargo run --release -p ezmix-content --example inspect -- "Plexi Crunch"
cargo run --release -p ezmix-dsp --example render_all     # render + sanity-check all presets
cargo test --workspace                                     # NAM vs. reference, rack audio, headless UI
```

## Release

`./scripts/release.sh` builds a universal (arm64 + x86_64) installer,
`out/EZ-er-Mix-<version>-macOS.pkg`, that installs the VST3 and CLAP plug-ins
into `/Library/Audio/Plug-Ins/{VST3,CLAP}` and the app into `/Applications`
(each can be deselected under Customize). It refuses to build from uncommitted
changes unless `ALLOW_DIRTY=1`.

One-time setup:

* `rustup target add x86_64-apple-darwin` (universal builds use rustup's
  toolchain even if a Homebrew Rust comes first on `PATH`).
* For a signed, notarized package (Apple Developer Program): create
  *Developer ID Application* and *Developer ID Installer* certificates
  (Xcode → Settings → Accounts → Manage Certificates), then store notary
  credentials with an app-specific password:
  `xcrun notarytool store-credentials ez-er-mix-notary --apple-id <id> --team-id <team>`.

```sh
export DEVELOPER_ID_APP="Developer ID Application: <name> (<team>)"
export DEVELOPER_ID_INSTALLER="Developer ID Installer: <name> (<team>)"
./scripts/release.sh   # signs, notarizes and staples; NOTARY_PROFILE overrides the profile name
```

Without the two `DEVELOPER_ID_*` variables the package is built unsigned, and
Gatekeeper makes users approve it in System Settings → Privacy & Security.
Remove copies left in `~/Library/Audio/Plug-Ins` by `install.sh` before
installing the package, or DAWs list the plug-in twice.

To uninstall, delete the three installed bundles, then:
`sudo pkgutil --forget com.trilsoft.ez-er-mix.pkg.vst3` (and `.clap`, `.app`).

## Layout

| Crate | Role |
|---|---|
| `crates/ezmix-content` | Pack discovery, `ttfs` container reader, preset parsers (all three generations), parameter range table, capture folders + WAV reader |
| `crates/ezmix-dsp` | Effect engine: ~95 effect types, routing graph, NAM WaveNet runtime (A1 + A2), partitioned convolution, IR decoder, cab IR slot |
| `crates/ezmix-plugin` | nih-plug plugin (preset rack) + egui editor + standalone entry point |

## What was reverse engineered

Everything below was recovered from the installed packs and the EZmix 3
binary (VST3, arm64 slice); see the code for exact details.

**Pack container (`ttfs`)** — `"ttfs"`, `u16` at 0x06 = root directory offset
− 0x10, a format tag (`EMX`, `EMX2`, `EMX3`) and an opaque signature block,
then directory tables: `u32 count` + entries of `u32 offset, u32 size, name\0`
(size 0 = sub-directory). File payloads are plain. Pack paths come from
`/Users/Shared/Toontrack/Preferences/com.toontrack.ezmix3.plist`
(`PresetPackPath`, `ResourcePath`).

**Presets** — three generations:
* EZmix 3: a brace-structured text file. `MixerChannel` holds a chain of
  effects: type 8 = multi-channel engine (base64 XML), 9 = NAM capture
  (key/value text), 10 = TFLite console (skipped). `PresetKnobs` maps each
  macro knob to `(slot = ix2, parameter = ix3, range lo hi curve)`; the knob
  curve is `t = x + c·x·(x−1)`.
* Engine XML (format 1, 4/6, 8): channels IN, AUX1–4 and OUT with five slots
  each; `TA_GLOBAL` holds volumes, pans, mute/solo/phase, sends with a tap
  point (`split`), multiband mode; nine `xfader` macros interpolate effect
  parameters: `n = min + x^(10^curve)·(max − min)`, `n` normalized.
* Normalized → real parameter ranges/tapers were fitted from (fader, value)
  pairs across all 1,384 presets (`data/param_ranges.json`); send levels are
  −40…0 dB and channel volumes −36…+12 dB, with 0 meaning off.

**NAM slot** — internal model index 1–18 maps to
`<ResourcePath>/Models` → `/NeuralAmpModeler/Amps/<name>.nam`; 20 means a pack
model in `/NAMModels/`. Input −20…+20 dB, output −40…+40 dB, gate −100…0 dB,
tone stack bass ±20 / mid ±15 / treble ±10 dB (ranges read from the wrapper's
constructor).

**Reverb IR codec** (`TA_REVIR`, decoder at 0x763550 in the binary) — header
`u32 channels, u32 frames, f32 peak`; word-aligned 512-sample blocks,
LSB-first bitstream: 9-bit length in words, two predictor coefficients,
3 × 24-bit warm-up samples, 16 groups of 32 residuals (5-bit shift + prefix
coded quotient −8…7 + low bits), 3rd-order prediction in 1/16 units, scale
2⁻²⁰; stereo = mid/side block pairs. The effect mixes wet at `0.08 × mix`.
Cab IRs (`TA_CABIR`) are raw float32; matching-EQ profiles are two 4097-bin
spectra.

## How faithful is it

Preset structure, routing, macro behaviour, IRs and NAM captures are
reproduced from the original data. The WaveNet runtime matches the output of
NAM core's reference renderer to float precision (A1 and A2 captures; a test
pins this with a small synthetic model). The ~95 Overloud-derived effect algorithms
(compressors, EQs, amps, reverbs, …) are clean re-implementations built
against the same parameters — they behave as intended but are not
sample-identical to EZmix. Known gaps:

* `TA_SAMPLE_PLAYER` (one preset) is bypassed.
* Sidechain inputs and the global LFO modulation used by ~35 presets are not
  implemented.
* Matching-EQ shape and reverb-IR length are applied when a preset loads;
  moving those knobs later only scales the wet level.
* Two EZmix 1-era presets ("V Line Bass", "G Kick") stack ~27 dB of EQ and
  will clip hot sources — as designed by the preset.
