#!/usr/bin/env bash
# Builds release VST3 + CLAP bundles and a macOS standalone app into
# target/bundled/. Pass --universal for arm64 + x86_64 binaries (needs the
# x86_64-apple-darwin Rust target); the default is the host architecture only.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "${1:-}" == "--universal" ]]; then
  # Put rustup's toolchain first: a Homebrew Rust earlier on PATH only has the
  # host std, and the xtask runs nested cargo/rustc by name.
  PATH="$(dirname "$(rustup which cargo)"):$PATH" cargo xtask bundle-universal ez-er-mix --release
  # That also built the standalone binary for both targets (the xtask doesn't
  # bundle it: it looks for "ez_er_mix"), so combine those ourselves.
  STANDALONE=target/ez-er-mix-universal
  lipo -create -output "$STANDALONE" target/x86_64-apple-darwin/release/ez-er-mix target/aarch64-apple-darwin/release/ez-er-mix
else
  cargo xtask bundle ez-er-mix --release
  cargo build --release -p ez-er-mix --bin ez-er-mix
  STANDALONE=target/release/ez-er-mix
fi

B=target/bundled
APP="$B/EZ-er Mix.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$STANDALONE" "$APP/Contents/MacOS/EZ-er Mix"
VERSION=$(cargo metadata --no-deps --format-version 1 | python3 -c 'import json,sys; print(next(p["version"] for p in json.load(sys.stdin)["packages"] if p["name"]=="ez-er-mix"))')

# nih-plug's bundler stamps every bundle com.nih-plug.<package> and 1.0.0;
# give them our own identifiers and the workspace version.
for kind in vst3 clap; do
  plist="$B/EZ-er Mix.$kind/Contents/Info.plist"
  plutil -replace CFBundleIdentifier -string "com.trilsoft.ez-er-mix.$kind" "$plist"
  plutil -replace CFBundleVersion -string "$VERSION" "$plist"
  plutil -replace CFBundleShortVersionString -string "$VERSION" "$plist"
done

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>EZ-er Mix</string>
  <key>CFBundleDisplayName</key><string>EZ-er Mix</string>
  <key>CFBundleIdentifier</key><string>com.trilsoft.ez-er-mix</string>
  <key>CFBundleExecutable</key><string>EZ-er Mix</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleVersion</key><string>${VERSION}</string>
  <key>CFBundleShortVersionString</key><string>${VERSION}</string>
  <key>LSMinimumSystemVersion</key><string>11.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSMicrophoneUsageDescription</key><string>EZ-er Mix processes audio from your audio interface input.</string>
</dict>
</plist>
PLIST

# Editing the plists broke the bundler's ad-hoc signatures; redo them.
for bundle in "$B/EZ-er Mix.vst3" "$B/EZ-er Mix.clap" "$APP"; do
  codesign --force -s - "$bundle" >/dev/null 2>&1 || true
done
echo "Built:"
ls -1d target/bundled/*
