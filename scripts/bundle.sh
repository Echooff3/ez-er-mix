#!/usr/bin/env bash
# Builds release VST3 + CLAP bundles and a macOS standalone app into
# target/bundled/.
set -euo pipefail
cd "$(dirname "$0")/.."

cargo xtask bundle ez-er-mix --release
cargo build --release -p ez-er-mix --bin ez-er-mix

APP="target/bundled/EZ-er Mix.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp target/release/ez-er-mix "$APP/Contents/MacOS/EZ-er Mix"
VERSION=$(cargo metadata --no-deps --format-version 1 | python3 -c 'import json,sys; print(next(p["version"] for p in json.load(sys.stdin)["packages"] if p["name"]=="ez-er-mix"))')
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>EZ-er Mix</string>
  <key>CFBundleDisplayName</key><string>EZ-er Mix</string>
  <key>CFBundleIdentifier</key><string>com.ez-er-mix.standalone</string>
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
codesign --force --deep -s - "$APP" >/dev/null 2>&1 || true
echo "Built:"
ls -1d target/bundled/*
