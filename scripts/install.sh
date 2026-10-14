#!/usr/bin/env bash
# Installs the built bundles for the current user (no sudo needed).
set -euo pipefail
cd "$(dirname "$0")/.."
B=target/bundled
mkdir -p ~/Library/Audio/Plug-Ins/VST3 ~/Library/Audio/Plug-Ins/CLAP ~/Applications
rm -rf ~/Library/Audio/Plug-Ins/VST3/"EZ-er Mix.vst3" ~/Library/Audio/Plug-Ins/CLAP/"EZ-er Mix.clap" ~/Applications/"EZ-er Mix.app"
cp -R "$B/EZ-er Mix.vst3" ~/Library/Audio/Plug-Ins/VST3/
cp -R "$B/EZ-er Mix.clap" ~/Library/Audio/Plug-Ins/CLAP/
cp -R "$B/EZ-er Mix.app" ~/Applications/
echo "Installed VST3, CLAP and ~/Applications/EZ-er Mix.app"
