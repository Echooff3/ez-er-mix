#!/usr/bin/env bash
# Builds a universal installer package into out/EZ-er-Mix-<version>-macOS.pkg
# that installs the VST3 and CLAP plug-ins into /Library/Audio/Plug-Ins and the
# standalone app into /Applications.
#
# Signing and notarization use:
#   DEVELOPER_ID_APP="Developer ID Application: <name> (<team>)"
#   DEVELOPER_ID_INSTALLER="Developer ID Installer: <name> (<team>)"
#   NOTARY_PROFILE=ez-er-mix-notary   (from xcrun notarytool store-credentials)
# Without both identities the package is unsigned and not notarized (dry run).
# ALLOW_DIRTY=1 allows building from uncommitted changes.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -z "${ALLOW_DIRTY:-}" && -n "$(git status --porcelain)" ]]; then
  echo "Uncommitted changes; commit them or set ALLOW_DIRTY=1." >&2
  exit 1
fi
SIGN=1
if [[ -z "${DEVELOPER_ID_APP:-}" || -z "${DEVELOPER_ID_INSTALLER:-}" ]]; then
  SIGN=0
  echo "WARNING: DEVELOPER_ID_APP / DEVELOPER_ID_INSTALLER not set; building an unsigned package." >&2
fi

./scripts/bundle.sh --universal

B=target/bundled
OUT=out
VERSION=$(cargo metadata --no-deps --format-version 1 | python3 -c 'import json,sys; print(next(p["version"] for p in json.load(sys.stdin)["packages"] if p["name"]=="ez-er-mix"))')
PKG="$OUT/EZ-er-Mix-$VERSION-macOS.pkg"
rm -rf "$OUT/stage" "$OUT/components" "$OUT/distribution.xml" "$PKG"
mkdir -p "$OUT/components"

# kind, bundle extension, install location
COMPONENTS=(
  "vst3 vst3 /Library/Audio/Plug-Ins/VST3"
  "clap clap /Library/Audio/Plug-Ins/CLAP"
  "app app /Applications"
)

for component in "${COMPONENTS[@]}"; do
  read -r kind ext location <<<"$component"
  bundle="$OUT/stage/$kind/EZ-er Mix.$ext"
  mkdir -p "$OUT/stage/$kind"
  ditto "$B/EZ-er Mix.$ext" "$bundle"
  # Extended attributes (e.g. com.apple.provenance) would end up in the payload.
  xattr -cr "$bundle"

  archs=$(lipo -archs "$bundle/Contents/MacOS/EZ-er Mix")
  if [[ "$archs" != *x86_64* || "$archs" != *arm64* ]]; then
    echo "$bundle is not universal (archs: $archs)" >&2
    exit 1
  fi
  if (( SIGN )); then
    entitlements=()
    [[ "$kind" == app ]] && entitlements=(--entitlements packaging/app.entitlements)
    codesign --force --options runtime --timestamp ${entitlements[@]+"${entitlements[@]}"} -s "$DEVELOPER_ID_APP" "$bundle"
  fi
  codesign --verify --strict "$bundle"

  # Install exactly where we say: don't follow a copy of the bundle found
  # elsewhere on disk, and don't skip it because an installed copy looks newer.
  plist="$OUT/components/$kind.plist"
  pkgbuild --analyze --root "$OUT/stage/$kind" "$plist" >/dev/null
  plutil -replace 0.BundleIsRelocatable -bool NO "$plist"
  plutil -replace 0.BundleIsVersionChecked -bool NO "$plist"
  pkgbuild --quiet --root "$OUT/stage/$kind" --component-plist "$plist" \
    --identifier "com.trilsoft.ez-er-mix.pkg.$kind" --version "$VERSION" \
    --install-location "$location" "$OUT/components/$kind.pkg"
done

cat > "$OUT/distribution.xml" <<XML
<?xml version="1.0" encoding="utf-8"?>
<installer-gui-script minSpecVersion="2">
  <title>EZ-er Mix $VERSION</title>
  <product id="com.trilsoft.ez-er-mix" version="$VERSION"/>
  <welcome file="welcome.html" mime-type="text/html"/>
  <options customize="allow" require-scripts="false" hostArchitectures="x86_64,arm64"/>
  <domains enable_anywhere="false" enable_currentUserHome="false" enable_localSystem="true"/>
  <allowed-os-versions><os-version min="11.0"/></allowed-os-versions>
  <choices-outline>
    <line choice="vst3"/>
    <line choice="clap"/>
    <line choice="app"/>
  </choices-outline>
  <choice id="vst3" title="VST3 plug-in" description="EZ-er Mix.vst3 in /Library/Audio/Plug-Ins/VST3">
    <pkg-ref id="com.trilsoft.ez-er-mix.pkg.vst3"/>
  </choice>
  <choice id="clap" title="CLAP plug-in" description="EZ-er Mix.clap in /Library/Audio/Plug-Ins/CLAP">
    <pkg-ref id="com.trilsoft.ez-er-mix.pkg.clap"/>
  </choice>
  <choice id="app" title="Standalone app" description="EZ-er Mix.app in /Applications">
    <pkg-ref id="com.trilsoft.ez-er-mix.pkg.app"/>
  </choice>
  <pkg-ref id="com.trilsoft.ez-er-mix.pkg.vst3" version="$VERSION" onConclusion="none">vst3.pkg</pkg-ref>
  <pkg-ref id="com.trilsoft.ez-er-mix.pkg.clap" version="$VERSION" onConclusion="none">clap.pkg</pkg-ref>
  <pkg-ref id="com.trilsoft.ez-er-mix.pkg.app" version="$VERSION" onConclusion="none">app.pkg</pkg-ref>
</installer-gui-script>
XML

sign_args=()
(( SIGN )) && sign_args=(--sign "$DEVELOPER_ID_INSTALLER" --timestamp)
productbuild --distribution "$OUT/distribution.xml" --resources packaging/resources \
  --package-path "$OUT/components" ${sign_args[@]+"${sign_args[@]}"} "$PKG"

if (( SIGN )); then
  PROFILE=${NOTARY_PROFILE:-ez-er-mix-notary}
  xcrun notarytool submit "$PKG" --keychain-profile "$PROFILE" --wait --output-format json > "$OUT/notary.json"
  read -r ID STATUS < <(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["id"], d["status"])' "$OUT/notary.json")
  if [[ "$STATUS" != Accepted ]]; then
    echo "Notarization $STATUS:" >&2
    xcrun notarytool log "$ID" --keychain-profile "$PROFILE" >&2
    exit 1
  fi
  xcrun stapler staple "$PKG"
  spctl --assess -vv --type install "$PKG"
fi

echo "Built $PKG"
shasum -a 256 "$PKG"
