#!/usr/bin/env bash
# Build Vector Baba as a signed (ad-hoc) Apple Silicon .app and wrap it in a DMG.
# Usage: scripts/build_dmg.sh            -> dist/VectorBaba-<version>-arm64.dmg
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PATH="/opt/homebrew/opt/rustup/bin:$HOME/.cargo/bin:$PATH"

VERSION="$(grep -m1 '^version' Cargo.toml | sed -E 's/.*"([^"]+)".*/\1/')"
TARGET="aarch64-apple-darwin"
APP_NAME="Vector Baba"
APP_DIR="dist/$APP_NAME.app"
DMG_NAME="VectorBaba-$VERSION-arm64.dmg"

echo "==> Building release binaries ($TARGET)"
rustup target add "$TARGET" >/dev/null 2>&1 || true
cargo build --release --target "$TARGET" --bins

echo "==> Generating icon"
mkdir -p assets
"target/$TARGET/release/gen-icon" assets/icon_1024.png
ICONSET="dist/icon.iconset"
rm -rf "$ICONSET" && mkdir -p "$ICONSET"
for s in 16 32 128 256 512; do
  sips -z $s $s assets/icon_1024.png --out "$ICONSET/icon_${s}x${s}.png" >/dev/null
  d=$((s*2))
  sips -z $d $d assets/icon_1024.png --out "$ICONSET/icon_${s}x${s}@2x.png" >/dev/null
done
iconutil -c icns "$ICONSET" -o dist/icon.icns

echo "==> Assembling $APP_DIR"
rm -rf "$APP_DIR"
mkdir -p "$APP_DIR/Contents/MacOS" "$APP_DIR/Contents/Resources"
cp "target/$TARGET/release/vector-baba" "$APP_DIR/Contents/MacOS/vector-baba"
cp "target/$TARGET/release/vector-baba-cli" "$APP_DIR/Contents/MacOS/vector-baba-cli"
cp dist/icon.icns "$APP_DIR/Contents/Resources/icon.icns"
sed "s/__VERSION__/$VERSION/g" macos/Info.plist > "$APP_DIR/Contents/Info.plist"
printf 'APPL????' > "$APP_DIR/Contents/PkgInfo"

echo "==> Signing (ad-hoc)"
codesign --force --deep --sign - "$APP_DIR"
codesign --verify --verbose=2 "$APP_DIR"

echo "==> Creating DMG"
STAGE="dist/dmg-root"
rm -rf "$STAGE" && mkdir -p "$STAGE"
cp -R "$APP_DIR" "$STAGE/"
ln -s /Applications "$STAGE/Applications"
rm -f "dist/$DMG_NAME"
hdiutil create -volname "$APP_NAME" -srcfolder "$STAGE" -ov -format UDZO "dist/$DMG_NAME" >/dev/null
rm -rf "$STAGE" "$ICONSET"

echo
echo "Done: $ROOT/dist/$DMG_NAME"
ls -lh "dist/$DMG_NAME"
