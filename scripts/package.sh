#!/usr/bin/env bash
# Package release binaries into dist/prata-v<version>-<target>.tar.gz (+ .sha256).
# Used by .github/workflows/release.yml and for local testing of the npm launcher.
#
#   scripts/package.sh <target> [version] [bin-dir]
#     target   darwin-arm64 | darwin-x64 | linux-x64
#     version  defaults to the workspace version in Cargo.toml
#     bin-dir  defaults to target/release
#
# The archive is flat: prata, prata-web, transcribe.py, requirements.txt, LICENSE, NOTICE, README.md
set -euo pipefail
cd "$(dirname "$0")/.."
TARGET=${1:?usage: package.sh <target> [version] [bin-dir]}
VERSION=${2:-$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -1)}
BIN_DIR=${3:-target/release}
NAME="prata-v${VERSION}-${TARGET}"
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT

for b in prata prata-web; do
  [ -x "$BIN_DIR/$b" ] || { echo "missing $BIN_DIR/$b – run cargo build --release first" >&2; exit 1; }
  cp "$BIN_DIR/$b" "$STAGE/"
done
cp python/transcribe.py python/requirements.txt LICENSE NOTICE README.md "$STAGE/"
chmod 755 "$STAGE/prata" "$STAGE/prata-web"
if command -v strip >/dev/null; then strip "$STAGE/prata" "$STAGE/prata-web" 2>/dev/null || true; fi

mkdir -p dist
# Reproducible-ish: fixed owner, sorted entries
tar -C "$STAGE" -czf "dist/${NAME}.tar.gz" --owner=0 --group=0 --numeric-owner 2>/dev/null . \
  || tar -C "$STAGE" -czf "dist/${NAME}.tar.gz" .     # bsdtar (macOS) has no --owner
( cd dist && { sha256sum "${NAME}.tar.gz" 2>/dev/null || shasum -a 256 "${NAME}.tar.gz"; } > "${NAME}.tar.gz.sha256" )
echo "dist/${NAME}.tar.gz"
