#!/usr/bin/env bash
# build tools/bin (zstd): download zstd 1.4.8 source and make.
# evidence: old tools/bin/zstd reports "v1.4.8" (zstd --version).
set -euo pipefail
set -x
trap 'echo "FAILED: $BASH_COMMAND rc=$?"' ERR
T=$CYCLESCOPE_ARTIFACT_ROOT/tools/bin
rm -rf "$T"
mkdir -p "$T" "$T/src"
cd "$T/src"
URL="https://github.com/facebook/zstd/releases/download/v1.4.8/zstd-1.4.8.tar.gz"
# fallback: codeload tarball of the same tag (bit-identical repo tree, gzip
# stream differs -> sha256 differs; both recorded in capture_tools.json note)
FB="https://codeload.github.com/facebook/zstd/tar.gz/refs/tags/v1.4.8"
curl -fL --retry 2 "$URL" -o zstd-1.4.8.tar.gz   || curl -fL --retry 2 "$FB" -o zstd-1.4.8.tar.gz
tar xzf zstd-1.4.8.tar.gz
make -C zstd-1.4.8 -j8 zstd
cp zstd-1.4.8/programs/zstd "$T/zstd"
chmod +x "$T/zstd"
