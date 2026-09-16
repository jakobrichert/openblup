#!/usr/bin/env bash
# Build OpenBLUP Studio: compile the full engine to WebAssembly, generate the
# JavaScript bindings into studio/pkg and stage the example data.
#
#   rustup target add wasm32-unknown-unknown
#   cargo install wasm-bindgen-cli --version <version in Cargo.lock> --locked
#   studio/build.sh
#   python3 -m http.server --directory studio 8000
set -euo pipefail
cd "$(dirname "$0")/.."

target_dir=${CARGO_TARGET_DIR:-target}
want=$(grep -A1 '^name = "wasm-bindgen"$' Cargo.lock | sed -n 's/^version = "\(.*\)"$/\1/p')
have=$(wasm-bindgen --version 2>/dev/null | awk '{print $2}' || true)
if [ "$want" != "$have" ]; then
  echo "error: wasm-bindgen-cli $want is required (found: ${have:-none})" >&2
  echo "       cargo install wasm-bindgen-cli --version $want --locked" >&2
  exit 1
fi

cargo build -p openblup-studio --target wasm32-unknown-unknown --release

rm -rf studio/pkg
wasm-bindgen --target web --no-typescript --out-dir studio/pkg \
  "$target_dir/wasm32-unknown-unknown/release/openblup_studio.wasm"
if command -v wasm-opt >/dev/null 2>&1; then
  wasm-opt -O3 --enable-bulk-memory --enable-nontrapping-float-to-int \
    -o studio/pkg/openblup_studio_bg.wasm studio/pkg/openblup_studio_bg.wasm
fi

mkdir -p studio/examples
cp examples/*.csv studio/examples/

echo "Built studio/ ($(du -h studio/pkg/openblup_studio_bg.wasm | cut -f1) engine)."
echo "Serve it with: python3 -m http.server --directory studio 8000"
