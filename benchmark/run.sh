#!/usr/bin/env bash
# Build and run the CLIc C++ benchmark, then run the clic-rs Rust benchmark.
#
# Usage:  bash benchmark/run.sh [clic_build_dir]
#
# clic_build_dir defaults to /tmp/clic_build (where CLAUDE.md says to build CLIc).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BUILD_DIR="${1:-/tmp/clic_build}"

# ── Locate required paths ─────────────────────────────────────────────────────

CLIC_LIB="$BUILD_DIR/clic/libCLIc.dylib"
CLIC_INCLUDE="$REPO_ROOT/CLIc/clic/include"
EIGEN_INCLUDE="$BUILD_DIR/_deps/eigen-src"
KERNEL_INCLUDE="$BUILD_DIR/_deps/clekernels-src/kernels"   # generated kernel headers
CLIC_BUILD_INCLUDE="$BUILD_DIR/clic/include"               # generated clic.hpp etc.
VKFFT_INCLUDE="$BUILD_DIR/_deps/vkfft-src/vkFFT"

if [[ ! -f "$CLIC_LIB" ]]; then
    echo "error: $CLIC_LIB not found."
    echo "Build CLIc first:"
    echo "  cmake --preset macos-make-release -B $BUILD_DIR"
    echo "  cmake --build $BUILD_DIR --parallel 4"
    exit 1
fi

# ── Build C++ benchmark ───────────────────────────────────────────────────────

OUT="$BUILD_DIR/clic_bench"
echo "=== Building C++ benchmark ==="
c++ -std=c++17 -O2 \
    -I"$CLIC_INCLUDE" \
    -I"$CLIC_BUILD_INCLUDE" \
    -I"$EIGEN_INCLUDE" \
    -I"$KERNEL_INCLUDE" \
    -I"$VKFFT_INCLUDE" \
    "$SCRIPT_DIR/clic_bench.cpp" \
    -L"$BUILD_DIR/clic" -lCLIc \
    -Wl,-rpath,"$BUILD_DIR/clic" \
    -framework OpenCL \
    -o "$OUT"

echo ""
echo "=== CLIc (C++) ==="
# Run each benchmark in its own process so a GPU hang in one cannot crash the next.
DEVICE_NAME=$("$OUT" gaussian_blur 2>/dev/null | grep "^Device:" || true)
[[ -n "$DEVICE_NAME" ]] && echo "$DEVICE_NAME"
echo ""
for bench in gaussian_blur add_images_weighted mean_of_all_pixels push_pull; do
    set +e
    "$OUT" "$bench" 2>/dev/null
    [[ $? -ne 0 ]] && echo "  $bench: GPU error (skipped)"
    set -e
    echo ""
done

# ── Run Rust benchmark ────────────────────────────────────────────────────────

echo ""
echo "=== clic-rs (Rust) ==="
cd "$REPO_ROOT"
cargo bench --bench gpu 2>/dev/null \
    | awk '
        /^(gaussian_blur|add_images_weighted|mean_of_all_pixels|push_pull)\// {
            # name only on this line (long name, time follows)
            if ($0 ~ /time:/) { printf "  %s\n", $0 }
            else { name = $1 }
            next
        }
        /^[[:space:]]+time:/ && name != "" {
            printf "  %-42s %s\n", name, $0
            name = ""
        }
    '
