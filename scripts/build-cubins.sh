#!/usr/bin/env bash
# Compiles every device kernel under kernels/ into the cubin directory `sparkflash serve
# --tp4-device <dir>` loads (one <name>.cubin per kernels/<name>.cu, for sm_121a by default).
#
#   scripts/build-cubins.sh [out dir]        (default: target/cubins)
#
# Environment: NVCC (default: nvcc on PATH, then /usr/local/cuda/bin/nvcc), SPARKFLASH_CUDA_ARCH
# (default sm_121a, the DGX Spark's GB10).
set -euo pipefail
ROOT=$(cd "${0%/*}/.." && pwd)
OUT="${1:-$ROOT/target/cubins}"
ARCH="${SPARKFLASH_CUDA_ARCH:-sm_121a}"
NVCC="${NVCC:-$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)}"
[ -x "$NVCC" ] || { echo "build-cubins: nvcc not found (set NVCC)" >&2; exit 1; }
mkdir -p "$OUT"
jobs="${JOBS:-$(nproc 2>/dev/null || echo 4)}"
ls "$ROOT"/kernels/*.cu | xargs -P "$jobs" -I{} sh -c '
    name=$(basename "{}" .cu)
    "$0" -std=c++17 -arch="$1" -cubin -I "$2/kernels" "{}" -o "$3/$name.cubin" || { echo "build-cubins: $name failed" >&2; exit 255; }
' "$NVCC" "$ARCH" "$ROOT" "$OUT"
echo "build-cubins: $(ls "$OUT"/*.cubin | wc -l) cubins in $OUT"
