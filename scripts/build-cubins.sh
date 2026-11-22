#!/usr/bin/env bash
# Copyright (c) 2026 Hunter L. Allen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
python3 "$ROOT/scripts/instances.py" --check >/dev/null

# Sources with instances are compiled from kernels/instances.toml: one compile per (source, cubin)
# group with that group's shape macros. A source with a second-shape proof instance therefore
# builds twice, into two cubins, and the released cubin keeps its own symbol set. Every other
# source is compiled exactly as before, once, with no defines.
plan=$(mktemp)
trap 'rm -f "$plan"' EXIT
python3 "$ROOT/scripts/instances.py" --cubin-plan > "$plan"
instanced=$(python3 "$ROOT/scripts/instances.py" --sources | sed "s|^|$ROOT/|")

while IFS=$'\t' read -r cubin source defines; do
    [ -n "$cubin" ] || continue
    # The defines are one flag per shape and never contain whitespace; the split is intended.
    # shellcheck disable=SC2086
    "$NVCC" -std=c++17 -arch="$ARCH" -cubin -I "$ROOT/kernels" $defines "$ROOT/$source" \
        -o "$OUT/$cubin.cubin" || { echo "build-cubins: $cubin failed" >&2; exit 255; }
done < "$plan"

for source in "$ROOT"/kernels/*.cu; do
    case $'\n'"$instanced"$'\n' in
        *$'\n'"$source"$'\n'*) continue ;;
    esac
    name=$(basename "$source" .cu)
    "$NVCC" -std=c++17 -arch="$ARCH" -cubin -I "$ROOT/kernels" "$source" \
        -o "$OUT/$name.cubin" || { echo "build-cubins: $name failed" >&2; exit 255; }
done
echo "build-cubins: $(ls "$OUT"/*.cubin | wc -l) cubins in $OUT"
