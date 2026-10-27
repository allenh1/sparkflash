#!/bin/sh
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

# Syntax-checks the kernel sources that are not yet in kernels/ (pending_kernels/*.cu), with the
# same clang CUDA frontend invocation build.rs uses for the shipped set.
#
# WHY THIS EXISTS. A kernel in pending_kernels/ is not in build.rs's PROBED_KERNELS and has no
# baseline entry, because the baseline records an nvcc cubin sha256 that a host without nvcc cannot
# produce. That must not mean it goes unchecked: this script runs the identical stub-header,
# device-only syntax check, so a CPU-only host still type-checks the body, the ptxas-specific parts
# excluded exactly as for a landed kernel on the same host. Run it with nvcc available and the
# baseline tools instead when moving the source into kernels/.
#
# Usage: scripts/check-pending-kernels.sh [clang]
#   $SPARKFLASH_CLANG overrides the clang to run; $SPARKFLASH_CUDA_ARCH overrides the arch.
set -eu

crate_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
clang=${1:-${SPARKFLASH_CLANG:-clang}}
arch=${SPARKFLASH_CUDA_ARCH:-sm_121a}

if ! command -v "$clang" >/dev/null 2>&1; then
    echo "no clang ($clang) on PATH: the pending kernels cannot be syntax-checked here" >&2
    exit 127
fi
if [ ! -d "$crate_dir/pending_kernels" ]; then
    echo "no pending_kernels directory: nothing to check"
    exit 0
fi

status=0
for source in "$crate_dir"/pending_kernels/*.cu; do
    [ -f "$source" ] || continue
    if "$clang" -x cuda --cuda-device-only "--cuda-gpu-arch=$arch" -nocudainc -nocudalib \
        -std=c++17 -fsyntax-only -DSPARKFLASH_CUDA_SYNTAX_STUB=1 \
        -I "$crate_dir/../../kernels" -I "$crate_dir/cuda_stub" "$source"; then
        echo "ok   $(basename "$source")"
    else
        echo "FAIL $(basename "$source")" >&2
        status=1
    fi
done
exit "$status"
