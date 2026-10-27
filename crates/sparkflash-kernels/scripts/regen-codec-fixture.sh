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

# Regenerate the exhaustive bf16 -> FP8/FP4 codec fixture with the real CUDA toolkit and diff it
# against the committed one.
#
# The fixture (tests/fixtures/m4_codec_cuda_12_9.hex) is the ground truth the codec tests compare
# against, so its provenance has to be reproducible rather than a claim: this script runs
# tests/fixtures/m4_codec_gen.cu under nvcc, writes the table to a temporary directory and fails if
# the result differs from the committed bytes. A CI job on a CUDA host can run it as-is; on a host
# without nvcc, run it inside the CUDA container with the repo bound in, for example:
#
#   docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/w" -w /w \
#       nvidia/cuda:12.9.1-devel-ubuntu22.04 \
#       /bin/sh crates/sparkflash-kernels/scripts/regen-codec-fixture.sh
#
# The generator's own header lines are part of the fixture, so they are deliberately unchanged.
set -eu

crate_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
fixture="$crate_dir/tests/fixtures/m4_codec_cuda_12_9.hex"
generator="$crate_dir/tests/fixtures/m4_codec_gen.cu"

if ! command -v nvcc >/dev/null 2>&1; then
    echo "nvcc is not on PATH: run this script on a CUDA host or inside the CUDA container" >&2
    exit 127
fi

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

nvcc -arch=sm_121a -o "$work/m4_codec_gen" "$generator"
"$work/m4_codec_gen" "$work/m4_codec.hex"

if diff -u "$fixture" "$work/m4_codec.hex"; then
    echo "codec fixture matches nvcc output byte for byte ($(wc -c < "$fixture") bytes)"
else
    echo "FAIL: the committed codec fixture does not match what nvcc produces" >&2
    echo "regenerate it only with a reviewed toolkit change, never to silence this check" >&2
    exit 1
fi
