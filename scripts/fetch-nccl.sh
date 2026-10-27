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

# Vendors libnccl.so.2 for aarch64/CUDA 13 into a deps directory - no system install, no root.
#
#   scripts/fetch-nccl.sh [deps dir]
#
# The four nodes have no system libnccl (host ldconfig is empty; the library only exists inside the
# deployed vllm-node image, NCCL 2.28.3). The layer port therefore ships its own copy: the wheel is
# pinned by version and SHA-256, the extracted library is checked to be aarch64 and to carry the
# NCCL 2.x SONAME, and the result lives under target/deps/nccl (gitignored, never committed - it is
# a 346 MB artifact) where scripts/gpu-reduce-bench.sh stages it from.
#
# Two pins, because there are two artifacts: WHEEL_SHA256 is the nvidia-nccl-cu13 2.32.3 aarch64
# wheel published on PyPI, and LIB_SHA256 is the nvidia/nccl/lib/libnccl.so.2 extracted from it (the
# file the nodes actually dlopen). Both are checked.
#
# PIN HISTORY, because a mismatch here must never be silenced by deleting the check. On 2026-10-22 the
# single pin was c77f5ced2c4b538aad2b0ab88a9838e9eac9a68e22017863ffb0cff9c4f58745 and it was checked
# against the WHEEL - so a good download failed. Downloaded and verified: the wheel is 305,107,282 B
# with sha256 a5bee92b...724c; the libnccl.so.2 inside it is 346,523,448 B with sha256 c77f5ced...8745
# (the old pin was the extracted library, the right artifact under the wrong check). Neither the URL
# nor the version was wrong. The checks stay mandatory: on a future mismatch, download the pinned URL,
# verify version/arch/SONAME and the printed shas, then correct the pin with the evidence - never
# delete the check.
set -euo pipefail
ROOT=$(cd "${0%/*}/.." && pwd)
DEPS="${1:-$ROOT/target/deps/nccl}"
VERSION=2.32.3
WHEEL="nvidia_nccl_cu13-${VERSION}-py3-none-manylinux_2_27_aarch64.whl"
WHEEL_SHA256=a5bee92b2f4af218c109f8d221c3c9adcc752b94ae0ffcb0ed5abf9341a7724c
LIB_SHA256=c77f5ced2c4b538aad2b0ab88a9838e9eac9a68e22017863ffb0cff9c4f58745
URL="https://files.pythonhosted.org/packages/58/0a/c29c302036a06d27dd732588f3fd8ee89b1f7b0087e38c668ae7be8ff7b2/$WHEEL"

mkdir -p "$DEPS"
if [ -s "$DEPS/libnccl.so.2" ] && [ -s "$DEPS/.version" ] && [ "$(cat "$DEPS/.version")" = "$VERSION" ]; then
    echo "nccl: $DEPS/libnccl.so.2 already present (version $VERSION)"
    exit 0
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
echo "nccl: downloading $WHEEL"
curl -sSfL -o "$tmp/nccl.whl" "$URL"
echo "$WHEEL_SHA256  $tmp/nccl.whl" | sha256sum -c - >/dev/null
unzip -q -o "$tmp/nccl.whl" 'nvidia/nccl/lib/libnccl.so.2' -d "$tmp/x"
install -m 0644 "$tmp/x/nvidia/nccl/lib/libnccl.so.2" "$DEPS/libnccl.so.2"
printf '%s\n' "$VERSION" > "$DEPS/.version"
# The extract is checked, not assumed: the wrong bytes, architecture or SONAME is a refusal.
echo "$LIB_SHA256  $DEPS/libnccl.so.2" | sha256sum -c - >/dev/null || {
    echo "nccl: extracted libnccl.so.2 does not match LIB_SHA256 (expected $LIB_SHA256)" >&2
    exit 1
}
file "$DEPS/libnccl.so.2" | grep -q 'ARM aarch64' || { echo "nccl: extracted library is not aarch64" >&2; exit 1; }
readelf -d "$DEPS/libnccl.so.2" | grep -q 'SONAME.*libnccl.so.2' || { echo "nccl: extracted library has no libnccl.so.2 SONAME" >&2; exit 1; }
echo "nccl: $DEPS/libnccl.so.2 version $VERSION sha256 $(sha256sum "$DEPS/libnccl.so.2" | cut -d' ' -f1)"
