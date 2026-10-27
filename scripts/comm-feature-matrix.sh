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

# Compile matrix for the all-reduce arm features: the crate must format, lint and test with the RDMA
# arm feature off AND on, so the ARMED shape cannot rot while nvidia-peermem is root-blocked.
#
#   scripts/comm-feature-matrix.sh
#
# The matrix is the point: a feature that only builds in the configuration the developer happens to
# use is not a designed arm, it is a rot risk. Both configurations must be green before either is
# trusted, and the typed-refusal tests pin the arm's behaviour in both.
set -euo pipefail
ROOT=$(cd "${0%/*}/.." && pwd)
CARGO=${CARGO:-$HOME/.cargo/bin/cargo}
run() {
    local label=$1
    shift
    echo "== sparkflash-comm: $label"
    "$CARGO" fmt -p sparkflash-comm --all -- --check
    "$CARGO" clippy -q -p sparkflash-comm --all-targets "$@" -- -D warnings
    "$CARGO" test -q -p sparkflash-comm --all-targets "$@"
    echo "== $label: ok"
}
run "feature off" 
run "gpu-rdma-reduce on" --features gpu-rdma-reduce
echo "comm-feature-matrix: both configurations are green"
