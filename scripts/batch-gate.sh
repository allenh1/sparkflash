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

# The Lead's batch gate sweep: the one list that must pass before a batch branch is pushed.
#
# Every command here failed at least once in this project's history or is a frozen invariant. The
# rustdoc line in particular exists because a batch once passed fmt/check/clippy/tests and was still
# unmergeable: the new public API had no doc comments, and `cargo doc` without -D missing_docs only
# warns. A gate list that lives in an agent's memory gets edited by accident; this one is in the repo.
#
# Usage: scripts/batch-gate.sh [--fast]
#   --fast skips the two slowest entries (workspace tests twice, feature pairs) - never use it for a push.
#
# The container step is NOT here because it needs docker and the pinned image; run it separately:
#   docker run --rm -v "$PWD":/w -w /w nvidia/cuda:13.0.3-devel-ubuntu22.04 \
#     sh -c 'SPARKFLASH_NVCC=/usr/local/cuda/bin/nvcc sh crates/sparkflash-kernels/scripts/check-kernel-baseline.sh'
set -eu

fast=0
[ "${1:-}" = "--fast" ] && fast=1

run() {
    printf '%s ... ' "$*"
    if "$@" >/tmp/batch-gate.log 2>&1; then
        printf 'ok\n'
    else
        printf 'FAIL (rc=%s)\n' "$?"
        tail -40 /tmp/batch-gate.log
        exit 1
    fi
}

run cargo fmt --all -- --check
run cargo check --workspace --all-targets
run cargo clippy --workspace --all-targets -- -D warnings
# The line that caught the missing docs in batch 8. Keep it before the slow tests.
run env RUSTDOCFLAGS=-Dmissing_docs cargo doc --workspace --no-deps
run sh ci/check-m4-abi-size.sh
run python3 ci/verify-m4-abi-split.py
run cargo test -p sparkflash-kernels
run cargo test -p sparkflash-kv
run cargo test -p sparkflash-device
run cargo test -p sparkflash-engram
run cargo test -p sparkflash-engine
run python3 -m unittest discover -s scripts/tests

if [ "$fast" -eq 0 ]; then
    run cargo clippy -p sparkflash-cli --all-targets --features cuda -- -D warnings
    run cargo test -p sparkflash-cli --features cuda
    run cargo clippy -p sparkflash-cli --all-targets --features rdma -- -D warnings
    run cargo test -p sparkflash-cli --features rdma
    run cargo clippy -p sparkflash-comm --all-targets --features rdma -- -D warnings
    run cargo test -p sparkflash-comm --features rdma
    run python3 ci/check_features.py
    run cargo test --workspace
fi

printf 'batch gate: all commands passed\n'
