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

# The one gate every branch runs before it is pushed.
#
# Why this exists: a gate whose failure is printed but not acted on is worse than no gate, because it
# reports "checked" while nothing was checked. The way that happens in practice is a pipeline:
#   cargo clippy ... | grep error && git commit
# which commits when grep matches. This script sets -e, -u and -o pipefail, never pipes a gate through
# anything, and exits non-zero on the first failure.
#
# Usage:
#   bash scripts/branch-gate.sh              # fmt, clippy (default + cuda), rustdoc (the CI bar),
#                                            # cli lib tests
#   bash scripts/branch-gate.sh --full       # adds the whole workspace test suite
#   bash scripts/branch-gate.sh --push       # run the gate, then push ONLY if it passed
#   bash scripts/branch-gate.sh --push-force # same, but `git push --force-with-lease` (needed after an
#                                            # amendment: a plain push is rejected, which left a green
#                                            # gate sitting on an unpushed commit)
#   SPARKFLASH_NVCC=/home/user/cuda-tk/bin/nvcc bash scripts/branch-gate.sh --push
#
# Use --push, not `bash scripts/branch-gate.sh && git push`. The chained form can itself be defeated by
# piping the script (... 2>&1 | tail -3 && git push makes the pipeline status tail's), which is exactly how
# a red commit was pushed twice on this branch. With --push the push happens inside this process, after the
# gate's own exit status is known, so no external pipe can mask it and no chain can reorder it.
#   bash scripts/branch-gate.sh --push 2>&1 | tail -20      # safe: the push is already inside
set -euo pipefail

cd "$(dirname "$0")/.."

FULL=0
PUSH=0
PUSH_FORCE=0
for arg in "$@"; do
  case "$arg" in
    --full) FULL=1 ;;
    --push) PUSH=1 ;;
    --push-force) PUSH=1; PUSH_FORCE=1 ;;
    *) echo "branch-gate: unknown argument $arg" >&2; exit 2 ;;
  esac
done

step() {
  echo
  echo "=== $* ==="
}

step "git status must be clean (the gate runs on the committed tree)"
if [ -n "$(git status --porcelain)" ]; then
  echo "branch-gate: the working tree is dirty; commit or stash first." >&2
  echo "  (a gate run on a dirty tree can pass while the commit is red - this is the exact failure" >&2
  echo "   mode that let a clippy failure be committed once already)" >&2
  git status --short >&2
  exit 1
fi

step "cargo fmt --check"
cargo fmt --check

step "cargo clippy -p sparkflash-cli --lib -- -D warnings"
cargo clippy -p sparkflash-cli --lib -- -D warnings

step "cargo clippy -p sparkflash-cli --lib --features cuda -- -D warnings"
cargo clippy -p sparkflash-cli --lib --features cuda -- -D warnings

step "RUSTDOCFLAGS=-D missing_docs cargo doc --workspace --no-deps (the rustdoc CI job, same flags)"
# This is the CI job's exact command, and it is here because its absence let a red rustdoc job reach
# the pipeline: the batch documented a new public struct without field docs and the local gate was
# green. cargo doc writes into target/doc like any other build artifact, so it needs no special setup.
RUSTDOCFLAGS="-D missing_docs" cargo doc --workspace --no-deps

step "cargo test -p sparkflash-cli --lib"
cargo test -p sparkflash-cli --lib

if [ -n "${SPARKFLASH_NVCC:-}" ]; then
  step "kernel baseline (SPARKFLASH_NVCC=${SPARKFLASH_NVCC})"
  bash crates/sparkflash-kernels/scripts/check-kernel-baseline.sh
else
  echo
  echo "note: SPARKFLASH_NVCC is unset, so the kernel baseline is not checked in this run."
  echo "  set SPARKFLASH_NVCC=/home/user/cuda-tk/bin/nvcc to include it."
fi

if [ "$FULL" -eq 1 ]; then
  step "cargo test --workspace"
  cargo test --workspace
fi

step "gate passed"

if [ "$PUSH" -eq 1 ]; then
  if [ "$PUSH_FORCE" -eq 1 ]; then
    step "git push --force-with-lease (the gate passed, so this push cannot carry a red tree)"
    git push --force-with-lease
  else
    step "git push (the gate passed, so this push cannot carry a red tree)"
    git push
  fi
  echo
  echo "=== gate passed and pushed ==="
fi
