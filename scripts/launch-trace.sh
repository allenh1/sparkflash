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

# Every launch of an arm's bit-record fixture, saved once and compared after.
#
# usage: scripts/launch-trace.sh <trace-dir> <crate>...
#
# For each crate the script runs its bit-record tests (every `bits_equal_the_record`, at TP 1, 2
# and 4) with SPARKFLASH_LAUNCH_TRACE on (sparkflash_kernels::launch_trace), then sorts the trace
# stably by launching thread, so a local group's ranks each read in issue order. The first run
# writes <trace-dir>/<crate>.trace.gz; a later run compares with it and prints `equal`, or
# `DIFFERS` with the first differing lines. The record tests compare their bits only when
# SPARKFLASH_BITS_RECORD names the record directory: set it (and the fixtures' own switches) as
# for the record itself. A trace with no launch is refused, never saved.

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 <trace-dir> <crate>..." >&2
    exit 2
fi
dir=$1
shift
mkdir -p "$dir"
status=0
for crate in "$@"; do
    raw="$dir/$crate.raw"
    rm -f "$raw"
    echo "== $crate"
    SPARKFLASH_LAUNCH_TRACE="$raw" timeout 3000 cargo test -q -p "$crate" --lib bits_equal_the_record \
        -- --include-ignored --nocapture --test-threads 1 2>&1 |
        grep -E "^record:|panicked|^error|test result" | head -8 || true
    if [[ ! -s "$raw" ]]; then
        echo "trace: $crate: no launch was traced" >&2
        status=1
        continue
    fi
    new="$dir/$crate.new"
    LC_ALL=C sort -s -k1,1 "$raw" >"$new"
    rm -f "$raw"
    kept="$dir/$crate.trace.gz"
    lines=$(wc -l <"$new")
    if [[ -f "$kept" ]]; then
        if gzip -dc "$kept" | cmp -s - "$new"; then
            echo "trace: $crate: $lines launches equal"
            rm -f "$new"
        else
            echo "trace: $crate: DIFFERS from $kept (new trace kept at $new)"
            diff <(gzip -dc "$kept") "$new" | head -20 || true
            status=1
        fi
    else
        gzip -c "$new" >"$kept"
        rm -f "$new"
        echo "trace: $crate: recorded $lines launches"
    fi
done
exit $status
