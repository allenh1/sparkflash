#!/usr/bin/env python3
"""Compile and test every non-default cargo feature in the workspace.

`cargo test --workspace` builds only the default feature sets, so code behind a non-default
feature is never compiled by any job in this pipeline unless a job names it explicitly.
Milestone 2's io_uring ring lives behind `sparkflash-engram/io-uring`, which no crate in the
workspace enables: the ring and its tests were invisible to every green run for six review
rounds, and the defect review round 3 found (accepted requests orphaned in the submission
queue when a completion was lost) had survived all of them because nothing ever compiled that
file. Milestone 3's `sparkflash-comm/rdma` was in the same position.

This script discovers the features rather than listing them, runs `cargo test` and
`cargo clippy --all-targets -- -D warnings` for each, and fails when a feature appears that is
not in one of the two lists below. A new feature therefore forces an explicit decision:
either this pipeline compiles it, or somebody writes down here why it cannot be compiled in a
container with no GPU, no RDMA fabric and no checkpoint.
"""

import json
import subprocess
import sys

# Features that need hardware this pipeline does not have. The cuda-abi job compiles CUDA in
# an image that does have a toolkit, so this one is covered elsewhere in the pipeline.
HARDWARE_ONLY = {
    ("sparkflash-kernels", "cuda"): "needs a toolkit and a target; the cuda-abi job compiles it",
}


def is_dependency_toggle(name, values):
    """True for the implicit feature cargo generates for an optional dependency."""
    return values == [f"dep:{name}"]


def workspace_features():
    out = subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"],
        capture_output=True,
        text=True,
        check=True,
    )
    found = []
    for package in json.loads(out.stdout)["packages"]:
        features = package.get("features", {})
        defaults = set(features.get("default", []))
        for name, values in features.items():
            if name == "default" or name in defaults:
                continue
            if is_dependency_toggle(name, values):
                continue
            found.append((package["name"], name))
    return sorted(found)


def run(command):
    print("$ " + " ".join(command), flush=True)
    return subprocess.run(command).returncode


def main():
    found = workspace_features()
    print("non-default, non-dependency features in this workspace:")
    for package, feature in found:
        hardware = HARDWARE_ONLY.get((package, feature))
        print(f"  {package}/{feature}" + (f"  [not compiled here: {hardware}]" if hardware else ""))

    dead = sorted(set(HARDWARE_ONLY) - set(found))
    if dead:
        print(
            "these features are classified as hardware-only but no longer exist: "
            + ", ".join(f"{p}/{f}" for p, f in dead),
            file=sys.stderr,
        )
        return 1

    failures = 0
    for package, feature in found:
        if (package, feature) in HARDWARE_ONLY:
            continue
        failures += run(["cargo", "test", "-p", package, "--features", feature]) != 0
        failures += run(
            ["cargo", "clippy", "-p", package, "--all-targets", "--features", feature, "--", "-D", "warnings"]
        ) != 0

    if failures:
        print(f"{failures} feature-scoped command(s) failed", file=sys.stderr)
        return 1
    print("every non-default feature compiles and passes its tests")
    return 0


if __name__ == "__main__":
    sys.exit(main())
