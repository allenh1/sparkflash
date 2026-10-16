#!/usr/bin/env python3
"""Compile and test every non-default cargo feature in the workspace, and fail on a dead one.

`cargo test --workspace` builds only the default feature sets, so code behind a non-default
feature is never compiled by any job in this pipeline unless a job names it explicitly.
Milestone 2's io_uring ring lives behind `sparkflash-engram/io-uring`, which no crate in the
workspace enables: the ring and its tests were invisible to every green run for six review
rounds, and the defect review round 3 found (accepted requests orphaned in the submission
queue when a completion was lost) had survived all of them because nothing ever compiled that
file. Milestone 3's `sparkflash-comm/rdma` was in the same position.

The first version of this script skipped a feature whose value list was exactly `["dep:<name>"]`,
meaning to skip the implicit feature cargo generates for an optional dependency. That test is
wrong for a feature a manifest declares explicitly with the same shape: sparkflash-engram
declares `io-uring = ["dep:io-uring"]` on purpose, the script classified it as an implicit
toggle, and the ring stayed invisible - the CI job claimed the coverage that the review had just
found missing. A feature is implicit only if the manifest does not declare it, so this reads the
manifests.

It also fails on a feature that gates nothing: if no `cfg` in the package names a declared
feature and the feature forwards to no other feature, then enabling it changes no code, which is
how `sparkflash-trace`'s `trace-gpu-in-graph` and `trace-nvtx` read as capabilities that did
not exist. A new feature therefore forces an explicit decision here: either this pipeline
compiles it, or somebody writes down why it cannot be compiled in a container with no GPU, no
RDMA fabric and no checkpoint.
"""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Features whose TESTS need hardware this pipeline does not have. The cuda-abi job compiles CUDA in
# an image that does have a toolkit, so this one is covered elsewhere in the pipeline - but only the
# tests are what needs the GPU, and the lint is what needs the TOOLKIT, because cudarc's build script
# runs `nvcc --version` and refuses to configure a version without one.
#
# Round-19 A4: the skip below is for the tests, and the feature-scoped clippy call stands BEFORE it,
# so on a host with a toolkit the sweep lints the feature as well. Before that the `continue` skipped
# the lint along with the tests, and the workspace clippy resolves default features only - so
# sparkflash-kernels/cuda was linted by no gate at all while this file's docstring claimed it was.
# When no toolkit is on PATH the command cannot run here and the sweep SAYS SO and names the job that
# owns the feature, rather than dropping it quietly.
HARDWARE_ONLY = {
    ("sparkflash-kernels", "cuda"): "needs a toolkit and a target; the cuda-abi job compiles it",
}

SECTION = re.compile(r"^\s*\[([^\]]+)\]\s*$")
KEY = re.compile(r"^\s*([A-Za-z0-9_-]+)\s*=")


def parse_declared_features(manifest_text):
    """Feature names from a manifest's [features] table, in declaration order."""
    declared = []
    in_features = False
    for raw in manifest_text.splitlines():
        stripped = raw.split("#", 1)[0].rstrip()
        section = SECTION.match(stripped)
        if section:
            in_features = section.group(1).strip() == "features"
            continue
        if not in_features:
            continue
        key = KEY.match(stripped)
        if key:
            declared.append(key.group(1))
    return declared


def is_implicit_dependency_toggle(name, values, declared):
    """True only for the implicit feature cargo generates for an optional dependency.

    cargo adds `<dep> = ["dep:<dep>"]` for an optional dependency that no feature mentions.
    A manifest may declare the same shape by hand to name its own feature; that one is real.
    """
    return name not in declared and values == ["dep:" + name]


def gates_nothing(name, values, sources, referenced=()):
    """True when enabling the feature cannot change any code in its own package.

    A feature with any value is left alone: a value either forwards to another feature or names
    an optional dependency, and both change what gets built even when no cfg mentions the name.
    sparkflash-kernels/cuda is the second shape - it selects the cudarc dependency without any
    `cfg(feature = "cuda")` in the crate - so a rule that only looked for cfg would call it dead.
    What is left to catch is the empty feature nothing names, which is exactly the shape
    sparkflash-trace used for two GPU trace capabilities that had no code behind them at all.
    """
    if values or name in referenced:
        return False
    pattern = re.compile(r"feature\s*=\s*[\"']" + re.escape(name) + r"[\"']")
    return not any(pattern.search(text) for text in sources)


def classify(packages, manifests, sources):
    """Split discovered features into those to test and the findings that fail the job."""
    to_test, dead = [], []
    for package in packages:
        features = package.get("features", {})
        defaults = set(features.get("default", []))
        declared = parse_declared_features(manifests[package["name"]])
        referenced = {
            value for values in features.values() for value in values if not value.startswith("dep:")
        }
        for name, values in features.items():
            if name == "default" or name in defaults:
                continue
            if is_implicit_dependency_toggle(name, values, declared):
                continue
            if gates_nothing(name, values, sources.get(package["name"], []), referenced):
                dead.append((package["name"], name, values))
                continue
            to_test.append((package["name"], name))
    return sorted(to_test), sorted(dead)


def read_source_text(package_dir):
    """Every .rs file under a package, for the cfg search."""
    root = Path(package_dir)
    return [path.read_text(errors="replace") for path in root.rglob("*.rs")]


def workspace():
    out = subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"],
        capture_output=True,
        text=True,
        check=True,
    )
    packages = json.loads(out.stdout)["packages"]
    manifests, sources = {}, {}
    for package in packages:
        manifest = Path(package["manifest_path"])
        manifests[package["name"]] = manifest.read_text()
        sources[package["name"]] = read_source_text(manifest.parent)
    return packages, manifests, sources


def run(command):
    print("$ " + " ".join(command), flush=True)
    return subprocess.run(command).returncode


# The clippy gate, verbatim as CI runs it. It is the WORKSPACE invocation on purpose: a crate-scoped
# `cargo clippy -p <package>` does not see the workspace-level documentation lints, so a doc-list
# continuation in a crate this sweep does not scope over reached CI (round-18 A7). Keep it equal to the
# clippy step in .gitlab-ci.yml, and see ci/test_check_features.py::ClippyGate, which pins it.
CLIPPY_GATE = ["cargo", "clippy", "--workspace", "--all-targets", "--", "-D", "warnings"]


def feature_clippy(package, feature):
    """The feature-scoped clippy run, which the workspace invocation cannot replace.

    A workspace clippy resolves DEFAULT features only, so a path behind a non-default feature
    (sparkflash-engram/io-uring, sparkflash-comm/rdma, sparkflash-kernels/cuda) is not linted by it at
    all. The sweep runs BOTH: this one per feature, and CLIPPY_GATE once for the workspace, which is
    the command CI runs and the only one that sees cross-crate documentation lints (round-18 A7).
    """
    return [
        "cargo",
        "clippy",
        "-p",
        package,
        "--all-targets",
        "--features",
        feature,
        "--",
        "-D",
        "warnings",
    ]


def toolkit():
    """The nvcc the feature-scoped lint of a hardware-only feature needs, or None when there is none.

    `cargo clippy` needs no GPU, but it does need to BUILD the feature, and cudarc's build script runs
    `nvcc --version`: without a toolkit the cuda feature cannot even be configured, in this job's rust
    image or on a developer host. So the lint is not dropped, it is conditional on the toolkit - and
    when it is missing, main says which command it is not running and why (round-19 A4).
    """
    return shutil.which(os.environ.get("SPARKFLASH_NVCC", "nvcc"))


def commands_for(found, toolkit_present):
    """The commands the sweep runs, in the order it runs them, for `classify`'s list of features.

    The feature-scoped clippy stands BEFORE the hardware-only skip on purpose (round-19 A4). Clippy
    needs no GPU: only the tests do, and the workspace gate resolves default features only, so skipping
    the feature-scoped lint left sparkflash-kernels/cuda linted by neither gate while this file claimed
    otherwise. The lint runs whenever one can - `toolkit_present` is what decides that for a
    hardware-only feature - and this list is what the tests in ci/test_check_features.py pin, so a
    command that stops being produced fails a test rather than the pipeline.
    """
    commands = []
    for package, feature in found:
        hardware_only = (package, feature) in HARDWARE_ONLY
        if not hardware_only or toolkit_present:
            commands.append(feature_clippy(package, feature))
        if hardware_only:
            continue
        commands.append(["cargo", "test", "-p", package, "--features", feature])
    # And one workspace-wide clippy run, exactly the command CI runs, which is the only one that sees
    # cross-crate documentation lints - the reason a clippy::doc_lazy_continuation in one crate's test
    # code reached CI while the feature-scoped runs stayed green (round-18 A7).
    commands.append(CLIPPY_GATE)
    return commands


def main():
    packages, manifests, sources = workspace()
    found, dead = classify(packages, manifests, sources)

    print("non-default, non-dependency features in this workspace:")
    for package, feature in found:
        hardware = HARDWARE_ONLY.get((package, feature))
        print(f"  {package}/{feature}" + (f"  [not compiled here: {hardware}]" if hardware else ""))

    failed = False
    for package, feature, values in dead:
        print(
            f"{package}/{feature} gates nothing: no cfg in the package names it and its value "
            f"list {values} forwards to no feature. Either use it or delete it.",
            file=sys.stderr,
        )
        failed = True

    missing = sorted(set(HARDWARE_ONLY) - set(found))
    if missing:
        print(
            "these features are classified as hardware-only but no longer exist: "
            + ", ".join(f"{p}/{f}" for p, f in missing),
            file=sys.stderr,
        )
        failed = True

    if failed:
        return 1

    # BOTH clippy gates, and for a hardware-only feature the lint still runs when one can: only its
    # tests are skipped. Without a toolkit the lint cannot build the feature, and that is said loudly.
    toolkit_present = toolkit()
    if toolkit_present is None and any((package, feature) in HARDWARE_ONLY for package, feature in found):
        print(
            "no CUDA toolkit on PATH, so the feature-scoped lint of a hardware-only feature cannot run "
            "here (cudarc's build script runs nvcc --version). The cuda-abi job owns that feature and "
            "should lint it there; this is the one gate that cannot run in this image.",
            file=sys.stderr,
        )
    failures = 0
    for command in commands_for(found, toolkit_present):
        failures += run(command) != 0

    if failures:
        print(f"{failures} feature-scoped command(s) failed", file=sys.stderr)
        return 1
    print("every non-default feature compiles and passes its tests")
    return 0


if __name__ == "__main__":
    sys.exit(main())
