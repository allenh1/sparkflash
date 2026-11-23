#!/usr/bin/env python3
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

"""Reads kernels/instances.toml for the shell build scripts.

The file is the single list of kernel instances (source, entry symbol, the shapes one rank
sees). The Rust build scripts parse it too; this is the shell scripts' reader, so
scripts/build-cubins.sh and the baseline scripts compile a source with the same defines the
Cargo build probes it with.

    scripts/instances.py --cubin-plan          cubin<TAB>source<TAB>defines per compile group
    scripts/instances.py --sources             the sources that have instances, one per line
    scripts/instances.py --defines-for-cubin N the defines for one cubin (empty when not listed)
    scripts/instances.py --check               validate; print a summary or fail

A define is SPARK_INSTANCE_<ENTRY>_<ARG>=<value>u. Instances that share a cubin are compiled in
one translation unit, which is what keeps the released multi-symbol cubins whole.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INSTANCES = ROOT / "kernels" / "instances.toml"
IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
FAMILIES = {
    "router": ("hidden", "experts", "top_k"),
    "routed": ("hidden", "intermediate", "padded", "experts", "top_k"),
    # The rows and bulk family: the multi-row decode twins, the bulk-prefill MMA bodies and the
    # generic dense/GEMV projections. Every one of its sources takes the hidden width; the routed
    # expert bodies also take the intermediate width, the expert count and the selected slots.
    "rows": ("hidden", "intermediate", "experts", "top_k"),
}
ALLOWED = {
    "source",
    "entry",
    "cubin",
    "family",
    "catalogue",
    "catalogue_intermediate",
    "catalogue_dims",
    "intermediate_multiple",
    "max_top_k",
    *{key for keys in FAMILIES.values() for key in keys},
}
CATALOGUE_DIMS = {"hidden", "intermediate", "experts", "top_k"}


def load() -> list[dict]:
    """The validated instances, with a cubin default and a per-entry define list."""

    if not INSTANCES.is_file():
        raise SystemExit(f"{INSTANCES} is missing: the kernel instance list is the build's input")
    document = tomllib.loads(INSTANCES.read_text())
    if document.get("version") != 1:
        raise SystemExit(f"{INSTANCES}: expected version = 1")
    rows = document.get("instance")
    if not isinstance(rows, list) or not rows:
        raise SystemExit(f"{INSTANCES}: no [[instance]] tables")

    kernels = (ROOT / "kernels").resolve()
    seen: set[str] = set()
    result: list[dict] = []
    for number, row in enumerate(rows, 1):
        unknown = set(row) - ALLOWED
        if unknown:
            raise SystemExit(f"{INSTANCES}: instance {number}: unknown fields {sorted(unknown)}")
        source = row.get("source")
        entry = row.get("entry")
        family = row.get("family")
        if not isinstance(source, str) or not isinstance(entry, str):
            raise SystemExit(f"{INSTANCES}: instance {number}: source and entry are strings")
        if family not in FAMILIES:
            raise SystemExit(
                f"{INSTANCES}: instance {number}: family {family!r} is not one of {sorted(FAMILIES)}"
            )
        if not IDENTIFIER.match(entry):
            raise SystemExit(f"{INSTANCES}: instance {number}: {entry!r} is not a symbol")
        if entry in seen:
            raise SystemExit(f"{INSTANCES}: instance {number}: {entry} is listed twice")
        seen.add(entry)
        path = (kernels / source).resolve()
        if path.parent != kernels or not path.is_file():
            raise SystemExit(f"{INSTANCES}: instance {number}: kernels/{source} does not exist")
        text = path.read_text()
        if entry not in text:
            raise SystemExit(
                f"{INSTANCES}: instance {number}: {entry} does not appear in kernels/{source}"
            )
        cubin = row.get("cubin", path.stem)
        if not isinstance(cubin, str) or "/" in cubin or not IDENTIFIER.match(cubin):
            raise SystemExit(f"{INSTANCES}: instance {number}: bad cubin {cubin!r}")
        defines = []
        for key in FAMILIES[family]:
            value = row.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise SystemExit(
                    f"{INSTANCES}: instance {number} ({entry}): {key} is a positive integer"
                )
            defines.append(
                f"-DSPARK_INSTANCE_{entry.upper()}_{key.upper()}={value}u"
            )
        catalogue = row.get("catalogue")
        if catalogue is not None and (not isinstance(catalogue, str) or not catalogue):
            raise SystemExit(f"{INSTANCES}: instance {number}: bad catalogue key")
        dims = row.get("catalogue_dims")
        if dims is not None:
            if (
                not isinstance(dims, list)
                or not dims
                or any(not isinstance(dim, str) or dim not in CATALOGUE_DIMS for dim in dims)
                or len(set(dims)) != len(dims)
            ):
                raise SystemExit(
                    f"{INSTANCES}: instance {number}: catalogue_dims is a non-empty subset of "
                    f"{sorted(CATALOGUE_DIMS)}"
                )
            if catalogue is None:
                raise SystemExit(
                    f"{INSTANCES}: instance {number}: catalogue_dims needs a catalogue key"
                )
        intermediate_multiple = row.get("intermediate_multiple")
        if intermediate_multiple is not None and (
            not isinstance(intermediate_multiple, int)
            or isinstance(intermediate_multiple, bool)
            or intermediate_multiple <= 0
        ):
            raise SystemExit(
                f"{INSTANCES}: instance {number}: intermediate_multiple is a positive integer"
            )
        max_top_k = row.get("max_top_k")
        if max_top_k is not None and (
            not isinstance(max_top_k, int) or isinstance(max_top_k, bool) or max_top_k <= 0
        ):
            raise SystemExit(f"{INSTANCES}: instance {number}: max_top_k is a positive integer")
        result.append(
            {
                "source": f"kernels/{source}",
                "entry": entry,
                "cubin": cubin,
                "family": family,
                "catalogue": catalogue,
                "catalogue_dims": dims,
                "max_top_k": max_top_k,
                "defines": defines,
                "shapes": {key: row[key] for key in FAMILIES[family]},
            }
        )
    return result


def groups(rows: list[dict]) -> list[dict]:
    """One compile group per (source, cubin), in file order."""

    order: list[tuple[str, str]] = []
    defines: dict[tuple[str, str], list[str]] = {}
    for row in rows:
        key = (row["source"], row["cubin"])
        if key not in defines:
            order.append(key)
            defines[key] = []
        defines[key].extend(row["defines"])
    return [
        {"cubin": cubin, "source": source, "defines": defines[(source, cubin)]}
        for source, cubin in order
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cubin-plan", action="store_true", help="print compile groups")
    parser.add_argument("--sources", action="store_true", help="print sources with instances")
    parser.add_argument("--defines-for-cubin", metavar="NAME", help="print one cubin's defines")
    parser.add_argument("--check", action="store_true", help="validate and summarise")
    args = parser.parse_args()

    rows = load()
    if args.sources:
        for source in sorted({row["source"] for row in rows}):
            print(source)
        return 0
    if args.defines_for_cubin is not None:
        found = [group for group in groups(rows) if group["cubin"] == args.defines_for_cubin]
        if len(found) > 1:
            raise SystemExit(
                f"{args.defines_for_cubin} names more than one compile group: "
                + ", ".join(group["source"] for group in found)
            )
        print(" ".join(found[0]["defines"]) if found else "")
        return 0
    if args.cubin_plan:
        for group in groups(rows):
            print(f"{group['cubin']}\t{group['source']}\t{' '.join(group['defines'])}")
        return 0
    if args.check or not any(vars(args)[name] for name in vars(args)):
        compile_groups = groups(rows)
        print(
            f"{len(rows)} instances in {len({row['source'] for row in rows})} sources, "
            f"{len(compile_groups)} compile groups"
        )
        return 0
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())