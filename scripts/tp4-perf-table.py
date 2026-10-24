#!/usr/bin/env python3
"""Render the per-stage GPU-time table from a `--forward-perf-probe-nosync` row.

The status doc's per-stage table (`docs/build/s1-result.md` section 8.2) is the input to the kernel
re-tile: it names which stages own the step's kernel time. Until this script existed no in-tree tool
produced it -- `scripts/tp4-probe-attribution.py` reads only rank 0, sums the per-layer `wall_ms`
(host wall, not `kernel_ns`), and skips every `head`-flagged position, which are the decode positions
this table is about. This renders what 8.2 renders, from the probe documents themselves.

What it computes, per stage, as a mean over a set of decode positions:

  launches / step    sum of `launches` over all 40 layers, per position, averaged
  kernel ms / step   sum of `kernel_ns` over all 40 layers, per position, averaged
  kernel us / layer  the per-step kernel time spread over 40 layers (blank where the row is not a
                     per-layer stage: the head and the step total)
  wall ms / step     sum of `wall_ms` over all 40 layers (host dispatch + collective wall)
  share              share of the summed 40-layer kernel time

plus the `40-layer subtotal`, the head's own stages (H1-H3, recorded in `positions[].head_blocks`
outside `layers`) and the `Total step` row, which is the subtotal plus the head - the step's kernel
time, not the 40-layer subtotal alone.

Rounding: stage values are summed unrounded and printed to three decimals. A total can therefore
differ in the last digit from a published sum of round-to-three-decimal rows (the archived S1 row's
step wall prints 203.945 here against the document's 203.944, and its H1 wall 0.006 against 0.007,
which is half-up of 0.0065).

The three bases match the document's own accounting: layers are summed in the probe's own `layers`
array (not inferred from seams), `kernel_ns` is the deferred `cuEventElapsedTime` harvest (so the
value carries no per-block boundary drain), and `wall_ms` is the host-only wall the probe records
beside it. Both columns of 8.2 therefore come from one pass.

Trimmed rows: the emitter writes each block row trimmed at its last non-zero field, with `wall_ms`
always first, so a valid row is any length from 1 to the legend's and the missing trailing fields are
zero (`crates/sparkflash-cli/src/tp4_forward.rs:9400-9402`). Short rows are padded, not refused.
Only a host-only row is accepted (`--forward-perf-probe-nosync`): the probe's legend says a
device-inclusive row's wall and `kernel_ns` are on different bases and must not be mixed with these.

Usage:
    tp4-perf-table.py ROW_DIR [--rank N] [--steady N] [--json] [--expect STAGE=MS ...]

ROW_DIR is either the row directory (with `rank0/`, `rank1/`, ... inside) or a single rank directory.
`--steady N` keeps only the last N decode positions (8.2 used all eight of that row). `--expect`
asserts a stage's `kernel ms/step` within 0.0005 and exits non-zero on a mismatch, which is how the
table's kernel column is pinned to the published doc (wall, launches, us/layer and share are printed
but not asserted).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCHEMA_PREFIX = "sparkflash-tp4-forward-perf/"
LAYERS_PER_STEP = 40


def load_doc(row_dir: Path, rank: int) -> dict:
    """Find and load one rank's probe document under a row directory."""
    candidates = []
    if (row_dir / f"tp4-forward-perf-r{rank}.json").is_file():
        candidates.append(row_dir / f"tp4-forward-perf-r{rank}.json")
    candidates.extend(sorted(row_dir.glob(f"rank{rank}/tp4-forward-perf-r*.json")))
    candidates.extend(sorted(row_dir.glob(f"**/tp4-forward-perf-r{rank}.json")))
    for path in candidates:
        if path.is_file():
            with path.open() as handle:
                doc = json.load(handle)
            doc["_path"] = str(path)
            return doc
    raise SystemExit(f"no tp4-forward-perf-r{rank}.json under {row_dir}")


def check_schema(doc: dict) -> None:
    schema = doc.get("schema")
    if not isinstance(schema, str) or not schema.startswith(SCHEMA_PREFIX):
        raise SystemExit(f"unexpected probe schema {schema!r}; expected {SCHEMA_PREFIX}*")


def check_mode(doc: dict) -> None:
    """Refuse a row whose wall and kernel columns are on different bases.

    The probe's own legend says a device-inclusive row reads the wall before a boundary drain and
    takes `kernel_ns` from the meter, while a host-only row never drains and takes `kernel_ns` from
    one deferred harvest; the two must not be averaged together, and the published table is
    host-only (`--forward-perf-probe-nosync`).
    """
    wall_clock = (doc.get("mode") or {}).get("wall_clock")
    if wall_clock != "host-only":
        raise SystemExit(
            f"probe row is {wall_clock!r}, not host-only: its wall is device-inclusive and its "
            "kernel_ns comes from the meter, so it is not comparable to the published host-only "
            "table; re-run with --forward-perf-probe-nosync"
        )


def block_fields(doc: dict) -> list[str]:
    legend = doc.get("legend") or {}
    fields = legend.get("block_fields")
    if not fields:
        raise SystemExit("document has no legend.block_fields; cannot name the block columns")
    return fields


def as_record(row: list, fields: list[str]) -> dict:
    """Decode one block row.

    The emitter trims a row at its last non-zero field with `wall_ms` always first
    (`crates/sparkflash-cli/src/tp4_forward.rs:9400-9402`), so a valid row can be any length from 1
    to the legend's, and the missing trailing fields are zero. Padding is therefore decoding, not
    guessing: a wait-only host block whose wall sits entirely in `wait_ms` arrives as 19 values, a
    host-only block as one. An empty row or a row longer than the legend is refused.
    """
    if not row:
        raise SystemExit("block row has no values")
    if len(row) > len(fields):
        raise SystemExit(f"block row has {len(row)} values but the legend names only {len(fields)}")
    padded = list(row) + [0] * (len(fields) - len(row))
    return dict(zip(fields, padded))


def decode_positions(doc: dict) -> list[dict]:
    """The decode positions: those the document flags with `head` truthy."""
    return [position for position in doc.get("positions", []) if position.get("head")]


def aggregate(doc: dict, positions: list[dict]) -> dict[str, dict]:
    fields = block_fields(doc)
    stages: dict[str, dict] = {}
    for position in positions:
        for layer in position.get("layers") or []:
            for name, row in (layer.get("blocks") or {}).items():
                record = as_record(row, fields)
                stage = stages.setdefault(name, {"kernel_ns": 0, "wall_ms": 0.0, "launches": 0})
                stage["kernel_ns"] += record["kernel_ns"]
                stage["wall_ms"] += record["wall_ms"]
                stage["launches"] += record["launches"]
    return stages


HEAD_STAGES = ("hc_pre", "screen", "gather_logits")


def head_aggregate(doc: dict, positions: list[dict]) -> dict[str, dict]:
    """The head's stages, recorded per position in `positions[].head_blocks`.

    The published table lists them as H1-H3 in the same units as the layer stages, so they are
    aggregated the same way. `head_total` is the head's own sum and is used only for the step total,
    never as a row of its own (it would double-count).
    """
    fields = block_fields(doc)
    stages: dict[str, dict] = {}
    for position in positions:
        for name, row in (position.get("head_blocks") or {}).items():
            if name not in HEAD_STAGES:
                continue
            record = as_record(row, fields)
            stage = stages.setdefault(name, {"kernel_ns": 0, "wall_ms": 0.0, "launches": 0})
            stage["kernel_ns"] += record["kernel_ns"]
            stage["wall_ms"] += record["wall_ms"]
            stage["launches"] += record["launches"]
    return stages


def rows(doc: dict, positions: list[dict]) -> tuple[list[dict], dict]:
    """Render the published table: layer stages, the 40-layer subtotal, the head, the step total."""
    stages = aggregate(doc, positions)
    steps = len(positions)
    if not steps:
        raise SystemExit("no decode positions in the document (every position has head=false)")
    subtotal_kernel = sum(stage["kernel_ns"] for stage in stages.values()) / steps / 1e6
    out = []
    for name, stage in sorted(stages.items(), key=lambda item: -item[1]["kernel_ns"]):
        kernel_ms = stage["kernel_ns"] / steps / 1e6
        out.append(
            {
                "key": name,
                "stage": name,
                "launches_per_step": stage["launches"] / steps,
                "kernel_ms_per_step": kernel_ms,
                "kernel_us_per_layer": kernel_ms * 1000.0 / LAYERS_PER_STEP,
                "wall_ms_per_step": stage["wall_ms"] / steps,
                "share": (100.0 * kernel_ms / subtotal_kernel) if subtotal_kernel else 0.0,
                "layers_with_kernel_time": _layers_with_kernel_time(doc, positions, name),
            }
        )
    subtotal_wall = sum(row["wall_ms_per_step"] for row in out)
    subtotal_launches = sum(row["launches_per_step"] for row in out)
    subtotal = {
        "key": "subtotal_40l",
        "stage": f"{LAYERS_PER_STEP}-layer subtotal",
        "launches_per_step": subtotal_launches,
        "kernel_ms_per_step": subtotal_kernel,
        "kernel_us_per_layer": subtotal_kernel * 1000.0 / LAYERS_PER_STEP,
        "wall_ms_per_step": subtotal_wall,
        "share": 100.0,
        "layers_with_kernel_time": LAYERS_PER_STEP,
    }
    heads = head_aggregate(doc, positions)
    head_rows = []
    for name in HEAD_STAGES:
        stage = heads.get(name)
        if stage is None:
            continue
        head_rows.append(
            {
                "key": f"head_{name}",
                "stage": f"head.{name}",
                "launches_per_step": stage["launches"] / steps,
                "kernel_ms_per_step": stage["kernel_ns"] / steps / 1e6,
                "kernel_us_per_layer": stage["kernel_ns"] / steps / 1e6 * 1000.0 / LAYERS_PER_STEP,
                "wall_ms_per_step": stage["wall_ms"] / steps,
                "share": None,
                "layers_with_kernel_time": None,
            }
        )
    # The step total is the 40-layer subtotal plus the head's own stages. The record also carries a
    # `head_total` block; it agrees on kernel time and differs on wall by the difference between the
    # staged rows and the block total, so the rendered head rows are summed to keep the page additive.
    head_kernel = sum(row["kernel_ms_per_step"] for row in head_rows)
    head_wall = sum(row["wall_ms_per_step"] for row in head_rows)
    total = {
        "key": "total_step",
        "stage": "Total step (40 layers + head)",
        "launches_per_step": subtotal_launches + sum(row["launches_per_step"] for row in head_rows),
        "kernel_ms_per_step": subtotal_kernel + head_kernel,
        "kernel_us_per_layer": (subtotal_kernel + head_kernel) * 1000.0 / LAYERS_PER_STEP,
        "wall_ms_per_step": subtotal_wall + head_wall,
        "share": None,
        "layers_with_kernel_time": None,
    }
    totals = {
        "positions": steps,
        "first_position": positions[0].get("position"),
        "last_position": positions[-1].get("position"),
        "kernel_ms_per_step": subtotal_kernel,
        "wall_ms_per_step": subtotal_wall,
        "launches_per_step": subtotal_launches,
        "head_kernel_ms_per_step": head_kernel,
        "head_wall_ms_per_step": head_wall,
        "total_kernel_ms_per_step": total["kernel_ms_per_step"],
        "total_wall_ms_per_step": total["wall_ms_per_step"],
    }
    return out + [subtotal] + head_rows + [total], totals


def _layers_with_kernel_time(doc: dict, positions: list[dict], name: str) -> int:
    """How many of the document's layers carry non-zero kernel time for this stage (mean rounded)."""
    fields = block_fields(doc)
    total = 0
    for position in positions:
        count = 0
        for layer in position.get("layers") or []:
            row = (layer.get("blocks") or {}).get(name)
            if row is not None and as_record(row, fields)["kernel_ns"] != 0:
                count += 1
        total += count
    return round(total / len(positions))


def clock_window(row_dir: Path, rank: int) -> str:
    """Read the row's own gpu.csv when it has one: clocks are part of the number, not of the doc.

    The probe document does not record clocks (and the S2 rows do not ship a gpu.csv either), so a
    row that wants its table quoted beside a clock window has to carry one. S1's archive rows do.
    """
    preferred = sorted(row_dir.glob(f"**/rank{rank}/gpu.csv"))
    fallback = [] if preferred else sorted(row_dir.glob("**/gpu.csv"))
    for path in preferred or fallback:
        note = "" if path in preferred else " (requested rank's gpu.csv absent; showing this rank's)"
        values = []
        with path.open() as handle:
            for line in handle:
                parts = line.replace(",", " ").split()
                numbers = [part for part in parts if part.replace(".", "", 1).isdigit()]
                if len(numbers) >= 2:
                    values.append(float(numbers[1]))
        if values:
            median = sorted(values)[len(values) // 2]
            return (
                f"clocks: {min(values):.0f}-{max(values):.0f} MHz "
                f"(median {median:.0f}, {len(values)} samples, {path}{note})"
            )
    return (
        "clocks: not recorded in this row (no readable gpu.csv); quote the window from the run sheet"
    )


def provenance(doc: dict, row_dir: Path, rank: int) -> list[str]:
    """State what the table is measured on, so the number cannot travel without its mode."""
    mode = doc.get("mode") or {}
    legend_mode = (doc.get("legend") or {}).get("mode") or {}
    lines = [
        f"rank {doc.get('rank', rank)} of {doc.get('world', '?')}",
        f"schema {doc.get('schema', '?')}",
        f"probe mode: lean={mode.get('lean')} synchronize_per_block={mode.get('synchronize_per_block')} "
        f"wall_clock={mode.get('wall_clock')}",
        f"doc: {doc.get('_path', row_dir)}",
    ]
    if legend_mode:
        lines.append(f"legend mode: {legend_mode}")
    lines.append("arm/commit: not recorded in the probe document; cite rank*/run.log for it")
    lines.append(clock_window(row_dir, rank))
    return lines


def render(doc: dict, row_dir: Path, rank: int, positions: list[dict], steady: int) -> dict:
    out, totals = rows(doc, positions)
    return {
        "provenance": provenance(doc, row_dir, rank),
        "steady": steady,
        "rows": out,
        "totals": totals,
    }


def print_table(report: dict) -> None:
    for line in report["provenance"]:
        print(f"# {line}")
    totals = report["totals"]
    print(
        f"# decode positions {totals['first_position']}..{totals['last_position']} "
        f"({totals['positions']} steps, {LAYERS_PER_STEP} layers each)"
    )
    print()
    print(
        f"{'stage':26s} {'launch/step':>11s} {'kern ms/step':>13s} {'us/layer':>9s} "
        f"{'wall ms/step':>13s} {'share':>7s}"
    )
    for row in report["rows"]:
        share = "      -" if row["share"] is None else f"{row['share']:6.1f}%"
        us_per_layer = (
            "        -"
            if row["layers_with_kernel_time"] is None
            else f"{row['kernel_us_per_layer']:9.1f}"
        )
        print(
            f"{row['stage']:26s} {row['launches_per_step']:11.1f} {row['kernel_ms_per_step']:13.3f} "
            f"{us_per_layer} {row['wall_ms_per_step']:13.3f} {share}"
        )


def check_expectations(report: dict, expectations: dict[str, float]) -> int:
    by_stage = {}
    for row in report["rows"]:
        by_stage[row["key"]] = row
        by_stage[row["stage"]] = row
    failures = 0
    for stage, expected in expectations.items():
        row = by_stage.get(stage)
        if row is None:
            print(f"expect {stage}: absent from the table", file=sys.stderr)
            failures += 1
            continue
        actual = row["kernel_ms_per_step"]
        if abs(actual - expected) > 0.0005:
            print(f"expect {stage}: {actual:.3f} != {expected:.3f}", file=sys.stderr)
            failures += 1
    return failures


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("row_dir", type=Path)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument(
        "--steady",
        type=int,
        default=0,
        help="keep only the last N decode positions (0 = every decode position the row has)",
    )
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    parser.add_argument(
        "--expect",
        action="append",
        default=[],
        metavar="STAGE=MS",
        help="assert a stage's kernel ms/step (repeatable)",
    )
    args = parser.parse_args(argv)

    row_dir = args.row_dir
    if not row_dir.is_dir():
        print(f"not a directory: {row_dir}", file=sys.stderr)
        return 2
    try:
        doc = load_doc(row_dir, args.rank)
        check_schema(doc)
        check_mode(doc)
        positions = decode_positions(doc)
    except SystemExit as refusal:
        # Every refusal is a clean exit 2 with its reason on stderr, not a traceback or a bare exit 1.
        print(refusal.code, file=sys.stderr)
        return 2
    if args.steady < 0:
        print(f"--steady must not be negative (got {args.steady})", file=sys.stderr)
        return 2
    if args.steady:
        if args.steady > len(positions):
            print(
                f"note: --steady {args.steady} exceeds the row's {len(positions)} decode positions; "
                "using all of them",
                file=sys.stderr,
            )
        positions = positions[-args.steady :]
    report = render(doc, row_dir, args.rank, positions, args.steady)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_table(report)
    expectations = {}
    for item in args.expect:
        stage, _, value = item.partition("=")
        try:
            expectations[stage] = float(value)
        except ValueError:
            print(f"--expect {item!r} is not STAGE=MS", file=sys.stderr)
            return 2
    return 1 if check_expectations(report, expectations) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
