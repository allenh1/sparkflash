#!/usr/bin/env bash
# The S1 layer-localisation tool: host arm vs device arm on the same layer and position.
#
#   scripts/qa-layer-diff.sh --layer N --host DIR --device DIR [--position P] [--bounds FILE]
#   scripts/qa-layer-diff.sh --self-test
#
# The device arm deliberately has no per-stage observation read in its product path - that is what
# makes it fast (docs/build/ws-device-arm-numerical-gate-20261022.md section 3). Attribution is
# therefore a DEBUG ROW: a dev-gated build writes the same per-boundary f32 rows the host arm's
# boundary trace already writes, one row per stage boundary, and this script diffs the two arms row
# by row. It never quotes a number; its output is a localisation.
#
# Observation layout (both arms, same file names):
#
#   DIR/pos{P}/row{L}.{name}.f32     little-endian f32, one boundary row
#
# with these rows, in execution order:
#
#   residual_in  hc_in_pre                    layer input (written before this layer: L-1's
#                                             ffn_epilogue / begin_position - divergence here is
#                                             upstream of layer L, not a stage of it)
#   hc_attn_pre  hc_attn_post  hc_attn_comb
#   pre_attention_hidden                      mhc_entry
#   attention_out                             all_reduce (the reduced attention delta)
#   post_attention_hidden                     attention_residual
#   hc_ffn_pre  hc_ffn_post  hc_ffn_comb
#   pre_ffn_hidden                            ffn_seams
#   ffn_out  residual_out                     ffn_epilogue
#
# The host arm writes this set today from `write_boundary_trace_row` under
# `SPARKFLASH_BOUNDARY_TRACE` + `SPARKFLASH_TRACE_POSITIONS` (rank 0 only). The device arm's copy of
# the set is NOT landed: the product arm has no observation read, so a dev-gated writer has to be
# wired before this tool can run. Until it is, the tool fails closed (exit 2) at the artifact check
# instead of inventing one.
#
# The bisection rule: a row diverges when its max |host - device| exceeds 10x the unit-test bound
# declared for it (and, when an ULP bound is declared, its max ULP distance exceeds 10x that bound).
# The first stage in execution order with a divergent row is the reported fault; the rule does not
# guess between stages and does not average.
#
# Exit codes (fail-closed contract):
#   0  every observed row is within 10x its declared unit-test bound
#   1  a divergent stage was found; the first one and the row that jumped are printed
#   2  no GPU / no observation copies / no declared bounds / usage: NOTHING was compared
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LAYER=""
HOST_DIR=""
DEVICE_DIR=""
POSITION=4
BOUNDS="$ROOT/docs/build/reference/s1-layer-diff-bounds.json"
SELF_TEST=0

usage() {
  printf '%s\n' \
    "usage: scripts/qa-layer-diff.sh --layer N --host DIR --device DIR [--position P] [--bounds FILE]" \
    "       scripts/qa-layer-diff.sh --self-test" >&2
  exit 2
}

fail_closed() {
  printf 'qa-layer-diff: %s\n' "$1" >&2
  shift
  printf '  %s\n' "$@" >&2
  exit 2
}

gpu_present() {
  if compgen -G '/dev/nvidia*' >/dev/null 2>&1; then
    return 0
  fi
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    return 0
  fi
  return 1
}

while [ $# -gt 0 ]; do
  case "$1" in
    --layer)
      [ $# -ge 2 ] || usage
      LAYER=$2
      shift 2
      ;;
    --host)
      [ $# -ge 2 ] || usage
      HOST_DIR=$2
      shift 2
      ;;
    --device)
      [ $# -ge 2 ] || usage
      DEVICE_DIR=$2
      shift 2
      ;;
    --position)
      [ $# -ge 2 ] || usage
      POSITION=$2
      shift 2
      ;;
    --bounds)
      [ $# -ge 2 ] || usage
      BOUNDS=$2
      shift 2
      ;;
    --self-test)
      SELF_TEST=1
      shift
      ;;
    -h | --help) usage ;;
    *) usage ;;
  esac
done

python_helper() {
  python3 - "$@" <<'PY'
import array
import json
import math
import os
import struct
import sys
import tempfile

# (producing stage, observation row) in execution order. The stage label is the bisection unit: the
# first stage with a divergent row is the reported fault. `layer_input` rows are produced before this
# layer by the previous layer's epilogue / the position's begin_position, so a divergence there is
# upstream of layer L.
ROWS = [
    ("layer_input", "residual_in"),
    ("layer_input", "hc_in_pre"),
    ("mhc_entry", "hc_attn_pre"),
    ("mhc_entry", "hc_attn_post"),
    ("mhc_entry", "hc_attn_comb"),
    ("mhc_entry", "pre_attention_hidden"),
    ("all_reduce", "attention_out"),
    ("attention_residual", "post_attention_hidden"),
    ("ffn_seams", "hc_ffn_pre"),
    ("ffn_seams", "hc_ffn_post"),
    ("ffn_seams", "hc_ffn_comb"),
    ("ffn_seams", "pre_ffn_hidden"),
    ("ffn_epilogue", "ffn_out"),
    ("ffn_epilogue", "residual_out"),
]
SCHEMA = "s1-layer-diff-bounds-v1"
DECODER_LAYERS = 40
RULE_FACTOR = 10.0


def die(message):
    print(f"qa-layer-diff: {message}", file=sys.stderr)
    sys.exit(2)


def row_path(root, layer, position, name):
    return os.path.join(root, f"pos{position}", f"row{layer}.{name}.f32")


def read_row(path):
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as handle:
        data = handle.read()
    if len(data) % 4:
        die(f"observation row {path} is {len(data)} bytes, not a whole number of f32 values")
    values = array.array("f")
    values.frombytes(data)
    if sys.byteorder != "little":
        values.byteswap()
    return values


def write_row(path, values):
    data = array.array("f", values)
    if sys.byteorder != "little":
        data.byteswap()
    with open(path, "wb") as handle:
        handle.write(data.tobytes())


def load_bounds(path):
    if not os.path.isfile(path):
        die(
            "no unit-test bounds: "
            f"{path} is absent. The bisection rule compares each row's deviation to 10x its "
            "declared unit-test bound; a guessed bound would turn the rule into an opinion. "
            "Freeze the bounds (schema and derivation in docs/build/s1-qa-harness.md) and re-run. "
            "Nothing was compared."
        )
    try:
        with open(path) as handle:
            document = json.load(handle)
    except (OSError, ValueError) as error:
        die(f"cannot read the bounds file {path}: {error}")
    if document.get("schema") != SCHEMA:
        die(f"bounds schema is {document.get('schema')!r}, expected {SCHEMA!r}")
    bounds = document.get("bounds")
    if not isinstance(bounds, dict):
        die(f"{path}: bounds must be an object keyed by observation row")
    resolved = {}
    for _stage, name in ROWS:
        entry = bounds.get(name)
        if not isinstance(entry, dict):
            die(
                f"{path}: no unit-test bound for observation row {name!r}; the rule cannot be applied "
                "to a row with no declared bound (fill every row or do not run the bisection)"
            )
        max_abs = entry.get("max_abs")
        if not isinstance(max_abs, (int, float)) or not math.isfinite(float(max_abs)) or max_abs <= 0:
            die(f"{path}: observation row {name!r} needs a positive finite max_abs bound")
        max_ulp = entry.get("max_ulp")
        if max_ulp is not None and (
            not isinstance(max_ulp, int) or isinstance(max_ulp, bool) or max_ulp < 0
        ):
            die(f"{path}: observation row {name!r} has a non-integer max_ulp bound")
        resolved[name] = {"max_abs": float(max_abs), "max_ulp": max_ulp}
    return resolved


def ordered_bits(value):
    """Map an f32 to a monotone integer so ULP distance is |a - b| of the mapped ints."""
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    if bits & 0x8000_0000:
        return -(bits & 0x7FFF_FFFF)
    return bits


def ulp_distance(left, right):
    if math.isnan(left) or math.isnan(right):
        return float("inf")
    if left == right:
        return 0
    return abs(ordered_bits(left) - ordered_bits(right))


def compare_row(host_values, device_values, name):
    if len(host_values) != len(device_values):
        die(
            f"observation row {name}: host has {len(host_values)} f32 value(s), device has "
            f"{len(device_values)}; the two arms did not dump the same span"
        )
    max_abs = 0.0
    max_ulp = 0.0
    non_finite = 0
    for left, right in zip(host_values, device_values):
        if not (math.isfinite(left) and math.isfinite(right)):
            non_finite += 1
            max_abs = float("inf")
            max_ulp = float("inf")
            continue
        max_abs = max(max_abs, abs(left - right))
        max_ulp = max(max_ulp, ulp_distance(left, right))
    return max_abs, max_ulp, non_finite


def compare_roots(host_root, device_root, layer, position, bounds):
    if not os.path.isdir(host_root):
        die(
            f"host-arm observation root is absent ({host_root}); run the host arm with "
            "SPARKFLASH_BOUNDARY_TRACE=<DIR> and SPARKFLASH_TRACE_POSITIONS=<position> first"
        )
    missing_device = [
        name
        for _stage, name in ROWS
        if read_row(row_path(device_root, layer, position, name)) is None
    ]
    if not os.path.isdir(device_root) or len(missing_device) == len(ROWS):
        die(
            "the device arm's per-stage observation copies are absent "
            f"({device_root}/pos{position}/row{layer}.*.f32). The product arm has no per-stage "
            "readback by design, so the debug row needs a dev-gated writer that lands the same "
            "observation rows the host writer lands. That writer is not landed on this tree; "
            "nothing was compared (exit 2)."
        )
    results = []
    for stage, name in ROWS:
        path = row_path(host_root, layer, position, name)
        host_values = read_row(path)
        if host_values is None:
            die(f"host-arm observation row is missing ({path}); nothing was compared")
        device_path = row_path(device_root, layer, position, name)
        device_values = read_row(device_path)
        if device_values is None:
            die(
                f"device-arm observation row is missing ({device_path}); the dev-gated observation "
                "writer is not complete for this layer/position (exit 2)"
            )
        max_abs, max_ulp, non_finite = compare_row(host_values, device_values, name)
        bound = bounds[name]
        ratio_abs = math.inf if max_abs == math.inf else max_abs / bound["max_abs"]
        divergent = non_finite > 0 or ratio_abs > RULE_FACTOR
        reason = ""
        if non_finite > 0:
            reason = f"{non_finite} non-finite value(s) in a row of {len(host_values)}"
        elif ratio_abs > RULE_FACTOR:
            reason = f"max_abs {max_abs:.3e} > {RULE_FACTOR:g} x {bound['max_abs']:.3e}"
        ratio_ulp = None
        if bound["max_ulp"] is not None:
            ratio_ulp = math.inf if math.isinf(max_ulp) else max_ulp / bound["max_ulp"]
            if ratio_ulp > RULE_FACTOR:
                divergent = True
                ulp_reason = f"max_ulp {max_ulp:.0f} > {RULE_FACTOR:g} x {bound['max_ulp']}"
                reason = f"{reason}; {ulp_reason}" if reason else ulp_reason
        results.append(
            {
                "row": name,
                "stage": stage,
                "max_abs": max_abs,
                "max_ulp": max_ulp,
                "abs_bound": bound["max_abs"],
                "ulp_bound": bound["max_ulp"],
                "ratio_abs": ratio_abs,
                "ratio_ulp": ratio_ulp,
                "non_finite": non_finite,
                "divergent": divergent,
                "reason": reason,
            }
        )
    return results


def bisection(results):
    """The first stage in execution order with a divergent row, and that stage's worst row."""
    order = []
    for result in results:
        if result["stage"] not in order:
            order.append(result["stage"])
    for stage in order:
        stage_rows = [row for row in results if row["stage"] == stage and row["divergent"]]
        if stage_rows:
            worst = max(stage_rows, key=lambda row: row["ratio_abs"])
            return stage, worst
    return None, None


def print_report(layer, position, host_root, device_root, results):
    print(f"layer {layer} position {position}: host {host_root} vs device {device_root}")
    print(
        f"{'row':<24} {'stage':<18} {'max_abs':>11} {'max_ulp':>10} {'abs_bound':>11} "
        f"{'ratio':>8}  verdict"
    )
    for result in results:
        ratio = result["ratio_abs"]
        ratio_text = "inf" if math.isinf(ratio) else f"{ratio:.2f}"
        verdict = "DIVERGENT" if result["divergent"] else "ok"
        print(
            f"{result['row']:<24} {result['stage']:<18} {result['max_abs']:>11.3e} "
            f"{result['max_ulp']:>10.0f} {result['abs_bound']:>11.3e} {ratio_text:>8}  {verdict}"
            + (f"  ({result['reason']})" if result["reason"] else "")
        )


def main():
    mode = sys.argv[1]
    if mode == "diff":
        layer, host_root, device_root, position, bounds_path = (
            int(sys.argv[2]),
            sys.argv[3],
            sys.argv[4],
            int(sys.argv[5]),
            sys.argv[6],
        )
        if not 0 <= layer < DECODER_LAYERS:
            die(f"--layer {layer} is outside the released {DECODER_LAYERS}-layer stack")
        if position < 0:
            die(f"--position {position} is negative")
        bounds = load_bounds(bounds_path)
        results = compare_roots(host_root, device_root, layer, position, bounds)
        print_report(layer, position, host_root, device_root, results)
        stage, worst = bisection(results)
        if stage is None:
            print(
                f"BISECTION: no stage exceeds {RULE_FACTOR:g}x its unit-test bound; "
                "the layer is within its declared per-stage noise"
            )
            return 0
        print(
            f"BISECTION: first divergent stage = {stage} (row {worst['row']}: {worst['reason']})"
        )
        return 1
    if mode == "self-test":
        self_test()
        return 0
    die(f"unknown helper mode {mode!r}")


def write_all(root, layer, position, values):
    directory = os.path.join(root, f"pos{position}")
    os.makedirs(directory, exist_ok=True)
    for name, row in values.items():
        write_row(row_path(root, layer, position, name), row)


def synthetic_root(directory, layer, position, overrides=None):
    base = [math.sin(index * 0.37) * 0.5 for index in range(40)]
    values = {name: list(base) for _stage, name in ROWS}
    if overrides:
        for name, mutate in overrides.items():
            mutator = mutate(list(base))
            values[name] = mutator
    write_all(directory, layer, position, values)


def self_test():
    work = tempfile.mkdtemp(prefix="qa-layer-diff-")
    try:
        layer, position = 7, 4
        bounds_path = os.path.join(work, "bounds.json")
        bounds_document = {
            "schema": SCHEMA,
            "bounds": {
                name: {"stage": stage, "max_abs": 1.0e-6, "max_ulp": 8}
                for stage, name in ROWS
            },
        }
        with open(bounds_path, "w") as handle:
            json.dump(bounds_document, handle)
        bounds = load_bounds(bounds_path)

        def run(host_root, device_root):
            return compare_roots(host_root, device_root, layer, position, bounds)

        # 1. Identical copies are within bounds and bisect to nothing.
        host = os.path.join(work, "host")
        device = os.path.join(work, "device")
        synthetic_root(host, layer, position)
        synthetic_root(device, layer, position)
        results = run(host, device)
        stage, _worst = bisection(results)
        assert stage is None, f"identical copies must not bisect, got {stage}"
        print("  self-test: identical arms -> no divergent stage")

        # 2. A 50x jump in ffn_seams/pre_ffn_hidden is the first divergent stage.
        device2 = os.path.join(work, "device-ffn")
        synthetic_root(device2, layer, position,
                       {"pre_ffn_hidden": lambda row: [value + 50.0e-6 for value in row]})
        stage, worst = bisection(run(host, device2))
        assert stage == "ffn_seams", f"expected ffn_seams, got {stage}"
        assert worst["row"] == "pre_ffn_hidden", worst
        print("  self-test: ffn_seams jump -> bisection at ffn_seams")

        # 3. A jump in an earlier stage wins over a later one (execution order, not magnitude).
        device3 = os.path.join(work, "device-both")
        synthetic_root(
            device3,
            layer,
            position,
            {
                "hc_attn_post": lambda row: [value + 50.0e-6 for value in row],
                "pre_ffn_hidden": lambda row: [value + 500.0e-6 for value in row],
            },
        )
        stage, _worst = bisection(run(host, device3))
        assert stage == "mhc_entry", f"expected the earlier mhc_entry, got {stage}"
        print("  self-test: earlier stage wins over a larger later jump")

        # 4. A deviation inside 10x the bound is not a divergence. Perturb one element at the row's
        # largest magnitude: a global absolute epsilon would move near-zero elements by hundreds of
        # ULP and legitimately trip the declared ULP bound.
        device4 = os.path.join(work, "device-small")
        base_values = [math.sin(index * 0.37) * 0.5 for index in range(40)]
        peak = max(range(len(base_values)), key=lambda index: abs(base_values[index]))

        def small_shift(row):
            row = list(row)
            row[peak] += 5.0e-7
            return row

        synthetic_root(device4, layer, position, {"pre_ffn_hidden": small_shift})
        stage, _worst = bisection(run(host, device4))
        assert stage is None, f"a sub-10x deviation must stay within bounds, got {stage}"
        print("  self-test: sub-10x deviation -> within bounds")

        # 5. A non-finite value diverges even when the arithmetic distance is not meaningful.
        device5 = os.path.join(work, "device-nan")
        synthetic_root(
            device5,
            layer,
            position,
            {"residual_out": lambda row: [float("inf")] + row[1:]},
        )
        stage, _worst = bisection(run(host, device5))
        assert stage == "ffn_epilogue", f"expected ffn_epilogue, got {stage}"
        print("  self-test: non-finite row -> divergent at its stage")

        # 6. The fail-closed paths: no bounds, missing bounds row, missing host copy.
        missing_bounds = os.path.join(work, "absent-bounds.json")
        try:
            load_bounds(missing_bounds)
            raise AssertionError("absent bounds must refuse")
        except SystemExit as refusal:
            assert refusal.code == 2, refusal.code
        incomplete = {"schema": SCHEMA, "bounds": {"residual_in": {"max_abs": 1.0e-6}}}
        incomplete_path = os.path.join(work, "incomplete-bounds.json")
        with open(incomplete_path, "w") as handle:
            json.dump(incomplete, handle)
        try:
            load_bounds(incomplete_path)
            raise AssertionError("an incomplete bounds table must refuse")
        except SystemExit as refusal:
            assert refusal.code == 2, refusal.code
        try:
            compare_roots(os.path.join(work, "absent-host"), device, layer, position, bounds)
            raise AssertionError("absent host copies must refuse")
        except SystemExit as refusal:
            assert refusal.code == 2, refusal.code
        print("  self-test: absent bounds / incomplete bounds / absent host -> exit 2")
        print("QA-LAYER-DIFF SELF-TEST PASS")
    finally:
        import shutil
        shutil.rmtree(work, ignore_errors=True)


sys.exit(main())
PY
}

if [ "$SELF_TEST" = "1" ]; then
  python_helper self-test
  exit 0
fi

if [ -z "$LAYER" ]; then
  usage
fi
case "$LAYER" in
  '' | *[!0-9]*) usage ;;
esac
[ "$LAYER" -ge 0 ] && [ "$LAYER" -lt 40 ] || usage
case "$POSITION" in
  '' | *[!0-9]*) usage ;;
esac

# Fail closed before comparing anything: this tool needs both arms run on a node. The host arm is
# forced by evidence recording, the device arm is SPARKFLASH_LAYER_DEVICE=1, and both need the real
# kernels; a CPU-only checkout cannot produce the observation copies.
if ! gpu_present; then
  fail_closed \
    "no GPU on this host, so neither the host arm nor the device arm can be run and no observation copies exist." \
    "/dev/nvidia*: absent; nvidia-smi: absent." \
    "The host-arm vs device-arm diff runs on a node: run the host arm with" \
    "  SPARKFLASH_BOUNDARY_TRACE=<host-trace> SPARKFLASH_TRACE_POSITIONS=<position>" \
    "(rank 0 writes the rows; evidence recording selects the host arm), run the device arm with" \
    "  SPARKFLASH_LAYER_DEVICE=1 and the dev-gated observation writer pointed at <device-trace>," \
    "then diff them here with --host <host-trace> --device <device-trace>." \
    "Failing closed: exit 2. Nothing was compared."
fi

if [ -z "$HOST_DIR" ] || [ -z "$DEVICE_DIR" ]; then
  usage
fi

set +e
python_helper diff "$LAYER" "$HOST_DIR" "$DEVICE_DIR" "$POSITION" "$BOUNDS"
STATUS=$?
set -e
exit "$STATUS"
