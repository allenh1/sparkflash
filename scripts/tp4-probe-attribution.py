#!/usr/bin/env python3
"""Reproduce the TP4 probe attribution table from two `tp4-forward-perf-r*.json` rows.

The probe JSON records, on rank 0, one entry per position and one entry per layer. Three totals are
in the artifact and they differ because they measure different things; every table that quotes a
`ms/layer` number must say which one it uses:

  seam-layers   mean of the per-layer `total_ms` over the ordinary layers only. Layers 1 and 14 are
                the enigma layers (their stage sequence differs), so they are excluded to keep a
                per-stage table comparable. This is the attribution basis in the status doc.
  all-layers    the same mean over every layer, enigma included.
  probe-scope   `run.block_summary[layer_total].wall_ms / count`: the probe's own whole-layer
                timer, which includes the probe's per-block bracketing overhead, so it is larger
                than the sum of the block walls.

Measured (m1w37-probe-lean -> m2p1-probe, rank 0): seam-layers 25.722 -> 19.295 ms/layer (-6.427);
probe-scope 27.366 -> 20.803 ms/layer (-6.563). The per-block table uses seam-layers.

Run counters come from `run.loop_total`, an 18-element array whose field order is
`legend.block_fields` (wall_ms, kernel_ns, overhead_ms, launches, syncs, sync_ms, h2d_count,
h2d_bytes, h2d_ms, d2h_count, d2h_bytes, d2h_ms, dtod_count, dtod_bytes, dtod_ms, alloc_count,
free_count, module_load_count). They are NOT in `run.counters`.

Usage:
  python3 scripts/tp4-probe-attribution.py NEW.json|DIR [BASE.json|DIR] [--markdown] [--json]
"""
import json
import sys
from collections import defaultdict

ENIGMA_LAYERS = (1, 14)


def load(path):
    if path.endswith(".json"):
        with open(path) as handle:
            return json.load(handle)
    root = path.rstrip("/")
    for candidate in (
        f"{root}/tp4-forward-perf-r0.json",
        f"{root}/rank0/tp4-forward-perf-r0.json",
    ):
        try:
            with open(candidate) as handle:
                return json.load(handle)
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f"no tp4-forward-perf-r0.json under {root}")


def aggregate(perf):
    all_blocks = defaultdict(float)
    all_blocks_n = defaultdict(int)
    seam_blocks = defaultdict(float)
    seam_blocks_n = defaultdict(int)
    all_tot = []
    seam_tot = []
    for position in perf["positions"]:
        if position.get("head"):
            continue
        for layer in position["layers"]:
            all_tot.append(layer["total_ms"])
            for name, values in layer["blocks"].items():
                all_blocks[name] += values[0]
                all_blocks_n[name] += 1
            if layer["layer"] in ENIGMA_LAYERS:
                continue
            seam_tot.append(layer["total_ms"])
            for name, values in layer["blocks"].items():
                seam_blocks[name] += values[0]
                seam_blocks_n[name] += 1
    probe_scope = {}
    for entry in perf["run"]["block_summary"]:
        if entry.get("name") == "layer_total":
            probe_scope = {
                "wall_ms": entry["wall_ms"],
                "count": entry["count"],
                "ms_per_layer": entry["wall_ms"] / entry["count"] if entry["count"] else None,
            }
            break
    return {
        "positions": sum(1 for p in perf["positions"] if not p.get("head")),
        "layer_positions": len(all_tot),
        "all_layers_total": sum(all_tot) / len(all_tot) if all_tot else 0.0,
        "seam_layers_total": sum(seam_tot) / len(seam_tot) if seam_tot else 0.0,
        "probe_scope": probe_scope,
        "all_blocks": {
            name: (all_blocks[name] / all_blocks_n[name], all_blocks_n[name])
            for name in all_blocks
        },
        "seam_blocks": {
            name: (seam_blocks[name] / seam_blocks_n[name], seam_blocks_n[name])
            for name in seam_blocks
        },
        "counters": counter_table(perf),
        "probe_overhead": perf["run"].get("probe_overhead", {}),
    }


def counter_table(perf):
    legend = perf["legend"]["block_fields"]
    return {name: value for name, value in zip(legend, perf["run"]["loop_total"])}


def rows(stats, base=None):
    names = set(stats["seam_blocks"]) | (set(base["seam_blocks"]) if base else set())
    ordered = sorted(names, key=lambda n: -stats["seam_blocks"].get(n, (0.0, 0))[0])
    out = []
    for name in ordered:
        new_all = stats["all_blocks"].get(name, (None, 0))[0]
        old_all = base["all_blocks"].get(name, (None, 0))[0] if base else None
        new_seam = stats["seam_blocks"].get(name, (None, 0))[0]
        old_seam = base["seam_blocks"].get(name, (None, 0))[0] if base else None
        out.append({
            "block": name,
            "all": new_all,
            "all_base": old_all,
            "seam": new_seam,
            "seam_base": old_seam,
        })
    return out


def delta(a, b):
    if a is None or b is None:
        return None
    return a - b


def markdown(stats, base):
    out = ["| block | seam-layers ms | seam baseline | seam delta | all-layers ms | all baseline |",
           "| :-- | --: | --: | --: | --: | --: |"]
    for row in rows(stats, base):
        def f(value):
            return "--" if value is None else f"{value:.3f}"
        out.append(
            f"| {row['block']} | {f(row['seam'])} | {f(row['seam_base'])} "
            f"| {f(delta(row['seam'], row['seam_base']))} | {f(row['all'])} | {f(row['all_base'])} |"
        )
    seam_base = base["seam_layers_total"] if base else None
    all_base = base["all_layers_total"] if base else None
    out.append(
        f"| **seam-layers total** | {stats['seam_layers_total']:.3f} "
        f"| {(('%.3f' % seam_base) if base else '--')} "
        f"| {('%+.3f' % (stats['seam_layers_total'] - seam_base)) if base else '--'} | | |"
    )
    out.append(
        f"| **all-layers total** | | | | {stats['all_layers_total']:.3f} "
        f"| {(('%.3f' % all_base) if base else '--')} |"
    )
    if stats["probe_scope"]:
        base_probe = base["probe_scope"] if base and base["probe_scope"] else None
        out.append(
            "| **probe-scope layer_total** | | | | "
            f"{stats['probe_scope']['ms_per_layer']:.3f} "
            f"(wall {stats['probe_scope']['wall_ms']:.3f} / count {stats['probe_scope']['count']}) "
            "| "
            + (
                f"{base_probe['ms_per_layer']:.3f} "
                f"(wall {base_probe['wall_ms']:.3f} / count {base_probe['count']})"
                if base_probe
                else "--"
            )
            + " |"
        )
    return "\n".join(out)


def counters_text(stats, base):
    fields = ["launches", "syncs", "alloc_count", "free_count", "module_load_count",
              "h2d_count", "h2d_bytes", "d2h_count", "d2h_bytes", "dtod_count", "dtod_bytes"]
    return {
        field: (stats["counters"].get(field), base["counters"].get(field) if base else None)
        for field in fields
    }


def show(new_path, base_path=None, as_markdown=False):
    stats = aggregate(load(new_path))
    base = aggregate(load(base_path)) if base_path else None
    if as_markdown:
        print(markdown(stats, base))
        print("")
        print("counters (run.loop_total), new -> base")
        for field, (new_value, old_value) in counters_text(stats, base).items():
            print(f"- {field}: {new_value} -> {old_value}")
        print(f"- probe_overhead: {json.dumps(stats['probe_overhead'])}")
        return
    print(f"positions={stats['positions']} layer_positions={stats['layer_positions']}")
    for label, key in (
        ("seam-layers", "seam_layers_total"),
        ("all-layers", "all_layers_total"),
    ):
        suffix = f" (base {base[key]:.3f}, delta {stats[key] - base[key]:+.3f})" if base else ""
        print(f"{label} total {stats[key]:.3f} ms/layer{suffix}")
    if stats["probe_scope"]:
        print(f"probe-scope layer_total {stats['probe_scope']['ms_per_layer']:.3f} ms/layer "
              f"(wall {stats['probe_scope']['wall_ms']:.3f} / count {stats['probe_scope']['count']})")
    print(f"{'block':30s} {'seam':>8s} {'seam-base':>9s} {'d':>7s} {'all':>8s} {'all-base':>8s}")
    for row in rows(stats, base):
        def f(value):
            return "       --" if value is None else f"{value:8.3f}"
        base_seam = base["seam_blocks"].get(row["block"], (None, 0))[0] if base else None
        base_all = base["all_blocks"].get(row["block"], (None, 0))[0] if base else None
        print(
            f"{row['block']:30s} {f(row['seam'])} {f(base_seam)} {f(delta(row['seam'], base_seam))} "
            f"{f(row['all'])} {f(base_all)}"
        )
    print("counters:", json.dumps(counters_text(stats, base)))
    print("probe_overhead:", json.dumps(stats["probe_overhead"]))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) not in (1, 2):
        sys.exit("usage: tp4-probe-attribution.py NEW[.json|DIR] [BASE[.json|DIR]] [--markdown]")
    if "--json" in sys.argv:
        stats = aggregate(load(args[0]))
        stats["base"] = aggregate(load(args[1])) if len(args) == 2 else None
        print(json.dumps(stats, indent=1))
    else:
        show(args[0], args[1] if len(args) == 2 else None, "--markdown" in sys.argv)
