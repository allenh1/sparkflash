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
"""producer-capture.py: the producer's logits on a checkpoint, and the host reference's distance.

Stage 7 of the adding-a-model skill. The producer's own classes (`transformers` at the revision
the layout cites) run the checkpoint in the config's dtype on the same token prompt the host
reference ran; the two logit files are compared inside the tolerance the model doc wrote before
either ran. The tolerance is an argument here, never a constant the script can raise.

  scripts/producer-capture.py --checkpoint DIR --host host-logits.json \
      --out DIR [--revision SHA] [--rel-l2 1e-2] [--nll 0.02]

The host file is `dump-host-logits`'s `sparkflash-host-logits/1` document. The producer file is
written beside it as `producer-logits.json` with the same shape, and `capture.json` holds the
per-position distances, the verdict and the versions that produced it. Exit 0 only on PASS.
"""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path


def fail(message: str) -> "NoReturn":
    print(f"producer-capture: {message}", file=sys.stderr)
    raise SystemExit(2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--host", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--revision",
        default="42efcfde46e668067a74bbd439d33f107b4a7c6c",
        help="the transformers commit the layout cites (recorded, not fetched)",
    )
    parser.add_argument("--rel-l2", type=float, default=1e-2)
    parser.add_argument("--nll", type=float, default=0.02)
    parser.add_argument("--top1", type=int, default=None, help="defaults to every position")
    parser.add_argument(
        "--boundaries",
        type=Path,
        default=None,
        help="optional dump-host-logits --boundaries directory for same-input per-sublayer checks",
    )
    return parser.parse_args()


def load_f32(path: Path) -> list[float]:
    import numpy as np

    return np.fromfile(path, dtype="<f4").tolist()


def argmax(row: list[float]) -> int:
    return max(range(len(row)), key=row.__getitem__)


def log_softmax(row: list[float]) -> list[float]:
    top = max(row)
    shifted = [value - top for value in row]
    total = sum(math.exp(value) for value in shifted)
    return [value - math.log(total) for value in shifted]


def producer_logits(checkpoint: Path, host: dict, dtype_name: str):
    """The producer's rows for the host's token prompt, from the checkpoint alone."""
    import torch
    from transformers import AutoConfig, Glm5NextForConditionalGeneration

    config = AutoConfig.from_pretrained(str(checkpoint), trust_remote_code=False)
    if getattr(config, "model_type", None) != "glm5_next":
        fail(f"{checkpoint} opens as {getattr(config, 'model_type', None)}, not the producer's glm5_next")
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[dtype_name]
    model = Glm5NextForConditionalGeneration.from_pretrained(
        str(checkpoint),
        config=config,
        torch_dtype=dtype,
        attn_implementation="eager",
        ignore_mismatched_sizes=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    tokens = host["tokens"]
    with torch.no_grad():
        ids = torch.tensor([tokens], dtype=torch.long)
        out = model(input_ids=ids, use_cache=False).logits
    rows = out[0].to(torch.float32).tolist()
    if len(rows) != len(tokens):
        fail(f"the producer returned {len(rows)} positions for {len(tokens)} tokens")
    return rows, config, model


def check_boundaries(model, host: dict, boundaries: Path) -> list[dict]:
    """Compare each layer's attention and MLP on the host's own sublayer inputs."""
    import numpy as np
    import torch

    shapes = json.loads((boundaries / "shapes.json").read_text())
    seq_len = len(host["tokens"])
    mask = torch.ones((1, seq_len), dtype=torch.bool)
    pos = torch.arange(seq_len).unsqueeze(0)
    results = []
    shared_topk = None
    with torch.no_grad():
        for i, layer in enumerate(model.model.language_model.layers):
            for kind, in_key, out_key in (
                ("attn", f"L{i}-pre_attention_hidden.f32", f"L{i}-attention_out.f32"),
                ("ffn", f"L{i}-pre_ffn_hidden.f32", f"L{i}-ffn_out.f32"),
            ):
                x = torch.from_numpy(
                    np.fromfile(boundaries / in_key, dtype="<f4").reshape(1, *shapes[in_key])
                ).to(torch.bfloat16)
                host_out = np.fromfile(boundaries / out_key, dtype="<f4").reshape(shapes[out_key])
                if kind == "attn":
                    if hasattr(layer.self_attn, "conv1d"):
                        prod_t = layer.self_attn(x)
                    else:
                        captured = []
                        handle = None
                        if getattr(layer.self_attn, "indexer", None) is not None:
                            handle = layer.self_attn.indexer.register_forward_hook(
                                lambda _m, _i, o: captured.append(o.detach().clone())
                            )
                        try:
                            prod_t = layer.self_attn(
                                x,
                                position_embeddings=None,
                                attention_mask=mask,
                                position_ids=pos,
                                prev_topk_indices=shared_topk,
                            )[0]
                        finally:
                            if handle is not None:
                                handle.remove()
                        if captured:
                            shared_topk = captured[-1]
                else:
                    prod_t = layer.mlp(x)
                    if isinstance(prod_t, tuple):
                        prod_t = prod_t[0]
                prod_out = prod_t[0].to(torch.float32).cpu().numpy()
                rels = [
                    float(np.linalg.norm(host_out[p] - prod_out[p]) / np.linalg.norm(prod_out[p]))
                    for p in range(seq_len)
                ]
                results.append(
                    {
                        "layer": i,
                        "sublayer": kind,
                        "pos0_rel_l2": rels[0],
                        "median_rel_l2": float(np.median(rels)),
                        "max_rel_l2": max(rels),
                    }
                )
        last = len(model.model.language_model.layers) - 1
        r_key = f"L{last}-residual_out.f32"
        r_last = torch.from_numpy(
            np.fromfile(boundaries / r_key, dtype="<f4").reshape(1, *shapes[r_key])
        ).to(torch.bfloat16)
        h = model.model.language_model.norm(model.model.language_model.hc_head(r_last))
        prod_f32_logits = (h.float() @ model.lm_head.weight.float().T)[0].cpu().numpy()
        host_logits = np.array(host["logits"], dtype=np.float32)
        h_rels = [
            float(np.linalg.norm(host_logits[p] - prod_f32_logits[p]) / np.linalg.norm(prod_f32_logits[p]))
            for p in range(seq_len)
        ]
        results.append(
            {
                "layer": last,
                "sublayer": "head_f32",
                "pos0_rel_l2": h_rels[0],
                "median_rel_l2": float(np.median(h_rels)),
                "max_rel_l2": max(h_rels),
            }
        )
    return results


def main() -> int:
    args = parse_args()
    host = json.loads(args.host.read_text())
    if host.get("schema") != "sparkflash-host-logits/1":
        fail(f"{args.host} is not a sparkflash-host-logits/1 document")
    args.out.mkdir(parents=True, exist_ok=True)
    config_text = (args.checkpoint / "config.json").read_text()
    config = json.loads(config_text)
    dtype_name = config.get("text_config", config).get("dtype", "bfloat16")
    rows, loaded, model = producer_logits(args.checkpoint, host, dtype_name)
    sublayers = check_boundaries(model, host, args.boundaries) if args.boundaries else None
    (args.out / "producer-logits.json").write_text(
        json.dumps(
            {
                "schema": "sparkflash-host-logits/1",
                "source": "producer",
                "tokens": host["tokens"],
                "vocab": len(rows[0]),
                "ids": [argmax(row) for row in rows],
                "logits": rows,
            }
        )
    )

    ours = host["logits"]
    table = []
    for position, (mine, theirs) in enumerate(zip(ours, rows)):
        if len(mine) != len(theirs):
            fail(f"position {position}: host row is {len(mine)} wide, producer {len(theirs)}")
        difference = math.sqrt(sum((a - b) ** 2 for a, b in zip(mine, theirs)))
        norm = math.sqrt(sum(b * b for b in theirs))
        relative = difference / norm if norm else difference
        mine_nll = -log_softmax(mine)[argmax(theirs)]
        theirs_nll = -log_softmax(theirs)[argmax(theirs)]
        table.append(
            {
                "position": position,
                "token": host["tokens"][position],
                "host_id": argmax(mine),
                "producer_id": argmax(theirs),
                "same_id": argmax(mine) == argmax(theirs),
                "relative_l2": relative,
                "nll_delta": abs(mine_nll - theirs_nll),
            }
        )
    same = sum(1 for row in table if row["same_id"])
    agreement = same / len(table) if table else 0.0
    largest = max(row["relative_l2"] for row in table)
    mean_nll = statistics.fmean(row["nll_delta"] for row in table)
    top1_bar = len(table) if args.top1 is None else min(args.top1, len(table))
    passed = same >= top1_bar and largest <= args.rel_l2 and mean_nll <= args.nll
    report = {
        "schema": "sparkflash-producer-capture/1",
        "checkpoint": str(args.checkpoint),
        "host": str(args.host),
        "revision": args.revision,
        "config_model_type": loaded.model_type,
        "dtype": dtype_name,
        "positions": len(table),
        "top1_agreement": agreement,
        "top1_same": same,
        "top1_bar": top1_bar,
        "largest_relative_l2": largest,
        "rel_l2_bar": args.rel_l2,
        "mean_abs_nll_delta": mean_nll,
        "nll_bar": args.nll,
        "verdict": "PASS" if passed else "FAIL",
        "positions_table": table,
        **({"sublayers": sublayers} if sublayers is not None else {}),
    }
    (args.out / "capture.json").write_text(json.dumps(report, indent=2))
    print(
        f"producer-capture: {same}/{len(table)} positions agree, largest relative L2 "
        f"{largest:.3e} (bar {args.rel_l2:.1e}), mean |NLL delta| {mean_nll:.3e} "
        f"(bar {args.nll:.2f}): {'PASS' if passed else 'FAIL'}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
