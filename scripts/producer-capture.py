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
      --out DIR [--revision SHA] [--rel-l2 1e-2] [--nll 0.02] [--device cpu|cuda]

  --device is the device the producer runs on. A checkpoint that declares the release's fp8
  quantization is dequantized to bfloat16 when it loads on a CPU, so a capture of the release's
  dynamic scheme must name cuda: on a CPU the comparison would be the host's quantized arithmetic
  against an unquantized one, and the report says which the run actually got.

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
    parser.add_argument(
        "--device",
        default="cpu",
        help=(
            "where the producer runs (cpu or cuda). A checkpoint that declares the release's fp8 "
            "quantization is dequantized to bfloat16 on a CPU, so a run of the release's dynamic "
            "scheme needs cuda: it is the device that makes transformers keep the quantizer"
        ),
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


def stored_bf16_modules(checkpoint: Path) -> list[str]:
    """The modules this checkpoint stores without an FP8 scale, from its own index.

    `transformers` builds a fine-grained FP8 model from the quantization config's
    `modules_to_not_convert` list alone: it decides which linears become FP8 modules by name, and
    never reads the checkpoint's stored dtypes. This release stores its KDA projections, `kv_b`,
    the sparse indexer and the whole vision tower in BF16, with no `weight_scale_inv` beside them,
    and its config names none of them. Built as FP8 modules those weights carry a zero inverse
    scale, so every value the producer computes from them is NaN, and the capture measures
    nothing at all.

    The index is the authority, and this reads it rather than a hand-written list: a module whose
    `.weight_scale_inv` is absent is stored unquantised, and its name goes to
    `modules_to_not_convert`. The same reading applies to the real checkpoint, which stores the
    same modules the same way.
    """
    index = checkpoint / "model.safetensors.index.json"
    if not index.is_file():
        return []
    weight_map = json.loads(index.read_text())["weight_map"]
    scales = {name for name in weight_map if name.endswith(".weight_scale_inv")}
    # A norm, an embedding, the head and a bias are not linears a quantizer would take; naming
    # them would be noise in the report. Everything else the index stores without a scale is a
    # BF16 linear.
    not_a_linear = ("norm", "embed_tokens", "lm_head", "ape", "bias")
    bare = {
        name[: -len(".weight")]
        for name in weight_map
        if name.endswith(".weight")
        and name[: -len(".weight")] + ".weight_scale_inv" not in scales
        and not any(part in name for part in not_a_linear)
    }
    return sorted(bare)

def producer_logits(checkpoint: Path, host: dict, dtype_name: str, device: str = "cpu"):
    """The producer's rows for the host's token prompt, from the checkpoint alone."""
    import torch
    from transformers import AutoConfig, Glm5NextForConditionalGeneration

    config = AutoConfig.from_pretrained(str(checkpoint), trust_remote_code=False)
    if getattr(config, "model_type", None) != "glm5_next":
        fail(f"{checkpoint} opens as {getattr(config, 'model_type', None)}, not the producer's glm5_next")
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[dtype_name]
    # The modules the checkpoint stores unquantised, told to the quantizer by name: without them
    # its FP8 modules carry a zero inverse scale and every value derived from them is NaN.
    stored_bf16 = stored_bf16_modules(checkpoint)
    quantized = False
    if device == "cuda":
        if not torch.cuda.is_available():
            fail("--device cuda was asked for and torch reports no CUDA device")
        device_map = {"": "cuda"}
    elif device == "cpu":
        device_map = None
    else:
        fail(f"--device must be cpu or cuda, got {device!r}")
    if stored_bf16:
        quantization = getattr(config, "quantization_config", None)
        if isinstance(quantization, dict):
            named = list(quantization.get("modules_to_not_convert") or [])
            quantization["modules_to_not_convert"] = sorted(set(named) | set(stored_bf16))
        elif quantization is not None:
            named = list(getattr(quantization, "modules_to_not_convert", None) or [])
            quantization.modules_to_not_convert = sorted(set(named) | set(stored_bf16))
    model = Glm5NextForConditionalGeneration.from_pretrained(
        str(checkpoint),
        config=config,
        torch_dtype=dtype,
        attn_implementation="eager",
        ignore_mismatched_sizes=True,
        low_cpu_mem_usage=True,
        **({"device_map": device_map} if device_map else {}),
    )
    quantized = bool(getattr(model, "is_quantized", False))
    if device == "cuda" and getattr(config, "quantization_config", None) and not quantized:
        fail(
            "the checkpoint declares a quantization this producer loaded unquantized: the run "
            "would compare the host's quantized arithmetic with an unquantized one"
        )
    model.eval()
    # The model's own device, which is the GPU under --device cuda: a CPU id tensor against a
    # CUDA model fails inside the first index_select with `Expected all tensors to be on the same
    # device`, which reads as a producer fault and is the caller's.
    where = next(model.parameters()).device
    tokens = host["tokens"]
    with torch.no_grad():
        ids = torch.tensor([tokens], dtype=torch.long, device=where)
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
    where = next(model.parameters()).device
    mask = torch.ones((1, seq_len), dtype=torch.bool, device=where)
    pos = torch.arange(seq_len, device=where).unsqueeze(0)
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
    rows, loaded, model = producer_logits(args.checkpoint, host, dtype_name, args.device)
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
        "device": args.device,
        "producer_quantized": bool(getattr(model, "is_quantized", False)),
        "stored_unquantised_modules": len(stored_bf16_modules(args.checkpoint)),
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
