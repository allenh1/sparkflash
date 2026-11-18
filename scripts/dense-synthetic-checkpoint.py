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

"""dense-synthetic-checkpoint.py: a random checkpoint of a family no pack names.

The shared dense device arm's served gate (scripts/served-gate.py) checks the serving contract:
fused lanes equal alone, prompt parts, kept conversations and the rewind point. None of that
needs a trained model, only a checkpoint the arm takes and a tokenizer the server can speak, so
this writes one with random weights beside a ChatML tokenizer copied from TOKENIZER_DIR (its
tokenizer.json, vocab.json, chat template and generation config):

  dense-synthetic-checkpoint.py TOKENIZER_DIR OUT_DIR [--layers 8] [--hidden 1024]
      [--experts 32 --top-k 4 --expert-intermediate 512 [--shared-intermediate 1024]] [--fp8]
      [--gdn]

With --experts every layer routes: a softmax router renormalising its top-k picks over that many
BF16 experts, and a sigmoid-gated shared expert when --shared-intermediate names its width (the
Qwen MoE spelling), so the served gate covers the arm's routed layers too. With --fp8 every
projection and expert is stored as the Qwen3 FP8 releases store them: E4M3 codes under one BF16
`weight_scale_inv` a 128 x 128 tile (the embedding, head, norms and router stay BF16). With
--gdn every even layer is a Gated DeltaNet (Qwen3.5's `linear_attn` spelling: four key heads and
eight value heads of 128, four conv taps, a SiLU output gate) between attention layers, so the
gate covers the recurrent state in the lanes and the rewind point's copy of it.

The head scores only plain word tokens (a space and three to eight lowercase letters), at a
scale that outweighs every other token's zero, so greedy and sampled replies are words that
re-encode to the ids the server picked, and no reply writes a control token. Nothing here needs
numpy: the random BF16 values are random bits with the exponent pinned by a byte table.
"""

import argparse
import json
import os
import re
import shutil
import struct


def bf16_random(count, exponent):
    """`count` BF16 values of random sign, each within a factor of four of 2**exponent."""
    raw = bytearray(os.urandom(count * 2))
    # Little endian: the high byte holds the sign and the exponent's top seven bits, the low
    # byte the exponent's last bit and the mantissa. Pin the top seven exponent bits.
    biased = exponent + 127
    top = (biased >> 1) & 0x7F
    table = bytes(((byte & 0x80) | top) for byte in range(256))
    raw[1::2] = bytes(raw[1::2]).translate(table)
    return bytes(raw)


def bf16_constant(count, value):
    word = struct.pack("<f", value)[2:]
    return word * count


def e4m3_random(count):
    """`count` E4M3 codes of random sign and mantissa, each in [1, 2)."""
    # Sign bit, the exponent field pinned at the bias (7), and three mantissa bits.
    table = bytes(((byte & 0x87) | 0x38) for byte in range(256))
    return os.urandom(count).translate(table)


def write(path, tensors):
    """A safetensors file of `tensors`: (name, shape, bytes[, dtype]) in order, BF16 unless named."""
    header, offset = {}, 0
    for name, shape, data, *dtype in tensors:
        header[name] = {"dtype": dtype[0] if dtype else "BF16", "shape": shape,
                        "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    with open(path, "wb") as out:
        out.write(struct.pack("<Q", len(blob)))
        out.write(blob)
        for _, _, data, *_ in tensors:
            out.write(data)


def projection(name, rows, columns, exponent, fp8):
    """One `[rows, columns]` projection about 2**exponent: BF16, or with `fp8` E4M3 codes in
    [1, 2) under a BF16 inverse scale of 2**exponent a 128 x 128 tile."""
    if not fp8:
        return [(name, [rows, columns], bf16_random(rows * columns, exponent))]
    grid = [-(-rows // 128), -(-columns // 128)]
    return [
        (name, [rows, columns], e4m3_random(rows * columns), "F8_E4M3"),
        (name.replace(".weight", ".weight_scale_inv"), grid,
         bf16_constant(grid[0] * grid[1], 2.0 ** exponent)),
    ]


def mlp(prefix, width, hidden, inner, fp8):
    """One SwiGLU MLP of `width` under `prefix`."""
    return (projection(prefix + "gate_proj.weight", width, hidden, inner(hidden), fp8)
            + projection(prefix + "up_proj.weight", width, hidden, inner(hidden), fp8)
            + projection(prefix + "down_proj.weight", hidden, width, inner(width), fp8))


def gdn(prefix, hidden, inner, fp8, key_heads=4, heads=8, dim=128, taps=4):
    """One Gated DeltaNet layer under `prefix`: the fused q/k/v, output gate and out projections
    (FP8 with `fp8`), the beta and decay projections, a `[width, 1, taps]` conv, decay rates of
    exp(-0.5) and the gated norm, all BF16."""
    width = 2 * key_heads * dim + heads * dim
    return (projection(prefix + "in_proj_qkv.weight", width, hidden, inner(hidden), fp8)
            + projection(prefix + "in_proj_z.weight", heads * dim, hidden, inner(hidden), fp8)
            + projection(prefix + "in_proj_b.weight", heads, hidden, inner(hidden), False)
            + projection(prefix + "in_proj_a.weight", heads, hidden, inner(hidden), False)
            + [(prefix + "conv1d.weight", [width, 1, taps], bf16_random(width * taps, -2)),
               (prefix + "A_log", [heads], bf16_constant(heads, -0.5)),
               (prefix + "dt_bias", [heads], bf16_constant(heads, 0.0)),
               (prefix + "norm.weight", [dim], bf16_constant(dim, 1.0))]
            + projection(prefix + "out_proj.weight", hidden, heads * dim, inner(heads * dim), fp8))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("tokenizer_dir")
    parser.add_argument("out_dir")
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--intermediate", type=int, default=2816)
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--experts", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--expert-intermediate", type=int, default=512)
    parser.add_argument("--shared-intermediate", type=int, default=0)
    parser.add_argument("--fp8", action="store_true")
    parser.add_argument("--gdn", action="store_true")
    args = parser.parse_args()
    fp8 = args.fp8
    os.makedirs(args.out_dir, exist_ok=True)
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                 "chat_template.jinja", "generation_config.json"):
        source = os.path.join(args.tokenizer_dir, name)
        if os.path.isfile(source):
            shutil.copyfile(source, os.path.join(args.out_dir, name))
    with open(os.path.join(args.tokenizer_dir, "tokenizer.json")) as text:
        tokenizer = json.load(text)
    vocab = tokenizer["model"]["vocab"]
    size = max(vocab.values()) + 1
    for token in tokenizer.get("added_tokens", []):
        size = max(size, token["id"] + 1)
    size += -size % 32
    words = sorted(i for t, i in vocab.items() if re.fullmatch("Ġ[a-z]{3,8}", t))[:3000]
    generation = {}
    path = os.path.join(args.tokenizer_dir, "generation_config.json")
    if os.path.isfile(path):
        with open(path) as text:
            generation = json.load(text)
    hidden, head_dim = args.hidden, args.hidden // args.heads
    config = {
        "model_type": "acme_dense",
        "architectures": ["AcmeDenseForCausalLM"],
        "hidden_size": hidden,
        "intermediate_size": args.intermediate,
        "num_hidden_layers": args.layers,
        "num_attention_heads": args.heads,
        "num_key_value_heads": args.kv_heads,
        "head_dim": head_dim,
        "vocab_size": size,
        "max_position_embeddings": args.context,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "hidden_act": "silu",
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "bos_token_id": generation.get("bos_token_id"),
        "eos_token_id": generation.get("eos_token_id"),
    }
    if args.experts:
        config.update({
            "model_type": "acme_moe",
            "architectures": ["AcmeMoeForCausalLM"],
            "num_experts": args.experts,
            "num_experts_per_tok": args.top_k,
            "moe_intermediate_size": args.expert_intermediate,
            "norm_topk_prob": True,
        })
        if args.shared_intermediate:
            config["shared_expert_intermediate_size"] = args.shared_intermediate
    linear = lambda layer: args.gdn and layer % 2 == 0
    if args.gdn:
        config.update({
            "model_type": config["model_type"].replace("acme_", "acme_hybrid_"),
            "layer_types": ["linear_attention" if linear(layer) else "full_attention"
                            for layer in range(args.layers)],
            "linear_conv_kernel_dim": 4,
            "linear_num_key_heads": 4,
            "linear_num_value_heads": 8,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
        })
    if fp8:
        config["quantization_config"] = {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
        }
    with open(os.path.join(args.out_dir, "config.json"), "w") as out:
        json.dump(config, out, indent=2)
    # Each projection's magnitude about one over the root of its inner width.
    inner = lambda width: max(-12, round(-0.5 * (width.bit_length() - 1)) - 1)
    tensors = [
        ("model.embed_tokens.weight", [size, hidden], bf16_random(size * hidden, -1)),
        ("model.norm.weight", [hidden], bf16_constant(hidden, 1.0)),
    ]
    head = bytearray(size * hidden * 2)
    row = hidden * 2
    for token in words:
        head[token * row:(token + 1) * row] = bf16_random(hidden, -2)
    tensors.append(("lm_head.weight", [size, hidden], bytes(head)))
    q, kv = args.heads * head_dim, args.kv_heads * head_dim
    for layer in range(args.layers):
        at = "model.layers.%d." % layer
        tensors += [
            (at + "input_layernorm.weight", [hidden], bf16_constant(hidden, 1.0)),
            (at + "post_attention_layernorm.weight", [hidden], bf16_constant(hidden, 1.0)),
        ]
        if linear(layer):
            tensors += gdn(at + "linear_attn.", hidden, inner, fp8)
        else:
            attn = at + "self_attn."
            tensors += projection(attn + "q_proj.weight", q, hidden, inner(hidden), fp8)
            tensors += projection(attn + "k_proj.weight", kv, hidden, inner(hidden), fp8)
            tensors += projection(attn + "v_proj.weight", kv, hidden, inner(hidden), fp8)
            tensors += projection(attn + "o_proj.weight", hidden, q, inner(q), fp8)
        if not args.experts:
            tensors += mlp(at + "mlp.", args.intermediate, hidden, inner, fp8)
            continue
        tensors.append((at + "mlp.gate.weight", [args.experts, hidden],
                        bf16_random(args.experts * hidden, inner(hidden))))
        for expert in range(args.experts):
            tensors += mlp(at + "mlp.experts.%d." % expert, args.expert_intermediate, hidden,
                           inner, fp8)
        if args.shared_intermediate:
            tensors += mlp(at + "mlp.shared_expert.", args.shared_intermediate, hidden, inner, fp8)
            tensors.append((at + "mlp.shared_expert_gate.weight", [1, hidden],
                            bf16_random(hidden, inner(hidden))))
    write(os.path.join(args.out_dir, "model.safetensors"), tensors)
    print("wrote %s: %d layers, hidden %d, vocab %d, %d word tokens in the head%s%s%s" % (
        args.out_dir, args.layers, hidden, size, len(words),
        ", %d experts (top %d)" % (args.experts, args.top_k) if args.experts else "",
        ", FP8 projections" if fp8 else "",
        ", Gated DeltaNet on even layers" if args.gdn else ""))


if __name__ == "__main__":
    main()
