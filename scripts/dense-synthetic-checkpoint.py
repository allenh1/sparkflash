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

"""dense-synthetic-checkpoint.py: a random dense BF16 checkpoint of a family no pack names.

The shared dense device arm's served gate (scripts/served-gate.py) checks the serving contract:
fused lanes equal alone, prompt parts, kept conversations and the rewind point. None of that
needs a trained model, only a checkpoint the arm takes and a tokenizer the server can speak, so
this writes one with random weights beside a ChatML tokenizer copied from TOKENIZER_DIR (its
tokenizer.json, vocab.json, chat template and generation config):

  dense-synthetic-checkpoint.py TOKENIZER_DIR OUT_DIR [--layers 8] [--hidden 1024]

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


def write(path, tensors):
    """A safetensors file of `tensors`: (name, shape, bytes) in order."""
    header, offset = {}, 0
    for name, shape, data in tensors:
        header[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    with open(path, "wb") as out:
        out.write(struct.pack("<Q", len(blob)))
        out.write(blob)
        for _, _, data in tensors:
            out.write(data)


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
    args = parser.parse_args()
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
            (at + "self_attn.q_proj.weight", [q, hidden], bf16_random(q * hidden, inner(hidden))),
            (at + "self_attn.k_proj.weight", [kv, hidden], bf16_random(kv * hidden, inner(hidden))),
            (at + "self_attn.v_proj.weight", [kv, hidden], bf16_random(kv * hidden, inner(hidden))),
            (at + "self_attn.o_proj.weight", [hidden, q], bf16_random(hidden * q, inner(q))),
            (at + "mlp.gate_proj.weight", [args.intermediate, hidden],
             bf16_random(args.intermediate * hidden, inner(hidden))),
            (at + "mlp.up_proj.weight", [args.intermediate, hidden],
             bf16_random(args.intermediate * hidden, inner(hidden))),
            (at + "mlp.down_proj.weight", [hidden, args.intermediate],
             bf16_random(hidden * args.intermediate, inner(args.intermediate))),
        ]
    write(os.path.join(args.out_dir, "model.safetensors"), tensors)
    print("wrote %s: %d layers, hidden %d, vocab %d, %d word tokens in the head" % (
        args.out_dir, args.layers, hidden, size, len(words)))


if __name__ == "__main__":
    main()
