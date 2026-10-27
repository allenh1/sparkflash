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

"""Regenerate the Engram hashing golden fixtures.

Usage (from anywhere):

    cd /home/user/wt/m2 && uv run --with 'numpy,sympy,torch,tokenizers>=0.23,<0.24' \
        python3 crates/sparkflash-engram/tests/fixtures/hash/generate_fixtures.py

Outputs (all next to this script):

    token_map.bin     48-byte header + 129_280 little-endian u32 compressed ids
    token_map.json    human-readable metadata (versions, sha256, counts)
    hash_cases.json   reference hash columns for the streaming scenarios

The expected columns come from the released reference implementation
(/home/user/dsv41-research/inference/engram.py, class NgramHashState) running under
torch.  A second, independent numpy port of the same algorithm is evaluated alongside it
and the two must agree bit for bit; that is what makes the fixture trustworthy rather
than a transcription of my own reading of the reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# here = <worktree>/crates/sparkflash-engram/tests/fixtures/hash
WORKTREE = HERE.parents[4]
TOKENIZER_JSON = WORKTREE / "reference" / "tokenizer.json"
REF_DIR = Path("/home/user/dsv41-research/inference")
GOLDEN_JSON = Path("/home/user/dsv41-research/engram_golden.json")

IMAGE_TOKEN_ID = 129_264
PAD_TOKEN_ID = 2
COMPRESSED_VOCAB_SIZE = 99_092
TOKEN_MAP_MAGIC = b"EGTM"
TOKEN_MAP_VERSION = 1


class _TokenizerShim:
    """Mimic the transformers PreTrainedTokenizerFast surface engram.py uses."""

    def __init__(self, backend):
        self.backend_tokenizer = backend

    def __len__(self) -> int:
        return self.backend_tokenizer.get_vocab_size(with_added_tokens=True)


def import_reference():
    if not (REF_DIR / "engram.py").is_file():
        raise SystemExit(f"reference implementation not found at {REF_DIR / 'engram.py'}")
    sys.path.insert(0, str(REF_DIR))
    import engram

    return engram


def numpy_port(layout, multipliers, primes, offsets, token_map, pad_id, ids, start_pos, full_ids, full_mask):
    """Independent numpy transcription of NgramHashState.forward (no torch).

    full_ids/full_mask are the whole sequence pushed so far (the reference's persistent cache);
    ids is the slice this step appends at start_pos.
    """
    full = np.asarray(full_ids, dtype=np.int64)
    fmask = np.asarray(full_mask, dtype=bool)
    cache = np.where(fmask, np.asarray(token_map, dtype=np.int64)[full], -1)
    n = len(ids)
    positions = np.arange(start_pos, start_pos + n, dtype=np.int64)
    blocked = np.zeros(n, dtype=bool)
    tokens = []
    for shift in range(layout.max_ngram_size):
        source = cache[np.clip(positions - shift, 0, None)]
        blocked = blocked | (positions < shift) | (source == -1)
        tokens.append(np.where(blocked, int(pad_id), source))
    tokens = np.stack(tokens, axis=-1)  # [L, max_ngram_size]
    products = tokens[:, None, :] * multipliers[None, :, :]  # [L, layers, max_ngram_size]
    rolling = products[..., 0]
    hashes = []
    for i in range(1, layout.max_ngram_size):
        rolling = np.bitwise_xor(rolling, products[..., i])
        hashes.append(rolling[..., None] % primes[None, :, i - 1, :])
    out = np.concatenate(hashes, axis=-1) + offsets[None, :, :]
    return out.astype(np.int64)


def build_layout(engram, args):
    layout = engram.EngramLayout.from_args(args)
    if layout is None:
        raise SystemExit("EngramLayout.from_args returned None")
    return layout


class _Args:
    engram_layer_ids = (1, 14)
    engram_max_ngram_size = 4
    engram_n_heads = 8
    engram_vocab_size = 16_000_000
    engram_compressed_vocab_size = COMPRESSED_VOCAB_SIZE
    engram_pad_id = PAD_TOKEN_ID
    engram_num_embeddings = [384_006_168, 384_016_682]
    engram_head_dim = 256
    max_batch_size = 1
    max_seq_len = 512


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-numpy-fallback",
        action="store_true",
        help="keep going if torch is unavailable and use only the numpy port",
    )
    opts = parser.parse_args()

    import tokenizers

    engram = import_reference()
    golden = json.loads(GOLDEN_JSON.read_text())

    backend = tokenizers.Tokenizer.from_file(str(TOKENIZER_JSON))
    tokenizer = _TokenizerShim(backend)
    vocab_size = len(tokenizer)
    token_map, compressed_size = engram.build_compressed_token_map(tokenizer)
    assert compressed_size == COMPRESSED_VOCAB_SIZE, (compressed_size, COMPRESSED_VOCAB_SIZE)
    assert len(token_map) == vocab_size

    args = _Args()
    layout = build_layout(engram, args)

    primes = np.asarray(layout.primes, dtype=np.int64)
    assert primes.shape == (2, 3, 8)
    assert [int(p) for p in primes.reshape(2, -1).sum(axis=1)] == args.engram_num_embeddings
    for layer in range(2):
        expected_flat = [int(p) for order in golden["primes"][layer] for p in order]
        assert [int(p) for p in primes[layer].reshape(-1)] == expected_flat, layer

    multipliers = (
        engram.compute_hash_multipliers(layout.layer_ids, layout.max_ngram_size, compressed_size)
        .numpy()
        .astype(np.int64)
    )
    assert multipliers.tolist() == golden["multipliers"], multipliers.tolist()

    flat = primes.reshape(2, 24)
    offsets = np.stack([np.concatenate([[0], np.cumsum(sizes[:-1])]) for sizes in flat])
    assert offsets[0].tolist() == golden["offsets"][0]

    pad_id = token_map[PAD_TOKEN_ID]

    # ---- scenarios ---------------------------------------------------------
    plain = backend.encode(
        " The quick brown fox jumps over the lazy dog.  Hello, WORLD!", add_special_tokens=False
    ).ids
    # ids that decode to partial UTF-8 byte tokens in this vocabulary
    weird = [128_000, 128_001, 128_002]
    image_span = [100, 200, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 300, 400, 500]
    image_mask = [True, True, False, False, False, True, True, True]
    nine = plain[:9]
    preamble, tail = plain[:5], plain[5:9]

    def step(ids, mask=None):
        return {
            "ids": [int(i) for i in ids],
            "mask": None if mask is None else [bool(m) for m in mask],
        }

    scenarios = [
        ("plain_short_prompt", [step(plain)]),
        ("partial_utf8_ids", [step(weird)]),
        ("image_span_masked", [step(image_span, image_mask)]),
        ("all_masked_pads", [step(nine, [False] * len(nine))]),
        ("split_8_plus_1", [step(nine[:8]), step(nine[8:])]),
        ("decode_start_pos_5", [step(preamble), step(tail)]),
        ("single_step_9", [step(nine)]),
    ]

    try:
        import torch

        have_torch = True
    except ImportError:
        if not opts.allow_numpy_fallback:
            raise
        have_torch = False

    cases = []
    for name, steps in scenarios:
        state = engram.NgramHashState(args, layout, tokenizer) if have_torch else None
        full_ids, full_mask = [], []
        for s in steps:
            full_ids.extend(s["ids"])
            full_mask.extend(s["mask"] if s["mask"] is not None else [True] * len(s["ids"]))
        expected_steps = []
        start = 0
        for s in steps:
            ids = s["ids"]
            mask = s["mask"]
            out = numpy_port(
                layout, multipliers, primes, offsets, token_map, pad_id, ids, start,
                full_ids, full_mask,
            )
            if have_torch:
                t_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
                t_mask = None
                if mask is not None:
                    t_mask = torch.tensor(mask, dtype=torch.bool).unsqueeze(0)
                with torch.inference_mode():
                    t_out = state.forward(t_ids, start, t_mask)
                t_out = t_out.squeeze(0).numpy().astype(np.int64)
                if not np.array_equal(t_out, out):
                    raise SystemExit(f"torch and numpy ports disagree on scenario {name}")
                out = t_out
            expected_steps.append(out.tolist())  # [L, 2, 24]
            start += len(ids)
        cases.append({"name": name, "steps": steps, "expected": expected_steps})

    # ---- write fixtures ----------------------------------------------------
    tokenizer_bytes = TOKENIZER_JSON.read_bytes()
    sha256 = hashlib.sha256(tokenizer_bytes).hexdigest()

    header = TOKEN_MAP_MAGIC + struct.pack("<III", TOKEN_MAP_VERSION, vocab_size, compressed_size)
    header += bytes.fromhex(sha256)
    assert len(header) == 48
    blob = header + struct.pack(f"<{len(token_map)}I", *token_map)
    (HERE / "token_map.bin").write_bytes(blob)
    (HERE / "token_map.json").write_text(
        json.dumps(
            {
                "magic": TOKEN_MAP_MAGIC.decode(),
                "version": TOKEN_MAP_VERSION,
                "tokenizer": str(TOKENIZER_JSON.relative_to(WORKTREE)),
                "tokenizer_sha256": sha256,
                "tokenizer_bytes": len(tokenizer_bytes),
                "tokenizers_python_version": tokenizers.__version__,
                "vocab_size": vocab_size,
                "compressed_vocab_size": compressed_size,
                "pad_id": int(pad_id),
                "token_map_entries": len(token_map),
                "generator": "crates/sparkflash-engram/tests/fixtures/hash/generate_fixtures.py",
                "reference": "/home/user/dsv41-research/inference/engram.py",
                "numpy_port_cross_checked": True,
            },
            indent=2,
        )
        + "\n"
    )

    (HERE / "hash_cases.json").write_text(
        json.dumps(
            {
                "note": "NgramHashState.forward columns from the reference implementation",
                "tokenizer_sha256": sha256,
                "torch_checked": have_torch,
                "primes": primes.tolist(),
                "offsets": offsets.tolist(),
                "multipliers": multipliers.tolist(),
                "cases": cases,
            }
        )
        + "\n"
    )

    print(f"tokenizers {tokenizers.__version__} torch {'yes' if have_torch else 'no (numpy only)'}")
    print(f"vocab_size={vocab_size} compressed={compressed_size} pad_id={pad_id}")
    print(f"token_map.bin {len(blob)} bytes, sha256(tokenizer.json)={sha256}")
    for name, steps in scenarios:
        n = sum(len(s["ids"]) for s in steps)
        print(f"  {name}: {len(steps)} step(s), {n} token(s)")
    print(f"hash_cases.json: {len(cases)} scenarios")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



