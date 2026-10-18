# Phase 2 golden reference fixture (node, main f72738b)

Status: **the real-checkpoint sequence fixture exists.** It was produced on the node `node0` on
2026-10-18 from main `f72738b`, in a Lead-sequenced vLLM-down window, and it is the **reference
truth** for milestone 2B/2C. It is **not** a device run and it closes no exit criterion: 2B's device
plumbing ran on the node on 2026-10-18 but the device comparison (tokens then logits against this
fixture) has not run, and Phase 2 exit criterion 1
([roadmap-to-serving.md](roadmap-to-serving.md) section 5.5.1) remains OPEN. The CSA-union format-2
semantics that can reproduce this fixture's attended set are [DECISIONS.md](../DECISIONS.md) D24; the
decision that this fixture is the truth is D25.

Sources: the task-9 2A/2B plan; memorix observations #110-#112; the run's evidence files under
`/home/user/phase2-evidence/devhost-copy-f72738b/` (`status-f72738b.txt`, `run-f72738b.log`,
`run-info-f72738b.txt`, `io-f72738b.log`); the committed
[phase2-sequence-harness.md](phase2-sequence-harness.md) sections 3-4; [DECISIONS.md](../DECISIONS.md)
D24 and D25.

## 1. What the fixture is

The engine's CPU reference forward (`ReferenceWeights`) over the **real released checkpoint**, not a
synthetic source and not a device output. One sequence: a **130-token prompt** (it crosses the
128-token sliding window) followed by **8 greedy decode steps**. Schema version 1. The prompt is the
generator's deterministic prompt for seed 7 (its 130 ids are recorded in the file).

| field | value |
|:--|:--|
| schema_version | 1 |
| generator | `sparkflash-engine sequence_fixture v1` |
| source | `reference_checkpoint`, seed 0, released checkpoint snapshot |
| config_label | `released` (40 backbone layers, 3 MTP stages, window 128) |
| prompt | 130 token ids, recorded in the file |
| decode_steps | 8 |
| compared_positions | `[129, 130, ..., 137]` (last prefill position + every decode position) |
| prefill next_token | **491** |
| decode next_tokens | **367, 201, 80, 3302, 201, 54, 201, 4341** |
| logits | the prefill step's full 129,280 f32 logits; every step has a sha256 digest |
| kv_rows | empty in 2A; 2C populates them |
| provenance | section 3 |

## 2. Identity and where it lives

The same bytes exist in two places, both sha-verified:

| copy | path | bytes | sha256 |
|:--|:--|--:|:--|
| node (written by the run) | `/home/user/phase2-evidence/sequence-130-released.json` | 2,536,194 | `e6df8290dcc9cfc96af27ad1a4da165e5f3d8d4e9aac0f98819b0030b738b308` |
| dev host (copy of record) | `/home/user/phase2-evidence/devhost-copy-f72738b/sequence-130-released.json` | 2,536,194 | same |

The node-side sha is recorded in `status-f72738b.txt`; the dev-host copy was re-hashed with
`sha256sum` when this note was written and matched. The sha256 and size are the fixture's identity: a
run that produces different bytes is a different fixture even if the tokens match.

## 3. Provenance recorded in the fixture

- **host** `node0`, `Linux node0 6.17.0-1029-nvidia ... aarch64`, 20 CPUs, run timestamp
  `2026-10-18T14:11:28Z`, working directory `/home/user/phase2-evidence`;
- **source main** `f72738b5a63996a8425d11d744c4abb5f321e007` (the !60 parallel reference pass; !57 and
  !58 were already on main, and the run does not depend on !61);
- **binary** `/home/user/phase2-evidence/sequence_fixture_f72738b_aarch64`,
  `aarch64-unknown-linux-gnu` release, **7,265,936 B**, sha256
  `ed3838df5ca28722a19d6dda689d7b90c4dd67c8faf205cafcc3469dcb459ffa`;
- **toolchain** rustc 1.98.1 (48a229cea 2026-09-01), cargo 1.98.1 (797e8a9bc 2026-08-05), zig 0.17.0;
- **checkpoint** `/home/user/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170`;
- **command** the shell-runnable argv is stored verbatim in `provenance.command`; the generator also
  records host, cwd and binary sha itself;
- **notes** 8 provenance notes: the generator's three defaults (what the fixture is, the CED caveat,
  the run cwd) plus five `--note` entries (main, binary sha+size, toolchain, the squash note, the
  fix-commit note).

The recorded binary sha is of the staged node copy. As with the dev-store probe binary
([phase2-device-kv-store.md](phase2-device-kv-store.md) section 9), an aarch64 binary's hash is
path-dependent, so the binary is provenance for the run, not a build artifact a rebuild must
reproduce byte-for-byte.

## 4. The measured run (node `node0`)

| measure | value | source |
|:--|--:|:--|
| wall clock | **613.54 s** (`10:13.54`) | `/usr/bin/time -v` in `run-f72738b.log` |
| CPU | 501 % | `run-f72738b.log` |
| user / system time | 2,214.85 s / 864.52 s | `run-f72738b.log` |
| peak RSS | 12,364,220 KiB | `run-f72738b.log` |
| rchar | 268,070,647,545 B | final sample, `io-f72738b.log` |
| read_bytes | 124,332,576,768 B | final sample, `io-f72738b.log` — **warm cache, see below** |
| exit / abort | 0 / none | `status-f72738b.txt` |
| fixture | 2,536,194 B, sha256 `e6df8290...` | `status-f72738b.txt` |

Two caveats, both required when these numbers are quoted:

1. **The prefill/decode split is inferred, not measured.** The run did not instrument the phases. The
   io log shows eight roughly-equal read bursts of about **31.3 s** during the decode phase, which puts
   the eight decodes at roughly **251.9 s** and the 130-token prefill at roughly **361.6 s**
   (613.54 - 251.9). Quote the split as inferred, or quote only the total.
2. **`read_bytes` is not a cold-cache number.** It was measured with a warm page cache and must not be
   quoted as the fixture's cold-cache I/O requirement.

## 5. How it was produced, and how to re-verify it

Generation (the executable form on the dev host is
`/home/user/phase2-evidence/run-golden-f72738b.sh`, not in the repo; run only in a Lead-sequenced
vLLM-down window):

1. Cross-build the example from main `f72738b`:
   `cargo build --release --target aarch64-unknown-linux-gnu -p sparkflash-engine --example sequence_fixture`.
2. `scp` the binary to the node as `/home/user/phase2-evidence/sequence_fixture_f72738b_aarch64`
   and `sha256sum` it on the node (`ed3838df...`, 7,265,936 B).
3. On the node, from `/home/user/phase2-evidence`, run it under `/usr/bin/time -v` with the
   recorded argv:

   ```text
   ./sequence_fixture_f72738b_aarch64 --config released \
     --checkpoint /home/user/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170 \
     --prompt-tokens 130 --decode-steps 8 --seed 7 --logits prefill --provenance node \
     --out /home/user/phase2-evidence/sequence-130-released.json \
     --note 'main=f72738b5a63996a8425d11d744c4abb5f321e007' \
     --note 'binary sha256=ed3838df5ca28722a19d6dda689d7b90c4dd67c8faf205cafcc3469dcb459ffa size=7265936' \
     --note 'toolchain rustc 1.98.1 (48a229cea 2026-09-01), cargo 1.98.1 (797e8a9bc 2026-08-05), zig 0.17.0, aarch64-unknown-linux-gnu release' \
     --note 'squash: !60 squash-merged as main f72738b; !58 and !57 already on main; does not depend on !61' \
     --note 'fix commit f72738b; argv/host/cwd/binary sha recorded by the generator'
   ```

4. The generator prints the file's own sha256 and size; the wrapper records the exit status and the
   node-side sha (`status-f72738b.txt`).
5. Copy the fixture and the logs to the dev host (`scp`), then `sha256sum` the copy and compare.

Re-verification needs no checkpoint and no node:

```text
sha256sum /home/user/phase2-evidence/sequence-130-released.json
# e6df8290dcc9cfc96af27ad1a4da165e5f3d8d4e9aac0f98819b0030b738b308  (2,536,194 B)
sha256sum /home/user/phase2-evidence/devhost-copy-f72738b/sequence-130-released.json
# same
```

The comparison contract consumes the file through
`sparkflash_engine::sequence_fixture::SequenceFixture` (`validate`, then `compare_steps` token-first
and `compare_logits` at `LOGITS_ABS_BAR` = 2e-2; [phase2-sequence-harness.md](phase2-sequence-harness.md)
section 4). The fixture is comparable only on its own `compared_positions`: the engine reference does
not implement CED (D7), and positions older than the window have no comparable decoder output.

## 6. What it is not, and the rule that protects it

* Not a device run: no kernel launch, no CUDA, no GPU output, nothing here is device evidence.
* Not exit-criterion evidence: no device run has compared tokens or logits against this fixture (the
  2B plumbing run of 2026-10-18T18:32Z measured rotation and dispatch only), so Phase 2 exit
  criterion 1 stays OPEN; criteria 2-5 are not claimed either.
* Not a CED-parity claim: only the compared set is comparable, and the harness supplies the
  encoder/decoder split.
* Not a claim that the device will match: the fixture is what the device run is measured against.

**Rule: this fixture is reference truth and must not be regenerated as a side effect of a device
change.** If a device run disagrees, the device work is wrong or the gap is real; the fixture is not
edited to make it pass. A legitimate regeneration (a schema change or a corrected reference) is a
separate Lead decision that records a new sha256/size here and marks the old one superseded - never a
by-product of a device milestone.

## References

- main `d3dcc3f` (the CSA-union format-2 merge):
  [commit](https://example.com/sparkflash/-/commit/d3dcc3fea68061ddf0b47e02eb6a1abf1db6fbed)
- golden-run main `f72738b`:
  [commit](https://example.com/sparkflash/-/commit/f72738b5a63996a8425d11d744c4abb5f321e007)
- MR !62 (CSA-union kernel, merged):
  [merge request 62](https://example.com/sparkflash/-/merge_requests/62)
- MR !63 (2B harness, merged as 1ceeb54):
  [merge request 63](https://example.com/sparkflash/-/merge_requests/63)
- pipeline 2763 (!62, green):
  [pipeline 2763](https://example.com/sparkflash/-/pipelines/2763)
- [phase2-sequence-harness.md](phase2-sequence-harness.md) sections 3-5;
  [phase2-device-kv-store.md](phase2-device-kv-store.md) section 9;
  [roadmap-to-serving.md](roadmap-to-serving.md) section 5.5; [DECISIONS.md](../DECISIONS.md) D24, D25.
