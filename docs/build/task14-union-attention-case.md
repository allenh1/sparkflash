# Task-14: device verification of CSA-union attention (2B-i(a))

Status: implementation, evidence pending the Lead-sequenced node window.
Scope: one attention stage, on engine-produced inputs. This is **not** the 2B sequence
milestone (no sequence, no KV persistence, no chain) and **not** a serving claim.

## What is being verified

The merged format-2 union path of `kernels/flash_mla_csa2_sm121.cu` (window rows plus selected
compressed candidates, one three-pass softmax with the sink term in the denominator) is executed on
a device with inputs produced by the engine reference forward, not by the harness. The comparison is
the bf16 device output against the bf16 rounding of the engine reference attention output for one
query position, with the 2e-2 bar the 2B contract states.

Inputs come from the released 130-token prompt (last prefill position 129), captured for decoder
layers 21, 30 and 39. The harness also runs kernel format 1 (engine window format) over the same
window rows. Format 1 cannot attend the selected compressed candidates by construction, so its
distance from the reference is the contribution the union carries - the structural-exclusion
demonstration, and the reason format 2 exists.

## Part A: the engine-produced attention case fixture

`crates/sparkflash-engine/examples/attention_case_fixture.rs` runs the real reference forward and
records, per requested layer at one prefill query:

- the rotated query `[heads][head_dim]` f32 and the per-head sink;
- the window ids the reference attends, and the window KV rows it stored (f32 after the engine's FP8
  round trip) with the engine-format bytes the device takes: 512 B E4M3 codes plus 16 B E8M0 scales
  per row, produced by bf16-rounding each stored row and running the engine's released activation
  quantiser (the same rule as `cuda_harness::engine_window_row`);
- the compressed ids the reference selected in its shared index space (NO_INDEX entries included),
  the engine buffer rows, and each reachable store row in the device main-KV layout: 256 B packed
  FP4 plus 32 B E4M3 scales per 16 channels, from `quantize_main_kv`;
- the selection mapped to absolute token positions (store row times the source layer's compression
  ratio), which is the kernel's selected list;
- the reference attention output, pre-inverse-RoPE, `[heads][head_dim]` f32.

The capture hook is `crates/sparkflash-engine/src/forward/attention_case.rs`: a thread-local recorder
that is inactive unless explicitly armed. It copies values the reference pass has already computed
and adds no arithmetic, so the reference output is unchanged whether it fires or not. The engine
crate's own tests run with it inactive.

The generator refuses to mix provenance: `--provenance ci` requires the tiny finite-weight
configuration, `--provenance node` requires the released checkpoint. A node fixture records argv,
host, binary sha256, checkpoint path, notes, and the golden sequence fixture it cross-checked
(path, sha256, prompt equality, prefill next-token agreement, logits delta).

### Command (node, CPU only, inside the Lead-sequenced window)

```
target/aarch64-unknown-linux-gnu/release/examples/attention_case_fixture \
  --config released \
  --checkpoint /home/user/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170 \
  --prompt-tokens 130 --seed 7 --layers 21,30,39 \
  --golden-fixture /home/user/phase2-evidence/sequence-130-released.json \
  --provenance node \
  --note "task-14 Part A: engine reference attention case for format-2 union device check" \
  --out /home/user/phase2-evidence/attention-case-released.json
```

The generator writes the fixture, prints its sha256 and size, and prints the golden prefill
cross-check (the same prompt and configuration on the same host should agree exactly; the golden
fixture was produced on the same node from the same checkpoint). The fixture is then copied to the
dev host with its sha256 and size; the checkpoint is not copied.

## Part B: the device comparison

`crates/sparkflash-kernels/examples/m8_union_attention.rs` is a new, isolated example on
`origin/main` that uses the typed `cuda_driver`/`cuda_harness` surface directly
(`load_flash_mla`, `flash_mla_params` with format word 2, `launch_flash_mla`); it does not use the
m8 sequence chain. It validates the fixture, rebuilds every window row with
`cuda_harness::engine_window_row` and requires byte equality with the fixture, checks the main-KV
store-row decode against the captured engine f32 rows, places each store row at its absolute
position in the position-indexed device buffers, and launches the kernel once per format. Each
launch runs twice and the two output byte strings must match.

The report records: fixture sha256 and provenance echo, cubin path/size/sha256, binary sha256, host,
argv, per-layer input-seam maxima (window decode vs stored f32, store decode vs engine f32), format-2
comparison (exact bf16 count, max abs delta, max ULP, max relative delta, non-finite count, pass),
format-1 window-only comparison against the same reference, format-1 vs format-2 contribution, and
the deterministic-launch result.

### Cubin (pinned toolkit, sm_121a)

Build in the pinned container on the dev host, not on the node:

```
docker run --rm -v <repo>:/repo:ro -v <evidence>:/out nvidia/cuda:13.0.3-devel-ubuntu22.04 bash -lc '
  cd /repo && nvcc -cubin -arch=sm_121a -I kernels -o /out/flash_mla_csa2_sm121.cubin kernels/flash_mla_csa2_sm121.cu'
```

The task-14 cubin was built from merged main `d748b98` with nvcc V13.0.88
(cuda_13.0.r13.0/compiler.36424714_0): 60,792 B, sha256
`64fc2944d510885490ef1caeb680600f30b1d552c6ec31b0f6ed840ad0f09d2b`. That is the same digest the
kernel baseline records for this kernel, so the cubin is the reviewed build.

### Command

```
target/aarch64-unknown-linux-gnu/release/examples/m8_union_attention \
  --fixture /home/user/phase2-evidence/attention-case-released.json \
  --cubin-dir /home/user/phase2-evidence/task14 \
  --layers 21,30,39 \
  --out /home/user/phase2-evidence/task14/union-attention-report.json \
  --note "task-14 Part B: device format-2 union check on engine-produced inputs"
```

`--dry-run` performs the whole host-side validation and buffer build without touching CUDA; it is
the dev-host check of the fixture and the mapping.

## Honesty

- One attention stage on engine-produced inputs. The fixture is host evidence; the report is device
  evidence for that stage only.
- The device inputs are the store formats: window rows bf16 then FP8 (E4M3/E8M0), compressed rows
  FP4 with E4M3 scales, query bf16 on the device side. The reference output used f32 inputs, so the
  comparison carries that quantisation seam; it is not bit-identity with the released kernel.
- Format 2 is the merged kernel behaviour; this changes no format-2 semantics and edits no existing
  example or reference module.
- No token or logits claim: the golden fixture cross-check in Part A is provenance for the same
  prompt, not a new comparison bar.
