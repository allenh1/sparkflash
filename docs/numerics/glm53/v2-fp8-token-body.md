# GLM-5.3-Flash, numerics version 2: decode on the FP8 E4M3 token body

Version 2 of the world-4 sm_121a configuration moves GLM's one-to-eight-row block-scaled FP8
projections from the tiled row GEMV (`q4e_gemv_tiled_e4m3_block`) to the shared tile's few-row
body (`glm_decode_fp8b128_rows`). It is the first declared change to GLM's arithmetic after the
shipped version 1 (the E4M3 attention window, [v1-e4m3-window.md](v1-e4m3-window.md)).

## What changed

A decode step, a speculative verify and any FP8 projection of at most eight rows now launch the
body on the same E4M3 codes, the same f32 128 x 128 block scales and the same f32 activations the
GEMV read. Nothing is converted: the body takes the GEMV's operands and writes its part-by-part
output layout, one launch for each part of a fused projection. The body runs the prompt
tile's chain - the E4M3 codes widened to bf16, the activation split into a bf16 head and
remainder, two MMAs a 16-column step with K ascending, and the 128-column block scale folded in
ascending block order - so its row is the prompt tile's row bit for bit. That tile is
`glm_prefill_fp8b128_mma`, which takes f32 activations; the shipped release's bulk prefill
takes `glm_prefill_q8_mma` for its dynamic E4M3 activations, so this change does not by itself
make a continued turn equal a cold q8 prefill.

The routing boundary is the arm's existing one: `MAX_ROWS` is eight, one to eight rows take the
body, and a larger batch keeps the bulk tensor-core GEMM. Projections whose shape the body does
not take (a BF16 weight, a K run that is not whole 128-column blocks, or a part that is not
whole eight-column outputs) keep the tiled GEMV. The routed experts' `q4e_expert_gemv_e4m3_block`
is outside this change and still serves the token rows.

## What was measured

- **The body is the tile's row.** On the sm_121a device, for every token count from 1 to 8, a
  two-part fused projection launched through the arm's own body path is bit-equal, row by row,
  to `glm_prefill_fp8b128_mma` over the same weights, scales and activations
  (`device::tests::a_decode_step_on_the_token_body_is_the_prefill_tiles_row`, ignored unless a
  CUDA device is present). The kernel-level harness
  (`crates/sparkflash-kernels/tests/token_bodies_device.rs`) also holds the body to the tile at
  1, 2, 4 and 8 rows, on a 256 x 512 shape and a 96 x 384 partial tile.
- **The routing boundary.** A CPU test holds the band to one through eight rows and the ninth to
  the bulk path, and rejects a BF16 weight, a part that is not whole eight-column writes, and a
  part whose block scale does not start on a 128-row boundary
  (`device::tests::the_fp8_token_body_takes_one_to_eight_rows`).

The device run, on the sm_121a host:

```text
$ cargo test -p sparkflash-model-glm5-next --lib a_decode_step_on_the_token_body_is_the_prefill_tiles_row -- --ignored --nocapture
running 1 test
device 0: NVIDIA RTX Spark N1X (6144-core Blackwell RTX GPU) of 1 (total 118815 MiB, MemAvailable 106777 MiB)
test device::tests::a_decode_step_on_the_token_body_is_the_prefill_tiles_row ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 74 filtered out; finished in 0.44s
```

The device arm's device-versus-host fixture moves with the declared change: with the FP8
weights' inputs unquantised, the three worlds' worst seams are 1.3e-5, 1.8e-5 and 1.3e-5
(`TOKEN_BODY_REL_L2_BAR` is 2e-5); with the release's activation scheme the host quantises each
input per 128 columns and the body takes the f32 row, so the median seam is 4.8e-2
(`TOKEN_BODY_QUANTISED_MEDIAN_BAR` is 6e-2) and the test now asserts rank agreement rather than
host parity. These are fixture distances, not the served model's.

Both checks compare bits. Neither is a speed measurement, and neither runs a checkpoint.

## What could not be measured

- **The served delta.** No cluster window and no checkpoint were available, so the change was not
  captured against the frozen GLM anchor (`numerics/references/glm53-anchor.json`). The mean NLL
  delta, top-1 agreement, KL lower bound and greedy-text flips of version 2 against version 1 are
  therefore **unmeasured**. The registry's `reference` and `fingerprint` still name the version-1
  capture until the cluster recaptures; the next capture must be compared against the anchor with
  `scripts/numerics.py compare`.
- **The speed A/B.** Whether the one-row step is flat or slightly slower and whether the
  several-row step is faster are both **unmeasured**. The expected shape from the plan is flat or
  slightly slower at one row and faster at several, as on DeepSeek and Qwen; that expectation is
  not evidence.

To measure the served configuration, on the cluster with GLM served at world 4:

```bash
# The version-2 reference, then the bit check against it.
python3 scripts/numerics.py capture --url http://127.0.0.1:8001 --out numerics/references/glm53-tp4-sm121a.json
python3 scripts/numerics.py check --url http://127.0.0.1:8001 --reference numerics/references/glm53-tp4-sm121a.json

# The served delta against the frozen anchor.
python3 scripts/numerics.py capture --url http://127.0.0.1:8001 --forced-from numerics/references/glm53-anchor.json --top-logprobs 20 --out numerics/captures/glm53/v2-fp8-token-body.json
python3 scripts/numerics.py compare --anchor numerics/references/glm53-anchor.json --candidate numerics/captures/glm53/v2-fp8-token-body.json --markdown docs/numerics/glm53/v2-fp8-token-body.md
```

`scripts/numerics.py check` is the command that decides whether the server's bits are the
declared version's.

The declaration check itself (`python3 scripts/numerics.py ci`) reports two findings until that
capture exists: version 2 names version 1's fingerprint, so it is a declared non-change, and the
`anchor`, `coverage` and `monitors` prerequisites are still false. Both are expected here - the
first clears with the recapture, the second with the prerequisite work - and neither can pass
without the cluster.

## Reading

The important property this version establishes is the one the device test measures: a GLM decode
step on the body is the prompt tile's row, not a second arithmetic image. What it does not
establish, until the cluster capture, is how far the served logits move and whether the step got
faster. Version 1 remains the only measured GLM reference.

The body covers the FP8 precision of GLM's decode; BF16 projections still take the tiled
GEMV, and the `WAIVERS` lines in `crates/sparkflash-ops/src/parts.rs` now say so and cite
`glm_decode_fp8b128_rows`.
