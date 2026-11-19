# GLM-5.3-Flash, numerics version 2: decode on the FP8 E4M3 token body

**Registry state: version 1 is still the captured entry; version 2's capture exists as evidence.**
The capture was taken on the served world-4 configuration on 2026-11-19 and is committed under
`numerics/captures/glm53/`. It is not the entry's `reference`: naming new bits at version 1 is an
undeclared change, and the registry may not bump while `coverage` and `monitors` are false
(`scripts/numerics.py ci`). GLM-5.3's own `[[anchor]]` and the captures teacher-forced from it
are in place, so the per-model anchor requirement does not block this bump; the two dumps do.
Version 2 lands as one diff - the reference, the bump and this report - once they are in place.
Until then version 1's reference stands, and this page records how far the tip's bits move from it.

Version 2 of the world-4 sm_121a configuration moves GLM's one-to-eight-row block-scaled FP8
projections from the tiled row GEMV (`q4e_gemv_tiled_e4m3_block`) to the shared tile's few-row
body (`glm_decode_fp8b128_rows`) under `decode.tensor_cores`'s on default, which the shipped
`glm53_flash.toml` implies. It is the first declared change to GLM's arithmetic after the
shipped version 1 (the E4M3 attention window, [v1-e4m3-window.md](v1-e4m3-window.md)).

## What changed

A decode step, a speculative verify and any FP8 projection of at most eight rows now launch the
body on the same E4M3 codes, the same f32 128 x 128 block scales and the same f32 activations the
GEMV read. Nothing is converted: the body takes the GEMV's operands and writes its part-by-part
output layout, one launch for each part of a fused projection. The body runs the prompt tile's
chain - the E4M3 codes widened to bf16, the activation split into a bf16 head and remainder, two
MMAs a 16-column step with K ascending, and the 128-column block scale folded in ascending block
order - so its row is the prompt tile's row bit for bit. That tile is `glm_prefill_fp8b128_mma`,
which takes f32 activations; the shipped release's bulk prefill takes `glm_prefill_q8_mma` for its
dynamic E4M3 activations, so this change does not by itself make a continued turn equal a cold q8
prefill.

The routing boundary is the arm's existing one, now gated by the decode switch:
`decode.tensor_cores` (`SPARKFLASH_DECODE_MMA`), on by default and left unset by the shipped
`glm53_flash.toml`, routes a one-to-eight-row FP8 part to the body; `off` puts it back on the
tiled GEMV (`gemv_small`). `MAX_ROWS` is eight and a larger batch keeps the bulk tensor-core
GEMM whatever the switch says. Projections whose shape the body does not take (a BF16 weight, a
K run that is not whole 128-column blocks, or a part that is not whole eight-column outputs)
keep the tiled GEMV in both switch states. The routed experts'
`q4e_expert_gemv_e4m3_block` is outside this change and still serves the token rows.

## What was measured before the cluster

- **The body is the tile's row.** On the sm_121a device, for every token count from 1 to 8, a
  two-part fused projection launched through the arm's own body path is bit-equal, row by row, to
  `glm_prefill_fp8b128_mma` over the same weights, scales and activations
  (`device::tests::a_decode_step_on_the_token_body_is_the_prefill_tiles_row`, ignored unless a
  CUDA device is present). The kernel-level harness
  (`crates/sparkflash-kernels/tests/token_bodies_device.rs`) also holds the body to the tile at
  1, 2, 4 and 8 rows, on a 256 x 512 shape and a 96 x 384 partial tile.
- **The routing boundary.** A CPU test holds the band to one through eight rows and the ninth to
  the bulk path, and rejects a BF16 weight, a part that is not whole eight-column writes, and a
  part whose block scale does not start on a 128-row boundary
  (`device::tests::the_fp8_token_body_takes_one_to_eight_rows`). The switch test covers both
  values: on routes the FP8 parts to the body, off to the tiled GEMV, and a BF16 part or a shape
  the body does not fit stays on the tiled GEMV either way
  (`the_decode_switch_picks_the_token_body_or_the_tiled_gemv`); the shipped profile's unset
  field is pinned to the body's route (`the_shipped_profile_decodes_on_the_token_body`).

The device arm's device-versus-host fixture moves with the declared change: with the FP8 weights'
inputs unquantised, the three worlds' worst seams are 1.3e-5, 1.8e-5 and 1.3e-5
(`TOKEN_BODY_REL_L2_BAR` is 2e-5); with the release's activation scheme the host quantises each
input per 128 columns and the body takes the f32 row, so the median seam is 4.8e-2
(`TOKEN_BODY_QUANTISED_MEDIAN_BAR` is 6e-2) and the test now asserts rank agreement rather than
host parity. These are fixture distances, not the served model's.

## Measured on the cluster, 2026-11-19

Served at world 4 on the four DGX Sparks from the branch built at `f1f86bf4` (release,
`5a092d5a`), GLM-5.3-Flash over 1,048,576 tokens, the shipped `glm53_flash.toml` profile. The
four ranks agreed on the plan agreement word before the gather (`5b3e9a9f0b067407`, context
`1048576`, configuration `bc832624869ede08`), all four carried the same binary before the swap,
and `/readyz` answered in 21 s.

Two free captures are bit-identical, so the served configuration is deterministic at this tip:

```text
451be41bb5fbe34db4646d2167647fad282be34b2f144523fe4549d101be30fa  e4m3-token-body-v2.json
451be41bb5fbe34db4646d2167647fad282be34b2f144523fe4549d101be30fa  the same capture, second run
41dc724dd71ca2b8e5df1dd2fc32c48a12987042bf5c88657763ea9458d44011  e4m3-token-body-v2-forced.json
```

The forced capture is teacher-forced on version 1's tokens, top-8 logprobs, and compared against
the version 1 reference (`827ec5f682d512cc`):

| metric | value |
| :-- | --: |
| tokens scored | 997 |
| version 1 NLL / token | 0.1504 |
| mean NLL delta | +1.373e-01 |
| mean absolute NLL delta | 1.559e-01 |
| p99 absolute NLL delta | 4.780e+00 |
| max absolute NLL delta | 1.446e+01 |
| top-1 agreement | 93.3801 % |
| mean KL lower bound | 1.075e-01 |

Per prompt the delta is not uniform: retrieval is on the floor (+4.2e-03 mean NLL, top-1 100 %),
logic +6.8e-03 (98.4 %), tool-json +6.4e-02 (96.9 %), summary +1.0e-01 (95.3 %), arithmetic
+1.2e-01 (94.5 %), code +1.6e-01 (89.8 %), translation +2.0e-01 (97.7 %) and raw continuation
+4.1e-01 (75.8 %).

The served gate's lanes check also passed on the same boot: 41.4 tok/s for a prompt alone, 54.7
and 54.8 tok/s with four at once, and every together answer equal to its alone answer.

**What this number is.** It is the tip against version 1, not this body's share. Version 1's
reference predates the arm's move onto `Runtime<Walk>` (phase 1) and this decode body (phase 5),
so the delta is the sum of those changes. For scale: Qwen's tensor-core decode moved +7.0e-04
mean NLL with 99.39 % top-1 agreement ([../qwen38/v1-tc-decode.md](../qwen38/v1-tc-decode.md)),
two orders of magnitude less. Two causes are consistent with the per-prompt pattern and neither
is established: the body's accumulation order (the 128-column block scale folded in ascending
block order, two MMAs a 16-column step) against the row GEMV's, and the arm's prefill chunking
at the profile's 4,096 rows where version 1's build ran the code default.

**The separation capture taken in that window is not evidence.** The same window served the build
once with `SPARKFLASH_DECODE_MMA` unset and once with it set to `off`, and the two captures were
**bit-identical**. That is the defect the window found, not the body agreeing with the row GEMV:
on that build the switch was inert. Nothing in the decode path read it; `DeviceRank::gemv` chose
the tile's few-row body by shape alone (`fp8_token_body_fits`), so `off` served exactly the same
kernel. This fix gates the route on the switch. The separation capture has to be retaken on a
build that carries it, against this capture, before version 2 declares: only then does `off`
measure the tiled GEMV's bits and the body's own share separate from the move's.

## What is still not measured

- **The body's own share of the delta.** The switch is live now, so the separating capture -
  the shipped default against `SPARKFLASH_DECODE_MMA=off` - must be retaken on the fixed build;
  the window's `off` capture was bit-identical only because the switch was inert.
- **The speed A/B.** Whether the one-row step is flat or slightly slower and whether the
  several-row step is faster are both **unmeasured**. The expected shape from the plan is flat or
  slightly slower at one row and faster at several, as on DeepSeek and Qwen; that expectation is
  not evidence. The lanes check above is a functional gate, not a speed comparison.
- **Greedy-text flips.** `numerics.py` at this tip reports top-1 agreement, not a flipped-token
  count; 93.38 % agreement over 997 positions is about 66 changed argmaxes, which is derived and
  not the tool's own count.

## Reading

The device tests establish that a GLM decode step on the body is the prompt tile's row, not a
second arithmetic image. The cluster capture establishes something the device tests could not: on
the released checkpoint the tip's bits are deterministic, they are measurably different from
version 1's, and the difference is large enough that it is a served change and not a rounding
note. Retrieval is essentially unchanged; the raw continuation is the worst prompt at 75.8 %
top-1.

Version 1 remains the only reference the registry names until the coverage and monitor dumps
allow version 2 to declare this capture.

The body covers the FP8 precision of GLM's decode while `decode.tensor_cores` is on (the
shipped default); `off` puts those parts on the same tiled GEMV the BF16 projections always take,
and the `WAIVERS` lines in `crates/sparkflash-ops/src/parts.rs` say so and cite
`glm_decode_fp8b128_rows`.
