# The host `kv_path` semantics — the contract a device port must not drift

Status: **acceptance reference** for the device KV-path port (`kv_path` in
`crates/sparkflash-cli/src/layer_device.rs`, three kernels: `kv_append_bf16`,
`window_quant_fp8_bf16`, `rope_rotate_bf16`). Written 2026-10-22 from `origin/win/layer-device`
@ `159df686`; every claim carries a `file:line` so an implementer can check it rather than trust it.

device-lead began this document, established items 1-5 below from the code, and then ran out of
session budget before reading four sites it correctly refused to hand-wave. The Lead closed those four
(`attention::rope_tables_for_layer` body, `cuda_harness::engine_window_row` body, the `cache.push`
line, and whether the seam reads the cache) and wrote them up; the appendix records which were closed
and, in one case, corrected.

## 1. Where the cache lives today

- The cache is a **loop-local host structure**: `Vec<Vec<(Vec<u8>, Vec<u8>)>>` indexed by layer, one
  row list per layer (`crates/sparkflash-cli/src/tp4_forward.rs:10685`), mutated in place as positions
  advance (`:11104` and `:11113`).
- Each row is a **pair of already-quantised byte buffers**: `(window_codes, window_scales)` - the
  engine-format E4M3FN code bytes and the E8M0 scale bytes - produced by `kv_path` and returned in
  `run_layer_kv_path`'s result (`:2948-2951`).
- The device `AttentionBody` holds its **own** `window_codes`/`window_scales` buffers (`:3082-3085`
  region), fed per launch from the staged window rows. So the device side already has a place for these
  bytes; what it does not have is ownership of the cache.

**Ownership move required by the port**: the device arm must own its own cache, the host arm keeps this
host-resident one, and the two must not share mutable state. Nothing in the steady-state device loop may
read the cache back to the host.

## 2. `rope`

- `mla_width = config.head_dim`, `rope_head_dim = config.rope_head_dim`; it refuses
  `rope_head_dim == 0 || rope_head_dim > mla_width || rope_head_dim odd`
  (`tp4_forward.rs:3005-3006, 3022-3031`).
- `start = mla_width - rope_head_dim`, and **only the tail** `kv[start..]` rotates:
  `linalg::rope_rotate(&mut kv[start..], &tables.cos, &tables.sin)` (`:3031`).
- **Pairing is adjacent**: `linalg::rope_rotate` (`crates/sparkflash-engine/src/linalg.rs:1183`) is
  `(a, b) -> (a cos - b sin, a sin + b cos)` over `x[2i], x[2i+1]` (`:1205-1216`). It is *not* the
  half-split (NeoX-style first-half/second-half) convention; a kernel that assumes the split form is
  wrong by construction.
- **Tables are per position and built from a one-element position list**:
  `rope_tables_for_layer(config, layer, &[position])` (`tp4_forward.rs:3037`,
  `forward/attention.rs:300`). Inside `tables_for` (`attention.rs:326`):
  - `freqs[k] = 1 / base^(2k/dim)` in **f32** (`:334-336`);
  - for layers with `layer_ratio > 0` the base is `config.compress_rope_theta` and the YaRN ramp is
    applied in f32 with the band edges computed in **f64** from `beta_fast`/`beta_slow`,
    `original_seq_len` and `rope_factor` (`:304-311, :340-364`);
  - the angle is `position as f32 * freq` then `.cos()`/`.sin()` in f32 (`:369-375`).
- **Why "per position, one-element list" matters for a device port**: the values are bit-computed as
  `(position as f32) * freq` in f32, so a precomputed range table *can* reproduce them exactly if it
  applies the same f32 multiply per position and does not reuse a cumulative product. What must not
  differ: the f32 multiply, the f32 trig, the f32 ramp arithmetic (including re-deriving `1 - smooth`
  from the rounded `smooth` rather than reusing `ramp`, `:359-363`), and the f64 band edges. A kernel
  may hold a range table; it may not change the arithmetic that fills it.

## 3. The fp8 window round-trip

- The round-trip is applied to the **whole rotated row**, not to the window slice
  (`tp4_forward.rs:2903-2922`, called at `:3038`). The window is a slice of **already-quantised rows** -
  there is no second quantisation of a window at attention time.
- The construction is single-sourced: `window_row_bytes` validates the width (`MLA_LATENT = 512`) and
  finiteness, then delegates to `sparkflash_kernels::cuda_harness::engine_window_row`
  (`crates/sparkflash-kernels/src/cuda_harness.rs:220`), which:
  1. rounds the caller's f32 row to **bf16 exactly once** (`:224-228`;
     `quantize_activation_fp8` returns the input dtype, so the bf16 row *is* the stored row - no second
     bf16 round);
  2. quantises with the engine's `sparkflash_engine::quant::quantize_activation_fp8`;
  3. **asserts byte equality** against the format oracle `m4_quant::act_quant_fp8_row` on the same bf16
     row (`:230-240`) - a disagreement is a hard error, never a silent divergence.
- The rule itself (`crates/sparkflash-kernels/src/m4_quant.rs:396`, documented at `:377-395`):
  - **block = 32 values**; **one** absolute maximum per block (not a running row maximum);
  - **amax floor 1e-4** (`ACT_QUANT_AMAX_FLOOR`), then `scale_code = ue8m0_code_ceil(amax / 448)` -
    the round-**up**-to-a-power-of-two E8M0 code, total on every f32 (NaN -> reserved code 255,
    non-positive -> 0, subnormals ceiled from the highest mantissa bit, clamped to [0, 254]);
  - each value is `value / decode(code)`, **clamped to +-448**, then E4M3 round-to-nearest-even;
  - **codes are 512 bytes per row** (`WINDOW_CODE_BYTES = MLA_LATENT`, `tp4_forward.rs:2886`) and
    **scales are 16 bytes** per row (512 / 32).
- The round-trip is not a tolerance-freedom area: the f32->bf16->fp8 path is *the released format*, and
  the oracle assertion is the reason two harnesses cannot drift. A device kernel must reproduce the bf16
  round, the per-32 amax, the 1e-4 floor, the ceil-power-of-two scale and the +-448 clamp; D54's
  tolerance arm covers the arithmetic *around* the format, not the format.

## 4. The append rule and visibility

- Order inside the position loop, verbatim (`tp4_forward.rs:11104-11113`):
  1. `cache.len() != position` -> **hard error** ("the position loop left the cache out of step");
  2. the reference seam runs (`:11119`/`:11126`), producing `seam.window_width`;
  3. `window_width == 0 || window_width > cache.len()` -> hard error;
  4. `window_start = cache.len() - window_width`; `window_rows = cache[window_start..].to_vec()`
     (**deep clone** of the already-quantised rows);
  5. the staging + attention body consume `window_rows`;
  6. the new row is pushed **after** that: `cache.push((kv_path.window_codes.clone(),
     kv_path.window_scales.clone()))` (`:11113`).
- So an appended row becomes visible to the **next** position's window, never to the current one. For
  the device port the rule is the same but the ordering must be explicit: the append is queued on the
  stream *after* the window read of this position and *before* the next position's window read. A
  kernel that appends then reads in the same position sees a window one row too long.
- **Device arm, decode only (stage-order equivalence, added 2026-10-22).** `run_layer` queues the
  rotate/quantise/append *before* `reference_seam`/`stage_attention`/`attention_body`, which is not
  the append-after-read order the host uses. For decode the two produce the same number: the window is
  `[position - window_width ..= position - 1]`, so it excludes the row appended at `position` and no
  read in this position can observe it. The equivalence is a property of the single-position window,
  not a licence to reorder in general: **multi-position/prefill needs the split** (append after the
  last read of the group), and that is the condition for revisiting the order.
- Boundary consequence, worth stating because it is a real constraint: at `position == 0` the cache is
  empty, so the check in (3) makes **any non-zero window an error**. The served loop is only valid once
  the cache has rows; the port does not change that, and a device implementation must keep the same
  refusal rather than "fixing" it with a zero-length window.

## 5. The window slice arithmetic

- `window_start = cache.len() - window_width`, and the window is `cache[window_start..]` - the
  **tail** of the cache, i.e. the most recent `window_width` rows (`:11141-11142`).
- `window_width` is **decided by the reference seam**, not by the caller: `window_width > cache.len()`
  refuses, so the seam can never ask for a longer window than the cache holds even though it attends
  across ranks.
- The invariant that makes it valid is exactly `cache.len() == position` at the top of the position -
  one row per completed position, no gaps, no duplicates.

## 6. The seam does **not** read the cache (correction)

device-lead flagged "whether `reference_attention.step(...)` actually reads the cache" as an unverified
assumption, and it was right to. It does **not**: the call is
`reference_attention.step(layer, position, &entry.normed, &query_path.q_latent_bf16)` (and the
`step_with_backend` form adds only the device handle) - `tp4_forward.rs:11119-11132`. The seam sees the
block's normed row and this rank's q latent, and returns `window_width` plus the compressed ids/rows; the
cache is sliced afterwards by the loop. Anything written in an earlier draft about "the seam reads the
cache too" is withdrawn.

## 7. What the device port must reproduce, and where it is free

| must be identical | free under D54 (declared arm) |
| :-- | :-- |
| the rotated *tail only*, adjacent-pair convention, `start = mla_width - rope_head_dim` | the trig implementation, if it reproduces f32 `cos`/`sin` to the declared logits bars |
| the f32 `position * freq` multiply and the per-layer base (YaRN ramp for compressed layers) | whether tables are computed per position or precomputed as a range - provided the fill arithmetic is unchanged |
| bf16 round once, then block-32 amax, 1e-4 floor, ceil-power-of-two E8M0 scale, +-448 clamp, E4M3 RN-even | the kernel's internal reduction order (tree allowed), never the format |
| append **after** the current position's window read; `cache.len() == position` invariant | where the device cache lives in the slab, as long as the device arm owns it |
| window = tail slice, `window_width` taken from the seam, empty-cache refusal | the staging path into `AttentionBody`'s existing window buffers |

Reproduce the format exactly and the rest is arithmetic the numerical gate (D54) will judge on a row;
reproduce the *format* approximately and the layer is silently wrong in a way no logits bar will
attribute to the KV path.

## Appendix: the four gaps device-lead refused to hand-wave

1. **`rope_tables_for_layer` body** - closed (§2): per position, one-element list, f32 multiply and f32
   trig, f64 YaRN band edges, f32 ramp; the "reuse `ramp` instead of re-deriving `1 - smooth`" trap is
   in the code with its reason.
2. **`engine_window_row` body** - closed (§3): single bf16 round + engine quantiser + byte-equality
   assertion against the `m4_quant` oracle; block 32, floor 1e-4, ceil-power-of-two scale, +-448 clamp.
3. **`cache.push` position relative to the `len == position` check** - closed (§4): the check is at the
   top of the position, the push is at the bottom, after the window has been read; the row is visible
   from the next position only.
4. **Whether the seam reads the cache** - closed (§6), and the assumption was **wrong**: it does not.

## The window slice, pinned (2026-10-22, after the interface wording drifted)

The window arithmetic is the one place where a one-off index error is silent: an off-by-one window reads a
row that has not been appended yet, or drops the oldest row, and the result is a plausible attention
output rather than a failure. Pinned here, from the host loop and the frozen interface:

- At the top of position `p` the cache holds **exactly `p` rows** - `cache.len() != position` is the hard
  error (`tp4_forward.rs:11104`).
- `window_start = cache.len() - window_width = p - window_width`; the window is `cache[window_start..]`,
  deep-cloned **before** the push (`:11141-11142`). The append is at the bottom of the position
  (`:11113`).
- Therefore the window is rows **before `p`**: attribute indices `[position - window_width ..= position - 1]`.
  Position `p`'s own row is **not** in the window: it does not exist yet for this position.
- `window_width == 0 || window_width > cache.len()` is a hard error, which is also why position 0 refuses
  any non-zero window and why the served loop is only valid once the cache has rows.

An earlier draft of the `DeviceKvCache` doc said `[position - window_width + 1 ..= position]`. That is
wrong at both ends and includes the unappended row; it was corrected to the form above. The code, the
kernel and the ring slot follow this section, not the earlier string.

## Cross-rank collective counting: the runner, not the process

The per-rank check (`verify_collective_count`) refuses loudly at the run boundary; the **cross-rank**
check is deliberately *not* an in-process collective, because enqueuing a collective to compare collective
counts changes the count and would let the rank whose skip is the bug be the rank whose extra check makes
the numbers agree. It lives in `scripts/check-collective-count.py`, run by the four-rank harness after the
per-rank records are on disk and before anything is published (exit 0 agree, 1 divergence naming every
rank's count, 2 a record that cannot testify). Verified locally in both directions.
