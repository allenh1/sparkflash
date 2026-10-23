# S1 P1 — the `engram_write` device stage: what landed, and the wiring the Lead adds

Branch `integration/s1-layer-device`, owner **device-lead** (shared task `task-1`). This note is the
deliverable the Lead asked for in the P1 brief: (1) the kernel and its typed launcher, which are
landed in this task, and (2) the exact wiring the Lead must add to `DeviceLayer::engram_write`, plus
the verified `stage_attention` wiring note.

No claim in this file about device behaviour, numerics, the D54 gate or step timing is measured: this
host has no CUDA device (`/dev/nvidia*` absent), so every one of those is **NOT RUN**. What *is*
verified is listed in §1 with the exact command and output.

## 0. Summary

* `kernels/engram_write_sm121.cu` + `crates/sparkflash-kernels/src/engram_write_launch.rs` land the
  symbol the stage contract names (`engram_write_bf16`, `layer_device.rs:764-768`) with the contract's
  ABI, a one-block-per-1024-column launch and span guards. This is the write only: the multiply, the
  add and the one bf16 cast of `model.py:363-365`.
* The write alone does not make the stage runnable with the kernels that are landed today: the landed
  gate/write kernel (`engram_gate_write_sm121`) is a **bf16-stream / bf16-out** kernel, and the arm's
  stream span is **f32** (`LayerResidual::stream`, `layer_device.rs:372-382`). Two exact conversion
  kernels therefore bridge the dtype, and **both are now landed** in one new cubin
  (`kernels/engram_stream_convert_sm121.cu`: `engram_narrow_bf16`, `engram_widen_f32`) with their
  launcher and by-name loaders; §4.4 lists them and gives the two declaration lines for your
  probe/pin step. They are conversions, not gate kernels.
* With the at-open arena of §4.1 the per-position engram path needs **no gather launch and no host
  operand**: the 24 table rows a position looks up are a pure function of the token history, so their
  bytes are read at open (the same ranges `warm_engram_rows` already touches) into a per-position
  arena; the per-step chain is then the landed `engram_embed_sm121` (dequantise + bf16 cast +
  activation quantise), the landed `engram_wkv_sm121`, the landed gate, and the declared write: 6
  launches and zero copies, which is the S1 shape.
* The gather-index scalar the Lead asked about is **not needed** for that wiring: the position is a
  host-known launch offset into the arena. §4.7 gives the exact `begin_position` change for the
  *device-gather* variant (D39 rule 4's target) instead, and says which launches would read it.

## 1. What landed here, and what was verified on this host

Files added by this task (all committed by the Lead, never by me):

| File | Contents |
| --- | --- |
| `kernels/engram_write_sm121.cu` | symbol `engram_write_bf16` (`:68`), the stage's declared kernel |
| `crates/sparkflash-kernels/src/engram_write_launch.rs` | typed params, grid, span guards, declaration pin, CPU twin test, load/launch |
| `crates/sparkflash-kernels/src/lib.rs:44` | `pub mod engram_write_launch;` |
| `crates/sparkflash-kernels/src/cuda_driver.rs:3047` | `Cuda::load_engram_write` (by-name loader; the symbol has no `KernelId`) |
| `kernels/engram_stream_convert_sm121.cu` | `engram_narrow_bf16` (f32 -> bf16) and `engram_widen_f32` (bf16 -> f32), one cubin, non-stage so no trace trio (§4.4) |
| `crates/sparkflash-kernels/src/engram_convert_launch.rs` | the two declarations pinned, `Direction`, params/spans, validate/grid/check_spans, `load_narrow`/`load_widen`, `launch_narrow`/`launch_widen`, CPU-twin test |
| `crates/sparkflash-kernels/src/lib.rs:39` | `pub mod engram_convert_launch;` |
| `crates/sparkflash-kernels/src/cuda_driver.rs:3063,3079` | `Cuda::load_engram_narrow` / `Cuda::load_engram_widen` (by-name) |

The pinned declaration (`engram_write_launch::PARAMS`, pinned by `declaration_matches`):

```
extern "C" __global__ void engram_write_bf16(
    const float *__restrict__ stream,       // [copies][dim] f32 residual stream
    const float *__restrict__ gate,         // [copies]       f32 gates
    const float *__restrict__ value,        // [dim]          f32 value row
    float *__restrict__ stream_out,         // [copies][dim] f32 out
    const int dim,                          // columns per copy
    GpuTraceRing *__restrict__ trace_ring,  // ABI uniformity only; no event emitted
    const uint32_t layer_id,
    const uint64_t step_seq)
```

Launch: grid `(dim.div_ceil(1024), copies, 1)`, block `(256, 1, 1)`. `dim` is the last argument word;
`copies` (hc) is grid.y because the contract's declaration has no hc word — that is the Lead's ruling
(A), contract ABI authoritative. The copy is loaded before it is stored, so `stream_out == stream`
(in-place) is exact.

Verification actually run on this host:

| Command | Result |
| --- | --- |
| `/home/user/cuda-tk/bin/nvcc -std=c++17 -arch=sm_121a --ptx -I kernels kernels/engram_write_sm121.cu -o /tmp/engram_write.ptx` | exit 0; PTX has `mul.rn.f32`, `add.rn.f32`, `cvt.rn.bf16.f32`, `cvt.f32.bf16` and **no** `fma`/`mad` (so nvcc contraction cannot fuse the host twin's two roundings into one) |
| `/home/user/cuda-tk/bin/nvcc -std=c++17 -arch=sm_121a --ptx -I kernels kernels/engram_stream_convert_sm121.cu -o /tmp/engram_convert.ptx` | exit 0; both entries, `cvt.rn.bf16.f32` and `cvt.f32.bf16` present, **no** `fma` |
| `cargo test -p sparkflash-kernels --lib` | **584 passed, 0 failed, 2 ignored** (8 `engram_write_launch::tests` + 8 `engram_convert_launch::tests`; the 2 ignored are device tests that need a GPU) |
| `cargo clippy -p sparkflash-kernels --lib --all-targets -- -D warnings` | clean |
| `cargo fmt -p sparkflash-kernels -- --check` | clean |

The 16 module tests (8 for the write, 8 for the conversions) pin: the declarations (parameters,
order, `__restrict__`, `const` pointees, `int dim` / `uint32_t count`), non-vacuity of each pin (a
removed `__restrict__`, a dropped pointee `const` and a widened count/`dim` all fail it), the
argument word order, the conversion direction sizes and intrinsics (and that they carry no trace
ring), the twin round-trips of both conversions, grid edge cases (`dim` 1 / 1024 / 1025 → 1 / 1 / 2 blocks), the four geometry
refusals, the four span refusals by name, the source's rounded-multiply-then-add and
`__float2bfloat16_rn` (and no FMA), the trace tail being inert, and — the arithmetic pin — that this
module's write reference reproduces the **landed CPU twin's write half**
(`engram_native_ref::gate_and_write`, `engram_native_ref.rs:177`) bit for bit on a synthetic row.

NOT RUN here, and not claimed: any device launch, any comparison of the kernel against the twin on a
device, the D54 numerical gate, and any timing. The Lead's registration step (a `PROBED_KERNELS` entry
`("engram_write_sm121.cu", "engram_write_bf16")`, the baseline pin for the cubin, and the nvcc
baseline build) is the Lead's; the file+symbol+declaration line is above.

## 2. The landed Engram kernels and their exact ABIs

The stage's declared kernel is only the write. The kernels that are pinned today, with the operands
the write consumes produced by the ones above it:

| Kernel | ABI (from its pinned launcher) | Produces / consumes |
| --- | --- | --- |
| `engram_embed_sm121` | `(row_codes [rows_cols][head_dim] E4M3, row_scales [rows_cols][head_dim/32] E8M0, fetched bf16 [rows_cols*head_dim], act_codes E4M3, act_scales E8M0, rows_cols, head_dim, …, trace trio)` (`engram_embed_launch.rs:36-58`) | the 24 gathered rows dequantised to bf16 **and** quantised to the activation pair in one launch |
| `engram_wkv_sm121` | `(act_codes, act_scales, weight, weight_scales, out bf16 [out_width], out_width, inner, …, trace trio)` (`engram_wkv_launch.rs`) | the W8A8 projection: `out_width = dim * (hc + 1)`, key `[0, hc*dim)` then value `[hc*dim, hc*dim+dim)` |
| `engram_gate_write_sm121` | `(kv bf16 [out_width], stream bf16 [hc*dim], product f32 [hc*dim], out bf16 [hc*dim], gates f32 [hc], dim, hc, eps, …, trace trio)` (`engram_gate_write_sm121.cu:45`, `engram_gate_write_launch.rs:35`) | the gate over (`kv` key, `stream`, `product`) **and** the same write, into bf16 |
| `engram_write_bf16` (this task) | `(stream f32, gate f32, value f32, stream_out f32, dim, …, trace trio)` | **the write only**, into the arm's f32 stream |

Facts about those ABIs that constrain the wiring, each verified in the sources:

* `engram_gate_write_sm121` writes `gates[hc]` (`engram_gate_write_sm121.cu:50,86`) — the gate values
  the write needs are already on device after that launch.
* Its `stream` operand is `__nv_bfloat16` and it computes the gate from
  `h = __bfloat162float(stream[...])` (`engram_gate_write_sm121.cu:68`), i.e. from bf16 values; the
  host gate likewise rounds its copy first (`let h = quant::bf16_round(stream[column])`,
  `engram.rs:973`). A narrow is therefore *exactly* the host's `h`.
* Its write phase runs after a `__syncthreads()` and reads `stream[index]` before storing
  `out[index]` (`engram_gate_write_sm121.cu:88-98`), so `out == stream` is a safe alias for that
  launch — useful below because the stage discards `out` in branch (i).
* `engram_embed_sm121` is the whole host `read_table`+`quantise_fetched` pair on device: it reads the
  gathered row bytes, dequantises with the row scales, applies the released bf16 cast and writes the
  activation codes/scales the wkv kernel takes (`engram_embed_sm121.cu`, twin
  `engram_native_ref::embed_row`), which is why §4.1 only has to move the *gather* to open.
* `product` is the constant `q_weight * k_weight` (`engram.rs:906-911`): weights only, no position
  dependence, so it is uploaded once at open.
* The `value` row of `engram_write_bf16` is `kv[hc*dim .. hc*dim+dim]` (`engram.rs:962,1006`):
  the wkv tail, which is bf16 on device. Hence the widen in §4.4.
* A position whose hash row contains `HASH_SENTINEL` is **skipped entirely** by the host seam
  (`engram.rs:919-921`): the device stage must launch nothing for such a position (§4.1 flags).

## 3. Where the stage sits in the arm

* `LayerStages::run_layer` (`layer_device.rs:1579-1599`) calls `self.engram_write(layer, position)`
  **first**, before `mhc_entry`: the write must land on the stream the layer block starts from,
  exactly as `engram.rs:365` does.
* The device arm's stub is `tp4_forward.rs:11884-11889`: a no-op for layers without rows, otherwise
  `Err(Self::refusal("engram_write"))`. `self.engram_rows` (`tp4_forward.rs:10563`, filled from
  `config.engram_rows` at `10825`) is the "layers 1 and 14" filter.
* The stage's stream is `self.buffers.residual.stream` (`layer_device.rs:375`): `hc_mult * dim` f32.
  There is **no** engram span in `LayerBuffers` today (`layer_device.rs:387-455`), and none is needed:
  §4.5 puts all engram scratch in one run-scope handle.
* The blocker text the stage must retire: `layer_device.rs:1389-1390` — "the host gather and stream
  add (engram_forward_with_projection) must become the device gather D39 rule 4 targets; the
  projection is already on device". §4.1 moves the per-step gather to open (the table bytes are read
  once, there) and §4.2 moves the stream add onto the device (the declared kernel); what remains of
  that sentence is the *device* gather (the hot rows staying resident on the device, indexed by a
  device scalar) — that is D39 rule 4's target and the one part this slice does not move (§7, item 1).

### 3.1 The D39 rule-4 declaration (your ruling (c)): what this slice does and does not close

S1 takes the at-open arena **with this declaration**. It belongs in the stage record and in
`docs/build/s1-result.md` (yours), and it is deliberately not a claim that rule 4 is closed:

* **Retired by this slice:** the per-step host *gather* (the paged row read) and the per-step host
  *stream add*. The gather becomes one open-time read of the selected table ranges into a per-position
  arena (§4.1); the add becomes the declared `engram_write_bf16` launch over the arm's stream (§4.2).
* **Run-shape assumption, stated:** the arena is buildable because a *fixture* run knows its token
  history at open (`--token-ids` is the whole prompt, `tp4_forward.rs:13414`). A **serving** run does
  not: its next token is unknown until the step produces it, so the same stage would need the gather
  per step. The arena therefore encodes the benchmark's run shape, not a general serving path.
* **Deferred, explicitly:** the *device* gather - hot table rows resident on device, a hash-id slot
  map, and the gather kernel that fills the arena entry from a device position scalar. That is D39
  rule 4's target, it is **not landed**, and it is not in this slice.
* **`begin_position`'s position scalar** (§4.7) exists only for that later variant. In this slice the
  position reaches the stage as a host launch offset; `begin_position` keeps exactly its two copies.
* **Do not claim rule 4 closed.** The blocker at `layer_device.rs:1389-1390` should read as retired
  for the host per-step work and open for the device gather. Draft stage-record wording for your file:

```
The per-step host gather and stream add are retired: the selected table ranges are read once at open
into a per-position arena (a run-shape assumption of the fixture's known token history), and the
stream add is the declared engram_write_bf16 launch on the device. The device gather (hot rows and a
hash-id slot map on the device, indexed by a position scalar written in begin_position) is explicitly
deferred; D39 rule-4 remains open.
```

## 4. The wiring the Lead adds

### 4.1 The at-open arena: the gather moves to open, the quantiser stays on device

(This is the wiring your ruling (c) accepted under the declaration in §3.1: a run-shape assumption,
device gather deferred, rule 4 not closed.)

The per-step host operands of the engram head today are the gather and the activation quantiser:
`engram_forward` pages this position's table rows through the source's range read
(`read_paged_table`, `engram.rs:508-556`, offsets `id * width` and `id * width.div_ceil(32)` at
`engram.rs:517-524`) and quantises the fetched row (`quantise_fetched`, `engram.rs:947`). Both are
functions of the token history only, never of the stream, so they can leave the step:

* the **gather** moves to **open** — the same reads the warm pass already performs
  (`warm_engram_rows`, `tp4_forward.rs:5787-5805`, through `read_ranges_paged`), with the bytes kept
  instead of discarded;
* the **dequantise + bf16 cast + activation quantise** stay **on device**, because the landed
  `engram_embed_sm121` does all three in one launch: 24 rows of code/scale bytes in, the fetched bf16
  row and the activation code/scale arrays out (`engram_embed_launch.rs:36-58`).

So the arena holds, per (engram layer, position), the 24 gathered table rows in the layout the embed
kernel reads — `[24][256]` E4M3 codes then `[24][8]` E8M0 scales, 256-B aligned, 6,400 B per entry —
plus one skip flag for sentinel rows. Builder, at run scope beside
`EngramResidentSource`/`EngramWkvDevice` (`tp4_forward.rs:13070-13073`):

```rust
// columns = hash_columns(config) (24), head_dim = config.engram_head_dim (256), width = head_dim,
// scale_width = head_dim.div_ceil(32) (8). One arena entry per (layer slot, position):
//   [columns * head_dim] E4M3 codes, then [columns * scale_width] E8M0 scales.
let codes_len = columns * head_dim;                       // 6,144
let scales_len = columns * scale_width;                   // 192
let stride = align_up(codes_len, ALIGN) + align_up(scales_len, ALIGN); // 6,400
let mut arena = vec![0u8; config.engram_layer_ids.len() * total_positions * stride];
let mut skip = vec![vec![false; total_positions]; config.engram_layer_ids.len()];
let mut hashes = Vec::new();
for position in 0..total_positions {
    let history = prompt_tokens.get(..=position)?;          // the loop's own history, tp4_forward.rs:13414
    ngram_hashes(config, history, position, &mut hashes)?;  // the loop's own call, tp4_forward.rs:13420
    for (slot, &layer) in config.engram_layer_ids.iter().enumerate() {
        let row = &hashes[slot * columns..(slot + 1) * columns];
        let Some(table_rows) = config.engram_rows(layer) else { continue };
        if row.contains(&HASH_SENTINEL) || row.iter().any(|&id| id as usize >= table_rows) {
            skip[slot][position] = true;    // engram.rs:919-921: the seam skips the whole row
            continue;
        }
        // One range read per array per position: the ranges concatenate in row order, which is
        // exactly the arena's layout (read_paged_table's offsets, engram.rs:517-524).
        let codes_ranges: Vec<(u64, usize)> = row.iter().map(|&id| (u64::from(id) * width as u64, width)).collect();
        let scale_ranges: Vec<(u64, usize)> = row.iter().map(|&id| (u64::from(id) * scale_width as u64, scale_width)).collect();
        // read_ranges_paged returns Option<Result<..>>: None means the source serves whole tensors
        // only, and the same rows then come from the read_table fallback read_paged_table uses.
        let codes = source.read_ranges_paged(&format!("layers.{layer}.engram.embed.weight"), &codes_ranges)
            .ok_or("source serves whole tensors only")?.map_err(..)?;
        let scales = source.read_ranges_paged(&format!("layers.{layer}.engram.embed.scale"), &scale_ranges)
            .ok_or("source serves whole tensors only")?.map_err(..)?;
        let entry = (slot * total_positions + position) * stride;
        arena[entry..entry + codes_len].copy_from_slice(&codes);
        arena[entry + codes_len..entry + codes_len + scales_len].copy_from_slice(&scales);
    }
}
// One H2D of the whole arena at open, into the run-scope handle.
```

Costs: 2 layers × 138 positions × 6,400 B ≈ **1.77 MB** device, one upload, and no per-step I/O. The
bytes are the ones `warm_engram_rows` already touches (`tp4_forward.rs:13133`), so the run adds no new
I/O family; the only change is that the bytes are kept. (Alternative the Lead may prefer if it wants
fewer host lines: capture the *quantised* activations from `engram_forward_with_projection` with a
projection closure and skip the embed launch — correct as well, but it moves the released quantiser to
the host, so the arena-of-gathered-rows form above is the one recommended.)

Also build at open: `product` per engram layer (f32 `[hc*dim]`, 80 KiB, `q_weight * k_weight`,
`engram.rs:906-911`) and the wkv weight/scale device slices exactly as `EngramWkvDevice::build` does
(`tp4_forward.rs:5900-5980`, from `resident_bytes` of `layers.{N}.engram.wkv.weight/.scale`). The
handle owns: the arena, the skip flags, `product[2]`, the wkv pair per engram layer, and one scratch
set (§4.5).

Sentinel skip at runtime: `skip[slot][position]` is host metadata fixed at open (the plan and the token
ids are), so the stage returns `Ok(())` before any launch — deterministic on all four ranks, which is
the divergence rule (`layer_device.rs:1572-1578`).

### 4.2 Branch (i) — recommended: the declared write kernel

Per engram layer and position, after the skip check, six launches on the arm's stream; no copy, no
allocation, no readback:

```rust
let e = self.engram.as_ref().ok_or("engram_write: the arena was not built at open")?;
let slot = e.layer_slot(layer)?;                 // 0 for layer 1, 1 for layer 14 (config order)
let stream = self.buffers.residual.stream;       // f32 [hc*dim]
let (codes, scales) = e.rows_at(slot, position); // the arena entry, both 256-B aligned

// 1. dequantise the 24 table rows and quantise the fetched row: the landed embed kernel does the
//    E4M3/E8M0 -> f32 -> bf16 -> E4M3/E8M0 chain that quantise_fetched does on the host.
engram_embed_launch::launch(self.cuda, self.engram_embed, &EngramEmbedLaunchParams {
    row_codes: codes,                                // [columns][head_dim] E4M3
    row_scales: scales,                              // [columns][head_dim / 32] E8M0
    fetched: e.fetched_bf16,                         // [columns * head_dim] bf16 scratch
    act_codes: e.act_codes,                          // [columns * head_dim] E4M3 scratch
    act_scales: e.act_scales,                        // [columns * head_dim / 32] E8M0 scratch
    rows_cols: e.columns as u32,                     // 24
    head_dim: e.head_dim as u32,                     // 256
    trace_ring: 0, layer_id: layer as u32, step_seq: 0,
})?;

// 2. the W8A8 projection over that activation row (landed kernel).
engram_wkv_launch::launch(self.cuda, self.engram_wkv, &EngramWkvLaunchParams {
    act_codes: e.act_codes,
    act_scales: e.act_scales,
    weight: e.wkv_codes[slot],                       // resident, uploaded at open
    weight_scales: e.wkv_scales[slot],
    out: e.kv_bf16,                                  // bf16 [dim * (hc + 1)]
    out_width: e.out_width as u32,                   // dim * (hc + 1) = 25,600
    inner: e.inner as u32,                           // 6,144
    trace_ring: 0, layer_id: layer as u32, step_seq: 0,
})?;

// 3. narrow the arm's f32 stream to the bf16 the gate kernel reads (exact: bf16_rne; see §4.4).
engram_convert_launch::launch_narrow(
    self.cuda,
    self.engram_narrow,
    &EngramConvertParams {
        src: Deviceptr(stream.device),               // f32 [hc*dim]
        dst: e.hs_bf16,                              // bf16 [hc*dim]
        count: (self.hc_mult * self.dim) as u32,
    },
    &EngramConvertSpans { src_bytes: stream.len, dst_bytes: e.hs_bytes },
)?;

// 4. the gate (landed kernel; its own write is discarded into hs_bf16, a safe alias, §2).
engram_gate_write_launch::launch(self.cuda, self.engram_gate, &EngramGateWriteLaunchParams {
    kv: e.kv_bf16, stream: e.hs_bf16, product: e.product[slot],
    out: e.hs_bf16,        // discarded: alias of the stream operand, safe after the kernel's barrier
    gates: e.gates,        // f32 [hc]
    dim: self.dim as u32, hc: self.hc_mult as u32, eps: self.engram_eps,
    trace_ring: 0, layer_id: layer as u32, step_seq: 0,
})?;

// 5. widen the wkv value tail to the f32 the declared write takes (exact; §4.4).
engram_convert_launch::launch_widen(
    self.cuda,
    self.engram_widen,
    &EngramConvertParams {
        src: e.kv_bf16.byte_offset((self.hc_mult * self.dim) as u64 * 2), // the value row, bf16
        dst: e.value_f32,                                                 // f32 [dim]
        count: self.dim as u32,
    },
    // The source span is the kv span from the value row's offset; the guard only needs it to be
    // at least dim bf16 values long.
    &EngramConvertSpans { src_bytes: e.kv_bytes, dst_bytes: e.value_bytes },
)?;

// 6. the stage's declared write (this task's kernel).
let params = EngramWriteParams {
    stream: Deviceptr(stream.device),
    gate: e.gates,
    value: e.value_f32,
    stream_out: Deviceptr(stream.device),   // in place: the stage's contract
    dim: self.dim as u32,
    copies: self.hc_mult as u32,
    trace_ring: 0,
    layer_id: layer as u32,
    step_seq: 0,
};
engram_write_launch::launch(self.cuda, self.engram_write, &params, &EngramWriteSpans {
    stream_bytes: stream.len,
    gate_bytes: self.hc_mult * 4,
    value_bytes: self.dim * 4,
    stream_out_bytes: stream.len,
})?;
```

Each launcher above validates its own geometry before the driver call (the landed
`engram_embed_launch`/`engram_wkv_launch`/`engram_gate_write_launch` check their shapes; this task's
`engram_write_launch` additionally refuses a short span by name, §1). The arena/scratch pointers are
the handle's own fields, so their sizes are checked once in its builder, not per launch.

Numeric equivalence with the host seam: steps 1–2 are the landed kernels whose CPU twins
(`engram_native_ref::embed_row`/`act_quantise` and `wkv`) reproduce them op for op and reduction for
reduction (module doc, `engram_native_ref.rs:1-30`); step 4's gates are the host's gates (same
`bf16_round(h)`, same product, same ascending sum — `engram_native_ref::gate`, and the head's only
declared deviation is the sigmoid's F8-class exponential); step 6 is bit-identical to `engram_write`
by §1's twin test. So the stage's stream equals the host arm's stream up to the declared sigmoid class
— that is the D54 claim the Lead can measure on a node.

### 4.3 Branch (ii) — the landed fused `engram_gate_write`, declared kernel left unwired

The Lead's alternative: drop steps 5–6 and take `engram_gate_write_sm121`'s bf16 result back to the
f32 stream with one widen — steps 1, 2, 3, then 4 with `out` in a bf16 scratch (`e.hs_bf16` is the
safe alias), then `widen(e.hs_bf16 -> stream f32)`. Five launches instead of six, no
`engram_write_bf16` launch — the declared symbol stays the interface the contract names, but the
running arm does not execute it.

Differences the Lead should weigh:

* branch (ii) needs **no** `value_f32` scratch and no step 6, and it *does* need a second bf16 scratch
  for `out` if it prefers not to alias `hs_bf16` (both are 40 KiB); §4.1/§4.4/§4.5 otherwise
  unchanged.
* branch (ii) writes the bf16 result and then widens it; the widen is exact (bf16 -> f32), so the
  stream values are identical to branch (i)'s. The costs are one extra 80 KiB read+write per engram
  layer-position and a stage whose declared kernel is never launched.
* branch (i) exercises the declared ABI and leaves the landed fused kernel as the *cross-check* (the
  two writes can be compared bit for bit on a node, since both are the released arithmetic).

Recommendation: branch (i). It is one launch more, it makes the contract's kernel the one that runs,
and the extra widen is a 20 KiB conversion. Branch (ii) is the fallback if the Lead prefers not to
keep two write paths alive.

### 4.4 The two conversion launches (landed): one cubin, two symbols

The arm's stream is f32; the released stream (and every landed gate/write kernel) is bf16. The
conversions are exact in both directions for this data (an f32 that is already a bf16 value narrows
to itself; bf16 → f32 is a bit insertion), so they add no numeric class. They cannot live in the
pinned engram cubins (that would change their digests), so they are their own cubin, landed in this
task:

| File | Contents |
| --- | --- |
| `kernels/engram_stream_convert_sm121.cu` | `engram_narrow_bf16` (f32 -> bf16) and `engram_widen_f32` (bf16 -> f32), both grid-strided |
| `crates/sparkflash-kernels/src/engram_convert_launch.rs` | pinned declarations for both, `Direction`, `EngramConvertParams`/`EngramConvertSpans`, validate/grid/check_spans, `load_narrow`/`load_widen`, `launch_narrow`/`launch_widen`, CPU-twin test |
| `crates/sparkflash-kernels/src/lib.rs:39` | `pub mod engram_convert_launch;` |
| `crates/sparkflash-kernels/src/cuda_driver.rs:3063,3079` | `Cuda::load_engram_narrow` / `Cuda::load_engram_widen` (by name; neither symbol has a `KernelId`) |

The two declarations, exactly as the launcher pins them (give these to the probe/pin step):

```
extern "C" __global__ void engram_narrow_bf16(
    const float *__restrict__ src, __nv_bfloat16 *__restrict__ dst, const uint32_t count)

extern "C" __global__ void engram_widen_f32(
    const __nv_bfloat16 *__restrict__ src, float *__restrict__ dst, const uint32_t count)
```

Launch shape: `count` elements, grid `(min(ceil(count / 256), 65_535), 1, 1)` (both kernels are
grid-strided, so the cap only bounds the grid), block `(256, 1, 1)`. Neither carries the trace trio:
the repo's rule for a non-stage kernel is `shared_swiglu_f32`'s (`layer_seams_launch.rs:20-23,41-43`)
- no ring, and the stage that launches it records the trace. A test pins that absence.

**Lead registration note (build.rs/baseline are yours).** The declarations are the two above; the
cubin stem is `engram_stream_convert_sm121`. One wrinkle to be aware of: `PROBED_KERNELS`
(`crates/sparkflash-kernels/build.rs:172-212`) is one `(file, symbol)` pair per file and the probe
`#include`s every file exactly once, so a second symbol in the same file needs either an entry type
that carries a symbol list or a probe that only checks this file compiles. I did not touch build.rs,
the baseline manifest or the staged cubin dir.

Wiring for the stage (both branches): narrow `residual.stream` before the gate kernel (step 3 of
§4.2), and widen the wkv value tail after the projection (step 5). The narrow is *lossless by
construction* under `--native-rounding mhc` (`native_bf16`, `tp4_forward.rs:10568`), and it is still a
launch because the issue is the span layout (f32 vs bf16), not the rounding.

### 4.5 Spans: one run-scope handle, no `LayerBuffers` change

Do **not** add per-layer spans for this stage: `LayerBuffers` is one recipe for all 40 layers
(`layer_device.rs:615-748`: `bytes` and `carve` must stay in lockstep, and
`assert_contract_widths`'s 25-entry array, `layer_device.rs:501-607`, has to be kept in step), and the
engram spans would be dead for 38 of them. Only one engram layer runs at a time, so **one** scratch
set serves both:

| Run-scope allocation (new `EngramWriteDevice` handle) | Size (released) |
| --- | --- |
| arena: the 24 gathered rows per (layer, position), `[2][total_positions][6,400]` | 1.77 MB |
| `fetched_bf16` bf16 `[columns*head_dim]` (embed output) | 12 KiB |
| `act_codes` E4M3 `[columns*head_dim]`, `act_scales` E8M0 `[columns*head_dim/32]` (embed output) | 6.2 KiB |
| `hs_bf16` bf16 `[hc*dim]` (narrow out; also gate_write's discarded out) | 40 KiB |
| `kv_bf16` bf16 `[dim*(hc+1)]` (wkv out) | 50 KiB |
| `value_f32` f32 `[dim]` (widen out, branch (i) only) | 20 KiB |
| `gates` f32 `[hc]` | 256 B (aligned) |
| `product[2]` f32 `[hc*dim]` (constant, uploaded at open) | 160 KiB |
| wkv (codes, scales) per engram layer — as `EngramWkvDevice` already uploads | 314 MB (two 157 MB matrices) |
| skip flags, `[2][total_positions]` (host-side only) | — |

The 314 MB is the same run-scope upload the host arm's device-wkv overlay already performs
(`tp4_forward.rs:5900-5980`), so it is an accepted memory charge, not a new one — but the Lead should
confirm the device arm has the budget side by side with the host arm (they are different arms; only
one runs).

### 4.6 Open-time plumbing

* Build the handle where the arm builds its other run-scope device state (next to
  `EngramWkvDevice::build`, `tp4_forward.rs:13070-13073`), and pass it into `DeviceLayer::open`
  (`tp4_forward.rs:10732`, which already takes `Option<&RankBind>` and the plan — one more
  `Option<&EngramWriteDevice>` fits the existing arity note at `10726-10731`). `None` keeps the
  stage's refusal (no silent host fallback).
* Load the kernels at open with `load_pinned_addition` (`tp4_forward.rs:8007`, one call per cubin
  stem, `baseline::pin` by stem): `engram_embed_sm121`, `engram_wkv_sm121`, `engram_gate_write_sm121`
  (all three already pinned; the host arm loads the first two by `KernelId` in their launchers),
  `engram_write_sm121` — the **stem is `engram_write_sm121` while the symbol is
  `engram_write_bf16`**, so pass the stem to `load_pinned_addition` and the returned bytes to
  `engram_write_launch::load` — and `engram_stream_convert_sm121`, whose bytes go to
  `engram_convert_launch::load_narrow` and `::load_widen` (two symbols, one cubin). All five
  loaders resolve by name; no new `KernelId` is needed.
* `eps`: the released Engram eps is the config's `norm_eps` as f32 (`engram.rs:911`), not the mHC
  eps the arm already carries (`self.hc_eps`) — take it from config in the handle.
* `step_seq`: the arm has no per-launch sequence counter today (the seam kernels pass 0); pass 0 and
  let the trace arm add it, consistent with `kv_path`'s launches.

### 4.7 `begin_position`: where the gather index scalar goes (and why this slice does not need it)

**This slice (at-open arena):** no scalar, no `begin_position` change. The arena offset is
`layer_slot * stride + position * stride` — a host-side launch argument, exactly like the rope table
offset `kv_path` already computes from `position` (`tp4_forward.rs:12141-12160`). Writing a 4-byte
device scalar would be a second H2D per position for no reader. `begin_position`
(`tp4_forward.rs:11851-11881`) keeps exactly its two copies: the stream row (`copy_in`,
`tp4_forward.rs:11874`) and the identity seed (`11876-11881`).

**Device-gather variant (D39 rule 4's target, next slice):** the gather needs the position *inside*
the launch, so the launch's argument array can be position-independent. Then:

* span: one new 4-byte `u32` span, 256-byte aligned, e.g. `LayerResidual::engram_position` (added to
  `LayerResidual`, `LayerBuffers::bytes` and `carve`, and to `assert_contract_widths`);
* write, immediately after the stream upload in `begin_position`:

```rust
self.cuda.copy_htod(
    &Deviceptr(self.buffers.residual.engram_position.device),
    &u32::try_from(position)
        .map_err(|_| format!("layer device begin_position: position {position} does not fit u32"))?
        .to_le_bytes(),
)?;
```

* readers: exactly one launch — the gather (new kernel, not landed). The embed/wkv/gate/write
  launches of §4.2 take position through host-computed offsets and never read the scalar. The gather
  reads `position` as `const uint32_t *`, indexes the per-position hash-id row and the hot-row slot
  map, and fills the arena entry the embed kernel then reads. This is the piece D39 rule 4 names; it
  is not in P1 and I would take it as its own task (it needs the hot-row arena and the slot map, both
  of which live in the resident/paging slice, not this one).
* ordering: the scalar write must precede the position's gather on the same stream; because it is
  enqueued in `begin_position`, before `run_layer`, that holds for every layer.

## 5. `stage_attention`: verified wiring note (validation-only)

The contract decision is in the module header: `stage_attention` stays and becomes **validation-only**;
it launches nothing and copies nothing (`crates/sparkflash-kernels/src/attention_stage.rs:1-13`). The
Lead's instruction was to verify the semantics other slices will wire against. Verified facts, with
sources:

* `WINDOW_ROWS = 128` (`attention_stage.rs:44`).
* `window_span(position, window)` (`attention_stage.rs:149-153`): `window_rows = min(position + 1,
  window)` — the **current token is included** — and `window_start = position + 1 - window_rows`.
  `position` is the 0-based token index p, not p+1.
* `assemble(...)` (`attention_stage.rs:164-202`) runs its five checks in this order and returns
  nothing until all pass: `ring_rows <= max_positions`; `position < ring_rows` (the append must
  precede the stage); `cursor == store.rows_written` (the cursor word and the store must agree, or
  `store_zero` would be a guess); then `selected_rows(ids, window_rows, ratio)`; and
  `window_rows + selected_count` room is the body's own check
  (`flash_mla_csa2_seq_sm121.cu:217-218`).
* The id space: `StageOperands::selected` (`attention_stage.rs:63-78`) is in the **compressed-position
  space offset by the window COUNT**, `id = window_rows + j` (j = compressed row), **not** by
  `window_start`. Mapping to store rows is `(id - window_rows) * ratio`
  (`compressed_cache_ref.rs:313`, `selected_rows` at `:454`), which is exactly
  `compressed_selected_fill_sm121`'s contract: `(ids, n_ids, window_rows, ratio, out, out_count)`
  (`compressed_append_launch.rs:67-100`) — pass the same `window_rows` there. Do not pass
  `window_start`: it slides and would shift every id by up to 128.
* Ring addressing (already used by `kv_path`, `tp4_forward.rs:12200-12208`):
  `slot = kv_ring.slot_for(position)`, `codes = kv_ring_base + kv_ring.layer_offset(layer) +
  kv_ring.codes_offset(slot)`, scales likewise (`kv_path_ref.rs:432-537`, `slot_for` at `:504`,
  `codes_offset` at `:513`, `scales_offset` at `:522`, `layer_offset` at `:486`). The window reads
  rows `window_start .. window_start + window_rows` in ring order, so a window that wraps crosses the
  slot boundary once; the body's `window_codes`/`window_scales` operands are **contiguous**
  `[window_rows][ROW_BYTES]` (`FlashMlaSeqParams::new`, `flash_mla_csa2_seq_launch.rs:110-130`; the
  host arm allocates exactly that, `tp4_forward.rs:3292-3295`), so the device arm must linearise the
  ring window into a contiguous staging span — at most two device-to-device runs, no host copy, and
  the stage records `window_rows` = the committed operand.
* `store_zero`: true while the compressed store has published no row. The body takes
  `selected_count` as a **host word** and reads exactly that many ids, each as an absolute compressed
  position, with no `NO_INDEX` skip (`flash_mla_csa2_seq_sm121.cu:177-185,335,465`). So the device
  arm's select must publish a **dense** id list with a host-known count; the count is computable from
  the plan and the position (`min(topk, published compressed candidates)`) with no readback, or —
  if the select can return `NO_INDEX` padding — the fill kernel's device `out_count` is not usable as
  the body's word and the Lead must either guarantee density or accept one 4-byte D2H. Flag it, do
  not silently pass `topk`.
* `FlashMlaSeqParams` words (`flash_mla_csa2_seq_launch.rs:104-152`): `window_kv_format =
  FORMAT_CSA_UNION` (2, `:55`), `window_rows`, `position` (the 0-based p), `selected`,
  `selected_count`, trace trio. Grid `(HEADS, 1, 1)`, 19 words — the two words that must come from
  `attention_stage` are `window_rows` and `position`; `window_start` is only the ring/window base.
* Ordering in `run_layer` (`layer_device.rs:1579-1599`): `kv_path` (append) → `ref_select` →
  `stage_attention` (validate) → `attention_body` (launch) → `attention_output`. `kv_path`
  increments `appended_rows[layer]` after its append (`tp4_forward.rs:12231-12233`), so
  `ring_rows = self.appended_rows[layer]` at the stage is exactly `position + 1`, which is the input
  `assemble`'s `position < ring_rows` check expects.
* There is no host `cache::Plan`/`cache::Store` in the device arm (rule 7: the arm owns the device
  cache, the host cache is not a second view). So the device `stage_attention` body should call
  `window_span` and `window_rows` room checks directly, and take `cursor`/`store_zero` from the same
  run-scope counters the compressed-append stage maintains (mirroring
  `compressed_cache_ref::Store::rows_written`/`zero`). Do not construct a host `Store` to call
  `assemble`; that would be a second view.

`tp4_forward.rs:12246-12248` is the stub that shrinks to this note's body; `attention_body`
(`:12250`) stays a refusal until its slice lands.

## 6. What is NOT verified on this host

* No device launch of any Engram kernel (no GPU, no `/dev/nvidia*`), so: no bit comparison of
  `engram_write_bf16` against the twin on device, no D54 numerics, no gate, no step timing, and no
  check that the launch's `words()` map correctly through the driver's `cuLaunchKernel` packing
  beyond the declaration pin and the PTX compile.
* No end-to-end arm run: the wiring in §4 is a design plus API-exact snippets, not compiled code —
  it touches files the Lead owns (`tp4_forward.rs`, `layer_device.rs`) and was **not applied**. After
  the conversion kernels, the wiring is still yours: arena + handle + open plumbing + the six-launch
  stage body. I did not edit either file.
* The conversion kernels of §4.4 are compiled to PTX and unit-tested here, but not launched, not
  pinned, and not registered: the cubin/staged-dir/baseline step is yours (build.rs is untouched).
* Everything device-side needs the node: `cargo test`/clippy/fmt are the local bar, and the Lead's
  nvcc baseline build is the next gate for `kernels/engram_write_sm121.cu`.

## 7. Open decisions for the Lead

1. **D39 rule 4** — resolved by your ruling (c): the at-open arena stands **with the declaration in
   §3.1** (run-shape assumption, device gather explicitly deferred, rule 4 not closed). §3.1 also has
   the draft stage-record wording; if you want the gather kernel as its own slice after S1, that is a
   separate task and I can take it.
2. **Branch (i) vs (ii)** (§4.2/§4.3). I recommend (i).
3. **The two conversion launches** — **resolved: landed** (§4.4). Your remaining step is the
   registration: the two declaration lines are in §4.4 (watch the one `PROBED_KERNELS` pair-per-file
   wrinkle noted there).
4. **The `--native-rounding mhc` bf16 stream shadow**: if the arm later keeps the stream as bf16 end
   to end, the narrow launch can disappear (the value widen stays, the wkv tail is bf16 and the
   declared write takes f32). Not needed for this slice.
5. **`selected_count` density** for `stage_attention` (§5): whose slice guarantees a dense id list.
