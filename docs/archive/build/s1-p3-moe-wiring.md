# S1 P3 — the `moe_routed` device data path: what landed, and the wiring the Lead adds

Branch `integration/s1-layer-device`, owner **moe-owner** (shared task `task-3`). This is the P3
deliverable: (1) the two kernels that close the stage's data path, (2) the typed module that owns the
run-scope device state and transcribes the arithmetic bit for bit, (3) this note — the exact wiring for
`DeviceLayer::moe_routed`, the open-time expert-pointer-table build, the span checks, the refusal
removal, and the kernel registration the Lead owns.

**No device claim in this file is measured.** This host has no CUDA device (`/dev/nvidia*` absent), so
every device launch, every numeric comparison, the D54 gate and every timing number is **NOT RUN**.
What *is* verified is listed in §7 with the exact command and output. Nothing here should be read as
evidence that the stage runs, matches the host arm numerically, or meets the 120 ms/step budget.

## 0. Summary

* `kernels/moe_routed_device_sm121.cu` lands the two symbols the stage is missing: a staging gather
  (`moe_expert_stage_sm121`) and the routed combine epilogue (`moe_routed_combine_sm121`).
* `crates/sparkflash-kernels/src/moe_routed_device.rs` lands the typed plan, the device buffer plan,
  the expert pointer table, the two launch wrappers, the landed-launch wrappers the stage drives, the
  run-scope `MoeRoutedDevice` (allocate once, eight launches per stage call, zero copies), and the CPU
  twins of the two kernels' arithmetic, pinned bit-for-bit against `mxfp4_ref`.
* The stage as the layer port will call it is **eight launches and zero copies per layer**: three
  staging gathers, the two landed gate/up launches, the landed weighted SwiGLU, the landed stored down
  projection, and this module's combine. The host arm's per-layer work that disappears is: the six
  slot-id readbacks, the three slot-ordered staging `copy_dtod` sets (issued by the host from ids it
  read back), the 122,880 B slot-accumulator readback, the host `moe_combine_routed_f32` +
  `bf16_le_bytes`, and the 10,240 B payload upload — 133 KB of PCIe and one host combine per layer.
* The stage's refusal becomes a refusal only for a *missing* prewarm (`attach_moe_routed` not called),
  a *stale* id row (the router did not run for this position), or a *short* carved span. The five
  launches plus the two new ones need no host-visible intermediate, so the refusal text
  `the five device MoE launches exist; the host combine of their outputs is the step`
  (`layer_device.rs:1449-1451`) stops being true and is deleted (§5.7).
* **The Lead's integration steps** are: five wiring edits in `tp4_forward.rs` (§5), two by-name loaders
  in `cuda_driver.rs` (§6.2), one cubin registration across nine files (§6), and the `layer_device.rs`
  contract/blocker text (§5.7). Everything is written out below as pasteable code. The Lead owns all of
  it, in one registration pass for this milestone's three new cubins (§6).
* Measured pin for the registration (this host's toolkit, same release/build the manifest records):

  ```
  kernel moe_routed_device_sm121 cubin_bytes=28144 cubin_sha256=19bcadd0f38a9886fa04aa935445eb48f6bfcbcc4a3dd422ace6d2cdfaadc136 ptx_membar_lines=0
  ```

## 1. What landed, and what was verified on this host

Files added by this task (committed by the Lead, never by me):

| File | Contents |
| --- | --- |
| `kernels/moe_routed_device_sm121.cu` | `moe_expert_stage_sm121` (`:155`), `moe_routed_combine_sm121` (`:199`), the two rounding helpers and the staged copy helper; geometry `static_assert`s against `m4_common_sm121.cuh` |
| `crates/sparkflash-kernels/src/moe_routed_device.rs` | geometry/plan/sizes, `ExpertPointerTable`, the two ABI pins (declarations, constants, static asserts, measured pin), `MoeRoutedKernels`, `MoeRoutedDevice` (`open`/`run`/`release`), the launch wrappers, the CPU twins, 24 CPU tests + 2 `#[ignore]` device tests |
| `crates/sparkflash-kernels/src/lib.rs:63` | `pub mod moe_routed_device;` (after `mla_ref`) |
| `docs/archive/build/s1-p3-moe-wiring.md` | this note |

### 1.1 The two pinned declarations

`moe_expert_stage_sm121` — 10 words, grid `(rows, 1, 1)`, block `(256, 1, 1)`, one CTA per routed row:

```
extern "C" __global__ void moe_expert_stage_sm121(
    const uint64_t *__restrict__ table,        // [layer][matrix][expert] (weight, scales) device addresses
    const uint32_t *__restrict__ expert_ids,   // [rows] global expert ids, device-resident
    uint8_t *__restrict__ dst_weight,          // [rows][weight_bytes] staged packed FP4
    uint8_t *__restrict__ dst_scales,          // [rows][scale_bytes] staged ue8m0 grids
    float *__restrict__ slot_acc,              // [rows][acc_row_floats] f32, zeroed when acc_row_floats != 0
    const uint32_t rows,
    const uint32_t experts,
    const uint32_t weight_bytes,
    const uint32_t scale_bytes,
    const uint32_t acc_row_floats)
```

`moe_routed_combine_sm121` — 6 words, grid `(tokens, ceil(hidden / 256), 1)`, block `(256, 1, 1)`:

```
extern "C" __global__ void moe_routed_combine_sm121(
    const float *__restrict__ slot_acc,        // [tokens * top_k][hidden] f32
    const uint32_t *__restrict__ expert_ids,   // [tokens * top_k] global expert ids
    float *__restrict__ delta_f32,             // [tokens][hidden] f32 routed sum
    uint16_t *__restrict__ delta_bf16,         // [tokens][hidden] bf16 payload
    const uint32_t tokens,
    const uint32_t hidden)
```

Both are pinned inside the module the same way the landed ABI modules pin theirs: the declaration is
parsed out of the source (`c_decl::kernel_declaration`), compared parameter by parameter
(`declaration_matches`, which requires the exact name/type/pointee-const/`__restrict__` shape), and a
mutation test proves the comparison is not vacuous. The `static_assert`s are evaluated out of
`Consts::parse` + the shared M4 header, so a kernel that stops pinning its geometry fails a unit test.

### 1.2 The measured pin

The cubin and the PTX this host's CUDA 13.0 front end produce from the file:

```
$ /home/user/cuda-tk/bin/nvcc -std=c++17 -arch=sm_121a -cubin -I kernels kernels/moe_routed_device_sm121.cu -o /tmp/moe_routed_device.cubin
$ wc -c < /tmp/moe_routed_device.cubin   ->  28144
$ sha256sum /tmp/moe_routed_device.cubin ->  19bcadd0f38a9886fa04aa935445eb48f6bfcbcc4a3dd422ace6d2cdfaadc136
$ /home/user/cuda-tk/bin/nvcc -std=c++17 -arch=sm_121a --ptx ...
$ grep -c membar /tmp/moe_routed_device.ptx ->  0
$ grep '\.visible \.entry' /tmp/moe_routed_device.ptx
  .visible .entry moe_expert_stage_sm121(
  .visible .entry moe_routed_combine_sm121(
```

The toolkit is `V13.0.88` / `cuda_13.0.r13.0/compiler.36424714_0` — the exact release and build the
manifest already records, so the Lead can regenerate the baseline **on this host** (§7.2) instead of
transcribing the line by hand.

## 2. Why these two kernels: the five landed launches and the two host steps

### 2.1 The five launches the stage already had

`run_layer_moe_routed` (the host arm, `tp4_forward.rs:4994-5400`) issues, on device:

1. `moe_router_sm121` + `moe_router_topk_sm121` — six distinct in-range global ids and six routing
   weights, in `workspace.moe_ids` / `workspace.moe_weights`. Already on device in the arm.
2. `routed_gate_up_sm121` twice — gate and up, indexing their weight operand by a **staged slot**
   (`weight_row = expert * width + column`), consuming `workspace.act_codes` / `act_scales`.
3. `swiglu_mxfp8_quant_weighted_sm121` — SwiGLU, routing weight applied **before** the bf16 cast, then
   the E4M3/ue8m0 quantise into `rows x 640` codes and `rows x 20` scales.
4. `b12x_mxfp4_moe_stored_sm121` — the down projection, added into the f32 slot accumulator
   (`slot_acc[token * top_k + slot] += acc`), again by staged slot.

All five are pinned, loaded and unit-tested already. None of them can run in the arm, for two reasons.

### 2.2 The two host steps that kept the stage refused

* **The staging.** The gate/up/stored launches index a slot-ordered buffer, and building that buffer is
  a per-layer copy of six `(weight, scale)` slices from each selected expert's resident-arena shard.
  The host does it with `copy_dtod` per slot (`tp4_forward.rs:5128-5180`) **because the ids are host
  values there** (`copy_dtoh` of `moe_ids`, then `moe_slots`). In the arm the ids exist only on device,
  so a host copy cannot express the gather.
* **The combine.** After the stored launch the per-slot f32 accumulators are on device, and the host
  arm reads all 122,880 B back, runs `mxfp4_ref::moe_combine_routed_f32` (per-slot bf16 round, then the
  six f32 adds in ascending expert-id order), narrows with `bf16_le_bytes`, and uploads the 10,240 B
  payload for the second collective. That readback *is* the step the refusal names.

### 2.3 Why a gather kernel and not "just a combine kernel"

A combine kernel alone is not enough: with the ids on device, nothing on the host can build the
slot-ordered weight buffer the landed launches read. The gather has to be a kernel for the same reason
the combine does — the ids are a device value. Conversely, the gather alone is not enough either: no
landed kernel performs the weighted sum over slots (the reference combine exists only as a host
function in `mxfp4_ref`, and `mxfp4_ref::moe_combine_finalize` is the host's post-reduce epilogue). So
both were added, and the module reuses every landed launch for the expert math.

### 2.4 One staging buffer pair instead of three

At the released geometry the staged gate/up slice (`576 x 2560 = 1,474,560 B`) and the staged down
slice (`5120 x 288 = 1,474,560 B`) are the **same size**; so are their scale slices
(`576 x 160 = 92,160 B` and `5120 x 18 = 92,160 B`). Both projections are transposes of one another.
The two landed launches are on one stream, so one staging buffer pair is written, read, and then
written again by the next gather (step 3 of §3.1) — stream order is what makes the reuse safe. That is
why the run-scope state is 9.8 MiB instead of 3x that; `run` refuses a geometry where the equality
stops holding, before a launch could stage over a shorter slice than its reader.

## 3. The device data path

### 3.1 The eight launches, in stream order, per stage call

| # | launch | operands | notes |
| --- | --- | --- | --- |
| 1 | `moe_expert_stage_sm121` (gate) | table slice `[layer][Gate]`, `moe_ids`, staged weight/scales | one CTA per routed row; copies each row's slice |
| 2 | `routed_gate_up_sm121` | `act_codes`, `act_scales`, staged gate pair, **staged slot ids**, `gate_out` | output `[rows][640]` bf16 |
| 3 | `moe_expert_stage_sm121` (up) | table slice `[layer][Up]`, same buffers | reuses the staging buffer after launch 2 read it |
| 4 | `routed_gate_up_sm121` | same, `up_out` | |
| 5 | `swiglu_mxfp8_quant_weighted_sm121` | `gate_out`, `up_out`, `moe_weights`, codes/scales | routing weight applied, then quantised |
| 6 | `moe_expert_stage_sm121` (down) | table slice `[layer][Down]`, **zeroes `slot_acc`** | `acc_row_floats = dim` |
| 7 | `b12x_mxfp4_moe_stored_sm121` | codes/scales, staged down pair, staged slot ids, `slot_acc` | adds one rank's partial per slot |
| 8 | `moe_routed_combine_sm121` | `slot_acc`, `moe_ids`, `moe_delta`, `moe_delta_bf16` | the epilogue; reads no weight |

Eight launches, zero copies. The "staged slot ids" are **constant**: row `r` uses slot `r % top_k`,
i.e. `[0, 1, 2, 3, 4, 5]` for the decode, uploaded once at open. The slot ids are an addressing device,
not the expert selection: the gather already picked the expert from the device id row, so the landed
launches' slot indexing lands on exactly the slice that row needs. This is the same value the host arm
computes and uploads, and it is position-invariant, which is why it is uploaded at open.

`workspace.moe_delta` (f32) and `workspace.moe_delta_bf16` (the pre-reduce payload) are the stage's two
outputs, exactly as the buffer table says (`layer_device.rs:25-26`). `all_reduce_moe` then reduces the
bf16 row; nothing in this module reduces or touches the shared expert.

### 3.2 The run-scope buffers (released, one token = six routed rows)

| buffer | bytes | why |
| --- | --- | --- |
| pointer table | 737,280 | 40 layers x 3 matrices x 384 experts x 2 addresses x 8 B |
| staged weight | 8,847,360 | 6 rows x 1,474,560 |
| staged scales | 552,960 | 6 rows x 92,160 |
| gate out | 7,680 | 6 rows x 640 bf16 |
| up out | 7,680 | 6 rows x 640 bf16 |
| codes | 3,840 | 6 rows x 640 E4M3 |
| scales | 120 | 6 rows x 20 ue8m0 |
| slot accumulator | 122,880 | 6 rows x 5120 f32 |
| slot ids | 24 | 6 u32 |
| **total** | **10,279,824** | 9.8 MiB, once per run |

The capacity is the routed-row count, not the expert count: `MAX_ROUTED_ROWS = 256` is the brief's
"up to 256 x (w_ptr, s_ptr)" read as a row capacity (see §9.1). The decode needs 6; the plan admits up
to 42 tokens' rows.

### 3.3 The expert pointer table

Built once at open, from the arena addresses `attach_routed` already resolved:

* layout `[layer][matrix][expert]`, two `u64` words per entry `(weight, scales)`;
* word offset of a slice base: `(layer * 3 + matrix.index()) * experts * 2` — the staging launch's
  `table` word is that base, and the kernel indexes `table[2 * id]` / `table[2 * id + 1]`;
* refusal at build: zero layers, zero experts, a null address (the arena resolves every bound slice, so
  a null is a wiring defect that would otherwise become a kernel reading the null page);
* `matrix.index()` is `Gate = 0, Up = 1, Down = 2`, the same order as
  `DeviceLayer::routed[layer][expert]` (`attach_routed`'s `[Gate, Up, Down]` array), so the builder
  closure is a plain lookup and the two orders cannot drift.

Only the table and the constant slot ids cross PCIe, both at open. The kernel's id clamp (`id < experts
? id : 0`) is documented in the source as unreachable from `moe_router_topk`, whose ids are guaranteed
distinct and in range: it exists so a corrupt table index cannot become an out-of-bounds *address*
read, and it is a clamp rather than a refusal because a refusal would need a device-side trap.

## 4. Arithmetic: the combine is bit-identical to the host arm

The expert math is entirely the landed kernels'. The only new arithmetic is the combine, and it
transcribes two host formulas **as integer bit arithmetic**, not as the nearest CUDA intrinsic:

* **Per-slot round** — `mxfp4_ref::round_expert_row`:
  `if (bits & 0x7fff_ffff) > 0x7f80_0000 { (bits >> 16) | 0x0040 } else { bits.wrapping_add(0x7fff + lsb) >> 16 }`,
  then `<< 16`.
* **Payload narrow** — the CLI's `bf16_le_bytes` on the fused total: `bits.wrapping_add(0x7fff + lsb) & 0xffff_0000`,
  with **no** NaN special case.
* **Adds** — `__fadd_rn(total, value)` from `+0.0f`, matching the host's `*total += value` from `0.0f`.
  (No multiply: the routing weight was already consumed by the weighted SwiGLU, so the combine must not
  re-apply it — a test greps the combine's body for the absence of a weight operand.)

The two roundings differ **only** on a NaN input, which is why the kernel spells them out and why
`the_two_roundings_agree_on_finite_values_and_differ_only_on_a_nan` walks all 65,536 bf16 values plus
the boundary cases. The CPU twins (`combine_routed_f32`, `combine_slot_order`, `round_slot_bf16`,
`narrow_payload_bf16`) are the kernel's arithmetic in Rust, and they are tested:

* equal to `mxfp4_ref::moe_combine_slot_order` and `mxfp4_ref::moe_combine_routed_f32` on a corpus that
  includes every exponent class, both zeros, subnormals, infinities and NaN payloads, with in-order,
  reverse, duplicate and out-of-order id rows;
* the payload equal to `bf16_le_bytes` of the same f32 row;
* the single-token payload equal to `mxfp4_ref::moe_combine_finalize` with a zero shared row (the
  post-reduce epilogue's input, i.e. the contract between this stage and `ffn_epilogue`).

**Why the slot order is the host's order.** The host arm passes `moe_combine_routed_f32` the
slot-mapped id row, not the router's raw ids. `moe_slots` sorts and dedups the six global ids before
assigning slot indices, so slot `i` is the `i`-th smallest id. The device combine sorts the raw ids by
ascending global id instead — the same permutation — so the six rounded values are added in the same
order. Duplicate experts are staged and added per routed row on both paths (the host's dedup changes
*which copies* are made, not the values added), so a duplicate id row produces the same sum.

## 5. The wiring the Lead adds to `DeviceLayer`

Everything below is `tp4_forward.rs` unless stated. Line numbers are the ones at the time of writing
(the Lead is editing this file); function names are the anchors.

### 5.1 Two new fields on `DeviceLayer` (`struct DeviceLayer`, `:10558`)

```rust
    /// The routed data path's five kernel handles, loaded once at open (two new symbols, three landed
    /// launches). `None` means the run has no staged cubin directory; `moe_routed` refuses by name.
    moe_routed_kernels: Option<sparkflash_kernels::moe_routed_device::MoeRoutedKernels>,
    /// The routed stage's run-scope state: the expert pointer table and the staging buffers.
    /// `attach_moe_routed` builds it, after `attach_routed` has resolved the arena pointers.
    moe_routed_device: Option<sparkflash_kernels::moe_routed_device::MoeRoutedDevice>,
    /// The `moe_router` probe count the last `moe_routed` call consumed, so a stale id row is refused.
    moe_router_seen: u64,
```

Initialize `moe_routed_kernels: moe_routed_kernels` (from §5.2), `moe_routed_device: None`,
`moe_router_seen: 0` in the `open` constructor.

### 5.2 Load the five handles at open

Insert next to the router kernels' load (`let router_kernels = match kernel_cubin_dir {`, `:10926`), in
the same `Some(dir)` arm:

```rust
        // The routed data path's two new symbols and the three landed launches the stage drives. One
        // load at open, next to the router's, so a staged directory without the new cubin refuses at
        // open instead of at layer 0's `moe_routed`. Two symbols live in the one new cubin, and no
        // KernelId names either of them (a KernelId names one symbol), so both go through by-name
        // loaders; the three landed kernels use their own pinned loaders and their own pins.
        let moe_routed_kernels = match kernel_cubin_dir {
            Some(dir) => {
                let (stage_cubin, _sha) = load_pinned_addition(
                    dir,
                    "moe_routed_device_sm121",
                    "layer device routed data path",
                )?;
                let stage = cuda.load_moe_routed_stage(&stage_cubin)?;
                let combine = cuda.load_moe_routed_combine(&stage_cubin)?;
                let (gate_up_cubin, _sha) = load_pinned_addition(
                    dir,
                    "routed_gate_up_sm121",
                    "layer device routed data path",
                )?;
                let gate_up =
                    sparkflash_kernels::routed_gate_up_launch::load(cuda, &gate_up_cubin)?;
                let (weighted_cubin, _sha) = load_pinned_addition(
                    dir,
                    "swiglu_mxfp8_quant_weighted_sm121",
                    "layer device routed data path",
                )?;
                let weighted = cuda.load_kernel(
                    &weighted_cubin,
                    sparkflash_kernels::cuda_driver::KernelId::SwigluMxfp8QuantWeighted,
                )?;
                let (stored_cubin, _sha) = load_pinned_addition(
                    dir,
                    sparkflash_kernels::mxfp4_moe_stored_launch::KERNEL,
                    "layer device routed data path",
                )?;
                let stored =
                    sparkflash_kernels::mxfp4_moe_stored_launch::load(cuda, &stored_cubin)?;
                Some(
                    sparkflash_kernels::moe_routed_device::MoeRoutedKernels {
                        stage,
                        combine,
                        gate_up,
                        weighted,
                        stored,
                    },
                )
            }
            None => None,
        };
```

The landed cubins are already in the run's module cache (the host arm and `KernelCache` load them), so
these three loads are cache hits: same bytes, same module, no second image on the device.

### 5.3 `attach_moe_routed`, right after `attach_routed` in the attach block (`:13390-13405`)

```rust
    /// Builds and opens the routed data path's run-scope state: the expert pointer table from the
    /// arena pointers `attach_routed` resolved, and the staging buffers.
    ///
    /// Called once, after `attach_routed` and before the first position. Nothing here is per layer or
    /// per step: the table and the constant slot ids are uploaded once, and from then on a stage call
    /// touches device memory only.
    fn attach_moe_routed(&mut self) -> Result<(), String> {
        use sparkflash_kernels::moe_routed_device::{
            ExpertPointerTable, MoeRoutedDevice, MoeRoutedGeometry, MoeRoutedPlan,
        };
        if self.moe_routed_device.is_some() {
            return Err("layer device routed data path: attached twice".to_owned());
        }
        let Some(routed) = self.routed.as_ref() else {
            return Err(
                "layer device routed data path: no routed shard was attached, so there is no \
                 resident (layer, expert, matrix) slice to table"
                    .to_owned(),
            );
        };
        let Some(kernels) = self.moe_routed_kernels else {
            return Err(
                "layer device routed data path: the kernels were not loaded at open (the staged \
                 cubin directory carries no moe_routed_device_sm121)"
                    .to_owned(),
            );
        };
        let geometry = self.routed_geometry;
        let n_layers = routed.len();
        // The same released-shape refusal the host arm applies: a config the pinned launches were
        // not compiled for is an open-time error naming the field, not a per-step surprise.
        let moe_geometry = MoeRoutedGeometry::new(
            geometry.dim,
            geometry.inter,
            geometry.world,
            geometry.experts,
            geometry.top_k,
        )?;
        if n_layers != self.compress_ratio.len() {
            return Err(format!(
                "layer device routed data path: the shard has {n_layers} layer(s), the arm has {}",
                self.compress_ratio.len()
            ));
        }
        let table = ExpertPointerTable::build(n_layers, geometry.experts, |layer, matrix, expert| {
            let bind = routed[layer][expert][matrix.index()];
            (bind.weight.0, bind.scales.0)
        })?;
        // The arm stages one token per position (the carved spans are one row wide), so the capacity
        // is exactly one token's top_k routed rows; `run` refuses a wider call.
        let plan = MoeRoutedPlan::new(moe_geometry, n_layers, geometry.top_k, self.swiglu_limit)?;
        let device = MoeRoutedDevice::open(self.cuda, plan, &table, kernels)?;
        self.moe_routed_device = Some(device);
        Ok(())
    }
```

and in the attach block:

```rust
        if let Some(bind) = bound.as_ref().map(|(_arena, bind)| bind) {
            device.attach_routed(bind, request.rank)?;
            device.attach_router(bind)?;
            // The routed data path needs both: attach_routed for the pointers, the kernels for the
            // launches. It allocates 10,279,824 B once (9.8 MiB); §5.3.1 folds that number into the
            // shared prewarm's budget projection as a record, and §9.3 states the rule it must not break.
            device.attach_moe_routed()?;
        }
```

Ordering matters in one direction only: `attach_moe_routed` reads `self.routed`, so it must come after
`attach_routed`; it is prewarm work, so it must come before `loop_total_timer`.

### 5.3.1 The budget projection (record only — Lead's ruling)

The routed state is allocated in `attach_moe_routed`, outside `build_shared_expert_device`'s
`BudgetAdmission` call. The Lead's ruling is to fold its size into that projection **as a record**:
the same admission point then accounts for both run-scope states, while the bytes the shared prewarm
allocates and the number it records (`SharedExpertDevice::device_bytes` -> `shared_expert_bytes`,
`shared_prewarm_bytes`) stay exactly as they are today. The fold must therefore not change any
allocation or measured byte total; it changes only `BudgetRequest::projected_device_bytes`.

Site: `build_shared_expert_device` (`tp4_forward.rs:7263`), the `let projected_device = ...`
(`:7321-7322`). Exact expression:

```rust
    // The routed stage's run-scope state (P3): its own allocation, folded into this projection so one
    // admission accounts for both states. RECORD ONLY: it changes the projection, not the shared
    // prewarm's allocations or its measured byte total.
    let routed_device_bytes = sparkflash_kernels::moe_routed_device::MoeRoutedSizes::new(
        &sparkflash_kernels::moe_routed_device::MoeRoutedGeometry::new(
            config.dim,
            config.moe_inter_dim,
            request.world,
            config.n_routed_experts,
            config.n_activated_experts,
        )?,
        request.n_layers,
        config.n_activated_experts,
    )?
    .total as u64;
    let projected_device = request.n_layers as u64 * (weight_bytes + scale_bytes)
        + workspace_bytes as u64
        + routed_device_bytes;
```

`MoeRoutedSizes::new` is the same arithmetic `MoeRoutedDevice::open` allocates (10,279,824 B at the
released geometry and the one-token capacity), so the projection cannot drift from the allocation. If
the config is not the released shape, `MoeRoutedGeometry::new` refuses here — at setup, with the same
field-naming message `attach_moe_routed` would give.

### 5.4 The stage body: replace the refusal (`fn moe_routed`, `:12523`)

```rust
    fn moe_routed(&mut self, layer: usize, position: usize) -> Result<(), String> {
        use sparkflash_kernels::cuda_driver::Deviceptr;
        use sparkflash_kernels::moe_routed_device::{DeviceSpan, MoeRoutedOperands};
        // The ids are this position's only if the router ran since this stage last looked: the same
        // freshness rule `moe_router` applies to `ffn_seams`, for the same reason — stale ids produce
        // plausible, wrong routing, and therefore a plausible, wrong delta that no later stage can
        // detect.
        let now = probe::stage_count("moe_router");
        if now == self.moe_router_seen {
            return Err(format!(
                "layer device stage moe_routed (layer {layer}, position {position}): moe_router has \
                 not run since this stage last ran (probe count unchanged at {now}), so this \
                 position's ids are not on device"
            ));
        }
        self.moe_router_seen = now;
        let Some(device) = self.moe_routed_device.as_ref() else {
            return Err(format!(
                "layer device stage moe_routed (layer {layer}, position {position}): the routed data \
                 path was not opened at setup (attach_moe_routed was not called, or the run resolved \
                 no resident routed shards); refusing rather than launch the staged kernels without \
                 the expert pointer table"
            ));
        };
        let workspace = self.buffers.workspace;
        // Two `DeviceSpan` types meet here: the arm's carved span (`device: u64`, `len: usize`) and
        // the kernels crate's operand span (`device: Deviceptr`, `bytes`). This closure is the only
        // conversion between them, and it keeps the carved length so the module can refuse a short
        // span by its workspace name.
        let to_operand = |carved: crate::layer_device::DeviceSpan| {
            DeviceSpan::new(Deviceptr(carved.device), carved.len)
        };
        let operands = MoeRoutedOperands {
            act_codes: to_operand(workspace.act_codes),
            act_scales: to_operand(workspace.act_scales),
            ids: to_operand(workspace.moe_ids),
            weights: to_operand(workspace.moe_weights),
            delta_f32: to_operand(workspace.moe_delta),
            delta_bf16: to_operand(workspace.moe_delta_bf16),
        };
        // One token per stage call: the carved spans are one row wide, and the plan's capacity is
        // that row's top_k. `run` checks every span and refuses a short one by name.
        device.run(self.cuda, layer, position, 1, &operands)?;
        probe::count("moe_routed");
        Ok(())
    }
```

`Deviceptr` conversion: the arm's `DeviceSpan.device` is `u64`, the kernels crate's is a `Deviceptr`
tuple struct — the `span` closure above is the only conversion. If the Lead prefers, the module's
`DeviceSpan` can take `u64`/`len` instead; the current split keeps `sparkflash-kernels` free of CLI
types.

The refusal is now a *setup* refusal (or the freshness gate), which is what the divergence rule wants:
both are identical on all four ranks, and neither can fire on one rank alone. The other refusals are
the span checks inside `run` (also rank-identical, from the plan and the carved slab).

### 5.5 Teardown (`impl Drop for DeviceLayer`, `:11806`)

The layer's `Drop` frees its own modules and slab; the run-scope state needs its own `release`, so in
the loop's teardown (where `drop(layer_device)` / the shared-expert state is released) add:

```rust
    if let Some(device) = layer_device.take() {
        let cuda = device.cuda_handle();
        if let Some(moe) = device.take_moe_routed_device() {
            moe.release(cuda);
        }
    }
```

That needs two small accessors on `DeviceLayer` (`cuda_handle` returning the `&Cuda` it borrows, and
`take_moe_routed_device` returning the `Option` by value). If the Lead prefers not to add them,
`MoeRoutedDevice` can instead hold its buffers in a `Vec<Deviceptr>` and be freed by the same
`free_buffers` call at run teardown — the module's `release` is written for the explicit path because
the layer arm owns the lifetime. (The module cache owns the five modules: `free` on a cached module is
a documented no-op, so `release` is safe either way.)

### 5.6 The record (`"layer_device_memory"`, `:14372`)

Add next to `"shared_expert_bytes"` (`:14421`):

```rust
                "moe_routed_device_bytes": layer_device
                    .as_ref()
                    .map_or(0, |device| device.moe_routed_device_bytes()),
```

with

```rust
    /// The routed data path's device bytes, as `attach_moe_routed` allocated them.
    fn moe_routed_device_bytes(&self) -> usize {
        self.moe_routed_device
            .as_ref()
            .map_or(0, |device| device.device_bytes())
    }
```

and extend the record's `note` to say the number is the pointer table (737,280 B) plus the staging
buffers (9,542,544 B at the one-token capacity), allocated once at setup and not per layer.

### 5.7 `crates/sparkflash-cli/src/layer_device.rs` (Lead-owned; read-only for me)

1. `STAGE_CONTRACTS`, `moe_routed` (`:847-851`): the kernel and signature are the host arm's single
   stored launch. Replace with the arm's real chain:

   ```rust
    StageContract {
        stage: "moe_routed",
        kernel: "moe_expert_stage_sm121 (10-word ABI) x3, routed_gate_up_sm121 (10-word ABI) x2, swiglu_mxfp8_quant_weighted_sm121 (11-word ABI), b12x_mxfp4_moe_stored_sm121 (10-word ABI), moe_routed_combine_sm121 (6-word ABI)",
        signature: "staging: (const uint64_t* table, const uint32_t* ids, uint8_t* dst_weight, uint8_t* dst_scales, float* slot_acc, uint32_t rows, uint32_t experts, uint32_t weight_bytes, uint32_t scale_bytes, uint32_t acc_row_floats); gate/up: (const uint8_t* act_fp8, const uint8_t* act_scales, const uint8_t* w_fp4, const uint8_t* w_scales, const uint32_t* slot_ids, __nv_bfloat16* out, int tokens, ...trace tail); weighted: (const __nv_bfloat16* gate, const __nv_bfloat16* up, const float* weights, uint8_t* out_fp8, uint8_t* out_scales, uint32_t rows, uint32_t cols, float limit, ...trace tail); stored: (const uint8_t* codes, const uint8_t* scales, const uint8_t* w2_fp4, const uint8_t* w2_scales, const uint32_t* slot_ids, float* slot_acc, int tokens, ...trace tail); combine: (const float* slot_acc, const uint32_t* ids, float* delta_f32, uint16_t* delta_bf16, uint32_t tokens, uint32_t hidden)",
    },
   ```

   The pointer table means no operand is a `[slot]` weight buffer the arm builds: the seven launches
   read the arena and this stage's own buffers only, which is the S1 shape the contract table documents.
2. `STAGE_BLOCKERS`, the `moe_routed` entry (`:1449-1451`) — **delete it**:
   `"the five device MoE launches exist; the host combine of their outputs is the step"` is no longer
   true, and a blocker that outlives its cause is worse than no blocker.
3. The buffer table's `moe_delta` row (`:25`) says "`moe_routed`'s bf16 cast before the second
   collective": still true, now cast on device. Optionally note "(device combine + narrow)".

## 6. Kernel registration (the Lead's integration step)

Adding one `kernels/*.cu` touches nine files plus two Rust-side loaders. The exact measured line is in
§1.2. The order below is the one that leaves the tree green after each step.

**The Lead owns this pass**, and is doing it once for all three new cubins of this milestone
(`engram_write`, `engram_stream_convert`, `moe_routed_device`) plus the by-name loaders, because
`cuda_driver.rs` is being edited by another package too. The list below is the precise inventory for
that single pass, not a request for three separate ones.

### 6.1 The nine files

1. **`crates/sparkflash-kernels/build.rs:172`** — `PROBED_KERNELS: [(&str, &str); 27]` -> `28`, with the
   entry `("moe_routed_device_sm121.cu", "two symbols, no trace words; the routed staging gather and the \
   routed combine epilogue")` (match the existing entry shape), and the doc prose at the top of the file
   that counts the kernels.
2. **`crates/sparkflash-kernels/scripts/regen-kernel-baseline.sh`** — two prose counts (`:2`, `:46`) and
   the kernel loop list (`:69`): append `moe_routed_device_sm121`.
3. **`crates/sparkflash-kernels/baseline/kernel_baseline.txt`** — regenerate rather than hand-edit
   (§7.2 reproduces the committed file byte for byte on this host, so the only diff should be the header
   count and the new line).
4. **`crates/sparkflash-kernels/scripts/check-kernel-baseline.sh`** — the `kernels="..."` list (`:19`)
   and the header's "twenty-five baseline kernels" prose (`:2`).
5. **`crates/sparkflash-kernels/src/baseline.rs:153-157`** — `assert_eq!(pins.len(), 27)` -> `28`, prose
   "twenty-seven" -> "twenty-eight".
6. **`crates/sparkflash-kernels/tests/kernel_baseline.rs`** — `:85` `assert_eq!(..., 27)` -> `28`; `:86`
   prose; `:27`, `:39` "twenty-five" prose; and the by-name skip list at `:362-372` must gain
   `"moe_routed_device_sm121"` (it has no `KernelId`; `checked` stays 19), with the comment at `:381`
   "twenty-five entries" -> "twenty-eight".
7. **The three ABI modules' manifest-header tests** — `routed_gate_up_abi.rs:233`,
   `mxfp4_moe_stored_abi.rs:278`, `router_abi.rs:291` assert the manifest says "twenty-seven baseline
   kernels": all three must say "twenty-eight". (The Lead's `grep -rl "twenty-seven baseline kernels"
   crates/` is the complete list.)

### 6.2 The two by-name loaders in `cuda_driver.rs`

Next to `load_router_topk` (`:2861`) / `load_kv_norm_rope` (`:2991`), same house style:

```rust
    /// moe_routed_device_sm121: the routed staging gather, loaded by name from its own cubin.
    ///
    /// The cubin carries two symbols and no `KernelId` names either of them (a `KernelId` names one
    /// symbol), so both are loaded by name; the launch module (`moe_routed_device`) owns the pin.
    pub fn load_moe_routed_stage(&self, cubin: &[u8]) -> Result<(Module, Kernel), String> {
        let (module, function) = self.load_function(cubin, "moe_expert_stage_sm121")?;
        Ok((
            Module(module),
            Kernel {
                function,
                name: "moe_expert_stage_sm121",
            },
        ))
    }

    /// moe_routed_device_sm121: the routed combine epilogue, loaded by name from the same cubin.
    pub fn load_moe_routed_combine(&self, cubin: &[u8]) -> Result<(Module, Kernel), String> {
        let (module, function) = self.load_function(cubin, "moe_routed_combine_sm121")?;
        Ok((
            Module(module),
            Kernel {
                function,
                name: "moe_routed_combine_sm121",
            },
        ))
    }
```

### 6.3 The staged directory

`--kernel-cubin-dir` must contain `moe_routed_device_sm121.cubin` next to the 27 existing cubins;
`load_pinned_addition` refuses a missing or unpinned file at open, which is why §6.1 comes before §5.2
is exercised.

## 7. Verification run on this host

### 7.1 What ran

| command | result |
| --- | --- |
| `/home/user/cuda-tk/bin/nvcc -std=c++17 -arch=sm_121a -cubin -I kernels kernels/moe_routed_device_sm121.cu` | exit 0, 28,144 B, sha256 `19bcadd0…d136` |
| `/home/user/cuda-tk/bin/nvcc -std=c++17 -arch=sm_121a --ptx -I kernels kernels/moe_routed_device_sm121.cu` | exit 0, both `.visible .entry`s present, 0 `membar` lines |
| `/usr/lib/llvm/22/bin/clang++ -std=c++17 --cuda-device-only -nocudainc -nocudalib -fsyntax-only -DSPARKFLASH_CUDA_SYNTAX_STUB -I kernels -I crates/sparkflash-kernels/cuda_stub --target=x86_64-unknown-linux-gnu kernels/moe_routed_device_sm121.cu` | exit 0 (the stub probe the crate's `build.rs` uses, run with full nvcc semantics too) |
| `cargo test -p sparkflash-kernels --lib` | **576 passed, 0 failed, 2 ignored** at the time of this run (the two `#[ignore]` device tests in this module). The crate's total moves as teammates land modules on this shared tree; the 24 CPU tests this task adds are the ones §7.1's next row names |
| `cargo test -p sparkflash-kernels --lib moe_routed_device` | **24 passed, 0 failed, 2 ignored** |
| `cargo clippy -p sparkflash-kernels --lib -- -D warnings` | clean |
| `cargo fmt --check` | clean |
| `SPARKFLASH_NVCC=/home/user/cuda-tk/bin/nvcc /bin/sh crates/sparkflash-kernels/scripts/regen-kernel-baseline.sh /tmp/measured_baseline.txt` | exit 0, output **identical** to the committed `baseline/kernel_baseline.txt` (1m16s) |

### 7.2 The registration can be regenerated here

Because this host's `/home/user/cuda-tk/bin/nvcc` is the release/build the manifest was generated
with, the Lead can regenerate the whole 28-entry baseline on this host, no GPU needed:

```sh
SPARKFLASH_NVCC=/home/user/cuda-tk/bin/nvcc /bin/sh crates/sparkflash-kernels/scripts/regen-kernel-baseline.sh \
    crates/sparkflash-kernels/baseline/kernel_baseline.txt
SPARKFLASH_NVCC=/home/user/cuda-tk/bin/nvcc /bin/sh crates/sparkflash-kernels/scripts/check-kernel-baseline.sh
```

The first run today reproduces the 27 committed lines byte for byte, so after the two script lists and
the header prose are updated, the only manifest diff is the header count and the new
`kernel moe_routed_device_sm121 …` line quoted in §1.2.

### 7.3 NOT RUN (no CUDA device on this host)

* any launch of any kernel in this note (the two new ones and the five landed ones);
* the pointer-table upload, the staging gather, the combine, and every numeric comparison against the
  host arm;
* the D54 numerical gate (any tolerance for the code path, and the f32-reference shared arm's declared
  item);
* step latency, launch-count and PCIe-byte measurements, including the "no per-layer PCIe copies" claim
  (it is an architecture claim here, not a measurement);
* CUDA-graph capture of the eight-launch group (`tp4_graph::segment`); the launches are plain stream
  launches and are capturable, but that was not tested;
* the two `#[ignore]` tests in the module (`cargo test -- --ignored` on a GPU host).

## 8. GPU verification procedure (when a device is available)

In order, each step a refusal rather than a tolerance:

1. `cargo test -p sparkflash-kernels --lib -- --ignored moe_routed_device` — the table upload round-trip
   and the released-size allocation.
2. Table layout on device: for one layer, copy the table back and compare against
   `routed[layer][expert][matrix]` from `attach_routed` (equality of `u64` address pairs); a permuted
   table is a silent wrong-expert path, so this is a required check, not a smoke test.
3. Staging gather: write six known ids into `moe_ids`, run one staging launch into a readback buffer, and
   compare byte for byte against the host's `copy_dtod` of the same six slices — the host arm contains
   the reference sequence.
4. The combine, end to end: run the eight launches for layer 0 on a fixed token, read back
   `moe_delta`/`moe_delta_bf16`, and compare against `run_layer_moe_routed`'s f32 delta and payload for
   the same token. Expected: **bit-identical** (`to_bits` equality of every f32, exact `u16` payload),
   not a tolerance pair. Any mismatch is a defect in this module's ordering or the transposed rounding
   formulas, not a numerical tolerance item.
5. Then the D54 gate and the step-timing run — the milestone owners', not this note's.

## 9. Risks, decisions and open questions for the Lead

### 9.1 "up to 256 x (w_ptr, s_ptr)" — resolved

The brief's phrase cannot be the expert count of the released layer (384 resident experts), so P3 read
256 as the **routed-row capacity** and flagged the reading. **The Lead confirmed it** (2026-10-22): the
released model runs 384 routed experts at top-6 (`docs/archive/build/gpu-device-roadmap.md:84`,
`docs/archive/build/kernel-occupancy-inventory.md:26`), so 256 is the staging-row capacity, kept as
`MAX_ROUTED_ROWS = 256`, and the table stays `[layer][matrix][expert]` at 40 x 3 x 384 x 2 u64 =
737,280 B. The resolution is restated in §10. The staging design (device ids -> device pointer table ->
landing kernels' slot ABI) is what the brief was describing either way.

### 9.2 Per-routed-row staging instead of per-unique-expert

The host arm dedups ids and stages one slice per unique expert; the device path stages one slice per
routed row. Advantages: no device-side dedup/sort kernel (the combine sorts anyway, and a dedup pass
would be a read-modify-write across CTAs), and the landing kernels' slot ABI is per row. Costs: with
duplicate ids some slices are copied twice (a `copy_dtod` the host arm skips — a DtoD copy inside the
staging kernel, not PCIe traffic), and the row capacity is 6 rather than the unique count. Negligible
at top_k = 6, and it keeps the two sides' arithmetic identical (§4).

### 9.3 Memory: the 9.8 MiB is folded into the budget projection as a **record**

`attach_moe_routed` allocates 10,279,824 B; the Lead's ruling is that the shared prewarm's
`BudgetAdmission` projection carries that number, **as a record only** — the fold must not change the
bytes the shared prewarm allocates nor its measured total (`shared_expert_bytes` /
`shared_prewarm_bytes` are `SharedExpertDevice::device_bytes`, which stays exactly its own
allocations). The exact site and expression are in §5.3.1. Nothing in `build_shared_expert_device`'s
allocation path changes, and the shared prewarm's recorded number is byte-for-byte what it is today.

### 9.4 Which rank's shard the table points at

The table is built per rank from that rank's `routed` shard (the arena pointers `attach_routed` resolved
with `request.rank`), so each rank stages its own column/row slice of every expert, as the host arm does.
The combine then sums the six rows of *one rank's partial*; the cross-rank sum is `all_reduce_moe`'s, not
this stage's. That is the host arm's structure exactly (its `slot_acc` is one rank's partial, reduced
afterwards), so the native path's "all four ranks sum their partials" is unchanged.

### 9.5 Trace words — approved

The landed launches take `trace_ring`/`layer_id`/`step_seq`; the shipped cubins are trace-disabled
builds (the same finding the router stage records), so the module passes `trace_ring = 0` and the real
`layer`/`position` in the other two words — better evidence if a traced build is ever pinned, and a
no-op today. The host arm passes `layer_id = 0, step_seq = rank`, which is meaningless outside a probe
build. The Lead approved this deviation (2026-10-22).

### 9.6 The 10,240 B payload upload disappears but the collective does not

`all_reduce_moe` still reads `workspace.moe_delta_bf16` and writes `workspace.moe_reduced_bf16`; this
module only replaces the *producer* of that row with a device kernel. The collective count assertion
(`COLLECTIVES_PER_LAYER`) is untouched.

## 10. Out of scope: the 640/576 padding is S4

The routed gate/up/SwiGLU operands are `576` real columns padded to `640` (the landed kernels' pinned
`SPARK_M4_MOE_WIDTH_PADDED`): the two gate/up launches write zeros into columns `576..640`, the weighted
SwiGLU quantises all 640, and the stored launch's K width is the same padded width. The padding is
**reproduced exactly**, because the incumbent ABI is the pinned one; removing or reshaping it (a real
576-wide chain, or a 256-byte-aligned padded width) is the S4 padding work, and it moves the landed
kernels' geometry, not this module. `MoeRoutedGeometry` refuses anything but the released shape
precisely so that this task cannot silently depend on a padded-width change that has not landed.

## 10.1 Resolution of the "256" ambiguity (Lead, 2026-10-22)

The P3 brief asked for an expert pointer table of "256 x (w_ptr, s_ptr)". The released model has
**384 routed experts at top-6** (`docs/archive/build/gpu-device-roadmap.md:84` and
`docs/archive/build/kernel-occupancy-inventory.md:26` name the released choices), so 256 cannot be an expert
count for this layer. The Lead confirmed P3's reading: 256 is the **routed-row staging capacity**,
landed as `MAX_ROUTED_ROWS = 256`, and the pointer table is `[layer][matrix][expert]` = 40 x 3 x 384 x 2
`u64` = 737,280 B (§3.3). Nothing in the module changes from this note; it records that the ambiguity is
settled so a later reader does not re-raise it. The one-token decode stages 6 rows and the plan admits
up to 42 tokens (252 rows); a call that asks for more is refused by `MoeRoutedPlan::routed_rows`.
