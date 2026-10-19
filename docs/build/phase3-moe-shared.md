# Phase 3 m17b - the shared expert at real layer-0 geometry

## Status

**PASS at both token counts (window b28).** The harness ran the released shared-expert path and all
eleven declared checks passed at tokens 8 and 130, with the binary's sha256 verified by the window
script first: 11 checks per run, three device stages plus the declared host shared-add, two pinned
cubins. This is a **stage result only**: the shared-add epilogue is host arithmetic by declaration,
the activation is a declared stand-in, and the routed operand is a declared stand-in, so nothing here
is a chain, token or per-layer claim.

Source: `crates/sparkflash-kernels/examples/m17b_moe_shared.rs`, aarch64 binary
`m17b_moe_shared_7f9719c9_aarch64` (4,466,096 B, sha256
`7f9719c91febe0fc4763c9affc4ed9683e59995382d1bae806f3fd4cbeb74ace`),
built by `scripts/build-aarch64.sh --example=m17b_moe_shared` from branch
`task/m17b-shared`, rebased onto `origin/main` 51e7954 (batch 15, the m17a merge), with build cwd
`/home/user/wt/p3-m17` (the build script is cwd-sensitive: the same source from another directory
gives different bytes and size, so the sha above is reproducible only from that cwd). The window b28
binary was built from the same m17b source before that rebase, and the rebase changed only the base.
The branch also carries the batch-16 m17a harness hardening and the m16 four-shard argmax fixture
extension:
four shard vectors with the maximum only in rank 1 and the second entry in rank 3, so a scan that
takes only two shards returns 2.0 instead of 2.5 and fails `top_indices`; it is a test-only change
and rides with this batch.

## Subject and launch

One harness runs the released shared-expert path on one node, in the released order, with the
released checkpoint bytes:

| # | stage | kernel | cubin pin | launches |
| :-- | :--- | :--- | :--- | :--- |
| 1 | shared w1 and w3 | `b12x_mxfp8_dense_sm121` | 11,824 B / `03e4c422...` | 2 |
| 2 | SwiGLU + MXFP8 quantise | `swiglu_mxfp8_quant_sm121` | 21,776 B / `4dbef63b...` | 1 |
| 3 | shared w2 | `b12x_mxfp8_dense_sm121` | 11,824 B / `03e4c422...` | 1 |
| 4 | shared-add epilogue | none - **HOST** | - | host |

Both cubins are refused by size and sha256 before any device allocation, and the constants are
cross-checked against the crate's embedded baseline manifest (`baseline::pin`, the single source of
pins); a mismatch is an error before the path is measured, never a tolerance.

Geometry, from the released `config.json`: hidden 5120, `moe_intermediate_size` 2304,
`n_shared_experts` 1. The shared expert stays **whole and replicated** at 2304 (D30); there is no
rank slicing and no routed expert is touched. The released layer-0 FFN is routed experts plus this
one shared expert, so the shared path is a real part of the layer rather than an extra.

## Released operands (bytes only, no synthetic weights)

| operand | released tensor | dtype / shape | read as |
| :--- | :--- | :--- | :--- |
| shared w1 | `layers.<l>.ffn.shared_experts.w1.weight` | F8_E4M3 [2304, 5120] | bytes, repacked to B12x order on the host |
| shared w1 scales | `layers.<l>.ffn.shared_experts.w1.scale` | F8_E8M0 [72, 160] | bytes, verbatim (the stored grid is already B12x's) |
| shared w3 | `layers.<l>.ffn.shared_experts.w3.weight` | F8_E4M3 [2304, 5120] | bytes, repacked |
| shared w3 scales | `layers.<l>.ffn.shared_experts.w3.scale` | F8_E8M0 [72, 160] | bytes, verbatim |
| shared w2 | `layers.<l>.ffn.shared_experts.w2.weight` | F8_E4M3 [5120, 2304] | bytes, repacked |
| shared w2 scales | `layers.<l>.ffn.shared_experts.w2.scale` | F8_E8M0 [160, 72] | bytes, verbatim |
| activation rows | `embed.weight` rows 0..tokens | BF16 | **declared stand-in** for the post-attention hidden |
| routed operand | the same `embed.weight` rows | BF16 | **declared stand-in**, explicitly NOT m17a's routed output |

The activation stand-in is the same choice m17a, `m9_router.rs` and `m10_gate_up.rs` make: this tree
does not carry layer 0's post-attention hidden into an isolated harness. The activation operand is
the D8 quantiser applied per row (`act_quant_fp8_row`), exactly as the mHC pass leaves the FFN block
input. The shared weights are repacked to B12x order by a value-preserving index permutation on the
host (no arithmetic), the same declaration m12 and the M7 stage-7 chain make.

## Comparison classes (declared before any number)

1. **Shared w1/w3 dense.** Every output bf16 bit must equal `mxfp8_ref::dense_gemm` on the SAME
   activation codes/scales, the SAME repacked weights and the SAME weight scales - the bit-identical
   class of the M7 stage-7 chain and m12 (the scalar path has one f32 accumulation in K-ascending
   32-value blocks and one bf16 rounding, exactly the reference's order, so there is no
   reduction-order bar). Both projections run at their released [tokens][2304] over inner 5120; two
   launches byte-identical per projection. A difference is a finding.
2. **SwiGLU + quantise.** E4M3 codes and ue8m0 scales BIT-EXACT against `swiglu_ref::quantise_row`
   over the same bf16 values at the released 2304-wide intermediate - the stage-8 B1 class. Two
   launches byte-identical. The launch is unweighted: the shared expert is not multiplied by a
   routing weight.
3. **Shared w2 dense.** Every output bf16 bit must equal `mxfp8_ref::dense_gemm` on stage 2's device
   codes/scales, the same repacked w2 and the same weight scales - the same bit-identical class as
   stage 1, at the second released shape [tokens][5120] over inner 2304. Two launches byte-identical.
4. **Shared-add epilogue - HOST, by declaration.** No pinned kernel owns the elementwise add, so the
   epilogue is host arithmetic and is NOT a device measurement. Its class: the reference's own
   combination (`combine_bf16_routed_with_shared`: routed bf16 widened + shared bf16 widened, f32
   sum) applied to the same bytes, with the check that an independent f32 loop in the harness
   reproduces it bit-for-bit; the operand provenance is printed. This keeps the seam named rather
   than hidden.

## The shared-add seam (named, not hidden)

The released block's last step is `ffn_out = routed + shared`: two bf16 vectors added in f32 and
handed on to the FFN mHC pass (the M7 stage-7 chain records this as seam S9). The routed operand
belongs to m17a's measured routed branch; this harness does not re-run that branch and does not claim
its output. It therefore applies the reference's combination to the device's real stage-3 bf16
values and a **declared** routed stand-in (released `embed.weight` rows), and prints exactly which
bytes cross the seam: routed bf16 `[tokens][5120]` (stand-in) plus shared bf16 `[tokens][5120]`
(device stage 3) -> f32 `[tokens][5120]`. The result is a seam check, not an FFN row: the
NOT-claimed list excludes any end-to-end or per-layer claim.

## Window command

```sh
cd ~/phase2-evidence/window-2026-10-18/bin
./m17b_moe_shared_7f9719c9_aarch64 --checkpoint ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170 \
    --cubin-dir ~/phase2-evidence/window-2026-10-18/cubins --tokens 8 --layer 0 --floor-gib 32
./m17b_moe_shared_7f9719c9_aarch64 --checkpoint ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170 \
    --cubin-dir ~/phase2-evidence/window-2026-10-18/cubins --tokens 130 --layer 0 --floor-gib 32
```

The cubin dir must hold `b12x_mxfp8_dense_sm121.cubin` (11,824 B) and
`swiglu_mxfp8_quant_sm121.cubin` (21,776 B); both are already staged there from earlier windows.

Measured wall time in b28: tokens 8 ran 05:02:14 -> 05:02:15Z (1 s) and tokens 130 ran
05:02:15 -> 05:02:20Z (5 s), both below the estimate; the harness makes no timing claim, so the page
records the wall times without attributing them to a component. The 32-GiB MemAvailable floor is checked at start, before the device context and
before/after the stage-2 launch; every reading in the window was at least 116.2 GiB, and the node
needs vLLM stopped as usual for a window.

## Host dry run and tests

`cargo test -p sparkflash-kernels --example m17b_moe_shared`: 18 tests pass (the shared b2_harness
suite plus four m17b tests: argument parsing of the documented command, the B12x repack being a
value-preserving permutation, the shared-add loop reproducing the reference combination, and the
two pins being the baseline-manifest lines). `cargo clippy -p sparkflash-kernels --example
m17b_moe_shared -- -D warnings` is clean. The workspace gates are run before review. No device code
path is executed by these tests.

## Window record

### Window b28 (2026-10-19): PASS at both token counts

Window `window-b28-20261019T050029Z.log`. The window script verified the binary sha256 before the
runs (the log records `7f9719c91febe0fc4763c9affc4ed9683e59995382d1bae806f3fd4cbeb74ace`), and both
cubins' sizes and shas are printed by the harness before allocation and again cross-checked against
the embedded baseline manifest.

```text
=== window b28 (c3b retry with the row-order fix; then m17b shared expert) start 2026-10-19T05:00:29Z host=node0 ===
=== RUN m17b-tokens8 start 2026-10-19T05:02:14Z
=== RUN m17b-tokens8 exit=0 end 2026-10-19T05:02:15Z
=== RUN m17b-tokens130 start 2026-10-19T05:02:15Z
=== RUN m17b-tokens130 exit=0 end 2026-10-19T05:02:20Z
```

The exact per-run records from `runs28/m17b-tokens8.log` and `runs28/m17b-tokens130.log`:

#### tokens 8 (35,424,000 released shared-expert bytes read)

```text
MemAvailable at m17b start: 122094512 kB (116.4 GiB), floor 32 GiB - ok
  check geometry matches the released config.json: ok (hidden 5120, shared intermediate 2304, exactly 1 shared expert, fp4 experts)
MemAvailable at m17b before context: 122038836 kB (116.4 GiB), floor 32 GiB - ok
  check stage 1 w1: bit-identical to mxfp8_ref::dense_gemm: ok (0 mismatched bf16 values of 18432)
  check stage 1 w1: two launches byte-identical: ok (36864 bf16 bytes compared)
  check stage 1 w3: bit-identical to mxfp8_ref::dense_gemm: ok (0 mismatched bf16 values of 18432)
  check stage 1 w3: two launches byte-identical: ok (36864 bf16 bytes compared)
MemAvailable at m17b stage 2 pre-launch: 121847312 kB (116.2 GiB), floor 32 GiB - ok
MemAvailable at m17b stage 2 post-launch: 121847316 kB (116.2 GiB), floor 32 GiB - ok
  check stage 2: E4M3 codes bit-exact: ok (0 mismatched of 18432)
  check stage 2: ue8m0 scales bit-exact: ok (0 mismatched of 576)
  check stage 2: two launches byte-identical: ok (18432 code + 576 scale bytes compared)
  check stage 3 w2: bit-identical to mxfp8_ref::dense_gemm: ok (0 mismatched bf16 values of 40960)
  check stage 3 w2: two launches byte-identical: ok (81920 bf16 bytes compared)
  check stage 4 host seam: the harness reproduces the reference's combination bit-for-bit: ok (0 mismatched f32 of 40960 (host seam, not a device check))
```

#### tokens 130

```text
MemAvailable at m17b start: 122072436 kB (116.4 GiB), floor 32 GiB - ok
  check geometry matches the released config.json: ok (hidden 5120, shared intermediate 2304, exactly 1 shared expert, fp4 experts)
MemAvailable at m17b before context: 122023016 kB (116.4 GiB), floor 32 GiB - ok
  check stage 1 w1: bit-identical to mxfp8_ref::dense_gemm: ok (0 mismatched bf16 values of 299520)
  check stage 1 w1: two launches byte-identical: ok (599040 bf16 bytes compared)
  check stage 1 w3: bit-identical to mxfp8_ref::dense_gemm: ok (0 mismatched bf16 values of 299520)
  check stage 1 w3: two launches byte-identical: ok (599040 bf16 bytes compared)
MemAvailable at m17b stage 2 pre-launch: 121836044 kB (116.2 GiB), floor 32 GiB - ok
MemAvailable at m17b stage 2 post-launch: 121836056 kB (116.2 GiB), floor 32 GiB - ok
  check stage 2: E4M3 codes bit-exact: ok (0 mismatched of 299520)
  check stage 2: ue8m0 scales bit-exact: ok (0 mismatched of 9360)
  check stage 2: two launches byte-identical: ok (299520 code + 9360 scale bytes compared)
  check stage 3 w2: bit-identical to mxfp8_ref::dense_gemm: ok (0 mismatched bf16 values of 665600)
  check stage 3 w2: two launches byte-identical: ok (1331200 bf16 bytes compared)
  check stage 4 host seam: the harness reproduces the reference's combination bit-for-bit: ok (0 mismatched f32 of 665600 (host seam, not a device check))
```

The record's numbers, stated plainly: w1 and w3 are bit-identical to `mxfp8_ref::dense_gemm` on all
18,432 / 299,520 bf16 values per projection, w2 on all 40,960 / 665,600, with two-launch
determinism on every stage; the 2304-wide SwiGLU quantiser's E4M3 codes (18,432 / 299,520) and
ue8m0 scales (576 / 9,360) are bit-exact; and the host shared-add reproduces the reference
combination on all 40,960 / 665,600 f32 values. The crossing shapes print as declared: routed bf16
[tokens][5120] (released embed rows, stand-in) + shared bf16 [tokens][5120] (device stage 3) ->
f32 [tokens][5120] (reference order, host). This is a **stage result only** - host shared-add seam,
declared stand-in activation and routed operand, no chain/token/per-layer claim.

The same window's c3b rerun failed on an unrelated Engram-split twin check at layer 1; that is the
engine-dev arm and does not affect m17b.

## NOT claimed

- Not a device shared-add: the epilogue is host arithmetic and this run does not measure a kernel
  for it.
- Not a chain, token, per-layer or end-to-end FFN claim: one layer's shared expert, one declared
  stand-in activation and a declared stand-in routed operand; the routed block is m17a and is not
  re-measured here.
- Not layer text: the activation and the routed operand are released `embed.weight` rows.
- Not the shared expert at any other layer and not a routed-expert claim.
- Not engine integration and not a host-time-saving claim: no engine seam is changed.
- Not a performance claim: the harness makes no timing claim.
