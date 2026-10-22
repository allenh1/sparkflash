# Attention operand path on the four-rank arm: retraction, host gate, and the wiring contract

Status: 2026-10-20, host-only (no GPU). Branch task/attn-operand-repack, cut from integration tip
bf1b33a. This note answers the retargeted task-10: the released attention projections are already
the device operands the chain binds, so the deliverable is evidence hardening, not a repack kernel.

## 1. Premise retraction

The task-10 brief inherited a claim that wq_a and wkv are stored BF16 on the released checkpoint and
therefore have no B12x-weight device kernel. That claim was an unmeasured inference. It is retracted;
this note replaces it with a header read and a plan gate that can be rerun.

## 2. Measured evidence (released checkpoint)

Checkpoint: DeepSeek-V4.1-Flash-1017, snapshot 1bae0e1088993b9b82de08f59de5be5a34e42170. Header
dump read in place on node0 from the 48 safetensors headers: 96,085 tensors,
sha256 f459bda49bbbba28d9d73570bb7bbf2beba6d71e0a96343644a97839ffa10786 (see section 8).

- All 43 attention layers (40 chain + 3 MTP) carry wq_a and wkv as F8_E4M3 with F8_E8M0 scales:
  43/43 each, zero exceptions. wq_b, wo_a and wo_b are F8_E4M3 as well.
- The plan's B12x predicate (crates/sparkflash-device/src/arena.rs) is true for every one of the 215
  attention projection weights: dtype F8_E4M3, name ends .weight, attention/shared group, sharded
  shape with a 32-divisible pair of axes.
- Window evidence from the 2026-10-20 bind run (window/m1-bind/rank0/tp4-loop-r0.json) matches the
  plan: attention-replicated 258 tensors, 86 tiled, 395,066,112 B; attention-column 172/86 tiled;
  attention-row 86/43 tiled; tiled_tensors 344. The replicated byte total decomposes as
  43 x (wq_a 6,553,600 + wkv 2,621,440 + scales 8,960) + BF16 q/kv norms 154,112 = 395,066,112 B.
- bind_rank_attention already binds a B12x weight and its scale as two distinct pointers; the
  single-node device path reads the checkpoint F8 bytes and applies b12x_weight_index, a pure index
  permutation. No BF16->FP8 weight quantiser exists in the tree and none is used.

## 3. Decision: no boot repack (the chosen "option C")

No repack kernel, no new cubin, no pin change, no baseline entry. The twenty-entry baseline stays
twenty. The arena's attention groups already hold the checkpoint's B12x-tiled F8_E4M3 codes and the
F8_E8M0 scale grids; the forward loop binds those pointers directly.

Exactness: the bind is value-preserving by construction. The checkpoint bytes are copied into the
arena in the checkpoint's own dtype; the only transform is the B12x index permutation, an assertion
about which byte lands where, with no arithmetic, no dtype conversion and no scale transform. The
operand pair the D37 bf16a kernels (b12x_bf16a_mxfp8w_dense_sm121, b12x_bf16a_mxfp8w_wo_a_sm121)
read from the arena is bit-for-bit the checkpoint's F8_E4M3 codes and E8M0 scales.

Options A and B from the brief (one-time in-arena B12x repack at boot, or a BF16-weight GEMM family)
are not "infeasible": they are unnecessary, because the premise that any attention projection is
BF16-stored is false. A future checkpoint that stores BF16 attention weights must fail this gate by
name (section 4), and only then would a real quantiser or a BF16-weight kernel family be the answer.

## 4. Host gate: location, argv, checks, exact lines

The gate is a host-only stage, no driver, no kernel launch, no output claim:

    crates/sparkflash-cli/src/tp4_operands.rs                  (stage + unit tests)
    sparkflash device-chain --run-stage attn-operands \
      (--tensor-index HEADER_DUMP.json | --checkpoint DIR) \
      --tp-rank R --tp-world-size 4 --evidence-dir DIR [--expect-plan-hash 0x...]

It writes tp4-attn-operands-rR.json (schema sparkflash-tp4-attn-operands/1) and exits 0 on PASS, 2
on FAIL. It builds the rank's ResidentPlan from headers, derives each projection's operand layout
from the plan record (dtype + group + shard shape) and checks: the 200 chain projections against the
rank geometry; all 215 attention weights (chain + MTP) and their 215 scale grids; the five layer-0
operands; group tiled parity 86/86/43; group members 258/172/86; and, at world 4, the exact group
and operand byte totals the bind window recorded. A stored (non-B12x) attention projection is named
in the blocked list and fails the gate instead of becoming a per-layer host repack at run time.

Real run, released headers, all four ranks exit 0, rank 0 printed:

    tp4-attn-operands rank 0 world 4: plan 95822 shards (95818 resident / 4 disk), 43 attention layers, plan hash 0x9a59035fc40776c8
    tp4-attn-operands: attention projections 215/215 B12x-launchable (86 replicated + 86 column + 43 row), 0 stored, 0 blocked
    tp4-attn-operands: chain geometry 200/200 B12x; layer-0 five operands B12x
    tp4-attn-operands: wkv 43 x [512x5120] F8_E4M3 -> B12x 2621440 B codes + 2560 B F8_E8M0 scales (b12x_bf16a_mxfp8w_dense_sm121, activation BF16)
    tp4-attn-operands: wo_a 43 x [2048x4096] F8_E4M3 -> B12x 8388608 B codes + 8192 B F8_E8M0 scales (b12x_bf16a_mxfp8w_wo_a_sm121, activation BF16)
    tp4-attn-operands: wo_b 43 x [5120x2048] F8_E4M3 -> B12x 10485760 B codes + 10240 B F8_E8M0 scales (b12x_bf16a_mxfp8w_wo_a_sm121, activation BF16)
    tp4-attn-operands: wq_a 43 x [1280x5120] F8_E4M3 -> B12x 6553600 B codes + 6400 B F8_E8M0 scales (b12x_bf16a_mxfp8w_dense_sm121, activation BF16)
    tp4-attn-operands: wq_b 43 x [8192x1280] F8_E4M3 -> B12x 10485760 B codes + 10240 B F8_E8M0 scales (b12x_bf16a_mxfp8w_dense_sm121, activation BF16)
    tp4-attn-operands: operand bytes replicated 394912000, column 812390400, row 451328000, total 1658630400 B; repack bytes 0
    tp4-attn-operands gate: PASS (215/215 attention projections on 43 layers are B12x MXFP8; no BF16-stored operand; no boot repack; per-token weight H2D stays 0)

The plan hash 0x9a59035fc40776c8 is the fingerprint the bind window recorded, so the gate's plan is
the same hundred-thousand-shard plan the device arm ran; --expect-plan-hash pins it explicitly.
18/18 evidence checks pass on the released headers.

Negative test (deliberately inconsistent fixture, refused by name, nothing launched):
tp4_operands::tests::a_bf16_stored_projection_fails_the_gate_by_name rewrites one wq_a header to
BF16; the gate fails and names layers.0.attn.wq_a.weight. On the device side the same rule lives in
sparkflash-device::resident::check_operand_bind, whose two tests
(the_bind_census_accepts_the_planned_pair_and_names_an_inconsistent_one and
a_fixture_bind_that_disagrees_with_the_plan_is_named_not_launched) refuse a shared pointer for a B12x
operand, a null pointer, and a distinct scale pointer for a stored tensor, each with the tensor name.

## 5. Census and chain wiring contract for the four-rank forward

Device-lead's census should stop inferring layout from the bind pointer pair alone and ask the plan.
The plan API (crates/sparkflash-device/src/resident.rs) is:

    pub enum OperandLayout { B12x { cols, inner, weight_bytes, scale_bytes }, Stored { dtype, bytes } }
    ResidentShard::operand_layout() -> OperandLayout        // plan record only
    ResidentPlan::operand_layout(name) -> Option<OperandLayout>
    check_operand_bind(&ResidentShard, weight_ptr: u64, scale_ptr: u64)
        -> Result<OperandLayout, String>                    // names the tensor on refusal

The forward loop's census should then print, per rank, the plan-vs-bind comparison rather than a
pointer guess. Required lines (values for world 4, released plan; other worlds keep counts and
recompute bytes from the rank geometry):

    tp4-chain rank 0: attention census 215/215 b12x by plan dtype (F8_E4M3 weights, F8_E8M0 scales), 0 stored, 0 plan-vs-bind disagreements
    tp4-chain rank 0: attention-replicated 258 tensors (86 tiled, 395066112 B), column 172 (86 tiled, 812390400 B), row 86 (43 tiled, 451328000 B)
    tp4-chain rank 0: per-token weight H2D 0 B; repacked attention weight bytes 0

Refusal contract: when check_operand_bind returns Err, the loop prints the returned string (it names
the tensor, the planned layout and the observed pair) and the layer does not launch. There is no
fallback path that repacks or converts a weight inside the loop.

Per-projection launch contract (world 4, rank geometry; all four ranks share it because every axis
divides by 4):

    projection | kernel family                          | activation | rank-0 operand geometry | scale rule
    wq_a       | b12x_bf16a_mxfp8w_dense_sm121          | BF16       | 1280 x 5120             | F8_E8M0 [40,160] stored as-is
    wkv        | b12x_bf16a_mxfp8w_dense_sm121          | BF16       |  512 x 5120             | F8_E8M0 [16,160] stored as-is
    wq_b       | b12x_bf16a_mxfp8w_dense_sm121          | BF16       | 8192 x 1280 (Column)    | F8_E8M0 [256,40] stored as-is
    wo_a       | b12x_bf16a_mxfp8w_wo_a_sm121           | BF16       | 2048 x 4096 (Column)    | F8_E8M0 [64,128] stored as-is
    wo_b       | b12x_bf16a_mxfp8w_wo_a_sm121           | BF16       | 5120 x 2048 (Row)       | F8_E8M0 [160,64] stored as-is

Where the repack runs: nowhere. There is no boot pass, no staging buffer, no second copy. The two
arena pointers per projection (tiled weight, scale grid) are the binding contract; both come from
the one streaming arena upload the chain already performs.

## 6. Memory accounting

No new allocation. The operand bytes below are a subset of the arena the plan already uploads; the
plan's resident total is 77,640,198,984 B (72.31 GiB), exactly the design budget's resident figure.

    replicated operand bytes  394,912,000 B  (wq_a 281,804,800 + wkv 112,721,920 + scales 385,280)
    column operand bytes      812,390,400 B
    row operand bytes         451,328,000 B
    attention operand total 1,658,630,400 B, repacked bytes 0, per-token weight H2D 0 B

## 7. What this does not prove

The gate proves the plan offers the right operands; it does not prove a bind or a launch. The bind
side is already evidenced by window/m1-bind (86/86/43 tiled, 344 total) and the gate reproduces the
same totals from the plan, which is why no GPU window is needed for task-10. The remaining M1
critical path is the 40-layer sequence itself (seams, attention state, collectives, stored MoE
combine), not an attention operand dtype blocker.

## 8. Reproduce

    # header dump, read-only on a node with the released checkpoint
    ssh allenh1@192.0.2.80 scripts/dump_safetensors_headers.py /path/to/checkpoint > released-headers.json
    sha256sum released-headers.json   # f459bda49bbbba28d9d73570bb7bbf2beba6d71e0a96343644a97839ffa10786

    cargo build -p sparkflash-cli --bin sparkflash
    ./target/debug/sparkflash device-chain --run-stage attn-operands \
      --tensor-index released-headers.json --tp-rank 0 --tp-world-size 4 \
      --evidence-dir /tmp/attn-evidence --expect-plan-hash 0x9a59035fc40776c8
    # exit 0, 18/18 checks PASS, evidence /tmp/attn-evidence/tp4-attn-operands-r0.json

Host tests: cargo test -p sparkflash-device --lib resident:: (11/11, includes the two bind-refusal
tests) and cargo test -p sparkflash-cli --lib tp4_operands (5/5). No pin, cubin, or baseline change.
