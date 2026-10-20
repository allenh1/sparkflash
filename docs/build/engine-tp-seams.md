# The engine TP seam: Phase-1 sharded reference forward

Task-54. The engine half of the tensor-parallel path in `docs/build/tp4-design.md` section 3: the
reference forward runs on ONE RANK's shards, with one collective per layer and weights resident for
the life of the engine. This document records the seam list, the deviations from the full section-3
map as numbers, and the bars declared on task-54 before the run, with the measurements.

The transport half, the TCP adapter, the launch and the evidence ledger are task-53 (tp-dev); the
engine never names a comm type. The trait is `sparkflash_engine::parallel_tp::TpCollective` and the
entry point is `InferenceEngine::new_tp`.

## 1. Phase-1 seam list

| # | tensor(s) | section-3 rule | Phase 1 | collective |
|---|-----------|----------------|---------|------------|
| S1 | routed experts: `w1`/`w3` rows, `w2` input columns | column / row parallel, local inter = `moe_inter_dim / world` | implemented | one all-reduce of the routed FFN output `[T, dim]` per layer |
| — | attention `wq_a`, `wkv`, `wq_b`, `wo_a`, `wo_b` | replicated + column/row in section 3 | replicated | none |
| — | embedding, head | vocab rows / world | replicated | none |
| — | shared expert, router, mHC, norms, indexer, compressor, Engram | replicated | replicated | none |

The one seam per layer is the routed FFN output; the shared expert is added host-f32 after the
reduce, in the reference's order (routed sum first, then shared add), so no extra rounding is
introduced by that step. With `world() == 1` the adapter must be a strict no-op (no bf16 round, no
write), which is what makes the one-rank engine bit-identical to today's reference.

## 2. Deviations from the section-3 map, recorded as numbers

1. **Attention replicated** (no `wq_b`/`wo_a` column split, no `wo_b` all-reduce): zero numeric
   delta by construction (the tensor is not split; every rank computes the same units). Cost: the
   attention weights stay at full size on every rank.
2. **Embedding and head replicated** (no vocabulary all-reduce, no head gather): zero numeric delta
   by construction. Cost: the 32320-row embedding and head stay at full size on every rank.
3. **The routed seam is quantised to bf16** while the single-rank reference stays f32: this is the
   only numeric class in Phase 1. Measured per seam below.
4. **Resident cache is raw bytes, eager**: `new_tp` reads every tensor the plan names once and keeps
   the bytes for the life of the engine; the per-token forward reads nothing from the caller's
   source. On the node the decoded-f32 form does not fit (68.9 GiB/rank of expert bytes is ~275 GiB
   decoded to f32 against 117-118 GB/node), so the node run keeps the CLI-side streaming runner; this
   host engine path is for the identity/equivalence proof and the fixture. The read-count test
   asserts the residency property for the host path.

## 3. Bars declared on task-54 BEFORE the run

Relative L2 is `||a-b||2 / max(||a||2, ||b||2)`.

* **identity**: `world == 1` logits bit-identical to the reference (0 differing elements).
* **per seam**: routed FFN output at each layer against the single-rank f32 reference, relative L2
  <= **5.0e-3** (about 2.5 bf16 ulp of the seam).
* **end-to-end**: logits against the single-rank reference, relative L2 <= **2.0e-2** for
  `world = 2` and for `world = 4`; `world = 2` vs `world = 4` logits relative L2 <= **2.0e-2**.
* **resident**: after `new_tp` returns, a per-token forward issues zero reads against the caller's
  source.

## 4. Measured on the tiny fixture (host, dev machine)

Fixture: `ModelConfig::tiny()`, `n_layers = 2`, `moe_inter_dim = 32`, seed 7,
`FiniteSource` from `tests/common`, prompt ids `[3, 7, 11]`, compared position = final. Command:
`cargo test -p sparkflash-engine --test tp_seams -- --nocapture`.

| measurement | declared bar | measured |
|---|---|---|
| identity logits, world 1 | 0 differing | **0 of 128 differ**, bit-identical |
| world 2 logits rel L2, both ranks | <= 2.0e-2 | **0e0**, 0 of 128 differ |
| world 4 logits rel L2, all ranks | <= 2.0e-2 | **0e0**, 0 of 128 differ |
| world 2 vs world 4 logits | <= 2.0e-2 | **0e0**, bit-identical |
| world 2 seam 0 / seam 1 rel L2 | <= 5.0e-3 | **2.197e-3 / 2.413e-3** |
| world 4 seam 0 / seam 1 rel L2 | <= 5.0e-3 | **1.995e-3 / 1.833e-3** |
| resident reads after `new_tp` | unchanged | unchanged across two prefill calls |

The end-to-end logits measured exactly 0 because the tiny fixture's routed FFN outputs are O(1e-9)
absolute (the fixture weights are O(1e-2) and the layers are two), so a bf16 seam perturbation of
~4e-11 absolute is below the f32 logit ulp (~2e-9 near 0.03). The per-seam bar is the binding
measurement on this fixture; on the released model the same seam perturbation is O(1e-3) of the
activation and the end-to-end bar becomes the binding one. Neither number is a claim about the
released model: they are the fixture's measured values against the declared bars.

## 5. Local-spec contract (agreed with task-53 before either side wrote the seam)

The source passed to `new_tp` answers global tensor names; routed-expert tensors carry this rank's
local specs, every other tensor is the global (replicated) spec:

* `w1`/`w3`: local logical shape `[moe_inter_dim/world, dim]`; scales follow the weight on the row
  axis (`[moe_inter_dim/world, ceil(dim/32)]`).
* `w2`: local logical shape `[dim, moe_inter_dim/world]`; scales follow on the column axis
  (`[dim, ceil((moe_inter_dim/world)/32)]`).
* everything else: the global spec.

The adapter trait is `TpCollective`: `world`, `rank`, `label`, `all_reduce_assign`,
`all_gather_assign`, and `send`/`recv` (default bodies refuse; the Phase-1 seam has no send/recv
leg). The canonical wire rule for `world > 1`: each rank's f32 values are rounded once to bf16 (RNE),
accumulated in f32 in ascending rank order from 0.0, rounded once to bf16 (RNE) and widened back;
`world == 1` is a strict no-op. `all_gather_assign` is the same rule with a plain concatenation in
ascending rank order; the Phase-1 forward does not call it.

The engine-side reference arithmetic for that rule is `parallel_tp::canonical_reduce_into`, so a
test can check an adapter against the same numbers without a transport.

## 6. What is not claimed

No GPU, no vLLM, no node-side run, no device kernel, no speed claim. The fixture numbers are not the
released model's. The node path (task-53) streams its shards and is not wired to `new_tp`; the CLI
runner there exercises `TpCollective` through the TCP adapter independently, and the two paths are
cross-checked on the same fixture by task-53.
