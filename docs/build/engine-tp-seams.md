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
4. **Resident cache is raw bytes, eager, and covers every non-Engram tensor**: `new_tp` reads every
   planned tensor once at construction - the rank's routed-expert slices AND every replicated tensor -
   and keeps the raw stored bytes for the life of the engine; the per-token forward decodes them on
   use and issues no whole-tensor read against the caller's source. The two Engram table tensors per
   Engram layer are deliberately excluded: they are ~98 GB each, so they stay in the caller's source
   and are paged per token through `WeightSource::read_ranges`. The node target is ~84.7 GiB of shard
   plus ~9.8 GiB of replicated tensors (~94.5 GiB per rank) of raw bytes, not the ~4x a decoded-f32
   copy would need. The resident cache is the ONE cache: the CLI must pass the per-rank source and let
   `new_tp` build it, not pre-cache the same bytes (task-59 deletes the CLI `ResidentCache`). The
   read-count test asserts per-name: every non-Engram plan tensor read exactly once at construction
   and zero times after; each Engram table read zero times whole and paged at least once by a forward.

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
| world 2 seam 0 / seam 1 rel L2 | <= 5.0e-3 | **2.197e-3 / 2.421e-3** |
| world 4 seam 0 / seam 1 rel L2 | <= 5.0e-3 | **1.995e-3 / 1.793e-3** |
| resident reads after `new_tp` | unchanged | unchanged across two prefill calls (172 tensors resident, 2 paged) |
| replicated tensor read after construction | 0 more | 0 more (per name) |
| Engram table reads after construction | paged, not whole | 0 whole-tensor, > 0 paged ranges |
| resident bytes on the tiny fixture | every non-Engram plan tensor | **530638 bytes resident, 9216 bytes paged/excluded** |

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

No GPU, no vLLM, no node-side run of `new_tp`, no device kernel, no speed claim. The fixture numbers
are not the released model's. Tonight's four-node run uses tp-dev's CLI-resident runner, not
`new_tp`; the two paths are cross-checked on the fixture by task-53. The swap to `new_tp` is task-59
(tomorrow-shaped): one branch, the CLI `ResidentCache` deleted, the four-rank run reproduces the same
tokens, and the read-count test here proves replicated-resident / Engram-paged.

## 7. The task-59 swap: the exact constructor

`Tp4Rank::open` calls exactly this, with no CLI cache in front:

```text
let engine = sparkflash_engine::forward::InferenceEngine::new_tp(
    global_config.clone(),   // the same unsharded ModelConfig on every rank
    Box::new(rank_source),   // global names; LOCAL routed-expert specs/bytes; read_ranges for the Engram tables
    element_limit,           // same on every rank
    sparkflash_engine::parallel_tp::TpLayout { rank, world_size },
    Box::new(collective),    // the CLI adapter over the TCP transport
)?;
```

`new_tp` then builds the ONE `ResidentSource`: every non-Engram plan tensor is read once, raw, and
cached; the two Engram table names stay in `rank_source` and are paged per token through
`read_ranges`. The source must be complete at construction: every plan name present, routed `w1`/`w3`
local `[moe_inter_dim/world, dim]` with scales following on the row axis, routed `w2` local
`[dim, moe_inter_dim/world]` with scales on the column axis, every other tensor global, and
`read_ranges` answering `*.engram.embed.weight` / `*.engram.embed.scale` by row. The CLI must delete
its own `ResidentCache` in the same branch: passing an already-cached source would duplicate ~94.5 GiB.

For accounting and for a standalone check, the loader accepts any [ResidentPlan] rather than only
`TensorPlan`:

```text
ResidentSource::load(Box<dyn WeightSource>, &dyn ResidentPlan)
```

`TensorPlan` implements `ResidentPlan` with every planned name (what `new_tp` passes). The CLI
implements it for a thin local newtype over its `ShardPlan`, returning the plan rule names plus every
index name not in the plan; the local newtype satisfies the orphan rule, so no device crate change is
needed. The accounting to print next to RSS is `len()`, `resident_bytes()`, `paged_names()` and
`is_resident(name)`.

## 8. Released-checkpoint preflight for new_tp (prepared, NOT run)

A four-rank node run may construct `new_tp` with resident weights only when all of these hold:

1. **Source contract.** The rank source serves global names, the local routed-expert specs/bytes of
   section 5, and `read_ranges` for the two Engram tables per Engram layer. `new_tp` must return
   without a spec/read error: the plan load is the first whole-plan check the checkpoint sees.
2. **Memory.** Expected resident is ~84.70 GiB of rank shard plus ~9.8 GiB of replicated tensors,
   ~94.5 GiB of raw bytes (the CLI's current resident plan prints the same set). With the ~5.6 GiB
   shared-expert f32 and ~2.65 GiB head f32 transients, VmHWM ~95-105 GiB against 117-118 GB nodes.
   The Engram tables are not resident; they are read by row through the paged reader.
3. **Load time.** One cold sequential pass over the rank's bytes before any token. tp-dev measured
   85.5 GB of cold fill; expect `load_ms` in the minutes, and no collective is used during the load,
   so no TCP timeout applies to it.
4. **Adapter.** `TpCollective` is constructed and connected before `new_tp`; `world()`/`rank()` agree
   with `TpLayout`; `world_size` divides `moe_inter_dim` or `new_tp` refuses at construction.

Smallest evidence that construction succeeded, before any forward: (a) the resident accounting above
(tensor count, `resident_bytes` ~94.5 GiB computed from the local specs, `paged_names` exactly the
Engram tables, elapsed `load_ms`); (b) one cache probe - a replicated tensor served from the cache
with no further underlying read, and one Engram `read_ranges` call returning the expected row bytes;
(c) one all-reduce handshake at seq 0 with all ranks present; (d) `VmHWM` from `/proc/self/status` at
Ready. A run missing any of them should fail before the first token.
