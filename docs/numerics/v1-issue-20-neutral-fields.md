# Version 1: issue #20's additions to the profile fields that cannot move the bits

This is the report `scripts/numerics.py ci` asks for when a run adds fields to
`sparkflash_model::numerics::NEUTRAL_FIELDS`. It covers the switches issue #20 added to the typed
profile table, one section a field, each naming the code that reads it, why changing it cannot
change a token's arithmetic, and the gate that holds that claim: a named test, an invariance in
[docs/NUMERICS.md](../NUMERICS.md) section 1, or an argument from the code a reader can check
without a GPU. The standard is section 5's: a field is neutral only if a profile may change it
without recapturing a reference.

The sibling report [v1-profile-neutral-fields.md](v1-profile-neutral-fields.md) covers the fields
the profile migration added; this one is the performance workstreams' own.

## Prefill

### `prefill.index_pack_append`

Read at `crates/sparkflash-deepseek-forward/src/tp4_forward/bulk.rs:49`
(`index_pack_append`). A prefill chunk either re-packs its whole key history into the persistent
fragment-order store or appends only the positions it completed. The store holds the same E4M3
codes either way: the pack kernel's address is a function of the absolute key index from a
multiple of eight (`index_pack_start` rounds the append point down to one), so the positions
before the append are untouched and the positions after it are the same expression over the same
source. The equality is measured, not structural: three identical requests an arm returned the
re-pack's own sha at 7,538, 30,246 and 60,488 tokens (docs/performance.md, WS2), and
`SPARKFLASH_INDEX_PACK_VERIFY=1` checksums the store after each pack. Gate: the measured sha
equality above; the host-side state machine (watermark, sequence identity, eight-key rounding,
reset at placement) is pinned by `index_pack_tests` in the same file.

### `prefill.index_pack_dedupe`

Read at `bulk.rs:145` (`index_pack_dedupe`). It skips the four later owner-3 packs inside one
chunk pass, which the checksum probe shows write the same bytes as the first. The skip is scoped
to one chunk pass (`pack_scope_is_new`), so it cannot carry a previous request's store into a
later one. Off by default: an unset profile takes the every-pack path. Gate: the same measured
sha equality (a run with the skip on reproduced the every-pack text at all three lengths).

### `prefill.index_pack_log`, `prefill.index_pack_verify`

Read at `bulk.rs:80` and `:170`. They print the blocks a pack asks for and checksum the packed
store after each pack: a host-side copy and a print after the launch, with every launched
argument computed before either branch. Gate: code argument.

## Speculation

### `speculation.row_ms`

Read at `crates/sparkflash-deepseek-forward/src/tp4_forward/spec.rs:1710` (`row_cost_ms`). It
replaces `StepCost::row_ms`, the price the depth policy charges a verified row. The price enters
the policy's ratio of expected tokens to modelled milliseconds; it never enters a kernel. The
invariance that holds it is `speculation never changes the output` (NUMERICS §1): the depth
policy chooses *how many* drafts are verified, and the verified rows' arithmetic and the emitted
tokens are the step's own. Measured: D106 has 85.2 against 85.6 tok/s at c1 and 29.7 either arm
at c16 with the tokens a step identical to two decimals; D113 priced the row at the measured
5.2 ms for **+0.0 %** paired over six prompts, text byte-identical (`a629cb2d93922881`).

### `speculation.batch_spec_cost`

Read at `crates/sparkflash-deepseek-forward/src/tp4_forward/spec.rs:1710`
(`batch_step_cost_from`), consumed by `share_rows`. It is the same mechanism as
`speculation.row_ms` one level up: the fitted two-band curve decides how many verify rows the
shared step gives each sequence, and the rows themselves are verified against the target exactly as
before. D110 measured the fitted arm against the flat one with text byte-identical
(`a629cb2d93922881`) at c4/c8/c16 and the per-stream rates flat within the 4-5 % spread. With the
knob unset the model is the pre-bend flat one exactly (`23 + 4 * rows`, which at a c=1 step is
27 + 4 x drafts), so the control arm is the engine's own.

## Diagnostics

### `diagnostics.engram_dedupe`

Read at `crates/sparkflash-deepseek-forward/src/tp4_forward/spec.rs` (`EngramReader` /
`EngramGather::settle`). A read whose (slot, table bytes) an earlier read of the same slot landed
is copied from that landing buffer instead of read again; the bytes copied are the bytes the second
read would have returned, and the upload that follows is unchanged. D108 measured identical text
with the switch on and off (`efb0f3d38a387f1d`), and `docs/serving.md` records the path as
bit-identical either way.

### `diagnostics.moe_dedup`

Read at `crates/sparkflash-kernels/src/moe_routed_device.rs` (`MoeRoutedOptions::dedup`,
resolved from the profile by the forward). `0` runs the routed experts once per routed row
instead of once per distinct expert; both paths are the same per-row arithmetic over the same
expert matrices (the dedup path groups rows by expert and runs each group's gate/up and down with
the per-row kernel's terms), which `docs/serving.md` records as producing the same bits.

### `diagnostics.moe_plan_in_cta`

Read at `crates/sparkflash-deepseek-forward/src/tp4_forward.rs:13643` (resolved from the profile
and passed into `MoeRoutedDevice::open`, because `sparkflash-kernels` does not depend on
`sparkflash-model`). It selects the in-block plan build (the A/B arm) over the hoisted
`opt_moe_plan_sm121` launch. Both arms call the same `opt_moe_plan_ids`/`opt_moe_plan_build`, so
the grouping, the group order and each group's row order are the same bytes, and the dedup
gate/up, down and combine read the same plan (DECISIONS.md D102). Measured text is byte-identical
across the hoisted arm, the in-block arm and the previous build at one and sixteen streams. Gate:
the single shared body (`kernels/opt_moe_sm121.cu:670`) plus D102's measurement.

### `diagnostics.moe_plan_stats`

Read at `crates/sparkflash-kernels/src/moe_routed_device.rs` (the stats buffer is allocated only
when on, and `read_plan_stats` reads it at drain). The armed plan kernel adds each call's
distinct count and one call marker into its band's pair after the plan is written; the additions
go to a separate buffer no kernel reads. Gate: code argument.

### `diagnostics.moe_ids_dump`

Read at `crates/sparkflash-deepseek-forward/src/tp4_forward/rows.rs:3701` (`rows_moe_routed`).
It copies the step's routed ids `[n][top_k]` to the host and prints them after the layer's
kernels are enqueued. Gate: code argument (a `copy_dtoh` of a buffer the layer wrote; no launch
or argument changes).
