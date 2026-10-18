# The m7_layer0 harness split (no behaviour change)

Status: **landed on branch `task/review-hygiene` as 954d7af** (one commit; not merged at the time of
writing - the Lead merges). Measured 2026-10-18 on that branch: the one file
`crates/sparkflash-kernels/examples/m7_layer0.rs` was **8,224 lines** at main `2245a4e` (`wc -l`;
the review said 8,100, which is not what the file measured) and is now **8,273 lines across 11
files** under `crates/sparkflash-kernels/examples/m7_layer0/`. The +49 net lines are the crate-root
glue (a 4-line layout comment, 10 `mod` declarations, 10 `use crate::stageN::stageN;` re-exports and
3 blank lines: 27 lines), the per-module import headers and their surrounding blank lines, minus the
5 dropped banner lines and rustfmt's tidying of blank lines at the new file joins. 103 items gained
`pub(crate)`; the original file had none.

This note is the claim, the proof, and - equally important - what was *not* run.

## 1. What moved where

Line ranges are 1-based inclusive in the original file. The slice boundaries are where a stage's
doc comment started in the original, found by walking back over its `///` lines and attributes so
no doc comment is orphaned from its item.

| new file | new lines | original lines | lines | contents |
| :--- | ---: | :--- | ---: | :--- |
| `main.rs` | 908 | 1-850 and 4332-4363 | 882 | crate docs, imports, `Args`/`parse_args`, memory floor, shared comparison and formatting helpers, `fn main` |
| `stage0.rs` | 244 | 851-1093 | 243 | stage 0, CPU fold-closure cross-check |
| `stage1.rs` | 436 | 1094-1527 | 434 | stage 1, Mega-mHC fused kernel vs reference |
| `stage2.rs` | 1040 | 1528-2566 | 1039 | stage 2, attention sublayer (engine/FP4 window formats) |
| `stage3.rs` | 747 | 2567-3311 | 745 | stage 3, MoE + shared expert |
| `stage4.rs` | 312 | 3312-3622 | 311 | stage 4, determinism repeat run |
| `stage5.rs` | 294 | 3623-3915 | 293 | stage 5, `cert_lm_head_sm121` |
| `stage6.rs` | 94 | 3916-4007 | 92 | stage 6, RoCEnante one-shot CPU/device seam |
| `stage8.rs` | 326 | 4008-4331 | 324 | stage 8, B1 probes of never-executed kernels |
| `stage7.rs` | 1924 | 4364-6276 | 1913 | stage 7, K5 flash_mla window attention |
| `stage9.rs` | 1948 | 6277-8219 | 1943 | stage 9, the 40-layer released-checkpoint column |
| (dropped) | - | 8220-8224 | 5 | a dangling section banner (`// Raw CUDA driver API ...`) whose code had already moved to `src/cuda_driver.rs` in D12's Task-8 extension. Comments only; nothing below it. |

Every module starts with `use super::*;` (the crate root's items, including its `use` bindings)
plus the sibling stages whose items it calls, at a `crate::` path because a sibling module cannot
be reached through `super`:

| module | also imports |
| :--- | :--- |
| `stage0`, `stage2`, `stage4`, `stage5` | - |
| `stage1`, `stage3` | `stage2` |
| `stage6` | `stage5` |
| `stage8` | `stage3` |
| `stage7` | `stage2`, `stage3`, `stage8` |
| `stage9` | `stage2`, `stage3`, `stage7`, `stage8` |

Only items used across a module boundary were given `pub(crate)` (103 items: stage entry points,
the shared `B2_STREAM_TENSORS`/`RoutedOperands` data, and the `LayerMhc` methods stages 7/8/9 call
on each other's values). The added keyword is the only token that changed; the verifier below
strips it back before comparing. The CLI is unchanged: `--stage` 0-9, the same flags, the same
usage and error strings, and `scripts/build-aarch64.sh --example=m7_layer0` still names the same
example target (`examples/<name>/main.rs` is auto-discovered as example `m7_layer0`).

## 2. Proof that the move is verbatim: `ci/verify-m7-layer0-split.py`

The check is checked in, so it can be re-run rather than believed:

```sh
python3 ci/verify-m7-layer0-split.py            # defaults to base commit 2245a4e
```

It reads the original from `git show 2245a4e:crates/sparkflash-kernels/examples/m7_layer0.rs`,
re-slices every stage range from the table above, removes only the lines the split added (import
headers, the crate-root layout comment and `mod`/`use` glue, `pub(crate) ` prefixes), normalises
blank-line runs at the file joins, and compares. Output on the split commit:

```text
OK    stage0 verbatim 243 lines
OK    stage1 verbatim 434 lines
OK    stage2 verbatim 1039 lines
OK    stage3 verbatim 745 lines
OK    stage4 verbatim 311 lines
OK    stage5 verbatim 293 lines
OK    stage6 verbatim 92 lines
OK    stage8 verbatim 324 lines
OK    stage7 verbatim modulo rustfmt re-wrapping and 1 inserted trailing comma(s)
OK    stage9 verbatim 1943 lines
OK   main.rs verbatim 882 lines (root 850 + fn main 32 lines, one trailing blank)
OK   dropped original lines 8220 to 8224 are comments only
PASS split is verbatim
```

The one formatting exception is stated rather than hidden: adding `pub(crate) ` to two constructs in
stage 7 pushed them past `max_width = 100`, so `cargo fmt` re-wrapped them
(`const B2_STREAM_TENSORS` and one multi-line method signature). The comparison there is
whitespace- and comma-insensitive and the only inserted punctuation is **one trailing comma** inside
a wrapped parameter list; no identifier, literal or operator differs. Every other module is
line-for-line identical after the glue is removed, including blank lines and comments.

The dropped lines are verified to be comments only and are checked not to reappear anywhere in the
split directory.

## 3. Behaviour proof on the paths this host can run

This dev host has **no GB10, no checkpoint, no cubins and no `libcuda.so.1`**, so the device stages
cannot run here and no stage numerics were compared. What could run was the CLI/validation surface.
A binary built from the pre-split HEAD was kept at `/tmp/m7_layer0_old` and the split binary was
built with `cargo build -p sparkflash-kernels --example m7_layer0`; both were run as
`./m7_layer0` from their own directory so `argv[0]` matches too. 17 argv paths were compared -
stdout, stderr and exit code, with only the live `MemAvailable` value normalised:

```text
[]                                                          exit 1
['--stage']                                                  exit 1
['--stage', 'xyz']                                           exit 1
['--stage', '0']                                             exit 1
['--checkpoint', '/nonexistent', '--stage', '0']             exit 1
['--checkpoint', '/nonexistent', '--stage', '1']             exit 1
['--bogus']                                                  exit 1
['--stage', '8']                                             exit 1
['--stage', '6']                                             exit 1
['--stage', '2', '--tokens', '130']                          exit 1
['--stage', '9', '--stage9-out', '/tmp/stage9-out']          exit 1
['--stage', '-1']                                            exit 1
['--mem-floor-gib', '1', '--stage', '0']                     exit 1
['--stage', '0', '--host-row', '/nonexistent/row']           exit 1
['--checkpoint', '/nonexistent', '--stage', '8']             exit 1
['--checkpoint', '/nonexistent', '--stage', '9', '--stage9-out', '/tmp/stage9-out']  exit 1
['--checkpoint', '/nonexistent', '--stage', '2', '--tokens', '7', '--cubin-dir', '/nonexistent/cubins']  exit 1
cases 17 differences 0
```

That covers argument parsing, stage selection, the per-stage header lines the harness prints before
any number, the memory floor reads and the first failure each path reaches on a host with no
checkpoint. It is **not** a numeric comparison of any stage.

Builds on the split commit, all green:

* `cargo check -p sparkflash-kernels --examples`;
* `cargo test --workspace` (the m4_abi pins and the two integration tests that read the example
  source all pass; see section 5);
* `cargo clippy --workspace --all-targets -- -D warnings` (only the expected no-nvcc build-script
  warning);
* `scripts/build-aarch64.sh --example=m7_layer0` (rust 1.98.1 + zig 0.17.0 wrappers): arm64 ELF
  built in 38.5 s, sha256 `30012b9b2d03fafe147789f73aeabaf0b9065dbf7c6224703082270ea03c3ef3`.

## 4. What was NOT run, and why that is the honest boundary

* Device stages 1, 2, 3, 5, 7, 8, 9 (and stage 4's repeat, 6's device half): need a CUDA context,
  the cubins and, for most, the released checkpoint. None is on this host. The split is proved by
  source identity plus the CLI paths above, not by re-running the device comparisons.
* Stage 0 and stage 6 CPU halves: need the released checkpoint and the engine's captured host row
  (`--host-row`, `--head-input`, `--logits`). The checkpoint and those artifacts live on the node /
  under `/home/user/phase2-evidence`, not in this checkout, so even the CPU stages were not run
  here.
* No cluster access, no window and no network action was involved.

A reviewer should read this as "the code that ran before is byte-identical and the CLI behaves
identically"; it is not a claim that any stage's numbers were reproduced after the split.

## 5. Forced follow-ups in the same commit (and why)

The split deletes the path that three pins read, so those pins were made to read the same text in
its new layout. All three changes are mechanical; the diffs are in 954d7af.

* `src/m4_abi.rs` read the example with two `include_str!("../examples/m7_layer0.rs")`. They were
  replaced by one `#[cfg(test)] const M7_LAYER0_SOURCE: &str = concat!(include_str!(...main.rs),
  include_str!(...stage0.rs), ... )` so the pins scan **the same text, concatenated**. No pinned
  count changed: the four `flash_mla_params(` builders, six `flash_mla_launch(` calls, one
  `flash_mla_launch_repeated(`, the two `engine_window_row(&` sites and the `stage9_attention_block`
  checks all still pass. The module grew by the 19-line constant: **13,893 lines**, so
  `ci/check-m4-abi-size.sh`'s `PINNED_LINES` and D26's measured numbers moved in the same commit -
  exactly the deliberate, reviewable act D26 describes.
* the driver-example pin (`examples_using_the_shared_cuda_driver_are_unsafe_free...`) looks for the
  example that imports `sparkflash_kernels::cuda_driver`; that is now `m7_layer0/main.rs`, so the
  vacuity check asserts on `main.rs` instead of `m7_layer0.rs`. It is still "the m7 example must be
  among the pinned driver users", and the `#![forbid(unsafe_code)]` check still covers the file that
  carries the import.
* `tests/mhc_released_rows.rs` and `tests/determinism_contract.rs` read the example source at run
  time. They now read every `*.rs` under `examples/m7_layer0/`, sorted, concatenated. The assertions
  are unchanged (four `mega_mhc_layer_with_scale(` call sites with checkpoint-order bias, one
  kernel-order derivation, the D-a/D-b/D-c citations). These two files are outside the declared
  write scope of the task; the Lead reviewed and accepted the mechanical edit.

Known stale references **not** fixed here because they are outside this task's scope (all
comments/docs, no code): historical DECISIONS entries D12's Task-8 extension (lines 239, 248) and the
D24 consequence at `docs/DECISIONS.md:1204`, both describing the file as it was at their own commit. The three stale `examples/m7_layer0.rs`
comments left here at !69 time (`crates/sparkflash-kernels/Cargo.toml:42`, `src/cuda_harness.rs:3`,
`src/mega_mhc_ref.rs:146`) were fixed in the !72 follow-up. `src/lib.rs` and `src/m7_layer0_shapes.rs` name `m7_layer0` (the example target) without the
file extension and stay correct.

## 6. Citing lines after the split

* Original lines **1-850** (crate docs, `parse_args`, shared helpers) keep their numbers in
  `main.rs`: `m7_layer0.rs:674-679` becomes `examples/m7_layer0/main.rs:674-679`. `fn main`
  (original 4332-4362; line 4363 was the blank join) moved to `main.rs:878-908` (the glue sits
  between the two original ranges).
* A line `L` inside a stage slice `[start, end]` is at `file line = 1 + len(imports) + 1 + (L -
  start)`: the import block, one blank line, then the original text. For example original
  `stage9` line 6277 is `stage9.rs` line 7 (5 imports + 1 blank line).
