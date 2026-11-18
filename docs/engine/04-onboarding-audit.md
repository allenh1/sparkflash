# 04-onboarding — audit

An audit of [04-onboarding.md](04-onboarding.md) and the open items in
[04-onboarding-review.md](04-onboarding-review.md), at `1e3cb3a9` (branch
`runtime-shared-plans`). Read-only on source: no source file was modified. This file is the only
addition.

## Scope and method

The plan says it reads the tree at `101ccb9` (plan line 3). Between `101ccb9` and `1e3cb3a9`
the only change is the plan itself (`git diff --stat 101ccb9..1e3cb3a9`: `docs/engine/04-onboarding.md`,
646 insertions), so every code citation is still at the line the plan saw. The review is an
untracked file at the same tip.

Counts below are measurements unless a row says otherwise. Where a plan count cannot be
reproduced, the method tried is named and the closest measurement is given; it is not massaged
into agreement. Commands are shown as run from the repository root.

Two conventions matter for the counts:

* A **"model-family enum"** here is one of the three enums whose variants name a family:
  `Pack` (`crates/sparkflash-cli/src/serve/pack.rs:639`, 7 variants), `Dialect`
  (`crates/sparkflash-cli/src/serve/dynamic.rs:75`, 4 variants), and `Image`
  (`crates/sparkflash-kernels/src/images.rs:31`, 15 variants; one image per family body).
  A match is **exhaustive** if it has no `_ =>` arm, so adding a variant stops compilation.
* **"Settings"** are distinct environment-variable names, not occurrences.

## Measurement table

| Plan claim | Measured at `1e3cb3a9` | Command / method | Verdict |
|---|---|---|---|
| A GLM-style family creates ~32 files | **32** EXACT | `git log --all --diff-filter=A --name-only --format='' \| grep -i glm \| sort -u \| wc -l` (files whose path names GLM, ever added) | holds |
| …and touches ~52 existing files | **55** non-GLM files in the union of the 15 GLM-titled commits; **42** in the narrower six-commit window; **74** total in that union | `for c in 01f757ee 1f1f127f 5d1490dc b57da442 1e836ee3 7e94f4e8 449a6f8e cb266f6a cfa19ad1 d75102c7 b091e1ce 8e001b2c f8e2311f 015c5885 e1574b84; do git show --name-only --format='' $c; done \| sort -u` | not reproduced exactly; 52 is within 3 of 55, and 55 includes 9 docs + `Cargo.lock`. No single `git log` method yields 52 |
| Only 9 match sites in 3 files stop compiling | **8 exhaustive** in 3 files: `Pack` 4 (pack.rs:660, 768, 798, 822), `Dialect` 1 (dynamic.rs:996), `Image` 3 (images.rs:216, 708, 737). A 9th family match, `Pack::is_arm` (pack.rs:748), is a `!matches!` and **does not** fail on a new variant | read every `match` over the three enums; wildcard list in Corrections 5 | off by one; 9 is reachable only by counting the one that does not stop compilation |
| 8 serving entry paths | **9 concrete server-construction sites**, or 8 if the pack host and group are counted as one entry (`serve_arm`) | read the dispatch chain `serve.rs:6471` → pack/dynamic; enumerate below | imprecise; the plan's eight is a categorisation, and none of the eight is derived from `ARMS` as written |
| 19 `SPARKFLASH_GLM_*` settings | **19** repo-wide; **19** in `crates/` | `grep -rhoE 'SPARKFLASH_GLM_[A-Z0-9_]+' --exclude-dir=.git --exclude-dir=target . \| sort -u \| wc -l` | holds |
| 22 `SPARKFLASH_QWEN4_*` | **22** repo-wide; **20** in `crates/` (2 names appear only in docs/scripts/site) | same with `QWEN4` | holds only repo-wide; cite the narrower number if the claim is "in the arms" |
| 19 `SPARKFLASH_INKLING_*` | **19** repo-wide; **18** in `crates/` (1 name only in docs/scripts) | same with `INKLING` | holds only repo-wide |
| Direct `env::var` reads: Inkling 20, GLM 2, Qwen 1 | **20 / 2 / 1** for `env::var` in `crates/sparkflash-model-*`; Qwen becomes **2** if `env::var_os` in `tests/vision_released.rs:31` is counted; `sparkflash-deepseek-forward` has 5 | `grep -rhoE '(std::)?env::var(_os)?' crates/sparkflash-model-*/ --include='*.rs' \| wc -l` | holds with the `var_os` caveat |
| 19 `load_named` sites in `sparkflash-deepseek-forward` | **19** | `grep -rn 'load_named' crates/sparkflash-deepseek-forward --include='*.rs' \| wc -l` | holds |
| About 205 distinct symbols | **219** distinct symbols in the `Self::Deepseek` arm of `Image::exported_symbols` (images.rs:461-…); **127** of them appear as string literals anywhere in the forward crate | parse the `Self::Deepseek => &[...]` block; `grep -rhoE '"[a-zA-Z0-9_]+"' crates/sparkflash-deepseek-forward/src` | off by +14 |
| About 105 of them in the catalogue | **76** of the 219 exported symbols are named in `CATALOGUE`; **51** of the 127 actually referenced are. The catalogue has **78** `PartKernel` rows, **203** `sym(...)` entries and **149** distinct names | `grep -oE 'sym\("[a-zA-Z0-9_]+"' parts.rs \| sort -u`; `comm -12` against the two symbol sets | off by −29 (the catalogue covers 35% of exports, not ~51%) |
| About 100 uncatalogued symbols | **143** exported symbols are not in `CATALOGUE`; **76** referenced ones are not | `comm -23` | undercounts by ~43 |
| Crate source lines in the table | EXACT for all seven, using `src/**/*.rs` only: forward 66,757; deepseek-v41 31,703; inkling2 24,269; qwen4-exp 20,372; glm5-next 14,002; dory 9,934; dense 3,095 | `find crates/<crate>/src -name '*.rs' \| xargs cat \| wc -l` | holds; the table deliberately excludes `tests/` |
| Forward: ~half layer arithmetic/kernel launches | **38,246-41,281** lines (57-62% of 66,757) in the layer/attention/MoE/tail files (`tp4_forward.rs`, `rows.rs`, `attention_device.rs`, `moe_device.rs`, `layer_device.rs`, `layer_attention_stages.rs`, `tp4_combine.rs`, `tp4_prefill.rs`, `tp4_token.rs`, `tp4_tail.rs`, `tp4_operands.rs`), before/after subtracting inline tests | `wc -l` on that set | "about half" holds loosely; it is nearer three-fifths |
| Forward: ~4,500 scheduling | **5,703** non-test lines / **6,321** with tests in `batch.rs` + `spec.rs` + `lanes.rs` | `wc -l`; subtract inline `#[cfg(test)]` blocks | low by 12-29% |
| Forward: 11,154 tests | **7,713** lines from the last `#[cfg(test)]` to EOF (an upper bound on test code); **371** `#[test]` functions; there is **no** `tests/` directory | sum of last `#[cfg(test)]`-to-EOF per file; `grep -rn '#\[test\]'` | not reproduced; the only test code that exists is inline and 7.7k lines |
| DeepSeek's CLI serving path is 14,545 lines | **14,545** EXACT (5,180 + 5,245 + 2,001 + 236 + 1,883) | `wc -l` on `device_chain.rs`, `tp4_host.rs`, `tp4_chain.rs`, `tp4_device_runner.rs`, `serve/device.rs` | holds |
| CLI is 35,657 lines, over a 34,000 cap | **35,657** under `crates/sparkflash-cli/src`; cap is `34_000` at `one_stack.rs:130`; the test at :132 therefore **fails today** | `find crates/sparkflash-cli/src -name '*.rs' \| xargs cat \| wc -l`; `grep -n CLI_SOURCE_LINE_LIMIT` | holds; note the cap test is red, and the crate is not in CI |
| `adding-a-model.md` §6 is a 15-item checklist | **15** items (15 `- [ ]` lines, `docs/adding-a-model.md:386-410`) | `awk '/^## 6\./,/^## 7\./' docs/adding-a-model.md \| grep -cE '^- \[ \]'` | holds |
| "The code allows 32" speculative lanes | `R_MAX = 64` (`rows.rs:44`), and the multi-sequence speculative branch runs only when `2 * many.len() <= R_MAX` (`batch.rs:2214`), i.e. 32; documented at `batch.rs:38`, `batch.rs:2619` | read constants and branch | holds |
| `design.md:117` says eight | the sentence is at **`design.md:118-119`**; line 117 is blank | `nl -ba docs/design.md \| sed -n '113,121p'` | wrong line, right statement |
| Resume statement at `NUMERICS.md:281` | the bullet starts at **281**, but the claim "a request resumed from the prefix a reused lane holds…" is at **284-285** | `nl -ba docs/NUMERICS.md \| sed -n '279,286p'` | line points at the paragraph, not the sentence; cite 284 |
| `03-runtime.md:266` says DeepSeek keeps its own `tp4_reduce::DeviceExchange` | line 266 is row 30 and the sentence is exactly there; the type is defined at `crates/sparkflash-deepseek-forward/src/tp4_reduce.rs:235`, and the CLI re-exports it at `crates/sparkflash-cli/src/lib.rs:52`. Only `serve.rs:4658` and `glm_tp4.rs:394` construct it. DeepSeek's serving path does open the shared `write_exchange` (`rdma_exchange.rs:34-44`) | read | statement is **literally true but misleading**, not false: the type lives in DeepSeek's crate, but DeepSeek's own path does not use the type; the pack paths do |
| Three bit dependences | `design.md:113` (bulk vs one-row last bits), `optimization-strategy.md:497` (chunk boundary moves last bits), `NUMERICS.md:281/284` (resumed lane differs from whole prefill) | read | holds; `NUMERICS` line should be 284 |

### The eight/nine entry paths, concretely

The dispatch chain is `sparkflash serve` → `serve::run` (`crates/sparkflash-cli/src/serve.rs:6471`)
→ either the DeepSeek path or `serve_pack` (`:6650`) → pack/dynamic. The construction sites are:

| # | Function | file:line | What it constructs |
|---|---|---|---|
| 1 | `run_tp4` / `serve_with_engine` (DeepSeek) | `serve.rs:6502`, `:6642` | DeepSeek reference or TP4 device server |
| 2 | `serve_arm<A>` → `serve_host<A>` | `pack.rs:844`, `:1178` | pack single-node backend behind `HostDriver` |
| 3 | `serve_arm<A>` → `serve_group<A>` / `serve_group_lanes<A>` | `pack.rs:1388`, `:1487` | pack TP-group / fused-lane group |
| 4 | `serve_arm_identity<A>` | `pack.rs:1309` (called from `dynamic.rs:518`) | dynamic id resolved to a released pack identity |
| 5 | `serve_inkling_host` | `dynamic.rs:837` | Inkling f32 host reference |
| 6 | `serve_inkling_group` → `inkling_group::serve` | `dynamic.rs:903`, `inkling_group.rs:36` | Inkling four-rank device group |
| 7 | `serve_dory_host` | `dynamic.rs:1011` | Dory device (or host fallback) |
| 8 | `recurrence::serve` | `recurrence.rs:323` | Dory ring schedule |
| 9 | `serve_novel` → `serve_dense_group` / `serve_host` | `dynamic.rs:602`, `:708`, `:925` | dynamic dense device or CPU host reference |

The plan's "8" merges rows 2 and 3 into one `serve_arm` entry and keeps the two Dory rows.
As written, **mechanism 5 collapses none of these**: the plan says `ARMS` derives "the registry,
the console's model list, conversion, the launch scripts, the test lists and the CI shards"
(plan lines 361-363), not serving dispatch. If `Arm` gains a `serve(mode)` method, rows 2-9 can
become table entries, but the *mode* branches (host/group/recurrence/dense-device) remain code;
row 1 stays separate until phase 6 step 6, and row 8 is explicitly deferred. So "one registration"
makes the table one; it does not make the serving paths one.

## Citations checked

The review verified the table of citations at its lines 14-35. Rechecked here, all hold except
where marked. Additional/uncertain ones:

| Citation | Result |
|---|---|
| `fused.rs:185`, `:246`, `:259`, `:287`, `:311`, `:374`, `:398` | hold exactly; 7 required, 14 defaulted methods confirmed by reading the trait |
| `fused.rs:35` "prompt chunks of two requests are not batched" | holds (line 35) |
| `pack.rs:73`, `:121`, `:748` | hold exactly |
| `serve.rs:4161` GLM context fallback | holds: `unwrap_or(sparkflash_model_glm5_next::DEFAULT_DEVICE_CONTEXT)` |
| `web/config.rs:1773` GLM spec for any non-Inkling lane group | holds; `false` in that match arm means "not Inkling", so Dory/Qwen are also given `SPARKFLASH_GLM_SPEC` |
| `dense/lib.rs:110` fixed 256-row chunk | holds (`CHUNK_ROWS = 256`) |
| `dense/lib.rs:2661` "the walk already has this signature" | holds only in the loose sense: `fn walk(shared, b, segments: &mut [(&mut Sequence, &[u32])], policy: Option<WidePolicy>, trace)` is segments+policy, but it is a free function, not `&self`, and has no `Pass`/`Hidden`/`head` |
| `dense/lib.rs:2749` "copies the full logits to the host" | **wrong line**: the doc comment for the head starts at 2749; the `copy_dtoh` is at **2778** |
| `dense/lib.rs:2785` "decodes step passes no tile policy" | **wrong line**: 2785 is `fn step`; the `walk(..., None, ...)` call is at **2799**, and the head also passes `None` at 2774 |
| `dense/lib.rs:3043` "states no drafter" | holds: `ledger.drafting(None, false, false)` is at 3043 |
| `projection.rs:95-117` | holds: `GLM` 95, `DORY` 102, `INKLING_TENSOR` 109, `DENSE` 116 |
| `plan.rs:508` `KernelSymbol` has name/precision/phase and no compute class | holds (508-516) |
| `tp.rs:60-75` `RowSums` refuses every wire but F32 | holds (refusal at 62-80) |
| `one_stack.rs:130` 34,000 cap | holds; the test at 132 is the one that now fails |
| `serve.rs:4658`, `glm_tp4.rs:394` `DeviceExchange` users | hold as construction sites; the type is the forward's, re-exported at `cli/src/lib.rs:52`; `recurrence.rs:278` uses `open_exchange_reduce` but not `DeviceExchange` |
| `serve.rs:1202-1246` GLM resume, `device.rs:6319-6360` Qwen resume | hold as ranges; `begin_prompt` starts at `model-glm5-next/src/serve.rs:1192` and `model-qwen4-exp/src/device.rs:6306`, and both call `sparkflash_runtime::resume` with near-identical comments |
| `group.rs:592-700` Inkling's second single-lane group path | holds for `crates/sparkflash-model-inkling2/src/group.rs` (the `GroupSession` impl of `ModelSession` is 594-668). The bare filename is ambiguous with `crates/sparkflash-runtime/src/group.rs:473` (`GroupLanes`), which is a different object |
| `batch.rs:886` `tick`, `:2035` `prefill_together`, `:1348` `adopt` | hold exactly |
| `03-runtime.md:266` | holds at 266 (see measurement table for the misleading part) |
| `design.md:113`, `optimization-strategy.md:497`, `:499` | hold exactly; `:499` carries 2,457 → 1,944 ms |
| `design.md:117` | **wrong line**: 117 is blank; eight is at 118-119 |
| `NUMERICS.md:281` | paragraph starts at 281; the cited sentence is 284-285 |
| `Cargo.toml` has no `sparkflash-ops` dependency | holds: `crates/sparkflash-deepseek-forward/Cargo.toml:29-48` |
| `opt_gemv_mma_*` not in the catalogue | holds: exported at `images.rs:562-564` and launched from `mxfp8_gemv_launch.rs:466-468`; `grep opt_gemv_mma crates/sparkflash-ops/src/parts.rs` → 0 |

## Corrections

1. **Line drift in the dense arm.** Replace `lib.rs:2749` with **2778** (the `copy_dtoh`) and
   `lib.rs:2785` with **2799** (the `walk(..., None, ...)` call). `:3043` is correct.
2. **`design.md:117` → 118-119.** The plan cites a blank line. The review also says 117 holds;
   it does not.
3. **`NUMERICS.md:281` → 284** for the actual resumed-lane sentence, or say "the bullet at 281".
4. **The match count is 8 compile-stopping sites, not 9.** Four in `pack.rs`, one in
   `dynamic.rs`, three in `images.rs`. The ninth family match the plan likely counted,
   `Pack::is_arm` at `pack.rs:748`, is a `!matches!` that compiles with any new variant. The
   plan's own text relies on that (it says a new variant becomes an arm without a decision), so
   the "9 … stop compiling" sentence contradicts the paragraph that introduces it.
5. **The wildcard sites are at least these** and none of them stops compilation:
   `profile_toml_for_family` (`pack.rs:74`, `_ => None`), `from_descriptor`
   (`pack.rs:675`, `_ => None`), `device_family` (`dynamic.rs:364`, `_ => false`),
   the family serve match (`dynamic.rs:518`, `_ => serve_novel`), `default_convert_world`
   (`convert.rs:425`, `_ => 1`), `shape_for` (`parts.rs:142`, `_ => ShapeConstraint::ANY`),
   and the web's `fit`/`hf` family matches (`fit.rs:242`, `fit.rs:357`, `hf.rs:236`).
   A forgotten family falls into one of those silently; mechanism 5 as written only removes the
   `Pack` table, not the string matches.
6. **Symbol counts.** 219 distinct DeepSeek-image exports, 127 referenced by the forward,
   76 in the catalogue, 143 exported-but-uncatalogued. The plan's "~205 / ~105 / ~100" is
   internally consistent for no measurement I can find. If the intended denominator was "symbols
   the forward names", the catalogue intersection is 51, not 105.
7. **The forward's test line count.** 11,154 is not line-countable here: there is no
   `crates/sparkflash-deepseek-forward/tests`, and the test tails sum to at most **7,713**
   lines (371 tests). Either the survey counted a different revision or counted test *functions*
   at an average that does not match.
8. **Scheduling is ~5.7k lines, not ~4.5k**, and layer/kernel code is ~57-62%, not "about a
   third" for the whole crate. (The plan's "about a third" sentence is about GLM's `device.rs`
   specifically, which this audit did not classify.)
9. **Qwen's setting count is 22 repo-wide but 20 inside `crates/`; Inkling 19/18.** If the
   intended claim is "each arm carries its own names", the smaller numbers are the honest ones.
   Likewise, Qwen has one `env::var` plus one `env::var_os` (in a test): say "1 in the arm,
   2 if the test helper is counted".
10. **GLM's created count is exactly 32; the existing count is 55 (or 42 on the narrower
    window), not 52.** The 32 is reproducible; the 52 is not.
11. **"8 serving entry paths" is 9 functions or 8 categories.** The ambiguity matters because
    the review's item 8 asks which collapse. None do under mechanism 5 alone (see table above).
12. **The CLI cap is not merely exceeded, its test is red.** `one_stack.rs:132` asserts
    `total < 34_000` against a measured 35,657. Adding `sparkflash-ops` to CI without raising
    the cap in the same commit turns CI red on the first run — phase 0's smallest first commit
    must include the cap change.
13. **The crate table is `src/` only.** That is fine, but it is not the crate's whole size:
    e.g. qwen4-exp is 31,076 with tests, inkling2 28,014. Say "source lines" if that is meant.
14. **The `03-runtime.md:266` "fix" is a clarification, not a correction.** The sentence is true
    of the type's crate; what is misleading is that the only constructor sites are pack paths
    and the DeepSeek path uses `write_exchange` directly. The edit should say that, not "the
    statement is stale".

## Missed by the plan

These are omissions in the mechanism list, not repeats of the review's ten items.

1. **`ModelSession` and `HostDriver` are the other engine, and no mechanism replaces them.**
   The plan itself notices the shape ("`ModelSession` has the same shape: no reusable prefix
   and zero cached tokens unless overridden", plan lines 83-84) but every mechanism then talks
   about `FusedArm`. The host path that serves pack and dynamic families is
   `sparkflash_serve::HostDriver` (`crates/sparkflash-serve/src/host.rs:165`), started from
   `pack.rs:1255` and `dynamic.rs:985`; it drives `ModelSession` (`model/src/backend.rs:140`),
   whose defaults are exactly the silent ones the plan criticises (`reusable_prefix` :201,
   `rewind_prefix` :212, `cached_tokens` :232). `Walk`/phase 1 does nothing for this path.
   A new family that lands on the host or dynamic path can still omit every strategy after the
   whole plan is implemented.
2. **Tokenizer, chat template and response-engine registration are outside `ARMS`.**
   `codec()` (`dynamic.rs:278-355`) is a hardcoded chain: GLM's `[gMASK]`/`<sop>`, then
   `family == "inkling2"`, then Qwen/Dory ChatML, then DeepSeek, else a refusal ("not a
   tokenizer this path can name"). `Dialect` (4 variants) is then matched to four response
   engines at `dynamic.rs:996-1001`; each API engine is its own module
   (`sparkflash-api/src/{glm,qwen,inkling}.rs`, plus `reasoning.rs` and `tool_parser.rs`).
   The "done" test says a family supplies "its tokenizer and chat template", but no mechanism
   registers it. `ARMS` as described only covers the model variant table.
3. **Vision is named in the trait sketch and absent from every mechanism.** `Walk` has
   `type Vision: Vision<Self>` (plan lines 205-206), but no code has a `Vision` trait for
   that role; what exists is `VisionCodec` (`model/src/vision.rs:240`) and the
   `ModelSession::prefill_images`/`held_images` pair (`backend.rs:163`, `:224`). The
   catalogue has exactly one `Part::Vision` entry, for `Image::Deepseek`
   (`ops/src/parts.rs:1463`); Inkling's tower is its own `src/vision.rs` and cubin, Qwen's is
   `src/vision.rs`, and the dense arm has no tower. `PartExec` has no image input, and
   `Walk::walk`/`head` have no image or span parameter. A vision family cannot be expressed
   by either.
4. **The dynamic HF path is described as "with the patterns added", but it cannot carry two of
   them today.** Non-dense dynamic families serve one request at a time: `dynamic.rs:490-515`
   refuses `--concurrency` and `--fuse-batch` unless the family is the shared dense arm, and
   the dense arm's group is the only dynamic `--tp4-rank` path. `dynamic.rs:42-47` /
   `ops/src/parts.rs:1660` deliberately ignore `Part::Mtp` and `Part::Vision`, so a dynamic
   checkpoint can have a drafter or a tower and the plan will not run either. `dense_device_refusal`
   (`ops/src/parts.rs:1894`) is a long shape/feature refusal list, so "made of catalogue parts"
   does not imply the dense arm takes it. Naming/conversion are separate per-family surfaces
   (`ops/src/naming.rs:260`, `cli/src/convert.rs:211-238`) with no `PartExec` counterpart.
5. **`sparkflash-engine` and `sparkflash-serve` are not in the plan at all.**
   `sparkflash-engine` is only a re-export facade over `sparkflash-model-deepseek-v41`
   (`crates/sparkflash-engine/src/lib.rs:22-37`), and `sparkflash-serve` is already
   "model-agnostic serving lifecycle" with a queue, concurrency and prefix reuse
   (`serve/src/lib.rs:20-27`, `host.rs:165`). The plan puts the one engine in
   `sparkflash-runtime` and phase 6 moves `Batch` there, but never says whether `HostDriver`
   is kept, deleted or made the engine. A reader implementing phase 1 will build a third loop.
6. **`sparkflash-scheduler` is a shadow scheduler.** 7,331 lines of batching/prefill/DSpark
   decision code (`crates/sparkflash-scheduler/src/lib.rs:20-37`, `batching.rs`,
   `prefill.rs`, `dspark.rs`) that the plan never mentions. It is a CLI dependency
   (`cli/Cargo.toml:70`), so it is in the build, even if today's serve loop does not call it.
   "One engine" has to say what happens to it, or phase 6 leaves a fourth model of scheduling.
7. **Per-arm CLI wiring survives `ARMS`.** `crates/sparkflash-cli/src/serve.rs` names GLM and Qwen device
   types directly (`serve.rs:4715-5200`: `load_glm_tp4_rank` at 4760, `load_qwen_tp4_rank` at 5128, and
   `Qwen4ExpDeviceBackend` at 4987), and `convert.rs:211/238/426` hardcodes family lists. Phase 2 says
   "the rank openers (about 900 lines) leave serve.rs", but the typed backend references and the
   conversion family list are separate, and a new family still edits them. The `ARMS` table alone
   does not make them compile errors.
8. **Neither `Walk` nor `PartExec` can express four things the released arms need:**
   * **mid-walk drafter taps.** `Walk::walk` returns `Hidden` after all layers and `head`
     runs afterwards (plan lines 214-217), but DeepSeek seeds the drafter from layers 37-39
     during the walk (`batch.rs:1658`, `:1825`, `:1950`). The plan adds "a hook between the
     layers and the head" only in mechanism 7 (line 433), not in the trait or `Pass`.
   * **Engram state.** A paged per-layer n-gram table with device reads interleaved in the layer
     walk (`crates/sparkflash-engram`, `engram_device_state.rs`, 681 lines) is neither
     positional nor recurrent, which are the only two regions `StateLayout` names (plan lines
     208-209).
   * **constraint state.** DeepSeek keeps a per-sequence `TokenConstraint` and verifies drafted
     rows under it (`Seq.constraint` at `batch.rs:253-257`, the per-row verify constraints at `batch.rs:2322-2335`, `sparkflash-api/src/constrained.rs`); `HeadRow`
     in the sketch has no constraint, and `head` is the only place the head sees a row.
   * **sampling/logprobs/stop/reasoning/tool parsing.** `StepOutput` carries the token and
     logits, but the request-level transformations live in `sparkflash-api` and
     `sparkflash-serve`; `Walk::head` cannot express them, and phase 1 does not move them.
9. **Drafter/vision types are required, but nothing says how an arm gets one.**
   `type Drafter: Drafter<Self>` and `type Vision: Vision<Self>` are "required" (plan lines
   203-206), yet the released families' drafters are wired through profiles
   (`sparkflash-model/src/profile.rs:265`, `DrafterProfile`) and the plan has no `Drafter`
   implementation for DeepSeek's MTP or Inkling's DSpark. The trait is a good forcing function;
   the mechanism that maps a published head to it is missing.
10. **The "compute class" covers only catalogue parts, and vision/MTP are catalogue parts only
    for DeepSeek.** `CATALOGUE` has `Part::Mtp` for DeepSeek and Dflash2, and `Part::Vision`
    only for DeepSeek (parts.rs:1428-1503, 355). The const assertion in mechanism 3 therefore
    either skips Inkling's vision/attention bodies or forces them into the catalogue, which is a
    larger change than "add a compute class". State it.

## Ownership map

The review's split (0 ‖ 1, 3a early) is directionally right, but its file-conflict claims are
too clean: phase 0's ledger work edits the *arms'* `ledger()` implementations, not only
`runtime/src/ledger.rs`, and phase 0's refusals edit the same GLM/Inkling device files phase 1
moves.

| Phase | Smallest honest first commit | Main files | Can run beside | Collides with |
|---|---|---|---|---|
| 0 | Raise `one_stack` cap to the measured 35,657 and add the six missing crates to both CI files in the same commit, so the cap test at least does not fail on the first run; any other failure this exposes is the point of the phase | `ops/tests/one_stack.rs:130`, `.gitlab-ci.yml:194-216`, `.github/workflows/ci.yml:158-203` | 3a (ops only); 1 only until the ledger/refusal commits | 1 on `model-glm5-next/src/device.rs`, `model-inkling2/src/device.rs`, `runtime/src/fused.rs`; 2 on `cli/serve.rs` and `web/config.rs`; 5 on `numerics/registry.toml` and arm ledgers; 7 on the same arm files |
| 1 | Add `Walk`, `StateLayout`, `ChunkPolicy` and a CPU unit test in `sparkflash-runtime`; move no arm (no bits, no `FusedEngine` rewrite yet) | new `runtime/src/walk.rs`, `runtime/src/lib.rs`; then `fused.rs`, dense `lib.rs` | 0 (except ledger), 3a | 0 on `fused.rs`/arm ledgers; 4/5/7 on dense/GLM/Qwen/Inkling device files; 6 on `fused.rs` and the group engine |
| 2 | Add the `ARMS` table and one read-only consumer (`sparkflash arms --list`) using it; do not rename settings yet | new `model/src/arms.rs`, `cli/src/models.rs` | 3a; 1 if no shared model file | 0 on `serve.rs`/`web/config.rs`; 6 on `serve.rs`; 8 on CLI |
| 3a | Add `ComputeClass` + row band to `KernelSymbol`, a first `WAIVERS` table, and the const check over `CATALOGUE` with all current gaps waived | `ops/src/plan.rs:508`, `ops/src/parts.rs:398`, new `ops/src/waivers.rs` | 0, 1, 2 (ops-only) | none for 3a; 3b collides with 5 on GLM token bodies and 6 on `deepseek-forward`/catalogue |
| 4 | Dense's step passes a decode policy (one call at `dense/lib.rs:2799`) plus the generated check; bits move, first dense reference | `model-dense/src/lib.rs:2774-2799`, `numerics/registry.toml` | 2, 3a, 5 (different families) | 1 on dense (must follow the dense move); 5 on dense if the shared token body is picked up first |
| 5 | One arm: GLM token body through the shared typed kernel, its numerics entry and waiver deleted/added; then Inkling experts/head, then Dory | `model-glm5-next/src/device.rs`, `numerics/registry.toml`, `docs/numerics/glm53/*` | 4 (dense), 6 (different crates), other arms in 5 | 0 (ledger/refusal), 1 (device files), 7 (same call sites) |
| 6 | Move `Batch`, `serve_ticks` and `Ahead` into `sparkflash-runtime` behind `Walk`, re-exported from the forward crate, with no logic change; then tick trace | `deepseek-forward/src/tp4_forward/batch.rs:886`, `spec.rs`, runtime | 5 (different crates), 8 (docs) | 2 on `cli/serve.rs`; 3b and 7 on `deepseek-forward`; 1 on `fused.rs`/group engine |
| 7 | One arm (dense) converts its call sites to `Mat<F>`/bound handles; add the string-lookup ban with that arm exempted | `model-dense/src/lib.rs`, then each arm | by arm, once that arm's phase 5 is done | 5 on every arm's device file; 6 on the forward; 4 on dense |
| 8 | `sparkflash onboard <hf-id>` reading `ARMS` + `CATALOGUE`, printing parts/formats/fit; no enforcement yet | new `cli/src/onboard.rs`, `docs/adding-a-model.md` | 7 (docs/CLI) after 2 | 2 (CLI/ARMS), 0 (docs) |

The first honest window is therefore **0 (cap+CI) ‖ 3a**, then **1**; phase 2 can start once the
phase-0 leak commits touch `serve.rs`/`web/config.rs`, but its `serve.rs` work must wait for
phase 6 or be rebased over it. Phase 5's GLM commit and phase 6's `Batch` move can be parallel;
phase 7 starts only after the corresponding phase 5 arm.

## What would most likely sink the implementation

1. **Two engines become three.** Phase 1 writes the strategy glue inside `FusedEngine`
   (`runtime/src/fused.rs:512`), which phase 6 deletes, while `HostDriver` and
   `ModelSession` already run the host/dynamic path and are not in any mechanism. The plan's
   own regression clause only covers DeepSeek's tick. If the trait is built before choosing the
   final engine, phase 1 is paid for twice and the "one walk" promise is still false for the
   path most new families use. This is the single largest risk.
2. **The registration is still not one.** `ARMS` covers the model table, but a new family
   still edits `codec()`, `Dialect` + the response-engine match, the conversion family list,
   the CLI's typed rank loaders, and (for vision) the tower and image path. The plan's "done"
   test is stated as a property of the compiler, but five of those six surfaces are wildcard
   matches or hardcoded chains, so a miss is silent. The audit's entry-path table is the concrete
   list to turn into tests.
3. **The gates cannot run where the plan says they run.** `one_stack` is red today (35,657 >
   34,000) and neither CI file runs `sparkflash-ops`; the generated `conformance!` needs a
   device and a fixture, while CI is GPU-free. If phase 0 adds the crates without the cap change
   CI is red, and if phase 1 ships a two-variant `Pass` (the review's item 2) the conformance
   checks cannot cover the chunk boundaries that the whole bit argument depends on. That turns
   the "generated test" promise into a manual gate at exactly the points the plan is trying to
   mechanise.

## What could not be measured

* The **"52 existing files"** has no single reproducible `git` method; the closest are 55 (15
  GLM-titled commits) and 42 (the six initial GLM commits). The 32 created files reproduce
  exactly. Do not present 52 as measured.
* **"11,154 tests"** could not be reproduced; the crate has no `tests/` directory and 7,713
  inline test lines. If the intended unit was test *functions*, the measured count is 371.
* **"About half is layer arithmetic and kernel launches"** is a classification, not a line
  count. The 57-62% above is my file-set approximation; a per-function classification was not
  attempted.
* **"More than a third of GLM's `device.rs` is sequence, lane and prefill orchestration"**
  was not measured; it needs a function-span classification of `model-glm5-next/src/device.rs`.
* The **survey's 9-match-site notion** may have counted a different enum or the `matches!` in
  `is_arm`; 8 is what stops compilation, 9 is what a match-site count including `is_arm`
  gives.
* Historical performance numbers in the docs (the 5%, 8%, 2,457→1,944 ms and bit-dependence
  claims) are quoted, not re-measured; re-running them needs the cluster.