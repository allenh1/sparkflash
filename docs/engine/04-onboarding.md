# 04 — Onboarding a model: the common patterns by construction

This page is a plan. Nothing in it is built. It reads the tree at `101ccb9`. The decisions taken
on 2026-11-18 are folded in and listed under [Decisions](#decisions).

[03-runtime.md](03-runtime.md) wrote each serving strategy once. This page is about the step after
that: a new model must not be able to leave a strategy out, and nobody should have to ask for one.

Counts marked *(survey)* come from a read-only sweep of the tree and are estimates. Everything
with a `file:line` was checked directly.

## The test for "done"

Someone adds a model family by supplying only what that family alone has:

* how its released config maps onto parts (`Architecture`),
* the arithmetic of any part no other model has,
* its weight names, tokenizer and chat template.

Without being asked, the result then has all of:

| Pattern | Meaning |
|---|---|
| Tensor-core prompt pass | every GEMM-class part runs on the MMA tile when rows allow |
| Chunked prefill | a prompt of any length runs in bounded chunks |
| Concurrent requests | several sequences share a pass, alone and in a TP group |
| Tensor-core decode | the one-to-eight row step uses the tensor cores too |
| Stored-format arithmetic | NVFP4 stays NVFP4, FP8 stays FP8; nothing widens unseen |
| KV reuse | a continued turn extends the cache; a changed turn rewinds it |
| Drafting | a published draft head is used, in lanes and in joint rounds |
| Two-rail RDMA exchange | on for every TP > 1 group |
| Packed weights | the tiled packed layout, on by default |
| DeepSeek's defaults | unless a measurement says otherwise |

A missing pattern fails in the earliest place that can express it: the compiler, then boot, then a
generated test. A person reading a checklist is the last resort, not the mechanism.

Two rows already work this way. The RDMA exchange (step 30, `35f3a86`) is built once, on for every
TP group, and has one switch to turn it off. Packed weights are the same. They are the model for
the other rows.

## What onboarding costs today

* A GLM-style family touches about 52 existing files and creates about 32 *(survey)*.
* Only 9 `match` sites in 3 files stop compiling when one is forgotten *(survey)*. The other
  registration sites fail silently or not at all.
* There are 8 serving entry paths *(survey)*: DeepSeek's own, the pack host and group, dynamic ids
  of pack families, two for Inkling, two for Dory, and the dynamic dense or CPU path.
* `docs/adding-a-model.md` §6 is a 15-item checklist kept by hand.
* Each arm carries its own run loop:

| Crate | Source lines |
|---|---|
| `sparkflash-deepseek-forward` | 66,757 |
| `sparkflash-model-deepseek-v41` | 31,703 |
| `sparkflash-model-inkling2` | 24,269 |
| `sparkflash-model-qwen4-exp` | 20,372 |
| `sparkflash-model-glm5-next` | 14,002 |
| `sparkflash-model-dory` | 9,934 |
| `sparkflash-model-dense` | 3,095 |

About a third of GLM's `device.rs` is layer arithmetic. More than a third is sequence, lane and
prefill orchestration that the other arms also wrote *(survey)*.

## Why the patterns are not inherited

Seven causes. Each one maps to a mechanism below.

### A. Strategies are hooks with silent defaults

`FusedArm` (`crates/sparkflash-runtime/src/fused.rs:185`) has 7 required methods and 14 with a
default body. Each default is a correct but degraded server:

| Left at its default | What the server does |
|---|---|
| `begin_prompt`, `next_part` (`:246`, `:259`) | the whole prompt is one part: no chunking |
| `history` (`:287`) | the slot is renewed every request: no KV reuse |
| `rewind_prefix` (`:311`) | a changed turn starts cold |
| `give_back` (`:374`) | drafting past the accepted row is refused |
| `ledger` (`:398`) | boot prints "the arm does not state it" |

`PackArm::OPEN_LANES` defaults to `None` (`crates/sparkflash-cli/src/serve/pack.rs:121`), and a
group without it serves one request at a time. `ModelSession` has the same shape: no reusable
prefix and zero cached tokens unless overridden.

A new arm that implements the minimum compiles, serves, and passes a smoke test. The only signal
is one boot log line.

### B. The glue from a walk to the strategies is copied

The runtime owns the scheduling. Each arm still owns the code that turns "run these rows" into
each strategy *(survey)*:

* the prompt-part state machine, in 4 arms;
* the resume preamble, close to verbatim in GLM (`serve.rs:1202-1246`) and Qwen
  (`device.rs:6319-6360`);
* the rewind trio, in 4 arms;
* `enable_fusion`, `enable_group_fusion`, `mirror_group` and `wire`, in GLM, Qwen and dense;
* `feed`, `holds`, `open_round` and `JOINT_WALL`, in 3 arms.

Every arm also picks its own chunk size: Inkling 4096, 8192 or 65,535; GLM 4096; dense a fixed
256 (`crates/sparkflash-model-dense/src/lib.rs:110`); Qwen its own `PrefillPlan`.

### C. Kernel choice belongs to the arm

* Row-count rules are arm constants: GLM `MAX_ROWS`, Qwen `GEMM_MIN_BATCH = 64`, Inkling
  `BATCH_ROWS = 8`, Dory `TOKEN_ROWS = 8`.
* The tensor-core tile is reached through `route_bf16(policy: Option<WidePolicy>, …)`. `None`
  means no tile, and the policies are constants named after models
  (`crates/sparkflash-ops/src/projection.rs:95-117`).
* Tensor-core decode exists twice and separately: DeepSeek's `opt_gemv_mma_*` and Qwen's
  `q4p_dense_rows` with `q4p_moe_rows_gate_up/_down`.
* `decode.tensor_cores=true` is refused by dense and Dory. GLM and Inkling never read it, so the
  switch is ignored there.
* `KernelSymbol` (`crates/sparkflash-ops/src/plan.rs:508`) records a name, the precisions it
  accepts and a phase. It has no notion of "runs on the tensor cores", so nothing can ask whether
  a part has such a body.
* The resolved plan is validated at boot and then dropped. Arms launch their own kernels by name.

The arm a new family lands on is the dense arm, and it is the furthest behind. Its decode step
passes no tile policy (`lib.rs:2785`), copies the full logits to the host every step (`:2749`),
and states no drafter (`:3043`).

### D. Format is a run-time value

A device pointer is a bare address. `Call<N>` carries `[u64; N]`. Uploaders mostly check a byte
length, not a dtype. The link from a weight's format to a kernel is a bitmask compared at boot.
Nothing stops a load path from widening a tensor and handing it to a wider kernel; the boot check
passes because the formats agree again by then.

### E. Registration is by hand

* `Pack::is_arm` is a negative match (`pack.rs:748`), so a new variant is an arm without anyone
  deciding it.
* `profile_toml_for_family` (`pack.rs:73`) is a string match that ends in `None`.
* Model names leak into shared paths. Every `--tp4-rank` group falls back to GLM's context
  (`crates/sparkflash-cli/src/serve.rs:4161`). Every lane group that is not Inkling is launched
  with `SPARKFLASH_GLM_SPEC` (`crates/sparkflash-web/src/config.rs:1773`).
* Settings have per-model names: 19 `SPARKFLASH_GLM_*`, 22 `SPARKFLASH_QWEN4_*` and 19
  `SPARKFLASH_INKLING_*` *(survey)*. Inkling reads the environment directly in 20 places, GLM in
  2, Qwen in 1.
* `numerics/registry.toml` has entries for Qwen3.8, GLM and DeepSeek. Inkling and Dory have none.

### F. The gates are a checklist, and some do not run

Neither CI file lists `sparkflash-ops`, `sparkflash-runtime`, `sparkflash-model-dense`,
`sparkflash-model-inkling2`, `sparkflash-model-dory` or `sparkflash-deepseek-forward`. So
`sparkflash-ops/tests/one_stack.rs`, which holds the "one stack" rules, does not run per commit.
Its 34,000-line cap on the CLI is already exceeded (35,657).

The checks every arm should have exist once, in one arm: Qwen's `batch_exactness.rs`,
`few_rows.rs` and `checkpoint_roundtrip.rs`. The runtime and the dense arm have no test directory.

### G. The most optimized model is a separate program

"Every model takes DeepSeek's defaults" cannot hold by construction while those defaults live in
a scheduler no other model runs.

* `sparkflash-deepseek-forward` does not depend on `sparkflash-ops` (its `Cargo.toml`). It has no
  catalogue, no `Call<N>` and no shared dispatch; it resolves kernels by string *(survey: 19
  `load_named` sites, about 205 distinct symbols, about 105 of them in the catalogue)*. The three
  `opt_gemv_mma_*` tensor-core decode bodies are not in the catalogue.
* Its scheduler (`Batch::tick`, `tp4_forward/batch.rs:886`) has strategies the shared engine
  lacks *(survey)*:
  * the prompt chunks of several requests in one tick (`prefill_together`, `:2035`); the shared
    engine states "The prompt chunks of two requests are not batched" (`fused.rs:35`);
  * a time-based decode share during a long prompt, and shortest-rest-first prompt order;
  * deciding the next tick under the running one;
  * taking a prefix from another lane (`adopt`, `:1348`), preemption, and a 64-deep queue.
* Its group sends one small tick frame and every rank replays the same scheduler. The shared
  `GroupLanes` broadcasts each decision.
* Its serving path in the CLI is 14,545 lines of its own (`device_chain.rs`, `tp4_host.rs`,
  `tp4_chain.rs`, `tp4_device_runner.rs`, `serve/device.rs`). That is why the CLI is over its cap.
* Of the forward crate's 66,757 lines, about half is layer arithmetic and kernel launches, about
  4,500 is scheduling, and 11,154 is tests *(survey)*. The scheduler is the small part.

Three of DeepSeek's behaviours would fail the generated checks of mechanism 6 today. Each is
already written down:

* its bulk path differs from its one-row path in the last bits (`docs/design.md:113`);
* those bits move with the chunk boundary for an idle prompt over about 2.6K tokens
  (`docs/optimization-strategy.md:497`);
* a request resumed from a reused lane can give different greedy text than the same request
  prefilled whole (`docs/NUMERICS.md:281`).

Two statements in the docs are stale and get fixed in phase 0. `03-runtime.md:266` says DeepSeek
keeps its own `tp4_reduce::DeviceExchange`; that type's only users are the pack paths
(`serve.rs:4658`, `glm_tp4.rs:394`), and DeepSeek already opens the shared RDMA exchange.
`design.md:117` says eight sequences share a speculative step; the code allows 32.

## The design

03 made the strategies shared. This plan removes the choice. There are seven mechanisms.

### 1. An arm supplies one walk; the runtime supplies every strategy

Replace the per-strategy hooks with one small required trait. Sketch:

```rust
pub trait Walk: 'static {
    /// One sequence's device state.
    type Seq;
    /// The draft source. An arm without one names `Absent<Reason>`; there is no default.
    type Drafter: Drafter<Self>;
    /// The image path, on the same terms.
    type Vision: Vision<Self>;

    /// Which state grows with position (a KV cache) and which is recurrent.
    fn layout(&self) -> &StateLayout;
    fn open(&self, memory: LaneMemory) -> Result<Self::Seq, BackendError>;
    /// Most rows one pass may hold, and the context.
    fn limits(&self) -> Limits;
    /// Run these rows of these sequences through the layers, in one pass.
    fn walk(&self, segments: &mut [Segment<'_, Self::Seq>], pass: Pass)
        -> Result<Hidden, BackendError>;
    /// Turn the last rows into tokens, on the device.
    fn head(&self, hidden: &Hidden, rows: &[HeadRow<'_>]) -> Result<Vec<StepOutput>, BackendError>;
}
```

The trait carries no thread bounds. An engine that needs `Send + Sync` asks for them itself;
DeepSeek's device state is single-threaded and must fit.

`sparkflash-runtime` then holds the only `FusedArm` implementation, `Runtime<W: Walk>`, and
`FusedArm` is sealed. The runtime owns:

* history, resume and the prefix match;
* the prompt-part state machine and chunk sizing (one `ChunkPolicy` from `Limits` and free memory);
* the checkpoint and rewind, driven by `StateLayout`: a positional region is truncated, a
  recurrent region is checkpointed;
* lane memory (`LanePool`), give-back, and the joint-round glue;
* group mirroring for any TP > 1 group, with the RDMA exchange as it is today;
* the ledger, computed from the types. "Does not state it" stops being a possible value.

What this buys:

* **Chunked prefill.** An arm has no prefill method. The runtime calls `walk` in chunks.
* **Concurrent requests.** `walk` takes segments. A walk that handles one segment handles several,
  and `OPEN_LANES` is deleted.
* **KV reuse.** An arm does not implement reuse. It declares its state, and `layout` is required.
* **Drafting.** Leaving `Drafter` out is a compile error. `Absent<NoneReleased>` is visible in
  review and printed in the ledger.

The dense arm's `walk` already has this signature (`lib.rs:2661`), so the trait is extracted from
working code. It is designed against DeepSeek's driver from the start (mechanism 7), so that one
arm interface serves both engines.

### 2. New families get the walk too

`sparkflash_model::compose` already has a generic layer walk (`Architecture::steps`). Only the
plan reads it. Make it the executor: a shared `PlanWalk` that implements `Walk` by running each
step's part.

A part that only one model has is a plug-in:

```rust
pub trait PartExec {
    type Weights: Weights;                          // typed, see mechanism 4
    fn state(&self) -> &[Region];                   // what it adds to the layout
    fn split(&self, plan: &TpPlan) -> Split;        // how it divides over ranks
    fn rows(&self, pass: &mut PassCx, w: &Self::Weights) -> Result<(), BackendError>;
    fn token(&self, pass: &mut PassCx, w: &Self::Weights) -> Result<(), BackendError>;
}
```

Every method is required. A part whose token body is its prompt body says so in one visible line.

Two levels follow:

* A model made of catalogue parts needs no model crate. That is today's dynamic path, with the
  patterns added.
* A model with one new part writes one `PartExec` and its kernel, and nothing about serving.

The released arms keep their hand-written walks behind `Walk`. They can move to `PlanWalk` later
or never.

### 3. The stack chooses the kernel

* Add a compute class to `KernelSymbol`: `TensorCore`, `Row` or `Elementwise`, plus the row range
  the body is for.
* Add a `const` assertion over `CATALOGUE`: for every GEMM-class part and every precision it
  accepts, a tensor-core body exists for the prompt pass and for the token pass, or a line in one
  `WAIVERS` table names the part, a row band and a measurement. Adding a part with only a row
  GEMV then fails to compile.
* The resolved kernel becomes the launched kernel. `KernelResolver` returns bound handles and
  arms launch through them. A workspace test forbids looking a kernel up by string in a model
  crate.
* Row dispatch moves into `sparkflash-ops`. Thresholds become a property of the symbol, not an
  arm constant. Each released arm's present thresholds are kept as a named legacy policy so its
  bits do not move. A new arm cannot name a policy; it gets the default.
* The two tensor-core decode implementations become the shared token body per precision. BF16 and
  NVFP4 exist in Qwen's, MX FP8 in DeepSeek's. FP8 E4M3 and the packed BF16 form have to be
  written.

One rule comes with this, learned on Qwen (D68): **a part has one arithmetic image per
precision, and it is the tile's.** The prompt body and the token body must give the same bits for
the same row, or a continued turn differs from a cold one. A few-row body may be a different
kernel, as Qwen's `q4p_*_rows` twins are, but it must reproduce the tile's bits. The generated
tests in mechanism 6 check it.

The compiler cannot see inside a `.cu` file. A kernels test closes that gap mechanically: a symbol
declared `TensorCore` must include the shared tile header.

#### Why the tensor cores at every row count

Other servers have no separate decode path. Checked on 2026-11-18:

* PyTorch has no half-precision GEMV. `gemv<Half>` and `gemv<BFloat16>` call `gemm`; the source
  says "There's no such thing as "cublasHalfgemv", so here we hack gemv with a gemm". `gemm`
  calls `cublasGemmEx` with `CUBLAS_GEMM_DEFAULT_TENSOR_OP`
  ([CUDABlas.cpp](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cuda/CUDABlas.cpp)).
  So a BF16 linear layer in vLLM or SGLang asks for a tensor-op GEMM at one row and at ten
  thousand.
* vLLM picks its quantized kernel by GPU capability and quantization scheme
  ([scaled_mm kernels](https://docs.vllm.ai/en/latest/api/vllm/model_executor/kernels/linear/scaled_mm/)).
  Nothing in the selection reads the row count or the phase.
* SGLang picks its FP8 and FP4 GEMM backend by GPU architecture
  ([quantization](https://docs.sglang.io/docs/advanced_features/quantization)). Its small-row
  refinement is still a tensor-core GEMM (a "swapAB optimization for small M dimensions in
  decoding"), not a step off it.

What was checked is the call and its flags. What cuBLAS does inside at one row is its own
heuristic and was not checked.

This tree's own measurements say where the gain is on this hardware:

* DeepSeek's tensor-core decode: the step is 5% shorter at eight rows and flat at one.
* Qwen's: about 8% slower at one row than its token kernels when it landed (18.5 against 20.1
  tokens a second at TP 1), and kept, because it makes a continued turn bit-equal to a cold one.

So at one row the reason is one arithmetic image, and at several rows it is also speed. That is
why the rule above is about bits, and why a waiver has to name a row band and a measurement.

### 4. Formats are types

* Zero-sized markers `Bf16`, `Fp8E4m3`, `MxFp8`, `NvFp4`, `F32` implement a sealed `Format`.
  A weight is `Mat<F>` and carries the scales its format needs.
* The uploader is `upload::<F>(tensor) -> Result<Mat<F>>`, and it checks the stored dtype
  (`TensorSpec::is_stored_as`).
* A bound kernel takes `&Mat<F>` for the format it was resolved for. Handing an NVFP4 matrix to a
  BF16 body is a compile error.
* There is no conversion between formats, except one `widen` that takes a reason and writes a
  record. The small exact widenings that exist today (GLM's BF16 vectors, dense's router bias,
  Inkling's clip gains) become explicit calls.
* Boot prints a format ledger: each part's stored format, its arithmetic format, and its bytes
  read per token. An arithmetic format that differs from the stored one without a record is
  refused.
* A generated test compares the bytes read per token with the stored-bytes floor from
  `EnginePlan`. This catches what types cannot: a recorded widening that is exact but slow.

Activations stay f32. That is the accumulation format, not a storage choice. The KV cache's format
is declared in `StateLayout`.

New shared code is typed from its first line (phase 3). The released arms' call sites convert
later (phase 7), because that is churn without new behaviour.

### 5. One registration

* An `Arm` trait and one `ARMS` table replace the `Pack` variants, `is_arm` and the family string
  match. A missing associated item is a compile error. DeepSeek has an entry like any other.
* The registry, the console's model list, conversion, the launch scripts, the test lists and the
  CI shards are derived from `ARMS`. A workspace test fails when a `sparkflash-model-*` crate is
  not in CI.
* Settings get neutral names through `profile.rs`'s `ENV_OVERRIDES`. A per-model name is refused
  with a message that names its replacement. A workspace test forbids `env::var` in a model crate.
* A `Defaults` table holds DeepSeek's values. An arm's profile may override one only with a
  `measured = "docs/…"` key beside it; the parser refuses the override otherwise.
* The two leaks go: the group context comes from the arm, and the console writes the neutral
  drafting name.

### 6. The gates are generated

* `conformance!(Arm)` expands to the checks over the arm's fixture. Qwen's existing tests are the
  source:
  * whole prompt = chunked = token by token;
  * alone = four at once;
  * continued turn = cold; rewound turn = cold;
  * TP 1 = TP 2 = TP 4 within the arm's declared exactness;
  * no unrecorded widening; bytes per token within the floor.
* An `ARMS` entry needs a token that only the macro can construct. An arm without the tests does
  not compile.
* `scripts/onboard-gate.sh <arm>` runs `served-gate.py all` and writes the first numerics
  reference. The registry refuses a first reference for an arm with an unwaived gap.
* `sparkflash onboard <hf-id>` prints, before any code is written: which parts the catalogue
  already has, which need a `PartExec`, the stored formats, the fit per TP, and the expected
  bytes-per-token floor.

### 7. DeepSeek joins by donating its scheduler

The direction matters. Moving DeepSeek onto today's shared engine would:

* lose strategies it has. Batching the prompts of several requests alone took the median time to
  first token from 2,457 to 1,944 ms at eight concurrent prompts
  (`docs/optimization-strategy.md:499`);
* change its tick, and with it the chunk boundaries its bits depend on;
* put a broadcast per decision on a step that is already at its floor.

So the move goes the other way. The runtime takes DeepSeek's scheduler, and the other arms move
onto it. DeepSeek's tick is then the same code in a new home, and every other model gains what
only DeepSeek had.

In order:

1. **Registration and the record.** An `ARMS` entry, neutral setting names, the ledger rows, and
   its three bit dependences entered as waivers. No driver change.
2. **Kernels in the catalogue.** Its uncatalogued symbols (about 100 *(survey)*) enter with a
   compute class. `opt_gemv_mma_*` become the shared MX FP8 and BF16 token bodies, since they are
   where that design came from. The forward crate takes the `sparkflash-ops` dependency and
   launches through bound handles. The kernels are the same.
3. **Transport and memory.**
   * One opener for the RDMA exchange. There are three today, over one implementation: DeepSeek's,
     Inkling's and `ops::doorbell`. The row cap becomes a parameter (DeepSeek's is 655,360 bytes,
     the others' 262,144).
   * `tp::RowSums` gains the BF16 wire with DeepSeek's sum kernel. It refuses every wire but f32
     today (`crates/sparkflash-ops/src/tp.rs:60-75`), and moving DeepSeek to f32 would change its
     bits.
   * `lane_memory` gains what DeepSeek's lanes have and the pool lacks: taking a prefix from
     another lane, the slide, and growth to 32 lanes.
4. **The engine.** `Batch`, `serve_ticks` and the tick frame move into `sparkflash-runtime`,
   generic over `Walk`. DeepSeek's row step, bulk prefill, drafter, Engram and head stay in its
   crate behind `Walk`. Its constants (4096-row idle chunk, 2048 busy, 64 rows a step, the band
   wall, the queue) become the `Defaults` table of mechanism 5.
5. **The other arms move onto it**, one at a time, dense first. `FusedEngine` and `GroupLanes`
   are deleted when the last arm has left. An arm that measures slower on the tick engine stays
   where it is, and the plan records why.
6. **The CLI path folds** into the common entry.

What `Walk` needs for this, beyond the first sketch:

* no thread bounds on the arm;
* `Pass` variants for what DeepSeek's tick asks: token rows, a bulk chunk, the paired chunks of
  several prompts, and verify rows under an output constraint;
* a hook between the layers and the head, where DeepSeek decides the next tick;
* state that is all positional: its rewind moves counters and needs no checkpoint.

The gate for step 4 is stronger than greedy text, which is all DeepSeek's numerics reference is:

* **the tick trace**: the scheduler's decisions on a replayed workload (lanes, rows, chunk
  boundaries, bulk or exact) recorded before and after, and required to be identical;
* the numerics check on a freshly booted server, as `NUMERICS.md` requires;
* alternated launches of the old and new build, several of each, at 1, 8 and 16 streams and at 2K
  and 16K prompts. Differences under about 0.2 ms a step do not show in a single pair.

**The regression clause.** Any step that costs DeepSeek speed is reverted, and the plan records
the measurement and what stays separate because of it. Steps 1 to 3 change where code lives.
Step 4 is the one with risk, and its design (the same tick, relocated) is chosen so that the
expected change is none.

What this does not fix: DeepSeek's three bit dependences stay as recorded waivers. Closing them
changes its bits and is a separate decision.

Dory's ring is a second schedule, not a lane walk. It moves into the runtime the same way and
under the same clause, after DeepSeek. Nothing else waits on it.

### Where each pattern fails after the change

| What no longer has to be said | What makes it true | A miss shows up as |
|---|---|---|
| Use the tensor cores for the prompt pass | catalogue assertion; dispatch in the stack | compile error |
| Use the tensor cores for decode | the same assertion, token pass | compile error, or a cited waiver |
| Chunk the prefill | the runtime drives `walk`; an arm has no prefill | not expressible |
| Support concurrent requests | `walk` takes segments; the runtime owns lanes | not expressible |
| Support parallel requests in a TP group | group lanes for any `Walk` | not expressible |
| Use the KV cache | required `layout`; the runtime owns reuse and rewind | compile error |
| Do not widen NVFP4 or FP8 | `Mat<F>`, typed launches, recorded `widen` | compile error, then boot refusal |
| …and do not do it slowly either | bytes-per-token floor | generated test |
| Draft when a head exists | required `Drafter` type | compile error; `Absent` in the ledger |
| Schedule the way DeepSeek does | one engine, and it is DeepSeek's | not expressible |
| Use the RDMA exchange | already the default for TP > 1 | — |
| Pack the weights | already the default | — |
| Take DeepSeek's defaults | `Defaults`; overrides need a citation | profile refused at boot |
| Register it everywhere | `ARMS` | compile error |
| Add the gates | conformance token | compile error |

## Phases

Each phase ends in a tree that serves every released model. Tests follow the standing rule: only
the changed components per stage, one full battery at the end of a phase.

Sizes are guesses: S is about one working session, M two to three, L more than that.

The order puts speed for the next model first (phases 1, 3, 4), then the approved bit changes
(5), then DeepSeek (6), then the churn that only tightens what already works (7, 8).

| Phase | What | Bits | Size |
|---|---|---|---|
| 0 | Make the gaps visible | neutral | S |
| 1 | One walk; the runtime owns the strategies; generated checks | neutral | L |
| 2 | One registration | neutral | M |
| 3 | The stack chooses the kernel; typed API for new code | neutral | M to L |
| 4 | The default arm gets everything | dense arm moves | M |
| 5 | Tensor-core decode on GLM, Inkling, Dory | once per arm | M each |
| 6 | DeepSeek's scheduler becomes the runtime's engine | neutral | L |
| 7 | Released arms' call sites take the typed API | neutral | L |
| 8 | The onboard command, the mandatory token, the docs | neutral | S |

### Phase 0: make the gaps visible. Bit-neutral. S

* Run the `sparkflash-ops`, `-runtime`, `-model-dense`, `-model-inkling2`, `-model-dory` and
  `-deepseek-forward` tests in CI. State the CLI's real line count in `one_stack` until phases 2
  and 6 bring it down.
* GLM and Inkling refuse `decode.tensor_cores=true` instead of ignoring it.
* The ledger gains rows for chunked prefill, tensor-core prompt pass, tensor-core decode,
  stored-format arithmetic and rewind. An arm that states nothing is refused at boot.
* `sparkflash arms --ledger` prints the gap table for every arm, DeepSeek included. It goes into
  the docs.
* Numerics entries for Inkling and Dory.
* Fix the two cross-model leaks and the two stale statements.

Gate: unit tests, and each arm's boot line. Value: the list of what each model lacks stops living
in anyone's memory.

### Phase 1: one walk. Bit-neutral. L

1. `Walk`, `StateLayout`, `ChunkPolicy`, `Runtime<W>` and `conformance!` in `sparkflash-runtime`.
   The checks come first, lifted from Qwen's tests, so every later move is tested the same way.
   The dense arm moves first.
2. GLM and Qwen move. Their resume preambles, prompt-part machines, rewind trios, fusion wiring
   and round glue are deleted.
3. Inkling moves onto `Fusion` and `resume`, and its second single-lane group path
   (`group.rs:592-700`) goes.
4. `FusedArm` is sealed and `OPEN_LANES` is removed.

Each released arm's present chunk sizes become inputs to `ChunkPolicy`, so its bits hold.

Gate per stage: `conformance!` and `served-gate.py all` on the moved arm, bit-exact against the
registry. Value: chunked prefill, lanes, group lanes, reuse and rewind can no longer be left out.
Expected removal is a few thousand lines per arm *(estimate)*.

After this phase the strategy glue exists once. That is what makes phase 6 a change inside the
runtime and not a rewrite of four arms.

### Phase 2: one registration. Bit-neutral. M

`Arm`, `ARMS` (DeepSeek and Dory included), derived lists, neutral setting names with the old
ones refused, the `Defaults` table, the `env::var` ban. The rank openers (about 900 lines) leave
`serve.rs`.

The deployed site files use the old names. They change in the deploy that carries this phase, and
that deploy waits for direction like any other.

Gate: the CLI and web tests, one boot per arm. Value: adding a family is one table entry, and a
forgotten site is a compile error.

### Phase 3: the stack chooses the kernel. Bit-neutral. M to L

* The compute class, the catalogue assertion and its first `WAIVERS` table.
* Row dispatch in `sparkflash-ops`, with each released arm's thresholds as its legacy policy.
* The shared token bodies: Qwen's BF16 and NVFP4, DeepSeek's MX FP8, and the two to write (FP8
  E4M3, packed BF16).
* `Mat<F>`, the checking uploader, bound handles and the boot format ledger, used by all new
  shared code.
* DeepSeek's uncatalogued symbols enter the catalogue (mechanism 7, step 2).

The first waiver table is the honest list of today's gaps: GLM's and Inkling's row-GEMV decode,
Dory's, the heads on row GEMVs, and DeepSeek's three bit dependences. Each line cites why it
stands.

Gate: bit-exact on every registered configuration; the new token bodies against the tile's bits
in kernel tests. Value: a missing tensor-core body is a compile error.

### Phase 4: the default arm gets everything. Dense bits move. M

The dense arm becomes `PlanWalk`:

* tensor-core decode through the shared token bodies;
* the head and sampler on the device, with no logits copy per step;
* a `Drafter` when the repository ships a draft head;
* chunk sizes from `ChunkPolicy` instead of a fixed 256;
* NVFP4 and FP8 parts through `PartExec`.

No released model runs on the dense arm and it has no numerics entry, so its first reference is
taken at the end of this phase.

Gate: the generated checks, and a TP 4 served gate on a dense fixture. Value: this is the phase
that makes the next model fast without a conversation.

### Phase 5: tensor-core decode on the released arms. Bits move once per arm. M each

GLM, then Inkling's experts and head, then Dory. Each is one declared change: a version bump in
the numerics registry, a report of how far the outputs move, and new references. After it, the
arm's waiver lines are deleted and its continued turns are held to its cold ones by the generated
check.

GLM goes first because its FP8 token body is the one phase 4 also needs, and a released model
with real weights is a better test bed for it than a fixture.

Expect the one-row step to be flat or slightly slower and the several-row step to be faster, as
on DeepSeek and Qwen. An arm whose one-row step loses more than its few-row twin can win back
keeps a waiver for that row band, with the number.

### Phase 6: DeepSeek's scheduler becomes the runtime's engine. Bit-neutral. L

Mechanism 7, steps 3 to 6: transport and memory, the engine, the other arms, the CLI path. Step 4
carries the tick-trace gate and the alternated launches. Step 5 is one arm at a time, each with
its own A/B.

Value: DeepSeek stops being a separate program, the other models gain batched prompts,
tick-ahead, the time-based decode share and prefix adoption, and the CLI loses most of 14,545
lines.

### Phase 7: the released arms take the typed API. Bit-neutral. L

Every kernel call site in GLM, Qwen, Inkling, Dory and DeepSeek moves to `Mat<F>` and bound
handles. String lookups in model crates are then forbidden by test. This is the phase with the
most churn and no new behaviour, which is why it is late. It can be split by arm.

### Phase 8: the onboard command, the mandatory token, the docs. Bit-neutral. S

`sparkflash onboard`, `onboard-gate.sh`, the conformance token required by `ARMS`.
`adding-a-model.md` shrinks to "what you supply" and its checklist is deleted, because the
compiler holds it.

### If only part of this is done

* Phases 0, 1, 3 and 4 deliver the speed patterns for the next model.
* Phases 2 and 8 deliver the low effort.
* Phase 6 is the only one that changes how the production model is scheduled. It can wait without
  blocking any other phase.
* Phase 7 turns the last "gets" into "cannot avoid" for the models that already exist.

## Decisions

Taken on 2026-11-18.

1. **Per-model setting names are refused**, with a message naming the replacement. The deployed
   site files change in the same deploy.
2. **The dense arm's bits move** in phase 4. Its first numerics reference is taken there.
3. **Inkling and Dory get tensor-core decode**, as GLM does (phase 5). Other servers do the same
   at every row count; the evidence is under mechanism 3.
4. **DeepSeek does not stay on its own**, unless a step would cost it speed. Mechanism 7 is the
   path and carries the clause.
5. **The order** is the table above.

## What this does not do

* Types prove that a pattern is declared and wired. They do not prove it is fast. Speed is held
  by the bytes-per-token floor and the served gate.
* A layer no model has had still needs its kernels. The plan makes that one trait with every body
  required, and nothing about serving.
* The catalogue assertion is planned as a `const` check. If the catalogue's shape resists that,
  the same check is a test, and phase 0 is what makes sure the test runs.
* Line counts removed are estimates until phase 1 has moved one arm.
* Whether DeepSeek's scheduler generalises over `Walk` without a cost is the plan's largest
  unknown. It is not known until phase 6's step 4 is measured. The survey found no recorded
  regression from sharing a DeepSeek piece, and none of the pieces shared so far was the tick.
