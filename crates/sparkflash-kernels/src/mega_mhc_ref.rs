// Copyright (c) 2026 Hunter L. Allen
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! CPU reference for the Mega-mHC kernel (kernels/mega_mhc_sm121.cu), specification 2.4.
//!
//! Mega-mHC updates four residual streams X_l of width d with
//!
//!   X_{l+1} = B_l X_l + C_l F_l(A_{l-1} X_l),   (A_l, B_l, C_l) = H(X_l)
//!
//! and the whole point of the kernel is that it reads X_l ONCE. The A that scales the input to F is
//! the one the previous layer produced, which is still in registers, so there is no second read of
//! X_l to recompute it. That is why the traffic is (2n+2)d elements instead of the (4n+4)d of a
//! multi-pass formulation.
//!
//! This module therefore does two things:
//!
//! 1. it defines the arithmetic: H, the Sinkhorn-Knopp normalisation of B, the softmax for A and C,
//!    the RMS pre-norm and the per-32 ue8m0 FP8 quantisation of the block input;
//! 2. it COUNTS the traffic, mechanically. The streams live in a Streams value whose data is
//!    private and whose read accessor increments a counter, so a second pass over X_l cannot be
//!    written without being counted; the register copy is a separate type with its own counter, and
//!    the written buffers are Counted, so every write passes through a counter too. The ledger is
//!    filled by observe() from those counters at the end of the layer, and its passes() is derived
//!    from x_read rather than assigned - an earlier version set passes = 1 by hand, and a reviewer
//!    added a genuine second pass over the streams without moving the number. The tests assert the
//!    specification's own (2n+2)d expression, that X_l is read exactly once, and - for non-vacuity -
//!    that a deliberately three-pass variant of the same math blows past both this reference and the
//!    (4n+4)d baseline. A bound a broken implementation also satisfies would be worthless, so the
//!    three-pass variant is part of the test suite, not a comment.
//!
//! H is under-specified by the specification, which fixes its output shapes (n*n + 2n = 24 rows plus
//! the accumulators) but not the internals. This module states the project's reading explicitly and
//! implements it: the released (n*n + 2n) x (n*d) prediction matrix, whose rows are the released
//! pre(0..n) / post(n..2n) / comb(2n..2n+n*n) rows permuted into this reference's kernel order by
//! [`released_predictor_into_kernel_rows`]; the first 16 kernel rows become the 4x4 mixer B, the
//! next 4 the input mix A and the last 4 the block-output weight C. How they are activated and
//! normalised is not this module's choice: the released model uses
//! `sigmoid(...) + hc_eps`, `2*sigmoid(...)` and the released split-Sinkhorn schedule, and an
//! earlier version of this file softmaxed A and C and ran a generic Sinkhorn-Knopp on B. Those were
//! semantic differences, not refinements; the released arithmetic is implemented by
//! [`h_of_with_scale`] / [`mega_mhc_layer_with_scale`] and the softmax reading is still built,
//! and labelled, by the no-bias and bias-only entry points.
//!
//! ## The per-coefficient bias: hc_attn_base / hc_ffn_base
//!
//! The released checkpoint carries a 24-value fp32 **bias** per predictor - `hc_attn_base` for the
//! attention predictor and `hc_ffn_base` for the FFN one, one vector per layer. Neither the fused
//! kernel nor this reference used to take it, so an M7 comparison ran with it silently absent. It is
//! not an output offset and not a per-token term: `TransformerBlock.hc_mixes` computes
//!
//! `mixes = F.linear(x.flatten(2).float(), hc_fn) * rsqrt(mean(x^2) + norm_eps)` (model.py:952-954)
//! and then `hc_split_sinkhorn(mixes, hc_scale, hc_base, ...)` (model.py:955), whose body adds
//! `hc_base[row]` to the **raw logit** of row `row`, after the projection and after the rsqrt
//! scaling, and BEFORE any non-linearity (inference/kernel.py:426-431):
//!
//! `pre[j]     = sigmoid(mix[j]        * scale[0] + base[j])       + eps`
//! `post[j]    = 2 * sigmoid(mix[j + n] * scale[1] + base[j + n])`
//! `comb[j][k] = mix[j*n + k + 2n] * scale[2] + base[j*n + k + 2n]` then softmax + Sinkhorn
//!
//! So: it is added before the softmax and before the Sinkhorn, it is added to the coefficient logits
//! of all 24 rows (the 4 pre, the 4 post and the 16 comb), it is put on top of the scaled logit and
//! never multiplied by the scale, and it is a CONSTANT of the model - one vector per predictor,
//! shared by every token. `comb` is the 4x4 residual mixer (this reference's B), `pre` is the
//! input mix (this reference's A) and `post` is the block-output weight (this reference's C).
//!
//! Two layouts therefore have to be reconciled, and the naive one is wrong. The checkpoint's row
//! order is pre(0..n) / post(n..2n) / comb(2n..2n+n*n); this reference's H row order is B(0..n*n) /
//! A(n*n..n*n+n) / C(n*n+n..n*n+2n), because the first 16 rows are the ones Sinkhorned. On top of
//! that, `comb` is indexed `[source][destination]` (`hc_post` sums `comb[j][k] * residual[j]`
//! over j, so j is the source copy and k the destination), while this reference's B is indexed
//! `[destination][source]` (`b[i][j]` mixes stream j into stream i, and the kernel applies
//! `s_b[s*n + t] * x[t]` to output s). The 16 comb values therefore arrive transposed.
//! `bias_into_kernel_rows` performs that rearrangement, a test pins it value by value, and another
//! test shows that adding the vector index for index lands the pre/post values on the wrong
//! coefficients.
//!
//! ## The released pre/post activation and hc_scale: a semantic change
//!
//! Two released behaviours were missing here, and both change the numbers rather than refine them.
//! `hc_split_sinkhorn` is called with the checkpoint's `hc_scale` (model.py:948-955) and applies,
//! after the projection and its rsqrt and BEFORE any non-linearity (inference/kernel.py:426-431):
//!
//! `pre[j]     = sigmoid(mix[j]         * hc_scale[0] + base[j])       + hc_eps`
//! `post[j]    = 2 * sigmoid(mix[j + n] * hc_scale[1] + base[j + n])`
//! `comb[j][k] = mix[j*n + k + 2n]      * hc_scale[2] + base[j*n + k + 2n]`
//!
//! with `hc_scale` the `[3]` tensor of model.py:945-946, in the order pre, post, comb, and
//! `eps = self.hc_eps = 1e-6` (model.py:937, 955; the call site is model.py:955).
//!
//! * **The scale multiplies, then the bias is added.** The order is `mix * scale + base`, never
//!   `(mix + base) * scale`, and the scale is per coefficient GROUP - one scalar for the four pre
//!   rows, one for the four post rows, one for the sixteen comb rows - not per row.
//!   [`scale_into_kernel_rows`] places the three scalars on the 24 rows in this reference's order,
//!   [`h_of_with_scale`] and [`mega_mhc_layer_with_scale`] apply them, and a test pins the order
//!   by computing both wrong orders and showing they land on different numbers.
//! * **The pre and post activations are sigmoids, not softmaxes.** They do not normalise across the
//!   four copies: each `pre` entry is independently `sigmoid(...) + hc_eps`, in
//!   `(hc_eps, 1 + hc_eps)`, and each `post` entry is independently `2 * sigmoid(...)`, in
//!   `(0, 2)`. This module's earlier reading softmaxed the four pre logits and the four post
//!   logits, which forces each vector to sum to one and gives it a different shape entirely (at zero
//!   logits, `0.25` each instead of `0.5 + eps` and `1.0`). That reading is retained as
//!   [`PrePostActivation::Softmax`] because entry points were built on it, and every entry point now
//!   says which reading it is:
//!
//!   - [`h_of`] and [`mega_mhc_layer`]: no bias, no scale, softmax A and C - the ORIGINAL reading.
//!   - [`h_of_with_bias`] and [`mega_mhc_layer_with_bias`]: the checkpoint bias, no scale, softmax
//!     A and C - the bias-only reading, unchanged.
//!   - [`h_of_with_scale`] and [`mega_mhc_layer_with_scale`]: the checkpoint bias AND scale,
//!     the projection rstd, the released sigmoid activation and the released comb schedule - the
//!     reading a released-model comparison wants.
//!
//! The arithmetic AROUND the logits is now the released arithmetic, each piece with the source
//! lines above and a test that separates it from the older reading:
//!
//! * **the projection rstd**: `rsqrt(mean(x^2) + norm_eps)` over the flattened `n*d` stream,
//!   `norm_eps = 1e-20` (model.py:953-954; D4 in docs/architecture/02-mhc.md), applied BEFORE the
//!   group scale and the bias. `coefficient_rstd` is the single definition; a test pins the
//!   constant (1/sqrt(1e-20), not 1/sqrt(1e-6)) and the n*d mean.
//! * **the comb schedule**: the released softmax-then-alternate schedule in the released
//!   `[source][destination]` orientation (inference/kernel.py:436-458; D3), implemented by
//!   [`released_split_sinkhorn`] and [`released_comb`] rather than by the generic
//!   [`sinkhorn_knopp`]. A test transcribes the schedule by hand and compares every matrix entry.
//! * **the block RMS statistic**: the RMSNorm of the COLLAPSED block input normalises X_hat with
//!   X_hat's sum of squares over `d` and `norm_eps` (model.py:984, 991), not the streams' sum
//!   over `n*d`, and not the Sinkhorn epsilon.
//!
//! The predictor used to be the folded `[24, d]` matrix applied to the stream mean, which is the
//! release's flattened `[24, n*d]` projection divided by the copy count when the copies are equal.
//! **B0 (2026-10) closed that difference**: every entry point now takes the released `[24, n*d]`
//! predictor (rows permuted into kernel order, NOT averaged; see
//! [`released_predictor_into_kernel_rows`]) and projects all four copies, and there is no fold
//! helper left to silently reintroduce it. `kernels/mega_mhc_sm121.cu` took the same change in the
//! same window. Pre-B0 stage-1 device runs measured the folded semantics; those numbers are retired.
//!
//! ## Entry points
//!
//! [`h_of`] and [`mega_mhc_layer`] are the original no-bias, softmax-reading path;
//! [`h_of_with_bias`] and [`mega_mhc_layer_with_bias`] take a checkpoint vector, and an all-zero
//! vector reproduces the no-bias path bit for bit (a test asserts that for both the coefficients and
//! the residual streams). [`h_of_with_scale`] and [`mega_mhc_layer_with_scale`] additionally take
//! `hc_scale` and use the released sigmoid activation: with `hc_scale = [1, 1, 1]` and a zero base
//! they still differ from the no-bias path in A and C by construction, and that difference IS the
//! semantic change.
//!
//! The device side now carries the same tensors and the same arithmetic: kernels/mega_mhc_sm121
//! takes `hc_scale` [3] and `hc_base` [24] in this reference's row order, applies the projection
//! rstd, the group scale and the bias, the sigmoid pre/post and the released split-Sinkhorn, and
//! publishes the 26 accumulated statistics (the 24 projection rows plus the two RMS statistics) so
//! the host can compare the whole coefficient chain raw. `examples/m7_layer0/main.rs` permutes the
//! released rows into kernel order with [`released_predictor_into_kernel_rows`] (no averaging),
//! passes the bias through [`bias_into_kernel_rows`], passes the scale itself, and no longer
//! pre-folds it into the predictor (doing both would apply it twice). With the full released
//! projection on both sides, a device-vs-reference comparison at stage 1 is now a comparison of the
//! released arithmetic (modulo per-tile reduction order), not of a folded approximation.

use crate::m4_quant::{QUANT_BLOCK, act_quant_scale_code, fp8_e4m3_encode};

/// The hidden width. Mirrors SPARK_M4_HIDDEN.
pub const HIDDEN: usize = 5120;
/// The number of residual streams. Mirrors SPARK_M4_HC_MULT.
pub const HC_MULT: usize = 4;
/// Sinkhorn-Knopp iterations. Mirrors SPARK_M4_HC_SINKHORN_ITERS.
pub const SINKHORN_ITERS: usize = 20;
/// The Sinkhorn floor. Mirrors SPARK_M4_HC_EPS: the epsilon added after the row softmax and to
/// every later denominator, and the epsilon `pre` carries (inference/kernel.py:427, 443-458).
pub const SINKHORN_EPS: f32 = 1e-6;
/// The residual-stream RMS epsilon of the released model: `norm_eps = 1e-20` in the released
/// `config.json`, used by `hc_mixes`'s `rsqrt(mean(x^2) + norm_eps)` and by the RMSNorm
/// pre-norm over the collapsed block input (model.py:953-954; D4 in
/// docs/architecture/02-mhc.md). It is NOT `hc_eps`: this reference used the Sinkhorn floor in
/// both denominators, which is a defect the released entry points close.
pub const NORM_EPS: f32 = 1e-20;
/// The predictor rows H feeds on: n*n for B, 2n for A and C. Mirrors SPARK_M4_HC_PREDICTOR_ROWS.
pub const PREDICTOR_ROWS: usize = HC_MULT * HC_MULT + 2 * HC_MULT;

/// Row offset of the `pre` group (the input mix, this reference's A) in the RELEASED checkpoint's
/// 24-row layout (inference/kernel.py:426-431).
pub const PRE_ROWS_OFFSET: usize = 0;
/// Row offset of the `post` group (the block-output weight, this reference's C) in the released
/// layout.
pub const POST_ROWS_OFFSET: usize = HC_MULT;
/// Row offset of the `comb` group (the 4x4 residual mixer, this reference's B, transposed) in the
/// released layout.
pub const COMB_ROWS_OFFSET: usize = 2 * HC_MULT;

/// The number of per-group scales one predictor carries: pre, post, comb.
pub const HC_SCALE_LEN: usize = 3;
/// Position of the `pre` group's scalar in the checkpoint's `hc_scale`, the group this reference's
/// A comes from (inference/kernel.py:427).
pub const PRE_SCALE_INDEX: usize = 0;
/// Position of the `post` group's scalar, the group this reference's C comes from
/// (inference/kernel.py:429).
pub const POST_SCALE_INDEX: usize = 1;
/// Position of the `comb` group's scalar, the group this reference's B comes from
/// (inference/kernel.py:431).
pub const COMB_SCALE_INDEX: usize = 2;

/// The 24 per-coefficient biases of one predictor (`hc_attn_base` or `hc_ffn_base`), as the
/// checkpoint ships them: pre(0..n) / post(n..2n) / comb(2n..2n+n*n), one f32 per H row.
pub type HcBase = [f32; PREDICTOR_ROWS];

/// The three per-group scales of one predictor (`hc_attn_scale` or `hc_ffn_scale`), as the
/// checkpoint ships them: `[pre, post, comb]`, one f32 per coefficient GROUP (model.py:945-946,
/// inference/kernel.py:426-431). Not one per row; [`scale_into_kernel_rows`] expands it onto the
/// 24 rows of this reference's H order.
pub type HcScale = [f32; HC_SCALE_LEN];

/// A buffer whose writes are counted by its own accessor, so a write cannot be added without
/// moving the ledger. Nothing in this reference pushes into a bare Vec and then reports a count it
/// worked out by hand.
#[derive(Debug, Clone)]
pub struct Counted<T> {
    data: Vec<T>,
    writes: u64,
}

impl<T> Counted<T> {
    /// An empty counted buffer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            writes: 0,
        }
    }

    /// Append one element, counted.
    pub fn push(&mut self, value: T) {
        self.data.push(value);
        self.writes += 1;
    }

    /// How many elements have been written through this buffer.
    #[must_use]
    pub fn writes(&self) -> u64 {
        self.writes
    }

    /// The written elements.
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// How many elements the buffer holds, which equals writes().
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl<T> Default for Counted<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// The four residual streams X_l, owned here and reachable only through the counting read accessor.
///
/// This is what makes the traffic claim a measurement rather than a claim about it. The data is
/// private, so a second full pass over X_l cannot be written without going through the counter: an
/// earlier version of this reference took a plain slice, and a reviewer replaced the register copy
/// with a fresh slice read without moving a single number, which is the failure this type removes.
/// Two numbers follow from the counter itself: reads() is the total, and passes() is how many full
/// d-wide traversals it amounts to, computed rather than assigned.
#[derive(Debug, Clone)]
pub struct Streams {
    data: Vec<f32>,
    reads: u64,
}

impl Streams {
    /// Wrap the four streams, stream-major: stream * HIDDEN + channel.
    ///
    /// # Panics
    /// Panics if the buffer is not exactly n * d elements: a short buffer would make the pass
    /// counter meaningless.
    #[must_use]
    pub fn new(data: Vec<f32>) -> Self {
        assert_eq!(
            data.len(),
            HC_MULT * HIDDEN,
            "the streams must be n by d, read from a buffer of exactly that size"
        );
        Self { data, reads: 0 }
    }

    /// One element of one stream, counted.
    ///
    /// # Panics
    /// Panics if the index is out of range for the declared shape.
    pub fn read(&mut self, stream: usize, channel: usize) -> f32 {
        assert!(stream < HC_MULT, "the stream index is out of range");
        assert!(channel < HIDDEN, "the channel index is out of range");
        self.reads += 1;
        self.data[stream * HIDDEN + channel]
    }

    /// Every element of X_l ever read out of memory.
    #[must_use]
    pub fn reads(&self) -> u64 {
        self.reads
    }

    /// How many full d-wide traversals of X_l the counted reads amount to, ROUNDED UP so that even a
    /// partial second pass reports two. Derived from the counter, never stored: there is no field a
    /// caller could leave at one.
    #[must_use]
    pub fn passes(&self) -> u64 {
        self.reads.div_ceil((HC_MULT * HIDDEN) as u64)
    }

    /// The single stream-major pass that fills the register copy, which is the only memory read of
    /// X_l this reference performs.
    pub fn capture(&mut self) -> Registers {
        let mut data = Vec::with_capacity(HC_MULT * HIDDEN);
        for stream in 0..HC_MULT {
            for channel in 0..HIDDEN {
                data.push(self.read(stream, channel));
            }
        }
        Registers { data, reads: 0 }
    }
}

/// X_l in registers, as the kernel holds it. Reads through this type are counted separately and are
/// deliberately NOT memory traffic: registers are the currency the single pass buys, and keeping the
/// two counters apart is what lets a test show the reuse is real instead of asserting it.
#[derive(Debug, Clone)]
pub struct Registers {
    data: Vec<f32>,
    reads: u64,
}

impl Registers {
    /// One element of one stream out of the register copy, counted as a register read.
    ///
    /// # Panics
    /// Panics if the index is out of range for the declared shape.
    pub fn read(&mut self, stream: usize, channel: usize) -> f32 {
        assert!(stream < HC_MULT, "the stream index is out of range");
        assert!(channel < HIDDEN, "the channel index is out of range");
        self.reads += 1;
        self.data[stream * HIDDEN + channel]
    }

    /// How many elements were served out of registers.
    #[must_use]
    pub fn reads(&self) -> u64 {
        self.reads
    }

    /// The empty register file: what a multi-pass variant keeps, which is exactly what makes its
    /// traffic worse.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            data: Vec::new(),
            reads: 0,
        }
    }
}

/// What one Mega-mHC layer touched, in elements. Every field is copied from a counted buffer's own
/// accessor by observe() at the end of the layer, so none of them can be set from a literal, and
/// passes() is derived from x_read rather than stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TrafficLedger {
    /// Elements of X_l read out of memory. The kernel's claim is that this is exactly n * d.
    pub x_read: u64,
    /// Elements of X_{l+1} written.
    pub x_write: u64,
    /// Elements of the quantised block input written (one per element of one stream).
    pub xhat_write: u64,
    /// ue8m0 scale bytes written, one per 32-element block of the quantised input.
    pub scale_write: u64,
    /// Elements the reference served out of its register copy rather than by reading a stream
    /// again. Reported for the readers of a diff and deliberately NOT part of total(), because
    /// registers are not memory traffic.
    pub register_reuse: u64,
}

impl TrafficLedger {
    /// Copy every field from the counted buffers the layer actually used. This is the only writer of
    /// these fields.
    pub fn observe(
        &mut self,
        streams: &Streams,
        registers: &Registers,
        next: &Counted<f32>,
        codes: &Counted<u8>,
        scales: &Counted<u8>,
    ) {
        self.x_read = streams.reads();
        self.register_reuse = registers.reads();
        self.x_write = next.writes();
        self.xhat_write = codes.writes();
        self.scale_write = scales.writes();
    }

    /// The total traffic in elements.
    #[must_use]
    pub fn total(&self) -> u64 {
        self.x_read + self.x_write + self.xhat_write + self.scale_write
    }

    /// The stream traffic the specification's bound is stated over: reading X_l and writing X_{l+1}.
    #[must_use]
    pub fn stream_traffic(&self) -> u64 {
        self.x_read + self.x_write
    }

    /// Full traversals of X_l, derived from the counted reads. Nothing assigns this.
    #[must_use]
    pub fn passes(&self) -> u64 {
        self.x_read.div_ceil((HC_MULT * HIDDEN) as u64)
    }
}

/// The specification's own (2n+2)d bound, written as the specification writes it.
#[must_use]
pub fn spec_bound() -> u64 {
    ((2 * HC_MULT + 2) * HIDDEN) as u64
}

/// The (4n+4)d a multi-pass formulation costs, which is what the single pass is contrasted with.
#[must_use]
pub fn baseline_bound() -> u64 {
    ((4 * HC_MULT + 4) * HIDDEN) as u64
}

/// The coefficients of one layer: the 4x4 mixer B and the two 4-vectors A and C.
#[derive(Debug, Clone, PartialEq)]
pub struct Coefficients {
    /// By convention b\[i\]\[j\] mixes stream j into stream i.
    pub b: [[f32; HC_MULT]; HC_MULT],
    /// Scales the block input: X_hat = A .* X (elementwise, one factor per stream).
    pub a: [f32; HC_MULT],
    /// Weights the normalised block output back into the streams.
    pub c: [f32; HC_MULT],
}

/// The 24 raw predictor logits: one dot product per row against the FLATTENED `n*d` stream
/// (released `F.linear(x.flatten(2), hc_fn)`, model.py:951-954).
///
/// Raw is the point: the released model multiplies these numbers by hc_scale, adds hc_base and only
/// then applies its non-linearities, so the bias and scale paths ([`h_of_with_bias`],
/// [`h_of_with_scale`]) need them unactivated and unscaled. The projection's
/// `rsqrt(mean(x^2) + norm_eps)` factor is applied by [`h_of_with_scale`], not here.
///
/// The predictor is the released `[PREDICTOR_ROWS][HC_MULT * HIDDEN]` matrix in kernel row order,
/// produced from the checkpoint by [`released_predictor_into_kernel_rows`]. The accumulation order
/// is the flattened stream order (copy by copy, then channel), which is the engine reference's
/// `F.linear` order.
///
/// # Panics
/// Panics if the predictor is not the released (n*n + 2n) by (n*d) matrix.
fn predictor_rows(registers: &mut Registers, predictor: &[f32]) -> [f32; PREDICTOR_ROWS] {
    assert_eq!(
        predictor.len(),
        PREDICTOR_ROWS * HC_MULT * HIDDEN,
        "the released predictor must be (n*n + 2n) by (n*d)"
    );
    // One dot product per predictor row over the flattened four-stream input, out of the register
    // copy: no stream is read twice, and the projection is NOT the folded stream mean.
    let mut rows = [0.0f32; PREDICTOR_ROWS];
    for row in 0..PREDICTOR_ROWS {
        let mut acc = 0.0f32;
        for stream in 0..HC_MULT {
            for i in 0..HIDDEN {
                acc += registers.read(stream, i)
                    * predictor[row * HC_MULT * HIDDEN + stream * HIDDEN + i];
            }
        }
        rows[row] = acc;
    }
    rows
}

/// How the four `pre` logits become the A vector and the four `post` logits the C vector.
///
/// The B path is shared by both variants, so the two readings cannot diverge in the part that is not
/// the A/C activation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrePostActivation {
    /// The released `hc_split_sinkhorn` (inference/kernel.py:427,429): `pre[j] =
    /// sigmoid(logit[j]) + hc_eps` and `post[j] = 2 * sigmoid(logit[j])`, independently for each
    /// entry, with no normalisation across the four copies.
    ReleasedSigmoid,
    /// This module's EARLIER reading: an independent softmax over the four `pre` logits and another
    /// over the four `post` logits, so each vector sums to one. This is NOT the released arithmetic;
    /// it is kept because [`h_of`], [`h_of_with_bias`], [`mega_mhc_layer`] and
    /// [`mega_mhc_layer_with_bias`] were built on it and their numbers are pinned by tests.
    Softmax,
}

/// This reference's split of 24 logits into B, A and C, under either reading of the A/C activation.
///
/// B follows the activation too, because the two readings were built at different times: the softmax
/// reading (the original entry points, whose numbers are pinned) uses this module's own
/// [`sinkhorn_knopp`] on exponentiated logits, while the released reading uses
/// [`released_comb`], which is the released softmax-then-alternate schedule in the released
/// orientation. A and C: [`PrePostActivation`].
#[must_use]
pub fn coefficients_from_rows_with(
    rows: &[f32; PREDICTOR_ROWS],
    activation: PrePostActivation,
) -> Coefficients {
    let b = match activation {
        // The original reading, unchanged: exponentiate and run the generic Sinkhorn-Knopp.
        PrePostActivation::Softmax => {
            let mut b = [[0.0f32; HC_MULT]; HC_MULT];
            for i in 0..HC_MULT {
                for j in 0..HC_MULT {
                    b[i][j] = rows[i * HC_MULT + j].exp();
                }
            }
            sinkhorn_knopp(&mut b, SINKHORN_ITERS, SINKHORN_EPS);
            b
        }
        PrePostActivation::ReleasedSigmoid => released_comb(rows),
    };

    let (a, c) = match activation {
        PrePostActivation::Softmax => (
            softmax4(&rows[HC_MULT * HC_MULT..HC_MULT * HC_MULT + HC_MULT]),
            softmax4(&rows[HC_MULT * HC_MULT + HC_MULT..PREDICTOR_ROWS]),
        ),
        PrePostActivation::ReleasedSigmoid => (
            released_pre(&rows[HC_MULT * HC_MULT..HC_MULT * HC_MULT + HC_MULT]),
            released_post(&rows[HC_MULT * HC_MULT + HC_MULT..PREDICTOR_ROWS]),
        ),
    };
    Coefficients { b, a, c }
}

/// This reference's EARLIER split, for the no-bias and bias-only entry points: the softmax reading of
/// A and C. Kept as one named function so it is obvious which reading a caller gets.
#[must_use]
fn coefficients_from_rows(rows: &[f32; PREDICTOR_ROWS]) -> Coefficients {
    coefficients_from_rows_with(rows, PrePostActivation::Softmax)
}

/// The logistic function, in f32.
#[must_use]
pub fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

/// The released `pre` activation: `sigmoid(logit) + hc_eps` per entry (inference/kernel.py:427). It
/// is not a distribution; the four entries are independent.
#[must_use]
fn released_pre(logits: &[f32]) -> [f32; HC_MULT] {
    assert_eq!(
        logits.len(),
        HC_MULT,
        "the pre activation wants exactly n logits"
    );
    core::array::from_fn(|j| sigmoid(logits[j]) + SINKHORN_EPS)
}

/// The released `post` activation: `2 * sigmoid(logit)` per entry, no epsilon
/// (inference/kernel.py:429).
#[must_use]
fn released_post(logits: &[f32]) -> [f32; HC_MULT] {
    assert_eq!(
        logits.len(),
        HC_MULT,
        "the post activation wants exactly n logits"
    );
    core::array::from_fn(|j| 2.0 * sigmoid(logits[j]))
}

/// H(X_l): the predictor rows applied to the flattened four-copy stream (post-B0; the retired fold
/// applied them to the stream mean), then exp + Sinkhorn into B, softmax into A and into C. The
/// predictor is the released (n*n + 2n) x (n*d) matrix in the kernel's B/A/C row order, row-major,
/// with all four step copies kept: it is NOT averaged over the streams.
///
/// This is the ORIGINAL reading and it is unchanged: no bias, no scale, and A and C are SOFTMAXED,
/// which is NOT what the released `hc_split_sinkhorn` does (inference/kernel.py:427-429 uses
/// `sigmoid` and `2*sigmoid`; see the module docs, "The released pre/post activation and
/// hc_scale"). Use [`h_of_with_bias`] when only the checkpoint bias is to be added, and
/// [`h_of_with_scale`] for the released activation and scale.
///
/// It takes the REGISTER copy, which is how the kernel computes it: the reads are counted as
/// register reads, which are not memory traffic. There is no signature of this function that
/// accepts the streams themselves, so forming H a second time from memory is not something a caller
/// can do by accident - and if someone does it deliberately through Streams::read, the pass counter
/// sees it.
///
/// The result is deterministic and depends only on the input, so the kernel can compute it on the
/// last tile while the other tiles wait - which is what the grid rendezvous in the kernel is for.
///
/// # Panics
/// Panics if the predictor is not the released (n*n + 2n) by (n*d) matrix.
#[must_use]
pub fn h_of(registers: &mut Registers, predictor: &[f32]) -> Coefficients {
    coefficients_from_rows(&predictor_rows(registers, predictor))
}

/// The released checkpoint's 24-value `hc_base` rearranged into this reference's H row order.
///
/// The checkpoint's layout is pre(0..n) / post(n..2n) / comb(2n..2n+n*n)
/// (inference/kernel.py:426-431). This reference's row order is B(0..n*n) / A(n*n..) / C(n*n+n..),
/// and its B is indexed `[destination][source]` while `comb` is indexed `[source][destination]`
/// (see the module docs), so the 16 comb values are TRANSPOSED on the way in. The 4 pre values land
/// on A unchanged and the 4 post values on C unchanged.
///
/// Adding the checkpoint vector index for index instead is NOT equivalent - a test pins both the
/// arrangement and that difference.
#[must_use]
pub fn bias_into_kernel_rows(hc_base: &HcBase) -> [f32; PREDICTOR_ROWS] {
    let mut kernel = [0.0f32; PREDICTOR_ROWS];
    for destination in 0..HC_MULT {
        for source in 0..HC_MULT {
            kernel[destination * HC_MULT + source] =
                hc_base[COMB_ROWS_OFFSET + source * HC_MULT + destination];
        }
    }
    for j in 0..HC_MULT {
        kernel[HC_MULT * HC_MULT + j] = hc_base[PRE_ROWS_OFFSET + j];
        kernel[HC_MULT * HC_MULT + HC_MULT + j] = hc_base[POST_ROWS_OFFSET + j];
    }
    kernel
}

/// The released checkpoint's three per-group `hc_scale` values expanded onto this reference's 24 H
/// rows.
///
/// The checkpoint indexes the scales by COEFFICIENT GROUP, not by row: `hc_scale[0]` multiplies all
/// four `pre` rows, `hc_scale[1]` all four `post` rows and `hc_scale[2]` all sixteen `comb`
/// rows (inference/kernel.py:427-431). This reference's row order is B(0..n*n) / A(n*n..) /
/// C(n*n+n..), so the comb scalar lands on the B rows, the pre scalar on the A rows and the post
/// scalar on the C rows. The scale MULTIPLIES the logit and `hc_base` is added afterwards, so a zero
/// scale leaves only the bias, never the other way round.
#[must_use]
pub fn scale_into_kernel_rows(hc_scale: &HcScale) -> [f32; PREDICTOR_ROWS] {
    let mut kernel = [0.0f32; PREDICTOR_ROWS];
    for destination in 0..HC_MULT {
        for source in 0..HC_MULT {
            kernel[destination * HC_MULT + source] = hc_scale[COMB_SCALE_INDEX];
        }
    }
    for j in 0..HC_MULT {
        kernel[HC_MULT * HC_MULT + j] = hc_scale[PRE_SCALE_INDEX];
        kernel[HC_MULT * HC_MULT + HC_MULT + j] = hc_scale[POST_SCALE_INDEX];
    }
    kernel
}

/// The released checkpoint row that kernel row `kernel_row` comes from, in the permutation of
/// [`released_predictor_into_kernel_rows`]. Kernel rows are B(0..n*n) / A(n*n..n*n+n) /
/// C(n*n+n..); released rows are pre(0..n) / post(n..2n) / comb(2n..2n+n*n), and the comb terms
/// transpose because this reference's B is `[destination][source]`.
fn released_row_into_kernel_row(kernel_row: usize) -> usize {
    let comb_rows = HC_MULT * HC_MULT;
    if kernel_row < comb_rows {
        let destination = kernel_row / HC_MULT;
        let source = kernel_row % HC_MULT;
        COMB_ROWS_OFFSET + source * HC_MULT + destination
    } else if kernel_row < comb_rows + HC_MULT {
        PRE_ROWS_OFFSET + (kernel_row - comb_rows)
    } else {
        POST_ROWS_OFFSET + (kernel_row - comb_rows - HC_MULT)
    }
}

/// The released `[24, n*d]` predictor permuted into this reference's `[24, n*d]` kernel row
/// order: NO averaging, every stream copy kept.
///
/// The release applies `hc_fn` to the flattened `n*d` stream (model.py:952-954), so the reference
/// and the kernel both project all four copies ([specification 2.4]'s "RMSNorm weights folded
/// offline into the predictor projection" describes the checkpoint's storage of the projection, not
/// a d-wide fold). B0 closed the old fold; this function is the ONLY row rearrangement left, and it
/// must not average: a fold here would silently divide the released projection by the copy count
/// again, which a test now pins.
///
/// The row ORDER is the checkpoint's: reversed from `pre(0..n) / post(n..2n) / comb(2n..)` into
/// kernel order, with the comb terms transposed. Permuting the rows without this mapping makes the
/// kernel apply one group's projection to another group's activation, and a device comparison
/// cannot catch it because both sides share the permutation - a test pins this mapping instead
/// (docs/archive/build/roadmap-to-serving.md section 10, open unknown 2).
///
/// # Panics
/// Panics if `released` is not `PREDICTOR_ROWS * HC_MULT * HIDDEN` long or `kernel_rows` is not the
/// same length.
pub fn released_predictor_into_kernel_rows(released: &[f32], kernel_rows: &mut [f32]) {
    assert_eq!(
        released.len(),
        PREDICTOR_ROWS * HC_MULT * HIDDEN,
        "the released predictor is 24 rows by n*d"
    );
    assert_eq!(
        kernel_rows.len(),
        PREDICTOR_ROWS * HC_MULT * HIDDEN,
        "the kernel-order predictor is also 24 rows by n*d: B0 removed the d-wide fold"
    );
    for kernel_row in 0..PREDICTOR_ROWS {
        let released_row = released_row_into_kernel_row(kernel_row);
        let out = kernel_row * HC_MULT * HIDDEN;
        let source = released_row * HC_MULT * HIDDEN;
        kernel_rows[out..out + HC_MULT * HIDDEN]
            .copy_from_slice(&released[source..source + HC_MULT * HIDDEN]);
    }
}

/// `row = row * scale + base`, in that order: the released formula is `mix * hc_scale[group] +
/// hc_base[row]`, so the scale multiplies and the bias is added (inference/kernel.py:427-431).
fn apply_scale_then_bias(rows: &mut [f32; PREDICTOR_ROWS], hc_base: &HcBase, hc_scale: &HcScale) {
    let scale = scale_into_kernel_rows(hc_scale);
    let bias = bias_into_kernel_rows(hc_base);
    for index in 0..PREDICTOR_ROWS {
        rows[index] = rows[index] * scale[index] + bias[index];
    }
}

/// H(X_l) with the released model's per-coefficient bias: the 24 raw logits each get their
/// `hc_base` entry added BEFORE the exp/Sinkhorn and the two softmaxes, which is where
/// `hc_split_sinkhorn` adds it (inference/kernel.py:426-431).
///
/// This is the BIAS-ONLY reading: no `hc_scale`, and A and C are softmaxed, not sigmoided, so it is
/// not the released activation. Use [`h_of_with_scale`] for that.
///
/// `hc_base` is in the checkpoint's own row order; [`bias_into_kernel_rows`] does the
/// rearrangement, and an all-zero vector reproduces [`h_of`] bit for bit.
///
/// # Panics
/// Panics if the predictor is not the released (n*n + 2n) by (n*d) matrix.
#[must_use]
pub fn h_of_with_bias(
    registers: &mut Registers,
    predictor: &[f32],
    hc_base: &HcBase,
) -> Coefficients {
    let mut rows = predictor_rows(registers, predictor);
    let bias = bias_into_kernel_rows(hc_base);
    for (row, offset) in rows.iter_mut().zip(bias.iter()) {
        *row += offset;
    }
    coefficients_from_rows(&rows)
}

/// The sum of squares of the four streams over the whole flattened `n*d` stream: the statistic the
/// released `hc_mixes` normalises its projection by (model.py:952-954). These are register reads,
/// not stream traffic: the kernel holds the streams in registers when it needs this.
fn stream_sum_squares(registers: &mut Registers) -> f32 {
    let mut sum = 0.0f32;
    for stream in 0..HC_MULT {
        for channel in 0..HIDDEN {
            let value = registers.read(stream, channel);
            sum += value * value;
        }
    }
    sum
}

/// The released coefficient rstd: `1/sqrt(mean(x^2) + norm_eps)` over the flattened `n*d`
/// stream (model.py:953-954). The mean is over `n*d`, not `d`: all four copies are dimensions of
/// the flattened projection input, so every element contributes. D4 in docs/architecture/02-mhc.md
/// records the constant.
#[must_use]
fn coefficient_rstd(sum_squares: f32) -> f32 {
    1.0 / (sum_squares / (HC_MULT * HIDDEN) as f32 + NORM_EPS).sqrt()
}

/// The f32 block input the kernel now captures through its B0 `xhat_normed` output: the collapsed
/// input `X_hat` times the RMS weight times `1/rms`, in the KERNEL's multiplication order
/// (`xhat * weight * inv_rms`). This is the value BEFORE the per-32 quantization whose codes and
/// scales are [`Counted`] outputs of the layer functions; a comparison can therefore separate the
/// RMSNorm arithmetic from the quantizer instead of reconstructing the f32 value as
/// `code * scale`.
///
/// It is a capture, not a change to the specification's write set, so it is not counted in
/// [`TrafficLedger`]: the kernel's write set grew by `d` f32 when the capture was added, and that
/// is reported in the kernel's payload meta rather than charged to the (2n+2)d stream bound.
///
/// # Panics
/// Panics if `xhat` or `rms_weight` is not `d` wide.
#[must_use]
pub fn block_input_normed(xhat: &[f32], rms_weight: &[f32], inv_rms: f32) -> Vec<f32> {
    assert_eq!(xhat.len(), HIDDEN, "X_hat is d wide");
    assert_eq!(rms_weight.len(), HIDDEN, "the RMS weight is d wide");
    xhat.iter()
        .zip(rms_weight.iter())
        .map(|(value, weight)| value * weight * inv_rms)
        .collect()
}

/// H(X_l) with ALL THREE released predictor tensors and the released arithmetic: the projection is
/// first multiplied by `rsqrt(mean(x^2) + norm_eps)` over the flattened `n*d` stream
/// (model.py:953-954), then each row is multiplied by its group's `hc_scale` entry and has
/// `hc_base` added AFTER the product, then the four `pre` logits become
/// `sigmoid(...) + hc_eps`, the four `post` logits `2 * sigmoid(...)`, and the sixteen `comb`
/// logits the released softmax-then-alternate schedule (inference/kernel.py:426-458).
///
/// This is the reading a released-model comparison wants, and it differs from [`h_of_with_bias`]
/// in THREE deliberate ways, each pinned by a test: the projection rstd is applied, the scale is
/// applied, A and C are sigmoids instead of softmaxes, and B uses the released schedule instead of
/// this module's [`sinkhorn_knopp`].
///
/// Consequently a zero `hc_base` with `hc_scale = [1, 1, 1]` does NOT reproduce [`h_of`]: A and
/// C differ by the activation, and B differs by the schedule.
///
/// # Panics
/// Panics if the predictor is not the released (n*n + 2n) by (n*d) matrix.
#[must_use]
pub fn h_of_with_scale(
    registers: &mut Registers,
    predictor: &[f32],
    hc_base: &HcBase,
    hc_scale: &HcScale,
) -> Coefficients {
    let mut rows = predictor_rows(registers, predictor);
    let rstd = coefficient_rstd(stream_sum_squares(registers));
    for row in rows.iter_mut() {
        *row *= rstd;
    }
    apply_scale_then_bias(&mut rows, hc_base, hc_scale);
    coefficients_from_rows_with(&rows, PrePostActivation::ReleasedSigmoid)
}

/// The released `comb` normalisation, exactly as `hc_split_sinkhorn` runs it
/// (inference/kernel.py:436-458):
///
/// 1. a row softmax over the LAST axis, then `+ eps` (one row pass that adds eps once);
/// 2. one unconditional column pass, `/(colsum + eps)`;
/// 3. `iterations - 1` pairs of `/(rowsum + eps)` and `/(colsum + eps)`.
///
/// The matrix is in the released `[source][destination]` orientation, so "row" is a destination
/// sum for one source, the last pass is a COLUMN pass, and each column ends summing to 1. This is
/// not [`sinkhorn_knopp`]: the softmax is the first row normalisation and carries no eps in its
/// own denominator, eps is added to the matrix once before the first column pass, and every later
/// denominator carries it. D3 in docs/architecture/02-mhc.md states the same schedule.
pub fn released_split_sinkhorn(
    matrix: &mut [[f32; HC_MULT]; HC_MULT],
    iterations: usize,
    eps: f32,
) {
    for row in matrix.iter_mut() {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for value in row.iter_mut() {
            *value = (*value - max).exp();
            sum += *value;
        }
        for value in row.iter_mut() {
            *value = *value / sum + eps;
        }
    }
    normalise_columns(matrix, eps);
    for _ in 1..iterations {
        normalise_rows(matrix, eps);
        normalise_columns(matrix, eps);
    }
}

/// One `/(rowsum + eps)` pass of the released schedule.
fn normalise_rows(matrix: &mut [[f32; HC_MULT]; HC_MULT], eps: f32) {
    for row in matrix.iter_mut() {
        let denominator: f32 = row.iter().sum::<f32>() + eps;
        for value in row.iter_mut() {
            *value /= denominator;
        }
    }
}

/// One `/(colsum + eps)` pass of the released schedule.
fn normalise_columns(matrix: &mut [[f32; HC_MULT]; HC_MULT], eps: f32) {
    for column in 0..HC_MULT {
        let denominator: f32 = (0..HC_MULT).map(|row| matrix[row][column]).sum::<f32>() + eps;
        for row in matrix.iter_mut() {
            row[column] /= denominator;
        }
    }
}

/// The released 16 comb rows in this reference's `B[destination][source]` order.
///
/// The rows on the way in are the kernel's order, `B[destination][source]`; the release computes
/// `comb` in `[source][destination]` order (D3 and the workspace notes in
/// docs/architecture/02-mhc.md), so the matrix is transposed into that orientation, normalised by
/// [`released_split_sinkhorn`], and transposed back.
fn released_comb(rows: &[f32; PREDICTOR_ROWS]) -> [[f32; HC_MULT]; HC_MULT] {
    let mut comb = [[0.0f32; HC_MULT]; HC_MULT];
    for destination in 0..HC_MULT {
        for source in 0..HC_MULT {
            comb[source][destination] = rows[destination * HC_MULT + source];
        }
    }
    released_split_sinkhorn(&mut comb, SINKHORN_ITERS, SINKHORN_EPS);
    let mut b = [[0.0f32; HC_MULT]; HC_MULT];
    for source in 0..HC_MULT {
        for destination in 0..HC_MULT {
            b[destination][source] = comb[source][destination];
        }
    }
    b
}

/// Sinkhorn-Knopp: alternate row and column normalisation, with the specification's floor so a row
/// of zeros cannot produce a division by zero.
pub fn sinkhorn_knopp(matrix: &mut [[f32; HC_MULT]; HC_MULT], iterations: usize, eps: f32) {
    for _ in 0..iterations {
        for row in matrix.iter_mut() {
            let sum: f32 = row.iter().sum();
            let scale = if sum > eps { 1.0 / sum } else { 1.0 };
            for value in row.iter_mut() {
                *value *= scale;
            }
        }
        for column in 0..HC_MULT {
            let sum: f32 = (0..HC_MULT).map(|row| matrix[row][column]).sum();
            let scale = if sum > eps { 1.0 / sum } else { 1.0 };
            for row in matrix.iter_mut() {
                row[column] *= scale;
            }
        }
    }
}

/// Softmax over four values, in f32, subtracting the maximum first.
#[must_use]
pub fn softmax4(values: &[f32]) -> [f32; HC_MULT] {
    assert_eq!(values.len(), HC_MULT, "softmax4 wants exactly n values");
    let max = values.iter().copied().fold(f32::MIN, f32::max);
    let mut out = [0.0f32; HC_MULT];
    let mut sum = 0.0f32;
    for (index, value) in values.iter().enumerate() {
        out[index] = (value - max).exp();
        sum += out[index];
    }
    for value in out.iter_mut() {
        *value /= sum;
    }
    out
}

/// One Mega-mHC layer, single pass, counting every access through the counted buffers.
///
/// X_hat is formed from the A the PREVIOUS layer produced (passed in as a_prev), which is the whole
/// reason this needs one read of X: the kernel still has A_{l-1} X_l in registers from the previous
/// update, and recomputing it would mean reading X again.
///
/// This is the ORIGINAL reading and it is unchanged: no bias, no scale, and A and C softmaxed, which
/// is NOT the released activation (see the module docs). [`mega_mhc_layer_with_bias`] is the same
/// layer with the checkpoint's hc_base added; [`mega_mhc_layer_with_scale`] is the same layer with
/// hc_base AND hc_scale and the released sigmoid activation.
///
/// # Panics
/// Panics if the RMS weight is not d wide, or the predictor is not the released `[24, n*d]`
/// matrix in kernel row order.
#[must_use]
pub fn mega_mhc_layer(
    streams: &mut Streams,
    rms_weight: &[f32],
    a_prev: &[f32; HC_MULT],
    predictor: &[f32],
    ledger: &mut TrafficLedger,
) -> (Counted<f32>, Coefficients, Counted<u8>, Counted<u8>) {
    mega_mhc_layer_inner(
        streams,
        rms_weight,
        a_prev,
        predictor,
        CoefficientTensors::None,
        ledger,
    )
}

/// The same layer with the released checkpoint's per-coefficient bias: the 24 values of hc_base are
/// added to the raw coefficient logits before the exp/Sinkhorn and the softmaxes
/// ([`h_of_with_bias`]).
///
/// This is the BIAS-ONLY reading: no hc_scale, and A and C softmaxed, so it is not the released
/// activation. An all-zero vector reproduces [`mega_mhc_layer`] bit for bit, which a test asserts.
///
/// # Panics
/// As [`mega_mhc_layer`].
#[must_use]
pub fn mega_mhc_layer_with_bias(
    streams: &mut Streams,
    rms_weight: &[f32],
    a_prev: &[f32; HC_MULT],
    predictor: &[f32],
    hc_base: &HcBase,
    ledger: &mut TrafficLedger,
) -> (Counted<f32>, Coefficients, Counted<u8>, Counted<u8>) {
    mega_mhc_layer_inner(
        streams,
        rms_weight,
        a_prev,
        predictor,
        CoefficientTensors::Bias(hc_base),
        ledger,
    )
}

/// The same layer with BOTH released coefficient tensors and the released activation: hc_scale
/// multiplies each logit, hc_base is added after the product, and A and C are sigmoids
/// ([`h_of_with_scale`]).
///
/// This is the layer to compare against the released model. It is the same single pass and the same
/// traffic - only the coefficients differ - so the ledger's claims are unchanged.
///
/// # Panics
/// As [`mega_mhc_layer`].
#[must_use]
pub fn mega_mhc_layer_with_scale(
    streams: &mut Streams,
    rms_weight: &[f32],
    a_prev: &[f32; HC_MULT],
    predictor: &[f32],
    hc_base: &HcBase,
    hc_scale: &HcScale,
    ledger: &mut TrafficLedger,
) -> (Counted<f32>, Coefficients, Counted<u8>, Counted<u8>) {
    mega_mhc_layer_inner(
        streams,
        rms_weight,
        a_prev,
        predictor,
        CoefficientTensors::Released {
            base: hc_base,
            scale: hc_scale,
        },
        ledger,
    )
}

/// Which coefficient tensors the shared single pass applies. This is what keeps the three public layer
/// entry points on ONE implementation: only the formation of the coefficients differs.
enum CoefficientTensors<'a> {
    /// No checkpoint coefficients: this module's original H.
    None,
    /// The checkpoint's hc_base only, with the softmax reading of A and C.
    Bias(&'a HcBase),
    /// The checkpoint's hc_base and hc_scale, with the released sigmoid activation.
    Released {
        /// The 24 per-row biases, as the checkpoint ships them.
        base: &'a HcBase,
        /// The three per-group scales, as the checkpoint ships them.
        scale: &'a HcScale,
    },
}

/// The single pass all public layer entry points share. `tensors` says which coefficient tensors to
/// apply, so the no-bias path calls exactly what it called before the bias and scale existed.
#[must_use]
fn mega_mhc_layer_inner(
    streams: &mut Streams,
    rms_weight: &[f32],
    a_prev: &[f32; HC_MULT],
    predictor: &[f32],
    tensors: CoefficientTensors<'_>,
    ledger: &mut TrafficLedger,
) -> (Counted<f32>, Coefficients, Counted<u8>, Counted<u8>) {
    assert_eq!(rms_weight.len(), HIDDEN, "the RMS weight is d wide");

    // THE one pass over the streams: capture() reads every element exactly once through the counted
    // accessor and hands back the register copy. Everything below this line uses registers, which
    // is the whole content of the single-pass claim, and nothing below this line can reach the
    // stream buffer except through the counting accessor.
    let mut registers = streams.capture();

    // X_hat from the previous layer's A, plus the TWO different sums of squares the released model
    // needs: the streams' over the flattened n*d input (for the coefficient projection's rsqrt,
    // model.py:953-954) and X_hat's over d (for the RMSNorm the block input goes through,
    // model.py:984/991). The kernel accumulates both on the way past the register copy, one at the
    // thread's element and one per stream copy, and reduces them in a fixed order.
    let mut xhat = vec![0.0f32; HIDDEN];
    let mut stream_sum_squares = 0.0f32;
    let mut xhat_sum_squares = 0.0f32;
    for (index, slot) in xhat.iter_mut().enumerate() {
        let mut value = 0.0f32;
        for (stream, weight) in a_prev.iter().enumerate() {
            let element = registers.read(stream, index);
            stream_sum_squares += element * element;
            value += weight * element;
        }
        xhat_sum_squares += value * value;
        *slot = value;
    }
    // The older entry points' block RMS is pinned (including its epsilon) and is not the released
    // statistic; only the released entry point gets the released one. The coefficient path's
    // h_of_with_scale recomputes its own n*d sum from the registers, so it stays independent of this
    // loop's traversal.
    let block_inv_rms = match tensors {
        CoefficientTensors::Released { .. } => {
            1.0 / (xhat_sum_squares / HIDDEN as f32 + NORM_EPS).sqrt()
        }
        _ => 1.0 / (stream_sum_squares / (HC_MULT * HIDDEN) as f32 + SINKHORN_EPS).sqrt(),
    };

    // H(X_l) and the B X_l term both come from that register copy, exactly as they do in the
    // kernel, so they cost no further stream traffic. h_of takes the registers and nothing else;
    // h_of_counted exists only for the deliberately wasteful variant below, which is what the bound
    // test contrasts against.
    let coefficients = match tensors {
        CoefficientTensors::None => h_of(&mut registers, predictor),
        CoefficientTensors::Bias(base) => h_of_with_bias(&mut registers, predictor, base),
        CoefficientTensors::Released { base, scale } => {
            h_of_with_scale(&mut registers, predictor, base, scale)
        }
    };

    // F: the RMS pre-norm, the per-32 ue8m0 scales and the FP8 codes. Writing those is d elements
    // plus d / 32 scale bytes, which is the second half of the bound. The f32 pre-normed value is
    // computed through [`block_input_normed`] in the KERNEL's multiplication order, so the codes and
    // scales are quantized from the same f32 value the kernel's B0 capture writes.
    let inv_rms = block_inv_rms;
    let normed = block_input_normed(&xhat, rms_weight, inv_rms);
    let mut codes: Counted<u8> = Counted::new();
    let mut scales: Counted<u8> = Counted::new();
    for block in 0..HIDDEN / QUANT_BLOCK {
        let base = block * QUANT_BLOCK;
        let mut amax = 0.0f32;
        for value in &normed[base..base + QUANT_BLOCK] {
            let magnitude = value.abs();
            amax = if magnitude > amax { magnitude } else { amax };
        }
        // X_hat^FP8 is an ACTIVATION, so its scale is the reference's act_quant rule: amax floored at
        // 1e-4, times 1/448 in f32, then the scale exponent rounded UP (kernel.py:36-37, 76-80;
        // engine quant::e8m0_encode_ceil). The checkpoint weight-grid rule
        // (ue8m0_from_amax = floor(log2(amax)) - emax) is deliberately a different rule and must not
        // be used here: on a block whose amax lies above 1.75 * 2^floor(log2 amax) it picks a scale
        // one power of two too small, and the E4M3 operand differs from the engine's.
        // act_quant_scale_code is total, so a block containing inf takes code 254 instead of
        // aborting this oracle (the D8 review's B1).
        let scale_code = act_quant_scale_code(amax);
        scales.push(scale_code);
        // MX encodes the value DIVIDED by the shared power-of-two scale; the decoder multiplies it
        // back. This is the same value/scale order the reference uses (kernel.py:89-93).
        let scale = crate::m4_quant::ue8m0_decode(scale_code);
        for value in &normed[base..base + QUANT_BLOCK] {
            codes.push(fp8_e4m3_encode(value / scale));
        }
    }

    // X_{l+1} = B X_l + C F(X_hat), where F is read back from the codes it just wrote and X_l
    // comes from the register copy.
    let mut next: Counted<f32> = Counted::new();
    for stream in 0..HC_MULT {
        for i in 0..HIDDEN {
            let mut value = 0.0f32;
            for source in 0..HC_MULT {
                value += coefficients.b[stream][source] * registers.read(source, i);
            }
            let block = i / QUANT_BLOCK;
            let dequantised = crate::m4_quant::fp8_e4m3_decode(codes.as_slice()[i])
                * crate::m4_quant::ue8m0_decode(scales.as_slice()[block]);
            value += coefficients.c[stream] * dequantised;
            next.push(value);
        }
    }

    // The ledger is filled from the counters, at the end, from what actually happened.
    ledger.observe(streams, &registers, &next, &codes, &scales);
    (next, coefficients, codes, scales)
}

/// The same H, but taken from MEMORY: every element goes through the stream accessor, so its reads
/// are counted as stream traffic. This is a second full pass in this reference (the kernel does it
/// from registers and charges nothing), and it is what makes the wasteful variant below wasteful.
fn h_of_counted(streams: &mut Streams, predictor: &[f32]) -> Coefficients {
    assert_eq!(predictor.len(), PREDICTOR_ROWS * HC_MULT * HIDDEN);
    let mut rows = [0.0f32; PREDICTOR_ROWS];
    for row in 0..PREDICTOR_ROWS {
        let mut acc = 0.0f32;
        for stream in 0..HC_MULT {
            for i in 0..HIDDEN {
                acc += streams.read(stream, i)
                    * predictor[row * HC_MULT * HIDDEN + stream * HIDDEN + i];
            }
        }
        rows[row] = acc;
    }
    let mut b = [[0.0f32; HC_MULT]; HC_MULT];
    for i in 0..HC_MULT {
        for j in 0..HC_MULT {
            b[i][j] = rows[i * HC_MULT + j].exp();
        }
    }
    sinkhorn_knopp(&mut b, SINKHORN_ITERS, SINKHORN_EPS);
    let a = softmax4(&rows[HC_MULT * HC_MULT..HC_MULT * HC_MULT + HC_MULT]);
    let c = softmax4(&rows[HC_MULT * HC_MULT + HC_MULT..PREDICTOR_ROWS]);
    Coefficients { b, a, c }
}

/// A deliberately wasteful three-pass formulation of exactly the same mathematics: pass one reads X
/// to form A X, pass two reads X again to form H(X), pass three reads X a third time for B X. It
/// exists so the traffic bound has a counter-example inside the test suite.
///
/// # Panics
/// Panics if the slices are shorter than the shapes require.
#[must_use]
pub fn mega_mhc_layer_three_pass(
    streams: &mut Streams,
    rms_weight: &[f32],
    predictor: &[f32],
    ledger: &mut TrafficLedger,
) -> (Counted<f32>, Coefficients) {
    assert_eq!(rms_weight.len(), HIDDEN);

    // Pass 1: A .* X, recomputing A from this very X (the multi-pass mistake).
    let mut a_pass = 0.0f32;
    for i in 0..HIDDEN {
        for stream in 0..HC_MULT {
            a_pass += streams.read(stream, i);
        }
    }
    let a_prev = softmax4(&[a_pass; HC_MULT]);

    let mut xhat = vec![0.0f32; HIDDEN];
    for (index, slot) in xhat.iter_mut().enumerate() {
        let mut value = 0.0f32;
        for (stream, weight) in a_prev.iter().enumerate() {
            value += weight * streams.read(stream, index);
        }
        *slot = value;
    }
    let _ = xhat;

    // Pass 2: H(X) over the streams again, from memory rather than from registers.
    let coefficients = h_of_counted(streams, predictor);

    // Pass 3: B X, reading the streams a third time.
    let mut next: Counted<f32> = Counted::new();
    for stream in 0..HC_MULT {
        for i in 0..HIDDEN {
            let mut value = 0.0f32;
            for source in 0..HC_MULT {
                value += coefficients.b[stream][source] * streams.read(source, i);
            }
            next.push(value);
        }
    }
    // A variant that keeps no register copy at all, which is exactly what makes its traffic worse.
    ledger.observe(
        streams,
        &Registers::empty(),
        &next,
        &Counted::new(),
        &Counted::new(),
    );
    (next, coefficients)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::m4_quant::fp8_e4m3_decode;

    /// The zeros predictor: every predictor row is zero, so H is the uniform case and the golden
    /// numbers below can be worked out on paper.
    fn zero_predictor() -> Vec<f32> {
        vec![0.0f32; PREDICTOR_ROWS * HC_MULT * HIDDEN]
    }

    #[test]
    fn golden_zero_predictor_is_hand_computable() {
        // With a zero predictor every row of H is zero, so B = exp(0) = 1 everywhere and Sinkhorn
        // on the all-ones matrix leaves 1/n = 0.25; A and C are softmax(0) = 0.25 each.
        //
        // Take X_l = 1 in every stream and a_prev = 0.25, so X_hat = 0.25 * 4 * 1 = 1. The RMS
        // denominator is sqrt(1 + eps), whose deviation from 1 is a factor 1 - 5e-7, and with an RMS
        // weight of 3 the normalised value is 2.9999985 - not on a power of two and not near a
        // rounding boundary, so the activation scale rule picks 2^-7 (ue8m0 code 120: the smallest
        // power of two at or above 3/448) and the encoded element is 3 / 2^-7 = 384, which is E4M3
        // 0x7C. Read back, that is 384 * 2^-7 = 3, and X_{l+1} = B X + C F = 4 * 0.25 * 1 + 0.25 * 3
        // = 1.75.
        let x = vec![1.0f32; HC_MULT * HIDDEN];
        let rms_weight = vec![3.0f32; HIDDEN];
        let mut ledger = TrafficLedger::default();
        let mut streams = Streams::new(x.clone());
        let (next, coefficients, codes, scales) = mega_mhc_layer(
            &mut streams,
            &rms_weight,
            &[0.25; HC_MULT],
            &zero_predictor(),
            &mut ledger,
        );

        for row in 0..HC_MULT {
            for column in 0..HC_MULT {
                assert!(
                    (coefficients.b[row][column] - 0.25).abs() < 1e-6,
                    "a zero predictor gives a uniform mixer, got {}",
                    coefficients.b[row][column]
                );
            }
        }
        for stream in 0..HC_MULT {
            assert!((coefficients.a[stream] - 0.25).abs() < 1e-6);
            assert!((coefficients.c[stream] - 0.25).abs() < 1e-6);
        }
        // F reads back as decode(0x7C) * scale = 384 * 2^-7 = 3.0, so X_{l+1} = 1 + 0.25 * 3 = 1.75.
        assert_eq!(
            codes.as_slice()[0],
            0x7C,
            "2.9999985 / 2^-7 = 384 encodes as 0x7C"
        );
        assert_eq!(
            scales.as_slice()[0],
            120,
            "3.0 with an element emax of 8 is a 2^-7 scale"
        );
        assert_eq!(scales.len(), HIDDEN / QUANT_BLOCK);
        let expected = 1.75f32;
        for stream in 0..HC_MULT {
            assert!(
                (next.as_slice()[stream * HIDDEN] - expected).abs() < 1e-6,
                "stream {stream} gave {}, expected {expected}",
                next.as_slice()[stream * HIDDEN]
            );
        }

        // Determinism: the kernel computes H on one tile and lets the others read the result, which
        // is only sound if H does not depend on who computes it.
        let (again, again_coefficients, _, _) = mega_mhc_layer(
            &mut Streams::new(x.clone()),
            &rms_weight,
            &[0.25; HC_MULT],
            &zero_predictor(),
            &mut TrafficLedger::default(),
        );
        assert_eq!(coefficients, again_coefficients);
        assert_eq!(
            next.as_slice()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            again
                .as_slice()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "the update must be bit-reproducible"
        );
    }

    #[test]
    fn the_single_pass_ledger_stays_inside_the_specification_bound() {
        // The claim of specification 2.4: (2n+2)d elements instead of (4n+4)d, because X_l is read
        // once. The ledger is incremented by the reference itself, so these are counts and not a
        // derivation.
        let d = HIDDEN as u64;
        let n = HC_MULT as u64;
        let rms_weight = vec![1.0f32; HIDDEN];
        let mut ledger = TrafficLedger::default();
        let mut streams = Streams::new(vec![0.5f32; HC_MULT * HIDDEN]);
        let _ = mega_mhc_layer(
            &mut streams,
            &rms_weight,
            &[0.25; HC_MULT],
            &zero_predictor(),
            &mut ledger,
        );

        // passes() is derived from the counted reads, not stored, and x_read is copied from the
        // stream buffer's own counter, so these are measurements of the traversal that happened.
        assert_eq!(ledger.passes(), 1, "X_l must be walked exactly once");
        assert_eq!(streams.passes(), 1, "and the buffer agrees with the ledger");
        assert_eq!(
            ledger.x_read,
            n * d,
            "X_l is read once, element for element"
        );
        assert_eq!(ledger.x_read, streams.reads());
        assert_eq!(ledger.x_write, n * d, "X_{{l+1}} is written once");
        assert_eq!(ledger.xhat_write, d, "the block input is one stream wide");
        assert_eq!(ledger.scale_write, d / QUANT_BLOCK as u64);
        assert_eq!(ledger.stream_traffic(), 2 * n * d);
        // The specification's bound, written as the specification writes it, and the exact
        // decomposition of what this reference touches.
        assert_eq!(spec_bound(), (2 * n + 2) * d);
        assert_eq!(baseline_bound(), (4 * n + 4) * d);
        assert_eq!(
            ledger.total(),
            ledger.stream_traffic() + ledger.xhat_write + ledger.scale_write,
            "the total must be the sum of the counted parts"
        );
        assert_eq!(ledger.total(), 2 * n * d + d + d / QUANT_BLOCK as u64);
        assert!(
            ledger.total() <= spec_bound(),
            "total {} exceeds the (2n+2)d bound {}",
            ledger.total(),
            spec_bound()
        );
        assert!(
            ledger.register_reuse >= n * d,
            "the register copy must actually be used, or the single pass is fake"
        );
    }

    #[test]
    fn a_second_pass_over_the_streams_cannot_evade_the_ledger() {
        // The reviewer's counter-example against the first version of this ledger: form H from the
        // STREAMS rather than from the register copy, which is a genuine extra read of X_l. That
        // version assigned passes = 1 by hand and counted register reuse with +=, so the bound test
        // still passed. Here the only way to reach the stream data is the counting accessor, so the
        // same mutation moves the numbers and fails both this test and the bound test.
        let mut streams = Streams::new(vec![0.5f32; HC_MULT * HIDDEN]);
        let mut registers = streams.capture();
        let _ = h_of(&mut registers, &zero_predictor());
        let honest = streams.reads();
        assert_eq!(honest, (HC_MULT * HIDDEN) as u64);
        assert_eq!(streams.passes(), 1);

        // The mutation, written the only way it can be written now.
        let _ = h_of_counted(&mut streams, &zero_predictor());
        assert!(
            streams.passes() >= 2,
            "a second pass over X_l must be visible in the ledger, got {}",
            streams.passes()
        );
        assert!(streams.reads() > honest);
        // A partial second pass counts as a pass too, so a small peek cannot hide under the
        // rounding.
        let before = streams.passes();
        let _ = streams.read(0, 0);
        assert!(streams.passes() >= before);
        assert_eq!(
            streams.reads(),
            honest + (PREDICTOR_ROWS * HC_MULT * HIDDEN) as u64 + 1
        );
        // And the layer's ledger is the buffer's counter, not a number of its own.
        let mut streams = Streams::new(vec![0.5f32; HC_MULT * HIDDEN]);
        let mut ledger = TrafficLedger::default();
        let _ = mega_mhc_layer(
            &mut streams,
            &vec![1.0f32; HIDDEN],
            &[0.25; HC_MULT],
            &zero_predictor(),
            &mut ledger,
        );
        assert_eq!(
            ledger.x_read,
            streams.reads(),
            "the ledger must report the accessor's count"
        );
        assert_eq!(ledger.x_read, (HC_MULT * HIDDEN) as u64);
    }

    #[test]
    fn the_three_pass_variant_breaks_the_bound() {
        // Non-vacuity: the same mathematics, done the multi-pass way, must exceed both this
        // reference's traffic and the specification's own baseline. If a broken implementation also
        // satisfied the bound, the bound would prove nothing.
        let rms_weight = vec![1.0f32; HIDDEN];
        let predictor = zero_predictor();

        let mut single = TrafficLedger::default();
        let _ = mega_mhc_layer(
            &mut Streams::new(vec![0.5f32; HC_MULT * HIDDEN]),
            &rms_weight,
            &[0.25; HC_MULT],
            &predictor,
            &mut single,
        );
        let mut three = TrafficLedger::default();
        let _ = mega_mhc_layer_three_pass(
            &mut Streams::new(vec![0.5f32; HC_MULT * HIDDEN]),
            &rms_weight,
            &predictor,
            &mut three,
        );

        assert_eq!(single.passes(), 1);
        assert!(
            three.passes() >= 3,
            "the wasteful variant walks the streams at least three times, got {}",
            three.passes()
        );
        assert!(
            three.total() > single.total(),
            "the wasteful variant costs {} and the single pass {}",
            three.total(),
            single.total()
        );
        assert!(
            three.total() > baseline_bound(),
            "the wasteful variant must also exceed the (4n+4)d baseline it exists to contrast with"
        );
    }

    #[test]
    fn sinkhorn_is_doubly_stochastic_and_the_iteration_count_matters() {
        let base = [
            [4.0f32, 1.0, 0.5, 2.0],
            [1.0, 3.0, 0.25, 0.5],
            [0.5, 0.25, 2.0, 1.0],
            [2.0, 0.5, 1.0, 3.0],
        ];
        let mut normalised = base;
        sinkhorn_knopp(&mut normalised, SINKHORN_ITERS, SINKHORN_EPS);
        for (row, values) in normalised.iter().enumerate() {
            let row_sum: f32 = values.iter().sum();
            assert!((row_sum - 1.0).abs() < 1e-4, "row {row} sums to {row_sum}");
        }
        for column in 0..HC_MULT {
            let column_sum: f32 = normalised.iter().map(|values| values[column]).sum();
            assert!(
                (column_sum - 1.0).abs() < 1e-4,
                "column {column} sums to {column_sum}"
            );
        }
        // ...and the iteration count is load-bearing: without it the matrix is not normalised.
        let mut uniterated = base;
        sinkhorn_knopp(&mut uniterated, 0, SINKHORN_EPS);
        assert_eq!(
            uniterated, base,
            "zero iterations must leave the matrix alone"
        );
        let row_sum: f32 = uniterated[0].iter().sum();
        assert!(
            (row_sum - 1.0).abs() > 1e-3,
            "the raw matrix is not stochastic"
        );
    }

    #[test]
    fn softmax_is_normalised_and_translation_invariant() {
        let values = [1.5f32, -2.0, 0.25, 8.0];
        let soft = softmax4(&values);
        let sum: f32 = soft.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(soft.iter().all(|value| *value >= 0.0 && *value <= 1.0));
        // Translation invariance: adding a constant must not change the result, which is why the
        // kernel can subtract the maximum for stability.
        let shifted: Vec<f32> = values.iter().map(|value| value + 1000.0).collect();
        let shifted_soft = softmax4(&shifted);
        for index in 0..HC_MULT {
            assert!(
                (soft[index] - shifted_soft[index]).abs() < 1e-6,
                "coordinate {index} moved when the input was translated"
            );
        }
        // The largest input must get the largest weight.
        assert_eq!(soft.iter().cloned().fold(0.0f32, f32::max), soft[3]);
    }

    #[test]
    fn the_quantised_block_input_round_trips_within_one_code_step() {
        // F is read back from the codes the kernel wrote, so the reference's own dequantisation has
        // to be within one E4M3 step of the value it quantised - otherwise the F it applies would
        // not be the F the kernel applies.
        let x: Vec<f32> = (0..HC_MULT * HIDDEN)
            .map(|i| ((i % 17) as f32 - 8.0) / 8.0)
            .collect();
        let rms_weight = vec![1.0f32; HIDDEN];
        let mut ledger = TrafficLedger::default();
        let (_, _, codes, scales) = mega_mhc_layer(
            &mut Streams::new(x.clone()),
            &rms_weight,
            &[0.25; HC_MULT],
            &zero_predictor(),
            &mut ledger,
        );

        let sum_squares: f32 = x.iter().map(|value| value * value).sum();
        let inv_rms = 1.0f32 / (sum_squares / (HC_MULT * HIDDEN) as f32 + SINKHORN_EPS).sqrt();
        for (block, scale_code) in scales.as_slice().iter().enumerate() {
            let scale = crate::m4_quant::ue8m0_decode(*scale_code);
            let mut amax = 0.0f32;
            for i in 0..QUANT_BLOCK {
                let index = block * QUANT_BLOCK + i;
                let mut xhat = 0.0f32;
                for stream in 0..HC_MULT {
                    xhat += 0.25 * x[stream * HIDDEN + index];
                }
                let normalised = xhat * inv_rms * rms_weight[index];
                amax = amax.max(normalised.abs());
                let back = fp8_e4m3_decode(codes.as_slice()[index]) * scale;
                let tolerance = (normalised.abs() * 0.0625).max(2.0f32.powi(-10)) + 1e-30;
                assert!(
                    (back - normalised).abs() <= tolerance,
                    "block {block} value {index} came back as {back} instead of {normalised}"
                );
            }
            // The scale must at least cover the block: no code may saturate because the scale was
            // chosen too small (the E4M3 maximum is 448).
            assert!(amax <= 448.0, "the fixture must not need saturation");
        }
    }

    /// A predictor whose rows are all different, so the coefficients are not the zero-predictor
    /// uniform case and a misplaced bias cannot hide inside a symmetry.
    fn deterministic_predictor() -> Vec<f32> {
        (0..PREDICTOR_ROWS * HC_MULT * HIDDEN)
            .map(|i| ((i % 19) as f32 - 9.0) / 32.0)
            .collect()
    }

    /// Four different streams, for the same reason.
    fn deterministic_streams() -> Vec<f32> {
        (0..HC_MULT * HIDDEN)
            .map(|i| ((i % 23) as f32 - 11.0) / 16.0)
            .collect()
    }

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    #[test]
    fn bias_into_kernel_rows_pins_the_checkpoint_layout() {
        // The checkpoint layout is pre(0..n) / post(n..2n) / comb(2n..): give every row its own value
        // (1..=24) and pin where each one has to arrive.
        let hc_base: HcBase = core::array::from_fn(|index| (index + 1) as f32);
        let kernel = bias_into_kernel_rows(&hc_base);

        // This reference's B is [destination][source] and the checkpoint's comb is [source][
        // destination], so the 4x4 block arrives transposed:
        // kernel[destination * n + source] = hc_base[2n + source * n + destination].
        assert_eq!(kernel[0..4], [9.0, 13.0, 17.0, 21.0], "B row 0");
        assert_eq!(kernel[4..8], [10.0, 14.0, 18.0, 22.0], "B row 1");
        assert_eq!(kernel[8..12], [11.0, 15.0, 19.0, 23.0], "B row 2");
        assert_eq!(kernel[12..16], [12.0, 16.0, 20.0, 24.0], "B row 3");
        assert_eq!(kernel[16..20], [1.0, 2.0, 3.0, 4.0], "pre lands on A");
        assert_eq!(kernel[20..24], [5.0, 6.0, 7.0, 8.0], "post lands on C");

        // Index-for-index is a different vector, which is the assumption this test removes.
        assert_ne!(kernel, hc_base, "the identity order is not the mapping");
    }

    #[test]
    fn an_all_zero_bias_reproduces_the_unbiased_path_bit_for_bit() {
        let predictor = deterministic_predictor();
        let rms_weight = vec![1.0f32; HIDDEN];
        let zero: HcBase = [0.0f32; PREDICTOR_ROWS];

        let mut registers = Streams::new(deterministic_streams()).capture();
        let unbiased = h_of(&mut registers, &predictor);
        let mut registers = Streams::new(deterministic_streams()).capture();
        let biased = h_of_with_bias(&mut registers, &predictor, &zero);
        assert_eq!(unbiased, biased, "a zero bias must not touch H");

        let (unbiased_next, unbiased_coefficients, ..) = mega_mhc_layer(
            &mut Streams::new(deterministic_streams()),
            &rms_weight,
            &[0.25; HC_MULT],
            &predictor,
            &mut TrafficLedger::default(),
        );
        let (biased_next, biased_coefficients, ..) = mega_mhc_layer_with_bias(
            &mut Streams::new(deterministic_streams()),
            &rms_weight,
            &[0.25; HC_MULT],
            &predictor,
            &zero,
            &mut TrafficLedger::default(),
        );
        assert_eq!(unbiased_coefficients, biased_coefficients);
        assert_eq!(
            bits(unbiased_next.as_slice()),
            bits(biased_next.as_slice()),
            "the no-bias path must be bit-identical with a zero bias"
        );
    }

    #[test]
    fn a_pre_bias_enters_the_a_logits_before_the_softmax() {
        // With a zero predictor every logit is zero, so the expected answer is a softmax over the
        // bias itself: [1, 0, 0, 0] gives e/(e+3) on the first A entry and 1/(e+3) on the rest.
        let predictor = zero_predictor();
        let mut hc_base: HcBase = [0.0f32; PREDICTOR_ROWS];
        hc_base[PRE_ROWS_OFFSET] = 1.0;
        let mut registers = Streams::new(vec![1.0f32; HC_MULT * HIDDEN]).capture();
        let coefficients = h_of_with_bias(&mut registers, &predictor, &hc_base);

        let e = 1.0f32.exp();
        let expected_first = e / (e + 3.0);
        let expected_rest = 1.0 / (e + 3.0);
        assert!(
            (coefficients.a[0] - expected_first).abs() < 1e-6,
            "A[0] is {}, expected {expected_first}",
            coefficients.a[0]
        );
        for stream in 1..HC_MULT {
            assert!(
                (coefficients.a[stream] - expected_rest).abs() < 1e-6,
                "A[{stream}] is {}, expected {expected_rest}",
                coefficients.a[stream]
            );
        }
        // A pre-activation logit, not an added coefficient: adding 1.0 after the softmax would put
        // 1.25 here.
        assert!(coefficients.a[0] < 1.0);

        // The other two coefficient groups are untouched. In particular B is still the uniform
        // mixer, so a vector added index for index (which would have landed on B[0][0]) is a
        // different computation.
        for row in 0..HC_MULT {
            for column in 0..HC_MULT {
                assert!(
                    (coefficients.b[row][column] - 0.25).abs() < 1e-6,
                    "B[{row}][{column}] moved, the pre bias belongs on A"
                );
            }
        }
        for stream in 0..HC_MULT {
            assert!((coefficients.c[stream] - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn a_post_bias_lands_on_the_c_vector() {
        let predictor = zero_predictor();
        let mut hc_base: HcBase = [0.0f32; PREDICTOR_ROWS];
        hc_base[POST_ROWS_OFFSET + 3] = 1.0;
        let mut registers = Streams::new(vec![1.0f32; HC_MULT * HIDDEN]).capture();
        let coefficients = h_of_with_bias(&mut registers, &predictor, &hc_base);

        let e = 1.0f32.exp();
        assert!(
            (coefficients.c[3] - e / (e + 3.0)).abs() < 1e-6,
            "C[3] is {}",
            coefficients.c[3]
        );
        for stream in 0..3 {
            assert!((coefficients.c[stream] - 1.0 / (e + 3.0)).abs() < 1e-6);
        }
        for row in 0..HC_MULT {
            for column in 0..HC_MULT {
                assert!((coefficients.b[row][column] - 0.25).abs() < 1e-6);
            }
        }
        for stream in 0..HC_MULT {
            assert!((coefficients.a[stream] - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn a_comb_bias_lands_on_the_sinkhorn_matrix_transposed_and_before_the_sinkhorn() {
        let predictor = zero_predictor();
        let source = 1usize;
        let destination = 2usize;
        let mut hc_base: HcBase = [0.0f32; PREDICTOR_ROWS];
        hc_base[COMB_ROWS_OFFSET + source * HC_MULT + destination] = 2.0;
        let mut registers = Streams::new(vec![1.0f32; HC_MULT * HIDDEN]).capture();
        let coefficients = h_of_with_bias(&mut registers, &predictor, &hc_base);

        // comb[source][destination] is B[destination][source] in this reference's convention, and
        // with every other logit zero it has to be the largest entry. An implementation that added
        // the vector index for index would put this 2.0 at B[destination + 1][destination] instead.
        let largest = coefficients
            .b
            .iter()
            .flatten()
            .copied()
            .fold(f32::MIN, f32::max);
        assert_eq!(largest, coefficients.b[destination][source]);
        assert!(
            coefficients.b[destination][source] > coefficients.b[source][destination] * 1.5,
            "the transpose matters: B[{destination}][{source}] = {}, B[{source}][{destination}] = {}",
            coefficients.b[destination][source],
            coefficients.b[source][destination]
        );
        assert!(coefficients.b[destination][source] > 1.0 / HC_MULT as f32);

        // Added BEFORE the Sinkhorn, so the result is still normalised in both directions. An
        // addition after the normalisation would leave row {destination} summing to about 3.
        for (row, values) in coefficients.b.iter().enumerate() {
            let sum: f32 = values.iter().sum();
            assert!((sum - 1.0).abs() < 1e-4, "row {row} sums to {sum}");
        }
        for column in 0..HC_MULT {
            let sum: f32 = coefficients.b.iter().map(|values| values[column]).sum();
            assert!((sum - 1.0).abs() < 1e-4, "column {column} sums to {sum}");
        }

        // A and C are untouched: the value landed on the 4x4 block only.
        for stream in 0..HC_MULT {
            assert!((coefficients.a[stream] - 0.25).abs() < 1e-6);
            assert!((coefficients.c[stream] - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn the_biased_layer_moves_the_residual_streams() {
        // The end-to-end non-vacuity check: with a realistic (non-uniform) predictor, a bias on each
        // of the three groups has to move B, A, C and the streams the layer writes. This is the test
        // that fails if the bias is quietly ignored.
        let predictor = deterministic_predictor();
        let rms_weight = vec![1.0f32; HIDDEN];
        let mut hc_base: HcBase = [0.0f32; PREDICTOR_ROWS];
        hc_base[PRE_ROWS_OFFSET] = 1.5;
        hc_base[POST_ROWS_OFFSET + 2] = -0.75;
        hc_base[COMB_ROWS_OFFSET + 3 * HC_MULT + 1] = 2.0;

        let (plain_next, plain_coefficients, ..) = mega_mhc_layer(
            &mut Streams::new(deterministic_streams()),
            &rms_weight,
            &[0.25; HC_MULT],
            &predictor,
            &mut TrafficLedger::default(),
        );
        let (biased_next, biased_coefficients, ..) = mega_mhc_layer_with_bias(
            &mut Streams::new(deterministic_streams()),
            &rms_weight,
            &[0.25; HC_MULT],
            &predictor,
            &hc_base,
            &mut TrafficLedger::default(),
        );

        assert_ne!(plain_coefficients.a, biased_coefficients.a, "A must move");
        assert_ne!(plain_coefficients.c, biased_coefficients.c, "C must move");
        assert_ne!(plain_coefficients.b, biased_coefficients.b, "B must move");
        let largest_delta = plain_next
            .as_slice()
            .iter()
            .zip(biased_next.as_slice())
            .map(|(plain, biased)| (plain - biased).abs())
            .fold(0.0f32, f32::max);
        assert!(
            largest_delta > 1e-3,
            "the bias must move the residual streams, largest delta {largest_delta}"
        );
    }

    #[test]
    fn scale_into_kernel_rows_pins_the_group_layout() {
        // One scalar per GROUP, not per row: hc_scale = [pre, post, comb] (model.py:945-946,
        // inference/kernel.py:427-431). Give each group a distinct value and pin where it lands in
        // this reference's B(0..n*n) / A(n*n..) / C(n*n+n..) row order.
        let hc_scale: HcScale = [1.0, 2.0, 3.0];
        let kernel = scale_into_kernel_rows(&hc_scale);
        // This reference's row order: B at 0, A at n*n, C at n*n + n.
        let a_offset = HC_MULT * HC_MULT;
        let c_offset = a_offset + HC_MULT;
        assert_eq!((a_offset, c_offset), (16, 20), "the row layout moved");
        for (row, value) in kernel.iter().enumerate().take(a_offset) {
            assert_eq!(*value, 3.0, "comb scales B, row {row}");
        }
        for (row, value) in kernel.iter().enumerate().skip(a_offset).take(HC_MULT) {
            assert_eq!(*value, 1.0, "pre scales A, row {row}");
        }
        for (row, value) in kernel.iter().enumerate().skip(c_offset) {
            assert_eq!(*value, 2.0, "post scales C, row {row}");
        }
        assert_eq!(kernel.len(), PREDICTOR_ROWS);
        // The indices are the checkpoint's group order, not this reference's row order.
        assert_eq!(
            (PRE_SCALE_INDEX, POST_SCALE_INDEX, COMB_SCALE_INDEX),
            (0, 1, 2)
        );
    }

    #[test]
    fn the_released_scale_multiplies_each_group_before_the_bias_is_added() {
        // inference/kernel.py:427-431 reads `mix * scale[group] + base` - multiply first, add
        // second. Each group gets a different scale and every row a different bias, so a missing
        // scale, a scale applied AFTER the bias, and a bias applied BEFORE the scale all land on
        // different numbers. The expected values are written out here with a locally-defined
        // logistic rather than taken from the module, so this test cannot agree with its builder by
        // construction.
        // The deterministic stream pattern (period 23) and the predictor pattern (period 19) cancel
        // exactly on kernel A row 19, which would make the missing-scale assertion below vacuous at
        // A[3]; one nonzero entry gives that row a real logit and leaves every other row untouched.
        let mut predictor = deterministic_predictor();
        predictor[(HC_MULT * HC_MULT + 3) * HC_MULT * HIDDEN + 1] = 1.0;
        let hc_scale: HcScale = [0.5, -2.0, 3.0];
        let hc_base: HcBase = core::array::from_fn(|index| (index as f32 - 12.0) / 16.0);
        let logistic = |z: f32| 1.0 / (1.0 + (-z).exp());

        // The raw logits, which are the input of the released formula, and the projection rstd the
        // release multiplies them by before scale and bias (model.py:953-954).
        let rstd = coefficient_rstd(stream_sum_squares(
            &mut Streams::new(deterministic_streams()).capture(),
        ));
        let rows = predictor_rows(
            &mut Streams::new(deterministic_streams()).capture(),
            &predictor,
        );
        let got = h_of_with_scale(
            &mut Streams::new(deterministic_streams()).capture(),
            &predictor,
            &hc_base,
            &hc_scale,
        );

        // A: sigmoid(mix * rstd * scale[0] + base) + hc_eps (kernel.py:427, model.py:953-954).
        assert!(
            (rstd - 1.0).abs() > 1e-3,
            "the fixture cannot see the projection rstd: {rstd}"
        );
        for j in 0..HC_MULT {
            let row = rows[HC_MULT * HC_MULT + j] * rstd;
            let base = hc_base[PRE_ROWS_OFFSET + j];
            let expected = logistic(row * hc_scale[PRE_SCALE_INDEX] + base) + SINKHORN_EPS;
            assert!(
                (got.a[j] - expected).abs() < 1e-6,
                "A[{j}] is {}, expected {expected}",
                got.a[j]
            );
            let missing_scale = logistic(row + base) + SINKHORN_EPS;
            let scale_after_bias =
                logistic((row + base) * hc_scale[PRE_SCALE_INDEX]) + SINKHORN_EPS;
            assert!(
                (expected - missing_scale).abs() > 1e-3,
                "this fixture cannot see a missing scale at A[{j}]"
            );
            assert!(
                (expected - scale_after_bias).abs() > 1e-3,
                "this fixture cannot tell multiply-then-add from add-then-multiply at A[{j}]"
            );
        }

        // C: 2 * sigmoid(mix * rstd * scale[1] + base), no epsilon (kernel.py:429).
        for j in 0..HC_MULT {
            let row = rows[HC_MULT * HC_MULT + HC_MULT + j] * rstd;
            let base = hc_base[POST_ROWS_OFFSET + j];
            let expected = 2.0 * logistic(row * hc_scale[POST_SCALE_INDEX] + base);
            assert!(
                (got.c[j] - expected).abs() < 1e-6,
                "C[{j}] is {}, expected {expected}",
                got.c[j]
            );
            let missing_scale = 2.0 * logistic(row + base);
            let scale_after_bias = 2.0 * logistic((row + base) * hc_scale[POST_SCALE_INDEX]);
            assert!(
                (expected - missing_scale).abs() > 1e-3,
                "this fixture cannot see a missing scale at C[{j}]"
            );
            assert!(
                (expected - scale_after_bias).abs() > 1e-3,
                "this fixture cannot tell multiply-then-add from add-then-multiply at C[{j}]"
            );
        }

        // B: the sixteen comb logits are `mix * rstd * scale[2] + base`, transposed into the
        // released [source][destination] orientation, normalised by the released split-Sinkhorn
        // schedule, then transposed back into this reference's [destination][source]. Build every
        // step from the released formula here and compare; then build the no-scale matrix and show
        // the fixture can tell them apart.
        let mut expected = [[0.0f32; HC_MULT]; HC_MULT];
        let mut unscaled = [[0.0f32; HC_MULT]; HC_MULT];
        for source in 0..HC_MULT {
            for destination in 0..HC_MULT {
                let checkpoint_row = COMB_ROWS_OFFSET + source * HC_MULT + destination;
                let mix = rows[destination * HC_MULT + source] * rstd;
                expected[source][destination] =
                    mix * hc_scale[COMB_SCALE_INDEX] + hc_base[checkpoint_row];
                unscaled[source][destination] = mix + hc_base[checkpoint_row];
            }
        }
        released_split_sinkhorn(&mut expected, SINKHORN_ITERS, SINKHORN_EPS);
        released_split_sinkhorn(&mut unscaled, SINKHORN_ITERS, SINKHORN_EPS);
        let mut expected_b = [[0.0f32; HC_MULT]; HC_MULT];
        let mut unscaled_b = [[0.0f32; HC_MULT]; HC_MULT];
        for source in 0..HC_MULT {
            for destination in 0..HC_MULT {
                expected_b[destination][source] = expected[source][destination];
                unscaled_b[destination][source] = unscaled[source][destination];
            }
        }
        assert_eq!(
            got.b, expected_b,
            "the comb rows must be scaled before the released split-Sinkhorn"
        );
        assert_ne!(
            expected_b, unscaled_b,
            "this fixture cannot see a missing comb scale"
        );
    }

    #[test]
    fn the_released_pre_and_post_activations_are_sigmoids_not_softmaxes() {
        // kernel.py:427 and 429: pre = sigmoid(...) + hc_eps and post = 2 * sigmoid(...),
        // independently per entry, so neither vector is a distribution. At zero logits the release
        // gives 0.5 + eps and 1.0, where this module's earlier softmax reading gives 0.25 and 0.25.
        let predictor = zero_predictor();
        let zero_base: HcBase = [0.0f32; PREDICTOR_ROWS];
        let hc_scale: HcScale = [1.0; HC_SCALE_LEN];
        let released = h_of_with_scale(
            &mut Streams::new(vec![1.0f32; HC_MULT * HIDDEN]).capture(),
            &predictor,
            &zero_base,
            &hc_scale,
        );
        for j in 0..HC_MULT {
            assert!(
                (released.a[j] - (0.5 + SINKHORN_EPS)).abs() < 1e-6,
                "A[{j}] is {}, expected 0.5 + eps",
                released.a[j]
            );
            assert!(
                (released.c[j] - 1.0).abs() < 1e-6,
                "C[{j}] is {}, expected 2 * sigmoid(0) = 1",
                released.c[j]
            );
        }
        let a_sum: f32 = released.a.iter().sum();
        assert!(
            (a_sum - 2.0).abs() < 1e-5,
            "four independent sigmoids at zero logits sum to 2, got {a_sum}"
        );
        assert!(
            released.a.iter().all(|value| (value - 0.25).abs() > 1e-3),
            "the softmax reading would put 0.25 in every A entry"
        );

        // A nonzero logit moves ITS OWN entry only; the other three stay at sigmoid(0) + eps.
        let mut hc_base: HcBase = [0.0f32; PREDICTOR_ROWS];
        hc_base[PRE_ROWS_OFFSET] = 1.0;
        hc_base[POST_ROWS_OFFSET + 3] = 2.0;
        let logistic = |z: f32| 1.0 / (1.0 + (-z).exp());
        let biased = h_of_with_scale(
            &mut Streams::new(vec![1.0f32; HC_MULT * HIDDEN]).capture(),
            &predictor,
            &hc_base,
            &hc_scale,
        );
        assert!((biased.a[0] - (logistic(1.0) + SINKHORN_EPS)).abs() < 1e-6);
        for j in 1..HC_MULT {
            assert!(
                (biased.a[j] - (0.5 + SINKHORN_EPS)).abs() < 1e-6,
                "A[{j}] moved even though its logit did not"
            );
        }
        assert!((biased.c[3] - 2.0 * logistic(2.0)).abs() < 1e-6);
        for j in 0..3 {
            assert!((biased.c[j] - 1.0).abs() < 1e-6);
        }
        let expected_sum = logistic(1.0) + SINKHORN_EPS + 3.0 * (0.5 + SINKHORN_EPS);
        let a_sum: f32 = biased.a.iter().sum();
        assert!(
            (a_sum - expected_sum).abs() < 1e-6,
            "A is {a_sum}, and the independent sigmoids sum to {expected_sum}"
        );
        assert!(
            (a_sum - 1.0).abs() > 0.5,
            "a softmax would sum to one; the released activation must not"
        );

        // The earlier reading is still available, still labelled, and is a DIFFERENT computation -
        // this test says so rather than pretending both can hold.
        let old = h_of_with_bias(
            &mut Streams::new(vec![1.0f32; HC_MULT * HIDDEN]).capture(),
            &predictor,
            &hc_base,
        );
        let e = 1.0f32.exp();
        assert!(
            (old.a[0] - e / (e + 3.0)).abs() < 1e-6,
            "the old A remains a softmax"
        );
        assert!((old.a.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(
            (old.a[0] - biased.a[0]).abs() > 0.1,
            "the two readings must differ at A[0]"
        );
        assert!(
            (old.c[3] - biased.c[3]).abs() > 0.5,
            "the two readings must differ at C[3]"
        );

        // The semantic change itself: a unit scale and a zero base do not reproduce the old path,
        // because BOTH the activation and the comb schedule differ now. The comb difference is not a
        // side effect to be tolerated: the released schedule ends with a column pass and adds eps to
        // the matrix, so it cannot coincide with this module's generic Sinkhorn-Knopp.
        let modern = h_of_with_scale(
            &mut Streams::new(vec![1.0f32; HC_MULT * HIDDEN]).capture(),
            &predictor,
            &zero_base,
            &hc_scale,
        );
        let original = h_of(
            &mut Streams::new(vec![1.0f32; HC_MULT * HIDDEN]).capture(),
            &predictor,
        );
        assert_ne!(modern.a, original.a, "A changed semantically");
        assert_ne!(modern.c, original.c, "C changed semantically");
        assert_ne!(
            modern.b, original.b,
            "B changed semantically: the released schedule is not the generic Sinkhorn-Knopp"
        );
    }

    #[test]
    fn the_scaled_layer_moves_the_residual_streams() {
        // The scale has to reach the LAYER, not only the coefficient helper: with the same bias, a
        // scale of [3, 0.5, -2] against [1, 1, 1] must move B, A, C and the streams the layer writes.
        // This is the end-to-end non-vacuity check for the scale, as
        // `the_biased_layer_moves_the_residual_streams` is for the bias.
        let predictor = deterministic_predictor();
        let rms_weight = vec![1.0f32; HIDDEN];
        let hc_base: HcBase = core::array::from_fn(|index| (index as f32 - 12.0) / 32.0);
        let unit: HcScale = [1.0; HC_SCALE_LEN];
        let scaled: HcScale = [3.0, 0.5, -2.0];

        let (unit_next, unit_coefficients, ..) = mega_mhc_layer_with_scale(
            &mut Streams::new(deterministic_streams()),
            &rms_weight,
            &[0.25; HC_MULT],
            &predictor,
            &hc_base,
            &unit,
            &mut TrafficLedger::default(),
        );
        let (scaled_next, scaled_coefficients, ..) = mega_mhc_layer_with_scale(
            &mut Streams::new(deterministic_streams()),
            &rms_weight,
            &[0.25; HC_MULT],
            &predictor,
            &hc_base,
            &scaled,
            &mut TrafficLedger::default(),
        );

        assert_ne!(
            unit_coefficients.a, scaled_coefficients.a,
            "A must move with the scale"
        );
        assert_ne!(
            unit_coefficients.c, scaled_coefficients.c,
            "C must move with the scale"
        );
        assert_ne!(
            unit_coefficients.b, scaled_coefficients.b,
            "B must move with the scale"
        );
        let largest_delta = unit_next
            .as_slice()
            .iter()
            .zip(scaled_next.as_slice())
            .map(|(unit, scaled)| (unit - scaled).abs())
            .fold(0.0f32, f32::max);
        assert!(
            largest_delta > 1e-3,
            "the scale must move the residual streams, largest delta {largest_delta}"
        );
    }

    #[test]
    fn the_released_split_sinkhorn_adds_eps_once_before_the_first_column_pass() {
        // kernel.py:436-458, with the schedule split in its three parts so each one is visible:
        //   row softmax (no eps in its own denominator) + eps, one column pass /(colsum + eps),
        //   then iterations - 1 (row, column) pairs. With iterations = 1 the closed form below is the
        // whole schedule, so this test cannot agree with the implementation by construction: it
        // recomputes the softmax with exp/sum here and applies eps in exactly one place.
        let fixture = [
            [2.0f32, -1.0, 0.5, 0.0],
            [0.0, 1.5, -0.5, 1.0],
            [-2.0, 0.25, 3.0, -0.75],
            [1.0, 0.0, 0.0, -1.0],
        ];
        let mut got = fixture;
        released_split_sinkhorn(&mut got, 1, SINKHORN_EPS);
        let mut expected = [[0.0f32; HC_MULT]; HC_MULT];
        for source in 0..HC_MULT {
            let row_max = fixture[source]
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            let mut exponentials = [0.0f32; HC_MULT];
            let mut row_sum = 0.0f32;
            for destination in 0..HC_MULT {
                let value = (fixture[source][destination] - row_max).exp();
                exponentials[destination] = value;
                row_sum += value;
            }
            for (slot, exponential) in expected[source].iter_mut().zip(exponentials) {
                *slot = exponential / row_sum + SINKHORN_EPS;
            }
        }
        let column_sums: [f32; HC_MULT] =
            core::array::from_fn(|destination| expected.iter().map(|row| row[destination]).sum());
        for (destination, column_sum) in column_sums.iter().enumerate() {
            for row in expected.iter_mut() {
                row[destination] /= column_sum + SINKHORN_EPS;
            }
        }
        for source in 0..HC_MULT {
            for destination in 0..HC_MULT {
                assert!(
                    (got[source][destination] - expected[source][destination]).abs() < 1e-7,
                    "[{source}][{destination}] is {} and the hand transcription of the schedule says {}",
                    got[source][destination],
                    expected[source][destination]
                );
            }
        }
        // The eps is added to the whole matrix ONCE, so removing it changes the result even after
        // the last pass divides by a sum that also carries eps; the generic Sinkhorn-Knopp does not
        // have this shape at all.
        let mut generic = [[0.0f32; HC_MULT]; HC_MULT];
        for source in 0..HC_MULT {
            for destination in 0..HC_MULT {
                generic[source][destination] = fixture[source][destination].exp();
            }
        }
        sinkhorn_knopp(&mut generic, SINKHORN_ITERS, SINKHORN_EPS);
        let mut full = fixture;
        released_split_sinkhorn(&mut full, SINKHORN_ITERS, SINKHORN_EPS);
        assert_ne!(
            full, generic,
            "the released schedule is not the generic Sinkhorn-Knopp"
        );
        // Twenty iterations terminate on a column pass, so the columns of the released-orientation
        // matrix are the side that sums to one.
        let full_column_sums: [f32; HC_MULT] =
            core::array::from_fn(|destination| full.iter().map(|row| row[destination]).sum());
        for (destination, column_sum) in full_column_sums.iter().enumerate() {
            assert!(
                (column_sum - 1.0).abs() < 1e-5,
                "column {destination} sums to {column_sum}; the last released pass is a column pass"
            );
        }
    }

    #[test]
    fn the_released_split_sinkhorn_runs_iterations_minus_one_pairs() {
        // The sibling test above hand-transcribes the iterations = 1 schedule, which never enters
        // the (row, column) pair loop, so an off-by-one in `for _ in 1..iterations` is invisible to
        // it: 18 pairs also terminate on a column pass with unit columns. This test pins the loop
        // count numerically. `kernel.py:436-458` runs one unconditional column pass and then
        // `iterations - 1` (row, column) pairs, so iterations = 2 and 3 must run one and two pairs.
        // The expected matrices are the output of an independent float64 transcription of that
        // schedule for the fixture below, and they differ most where the extra pair lands; the
        // implementation is f32, hence the 1e-6 bar.
        let fixture = [
            [2.0f32, -1.0, 0.5, 0.0],
            [0.0, 1.5, -0.5, 1.0],
            [-2.0, 0.25, 3.0, -0.75],
            [1.0, 0.0, 0.0, -1.0],
        ];
        // row softmax + eps, one column pass, then ONE (row, column) pair.
        let expected_two = [
            [5.496816e-01, 5.8454554e-02, 1.1457338e-01, 2.515067e-01],
            [5.5371697e-02, 5.3003616e-01, 3.137294e-02, 5.0886584e-01],
            [5.1950157e-03, 1.0526046e-01, 7.201142e-01, 6.1295424e-02],
            [3.8975075e-01, 3.0624767e-01, 1.3393867e-1, 1.783309e-01],
        ];
        // the same, then TWO (row, column) pairs.
        let expected_three = [
            [5.6097523e-01, 6.2984936e-2, 1.0831817e-01, 2.7009445e-01],
            [4.89073e-02, 4.942847e-01, 2.5670015e-02, 4.729582e-01],
            [5.7912927e-03, 1.23891e-01, 7.4366117e-01, 7.190355e-02],
            [3.8432518e-01, 3.1883832e-01, 1.22349721e-01, 1.8504276e-01],
        ];
        for (iterations, expected) in [(2usize, expected_two), (3usize, expected_three)] {
            let mut got = fixture;
            released_split_sinkhorn(&mut got, iterations, SINKHORN_EPS);
            for source in 0..HC_MULT {
                for destination in 0..HC_MULT {
                    assert!(
                        (got[source][destination] - expected[source][destination]).abs() < 1e-6,
                        "iterations = {iterations}: [{source}][{destination}] is {} and the transcribed schedule says {}",
                        got[source][destination],
                        expected[source][destination]
                    );
                }
            }
        }
    }

    #[test]
    fn the_released_projection_rstd_uses_norm_eps_over_the_flattened_stream() {
        // model.py:953-954: mixes = F.linear(x) * rsqrt(mean(x^2) + norm_eps), the mean over the
        // flattened n*d = 20480 stream. Two fixture values pin the two halves of that sentence:
        // 1e-20 in the denominator (an hc_eps of 1e-6 would sit at 1e3, not 1e10) and the n*d
        // denominator with all four copies contributing (identical copies at 2 give mean 4).
        let zero_stream_squares = coefficient_rstd(0.0);
        assert!(
            (zero_stream_squares - 1e10).abs() / 1e10 < 1e-5,
            "coefficient_rstd(0) is {zero_stream_squares}; norm_eps = 1e-20 gives 1e10 and hc_eps = 1e-6 gives 1e3"
        );
        let squares = 4.0f32 * (HC_MULT * HIDDEN) as f32;
        assert!(
            (coefficient_rstd(squares) - 0.5).abs() < 1e-6,
            "four copies at 2 have mean square 4 over the flattened stream, so the rstd is 0.5"
        );

        // The rstd multiplies the logits BEFORE the group scale and the bias. The four copies are
        // identical at 2, the predictor is nonzero, so omitting the multiplier moves every group.
        let predictor = deterministic_predictor();
        let streams = || vec![2.0f32; HC_MULT * HIDDEN];
        let hc_base: HcBase = core::array::from_fn(|index| (index as f32 - 10.0) / 40.0);
        let hc_scale: HcScale = [1.5, -0.5, 2.0];
        let rows = predictor_rows(&mut Streams::new(streams()).capture(), &predictor);
        let got = h_of_with_scale(
            &mut Streams::new(streams()).capture(),
            &predictor,
            &hc_base,
            &hc_scale,
        );
        let logistic = |z: f32| 1.0 / (1.0 + (-z).exp());
        let mut without_rstd_differs = 0;
        for j in 0..HC_MULT {
            let row = rows[HC_MULT * HC_MULT + j];
            let expected =
                logistic(0.5 * row * hc_scale[PRE_SCALE_INDEX] + hc_base[PRE_ROWS_OFFSET + j])
                    + SINKHORN_EPS;
            assert!(
                (got.a[j] - expected).abs() < 1e-6,
                "A[{j}] is {}, expected {expected}",
                got.a[j]
            );
            let missing = logistic(row * hc_scale[PRE_SCALE_INDEX] + hc_base[PRE_ROWS_OFFSET + j])
                + SINKHORN_EPS;
            if (expected - missing).abs() > 1e-3 {
                without_rstd_differs += 1;
            }
        }
        assert!(
            without_rstd_differs > 0,
            "this fixture cannot see the projection rstd at A"
        );
    }

    #[test]
    fn the_released_block_rms_normalises_x_hat_not_the_streams() {
        // model.py:984/991 normalise the COLLAPSED block input, so the statistic is the sum of
        // squares of X_hat over d - not the sum over the four streams. With one stream at 1.0 and
        // a uniform a_prev, X_hat is uniform at 0.3, so the released rstd is 1/0.3; the old streams
        // statistic is 1/sqrt(0.25). The two normalised values are 1.0 and ~0.6, which the
        // reference's E4M3 quantiser maps to DIFFERENT codes (0x78 against 0x7A); the fixture must
        // keep that property or it cannot tell the statistics apart.
        let a_prev = [0.3f32; HC_MULT];
        let mut stream_values = vec![0.0f32; HC_MULT * HIDDEN];
        stream_values[..HIDDEN].fill(1.0);
        let rms_weight = vec![1.0f32; HIDDEN];
        let (_next, _coefficients, codes, scales) = mega_mhc_layer_with_scale(
            &mut Streams::new(stream_values.clone()),
            &rms_weight,
            &a_prev,
            &zero_predictor(),
            &[0.0f32; PREDICTOR_ROWS],
            &[1.0f32; HC_SCALE_LEN],
            &mut TrafficLedger::default(),
        );
        let expected_codes = |value: f32| {
            let mut out = Vec::with_capacity(HIDDEN);
            for _ in 0..HIDDEN / QUANT_BLOCK {
                let normalised = [value; QUANT_BLOCK];
                let scale_code = act_quant_scale_code(value.abs());
                let scale = crate::m4_quant::ue8m0_decode(scale_code);
                for value in normalised {
                    out.push(fp8_e4m3_encode(value / scale));
                }
            }
            out
        };
        let released_inv = 1.0 / (0.3f32 * 0.3f32 + NORM_EPS).sqrt();
        let old_inv = 1.0 / (0.25f32 + 1e-6).sqrt();
        let released = expected_codes(0.3f32 * 1.0f32 * released_inv);
        let old = expected_codes(0.3f32 * 1.0f32 * old_inv);
        assert_eq!(
            codes.as_slice(),
            released.as_slice(),
            "the released block RMS must normalise X_hat, not the streams"
        );
        assert_ne!(
            released, old,
            "this fixture cannot distinguish the two RMS statistics"
        );
        let _ = scales;
    }

    #[test]
    fn released_predictor_rows_are_permuted_and_never_averaged() {
        // The released row order is pre(0..n) / post(n..2n) / comb(2n..). The permutation must move
        // pre to kernel A rows (n*n..n*n+n), post to kernel C rows (n*n+n..) and comb to B rows
        // (0..n*n) transposed. Every released element carries its own value, so the copy index and
        // the column survive the mapping and the test names both. B0 removed the d-wide fold: a
        // helper that averages the copies, or that drops three of them, must fail here.
        let mut released = vec![0.0f32; PREDICTOR_ROWS * HC_MULT * HIDDEN];
        for released_row in 0..PREDICTOR_ROWS {
            for copy in 0..HC_MULT {
                for column in 0..HIDDEN {
                    released[released_row * HC_MULT * HIDDEN + copy * HIDDEN + column] =
                        released_row as f32 * 10.0 + copy as f32 + column as f32 * 1e-3;
                }
            }
        }
        let mut kernel_order = vec![0.0f32; PREDICTOR_ROWS * HC_MULT * HIDDEN];
        released_predictor_into_kernel_rows(&released, &mut kernel_order);
        for kernel_row in 0..PREDICTOR_ROWS {
            let expected_released_row = if kernel_row < HC_MULT * HC_MULT {
                // kernel B[destination][source] <- released comb [source][destination]
                let destination = kernel_row / HC_MULT;
                let source = kernel_row % HC_MULT;
                COMB_ROWS_OFFSET + source * HC_MULT + destination
            } else if kernel_row < HC_MULT * HC_MULT + HC_MULT {
                PRE_ROWS_OFFSET + (kernel_row - HC_MULT * HC_MULT)
            } else {
                POST_ROWS_OFFSET + (kernel_row - HC_MULT * HC_MULT - HC_MULT)
            };
            // No averaging: every copy is carried through, so copy c keeps value + c + column/1000.
            let base = expected_released_row as f32 * 10.0;
            for copy in 0..HC_MULT {
                for column in 0..HIDDEN {
                    let got = kernel_order[kernel_row * HC_MULT * HIDDEN + copy * HIDDEN + column];
                    let expected = base + copy as f32 + column as f32 * 1e-3;
                    assert!(
                        (got - expected).abs() < 1e-3,
                        "kernel row {kernel_row} copy {copy} column {column} is {got}, and released row {expected_released_row} says {expected}"
                    );
                }
            }
        }
        // Non-vacuity 1: the identity row mapping (no permutation, no transpose) does NOT fit.
        let mut wrong_mapping = 0;
        for index in 0..PREDICTOR_ROWS * HC_MULT * HIDDEN {
            if (released[index] - kernel_order[index]).abs() > 1e-3 {
                wrong_mapping += 1;
            }
        }
        assert!(
            wrong_mapping > HC_MULT * HIDDEN,
            "an identity row mapping must not match, and it differs at {wrong_mapping} entries"
        );
        // Non-vacuity 2: the retired fold (mean over the four copies) does NOT fit. If it did, this
        // test would not notice B0 being undone.
        let mut wrong_fold = 0;
        for kernel_row in 0..PREDICTOR_ROWS {
            let released_row = if kernel_row < HC_MULT * HC_MULT {
                let destination = kernel_row / HC_MULT;
                let source = kernel_row % HC_MULT;
                COMB_ROWS_OFFSET + source * HC_MULT + destination
            } else if kernel_row < HC_MULT * HC_MULT + HC_MULT {
                PRE_ROWS_OFFSET + (kernel_row - HC_MULT * HC_MULT)
            } else {
                POST_ROWS_OFFSET + (kernel_row - HC_MULT * HC_MULT - HC_MULT)
            };
            for column in 0..HIDDEN {
                let mut mean = 0.0f32;
                for copy in 0..HC_MULT {
                    mean += released[released_row * HC_MULT * HIDDEN + copy * HIDDEN + column];
                }
                mean /= HC_MULT as f32;
                let first_copy = kernel_order[kernel_row * HC_MULT * HIDDEN + column];
                if (mean - first_copy).abs() > 1e-3 {
                    wrong_fold += 1;
                }
            }
        }
        assert!(
            wrong_fold > HIDDEN,
            "an averaged (pre-B0 folded) row must not match the full projection row, and it differs at {wrong_fold} entries"
        );
    }
}
