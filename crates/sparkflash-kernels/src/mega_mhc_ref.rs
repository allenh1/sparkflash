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
//! the accumulator) but not the internals. This module states the project's reading explicitly and
//! implements it: the 24 predictor rows are applied to the mean over the four streams, the first 16
//! are exponentiated and Sinkhorn-Knopp-normalised into the 4x4 matrix B, the next 4 are softmaxed
//! into the 4-vector A, and the last 4 are softmaxed into C. A different reading would change these
//! numbers and nothing in this file claims otherwise; what it does claim is that the shape and the
//! traffic are the ones the kernel implements.

use crate::m4_quant::{E4M3_ELEMENT_EMAX, QUANT_BLOCK, fp8_e4m3_encode, ue8m0_from_amax};

/// The hidden width. Mirrors SPARK_M4_HIDDEN.
pub const HIDDEN: usize = 5120;
/// The number of residual streams. Mirrors SPARK_M4_HC_MULT.
pub const HC_MULT: usize = 4;
/// Sinkhorn-Knopp iterations. Mirrors SPARK_M4_HC_SINKHORN_ITERS.
pub const SINKHORN_ITERS: usize = 20;
/// The Sinkhorn floor. Mirrors SPARK_M4_HC_EPS.
pub const SINKHORN_EPS: f32 = 1e-6;
/// The predictor rows H feeds on: n*n for B, 2n for A and C. Mirrors SPARK_M4_HC_PREDICTOR_ROWS.
pub const PREDICTOR_ROWS: usize = HC_MULT * HC_MULT + 2 * HC_MULT;

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

/// H(X_l): the predictor rows applied to the stream mean, then exp + Sinkhorn into B, softmax into A
/// and into C. The predictor is the folded (n*n + 2n) x d matrix of the kernel, row-major.
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
/// Panics if the predictor is not the folded (n*n + 2n) by d matrix.
#[must_use]
pub fn h_of(registers: &mut Registers, predictor: &[f32]) -> Coefficients {
    assert_eq!(
        predictor.len(),
        PREDICTOR_ROWS * HIDDEN,
        "the folded predictor must be (n*n + 2n) by d"
    );
    // The stream mean, then one dot product per predictor row, all out of the register copy.
    let mut rows = [0.0f32; PREDICTOR_ROWS];
    for row in 0..PREDICTOR_ROWS {
        let mut acc = 0.0f32;
        for stream in 0..HC_MULT {
            for i in 0..HIDDEN {
                acc += registers.read(stream, i) * predictor[row * HIDDEN + i];
            }
        }
        rows[row] = acc / HC_MULT as f32;
    }

    // B: exponentiate the n*n rows and run Sinkhorn-Knopp.
    let mut b = [[0.0f32; HC_MULT]; HC_MULT];
    for i in 0..HC_MULT {
        for j in 0..HC_MULT {
            b[i][j] = rows[i * HC_MULT + j].exp();
        }
    }
    sinkhorn_knopp(&mut b, SINKHORN_ITERS, SINKHORN_EPS);

    // A and C: softmax over n rows each.
    let a = softmax4(&rows[HC_MULT * HC_MULT..HC_MULT * HC_MULT + HC_MULT]);
    let c = softmax4(&rows[HC_MULT * HC_MULT + HC_MULT..PREDICTOR_ROWS]);
    Coefficients { b, a, c }
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
/// # Panics
/// Panics if the RMS weight is not d wide, or the predictor is not the folded matrix.
#[must_use]
pub fn mega_mhc_layer(
    streams: &mut Streams,
    rms_weight: &[f32],
    a_prev: &[f32; HC_MULT],
    predictor: &[f32],
    ledger: &mut TrafficLedger,
) -> (Counted<f32>, Coefficients, Counted<u8>, Counted<u8>) {
    assert_eq!(rms_weight.len(), HIDDEN, "the RMS weight is d wide");

    // THE one pass over the streams: capture() reads every element exactly once through the counted
    // accessor and hands back the register copy. Everything below this line uses registers, which
    // is the whole content of the single-pass claim, and nothing below this line can reach the
    // stream buffer except through the counting accessor.
    let mut registers = streams.capture();

    // X_hat from the previous layer's A, and the residual sum of squares for the RMS norm: both are
    // formed on the way past the register copy, which is how the kernel forms them.
    let mut xhat = vec![0.0f32; HIDDEN];
    let mut sum_squares = 0.0f32;
    for (index, slot) in xhat.iter_mut().enumerate() {
        let mut value = 0.0f32;
        for (stream, weight) in a_prev.iter().enumerate() {
            let element = registers.read(stream, index);
            sum_squares += element * element;
            value += weight * element;
        }
        *slot = value;
    }

    // H(X_l) and the B X_l term both come from that register copy, exactly as they do in the
    // kernel, so they cost no further stream traffic. h_of takes the registers and nothing else;
    // h_of_counted exists only for the deliberately wasteful variant below, which is what the bound
    // test contrasts against.
    let coefficients = h_of(&mut registers, predictor);

    // F: the RMS pre-norm, the per-32 ue8m0 scales and the FP8 codes. Writing those is d elements
    // plus d / 32 scale bytes, which is the second half of the bound.
    let inv_rms = 1.0f32 / (sum_squares / (HC_MULT * HIDDEN) as f32 + SINKHORN_EPS).sqrt();
    let mut codes: Counted<u8> = Counted::new();
    let mut scales: Counted<u8> = Counted::new();
    for block in 0..HIDDEN / QUANT_BLOCK {
        let base = block * QUANT_BLOCK;
        let mut amax = 0.0f32;
        let mut normalised = [0.0f32; QUANT_BLOCK];
        for i in 0..QUANT_BLOCK {
            let value = xhat[base + i] * inv_rms * rms_weight[base + i];
            normalised[i] = value;
            let magnitude = value.abs();
            amax = if magnitude > amax { magnitude } else { amax };
        }
        let scale_code = ue8m0_from_amax(amax, E4M3_ELEMENT_EMAX);
        scales.push(scale_code);
        // MX encodes the value DIVIDED by the shared scale: the scale is 2^(floor(log2(amax)) - emax),
        // so value / scale lands near the element format's own exponent range and the decoder
        // multiplies it back. Encoding the value directly and multiplying by the scale on the way out
        // would shrink the block by a factor of 2^emax - which is what this reference did until the
        // round-trip test below caught it.
        let scale = crate::m4_quant::ue8m0_decode(scale_code);
        for value in normalised.iter() {
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
    assert_eq!(predictor.len(), PREDICTOR_ROWS * HIDDEN);
    let mut rows = [0.0f32; PREDICTOR_ROWS];
    for row in 0..PREDICTOR_ROWS {
        let mut acc = 0.0f32;
        for stream in 0..HC_MULT {
            for i in 0..HIDDEN {
                acc += streams.read(stream, i) * predictor[row * HIDDEN + i];
            }
        }
        rows[row] = acc / HC_MULT as f32;
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
        vec![0.0f32; PREDICTOR_ROWS * HIDDEN]
    }

    #[test]
    fn golden_zero_predictor_is_hand_computable() {
        // With a zero predictor every row of H is zero, so B = exp(0) = 1 everywhere and Sinkhorn
        // on the all-ones matrix leaves 1/n = 0.25; A and C are softmax(0) = 0.25 each.
        //
        // Take X_l = 1 in every stream and a_prev = 0.25, so X_hat = 0.25 * 4 * 1 = 1. The RMS
        // denominator is sqrt(1 + eps), whose deviation from 1 is a factor 1 - 5e-7, and with an RMS
        // weight of 3 the normalised value is 2.9999985 - not on a power of two and not near a
        // rounding boundary, so the block scale is 2^(floor(log2(3)) - 8) = 2^-7 (ue8m0 code 120)
        // and the encoded element is 3 / 2^-7 = 384, which is E4M3 0x7C. Read back, that is
        // 384 * 2^-7 = 3, and X_{l+1} = B X + C F = 4 * 0.25 * 1 + 0.25 * 3 = 1.75.
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
}
