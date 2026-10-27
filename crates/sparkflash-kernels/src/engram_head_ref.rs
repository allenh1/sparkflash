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

//! CPU reference for the Engram head (model.py:350-365), grounded in real checkpoint rows.
//!
//! The M2 store serves 264-byte rows: 256 F8_E4M3 weights and 8 F8_E8M0 block scales, one scale per
//! 32 columns (the PerRow32 rule). Nothing in this crate agreed with that layout - mxfp8_ref is the
//! b12x tiled GEMM format, which is a different thing with the same dtype names - so the consumer
//! measurement that decides decode viability rested on a reconstruction. This module is the
//! reference, and it is checked two ways:
//!
//! 1. AGAINST RELEASED BYTES. [REAL_ROWS] holds eight rows read read-only from the published
//!    checkpoint, including the LAST row of each table (384006167 on layer 1, 384016681 on layer
//!    14), because the final row is what proves the tail is data rather than padding. The expected
//!    values come from an independent decoder (stdlib Python, mantissa and exponent arithmetic by
//!    hand) and are compared BIT FOR BIT where bits are what is claimed; the floating point sums
//!    are compared with a stated tolerance.
//!
//! FIXTURE PROVENANCE. [REAL_ROWS] embeds 2,112 bytes of released checkpoint weights - eight rows of
//! 256 weight bytes with their 8 scale bytes each - read read-only from the published
//! deepseek-ai/DeepSeek-V4.1-Flash snapshot. The bytes are embedded rather than only their hashes on
//! purpose: it is what lets a test decode RELEASED bytes in CI instead of checking the decoder against
//! itself. reference/LICENSE is MIT (Copyright (c) 2023 DeepSeek) and reference/MODEL_CARD.md:183 says
//! "This repository and the model weights are licensed under the MIT License", so redistribution is
//! permitted with the notice kept; 2,112 bytes against a 510 GB checkpoint is de minimis, and this
//! repository is internal. THIS MUST BE RE-REVIEWED IF THE REPOSITORY IS EVER MADE PUBLIC. If the model
//! licence is ever found to restrict redistribution outright, these bytes come out and only the
//! FNV-1a64 and bits_at goldens stay, which costs the ability to decode released bytes in CI.
//! 2. AGAINST THE TREE. sparkflash-engine/src/forward/engram.rs is the executable definition of
//!    this arithmetic; the constants and the order of operations here mirror it, and the engine
//!    tests that pin the rule are the reason the formulas below are quoted with line numbers.
//!
//! WHAT IS CONFIRMED FROM THE TREE, and where:
//!
//! - the row layout is one E8M0 scale per 32 columns of each row (forward/engram.rs:462-467,
//!   "the PerRow32 rule of sparkflash-engine's dtype.rs"), applied as an f32 multiply (linalg.rs:176-184,
//!   `weight[row * cols + col] *= scale`);
//! - E4M3 decodes with bias 7, subnormals `mantissa * 2^-9`, and 0x7F/0xFF as NaN (quant.rs:109-128);
//!   E8M0 decodes as `2^(code - 127)` with 255 as NaN (quant.rs:135-142);
//! - one token fetches `hash_columns` rows and CONCATENATES them (forward/engram.rs:510-511,
//!   639-641), `hash_columns = orders * heads` (forward/engram.rs:63-70);
//! - wkv turns that concatenation into one key per hc copy plus one shared value
//!   (forward/engram.rs:512, 589-592, 642-643);
//! - the gate is a normalised dot product of the residual stream against the key, scaled by both
//!   root-mean-squares and by dim^-0.5, with a SIGNED square root before the sigmoid
//!   (forward/engram.rs:648-664, model.py:356-362);
//! - q_weight and k_weight are only ever used as their product (forward/engram.rs:598-615);
//! - the write adds gate x value per (hc copy, column), in place (forward/engram.rs:676-682).
//!
//! WHAT THIS MODULE CANNOT CONFIRM, stated rather than smoothed over:
//!
//! - the scale grid is applied to a WHOLE TABLE by the engine (rows x cols), not to one row at a
//!   time; [decode_row] is the same arithmetic restricted to one row, which is what the store
//!   hands out. The restriction is exact for the weight half, but a consumer that applies scales
//!   per row and a consumer that applies them per table would differ if a row's 8 scale bytes were
//!   shared or misaligned - the fixtures check that they are not;
//! - the released configuration's `engram_n_heads`/`orders` (hence hash_columns) is not a
//!   constant here: [head_ref] takes the ids it is given, so a wrong 24 is not detectable in this
//!   module. forward/engram.rs:63-70 derives it from the config;
//! - nothing here proves the values are loaded with the right ENDIANNESS or offsets by the store;
//!   [REAL_ROWS] proves the bytes at `data0 + id*256` and `data0 + id*8` decode to trained-looking
//!   weights, not that the rank files are addressed correctly.
//!
//! - agreement with the engine's own executable path is NOT executed here: this crate does not
//!   depend on sparkflash-engine, so forward/engram.rs is quoted with line numbers rather than
//!   called the way a cross-check would call it. Until such a check exists, "mirrors the tree"
//!   means "mirrors the lines cited above", not "was diffed against it".
//!
//! This is a reference: it is scalar, it allocates, and it is not how a serving path should compute
//! the head. Its job is to be checkable.

/// Weight bytes in one Engram row. Mirrors sparkflash_engram::layout::ENGRAM_ROW_BYTES.
pub const ROW_BYTES: usize = 264;
/// One head's width. Mirrors sparkflash_engram::layout::ENGRAM_HEAD_DIM.
pub const HEAD_DIM: usize = 256;
/// Columns sharing one E8M0 scale. Mirrors sparkflash-engine/src/dtype.rs SCALE_BLOCK.
pub const SCALE_BLOCK: usize = 32;
/// Scales in one row: HEAD_DIM / SCALE_BLOCK. Mirrors ENGRAM_SCALE_BYTES.
pub const SCALE_BYTES: usize = HEAD_DIM / SCALE_BLOCK;
/// The id that marks a position taking no part in an n-gram (forward/engram.rs:61).
pub const HASH_SENTINEL: u32 = u32::MAX;

/// Decodes one E4M3 code, mirroring quant.rs:109-128.
#[must_use]
pub fn e4m3_to_f32(code: u8) -> f32 {
    let negative = code & 0x80 != 0;
    let exponent = (code >> 3) & 0x0F;
    let mantissa = code & 0x07;
    if exponent == 0x0F && mantissa == 0x07 {
        return f32::NAN;
    }
    let magnitude = if exponent == 0 {
        f32::from(mantissa) * 2.0f32.powi(-9)
    } else {
        (1.0 + f32::from(mantissa) / 8.0) * 2.0f32.powi(i32::from(exponent) - 7)
    };
    if negative { -magnitude } else { magnitude }
}

/// Decodes one E8M0 scale, mirroring quant.rs:135-142.
#[must_use]
pub fn e8m0_to_f32(code: u8) -> f32 {
    if code == 255 {
        return f32::NAN;
    }
    2.0f32.powi(i32::from(code) - 127)
}

/// Decodes one stored row to its 256 values.
///
/// weight is the 256 weight bytes, scale the 8 scale bytes. Element `i` is multiplied by the scale
/// at `i / SCALE_BLOCK`, which is linalg.rs:176-184 restricted to a single row.
///
/// # Panics
///
/// Never: both slices are fixed-size.
#[must_use]
pub fn decode_row(weight: &[u8; HEAD_DIM], scale: &[u8; SCALE_BYTES]) -> [f32; HEAD_DIM] {
    // The eight scales are decoded once, exactly as the tree does it: the engine decodes the whole
    // scale tensor up front (quant.rs:277-279) and apply_scales only multiplies (linalg.rs:176-184).
    let mut decoded_scales = [0.0f32; SCALE_BYTES];
    for (slot, code) in decoded_scales.iter_mut().zip(scale.iter()) {
        *slot = e8m0_to_f32(*code);
    }
    let mut out = [0.0f32; HEAD_DIM];
    for (index, value) in out.iter_mut().enumerate() {
        *value = e4m3_to_f32(weight[index]) * decoded_scales[index / SCALE_BLOCK];
    }
    out
}

/// The gate for one (token, hc copy): the normalised dot product with a signed square root.
///
/// dot, stream and key are the quantities forward/engram.rs:650-662 computes; this function is the
/// last two lines of that loop, model.py:359-362.
#[must_use]
pub fn gate(stream: &[f32], key: &[f32], product: &[f32], dim: usize, eps: f32) -> f32 {
    let mut stream_square = 0.0f32;
    let mut key_square = 0.0f32;
    let mut dot = 0.0f32;
    for column in 0..dim {
        let h = stream[column];
        let k = key[column];
        stream_square += h * h;
        key_square += k * k;
        dot += h * product[column] * k;
    }
    let rstd = (1.0 / (stream_square / dim as f32 + eps).sqrt())
        * (1.0 / (key_square / dim as f32 + eps).sqrt());
    let dot = dot * rstd * (1.0 / (dim as f32).sqrt());
    sigmoid(dot.abs().max(1e-6).sqrt().copysign(dot))
}

/// The logistic function, spelled out so the reference does not depend on a helper.
#[must_use]
pub fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

/// The shape and numerics of one head, so the entry point stays under the argument limit.
#[derive(Clone, Copy, Debug)]
pub struct HeadShape {
    /// The residual width (5120 for the released model).
    pub dim: usize,
    /// Residual copies the write fans out to (4 for the released model).
    pub hc: usize,
    /// The normalisation epsilon.
    pub eps: f32,
    /// One head's width. 256 in the released checkpoint; ModelConfig and the tests can differ.
    pub head_dim: usize,
}

/// One token of the Engram head: fetch, concatenate, project, gate, write.
///
/// [head_ref] is forward/engram.rs:624-672 for a single token, with the row source injected so the
/// caller can serve it from the M2 store instead of from a 98 GB decode. `rows` is called once per
/// id and returns the 256 values of that row.
///
/// Returns `None` when any id is [HASH_SENTINEL]: the position takes no part in the n-gram, and the
/// engine skips both the lookup and the write (forward/engram.rs:626-628).
///
/// # Panics
///
/// Panics when `wkv` is shorter than `ids.len() * HEAD_DIM * (hc + 1)`, when `product` is shorter
/// than `hc * dim`, or when `x_hc` is not exactly `hc * dim` long: a reference that silently
/// computed on short buffers would certify nothing.
#[must_use]
pub fn head_ref(
    x_hc: &mut [f32],
    ids: &[u32],
    rows: &mut dyn FnMut(u32) -> Vec<f32>,
    wkv: &[f32],
    product: &[f32],
    shape: HeadShape,
) -> Option<Vec<f32>> {
    let HeadShape {
        dim,
        hc,
        eps,
        head_dim,
    } = shape;
    assert_eq!(x_hc.len(), hc * dim, "x_hc must be one whole hc row");
    assert!(product.len() >= hc * dim, "product must cover hc * dim");
    let fetched_width = ids.len() * head_dim;
    let out_width = dim * (hc + 1);
    assert!(
        wkv.len() >= out_width * fetched_width,
        "wkv must cover out_width x fetched_width"
    );
    if ids.contains(&HASH_SENTINEL) {
        return None;
    }
    let mut fetched = vec![0.0f32; fetched_width];
    for (column, &id) in ids.iter().enumerate() {
        let row = rows(id);
        assert_eq!(
            row.len(),
            head_dim,
            "the row source returned the wrong width"
        );
        fetched[column * head_dim..(column + 1) * head_dim].copy_from_slice(&row);
    }
    let mut kv = vec![0.0f32; out_width];
    for (out, slot) in kv.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        for (column, &value) in fetched.iter().enumerate() {
            acc += value * wkv[out * fetched_width + column];
        }
        *slot = acc;
    }
    let (key, value) = kv.split_at(hc * dim);
    let mut gates = vec![0.0f32; hc];
    for (copy, gate_slot) in gates.iter_mut().enumerate() {
        let stream = &x_hc[copy * dim..(copy + 1) * dim];
        let key_copy = &key[copy * dim..(copy + 1) * dim];
        *gate_slot = gate(
            stream,
            key_copy,
            &product[copy * dim..(copy + 1) * dim],
            dim,
            eps,
        );
    }
    for copy in 0..hc {
        for column in 0..dim {
            x_hc[copy * dim + column] += gates[copy] * value[column];
        }
    }
    Some(gates)
}

/// One real row from the released checkpoint, with its independently computed expectations.
pub struct RealRow {
    /// The Engram layer id the row came from (1 or 14).
    pub layer: u8,
    /// The row id, i.e. the value used at `data0 + id * 256`.
    pub id: u64,
    /// The 256 weight bytes, hex.
    pub weight: &'static str,
    /// The 8 scale bytes, hex.
    pub scale: &'static str,
    /// Sum of the 256 decoded values, accumulated in f64 by the independent decoder.
    pub sum: f32,
    /// Sum of squares of the 256 decoded values.
    pub sum_squares: f32,
    /// Smallest decoded value.
    pub min: f32,
    /// Largest decoded value.
    pub max: f32,
    /// FNV-1a 64 over the little-endian f32 bit patterns of all 256 decoded values.
    pub fnv1a64: u64,
    /// The bit patterns of decoded values 0, 127 and 255.
    pub bits_at: [u32; 3],
}

// 8 real rows from the released checkpoint, read-only, 264 bytes each.
// Bytes: layers.{1,14}.engram.embed.weight at data0 + id*256 (256 B) and
// layers.{1,14}.engram.embed.scale at data0 + id*8 (8 B). Row 384006167 is the LAST row
// of layer 1 and 384016681 the last row of layer 14, so the tail is proved to be data.
/// Eight rows of released checkpoint weights, read read-only, 264 bytes each.
///
/// Bytes: layers.{1,14}.engram.embed.weight at data0 + id*256 (256 B) and
/// layers.{1,14}.engram.embed.scale at data0 + id*8 (8 B). Row 384006167 is the LAST row
/// of layer 1 and 384016681 the last row of layer 14, so the tail is proved to be data.
/// The four float expectations are f64 accumulations from the independent decoder, printed to
/// six significant digits: the test compares them with a stated tolerance, and the exact
/// claims are fnv1a64 and bits_at, which are compared bit for bit.
///
/// Regenerate the independent side with tests/fixtures/engram_head/golden_decode.py.
pub const REAL_ROWS: [RealRow; 8] = [
    RealRow {
        layer: 1,
        id: 0,
        weight: "e36aed4fc6d9f5e8e0f6f2715af1cde27b44f949e4d1f0ee745b6370dcd7edbbf07470d9ec71e4f477716b57f8625bebeef0f8d8f8ef7478df5a6cdfdf75686a67717964705f58726c61fb7974ef73e3e57bdc78d7f371eb7571f7ecf07549435be7e26363f4eae86ceb6ce26d71f654e7526a576fee76ece87377fae974ea666d7669f3f26775ebefd5e5f8edddf071e077ef4b6860da6edff16af1f6e54b78cf6b75e873e7eaf5725dece5e958f8ec5bebf1e8646852666871ec7276f165f4f15ad55268e970f5726955f46edd62ec6ade6768e6f070725dd876eadef9e95e6962d075e3f6f0da51ed4ddaf172ecf2f4f770eed86bee6de2eb69627061f554",
        scale: "7878777878787878",
        sum: -8.93164,
        sum_squares: 229.353,
        min: -2.5,
        max: 2.75,
        fnv1a64: 0x6ed14a09038df838,
        bits_at: [0xbeb00000, 0x3ee00000, 0x3dc00000],
    },
    RealRow {
        layer: 1,
        id: 1,
        weight: "f25c70faeae969e9f6e270c8df66da7571dae0f8756c68f87bed72ee4be9ea5bf3f07078f3dff26c636d6ddd6eec5c6673f0ebe1726c654964717279e6f4ed7265736eedf8eb6df16f6af0ee70def75cdb665fe461626e6f6869e6f57af7db6c6a6570fa69f7eb64ee6ee7726d6bf4796f616b6c71f36ff0e47171f54f6b65e1f169f27778ed5c5ed4ebd3ddf5677665d572d5d5ed6764f1f469eef671e8de6de7deb56de3767362f8e3ed52f6e9e67365f471f063fa78797372f1fe70676ee2f0ec69e874f842ece6f3697368f36aedc2e3f56aece2d76ecef8cb736e5bf4f9fc50e1e6dc6eeb5e67f466d8e558eae862ef7bebebe877f66a666bfb74656bed",
        scale: "7878787878787878",
        sum: -5.44385,
        sum_squares: 308.968,
        min: -3.5,
        max: 2.75,
        fnv1a64: 0xffc552dfcbe60b86,
        bits_at: [0xbfa00000, 0xbe900000, 0xbf500000],
    },
    RealRow {
        layer: 1,
        id: 4095,
        weight: "705e3476f3dd5c3d596d556aed6fe8775871626d6b6adbd85c704bf0f46e6876716e65ed5b6162d3e4e6d4ebeaedef74db6d706f7078e5e2ecf56d7061e9dd7666f1776251765f67e7eb756af16f636d70f5ef66f05f61e9eeea63dff1e76cdb66f573dbe9dc58ee79e272e6654de3706af467e4f5585c6bdeeaeeedefed676a6c54d97167eac273eb6b68e6686273f1f470f0efe9d9dde2695a5c7774e7f3e9d9716c6ce943666ae364745dca71ebe6f273746471d562db70ec7076d8ea716f5869537163d97b687553eae46669e86a6eef69df5b79d9dcdaef6971ea79526aefe1e074ece3f2e565696ce4f36d75dbebe76cd964def4e6d8707763f5d8c8e5",
        scale: "7878787878787878",
        sum: 39.7939,
        sum_squares: 196.399,
        min: -1.625,
        max: 2.75,
        fnv1a64: 0x3823f567942ab994,
        bits_at: [0x3f800000, 0x3f200000, 0xbed00000],
    },
    RealRow {
        layer: 1,
        id: 12345678,
        weight: "69e1b86e71f2606b6e657ce2dc69f26176eeeef36276d6fb6af174faeada7360746cf3f168f7e87bf2f9f2f37df2617775f0f3e16ee6686360694f7666756cf07ae869737274f376494d6ef06e686ce7eae5eaf76defebf9735d56f9f2f2747a6c60eecb6cf7ebea70732575e66eeb62f674d0e5f2723d746276e86bf5e06a4af275e4785e6c66f7715ff453e2716fe7ec6cf8e4f8e3f5f0f9f56df17058dff0d96c716366eced69ecf56666f170eadcf6e86e7874e4686e706cf269e34df54ef86edd6071eaf8f26677f5e567566c7270e7f1ef6ce8e668697670e36edc71f6d9e4eb5870e3d2d5be6ae378d33c69e16262e3666aec50dce8636cf4f3edece9",
        scale: "7878787878787879",
        sum: 1.64709,
        sum_squares: 352.345,
        min: -3.0,
        max: 4.0,
        fnv1a64: 0x59d1910930ce8737,
        bits_at: [0x3f100000, 0x3d200000, 0xbf900000],
    },
    RealRow {
        layer: 1,
        id: 384006167,
        weight: "d7f26f726270e1e9687674776dec61f0d0c3e76e55e86edf6862ebe3e4e66174eaf55b60f678eafc79eb777b75fa5e7977d47769796f6fe5fd6853795f7bf773edeaede8e064f26e5d6a686275eff2d4eaec72f1e25d60f1f5e8785b4e5949e3f6e7e3faf67d7afa74f8fbf1f2fd7af8787efddf76f9f66676f9e078ebf0e1f7f369a5ec3b4f6361d1ecf170edf17a696e62735765f4596d68e96268f1e0ede7d0edd86c72eae8e9e865e96d78f268d36c6ef275e9f1665d6966f1e0eaee7674f168ed68e560626ce97872dce9ef72f0f6ee757eeb7c63e8506bf3706ef5f36977726f5461e8e1655df26470e5e0e972756368edf3455c71e8f2e3f265cc71e1",
        scale: "7877787778787778",
        sum: 11.5326,
        sum_squares: 186.568,
        min: -1.625,
        max: 2.5,
        fnv1a64: 0xc6ea0a0b18cedbe4,
        bits_at: [0xbdf00000, 0xbf700000, 0xbe900000],
    },
    RealRow {
        layer: 14,
        id: 0,
        weight: "69c0efe0686373e1ebf876f3d7706b7070b365d47bf7dc6de061e2f57463e0eee6f167625ddbe86fe5697675f4f9604859ea6ee8d06aef58696c7279f2f66d77f6effaec5d63b5ece9676668f2f3756de8e86772ec6deaf17267e47159dcf4ea5eddf171f16cee646b6f73e6e3e7ecce6963adc070e0efe9e2ea72e972ec77f86be74f7874bd6a5a28e96c6c707161f4707066606f686d76f6f6e268db5e63756b66e367f2d9dc5d49ebd870d5ee7070e4ea67df76f764ee77e4e9d36e74f56e6b7272e26579586465ebebf1f3f0ebf371ec5bf161e77957765d72cfdee4ed70f5dc65f06967ce6ee2e94ceef3eeec72d9e862dce1d7d2d7e1ef74ed57f85ce9",
        scale: "7878787878787878",
        sum: 11.1736,
        sum_squares: 231.744,
        min: -2.5,
        max: 2.75,
        fnv1a64: 0x2f6ddc257959dd55,
        bits_at: [0x3f100000, 0xc0000000, 0xbf100000],
    },
    RealRow {
        layer: 14,
        id: 7,
        weight: "eafadf67e573cfebd8ebe978f663ecea4aef6f727c745d6cf6666fe867d966e976f279e9df66735ec0d873f7696e67eceae4e4e6f06ee267f2776e7370e1e0e569756e4cfae843d17d75f3776ac4f0e1ddf77a5cf0747268e4f3666e47f472e5f5c9f564faf36f75ebe3d06ff169f4f17472eff9c5d667696065fd6be97666e5695ff1f1757171636d72f9f0ed48f3ef7757f564666e71f0795073dc61d9fa707779e6f65ef26cf06ffaf7e45df0dff871f971ebe5e9f4e0f769f56270ee4cebedf57663eb7455f0716ce0dae07af770e95c7ce7f4f3f0da685f7672747672eaf1766bf359e7726d7574f1afe0e07374e4f4ef7059f77c7170e25df4d0ee6961",
        scale: "7878787878787878",
        sum: 9.79517,
        sum_squares: 357.72,
        min: -3.25,
        max: 3.25,
        fnv1a64: 0xea65aec1a819c096,
        bits_at: [0xbf200000, 0xbed00000, 0x3e900000],
    },
    RealRow {
        layer: 14,
        id: 384016681,
        weight: "6d5150d1646968eedbd86b6f6cf179e2586a6e6aef6659da62e561f2ebe1f367ec7a6972f9d5fbf2f9f37df9f8f1dc74c1697eea6ed2f2f56379f172eb4ae2f4e74be267e7e9ce78f6f5eef2777ced777a757569bb7c79f0f3f2f9f2706de2eaf2e6e4e9694dedeb6e6ee3f5ec6b556df2e9be5aef6e625f79d565dbe3e86e6e6ce7f1eb6f6ddbe6f1d7e9d3e07b6cddf3ee6b62e26769da74546df04af2e16f646cb4e8f37ed2ee5af56ce97d5cfce760dade67706bf0f1e9ef7571e35369b4ed6d6af368ebe9e1ef586df4c9da6964dcd56cd372f1516df26c78c8dd65f06a596567e25bf4f8bbf2e86cf5eb5966f1645adce9dee3655cd766ee593f56d9dd",
        scale: "7877777878787878",
        sum: -1.58057,
        sum_squares: 185.814,
        min: -3.0,
        max: 3.5,
        fnv1a64: 0xda3db12034898603,
        bits_at: [0x3f500000, 0x3f600000, 0xbe500000],
    },
];
#[cfg(test)]
mod tests {
    use super::*;

    fn unhex(text: &str) -> Vec<u8> {
        assert!(text.len().is_multiple_of(2), "odd hex length");
        (0..text.len() / 2)
            .map(|i| u8::from_str_radix(&text[2 * i..2 * i + 2], 16).expect("hex"))
            .collect()
    }

    fn weight_bytes(row: &RealRow) -> [u8; HEAD_DIM] {
        let bytes = unhex(row.weight);
        assert_eq!(bytes.len(), HEAD_DIM);
        let mut out = [0u8; HEAD_DIM];
        out.copy_from_slice(&bytes);
        out
    }

    fn scale_bytes(row: &RealRow) -> [u8; SCALE_BYTES] {
        let bytes = unhex(row.scale);
        assert_eq!(bytes.len(), SCALE_BYTES);
        let mut out = [0u8; SCALE_BYTES];
        out.copy_from_slice(&bytes);
        out
    }

    fn fnv1a64_bits(values: &[f32; HEAD_DIM]) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for value in values {
            for byte in value.to_bits().to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        hash
    }

    /// The released bytes decode to the values an independent decoder computes, bit for bit.
    #[test]
    fn the_real_rows_decode_to_the_independent_goldens() {
        for row in &REAL_ROWS {
            let decoded = decode_row(&weight_bytes(row), &scale_bytes(row));
            assert_eq!(
                [
                    decoded[0].to_bits(),
                    decoded[127].to_bits(),
                    decoded[255].to_bits()
                ],
                row.bits_at,
                "layer {} row {}: decoded bits disagree with the independent decoder",
                row.layer,
                row.id
            );
            assert_eq!(
                fnv1a64_bits(&decoded),
                row.fnv1a64,
                "layer {} row {}: all 256 bit patterns disagree in aggregate",
                row.layer,
                row.id
            );
            // The aggregate goldens are f64 accumulations of these same values, so equality is not
            // claimed; the tolerance covers 256 f32 additions.
            let sum: f32 = decoded.iter().sum();
            let squares: f32 = decoded.iter().map(|v| v * v).sum();
            let finite: Vec<f32> = decoded.iter().copied().filter(|v| v.is_finite()).collect();
            let close = |a: f32, b: f32| (a - b).abs() <= 1e-4 * (1.0 + b.abs());
            assert!(
                close(sum, row.sum),
                "layer {} row {}: sum {sum} vs {}",
                row.layer,
                row.id,
                row.sum
            );
            assert!(
                close(squares, row.sum_squares),
                "layer {} row {}: sum of squares {squares} vs {}",
                row.layer,
                row.id,
                row.sum_squares
            );
            let min = finite.iter().copied().fold(f32::INFINITY, f32::min);
            let max = finite.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            assert!(
                close(min, row.min),
                "layer {} row {}: min {min} vs {}",
                row.layer,
                row.id,
                row.min
            );
            assert!(
                close(max, row.max),
                "layer {} row {}: max {max} vs {}",
                row.layer,
                row.id,
                row.max
            );
        }
    }

    /// The last row of each table is real data, not padding: that is why it is in the fixture.
    #[test]
    fn the_last_rows_are_data_and_not_padding() {
        for row in REAL_ROWS
            .iter()
            .filter(|r| r.id == 384_006_167 || r.id == 384_016_681)
        {
            let decoded = decode_row(&weight_bytes(row), &scale_bytes(row));
            let zeros = decoded.iter().filter(|v| **v == 0.0).count();
            let nonzero = decoded.iter().filter(|v| **v != 0.0).count();
            assert_eq!(
                nonzero, HEAD_DIM,
                "layer {} last row {}: {zeros} of {HEAD_DIM} values are zero, which is what padding would look like",
                row.layer, row.id
            );
            assert!(
                row.min < -0.5 && row.max > 0.5,
                "layer {} last row {}: decoded range {}..{} is too flat for trained weights",
                row.layer,
                row.id,
                row.min,
                row.max
            );
            // Every scale byte is a normal exponent, not the 0xFF that E8M0 reserves for NaN.
            assert!(
                unhex(row.scale).iter().all(|b| *b != 255),
                "layer {} last row {}: a scale byte is the E8M0 NaN code",
                row.layer,
                row.id
            );
        }
    }

    /// The scale is per 32 columns WITHIN a row: doubling the scale doubles the value it covers.
    #[test]
    fn the_scale_is_per_row_32_and_not_per_whole_row() {
        let weight = [0x38u8; HEAD_DIM]; // 1.0 in E4M3
        let mut scale = [127u8; SCALE_BYTES]; // 2^0
        scale[3] = 128; // 2^1 for columns 96..128 only
        let decoded = decode_row(&weight, &scale);
        for (index, value) in decoded.iter().enumerate() {
            let expected = if (96..128).contains(&index) { 2.0 } else { 1.0 };
            assert_eq!(*value, expected, "column {index} used the wrong scale");
        }
        // A scale change on one block must not move a value in another block.
        let mut other = [0x38u8; HEAD_DIM];
        other[255] = 0x38;
        assert_eq!(decode_row(&other, &scale)[255], 1.0);
    }

    /// The signed square root is load-bearing: dropping the sign flips the gate below 0.5.
    #[test]
    fn the_signed_square_root_changes_the_answer() {
        let dim = 8;
        let stream = vec![1.0f32; dim];
        let key = vec![1.0f32; dim];
        let product = vec![1.0f32; dim];
        let gate_value = gate(&stream, &key, &product, dim, 1e-6);
        // stream_square = key_square = dim, dot = dim, rstd = 1/(1+eps) = 1, scaled by dim^-0.5, so
        // the dot is sqrt(dim) and the signed root takes sqrt(sqrt(dim)): the gate is sigmoid of
        // that (forward/engram.rs test the_gate_is_the_reference_normalised_dot_product).
        let expected = sigmoid((dim as f32).sqrt().sqrt());
        assert!(
            (gate_value - expected).abs() < 1e-5,
            "{gate_value} vs {expected}"
        );

        let mut negative = stream.clone();
        for value in negative.iter_mut() {
            *value = -1.0;
        }
        let flipped = gate(&negative, &key, &product, dim, 1e-6);
        assert!(
            flipped < 0.5,
            "a negative dot must gate below one half, got {flipped}"
        );
        assert!(
            (flipped - (1.0 - expected)).abs() < 1e-5,
            "the signed root is odd, so the negative gate is 1 - sigmoid(sqrt(sqrt(dim)))"
        );
        // The mutant that drops the sign: |dot| first, so both signs collapse to the same gate.
        let mutant = sigmoid((dim as f32).sqrt().sqrt());
        assert_eq!(mutant, expected);
        assert!(
            (mutant - flipped).abs() > 1e-3,
            "if the mutant and the real rule agreed, the sign would be decorative"
        );
    }

    /// A sentinel id means the position takes no part in the n-gram: no lookup, no write.
    #[test]
    fn a_sentinel_id_skips_the_lookup_and_the_write() {
        let dim = 4;
        let hc = 2;
        let mut x_hc = vec![1.0f32; hc * dim];
        let before = x_hc.clone();
        let ids = [1u32, HASH_SENTINEL];
        let mut fetched = 0usize;
        let mut rows = |_id: u32| {
            fetched += 1;
            vec![0.5f32; HEAD_DIM]
        };
        let result = head_ref(
            &mut x_hc,
            &ids,
            &mut rows,
            &vec![0.0f32; dim * (hc + 1) * ids.len() * HEAD_DIM],
            &vec![1.0f32; hc * dim],
            HeadShape {
                dim,
                hc,
                eps: 1e-6,
                head_dim: HEAD_DIM,
            },
        );
        assert!(result.is_none());
        assert_eq!(fetched, 0, "a sentinel must not reach the row source");
        assert_eq!(x_hc, before, "a sentinel must not write");
    }

    /// The write is gate x value per (copy, column), in place, on the real row layout.
    #[test]
    fn the_head_writes_gate_times_value() {
        let dim = 4;
        let hc = 2;
        let columns = 2;
        let fetched_width = columns * HEAD_DIM;
        let out_width = dim * (hc + 1);
        // A wkv that copies the first column's first dim values into the value and zero elsewhere.
        let mut wkv = vec![0.0f32; out_width * fetched_width];
        for column in 0..dim {
            // value lives after the hc keys, at hc * dim + column.
            wkv[(hc * dim + column) * fetched_width + column] = 1.0;
        }
        let row_a: [f32; HEAD_DIM] =
            std::array::from_fn(|i| if i < dim { i as f32 + 1.0 } else { 0.0 });
        let row_b = [0.0f32; HEAD_DIM];
        let ids = [1u32, 2u32];
        let mut fetched_rows = [row_a, row_b];
        let mut rows = |id: u32| {
            std::mem::replace(&mut fetched_rows[(id - 1) as usize], [0.0f32; HEAD_DIM]).to_vec()
        };
        let x_before: Vec<f32> = (0..hc * dim).map(|i| i as f32 / 10.0).collect();
        let mut x_hc = x_before.clone();
        let product = vec![1.0f32; hc * dim];
        let result = head_ref(
            &mut x_hc,
            &ids,
            &mut rows,
            &wkv,
            &product,
            HeadShape {
                dim,
                hc,
                eps: 1e-6,
                head_dim: HEAD_DIM,
            },
        );
        let gates = result.expect("two real ids");
        assert_eq!(gates.len(), hc);
        // The key half is all zeros, so key_square = 0 and rstd is finite; dot is zero, so the gate
        // is sigmoid(sqrt(1e-6)) - just above one half, and identical for both copies here.
        for (index, value) in x_hc.iter().enumerate() {
            let copy = index / dim;
            let column = index % dim;
            let value_vec = (column + 1) as f32;
            let expected = x_before[index] + gates[copy] * value_vec;
            assert!(
                (value - expected).abs() < 1e-5,
                "element {index}: {value} vs {expected}"
            );
        }
        assert!(
            gates.iter().all(|g| *g > 0.5 && *g < 0.5003),
            "a zero dot must gate to just above one half after the signed root: {gates:?}"
        );
    }

    /// The E8M0 and E4M3 edge codes decode to what the tree says they do.
    #[test]
    fn the_edge_codes_are_the_tree_semantics() {
        assert_eq!(e8m0_to_f32(127), 1.0);
        assert_eq!(e8m0_to_f32(120), 2.0f32.powi(-7));
        assert!(e8m0_to_f32(255).is_nan());
        assert_eq!(e4m3_to_f32(0x00), 0.0);
        assert_eq!(e4m3_to_f32(0x80), 0.0);
        assert!(
            e4m3_to_f32(0x80).is_sign_negative(),
            "0x80 is negative zero"
        );
        assert_eq!(e4m3_to_f32(0x38), 1.0);
        assert_eq!(e4m3_to_f32(0xB8), -1.0);
        assert_eq!(e4m3_to_f32(0x7E), 448.0);
        assert!(e4m3_to_f32(0x7F).is_nan(), "0x7F is the E4M3 NaN code");
        assert!(e4m3_to_f32(0xFF).is_nan());
        // The smallest subnormal is mantissa 1 * 2^-9.
        assert_eq!(e4m3_to_f32(0x01), 2.0f32.powi(-9));
    }
}
