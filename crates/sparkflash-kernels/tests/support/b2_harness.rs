//! Offline readers and pure helpers shared by examples/m7_layer0.rs and the CPU tests in
//! tests/b2_harness_offline.rs.
//!
//! The harness's checkpoint and v1-boundary readers, its released-router implementation and its
//! free-vs-forced ffn_out metric live here so they are compiled and exercised on the CPU by cargo
//! test, not only inside the device run. The device example includes this file with a #[path]
//! module, so there is exactly one implementation of each reader.

#![allow(dead_code)]

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use sparkflash_kernels::m4_quant::{fp8_e4m3_decode, fp8_e4m3_encode, ue8m0_decode};
use sparkflash_kernels::{mxfp4_ref, swiglu_ref};

pub fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

pub fn bytes_to_bf16_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| u16::from_le_bytes(*c))
        .map(half::bf16::from_bits)
        .map(half::bf16::to_f32)
        .collect()
}

/// Slice each row's [skip, skip + take) window into a padded output row.
///
/// The same rule examples/moe_slice.rs:162 uses for its rank-0 slice, extended with a column offset
/// so the other ranks of the intermediate can be addressed too. The pad tail stays zero: that is
/// what the kernel masks, and what the reference masks identically.
pub fn slice_and_pad_rows(
    src: &[u8],
    rows: usize,
    src_row: usize,
    skip: usize,
    take: usize,
    pad: usize,
) -> Vec<u8> {
    assert_eq!(src.len(), rows * src_row, "the slice source is whole rows");
    assert!(
        skip + take <= src_row,
        "the rank window fits inside a stored row"
    );
    assert!(take <= pad, "the pad stride is at least the slice");
    let mut out = vec![0u8; rows * pad];
    for row in 0..rows {
        let start = row * src_row + skip;
        out[row * pad..row * pad + take].copy_from_slice(&src[start..start + take]);
    }
    out
}

/// One MXFP4 gate/up GEMM row block: out[n] = the reduction of the E4M3-cast FP4 weight against a
/// quantised f32 activation, with the ue8m0 weight scale applied to each 32-value PARTIAL product.
/// This is the arithmetic b12x_mxfp4_moe_sm121 documents for the same weight format; the gate/up
/// projection itself has no device kernel in this build, which is why it is host-side here. It moved
/// into the shared module so the forced-operand verifier and its CPU tests use one implementation.
pub fn gate_up_rows(
    activation: &[f32],
    weight: &[u8],
    scales: &[u8],
    rows: usize,
    inner: usize,
) -> Vec<f32> {
    let blocks = inner / 32;
    assert_eq!(
        weight.len(),
        rows * inner / 2,
        "the packed FP4 weight is whole rows"
    );
    assert_eq!(
        scales.len(),
        rows * blocks,
        "the weight scale grid is whole rows"
    );
    assert_eq!(activation.len(), inner, "the activation is one whole row");
    let mut out = vec![0.0f32; rows];
    for n in 0..rows {
        let weight_row = &weight[n * (inner / 2)..(n + 1) * (inner / 2)];
        let scale_row = &scales[n * blocks..(n + 1) * blocks];
        let mut acc = 0.0f32;
        for (block, scale) in scale_row.iter().enumerate().take(blocks) {
            let mut partial = 0.0f32;
            for i in 0..32 {
                let index = block * 32 + i;
                // The FP4-to-FP8 cast through float is exact on every E2M1 code (kernel.py:538-540).
                let w = fp8_e4m3_decode(fp8_e4m3_encode(mxfp4_ref::fp4_load(weight_row, index)));
                partial += activation[index] * w;
            }
            acc += partial * ue8m0_decode(*scale);
        }
        out[n] = acc;
    }
    out
}

#[derive(Clone)]
pub struct Header {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub start: u64,
    pub end: u64,
}

/// Minimal safetensors header reader. The package has no JSON dependency and three fields are
/// needed, so the header text is scanned directly - and every value it extracts is asserted against
/// the dtype and shape the checkpoint is known to store, so a parse error cannot pass silently.
pub fn read_header(path: &Path) -> Result<HashMap<String, Header>, String> {
    let mut file = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes).map_err(|e| e.to_string())?;
    let header_len = u64::from_le_bytes(len_bytes);
    let mut text = vec![0u8; header_len as usize];
    file.read_exact(&mut text).map_err(|e| e.to_string())?;
    let text = String::from_utf8(text).map_err(|e| e.to_string())?;
    let data_start = 8 + header_len;

    let mut out = HashMap::new();
    let bytes = text.as_bytes();
    let mut i = 0usize;
    while let Some(rel) = text[i..].find("\"dtype\"") {
        let dtype_at = i + rel;
        // Walk back to the opening quote of the tensor name.
        let mut name_end = dtype_at;
        while name_end > 0 && bytes[name_end - 1] != b'"' {
            name_end -= 1;
        }
        let mut name_start = name_end.saturating_sub(2);
        while name_start > 0 && bytes[name_start - 1] != b'"' {
            name_start -= 1;
        }
        let name = text[name_start..name_end - 1].to_string();
        let dq = text[dtype_at..]
            .find(':')
            .map(|p| dtype_at + p)
            .ok_or("no dtype colon")?;
        let v0 = text[dq..]
            .find('"')
            .map(|p| dq + p + 1)
            .ok_or("no dtype quote")?;
        let v1 = text[v0..].find('"').map(|p| v0 + p).ok_or("no dtype end")?;
        let dtype = text[v0..v1].to_string();
        let shape_at = text[v1..]
            .find("\"shape\"")
            .map(|p| v1 + p)
            .ok_or("no shape")?;
        let s0 = text[shape_at..]
            .find('[')
            .map(|p| shape_at + p + 1)
            .ok_or("no shape open")?;
        let s1 = text[s0..]
            .find(']')
            .map(|p| s0 + p)
            .ok_or("no shape close")?;
        let shape: Vec<usize> = text[s0..s1]
            .split(',')
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.trim().parse::<usize>().map_err(|e| e.to_string()))
            .collect::<Result<_, _>>()?;
        let off_at = text[s1..]
            .find("data_offsets")
            .map(|p| s1 + p)
            .ok_or("no offsets")?;
        let o0 = text[off_at..]
            .find('[')
            .map(|p| off_at + p + 1)
            .ok_or("no offsets open")?;
        let o1 = text[o0..]
            .find(']')
            .map(|p| o0 + p)
            .ok_or("no offsets close")?;
        let offs: Vec<u64> = text[o0..o1]
            .split(',')
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.trim().parse::<u64>().map_err(|e| e.to_string()))
            .collect::<Result<_, _>>()?;
        if offs.len() != 2 {
            return Err(format!(
                "{name}: expected two data offsets, got {}",
                offs.len()
            ));
        }
        out.insert(
            name,
            Header {
                dtype,
                shape,
                start: data_start + offs[0],
                end: data_start + offs[1],
            },
        );
        i = o1;
    }
    Ok(out)
}

/// The released checkpoint, opened through its own shard index. Every header field used here is
/// asserted against the dtype and shape the checkpoint is known to store, so a parse error cannot
/// pass silently, and every read is bounded by the caller: a whole-tensor read of embed.weight
/// would be 1.3 GB this harness does not need.
pub struct Snapshot {
    dir: PathBuf,
    index_text: String,
    headers: HashMap<String, HashMap<String, Header>>,
    files: HashMap<String, File>,
}

impl Snapshot {
    pub fn open(dir: &Path) -> Result<Self, String> {
        let index_path = dir.join("model.safetensors.index.json");
        let index_text = std::fs::read_to_string(&index_path)
            .map_err(|e| format!("{}: {e}", index_path.display()))?;
        Ok(Self {
            dir: dir.to_path_buf(),
            index_text,
            headers: HashMap::new(),
            files: HashMap::new(),
        })
    }

    /// The shard that stores one tensor, from the shard index.
    fn shard_of(&self, name: &str) -> Result<String, String> {
        let key = format!("\"{name}\"");
        let at = self
            .index_text
            .find(&key)
            .ok_or_else(|| format!("{name} is not in the shard index"))?;
        let rest = &self.index_text[at + key.len()..];
        let colon = rest
            .find(':')
            .ok_or_else(|| format!("{name}: malformed index entry"))?;
        let quote0 = rest[colon..]
            .find('"')
            .ok_or_else(|| format!("{name}: malformed index entry"))?
            + colon
            + 1;
        let quote1 = rest[quote0..]
            .find('"')
            .ok_or_else(|| format!("{name}: malformed index entry"))?
            + quote0;
        Ok(rest[quote0..quote1].to_string())
    }

    /// Open a tensor's shard if it is not open, and return its header.
    pub fn load(&mut self, name: &str) -> Result<Header, String> {
        let shard = self.shard_of(name)?;
        if !self.headers.contains_key(&shard) {
            let path = self.dir.join(&shard);
            let header = read_header(&path)?;
            let file = File::open(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            self.headers.insert(shard.clone(), header);
            self.files.insert(shard.clone(), file);
        }
        self.headers[&shard]
            .get(name)
            .cloned()
            .ok_or_else(|| format!("{name} is not in {shard}"))
    }

    /// A whole tensor as bytes, with its dtype and shape asserted first.
    pub fn read_raw(
        &mut self,
        name: &str,
        dtype: &str,
        shape: &[usize],
    ) -> Result<Vec<u8>, String> {
        let header = self.load(name)?;
        if header.dtype != dtype {
            return Err(format!(
                "{name}: expected {dtype}, the checkpoint stores {}",
                header.dtype
            ));
        }
        if header.shape != shape {
            return Err(format!(
                "{name}: expected shape {shape:?}, the checkpoint stores {:?}",
                header.shape
            ));
        }
        let len = (header.end - header.start) as usize;
        let shard = self.shard_of(name)?;
        let file = self
            .files
            .get_mut(&shard)
            .ok_or_else(|| format!("{name}: shard {shard} is not open"))?;
        file.seek(SeekFrom::Start(header.start))
            .map_err(|e| format!("{name}: {e}"))?;
        let mut bytes = vec![0u8; len];
        file.read_exact(&mut bytes)
            .map_err(|e| format!("{name}: {e}"))?;
        Ok(bytes)
    }

    /// One tensor stored as F32.
    pub fn read_f32(&mut self, name: &str, shape: &[usize]) -> Result<Vec<f32>, String> {
        let bytes = self.read_raw(name, "F32", shape)?;
        Ok(bytes_to_f32(&bytes))
    }

    /// One tensor stored as F32 or BF16, widened to f32. The declared dtype decides.
    pub fn read_f32_or_bf16(&mut self, name: &str, shape: &[usize]) -> Result<Vec<f32>, String> {
        match self.load(name)?.dtype.as_str() {
            "F32" => self.read_f32(name, shape),
            "BF16" => {
                let bytes = self.read_raw(name, "BF16", shape)?;
                Ok(bytes_to_bf16_f32(&bytes))
            }
            other => Err(format!(
                "{name}: expected F32 or BF16, the checkpoint stores {other}"
            )),
        }
    }

    /// One row of a two-dimensional BF16 tensor as raw little-endian bytes, into a reused buffer.
    /// The CPU head check walks 129,280 rows this way: reading the whole tensor would need 1.32 GB
    /// beside a running vLLM, while one row is 10 KB.
    pub fn read_row_bf16_into(
        &mut self,
        name: &str,
        row: usize,
        cols: usize,
        out: &mut [u8],
    ) -> Result<(), String> {
        let header = self.load(name)?;
        if header.dtype != "BF16" {
            return Err(format!(
                "{name}: expected BF16, the checkpoint stores {}",
                header.dtype
            ));
        }
        if header.shape.len() != 2 || header.shape[1] != cols {
            return Err(format!(
                "{name}: expected two dimensions with {cols} columns, the checkpoint stores {:?}",
                header.shape
            ));
        }
        if row >= header.shape[0] {
            return Err(format!("{name}: row {row} is out of range"));
        }
        let len = cols * 2;
        if out.len() != len {
            return Err(format!(
                "{name}: row buffer is {} bytes, expected {len}",
                out.len()
            ));
        }
        let start = header.start + (row * len) as u64;
        let shard = self.shard_of(name)?;
        let file = self
            .files
            .get_mut(&shard)
            .ok_or_else(|| format!("{name}: shard {shard} is not open"))?;
        file.seek(SeekFrom::Start(start))
            .map_err(|e| format!("{name}: {e}"))?;
        file.read_exact(out).map_err(|e| format!("{name}: {e}"))?;
        Ok(())
    }

    /// One row of a two-dimensional BF16 tensor, without reading or holding the rest of it.
    pub fn read_row_bf16(
        &mut self,
        name: &str,
        row: usize,
        cols: usize,
    ) -> Result<Vec<f32>, String> {
        let header = self.load(name)?;
        if header.dtype != "BF16" {
            return Err(format!(
                "{name}: expected BF16, the checkpoint stores {}",
                header.dtype
            ));
        }
        if header.shape.len() != 2 || header.shape[1] != cols {
            return Err(format!(
                "{name}: expected two dimensions with {cols} columns, the checkpoint stores {:?}",
                header.shape
            ));
        }
        if row >= header.shape[0] {
            return Err(format!("{name}: row {row} is out of range"));
        }
        let mut bytes = vec![0u8; cols * 2];
        self.read_row_bf16_into(name, row, cols, &mut bytes)?;
        Ok(bytes_to_bf16_f32(&bytes))
    }
}

/// One tensor of a v1 phase-1 boundary row, as the manifest records it.
#[derive(Debug)]
pub struct V1Tensor {
    pub dtype: String,
    pub dims: Vec<usize>,
    pub offset: usize,
    pub len: usize,
}

/// A phase-1 host boundary v1 pair: the text manifest plus the blob it indexes.
///
/// The format is \`docs/build/phase1-boundary-table.md\` section 3: a header, then \`row N\` markers and
/// one \`tensor <row> <name> <dtype> <dims> <offset> <bytes>\` line per tensor. Stage 0 reads two
/// f32 tensors from row 0 and the coefficient tensors that row captured, so it needs a reader that
/// checks the pair's own consistency rather than trusting the offsets.
pub struct HostBoundaryTable {
    prefix: PathBuf,
    manifest: String,
    pub blob: Vec<u8>,
}

impl HostBoundaryTable {
    /// Open \`<prefix>.manifest\` and \`<prefix>.blob\`.
    pub fn open(prefix: &Path) -> Result<Self, String> {
        let manifest_path = prefix.with_extension("manifest");
        let blob_path = prefix.with_extension("blob");
        let manifest = std::fs::read_to_string(&manifest_path)
            .map_err(|e| format!("{}: {e}", manifest_path.display()))?;
        let blob =
            std::fs::read(&blob_path).map_err(|e| format!("{}: {e}", blob_path.display()))?;
        if !manifest.starts_with("# phase1-boundary v1") {
            return Err(format!(
                "{}: not a phase1-boundary v1 manifest",
                manifest_path.display()
            ));
        }
        Ok(Self {
            prefix: prefix.to_path_buf(),
            manifest,
            blob,
        })
    }

    /// The tensor \`name\` of \`row\`, from its manifest line.
    pub fn tensor(&self, row: usize, name: &str) -> Result<V1Tensor, String> {
        let prefix = format!("tensor {row} {name} ");
        let line = self
            .manifest
            .lines()
            .find(|line| line.starts_with(&prefix))
            .ok_or_else(|| format!("{}: no tensor {name} in row {row}", self.prefix.display()))?;
        let mut fields = line.split_whitespace().skip(3);
        let dtype = fields
            .next()
            .ok_or_else(|| format!("row {row} {name}: no dtype"))?
            .to_string();
        let dims: Vec<usize> = fields
            .next()
            .ok_or_else(|| format!("row {row} {name}: no shape"))?
            .split(',')
            .map(|value| value.parse::<usize>().map_err(|e| e.to_string()))
            .collect::<Result<_, _>>()?;
        let offset = fields
            .next()
            .ok_or_else(|| format!("row {row} {name}: no offset"))?
            .parse::<usize>()
            .map_err(|e| e.to_string())?;
        let len = fields
            .next()
            .ok_or_else(|| format!("row {row} {name}: no length"))?
            .parse::<usize>()
            .map_err(|e| e.to_string())?;
        let end = offset.checked_add(len).ok_or_else(|| {
            format!("row {row} {name}: offset {offset} plus length {len} overflows usize")
        })?;
        if end > self.blob.len() {
            let blob = self.blob.len();
            return Err(format!(
                "row {row} {name}: [{offset}, {end}) is outside the {blob}-byte blob"
            ));
        }
        Ok(V1Tensor {
            dtype,
            dims,
            offset,
            len,
        })
    }

    /// The tensor's bytes.
    pub fn bytes(&self, row: usize, name: &str) -> Result<Vec<u8>, String> {
        let tensor = self.tensor(row, name)?;
        Ok(self.blob[tensor.offset..tensor.offset + tensor.len].to_vec())
    }

    /// The tensor's bytes as a u8 slice, with its manifest dtype checked to be a u8 format.
    pub fn u8(&self, row: usize, name: &str) -> Result<Vec<u8>, String> {
        let tensor = self.tensor(row, name)?;
        if !tensor.dtype.starts_with("u8") {
            return Err(format!(
                "row {row} {name}: expected a u8 format, the manifest says {}",
                tensor.dtype
            ));
        }
        let elements: usize = tensor.dims.iter().product();
        let bytes = self.bytes(row, name)?;
        if bytes.len() != elements {
            return Err(format!(
                "row {row} {name}: {elements} elements by shape, {} bytes",
                bytes.len()
            ));
        }
        Ok(bytes)
    }

    /// The tensor as f32 values, with its manifest dtype and shape checked.
    pub fn f32(&self, row: usize, name: &str) -> Result<Vec<f32>, String> {
        let tensor = self.tensor(row, name)?;
        if tensor.dtype != "f32" {
            return Err(format!(
                "row {row} {name}: expected f32, the manifest says {}",
                tensor.dtype
            ));
        }
        let values = bytes_to_f32(&self.bytes(row, name)?);
        let elements: usize = tensor.dims.iter().product();
        if values.len() != elements {
            return Err(format!(
                "row {row} {name}: {elements} elements by shape, {} values by dtype",
                values.len()
            ));
        }
        Ok(values)
    }
}

/// softplus(x) with beta 1 and PyTorch's threshold of 20 (model.py:816-817), the router's inner
/// function: the router's score function is the square root of this.
const SOFTPLUS_THRESHOLD: f32 = 20.0;

fn softplus(x: f32) -> f32 {
    if x > SOFTPLUS_THRESHOLD {
        x
    } else {
        x.exp().ln_1p()
    }
}

/// The router's score function for text_config.scoring_func "sqrtsoftplus".
fn sqrtsoftplus_router(x: f32) -> f32 {
    softplus(x).sqrt()
}

/// The router's parameters, so the released-router implementation can be compiled and tested
/// without the device harness's constants.
#[derive(Clone, Copy, Debug)]
pub struct RouterParams {
    pub hidden: usize,
    pub top_k: usize,
    pub gate_temp: f32,
    pub norm_eps: f32,
    pub scale: f32,
}

/// The measured relationship between the free-running and host-routing-forced ffn_out tensors.
///
/// pointwise_max_abs is the actual discrete contribution: the maximum of |free(i) - forced(i)|
/// over the row, with the signed difference at that index. free_host_max_abs and
/// forced_host_max_abs are the two host errors, kept separate: their difference is a change in max
/// error, NOT the flip's max-abs contribution, and this type never lets a caller label it as one.
#[derive(Debug, Clone)]
pub struct FreeForcedMetrics {
    pub pointwise_max_abs: f32,
    pub at: usize,
    pub signed_at: f32,
    pub free_host_max_abs: f32,
    pub free_host_at: usize,
    pub forced_host_max_abs: f32,
    pub forced_host_at: usize,
    pub all_equal: bool,
    pub differing_points: usize,
}

/// Compute the free-vs-forced metric. All three slices must have the same length.
pub fn free_forced_metrics(
    free: &[f32],
    forced: &[f32],
    host: &[f32],
) -> Result<FreeForcedMetrics, String> {
    if free.is_empty() {
        return Err("free_forced_metrics: empty tensors".to_string());
    }
    if free.len() != forced.len() || free.len() != host.len() {
        return Err(format!(
            "free_forced_metrics: lengths {} free, {} forced, {} host differ",
            free.len(),
            forced.len(),
            host.len()
        ));
    }
    let mut pointwise_max_abs = 0.0f32;
    let mut at = 0usize;
    let mut differing_points = 0usize;
    let mut all_equal = true;
    for (index, (free, forced)) in free.iter().zip(forced).enumerate() {
        let delta = (free - forced).abs();
        if !delta.is_finite() || delta > pointwise_max_abs {
            pointwise_max_abs = if delta.is_finite() {
                delta
            } else {
                f32::INFINITY
            };
            at = index;
        }
        if free.to_bits() != forced.to_bits() {
            all_equal = false;
            differing_points += 1;
        }
    }
    let mut free_host_max_abs = 0.0f32;
    let mut free_host_at = 0usize;
    let mut forced_host_max_abs = 0.0f32;
    let mut forced_host_at = 0usize;
    for (index, ((free, forced), host)) in free.iter().zip(forced).zip(host).enumerate() {
        let free_delta = (free - host).abs();
        if !free_delta.is_finite() || free_delta > free_host_max_abs {
            free_host_max_abs = if free_delta.is_finite() {
                free_delta
            } else {
                f32::INFINITY
            };
            free_host_at = index;
        }
        let forced_delta = (forced - host).abs();
        if !forced_delta.is_finite() || forced_delta > forced_host_max_abs {
            forced_host_max_abs = if forced_delta.is_finite() {
                forced_delta
            } else {
                f32::INFINITY
            };
            forced_host_at = index;
        }
    }
    Ok(FreeForcedMetrics {
        pointwise_max_abs,
        at,
        signed_at: free[at] - forced[at],
        free_host_max_abs,
        free_host_at,
        forced_host_max_abs,
        forced_host_at,
        all_equal,
        differing_points,
    })
}

/// Combine a routed-combine bf16 output with the shared-expert f32 output the way the reference's
/// seam S9 does (elementwise add in the reference's order). Both the free path and the forced
/// ffn_out verification go through this one function, so "the same combination" is structural.
pub fn combine_bf16_routed_with_shared(
    routed_bits: &[u16],
    shared_out: &[f32],
) -> Result<Vec<f32>, String> {
    if routed_bits.is_empty() {
        return Err("combine ffn_out: empty routed output".to_string());
    }
    if routed_bits.len() != shared_out.len() {
        return Err(format!(
            "combine ffn_out: {} routed values vs {} shared values",
            routed_bits.len(),
            shared_out.len()
        ));
    }
    Ok(routed_bits
        .iter()
        .zip(shared_out)
        .map(|(bits, shared)| half::bf16::from_bits(*bits).to_f32() + shared)
        .collect())
}

/// Tie the forced ffn_out tensor to the forced routed-combine output, and RETURN the verified
/// recomputation.
///
/// Recomputes ffn_out_forced from the forced routed bf16 bits and the shared-expert output with
/// combine_bf16_routed_with_shared (the free path's combination) and requires the tensor the device
/// launch produced to be bit-identical to it. The caller must use the returned tensor for every
/// reported number (documented as "the asserted-identical recomputation of the forced output"): a
/// copy inserted into the device tensor after this call cannot change any reported metric, and a
/// mutation that assigns the free ffn_out (or copies it per column) fails here with the loud abort.
pub fn verify_forced_ffn_out(
    forced_routed_bits: &[u16],
    shared_out: &[f32],
    ffn_out_forced: &[f32],
) -> Result<Vec<f32>, String> {
    let expected = combine_bf16_routed_with_shared(forced_routed_bits, shared_out)?;
    if expected.len() != ffn_out_forced.len() {
        return Err(format!(
            "forced ffn_out provenance: {} expected values vs {} declared",
            expected.len(),
            ffn_out_forced.len()
        ));
    }
    for (index, (expected, declared)) in expected.iter().zip(ffn_out_forced).enumerate() {
        if expected.to_bits() != declared.to_bits() {
            return Err(format!(
                "forced ffn_out provenance: index {index}: the declared forced tensor is {declared:.9e} ({:#010x}) but the forced routed bits plus shared output give {expected:.9e} ({:#010x}); the forced control is not diffing the forced run's own output",
                declared.to_bits(),
                expected.to_bits()
            ));
        }
    }
    Ok(expected)
}

/// The forced-routing control's verdict.
///
/// routing_identical means the ordered top-k IDS are equal: that is the routing choice. Weights can
/// still differ when the ids are equal (the two inputs are different f32 rows), and then the routed
/// bf16 bytes may legitimately round to the same output; only a difference in the IDS (a flip)
/// demands a nonzero routed difference.
#[derive(Debug, Clone)]
pub struct ForcedControlVerdict {
    pub routing_identical: bool,
    pub weights_changed: bool,
    pub routed_identical: bool,
}

/// Classify one forced-routing control run and enforce determinism. Errors if the ids differ but
/// the routed bytes are identical (a copy of the free path or a broken launch), and errors if the
/// ids and weights are both identical but the routed bytes differ (identical routing must reproduce
/// the free path's bytes exactly, so a different result is nondeterminism). When the ids are
/// identical and the weights changed, the routed bytes may be equal (the change rounded away) or
/// differ; only the ordered ids define the routing choice.
pub fn forced_control_verdict(
    device_ids: &[u32],
    host_ids: &[u32],
    device_weights: &[f32],
    host_weights: &[f32],
    routed_identical: bool,
) -> Result<ForcedControlVerdict, String> {
    if device_ids.len() != host_ids.len() || device_weights.len() != host_weights.len() {
        return Err(format!(
            "forced-routing control: lengths differ ({} device ids / {} host ids, {} device weights / {} host weights)",
            device_ids.len(),
            host_ids.len(),
            device_weights.len(),
            host_weights.len()
        ));
    }
    if device_ids.is_empty() {
        return Err("forced-routing control: empty routing".to_string());
    }
    let routing_identical = device_ids == host_ids;
    let weights_changed = routing_identical && device_weights != host_weights;
    if !routing_identical && routed_identical {
        return Err(format!(
            "forced-routing control: the routing choices differ (device ids {device_ids:?} vs host ids {host_ids:?}) but the routed-combine bytes are identical; a flip must produce a nonzero difference, so the forced launch cannot have used the host routing"
        ));
    }
    if routing_identical && !weights_changed && !routed_identical {
        return Err(format!(
            "forced-routing control: the routing ids and weights are identical ({device_ids:?}) but the routed-combine bytes differ; identical routing must reproduce the free path's bytes exactly, so this launch is nondeterministic and cannot certify the control"
        ));
    }
    Ok(ForcedControlVerdict {
        routing_identical,
        weights_changed,
        routed_identical,
    })
}

/// Verify that every routed rank's weight and scale bytes are exactly the expected bytes, in the
/// requested order. A forced launch calls this with the expected bytes re-read from the checkpoint
/// for the requested routing, before any launch: it catches an operand builder that ignores the
/// ids/weights it was handed, which comparing local variables after the launch would not.
pub fn verify_routed_operand_bytes(
    operand_w: &[Vec<u8>],
    operand_ws: &[Vec<u8>],
    expected_w: &[Vec<u8>],
    expected_ws: &[Vec<u8>],
) -> Result<(), String> {
    if operand_w.len() != expected_w.len() || operand_ws.len() != expected_ws.len() {
        return Err(format!(
            "routed operand provenance: rank counts differ ({} weight / {} scale operand ranks vs {} / {} expected)",
            operand_w.len(),
            operand_ws.len(),
            expected_w.len(),
            expected_ws.len()
        ));
    }
    if operand_w.is_empty() {
        return Err("routed operand provenance: no ranks".to_string());
    }
    for rank in 0..operand_w.len() {
        if operand_w[rank] != expected_w[rank] {
            return Err(format!(
                "routed operand provenance: rank {rank}: weight bytes differ from the bytes the requested routing selects ({} vs {} B)",
                operand_w[rank].len(),
                expected_w[rank].len()
            ));
        }
        if operand_ws[rank] != expected_ws[rank] {
            return Err(format!(
                "routed operand provenance: rank {rank}: scale bytes differ from the bytes the requested routing selects ({} vs {} B)",
                operand_ws[rank].len(),
                expected_ws[rank].len()
            ));
        }
    }
    Ok(())
}

/// The tensor geometry the forced-operand verifier needs, passed explicitly so the verifier (which
/// lives in this shared module) can be exercised on synthetic small tensors. The device example
/// supplies the checkpoint's real constants.
#[derive(Debug, Clone, Copy)]
pub struct ForcedOperandGeometry {
    pub hidden: usize,
    pub intermediate: usize,
    pub width: usize,
    pub width_padded: usize,
    pub top_k: usize,
    pub ranks: usize,
}

/// A checkpoint reader that returns one whole tensor's bytes, with its dtype and shape checked. The
/// trait exists so the full pre-launch verifier can be tested with a synthetic in-memory source as
/// well as the real Snapshot.
pub trait RawTensorSource {
    fn read_raw(&mut self, name: &str, dtype: &str, shape: &[usize]) -> Result<Vec<u8>, String>;
}

impl RawTensorSource for Snapshot {
    fn read_raw(&mut self, name: &str, dtype: &str, shape: &[usize]) -> Result<Vec<u8>, String> {
        Snapshot::read_raw(self, name, dtype, shape)
    }
}

/// Verify the forced-routing operands against a fresh read of the checkpoint, before any launch.
///
/// Covers every slot and every rank, not a spot check:
/// - the per-rank w2 weight and scale slices must be exactly the requested experts' bytes, in the
///   requested id order (re-read and re-sliced here);
/// - for each requested slot, the slot's w1/w3 weight and scale tensors are re-read and re-run
///   through the same gate/up GEMM and SwiGLU the operand builder uses, and for every rank the
///   resulting activation codes and scales must be bit-identical to the operand's slot/rank bytes.
///
/// A routed_operands that ignored its ids or weights fails here, before the launches.
#[allow(clippy::too_many_arguments)]
pub fn verify_forced_operand_provenance<S: RawTensorSource>(
    source: &mut S,
    prefix: &str,
    block_activation: &[f32],
    requested_ids: &[u32],
    requested_weights: &[f32],
    operand_w: &[Vec<u8>],
    operand_ws: &[Vec<u8>],
    operand_act_fp8: &[Vec<u8>],
    operand_act_scales: &[Vec<u8>],
    geometry: &ForcedOperandGeometry,
) -> Result<(), String> {
    let ForcedOperandGeometry {
        hidden,
        intermediate,
        width,
        width_padded,
        top_k,
        ranks,
    } = *geometry;
    if hidden == 0 || width == 0 || width_padded == 0 || top_k == 0 || ranks == 0 {
        return Err("forced operand provenance: empty geometry".to_string());
    }
    if width % 32 != 0 || width_padded % 32 != 0 || width_padded < width {
        return Err(format!(
            "forced operand provenance: width {width} / padded width {width_padded} are not whole 32-value blocks (padded >= width)"
        ));
    }
    if intermediate != width * ranks {
        return Err(format!(
            "forced operand provenance: intermediate {intermediate} is not {ranks} ranks x {width} width"
        ));
    }
    if requested_ids.len() != top_k || requested_weights.len() != top_k {
        return Err(format!(
            "forced operand provenance: {} ids / {} weights, expected {top_k} each",
            requested_ids.len(),
            requested_weights.len()
        ));
    }
    if block_activation.len() != hidden {
        return Err(format!(
            "forced operand provenance: block activation has {} values, expected {hidden}",
            block_activation.len()
        ));
    }
    if operand_w.len() != ranks
        || operand_ws.len() != ranks
        || operand_act_fp8.len() != ranks
        || operand_act_scales.len() != ranks
    {
        return Err(format!(
            "forced operand provenance: operand rank counts {}/{}/{}/{} for w/ws/act/act-scale, expected {ranks} each",
            operand_w.len(),
            operand_ws.len(),
            operand_act_fp8.len(),
            operand_act_scales.len()
        ));
    }
    let packed_row = width_padded / 2;
    let scale_row = width_padded / 32;
    let act_scale_len = width / 32;
    let expert_in_packed = hidden / 2;
    let expert_in_scales = hidden / 32;
    let in_packed = intermediate / 2;
    let in_scales = intermediate / 32;
    let expected_w_rank_len = top_k * hidden * packed_row;
    let expected_ws_rank_len = top_k * hidden * scale_row;
    for rank in 0..ranks {
        if operand_w[rank].len() != expected_w_rank_len {
            return Err(format!(
                "forced operand provenance: rank {rank} weight buffer is {} B, expected {expected_w_rank_len}",
                operand_w[rank].len()
            ));
        }
        if operand_ws[rank].len() != expected_ws_rank_len {
            return Err(format!(
                "forced operand provenance: rank {rank} scale buffer is {} B, expected {expected_ws_rank_len}",
                operand_ws[rank].len()
            ));
        }
        if operand_act_fp8[rank].len() != top_k * width_padded {
            return Err(format!(
                "forced operand provenance: rank {rank} activation buffer is {} B, expected {}",
                operand_act_fp8[rank].len(),
                top_k * width_padded
            ));
        }
        if operand_act_scales[rank].len() != top_k * scale_row {
            return Err(format!(
                "forced operand provenance: rank {rank} activation-scale buffer is {} B, expected {}",
                operand_act_scales[rank].len(),
                top_k * scale_row
            ));
        }
    }

    // (1) every rank's w2 weight and scale slices, re-read from the requested experts and re-sliced.
    let mut w2_all: Vec<Vec<u8>> = Vec::with_capacity(top_k);
    let mut s2_all: Vec<Vec<u8>> = Vec::with_capacity(top_k);
    for expert in requested_ids {
        let base = format!("{prefix}.ffn.experts.{expert}");
        w2_all.push(source.read_raw(&format!("{base}.w2.weight"), "I8", &[hidden, in_packed])?);
        s2_all.push(source.read_raw(
            &format!("{base}.w2.scale"),
            "F8_E8M0",
            &[hidden, in_scales],
        )?);
    }
    let mut expected_w: Vec<Vec<u8>> = Vec::with_capacity(ranks);
    let mut expected_ws: Vec<Vec<u8>> = Vec::with_capacity(ranks);
    for rank in 0..ranks {
        let mut weight = Vec::with_capacity(expected_w_rank_len);
        let mut scale = Vec::with_capacity(expected_ws_rank_len);
        for expert in 0..top_k {
            weight.extend_from_slice(&slice_and_pad_rows(
                &w2_all[expert],
                hidden,
                in_packed,
                rank * (width / 2),
                width / 2,
                packed_row,
            ));
            scale.extend_from_slice(&slice_and_pad_rows(
                &s2_all[expert],
                hidden,
                in_scales,
                rank * act_scale_len,
                act_scale_len,
                scale_row,
            ));
        }
        expected_w.push(weight);
        expected_ws.push(scale);
    }
    verify_routed_operand_bytes(operand_w, operand_ws, &expected_w, &expected_ws)?;

    // (2) every slot and every rank's activation operand, recomputed from the requested id and
    // weight through the same gate/up GEMM, SwiGLU and reference activation quantiser.
    for slot in 0..top_k {
        let base = format!("{prefix}.ffn.experts.{}", requested_ids[slot]);
        let w1 = source.read_raw(
            &format!("{base}.w1.weight"),
            "I8",
            &[intermediate, expert_in_packed],
        )?;
        let s1 = source.read_raw(
            &format!("{base}.w1.scale"),
            "F8_E8M0",
            &[intermediate, expert_in_scales],
        )?;
        let w3 = source.read_raw(
            &format!("{base}.w3.weight"),
            "I8",
            &[intermediate, expert_in_packed],
        )?;
        let s3 = source.read_raw(
            &format!("{base}.w3.scale"),
            "F8_E8M0",
            &[intermediate, expert_in_scales],
        )?;
        let gate = gate_up_rows(block_activation, &w1, &s1, intermediate, hidden);
        let up = gate_up_rows(block_activation, &w3, &s3, intermediate, hidden);
        let activation: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(gate, up)| swiglu_ref::swiglu(*gate, *up, swiglu_ref::SWIGLU_LIMIT))
            .collect();
        for rank in 0..ranks {
            let (codes, scales) = mxfp4_ref::routed_activation_operand(
                &activation[rank * width..(rank + 1) * width],
                requested_weights[slot],
            );
            if codes.len() != width || scales.len() != act_scale_len {
                return Err(format!(
                    "forced operand provenance: slot {slot} / rank {rank}: the reference operand is {} codes / {} scales, expected {width} / {act_scale_len}",
                    codes.len(),
                    scales.len()
                ));
            }
            let code_offset = slot * width_padded;
            if operand_act_fp8[rank][code_offset..code_offset + width] != codes {
                return Err(format!(
                    "forced operand provenance: slot {slot} / rank {rank}: activation codes differ from the reference operand for id {} and weight {}",
                    requested_ids[slot], requested_weights[slot]
                ));
            }
            let scale_offset = slot * scale_row;
            if operand_act_scales[rank][scale_offset..scale_offset + act_scale_len] != scales {
                return Err(format!(
                    "forced operand provenance: slot {slot} / rank {rank}: activation scales differ from the reference operand for id {} and weight {}",
                    requested_ids[slot], requested_weights[slot]
                ));
            }
        }
    }
    Ok(())
}

/// One token's routing decision, from the engine reference's Gate.forward (model.py:809-827):
/// scores = gate.weight @ x / gate_temp, then sqrtsoftplus; the noaux bias decides ONLY the
/// selection (model.py:818-823) and the weights come from the unbiased scores, normalised when
/// norm_topk_prob and top-k > 1, then multiplied by route_scale (model.py:824-826).
pub fn route_token_with(
    x: &[f32],
    gate_weight: &[f32],
    bias: &[f32],
    candidates: &[u32],
    params: RouterParams,
) -> (Vec<u32>, Vec<f32>) {
    let scores: Vec<f32> = candidates
        .iter()
        .map(|expert| {
            let expert = *expert as usize;
            let row = &gate_weight[expert * params.hidden..(expert + 1) * params.hidden];
            let mut acc = 0.0f32;
            for (value, weight) in x.iter().zip(row) {
                acc += value * weight;
            }
            sqrtsoftplus_router(acc / params.gate_temp)
        })
        .collect();
    let mut taken = vec![false; candidates.len()];
    let mut selected: Vec<usize> = Vec::with_capacity(params.top_k);
    for _ in 0..params.top_k.min(candidates.len()) {
        let mut best: Option<usize> = None;
        for index in 0..candidates.len() {
            if taken[index] {
                continue;
            }
            let signal = scores[index] + bias[candidates[index] as usize];
            // Strict inequality keeps the first candidate on a tie, and the caller passes
            // candidates in ascending expert-id order, which is the engine's tie rule.
            match best {
                Some(current) if signal <= scores[current] + bias[candidates[current] as usize] => {
                }
                _ => best = Some(index),
            }
        }
        match best {
            Some(index) => {
                taken[index] = true;
                selected.push(index);
            }
            None => break,
        }
    }
    let mut weights: Vec<f32> = selected.iter().map(|index| scores[*index]).collect();
    if weights.len() > 1 {
        let sum: f32 = weights.iter().sum();
        let denominator = sum + params.norm_eps;
        for weight in weights.iter_mut() {
            *weight /= denominator;
        }
    }
    for weight in weights.iter_mut() {
        *weight *= params.scale;
    }
    (
        selected.iter().map(|index| candidates[*index]).collect(),
        weights,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("b2_harness_test_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    fn write_safetensors(dir: &Path) {
        // Two tensors in one shard: "w" BF16 [2, 3] at bytes [0, 12), "b" F32 [3] at [12, 24).
        let json = concat!(
            "{\"w\":{\"dtype\":\"BF16\",\"shape\":[2,3],\"data_offsets\":[0,12]},",
            "\"b\":{\"dtype\":\"F32\",\"shape\":[3],\"data_offsets\":[12,24]}}"
        );
        let mut payload = Vec::new();
        for value in [1.0f32, -2.0, 0.5, 3.0, -0.25, 0.0] {
            payload.extend_from_slice(&half::bf16::from_f32(value).to_bits().to_le_bytes());
        }
        for value in [0.5f32, -1.5, 2.0] {
            payload.extend_from_slice(&value.to_le_bytes());
        }
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(json.len() as u64).to_le_bytes());
        bytes.extend_from_slice(json.as_bytes());
        bytes.extend_from_slice(&payload);
        std::fs::write(dir.join("model-00001.safetensors"), bytes).expect("write shard");
        let index = concat!(
            "{\"metadata\":{\"total_size\":24},\"weight_map\":{",
            "\"w\":\"model-00001.safetensors\",\"b\":\"model-00001.safetensors\"}}"
        );
        std::fs::write(dir.join("model.safetensors.index.json"), index).expect("write index");
    }

    #[test]
    fn snapshot_reads_tensors_by_declared_dtype() {
        let dir = temp_dir("snapshot");
        write_safetensors(&dir);
        let mut snap = Snapshot::open(&dir).expect("open snapshot");
        assert_eq!(
            snap.read_f32("b", &[3]).expect("f32 read"),
            vec![0.5, -1.5, 2.0]
        );
        assert_eq!(
            snap.read_f32_or_bf16("w", &[2, 3]).expect("bf16 read"),
            vec![1.0, -2.0, 0.5, 3.0, -0.25, 0.0]
        );
        assert_eq!(
            snap.read_row_bf16("w", 1, 3).expect("bf16 row"),
            vec![3.0, -0.25, 0.0]
        );
        let mut into = vec![0u8; 6];
        snap.read_row_bf16_into("w", 0, 3, &mut into)
            .expect("bf16 row into");
        assert_eq!(bytes_to_bf16_f32(&into), vec![1.0, -2.0, 0.5]);
        let header = snap.load("w").expect("load w");
        assert_eq!(header.dtype, "BF16");
        assert_eq!(header.shape, vec![2, 3]);
        assert_eq!(header.end - header.start, 12);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn snapshot_rejects_a_mismatched_dtype_shape_or_row() {
        let dir = temp_dir("snapshot_errors");
        write_safetensors(&dir);
        let mut snap = Snapshot::open(&dir).expect("open snapshot");
        assert!(snap.read_f32("w", &[2, 3]).is_err(), "BF16 read as F32");
        assert!(snap.read_row_bf16("w", 2, 3).is_err(), "row past the shape");
        assert!(
            snap.read_row_bf16("missing", 0, 3).is_err(),
            "missing tensor"
        );
        let mut small = vec![0u8; 2];
        assert!(snap.read_row_bf16_into("w", 0, 3, &mut small).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn write_boundary_pair(dir: &Path) {
        std::fs::write(
            dir.join("row.manifest"),
            concat!(
                "# phase1-boundary v1\n",
                "row 0\n",
                "tensor 0 t f32 2,2 0 16\n",
                "tensor 0 q u8 3,1 16 3\n"
            ),
        )
        .expect("write manifest");
        let mut blob = Vec::new();
        for value in [1.0f32, 2.0, 3.0, 4.0] {
            blob.extend_from_slice(&value.to_le_bytes());
        }
        blob.extend_from_slice(&[7u8, 8, 9]);
        std::fs::write(dir.join("row.blob"), blob).expect("write blob");
    }

    #[test]
    fn host_boundary_table_reads_by_class_and_checks_bounds() {
        let dir = temp_dir("boundary");
        write_boundary_pair(&dir);
        let table = HostBoundaryTable::open(&dir.join("row")).expect("open pair");
        assert_eq!(
            table.f32(0, "t").expect("f32 tensor"),
            vec![1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(table.u8(0, "q").expect("u8 tensor"), vec![7, 8, 9]);
        assert_eq!(table.bytes(0, "q").expect("bytes").len(), 3);
        let tensor = table.tensor(0, "q").expect("tensor line");
        assert_eq!(tensor.dtype, "u8");
        assert_eq!(tensor.dims, vec![3, 1]);
        assert_eq!((tensor.offset, tensor.len), (16, 3));
        assert!(table.tensor(0, "missing").is_err(), "missing tensor");
        assert!(table.f32(1, "t").is_err(), "missing row");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn host_boundary_table_rejects_an_offset_that_overflows_usize() {
        let dir = temp_dir("boundary_overflow");
        write_boundary_pair(&dir);
        let prefix = dir.join("row");
        let mut manifest =
            std::fs::read_to_string(prefix.with_extension("manifest")).expect("read manifest");
        manifest.push_str(&format!("tensor 0 huge f32 2,2 {} 8\n", usize::MAX - 1));
        std::fs::write(prefix.with_extension("manifest"), manifest).expect("write manifest");
        let table = HostBoundaryTable::open(&prefix).expect("open pair");
        let error = table.tensor(0, "huge").expect_err("overflow must fail");
        assert!(
            error.contains("overflows usize"),
            "unexpected error: {error}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn host_boundary_table_rejects_an_out_of_bounds_tensor() {
        let dir = temp_dir("boundary_bounds");
        write_boundary_pair(&dir);
        let prefix = dir.join("row");
        let mut manifest =
            std::fs::read_to_string(prefix.with_extension("manifest")).expect("read manifest");
        manifest.push_str("tensor 0 bad f32 2,2 20 8\n");
        std::fs::write(prefix.with_extension("manifest"), manifest).expect("write manifest");
        let table = HostBoundaryTable::open(&prefix).expect("open pair");
        let error = table.tensor(0, "bad").expect_err("out of bounds must fail");
        assert!(error.contains("outside the 19-byte blob"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn router_params() -> RouterParams {
        RouterParams {
            hidden: 2,
            top_k: 2,
            gate_temp: 1.0,
            norm_eps: 1e-20,
            scale: 1.5,
        }
    }

    #[test]
    fn route_token_selects_by_score_and_normalises_the_weights() {
        let x = [1.0f32, 0.0];
        let gate = vec![1.0, 0.0, 0.0, 1.0, -1.0, 0.0];
        let candidates = [0u32, 1, 2];
        let (ids, weights) = route_token_with(&x, &gate, &[0.0; 3], &candidates, router_params());
        assert_eq!(ids, vec![0, 1]);
        assert!(weights[0] > weights[1]);
        assert!((weights.iter().sum::<f32>() - 1.5).abs() < 1e-5);
    }

    #[test]
    fn route_token_selection_follows_the_bias_but_the_weights_do_not() {
        let x = [1.0f32, 0.0];
        let gate = vec![1.0, 0.0, 0.0, 1.0, -1.0, 0.0];
        let candidates = [0u32, 1, 2];
        let (ids, _) = route_token_with(&x, &gate, &[0.0, 5.0, 0.0], &candidates, router_params());
        assert_eq!(ids[0], 1);
    }

    #[test]
    fn route_token_breaks_ties_by_input_order() {
        let x = [1.0f32, 1.0];
        let gate = vec![1.0, 0.0, 1.0, 0.0];
        let candidates = [0u32, 1];
        let (ids, _) = route_token_with(&x, &gate, &[0.0; 2], &candidates, router_params());
        assert_eq!(ids, vec![0, 1]);
    }

    #[test]
    fn forced_control_verdict_covers_the_allowed_and_rejected_cases() {
        let ids = [250u32, 209];
        let weights = [0.75f32, 0.25];
        let same = forced_control_verdict(&ids, &ids, &weights, &weights, true).expect("identical");
        assert!(same.routing_identical);
        assert!(!same.weights_changed);
        assert!(same.routed_identical);
        // Equal ids, different weights, rounded-same routed bytes: allowed, and weights_changed says so.
        let weight_changed =
            forced_control_verdict(&ids, &ids, &weights, &[0.7500001, 0.2499999], true)
                .expect("equal ids with changed weights");
        assert!(weight_changed.routing_identical);
        assert!(weight_changed.weights_changed);
        // Different ids (a flip) with different routed bytes: allowed.
        let flip = forced_control_verdict(&ids, &[250, 210], &weights, &weights, false)
            .expect("ids differ and bytes differ");
        assert!(!flip.routing_identical);
        assert!(!flip.weights_changed);
        // Different ids but identical routed bytes: a copy of the free path, rejected.
        let copied = forced_control_verdict(&ids, &[250, 210], &weights, &weights, true);
        assert!(
            copied.is_err(),
            "a flip with identical routed bytes must fail"
        );
        assert!(forced_control_verdict(&ids, &ids[..1], &weights, &weights, true).is_err());
        // Identical ids AND identical weights must reproduce the bytes: different bytes are
        // nondeterminism, not a legitimate case.
        assert!(
            forced_control_verdict(&[1u32, 2], &[1, 2], &[0.5f32, 0.5], &[0.5, 0.5], false)
                .is_err(),
            "identical ids and weights with different routed bytes must fail"
        );
        let deterministic =
            forced_control_verdict(&[1u32, 2], &[1, 2], &[0.5f32, 0.5], &[0.5, 0.5], true)
                .expect("identical ids and weights with equal bytes");
        assert!(deterministic.routing_identical);
        assert!(!deterministic.weights_changed);
        // Ids equal, weights changed, routed bytes DIFFER: allowed (the changed weight need not
        // round away), and weights_changed records why the bytes may differ.
        let weight_changed_differing_bytes =
            forced_control_verdict(&[1u32, 2], &[1, 2], &[0.5f32, 0.5], &[0.25, 0.75], false)
                .expect("identical ids with changed weights and different bytes");
        assert!(weight_changed_differing_bytes.routing_identical);
        assert!(weight_changed_differing_bytes.weights_changed);
        assert!(!weight_changed_differing_bytes.routed_identical);
    }

    #[test]
    fn forced_ffn_out_tie_returns_the_verified_recomputation_and_metrics_use_it() {
        let forced_bits = [0x3f80u16, 0x4000]; // 1.0, 2.0
        let shared = [0.5f32, -0.5];
        let device_forced =
            combine_bf16_routed_with_shared(&forced_bits, &shared).expect("forced combine");
        assert_eq!(device_forced, vec![1.5, 1.5]);
        let returned = verify_forced_ffn_out(&forced_bits, &shared, &device_forced)
            .expect("the device tensor is the forced combination");
        // The returned tensor is bit-identical to the device tensor.
        assert_eq!(returned.len(), device_forced.len());
        for (returned, device) in returned.iter().zip(&device_forced) {
            assert_eq!(returned.to_bits(), device.to_bits());
        }
        // The metric consumes the RETURNED tensor: a copy inserted into the device variable after
        // the assertion cannot change the reported numbers. Contrast: feeding the polluted copy to
        // the metric would change them, so this test would catch a metric that reads the variable.
        let free = [1.0f32, 2.0];
        let host = [0.0f32, 0.0];
        let before = free_forced_metrics(&free, &returned, &host).expect("metrics from returned");
        let mut polluted_device = device_forced.clone();
        polluted_device.copy_from_slice(&free);
        let after = free_forced_metrics(&free, &returned, &host).expect("metrics from returned");
        assert_eq!(
            before.pointwise_max_abs.to_bits(),
            after.pointwise_max_abs.to_bits()
        );
        assert_eq!(before.differing_points, after.differing_points);
        let contaminated =
            free_forced_metrics(&free, &polluted_device, &host).expect("metrics from the copy");
        assert_ne!(
            before.pointwise_max_abs.to_bits(),
            contaminated.pointwise_max_abs.to_bits(),
            "the returned tensor, not the device variable, is what the metric consumes"
        );
        // A mutation that assigns the free path's tensor (or copies it per column) must fail.
        let free_bits = [0x4040u16, 0x4080]; // 3.0, 4.0
        let free_forced =
            combine_bf16_routed_with_shared(&free_bits, &shared).expect("free combine");
        let error = verify_forced_ffn_out(&forced_bits, &shared, &free_forced)
            .expect_err("the free tensor must not pass as the forced one");
        assert!(
            error.contains("forced ffn_out provenance"),
            "unexpected error: {error}"
        );
        let mut mixed = device_forced.clone();
        mixed[1] = free_forced[1];
        assert!(
            verify_forced_ffn_out(&forced_bits, &shared, &mixed).is_err(),
            "a per-column copy must fail"
        );
    }

    #[test]
    fn routed_operand_bytes_verification_accepts_exact_and_rejects_wrong_bytes() {
        let operand_w = vec![vec![1u8, 2, 3], vec![4u8, 5, 6]];
        let operand_ws = vec![vec![7u8, 8], vec![9u8, 10]];
        verify_routed_operand_bytes(
            &operand_w,
            &operand_ws,
            &operand_w.clone(),
            &operand_ws.clone(),
        )
        .expect("identical bytes");
        assert!(
            verify_routed_operand_bytes(
                &operand_w,
                &operand_ws,
                &operand_w,
                &vec![vec![0u8; 2]; 2]
            )
            .is_err(),
            "different scale bytes must fail"
        );
        let wrong_rank = vec![vec![1u8, 2, 3], vec![4u8, 5, 9]];
        assert!(
            verify_routed_operand_bytes(&operand_w, &operand_ws, &wrong_rank, &operand_ws).is_err(),
            "different weight bytes must fail"
        );
        assert!(
            verify_routed_operand_bytes(&operand_w[..1], &operand_ws, &operand_w, &operand_ws)
                .is_err(),
            "rank count mismatch must fail"
        );
    }

    type SyntheticTensor = (String, Vec<usize>, Vec<u8>);
    type SyntheticCase = (
        std::collections::HashMap<String, SyntheticTensor>,
        ForcedOperandGeometry,
        Vec<u32>,
        Vec<f32>,
        Vec<f32>,
    );
    type SyntheticOperands = (Vec<Vec<u8>>, Vec<Vec<u8>>, Vec<Vec<u8>>, Vec<Vec<u8>>);

    /// A tiny checkpoint for the full pre-launch verifier: hidden 64, intermediate 128 (four ranks
    /// of width 32, padded to 64) and a caller-chosen top-k (3 covers the small case, 6 matches the
    /// real MOE_TOP_K so every slot can be exercised). The tensors are deterministic pseudo-random
    /// bytes with ue8m0 scale codes of 127 (2^0), so the gate/up arithmetic is finite and
    /// reproducible.
    fn synthetic_forced_case_with_top_k(top_k: usize) -> SyntheticCase {
        assert!((3..=6).contains(&top_k), "top_k 3..6");
        let geometry = ForcedOperandGeometry {
            hidden: 64,
            intermediate: 128,
            width: 32,
            width_padded: 64,
            top_k,
            ranks: 4,
        };
        let all_ids = [7u32, 11, 13, 17, 19, 23];
        let all_weights = [0.5f32, 0.25, 0.125, 0.0625, 0.03125, 0.015625];
        let ids = all_ids[..top_k].to_vec();
        let weights = all_weights[..top_k].to_vec();
        let activation: Vec<f32> = (0..geometry.hidden)
            .map(|i| 0.01 * (i as f32) + 0.5)
            .collect();
        let mut seed = 17u32;
        let mut next_byte = move || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 24) as u8
        };
        let mut tensors: std::collections::HashMap<String, SyntheticTensor> =
            std::collections::HashMap::new();
        let prefix = "layers.0";
        for &id in &ids {
            let expert = format!("{prefix}.ffn.experts.{id}");
            for field in ["w1.weight", "w3.weight"] {
                let bytes: Vec<u8> = (0..geometry.intermediate * geometry.hidden / 2)
                    .map(|_| next_byte())
                    .collect();
                tensors.insert(
                    format!("{expert}.{field}"),
                    (
                        "I8".to_string(),
                        vec![geometry.intermediate, geometry.hidden / 2],
                        bytes,
                    ),
                );
            }
            for field in ["w1.scale", "w3.scale"] {
                tensors.insert(
                    format!("{expert}.{field}"),
                    (
                        "F8_E8M0".to_string(),
                        vec![geometry.intermediate, geometry.hidden / 32],
                        vec![127u8; geometry.intermediate * (geometry.hidden / 32)],
                    ),
                );
            }
            let w2_bytes: Vec<u8> = (0..geometry.hidden * (geometry.intermediate / 2))
                .map(|_| next_byte())
                .collect();
            tensors.insert(
                format!("{expert}.w2.weight"),
                (
                    "I8".to_string(),
                    vec![geometry.hidden, geometry.intermediate / 2],
                    w2_bytes,
                ),
            );
            tensors.insert(
                format!("{expert}.w2.scale"),
                (
                    "F8_E8M0".to_string(),
                    vec![geometry.hidden, geometry.intermediate / 32],
                    vec![127u8; geometry.hidden * (geometry.intermediate / 32)],
                ),
            );
        }
        (tensors, geometry, ids, weights, activation)
    }

    /// The small synthetic checkpoint (top-3) used by the general verifier tests.
    fn synthetic_forced_case() -> SyntheticCase {
        synthetic_forced_case_with_top_k(3)
    }

    /// Build the routed-combine operands exactly as the example's routed_operands does, for the
    /// synthetic tensors and routing.
    fn synthetic_forced_operands(
        tensors: &std::collections::HashMap<String, SyntheticTensor>,
        geometry: &ForcedOperandGeometry,
        ids: &[u32],
        weights: &[f32],
        activation: &[f32],
    ) -> SyntheticOperands {
        let g = *geometry;
        assert_eq!(ids.len(), g.top_k, "one id per slot");
        assert_eq!(weights.len(), g.top_k, "one weight per slot");
        let prefix = "layers.0";
        let packed_row = g.width_padded / 2;
        let scale_row = g.width_padded / 32;
        let get = |name: &str| -> Vec<u8> { tensors.get(name).expect(name).2.clone() };
        let mut w_rank: Vec<Vec<u8>> = Vec::with_capacity(g.ranks);
        let mut ws_rank: Vec<Vec<u8>> = Vec::with_capacity(g.ranks);
        for rank in 0..g.ranks {
            let mut weight = Vec::new();
            let mut scale = Vec::new();
            for &id in ids {
                let w2 = get(&format!("{prefix}.ffn.experts.{id}.w2.weight"));
                let s2 = get(&format!("{prefix}.ffn.experts.{id}.w2.scale"));
                weight.extend_from_slice(&slice_and_pad_rows(
                    &w2,
                    g.hidden,
                    g.intermediate / 2,
                    rank * (g.width / 2),
                    g.width / 2,
                    packed_row,
                ));
                scale.extend_from_slice(&slice_and_pad_rows(
                    &s2,
                    g.hidden,
                    g.intermediate / 32,
                    rank * (g.width / 32),
                    g.width / 32,
                    scale_row,
                ));
            }
            w_rank.push(weight);
            ws_rank.push(scale);
        }
        let mut act_fp8: Vec<Vec<u8>> = vec![Vec::new(); g.ranks];
        let mut act_scales: Vec<Vec<u8>> = vec![Vec::new(); g.ranks];
        for slot in 0..g.top_k {
            let id = ids[slot];
            let w1 = get(&format!("{prefix}.ffn.experts.{id}.w1.weight"));
            let s1 = get(&format!("{prefix}.ffn.experts.{id}.w1.scale"));
            let w3 = get(&format!("{prefix}.ffn.experts.{id}.w3.weight"));
            let s3 = get(&format!("{prefix}.ffn.experts.{id}.w3.scale"));
            let gate = gate_up_rows(activation, &w1, &s1, g.intermediate, g.hidden);
            let up = gate_up_rows(activation, &w3, &s3, g.intermediate, g.hidden);
            let swiglu: Vec<f32> = gate
                .iter()
                .zip(&up)
                .map(|(gate, up)| swiglu_ref::swiglu(*gate, *up, swiglu_ref::SWIGLU_LIMIT))
                .collect();
            for rank in 0..g.ranks {
                let (codes, scales) = mxfp4_ref::routed_activation_operand(
                    &swiglu[rank * g.width..(rank + 1) * g.width],
                    weights[slot],
                );
                let mut row = vec![0u8; g.width_padded];
                row[..g.width].copy_from_slice(&codes);
                act_fp8[rank].extend_from_slice(&row);
                let mut scale_row_bytes = vec![0u8; scale_row];
                scale_row_bytes[..scales.len()].copy_from_slice(&scales);
                act_scales[rank].extend_from_slice(&scale_row_bytes);
            }
        }
        (w_rank, ws_rank, act_fp8, act_scales)
    }

    struct MapSource(std::collections::HashMap<String, SyntheticTensor>);

    impl RawTensorSource for MapSource {
        fn read_raw(
            &mut self,
            name: &str,
            dtype: &str,
            shape: &[usize],
        ) -> Result<Vec<u8>, String> {
            let (stored_dtype, stored_shape, bytes) = self
                .0
                .get(name)
                .ok_or_else(|| format!("{name} is not in the synthetic source"))?;
            if stored_dtype != dtype {
                return Err(format!("{name}: expected {dtype}, found {stored_dtype}"));
            }
            if stored_shape.as_slice() != shape {
                return Err(format!(
                    "{name}: expected shape {shape:?}, found {stored_shape:?}"
                ));
            }
            Ok(bytes.clone())
        }
    }

    #[test]
    fn forced_operand_verifier_checks_every_slot_and_rank_on_synthetic_bytes() {
        let (tensors, geometry, ids, weights, activation) = synthetic_forced_case();
        let (w_rank, ws_rank, act_fp8, act_scales) =
            synthetic_forced_operands(&tensors, &geometry, &ids, &weights, &activation);
        verify_forced_operand_provenance(
            &mut MapSource(tensors.clone()),
            "layers.0",
            &activation,
            &ids,
            &weights,
            &w_rank,
            &ws_rank,
            &act_fp8,
            &act_scales,
            &geometry,
        )
        .expect("the synthetic operands are exactly the requested routing");

        // The same operands against a different id order must fail: w2 slices and activation
        // operands both come from the requested ids.
        let mut swapped = ids.clone();
        swapped.swap(0, 2);
        assert!(
            verify_forced_operand_provenance(
                &mut MapSource(tensors.clone()),
                "layers.0",
                &activation,
                &swapped,
                &weights,
                &w_rank,
                &ws_rank,
                &act_fp8,
                &act_scales,
                &geometry,
            )
            .is_err(),
            "operands built for one id order must not verify against another"
        );

        // A corrupted slot-1 w1 tensor changes slot 1's activation operand and must fail there.
        let mut corrupted = tensors.clone();
        let mut slot1_w1 = corrupted
            .get("layers.0.ffn.experts.11.w1.weight")
            .expect("slot 1 w1")
            .2
            .clone();
        for byte in &mut slot1_w1[..64] {
            *byte ^= 0xff;
        }
        corrupted
            .get_mut("layers.0.ffn.experts.11.w1.weight")
            .expect("slot 1 w1")
            .2 = slot1_w1;
        let error = verify_forced_operand_provenance(
            &mut MapSource(corrupted),
            "layers.0",
            &activation,
            &ids,
            &weights,
            &w_rank,
            &ws_rank,
            &act_fp8,
            &act_scales,
            &geometry,
        )
        .expect_err("a wrong slot-1 w1 must fail");
        assert!(error.contains("slot 1"), "unexpected error: {error}");

        // A changed activation byte at slot 1 / rank 0 must fail.
        let mut bad_act = act_fp8.clone();
        bad_act[0][geometry.width_padded] ^= 0x1;
        let error = verify_forced_operand_provenance(
            &mut MapSource(tensors.clone()),
            "layers.0",
            &activation,
            &ids,
            &weights,
            &w_rank,
            &ws_rank,
            &bad_act,
            &act_scales,
            &geometry,
        )
        .expect_err("a wrong slot-1 activation byte must fail");
        assert!(
            error.contains("slot 1") && error.contains("rank 0"),
            "unexpected error: {error}"
        );

        // A changed w2 byte in slot 1's rank-0 window must fail the per-rank weight check.
        let mut bad_w = w_rank.clone();
        bad_w[0][geometry.hidden * (geometry.width_padded / 2)] ^= 0x1;
        assert!(
            verify_forced_operand_provenance(
                &mut MapSource(tensors),
                "layers.0",
                &activation,
                &ids,
                &weights,
                &bad_w,
                &ws_rank,
                &act_fp8,
                &act_scales,
                &geometry,
            )
            .is_err(),
            "a wrong slot-1 w2 byte must fail"
        );
    }

    #[test]
    fn forced_operand_verifier_covers_every_slot_at_top_k_six() {
        // The real MOE_TOP_K is 6; a top-3 fixture cannot exercise slots 3..5. This fixture uses
        // six slots so corrupting slot 3's activation scale and slot 5's w1 proves every-slot
        // coverage (the earlier top-3 fixture hit an out-of-bounds here when asked for slot 3).
        let (tensors, geometry, ids, weights, activation) = synthetic_forced_case_with_top_k(6);
        assert_eq!(geometry.top_k, 6);
        assert_eq!(ids.len(), 6);
        let (w_rank, ws_rank, act_fp8, act_scales) =
            synthetic_forced_operands(&tensors, &geometry, &ids, &weights, &activation);
        verify_forced_operand_provenance(
            &mut MapSource(tensors.clone()),
            "layers.0",
            &activation,
            &ids,
            &weights,
            &w_rank,
            &ws_rank,
            &act_fp8,
            &act_scales,
            &geometry,
        )
        .expect("the top-6 synthetic operands are exactly the requested routing");

        // Slot 3's activation scale is a real slot at top-k 6: a changed byte must fail there.
        let scale_row = geometry.width_padded / 32;
        let mut bad_scales = act_scales.clone();
        bad_scales[0][3 * scale_row] ^= 0x1;
        let error = verify_forced_operand_provenance(
            &mut MapSource(tensors.clone()),
            "layers.0",
            &activation,
            &ids,
            &weights,
            &w_rank,
            &ws_rank,
            &act_fp8,
            &bad_scales,
            &geometry,
        )
        .expect_err("a wrong slot-3 activation scale must fail");
        assert!(error.contains("slot 3"), "unexpected error: {error}");

        // Slot 5's w1 tensor is a real slot at top-k 6: a corrupted source must fail at slot 5.
        let mut corrupted = tensors;
        let slot5_w1_name = format!("layers.0.ffn.experts.{}.w1.weight", ids[5]);
        let mut slot5_w1 = corrupted.get(&slot5_w1_name).expect("slot 5 w1").2.clone();
        for byte in &mut slot5_w1[..64] {
            *byte ^= 0xff;
        }
        corrupted.get_mut(&slot5_w1_name).expect("slot 5 w1").2 = slot5_w1;
        let error = verify_forced_operand_provenance(
            &mut MapSource(corrupted),
            "layers.0",
            &activation,
            &ids,
            &weights,
            &w_rank,
            &ws_rank,
            &act_fp8,
            &act_scales,
            &geometry,
        )
        .expect_err("a wrong slot-5 w1 must fail");
        assert!(error.contains("slot 5"), "unexpected error: {error}");
    }

    #[test]
    fn free_forced_metrics_separates_the_flip_from_the_host_errors() {
        let free = [1.0f32, 2.0, 3.0];
        let forced = [1.0f32, 2.0, 5.0];
        let host = [0.0f32, 2.0, 3.0];
        let metrics = free_forced_metrics(&free, &forced, &host).expect("metrics");
        assert!(!metrics.all_equal);
        assert_eq!(metrics.differing_points, 1);
        assert_eq!(metrics.at, 2);
        assert_eq!(metrics.pointwise_max_abs, 2.0);
        assert_eq!(metrics.signed_at, -2.0);
        assert_eq!(metrics.free_host_max_abs, 1.0);
        assert_eq!(metrics.free_host_at, 0);
        assert_eq!(metrics.forced_host_max_abs, 2.0);
        assert_eq!(metrics.forced_host_at, 2);
        let identical = free_forced_metrics(&free, &free, &host).expect("identical");
        assert!(identical.all_equal);
        assert_eq!(identical.pointwise_max_abs, 0.0);
        assert_eq!(identical.differing_points, 0);
        assert!(free_forced_metrics(&free, &forced[..2], &host).is_err());
    }
}
