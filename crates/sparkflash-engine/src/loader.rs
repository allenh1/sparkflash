//! The plan-verified loader: what the checkpoint holds, and what the arithmetic consumes.
//!
//! [SafeTensorsSource] answers questions about one shard; this module answers the question the
//! engine actually asks - is this directory the checkpoint the plan describes, and what are the
//! bytes of one tensor of it - and it does so without ever holding the model.
//!
//! # What it verifies
//!
//! [LoadedWeights::open] runs [LoadedWeights::audit] and refuses a checkpoint that does not match
//! the plan. The audit requires what convert.py:148-152 requires - every planned name must be
//! present - and differs from it in two opposite ways. It goes further, because it also checks each
//! stored descriptor against the plan's dtype, shape and byte length; and it stops short, because
//! convert.py asserts set equality (its `seen_names == expected_names`) while this collects the
//! names the plan does not know instead of refusing them, since the plan deliberately omits the
//! image path. It
//! reads only shard headers - a few kilobytes per shard, 48 headers for the released checkpoint -
//! so a refusal costs nothing and happens before any weight is materialised. The one thing the
//! audit compares is the container's raw descriptor against the plan's rule for that tensor:
//!
//! - an ordinary tensor must be stored with the plan's dtype word and shape;
//! - a routed or MTP expert must be stored as the packed int8 view, with the last dimension exactly
//!   half the logical one (tensor.rs:104-111 is why the plan's FP4 tensors are exactly the expert
//!   sets);
//! - a wo_a weight must have a scale grid that divides it into (32, 32) or (128, 128) blocks, the
//!   two pairs convert.py:164 accepts.
//!
//! In every case the stored byte length must be the plan's byte length, so a header whose offsets
//! disagree with its own shape is caught here rather than at the first read.
//!
//! The plan's own wo_a scale is the 32-block grid (dtype.rs SCALE_BLOCK, tensor.rs:148-151), so the
//! stored scale is compared against that shape: a checkpoint quantised on 128-blocks carries a
//! four-times-coarser scale and is refused as a plan disagreement before the conversion is reached.
//! The conversion accepts either pair anyway, because that is the check convert.py:164 makes and a
//! caller may reach it with stored shapes rather than with a plan, which is what its own tests do.
//!
//! A name the checkpoint has and the plan does not know is reported, not refused, and that is
//! deliberate: the released plan omits the image path - the vision tower, the aligner and the three
//! delimiters, 970,536,960 bytes of the index's 510,286,023,000 - so refusing the extra names would
//! refuse the released checkpoint. The report
//! says which names they are, so a checkpoint from a different revision is visible on the first
//! line of the log instead of being silently half-loaded.
//!
//! # The two conversions, and the one asymmetry
//!
//! The plan is the storage description, which is why it says F8E4M3 for wo_a and F4E2M1 for an
//! expert:
//!
//! - wo_a (convert.py:156-173) is stored as FP8 E4M3 codes plus a 2-D UE8M0 scale grid. The loader
//!   derives out_block and in_block from the two shapes, asserts the pair is (32, 32) or (128, 128)
//!   exactly as convert.py:164 does, multiplies each code by the scale of its block and hands back
//!   BF16, because that is what convert.py:173 hands the runtime. This is the one place where the
//!   dtype a caller sees differs from the dtype the plan names, and [Treatment] makes it explicit.
//! - an expert (convert.py:174-181) is stored as int8 that is really packed FP4 E2M1, two values
//!   per byte. No byte is transformed: the shape is reinterpreted, so the loader asserts the stored
//!   last dimension is exactly half the logical one and hands back the logical shape. A tensor that
//!   is already the logical width is a wrong checkpoint, and it is refused rather than loaded.
//!
//! # What it never loads
//!
//! There is no method that returns every tensor, and there is no eager path: 510,286,023,000 bytes
//! do not fit anywhere in this cluster. [LoadedWeights::get] reads one tensor on demand, under the
//! limit [SafeTensorsSource] enforces (2 GiB by default, enough for embed.weight at 1,323,827,200
//! bytes and head.weight at the same size, which are the largest tensors this loader may read
//! whole).
//!
//! The two Engram tables are not loaded here and must not be. They are
//! layers.1.engram.embed.weight and layers.14.engram.embed.weight, [384,006,168, 256] and
//! [384,016,682, 256] FP8, 98,305,579,008 and 98,308,270,592 bytes; with their .scale companions
//! (8 scale bytes per row, one per 32 columns) they are 202,758,032,400 bytes of the plan. The plan
//! covers the language model exactly: its total is 509,315,486,040, and that plus the 970,536,960 byte
//! image path is the index's 510,286,023,000 to the byte. The image path is the only thing the plan
//! leaves out, and it is what the unknown-name report above is for. crates/sparkflash-engram owns
//! those tables:
//! it keeps the packed per-rank store and loads the rows a token actually addresses, so the engine
//! never materialises a 98 GB tensor. This loader recognises them by name
//! ([is_engram_table]/[engram_tables]) and refuses to read them through
//! [SafeTensorsSource::with_refused], so a caller gets a typed refusal naming the tensor rather
//! than a 98 GB allocation or a confusing size error. They are still part of the audit, because a
//! checkpoint that lacks them is not the released checkpoint, and reading their headers is a few
//! kilobytes. For a caller that needs the rows themselves, [SafeTensorsSource::read_ranges] serves
//! named byte ranges of one tensor without a whole read (and deliberately does not consult the
//! refusal list: a range read is the alternative to the whole read the list refuses), and
//! [LoadedWeights::read_ranges] exposes it unchanged. The reference pass's Engram path uses it to
//! page the rows a chunk actually looks up.
//!
//! # The other safetensors reader, and why there are two
//!
//! crates/sparkflash-engram has a parallel safetensors reader (packer.rs: SafetensorsIndex,
//! SafetensorsHeader, and the same 64 MiB header bound this crate uses at
//! [crate::safetensors::MAX_HEADER_BYTES]). Recorded conclusion, not just the fact: the two cannot
//! be folded together without inverting the layering. Sharing the code in the engine means engram
//! depends on sparkflash-engine, which the layering rule forbids; sharing it in engram means the
//! engine depends on sparkflash-engram, which inverts the rule the other way and drags tokenizers,
//! memmap2, sha2 and the optional io-uring into a slice whose dependencies are serde,
//! serde_json and thiserror.
//! The contracts differ too: the packer's reader is an anyhow-typed build-time planner that mmaps
//! and bulk-copies whole shards, including the 98 GB tables, while this one is a WeightError-typed
//! random-access source that must refuse them. The only non-inverting fold is a new tiny crate
//! holding the header and index parse and nothing else, which is not justified for roughly 150
//! lines that need different error types and different IO strategies. The duplication is
//! deliberate.
//!
//! # What a header read confirms that a byte total merely corroborates
//!
//! The released plan's byte total is exactly the index's total minus the vision tower, which
//! confirms the *sizes*: every dtype and shape in the plan produces the number of bytes the
//! released checkpoint has. A byte total cannot confirm *layout*: it cannot say whether the
//! released wo_a grid is 32x32 or 128x128, whether the released experts are int8-packed FP4 or
//! already FP8, or whether a shard stores a tensor as F8_E4M3 with a scale or as BF16 without one.
//! A header read can, and that is exactly what the audit does: [crate::safetensors::SafeTensorsHeader]
//! gives the stored
//! dtype word, the stored shape and the declared byte range for every name, so the assertions
//! above - the (32, 32) grid, the half-width packed expert, the dtype and shape of every other
//! tensor - are confirmed directly against the container instead of inferred from a sum. The 48
//! released shards are not on this machine, so the tests build tiny shards with a known layout and
//! assert the relations by hand; the first run against the released checkpoint is where the audit
//! would confirm them for real.

use std::collections::BTreeSet;
use std::fmt;
use std::path::{Path, PathBuf};

use crate::dtype::DType;
use crate::quant;
use crate::safetensors::{DEFAULT_LIMIT, SafeTensorsSource};
use crate::tensor::{TensorPlan, TensorSpec};
use crate::weights::{WeightError, WeightSource};

/// The marker every Engram table name carries: layers.1.engram.embed.weight and its scale, and the
/// same for layer 14.
///
/// crates/sparkflash-engram owns those tensors; see the module docs for the sizes and for where
/// they are really loaded from.
pub const ENGRAM_TABLE_MARKER: &str = ".engram.embed.";

/// The suffix convert.py:157 matches for the one tensor it dequantises.
pub const WO_A_WEIGHT_SUFFIX: &str = "wo_a.weight";

/// The block pairs convert.py:164 accepts for a wo_a scale grid, as (out, in).
pub const WO_A_GRIDS: [(usize, usize); 2] = [(32, 32), (128, 128)];

/// Whether a name is one of the Engram embedding tables or its scale.
pub fn is_engram_table(name: &str) -> bool {
    name.contains(ENGRAM_TABLE_MARKER)
}

/// Every planned name this loader refuses to read: the Engram tables and their scales.
///
/// The plan names them, so the policy is derived from the plan rather than hardcoded - a
/// configuration with different engram_layer_ids is refused at the layers it actually has.
pub fn engram_tables(plan: &TensorPlan) -> Vec<String> {
    plan.names()
        .filter(|name| is_engram_table(name))
        .map(str::to_string)
        .collect()
}

/// The scale name that belongs to a wo_a weight name, the way convert.py:159 derives it.
pub fn wo_a_scale_name(weight: &str) -> String {
    weight.replace("weight", "scale")
}

/// How the loader treats one planned tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Treatment {
    /// Stored exactly as the plan says and handed back unchanged: the FP8 dense weights with their
    /// E8M0 scales, the BF16 embeddings and norms, and every scale tensor.
    Stored,
    /// A wo_a weight: FP8 E4M3 plus its UE8M0 scale grid, dequantised to BF16 on the way out
    /// (convert.py:156-173). The plan names the stored dtype, so this is the one treatment whose
    /// result dtype differs from the plan's.
    WoA,
    /// A routed or MTP expert weight: int8 that is packed FP4 E2M1, reinterpreted with no byte
    /// transformed (convert.py:174-181). The plan's FP4 entries are exactly the expert sets
    /// (tensor.rs:104-111).
    PackedExpert,
}

impl Treatment {
    /// How the loader treats one planned tensor.
    pub fn of(spec: &TensorSpec, name: &str) -> Self {
        if name.ends_with(WO_A_WEIGHT_SUFFIX) {
            Treatment::WoA
        } else if spec.dtype == DType::F4E2M1 {
            Treatment::PackedExpert
        } else {
            Treatment::Stored
        }
    }
}

/// One tensor, as the arithmetic will consume it.
///
/// The dtype and shape are the logical ones: BF16 for a wo_a weight, the plan's dtype and shape
/// for everything else. The bytes are in that layout, which for a packed expert means the packed
/// bytes with the logical shape - the view convert.py:181 takes, not an unpacked buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadedTensor {
    /// The name exactly as the plan spells it.
    pub name: String,
    /// The element format the bytes are in.
    pub dtype: DType,
    /// The logical shape.
    pub shape: Vec<usize>,
    /// The bytes, in the layout the dtype implies.
    pub data: Vec<u8>,
}

impl LoadedTensor {
    /// The number of logical elements.
    ///
    /// Saturating, like every element count in this crate: a shape comes from a plan, but this
    /// type can also be built by hand.
    pub fn elements(&self) -> u128 {
        self.shape
            .iter()
            .fold(1u128, |product, &dim| product.saturating_mul(dim as u128))
    }
}

/// One disagreement between the plan and a shard header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mismatch {
    /// The planned name.
    pub name: String,
    /// What the plan requires of the stored form.
    pub expected: String,
    /// What the checkpoint actually holds.
    pub found: String,
}

impl fmt::Display for Mismatch {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{}: expected {}, found {}",
            self.name, self.expected, self.found
        )
    }
}

/// What the plan and the checkpoint say about each other.
///
/// [LoadedWeights::audit] hands this back without refusing anything, and [LoadedWeights::open]
/// refuses on the parts a loader must not open with. The distinction matters for the extra names:
/// the released plan omits the vision tower, so unknown is reported and never refused, while a
/// planned name that is missing or stored wrongly is a checkpoint this loader will not serve.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Verification {
    /// How many planned names the checkpoint agrees with.
    pub matched: usize,
    /// Planned names the checkpoint does not have, in plan order.
    pub missing: Vec<String>,
    /// Planned names the checkpoint has in a form the plan does not accept, in plan order.
    pub mismatched: Vec<Mismatch>,
    /// Names the checkpoint has that the plan does not know, sorted.
    pub unknown: Vec<String>,
}

impl Verification {
    /// Whether every planned name is present and stored the way the plan says.
    pub fn is_conformant(&self) -> bool {
        self.missing.is_empty() && self.mismatched.is_empty()
    }

    /// Whether the checkpoint also holds nothing beyond the plan.
    ///
    /// False for the released checkpoint, which carries the vision tower the plan leaves out (see
    /// the module docs), so [Verification::is_conformant] is the check a loader acts on.
    pub fn is_exact(&self) -> bool {
        self.is_conformant() && self.unknown.is_empty()
    }

    /// One line for a log: how much of the plan the checkpoint accounts for, and how.
    pub fn summary(&self) -> String {
        format!(
            "{} matched, {} missing, {} mismatched, {} unknown",
            self.matched,
            self.missing.len(),
            self.mismatched.len(),
            self.unknown.len()
        )
    }
}

/// What the audit found for one planned name.
enum Finding {
    /// The stored form is what the plan requires.
    Agreed,
    /// The checkpoint does not have the name at all.
    Missing,
    /// The checkpoint has it in a form the plan does not accept.
    Disagreed {
        /// What the plan requires.
        expected: String,
        /// What the shard header says.
        found: String,
    },
}

/// The dtype word, shape and byte count the plan requires a tensor to be stored with.
///
/// The packed expert is the one place where the stored form is not the plan's entry: the plan
/// describes FP4 by its logical shape (tensor.rs:104-111) while the shard holds int8 with the last
/// dimension halved, which is the view convert.py:181 takes.
fn stored_form(spec: &TensorSpec) -> Result<(String, Vec<usize>, u128), String> {
    let bytes = spec.storage_bytes();
    if Treatment::of(spec, &spec.name) != Treatment::PackedExpert {
        return Ok((spec.dtype.name().to_string(), spec.shape.clone(), bytes));
    }
    let Some((&last, head)) = spec.shape.split_last() else {
        return Err("it has no dimensions at all".to_string());
    };
    if last % 2 != 0 {
        return Err(format!(
            "its logical last dimension is {last}, which is odd and so cannot be packed two values per byte"
        ));
    }
    let mut shape = head.to_vec();
    shape.push(last / 2);
    Ok(("I8".to_string(), shape, bytes))
}

/// How the plan's requirement reads in a report.
fn requirement(spec: &TensorSpec, word: &str, shape: &[usize]) -> String {
    if Treatment::of(spec, &spec.name) == Treatment::PackedExpert {
        format!(
            "the packed FP4 view: stored as {word} {shape:?}, the logical {:?} with its last dimension halved",
            spec.shape
        )
    } else {
        format!("stored as {word} {shape:?}")
    }
}

/// Compares one planned tensor's stored descriptor with what the plan requires.
///
/// Only the header is read. The container's raw words are the authority here, which is why the
/// audit does not go through [WeightSource::spec]: that maps the int8 word to a logical FP4 spec,
/// and the point of this check is the stored form itself.
fn inspect(source: &SafeTensorsSource, spec: &TensorSpec) -> Finding {
    let (word, shape, bytes) = match stored_form(spec) {
        Ok(form) => form,
        Err(detail) => {
            return Finding::Disagreed {
                expected: "a form the packed FP4 view can be taken of".to_string(),
                found: detail,
            };
        }
    };
    let expected = requirement(spec, &word, &shape);
    let header = match source.header(&spec.name) {
        Ok(header) => header,
        Err(WeightError::Missing { .. }) => return Finding::Missing,
        Err(error) => {
            return Finding::Disagreed {
                expected,
                found: error.to_string(),
            };
        }
    };
    let Ok(descriptor) = header.tensor(&spec.name) else {
        return Finding::Missing;
    };
    if descriptor.dtype != word {
        return Finding::Disagreed {
            expected,
            found: format!("the stored dtype word {}", descriptor.dtype),
        };
    }
    if descriptor.shape != shape {
        return Finding::Disagreed {
            expected,
            found: format!("the stored shape {:?}", descriptor.shape),
        };
    }
    if u128::from(descriptor.byte_len()) != bytes {
        return Finding::Disagreed {
            expected,
            found: format!("a declared range of {} bytes", descriptor.byte_len()),
        };
    }
    Finding::Agreed
}

/// Refuses a plan with no tensors in it.
///
/// The audit is a universal statement - every planned tensor is present and stored as described -
/// and a universal statement over an empty set is true. That is not a technicality worth tolerating
/// here: `TensorPlan::default()` is public, the audit would then call any directory conformant, and
/// the first read would be the thing that failed. A plan with nothing in it is a bug in the caller
/// and is named as one, before the checkpoint is opened.
fn refuse_an_empty_plan(plan: &TensorPlan) -> Result<(), WeightError> {
    if plan.is_empty() {
        return Err(WeightError::Plan {
            detail: "it has no tensors, so it would verify against any directory".to_string(),
        });
    }
    Ok(())
}

/// The plan against the checkpoint, reading only shard headers.
fn audit_source(source: &SafeTensorsSource, plan: &TensorPlan) -> Verification {
    let mut report = Verification::default();
    let mut agreed: BTreeSet<&str> = BTreeSet::new();
    for spec in plan.specs() {
        match inspect(source, spec) {
            Finding::Agreed => {
                agreed.insert(spec.name.as_str());
            }
            Finding::Missing => report.missing.push(spec.name.clone()),
            Finding::Disagreed { expected, found } => report.mismatched.push(Mismatch {
                name: spec.name.clone(),
                expected,
                found,
            }),
        }
    }
    // The wo_a grid is a relation between two planned tensors, so it is checked once both are known
    // to be stored as the plan says. Doing it here rather than only in the read path is what makes a
    // checkpoint with the wrong grid refused at open instead of at the first attention layer.
    for spec in plan.specs() {
        if Treatment::of(spec, &spec.name) != Treatment::WoA || !agreed.contains(spec.name.as_str())
        {
            continue;
        }
        let scale_name = wo_a_scale_name(&spec.name);
        if !agreed.contains(scale_name.as_str()) {
            continue;
        }
        let Some(scale) = plan.get(&scale_name) else {
            continue;
        };
        if let Err(detail) = grid_dims(&spec.shape, &scale.shape) {
            report.mismatched.push(Mismatch {
                name: spec.name.clone(),
                expected: "a scale grid that divides the weight into (32, 32) or (128, 128) blocks"
                    .to_string(),
                found: detail,
            });
        }
    }
    report.matched = plan
        .len()
        .saturating_sub(report.missing.len() + report.mismatched.len());
    let planned: BTreeSet<&str> = plan.names().collect();
    report.unknown = source
        .names()
        .into_iter()
        .filter(|name| !planned.contains(name.as_str()))
        .collect();
    report
}

/// One checkpoint, opened against one plan, read one tensor at a time.
///
/// The handle holds the source, the plan and the audit, and no tensor data. [LoadedWeights::get] is
/// the only way to bytes out of it, and it takes a name: there is deliberately no iterator, no
/// whole-model call and no eager path, because 510,286,023,000 bytes cannot be materialised and an
/// API that could try would be a foot-gun rather than a convenience.
#[derive(Debug)]
pub struct LoadedWeights {
    source: SafeTensorsSource,
    plan: TensorPlan,
    report: Verification,
}

impl LoadedWeights {
    /// Opens a checkpoint directory and refuses it unless the plan verifies against it.
    ///
    /// Only headers are read, so a refusal happens before any weight is materialised. A planned name
    /// the checkpoint does not have is [WeightError::Missing], and a planned name the checkpoint
    /// stores differently is [WeightError::Malformed] naming the tensor and saying what the plan
    /// requires and what the shard holds - unless the shard could not be read at all, in which case
    /// that error is returned as it happened. Extra names are reported in
    /// [LoadedWeights::report] rather than refused; see the module docs for why.
    ///
    /// # Errors
    ///
    /// [WeightError::Io] when the index cannot be read, and the errors described above.
    pub fn open(
        dir: impl Into<PathBuf>,
        index_path: &Path,
        plan: &TensorPlan,
    ) -> Result<Self, WeightError> {
        Self::open_with_limit(dir, index_path, plan, DEFAULT_LIMIT)
    }

    /// Opens with a per-tensor limit of the caller's choosing instead of
    /// [crate::safetensors::DEFAULT_LIMIT].
    ///
    /// The limit is the whole safety argument for reading a 510 GB checkpoint through one process,
    /// so it is an argument rather than a constant hidden in here: a host that cannot spare 1.32 GB
    /// for embed.weight can lower it, and a caller that wants to prove the guard works can lower it
    /// to a handful of bytes.
    ///
    /// # Errors
    ///
    /// As [LoadedWeights::open].
    pub fn open_with_limit(
        dir: impl Into<PathBuf>,
        index_path: &Path,
        plan: &TensorPlan,
        limit: u128,
    ) -> Result<Self, WeightError> {
        // Before the checkpoint is opened: an empty plan verifies against any directory, and a
        // conformance check that passes for that reason is worse than no check.
        refuse_an_empty_plan(plan)?;
        let source = SafeTensorsSource::open(dir, index_path)?
            .with_limit(limit)
            .with_refused(engram_tables(plan));
        let report = audit_source(&source, plan);
        if let Some(name) = report.missing.first() {
            return Err(WeightError::Missing { name: name.clone() });
        }
        if let Some(mismatch) = report.mismatched.first() {
            // A shard that cannot be read at all is reported as it happened; a shard that is there
            // but disagrees is reported as the disagreement, with the plan's requirement and the
            // header's contents both in the message.
            return Err(match source.spec(&mismatch.name) {
                Err(error) => error,
                Ok(_) => WeightError::Malformed {
                    name: mismatch.name.clone(),
                    detail: format!(
                        "the checkpoint disagrees with the plan: expected {}, found {}",
                        mismatch.expected, mismatch.found
                    ),
                },
            });
        }
        Ok(Self {
            source,
            plan: plan.clone(),
            report,
        })
    }

    /// The audit on its own: the plan against the checkpoint, refusing nothing.
    ///
    /// This is [LoadedWeights::open] without the refusal, for a caller that wants the whole report -
    /// every missing name, every disagreement, every extra name - before deciding what to do. It
    /// still reads only headers.
    ///
    /// # Errors
    ///
    /// [WeightError::Io] when the index cannot be read.
    pub fn audit(
        dir: impl Into<PathBuf>,
        index_path: &Path,
        plan: &TensorPlan,
    ) -> Result<Verification, WeightError> {
        refuse_an_empty_plan(plan)?;
        let source = SafeTensorsSource::open(dir, index_path)?
            .with_limit(DEFAULT_LIMIT)
            .with_refused(engram_tables(plan));
        Ok(audit_source(&source, plan))
    }

    /// What the audit found. Conformant by construction after a successful open; the extra names
    /// are what is left to look at.
    pub fn report(&self) -> &Verification {
        &self.report
    }

    /// The plan this loader serves.
    pub fn plan(&self) -> &TensorPlan {
        &self.plan
    }

    // There is deliberately no accessor for the shard reader underneath. [SafeTensorsSource::read]
    // returns raw stored bytes, so exposing it would hand a caller a way around the two conversions
    // and make "get is the only way to bytes" false. Everything a caller legitimately wants from it
    // is already here without the byte path: [LoadedWeights::limit],
    // [LoadedWeights::refused_names], [LoadedWeights::names] and [LoadedWeights::report].

    /// The per-tensor limit in force.
    pub fn limit(&self) -> u128 {
        self.source.limit()
    }

    /// The names this loader refuses to read, sorted: the Engram tables and their scales.
    pub fn refused_names(&self) -> Vec<String> {
        self.source.refused().iter().cloned().collect()
    }

    /// Every planned name, in plan order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.plan.names()
    }

    /// Reads one tensor, converting it if its [Treatment] says so.
    ///
    /// This is the only way to tensor bytes out of a [LoadedWeights]: there is no iterator and no
    /// accessor for the reader underneath, so a caller cannot reach the raw stored form of a tensor
    /// whose [Treatment] says it must be converted.
    ///
    /// A name the plan does not hold is [WeightError::Missing] even when the checkpoint has it: the
    /// loader serves the plan, and the plan is what the arithmetic was compiled against. An Engram
    /// table is refused before its shard is opened, as [WeightError::Refused] naming the tensor.
    ///
    /// # Errors
    ///
    /// [WeightError::Missing] for a name outside the plan, [WeightError::TooLarge] for a tensor over
    /// the limit, [WeightError::Refused] for a name on the refusal list, [WeightError::Io] when a
    /// shard cannot be read, and [WeightError::Malformed] when the stored descriptor contradicts the
    /// plan or the conversion cannot be made.
    pub fn get(&self, name: &str) -> Result<LoadedTensor, WeightError> {
        let Some(spec) = self.plan.get(name) else {
            return Err(WeightError::Missing {
                name: name.to_string(),
            });
        };
        match Treatment::of(spec, name) {
            Treatment::Stored => {
                let data = self.source.read(name)?;
                let expected = spec.storage_bytes();
                if u128::from(data.len() as u64) != expected {
                    return Err(WeightError::Malformed {
                        name: name.to_string(),
                        detail: format!(
                            "read {} bytes where the plan stores {expected}",
                            data.len()
                        ),
                    });
                }
                Ok(LoadedTensor {
                    name: name.to_string(),
                    dtype: spec.dtype,
                    shape: spec.shape.clone(),
                    data,
                })
            }
            Treatment::PackedExpert => {
                let header = self.source.header(name)?;
                let descriptor = header.tensor(name)?;
                let (word, shape) = (descriptor.dtype.clone(), descriptor.shape.clone());
                let stored = self.source.read(name)?;
                packed_expert(name, &word, &shape, &spec.shape, stored)
            }
            Treatment::WoA => {
                let scale_name = wo_a_scale_name(name);
                let header = self.source.header(name)?;
                let weight_shape = header.tensor(name)?.shape.clone();
                let scale_header = self.source.header(&scale_name)?;
                let scale_shape = scale_header.tensor(&scale_name)?.shape.clone();
                let weight = self.source.read(name)?;
                let scale = self.source.read(&scale_name)?;
                dequantise_wo_a(name, &weight, &weight_shape, &scale, &scale_shape)
            }
        }
    }

    /// Reads named byte ranges of one stored tensor, for the paged Engram path.
    ///
    /// This bypasses [LoadedWeights::get]'s whole-tensor treatments on purpose: the only tensors a
    /// caller pages are the two Engram tables, which are stored FP8 with an E8M0 per-row-32 scale
    /// row and need no conversion, and this method returns their raw bytes for the caller to decode
    /// row by row. The source's refusal list does not apply to a range read (see
    /// [crate::safetensors::SafeTensorsSource::read_ranges]); the element limit does not either,
    /// because holding one row at a time is what makes the released 98 GB table readable at all.
    ///
    /// # Errors
    /// As [crate::safetensors::SafeTensorsSource::read_ranges].
    pub fn read_ranges(&self, name: &str, ranges: &[(u64, usize)]) -> Result<Vec<u8>, WeightError> {
        self.source.read_ranges(name, ranges)
    }
}

/// A plan-consistent source for the CPU reference pass.
///
/// [LoadedWeights] hands back what a serving path wants: `wo_a` comes out as BF16 with its block
/// scales already applied, because that is the converted form the device kernels take. The
/// reference pass is the other way round - it decodes what the plan names (`F8E4M3` plus the
/// separate `wo_a.scale`) and applies the grid itself, as model.py does - so it needs the stored
/// bytes and the plan's [TensorSpec] for every tensor. The one reinterpretation it does share with
/// the loader is the packed expert: the file stores two FP4 values per int8 byte, and both the
/// device path and the reference's expert path want the logical `F4E2M1` shape, so [packed_expert]
/// runs here too.
///
/// The two Engram tables are only ever read through [WeightSource::read_ranges], which this source
/// serves straight from the shard; a whole read of them is still refused by name.
pub struct ReferenceWeights {
    source: SafeTensorsSource,
    plan: TensorPlan,
}

impl ReferenceWeights {
    /// Verifies the checkpoint against the plan, then opens the raw source the reference reads.
    ///
    /// The verification is [LoadedWeights::open]'s: every planned tensor's header must agree with
    /// the plan, and the report must be exact. Only then is the source opened, with the Engram
    /// tables on its refusal list and the default byte limit for every other tensor.
    ///
    /// # Errors
    /// As [LoadedWeights::open] and [SafeTensorsSource::open].
    pub fn open(
        dir: impl Into<PathBuf>,
        index_path: &Path,
        plan: &TensorPlan,
    ) -> Result<Self, WeightError> {
        let dir = dir.into();
        LoadedWeights::open(dir.clone(), index_path, plan)?;
        let source = SafeTensorsSource::open(dir, index_path)?.with_refused(engram_tables(plan));
        Ok(Self {
            source,
            plan: plan.clone(),
        })
    }

    /// The plan this source was opened against.
    pub fn plan(&self) -> &TensorPlan {
        &self.plan
    }
}

impl WeightSource for ReferenceWeights {
    /// Every planned name, in the plan's order.
    fn names(&self) -> Vec<String> {
        self.plan.names().map(str::to_owned).collect()
    }

    /// The plan's spec, which is what the engine decodes against for every tensor this source serves
    /// (the packed expert's logical shape included).
    fn spec(&self, name: &str) -> Result<TensorSpec, WeightError> {
        self.plan
            .get(name)
            .cloned()
            .ok_or_else(|| WeightError::Missing {
                name: name.to_owned(),
            })
    }

    /// The stored bytes of one tensor, with the packed expert reinterpreted to the logical shape.
    ///
    /// Everything else - `wo_a` included - is returned exactly as stored, because the engine
    /// applies the separate scale grids itself.
    fn read(&self, name: &str) -> Result<Vec<u8>, WeightError> {
        let Some(spec) = self.plan.get(name) else {
            return Err(WeightError::Missing {
                name: name.to_owned(),
            });
        };
        if Treatment::of(spec, name) != Treatment::PackedExpert {
            return self.source.read(name);
        }
        // The source's spec() is already the logical FP4 view, so the stored word and the packed
        // shape come from the header descriptor itself, which is what packed_expert checks.
        let header = self.source.header(name)?;
        let descriptor = header.tensor(name)?;
        let (word, shape) = (descriptor.dtype.clone(), descriptor.shape.clone());
        let bytes = self.source.read(name)?;
        packed_expert(name, &word, &shape, &spec.shape, bytes).map(|tensor| tensor.data)
    }

    /// Named byte ranges of one stored tensor, straight from the shard (the Engram paging path).
    fn read_ranges(
        &self,
        name: &str,
        ranges: &[(u64, usize)],
    ) -> Option<Result<Vec<u8>, WeightError>> {
        Some(self.source.read_ranges(name, ranges))
    }
}

/// The block layout of a wo_a scale grid, once it has been checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Grid {
    out: usize,
    in_dim: usize,
    out_blocks: usize,
    in_blocks: usize,
    out_block: usize,
    in_block: usize,
}

/// The grid check, as a caller of the conversion sees it.
fn wo_a_grid(
    name: &str,
    weight_shape: &[usize],
    scale_shape: &[usize],
) -> Result<Grid, WeightError> {
    grid_dims(weight_shape, scale_shape).map_err(|detail| WeightError::Malformed {
        name: name.to_string(),
        detail,
    })
}

/// The grid check itself: divisibility, then the two pairs convert.py:164 accepts.
fn grid_dims(weight_shape: &[usize], scale_shape: &[usize]) -> Result<Grid, String> {
    let [out_dim, in_dim] = weight_shape else {
        return Err(format!(
            "the weight shape {weight_shape:?} is not two-dimensional"
        ));
    };
    let [out_blocks, in_blocks] = scale_shape else {
        return Err(format!(
            "the scale shape {scale_shape:?} is not two-dimensional"
        ));
    };
    if *out_dim == 0 || *in_dim == 0 || *out_blocks == 0 || *in_blocks == 0 {
        return Err(format!(
            "the weight {weight_shape:?} and its scale grid {scale_shape:?} must have no empty dimension"
        ));
    }
    if out_dim % out_blocks != 0 || in_dim % in_blocks != 0 {
        return Err(format!(
            "the scale grid {scale_shape:?} does not divide the weight {weight_shape:?}"
        ));
    }
    let out_block = out_dim / out_blocks;
    let in_block = in_dim / in_blocks;
    if !WO_A_GRIDS.contains(&(out_block, in_block)) {
        return Err(format!(
            "the scale grid is ({out_block}, {in_block}) per block, neither (32, 32) nor (128, 128), which convert.py:164 refuses"
        ));
    }
    Ok(Grid {
        out: *out_dim,
        in_dim: *in_dim,
        out_blocks: *out_blocks,
        in_blocks: *in_blocks,
        out_block,
        in_block,
    })
}

/// Decodes one whole tensor, mapping a decoder complaint into the tensor's own malformed error.
fn decode(
    name: &str,
    dtype: DType,
    bytes: &[u8],
    elements: usize,
) -> Result<Vec<f32>, WeightError> {
    quant::decode(dtype, bytes, elements).map_err(|error| WeightError::Malformed {
        name: name.to_string(),
        detail: error.to_string(),
    })
}

/// Dequantises one wo_a weight: FP8 E4M3 codes and their UE8M0 block scales, as BF16.
///
/// This is convert.py:156-173. The block sizes come out of the two shapes rather than out of a
/// constant - out_block is out/scale_out and in_block is in/scale_in - and the pair must be (32, 32)
/// or (128, 128), which is the assertion convert.py:164 makes and the reason a checkpoint whose
/// grid is anything else is refused instead of scaled wrongly. Each code is multiplied by the scale
/// of the block it sits in, in f32, and the product is rounded to bf16 the way convert.py:173
/// rounds it.
///
/// The result is the one tensor whose dtype is not the plan's: the plan names F8E4M3 because that
/// is what is stored, and the arithmetic wants BF16 because that is what convert.py hands it.
///
/// # Errors
///
/// [WeightError::Malformed] when the shapes are not a grid convert.py accepts, when the grid does
/// not divide the weight, or when the bytes are not the length the shapes imply.
pub fn dequantise_wo_a(
    name: &str,
    weight: &[u8],
    weight_shape: &[usize],
    scale: &[u8],
    scale_shape: &[usize],
) -> Result<LoadedTensor, WeightError> {
    let grid = wo_a_grid(name, weight_shape, scale_shape)?;
    let codes = decode(
        name,
        DType::F8E4M3,
        weight,
        grid.out.saturating_mul(grid.in_dim),
    )?;
    let scales = decode(
        name,
        DType::F8E8M0,
        scale,
        grid.out_blocks.saturating_mul(grid.in_blocks),
    )?;
    let mut data = Vec::with_capacity(codes.len().saturating_mul(2));
    for (row, row_codes) in codes.chunks(grid.in_dim).enumerate() {
        let first = (row / grid.out_block) * grid.in_blocks;
        let row_scales = &scales[first..first + grid.in_blocks];
        for (column, &code) in row_codes.iter().enumerate() {
            let value = code * row_scales[column / grid.in_block];
            data.extend_from_slice(&f32_to_bf16(value).to_le_bytes());
        }
    }
    Ok(LoadedTensor {
        name: name.to_string(),
        dtype: DType::BF16,
        shape: vec![grid.out, grid.in_dim],
        data,
    })
}

/// Reinterprets one expert weight from the int8 view into the logical FP4 shape, byte for byte.
///
/// This is convert.py:174-181: the stored tensor is int8 that is really two FP4 E2M1 values per
/// byte, so nothing is transformed - only the element format and the shape change, and the last
/// dimension is exactly halved. A tensor already at the logical width is a checkpoint that was
/// written differently, and it is refused rather than reinterpreted: doubling its element count
/// would invent values that are not in the file.
///
/// convert.py's other expert branch (expert_dtype == "fp8", convert.py:175-179) casts the packed
/// tensor to E4M3 instead of viewing it as FP4. That is not this checkpoint: the released plan
/// names FP4, so an expert stored under the F8_E4M3 word is refused as a plan disagreement rather
/// than cast.
///
/// # Errors
///
/// [WeightError::Malformed] when the stored word is not I8, when the stored shape does not match the
/// logical one outside the last dimension, when the last dimension is not exactly half, or when the
/// bytes are not the packed length.
pub fn packed_expert(
    name: &str,
    stored_word: &str,
    stored_shape: &[usize],
    logical_shape: &[usize],
    stored: Vec<u8>,
) -> Result<LoadedTensor, WeightError> {
    let malformed = |detail: String| WeightError::Malformed {
        name: name.to_string(),
        detail,
    };
    if stored_word != "I8" {
        return Err(malformed(format!(
            "the expert weight is stored as {stored_word}, but the packed FP4 view convert.py:181 takes needs the int8 word I8"
        )));
    }
    let (Some((&logical_last, logical_head)), Some((&stored_last, stored_head))) =
        (logical_shape.split_last(), stored_shape.split_last())
    else {
        return Err(malformed(format!(
            "the expert weight has no dimensions: stored {stored_shape:?}, logical {logical_shape:?}"
        )));
    };
    if stored_head != logical_head {
        return Err(malformed(format!(
            "the stored shape {stored_shape:?} does not match the logical shape {logical_shape:?} outside the last dimension"
        )));
    }
    if logical_last % 2 != 0 || stored_last != logical_last / 2 {
        return Err(malformed(format!(
            "the expert weight is not packed: the stored last dimension {stored_last} is not half of the logical {logical_last}"
        )));
    }
    let elements = logical_shape
        .iter()
        .fold(1u128, |product, &dim| product.saturating_mul(dim as u128));
    let packed_bytes = elements / 2;
    if u128::from(stored.len() as u64) != packed_bytes {
        return Err(malformed(format!(
            "the expert weight holds {} bytes where the logical shape {logical_shape:?} packs into {packed_bytes}",
            stored.len()
        )));
    }
    Ok(LoadedTensor {
        name: name.to_string(),
        dtype: DType::F4E2M1,
        shape: logical_shape.to_vec(),
        data: stored,
    })
}

/// Rounds an f32 to the nearest bf16, ties to even, the way torch's bfloat16 cast does.
///
/// The value is an fp8 code times an e8m0 scale, computed in f32, and convert.py:173 rounds it.
/// Ties-to-even rather than a truncation matters: truncating costs half a bit of the mantissa
/// against the reference, and it makes a round trip through bf16 unstable for exactly the values
/// that sit on a midpoint.
fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    // A NaN keeps its high bits and is made quiet. The infinities have no set mantissa bit for the
    // rounding below to carry into, and an f32 above the largest finite bf16 lands on infinity, as
    // the cast does.
    if value.is_nan() {
        return ((bits >> 16) as u16) | 0x0040;
    }
    let round = 0x7FFF + ((bits >> 16) & 1);
    (bits.wrapping_add(round) >> 16) as u16
}

#[cfg(test)]
mod tests {
    use super::f32_to_bf16;

    #[test]
    fn the_bf16_cast_rounds_to_nearest_with_ties_to_even() {
        assert_eq!(f32_to_bf16(0.0), 0x0000);
        assert_eq!(f32_to_bf16(1.0), 0x3F80);
        assert_eq!(f32_to_bf16(-2.0), 0xC000);
        // 0.1f32 is 0x3DCCCCCD; rounding to nearest gives 0x3DCD, truncation would give 0x3DCC.
        assert_eq!(f32_to_bf16(0.1), 0x3DCD);
        // 1 + 2^-8 is exactly between 1.0 (0x3F80) and its successor (0x3F81): the even one wins.
        assert_eq!(f32_to_bf16(f32::from_bits(0x3F80_8000)), 0x3F80);
        // 1 + 3*2^-8 is exactly between 0x3F81 and 0x3F82: the even one wins again.
        assert_eq!(f32_to_bf16(f32::from_bits(0x3F81_8000)), 0x3F82);
        assert_eq!(f32_to_bf16(f32::INFINITY), 0x7F80);
        assert_eq!(f32_to_bf16(f32::NEG_INFINITY), 0xFF80);
        assert!(f32::from_bits(u32::from(f32_to_bf16(f32::NAN)) << 16).is_nan());
    }

    #[test]
    fn every_bf16_value_survives_a_round_trip_and_every_boundary_rounds_the_same_way() {
        // The sampler above pins seven values; this is the whole format, in two exhaustive
        // statements over the 65,536 possible bf16 bit patterns:
        //
        // 1. every non-NaN bf16 value is exactly representable in f32 and comes back unchanged, so
        //    the cast is the identity on the entire bf16 grid rather than on seven points of it;
        // 2. at each of the 65,535 boundaries between adjacent bf16 values, the f32 value just below
        //    the boundary rounds down, the one just above rounds up, and the exact midpoint - which
        //    is an f32, since bf16 has 8 significand bits and f32 has 24 - rounds to the even one.
        //
        // Points and boundaries together are every decision the rule can make, so a cast that was
        // wrong anywhere in the format would fail here: a truncating cast breaks statement 2 at every
        // midpoint, and a cast that rounded half away from zero breaks it at the odd ones.
        let mut grid = 0usize;
        for bits in 0u32..=0xFFFF {
            let value = f32::from_bits(bits << 16);
            if value.is_nan() {
                continue;
            }
            assert_eq!(f32_to_bf16(value), bits as u16, "round trip of {bits:#06x}");
            grid += 1;
        }
        // 128 NaN patterns per sign: all-ones exponent with a non-zero mantissa.
        assert_eq!(grid, 65_536 - 254, "every non-NaN pattern was checked");

        let mut boundaries = 0usize;
        for bits in 0u32..0xFFFF {
            let low = f32::from_bits(bits << 16);
            let high = f32::from_bits((bits + 1) << 16);
            // The two infinite patterns and the two NaN bands have no boundary to round across.
            if !low.is_finite() || !high.is_finite() {
                continue;
            }
            // The boundary is the f32 exactly halfway between two adjacent bf16 values, which is
            // their bit patterns half a step apart: 0x8000 below the next one.
            let midpoint = f32::from_bits((bits << 16) | 0x8000);
            let below = f32::from_bits((bits << 16) | 0x7FFF);
            let above = f32::from_bits((bits << 16) | 0x8001);
            assert_eq!(
                f32_to_bf16(below),
                bits as u16,
                "below the boundary at {bits:#06x}"
            );
            assert_eq!(
                f32_to_bf16(above),
                (bits + 1) as u16,
                "above the boundary at {bits:#06x}"
            );
            let even = if bits & 1 == 0 { bits } else { bits + 1 };
            assert_eq!(f32_to_bf16(midpoint), even as u16, "the tie at {bits:#06x}");
            boundaries += 1;
        }
        // Every adjacent pair except the 257 that touch an infinity or a NaN: for each sign, the pair
        // that ends at the infinity and the pair that starts at it, plus the pairs whose lower member
        // is itself a NaN - 127 of those on the positive side and 126 on the negative, because the
        // all-ones pattern is the last one the loop reaches and has no successor to pair with.
        assert_eq!(boundaries, 65_535 - 257, "every boundary was checked");
    }
}
