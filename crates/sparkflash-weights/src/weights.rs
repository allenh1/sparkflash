//! The weight source: the only interface the arithmetic code will ever see.
//!
//! A source answers three questions about a tensor - does it exist, what is its dtype and shape,
//! and what are its bytes - and nothing else. Two implementations are planned:
//!
//! - a synthetic source, which generates bytes from a seed and needs no checkpoint, and
//! - the checkpoint source, which memory-maps the safetensors shards named by the index.
//!
//! Keeping the trait this narrow is what makes the engine testable without the weights: every test
//! that is about arithmetic rather than about the file format runs against the synthetic source, and
//! the conformance check below is what proves the two sources describe the same model.

use crate::dtype::DType;
use crate::spec::TensorSpec;

/// What can go wrong while asking a source for a tensor.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum WeightError {
    /// The source does not hold a tensor with that name.
    #[error("no tensor named {name}")]
    Missing {
        /// The name that was asked for.
        name: String,
    },
    /// The tensor exists but is larger than this source is willing to materialise.
    ///
    /// The real checkpoint is 510 GB across four nodes, so a source that will happily allocate a
    /// tensor of any size is a source that can take a machine down by accident. This error is the
    /// guard, and the limit is a constructor argument.
    #[error("tensor {name} needs {bytes} bytes, over this source's limit of {limit}")]
    TooLarge {
        /// The name that was asked for.
        name: String,
        /// The number of bytes the tensor would occupy.
        bytes: u128,
        /// The source's configured limit.
        limit: u128,
    },
    /// The plan itself cannot be checked against anything.
    ///
    /// The audit asks whether a checkpoint is the model a plan describes, and an empty plan
    /// describes nothing: every planned name is trivially present, so it would report a directory
    /// full of unrelated files as conformant. A check that passes because it checked nothing is the
    /// failure this crate is built to avoid, so the empty plan is refused before the checkpoint is
    /// consulted at all.
    #[error("the tensor plan cannot be used: {detail}")]
    Plan {
        /// What is wrong with it.
        detail: String,
    },
    /// The source refuses this name outright, whatever its size.
    ///
    /// Not every tensor that must not be read is too large to read. The two Engram tables are 98 GB
    /// each and belong to another crate's packed store, so this source names them and refuses them;
    /// that refusal is a policy about the name, not a measurement of the bytes, and it is decided
    /// before the shard is opened so that a refused name costs no file handle. Reporting it as
    /// [WeightError::TooLarge] would have meant inventing a size for a tensor that was never
    /// described.
    #[error("tensor {name} is refused by this source: {detail}")]
    Refused {
        /// The name that was asked for.
        name: String,
        /// Why this source will not read it.
        detail: String,
    },
    /// The underlying storage could not be read.
    #[error("could not read {name}: {detail}")]
    Io {
        /// The name that was asked for.
        name: String,
        /// The operating system's message.
        detail: String,
    },
    /// The storage disagrees with the plan.
    #[error("tensor {name} is malformed: {detail}")]
    Malformed {
        /// The name that was asked for.
        name: String,
        /// What is wrong with it.
        detail: String,
    },
}

/// A borrowed view of one tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorView<'a> {
    /// The tensor's name.
    pub name: &'a str,
    /// Its element format.
    pub dtype: DType,
    /// Its logical shape: the shape the arithmetic sees, which for a packed FP4 weight is twice
    /// the last dimension of the bytes on disk.
    pub shape: &'a [usize],
    /// Its bytes, in the layout the dtype implies.
    pub data: &'a [u8],
}

/// An open weight file and the byte region `(start, len)` of one tensor in it.
pub type FileRegion = (std::sync::Arc<std::fs::File>, u64, u64);

/// A source of weights.
///
/// `Send + Sync` is part of the contract because the reference pass reads experts concurrently:
/// two independent expert bodies have no order between them, so the MoE block reads them from
/// several threads at once. A source is immutable from the caller's side (every method takes
/// `&self`), so `Sync` is the real requirement and `Send` follows from the boxed trait object
/// the engine owns. A source that needs interior mutability to serve a read must make it
/// thread-safe, exactly as [crate::safetensors::SafeTensorsSource] does for its header cache.
pub trait WeightSource: Send + Sync {
    /// Every tensor name this source holds, in a stable order.
    fn names(&self) -> Vec<String>;

    /// The dtype and logical shape of one tensor.
    fn spec(&self, name: &str) -> Result<TensorSpec, WeightError>;

    /// The bytes of one tensor.
    fn read(&self, name: &str) -> Result<Vec<u8>, WeightError>;

    /// The bytes of one tensor as a borrowed view, when the source already holds them.
    ///
    /// A resident source keeps every planned tensor in memory for the life of the process; handing
    /// the forward a borrow instead of a copy removes a per-tensor memcpy from every step. A source
    /// that cannot borrow (the default) keeps this `None` and the caller falls back to
    /// [WeightSource::read], so no implementation is forced to change.
    fn read_view(&self, name: &str) -> Result<Option<&[u8]>, WeightError> {
        let _ = name;
        Ok(None)
    }

    /// Named byte ranges of one tensor, for a source that can page a table too large to hold.
    ///
    /// The released Engram tables are 98 GB of FP8 each, far past any element limit the reference
    /// pass would accept, but a token looks up only a few rows of them. A source that can read a
    /// row on demand returns them here; one that only serves whole tensors keeps the default
    /// `None`, and the caller falls back to [`WeightSource::read`] under its limit (which refuses a
    /// released-size table rather than allocating it).
    ///
    /// The returned bytes are the ranges concatenated in the order given, with no decoding.
    fn read_ranges(
        &self,
        name: &str,
        ranges: &[(u64, usize)],
    ) -> Option<Result<Vec<u8>, WeightError>> {
        let _ = (name, ranges);
        None
    }

    /// [WeightSource::read_ranges] with every source-side whole-tensor fast path suppressed: each
    /// range is read on its own, and a source must not materialise the tensor to serve it. A paged
    /// caller uses this for a table whose stored region is tens of GB (the Engram embed tables are
    /// 98.3 GB each), where a whole-tensor shortcut is an out-of-memory bug rather than an
    /// optimisation. The default delegates to [WeightSource::read_ranges], which is correct for a
    /// source whose ranges are slices of memory it already holds.
    fn read_ranges_paged(
        &self,
        name: &str,
        ranges: &[(u64, usize)],
    ) -> Option<Result<Vec<u8>, WeightError>> {
        self.read_ranges(name, ranges)
    }

    /// The open file and the byte region `(start, len)` that hold a tensor's stored bytes, for a
    /// caller that issues its own concurrent reads (the runtime Engram gather, whose 256-byte
    /// lookups are latency-bound one at a time). `None` when the source does not serve from a file.
    fn file_region(&self, name: &str) -> Option<Result<FileRegion, WeightError>> {
        let _ = name;
        None
    }

    /// Whether the source holds a tensor with that name.
    fn contains(&self, name: &str) -> bool {
        self.spec(name).is_ok()
    }
}

/// The result of checking a source against a plan.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Conformance {
    /// How many tensors were present with the planned dtype and shape.
    pub matched: usize,
    /// Every disagreement, in plan order: a missing tensor, a dtype that differs, or a shape that
    /// differs.
    pub mismatches: Vec<String>,
}

impl Conformance {
    /// Whether the source and the plan describe the same model.
    pub fn is_conformant(&self) -> bool {
        self.mismatches.is_empty()
    }
}

/// Check every tensor in `specs` against `source`.
///
/// This is the check that lets a synthetic source stand in for the checkpoint in a test: it reports
/// whether the two describe the same names with the same dtypes and shapes, without reading tensor
/// data - only specs - so it is cheap even against a 510 GB checkpoint.
///
/// What it can and cannot prove depends entirely on the source. Against a real checkpoint it is a
/// conformance gate. Against a source built from the same specs it can only ever agree, because
/// that source clones the plan: a test that wants to show this check firing must hand it a source
/// that deviates, which is what the V4.1 `the_conformance_check_rejects_a_source_that_deviates`
/// does.
pub fn check_source_against_specs<S: WeightSource + ?Sized>(
    source: &S,
    specs: &[TensorSpec],
) -> Conformance {
    let mut report = Conformance::default();
    for wanted in specs {
        match source.spec(&wanted.name) {
            Err(error) => report.mismatches.push(format!("{}: {error}", wanted.name)),
            Ok(found) => {
                if found.dtype != wanted.dtype {
                    report.mismatches.push(format!(
                        "{}: dtype {} but the plan says {}",
                        wanted.name,
                        found.dtype.name(),
                        wanted.dtype.name()
                    ));
                } else if found.shape != wanted.shape {
                    report.mismatches.push(format!(
                        "{}: shape {:?} but the plan says {:?}",
                        wanted.name, found.shape, wanted.shape
                    ));
                } else {
                    report.matched += 1;
                }
            }
        }
    }
    report
}
