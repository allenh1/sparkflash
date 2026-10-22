//! The safetensors container and the checkpoint index: how 510 GB of weights is addressed.
//!
//! The released checkpoint is 96,085 tensors in 48 safetensors shards, 510,286,023,000 bytes,
//! described by model.safetensors.index.json. Nothing here ever reads a whole shard: a shard is
//! opened, its header is read, and only the bytes of the tensor that was asked for are read, by
//! seeking to that tensor's offset in the data section.
//!
//! The container:
//!
//! - the first eight bytes are the header length, little-endian;
//! - the next header_len bytes are a JSON object: each key is a tensor name whose value is
//!   {dtype, shape, data_offsets}, except the reserved key __metadata__, whose value is arbitrary
//!   JSON and is not a tensor;
//! - the data section starts at 8 + header_len, and every data_offsets pair is [begin, end)
//!   relative to that start.
//!
//! Why the header is read separately from the data: the two Engram tables are roughly 98 GB each
//! (crates/sparkflash-engram owns them, not the engine) and the pinned embedding is 1.32 GB, so a
//! reader that cannot tell a header from a tensor would allocate a 98 GB buffer by accident.
//! [SafeTensorsSource] therefore answers names() and spec() from the index and the headers alone,
//! and read(name) enforces a per-tensor limit - [DEFAULT_LIMIT] by default - before it allocates.
//!
//! Which names the loader declines to read, the Engram tables above all, is a policy the loader
//! owns; this module gives that policy its mechanism in three parts: the configurable limit, a
//! refusal list for names that must be refused whatever the limit, and header(name) so a policy can
//! see a tensor's declared size without reading a byte of it.
//!
//! Every malformed input is a [WeightError] and never a panic. Errors name what was asked for: a
//! container-level call names the file it was given, and the source names the tensor, with the
//! shard kept in the detail.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::Read;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};

use serde::Deserialize;

use crate::dtype::DType;
use crate::tensor::TensorSpec;
use crate::weights::{WeightError, WeightSource};

/// The largest header this reader will believe, and therefore the largest it will allocate for one.
///
/// A shard header is a few kilobytes (96,085 names across 48 shards is two thousand tensors each);
/// 64 MiB is far above any real header and far below anything that could exhaust a machine. A
/// declared length above this is rejected before a buffer is allocated, which is the point: the
/// length comes from the file and a truncated or hostile shard can claim any number at all.
pub const MAX_HEADER_BYTES: u64 = 64 * 1024 * 1024;

/// The default per-tensor limit: 2 GiB.
///
/// The largest tensor the engine may legitimately read whole is the pinned embedding, 129,280 x
/// 5,120 BF16 = 1.32 GB, so a 2 GiB limit admits every real tensor and refuses the two Engram
/// tables by a margin of two orders of magnitude. The limit is a constructor argument because the
/// loader's policy, not this module, decides what is too large.
pub const DEFAULT_LIMIT: u128 = 2 * 1024 * 1024 * 1024;

/// Bytes of the little-endian header length that opens every safetensors file.
const HEADER_LENGTH_BYTES: u64 = 8;

/// The one non-tensor key safetensors reserves: an object (or any JSON) describing the shard.
const METADATA_KEY: &str = "__metadata__";

/// The metadata key holding the checkpoint's total byte size.
const TOTAL_SIZE_KEY: &str = "total_size";

// ---------------------------------------------------------------------------------
// The index
// ---------------------------------------------------------------------------------

/// The subset of model.safetensors.index.json this reader needs.
///
/// The index is the authority on what the checkpoint contains: key -> shard file name, plus the
/// metadata block. The released one is 7.4 MB, names 96,085 tensors and totals 510,286,023,000
/// bytes, and reading it costs nothing next to reading a tensor.
#[derive(Debug, Clone, Deserialize)]
pub struct CheckpointIndex {
    /// Tensor name -> the shard file name that holds it.
    #[serde(deserialize_with = "without_duplicates")]
    pub weight_map: BTreeMap<String, String>,
    /// The shard-independent metadata block, kept verbatim. Hugging Face writes total_size here.
    #[serde(default, deserialize_with = "without_duplicates")]
    pub metadata: BTreeMap<String, serde_json::Value>,
}

/// Deserializes a JSON object into a map, refusing a repeated key.
///
/// A `BTreeMap` cannot notice one on its own: its `Deserialize` inserts as it goes, so a second entry
/// with the same key replaces the first and the parser is none the wiser. That matters here because
/// both maps are name tables - an index that names one tensor twice would be read as if it named it
/// once, and whichever shard lost the race would never be consulted. The keys arrive in order through
/// `MapAccess`, which is the only place the repeat is still visible.
fn without_duplicates<'de, D, V>(deserializer: D) -> Result<BTreeMap<String, V>, D::Error>
where
    D: serde::Deserializer<'de>,
    V: serde::Deserialize<'de>,
{
    struct Unique<V>(std::marker::PhantomData<V>);

    impl<'de, V: serde::Deserialize<'de>> serde::de::Visitor<'de> for Unique<V> {
        type Value = BTreeMap<String, V>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("a JSON object without a repeated key")
        }

        fn visit_map<A: serde::de::MapAccess<'de>>(
            self,
            mut map: A,
        ) -> Result<Self::Value, A::Error> {
            let mut out = BTreeMap::new();
            while let Some((key, value)) = map.next_entry::<String, V>()? {
                if out.contains_key(&key) {
                    return Err(serde::de::Error::custom(format!(
                        "the key {key:?} is defined more than once, so one of the two would be dropped"
                    )));
                }
                out.insert(key, value);
            }
            Ok(out)
        }
    }

    deserializer.deserialize_map(Unique(std::marker::PhantomData))
}

impl CheckpointIndex {
    /// Parses an index from a file.
    ///
    /// # Errors
    /// [WeightError::Io] when the file cannot be read, [WeightError::Malformed] when it is not a
    /// safetensors index.
    pub fn load(path: &Path) -> Result<Self, WeightError> {
        let source = file_label(path);
        let bytes = std::fs::read(path).map_err(|error| WeightError::Io {
            name: source.clone(),
            detail: format!("open {}: {error}", path.display()),
        })?;
        Self::from_slice(&bytes, &source)
    }

    /// Parses an index from bytes, naming the source in any error.
    ///
    /// This is the whole parser: [CheckpointIndex::load] only adds the file read, so an index can
    /// be exercised without a filesystem.
    ///
    /// # Errors
    /// [WeightError::Malformed] when the bytes are not a safetensors index or its total_size is not
    /// a non-negative integer.
    pub fn from_slice(bytes: &[u8], source: &str) -> Result<Self, WeightError> {
        let index: Self =
            serde_json::from_slice(bytes).map_err(|error| WeightError::Malformed {
                name: source.to_string(),
                detail: format!("the checkpoint index is not a safetensors index: {error}"),
            })?;
        if let Some(value) = index.metadata.get(TOTAL_SIZE_KEY)
            && value.as_u64().is_none()
        {
            return Err(WeightError::Malformed {
                name: source.to_string(),
                detail: format!(
                    "the index metadata.{TOTAL_SIZE_KEY} is {value}, not a non-negative integer"
                ),
            });
        }
        Ok(index)
    }

    /// The shard file that holds a tensor.
    ///
    /// # Errors
    /// [WeightError::Missing] when the index does not name that tensor.
    pub fn shard_of(&self, name: &str) -> Result<&str, WeightError> {
        self.weight_map
            .get(name)
            .map(String::as_str)
            .ok_or_else(|| WeightError::Missing {
                name: name.to_string(),
            })
    }

    /// The byte size the checkpoint declares for itself, when the metadata carries one.
    ///
    /// A malformed value is rejected by [CheckpointIndex::from_slice], so this accessor is
    /// infallible and simply absent when the index does not state a total.
    pub fn total_size(&self) -> Option<u128> {
        self.metadata
            .get(TOTAL_SIZE_KEY)
            .and_then(serde_json::Value::as_u64)
            .map(u128::from)
    }

    /// The number of tensors the index names.
    pub fn len(&self) -> usize {
        self.weight_map.len()
    }

    /// Whether the index names no tensors at all.
    pub fn is_empty(&self) -> bool {
        self.weight_map.is_empty()
    }

    /// Every tensor name, in sorted order (the map's order, which is stable).
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.weight_map.keys().map(String::as_str)
    }

    /// Every distinct shard file the index points at, in sorted order.
    pub fn shards(&self) -> Vec<&str> {
        self.weight_map
            .values()
            .map(String::as_str)
            .collect::<BTreeSet<&str>>()
            .into_iter()
            .collect()
    }
}

// ---------------------------------------------------------------------------------
// A shard header
// ---------------------------------------------------------------------------------

/// One tensor as its shard's header describes it: the container's dtype word, the shape as stored,
/// and the half-open byte range in the data section.
///
/// The dtype stays a string here because that is what the container says. Mapping it to a
/// [DType] - and doubling the last dimension of an I8 tensor that is really packed FP4 - is what
/// [SafeTensorsSource] does when it answers a [WeightSource].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorHeader {
    /// The container's dtype word: F32, BF16, F16, F8_E4M3, F8_E8M0 or I8.
    pub dtype: String,
    /// The shape as stored on disk. For a packed FP4 weight this is the packed shape, half the
    /// logical one.
    pub shape: Vec<usize>,
    /// [begin, end) relative to the start of the data section.
    pub data_offsets: [u64; 2],
}

impl TensorHeader {
    /// The number of bytes this tensor occupies in the shard.
    pub fn byte_len(&self) -> u64 {
        self.data_offsets[1].saturating_sub(self.data_offsets[0])
    }

    /// The number of elements the stored shape describes (not the logical count of a packed
    /// tensor). A shape whose product leaves u128 saturates rather than wrapping: a header may
    /// declare any dimensions at all, and a reader that overflowed on one would be a reader that
    /// fails on a corrupt file.
    pub fn elements(&self) -> u128 {
        self.shape
            .iter()
            .fold(1u128, |product, &dim| product.saturating_mul(dim as u128))
    }
}

/// The parsed header of one shard: its tensors, its metadata, where its data begins and how long
/// the file is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafeTensorsHeader {
    /// Tensor descriptors by full name, in sorted order. The reserved __metadata__ key is not here.
    pub tensors: BTreeMap<String, TensorHeader>,
    /// The shard's __metadata__ value, kept verbatim, when the shard has one.
    pub metadata: Option<serde_json::Value>,
    /// Absolute offset of the first data byte: 8 + header length.
    pub data_start: u64,
    /// The length of the shard file, so every offset can be checked against it.
    pub file_len: u64,
}

/// The raw JSON shape of one tensor entry.
#[derive(Deserialize)]
struct RawDescriptor {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

impl SafeTensorsHeader {
    /// Parses a header from the first bytes of a container, with the file's length for the bounds
    /// checks.
    ///
    /// prefix must start with the eight-byte little-endian header length and hold the whole header
    /// block; extra bytes are ignored, so a caller may pass a whole file. Nothing is allocated here
    /// that the declared length controls: the length is bounded by [MAX_HEADER_BYTES] and against
    /// file_len before any descriptor is read.
    ///
    /// # Errors
    /// [WeightError::Malformed] for a truncated prefix, a header length outside the bound, a header
    /// that is not a JSON object, an entry that is neither __metadata__ nor a valid descriptor, a
    /// reversed or out-of-file byte range, or a dimension this machine cannot address.
    pub fn from_container(prefix: &[u8], file_len: u64, source: &str) -> Result<Self, WeightError> {
        if (prefix.len() as u64) < HEADER_LENGTH_BYTES {
            return Err(WeightError::Malformed {
                name: source.to_string(),
                detail: format!(
                    "a safetensors file opens with an {HEADER_LENGTH_BYTES}-byte header length, but only {} bytes are present",
                    prefix.len()
                ),
            });
        }
        let mut length_bytes = [0u8; HEADER_LENGTH_BYTES as usize];
        length_bytes.copy_from_slice(&prefix[..HEADER_LENGTH_BYTES as usize]);
        let header_len = u64::from_le_bytes(length_bytes);
        let block = checked_header_len(header_len, file_len, source)?;
        if (prefix.len() as u64) < block {
            return Err(WeightError::Malformed {
                name: source.to_string(),
                detail: format!(
                    "the header block is {} bytes but the container declares {block}",
                    prefix.len()
                ),
            });
        }
        // Parsed as entries rather than as a map, because a map cannot see a repeated name: it
        // inserts as it goes and the second descriptor silently replaces the first, leaving the file
        // read as if it said something it did not. The safetensors header is a flat object of tensor
        // names, so a repeat is two descriptors claiming one tensor.
        let entries: HeaderEntries = serde_json::from_slice(
            &prefix[HEADER_LENGTH_BYTES as usize..block as usize],
        )
        .map_err(|error| WeightError::Malformed {
            name: source.to_string(),
            detail: format!("the safetensors header is not a JSON object: {error}"),
        })?;
        // Checked above: block <= file_len, so the subtraction cannot underflow.
        let data_len = file_len - block;
        let mut metadata = None;
        let mut tensors = BTreeMap::new();
        for (name, value) in entries.0 {
            if name == METADATA_KEY {
                if metadata.is_some() {
                    return Err(WeightError::Malformed {
                        name: name.clone(),
                        detail: format!("the header reserves {METADATA_KEY} twice"),
                    });
                }
                metadata = Some(value);
                continue;
            }
            if tensors.contains_key(&name) {
                return Err(WeightError::Malformed {
                    name: name.clone(),
                    detail: format!(
                        "the header describes {name} twice, so one of the two descriptors would be dropped"
                    ),
                });
            }
            let descriptor: RawDescriptor =
                serde_json::from_value(value).map_err(|error| WeightError::Malformed {
                    name: name.clone(),
                    detail: format!("{name} is not a safetensors tensor descriptor: {error}"),
                })?;
            let shape = descriptor
                .shape
                .iter()
                .map(|&dim| {
                    usize::try_from(dim).map_err(|_| WeightError::Malformed {
                        name: name.clone(),
                        detail: format!("dimension {dim} does not fit this machine"),
                    })
                })
                .collect::<Result<Vec<usize>, WeightError>>()?;
            let [begin, end] = descriptor.data_offsets;
            if begin > end {
                return Err(WeightError::Malformed {
                    name: name.clone(),
                    detail: format!("data_offsets [{begin}, {end}) are reversed"),
                });
            }
            if end > data_len {
                return Err(WeightError::Malformed {
                    name: name.clone(),
                    detail: format!(
                        "the tensor ends at byte {end} of the data section but the shard holds {data_len} bytes of data"
                    ),
                });
            }
            tensors.insert(
                name,
                TensorHeader {
                    dtype: descriptor.dtype,
                    shape,
                    data_offsets: [begin, end],
                },
            );
        }
        Ok(Self {
            tensors,
            metadata,
            data_start: block,
            file_len,
        })
    }

    /// Reads only the header of a shard on disk.
    ///
    /// The file is opened, its length is taken from the metadata, the eight-byte header length is
    /// read, and then exactly header_len more bytes: the tensor data is never touched, and the
    /// declared length is bounded before the buffer for it exists.
    ///
    /// # Errors
    /// [WeightError::Io] when the file cannot be opened, statted or read;
    /// [WeightError::Malformed] when it is not a safetensors container or a descriptor is broken.
    pub fn read(path: &Path) -> Result<Self, WeightError> {
        let source = file_label(path);
        let mut file = File::open(path).map_err(|error| WeightError::Io {
            name: source.clone(),
            detail: format!("open {}: {error}", path.display()),
        })?;
        let file_len = file
            .metadata()
            .map_err(|error| WeightError::Io {
                name: source.clone(),
                detail: format!("stat {}: {error}", path.display()),
            })?
            .len();
        let mut length_bytes = [0u8; HEADER_LENGTH_BYTES as usize];
        file.read_exact(&mut length_bytes).map_err(|error| {
            if error.kind() == std::io::ErrorKind::UnexpectedEof {
                // Too short to be a container at all: that is a fact about the file, not a failed
                // read of the medium.
                WeightError::Malformed {
                    name: source.clone(),
                    detail: format!(
                        "a safetensors file opens with an {HEADER_LENGTH_BYTES}-byte header length, but {} is only {file_len} bytes",
                        path.display()
                    ),
                }
            } else {
                WeightError::Io {
                    name: source.clone(),
                    detail: format!("read the header length of {}: {error}", path.display()),
                }
            }
        })?;
        let header_len = u64::from_le_bytes(length_bytes);
        // Bounded before the allocation below, from the same checks the pure parser repeats.
        let block = checked_header_len(header_len, file_len, &source)?;
        let mut prefix = vec![0u8; block as usize];
        prefix[..HEADER_LENGTH_BYTES as usize].copy_from_slice(&length_bytes);
        file.read_exact(&mut prefix[HEADER_LENGTH_BYTES as usize..])
            .map_err(|error| WeightError::Io {
                name: source.clone(),
                detail: format!("read the header of {}: {error}", path.display()),
            })?;
        Self::from_container(&prefix, file_len, &source)
    }

    /// The declared header length in bytes.
    pub fn header_len(&self) -> u64 {
        self.data_start - HEADER_LENGTH_BYTES
    }

    /// The number of bytes in the file after the header.
    pub fn data_len(&self) -> u64 {
        self.file_len - self.data_start
    }

    /// The number of tensors the header describes.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether the header describes no tensors (a shard that holds only metadata, for example).
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// One tensor's descriptor.
    ///
    /// # Errors
    /// [WeightError::Missing] when the header does not describe that tensor.
    pub fn tensor(&self, name: &str) -> Result<&TensorHeader, WeightError> {
        self.tensors.get(name).ok_or_else(|| WeightError::Missing {
            name: name.to_string(),
        })
    }
}

/// A parsed header kept as its entries, in file order, so a repeated key is still visible.
///
/// [BTreeMap] would lose it: its `Deserialize` inserts as it goes and a second entry with the same key
/// replaces the first. The header is the only description of what a shard holds, so a repeated tensor
/// name is a file that describes itself two ways and the reader must refuse it rather than pick one.
///
/// The check is on the header's own keys. A repeated key *inside* one tensor's descriptor -
/// `{"dtype":"BF16","dtype":"F32"}` - is not seen here, because the descriptor is parsed out of the
/// entry's JSON value, in which the repeat has already been collapsed; that is true of every JSON
/// parser there is. It is not a hole in the audit: whichever copy wins is then checked against the
/// plan's dtype, shape and byte length like any other descriptor, so a repeat cannot smuggle past the
/// loader a dtype or a shape the plan does not name.
struct HeaderEntries(Vec<(String, serde_json::Value)>);

impl<'de> serde::Deserialize<'de> for HeaderEntries {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Entries;

        impl<'de> serde::de::Visitor<'de> for Entries {
            type Value = HeaderEntries;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a JSON object of tensor names")
            }

            fn visit_map<A: serde::de::MapAccess<'de>>(
                self,
                mut map: A,
            ) -> Result<Self::Value, A::Error> {
                let mut entries = Vec::new();
                while let Some(entry) = map.next_entry::<String, serde_json::Value>()? {
                    entries.push(entry);
                }
                Ok(HeaderEntries(entries))
            }
        }

        deserializer.deserialize_map(Entries)
    }
}

/// Validates a declared header length against the bound and the file, and returns the offset where
/// the data section starts.
fn checked_header_len(header_len: u64, file_len: u64, source: &str) -> Result<u64, WeightError> {
    // The shortest possible JSON object is two bytes, "{}"; safetensors pads the header to eight
    // bytes, so a real one is longer still, but a header of two bytes or fewer cannot describe a tensor
    // and is refused here rather than parsed.
    if header_len <= 2 || header_len > MAX_HEADER_BYTES {
        return Err(WeightError::Malformed {
            name: source.to_string(),
            detail: format!(
                "the declared header length {header_len} is not a safetensors header (the bound is 3..={MAX_HEADER_BYTES} bytes)"
            ),
        });
    }
    let block = HEADER_LENGTH_BYTES + header_len;
    if block > file_len {
        return Err(WeightError::Malformed {
            name: source.to_string(),
            detail: format!(
                "the header claims {block} bytes but the shard is only {file_len} bytes"
            ),
        });
    }
    Ok(block)
}

// ---------------------------------------------------------------------------------
// The source
// ---------------------------------------------------------------------------------

/// Range count above which `read_ranges` reads the tensor's whole stored region in one pass and
/// gathers the wanted ranges out of that buffer, instead of one seek+read per range.
const GATHER_FROM_WHOLE_TENSOR_RANGES: usize = 16;

/// Largest stored tensor region [SafeTensorsSource::read_ranges] may materialise in one buffer to
/// serve a many-range read.
///
/// The shortcut exists for column slices of ordinary resident tensors and is only safe while the
/// region is small. A paged read of the Engram embed table asks for 24 rows of a 98.3 GB tensor;
/// materialising the table to gather them was an out-of-memory kill (m1y15, first Engram read). The
/// cap is far above every tensor the four-rank arena boot reads through this call - the largest
/// many-range one is an attention-row shard at 20,992,000 bytes - and far below the tables, so the
/// boot keeps its whole-gather path while a paged table goes range by range.
pub const GATHER_WHOLE_TENSOR_MAX_BYTES: u64 = 1 << 30;

/// Whether a range read of a total_bytes stored region with range_count ranges may materialise the
/// region in one buffer.
///
/// This is the policy [SafeTensorsSource::read_ranges] applies: a many-range read takes the
/// whole-tensor path only when the region fits the memory cap. It is one named rule so the reader
/// and its tests cannot disagree about the branch, and so the cap is set in exactly one place.
#[must_use]
pub fn gathers_whole_tensor(
    total_bytes: u64,
    range_count: usize,
    whole_tensor_max_bytes: u64,
) -> bool {
    range_count > GATHER_FROM_WHOLE_TENSOR_RANGES
        && total_bytes > 0
        && total_bytes <= whole_tensor_max_bytes
}

/// A [WeightSource] over a directory of safetensors shards plus the checkpoint index.
///
/// The index answers names() and which shard holds a name; the shard's header answers the dtype,
/// the shape and the byte range; only read(name) touches tensor data, and it reads exactly the
/// declared range - no more - after checking the limit.
///
/// Headers are cached by shard file name behind a mutex, so asking for a hundred thousand specs
/// parses 48 headers once each. A spec for a tensor whose shard is absent is an [WeightError::Io]:
/// the index is a promise about a directory, and this source does not populate it.
#[derive(Debug)]
pub struct SafeTensorsSource {
    dir: PathBuf,
    index: CheckpointIndex,
    limit: u128,
    refused: BTreeSet<String>,
    headers: Mutex<BTreeMap<String, Arc<SafeTensorsHeader>>>,
    files: Mutex<BTreeMap<String, Arc<File>>>,
}

impl SafeTensorsSource {
    /// A source over a directory, with [DEFAULT_LIMIT] and no refused names.
    pub fn new(dir: impl Into<PathBuf>, index: CheckpointIndex) -> Self {
        Self {
            dir: dir.into(),
            index,
            limit: DEFAULT_LIMIT,
            refused: BTreeSet::new(),
            headers: Mutex::new(BTreeMap::new()),
            files: Mutex::new(BTreeMap::new()),
        }
    }

    /// A source over a directory, with the index loaded from a file.
    ///
    /// # Errors
    /// [WeightError::Io] or [WeightError::Malformed] from [CheckpointIndex::load].
    pub fn open(dir: impl Into<PathBuf>, index_path: &Path) -> Result<Self, WeightError> {
        Ok(Self::new(dir, CheckpointIndex::load(index_path)?))
    }

    /// Sets the largest tensor this source will read.
    ///
    /// The limit is the loader's policy hook: the two Engram tables are roughly 98 GB each, so any
    /// limit a machine can allocate refuses them. Values above the address space are capped at
    /// usize::MAX internally, and the error reports the configured limit.
    pub fn with_limit(mut self, limit: u128) -> Self {
        self.limit = limit;
        self
    }

    /// Adds names this source refuses to read, whatever the limit.
    ///
    /// This is how the loader states a by-name policy - the two Engram tables belong to
    /// crates/sparkflash-engram and are never materialised here - without touching this module's
    /// size arithmetic. A refused name is reported as [WeightError::Refused], and it is refused
    /// before its shard is opened, because the refusal is about the name: it costs no file handle and
    /// no header read, and it does not need to invent a size for a tensor that was never described.
    /// Refusal applies to read only: names(), spec() and header() still describe a refused tensor,
    /// because a policy has to be able to see what it is refusing.
    pub fn with_refused<I, S>(mut self, names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.refused.extend(names.into_iter().map(Into::into));
        self
    }

    /// The largest tensor this source will read.
    pub fn limit(&self) -> u128 {
        self.limit
    }

    /// The names this source refuses.
    pub fn refused(&self) -> &BTreeSet<String> {
        &self.refused
    }

    /// The shard directory.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The checkpoint index.
    pub fn index(&self) -> &CheckpointIndex {
        &self.index
    }

    /// The path this source resolves a shard file name to.
    pub fn shard_path(&self, shard: &str) -> PathBuf {
        self.dir.join(shard)
    }

    /// The parsed header of the shard holding a name, read once and cached.
    ///
    /// This is the cheap way for a policy to see a tensor's declared byte length: it reads a few
    /// kilobytes of header and no tensor data at all. A failure names the tensor that was asked for,
    /// with the shard kept in the detail.
    ///
    /// # Errors
    /// [WeightError::Missing] when the index does not name the tensor,
    /// [WeightError::Malformed] when the index names a shard outside the directory or the shard's
    /// own header is malformed,
    /// [WeightError::Io] when the shard is missing or unreadable.
    pub fn header(&self, name: &str) -> Result<Arc<SafeTensorsHeader>, WeightError> {
        self.resolve(name).map(|(_, header)| header)
    }

    /// The header of one shard file name, read once and cached.
    fn header_of_shard(&self, shard: &str) -> Result<Arc<SafeTensorsHeader>, WeightError> {
        if let Some(cached) = self.headers().get(shard) {
            return Ok(Arc::clone(cached));
        }
        // The index names the file, so a name that is not a plain file name would escape the shard
        // directory. This checkpoint's index never does, and neither may a reader that trusts it.
        let escapes = shard.is_empty()
            || shard == "."
            || shard == ".."
            || shard
                .chars()
                .any(|c| c == '/' || c == std::path::MAIN_SEPARATOR);
        if escapes {
            return Err(WeightError::Malformed {
                name: shard.to_string(),
                detail: format!(
                    "the index names {shard} as a shard file, which is outside the shard directory"
                ),
            });
        }
        let header = Arc::new(SafeTensorsHeader::read(&self.shard_path(shard))?);
        self.headers()
            .insert(shard.to_string(), Arc::clone(&header));
        Ok(header)
    }

    /// The header cache, without a panic on a poisoned mutex.
    fn headers(&self) -> MutexGuard<'_, BTreeMap<String, Arc<SafeTensorsHeader>>> {
        self.headers
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// The open file handle cache, without a panic on a poisoned mutex.
    fn files(&self) -> MutexGuard<'_, BTreeMap<String, Arc<File>>> {
        self.files
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Opens or reuses a cached file handle for one shard.
    fn file_of_shard(&self, shard: &str, tensor_name: &str) -> Result<Arc<File>, WeightError> {
        if let Some(cached) = self.files().get(shard) {
            return Ok(Arc::clone(cached));
        }
        let path = self.shard_path(shard);
        let file = Arc::new(File::open(&path).map_err(|error| WeightError::Io {
            name: tensor_name.to_string(),
            detail: format!("open {}: {error}", path.display()),
        })?);
        self.files().insert(shard.to_string(), Arc::clone(&file));
        Ok(file)
    }

    /// Pre-parses and caches the safetensors header of every distinct shard file in the index.
    ///
    /// # Errors
    /// [WeightError::Malformed] or [WeightError::Io] if any shard header cannot be read or parsed.
    pub fn preload_headers(&self) -> Result<(), WeightError> {
        let shards: BTreeSet<String> = self.index.weight_map.values().cloned().collect();
        for shard in shards {
            let _ = self.header_of_shard(&shard)?;
        }
        Ok(())
    }

    /// The shard file name and parsed header for a tensor.
    ///
    /// A shard-level failure is relabelled here: [SafeTensorsHeader] is asked for a file, while
    /// [WeightSource] is asked for a tensor, and each error names what was asked for.
    fn resolve(&self, name: &str) -> Result<(&str, Arc<SafeTensorsHeader>), WeightError> {
        let shard = self.index.shard_of(name)?;
        let header = self
            .header_of_shard(shard)
            .map_err(|error| relabel_to(error, name, shard))?;
        Ok((shard, header))
    }
}

impl SafeTensorsSource {
    /// Reads named byte ranges of one tensor.
    ///
    /// This is the call a paged reader makes for a table too large to hold. It deliberately does not
    /// consult this source's refusal list: the list exists to stop a whole-table read under the
    /// element limit, and a range read is the alternative to that limit, not a bypass of a check that
    /// would otherwise apply. Each range is checked against the bytes the shard actually stores for
    /// the tensor - the header's data_offsets, not the plan's logical size, because a packed expert
    /// stores half as many bytes as its logical shape - and the ranges are concatenated in the order
    /// given. A many-range read may read the whole stored region to serve the ranges when the region
    /// fits [GATHER_WHOLE_TENSOR_MAX_BYTES]; [SafeTensorsSource::read_ranges_paged] suppresses that
    /// shortcut for a table that must never be materialised.
    ///
    /// # Errors
    /// [WeightError::Missing] when the name is not in the index or its shard, [WeightError::Malformed]
    /// when a range falls outside the tensor, [WeightError::Io] when the shard cannot be read.
    pub fn read_ranges(&self, name: &str, ranges: &[(u64, usize)]) -> Result<Vec<u8>, WeightError> {
        self.read_ranges_with_policy(name, ranges, GATHER_WHOLE_TENSOR_MAX_BYTES)
    }

    /// [SafeTensorsSource::read_ranges] with every whole-tensor fast path suppressed: each range is
    /// one seek and one read, whatever the range count, so a paged read can never materialise a
    /// table whose stored region is tens of GB.
    ///
    /// # Errors
    /// As [SafeTensorsSource::read_ranges].
    pub fn read_ranges_paged(
        &self,
        name: &str,
        ranges: &[(u64, usize)],
    ) -> Result<Vec<u8>, WeightError> {
        self.read_ranges_with_policy(name, ranges, 0)
    }

    /// Reads named byte ranges of one tensor directly into caller-provided `dst` without allocating
    /// an output `Vec<u8>`.
    ///
    /// # Errors
    /// As [SafeTensorsSource::read_ranges], plus [WeightError::Malformed] if `dst.len()` does not
    /// equal the sum of the range lengths.
    pub fn read_ranges_into(
        &self,
        name: &str,
        ranges: &[(u64, usize)],
        dst: &mut [u8],
    ) -> Result<(), WeightError> {
        self.read_ranges_into_with_policy(name, ranges, dst, GATHER_WHOLE_TENSOR_MAX_BYTES)
    }

    /// The policy-parameterised form of [SafeTensorsSource::read_ranges_into].
    ///
    /// # Errors
    /// As [SafeTensorsSource::read_ranges_into].
    pub fn read_ranges_into_with_policy(
        &self,
        name: &str,
        ranges: &[(u64, usize)],
        dst: &mut [u8],
        whole_tensor_max_bytes: u64,
    ) -> Result<(), WeightError> {
        let expected_out: usize = ranges.iter().map(|(_, len)| *len).sum();
        if dst.len() != expected_out {
            return Err(WeightError::Malformed {
                name: name.to_string(),
                detail: format!(
                    "destination slice is {} bytes, but the requested ranges sum to {expected_out}",
                    dst.len()
                ),
            });
        }
        let (shard, header) = self.resolve(name)?;
        let descriptor = header.tensor(name)?;
        let (begin, finish) = (descriptor.data_offsets[0], descriptor.data_offsets[1]);
        let total = finish.saturating_sub(begin);
        let start = header
            .data_start
            .checked_add(descriptor.data_offsets[0])
            .ok_or_else(|| WeightError::Malformed {
                name: name.to_string(),
                detail: format!(
                    "the tensor starts at byte {} of a data section that begins at {}",
                    descriptor.data_offsets[0], header.data_start
                ),
            })?;
        let path = self.shard_path(shard);
        let file = self.file_of_shard(shard, name)?;
        debug_assert!(
            !gathers_whole_tensor(total, ranges.len(), whole_tensor_max_bytes)
                || total <= whole_tensor_max_bytes
        );
        if gathers_whole_tensor(total, ranges.len(), whole_tensor_max_bytes) {
            let total_len = usize::try_from(total).map_err(|_| WeightError::Malformed {
                name: name.to_string(),
                detail: format!("tensor region of {total} bytes does not fit usize"),
            })?;
            let mut whole = vec![0u8; total_len];
            file.read_exact_at(&mut whole, start)
                .map_err(|error| match error.kind() {
                    std::io::ErrorKind::UnexpectedEof => WeightError::Malformed {
                        name: name.to_string(),
                        detail: format!(
                            "{} declared {total_len} bytes from byte {start} but holds fewer",
                            path.display()
                        ),
                    },
                    _ => WeightError::Io {
                        name: name.to_string(),
                        detail: format!("read {}: {error}", path.display()),
                    },
                })?;
            let mut cursor = 0usize;
            for &(offset, len) in ranges {
                let begin = usize::try_from(offset).map_err(|_| WeightError::Malformed {
                    name: name.to_string(),
                    detail: format!("range offset {offset} does not fit usize"),
                })?;
                let end = begin
                    .checked_add(len)
                    .ok_or_else(|| WeightError::Malformed {
                        name: name.to_string(),
                        detail: format!("range {offset}+{len} overflows"),
                    })?;
                if end > whole.len() {
                    return Err(WeightError::Malformed {
                        name: name.to_string(),
                        detail: format!(
                            "range {offset}..{end} is outside the tensor's {total} stored bytes"
                        ),
                    });
                }
                dst[cursor..cursor + len].copy_from_slice(&whole[begin..end]);
                cursor += len;
            }
            return Ok(());
        }
        let mut cursor = 0usize;
        for &(offset, len) in ranges {
            let end = offset
                .checked_add(len as u64)
                .ok_or_else(|| WeightError::Malformed {
                    name: name.to_string(),
                    detail: format!("range {offset}+{len} overflows a byte offset"),
                })?;
            if end > total {
                return Err(WeightError::Malformed {
                    name: name.to_string(),
                    detail: format!(
                        "range {offset}..{end} is outside the tensor's {total} stored bytes"
                    ),
                });
            }
            let at = start
                .checked_add(offset)
                .ok_or_else(|| WeightError::Malformed {
                    name: name.to_string(),
                    detail: format!("byte {start}+{offset} overflows a file offset"),
                })?;
            file.read_exact_at(&mut dst[cursor..cursor + len], at)
                .map_err(|error| match error.kind() {
                    std::io::ErrorKind::UnexpectedEof => WeightError::Malformed {
                        name: name.to_string(),
                        detail: format!(
                            "{} declared {end} bytes from byte {at} but holds fewer",
                            path.display()
                        ),
                    },
                    _ => WeightError::Io {
                        name: name.to_string(),
                        detail: format!("read {}: {error}", path.display()),
                    },
                })?;
            cursor += len;
        }
        Ok(())
    }

    /// The policy-parameterised form of [SafeTensorsSource::read_ranges].
    ///
    /// whole_tensor_max_bytes is the largest stored region a many-range read may materialise in one
    /// buffer. [SafeTensorsSource::read_ranges] passes [GATHER_WHOLE_TENSOR_MAX_BYTES] and
    /// [SafeTensorsSource::read_ranges_paged] passes 0, which forces every range to its own read.
    /// Public so a tiny container can exercise both branches of [gathers_whole_tensor].
    ///
    /// # Errors
    /// As [SafeTensorsSource::read_ranges].
    pub fn read_ranges_with_policy(
        &self,
        name: &str,
        ranges: &[(u64, usize)],
        whole_tensor_max_bytes: u64,
    ) -> Result<Vec<u8>, WeightError> {
        let total_out: usize = ranges.iter().map(|(_, len)| *len).sum();
        let mut out = vec![0u8; total_out];
        self.read_ranges_into_with_policy(name, ranges, &mut out, whole_tensor_max_bytes)?;
        Ok(out)
    }
}

impl WeightSource for SafeTensorsSource {
    fn read_ranges(
        &self,
        name: &str,
        ranges: &[(u64, usize)],
    ) -> Option<Result<Vec<u8>, WeightError>> {
        Some(SafeTensorsSource::read_ranges(self, name, ranges))
    }

    /// The paged form reads range by range and never buffers a whole region, so a source with a
    /// fast whole-region path cannot turn a paged read into a large allocation.
    fn read_ranges_paged(
        &self,
        name: &str,
        ranges: &[(u64, usize)],
    ) -> Option<Result<Vec<u8>, WeightError>> {
        Some(SafeTensorsSource::read_ranges_paged(self, name, ranges))
    }

    /// Every name in the index, in sorted order.
    fn names(&self) -> Vec<String> {
        self.index.names().map(str::to_string).collect()
    }

    /// The dtype and logical shape of one tensor, from the index and the shard header alone.
    ///
    /// # Errors
    /// [WeightError::Missing] when the index or the shard header does not have the name,
    /// [WeightError::Malformed] when the dtype is not one this checkpoint uses or the header's byte
    /// length disagrees with the dtype and shape it declares, [WeightError::Io] when the shard
    /// cannot be read.
    fn spec(&self, name: &str) -> Result<TensorSpec, WeightError> {
        let (_, header) = self.resolve(name)?;
        let descriptor = header.tensor(name)?;
        logical_spec(name, descriptor)
    }

    /// Reads exactly the bytes of one tensor.
    ///
    /// What is checked when, in order: a name on the refusal list is refused before its shard is even
    /// opened, because that list exists to keep the 98 GB Engram tables from being described at all;
    /// then the header supplies the declared size, which may cost a header read but never a tensor
    /// buffer; then the limit is applied before the buffer is allocated. The read spans the tensor's
    /// declared range only: a shard with padding or other tensors after it never contributes a byte to
    /// the result.
    ///
    /// # Errors
    /// [WeightError::Refused] when the name is on the refusal list,
    /// [WeightError::TooLarge] when the tensor is over the limit,
    /// [WeightError::Missing] when it is not in the index or not in its shard's header,
    /// [WeightError::Malformed] when the header disagrees with the index or the dtype,
    /// [WeightError::Io] when the shard cannot be read.
    fn read(&self, name: &str) -> Result<Vec<u8>, WeightError> {
        // Refused first and by name alone: this is the check that must not cost a file open. The
        // Engram tables are refused here, and a refusal that still opened their shard would defeat
        // the point of naming them.
        if self.refused.contains(name) {
            return Err(WeightError::Refused {
                name: name.to_string(),
                detail:
                    "it is on this source's refusal list, which is how the 98 GB Engram tables stay"
                        .to_string()
                        + " out of the engine's loader",
            });
        }
        let spec = self.spec(name)?;
        let bytes = spec.storage_bytes();
        let readable = self.limit.min(usize::MAX as u128);
        if bytes > readable {
            return Err(WeightError::TooLarge {
                name: name.to_string(),
                bytes,
                limit: self.limit,
            });
        }
        let (shard, header) = self.resolve(name)?;
        let descriptor = header.tensor(name)?;
        let start = header
            .data_start
            .checked_add(descriptor.data_offsets[0])
            .ok_or_else(|| WeightError::Malformed {
                name: name.to_string(),
                detail: format!(
                    "the tensor starts at byte {} of a data section that begins at {}",
                    descriptor.data_offsets[0], header.data_start
                ),
            })?;
        // Safe: bytes <= readable <= usize::MAX.
        let mut out = vec![0u8; bytes as usize];
        let path = self.shard_path(shard);
        let file = self.file_of_shard(shard, name)?;
        file.read_exact_at(&mut out, start)
            .map_err(|error| match error.kind() {
                std::io::ErrorKind::UnexpectedEof => WeightError::Malformed {
                    name: name.to_string(),
                    detail: format!(
                        "{} declared {} bytes at byte {start} but holds fewer",
                        path.display(),
                        out.len()
                    ),
                },
                _ => WeightError::Io {
                    name: name.to_string(),
                    detail: format!("read {}: {error}", path.display()),
                },
            })?;
        Ok(out)
    }

    /// Whether the index names this tensor.
    ///
    /// Overridden because the index answers it: a policy can ask whether a name exists over all
    /// 96,085 of them on a machine that holds none of the 48 shards, and the answer is the one
    /// [WeightSource::names] gives.
    fn contains(&self, name: &str) -> bool {
        self.index.weight_map.contains_key(name)
    }
}

/// Re-labels a shard-level failure as the tensor it was asked for.
///
/// [SafeTensorsHeader] is handed a file, so its errors name the file; a [WeightSource] is handed a
/// tensor, so its errors name the tensor and keep the shard in the detail.
fn relabel_to(error: WeightError, name: &str, shard: &str) -> WeightError {
    match error {
        WeightError::Io { detail, .. } => WeightError::Io {
            name: name.to_string(),
            detail: format!("shard {shard}: {detail}"),
        },
        WeightError::Malformed { detail, .. } => WeightError::Malformed {
            name: name.to_string(),
            detail: format!("shard {shard}: {detail}"),
        },
        WeightError::TooLarge { bytes, limit, .. } => WeightError::TooLarge {
            name: name.to_string(),
            bytes,
            limit,
        },
        WeightError::Refused { detail, .. } => WeightError::Refused {
            name: name.to_string(),
            detail,
        },
        // A plan-level failure has no tensor to relabel, and none is created by a shard read.
        plan @ WeightError::Plan { .. } => plan,
        WeightError::Missing { name } => WeightError::Missing { name },
    }
}

/// Maps a container descriptor to the logical tensor the plan and the arithmetic use.
///
/// The one transformation is the packed FP4 case. The checkpoint stores routed and MTP expert
/// weights as FP4 E2M1 packed two values per byte, and the container calls those bytes I8 with the
/// packed shape; the plan keeps the LOGICAL shape, so the last dimension doubles (dtype.rs,
/// quantisation). Everything else maps word for word. A word this checkpoint does not use is a
/// malformed tensor rather than a panic, and the header's own byte length has to agree with the
/// dtype and shape it declares, which catches a header whose offsets and descriptor disagree.
fn logical_spec(name: &str, descriptor: &TensorHeader) -> Result<TensorSpec, WeightError> {
    let mut shape = descriptor.shape.clone();
    let dtype = match descriptor.dtype.as_str() {
        "F32" => DType::F32,
        "F16" => DType::F16,
        "BF16" => DType::BF16,
        "F8_E4M3" => DType::F8E4M3,
        "F8_E8M0" => DType::F8E8M0,
        "I8" => {
            let last = shape.last_mut().ok_or_else(|| WeightError::Malformed {
                name: name.to_string(),
                detail: "an I8 tensor with no dimensions cannot be a packed FP4 weight".to_string(),
            })?;
            *last = last.checked_mul(2).ok_or_else(|| WeightError::Malformed {
                name: name.to_string(),
                detail: format!("the packed dimension {last} doubles past this machine's range"),
            })?;
            DType::F4E2M1
        }
        other => {
            return Err(WeightError::Malformed {
                name: name.to_string(),
                detail: format!(
                    "the dtype {other} is not one this checkpoint stores (F32, BF16, F16, F8_E4M3, F8_E8M0, or I8 packing FP4)"
                ),
            });
        }
    };
    // The container's byte count follows from the STORED shape, not the logical one, and it is
    // computed in u128 with saturation: a header may declare absurd dimensions, and this has to be
    // a typed error rather than an overflow. Once this check passes, the logical shape's own
    // storage_bytes is bounded by the file length and cannot overflow anything either.
    let stored = u128::from(descriptor.byte_len());
    let width = match dtype {
        DType::F32 => 4u128,
        DType::F16 | DType::BF16 => 2,
        DType::F8E4M3 | DType::F8E8M0 | DType::F4E2M1 => 1,
    };
    let needed = descriptor.elements().saturating_mul(width);
    if stored != needed {
        return Err(WeightError::Malformed {
            name: name.to_string(),
            detail: format!(
                "the shard stores {stored} bytes for this tensor but its dtype {} and shape {:?} need {needed}",
                descriptor.dtype, descriptor.shape
            ),
        });
    }
    Ok(TensorSpec {
        name: name.to_string(),
        dtype,
        shape,
    })
}

/// The name a file is reported under in an error: its file name, or its whole path when it has
/// none.
fn file_label(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}
