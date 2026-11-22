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

//! A checkpoint's tensor index, read from safetensors headers and nothing else.
//!
//! A scan has to know every tensor's name, stored shape and dtype before it can decide whether
//! the model description takes them all. It never needs the weights: the safetensors header of
//! a shard already names its tensors, their shapes and their dtypes. This module reads those
//! headers from a local snapshot directory or over HTTP range requests against the Hugging Face
//! Hub, so a 510 GB checkpoint costs a few hundred kilobytes of header reads.
//!
//! The index JSON is not the authority here. A local directory is enumerated from its
//! `*.safetensors` files, and a Hub repository from the index's shard list (or the Hub file
//! listing when it has none); each file's own header then supplies its tensors. That is why a
//! sharded checkpoint and a lone `model.safetensors` take the same path, and why a stale
//! `model.safetensors.index.json` cannot make the reader report a tensor the shards do not
//! describe.
//!
//! A Hub request carries the public repository id and the standard HTTP headers only. No token,
//! local path, config text or caller state is sent: the scan of a public checkpoint must not
//! leak whose machine is asking.

use std::collections::BTreeMap;
use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde_json::Value;

use crate::dtype::DType;
use crate::hub;
use crate::safetensors::{
    CheckpointIndex, INDEX_NAME, MAX_HEADER_BYTES, SINGLE_FILE_NAME, SafeTensorsHeader,
};
use crate::spec::TensorSpec;
use crate::weights::WeightError;

/// One tensor as its shard's header describes it: the container's dtype word, the stored shape
/// and the name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexedTensor {
    /// The checkpoint tensor name.
    pub name: String,
    /// The stored dtype word, exactly as the shard's header spells it (`I8` for packed FP4).
    pub dtype: String,
    /// The stored shape, outermost first.
    pub shape: Vec<usize>,
}

impl IndexedTensor {
    /// The logical tensor this stored descriptor describes, for comparison with a tensor plan.
    ///
    /// The one transformation is safetensors' packed FP4 convention: a shard stores FP4 E2M1
    /// codes as `I8` with the last dimension halved, while a plan keeps the logical shape. A
    /// dtype word this reader does not know is refused by name rather than guessed.
    ///
    /// # Errors
    /// [IndexError::Dtype] when the header's dtype word is not one this checkpoint format uses.
    pub fn logical_spec(&self) -> Result<TensorSpec, IndexError> {
        logical_spec(&self.name, &self.dtype, &self.shape)
    }
}

/// A checkpoint's tensor index, in name order.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TensorIndex {
    /// Where the index was read from: a directory, or `repo@revision` for the Hub.
    pub source: String,
    /// Every tensor, sorted by name.
    pub tensors: Vec<IndexedTensor>,
}

impl TensorIndex {
    /// An index over `tensors`, sorted by name.
    #[must_use]
    pub fn new(source: impl Into<String>, mut tensors: Vec<IndexedTensor>) -> Self {
        tensors.sort_by(|left, right| left.name.cmp(&right.name));
        Self {
            source: source.into(),
            tensors,
        }
    }

    /// The number of tensors the checkpoint describes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether the checkpoint describes no tensors.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Every tensor name, in name order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.iter().map(|tensor| tensor.name.as_str())
    }

    /// One tensor by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&IndexedTensor> {
        self.tensors
            .binary_search_by(|tensor| tensor.name.as_str().cmp(name))
            .ok()
            .map(|index| &self.tensors[index])
    }
}

/// Why a tensor index could not be read.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum IndexError {
    /// The local checkpoint directory could not be enumerated or one of its files read.
    #[error("the tensor index of {subject}: {detail}")]
    Local {
        /// The directory or file that failed.
        subject: String,
        /// What failed.
        detail: String,
    },
    /// The Hub request or its response could not be used.
    #[error("the tensor index of {subject} over the Hub: {detail}")]
    Hub {
        /// The repository and revision that failed.
        subject: String,
        /// What failed.
        detail: String,
    },
    /// A safetensors header is not a header this reader can trust.
    #[error("the safetensors header {subject} is malformed: {detail}")]
    Malformed {
        /// The file or URL whose header is malformed.
        subject: String,
        /// What is wrong with it.
        detail: String,
    },
    /// A header names a dtype this reader does not map.
    #[error(
        "the tensor {name} has the dtype {dtype}, which this reader does not map to a storage format"
    )]
    Dtype {
        /// The tensor name.
        name: String,
        /// The dtype word the header spells.
        dtype: String,
    },
}

/// The logical spec of one stored tensor descriptor.
///
/// # Errors
/// [IndexError::Dtype] when `dtype` is not one of the safetensors words this checkpoint format
/// uses, or when an `I8` tensor has no dimension to double.
pub fn logical_spec(name: &str, dtype: &str, shape: &[usize]) -> Result<TensorSpec, IndexError> {
    let mut shape = shape.to_vec();
    let dtype = match dtype {
        "I8" => {
            let last = shape.last_mut().ok_or_else(|| IndexError::Dtype {
                name: name.to_owned(),
                dtype: "I8 with no dimensions (not a packed FP4 weight)".to_owned(),
            })?;
            *last = last.checked_mul(2).ok_or_else(|| IndexError::Dtype {
                name: name.to_owned(),
                dtype: format!("I8 with a packed dimension {last} that doubles past this machine"),
            })?;
            DType::F4E2M1
        }
        other => DType::from_name(other).ok_or_else(|| IndexError::Dtype {
            name: name.to_owned(),
            dtype: other.to_owned(),
        })?,
    };
    Ok(TensorSpec {
        name: name.to_owned(),
        dtype,
        shape,
    })
}

// ---------------------------------------------------------------------------------
// A local snapshot directory
// ---------------------------------------------------------------------------------

/// Read every `*.safetensors` header in a local checkpoint directory.
///
/// Sharded and unsharded checkpoints take the same path: the directory's safetensors files are
/// enumerated, sorted and read header by header. `model.safetensors.index.json` is deliberately
/// ignored, because a header is the authority on what its file stores.
///
/// # Errors
/// [IndexError::Local] when the directory cannot be read or holds no safetensors file,
/// [IndexError::Malformed] when a header is malformed or one tensor is described by two shards.
pub fn read_local(dir: &Path) -> Result<TensorIndex, IndexError> {
    let entries = std::fs::read_dir(dir).map_err(|error| IndexError::Local {
        subject: dir.display().to_string(),
        detail: format!("read the directory: {error}"),
    })?;
    let mut files: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "safetensors")
        })
        .filter(|path| path.is_file())
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(IndexError::Local {
            subject: dir.display().to_string(),
            detail: format!(
                "no *.safetensors file is present (a checkpoint is either {SINGLE_FILE_NAME} or a sharded set of model-*.safetensors files)"
            ),
        });
    }
    let mut tensors: BTreeMap<String, (String, IndexedTensor)> = BTreeMap::new();
    for file in &files {
        let label = file.display().to_string();
        let header = SafeTensorsHeader::read(file).map_err(|error| header_error(&label, error))?;
        let shard = file
            .file_name()
            .map_or_else(|| label.clone(), |name| name.to_string_lossy().into_owned());
        for (name, descriptor) in header.tensors {
            // The loader checks that the declared byte range agrees with the stored dtype and
            // shape. Do it here too, so a header the index blesses is a header the loader reads.
            crate::safetensors::logical_spec(&name, &descriptor)
                .map_err(|error| header_error(&label, error))?;
            let indexed = IndexedTensor {
                name: name.clone(),
                dtype: descriptor.dtype,
                shape: descriptor.shape,
            };
            if let Some((first, _)) = tensors.get(&name) {
                return Err(IndexError::Malformed {
                    subject: dir.display().to_string(),
                    detail: format!(
                        "the tensor {name} is described by two shards ({first} and {shard}); one name must have one descriptor"
                    ),
                });
            }
            tensors.insert(name, (shard.clone(), indexed));
        }
    }
    Ok(TensorIndex::new(
        dir.display().to_string(),
        tensors.into_values().map(|(_, tensor)| tensor).collect(),
    ))
}

/// Re-labels a header parser's error as the file it was asked to read.
fn header_error(source: &str, error: WeightError) -> IndexError {
    match error {
        WeightError::Malformed { name, detail } => IndexError::Malformed {
            subject: format!("{source} ({name})"),
            detail,
        },
        other => IndexError::Malformed {
            subject: source.to_owned(),
            detail: other.to_string(),
        },
    }
}

// ---------------------------------------------------------------------------------
// The Hub
// ---------------------------------------------------------------------------------

/// Read a Hub repository's tensor index by ranged HTTP reads of each shard's header.
///
/// The repository's index JSON supplies the shard list when it has one; otherwise the Hub's file
/// listing supplies the `*.safetensors` files. Each shard's first eight bytes give the header
/// length and one more ranged read fetches exactly that block. Weights are never read.
///
/// The requests name only the public repository, revision and file. They send no token and no
/// local path, so the operation is safe to run from a console or a shared build host.
///
/// # Errors
/// [IndexError::Hub] when the id or revision is not one this reader accepts, the Hub cannot be
/// reached, the repository has no safetensors file, or a response cannot be used;
/// [IndexError::Malformed] when a header is malformed or a shard name escapes the repository.
pub fn read_hub(repo: &str, revision: &str) -> Result<TensorIndex, IndexError> {
    let source = format!("{repo}@{revision}");
    if !hub::is_model_id(repo) {
        return Err(IndexError::Hub {
            subject: source,
            detail: format!("{repo:?} is not a Hugging Face model id"),
        });
    }
    if !is_revision(revision) {
        return Err(IndexError::Hub {
            subject: source,
            detail: format!(
                "{revision:?} is not a revision this reader accepts (a ref or commit, one path component, with no '..' or separator)"
            ),
        });
    }
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(10))
        .timeout(Duration::from_secs(60))
        .build();
    let base = format!("https://huggingface.co/{repo}/resolve/{revision}");
    let mut shards = match fetch_bytes(&agent, &format!("{base}/{INDEX_NAME}")) {
        Ok(bytes) => {
            let index = CheckpointIndex::from_slice(&bytes, &format!("{source}/{INDEX_NAME}"))
                .map_err(|error| header_error(&format!("{source}/{INDEX_NAME}"), error))?;
            index.shards().into_iter().map(str::to_owned).collect()
        }
        Err(error) if error.status == Some(404) => match hub_safetensors(&agent, repo, revision) {
            Ok(files) if !files.is_empty() => files,
            Ok(_) => vec![SINGLE_FILE_NAME.to_owned()],
            Err(error) => {
                return Err(IndexError::Hub {
                    subject: source,
                    detail: error.detail,
                });
            }
        },
        Err(error) => {
            return Err(IndexError::Hub {
                subject: source,
                detail: error.detail,
            });
        }
    };
    shards.sort();
    shards.dedup();
    if shards.is_empty() {
        return Err(IndexError::Hub {
            subject: source,
            detail: "the repository names a sharded checkpoint with no shard files".to_owned(),
        });
    }
    let mut tensors: BTreeMap<String, (String, IndexedTensor)> = BTreeMap::new();
    for shard in shards {
        if !is_shard_name(&shard) {
            return Err(IndexError::Malformed {
                subject: source.clone(),
                detail: format!(
                    "the checkpoint names {shard} as a shard file, which is outside the repository"
                ),
            });
        }
        let url = format!("{base}/{shard}");
        let header = read_remote_header(&agent, &url)?;
        for (name, descriptor) in header.tensors {
            crate::safetensors::logical_spec(&name, &descriptor)
                .map_err(|error| header_error(&url, error))?;
            if let Some((first, _)) = tensors.get(&name) {
                return Err(IndexError::Malformed {
                    subject: source.clone(),
                    detail: format!(
                        "the tensor {name} is described by two shards ({first} and {shard}); one name must have one descriptor"
                    ),
                });
            }
            tensors.insert(
                name.clone(),
                (
                    shard.clone(),
                    IndexedTensor {
                        name,
                        dtype: descriptor.dtype,
                        shape: descriptor.shape,
                    },
                ),
            );
        }
    }
    Ok(TensorIndex::new(
        source,
        tensors.into_values().map(|(_, tensor)| tensor).collect(),
    ))
}

/// A failed Hub request: its HTTP status when the Hub answered.
#[derive(Debug, Clone, PartialEq, Eq)]
struct FetchError {
    status: Option<u16>,
    detail: String,
}

/// The `*.safetensors` paths in a Hub repository, from the public tree API.
fn hub_safetensors(
    agent: &ureq::Agent,
    repo: &str,
    revision: &str,
) -> Result<Vec<String>, FetchError> {
    let url = format!("https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=1");
    let bytes = fetch_bytes(agent, &url)?;
    let tree: Value = serde_json::from_slice(&bytes).map_err(|error| FetchError {
        status: None,
        detail: format!("{url} is not the Hub's file listing: {error}"),
    })?;
    let files = tree
        .as_array()
        .ok_or_else(|| FetchError {
            status: None,
            detail: format!("{url} is not a JSON array of repository entries"),
        })?
        .iter()
        .filter(|entry| entry.get("type").and_then(Value::as_str) == Some("file"))
        .filter_map(|entry| entry.get("path").and_then(Value::as_str))
        .filter(|path| path.ends_with(".safetensors"))
        .filter(|path| is_shard_name(path))
        .map(str::to_owned)
        .collect();
    Ok(files)
}

/// One shard's header over two ranged reads.
fn read_remote_header(agent: &ureq::Agent, url: &str) -> Result<SafeTensorsHeader, IndexError> {
    let (length_bytes, file_len) =
        fetch_range(agent, url, 0, 7).map_err(|error| IndexError::Hub {
            subject: url.to_owned(),
            detail: error.detail,
        })?;
    if length_bytes.len() != 8 {
        return Err(IndexError::Malformed {
            subject: url.to_owned(),
            detail: format!(
                "a safetensors file opens with an 8-byte header length, but the ranged read returned {} bytes",
                length_bytes.len()
            ),
        });
    }
    let mut declared = [0u8; 8];
    declared.copy_from_slice(&length_bytes);
    let header_len = u64::from_le_bytes(declared);
    if header_len <= 2 || header_len > MAX_HEADER_BYTES {
        return Err(IndexError::Malformed {
            subject: url.to_owned(),
            detail: format!(
                "the declared header length {header_len} is not a safetensors header (the bound is 3..={MAX_HEADER_BYTES} bytes)"
            ),
        });
    }
    let block = 8u64
        .checked_add(header_len)
        .ok_or_else(|| IndexError::Malformed {
            subject: url.to_owned(),
            detail: format!("the declared header length {header_len} overflows a byte offset"),
        })?;
    if block > file_len {
        return Err(IndexError::Malformed {
            subject: url.to_owned(),
            detail: format!(
                "the header claims {block} bytes but the shard is only {file_len} bytes"
            ),
        });
    }
    let (prefix, _) = fetch_range(agent, url, 0, block - 1).map_err(|error| IndexError::Hub {
        subject: url.to_owned(),
        detail: error.detail,
    })?;
    SafeTensorsHeader::from_container(&prefix, file_len, url)
        .map_err(|error| header_error(url, error))
}

/// A plain GET, with no caller identity attached.
fn fetch_bytes(agent: &ureq::Agent, url: &str) -> Result<Vec<u8>, FetchError> {
    let response = agent
        .get(url)
        .set("Accept", "application/json, application/octet-stream")
        .set("Accept-Encoding", "identity")
        .set("User-Agent", "sparkflash-tensor-index")
        .call()
        .map_err(|error| match error {
            ureq::Error::Status(status, _) => FetchError {
                status: Some(status),
                detail: format!("{url} returned HTTP {status}"),
            },
            ureq::Error::Transport(error) => FetchError {
                status: None,
                detail: format!("{url}: {error}"),
            },
        })?;
    let mut body = Vec::new();
    response
        .into_reader()
        .read_to_end(&mut body)
        .map_err(|error| FetchError {
            status: None,
            detail: format!("reading {url}: {error}"),
        })?;
    Ok(body)
}

/// One ranged GET. The returned bytes are `start..=end` and the second value is the file's total
/// length, from `Content-Range` or, when the server ignored the range, `Content-Length`.
fn fetch_range(
    agent: &ureq::Agent,
    url: &str,
    start: u64,
    end: u64,
) -> Result<(Vec<u8>, u64), FetchError> {
    let expected = usize::try_from(end - start + 1).map_err(|_| FetchError {
        status: None,
        detail: format!("the range {start}..={end} does not fit this machine"),
    })?;
    let response = agent
        .get(url)
        .set("Range", &format!("bytes={start}-{end}"))
        .set("Accept", "application/octet-stream")
        .set("Accept-Encoding", "identity")
        .set("User-Agent", "sparkflash-tensor-index")
        .call()
        .map_err(|error| match error {
            ureq::Error::Status(status, _) => FetchError {
                status: Some(status),
                detail: format!("{url} returned HTTP {status} for bytes={start}-{end}"),
            },
            ureq::Error::Transport(error) => FetchError {
                status: None,
                detail: format!("{url}: {error}"),
            },
        })?;
    let status = response.status();
    let content_range = response.header("Content-Range").map(str::to_owned);
    let content_length = response
        .header("Content-Length")
        .and_then(|value| value.parse::<u64>().ok());
    let mut body = Vec::with_capacity(expected);
    response
        .into_reader()
        .take(expected as u64)
        .read_to_end(&mut body)
        .map_err(|error| FetchError {
            status: None,
            detail: format!("reading {url}: {error}"),
        })?;
    if body.len() != expected {
        return Err(FetchError {
            status: None,
            detail: format!(
                "{url} returned {} bytes for bytes={start}-{end}, expected {expected}",
                body.len()
            ),
        });
    }
    let file_len = if status == 206 {
        content_range
            .as_deref()
            .and_then(total_of_content_range)
            .ok_or_else(|| FetchError {
                status: None,
                detail: format!(
                    "{url} answered a range request without a Content-Range total (header {content_range:?})"
                ),
            })?
    } else if status == 200 {
        content_length.ok_or_else(|| FetchError {
            status: None,
            detail: format!(
                "{url} ignored the range and sent no Content-Length, so the shard's length is unknown"
            ),
        })?
    } else {
        return Err(FetchError {
            status: Some(status),
            detail: format!("{url} returned HTTP {status} for bytes={start}-{end}"),
        });
    };
    Ok((body, file_len))
}

/// The total after the slash in `bytes 0-7/1234`.
fn total_of_content_range(value: &str) -> Option<u64> {
    value.rsplit('/').next()?.trim().parse().ok()
}

/// Whether `revision` is one plain path component this reader may put in a URL.
fn is_revision(revision: &str) -> bool {
    !revision.is_empty()
        && revision != "."
        && revision != ".."
        && !revision.contains('/')
        && !revision.contains('\\')
        && !revision.contains("..")
}

/// Whether `name` is one plain shard file name, not a path out of the repository.
fn is_shard_name(name: &str) -> bool {
    !name.is_empty()
        && name != "."
        && name != ".."
        && !name.contains('/')
        && !name.contains('\\')
        && !name.contains("..")
        && name.ends_with(".safetensors")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::safetensors::write_checkpoint;

    fn spec(name: &str, dtype: DType, shape: &[usize]) -> TensorSpec {
        TensorSpec {
            name: name.to_owned(),
            dtype,
            shape: shape.to_vec(),
        }
    }

    fn checkpoint(dir: &Path, shards: usize) -> Vec<TensorSpec> {
        let specs = vec![
            spec("model.embed_tokens.weight", DType::BF16, &[16, 8]),
            spec(
                "model.layers.0.self_attn.q_proj.weight",
                DType::BF16,
                &[8, 8],
            ),
            spec(
                "model.layers.0.mlp.experts.0.gate_proj.weight",
                DType::F4E2M1,
                &[4, 8],
            ),
        ];
        let tensors: Vec<(TensorSpec, Vec<u8>)> = specs
            .iter()
            .map(|spec| {
                let bytes = vec![0u8; spec.storage_bytes() as usize];
                (spec.clone(), bytes)
            })
            .collect();
        write_checkpoint(dir, &tensors, shards).expect("the fixture writes");
        specs
    }

    #[test]
    fn a_sharded_directory_reads_its_headers_and_ignores_the_index() {
        let dir = tempfile::tempdir().expect("tempdir");
        let specs = checkpoint(dir.path(), 2);
        // A stale index must not add or remove a tensor: the headers are the authority.
        std::fs::write(dir.path().join(INDEX_NAME), "{not json").expect("stale index");
        let index = read_local(dir.path()).expect("the headers read");
        assert_eq!(index.len(), specs.len());
        let names: Vec<&str> = index.names().collect();
        assert_eq!(
            names,
            vec![
                "model.embed_tokens.weight",
                "model.layers.0.mlp.experts.0.gate_proj.weight",
                "model.layers.0.self_attn.q_proj.weight",
            ]
        );
        let packed = index
            .get("model.layers.0.mlp.experts.0.gate_proj.weight")
            .expect("the packed expert");
        assert_eq!(
            (packed.dtype.as_str(), packed.shape.as_slice()),
            ("I8", &[4, 4][..])
        );
        assert_eq!(
            packed.logical_spec().expect("the logical spec").shape,
            vec![4, 8],
            "the packed dimension doubles in the logical view"
        );
    }

    #[test]
    fn an_unsharded_directory_reads_its_one_header() {
        let dir = tempfile::tempdir().expect("tempdir");
        let specs = checkpoint(dir.path(), 1);
        let index = read_local(dir.path()).expect("the header reads");
        assert_eq!(index.len(), specs.len());
        assert!(dir.path().join(SINGLE_FILE_NAME).is_file());
    }

    #[test]
    fn a_directory_with_no_safetensors_file_names_the_directory() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join(INDEX_NAME), "{}").expect("index");
        let error = read_local(dir.path()).expect_err("no shard");
        let IndexError::Local { subject, detail } = error else {
            panic!("expected a local error, got {error:?}");
        };
        assert_eq!(subject, dir.path().display().to_string());
        assert!(detail.contains("no *.safetensors"), "{detail}");
    }

    #[test]
    fn a_malformed_header_is_refused_by_name() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("model.safetensors");
        let mut file = Vec::new();
        file.extend_from_slice(&(MAX_HEADER_BYTES + 1).to_le_bytes());
        file.extend_from_slice(&[0u8; 8]);
        std::fs::write(&path, file).expect("fixture");
        let error = read_local(dir.path()).expect_err("the header is refused");
        let IndexError::Malformed { subject, detail } = error else {
            panic!("expected a malformed header, got {error:?}");
        };
        assert!(subject.contains("model.safetensors"), "{subject}");
        assert!(detail.contains("declared header length"), "{detail}");
    }

    #[test]
    fn a_header_byte_range_that_does_not_fit_its_dtype_is_refused() {
        let dir = tempfile::tempdir().expect("tempdir");
        let header = br#"{"x":{"dtype":"BF16","shape":[4,8],"data_offsets":[0,1]}}"#;
        let mut file = Vec::new();
        file.extend_from_slice(&(header.len() as u64).to_le_bytes());
        file.extend_from_slice(header);
        file.push(0);
        std::fs::write(dir.path().join(SINGLE_FILE_NAME), file).expect("fixture");
        let error = read_local(dir.path()).expect_err("one byte cannot store 32 BF16 values");
        let IndexError::Malformed { detail, .. } = error else {
            panic!("expected a malformed header, got {error:?}");
        };
        assert!(detail.contains("stores 1 bytes"), "{detail}");
    }

    #[test]
    fn one_name_in_two_shards_is_refused() {
        let dir = tempfile::tempdir().expect("tempdir");
        let spec = spec("model.embed_tokens.weight", DType::BF16, &[2, 2]);
        let bytes = vec![0u8; spec.storage_bytes() as usize];
        write_checkpoint(
            dir.path(),
            &[(spec.clone(), bytes.clone()), (spec.clone(), bytes)],
            2,
        )
        .expect("two shards, each naming the same tensor");
        let error = read_local(dir.path()).expect_err("the duplicate is refused");
        let IndexError::Malformed { detail, .. } = error else {
            panic!("expected a malformed header, got {error:?}");
        };
        assert!(detail.contains("described by two shards"), "{detail}");
    }

    #[test]
    fn a_header_dtype_this_reader_does_not_know_is_named() {
        let error = logical_spec("x", "BF16", &[2, 2]).expect("BF16 maps");
        assert_eq!(error.dtype, DType::BF16);
        let error = logical_spec("x", "I9", &[2, 2]).expect_err("I9 is not a dtype");
        assert_eq!(
            error,
            IndexError::Dtype {
                name: "x".to_owned(),
                dtype: "I9".to_owned()
            }
        );
    }

    #[test]
    fn a_revision_and_a_shard_name_stay_one_path_component() {
        for revision in ["main", "a..b", "", ".", "..", "a/b", "a\\b"] {
            assert_eq!(is_revision(revision), revision == "main", "{revision:?}");
        }
        for name in [
            "model.safetensors",
            "../model.safetensors",
            "a/model.safetensors",
            "model.bin",
            "",
        ] {
            assert_eq!(is_shard_name(name), name == "model.safetensors", "{name:?}");
        }
    }

    #[test]
    fn a_content_range_total_is_read_after_the_slash() {
        assert_eq!(total_of_content_range("bytes 0-7/1234"), Some(1234));
        assert_eq!(total_of_content_range("bytes 0-7/*"), None);
        assert_eq!(total_of_content_range("not a range"), None);
    }

    #[test]
    fn the_hub_reader_is_optional_and_needs_an_env_var() {
        let Ok(repo) = std::env::var("SPARKFLASH_HUB_INDEX_TEST") else {
            return;
        };
        let index = read_hub(&repo, "main").expect("the public repository's headers read");
        assert!(!index.is_empty());
    }
}
