//! Tokenizer compressor: 129280 token ids -> 99092 compressed ids. OWNER: engram-hash.
//!
//! Exact port of `build_compressed_token_map` from `reference/inference/engram.py`. Tokens
//! that normalise alike must hash alike (" The", "the" and "THE" share one bucket), so every
//! raw tokenizer id is mapped onto a dense compressed id space:
//!
//! 1. `text = tokenizer.decode([id], skip_special_tokens=false)` (the raw fast tokenizer, with
//!    no clean-up - see the reference's comment);
//! 2. if `text` contains U+FFFD (a partial UTF-8 byte token) the key is the raw
//!    `id_to_token(id)` string, because there is nothing to normalise;
//! 3. otherwise the key is the result of the normalizer chain below, falling back to the raw
//!    text when normalisation produces the empty string;
//! 4. keys are interned in ascending token-id order of first appearance.
//!
//! The normalizer chain, in order, is
//!
//! | # | normalizer |
//! |---:|---|
//! | 1 | NFKC |
//! | 2 | NFD |
//! | 3 | StripAccents |
//! | 4 | Lowercase |
//! | 5 | replace the regex `[ \t\r\n]+` with a single space |
//! | 6 | replace the regex `^ $` (a lone space) with the private-use sentinel U+E000 |
//! | 7 | Strip |
//! | 8 | replace U+E000 with a plain space |
//!
//! Step 6 is load bearing: without it a token that is exactly one space is stripped to the
//! empty string and merges with unrelated tokens, changing their compressed id.
//!
//! Two failure modes are silent and catastrophic, so both are rejected before a compressor
//! exists. The compressed size is not just a bound: every hash multiplier is derived from it
//! (`crate::primes::multiplier_bound` divides by `COMPRESSED_VOCAB_SIZE`), so the
//! constructors refuse to build a map that does not use exactly `COMPRESSED_VOCAB_SIZE`
//! classes. And because a *different* 129,280-token tokenizer can still produce 99,092
//! classes, the boot path `TokenizerCompressor::from_reference_tokenizer_json` also verifies
//! the SHA-256 of the tokenizer file against `REFERENCE_TOKENIZER_SHA256_HEX`. Either
//! mismatch would rehash the whole 189 GiB table.

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};
use tokenizers::normalizers::unicode::{NFD, NFKC};
use tokenizers::normalizers::utils::{Lowercase, Sequence};
use tokenizers::normalizers::{
    NormalizerWrapper, Replace, replace::ReplacePattern, strip::Strip, strip::StripAccents,
};
use tokenizers::{NormalizedString, Normalizer, Tokenizer};

use crate::layout::{COMPRESSED_VOCAB_SIZE, ENGRAM_PAD_TOKEN_ID};

/// Number of raw tokenizer ids the map covers (the tokenizer's vocabulary including added
/// tokens). The published DeepSeek-V4.1 tokenizer has 128,000 base tokens plus 1,280 added
/// tokens; the compressor rejects any other size because the checkpoint's hash table is
/// addressed by this exact id space.
pub const RAW_VOCAB_SIZE: usize = 129_280;

/// Private-use sentinel used by the normalizer chain (U+E000).
pub const WHITESPACE_SENTINEL: char = '\u{e000}';

/// The regex the reference collapses with (`normalizers.Replace(Regex(r"[ \t\r\n]+"), " ")`).
pub const WHITESPACE_RUN_PATTERN: &str = r"[ \t\r\n]+";

/// The regex the reference sentinels a lone space with (`normalizers.Replace(Regex(r"^ $"), ...)`).
pub const LONE_SPACE_PATTERN: &str = r"^ $";

/// SHA-256 of the published DeepSeek-V4.1-Flash `reference/tokenizer.json`, in lower-case
/// hex.
///
/// Verify against this before building a compressor on the boot path: two tokenizers can agree
/// on the vocabulary size and even on the class count and still disagree on individual ids,
/// and every disagreement moves rows in a 189 GiB table.
pub const REFERENCE_TOKENIZER_SHA256_HEX: &str =
    "c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b";

/// The same digest as raw bytes, for `TokenizerCompressor::from_tokenizer_json_verified`.
pub const REFERENCE_TOKENIZER_SHA256: [u8; 32] = [
    0xc9, 0x0d, 0xfa, 0x01, 0x24, 0x9d, 0xb1, 0xbe, 0x42, 0x45, 0x78, 0x0a, 0x05, 0x2e, 0xde, 0x75,
    0x2e, 0x13, 0x61, 0xc6, 0x12, 0xac, 0x6d, 0x08, 0xe2, 0xbd, 0xad, 0xa7, 0xd5, 0x99, 0x47, 0x6b,
];

/// Errors from the hashing half of the crate (tokenizer compression and n-gram hashing).
#[derive(Debug, thiserror::Error)]
pub enum EngramError {
    /// The tokenizer file could not be read or parsed.
    #[error("failed to load tokenizer {path}: {source}")]
    TokenizerLoad {
        /// Path that was passed to `Tokenizer::from_file`.
        path: PathBuf,
        /// Underlying tokenizers-crate error.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    /// The tokenizer's vocabulary is not the one the tables were built for.
    #[error("tokenizer vocabulary size mismatch: got {got}, expected {expected}")]
    VocabSize {
        /// Vocabulary size reported by the tokenizer (including added tokens).
        got: usize,
        /// Required size, `RAW_VOCAB_SIZE`.
        expected: usize,
    },
    /// A token id could not be decoded to text.
    #[error("failed to decode token id {token_id}: {source}")]
    Decode {
        /// Raw tokenizer id.
        token_id: u32,
        /// Underlying tokenizers-crate error.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    /// A token id present in the vocabulary had no raw token string.
    #[error("tokenizer has no token string for id {token_id}")]
    MissingToken {
        /// Raw tokenizer id.
        token_id: u32,
    },
    /// Normalising a token's text failed.
    #[error("failed to normalise a token: {source}")]
    Normalize {
        /// Underlying tokenizers-crate error.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    /// The compressed map does not use exactly `COMPRESSED_VOCAB_SIZE` classes.
    #[error(
        "compressed vocabulary size mismatch: got {got}, expected {expected}; every hash \
         multiplier is derived from {expected}, refusing to build"
    )]
    CompressedVocabSize {
        /// Number of classes the freshly built map uses.
        got: u32,
        /// Required number of classes, `COMPRESSED_VOCAB_SIZE`.
        expected: u32,
    },
    /// A lookup entry is not a valid compressed id.
    #[error(
        "lookup entry {token_id} is {value}, outside the compressed vocabulary of \
         {compressed_vocab_size}"
    )]
    LookupEntry {
        /// Raw tokenizer id whose entry is invalid.
        token_id: u32,
        /// The offending value.
        value: u32,
        /// Number of compressed classes the entry had to be below.
        compressed_vocab_size: u32,
    },
    /// A token mask does not have one entry per token id.
    #[error("token mask has {given} entries for {expected} token ids")]
    TokenMaskLength {
        /// Number of entries supplied.
        given: usize,
        /// Number of token ids pushed.
        expected: usize,
    },
    /// A push does not fit in the position cache.
    #[error("pushing {len} ids at position {start_pos} exceeds the {max_seq_len}-position cache")]
    CacheCapacity {
        /// Position the push would have started at.
        start_pos: usize,
        /// Number of ids pushed.
        len: usize,
        /// Cache capacity.
        max_seq_len: usize,
    },
    /// A compressed id handed to the hasher is not below the compressed vocabulary size.
    #[error(
        "compressed id {id} at batch position {position} is not below {compressed_vocab_size};          pass raw tokenizer ids to push() instead"
    )]
    CompressedIdOutOfRange {
        /// The offending id.
        id: u32,
        /// Offset of the offending id inside the pushed batch.
        position: usize,
        /// Number of compressed classes the id had to be below.
        compressed_vocab_size: u32,
    },
    /// A file could not be read (used by the digest helpers).
    #[error("failed to read {path}: {source}")]
    Io {
        /// The file that could not be read.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// The tokenizer file is not the one the hash table was built for.
    #[error(
        "tokenizer {path} has sha256 {actual}, expected {expected}; refusing to build a map          that would rehash the table"
    )]
    TokenizerDigestMismatch {
        /// The tokenizer file that was checked.
        path: PathBuf,
        /// Expected digest, lower-case hex.
        expected: String,
        /// Digest actually found, lower-case hex.
        actual: String,
    },
}

/// Surjective map from the 129,280 raw tokenizer ids onto 99,092 compressed hash ids.
///
/// Built once at boot by [`TokenizerCompressor::from_reference_tokenizer_json`], which
/// verifies the tokenizer digest first. The map is a plain
/// lookup table, so [`compress`](TokenizerCompressor::compress) is one bounds check and one
/// load, and is safe to call on the hot path.
#[derive(Debug, Clone)]
pub struct TokenizerCompressor {
    lookup: Box<[u32]>,
    compressed_vocab_size: u32,
    pad_id: u32,
}

impl TokenizerCompressor {
    /// Builds the compressor from a HuggingFace `tokenizer.json` **without checking its
    /// digest**.
    ///
    /// Prefer [`from_reference_tokenizer_json`](Self::from_reference_tokenizer_json) on the
    /// boot path. This constructor still rejects a wrong vocabulary size and a wrong class
    /// count, but it cannot tell the published tokenizer apart from another 129,280-token
    /// tokenizer that happens to land on 99,092 classes; such a tokenizer would be accepted
    /// silently and every hash id in the 189 GiB table would move. Use this entry point only
    /// when the caller has verified the file itself (for example against a checkpoint
    /// manifest).
    ///
    /// # Errors
    /// Fails when the file cannot be loaded, when its vocabulary is not exactly
    /// [`RAW_VOCAB_SIZE`] ids, when a token cannot be decoded or normalised, or when the
    /// resulting map does not use exactly `COMPRESSED_VOCAB_SIZE` classes.
    pub fn from_tokenizer_json(path: impl AsRef<Path>) -> Result<Self, EngramError> {
        let path = path.as_ref();
        let tokenizer =
            Tokenizer::from_file(path).map_err(|source| EngramError::TokenizerLoad {
                path: path.to_path_buf(),
                source,
            })?;
        Self::from_tokenizer(&tokenizer)
    }

    /// Builds the compressor from `path` after verifying that the file's SHA-256 is exactly
    /// `expected_sha256`.
    ///
    /// The digest is checked before the file is parsed, so a wrong tokenizer never reaches the
    /// lookup-table build.
    ///
    /// # Errors
    /// Fails with [`EngramError::TokenizerDigestMismatch`] when the digests differ (the error
    /// carries both digests), with [`EngramError::Io`] when the file cannot be read, and
    /// otherwise as [`from_tokenizer_json`](Self::from_tokenizer_json).
    pub fn from_tokenizer_json_verified(
        path: impl AsRef<Path>,
        expected_sha256: &[u8; 32],
    ) -> Result<Self, EngramError> {
        let path = path.as_ref();
        let actual = sha256_file(path)?;
        if actual != *expected_sha256 {
            return Err(EngramError::TokenizerDigestMismatch {
                path: path.to_path_buf(),
                expected: to_hex(expected_sha256),
                actual: to_hex(&actual),
            });
        }
        Self::from_tokenizer_json(path)
    }

    /// Builds the compressor from the published DeepSeek-V4.1-Flash tokenizer, verifying it
    /// against `REFERENCE_TOKENIZER_SHA256_HEX`. This is the boot-path constructor.
    ///
    /// # Errors
    /// As [`from_tokenizer_json_verified`](Self::from_tokenizer_json_verified).
    pub fn from_reference_tokenizer_json(path: impl AsRef<Path>) -> Result<Self, EngramError> {
        Self::from_tokenizer_json_verified(path, &REFERENCE_TOKENIZER_SHA256)
    }

    /// Builds the compressor from an already-loaded tokenizer. See
    /// [`from_tokenizer_json`](Self::from_tokenizer_json).
    ///
    /// # Errors
    /// As [`from_tokenizer_json`](Self::from_tokenizer_json).
    pub fn from_tokenizer(tokenizer: &Tokenizer) -> Result<Self, EngramError> {
        let vocab_size = tokenizer.get_vocab_size(true);
        if vocab_size != RAW_VOCAB_SIZE {
            return Err(EngramError::VocabSize {
                got: vocab_size,
                expected: RAW_VOCAB_SIZE,
            });
        }

        let normalizer = build_normalizer()?;
        let mut keys: HashMap<String, u32> = HashMap::new();
        let mut lookup = Vec::with_capacity(vocab_size);
        for token_id in 0..vocab_size as u32 {
            let text = tokenizer
                .decode(&[token_id], false)
                .map_err(|source| EngramError::Decode { token_id, source })?;
            let key = if text.contains('\u{fffd}') {
                // a partial UTF-8 byte token: nothing to normalise, key it by its raw form
                tokenizer
                    .id_to_token(token_id)
                    .ok_or(EngramError::MissingToken { token_id })?
            } else {
                let normalized = normalize_str(&normalizer, &text)?;
                if normalized.is_empty() {
                    text
                } else {
                    normalized
                }
            };
            let next_id = keys.len() as u32;
            let new_id = *keys.entry(key).or_insert(next_id);
            lookup.push(new_id);
        }

        let compressed_vocab_size = keys.len() as u32;
        if compressed_vocab_size != COMPRESSED_VOCAB_SIZE {
            return Err(EngramError::CompressedVocabSize {
                got: compressed_vocab_size,
                expected: COMPRESSED_VOCAB_SIZE,
            });
        }
        Self::from_lookup(lookup, compressed_vocab_size)
    }

    /// Builds a compressor from a ready-made lookup table, e.g. one loaded from a golden
    /// fixture.
    ///
    /// # Errors
    /// Fails when the table is shorter than `ENGRAM_PAD_TOKEN_ID + 1` entries, when any
    /// entry is not below `compressed_vocab_size`, or when `compressed_vocab_size` is not
    /// exactly `COMPRESSED_VOCAB_SIZE`.
    pub fn from_lookup(lookup: Vec<u32>, compressed_vocab_size: u32) -> Result<Self, EngramError> {
        let pad_id =
            *lookup
                .get(ENGRAM_PAD_TOKEN_ID as usize)
                .ok_or(EngramError::MissingToken {
                    token_id: ENGRAM_PAD_TOKEN_ID,
                })?;
        if compressed_vocab_size != COMPRESSED_VOCAB_SIZE {
            return Err(EngramError::CompressedVocabSize {
                got: compressed_vocab_size,
                expected: COMPRESSED_VOCAB_SIZE,
            });
        }
        if let Some((token_id, value)) = lookup
            .iter()
            .copied()
            .enumerate()
            .find(|&(_, id)| id >= compressed_vocab_size)
        {
            return Err(EngramError::LookupEntry {
                token_id: token_id as u32,
                value,
                compressed_vocab_size,
            });
        }
        Ok(Self {
            lookup: lookup.into_boxed_slice(),
            compressed_vocab_size,
            pad_id,
        })
    }

    /// Number of entries in the lookup table, i.e. the raw tokenizer vocabulary size.
    #[must_use]
    pub fn len(&self) -> usize {
        self.lookup.len()
    }

    /// Always false: a compressor always covers a non-empty vocabulary. Present because
    /// `len` exists.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.lookup.is_empty()
    }

    /// Compressed id of `raw_token_id`. Ids past the end of the vocabulary map to the pad
    /// id.
    ///
    /// This is a deliberate hardening **deviation** from the reference, which indexes
    /// `self.token_map[input_ids]` directly and raises `IndexError` for an id outside the
    /// vocabulary. Saturating to the pad id keeps one malformed request from aborting a whole
    /// batch, and the pad id is already the reference's substitute for blocked image-span
    /// positions, so the result is the well-defined "no memory for this token" value rather
    /// than a wrong one. The id space stays closed either way: no value returned here can
    /// reach `COMPRESSED_VOCAB_SIZE`.
    #[inline]
    #[must_use]
    pub fn compress(&self, raw_token_id: u32) -> u32 {
        match self.lookup.get(raw_token_id as usize) {
            Some(&id) => id,
            None => self.pad_id,
        }
    }

    /// The whole lookup table, one entry per raw tokenizer id.
    #[must_use]
    pub fn lookup(&self) -> &[u32] {
        &self.lookup
    }

    /// Compressed id of `ENGRAM_PAD_TOKEN_ID` (2), the id substituted for blocked positions.
    #[must_use]
    pub fn pad_id(&self) -> u32 {
        self.pad_id
    }

    /// Number of distinct compressed classes in the map; always `COMPRESSED_VOCAB_SIZE`.
    #[must_use]
    pub fn compressed_vocab_size(&self) -> u32 {
        self.compressed_vocab_size
    }

    /// Raw vocabulary size, i.e. `lookup().len()`.
    #[must_use]
    pub fn raw_vocab_size(&self) -> usize {
        self.lookup.len()
    }
}

/// SHA-256 of a file, read in 1 MiB chunks so memory stays bounded for a large tokenizer.
///
/// # Errors
/// Fails with [`EngramError::Io`] when the file cannot be opened or read.
pub fn sha256_file(path: impl AsRef<Path>) -> Result<[u8; 32], EngramError> {
    let path = path.as_ref();
    let io_error = |source: std::io::Error| EngramError::Io {
        path: path.to_path_buf(),
        source,
    };
    let mut file = File::open(path).map_err(io_error)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let read = file.read(&mut buffer).map_err(io_error)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    let mut digest = [0u8; 32];
    digest.copy_from_slice(&hasher.finalize());
    Ok(digest)
}

/// Lower-case hex of a byte string, used for digest messages.
fn to_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(char::from(HEX[(byte >> 4) as usize]));
        out.push(char::from(HEX[(byte & 0x0f) as usize]));
    }
    out
}

/// Builds the reference's eight-step normalizer chain (see the module docs for the order).
fn build_normalizer() -> Result<NormalizerWrapper, EngramError> {
    let normalize_err =
        |source: Box<dyn std::error::Error + Send + Sync>| EngramError::Normalize { source };
    let collapse = Replace::new(
        ReplacePattern::Regex(WHITESPACE_RUN_PATTERN.to_owned()),
        " ",
    )
    .map_err(normalize_err)?;
    let sentinel_in = Replace::new(
        ReplacePattern::Regex(LONE_SPACE_PATTERN.to_owned()),
        WHITESPACE_SENTINEL.to_string(),
    )
    .map_err(normalize_err)?;
    let sentinel_out = Replace::new(ReplacePattern::String(WHITESPACE_SENTINEL.to_string()), " ")
        .map_err(normalize_err)?;
    let chain = Sequence::new(vec![
        NormalizerWrapper::NFKC(NFKC),
        NormalizerWrapper::NFD(NFD),
        NormalizerWrapper::StripAccents(StripAccents),
        NormalizerWrapper::Lowercase(Lowercase),
        NormalizerWrapper::Replace(collapse),
        NormalizerWrapper::Replace(sentinel_in),
        NormalizerWrapper::StripNormalizer(Strip::new(true, true)),
        NormalizerWrapper::Replace(sentinel_out),
    ]);
    Ok(NormalizerWrapper::Sequence(chain))
}

/// Applies the chain to one string, the equivalent of the binding's `normalize_str`.
fn normalize_str(normalizer: &NormalizerWrapper, text: &str) -> Result<String, EngramError> {
    let mut normalized = NormalizedString::from(text);
    normalizer
        .normalize(&mut normalized)
        .map_err(|source| EngramError::Normalize { source })?;
    Ok(normalized.get().to_owned())
}
