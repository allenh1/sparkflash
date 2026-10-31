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

//! What bytes each vocabulary id contributes.
//!
//! A constrained decoder works in bytes (see [super::constrained]) but a sampler works in token
//! ids, so the two need one table: id -> the exact bytes that token appends. The checkpoint
//! tokenizer is byte-level BPE (tokenizer.json names ByteLevel as its decoder), so a vocabulary
//! entry is not the token text: it is the token bytes mapped through the byte alphabet, and
//! id_to_token hands back the mapped spelling. A token spelled with a leading U+0120 is the text
//! with a leading space, not the six characters it looks like. Feeding the mapped spelling to the
//! automaton would test the wrong bytes and accept documents that are not valid JSON.
//!
//! The table is built once per server from the codec and shared by every request through an Arc:
//! it is a few megabytes and does not depend on the request.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use sparkflash_model::PromptCodec;

/// The byte alphabet a ByteLevel tokenizer maps bytes through: printable ASCII, the Latin-1 high
/// range, then every remaining byte value assigned the next free code point (the table every
/// byte-level BPE checkpoint uses).
fn byte_alphabet() -> &'static HashMap<char, u8> {
    static TABLE: OnceLock<HashMap<char, u8>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut bytes: Vec<u32> = (33..=126u32).chain(161..=172).chain(174..=255).collect();
        let mut chars: Vec<u32> = bytes.clone();
        let mut next = 0u32;
        for byte in 0..256u32 {
            if !bytes.contains(&byte) {
                bytes.push(byte);
                chars.push(256 + next);
                next += 1;
            }
        }
        bytes
            .into_iter()
            .zip(chars)
            .filter_map(|(byte, code)| Some((char::from_u32(code)?, byte as u8)))
            .collect()
    })
}

/// Every vocabulary id bytes, plus a first-byte index for the constrained decoder.
///
/// The index is what keeps a per-step allowed set affordable: a JSON syntax state allows a handful
/// of bytes (a quote, a brace, a digit), so only the ids that start with one of them are worth
/// testing against the automaton.
#[derive(Debug)]
pub struct TokenBytes {
    entries: Vec<Option<Box<[u8]>>>,
    plain_string: Vec<bool>,
    by_first: Vec<Vec<u32>>,
}

impl TokenBytes {
    /// Builds the table from the checkpoint tokenizer, one entry per id below vocab.
    ///
    /// Special/control marker tokens (`<｜...｜>`, `<think>`, `</think>`) are omitted so they can
    /// never be admitted as content inside a JSON string; the end-of-sequence id is gated
    /// separately by [super::constrained::TokenConstraint].
    #[must_use]
    pub fn from_codec(codec: &dyn PromptCodec, vocab: usize) -> Self {
        let pairs = (0..vocab).filter_map(|id| {
            let id = u32::try_from(id).ok()?;
            let spelling = codec.token_str(id)?;
            if (spelling.starts_with("<｜") && spelling.ends_with("｜>"))
                || spelling == "<think>"
                || spelling == "</think>"
            {
                return None;
            }
            Some((id, spelling))
        });
        Self::from_pairs(pairs)
    }

    /// Builds the table from (id, vocabulary spelling) pairs.
    ///
    /// A spelling is decoded through the byte alphabet; a character outside it is taken as its own
    /// UTF-8 bytes, which is what an added token (a literal marker spelling) is.
    #[must_use]
    pub fn from_pairs(pairs: impl IntoIterator<Item = (u32, String)>) -> Self {
        let alphabet = byte_alphabet();
        let mut entries: Vec<Option<Box<[u8]>>> = Vec::new();
        for (id, spelling) in pairs {
            let id = id as usize;
            if entries.len() <= id {
                entries.resize(id + 1, None);
            }
            let mut bytes = Vec::with_capacity(spelling.len());
            for ch in spelling.chars() {
                match alphabet.get(&ch) {
                    Some(&byte) => bytes.push(byte),
                    None => {
                        let mut buf = [0u8; 4];
                        bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                    }
                }
            }
            entries[id] = Some(bytes.into_boxed_slice());
        }
        let plain_string = entries
            .iter()
            .map(|entry| {
                entry.as_deref().is_some_and(|bytes| {
                    !bytes.is_empty() && bytes.iter().all(|&b| b >= 0x20 && b != b'"' && b != 0x5c)
                })
            })
            .collect();
        let mut by_first = vec![Vec::new(); 256];
        for (id, entry) in entries.iter().enumerate() {
            if let Some(&first) = entry.as_deref().and_then(|bytes| bytes.first()) {
                by_first[first as usize].push(u32::try_from(id).unwrap_or(u32::MAX));
            }
        }
        Self {
            entries,
            plain_string,
            by_first,
        }
    }

    /// The ids in the vocabulary.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the table is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The bytes this id appends, or None for an id outside the table.
    #[must_use]
    pub fn bytes(&self, id: u32) -> Option<&[u8]> {
        self.entries.get(id as usize)?.as_deref()
    }

    /// Whether this id appends one or more printable non-quote, non-backslash bytes: inside an
    /// unescaped JSON string literal, such a token stays inside the string without mutating the
    /// container stack.
    #[must_use]
    pub fn is_plain_string(&self, id: u32) -> bool {
        self.plain_string.get(id as usize).copied().unwrap_or(false)
    }

    /// The ids whose first byte is byte, in ascending order.
    #[must_use]
    pub fn starting_with(&self, byte: u8) -> &[u32] {
        &self.by_first[byte as usize]
    }

    /// The table behind an Arc, building it once.
    #[must_use]
    pub fn shared(pairs: impl IntoIterator<Item = (u32, String)>) -> Arc<Self> {
        Arc::new(Self::from_pairs(pairs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(pairs: &[(u32, &str)]) -> TokenBytes {
        TokenBytes::from_pairs(pairs.iter().map(|&(id, text)| (id, text.to_owned())))
    }

    #[test]
    fn byte_level_spellings_decode_to_their_bytes() {
        // U+0120 is byte 0x20 (space) and U+010A is 0x0a (newline) in the byte alphabet; a
        // byte-level vocabulary stores them that way, so the automaton must see the byte.
        let space_marker = char::from_u32(0x120).unwrap();
        let newline_marker = char::from_u32(0x10a).unwrap();
        let vocab = TokenBytes::from_pairs([
            (0u32, format!("{space_marker}lead")),
            (1, newline_marker.to_string()),
            (2, "abc".to_owned()),
        ]);
        assert_eq!(vocab.bytes(0), Some(b" lead" as &[u8]));
        assert_eq!(vocab.bytes(1), Some(&[0x0au8][..]));
        assert_eq!(vocab.bytes(2), Some(b"abc" as &[u8]));
        assert_eq!(vocab.bytes(9), None);
    }

    #[test]
    fn the_first_byte_index_covers_every_entry() {
        let space_marker = char::from_u32(0x120).unwrap();
        let vocab = TokenBytes::from_pairs([
            (0u32, format!("{space_marker}a")),
            (1, format!("{space_marker}b")),
            (2, "z".to_owned()),
        ]);
        assert_eq!(vocab.starting_with(b' '), &[0, 1]);
        assert_eq!(vocab.starting_with(b'z'), &[2]);
        assert!(vocab.starting_with(b'q').is_empty());
    }

    #[test]
    fn an_added_token_keeps_its_literal_bytes() {
        let vocab = table(&[(7, "<|endoftext|>")]);
        assert_eq!(vocab.bytes(7), Some(b"<|endoftext|>" as &[u8]));
    }
}
