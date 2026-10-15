//! Decode-side N-gram and repeated-line loop-abort detector.
//!
//! Long-context decoders occasionally fall into a degenerate loop, emitting the same
//! N-gram or the same line forever. Both signatures are cheap to recognise from the token
//! stream alone, so the detector runs inline in the decode loop: one hash, one ring
//! comparison and a bounded line buffer per token.
//!
//! Detection is exact and structural:
//!
//! * An N-gram g of length n repeated R times back to back makes the last n*R tokens
//!   periodic with period n. That is equivalent to token\[i\] == token\[i - n\] holding for
//!   the last n*(R-1) tokens, which is a single counter plus a ring of n+1 token hashes.
//! * R identical consecutive newline-terminated lines, each at least min_line_bytes long,
//!   are tracked with two reusable string buffers.

use std::collections::VecDeque;

/// Maximum number of bytes buffered for the current line before it is disqualified from
/// repeated-line detection. Bounds detector memory at 2 * LINE_BUFFER_CAP bytes.
pub const LINE_BUFFER_CAP: usize = 8192;

/// Tunables for LoopAbortDetector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoopAbortConfig {
    /// Number of consecutive tokens that form the repeating unit.
    pub ngram: usize,
    /// Fire once the same n-gram has repeated this many times back to back.
    pub max_repeats: usize,
    /// Fire once the same line has repeated this many times back to back.
    pub max_line_repeats: usize,
    /// Lines whose payload (excluding the terminating newline) is shorter than this are
    /// never considered for repeated-line detection.
    pub min_line_bytes: usize,
}

impl Default for LoopAbortConfig {
    fn default() -> Self {
        Self {
            ngram: 8,
            max_repeats: 8,
            max_line_repeats: 4,
            min_line_bytes: 16,
        }
    }
}

/// The signature that tripped the detector.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoopAbort {
    /// The same n-gram block repeated back to back.
    RepeatedNgram {
        /// Configured N-gram length.
        ngram: usize,
        /// Number of consecutive identical blocks observed.
        repeats: usize,
    },
    /// The same non-trivial line repeated back to back.
    RepeatedLine {
        /// The repeated line, including its trailing newline.
        line: String,
        /// Number of consecutive identical lines observed.
        repeats: usize,
    },
}

impl LoopAbort {
    /// Short machine-readable reason, suitable for logs and error codes.
    #[must_use]
    pub fn reason(&self) -> &'static str {
        match self {
            Self::RepeatedNgram { .. } => "repeated_ngram",
            Self::RepeatedLine { .. } => "repeated_line",
        }
    }
}

/// Stateful loop detector fed one decoded token piece at a time.
///
/// The detector latches: once push returns a reason, every later call returns None (the
/// engine is expected to abort the request).
#[derive(Debug)]
pub struct LoopAbortDetector {
    config: LoopAbortConfig,
    ring: VecDeque<u64>,
    match_run: usize,
    line_buf: String,
    last_line: String,
    line_run: usize,
    line_overflow: bool,
    fired: Option<LoopAbort>,
}

impl LoopAbortDetector {
    /// Creates a detector. A zero n-gram (or max_repeats < 2) disables N-gram detection.
    #[must_use]
    pub fn new(config: LoopAbortConfig) -> Self {
        let capacity = config.ngram.saturating_add(1).max(1);
        Self {
            config,
            ring: VecDeque::with_capacity(capacity),
            match_run: 0,
            line_buf: String::new(),
            last_line: String::new(),
            line_run: 0,
            line_overflow: false,
            fired: None,
        }
    }

    /// The configuration in use.
    #[must_use]
    pub fn config(&self) -> LoopAbortConfig {
        self.config
    }

    /// Whether the detector has already fired.
    #[must_use]
    pub fn aborted(&self) -> bool {
        self.fired.is_some()
    }

    /// The reason the detector fired, if it has.
    #[must_use]
    pub fn abort_reason(&self) -> Option<&LoopAbort> {
        self.fired.as_ref()
    }

    /// Feeds one decoded token piece.
    ///
    /// Returns Some(reason) on the token that completes a pathological repeat, and None
    /// for every other token (including all tokens after the first abort).
    pub fn push(&mut self, piece: &str) -> Option<LoopAbort> {
        if self.fired.is_some() {
            return None;
        }
        if let Some(abort) = self.push_ngram(piece) {
            self.fired = Some(abort.clone());
            return Some(abort);
        }
        if let Some(abort) = self.push_lines(piece) {
            self.fired = Some(abort.clone());
            return Some(abort);
        }
        None
    }

    fn push_ngram(&mut self, piece: &str) -> Option<LoopAbort> {
        let ngram = self.config.ngram;
        if ngram == 0 || self.config.max_repeats < 2 {
            return None;
        }
        if self.ring.len() == ngram + 1 {
            self.ring.pop_front();
        }
        self.ring.push_back(hash_piece(piece));
        if self.ring.len() == ngram + 1 {
            if self.ring[ngram] == self.ring[0] {
                self.match_run += 1;
            } else {
                self.match_run = 0;
            }
        }
        if self.match_run >= ngram * (self.config.max_repeats - 1) {
            return Some(LoopAbort::RepeatedNgram {
                ngram,
                repeats: self.config.max_repeats,
            });
        }
        None
    }

    fn push_lines(&mut self, piece: &str) -> Option<LoopAbort> {
        let mut rest = piece;
        while !rest.is_empty() {
            let (segment, tail) = match rest.find('\n') {
                Some(index) => (&rest[..=index], &rest[index + 1..]),
                None => (rest, ""),
            };
            self.append_line_segment(segment);
            if let Some(abort) = self.finish_line() {
                return Some(abort);
            }
            rest = tail;
        }
        None
    }

    fn append_line_segment(&mut self, segment: &str) {
        if self.line_overflow {
            return;
        }
        if self.line_buf.len().saturating_add(segment.len()) > LINE_BUFFER_CAP {
            self.line_overflow = true;
            self.line_buf.clear();
            return;
        }
        self.line_buf.push_str(segment);
    }

    fn finish_line(&mut self) -> Option<LoopAbort> {
        // min_line_bytes counts the line payload, excluding the terminating newline.
        let payload_len = if self.line_buf.ends_with('\n') {
            self.line_buf.len() - 1
        } else {
            self.line_buf.len()
        };
        let eligible = !self.line_overflow && payload_len >= self.config.min_line_bytes;
        if eligible && self.line_buf == self.last_line {
            self.line_run += 1;
            if self.line_run >= self.config.max_line_repeats {
                let abort = LoopAbort::RepeatedLine {
                    line: self.line_buf.clone(),
                    repeats: self.line_run,
                };
                self.line_buf.clear();
                self.line_overflow = false;
                return Some(abort);
            }
        } else if eligible {
            std::mem::swap(&mut self.line_buf, &mut self.last_line);
            self.line_run = 1;
        } else {
            self.last_line.clear();
            self.line_run = 0;
        }
        self.line_buf.clear();
        self.line_overflow = false;
        None
    }
}

fn hash_piece(piece: &str) -> u64 {
    // FNV-1a: branch-free, allocation-free and good enough for token equality screening.
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in piece.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    fn detector(config: LoopAbortConfig) -> LoopAbortDetector {
        LoopAbortDetector::new(config)
    }

    #[test]
    fn default_config_matches_the_spec() {
        let config = LoopAbortConfig::default();
        assert_eq!(config.ngram, 8);
        assert_eq!(config.max_repeats, 8);
        assert_eq!(config.max_line_repeats, 4);
        assert_eq!(config.min_line_bytes, 16);
    }

    #[test]
    fn repeated_ngram_fires_exactly_at_the_threshold() {
        let config = LoopAbortConfig {
            ngram: 4,
            max_repeats: 3,
            ..LoopAbortConfig::default()
        };
        let block = ["alpha", "beta", "gamma", "delta"];
        let mut at_threshold = detector(config);
        let mut fired = None;
        for _ in 0..3 {
            for piece in block {
                if let Some(reason) = at_threshold.push(piece) {
                    fired = Some(reason);
                }
            }
        }
        assert_eq!(
            fired,
            Some(LoopAbort::RepeatedNgram {
                ngram: 4,
                repeats: 3
            })
        );
        assert!(at_threshold.aborted());

        let mut just_below = detector(config);
        for _ in 0..2 {
            for piece in block {
                assert_eq!(just_below.push(piece), None);
            }
        }
        // Three tokens into the third repetition: one token below the threshold.
        for piece in &block[..3] {
            assert_eq!(just_below.push(piece), None);
        }
        assert!(!just_below.aborted());
    }

    #[test]
    fn a_single_interleaved_token_resets_the_ngram_run() {
        let config = LoopAbortConfig {
            ngram: 4,
            max_repeats: 3,
            ..LoopAbortConfig::default()
        };
        let mut det = detector(config);
        for _ in 0..5 {
            for piece in ["a", "b", "c", "d"] {
                assert_eq!(det.push(piece), None);
            }
            assert_eq!(det.push("noise"), None);
        }
        assert!(!det.aborted());
    }

    #[test]
    fn interleaved_whitespace_still_counts_as_part_of_the_block() {
        let config = LoopAbortConfig {
            ngram: 2,
            max_repeats: 4,
            ..LoopAbortConfig::default()
        };
        let mut det = detector(config);
        let mut fired = false;
        for _ in 0..4 {
            for piece in ["hello", " "] {
                if det.push(piece).is_some() {
                    fired = true;
                }
            }
        }
        assert!(fired);
    }

    #[test]
    fn repeated_single_token_fires_on_the_last_repeat() {
        let config = LoopAbortConfig {
            ngram: 1,
            max_repeats: 5,
            ..LoopAbortConfig::default()
        };
        let mut det = detector(config);
        for _ in 0..4 {
            assert_eq!(det.push("\n"), None);
        }
        assert_eq!(
            det.push("\n"),
            Some(LoopAbort::RepeatedNgram {
                ngram: 1,
                repeats: 5
            })
        );
    }

    #[test]
    fn repeated_line_fires_exactly_at_the_threshold() {
        let config = LoopAbortConfig::default();
        let line = "0123456789abcdef\n";
        let mut at_threshold = detector(config);
        let mut fired = None;
        for _ in 0..4 {
            fired = at_threshold.push(line).or(fired);
        }
        assert_eq!(
            fired,
            Some(LoopAbort::RepeatedLine {
                line: line.to_owned(),
                repeats: 4
            })
        );

        let mut just_below = detector(config);
        for _ in 0..3 {
            assert_eq!(just_below.push(line), None);
        }
        assert!(!just_below.aborted());
    }

    #[test]
    fn short_lines_are_ignored() {
        let config = LoopAbortConfig::default();
        let short = "0123456789abcde\n"; // 15 payload bytes, one below min_line_bytes
        let mut det = detector(config);
        for _ in 0..10 {
            assert_eq!(det.push(short), None);
        }
        assert!(!det.aborted());
    }

    #[test]
    fn a_repeated_piece_spanning_several_lines_is_split_correctly() {
        let config = LoopAbortConfig::default();
        let line = "the quick brown fox\n";
        let chunk = format!("{line}{line}");
        let mut det = detector(config);
        assert_eq!(det.push(&line.repeat(2)), None);
        // The fourth identical line completes inside the second chunk.
        assert_eq!(
            det.push(&chunk),
            Some(LoopAbort::RepeatedLine {
                line: line.to_owned(),
                repeats: 4
            })
        );
    }

    #[test]
    fn detector_latches_after_firing() {
        let config = LoopAbortConfig {
            ngram: 1,
            max_repeats: 3,
            ..LoopAbortConfig::default()
        };
        let mut det = detector(config);
        assert_eq!(det.push("x"), None);
        assert_eq!(det.push("x"), None);
        let abort = det.push("x").expect("fires on the third repeat");
        assert_eq!(abort.reason(), "repeated_ngram");
        assert_eq!(det.push("x"), None);
        assert!(det.aborted());
        assert!(det.abort_reason().is_some());
    }

    #[test]
    fn legitimate_prose_over_two_thousand_tokens_does_not_fire() {
        // Public-domain prose (US Declaration of Independence), cycled to ~2000 word
        // tokens with a newline every twelve words: no 8-gram and no line repeats.
        let paragraph = "when in the course of human events it becomes necessary for one people \
            to dissolve the political bands which have connected them with another and to assume \
            among the powers of the earth the separate and equal station to which the laws of \
            nature and of natures god entitle them a decent respect to the opinions of mankind \
            requires that they should declare the causes which impel them to the separation we \
            hold these truths to be self evident that all men are created equal that they are \
            endowed by their creator with certain unalienable rights that among these are life \
            liberty and the pursuit of happiness that to secure these rights governments are \
            instituted among men deriving their just powers from the consent of the governed";
        let words: Vec<&str> = paragraph.split_whitespace().collect();
        let mut det = LoopAbortDetector::new(LoopAbortConfig::default());
        let mut emitted = 0usize;
        while emitted < 2100 {
            for (index, word) in words.iter().enumerate() {
                if det.push(word).is_some() {
                    panic!("false positive on legitimate prose at token {emitted}");
                }
                emitted += 1;
                if index % 12 == 11 {
                    assert_eq!(det.push("\n"), None);
                }
            }
        }
        assert!(emitted >= 2000);
        assert!(!det.aborted());
    }

    #[test]
    fn pathological_newline_flood_fires() {
        let mut det = LoopAbortDetector::new(LoopAbortConfig::default());
        let mut fired = false;
        for _ in 0..256 {
            if det.push("\n").is_some() {
                fired = true;
                break;
            }
        }
        assert!(fired);
        assert_eq!(
            det.abort_reason().map(LoopAbort::reason),
            Some("repeated_ngram")
        );
    }

    #[test]
    fn zero_ngram_disables_ngram_detection() {
        let config = LoopAbortConfig {
            ngram: 0,
            ..LoopAbortConfig::default()
        };
        let mut det = detector(config);
        for _ in 0..1000 {
            assert_eq!(det.push("\n"), None);
        }
        assert!(!det.aborted());
    }

    #[test]
    fn over_long_lines_are_skipped_without_unbounded_growth() {
        let config = LoopAbortConfig::default();
        let huge = "z".repeat(LINE_BUFFER_CAP * 2);
        let mut det = detector(config);
        for _ in 0..8 {
            assert_eq!(det.push(&huge), None);
            assert_eq!(det.push("\n"), None);
        }
        assert!(!det.aborted());
        assert!(det.line_buf.len() <= LINE_BUFFER_CAP);
        assert!(det.last_line.len() <= LINE_BUFFER_CAP);
    }
}
