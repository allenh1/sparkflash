Fixture: logits-dump-demo-scale.jsonl

This is a byte-for-byte copy of the existing D40-protocol demo dump
`/home/user/phase2-evidence/demo-scale/logits-scale.jsonl`, produced on 2026-10-19 before the
in-tree writer existed (sha256
`459bd9471f30629846f4859e124be2bbf2283369512a70c0df460fb788624a85`, 9 positions 129..137, vocab
129,280, 4,627 bytes, sidecar 4,654,080 bytes). It is kept as the *existing-dump* compatibility
fixture for `crates/sparkflash-cli/src/logits_dump.rs`: it was not written by that module, so it
shows the schema the merged chain's writer has to stay read-compatible with. Only the JSONL is
copied; the 4.6 MB sidecar is not, and the tests do not need it.
