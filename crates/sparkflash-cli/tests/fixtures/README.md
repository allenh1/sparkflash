Fixture: logits-dump-demo-scale.jsonl

This is a byte-for-byte copy of the existing D40-protocol demo dump
`/home/user/phase2-evidence/demo-scale/logits-scale.jsonl`, produced on 2026-10-19 before the
in-tree writer existed (sha256
`1ff1a29c7891bf9fce71c10c609ecf5b176c76871c3eb0c236e872963865a510`, 9 positions 129..137, vocab
129,280, 4,627 bytes, sidecar 4,654,080 bytes). It is kept as the *existing-dump* compatibility
fixture for `crates/sparkflash-cli/src/logits_dump.rs`: it was not written by that module, so it
shows the schema the merged chain's writer has to stay read-compatible with. Only the JSONL is
copied; the 4.6 MB sidecar is not, and the tests do not need it.

Fixture: m1y16-pre-head-hidden-pos129.f32

This is a byte-for-byte copy of rank 0's pre-head hidden row at position 129 from the m1w28/m1y16
four-rank token window: /home/user/ev-session-20261020/window/m1y16/rank0/head-input/pre_head_hidden-pos129.f32
(20,480 bytes: one row of 5,120 little-endian f32 values; sha256
e8ad36f1676f4f7a0765617b181b398366ccdbdc400bcf9e70952a5ec5dc91a0). It is the captured activation the
native-dense host twin test (tp4_forward::tests::the_native_dense_launch_arithmetic_equals_the_engine_coded_matmul_on_the_captured_row)
quantises and multiplies by a synthetic B12x weight, so the equivalence between the kernel's host
arithmetic twin and the engine's coded matmul runs on a real device activation row rather than a
synthetic one. The run's evidence is the m1y16 bundle, tag m1w28, binary sha
2d7cff6d61c797dc227458eaf1310f6b127c8283f5f62223b8cb60a0da94c493.

