# task-32 evidence bundle

All analysis records here were produced offline from frozen JSON files; nothing here needs a
server. Large inputs (fixtures, oracle dumps, device logits) are listed under "External inputs" and
are kept on the node; the sha256 below identifies them exactly.

## Analysis records in this directory

| file | what it is | sha256 |
|:--|:--|:--|
| control-p7.json | f32 prefill at 8 tokens vs served oracle at position 7 (random prompt) | ad1c209b6d12a751a4cac1308de0a079818edf03cb93edbadaf6ff07a8eb3a42 |
| control-p128.json | f32 prefill at 129 tokens vs oracle position 128 | 415f8983a195a5e1d2d3acb4819dbb0fb5cac8e4c09cdf68c62149768e230358 |
| control-p129.json | f32 prefill at 130 tokens vs oracle position 129 (the incident) | 04f9b8e234ff6bf782ea09f1b041abbe5297754847b339e8b06fa0e35095e1b9 |
| control-natural-p7.json | in-distribution short control (code-rust-borrow, 8 tokens) | see git blob |
| three-way-p129.json | f32 / device-native / oracle at position 129, pairwise overlap and correlation | d51b97103160da04850bbdfbcf6724f3c74517d92c5e46c191df3d121f7fb0cb |
| d41-floor.json | oracle run-to-run floor: rank-1, top-k set overlap, per-token delta percentiles, shared-band KL | see git blob |
| oracle-repeat.json | per-position repeat metrics 129..137 (139- vs 140-token request) | 28f802f1dce2f382e67aa3d40a672c3abaa9b450a8acf1a522ad6c1208862659 |
| d41-floor-warm.json | identical-shape warm repeats: bit-identical, floor 0 | 2434ae707eb97abaf6a12147328abb0c1ea9f034d84bd67186def96498726920 |
| d41-shape.json | batch-of-4 vs single-prompt shape probe (worst 4.06 nats, KL 0.93, one rank-1 flip at a flat position) | cc02a30383f60ceb6806f7dfb814c9852d528fdd66b1699b4a9fbff01240e874 |
| matched-shape-rows.json | minimal-shape oracle rows 7/128/129 (9/130/131-token requests) | 5e89c31d60fa220d9095401139d6efba4a8651734df64bab5d0c346c3474c9a7 |
| length-probe-rows.json | same rows 128/129 at prompt lengths p+2, p+3, p+4 (length envelope, busy batch state) | 5fae4ee9933a551e9a1da486971ceb6e1c37fd9b8c78402e7fc90d67607f6400 |
| batch-mode-probe.json | idle / concurrent-generation / idle blocks; same rows move deterministically | 8fbd4f76e63909ee37948d26d1289aa91501b5d278e8fa8646e883fdc364260e |

The sha256 of the analysis records equals the sha256 of the corresponding files on the node where
they were generated. Recompute with sha256sum in this directory if needed.

STATUS of the length envelope: length-probe-rows.json is a BUSY batch-state capture (an external
client was sending chat completions during it) and is therefore not the D41 floor. An idle-mode
re-capture with the same ids/shapes and idle_A -> busy -> idle_B blocks is staged on the node
(window-2026-10-18/oracle_length_envelope_idle_task32.py) but PENDING: window policy forbids server
contact while the user is using the vLLM instance. Do not freeze D41 on the busy table.

Warm-repeat raw artefacts (too large/duplicative for git) are on the node at
phase2-evidence/window-2026-10-18/task32-repeats/: native-oracle-repeats.json
(86731124ec814f1b103a42c04e682e2a38a371029185b61f07a24df1a53368ca), native-oracle-batch4.json
(d6e2e1e840972b92cde19609c33a8d98a1ae133e90aa24a0a368d49b56fa37ae) and the three raw responses.
The repeats were taken on the released server restarted 18:03:25Z, served root 1bae0e1089...,
requests: /v1/completions, prompt = 130 fixture ids + 8 continuation + sentinel 0 (139 tokens),
max_tokens 1, temperature 0.0, top_p 1.0, ignore_eos true, prompt_logprobs 1000.

## External inputs (node paths and sha256)

| file | node path | sha256 |
|:--|:--|:--|
| golden fixture (old binary) | /home/user/phase2-evidence/sequence-130-released.json | e6df8290dcc9cfc96af27ad1a4da165e5f3d8d4e9aac0f98819b0030b738b308 |
| current-engine fixture (prompt 130) | /tmp/task32/b69-130.json | 5f35be3e2c56abf96b743131e45570f2d6bc554d4a9ae49195945a229f7fd89f |
| current-engine fixture (prompt 8) | /tmp/task32/b69-8.json | (logits identical to f32-prompt8.json) |
| f32 8-token fixture | /tmp/task32/f32-prompt8.json | 321c2aaffee5b14b16c3bc1161c6eeb5556ecf03bd3a49bf78b1dfb17291a346 |
| f32 129-token fixture | /tmp/task32/f32-prompt128.json | 466d36b5a3627c8850adbb0f51ca1d799f9ceb5912321a9972504c39fa5a859f |
| oracle suite run (17:30:22Z, len 140) | /tmp/native-oracle-suite.json | efce16c2637aeab70475a43b8dc0c5cbe1f066b6bb9cd671480c7d7e03c4572a |
| oracle released run (17:28:30Z, len 138) | /tmp/native-oracle-released.json | 8ecc423d0e6240811421d95823caaeab396209a8cef8d7ef9b6fe54d3521f475 |
| oracle prompt suite | /tmp/suite.json | 42f66ad2... (recorded inside both oracle files) |
| device dump index | window-2026-10-18/runs70/logits-130x8.jsonl | f548d9c3115c0557cc277ed1b32ce2c82985ec38d299869163759f31fc30aea6 |
| device dump logits | window-2026-10-18/runs70/logits-130x8.jsonl.bin | 43b31438c01955c8359457e077322fc39e6c6a76f330bdf958e87d9ddcad07b5 |
| device binary | window-2026-10-18/bin/sparkflash_m3c_c4c_seqreuse_4f599b7_aarch64 | 7e0ce4bf2390ec078f70098eb5e77881b518e5d0ed58f9155b669a355a8d48cc |

## How the records were produced

Engine fixture (cross-compiled on the dev host, run on the node; the generator gained a
--prompt-ids option in this branch):

    sequence_fixture_task32 --config released --checkpoint <snapshot 1bae0e10...> \
        --prompt-ids ids.json --decode-steps 1 --logits prefill --provenance node \
        --out fixture.json

Offline comparison (one fixture prefill position against one oracle row):

    python3 scripts/compare_fixture_vs_oracle.py --fixture fixture.json \
        --oracle native-oracle-suite.json --suite suite.json \
        --prompt-id b69-fixture-130 \
        --continuation '[491,367,201,80,3302,201,54,201,4341,0]' --position 129 \
        --out control-p129.json

The script verifies the suite sha256, that the fixture prompt ids are a prefix of the oracle prompt
ids, and that the fixture's next token equals the oracle row's target token before comparing. It
writes top-1 agreement, top-k overlaps at 5/20/100/1000, max/mean absolute delta, KL(oracle to
device), JS divergence and the fixture/oracle provenance hashes.

Device comparison: the binary writes the jsonl index (position, argmax, reference_id, hybrid_id,
fixture_id, top20, logits_offset) plus one float32 row per position in the .bin. Read a row with
offset*4 and vocab 129280; the "reference vs fixture ... equal=true" and "logits 129: 128528 ...
max 15.009581 ... argmax 8060" lines in logits-130x8.log are the chain's own checks.
