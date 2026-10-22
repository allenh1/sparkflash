# S7 record draft - first served device token (host rehearsal, UNMEASURED)

Status (2026-10-21): **S7 is NOT achieved against a real device window.** The block advances S7
end to end to the refusal boundary and records it: the serving stack is wired on the M2
lean+hoist base, both m1y34 rows load and are refused by every claim form, the refusal is
fail-closed over all four OpenAI routes, and the device *serving* path itself is proven on a
synthetic claimed window (paper evidence). The m1y34 rows are measured M2 performance rows, not
D41 acceptance rows: **no token-acceptance, tok/s or TTFT claim is made from this record.**

## 1. What was asked and what this proves

S7 (docs/build/serving-s7-s8-plan.md sections 6 and 11) asks whether a claimed device token
window survives the HTTP path: provenance header, decoded text equals the tokenizer seam,
buffered and streamed agree, usage numbers, no engine entry, and no timing claim. S8 adds the
hard token evidence (nine reference matches and the writer's claim assertion).

This block:

1. Finishes the W3 CLI wiring on the M2 lean+hoist base (task/serve-openai-parity cherry-picked
   onto win/m1w25 6a40bfa5): --device-token-dir, --device-token-derived-claim, the claim-state
   startup line, and the seam decoder, plus the help text that documents them.
2. Reads the m1y34-identity and m1y34-lean windows with the real consumer and the python probe:
   both load; both are refused in asserted and derived form; the new forward_perf field does not
   reach the envelope (progress rows are not part of it) and no consumer assumption broke.
3. Runs the OpenAI routes end to end on a host loopback server: /v1/completions buffered and SSE,
   /v1/chat/completions buffered and SSE, provenance headers, claim validation and every refusal
   reason, host-twin fallback, and the DSML streaming tool calls, all with a device window
   installed (tests and a recorded host rehearsal).
4. Produces this record and the S8 delta list.

What the wire actually served from a real window: nothing device-side. The real window is
refused before any request, and the host twin answers with no device header (section 4.1). What
the wire serves from the *synthetic* claimed window is the full device path (section 4.2): that is
paper evidence from a fixture, not a device run.

## 2. Provenance

- Worktree: /home/user/wt/serve-parity (clean at record time: /home/user/ev-session-20261020/serve/s7-m1y34-20261021T122243Z/worktree-status.txt is empty).
- Branch: task/serve-s7-next at **0e87f54708043bd6df6139205a0ab276d571fcbe** (the tree the
  release binary and the rehearsal came from; this record is the commit on top of it), based on
  win/m1w25 **6a40bfa5** (M2 lean+hoist kit).
  The six serve commits from task/serve-openai-parity (tip 3a9a8286) were cherry-picked without
  conflicts: f4edb0af (consumer), 695e1057 (CLI+probe), db86ea7e (example), d96f473a (docs),
  43b58592 and e8e6382e (docs, m1y14 baseline), then 0e87f547 (this block: e2e tests, m1y34
  shape pin, fallback probe, plan section 12).
- Binary: target/release/sparkflash, sha256 **083a916006a998165831a42e11fadd86d11ff93128bebe707bcb3533c27b32ee** (/home/user/ev-session-20261020/serve/s7-m1y34-20261021T122243Z/binary.sha256).
- Record run: 2026-10-21T12:22:43Z, host only (no node, no GPU, no vLLM, loopback socket).
- Gates at the record tip (gate log /tmp/s7-gates.log, sha256
  894a181dd38e9af966a245a7359723b2f593819cc137f8b7348180a8ecacadef):
  * cargo fmt --all: clean;
  * cargo clippy --workspace --all-targets -- -D warnings: clean;
  * cargo test --workspace: 104 suites, **2148 passed, 0 failed** (includes the 27
    crates/sparkflash-cli/tests/serve_http.rs host e2e tests and the new device_chain unit test);
  * python3 -m unittest discover -s scripts/tests: **374 tests OK** (14 of them the probe script).

### 2.1 Windows (measured M2 perf rows, not D41 acceptance rows)

- `/home/user/ev-session-20261020/window/m1y34-identity`: run=m1y34-identity tag=m1w34
  kind=token, started 2026-10-21T11:46:34Z, binary
  e3470a141a886bdc3413c12073fd0a16d37e445bbc9b39aca4bb70ce452643eb, cubins manifest
  dd9b7b0cbf065acba02db7c65e3c658b908274e2c9c7fbe8e310224500eacfac, checkpoint snapshot
  1bae0e1088993b9b82de08f59de5be5a34e42170, peers 192.0.2.80/.88/.59/.68, argv
  --forward-reference-ids 491,367,201,80,3302,201,54,201,4341 (copy: run.meta-identity,
  argv-identity.txt).
- `/home/user/ev-session-20261020/window/m1y34-lean`: same tag/binary/cubins/peers, started
  11:56:11Z, argv additionally --forward-lean (copy: run.meta-lean, argv-lean.txt).
- Full file sha listings: /home/user/ev-session-20261020/serve/s7-m1y34-20261021T122243Z/window-identity.sha256 and /home/user/ev-session-20261020/serve/s7-m1y34-20261021T122243Z/window-lean.sha256.

Key record shas (the full listings are authoritative; these are the claim inputs):

| record | identity | lean |
| --- | --- | --- |
| run.meta | 7ff6a6a0c804f4563d1e5a6484a522b8e050b32112ce3f9928581e6cddeac604 | 7dee10a00bb883d7f4bde62ca9433029ddfefd849adf39a25eddceef2857dc57 |
| argv.txt | fdb601f6a46c04694f44761ed057ab383521e1486f3fb9d4f9d8e86e272c32b3 | 2f9b914ebe23fd405513217f37d7d456e601b1360c1800887eda26e233cde827 |

| rank | forward-embed identity | forward-embed lean | token identity | token lean |
| --- | --- | --- | --- | --- |
| rank0 | f42f093ac8d6138c052b2439627a004d8422f0d20fe2eb6aafebe4cb3df20db5 | 4f994ce781e22c01657c1d1b76b130643a62ca0fc992ee0377c6e5b8149b156e | 7651e53abb15e4398e89cac6867967ed39b618473caca75af7acabf1c7e0c81c | 67fb5a454abc835d6cbf8f6d430530407308cae05cd8735bc6e0fc8cb28fd74b |
| rank1 | 27b8eed2c41f41d5b130511766cfb62d21390cb13e7ba3429f4cdf570a2d19d7 | fcf6009f56d7ce6142de8bf1f90c677a0f8f270a6793f7f13200f2ab083757f3 | b32b91e99455079e1d1bd06d3e7a7b66a3fe4f177dbe215646fa0cdd8fefde39 | 91f3b261d39ebeb3c3db03190d731f24f8e0faacde034d9a487dc20829485373 |
| rank2 | a1160df09fd43b964be9959afd04924af95c7ab1984147ef00aec581b3740962 | a7fd421fdf987bc54589ab126006cb5580dfd780f40e718580157e1d672bbf70 | 4c9ddb2dc1ca25707f307d9881074414e85ebd7923fa1912438ab577a90ddac9 | 309b009f6a9fda592fa4283aa4a48fea916b58b55b14270e46b55a2e02ca3a8b |
| rank3 | 52701a0dd687156f3b4c49b79c694dbd1f63f933eacc80a638dbd165b391da1a | f78dd4a92791237ca4b4f7ee6043fdab7ffe557c9cfa0322b07361afbd3b58d0 | ffaeaa5e996744aaf7addc6ee17d9e67d8e0c16573fb46dae97c2493fd53483f | b59c8ed820561c4eea6787e60dc8db3513b68a3c2aa058ba2f7c64751b77f0aa |

The logits dump binary is the same file in both rows and on all four ranks: sha256
`aed6054d77d6704c29812463106039fcdd8e39f34920e2c67aa00e9b52e037c2` (tp4-logits-r{rank}.jsonl.bin, 4,654,080 bytes each).

## 3. Claim verdicts (python mirror; the Rust consumer is authoritative)

Both rows, both forms, `scripts/serve_window_probe.py inspect` (exit 2 = refused):

| row | form | claimable | reason (window-claim.json) |
| --- | --- | --- | --- |
| m1y34-identity | asserted | no | rank 0..3: embed does not assert token_claimed=true (the device writer currently records False) |
| m1y34-identity | derived | no | rank 0: matches_reference is [True, False, True, False, False, False, False, False, False], expected nine true matches |
| m1y34-lean | asserted | no | same four token_claimed reasons |
| m1y34-lean | derived | no | same 2/9 matches_reference reason |

Evidence: claim-identity-asserted/, claim-identity-derived/, claim-lean-asserted/,
claim-lean-derived/ (each with window-claim.json and the inspect log).

The server prints the same decision at startup and keeps the host twin:

    sparkflash serve: device token source .../m1y34-identity: explicit claim refused, serving the
    host twin for every request: no rank's forward record asserts token_claimed=true; ... rank 0:
    the forward token batch does not match the reference at every position: [Some(true),
    Some(false), Some(true), Some(false), ...] ...

    sparkflash serve: device token source .../m1y34-identity: derived claim refused, serving the
    host twin for every request: rank 0: the decoded ids do not all match the reference ids:
    [Some(true), Some(false), Some(true), Some(false), ...] ...

(Full lines: serve-identity-asserted.err, serve-identity-derived.err, serve-lean-derived.err.)

The window itself: prompt_ids 130, forward_ids 138, positions 129..137, dump present,
token_claimed false, evidence_written "loop-complete, position-complete, head-complete,
token-consensus-complete", ids [491, 92928, 201, 15, 4893, 16646, 15, 14265, 86], reference
[491, 367, 201, 80, 3302, 201, 54, 201, 4341], matches [T, F, T, F, F, F, F, F, F] on every
rank, head.argmax.token 86 (= ids[8]) with value 10.49007511138916 and certified false. So the
refusal is the whole loop evidence against the reference: the claim rules are not relaxed here and
must not be.

## 4. Served bytes

### 4.1 Real window: refusal is fail-closed on every route (recorded)

Three loopback server runs on the release binary:

| run | flags | url | startup | probe |
| --- | --- | --- | --- | --- |
| identity asserted | --device-token-dir identity | http://127.0.0.1:45731 | explicit claim refused (reasons above) | fallback probe PASS (14 checks) |
| identity derived | + --device-token-derived-claim | http://127.0.0.1:33445 | derived claim refused | fallback probe PASS (14 checks) |
| lean derived | --device-token-dir lean + derived flag | http://127.0.0.1:38159 | derived claim refused | fallback probe PASS (14 checks) |

Each fallback probe (`probe --expect fallback`) asserts: buffered 200, no
x-sparkflash-token-source, no x-sparkflash-token-claim, the host body carries the
x_deepseek_spark_stats block, object text_completion, finish_reason set, host usage shape; and for
the SSE request: 200, no device headers, [DONE], deltas join the buffered text, finish_reason set.
The probe refuses to run on a claimable window, so this record cannot be mislabelled.

Engine-entry counter (S7 protocol step 7) on the identity-asserted run:
`sparkflash_generation_tokens_total` 0 before the two probe requests and **8 after** (2 requests x
4 host tokens): the host twin did the work while the device source stayed installed. Files:
metrics-identity-asserted-before.txt / -after.txt.

Artifacts: fallback-identity-asserted/, fallback-identity-derived/, fallback-lean-derived/ each
with probe-buffered.json, probe-stream.sse, probe-summary.json and FALLBACK.md; server logs
serve-*.out/.err/.url.

### 4.2 Synthetic claimed window: the device route itself (paper, UNMEASURED)

Written by the device-token-fixtures example (`write_device_window ... asserted`); no device
produced these records. Path /home/user/ev-session-20261020/serve/s7-m1y34-20261021T122243Z/window-synthetic-asserted, shas in
window-synthetic.sha256 and below:

| rank | forward-embed | token |
| --- | --- | --- |
| rank0 | a4ac42da9c630b44a0748d3d1103606d5ba75dc3cc2347814ebc4d74532692e0 | 0889be9ed0be2b85518cf0799f6740f00acbe070042c3b2a8882f8ce10ef4bf4 |
| rank1 | 5f35f99a55de650bb9bbca9cf027a401ae13543ac29fa0d6f4859ff07020997b | 259b5e56a7112d04adc2af7a72065c2f1ba1ca7c4e0a7f927e8d535640abe1f3 |
| rank2 | a9dae8c260785ebfe58ff563e68e6ba63e60991d90ed835a5343b60a0c7d503d | b7121c1e6d23cd639665930262b5d4713c3ab6745e308a1ff0c8e65c90e816eb |
| rank3 | 0ec5c76357ec1fe0e0118223b74ad9d5f881e3d865f7505229628d32f099773b | d0773c67c0adccd2d351593fedc1acf8996e2f851171d390f687a1a16d92ccb5 |

Server startup: `explicit claim ok (9 tokens, prompt_len 130, dump present)` (server log
serve-synthetic-claimed.err). Probe --mode asserted: **PASS (18 checks)** and the same on the
metrics rerun (claimed-synthetic-metrics/): both provenance headers exact
(x-sparkflash-token-source: device-chain-envelope (UNMEASURED), x-sparkflash-token-claim:
explicit), no host stats key, buffered usage 130/9/139, finish_reason length, nine content frames,
stream usage equals buffered usage, [DONE], and the nine streamed pieces
[")", "4", "?", "J", "U", "`", "k", "v", "�"] join to the served text, which equals the
tokenizer seam decode of the synthetic ids [11, 22, 33, 44, 55, 66, 77, 88, 99].

Engine-entry counter on the claimed rerun: metrics-synthetic-before.txt and
metrics-synthetic-after.txt are **byte identical** (diff empty);
`sparkflash_generation_tokens_total 0` after both requests, so a claimed device serve does not
enter the engine. That is the S7 protocol step 7 evidence for the device path, on paper records.

This proves W3 and the wire decoder. It is NOT a device token: the ids are fixture ids and the
labels say UNMEASURED.

## 5. Routes and refusals exercised end to end (host)

crates/sparkflash-cli/tests/serve_http.rs (real binary, loopback socket; 27 tests, all pass):

- `the_serve_help_documents_the_device_token_flags`: the help lists --device-token-dir and
  --device-token-derived-claim (the CLI prints help on stderr with its usage exit code; the flag
  inventory is the contract).
- `a_device_token_window_serves_its_claimed_completion_and_labels_it_unmeasured`: buffered
  /v1/completions on a claimed synthetic window; both headers; usage 130/9/139; decoded text
  equals seam decode; startup says claim ok.
- `a_device_token_window_streams_its_claimed_completion_and_labels_it_unmeasured`: the same over
  SSE; nine content frames, length finish, usage chunk, [DONE].
- `chat_requests_fall_back_to_the_host_twin_with_a_device_window_installed`: /v1/chat/completions
  buffered and SSE; no device headers, host stats block, streamed content equals buffered content.
- `streaming_tool_calls_still_work_through_the_host_twin_with_a_device_window_installed`: DSML
  scripted tool calls (docs/build/streaming-tool-calls.md behaviour) through the host twin with a
  device window installed; buffered tool_calls and reassembled streamed calls are equal,
  finish_reason tool_calls, still no device headers.
- `a_refused_device_window_is_fail_closed_and_names_the_reason`: startup names the reasons; a
  request gets the host twin and no device header; the same schema claims in derived form with
  the flag and serves with `x-sparkflash-token-claim: derived`.
- `the_m1y34_forward_perf_progress_rows_are_not_part_of_the_envelope` (device_chain unit test):
  a window directory carrying m1y34-shaped tp4-forward-progress-r{rank}.json rows loads and
  claims exactly as before.

The device source participates only in /v1/completions when the prompt matches the recorded
prompt_ids; every chat request, every prompt mismatch, and every load error falls back to the host
twin (or refuses startup for a load error). The refusal reason for chat is recorded in the source
tests: a chat text prompt has no token ids to compare.

Probe script: 14 unit tests (including 4 for --expect fallback) in
scripts/tests/test_serve_window_probe.py; full scripts suite 374 tests OK.

## 6. m1y34 evidence-shape findings (task item 2)

1. **forward_perf is not in the envelope.** The field appears only inside
   tp4-forward-progress-r{rank}.json entries (40 per rank, lean false on identity and true on
   lean). The consumer loads only tp4-forward-embed-r{rank}.json and tp4-token-r{rank}.json and
   serde ignores unknown fields, so the new evidence shape cannot reach the claim. The python
   mirror ignores the file too; a unit test now pins this.
2. **The lean row differs only in diagnostics.** Verified by diffing rank0 records and by
   canonicalising the claim inputs:
   - attention body twin evidence is replaced by `{"skipped": true}` (identity records the full
     twin metrics); every layer_chain entry has twin_* null, twin_skipped true,
     forward_perf.lean true, and a lower wall_ms (perf evidence only, not a claim);
   - argv adds --forward-lean (4 rank lines);
   - head.input is the same five-field object in both (sha256, bytes, bf16_codes_sha256,
     bf16_codes_bytes, file), and both records pin the same file name
     pre_head_hidden-pos137.f32, differing only in the recording run directory. The consumer
     never opens it.
3. **No consumer assumption broke, and the claim inputs are identical.** Canonical sha of the
   embed tokens block is `e361d8c6a4282b10` and of the token record tokens block is
   `afd30b025130cba3` for both rows; prompt_ids_sha256 `e061b806...`, forward_ids_sha256
   `82494c40...`, dump bin sha `aed6054d...`, ids and matches identical. The consumer never stats
   or opens head.input.file (paths are node-absolute and may not exist on the serving host - the
   open evidence nit in M2 section 7.1), never reads the dump bytes, never reads progress cadence
   or twin fields. The identity and lean rows therefore produce the same claim decision, which
   is exactly what the startup logs show.
4. Both rows refuse: token_claimed false (asserted) and 2/9 reference matches (derived). The
   failure mode is the same first token 491 with a wrong continuation - the known D41 gap
   recorded in plan section 11.2, not a serving defect and not something the serving side may
   paper over.

## 7. What S8 (and a real S7) still needs

1. **Nine of nine reference matches** from the four-rank forward loop for the recorded prompt
   (m1y34: 2/9, m1y14 baseline: 2/9; the known Engram-write cause in plan section 11.2 unless the
   device side finds a different root cause first). No serving change substitutes for this.
2. **token_claimed=true on every rank** (the device writer assertion) for the asserted S8 form.
   The derived form (S7 rehearsal) does not need it once the nine matches hold; both forms also
   enforce the nine matches, so a derived run cannot bypass item 1.
3. Then, on the same host recipe as this record: `inspect --mode asserted` claimable, startup
   `explicit claim ok`, `probe --mode asserted` PASS (18 checks), metrics generation counter
   unchanged, plus the S8 plan section 7 items not yet recorded here: two runs reproducing the
   ids, head evidence partitions, prompt_ids equal to the request, and the accepted-token labels.
4. One doc nit for the device lead, already in plan section 11.3: the head argmax rule in plan
   section 7.3 says ids[0] but the recorded windows prove ids[8] (the last dumped position); the
   consumer implements ids[8]. Fix the plan text when the record becomes the S8 gate.

No serving work is expected before item 1 lands; if a real window appears with 9/9 and
token_claimed, this record's commands are the S7 recipe and the S8 run can reuse them with
--mode asserted.

## 8. Non-claims

- The m1y34 rows are M2 measured performance rows, not D41 acceptance rows. Nothing here says the
  device token is right or accepted; both rows are refused by the claim on purpose.
- No tok/s, TTFT, throughput or quality claim is made from any body served in this record.
- The synthetic window is paper: its ids, text and dump came from a fixture, not a device.
- The fail-closed fallback text is the host twin's synthetic-plan output (here the empty string for
  ids the released tokenizer has no text for); it is evidence about which path answered, not about
  model output.
- No merge, no staging, no node or GPU action happened in this block; the branch is host work.

## 9. Artifact index

All paths are under /home/user/ev-session-20261020/serve/s7-m1y34-20261021T122243Z unless noted:

| artifact | meaning |
| --- | --- |
| branch-tip.txt, commit.txt, worktree-status.txt | provenance at the run |
| binary.sha256 | release binary hash |
| window-identity.sha256, window-lean.sha256, window-synthetic.sha256 | every window file hash |
| run.meta-identity, argv-identity.txt, run.meta-lean, argv-lean.txt | recorded run metadata copies |
| claim-identity-asserted/, claim-identity-derived/, claim-lean-asserted/, claim-lean-derived/ | window-claim.json + inspect log |
| fallback-identity-asserted/, fallback-identity-derived/, fallback-lean-derived/ | fail-closed probe artifacts (probe-buffered.json, probe-stream.sse, probe-summary.json, FALLBACK.md) |
| serve-*.out, serve-*.err, serve-*.url | server logs and bound URLs for every run |
| metrics-identity-asserted-before.txt / -after.txt | host twin entered the engine (0 -> 8 tokens) |
| claimed-synthetic/, claimed-synthetic-metrics/ | the device route on the synthetic window (18 checks each) |
| metrics-synthetic-before.txt / -after.txt | byte-identical: no engine entry on a claimed serve |
| window-synthetic-asserted/ | the synthetic claimed envelope |

## 10. Reproduction commands (host only)

A release binary at 0e87f54708043bd6df6139205a0ab276d571fcbe served every window. The only
variable is W (identity, lean, or the synthetic fixture) and the claim flag; the record run used
these exact shapes:

    cargo build --release -p sparkflash-cli --bin sparkflash
    python3 scripts/serve_window_probe.py inspect --window W --arc-dir A --mode asserted
    sparkflash serve --synthetic --bind 127.0.0.1:0 --tokenizer reference/tokenizer.json \
        --device-token-dir W
    python3 scripts/serve_window_probe.py probe --window W --base-url URL --arc-dir A \
        --mode asserted --expect fallback   # refused window: fail-closed record
    # claimed window (synthetic fixture or a future real one):
    python3 scripts/serve_window_probe.py probe --window W --base-url URL --arc-dir A \
        --mode asserted                       # device route, 18 checks

