#!/usr/bin/env python3
"""Verdict tool for the tp4 device-layer frontier documents: refuse overclaims.

WHAT IT READS
    <row-dir>/tp4-layer-device-r{0,1,2,3}.json           (flat collection)
    <row-dir>/rank{0,1,2,3}/tp4-layer-device-r{0,1,2,3}.json  (collected four-rank layout)
One document per rank, written by write_layer_device_frontier in
crates/sparkflash-cli/src/tp4_forward.rs (win/layer-device @ 5d5a5ffd). The writer emits it on
both device-layer paths: the refused path (status "refused", with a "message" key) and the
completed path (status "completed", without "message").

THE FRONTIER, AND WHY PARSING IT IS THE POINT
    StageRuns::summary() in crates/sparkflash-cli/src/layer_device.rs is free text:
      "device-only through A -> B -> C (3 stage(s)); stopped at <stage>: <message>"
      "no stage ran; stopped at <stage>: <message>"                      (degenerate)
      "device-only through A -> ... -> ffn_epilogue (19 stage(s))"       (completed)
    The stage order is STAGE_CONTRACTS (layer_device.rs), listed once in STAGE_ORDER below: 19
    stages, with moe_shared between all_reduce_rdma_arm and ffn_epilogue (the shared expert had no
    stage of its own in the first 18-stage contract). Adding a stage is a one-line change to
    STAGE_ORDER; every prefix and length check in this tool reads that constant.
    kv_path stays ONE stage, and the reason is checkable from the read it performs: the window slice
    is [position - window_width ..= position - 1], so the row appended at position is outside every
    read the current step performs. For decode, append-before-read and append-after-read produce the
    same number, so splitting rotate/quant from append would add a stage without changing a value.
    Multi-position/prefill is the condition that will force that split (and a 20th stage); until it
    lands, the contract is 19.
    The check that makes this tool worth running: the chain must be an exact PREFIX of that
    order, and the stopped stage must be the NEXT stage after the chain (the chain length is its
    index). A refusal whose chain is "no stage ran" while it stopped at kv_path is a contradiction.
    The refused document is built by StageRuns::from_refusal(&error), which knows only the refused
    stage, so it LOSES the stages that actually ran. That is a known defect (gemv is fixing it by
    recording into DeviceLayer); this tool exits 1 on it and will pass unchanged once both
    documents carry the real runs. It is deliberately not softened to accept both shapes, because
    accepting a lost chain would hide exactly the defect the first probe row is meant to catch.

PROBE MODE (default) - a refused row's four documents
    a. all four documents present, parseable, same schema, same "frontier" string. A rank whose
       frontier differs is a real finding: it means the stage order or a refusal path differs per
       rank, so the row cannot serve as a probe.
    b. status == "refused"; arm == the device arm string (DEVICE_ARM below : layer_arm.why() from
       layer_device.rs). A host-arm document masquerading as a probe is exit 1, never a warning.
       The row's SPARKFLASH_LAYER_DEVICE is NOT recorded in the document (the arm string is the
       only arm evidence), so this tool does not pretend to look for it.
    c. frontier chain is the exact prefix of STAGE_ORDER and the stopped stage is the next stage;
       layer/position/world/rank bookkeeping is consistent across ranks.
    d. geometry: max_positions, rope_table_bytes, slab_bytes present as positive integers,
       rope_table_bytes == 512 * max_positions (two rope table classes x (cos, sin) x 32 f32 pairs
       at rope_head_dim 64, which is what the writer's 2 * rope_table_span_bytes computes), and the
       values agree across ranks. kv_cache_bytes/window_ring_bytes are not in the landed document;
       if a future document carries them they must form a pair and fit in slab_bytes.
    e. the refused note carries the not-host-validated wording. The writer's refused note is
       "the device arm stopped at the stage named in frontier; no device value in this row is
       validated against the host reference". Matched case-insensitively on needle substrings
       (--note-needle), because the wording is free text.
    A coherent probe is still NOT a measurement: no D54 arm, no ms/step, no numerical result. The
    verdict line says so.

COMPLETE MODE (--require-complete) - a later stage's completed layer
    status == "completed" on all four, no "message" key, and a complete frontier: all 19 stages
    exactly in STAGE_ORDER order and no "; stopped at". Geometry as above. THEN, and only then, the
    row is ELIGIBLE for the D54 numerical gate, so the tool verifies the gate's inputs actually
    exist: per rank a tp4-token-r{rank}.json (tokens.positions/tokens.prompt_len/tokens.ids,
    positions_run, logits_dump.rows/.vocab/.bin) and a logits bin of at least rows*vocab*4 bytes,
    resolved the way scripts/numerical-gate.sh resolves it (recorded path, else the
    tp4-logits-r{rank}.jsonl.bin beside the token document). It computes the fixture key exactly
    as the gate does and names the next command with the frozen reference path
    docs/build/reference/speed-gate/<fixture>/rank{0..3}.json. A missing gate input is exit 2.
    The tolerance-arm declaration is NOT machine-checkable from this document (it lives in the run
    recipe today): the tool prints that as a warning and does not invent a schema. If a future
    document carries tolerance_arm and declared_utc plus a row start timestamp, the ordering
    (declared before the row) is checked.

EXIT CODES (the rule that matters: 2 is a STOP, never a retry, never a pass)
    0  coherent: a real frontier probe (default), or a gate-eligible completed layer
    1  incoherent: a finding (rank disagreement, lost chain, host arm, wrong status, bad geometry,
       missing not-host-validated wording, incomplete completed frontier, unusable gate input)
    2  cannot testify: a rank document or required gate input is missing, unreadable, empty, not
       JSON, or not a JSON object. Nothing is inferred and nothing is passed.

WHAT IT DOES NOT DO
    It never contacts a node and never runs the numerical gate; it reads files in the row directory
    only. It does not validate a device number, a token id or a logit. A coherent probe row is a
    probe, not a measurement; a coherent completed row is eligible for the gate, not a pass of it.

USAGE
    python3 scripts/device-frontier.py --row-dir <row> [--json-out verdict.json]
    python3 scripts/device-frontier.py --row-dir <row> --require-complete
    python3 scripts/device-frontier.py --selftest
    python3 scripts/device-frontier.py --write-fixtures <dir>

No third-party packages; stdlib only. Tests: scripts/tests/test_device_frontier.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone

# --------------------------------------------------------------------------------------------------
# Contract constants. If the writer changes, this block is the only code that has to change.
# --------------------------------------------------------------------------------------------------

SCHEMA = "sparkflash-tp4-layer-device-frontier/1"
DOC_TEMPLATE = "tp4-layer-device-r{rank}.json"
RANK_DIR_TEMPLATE = "rank{rank}"
TOKEN_DOC_TEMPLATE = "tp4-token-r{rank}.json"
LOGITS_BESIDE_TEMPLATE = "tp4-logits-r{rank}.jsonl.bin"
DEFAULT_RANKS = (0, 1, 2, 3)
DEFAULT_WORLD = 4

STATUS_REFUSED = "refused"
STATUS_COMPLETED = "completed"

#: layer_arm.why() for LayerArm::Device, verbatim from layer_device.rs. The arm string is the only
#: arm evidence in the document: SPARKFLASH_LAYER_DEVICE is not recorded there.
DEVICE_ARM = "device layer (opt-in): no per-layer PCIe copy, no per-stage observation read"

#: STAGE_CONTRACTS order from layer_device.rs (19 stages). The chain must be a prefix of this, and
#: a refused chain must stop at the next stage; every length check below derives from len(STAGE_ORDER),
#: so the next amendment is one line here.
STAGE_ORDER = (
    "engram_write",
    "mhc_entry",
    "quantise_prologue",
    "attention_projections",
    "query_path",
    "kv_path",
    "reference_seam",
    "stage_attention",
    "attention_body",
    "attention_output",
    "all_reduce",
    "attention_residual",
    "ffn_seams",
    "moe_router",
    "moe_routed",
    "all_reduce_moe",
    "all_reduce_rdma_arm",
    "moe_shared",
    "ffn_epilogue",
)

#: Two rope table classes (plain, compressed), each (cos, sin), 32 f32 pairs per position at
#: rope_head_dim 64: 2 * 2 * 32 * 4 = 512 bytes per position. The writer computes
#: 2 * rope_table_span_bytes(max_positions), which is the same number.
ROPE_BYTES_PER_POSITION = 512

#: The frozen references the D54 numerical gate runs against (fixture key -> rank{0..3}.json).
REFERENCE_ROOT = os.path.join("docs", "build", "reference", "speed-gate")
NUMERICAL_GATE = os.path.join("scripts", "numerical-gate.sh")

#: The refused note's not-host-validated wording. Free text, so match needles, not equality.
NOTE_NEEDLES = (
    "validated against the host reference",
    "not host-validated",
    "not-host-validated",
    "not host validated",
)

EXIT_COHERENT = 0
EXIT_INCOHERENT = 1
EXIT_CANNOT_TESTIFY = 2

STATUS_PASS = "PASS"
STATUS_FINDING = "FINDING"
STATUS_STOP = "STOP"


class CannotTestify(Exception):
    """A rank document or gate input is missing/unreadable: exit 2, never a pass."""


class FrontierParseError(ValueError):
    """The frontier string does not match the writer's grammar."""


# --------------------------------------------------------------------------------------------------
# Frontier grammar
# --------------------------------------------------------------------------------------------------

STOPPED_MARKER = "; stopped at "
CHAIN_HEAD = "device-only through "
CHAIN_EMPTY = "no stage ran"
CHAIN_SUFFIX = re.compile(r"^(?P<chain>.+) \((?P<count>[0-9]+) stage\(s\)\)$")


def parse_frontier(text: str) -> dict:
    """Split the frontier into (chain, stopped_stage, stop_message).

    Raises FrontierParseError when the string cannot be the writer's summary; the caller records
    that as a finding, not as a pass.
    """
    if not isinstance(text, str) or not text.strip():
        raise FrontierParseError("frontier is missing or empty")
    head = text
    stopped = None
    stop_message = None
    if STOPPED_MARKER in text:
        head, rest = text.split(STOPPED_MARKER, 1)
        stage, separator, message = rest.partition(": ")
        if not separator:
            raise FrontierParseError("stopped-at part has no ': ' before the message")
        stopped = stage.strip()
        stop_message = message.strip()
    if head == CHAIN_EMPTY:
        chain = []
        declared = 0
    elif head.startswith(CHAIN_HEAD):
        match = CHAIN_SUFFIX.match(head[len(CHAIN_HEAD):])
        if not match:
            raise FrontierParseError("chain part is not '... -> stage (N stage(s))'")
        chain = [stage.strip() for stage in match.group("chain").split(" -> ")]
        declared = int(match.group("count"))
        if declared != len(chain):
            raise FrontierParseError(f"declares {declared} stages but names {len(chain)}")
    else:
        raise FrontierParseError("frontier does not start with the chain summary")
    return {"chain": chain, "stopped": stopped, "stop_message": stop_message}


def stopped_expected(chain_length: int):
    """The stage the arm must have stopped at after this chain, or None when the chain is whole."""
    if chain_length == len(STAGE_ORDER):
        return None
    return STAGE_ORDER[chain_length]


def render_frontier(chain, stopped, stop_message) -> str:
    """Build a writer-grammar frontier string (used by the synthetic fixtures and --selftest)."""
    if chain:
        head = "{}{} ({} stage(s))".format(CHAIN_HEAD, " -> ".join(chain), len(chain))
    else:
        head = CHAIN_EMPTY
    if stopped is not None:
        head += "{}{}: {}".format(STOPPED_MARKER, stopped, stop_message or "synthetic refusal text")
    return head


# --------------------------------------------------------------------------------------------------
# Synthetic fixtures (exposed so the tests and --selftest use one builder, never a hand-written blob)
# --------------------------------------------------------------------------------------------------

REFUSED_NOTE = (
    "the device arm stopped at the stage named in frontier; no device value in this row is "
    "validated against the host reference"
)
COMPLETED_NOTE = (
    "the whole device prefix ran for this layer; no device value is validated against the host "
    "reference until the cluster row"
)


def rank_document(
    rank: int,
    *,
    status: str = STATUS_REFUSED,
    chain=None,
    stopped=None,
    arm: str = DEVICE_ARM,
    note: str = None,
    layer: int = 0,
    position: int = 0,
    max_positions: int = 4096,
    slab_bytes: int = 64 << 20,
    world: int = DEFAULT_WORLD,
    schema: str = SCHEMA,
    rank_field: bool = True,
    drop_keys=(),
    extra: dict = None,
) -> dict:
    """One synthetic rank document in the writer's exact shape (synthetic: True marks it)."""
    if chain is None:
        if status == STATUS_COMPLETED:
            chain = list(STAGE_ORDER)
            stopped = None
        else:
            chain = list(STAGE_ORDER[:5])
            stopped = stopped if stopped is not None else STAGE_ORDER[5]
    document = {
        "synthetic": True,
        "schema": schema,
        "world": world,
        "arm": arm,
        "status": status,
        "layer": layer,
        "position": position,
        "frontier": render_frontier(chain, stopped, None),
        "max_positions": max_positions,
        "rope_table_bytes": ROPE_BYTES_PER_POSITION * max_positions,
        "slab_bytes": slab_bytes,
        "note": note if note is not None else (REFUSED_NOTE if status == STATUS_REFUSED else COMPLETED_NOTE),
    }
    if rank_field:
        document["rank"] = rank
    if status == STATUS_REFUSED:
        stage = stopped if stopped is not None else "unknown stage"
        document["message"] = (
            "layer device stage {}: synthetic refusal for the checker self-test".format(stage)
        )
    if status == STATUS_COMPLETED:
        document.pop("message", None)
    if drop_keys:
        for key in drop_keys:
            document.pop(key, None)
    if extra:
        document.update(extra)
    return document


def token_document(rank: int, bin_path: str, *, rows: int = 2, vocab: int = 16) -> dict:
    """One synthetic tp4-token document with the fields the numerical gate reads."""
    return {
        "synthetic": True,
        "schema": "sparkflash-tp4-token/1",
        "rank": rank,
        "positions_run": rows,
        "tokens": {"positions": [0, 1][:rows], "prompt_len": 3, "ids": [7, 8, 9][:rows]},
        "logits_dump": {"bin": os.path.abspath(bin_path), "rows": rows, "vocab": vocab},
    }


def write_row(root: str, documents: dict, token_documents: dict = None, *, layout: str = "rank-dirs") -> list:
    """Write synthetic rank documents (and optional token/logits files) and return written paths."""
    os.makedirs(root, exist_ok=True)
    written = []
    for rank, document in sorted(documents.items()):
        if layout == "rank-dirs":
            directory = os.path.join(root, RANK_DIR_TEMPLATE.format(rank=rank))
            os.makedirs(directory, exist_ok=True)
        else:
            directory = root
        path = os.path.join(directory, DOC_TEMPLATE.format(rank=rank))
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, sort_keys=True)
            handle.write("\n")
        written.append(path)
    for rank, token_document_body in sorted((token_documents or {}).items()):
        directory = os.path.join(root, RANK_DIR_TEMPLATE.format(rank=rank))
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, TOKEN_DOC_TEMPLATE.format(rank=rank))
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(token_document_body, handle, indent=2, sort_keys=True)
            handle.write("\n")
        written.append(path)
        bin_path = token_document_body["logits_dump"]["bin"]
        rows = token_document_body["logits_dump"]["rows"]
        vocab = token_document_body["logits_dump"]["vocab"]
        # The document may record the bin relative to the row directory (portable fixtures); the
        # bytes always land inside this row, never next to the generator's working directory.
        write_path = bin_path if os.path.isabs(bin_path) else os.path.join(root, bin_path)
        os.makedirs(os.path.dirname(write_path), exist_ok=True)
        with open(write_path, "wb") as handle:
            handle.write(b"\x00" * (rows * vocab * 4))
        written.append(write_path)
    return written


def synthetic_cases() -> list:
    """The self-test cases: (name, mode, expected_exit, documents, token_documents).

    Every case is synthetic and marked synthetic: True in each document. These are not rows.
    """
    def docs_for(**overrides):
        return {rank: rank_document(rank, **overrides) for rank in DEFAULT_RANKS}

    cases = []
    # 0. A coherent probe: five stages ran, the sixth (kv_path) refused. This is the real shape.
    cases.append(
        {
            "name": "coherent-6-stage-kv_path",
            "mode": "probe",
            "expected": EXIT_COHERENT,
            "documents": docs_for(chain=list(STAGE_ORDER[:5]), stopped="kv_path"),
        }
    )
    # 1a. The next expected frontier once the merge settles: engram_write is a no-op on layer 0,
    # the stages through kv_path have bodies, and reference_seam still refuses. reference_seam is
    # stage index 6, so the coherent chain is 6 stages (a literal "18-stage chain stopped at
    # reference_seam" would violate the unchanged index rule and is a finding, not this state).
    cases.append(
        {
            "name": "coherent-refusal-at-reference_seam",
            "mode": "probe",
            "expected": EXIT_COHERENT,
            "documents": docs_for(chain=list(STAGE_ORDER[:6]), stopped="reference_seam"),
        }
    )
    # 1a-neg. The literal reading of "an 18-stage chain stopping at reference_seam": chain length
    # 18 would put the next stage at ffn_epilogue, not reference_seam, so the tool reports a finding
    # under the unchanged index rule. Kept so that claim is checkable instead of argued: if such a
    # document ever appears, the merge or the frontier is wrong, and exit 1 is the correct answer.
    cases.append(
        {
            "name": "incoherent-18-stage-chain-at-reference_seam",
            "mode": "probe",
            "expected": EXIT_INCOHERENT,
            "documents": docs_for(chain=list(STAGE_ORDER[:18]), stopped="reference_seam"),
        }
    )
    # 1b. The state right after the moe_shared amendment lands and before its body is written: the
    # stages before moe_shared ran and the arm refused at moe_shared. moe_shared is the 18th stage
    # (index 17), so "chain length == index of the stopped stage" makes this a 17-stage chain. This
    # case would be a wrong-next-stage finding under a stale 18-stage order.
    cases.append(
        {
            "name": "coherent-refusal-at-moe_shared",
            "mode": "probe",
            "expected": EXIT_COHERENT,
            "documents": docs_for(chain=list(STAGE_ORDER[:17]), stopped="moe_shared"),
        }
    )
    # 1c. The boundary off-by-one a hand-written frontier would carry: an 18-stage chain stopping
    # at moe_shared. Chain length 18 puts the next stage at ffn_epilogue, so this is a finding, not
    # the post-amendment state. The tool must refuse to read the 18th stage as "ran" here.
    cases.append(
        {
            "name": "off-by-one-18-stage-chain-at-moe_shared",
            "mode": "probe",
            "expected": EXIT_INCOHERENT,
            "documents": docs_for(chain=list(STAGE_ORDER[:18]), stopped="moe_shared"),
        }
    )
    # 2. The known defect: the refused document lost the chain but still says kv_path stopped it.
    cases.append(
        {
            "name": "degenerate-no-stage-ran-kv_path",
            "mode": "probe",
            "expected": EXIT_INCOHERENT,
            "documents": docs_for(chain=[], stopped="kv_path"),
        }
    )
    # 2. One rank stopped somewhere else: a rank disagreement is a finding.
    documents = docs_for(chain=list(STAGE_ORDER[:5]), stopped="kv_path")
    documents[2] = rank_document(2, chain=list(STAGE_ORDER[:3]), stopped="query_path")
    cases.append(
        {
            "name": "rank-different-chain",
            "mode": "probe",
            "expected": EXIT_INCOHERENT,
            "documents": documents,
        }
    )
    # 3. A host-arm document masquerading as a probe.
    cases.append(
        {
            "name": "host-arm-document",
            "mode": "probe",
            "expected": EXIT_INCOHERENT,
            "documents": docs_for(
                chain=list(STAGE_ORDER[:5]),
                stopped="kv_path",
                arm="host arm (evidence recording forces it: the device arm has no per-position readback)",
            ),
        }
    )
    # 4. A rank document missing: cannot testify.
    documents = docs_for(chain=list(STAGE_ORDER[:5]), stopped="kv_path")
    documents.pop(3)
    cases.append(
        {
            "name": "missing-rank-document",
            "mode": "probe",
            "expected": EXIT_CANNOT_TESTIFY,
            "documents": documents,
        }
    )
    # 5. An empty rank document: a file that exists but cannot testify is a stop, not a pass.
    cases.append(
        {
            "name": "empty-rank-document",
            "mode": "probe",
            "expected": EXIT_CANNOT_TESTIFY,
            "documents": docs_for(chain=list(STAGE_ORDER[:5]), stopped="kv_path"),
            "corrupt": {"rank": 1, "content": ""},
        }
    )
    # 6. Geometry that does not follow the rope formula.
    cases.append(
        {
            "name": "bad-rope-geometry",
            "mode": "probe",
            "expected": EXIT_INCOHERENT,
            "documents": docs_for(
                chain=list(STAGE_ORDER[:5]),
                stopped="kv_path",
                extra={"rope_table_bytes": 12345},
            ),
        }
    )
    # 7. A refused note without the not-host-validated wording.
    cases.append(
        {
            "name": "note-without-not-host-validated",
            "mode": "probe",
            "expected": EXIT_INCOHERENT,
            "documents": docs_for(
                chain=list(STAGE_ORDER[:5]),
                stopped="kv_path",
                note="a probe row, values fine",
            ),
        }
    )
    # 9. A completed layer with the full 19-stage frontier and the numerical gate's inputs.
    token_documents = {}
    tokens_by_rank = {}
    for rank in DEFAULT_RANKS:
        bin_path = os.path.join("<ROW>", RANK_DIR_TEMPLATE.format(rank=rank), LOGITS_BESIDE_TEMPLATE.format(rank=rank))
        token_documents[rank] = token_document(rank, bin_path)
        tokens_by_rank[rank] = token_documents[rank]
    cases.append(
        {
            "name": "complete-19-stage",
            "mode": "complete",
            "expected": EXIT_COHERENT,
            "documents": {rank: rank_document(rank, status=STATUS_COMPLETED) for rank in DEFAULT_RANKS},
            "token_documents": tokens_by_rank,
        }
    )
    # 9. --require-complete on a refused row: present documents, wrong state, exit 1.
    cases.append(
        {
            "name": "require-complete-on-refused-row",
            "mode": "complete",
            "expected": EXIT_INCOHERENT,
            "documents": docs_for(chain=list(STAGE_ORDER[:5]), stopped="kv_path"),
        }
    )
    # 10. A completed frontier that stops before ffn_epilogue: incomplete, exit 1.
    cases.append(
        {
            "name": "completed-frontier-incomplete",
            "mode": "complete",
            "expected": EXIT_INCOHERENT,
            "documents": {
                rank: rank_document(rank, status=STATUS_COMPLETED, chain=list(STAGE_ORDER[:18]), stopped=None)
                for rank in DEFAULT_RANKS
            },
        }
    )
    # 11. Completed documents whose logits bins are missing: the gate cannot run, exit 2.
    cases.append(
        {
            "name": "complete-without-logits-bins",
            "mode": "complete",
            "expected": EXIT_CANNOT_TESTIFY,
            "documents": {rank: rank_document(rank, status=STATUS_COMPLETED) for rank in DEFAULT_RANKS},
        }
    )
    return cases


def write_fixture_set(root: str) -> list:
    """Materialise every synthetic case under root, one directory per case, plus a manifest."""
    manifest = []
    for case in synthetic_cases():
        case_dir = os.path.join(root, case["name"])
        documents = case["documents"]
        token_documents = None
        if case.get("token_documents"):
            token_documents = {}
            for rank, body in case["token_documents"].items():
                patched = json.loads(json.dumps(body))
                # Relative to the row directory (the checker's beside-the-token fallback), so the
                # generated fixture tree is portable and can be committed under scripts/tests/fixtures/.
                patched["logits_dump"]["bin"] = os.path.join(
                    RANK_DIR_TEMPLATE.format(rank=rank), LOGITS_BESIDE_TEMPLATE.format(rank=rank)
                )
                token_documents[rank] = patched
        write_row(
            case_dir,
            documents,
            token_documents,
            layout="flat" if case["name"] == "coherent-6-stage-kv_path" else "rank-dirs",
        )
        corruption = case.get("corrupt")
        if corruption:
            path = os.path.join(
                case_dir, RANK_DIR_TEMPLATE.format(rank=corruption["rank"]), DOC_TEMPLATE.format(rank=corruption["rank"])
            )
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(corruption["content"])
        manifest.append(
            {
                "case": case["name"],
                "mode": case["mode"],
                "expected_exit": case["expected"],
                "row_dir": os.path.abspath(case_dir),
                "synthetic": True,
            }
        )
    with open(os.path.join(root, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump({"synthetic": True, "cases": manifest}, handle, indent=2)
        handle.write("\n")
    return manifest


# --------------------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------------------

def load_document(path: str, label: str) -> dict:
    """Read one JSON object. Missing/empty/unparseable/non-object is CannotTestify (exit 2)."""
    if not os.path.isfile(path):
        raise CannotTestify("{} document missing: {}".format(label, os.path.abspath(path)))
    try:
        size = os.path.getsize(path)
    except OSError as error:
        raise CannotTestify("{} document unreadable: {} ({})".format(label, path, error)) from error
    if size == 0:
        raise CannotTestify("{} document is empty (0 bytes): {}".format(label, os.path.abspath(path)))
    try:
        with open(path, "r", encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise CannotTestify("{} document is not JSON: {} ({})".format(label, os.path.abspath(path), error)) from error
    if not isinstance(document, dict):
        raise CannotTestify("{} document is not a JSON object: {}".format(label, os.path.abspath(path)))
    return document


def locate_documents(row_dir: str, template: str, ranks) -> tuple:
    """Find one file per rank, flat or under rank{rank}/. Ambiguity is a stop, never a guess."""
    found = {}
    missing = []
    for rank in ranks:
        name = template.format(rank=rank)
        candidates = [
            os.path.join(row_dir, name),
            os.path.join(row_dir, RANK_DIR_TEMPLATE.format(rank=rank), name),
        ]
        existing = [path for path in candidates if os.path.isfile(path)]
        if not existing:
            missing.append({"rank": rank, "expected": [os.path.abspath(path) for path in candidates]})
        elif len(existing) > 1:
            raise CannotTestify(
                "rank {}: two possible documents, refusing to choose: {}".format(
                    rank, ", ".join(os.path.abspath(path) for path in existing)
                )
            )
        else:
            found[rank] = existing[0]
    return found, missing


def is_positive_int(value) -> bool:
    """True for a real JSON integer greater than zero (bool is not an integer here)."""
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def geometry_of(document: dict) -> dict:
    """The geometry fields, preferring a nested geometry block when one exists."""
    block = document.get("geometry")
    scope = block if isinstance(block, dict) else document
    keys = ("max_positions", "rope_table_bytes", "slab_bytes", "kv_cache_bytes", "window_ring_bytes")
    return {key: scope.get(key) for key in keys if key in scope}


def check_geometry(documents: dict, record: dict):
    """Record geometry checks. Returns the geometry dict of rank 0 for reporting."""
    first = {}
    for rank, document in documents.items():
        geometry = geometry_of(document)
        for key in ("max_positions", "rope_table_bytes", "slab_bytes"):
            value = geometry.get(key)
            if not is_positive_int(value):
                record.finding(
                    "geometry",
                    "rank {}: {} is not a positive integer ({!r})".format(rank, key, value),
                )
                continue
            first.setdefault(key, value)
        max_positions = geometry.get("max_positions")
        rope_bytes = geometry.get("rope_table_bytes")
        if is_positive_int(max_positions) and is_positive_int(rope_bytes):
            expected = ROPE_BYTES_PER_POSITION * max_positions
            if rope_bytes != expected:
                record.finding(
                    "geometry",
                    "rank {}: rope_table_bytes {} != {} * max_positions {} = {} (rope_head_dim 64)".format(
                        rank, rope_bytes, ROPE_BYTES_PER_POSITION, max_positions, expected
                    ),
                )
        kv = geometry.get("kv_cache_bytes")
        window = geometry.get("window_ring_bytes")
        if kv is not None or window is not None:
            if not (isinstance(kv, int) and isinstance(window, int)):
                record.finding(
                    "geometry",
                    "rank {}: kv_cache_bytes/window_ring_bytes must appear as an integer pair".format(rank),
                )
            else:
                slab = geometry.get("slab_bytes")
                if is_positive_int(slab) and kv + window > slab:
                    record.finding(
                        "geometry",
                        "rank {}: kv_cache_bytes {} + window_ring_bytes {} > slab_bytes {}".format(
                            rank, kv, window, slab
                        ),
                    )
    # Geometry must agree across ranks: four ranks of one row run the same configuration.
    for rank in documents:
        geometry = geometry_of(documents[rank])
        for key in ("max_positions", "rope_table_bytes", "slab_bytes", "kv_cache_bytes", "window_ring_bytes"):
            if key in geometry and key in first and geometry[key] != first[key]:
                record.finding(
                    "geometry-rank-agreement",
                    "rank {}: {}={} differs from rank {}: {}".format(
                        rank, key, geometry[key], min(documents), first[key]
                    ),
                )
            elif key in geometry:
                first.setdefault(key, geometry[key])
    if not any(check["status"] == STATUS_FINDING for check in record.checks if check["id"] == "geometry"):
        record.passing(
            "geometry",
            "max_positions={} rope_table_bytes={} slab_bytes={} (rope = {} * max_positions), equal across ranks".format(
                first.get("max_positions"),
                first.get("rope_table_bytes"),
                first.get("slab_bytes"),
                ROPE_BYTES_PER_POSITION,
            ),
        )
    return first


def note_has_needle(note, needles) -> bool:
    """Case-insensitive substring match against the not-host-validated wording."""
    if not isinstance(note, str):
        return False
    lowered = note.lower()
    return any(needle.lower() in lowered for needle in needles)


def check_frontier(documents: dict, record: dict, *, require_complete: bool):
    """Parse and validate the frontier of every rank. Returns the parsed frontier of rank min."""
    parsed_by_rank = {}
    for rank, document in documents.items():
        frontier = document.get("frontier")
        try:
            parsed = parse_frontier(frontier)
        except FrontierParseError as error:
            record.finding("frontier-grammar", "rank {}: {}".format(rank, error))
            continue
        parsed_by_rank[rank] = parsed
        if require_complete:
            if parsed["stopped"] is not None:
                record.finding(
                    "frontier-complete",
                    "rank {}: completed layer frontier names a stopped stage {!r}".format(
                        rank, parsed["stopped"]
                    ),
                )
            elif parsed["chain"] != list(STAGE_ORDER):
                record.finding(
                    "frontier-complete",
                    "rank {}: chain has {} stage(s), not the full {} in STAGE_CONTRACTS order "
                    "({})".format(rank, len(parsed["chain"]), len(STAGE_ORDER), " -> ".join(parsed["chain"])),
                )
        else:
            if parsed["stopped"] is None:
                record.finding(
                    "frontier-refused",
                    "rank {}: refused frontier has no '; stopped at <stage>' part".format(rank),
                )
                continue
            chain = parsed["chain"]
            if chain != list(STAGE_ORDER[: len(chain)]):
                first_bad = next(
                    (index for index, stage in enumerate(chain) if stage != STAGE_ORDER[index]),
                    None,
                )
                record.finding(
                    "frontier-prefix",
                    "rank {}: chain is not a prefix of STAGE_CONTRACTS; first difference at index "
                    "{} ({!r} != {!r})".format(
                        rank,
                        first_bad,
                        chain[first_bad] if first_bad is not None else None,
                        STAGE_ORDER[first_bad] if first_bad is not None else None,
                    ),
                )
                continue
            expected = stopped_expected(len(chain))
            if expected is None:
                record.finding(
                    "frontier-stopped",
                    "rank {}: frontier reports a stop at {!r} after all {} stages ran".format(
                        rank, parsed["stopped"], len(STAGE_ORDER)
                    ),
                )
            elif parsed["stopped"] != expected:
                detail = (
                    "rank {}: frontier says the chain is {!r} and it stopped at {!r}, but the next "
                    "stage after {} ran stage(s) is {!r}".format(
                        rank,
                        "no stage ran" if not chain else " -> ".join(chain),
                        parsed["stopped"],
                        len(chain),
                        expected,
                    )
                )
                if not chain:
                    detail += (
                        ". This is the known lost-chain defect: StageRuns::from_refusal keeps only "
                        "the refused stage, so a row that ran any stages reports 'no stage ran'. Exit "
                        "1 until the real runs are recorded (gemv's fix); do not soften it."
                    )
                record.finding("frontier-stopped", detail)
    if parsed_by_rank:
        first_rank = min(parsed_by_rank)
        record.passing(
            "frontier-grammar",
            "rank {}: {}".format(first_rank, parsed_by_rank[first_rank] and documents[first_rank].get("frontier")),
        )
    return parsed_by_rank


class Record:
    """Check accumulator: PASS / FINDING / STOP lines and the two verdicts."""

    def __init__(self, mode: str, row_dir: str, ranks):
        self.mode = mode
        self.row_dir = os.path.abspath(row_dir)
        self.ranks = list(ranks)
        self.checks = []
        self.warnings = []
        self.findings = []
        self.stop_reason = None

    def passing(self, check_id: str, detail: str):
        self.checks.append({"id": check_id, "status": STATUS_PASS, "detail": detail})

    def finding(self, check_id: str, detail: str):
        self.checks.append({"id": check_id, "status": STATUS_FINDING, "detail": detail})
        self.findings.append(detail)

    def warning(self, detail: str):
        self.warnings.append(detail)

    def stop(self, check_id: str, detail: str):
        self.checks.append({"id": check_id, "status": STATUS_STOP, "detail": detail})
        if self.stop_reason is None:
            self.stop_reason = detail

    def result(self, extra: dict = None) -> dict:
        if self.stop_reason is not None:
            verdict, code = "CANNOT-TESTIFY", EXIT_CANNOT_TESTIFY
        elif self.findings:
            verdict, code = "INCOHERENT", EXIT_INCOHERENT
        else:
            verdict = "COHERENT-PROBE" if self.mode == "probe" else "COHERENT-COMPLETE-LAYER"
            code = EXIT_COHERENT
        payload = {
            "schema": "sparkflash-device-frontier-check/1",
            "mode": self.mode,
            "row_dir": self.row_dir,
            "ranks": self.ranks,
            "checks": self.checks,
            "findings": self.findings,
            "warnings": self.warnings,
            "stop_reason": self.stop_reason,
            "verdict": verdict,
            "exit_code": code,
            "not_a_measurement": self.mode == "probe",
        }
        if extra:
            payload.update(extra)
        return payload


def evaluate(
    row_dir: str,
    *,
    ranks=DEFAULT_RANKS,
    require_complete: bool = False,
    note_needles=None,
    expect_schema: str = SCHEMA,
    reference_root: str = None,
) -> dict:
    """Evaluate a row directory and return the verdict document (does not print, does not raise)."""
    mode = "complete" if require_complete else "probe"
    record = Record(mode, row_dir, ranks)
    note_needles = tuple(note_needles or NOTE_NEEDLES)
    reference_root = reference_root or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), REFERENCE_ROOT
    )
    if not os.path.isdir(row_dir):
        record.stop("row-dir", "row directory does not exist: {}".format(os.path.abspath(row_dir)))
        return record.result()
    try:
        paths, missing = locate_documents(row_dir, DOC_TEMPLATE, ranks)
        if missing:
            detail = "; ".join(
                "rank {}: expected one of {}".format(item["rank"], ", ".join(item["expected"]))
                for item in missing
            )
            record.stop("rank-documents-present", "rank document(s) missing: " + detail)
            return record.result({"paths": paths, "missing": missing})
        record.passing(
            "rank-documents-present",
            "{}/{} documents found ({})".format(
                len(paths), len(ranks), ", ".join("rank{}: {}".format(rank, path) for rank, path in sorted(paths.items()))
            ),
        )
        documents = {}
        for rank, path in sorted(paths.items()):
            documents[rank] = load_document(path, "rank {}".format(rank))
    except CannotTestify as error:
        record.stop("documents-readable", str(error))
        return record.result()

    # (b) arm, (a) schema, rank-world bookkeeping.
    for rank, document in documents.items():
        schema = document.get("schema")
        if schema != expect_schema:
            record.finding(
                "schema",
                "rank {}: schema {!r} != {!r}".format(rank, schema, expect_schema),
            )
        arm = document.get("arm")
        if arm != DEVICE_ARM:
            record.finding(
                "arm-device",
                "rank {}: arm {!r} is not the device arm string; a host-arm document cannot be a "
                "device probe (expected {!r})".format(rank, arm, DEVICE_ARM),
            )
        if "rank" in document and document.get("rank") != rank:
            record.finding(
                "rank-field",
                "rank {}: document says rank {!r}".format(rank, document.get("rank")),
            )
        world = document.get("world")
        if isinstance(world, int) and not isinstance(world, bool) and world != len(ranks):
            record.finding(
                "world",
                "rank {}: world {} != {} ranks under test".format(rank, world, len(ranks)),
            )
        status = document.get("status")
        expected_status = STATUS_COMPLETED if require_complete else STATUS_REFUSED
        if status != expected_status:
            hint = ""
            if not require_complete and status == STATUS_COMPLETED:
                hint = "; this row claims a completed layer, so run with --require-complete"
            if require_complete and status == STATUS_REFUSED:
                hint = "; a refused row is a frontier probe, not a completed layer"
            record.finding(
                "status",
                "rank {}: status {!r} != {!r}{}".format(rank, status, expected_status, hint),
            )
    if not any(check["id"] == "schema" and check["status"] == STATUS_FINDING for check in record.checks):
        record.passing("schema", "all ranks carry schema {!r}".format(expect_schema))
    if not any(check["id"] == "arm-device" and check["status"] == STATUS_FINDING for check in record.checks):
        record.passing("arm-device", "all ranks carry the device arm string")
    if not require_complete and not any(
        check["id"] == "status" and check["status"] == STATUS_FINDING for check in record.checks
    ):
        record.passing("status", "all ranks are status 'refused' (a frontier probe row)")

    # Cross-rank bookkeeping: same frontier, layer, position.
    frontiers = {rank: document.get("frontier") for rank, document in documents.items()}
    distinct_frontiers = {value for value in frontiers.values()}
    if len(distinct_frontiers) > 1:
        record.finding(
            "frontier-rank-agreement",
            "ranks disagree on the frontier (a real finding: stage order or refusal path differs "
            "per rank): " + json.dumps({str(rank): value for rank, value in sorted(frontiers.items())}),
        )
    else:
        record.passing("frontier-rank-agreement", "all ranks carry the same frontier string")
    for key in ("layer", "position"):
        values = {rank: document.get(key) for rank, document in documents.items()}
        if len(set(values.values())) > 1:
            record.finding(
                "bookkeeping",
                "ranks disagree on {!r}: {}".format(key, json.dumps({str(rank): value for rank, value in sorted(values.items())})),
            )
    if not any(check["id"] == "bookkeeping" for check in record.checks):
        record.passing(
            "bookkeeping",
            "layer={} position={} on all ranks".format(
                documents[min(documents)].get("layer"), documents[min(documents)].get("position")
            ),
        )

    # (c) frontier grammar and stage order.
    check_frontier(documents, record, require_complete=require_complete)
    if not require_complete:
        # (e) the refused note's not-host-validated wording.
        bad_notes = [
            rank
            for rank, document in documents.items()
            if not note_has_needle(document.get("note"), note_needles)
        ]
        if bad_notes:
            record.finding(
                "note-not-host-validated",
                "rank(s) {}: refused note does not carry the not-host-validated wording; needles "
                "{}".format(bad_notes, list(note_needles)),
            )
        else:
            record.passing("note-not-host-validated", "every rank's refused note states the values are not host-validated")

    # (d) geometry.
    geometry = check_geometry(documents, record)

    # Message key: refused requires it; completed must not carry it.
    for rank, document in documents.items():
        has_message = "message" in document
        if require_complete and has_message:
            record.finding(
                "message-absent",
                "rank {}: completed document still carries a 'message' key".format(rank),
            )
        if not require_complete and not has_message:
            record.finding(
                "message-present",
                "rank {}: refused document has no 'message' (the refusal text) ".format(rank),
            )
    if require_complete and not any(
        check["id"] == "message-absent" and check["status"] == STATUS_FINDING for check in record.checks
    ):
        record.passing("message-absent", "completed documents carry no 'message' key")

    extra = {
        "paths": {str(rank): path for rank, path in sorted(paths.items())},
        "rank_report": {
            str(rank): {
                "path": path,
                "status": documents[rank].get("status"),
                "arm": documents[rank].get("arm"),
                "frontier": documents[rank].get("frontier"),
                "layer": documents[rank].get("layer"),
                "position": documents[rank].get("position"),
            }
            for rank, path in sorted(paths.items())
        },
    }
    if require_complete:
        _check_tolerance_arm(documents, record)
        if record.findings:
            record.warning(
                "D54 numerical-gate readiness not evaluated: the row already has {} finding(s), "
                "so it is not a completed layer to certify".format(len(record.findings))
            )
        else:
            extra.update(_complete_gate_readiness(row_dir, ranks, record, reference_root, documents))
        if record.stop_reason is None and not record.findings:
            reference = extra.get("reference")
            next_step = "python3 {} --run {}".format(NUMERICAL_GATE, os.path.abspath(row_dir))
            if reference and reference.get("present"):
                next_step += "   (frozen reference: {})".format(reference.get("directory"))
            else:
                next_step += "   (frozen reference NOT present in this tree; expected docs/build/reference/speed-gate/<fixture>/rank{0..3}.json)"
            if not any("arm declaration" in warning for warning in record.warnings):
                record.warning(
                    "arm declaration: not machine-checkable from this document (it lives in the run "
                    "recipe today); add tolerance_arm + declared_utc to the frontier document to make "
                    "this a real check"
                )
            extra["next_step"] = next_step
    extra["geometry"] = geometry
    return record.result(extra)


def parse_utc(text):
    """Parse an ISO-8601 UTC timestamp; None when it is not one."""
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _check_tolerance_arm(documents: dict, record: Record):
    """Forward-compatible check for a tolerance_arm/declared_utc field the writer may add.

    Today the declaration lives in the run recipe, so absence is a warning, not a finding. When
    the field does appear, it must agree across ranks and must have been declared before the row
    started (declared_utc <= row start). Present-but-contradictory evidence is a finding.
    """
    arms = {rank: document.get("tolerance_arm") for rank, document in documents.items()}
    if all(value is None for value in arms.values()):
        return
    if any(value is None for value in arms.values()):
        record.warning(
            "tolerance_arm is recorded on some ranks only: {}".format(
                json.dumps({str(rank): value for rank, value in sorted(arms.items())})
            )
        )
        return

    def arm_name(value):
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            return value.get("name")
        return None

    names = {rank: arm_name(value) for rank, value in arms.items()}
    if len(set(names.values())) > 1:
        record.finding(
            "tolerance-arm-rank-agreement",
            "ranks disagree on the tolerance arm: {}".format(
                json.dumps({str(rank): name for rank, name in sorted(names.items())})
            ),
        )
    declared = {}
    started = {}
    for rank, document in documents.items():
        value = arms[rank]
        declared[rank] = document.get("declared_utc") or (
            value.get("declared_utc") if isinstance(value, dict) else None
        )
        started[rank] = document.get("row_started_utc") or document.get("started_utc")
    declared_moments = {rank: parse_utc(text) for rank, text in declared.items()}
    started_moments = {rank: parse_utc(text) for rank, text in started.items()}
    if any(moment is None for moment in declared_moments.values()) or any(
        moment is None for moment in started_moments.values()
    ):
        record.warning(
            "tolerance_arm/timestamps incomplete: cannot prove the arm was declared before the row "
            "(declared={}, row start={})".format(
                json.dumps({str(rank): text for rank, text in sorted(declared.items())}),
                json.dumps({str(rank): text for rank, text in sorted(started.items())}),
            )
        )
        return
    late = [
        rank
        for rank in documents
        if declared_moments[rank] > started_moments[rank]
    ]
    if late:
        record.finding(
            "tolerance-arm-declared-before-row",
            "rank(s) {}: tolerance arm declared AFTER the row started (declared {} > started {})".format(
                late,
                declared_moments[late[0]].isoformat(),
                started_moments[late[0]].isoformat(),
            ),
        )
    else:
        record.passing(
            "tolerance-arm-declared-before-row",
            "tolerance arm {!r} declared {} before the row started {} on every rank".format(
                names[min(documents)],
                declared_moments[min(documents)].isoformat(),
                started_moments[min(documents)].isoformat(),
            ),
        )


def _complete_gate_readiness(row_dir: str, ranks, record: Record, reference_root: str, documents: dict) -> dict:
    """Check the D54 numerical gate's inputs exist; compute the fixture key; locate the reference.

    Every failure here is CannotTestify (exit 2): the gate cannot run, so the row cannot be called
    gate-eligible. Present-but-contradictory inputs are findings (exit 1).
    """
    fixture_keys = {}
    token_paths = {}
    try:
        token_paths, missing = locate_documents(row_dir, TOKEN_DOC_TEMPLATE, ranks)
        if missing:
            record.stop(
                "numerical-gate-inputs",
                "the D54 numerical gate needs per-rank token documents and logits bins; missing: "
                + "; ".join(
                    "rank {}: expected one of {}".format(item["rank"], ", ".join(item["expected"]))
                    for item in missing
                ),
            )
            return {"numerical_gate": {"available": False}}
    except CannotTestify as error:
        record.stop("numerical-gate-inputs", str(error))
        return {"numerical_gate": {"available": False}}

    for rank in ranks:
        try:
            token_document = load_document(token_paths[rank], "rank {} token".format(rank))
        except CannotTestify as error:
            record.stop("numerical-gate-inputs", str(error))
            return {"numerical_gate": {"available": False}}
        tokens = token_document.get("tokens")
        if not isinstance(tokens, dict):
            record.stop(
                "numerical-gate-inputs",
                "rank {}: token document has no tokens object (the gate's fixture key needs "
                "tokens.positions/prompt_len/ids)".format(rank),
            )
            return {"numerical_gate": {"available": False}}
        key_basis = {
            "positions": tokens.get("positions"),
            "prompt_len": tokens.get("prompt_len"),
            "ids": tokens.get("ids"),
        }
        if key_basis["positions"] is None or key_basis["prompt_len"] is None or key_basis["ids"] is None:
            record.stop(
                "numerical-gate-inputs",
                "rank {}: token document lacks tokens.positions/prompt_len/ids; the gate cannot "
                "compute a fixture key".format(rank),
            )
            return {"numerical_gate": {"available": False}}
        digest = hashlib.sha256(
            json.dumps(key_basis, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        fixture_keys[rank] = digest
        dump = token_document.get("logits_dump")
        if not isinstance(dump, dict):
            record.stop(
                "numerical-gate-inputs",
                "rank {}: token document has no logits_dump (rows/vocab/bin)".format(rank),
            )
            return {"numerical_gate": {"available": False}}
        rows, vocab = dump.get("rows"), dump.get("vocab")
        if not is_positive_int(rows) or not is_positive_int(vocab):
            record.stop(
                "numerical-gate-inputs",
                "rank {}: logits_dump.rows/vocab are not positive integers".format(rank),
            )
            return {"numerical_gate": {"available": False}}
        candidates = []
        recorded = dump.get("bin")
        if isinstance(recorded, str) and recorded:
            candidates.append(recorded)
        candidates.append(
            os.path.join(row_dir, RANK_DIR_TEMPLATE.format(rank=rank), LOGITS_BESIDE_TEMPLATE.format(rank=rank))
        )
        candidates.append(
            os.path.join(os.path.dirname(token_paths[rank]), LOGITS_BESIDE_TEMPLATE.format(rank=rank))
        )
        bin_path = next((path for path in candidates if os.path.isfile(path)), None)
        if bin_path is None:
            record.stop(
                "numerical-gate-inputs",
                "rank {}: logits bin missing; the gate would refuse. Expected one of: {}".format(
                    rank, ", ".join(os.path.abspath(path) for path in candidates)
                ),
            )
            return {"numerical_gate": {"available": False}}
        need = rows * vocab * 4
        size = os.path.getsize(bin_path)
        if size < need:
            record.finding(
                "numerical-gate-inputs",
                "rank {}: logits bin {} has {} bytes, needs rows*vocab*4 = {}; the gate would "
                "refuse it".format(rank, os.path.abspath(bin_path), size, need),
            )
        else:
            record.passing(
                "logits-bin-rank{}".format(rank),
                "{} bytes >= rows*vocab*4 = {} ({})".format(size, need, os.path.abspath(bin_path)),
            )

    if len(set(fixture_keys.values())) > 1:
        record.finding(
            "fixture-key-agreement",
            "ranks disagree on the fixture key (different tokens/prompt): {}".format(
                json.dumps({str(rank): key for rank, key in sorted(fixture_keys.items())})
            ),
        )
    else:
        fixture = next(iter(fixture_keys.values()))
        directory = os.path.join(reference_root, fixture)
        present = all(
            os.path.isfile(os.path.join(directory, "rank{}.json".format(rank))) for rank in ranks
        )
        record.passing(
            "fixture-key",
            "all ranks share fixture key {} (the numerical gate's sha256(positions+prompt_len+ids)[:16])".format(
                fixture
            ),
        )
        if present:
            record.passing("frozen-reference", "frozen reference present: {}".format(directory))
        else:
            record.warning(
                "frozen reference NOT present for fixture {}: expected {}; the gate will refuse "
                "until it is frozen with scripts/numerical-gate.sh --freeze".format(
                    fixture, os.path.join(directory, "rank{0..3}.json")
                )
            )
        return {
            "numerical_gate": {"available": True, "command": "python3 {} --run {}".format(NUMERICAL_GATE, os.path.abspath(row_dir))},
            "fixture": fixture,
            "reference": {"directory": directory, "present": present},
        }
    return {"numerical_gate": {"available": True}}


# --------------------------------------------------------------------------------------------------
# Rendering and CLI
# --------------------------------------------------------------------------------------------------

def render(result: dict) -> str:
    """Human-readable rendering of the verdict document."""
    lines = []
    label = "COMPLETE LAYER (D54 numerical-gate eligibility)" if result["mode"] == "complete" else "FRONTIER PROBE (refused row)"
    lines.append("device-frontier check: " + label)
    lines.append("row: " + result["row_dir"])
    lines.append("ranks under test: " + ", ".join(str(rank) for rank in result["ranks"]))
    for rank, report in sorted((result.get("rank_report") or {}).items(), key=lambda item: int(item[0])):
        lines.append(
            "  rank {}: status={} layer={} position={} arm={!r}".format(
                rank, report.get("status"), report.get("layer"), report.get("position"), report.get("arm")
            )
        )
        lines.append("          path={}".format(report.get("path")))
        lines.append("          frontier={}".format(report.get("frontier")))
    lines.append("")
    lines.append("checks:")
    for check in result["checks"]:
        lines.append("  {:<7} {}: {}".format(check["status"], check["id"], check["detail"]))
    for warning in result["warnings"]:
        lines.append("  WARN    " + warning)
    lines.append("")
    if result["verdict"] == "COHERENT-PROBE":
        lines.append(
            "verdict: COHERENT PROBE - this row is a probe, NOT a measurement: it declares no D54 "
            "arm, no ms/step and no numerical result, and it must not be quoted as one."
        )
    elif result["verdict"] == "COHERENT-COMPLETE-LAYER":
        lines.append(
            "verdict: COHERENT COMPLETE LAYER - the row is eligible for the D54 numerical gate; "
            "eligibility is not a gate pass and this tool does not run the gate."
        )
    elif result["verdict"] == "INCOHERENT":
        lines.append(
            "verdict: INCOHERENT - {} finding(s); a row that cannot be a probe must not be "
            "reported as one.".format(len(result["findings"]))
        )
    else:
        lines.append(
            "verdict: CANNOT TESTIFY - stop, never a retry and never a pass: {}".format(
                result.get("stop_reason") or "a required document is missing"
            )
        )
    if result.get("next_step"):
        lines.append("next: " + result["next_step"])
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI."""
    parser = argparse.ArgumentParser(description="Verdict tool for tp4 device-layer frontier documents")
    parser.add_argument("--row-dir", help="row directory holding tp4-layer-device-r{rank}.json")
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="later stage: require status=completed, full 19-stage frontier, gate inputs present",
    )
    parser.add_argument(
        "--ranks",
        default=",".join(str(rank) for rank in DEFAULT_RANKS),
        help="comma-separated ranks under test (default 0,1,2,3)",
    )
    parser.add_argument("--json-out", help="write the verdict document to this path")
    parser.add_argument(
        "--note-needle",
        action="append",
        default=None,
        help="extra case-insensitive not-host-validated needle for the refused note",
    )
    parser.add_argument("--expect-schema", default=SCHEMA, help="expected document schema string")
    parser.add_argument(
        "--reference-root",
        default=None,
        help="frozen numerical-gate reference root (default docs/build/reference/speed-gate)",
    )
    parser.add_argument("--write-fixtures", metavar="DIR", help="write the synthetic self-test fixtures and exit")
    parser.add_argument("--selftest", action="store_true", help="run the synthetic self-test and exit")
    return parser


def parse_ranks(text: str) -> tuple:
    """Parse --ranks. A rank set must be unique positive-or-zero integers."""
    try:
        ranks = tuple(int(part) for part in text.split(",") if part.strip() != "")
    except ValueError as error:
        raise SystemExit("--ranks needs comma-separated integers: {}".format(error)) from error
    if not ranks or len(set(ranks)) != len(ranks) or any(rank < 0 for rank in ranks):
        raise SystemExit("--ranks must be a non-empty set of distinct non-negative integers")
    return ranks


def run_selftest() -> int:
    """Build the synthetic fixtures, check every case's exit code, and prove both directions."""
    workspace = tempfile.mkdtemp(prefix="device-frontier-selftest-")
    try:
        manifest = write_fixture_set(workspace)
        failures = 0
        print("device-frontier selftest: {} synthetic case(s) under {}".format(len(manifest), workspace))
        for case in manifest:
            result = evaluate(
                case["row_dir"],
                require_complete=case["mode"] == "complete",
            )
            got = result["exit_code"]
            ok = got == case["expected_exit"]
            if not ok:
                failures += 1
            print(
                "  {:<7} {:<36} mode={:<8} expected={} got={} verdict={}".format(
                    "PASS" if ok else "FAIL",
                    case["case"],
                    case["mode"],
                    case["expected_exit"],
                    got,
                    result["verdict"],
                )
            )
            if not ok:
                for finding in result["findings"]:
                    print("          finding: " + finding)
                if result.get("stop_reason"):
                    print("          stop: " + result["stop_reason"])
        print(
            "selftest: {} of {} case(s) matched their expected exit code".format(
                len(manifest) - failures, len(manifest)
            )
        )
        return EXIT_COHERENT if failures == 0 else EXIT_INCOHERENT
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def main(argv=None) -> int:
    """CLI entry point. Prints the verdict; the exit code is the interface."""
    args = build_arg_parser().parse_args(argv)
    if args.write_fixtures:
        manifest = write_fixture_set(args.write_fixtures)
        print(
            "wrote {} synthetic case(s) to {} (SYNTHETIC: not device rows)".format(
                len(manifest), os.path.abspath(args.write_fixtures)
            )
        )
        for case in manifest:
            print(
                "  {}: mode={} expected_exit={} row_dir={}".format(
                    case["case"], case["mode"], case["expected_exit"], case["row_dir"]
                )
            )
        return EXIT_COHERENT
    if args.selftest:
        return run_selftest()
    if not args.row_dir:
        print("ERROR: --row-dir is required (or use --selftest / --write-fixtures)", file=sys.stderr)
        return EXIT_CANNOT_TESTIFY
    ranks = parse_ranks(args.ranks)
    needles = None
    if args.note_needle:
        needles = tuple(NOTE_NEEDLES) + tuple(args.note_needle)
    try:
        result = evaluate(
            args.row_dir,
            ranks=ranks,
            require_complete=args.require_complete,
            note_needles=needles,
            expect_schema=args.expect_schema,
            reference_root=args.reference_root,
        )
    except Exception as error:  # noqa: BLE001 - an internal error can never be a pass
        print("INTERNAL ERROR: {}: {}".format(type(error).__name__, error), file=sys.stderr)
        print("verdict: CANNOT TESTIFY - an internal error is a stop, never a pass", file=sys.stderr)
        return EXIT_CANNOT_TESTIFY
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
    print(render(result))
    return result["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
