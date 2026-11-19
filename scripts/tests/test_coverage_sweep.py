# Copyright (c) 2026 Hunter L. Allen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the coverage suite: the missing-expert report, the generator and the sweep."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent.parent / "numerics.py"
SPEC = importlib.util.spec_from_file_location("numerics", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
NUMERICS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(NUMERICS)

MODEL = "toy/coverage"
EXPERTS = [4, 4]
CORPUS = Path(__file__).resolve().parent.parent.parent / "numerics" / "coverage" / "corpus.json"


def coverage_dump(inputs: list[str] | None = None, hits: list[list[int]] | None = None) -> dict:
    return {
        "schema": NUMERICS.COVERAGE_SCHEMA,
        "model": MODEL,
        "inputs": [{"name": name, "tokens": 1} for name in (inputs or [])],
        "layers": [
            {"layer": layer, "hits": list(hits[layer]) if hits else [0] * experts}
            for layer, experts in enumerate(EXPERTS)
        ],
    }


def coverage_layout(model: str = MODEL) -> dict:
    return {
        "schema": NUMERICS.LAYOUT_SCHEMA,
        "model": model,
        "layers": [{"first": 0, "count": len(EXPERTS), "experts": EXPERTS[0]}],
    }


def corpus_document(prompts: list[tuple[str, str]], schema: str = NUMERICS.COVERAGE_CORPUS_SCHEMA) -> dict:
    return {
        "schema": schema,
        "prompts": [{"name": name, "prompt": text} for name, text in prompts],
    }


class FakeCoverageServer(BaseHTTPRequestHandler):
    """A tiny server that routes every token to an expert and writes the dump like the recorder.

    Tokenisation is one id per character, so a one-character prompt names a small, known set of
    slots: token 'A' reaches expert ('A' + layer) % experts at every layer.
    """

    model = MODEL
    experts = EXPERTS
    dump_path: Path
    completions: list[list[int]] = []

    def log_message(self, *_args: object) -> None:
        pass

    def _send(self, document: dict, status: int = 200) -> None:
        body = json.dumps(document).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - the HTTP server's interface
        if self.path == "/v1/models":
            self._send(
                {
                    "data": [
                        {
                            "id": self.model,
                            "numerics": {
                                "status": "registered",
                                "profile": "reference",
                                "version": 1,
                                "registered_version": 1,
                                "fingerprint": "0" * 16,
                                "evidence": "logprobs",
                                "world": 1,
                                "plan": "tp",
                                "arch": "sm_121a",
                                "overrides": [],
                            },
                            "build": {"version": "0", "id": "test"},
                            "capabilities": {},
                        }
                    ]
                }
            )
            return
        self._send({"error": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802 - the HTTP server's interface
        length = int(self.headers.get("content-length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/tokenize":
            if "messages" in body:
                text = "\n".join(message.get("content", "") for message in body["messages"])
            else:
                text = body.get("prompt", "")
            self._send({"tokens": [ord(character) for character in text] or [0]})
            return
        if self.path != "/v1/completions":
            self._send({"error": "not found"}, 404)
            return
        ids = [int(token) for token in body.get("prompt") or []]
        self.record(ids)
        self._send({"choices": [{"index": 0, "text": "", "finish_reason": "length"}]})

    def record(self, ids: list[int]) -> None:
        with self.server.lock:  # type: ignore[attr-defined]
            document = json.loads(self.dump_path.read_text()) if self.dump_path.exists() else coverage_dump()
            if document.get("model") != self.model:
                document = coverage_dump()
            for layer, layer_document in enumerate(document["layers"]):
                for token in ids:
                    layer_document["hits"][(token + layer) % self.experts[layer]] += 1
            name = NUMERICS.token_input_name(ids)
            if not any(entry["name"] == name for entry in document["inputs"]):
                document["inputs"].append({"name": name, "tokens": len(ids)})
            temporary = self.dump_path.with_name(self.dump_path.name + ".tmp")
            temporary.write_text(json.dumps(document))
            os.replace(temporary, self.dump_path)
            self.completions.append(ids)  # type: ignore[attr-defined]


class CoverageGapsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.layout = coverage_layout()
        self.dump = coverage_dump(
            inputs=["input-0", "input-1"],
            hits=[[1, 0, 2, 0], [0, 0, 0, 3]],
        )

    def test_the_report_names_every_missing_layer(self) -> None:
        result = NUMERICS.coverage_inspect(self.dump, self.layout, "layout.json", 0)
        self.assertEqual(len(result["missing"]), 5)
        self.assertEqual(result["missing_by_layer"], {0: [1, 3], 1: [0, 1, 2]})
        self.assertEqual(result["covered_experts"], 3)
        self.assertEqual(result["experts"], 8)
        document = NUMERICS.coverage_gaps_document(result, "dump.json")
        self.assertEqual(document["missing"], 5)
        self.assertFalse(document["complete"])
        self.assertEqual([layer["layer"] for layer in document["layers"]], [0, 1])
        markdown = NUMERICS.render_coverage_gaps(result)
        self.assertIn("5 missing", markdown)
        self.assertIn("| 1 | 4 | 3 |", markdown)

    def test_the_judge_reports_the_first_missing_layer_and_fails_below_the_bar(self) -> None:
        result = NUMERICS.judge_coverage(self.dump, self.layout, "layout.json", 1, 0.5)
        self.assertFalse(result["passes"])
        self.assertEqual(result["missing_by_layer"], {0: [1, 3], 1: [0, 1, 2]})
        markdown = NUMERICS.render_coverage(result)
        self.assertIn("FAIL coverage", markdown)
        self.assertIn("37.50%", markdown)
        self.assertIn("50.00%", markdown)
        self.assertIn("1, 3", markdown)

    def test_a_complete_dump_reports_no_gaps(self) -> None:
        complete = coverage_dump(inputs=["input-fd1f0f4381eb0395"], hits=[[1, 1, 1, 1], [1, 1, 1, 1]])
        result = NUMERICS.coverage_inspect(complete, self.layout, "layout.json", 1)
        self.assertEqual(result["missing"], [])
        self.assertIn("PASS coverage", NUMERICS.render_coverage_gaps(result))
        self.assertTrue(NUMERICS.coverage_gaps_document(result, "dump.json")["complete"])

    def test_the_command_writes_the_full_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dump_path = Path(directory) / "coverage.json"
            layout_path = Path(directory) / "layout.json"
            report_path = Path(directory) / "gaps.json"
            dump_path.write_text(json.dumps(self.dump))
            layout_path.write_text(json.dumps(self.layout))
            with contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(
                    NUMERICS.main(
                        [
                            "coverage-gaps",
                            "--dump",
                            str(dump_path),
                            "--layout",
                            str(layout_path),
                            "--json",
                            str(report_path),
                        ]
                    ),
                    0,
                )
            self.assertIn("5 missing", out.getvalue())
            report = json.loads(report_path.read_text())
            self.assertEqual(report["missing"], 5)
            self.assertEqual(report["layers"][1]["missing"], [0, 1, 2])


class CoverageGeneratorTests(unittest.TestCase):
    def test_the_checked_in_corpus_covers_the_documented_families(self) -> None:
        entries, duplicates = NUMERICS.load_coverage_corpus(CORPUS)
        self.assertEqual(duplicates, [])
        self.assertGreaterEqual(len(entries), 40)
        families = {entry["family"] for entry in entries}
        for family in ("charset", "code", "numerals", "scripts", "long", "misc"):
            self.assertIn(family, families)
        self.assertEqual(len({entry["name"] for entry in entries}), len(entries))
        self.assertEqual(len({NUMERICS.coverage_prompt_digest(entry) for entry in entries}), len(entries))

    def test_generation_is_deterministic_and_uses_every_family(self) -> None:
        first = NUMERICS.generate_coverage_prompts(7, 21)
        second = NUMERICS.generate_coverage_prompts(7, 21)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 21)
        self.assertEqual({entry["family"] for entry in first}, {"charset", "numerals", "code", "scripts", "long", "repetition", "mixed"})
        self.assertEqual(len({entry["name"] for entry in first}), 21)
        self.assertTrue(all(entry["prompt"].strip() for entry in first))
        self.assertNotEqual(first, NUMERICS.generate_coverage_prompts(8, 21))

    def test_the_python_name_matches_the_rust_recorder(self) -> None:
        self.assertEqual(NUMERICS.token_input_name([1, 2, 3]), "input-fd1f0f4381eb0395")
        self.assertEqual(NUMERICS.token_input_name([4]), "input-cd3ac65e44f721b1")
        self.assertEqual(NUMERICS.token_input_name([]), "input-cbf29ce484222325")


class CoverageSweepTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.dump = root / "coverage.json"
        self.layout = root / "layout.json"
        self.suite = root / "next.json"
        self.report = root / "sweep.json"
        self.dump.write_text(json.dumps(coverage_dump()))
        self.layout.write_text(json.dumps(coverage_layout()))
        FakeCoverageServer.completions = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeCoverageServer)
        self.server.lock = threading.Lock()  # type: ignore[attr-defined]
        FakeCoverageServer.dump_path = self.dump
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def write_corpus(self, prompts: list[tuple[str, str]]) -> Path:
        path = Path(self.directory.name) / "corpus.json"
        path.write_text(json.dumps(corpus_document(prompts)))
        return path

    def run_sweep(self, corpus: Path, *extra: str) -> tuple[int, str, dict]:
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = NUMERICS.main(
                [
                    "coverage-sweep",
                    "--url",
                    self.url,
                    "--dump",
                    str(self.dump),
                    "--layout",
                    str(self.layout),
                    "--corpus",
                    str(corpus),
                    "--order",
                    "corpus",
                    "--batch",
                    "2",
                    "--max-candidates",
                    "32",
                    "--out",
                    str(self.suite),
                    "--json",
                    str(self.report),
                    *extra,
                ]
            )
        return code, out.getvalue(), json.loads(self.report.read_text())

    def test_the_sweep_keeps_only_prompts_that_add_a_slot(self) -> None:
        corpus = self.write_corpus(
            [("A", "A"), ("dup-A", "A"), ("AA", "AA"), ("B", "B"), ("C", "C"), ("D", "D")]
        )
        code, output, report = self.run_sweep(corpus)
        self.assertEqual(code, 0, output)
        self.assertTrue(report["complete"])
        self.assertEqual(report["missing_before"], 8)
        self.assertEqual(report["missing_after"], 0)
        self.assertEqual(report["selected"], 4)
        self.assertEqual(report["selected_names"], ["A", "B", "C", "D"])
        self.assertEqual(report["sent"], 5, "the duplicate content is never sent")
        self.assertEqual(report["duplicates"], ["dup-A"])
        self.assertEqual([batch["new_slots"] for batch in report["batches"]], [2, 4, 2])
        self.assertEqual(report["batches"][0]["kept"], 1)
        self.assertEqual(report["batches"][0]["discarded"], 1)
        self.assertEqual(report["batches"][0]["coverage_before"], 0.0)
        self.assertEqual(report["batches"][0]["coverage_after"], 0.25)
        self.assertEqual(len(FakeCoverageServer.completions), 5)
        suite = json.loads(self.suite.read_text())
        self.assertEqual(suite["schema"], NUMERICS.SUITE_SCHEMA)
        self.assertEqual([prompt["name"] for prompt in suite["prompts"]], ["A", "B", "C", "D"])
        self.assertTrue(suite["coverage"]["complete"])
        self.assertIn("selected 4 prompt(s)", output)

    def test_a_second_sweep_does_not_duplicate_an_input_already_in_the_dump(self) -> None:
        corpus = self.write_corpus([("A", "A"), ("B", "B"), ("C", "C"), ("D", "D")])
        self.run_sweep(corpus, "--max-candidates", "1")
        sent = len(FakeCoverageServer.completions)
        self.assertEqual(sent, 1)
        # Drop the seen manifest: the dump's content-addressed input name must be enough.
        Path(str(self.dump) + ".seen.json").unlink()
        code, output, report = self.run_sweep(corpus)
        self.assertEqual(code, 0, output)
        self.assertEqual(report["skipped"]["already-in-dump"], 1)
        self.assertEqual(len(FakeCoverageServer.completions), sent + 3)
        self.assertNotIn([ord("A")], FakeCoverageServer.completions[sent:])
        self.assertEqual([prompt["name"] for prompt in json.loads(self.suite.read_text())["prompts"]], ["B", "C", "D"])

    def test_the_seen_manifest_skips_a_prompt_a_previous_window_sent(self) -> None:
        first = self.write_corpus([("A", "A"), ("B", "B")])
        self.run_sweep(first, "--max-candidates", "1")
        second = Path(self.directory.name) / "corpus-second.json"
        second.write_text(json.dumps(corpus_document([("A", "A"), ("C", "C")])))
        code, output, report = self.run_sweep(second)
        self.assertEqual(code, 0, output)
        self.assertEqual(report["skipped"]["already-seen"], 1)
        self.assertEqual(report["selected_names"], ["C"])
        self.assertNotIn([ord("A")], FakeCoverageServer.completions[1:])

    def test_a_legacy_dump_needs_its_input_suite(self) -> None:
        self.dump.write_text(
            json.dumps(coverage_dump(inputs=["input-0", "input-1"], hits=[[1, 0, 0, 0], [0, 0, 0, 0]]))
        )
        corpus = self.write_corpus([("A", "A"), ("B", "B")])
        with contextlib.redirect_stdout(io.StringIO()):
            code = NUMERICS.main(
                [
                    "coverage-sweep",
                    "--url",
                    self.url,
                    "--dump",
                    str(self.dump),
                    "--layout",
                    str(self.layout),
                    "--corpus",
                    str(corpus),
                    "--out",
                    str(self.suite),
                ]
            )
        self.assertEqual(code, 1)
        self.assertEqual(FakeCoverageServer.completions, [])
        # The declaration says which prompts the legacy inputs ran; the sweep skips their content.
        suite_path = Path(self.directory.name) / "input-suite.json"
        suite_path.write_text(json.dumps(corpus_document([("A", "A")], schema=NUMERICS.SUITE_SCHEMA)))
        self.dump.write_text(json.dumps(coverage_dump(inputs=["input-0"], hits=[[1, 0, 0, 0], [0, 0, 0, 0]])))
        code, output, report = self.run_sweep(corpus, "--input-suite", str(suite_path))
        self.assertEqual(code, 0, output)
        self.assertEqual(report["skipped"]["already-seen"], 1)
        self.assertNotIn([ord("A")], FakeCoverageServer.completions)
        self.assertEqual(report["selected_names"], ["B"])

    def test_a_complete_dump_makes_no_request(self) -> None:
        self.dump.write_text(json.dumps(coverage_dump(hits=[[1, 1, 1, 1], [1, 1, 1, 1]])))
        corpus = self.write_corpus([("A", "A")])
        code, output, report = self.run_sweep(corpus)
        self.assertEqual(code, 0, output)
        self.assertTrue(report["complete"])
        self.assertEqual(report["sent"], 0)
        self.assertEqual(FakeCoverageServer.completions, [])

    def test_the_sweep_refuses_a_layout_or_server_model_mismatch(self) -> None:
        corpus = self.write_corpus([("A", "A")])
        other_layout = Path(self.directory.name) / "other.json"
        other_layout.write_text(json.dumps(coverage_layout("toy/other")))
        with contextlib.redirect_stdout(io.StringIO()):
            code = NUMERICS.main(
                [
                    "coverage-sweep",
                    "--url",
                    self.url,
                    "--dump",
                    str(self.dump),
                    "--layout",
                    str(other_layout),
                    "--corpus",
                    str(corpus),
                    "--out",
                    str(self.suite),
                ]
            )
        self.assertEqual(code, 1)
        self.assertEqual(FakeCoverageServer.completions, [])

        FakeCoverageServer.model = "toy/other"
        self.addCleanup(setattr, FakeCoverageServer, "model", MODEL)
        with contextlib.redirect_stdout(io.StringIO()):
            code = NUMERICS.main(
                [
                    "coverage-sweep",
                    "--url",
                    self.url,
                    "--dump",
                    str(self.dump),
                    "--layout",
                    str(self.layout),
                    "--corpus",
                    str(corpus),
                    "--out",
                    str(self.suite),
                ]
            )
        self.assertEqual(code, 1)
        self.assertEqual(FakeCoverageServer.completions, [])

    def test_a_bad_corpus_or_a_negative_generate_is_refused(self) -> None:
        bad = Path(self.directory.name) / "bad.json"
        bad.write_text(json.dumps({"schema": "not-a-corpus", "prompts": []}))
        with contextlib.redirect_stdout(io.StringIO()):
            code = NUMERICS.main(
                [
                    "coverage-sweep",
                    "--url",
                    self.url,
                    "--dump",
                    str(self.dump),
                    "--layout",
                    str(self.layout),
                    "--corpus",
                    str(bad),
                    "--out",
                    str(self.suite),
                ]
            )
        self.assertEqual(code, 1)
        corpus = self.write_corpus([("A", "A")])
        with self.assertRaisesRegex(NUMERICS.NumericsFailure, "cannot be negative"):
            NUMERICS.generate_coverage_prompts(0, -1)
        with contextlib.redirect_stdout(io.StringIO()):
            code = NUMERICS.main(
                [
                    "coverage-sweep",
                    "--url",
                    self.url,
                    "--dump",
                    str(self.dump),
                    "--layout",
                    str(self.layout),
                    "--corpus",
                    str(corpus),
                    "--generate",
                    "-1",
                    "--out",
                    str(self.suite),
                ]
            )
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
