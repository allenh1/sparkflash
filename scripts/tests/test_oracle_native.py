#!/usr/bin/env python3
"""Unit tests for scripts/oracle_native.py and the pinned prompt suite.

Host-only: no server, no GPU, no network. Run with:

    python3 -m unittest discover -s scripts/tests -v
"""

import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
sys.path.insert(0, SCRIPTS)

import oracle_native as oracle


class BuildPromptTest(unittest.TestCase):
    def test_continuation_and_sentinel(self):
        self.assertEqual(oracle.build_prompt([1, 2], [3, 4], 0), [1, 2, 3, 4, 0])

    def test_empty_continuation_still_appends_sentinel(self):
        self.assertEqual(oracle.build_prompt([7], [], 0), [7, 0])

    def test_sentinel_is_recorded(self):
        body = oracle.build_request_body("m", [1, 2, 3], 1000)
        self.assertEqual(body["prompt"], [1, 2, 3])
        self.assertEqual(body["prompt_logprobs"], 1000)
        self.assertEqual(body["temperature"], 0.0)
        self.assertEqual(body["max_tokens"], 1)
        self.assertTrue(body["ignore_eos"])


class ServedRootTest(unittest.TestCase):
    def test_parse_served(self):
        served_id, root = oracle.parse_served({"data": [{"id": "m", "root": "/x/1bae0e10"}]})
        self.assertEqual(served_id, "m")
        self.assertEqual(root, "/x/1bae0e10")

    def test_parse_served_rejects_empty(self):
        with self.assertRaises(oracle.OracleError):
            oracle.parse_served({"data": []})

    def test_root_matches(self):
        root = "/cache/models--ds/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170"
        self.assertTrue(oracle.root_matches(root, oracle.DEFAULT_EXPECT_ROOT))
        self.assertTrue(oracle.root_matches(root, None))
        self.assertFalse(oracle.root_matches("/cache/other", oracle.DEFAULT_EXPECT_ROOT))
        self.assertFalse(oracle.root_matches(None, oracle.DEFAULT_EXPECT_ROOT))


class PromptLogprobsTest(unittest.TestCase):
    def _entry(self, **logprobs):
        return {str(token): {"logprob": value} for token, value in logprobs.items()}

    def test_position_equals_entry_index_minus_one(self):
        prompt = [10, 11, 12, 13]
        plp = [
            None,
            self._entry(**{"11": -0.5, "12": -2.0}),
            self._entry(**{"12": -0.1, "11": -3.0}),
            self._entry(**{"13": -0.25}),
        ]
        positions, first_is_null = oracle.parse_prompt_logprobs(plp, prompt)
        self.assertTrue(first_is_null)
        self.assertEqual([row["position"] for row in positions], [0, 1, 2])
        self.assertEqual([row["target_token"] for row in positions], [11, 12, 13])
        self.assertEqual(positions[0]["top"][0], [11, -0.5])
        self.assertTrue(positions[0]["target_in_distribution"])
        self.assertEqual(positions[1]["logprob_count"], 2)
        self.assertEqual(positions[0]["logprobs"], {"11": -0.5, "12": -2.0})

    def test_first_entry_may_be_a_dict(self):
        positions, first_is_null = oracle.parse_prompt_logprobs(
            [self._entry(**{"1": -1.0}), self._entry(**{"2": -1.0})], [1, 2]
        )
        self.assertFalse(first_is_null)
        self.assertEqual(len(positions), 1)

    def test_length_mismatch_is_an_error(self):
        with self.assertRaises(oracle.OracleError):
            oracle.parse_prompt_logprobs([None], [1, 2])

    def test_null_position_is_an_error(self):
        with self.assertRaises(oracle.OracleError):
            oracle.parse_prompt_logprobs([None, None], [1, 2])

    def test_non_finite_logprob_is_an_error(self):
        with self.assertRaises(oracle.OracleError):
            oracle.parse_prompt_logprobs([None, {"2": {"logprob": float("inf")}}], [1, 2])

    def test_missing_logprob_key_is_an_error(self):
        with self.assertRaises(oracle.OracleError):
            oracle.parse_prompt_logprobs([None, {"2": {"not_logprob": 1.0}}], [1, 2])


class ResponseParseTest(unittest.TestCase):
    def test_missing_prompt_logprobs_is_an_error(self):
        with self.assertRaises(oracle.OracleError):
            oracle.parse_response(json.dumps({"choices": [{"text": "x"}]}).encode())

    def test_bad_json_is_an_error(self):
        with self.assertRaises(oracle.OracleError):
            oracle.parse_response(b"not json")


class SuiteTest(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(SCRIPTS, "oracle_prompt_suite.json")

    def test_pinned_suite_shape(self):
        suite = oracle.load_suite(self.path)
        prompts = suite["prompts"]
        self.assertGreaterEqual(len(prompts), 8)
        lengths = [prompt["n_tokens"] for prompt in prompts]
        self.assertEqual(lengths, [len(prompt["tokens"]) for prompt in prompts])
        self.assertGreaterEqual(min(lengths), 130)
        self.assertLessEqual(max(lengths), 500)
        self.assertEqual(len({prompt["id"] for prompt in prompts}), len(prompts))
        self.assertEqual(suite["fixture_prompt"]["n_tokens"], 130)
        tok = suite["tokenizer"]["sha256"]
        self.assertEqual(len(tok), 64)
        self.assertEqual(prompts[0]["id"], "b69-fixture-130")

    def test_suite_sha256_is_stable(self):
        first = oracle.sha256_file(self.path)
        second = oracle.sha256_file(self.path)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)

    def test_bad_schema_is_an_error(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump({"schema": "other", "prompts": []}, handle)
            path = handle.name
        try:
            with self.assertRaises(oracle.OracleError):
                oracle.load_suite(path)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
