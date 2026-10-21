"""Tests for the four-node TP=4 serving launcher skeleton.

The script's `print` mode is pure command construction: it never opens a socket, so a host with
no cluster can still pin the command line each rank would run. A launch itself is not tested here
(that is the device window's job); the contract this file guards is the four rank commands and the
refusals for a malformed topology.
"""

from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent.parent / "serve-four-node.sh"
NODES = "10.0.0.1 10.0.0.2 10.0.0.3 10.0.0.4"


def base_env(**overrides: str) -> dict[str, str]:
    """The environment a print-mode run needs, with the caller's overrides applied."""
    env = dict(os.environ)
    env.update(
        {
            "SPARK_NODES": NODES,
            "SPARKFLASH_CHECKPOINT": "/models/ckpt",
            "SPARKFLASH_TOKENIZER": "/models/tokenizer.json",
            "SPARKFLASH_BIN": "/opt/sparkflash",
            "SPARKFLASH_HTTP_BIND": "0.0.0.0:8000",
        }
    )
    env.update(overrides)
    return env


def run_script(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        check=False,
        capture_output=True,
        text=True,
        env=env if env is not None else base_env(),
    )


class PrintPlanTests(unittest.TestCase):
    """The offline plan is the contract a four-node launch depends on."""

    def test_one_command_per_rank_in_rank_order(self) -> None:
        result = run_script("print")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("ssh -n"), 4, result.stdout)
        for rank in range(4):
            self.assertIn(f"--tp4-rank {rank}", result.stdout)
        self.assertIn("--tp4-world-size 4", result.stdout)
        self.assertIn("--tp4-peers 10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4", result.stdout)
        self.assertIn("/models/ckpt", result.stdout)
        self.assertIn("/models/tokenizer.json", result.stdout)

    def test_rank_zero_owns_http_and_workers_stay_on_loopback(self) -> None:
        result = run_script("print")
        lines = [line for line in result.stdout.splitlines() if line.strip().startswith("ssh -n")]
        self.assertEqual(len(lines), 4)
        self.assertIn("--bind 0.0.0.0:8000", lines[0])
        for line in lines[1:]:
            self.assertIn("--bind 127.0.0.1:0", line)
            self.assertNotIn("--bind 0.0.0.0:8000", line)

    def test_port_overrides_reach_the_commands(self) -> None:
        result = run_script(
            "print",
            env=base_env(
                SPARKFLASH_TP4_BASE_PORT="31000",
                SPARKFLASH_TP4_CONTROL_PORT="31050",
            ),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("--tp4-base-port 31000"), 4, result.stdout)
        self.assertEqual(result.stdout.count("--tp4-control-port 31050"), 4, result.stdout)

    def test_control_port_defaults_to_base_plus_one_hundred(self) -> None:
        result = run_script("print", env=base_env(SPARKFLASH_TP4_BASE_PORT="31000"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("--tp4-control-port 31100"), 4, result.stdout)

    def test_drain_knobs_are_exported_per_rank(self) -> None:
        result = run_script(
            "print",
            env=base_env(
                SPARKFLASH_DRAIN_DELAY_MS="2500",
                SPARKFLASH_DRAIN_TIMEOUT_MS="45000",
            ),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("SPARKFLASH_SERVE_DRAIN_DELAY_MS=2500"), 4)
        self.assertEqual(result.stdout.count("SPARKFLASH_SERVE_DRAIN_TIMEOUT_MS=45000"), 4)

    def test_dry_run_environment_uses_the_print_path(self) -> None:
        result = run_script("launch", env=base_env(SPARKFLASH_DRY_RUN="1"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("ssh -n"), 4)
        self.assertIn("--tp4-rank 3", result.stdout)


class RefusalTests(unittest.TestCase):
    """A malformed topology must fail before any ssh command is built."""

    def test_three_nodes_are_refused(self) -> None:
        result = run_script("print", env=base_env(SPARK_NODES="10.0.0.1 10.0.0.2 10.0.0.3"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("exactly 4 addresses", result.stderr)

    def test_five_nodes_are_refused(self) -> None:
        result = run_script(
            "print",
            env=base_env(SPARK_NODES="10.0.0.1 10.0.0.2 10.0.0.3 10.0.0.4 10.0.0.5"),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("exactly 4 addresses", result.stderr)

    def test_missing_checkpoint_is_refused(self) -> None:
        result = run_script("print", env=base_env(SPARKFLASH_CHECKPOINT=""))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SPARKFLASH_CHECKPOINT is required", result.stderr)

    def test_missing_tokenizer_is_refused(self) -> None:
        result = run_script("print", env=base_env(SPARKFLASH_TOKENIZER=""))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SPARKFLASH_TOKENIZER is required", result.stderr)

    def test_unknown_mode_is_refused(self) -> None:
        result = run_script("deploy", env=base_env())
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unknown mode", result.stderr)

    def test_no_command_is_printed_on_refusal(self) -> None:
        result = run_script("print", env=base_env(SPARK_NODES="10.0.0.1"))
        self.assertNotIn("ssh -n", result.stdout)


if __name__ == "__main__":
    unittest.main()
