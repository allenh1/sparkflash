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

    def test_context_and_lane_settings_reach_every_rank(self) -> None:
        """The web control plane owns these two knobs; a manual launch must forward them too."""
        result = run_script(
            "print",
            env=base_env(SPARK_MAX_SEQ_LEN="262144", SPARKFLASH_LANES="8"),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("SPARK_MAX_SEQ_LEN=262144"), 4)
        self.assertEqual(result.stdout.count("SPARKFLASH_LANES=8"), 4)

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

    def test_device_token_window_reaches_rank_zero_only(self) -> None:
        result = run_script(
            "print", env=base_env(SPARKFLASH_DEVICE_TOKEN_DIR="/var/lib/spark/windows")
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = [line for line in result.stdout.splitlines() if line.strip().startswith("ssh -n")]
        self.assertEqual(len(lines), 4)
        self.assertIn("--device-token-dir /var/lib/spark/windows", lines[0])
        for line in lines[1:]:
            self.assertNotIn("--device-token-dir", line)

    def test_live_follower_flag_is_opt_in(self) -> None:
        plain = run_script("print", env=base_env(SPARKFLASH_DEVICE_TOKEN_DIR="/var/lib/spark/w"))
        self.assertNotIn("--device-token-live", plain.stdout)
        live = run_script(
            "print",
            env=base_env(
                SPARKFLASH_DEVICE_TOKEN_DIR="/var/lib/spark/w",
                SPARKFLASH_DEVICE_TOKEN_LIVE="1",
            ),
        )
        self.assertEqual(live.returncode, 0, live.stderr)
        lines = [line for line in live.stdout.splitlines() if line.strip().startswith("ssh -n")]
        self.assertIn("--device-token-live", lines[0])
        self.assertNotIn("--device-token-live", lines[1])

    def test_device_token_flags_are_absent_without_a_window_directory(self) -> None:
        result = run_script(
            "print",
            env=base_env(
                SPARKFLASH_DEVICE_TOKEN_LIVE="1",
                SPARKFLASH_DEVICE_TOKEN_DERIVED_CLAIM="1",
            ),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("--device-token", result.stdout)


class LocalRankTests(unittest.TestCase):
    """A rank in SPARK_LOCAL_RANKS starts on this node without ssh."""

    def test_local_rank_is_started_without_ssh(self) -> None:
        result = run_script("print", env=base_env(SPARK_LOCAL_RANKS="0"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("ssh -n"), 3, result.stdout)
        local = [line for line in result.stdout.splitlines() if line.strip().startswith("bash -c")]
        self.assertEqual(len(local), 1, result.stdout)
        self.assertIn("--tp4-rank 0 ", local[0])
        self.assertIn("rank 0 -> 10.0.0.1 (this node)", result.stdout)

    def test_every_start_is_exec_detached(self) -> None:
        result = run_script("print", env=base_env(SPARK_LOCAL_RANKS="0"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("exec nohup env"), 4, result.stdout)

    def test_status_needs_no_checkpoint(self) -> None:
        # Stop and status never read the weights: a model missing from the cache must not block
        # them. Every rank is local here, so nothing is reached over ssh.
        result = run_script(
            "status",
            env=base_env(
                SPARKFLASH_CHECKPOINT="",
                SPARKFLASH_TOKENIZER="",
                SPARK_LOCAL_RANKS="0 1 2 3",
                SPARK_NODES="n0 n1 n2 n3",
            ),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("not running"), 4, result.stdout)

    def test_out_of_range_local_rank_is_refused(self) -> None:
        result = run_script("print", env=base_env(SPARK_LOCAL_RANKS="0 4"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SPARK_LOCAL_RANKS", result.stderr)
        self.assertNotIn("ssh -n", result.stdout)


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
