#!/usr/bin/env python3
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

"""Copies files from the lead Spark to the others, at the same absolute paths, in parallel.

    scripts/sync-nodes.py --host user@198.51.100.12,spark-1 [--host ...] --path /abs/dir [--path ...]
                          [--exclude '*.incomplete'] [--ensure-dir /abs/dir] [--ssh-arg=-oX=Y]
                          [--streams 4]

The web control plane runs this to install the server binary, kernels and NCCL library on the other
Sparks, and to copy the model there. Each `--host` is `destination,ssh-name`: the destination may
be a fast-network address, and the SSH name is the one whose host key is already trusted
(`HostKeyAlias`), so nothing new is written to known_hosts.

The files are split into `--streams` lists of about equal size, and every host gets one `rsync`
per list, all at once. A single SSH stream tops out well below the 200 Gb/s link, so several in
parallel are what fill it, and every host reads the same files in the same order so the lead's
page cache serves the second and third copy. rsync writes each file under a temporary name and
renames it when complete, so an interrupted copy never leaves a short file under its real name; a
rerun skips the files that already match in size and time.

Progress goes to stdout as `@progress {"done": N, "total": N, "detail": "..."}` lines.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import threading
import time

PROGRESS = re.compile(r"^\s*([\d,]+)\s+\d+%")


def collect(paths: list[str], excludes: list[str]) -> list[tuple[str, int]]:
    """Every file and symlink under `paths`, with its size, skipping excluded names."""

    def wanted(name: str) -> bool:
        return not any(fnmatch.fnmatch(name, pattern) for pattern in excludes)

    entries: list[tuple[str, int]] = []
    for root in paths:
        root = os.path.abspath(root)
        if os.path.islink(root) or os.path.isfile(root):
            entries.append((root, os.lstat(root).st_size))
            continue
        if not os.path.isdir(root):
            raise SystemExit(f"{root} does not exist on this Spark")
        for directory, dirs, files in os.walk(root):
            # A symlink to a directory is copied as the link, not followed.
            for name in [name for name in dirs if os.path.islink(os.path.join(directory, name))]:
                dirs.remove(name)
                files.append(name)
            dirs[:] = [name for name in dirs if wanted(name)]
            for name in sorted(files):
                if wanted(name):
                    path = os.path.join(directory, name)
                    size = 0 if os.path.islink(path) else os.lstat(path).st_size
                    entries.append((path, size))
    return entries


def split(entries: list[tuple[str, int]], streams: int) -> list[list[tuple[str, int]]]:
    """`streams` lists of about equal total size, largest files placed first."""
    buckets: list[list[tuple[str, int]]] = [[] for _ in range(max(1, streams))]
    sizes = [0] * len(buckets)
    for entry in sorted(entries, key=lambda entry: (-entry[1], entry[0])):
        index = sizes.index(min(sizes))
        buckets[index].append(entry)
        sizes[index] += entry[1]
    return [bucket for bucket in buckets if bucket]


def human(count: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if count < 1000:
            return f"{count:.0f} {unit}"
        count /= 1000
    return f"{count:.1f} TB"


def eta(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


class Transfer:
    """One rsync process: one host, one file list."""

    def __init__(self, alias: str, total: int, command: list[str]):
        self.alias = alias
        self.total = total
        self.command = command
        self.done = 0
        self.process: subprocess.Popen | None = None

    def start(self) -> None:
        self.process = subprocess.Popen(
            self.command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, bufsize=0,
        )
        threading.Thread(target=self.read, daemon=True).start()

    def read(self) -> None:
        assert self.process and self.process.stdout
        pending = b""
        while chunk := self.process.stdout.read(65536):
            pending += chunk
            # --info=progress2 redraws its line with carriage returns.
            *lines, pending = re.split(rb"[\r\n]", pending)
            for line in lines:
                self.line(line.decode(errors="replace"))
        self.line(pending.decode(errors="replace"))

    def line(self, text: str) -> None:
        match = PROGRESS.match(text)
        if match:
            self.done = min(self.total, int(match.group(1).replace(",", "")))
        elif text.strip():
            print(f"{self.alias}: {text.strip()}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--host", action="append", required=True,
                        help="destination,ssh-name (destination may be user@address)")
    parser.add_argument("--path", action="append", required=True, help="absolute path to copy")
    parser.add_argument("--exclude", action="append", default=[], help="file name pattern to skip")
    parser.add_argument("--ensure-dir", action="append", default=[],
                        help="absolute directory to create on every host")
    parser.add_argument("--ssh-arg", action="append", default=[], help="one ssh argument")
    parser.add_argument("--streams", type=int, default=4, help="rsync processes per host")
    parser.add_argument("--ssh", default="ssh", help=argparse.SUPPRESS)
    parser.add_argument("--dest-root", default="/", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    for path in args.path + args.ensure_dir:
        if not os.path.isabs(path):
            parser.error(f"{path} is not an absolute path")
    version = subprocess.run(["rsync", "--version"], capture_output=True, text=True, check=False)
    match = re.search(r"version\s+(\d+)\.(\d+)", version.stdout)
    if not match or (int(match.group(1)), int(match.group(2))) < (3, 1):
        raise SystemExit("rsync 3.1 or newer is needed on this Spark (sudo apt install rsync)")

    entries = collect(args.path, args.exclude)
    total = sum(size for _, size in entries)
    buckets = split(entries, args.streams)
    hosts = []
    for spec in args.host:
        destination, _, alias = spec.partition(",")
        hosts.append((destination, alias or destination.rpartition("@")[2]))
    print(
        f"Copying {len(entries)} files ({human(total)}) to {', '.join(a for _, a in hosts)} "
        f"in {len(buckets)} streams each",
        flush=True,
    )

    dest_root = args.dest_root.rstrip("/") or ""
    workdir = tempfile.mkdtemp(prefix="sparkflash-sync-")
    lists = []
    for index, bucket in enumerate(buckets):
        path = os.path.join(workdir, f"files-{index}")
        with open(path, "wb") as handle:
            for name, _ in bucket:
                handle.write(name.lstrip("/").encode() + b"\0")
        lists.append((path, sum(size for _, size in bucket)))

    # Every host first: rsync must be there, and the directories the launcher needs must exist.
    for destination, alias in hosts:
        ssh = [*shlex.split(args.ssh), *args.ssh_arg, "-o", f"HostKeyAlias={alias}"]
        dirs = " ".join(shlex.quote(dest_root + path) for path in args.ensure_dir)
        check = "command -v rsync >/dev/null || { echo 'rsync is not installed' >&2; exit 3; }"
        if dirs:
            check += f" && mkdir -p -- {dirs}"
        result = subprocess.run([*ssh, destination, check], stdin=subprocess.DEVNULL,
                                capture_output=True, text=True, check=False)
        if result.returncode != 0:
            detail = result.stderr.strip() or f"exit {result.returncode}"
            raise SystemExit(f"Can't prepare {alias}: {detail}")

    transfers = []
    for destination, alias in hosts:
        rsh = shlex.join([*shlex.split(args.ssh), *args.ssh_arg, "-o", f"HostKeyAlias={alias}"])
        for path, size in lists:
            transfers.append(Transfer(alias, size, [
                "rsync", "-a", "--whole-file", "--from0", f"--files-from={path}",
                "--info=progress2", "--no-human-readable", "-e", rsh,
                "/", f"{destination}:{dest_root}/",
            ]))
    for transfer in transfers:
        transfer.start()

    grand = total * len(hosts)
    started = time.monotonic()

    def report() -> None:
        done = sum(transfer.done for transfer in transfers)
        rate = done / max(time.monotonic() - started, 1e-3)
        detail = f"{human(done)} of {human(grand)}"
        if rate > 0 and done < grand:
            detail += f", {rate / 1e6:.0f} MB/s, about {eta((grand - done) / rate)} left"
        line = json.dumps({"done": done, "total": grand, "detail": detail})
        print(f"@progress {line}", flush=True)

    failed = []
    while any(transfer.process.poll() is None for transfer in transfers):
        time.sleep(1.0)
        report()
    for transfer in transfers:
        code = transfer.process.wait()
        if code == 0:
            transfer.done = transfer.total
        else:
            failed.append(f"{transfer.alias} (rsync exit {code})")
    report()
    for path, _ in lists:
        os.remove(path)
    os.rmdir(workdir)
    if failed:
        raise SystemExit("The copy failed for " + ", ".join(sorted(set(failed))))
    print(f"Copied {human(total)} to {len(hosts)} Sparks", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
