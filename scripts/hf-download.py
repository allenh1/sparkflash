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

"""Downloads a Hugging Face model into the hub cache, resumably, for the web control plane.

    scripts/hf-download.py --repo deepseek-ai/DeepSeek-V4.1-Flash [--cache DIR] [--jobs 8]

Standard library only, so a fresh Spark needs no `pip install`. The files land in the layout
`huggingface_hub` writes, which `sparkflash --web` and `huggingface_hub` itself both read:

    models--org--name/blobs/<etag>                      the file's bytes
    models--org--name/snapshots/<commit>/<path>         a relative symlink to its blob
    models--org--name/refs/<revision>                   the commit the revision names

A blob's name is its LFS SHA-256, or its git blob SHA-1 for a small file, and each one is checked
against that hash before it is renamed into place. Every file still to fetch is marked with an
empty `blobs/<etag>.incomplete` up front, so a cache scan sees an unfinished download as unfinished
even for files not started yet. An interrupted run resumes each file where it stopped.

Progress goes to stdout as `@progress {"done": N, "total": N, "detail": "..."}` lines, which the
control plane turns into a progress bar. The token comes from `HF_TOKEN` or the token file
`huggingface-cli login` writes; public models need none. It is sent to the Hugging Face host only,
never to the storage host a file download redirects to.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

CHUNK = 8 << 20
MARGIN = 1 << 30
DEFAULT_ENDPOINT = "https://huggingface.co"
MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*(/[A-Za-z0-9][A-Za-z0-9_.-]*)?$")


def default_cache() -> str:
    """The hub cache huggingface_hub uses, by the same environment precedence."""
    for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if os.environ.get(name):
            return os.path.expanduser(os.environ[name])
    if os.environ.get("HF_HOME"):
        return os.path.join(os.path.expanduser(os.environ["HF_HOME"]), "hub")
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "huggingface", "hub")


def find_token() -> str | None:
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(name, "").strip():
            return os.environ[name].strip()
    home = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    path = os.environ.get("HF_TOKEN_PATH") or os.path.join(home, "token")
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read().strip() or None
    except OSError:
        return None


class Hub:
    """The few Hugging Face endpoints a download needs."""

    def __init__(self, endpoint: str, token: str | None):
        self.endpoint = endpoint.rstrip("/")
        self.token = token

    def request(self, url: str, headers: dict[str, str] | None = None) -> urllib.request.Request:
        request = urllib.request.Request(url, headers={"User-Agent": "sparkflash-hf-download/1"})
        for key, value in (headers or {}).items():
            request.add_header(key, value)
        if self.token:
            # An unredirected header is not copied onto a redirect, so the token never reaches
            # the storage host a file download is sent to.
            request.add_unredirected_header("Authorization", f"Bearer {self.token}")
        return request

    def get_json(self, url: str) -> tuple[object, str | None]:
        try:
            with urllib.request.urlopen(self.request(url), timeout=60) as response:
                return json.load(response), next_link(response.headers.get("Link", ""))
        except urllib.error.HTTPError as error:
            if error.code in (401, 403):
                raise SystemExit(
                    f"Hugging Face refused access ({error.code}). If the model is gated, accept its "
                    "terms on huggingface.co and set HF_TOKEN."
                ) from None
            if error.code == 404:
                raise SystemExit(f"Hugging Face has no such model or revision: {url}") from None
            raise

    def commit(self, repo: str, revision: str) -> str:
        quoted = urllib.parse.quote(revision, safe="")
        info, _ = self.get_json(f"{self.endpoint}/api/models/{repo}/revision/{quoted}")
        return info["sha"]

    def files(self, repo: str, commit: str) -> list[dict]:
        """Every file of the commit: path, size and blob name."""
        url: str | None = f"{self.endpoint}/api/models/{repo}/tree/{commit}?recursive=true"
        files = []
        while url:
            page, url = self.get_json(url)
            for entry in page:
                if entry.get("type") != "file":
                    continue
                lfs = entry.get("lfs") or {}
                files.append(
                    {
                        "path": entry["path"],
                        "size": int(lfs.get("size", entry.get("size", 0))),
                        "etag": lfs.get("oid") or entry["oid"],
                        "lfs": bool(lfs),
                    }
                )
        return files

    def file_url(self, repo: str, commit: str, path: str) -> str:
        return f"{self.endpoint}/{repo}/resolve/{commit}/{urllib.parse.quote(path)}"


def next_link(header: str) -> str | None:
    for part in header.split(","):
        if 'rel="next"' in part and "<" in part and ">" in part:
            return part[part.index("<") + 1 : part.index(">")]
    return None


def new_hash(entry: dict):
    if entry["lfs"]:
        return hashlib.sha256()
    digest = hashlib.sha1()  # noqa: S324 - git's blob id, not a security check
    digest.update(f"blob {entry['size']}\0".encode())
    return digest


class Progress:
    """Byte and file counters shared by the workers, printed once a second."""

    def __init__(self, total: int, files: int, done: int, done_files: int):
        self.total = total
        self.files = files
        self.done = done
        self.done_files = done_files
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.abort = threading.Event()
        self.started = time.monotonic()
        self.started_bytes = done

    def add(self, count: int) -> None:
        with self.lock:
            self.done += count

    def file_done(self) -> None:
        with self.lock:
            self.done_files += 1

    def emit(self) -> None:
        with self.lock:
            done, files = self.done, self.done_files
        elapsed = max(time.monotonic() - self.started, 1e-3)
        rate = (done - self.started_bytes) / elapsed
        detail = f"{files} of {self.files} files"
        if rate > 0 and done < self.total:
            detail += f", {rate / 1e6:.0f} MB/s, about {eta((self.total - done) / rate)} left"
        line = json.dumps({"done": done, "total": self.total, "detail": detail})
        print(f"@progress {line}", flush=True)

    def run(self) -> None:
        while not self.stop.wait(1.0):
            self.emit()


def eta(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def human(count: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if count < 1000:
            return f"{count:.0f} {unit}"
        count /= 1000
    return f"{count:.1f} TB"


def fetch(hub: Hub, repo: str, commit: str, entry: dict, blobs: str, progress: Progress,
          attempts: int, retry_wait: float) -> None:
    """Downloads one blob, resuming its `.incomplete` file, and renames it into place."""
    final = os.path.join(blobs, entry["etag"])
    partial = final + ".incomplete"
    size = entry["size"]
    url = hub.file_url(repo, commit, entry["path"])
    for attempt in range(1, attempts + 1):
        have = os.path.getsize(partial) if os.path.exists(partial) else 0
        if have > size:
            os.truncate(partial, 0)
            progress.add(-have)
            have = 0
        try:
            if have < size:
                headers = {"Range": f"bytes={have}-"} if have else {}
                with urllib.request.urlopen(hub.request(url, headers), timeout=60) as response:
                    if have and response.status != 206:
                        # The server ignored the range: start over.
                        os.truncate(partial, 0)
                        progress.add(-have)
                        have = 0
                    with open(partial, "ab") as out:
                        while chunk := response.read(CHUNK):
                            if progress.abort.is_set():
                                return
                            out.write(chunk)
                            progress.add(len(chunk))
            if os.path.getsize(partial) != size:
                raise OSError(f"got {os.path.getsize(partial)} of {size} bytes")
            digest = new_hash(entry)
            with open(partial, "rb") as handle:
                while chunk := handle.read(CHUNK):
                    digest.update(chunk)
            if digest.hexdigest() != entry["etag"]:
                os.truncate(partial, 0)
                progress.add(-size)
                raise OSError("the bytes do not match the published hash; downloading again")
            os.replace(partial, final)
            progress.file_done()
            print(f"{entry['path']}: done ({human(size)})", flush=True)
            return
        except urllib.error.HTTPError as error:
            if error.code in (401, 403, 404):
                raise SystemExit(f"{entry['path']}: Hugging Face answered {error.code}") from None
            problem = f"HTTP {error.code}"
        except (OSError, urllib.error.URLError, TimeoutError) as error:
            problem = str(error)
        if attempt == attempts:
            raise SystemExit(f"{entry['path']}: giving up after {attempts} tries: {problem}")
        wait = min(60.0, retry_wait * 2 ** (attempt - 1))
        print(f"{entry['path']}: {problem}; retrying in {wait:.0f} s", flush=True)
        time.sleep(wait)


def link(snapshot: str, blobs: str, entry: dict) -> None:
    path = os.path.join(snapshot, *entry["path"].split("/"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    target = os.path.relpath(os.path.join(blobs, entry["etag"]), os.path.dirname(path))
    if os.path.islink(path) and os.readlink(path) == target:
        return
    if os.path.lexists(path):
        os.remove(path)
    os.symlink(target, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--repo", required=True, help="model id, e.g. org/name")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--cache", default=None, help="hub cache directory")
    parser.add_argument("--jobs", type=int, default=8, help="files fetched at once")
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", DEFAULT_ENDPOINT))
    parser.add_argument("--attempts", type=int, default=8, help="tries per file")
    parser.add_argument("--retry-wait", type=float, default=2.0, help="first retry delay, s")
    args = parser.parse_args(argv)

    if not MODEL_ID.match(args.repo) or ".." in args.repo or "--" in args.repo:
        parser.error(f"{args.repo!r} is not a Hugging Face model id")
    cache = os.path.abspath(os.path.expanduser(args.cache or default_cache()))
    hub = Hub(args.endpoint, find_token())
    print(f"Looking up {args.repo} ({args.revision}) on {hub.endpoint}", flush=True)
    commit = hub.commit(args.repo, args.revision)
    files = hub.files(args.repo, commit)
    if not files:
        raise SystemExit(f"{args.repo} at {commit} has no files")

    repo_dir = os.path.join(cache, "models--" + args.repo.replace("/", "--"))
    blobs = os.path.join(repo_dir, "blobs")
    snapshot = os.path.join(repo_dir, "snapshots", commit)
    os.makedirs(blobs, exist_ok=True)
    os.makedirs(snapshot, exist_ok=True)
    os.makedirs(os.path.join(repo_dir, "refs"), exist_ok=True)
    if args.revision != commit:
        with open(os.path.join(repo_dir, "refs", args.revision), "w", encoding="utf-8") as handle:
            handle.write(commit)

    # One blob can back several paths; fetch each once.
    unique: dict[str, dict] = {}
    for entry in files:
        unique.setdefault(entry["etag"], entry)
    total = sum(entry["size"] for entry in unique.values())
    pending = []
    have = 0
    for etag, entry in unique.items():
        if os.path.exists(os.path.join(blobs, etag)):
            have += entry["size"]
            continue
        partial = os.path.join(blobs, etag + ".incomplete")
        if os.path.exists(partial):
            have += min(os.path.getsize(partial), entry["size"])
        else:
            open(partial, "ab").close()
        pending.append(entry)
    need = total - have
    free = shutil.disk_usage(blobs).free
    print(
        f"{args.repo} at {commit[:12]}: {len(unique)} files, {human(total)}; "
        f"{human(need)} to fetch, {human(free)} free in {cache}",
        flush=True,
    )
    if need + MARGIN > free:
        raise SystemExit(
            f"Not enough disk space: the model needs {human(need)} more but {cache} has "
            f"{human(free)} free. Free some space or set model.hub_cache to a bigger disk."
        )

    progress = Progress(total, len(unique), have, len(unique) - len(pending))
    reporter = threading.Thread(target=progress.run, daemon=True)
    reporter.start()
    # Largest first, so the long transfers overlap instead of trailing at the end.
    pending.sort(key=lambda entry: -entry["size"])
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs))
    try:
        futures = [
            pool.submit(fetch, hub, args.repo, commit, entry, blobs, progress,
                        max(1, args.attempts), args.retry_wait)
            for entry in pending
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    except BaseException:
        # One file failed for good: stop the others where they are (they resume next time).
        progress.abort.set()
        pool.shutdown(wait=True, cancel_futures=True)
        progress.stop.set()
        raise
    pool.shutdown(wait=True)
    progress.stop.set()
    reporter.join()
    for entry in files:
        link(snapshot, blobs, entry)
    progress.emit()
    print(f"Downloaded {args.repo} to {snapshot}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
