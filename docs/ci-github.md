# GitHub Actions CI

The GitHub pipeline is a transcription of [.gitlab-ci.yml](../.gitlab-ci.yml), not a
replacement design.  The jobs, images, shards, cache boundaries and nightly split are the
same; only the vocabulary differs (merge request -> pull request, runner cache -> runner-local
volume/directory).  Read the comments in `.gitlab-ci.yml` for why the jobs are shaped this way.

| File | Upstream pipeline |
| :-- | :-- |
| [.github/workflows/ci.yml](../.github/workflows/ci.yml) | check + test stages |
| [.github/workflows/accuracy.yml](../.github/workflows/accuracy.yml) | .accuracy jobs (schedule / manual) |

## The runners

Two persistent repository-level runners share the work:

| Label | Host | Executor | Jobs |
| :-- | :-- | :-- | :-- |
| `sparkflash` | `gentoo-server` (x86_64) | Docker container `rust:1.98` | check, test, scripts, feature-sweep, gate-artifact |
| `gb10` | `gentoo-p16` (aarch64, GB10) | native host toolchain | cuda, cuda-baseline |

The x64 host has Docker but no CUDA toolkit, so the container jobs ask for
`[self-hosted, linux, x64, sparkflash]`.  The GB10 host has the CUDA 13 toolkit and an RTX
Spark GPU but no Docker, so it runs the CUDA jobs natively and they ask only for
`[self-hosted, gb10]`.

```sh
# on gentoo-server
systemctl status actions-runner-sparkflash
journalctl -u actions-runner-sparkflash -f

# on gentoo-p16
systemctl status actions-runner-gb10
journalctl -u actions-runner-gb10 -f
```

Install and configuration live in `~/actions-runner` on each host; the work tree is
`~/actions-runner/_work`.  The x64 service runs as the login user (a member of the `docker`
group) because its jobs create sibling containers through the host daemon.  The GB10 service
runs as the same user and uses the rustup toolchain in `~/.cargo`/`~/.rustup`; its CUDA
toolkit is `/opt/cuda/bin`.

The GB10 host ships OpenSSL 4, which the runner's bundled .NET probes for as version 3; a
small `~/openssl-compat` directory with `libssl.so.3`/`libcrypto.so.3` symlinks and an
`LD_LIBRARY_PATH` in the unit bridge that gap.

While no `.runner` file exists, `actions-runner-onboard.service` serves
<http://127.0.0.1:8097> with a fresh GitHub device code; open it, press Authorize, and the
service registers the runner and starts the unit by itself.  The code is rotated before it
expires, so the page can be left open.  Stop it with
`systemctl disable --now actions-runner-onboard` once the runner is registered.

```sh
# one-time (or after a GitHub-side removal): authorize, register, start
~/actions-runner/setup-github-runner.sh

# or non-interactively, with a token that has repo + workflow scopes
GH_TOKEN=github_pat_... ~/actions-runner/setup-github-runner.sh
```

## Build cache

GitHub's cache service caps a repository at 10 GB; this workspace's `target/` is well over
100 GB.  The cache stays on the runner, and no host path is written into the workflow:

* The x64 container jobs mount the Docker named volume `sparkflash-ci-cache` at `/cache`, with
  `CARGO_HOME=/cache/cargo-home`, `RUSTUP_HOME=/cache/rustup` and
  `CARGO_TARGET_DIR=/cache/target`.  The first job installs the 1.98.1 toolchain and its
  components once; later jobs reuse it.
* The native GB10 jobs use `$GITHUB_WORKSPACE/../ci-cache`, with
  `CARGO_TARGET_DIR=$GITHUB_WORKSPACE/../ci-cache/target-cuda`; the rustup toolchain and
  registry stay in `~/.cargo` and `~/.rustup`.

This is the same reuse the GitLab runner's cache archive provides, without a 10 GB ceiling and
without an upload.  Because the cache is shared by every job on a runner, cargo's own file
locks serialise builds if two jobs overlap; the x64 runner executes one job at a time, so in
practice the shards run back to back and reuse each other's artifacts.

## Schedule

Both workflows carry a cron entry (09:00 and 09:15 UTC).  GitHub disables scheduled
workflows after 60 days without repository activity; a push or a manual run re-enables them.
The nightly run is the equivalent of the GitLab pipeline schedule: the per-commit gates plus
the accuracy jobs.

## Security

`allenh1/sparkflash` is public and the runners are persistent, so a fork pull request can
execute its own workflow file on either host.  The x64 container jobs mount only the checkout,
the cache volume and the runner's temporary action directories; the Docker socket is not
mounted.  The GB10 jobs run natively, so a fork PR there has the runner user's shell.  Keep
**Require approval for all outside collaborators** enabled in *Settings -> Actions -> General*,
and treat a fork PR as code that will run on the host once approved.  Ephemeral runners or a
VM-per-job executor are the next step if that is not enough.
