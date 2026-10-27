<!--
Copyright (c) 2026 Hunter L. Allen

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Web control plane

`sparkflash --web` starts a lead-node control plane for the four-Spark cluster. It reads
`sparkflash.toml`, discovers the other nodes, proves passwordless SSH, drives the existing
converter and four-node launcher, and graphs rank 0's live telemetry. The intended clean-cluster
flow is: copy the example config, edit four nodes, open the UI, pick the model, **Convert all
ranks**, **Launch cluster**.

```bash
cp sparkflash.toml.example sparkflash.toml
target/release/sparkflash --web --config sparkflash.toml
# control plane on http://spark-0.local:8790 (any address of the lead node)
```

`sparkflash web` is the same entry point. `--bind ADDR` overrides `web.bind`; `--no-mdns` keeps the
configured-node SSH checks but disables discovery and advertising. `SPARKFLASH_CONFIG` and
`SPARKFLASH_WEB_BIND` are the environment equivalents.

The control plane is a management surface, not an inference proxy. Clients keep talking to rank
0's normal API at `cluster.http_bind`. Sparks run headless, so the UI listens on all interfaces
(`0.0.0.0:8790`) by default and is opened from another machine. It has no login of its own:
anyone who can reach the port can launch, stop or reconfigure the cluster, so keep it on a trusted
network. To limit it to the lead node, set `web.bind = "127.0.0.1:8790"` and tunnel:

```bash
ssh -N -L 8790:127.0.0.1:8790 user@spark-0
# then open http://127.0.0.1:8790
```

Two guards protect the listener from other pages in the operator's browser:

- Every request that changes something (`PUT`, `POST`) must carry `x-sparkflash-web: 1`. The UI
  sends it; a cross-site form or a simple cross-origin request cannot, so it gets 403.
- The `Host` header must be an IP literal, `localhost` (or `*.localhost`), this machine's hostname
  (or `<hostname>.local`), or an entry in `web.allowed_hosts`. Anything else gets 421, which stops
  a DNS-rebinding page from reading or driving the API. Add a reverse proxy's public name to
  `web.allowed_hosts`.

The UI has three sections: **Overview** (status, serving figures, graphs, ranks and tokens
served), **Setup** (the configuration form, bring-up checklist and discovery) and **Activity** (jobs
and their logs).

## Configuration

The file is the source of truth. The web form edits the same typed document and writes it
atomically (a sibling temp file, then rename). See
[`sparkflash.toml.example`](../sparkflash.toml.example) for every field with comments.

| Section | What it controls |
| :-- | :-- |
| `[model]` | Architecture (`weights`), Hugging Face model `id`, checkpoint (empty: looked up in the hub cache), hub cache, tokenizer, converted weights, cubins |
| `[engine]` | How the weights run (`kind`, `device`) |
| `[runtime]` | `SPARK_MAX_SEQ_LEN` and `SPARKFLASH_LANES` |
| `[cluster]` | Rank 0 HTTP bind, collective base port, SSH user/options, exactly four nodes (`ssh`, RoCE `peer`, optional `local`) |
| `[paths]` | Checkout, binary, launcher, NCCL library, log root on every node |
| `[network]` | `NCCL_IB_HCA`, `NCCL_SOCKET_IFNAME`, `NCCL_IB_GID_INDEX`, extra rank environment |
| `[web]` | Control-plane bind, rank 0 URL, extra allowed `Host` names, discovery/SSH timeouts, advertising |
| `[stats]` | Sampling interval, in-memory retention, idle snapshot path and flush interval |

Validation refuses an unknown weight/engine selection, a context outside 1..=1,048,576, a
concurrency outside 1..=32, duplicate SSH names or peers, a non-IP peer, a malformed socket
address/URL, and anything other than exactly four nodes. It also refuses SSH names and users that
SSH would read as options (a leading `-`, whitespace, `@` in the user), extra environment keys
that are not shell identifiers, `web.allowed_hosts` entries with a scheme or port, and an empty
`model.checkpoint` with a `model.id` that is not a Hugging Face model ID. The model and engine selectors are
separate values: "model" is the weights, "engine" is the runtime. Only `deepseek-v4.1-flash` is
implemented for either today; adding a second choice is a deliberate change to
`SUPPORTED_WEIGHTS`/`SUPPORTED_ENGINES` plus the code that can serve it.

Relative paths resolve against `paths.repo_root`. Relative `paths.repo_root` resolves against the
process working directory. The sample is checkout-relative; a production install usually uses
absolute paths.

## Model and checkpoint

`model.id` is the checkpoint's Hugging Face model ID, and an empty `model.checkpoint` means "find
it in the Hugging Face hub cache". The cache has one fixed layout, the one `huggingface_hub` writes:

```text
<cache>/models--<org>--<name>/refs/main          commit hash that main points at
<cache>/models--<org>--<name>/snapshots/<hash>/  the checkout used as the checkpoint
```

The cache directory is `model.hub_cache`, else `HF_HUB_CACHE`, else `$HF_HOME/hub`, else
`~/.cache/huggingface/hub` of the user running the control plane. The ID maps to its snapshot
through `refs/main`; when a download recorded no ref, the newest snapshot is used. The lookup runs
on the lead node and the resulting path is passed to every rank, so download the model on every
node into the same cache path (`huggingface-cli download <id>` as the same user does that).

The **Model** list on the Setup page is the lead node's cache. Each model shows its size, and a
download that is still in progress is marked incomplete. A model whose `config.json` `model_type`
this build cannot load is listed but disabled. The line under the list shows the snapshot the next
launch or conversion will use. A launch or conversion for a model that is not in the cache is refused with
the cache path in the message; stop and status still work. A non-empty `model.checkpoint` skips the
lookup. Use it for a checkpoint outside the cache, and then `model.id` may be any name.

Switching models means converting every rank again: the converted weights belong to one
checkpoint.

## Applying changes

A saved `web.serve_url` and `stats.sample_interval_secs` take effect on the next sample without a
restart. `web.bind`, the advertisement, the stats retention/flush policy and `stats.persist_path`
are fixed at process start; restart the control plane after changing them.

## Node discovery and SSH

The process advertises `_sparkflash._tcp.local.` and browses:

- `_sparkflash._tcp.local.` (another lead)
- `_ssh._tcp.local.` / `_sftp-ssh._tcp.local.` (Avahi's SSH services)
- `_workstation._tcp.local.` / `_device-info._tcp.local.` (machine info)

Every answer is merged with the configured ranks, first by hostname and then by RoCE address, so
a renamed mDNS host still lands on its configured rank. Discovery never edits the file: the UI
shows a **Use for R*N*** action and the operator saves the change.

Each candidate is probed with `ssh -n -o BatchMode=yes -o ConnectTimeout=<n> <target> true`.
`BatchMode=yes` is always present; without it SSH could block on a password prompt, which is the
condition being reported. Probes run concurrently. A configured `local = true` rank is not probed:
its commands run on the lead node itself, so it shows as **this node**. A failure displays a
copyable command:

```bash
if [ ! -f ~/.ssh/id_ed25519 ]; then ssh-keygen -t ed25519 -N '' -f ~/.ssh/id_ed25519; fi
ssh-copy-id -i ~/.ssh/id_ed25519.pub spark@spark1
```

`ssh-copy-id` still asks for the target's password once. That is the only step passwordless SSH
cannot remove.

## Actions

| Action | What it runs |
| :-- | :-- |
| **Convert all ranks** / per-rank **Convert** | One job per rank: rank 0 locally when `local = true`, ranks 1–3 over SSH. Each runs `sparkflash convert --checkpoint … --output-dir … --tp-world-size 4 --tp-rank R --no-pad-moe-640 --no-verify` |
| **Launch cluster** | `scripts/serve-four-node.sh launch` with `SPARK_NODES`, `SPARK_PEERS`, `SPARK_LOCAL_RANKS` (the `local = true` ranks, started on the lead without SSH), `SPARK_MAX_SEQ_LEN`, `SPARKFLASH_LANES`, paths, and every `NCCL_*` value from the TOML. Each rank is started detached, so the job ends once all four are running |
| **Stop** | `scripts/serve-four-node.sh stop` (SIGTERM; rank 0 drains first) |
| **Rank status** | `scripts/serve-four-node.sh status` |

Conversion is refused while rank 0 is reachable; stop the cluster first. Launch is refused while
rank 0 is already reachable. Launch, stop and conversion exclude each other: the check for a
conflicting running job and the new job's registration happen under one lock, so two quick clicks
(or two browser tabs) cannot both start. These are safety guards around a one-time conversion,
not a general job scheduler. Launch, stop and conversion ask for confirmation in the UI.

Each job has a monotonically increasing id, a stable kind (`convert-r2`, `launch`, …), a status, an
exit code and a bounded 400-line log tail. The launcher's remote log files remain the durable
record. The UI polls job state; no request stays open for a five-minute conversion.

## Telemetry

A background task samples rank 0 once per `stats.sample_interval_secs`:

- `GET /metrics` for counters and the new last-request gauges
- `GET /v1/cluster/stats` for per-rank memory, clocks, rails, KV hit rate and DSpark acceptance
- `GET /health`, whose 503 body (the device loop failed or stopped making progress) turns the
  header status to **Stalled** and shows the reason in a banner

The new Prometheus families are:

| Family | Meaning |
| :-- | :-- |
| `sparkflash_draft_tokens_accepted_total` | DSpark drafts accepted, cumulative |
| `sparkflash_draft_tokens_proposed_total` | DSpark drafts proposed, cumulative |
| `sparkflash_last_decode_tokens_per_second` | Last completed request's decode speed |
| `sparkflash_last_prefill_tokens_per_second` | Last completed request's prefill speed |
| `sparkflash_last_acceptance_rate` | Last completed request's draft acceptance |
| `sparkflash_last_time_to_first_token_seconds` | Last completed request's TTFT |

What the overview graphs:

- **Decode** and **Prefill** are the last completed request's speeds.
- **Throughput** is tokens generated per second across all requests, from the
  `sparkflash_generation_tokens_total` delta between scrapes. Prompt tokens are left out because
  the server counts a prompt only when its request completes: a 300K-token prompt would otherwise
  land in one 2-second interval as 150K tok/s. Prompt tokens still count toward the totals and the
  daily bars.
- **Draft acceptance** and **Prefix cache hit** show "—" with the reason, rather than 0%, while
  there is nothing to measure: no draft tokens proposed yet (or the drafter is off), or no cache
  hits or lookups reported. On the TP4 path `/v1/cluster/stats` leaves its `dspark` and `kv`
  sections at zero, so the draft figures come from the Prometheus counters.

The sample is folded into an in-memory ring (`stats.retention_secs`, default 24 hours) and the
graphs request a window of 5m/15m/1h/6h/24h. The server averages the ring into at most 600 points
for the requested window, so a long window stays cheap to render.

Tokens served are tracked from counter deltas with reset handling: if a serving process restarts
and its counters go backwards, the new value is treated as new work. Totals and a UTC by-day map
are kept in `stats.persist_path`. **The file is written only when the last scrape says the cluster
is idle** (`sparkflash_num_requests_running == 0` and `sparkflash_num_requests_waiting == 0`) and at
most once per `stats.flush_interval_secs`. Inference never waits on this disk I/O. An explicit
idle snapshot is attempted on a clean control-plane shutdown; a busy cluster skips it.

The file is a small JSON document with a schema version, cumulative totals, the by-day map and the
last raw counter baseline. A control-plane restart continues the deltas; a serving restart is
detected by the counter going backwards. With no saved baseline, the first scrape counts the
server's existing counters into the totals but reports no throughput for that interval, so a
fresh control plane does not draw everything served so far as a one-sample spike.

## HTTP routes

| Route | Purpose |
| :-- | :-- |
| `GET /` | Embedded single-page dark UI (`#overview`, `#setup`, `#activity`) |
| `GET /assets/app.css`, `/assets/app.js` | Embedded assets, no CDN |
| `GET /api/state` | Config summary, latest sample, totals, jobs, cached discovery, advertised name |
| `GET /api/config` | `{path, exists, load_error, config}` |
| `PUT /api/config` | Validate and atomically save `{config: …}` |
| `POST /api/config/reload` | Re-read the file (an invalid file leaves the running config in place) |
| `POST /api/discover` | mDNS browse + concurrent SSH probes |
| `GET /api/series?window_secs=N` | Downsampled graph window |
| `GET /api/jobs`, `GET /api/jobs/{id}` | Job list and full log |
| `POST /api/actions/convert` | `{ranks?: [0,1,2,3]}` |
| `POST /api/actions/launch`, `/stop`, `/status` | Launcher actions |

Every `PUT` and `POST` needs the `x-sparkflash-web: 1` header, and every route checks `Host` as
described above. A script can drive the API the same way:

```bash
curl -X POST -H 'x-sparkflash-web: 1' http://127.0.0.1:8790/api/actions/status
```

## Testing

Everything above is exercised without a GPU, a checkpoint, multicast or an SSH server. The
control-plane crate uses trait-backed discovery and SSH probes, a temp-directory config, and axum
`oneshot` route tests. Run:

```bash
cargo test -p sparkflash-web
```

The sampler is tested with literal Prometheus text and cluster JSON, the store with synthetic
counters (including a restart), and the command builder with a quoted remote conversion.
