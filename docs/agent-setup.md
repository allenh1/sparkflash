# Agent Setup Guide

Instructions for an AI agent running on the lead DGX Spark (rank 0) to build, configure, stage, and
launch a four-node SparkFlash cluster without a browser.

## Prerequisites

- Four NVIDIA DGX Sparks (GB10, `aarch64`, DGX OS with CUDA 13 in `/usr/local/cuda/bin`), each with
  one QSFP port cabled to a 200 GbE switch and ~600 GB free disk space.
- Same user account name on all four Sparks.
- Internet access on the lead Spark.

## Step 1: Clone, build, and start the control plane

Run on the lead Spark (rank 0):

```bash
git clone https://github.com/allenh1/sparkflash ~/sparkflash
cd ~/sparkflash
./scripts/setup.sh --no-ssh
```

`./scripts/setup.sh --no-ssh` is idempotent. If `rustup` is not yet installed and stdin is not a
TTY, install Rust first:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path --profile minimal
export PATH="/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH"
./scripts/setup.sh --no-ssh
```

This builds `target/release/sparkflash`, compiles `target/cubins/*.cubin`, fetches
`target/deps/nccl/libnccl.so.2`, and starts the control plane in the background on
`http://127.0.0.1:8790`.

Every `PUT` and `POST` to `127.0.0.1:8790` requires the header `-H 'x-sparkflash-web: 1'`.

## Step 2: Discover the other three Sparks and verify SSH

Discover Sparks on the local network:

```bash
curl -s -X POST -H 'x-sparkflash-web: 1' http://127.0.0.1:8790/api/discover
```

Identify the three peer Sparks (e.g. `spark-1.local`, `spark-2.local`, `spark-3.local`, or ask the
user if ambiguous).

Verify passwordless SSH to each peer:

```bash
for host in spark-1.local spark-2.local spark-3.local; do
  ssh -n -o BatchMode=yes -o ConnectTimeout=5 "$host" true
done
```

If any peer fails `BatchMode=yes`, generate an SSH key if needed (`[ -f ~/.ssh/id_ed25519 ] || ssh-keygen -t ed25519 -N '' -f ~/.ssh/id_ed25519`) and ask the user to run `ssh-copy-id -i ~/.ssh/id_ed25519.pub <host>` once for each peer that needs it.

Once passwordless SSH works to all three peers, write `sparkflash.toml` from `sparkflash.toml.example` with rank 0 marked `local = true` and ranks 1–3 set to the three peer SSH names (`local = false`), then reload:

```bash
cp sparkflash.toml.example sparkflash.toml
# Edit cluster.nodes in sparkflash.toml so rank 0 is this node (local = true)
# and ranks 1..3 are the three peer SSH hostnames (local = false).
curl -s -X POST -H 'x-sparkflash-web: 1' http://127.0.0.1:8790/api/config/reload
```

## Step 3: Inspect nodes and configure the fast network

Run the parallel read-only node check across all four Sparks:

```bash
curl -s -X POST -H 'x-sparkflash-web: 1' http://127.0.0.1:8790/api/inspect
```

Check the JSON response:

- `nodes[].network_fix`: if non-null on any rank, it contains the exact `netplan` / `sysctl` commands
  needed on that Spark (MTU 9000, fast-subnet IPv4 address, or `arp_ignore=1` / `arp_announce=2`).
  Run those commands on the respective Spark (or ask the user to run them if `sudo` requires a
  password), then re-run `POST /api/inspect`.
- `findings`: verify there are no entries with `"severity": "error"`.
- `plan`: once present and error-free, apply the detected dual-rail RoCE settings (`NCCL_IB_HCA`,
  `NCCL_SOCKET_IFNAME`, `NCCL_IB_GID_INDEX`, and per-rank `peer` IPs) to `sparkflash.toml`:

```bash
curl -s -X POST -H 'x-sparkflash-web: 1' http://127.0.0.1:8790/api/network/apply
```

## Step 4: Stage software, download and copy the model, and convert weights

Each action below starts an asynchronous job and returns `{"jobs": [{"id": <job_id>, "status": "running", ...}]}`.
Poll a job until `status` is `"succeeded"` or `"failed"`:

```bash
curl -s http://127.0.0.1:8790/api/jobs/<id>
```

Run the staging steps in order (steps 4a and 4b can run in parallel):

1. **4a. Copy the binary, cubins, and NCCL library to ranks 1–3:**
   ```bash
   curl -s -X POST -H 'x-sparkflash-web: 1' -H 'content-type: application/json' \
     -d '{"what":"software"}' http://127.0.0.1:8790/api/actions/sync
   ```

2. **4b. Download `deepseek-ai/DeepSeek-V4.1-Flash` (510 GB) into the lead Spark's HF cache:**
   *(If a Hugging Face token is needed, save it to `~/.cache/huggingface/token` first.)*
   ```bash
   curl -s -X POST -H 'x-sparkflash-web: 1' http://127.0.0.1:8790/api/actions/download
   ```

3. **4c. Copy the downloaded model to ranks 1–3 over the fast network (after 4b succeeds):**
   ```bash
   curl -s -X POST -H 'x-sparkflash-web: 1' -H 'content-type: application/json' \
     -d '{"what":"model"}' http://127.0.0.1:8790/api/actions/sync
   ```

4. **4d. Convert each rank's quarter of the weights in parallel (~5 minutes, after 4a and 4c succeed):**
   ```bash
   curl -s -X POST -H 'x-sparkflash-web: 1' -H 'content-type: application/json' \
     -d '{}' http://127.0.0.1:8790/api/actions/convert
   ```
   Wait for all four conversion jobs (`convert-r0` .. `convert-r3`) to reach `"succeeded"`.

## Step 5: Launch and verify the server

Start all four ranks:

```bash
curl -s -X POST -H 'x-sparkflash-web: 1' http://127.0.0.1:8790/api/actions/launch
```

Wait for rank 0 to finish loading (~25 seconds) by polling `/readyz` on port 8000:

```bash
until curl -sf http://127.0.0.1:8000/readyz >/dev/null; do sleep 1; done
```

Send a test completion:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "deepseek-ai/DeepSeek-V4.1-Flash",
  "messages": [{"role": "user", "content": "Reply with OK."}],
  "max_tokens": 16
}'
```

## Managing the cluster

- **Check rank status:** `curl -s -X POST -H 'x-sparkflash-web: 1' http://127.0.0.1:8790/api/actions/status`
- **Cluster stats & health:** `curl -s http://127.0.0.1:8000/v1/cluster/stats` and `curl -s http://127.0.0.1:8000/health`
- **Graceful stop (drains in-flight requests):** `curl -s -X POST -H 'x-sparkflash-web: 1' http://127.0.0.1:8790/api/actions/stop`
