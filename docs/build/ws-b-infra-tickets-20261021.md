# WS-B infra tickets: RDMA prerequisites for the 90 tok/s phase (2026-10-21)

**Status:** draft for the Lead. No privileged action has been taken from this branch: no package
install, no config edit, no module load, no service or rank start/stop, no MTU or clock change, no
node reboot. Every node interaction recorded here was read-only ssh. The only live side effect was
the read-only survey in `scripts/ws-b-allreduce-bench.sh preflight` (section 8).

**Why this document exists.** The one-shot RDMA data plane is the WS-B latency path, and three of
its prerequisites are still missing. Two are root/switch-admin work with external lead time, so
the tickets are written now, with exact commands, acceptance checks and rollback, even though the
changes themselves have to happen in a maintenance window. The slowest step is the switch; start
it first.

**Inputs.**

- Read-only audit 2026-10-21 09:15-09:25 EDT:
  `/home/user/ev-session-20261020/analysis/rdma-prereq-20261021.md` (outside this repo; the
  report is the source of the PASS/FAIL matrix and the per-node captures).
- `docs/STATUS-2026-10-21.md` Infra row (same findings, one line).
- `docs/build/rdma-bringup.md`: the 2026-10-20 four-node bring-up, the only measured fabric run
  on this cluster (10 KiB one-shot All-Reduce median 36.0-37.1 us dual rail, 29.5 us single rail;
  the spec's 20 us bar was missed and 1 of 5 dual-rail 10k replays failed with a transport error).
- Re-confirmed read-only 2026-10-21 16:36 UTC by this branch's preflight (section 8): fabric
  netdevs MTU 1500 on both rails, `nvidia_peermem` absent, one GPU compute process per node,
  `:8000` listening on rank 0, f1 rails `4: ACTIVE`, f0 twins `1: DOWN`, host `libnccl` 0,
  `clocks.max.sm` 3003 MHz.

Every number below is labelled **measured** (with a path or command) or **estimate/hypothesis**.
There is no measured post-fix latency; the 25 us p50 gate has never been met on this fabric.

## 1. Ticket summary and ownership

| ID | Ticket | Blocked on | Lead time | Window needed | Touches live vLLM? |
| --- | --- | --- | --- | --- | --- |
| B1 | MTU 9000 end-to-end (four hosts + switch) | **root** (hosts) + **switch admin** | switch: days; hosts: minutes | yes, one window | **yes** (rail-1 netdev carries vLLM NCCL/socket traffic) |
| B2 | Load and persist `nvidia-peermem` (GPUDirect) | **root** | minutes | yes, one window | module load itself does not restart vLLM; rollback may fail while MRs are live |
| B3 | f0/f1 rail naming: standardise on f1 (A) or bring f0 up (B) | **Lead/spec owner** (A) or **root + switch admin + physical** (B) | A: 1 day; B: days/unknown | B only | B changes cabling/ports, so yes |
| B4a | SM clock cap check | **spec owner** (the check reads a hardware max a lock cannot change) + **root** for any lock | decision: immediate; change: minutes | if locked: yes | **yes** (clock policy is cluster-wide) |
| B4b | nvidia-uvm 64K packing | **spec owner** (no knob exists on 580-series, measured) | decision only | no | no |
| B4c | Engram shards | **root** (write `/var/lib/sparkflash/engram`) | ~1-2 h per node (estimate) | yes, or idle node | **no**, but it fills disk and is a node-local job |
| B5 | Host NCCL 2.28+ (`libnccl.so.2`) | **root** (package/ldconfig) | minutes per node | no restart of vLLM; host install only | no, unless NCCL is loaded into a serving process (do not) |
| B6 | Device-buffer one-shot seam (S5.3) + first qualifying run | **device-lead** (code) then B1+B2 | code: days; run: one window | run yes | run **yes**, never while vLLM serves |

Two prerequisites from the audit are **not** tickets because they already pass on all four nodes:
NVIDIA driver 580.178.04 (exact), kernel 7.0.0-1019-nvidia, RDMA devices present with 4 HCAs,
`/dev/infiniband/uverbs0..3` world-writable, memlock unlimited, 118.08-118.12 GiB MemAvailable.
Do not "fix" those.

## 2. Measured baseline (read-only, 2026-10-21)

| Fact | Value on .80 / .88 / .59 / .68 | Evidence |
| --- | --- | --- |
| fabric netdev MTU | 1500 / 1500 on `enp1s0f1np1` and `enP2p1s0f1np1` | preflight, audit §1 row 3 |
| DF-8972 ping | fails (needs MTU 9000) | audit `COMMANDS.md`, raw capture |
| RDMA `active_mtu` / `max_mtu` | 1024 / 4096 | `docs/build/rdma-bringup.md` §2 (2026-10-20) |
| `nvidia_peermem` | module present, **not loaded** | audit §1 row 5; preflight 16:36Z |
| GPUDirect | not available (module absent) | audit §1 row 5 |
| f1 rails `rocep1s0f1`, `roceP2p1s0f1` | `4: ACTIVE`, 200 Gb/s | preflight; bring-up §2 |
| f0 twins `rocep1s0f0`, `roceP2p1s0f0` | `1: DOWN` on all four | preflight; audit §2 |
| host NCCL | none (`ldconfig -p | grep -c libnccl` = 0) | preflight; audit §1 row 6 |
| NCCL in vLLM image | 2.28.3 in `vllm-node:latest` | audit §1 row 6 |
| `/etc/netplan/40-cx7.yaml` | present, root-only; backup `40-cx7.yaml~` beside it | preflight |
| `systemd-networkd` / `NetworkManager` | both active | preflight |
| `clocks.max.sm` / `clocks.current.sm` | 3003 / 2437-2496 MHz | preflight; audit §1 row 9 |
| `nvidia-smi --query-gpu=clocks.min.sm` | **invalid field on this driver** | measured 2026-10-21 12:37 EDT |
| Engram manifest `/var/lib/sparkflash/engram/manifest.json` | absent | audit §1 row 10 |
| live vLLM | GPU compute process on 4/4, `:8000` on .80 | preflight 16:36Z |

**Estimate/hypothesis, not measured:** raising MTU 1500 -> 9000 removes fragmentation and the
per-packet header overhead; enabling GPUDirect removes the host-staging copy on the device path.
Neither has been measured on this branch. The 2026-10-20 numbers were taken at MTU 1500 with
GPUDirect off on host staging, so they are the correct "before" row, not a prediction.

## 3. Rules that apply to every ticket

1. **No change from this branch.** The comm-lead only writes tickets; an operator with root or a
   switch admin executes them.
2. **The live TP4 vLLM deployment is never stopped, restarted or reconfigured by this work.** A
   change that breaks its connections is scheduled with the serving owner, who owns the restart.
3. **One change at a time, one evidence directory.** Each ticket's acceptance is captured before
   the next ticket starts, so a later failure has one suspect.
4. **Persist it or it did not happen.** Host settings go through netplan/modules-load.d, not just
   `ip link`/`modprobe`, and the persistence file is part of acceptance.
5. **Rollback is rehearsed before the change.** Back up the file, print the exact rollback, keep
   the backup with the evidence.
6. **Read-only checks only before the window.** `sparkflash doctor --local --json` (without
   `--apply`), `ping -M do`, `ibv_devinfo`, `ss`, `cat /sys` are safe; they change nothing.

### 3.1 Operational blocker: who holds the GPUs

There is a second owner of these nodes, and it is us. This session's own agent harness is served
by the local TP4 vLLM deployment on `192.0.2.80:8000` (the Lead reports that the node-MainThread
PID on this host holds the GPU streams). A "free window" is therefore **not** "vLLM is stopped":
stopping vLLM cuts the model backend of every agent session coordinating the work, including the
one that would collect the evidence.

- **Authorisation:** stopping or restarting vLLM needs explicit human authorisation, not a
  teammate's judgement and not "the fabric looked idle". Do not run `docker stop`/`restart` on
  your own initiative.
- **Reported stop/restore path (Lead's report 2026-10-21; not measured by this branch):** stop
  with `docker stop vllm_node` on all four nodes; restore with
  `python3 run-recipe.py deepseek-v4.1-flash-1017-heretic` on `.80`; allow ~13 minutes to
  health. Confirm container name, recipe path and health check against the serving owner's
  runbook before use - the timings are the Lead's, not a measurement here.
- **Sequencing consequence:** a window needs at least two humans (one to authorise/restore
  serving, one to run the benchmark and collect), and the session that measures loses its model
  backend the moment serving stops.
- **What the runner does about it:** `scripts/ws-b-allreduce-bench.sh preflight` detects a GPU
  compute process or a `:8000` listener and refuses to launch unless
  `SPARK_WSB_ALLOW_BUSY=1` is deliberately set; the script never stops, restarts or kills vLLM
  and its cleanup only matches `ws-b-allreduce-bench`. That refusal is the correct behaviour, and
  it is why no fabric number exists today: the alternative would be to stop the service that
  serves this session.

## 4. B1 - MTU 9000 end to end (hosts and switch)

**Blocker class:** #1 according to the audit. Without it the one-shot path fragments at decode
sizes and the 2 MiB prefill payload on the same wire is impossible without IP fragmentation.

**Order:** switch first, then hosts. Setting hosts to 9000 while the switch still drops >1500
silently black-holes large packets, including vLLM's control traffic.

### B1.1 Switch half (blocked on switch admin; switch model is not recorded in this repo)

The switch model and its configuration are **not** in this repository or the audit captures, so
the exact vendor command cannot be written from here. Required settings on the four cluster ports:

- jumbo frames: L2 MTU 9216 (or at least 9000) on the four server-facing ports and on the bridge /
  VLAN they belong to;
- 200G per port (they already negotiate 200G on f1, measured 2026-10-20);
- lossless RoCEv2: PFC enabled on the relevant priority and ECN marking configured, or the RoCEv2
  traffic is not lossless and a QP timeout under burst is expected;
- no MTU mismatch between the server ports and any inter-switch/uplink path the traffic crosses.

Copy-pasteable discovery for the switch admin (run on the switch, read-only):

~~~text
# identify model/firmware first; the command syntax below is vendor-specific
show version
show interfaces status
show running-config
~~~

If the switch is the MikroTik CRS804-4DDQ / CRS812 from the spec, RouterOS needs jumbo MTU on both
the ethernet interfaces and the bridge (confirm the model before running this; the exact interface
names are the four downlinks):

~~~text
/interface ethernet set [find name=etherX] mtu=9216 l2mtu=9216
/interface bridge set [find name=bridge] mtu=9216
~~~

**Acceptance (switch):** after the host half below, all ordered pairs pass the DF-8972 probe and
no port reports frame errors/drops attributable to MTU. A failed probe is a switch MTU problem, not
a host MTU problem, until the host readings are 9000 on both rails.

### B1.2 Host half (root, four nodes)

Preflight, read-only (run from any host; all four addresses):

~~~bash
for ip in 192.0.2.80 192.0.2.88 192.0.2.59 192.0.2.68; do
  echo "=== $ip"
  ssh -o BatchMode=yes allenh1`$ip '
    for d in enp1s0f1np1 enP2p1s0f1np1; do printf "%s mtu=" "$d"; cat /sys/class/net/$d/mtu; done
    ip -br link | grep -E "enp1s0f1np1|enP2p1s0f1np1"
    sudo -n true 2>/dev/null && echo sudo=ready || echo sudo=needs-password'
done
~~~

Read the current config before editing it (root; this branch could not read it, it is mode 0600):

~~~bash
sudo cat /etc/netplan/40-cx7.yaml
~~~

The file exists on all four nodes and a `40-cx7.yaml~` backup already sits beside it, which means
it has been hand-edited before: take a fresh timestamped backup even so.

Change (on each node, in the window, rail-2 netdev first because the deployed vLLM recipe uses
rail 1 - `NCCL_IB_HCA=rocep1s0f1`, measured in `docs/build/rdma-bringup.md` §2):

~~~bash
# 0. back up the file that is actually in force
sudo cp -a /etc/netplan/40-cx7.yaml "/etc/netplan/40-cx7.yaml.pre-mtu-$(date -u +%Y%m%dT%H%M%SZ)"

# 1. add "mtu: 9000" to each of the four ethernet blocks (enp1s0f0np0, enp1s0f1np1,
#    enP2p1s0f0np0, enP2p1s0f1np1) in /etc/netplan/40-cx7.yaml, then:
sudo netplan generate
sudo netplan apply

# 2. if netplan is not the effective renderer for these interfaces (both networkd and
#    NetworkManager are active on these nodes), the runtime form is:
sudo ip link set dev enP2p1s0f1np1 mtu 9000
sudo ip link set dev enp1s0f1np1 mtu 9000
# ...and then persist the same value in the file that renders the interface.

# 3. confirm and persist check
for d in enp1s0f1np1 enP2p1s0f1np1; do printf "%s mtu=" "$d"; cat /sys/class/net/$d/mtu; done
sudo netplan get ethernets 2>/dev/null | grep -A1 mtu || true
~~~

**Acceptance (hosts, measured on the node, all four):**

~~~bash
# 1. runtime MTU on both rails is 9000, persistently
for d in enp1s0f1np1 enP2p1s0f1np1; do printf "%s mtu=" "$d"; cat /sys/class/net/$d/mtu; done
# expect: mtu=9000 for both on all four nodes

# 2. RDMA negotiated MTU rises from 1024 to 4096 (max_mtu, measured 4096 today)
ibv_devinfo -d rocep1s0f1 -v | grep -E 'max_mtu|active_mtu'
ibv_devinfo -d roceP2p1s0f1 -v | grep -E 'max_mtu|active_mtu'
# expect: active_mtu: 4096 on both rails

# 3. DF-8972 succeeds on every ordered pair of nodes, both rails
#    rank addresses: .80 .88 .59 .68 (rail-1); peers must be reachable on the rail under test
ping -M do -s 8972 -c 3 192.0.2.88
ping -M do -s 8972 -c 3 192.0.2.59
ping -M do -s 8972 -c 3 192.0.2.68
# expect: 3/3 received, "Frag needed and DF set" must NOT appear

# 4. vLLM is still healthy (read-only; never restart it from this ticket)
curl -sS -o /dev/null -w '%{http_code}\n' http://192.0.2.80:8000/v1/models
# expect: 200. If this fails after an MTU change on rail 1, the serving owner restarts in the
# window; this branch does not.
~~~

**Rollback (immediate, no reboot):**

~~~bash
# runtime rollback on each node
sudo ip link set dev enP2p1s0f1np1 mtu 1500
sudo ip link set dev enp1s0f1np1 mtu 1500

# persistence rollback
sudo cp -a /etc/netplan/40-cx7.yaml.pre-mtu-<timestamp> /etc/netplan/40-cx7.yaml
sudo netplan apply

# verify the old state: MTU 1500 and DF-1472 succeeds, DF-8972 fails again
for d in enp1s0f1np1 enP2p1s0f1np1; do printf "%s mtu=" "$d"; cat /sys/class/net/$d/mtu; done
ping -M do -s 1472 -c 3 192.0.2.88
~~~

**Risk to the live service:** changing MTU on a netdev carrying vLLM's NCCL/socket path resets or
breaks its established connections. Rail 2 is the safer first target only if it carries no live
traffic - verify with `ss`/`ip -s link` before assuming that. The window must include the
serving owner's restart plan; the comm-lead never restarts vLLM.

## 5. B2 - Load `nvidia-peermem` so GPUDirect RDMA is on

**Blocker class:** #2. Without it a GPU buffer cannot be registered with the HCA, so the
device-buffer one-shot seam (B6) cannot be tested and every GPU-side collective pays host staging.

**What is already true (measured):** `modinfo nvidia-peermem` resolves to
`/lib/modules/7.0.0-1019-nvidia/updates/dkms/nvidia-peermem.ko.zst`, version 580.178.04 on all
four nodes. It is simply not loaded. `/etc/modules-load.d/` exists with `modules.conf` and
`nvidia-mstflint.conf`; there is no peermem entry.

Preflight, read-only:

~~~bash
lsmod | grep -E '^nvidia_peermem' || echo 'not loaded'
ls -l /sys/module/nvidia_peermem 2>/dev/null || echo 'no sysfs entry'
dmesg | grep -i peermem | tail -5
~~~

Change (root, each node, in a window):

~~~bash
sudo modprobe nvidia-peermem

# persist across reboot: the module name has a hyphen; modules-load.d uses the same name
echo nvidia-peermem | sudo tee /etc/modules-load.d/nvidia-peermem.conf

# confirm
lsmod | grep -E '^nvidia_peermem'
cat /etc/modules-load.d/nvidia-peermem.conf
dmesg | grep -i peermem | tail -5
~~~

**Acceptance:**

~~~bash
# phase 1 (module): loaded on 4/4 nodes
for ip in 192.0.2.80 192.0.2.88 192.0.2.59 192.0.2.68; do
  ssh -o BatchMode=yes allenh1`$ip 'lsmod | grep -E "^nvidia_peermem" || echo MISSING'
done
~~~

**Phase 2 (real GPUDirect data path) is blocked on the device-buffer seam (B6).** A loaded module
is not proof that a GPU pointer registers: the proof is an `ibv_reg_mr` on a CUDA allocation
succeeding and a peer reading it, which is exactly what the S5.3 device gate has to do. Do not
claim GPUDirect is working from `lsmod` alone.

**Rollback:**

~~~bash
sudo rm -f /etc/modules-load.d/nvidia-peermem.conf
sudo modprobe -r nvidia-peermem   # may fail EBUSY while GPU MRs are registered; report, do not force
~~~

**Risk to the live service:** the load does not restart vLLM and new MR registrations are the only
behaviour change, but the module touches the GPU driver, so schedule it in a window. Unloading
while anything holds a peer-memory registration fails; that is a reportable state, not something
to force.

## 6. B3 - f0 twins DOWN at 40 Gb/s vs f1 rails ACTIVE at 200 Gb/s

**Measured:** `rocep1s0f0` and `roceP2p1s0f0` are `1: DOWN` on all four nodes today (read-only
preflight 16:36Z; same in the audit and the 2026-10-20 bring-up). The f1 twins are `4: ACTIVE` at
200 Gb/s and every measured collective to date ran on f1. The deployed vLLM recipe selects
`rocep1s0f1` (one rail). The spec names `rocep1s0f0`/`roceP2p1s0f0` in places, so code,
docs and reality disagree.

**Decision A - standardise on the f1 rails (recommended; no root, no window).**
Owner: Lead/spec owner. Scope: spec, `sparkflash doctor` rail table, and any code comment that
calls f0 the spec rail; record f0 as "not cabled/not used on this cluster" with today's measured
state. Acceptance: no runnable path selects f0; the doctor's rail check names the f1 twins; a
read-only capture shows them ACTIVE 200G. Rollback: revert the doc/spec commit, no node impact.
Lead time: one doc commit (estimate: under a day).

**Decision B - bring f0 up at 200G.** Blocked on root + switch admin + physical inspection.
Read-only diagnosis first (no change):

~~~bash
for ip in 192.0.2.80 192.0.2.88 192.0.2.59 192.0.2.68; do
  ssh -o BatchMode=yes allenh1`$ip '
    for d in rocep1s0f0 roceP2p1s0f0; do
      printf "%s state=" "$d"; cat /sys/class/infiniband/$d/ports/1/state
      printf "%s phys=" "$d"; cat /sys/class/infiniband/$d/ports/1/phys_state
      printf "%s rate=" "$d"; cat /sys/class/infiniband/$d/ports/1/rate
    done
    ip -br link | grep -E "enp1s0f0np0|enP2p1s0f0np0"
    dmesg | grep -iE "mlx5.*(f0|link down)" | tail -5'
done
~~~

Change class: re-seat/replace the f0 cables and enable the switch ports (physical/admin, cannot be
scripted from here); if the netdev is administratively down, `sudo ip link set dev <f0 netdev> up`
is the only host-side action. Acceptance: both f0 twins `4: ACTIVE` at 200 Gb/s on all four,
`ibv_devinfo` reports them, and the doctor rail check passes with four rails. Rollback: return
the f0 cabling/ports to today's state; f1 is untouched so the service is unaffected either way.

## 7. B4 - host-doctor items

### B4a SM clock cap (blocked on a spec decision, then root)

**Measured:** `clocks.max.sm` = 3003 MHz on all four nodes; current 2437-2496 MHz.
`sparkflash doctor` check 4 fails because `MAX_SM_CLOCK_MHZ = 2200` and the check compares the
queried maximum (3003) to it (`crates/sparkflash-cli/src/doctor.rs`).
`nvidia-smi --query-gpu=clocks.min.sm` is **not a valid field on this driver** (measured
2026-10-21 12:37 EDT), so the check falls back to `clocks.max.sm`.

The obvious root command, `sudo nvidia-smi -lgc 0,2200`, sets the clock *policy*; it does not
change the `Max Clocks` row that `clocks.max.sm` reports (that row is a hardware limit). So no
node change can satisfy the current check. **This is a check/spec mismatch, not a node defect.**

Decision needed (spec owner): either the check reads the configured applications-clock range
(`nvidia-smi -q -d CLOCK` "Applications Clocks" 2418 MHz today) or the spec accepts the
hardware max and the pass criterion becomes "a lock at or below 2200 was requested and accepted".
After that decision, if a lock is wanted: `sudo nvidia-smi -lgc 0,2200` (root, window; it is
cluster-wide and changes vLLM's clock policy), rollback `sudo nvidia-smi -rgc`. Acceptance is then
the corrected check, captured read-only. **Do not run the lock before the decision** - it changes
serving behaviour for a check that still fails.

### B4b nvidia-uvm 64K packing (blocked on the spec owner; no node command exists)

The spec requires "nvidia-uvm 64K page packing" but names no knob, and the 580.178.04 driver on
these nodes exposes no matching parameter: the doctor enumerates
`/sys/module/nvidia_uvm/parameters` (56 parameters seen under 580-series) and found none matching
`*pack*` or `*64k*`, and `/proc/driver/nvidia/params` has no such line (measured; the probe
reports `Unknown`). There is no root command to write down. Blocked on the spec owner defining
the knob or dropping the claim; the ticket is a decision, not an infrastructure change.

### B4c Engram shards (root, or a writable output dir the doctor reads)

**Measured:** `/var/lib/sparkflash/engram/manifest.json` does not exist; check 5 is
not-applicable. The packer is in-tree and its output path is the doctor's
`DEFAULT_ENGRAM_DIR = /var/lib/sparkflash/engram`.

The binary today is the standalone `pack-engram` (`crates/sparkflash-engram/src/bin/pack_engram.rs`; the
spec calls it `sparkflash pack-engram` but the CLI subcommand is not wired yet, so build the bin):

~~~bash
cargo build --release -p sparkflash-engram --bin pack-engram

# dry-run first: reads the model index, opens no shard, prints the plan and free-space need
./target/release/pack-engram \
  --index /cache/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/<snapshot>/model.safetensors.index.json \
  --dry-run

# real run (root for the default dir, or --output-dir under a writable path plus a doctor override)
sudo ./target/release/pack-engram \
  --index <same index> --world-size 4 --layers 1,14 --verify
~~~

**Acceptance:** `manifest.json` exists, every listed shard has the recorded sha256
(`--verify` prints the count), and `sparkflash doctor --local --json` reports
`engram-shards: pass`. **Rollback:** `sudo rm -rf /var/lib/sparkflash/engram`. **Risk:** disk
fill and node-local I/O; run it on an idle node or before the serving window, never during a
latency run. Exact shard size is a measured-after-run number, not estimated here (the spec table
budgets ~1.8 GiB Tier-1 DRM plus ~4 GiB Tier-2 coherent cache, but the packed files are row copies
of the layer-1/layer-14 tensors and the dry-run is what states the byte count).

## 8. B5 - Host NCCL 2.28+ (`libnccl.so.2`)

**Why:** the large-tensor dispatch path (>2 MiB, chunked prefill) routes to the patched NCCL ring.
Today NCCL 2.28.3 exists only inside `vllm-node:latest`; the host has none (measured 0 hits in
`ldconfig -p` on all four). `sparkflash-comm` now has the dlopen binding
(`crates/sparkflash-comm/src/nccl.rs`, NCCL 2.28 floor, no build-time dependency) and it refuses
with a named error when the library is absent, so the gap is visible rather than silent.

**Change (root, each node):** install an NCCL 2.28.x host library matching the container's ABI on
the host (distribution package, or the libnccl shipped for the 580/CUDA-13 stack), then
`sudo ldconfig`. Do not copy libraries out of the running container in place and do not put a
different major/minor on the host than the image's 2.28.3 without recording the difference: the
dispatch contract claims only "NCCL 2.28+", not a version.

**Acceptance (read-only, no GPU needed):**

~~~bash
ldconfig -p | grep libnccl.so.2
target/release/ws-b-allreduce-bench --check-nccl
# expect: library path + version 2.28.x, exit 0; today this exits 3 with the image/ticket message
~~~

**Rollback:** remove the package (or the `/etc/ld.so.conf.d` entry) and `sudo ldconfig`. **Risk:**
a host library is inert until a process dlopens it; this ticket does not change the serving
container. A first real NCCL collective needs device buffers and is B6, not this ticket.

## 9. B6 - After the infra tickets: the device seam and the first qualifying run

Not an infra ticket, but the dependency that makes B1/B2/B5 measurable. Sequence:

1. device-lead lands the device-buffer one-shot seam (S5.3): `Transport::all_reduce_bf16_device`
   for the RDMA backend, replacing today's `CommError::Unsupported`.
2. B1 + B2 accepted (MTU 9000, peermem loaded): device-buffer probe/gate can run.
3. `scripts/ws-b-allreduce-bench.sh run` in a free window; the report must say
   `fabric=true` and the gate is p50 <= 25 us on the decode rows. The bench is
   not-yet-measured today; no number may be quoted from a loopback run.
4. If the gate misses, the measured gap and the next single hypothesis go in
   `docs/build/ws-b-collectives-20261021.md` section 9; one change at a time.

## 10. Blocked-on lists (for the Lead's integration plan)

**Blocked on root, four nodes:** B1 host MTU (after switch), B2 peermem, B4c Engram (unless the
doctor is given a writable dir), B5 host NCCL.

**Blocked on switch admin:** B1 switch jumbo/PFC/ECN (model unknown), B3-B f0 ports/cabling.

**Blocked on Lead/spec owner (no root):** B3-A rail naming decision; B4a clock-check definition;
B4b uvm-64K claim.

**Blocked on device-lead:** B6 device-buffer seam; real GPUDirect proof (B2 phase 2).

**Blocked on explicit human authorisation:** any step that stops or restarts vLLM, including the
measurement window itself (B6 step 3), because that deployment also serves this session's agent
harness (section 3.1).

**Safe to do now without touching a node:** B3-A (doc/code rename), B4a/B4b decisions, and all
read-only captures. Nothing else in this document.

## 11. Read-only verification block (safe any time, no changes)

~~~bash
ssh -o BatchMode=yes allenh1`192.0.2.80 '
  echo "=== fabric"
  for d in enp1s0f1np1 enP2p1s0f1np1; do printf "%s mtu=" "$d"; cat /sys/class/net/$d/mtu; done
  for d in rocep1s0f1 roceP2p1s0f1 rocep1s0f0 roceP2p1s0f0; do
    printf "%s state=" "$d"; cat /sys/class/infiniband/$d/ports/1/state || true
  done
  echo "=== peermem"; lsmod | grep -E "^nvidia_peermem" || echo absent
  echo "=== nccl host"; ldconfig -p 2>/dev/null | grep -c libnccl.so.2
  echo "=== vllm"; ss -ltnH "sport = :8000" | wc -l
  echo "=== clock"; nvidia-smi --query-gpu=clocks.max.sm,clocks.current.sm --format=csv,noheader
'
~~~

Expected today: MTU 1500/1500, f1 `4: ACTIVE`, f0 `1: DOWN`, peermem absent, host NCCL 0,
`:8000` listener 1, max SM 3003 MHz. Anything else means a ticket moved; record it with a
timestamp before believing it.
