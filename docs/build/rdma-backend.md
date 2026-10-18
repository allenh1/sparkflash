# RDMA rails, addresses and the deployed reference

Status: **read-only measurement, not a gate run.** The rail/IP facts below were read on 2026-10-18
from the live cluster by the Lead (no ssh and no cluster action is permitted from this dev host), and
the device/NCCL facts were read from the live vLLM container on `node0` on the same day. Nothing
in this file is a bandwidth, latency or dual-rail result: the qualification gates that measure those
are still `never run` (see `docs/QUALIFICATION.md` section 3).

Why this file exists: `docs/QUALIFICATION.md` 2.1 previously said the rail-2 addresses were "not
recorded anywhere in this tree", and D12 recorded only the head node's pair (`192.0.2.80` /
`192.0.2.85`). A dual-rail gate cannot be planned against unrecorded addresses, so the measurement
is written down here with its provenance, and QUALIFICATION/D12 now point at it.

## 1. Address table

Each node holds one address per rail; both rails share `192.0.2.0/24`.

| node | rail 1 netdev (`rocep1s0f1`) | rail 1 IP | rail 2 netdev (`roceP2p1s0f1`) | rail 2 IP |
| :--- | :--- | :--- | :--- | :--- |
| node0 | enp1s0f1np1 | 192.0.2.80 | enP2p1s0f1np1 | 192.0.2.85 |
| node1 | enp1s0f1np1 | 192.0.2.84 | enP2p1s0f1np1 | 192.0.2.88 |
| node2 | enp1s0f1np1 | 192.0.2.64 | enP2p1s0f1np1 | 192.0.2.68 |
| node3 | enp1s0f1np1 | 192.0.2.59 | enP2p1s0f1np1 | 192.0.2.63 |

Provenance: Lead's read-only measurement of the live cluster, 2026-10-18 (interface/address listing
on each node). The head-node pair agrees with the earlier D12 measurement; the other three rows are
the new information. No artifact from that listing is stored in this tree (the dev host cannot reach
the nodes), so this table is labelled as a measurement report, not as a re-runnable gate.

The running vLLM instance uses four of these addresses: `192.0.2.80` (master), `192.0.2.88`,
`192.0.2.59`, `192.0.2.68`. That is the instance's own node list, not a rail assignment.

## 2. Link state

Both rails' **f1** functions are ACTIVE at 200000 Mb/s; the **f0** siblings the specification names
(`rocep1s0f0/enp1s0f0np0`, `roceP2p1s0f0/enP2p1s0f0np0`) are DOWN at 40000 Mb/s. This is the
discrepancy D12 records, and it is visible in the dev host's own doctor JSON:
`/home/user/phase2-evidence/window-2026-10-18/doctor-local-2026-10-18.json` lines 33-43 list
`roceP2p1s0f1/enP2p1s0f1np1 ACTIVE 200000 Mb/s` and `rocep1s0f1/enp1s0f1np1 ACTIVE 200000 Mb/s`, with
the f0 functions DOWN and the f1 functions on the same PCI devices. MTU is reported as 1500 on the
active netdevs, not the required 9000.

Devices and netdevs:

| rail | RDMA device | netdev | state |
| :--- | :--- | :--- | :--- |
| rail 1 | rocep1s0f1 | enp1s0f1np1 | ACTIVE, 200000 Mb/s |
| rail 2 | roceP2p1s0f1 | enP2p1s0f1np1 | ACTIVE, 200000 Mb/s |
| spec's rail 1 | rocep1s0f0 | enp1s0f0np0 | DOWN, 40000 Mb/s |
| spec's rail 2 | roceP2p1s0f0 | enP2p1s0f0np0 | DOWN, 40000 Mb/s |

## 3. The deployed vLLM reference is single-rail

Measured 2026-10-18 from the live container on `node0` (`docker exec vllm_node env`): the
environment contains

```text
NCCL_IB_HCA=rocep1s0f1
NCCL_SOCKET_IFNAME=enp1s0f1np1
NCCL_IB_GID_INDEX=3
NCCL_IB_DISABLE=0
```

and **only one HCA**, `rocep1s0f1`. The preserved launch recipe
`/home/user/phase2-evidence/window-2026-10-18/vllm-launch-recipe.sh` (1,809 B, sha256
`2a3e2c7719532149d3291d11964ace46bf27d9a0b656c39258d5a6295c552574`) sets the same value on line 10.
The cluster launcher (`launch-cluster.sh` on `node0`) passes
`-e NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1`, but the recipe's own export runs later inside the container
and wins - the live container shows the single-rail value. So the deployed instance carries its
traffic over **one** rail; the twin `roceP2p1s0f1` is link-ACTIVE but not selected by NCCL.

Consequence for our gates: the deployed reference is **single-rail**, while our specification and
qualification target a **dual-rail** aggregate (spec lines 190, 1322, 1323: both halves active,
aggregate bandwidth). The two are not interchangeable. A single-rail number from the vLLM instance
cannot be compared with the dual-rail bar, and our dual-rail gate must exercise both ACTIVE f1
functions on every node rather than copy the deployed environment.

## 4. What is not claimed

* No bandwidth or latency was measured for this note; the f1 link speed is the device's reported link
  rate, not an achieved throughput.
* The address table is read-only configuration, not a connectivity test: nothing here proves that a
  given pair of addresses exchanges packets.
* The MTU and driver/kernel gate failures remain exactly as recorded in D12 and QUALIFICATION section
  2.3; this note does not re-run them.
