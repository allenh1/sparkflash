# WS-B: apply MTU 9000 + nvidia-peermem (2026-10-21, root access granted)

State verified on 4/4 nodes before any change (comm-lead survey, `verify-20261021-1747EDT.txt`):

| item | state |
| :-- | :-- |
| rail links | `enp1s0f1np1` (192.0.2.x0), `enP2p1s0f1np1` (192.0.2.x5), both UP |
| RDMA devices | `rocep1s0f1`, `roceP2p1s0f1` ACTIVE at 200 Gb/s; f0 ports DOWN |
| netdev MTU | 1500 on every link (maxmtu 9978) |
| RDMA active_mtu | 1024 / 4096 |
| `nvidia-peermem` | module present at `/lib/modules/7.0.0-1019-nvidia/updates/dkms/nvidia-peermem.ko.zst` (580.178.04), not loaded |
| DF-8972 probe | fails "frag needed" |

Switch (LLDP, `lldpcli show neighbors -f keyvalue`): **MikroTik CRS812-8DS-2DQ-2DDQ**, RouterOS
7.20.7, mgmt 192.0.2.92, bridge enabled, port `qsfp56-dd-2-1`. Jumbo therefore needs the bridge
MTU raised to 9000 on the switch, otherwise host-side 9000 frames are dropped by the bridge.

## Order of operations (host side)

1. Load the peer-memory module and persist it (safe, reversible):
   `modprobe nvidia-peermem`, `printf 'nvidia-peermem\n' > /etc/modules-load.d/nvidia-peermem.conf`,
   verify `lsmod | grep nvidia_peermem` and that `nvidia-smi -L` is unchanged.
2. Persist MTU in netplan `/etc/netplan/40-cx7.yaml`: add `mtu: 9000` under each of the four
   ethernet blocks (dhcp4 keys stay). Keep `40-cx7.yaml~` as the restore copy.
3. Runtime first, one pair only: `ip link set dev enp1s0f1np1 mtu 9000` on .80 and .88, then
   `ping -M do -s 8972` both directions on both rails. Only after that, all four nodes and
   `netplan apply` (with a rollback timer armed).

## Rollback

`ip link set dev <if> mtu 1500`; `rm /etc/modules-load.d/nvidia-peermem.conf`; `rmmod nvidia_peermem`;
restore `40-cx7.yaml` from `40-cx7.yaml~` and `netplan apply`.

## What this is measured against

Last evidence (WS-B B2, host buffers, GPUDirect off): 10 KiB p50 36.0-37.2 us dual rail / 29.5-29.6 us
single, 2 MiB 8.76 ms, **no GB/s number exists yet**. The apply is only worth keeping if a re-run of
the one-shot probe shows a real jump and `active_mtu` reports 9000 (4096 is the current ceiling).
