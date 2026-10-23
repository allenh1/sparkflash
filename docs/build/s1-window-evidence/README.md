# S1 device-arm window evidence (2026-10-23)

The rows below decided D55 and the state in [`../s1-result.md`](../s1-result.md). Following
[`../evidence.md`](../evidence.md), **the raw dumps are not committed**: each row is a ~20 MB tree of
per-rank logs, `gpu.csv` clock/power samples, bind lists, the forward-progress records and the f32
logits dumps. Their node paths are recorded here; the fixtures the gates read are in
[`../reference/speed-gate/`](../reference/speed-gate/) and are small enough to live in the tree.

All rows ran on the four DGX Spark nodes (`node0` 192.0.2.80, `node1` .88,
`node3` .59, `node2` .68), one rank per node, TP=4, decode M=1, from the branch build
staged by `scripts/build-aarch64.sh --features=cuda` (aarch64, `sm_121a`). Row directories are
`~/ev-session-window/s1w17/<row>/` on each node.

| row | arm | what it is |
| :-- | :-- | :-- |
| `s1-hostq` | host | the host arm on the 138-token teacher-forced list, 9 decode positions, 1500 MHz clock lock (243.08 ms mean / 244.00 ms median); frozen as the `b08cbf341ad85320` reference |
| `s1-c-rep1` (`s1-c`) | device | the S1 40-layer device arm replicate 1 under `nvidia-smi -lgc 1500,1500` (202.67 ms mean / 191.75 ms median, `d2h=1 alloc=0 sync=0`, GPU 1462-1488 MHz, 8.84-11.43 W, D56 gate PASS, logits sha256 `12a03a861bb6f4644e65a525bb55870c6b528581c28845a371c48dcd4ed17623`) |
| `s1-c-rep2` | device | the S1 40-layer device arm replicate 2 under `nvidia-smi -lgc 1500,1500` (202.75 ms mean / 193.50 ms median, within 2 ms of `s1-c-rep1`, `d2h=1 alloc=0 sync=0`, GPU 1462-1488 MHz, 8.89-11.53 W, D56 gate PASS, byte-identical logits sha256 `12a03a861bb6f464...`) |
| `s1-c-probe` | device + `--forward-perf-probe-nosync` | the S1 40-layer per-stage deferred CUDA event timing probe under `nvidia-smi -lgc 1500,1500` (`tp4-forward-perf-r0.json` sha256 `e0ca57387f03bd32ac0cb427eb66dcc95bdcfcc8fa8a4bd8d34ccf1af8f60c79`, byte-identical logits sha256 `12a03a861bb6f464...`) |
| `s1-c1-boost` | device | the 40-layer device step at unlocked boost clocks (2405 MHz, 142.94 ms mean / 135.00 ms median, 134-140 ms steady-state) |
| `s1-c2` | device | the earlier coalesced `layer_seams_sm121` step under `nvidia-smi -lgc 1500,1500` (203.00 ms mean / 191.00 ms median, GPU clock 1469-1482 MHz) |
| `s1-devq` | device | the serial-seam experiment on the same list (338.67 ms mean / 320.00 ms median; rejected for +127 ms/step regression, D56) |
| `s1-host` | host | the earlier host arm on the 129-token list, frozen as `fab52c00a0dc6f6c`; host floor 0.0 on every metric |
| `s1-ord3` | device | per-layer boundary diffs, layers 0-5, positions 0-5: the layer-0 attribution |
| `s1-rdx3` | device + `SPARKFLASH_REDUCE_DUMP` | the four wo_b partials, replayed through the D51 canonical reduce |
| `s1-diff-full` | device + `SPARKFLASH_DEVICE_DIFF` | the full 40-layer per-stage diff sweep |

Binary identity: `s1-c-rep1`, `s1-c-rep2`, and `s1-c-probe` ran staged binary
`1f80580179b493da6e4a64339117117d6f5ffcd57bbfc156935344ac60103acc` with pinned cubin set `nc1`
(`layer_seams_sm121.cubin` `e3e1b4821da9bfafcdefce73d325ab0cf2ef85226020c8d069cdf880c310ad06`,
`pool_group_sm121.cubin` `ed16a3a00d1150ea7b32de59c9dc7f44011faa8a8a45cb3ea518c1a84c6175b7`).
`s1-hostq` and `s1-devq` ran staged binary
`6b0bd4c7dc08193bf7aad59f12a573bf9116dae54b699f4929ff4380828cccf3` (aarch64 release, branch
`integration/s1-layer-device-attn-moe`). Each
earlier row ran the binary staged for its own window; the sha256 is in the row's staging log on the
node (the build script prints it and the stage script copies that line into the window record).

Reproduce the gate that reads these rows:

```bash
scripts/numerical-gate.sh --run <device-row> --host-baseline <host-row>
```

with the row directories copied back from the nodes to the same relative paths under
`docs/archive/build/` (they are not in the tree, so a gate run needs the rsync first).
