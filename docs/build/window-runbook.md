# vLLM window runbook (Lead-owned)

Status: recorded 2026-10-18 from the Lead's window protocol. The window itself is **Lead-owned**: the
Lead stops and restarts the reference vLLM instance; teammates ask for a window and get one, they do
not stop the server. This file exists so the protocol survives the window: a restart that is
improvised from memory is how a live client loses its server.

## Why the window has to be short and why it always restarts

A live client at `192.0.2.71` uses the vLLM server at `http://192.0.2.80:8000`. While vLLM is
stopped, that client has no model. So:

* the window is as short as the evidence run needs (device work is prepared and staged on the nodes
  before the stop);
* **the server is always restarted at the end of the window**, from the recorded launch command, and
  the health check below is run before the window is called finished - not left to the next person.

## The canonical launch recipe

**Update 2026-10-19 (D38 actioned, D39):** the canonical recipe now serves the **released** checkpoint. /home/user/phase2-evidence/window-2026-10-18/vllm-launch-recipe-released.sh (sha256 a3ac461fa02a945181a17b509f7695ca39b38930eee529a0f9fd9151061072c4) is the previous served recipe (vllm-launch-recipe-heretic-12d58ad0.sh, sha256 2a3e2c7719532149d3291d11964ace46bf27d9a0b656c39258d5a6295c552574, retained) with the model path switched to deepseek-ai/...-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170 and --max-logprobs 1000 added for the numeric comparison; window-vllm.sh (sha256 a3ba6763cc8f5087ab65dc7771e28e37e9b3a18842d2e72542c3f22f7344ef70) starts it. The served root is recorded from /v1/models in every evidence log. The old paragraph below is kept as history.

Preserved at
`/home/user/phase2-evidence/window-2026-10-18/vllm-launch-recipe.sh` (1,809 B, sha256
`2a3e2c7719532149d3291d11964ace46bf27d9a0b656c39258d5a6295c552574`), generated from the recipe
`deepseek-v4.1-flash-1017-heretic`. It exports the model/NCCL environment and ends in the
`vllm serve` command; the NCCL part is single-rail (`NCCL_IB_HCA=rocep1s0f1`, see
`rdma-backend.md`, section 3, for why the launcher's twin-pair value does not win).

## Start (the Lead's command)

On the head node (`node0`), from `/home/user/spark-vllm-docker/` - the launcher directory
lives on the node, not on the dev host - the recorded start is:

```sh
./launch-cluster.sh --apply-mod deepseek-v4.1-flash-1017-heretic \
  -n 192.0.2.80,192.0.2.88,192.0.2.59,192.0.2.68 \
  --eth-if enp1s0f1np1 --ib-if rocep1s0f1,roceP2p1s0f1
```

The flag spelling and the `--apply-mod` recipe name are as reported by the Lead from the 2026-10-18
window; the launcher script is not on this dev host, so this file cannot re-check its usage text. The
preserved recipe is the canonical environment (`--apply-mod` is what regenerates it); if the launcher
and the recipe disagree on `NCCL_IB_HCA`, the recipe's export is what the running container shows.

## Stop (the Lead's action)

The `launch-cluster.sh` `stop` action, which runs `docker stop vllm_node` on all four nodes. Two
practical points:

* drain/quiet the client first if the window is long; the client at `192.0.2.71` fails from the
  moment the server stops;
* the device evidence run must be staged before the stop, because staging over the network while the
  server is down is window time wasted.

## Restart and health check

Restart with exactly the recorded start command above (not a hand-typed variant: `vllm serve`
parameter drift is a silent behaviour change). Then:

```sh
curl -fsS -o /dev/null -w '%{http_code}\n' http://192.0.2.80:8000/v1/models
```

Expect `200`. The window is finished only after this returns 200 and the client's next request
succeeds; if the health check fails, the window is not over and the server is restarted again.

## What this file does not do

* It does not automate the stop/start from the dev host; the dev host has no access to the nodes.
* It records no timings or throughput of the vLLM instance (`rdma-backend.md` and the qualification
  gates own those questions).
* The exact `launch-cluster.sh` usage text and any other `--apply-mod` recipes are on
  `node0`; this runbook records the one window command the Lead reported, not a catalogue.
