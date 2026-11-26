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

# The loop on the device walk (Inkling-2-Mini)

The host reference runs the looped release today (`docs/models/inkling-2-mini.md`). This note
is the device plan: what the arm's schedule has to become so the same arithmetic runs on the
GB10 at .100.

## What the host path proved

`Inkling2Config::applications()` returns `(stored layer, pass)`; the host forward runs that
list, saves the prelude's hidden state and summary stack, rewinds to them on every pass after
the first, and computes each core layer's four messages from the previous pass's summary slice
plus its final hidden. On the real checkpoint (`thinkingmachines/Inkling-2-Mini`, revision
`c9d0283`) the f32 reference prefills a three-token prompt and returns finite logits, and the
converter writes one 18.9 GB TP1 rank file with all 828 weights (including the 146 loop
tensors and the drafter).

## The device shape

1. **Applications, not layers.** The plan builds one `LayerSchedule` per application (48 for
   this release), each carrying `stored` (the checkpoint layer whose weights it reads) and
   `cache` (the slot `Inkling2Config::cache_slot` names). The schedules are the host path's
   `applications()` order, so the two cannot disagree.
2. **Caches.** Keys, values and the four convolution histories are addressed by `cache`, not by
   `stored`; a later pass reads its own copy, which is the producer's
   `cache.layers += copy.deepcopy(loop_layers)`. The allocator sizes
   `config.cache_slots()` a rank instead of `num_hidden_layers`.
3. **Summaries.** The summary stack is the producer's `block_states`. A pass after the first
   resets the stack length to the prelude's and overwrites the append slots, exactly as the
   producer does; the previous pass's slice is read before the reset.
4. **Messages.** Two ops before a core application's attention on pass > 0:
   `MessageRead` (the four queries over the previous pass's summaries and final hidden, into a
   per-application message buffer) and `MessageGate(stream)` (the rank-bottlenecked tanh pair
   mixing the message into the attention input and, for the MLP stream, into the MLP input).
   They are elementwise and small GEMVs; the gate's shapes are `[gate_rank, 2*hidden]`,
   `[hidden, gate_rank]` twice a stream.
5. **Rewind.** The row path saves the prelude's hidden row and summary rows and restores them at
   a pass boundary; the prompt path does the same per chunk row. This is the one point the
   runtime's linear layer walk cannot express, so the arm drives it in `RowRun`/`ChunkRun`
   rather than through `PlanHost::branches`.
6. **Drafter and tower** keep their existing schedules: the tower is rank 0 and independent of
   the loop, and the drafter reads the backbone's target-layer taps, which the applications
   list makes explicit (the producer's `target_layer_ids` counts applications, not stored
   layers: 5, 23, 41, 48).

## Order of work

1. **Done.** `LayerSchedule { index, stored, pass }` and the plan builder: the plan now holds
   one schedule an application, per-application caches come from the existing lane memory, and
   the row path saves the prelude's hidden row and rewinds the summary depth at a pass
   boundary. Every single-pass device test still passes on .100, and the looped tiny fixture
   now walks all 48 applications and answers (the messages are the next step).
2. The row path (`RowRun`): applications, rewind, message ops, gates. Compare with the host
   reference on the tiny looped fixture.
3. The prompt path (`ChunkRun`): the same, chunked; the summary slice is per row.
4. The drafter taps move to application indices.
5. Run on .100: device vs the host reference on the real checkpoint for a short prompt, then
   vision and the drafter.

## Gates

* exact ids on the looped tiny fixture at worlds 1, 2 and 4 (the conformance suite);
* device logits against the host reference on the real checkpoint inside a written tolerance
  (the host sums in a different order, so the bar is the one the other arms use);
* the drafter's acceptance and the tower's image answers on the served model.