# Launch-count reductions available WITHOUT a kernel rewrite

One page, requested by the Lead. Every figure is launch-count or recorded-timing arithmetic from
`docs/build/kernel-occupancy-inventory.md` (the measurement anchors are recorded there); no timing
claim is made for any change not yet run. The point of this page is that the 1,168-launch serial
chain has reductions that need no kernel source change, so they do not compete with the occupancy
work for the same window budget.

The chain is 1,168 host launches per position, and the b53 record shows total_chain_ms = 17,452
against device_us = 5,506,333 (5.5 s device): roughly 12 s/position is host launch, staging and
readback serialisation, not kernel time. In order of launches removed:

1. **CUDA graph replay of the whole position** - no kernel or arithmetic change. 1,168 launches
   become one graph launch per position plus graph-internal node dispatch. This is the roadmap's
   "RDMA one-shot + CUDA graphs" stage and the largest single reduction available. It needs
   graph-capture-safe buffers: no per-launch host readback inside the graph (router ids, cert shard
   results and mHC coefficients are read back per layer today and must become device-resident or
   graph nodes).
2. **Remove the per-position H2D staging** (device residency) - not a launch-count reduction, but
   the same record puts staged_matrix_bytes at 16.07 GB/position and staging at 12.8 s; every
   staged weight is one host-side copy per launch. Residency removes the copies, not the launches.
3. **Multi-stream the independent launches** - no kernel change. The eight gate/up launches per
   layer are independent (same activation, disjoint weights and outputs; m17a's header states
   this), as are the five attention projections and the four rank-slice attention passes. Issuing
   them on 2-4 streams removes serialisation, not launches; graph capture plus streams is the same
   win.
4. **Use the already-measured batched entry points where a batch axis exists** - the kernel code
   exists and is measured, so this is a launch/wiring change: cert_lm_head_batched (m22, 6 rows,
   2.853x), mega_mhc batched (m23, 4.32x), flash_mla_csa2_seq batched (m20, 3.567x),
   lm_head_reduce batched (m24, 1.01x, no win). At decode tokens=1 there is no row batch today,
   but the DSpark verify step brings 6 rows and the four ranks can share a launch where the ABI
   takes row offsets. This is the one reduction on the list that needs no graph.
5. **Drop the two extra identity mHC entries** at the Engram layers (1 and 14): the Engram seam is
   host-side, so those layers run the mHC entry twice. Device Engram removes 2 launches/position.

What cannot be removed without a kernel/ABI change (named so the boundary is explicit): the
8 gate/up -> 12 per-layer launches (fuse w1+w3 into one launch, then fuse the weighted SwiGLU +
quantise into the epilogue: 12 -> 4 launches/layer, 480 -> 160 per position), the 4 serial combine
launches (4 -> 1 launch looping ranks inside, no numeric change), and the per-layer router-id
readback (device-side routing, which also removes 40 D2H syncs/position). Those are the fusion
targets for the next session's design work, not tonight's measurement.
