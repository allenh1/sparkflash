// Copyright (c) 2026 Hunter L. Allen
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! The RoCEnante dispatcher must emit its scoped trace spans with the layer id as meta, so the
//! trace exporter can attribute collectives to layers. This file holds a single test because the
//! CPU trace ring is process global.

use sparkflash_comm::{DualPathCommunicator, ROCENANTE_MAX_BYTES, SparkFabricMode};
use sparkflash_trace::{TraceScope, arm, clear_cpu_events, disarm, drain_cpu_events};

#[test]
fn dispatch_emits_scoped_spans_carrying_the_layer_id() {
    disarm();
    clear_cpu_events();
    arm(TraceScope::ROCENANTE_RDMA | TraceScope::NCCL_COLLECTIVE, 64);

    let communicator = DualPathCommunicator::new(0, 4, SparkFabricMode::Switched200G);
    let small = communicator.dispatch_all_reduce_bf16(41, 4096, true);
    let big = communicator.dispatch_all_reduce_bf16(42, ROCENANTE_MAX_BYTES + 1, false);
    let gather = communicator.dispatch_all_gather_bf16(43, 128);
    assert!(small.is_rocenante());
    assert!(!big.is_rocenante());
    assert!(gather.is_rocenante());

    let mut observed: Vec<(&str, u64, u32)> = drain_cpu_events()
        .iter()
        .map(|event| (event.label, event.scope, event.meta))
        .collect();
    observed.sort_unstable();
    assert_eq!(
        observed,
        vec![
            ("dispatch_nccl_ring", TraceScope::NCCL_COLLECTIVE, 42),
            (
                "dispatch_rocenante_allgather",
                TraceScope::ROCENANTE_RDMA,
                43
            ),
            ("dispatch_rocenante_oneshot", TraceScope::ROCENANTE_RDMA, 41),
        ]
    );

    // A dispatch outside the armed scope records nothing.
    disarm();
    clear_cpu_events();
    arm(TraceScope::ROCENANTE_RDMA, 64);
    communicator.dispatch_all_reduce_bf16(44, ROCENANTE_MAX_BYTES + 1, false);
    assert!(
        drain_cpu_events().is_empty(),
        "the NCCL span must respect the armed scope mask"
    );

    // And once disarmed nothing is recorded at all.
    disarm();
    clear_cpu_events();
    communicator.dispatch_all_reduce_bf16(45, 4096, false);
    assert!(drain_cpu_events().is_empty());
}
