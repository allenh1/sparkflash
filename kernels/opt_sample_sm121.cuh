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

// kernels/opt_sample_sm121.cuh
//
// Attribution: Stateless counter-seeded Gumbel-max sampling so speculative draft and target
// verification share identical random draws without RNG state synchronization follows the
// sampling design in NVIDIA Flare (https://github.com/NVIDIA/flare-local-inference,
// Apache-2.0; Luke Alonso @lukealonso, @original-el8, and contributors). See NOTICE.
//
// The serving sampler's noise: a Gumbel variate per (seed, position, vocabulary id), counter based,
// so every rank draws the same value for the same token without any state or exchange. A token is
// sampled from softmax(logits / T) as the argmax of logit / T + G (the Gumbel-max rule); the
// verify head and the drafter both perturb with this one function, so a draft that agrees with the
// target's distribution tends to pick the same token under the same noise, and accept-while-equal
// stays exact: the emitted token at a position is the same function of the target's logits whatever
// the drafter proposed.
//
// The host twin is crate::tp4_forward::sample::gumbel (same integer hash, same uniform); the two
// may differ in the last ulp of the logarithms, which no consumer compares across them.

#pragma once

#include <stdint.h>

// splitmix64's finaliser over the packed counter: seed, then (position, id) as one 64-bit word.
__device__ __forceinline__ uint64_t opt_sample_mix(uint64_t x) {
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBull;
    x ^= x >> 31;
    return x;
}

// G(seed, position, id) = -ln(-ln(u)), u in (0, 1) from the top 23 bits of the hash, centred in its
// bin so it is never 0 or 1 (23 bits keep the centred value exact in an f32).
__device__ __forceinline__ float opt_sample_gumbel(uint64_t seed, uint32_t position, uint32_t id) {
    const uint64_t counter = ((uint64_t)position << 32) | (uint64_t)id;
    const uint64_t h = opt_sample_mix(opt_sample_mix(seed) ^ counter);
    const float u = ((float)(uint32_t)(h >> 41) + 0.5f) * (1.0f / 8388608.0f);
    return -logf(-logf(u));
}
