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

// clang-probe stand-in for CUDA's <cuda_runtime.h>.

// Used only by the -fsyntax-only clang probe. The values carry no semantics.
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "cuda_stub.h"

// The runtime API's launch bounds and stream types the kernel sources mention.
using cudaStream_t = void *;
using cudaError_t = int;
#define cudaSuccess 0
