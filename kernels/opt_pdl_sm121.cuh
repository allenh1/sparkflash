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

// kernels/opt_pdl_sm121.cuh
//
// Programmatic dependent launch for the opt kernels. The host launches every opt_* symbol with
// programmatic stream serialization, so a kernel may start while its predecessor is still running.
// OPT_PDL_ENTRY() is the first statement of every opt kernel: it lets the next kernel launch at
// once (the next kernel's launch latency then overlaps this one's run), and then waits until the
// previous grid has completed and its writes are visible. Because every opt kernel waits before it
// reads anything, and the wait covers the whole previous grid, stream order is unchanged.
//
// A kernel whose first reads are constant weights may split the entry: OPT_PDL_LAUNCH() first,
// the weight loads, then OPT_PDL_WAIT() before anything a previous kernel writes. The weights then
// stream while the previous grid finishes, and nothing it writes is read early.
#pragma once

#define OPT_PDL_LAUNCH() asm volatile("griddepcontrol.launch_dependents;" ::: "memory")
#define OPT_PDL_WAIT() asm volatile("griddepcontrol.wait;" ::: "memory")
#define OPT_PDL_ENTRY() asm volatile("griddepcontrol.launch_dependents;\n\tgriddepcontrol.wait;" ::: "memory")
