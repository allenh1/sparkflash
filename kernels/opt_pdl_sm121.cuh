// kernels/opt_pdl_sm121.cuh
//
// Programmatic dependent launch for the opt kernels. The host launches every opt_* symbol with
// programmatic stream serialization, so a kernel may start while its predecessor is still running.
// OPT_PDL_ENTRY() is the first statement of every opt kernel: it lets the next kernel launch at
// once (the next kernel's launch latency then overlaps this one's run), and then waits until the
// previous grid has completed and its writes are visible. Because every opt kernel waits before it
// reads anything, and the wait covers the whole previous grid, stream order is unchanged.
#pragma once

#define OPT_PDL_ENTRY() asm volatile("griddepcontrol.launch_dependents;\n\tgriddepcontrol.wait;" ::: "memory")
