# task-7 F2 kernel pin - measured, batch-5 candidate

This branch carries the device half of F2: `kernels/flash_mla_csa2_seq_sm121.cu` now rounds every
softmax probability to bf16 before the P*V value sum and keeps the normalising sum on the f32
probability, which is the released body's arithmetic (`kernel.py:339-341` cast into the bf16 tile,
`kernel.py:342-353` f32 `reduce_sum` before it). The host twins on this branch (engine
`sparse_attention` and the `mla_ref` union twin) already carry the same rule.

## Measured pin

Produced in the pinned CUDA image (`nvidia/cuda:13.0.3-devel-ubuntu22.04`, image id
`3869b846a8cc`), CPU-only with no GPU attached, which is enough: nvcc compiles sm_121a without a
device. The command was the repository's own generator, run from the worktree mounted at `/w`:

```
docker run --rm -v /home/user/wt/n3-woa:/w -w /w \
    -e SPARKFLASH_NVCC=/usr/local/cuda/bin/nvcc 3869b846a8cc \
    /bin/sh crates/sparkflash-kernels/scripts/regen-kernel-baseline.sh
```

The toolkit the script read out of nvcc: `nvcc_release=V13.0.88`,
`nvcc_build=cuda_13.0.r13.0/compiler.36424714_0` - the same build the baseline was recorded
under, so the pin is attributable to that exact V13.0.88 build and not to a toolkit drift.

The diff against the previous (f32-kernel) baseline was exactly the predicted row and nothing
else - no second kernel moved, the header's toolkit lines are identical:

```
kernel flash_mla_csa2_seq_sm121 cubin_bytes=38232 cubin_sha256=b12c68e68398138021d329dc8fb204581015ad7e8f07cf6d94c85e7bae272f3d ptx_membar_lines=0
```

was `cubin_bytes=38080 cubin_sha256=f6dfc738... ptx_membar_lines=0`, so +152 bytes, a new
digest, and `ptx_membar_lines` still 0: the edit adds an arithmetic cast, not a fence, which is
exactly what an F2 change predicts. Verification with the new baseline, in the same container:
`sh crates/sparkflash-kernels/scripts/check-kernel-baseline.sh` recompiled all fifteen kernels and
printed that they all match the recorded baseline (V13.0.88, cuda_13.0.r13.0/compiler.36424714_0),
exit 0. `crates/sparkflash-kernels/baseline/kernel_baseline.txt` carries the row above.

A second row moving, a non-zero `ptx_membar_lines`, or a different toolkit string would each have been
a finding to report rather than a baseline to regenerate; none of them happened.

## What still needs a window (not this pin)

The window leg is confirmation, not the pin itself: with this kernel the device-vs-host body
twin comparison should return to matching in the armed positions. Before this kernel, that
comparison reported the bf16 round as a difference **by design** (the host twin is native, the old
kernel kept f32); after it, a clean comparison is the confirmation that the kernel implements the
same rule, and a dirty one is a real finding. Landing does not depend on that window; the pin
above is what the batch needs.

## Decision (taken, so the review cannot relitigate it)

Round **both** branches; do not narrow the edit to the union branch. The released body rounds
every softmax probability regardless of the KV format, and the format is our staging choice, not
the checkpoint's arithmetic; a device path that kept f32 in a legacy format branch would be a
non-native path with no reference justification, and D39's rule is native arithmetic rather than
native where convenient.

The format-0/1 CPU twins therefore stay on their historical f32 arithmetic and are annotated as
pre-F2, exactly like the union twin was handled: each of `attention`, `attention_with_sink`,
`attention_window` and `attention_with_sink_window` carries a Historical arithmetic (pre-F2)
paragraph, both pre-F2 pass-three loops say why the f32 probability is deliberate, and the test
`the_pre_f2_window_path_keeps_the_f32_probability_by_design` asserts **both** halves - the twin
equals the f32 arithmetic bit for bit and differs from the bf16-native answer - so the
device-vs-twin difference is pinned as expected instead of being read as a defect. The
annotation lands here with the kernel edit and the pin.

