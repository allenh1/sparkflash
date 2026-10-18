#!/bin/sh
# Verifies that nvcc still compiles the eight baseline kernels to the recorded baseline.
#
# This is the check that would catch a kernel silently changing: it recompiles each kernel to a cubin
# and to PTX, and compares the cubin bytes, the cubin sha256 and the PTX membar count with
# baseline/kernel_baseline.txt. The kernel list below is the one thing a human has to keep honest: the
# script refuses to run when it is empty, and it checks that the list, the manifest and the
# measurement name the same kernels, so a drifted list cannot print the success line. The Rust test
# the_baseline_list_matches_the_manifest_and_the_kernels_on_disk pins it against kernels/ as well.
#
# A different toolkit is reported as a toolkit change (exit 2) and not
# as a kernel change, because the recorded numbers are only comparable under the toolkit that made
# them; a false "your kernel changed" is how a baseline check gets deleted.
set -eu

crate_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
baseline="$crate_dir/baseline/kernel_baseline.txt"
nvcc=${SPARKFLASH_NVCC:-nvcc}
kernels="mega_mhc_sm121 b12x_mxfp8_dense_sm121 b12x_mxfp4_moe_sm121 flash_mla_csa2_sm121 b12x_mxfp8_wo_a_sm121 cert_lm_head_sm121 swiglu_mxfp8_quant_sm121 moe_router_sm121"

if [ ! -f "$baseline" ]; then
    echo "no baseline at $baseline" >&2
    exit 1
fi
if ! command -v "$nvcc" >/dev/null 2>&1; then
    echo "nvcc is not on PATH: run this script on a CUDA host or inside the CUDA container" >&2
    exit 127
fi

recorded_release=$(sed -n 's/^nvcc_release=//p' "$baseline")
recorded_build=$(sed -n 's/^nvcc_build=//p' "$baseline")
present_release=$("$nvcc" --version | sed -n 's/^Cuda compilation tools, release [^,]*, \(V[^ ]*\)$/\1/p')
present_build=$("$nvcc" --version | sed -n 's/^Build \(.*\)$/\1/p')

if [ "$present_release" != "$recorded_release" ] || [ "$present_build" != "$recorded_build" ]; then
    echo "the toolkit changed, regenerate the baseline" >&2
    echo "  baseline was generated with: release=$recorded_release build=$recorded_build" >&2
    echo "  this toolkit is:             release=$present_release build=$present_build" >&2
    echo "  regenerate with: /bin/sh $crate_dir/scripts/regen-kernel-baseline.sh $baseline" >&2
    exit 2
fi

if [ -z "$kernels" ]; then
    echo "the kernel list is empty: there is nothing to compare, and the success line would be a lie" >&2
    exit 1
fi

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
/bin/sh "$crate_dir/scripts/regen-kernel-baseline.sh" "$work/measured.txt"

# The list, the manifest and the measurement must name the same kernels. Without this, a drifted list
# would compare a subset (or nothing) and still print the success line, and a kernel the manifest
# records but the list omits would never be checked.
listed=$(printf '%s\n' $kernels | sort)
recorded_names=$(sed -n 's/^kernel \([^ ]*\) .*/\1/p' "$baseline" | sort)
measured_names=$(sed -n 's/^kernel \([^ ]*\) .*/\1/p' "$work/measured.txt" | sort)
if [ -z "$listed" ] || [ "$listed" != "$recorded_names" ] || [ "$listed" != "$measured_names" ]; then
    echo "the kernel list, the baseline and the measurement do not name the same kernels" >&2
    echo "  list:     $(printf '%s ' $kernels)" >&2
    echo "  baseline: $(printf '%s ' $recorded_names)" >&2
    echo "  measured: $(printf '%s ' $measured_names)" >&2
    exit 1
fi

status=0
for kernel in $kernels; do
    recorded=$(grep "^kernel $kernel " "$baseline" || true)
    measured=$(grep "^kernel $kernel " "$work/measured.txt" || true)
    if [ -z "$recorded" ]; then
        echo "the baseline has no record of $kernel" >&2
        status=1
    elif [ "$recorded" = "$measured" ]; then
        echo "ok   $measured"
    else
        echo "CHANGED $kernel" >&2
        echo "  recorded: $recorded" >&2
        echo "  measured: $measured" >&2
        status=1
    fi
done
if [ "$status" -ne 0 ]; then
    echo "a kernel's compiled form changed (cubin bytes, cubin sha256 or PTX membar count)." >&2
    echo "If the change is intended, regenerate the baseline and review the diff:" >&2
    echo "  /bin/sh $crate_dir/scripts/regen-kernel-baseline.sh $baseline" >&2
    exit 1
fi
echo "all kernels match the recorded baseline ($recorded_release, $present_build)"
