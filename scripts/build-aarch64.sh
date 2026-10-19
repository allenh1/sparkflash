#!/usr/bin/env bash
# Cross-compile the SparkFlash workspace for the aarch64 head node from this x86_64 dev box.
#
# WHY THIS EXISTS
#   The head node (node0 / 192.0.2.80) is aarch64 with no cargo and no rustc, and while the
#   vLLM inference runs it has roughly 3 GB of memory available out of about 121 GB. Compiling there
#   is forbidden: a cargo build would thrash that headroom and endanger the running inference. Every
#   binary we put on that node is built here and copied over.
#
# WHY CROSS-COMPILING HERE NEEDS A SCRIPT AT ALL
#   Measured on this box on 2026-10-16:
#     - rustc/cargo are 1.98.1 from a plain /opt/rust-bin-1.98.1 install. There is NO rustup, so
#       there is no "rustup target add".
#     - There is no aarch64-linux-gnu-gcc, no aarch64-linux-gnu-ld, and no ld.lld anywhere.
#     - The container sets the no-new-privileges flag, so sudo and apt-get cannot run at all.
#
#   The textbook route is the distro cross toolchain:
#       apt-get install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu libc6-dev-arm64-cross
#   The g++ and libc6-dev packages are required, not optional: the dependency tree contains both C
#   and C++ (tokenizers -> onig is C, esaxx-rs is C++, ring is C, all built by the cc crate), and
#   the C++ objects need an aarch64 libstdc++ at link time. That route is REJECTED HERE ONLY because
#   this box cannot install packages - it is not the wrong answer. On any box with root, use it: it
#   is one apt line and needs no download pin.
#
#   What this script does instead, with no root at all:
#     1. installs rustup under $HOME (--no-modify-path, minimal profile) pinned to $RUST_VERSION,
#        and adds the aarch64 std for exactly that version;
#     2. downloads zig pinned to $ZIG_VERSION and checked against $ZIG_SHA256, and uses it as the
#        cross C compiler, C++ compiler, linker and archiver. Zig is self-contained - it carries its
#        own libc, libc++ and lld - so no sysroot package is needed.
#   Both steps are idempotent, and both refuse to continue if a pin moves: a cross toolchain that
#   silently changes underneath us is worse than one that is missing.
#
# USAGE
#   scripts/build-aarch64.sh                 # release binary for sparkflash-cli (the preflight CLI)
#   scripts/build-aarch64.sh --workspace     # every workspace binary
#   scripts/build-aarch64.sh --features=cuda # the CLI with its cuda-gated device arms
#   scripts/build-aarch64.sh --example=m7_layer0   # one example (the M7 layer-0 harness)
#   scripts/build-aarch64.sh --print-env     # show the resolved toolchain and exit
#
# OUTPUT
#   <target-dir>/aarch64-unknown-linux-gnu/release/sparkflash
#   CARGO_TARGET_DIR selects the target dir; it defaults to the repository's own target/.

set -euo pipefail

# ----- pins: changing any of these is a deliberate act, not a refresh -----------------------------
RUST_VERSION="1.98.1"
TARGET="aarch64-unknown-linux-gnu"
ZIG_TARGET="aarch64-linux-gnu"
ZIG_VERSION="0.17.0"
ZIG_SHA256="1cbe9df9f27e6b78d14ccbca43b6703a404ef79ef1c463de901d7f088d4e2026"

# ----- layout ------------------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS_DIR="${SPARKFLASH_TOOLCHAIN_DIR:-$HOME/opt}"
ZIG_TOP="$TOOLS_DIR/zig-x86_64-linux-$ZIG_VERSION"
ZIG="$ZIG_TOP/zig"
ZIG_TARBALL="$TOOLS_DIR/zig-$ZIG_VERSION.tar.xz"
ZIG_URL="https://ziglang.org/download/$ZIG_VERSION/zig-x86_64-linux-$ZIG_VERSION.tar.xz"
RUSTUP="$HOME/.cargo/bin/rustup"
CARGO="$HOME/.cargo/bin/cargo"
WRAP_DIR="$TOOLS_DIR/bin"
PREFIX="sparkflash-aarch64"
CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$REPO_ROOT/target}"
export CARGO_TARGET_DIR

BUILD_ARGS=(-p sparkflash-cli)
PRINT_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --workspace) BUILD_ARGS=(--workspace) ;;
    --features=*) BUILD_ARGS+=(--features "${arg#--features=}") ;;
    --example=*) BUILD_ARGS=(-p sparkflash-kernels --example "${arg#--example=}") ;;
    --print-env) PRINT_ONLY=1 ;;
    -h|--help) sed -n '2,45p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

say() { printf '==> %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

# ----- 1. rustup, pinned, user-local -------------------------------------------------------------
if [ ! -x "$RUSTUP" ]; then
  say "installing rustup under $HOME (no PATH changes, minimal profile, $RUST_VERSION)"
  tmp="$(mktemp -d)"
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o "$tmp/rustup-init.sh"
  sh "$tmp/rustup-init.sh" -y --no-modify-path --profile minimal --default-toolchain "$RUST_VERSION"
  rm -rf "$tmp"
fi
"$RUSTUP" toolchain list | grep -q "^$RUST_VERSION" || "$RUSTUP" toolchain install "$RUST_VERSION" --profile minimal
"$RUSTUP" target list --installed --toolchain "$RUST_VERSION" | grep -qx "$TARGET" \
  || "$RUSTUP" target add "$TARGET" --toolchain "$RUST_VERSION"

have_rust="$("$CARGO" +"$RUST_VERSION" --version | awk '{print $2}')"
[ "$have_rust" = "$RUST_VERSION" ] || die "cargo reports $have_rust, pinned $RUST_VERSION"
rustc_host="$("$RUSTUP" run "$RUST_VERSION" rustc --version)"
say "rust: $rustc_host"

# ----- 2. zig, pinned and checksummed ------------------------------------------------------------
mkdir -p "$TOOLS_DIR"
if [ ! -f "$ZIG_TARBALL" ]; then
  say "downloading zig $ZIG_VERSION"
  curl -sSL --retry 3 -o "$ZIG_TARBALL" "$ZIG_URL"
fi
echo "$ZIG_SHA256  $ZIG_TARBALL" | sha256sum -c - >/dev/null \
  || die "zig $ZIG_VERSION tarball at $ZIG_TARBALL does not match the pinned sha256 $ZIG_SHA256; delete it and re-run"
[ -x "$ZIG" ] || tar -xf "$ZIG_TARBALL" -C "$TOOLS_DIR"
[ -x "$ZIG" ] || die "zig $ZIG_VERSION unpacked without $ZIG"
zig_have="$("$ZIG" version)"
[ "$zig_have" = "$ZIG_VERSION" ] || die "zig reports $zig_have, pinned $ZIG_VERSION"
say "zig: $zig_have at $ZIG"

# ----- 3. the cross cc/c++/ar wrappers -----------------------------------------------------------
# Three translations happen here, and all three are forced by the first real build rather than by
# taste:
#
#   * The target flag. cargo passes --target=<rust triple> to the C compiler, and zig parses target
#     queries in its own spelling, so it reads aarch64-unknown-linux-gnu as an unknown operating
#     system. That is not hypothetical: the first cross build died inside ring's build script with
#     "unable to parse target query 'aarch64-unknown-linux-gnu': UnknownOperatingSystem". The
#     wrappers therefore DROP whatever target flag the caller passes and append zig's own
#     -target aarch64-linux-gnu.
#   * The C++ standard library. esaxx-rs compiles C++ through this wrapper, so its objects reference
#     libc++'s ABI, while the cc crate still tells cargo to link -lstdc++. Mapping -lstdc++ to -lc++
#     links the objects against the library that produced them; leaving it fails the link, because
#     zig ships libc++ and no libstdc++.
#   * The Cortex-A53 linker flag. rustc asks for -Wl,--fix-cortex-a53-843419 on every aarch64 target,
#     and lld refuses it ("error: unsupported linker arg: --fix-cortex-a53-843419"). Dropping it is
#     safe here because the flag works around ARM erratum 843419 in the Cortex-A53 core and the head
#     node is an NVIDIA GB10 Grace (Neoverse-class), not a Cortex-A53. The link is the last step, so
#     a wrapper that keeps this flag fails a build in which everything else already compiled.
mkdir -p "$WRAP_DIR"
write_c_wrapper() {  # $1 = wrapper suffix, $2 = zig driver (cc or c++)
  cat > "$WRAP_DIR/$PREFIX-$1" <<WRAP
#!/usr/bin/env bash
# Generated by scripts/build-aarch64.sh: edit the script, not this file.
set -euo pipefail
args=()
skip_target=0
for arg in "\$@"; do
  if [ "\$skip_target" = 1 ]; then skip_target=0; continue; fi
  case "\$arg" in
    --target|-target) skip_target=1 ;;
    --target=*|-target=*) ;;
    -lstdc++) args+=("-lc++") ;;
    -static-libstdc++|-static-libgcc) ;;
    -Wl,--fix-cortex-a53-843419) ;;
    *) args+=("\$arg") ;;
  esac
done
exec "$ZIG" $2 -target $ZIG_TARGET "\${args[@]}"
WRAP
  chmod +x "$WRAP_DIR/$PREFIX-$1"
}
write_c_wrapper cc  cc
write_c_wrapper cxx c++
cat > "$WRAP_DIR/$PREFIX-ar" <<WRAP
#!/usr/bin/env bash
# Generated by scripts/build-aarch64.sh: edit the script, not this file.
set -euo pipefail
exec "$ZIG" ar "\$@"
WRAP
chmod +x "$WRAP_DIR/$PREFIX-ar"

# cc/cxx/ar for the target: the cc crate reads the target-specific names (lowercase triple, dashes
# to underscores), and cargo reads the linker as CARGO_TARGET_<TARGET>_LINKER with the triple
# uppercased. Both spellings are derived from $TARGET so neither can drift from it.
TARGET_LOWER="$(printf '%s' "$TARGET" | tr '-' '_')"
TARGET_UPPER="$(printf '%s' "$TARGET" | tr '[:lower:]-' '[:upper:]_')"
export "CC_${TARGET_LOWER}=$WRAP_DIR/$PREFIX-cc"
export "CXX_${TARGET_LOWER}=$WRAP_DIR/$PREFIX-cxx"
export "AR_${TARGET_LOWER}=$WRAP_DIR/$PREFIX-ar"
export "CARGO_TARGET_${TARGET_UPPER}_LINKER=$WRAP_DIR/$PREFIX-cc"

say "cc  = $WRAP_DIR/$PREFIX-cc"
say "cxx = $WRAP_DIR/$PREFIX-cxx"
say "ar  = $WRAP_DIR/$PREFIX-ar"

if [ "$PRINT_ONLY" = 1 ]; then
  say "target dir: $CARGO_TARGET_DIR"
  exit 0
fi

# ----- 4. build ----------------------------------------------------------------------------------
say "building $RUST_VERSION for $TARGET in $REPO_ROOT"
say "cargo build --release --target $TARGET ${BUILD_ARGS[*]}"
cd "$REPO_ROOT"
"$CARGO" +"$RUST_VERSION" build --release --target "$TARGET" "${BUILD_ARGS[@]}"

# ----- 5. what came out --------------------------------------------------------------------------
OUT="$CARGO_TARGET_DIR/$TARGET/release"
say "artifacts in $OUT"
for binary in "$OUT"/sparkflash "$OUT"/rocenante-probe "$OUT"/examples/*; do
  [ -f "$binary" ] || continue
  printf '%s\n' "$(file -b "$binary")"
  printf 'sha256 %s  %s\n' "$(sha256sum "$binary" | cut -d' ' -f1)" "$binary"
done
