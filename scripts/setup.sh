#!/usr/bin/env bash
# Copyright (c) 2026 Hunter L. Allen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# One-time setup on the first DGX Spark. Run it from the checkout:
#
#   ./scripts/setup.sh                build, let this Spark sign in to the others, start the console
#   ./scripts/setup.sh --no-ssh       skip the sign-in step
#   ./scripts/setup.sh --foreground   run the console in this terminal instead of the background
#   ./scripts/setup.sh --stop         stop the console
#
# Every step is skipped when it is already done, so running it again is safe:
#
#   1. checks this is a DGX Spark (aarch64 Linux with the CUDA toolkit DGX OS ships)
#   2. installs Rust for this user with rustup when it is missing (no root)
#   3. builds the server, its GPU kernels, and fetches the pinned NCCL library
#   4. finds the other Sparks on the network and lets this one sign in to them (ssh-copy-id
#      asks once for each Spark's password)
#   5. starts the web console and prints its address
#
# The other Sparks need nothing installed by hand: the console's Setup page copies the server to
# them, downloads the model once and copies it over the fast network, and starts everything.
set -euo pipefail

ROOT=$(cd "${0%/*}/.." && pwd)
cd "$ROOT"
WEB_PORT=8790
STATE_DIR="$ROOT/target/web"
PIDFILE="$STATE_DIR/console.pid"
LOGFILE="$STATE_DIR/console.log"
DO_SSH=1
FOREGROUND=0

for arg in "$@"; do
    case "$arg" in
        --no-ssh) DO_SSH=0 ;;
        --foreground) FOREGROUND=1 ;;
        --stop)
            if [ -s "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
                kill "$(cat "$PIDFILE")" && rm -f "$PIDFILE"
                echo "Stopped the console."
            else
                echo "The console is not running from this checkout."
            fi
            exit 0
            ;;
        -h | --help)
            sed -n '16,33p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "setup: unknown option $arg (try --help)" >&2
            exit 2
            ;;
    esac
done

step() { printf '\n\033[1m%s\033[0m\n' "$*"; }
ok() { printf '  \033[32m✓\033[0m %s\n' "$*"; }
note() { printf '  %s\n' "$*"; }
fail() {
    printf '\n\033[31m%s\033[0m\n' "$*" >&2
    exit 1
}
interactive() { [ -t 0 ] && [ -t 1 ]; }

# Answers yes unless the person types n.
confirm() {
    local answer
    interactive || return 0
    read -r -p "  $1 [Y/n] " answer
    case "$answer" in [nN]*) return 1 ;; *) return 0 ;; esac
}

console_running() {
    curl -fsS --max-time 2 "http://127.0.0.1:$WEB_PORT/api/health" >/dev/null 2>&1
}

print_address() {
    local lan
    # The address other computers reach this Spark at: the one its default route leaves from (the
    # fast-network addresses are not reachable from a laptop). Nothing is sent.
    lan=$(ip -4 route get 1.1.1.1 2>/dev/null | sed -n 's/.* src \([0-9.]*\).*/\1/p' | head -n 1) || true
    [ -n "$lan" ] || lan=$(hostname -I 2>/dev/null | awk '{ print $1 }') || true
    echo
    printf '\033[1mOpen the SparkFlash console in a browser on any computer on this network:\033[0m\n\n'
    printf '    http://%s.local:%s\n' "$(hostname)" "$WEB_PORT"
    [ -z "$lan" ] || printf '    http://%s:%s   (if the name does not open)\n' "$lan" "$WEB_PORT"
    echo
    echo "  The Setup page walks through the rest: the fast network, the model download, and the start."
}

# 1. This machine.
step "1/5  Checking this Spark"
[ "$(uname -s)" = Linux ] || fail "SparkFlash runs on DGX Spark (Linux). This is $(uname -s)."
[ "$(uname -m)" = aarch64 ] || fail "SparkFlash runs on DGX Spark (aarch64). This machine is $(uname -m)."
export PATH="/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH"
command -v nvcc >/dev/null || fail "The CUDA toolkit is missing (no nvcc in /usr/local/cuda/bin). DGX OS includes it; on another install, add the CUDA 13 toolkit first."
ok "$(nvcc --version | sed -n 's/.*release \([0-9.]*\).*/CUDA \1/p' | head -n 1) at $(dirname "$(command -v nvcc)")"
if command -v nvidia-smi >/dev/null; then
    ok "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1)"
fi
missing=()
for tool in cc curl unzip rsync python3 ssh sha256sum; do
    command -v "$tool" >/dev/null || missing+=("$tool")
done
if [ ${#missing[@]} -gt 0 ]; then
    packages=$(printf '%s\n' "${missing[@]}" | sed 's/^cc$/build-essential/; s/^ssh$/openssh-client/; s/^sha256sum$/coreutils/' | sort -u | tr '\n' ' ')
    fail "Missing: ${missing[*]}. Install them with: sudo apt install $packages"
fi
ok "build tools, curl, unzip, rsync, python3 and ssh are installed"

# 2. Rust.
step "2/5  Rust"
if ! command -v rustup >/dev/null; then
    note "Rust is not installed. It installs into $HOME/.cargo for this user only (no root)."
    confirm "Install Rust now?" || fail "Rust is needed to build SparkFlash."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs |
        sh -s -- -y --no-modify-path --profile minimal --default-toolchain none
fi
channel=$(sed -n 's/^channel *= *"\(.*\)"/\1/p' rust-toolchain.toml)
if ! rustup run "$channel" rustc --version >/dev/null 2>&1; then
    note "Installing Rust $channel (pinned by rust-toolchain.toml)…"
    rustup toolchain install "$channel" --profile minimal --component rustfmt --component clippy
fi
ok "$(rustc --version)"

# 3. Build.
step "3/5  Building SparkFlash (the first build takes several minutes)"
cargo build --release -p sparkflash-cli --features cuda,tp4-serve
ok "server: target/release/sparkflash"
# The kernels rebuild only when a source under kernels/ is newer than its cubin.
newest="kernels/$(ls -t kernels | head -n 1)" || true
stale=0
for source in kernels/*.cu; do
    cubin="target/cubins/$(basename "$source" .cu).cubin"
    [ "$cubin" -nt "$newest" ] || stale=1
done
if [ "$stale" = 1 ]; then
    scripts/build-cubins.sh target/cubins | sed 's/^/  /'
else
    ok "GPU kernels: target/cubins is up to date"
fi
scripts/fetch-nccl.sh | sed 's/^/  /'

# 4. Sign-in to the other Sparks.
step "4/5  Letting this Spark sign in to the other three"
if [ "$DO_SSH" = 0 ] || ! interactive; then
    note "Skipped. The console's first step shows the command for any Spark it cannot sign in to."
else
    if [ ! -f "$HOME/.ssh/id_ed25519" ]; then
        mkdir -p -m 700 "$HOME/.ssh"
        ssh-keygen -q -t ed25519 -N '' -f "$HOME/.ssh/id_ed25519"
        ok "made a key for this Spark: ~/.ssh/id_ed25519"
    fi
    me=$(hostname | tr '[:upper:]' '[:lower:]')
    found=()
    if command -v avahi-browse >/dev/null; then
        note "Looking for computers on the network…"
        while IFS= read -r name; do
            [ "${name%.local}" = "$me" ] || found+=("$name")
        done < <(timeout 8 avahi-browse -tpr _ssh._tcp 2>/dev/null |
            awk -F';' '$1 == "=" && $3 == "IPv4" { print tolower($7) }' | sort -u)
    fi
    hosts=()
    if [ ${#found[@]} -gt 0 ]; then
        note "Found:"
        suggested=()
        for i in "${!found[@]}"; do
            printf '    %d) %s\n' "$((i + 1))" "${found[$i]}"
            case "${found[$i]}" in *spark*) suggested+=("$((i + 1))") ;; esac
        done
        default=""
        [ ${#suggested[@]} -eq 3 ] && default="${suggested[*]}"
        read -r -p "  Numbers of your three other Sparks${default:+ [$default]} (Enter to skip): " picks || picks=""
        picks=${picks:-$default}
        for pick in $picks; do
            if [[ "$pick" =~ ^[0-9]+$ ]] && [ "$pick" -ge 1 ] && [ "$pick" -le ${#found[@]} ]; then
                hosts+=("${found[$((pick - 1))]}")
            fi
        done
    else
        read -r -p "  Names of your three other Sparks, e.g. spark-1.local spark-2.local spark-3.local (Enter to skip): " names || names=""
        for name in $names; do hosts+=("$name"); done
    fi
    for host in ${hosts[@]+"${hosts[@]}"}; do
        if ssh -n -o BatchMode=yes -o ConnectTimeout=5 "$host" true 2>/dev/null; then
            ok "$host: already allowed"
            continue
        fi
        note "$host: type yes if asked to trust it, then the password of your account there."
        if ssh-copy-id -i "$HOME/.ssh/id_ed25519.pub" "$host"; then
            ok "$host: allowed"
        else
            note "$host: not set up. The console's first step shows how to retry."
        fi
    done
fi

# 5. The console.
step "5/5  Starting the console"
mkdir -p "$STATE_DIR"
if console_running; then
    ok "already running"
    note "After updating SparkFlash, run ./scripts/setup.sh --stop (when nothing is running in"
    note "the console) and then ./scripts/setup.sh again to use the new version."
    print_address
    exit 0
fi
if [ "$FOREGROUND" = 1 ]; then
    print_address
    exec target/release/sparkflash --web --config sparkflash.toml
fi
# Detached from this terminal, so closing it does not stop a download the console is running.
nohup setsid target/release/sparkflash --web --config sparkflash.toml >>"$LOGFILE" 2>&1 </dev/null &
echo $! >"$PIDFILE"
for _ in $(seq 1 50); do
    console_running && break
    sleep 0.2
done
console_running || fail "The console did not start. Its log is $LOGFILE."
ok "running in the background (log: target/web/console.log, stop: ./scripts/setup.sh --stop)"
print_address
