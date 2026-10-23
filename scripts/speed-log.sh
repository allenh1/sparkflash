#!/usr/bin/env bash
# One-number speed log: append or print the lean decode ms/step of a four-rank window run.
#
#   scripts/speed-log.sh append --run <run dir> [--commit SHA] [--note TEXT]
#   scripts/speed-log.sh print
#
# The run directory is a window evidence dir holding rank0..rank3/tp4-token-rN.json (the layout the
# run-fourrank.sh wrapper writes). The step mean is the mean of the four ranks' step_ms.mean_ms, the
# same number the status doc publishes. The CSV lives in docs/build/speed-log.csv so the daily
# number is a committed artifact, not a chat message.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CSV="$ROOT/docs/build/speed-log.csv"
[ -f "$CSV" ] || printf 'date,commit,run,step_mean_ms,tokens_ok,logits_sha_ok,note\n' > "$CSV"
usage(){ echo "usage: speed-log.sh append --run DIR [--commit SHA] [--note TEXT] | print" >&2; exit 2; }
[ $# -ge 1 ] || usage
case "$1" in
  print)
    python3 - "$CSV" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
print("| date | commit | run | lean ms/step | identity | note |")
print("| :-- | :-- | :-- | --: | :-- | :-- |")
for r in rows:
    print(f"| {r['date']} | {r['commit']} | {r['run']} | {r['step_mean_ms']} | tokens {r['tokens_ok']} / sha {r['logits_sha_ok']} | {r['note']} |")
PY
    ;;
  append)
    shift
    RUN=""; COMMIT="-"; NOTE=""
    while [ $# -gt 0 ]; do
      case "$1" in
        --run) RUN=$2; shift 2;;
        --commit) COMMIT=$2; shift 2;;
        --note) NOTE=$2; shift 2;;
        *) usage;;
      esac
    done
    [ -n "$RUN" ] || usage
    python3 - "$CSV" "$RUN" "$COMMIT" "$NOTE" <<'PY'
import csv, datetime, json, os, statistics, sys
csv_path, run, commit, note = sys.argv[1:5]
# The published number must name the commit that produced it, not whatever branch happens to be
# checked out where the row is filed. The row's run.meta names its tag; the staged provenance for
# that tag is the measured commit. --commit still wins when given.
if commit in ("", "-"):
    tag = os.path.basename(run.rstrip("/")).split("-")[0]
    meta = os.path.join(run, "run.meta")
    if os.path.exists(meta):
        for line in open(meta):
            if line.startswith("tag="):
                tag = line.split("=", 1)[1].strip()
    stage_root = os.environ.get("SPARKFLASH_STAGE_ROOT", "/home/user/ev-session-window")
    provenance = os.path.join(stage_root, tag, "stage", "provenance.txt")
    if os.path.exists(provenance):
        commit = open(provenance).read().strip()[:12]
means = []
for r in range(4):
    path = os.path.join(run, f"rank{r}", f"tp4-token-r{r}.json")
    if not os.path.exists(path):
        sys.exit(f"speed-log: missing {path}")
    data = json.load(open(path))
    means.append(float(data["step_ms"]["mean_ms"]))
tokens = {json.dumps(json.load(open(os.path.join(run, f"rank{r}", f"tp4-token-r{r}.json")))["tokens"]["ids"]) for r in range(4)}
shas = {json.load(open(os.path.join(run, f"rank{r}", f"tp4-token-r{r}.json"))).get("logits_dump", {}).get("bin_sha256") for r in range(4)}
tokens_ok = "yes" if len(tokens) == 1 else "NO"
sha_ok = "yes" if len(shas) == 1 and None not in shas else "NO"
with open(csv_path, "a", newline="") as handle:
    writer = csv.writer(handle, lineterminator="\n")
    writer.writerow([datetime.date.today().isoformat(), commit, os.path.basename(run.rstrip("/")),
                     f"{statistics.fmean(means):.2f}", tokens_ok, sha_ok, note])
print(f"logged {os.path.basename(run.rstrip('/'))}: {statistics.fmean(means):.2f} ms/step "
      f"(tokens {tokens_ok}, logits sha {sha_ok}) -> {csv_path}")
PY
    ;;
  *) usage;;
esac
