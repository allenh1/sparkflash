#!/usr/bin/env python3
"""Assert the four-rank collective-count agreement from the per-rank run records.

Why this exists: a rank that skips a collective hangs the reduce rather than failing a check, so
the per-rank `collective_check` counter is the only host-visible trace that the four ranks ran the
same sequence. The in-process alternative is deliberately not used - comparing counts by enqueuing
a collective changes the count being checked, and on the buggy rank the extra collective would make
the numbers agree. This reads what each rank already wrote and refuses the row before publication.

Usage:
    check-collective-count.py RANK0.json RANK1.json RANK2.json RANK3.json [--expected N]

Each file is a run record (the loop's evidence JSON) or a whole evidence directory's summary; the
`collective_check` object is extracted from the top level or from the first object that carries it.
Exit 0 only when every rank reports the same `enqueued` value and, if --expected or the records'
own `expected` field is present, that value equals it.
"""
import json
import sys


def find_check(node):
    if isinstance(node, dict):
        if "collective_check" in node and isinstance(node["collective_check"], dict):
            return node["collective_check"]
        for value in node.values():
            found = find_check(value)
            if found is not None:
                return found
    elif isinstance(node, list):
        for value in node:
            found = find_check(value)
            if found is not None:
                return found
    return None


def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def main(argv):
    expected = None
    paths = []
    index = 0
    while index < len(argv):
        if argv[index] == "--expected":
            index += 1
            expected = int(argv[index])
        else:
            paths.append(argv[index])
        index += 1
    if len(paths) < 2:
        print("check-collective-count: need at least two rank records", file=sys.stderr)
        return 2
    checks = []
    for path in paths:
        check = find_check(load(path))
        if check is None:
            print(
                f"check-collective-count: {path} carries no collective_check object; "
                "a record without it cannot testify for its rank",
                file=sys.stderr,
            )
            return 2
        checks.append((path, check))
    if expected is None:
        reported = {int(check.get("expected", -1)) for _, check in checks}
        if len(reported) != 1:
            print(
                f"check-collective-count: the ranks disagree on the expected count: {sorted(reported)}",
                file=sys.stderr,
            )
            return 1
        expected = reported.pop()
    enqueued = [(path, int(check.get("enqueued", -1))) for path, check in checks]
    disagreed = [(path, count) for path, count in enqueued if count != expected]
    if disagreed:
        details = ", ".join(f"{path}: {count}" for path, count in enqueued)
        print(
            f"check-collective-count: REFUSED - the ranks did not enqueue the same sequence "
            f"(expected {expected}): {details}. A skipped collective hangs NCCL instead of failing "
            f"it; this run's numbers must not be published.",
            file=sys.stderr,
        )
        return 1
    ranks = {path: count for path, count in enqueued}
    print(
        f"check-collective-count: OK - {len(ranks)} ranks each enqueued {expected} collectives"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
