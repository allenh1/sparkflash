---
name: adversarial-review-loop
description: Run a branch or change through repeated independent adversarial reviews until a fresh critic approves. Use when a task asks for review-driven iteration, a critic loop, "don't self-review", or when a large branch needs an unbiased correctness and architecture gate.
---

<!--
Copyright (c) 2026 Hunter L. Allen

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Adversarial review loop

The point of this loop is **independence**, not agreement. Each round gets one fresh critic who
sees the current diff and nothing about earlier rounds. The lead fixes what is real, proves it,
and only then asks a new critic.

## Non-negotiable rules

- **One isolated worktree.** Never review or edit the primary checkout. Record the branch and the
  base commit before the first review.
- **Fresh critic every round.** Use a new subagent, not the previous critic and not a fork of the
  conversation. A forked agent inherits the lead's assumptions and prior findings.
- **No self-review.** While the critic works, do not "also" review the same diff or fix things
  they have not reported. Wait.
- **No pressure words.** Never say "final", "ship it", "last round", "this should pass now", or
  give an iteration count. Do not tell the critic what earlier critics found.
- **Do not busy-wait.** After spawning, end the turn. The runtime wakes you when the critic
  finishes. Use a job wait only when you are genuinely blocked.
- **Commit before review.** The critic diffs the branch against the base; uncommitted files are
  invisible. Leave the worktree clean while they read.
- **Rebase before each new round.** Fetch the latest main, rebase the branch, resolve/build
  fallout, rerun the affected tests, then spawn the critic. Do not ask a critic to review a stale
  diff.
- **Approval is the exit.** Stop when a fresh critic, looking at the current HEAD, returns
  `APPROVE` with no blocking findings. A stale approval, or an approval of an earlier commit,
  does not count.

## Round workflow

1. **Set up the review target**
   - Create the worktree, check out the branch, fetch and fast-forward/rebase.
   - Record: worktree path, branch, base SHA, one-paragraph scope.
   - Run the relevant build/tests once so the critic is not the first person to discover a broken
     tree. Fix mechanical breakage before review.

2. **Spawn one fresh critic**
   - Use `subagent` with a new name. Give it the worktree, branch, base, scope, rules to check,
     commands it may run, and the report format. Do not attach the lead's conclusions.
   - Tell it to be read-only and not to use sandbox escalation.

3. **Wait**
   - End the turn. Do not poll. When the completion arrives, read the verdict and every finding.
   - If the critic fails before finishing, spawn another fresh critic with the same prompt. Do
     not reuse the failed name.

4. **Triage**
   - Reproduce or verify every finding against the code and tests before changing anything.
   - Classify: blocking correctness/architecture issue, integration failure, test gap, or
     documentation drift. Fix blocking issues; fix cheap non-blocking issues if they reduce risk.
   - If a finding is wrong, prove it with a test, a command, or a precise code path; do not just
     assert the critic is wrong.
   - Never "fix" a finding by weakening the assertion, deleting the honest skip reason, or
     changing the docs to describe the bug as intended.

5. **Fix and prove**
   - Make the change in the worktree, add or update the test that would have caught it, run the
     affected suites and repo gates, and commit. Keep commits scoped to the round so the diff is
     reviewable.
   - Rebase onto the latest main. Rerun the tests the rebase touched.

6. **Spawn the next fresh critic**
   - New name, same neutral prompt, current HEAD. Do not mention prior rounds or the fixes.

## Reviewer prompt

Use this verbatim, filling only the bracketed fields. Keep the "first look" framing.

```text
You are reviewing a code branch in an isolated git worktree. Give it a fresh, adversarial first
look.

Repository / worktree: [PATH]
Branch: [BRANCH] (HEAD is the tip)
Base: origin/main at [SHA] (use `git merge-base origin/main HEAD`)

[ONE PARAGRAPH SCOPE: what the branch adds, and which files/areas matter.]

Before reviewing:
- Read [PROJECT DOCS] and load the [SKILL] skill with the skill tool.
- Inspect `git log --oneline $(git merge-base origin/main HEAD)..HEAD` and
  `git diff $(git merge-base origin/main HEAD)..HEAD`, including new files.

What to check:
- The project's hard architectural rules (dependencies, kernel location, typed config/profile,
  plans/streams, no silent fallbacks).
- Correctness paths and their error handling: indexing, arithmetic, state machines,
  concurrency, unsafe code, silent fallbacks, overflow.
- Integration: registry/control plane, build gates, catalogues, detection, boot validation,
  docs claims.
- Tests: do they exercise the new path? Are device/GPU skips explicit? Do tests assert exact
  output where the contract is exact, rather than substrings?

Run targeted commands where feasible:
[COMMANDS]
If a command cannot run, say so and continue with static review. Do not edit files; this is a
read-only review.

Report:
1. Verdict: APPROVE or REJECT (approve only if no blocking correctness or architecture issue
   remains).
2. Concrete findings ordered by severity, each with file:line, the failure mode, and the
   evidence used.
3. Commands run and their results.
4. Residual risks or gaps you could not verify.

Be specific and skeptical. Do not modify the worktree. Do not use sandbox escalation.
```

## Lead's ledger

Keep a local ledger (outside the branch) with, per round:

- round number, base SHA, HEAD SHA;
- verdict and each finding;
- what was verified, what was fixed, what was rejected and why;
- commands and results;
- the fresh critic name.

The ledger is for the lead. It must not leak into the next critic's prompt.

## Definition of done

- The worktree is clean and the branch is rebased on the latest main.
- The affected suites and repo gates pass, or every skip is explicit and justified.
- A fresh critic, reviewing the current HEAD with no knowledge of prior rounds, returns
  `APPROVE` with no blocking findings.
- Residual risks are written down, not hidden.

## Anti-patterns

- Reusing the same critic because they "already know the code".
- Telling the critic "this is the final check" or how many rounds have run.
- Fixing while the critic reads, then forgetting to rebuild or rerun the tests.
- Turning a red test green by making the assertion laxer.
- Letting a GPU test print a skip and return, then calling the suite green without saying so.
- Reviewing the diff yourself instead of waiting, then biasing the next prompt with your own
  conclusions.
