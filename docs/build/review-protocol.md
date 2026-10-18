# Review protocol for merge requests

Status: recorded 2026-10-18 on the Lead's instruction, after the review cycle this file's sibling
(`review-2026-10-18-feedback.md`) audits. It states the protocol the team is expected to follow; it is
not a description of a tool that enforces it.

## The protocol

1. **Every merge request is reviewed by a fresh adversarial critic.** "Fresh" means a reviewer with
   an independent context and no authorship of the change - it starts from the diff, the task text
   and the repository, not from the author's summary or this session's history. "Adversarial" means
   the reviewer's job is to falsify the change's claims: find the claim that is not backed by a test
   or a measurement, the line that looks moved but changed, the doc that is now stale, the number
   that was copied from a task text instead of measured.
2. **Iterate until the critic approves.** The author answers each finding with a fix, a
   demonstration that it is already true, or a reasoned rejection; then the critic re-reviews. The
   MR merges only after the critic approves. A finding that is answered by "will fix later" is not
   approved; it is either fixed or explicitly recorded as an open, owned follow-up.
3. **Only the Lead merges.** Teammates push branches and report; the Lead owns the merge point and
   the final diff/test run.

## What the critic checks, at minimum

These are the repository's own house rules, so the review can name the rule it is enforcing:

* every claim is tied to a test, a measurement or a specification line; nothing is reported as
  measured that did not run;
* measured numbers carry the commit they were measured at, and a number that disagrees with the task
  text is reported with the measured value (this cycle: `m7_layer0.rs` measured 8,224 lines, the
  task said 8,100);
* hypotheses and readings are labelled as such; a device claim without a device run says so;
* a non-trivial behaviour change updates `docs/`, and an architectural one adds a DECISIONS entry;
* the diff is inside the declared write scope, and anything outside it is called out and justified;
* the verification commands in the report actually pass on the branch under review, including
  `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings` and
  `cargo test --workspace`.

## Documentation-only deltas are non-blocking

A delta is documentation-only when it changes only `docs/**` files or comment text and no code,
test assertion, string the code prints, or build/CI behaviour. Fixing a path or a line citation in a
comment is documentation-only; changing an error-message string is not (it is a behaviour the tests
may pin). A documentation-only finding does not block the merge. It is still recorded in the MR
description as a follow-up with an owner, and it must not be used to carry a code change: the moment
a doc claim is only true if code changes, the code change is the real work and it is blocking.

## Provenance and limits

The protocol is the Lead's instruction of 2026-10-18; this file records it so a new reviewer can
follow it rather than re-derive it. It has no automated enforcement - the critic, the Lead and the CI
jobs are the enforcement. The `review-2026-10-18-feedback.md` audit in this directory is a worked
example of the claim-by-claim half.
