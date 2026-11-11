# Version 1: the head screen's two fields

This report is the evidence the declaration check asks for when a serving side adds a field to
`sparkflash_model::numerics::NEUTRAL_FIELDS`. It covers `diagnostics.head_screen` and
`diagnostics.head_screen_check`, which joined the list when the verify head moved behind an int8
tensor-core screen. The standard is section 5 of [docs/NUMERICS.md](../NUMERICS.md): a field is
neutral only if a profile may change it without recapturing a reference, and the entry has to say
which statement in the code makes that true.

The head screen is the one entry in that list whose replaced launch is not re-run expression for
expression over everything: it decides which entries the head's arithmetic is run for. The
argument for the two fields is therefore about the decision, not the launch, and it is given in
full below.

### `diagnostics.head_screen`

Read at `TailStage::open` (`crates/sparkflash-deepseek-forward/src/tp4_tail.rs`), which builds
the `HeadScreen`, and used in `head_rows_multi_c`
(`crates/sparkflash-deepseek-forward/src/tp4_forward/rows.rs`). On a step with no constrained
and no truncated (top-k, top-p) row it replaces the one `opt_rows_lm_head` launch with
`opt_head_screen_rows`, `opt_head_screen_mma` and `opt_head_screen_exact`
(`kernels/opt_rows_sm121.cu`); every other step takes the head whole. Unlike most entries here
the replaced launch is not re-run expression for expression over everything: the screen decides
which entries the head's arithmetic is run for. It is neutral because of what is proven about
that decision, not because nothing changed:

- The exact pass is `opt_rows_lm_head` over a subset of the entries: the same weight loads, the
  same 160 FMAs a lane, the same lane sum, the same noise for a temperature-sampled row and the
  same per-warp and cross-warp tie rules. A score is that kernel's bits.
- The subset contains every entry that can be the row's maximum on the rank. The screen computes
  an exact integer dot product of int8 codes of the entry with a two-digit int8 form of the row
  and puts an interval around it that contains the head's f32 logit: the kernel section's header
  derives it (rounding of the 165 additions, the entry's quantisation residual, the row's, and
  the screen's own three roundings), with every stored norm on the safe side. An entry is skipped
  only when its upper bound is below the best lower bound, and the comparisons are written so a
  NaN keeps the entry. A row or an entry the bound does not cover is never ruled out.
- A temperature-sampled row's score is `fmaf(logit, inv_t, noise)`, the noise a function of
  (seed, position, id) alone. A correctly rounded FMA is monotone in the logit for a positive
  `inv_t`, so the same FMA applied to the interval's two ends bounds the score with nothing
  added, and the argument runs on the score. A row whose `inv_t` is not positive and finite is
  never screened.
- Hence the chunk that holds the lowest entry tying the maximum reports the full kernel's
  (value, index), every other chunk a value no larger (and strictly smaller before that chunk),
  and `opt_rows_argmax_fold`'s first-then-strictly-greater rule returns the same pair. Nothing
  else reads the chunk maxima.

Gate: `sparkflash_kernels::head_screen_ref`'s tests run the interval against an emulation of the
head's f32 arithmetic on rows built to stress it, show by mutation that each term is needed
(including a case constructed to be tight), and check the screened arg-max, ties included,
against the full one; `diagnostics.head_screen_check` holds the device to that reference and to
the full head on live traffic, sampled rows included (the reference does not reproduce the
device's `logf`, so the noise is checked there and not on the host). The serve gate's hashes are those of the unscreened path. Caveat,
as for `diagnostics.lean_head`: a rank's losing chunks no longer carry their own maxima, so this
is neutral for a path that serves the arg-max and would not be for one that read them.

### `diagnostics.head_screen_check`

Read at `TailStage::open`. With it the tail refuses to open unless the device's table (every
entry's codes and scale to the bit, its coefficients on the safe side) and a screen of made-up
rows (every digit to the bit, every bound against the exact integer dot product) match
`sparkflash_kernels::head_screen_ref` (`HeadScreen::self_test`), and every screened step is
also run through `opt_rows_lm_head`; the step refuses if any row's folded (value bits, index)
differs, and continues from the full head's maxima. A checked run's output is therefore the
unscreened path's by construction. Gate: the code path (`head_rows_multi_c`'s checked arm). It
is a check beside the served path, and it refuses rather than runs when the screen is off.
