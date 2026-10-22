# The m4_abi quarantine split (D26 refactor, task-27)

Status: landed on branch `task/task27-host`, base `0639413` (main). It is a pure refactor of the
frozen Milestone-4 ABI lint: no behaviour change, no test renamed, no scanner edited. The proof is
the byte-identical test list plus the verbatim regeneration in `ci/verify-m4-abi-split.py`.

## Why

D26 froze `crates/sparkflash-kernels/src/m4_abi.rs` as one 13,906-line artifact whose length is
pinned by `ci/check-m4-abi-size.sh`. The audit in
`docs/build/review-2026-10-19-feedback.md` (C4) measured what the length is made of: the const API
that launch, driver and example modules import is ~54 lines, while the rest is source-text scanners
and their tests, compiled into every build. Moving the whole module under `tests/` was rejected
there because production code imports it; the viable change is to keep the API in `src/` and
quarantine the scanners in pinned submodules. task-27 asked for that split with a Lead-owned
decision on the D26 pin.

## Composition, before and after

Before (`0639413`, one file, 13,906 lines, 64 tests):

| lines | content |
| --- | --- |
| 1-848 | module documentation: the freeze notice and the recorded 36-round review history |
| 849-902 | public const API (`COMMON_HEADER`, `MEGA_MHC`, `MXFP8_DENSE`, `MXFP4_MOE`, `FLASH_MLA_CSA2`, `MXFP8_WO_A`, `CERT_LM_HEAD`, `KERNELS`, `header_source`, `consts`, `header_constant`) |
| 905-2964 | root `cfg(test)`: the signature/static-assert/geometry parsers, `pin_helpers`, and the 21 root tests (`m4_abi::*`) |
| 2966-7247 | C/C++ tokeniser and control-flow scanner, including the five `pub fn` entry points and the acquire-guard scans (`acquire_fence_after_poll`, ...) |
| 7249-13,906 | `cfg(test) mod acquire_guard`: its 43 tests and fixture tables (`m4_abi::acquire_guard::*`) |

After (aggregate 13,924 lines, same 64 tests at the same ids):

| file | lines | content |
| --- | --- | --- |
| `src/m4_abi.rs` | 2,979 | the model documentation, the const API, the 21 root tests and helpers, `pin_helpers`, and the glue: `mod scanner;`, the `pub use` re-exports of the five moved public entry points, `#[cfg(test)] use scanner::*;`, `#[cfg(test)] mod acquire_guard;` |
| `src/m4_abi/scanner.rs` | 4,313 | lines 2966-7247 of the base file, verbatim; it gained only `pub(crate)` visibility and the rustfmt re-wraps that the new indentation forces |
| `src/m4_abi/acquire_guard.rs` | 6,632 | lines 7249-13,906 of the base file, verbatim, with the inline module wrapper removed (the module path, and therefore every test id, is unchanged) |

The +18 lines between 13,906 and 13,924 are the glue block, the freeze notice that now names the
aggregate, and formatting; the two moved files are the base text modulo rustfmt and visibility.

## What moved, and what did not

* Moved to `scanner.rs`: the token model and tokeniser, preprocessor conditional masks, literal and
  control-flow analysis, macro definitions/scopes, poll/fence/designator detection, and the five
  public entry points `acquire_fence_after_poll`, `observing_paths_call_the_acquiring_poll`,
  `acquire_helper_fences_unconditionally`, `guarded_names_are_not_redefined`,
  `the_acquiring_poll_is_defined_exactly_once`. The root re-exports the last five, so the public
  paths `m4_abi::...` are unchanged for every caller (`mxfp8_dense_launch.rs`, `m4_quant.rs`,
  `baseline.rs`, the cert/lm-head/Flash-MLA launch modules, `sparkflash-device`, examples).
* Moved to `acquire_guard.rs`: only the `cfg(test) mod acquire_guard` block, at the same module
  path.
* Not moved: the module documentation, the const API, the 21 root tests, `pin_helpers`, every test
  name, every assertion. Three tests read this module family's own source (see below); their
  adapters are the only edited test text.

## Proof 1: the test list is byte-identical

```sh
cargo test -p sparkflash-kernels --lib -- --list | sort > before.txt
# ... split ...
cargo test -p sparkflash-kernels --lib -- --list | sort > after.txt
diff before.txt after.txt
```

`diff` is empty: 338 lines, 336 listed tests, and all 64 `m4_abi` tests at the same ids (21
`m4_abi::*` + 43 `m4_abi::acquire_guard::*`). No line moved.

## Proof 2: the move is a verbatim regeneration of the base commit

`ci/verify-m4-abi-split.py` reads `git show 0639413:crates/sparkflash-kernels/src/m4_abi.rs` (never
a working-tree copy, so a later edit cannot make the verbatim claim true retroactively), cuts it at
the slice boundaries above, applies the glue and the allow-listed adapters, formats the result with
the repo's rustfmt, and requires all three working-tree files to be byte-identical. The allow-list
is exactly two classes of text:

1. self-scan adapters in the moved tests. Three tests in `acquire_guard.rs` read this module family's
   own source: `every_rejection_routes_through_the_one_constructor` (now reads
   `src/m4_abi/scanner.rs` from `CANONICAL_HINT` to EOF, because the guard code it scans moved
   there), `the_documentation_counts_are_derived_from_the_tables` (now uses the family
   `THIS_SOURCE` concatenation, because the tables moved while the documented counts stayed in the
   root docs), and `the_documentation_blocks_have_no_lazy_continuation` (same `THIS_SOURCE` so its
   coverage does not shrink). Their assertions are unchanged.
2. the root freeze notice, which now says the family's aggregate length is pinned and points at
   this document.

Negative control: changing one scanner comment phrase makes the verifier fail at that line
(`scanner: first difference at line 21`); restoring it exits 0.

## Proof 3: D26 is no weaker

`ci/check-m4-abi-size.sh` no longer pins one file. It pins the aggregate over the family and checks
the file set exactly (`m4_abi.rs`, `m4_abi/acquire_guard.rs`, `m4_abi/scanner.rs`): any net line
change, growth or shrink, any added or removed family file, fails. The pin moved 13,906 -> 13,924
in the same commit that performs the split, and D26 in `docs/DECISIONS.md` records the change, the
reason, and keeps 13,906 as the pre-split baseline.

## Gates (host-only; no GPU, no node, no checkpoint, no network)

| gate | command | exit |
| --- | --- | --- |
| fmt | `cargo fmt --all -- --check` | 0 |
| check | `cargo check --workspace --all-targets` | 0 |
| clippy | `cargo clippy --workspace --all-targets -- -D warnings` | 0 |
| rustdoc | `RUSTDOCFLAGS='-D missing_docs' cargo doc --workspace --no-deps` | 0 |
| D26 freeze | `sh ci/check-m4-abi-size.sh` | 0 |
| split proof | `python3 ci/verify-m4-abi-split.py` | 0 |
| workspace tests | `cargo test --workspace` | 0 |
| trace no-default | `cargo test -p sparkflash-trace --no-default-features` | 0 |
| example harnesses | `cargo test -p sparkflash-kernels --examples` | 0 |
| feature sweep self-test | `python3 ci/test_check_features.py` | 0 |
| feature sweep | `python3 ci/check_features.py` | 0 |

The same set was then run once through `python3 ci/host-gates.py`, which writes the machine-readable
artifact (see `docs/build/gate-artifact.md`); its exit codes and suite counts are the batch record.

## Files touched

`crates/sparkflash-kernels/src/m4_abi.rs` (modified), `crates/sparkflash-kernels/src/m4_abi/scanner.rs`
and `crates/sparkflash-kernels/src/m4_abi/acquire_guard.rs` (new), `ci/check-m4-abi-size.sh`,
`ci/verify-m4-abi-split.py` (new), `docs/DECISIONS.md` (D26), this document, and the D26 comment in
`.gitlab-ci.yml`. No kernel, no other crate, no device or node path changed.
