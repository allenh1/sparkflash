//! # FREEZE (D26, 2026-10-18): no new source-text scanners in this module family
//!
//! This module is frozen. It is the Milestone-4 ABI conformance lint, and its own contract below says
//! the scan is "explicitly NOT the proof" - the committed cubin baseline is - while review round after
//! review round kept finding false accepts and false rejections in it (the history below reaches round
//! 36). Each fix bought another parser feature for a proof a text scan cannot make, so the class is
//! closed rather than extended:
//!
//!   * no new scanner, parser or lint for a new kernel invariant is added to this family; a new
//!     invariant goes to compiled-artifact inspection
//!     (crates/sparkflash-kernels/scripts/check-kernel-baseline.sh recompiles every Milestone 4
//!     kernel and compares cubin bytes, sha256 and PTX membar lines against
//!     crates/sparkflash-kernels/baseline/kernel_baseline.txt in the cuda-abi CI job) or to a
//!     runtime device assertion in a harness (the m7_layer0 device stages), where the shipped
//!     binary is what is measured;
//!   * defects in the scanners already here are still fixed, with the mutation proof the file requires,
//!     because they gate the kernels today: the freeze is on length, not on correctness;
//!   * the family's total length is pinned by ci/check-m4-abi-size.sh, wired into .gitlab-ci.yml: the
//!     root file plus the pinned submodules under src/m4_abi/ (the quarantine split recorded in
//!     docs/build/m4-abi-split.md). Any net change fails that check, and a legitimate shrink updates the
//!     pin in the same commit.
//!
//! Decision record: docs/DECISIONS.md D26.
//!
//! Parsed-declaration conformance for the Milestone 4 sm_121a kernels.
//!
//! Every kernel in this milestone is CUDA that cannot run on the development host, so its ABI is
//! pinned by parsing its own source text: the tests below re-derive each geometry number from
//! the #defines, re-parse each __global__ signature from the declaration, re-evaluate each
//! static_assert in Rust, and check that every kernel is wrapped in the trace scopes that
//! specification 5.8 assigns to it. A mutation proof accompanies each parser, because a parser
//! that always returned "fine" would pass every positive test in this file.
//!
//! The three verification gates this milestone can actually close on a GPU-less host are:
//!   1. nvcc compiles each kernel for sm_121a (build.rs: the trace-header probe, the RoCEnante ABI
//!      probe, each of the six kernels and the combined Milestone 4 probe, all to PTX in OUT_DIR; the
//!      cuda-abi CI job asserts the probe PTX exists, and scripts/check-kernel-baseline.sh recompiles
//!      all six against the committed baseline);
//!   2. the static asserts here and in the headers hold under both toolchains;
//!   3. the Rust side re-derives the same shapes, and the CPU references define the numerics.
//!
//! # The acquire guard is a canonical-form checker, not a proof
//!
//! Round after round of review has found a hole or a false rejection here, and every one was real.
//! That is what asking a text scanner for an exhaustive proof that a fence is unconditional on every
//! path produces, so this file no longer claims to be one. The contract is:
//!
//!   * the shipped kernels write the acquire the canonical way, and the canonical spellings are
//!     enumerated below;
//!   * there are THREE lists below, and they are the contract: what the checks accept and pin green,
//!     what they deliberately reject with a reason, and what they admit they cannot see. For everything
//!     the scan can read, a shape in neither the first nor the second list is REJECTED - and the third
//!     list is exactly the set of constructs that can pass unread, which is why it carries what each of
//!     them costs. EVERY rejection this module produces ends with `CANONICAL_HINT`, which names the form
//!     to write instead: the fence's own spelling and placement, a construct the scan cannot certify,
//!     and the helper-shape, definition and translation-unit rules alike. Rejecting an exotic but valid
//!     spelling, or a construct whose body this scan does not read, is therefore a policy decision the
//!     developer resolves by writing the canonical form, not a bug to work around - and that is only
//!     honest because the message says so. The promise is not taken on trust and it cannot go stale,
//!     because it is structural rather than remembered: EVERY rejection is built by the single
//!     constructor `rejection`, which appends the hint when it is absent and refuses in debug a message
//!     that carries it anywhere but at its end. `every_rejection_routes_through_the_one_constructor`
//!     reads this module's own source and fails when a rejection-producing construct does not go
//!     through it, when the hint is named anywhere but in its definition and the constructor, or when
//!     the constructor stops appending it - and `every_rejection_carries_the_canonical_hint` names the
//!     branch each fixture has to reach and asserts the hint on it (round-13 A-1 found a branch without
//!     the hint; round-14 A-2 replaced the hint-scanning version, which four physical mutations
//!     escaped, with the constructor and this routing scan; both are proofed by mutation);
//!   * every claim below is marked with how it is enforced. `pinned` means a test in this file asserts
//!     the verdict and the reason, and the test is named where the claim is made; `measured` means the
//!     number came from one nvcc compile of the shape written beside it and is checked by NO test,
//!     because a cubin byte count cannot be reproduced inside `cargo test`. A measurement is stated as a
//!     measurement, and every number names the shape that produced it, so a claim cannot be read as a
//!     pin it does not have (the round-12 structural finding: two rounds of findings were exactly that
//!     mistake);
//!   * what enforces the compiled acquire is the committed cubin baseline, not this file: any change
//!     to the six kernels' cubin byte count, sha256 or PTX `membar` count fails the `cuda-abi` CI
//!     job, and the on-device tests that would check the acquire at run time do not exist yet. The
//!     lint's job is to catch the mistake early and explain it; it is explicitly NOT the proof;
//!   * the lists below are complete by policy: every spelling the checks accept is listed, every
//!     conservative rejection is listed with its reason, and every construct they cannot see is in the
//!     third list with its measured consequence.
//!
//! # Which cubin numbers are current
//!
//! Every cubin number in this module was re-measured in the CUDA 13.0.3 container (nvcc V13.0.88)
//! when the released mHC arithmetic changed the shared header (spark_released_split_sinkhorn,
//! spark_sigmoid, SPARK_M4_NORM_EPS), and the measured_texts table is the authoritative record: each
//! text's bytes / membar-lines / sha16 triple in that table is the number the checks use and the
//! number the prose quotes for that text. The shared baseline is the released kernel's shipped text,
//! 70432 / 4 / 605321041ac5ec1e: every text whose body compiles to the shipped code carries it,
//! including the round-32 guard texts the merged !20 (main c545ca8) added, which were re-measured
//! under the same released kernel rather than quoted from the pre-change 38640 / 4 / e9cc6998. A few
//! variants that were measured with an extra -DM4_FLAG=1 define, or in an earlier round before the
//! release header existed, cannot be re-derived from the table because no text for them is stored;
//! where those appear they are marked pre-change, their numbers are the historical values that
//! motivated the rule, and nothing current depends on them. A pre-change number quoted beside a
//! current one is history by construction, not a second claim: the table holds one triple per text,
//! and where a prose number and the table disagree, the table wins.
//!
//! The two checks read different amounts of text, and the difference matters where their verdicts are
//! compared (round-12 A1): `acquire_helper_fences_unconditionally` reads the WHOLE body of the one
//! helper that carries the guarantee, before the poll and after it, while `acquire_fence_after_poll`
//! finds the poll, reads the block the poll stands in BEFORE the poll, and then walks forward from the
//! poll for the first `__threadfence()` that is a bare call statement no recognised control flow can
//! skip. Both locate the poll by the SAME rule - the helper's body must MENTION the raw poll exactly
//! once, and that mention must be a CALL of it: an argument list follows the designator, however the
//! designator is parenthesised, dereferenced or addressed, and whatever expression the call stands in
//! (round 18; round 17 corrected the rule the two rounds before it stated) - so the poll's own
//! spelling is not a place they can disagree, and the lint's `poll` argument is the label its
//! messages print rather than a pattern to match.
//!
//! The lint then follows the HELPER's own poll, not the first call of that name in the text: a forward
//! declaration, a function-pointer parameter named like the poll, and an earlier function that polls
//! all name the raw poll without being the observation, and an earlier round anchored on exactly that
//! and reported a `return` inside the poll's own definition as standing before the poll (round-14
//! A-1; each of the three fixtures carries the `#include <stdint.h>` its own types need, because it
//! stands in front of the header's own includes - the text pinned in the test is the text nvcc
//! compiles, and all three compile to the baseline cubin, 70432 / 4 / 605321041ac5ec1e, measured in the CUDA
//! 13.0.3 container - so reading them as a poll was a false rejection; round-15 A-2 found the claim
//! had been measured on a differently placed fixture and made the fixtures self-contained).
//!
//! INSIDE THE HELPER'S BODY the rule is one line (round 17: round 16 stated it wrongly and this is the
//! correction). The body must mention the raw poll exactly once, and that mention must be a CALL of it.
//! Exactly what that means:
//!
//!   * the designator is followed by an argument list, and nothing marks the same tokens as a
//!     DECLARATION of something named like the poll - no type in front of it, in the narrow sense the
//!     call-site check below uses (an identifier that is not a statement keyword, a `>` that closes a
//!     template argument list, a `)` that closes a `decltype`, an attribute-specifier between the type
//!     and the name, or the `(*name)` of a function-pointer declarator);
//!   * nothing else. The ARGUMENTS are not read: `spark_wait_for_flag(f, d)` under renamed parameters,
//!     `spark_wait_for_flag((flag), (deadline))` and a cast argument are the same call, and the shipped
//!     spelling in the messages is an ILLUSTRATION - the hint never tells a writer to use a name that
//!     does not exist. The EXPRESSION the call stands in is not read either: a scan that starts at the
//!     name cannot see the prefix at all, which is what round 16 got wrong when it claimed to require
//!     the canonical tokens exactly.
//!
//! Those are the two lists. ACCEPTED, each measured in the CUDA 13.0.3 container on the mega_mhc kernel
//! (sm_121a, as the committed baseline is), with the cubin its text compiles to:
//!
//!   * the shipped form, `const uint32_t observed = spark_wait_for_flag(flag, deadline);` -
//!     70432 / 4 / 605321041ac5ec1e (the baseline);
//!   * a bare call, a `(void)` cast, an attribute on the statement `[[maybe_unused]] poll(...);`, and
//!     `0u > poll(...)` - all 70960 / 3 / bbafa1b88b3c3be7;
//!   * a unary operator `!poll(...)` and `0u == poll(...)` - both 70432 / 4 / f715d21b4aef0271;
//!   * `poll(...) < deadline` and a chained comparison `0u < poll(...) < deadline` - both 70432 / 3 /
//!     439d36ffff9ee9d5;
//!   * a cast of the result `(uint32_t) poll(...)` and an attribute on the declaration
//!     `[[maybe_unused]] const uint32_t observed = poll(...)` - both 70432 / 4 / 605321041ac5ec1e;
//!   * the argument spellings above, spacing, the renamed-parameter call and the macro alias
//!     (`#define POLL_ALIAS spark_wait_for_flag`, one level) - all 70432 / 4 / 605321041ac5ec1e;
//!   * round 18 - how the DESIGNATOR itself is spelled is not part of the rule either, because
//!     `(poll)(flag, deadline)`, `(*poll)(flag, deadline)` and `(&poll)(flag, deadline)` are the same
//!     call: all three 70432 / 4 / 605321041ac5ec1e;
//!   * round 18 - the poll as the CONDITION of a control statement or an operand of the conditional
//!     operator, where the fence after that whole statement runs unconditionally: `true ? poll(...) :
//!     0u` is 70432 / 4 / 605321041ac5ec1e, `if (poll(...)) {}` is 70960 / 3 / bbafa1b88b3c3be7 and
//!     `while (poll(...)) {}` is 71088 / 3 / a9ef975cf8356c47;
//!   * round 18 - an earlier `<` of a comparison in the poll's own statement, parenthesised
//!     (`(0u < 1u) > poll(...)`) or bare (`0u < 1u > poll(...)`): both 70432 / 4 / f715d21b4aef0271;
//!   * round 19 - a control OPERATOR in an EARLIER, COMPLETED statement, whose own statement ended at
//!     its semicolon so it cannot bound the poll: `const bool ran = (deadline > 0u) && true;`,
//!     `|| true;`, `& true;` and an earlier ternary, all four 70432 / 4 / 605321041ac5ec1e (measured);
//!     and a macro whose replacement list is the designator - `#define P (spark_wait_for_flag)`,
//!     `#define P (*spark_wait_for_flag)`, `#define P (&spark_wait_for_flag)`, all three
//!     70432 / 4 / 605321041ac5ec1e. The parentheses are optional in the replacement list, exactly as they
//!     are around the designator itself, and the object-like/function-like distinction is the
//!     preprocessor's own: `#define P(x) ...` has no space, `#define P (x) ...` does.
//!   * round 20 - three more spellings, all 70432 / 4 / 605321041ac5ec1e: a DOUBLE-parenthesised
//!     replacement `#define P ((spark_wait_for_flag))`; a replacement that is the WHOLE call
//!     `#define P spark_wait_for_flag(flag, deadline)` used as `const uint32_t observed = P;`, where
//!     the observation is the macro's expansion rather than a call at the use; and `#define PRE
//!     (deadline > 0u) && true` used as `const bool ran = PRE;` in an earlier, COMPLETED statement, so
//!     the completed-statement scoping excuses its operator exactly as it excuses the literal operator.
//!     All three are accepted by BOTH checks.
//!   * round 21 - four more spellings, all accepted by BOTH, plus the one the review listed as a false
//!     refusal. `#define P (((spark_wait_for_flag)))` used as `P(flag, deadline)` is the same alias,
//!     because the designator rule unwraps any number of balanced pairs; `#define P(a, b)
//!     spark_wait_for_flag(a, b)` is the whole call with the macro's own parameters in the arguments
//!     (the arguments are not read, at a use or in a replacement list); and `#define P
//!     (spark_wait_for_flag(flag, deadline))` is the whole call WRAPPED in parentheses, refused until
//!     round 21 although `const uint32_t observed = P;` compiles to 70432 / 4 / 605321041ac5ec1e. All four
//!     are 70432 / 4 / 605321041ac5ec1e. The discarded-value spelling of the last one, `(P);` as a whole
//!     statement, is 70960 / 3 / bbafa1b88b3c3be7 - the number every discarded-value call in this list
//!     measures (`bare_call`, `void_cast`, `attribute_stmt`) - so the two spellings' numbers are
//!     stated together rather than one standing for both.
//!   * round 22 - two more accepted shapes, both of them false refusals the review found. A helper body
//!     that carries a fence BEFORE the poll and the acquire after it: the structural check took the
//!     FIRST fence anywhere as the acquire and refused the shape with "precedes the poll" while the lint
//!     accepted it, and the shape is not the baseline - 70448 / 5 / 488031bfabdebb09, the extra fence being the
//!     only delta - so the lint was right. Both checks now take the first fence AFTER the poll
//!     (accepted_fence_before_the_poll). And `#define P (spark_wait_for_flag)(flag, deadline)` used as
//!     `const uint32_t observed = P;`: the whole call with its DESIGNATOR parenthesised, which neither
//!     the designator rule (the name is not the whole replacement list) nor the whole-call rule (the
//!     body does not start with the bare name) read, so both checks refused it although it compiles to
//!     70432 / 4 / 605321041ac5ec1e (macro_parenthesised_designator_whole_call) - the mention rule now skips
//!     the designator's own parentheses and reads the argument list after them.
//!   * round 23 - one more accepted shape, and the root fix that made it one. A fence inside the poll's
//!     ARGUMENTS with the real acquire after the call (accepted_fence_in_the_arguments_and_after, 70280 /
//!     5 / 488031bf, the cubin of the accepted fence-before-and-after shape): the structural check took
//!     the in-arguments fence as the acquire and refused the body while the lint accepted it, because its
//!     fence search started at the poll's designator instead of after the call's argument list. It starts
//!     where the lint's walk starts now, so a fence in the arguments is never the acquire.
//!   * round 24 - and the sentence above is what round 24 had to finish. Skipping the span for fences
//!     also stopped the scan READING it, so control flow inside the poll's own arguments became
//!     invisible: `spark_wait_for_flag((__trap(), flag), deadline); __threadfence(); return observed;`
//!     was ACCEPTED by both checks although it compiles to 69536 / 4 / 5826994128b34ed0, byte-identical to the
//!     trap-before-the-poll statement body both refuse (refused_trap_in_the_poll_arguments), and
//!     `({ return 0u; flag; })` as the first argument compiles to 70464 / 2 / 2b56c510b5f4a945
//!     (refused_return_in_the_poll_arguments). The span is skipped for FENCE detection - a fence written
//!     there runs before the call, so it is never the acquire - and read for control flow, in both
//!     checks. A conditional fence BEHIND an unconditional acquire is accepted now, because without the
//!     define that body preprocesses to the shipped baseline.
//!
//! REFUSED, because a name that is only declared is not an observation: a declaration in any return-type
//! spelling - `uint32_t poll(const volatile uint32_t *, uint64_t);`, `wrapper<uint32_t> poll(...)`,
//! `decltype(0u) poll(...)`, a macro-defined return type, an attribute between the type and the name -
//! and the function-pointer declarator `uint32_t (*poll)(...)`. Every one of those helper bodies omits
//! the call. The spellings that ARE translation units - the `uint32_t` declaration, the `decltype` and
//! source-macro spellings of its return type, the attribute spelling and the function-pointer
//! declarator - compile to 70600 bytes / 3 PTB membar lines / sha 1f2503fd7ecaf189 against the 70264 /
//! 4 / 605321041ac5ec1e baseline (measured), which is why refusing them costs nothing: the code they
//! produce is not the baseline. Two spellings are NOT measured because nvcc refuses them, and are
//! pinned as shapes instead: `wrapper<uint32_t> poll(...)` (the declaration is incompatible with the
//! poll's own declaration) and `uint32_t spark_wait_for_flag(flag, deadline);` (the shipped tokens as
//! a VARIABLE declaration with two initializer values: "too many initializer values"). Every refusal
//! names the call the shared header writes and ends with the canonical hint.
//!
//! Round 23 added three refusals, each with both verdicts and, where the text compiles, a cubin. (a) The
//! shape above with the acquire REMOVED, `spark_wait_for_flag(({ __threadfence(); flag; }), deadline);`,
//! which the structural check used to ACCEPT: it compiles to 70280 / 3 / 18b66cc5f312046c, the cubin of the
//! plain fence-before-the-poll body both checks refuse, so the old verdict was not a function of the
//! compiled shape (refused_fence_in_the_poll_arguments). (b) A poll the preprocessor may not compile
//! beside a second, unconditional poll and the fence (refused_conditional_poll_with_a_second_poll): the
//! baseline 70432 / 4 / 605321041ac5ec1e without -DM4_FLAG=1, and 39072 / 95e0cf41 (pre-change) with it. (c)
//! `void (spark_wait_for_flag)(flag, deadline);` - a mention spelled with a `void` return type. The
//! parentheses around the designator are also the shape of a call through a parenthesised designator, so
//! both checks read it as a call; the tokens are a block-scope REDECLARATION of the poll, and nvcc
//! refuses the translation unit with "error: incomplete type void is not allowed", so there is no cubin
//! to compare and both checks now refuse it with one sentence that names the type and the compiler's
//! answer. Round 24 added three more: a `return`, a control keyword or a call that never returns in the
//! poll's OWN argument list, in either argument, refused by both in the same sentence (the stored texts
//! above, and `(__builtin_unreachable(), flag)` is 68576 / 2 / bfd9e8e177f4f76d, the cubin this documentation
//! already carried for a return before the poll); a conditional fence with nothing unconditional after
//! the poll, refused by both in the same sentence; and `__threadfence(0);`, which ONE predicate now
//! decides on both sides - the CUDA barrier takes no arguments, so the spelling is refused rather than
//! read as a bare call statement.
//!
//! One more REFUSAL, for the other reason (round-18 A8): a poll and a fence inside a NESTED bare block,
//! `{ poll(...); __threadfence(); }`. Its cubin IS the baseline, 70432 / 4 / 605321041ac5ec1e (measured),
//! and it is refused anyway because neither check can relate a block inside the helper's body to the
//! helper's straight-line path: what the helper expands to is then not the fixed poll-then-fence shape
//! the guarantee is stated about. Both checks refuse it with the same words, which is the round-18 fix -
//! the lint used to accept it while the structural check refused it.
//!
//! A third REFUSED family (round-18 A2/A3, corrected in round 19): the poll is called, but the path to
//! the fence is not the one the guarantee is stated about. Each shape has its own reason, and each cubin
//! is measured:
//!
//!   * `if (deadline > 0u) poll(...); __threadfence();` - 71096 / 3 / 7ccedc01e1b172e5: the poll is
//!     evaluated only when the branch is taken, so the fence may follow no observation at all;
//!   * `(deadline > 0u) && poll(...)` - 70432 / 4 / 28939293aa1b8fc2: the same reason, through the
//!     short-circuit operator;
//!   * `return poll(...); __threadfence();` - 70264 / 2 / 42454ad1740a315c: the call here is NOT
//!     conditional, it is the return value; what is wrong is that the fence after it is unreachable,
//!     because the return leaves the helper first. Two PTX membar lines against the baseline's four is
//!     that unreachability, and it is why the reason for this shape is not the one above.
//!
//! A fourth REFUSED family (round 20): two texts where the macro reading and the call reading collide.
//! `#define BAIL return` then `BAIL;` before the poll is a USE of a macro that can bound the path, and
//! nvcc compiles it to 68576 bytes / 2 PTB membar lines / sha bfd9e8e177f4f76d - not the baseline - so
//! refusing it is required; the lint refused it and the structural check accepted it, which was the
//! wrong verdict, and both now say so in the same words. `#define P(spark_wait_for_flag)
//! spark_wait_for_flag` used as `P(spark_wait_for_flag)(flag, deadline)` is refused although nvcc DOES
//! compile it to 70432 / 4 / 605321041ac5ec1e: the scan reads two mentions of the designator - the macro's
//! argument and the macro's own name, whose body is that parameter - and will not guess which one is the
//! observation. That second text is a conservative refusal of a spelling the compiler accepts, and it is
//! written down as one rather than claimed to be a defect. The operator-only macro of the ACCEPTED list
//! is also refused when it stands in the poll's OWN statement, where no `;` separates the use from the
//! observation: `#define MAYBE(x) ((deadline > 0u) && (x))` then
//! `const uint32_t observed = MAYBE(spark_wait_for_flag(flag, deadline));` is 70432 / 4 / 28939293aa1b8fc2, the
//! same code as the literal short circuit it expands to, and both checks refuse it.
//!
//! Round 21 added the text of this family that the review found on the OTHER side of the poll, where the
//! two checks disagreed in the opposite direction. `#define BAIL return` then `BAIL;` AFTER the poll,
//! in a later, completed statement, is refused by the lint and was ACCEPTED by the structural check,
//! whose suspicion rule stopped at "before the poll, or in the poll's own statement" - and it is not the
//! baseline: 69072 bytes / 2 PTB membar lines / sha 18f93bdc42897348, because the fence after the macro
//! is unreachable, so the lint was right and the structural check was wrong. The structural check now
//! reads the span the lint reads - every use before the FENCE - and answers with the lint's own sentence
//! for the position: the pre-poll scan's "the acquire after it runs" before the poll, the forward walk's
//! "the fence runs after it" after it. A use AFTER the fence is read by neither check, because the
//! acquire has already run by then, and that is the one place the scope stops on purpose.
//!
//! Before round 18 the lint refused all three while the structural check refused fewer of them: a `;`
//! between the token and the fence made an operator blocker irrelevant, which excused the
//! short-circuit operand. Round 18 made both checks refuse all three; round 19 scoped the rule so an
//! operator in an EARLIER, COMPLETED statement is not a blocker either (round-19 A1), and gave each
//! shape the reason above. The two verdicts are equal for every text in `measured_texts`.
//!
//! Verdicts are pinned shape by shape in `measured_texts`, whose table
//! `the_helper_body_accepts_every_call_of_the_poll` and
//! `a_declaration_in_the_helper_body_is_not_an_observation` walk for both checks, and whose stored
//! texts `the_stored_measured_texts_are_the_texts_the_tests_build` checks byte for byte under
//! /home/user/wt/targets/kd17_texts. The renamed-parameter call and the macro alias have their own
//! focused test, `a_call_under_renamed_parameters_or_through_an_alias_is_a_call`.
//!
//! The comparison case has its own bounds (round-17 A3, round-18 A4): the hunt for the `<` that would
//! make a `>` a template close stops at a `;`, a brace or a preprocessor line, because a template
//! argument list cannot span one; it skips a BALANCED parenthesised or bracketed group, so an earlier
//! `<` inside the poll's own statement - `(0u < 1u) > poll(...)` - is not read as a template open;
//! and a `<` opens a template argument list only when a template NAME stands before it, so the `<` of
//! `0u < 1u > poll(...)`, which has a literal in front of it, is a comparison. Each of those was a
//! false refusal of a call nvcc compiles: the bound alone left the parenthesised case refused, and the
//! two cases measure the same 70432 / 4 / f715d21b4aef0271.
//!
//! OUTSIDE the helper's body a DECLARATION of the poll is still accepted and ignored - which is where a
//! declaration is uninteresting, because it cannot change what the body does. The call-site check
//! (`observing_paths_call_the_acquiring_poll`) reads a kernel translation unit and rejects a CALL, so
//! it reads a declaration as a declaration, including a function-pointer declaration and one with a
//! `noexcept` specifier, both of which nvcc accepts and which this scan used to report as a bare call
//! (round-15 A6; pinned in `a_declaration_in_a_kernel_is_not_a_call_whatever_its_specifiers`). A call
//! through a function pointer, `(*name)(args)`, is token for token the function-pointer declaration,
//! so this scan cannot separate the two: a stated limit whose guarantee is carried by the helper's own
//! body above.
//!
//! A name the source `#define`s is the one case the tokens cannot settle: `#define uint32_t unsigned`
//! with `uint32_t poll(...);` and `#define BAIL return` with `BAIL poll(...);` are token for token
//! the same shape and have to be answered differently, so the scan reads the macro's BODY - a potential
//! type, unless the body holds a control keyword, a non-returning call, or another such macro. BOTH
//! checks apply that reading to the same span - every USE of such a macro before the fence, in the same
//! words: the lint reads a use before the poll in the poll's block, a use in the poll's own statement,
//! and every use between the poll and the fence in its forward walk, and the structural check reads
//! exactly that span since round 21 (below). A use AFTER the fence is read by neither, because the
//! acquire has already run by then. A body
//! headed by `decltype(` is exempt because it names a type, which is what stopped a macro-defined
//! return type from being read as a macro use (round-15 A3), and a redefinition replaces the mark the
//! previous definition left (round-15 A3), so a name defined again with a harmless body is a type
//! again. That is what keeps the guard's own `#define`d BAIL escapes rejected while a source
//! `#define`d return type is read as one. Every header-level declaration shape is pinned in
//! `a_declaration_is_a_declaration_whatever_its_return_type`, and the cubin of each one nvcc accepts
//! is the baseline (70432 / 4 / 605321041ac5ec1e, measured - the typedef, `decltype` and source-macro
//! declarations and the function-template form of the class-template one; the non-template
//! class-template declaration redeclares the poll with a different return type, which nvcc refuses as a
//! redefinition, so it is pinned as a shape and not measured).
//!
//! They also answer the same way about a fence's spelling and placement, about control flow before the
//! poll, about control flow between the poll and the fence, and about the bodies a fence can sit in - a
//! fence in a `try` body, a fence under an attribute, a fence inside a lambda. The round-21 review asked
//! for the macro-use matrix rather than for a claim, so the matrix is a test:
//! `the_two_checks_agree_on_every_shape_of_macro_use` covers a use before the poll (in an earlier
//! completed statement and in the poll's own statement), after the poll (in the same statement and in a
//! later, completed statement), after the fence, inside a nested block before the poll, inside the
//! fence's own statement, a macro defined after its use, one redefined benign-then-control and
//! control-then-benign, a function-like macro with the designator as a parameter, a nested macro, a
//! macro whose body is a non-returning call, and now a fence before the poll with the acquire after it
//! and a whole call whose designator is parenthesised, and asserts the SAME VERDICT from both checks for
//! every one. Where the two checks claim the same sentence the matrix asserts the whole messages are
//! EQUAL, not merely that they share a substring: every macro-rule and poll-statement refusal is equal
//! byte for byte, and so is every acceptance. The shapes refused by the macro rule share the rule's
//! sentence, and the parameter-swapping alias shares the mention count's: that one used to be a family
//! the two answered in their own voice - the lint saying "this lint" and naming the canonical spelling,
//! this check saying "this scan" and naming the designator - and round 29 ended it by unifying the
//! wording, so the two COUNT messages are byte-identical for the same text and the matrix asserts they
//! are. TWO families still answer differently and the module documentation lists both: the DECLARATION
//! family and the MISSING-POLL family, which round 30 found - a body with no poll reaches the
//! structural check as "does not mention the poll in its body", and the lint as "no poll of that call
//! in the compiled body", two true sentences that are not the same words.
//! `the_missing_poll_family_speaks_in_two_voices` pins both.
//!
//! The round-22 review replaced the claim with a SEARCH, because its 1500-case fuzz found 65
//! disagreements, every one of them the same class: a fence BEFORE the poll was read as the acquire.
//! `the_two_checks_agree_on_every_grid_cell_of_fence_and_macro_placements` therefore crosses the fence
//! placements (none, before the poll, inside the poll's statement, between the poll and the fence, after
//! the fence, before-and-after), the macro placements (none, before the poll, in the poll's own
//! statement, after the poll, after the fence), four macro bodies (a control keyword, a non-returning
//! call, operators only, benign) and five things in front of the poll (nothing, an `if` that returns, a
//! `?` expression, an `&&` expression, a `for` loop) - 600 cells - and asserts the same verdict for every
//! one of them. Round 23 cut the statements in front of the poll to two and moved the grid onto a
//! ten-line fixture, and round 24 removed the 24 cells that were not distinct sources - the "after the
//! fence" macro placement writes the same tokens as "after the poll" when no fence stands between them,
//! so those three fence placements skip it - which leaves 216 cells whose sources a `HashSet` assertion
//! proves distinct. The saving went on positions the grid did not have, each an explicit row rather than
//! only a cell and each asserting the WHOLE message as well as the verdict: a fence in the FIRST
//! argument and in the SECOND, bare and as a statement expression (all four run before the call, so none
//! is an acquire); a fence between the poll and the end of its own statement; a preprocessor-conditional
//! poll with a second compiled poll and on its own; control flow in the FIRST and in the SECOND argument
//! (a non-returning call and a `return`); an unconditional acquire followed by a conditional fence; a
//! conditional acquire with nothing after it; a fence written with arguments; and the void-typed
//! mention. Forty rows in all (round 27 added the eight directive shapes below, the four
//! argument-span shapes, the macro-lambda and the capture). Behind them the grid runs a seeded 160-round corpus of its own pieces,
//! so the round-22 fuzz has a deterministic successor that is reproducible by rerunning the test. The two
//! shapes the round-21 review had me pin for the stricter-direction sentence (a fence inside an
//! unresolved conditional, a fence inside a `try` body) are pinned with a whole-message equality in
//! `the_two_checks_agree_about_the_fence_shapes_the_structural_check_reads`. WHICH BODIES ARE
//! TRANSLATION UNITS: every row of the 81-row matrix and every text in the measured table is a
//! translation unit. The matrix rows are written out with SPARKFLASH_MATRIX_DUMP and compiled in the
//! CUDA 13.0.3 container, and the measured texts are compiled there by the measurement harness, which
//! is the same container: the counts are the matrix dump's 81 compiled / 0 refused and the harness's
//! 116 texts / 0 refused. What the matrix compile proves is PARSEABILITY: the rows carry no entry
//! point, so each row is compiled to an empty cubin, and a row that nvcc rejects is a row this
//! documentation may not quote a cubin for. The cubins quoted in this file are measured on the stored
//! texts of the measured table, which are built from the shipped header by `with_the_shipped_body`,
//! compiled with that substitution and compared against the committed baseline; the matrix rows are
//! written as shapes and claim no cubin of their own (round 25 corrected this sentence: it used to be
//! written about the whole
//! file, and the grid's ten-line fixture, the poll's declaration plus the helper with no includes, is
//! NOT a translation unit: nvcc rejects it for the missing `uint32_t` and `uint64_t`, so neither the
//! 216 grid cells nor the rows of the positions table may claim a cubin, and none of them is described
//! as compiled anywhere in this documentation). The grid's bodies use the device-callable `__trap()`
//! because the matrix rows they mirror are compiled.
//!
//! The sentence that used to stand here - "where they can differ they differ in one direction only,
//! because the structural check is STRICTER about the helper's own shape" - is not kept as a claim, and
//! neither is the round-21 sentence that no shape is known where one refuses and the other accepts: that
//! sentence was WRONG, and the fence before the poll is the shape that falsifies it. It is refused by
//! neither check now, and the two rules that made it a divergence are pinned - the structural check
//! takes the first fence AFTER the poll, and the lint's poll-statement wording is the structural
//! check's wording too. What is left in this documentation when the two checks answer differently is
//! different REASONS for the same verdict. The poll's own placement is not one of those
//! asymmetries either: both
//! checks require the poll to be a top-level statement of the helper's body and refuse a nested one in
//! byte-identical words (round-18 A8). A helper body with more than one compiled poll, and a poll inside
//! a preprocessor conditional that may not hold, are refused by BOTH - and that sentence used to end
//! "with different reasons", which round 23 found to be a divergence rather than a difference of words:
//! the structural check refused the conditional poll while the lint, counting only compiled mentions,
//! accepted any body that held one too, so `#if M4_FLAG` around a first poll with a second unconditional
//! poll compiled under -DM4_FLAG=1 to two real polls (39072 / 95e0cf41 (pre-change), measured with that define) whose
//! one acquire pairs with the second only: a silent second observation. Both checks refuse a poll the
//! scan cannot place now, and with the same sentence. What was left then was the COUNT, where the two
//! checks shared the "2 times" clause and spoke in their own voices about it; round 29 unified that
//! wording as well. Round 25 found that the earlier claim was
//! still too strong and finished the enumeration: the families are now byte-identical on both sides -
//! not-bare-call, missing-fence, larger-expression, nested-body, conditional-fence (both orders), the
//! blocker family (a control keyword or a non-returning call between the poll and the fence), the
//! macro-body family (in the arguments and between the poll and the fence), and the pre-poll family
//! (a literal control statement before the poll, whose message no longer names the helper on one side
//! and not the other). The MATRIX rows and the POSITIONS rows assert WHOLE messages, not substrings,
//! and a family is claimed here only where such a row asserts both messages equal. The MEASURED table
//! pins a different thing and says so: for each compiled text it pins the verdict both checks must
//! give, the measured cubin and a NEEDLE - one sentence of the reason, which is what a sentence-level
//! assertion over a text that is compiled and measured can carry. Its needle cannot be a whole message
//! for the matrix's reason, because the same refusal reaches different texts with different line
//! numbers in it.
//! TWO families still answer in two voices. The DECLARATION family: the structural check says a name
//! that is only declared is not an observation, the lint says the only mention it can see is not a
//! CALL of the poll. And the MISSING-POLL family, which round 30 found: the structural check says the
//! helper body does not mention the poll, so there is no observation for the fence to pair with, while
//! the lint says there is no poll of that call in the compiled body - true of the same text, not the
//! same words. `the_missing_poll_family_speaks_in_two_voices` pins both sentences. Every other family
//! is byte-identical on both sides.
//! The mention COUNT used to be a third such family, in the opposite direction: the structural check
//! said the last mention would be read with no fence after it, which was false about a text with one
//! call and the fence after it, so round 27 replaced that sentence with the lint's own - the scan
//! cannot tell which observation the fence pairs with. Round 28 fixed the documentation and the test
//! comment that still described the old one and round 29 finished the unification: the lint said "this
//! lint" where the structural check says "this scan", so the COUNT was still two-voiced in that word;
//! both say "this scan" now and the matrix asserts the two COUNT messages equal. One reading is
//! conservative rather than
//! complete and is stated as such: a macro whose REPLACEMENT LIST contains a lambda with a return is
//! refused, although nvcc compiles that text to the baseline (70432 / 4 / 605321041ac5ec1e), because the
//! replacement is read textually with no lambda skip; the lambda promise below covers a lambda written
//! in the source, not one hidden inside a macro.
//!
//! WHY THE MACRO PATH IS READ TEXTUALLY, and where that stops: both checks are text scanners with a
//! small `#define` table rather than a preprocessor, so a macro USE is judged by the body the table
//! holds for it at that point. That is what lets `#define BAIL return` then `BAIL;` be refused before
//! the poll (it compiles to 68576 / 2 / bfd9e8e177f4f76d) and what makes the conservative refusal above
//! possible. Round 28 aligned the macro branch with the literal one in the one place the two could be
//! compared at all: a use inside a LAMBDA BODY is now skipped by both checks, exactly as a literal
//! exit inside a lambda body is, so `#define BAIL return` used inside `auto f = [&] { BAIL; };` before
//! the poll is ACCEPTED by both - it compiles to the baseline 70432 / 4 / 605321041ac5ec1e, measured on the
//! stored text `accepted_macro_return_in_a_lambda_before_the_poll`. The lambda blindness that remains
//! is the one inside a macro's REPLACEMENT LIST, which no reading of the use site can see.
//! The two-poll refusal is conservative and its measurement is
//! stated with it: a helper body that calls the poll twice is valid C++ and nvcc compiles it - to 70728
//! bytes / 3 PTX membar lines / sha d5a856a8953895d6, against the 70432 / 4 / 605321041ac5ec1e baseline
//! (measured on the stored text refused_two_polls; the round-13 fixture for this shape duplicated one
//! const and was not a translation unit at all, so the number it quoted belonged to no text) - but the
//! shipped helper observes once and this guard will not certify a second observation on the
//! developer's behalf. Both checks are run on the same text, and the lint's
//! frontier is that text: a text in which it cannot find the helper's compiled poll is refused rather
//! than passed, so the two can diverge only in the direction the stricter structural rule above
//! describes. Every pair of verdicts this documentation names is pinned in
//! `the_documented_asymmetry_is_what_the_tests_assert`.
//! It is a conservative lint; reachability over C++ is not decidable, so its frontier is bounded on
//! purpose. Nothing here is exhaustive, and this is the third list: the constructs the scan admits it
//! cannot see, which are exactly the ones that can therefore pass unread. Each carries the consequence
//! that was measured for it:
//!
//!   * a macro that is not defined in the scanned source (the included header's macros are
//!     invisible here), or a control keyword assembled by token pasting. Both remove real barriers: an
//!     included-header macro that expands to a `return`, and a `##`-assembled one, each compile the
//!     helper to 69072 bytes / 2 PTB membar lines / sha 18f93bdc... against the 70432 / 4 / 605321041ac5ec1e
//!     baseline (measured, not pinned: the two fixtures' VERDICT - that both checks accept them, and so
//!     miss the removed barrier - is pinned in `the_written_claims_are_exact`, and the cubin numbers are
//!     the measurement those two shapes produce; round-11 A4). The names of the macros a source DOES
//!     define are followed to a fixed point, and only a definition that stands BEFORE a use is in
//!     effect at it, because the
//!     preprocessor is positional: `#define OUTER INNER` beside
//!     `#define INNER if (...)` no longer launders a conditional fence, while a definition after
//!     the use is not treated as if it stood at the top of the file (round-10 A-5). A replacement
//!     list of exactly `do { } while (0)` is the one recognised no-op spelling;
//!   * a non-returning callee that is off `NON_RETURNING_CALLS`, a callee reached through a
//!     function pointer or a cast, a macro or inline asm, or a user `[[noreturn]]` annotation. The
//!     test is whether the designator is CALLED, not whether the call is a statement of its own:
//!     `(__builtin_unreachable(), 0u)` in the middle of an expression never returns either, and
//!     requiring a statement was the round-8 A1 hole. Only the LIBRARY function counts: a member
//!     (`s.abort(1u)`), another scope's function (`X::_Exit(1)`), a declaration
//!     (`void abort();`) and a name this source defines as a macro are not it (round-10 A-4), and a
//!     qualified call the scanner cannot resolve (`std::abort()`) is deliberately not a blocker
//!     either. That one is the stated frontier rather than an accident, and it costs nothing: nvcc
//!     refuses a host `abort` call in device code altogether (`calling a __host__ function abort
//!     from a __device__ function is not allowed`), so the permissive reading cannot remove an
//!     acquire from a shipping kernel, while the conservative one rejected real code;
//!   * an unbounded loop, a computed jump, or any reachability that depends on a runtime value;
//!   * a preprocessor conditional it cannot resolve (`#ifdef`, `#if X`, `#if 1 + 1`): the region
//!     is treated as may-not-hold, so a poll there is not found and a fence there is rejected, which
//!     is conservative and never a proof that the region holds. A literal 0 or 1 IS resolved in every
//!     spelling a compiler folds - `#if (1)`, `#if ((0))`, `#if 1u` - because rejecting those was
//!     the round-5 E5 false rejection;
//!   * an arithmetic tautology in a static_assert (`SPARK_X + 0u == SPARK_X`) or any statement
//!     equivalence the textual normaliser does not model.
//!
//! # What the acquire rests on, and the limits this scan accepts
//!
//! No check in this file proves that an acquire exists in the compiled kernel, and none is meant to:
//! each check catches the shapes it names, on the input it is handed, and stops there. The shape
//! that carries the guarantee is the shared header's `spark_wait_for_flag_acquire` - one body that
//! polls and then fences, which nvcc compiles into every observing kernel. The checks here keep that
//! shape from drifting and refuse the rewrites that would make the header irrelevant:
//! `observing_paths_call_the_acquiring_poll` (no kernel translation unit names the raw poll, in the
//! plain, parenthesised or dereferenced spelling), `acquire_helper_fences_unconditionally` (the
//! helper's body is one poll statement followed by one fence statement that always runs - at the top
//! level, alone in a bare block or behind a label - with no control token and no non-returning call
//! between them at ANY brace depth), and the two translation-unit checks
//! `guarded_names_are_not_redefined` and `the_acquiring_poll_is_defined_exactly_once`.
//!
//! What makes the acquire real is therefore not this scan: it is the helper's one-body shape, the
//! nvcc compile of that body into the observing kernel's PTX, and the CPU-side conformance tests. The
//! compile is checked too, rather than assumed: `scripts/check-kernel-baseline.sh` recompiles all
//! six kernels with `-std=c++17 -arch=sm_121a` and compares each cubin's byte count and sha256 and
//! the number of `membar` lines in its PTX with `baseline/kernel_baseline.txt`, which records the
//! toolkit that produced them (release `V13.0.88`, build
//! `cuda_13.0.r13.0/compiler.36424714_0`). A different toolkit fails with "the toolkit changed,
//! regenerate the baseline" instead of reporting a kernel change, so a false failure cannot make the
//! check useless; a moved cubin or membar count is reported per kernel. `tests/kernel_baseline.rs`
//! runs the same script when nvcc is available and says it skipped when it is not, and the cuda-abi
//! CI job runs it as a hard gate. A green guard means "the shapes this file names did not occur".
//!
//! The baseline compile is also what makes the frontier of this lint acceptable. The committed
//! baseline is a CI gate: any change to the six kernels' compiled form - a cubin's byte count, its
//! sha256, or the number of `membar` lines in its PTX - fails the job, and the diff of
//! `baseline/kernel_baseline.txt` is where such a change gets reviewed. A shape the scan cannot
//! read therefore cannot silently remove an acquire from the shipped kernel. What the lint owes is
//! to tell the developer early and precisely, not to be the proof: where it cannot read a shape it
//! refuses it rather than trusting it.
//!
//! The lists below are the contract, and `the_accepted_limits_are_pinned`,
//! `the_holes_and_the_false_rejections_are_closed` and `the_written_claims_are_exact` keep them
//! apart. There are three:
//!
//!   1. the accepted spellings: shapes that are green ON PURPOSE, each pinned green;
//!   2. the conservative rejections: shapes the scan reads and refuses rather than trusts, each pinned
//!      as an error with the reason that makes the refusal honest;
//!   3. the not-seen list above: constructs the scan cannot read at all, each with the compiled
//!      consequence of missing it.
//!
//! A shape the scan can read that is in neither the first nor the second list is REJECTED; an
//! unrecognised spelling does not pass by default. The third list is the only place where something
//! passes unread, which is why it is a list of what it costs rather than a footnote.
//!
//! Every VERDICT and every REASON below is pinned by a test named in this file. Every cubin number
//! below is a MEASUREMENT of the shape written beside it, produced by nvcc (release `V13.0.88`, build
//! `cuda_13.0.r13.0/compiler.36424714_0`, `-std=c++17 -arch=sm_121a`, the same flags
//! `scripts/check-kernel-baseline.sh` uses) and checked by no unit test, because `cargo test` cannot
//! compile a cubin. The one compiled claim that IS a CI gate is the baseline itself: that script
//! recompiles the six kernels and compares each cubin's byte count and sha256 and its PTX `membar`
//! line count with `baseline/kernel_baseline.txt`, `tests/kernel_baseline.rs` runs it whenever nvcc is
//! available and reports a skip when it is not, and the `cuda-abi` CI job runs it as a hard gate.
//!
//! The accepted spellings are:
//!
//!   * the checks read unpreprocessed text: a `#define` of one of the three guarded names in a
//!     source under `kernels/` is rejected by name, and so is a using-declaration or using-directive
//!     that re-binds one of them (round-10 A-2), because both rewrite what the unqualified name
//!     denotes. A use of a macro whose body contains a control keyword, a non-returning call, or the
//!     name of another such macro is rejected too, because the names this source defines are followed
//!     to a fixed point - and only the definitions that stand before the use are in effect at it
//!     (round-10 A-5). A definition in a file this scan does not read, a namespace a using-directive
//!     imports from such a file, and a keyword assembled by token pasting, are still not seen. A macro
//!     whose name is not guarded is not a definition of a guarded name: one whose body spells the raw
//!     poll is caught by the call-site check, which reads directive bodies too, and one that aliases
//!     the helper keeps the acquire in its expansion and is pinned green. A macro whose replacement
//!     list is exactly `do { } while (0)` is read as the no-op it is, so a fence after its use still
//!     runs and `#define abort() do {} while(0)` beside `abort();` is pinned green (round-10 A-4);
//!   * no check reads the translation unit the compiler actually sees, so an `#include` that pulled
//!     a second definition of the helper in from outside `kernels/` would be invisible. That limit
//!     is pinned green with a synthetic kernel that includes an out-of-tree header. What covers the
//!     real tree is the directory walk plus the baseline compile, whose input is the translation unit
//!     nvcc sees. The walk hands every file under `kernels/` to `guarded_names_are_not_redefined` and
//!     to `the_acquiring_poll_is_defined_exactly_once`, and every `.cu` file there to
//!     `observing_paths_call_the_acquiring_poll`: the shared header is not a translation unit, since
//!     it defines and calls the raw poll inside the helper, so it is not handed to the call-site
//!     check;
//!   * a comment may mention any of the three names, and a definitely-disabled `#if 0` region between
//!     the poll and the fence is ignored rather than obeyed, because it executes nothing;
//!   * everything after the fence is not read: both checks stop at the acquire, so the helper's tail -
//!     the `return observed;` the shared header writes, or anything else a developer puts there - is
//!     outside what they claim, and only the fence's own placement and the statements before it are
//!     constrained;
//!   * a fence that is unconditional in a spelling the scan does not have to trust: the designator may
//!     be parenthesised, dereferenced or have its address taken in ANY nesting
//!     (`(__threadfence)();`, `(*__threadfence)();`, `(&__threadfence)();`,
//!     `(*(&__threadfence))();`), the fence may be the only statement of a bare block
//!     (`{ __threadfence(); }`, which is not a branch or a loop body), it may carry a label
//!     (`obs: __threadfence();`) or the global `::` (`::__threadfence();`), an attribute-specifier may
//!     stand in front of it on the same line or on the line above (`[[likely]] __threadfence();`), and a
//!     comment or a blank line around it changes nothing. nvcc compiled both attribute placements to the
//!     baseline cubin byte for byte (70432 / 4 / 605321041ac5ec1e, measured) and the statement-start rule looks
//!     past the whole `[[...]]` rather than reading its closing bracket as an expression (round-12 A4),
//!     which is pinned for both placements in `the_written_claims_are_exact`. A conditional OPERATOR in
//!     an earlier, completed statement (`bool ready2 = ready ? true : false;` before the fence) does not
//!     make the fence conditional either, so it is not a blocker: an operator counts only in the fence's
//!     own statement, and a
//!     `;` that ends that statement is a `;` at the operator's own brace depth, so the `;` inside
//!     a statement expression does not end it (round-10 A-1, a regression of the round-8 A6 scoping
//!     that let `(observed == 0u) && ({ int z = 0; __threadfence(); z; });` through while nvcc
//!     dropped a barrier: 38624 bytes / 3 PTB membar lines / sha d64426bf... (pre-change) against the 70432 / 4 /
//!     605321041ac5ec1e baseline). nvcc compiles the designator wrappers, the bare block, the label, the global
//!     `::` and both attribute placements to the same barriers, so rejecting them was the round-7,
//!     round-8, round-9, round-10 and round-12 false rejections, and each is
//!     pinned green; a fence in a branch or a loop body, a fence in a conditional expression, and an
//!     operator in the fence's own statement are still rejected;
//!   * a fence whose placement a compiler accepts verbatim, which this list now names (round-11 A1):
//!     the fence may be the only statement of a bare block that also holds earlier statements
//!     (`{ int i = 0; (void)i; __threadfence(); }`), it may be the whole body of a statement
//!     expression (`({ __threadfence(); });`), and bare blocks may nest
//!     (`{ { __threadfence(); } }`). nvcc compiles each of those three bodies to the baseline cubin
//!     byte for byte (70432 / 4 / 605321041ac5ec1e), so they are enumerated rather than rejected. A
//!     `co_await` or `co_yield` before the fence is accepted for the same reason the scan gives no
//!     verdict on suspension at all: the fence still runs when the coroutine resumes, and this scanner
//!     does not model coroutines;
//!   * a `try`/`catch` or a `requires`-expression that does not CONTAIN the fence, standing before it
//!     (`try { } catch (...) { }` then `__threadfence();`, or `requires { };` then `__threadfence();`):
//!     both checks accept them, and both shapes are compile failures rather than code - nvcc refuses the
//!     try body in a device function with "device code does not support exception handling" and the
//!     requires prefix with "identifier requires is undefined" (both measured) - so accepting them costs
//!     nothing (round-12 A6, pinned in `the_written_claims_are_exact`). The same constructs HOLDING the
//!     fence are rejected as nested bodies, below;
//!   * a non-returning call this scan can tell apart from the library function, so it is not read as
//!     one: a member call, a qualified call to another scope, a declaration and a name the source
//!     `#define`s (round-10 A-4). The last two are also why `the name is a declaration, not a call`
//!     is a rule about the token before the name;
//!
//! The conservative rejections are shapes the scan refuses rather than trusts, and each is rejected for
//! the reason stated with it. A fence spelled as inline asm and a fence hidden behind a macro of its
//! own leave the body with no `__threadfence()` call at all, so
//! `acquire_helper_fences_unconditionally` reports the body for having none. A QUALIFIED name belongs
//! with them: only the unqualified or globally qualified `__threadfence` is this barrier, so
//! `X::__threadfence();` and `x.__threadfence();` are read as a body with no barrier - which is
//! what they are, and with a namespace member of that name nvcc compiles the no-op and the kernel loses
//! two barriers (70264 bytes / 2 PTB membar lines / sha 42454ad1... against the 70432 / 4 /
//! 605321041ac5ec1e baseline), so certifying them would be worse than refusing them. A fence reached through a
//! function pointer or a CAST expression keeps the name as part of a larger expression rather than as a
//! bare call statement, which is the reason both checks report; nvcc does compile
//! `((void (*)(void))__threadfence)();` to the baseline cubin byte for byte, so this one is a policy
//! rejection - the scanner does not resolve the cast, the message names the canonical form, and the
//! baseline is what settles the compiled acquire (round-10 A-6). A using-declaration or using-directive
//! that imports a guarded name is rejected with the file and the line by all three checks, and the two
//! cases are not the same one (round-10 A-2, corrected in round-11 A5): a using-declaration for a
//! namespace's member (`using X::__threadfence;`) really does re-bind the name, and nvcc compiled that
//! body to 70264 bytes / 2 PTB membar lines / sha 42454ad1... against the baseline, while a
//! using-declaration for the global barrier itself (`using ::__threadfence;`) still names the barrier
//! and nvcc compiles that body to the baseline cubin (70432 / 4 / 605321041ac5ec1e, measured) - so that one is a
//! POLICY rejection, and the message says so instead of claiming the name changed meaning. The BARE
//! spelling is a third case, and what this doc said about it was false: `using __threadfence;` is not
//! namespace-qualified, and nvcc refuses the translation unit with "namespace-qualified name is
//! required" (measured), so refusing it costs nothing and the message says which spelling it was
//! instead of claiming the name still denotes the barrier (round-11 A5, corrected in round-12 A5).
//! Three further spellings each get their own reason rather than that sentence, because nvcc answers
//! each differently (all measured, all compile failures except the last): `using X::template
//! __threadfence;` names a function template - "no instance of function template X::__threadfence
//! matches the argument list"; `using X::__threadfence<int>;` is a template-id - "a template-id is not
//! allowed"; and `using __threadfence = X;` is an ALIAS-declaration, which is not a using-declaration
//! at all: nvcc refuses it when the right-hand side is not a type ("namespace X is not a type name")
//! and COMPILES the re-binding when it is, where the barrier is gone - 70264 bytes / 2 PTB membar lines
//! / sha 42454ad1..., the same cubin as the member re-bind, against the 70432 / 4 / 605321041ac5ec1e baseline
//! (round-13 A-3, pinned in `the_using_spellings_are_refused_with_their_own_reason`).
//! A using-directive whose namespace the scanned source defines is rejected too, and nvcc refuses the
//! ambiguity it produces ("more than one instance of overloaded function __threadfence matches"); a
//! using-directive for a namespace that no scanned source defines is not seen, which is the same limit
//! as a macro from an included header. `throw 0u;` and `co_return;` between the poll and the fence are
//! rejected as control tokens: neither can appear in a shipping device function (nvcc refuses `throw`
//! in device code), so the rejection costs nothing
//! and closes the textual gap. A fence inside a lambda's body or a class, struct, namespace, enum or
//! union body is rejected as a nested body rather than a bare block: it runs only if the closure is
//! called or the member is reached, and nvcc compiled the uncalled closure to 70264 bytes / 2 PTB
//! membar lines / sha 42454ad1... while the called one was byte-identical to the baseline, so even the
//! called spelling is refused by policy and the message says which form to write (round-10 A-3).
//! The `do { } while (0)` idiom is the one control-looking body that is NOT suspicious, and a
//! qualified call to a non-returning name (`std::abort()`) is not treated as a blocker at all; both
//! are the stated policy above, and both are pinned.
//! A conditional construct or a loop between the poll and the fence is rejected even when its body
//! cannot bound the path: `if (observed != 0u) { } __threadfence();` compiles to the baseline cubin.
//! The scan reads placement rather than the body of a construct, so it refuses those instead of
//! certifying them, and the message says exactly that rather than claiming the fence cannot run.
//! An exit inside a body the HELPER DOES NOT OWN is NOT in that class and is accepted, because the
//! exit leaves the lambda or the member, the helper's path still reaches the acquire, and nvcc
//! compiles it to the baseline. Round 27 established it for a lambda body written directly before the
//! poll and round 28 extended it to a lambda body nested at any depth and to the macro branch -
//! `auto f = [&] { return flag; };`, `[&]{ { return; } }`, `[&]{ if (flag) { return; } }` and
//! `[&]{ __trap(); }` before AND after the poll are all stored and measured at 70432 / 4 / 605321041ac5ec1e,
//! while a bare block at statement level stays refused because the `return` in it IS the helper's
//! (70464 / 2 / 2b56c510b5f4a945, measured), as is every attributed block below. Round 31 A2 added the second
//! member of the class: a local TYPE body. `struct S { S() { return; } };`, `struct S { ~S() {
//! return; } };` and `struct S { void run() { return; } };` before the poll are exits of the MEMBER,
//! not of the helper, and nvcc compiles all three to the baseline 70432 / 4 / 605321041ac5ec1e with the
//! acquire present (accepted_return_in_a_local_constructor, ..._destructor, ..._member_function);
//! both checks used to refuse them with the unreachability sentence, which was false of that body.
//! Round 32 A2 extended the same reading to the class-head spellings the closing reviews found
//! refused: a BASE CLAUSE with a comma (`struct S : A, B`, `class S : public A, public B`) and an
//! ATTRIBUTE specifier (`struct [[maybe_unused]] S`, `struct __attribute__((packed)) S`, and
//! `struct alignas(8) S`, which round 32 added after the second review) all stand between the
//! keyword and the brace, and the walk-back stopped at their `,`, `]` or `)`. It skips a balanced
//! bracket group now, and all five spellings are stored and measured at
//! 70432 / 4 / 605321041ac5ec1e (38640 / 4 / e9cc6998 pre-change)
//! (accepted_return_in_a_local_type_with_two_bases, ..._with_public_bases, ..._with_an_attribute,
//! ..._with_a_gnu_attribute, ..._with_alignas).
//! Round 32 A3 is the same rule read in the CALLED direction: a lambda the source CALLS where it
//! stands runs, so a body of its own that holds a non-returning call cannot hand control back and
//! the statement after the call never completes. `(void)([&]{ __trap(); return 0u; })();` before the
//! poll is that shape; nvcc compiles it to 69536 / 4 / 5826994128b34ed0 (37872 / 4 / 54cda5bc pre-change),
//! the cubin this file already refuses as `__trap();` before the poll. That is the
//! ONE-CUBIN-ONE-VERDICT rule this module reads
//! the call direction with: two texts that compile to the same cubin may not be certified
//! differently when the refusal asserts something about the compiled code - here, that the acquire
//! after the call never runs. A CONSERVATIVE refusal asserts less than that ("the scan will not
//! certify this text") and conflicts with nothing: four texts in the store are refused with the
//! baseline 70432 / 4 / 605321041ac5ec1e recorded beside them (38640 / 4 / e9cc6998 pre-change): two
//! for how their macros' replacement lists read textually rather than preprocessed, one for the
//! preprocessor conditional the scan will not resolve, and one, refused_poll_in_a_bare_block,
//! because its poll stands in a nested block rather than a top-level statement.
//! Both checks ACCEPTED it until round 32 - the false-accept direction of the same defect - and
//! both refuse it now (refused_trap_in_a_called_lambda_before_the_poll, measured at that cubin). A
//! `return` in the called lambda returns to the caller, so `(void)([&]{ return; })();` is still
//! accepted, and so is the lambda nobody calls: `auto f = [&] { __trap(); }; (void)f;` before the
//! poll is measured at the BASELINE (accepted_nonreturning_call_in_a_lambda_before_the_poll),
//! because a body that never runs cannot bound the path. A trap reached through a macro, a
//! function pointer or an inline-asm block, and a body that cannot return only because of an
//! unbounded loop, are the same stated limits the non-returning name list carries. So is a closure
//! the source STORES in a variable and calls later - `auto f = [&] { __trap(); }; f();` before the
//! poll - because the reading asks whether the lambda is called where it stands and does not follow
//! the variable the closure is bound to; that body is excused today, and the excuse is a stated
//! limit rather than a measured cubin claim.
//! THE SPELLINGS THE DETECTOR READS. "Any depth inside a lambda" holds for every spelling with a
//! capture list - `[&]`, `[=]`, `[x]`, `[&, x]`, `[]` and the trailing-return forms - and round 30
//! made it hold for the two spellings round 29 had refused conservatively: `[&]<typename T>()` (the
//! walk-back now steps over the balanced template parameter list) and `[&]() -> decltype(auto)` (it
//! steps over the balanced `decltype(...)` in the return type). Round 32 A4 pinned the two specifier
//! spellings that were accepted and untested: `[&]() noexcept { return; }` and
//! `[&]() mutable { return; }` are stored and measured at 70432 / 4 / 605321041ac5ec1e
//! (38640 / 4 / e9cc6998 pre-change; accepted_noexcept_lambda_with_a_return,
//! accepted_mutable_lambda_with_a_return). Both of round 30's
//! spellings are stored ACCEPTED texts as well
//! (accepted_template_parameter_list_lambda_with_a_return and
//! accepted_decltype_auto_lambda_with_a_return), and nvcc compiles all four to the BASELINE
//! 70432 / 4 / 605321041ac5ec1e, as it always did - the round-29 refusals said the acquire after the exit was
//! unreachable, and that sentence was false of the body nvcc compiled.
//! Round 31 A1 closed the way that claim was still false: the walk-back kept ONE depth counter for
//! round, square and angle brackets, so a `<` or `>` written inside a parenthesised or
//! square-bracketed group - a COMPARISON in a type, not a delimiter - cancelled the `)` or `]` of the
//! group it sat in, the scan never reached the arrow, and both checks refused the whole spelling with
//! the unreachability sentence. Each bracket kind has its own counter now, a bracket nested in
//! another kind of group is ignored, and the angle matcher skips the groups nested in a template
//! argument or parameter list. Eleven stored and measured texts pin it at 70432 / 4 / 605321041ac5ec1e:
//! `[&]() -> std::enable_if_t<(1 < 2), int> { return 0; }` and its `(2 > 1)` twin, five
//! `decltype(...)` forms (`1 << 2`, `(1 < 2)`, `(1 <= 2)`, `(1 > 2)`, `(1 >= 2)`), and
//! `[&]<typename T = std::enable_if_t<(1 < 2), int>>()`, the same with `(1 << 2) != 0`, plus
//! `[&]<auto N = (1 < 2)>()` and `[&]<auto N = (1 << 2)>()` in a template parameter list. A genuine
//! `>` at the top level of the walk-back is still read as a template close, which is what keeps the
//! acceptance and the refusal both honest; a requires-clause is NOT read (nvcc 12.9 refuses that
//! spelling itself, so it has no cubin to pin). Round 32 A1 closed the last of it, and it was a
//! test-quality defect before it was a code one: every one of round 31's angle texts wrote its
//! comparison PARENTHESISED, so the guard had never been asked about a BARE one, and
//! `std::enable_if_t<1 < 2, int>`, `1 <= 2` and `1 >= 2` written bare in a template argument or
//! parameter list were still refused with the unreachability sentence while nvcc compiled all five
//! to 70432 / 4 / 605321041ac5ec1e (38640 / 4 / e9cc6998 pre-change). In BOTH angle walks a `<` is a
//! delimiter only when a template NAME stands in front of it, and the `>` of a `>=` is a comparison
//! rather than a close: the guard is
//! the one function round-18 A4 wrote for the call reading, now with three call sites. The five
//! bare forms are stored and measured (accepted_bare_less_than_in_a_trailing_return, the
//! ..._less_equal one beside it, and the three ..._in_a_template_parameter_list forms).
//! AN ATTRIBUTE IS NOT A CAPTURE LIST, and `]]` alone does not decide it. Round 28's outward walk read
//! a compound statement carrying an attribute as a lambda body, because `[[maybe_unused]]` closes with
//! two `]` tokens: the walk saw the first `]` before the brace and took it for a capture list - the
//! round-29 regression. Round 29 then refused any brace preceded by `]]`, and round 30 found that cuts
//! the other way: an init-capture whose initialiser ends in a subscript (`[x = (&deadline)[0]]`) and a
//! capture list with a trailing attribute (`[&] [[maybe_unused]]`) both end `]]` and both are LAMBDAS
//! that nvcc compiles to the baseline. The test is what the bracket group IS: `an_attribute_closes_at`
//! skips a balanced `[[...]]` pair - adjacent brackets on both sides, which an init-capture list is
//! not - and the walk-back keeps looking for the capture list before it. So `[[maybe_unused]] { {
//! return 0u; } }` before the poll, the same with an `if` and with a `while`, and the after-poll
//! `[[maybe_unused]] { { return observed; } }` are stored, measured and REFUSED (70464 / 2 /
//! 2b56c510 and 70264 / 2 / 42454ad1740a315c), as is the plain nested block `{ { return 0u; } }` that is the
//! control, while accepted_init_capture_ending_in_a_subscript and
//! accepted_capture_list_with_a_trailing_attribute are stored and measured at the baseline
//! 70432 / 4 / 605321041ac5ec1e.
//! The tokens that really do bound the path stay rejected: a `return`, `goto`, `throw` or `co_return`
//! outside a lambda body, a loop, and a non-returning call. The same tokens BEFORE the poll are rejected
//! too, by both checks and with the same reason - the structural check reads the whole helper body and
//! the lint reads the block the poll stands in, and the lint accepting them was the second disagreement
//! round-12 A1 found (both verdicts pinned in `the_written_claims_are_exact`).
//! A nested fence does not stop either check while a top-level fence follows it: `auto l = [&] {
//! __threadfence(); }; (void)l; __threadfence();` is accepted by both checks, and nvcc compiled it to the
//! baseline cubin byte for byte (70432 / 4 / 605321041ac5ec1e, measured); the lint used to reject the lambda's
//! fence before it ever reached the real acquire, which was the first disagreement of round-12 A1.
//! A fence inside a `requires`-expression or a `try` block is rejected as a nested body, and the two
//! checks agree about both - the lint used to step into a `try` block and accept the fence there while
//! the structural check refused it (round-11 A1). An attribute between the head and the brace changes
//! neither verdict: `try [[likely]] { __threadfence(); }` and `catch (...) [[likely]] {
//! __threadfence(); }` are refused by both checks as well, which they were not while both
//! determinations read the attribute's closing bracket as the token before the brace (round-36 A1).
//! Neither shape can ship: a `requires`-expression in that
//! position is not valid C++ ("identifier requires is undefined"), and the attributed `try` and
//! `catch` spellings do not parse at all - the grammar has no attribute there, and g++ 16 and clang 22
//! reject both in c++17, c++20, c++23 and c++2c - so no compiler ever reaches the "device code does not
//! support exception handling" refusal the plain spellings get. Refusing them costs nothing.
//! Every rejection in this module ends with the canonical hint, and `every_rejection_carries_the_-
//! `canonical_hint` asserts that branch by branch: the lint's missing-poll, macro, non-returning,
//! spelling, placement, conditional and pre-poll branches, the structural check's shape rules (no poll,
//! two polls, a poll that is not top-level, a fence before the poll, a fence in a preprocessor
//! conditional, no helper body), and the call-site, `#define` and definition-once rules. The lint's
//! non-returning branch, both of its macro branches, the structural check's pre-poll branch and its
//! helper-shape and definition branches carried no hint until round-12 A2 closed that gap; the sweep
//! test is what keeps the promise from going stale again.
//! The PTX `membar` count is not monotone with correctness, which is why the gate compares bytes and
//! sha256 as well - and both numbers here are MEASUREMENTS of the shape written beside them, not pins:
//! `__threadfence();` followed by `if (observed != 0u) { } else { __threadfence(); }` compiles to 70448
//! bytes / 5 PTB membar lines / sha 488031bf... which is MORE membar lines than the 70432 / 4 baseline
//! while the acquire is present, and `if (observed != 0u) { __threadfence(); }` with the top-level fence
//! deleted compiles to 38624 bytes / 3 PTB membar lines / sha e5b9d6cc... (pre-change), which is FEWER lines with the
//! acquire gone (round-12 A3 re-measured both, after the round-11 sentence attributed the 5-line number
//! to the wrong shape). A gate that watched only the count would call the first of those safer and the
//! second unsafe.
//! `the_accepted_limits_are_pinned`, `the_holes_and_the_false_rejections_are_closed` and
//! `the_written_claims_are_exact` assert each reason, not merely that something failed. Refusing to
//! certify a shape the scan cannot read is not a hole; it is the guard declining to guess.

use crate::c_decl::Consts;

mod scanner;
pub use scanner::{
    acquire_fence_after_poll, acquire_helper_fences_unconditionally,
    guarded_names_are_not_redefined, observing_paths_call_the_acquiring_poll,
    the_acquiring_poll_is_defined_exactly_once,
};

#[cfg(test)]
use scanner::*;

#[cfg(test)]
mod acquire_guard;

/// The shared header all six kernels include: geometry, codecs, Sinkhorn-Knopp, sync helpers.
pub const COMMON_HEADER: &str = include_str!("../../../kernels/m4_common_sm121.cuh");
/// Mega-mHC, the fused single-pass hyper-connection kernel of specification 2.4.
pub const MEGA_MHC: &str = include_str!("../../../kernels/mega_mhc_sm121.cu");
/// The MXFP8 dense twin of specification 4.1.
pub const MXFP8_DENSE: &str = include_str!("../../../kernels/b12x_mxfp8_dense_sm121.cu");
/// The MXFP4 routed-expert MoE combine of specification 2.1.
pub const MXFP4_MOE: &str = include_str!("../../../kernels/b12x_mxfp4_moe_sm121.cu");
/// FlashMLA with CSA2 candidate selection of specification 2.3.
pub const FLASH_MLA_CSA2: &str = include_str!("../../../kernels/flash_mla_csa2_sm121.cu");
/// The grouped wo_a output projection, MXFP8 with the same B12x layout.
pub const MXFP8_WO_A: &str = include_str!("../../../kernels/b12x_mxfp8_wo_a_sm121.cu");
/// The certified LM head of specification 2.1: a bound screen plus exact dots.
pub const CERT_LM_HEAD: &str = include_str!("../../../kernels/cert_lm_head_sm121.cu");

/// Every M4 kernel source, by file name. The list is the contract of section 6.2, line 1337.
pub const KERNELS: [(&str, &str); 6] = [
    ("mega_mhc_sm121.cu", MEGA_MHC),
    ("b12x_mxfp8_dense_sm121.cu", MXFP8_DENSE),
    ("b12x_mxfp4_moe_sm121.cu", MXFP4_MOE),
    ("flash_mla_csa2_sm121.cu", FLASH_MLA_CSA2),
    ("b12x_mxfp8_wo_a_sm121.cu", MXFP8_WO_A),
    ("cert_lm_head_sm121.cu", CERT_LM_HEAD),
];

/// The text of the shared header, for the modules that check their own constants against it.
#[must_use]
pub fn header_source() -> &'static str {
    COMMON_HEADER
}

/// An evaluator over the #defines of the header plus every kernel.
#[must_use]
pub fn consts() -> Consts {
    let mut table = Consts::parse(COMMON_HEADER);
    for (_, source) in KERNELS {
        table.extend(source);
    }
    table
}

/// One #define of a source, evaluated as an integer constant expression.
///
/// # Panics
/// Panics when the name is absent or not an integer expression.
#[must_use]
pub fn header_constant(source: &str, name: &str) -> u64 {
    let mut table = Consts::parse(source);
    table.extend(COMMON_HEADER);
    u64::try_from(table.value(name))
        .unwrap_or_else(|_| panic!("#define {name} is not a non-negative integer"))
}

/// The parameter shape of one kernel, as the ABI tests expect it.
#[cfg(test)]
struct ExpectedParam {
    raw_type: &'static str,
    name: &'static str,
}

/// Re-parses a kernel's declaration and compares the whole parameter list, in order, including
/// pointer constness and the __restrict__ qualifiers: a change to any of them is an ABI break for
/// the launcher, which is written in Rust and would not be recompiled by a CUDA-only change.
#[cfg(test)]
fn assert_signature(source: &str, kernel: &str, expected: &[ExpectedParam]) {
    let decl = crate::c_decl::kernel_declaration(&crate::c_decl::strip_comments(source), kernel);
    assert!(decl.extern_c, "{kernel} must be extern \"C\"");
    assert!(decl.global, "{kernel} must be __global__");
    assert_eq!(
        decl.params.len(),
        expected.len(),
        "{kernel} parameter count"
    );
    for (index, (param, want)) in decl.params.iter().zip(expected).enumerate() {
        assert_eq!(
            param.raw_type, want.raw_type,
            "{kernel} parameter {index} type changed (name {})",
            param.name
        );
        assert_eq!(
            param.name, want.name,
            "{kernel} parameter {index} name changed"
        );
    }
}

#[test]
fn mega_mhc_signature_is_pinned() {
    assert_signature(
        MEGA_MHC,
        "mega_mhc_fused_bf16_sm121",
        &[
            ExpectedParam {
                raw_type: "__nv_bfloat16 *__restrict__ x",
                name: "x",
            },
            ExpectedParam {
                raw_type: "const __nv_bfloat16 *__restrict__ y_prev",
                name: "y_prev",
            },
            ExpectedParam {
                raw_type: "const float *__restrict__ b_prev",
                name: "b_prev",
            },
            ExpectedParam {
                raw_type: "const float *__restrict__ c_prev",
                name: "c_prev",
            },
            ExpectedParam {
                raw_type: "const float *__restrict__ a_prev",
                name: "a_prev",
            },
            ExpectedParam {
                raw_type: "const float *__restrict__ predictor",
                name: "predictor",
            },
            ExpectedParam {
                raw_type: "const float *__restrict__ hc_scale",
                name: "hc_scale",
            },
            ExpectedParam {
                raw_type: "const float *__restrict__ hc_base",
                name: "hc_base",
            },
            ExpectedParam {
                raw_type: "const float *__restrict__ rms_weight",
                name: "rms_weight",
            },
            ExpectedParam {
                raw_type: "uint8_t *__restrict__ xhat_fp8",
                name: "xhat_fp8",
            },
            ExpectedParam {
                raw_type: "uint8_t *__restrict__ xhat_scales",
                name: "xhat_scales",
            },
            ExpectedParam {
                raw_type: "float *__restrict__ hc_stats",
                name: "hc_stats",
            },
            ExpectedParam {
                raw_type: "uint32_t *__restrict__ hc_sync",
                name: "hc_sync",
            },
            ExpectedParam {
                raw_type: "float *__restrict__ hc_coeffs",
                name: "hc_coeffs",
            },
            ExpectedParam {
                raw_type: "float *__restrict__ hc_rms_out",
                name: "hc_rms_out",
            },
            ExpectedParam {
                raw_type: "float *__restrict__ xhat_normed",
                name: "xhat_normed",
            },
            ExpectedParam {
                raw_type: "GpuTraceRing *__restrict__ trace_ring",
                name: "trace_ring",
            },
            ExpectedParam {
                raw_type: "const uint32_t layer_id",
                name: "layer_id",
            },
            ExpectedParam {
                raw_type: "const uint64_t step_seq",
                name: "step_seq",
            },
        ],
    );
}

/// The complete parameter list of every Milestone 4 kernel, as one string in declaration order,
/// with every qualifier the ABI depends on: pointer-ness, const-ness of the pointee, __restrict__,
/// the element type and the parameter name. Pinning the joined text rather than a list of fields
/// keeps the pin readable while still failing on a dropped qualifier, a reordered pair of
/// parameters, or a renamed parameter, which is what the Rust and CUDA halves must agree on.
#[cfg(test)]
const SIGNATURES: [(&str, &str, &str); 6] = [
    (
        "mega_mhc_sm121.cu",
        "mega_mhc_fused_bf16_sm121",
        "__nv_bfloat16 *__restrict__ x; const __nv_bfloat16 *__restrict__ y_prev; const float *__restrict__ b_prev; const float *__restrict__ c_prev; const float *__restrict__ a_prev; const float *__restrict__ predictor; const float *__restrict__ hc_scale; const float *__restrict__ hc_base; const float *__restrict__ rms_weight; uint8_t *__restrict__ xhat_fp8; uint8_t *__restrict__ xhat_scales; float *__restrict__ hc_stats; uint32_t *__restrict__ hc_sync; float *__restrict__ hc_coeffs; float *__restrict__ hc_rms_out; float *__restrict__ xhat_normed; GpuTraceRing *__restrict__ trace_ring; const uint32_t layer_id; const uint64_t step_seq",
    ),
    (
        "b12x_mxfp8_dense_sm121.cu",
        "b12x_mxfp8_dense_sm121",
        "const uint8_t *__restrict__ a_fp8; const uint8_t *__restrict__ a_scales; const uint8_t *__restrict__ w_fp8; const uint8_t *__restrict__ w_scales; __nv_bfloat16 *__restrict__ out; const uint32_t rows; const uint32_t cols; const uint32_t inner; GpuTraceRing *__restrict__ trace_ring; const uint32_t layer_id; const uint64_t step_seq",
    ),
    (
        "b12x_mxfp4_moe_sm121.cu",
        "b12x_mxfp4_moe_sm121",
        "const uint8_t *__restrict__ act_fp8; const uint8_t *__restrict__ act_scales; const uint8_t *__restrict__ w_fp4; const uint8_t *__restrict__ w_scales; const uint32_t *__restrict__ expert_ids; float *__restrict__ slot_acc; const uint32_t tokens; GpuTraceRing *__restrict__ trace_ring; const uint32_t layer_id; const uint64_t step_seq",
    ),
    (
        "flash_mla_csa2_sm121.cu",
        "flash_mla_csa2_sm121",
        "const __nv_bfloat16 *__restrict__ q; const uint8_t *__restrict__ kv_fp4; const uint8_t *__restrict__ kv_scales; const uint8_t *__restrict__ window_codes; const uint8_t *__restrict__ window_scales; const uint32_t window_kv_format; const uint32_t window_rows; const uint32_t *__restrict__ selected; const __nv_bfloat16 *__restrict__ index_key; const __nv_bfloat16 *__restrict__ index_query; const float *__restrict__ sink; float *__restrict__ block_scores; uint32_t *__restrict__ block_best; __nv_bfloat16 *__restrict__ out; const uint32_t position; const uint32_t selected_count; GpuTraceRing *__restrict__ trace_ring; const uint32_t layer_id; const uint64_t step_seq",
    ),
    (
        "b12x_mxfp8_wo_a_sm121.cu",
        "b12x_mxfp8_wo_a_sm121",
        "const uint8_t *__restrict__ attn_fp8; const uint8_t *__restrict__ attn_scales; const uint8_t *__restrict__ w_fp8; const uint8_t *__restrict__ w_scales; __nv_bfloat16 *__restrict__ lo_out; const uint32_t groups; GpuTraceRing *__restrict__ trace_ring; const uint32_t layer_id; const uint64_t step_seq",
    ),
    (
        "cert_lm_head_sm121.cu",
        "cert_lm_head_sm121",
        "const __nv_bfloat16 *__restrict__ hidden; const __nv_bfloat16 *__restrict__ w; const __nv_bfloat16 *__restrict__ w_block_max; float *__restrict__ logits; uint8_t *__restrict__ certified; unsigned int *__restrict__ certified_count; float *__restrict__ block_max; uint32_t *__restrict__ block_argmax; const uint32_t vocab; GpuTraceRing *__restrict__ trace_ring; const uint32_t layer_id; const uint64_t step_seq",
    ),
];

/// The kernel name that a source file defines, taken from the same table as the signature pins.
#[cfg(test)]
fn kernel_name_of(file: &str) -> &'static str {
    SIGNATURES
        .iter()
        .find(|(name, _, _)| *name == file)
        .map(|(_, kernel, _)| *kernel)
        .unwrap_or_else(|| panic!("{file} is not a pinned Milestone 4 kernel"))
}

/// Re-parses a kernel declaration and joins its parameters into the pinned form.
#[cfg(test)]
fn parsed_signature(source: &str, kernel: &str) -> String {
    let stripped = crate::c_decl::strip_comments(source);
    let declaration = crate::c_decl::kernel_declaration(&stripped, kernel);
    declaration
        .params
        .iter()
        .map(|param| param.raw_type.as_str())
        .collect::<Vec<_>>()
        .join("; ")
}

#[test]
fn every_milestone_4_signature_is_pinned() {
    for (name, source) in KERNELS {
        let kernel = kernel_name_of(name);
        let expected = SIGNATURES
            .iter()
            .find(|(file, _, _)| *file == name)
            .map(|(_, _, signature)| *signature)
            .unwrap_or_else(|| panic!("{name} has no pinned signature"));
        assert_eq!(
            parsed_signature(source, kernel),
            expected,
            "{name}'s parameter list changed; the Rust side and the CUDA side no longer agree"
        );
    }
}

#[test]
fn the_joined_signature_pin_notices_a_dropped_qualifier() {
    // Mutation proof for the joined pin, on a kernel other than the one the field-wise test uses:
    // dropping __restrict__ from the MoE weight pointer must change the joined text.
    let mutated = MXFP4_MOE.replace("const uint8_t *__restrict__ w_fp4", "const uint8_t *w_fp4");
    assert_ne!(mutated, MXFP4_MOE, "the mutation must change the text");
    let joined = parsed_signature(&mutated, "b12x_mxfp4_moe_sm121");
    let pinned = SIGNATURES
        .iter()
        .find(|(file, _, _)| *file == "b12x_mxfp4_moe_sm121.cu")
        .map(|(_, _, signature)| *signature)
        .expect("pinned");
    assert_ne!(
        joined, pinned,
        "the joined signature must expose the dropped qualifier"
    );
    assert!(joined.contains("const uint8_t *w_fp4;"));
}

#[test]
fn every_kernel_takes_the_trace_ring_and_the_layer_identity() {
    // A kernel that forgets the trace ring or the layer id cannot be instrumented, and the trace
    // ABI is how the M4 phases are measured. This is the cheap version of the scope test below:
    // it applies to every kernel, including any added later.
    for (name, source) in KERNELS {
        let signature = parsed_signature(source, kernel_name_of(name));
        assert!(
            signature.contains("GpuTraceRing *__restrict__ trace_ring"),
            "{name} does not take the trace ring"
        );
        assert!(
            signature.contains("const uint32_t layer_id"),
            "{name} does not take the layer id"
        );
        assert!(
            signature.contains("const uint64_t step_seq"),
            "{name} does not take the step sequence"
        );
    }
}

#[test]
fn the_signature_parser_notices_a_dropped_qualifier() {
    // Mutation proof: drop __restrict__ from one parameter in the parsed text and require the
    // comparison to fail. Without this, a parser that ignored qualifiers would look correct.
    let mutated = MEGA_MHC.replace("uint8_t *__restrict__ xhat_scales", "uint8_t *xhat_scales");
    assert_ne!(
        mutated, MEGA_MHC,
        "the mutation must actually change the text"
    );
    let decl = crate::c_decl::kernel_declaration(
        &crate::c_decl::strip_comments(&mutated),
        "mega_mhc_fused_bf16_sm121",
    );
    assert_eq!(
        decl.params.len(),
        19,
        "the parameter count is unchanged by the mutation"
    );
    // xhat_scales is parameter 10: x, y_prev, b_prev, c_prev, a_prev, predictor, hc_scale,
    // hc_base, rms_weight, xhat_fp8, xhat_scales.
    assert!(
        !decl.params[10].raw_type.contains("__restrict__"),
        "the parser must expose the missing qualifier: {}",
        decl.params[10].raw_type
    );
}

#[test]
fn mega_mhc_geometry_is_self_consistent() {
    let table = consts();
    let hidden = table.value("SPARK_M4_HIDDEN");
    let tile = table.value("SPARK_M4_MHC_TILE_ELEMS");
    let tiles = table.value("SPARK_M4_MHC_TILES");
    let threads = table.value("SPARK_M4_MHC_THREADS");
    let stats = table.value("SPARK_M4_MHC_STATS");
    assert_eq!(tile * tiles, hidden, "the four tiles must cover d exactly");
    assert_eq!(
        tiles, 4,
        "specification 2.4 leaves four streams, swept as four tiles"
    );
    assert_eq!(threads % 32, 0, "a block is a whole number of warps");
    assert_eq!(
        stats, 26,
        "n*n + 2n dot products plus the block RMS and the coefficient RMS statistics"
    );
    assert_eq!(table.value("SPARK_M4_HC_PREDICTOR_ROWS"), 24);
    assert_eq!(
        tile % table.value("SPARK_M4_QUANT_BLOCK"),
        0,
        "tile vs MX block"
    );
}

#[test]
fn the_geometry_parser_re_derives_rather_than_restates() {
    // Mutation proof for the #define evaluator: rewrite SPARK_M4_HC_MULT to 3 and require
    // SPARK_M4_HC_PREDICTOR_ROWS to follow it (3*3 + 2*3 = 15), which only happens if the
    // expression is genuinely evaluated through the identifier table.
    let mutated =
        COMMON_HEADER.replace("#define SPARK_M4_HC_MULT 4u", "#define SPARK_M4_HC_MULT 3u");
    assert_ne!(
        mutated, COMMON_HEADER,
        "the mutation must actually change the text"
    );
    let table = Consts::parse(&mutated);
    assert_eq!(table.value("SPARK_M4_HC_MULT"), 3);
    assert_eq!(table.value("SPARK_M4_HC_PREDICTOR_ROWS"), 15);
    // The static asserts that depended on it must now fail, which is what makes them useful.
    let original = Consts::parse(COMMON_HEADER);
    assert!(original.value("SPARK_M4_HC_PREDICTOR_ROWS") == 24);
}

/// The number of static asserts each source must carry. Pinning the count is what stops a future
/// edit from quietly deleting an assertion that the CI cuda-abi job exists to enforce.
#[cfg(test)]
const ASSERT_COUNTS: [(&str, usize); 7] = [
    ("m4_common_sm121.cuh", 20),
    ("mega_mhc_sm121.cu", 7),
    ("b12x_mxfp8_dense_sm121.cu", 15),
    ("b12x_mxfp4_moe_sm121.cu", 12),
    ("flash_mla_csa2_sm121.cu", 17),
    ("b12x_mxfp8_wo_a_sm121.cu", 20),
    ("cert_lm_head_sm121.cu", 9),
];

/// Why an assertion is not a conformance check, as a function so that the mutation test below runs
/// the real rule instead of a copy of it. Two shapes are rejected:
///
///   * a condition that names no SPARK_ symbol, unless it is one of the pinned toolchain facts:
///     replacing any assertion with static_assert(1, "...") keeps the count and the evaluation, so
///     the symbol is what makes the count mean anything;
///   * a condition whose `==` or `!=` has two operands that are the same after normalisation.
///     That is true whatever the project constant is, it names the project's own symbol, it keeps
///     the count and it evaluates to 1, so no other check in this file can see it:
///     SPARK_M4_HC_MULT == SPARK_M4_HC_MULT was accepted before this rule existed.
///
/// A side is normalised by removing whitespace, unwrapping a fully enclosing pair of balanced
/// parentheses, and peeling a leading unary `+`/`-`, a leading `~`, a C-style cast, one of the
/// four named casts (`static_cast<unsigned>(SPARK_X)`) or a functional cast
/// (`unsigned(SPARK_X)`), so `SPARK_X == (SPARK_X)`, `SPARK_X == +SPARK_X`,
/// `SPARK_X == (uint32_t)SPARK_X`, `SPARK_X == static_cast<unsigned>(SPARK_X)`,
/// `SPARK_X == ~(~SPARK_X)` and `SPARK_X != SPARK_X` are all defects. A `?:` is judged by its
/// condition, so `SPARK_X == SPARK_X ? 1 : 0` is a defect too. The rule stays textual: a tautology
/// built by an operator the normaliser does not model (`SPARK_X + 0u == SPARK_X`,
/// `SPARK_X * 1u == SPARK_X`) is out of its reach, and that limit is pinned by a case in the
/// mutation test rather than left implicit.
#[cfg(test)]
fn static_assert_defect(name: &str, condition: &str) -> Option<String> {
    if let Some(side) = self_comparison(condition) {
        return Some(format!(
            "static_assert({condition}) compares {side} with itself, so it says nothing about the project"
        ));
    }
    let toolchain_fact = NON_SPARK_ASSERTIONS
        .iter()
        .any(|(file, pinned)| *file == name && condition == *pinned);
    if !condition.contains("SPARK_") && !toolchain_fact {
        return Some(format!(
            "static_assert({condition}) references no SPARK_ symbol and is not one of the {} pinned toolchain facts, so it checks nothing about this project",
            NON_SPARK_ASSERTIONS.len()
        ));
    }
    None
}

/// The operand text of each `==`/`!=` comparison in `condition`, normalised, in order.
#[cfg(test)]
fn comparison_operands(condition: &str) -> Vec<String> {
    let mut operands = Vec::new();
    let mut current = String::new();
    let mut chars = condition.chars().peekable();
    while let Some(c) = chars.next() {
        let next = chars.peek().copied();
        let is_comparison = (c == '=' && next == Some('=')) || (c == '!' && next == Some('='));
        if is_comparison {
            chars.next();
            operands.push(normalise_operand(&current));
            current.clear();
            continue;
        }
        current.push(c);
    }
    operands.push(normalise_operand(&current));
    operands
}

/// A comparison side with its textual wrappers peeled: whitespace, fully enclosing parentheses, a
/// leading unary sign, a leading bitwise complement, a C-style or named cast, and a functional
/// cast. The wrappers nest, so this repeats until the text stops shrinking.
#[cfg(test)]
fn normalise_operand(operand: &str) -> String {
    let mut text: String = operand.chars().filter(|c| !c.is_whitespace()).collect();
    loop {
        let before = text.clone();
        text = unwrap_fully_parenthesized(&text);
        text = strip_leading_sign(&text);
        text = strip_leading_bitnot(&text);
        text = strip_leading_cast(&text);
        text = strip_leading_named_cast(&text);
        text = strip_leading_functional_cast(&text);
        if text == before {
            return text;
        }
    }
}

/// Removes one pair of parentheses when it encloses the whole text, e.g. `(SPARK_X)`.
#[cfg(test)]
fn unwrap_fully_parenthesized(text: &str) -> String {
    if !text.starts_with('(') {
        return text.to_string();
    }
    let mut depth = 0i32;
    for (index, c) in text.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return if index + 1 == text.len() {
                        text[1..index].to_string()
                    } else {
                        text.to_string()
                    };
                }
            }
            _ => {}
        }
    }
    text.to_string()
}

/// Removes one leading unary `+` or `-`.
#[cfg(test)]
fn strip_leading_sign(text: &str) -> String {
    match text.strip_prefix('+').or_else(|| text.strip_prefix('-')) {
        Some(rest) => rest.to_string(),
        None => text.to_string(),
    }
}

/// Removes one leading bitwise complement `~`, so `~~SPARK_X` and `~(~SPARK_X)` peel to the
/// symbol.
#[cfg(test)]
fn strip_leading_bitnot(text: &str) -> String {
    match text.strip_prefix('~') {
        Some(rest) => rest.to_string(),
        None => text.to_string(),
    }
}

/// Removes a leading C-style cast to a recognised type, e.g. `(uint32_t)` in `(uint32_t)SPARK_X`.
#[cfg(test)]
fn strip_leading_cast(text: &str) -> String {
    let Some(rest) = text.strip_prefix('(') else {
        return text.to_string();
    };
    let mut depth = 0i32;
    for (offset, c) in rest.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                if depth == 0 {
                    return if is_type_name(&rest[..offset]) {
                        rest[offset + 1..].to_string()
                    } else {
                        text.to_string()
                    };
                }
                depth -= 1;
            }
            _ => {}
        }
    }
    text.to_string()
}

/// The four named casts, which wrap their operand in a second pair of parentheses.
#[cfg(test)]
const NAMED_CASTS: [&str; 4] = [
    "static_cast",
    "dynamic_cast",
    "reinterpret_cast",
    "const_cast",
];

/// Removes a leading named cast, e.g. `static_cast<unsigned>(SPARK_X)`.
///
/// A C++ kernel spells its casts this way rather than `(unsigned)`, and the round-4 A4 shape hid the
/// same tautology behind one. The cast must enclose the whole text: a cast with an arithmetic tail
/// is a different expression and is left alone.
#[cfg(test)]
fn strip_leading_named_cast(text: &str) -> String {
    let Some(rest) = NAMED_CASTS
        .iter()
        .find_map(|keyword| text.strip_prefix(keyword))
    else {
        return text.to_string();
    };
    let Some(rest) = rest.strip_prefix('<') else {
        return text.to_string();
    };
    // Find the closing `>` of the type argument. A nested `<` is counted so a template id in the
    // type cannot end the cast early.
    let mut depth = 1i32;
    let mut type_end = None;
    for (offset, c) in rest.char_indices() {
        match c {
            '<' => depth += 1,
            '>' => {
                depth -= 1;
                if depth == 0 {
                    type_end = Some(offset);
                    break;
                }
            }
            _ => {}
        }
    }
    let Some(type_end) = type_end else {
        return text.to_string();
    };
    let Some(rest) = rest[type_end + 1..].strip_prefix('(') else {
        return text.to_string();
    };
    strip_wrapping_parentheses(text, rest)
}

/// Removes a leading functional cast, e.g. `unsigned(SPARK_X)` or `uint32_t(SPARK_X)`.
///
/// The prefix before the `(` has to read as a type name, which is what tells the cast from a call to
/// a function. The parenthesised expression must enclose the whole text.
#[cfg(test)]
fn strip_leading_functional_cast(text: &str) -> String {
    let Some(open) = text.find('(') else {
        return text.to_string();
    };
    if open == 0 || !is_type_name(&text[..open]) {
        return text.to_string();
    }
    strip_wrapping_parentheses(text, &text[open + 1..])
}

/// If `rest` is exactly a parenthesised expression, returns its contents; otherwise returns
/// `original` unchanged. `rest` is the text after an already-consumed `(`, so the initial depth
/// is one.
#[cfg(test)]
fn strip_wrapping_parentheses(original: &str, rest: &str) -> String {
    let mut depth = 1i32;
    for (offset, c) in rest.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return if offset + 1 == rest.len() {
                        rest[..offset].to_string()
                    } else {
                        original.to_string()
                    };
                }
            }
            _ => {}
        }
    }
    original.to_string()
}

/// Whether `text` reads as a C++ type name, which is what tells a cast `(uint32_t)SPARK_X` from
/// a parenthesised expression `(SPARK_X)`.
#[cfg(test)]
fn is_type_name(text: &str) -> bool {
    if text.is_empty() {
        return false;
    }
    text.split_whitespace().all(|word| {
        matches!(
            word,
            "const"
                | "volatile"
                | "unsigned"
                | "signed"
                | "short"
                | "long"
                | "int"
                | "float"
                | "double"
                | "bool"
                | "char"
                | "void"
                | "size_t"
                | "ptrdiff_t"
                | "uint8_t"
                | "uint16_t"
                | "uint32_t"
                | "uint64_t"
                | "int8_t"
                | "int16_t"
                | "int32_t"
                | "int64_t"
                | "__nv_bfloat16"
                | "__half"
                | "half"
                | "*"
        ) || word.ends_with("_t")
            || word.starts_with("__nv_")
    })
}

/// The condition of the outermost `?:` in `text`, if it has one at bracket depth zero.
///
/// A `?:` whose condition compares a symbol with itself is the same defect as the bare comparison:
/// the whole expression is constant however the symbol resolves, which is the round-4 A4
/// `SPARK_X == SPARK_X ? 1 : 0` shape.
#[cfg(test)]
fn ternary_condition(text: &str) -> Option<&str> {
    let mut depth = 0i32;
    for (index, c) in text.char_indices() {
        match c {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth -= 1,
            '?' if depth == 0 => return Some(&text[..index]),
            _ => {}
        }
    }
    None
}

/// The operand text of an `==` or `!=` whose two sides are the same after normalisation, which
/// makes it trivially true however the symbols resolve. A `?:` is judged by its condition.
#[cfg(test)]
fn self_comparison(condition: &str) -> Option<String> {
    let comparison = ternary_condition(condition).unwrap_or(condition);
    let operands = comparison_operands(comparison);
    operands
        .windows(2)
        .find(|pair| !pair[0].is_empty() && pair[0] == pair[1])
        .map(|pair| pair[0].clone())
}

#[test]
fn every_static_assert_in_the_milestone_holds_under_rust_too() {
    let mut checked = 0usize;
    for (name, source) in std::iter::once(("m4_common_sm121.cuh", COMMON_HEADER)).chain(KERNELS) {
        let stripped = crate::c_decl::strip_comments(source);
        let mut table = Consts::parse(&stripped);
        table.extend(COMMON_HEADER);
        let asserts = crate::c_decl::static_asserts(&stripped);
        let expected = ASSERT_COUNTS
            .iter()
            .find(|(file, _)| *file == name)
            .map(|(_, count)| *count)
            .unwrap_or_else(|| panic!("{name} has no pinned assert count"));
        assert_eq!(
            asserts.len(),
            expected,
            "{name} static_assert count changed"
        );
        for assertion in &asserts {
            let value = table.eval(&assertion.condition);
            assert_eq!(
                value, 1,
                "{name}: static_assert({}, \"{}\") is false in Rust's reading",
                assertion.condition, assertion.message
            );
            // The conformance rule lives in static_assert_defect so the mutation test exercises the
            // real predicate: each assertion has to name a project symbol, and no assertion may
            // compare something with itself.
            if let Some(defect) = static_assert_defect(name, &assertion.condition) {
                panic!("{name}: {defect} (message: {})", assertion.message);
            }
            checked += 1;
        }
    }
    assert_eq!(
        checked, 100,
        "the milestone's static asserts must all be evaluated: 20 in the shared header plus 7, 15, \
         12, 17, 20 and 9 in the six kernels (mega_mhc, dense, MoE, flash_mla, wo_a, cert). The MoE \
         count rose from 10 to 12 with the split-K tiling: the 256-thread block identity and the \
         one-reducer-per-(slot, column) bound. D14 had raised it from 7 to 10 for the FP8 E4M3 \
         activation operand's byte width; flash_mla's rose from 13 to 15 with the engine window \
         format's two layout asserts, then to 17 with the union capacity's two. The format VALUE is \
         not among them: it is a runtime launch parameter and is guarded at kernel entry instead"
    );
    // Every exception must still exist: a stale entry would quietly widen the rule above.
    for (name, condition) in NON_SPARK_ASSERTIONS {
        let source = if name == "m4_common_sm121.cuh" {
            COMMON_HEADER
        } else {
            KERNELS
                .iter()
                .find(|(file, _)| *file == name)
                .map(|(_, source)| *source)
                .unwrap_or_else(|| panic!("no source for {name}"))
        };
        let present = crate::c_decl::static_asserts(source)
            .iter()
            .any(|assertion| assertion.condition == condition);
        assert!(
            present,
            "{name} no longer contains the pinned toolchain assertion {condition:?}, so the \
             exception list must shrink"
        );
    }
}

/// The only assertions allowed to name no SPARK_ symbol: they pin a toolchain fact (the width of a
/// storage type the ABI depends on) rather than a project constant. Listed by exact text so a new
/// one has to be added deliberately, which is what stops the exception from being the hole that a
/// tautology slips through.
#[cfg(test)]
const NON_SPARK_ASSERTIONS: [(&str, &str); 4] = [
    ("m4_common_sm121.cuh", "sizeof(uint8_t) == 1"),
    ("m4_common_sm121.cuh", "sizeof(uint16_t) == 2"),
    ("m4_common_sm121.cuh", "sizeof(float) == 4"),
    ("m4_common_sm121.cuh", "sizeof(double) == 8"),
];

#[test]
fn a_static_assert_that_checks_nothing_is_rejected() {
    // Mutation proof for the SPARK_ requirement above, with the reviewer's own escape: replace a
    // kernel's assertion with a tautology. The count and the evaluation both still pass, so only
    // the symbol requirement can catch it - and it does.
    let source = crate::c_decl::strip_comments(CERT_LM_HEAD);
    let asserts = crate::c_decl::static_asserts(&source);
    let target = asserts
        .first()
        .expect("the certified head carries static asserts");
    assert!(
        target.condition.contains("SPARK_"),
        "the real assertion names a project symbol: {}",
        target.condition
    );
    let mutated = CERT_LM_HEAD.replace(&target.condition, "1");
    assert_ne!(mutated, CERT_LM_HEAD, "the mutation must apply");
    let mutated_source = crate::c_decl::strip_comments(&mutated);
    let mutated_asserts = crate::c_decl::static_asserts(&mutated_source);
    assert_eq!(
        mutated_asserts.len(),
        asserts.len(),
        "the count must not change, so the count cannot catch this"
    );
    let mut table = Consts::parse(&mutated_source);
    table.extend(COMMON_HEADER);
    let hollow = mutated_asserts
        .iter()
        .find(|assertion| assertion.condition.trim() == "1")
        .expect("the tautology must be recovered from the text");
    assert_eq!(table.eval(&hollow.condition), 1, "and it is trivially true");
    assert!(
        !hollow.condition.contains("SPARK_"),
        "which is exactly what the conformance test rejects"
    );
    assert!(
        static_assert_defect("cert_lm_head_sm121.cu", &hollow.condition).is_some(),
        "the real conformance rule must reject the constant"
    );

    // The round-3 escape, run through the same real rule: a self-comparison. It keeps the count,
    // names a SPARK_ symbol and evaluates to 1, so the symbol requirement alone cannot see it.
    let header_source = crate::c_decl::strip_comments(COMMON_HEADER);
    let header_asserts = crate::c_decl::static_asserts(&header_source);
    let mhc = header_asserts
        .iter()
        .find(|assertion| assertion.condition.trim() == "SPARK_M4_HC_MULT == 4u")
        .expect("the header pins the mHC multiplicity");
    assert_eq!(
        static_assert_defect("m4_common_sm121.cuh", &mhc.condition),
        None,
        "the real assertion must pass the rule the mutation is about to break"
    );
    let tautology = mhc.condition.replace("4u", "SPARK_M4_HC_MULT");
    assert_eq!(tautology, "SPARK_M4_HC_MULT == SPARK_M4_HC_MULT");
    let mutated_header = COMMON_HEADER.replace(&mhc.condition, &tautology);
    assert_ne!(mutated_header, COMMON_HEADER, "the mutation must apply");
    let mutated_source = crate::c_decl::strip_comments(&mutated_header);
    let mutated_asserts = crate::c_decl::static_asserts(&mutated_source);
    assert_eq!(
        mutated_asserts.len(),
        header_asserts.len(),
        "the count must not change, so the count cannot catch this"
    );
    let mutated = mutated_asserts
        .iter()
        .find(|assertion| assertion.condition.trim() == tautology)
        .expect("the mutated assertion is still parsed");
    let mut mutated_table = Consts::parse(&mutated_source);
    assert_eq!(
        mutated_table.eval(&mutated.condition),
        1,
        "and it is trivially true, so the evaluator cannot catch it either"
    );
    mutated_table.extend(COMMON_HEADER);
    assert_eq!(mutated_table.value("SPARK_M4_HC_MULT"), 4);
    assert!(
        static_assert_defect("m4_common_sm121.cuh", &mutated.condition).is_some(),
        "the real conformance rule must reject a condition that compares a symbol with itself"
    );

    // The documented limit, asserted rather than left implicit: the rule is textual, so an
    // arithmetic tautology is out of its reach. If this ever becomes caught, the doc comment on
    // static_assert_defect has to change with it.
    let arithmetic = "SPARK_M4_HC_MULT + 0u == SPARK_M4_HC_MULT";
    assert_eq!(self_comparison(arithmetic), None);
    assert_eq!(
        static_assert_defect("m4_common_sm121.cuh", arithmetic),
        None
    );
}

#[test]
fn the_self_comparison_rule_rejects_the_reviewers_wrapped_shapes() {
    // Round-3 A6: the rule was purely textual, so wrapping one side hid the same tautology. Each
    // shape below is true whatever SPARK_X is, keeps the count and evaluates to 1, so the exact
    // predicate has to reject all of them - not just the documented SPARK_X + 0u form.
    for condition in [
        "SPARK_X == SPARK_X",
        "SPARK_X == (SPARK_X)",
        "SPARK_X == +SPARK_X",
        "SPARK_X == (uint32_t)SPARK_X",
        "SPARK_X != SPARK_X",
        "SPARK_X == -(-(SPARK_X))",
        // Round-4 A4: the named cast, its extra-parenthesised form, the bitwise complement and a
        // `?:` whose condition is the comparison.
        "SPARK_X == static_cast<unsigned>(SPARK_X)",
        "SPARK_X == (static_cast<unsigned>(SPARK_X))",
        "SPARK_X == ~(~SPARK_X)",
        "SPARK_X == SPARK_X ? 1 : 0",
    ] {
        assert!(
            self_comparison(condition).is_some(),
            "{condition} compares an operand with itself after normalisation"
        );
        assert!(
            static_assert_defect("k.cu", condition).is_some(),
            "{condition} is not a conformance check"
        );
    }
    // Non-vacuity: distinct operands, a cast on one side and a parenthesised sub-expression are not
    // self-comparisons, so the rule does not reject every condition.
    for condition in [
        "SPARK_X == (uint32_t)SPARK_Y",
        "SPARK_X == (SPARK_Y + 1u)",
        "SPARK_X == +SPARK_Y",
        "SPARK_X == static_cast<unsigned>(SPARK_Y)",
        "SPARK_X == (SPARK_Y ? 1u : 0u)",
        "(SPARK_X) == SPARK_Y",
        "SPARK_X <= SPARK_X",
    ] {
        assert_eq!(
            self_comparison(condition),
            None,
            "{condition} has two distinct operands"
        );
    }
}

#[test]
fn the_round_four_self_comparison_spellings_are_rejected_with_the_stated_reason() {
    // Round-4 A4: each shape is true whatever SPARK_X is, names SPARK_, keeps the count and
    // evaluates to 1, so only this rule can see it. The exact verdict is pinned, not just "some
    // error", so a guard that rejects for a different reason fails here.
    for (condition, side) in [
        ("SPARK_X == static_cast<unsigned>(SPARK_X)", "SPARK_X"),
        ("SPARK_X == (static_cast<unsigned>(SPARK_X))", "SPARK_X"),
        ("SPARK_X == reinterpret_cast<unsigned>(SPARK_X)", "SPARK_X"),
        ("SPARK_X == const_cast<unsigned>(SPARK_X)", "SPARK_X"),
        ("SPARK_X == dynamic_cast<unsigned>(SPARK_X)", "SPARK_X"),
        ("SPARK_X == ~(~SPARK_X)", "SPARK_X"),
        ("SPARK_X == SPARK_X ? 1 : 0", "SPARK_X"),
    ] {
        assert_eq!(
            self_comparison(condition),
            Some(side.to_string()),
            "{condition} must normalise to the same operand on both sides"
        );
        let verdict = static_assert_defect("k.cu", condition)
            .unwrap_or_else(|| panic!("{condition} must be a defect"));
        assert_eq!(
            verdict,
            format!(
                "static_assert({condition}) compares {side} with itself, so it says nothing about the project"
            ),
            "{condition} must be rejected as a self-comparison"
        );
    }
    // The functional-cast spelling the same normaliser learns, and a named cast on one side of a
    // real comparison, which must stay accepted.
    assert_eq!(
        self_comparison("SPARK_X == unsigned(SPARK_X)"),
        Some("SPARK_X".to_string())
    );
    assert_eq!(
        self_comparison("SPARK_X == static_cast<unsigned>(SPARK_Y)"),
        None
    );
    assert_eq!(self_comparison("SPARK_X == unsigned(SPARK_Y)"), None);
}

#[test]
fn the_static_assert_reader_is_not_vacuous() {
    // Mutation proof: break a geometry constant in the text and require the reader to report the
    // corresponding assertion as false. An odd vocabulary breaks both the TP divisibility assert
    // and the per-rank assert, and nothing else in the file is allowed to hide that.
    let mutated = COMMON_HEADER.replace(
        "#define SPARK_M4_VOCAB 129280u",
        "#define SPARK_M4_VOCAB 129279u",
    );
    assert_ne!(mutated, COMMON_HEADER);
    let stripped = crate::c_decl::strip_comments(&mutated);
    let table = Consts::parse(&stripped);
    let asserts = crate::c_decl::static_asserts(&stripped);
    let failures: Vec<&crate::c_decl::StaticAssert> = asserts
        .iter()
        .filter(|assertion| table.eval(&assertion.condition) != 1)
        .collect();
    assert_eq!(
        failures.len(),
        2,
        "exactly the two header vocabulary assertions must fail, found {} of {} asserts",
        failures.len(),
        asserts.len()
    );
    assert!(
        failures
            .iter()
            .all(|assertion| assertion.condition.contains("SPARK_M4_VOCAB")),
        "only the vocabulary assertions may depend on that constant: {failures:?}"
    );

    // The same mutated header is then used to evaluate a kernel's own asserts. The certified head
    // carries a vocabulary shard assertion, and a mutation proof that only ever looked at the
    // header would not notice if that kernel's assert were deleted.
    let lm_source = crate::c_decl::strip_comments(CERT_LM_HEAD);
    let mut kernel_table = Consts::parse(&lm_source);
    kernel_table.extend(&mutated);
    let lm_asserts = crate::c_decl::static_asserts(&lm_source);
    let kernel_failures: Vec<&crate::c_decl::StaticAssert> = lm_asserts
        .iter()
        .filter(|assertion| kernel_table.eval(&assertion.condition) != 1)
        .collect();
    assert_eq!(
        kernel_failures.len(),
        1,
        "the certified head has exactly one assert that depends on the vocabulary, found {}",
        kernel_failures.len()
    );
    assert!(
        kernel_failures[0].condition.contains("SPARK_M4_VOCAB"),
        "the failing kernel assert must be the shard one: {:?}",
        kernel_failures[0].condition
    );
}

#[test]
fn the_scope_assigned_to_each_kernel_is_the_one_the_spec_asks_for() {
    // The trace scope names come from the trace ABI table in src/abi.rs; these are the scopes
    // specification 5.8 assigns to the M4 phases (Mega-mHC, dense MXFP8 including the wo_a twin,
    // MoE MXFP4, CSA2 attention, and the certified head's dense screen).
    let expected = [
        ("mega_mhc_sm121.cu", "TRACE_SCOPE_MEGA_MHC"),
        ("b12x_mxfp8_dense_sm121.cu", "TRACE_SCOPE_DENSE_MXFP8"),
        ("b12x_mxfp4_moe_sm121.cu", "TRACE_SCOPE_MOE_MXFP4"),
        ("flash_mla_csa2_sm121.cu", "TRACE_SCOPE_CSA2_ATTN"),
        // The grouped output projection and the certified head both run under the dense MXFP8
        // scope: they are dense MXFP8 GEMMs, not phases with a scope of their own.
        ("b12x_mxfp8_wo_a_sm121.cu", "TRACE_SCOPE_DENSE_MXFP8"),
        ("cert_lm_head_sm121.cu", "TRACE_SCOPE_DENSE_MXFP8"),
    ];
    for (name, scope) in expected {
        let source = KERNELS
            .iter()
            .find(|(file, _)| *file == name)
            .map(|(_, source)| *source)
            .unwrap_or_else(|| panic!("{name} is not in the kernel list"));
        let enter = format!("SPARK_GPU_TRACE_ENTER(trace_ring, {scope},");
        assert!(
            source.contains(&enter),
            "{name} must open {scope} with {enter}"
        );
        let exit = format!("SPARK_GPU_TRACE_EXIT(trace_ring, {scope},");
        assert!(
            source.contains(&exit),
            "{name} must close {scope} with {exit}"
        );
    }
}

#[test]
fn the_new_kernels_geometry_re_derives_from_the_header() {
    // Everything here is re-derived from the parsed #defines, so the test fails if a kernel
    // restates a header number instead of using it. The values themselves are specification 2.1's.
    let table = consts();
    // MoE: 576 wide per rank, padded to 640 for the shared expert.
    assert_eq!(table.value("SPARK_M4_MOE_WIDTH"), 576);
    assert_eq!(table.value("SPARK_M4_MOE_WIDTH_PADDED"), 640);
    assert_eq!(
        table.value("SPARK_M4_MOE_WIDTH_PADDED") % table.value("SPARK_M4_QUANT_BLOCK"),
        0,
        "the padded width must tile into whole MX blocks"
    );
    assert_eq!(table.value("SPARK_M4_ROUTED_EXPERTS"), 384);
    assert_eq!(table.value("SPARK_M4_MOE_TOP_K"), 6);
    // wo_a: 8 heads per group of 512 dims each is a 4096-wide reduction, 1024 out.
    assert_eq!(
        table.value("SPARK_M4_HEADS") / table.value("SPARK_M4_O_GROUPS")
            * table.value("SPARK_M4_HEAD_DIM"),
        4096
    );
    assert_eq!(table.value("SPARK_M4_O_LORA_RANK"), 1024);
    // MLA: the candidate list is index_topk, the pool is 2048 blocks of 8 positions.
    assert_eq!(table.value("SPARK_M4_INDEX_TOPK"), 512);
    assert_eq!(
        table.value("SPARK_M4_HSI_BLOCKS") * table.value("SPARK_M4_HSI_BLOCK_TOKENS"),
        table.value("SPARK_M4_HSI_POSITIONS")
    );
    assert_eq!(table.value("SPARK_M4_HSI_POSITIONS"), 16384);
    assert_eq!(table.value("SPARK_M4_KV_LORA_RANK"), 512);
    assert_eq!(
        table.value("SPARK_M4_KV_LORA_RANK") % table.value("SPARK_M4_KV_SCALE_CHANNELS"),
        0
    );
    // The certified head: 129,280 logits over 4 ranks is 32,320 each.
    assert_eq!(table.value("SPARK_M4_VOCAB"), 129280);
    assert_eq!(
        table.value("SPARK_M4_VOCAB") / table.value("SPARK_M4_VOCAB_PER_RANK"),
        4
    );
    assert_eq!(table.value("SPARK_M4_LM_SCREEN_TILE"), 128);
    assert_eq!(
        table.value("SPARK_M4_HIDDEN") % table.value("SPARK_M4_LM_SCREEN_TILE"),
        0
    );
    // The dense kernel's tile and the shared reduce width.
    assert_eq!(table.value("SPARK_M4_REDUCE_SLOTS"), 256);
    assert_eq!(table.value("SPARK_M4_DENSE_TILE_M"), 64);
    assert_eq!(table.value("SPARK_M4_DENSE_TILE_N"), 64);
}

#[test]
fn kernels_include_the_common_header_before_using_its_helpers() {
    for (name, source) in KERNELS {
        let stripped = crate::c_decl::strip_comments(source);
        let include = stripped.find("#include \"m4_common_sm121.cuh\"");
        assert!(
            include.is_some(),
            "{name} must include the shared M4 header"
        );
        let include = include.expect("checked");
        for helper in [
            "spark_ue8m0_from_amax",
            "spark_fp8_e4m3_encode",
            "spark_wait_for_flag_acquire",
            "SPARK_GPU_TRACE_ENTER",
        ] {
            let use_site = stripped.find(helper);
            if let Some(use_site) = use_site {
                assert!(
                    use_site > include,
                    "{name} uses {helper} before including the header that declares it"
                );
            }
        }
    }
}

#[cfg(test)]
mod pin_helpers {
    use std::path::{Path, PathBuf};

    /// A source file the pin scans read at test time: a display name and its text.
    pub type ScannedSource = (String, String);

    /// Counts definitions of fn <function>( in source, whatever the visibility or nesting.
    ///
    /// A private or nested module that redeclares a shared harness function is exactly the drift
    /// these pins forbid, so the count includes fn, pub fn, pub(crate) fn and every other
    /// qualifier. Comments are stripped first and the match is line-anchored, so a call site or a
    /// string literal that merely contains the token does not count as a definition.
    pub fn definition_count(source: &str, function: &str) -> usize {
        let code = mask_string_literals(&crate::c_decl::strip_comments(source));
        let compact: String = code.chars().filter(|ch| !ch.is_whitespace()).collect();
        compact.matches(&format!("fn{function}(")).count()
    }

    /// Replaces the contents of Rust string literals with spaces (newlines kept), so a definition
    /// token that only appears inside a message or a fixture string is not counted. Comments are
    /// stripped before this is called.
    pub fn mask_string_literals(source: &str) -> String {
        let mut masked = String::with_capacity(source.len());
        let mut in_string = false;
        let mut escaped = false;
        for ch in source.chars() {
            if in_string {
                if ch == '\n' {
                    masked.push('\n');
                } else {
                    masked.push(' ');
                }
                if escaped {
                    escaped = false;
                } else if ch == '\\' {
                    escaped = true;
                } else if ch == '"' {
                    in_string = false;
                }
            } else if ch == '"' {
                in_string = true;
                masked.push(' ');
            } else {
                masked.push(ch);
            }
        }
        masked
    }

    /// All Rust sources under dir, recursively, sorted for deterministic messages.
    pub fn rust_sources_under(dir: &Path) -> Vec<ScannedSource> {
        fn visit(dir: &Path, found: &mut Vec<ScannedSource>) {
            let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)
                .expect("a source directory must be readable")
                .map(|entry| entry.expect("a directory entry must be readable").path())
                .collect();
            entries.sort();
            for path in entries {
                if path.is_dir() {
                    visit(&path, found);
                } else {
                    let bytes = std::fs::read(&path).expect("a scanned file must be readable");
                    let text = String::from_utf8_lossy(&bytes).into_owned();
                    found.push((path.to_string_lossy().into_owned(), text));
                }
            }
        }
        let mut found = Vec::new();
        visit(dir, &mut found);
        found
    }

    /// The crate source tree, src/**/*.rs.
    pub fn crate_sources() -> Vec<ScannedSource> {
        rust_sources_under(&Path::new(env!("CARGO_MANIFEST_DIR")).join("src"))
    }

    /// The example source tree, examples/**/*.rs.
    pub fn example_sources() -> Vec<ScannedSource> {
        rust_sources_under(&Path::new(env!("CARGO_MANIFEST_DIR")).join("examples"))
    }

    /// Exactly one definition of function across the scanned sources, and it must be
    /// expected_path. Paths are canonicalised and compared exactly: a differently named file
    /// beside the harness (src/not_cuda_harness.rs) cannot stand in, and a duplicate anywhere
    /// fails the count.
    pub fn require_single_shared_definition(
        sources: &[ScannedSource],
        function: &str,
        expected_path: &Path,
    ) -> Result<(), String> {
        let expected = std::fs::canonicalize(expected_path)
            .map_err(|error| format!("{}: {error}", expected_path.display()))?;
        let mut found: Vec<(&str, usize)> = Vec::new();
        for (name, source) in sources {
            let count = definition_count(source, function);
            if count != 0 {
                found.push((name, count));
            }
        }
        let total: usize = found.iter().map(|(_, count)| *count).sum();
        if total != 1 {
            return Err(format!(
                "expected exactly one fn {function}( definition in the crate source, found {found:?}"
            ));
        }
        let actual = std::fs::canonicalize(found[0].0)
            .map_err(|error| format!("{}: {error}", found[0].0))?;
        if actual != expected {
            return Err(format!(
                "the one fn {function}( definition must live in {}, found {}",
                expected.display(),
                found[0].0
            ));
        }
        Ok(())
    }

    /// No source re-exports function under its own name.
    pub fn require_no_alias(sources: &[ScannedSource], function: &str) -> Result<(), String> {
        for (name, source) in sources {
            if has_alias(source, function) {
                return Err(format!(
                    "{name} re-exports {function} under the same name; there is one definition and no alias"
                ));
            }
        }
        Ok(())
    }

    /// True when source re-exports function under its own name (use m::x as x).
    pub fn has_alias(source: &str, function: &str) -> bool {
        let code = mask_string_literals(&crate::c_decl::strip_comments(source));
        let compact: String = code.chars().filter(|ch| !ch.is_whitespace()).collect();
        compact.contains(&format!("as{function}"))
    }

    /// The file name alone, for pin messages.
    pub fn display_name(path: &str) -> String {
        path.rsplit('/').next().unwrap_or(path).to_string()
    }

    /// True for the unsafe keyword outside identifiers, so #![forbid(unsafe_code)] does not count.
    pub fn has_unsafe_keyword(code: &str) -> bool {
        let bytes = code.as_bytes();
        let is_ident = |byte: u8| byte.is_ascii_alphanumeric() || byte == b'_';
        for (at, _) in code.match_indices("unsafe") {
            let before = at.checked_sub(1).map_or(b' ', |index| bytes[index]);
            let after = bytes.get(at + "unsafe".len()).copied().unwrap_or(b' ');
            if !is_ident(before) && !is_ident(after) {
                return true;
            }
        }
        false
    }

    /// Examples whose code imports the shared CUDA driver, as their full source paths so a pin can
    /// name the exact file - after the m7_layer0 split the importing file is
    /// `examples/m7_layer0/main.rs`, and a basename-only witness would accept any other `main.rs`.
    /// String literals are masked first, so a message that merely quotes the import path is not a
    /// user.
    pub fn example_driver_users(sources: &[ScannedSource]) -> Vec<String> {
        sources
            .iter()
            .filter(|(_, source)| {
                mask_string_literals(&crate::c_decl::strip_comments(source))
                    .contains("sparkflash_kernels::cuda_driver")
            })
            .map(|(name, _)| name.clone())
            .collect()
    }

    /// The defense-in-depth rule for an example that uses the shared CUDA driver: no unsafe code
    /// at all and no raw driver token. The compiler already closes the raw launch (the driver
    /// launcher only accepts opaque Kernel values and flash_mla is an opaque FlashMlaKernel); this
    /// pin keeps a future example from opening a second FFI path beside the shared one.
    pub fn check_driver_example(source: &str) -> Result<(), String> {
        let code = mask_string_literals(&crate::c_decl::strip_comments(source));
        if !code.contains("sparkflash_kernels::cuda_driver") {
            return Ok(());
        }
        if !code.contains("#![forbid(unsafe_code)]") {
            return Err(
                "an example that uses the shared CUDA driver must start with #![forbid(unsafe_code)]"
                    .to_string(),
            );
        }
        if has_unsafe_keyword(&code) {
            return Err(
                "an example that uses the shared CUDA driver must not contain the unsafe keyword"
                    .to_string(),
            );
        }
        for token in [
            "mla_func",
            "cuLaunchKernel",
            "cuModuleGetFunction",
            "extern",
        ] {
            if code.contains(token) {
                return Err(format!(
                    "an example that uses the shared CUDA driver must not name the raw driver token {token}"
                ));
            }
        }
        Ok(())
    }

    /// Definitions of function across every example source.
    pub fn example_definition_count(sources: &[ScannedSource], function: &str) -> usize {
        sources
            .iter()
            .map(|(_, source)| definition_count(source, function))
            .sum()
    }
}

/// The split m7_layer0 example, concatenated for the text pins below: `main.rs` plus one module
/// per stage. The split moved statements between files without changing them, so a pin that used to
/// scan one path scans the concatenation; every module file ends with a newline, so no statement is
/// joined across a file boundary.
#[cfg(test)]
const M7_LAYER0_SOURCE: &str = concat!(
    include_str!("../examples/m7_layer0/main.rs"),
    include_str!("../examples/m7_layer0/stage0.rs"),
    include_str!("../examples/m7_layer0/stage1.rs"),
    include_str!("../examples/m7_layer0/stage2.rs"),
    include_str!("../examples/m7_layer0/stage3.rs"),
    include_str!("../examples/m7_layer0/stage4.rs"),
    include_str!("../examples/m7_layer0/stage5.rs"),
    include_str!("../examples/m7_layer0/stage6.rs"),
    include_str!("../examples/m7_layer0/stage7.rs"),
    include_str!("../examples/m7_layer0/stage8.rs"),
    include_str!("../examples/m7_layer0/stage9.rs"),
);

/// Every flash_mla launch site goes through the typed parameter builder and the library's typed
/// launches, so no site can keep a stale arity or a hand-built argument order. The builder and the
/// launches live in sparkflash_kernels::cuda_harness and the example only aliases them; a raw
/// launch is not expressible in the example, because cuda_driver::Cuda::launch takes an opaque
/// Kernel and flash_mla is only ever an opaque FlashMlaKernel. Mutation: the stage-2 main site is
/// put back on a 15-element array and the pin must reject it.
#[test]
fn every_flash_mla_launch_site_uses_the_typed_builder_and_typed_launches() {
    let pin = |source: &str| -> bool {
        let code = crate::c_decl::strip_comments(source);
        // Four constructor calls: stage-2 indexer self-check, stage-2 attention loop, stage-7 K5
        // and stage-9 K5.
        let builders = code.matches("= flash_mla_params(").count();
        // Six typed launches: stage-2 indexer, stage-2 main, the two injected launches, stage-7 K5
        // and stage-9 K5; one timed launch in the stage-2 timing loop.
        let launches = code.matches("flash_mla_launch(").count();
        let timed = code.matches("flash_mla_launch_repeated(").count();
        builders == 4
            && launches == 6
            && timed == 1
            && code.contains("launch_flash_mla as flash_mla_launch")
            && !code.contains("[u64; 15]")
            && !code.contains("cuda.launch(mla_func")
            && !code.contains("cuda.launch(dev.mla_func")
    };
    let source = M7_LAYER0_SOURCE;
    assert!(
        crate::c_decl::strip_comments(source).contains("use sparkflash_kernels::cuda_harness::"),
        "the example must import the one flash_mla parameter builder from the library harness module"
    );
    assert!(
        pin(source),
        "a flash_mla launch site bypasses the typed builder/launch wrapper (expected 4 builder calls, 6 typed launches and 1 timed launch)"
    );
    let stale = source.replacen(
        "let params = flash_mla_params(",
        "let params: [u64; 15] = [0; 15];",
        1,
    );
    assert_ne!(stale, source, "the mutation must apply");
    assert!(
        !pin(&stale),
        "a hand-built 15-parameter array is the stale-ABI defect this pin exists for"
    );
}

/// Examples that use the shared CUDA driver are unsafe-free and never name a raw driver token.
/// This is defense in depth next to the type-level enforcement: the driver's launch entry points
/// only accept opaque handles (cuda_driver::Kernel and cuda_harness::FlashMlaKernel), and this pin
/// keeps a future example from opening its own FFI path beside the shared one.
#[test]
fn examples_using_the_shared_cuda_driver_are_unsafe_free_and_name_no_raw_driver_token() {
    let examples = pin_helpers::example_sources();
    let users = pin_helpers::example_driver_users(&examples);
    assert!(
        users
            .iter()
            .any(|name| name.ends_with("examples/m7_layer0/main.rs")),
        "the split m7_layer0 example's main.rs must be one of the pinned driver examples, else the pin is vacuous; found {users:?}"
    );
    for (name, source) in &examples {
        pin_helpers::check_driver_example(source)
            .unwrap_or_else(|error| panic!("{}: {error}", pin_helpers::display_name(name)));
    }
    // The split moved the driver import into main.rs but left the code it guards in the per-stage
    // modules. The four-token check above only sees the file containing the import, so run it once
    // over the concatenation the other m7 pins already scan: a raw driver token in a stage module
    // must not become invisible just because the import lives one file over.
    pin_helpers::check_driver_example(M7_LAYER0_SOURCE)
        .unwrap_or_else(|error| panic!("examples/m7_layer0/: {error}"));
    let spoof = r#"const SPOOF: &str = "use sparkflash_kernels::cuda_driver::Cuda;";\nfn f() {}"#;
    assert!(
        !pin_helpers::example_driver_users(&[("spoof.rs".to_string(), spoof.to_string())])
            .iter()
            .any(|name| name == "spoof.rs"),
        "a string literal must not make an example a driver user"
    );
    assert!(
        pin_helpers::check_driver_example(
            "use sparkflash_kernels::cuda_driver::Cuda;\nfn f() {}\n"
        )
        .is_err(),
        "a real driver import without #![forbid(unsafe_code)] must fail"
    );
    assert!(
        pin_helpers::check_driver_example(
            "#![forbid(unsafe_code)]\nuse sparkflash_kernels::cuda_driver::Cuda;\nfn f() { let _ = mla_func; }\n"
        )
        .is_err(),
        "a raw handle name must fail the driver example pin"
    );
    assert!(
        pin_helpers::check_driver_example(
            "#![forbid(unsafe_code)]\nuse sparkflash_kernels::cuda_driver::Cuda;\nfn f() { unsafe {} }\n"
        )
        .is_err(),
        "an unsafe block must fail the driver example pin"
    );
}

/// Stage 2 and stage 9 must build their engine-format window rows with the one shared helper, in
/// the pinned order: rms_norm + rope tail (caller, f32) -> bf16 storage round exactly once ->
/// engine quantiser -> assert against m4_quant::act_quant_fp8_row. The helper lives in the library
/// harness module, sparkflash_kernels::cuda_harness, so the m7 example and the forthcoming m8
/// sequence example compile one implementation instead of a copy each. Definitions are counted at
/// every visibility, module and whitespace variant in every file under src, and the one definition
/// must be the exact path src/cuda_harness.rs. Stage 9's K5 reference must be format-dispatched to
/// mla_ref::attention_with_sink_window in engine mode (feeding window bytes to the FP4 reference is
/// the defect this pins).
#[test]
fn stage2_and_stage9_share_one_library_engine_window_row_builder() {
    let crate_sources = pin_helpers::crate_sources();
    let harness_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/cuda_harness.rs");
    pin_helpers::require_single_shared_definition(
        &crate_sources,
        "engine_window_row",
        &harness_path,
    )
    .unwrap_or_else(|error| panic!("{error}"));
    pin_helpers::require_no_alias(&crate_sources, "engine_window_row")
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(
        pin_helpers::example_definition_count(&pin_helpers::example_sources(), "engine_window_row"),
        0,
        "no example may define its own engine-window row helper"
    );
    let example = crate::c_decl::strip_comments(M7_LAYER0_SOURCE);
    assert!(
        example.contains("use sparkflash_kernels::cuda_harness::"),
        "the example must import the shared engine-row helper from the library harness"
    );
    assert_eq!(
        example.matches("engine_window_row(&").count(),
        2,
        "stage 2 and stage 9 must both call the one shared engine-row helper; a re-implementation has drifted"
    );
    let harness =
        std::fs::read_to_string(&harness_path).expect("the harness source must be readable");
    let start = harness
        .find("fn engine_window_row(")
        .expect("the shared engine-row builder must exist in the library harness");
    let body = &harness[start..];
    let body_end = body
        .find("\n}\n")
        .expect("the shared engine-row helper has a body terminator");
    let body = &body[..body_end];
    let round_at = body
        .find("half::bf16::from_bits(bf16_bits(")
        .expect("the helper must apply the bf16 storage round");
    let quantise_at = body
        .find("sparkflash_engine::quant::quantize_activation_fp8")
        .expect("the helper must use the engine's released quantiser");
    let oracle_at = body
        .find("act_quant_fp8_row(")
        .expect("the helper must assert the m4_quant oracle");
    assert!(
        round_at < quantise_at && quantise_at < oracle_at,
        "the shared helper order is pinned: bf16 storage round, then engine quantiser, then the m4_quant oracle"
    );
    assert_eq!(
        body.matches("half::bf16::from_bits(bf16_bits(").count(),
        1,
        "the bf16 round must happen exactly once, at the engine's storage dtype boundary"
    );
    // Stage 9's engine K5 reference dispatch: engine -> attention_with_sink_window, FP4 -> the FP4
    // reference. Both names must be in the attention block and the dispatch must use the format
    // value, not a constant.
    let block_start = example
        .find("fn stage9_attention_block(")
        .expect("stage 9's attention block must exist");
    let block_end = example[block_start..]
        .find("fn stage9_ffn_block(")
        .expect("the attention block is followed by the FFN block");
    let block = &example[block_start..block_start + block_end];
    assert!(
        block.contains("attention_with_sink_window("),
        "stage 9 engine mode must check K5 against mla_ref::attention_with_sink_window"
    );
    assert!(
        block.contains("attention_with_sink("),
        "stage 9 fp4 mode must keep the FP4 reference"
    );
    assert!(
        block.contains("if format == WindowKvFormat::Engine"),
        "stage 9's K5 reference must be format-dispatched, not hardcoded"
    );
    assert!(
        example.contains("--stage9-kv-format"),
        "stage 9 must expose --stage9-kv-format"
    );
}

/// The flash_mla parameter builder has one definition, in the exact path src/cuda_harness.rs, no
/// alias, and no example reimplements it: a second copy is the drift the builder exists to prevent.
#[test]
fn the_flash_mla_builder_has_one_unaliased_definition_in_the_library_harness() {
    let crate_sources = pin_helpers::crate_sources();
    let harness_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/cuda_harness.rs");
    pin_helpers::require_single_shared_definition(
        &crate_sources,
        "flash_mla_params",
        &harness_path,
    )
    .unwrap_or_else(|error| panic!("{error}"));
    pin_helpers::require_no_alias(&crate_sources, "flash_mla_params")
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(
        pin_helpers::example_definition_count(&pin_helpers::example_sources(), "flash_mla_params"),
        0,
        "no example may define its own flash_mla parameter builder"
    );
}

/// The definition pins against the reviewed evasions: a differently named file beside the harness,
/// an inline or space-before-paren duplicate in a .inc file, and a re-export under the same name.
/// Every case is written to a temp tree and scanned with the same helpers the real pins use.
#[test]
fn the_definition_pins_reject_the_reviewed_evasions() {
    let base =
        std::env::temp_dir().join(format!("sparkflash-pin-definitions-{}", std::process::id()));
    let pristine = base.join("pristine/src");
    std::fs::create_dir_all(&pristine).expect("the temp source dir must be creatable");
    let harness = pristine.join("cuda_harness.rs");
    std::fs::write(
        &harness,
        "pub fn engine_window_row(latent: &[f32]) {}\npub fn flash_mla_params() {}\n",
    )
    .expect("the temp pristine harness must be writable");
    let sources = pin_helpers::rust_sources_under(&pristine);
    pin_helpers::require_single_shared_definition(&sources, "engine_window_row", &harness)
        .expect("one definition in the harness file is the pristine case");
    pin_helpers::require_single_shared_definition(&sources, "flash_mla_params", &harness)
        .expect("one builder definition in the harness file is the pristine case");

    // (c) a differently named file beside the harness cannot stand in for the harness path, and a
    // duplicate in it fails the count.
    let suffixed = base.join("suffixed/src");
    std::fs::create_dir_all(&suffixed).expect("the temp source dir must be creatable");
    std::fs::write(
        suffixed.join("not_cuda_harness.rs"),
        "fn engine_window_row() {}\n",
    )
    .expect("the temp suffixed source must be writable");
    let suffixed_sources = pin_helpers::rust_sources_under(&suffixed);
    assert!(
        pin_helpers::require_single_shared_definition(
            &suffixed_sources,
            "engine_window_row",
            &suffixed.join("cuda_harness.rs"),
        )
        .is_err(),
        "a definition in src/not_cuda_harness.rs must not satisfy the exact harness path"
    );
    std::fs::write(
        pristine.join("not_cuda_harness.rs"),
        "fn engine_window_row() {}\n",
    )
    .expect("the temp duplicate source must be writable");
    let duplicated = pin_helpers::rust_sources_under(&pristine);
    assert!(
        pin_helpers::require_single_shared_definition(&duplicated, "engine_window_row", &harness)
            .is_err(),
        "a second definition in a differently named file must fail the count"
    );
    std::fs::remove_file(pristine.join("not_cuda_harness.rs")).expect("cleanup");

    // (b)+(d) an inline or space-before-paren duplicate in a .inc file is still counted, because
    // every file and every whitespace variant is scanned.
    let inc = base.join("inc/src");
    std::fs::create_dir_all(&inc).expect("the temp source dir must be creatable");
    let inc_harness = inc.join("cuda_harness.rs");
    std::fs::write(
        &inc_harness,
        "pub fn engine_window_row(latent: &[f32]) {}\n",
    )
    .expect("the temp harness must be writable");
    std::fs::write(
        inc.join("private_copy.inc"),
        "mod hidden {\n    fn engine_window_row (latent: &[f32]) {}\n}\n",
    )
    .expect("the temp inc source must be writable");
    let inc_sources = pin_helpers::rust_sources_under(&inc);
    assert!(
        pin_helpers::require_single_shared_definition(
            &inc_sources,
            "engine_window_row",
            &inc_harness
        )
        .is_err(),
        "the .inc duplicate with a space before the parameter list must be counted"
    );
    assert_eq!(
        pin_helpers::definition_count(
            "mod hidden {\n    fn engine_window_row () {}\n}",
            "engine_window_row"
        ),
        1,
        "the whitespace-insensitive token must see a space-before-paren definition"
    );

    // (d) a re-export under the same name is rejected even where the definition count is one.
    assert!(
        pin_helpers::has_alias(
            "pub use private_copy::engine_window_row as engine_window_row;\n",
            "engine_window_row"
        ),
        "an as-rename re-export of the shared helper must be detected"
    );
    assert!(
        pin_helpers::require_no_alias(
            &[(
                "src/cuda_harness.rs".to_string(),
                "pub fn engine_window_row(latent: &[f32]) {}\npub use private_copy::engine_window_row as engine_window_row;\n".to_string(),
            )],
            "engine_window_row",
        )
        .is_err(),
        "a re-export under the same name must fail the alias pin"
    );

    let _ = std::fs::remove_dir_all(&base);
}

/// Checks that a kernel's observing path executes an acquire fence unconditionally.
///
#[test]
fn the_reviewed_invariants_are_pinned_in_the_kernel_sources() {
    // The round-1 review found five defects that no test could see. Each of them is now a property
    // of the source text, and each property is paired with the mutation that removes it, so a
    // future edit that reintroduces the defect fails here as well as in the reference tests.

    // A1: the wo_a group strides. A weight buffer and a scale grid have different sizes, and the
    // bug was using the weight stride for both.
    let wo_a_has_two_strides = |source: &str| -> bool {
        let code = crate::c_decl::strip_comments(source);
        code.contains("const uint64_t group_weight_base =")
            && code.contains("const uint64_t group_scale_base =")
            && code.contains("(uint64_t)group * spark_b12x_scale_bytes(SPARK_M4_O_LORA_RANK")
            && code.contains("w_fp8[group_weight_base +")
            && code.contains("w_scales[group_scale_base +")
    };
    assert!(wo_a_has_two_strides(MXFP8_WO_A));
    let shared_stride = MXFP8_WO_A.replace(
        "(uint64_t)group * spark_b12x_scale_bytes(SPARK_M4_O_LORA_RANK",
        "(uint64_t)group * spark_b12x_buffer_bytes(SPARK_M4_O_LORA_RANK",
    );
    assert_ne!(shared_stride, MXFP8_WO_A, "the mutation must apply");
    assert!(
        !wo_a_has_two_strides(&shared_stride),
        "the weight stride must not be reused for the scales"
    );

    // A2: the MLA score scale is applied exactly once in each attention path: the historical
    // shared loop and the union mode's window and compressed score sites. The union branch adds two
    // sites, so the count is pinned rather than left as a lower bound.
    let mla_scales_the_score = |source: &str| -> bool {
        let code = crate::c_decl::strip_comments(source);
        code.contains("score *= SPARK_M4_MLA_SCORE_SCALE;")
            && code.matches("score *= SPARK_M4_MLA_SCORE_SCALE;").count() == 3
            && code.matches("SPARK_M4_MLA_SCORE_SCALE").count() == 3
    };
    assert!(mla_scales_the_score(FLASH_MLA_CSA2));
    let unscaled = FLASH_MLA_CSA2.replace("score *= SPARK_M4_MLA_SCORE_SCALE;", "score *= 1.0f;");
    assert_ne!(unscaled, FLASH_MLA_CSA2, "the mutation must apply");
    assert!(
        !mla_scales_the_score(&unscaled),
        "an unscaled score is the defect the reference used to bless"
    );

    // A3: the maximum is a per-thread value reduced as a tree, not a shared read-modify-write.
    let mla_reduces_the_max = |source: &str| -> bool {
        let code = crate::c_decl::strip_comments(source);
        code.contains("local_max = spark_block_max(local_max, lane, SPARK_M4_MLA_THREADS)")
            && code.contains("if (lane == 0u) {\n        s_max = local_max;")
            && !code.contains("s_max = s_scores[slot] > s_max")
    };
    assert!(mla_reduces_the_max(FLASH_MLA_CSA2));
    let racy_max = FLASH_MLA_CSA2.replace(
        "    local_max = spark_block_max(local_max, lane, SPARK_M4_MLA_THREADS);\n    if (lane == 0u) {\n        s_max = local_max;\n    }",
        "    s_max = s_scores[slot] > s_max ? s_scores[slot] : s_max;",
    );
    assert_ne!(racy_max, FLASH_MLA_CSA2, "the mutation must apply");
    assert!(
        !mla_reduces_the_max(&racy_max),
        "the racy shared accumulation must be rejected"
    );

    // A5: the indexer is addressed by index head, not by query head.
    let mla_uses_the_index_head = |source: &str| -> bool {
        let code = crate::c_decl::strip_comments(source);
        code.contains(
            "const uint32_t index_head = head / (uint32_t)SPARK_M4_MLA_HEADS_PER_INDEX_HEAD;",
        ) && code.contains("index_query + (uint64_t)index_head * SPARK_M4_INDEX_HEAD_DIM")
    };
    assert!(mla_uses_the_index_head(FLASH_MLA_CSA2));
    let query_head = FLASH_MLA_CSA2.replace(
        "const uint32_t index_head = head / (uint32_t)SPARK_M4_MLA_HEADS_PER_INDEX_HEAD;",
        "const uint32_t index_head = head;",
    );
    assert_ne!(query_head, FLASH_MLA_CSA2, "the mutation must apply");
    assert!(
        !mla_uses_the_index_head(&query_head),
        "the query head must not index the indexer query buffer"
    );

    // A6: the normed X_hat row is actually written. The write is not on the quantiser's own path
    // (that recomputes the product), so deleting it left the whole suite green even though the
    // exported row - mega_mhc_sm121.cu:329's xhat_normed - was empty. The reference's own row
    // comparison cannot see a missing device write that no test reads, so the source text pins it.
    let mhc_writes_the_normed_row = |source: &str| -> bool {
        let code = crate::c_decl::strip_comments(source);
        code.contains("xhat_normed[index + k] = value;")
    };
    assert!(mhc_writes_the_normed_row(MEGA_MHC));
    let dropped_normed_write = MEGA_MHC.replace("xhat_normed[index + k] = value;", "");
    assert_ne!(dropped_normed_write, MEGA_MHC, "the mutation must apply");
    assert!(
        !mhc_writes_the_normed_row(&dropped_normed_write),
        "the normed X_hat write is the kernel's exported row, not dead code"
    );

    // A7: the MoE masks the padded region instead of trusting the caller to have zeroed it.
    let moe_masks_the_padding = |source: &str| -> bool {
        let code = crate::c_decl::strip_comments(source);
        // The trailing newline matters: the padded constant starts with the real one, so a bare
        // prefix check would accept the unmasked mutant.
        code.contains("const float act_value = index < (uint32_t)SPARK_M4_MOE_WIDTH\n")
            && code.contains("? spark_fp8_e4m3_decode(act_row[index])")
    };
    assert!(moe_masks_the_padding(MXFP4_MOE));
    let unmasked = MXFP4_MOE.replace(
        "index < (uint32_t)SPARK_M4_MOE_WIDTH",
        "index < (uint32_t)SPARK_M4_MOE_WIDTH_PADDED",
    );
    assert_ne!(unmasked, MXFP4_MOE, "the mutation must apply");
    assert!(
        !moe_masks_the_padding(&unmasked),
        "walking the padded width unmasked is the defect"
    );

    // D14: the activation operand is FP8 E4M3 with per-32 ue8m0 scales, and the MMA is the
    // reference's FP8xFP8 path (kernel.py:486-499). The packed E2M1 operand this kernel read before
    // D14 must not be able to come back quietly: the byte width is a define that a static_assert
    // reads, the activation is decoded as E4M3, and the routing weight is not folded into the GEMM.
    let moe_activation_is_e4m3 = |source: &str| -> bool {
        let code = crate::c_decl::strip_comments(source);
        code.contains("#define SPARK_M4_MOE_ACT_BYTES_PER_VALUE 1u")
            && code.contains("static_assert(SPARK_M4_MOE_ACT_BYTES_PER_VALUE == 1u,")
            && code.contains(
                "static_assert(SPARK_M4_MOE_ACT_BYTES_PER_ROW == SPARK_M4_MOE_WIDTH_PADDED,",
            )
            && code.contains("const uint8_t *__restrict__ act_fp8")
            && code.contains("? spark_fp8_e4m3_decode(act_row[index])")
            && code.contains("terms[slot][block][col_local] =")
            && code.contains("__fmul_rn(__fmul_rn(partial, act_scale), w_scale);")
            && !code.contains("spark_fp4_load(act_row")
            && !code.contains("expert_weights")
            && !code.contains("routing *")
    };
    assert!(moe_activation_is_e4m3(MXFP4_MOE));
    let packed_activation = MXFP4_MOE.replace(
        "#define SPARK_M4_MOE_ACT_BYTES_PER_VALUE 1u",
        "#define SPARK_M4_MOE_ACT_BYTES_PER_VALUE SPARK_M4_FP4_PER_BYTE",
    );
    assert_ne!(packed_activation, MXFP4_MOE, "the mutation must apply");
    assert!(
        !moe_activation_is_e4m3(&packed_activation),
        "the byte-width pin and the static_assert that reads it are what reject a packed operand"
    );
    let e2m1_operand = MXFP4_MOE.replace(
        "? spark_fp8_e4m3_decode(act_row[index])",
        "? spark_fp4_load(act_row, index)",
    );
    assert_ne!(e2m1_operand, MXFP4_MOE, "the mutation must apply");
    assert!(
        !moe_activation_is_e4m3(&e2m1_operand),
        "reading the activation as packed E2M1 is D14's defect"
    );
    let routing_folded = MXFP4_MOE.replace(
        "__fmul_rn(__fmul_rn(partial, act_scale), w_scale);",
        "__fmul_rn(__fmul_rn(routing * partial, act_scale), w_scale);",
    );
    assert_ne!(routing_folded, MXFP4_MOE, "the mutation must apply");
    assert!(
        !moe_activation_is_e4m3(&routing_folded),
        "folding the routing weight into the GEMM would quantise an unrouted tensor \
         (model.py:849-851), which is a different arithmetic"
    );

    // A8: the LM certificate breaks ties by the lower entry index, in both reductions.
    let lm_breaks_ties_by_index = |source: &str| -> bool {
        let code = crate::c_decl::strip_comments(source);
        code.contains("acc > running_max || (acc == running_max && entry < running_argmax)")
            && code.contains("s_thread_max[i] == best && s_thread_argmax[i] < best_index")
            && code.contains("#define SPARK_M4_LM_TIE_BREAK_BY_INDEX 1u")
    };
    assert!(lm_breaks_ties_by_index(CERT_LM_HEAD));
    let first_wins = CERT_LM_HEAD
        .replace("entry < running_argmax", "entry <= running_argmax")
        .replace(
            "s_thread_argmax[i] < best_index",
            "s_thread_argmax[i] > best_index",
        );
    assert_ne!(first_wins, CERT_LM_HEAD, "the mutation must apply");
    assert!(
        !lm_breaks_ties_by_index(&first_wins),
        "a later entry winning a tie is the defect"
    );

    // A10: the engine window-KV format is selected by an explicit runtime argument, guarded at
    // kernel entry, and never inferred from another argument's value. A format VALUE cannot be a
    // static_assert - static_asserts see compile-time constants, not launch parameters - so the
    // guard is runtime code and this pin is the source text that proves it exists. Format 2 (CSA
    // union) is a third runtime mode, not a widening of the window: this same helper also pins the
    // union dispatch, its capacity expression and the formats 0/1 predicate text the union branch
    // must not disturb. Each mutation below asserts it applied and that this helper then fails.
    let mla_guards_the_window_format = |source: &str| -> bool {
        let code = crate::c_decl::strip_comments(source);
        code.contains("#define SPARK_M4_MLA_WINDOW_KV_FORMAT_FP4 0u")
            && code.contains("#define SPARK_M4_MLA_WINDOW_KV_FORMAT_ENGINE_FP8 1u")
            && code.contains("#define SPARK_M4_MLA_WINDOW_KV_FORMAT_CSA_UNION 2u")
            && code.contains("if (window_kv_format > SPARK_M4_MLA_WINDOW_KV_FORMAT_CSA_UNION) {")
            && code.contains("__float2bfloat16_rn(__uint_as_float(0x7FC00000u))")
            && !code.contains("static_assert(window_kv_format")
            // The formats 0/1 dispatch text the union branch does not enter: the attended
            // predicate and the two ternary load sites.
            && code.contains("? spark_mla_in_window(position, candidate)")
            && code.contains(": (candidate < window_rows ? 1u : 0u);")
            && code.contains(
                "const float key = window_kv_format == SPARK_M4_MLA_WINDOW_KV_FORMAT_FP4",
            )
            && code.contains(
                "const float value = window_kv_format == SPARK_M4_MLA_WINDOW_KV_FORMAT_FP4",
            )
            // The union mode's own dispatch: the absolute-position causal predicate, every supplied
            // window row attended, and the compile-time capacity shared with the Rust twin. The
            // three capacity arms together refuse 641 while accepting exactly 640 (128 + 512).
            && code.contains(
                "#define SPARK_M4_MLA_MAX_ATTENDED (SPARK_M4_MLA_MAX_SELECTED + SPARK_M4_SWA_WINDOW)",
            )
            && code.contains("static_assert(SPARK_M4_MLA_MAX_ATTENDED == 640u,")
            && code.contains("__shared__ float s_scores[SPARK_M4_MLA_MAX_ATTENDED];")
            && code.contains("__device__ __forceinline__ uint32_t spark_mla_union_attends(")
            && code.contains("return candidate <= position ? 1u : 0u;")
            && code.contains("if (window_kv_format == SPARK_M4_MLA_WINDOW_KV_FORMAT_CSA_UNION) {")
            && code.contains("spark_mla_union_attends(position, candidate) != 0u")
            && code.contains("spark_mla_load_window_stored(window_codes, window_scales, slot, i)")
            && code.contains("window_rows > (uint32_t)SPARK_M4_SWA_WINDOW")
            && code.contains("selected_count > (uint32_t)SPARK_M4_MLA_MAX_SELECTED")
            && code.contains("window_rows + selected_count > (uint32_t)SPARK_M4_MLA_MAX_ATTENDED")
            // The union branch must END the kernel: deleting its terminal return would fall
            // through into historical Stage 2 with a format-2 result already written. Pin the
            // 12-space union trace exit and the exact tail sequence, not just the branch header;
            // the historical Stage-2 exit has different indentation and must not satisfy these.
            && code.contains("            SPARK_GPU_TRACE_EXIT(trace_ring, TRACE_SCOPE_CSA2_ATTN, layer_id, meta, t_enter);")
            && code.contains(
                "            (void)heads;\n            (void)step_seq;\n        }\n        (void)t_enter;\n        return;\n    }",
            )
    };
    assert!(mla_guards_the_window_format(FLASH_MLA_CSA2));

    // The sentinel selector the design rejects: window_rows == 0 cannot express an empty engine
    // window, and it reads badly in the ABI pin because the row count is not a format.
    let sentinel_format = FLASH_MLA_CSA2.replace(
        "if (window_kv_format > SPARK_M4_MLA_WINDOW_KV_FORMAT_CSA_UNION) {",
        "if (window_rows == 0u) {",
    );
    assert_ne!(sentinel_format, FLASH_MLA_CSA2, "the mutation must apply");
    assert!(
        !mla_guards_the_window_format(&sentinel_format),
        "a row-count sentinel is exactly the format selector this design rejects"
    );

    // The format value itself: renumbering the union mode onto the engine value must fail the pin.
    let wrong_union_value = FLASH_MLA_CSA2.replace(
        "#define SPARK_M4_MLA_WINDOW_KV_FORMAT_CSA_UNION 2u",
        "#define SPARK_M4_MLA_WINDOW_KV_FORMAT_CSA_UNION 1u",
    );
    assert_ne!(wrong_union_value, FLASH_MLA_CSA2, "the mutation must apply");
    assert!(
        !mla_guards_the_window_format(&wrong_union_value),
        "the union format must keep its own value, not alias engine format 1"
    );

    // Accepting 3 through a too-wide guard, or falling back to the old two-value guard, is the
    // silent-wrong-arithmetic defect: both must fail.
    let wide_guard = FLASH_MLA_CSA2.replace(
        "if (window_kv_format > SPARK_M4_MLA_WINDOW_KV_FORMAT_CSA_UNION) {",
        "if (window_kv_format > SPARK_M4_MLA_WINDOW_KV_FORMAT_CSA_UNION + 1u) {",
    );
    assert_ne!(wide_guard, FLASH_MLA_CSA2, "the mutation must apply");
    assert!(
        !mla_guards_the_window_format(&wide_guard),
        "a fourth format must not pass the entry guard"
    );
    let two_value_guard = FLASH_MLA_CSA2.replace(
        "if (window_kv_format > SPARK_M4_MLA_WINDOW_KV_FORMAT_CSA_UNION) {",
        "if (window_kv_format > SPARK_M4_MLA_WINDOW_KV_FORMAT_ENGINE_FP8) {",
    );
    assert_ne!(two_value_guard, FLASH_MLA_CSA2, "the mutation must apply");
    assert!(
        !mla_guards_the_window_format(&two_value_guard),
        "the old two-value guard would hand format 2 to the format-1 branch"
    );

    // A refusal that is not visible is worse than no refusal: the poison must stay the NaN output.
    let no_poison = FLASH_MLA_CSA2.replace("0x7FC00000u", "0x0u");
    assert_ne!(no_poison, FLASH_MLA_CSA2, "the mutation must apply");
    assert!(
        !mla_guards_the_window_format(&no_poison),
        "the refused formats must still poison with NaN"
    );

    // The negative pin: a runtime format cannot be a static_assert even when the guard remains.
    let asserted_format =
        format!("{FLASH_MLA_CSA2}\nstatic_assert(window_kv_format > 2u, \"format\");\n");
    assert_ne!(asserted_format, FLASH_MLA_CSA2, "the mutation must apply");
    assert!(
        !mla_guards_the_window_format(&asserted_format),
        "a runtime format value cannot be a static_assert"
    );

    // The formats 0/1 dispatch text: both ternary load sites are pinned, so deleting either one
    // fails even while the other (and the union branch) remains.
    let no_key_site = FLASH_MLA_CSA2.replace(
        "const float key = window_kv_format == SPARK_M4_MLA_WINDOW_KV_FORMAT_FP4",
        "const float key = 1u",
    );
    assert_ne!(no_key_site, FLASH_MLA_CSA2, "the mutation must apply");
    assert!(
        !mla_guards_the_window_format(&no_key_site),
        "the format-0/1 key load dispatch must stay"
    );
    let no_value_site = FLASH_MLA_CSA2.replace(
        "const float value = window_kv_format == SPARK_M4_MLA_WINDOW_KV_FORMAT_FP4",
        "const float value = 1u",
    );
    assert_ne!(no_value_site, FLASH_MLA_CSA2, "the mutation must apply");
    assert!(
        !mla_guards_the_window_format(&no_value_site),
        "the format-0/1 value load dispatch must stay"
    );

    // The union rule is a causal predicate, not a window: changing it to the window predicate or
    // to an exclusive comparison would silently drop out-of-window candidates (the Criterion-1
    // gap) or the query's own position.
    let union_windowed = FLASH_MLA_CSA2.replace(
        "return candidate <= position ? 1u : 0u;",
        "return candidate <= position && spark_mla_in_window(position, candidate) ? 1u : 0u;",
    );
    assert_ne!(union_windowed, FLASH_MLA_CSA2, "the mutation must apply");
    assert!(
        !mla_guards_the_window_format(&union_windowed),
        "an out-of-window selected id must stay attendable in union mode"
    );
    let union_exclusive = FLASH_MLA_CSA2.replace(
        "return candidate <= position ? 1u : 0u;",
        "return candidate < position ? 1u : 0u;",
    );
    assert_ne!(union_exclusive, FLASH_MLA_CSA2, "the mutation must apply");
    assert!(
        !mla_guards_the_window_format(&union_exclusive),
        "an id at the query's own position is attendable"
    );

    // Capacity: exactly 640 is the union of a full candidate list and a full window; a narrower
    // expression (or no sum guard) would write past s_scores.
    let shrunk_capacity = FLASH_MLA_CSA2.replace(
        "#define SPARK_M4_MLA_MAX_ATTENDED (SPARK_M4_MLA_MAX_SELECTED + SPARK_M4_SWA_WINDOW)",
        "#define SPARK_M4_MLA_MAX_ATTENDED (SPARK_M4_MLA_MAX_SELECTED + 64u)",
    );
    assert_ne!(shrunk_capacity, FLASH_MLA_CSA2, "the mutation must apply");
    assert!(
        !mla_guards_the_window_format(&shrunk_capacity),
        "the union capacity expression is one candidate list plus one window"
    );
    let no_capacity_guard = FLASH_MLA_CSA2.replace(
        "window_rows + selected_count > (uint32_t)SPARK_M4_MLA_MAX_ATTENDED",
        "window_rows + selected_count > (uint32_t)SPARK_M4_MLA_LATENT",
    );
    assert_ne!(no_capacity_guard, FLASH_MLA_CSA2, "the mutation must apply");
    assert!(
        !mla_guards_the_window_format(&no_capacity_guard),
        "a union longer than the score array would be an out-of-bounds write"
    );
    let no_selected_cap = FLASH_MLA_CSA2.replace(
        "selected_count > (uint32_t)SPARK_M4_MLA_MAX_SELECTED",
        "selected_count > 1024u",
    );
    assert_ne!(no_selected_cap, FLASH_MLA_CSA2, "the mutation must apply");
    assert!(
        !mla_guards_the_window_format(&no_selected_cap),
        "the selected list is index_topk long"
    );
    let no_window_cap = FLASH_MLA_CSA2.replace(
        "window_rows > (uint32_t)SPARK_M4_SWA_WINDOW",
        "window_rows > 256u",
    );
    assert_ne!(no_window_cap, FLASH_MLA_CSA2, "the mutation must apply");
    assert!(
        !mla_guards_the_window_format(&no_window_cap),
        "the union window is at most the sliding window"
    );

    // The union branch's terminal return: dropping it falls through into historical Stage 2 with a
    // format-2 result already written. The mutation removes the exact tail the pin names, and a
    // second mutation removes only the union trace exit so the historical Stage-2 exit cannot
    // satisfy the pin by itself.
    let union_falls_through = FLASH_MLA_CSA2.replace(
        "        (void)t_enter;\n        return;\n    }\n\n    // Stage 2: attention over the selected candidates.",
        "        (void)t_enter;\n    }\n\n    // Stage 2: attention over the selected candidates.",
    );
    assert_ne!(
        union_falls_through, FLASH_MLA_CSA2,
        "the mutation must apply"
    );
    assert!(
        !mla_guards_the_window_format(&union_falls_through),
        "the union branch must return before historical Stage 2"
    );
    let union_no_trace_exit = FLASH_MLA_CSA2.replace(
        "            SPARK_GPU_TRACE_EXIT(trace_ring, TRACE_SCOPE_CSA2_ATTN, layer_id, meta, t_enter);\n",
        "",
    );
    assert_ne!(
        union_no_trace_exit, FLASH_MLA_CSA2,
        "the mutation must apply"
    );
    assert!(
        !mla_guards_the_window_format(&union_no_trace_exit),
        "the union branch's trace exit is part of its terminal sequence"
    );

    // A6 and A9: the header carries the two scale grids and the row-scale helpers the kernels use,
    // and the MLA constants are defined there rather than repeated in a kernel.
    let header_has_the_grids = |source: &str| -> bool {
        let code = crate::c_decl::strip_comments(source);
        code.contains("#define SPARK_B12X_NTILE 32u")
            && code.contains("#define SPARK_B12X_KTILE 32u")
            && code.contains("spark_b12x_scale_index")
            && code.contains("spark_b12x_row_scale_index")
            && code.contains("spark_b12x_row_scale_bytes")
            && code.contains("spark_b12x_scale_bytes")
    };
    assert!(header_has_the_grids(COMMON_HEADER));
    let one_grid = COMMON_HEADER.replace("spark_b12x_row_scale_index", "spark_b12x_scale_index");
    assert_ne!(one_grid, COMMON_HEADER, "the mutation must apply");
    assert!(
        !header_has_the_grids(&one_grid),
        "the fp4 per-row grid must not collapse into the fp8 per-tile grid"
    );
    assert!(
        FLASH_MLA_CSA2.contains("SPARK_M4_MLA_HEADS_PER_INDEX_HEAD")
            && COMMON_HEADER.contains("#define SPARK_M4_MLA_HEADS_PER_INDEX_HEAD"),
        "the head mapping must be a header constant used by name"
    );
    assert!(
        COMMON_HEADER.contains("#define SPARK_M4_MLA_SCORE_SCALE"),
        "the score scale must be a header constant used by name"
    );
}
