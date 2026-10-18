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
        "const __nv_bfloat16 *__restrict__ hidden; const uint8_t *__restrict__ act_fp8; const uint8_t *__restrict__ act_scales; const uint8_t *__restrict__ w_fp4; const uint8_t *__restrict__ w_scales; const uint32_t *__restrict__ expert_ids; __nv_bfloat16 *__restrict__ combined; const uint32_t tokens; GpuTraceRing *__restrict__ trace_ring; const uint32_t layer_id; const uint64_t step_seq",
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
        "const __nv_bfloat16 *__restrict__ hidden; const __nv_bfloat16 *__restrict__ w; float *__restrict__ logits; uint8_t *__restrict__ certified; unsigned int *__restrict__ certified_count; float *__restrict__ block_max; uint32_t *__restrict__ block_argmax; const uint32_t vocab; GpuTraceRing *__restrict__ trace_ring; const uint32_t layer_id; const uint64_t step_seq",
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
    ("mega_mhc_sm121.cu", 6),
    ("b12x_mxfp8_dense_sm121.cu", 6),
    ("b12x_mxfp4_moe_sm121.cu", 10),
    ("flash_mla_csa2_sm121.cu", 15),
    ("b12x_mxfp8_wo_a_sm121.cu", 11),
    ("cert_lm_head_sm121.cu", 8),
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
        checked, 76,
        "the milestone's static asserts must all be evaluated: 20 in the shared header plus 6, 6, \
         10, 15, 11 and 8 in the six kernels. The MoE count rose from 7 to 10 with D14, which pins \
         the FP8 E4M3 activation operand's byte width in the build itself; flash_mla's rose from 13 \
         to 15 with the engine window format's two layout asserts. The format VALUE is not among \
         them: it is a runtime launch parameter and is guarded at kernel entry instead"
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

/// Every flash_mla launch site goes through the typed parameter builder and its launch wrappers,
/// so no site can keep a stale arity or a hand-built argument order. Mutation: the stage-2 main
/// site is put back on a 15-element array and the pin must reject it.
#[test]
fn every_flash_mla_launch_site_uses_the_typed_parameter_builder() {
    let pin = |source: &str| -> bool {
        let code = crate::c_decl::strip_comments(source);
        // Four constructor calls: the stage-2 indexer self-check, the stage-2 attention loop and
        // the stage-7 and stage-9 K5 launches.
        let builders = code.matches("= flash_mla_params(").count();
        // The typed wrapper is the only launch path: one definition plus six call sites (indexer,
        // stage-2 main, the two injected launches, stage-2 timing, stage-7 K5, stage-9 K5).
        let launches = code.matches("flash_mla_launch(").count();
        let timed = code.matches("flash_mla_launch_repeated(").count();
        builders == 4
            && launches == 7
            && timed == 2
            && !code.contains("[u64; 15]")
            && !code.contains("cuda.launch(mla_func")
            && !code.contains("cuda.launch(dev.mla_func")
    };
    let source = include_str!("../examples/m7_layer0.rs");
    assert!(
        pin(source),
        "a flash_mla launch site bypasses the typed builder/launch wrapper (expected 4 builder calls and 6 wrapper call sites)"
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

    // A2: the MLA score scale is applied, exactly once.
    let mla_scales_the_score = |source: &str| -> bool {
        let code = crate::c_decl::strip_comments(source);
        code.contains("score *= SPARK_M4_MLA_SCORE_SCALE;")
            && code.matches("SPARK_M4_MLA_SCORE_SCALE").count() == 1
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
            && code.contains("acc += (partial * act_scale) * w_scale;")
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
        "acc += (partial * act_scale) * w_scale;",
        "acc += routing * (partial * act_scale) * w_scale;",
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
    // guard is runtime code and this pin is the source text that proves it exists. The mutation is
    // the sentinel selector the design rejects: window_rows == 0 cannot express an empty engine
    // window, and it reads badly in the ABI pin because the row count is not a format.
    let mla_guards_the_window_format = |source: &str| -> bool {
        let code = crate::c_decl::strip_comments(source);
        code.contains("#define SPARK_M4_MLA_WINDOW_KV_FORMAT_FP4 0u")
            && code.contains("#define SPARK_M4_MLA_WINDOW_KV_FORMAT_ENGINE_FP8 1u")
            && code.contains("if (window_kv_format > SPARK_M4_MLA_WINDOW_KV_FORMAT_ENGINE_FP8) {")
            && code.contains("__float2bfloat16_rn(__uint_as_float(0x7FC00000u))")
            && !code.contains("static_assert(window_kv_format")
    };
    assert!(mla_guards_the_window_format(FLASH_MLA_CSA2));
    let sentinel_format = FLASH_MLA_CSA2.replace(
        "if (window_kv_format > SPARK_M4_MLA_WINDOW_KV_FORMAT_ENGINE_FP8) {",
        "if (window_rows == 0u) {",
    );
    assert_ne!(sentinel_format, FLASH_MLA_CSA2, "the mutation must apply");
    assert!(
        !mla_guards_the_window_format(&sentinel_format),
        "a row-count sentinel is exactly the format selector this design rejects"
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

/// One C token, the two lines it can be reported by, and where it started.
///
/// `line` is the line of the text `code_tokens` was handed, and the directive structure is read from
/// it: `end_of_directive` joins a backslash continuation by comparing token lines, so it has to stay
/// the line of the phase-2-spliced text.
///
/// `physical_line` is the line a reader of the file would find the token on. Every diagnostic uses
/// it, because the spliced line is not the file line: the shipped header has 23 splices before the
/// helper, so the helper was reported at m4_common_sm121.cuh:484 while it sits on line 507.
/// `code_tokens` sets it equal to `line` (there is no map for text that was not spliced here);
/// [`code_tokens_of`] replaces it with the mapped line. `offset` is the byte the token starts at in
/// that same text, which is what the map is keyed by.
#[derive(Debug, Clone, PartialEq, Eq)]
struct CodeToken {
    line: usize,
    physical_line: usize,
    offset: usize,
    text: String,
}

/// Splits comment-free C into identifiers, numbers, string or character literals and punctuation,
/// keeping the line of each token so an error can point at it.
fn code_tokens(source: &str) -> Vec<CodeToken> {
    let bytes = source.as_bytes();
    let mut tokens = Vec::new();
    let mut index = 0usize;
    let mut line = 1usize;
    while index < bytes.len() {
        let current = bytes[index];
        if current == b'\n' {
            line += 1;
            index += 1;
            continue;
        }
        if current.is_ascii_whitespace() {
            index += 1;
            continue;
        }
        if current.is_ascii_alphabetic() || current == b'_' {
            let start = index;
            while index < bytes.len()
                && (bytes[index].is_ascii_alphanumeric() || bytes[index] == b'_')
            {
                index += 1;
            }
            // A raw string opener may carry an encoding prefix (u8R"..", uR"..", UR"..", LR"..")
            // or none (R".."). The whole literal, prefix included, is one opaque token: its
            // contents are never code, so a fence or a control keyword spelled inside one cannot
            // be mistaken for the real thing.
            let word = &source[start..index];
            if matches!(word, "R" | "u8R" | "uR" | "UR" | "LR") && bytes.get(index) == Some(&b'"') {
                let end = crate::c_decl::raw_string_end(source, index - 1);
                let text = &source[start..end];
                let start_line = line;
                line += text.matches('\n').count();
                tokens.push(CodeToken {
                    line: start_line,
                    physical_line: start_line,
                    offset: start,
                    text: text.to_string(),
                });
                index = end;
                continue;
            }
            tokens.push(CodeToken {
                line,
                physical_line: line,
                offset: start,
                text: word.to_string(),
            });
            continue;
        }
        if current.is_ascii_digit() {
            let start = index;
            while index < bytes.len()
                && (bytes[index].is_ascii_alphanumeric()
                    || bytes[index] == b'.'
                    || bytes[index] == b'_')
            {
                index += 1;
            }
            tokens.push(CodeToken {
                line,
                physical_line: line,
                offset: start,
                text: source[start..index].to_string(),
            });
            continue;
        }
        if current == b'"' || current == b'\'' {
            // A literal is one token, so a word inside a message can never be read as a keyword.
            let quote = current;
            let start = index;
            index += 1;
            while index < bytes.len() {
                if bytes[index] == b'\\' {
                    index += 2;
                    continue;
                }
                let byte = bytes[index];
                index += 1;
                if byte == quote {
                    break;
                }
            }
            let end = index.min(bytes.len());
            let text = String::from_utf8_lossy(&bytes[start..end]).into_owned();
            let start_line = line;
            line += text.matches('\n').count();
            tokens.push(CodeToken {
                line: start_line,
                physical_line: start_line,
                offset: start,
                text,
            });
            continue;
        }
        if (current == b'&' || current == b'|') && bytes.get(index + 1) == Some(&current) {
            // C's translator has && and || as single tokens, so the guard keeps them whole: split
            // into two characters a short-circuit operator is indistinguishable from something
            // harmless, and is_control has to be able to name it.
            tokens.push(CodeToken {
                line,
                physical_line: line,
                offset: index,
                text: source[index..index + 2].to_string(),
            });
            index += 2;
            continue;
        }
        tokens.push(CodeToken {
            line,
            physical_line: line,
            offset: index,
            text: (current as char).to_string(),
        });
        index += 1;
    }
    tokens
}

/// Tokens of a source file, with every token's `line` set to the PHYSICAL line of `source`.
///
/// [`code_tokens`] counts newlines in the text it is handed, and the guard hands it the phase-2
/// spliced, comment-stripped text. Both stages delete bytes - the splice deletes the newline, the
/// comment scanner deletes the comment - so every token after the first splice in a file was reported
/// a few lines early: the shipped header has 23 splices before the helper, which put the acquiring
/// poll at 484 while `grep -n` finds it at 507. A diagnostic that names a line the developer cannot
/// find teaches them to distrust the guard, so this is what every check that reports a line uses.
fn code_tokens_of(source: &str) -> Vec<CodeToken> {
    let (stripped, lines) = crate::c_decl::strip_comments_with_lines(source);
    let mut tokens = code_tokens(&stripped);
    for token in &mut tokens {
        token.physical_line = lines[token.offset] as usize;
    }
    tokens
}

/// The directive keyword that follows the `#` at `index`, or `""` when the `#` ends the input.
fn directive_keyword(tokens: &[CodeToken], index: usize) -> &str {
    tokens.get(index + 1).map_or("", |next| next.text.as_str())
}

/// The compile-time state of one arm of a preprocessor conditional, from the point of view of a
/// text guard that cannot generally evaluate the condition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Arm {
    /// The arm is definitely compiled.
    Active,
    /// The arm is definitely not compiled.
    Disabled,
    /// The guard cannot resolve the condition, so the arm may or may not be compiled.
    Unknown,
}

/// One open preprocessor conditional.
#[derive(Debug, Clone, Copy)]
struct Conditional {
    /// The state of the arm currently being read.
    arm: Arm,
    /// Whether some previous arm of this conditional may have been taken.
    prior_maybe_taken: bool,
    /// Whether some previous arm of this conditional was definitely taken.
    prior_definitely_taken: bool,
}

/// The arm a `#if`/`#elif` condition denotes when that condition is a literal 0 or 1, and
/// [`Arm::Unknown`] for anything else.
///
/// The guard does not evaluate expressions, but it must not reject a region the compiler certainly
/// compiles either. `#if (1)`, `#if ((0))` and `#if 1u` are all literal: parentheses group, and an
/// integer suffix is part of the same preprocessing token. Reading only a bare `0`/`1` token
/// rejected those, which is the round-5 E5 finding, two rounds after the round-4 A3 false rejection
/// of `#if 1`. Anything else on the directive line (`#if 1 + 1`) stays unknown: folding further
/// would mean evaluating, which a text scan cannot do. That is stricter than the old bare-token read
/// in the other direction as well - `#if 1 - 1` used to be read as active because only its first
/// token was looked at, and is now unknown.
fn literal_arm(tokens: &[CodeToken], index: usize) -> Arm {
    let end = end_of_directive(tokens, index);
    let mut cursor = index + 2;
    let mut open = 0usize;
    while tokens.get(cursor).is_some_and(|token| token.text == "(") {
        open += 1;
        cursor += 1;
    }
    let Some(token) = tokens.get(cursor) else {
        return Arm::Unknown;
    };
    let digits_end = token
        .text
        .find(|byte: char| !byte.is_ascii_digit())
        .unwrap_or(token.text.len());
    let (digits, suffix) = token.text.split_at(digits_end);
    let arm = match digits {
        "0" => Arm::Disabled,
        "1" => Arm::Active,
        _ => return Arm::Unknown,
    };
    if !suffix
        .chars()
        .all(|byte| matches!(byte, 'u' | 'U' | 'l' | 'L'))
    {
        return Arm::Unknown;
    }
    cursor += 1;
    let mut close = 0usize;
    while close < open && tokens.get(cursor).is_some_and(|token| token.text == ")") {
        close += 1;
        cursor += 1;
    }
    if close != open || cursor != end {
        return Arm::Unknown;
    }
    arm
}

/// Applies one preprocessor directive to the open-conditional stack.
///
/// The guard cannot run the preprocessor, so it tracks a three-way state per level rather than the
/// boolean "disabled" it used before. A literal `#if 0` is definitely disabled, a literal `#if 1`
/// is definitely active, and anything else (`#ifdef`, `#if defined(...)`, an arithmetic condition)
/// is [`Arm::Unknown`]: it may hold, and the guard must not pretend either way. The previous
/// version collapsed `#if 1` and `#ifdef` into the same "not 0" state and then treated any open
/// level as not compiled, which rejected a poll or a fence that a literal `#if 1` really does
/// compile; that was the round-4 A3 false rejection. `#else` and `#elif` invert the arms that came
/// before, and a level whose earlier arm may have been taken stays [`Arm::Unknown`] even when its
/// own literal is `1`.
fn update_conditionals(tokens: &[CodeToken], index: usize, conditionals: &mut Vec<Conditional>) {
    match directive_keyword(tokens, index) {
        "if" => {
            let arm = literal_arm(tokens, index);
            conditionals.push(Conditional {
                arm,
                prior_maybe_taken: arm != Arm::Disabled,
                prior_definitely_taken: arm == Arm::Active,
            });
        }
        "ifdef" | "ifndef" => conditionals.push(Conditional {
            arm: Arm::Unknown,
            prior_maybe_taken: true,
            prior_definitely_taken: false,
        }),
        "elif" => {
            if let Some(level) = conditionals.last_mut() {
                level.arm = if level.prior_definitely_taken {
                    Arm::Disabled
                } else if level.prior_maybe_taken {
                    Arm::Unknown
                } else {
                    literal_arm(tokens, index)
                };
                level.prior_maybe_taken |= level.arm != Arm::Disabled;
                level.prior_definitely_taken |= level.arm == Arm::Active;
            }
        }
        "else" => {
            if let Some(level) = conditionals.last_mut() {
                level.arm = if level.prior_definitely_taken {
                    Arm::Disabled
                } else if level.prior_maybe_taken {
                    Arm::Unknown
                } else {
                    Arm::Active
                };
            }
        }
        "endif" => {
            conditionals.pop();
        }
        _ => {}
    }
}

/// Whether every enclosing conditional level is definitely compiled, so a token there is certainly
/// in the translation unit.
fn definitely_compiled(conditionals: &[Conditional]) -> bool {
    conditionals.iter().all(|level| level.arm == Arm::Active)
}

/// Whether some enclosing level is definitely disabled, so a token there is certainly not compiled.
fn definitely_disabled(conditionals: &[Conditional]) -> bool {
    conditionals.iter().any(|level| level.arm == Arm::Disabled)
}

/// The index just past a preprocessor directive's whole logical line, backslash continuations
/// included. A fence or a poll on a continuation line is part of the directive, not code.
fn end_of_directive(tokens: &[CodeToken], index: usize) -> usize {
    let mut directive_line = tokens[index].line;
    let mut cursor = index;
    loop {
        while cursor < tokens.len() && tokens[cursor].line == directive_line {
            cursor += 1;
        }
        let continued = cursor
            .checked_sub(1)
            .and_then(|previous| tokens.get(previous))
            .is_some_and(|previous| previous.text == "\\");
        if !continued {
            return cursor;
        }
        match tokens.get(cursor) {
            Some(next) => directive_line = next.line,
            None => return cursor,
        }
    }
}

/// Marks every token that is definitely compiled: not on a preprocessor directive line and in a
/// region where every enclosing conditional level is definitely active.
///
/// A poll is only a poll on such a token, because otherwise the compiler may never reach it. A
/// region the guard cannot resolve (`#ifdef`, an arithmetic `#if`) is NOT marked, so a poll that
/// lives only there is reported as no poll: that is the conservative half of the rule and it is
/// deliberate, not a limitation to be removed. A literal `#if 1` and the `#else` arm of a literal
/// `#if 0` ARE marked, because the compiler certainly compiles them; treating those as unreachable
/// was the round-4 A3 false rejection.
///
/// This is deliberately stricter than the post-poll fence walk, which accepts a fence whenever
/// every enclosing level is definitely active but rejects one in a may-not-hold region rather than
/// reading it as unconditional.
fn active_code_mask(tokens: &[CodeToken]) -> Vec<bool> {
    let mut active = vec![false; tokens.len()];
    let mut conditionals: Vec<Conditional> = Vec::new();
    let mut index = 0usize;
    while index < tokens.len() {
        if tokens[index].text == "#" {
            update_conditionals(tokens, index, &mut conditionals);
            index = end_of_directive(tokens, index);
            continue;
        }
        if definitely_compiled(&conditionals) {
            active[index] = true;
        }
        index += 1;
    }
    active
}

/// The conditional stack in effect at the code token `stop`, recovered by scanning from the top of
/// the token list.
///
/// The fence walk starts at the poll, so without this it would begin with an empty stack and read
/// the `#else` of an enclosing `#if 1` as unconditional code. Recovering the stack at the poll
/// keeps the two scans of the same token list consistent.
fn conditional_stack_before(tokens: &[CodeToken], stop: usize) -> Vec<Conditional> {
    let mut conditionals = Vec::new();
    let mut index = 0usize;
    while index < stop && index < tokens.len() {
        if tokens[index].text == "#" {
            update_conditionals(tokens, index, &mut conditionals);
            index = end_of_directive(tokens, index);
            continue;
        }
        index += 1;
    }
    conditionals
}

/// Whether a token ends the fall-through path, or makes the fence's execution conditional.
///
/// `throw` and `co_return` are here because they end the fall-through path where they are legal: a
/// device function may not throw (nvcc refuses `throw` in device code) and a device function may not
/// be a coroutine, so neither can actually ship a missing acquire - but the guard reads the shape of
/// the source, and a body that spells `throw 0u;` between the poll and the fence has an exit it
/// cannot see through, so it is treated like the other control tokens rather than left to a proof
/// about which constructs nvcc accepts.
fn is_control(text: &str) -> bool {
    matches!(
        text,
        "if" | "else"
            | "for"
            | "while"
            | "do"
            | "switch"
            | "case"
            | "default"
            | "return"
            | "break"
            | "continue"
            | "goto"
            | "throw"
            | "co_return"
            | "?"
            // A short-circuit operator makes what follows conditional, which is exactly the hazard
            // here: (ready == 0u) && ({ __threadfence(); 1; }) runs the fence only on one branch.
            // The single-character forms are rejected as well, because they are the halves of the
            // same operators if the tokenizer is ever changed again, and because a bitwise & or |
            // with a statement-expression operand is the same hazard. Rejecting them is deliberately
            // conservative: this guard is allowed to refuse a shape it cannot prove, and the kernel
            // it protects has neither operator between its poll and its fence.
            | "&&"
            | "||"
            | "&"
            | "|"
    )
}

/// Whether the control token is an OPERATOR, which is the only kind the statement scoping of
/// [`a_statement_completed_between`] may excuse. A control KEYWORD is never excused: `if`, `while`,
/// `for`, `switch` can have the poll inside their body, and `return`, `break`, `continue`, `goto`,
/// `throw` can leave the helper before it, so `if (ready) return 0u;` before the poll is still a
/// blocker even though that statement completed.
fn is_control_operator(text: &str) -> bool {
    matches!(text, "&&" | "||" | "&" | "|" | "?")
}

/// Whether a control keyword leaves the path rather than making it conditional. A `return`, `break`,
/// `continue`, `goto`, `throw` or `co_return` before the poll sends the path elsewhere, so the
/// acquire after it is UNREACHABLE, which is a different fact from "the acquire may not be reached"
/// (round-20 A6). The distinction is not cosmetic: `return poll(...); __threadfence();` calls the poll
/// unconditionally and the fence after it is dead code, and reporting that as a conditional evaluation
/// was false of it.
fn is_an_exit_keyword(text: &str) -> bool {
    matches!(
        text,
        "return" | "break" | "continue" | "goto" | "throw" | "co_return"
    )
}

/// Whether an exit keyword ends the HELPER's path, as the argument-span reading counts them: `return`,
/// `goto`, `throw` and `co_return`. `break` and `continue` are loop-scoped and cannot end the helper
/// from inside an argument list, and a `?` or a `&&` is not an exit at all (round-25 A3).
fn is_an_argument_exit(text: &str) -> bool {
    matches!(text, "return" | "goto" | "throw" | "co_return")
}

/// Whether the `{` at `index` opens a LAMBDA body rather than a block of the function being read. A
/// `return` inside it exits the lambda, not the helper, which the documentation has promised since
/// round 10, so the argument-span reading skips such a body (round-25 A3).
///
/// Round 30 rewrote the walk-back to read BALANCED GROUPS instead of guessing from the token before
/// the brace, because the guess produced two false verdicts. Working backwards from the body brace,
/// one piece at a time: a TRAILING RETURN TYPE is skipped by finding the `->` that introduces it,
/// so the `)` of `decltype(auto)` inside the type is a group rather than the end of the scan; then
/// the specifiers; then a balanced `(...)` parameter list; then a balanced `<...>` explicit template
/// parameter list; and finally the `]` of the CAPTURE LIST, which is what makes it a lambda. An
/// attribute is a `[[...]]` pair, so a `]` that closes one is skipped instead of being read as a
/// capture list: `[[maybe_unused]] { ... }` is a plain block and stays refused, while
/// `[x = (&deadline)[0]] { ... }` and `[&] [[maybe_unused]] { ... }` are lambdas and are accepted.
fn a_lambda_body_opens_at(tokens: &[CodeToken], index: usize) -> bool {
    if tokens.get(index).map(|token| token.text.as_str()) != Some("{") {
        return false;
    }
    // (1) The trailing return type, if the body has one: the brace follows `-> type`, and a type can
    // hold `(...)`, `[...]`, `{...}` and `<...>` groups as well as `*`, `const`, `::` and
    // identifiers. Round 31 A1: each bracket KIND gets its own counter, because a `<` or a `>`
    // written inside a parenthesised group is a comparison in a type - `std::enable_if_t<(1 < 2),
    // int>`, `decltype(1 << 2, int{})`, `decltype((1 <= 2), int{})` - and one shared counter let
    // it cancel the `)` of that group, so the scan never reached the arrow and the trailing-return
    // forms were refused with an unreachability sentence that was false of the compiled body. A
    // `;`, a top-level brace or a top-level opener ends the scan with no arrow.
    let mut cursor = index;
    let mut probe = index;
    let mut parens = 0usize;
    let mut squares = 0usize;
    let mut angles = 0usize;
    let mut braces = 0usize;
    let mut walked = 0usize;
    while probe > 0 {
        probe -= 1;
        walked += 1;
        if walked > 256 {
            return false;
        }
        let text = tokens[probe].text.as_str();
        // The tokenizer splits the arrow into `-` and `>`, so the pair is what introduces a
        // trailing return type. This is checked before the `>` arm below, which counts a `>` as
        // the close of a template group. An arrow is never inside a group.
        if text == ">"
            && parens == 0
            && squares == 0
            && braces == 0
            && probe > 0
            && tokens[probe - 1].text == "-"
        {
            cursor = probe - 1;
            break;
        }
        match text {
            // A `]` that closes no group of its own is the capture list, so the brace is the body
            // of a lambda (or of nothing this detector reads) and phase 1 is over.
            "]" if parens == 0 && squares == 0 && angles == 0 && braces == 0 => break,
            ")" => parens += 1,
            "]" => squares += 1,
            "}" => braces += 1,
            "(" => {
                if parens == 0 {
                    break;
                }
                parens -= 1;
            }
            "[" => {
                if squares == 0 {
                    break;
                }
                squares -= 1;
            }
            "{" => {
                if braces == 0 {
                    break;
                }
                braces -= 1;
            }
            // An angle bracket inside another kind of group is a comparison, not a delimiter of
            // the type: it is ignored so the group it sits in still closes on its own counter.
            // The `>` of a `>=` is a comparison as well (round 32 A1): counting it left the
            // group it sits in open, so the scan never reached the arrow.
            ">" if parens == 0 && squares == 0 && braces == 0 => {
                if tokens.get(probe + 1).map(|next| next.text.as_str()) != Some("=") {
                    angles += 1;
                }
            }
            // Round 32 A1: a top-level `<` is a delimiter only when a template NAME stands in
            // front of it. `std::enable_if_t<1 < 2, int>` and `enable_if_t<N < 2, int>` write a
            // COMPARISON there, and counting it closed the group the comparison sat in, so the
            // scan stopped at the real opener and the trailing-return forms were refused with
            // the unreachability sentence. A comparison neither opens nor closes anything, and
            // it does not end the scan either - the scan ends at a `;`, a top-level brace, the
            // beginning of the text or the step bound, so the walk still reaches the arrow.
            "<" if parens == 0 && squares == 0 && braces == 0 => {
                if a_template_name_stands_before(tokens, probe) && angles > 0 {
                    angles -= 1;
                }
            }
            ";" if parens == 0 && squares == 0 && angles == 0 && braces == 0 => break,
            _ => {}
        }
    }
    // (2) The pieces between the capture list and the body, in reverse order. The first `]` that is
    // NOT the close of an attribute is the capture list, so it is the one that answers the question.
    let mut pieces = 0usize;
    loop {
        pieces += 1;
        if pieces > 64 || cursor == 0 {
            return false;
        }
        match tokens[cursor - 1].text.as_str() {
            "mutable" | "noexcept" | "constexpr" | "consteval" => cursor -= 1,
            "]" => match an_attribute_closes_at(tokens, cursor - 1) {
                Some(opener) => cursor = opener,
                None => return true,
            },
            ")" => match the_matching_opener(tokens, cursor - 1, "(", ")") {
                Some(opener) => cursor = opener,
                None => return false,
            },
            // The angle matcher skips the groups nested inside the template argument or parameter
            // list, so a comparison there is not read as a delimiter (round 31 A1).
            ">" => match the_matching_angle_opener(tokens, cursor - 1) {
                Some(opener) => cursor = opener,
                None => return false,
            },
            _ => return false,
        }
    }
}

/// The index of the `<` that matches the template closer at `close`, walking backwards. Unlike
/// `the_matching_opener` this one tracks every bracket kind, because a template argument or
/// parameter list may CONTAIN groups - `std::enable_if_t<(1 < 2), int>`, `[&]<auto N = (1 << 2)>()`,
/// `decltype(int{})` - and a `<` or `>` written inside one of them is a comparison rather than a
/// delimiter, which is exactly the round-31 A1 defect: one shared counter let `(1 < 2)` cancel the
/// `)` it sits in, so the scan walked past the arrow. Inside a group the angle brackets here are
/// ignored; at group depth zero they nest as usual. The arrow's own `>` (`-` then `>` in this token
/// stream) is not a template close. `None` when the brackets do not balance.
fn the_matching_angle_opener(tokens: &[CodeToken], close: usize) -> Option<usize> {
    let mut parens = 0usize;
    let mut squares = 0usize;
    let mut braces = 0usize;
    let mut angles = 1usize;
    let mut cursor = close;
    // Round 32 A1: `>` and `<` are comparisons as well, so the first `<` that would close the
    // count is not always the opener - in `[&]<auto N = M < 2>()` it is the comparison after
    // `M`, in front of which no `]` stands, so phase 2 cannot consume it. The walk therefore
    // REMEMBERS the leftmost opener it has matched and keeps looking: a comparison inside the
    // group must not stand in for its opener, which is the reading the standard gives when the
    // name in front of the inner `<` is a value rather than a template.
    let mut opener: Option<usize> = None;
    while cursor > 0 {
        cursor -= 1;
        let text = tokens[cursor].text.as_str();
        if text == ">"
            && parens == 0
            && squares == 0
            && braces == 0
            && cursor > 0
            && tokens[cursor - 1].text == "-"
        {
            continue;
        }
        match text {
            ")" => parens += 1,
            "]" => squares += 1,
            "}" => {
                if braces == 0 {
                    return opener;
                }
                braces -= 1;
            }
            "(" => {
                if parens == 0 {
                    return None;
                }
                parens -= 1;
            }
            "[" => {
                if squares == 0 {
                    return None;
                }
                squares -= 1;
            }
            "{" => {
                // The start of a block ends the statement, exactly as the `;` below does.
                if braces == 0 {
                    return opener;
                }
                braces -= 1;
            }
            // A `;` or a preprocessor line ends the statement, so no group can open before
            // it: the candidate found inside the statement is the answer.
            ";" | "#" if parens == 0 && squares == 0 && braces == 0 => return opener,
            // The `>` of a `>=` is a comparison, not a template close (round 32 A1).
            ">" if parens == 0 && squares == 0 && braces == 0 => {
                if tokens.get(cursor + 1).map(|next| next.text.as_str()) != Some("=") {
                    angles += 1;
                }
            }
            // Round 32 A1: only a `<` with a template NAME in front of it, or the one that
            // follows a generic lambda's capture list, opens a template group; `1 < 2` and
            // `1 <= 2` write a comparison there.
            "<" if parens == 0 && squares == 0 && braces == 0 => {
                let named = a_template_name_stands_before(tokens, cursor)
                    || tokens
                        .get(cursor.wrapping_sub(1))
                        .map(|before| before.text.as_str())
                        == Some("]");
                if named {
                    angles -= 1;
                    if angles == 0 {
                        opener = Some(cursor);
                        angles = 1;
                    }
                }
            }
            _ => {}
        }
    }
    opener
}
/// The index of the opener that matches the closer at `close`, walking backwards. `None` when the
/// tokens do not balance, which is also how a caller refuses to guess (round-30 A2).
fn the_matching_opener(
    tokens: &[CodeToken],
    close: usize,
    open: &str,
    shut: &str,
) -> Option<usize> {
    let mut depth = 0usize;
    let mut cursor = close + 1;
    while cursor > 0 {
        cursor -= 1;
        let text = tokens[cursor].text.as_str();
        if text == shut {
            depth += 1;
        } else if text == open {
            if depth == 0 {
                return None;
            }
            depth -= 1;
            if depth == 0 {
                return Some(cursor);
            }
        }
    }
    None
}

/// Whether the `]` at `close` ends an ATTRIBUTE-SPECIFIER, `[[...]]`, which is written with its two
/// brackets adjacent on both sides. An init-capture whose initialiser ends in a subscript ends `]]`
/// as well - `[x = (&deadline)[0]]` - but its group starts `[x`, so it is a capture list and this
/// returns `None`, which makes the walk-back answer `true`. Round 29 tested the two tokens before the
/// brace instead and false-refused that lambda and `[&] [[maybe_unused]] { ... }` (round-30 A2).
fn an_attribute_closes_at(tokens: &[CodeToken], close: usize) -> Option<usize> {
    if tokens.get(close).map(|token| token.text.as_str()) != Some("]") {
        return None;
    }
    let opener = the_matching_opener(tokens, close, "[", "]")?;
    if opener + 1 < close && tokens[opener + 1].text == "[" && tokens[close - 1].text == "]" {
        Some(opener)
    } else {
        None
    }
}

/// The last token before the attribute groups standing at `last`, if there are any.
///
/// C++ lets a balanced attribute group stand between a control statement's head and the brace of its
/// body: `if (deadline) [[likely]] { ... }`, and the same after `else`, `do`, `for`, `while` and
/// `switch`. It does NOT for `try` or `catch`: the grammar has no attribute between `try` and its
/// compound statement, nor between a catch clause's `)` and its compound statement, and both g++ 16 and
/// clang 22 reject `try [[likely]] {` and `catch (...) [[likely]] {` at parse in c++17, c++20, c++23 and
/// c++2c ("expected `{` before `[`"). Those two spellings are still inputs here because the readings
/// below run on a token stream, and the fixtures pin this walk with them. The token at `last` is then
/// the attribute's closing bracket, which is no control token, and a rule that tests it alone reads the
/// body as a bare block. What that hid: the refusal the plain spellings rest on is keyed on the token
/// that ends the head - the `)` of an if/for/while/switch/catch condition - or on `try`, `do` and
/// `requires` naming a body directly, so for `catch (...) [[likely]] {` it is the `)` of the parameter
/// list that the attribute bracket merely hid, and the attributed `try` and `catch` bodies were ACCEPTED
/// while the plain ones were refused (round-36 A1). Neither `try` nor `catch` is in [is_control], so no
/// later blocker rescues them.
/// [a_bare_block_opens_at] has stepped over the group since round-34 A1; this is the same walk, shared
/// so the three rules that need it cannot drift apart again.
fn the_token_before_the_attributes(tokens: &[CodeToken], last: Option<usize>) -> Option<usize> {
    let mut before = last;
    while let Some(close) = before {
        let Some(opener) = an_attribute_closes_at(tokens, close) else {
            break;
        };
        before = opener.checked_sub(1);
    }
    before
}

/// Whether a `;` at the same brace nesting as `from` stands between `from` and `to`, which means the
/// token at `from` belongs to a statement that was already complete before `to` began.
///
/// Round-19 A1: this is what keeps a control OPERATOR in an earlier, finished statement from being read
/// as a blocker of the poll's own statement. `const bool ran = (deadline > 0u) && true;` before the poll
/// ends at its `;`, so by the time the poll is evaluated nothing about that operator bounds the path -
/// and calling its evaluation conditional would be false. A control KEYWORD (`if`, `while`, `return`)
/// before the poll is NOT excused by this: its body can still bound the path, and the poll may be
/// inside it. A macro whose body carries only an operator obeys the same scoping as the literal
/// operator (round-20 A1), which is what makes `#define PRE (deadline > 0u) && true` before the poll
/// accepted while `#define BAIL return` before it stays refused.
fn a_statement_completed_between(tokens: &[CodeToken], from: usize, to: usize) -> bool {
    let mut depth = 0i32;
    for token in tokens.iter().take(to).skip(from + 1) {
        match token.text.as_str() {
            "{" => depth += 1,
            "}" => depth -= 1,
            ";" if depth == 0 => return true,
            _ => {}
        }
    }
    false
}

/// Whether the control token at `control` OWNS the poll at `poll` as the condition the poll is
/// evaluated in, rather than standing before it the way a blocker means.
///
/// Round-18 A2/A3: the poll may be the condition of `if`, `while`, `switch`, `for` or `catch`, or an
/// operand of the conditional operator `?`, and the fence after that whole statement runs
/// unconditionally - so a control token that owns the statement the poll stands in is not a blocker.
/// What is still a blocker is everything that can take the path away BEFORE the poll: `return
/// poll(...)` (a `return` is never a condition), `if (x) poll(...);` (the poll is in the body, and
/// `x` may be false), a short-circuit operator `ready && poll(...)` (the poll may not be evaluated)
/// and a `?` in an earlier, completed statement. The `(` is looked for between the keyword and the
/// poll, so a keyword whose condition has already closed does not qualify.
fn the_control_token_governs_the_poll(tokens: &[CodeToken], control: usize, poll: usize) -> bool {
    if control >= poll {
        return false;
    }
    let text = tokens[control].text.as_str();
    if text == "?" {
        // The conditional expression's own statement: a `;` or a brace between the `?` and the poll
        // puts the `?` in an earlier statement, where it cannot govern anything.
        return !tokens[control..poll]
            .iter()
            .any(|token| matches!(token.text.as_str(), ";" | "{" | "}"));
    }
    if !matches!(text, "if" | "while" | "switch" | "for" | "catch") {
        return false;
    }
    let Some(open) = (control + 1..poll).find(|&index| tokens[index].text == "(") else {
        return false;
    };
    matching_close_bracket(tokens, open).is_some_and(|close| close >= poll)
}

/// Calls that never return, so a fence after one is unreachable rather than unconditional. The
/// list names the standard spellings a kernel could reach: the compiler builtins
/// (`__builtin_trap`, `__builtin_unreachable`), the C/POSIX process exits (`abort`, `exit`,
/// `_Exit`, `_exit`, `quick_exit`), the C11 thread exit (`thrd_exit`), the POSIX thread exit
/// (`pthread_exit`) and `longjmp`. A name reached through a macro, a function pointer or an
/// inline-asm block is not covered, which is a stated limit.
const NON_RETURNING_CALLS: [&str; 11] = [
    "__builtin_trap",
    "__builtin_unreachable",
    "__trap",
    "abort",
    "exit",
    "_Exit",
    "_exit",
    "quick_exit",
    "thrd_exit",
    "pthread_exit",
    "longjmp",
];

/// Whether the name at `index` is used as a called designator, in every spelling the preprocessor
/// leaves alone: `f(...)`, `(f)(...)`, `(*f)(...)` and `(((f)))(...)`.
///
/// This is the same unwrap as [`call_argument_list`] without the "and a semicolon follows the
/// argument list" part, because a call inside an expression is still a call. It exists because the
/// round-5 review replaced the acquiring call with `(spark_wait_for_flag)(&hc_sync[1], deadline)`:
/// parenthesising a function designator does not stop it from being called, and the preprocessor does
/// not rewrite it, so a check that only looked at the token after the name saw no call at all.
fn designator_is_a_call(tokens: &[CodeToken], index: usize) -> bool {
    call_argument_list(tokens, index).is_some()
}

/// The indices of the raw poll designator being CALLED, in source order.
///
/// Both checks locate the poll by this one rule - the raw poll designator is called - so their
/// verdicts cannot diverge on the poll's spelling (round-13 A-2). The lint used to take a literal
/// token pattern as its `poll` argument, so `spark_wait_for_flag((flag), (deadline))`, a cast
/// argument and plain spacing differences were located by the structural check and missed by the
/// lint, which then reported "no poll of spark_wait_for_flag(flag, deadline) in this source's code"
/// for code nvcc compiles to the baseline cubin byte for byte (70432 / 4 / 605321041ac5ec1e, measured).
fn raw_poll_call_indices(tokens: &[CodeToken]) -> Vec<usize> {
    let directives = directive_token_mask(tokens);
    // The macros in effect at each position, in source order: whether a name in front of an argument
    // list is a macro USE (round-4 A2: `BAIL poll(...)`) or the return TYPE of a declaration
    // (`#define uint32_t unsigned` then `uint32_t spark_wait_for_flag(...);`, round-15 A-1) is a
    // question about that macro, so the scope walks the text with this scan.
    let mut macros = MacroScope::from_tokens(tokens);
    let mut found = Vec::new();
    for index in 0..tokens.len() {
        macros.advance_to(index);
        if tokens[index].text == RAW_POLL
            && designator_is_a_call(tokens, index)
            && !is_a_declaration_of(tokens, index, &macros, &directives)
        {
            found.push(index);
        }
    }
    found
}

/// Whether the called designator at `index` is a DECLARATION of that function rather than a call of
/// it, whatever its return type: the argument list is followed by `;` (a prototype), by `=` (a
/// defaulted or pure-virtual declaration) or by the `{` of a body, or a declaration specifier
/// stands before the name (`void abort();`).
///
/// The `;` rule is what makes the return type irrelevant. Round-14 A-1: a plain forward
/// declaration `uint32_t spark_wait_for_flag(const volatile uint32_t *, uint64_t);` was read as a
/// CALL, because `uint32_t` is not in `DECLARATION_SPECIFIERS` and the declaration does not open a
/// body - so the lint anchored on plumbing and reported a `return` inside this very definition as
/// standing before the poll, for a header nvcc compiles to the baseline cubin.
///
/// A function-pointer PARAMETER is deliberately not covered: `(*f)(args)` is the same tokens
/// whether it declares a parameter or calls through a pointer, so the lint anchors on the helper's
/// own body rather than guessing from the following bracket.
/// Whether the mention at `index` is spelled with a VOID return type: `void (spark_wait_for_flag)
/// (flag, deadline);`. The parentheses around the designator are also the shape of a call through a
/// parenthesised designator, so the parentheses alone cannot decide it; the declaration specifier in
/// front of them can. The tokens redeclare the poll at block scope with a `void` return type, and nvcc
/// refuses the translation unit rather than compiling it: "error: incomplete type void is not allowed"
/// (round-23 finding 5, measured in the CUDA 13.0.3 container on sm_121a). Both checks refuse it, with
/// the same sentence, because a body that is not a translation unit has no cubin to compare with the
/// baseline.
fn a_void_type_precedes_the_designator(tokens: &[CodeToken], index: usize) -> bool {
    let Some(open) = index
        .checked_sub(1)
        .filter(|&previous| tokens[previous].text == "(")
    else {
        return false;
    };
    if matching_close_bracket(tokens, open) != Some(index + 1) {
        return false;
    }
    open.checked_sub(1)
        .is_some_and(|before| tokens[before].text == "void")
}

/// The shared message for that shape. The callers wrap it in the one constructor, so the two checks
/// answer the spelling with the same words.
fn void_typed_poll_message(line: usize) -> String {
    format!(
        "line {line}: {RAW_POLL} is spelled with a void return type here - `void ({RAW_POLL})(...)` \
         redeclares the observation instead of calling it, and nvcc refuses the translation unit \
         (\"error: incomplete type void is not allowed\"); the shared header writes the call as \
         `{CANONICAL_POLL_CALL}`"
    )
}

fn is_a_declaration_of(
    tokens: &[CodeToken],
    index: usize,
    macros: &MacroScope,
    directives: &[bool],
) -> bool {
    if a_void_type_precedes_the_designator(tokens, index) {
        return true;
    }
    if index > 0 && DECLARATION_SPECIFIERS.contains(&tokens[index - 1].text.as_str()) {
        return true;
    }
    let after_the_arguments = call_argument_list(tokens, index)
        .and_then(|(_, open)| matching_close_bracket(tokens, open))
        .map(|close| skip_noexcept_clause(tokens, close + 1))
        .and_then(|next| tokens.get(next))
        .map(|next| next.text.as_str());
    match after_the_arguments {
        // A body brace cannot follow a call's argument list, so this needs no type: it is the
        // definition of the poll itself.
        Some("{") => true,
        // A prototype or a defaulted/pure-virtual declaration. A bare call statement ends with `;`
        // too, so the token before the name has to say which one this is: a TYPE name means the name
        // is being declared, anything else (`=`, a bracket, an operator, a statement keyword) means
        // it is being called.
        Some(";" | "=") => a_type_stands_before(tokens, index, macros, directives),
        _ => false,
    }
}

/// The token after an optional `noexcept` or `noexcept(...)` clause. A declaration of the poll may
/// carry one - `uint32_t poll(...) noexcept;` - and it is still a declaration, not a call (round-15
/// A6).
fn skip_noexcept_clause(tokens: &[CodeToken], mut index: usize) -> usize {
    let is_noexcept = tokens
        .get(index)
        .is_some_and(|token| token.text == "noexcept");
    if is_noexcept {
        index += 1;
    }
    // The clause is either bare or parenthesised; the parentheses are the only part with a shape.
    let parenthesised = if is_noexcept && tokens.get(index).is_some_and(|token| token.text == "(") {
        matching_close_bracket(tokens, index)
    } else {
        None
    };
    if let Some(close) = parenthesised {
        index = close + 1;
    }
    index
}

/// Whether a type name stands before the called designator at `index`, through a pointer or
/// reference chain: the return type of a declaration whose type is a typedef or a class name
/// (`uint32_t spark_wait_for_flag(...)`), which `DECLARATION_SPECIFIERS` cannot see.
///
/// The token in front of a CALL is `=`, a bracket, an operator or a statement keyword
/// (`... = poll;`, `(poll)`, `return poll;`, `ready > poll(...)`), and the shapes that have
/// something else there are the return types a declaration can have:
///
///   * an identifier that is not a statement keyword - a typedef, a class name, an unknown name -
///     which is a TYPE unless the source defines it as a macro that can be INVOKED. A macro name can
///     be a return type (`#define uint32_t unsigned` then `uint32_t spark_wait_for_flag(...);` is
///     a declaration, round-15 A-1), and the shapes are otherwise identical, so the macro's own body
///     decides: a macro whose expansion contains a control keyword or a non-returning call is a
///     control STATEMENT here (`#define BAIL return` then `BAIL poll(...);` must stay a call,
///     round-4 A2), and any other macro is read as the type it stands for;
///   * a `>` that closes a template argument list (`wrapper<uint32_t> spark_wait_for_flag(...);`)
///     or a `)` that closes a `decltype(...)` - the two return types that end in punctuation.
///     Neither is accepted on a guess: the `<` has to be there (so a comparison, `ready >
///     poll(...)`, stays a call) and the token in front of the `<` must not make it an arrow
///     (`p->poll(...)`), and only `decltype` may open the parentheses.
///
/// A token of a `#define` line is excluded: it belongs to that directive (`#define CHAIN1 CHAIN0`
/// followed by the poll), not to a declaration.
fn a_type_stands_before(
    tokens: &[CodeToken],
    index: usize,
    macros: &MacroScope,
    directives: &[bool],
) -> bool {
    let mut cursor = index;
    loop {
        if cursor == 0 {
            return false;
        }
        let previous = cursor - 1;
        match tokens[previous].text.as_str() {
            // A pointer or reference declarator belongs to the type.
            "*" | "&" => cursor = previous,
            // An attribute-specifier between the type and the name belongs to the declaration:
            // `uint32_t [[maybe_unused]] poll(...);` declares the name (round-15 A2). The tokenizer
            // emits one token per punctuation character, so the closing pair is two `]` tokens and the
            // one in front of the name is a single `]` - matching `"]]"` here matched nothing, which
            // made this arm dead and let a declaration through as a call (round-17 A1).
            "]" => match attribute_specifier_open(tokens, previous) {
                Some(open) => cursor = open,
                None => return false,
            },
            // The opening parenthesis of a function-pointer declarator, `TYPE (*name)(args)`: the
            // name stands inside `(* ... )` and the type is what stands before the `(`. The same
            // tokens are a call THROUGH a function pointer, so this accepts the declarator and cannot
            // separate the two; that is stated as out of scope in the module documentation.
            "(" if tokens
                .get(previous + 1)
                .is_some_and(|next| matches!(next.text.as_str(), "*" | "&")) =>
            {
                cursor = previous;
            }
            _ => break,
        }
    }
    let previous = cursor - 1;
    if directives[previous] {
        return false;
    }
    let text = tokens[previous].text.as_str();
    if text == ">" {
        let not_an_arrow = previous == 0 || tokens[previous - 1].text != "-";
        return not_an_arrow && matching_open_angle(tokens, previous).is_some();
    }
    if text == ")" {
        return decltype_stands_before(tokens, previous);
    }
    if !is_identifier(text) || STATEMENT_KEYWORDS.contains(&text) {
        return false;
    }
    if macros.is_defined(text) {
        return !macros.is_suspicious(text);
    }
    true
}

/// The index of the `[[` that opens the attribute-specifier the `]]` at `close` ends, or None. A
/// pair of adjacent brackets is what makes this an attribute-specifier rather than `a[i]`.
fn attribute_specifier_open(tokens: &[CodeToken], close: usize) -> Option<usize> {
    let mut depth = 0usize;
    let mut cursor = close;
    loop {
        match tokens[cursor].text.as_str() {
            "]" => depth += 1,
            "[" => {
                depth -= 1;
                if depth == 0 {
                    return tokens
                        .get(cursor + 1)
                        .is_some_and(|next| next.text == "[")
                        .then_some(cursor);
                }
            }
            _ => {}
        }
        cursor = cursor.checked_sub(1)?;
    }
}

/// The index of the `(` or `[` that matches the `)` or `]` at `close`, walking backwards. Bounded by
/// the statement, like the angle hunt it serves: an unbalanced group is not a group.
fn matching_open_bracket(tokens: &[CodeToken], close: usize) -> Option<usize> {
    let (open, shut) = match tokens[close].text.as_str() {
        ")" => ("(", ")"),
        "]" => ("[", "]"),
        _ => return None,
    };
    let mut depth = 0usize;
    let mut cursor = close;
    loop {
        let text = tokens[cursor].text.as_str();
        if text == shut {
            depth += 1;
        } else if text == open {
            depth -= 1;
            if depth == 0 {
                return Some(cursor);
            }
        } else if text == ";" || text == "#" {
            return None;
        }
        cursor = cursor.checked_sub(1)?;
    }
}

/// Whether a template NAME stands in front of the `<` at `open`: an identifier, a `>`, a `::`,
/// the `template` keyword or a type keyword (every one of those is an identifier to
/// `is_identifier`). A `<` with none of them in front of it is a COMPARISON, and a comparison
/// neither opens a template group nor closes one. Round-18 A4 needed the test to tell
/// `0u < 1u > poll(...)` from `name<0u>`; round 32 A1 needs the SAME test in the two walks that
/// look for a lambda body, where counting `std::enable_if_t<1 < 2, int>` as an opener closed the
/// group the comparison sat in and refused a textbook nvcc compiles to the baseline. One
/// definition, three call sites: round 18 introduced it inside `matching_open_angle`, and round 32
/// put the two lambda walks that had their own angle handling onto the same definition.
fn a_template_name_stands_before(tokens: &[CodeToken], open: usize) -> bool {
    open.checked_sub(1).is_some_and(|before| {
        let text = tokens[before].text.as_str();
        is_identifier(text) || text == ">" || text == ":"
    })
}

/// The `<` that matches the `>` at `close`, or None when that `>` does not close a template
/// argument list. The tokenizer emits one token per punctuation character, so `>>` is two nested
/// `>` and a nested template id is counted correctly.
fn matching_open_angle(tokens: &[CodeToken], close: usize) -> Option<usize> {
    let mut depth = 0usize;
    let mut index = close;
    loop {
        match tokens[index].text.as_str() {
            ">" => depth += 1,
            "<" => {
                depth -= 1;
                if depth == 0 {
                    // A `<` opens a template argument list only when a template NAME stands before it
                    // (round-18 A4): the token in front of the `<` of `0u < 1u > poll(...)` is a
                    // numeric literal, so that `<` is a comparison and the `>` in front of the poll is
                    // not a template close. Reading it as one refused a call nvcc compiles to the same
                    // size as the shipped comparison.
                    return if a_template_name_stands_before(tokens, index) {
                        Some(index)
                    } else {
                        None
                    };
                }
            }
            // A balanced group is skipped rather than walked through (round-18 A4), so an earlier `<`
            // inside the poll's OWN statement - even inside parentheses, `(0u < 1u) > poll(...)` - is
            // not read as a template open. The statement bound below still ends the hunt.
            ")" | "]" => index = matching_open_bracket(tokens, index)?,
            // The hunt is bounded by the STATEMENT (round-17 A3): a `;`, a brace or a preprocessor
            // line ends it, because a template argument list cannot span one. Without the bound, the
            // walk reached an unmatched `<` anywhere earlier in the file - an include, a comparison in
            // another function - so the SAME comparison was a call in one source and a declaration in
            // another, and the documented claim "a comparison stays a call" was false.
            ";" | "{" | "}" | "#" => return None,
            _ => {}
        }
        if index == 0 {
            return None;
        }
        index -= 1;
    }
}

/// Whether the `)` at `close` closes a `decltype(...)` type: a type position in front of a name.
/// `sizeof(...)` is not a type and neither is a parenthesised expression, so those stay calls.
fn decltype_stands_before(tokens: &[CodeToken], close: usize) -> bool {
    let mut depth = 0usize;
    let mut index = close;
    loop {
        match tokens[index].text.as_str() {
            ")" => depth += 1,
            "(" => {
                depth -= 1;
                if depth == 0 {
                    return index > 0 && tokens[index - 1].text == "decltype";
                }
            }
            _ => {}
        }
        if index == 0 {
            return false;
        }
        index -= 1;
    }
}

/// The keywords that can stand where a type name cannot, immediately before a call: a declaration
/// does not start with any of them, and a statement very often does.
const STATEMENT_KEYWORDS: [&str; 32] = [
    "alignof",
    "case",
    "catch",
    "co_await",
    "co_return",
    "co_yield",
    "decltype",
    "delete",
    "do",
    "else",
    "false",
    "for",
    "goto",
    "if",
    "new",
    "noexcept",
    "nullptr",
    "operator",
    "requires",
    "return",
    "sizeof",
    "static_assert",
    "switch",
    "template",
    "this",
    "throw",
    "true",
    "try",
    "typename",
    "using",
    "while",
    "namespace",
];

/// The index of the bracket that closes the one at `open`, or None when the tokens are unbalanced.
/// The three bracket kinds share one counter, which is enough for a scanner that only needs the end
/// of a call's argument list.
fn matching_close_bracket(tokens: &[CodeToken], open: usize) -> Option<usize> {
    let mut depth = 0usize;
    for (index, token) in tokens.iter().enumerate().skip(open) {
        match token.text.as_str() {
            "(" | "[" | "{" => depth += 1,
            ")" | "]" | "}" => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(index);
                }
            }
            _ => {}
        }
    }
    None
}

/// The callee designator and the argument list of the call whose name is at `index`: the index of
/// the designator's first token and the index of the `(` that opens the arguments. None when the
/// name is not called. This is the one place the designator spelling is unwrapped, so a check that
/// needs the whole call (the bare-call rule below, the statement-start rule in the reachability
/// lint, the structural helper check) cannot drift from [`designator_is_a_call`].
///
/// The designator may be dereferenced or have its address taken inside any nesting of parentheses:
/// `(*f)(...)`, `(&f)(...)`, `(*(&f))(...)` and `(&(*f))(...)` all call `f` through that
/// pointer, and nvcc compiles them all to the same call. The round-8 review compiled
/// `(*(&__threadfence))();` to the baseline cubin byte for byte while both checks rejected it, so
/// the unwrap walks the whole wrapper rather than one step.
fn call_argument_list(tokens: &[CodeToken], index: usize) -> Option<(usize, usize)> {
    // Forward: the wrapper's closing parentheses, then the argument list's `(`.
    let mut cursor = index + 1;
    let mut closes = 0usize;
    while tokens.get(cursor).is_some_and(|next| next.text == ")") {
        closes += 1;
        cursor += 1;
    }
    if !tokens.get(cursor).is_some_and(|next| next.text == "(") {
        return None;
    }
    let open = cursor;
    // Backward: the wrapper's parentheses, `*` and `&`, in any order and nesting.
    let mut needed = closes;
    let mut start = index;
    while needed > 0 {
        let previous = start.checked_sub(1)?;
        match tokens[previous].text.as_str() {
            "(" => needed -= 1,
            ")" => needed += 1,
            "*" | "&" => {}
            _ => return None,
        }
        start = previous;
    }
    Some((start, open))
}

/// Whether the name at `index` is a call with an empty argument list that ends the statement, in
/// every designator spelling: `f();`, `(f)();`, `((f))();`, `(*f)();`. This is the shape of the
/// fence itself - `__threadfence` takes no arguments - and the round-7 review found that both places
/// that used to spell it out as `name ( ) ;` rejected `(__threadfence)();`, which nvcc accepts and
/// compiles to the same PTX membar as the bare spelling. Rejecting valid C++ makes the guard a hazard
/// to the developer debugging it, so the spelling is accepted rather than documented as a limit.
fn is_bare_fence_call(tokens: &[CodeToken], index: usize) -> bool {
    let Some((_, open)) = call_argument_list(tokens, index) else {
        return false;
    };
    tokens.get(open + 1).is_some_and(|next| next.text == ")")
        && tokens.get(open + 2).is_some_and(|next| next.text == ";")
}

/// The index of the `__threadfence` name of a designator that starts at `index`, or None. The `&`
/// or `*` that takes the address of or dereferences the callee is PART OF THE DESIGNATOR, so
/// `(&__threadfence)();` and `(*__threadfence)();` are ordinary calls and not the short-circuit or
/// bitwise operator the control list rejects: both checks look past the designator instead of
/// refusing valid C++.
///
/// The wrapper is walked as a whole, not one step at a time, so `(*(&__threadfence))();` and
/// `(&(*__threadfence))();` are looked past as well. A genuine operator is still an operator: the
/// run stops at any other token, and a call the run reaches but that does not start its statement is
/// rejected by the statement-start rule (`x & (__threadfence)()` stays rejected).
fn fence_designator_starting_at(tokens: &[CodeToken], index: usize) -> Option<usize> {
    let mut cursor = index;
    while tokens
        .get(cursor)
        .is_some_and(|token| matches!(token.text.as_str(), "(" | "*" | "&"))
    {
        cursor += 1;
    }
    tokens
        .get(cursor)
        .is_some_and(|token| token.text == "__threadfence")
        .then_some(cursor)
}

/// The sentence every shape rejection ends with. The guard is a canonical-form checker rather than a
/// proof (see the module documentation): when it does not recognise a spelling it says so and names
/// the form to write, so a rejection of an exotic but valid spelling is a policy decision the
/// developer resolves by writing the canonical spelling instead of a bug to be worked around.
const CANONICAL_HINT: &str = concat!(
    "write the acquire the canonical way - a bare __threadfence(); statement after the poll - and ",
    "see the acquire-guard documentation for the complete list of spellings this check accepts"
);

/// The one constructor of a rejection. Every rejection this module builds goes through it, so the
/// canonical hint is attached by construction instead of by remembering: it is appended when the
/// message does not already end with it, and a debug build refuses a message that carries the hint
/// anywhere but at its end.
///
/// Round-14 A-2: the source scan beside the tests used to search an `Err(...)` span for the hint,
/// and four physical mutations escaped it (a rejection built by `ok_or_else`, a helper whose
/// if-branch carried the hint and whose else-branch did not, a hint followed by more message, and a
/// hint buried inside an if/else). With one constructor the hint cannot be forgotten by
/// construction, and the scan only has to check that every rejection site calls this function.
fn rejection(message: impl Into<String>) -> String {
    let message = message.into();
    debug_assert!(
        !message.contains(CANONICAL_HINT) || message.ends_with(CANONICAL_HINT),
        "a rejection message must not carry the canonical hint anywhere but at its end"
    );
    if message.ends_with(CANONICAL_HINT) {
        return message;
    }
    format!("{message}; {CANONICAL_HINT}")
}

/// What a using-declaration or using-directive names in front of one of the guarded names. The
/// reason a rejection is honest depends on which one it is, so the message is built from this rather
/// than from a bare `Option<namespace>` (round-11 A5, corrected in round-12 A5: the bare
/// `using __threadfence;` was claimed to still name the barrier, and nvcc refuses it).
#[derive(Debug, PartialEq, Eq)]
enum UsingTarget {
    /// `using X::__threadfence;` names a member of X, so an unqualified call can resolve to X's
    /// declaration and the meaning really does change. Measured: nvcc compiled that helper to 70264
    /// bytes / 2 PTB membar lines / sha 42454ad1... against the 70432 / 4 / 605321041ac5ec1e baseline
    /// (round-10 A-2).
    Member(String),
    /// `using namespace X;` where a scanned source defines X with a member carrying a guarded name:
    /// the same re-binding, reached indirectly. Measured: nvcc refuses the ambiguous call the
    /// directive produces ("more than one instance of overloaded function __threadfence matches").
    Directive(String),
    /// `using ::__threadfence;` names the global barrier itself, so the name still denotes the
    /// barrier and nvcc compiles that body to the baseline cubin byte for byte. This one is a POLICY
    /// rejection: a text scanner does not resolve using-declarations.
    Global,
    /// `using __threadfence;` is not namespace-qualified, so it is not valid C++ here at all: nvcc
    /// refuses the translation unit with "namespace-qualified name is required" (round-12 A5). The
    /// shape cannot ship, so refusing it costs nothing, and the message says which spelling it was.
    Unqualified,
    /// `using X::template __threadfence;` names a function template rather than a function: nvcc
    /// refuses it with "no instance of function template X::__threadfence matches the argument list"
    /// (measured). The spelling is not a re-binding, so the message does not claim it is one.
    TemplateQualified(String),
    /// `using X::__threadfence<int>;` is a template-id, which nvcc refuses with "a template-id is not
    /// allowed" (measured). Like the template-qualified form it is a malformed using-declaration
    /// rather than a re-binding, so the message says that instead (round-13 A-3).
    TemplateId,
    /// `using __threadfence = X;` is an alias-declaration, not a using-declaration, and it RE-BINDS
    /// the unqualified name in its scope: nvcc refuses it when the right-hand side is not a type
    /// ("namespace X is not a type name", measured) and compiles the re-binding when it is, where the
    /// barrier is gone - 70264 bytes / 2 PTB membar lines / sha 42454ad1..., the same cubin as the
    /// `using X::__threadfence;` member re-bind, against the 70432 / 4 / 605321041ac5ec1e baseline
    /// (measured, round-13 A-3).
    Alias,
}

/// The rejection a using-declaration or using-directive earns, in one place so the three checks say
/// the same thing. Every case ends with the canonical hint, like every other rejection of a shape
/// this scan cannot certify.
fn using_declaration_rejection(position: &str, name: &str, target: &UsingTarget) -> String {
    match target {
        UsingTarget::Member(namespace) => format!(
            "{position}: a using-declaration that brings {namespace}'s {name} into scope changes what the unqualified {name} can denote, so this scan refuses the shape rather than certify it"
        ),
        UsingTarget::Directive(namespace) => format!(
            "{position}: a using-directive for {namespace}, which defines its own {name}, changes what the unqualified {name} can denote, so this scan refuses the shape rather than certify it"
        ),
        UsingTarget::Global => format!(
            "{position}: `using ::{name};` still names the barrier itself and nvcc compiles that body to the baseline cubin, but this scan does not resolve using-declarations, so it refuses the shape rather than certify it. This is a policy rejection, not a claim that {name} changed meaning"
        ),
        UsingTarget::Unqualified => format!(
            "{position}: `using {name};` is not a namespace-qualified using-declaration, so nvcc refuses the translation unit with \"namespace-qualified name is required\" and this scan refuses the shape too. This is a policy rejection, not a claim that {name} changed meaning"
        ),
        UsingTarget::TemplateQualified(namespace) => format!(
            "{position}: `using {namespace}::template {name};` names a function template rather than the barrier, and nvcc refuses it with \"no instance of function template {namespace}::{name} matches the argument list\" (measured). This is a policy rejection, not a claim that {name} changed meaning"
        ),
        UsingTarget::TemplateId => format!(
            "{position}: `using X::{name}<...>;` is a template-id, which nvcc refuses with \"a template-id is not allowed\" (measured). This is a policy rejection, not a claim that {name} changed meaning"
        ),
        UsingTarget::Alias => format!(
            "{position}: `using {name} = ...;` is an alias-declaration that re-binds the unqualified {name} in its scope: nvcc refuses it when the right-hand side is not a type (\"namespace X is not a type name\") and compiles the re-binding when it is, which loses the barrier - 70264 bytes / 2 PTB membar lines / sha 42454ad1... against the 70432 / 4 / 605321041ac5ec1e baseline (measured). This scan refuses the shape rather than certify it"
        ),
    }
}

/// Whether the text is a C++ identifier, which is what qualifies a designator as a member of a
/// namespace rather than the global name.
fn is_identifier(text: &str) -> bool {
    let mut characters = text.chars();
    characters
        .next()
        .is_some_and(|first| first.is_ascii_alphabetic() || first == '_')
        && characters.all(|character| character.is_ascii_alphanumeric() || character == '_')
}

/// The declaration specifiers that make a name a declaration rather than a call. `void abort();`
/// declares a name; `(void)abort()`, `return abort()` and `f(abort())` call one. The difference
/// is the token immediately before the name (round-10 A-4).
const DECLARATION_SPECIFIERS: [&str; 11] = [
    "void", "bool", "char", "short", "int", "long", "float", "double", "signed", "unsigned", "auto",
];

/// Whether the designator at `index` is qualified, so it names a member or another scope's function
/// rather than the library function this guard knows: `s.abort(1u)`, `x->abort()`, `X::_Exit(1)`
/// and `std::abort()` are not the non-returning function, and treating them as it was a false
/// rejection (round-10 A-4). A global `::abort` IS the global function and does count, which is why
/// a doubled colon only qualifies when an identifier or a template's `>` stands in front of it.
fn designator_is_qualified(tokens: &[CodeToken], index: usize) -> bool {
    match tokens
        .get(index.wrapping_sub(1))
        .map(|token| token.text.as_str())
    {
        Some(".") => true,
        Some(">") => tokens
            .get(index.wrapping_sub(2))
            .is_some_and(|token| token.text == "-"),
        Some(":") => {
            tokens
                .get(index.wrapping_sub(2))
                .is_some_and(|token| token.text == ":")
                && tokens
                    .get(index.wrapping_sub(3))
                    .is_some_and(|token| is_identifier(&token.text) || token.text == ">")
        }
        _ => false,
    }
}

/// Whether the name at `index` is the library non-returning function this guard knows, rather than
/// one of the four shapes that only look like it (round-10 A-4).
///
/// `s.abort(1u)` is a member function, `X::_Exit(1)` and `std::abort()` are other scopes',
/// `void abort();` is a declaration and a name this source `#define`s is a macro use, and none of
/// them is the function that never returns. The qualified case the scanner cannot resolve is a stated
/// policy: the documentation lists it, the compiled cubin baseline is what settles it, and the guard
/// is a text scanner rather than a name resolver, so `std::abort()` is deliberately not a blocker -
/// and that costs nothing, because nvcc refuses a host `abort` call in device code ("calling a
/// __host__ function abort from a __device__ function is not allowed"). The conservative reading was
/// refusing real code (a declaration, a member, another scope's function) while the permissive one
/// cannot ship a kernel without its fence.
fn is_the_library_non_returning_call(
    tokens: &[CodeToken],
    index: usize,
    macros: &MacroScope,
) -> bool {
    if index == 0 {
        return designator_is_a_call(tokens, index);
    }
    !designator_is_qualified(tokens, index)
        && !DECLARATION_SPECIFIERS.contains(&tokens[index - 1].text.as_str())
        && !macros.is_defined(&tokens[index].text)
        && designator_is_a_call(tokens, index)
}

/// Whether the brace that continues the statement in `history` opens a lambda's or a type's body
/// rather than a bare block.
///
/// A bare block runs unconditionally like the statements around it; a lambda's body does not run until
/// the closure is called, and a class, struct, namespace, enum or union body is not on the helper's
/// path at all. The round-10 reviewer compiled `auto l = [] { __threadfence(); }; (void)l;` into
/// 70264 bytes with 2 PTB membar lines (sha 42454ad1...), one barrier short of the 70432 / 4 /
/// 605321041ac5ec1e baseline, while the called version was byte-identical to it - so the fence in the closure
/// is a nested, conditional fence, not the acquire.
///
/// A `]` anywhere in the statement catches every lambda spelling, including `[&] { ... }` and
/// `[]() -> int { ... }`, and a type keyword catches the bodies whose last token before the brace is
/// the type's name rather than a keyword. A statement that reaches neither is the bare block the
/// round-8 review asked for.
fn opens_a_lambda_or_a_type_body(tokens: &[CodeToken], history: &[usize]) -> bool {
    let mut cursor = history.len();
    while cursor > 0 {
        cursor -= 1;
        match tokens[history[cursor]].text.as_str() {
            ";" | "{" | "}" => return false,
            "]" => {
                // An ATTRIBUTE in front of the brace ends in the same ] a capture list does, and it
                // introduces no body: `[[likely]] { __threadfence(); }` runs its fence exactly where
                // the same block without the attribute does, so refusing it because the ] looks like
                // a capture list is the compile-to-the-baseline direction this project calls the
                // worse error (round-35 A3). Step over the whole attribute group and keep looking;
                // anything else that ends in ] is still a capture list or a subscript, which is the
                // lambda body this test exists to find.
                if let Some(opener) = an_attribute_closes_at(tokens, history[cursor]) {
                    while cursor > 0 && history[cursor - 1] >= opener {
                        cursor -= 1;
                    }
                    continue;
                }
                return true;
            }
            "class" | "struct" | "namespace" | "enum" | "union" => return true,
            _ => {}
        }
    }
    false
}

/// Whether `index` sits inside a body the HELPER does not own, at any depth: a LAMBDA's body
/// (`[&]{ ... }` or `[&]<typename T>() -> T { ... }`, round 27 A2, round 28 A2) or a TYPE body
/// (`struct`, `class`, `union` - round 31 A2).
///
/// A `return` inside a lambda exits the LAMBDA, and a `return` inside a member function of a local
/// type exits that MEMBER. Neither is the helper's path - the helper reaches its acquire either way -
/// so every reading that looks for an exit standing on the helper's path skips both. nvcc compiles
/// `struct S { void run() { return; } };` before the poll to the baseline 70432 / 4 / 605321041ac5ec1e with
/// the acquire present, and round 31 A2 found that the class-body spelling was refused with the
/// unreachability sentence, which was false of it.
///
/// EVERY enclosing brace is examined rather than only the innermost one: an exit nested in a block,
/// an `if` or a loop inside the lambda still exits the lambda, and stopping at the innermost brace
/// refused those bodies with "the acquire after it is unreachable" while nvcc compiled them to the
/// baseline (round 28 A2). The walk stops at `floor`, the opening brace of the block the caller is
/// reading - the helper's own body in all three readings - so a lambda or a type defined outside the
/// helper cannot make this true, which is what keeps a bare block at statement level refused (it
/// compiles to 70464 / 2 / 2b56c510b5f4a945, and the reason it is refused with is then true).
///
/// Round 32 A3: the excuse is that the exit leaves the LAMBDA and the helper still reaches its
/// acquire. For a lambda the source does not call that holds, because the body never runs; for a
/// lambda the source CALLS right there it only holds if the body can RETURN.
/// `(void)([&]{ __trap(); return 0u; })();` before the poll is the counter-example: the trap ends
/// the helper, the acquire after it never runs, and nvcc compiles the text to
/// 69536 / 4 / 5826994128b34ed0 (37872 / 4 / 54cda5bc pre-change) - the cubin this file already
/// refuses as `__trap();` before the poll, so
/// the module's one-cubin-one-verdict rule applies: a CALLED lambda body that holds a
/// non-returning call of its own is not a body the helper does not own. A `return` does not make
/// the body non-returning - it returns to the caller - so `(void)([&]{ return; })();` stays
/// accepted, and a lambda nobody calls (`auto f = [&]{ __trap(); }; (void)f;`) stays accepted
/// too, which is what its baseline cubin measures.
fn inside_a_body_the_helper_does_not_own(tokens: &[CodeToken], index: usize, floor: usize) -> bool {
    let mut cursor = index;
    while cursor > floor {
        let Some(open) = the_enclosing_brace(tokens, cursor, floor) else {
            return false;
        };
        // Only the precise detectors, not the walk's looser one: an attribute's brackets would
        // make the looser test true for the whole helper.
        if a_lambda_body_opens_at(tokens, open) {
            return !(a_lambda_body_is_called_at(tokens, open)
                && a_called_lambda_body_cannot_return(tokens, open));
        }
        if a_type_body_opens_at(tokens, open) {
            return true;
        }
        // Neither: keep walking outward, because this brace may be a nested block INSIDE a lambda
        // or a type whose opener is further out.
        cursor = open;
    }
    false
}

/// Whether the lambda whose body starts at `open` is CALLED where it stands: `[&]{ ... }()` or
/// `([&]{ ... })()`. The tokenizer emits one token per punctuation character, so the wrap-around
/// form is `}` `)` `(`. Round 32 A3.
fn a_lambda_body_is_called_at(tokens: &[CodeToken], open: usize) -> bool {
    let Some(close) = matching_close_bracket(tokens, open) else {
        return false;
    };
    match tokens.get(close + 1).map(|token| token.text.as_str()) {
        Some("(") => true,
        Some(")") => tokens.get(close + 2).map(|token| token.text.as_str()) == Some("("),
        _ => false,
    }
}

/// Whether the body of a CALLED lambda at `open` cannot return: it calls one of the
/// `NON_RETURNING_CALLS` names on a path that runs when the body runs, so control never comes back
/// to the statement the call stands in and everything after that statement is unreachable (round
/// 32 A3). The same name test the two walks use decides it, so a declaration, a member, another
/// scope's function or a name this source `#define`s is not a non-returning call here either. A
/// trap reached through a macro, a function pointer or an inline-asm block is the same stated limit
/// the name list carries.
///
/// The scan descends through BARE BLOCKS and stops at every other brace: a lambda's or a type's
/// body, whose calls belong to another function, and a control statement's body, whose statements
/// run only if the statement does. `(void)([&]{ { __trap(); } return 0u; })();` before the poll is
/// the round-33 A1 counter-example to the narrower rule round 32 shipped: the brace is a bare
/// block, the trap runs exactly as it does unbraced, and the text is the same program as the
/// unbraced spelling this file already refuses as 69536 / 4 / 5826994128b34ed0 (37872 / 4 /
/// 54cda5bc pre-change). A trap behind a control
/// statement - `[&]{ if (c) { __trap(); } return 0u; }()` - is still not counted, because it is
/// conditional on `c` and the acquire after it still runs.
fn a_called_lambda_body_cannot_return(tokens: &[CodeToken], open: usize) -> bool {
    let Some(close) = matching_close_bracket(tokens, open) else {
        return false;
    };
    let mut macros = MacroScope::from_tokens(tokens);
    // One frame per open brace, innermost last: true where the brace opens a scope whose statements
    // this scan must not read as the lambda's own. `opaque` counts the true ones still open.
    let mut frames: Vec<bool> = Vec::new();
    let mut opaque = 0usize;
    let mut inner = open + 1;
    while inner < close {
        macros.advance_to(inner);
        let text = tokens[inner].text.as_str();
        match text {
            "{" => {
                let hides = !a_bare_block_opens_at(tokens, inner);
                if hides {
                    opaque += 1;
                }
                frames.push(hides);
            }
            "}" => {
                if frames.pop() == Some(true) {
                    opaque -= 1;
                }
            }
            _ => {
                if opaque == 0
                    && NON_RETURNING_CALLS.contains(&text)
                    && is_the_library_non_returning_call(tokens, inner, &macros)
                {
                    return true;
                }
            }
        }
        inner += 1;
    }
    false
}

/// Whether the `{` at `index` opens a BARE BLOCK: a scope that runs where it stands, so the
/// statements in it are on the path the code around them is on (round 33 A1).
///
/// This is the difference the round-32 fix turned on and got half right. `{ __trap(); }` inside an
/// immediately-called lambda runs in the same order as `__trap();` there - the brace adds no
/// condition and no function - so a scan that stops at the brace certifies a text that is the same
/// program as one it already refuses. Three kinds of brace are therefore NOT bare blocks:
///
///   * a lambda's or a type's body, which the precise detectors already identify, because a call in
///     there belongs to another function that may never run;
///   * a control statement's body, whose brace follows `)`, `else`, `do`, `try` or `requires`,
///     because a trap in there is conditional and refusing a conditional trap would refuse a text
///     nvcc compiles to the baseline - the direction this project treats as the worse error;
///   * a lambda's or a type's body behind a capture list or behind attributes, which is what the `]
///     case in those detectors covers - note that the attributes are stepped over first, so an
///     attribute standing on a bare block does not hide that block.
///
/// Everything else is a bare block. The rule is deliberately the narrow one: it descends only where
/// nothing can keep the statements from running, so it closes the false accept without opening a
/// false refusal. The limit that buys is stated rather than hidden: the rule reads braces, not
/// values, so `(void)([&]{ if (1) { __trap(); } return 0u; })();` before the poll stays accepted
/// even though the condition is constantly true, because deciding it needs the condition's value.
/// Round-32's narrower rule was the same kind of limit - it just put the boundary at the wrong
/// brace, refusing nothing and accepting a text whose cubin the file already refuses.
fn a_bare_block_opens_at(tokens: &[CodeToken], index: usize) -> bool {
    if a_lambda_body_opens_at(tokens, index) || a_type_body_opens_at(tokens, index) {
        return false;
    }
    // An ATTRIBUTE may stand between a control keyword and the brace of its body, which C++ allows on
    // a selection or iteration statement, as in: if (deadline) [[likely]] { ... }. The brace then
    // follows a ], which is not a control token, so an attribute-blind test read the body as a bare
    // block and refused a conditional trap that nvcc compiles to the baseline - the worse direction,
    // and the one this rule exists to avoid (round-34 A1). Step over every attribute standing there
    // and test the token before the first of them.
    !matches!(
        the_token_before_the_attributes(tokens, index.checked_sub(1))
            .map(|before| tokens[before].text.as_str()),
        Some(")" | "else" | "do" | "try" | "requires")
    )
}

/// The index of the `{` that opens the innermost block holding `index`, or `None` when the walk
/// reaches `floor` first - the caller's own block, which is the helper's body in every reading.
fn the_enclosing_brace(tokens: &[CodeToken], index: usize, floor: usize) -> Option<usize> {
    let mut depth = 0usize;
    let mut cursor = index;
    while cursor > floor {
        cursor -= 1;
        match tokens[cursor].text.as_str() {
            "}" => depth += 1,
            "{" => {
                if depth == 0 {
                    return Some(cursor);
                }
                depth -= 1;
            }
            _ => {}
        }
    }
    None
}

/// Whether the `{` at `index` opens a TYPE body, `struct S { ... }`, `class X : public A, public B
/// { ... }` or `union U { ... }`. The walk back stops at the first token that ends a declaration or
/// starts another construct, so an ordinary function body, a lambda body, a namespace and a control
/// statement all answer false (round 31 A2).
///
/// Round 32 A2: a balanced bracket group is SKIPPED instead of ending the walk, and a `,` no
/// longer ends it. A base clause (`struct S : A, B`, `class S : public A, public B`) and an
/// attribute specifier (`struct [[maybe_unused]] S`, `struct __attribute__((packed)) S`) both
/// stand between the keyword and the brace in declarations nvcc compiles to the baseline, and
/// stopping at their `,`, `]` or `)` refused those type bodies with the unreachability sentence.
/// The skip is bounded: `matching_open_bracket` answers `None` for an unbalanced group, so a
/// stray closer still ends the walk.
fn a_type_body_opens_at(tokens: &[CodeToken], index: usize) -> bool {
    if tokens.get(index).map(|token| token.text.as_str()) != Some("{") {
        return false;
    }
    let mut cursor = index;
    let mut looked = 0usize;
    while cursor > 0 && looked < 48 {
        cursor -= 1;
        looked += 1;
        match tokens[cursor].text.as_str() {
            "struct" | "class" | "union" => return true,
            ")" | "]" => match matching_open_bracket(tokens, cursor) {
                Some(opener) => cursor = opener,
                None => return false,
            },
            // A declaration, a body, an assignment, a namespace or a control statement:
            // everything before this point belongs to another construct.
            ";" | "{" | "}" | "=" | "namespace" | "enum" | "if" | "else" | "for" | "while"
            | "switch" | "do" | "try" | "catch" | "return" => return false,
            _ => {}
        }
    }
    false
}

/// Whether the operator at `index` is written inside a LAMBDA CAPTURE LIST, `[&]` or `[=]` or
/// `[&x]`, and therefore is not a path-bounding operator at all (round 26 A4). The capture list is
/// the one `[` ... `]` pair directly before a lambda's parameter list or body, so the name after
/// the closing bracket decides it.
fn inside_a_lambda_capture(tokens: &[CodeToken], index: usize) -> bool {
    // `[&]`, `[=]`, `[&, x]`: the operator sits between the capture's brackets, which is the test.
    if tokens
        .get(index.wrapping_sub(1))
        .is_some_and(|token| token.text == "[")
        && tokens
            .get(index + 1)
            .is_some_and(|token| token.text == "]" || is_identifier(&token.text))
    {
        return true;
    }
    let mut cursor = index;
    while cursor > 0 {
        cursor -= 1;
        match tokens[cursor].text.as_str() {
            ";" | "{" | "}" => return false,
            "[" => {
                let Some(close) = matching_close_bracket(tokens, cursor) else {
                    return false;
                };
                if close < index {
                    return false;
                }
                return matches!(
                    tokens.get(close + 1).map(|token| token.text.as_str()),
                    Some("(" | "{" | "mutable" | "noexcept" | "->")
                ) || tokens
                    .get(close + 1)
                    .is_some_and(|token| token.text == "->");
            }
            _ => {}
        }
    }
    false
}

/// Where the statement that starts with the designator begins, or None when the name is not the
/// global barrier at all.
///
/// `::__threadfence` is the global function, so its leading `::` - two `:` tokens to this
/// tokenizer - belongs to the designator and the statement starts before it. `X::__threadfence()`
/// and `x.__threadfence()` name a namespace's or an object's function, which is not this barrier:
/// reading the `::` as a label colon made a labelled statement out of `X::__threadfence();`, and
/// with a namespace member named `__threadfence` nvcc compiled that no-op, dropped two barriers and
/// the check stayed green (round-8 A3). For a qualified name the truthful verdict is that this body
/// contains no barrier, which is what None means to the caller.
fn global_fence_designator(
    tokens: &[CodeToken],
    visited: &[usize],
    designator: usize,
) -> Option<usize> {
    let Some(previous) = code_token_before(visited, designator) else {
        return Some(designator);
    };
    match tokens[previous].text.as_str() {
        // `x.__threadfence` and `x->__threadfence` name another object's function.
        "." => None,
        ">" if code_token_before(visited, previous)
            .is_some_and(|before| tokens[before].text == "-") =>
        {
            None
        }
        ":" => {
            // One colon before the name is a LABEL (`obs: __threadfence();`), which the
            // statement-boundary rule below understands; two are `::`, a scope qualifier.
            let Some(first) = code_token_before(visited, previous) else {
                return Some(designator);
            };
            if tokens[first].text != ":" {
                return Some(designator);
            }
            match code_token_before(visited, first) {
                // A name, a template's `>` or a `)` before the `::` qualifies the designator:
                // only `::__threadfence` with nothing before it is the global one.
                Some(before)
                    if tokens[before].text == ")"
                        || tokens[before].text == ">"
                        || is_identifier(&tokens[before].text) =>
                {
                    None
                }
                _ => Some(first),
            }
        }
        _ => Some(designator),
    }
}

/// The last code token visited before `start`, or None at the start of the block.
fn code_token_before(visited: &[usize], start: usize) -> Option<usize> {
    visited.iter().rev().copied().find(|&code| code < start)
}

/// Whether the colon at `colon` ends a label rather than the false branch of a conditional
/// expression. The two are told apart by the `?` a conditional expression must have at the same
/// parenthesis depth: `ready ? a : b` always has one before its colon, a label never does. Without
/// this, accepting `obs: __threadfence();` would also accept the fence as the false branch of a
/// conditional, which is exactly the conditional acquire the check exists to reject.
///
/// The search stops at the statement's start, because a conditional expression cannot span a `;` or a
/// brace: without that bound a `?` in an earlier statement would make a later label look like a
/// conditional expression. A colon passed on the way consumes one `?`, so `a ? b : c ? d : e` reads
/// correctly from either of its colons.
///
/// A colon that is part of `::` is not a label either: `X::__threadfence();` is a qualified call,
/// and reading its `::` as a label colon made the statement look like a labelled one (round-8 A3).
fn colon_is_a_label(tokens: &[CodeToken], visited: &[usize], colon: usize) -> bool {
    if code_token_before(visited, colon).is_some_and(|before| tokens[before].text == ":")
        || visited
            .iter()
            .copied()
            .find(|&code| code > colon)
            .is_some_and(|after| tokens[after].text == ":")
    {
        return false;
    }
    let mut depth = 0i32;
    let mut colons = 0i32;
    let mut cursor = colon;
    while let Some(previous) = code_token_before(visited, cursor) {
        match tokens[previous].text.as_str() {
            ")" => depth += 1,
            "(" => depth -= 1,
            ";" | "{" | "}" if depth == 0 => return true,
            ":" if depth == 0 => colons += 1,
            "?" if depth == 0 => {
                if colons == 0 {
                    return false;
                }
                colons -= 1;
            }
            _ => {}
        }
        cursor = previous;
    }
    true
}

/// Whether the code token at `start` starts a statement, given the code tokens visited before it.
/// `start` is the first token of a whole designator, so `(__threadfence)();` is decided by what
/// precedes its `(`, not by the `(` itself (round 7).
///
/// A labelled statement starts where its label starts, so `obs: __threadfence();` is accepted: the
/// label does not join the fence to an expression, and nvcc compiles the call unconditionally. The
/// reachability lint and the structural check must agree about that, so both call this function.
/// `case 1: __threadfence();` reaches the statement boundary too, and is then rejected by the
/// `case` token as the control it is.
fn starts_a_statement(tokens: &[CodeToken], visited: &[usize], start: usize) -> bool {
    let Some(mut previous) = code_token_before(visited, start) else {
        return true;
    };
    // An attribute-specifier is not an expression: `[[likely]] __threadfence();`, with the attribute
    // on the same line or the line above, is still the fence's own statement, and nvcc compiles that
    // helper to the baseline cubin byte for byte (round-12 A4). The whole `[[...]]` is walked over
    // before the boundary is decided, and both checks call this function, so they agree about it.
    while tokens[previous].text == "]" {
        let Some(first) = attribute_specifier_start(tokens, visited, previous) else {
            break;
        };
        let Some(before) = code_token_before(visited, first) else {
            return true;
        };
        previous = before;
    }
    match tokens[previous].text.as_str() {
        ";" | "{" | "}" => true,
        ":" if colon_is_a_label(tokens, visited, previous) => {
            // Walk back over the label's own name (and a `case`/`default` keyword, with the
            // constant of a `case`) to the boundary the whole label starts at.
            let mut cursor = previous;
            while let Some(name) = code_token_before(visited, cursor) {
                match tokens[name].text.as_str() {
                    ";" | "{" | "}" | "case" | "default" => return true,
                    _ => cursor = name,
                }
            }
            true
        }
        _ => false,
    }
}

/// The index of the opening `[` of the `[[...]]` attribute-specifier whose closing `]` is at `close`,
/// or None when those brackets are not an attribute.
///
/// An attribute is a double bracket, which is what tells `[[likely]] __threadfence();` from an array
/// subscript or a lambda capture: a single `[` is neither an attribute nor an expression the fence
/// can be part of, so this returns None for it and the statement-start rule keeps its usual answer.
fn attribute_specifier_start(
    tokens: &[CodeToken],
    visited: &[usize],
    close: usize,
) -> Option<usize> {
    let mut depth = 0usize;
    let mut cursor = close;
    loop {
        match tokens[cursor].text.as_str() {
            "]" => depth += 1,
            "[" => {
                depth -= 1;
                if depth == 0 {
                    // The opening brackets are adjacent in the token stream: that pair is what makes
                    // this an attribute-specifier rather than `a[i]` or a lambda capture.
                    return tokens
                        .get(cursor + 1)
                        .is_some_and(|next| next.text == "[")
                        .then_some(cursor);
                }
            }
            _ => {}
        }
        cursor = code_token_before(visited, cursor)?;
    }
}

/// The range of a function declarator's parameter list: the indices of its opening `(` and matching
/// `)`, for the name at `index`. None when the name is not directly followed by a parameter list,
/// which is what a call through a parenthesised or dereferenced designator looks like.
fn parameter_list(tokens: &[CodeToken], index: usize) -> Option<(usize, usize)> {
    if tokens.get(index + 1).map(|token| token.text.as_str()) != Some("(") {
        return None;
    }
    let open = index + 1;
    let mut cursor = open;
    let mut depth = 0usize;
    while cursor < tokens.len() {
        match tokens[cursor].text.as_str() {
            "(" => depth += 1,
            ")" => {
                depth -= 1;
                if depth == 0 {
                    return Some((open, cursor));
                }
            }
            _ => {}
        }
        cursor += 1;
    }
    None
}

/// Whether the name at `index` sits where a callee designator can sit, which is what separates a call
/// of the acquiring poll from a declaration or a definition of the same name. A declarator is
/// preceded by its return type (`uint32_t`, `void`, `*`); a call is preceded by an operator, a
/// separator, a keyword that starts an expression, another call's parenthesis, or nothing.
fn looks_like_a_callee(tokens: &[CodeToken], index: usize) -> bool {
    match index
        .checked_sub(1)
        .and_then(|previous| tokens.get(previous))
    {
        None => true,
        Some(previous) => matches!(
            previous.text.as_str(),
            "=" | "("
                | ","
                | ";"
                | "{"
                | "}"
                | "return"
                | "?"
                | ":"
                | "&&"
                | "||"
                | "!"
                | ")"
                | "+"
                | "-"
        ),
    }
}

/// One macro definition of a source: its name, where it takes effect, and what its replacement list
/// says.
struct MacroDefinition {
    name: String,
    /// The token index of the `#` that starts the definition. A use before this position is not
    /// affected by it, because the preprocessor is positional.
    position: usize,
    /// The identifiers the replacement list mentions, which is where a chain of names shows up.
    mentions: Vec<String>,
    /// The replacement list's tokens, so a check can ask whether the body IS a name
    /// (`#define POLL_ALIAS spark_wait_for_flag`, round-17 A4) rather than merely mentioning it.
    body: Vec<String>,
    /// Whether the replacement list can end the fall-through path or never return: it carries a control
    /// KEYWORD (`#define BAIL return`) or a non-returning call. Operators are NOT here (round-20 A1),
    /// because a body that carries only an operator does not bound the path by itself.
    bounds_the_path: bool,
    /// Whether the replacement list carries an OPERATOR of the control list (`&&`, `||`, `&`, `|`, `?`)
    /// and no control keyword, so a USE of the macro is a blocker under the same completed-statement
    /// scoping as the literal operator (round-20 A1).
    carries_an_operator: bool,
}

/// The macros a source defines, in definition order, with the ones whose use cannot be trusted kept
/// up to date as a scan advances.
///
/// The preprocessor is positional, and the round-10 review was right to say so: in `int FLAG; ...
/// (void)FLAG; __threadfence(); #define FLAG return` the definition follows the use, so it does not
/// put a `return` in front of the fence; treating every definition as if it stood at the top of the
/// file rejected real code (round-10 A-5). [`MacroScope::advance_to`] therefore publishes a
/// definition only once a scan has passed its position, and the checks ask it about the token they are
/// looking at.
///
/// Suspicion still propagates through the names a replacement list mentions to a fixed point, exactly
/// as before - `#define OUTER INNER` beside `#define INNER if (observed != 0u)` is the round-8 shape,
/// which nvcc compiled to 38624 bytes with 3 PTX membar lines (sha e5b9d6...) (pre-change) while the baseline is
/// 70432 / 4 / 605321041ac5ec1e - but the reverse edges are indexed by name and each one is walked once. The
/// previous implementation re-scanned every body each time it learned a name, so a chain of N
/// definitions cost O(N^2) body scans over a growing name list: the round-10 review measured a
/// 400-link chain at 0.7-1.0 s and a 4000-link chain that did not finish inside 120 s. This one
/// visits each definition once when it takes effect and each edge once when a name turns suspicious,
/// so `a_long_macro_chain_does_not_stall_the_lint` pins a 4000-link chain at well under a second.
///
/// What is still not expanded, and is the honest limit of this reading: a macro defined in the
/// included header (invisible to this source), a keyword assembled by token pasting, and a
/// function-like macro whose ARGUMENT carries the control keyword (`#define WRAP(x) x` with
/// `WRAP(if (c))` is caught by the walk over the use, not by this body reading).
struct MacroScope {
    definitions: Vec<MacroDefinition>,
    /// A name a replacement list mentions, and the definitions that mention it, so marking one name
    /// walks its users exactly once instead of rescanning every body.
    users: std::collections::BTreeMap<String, Vec<usize>>,
    marked: Vec<bool>,
    /// How many definitions have taken effect at the scan's current position.
    in_effect: usize,
    defined: std::collections::BTreeSet<String>,
    suspicious: std::collections::BTreeSet<String>,
}

impl MacroScope {
    /// Reads the `#define`s of a source, in definition order.
    fn new(source: &str) -> Self {
        Self::from_tokens(&code_tokens_of(source))
    }

    /// The same, from tokens a check has already produced: a check that needs both the token stream
    /// and the macro scope (round-15 A-1) must not tokenize the source twice.
    fn from_tokens(tokens: &[CodeToken]) -> Self {
        let definitions = macro_definitions(tokens);
        let mut users: std::collections::BTreeMap<String, Vec<usize>> =
            std::collections::BTreeMap::new();
        for (index, definition) in definitions.iter().enumerate() {
            for name in &definition.mentions {
                users.entry(name.clone()).or_default().push(index);
            }
        }
        let marked = vec![false; definitions.len()];
        Self {
            definitions,
            users,
            marked,
            in_effect: 0,
            defined: std::collections::BTreeSet::new(),
            suspicious: std::collections::BTreeSet::new(),
        }
    }

    /// Publishes every definition that stands before `position`, and propagates suspicion through
    /// the definitions now in effect.
    fn advance_to(&mut self, position: usize) {
        while self.in_effect < self.definitions.len()
            && self.definitions[self.in_effect].position < position
        {
            let index = self.in_effect;
            self.in_effect += 1;
            let name = self.definitions[index].name.clone();
            self.defined.insert(name.clone());
            if self.definition_is_suspicious(index) {
                self.mark(index);
            } else {
                // A redefinition REPLACES the previous mark (round-15 A3): a name defined again with a
                // harmless body is no longer a macro that can hide control flow or a `return`, so a
                // stale mark must not make it read as one. Marks that other definitions took from the
                // name before the redefinition are not revisited; that is the stated limit of this
                // fixed-point walk, and it errs toward refusing.
                self.suspicious.remove(&name);
            }
        }
    }

    /// Whether a definition in effect is suspicious on its own or through a name it mentions.
    fn definition_is_suspicious(&self, index: usize) -> bool {
        let definition = &self.definitions[index];
        // Round-19 A5: an alias of the poll is not suspicious. Its body may contain a token the control
        // list names - `(&spark_wait_for_flag)` has an `&` - but that is the designator's address-of
        // operator, not a control operator in front of a statement, and nvcc compiles a helper that
        // calls through the alias to the baseline cubin. Marking it suspicious refused a call.
        // Round-20 A1: only a body that can BOUND the path is suspicious. A body that carries only an
        // operator keeps the operator's own rule instead, so `#define PRE (deadline > 0u) && true` before
        // the poll is not a blocker once its own statement has completed.
        (definition.bounds_the_path && !replacement_is_the_designator(&definition.body))
            || definition
                .mentions
                .iter()
                .any(|name| self.suspicious.contains(name))
    }

    /// Marks a definition suspicious, then the definitions that mention its name, once each, through
    /// an explicit worklist rather than a recursion: a chain of thousands of links is a shape the
    /// tests pin, and a recursion of that depth is a stack to fall off.
    ///
    /// Only definitions already in effect are marked: a later definition is judged when it takes
    /// effect, with this name already suspicious, which is what the preprocessor would do at a use
    /// after it.
    fn mark(&mut self, index: usize) {
        let mut queue = vec![index];
        while let Some(current) = queue.pop() {
            if self.marked[current] {
                continue;
            }
            self.marked[current] = true;
            let name = self.definitions[current].name.clone();
            self.suspicious.insert(name.clone());
            let users = self.users.get(&name).cloned().unwrap_or_default();
            for user in users {
                if user < self.in_effect && !self.marked[user] {
                    queue.push(user);
                }
            }
        }
    }

    /// Whether the source defines this name as a macro at the scan's current position.
    fn is_defined(&self, name: &str) -> bool {
        self.defined.contains(name)
    }

    /// Whether a macro of this name takes effect before the scan's current position and cannot be
    /// trusted: its expansion contains a control keyword, a non-returning call, or another such
    /// macro's name.
    fn is_suspicious(&self, name: &str) -> bool {
        self.suspicious.contains(name)
    }

    /// Whether a USE of `name` is an OPERATOR blocker: the definition in effect is a macro whose
    /// replacement list carries a control operator and no control keyword, so the use obeys the
    /// completed-statement scoping of the literal operator (round-20 A1).
    fn carries_only_operators(&self, name: &str) -> bool {
        self.definitions[..self.in_effect]
            .iter()
            .rev()
            .find(|definition| definition.name == name)
            .is_some_and(|definition| definition.carries_an_operator)
    }

    /// The replacement list of the definition of `name` that is in effect at the scan's position -
    /// the last definition of that name that has taken effect, because a redefinition replaces the
    /// one before it. `None` when the name is not in effect as a macro here.
    fn replacement_tokens(&self, name: &str) -> Option<&[String]> {
        self.definitions[..self.in_effect]
            .iter()
            .rev()
            .find(|definition| definition.name == name)
            .map(|definition| definition.body.as_slice())
    }
}

/// The `#define`s of a token stream, in definition order, with the position each one takes effect at
/// and the identifiers its replacement list mentions.
/// Whether `after` starts exactly where `before` ends, with no whitespace between them. The tokenizer
/// drops whitespace, so this is the only way to tell an object-like macro from a function-like one:
/// `#define P(x) ...` is function-like and `#define P (x) ...` is object-like, which is the
/// preprocessor's own rule (C 6.10.3p10) and what round-19 A5 needed to accept a parenthesised alias.
fn follows_without_space(before: &CodeToken, after: &CodeToken) -> bool {
    before.offset + before.text.len() == after.offset
}

fn macro_definitions(tokens: &[CodeToken]) -> Vec<MacroDefinition> {
    let mut definitions = Vec::new();
    let mut index = 0usize;
    while index < tokens.len() {
        if tokens[index].text != "#" || directive_keyword(tokens, index) != "define" {
            index += 1;
            continue;
        }
        let end = end_of_directive(tokens, index).min(tokens.len());
        if let Some(name) = tokens.get(index + 2) {
            let mut cursor = index + 3;
            // The `(` of a function-like macro has to follow the NAME with no space between them:
            // `#define P (spark_wait_for_flag)` is an OBJECT-like macro whose replacement list is the
            // parenthesised designator. Dropping the space made it function-like with an empty body, so
            // a call through it was not a mention of the poll and the alias was refused (round-19 A5).
            if tokens
                .get(cursor)
                .is_some_and(|token| token.text == "(" && follows_without_space(name, token))
            {
                // A function-like macro's parameter list; the replacement list starts after it.
                let mut depth = 0usize;
                while cursor < end {
                    match tokens[cursor].text.as_str() {
                        "(" => depth += 1,
                        ")" => {
                            depth -= 1;
                            if depth == 0 {
                                cursor += 1;
                                break;
                            }
                        }
                        _ => {}
                    }
                    cursor += 1;
                }
            }
            let body = &tokens[cursor.min(end)..end];
            definitions.push(MacroDefinition {
                name: name.text.clone(),
                position: index,
                mentions: body
                    .iter()
                    .filter(|token| is_identifier(&token.text))
                    .map(|token| token.text.clone())
                    .collect(),
                body: body.iter().map(|token| token.text.clone()).collect(),
                bounds_the_path: macro_body_bounds_the_path(body),
                carries_an_operator: macro_body_carries_only_operators(body),
            });
        }
        index = end.max(index + 1);
    }
    definitions
}

/// Whether a replacement list is known to end the fall-through path or never return.
///
/// A control keyword is enough, and so is a non-returning call's name. The one recognised no-op
/// spelling is the `do { } while (0)` idiom: it runs its body once and falls through, so a fence
/// after its use still runs, while reading its `do`/`while` as a control token made a harmless
/// `#define abort() do {} while(0)` a false rejection (round-10 A-4). Every other control token
/// counts, including a `do`/`while` whose condition is not the literal zero, which may never
/// terminate.
/// Whether a replacement list NAMES A TYPE rather than control flow: the `?` of
/// `decltype(true ? 0u : 1u)` counted as a control keyword and turned a return-type macro into a macro
/// USE, which made a declaration of the poll look like a call (round-15 A3). Only the `decltype`-headed
/// spelling is exempt; a `?` anywhere else still counts.
fn macro_body_is_a_type(body: &[CodeToken]) -> bool {
    body.first().is_some_and(|token| token.text == "decltype")
        && body.get(1).is_some_and(|token| token.text == "(")
}

/// Whether a replacement list can end the fall-through path or never return, which is what makes a USE
/// of the macro a blocker whatever statement it stands in: a control KEYWORD or a non-returning call.
///
/// Round-20 A1 split this out of the old `macro_body_is_control`, which counted the operators too. A
/// body that carries only an operator does not bound the path by itself - `#define PRE (deadline > 0u)
/// && true` then `const bool ran = PRE;` before the poll compiles to the 70432 / 4 / 605321041ac5ec1e baseline,
/// while `#define BAIL return` then `BAIL;` before the poll compiles to 68576 / 2 / bfd9e8e177f4f76d - not the
/// baseline. The operators keep the rule below.
///
/// The one recognised no-op spelling is the `do { } while (0)` idiom: it runs its body once and falls
/// through, so a fence after its use still runs, while reading its `do`/`while` as a control token made
/// a harmless `#define abort() do {} while(0)` a false rejection (round-10 A-4). Every other control
/// token counts, including a `do`/`while` whose condition is not the literal zero, which may never
/// terminate.
fn macro_body_bounds_the_path(body: &[CodeToken]) -> bool {
    if macro_body_is_a_type(body) {
        return false;
    }
    if body
        .iter()
        .any(|token| NON_RETURNING_CALLS.contains(&token.text.as_str()))
    {
        return true;
    }
    body.iter()
        .any(|token| is_control(&token.text) && !is_control_operator(&token.text))
        && !the_do_while_zero_idiom(body)
}

/// Whether a replacement list carries a control OPERATOR and no control keyword, so a use of the macro
/// is a blocker under the same completed-statement scoping as the literal operator (round-20 A1):
/// `const bool ran = PRE;` before the poll is accepted because that statement completed, while
/// `MAYBE(spark_wait_for_flag(flag, deadline));` is refused because the poll is the macro's argument and
/// no `;` stands between the use and the observation.
fn macro_body_carries_only_operators(body: &[CodeToken]) -> bool {
    if macro_body_is_a_type(body) || macro_body_bounds_the_path(body) {
        return false;
    }
    body.iter().any(|token| is_control_operator(&token.text)) && !the_do_while_zero_idiom(body)
}

/// Whether the replacement list is exactly the `do { } while (0)` no-op, in either spelling of the
/// literal zero.
fn the_do_while_zero_idiom(body: &[CodeToken]) -> bool {
    let texts: Vec<&str> = body
        .iter()
        .map(|token| token.text.as_str())
        .filter(|text| *text != ";")
        .collect();
    matches!(
        texts.as_slice(),
        ["do", "{", "}", "while", "(", "0", ")"]
            | ["do", "{", "}", "while", "(", "0u", ")"]
            | ["do", "{", "}", "while", "(", "false", ")"]
    )
}

/// Whether each token belongs to a preprocessor directive's logical line.
///
/// A macro definition's replacement list can spell the macro's own name (`#define BAIL(x) ...`), so a
/// backward walk over the tokens has to know which of them are directive text and not a use.
fn directive_token_mask(tokens: &[CodeToken]) -> Vec<bool> {
    let mut mask = vec![false; tokens.len()];
    let mut index = 0usize;
    while index < tokens.len() {
        if tokens[index].text == "#" {
            let end = end_of_directive(tokens, index).min(tokens.len());
            for slot in &mut mask[index..end] {
                *slot = true;
            }
            index = end;
            continue;
        }
        index += 1;
    }
    mask
}

/// The name of a suspicious macro used on the same statement as the poll at `start`, if any.
///
/// A poll passed as a macro argument (`BAIL(spark_wait_for_flag(...))`) is expanded inside the macro
/// body, which can put a `return` or an `abort()` in front of the fence; the round-4 A2 shape. The
/// same hazard exists when an object-like macro stands in front of the poll as a control statement
/// (`#define BAIL return` then `BAIL spark_wait_for_flag(...);`). The walk after the poll starts at
/// the poll and never sees the macro name in either case, so the use has to be caught here: scan
/// back to the start of the poll's statement (`;`, `{`, `}` or the start of the source), skipping
/// directive lines so the definition's own spelling of the name is not mistaken for a use. A nested
/// spelling such as `BAIL((poll))` is found because every token of the statement is visited.
fn suspicious_macro_in_poll_statement(
    tokens: &[CodeToken],
    start: usize,
    macros: &MacroScope,
    directive: &[bool],
) -> Option<String> {
    let mut cursor = start;
    while cursor > 0 {
        cursor -= 1;
        if directive.get(cursor).copied().unwrap_or(false) {
            continue;
        }
        match tokens[cursor].text.as_str() {
            ";" | "{" | "}" => return None,
            name if macros.is_suspicious(name) => {
                return Some(name.to_string());
            }
            _ => {}
        }
    }
    None
}

/// A kernel that polls a flag a peer block wrote, and then reads what that flag released, needs the
/// acquire half of the release/acquire pair on the observing path.
///
/// This is the belt and braces beside the guarantee. The guarantee is the set of checks in the
/// module documentation above: [`observing_paths_call_the_acquiring_poll`] (no kernel names the raw
/// poll, in any spelling of the call), [`acquire_helper_fences_unconditionally`] (the helper's own
/// body is exactly one poll and then a fence, at one brace depth, with nothing that can intervene),
/// [`guarded_names_are_not_redefined`] and [`the_acquiring_poll_is_defined_exactly_once`]. Together
/// with the helper's body those make the fence part of the expansion every call site gets, in every
/// spelling of that call, and they name the rewrites that would replace the call instead of merely
/// assuming none exists. What follows is the reachability lint that the earlier rounds grew; it is
/// kept because it is the only check that reads the *text around* a poll and asks whether a fence
/// after it really runs, which is what protects the helper if its body is ever reshaped. It is not
/// the guarantee, and the list at the end of this comment is why: it reasons about source shape, so
/// it has to enumerate the tricks, and an enumeration is only as good as its list.
/// The Milestone 3 review defeated the earlier, line-based version of this guard six ways, and the
/// round-2 and round-3 reviews five more, all of them text tricks that a compiler ignores: a fence
/// hidden in a comment, a poll mention in a comment used to re-anchor the search before the release
/// fence, a fence wrapped in if (false), a fence inside an unrelated branch, an early return, a
/// control statement the line scan did not recognise (switch/case, do/while, a label), a
/// short-circuit && or || whose right operand holds the fence, a fence on a #define continuation
/// line, a fence behind a call that never returns, a keyword or comment opener split by a
/// translation-phase-2 splice, and a fence or a poll spelled inside a raw string literal. Round 4
/// added the horizontal-whitespace form of that splice (which clang and nvcc also make), a
/// function-like macro body, a poll passed as a suspicious macro's argument, and a literal `#if 1`
/// the earlier mask treated as not compiled. So this version works on tokens, after the phase-2
/// splice and comment stripping:
///
///   * the poll is located in CODE that is definitely compiled: a mention inside a comment, on a
///     preprocessor directive line, in a #define body, or in a region that is definitely disabled or
///     whose condition the guard cannot resolve (a literal #if 0, an #ifdef, an arithmetic #if) is
///     not a poll. A literal #if 1 and the #else arm of a literal #if 0 ARE compiled, so a poll
///     there counts. A poll whose text lives only in a macro body is still reported as no poll,
///     because the guard does not expand a macro that has no call site;
///   * the tokeniser makes each string, character and raw string literal one opaque token, so a
///     keyword inside literal content is never a keyword;
///   * the first __threadfence after the poll must be a bare call statement, not part of any larger
///     expression; the designator may be parenthesised, dereferenced or have its address taken in any
///     nesting (`(__threadfence)();`, `(*__threadfence)();`, `(&__threadfence)();`,
///     `(*(&__threadfence))();`), a fence alone in a bare block is that statement, and a label before
///     it is a statement boundary (`obs: __threadfence();`) rather than an expression, as long as the
///     colon is not the false branch of a `? :` and is not half of a `::`. Only the unqualified or
///     globally qualified name is this barrier: `X::__threadfence();` and `x.__threadfence();` name
///     a namespace's or an object's function, so the body is read as having no barrier;
///   * nothing that ends the fall-through path or makes execution conditional may stand between the
///     poll and the fence: if, else, for, while, do, switch, case, default, return, break, continue,
///     goto, throw, co_return, ?, &&, ||, and the single-character & and | (rejected because either
///     half of a short-circuit operator, and the bitwise operator with a statement-expression operand,
///     has the same hazard). The four operators are scoped to the fence's OWN statement: the `?` of a
///     completed `bool ready2 = ready ? true : false;` in front of the fence cannot make it
///     conditional and is accepted, while an operator with no `;` between it and the fence is in the
///     fence's statement and is rejected. The keywords keep the conservative reading at any position,
///     because nothing here bounds a branch or a loop in front of the fence;
///   * a call to a known non-returning function is rejected, since a fence after it cannot run, and
///     the test is whether the designator is CALLED rather than whether the call is a statement, so
///     `(abort)()`, `(__builtin_unreachable(), 0u)` inside a comma expression, an initialiser list
///     and a nested argument are all rejected too;
///   * a use of a macro defined in this source - object-like or function-like, with the parameter
///     list stripped - whose own body contains a control keyword, a non-returning call, or the name
///     of another such macro (read to a fixed point, so `#define OUTER INNER` cannot launder it) is
///     rejected. That covers a poll used on the same statement as such a macro, whether the poll is
///     its argument (`BAIL((poll))`) or follows it as a control statement (`BAIL poll;`), and a use
///     that stands BEFORE the poll anywhere in the poll's own block (round-13 A-2): the expansion can
///     return from the helper before the poll, so the lint reads the block rather than only the text
///     after the poll - and since round 20 A1 the structural check applies the same rule to the same
///     span, so such a text is refused by BOTH in the same words;
///   * a fence inside a preprocessor conditional that may not hold is rejected, while a fence inside
///     a level that is definitely active (a literal #if 1) is accepted; a region that is definitely
///     disabled (#if 0) is ignored rather than obeyed; and a whole logical directive line -
///     continuations included by phase 2 - is skipped instead of being scanned as code. Each of those
///     branches has its own case in `every_rejection_carries_the_canonical_hint`, with a needle only
///     that branch prints (round-14 A-3: the fence-in-a-conditional branch was claimed covered and was
///     not, because two branches shared the wording the case asserted);
///   * a text whose helper body holds more than one compiled poll, or whose helper body holds none
///     while the text defines the helper, is rejected instead of guessed at: the acquire pairs with
///     one observation, and the lint will not decide on the developer's behalf which call that is
///     (round-14 A-1; the two-poll refusal is conservative - a helper body that calls the poll twice
///     is valid C++ and nvcc compiles it, to 39112 bytes / 3 PTX membar lines / sha 07aa2c1d... (pre-change)
///     against the 70432 / 4 / 605321041ac5ec1e baseline, measured - the shipped helper observes once, and a
///     second observation is a change this guard will not certify on the developer's behalf). The
///     poll it follows is the HELPER's own call, not the first call of that name
///     in the text - a forward declaration, a function-pointer parameter and an earlier polling
///     function all name the raw poll without being the observation, and each of those headers
///     compiles to the baseline cubin (70432 / 4 / 605321041ac5ec1e, measured);
///
/// What this does NOT catch, stated so that the claim is no wider than the code: a non-returning
/// callee that is not on NON_RETURNING_CALLS, or one reached through a function pointer, a macro or
/// inline asm; a macro this source does not itself define (the included header's macros are invisible
/// here), or a control keyword assembled by token pasting - the names this source DOES define are
/// followed to a fixed point, so a nested expansion is caught; an unbounded loop before the fence; a
/// computed jump; and any reachability that depends on a value, because this is a check on the shape
/// of the source and not a dataflow analysis. Trigraphs are not a splice in C++17, the
/// level this milestone builds with, so `??/` before a newline is not treated as a comment opener;
/// the backslash-newline form is, and it is valid C++17.
///
/// It is also a guard on the SHAPE of the source in this repository, and it is not a proof of
/// anything about the hardware: only the release/acquire semantics of the fence itself are CUDA's
/// business, and on coherent UMA a reviewer may argue the acquire is unnecessary. The kernel says
/// in a comment that it runs the fence because a conditional one can be wrong while an
/// unconditional one cannot; the guard's job is to keep that claim true as the code changes.
///
/// The raw poll is located by the same rule the structural check uses - the helper's body must mention
/// the designator exactly once, and that mention must be a CALL of it (an argument list follows the
/// name, and nothing in front of it marks the same tokens as a declaration). Neither the arguments nor
/// the expression the call stands in are read, so every call is accepted whatever its prefix and
/// whatever the arguments are spelled like (round 17; the module documentation carries the accepted and
/// refused lists with the cubin each shape compiles to). A mention that is not a call - a declaration,
/// a function-pointer declarator, an attribute between the type and the name - is refused, and the
/// shipped spelling the message names is an illustration rather than a substitution for a helper whose
/// parameters are named differently. The `poll` argument is the label those messages print, not a
/// pattern to match (round-13 A-2).
/// Of those calls this lint follows the one inside the helper's own body, so plumbing that merely
/// names the raw poll cannot hijack it, and a rejection always names a line of that body rather than
/// a line of some other function (round-14 A-1).
///
/// # Errors
/// Returns the reason the fence is not reachable, so a test can assert on the specific evasions.
pub fn acquire_fence_after_poll(source: &str, poll: &str) -> Result<(), String> {
    let tokens = code_tokens_of(source);
    assert!(!poll.is_empty(), "the poll label must not be empty");
    // A using-declaration or using-directive that re-binds one of the guarded names is the same
    // rewrite as a #define, so this check refuses it too: `using X::__threadfence;` makes the bare
    // call below name X's no-op while its text is unchanged (round-10 A-2).
    if let Some((_, line, name, target)) =
        guarded_names_brought_into_scope(&[("this source", source)])
    {
        return Err(rejection(using_declaration_rejection(
            &format!("line {line}"),
            &name,
            &target,
        )));
    }
    // The macros of this source, in definition order. Their uses are rejected below when the
    // definition stands before the use, because the preprocessor is positional (round-10 A-5); a
    // named chain is followed to a fixed point, and a macro defined in the included header is not
    // expanded, which is the stated limit of the guard.
    let mut macros = MacroScope::new(source);
    let active = active_code_mask(&tokens);
    // The poll this lint follows is the HELPER's own observation, not the first raw-poll call in the
    // text. Round-14 A-1: a forward declaration, a function-pointer parameter and an earlier function
    // that polls all name the raw poll without being the observation, and anchoring on the first call
    // made the lint read plumbing and print a line that was not the helper's poll. The rule that
    // LOCATES the raw poll is still the one the structural check uses, so the two cannot diverge on
    // its spelling; this decides which of those calls is the observation.
    let calls: Vec<usize> = raw_poll_call_indices(&tokens)
        .into_iter()
        .filter(|&index| active[index])
        .collect();
    let helper_body = function_body_tokens(&tokens, ACQUIRE_HELPER);
    let start = match helper_body {
        Some((open, close)) => {
            // Inside the helper's own body this lint reads ONE rule (round 17, corrected in round 18):
            // the body must mention the poll exactly once, and that mention must be a CALL of it - the
            // designator followed by an argument list, however the designator is parenthesised,
            // dereferenced or addressed, and whatever expression the call stands in. What is refused
            // is a mention that is NOT a call: a declaration in any return-type spelling, a
            // function-pointer declarator, an attribute between the type and the name. Those bodies
            // omit the call and compile to something other than the baseline (70600 / 3 / 1f2503fd7ecaf189,
            // measured), while every call shape - including the ones an earlier rule refused as
            // "wrapped in a comparison" - is a translation unit whose cubin the module documentation
            // lists.
            let inside = compiled_raw_poll_mentions(&tokens, open, close);
            // Round-23 A3 (HIGH): a poll the preprocessor may not compile is not a poll this lint
            // follows - but it IS a poll the scan cannot place, and the structural check refuses the
            // body for it. Accepting the shape whenever a compiled poll also existed made a SILENT
            // second observation possible: `#if M4_FLAG` around a first poll, then an unconditional
            // poll and the fence, compiles under -DM4_FLAG=1 to two real polls (39072 / 95e0cf41 (pre-change))
            // whose one acquire pairs with the second only, while without the define it is the
            // baseline. Both checks now refuse it, and with the same sentence, so the agreement is
            // whole-message equality rather than a matching verdict.
            if let Some(&unplaced) = raw_poll_mentions(&tokens, open, close)
                .iter()
                .find(|mention| !inside.contains(mention))
            {
                return Err(rejection(format!(
                    "line {}: the poll in {ACQUIRE_HELPER} sits inside a preprocessor conditional, so it is not guaranteed to be compiled",
                    tokens[unplaced].physical_line
                )));
            }
            match inside.len() {
                1 => {
                    if a_void_type_precedes_the_designator(&tokens, inside[0]) {
                        return Err(rejection(void_typed_poll_message(
                            tokens[inside[0]].physical_line,
                        )));
                    }
                    let directives = directive_token_mask(&tokens);
                    let mut at_the_mention = MacroScope::from_tokens(&tokens);
                    at_the_mention.advance_to(inside[0]);
                    if !the_mention_is_a_call_of_the_poll(
                        &tokens,
                        inside[0],
                        &at_the_mention,
                        &directives,
                    ) {
                        return Err(rejection(format!(
                            "line {}: the only mention of {RAW_POLL} in {ACQUIRE_HELPER}'s body is not \
                             a CALL of it, so this lint cannot certify that the fence follows an \
                             observation; the shared header writes the call as \
                             `{CANONICAL_POLL_CALL}` (with the helper's own parameter names - the \
                             arguments are not read), and this scan reads a mention as the observation \
                             when the designator is called and nothing in front of it makes the same \
                             tokens a declaration",
                            tokens[inside[0]].physical_line
                        )));
                    }
                    inside[0]
                }
                0 => {
                    return Err(rejection(format!(
                        "no poll of {poll} in {ACQUIRE_HELPER}'s compiled body (a mention inside a \
                         comment is not a poll, and a poll the preprocessor may not compile is not one \
                         this lint follows)"
                    )));
                }
                count => {
                    return Err(rejection(format!(
                        // Round-29 A2: "this lint" used to stand here where the structural check
                        // says "this scan", which made the COUNT a family the two answered in their
                        // own voices - the module documentation claimed one family while this
                        // sentence acknowledged two. Both now say "this scan", so the two messages
                        // are byte-identical for the same text and the matrix can assert equality.
                        "line {} and line {}: {ACQUIRE_HELPER} mentions the poll {count} times, so \
                         this scan cannot tell which observation the fence pairs with",
                        tokens[inside[0]].physical_line,
                        tokens[inside[count - 1]].physical_line
                    )));
                }
            }
        }
        None => match calls.len() {
            1 => calls[0],
            0 => {
                return Err(rejection(format!(
                    "no poll of {poll} in this source's code (a mention inside a comment is not a \
                     poll)"
                )));
            }
            count => {
                return Err(rejection(format!(
                    "line {} and line {}: this text calls {poll} {count} times and does not define \
                     {ACQUIRE_HELPER}, so this lint cannot tell which call is the observation: run it \
                     on the helper's own text",
                    tokens[calls[0]].physical_line,
                    tokens[calls[count - 1]].physical_line
                )));
            }
        },
    };
    // A poll used with one of those macros is expanded inside it (or follows it as a control
    // statement), so the macro body can put a `return` or an `abort()` in front of the fence while
    // the walk below, which starts after the poll, never sees the macro name.
    let directive_tokens = directive_token_mask(&tokens);
    // Only the definitions in effect at the poll can be expanded by it. The poll wrapped in a
    // suspicious macro keeps its own reason, so this check runs before the pre-poll scan below: the
    // macro-use reason would otherwise shadow it (round-13 D1).
    macros.advance_to(start);
    if let Some(name) =
        suspicious_macro_in_poll_statement(&tokens, start, &macros, &directive_tokens)
    {
        return Err(rejection(format!(
            "line {}: the poll is used with {}, a macro whose body contains a control keyword, a \
             non-returning call, or another such macro, so this scan does not certify that the fence \
             runs after it",
            tokens[start].physical_line, name
        )));
    }
    // The poll's enclosing block, BEFORE the poll. The structural check reads the whole helper body,
    // so it refuses a control token or a non-returning call that stands before the poll; this walk
    // started AT the poll and accepted the same shape, which was the second disagreement in round-12
    // A1. The block is the poll's own, so an unrelated function's control flow is not read.
    let block_start = enclosing_block_start(&tokens, start);
    // The poll's own designator, from its first token to the name: `(poll)`, `(*poll)`, `(&poll)`.
    // Round-18 A1: the `&` of `(&poll)(...)` is part of the call, not a blocker in front of it, and
    // reading it as the bitwise operator refused a call nvcc compiles to the baseline. The fence has
    // had the same unwrap since round 8 (`fence_designator_starting_at`); this is the poll's.
    let designator_start =
        call_argument_list(&tokens, start).map_or(start, |(designator, _)| designator);
    // Round-18 A8: the poll has to be a top-level statement of the helper's own body. The structural
    // check refuses a poll at a nested brace depth (`poll_depth != 0`) and this walk accepted the same
    // text, so the two checks disagreed about a helper whose observation and fence sit inside a block
    // neither can relate to the helper's straight-line path. They answer the same way now, with the
    // structural check's own words.
    if helper_body.is_some_and(|(open, _)| block_start != open) {
        return Err(rejection(format!(
            "line {}: the poll in {ACQUIRE_HELPER} is not a top-level statement of its body, so the \
             helper's expansion is not the fixed poll-then-fence shape",
            tokens[start].physical_line
        )));
    }
    let mut earlier_conditionals = conditional_stack_before(&tokens, block_start);
    let mut earlier = block_start;
    while earlier < start {
        let text = tokens[earlier].text.as_str();
        if text == "#" {
            update_conditionals(&tokens, earlier, &mut earlier_conditionals);
            earlier = end_of_directive(&tokens, earlier);
            continue;
        }
        if definitely_disabled(&earlier_conditionals) {
            earlier += 1;
            continue;
        }
        if earlier >= designator_start && earlier < start {
            earlier += 1;
            continue;
        }
        macros.advance_to(earlier);
        if fence_designator_starting_at(&tokens, earlier).is_none() {
            // A macro whose body can BOUND the path - a control keyword or a non-returning call - is
            // the same blocker before the poll as after it, and BOTH checks read it now: the structural
            // check applies this rule to its own span (round-20 A1), which is why the verdicts agree. It
            // is how `BAIL(<poll>);` is caught: the poll is the macro's ARGUMENT, so the walk from the
            // poll never sees the name. A macro whose body carries only an OPERATOR keeps the operator's
            // own rule instead, in the branch below.
            // Round-28 A5: a macro use inside a LAMBDA's body is not this helper's path either, so
            // the macro branch skips a lambda body exactly as the literal branches do. `#define BAIL
            // return` used inside a lambda before the poll exits the LAMBDA and compiles to the
            // baseline; the macro path is read textually, and the documentation says why.
            if macros.is_suspicious(text)
                && !inside_a_body_the_helper_does_not_own(&tokens, earlier, block_start)
            {
                return Err(rejection(format!(
                    "line {}: {} is a macro whose body contains a control keyword, a non-returning \
                     call, or another such macro, so this scan does not certify that the acquire after \
                     it runs",
                    tokens[earlier].physical_line, text
                )));
            }
            // Round-28 A1: the lambda guard the control-token branch below has had since round 27.
            // Without it this branch refused a non-returning call inside a lambda before the poll -
            // `auto f = [&]{ __trap(); }; (void)f;` - which nvcc compiles to the baseline, while the
            // structural check accepted it: a verdict split the lint lost.
            if NON_RETURNING_CALLS.contains(&text)
                && !inside_a_body_the_helper_does_not_own(&tokens, earlier, block_start)
                && is_the_library_non_returning_call(&tokens, earlier, &macros)
            {
                return Err(rejection(format!(
                    "line {}: {} does not return, so this scan does not certify that the acquire after it runs",
                    tokens[earlier].physical_line, text
                )));
            }
            // Round-18 A2/A3: a control token that GOVERNS the poll - the condition the poll is, or
            // the conditional expression it stands in - belongs to the poll's own statement, and the
            // fence after that statement still runs unconditionally, so it is not a blocker. This is
            // also what makes this walk agree with the structural check about a conditional
            // expression, which it used to refuse on the `?` alone.
            // Round-19 A1: an operator in an earlier statement that already completed is not a blocker.
            // `const bool ran = (deadline > 0u) && true;` before the poll compiles to the baseline, and
            // the operator cannot bound the poll's own statement because its own statement ended at its
            // semicolon. The structural check applies the same rule to its operator blockers.
            // Round-20 A1: a macro whose body carries only an operator is a blocker with the
            // operator's own rule, so the completed-statement scoping decides it exactly as it does for
            // the literal operator. That keeps `MAYBE(spark_wait_for_flag(flag, deadline));` refused
            // (the poll is the macro's argument and no `;` stands between the use and the observation)
            // while `const bool ran = PRE;` before the poll, whose own statement ended, is not a blocker
            // at all - nvcc compiles that text to the baseline cubin.
            let operator_blocker = (is_control_operator(text)
                || macros.carries_only_operators(text))
                && !inside_a_lambda_capture(&tokens, earlier);
            if (is_control(text) || operator_blocker)
                && !inside_a_lambda_capture(&tokens, earlier)
                && !inside_a_body_the_helper_does_not_own(&tokens, earlier, block_start)
                && !the_control_token_governs_the_poll(&tokens, earlier, start)
                && !(operator_blocker && a_statement_completed_between(&tokens, earlier, start))
            {
                if is_an_exit_keyword(text) {
                    return Err(rejection(format!(
                        "line {}: {:?} stands before the poll, so the acquire after it is unreachable",
                        tokens[earlier].physical_line, text
                    )));
                }
                return Err(rejection(format!(
                    "line {}: {:?} stands before the poll, so the acquire after it may not be reached",
                    tokens[earlier].physical_line, text
                )));
            }
        }
        earlier += 1;
    }

    // The conditional stack enclosing the poll. Recovering it here, rather than starting empty at
    // the poll, keeps the walk from reading the `#else` of an enclosing `#if 1` as unconditional
    // code.
    let mut conditionals = conditional_stack_before(&tokens, start);
    // The code tokens visited so far, in order. A preprocessor directive is not a statement, and
    // neither is a token inside a definitely-disabled region, so neither is recorded: a fence that
    // follows a #endif still follows whatever statement stood before the directive. The indices are
    // kept rather than the last token's text so the fence can look past the parentheses of its own
    // designator for the statement it starts.
    let mut visited_code: Vec<usize> = Vec::new();
    // Whether each open brace encloses a body that is not the helper's own straight-line path: a
    // lambda's or a type's body (round-10 A-3) and any branch or loop body. A fence inside one of
    // them is not the unconditional acquire, however unconditional it looks in isolation: the closure
    // of `auto l = [] { __threadfence(); };` runs only if it is called.
    let mut nested_bodies: Vec<bool> = Vec::new();
    // The first fence this walk saw inside a nested body, recorded rather than reported. The
    // structural check records the same thing and only reports it when the body has NO top-level
    // fence, and the two checks have to agree: `auto l = [] { __threadfence(); }; (void)l;
    // __threadfence();` is the baseline cubin byte for byte (70432 / 4 / 605321041ac5ec1e) and this walk used
    // to reject it on the lambda's fence before ever reaching the real acquire (round-12 A1).
    let mut nested_fence: Option<usize> = None;
    // Round-25 A2: a fence the preprocessor may not compile is a FALLBACK candidate, not an immediate
    // refusal. `poll; #if M4_FLAG __threadfence(); #endif __threadfence(); return observed;` has the
    // acquire - the unconditional fence - and preprocesses without the define to the shipped baseline,
    // so returning at the conditional one refused a body whose acquire is present. It is reported only
    // when no unconditional compiled fence followed it, exactly as the structural check does it.
    let mut conditional_fence: Option<usize> = None;
    // The first control token or non-returning call that stands between the poll and the fence,
    // recorded rather than reported at once. The structural check evaluates the fence first and only
    // then the blockers, so a body with no top-level fence is reported as exactly that; reporting the
    // blocker here as soon as it appeared made the two checks answer with different reasons for the
    // same body (round-12 A1). It is still an error - when a top-level fence is found, any recorded
    // blocker is between the poll and the acquire and is reported before the walk returns.
    let mut blockers: Vec<(usize, bool, bool, usize)> = Vec::new();
    // Every `;` with the brace depth it sits at. The structural check keeps exactly these two lists
    // and decides which blocker is relevant with them; the lint keeping them too is what makes the two
    // checks answer the same way about an operator in an earlier, completed statement (round-12 A1)
    // instead of the lint reading the fence inside a lambda as the operator's own statement.
    let mut semicolons: Vec<(usize, usize)> = Vec::new();
    // Round-24 A1: the walk starts after the poll's argument list - a fence written there runs BEFORE
    // the call and is never the acquire - but the arguments are still READ for control flow. A `return`,
    // a control keyword or a call that never returns inside them ends the path just as it would in front
    // of the poll: `spark_wait_for_flag((__trap(), flag), deadline); __threadfence(); return observed;`
    // compiles to 69536 / 4 / 5826994128b34ed0, byte-identical to the trap-before-the-poll statement body that
    // both checks refuse, and `({ return 0u; flag; })` as the first argument compiles to 70464 / 2 /
    // 2b56c510. Reading the span is what keeps the verdict a function of the compiled shape.
    let argument_span = call_argument_list(&tokens, start)
        .and_then(|(_, open)| matching_close_bracket(&tokens, open).map(|close| (open, close)));
    let mut index = call_argument_list(&tokens, start)
        .and_then(|(_, open)| matching_close_bracket(&tokens, open))
        .map_or(start + 1, |close| close + 1);
    if let Some((open, close)) = argument_span {
        // Round-25 A1/A3: the reading has to mean in the arguments what it means in front of the poll.
        // A macro whose body returns or traps is a blocker here too - the structural check read it and
        // the lint did not, so the two answered the same compiled body differently
        // (`spark_wait_for_flag(({ BAIL; flag; }), deadline);` with `#define BAIL return 0u` is
        // 70464 / 2 / 2b56c510b5f4a945, the cubin of the literal return-in-argument shape both refuse). A
        // LAMBDA body is skipped, because a `return` there exits the lambda (the docs have promised
        // that since round 10, and `spark_wait_for_flag(([&]{ return flag; })(), deadline);` compiles
        // to the baseline with the acquire present), and a bare `if`/`while`/`for`/`switch` is not a
        // blocker by itself: only the exit keyword inside it is, and the same reading finds that.
        let mut inner = open + 1;
        while inner < close {
            let text = tokens[inner].text.as_str();
            // Round-26 A2: a DEFINITELY DISABLED region in the arguments is not read at all, exactly
            // as the walks do not read it. `#if 0 return 0u; #endif` and a `#if 0`-disabled macro use
            // are removed by the compiler, so refusing them refused a body nvcc compiles to the
            // baseline. A conditional the scan cannot resolve is NOT disabled, and a `return` in one
            // still blocks - under the define it really runs.
            if text == "#" {
                update_conditionals(&tokens, inner, &mut conditionals);
                inner = end_of_directive(&tokens, inner);
                continue;
            }
            if definitely_disabled(&conditionals) {
                inner += 1;
                continue;
            }
            macros.advance_to(inner);
            // Round-26 A1: the narrow backward scan misses a TRAILING RETURN TYPE
            // (`[&]() -> const volatile uint32_t * { ... }`), so the walk's own detector decides a
            // body the narrow one does not recognise.
            if text == "{"
                && (a_lambda_body_opens_at(&tokens, inner)
                    || opens_a_lambda_or_a_type_body(
                        &tokens,
                        &(start..inner).collect::<Vec<usize>>(),
                    ))
            {
                inner = matching_close_bracket(&tokens, inner).map_or(inner + 1, |end| end + 1);
                continue;
            }
            let never_returns = NON_RETURNING_CALLS.contains(&text)
                && is_the_library_non_returning_call(&tokens, inner, &macros);
            if never_returns || is_an_argument_exit(text) {
                blockers.push((inner, never_returns, false, 0));
            } else if macros.is_suspicious(text) {
                return Err(rejection(format!(
                    "line {}: {} is a macro whose body contains a control keyword, a non-returning \
                     call, or another such macro, so this scan does not certify that the fence runs after \
                     it",
                    tokens[inner].physical_line, text
                )));
            }
            inner += 1;
        }
    }
    while index < tokens.len() {
        let token = &tokens[index];
        if token.text == "#" {
            update_conditionals(&tokens, index, &mut conditionals);
            // The whole LOGICAL line belongs to the preprocessor, including a #define body or an
            // #else keyword, and none of it is a statement on this path. The backslash-newline
            // splice has already joined a continuation line into this logical line at translation
            // phase 2, so a fence on a continuation line is part of the directive and never runs
            // either: skipping only the line that carries the # is the reviewer's fourth shape.
            index = end_of_directive(&tokens, index);
            continue;
        }
        if definitely_disabled(&conditionals) {
            // Inside #if 0: nothing here executes, so neither a control nor a fence counts.
            index += 1;
            continue;
        }
        // Only the definitions that stand before this token are in effect, as the preprocessor would
        // have it at this point in the file (round-10 A-5).
        macros.advance_to(index);
        if token.text == "{" {
            let lambda_or_type = opens_a_lambda_or_a_type_body(&tokens, &visited_code);
            // A `)` is the end of an if/for/while/switch/catch condition, and `try`, `do` and
            // `requires` name a body directly. The structural check has always treated a `try` body
            // as a nested body while this walk stepped into it and accepted a fence there, which was
            // the one place the two checks disagreed (round-11 A1); `requires` was accepted by both
            // even though nothing inside a requires-expression is executed (round-11 A1).
            // The token read here can be an attribute's closing bracket, as in `try [[likely]] { ... }`
            // or `catch (...) [[likely]] { ... }`, so the read steps over any attribute group first
            // (round-36 A1). Reading the `]]` alone let BOTH checks accept those two spellings, which
            // the plain ones are refused for, and no later blocker rescues them: neither `try` nor
            // `catch` is in [is_control].
            let branch = the_token_before_the_attributes(&tokens, visited_code.last().copied())
                .is_some_and(|previous| {
                    matches!(
                        tokens[previous].text.as_str(),
                        ")" | "try" | "do" | "requires"
                    )
                });
            nested_bodies.push(lambda_or_type || branch);
            visited_code.push(index);
            index += 1;
            continue;
        }
        if token.text == "}" {
            nested_bodies.pop();
            visited_code.push(index);
            index += 1;
            continue;
        }
        if macros.is_suspicious(&token.text)
            && !inside_a_body_the_helper_does_not_own(&tokens, index, block_start)
        {
            return Err(rejection(format!(
                "line {}: {} is a macro whose body contains a control keyword, a non-returning \
                 call, or another such macro, so this scan does not certify that the fence runs after \
                 it",
                token.physical_line, token.text
            )));
        }
        if NON_RETURNING_CALLS.contains(&token.text.as_str())
            && is_the_library_non_returning_call(&tokens, index, &macros)
            // Round-27 A2: a call inside a LAMBDA's body is not this helper's path either, exactly as
            // the structural check now reads it and as the argument-span reading already did.
            && !inside_a_body_the_helper_does_not_own(&tokens, index, block_start)
        {
            // A call that never returns makes everything after it unreachable, so the fence is not
            // an unconditional acquire on this path: the reviewer's fifth shape. The test is the
            // designator being called, NOT the call being a whole statement: `(__trap(), 0u)` is
            // called inside a comma expression, so requiring a `;` after the argument list missed a
            // whole class - the round-8 A1 hole, where nvcc removed two of the four barriers. The
            // verdict waits for the fence, so the reason matches the structural check's reason.
            blockers.push((index, true, false, nested_bodies.len()));
        }
        // The barrier this guard is about is the unqualified or globally qualified function:
        // `X::__threadfence()` and `x.__threadfence()` name a namespace's or an object's function,
        // which is not this barrier at all, and reading their `::` as a label colon is how a
        // namespaced no-op passed for the acquire (round-8 A3). None means "no barrier here", and the
        // loop then reports that no reachable fence follows the poll.
        let fence_designator = if token.text == "__threadfence" {
            let start = call_argument_list(&tokens, index).map_or(index, |(start, _)| start);
            global_fence_designator(&tokens, &visited_code, start)
        } else {
            None
        };
        if let Some(designator) = fence_designator {
            if nested_bodies.iter().any(|&nested| nested) {
                // A lambda's or a type's body (round-10 A-3), or a branch, loop, try block or
                // requires-expression: the fence runs only on some paths, so it is not the acquire
                // the helper owes its caller. It is NOT reported here, though - the walk keeps going
                // so a real top-level fence after it can still be found, which is what the structural
                // check does and what round-12 A1 requires of both.
                nested_fence.get_or_insert(index);
                visited_code.push(index);
                index += 1;
                continue;
            }
            if !is_bare_fence_call(&tokens, index) {
                return Err(rejection(format!(
                    "line {}: the __threadfence is not a bare call statement, so it may be guarded",
                    token.physical_line
                )));
            }
            if !definitely_compiled(&conditionals) {
                // Round-25 A2: record it and keep walking. Reporting here refused a body whose acquire
                // is present: with an unconditional fence later - or earlier, which the structural
                // check already accepted - the conditional one is irrelevant, and without the define
                // the body IS the shipped baseline. The verdict waits for the end of the walk.
                conditional_fence.get_or_insert(token.physical_line);
                visited_code.push(index);
                index += 1;
                continue;
            }
            // The statement this call starts. `__threadfence` and `(__threadfence)` are the same
            // call in C++, so the search starts before the whole designator: looking at the token
            // immediately before the name found a `(` and rejected valid C++ (round 7). A label is a
            // statement boundary as well (round 8) and a global `::` belongs to the designator, and
            // this is the same helper the structural check calls, so the two cannot disagree about a
            // spelling the way they did for `(*f)();`.
            if !starts_a_statement(&tokens, &visited_code, designator) {
                let previous = code_token_before(&visited_code, designator)
                    .map_or("the start of the block", |code| tokens[code].text.as_str());
                return Err(rejection(format!(
                    "line {}: the __threadfence is part of a larger expression that follows \
                     {previous:?}, so \
                     it may not run",
                    token.physical_line
                )));
            }
            // The first blocker that is not an operator in an earlier, completed statement. This is
            // the structural check's `relevant` predicate, on the same lists: the `;` that ends the
            // operator's statement is a `;` at the operator's own depth, so the `;` of `int z = 0;`
            // inside a statement expression or a lambda body does not end it (round-10 A-1).
            let fence_index = index;
            let relevant = |&(blocked, _, operator, blocker_depth): &(usize, bool, bool, usize)| {
                !operator
                    || !semicolons.iter().any(|&(code, semicolon_depth)| {
                        code > blocked && code < fence_index && semicolon_depth == blocker_depth
                    })
            };
            if let Some(&(blocked, never_returns, _, _)) =
                blockers.iter().find(|blocker| relevant(blocker))
            {
                let line = tokens[blocked].physical_line;
                let text = tokens[blocked].text.as_str();
                if never_returns {
                    return Err(rejection(format!(
                        "line {line}: {text} does not return and stands between the poll and the acquire in {ACQUIRE_HELPER}, so this scan does not certify that the acquire runs on every path"
                    )));
                }
                return Err(rejection(format!(
                    "line {line}: {text:?} stands between the poll and the acquire in {ACQUIRE_HELPER}, and this scan reads placement rather than a construct's body, so it does not certify the acquire on every path"
                )));
            }
            return Ok(());
        }
        if is_control(&token.text) {
            // The `&` or `*` of a fence designator is not the operator this rule is about, so
            // `(&__threadfence)();` and `(*__threadfence)();` fall through to the fence below. An
            // operator is recorded like any other blocker and scoped at the fence, by the same rule
            // the structural check uses: an operator with a `;` of its own depth between it and the
            // fence is in an earlier, completed statement, which is why a `&` in a lambda's capture
            // list does not block the real acquire after the lambda (round-12 A1).
            if fence_designator_starting_at(&tokens, index).is_none()
                && !(is_control(&token.text)
                    && inside_a_body_the_helper_does_not_own(&tokens, index, block_start))
            {
                let operator = matches!(token.text.as_str(), "?" | "&&" | "||" | "&" | "|");
                blockers.push((index, false, operator, nested_bodies.len()));
            }
            // A conditional OPERATOR only matters in the fence's own statement: the `?` of a
            // completed `bool b = ready ? x : y;` cannot make a later fence conditional, and
            // rejecting it was a false rejection (round-8 A6), while `ready && __threadfence()`
            // or `ready ? __threadfence() : ...` puts the fence in the same statement and stays
            // rejected. The keywords and the early exits keep the conservative reading, because
            // nothing here bounds a branch or a loop standing in front of the fence. The reason is a
            // limitation, not a claim about the path: `if (observed != 0u) { } __threadfence();`
            // compiles to the baseline cubin, so saying the fence is not guaranteed to run would be
            // false (round-11 A2). It is reported at the fence, or as the missing-fence verdict.
        }
        if token.text == ";" {
            semicolons.push((index, nested_bodies.len()));
        }
        visited_code.push(index);
        index += 1;
    }
    // No top-level fence. A nested one was seen, so the reason names it - the same reason the
    // structural check gives for the same body - and only then the plain "no fence" case.
    if let Some(line) = conditional_fence {
        return Err(rejection(format!(
            "line {line}: the fence sits inside a preprocessor conditional, so it is not guaranteed to run"
        )));
    }
    if let Some(nested) = nested_fence {
        return Err(rejection(format!(
            "line {}: the __threadfence in {ACQUIRE_HELPER} sits at a nested brace depth, inside a body that is not the helper's straight-line path - a branch, a loop, a try block, a requires-expression, a lambda or a type body - so a path through the helper can reach the caller without the fence; the acquire has to be a top-level statement after the poll",
            tokens[nested].physical_line
        )));
    }
    Err(rejection(format!(
        "no reachable __threadfence() follows the poll of {RAW_POLL}, so the observing path has no \
         acquire"
    )))
}

/// The index just after the `{` of the block that encloses the token at `index`, or 0 when the token
/// is not inside a brace.
///
/// The reachability lint reads the poll's OWN block before the poll, so control flow in an unrelated
/// function or block above it is not attributed to the poll (round-12 A1).
fn enclosing_block_start(tokens: &[CodeToken], index: usize) -> usize {
    let mut depth = 0i32;
    let mut cursor = index;
    while let Some(previous) = cursor.checked_sub(1) {
        match tokens[previous].text.as_str() {
            "}" => depth += 1,
            "{" if depth == 0 => return previous + 1,
            "{" => depth -= 1,
            _ => {}
        }
        cursor = previous;
    }
    0
}

/// The name of the helper that carries the acquire with the poll. This is where the guarantee
/// lives: an observing path names one function whose body never changes shape.
const ACQUIRE_HELPER: &str = "spark_wait_for_flag_acquire";
/// The raw poll. It stays available to the publisher side, which needs the bounded wait without an
/// acquire; a bare call to it in a kernel translation unit is exactly what half one below rejects.
const RAW_POLL: &str = "spark_wait_for_flag";

/// The shipped observation, spelled the way the shared header spells it. It is an ILLUSTRATION in the
/// messages and never a substitution: the arguments are not part of the rule (round-17 A2), so a helper
/// whose parameters are named differently is never told to write `flag` and `deadline`.
const CANONICAL_POLL_CALL: &str = "spark_wait_for_flag(flag, deadline)";

/// Whether the mention of the raw poll at `index` is a CALL of it. This is the whole rule inside the
/// helper's body, and it is the narrowest statement the scan can make honestly.
///
/// The scan certifies one thing: the designator is CALLED - an argument list follows it, and nothing
/// marks the same tokens as a declaration of something named like the poll. It does NOT read the
/// arguments (round-17 A2: `spark_wait_for_flag(f, d)` under renamed parameters,
/// `spark_wait_for_flag((flag), (deadline))` and a cast argument are the same call, and each compiles
/// the header to the baseline cubin), and it does NOT read the expression the call stands in
/// (round-17 A1: `!poll(...)`, `0u == poll(...)`, `(void) poll(...)` and a bare call each EXECUTE
/// the poll, and each is listed as accepted with its measured cubin in the module documentation).
///
/// Round 16 read a canonical spelling here instead, and the claim that it did so exactly was false
/// (round-17 root): a scan that starts at the name cannot see the prefix at all, so the prefix is not
/// part of the rule. What remains - is this a call rather than a declaration - is what
/// [`a_type_stands_before`] answers.
fn the_mention_is_a_call_of_the_poll(
    tokens: &[CodeToken],
    index: usize,
    macros: &MacroScope,
    directives: &[bool],
) -> bool {
    if designator_is_a_call(tokens, index)
        && !a_type_stands_before(tokens, index, macros, directives)
    {
        return true;
    }
    // Round-20 A3: a macro whose replacement list is the whole call `spark_wait_for_flag(flag,
    // deadline)` IS the observation at its own use - `const uint32_t observed = P;` calls the poll - so
    // the mention does not need an argument list of its own. The arguments are not read either way.
    macros
        .replacement_tokens(tokens[index].text.as_str())
        .is_some_and(replacement_is_the_call)
}

/// Whether the token at `index` names the raw poll: the designator itself, or a macro of this source
/// whose replacement list is exactly the designator (one level: `#define POLL_ALIAS
/// spark_wait_for_flag`, round-17 A4). An alias is a normal way to write the call and nvcc compiles it
/// to the baseline cubin, so counting it as a mention keeps the verdict about the text the compiler
/// sees rather than about the spelling the guard prefers.
fn token_names_the_raw_poll(tokens: &[CodeToken], index: usize, macros: &MacroScope) -> bool {
    let text = tokens[index].text.as_str();
    text == RAW_POLL
        || macros.replacement_tokens(text).is_some_and(|body| {
            replacement_is_the_designator(body) || replacement_is_the_call(body)
        })
}

/// The index just past a designator of the raw poll that may itself be parenthesised, or [`None`]:
/// `spark_wait_for_flag`, `(spark_wait_for_flag)`, `((spark_wait_for_flag))`. A pair counts only when
/// the designator is the WHOLE of what it wraps, so `(spark_wait_for_flag)(flag)` answers with the
/// index past the `)` and the argument list is what follows it.
fn poll_designator_end(body: &[String]) -> Option<usize> {
    if body.first().is_some_and(|first| first == RAW_POLL) {
        return Some(1);
    }
    if body.len() < 3 || body.first().is_none_or(|first| first != "(") {
        return None;
    }
    let mut depth = 0usize;
    for (position, token) in body.iter().enumerate() {
        match token.as_str() {
            "(" => depth += 1,
            ")" => {
                if depth == 0 {
                    return None;
                }
                depth -= 1;
                if depth == 0 {
                    return (poll_designator_end(&body[1..position]) == Some(position - 1))
                        .then_some(position + 1);
                }
            }
            _ => {}
        }
    }
    None
}

/// Whether a macro's replacement list is the WHOLE canonical call of the poll
/// (`#define P spark_wait_for_flag(flag, deadline)`, round-20 A3). A use of it is the observation
/// itself, so `const uint32_t observed = P;` is a call of the poll rather than a mention with no
/// argument list. The arguments are not read, exactly as they are not read for a literal call; the rule
/// is that the replacement starts with the designator and its argument list closes at the end of the
/// replacement list, so the call is the whole of what the macro expands to.
///
/// Round-22 A2: the designator may be parenthesised on its own - `#define P
/// (spark_wait_for_flag)(flag, deadline)` - which is the same call as the bare name and compiles to the
/// same baseline cubin (70432 / 4 / 605321041ac5ec1e, measured). Neither rule read it: the designator rule
/// wants the name to be the WHOLE replacement list, and this rule wanted the bare name to come first,
/// so an ordinary spelling of the call fell between them and was refused.
/// The slice inside one pair of parentheses that wraps the WHOLE slice, or [`None`] when the slice is
/// not exactly such a pair: the first token opens and its matching `)` is the last token. A pair that
/// closes early (`(a) + (b)`) is not a wrapper, and a slice that starts with `)` or never balances is
/// rejected rather than unwrapped.
fn pair_wrapping(body: &[String]) -> Option<&[String]> {
    if body.len() < 3 || body.first().is_none_or(|first| first != "(") {
        return None;
    }
    let mut depth = 0usize;
    for (position, token) in body.iter().enumerate() {
        match token.as_str() {
            "(" => depth += 1,
            ")" => {
                if depth == 0 {
                    return None;
                }
                depth -= 1;
                if depth == 0 {
                    return (position + 1 == body.len()).then_some(&body[1..position]);
                }
            }
            _ => {}
        }
    }
    None
}

fn replacement_is_the_call(body: &[String]) -> bool {
    // Round-21 A3: any number of balanced pairs around the WHOLE replacement list are unwrapped first,
    // the way the designator rule below unwraps them, because
    // `#define P (spark_wait_for_flag(flag, deadline))` then `P;` is the same observation and nvcc
    // compiles it to the baseline cubin. This check refused it until round 21.
    let mut inner: &[String] = body;
    while let Some(unwrapped) = pair_wrapping(inner) {
        inner = unwrapped;
    }
    let Some(designator_end) = poll_designator_end(inner) else {
        return false;
    };
    let rest = &inner[designator_end..];
    if rest.first().is_none_or(|token| token != "(") {
        return false;
    }
    let mut depth = 0usize;
    for (position, token) in rest.iter().enumerate() {
        match token.as_str() {
            "(" => depth += 1,
            ")" => {
                if depth == 0 {
                    return false;
                }
                depth -= 1;
                if depth == 0 {
                    return position == rest.len() - 1;
                }
            }
            _ => {}
        }
    }
    false
}

/// Whether a macro's replacement list is the raw poll's DESIGNATOR, so a use of the macro is a mention
/// of the poll: the name itself, the name in ANY number of balanced pairs of parentheses
/// (`#define P (spark_wait_for_flag)`, `#define P ((spark_wait_for_flag))`), and any of those
/// dereferenced or addressed inside a pair (`(*spark_wait_for_flag)`, `(&spark_wait_for_flag)`).
///
/// Round-19 A5 introduced this for one pair of parentheses; round-20 A3 unwraps them all, because
/// `(( ... ))` is an ordinary thing to write and nvcc compiles it to the baseline cubin. The `*` or
/// `&` only counts when at least one pair of parentheses was there, because
/// `&spark_wait_for_flag(flag, deadline)` is the address of a call and is not the designator of one.
fn replacement_is_the_designator(body: &[String]) -> bool {
    let mut inner: &[String] = body;
    let mut parenthesised = false;
    while inner.len() >= 3
        && inner.first().is_some_and(|first| first == "(")
        && inner.last().is_some_and(|last| last == ")")
    {
        inner = &inner[1..inner.len() - 1];
        parenthesised = true;
    }
    let named = match inner {
        [wrapper, rest @ ..] if parenthesised && (wrapper == "*" || wrapper == "&") => rest,
        _ => inner,
    };
    named.len() == 1 && named[0] == RAW_POLL
}

/// The mentions of the raw poll inside `[open, close)` that are compiled code: every occurrence of the
/// name that is not a token of a preprocessor line and not inside a region the guard can see is
/// disabled. Comments are not tokens, so a mention in a comment is not one.
///
/// Both checks count mentions rather than calls (round 15): the body must mention the poll exactly
/// once and that mention must be canonical, so a declaration, a function-pointer declarator, an
/// attribute between the type and the name, a macro-defined return type or a wrapped call all count
/// as the one mention and are then refused as shapes the scan does not certify.
fn compiled_raw_poll_mentions(tokens: &[CodeToken], open: usize, close: usize) -> Vec<usize> {
    let active = active_code_mask(tokens);
    raw_poll_mentions(tokens, open, close)
        .into_iter()
        .filter(|&index| active[index])
        .collect()
}

/// The mentions of the raw poll inside `[open, close)` that are not in a region the guard can see is
/// disabled: what the STRUCTURAL check counts, so a mention inside a conditional it cannot resolve is
/// still counted and then refused by the conditional rule that names it. The lint reads the narrower
/// set above, which is the documented asymmetry: the structural check refuses that body and says the
/// observation may not be compiled, the lint reads no compiled poll at all.
fn raw_poll_mentions(tokens: &[CodeToken], open: usize, close: usize) -> Vec<usize> {
    let mut conditionals: Vec<Conditional> = conditional_stack_before(tokens, open);
    // The macro scope is advanced with the walk because a mention may be an ALIAS of the designator
    // (round-17 A4) and the preprocessor is positional: `#define POLL_ALIAS spark_wait_for_flag` only
    // counts after its own line.
    let mut macros = MacroScope::from_tokens(tokens);
    let mut mentions = Vec::new();
    let mut index = open;
    while index < close {
        if tokens[index].text == "#" {
            update_conditionals(tokens, index, &mut conditionals);
            index = end_of_directive(tokens, index);
            continue;
        }
        macros.advance_to(index);
        // A region the guard can see is disabled (#if 0) is not compiled code at all, so a mention
        // there is not one; a region it cannot resolve IS counted, and then refused by the caller's
        // own conditional rule, which says so in its message.
        if !definitely_disabled(&conditionals) && token_names_the_raw_poll(tokens, index, &macros) {
            mentions.push(index);
        }
        index += 1;
    }
    mentions
}

/// The structural half of the acquire guarantee: no kernel translation unit calls the raw poll.
///
/// This is the guarantee; [`acquire_fence_after_poll`] is the belt and braces beside it, and the
/// difference between them is the point. The reachability lint asks "is there a fence after this
/// poll, in this text", which is a question a call site can answer wrongly - and did, once per
/// review round - because when the fence is written at the call site the answer depends on the call
/// site's shape. This check removes the question: the observing path must name
/// `spark_wait_for_flag_acquire`, whose body is checked by
/// [`acquire_helper_fences_unconditionally`], so the fence is part of the expansion the call site
/// gets, in every spelling of that call - plain, parenthesised or dereferenced. Replacing the call
/// itself (a `#define` of the helper's name, or a second overload that wins overload resolution) is
/// what [`guarded_names_are_not_redefined`] and [`the_acquiring_poll_is_defined_exactly_once`]
/// reject; a text scan cannot see past a macro it does not read, and the module documentation
/// enumerates what is left.
///
/// The rule is deliberately about kernel translation units and not about the header: the raw poll is
/// correct on the publisher side, which only needs the wait.
///
/// A DECLARATION of the raw poll is accepted and ignored, because a declaration is not an observation
/// and cannot change what any body does: a plain prototype, a function-pointer declaration and one with
/// a `noexcept` specifier all pass - the first two shapes and the specifier are text nvcc accepts, and
/// round-15 A6 found the function-pointer and `noexcept` spellings reported as bare calls. What is
/// rejected is a CALL of the raw poll - and a definition of it would be one too, which
/// [`guarded_names_are_not_redefined`] refuses separately. A call through a function pointer,
/// `(*name)(args)`, is token for token the function-pointer declaration, so this scan cannot separate
/// the two; the module documentation states that limit, and the guarantee it could weaken is carried by
/// the helper's own body.
///
/// # Errors
/// Returns the file and line of the first bare poll.
pub fn observing_paths_call_the_acquiring_poll(kernels: &[(&str, &str)]) -> Result<(), String> {
    for (name, source) in kernels {
        let tokens = code_tokens_of(source);
        let directives = directive_token_mask(&tokens);
        let mut macros = MacroScope::from_tokens(&tokens);
        for (index, token) in tokens.iter().enumerate() {
            macros.advance_to(index);
            if token.text == RAW_POLL
                && designator_is_a_call(&tokens, index)
                && !is_a_declaration_of(&tokens, index, &macros, &directives)
            {
                return Err(rejection(format!(
                    "{name}:{}: a bare {RAW_POLL} call is the raw poll; an observing path must call \
                     {ACQUIRE_HELPER}, whose body carries the acquire fence in the same expansion; \
                    ",
                    token.physical_line
                )));
            }
        }
    }
    Ok(())
}

/// The token range of a function body's interior, by brace depth, or None when there is no such
/// definition with a body.
///
/// The definition is the occurrence of the name whose ARGUMENT LIST is followed by the body's `{`;
/// a call of the same name and a prototype are not bodies, and scanning forward from either to the
/// next brace anywhere in the file returned an unrelated range - which made the lint look for the
/// poll outside the helper it had found (round-14 A-1).
fn function_body_tokens(tokens: &[CodeToken], name: &str) -> Option<(usize, usize)> {
    for (index, token) in tokens.iter().enumerate() {
        if token.text != name || tokens.get(index + 1).map(|next| next.text.as_str()) != Some("(") {
            continue;
        }
        let mut cursor = index + 1;
        let mut parens = 0usize;
        while cursor < tokens.len() {
            match tokens[cursor].text.as_str() {
                "(" => parens += 1,
                ")" => {
                    parens -= 1;
                    if parens == 0 {
                        break;
                    }
                }
                _ => {}
            }
            cursor += 1;
        }
        cursor += 1;
        if cursor >= tokens.len() || tokens[cursor].text != "{" {
            continue;
        }
        let open = cursor;
        let mut depth = 0usize;
        while cursor < tokens.len() {
            match tokens[cursor].text.as_str() {
                "{" => depth += 1,
                "}" => {
                    depth -= 1;
                    if depth == 0 {
                        return Some((open + 1, cursor));
                    }
                }
                _ => {}
            }
            cursor += 1;
        }
        return None;
    }
    None
}

/// The structural half of the acquire guarantee, on the helper's own body: the body must mention the
/// raw poll exactly once, and that mention must be a CALL of it - an argument list follows the
/// designator and no type stands in front of it. The call's arguments are not read, and neither is the
/// expression it stands in (round 17; the module documentation lists what is accepted with each shape's
/// measured cubin). What is refused is a mention that is not a call: a declaration in any return-type
/// spelling, a function-pointer declarator, an attribute between the type and the name. An
/// unconditional `__threadfence();` must follow as a top-level statement of the same body.
///
/// Why this is stronger than the reachability lint: the lint re-derives a fence's reachability from
/// the text around a poll, so it has to enumerate every way a source can make a statement
/// conditional, and its comment has to admit what the enumeration misses. This check reads one body
/// that is written once, so no enumeration of shapes is needed: it refuses a control token or a
/// non-returning call at ANY brace depth between the poll and the fence, and it accepts a fence in a
/// bare block or behind a label, which are unconditional too. The two halves together state the
/// guarantee as: the observing path names the helper, and the helper's expansion is poll then fence
/// with nothing compiled between them that can intervene.
///
/// A definitely-disabled region (#if 0) between the poll and the fence is ignored rather than
/// obeyed, exactly as in the reachability lint, because none of it executes; a fence inside a
/// conditional the guard cannot resolve is rejected, because the compiler may not take it. A macro
/// whose body hides a control keyword IS analysed here too, in the lint's own words (round-20 A1): the
/// round-19 review found that the lint refused `#define BAIL return` then `BAIL;` before the poll while
/// this check accepted it, and nvcc compiles that text to 68576 / 2 / bfd9e8e177f4f76d rather than to the
/// 70432 / 4 / 605321041ac5ec1e baseline, so accepting it was the wrong verdict rather than a stricter one.
///
/// # Errors
/// Returns the line of the defect and the reason, so a test can assert on each mutation.
pub fn acquire_helper_fences_unconditionally(header: &str) -> Result<(), String> {
    let tokens = code_tokens_of(header);
    // A using-declaration that re-binds a guarded name is the same rewrite as a #define, and it has
    // to be refused here because the bare call in this body is the thing it redirects (round-10 A-2):
    // `using X::__threadfence;` beside `namespace X { void __threadfence() {} }` made the acquire
    // call X's no-op, and nvcc compiled that kernel to 70264 bytes / 2 PTB membar lines / sha
    // 42454ad1... against the 70432 / 4 / 605321041ac5ec1e baseline.
    if let Some((_, line, name, target)) =
        guarded_names_brought_into_scope(&[(COMMON_HEADER_NAME, header)])
    {
        return Err(rejection(using_declaration_rejection(
            &format!("line {line}"),
            &name,
            &target,
        )));
    }
    // The header's own macros, so a name the header defines is not read as the library function it
    // shadows (round-10 A-4), and so a use of a header macro with a control body is refused.
    let mut macros = MacroScope::new(header);
    let (open, close) = function_body_tokens(&tokens, ACQUIRE_HELPER).ok_or_else(|| {
        rejection(format!(
            "{ACQUIRE_HELPER} must be defined as a function body in the shared header: the acquiring poll is what makes the fence structural rather than a call-site promise"
        ))
    })?;
    let mut conditionals: Vec<Conditional> = conditional_stack_before(&tokens, open);
    let mut depth = 0usize;
    // Every poll in the body, with its brace depth. The count is part of the check: one observation
    // pairs with one acquire, and recording only the first poll let `poll; fence; poll;` through.
    let mut polls: Vec<(usize, usize)> = Vec::new();
    // The body's mentions of the raw poll, in every shape. The body must mention it exactly once and
    // that mention must be a CALL of it (round 17, corrected in round 18): a designator followed by an
    // argument list, whatever the designator's parentheses and whatever expression the call stands in.
    // A mention that is not a call - a declaration, a function-pointer declarator, an attribute
    // between the type and the name - is refused rather than read as a call, because those bodies
    // compile to something other than the baseline (70600 / 3 / 1f2503fd7ecaf189, measured).
    let mentions = raw_poll_mentions(&tokens, open, close);
    // The fence and the index its own statement starts at (a global `::` belongs to the designator).
    let mut top_fence: Option<(usize, usize)> = None;
    // The code tokens of the body in order. Directives are tokens in this stream but produce no
    // statements, so a fence that follows an #endif on its own line still counts as a bare statement;
    // keeping the visited indices also lets `(__threadfence)();` be read as the call it is rather
    // than as a fence preceded by a `(` (round 7), and lets the statement-boundary rule see a label
    // (round 8).
    let mut code_history: Vec<usize> = Vec::new();
    // For each open brace of the body, whether it opens a BARE block rather than the body of an if,
    // else, for, while, do, switch or try. A fence alone in a bare block runs unconditionally - the
    // round-8 review found `{ __threadfence(); }` rejected with the reason for a branch that was not
    // there - while a fence in a branch or a loop body does not.
    let mut bare_blocks: Vec<bool> = Vec::new();
    let mut nested_fence: Option<usize> = None;
    // A fence the preprocessor may not compile: a candidate only while nothing unconditional has been
    // found after the poll, because a later unconditional fence is the acquire and makes it irrelevant
    // (round-24 A2).
    let mut conditional_fence: Option<(usize, usize)> = None;
    // Everything that can keep the fence from running: the index, whether it never returns, whether it
    // is a conditional OPERATOR (scoped to the fence's own statement below), and the brace depth the
    // blocker sits at, because a `;` nested inside a brace does not end that statement (round-10
    // A-1). Every candidate is kept, because one of them may turn out to be an operator in an earlier,
    // completed statement that must not be reported.
    let mut blockers: Vec<(usize, bool, bool, usize)> = Vec::new();
    // Every USE of a macro that can bound the path before the poll - `#define BAIL return` then `BAIL;`.
    // Round-20 A1: the lint refused these and this check accepted them, which was the wrong verdict: nvcc
    // compiles that text to 68576 / 2 / bfd9e8e177f4f76d rather than to the 70432 / 4 / 605321041ac5ec1e baseline.
    let mut suspicious_macro_uses: Vec<usize> = Vec::new();
    // Every `;` of the body with its brace depth: the separator that ends the fence's statement is a
    // `;` at the same depth as the operator, and the `;` of `int z = 0;` inside a statement
    // expression is at a deeper one.
    let mut semicolons: Vec<(usize, usize)> = Vec::new();
    let mut cursor = open;
    while cursor < close {
        let text = tokens[cursor].text.as_str();
        macros.advance_to(cursor);
        if text == "#" {
            update_conditionals(&tokens, cursor, &mut conditionals);
            cursor = end_of_directive(&tokens, cursor);
            continue;
        }
        if text == "{" {
            // A `)` before the brace is the end of an if/for/while/switch condition and `else`,
            // `do` and `try` name a body directly; anything else opens a bare block, which executes
            // unconditionally like the statements around it. A lambda's or a type's body is not bare
            // either (round-10 A-3): the fence in `auto l = [] { __threadfence(); };` runs only if
            // the closure is called, and nvcc compiled that helper to 70264 bytes with 2 PTB membar
            // lines (sha 42454ad1...) while the called version was byte-identical to the 70432 / 4 /
            // 605321041ac5ec1e baseline.
            // `requires` is in the list because nothing inside a requires-expression is executed,
            // so a fence there is not the acquire even though it sits in braces (round-11 A1).
            // An attribute standing between a head and its brace does not change which body this
            // is, and its closing bracket is not one of the tokens below: the read steps over the group
            // first, so `try [[likely]] { ... }` is still not a bare block (round-36 A1).
            let bare = the_token_before_the_attributes(&tokens, code_history.last().copied())
                .is_none_or(|previous| {
                    !matches!(
                        tokens[previous].text.as_str(),
                        ")" | "else" | "do" | "try" | "requires"
                    )
                })
                && !opens_a_lambda_or_a_type_body(&tokens, &code_history);
            bare_blocks.push(bare);
            depth += 1;
            code_history.push(cursor);
            cursor += 1;
            continue;
        }
        if text == "}" {
            bare_blocks.pop();
            depth = depth.saturating_sub(1);
            code_history.push(cursor);
            cursor += 1;
            continue;
        }
        if definitely_disabled(&conditionals) {
            cursor += 1;
            continue;
        }
        if text == "__threadfence" {
            // Only the unqualified or globally qualified function is the barrier: `X::__threadfence()`
            // and `x.__threadfence()` name a namespace's or an object's function, and reading their
            // `::` as a label colon is how a namespaced no-op passed for the acquire (round-8 A3).
            // For those the branch is skipped and the body is reported as having no barrier, which is
            // the truthful verdict.
            let designator = call_argument_list(&tokens, cursor).map_or(cursor, |(start, _)| start);
            if let Some(designator) = global_fence_designator(&tokens, &code_history, designator) {
                // Round-24 A2: a fence the guard cannot resolve is a FALLBACK candidate, not a refusal.
                // `poll; __threadfence(); #if M4_FLAG __threadfence(); #endif; return observed;`
                // preprocesses without the define to exactly the shipped body - 70432 / 4 / 605321041ac5ec1e,
                // the committed baseline - so the unconditional fence is the acquire and the later
                // conditional one is irrelevant. Refusing on sight made this check refuse a baseline
                // body while the lint accepted it, with a reason that was simply false. This candidate
                // is reported only when no unconditional top-level fence is found after the poll.
                if !definitely_compiled(&conditionals) {
                    conditional_fence.get_or_insert((cursor, designator));
                    code_history.push(cursor);
                    cursor += 1;
                    continue;
                }
                // Every enclosing brace must be a bare block for the fence to run unconditionally: a
                // bare block around the fence is not a branch or a loop, so `{ __threadfence(); }`
                // qualifies, while anything inside `if (...) { ... }` does not.
                //
                // Round-22 A1: the acquire is the first fence AFTER the poll. A helper body may carry a
                // fence BEFORE the poll - it is not the acquire of that observation, and taking it made
                // this check refuse `__threadfence(); spark_wait_for_flag(flag, deadline);
                // __threadfence(); return observed;` with "precedes the poll" while the lint accepted
                // it. The reviewer measured that shape: the extra fence before the poll is the only
                // delta and it is 70448 bytes / 5 PTB membar lines / sha 488031bf, so the acquire after
                // the poll is present and the LINT is right. Fences before the poll are ignored here,
                // exactly as the lint ignores them.
                if bare_blocks.iter().all(|&bare| bare) {
                    if mentions.iter().any(|&mention| mention < cursor) {
                        top_fence.get_or_insert((cursor, designator));
                    }
                } else {
                    nested_fence.get_or_insert(cursor);
                }
                code_history.push(cursor);
                cursor += 1;
                continue;
            }
        }
        if mentions.contains(&cursor) {
            if !definitely_compiled(&conditionals) {
                return Err(rejection(format!(
                    "line {}: the poll in {ACQUIRE_HELPER} sits inside a preprocessor conditional, so it is not guaranteed to be compiled",
                    tokens[cursor].physical_line
                )));
            }
            polls.push((cursor, depth));
            code_history.push(cursor);
            // Round-23 root: the scan resumes after the poll's ARGUMENT LIST, exactly where the lint's
            // walk starts. It used to resume at the designator itself, so a fence inside the arguments -
            // which executes BEFORE the call - was read as the acquire of that observation, and the
            // verdict stopped being a function of the compiled shape: the statement-expression fence
            // `spark_wait_for_flag(({ __threadfence(); flag; }), deadline); return observed;` was
            // ACCEPTED here and refused by the lint, while it compiles to 70280 / 3 / 18b66cc5f312046c, the same
            // cubin as the plain fence-before-the-poll body both checks refuse. With the real acquire
            // after it, `spark_wait_for_flag((__threadfence(), flag), deadline); __threadfence();` was
            // REFUSED here and accepted by the lint, and it compiles to 70448 / 5 / 488031bfabdebb09, the cubin
            // of the accepted fence-before-and-after shape. A fence in the arguments is never the
            // acquire, and nothing inside the arguments stands between the poll and its fence.
            let (after_the_call, argument_span) = call_argument_list(&tokens, cursor)
                .and_then(|(_, open)| {
                    matching_close_bracket(&tokens, open)
                        .map(|close| (close + 1, Some((open, close))))
                })
                .unwrap_or((cursor + 1, None));
            if let Some((open, close)) = argument_span {
                // Round-24 A1: the span is not read for FENCES - a fence written in the arguments runs
                // before the call, so it is never the acquire of that observation - but it IS read for
                // control flow. Round 23 skipped it entirely, which made the scan stop reading a
                // `return`, a control keyword or a non-returning call inside the poll's own arguments:
                // `spark_wait_for_flag((__trap(), flag), deadline); __threadfence(); return observed;`
                // was ACCEPTED although it compiles to 69536 / 4 / 5826994128b34ed0, the cubin of the
                // trap-before-the-poll statement body both checks refuse, and `({ return 0u; flag; })`
                // as the first argument compiles to 70464 / 2 / 2b56c510b5f4a945. The lint reads the same span
                // now, so the two cannot diverge on the shape the compiler sees.
                // Round-25 A3: the reading is the lint's reading. A lambda body is skipped - a
                // `return` there exits the lambda, not the helper, and refusing
                // `spark_wait_for_flag(([&]{ return flag; })(), deadline);` contradicted the
                // documentation, which has promised since round 10 that a lambda's return is not a
                // blocker. A bare `if`/`while`/`for`/`switch` is not a blocker either: only an EXIT
                // (`return`, `goto`, `throw`, `co_return`) or a call that never returns is, so
                // `if (ready) return 0u;` still blocks while `if (0) { }` does not - and each of those
                // bodies compiles to the baseline with the acquire present.
                let mut inner = open + 1;
                while inner < close {
                    let text = tokens[inner].text.as_str();
                    // Round-26 A2: a DEFINITELY DISABLED region in the arguments is not read, exactly
                    // as the walks do not read one; a conditional this scan cannot resolve is not
                    // disabled, so a `return` under `#if M4_FLAG` still blocks - under the define it
                    // really runs and the fence after the poll is really dead.
                    if text == "#" {
                        update_conditionals(&tokens, inner, &mut conditionals);
                        inner = end_of_directive(&tokens, inner);
                        continue;
                    }
                    if definitely_disabled(&conditionals) {
                        inner += 1;
                        continue;
                    }
                    macros.advance_to(inner);
                    // Round-26 A1: the narrow backward scan misses a TRAILING RETURN TYPE, so the
                    // walk's own detector decides a body the narrow one does not recognise.
                    if text == "{"
                        && (a_lambda_body_opens_at(&tokens, inner)
                            || opens_a_lambda_or_a_type_body(
                                &tokens,
                                &(cursor..inner).collect::<Vec<usize>>(),
                            ))
                    {
                        inner =
                            matching_close_bracket(&tokens, inner).map_or(inner + 1, |end| end + 1);
                        continue;
                    }
                    let never_returns = NON_RETURNING_CALLS.contains(&text)
                        && is_the_library_non_returning_call(&tokens, inner, &macros);
                    if never_returns || is_an_argument_exit(text) {
                        blockers.push((inner, never_returns, false, depth));
                    } else if macros.is_suspicious(text) {
                        // Round-25 A4: one sentence, the same one the lint's argument-span reading and
                        // both walks use for this reason.
                        return Err(rejection(format!(
                            "line {}: {} is a macro whose body contains a control keyword, a non-returning \
                             call, or another such macro, so this scan does not certify that the fence runs \
                             after it",
                            tokens[inner].physical_line, text
                        )));
                    }
                    inner += 1;
                }
            }
            cursor = after_the_call;
            continue;
        }
        // Anything that ends the fall-through path, or makes execution conditional, stops the fence
        // from being an unconditional acquire. This is recorded at EVERY brace depth: `{ return 0u;
        // }`, `{ if (...) return 0u; }`, `{ abort(); }`, `{ __trap(); }` and `{ goto done; }`
        // between the poll and the fence all leave the helper without its acquire, and recording only
        // depth 0 was the round-8 A1 hole. A bare block that holds nothing but the fence has no such
        // token, so accepting it costs nothing. Every candidate is kept, because one of them may turn
        // out to be an operator in an earlier statement that must not be reported.
        if fence_designator_starting_at(&tokens, cursor).is_none() {
            // Called, not necessarily a statement of its own: `(__builtin_unreachable(), 0u)` and
            // `f(__trap())` never return either, and the fence after them never runs (round-8 A1).
            // Only the LIBRARY function counts: a member, a qualified name, a declaration and a name
            // this header #defines are not the function that never returns (round-10 A-4).
            let never_returns = NON_RETURNING_CALLS.contains(&text)
                && is_the_library_non_returning_call(&tokens, cursor, &macros);
            // Round-27 A2: an exit inside a LAMBDA's body is not this helper's path - it exits the
            // lambda - so it is not a blocker. The lint's pre-poll reading skips the same bodies;
            // refusing `const auto ran = [&]{ return flag; }();` before the poll refused a body nvcc
            // compiles to the baseline.
            if (never_returns || is_control(text))
                && !inside_a_body_the_helper_does_not_own(&tokens, cursor, open)
            {
                let operator = matches!(text, "?" | "&&" | "||" | "&" | "|");
                blockers.push((cursor, never_returns, operator, depth));
            } else if macros.carries_only_operators(text) {
                // Round-20 A1: a use of a macro whose body carries only an operator is a blocker with the
                // OPERATOR's own rule, so the `;` scoping in `relevant` decides it exactly as it does for
                // the literal operator: `MAYBE(spark_wait_for_flag(flag, deadline));` is refused because
                // the poll is the macro's argument and no `;` stands between the use and the observation,
                // while `const bool ran = PRE;` before the poll is not a blocker once that statement has
                // completed.
                blockers.push((cursor, false, true, depth));
            }
            // The same macro-suspicion rule the lint applies to this span, so the two checks refuse the
            // same texts in the same words (round-20 A1) - including the lambda bodies both skip since
            // round 28 A5, which is what keeps the two verdicts equal.
            if macros.is_suspicious(text)
                && !inside_a_body_the_helper_does_not_own(&tokens, cursor, open)
            {
                suspicious_macro_uses.push(cursor);
            }
        }
        if text == ";" {
            semicolons.push((cursor, depth));
        }
        code_history.push(cursor);
        cursor += 1;
    }
    let Some(&(poll, poll_depth)) = polls.first() else {
        return Err(rejection(format!(
            "{ACQUIRE_HELPER} does not mention {RAW_POLL} in its body, so there is no observation for the fence to pair with"
        )));
    };
    // Exactly one observation pairs with the acquire. Two polls with one fence between them is the
    // round-5 E2 shape: the old check recorded only the FIRST poll, so `poll; fence; poll;` was
    // green while the second observation was read with no fence after it.
    // The count comes from the SAME scan the lint counts with (round-23 root): `mentions`, not the
    // walk's `polls`. The walk resumes after the poll's argument list, so in
    // `#define P(spark_wait_for_flag) spark_wait_for_flag` + `P(spark_wait_for_flag)(flag, deadline)`
    // it visits only the macro-name mention and used to report one observation where this scan and the
    // lint both count two spellings - the alias `P` and the designator written as its argument. The
    // verdict was right either way, but the counted number has to be the same number the lint counts.
    if mentions.len() > 1 {
        return Err(rejection(format!(
            "line {} and line {}: {ACQUIRE_HELPER} mentions the poll {} times, so this scan cannot tell which observation the fence pairs with",
            tokens[mentions[0]].physical_line,
            tokens[mentions[1]].physical_line,
            mentions.len()
        )));
    }
    // The one mention has to be a CALL of the poll. What is refused is a mention that is not one - a
    // declaration in any of its return-type spellings, a function-pointer declarator, an attribute
    // between the type and the name - because a name that is only declared is not an observation. The
    // call's arguments and the expression around it are NOT read (round-17 A1/A2): every call executes
    // the poll, and the accepted list in the module documentation carries each shape's measured cubin.
    let directives = directive_token_mask(&tokens);
    let mut at_the_poll = MacroScope::from_tokens(&tokens);
    at_the_poll.advance_to(poll);
    if a_void_type_precedes_the_designator(&tokens, poll) {
        return Err(rejection(void_typed_poll_message(
            tokens[poll].physical_line,
        )));
    }
    if !the_mention_is_a_call_of_the_poll(&tokens, poll, &at_the_poll, &directives) {
        return Err(rejection(format!(
            "line {}: {ACQUIRE_HELPER}'s body mentions {RAW_POLL} in a shape this scan will not certify as a call of it - a name that is only declared is not an observation; the shared header writes the call as `{CANONICAL_POLL_CALL}` (with the helper's own parameter names, since the arguments are not read)",
            tokens[poll].physical_line
        )));
    }
    if poll_depth != 0 {
        return Err(rejection(format!(
            "line {}: the poll in {ACQUIRE_HELPER} is not a top-level statement of its body, so the helper's expansion is not the fixed poll-then-fence shape",
            tokens[poll].physical_line
        )));
    }
    let Some((fence, designator)) = top_fence else {
        if let Some((conditional, _)) = conditional_fence {
            return Err(rejection(format!(
                "line {}: the fence sits inside a preprocessor conditional, so it is not guaranteed to run",
                tokens[conditional].physical_line
            )));
        }
        if let Some(nested) = nested_fence {
            return Err(rejection(format!(
                "line {}: the __threadfence in {ACQUIRE_HELPER} sits at a nested brace depth, inside a body that is not the helper's straight-line path - a branch, a loop, a try block, a requires-expression, a lambda or a type body - so a path through the helper can reach the caller without the fence; the acquire has to be a top-level statement after the poll",
                tokens[nested].physical_line
            )));
        }
        return Err(rejection(format!(
            "no reachable __threadfence() follows the poll of {RAW_POLL}, so the observing path has no acquire"
        )));
    };
    // Round-20 A1 and round-21 A1: a use of a suspicious macro is refused in the LINT's own words,
    // because the lint refuses these texts and this check used to accept some of them. The macro can
    // leave the helper before the acquire runs, and every measured one differs from the 70432 / 4 /
    // 605321041ac5ec1e baseline: `BAIL;` before the poll is 68576 / 2 / bfd9e8e177f4f76d (round 20), and `BAIL;`
    // after it in a later, completed statement is 69072 / 2 / 18f93bdc42897348 (round 21, where this check
    // accepted the text and the lint refused it).
    //
    // The scope is the LINT's, exactly: every use that stands before the FENCE. The lint reads a use
    // before the poll in the poll's block, a use in the poll's own statement, and - in its forward walk
    // from the designator - every use between the poll and the fence. A use after the fence is read by
    // neither check, because the acquire has already run by then. Round 20 scoped this to "before the
    // poll, or in the poll's own statement"; that is what let `poll(...); BAIL; __threadfence();`
    // through while the lint refused it.
    //
    // The words are the lint's own for the position: its pre-poll scan says "the acquire after it
    // runs", its forward walk from the poll says "the fence runs after it". The rule stands BEFORE the
    // fence-placement rules below, because that is where the lint's forward walk meets a use: both then
    // answer a shape like `BAIL, __threadfence();` with the macro sentence rather than with the
    // fence-placement sentence.
    // The lint's poll-statement rule, in the lint's own words (round-22 finding 3). The lint refuses a
    // poll used WITH a suspicious macro - the macro is an argument of the poll, or it stands in the
    // poll's own statement before it - naming that macro and saying the scan does not certify that the
    // fence runs after it. This check answered the same shape with its own prefix and its own tail, so
    // the two messages agreed on the substring and on nothing else; they are whole-message equal now,
    // which is what the matrix asserts. The rule comes first for the lint's own reason (round-13 D1):
    // the macro-use reason would otherwise shadow the poll-statement one.
    if let Some(name) = suspicious_macro_in_poll_statement(&tokens, poll, &at_the_poll, &directives)
    {
        return Err(rejection(format!(
            "line {}: the poll is used with {}, a macro whose body contains a control keyword, a \
             non-returning call, or another such macro, so this scan does not certify that the fence \
             runs after it",
            tokens[poll].physical_line, name
        )));
    }
    if let Some(&index) = suspicious_macro_uses.iter().find(|&&index| index < fence) {
        let tail = if index < poll {
            "the acquire after it runs"
        } else {
            "the fence runs after it"
        };
        return Err(rejection(format!(
            "line {}: {} is a macro whose body contains a control keyword, a non-returning call, or \
             another such macro, so this scan does not certify that {tail}",
            tokens[index].physical_line, tokens[index].text
        )));
    }
    // "The fence follows the poll" is not checked here any more: round-22 A1 made the selection above
    // take only fences that have a poll mention before them, so a fence that precedes the poll is
    // never the candidate and there is nothing left to report about it. What a body with a fence
    // before the poll and none after it gets is the "no reachable __threadfence() follows the poll"
    // refusal below, which is the true reason for that shape.
    if !is_bare_fence_call(&tokens, fence) {
        return Err(rejection(format!(
            "line {}: the __threadfence is not a bare call statement, so it may be guarded",
            tokens[fence].physical_line
        )));
    }
    // The code token before the whole designator, not the raw previous token: a directive is a token
    // in this stream but produces no statement, so a fence on its own line after an #endif is a bare
    // statement like any other. The round-6 limit fixture found that the raw-previous-token version
    // rejected that shape as "part of a larger expression"; round 8 added the label case, and the
    // reachability lint answers it with the same helper so the two checks cannot disagree.
    if !starts_a_statement(&tokens, &code_history, designator) {
        let previous = code_token_before(&code_history, designator)
            .map_or("the start of the block", |code| tokens[code].text.as_str());
        return Err(rejection(format!(
            "line {}: the __threadfence is part of a larger expression that follows {previous:?}, so it may not run",
            tokens[fence].physical_line
        )));
    }
    // A conditional OPERATOR is scoped to the fence's own statement: the `?` of a completed
    // `bool b = ready ? x : y;` in front of a labelled fence cannot make the fence conditional, and
    // reporting it was a false rejection (round-8 A6), while an operator with no `;` between it and
    // the fence is in the fence's own statement and does block. The `;` that ends that statement is a
    // `;` at the operator's own brace depth: the `;` inside a statement expression or a lambda body
    // belongs to the inner statement, and reading it as the end of the operator's statement let
    // `(observed == 0u) && ({ int z = 0; __threadfence(); z; });` through (round-10 A-1, where nvcc
    // dropped a barrier: 38624 bytes / 3 PTB membar lines / sha d64426bf... (pre-change) against the 70432 / 4 /
    // 605321041ac5ec1e baseline). The keywords and the early exits keep the conservative reading, because
    // nothing here bounds a branch or a loop in front of the fence.
    // The poll's own designator, from its first token to the name: `(poll)`, `(*poll)`, `(&poll)` are
    // calls, and the `&` of the last one is not the bitwise operator a blocker scan would read it as
    // (round-18 A1). The lint skips the same span.
    let poll_designator =
        call_argument_list(&tokens, poll).map_or(poll, |(designator, _)| designator);
    let relevant = |&(index, _, operator, blocker_depth): &(usize, bool, bool, usize)| {
        index < fence
            && !(index >= poll_designator && index < poll)
            // Round-18 A2/A3: a control token that OWNS the poll - the `if`, `while`, `switch`, `for`
            // or `catch` whose condition the poll is, or the `?` of the conditional expression the
            // poll stands in - is not a blocker: the fence after that whole statement runs
            // unconditionally. Anything that can take the path away before the poll still counts, and
            // this is what keeps this check's verdict equal to the lint's for `true ? poll(...) : 0u`.
            && !the_control_token_governs_the_poll(&tokens, index, poll)
            // Round-18 A2/A3: a CONTROL token that stands BEFORE the poll is a blocker even when a
            // `;` separates it from the fence, because what it controls is whether the poll runs at
            // all - `if (ready) poll(...);`, `ready && poll(...)` and `return poll(...);` all compile
            // and none of them is the fixed shape. The `;` exclusion is for the operators that merely
            // wrap the call (`!poll(...)`, `0u == poll(...)`), which is why it is written as
            // `operator` rather than "anything". After the poll the `;` rule is untouched: a text
            // like `poll(...); &x;` is still the round-8 case. This makes the structural verdict equal
            // to the lint's for every shape in `measured_texts`, refused and accepted alike.
            && (!operator
                || (index < poll
                    && (is_control_operator(&tokens[index].text)
                        || macros.carries_only_operators(&tokens[index].text))
                    && !inside_a_lambda_capture(&tokens, index)
                    && !a_statement_completed_between(&tokens, index, poll))
                || !semicolons.iter().any(|&(code, semicolon_depth)| {
                    code > index && code < fence && semicolon_depth == blocker_depth
                }))
    };
    if let Some(&(index, never_returns, _, _)) = blockers.iter().find(|blocker| relevant(blocker)) {
        let line = tokens[index].physical_line;
        let text = tokens[index].text.as_str();
        if never_returns {
            if index > poll {
                return Err(rejection(format!(
                    "line {line}: {text} does not return and stands between the poll and the acquire in {ACQUIRE_HELPER}, so this scan does not certify that the acquire runs on every path"
                )));
            }
            return Err(rejection(format!(
                "line {line}: {text} does not return, so this scan does not certify that the acquire after it runs"
            )));
        }
        if index > poll {
            return Err(rejection(format!(
                "line {line}: {text:?} stands between the poll and the acquire in {ACQUIRE_HELPER}, and this scan reads placement rather than a construct's body, so it does not certify the acquire on every path"
            )));
        }
        if is_an_exit_keyword(text) {
            // Round-20 A6: the acquire after a `return` (or a `break`, `goto`, `throw`, ...) is
            // UNREACHABLE, which is not the same fact as "it may not be reached": `return poll(...);` is
            // an unconditional call whose fence is dead code.
            return Err(rejection(format!(
                "line {line}: {text:?} stands before the poll, so the acquire after it is unreachable"
            )));
        }
        return Err(rejection(format!(
            "line {line}: {text:?} stands before the poll, so the acquire after it may not be reached"
        )));
    }
    Ok(())
}

/// A `#define` of one of these three names rewrites the acquire out of the translation unit without
/// touching a single call site. The reviewer's round-5 mutations 1 and 2 are exactly the first and
/// the third of them.
const GUARDED_NAMES: [&str; 3] = [ACQUIRE_HELPER, RAW_POLL, "__threadfence"];

/// The file name of the shared header: the one source that may define the acquiring poll.
const COMMON_HEADER_NAME: &str = "m4_common_sm121.cuh";

/// The first `using` that re-binds one of the guarded names, as `(file, line, name, namespace)`.
///
/// A using-declaration rewrites what an unqualified name denotes exactly as a `#define` does, and
/// the round-10 A-2 hole shows why that matters: with `namespace X { __device__ void
/// __threadfence() {} }` and `using X::__threadfence;` beside the bare call, the helper's
/// `__threadfence();` calls X's no-op, and nvcc compiled the kernel to 70264 bytes with 2 PTB membar
/// lines (sha 42454ad1...) where the baseline is 70432 / 4 / 605321041ac5ec1e. Both checks accepted it,
/// because the text of the call is unchanged.
///
/// A using-DIRECTIVE cannot be resolved by a text scanner in general, so it is caught in the case the
/// scanner can see: `using namespace X;` where a scanned source defines `namespace X` with a member
/// carrying a guarded name. The documentation names the remaining case - a namespace from a header
/// outside the scan - as a limit, alongside the macro that comes from an included header.
fn guarded_names_brought_into_scope(
    sources: &[(&str, &str)],
) -> Option<(String, usize, String, UsingTarget)> {
    let mut namespaces: Vec<(String, String)> = Vec::new();
    for &(_, source) in sources {
        namespaces.extend(namespaces_with_guarded_members(source));
    }
    for &(file, source) in sources {
        let tokens = code_tokens_of(source);
        let mut index = 0usize;
        while index < tokens.len() {
            if tokens[index].text != "using" {
                index += 1;
                continue;
            }
            let mut cursor = index + 1;
            let directive = tokens
                .get(cursor)
                .is_some_and(|token| token.text == "namespace");
            if directive {
                cursor += 1;
            }
            let line = tokens[index].physical_line;
            while cursor < tokens.len() {
                let text = tokens[cursor].text.as_str();
                if matches!(text, ";" | "{" | "}" | "(" | ")") {
                    break;
                }
                if GUARDED_NAMES.contains(&text) {
                    // A using-DECLARATION reports which spelling it is, so the message can tell a
                    // re-bound name (using X::__threadfence;, the round-10 A-2 hole) from the global
                    // barrier itself (using ::__threadfence;, a policy rejection because nvcc still
                    // compiles it to the baseline cubin) from the bare spelling (using
                    // __threadfence;, which nvcc refuses - round-11 A5, corrected in round-12 A5).
                    let target = if directive {
                        UsingTarget::Directive(text.to_string())
                    } else {
                        using_declaration_target(&tokens, cursor)
                    };
                    return Some((file.to_string(), line, text.to_string(), target));
                }
                if let Some((_, member)) = directive
                    .then(|| namespaces.iter().find(|(namespace, _)| namespace == text))
                    .flatten()
                {
                    return Some((
                        file.to_string(),
                        line,
                        member.clone(),
                        UsingTarget::Directive(text.to_string()),
                    ));
                }
                cursor += 1;
            }
            index += 1;
        }
    }
    None
}

/// What a using-declaration names in front of a guarded name at `index`.
///
/// A doubled colon with an identifier in front of it is a member of that namespace (`using
/// X::__threadfence;`), the case where the meaning really changes. A doubled colon with nothing in
/// front of it is the global barrier itself (`using ::__threadfence;`), which nvcc still compiles to
/// the baseline cubin. No colon at all is the bare spelling (`using __threadfence;`), which nvcc
/// refuses outright (round-12 A5). `using` is an identifier to the tokenizer, so it is excluded from
/// the namespace slot: reporting the namespace "using" is the nonsense the first spelling of this
/// helper printed.
fn using_declaration_target(tokens: &[CodeToken], index: usize) -> UsingTarget {
    let after = tokens.get(index + 1).map(|token| token.text.as_str());
    // An alias-declaration is not a using-declaration at all, and it RE-BINDS the name in its scope,
    // so it is classified before the qualified forms (round-13 A-3).
    if after == Some("=") {
        return UsingTarget::Alias;
    }
    // `using X::template __threadfence;` - the qualified name is preceded by the `template`
    // keyword, which makes it a template rather than the barrier. This is checked first because the
    // name's own neighbour is the keyword, not the `::` (round-13 A-3).
    if index >= 4
        && tokens[index - 1].text == "template"
        && tokens[index - 2].text == ":"
        && tokens[index - 3].text == ":"
        && is_identifier(&tokens[index - 4].text)
        && !matches!(tokens[index - 4].text.as_str(), "using" | "namespace")
    {
        return UsingTarget::TemplateQualified(tokens[index - 4].text.clone());
    }
    if index >= 2 && tokens[index - 1].text == ":" && tokens[index - 2].text == ":" {
        if index >= 3
            && is_identifier(&tokens[index - 3].text)
            && !matches!(tokens[index - 3].text.as_str(), "using" | "namespace")
        {
            // `using X::__threadfence<int>;` names a specialization, which is not a valid
            // using-declaration, so it is not the member re-binding the next arm describes.
            if after == Some("<") {
                return UsingTarget::TemplateId;
            }
            return UsingTarget::Member(tokens[index - 3].text.clone());
        }
        if after == Some("<") {
            return UsingTarget::TemplateId;
        }
        return UsingTarget::Global;
    }
    if after == Some("<") {
        return UsingTarget::TemplateId;
    }
    UsingTarget::Unqualified
}

/// The namespaces this source defines whose body mentions one of the guarded names, as
/// `(namespace, guarded name)`. A using-directive for one of them can make the unqualified name
/// resolve to that member, which is the case a text scanner can see.
fn namespaces_with_guarded_members(source: &str) -> Vec<(String, String)> {
    let tokens = code_tokens_of(source);
    let mut found = Vec::new();
    for (index, token) in tokens.iter().enumerate() {
        if token.text != "namespace" {
            continue;
        }
        let mut cursor = index + 1;
        // A nested namespace (`namespace a::b`) is read by its last part, and an unnamed namespace
        // has no name to import.
        let mut name: Option<&str> = None;
        while tokens.get(cursor).is_some_and(|next| next.text == ":") {
            cursor += 1;
        }
        if let Some(next) = tokens.get(cursor).filter(|next| is_identifier(&next.text)) {
            name = Some(next.text.as_str());
        }
        let (Some(name), Some(body)) = (name, namespace_body(&tokens, cursor)) else {
            continue;
        };
        if let Some(member) = tokens[body.0..body.1]
            .iter()
            .find(|slot| GUARDED_NAMES.contains(&slot.text.as_str()))
        {
            found.push((name.to_string(), member.text.clone()));
        }
    }
    found
}

/// The token range inside a namespace body at or after `from`, exclusive of the braces.
fn namespace_body(tokens: &[CodeToken], from: usize) -> Option<(usize, usize)> {
    let mut cursor = from;
    while cursor < tokens.len() {
        match tokens[cursor].text.as_str() {
            ";" => return None,
            "{" => {
                let open = cursor;
                let mut depth = 0usize;
                while cursor < tokens.len() {
                    match tokens[cursor].text.as_str() {
                        "{" => depth += 1,
                        "}" => {
                            depth -= 1;
                            if depth == 0 {
                                return Some((open + 1, cursor));
                            }
                        }
                        _ => {}
                    }
                    cursor += 1;
                }
                return None;
            }
            _ => {}
        }
        cursor += 1;
    }
    None
}

/// Rejects a `#define` of the acquiring poll, the raw poll or the fence in any scanned source.
///
/// This is the translation-unit half of the guarantee, and it closes a whole class of evasion that no
/// call-site check can see: `#define spark_wait_for_flag_acquire(f, d) (1u)` in a kernel and
/// `#define __threadfence() ((void)0)` above the include each leave every call site looking exactly
/// right, while the kernel reads the published payload with no acquire (the reviewer's round-5
/// mutations 1 and 2, both guard-green before this check existed). A macro *use* is not analysed
/// here - that is the reachability lint's job - but the definition of one of these three names is a
/// failure naming the file and the line.
///
/// A using-declaration or using-directive that brings one of the three names into scope is the same
/// rewrite and is rejected by the same rule (round-10 A-2).
///
/// `#undef` is deliberately not rejected: undefining a name cannot put a fence into code that has
/// none, and the fence this header defines is already inside the helper's body, whose shape is
/// checked separately.
///
/// # Errors
/// Returns the offending `file:line` and the name that was defined.
pub fn guarded_names_are_not_redefined(sources: &[(&str, &str)]) -> Result<(), String> {
    for &(name, source) in sources {
        let tokens = code_tokens_of(source);
        for (index, token) in tokens.iter().enumerate() {
            if token.text != "#" || directive_keyword(&tokens, index) != "define" {
                continue;
            }
            let Some(defined) = tokens.get(index + 2) else {
                continue;
            };
            if GUARDED_NAMES.contains(&defined.text.as_str()) {
                return Err(rejection(format!(
                    "{name}:{}: #define {} rewrites the acquire out of the translation unit; the raw poll, the acquiring poll and the fence are not macro names under kernels/",
                    defined.physical_line, defined.text
                )));
            }
        }
    }
    if let Some((file, line, name, target)) = guarded_names_brought_into_scope(sources) {
        return Err(rejection(using_declaration_rejection(
            &format!("{file}:{line}"),
            &name,
            &target,
        )));
    }
    Ok(())
}

/// Requires exactly one definition of the acquiring poll, and that it is in the shared header.
///
/// The helper is the guarantee, so a second spelling of it is a defect even when the first is
/// intact. An overload whose parameter is `uint32_t *` rather than `const volatile uint32_t *` wins
/// overload resolution for a non-volatile flag - the reviewer's round-5 mutation 3, whose selected
/// callee clang -S confirmed - and can carry a body with no fence; a redeclaration that changes the
/// signature does the same from a distance. A definition is recognised by the body brace that follows
/// its parameter list, and a parameter list followed by a semicolon is a declaration unless it sits
/// where a callee normally sits.
///
/// # Errors
/// Returns every offending `file:line`, and why it is one.
pub fn the_acquiring_poll_is_defined_exactly_once(sources: &[(&str, &str)]) -> Result<(), String> {
    let mut definitions: Vec<(&str, usize)> = Vec::new();
    for &(name, source) in sources {
        let tokens = code_tokens_of(source);
        for (index, token) in tokens.iter().enumerate() {
            if token.text != ACQUIRE_HELPER {
                continue;
            }
            let Some((_, close)) = parameter_list(&tokens, index) else {
                continue;
            };
            match tokens.get(close + 1).map(|next| next.text.as_str()) {
                Some("{") => definitions.push((name, token.physical_line)),
                Some(";") if !looks_like_a_callee(&tokens, index) => {
                    return Err(rejection(format!(
                        "{name}:{}: a declaration of {ACQUIRE_HELPER} outside the shared header can change which definition a call binds to; the helper is defined once, in {COMMON_HEADER_NAME}",
                        token.physical_line
                    )));
                }
                _ => {}
            }
        }
    }
    if definitions.len() != 1 {
        return Err(rejection(format!(
            "{ACQUIRE_HELPER} must be defined exactly once, in the shared header; found {} definition(s): {}",
            definitions.len(),
            definitions
                .iter()
                .map(|(name, line)| format!("{name}:{line}"))
                .collect::<Vec<_>>()
                .join(", ")
        )));
    }
    if definitions[0].0 != COMMON_HEADER_NAME {
        return Err(rejection(format!(
            "{ACQUIRE_HELPER} is defined in {}, not in {COMMON_HEADER_NAME}",
            definitions[0].0
        )));
    }
    Ok(())
}

#[cfg(test)]
mod acquire_guard {
    use super::*;

    /// The poll of the Mega-mHC coefficients-ready flag: the only observing path in this milestone.
    const MEGA_POLL: &str = "spark_wait_for_flag(&hc_sync[1]";
    /// The poll as the shipped helper writes it. This is where the acquire now lives, so this is the
    /// text the token-level evasions below are applied to: the real header, not a fixture.
    const HELPER_POLL: &str = "spark_wait_for_flag(flag, deadline)";
    /// The two lines of the helper body that carry the acquire and the statement after it. Every
    /// evasion replaces this anchor, so each one is a mutation of the shipped text.
    const HELPER_ANCHOR: &str = r"    __threadfence();
    return observed;";
    /// The rest of the helper body, appended after each replacement so the fixture stays a body.
    const HELPER_TAIL: &str = r"
    return observed;";

    #[test]
    fn the_mega_mhc_observing_path_uses_the_acquiring_poll() {
        // Half one of the structural guarantee, read at the call site: the observing path names the
        // helper, so the fence cannot be separated from the poll by an edit here.
        let call = format!("{ACQUIRE_HELPER}(&hc_sync[1], deadline)");
        assert!(
            MEGA_MHC.contains(&call),
            "the observing path must call {call}, not the raw poll"
        );
        // The publisher's comment names the raw poll on purpose, so the scan is on code.
        let code = crate::c_decl::strip_comments(MEGA_MHC);
        for (index, _) in code.match_indices(RAW_POLL) {
            assert!(
                code[index..].starts_with(&format!("{ACQUIRE_HELPER}(")),
                "every poll in the kernel must be the acquiring helper: {call}"
            );
        }
        // The hand-rolled acquire that used to sit here is gone: with the helper in place a fence at
        // the call site would be a second, independently editable copy of the guarantee.
        let observing = MEGA_MHC.find(&call).expect("the call exists");
        assert!(
            !MEGA_MHC[observing..].contains("__threadfence"),
            "the acquire must not be duplicated at the call site"
        );
        // The read the acquire protects still happens after the call, and the call is what puts the
        // fence before it.
        let read = MEGA_MHC[observing..]
            .find("*hc_rms_out")
            .map(|offset| observing + offset)
            .expect("the observing path reads hc_rms_out after the poll");
        assert!(observing < read, "the helper call must precede the read");
        // ...and the release half of the pair must exist too: a fence before the flag store, not just
        // an acquire after the poll.
        let publish = MEGA_MHC
            .find("hc_sync[1] = 1u")
            .expect("the ready flag is published");
        let release = MEGA_MHC[..publish]
            .rfind("__threadfence();")
            .expect("a release fence precedes the publish");
        assert!(release < publish, "the release must precede the flag store");
    }

    #[test]
    fn the_other_five_kernels_have_no_observing_path_at_all() {
        // This is a raw SHAPE CHECK, not an analysis, and it is deliberately so. The Lead's
        // instruction is: an unconditional acquire on the observing path, or code that states why it
        // is provably unnecessary. These five kernels synchronise inside a block and never read a
        // peer-written flag, so they have no observing path to guard. The two contains() calls below
        // are the cheap tripwire that keeps that true: a future `spark_wait_for_flag` (a poll) or a
        // future `__threadfence` (a fence that may belong to an observing path) makes the author
        // come back and extend `acquire_fence_after_poll` on purpose. It can therefore reject a
        // kernel that legitimately needs a fence somewhere else, and that is the intended failure
        // mode: the fix is to make the new path explicit here, not to weaken the tripwire. The
        // source text is used on purpose, because the question is "did the shape change", which a
        // textual check answers exactly; the acquire guard itself is the analysis for the one path
        // that does exist.
        for (name, source) in [
            ("b12x_mxfp8_dense_sm121.cu", MXFP8_DENSE),
            ("b12x_mxfp4_moe_sm121.cu", MXFP4_MOE),
            ("flash_mla_csa2_sm121.cu", FLASH_MLA_CSA2),
            ("b12x_mxfp8_wo_a_sm121.cu", MXFP8_WO_A),
            ("cert_lm_head_sm121.cu", CERT_LM_HEAD),
        ] {
            assert!(
                !source.contains("spark_wait_for_flag"),
                "{name} polls a peer flag but has no acquire guard"
            );
            assert!(
                !source.contains("__threadfence()"),
                "{name} has a fence, so the guard above must be extended to cover it"
            );
        }
    }

    /// Every evasion the three reviews used, one case each, with the mutation applied to the real
    /// text where the poll and the acquire now live: the body of spark_wait_for_flag_acquire in the
    /// shared header. A rejection is therefore never a rejection of a fixture nobody ships. The
    /// coverage is what the list below contains and nothing more: the comment shapes, the branch
    /// shapes, the control statements, the short-circuit operators, the preprocessor shapes and the
    /// non-returning calls. What the guard does not catch is stated on acquire_fence_after_poll.
    #[test]
    fn the_acquire_guard_catches_every_evasion_the_reviews_used() {
        // Wrapped in a false condition: C's if (false) does not remove the token, but it removes the
        // guarantee, and the guard must notice.
        let hidden_in_false = COMMON_HEADER.replace(
            HELPER_ANCHOR,
            r"    if (false) __threadfence();
    return observed;",
        );
        assert_ne!(hidden_in_false, COMMON_HEADER, "the mutation must apply");
        assert!(acquire_fence_after_poll(&hidden_in_false, HELPER_POLL).is_err());

        // Hidden in an unrelated branch.
        let in_branch = COMMON_HEADER.replace(
            HELPER_ANCHOR,
            r"    if (lane == 0u) {
        __threadfence();
    }
    return observed;",
        );
        assert_ne!(in_branch, COMMON_HEADER);
        assert!(acquire_fence_after_poll(&in_branch, HELPER_POLL).is_err());

        // An early return in front of it.
        let early_exit = COMMON_HEADER.replace(
            HELPER_ANCHOR,
            r"    return;
    __threadfence();
    return observed;",
        );
        assert_ne!(early_exit, COMMON_HEADER);
        assert!(acquire_fence_after_poll(&early_exit, HELPER_POLL).is_err());

        // No fence at all: the statement that follows the poll is not an acquire.
        let removed = COMMON_HEADER.replace(HELPER_ANCHOR, "    return observed;");
        assert_ne!(removed, COMMON_HEADER);
        assert!(acquire_fence_after_poll(&removed, HELPER_POLL).is_err());

        // A control statement the line scan did not recognise: switch/case with a return inside.
        let hidden_in_switch = COMMON_HEADER.replace(
            HELPER_ANCHOR,
            r"    switch (lane) {
        case 0u:
            return;
        default:
            break;
    }
    __threadfence();
    return observed;",
        );
        assert_ne!(hidden_in_switch, COMMON_HEADER, "the mutation must apply");
        assert!(acquire_fence_after_poll(&hidden_in_switch, HELPER_POLL).is_err());

        // A do/while(0) around an exit, which is an ordinary statement to a line scan.
        let hidden_in_do = COMMON_HEADER.replace(
            HELPER_ANCHOR,
            r"    do {
        return;
    } while (0);
    __threadfence();
    return observed;",
        );
        assert_ne!(hidden_in_do, COMMON_HEADER);
        assert!(acquire_fence_after_poll(&hidden_in_do, HELPER_POLL).is_err());

        // A label and a goto, which are not branch keywords at all.
        let hidden_behind_label = COMMON_HEADER.replace(
            HELPER_ANCHOR,
            r"retry:
    __trap();
    goto retry;
    __threadfence();
    return observed;",
        );
        assert_ne!(hidden_behind_label, COMMON_HEADER);
        assert!(acquire_fence_after_poll(&hidden_behind_label, HELPER_POLL).is_err());

        // The fence compiled away by the preprocessor. A definitely-disabled region is ignored
        // rather than obeyed, so this one has no fence at all on the reachable path.
        let inside_if_zero = COMMON_HEADER.replace(
            HELPER_ANCHOR,
            r"    #if 0
    __threadfence();
    #endif
    return observed;",
        );
        assert_ne!(inside_if_zero, COMMON_HEADER);
        assert!(acquire_fence_after_poll(&inside_if_zero, HELPER_POLL).is_err());

        // A conditional the compiler may or may not take makes the fence conditional, so a fence
        // inside #ifdef is rejected even though the token is there.
        let inside_ifdef = COMMON_HEADER.replace(
            HELPER_ANCHOR,
            r"    #ifdef SPARK_M4_ACQUIRE
    __threadfence();
    #endif
    return observed;",
        );
        assert_ne!(inside_ifdef, COMMON_HEADER);
        assert!(acquire_fence_after_poll(&inside_ifdef, HELPER_POLL).is_err());

        // A fence hidden behind a macro: the call site never mentions __threadfence, so there is no
        // fence on the path as far as the source is concerned.
        let behind_macro = COMMON_HEADER.replace(
            HELPER_ANCHOR,
            r"    #define SPARK_ACQUIRE() __threadfence()
    SPARK_ACQUIRE();
    return observed;",
        );
        assert_ne!(behind_macro, COMMON_HEADER);
        assert!(acquire_fence_after_poll(&behind_macro, HELPER_POLL).is_err());

        // A fence that exists only inside a comment. The token scan has no fence to find.
        let commented_fence = COMMON_HEADER.replace(
            HELPER_ANCHOR,
            r"    /* the acquire used to be here: __threadfence(); */
    return observed;",
        );
        assert_ne!(commented_fence, COMMON_HEADER);
        assert!(acquire_fence_after_poll(&commented_fence, HELPER_POLL).is_err());

        // The re-anchoring evasion: the real poll is deleted and a comment that merely mentions it
        // is planted where the observation used to be. A search anchored on text rather than code
        // finds the mention first; the token search finds no poll in code at all.
        let reanchored = COMMON_HEADER
            .replace(
                "    const uint32_t observed = spark_wait_for_flag(flag, deadline);",
                r"    // spark_wait_for_flag(flag, deadline) is called by the observer later
    const uint32_t observed = 1u;",
            )
            .replace(HELPER_ANCHOR, "    return observed;");
        assert_ne!(reanchored, COMMON_HEADER, "the mutation must apply");
        assert!(
            reanchored.contains("// spark_wait_for_flag(flag, deadline)"),
            "the decoy comment is planted"
        );
        let error = acquire_fence_after_poll(&reanchored, HELPER_POLL)
            .expect_err("a mention in a comment cannot re-anchor the search");
        assert!(
            error.contains("no poll of"),
            "the decoy must not count: {error}"
        );

        // A poll that does not exist is an error rather than a silent pass.
        // The poll is located by the raw poll designator being called, not by the `poll` label, so a
        // label names nothing into existence: a source with no call is the error, and the message
        // prints the label it was given (round-13 A-2).
        let error =
            acquire_fence_after_poll("uint32_t f() { return 1u; }", "spark_wait_for_no_such_flag")
                .expect_err("a source with no poll is an error");
        assert!(
            error.contains("no poll of spark_wait_for_no_such_flag"),
            "the message names the label it was given: {error}"
        );
        // A mention of the poll inside a comment is not a poll either.
        let mention_only = COMMON_HEADER.replace(
            "    const uint32_t observed = spark_wait_for_flag(flag, deadline);",
            r"    // const uint32_t observed = spark_wait_for_flag(flag, deadline);
    const uint32_t observed = 1u;",
        );
        assert_ne!(mention_only, COMMON_HEADER);
        assert!(acquire_fence_after_poll(&mention_only, HELPER_POLL).is_err());

        // A short-circuit && : the fence is in the right operand, so it runs only when the left one
        // is true, and the statement expression around it is not a branch keyword.
        let short_circuit_and = COMMON_HEADER.replace(
            HELPER_ANCHOR,
            r"    (observed == 0u) && ({ __threadfence(); 1; });
    return observed;",
        );
        assert_ne!(short_circuit_and, COMMON_HEADER, "the mutation must apply");
        assert!(acquire_fence_after_poll(&short_circuit_and, HELPER_POLL).is_err());

        // A short-circuit || , which is the same hazard with the operands the other way round.
        let short_circuit_or = COMMON_HEADER.replace(
            HELPER_ANCHOR,
            r"    (observed == 0u) || ({ __threadfence(); 0; });
    return observed;",
        );
        assert_ne!(short_circuit_or, COMMON_HEADER, "the mutation must apply");
        assert!(acquire_fence_after_poll(&short_circuit_or, HELPER_POLL).is_err());

        // The bitwise spelling of the same shape: the guard rejects the single-character forms too,
        // so this cannot be used to get the statement expression back in.
        let bitwise_and = COMMON_HEADER.replace(
            HELPER_ANCHOR,
            r"    (observed == 0u) & ({ __threadfence(); 1; });
    return observed;",
        );
        assert_ne!(bitwise_and, COMMON_HEADER, "the mutation must apply");
        assert!(acquire_fence_after_poll(&bitwise_and, HELPER_POLL).is_err());

        // A fence on a #define continuation line. The preprocessor splices the next physical line
        // into the same logical line, so the fence is part of the macro body: it is not code here,
        // and skipping only the line that carries the # would read it as an unconditional fence.
        let behind_continuation = COMMON_HEADER.replace(
            HELPER_ANCHOR,
            "    #define SPARK_ACQUIRE() \\\n        __threadfence();\n    SPARK_ACQUIRE();\n    return observed;",
        );
        assert_ne!(
            behind_continuation, COMMON_HEADER,
            "the mutation must apply"
        );
        assert!(
            behind_continuation.contains(r"() \"),
            "the fixture must really continue the directive onto the next physical line, or this \
             case would test the same line rather than the continuation"
        );
        assert!(acquire_fence_after_poll(&behind_continuation, HELPER_POLL).is_err());

        // ...and the continuation handling itself, on a source where nothing else can be the reason:
        // the fence on the continuation line IS a bare call statement, so a guard that skipped only
        // the line carrying the # would find it and accept. The rejection has to be the missing
        // fence, which is what makes this case about the splice rather than about the code after it.
        let synthetic =
            "spark_wait_for_flag(&flag, deadline);\n#define SPARK_ACQUIRE() \\\n    __threadfence();\nSPARK_ACQUIRE();\n"
                .to_string();
        let error = acquire_fence_after_poll(&synthetic, "spark_wait_for_flag(&flag, deadline)")
            .expect_err("a fence inside a #define continuation line is not code on this path");
        assert!(
            error.contains("no reachable __threadfence"),
            "the splice is the reason: {error}"
        );

        // Calls that never return: a fence after one of these is unreachable rather than
        // unconditional, so each has to be rejected by name.
        for never_returns in [
            "__builtin_trap()",
            "__builtin_unreachable()",
            "__trap()",
            "abort()",
            "exit(1)",
            "_Exit(1)",
            "_exit(1)",
            "quick_exit(0)",
            "thrd_exit(0)",
            "pthread_exit(nullptr)",
            "longjmp(env, 1)",
            "(abort)()",
            "((abort))()",
        ] {
            let mutation = format!("    {never_returns};\n    __threadfence();{HELPER_TAIL}");
            let behind_non_returning = COMMON_HEADER.replace(HELPER_ANCHOR, &mutation);
            assert_ne!(
                behind_non_returning, COMMON_HEADER,
                "the mutation for {never_returns} must apply"
            );
            let error = acquire_fence_after_poll(&behind_non_returning, HELPER_POLL)
                .expect_err("a fence behind a call that never returns is unreachable");
            assert!(
                error.contains("does not return"),
                "{never_returns} must be rejected as non-returning rather than for some later \
                 reason: {error}"
            );
        }
    }

    /// The seven round-3 probes, each as its own source, with the exact reason the guard now gives.
    /// A guard that rejected everything would pass the old is_err() suite, so every case here pins
    /// the verdict text as well as the rejection.
    #[test]
    fn the_round_three_probes_are_rejected_for_the_stated_reason() {
        const POLL: &str = "spark_wait_for_flag(&flag, deadline)";
        let baseline = "spark_wait_for_flag(&flag, deadline);\n__threadfence();\n";
        assert_eq!(
            acquire_fence_after_poll(baseline, POLL),
            Ok(()),
            "the baseline is a real poll with a real acquire"
        );

        // A2 first half: ret<backslash><newline>urn is the keyword return after phase 2, so the
        // fence that follows is unreachable. The verdict names return. Naming the backslash-newline
        // as a String keeps the literal readable.
        let splice: String = ['\\', '\n'].into_iter().collect();
        let spliced_return =
            format!("spark_wait_for_flag(&flag, deadline);\nret{splice}urn;\n__threadfence();\n");
        let error = acquire_fence_after_poll(&spliced_return, POLL)
            .expect_err("a spliced return is a return");
        assert!(
            error.contains("return") && error.contains("stands between"),
            "the split identifier must be joined and named: {error}"
        );

        // A2 second half: an object-like macro whose body is a control keyword. The guard cannot
        // evaluate the preprocessor, but it refuses a use it cannot trust instead of guessing.
        let macro_return =
            "spark_wait_for_flag(&flag, deadline);\n#define BAIL return\nBAIL;\n__threadfence();\n";
        let error = acquire_fence_after_poll(macro_return, POLL)
            .expect_err("a macro that expands to return is not a statement the guard can trust");
        assert!(
            error.contains("BAIL") && error.contains("macro"),
            "the macro use must be named and its reason given: {error}"
        );

        // A3: the fence is inside a raw string, so the compiled kernel has no acquire at all. The
        // tokeniser must see one opaque literal, leaving no reachable fence.
        let raw_fence = "spark_wait_for_flag(&flag, deadline);\nconst char *s = R\"x(;\" ;__threadfence(); )x\";\n";
        let error = acquire_fence_after_poll(raw_fence, POLL)
            .expect_err("a fence inside a raw string is not a fence");
        assert!(
            error.contains("no reachable __threadfence"),
            "the raw string must hide the fence entirely: {error}"
        );

        // A4 first half: a comment opener split across lines. The whole line is a // comment, so the
        // fence in it does not run and there is no reachable acquire.
        let split_comment_fence =
            format!("spark_wait_for_flag(&flag, deadline);\n/{splice}/ ; __threadfence();\n");
        let error = acquire_fence_after_poll(&split_comment_fence, POLL)
            .expect_err("a fence in a split // comment is not code");
        assert!(
            error.contains("no reachable __threadfence"),
            "the split opener must make the fence a comment: {error}"
        );

        // A4 second half: the same shape hides the poll inside a comment, so the only "poll" in the
        // text is the comment's, and the unconditional fence after it must not be accepted.
        let split_comment_poll =
            format!("/{splice}/ spark_wait_for_flag(&flag, deadline);\n__threadfence();\n");
        let error = acquire_fence_after_poll(&split_comment_poll, POLL)
            .expect_err("a poll spelled in a split // comment is not a poll");
        assert!(
            error.contains("no poll of"),
            "the split opener must hide the poll: {error}"
        );

        // A5 first half: the poll lives in a #define body, which the guard cannot evaluate. It
        // reports no poll rather than treating the body as the observing path.
        let poll_in_define = "#define SPARK_POLL() spark_wait_for_flag(&flag, deadline)\nSPARK_POLL();\n__threadfence();\n";
        let error = acquire_fence_after_poll(poll_in_define, POLL)
            .expect_err("a poll inside a macro body is not a located poll");
        assert!(
            error.contains("no poll of"),
            "the macro body must not count as the poll: {error}"
        );

        // A5 second half: the poll is in a #if 0 region, so it is not in the compiled kernel. The
        // #endif and the bare fence after it must not be accepted as the observing path.
        let poll_inside_if_zero =
            "#if 0\nspark_wait_for_flag(&flag, deadline);\n#endif\n__threadfence();\n";
        let error = acquire_fence_after_poll(poll_inside_if_zero, POLL)
            .expect_err("a poll inside #if 0 is not compiled");
        assert!(
            error.contains("no poll of"),
            "the disabled region must not count as the poll: {error}"
        );

        // A1: __builtin_unreachable closes the fall-through path exactly as abort does, so a fence
        // after it is rejected by name.
        let builtin_unreachable =
            "spark_wait_for_flag(&flag, deadline);\n__builtin_unreachable();\n__threadfence();\n";
        let error = acquire_fence_after_poll(builtin_unreachable, POLL)
            .expect_err("__builtin_unreachable does not return");
        assert!(
            error.contains("__builtin_unreachable") && error.contains("does not return"),
            "the builtin must be named: {error}"
        );

        // A1: a fully parenthesized callee designator is still a call to the non-returning name.
        let parenthesized_callee =
            "spark_wait_for_flag(&flag, deadline);\n(abort)();\n__threadfence();\n";
        let error = acquire_fence_after_poll(parenthesized_callee, POLL)
            .expect_err("(abort)() is a call to abort");
        assert!(
            error.contains("does not return"),
            "the parenthesized callee must be unwrapped: {error}"
        );
        let doubly_parenthesized_callee =
            "spark_wait_for_flag(&flag, deadline);\n((abort))();\n__threadfence();\n";
        assert!(acquire_fence_after_poll(doubly_parenthesized_callee, POLL).is_err());
    }

    #[test]
    fn the_guard_accepts_a_fence_that_really_is_unconditional() {
        // Non-vacuity for every rejection above: the same helper body with the fence in place, and
        // with a definitely-disabled region (which executes nothing) or a string literal containing
        // a keyword (which is not a keyword) in front of it, must still pass. The text is the
        // header's helper body, which is where the poll and the acquire now live.
        assert_eq!(acquire_fence_after_poll(COMMON_HEADER, HELPER_POLL), Ok(()));
        let disabled_region = COMMON_HEADER.replace(
            HELPER_ANCHOR,
            r"    #if 0
    return;
    #endif
    __threadfence();
    return observed;",
        );
        assert_ne!(disabled_region, COMMON_HEADER);
        assert_eq!(
            acquire_fence_after_poll(&disabled_region, HELPER_POLL),
            Ok(()),
            "code that is definitely not compiled cannot make the fence conditional"
        );
        let string_keyword = COMMON_HEADER.replace(
            HELPER_ANCHOR,
            r#"    spark_note("if this returns there is no acquire");
    __threadfence();
    return observed;"#,
        );
        assert_ne!(string_keyword, COMMON_HEADER);
        assert_eq!(
            acquire_fence_after_poll(&string_keyword, HELPER_POLL),
            Ok(()),
            "a keyword inside a message is not a keyword"
        );
        // A raw string is one opaque token too, so neither the control keyword nor the fence it
        // spells out is code; the real fence after it must still be found and accepted.
        let raw_string_keyword = COMMON_HEADER.replace(
            HELPER_ANCHOR,
            r#"    const char *s = R"x(if (lane) return; __threadfence();)x";
    __threadfence();
    return observed;"#,
        );
        assert_ne!(raw_string_keyword, COMMON_HEADER);
        assert_eq!(
            acquire_fence_after_poll(&raw_string_keyword, HELPER_POLL),
            Ok(()),
            "a keyword and a fence inside a raw string are content, not code"
        );
        // Non-vacuity for the non-returning rule: a call that DOES return cannot make the fence
        // unreachable, so the rule is a list of names and not "any call before the fence".
        let returning_call = COMMON_HEADER.replace(
            HELPER_ANCHOR,
            r#"    spark_note("ready, no acquire taken");
    __threadfence();
    return observed;"#,
        );
        assert_ne!(returning_call, COMMON_HEADER);
        assert_eq!(
            acquire_fence_after_poll(&returning_call, HELPER_POLL),
            Ok(()),
            "a call that returns cannot make the fence conditional"
        );
    }

    /// The round-4 probes, each as its own source and each pinning the verdict. A1 is a splice the
    /// compiler makes and the guard did not; A2 is the function-like spelling of the same macro
    /// escape; A3 is the false rejection a literal `#if 1` used to cause. The fixtures are raw
    /// strings so the backslash and the physical newline are exactly what the compiler sees.
    #[test]
    fn the_round_four_probes_are_handled_for_the_stated_reason() {
        const POLL: &str = "spark_wait_for_flag(&flag, deadline)";

        // A1: a backslash separated from its newline by spaces is still a splice to clang and nvcc,
        // so `ret\   \nurn` is the keyword return and the fence behind it is unreachable. clang
        // -E -P on this fixture emits `return;` then the fence.
        let whitespace_splice = r"spark_wait_for_flag(&flag, deadline);
ret\   
urn;
__threadfence();
";
        let error = acquire_fence_after_poll(whitespace_splice, POLL)
            .expect_err("a whitespace splice is a splice");
        assert!(
            error.starts_with(
                r#"line 2: "return" stands between the poll and the acquire in spark_wait_for_flag_acquire, and this scan reads placement"#
            ) && error.ends_with(CANONICAL_HINT),
            "A1: the joined keyword must be named, and the reason must be a limitation rather than a \n             claim about the path: {error}"
        );

        // A2 first half: a function-like macro whose body is a control keyword, called after the
        // poll. The object-like-only scan could not see it.
        let function_like_after = r"spark_wait_for_flag(&flag, deadline);
#define BAIL() return
BAIL();
__threadfence();
";
        let error = acquire_fence_after_poll(function_like_after, POLL)
            .expect_err("a function-like macro body is a macro body");
        assert!(
            error.starts_with(
                "line 3: BAIL is a macro whose body contains a control keyword, a non-returning call, or another such macro, so this scan does not certify that the fence runs after it"
            ) && error.ends_with(CANONICAL_HINT),
            "A2: the function-like use must be named, with the limitation and the hint: {error}"
        );

        // A2 second half: the poll passed as the macro's argument. The walk starts after the poll
        // and so never sees the macro name; the invocation has to be checked before the walk.
        let function_like_argument = r"#define BAIL(x) return x
BAIL(spark_wait_for_flag(&flag, deadline));
__threadfence();
";
        let error = acquire_fence_after_poll(function_like_argument, POLL)
            .expect_err("a poll expanded inside a returning macro is not an observing path");
        assert!(
            error.starts_with(
                "line 2: the poll is used with BAIL, a macro whose body contains a control keyword, a non-returning call, or another such macro, so this scan does not certify that the fence runs after it"
            ) && error.ends_with(CANONICAL_HINT),
            "A2: a poll passed to a suspicious macro must be rejected, with the limitation and the hint: {error}"
        );

        // A2 with a non-returning body rather than a control keyword, and the macro defined in the
        // same source: the same argument-list shape.
        let function_like_abort = r"#define BAIL(x) abort()
BAIL(spark_wait_for_flag(&flag, deadline));
__threadfence();
";
        let error = acquire_fence_after_poll(function_like_abort, POLL)
            .expect_err("a poll expanded inside an aborting macro is not an observing path");
        assert!(
            error.starts_with(
                "line 2: the poll is used with BAIL, a macro whose body contains a control keyword, a non-returning call, or another such macro, so this scan does not certify that the fence runs after it"
            ) && error.ends_with(CANONICAL_HINT),
            "A2: the non-returning body is suspicious in the function-like spelling too, with the limitation and the hint: {error}"
        );

        // A2 with the argument parenthesised twice: every token of the statement is visited, so the
        // extra parentheses cannot hide the macro name.
        let double_parenthesised_argument = r"#define BAIL(x) return x
BAIL((spark_wait_for_flag(&flag, deadline)));
__threadfence();
";
        let error = acquire_fence_after_poll(double_parenthesised_argument, POLL)
            .expect_err("parentheses around the poll argument do not hide the macro");
        assert!(
            error.starts_with(
                "line 2: the poll is used with BAIL, a macro whose body contains a control keyword, a non-returning call, or another such macro, so this scan does not certify that the fence runs after it"
            ) && error.ends_with(CANONICAL_HINT),
            "A2: a nested spelling is the same use, with the limitation and the hint: {error}"
        );

        // A2 object-like, standing in front of the poll as a control statement. The macro name is
        // before the poll, so only the same-statement scan can see it.
        let object_like_before_poll = r"#define BAIL return
BAIL spark_wait_for_flag(&flag, deadline);
__threadfence();
";
        let error = acquire_fence_after_poll(object_like_before_poll, POLL)
            .expect_err("an object-like control macro in front of the poll is the same escape");
        assert!(
            error.starts_with(
                "line 2: the poll is used with BAIL, a macro whose body contains a control keyword, a non-returning call, or another such macro, so this scan does not certify that the fence runs after it"
            ) && error.ends_with(CANONICAL_HINT),
            "A2: the object-like spelling is covered by the same scan, with the limitation and the hint: {error}"
        );

        // A2 non-vacuity: a macro that only parenthesises its argument is not suspicious, and a
        // suspicious macro's DEFINITION (with no use) must not be mistaken for a use.
        let harmless_macro_before_poll = r"#define BAIL(x) (x)
BAIL(spark_wait_for_flag(&flag, deadline));
__threadfence();
";
        assert_eq!(
            acquire_fence_after_poll(harmless_macro_before_poll, POLL),
            Ok(()),
            "a macro that only parenthesises its argument is not suspicious"
        );
        let definition_only = r"#define BAIL return
spark_wait_for_flag(&flag, deadline);
__threadfence();
";
        assert_eq!(
            acquire_fence_after_poll(definition_only, POLL),
            Ok(()),
            "a #define is not a use"
        );

        // A3: a poll inside a literal #if 1 IS compiled, so the guard must find it. The fence after
        // the #endif is unconditional.
        let poll_in_if_one = r"#if 1
spark_wait_for_flag(&flag, deadline);
#endif
__threadfence();
";
        assert_eq!(acquire_fence_after_poll(poll_in_if_one, POLL), Ok(()));

        // A3: the #else arm of a literal #if 0 is the compiled arm.
        let poll_in_else_of_if_zero = r"#if 0
spark_wait_for_flag(&flag, deadline);
#else
spark_wait_for_flag(&flag, deadline);
#endif
__threadfence();
";
        assert_eq!(
            acquire_fence_after_poll(poll_in_else_of_if_zero, POLL),
            Ok(())
        );

        // A3: a poll AND its fence inside a literal #if 1 are both compiled.
        let poll_and_fence_in_if_one = r"#if 1
spark_wait_for_flag(&flag, deadline);
__threadfence();
#endif
";
        assert_eq!(
            acquire_fence_after_poll(poll_and_fence_in_if_one, POLL),
            Ok(())
        );

        // A3 negative: the fence in the #else arm of a literal #if 1 is definitely not compiled, so
        // there is no acquire on the path. Accepting it would be the unsound half of the same rule.
        let fence_in_else_of_if_one = r"#if 1
spark_wait_for_flag(&flag, deadline);
#else
__threadfence();
#endif
";
        let error = acquire_fence_after_poll(fence_in_else_of_if_one, POLL)
            .expect_err("the #else of #if 1 is not compiled");
        assert!(
            error.contains("no reachable __threadfence"),
            "the disabled arm has no fence: {error}"
        );

        // A3: a poll in a region the guard cannot resolve (#ifdef) is still not a located poll.
        // This conservative half is kept on purpose and the exact verdict names it.
        let poll_in_ifdef = r"#ifdef SPARK_M4_ACQUIRE
spark_wait_for_flag(&flag, deadline);
#endif
__threadfence();
";
        let error = acquire_fence_after_poll(poll_in_ifdef, POLL)
            .expect_err("a may-not-hold poll is not a poll");
        assert!(
            error.starts_with(
                "no poll of spark_wait_for_flag(&flag, deadline) in this source's code (a mention inside a comment is not a poll)"
            ) && error.ends_with(CANONICAL_HINT),
            "A3: the may-not-hold region is conservative, not compiled: {error}"
        );
    }
    /// The structural guarantee in its positive direction, on the two shipped texts it reads: the
    /// six kernel translation units and the helper body in the header. The reachability lint runs
    /// over the same header body at the end, so the two checks are visible as separate predicates
    /// rather than one check counted twice: the lint reasons about the text around a poll, the
    /// structural check reasons about the brace depth inside one fixed body.
    #[test]
    fn the_acquire_is_structural_in_the_shipped_helper() {
        assert_eq!(call_site_check_over_the_walk(), Ok(()));
        assert_eq!(acquire_helper_fences_unconditionally(COMMON_HEADER), Ok(()));
        assert_eq!(acquire_fence_after_poll(COMMON_HEADER, HELPER_POLL), Ok(()));

        // A9: the call-site check is handed the walk, not only the six names compiled into this
        // crate, so a kernel file that lands under kernels/ is scanned before anyone adds it to
        // KERNELS or to the baseline.
        let mut extended = kernels_directory();
        extended.push((
            "new_kernel_sm121.cu".to_string(),
            "void k() { spark_wait_for_flag(&flag, deadline); }\n".to_string(),
        ));
        let error =
            observing_paths_call_the_acquiring_poll(&kernels_as_refs(&kernel_units(&extended)))
                .expect_err("a kernel file the walk finds must be scanned");
        assert!(
            error.starts_with("new_kernel_sm121.cu:1:"),
            "the message must name the new file and its line: {error}"
        );
    }

    /// The four mutations the Lead asked for, each applied to a copy of the shipped text and each
    /// with its verdict pinned: a check that rejected everything would pass a bare is_err() suite.
    /// (a) the fence deleted from the helper body; (b) the fence wrapped in the observed-flag
    /// condition, so the timeout path would have no acquire; (c) a bare poll back in the observing
    /// kernel; (d) the unmutated text, which is green.
    #[test]
    fn the_structural_check_rejects_the_four_mutations() {
        // (a) the fence deleted.
        let without_fence = COMMON_HEADER.replace(HELPER_ANCHOR, "    return observed;");
        assert_ne!(without_fence, COMMON_HEADER, "the mutation must apply");
        let error = acquire_helper_fences_unconditionally(&without_fence)
            .expect_err("(a) a helper without its fence must be rejected");
        assert_eq!(
            error,
            format!(
                "no reachable __threadfence() follows the poll of spark_wait_for_flag, so the observing path has no acquire; {CANONICAL_HINT}"
            ),
            "(a) the missing fence must be named, and the rejection must name the canonical form"
        );

        // (b) the fence wrapped in the observed-flag condition: it would run only when the flag was
        // observed, so the timeout path would have no acquire at all.
        let conditional = COMMON_HEADER.replace(
            HELPER_ANCHOR,
            r"    if (observed != 0u) {
        __threadfence();
    }
    return observed;",
        );
        assert_ne!(conditional, COMMON_HEADER, "the mutation must apply");
        let error = acquire_helper_fences_unconditionally(&conditional)
            .expect_err("(b) a fence inside a branch must be rejected");
        assert!(
            error.contains("nested brace depth")
                && error.contains(
                    "a branch, a loop, a try block, a requires-expression, a lambda or a type body"
                ),
            "(b) the conditional acquire must be named as conditional: {error}"
        );

        // (c) a bare poll back in the observing kernel.
        let bare_poll = MEGA_MHC.replace(
            "spark_wait_for_flag_acquire(&hc_sync[1], deadline)",
            "spark_wait_for_flag(&hc_sync[1], deadline)",
        );
        assert_ne!(bare_poll, MEGA_MHC, "the mutation must apply");
        let error = observing_paths_call_the_acquiring_poll(&[("mega_mhc_sm121.cu", &bare_poll)])
            .expect_err("(c) a bare poll in a kernel translation unit must be rejected");
        let offset = bare_poll
            .find("spark_wait_for_flag(&hc_sync[1]")
            .expect("the bare poll is there");
        let line = bare_poll[..offset].matches('\n').count() + 1;
        assert!(
            error.starts_with(&format!("mega_mhc_sm121.cu:{line}:")),
            "(c) the message must name the file and the line: {error}"
        );
        assert!(
            error.contains("must call spark_wait_for_flag_acquire"),
            "(c) the message must say what to call instead: {error}"
        );

        // (d) restored: the unmutated text is green. Nothing was left behind by the three cases
        // above, because each one is a copy of the shipped text.
        assert_eq!(call_site_check_over_the_walk(), Ok(()));
        assert_eq!(acquire_helper_fences_unconditionally(COMMON_HEADER), Ok(()));
    }

    /// The round-8 structural findings, on the shipped helper body.
    ///
    /// (A1) The hole: the check recorded the first thing that can keep the fence from running only at
    /// brace depth 0, so an early exit inside a top-level compound block was invisible while a later
    /// top-level fence was accepted: `{ return 0u; }`, `{ if (observed != 0u) return 0u; }`,
    /// `{ abort(); }`, `{ __trap(); }` and `{ goto done; }` between the poll and the fence were all
    /// green. Each is rejected now, at any depth, with the reason that names it.
    ///
    /// (A2)-(A4) The false rejections: the dereferenced designator `(*__threadfence)();` (the lint
    /// accepted it, the structural check did not), the fence alone in a bare block (rejected with the
    /// reason for a branch that was not there) and the fence behind a label. nvcc compiles all three
    /// and each runs unconditionally, so all three are accepted now. What stays rejected is what is
    /// genuinely conditional: a fence in a branch or a loop body, and a fence in a conditional
    /// expression.
    #[test]
    fn the_structural_check_reads_every_depth_and_accepts_unconditional_spellings() {
        let poll_statement = "    const uint32_t observed = spark_wait_for_flag(flag, deadline);";
        let insert_after_poll = |insert: &str| {
            COMMON_HEADER.replace(poll_statement, &format!("{poll_statement}\n{insert}"))
        };

        // (A1) an early exit inside a compound block between the poll and the fence.
        for (insert, reason) in [
            (
                "    { return 0u; }",
                "\"return\" stands between the poll and the acquire",
            ),
            (
                "    { if (observed != 0u) return 0u; }",
                "\"if\" stands between the poll and the acquire",
            ),
            (
                "    { abort(); }",
                "does not return and stands between the poll and the acquire",
            ),
            (
                "    { __trap(); }",
                "__trap does not return and stands between the poll and the acquire",
            ),
            (
                "    { goto done; }",
                "\"goto\" stands between the poll and the acquire",
            ),
        ] {
            let mutation = insert_after_poll(insert);
            assert_ne!(mutation, COMMON_HEADER, "the mutation must apply: {insert}");
            let error = acquire_helper_fences_unconditionally(&mutation)
                .expect_err("an early exit at any depth must be rejected");
            assert!(error.contains(reason), "{insert} must be named: {error}");
            assert!(
                acquire_fence_after_poll(&mutation, HELPER_POLL).is_err(),
                "{insert} must be rejected by the reachability lint as well"
            );
        }
        // The same statement at depth 0 is rejected as it was before: the rule widened what is seen,
        // not what is accepted.
        assert!(
            acquire_helper_fences_unconditionally(&insert_after_poll("    return 0u;")).is_err()
        );
        // A bare block that holds ONLY the fence has no blocker in it, which is why (A3) below can
        // accept it without weakening this rule.
        assert_eq!(
            acquire_helper_fences_unconditionally(&insert_after_poll("    { __threadfence(); }")),
            Ok(())
        );

        // (A2) the dereferenced and address-of designators are the same call.
        for spelling in ["(*__threadfence)();", "(&__threadfence)();"] {
            let mutation =
                COMMON_HEADER.replace("    __threadfence();", &format!("    {spelling}"));
            assert_ne!(
                mutation, COMMON_HEADER,
                "the mutation must apply: {spelling}"
            );
            assert_eq!(
                acquire_helper_fences_unconditionally(&mutation),
                Ok(()),
                "{spelling} is the same unconditional call"
            );
            assert_eq!(
                acquire_fence_after_poll(&mutation, HELPER_POLL),
                Ok(()),
                "the lint and the structural check must agree about {spelling}"
            );
        }

        // (A3) the fence alone in a bare block: a bare block is not a branch and not a loop.
        let bare_block = COMMON_HEADER.replace("    __threadfence();", "    { __threadfence(); }");
        assert_ne!(bare_block, COMMON_HEADER, "the mutation must apply");
        assert_eq!(acquire_helper_fences_unconditionally(&bare_block), Ok(()));
        assert_eq!(acquire_fence_after_poll(&bare_block, HELPER_POLL), Ok(()));

        // ...while a branch or a loop body stays rejected, with the reason that fits it.
        for wrapped in [
            "    if (observed != 0u) { __threadfence(); }",
            "    for (uint32_t i = 0; i < 1u; ++i) { __threadfence(); }",
        ] {
            let mutation = COMMON_HEADER.replace("    __threadfence();", wrapped);
            assert_ne!(
                mutation, COMMON_HEADER,
                "the mutation must apply: {wrapped}"
            );
            let error = acquire_helper_fences_unconditionally(&mutation)
                .expect_err("a fence in a branch or a loop body must be rejected");
            assert!(
                error.contains("nested brace depth")
                    && error.contains("a branch, a loop, a try block, a requires-expression, a lambda or a type body"),
                "{wrapped}: {error}"
            );
        }

        // (A4) a label before the fence is a statement boundary in both checks.
        let labelled = COMMON_HEADER.replace("    __threadfence();", "    obs: __threadfence();");
        assert_ne!(labelled, COMMON_HEADER, "the mutation must apply");
        assert_eq!(
            acquire_helper_fences_unconditionally(&labelled),
            Ok(()),
            "a labelled statement runs unconditionally"
        );
        assert_eq!(acquire_fence_after_poll(&labelled, HELPER_POLL), Ok(()));

        // A conditional expression earlier in the same source does not make a later label look like
        // one: the search for the matching `?` stops at the statement's start.
        let labelled_after_ternary =
            format!("uint32_t earlier = (observed != 0u) ? 1u : 0u;\n{labelled}");
        assert_eq!(
            acquire_helper_fences_unconditionally(&labelled_after_ternary),
            Ok(()),
            "an earlier ? must not turn a later label into a conditional"
        );

        // ...but the colon of a conditional expression is not a label, so a fence in the false branch
        // of `? :` stays rejected by both checks.
        let ternary = COMMON_HEADER.replace(
            "    __threadfence();",
            "    (void)(observed != 0u ? 1u : (__threadfence(), 0u));",
        );
        assert_ne!(ternary, COMMON_HEADER, "the mutation must apply");
        let error = acquire_helper_fences_unconditionally(&ternary)
            .expect_err("a fence in a conditional expression must be rejected");
        assert!(error.contains("not a bare call statement"), "{error}");
        assert!(
            acquire_fence_after_poll(&ternary, HELPER_POLL).is_err(),
            "the lint must reject the conditional-expression fence as well"
        );
    }

    /// The shapes the round-8 review named, pinned one by one.
    ///
    /// Every rejection here is a shape nvcc compiles to FEWER barriers than the committed baseline,
    /// and every acceptance is one it compiles to the baseline cubin byte for byte (nvcc 13.0.3,
    /// sm_121a, measured in the container: 70432 bytes, 4 PTX membar lines, sha256 605321041ac5ec1e), so an
    /// unnecessary rejection and a real hole both surface as a failing assertion here rather than as a
    /// claim in a comment.
    #[test]
    fn the_round_eight_findings_are_closed() {
        let poll_statement = "    const uint32_t observed = spark_wait_for_flag(flag, deadline);";
        let insert_after_poll = |insert: &str| {
            COMMON_HEADER.replace(poll_statement, &format!("{poll_statement}\n{insert}"))
        };

        // (A1) a non-returning call that is NOT a statement of its own. Requiring a semicolon after
        // the argument list missed the whole class, and nvcc moved in every measurable spelling of
        // it: with (__builtin_unreachable(), 0u), with uint32_t bail = (__builtin_unreachable(), 0u)
        // and with (observed, __builtin_unreachable()) mega_mhc compiled to 36696 bytes with 2 membar (pre-change)
        // lines (sha 363f46...) instead of the baseline 70432 / 4 / 605321041ac5ec1e, and (__trap(), 0u)
        // compiled to 38232 bytes (sha 7455986b...; pre-change) with the four barrier instructions present but
        // unreachable behind the trap. The last fixture, wrap(__builtin_unreachable()), pins the same
        // rule in a position nvcc refuses outright - the builtin returns void, so it cannot be an
        // argument at all - which is also why pinning it costs nothing: a shape that does not compile
        // cannot ship a missing acquire.
        for insert in [
            "    (void)(__trap(), 0u);",
            "    (void)(__builtin_unreachable(), 0u);",
            "    uint32_t bail = (__builtin_unreachable(), 0u);",
            "    (void)(observed, __builtin_unreachable());",
            "    (void)wrap(__builtin_unreachable());",
        ] {
            let mutation = insert_after_poll(insert);
            assert_ne!(mutation, COMMON_HEADER, "the mutation must apply: {insert}");
            let error = acquire_helper_fences_unconditionally(&mutation)
                .expect_err("a call that never returns blocks the acquire wherever it is written");
            assert!(
                error.contains("does not return and stands between the poll and the acquire"),
                "{insert} must be named as non-returning: {error}"
            );
            let lint = acquire_fence_after_poll(&mutation, HELPER_POLL)
                .expect_err("the reachability lint must reject it too");
            assert!(lint.contains("does not return"), "{insert}: {lint}");
        }

        // (A2) any nesting of parentheses, * and & around the designator is the same call, and nvcc
        // compiles all three spellings below to the baseline cubin byte for byte.
        for spelling in [
            "(*(&__threadfence))();",
            "(&(*__threadfence))();",
            "((*(&__threadfence)))();",
        ] {
            let mutation =
                COMMON_HEADER.replace("    __threadfence();", &format!("    {spelling}"));
            assert_ne!(
                mutation, COMMON_HEADER,
                "the mutation must apply: {spelling}"
            );
            assert_eq!(
                acquire_helper_fences_unconditionally(&mutation),
                Ok(()),
                "{spelling} calls the same fence"
            );
            assert_eq!(
                acquire_fence_after_poll(&mutation, HELPER_POLL),
                Ok(()),
                "the lint and the structural check must agree about {spelling}"
            );
        }
        // A genuine operator is still an operator in both checks: the wrapper walk must not turn &
        // or + into address-of.
        for larger in [
            "    (void)(observed & (__threadfence)());",
            "    (void)(1u + (__threadfence)());",
        ] {
            let mutation = COMMON_HEADER.replace("    __threadfence();", larger);
            assert_ne!(mutation, COMMON_HEADER, "the mutation must apply: {larger}");
            assert!(
                acquire_helper_fences_unconditionally(&mutation).is_err(),
                "{larger} is the fence inside a larger expression"
            );
            assert!(
                acquire_fence_after_poll(&mutation, HELPER_POLL).is_err(),
                "the lint must reject {larger} as well"
            );
        }

        // (A3) only the unqualified or globally qualified name is this barrier. With a namespace
        // member named __threadfence, X::__threadfence(); compiled mega_mhc to 70264 bytes with 2
        // membar lines (sha 42454ad1...) - a silently missing acquire - while ::__threadfence(); is the
        // real global function and compiles to the baseline cubin byte for byte.
        let qualified = COMMON_HEADER.replace("    __threadfence();", "    X::__threadfence();");
        assert_ne!(qualified, COMMON_HEADER, "the mutation must apply");
        let error = acquire_helper_fences_unconditionally(&qualified)
            .expect_err("a namespaced no-op is not the barrier");
        assert!(
            error.contains("no reachable __threadfence() follows the poll of spark_wait_for_flag"),
            "the truthful verdict is that the body has no barrier: {error}"
        );
        // The lint refuses the shipped body too, by the early return that now stands where the
        // acquire should be; the precise verdict for a qualified name is pinned on the small bodies
        // below, where the name is the only thing after the poll.
        assert!(
            acquire_fence_after_poll(&qualified, HELPER_POLL).is_err(),
            "the lint must not certify a body whose fence is a namespaced no-op"
        );
        // A qualified name is read as "no barrier here" rather than as an error, so a REAL barrier
        // after it is still found by both checks - which is what keeps them agreeing.
        let raw_poll = "spark_wait_for_flag(&flag, deadline)";
        for member in [
            "X::__threadfence();",
            "x.__threadfence();",
            "x->__threadfence();",
        ] {
            let small = format!("{raw_poll}\n{member}\n");
            let lint = acquire_fence_after_poll(&small, raw_poll)
                .expect_err("a qualified name is not the barrier");
            assert!(
                lint.contains("no reachable __threadfence()"),
                "{member} must be reported as no barrier: {lint}"
            );
        }
        let small = format!("{raw_poll}\nX::__threadfence();\n__threadfence();\n");
        assert_eq!(
            acquire_fence_after_poll(&small, raw_poll),
            Ok(()),
            "the global barrier after the qualified name is still read"
        );
        let qualified_then_global = COMMON_HEADER.replace(
            "    __threadfence();",
            "    X::__threadfence();\n    __threadfence();",
        );
        assert_ne!(qualified_then_global, COMMON_HEADER, "must apply");
        assert_eq!(
            acquire_helper_fences_unconditionally(&qualified_then_global),
            Ok(()),
            "and the structural check must agree"
        );
        for member in ["    x.__threadfence();", "    x->__threadfence();"] {
            let mutation = COMMON_HEADER.replace("    __threadfence();", member);
            assert_ne!(mutation, COMMON_HEADER, "the mutation must apply: {member}");
            assert!(
                acquire_helper_fences_unconditionally(&mutation).is_err(),
                "{member} names an object's function, not the barrier"
            );
        }
        let global = COMMON_HEADER.replace("    __threadfence();", "    ::__threadfence();");
        assert_ne!(global, COMMON_HEADER, "the mutation must apply");
        assert_eq!(
            acquire_helper_fences_unconditionally(&global),
            Ok(()),
            "::__threadfence() is the global function itself"
        );
        assert_eq!(
            acquire_fence_after_poll(&global, HELPER_POLL),
            Ok(()),
            "the lint must agree about ::__threadfence()"
        );

        // (A4) a nested expansion no longer launders a conditional fence. The inner macro carries the
        // if, so the fence inside OUTER { ... } is conditional; nvcc compiled that body to 38624
        // bytes with 3 membar lines (sha e5b9d6...) (pre-change), one barrier short of the baseline. The names this
        // source defines are followed to a fixed point, and the message says which macro is which.
        let nested = format!(
            "#define OUTER INNER\n#define INNER if (observed != 0u)\n{}",
            COMMON_HEADER.replace("    __threadfence();", "    OUTER { __threadfence(); }")
        );
        let error = acquire_fence_after_poll(&nested, HELPER_POLL)
            .expect_err("a macro that names a suspicious macro is suspicious");
        assert!(
            error.contains(
                "OUTER is a macro whose body contains a control keyword, a \
                            non-returning call, or another such macro"
            ),
            "the nested definition must be named as the reason: {error}"
        );
        // The direct form is still caught, so the fixed point widened the reading rather than
        // replacing it.
        let direct = format!(
            "#define INNER if (observed != 0u)\n{}",
            COMMON_HEADER.replace("    __threadfence();", "    INNER { __threadfence(); }")
        );
        assert!(acquire_fence_after_poll(&direct, HELPER_POLL).is_err());

        // (A5) throw and co_return are control tokens: neither can appear in a shipping device
        // function, so rejecting them costs nothing and closes the textual gap. nvcc 12.9.1 refuses
        // both - "device code does not support exception handling" for throw and "identifier
        // co_return is undefined" for co_return (rc=1 each in the container).
        for exit in ["    throw 0u;", "    co_return;"] {
            let mutation = insert_after_poll(exit);
            assert_ne!(mutation, COMMON_HEADER, "the mutation must apply: {exit}");
            let error = acquire_helper_fences_unconditionally(&mutation)
                .expect_err("an exit the guard cannot see through is a blocker");
            assert!(
                error.contains("stands between the poll and the acquire"),
                "{exit}: {error}"
            );
            let lint = acquire_fence_after_poll(&mutation, HELPER_POLL)
                .expect_err("the lint must reject it too");
            assert!(
                lint.contains("stands between the poll and the acquire"),
                "{exit}: {lint}"
            );
        }

        // (A6) a conditional operator in an earlier, COMPLETED statement cannot make the fence
        // conditional. nvcc compiles this body to the baseline cubin byte for byte, so rejecting it
        // was a false rejection; the operator counts only in the fence's own statement, which is why
        // the ternary the round-8 test pins above is still rejected.
        let earlier = COMMON_HEADER
            .replace(
                poll_statement,
                &format!(
                    "{poll_statement}\n    uint32_t masked = observed & 1u;\n    bool ready2 = \
                     observed != 0u ? true : false;"
                ),
            )
            .replace("    __threadfence();", "    obs: __threadfence();");
        assert_ne!(earlier, COMMON_HEADER, "the mutation must apply");
        assert_eq!(
            acquire_helper_fences_unconditionally(&earlier),
            Ok(()),
            "an operator in an earlier statement is not a blocker"
        );
        assert_eq!(
            acquire_fence_after_poll(&earlier, HELPER_POLL),
            Ok(()),
            "the lint must agree about an earlier statement's operators"
        );
    }

    /// The two checks did not collapse into one, from both directions. The reachability lint still
    /// rejects its own round-3 and round-4 fixtures (those tests are unchanged and run beside this
    /// one). On the shipped text the separation is this: the lint, handed the observing kernel
    /// alone, now reports a poll it cannot find, because the acquire moved into the helper's
    /// expansion, while the structural check reads the pair - the kernel translation units and the
    /// helper body - and is green.
    #[test]
    fn the_structural_check_and_the_reachability_lint_are_separate_checks() {
        let error = acquire_fence_after_poll(MEGA_MHC, MEGA_POLL)
            .expect_err("the lint on the kernel alone has no acquire to find");
        assert!(
            error.contains("no poll of"),
            "the kernel no longer holds the raw poll in its own text: {error}"
        );
        assert!(
            MEGA_MHC.contains("__threadfence"),
            "the release fence is still in the kernel, so the lint is not being handed an empty file"
        );
        assert_eq!(acquire_fence_after_poll(COMMON_HEADER, HELPER_POLL), Ok(()));
        assert_eq!(acquire_helper_fences_unconditionally(COMMON_HEADER), Ok(()));
    }

    /// The include line of every kernel, as a raw string so the quotes below are literal.
    const INCLUDE: &str = r#"#include "m4_common_sm121.cuh""#;

    /// Every source under kernels/, read from disk. The translation-unit checks take their input as
    /// an argument, so the test hands them the directory this milestone ships rather than a list kept
    /// by hand: a file that lands there later is read too, and a check cannot pass by not being
    /// handed the file that breaks it.
    fn kernels_directory() -> Vec<(String, String)> {
        let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../kernels");
        let mut sources: Vec<(String, String)> = std::fs::read_dir(&directory)
            .unwrap_or_else(|error| panic!("cannot read {}: {error}", directory.display()))
            .map(|entry| {
                let path = entry.expect("a readable directory entry").path();
                let name = path
                    .file_name()
                    .expect("a file name")
                    .to_string_lossy()
                    .into_owned();
                let text = std::fs::read_to_string(&path)
                    .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
                (name, text)
            })
            .collect();
        sources.sort();
        assert!(
            sources.iter().any(|(name, _)| name == COMMON_HEADER_NAME),
            "the walk must find the shared header: {sources:?}"
        );
        assert!(
            sources.iter().any(|(name, _)| name == "mega_mhc_sm121.cu"),
            "the walk must find the observing kernel: {sources:?}"
        );
        sources
    }

    fn kernels_as_refs(sources: &[(String, String)]) -> Vec<(&str, &str)> {
        sources
            .iter()
            .map(|(name, text)| (name.as_str(), text.as_str()))
            .collect()
    }

    /// The kernel translation units of a walk: every `.cu` file under kernels/, but not the shared
    /// header. The call-site check reads translation units - the raw poll is defined AND called inside
    /// the header's helper body - so the header must not be handed to it. Taking the units from the
    /// walk rather than a compiled-in array is the round-8 A9 fix: a kernel file added under kernels/
    /// that calls the raw poll is scanned instead of being invisible.
    fn kernel_units(sources: &[(String, String)]) -> Vec<(String, String)> {
        sources
            .iter()
            .filter(|(name, _)| name.ends_with(".cu"))
            .cloned()
            .collect()
    }

    /// The call-site check over the real directory, the way the other translation-unit checks are run.
    fn call_site_check_over_the_walk() -> Result<(), String> {
        let sources = kernels_directory();
        let units = kernel_units(&sources);
        assert!(
            units.len() >= 6,
            "the walk must hand over at least the six kernels: {units:?}"
        );
        observing_paths_call_the_acquiring_poll(&kernels_as_refs(&units))
    }

    /// The translation-unit half of the guarantee, over the real directory: no source under kernels/
    /// defines the raw poll, the acquiring poll or the fence as a macro, and the shipped text is
    /// green. Cases (1) and (2) of the round-5 review are the two mutations that leave every call
    /// site looking exactly right.
    #[test]
    fn no_source_under_kernels_redefines_the_poll_or_the_fence() {
        let sources = kernels_directory();
        let refs = kernels_as_refs(&sources);
        assert_eq!(guarded_names_are_not_redefined(&refs), Ok(()));

        // (1) the helper redefined away, after the include, so the call site still names the helper.
        let mutation = MEGA_MHC.replacen(
            INCLUDE,
            &format!("#define {ACQUIRE_HELPER}(f, d) (1u)\n{INCLUDE}"),
            1,
        );
        assert_ne!(mutation, MEGA_MHC, "the mutation must apply");
        let error = guarded_names_are_not_redefined(&[("mega_mhc_sm121.cu", &mutation)])
            .expect_err("(1) defining the helper away must be rejected");
        assert!(
            error.starts_with("mega_mhc_sm121.cu:")
                && error.contains("#define spark_wait_for_flag_acquire"),
            "(1) the message must name the file, the line and the name: {error}"
        );

        // (2) the fence redefined away, above the include so the header sees it as well.
        let mutation = format!("#define __threadfence() ((void)0)\n{MEGA_MHC}");
        assert_ne!(mutation, MEGA_MHC, "the mutation must apply");
        let error = guarded_names_are_not_redefined(&[("mega_mhc_sm121.cu", &mutation)])
            .expect_err("(2) defining the fence away must be rejected");
        assert!(
            error.contains("mega_mhc_sm121.cu:1") && error.contains("#define __threadfence"),
            "(2) the message must name the file, the line and the name: {error}"
        );

        // The raw poll and the fence are declared and called in the shipped sources; only a #define of
        // them is a defect, so the real directory stays green.
        assert_eq!(guarded_names_are_not_redefined(&refs), Ok(()));
    }

    /// The other translation-unit check: the acquiring poll is defined once, in the shared header,
    /// and nowhere else. Case (3) of the round-5 review - a better-matching overload whose body has
    /// no fence - is the reason it exists.
    #[test]
    fn the_acquiring_poll_is_defined_once_and_in_the_header() {
        let sources = kernels_directory();
        let refs = kernels_as_refs(&sources);
        assert_eq!(the_acquiring_poll_is_defined_exactly_once(&refs), Ok(()));

        // (3) an overload after the include. Its first parameter is uint32_t *, which beats the
        // header's const volatile uint32_t * for the non-volatile hc_sync, and its body has no fence.
        let overload = MEGA_MHC.replacen(
            INCLUDE,
            &format!(
                "{INCLUDE}\n__device__ __forceinline__ uint32_t {ACQUIRE_HELPER}(uint32_t *flag, uint64_t deadline) {{ return {RAW_POLL}((const volatile uint32_t *)flag, deadline); }}"
            ),
            1,
        );
        assert_ne!(overload, MEGA_MHC, "the mutation must apply");
        let error = the_acquiring_poll_is_defined_exactly_once(&[
            (COMMON_HEADER_NAME, COMMON_HEADER),
            ("mega_mhc_sm121.cu", &overload),
        ])
        .expect_err("(3) a second definition must be rejected");
        assert!(
            error.contains("exactly once")
                && error.contains("m4_common_sm121.cuh:")
                && error.contains("mega_mhc_sm121.cu:"),
            "(3) the message must name both definitions: {error}"
        );

        // (3b) a declaration that changes the signature does the same from a distance.
        let declaration = MEGA_MHC.replacen(
            INCLUDE,
            &format!(
                "{INCLUDE}\n__device__ uint32_t {ACQUIRE_HELPER}(uint32_t *flag, uint64_t deadline);"
            ),
            1,
        );
        assert_ne!(declaration, MEGA_MHC, "the mutation must apply");
        let error =
            the_acquiring_poll_is_defined_exactly_once(&[("mega_mhc_sm121.cu", &declaration)])
                .expect_err("(3b) a declaration of the helper outside the header must be rejected");
        assert!(
            error.contains("declaration of") && error.contains(COMMON_HEADER_NAME),
            "(3b) the message must say what the problem is: {error}"
        );
        assert_eq!(the_acquiring_poll_is_defined_exactly_once(&refs), Ok(()));
    }

    /// The five guard-green evasions from the round-5 review, each applied to a copy of the shipped
    /// text and each asserting the verdict of the check that now owns it. Before this round all five
    /// left the guard green: (1) and (2) rewrote the helper or the fence out of the translation unit,
    /// (3) added a better-matching overload, (4) parenthesised or dereferenced the callee designator,
    /// (5) hid an early return behind a form feed in the splice the text parser used to miss.
    #[test]
    fn the_round_five_evasions_are_rejected_by_the_named_check() {
        // (1) the helper defined away after the include.
        let mutation = MEGA_MHC.replacen(
            INCLUDE,
            &format!("{INCLUDE}\n#define {ACQUIRE_HELPER}(f, d) (1u)"),
            1,
        );
        assert_ne!(mutation, MEGA_MHC, "the mutation must apply");
        let error = guarded_names_are_not_redefined(&[("mega_mhc_sm121.cu", &mutation)])
            .expect_err("(1) a define of the helper must be rejected");
        assert!(
            error.contains("#define spark_wait_for_flag_acquire"),
            "(1) {error}"
        );

        // (2) the fence defined away above the include.
        let mutation = format!("#define __threadfence() ((void)0)\n{MEGA_MHC}");
        let error = guarded_names_are_not_redefined(&[("mega_mhc_sm121.cu", &mutation)])
            .expect_err("(2) a define of the fence must be rejected");
        assert!(error.contains("#define __threadfence"), "(2) {error}");

        // (3) a better-matching overload after the include, with no fence in its body.
        let mutation = MEGA_MHC.replacen(
            INCLUDE,
            &format!(
                "{INCLUDE}\n__device__ __forceinline__ uint32_t {ACQUIRE_HELPER}(uint32_t *flag, uint64_t deadline) {{ return {RAW_POLL}((const volatile uint32_t *)flag, deadline); }}"
            ),
            1,
        );
        assert_ne!(mutation, MEGA_MHC, "the mutation must apply");
        let error = the_acquiring_poll_is_defined_exactly_once(&[
            (COMMON_HEADER_NAME, COMMON_HEADER),
            ("mega_mhc_sm121.cu", &mutation),
        ])
        .expect_err("(3) a second definition must be rejected");
        assert!(error.contains("found 2 definition(s)"), "(3) {error}");

        // (4) the callee designator parenthesised, then dereferenced: both are still calls to the raw
        // poll, and the preprocessor rewrites neither of them.
        let call = format!("{ACQUIRE_HELPER}(&hc_sync[1], deadline)");
        for spelling in [
            format!("({RAW_POLL})(&hc_sync[1], deadline)"),
            format!("(*{RAW_POLL})(&hc_sync[1], deadline)"),
        ] {
            let mutation = MEGA_MHC.replace(&call, &spelling);
            assert_ne!(mutation, MEGA_MHC, "the mutation must apply: {spelling}");
            let error =
                observing_paths_call_the_acquiring_poll(&[("mega_mhc_sm121.cu", &mutation)])
                    .expect_err(
                        "(4) the parenthesised or dereferenced designator is a raw poll call",
                    );
            assert!(
                error.contains("must call spark_wait_for_flag_acquire"),
                "(4) {error}"
            );
        }

        // (5) an early return spliced in front of the fence with a form feed between the backslash and
        // the newline. clang, gcc and nvcc all splice it, so the parser has to as well; once it does,
        // the return that stands between the poll and the fence is visible.
        let poll_statement = "spark_wait_for_flag(flag, deadline);";
        let spliced = format!("    ret\\{}\nurn observed;", '\u{c}');
        let mutation =
            COMMON_HEADER.replace(poll_statement, &format!("{poll_statement}\n{spliced}"));
        assert_ne!(mutation, COMMON_HEADER, "the mutation must apply");
        let error = acquire_helper_fences_unconditionally(&mutation)
            .expect_err("(5) a spliced early return must be seen");
        assert!(
            error.contains("stands between the poll and the acquire"),
            "(5) {error}"
        );
        assert!(
            acquire_fence_after_poll(&mutation, HELPER_POLL).is_err(),
            "(5) the reachability lint must reject it too"
        );

        // The shipped text is still green after all five fixtures: nothing was written back.
        assert_eq!(call_site_check_over_the_walk(), Ok(()));
        assert_eq!(acquire_helper_fences_unconditionally(COMMON_HEADER), Ok(()));
        assert_eq!(guarded_names_are_not_redefined(&KERNELS), Ok(()));
    }

    /// Round-30 A3: the MISSING-POLL family, pinned by name. A helper body with no poll at all - and
    /// one whose only poll stands in a definitely-disabled region - is refused by both checks, and the
    /// two refusals are two different true sentences rather than one shared sentence: the structural
    /// check reports the body's missing mention, the lint reports that there is no poll of that call in
    /// the compiled body. The documentation listed the DECLARATION family as the only one the two
    /// answered differently, which this probe falsifies; both are listed now.
    #[test]
    fn the_missing_poll_family_speaks_in_two_voices() {
        let poll_line = "    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n";
        let without_the_poll = COMMON_HEADER.replace(poll_line, "");
        assert_ne!(without_the_poll, COMMON_HEADER, "the mutation must apply");
        let structural = acquire_helper_fences_unconditionally(&without_the_poll)
            .expect_err("a body with no poll is refused by the structural check");
        // Round-31 A4: the WHOLE sentence, not a prefix and a suffix. A family is pinned when its
        // message is; a prefix assertion is how a changed reason hides behind a matching opening.
        assert_eq!(
            structural,
            format!(
                "{ACQUIRE_HELPER} does not mention {RAW_POLL} in its body, so there is no \
                 observation for the fence to pair with; {CANONICAL_HINT}"
            ),
            "the structural check reports the missing mention in full"
        );
        let lint = acquire_fence_after_poll(&without_the_poll, HELPER_POLL)
            .expect_err("a body with no poll is refused by the lint too");
        assert_eq!(
            lint,
            format!(
                "no poll of {HELPER_POLL} in {ACQUIRE_HELPER}'s compiled body (a mention inside a \
                 comment is not a poll, and a poll the preprocessor may not compile is not one this \
                 lint follows); {CANONICAL_HINT}"
            ),
            "the lint reports that there is no poll of that call in full"
        );
        assert_ne!(
            lint, structural,
            "the two sentences differ, which is what makes this a two-voice family"
        );
        // The same shape with the poll behind `#if 0`. The lint reads the compiled body, so it
        // refuses there with the same missing-poll sentence; what the structural check does with the
        // text is pinned below rather than assumed, because the two readings disagree about it and a
        // text the two disagree about belongs in no table that asserts their messages are equal.
        let disabled = COMMON_HEADER.replace(
            poll_line,
            "#if 0\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n#endif\n",
        );
        assert_ne!(disabled, COMMON_HEADER, "the mutation must apply");
        let lint = acquire_fence_after_poll(&disabled, HELPER_POLL)
            .expect_err("a definitely-disabled poll is not a poll");
        assert_eq!(
            lint,
            format!(
                "no poll of {HELPER_POLL} in {ACQUIRE_HELPER}'s compiled body (a mention inside a \
                 comment is not a poll, and a poll the preprocessor may not compile is not one this \
                 lint follows); {CANONICAL_HINT}"
            ),
            "the lint names the missing poll in full"
        );
        assert_eq!(
            acquire_helper_fences_unconditionally(&disabled),
            Err(format!(
                "{ACQUIRE_HELPER} does not mention {RAW_POLL} in its body, so there is no \
                 observation for the fence to pair with; {CANONICAL_HINT}"
            )),
            "the structural check reads the definitely-disabled poll as no poll either"
        );
        // Round-31 A4, the coverage the battery's kd18 case 1 found missing: a MENTION of the
        // designator is not a CALL of it, so a body that only takes its address has no poll and both
        // checks refuse. Without this text both checks could start counting every mention as a poll
        // and every table in this file would still be green.
        let mentioned_not_called = COMMON_HEADER.replace(
            poll_line,
            "    const auto *mentioned = &spark_wait_for_flag;\n    (void)mentioned;\n",
        );
        assert_ne!(
            mentioned_not_called, COMMON_HEADER,
            "the mutation must apply"
        );
        let structural = acquire_helper_fences_unconditionally(&mentioned_not_called)
            .expect_err("a mention that is not a call is not a poll");
        // The sentence names the line the scan read. That number is derived from this text's own
        // numbering, so the pin covers every word of the reason and checks the number is a number.
        let (line, sentence) = structural
            .split_once(": ")
            .expect("the sentence names the line it read first");
        assert!(
            line.strip_prefix("line ")
                .is_some_and(|number| number.chars().all(|c| c.is_ascii_digit())),
            "the sentence starts with the physical line it read: {structural}"
        );
        assert_eq!(
            sentence,
            format!(
                "{ACQUIRE_HELPER}'s body mentions {RAW_POLL} in a shape this scan will not certify \
                 as a call of it - a name that is only declared is not an observation; the shared \
                 header writes the call as `spark_wait_for_flag(flag, deadline)` (with the helper's \
                 own parameter names, since the arguments are not read); {CANONICAL_HINT}"
            ),
            "the structural check reports the uncalled mention in full"
        );
        let lint = acquire_fence_after_poll(&mentioned_not_called, HELPER_POLL)
            .expect_err("a mention that is not a call is refused by the lint too");
        let (line, sentence) = lint
            .split_once(": ")
            .expect("the lint names the line it read first too");
        assert!(
            line.strip_prefix("line ")
                .is_some_and(|number| number.chars().all(|c| c.is_ascii_digit())),
            "the lint names the physical line it read: {lint}"
        );
        assert_eq!(
            sentence,
            format!(
                "the only mention of {RAW_POLL} in {ACQUIRE_HELPER}'s body is not a CALL of it, \
                 so this lint cannot certify that the fence follows an observation; the shared header \
                 writes the call as `spark_wait_for_flag(flag, deadline)` (with the helper's own \
                 parameter names - the arguments are not read), and this scan reads a mention as the \
                 observation when the designator is called and nothing in front of it makes the same \
                 tokens a declaration; {CANONICAL_HINT}"
            ),
            "the lint reports the uncalled mention in full"
        );
    }
    /// The round-5 E2 finding: two polls with one fence between them. The acquire pairs with exactly
    /// one observation, so a later poll would be read with no fence after it.
    #[test]
    fn a_second_poll_in_the_helper_body_is_rejected() {
        let second = "spark_wait_for_flag(flag, deadline);";
        let mutation = COMMON_HEADER.replace(
            second,
            "spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    (void)spark_wait_for_flag(flag, deadline);",
        );
        assert_ne!(mutation, COMMON_HEADER, "the mutation must apply");
        let error = acquire_helper_fences_unconditionally(&mutation)
            .expect_err("a poll-fence-poll body must be rejected");
        assert!(
            error.contains("mentions the poll 2 times")
                && error.contains("cannot tell which observation the fence pairs with"),
            "the message must name the count and the rule: {error}"
        );
    }

    /// The round-5 E5 finding: a parenthesised or integer-suffixed literal condition is folded, so a
    /// poll and a fence inside #if (1) are certainly compiled and inside #if ((0)) certainly not. A
    /// bare-token read rejected the first two, which is the false rejection this test pins.
    #[test]
    fn a_parenthesised_or_suffixed_literal_condition_is_folded() {
        let conditional = |condition: &str| {
            format!(
                "void k() {{\n#if {condition}\n    {MEGA_POLL});\n    __threadfence();\n#endif\n}}\n"
            )
        };
        for condition in ["1", "(1)", "((1))", "1u", "1L", "1UL", " ( 1 ) "] {
            let source = conditional(condition);
            assert_eq!(
                acquire_fence_after_poll(&source, MEGA_POLL),
                Ok(()),
                "#if {condition} is definitely compiled, so the poll and its fence are both there"
            );
        }
        for condition in ["0", "(0)", "((0))", "0u", "0L"] {
            let source = conditional(condition);
            assert!(
                acquire_fence_after_poll(&source, MEGA_POLL).is_err(),
                "#if {condition} is definitely not compiled, so there is no poll to find"
            );
        }
        // An expression is still unknown: nothing is read as active just because it starts with 1.
        let expression = conditional("1 - 1");
        assert!(
            acquire_fence_after_poll(&expression, MEGA_POLL).is_err(),
            "#if 1 - 1 must not be folded to active"
        );
    }

    /// The accepted limits and the conservative rejections, pinned one by one.
    ///
    /// The first half pins shapes that are green ON PURPOSE, so a later round cannot mistake a limit
    /// for an oversight without deleting the assertion that says so. The second half pins shapes the
    /// scan cannot read: those are ERRORS by design, and this test says so with its assertions rather
    /// than claiming every assertion in it is green. The module documentation lists both kinds.
    #[test]
    fn the_accepted_limits_are_pinned() {
        // A comment may name the poll, the helper or the fence: comments are stripped before any
        // check reads tokens, so a name in prose is neither a call nor a definition.
        let commented = format!(
            "// the publisher calls {RAW_POLL} without an acquire; the observer calls {ACQUIRE_HELPER}\n{INCLUDE}\n"
        );
        assert!(
            observing_paths_call_the_acquiring_poll(&[("commented_sm121.cu", &commented)]).is_ok(),
            "a comment must not read as a call"
        );
        assert!(guarded_names_are_not_redefined(&[("commented_sm121.cu", &commented)]).is_ok());

        // A definitely-disabled region between the poll and the fence is ignored, because it executes
        // nothing: the fence after the #endif is the acquire that counts.
        let disabled = COMMON_HEADER.replace(
            HELPER_ANCHOR,
            "#if 0\n    __threadfence();\n#endif\n    __threadfence();\n    return observed;",
        );
        assert_ne!(disabled, COMMON_HEADER, "the mutation must apply");
        assert!(
            acquire_helper_fences_unconditionally(&disabled).is_ok(),
            "an #if 0 region is not compiled, so the fence outside it is the acquire"
        );
        assert!(acquire_fence_after_poll(&disabled, HELPER_POLL).is_ok());

        // A #define whose NAME is not one of the three guarded names is not itself a failure. If it
        // aliases the helper, the acquire survives the expansion, and what pins the shipped kernel is
        // the call-site shape test, which requires the helper by name. A macro whose BODY spells the
        // raw poll is a different matter and is rejected, because that token is in the source and no
        // expansion is needed to see it - the check reads directive bodies too.
        let alias = "#define SPARK_M4_ACQ spark_wait_for_flag_acquire\n__global__ void k() { const uint32_t ready = SPARK_M4_ACQ(&hc_sync[1], deadline); }\n";
        assert!(guarded_names_are_not_redefined(&[("alias_sm121.cu", alias)]).is_ok());
        assert!(observing_paths_call_the_acquiring_poll(&[("alias_sm121.cu", alias)]).is_ok());
        let body_with_poll = "#define SPARK_M4_LOCAL_POLL() spark_wait_for_flag(&hc_sync[1], deadline)\n__global__ void k() { const uint32_t ready = SPARK_M4_LOCAL_POLL(); }\n";
        assert!(
            observing_paths_call_the_acquiring_poll(&[("aliased_sm121.cu", body_with_poll)])
                .is_err(),
            "a macro body that spells the raw poll is visible to the call-site check"
        );

        // (4) The checks read the text they are handed and never follow an #include, so a kernel that
        // includes a header from outside kernels/ is judged on what is written in it. This is the
        // limit, pinned green: what covers the real tree is the directory walk below - every file
        // under kernels/ reaches these checks - plus the baseline compile, which compiles the
        // translation unit nvcc sees (the include is followed there, by the compiler).
        let out_of_tree = format!(
            "#include \"spark_m4_elsewhere.h\"\n{INCLUDE}\n__global__ void k() {{ const uint32_t ready = {ACQUIRE_HELPER}(&hc_sync[1], deadline); }}\n"
        );
        assert_eq!(
            guarded_names_are_not_redefined(&[("out_of_tree_sm121.cu", &out_of_tree)]),
            Ok(())
        );
        assert_eq!(
            observing_paths_call_the_acquiring_poll(&[("out_of_tree_sm121.cu", &out_of_tree)]),
            Ok(())
        );
        // The definition the include would pull in is only read when the file itself is handed over.
        let elsewhere = format!("#define {RAW_POLL}(f, d) (0u)\n");
        assert!(guarded_names_are_not_redefined(&[("elsewhere.h", &elsewhere)]).is_err());
        // The walk is not truncated: all six kernels and the shared header are in it, and the whole
        // directory is green.
        let sources = kernels_directory();
        for name in [
            COMMON_HEADER_NAME,
            "mega_mhc_sm121.cu",
            "b12x_mxfp8_dense_sm121.cu",
            "b12x_mxfp4_moe_sm121.cu",
            "flash_mla_csa2_sm121.cu",
            "b12x_mxfp8_wo_a_sm121.cu",
            "cert_lm_head_sm121.cu",
        ] {
            assert!(
                sources.iter().any(|(found, _)| found == name),
                "the walk must find {name}: {sources:?}"
            );
        }
        assert_eq!(
            guarded_names_are_not_redefined(&kernels_as_refs(&sources)),
            Ok(())
        );

        // A comment or a blank line between the poll and the fence changes nothing: comments are
        // stripped before tokens exist and a blank line produces none.
        let commented_acquire = COMMON_HEADER.replace(
            "    __threadfence();",
            "    // the acquire\n\n    __threadfence();",
        );
        assert_ne!(commented_acquire, COMMON_HEADER, "the mutation must apply");
        assert_eq!(
            acquire_helper_fences_unconditionally(&commented_acquire),
            Ok(()),
            "a comment and a blank line are not statements"
        );
        assert_eq!(
            acquire_fence_after_poll(&commented_acquire, HELPER_POLL),
            Ok(())
        );

        // The fence may be the last statement of the body: nothing has to follow the acquire.
        let fence_last = COMMON_HEADER.replace(HELPER_ANCHOR, "    __threadfence();");
        assert_ne!(fence_last, COMMON_HEADER, "the mutation must apply");
        assert_eq!(acquire_helper_fences_unconditionally(&fence_last), Ok(()));
        assert_eq!(acquire_fence_after_poll(&fence_last, HELPER_POLL), Ok(()));

        // An unused macro whose body contains a control keyword is not a defect: only a USE of it can
        // change the shape of the source, which is the limit the lint documents.
        let unused_macro = format!("#define SPARK_M4_UNUSED_BAIL() return 0u\n{COMMON_HEADER}");
        assert_eq!(
            acquire_fence_after_poll(&unused_macro, HELPER_POLL),
            Ok(()),
            "an unused macro is not a statement"
        );
        assert_eq!(acquire_helper_fences_unconditionally(&unused_macro), Ok(()));

        // --- Conservative rejections: errors by design, not limits. Each one asserts its REASON,
        // not just that something failed: a bare is_err() stays green if the check starts rejecting
        // the shape for an unrelated mistake. ---
        // (5) A fence spelled as inline asm is not recognised as the fence, so the reason is the
        // missing fence.
        let inline_asm = COMMON_HEADER.replace("__threadfence();", "asm volatile(\"membar.gl;\");");
        assert_ne!(inline_asm, COMMON_HEADER, "the mutation must apply");
        let error = acquire_helper_fences_unconditionally(&inline_asm)
            .expect_err("a fence the scan cannot recognise is rejected rather than trusted");
        assert!(
            error.contains("no reachable __threadfence() follows the poll of spark_wait_for_flag"),
            "the reason must be the missing fence: {error}"
        );
        assert!(
            acquire_fence_after_poll(&inline_asm, HELPER_POLL).is_err(),
            "the reachability lint rejects it too, for the same reason"
        );

        // (6) A fence reached through a function pointer: the name is there as an initialiser and the
        // call is to the pointer, so the reason is the bare-call rule, not a missing fence.
        let through_pointer = COMMON_HEADER.replace(
            "    __threadfence();",
            "    void (*spark_fence)(void) = __threadfence;\n    spark_fence();",
        );
        assert_ne!(through_pointer, COMMON_HEADER, "the mutation must apply");
        let error = acquire_helper_fences_unconditionally(&through_pointer)
            .expect_err("a fence reached through a pointer is not a fence the scan can read");
        assert!(
            error.contains("is not a bare call statement"),
            "the reason must be the bare-call rule: {error}"
        );
        assert!(
            acquire_fence_after_poll(&through_pointer, HELPER_POLL).is_err(),
            "the reachability lint rejects it too"
        );

        // (7) A fence hidden behind a macro of its own: SPARK_M4_FENCE is not one of the guarded
        // names, so the definition is not what rejects this - the body no longer calls the fence.
        let through_macro = format!(
            "#define SPARK_M4_FENCE() __threadfence()\n{}",
            COMMON_HEADER.replace("    __threadfence();", "    SPARK_M4_FENCE();")
        );
        let error = acquire_helper_fences_unconditionally(&through_macro)
            .expect_err("a fence behind a macro is not a fence the scan can read");
        assert!(
            error.contains("no reachable __threadfence() follows the poll of spark_wait_for_flag"),
            "the reason must be the missing fence: {error}"
        );
        assert!(
            acquire_fence_after_poll(&through_macro, HELPER_POLL).is_err(),
            "the reachability lint rejects it too"
        );
    }

    /// The round-7 A2 false rejection: the fence spelled `(__threadfence)();`. That is the same call
    /// in C++ - nvcc accepts it and the PTX carries the same four membar lines as the bare spelling -
    /// so both checks that read the fence's own spelling accept it. Rejecting it made the guard a
    /// hazard to a developer debugging a real acquire.
    #[test]
    fn a_parenthesised_fence_call_is_the_fence() {
        for spelling in ["(__threadfence)();", "((__threadfence))();"] {
            let source = COMMON_HEADER.replace("__threadfence();", spelling);
            assert_ne!(source, COMMON_HEADER, "the mutation must apply");
            assert_eq!(
                acquire_helper_fences_unconditionally(&source),
                Ok(()),
                "{spelling} is still the fence, so the helper's body is poll-then-fence"
            );
            assert!(
                acquire_fence_after_poll(&source, HELPER_POLL).is_ok(),
                "{spelling} must read as a bare call statement of the fence"
            );
        }
        // The spelling is accepted as a *statement*. Glued onto an expression it is still part of
        // that expression, so the statement-start rule keeps rejecting it: accepting the spelling
        // must not accept `1 + (__threadfence)();`.
        let glued = COMMON_HEADER.replace("__threadfence();", "1 + (__threadfence)();");
        assert_ne!(glued, COMMON_HEADER, "the mutation must apply");
        assert!(
            acquire_helper_fences_unconditionally(&glued).is_err(),
            "a fence glued to an expression is not a statement"
        );
        assert!(
            acquire_fence_after_poll(&glued, HELPER_POLL).is_err(),
            "the lint must reject a fence that is part of an expression"
        );
    }

    /// Diagnostics name the line in the FILE, not the line in the phase-2-spliced text.
    ///
    /// The shared header has 23 backslash-newline splices above the helper, so a spliced-text line is
    /// 23 lines early there: the round-6 guard reported the shipped poll at
    /// m4_common_sm121.cuh:484 while `grep -n` finds it at 507. A reader who follows a guard message
    /// to a line that does not hold the defect stops trusting the guard, so the physical line is
    /// pinned on both kinds of message.
    #[test]
    fn diagnostics_name_the_physical_line_not_the_spliced_one() {
        // A #define whose spliced line is 3 and whose physical line is 4: the first line continuesto
        // the second, so the splice removes one newline above the define.
        let spliced =
            format!("// filler \\\n more filler\n// third\n#define {RAW_POLL}(f, d) (0u)\n");
        let error = guarded_names_are_not_redefined(&[("spliced_sm121.cu", &spliced)])
            .expect_err("a #define of the raw poll must be rejected");
        assert!(
            error.starts_with("spliced_sm121.cu:4:"),
            "the define is on physical line 4 and spliced line 3: {error}"
        );

        // The shipped helper: two polls in its body, reported at their physical lines.
        let header = COMMON_HEADER;
        let poll_line = "    const uint32_t observed = spark_wait_for_flag(flag, deadline);";
        let physical = header
            .lines()
            .position(|line| line == poll_line)
            .map(|index| index + 1)
            .expect("the shipped helper polls exactly once on its own line");
        let second_poll = header.replace(
            poll_line,
            "    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    (void)spark_wait_for_flag(flag, deadline);",
        );
        assert_ne!(second_poll, header, "the mutation must apply");
        let error = acquire_helper_fences_unconditionally(&second_poll)
            .expect_err("two polls are rejected");
        assert!(
            error.contains(&format!("line {physical} and line {}", physical + 2)),
            "the two polls are at physical lines {physical} and {}: {error}",
            physical + 2
        );
        assert!(
            !error.contains("line 484"),
            "the spliced-text line must not appear in a diagnostic: {error}"
        );
    }

    /// The six findings of the review that asked for the canonical-form contract: the three holes,
    /// each of which nvcc compiled to fewer barriers than the baseline without a word from either
    /// check, and the three false rejections, each of which nvcc compiles to the baseline.
    ///
    /// Every verdict below is one the reviewer measured; the comments carry the numbers.
    #[test]
    fn the_holes_and_the_false_rejections_are_closed() {
        // A-1: a `;` nested inside a brace does not end the operator's statement. nvcc compiled this
        // body to 38624 bytes / 3 PTB membar lines / sha d64426bf... (pre-change), byte for byte the conditional
        // fence the round-3 shape produces and one barrier short of the 70432 / 4 / 605321041ac5ec1e baseline.
        let statement_expression = COMMON_HEADER.replace(
            HELPER_ANCHOR,
            r"    (observed == 0u) && ({ int z = 0; __threadfence(); z; });
    return observed;",
        );
        assert_ne!(statement_expression, COMMON_HEADER, "A-1 must apply");
        let error = acquire_fence_after_poll(&statement_expression, HELPER_POLL)
            .expect_err("A-1 lint: the fence is the \"&&\" operand's own statement");
        assert!(
            error.contains("\"&&\" stands between the poll and the acquire"),
            "A-1 lint: {error}"
        );
        let error = acquire_helper_fences_unconditionally(&statement_expression)
            .expect_err("A-1 structural: the same operator is in front of the acquire");
        assert!(
            error.contains("\"&&\" stands between the poll and the acquire"),
            "A-1 structural: {error}"
        );
        // The round-3 shape, whose inner statement is the fence itself, stays rejected for the same
        // reason: the brace is not a statement boundary.
        let round_three = COMMON_HEADER.replace(
            HELPER_ANCHOR,
            r"    (observed == 0u) && ({ __threadfence(); 1; });
    return observed;",
        );
        assert!(
            acquire_fence_after_poll(&round_three, HELPER_POLL)
                .expect_err("A-1: the statement-expression fence")
                .contains("stands between the poll and the acquire"),
            "A-1: the round-3 statement expression must stay rejected"
        );
        // Non-vacuity: the round-8 A6 acceptance is untouched. A completed statement's operator is
        // still not a blocker, in either check.
        let earlier_statement = COMMON_HEADER.replace(
            HELPER_ANCHOR,
            r"    uint32_t masked = observed & 1u;
    bool ready = observed != 0u ? true : false;
    __threadfence();
    return observed;",
        );
        assert_eq!(
            acquire_helper_fences_unconditionally(&earlier_statement),
            Ok(())
        );
        assert_eq!(
            acquire_fence_after_poll(&earlier_statement, HELPER_POLL),
            Ok(())
        );

        // A-2: a using-declaration re-binds the name exactly as a #define does. With
        // `namespace X { __device__ void __threadfence() {} }` and this declaration, nvcc compiled
        // the kernel to 70264 bytes / 2 PTB membar lines / sha 42454ad1..., the same hole as the
        // qualified call the round-9 review found, and both checks accepted it.
        let using_declaration = COMMON_HEADER.replace(
            HELPER_ANCHOR,
            r"    using X::__threadfence;
    __threadfence();
    return observed;",
        );
        assert_ne!(using_declaration, COMMON_HEADER, "A-2 must apply");
        let error = acquire_helper_fences_unconditionally(&using_declaration)
            .expect_err("A-2 structural: the declaration re-binds the fence's name");
        assert!(
            error.contains("using-declaration") && error.contains("__threadfence"),
            "A-2 structural: {error}"
        );
        let error = acquire_fence_after_poll(&using_declaration, HELPER_POLL)
            .expect_err("A-2 lint: the declaration re-binds the fence's name");
        assert!(error.contains("using-declaration"), "A-2 lint: {error}");
        let error = guarded_names_are_not_redefined(&[("m4_common_sm121.cuh", &using_declaration)])
            .expect_err("A-2 translation unit: the declaration re-binds a guarded name");
        assert!(
            error.starts_with("m4_common_sm121.cuh:") && error.contains("using-declaration"),
            "A-2 must name the file and the line: {error}"
        );
        // The same declaration above the poll, which is the other placement the review asked for.
        let using_before_the_poll = format!("using X::__threadfence;\n{COMMON_HEADER}");
        assert!(
            guarded_names_are_not_redefined(&[("m4_common_sm121.cuh", &using_before_the_poll)])
                .is_err()
        );
        assert!(acquire_helper_fences_unconditionally(&using_before_the_poll).is_err());
        assert!(acquire_fence_after_poll(&using_before_the_poll, HELPER_POLL).is_err());
        // A using-DIRECTIVE is caught in the case a text scanner can see: the namespace is defined in
        // the scanned source and holds a member with the guarded name.
        let using_directive = format!(
            "namespace X {{ __device__ __forceinline__ void __threadfence() {{}} }}\nusing namespace X;\n{COMMON_HEADER}"
        );
        let error = acquire_helper_fences_unconditionally(&using_directive)
            .expect_err("A-2 using-directive: it imports the namespace that holds the name");
        assert!(
            error.contains("using-directive")
                && error.contains("changes what the unqualified __threadfence can denote"),
            "A-2 using-directive: {error}"
        );

        // A-3: a lambda's body is not a bare block. nvcc compiled the uncalled closure into 70264
        // bytes / 2 PTB membar lines / sha 42454ad1... and the called one byte for byte to the
        // 70432 / 4 / 605321041ac5ec1e baseline; the checks reject both, which is the canonical-form policy
        // (the closure is not the acquire, and the message says what to write).
        let uncalled_closure = COMMON_HEADER.replace(
            HELPER_ANCHOR,
            r"    auto closure = [] { __threadfence(); };
    (void)closure;
    return observed;",
        );
        let error = acquire_helper_fences_unconditionally(&uncalled_closure)
            .expect_err("A-3 structural: the fence is inside the closure");
        assert!(
            error.contains("nested brace depth"),
            "A-3 structural: {error}"
        );
        let error = acquire_fence_after_poll(&uncalled_closure, HELPER_POLL)
            .expect_err("A-3 lint: the fence is inside the closure");
        assert!(
            error.contains("nested brace depth"),
            "A-3 lint: the two checks share the sentence now: {error}"
        );
        let called_closure = COMMON_HEADER.replace(
            HELPER_ANCHOR,
            r"    auto closure = [] { __threadfence(); };
    closure();
    return observed;",
        );
        let error = acquire_fence_after_poll(&called_closure, HELPER_POLL)
            .expect_err("A-3 lint: a closure body is not the canonical acquire");
        assert!(
            error.contains("nested brace depth") && error.ends_with(CANONICAL_HINT),
            "A-3 lint, called: {error}"
        );
        // A type's body is the same class of body, in both checks.
        let type_body = COMMON_HEADER.replace(
            HELPER_ANCHOR,
            r"    struct Probe { void run() { __threadfence(); } };
    return observed;",
        );
        assert!(acquire_helper_fences_unconditionally(&type_body).is_err());
        assert!(acquire_fence_after_poll(&type_body, HELPER_POLL).is_err());

        // A-4: four false rejections. A member function, another scope's function, a declaration and a
        // name this source defines as a harmless macro are not the library function that never
        // returns; nvcc compiled each of these four bodies to the baseline cubin byte for byte
        // (70432 / 4 / 605321041ac5ec1e), while the bare `abort();` and the host-qualified forms do not
        // compile in device code at all.
        let member_abort =
            COMMON_HEADER.replace(HELPER_ANCHOR, &format!("    s.abort(1u);\n{HELPER_ANCHOR}"));
        let qualified_exit =
            COMMON_HEADER.replace(HELPER_ANCHOR, &format!("    X::_Exit(1);\n{HELPER_ANCHOR}"));
        let declaration = COMMON_HEADER.replace(
            HELPER_ANCHOR,
            &format!("    void abort();\n{HELPER_ANCHOR}"),
        );
        let macro_noop = format!("#define abort() do {{}} while(0)\n{COMMON_HEADER}")
            .replace(HELPER_ANCHOR, &format!("    abort();\n{HELPER_ANCHOR}"));
        for (label, fixture) in [
            ("a member function", &member_abort),
            ("a qualified name", &qualified_exit),
            ("a declaration", &declaration),
            ("a macro that expands to nothing", &macro_noop),
        ] {
            assert_ne!(fixture.as_str(), COMMON_HEADER, "A-4 {label} must apply");
            assert_eq!(
                acquire_helper_fences_unconditionally(fixture),
                Ok(()),
                "A-4 structural must accept {label}"
            );
            assert_eq!(
                acquire_fence_after_poll(fixture, HELPER_POLL),
                Ok(()),
                "A-4 lint must accept {label}"
            );
        }
        // The stated frontier, pinned as a choice rather than left silent: a qualified call this text
        // scanner cannot resolve is not a blocker, and the committed cubin baseline is what settles
        // the compiled acquire. The global `::abort` and the bare `abort()` stay blockers, so the
        // rule is not "the name never counts".
        let qualified_std = COMMON_HEADER.replace(
            HELPER_ANCHOR,
            &format!("    std::abort();\n{HELPER_ANCHOR}"),
        );
        assert_eq!(
            acquire_helper_fences_unconditionally(&qualified_std),
            Ok(())
        );
        assert_eq!(
            acquire_fence_after_poll(&qualified_std, HELPER_POLL),
            Ok(())
        );
        let global_abort =
            COMMON_HEADER.replace(HELPER_ANCHOR, &format!("    ::abort();\n{HELPER_ANCHOR}"));
        assert!(acquire_helper_fences_unconditionally(&global_abort).is_err());
        assert!(acquire_fence_after_poll(&global_abort, HELPER_POLL).is_err());
        let bare_abort =
            COMMON_HEADER.replace(HELPER_ANCHOR, &format!("    abort();\n{HELPER_ANCHOR}"));
        assert!(acquire_helper_fences_unconditionally(&bare_abort).is_err());
        assert!(acquire_fence_after_poll(&bare_abort, HELPER_POLL).is_err());

        // A-5: the preprocessor is positional, so a definition after the use cannot make the use a
        // `return`; treating it as if it stood at the top rejected real code, and nvcc compiled this
        // body to the baseline cubin byte for byte (70432 / 4 / 605321041ac5ec1e).
        const POLL: &str = "spark_wait_for_flag(&flag, deadline)";
        let definition_after_the_use = r"int FLAG;
spark_wait_for_flag(&flag, deadline);
(void)FLAG;
__threadfence();
#define FLAG return
";
        assert_eq!(
            acquire_fence_after_poll(definition_after_the_use, POLL),
            Ok(()),
            "A-5: a definition after the use is not in effect at the use"
        );
        // Non-vacuity: the same spelling with the definition before the use is still rejected, so the
        // rule is positional and not "macros are never suspicious".
        let definition_before_the_use = r"#define FLAG return
int FLAG;
spark_wait_for_flag(&flag, deadline);
(void)FLAG;
__threadfence();
";
        assert!(
            acquire_fence_after_poll(definition_before_the_use, POLL).is_err(),
            "A-5: a definition before the use is in effect at it"
        );

        // A-6: a fence reached through a cast expression. nvcc compiled this body to the baseline
        // cubin byte for byte (70432 / 605321041ac5ec1e / 4), so the rejection is a policy decision and not a
        // claim that the fence is conditional: the scanner does not resolve a cast, and both messages
        // name the canonical form to write instead.
        let cast_call = COMMON_HEADER.replace(
            HELPER_ANCHOR,
            r"    ((void (*)(void))__threadfence)();
    return observed;",
        );
        let error = acquire_helper_fences_unconditionally(&cast_call)
            .expect_err("A-6 structural: the cast is not resolved");
        assert!(
            (error.contains("not a bare call statement")
                || error.contains("part of a larger expression"))
                && error.ends_with(CANONICAL_HINT),
            "A-6 structural: {error}"
        );
        let error = acquire_fence_after_poll(&cast_call, HELPER_POLL)
            .expect_err("A-6 lint: the cast is not resolved");
        assert!(
            (error.contains("not a bare call statement")
                || error.contains("part of a larger expression"))
                && error.ends_with(CANONICAL_HINT),
            "A-6 lint: {error}"
        );
    }

    /// A long chain of macro names must not make the lint quadratic.
    ///
    /// The round-10 review measured the previous implementation at 0.7-1.0 s for a 400-link chain and
    /// still running after 120 s for a 4000-link one, because every pass re-scanned every body each
    /// time it learned a name. The reverse-edge index in [`MacroScope`] visits each definition once
    /// when it takes effect and each edge once when a name turns suspicious, so both sizes are
    /// instant; the assertion is deliberately loose, and the printed numbers are the evidence.
    #[test]
    fn a_long_macro_chain_does_not_stall_the_lint() {
        const POLL: &str = "spark_wait_for_flag(&flag, deadline)";
        for links in [400usize, 4000] {
            let mut source = String::from("#define CHAIN0 return\n");
            for link in 1..links {
                source.push_str(&format!("#define CHAIN{link} CHAIN{}\n", link - 1));
            }
            source.push_str(&format!(
                "spark_wait_for_flag(&flag, deadline);\nCHAIN{};\n__threadfence();\n",
                links - 1
            ));
            let started = std::time::Instant::now();
            let error =
                acquire_fence_after_poll(&source, POLL).expect_err("the chain ends in a return");
            let elapsed = started.elapsed();
            println!("{links}-link macro chain: {elapsed:?}");
            assert!(
                error.contains(&format!("CHAIN{} is a macro", links - 1)),
                "{links} links: {error}"
            );
            assert!(
                elapsed < std::time::Duration::from_secs(10),
                "{links} links took {elapsed:?}"
            );
        }
    }

    /// Round 11: every claim in the module documentation is pinned here, so the contract cannot drift
    /// away from the code again. The reviewer's finding was that the doc overstated what it
    /// enumerates; these are the shapes it enumerates, each with the verdict the checks actually give.
    #[test]
    fn the_written_claims_are_exact() {
        // ---- A1, accepted list: the shapes the code accepts that the doc did not name. nvcc compiles
        // every one of these to the baseline cubin (70432 bytes / 4 PTB membar lines / sha
        // 605321041ac5ec1e), which is why they are enumerated rather than rejected.
        for (label, body) in [
            (
                "a bare block with a statement before the fence",
                r"    { int i = 0; (void)i; __threadfence(); }",
            ),
            (
                "a statement expression whose fence is the whole statement",
                r"    ({ __threadfence(); });",
            ),
            ("nested bare blocks", r"    { { __threadfence(); } }"),
        ] {
            let source =
                COMMON_HEADER.replace(HELPER_ANCHOR, &format!("{body}\n    return observed;"));
            assert_ne!(
                source, COMMON_HEADER,
                "A1: the mutation must apply: {label}"
            );
            assert_eq!(
                acquire_helper_fences_unconditionally(&source),
                Ok(()),
                "A1 structural must accept {label}"
            );
            assert_eq!(
                acquire_fence_after_poll(&source, HELPER_POLL),
                Ok(()),
                "A1 lint must accept {label}"
            );
        }
        // A suspension point is not a path bound: the fence still runs when the coroutine resumes, and
        // the scan sees neither co_await nor co_yield. The doc lists it with this reason.
        let suspended = COMMON_HEADER.replace(
            HELPER_ANCHOR,
            r"    co_await observed;
    __threadfence();
    return observed;",
        );
        assert_eq!(
            acquire_helper_fences_unconditionally(&suspended),
            Ok(()),
            "A1 structural: co_await is not a path bound"
        );
        assert_eq!(
            acquire_fence_after_poll(&suspended, HELPER_POLL),
            Ok(()),
            "A1 lint: co_await is not a path bound"
        );

        // ---- A1 rejected list: a requires-expression is not executed and a try body is a nested
        // body, and the two checks AGREE about both. The lint used to step into a try body and accept
        // the fence there while the structural check rejected it; that disagreement is gone. Neither
        // shape can ship either: nvcc refuses the requires-expression with "identifier requires is
        // undefined" and the try body with "device code does not support exception handling".
        for (label, body, needle) in [
            (
                "a requires-expression holding the fence",
                r"    requires { __threadfence(); };",
                "requires-expression",
            ),
            (
                "a fence in a try body",
                r"    try { __threadfence(); } catch (...) { }",
                "try block",
            ),
        ] {
            let source =
                COMMON_HEADER.replace(HELPER_ANCHOR, &format!("{body}\n    return observed;"));
            let structural = acquire_helper_fences_unconditionally(&source)
                .expect_err("A1 structural: a nested body is not the acquire");
            let lint = acquire_fence_after_poll(&source, HELPER_POLL)
                .expect_err("A1 lint: a nested body is not the acquire");
            assert!(
                structural.contains(needle) && lint.contains(needle),
                "A1: both checks must name {label}: {structural} | {lint}"
            );
        }

        // ---- A2 rejected list, with a reason that is true. An empty if body cannot bound the path,
        // and nvcc compiled it to the baseline cubin byte for byte (70432 / 4 / 605321041ac5ec1e), so the
        // message must not claim the fence is not guaranteed: it says the scan does not read a
        // construct's body. A lambda whose return belongs to the LAMBDA is not in this list at all -
        // round 27 moved it to the ACCEPTANCE assertions below, where its baseline cubin belongs. The
        // conservative extras below are the same class.
        for (label, body) in [
            ("an empty if body", r"    if (observed != 0u) { }"),
            (
                "a switch that only breaks",
                r"    switch (observed) { case 1u: break; default: break; }",
            ),
            // The row that used to stand here - `auto f = [&] { if (...) return; }; (void)f;` before the
            // poll - is no longer in this list: round 27 A2 asserts it as ACCEPTANCE below, because the
            // return exits the LAMBDA and nvcc compiles the body to the baseline 70432 / 4 / 605321041ac5ec1e.
            (
                "an if whose condition is an operator",
                r"    if (observed != 0u && deadline == 0u) { }",
            ),
        ] {
            let source = COMMON_HEADER.replace(HELPER_ANCHOR, &format!("{body}\n{HELPER_ANCHOR}"));
            assert_ne!(
                source, COMMON_HEADER,
                "A2: the mutation must apply: {label}"
            );
            let structural = acquire_helper_fences_unconditionally(&source)
                .expect_err("A2 structural: a conditional construct is refused");
            let lint = acquire_fence_after_poll(&source, HELPER_POLL)
                .expect_err("A2 lint: a conditional construct is refused");
            for (check, error) in [("structural", &structural), ("lint", &lint)] {
                assert!(
                    error.contains("does not certify"),
                    "A2 {check}: the reason is a limitation, not a claim about the path: {error}"
                );
                assert!(
                    error.ends_with(CANONICAL_HINT),
                    "A2 {check}: a shape rejection names the canonical form: {error}"
                );
            }
        }
        // Round-27 A2: the lambda case asserts ACCEPTANCE, on both sides, because the return exits the
        // lambda, the statement completes, and nvcc compiles the body to the baseline 70432 / 4 /
        // 605321041ac5ec1e. A lambda body's exit before the poll is not this helper's path.
        {
            let body = r"    auto f = [&] { if (observed != 0u) return; };
    (void)f;
";
            let source = COMMON_HEADER.replace(HELPER_ANCHOR, &format!("{body}\n{HELPER_ANCHOR}"));
            assert_ne!(source, COMMON_HEADER, "A2: the mutation must apply");
            assert_eq!(
                acquire_helper_fences_unconditionally(&source),
                Ok(()),
                "Round-27 A2 structural: a lambda body's exit before the poll is not the helper's path"
            );
            assert_eq!(
                acquire_fence_after_poll(&source, HELPER_POLL),
                Ok(()),
                "Round-27 A2 lint: a lambda body's exit before the poll is not the helper's path"
            );
        }
        // Non-vacuity: the tokens that really do bound the path to the fence stay rejected.
        for (label, body) in [
            (
                "a return before the fence",
                r"    if (observed != 0u) return 0u;",
            ),
            ("a bare block holding a return", r"    { return 0u; }"),
            ("a loop before the fence", r"    for (;;) { }"),
        ] {
            let source = COMMON_HEADER.replace(HELPER_ANCHOR, &format!("{body}\n{HELPER_ANCHOR}"));
            assert!(
                acquire_helper_fences_unconditionally(&source).is_err(),
                "A2 structural must still reject {label}"
            );
            assert!(
                acquire_fence_after_poll(&source, HELPER_POLL).is_err(),
                "A2 lint must still reject {label}"
            );
        }

        // ---- A3: the fence-precedes-poll rejection is a placement rejection, so it carries the
        // canonical hint like the others. It did not, while the doc promised that every placement
        // rejection does.
        let reversed = format!(
            "__device__ __forceinline__ unsigned {ACQUIRE_HELPER}(const unsigned* flag, unsigned deadline) {{\n    __threadfence();\n    const uint32_t observed = {HELPER_POLL};\n    return observed;\n}}\n"
        );
        let error = acquire_helper_fences_unconditionally(&reversed)
            .expect_err("A3: a fence before the poll is not the acquire of that observation");
        assert!(
            error.ends_with(CANONICAL_HINT),
            "A3: the placement rejection must name the canonical form: {error}"
        );

        // ---- A4: the two limits the doc lists and the tests did not exercise. Neither is a shape the
        // scan could have recognised: the scan reads one source at a time, so a name defined in an
        // included file is not expanded, and a keyword built by token pasting is two identifiers to
        // the tokenizer. Measured here, against the 70432 / 4 / 605321041ac5ec1e baseline: a name defined in an
        // included file as a return and a ##-assembled return each compile the helper to 69072 bytes /
        // 2 PTB membar lines / sha 18f93bdc... - the reviewer's numbers, reproduced - which is why the
        // not-seen list is a list of consequences rather than a footnote.
        let from_an_included_header = COMMON_HEADER.replace(
            HELPER_ANCHOR,
            r"    M4_BAIL_IF_READY;
    __threadfence();
    return observed;",
        );
        assert_eq!(
            acquire_helper_fences_unconditionally(&from_an_included_header),
            Ok(()),
            "A4: a name this source does not define is not read as a macro"
        );
        assert_eq!(
            acquire_fence_after_poll(&from_an_included_header, HELPER_POLL),
            Ok(()),
            "A4: the lint expands only the macros of the source it is given"
        );
        let token_pasted_keyword = COMMON_HEADER.replace(
            HELPER_ANCHOR,
            r"#define GLUE(a, b) a##b
    GLUE(re, turn);
    __threadfence();
    return observed;",
        );
        assert_eq!(
            acquire_helper_fences_unconditionally(&token_pasted_keyword),
            Ok(()),
            "A4: a pasted keyword is two identifiers to the scanner"
        );
        assert_eq!(
            acquire_fence_after_poll(&token_pasted_keyword, HELPER_POLL),
            Ok(()),
            "A4: a pasted keyword is two identifiers to the scanner"
        );

        // ---- A5: the using-declaration reason. A using-declaration for the global barrier still
        // names the barrier and nvcc compiles that body to the baseline cubin byte for byte (70264 /
        // 4 / 605321041ac5ec1e), so the message must say the rejection is a policy one rather than claim the
        // name changed meaning; a using-declaration for a namespace member is the case where the
        // meaning really does change (70264 / 2 / 42454ad1740a315c).
        let global_using = COMMON_HEADER.replace(
            HELPER_ANCHOR,
            r"    using ::__threadfence;
    __threadfence();
    return observed;",
        );
        for (check, error) in [
            (
                "structural",
                acquire_helper_fences_unconditionally(&global_using)
                    .expect_err("A5: a using-declaration is refused"),
            ),
            (
                "lint",
                acquire_fence_after_poll(&global_using, HELPER_POLL)
                    .expect_err("A5: a using-declaration is refused"),
            ),
        ] {
            assert!(
                error.contains("policy rejection")
                    && error.contains("does not resolve using-declarations"),
                "A5 {check}: the reason must be the unresolved declaration: {error}"
            );
            assert!(
                error.ends_with(CANONICAL_HINT),
                "A5 {check}: the placement rejection must name the canonical form: {error}"
            );
        }
        let member_using = COMMON_HEADER.replace(
            HELPER_ANCHOR,
            r"    using X::__threadfence;
    __threadfence();
    return observed;",
        );
        let namespaced = format!(
            "namespace X {{ __device__ __forceinline__ void __threadfence() {{}} }}\n{member_using}"
        );
        let error = acquire_helper_fences_unconditionally(&namespaced)
            .expect_err("A5: a namespace member's name can resolve elsewhere");
        assert!(
            error.contains("brings X's __threadfence into scope"),
            "A5: the namespace case is named: {error}"
        );

        // ---- R12 A4: an attribute-specifier before the fence. Both placements compile to the
        // baseline cubin byte for byte (70432 / 4 / 605321041ac5ec1e, measured), so the fence is accepted
        // rather than refused, and the statement-start rule the two checks share now looks past the
        // whole [[...]] instead of reading its closing bracket as an expression.
        for (label, body) in [
            (
                "an attribute on the fence's own line",
                r"    [[likely]] __threadfence();",
            ),
            (
                "an attribute on the line above the fence",
                r"    [[likely]]
    __threadfence();",
            ),
        ] {
            let source =
                COMMON_HEADER.replace(HELPER_ANCHOR, &format!("{body}\n    return observed;"));
            assert_eq!(
                acquire_helper_fences_unconditionally(&source),
                Ok(()),
                "R12 A4 structural must accept {label}"
            );
            assert_eq!(
                acquire_fence_after_poll(&source, HELPER_POLL),
                Ok(()),
                "R12 A4 lint must accept {label}"
            );
        }

        // ---- R12 A1, first disagreement: a nested fence is not reported while a top-level fence
        // follows it. Both checks accept the shape, and nvcc compiled it to the baseline cubin byte
        // for byte (70432 / 4 / 605321041ac5ec1e); the lint used to reject the lambda's fence before it ever
        // reached the real acquire.
        let lambda_then_top = COMMON_HEADER.replace(
            HELPER_ANCHOR,
            r"    auto l = [&] { __threadfence(); };
    (void)l;
    __threadfence();
    return observed;",
        );
        assert_eq!(
            acquire_helper_fences_unconditionally(&lambda_then_top),
            Ok(()),
            "R12 A1 structural: the top-level fence is the acquire"
        );
        assert_eq!(
            acquire_fence_after_poll(&lambda_then_top, HELPER_POLL),
            Ok(()),
            "R12 A1 lint: a nested fence does not hide the top-level fence"
        );
        // The same door from the other side: with no top-level fence, the nested one is named.
        let lambda_only = COMMON_HEADER.replace(
            HELPER_ANCHOR,
            r"    auto l = [&] { __threadfence(); };
    (void)l;
    return observed;",
        );
        let structural_only = acquire_helper_fences_unconditionally(&lambda_only)
            .expect_err("R12 A1: no top-level fence");
        let lint_only = acquire_fence_after_poll(&lambda_only, HELPER_POLL)
            .expect_err("R12 A1: no top-level fence");
        // Round-24 A4: the two checks used to word this differently on purpose; the reason is the same
        // on both sides, so the sentence is shared. Round-27 A4 makes that shared sentence a PIN rather
        // than a claim: the whole message is asserted equal, so a rewording on one side cannot pass.
        assert_eq!(
            structural_only, lint_only,
            "R12 A1: the lambda-body-only fence is refused in the same words on both sides"
        );
        assert!(
            structural_only.contains("nested") && structural_only.ends_with(CANONICAL_HINT),
            "R12 A1: the nested fence is named, with the hint: {structural_only}"
        );

        // ---- R12 A1, second disagreement: control flow BEFORE the poll. The structural check reads
        // the whole helper body and the lint now reads the block the poll is in, so both refuse it
        // with the same reason and the same hint.
        for (label, body) in [
            (
                "a return before the poll",
                r"    if (deadline == 0u) return 0u;",
            ),
            ("a goto before the poll", r"    goto done;"),
            ("an unbounded loop before the poll", r"    for (;;) { }"),
        ] {
            // The control token goes BEFORE the poll, which is what this case is about: the fixture
            // inserts it in front of the poll statement, not in front of the fence.
            let poll_line = "    const uint32_t observed = spark_wait_for_flag(flag, deadline);";
            let source = COMMON_HEADER.replace(poll_line, &format!("{body}\n{poll_line}"));
            assert_ne!(
                source, COMMON_HEADER,
                "R12 A1: the mutation must apply: {label}"
            );
            for (check, error) in [
                (
                    "structural",
                    acquire_helper_fences_unconditionally(&source)
                        .expect_err("R12 A1: control before the poll"),
                ),
                (
                    "lint",
                    acquire_fence_after_poll(&source, HELPER_POLL)
                        .expect_err("R12 A1: control before the poll"),
                ),
            ] {
                assert!(
                    error.contains("before the poll") && error.ends_with(CANONICAL_HINT),
                    "R12 A1 {check} must refuse {label} with the same reason: {error}"
                );
            }
        }

        // ---- R12 A6: the two compile-fail prefixes before the fence are accepted by both checks and
        // named in the accepted list. Neither shape can ship - nvcc refuses the try body with "device
        // code does not support exception handling" and the requires prefix with "identifier requires
        // is undefined" - so accepting them costs nothing.
        for (label, body) in [
            ("a try/catch prefix", r"    try { } catch (...) { }"),
            ("a requires prefix", r"    requires { };"),
        ] {
            let source = COMMON_HEADER.replace(
                HELPER_ANCHOR,
                &format!("{body}\n    __threadfence();\n    return observed;"),
            );
            assert_eq!(
                acquire_helper_fences_unconditionally(&source),
                Ok(()),
                "R12 A6 structural must accept {label}"
            );
            assert_eq!(
                acquire_fence_after_poll(&source, HELPER_POLL),
                Ok(()),
                "R12 A6 lint must accept {label}"
            );
        }

        // ---- R12 A2: the non-returning branch carries the hint too. The conditional shapes above
        // pin the four conditional branches; this is the fifth promised branch, and the two macro
        // branches are pinned with \`ends_with(CANONICAL_HINT)\` in
        // \`the_round_four_probes_are_handled_for_the_stated_reason\`.
        let aborting = COMMON_HEADER.replace(
            HELPER_ANCHOR,
            r"    abort();
    __threadfence();
    return observed;",
        );
        for (check, error) in [
            (
                "structural",
                acquire_helper_fences_unconditionally(&aborting)
                    .expect_err("R12 A2: a non-returning call is a blocker"),
            ),
            (
                "lint",
                acquire_fence_after_poll(&aborting, HELPER_POLL)
                    .expect_err("R12 A2: a non-returning call is a blocker"),
            ),
        ] {
            assert!(
                error.contains("does not return") && error.ends_with(CANONICAL_HINT),
                "R12 A2 {check}: the non-returning branch names the canonical form: {error}"
            );
        }

        // ---- R12 A5: the bare spelling. nvcc refuses it ("namespace-qualified name is required"),
        // so it cannot ship, and the message says which spelling it was instead of claiming that the
        // name still denotes the barrier.
        let bare_using = COMMON_HEADER.replace(
            HELPER_ANCHOR,
            r"    using __threadfence;
    __threadfence();
    return observed;",
        );
        let error = acquire_helper_fences_unconditionally(&bare_using)
            .expect_err("R12 A5: the bare spelling is not a valid using-declaration");
        assert!(
            error.contains("namespace-qualified") && error.ends_with(CANONICAL_HINT),
            "R12 A5: the bare spelling is named as such: {error}"
        );

        // ---- R12 A1's asymmetry, CLOSED in round 18 A8: a poll inside a bare block used to be
        // accepted by the lint, which read the block the poll stands in, and refused by the structural
        // check, which requires the poll to be a top-level statement of the helper's body. The two
        // checks now agree, with the structural check's own words, and this pins both verdicts.
        let poll_in_a_bare_block = COMMON_HEADER.replace(
            "    const uint32_t observed = spark_wait_for_flag(flag, deadline);",
            r"    { const uint32_t observed = spark_wait_for_flag(flag, deadline);
      __threadfence();
      return observed; }",
        );
        let error = acquire_helper_fences_unconditionally(&poll_in_a_bare_block)
            .expect_err("R12 A1: the structural check requires the fixed helper shape");
        assert!(
            error.contains("not a top-level statement") && error.ends_with(CANONICAL_HINT),
            "R12 A1: the stricter rule is named, with the hint: {error}"
        );
        let lint = acquire_fence_after_poll(&poll_in_a_bare_block, HELPER_POLL)
            .expect_err("R18 A8: the lint requires a top-level statement too");
        assert_eq!(
            lint, error,
            "R18 A8: the two checks answer the same text with the same words"
        );
    }

    /// Round 12, the structural fix's second half: the doc promises that EVERY rejection this module
    /// produces ends with the canonical hint, so the promise cannot quietly become false again the way
    /// it did for the lint's non-returning and macro branches. Each branch below is a different
    /// rejection: the lint's missing-poll, macro, non-returning, spelling, placement, conditional and
    /// pre-poll branches, and the structural check's shape rules, definition rules and translation-unit
    /// rules. A branch added without the hint fails here.
    #[test]
    fn every_rejection_carries_the_canonical_hint() {
        let poll_line = "    const uint32_t observed = spark_wait_for_flag(flag, deadline);";
        // Each case names the branch it has to reach as well as the hint, so a fixture that falls
        // into another branch fails instead of passing on the hint alone (round-13 D1).
        let poll = "spark_wait_for_flag(&flag, deadline);";
        let lint_cases: [(&str, String, &str); 13] = [
            (
                "no poll in the source",
                "uint32_t x = 1u;".to_string(),
                "no poll of",
            ),
            (
                "the poll used with a suspicious macro",
                format!("#define BAIL(x) return x\nBAIL({poll})"),
                "the poll is used with",
            ),
            (
                "a suspicious macro after the poll",
                format!("#define BAIL return\n{poll}\nBAIL;\n__threadfence();"),
                "does not certify that the fence runs after it",
            ),
            (
                "a non-returning call between the poll and the fence",
                format!("{poll}\nabort();\n__threadfence();"),
                "does not return and stands between the poll",
            ),
            (
                "a non-returning call before the poll",
                format!("uint32_t f() {{\n    abort();\n    {poll}\n    __threadfence();\n}}"),
                "abort does not return, so this scan does not certify that the acquire after it runs",
            ),
            (
                "a fence that is not a bare call statement",
                format!("{poll}\n(__threadfence(), 0u);"),
                "is not a bare call statement",
            ),
            (
                "a fence inside a larger expression",
                format!("{poll}\nready & (__threadfence)();"),
                "is part of a larger expression",
            ),
            (
                "no fence after the poll at all",
                poll.to_string(),
                "no reachable __threadfence()",
            ),
            (
                "a fence inside a nested body",
                format!("{poll}\nif (ready) {{ __threadfence(); }}"),
                "the __threadfence in spark_wait_for_flag_acquire sits at a nested brace depth",
            ),
            (
                "control before the poll",
                format!(
                    "uint32_t f() {{\n    if (ready) return 0u;\n    {poll}\n    __threadfence();\n}}"
                ),
                "stands before the poll",
            ),
            (
                "a macro use before the poll",
                format!(
                    "#define BAIL return\nuint32_t f() {{\n    BAIL;\n    {poll}\n    __threadfence();\n}}"
                ),
                "is a macro whose body contains a control keyword, a non-returning call, or another \
                 such macro, so this scan does not certify that the acquire after it runs",
            ),
            (
                "a fence inside a preprocessor conditional",
                format!("{poll}\n#if M4_FLAG\n__threadfence();\n#endif\n"),
                "the fence sits inside a preprocessor conditional",
            ),
            (
                "the fence before the poll (the same branch as no fence after it)",
                format!(
                    "uint32_t f() {{\n    __threadfence();\n    {poll}\n    return observed;\n}}"
                ),
                "no reachable __threadfence()",
            ),
        ];
        for (label, source, needle) in lint_cases {
            let error = acquire_fence_after_poll(&source, HELPER_POLL)
                .expect_err("every rejection must carry the hint");
            assert!(
                error.contains(needle) && error.ends_with(CANONICAL_HINT),
                "lint branch {label} must be the {needle:?} branch and end with the hint: {error}"
            );
        }

        let structural_cases: [(&str, String, &str); 6] = [
            (
                "a body with no poll",
                COMMON_HEADER.replace(poll_line, ""),
                "does not mention spark_wait_for_flag in its body",
            ),
            (
                "a body with two polls",
                COMMON_HEADER.replace(poll_line, &format!("{poll_line}\n{poll_line}")),
                "mentions the poll 2 times",
            ),
            (
                "the fence before the poll and none after it",
                COMMON_HEADER
                    .replace(HELPER_ANCHOR, "    return observed;")
                    .replace(poll_line, &format!("    __threadfence();\n{poll_line}")),
                "no reachable __threadfence() follows the poll of spark_wait_for_flag",
            ),
            (
                "no helper body in the header",
                "uint32_t nothing_here(void) { return 0u; }".to_string(),
                "must be defined as a function body",
            ),
            (
                "the fence inside a preprocessor conditional",
                COMMON_HEADER.replace(
                    HELPER_ANCHOR,
                    "#if M4_FLAG\n    __threadfence();\n#endif\n    return observed;",
                ),
                "the fence sits inside a preprocessor conditional",
            ),
            (
                "the POLL inside a preprocessor conditional",
                COMMON_HEADER.replace(poll_line, &format!("#if M4_FLAG\n{poll_line}\n#endif")),
                "the poll in spark_wait_for_flag_acquire sits inside a preprocessor conditional",
            ),
        ];
        for (label, source, needle) in structural_cases {
            let error = acquire_helper_fences_unconditionally(&source)
                .expect_err("every rejection must carry the hint");
            assert!(
                error.contains(needle) && error.ends_with(CANONICAL_HINT),
                "structural branch {label} must be the {needle:?} branch and end with the hint: {error}"
            );
        }
        // The three checks that read a whole set of sources, not one text.
        let call_site = observing_paths_call_the_acquiring_poll(&[(
            "k.cu",
            "spark_wait_for_flag(&flag, deadline);",
        )])
        .expect_err("a bare raw poll is rejected");
        assert!(
            call_site.ends_with(CANONICAL_HINT),
            "the call-site branch must end with the hint: {call_site}"
        );
        let redefined = guarded_names_are_not_redefined(&[("k.cu", "#define __threadfence()\n")])
            .expect_err("a #define of a guarded name is rejected");
        assert!(
            redefined.ends_with(CANONICAL_HINT),
            "the #define branch must end with the hint: {redefined}"
        );
        let declared = the_acquiring_poll_is_defined_exactly_once(&[(
            "other.cuh",
            "void spark_wait_for_flag_acquire(const uint32_t*, uint64_t);",
        )])
        .expect_err("a declaration outside the header is rejected");
        assert!(
            declared.ends_with(CANONICAL_HINT),
            "the declaration branch must end with the hint: {declared}"
        );
        let elsewhere = the_acquiring_poll_is_defined_exactly_once(&[(
            "other.cuh",
            "void spark_wait_for_flag_acquire(const uint32_t*, uint64_t) { }",
        )])
        .expect_err("a definition in the wrong file is rejected");
        assert!(
            elsewhere.ends_with(CANONICAL_HINT),
            "the wrong-file branch must end with the hint: {elsewhere}"
        );
        let none = the_acquiring_poll_is_defined_exactly_once(&[("k.cu", "uint32_t x = 1u;")])
            .expect_err("no definition at all is rejected");
        assert!(
            none.ends_with(CANONICAL_HINT),
            "the exactly-once branch must end with the hint: {none}"
        );
    }
    /// The argument text of a rejection site without its closure header: `||` for `ok_or_else` and
    /// `|_| ...` for `map_err`, trimmed at both ends. What is left has to be the constructor call and
    /// nothing else (round-15 A6).
    fn strip_closure_header(argument: &str) -> &str {
        let trimmed = argument.trim();
        if let Some(rest) = trimmed.strip_prefix("||") {
            return without_braces(rest.trim());
        }
        let after_pipe = trimmed
            .strip_prefix('|')
            .and_then(|rest| rest.find('|').map(|close| &rest[close + 1..]));
        if let Some(rest) = after_pipe {
            return without_braces(rest.trim());
        }
        without_braces(trimmed)
    }

    /// The text without one wrapping pair of braces, trimmed: the block body of a closure.
    fn without_braces(text: &str) -> &str {
        if text.starts_with('{') && text.ends_with('}') {
            return text[1..text.len() - 1].trim();
        }
        text
    }

    /// The index of the `)` that closes the `(` the caller has just stepped over: `open` is the
    /// byte AFTER that bracket, so the count starts at one. Counting only real brackets, because a
    /// bracket inside a string literal is text and a backslash-escaped quote does not end the literal.
    ///
    /// Round-15 A-5: starting the count at zero made every site whose argument does not itself open a
    /// bracket (`Err(variable)`, `Err("message".to_string())`) look unbalanced, and the scan
    /// panicked instead of reporting the routing diagnosis it exists to report.
    fn matching_paren(source: &str, open: usize) -> Option<usize> {
        let mut depth = 1usize;
        let mut in_string = false;
        let mut escaped = false;
        for (index, byte) in source.as_bytes().iter().enumerate().skip(open) {
            if in_string {
                if escaped {
                    escaped = false;
                } else if *byte == 0x5cu8 {
                    escaped = true;
                } else if *byte == b'"' {
                    in_string = false;
                }
                continue;
            }
            match byte {
                b'"' => in_string = true,
                b'(' => depth += 1,
                b')' => {
                    depth = depth.checked_sub(1)?;
                    if depth == 0 {
                        return Some(index);
                    }
                }
                _ => {}
            }
        }
        None
    }

    /// Round-15 A-1: the declaration rule has to read every return type a declaration can have, not
    /// only the ones spelled TYPE name(...). A return type that ends in '>' (a class template) or in
    /// ')' (decltype) has no identifier in front of the name, and a name the source defines as a macro
    /// was excluded as a macro use - so those declarations were read as CALLS. The consequence was a
    /// fixed verdict flip: the same helper was refused when its local declaration used a typedef and
    /// accepted when it used the macro or template twin, and with decltype both checks certified a
    /// helper that never observes the flag, for a cubin that is not the baseline (measured in the
    /// round-15 evidence).
    ///
    /// Each shape is tested in BOTH directions, because either direction alone can be satisfied by a
    /// broken rule: beside the canonical helper the declaration must not be mistaken for a poll, and
    /// in the helper's place the helper observes nothing, so both checks must refuse the mention.
    ///
    /// Round-15 A5 corrected what this test says about the second direction. `TYPE
    /// spark_wait_for_flag(flag, deadline);` spells the canonical tokens and nvcc reads it as a
    /// VARIABLE declaration with two initializer values ("too many initializer values"), so it is not
    /// a translation unit and no cubin is claimed for it; a block-scope function declaration is
    /// accepted by nvcc, so the twin that spells the parameters as types IS a translation unit, and the
    /// round-15 evidence compiles it to 70600 bytes / 3 PTB membar lines / sha 1f2503fd7ecaf189 against
    /// the 70432 / 4 / 605321041ac5ec1e baseline for THREE of the four return types: the typedef, the
    /// `decltype` and the source-macro declarations. The class-template twin is not a translation unit -
    /// nvcc reports the non-template declaration as incompatible with the poll's own - so that one is
    /// pinned as tokens only, and the function-template form beside the helper is what is measured for
    /// the class-template shape. Both texts are pinned, and labelled with which of the two nvcc accepts.
    #[test]
    fn a_declaration_is_a_declaration_whatever_its_return_type() {
        let poll_line = "    const uint32_t observed = spark_wait_for_flag(flag, deadline);";
        let shapes = [
            ("a typedef return type", String::new(), "uint32_t"),
            (
                "a class-template return type",
                String::from("template <typename T> struct wrapper { };\n"),
                "wrapper<uint32_t>",
            ),
            ("a decltype return type", String::new(), "decltype(0u)"),
            (
                "a return type the source defines as a macro",
                String::from("#define SPARK_M4_RETURN uint32_t\n"),
                "SPARK_M4_RETURN",
            ),
        ];
        for (label, prefix, return_type) in shapes {
            let prototype =
                format!("{return_type} spark_wait_for_flag(const volatile uint32_t *, uint64_t);");
            let beside = format!("#include <stdint.h>\n{prefix}{prototype}\n{COMMON_HEADER}");
            assert!(
                acquire_helper_fences_unconditionally(&beside).is_ok(),
                "the helper's own shape is unchanged by {label}"
            );
            assert_eq!(
                acquire_fence_after_poll(&beside, HELPER_POLL),
                Ok(()),
                "the lint follows the helper's own poll past {label}"
            );
            // The call-site check reads a KERNEL translation unit, so this text is the
            // declaration plus an observing path that calls the acquiring poll - not the header,
            // whose helper body contains the raw poll on purpose.
            let kernel = format!(
                "#include <stdint.h>\n{prefix}{prototype}\n__global__ void k(const volatile                  uint32_t *flag, uint64_t deadline) {{ {ACQUIRE_HELPER}(flag, deadline); }}\n"
            );
            assert!(
                observing_paths_call_the_acquiring_poll(&[("k.cu", kernel.as_str())]).is_ok(),
                "a declaration in a kernel is a declaration, not a bare poll call ({label})"
            );
            // (ii) In the helper's place, two texts. The first is the one the review named: the
            // canonical tokens standing as a declaration. nvcc reads it as a variable declaration with
            // two initializer values, so it is NOT a translation unit and no cubin is quoted for it -
            // and the scan still has to refuse it, because its tokens are the canonical call's.
            let canonical_tokens_as_a_declaration =
                format!("#include <stdint.h>\n{prefix}{COMMON_HEADER}")
                    .replace(
                        poll_line,
                        &format!("    {return_type} spark_wait_for_flag(flag, deadline);"),
                    )
                    .replace("    return observed;", "    return 0u;");
            // The compilable twin: a block-scope function declaration, which nvcc accepts, and the
            // exact text the round-15 evidence measures at 70600 / 3 / 1f2503fd7ecaf189.
            let type_parameter_twin = format!("#include <stdint.h>\n{prefix}{COMMON_HEADER}")
                .replace(
                    poll_line,
                    &format!(
                        "    {return_type} spark_wait_for_flag(const volatile uint32_t *, uint64_t);"
                    ),
                )
                .replace("    return observed;", "    return 0u;");
            for (form, instead) in [
                (
                    "the canonical tokens as a declaration",
                    &canonical_tokens_as_a_declaration,
                ),
                ("the type-parameter twin", &type_parameter_twin),
            ] {
                let structural = acquire_helper_fences_unconditionally(instead).expect_err(
                    "a helper whose only mention of the poll is a declaration observes nothing",
                );
                assert!(
                    structural.contains(CANONICAL_POLL_CALL)
                        && structural.ends_with(CANONICAL_HINT),
                    "the structural check refuses {label} as {form}, naming the canonical form: \
                     {structural}"
                );
                let lint = acquire_fence_after_poll(instead, HELPER_POLL).expect_err(
                    "the lint follows the helper's own body, where the poll is only declared",
                );
                assert!(
                    lint.contains(CANONICAL_POLL_CALL) && lint.ends_with(CANONICAL_HINT),
                    "the lint refuses {label} as {form}, naming the canonical form: {lint}"
                );
            }
        }
        // The class-template return type as nvcc actually accepts it beside the poll's own
        // definition: a function TEMPLATE, whose return type ends in '>' and which therefore does not
        // collide with the poll. The checks read the declaration shape and accept it, and the
        // round-15 evidence compiles this exact text to the baseline cubin. (The non-template form
        // the shapes loop uses is the same shape to the checks but not a translation unit nvcc
        // accepts: it redeclares the poll with a different return type, which is not an overload.)
        let function_template = format!(
            "#include <stdint.h>\ntemplate <typename T> struct wrapper {{ }};\ntemplate <typename              T> wrapper<T> spark_wait_for_flag(const volatile uint32_t *, uint64_t);\n{COMMON_HEADER}"
        );
        assert!(
            acquire_helper_fences_unconditionally(&function_template).is_ok(),
            "a function-template declaration of the poll does not change the helper's shape"
        );
        assert_eq!(
            acquire_fence_after_poll(&function_template, HELPER_POLL),
            Ok(()),
            "the lint follows the helper's own poll past a function-template declaration"
        );
        let template_kernel = format!(
            "#include <stdint.h>\ntemplate <typename T> struct wrapper {{ }};\ntemplate <typename              T> wrapper<T> spark_wait_for_flag(const volatile uint32_t *, uint64_t);\n__global__ void              k(const volatile uint32_t *flag, uint64_t deadline) {{ {ACQUIRE_HELPER}(flag, deadline); }}\n"
        );
        assert!(
            observing_paths_call_the_acquiring_poll(&[("k.cu", template_kernel.as_str())]).is_ok(),
            "a function-template declaration in a kernel is a declaration, not a bare poll call"
        );
        // The compilable twin of that second form: a class template whose two-argument constructor
        // makes `wrapper<uint32_t> spark_wait_for_flag(flag, deadline);` a local VARIABLE declaration,
        // and so a translation unit nvcc compiles. This is one of the texts the round-15 evidence
        // measures a cubin for, and both checks must still refuse it: the helper observes nothing.
        let wrapper_twin = format!(
            "#include <stdint.h>\ntemplate <typename T> struct wrapper {{ __device__ wrapper(const \
             volatile uint32_t *, uint64_t) {{}} }};\n{COMMON_HEADER}"
        )
        .replace(
            poll_line,
            "    wrapper<uint32_t> spark_wait_for_flag(flag, deadline);",
        )
        .replace("    return observed;", "    return 0u;");
        assert!(
            acquire_helper_fences_unconditionally(&wrapper_twin).is_err(),
            "the structural check refuses a helper that only declares a variable named like the poll"
        );
        assert!(
            acquire_fence_after_poll(&wrapper_twin, HELPER_POLL).is_err(),
            "the lint refuses a helper whose only mention of the poll declares a local variable"
        );
        // The two shapes the new type positions must NOT swallow, or the rule would be a hole of its
        // own: a comparison's '>' does not close a template argument list and an arrow's '>' is not
        // one either, so both calls stay calls. The lint's verdict pins it - a call read as a
        // declaration leaves the source with no poll at all, which the lint refuses.
        for (label, source) in [
            (
                "a comparison",
                "uint32_t f() {\n    bool later = deadline > spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return later ? 1u : 0u;\n}",
            ),
            (
                "a call through an arrow",
                "uint32_t f(struct holder *it) {\n    bool later = it->spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return later ? 1u : 0u;\n}",
            ),
        ] {
            assert_eq!(
                acquire_fence_after_poll(source, HELPER_POLL),
                Ok(()),
                "the lint still reads the poll as a call in {label}"
            );
        }
    }

    /// Round-15 A-3: the constructor's promise - exactly one hint, appended only when it is absent -
    /// is what makes every rejection honest, and no test called the constructor directly, so deleting
    /// its early return left the whole suite green and the debug assertion never ran. This test is
    /// what makes that early return load-bearing in both directions: a message WITHOUT the hint gets
    /// it once, a message WITH it is returned unchanged instead of doubled, and a message whose hint
    /// is not last is refused rather than shipped.
    #[test]
    fn the_constructor_appends_the_hint_exactly_once() {
        let plain = rejection("a rejection without the hint");
        assert_eq!(
            plain,
            format!("a rejection without the hint; {CANONICAL_HINT}"),
            "the constructor appends the hint to a message that does not carry one"
        );
        assert_eq!(
            plain.matches(CANONICAL_HINT).count(),
            1,
            "the hint appears exactly once: {plain}"
        );
        let hinted = format!("a rejection that already ends with the hint; {CANONICAL_HINT}");
        assert_eq!(
            rejection(hinted.clone()),
            hinted,
            "a message that already ends with the hint is returned unchanged, not doubled"
        );
        assert_eq!(
            rejection(hinted).matches(CANONICAL_HINT).count(),
            1,
            "the early return is what keeps the hint from being appended twice"
        );
        // A hint that is not last is a message bug, and in a debug build the constructor says so
        // instead of shipping a rejection whose hint is buried.
        let refused = std::panic::catch_unwind(|| {
            rejection(format!("{CANONICAL_HINT} and then some more message"))
        });
        assert!(
            refused.is_err(),
            "a message that carries the hint anywhere but at its end must be refused"
        );
    }

    /// Round 13 A-1 found a rejection built without the hint, and round 13 answered with a source
    /// scan that looked for the hint anywhere in an `Err(...)` span. Round-14 A-2 showed four
    /// physical mutations escaping that scan (a rejection built by `ok_or_else`, a message that
    /// carried the hint in one branch, a hint with more message after it, a hint buried in an
    /// if/else), so round 14 makes the invariant STRUCTURAL instead: every rejection goes through
    /// `rejection`, which appends the hint when it is absent, and a debug build refuses a message
    /// that carries the hint anywhere but at its end.
    ///
    /// This scan therefore checks the ROUTING rather than searching for the hint, which is a much
    /// smaller and more honest claim:
    ///
    ///   * every rejection-producing construct in the guard's code calls `rejection(`: `Err(`,
    ///     `ok_or(`, `ok_or_else(` and `map_err(` (round-15 A-5: `ok_or(` was missing from the
    ///     keyword set, so `ok_or(raw_message)` escaped);
    ///   * the hint is named nowhere in that code except its own definition and inside `rejection`,
    ///     so no message can smuggle a hint in, in any branch;
    ///   * `rejection` still appends the hint at the end.
    ///
    /// A keyword occurrence that is part of a LONGER identifier is not a site: a function named
    /// `my_ok_or_else` is a definition, not a rejection, and reading it as one made this scan panic
    /// on text that has no rejection in it at all (round-15 A-5).
    ///
    /// Out of scope, and said rather than claimed: a rejection built by code OUTSIDE this region (in
    /// another module of this file, or by a callee that returns a message built elsewhere) is not
    /// inspected, and neither is a message built somewhere else and passed in a VARIABLE
    /// (`let message = ...; Err(message)`). That form is refused rather than allowed, by policy: the
    /// message of a rejection is built at the rejection, through the constructor, so the argument of
    /// an `Err` here is either a call of `rejection` or it is reported.
    #[test]
    fn every_rejection_routes_through_the_one_constructor() {
        let source = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/src/m4_abi.rs"))
            .expect("this module's own source is readable");
        // Round-15 A-5: the region split is a text search, and this file is checked out on platforms
        // whose line endings are CRLF, so the newlines are normalised first.
        let source = source.replace("\r\n", "\n");
        // The guard's own code: from the hint constant that every rejection ends with, up to its
        // test module. Other modules in this file have their own errors and their own vocabulary.
        let code = source
            .split_once("const CANONICAL_HINT: &str")
            .expect("the guard defines the canonical hint")
            .1
            .split_once("\n#[cfg(test)]\nmod acquire_guard {")
            .expect("the guard's code comes before its test module")
            .0;
        assert!(
            code.contains("pub fn acquire_fence_after_poll") && code.contains("fn rejection("),
            "the scan reads the guard and not the tests"
        );
        // 1. Every rejection-producing construct routes through the constructor.
        let mut sites = 0usize;
        for keyword in ["Err(", "ok_or(", "ok_or_else(", "map_err("] {
            for (index, _) in code.match_indices(keyword) {
                // A keyword inside a longer identifier is that identifier, not this construct.
                let previous_byte = index.checked_sub(1).map(|before| code.as_bytes()[before]);
                if previous_byte.is_some_and(|byte| byte.is_ascii_alphanumeric() || byte == b'_') {
                    continue;
                }
                let line_start = code[..index].rfind('\n').map_or(0, |position| position + 1);
                if code[line_start..index].trim_start().starts_with("//") {
                    continue;
                }
                let Some(close) = matching_paren(code, index + keyword.len()) else {
                    panic!("the {keyword} at byte {index} is unbalanced, so this scan is broken");
                };
                sites += 1;
                // The ARGUMENT has to be the constructor call itself, not merely contain one: a
                // substring search let `Err(if c { rejection(x) } else { String::from(y) })` through
                // (round-15 A6), whose else arm never gets the hint. A closure header is allowed,
                // because `ok_or_else(|| ...)` and `map_err(|_| ...)` require one; anything else -
                // an if/else, a match, a variable, a second arm - is reported.
                let argument = strip_closure_header(&code[index + keyword.len()..close]);
                let routed = argument.starts_with("rejection(")
                    && matching_paren(argument, "rejection(".len())
                        .is_some_and(|end| end + 1 == argument.len());
                assert!(
                    routed,
                    "the {keyword} at byte {index} does not route through the constructor: {}",
                    &code[index..=close.min(index + 200)]
                );
            }
        }
        assert!(
            sites >= 20,
            "the scan must see the guard's rejection sites, saw {sites}"
        );
        // 2. Only the definition and the constructor may name the hint, so no message can carry one.
        let constructor_start = code.find("fn rejection(").expect("the constructor");
        let constructor_body = &code[constructor_start..];
        let constructor_end = constructor_start
            + constructor_body
                .find("\n}\n")
                .expect("the constructor's body ends");
        for (index, _) in code.match_indices("CANONICAL_HINT") {
            let line_start = code[..index].rfind('\n').map_or(0, |position| position + 1);
            let line_end = code[index..]
                .find('\n')
                .map_or(code.len(), |end| index + end);
            let line = &code[line_start..line_end];
            if line.trim_start().starts_with("//") || line.contains("const CANONICAL_HINT") {
                continue;
            }
            assert!(
                (constructor_start..constructor_end).contains(&index),
                "the hint is named outside its definition and the constructor: {line}"
            );
        }
        // 3. The constructor still appends the hint at the end.
        let constructor = &code[constructor_start..constructor_end];
        assert!(
            constructor.contains("format!(\"{message}; {CANONICAL_HINT}\")")
                && constructor.contains("message.ends_with(CANONICAL_HINT)"),
            "the constructor appends the hint at the end: {constructor}"
        );
    }

    /// Round 13, D3: the one-direction claim in the documentation - no shape passes the structural
    /// check and fails the lint - is only meaningful when BOTH sides of every documented asymmetry
    /// are asserted. Each pair below pins the structural verdict and the lint verdict for the same
    /// text, so the documentation is checked rather than believed.
    #[test]
    fn the_documented_asymmetry_is_what_the_tests_assert() {
        let poll_line = "    const uint32_t observed = spark_wait_for_flag(flag, deadline);";
        // Two polls: exactly one observation pairs with the acquire, and since round 14 the lint
        // refuses to guess which of the two the fence belongs to, so BOTH checks refuse the body and
        // this asymmetry is gone rather than documented.
        let two_polls = COMMON_HEADER.replace(poll_line, &format!("{poll_line}\n{poll_line}"));
        let structural = acquire_helper_fences_unconditionally(&two_polls)
            .expect_err("two polls are not the fixed helper shape");
        assert!(
            structural.contains("2 times") && structural.ends_with(CANONICAL_HINT),
            "the structural side names the count: {structural}"
        );
        let lint = acquire_fence_after_poll(&two_polls, HELPER_POLL)
            .expect_err("the lint cannot tell which observation the fence pairs with");
        assert!(
            lint.contains("cannot tell which observation") && lint.ends_with(CANONICAL_HINT),
            "the lint side names the ambiguity: {lint}"
        );
        // The poll inside a bare block: the structural check requires the poll to be a top-level
        // statement of the helper's body, the lint reads the poll's own block.
        let poll_in_a_bare_block =
            COMMON_HEADER.replace(poll_line, &format!("    {{\n    {poll_line}\n    }}"));
        let structural = acquire_helper_fences_unconditionally(&poll_in_a_bare_block)
            .expect_err("the poll is not a top-level statement");
        assert!(
            structural.contains("not a top-level statement")
                && structural.ends_with(CANONICAL_HINT),
            "the structural side names the shape: {structural}"
        );
        let lint = acquire_fence_after_poll(&poll_in_a_bare_block, HELPER_POLL)
            .expect_err("R18 A8: the lint requires a top-level statement too");
        assert_eq!(
            lint, structural,
            "R18 A8: the two checks answer the same text with the same words"
        );
        // The poll inside a preprocessor conditional: BOTH checks refuse the body, and with the SAME
        // sentence. Round-23 A3 corrected this. The lint used to read the text as "no compiled poll"
        // and accept any body that also held a compiled poll, while the structural check refused the
        // body, so `#if M4_FLAG` around a first poll with a second unconditional poll and the fence
        // compiled under -DM4_FLAG=1 to two real polls (39072 / 95e0cf41 (pre-change)) with one acquire between
        // them: a silent second observation. A poll the scan cannot place is refused in both checks now.
        let poll_in_a_conditional =
            COMMON_HEADER.replace(poll_line, &format!("#if M4_FLAG\n{poll_line}\n#endif"));
        let structural = acquire_helper_fences_unconditionally(&poll_in_a_conditional)
            .expect_err("the poll may not be compiled");
        assert!(
            structural.contains("the poll in spark_wait_for_flag_acquire sits inside")
                && structural.ends_with(CANONICAL_HINT),
            "the structural side names the conditional: {structural}"
        );
        let lint = acquire_fence_after_poll(&poll_in_a_conditional, HELPER_POLL)
            .expect_err("the lint refuses a poll it cannot place too");
        assert_eq!(
            lint, structural,
            "the two checks answer an unplaceable poll with the same sentence"
        );
        // Round-22 A1: a fence BEFORE the poll is not the acquire, and the shape the round-21 review
        // found has both fences. With the canonical fence still in place BOTH checks ACCEPT it, because
        // the structural check now takes the first fence AFTER the poll exactly as the lint does. The
        // reviewer measured the shape: the acquire after the poll is present and the extra fence before
        // it is the only delta, 70448 bytes / 5 PTB membar lines / sha 488031bf, against the 70432 / 4 /
        // 605321041ac5ec1e baseline - so the lint's acceptance was right and this check's refusal was wrong.
        let fence_before_and_after =
            COMMON_HEADER.replace(poll_line, &format!("    __threadfence();\n{poll_line}"));
        let structural = acquire_helper_fences_unconditionally(&fence_before_and_after);
        let lint = acquire_fence_after_poll(&fence_before_and_after, HELPER_POLL);
        assert!(
            structural.is_ok(),
            "the acquire after the poll is there: {structural:?}"
        );
        assert_eq!(
            structural, lint,
            "both checks accept a fence before the poll when the acquire after it is still there"
        );
        // The same insertion with the acquire REMOVED is still refused, and for the true reason: there
        // is no fence after the poll at all. A fence before the poll is not the acquire, so the
        // structural check reports the absence rather than saying the extra fence "precedes the poll".
        let fence_before_the_poll = COMMON_HEADER
            .replace(HELPER_ANCHOR, "    return observed;")
            .replace(poll_line, &format!("    __threadfence();\n{poll_line}"));
        let structural = acquire_helper_fences_unconditionally(&fence_before_the_poll)
            .expect_err("the only fence precedes the poll, so there is no acquire after it");
        assert!(
            structural
                .contains("no reachable __threadfence() follows the poll of spark_wait_for_flag")
                && structural.ends_with(CANONICAL_HINT),
            "the structural side says there is no acquire after the poll: {structural}"
        );
        let lint = acquire_fence_after_poll(&fence_before_the_poll, HELPER_POLL)
            .expect_err("no fence follows the poll");
        assert!(
            lint.contains("no reachable __threadfence()") && lint.ends_with(CANONICAL_HINT),
            "the lint side says no fence follows the poll: {lint}"
        );
        // Round-14 A-1: the poll this lint follows is the HELPER's own, so plumbing that names the
        // raw poll cannot hijack it. nvcc compiles each of these three headers to the baseline cubin
        // (70432 bytes / 4 PTB membar lines / sha 605321041ac5ec1e, measured in the CUDA 13.0.3
        // container), which is why reading them as a poll was a false rejection and not a
        // conservative one.
        // Each fixture carries the include its own types need, because it stands in front of the
        // header's own includes: the text as written here is the text nvcc compiles in the
        // round-14/15 evidence, and a fixture that does not compile has no measured cubin to claim
        // (round-15 A-2).
        let forward_declaration = format!(
            "#include <stdint.h>\nuint32_t spark_wait_for_flag(const volatile uint32_t *, \
             uint64_t);\n{COMMON_HEADER}"
        );
        let earlier_polling_function = format!(
            "#include <stdint.h>\nuint32_t spark_wait_for_flag(const volatile uint32_t *, \
             uint64_t);\nuint32_t earlier_observer(const uint32_t *flag, uint64_t deadline) {{\n    \
             return spark_wait_for_flag(flag, deadline);\n}}\n{COMMON_HEADER}"
        );
        let pointer_parameter = format!(
            "#include <stdint.h>\nuint32_t through_a_pointer(uint32_t (*spark_wait_for_flag)(const \
             volatile uint32_t *, uint64_t)) {{\n    return spark_wait_for_flag(0, 0);\n}}\n\
             {COMMON_HEADER}"
        );
        for (label, source) in [
            ("a forward declaration of the raw poll", forward_declaration),
            ("an earlier function that polls", earlier_polling_function),
            (
                "a function-pointer parameter named like the poll",
                pointer_parameter,
            ),
        ] {
            assert!(
                acquire_helper_fences_unconditionally(&source).is_ok(),
                "the helper's own shape is unchanged by {label}"
            );
            assert_eq!(
                acquire_fence_after_poll(&source, HELPER_POLL),
                Ok(()),
                "the lint anchors on the helper's own poll, not on {label}"
            );
        }
        // Round-14 A-1: a rejection must be true about its input. This one names the line of the
        // statement standing before the poll, inside the helper's own body - the buggy lint named
        // line 146, inside the raw poll's DEFINITION, and called that "before the poll".
        let guarded_body = COMMON_HEADER.replace(
            poll_line,
            &format!("    if (ready) return 0u;\n{poll_line}"),
        );
        let poll_line_number = line_number_of(&guarded_body, poll_line);
        let helper_line_number = line_number_of(&guarded_body, &format!("{ACQUIRE_HELPER}("));
        let error = acquire_fence_after_poll(&guarded_body, HELPER_POLL)
            .expect_err("a control statement before the poll is refused");
        let named: usize = error
            .strip_prefix("line ")
            .and_then(|rest| rest.split(':').next())
            .and_then(|number| number.parse().ok())
            .unwrap_or_else(|| panic!("the rejection names a line: {error}"));
        assert!(
            named > helper_line_number && named < poll_line_number,
            "line {named} is inside the helper's body (from line {helper_line_number}) and before its \
             poll (line {poll_line_number}): {error}"
        );
        assert!(
            guarded_body
                .lines()
                .nth(named - 1)
                .is_some_and(|text| text.contains("if (ready)")),
            "the named line really holds the statement the message quotes: {error}"
        );
        // Round-20 A1: a use of a macro that can BOUND the path before the poll is refused by BOTH checks
        // now, in the same words. The round-19 review found the lint refusing it while the structural
        // check accepted it, which was not a documented asymmetry but a wrong verdict: nvcc compiles this
        // text to 68576 / 2 / bfd9e8e177f4f76d rather than to the 70432 / 4 / 605321041ac5ec1e baseline (measured; the
        // stored text refused_suspicious_macro_before_poll is this text).
        let bail_before_the_poll = format!("#define BAIL return\n{COMMON_HEADER}").replace(
            SHIPPED_HELPER_BODY,
            "    BAIL;\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
        );
        let structural = acquire_helper_fences_unconditionally(&bail_before_the_poll)
            .expect_err("a use of a macro that can return before the poll is refused");
        assert_eq!(
            acquire_fence_after_poll(&bail_before_the_poll, HELPER_POLL),
            Err(structural.clone()),
            "R20 A1: both checks refuse it in the same words"
        );
        assert!(
            structural.contains("is a macro whose body contains a control keyword")
                && structural.ends_with(CANONICAL_HINT),
            "the shared reason names the macro rule: {structural}"
        );
        // The other side of the same rule: a macro whose body carries only an OPERATOR is a blocker under
        // the completed-statement scoping of the literal operator, so this text - whose macro use stands
        // in an earlier statement that ended at its `;` - is ACCEPTED by both. nvcc compiles it to the
        // baseline cubin (measured; the stored text is macro_operator_before_the_poll).
        let operator_macro_in_an_earlier_statement =
            format!("#define PRE (deadline > 0u) && true\n{COMMON_HEADER}").replace(
                SHIPPED_HELPER_BODY,
                "    const bool ran = PRE;\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
            );
        assert_eq!(
            acquire_helper_fences_unconditionally(&operator_macro_in_an_earlier_statement),
            Ok(()),
            "an operator-only macro in a completed statement is not a blocker (70432 / 4 / 605321041ac5ec1e)"
        );
        assert_eq!(
            acquire_fence_after_poll(&operator_macro_in_an_earlier_statement, HELPER_POLL),
            Ok(()),
            "the lint agrees (70432 / 4 / 605321041ac5ec1e)"
        );
        // And the refusal side of the macro rule in the poll's OWN statement: the poll passed as the
        // argument of a macro that can return. Both checks refuse it - the lint names the use with the
        // poll, the structural check names the use - and each says why in the same words about the body.
        let poll_inside_a_returning_macro = format!("#define BAIL(x) return x\n{COMMON_HEADER}").replace(
            SHIPPED_HELPER_BODY,
            "    BAIL((spark_wait_for_flag(flag, deadline)));\n    __threadfence();\n    return 0u;\n",
        );
        for (check, error) in [
            (
                "structural",
                acquire_helper_fences_unconditionally(&poll_inside_a_returning_macro).expect_err(
                    "a poll expanded inside a returning macro is not an observing path",
                ),
            ),
            (
                "lint",
                acquire_fence_after_poll(&poll_inside_a_returning_macro, HELPER_POLL)
                    .expect_err("the lint refuses the same text"),
            ),
        ] {
            assert!(
                error.contains("a macro whose body contains a control keyword")
                    && error.ends_with(CANONICAL_HINT),
                "the {check} check refuses a poll inside a returning macro: {error}"
            );
        }
        // The same rule on the OTHER side of the poll, in the poll's own statement, where no control
        // token of the structural check's own list stands: `+` is not one, so only the macro rule can
        // refuse this text, and the lint refuses it with its poll-statement rule. The shape is written as
        // a shape rather than stored and measured, so no cubin is claimed for it.
        let macro_after_the_poll_in_the_same_statement =
            format!("#define BAIL (abort(), 0u)\n{COMMON_HEADER}").replace(
                SHIPPED_HELPER_BODY,
                "    const uint32_t observed = spark_wait_for_flag(flag, deadline) + BAIL;\n    __threadfence();\n    return observed;\n",
            );
        for (check, error) in [
            (
                "structural",
                acquire_helper_fences_unconditionally(&macro_after_the_poll_in_the_same_statement)
                    .expect_err("a non-returning macro in the poll's own statement is refused"),
            ),
            (
                "lint",
                acquire_fence_after_poll(&macro_after_the_poll_in_the_same_statement, HELPER_POLL)
                    .expect_err("the lint refuses the use in the poll's own statement"),
            ),
        ] {
            assert!(
                error.contains("a macro whose body contains a control keyword")
                    && error.ends_with(CANONICAL_HINT),
                "the {check} check refuses the macro in the poll's own statement: {error}"
            );
        }
        // Round-21 A1: the SAME macro AFTER the poll, in a later, completed statement. The round-20 rule
        // scoped suspicion to "before the poll, or in the poll's own statement", and this check accepted
        // the text while the lint refused it - the mirror image of round 20, and again the lint was
        // right: nvcc compiles the text to 69072 / 2 / 18f93bdc42897348 rather than to the 70432 / 4 / 605321041ac5ec1e
        // baseline, because the fence after the macro is unreachable (the stored text is
        // refused_suspicious_macro_after_the_poll). The scope is now the lint's own - every use before
        // the FENCE - and the shared reason is the lint's forward-walk sentence.
        let bail_after_the_poll = format!("#define BAIL return\n{COMMON_HEADER}").replace(
            SHIPPED_HELPER_BODY,
            "    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    BAIL;\n    __threadfence();\n    return observed;\n",
        );
        let structural = acquire_helper_fences_unconditionally(&bail_after_the_poll)
            .expect_err("a use of a macro that can return after the poll is refused");
        assert_eq!(
            acquire_fence_after_poll(&bail_after_the_poll, HELPER_POLL),
            Err(structural.clone()),
            "R21 A1: both checks refuse it in the same words"
        );
        assert!(
            structural.contains("is a macro whose body contains a control keyword")
                && structural.contains("the fence runs after it")
                && structural.ends_with(CANONICAL_HINT),
            "the shared reason is the lint's forward-walk sentence, not the pre-poll one: {structural}"
        );
        // A use AFTER the fence is read by neither check and is NOT refused: the acquire has already run
        // by then, so a macro that can leave the helper afterwards cannot skip it.
        let bail_after_the_fence = format!("#define BAIL return\n{COMMON_HEADER}").replace(
            SHIPPED_HELPER_BODY,
            "    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    BAIL;\n    return observed;\n",
        );
        assert_eq!(
            acquire_helper_fences_unconditionally(&bail_after_the_fence),
            Ok(()),
            "a use after the fence cannot skip the acquire"
        );
        assert_eq!(
            acquire_fence_after_poll(&bail_after_the_fence, HELPER_POLL),
            Ok(()),
            "the lint agrees, and that is the scope the two checks share"
        );
    }

    /// Round-21 finding 2: the sentence that said the two checks "differ in one direction only, because
    /// the structural check is STRICTER" had to be re-derived from shapes instead of kept as a claim.
    /// These are the fence-placement shapes where the structural check is the one that could be
    /// stricter - a fence the compiler may not reach, and a fence in a body that is not the helper's
    /// straight-line path - and the two checks give the SAME VERDICT on both: both refuse, each naming
    /// the shape it read. A divergence here would be the structural check refusing alone, so this is the
    /// direction that sentence claims, written as a shape rather than as prose.
    #[test]
    fn the_two_checks_agree_about_the_fence_shapes_the_structural_check_reads() {
        let poll_line = "    const uint32_t observed = spark_wait_for_flag(flag, deadline);";
        let conditional_fence = format!(
            "{poll_line}\n#if M4_FLAG\n    __threadfence();\n#endif\n    return observed;\n"
        );
        let fence_in_a_try_body = format!(
            "{poll_line}\n    try {{\n        __threadfence();\n    }} catch (...) {{\n    }}\n    return observed;\n"
        );
        // The same two bodies with an attribute between the head and the brace. NEITHER spelling is
        // valid C++ - the grammar has no attribute between `try` and its compound statement, nor
        // between a catch clause's `)` and its compound statement, and both g++ 16 and clang 22 reject
        // `try [[likely]] {` and `catch (...) [[likely]] {` at parse in c++17, c++20, c++23 and c++2c
        // ("expected `{` before `[`") - so no compiler reaches the "device code does not support
        // exception handling" refusal nvcc gives the PLAIN spellings. The guard reads tokens rather
        // than compiling, so the question here is still the one it answers for the plain spelling:
        // these must get the same verdict (round-36 A1).
        let attributed_try_body = format!(
            "{poll_line}\n    try [[likely]] {{\n        __threadfence();\n    }} catch (...) {{\n    }}\n    return observed;\n"
        );
        let attributed_catch_body = format!(
            "{poll_line}\n    try {{\n    }} catch (...) [[likely]] {{\n        __threadfence();\n    }}\n    return observed;\n"
        );
        // An UNCALLED lambda with an attribute between the capture list and the brace. The walk-back
        // must step over the attribute group and keep looking for the capture list; the variant that
        // stops at the attribute, which is the mutation of the step-over's `continue`, reads the
        // closure body as the helper's own bare block and certifies a fence that never runs. Both
        // spellings are legitimate C++: g++ accepts `[] [[maybe_unused]] { ... }` at c++17 and c++20,
        // and clang accepts it with a C++23-extension warning.
        let uncalled_attributed_lambda = format!(
            "{poll_line}\n    auto l = [] [[maybe_unused]] {{ __threadfence(); }};\n    (void)l;\n    return observed;\n"
        );
        for (label, body) in [
            (
                "a fence inside a preprocessor conditional the guard cannot resolve",
                conditional_fence,
            ),
            ("a fence inside a try body", fence_in_a_try_body),
            (
                "a fence inside a try body carrying an attribute",
                attributed_try_body,
            ),
            (
                "a fence inside a catch body carrying an attribute",
                attributed_catch_body,
            ),
            (
                "a fence inside an uncalled lambda with an attribute before its brace",
                uncalled_attributed_lambda,
            ),
        ] {
            let source = COMMON_HEADER.replace(SHIPPED_HELPER_BODY, &body);
            let structural = acquire_helper_fences_unconditionally(&source);
            let lint = acquire_fence_after_poll(&source, HELPER_POLL);
            assert_eq!(
                structural.is_ok(),
                lint.is_ok(),
                "the two checks disagree about {label}: structural={structural:?} lint={lint:?}"
            );
            assert!(
                structural.is_err(),
                "the shape is one the structural check reads as not straight-line: {label}"
            );
            // Round-24: the doc calls these two an EQUALITY, so the whole message is asserted, not just
            // the verdict. The two checks give the same reason for both shapes, in the same words.
            assert_eq!(
                structural, lint,
                "the two checks answer {label} with the same words"
            );
        }
    }

    /// The macro-use shapes of the round-21 review's disagreement matrix, and the one thing this test
    /// asserts about all of them: the two checks give the same VERDICT. The round-20 and round-21 fixes
    /// removed the two disagreements those reviews found; this table is what keeps a third one from
    /// appearing, because a shape the lint refuses and the structural check accepts (or the other way
    /// round) is a wrong verdict on one side rather than a documented asymmetry. Every shape is written
    /// as a shape rather than stored and measured, so no cubin is claimed for it beyond the two the
    /// stored texts already carry - and every row IS a translation unit, which is evidence rather than a
    /// reading: with SPARKFLASH_MATRIX_DUMP set the loop writes the very texts it runs, and all
    /// eighty-one compile in the CUDA 13.0.3 container (the rows whose macro called abort() call the
    /// device-callable `__trap()`, because nvcc compiles abort() and `__builtin_trap()` as calls to a
    /// `__host__` function from a device body). What that compile proves is PARSEABILITY and nothing
    /// more: a row is a translation unit with no entry point, so its cubin is empty, and the cubins this
    /// file quotes belong to the measured table, which compiles through the same body substitution
    /// against the committed baseline. No row here claims a cubin.
    ///
    /// The shapes, in the order the round-21 review named them:
    /// - a use before the poll, in an earlier, completed statement;
    /// - a use before the poll, in the poll's own statement;
    /// - a use after the poll, in the same statement;
    /// - a use after the poll, in a later, completed statement;
    /// - a use after the fence, which neither check reads;
    /// - a use inside a nested block before the poll;
    /// - a use in the fence's own statement;
    /// - a macro defined after its use;
    /// - a macro redefined benign then control, and control then benign;
    /// - a function-like macro with the designator as a parameter;
    /// - a nested macro;
    /// - a macro whose body is a non-returning call.
    ///
    /// The third field says whether the row claims a REFUSAL by the macro rule; a row that says false
    /// claims ACCEPTANCE by both checks and is asserted to be accepted by both (round 26), because
    /// comparing two refusals was passing rows that claimed the opposite verdict.
    ///
    /// The two counts the module documentation states are DERIVED here rather than trusted: the source
    /// is read back and the rows counted, so a stale number cannot survive a round (round 27 found
    /// "21 rows" and "Seventeen rows in all" in the documentation of a 27-row and a 40-row table).
    #[test]
    fn the_documentation_counts_are_derived_from_the_tables() {
        let source = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/m4_abi.rs"));
        let positions = source
            .split_once(concat!(
                "let positions: Vec<(&str, String, bool)> = ",
                "vec!["
            ))
            .expect("the positions table")
            .1;
        let positions = positions
            .split_once(concat!("for (label, source, must_refuse) in ", "positions"))
            .expect("the positions loop")
            .0;
        let matrix = source
            .split_once(concat!("let shapes: Vec<(&str, String, bool)> = ", "vec!["))
            .expect("the matrix table")
            .1;
        let matrix = matrix
            .split_once(concat!(
                "for (index, (label, source, refuses_with_the_macro_rule)) in ",
                "shapes"
            ))
            .expect("the matrix loop")
            .0;
        assert_eq!(
            positions.matches("grid_source(").count(),
            40,
            "the positions table has the forty rows the documentation states"
        );
        assert_eq!(
            matrix.matches("with_define(").count(),
            81,
            "the matrix has the eighty-one rows the documentation states"
        );
        assert!(
            source.contains(concat!("Forty rows in ", "all")),
            "the documented positions count"
        );
        assert!(
            source.contains(concat!("the 81-row ", "matrix")),
            "the documented matrix count"
        );
    }

    #[test]
    fn the_two_checks_agree_on_every_shape_of_macro_use() {
        let poll_line = "    const uint32_t observed = spark_wait_for_flag(flag, deadline);";
        let observed = "    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n";
        let before_the_poll =
            format!("    BAIL;\n{poll_line}\n    __threadfence();\n    return observed;\n");
        let after_the_poll =
            format!("{poll_line}\n    BAIL;\n    __threadfence();\n    return observed;\n");
        let with_define = |define: &str, body: &str| {
            format!("{define}\n{COMMON_HEADER}").replace(SHIPPED_HELPER_BODY, body)
        };
        let shapes: Vec<(&str, String, bool)> = vec![
            (
                "a use before the poll, in an earlier, completed statement",
                with_define("#define BAIL return", &before_the_poll),
                true,
            ),
            (
                "a use before the poll, in the poll's own statement",
                with_define(
                    "#define BAIL return",
                    "    BAIL spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return 0u;\n",
                ),
                true,
            ),
            (
                "a use after the poll, in the same statement",
                with_define(
                    "#define BAIL (__trap(), 0u)",
                    "    const uint32_t observed = spark_wait_for_flag(flag, deadline) + BAIL;\n    __threadfence();\n    return observed;\n",
                ),
                true,
            ),
            (
                "a use after the poll, in a later, completed statement",
                with_define("#define BAIL return", &after_the_poll),
                true,
            ),
            (
                "a use after the fence",
                with_define(
                    "#define BAIL return",
                    &format!(
                        "{poll_line}\n    __threadfence();\n    BAIL;\n    return observed;\n"
                    ),
                ),
                false,
            ),
            (
                "a use inside a nested block before the poll",
                with_define(
                    "#define BAIL return",
                    &format!(
                        "    {{\n        BAIL;\n    }}\n{poll_line}\n    __threadfence();\n    return observed;\n"
                    ),
                ),
                true,
            ),
            (
                "a use in the fence's own statement",
                with_define(
                    "#define BAIL (__trap(), 0u)",
                    &format!("{poll_line}\n    BAIL, __threadfence();\n    return observed;\n"),
                ),
                true,
            ),
            (
                "a macro defined after its use",
                // The use is a translation unit: the header defines BAIL benignly before it, and the
                // suspicious redefinition comes after the function, so the scan at the use reads the
                // benign definition while the compiler still sees a valid call.
                format!(
                    "{}\n#undef BAIL\n#define BAIL return\n",
                    with_define("#define BAIL 0u", &after_the_poll)
                ),
                false,
            ),
            (
                "a macro redefined benign then control",
                with_define(
                    "#define BAIL 0u\n#undef BAIL\n#define BAIL return",
                    &after_the_poll,
                ),
                true,
            ),
            (
                "a macro redefined control then benign",
                with_define(
                    "#define BAIL return\n#undef BAIL\n#define BAIL 0u",
                    &after_the_poll,
                ),
                false,
            ),
            (
                "a function-like macro with the designator as a parameter",
                with_define(
                    "#define P(spark_wait_for_flag) spark_wait_for_flag",
                    "    const uint32_t observed = P(spark_wait_for_flag)(flag, deadline);\n    __threadfence();\n    return observed;\n",
                ),
                true,
            ),
            (
                "a nested macro",
                with_define(
                    "#define INNER return 0u\n#define OUTER INNER",
                    &format!(
                        "{poll_line}\n    OUTER;\n    __threadfence();\n    return observed;\n"
                    ),
                ),
                true,
            ),
            (
                "a macro whose body is a non-returning call",
                // __trap() rather than abort() or __builtin_trap(): nvcc compiles abort() and
                // __builtin_trap() as calls to a __host__ function from the device body, so neither row
                // would be a translation unit (round-22 finding 3), while __trap() is the
                // device-callable never-returning call the guard's own list names.
                with_define("#define BAIL __trap()", &after_the_poll),
                true,
            ),
            (
                "a fence before the poll with the acquire after it",
                with_define(
                    "",
                    &format!(
                        "    __threadfence();\n{poll_line}\n    __threadfence();\n    return observed;\n"
                    ),
                ),
                false,
            ),
            (
                "a whole call whose designator is parenthesised",
                with_define(
                    "#define P (spark_wait_for_flag)(flag, deadline)",
                    "    const uint32_t observed = P;\n    __threadfence();\n    return observed;\n",
                ),
                false,
            ),
            ("the shipped body itself", with_define("", observed), false),
            // Round-25 A1: a MACRO whose body returns, used INSIDE the poll's own argument list. The
            // structural check read it and the lint did not, so the two answered one compiled body
            // differently: the literal `({ return 0u; flag; })` in the argument is 70464 / 2 /
            // 2b56c510, while THIS form - `#define BAIL return` inside `({ BAIL; flag; })` - is
            // 68576 / 2 / bfd9e8e177f4f76d, the cubin of a `return` before the poll that both checks refuse
            // for the same reason. Round 26 corrected this comment, which had swapped the two.
            (
                "a macro that returns inside the poll's first argument",
                with_define(
                    "#define BAIL return",
                    "    const uint32_t observed = spark_wait_for_flag(({ BAIL; flag; }), deadline);\n    __threadfence();\n    return observed;\n",
                ),
                true,
            ),
            // Round-25 A2: a CONDITIONAL fence followed by an unconditional one. The acquire is
            // present, and without the define the text preprocesses to the shipped baseline; the lint
            // returned at the first, conditional fence while the structural check scanned on.
            (
                "a conditional fence followed by an unconditional one",
                with_define(
                    "",
                    &format!(
                        "{poll_line}\n#if M4_FLAG\n    __threadfence();\n#endif\n    __threadfence();\n    return observed;\n"
                    ),
                ),
                false,
            ),
            // Round-25 A3: a LAMBDA body as the first argument. A `return` inside it exits the lambda,
            // not the helper - the documentation has promised that since round 10 - and the body
            // compiles to the baseline with the acquire present.
            (
                "a lambda body as the poll's first argument",
                with_define(
                    "",
                    "    const uint32_t observed = spark_wait_for_flag(([&]{ return flag; })(), deadline);\n    __threadfence();\n    return observed;\n",
                ),
                false,
            ),
            // Round-25 A3: a control keyword that does NOT bound the path - `if (0) { }` cannot make
            // the fence conditional, and the acquire is present.
            (
                "a non-bounding if in the poll's first argument",
                with_define(
                    "",
                    "    const uint32_t observed = spark_wait_for_flag(({ if (0) { } flag; }), deadline);\n    __threadfence();\n    return observed;\n",
                ),
                false,
            ),
            // Round-25 A4: the LITERAL pre-poll control statement, whose macro twin is above. This row
            // asserts the verdict and the whole-message equality below; the positions table carries the
            // same shape as a grid cell. (Round 30 corrected this comment: it used to say the matrix
            // asserts the verdict only, which stopped being true in round 22, when this table began
            // comparing whole messages for every refusal.)
            (
                "a literal pre-poll control statement",
                with_define(
                    "",
                    &format!(
                        "    if (flag == flag) return 0u;\n{poll_line}\n    __threadfence();\n    return observed;\n"
                    ),
                ),
                // Round-26: this row claims a REFUSAL - by the pre-poll rule, which is why it was
                // labelled accepted in round 25 and slipped through the old comparison.
                true,
            ),
            // Round-26 A1: a lambda with a TRAILING RETURN TYPE. The narrow backward scan from the
            // body's `{` stopped at `*`, `uint32_t` or an identifier, so the lambda's `return` was
            // read as the helper's and both checks refused a body nvcc compiles to 70432 / 4 /
            // 605321041ac5ec1e, the BASELINE. The walk's own detector reads every lambda spelling.
            (
                "a trailing-return-type lambda as the poll's first argument",
                with_define(
                    "",
                    "    const uint32_t observed = spark_wait_for_flag(([&]() -> const volatile uint32_t * { return flag; }()), deadline);\n    __threadfence();\n    return observed;\n",
                ),
                false,
            ),
            (
                "a generic trailing-return lambda as the poll's first argument",
                with_define(
                    "",
                    "    const uint32_t observed = spark_wait_for_flag(([&](auto x) -> const volatile uint32_t * { return x; }(flag)), deadline);\n    __threadfence();\n    return observed;\n",
                ),
                false,
            ),
            // Round-26 A2: a DEFINITELY DISABLED region inside the arguments. `#if 0` removes it, the
            // acquire is present and the cubin is the baseline; the walks read this and the
            // argument-span loops did not.
            (
                "a disabled return in the poll's first argument",
                with_define(
                    "",
                    "    const uint32_t observed = spark_wait_for_flag(({\n#if 0\n        return 0u;\n#endif\n        flag;\n    }), deadline);\n    __threadfence();\n    return observed;\n",
                ),
                false,
            ),
            (
                "a disabled macro in the poll's first argument",
                with_define(
                    "#define BAIL return 0u",
                    "    const uint32_t observed = spark_wait_for_flag(({\n#if 0\n        BAIL;\n#endif\n        flag;\n    }), deadline);\n    __threadfence();\n    return observed;\n",
                ),
                false,
            ),
            // Round-26 A3: a macro whose REPLACEMENT is a lambda holding a return. The replacement is
            // read textually with no lambda skip, so this is a CONSERVATIVE refusal: the text compiles
            // to the baseline and the only reason is that the scan does not read inside a replacement
            // list's lambda. The documentation says that instead of promising otherwise.
            (
                "a macro whose replacement is a lambda with a return",
                with_define(
                    "#define LL ([&]{ return flag; })",
                    "    const uint32_t observed = spark_wait_for_flag((LL)(), deadline);\n    __threadfence();\n    return observed;\n",
                ),
                true,
            ),
            // Round-26 A4: the `&` of a lambda CAPTURE before the poll in one declaration. A capture
            // list is not a path-bounding operator, so this is accepted.
            // Round-28 A1/A2: an exit that belongs to a LAMBDA, before and after the poll, at every
            // nesting the review named. The exit exits the LAMBDA, the helper's path reaches the
            // acquire, and nvcc compiles every one of these to the baseline 70432 / 4 / 605321041ac5ec1e. The
            // bare block at the end is the same text with the lambda removed, and it compiles to
            // 70464 / 2 / 2b56c510b5f4a945 - so the row that must stay refused is refused for a true reason.
            (
                "a non-returning call inside a lambda before the poll",
                with_define(
                    "",
                    "    auto f = [&] { __trap(); };\n    (void)f;\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                ),
                false,
            ),
            (
                "a return in a nested block inside a lambda before the poll",
                with_define(
                    "",
                    "    auto f = [&] { { return; } };\n    (void)f;\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                ),
                false,
            ),
            (
                "a return in an if inside a lambda before the poll",
                with_define(
                    "",
                    "    auto f = [&] { if (flag != nullptr) { return; } };\n    (void)f;\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                ),
                false,
            ),
            (
                "a return in a loop inside a lambda before the poll",
                with_define(
                    "",
                    "    auto f = [&] { for (;;) { return; } };\n    (void)f;\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                ),
                false,
            ),
            (
                "a trap in a nested block inside a lambda before the poll",
                with_define(
                    "",
                    "    auto f = [&] { { __trap(); } };\n    (void)f;\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                ),
                false,
            ),
            (
                "a macro return inside a lambda before the poll",
                with_define(
                    "#define BAIL return",
                    "    auto f = [&] { BAIL; };\n    (void)f;\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                ),
                false,
            ),
            (
                "a return in a nested block inside a lambda after the poll",
                with_define(
                    "",
                    "    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    auto f = [&] { { return; } };\n    (void)f;\n    __threadfence();\n    return observed;\n",
                ),
                false,
            ),
            (
                "a return in an if inside a lambda after the poll",
                with_define(
                    "",
                    "    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    auto f = [&] { if (observed != 0u) { return; } };\n    (void)f;\n    __threadfence();\n    return observed;\n",
                ),
                false,
            ),
            (
                "a trap in a nested block inside a lambda after the poll",
                with_define(
                    "",
                    "    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    auto f = [&] { { __trap(); } };\n    (void)f;\n    __threadfence();\n    return observed;\n",
                ),
                false,
            ),
            (
                "a bare block before the poll",
                with_define(
                    "",
                    "    { return 0u; }\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                ),
                true,
            ),
            // Round-29 A1: an ATTRIBUTE before a compound statement is not a lambda body. Two `]` in
            // a row is never a capture list, so the exit inside these is the HELPER's and both checks
            // must refuse them - round 28 accepted them, which was the regression. The plain nested
            // block at the end is the control: it was refused before and stays refused.
            (
                "an attribute before a nested block before the poll",
                with_define(
                    "",
                    "    [[maybe_unused]] { { return 0u; } }\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                ),
                true,
            ),
            (
                "an attribute before an if before the poll",
                with_define(
                    "",
                    "    [[maybe_unused]] { if (flag != nullptr) { return 0u; } }\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                ),
                true,
            ),
            (
                "an attribute before a while before the poll",
                with_define(
                    "",
                    "    [[maybe_unused]] { while (flag != nullptr) { return 0u; } }\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                ),
                true,
            ),
            (
                "an attribute before a return before the poll",
                with_define(
                    "",
                    "    [[maybe_unused]] { return 0u; }\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                ),
                true,
            ),
            (
                "an attribute before a nested block after the poll",
                with_define(
                    "",
                    "    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    [[maybe_unused]] { { return observed; } }\n    __threadfence();\n    return observed;\n",
                ),
                true,
            ),
            (
                "an attribute before a return after the poll",
                with_define(
                    "",
                    "    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    [[maybe_unused]] { return observed; }\n    __threadfence();\n    return observed;\n",
                ),
                true,
            ),
            (
                "a nested block before the poll",
                with_define(
                    "",
                    "    { { return 0u; } }\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                ),
                true,
            ),
            // Round-30 A1/A2: the spellings the walk-back reads now, each with a return inside the
            // lambda before the poll, so both checks must ACCEPT them. The first two are the ones
            // round 29 refused conservatively, the next three the round-29 `]]` test false-refused.
            (
                "an explicit template parameter list on a lambda",
                with_define(
                    "",
                    "    auto f = [&]<typename T>() { return; };\n    (void)f;\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                ),
                false,
            ),
            (
                "a decltype(auto) return type on a lambda",
                with_define(
                    "",
                    "    auto f = [&]() -> decltype(auto) { return; };\n    (void)f;\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                ),
                false,
            ),
            (
                "an init-capture ending in a subscript",
                with_define(
                    "",
                    "    auto f = [x = (&deadline)[0]] { return; };\n    (void)f;\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                ),
                false,
            ),
            (
                "a capture list with a trailing attribute",
                with_define(
                    "",
                    "    auto f = [&] [[maybe_unused]] { return; };\n    (void)f;\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                ),
                false,
            ),
            (
                "an empty capture list",
                with_define(
                    "",
                    "    auto f = [] { return; };\n    (void)f;\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                ),
                false,
            ),
            // Round-31 A1/A2: the shapes the per-kind bracket counters and the type-body predicate
            // read. Round 30's single depth counter let a `<` or `>` inside `(...)` cancel that
            // group, so the walk-back never reached the arrow and both checks refused these with
            // the unreachability sentence, which was false of the compiled body: nvcc compiles every
            // one of them to the BASELINE 70432 / 4 / 605321041ac5ec1e with the acquire present.
            (
                "an enable_if with a less-than in the trailing return type",
                with_define(
                    "",
                    "    auto f = [&]() -> std::enable_if_t<(1 < 2), int> { return 0; };\n    (void)f;\n\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                ),
                false,
            ),
            (
                "an enable_if with a greater-than in the trailing return type",
                with_define(
                    "",
                    "    auto f = [&]() -> std::enable_if_t<(2 > 1), int> { return 0; };\n    (void)f;\n\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                ),
                false,
            ),
            (
                "a decltype with a shift in the trailing return type",
                with_define(
                    "",
                    "    auto f = [&]() -> decltype(1 << 2, int{}) { return 0; };\n    (void)f;\n\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                ),
                false,
            ),
            (
                "a decltype with a less-than in the trailing return type",
                with_define(
                    "",
                    "    auto f = [&]() -> decltype((1 < 2), 0) { return 0; };\n    (void)f;\n\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                ),
                false,
            ),
            (
                "a decltype with a less-equal in the trailing return type",
                with_define(
                    "",
                    "    auto f = [&]() -> decltype((1 <= 2), int{}) { return 0; };\n    (void)f;\n\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                ),
                false,
            ),
            (
                "a decltype with a greater-than in the trailing return type",
                with_define(
                    "",
                    "    auto f = [&]() -> decltype((1 > 2), int{}) { return 0; };\n    (void)f;\n\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                ),
                false,
            ),
            (
                "a decltype with a greater-equal in the trailing return type",
                with_define(
                    "",
                    "    auto f = [&]() -> decltype((1 >= 2), int{}) { return 0; };\n    (void)f;\n\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                ),
                false,
            ),
            (
                "an enable_if comparison in a template parameter list",
                with_define(
                    "",
                    "    auto f = [&]<typename T = std::enable_if_t<(1 < 2), int>>() { return ; };\n    (void)f;\n\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                ),
                false,
            ),
            (
                "an enable_if shift in a template parameter list",
                with_define(
                    "",
                    "    auto f = [&]<typename T = std::enable_if_t<(1 << 2) != 0, int>>() { return ; };\n    (void)f;\n\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                ),
                false,
            ),
            (
                "a non-type parameter with a less-than in a template parameter list",
                with_define(
                    "",
                    "    auto f = [&]<auto N = (1 < 2)>() { return ; };\n    (void)f;\n\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                ),
                false,
            ),
            (
                "a non-type parameter with a shift in a template parameter list",
                with_define(
                    "",
                    "    auto f = [&]<auto N = (1 << 2)>() { return ; };\n    (void)f;\n\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                ),
                false,
            ),
            (
                "a return in a local constructor body",
                with_define(
                    "",
                    "    struct S { S() { return; } };\n    (void)sizeof(S);\n\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                ),
                false,
            ),
            (
                "a return in a local destructor body",
                with_define(
                    "",
                    "    struct S { ~S() { return; } };\n    (void)sizeof(S);\n\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                ),
                false,
            ),
            (
                "a return in a local member function body",
                with_define(
                    "",
                    "    struct S { void run() { return; } };\n    (void)sizeof(S);\n\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                ),
                false,
            ),
            // Round-32 A1/A2/A3/A4: the shapes the closing reviews of the merged !8 reported. A1/A2/A4
            // are ACCEPTED by both checks and compile to the BASELINE 70432 / 4 / 605321041ac5ec1e
            // (measured in the stored table; 38640 / 4 / e9cc6998 pre-change); A3 is the FALSE ACCEPT
            // they found - the lambda is CALLED where it stands, so its trap ends the helper, and the
            // text compiles to the cubin this file already refuses as `__trap();` before the poll
            // (69536 / 4 / 5826994128b34ed0, 37872 / 4 / 54cda5bc pre-change). Under the module's
            // one-cubin-one-verdict rule the row claims a REFUSAL.
            (
                "a bare less-than comparison in a trailing return type",
                with_define(
                    "",
                    "    auto f = [&]() -> std::enable_if_t<1 < 2, int> { return 0; };\n    (void)f;\n\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                ),
                false,
            ),
            (
                "a bare less-equal comparison in a trailing return type",
                with_define(
                    "",
                    "    auto f = [&]() -> std::enable_if_t<1 <= 2, int> { return 0; };\n    (void)f;\n\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                ),
                false,
            ),
            (
                "a bare less-than comparison in a template parameter list",
                with_define(
                    "",
                    "    auto f = [&]<int N = 1 < 2>() { return ; };\n    (void)f;\n\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                ),
                false,
            ),
            (
                "a bare less-equal comparison in a template parameter list",
                with_define(
                    "",
                    "    auto f = [&]<auto N = 1 <= 2>() { return ; };\n    (void)f;\n\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                ),
                false,
            ),
            (
                "a bare greater-equal comparison in a template parameter list",
                with_define(
                    "",
                    "    auto f = [&]<auto N = 1 >= 2>() { return ; };\n    (void)f;\n\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                ),
                false,
            ),
            (
                "a return in a local type with two private bases",
                with_define(
                    "",
                    "    struct A {}; struct B {}; struct S : A, B { void run() { return; } };\n    (void)sizeof(S);\n\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                ),
                false,
            ),
            (
                "a return in a local type with two public bases",
                with_define(
                    "",
                    "    struct A {}; struct B {}; class S : public A, public B { public: void run() { return; } };\n    (void)sizeof(S);\n\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                ),
                false,
            ),
            (
                "a return in a local type with an attribute",
                with_define(
                    "",
                    "    struct [[maybe_unused]] S { void run() { return; } };\n    (void)sizeof(S);\n\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                ),
                false,
            ),
            (
                "a return in a local type with a GNU attribute",
                with_define(
                    "",
                    "    struct __attribute__((packed)) S { void run() { return; } };\n    (void)sizeof(S);\n\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                ),
                false,
            ),
            (
                "a return in a local type with an alignas specifier",
                with_define(
                    "",
                    "    struct alignas(8) S { void run() { return; } };\n    (void)sizeof(S);\n\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                ),
                false,
            ),
            (
                "a noexcept lambda with a return",
                with_define(
                    "",
                    "    auto f = [&]() noexcept { return ; };\n    (void)f;\n\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                ),
                false,
            ),
            (
                "a mutable lambda with a return",
                with_define(
                    "",
                    "    auto f = [&]() mutable { return ; };\n    (void)f;\n\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                ),
                false,
            ),
            (
                "a trap in a lambda the source calls before the poll",
                with_define(
                    "",
                    "    (void)([&]{ __trap(); return 0u; })();\n\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                ),
                true,
            ),
            // Round-31 A1, the other direction: a `>` still has to be read as a template close where
            // it is one, and NOT as one where it is a comparison. The first row exercises the `>`
            // piece of the walk-back with no parameter list after it; the second is the control, where
            // a completed comparison statement precedes a body that is NOT a lambda, so the exit in it
            // is still the helper's and both checks refuse (in the same words, which the loop asserts).
            (
                "a template parameter list with no parameter list",
                with_define(
                    "",
                    "    auto f = [&]<typename T> { return; };\n    (void)f;\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                ),
                false,
            ),
            (
                "a greater-than comparison before a bare block",
                with_define(
                    "",
                    "    const bool ready = *flag > 0u;\n    (void)ready;\n    { { return 0u; } }\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                ),
                true,
            ),
            // Round-29 A1, the other direction: the `]]` guard must not break a real capture list.
            // Every capture spelling the promise names is exercised here - `[&]` in the round-28 rows
            // above, and `[=]`, `[x]` and `[&, x]` here, each with a return inside the lambda and the
            // poll after it, so the detector has to read the lambda and both checks have to accept.
            (
                "a by-copy capture list with a return inside",
                with_define(
                    "",
                    "    auto f = [=] { (void)flag; return; };\n    (void)f;\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                ),
                false,
            ),
            (
                "a single-name capture list with a return inside",
                with_define(
                    "",
                    "    auto f = [flag] { (void)flag; return; };\n    (void)f;\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                ),
                false,
            ),
            (
                "a mixed capture list with a return inside",
                with_define(
                    "",
                    "    auto f = [&, flag] { (void)flag; return; };\n    (void)f;\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                ),
                false,
            ),
            (
                "a lambda capture before the poll in the same statement",
                with_define(
                    "",
                    "    const uint32_t capture = (([&]{ flag; }()), 0u), observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed + capture;\n",
                ),
                false,
            ),
        ];
        // Round-22 finding 3: with SPARKFLASH_MATRIX_DUMP set, the very shapes this loop runs are
        // written out, so the container can compile each one and the claim that every row is a
        // translation unit is evidence rather than a reading. The dump is off in every ordinary run.
        let dump = std::env::var("SPARKFLASH_MATRIX_DUMP").ok();
        for (index, (label, source, refuses_with_the_macro_rule)) in shapes.into_iter().enumerate()
        {
            if let Some(directory) = dump.as_deref() {
                std::fs::create_dir_all(directory).expect("the dump directory");
                let name: String = label
                    .chars()
                    .map(|character| {
                        if character.is_ascii_alphanumeric() {
                            character
                        } else {
                            '_'
                        }
                    })
                    .collect();
                std::fs::write(
                    std::path::Path::new(directory).join(format!("{index:02}_{name}.cuh")),
                    &source,
                )
                .expect("the dumped text");
            }
            let structural = acquire_helper_fences_unconditionally(&source);
            let lint = acquire_fence_after_poll(&source, HELPER_POLL);
            assert_eq!(
                structural.is_ok(),
                lint.is_ok(),
                "the two checks disagree about {label}: structural={structural:?} lint={lint:?}"
            );
            if !refuses_with_the_macro_rule {
                // Round-26: the row claims ACCEPTANCE, so both checks have to accept it. Round 25
                // only compared the two results here, and a row both checks REFUSED with identical
                // words therefore passed a row labelled accepted - which is exactly how the
                // trailing-return-type lambdas slipped through as "both checks agree".
                assert!(
                    structural.is_ok() && lint.is_ok(),
                    "the row claims acceptance, so both checks have to accept {label}: \
                     structural={structural:?} lint={lint:?}"
                );
                assert_eq!(
                    structural, lint,
                    "both checks accept {label}, so their whole results have to be equal"
                );
                continue;
            }
            {
                let structural = structural.expect_err(&format!(
                    "the shape is refused, but the structural check accepted {label}"
                ));
                // Most of these shapes are refused by the macro rule, whose sentence the two checks
                // share. The parameter-swapping alias is refused by the mention COUNT instead ("2
                // times"), which the two checks also answer in the same words, so the sentence the
                // structural check used is the one the lint has to carry as well.
                let shared = if structural.contains("a macro whose body contains a control keyword")
                {
                    "a macro whose body contains a control keyword"
                } else if structural.contains("stands before the poll") {
                    "stands before the poll"
                } else if structural.contains("stands between the poll and the acquire") {
                    "stands between the poll and the acquire"
                } else if structural.contains("does not return, so this scan") {
                    // Round-32 A3: the trap rule's sentence, which both checks write word for word
                    // for the called-lambda body and for the bare statement alike.
                    "does not return, so this scan"
                } else {
                    "2 times"
                };
                assert!(
                    structural.ends_with(CANONICAL_HINT),
                    "the structural check's refusal of {label} ends with the hint: {structural}"
                );
                let lint = lint.expect_err(&format!(
                    "the shape is refused, but the lint accepted {label}"
                ));
                assert!(
                    lint.contains(shared) && lint.ends_with(CANONICAL_HINT),
                    "the lint refuses {label} with the same sentence as the structural check \
                     ({shared}): {lint}"
                );
                // Round-22 finding 3: where the two checks claim the same sentence, the whole messages
                // have to be equal, not just a shared substring - a substring assertion is what let a
                // shape look agreed while one side said something else around it. Every refusal by the
                // macro rule, by the poll-statement rule and by the mention COUNT is asserted that way:
                // round 27 replaced the structural check's false "the last mention would be read with
                // no fence after it" with the lint's own sentence, round 29 unified the one word that
                // still differed ("this lint" against "this scan"), and round 30 corrected this comment,
                // which had named a DECLARATION row this table does not carry - the DECLARATION family
                // is two-voiced and the equality below would have failed on it. Every row in this table
                // is compared whole; the two families that still answer differently are the DECLARATION
                // and the MISSING-POLL ones, the module documentation lists both, and
                // `the_missing_poll_family_speaks_in_two_voices` pins the second by name.
                assert_eq!(
                    lint, structural,
                    "the two checks refuse {label} with the same sentence, so their whole messages \
                     have to be equal"
                );
            }
        }
    }

    /// Round-22 finding 1, the systematic part: a GRID rather than a list of named shapes, because
    /// the round-21 review's 1500-case fuzz found 65 disagreements and every one of them was the same
    /// class:
    ///
    ///     - a fence before the poll that the structural check read as the acquire - one wrong fence
    ///       choice rather than 65 wrong rules.
    ///
    /// A grid pins that class rather than one instance of it.
    ///
    /// The grid crosses the fence placements the scan can see (none, before the poll, inside the poll's
    /// own statement, between the poll and the fence, after the fence, and before-and-after) with the
    /// macro placements it reads (none, a completed statement before the poll, the poll's own statement,
    /// a completed statement after the poll, and after the fence), with four macro bodies (a control
    /// keyword, a non-returning call, operators only, and a benign body) and with four statements in
    /// front of the poll (an `if` that returns, a `?` expression, an `&&` expression, and a `for` loop),
    /// and asserts the SAME VERDICT for every cell. The cells are scan shapes rather than stored texts:
    /// what the grid asserts is agreement about a POSITION, and the shapes whose cubin is claimed are in
    /// the measured table, where each of them is a translation unit.
    #[test]
    fn the_two_checks_agree_on_every_grid_cell_of_fence_and_macro_placements() {
        let poll_expression = "spark_wait_for_flag(flag, deadline)";
        let macro_use = "    BAIL;\n";
        let fence_placements = [
            ("no fence at all", 0u8),
            ("a fence before the poll", 1),
            ("a fence inside the poll's statement", 2),
            ("a fence between the poll and the fence", 3),
            ("a fence after the fence", 4),
            ("a fence before and after the poll", 5),
        ];
        let macro_placements = [
            ("no macro use", 0u8),
            ("a use before the poll", 1),
            ("a use in the poll's own statement", 2),
            ("a use after the poll", 3),
            ("a use after the fence", 4),
        ];
        let macro_bodies = [
            ("a macro whose body returns", "#define BAIL return 0u"),
            ("a macro whose body never returns", "#define BAIL __trap()"),
            (
                "a macro whose body is operators only",
                "#define BAIL (deadline > 0u) && true",
            ),
            ("a benign macro", "#define BAIL 0u"),
        ];
        // Round-23: two statements in front of the poll instead of five, because the cells that cost
        // the time are the fence and macro placements and the statements in front of the poll are
        // already covered by the 16-shape matrix over the shipped header. 6 x 4 x 5 x 2 = 240 cells.
        let controls = [
            ("nothing", ""),
            ("an if that returns", "    if (ready) return 0u;\n"),
        ];
        let mut cells = 0usize;
        // Every source the grid runs, so a cell can never silently be a duplicate of another.
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut disagreements: Vec<String> = Vec::new();
        for (fence_label, fence) in fence_placements {
            for (body_label, macro_body) in macro_bodies {
                for (placement_label, placement) in macro_placements {
                    for (control_label, control) in controls {
                        // Round-24: with no fence between the poll and the macro use, the 'after the
                        // fence' placement and the 'after the poll' one write the SAME tokens, so the
                        // cell is not a second shape. It is skipped rather than counted twice, and the
                        // distinctness assertion below is what keeps a duplicate from creeping back.
                        if placement == 4 && fence < 3 {
                            continue;
                        }
                        let mut body = String::from(control);
                        if placement == 1 {
                            body.push_str(macro_use);
                        }
                        if fence == 1 || fence == 5 {
                            body.push_str("    __threadfence();\n");
                        }
                        // The poll's own statement, with the fence and the macro use placed INSIDE it
                        // when those are the placements this cell pairs.
                        body.push_str(&match (fence == 2, placement == 2) {
                            (true, true) => format!(
                                "    const uint32_t observed = (__threadfence(), BAIL, {poll_expression});\n"
                            ),
                            (true, false) => format!(
                                "    const uint32_t observed = (__threadfence(), {poll_expression});\n"
                            ),
                            (false, true) => format!(
                                "    const uint32_t observed = (BAIL, {poll_expression});\n"
                            ),
                            (false, false) => {
                                format!("    const uint32_t observed = {poll_expression};\n")
                            }
                        });
                        if placement == 3 {
                            body.push_str(macro_use);
                        }
                        if fence == 3 || fence == 4 || fence == 5 {
                            body.push_str("    __threadfence();\n");
                        }
                        if fence == 4 {
                            body.push_str("    __threadfence();\n");
                        }
                        if placement == 4 {
                            body.push_str(macro_use);
                        }
                        body.push_str("    return observed;\n");
                        // The grid's own minimal fixture rather than the shipped header: these two
                        // checks read the helper's body and the poll's declaration, so a ten-line text
                        // asks the same positional question for a fraction of the cost - which is what
                        // keeps this grid at 240 cells and the crate suite near the ten seconds the
                        // round-23 review asks for. The 16-shape matrix above runs the shipped header.
                        let source = grid_source(macro_body, &body);
                        let structural = acquire_helper_fences_unconditionally(&source);
                        let lint = acquire_fence_after_poll(&source, HELPER_POLL);
                        cells += 1;
                        seen.insert(source.clone());
                        if structural.is_ok() != lint.is_ok() {
                            disagreements.push(format!(
                                "{fence_label} x {placement_label} x {body_label} x \
                                 {control_label}: structural={structural:?} lint={lint:?}"
                            ));
                        }
                    }
                }
            }
        }
        // 6 x 4 x 5 x 2 = 240 cells, less the 3 x 4 x 2 = 24 the skip above removes: 216 distinct
        // sources.
        assert_eq!(cells, 216, "every distinct cell of the grid was visited");
        assert_eq!(seen.len(), cells, "the grid must not repeat a source");
        assert!(
            disagreements.is_empty(),
            "{} of {cells} grid cells disagree:\n{}",
            disagreements.len(),
            disagreements.join("\n")
        );
        // Round-24: the successor of the round-22 randomized fuzz, deterministic and cheap. The pieces
        // are the grid's own and the seed is fixed, so any failure is reproducible by rerunning this
        // test; the corpus asserts that the two VERDICTS agree, which is the property the fuzz was
        // about, and it reaches arrangements the grid's axes do not pair (a bare `__threadfence(0);`
        // after the poll, a conditional fence behind a compiled one, a control keyword in the poll's
        // own statement).
        let mut state = 0x2545_f491_4f6c_dd1du64;
        for round in 0..160usize {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let mut body = String::new();
            if state & 1 == 1 {
                body.push_str("    if (ready) return 0u;\n");
            }
            body.push_str(if state & 2 == 2 {
                "#define BAIL __trap()\n"
            } else {
                "#define BAIL return 0u\n"
            });
            if state & 4 == 4 {
                body.push_str(macro_use);
            }
            if state & 8 == 8 {
                body.push_str("    __threadfence();\n");
            }
            if state & 16 == 16 {
                body.push_str(&format!(
                    "    const uint32_t observed = (BAIL, {poll_expression});\n"
                ));
            } else {
                body.push_str(&format!(
                    "    const uint32_t observed = {poll_expression};\n"
                ));
            }
            match (state >> 5) % 5 {
                0 => {}
                1 => body.push_str("    __threadfence();\n"),
                2 => body.push_str("    __threadfence(0);\n"),
                3 => body.push_str("#if M4_FLAG\n    __threadfence();\n#endif\n"),
                _ => body.push_str(macro_use),
            }
            body.push_str("    return observed;\n");
            let source = grid_source("", &body);
            let structural = acquire_helper_fences_unconditionally(&source);
            let lint = acquire_fence_after_poll(&source, HELPER_POLL);
            assert_eq!(
                structural.is_ok(),
                lint.is_ok(),
                "the seeded corpus round {round} disagrees:\n{body}\nstructural={structural:?}\nlint={lint:?}"
            );
        }
        // Round-23: the positions the grid does not have, each an explicit row with BOTH verdicts and
        // the WHOLE message where the two checks give the same reason. A fence in the first argument and
        // in the second, bare and as a statement expression, runs BEFORE the call, so it is never the
        // acquire of that observation; a fence between the poll and the end of its own statement is not a
        // bare call statement, and both checks say so in the same words now; and a poll the preprocessor
        // may not compile is refused in both checks, with the same sentence, whether or not a second
        // compiled poll exists. Three of these rows are also stored and measured, and their cubins are
        // the ones the measured table carries: refused_fence_in_the_poll_arguments 70280 / 3 / 18b66cc5f312046c,
        // accepted_fence_in_the_arguments_and_after 70448 / 5 / 488031bfabdebb09, and
        // refused_conditional_poll_with_a_second_poll, the 70432 / 4 / 605321041ac5ec1e baseline without
        // -DM4_FLAG=1 (and 39072 / 95e0cf41 (pre-change) with it, measured with that define).
        let positions: Vec<(&str, String, bool)> = vec![
            (
                "a fence in the first argument, as a statement expression",
                grid_source(
                    "",
                    "    const uint32_t observed = spark_wait_for_flag(({ __threadfence(); flag; }), deadline);\n    return observed;\n",
                ),
                true,
            ),
            (
                "a fence in the first argument, as a bare call",
                grid_source(
                    "",
                    "    const uint32_t observed = spark_wait_for_flag((__threadfence(), flag), deadline);\n    return observed;\n",
                ),
                true,
            ),
            (
                "a fence in the second argument, as a statement expression",
                grid_source(
                    "",
                    "    const uint32_t observed = spark_wait_for_flag(flag, ({ __threadfence(); deadline; }));\n    return observed;\n",
                ),
                true,
            ),
            (
                "a fence in the second argument, as a bare call",
                grid_source(
                    "",
                    "    const uint32_t observed = spark_wait_for_flag(flag, (__threadfence(), deadline));\n    return observed;\n",
                ),
                true,
            ),
            (
                "a fence in the arguments with the acquire after the call",
                grid_source(
                    "",
                    "    const uint32_t observed = spark_wait_for_flag((__threadfence(), flag), deadline);\n    __threadfence();\n    return observed;\n",
                ),
                false,
            ),
            (
                "a fence between the poll and the end of its own statement",
                grid_source(
                    "",
                    "    spark_wait_for_flag(flag, deadline), __threadfence();\n    return 0u;\n",
                ),
                true,
            ),
            (
                "a conditional poll with a second compiled poll",
                grid_source(
                    "",
                    "    #if M4_FLAG\n    spark_wait_for_flag(flag, deadline);\n    #endif\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                ),
                true,
            ),
            (
                "a conditional poll on its own",
                grid_source(
                    "",
                    "    #if M4_FLAG\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    #endif\n    __threadfence();\n    return observed;\n",
                ),
                true,
            ),
            // Round-23 finding 5: a mention spelled with a VOID return type. The parentheses around the
            // designator are also a call's shape, so both checks read this as a call; it is a
            // block-scope redeclaration of the poll, and nvcc refuses the translation unit with
            // "error: incomplete type void is not allowed", so there is no cubin to measure. Both
            // checks refuse it, with the same sentence, which names the type and the compiler's answer.
            // Round-24 A2/A3: an unconditional acquire followed by a CONDITIONAL one is the shipped
            // body once the preprocessor resolves it (70432 / 4 / 605321041ac5ec1e), so it is accepted; a
            // conditional fence with nothing unconditional after the poll is refused by both, in the
            // same sentence; and a fence WRITTEN WITH ARGUMENTS is not the barrier the shared header
            // writes, which one predicate now decides on both sides.
            (
                "an unconditional acquire and a later conditional fence",
                grid_source(
                    "",
                    "    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n#if M4_FLAG\n    __threadfence();\n#endif\n    return observed;\n",
                ),
                false,
            ),
            (
                "a conditional acquire with nothing after it",
                grid_source(
                    "",
                    "    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n#if M4_FLAG\n    __threadfence();\n#endif\n    return observed;\n",
                ),
                true,
            ),
            (
                "a fence written with arguments",
                grid_source(
                    "",
                    "    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence(0);\n    return observed;\n",
                ),
                true,
            ),
            // Round-24 A1: control flow inside the poll's OWN argument list. The span is skipped for
            // fence detection (a fence there runs before the call, so it is never the acquire) but it is
            // still read for control flow, in BOTH checks. `(__trap(), flag)` as the first argument
            // compiles to 69536 / 4 / 5826994128b34ed0, byte-identical to the trap-before-the-poll statement
            // body both checks refuse; `({ return 0u; flag; })` as the first argument to 70464 / 2 /
            // 2b56c510; `(__builtin_unreachable(), flag)` to 68576 / 2 / bfd9e8e177f4f76d, the documented cubin
            // of a return before the poll. All of them are refused by both, in the same words.
            (
                "a trap in the first argument",
                grid_source(
                    "",
                    "    spark_wait_for_flag((__trap(), flag), deadline);\n    __threadfence();\n    return 0u;\n",
                ),
                true,
            ),
            (
                "a return in the first argument",
                grid_source(
                    "",
                    "    spark_wait_for_flag(({ return 0u; flag; }), deadline);\n    __threadfence();\n    return 0u;\n",
                ),
                true,
            ),
            (
                "an unreachable call in the first argument",
                grid_source(
                    "",
                    "    spark_wait_for_flag((__builtin_unreachable(), flag), deadline);\n    __threadfence();\n    return 0u;\n",
                ),
                true,
            ),
            (
                "a trap in the second argument",
                grid_source(
                    "",
                    "    const uint32_t observed = spark_wait_for_flag(flag, (__trap(), deadline));\n    __threadfence();\n    return observed;\n",
                ),
                true,
            ),
            (
                "a return in the second argument",
                grid_source(
                    "",
                    "    const uint32_t observed = spark_wait_for_flag(flag, ({ return deadline; }));\n    __threadfence();\n    return observed;\n",
                ),
                true,
            ),
            // Round-25 A1: the macro twin of the literal shape above, in the first argument and in the
            // second. The structural check read the macro; the lint's argument-span loop did not, so the
            // same compiled body (70464 / 2 / 2b56c510b5f4a945 for the literal `return 0u` argument, 68576 / 2 /
            // bfd9e8e1 for `#define BAIL return`) got two verdicts.
            (
                "a macro that returns in the first argument",
                grid_source(
                    "#define BAIL return 0u",
                    "    spark_wait_for_flag(({ BAIL; flag; }), deadline);\n    __threadfence();\n    return 0u;\n",
                ),
                true,
            ),
            (
                "a macro that returns in the second argument",
                grid_source(
                    "#define BAIL return 0u",
                    "    const uint32_t observed = spark_wait_for_flag(flag, ({ BAIL; deadline; }));\n    __threadfence();\n    return observed;\n",
                ),
                true,
            ),
            // Round-26 A1: the trailing-return-type lambdas, each of which nvcc compiles to the
            // BASELINE 70432 / 4 / 605321041ac5ec1e with the acquire present, and each of which both checks
            // refused because the narrow scan did not walk past the return type.
            (
                "a trailing-return-type lambda in the first argument",
                grid_source(
                    "",
                    "    spark_wait_for_flag(([&]() -> const volatile uint32_t * { return flag; }()), deadline);\n    __threadfence();\n    return 0u;\n",
                ),
                false,
            ),
            (
                "a generic trailing-return lambda in the first argument",
                grid_source(
                    "",
                    "    spark_wait_for_flag(([&](auto x) -> const volatile uint32_t * { return x; }(flag)), deadline);\n    __threadfence();\n    return 0u;\n",
                ),
                false,
            ),
            // Round-26 A2: definitely disabled regions in the arguments, removed by the compiler.
            (
                "a disabled return in the first argument",
                grid_source(
                    "",
                    "    spark_wait_for_flag(({\n#if 0\n        return 0u;\n#endif\n        flag;\n    }), deadline);\n    __threadfence();\n    return 0u;\n",
                ),
                false,
            ),
            (
                "a disabled macro in the first argument",
                grid_source(
                    "#define BAIL return 0u",
                    "    spark_wait_for_flag(({\n#if 0\n        BAIL;\n#endif\n        flag;\n    }), deadline);\n    __threadfence();\n    return 0u;\n",
                ),
                false,
            ),
            // Round-26 A3: the documented CONSERVATIVE refusal - a macro whose replacement is a lambda
            // with a return compiles to the baseline, and the scan refuses it because the replacement is
            // read textually.
            (
                "a macro whose replacement is a lambda with a return",
                grid_source(
                    "#define LL ([&]{ return flag; })",
                    "    spark_wait_for_flag((LL)(), deadline);\n    __threadfence();\n    return 0u;\n",
                ),
                true,
            ),
            // Round-26 A4: the `&` of a capture before the poll in one declaration is not an operator
            // blocker; the body carries no return so the row reads the capture alone.
            // Round-27 A1/A3: a LIVE directive whose next line starts with an exit. Round 26's
            // `end_of_directive(...) + 1` skipped that first token, so neither check saw the exit and
            // both ACCEPTED a body whose compiled acquire is dead (nvcc: 70464 / 2 / 2b56c510b5f4a945, or
            // 69536 / 4 / 5826994128b34ed0 for the trap). Every spelling of a live condition is here, and the
            // second-line control that already worked is here too so the direction is proved both ways.
            (
                "an exit on the first line after #if 1",
                grid_source(
                    "",
                    "    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n#if 1\n    return 0u;\n#endif\n    __threadfence();\n    return observed;\n",
                ),
                true,
            ),
            (
                "an exit on the first line after #if 1u",
                grid_source(
                    "",
                    "    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n#if 1u\n    return 0u;\n#endif\n    __threadfence();\n    return observed;\n",
                ),
                true,
            ),
            (
                "an exit on the first line after #if (1)",
                grid_source(
                    "",
                    "    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n#if (1)\n    return 0u;\n#endif\n    __threadfence();\n    return observed;\n",
                ),
                true,
            ),
            (
                "an exit on the first line of a live #elif",
                grid_source(
                    "",
                    "    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n#if 0\n    flag;\n#elif 1\n    return 0u;\n#endif\n    __threadfence();\n    return observed;\n",
                ),
                true,
            ),
            (
                "an exit on the first line of a live #else",
                grid_source(
                    "",
                    "    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n#if 0\n    flag;\n#else\n    return 0u;\n#endif\n    __threadfence();\n    return observed;\n",
                ),
                true,
            ),
            (
                "a trap on the first line after #if 1",
                grid_source(
                    "",
                    "    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n#if 1\n    __trap();\n#endif\n    __threadfence();\n    return observed;\n",
                ),
                true,
            ),
            // The second-line control, which the off-by-one never affected: it must stay refused, so
            // the fix is proved to have moved the FIRST line only.
            (
                "an exit on the second line after #if 1",
                grid_source(
                    "",
                    "    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n#if 1\n    flag;\n    return 0u;\n#endif\n    __threadfence();\n    return observed;\n",
                ),
                true,
            ),
            (
                "a nested #if 0 immediately after a live directive",
                grid_source(
                    "",
                    "    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n#if 1\n#if 0\n    return 0u;\n#endif\n#endif\n    __threadfence();\n    return observed;\n",
                ),
                false,
            ),
            (
                "a lambda capture before the poll in the same statement",
                grid_source(
                    "",
                    "    const uint32_t capture = (([&]{ flag; }()), 0u), observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed + capture;\n",
                ),
                false,
            ),
            // Round-25 A2: the order the lint got wrong - a conditional fence FIRST and the
            // unconditional acquire after it. Without the define the body preprocesses to the shipped
            // baseline, so both must accept it.
            (
                "a conditional fence followed by an unconditional one",
                grid_source(
                    "",
                    "    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n#if M4_FLAG\n    __threadfence();\n#endif\n    __threadfence();\n    return observed;\n",
                ),
                false,
            ),
            // Round-25 A3: the bodies the argument-span reading was REFUSING although the compiled
            // acquire is present and the cubin is the baseline. A lambda's `return` exits the lambda; a
            // bare `if`/`while`/`switch` that holds no exit does not bound the path at all.
            (
                "a lambda body in the first argument",
                grid_source(
                    "",
                    "    const uint32_t observed = spark_wait_for_flag(([&]{ return flag; })(), deadline);\n    __threadfence();\n    return observed;\n",
                ),
                false,
            ),
            (
                "a non-bounding if in the first argument",
                grid_source(
                    "",
                    "    const uint32_t observed = spark_wait_for_flag(({ if (0) { } flag; }), deadline);\n    __threadfence();\n    return observed;\n",
                ),
                false,
            ),
            (
                "a non-bounding while in the first argument",
                grid_source(
                    "",
                    "    const uint32_t observed = spark_wait_for_flag(({ while (0) { } flag; }), deadline);\n    __threadfence();\n    return observed;\n",
                ),
                false,
            ),
            (
                "a non-bounding switch in the first argument",
                grid_source(
                    "",
                    "    const uint32_t observed = spark_wait_for_flag(({ switch (0) { } flag; }), deadline);\n    __threadfence();\n    return observed;\n",
                ),
                false,
            ),
            // And the exit those keywords can hold: the keyword itself is the blocker, which is why
            // `if (ready) return 0u;` in an argument still refuses while `if (0) { }` does not.
            (
                "an exit inside an if in the first argument",
                grid_source(
                    "",
                    "    const uint32_t observed = spark_wait_for_flag(({ if (0 == 0) return 0u; flag; }), deadline);\n    __threadfence();\n    return observed;\n",
                ),
                true,
            ),
            // Round-25 A4: the LITERAL pre-poll control statement, whose macro twin the matrix has.
            (
                "a literal pre-poll control statement",
                grid_source(
                    "",
                    "    if (0 == 0) return 0u;\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                ),
                true,
            ),
            (
                "a mention spelled with a void return type",
                grid_source(
                    "",
                    "    void (spark_wait_for_flag)(flag, deadline);\n    __threadfence();\n    return 0u;\n",
                ),
                true,
            ),
        ];
        for (label, source, must_refuse) in positions {
            let structural = acquire_helper_fences_unconditionally(&source);
            let lint = acquire_fence_after_poll(&source, HELPER_POLL);
            assert_eq!(
                structural.is_ok(),
                lint.is_ok(),
                "the two checks disagree about {label}: structural={structural:?} lint={lint:?}"
            );
            assert_eq!(
                structural.is_err(),
                must_refuse,
                "{label} has to be refused: structural={structural:?}"
            );
            assert_eq!(
                structural, lint,
                "both checks answer {label} with the same words"
            );
        }
    }

    /// The 1-based line of the first occurrence of `needle`, for asserting what a message is true
    /// about.
    fn line_number_of(text: &str, needle: &str) -> usize {
        text[..text.find(needle).expect("the needle is in the text")]
            .matches('\n')
            .count()
            + 1
    }

    /// Round-15 A6 and A3: the shapes the call-site check used to false-reject, and the macro rule one
    /// of them needed.
    ///
    /// A kernel translation unit may DECLARE the raw poll - what it must not do is call it. A
    /// function-pointer declaration and a declaration with a noexcept specifier are declarations that
    /// nvcc accepts, and the call-site check now reads them as declarations instead of reporting a bare
    /// poll call. The function-pointer spelling is also the shape of a CALL through a pointer -
    /// \`(*name)(args)\` against \`TYPE (*name)(args)\` - so accepting the declaration means this check
    /// cannot separate the two; that is stated as out of scope in the module documentation rather than
    /// claimed away.
    ///
    /// The macro half is round-15 A3: a replacement list headed by \`decltype(\` names a TYPE, so the
    /// \`?\` inside it must not make the macro suspicious - it made a declaration of the poll look like
    /// a call - and a name defined AGAIN with a harmless body replaces the mark its first definition
    /// left, so a stale mark cannot refuse a type.
    #[test]
    fn a_declaration_in_a_kernel_is_not_a_call_whatever_its_specifiers() {
        let observing = format!(
            "__global__ void k(const volatile uint32_t *flag, uint64_t deadline) {{ \
             {ACQUIRE_HELPER}(flag, deadline); }}\n"
        );
        for (label, declaration) in [
            (
                "a function-pointer declaration",
                "uint32_t (*spark_wait_for_flag)(const volatile uint32_t *, uint64_t);",
            ),
            (
                "a noexcept declaration",
                "uint32_t spark_wait_for_flag(const volatile uint32_t *, uint64_t) noexcept;",
            ),
            (
                "a declaration whose return type is a macro headed by decltype",
                "#define SPARK_M4_RETURN decltype(true ? 0u : 1u)\nSPARK_M4_RETURN \
                 spark_wait_for_flag(const volatile uint32_t *, uint64_t);",
            ),
        ] {
            let kernel = format!("#include <stdint.h>\n{declaration}\n{observing}");
            assert!(
                observing_paths_call_the_acquiring_poll(&[("k.cu", kernel.as_str())]).is_ok(),
                "{label} in a kernel is a declaration, not a bare poll call"
            );
        }
        // The other direction, so the acceptance above is not vacuous: a name that expands to a
        // \`return\` is a macro use, and the mention is refused as the call it is.
        let returns = format!(
            "#include <stdint.h>\n#define SPARK_M4_RETURN uint32_t\n#define SPARK_M4_RETURN \
             return\nSPARK_M4_RETURN spark_wait_for_flag(flag, deadline);\n{observing}"
        );
        let error = observing_paths_call_the_acquiring_poll(&[("k.cu", returns.as_str())])
            .expect_err("a macro that expands to a return cannot be a return type");
        assert!(
            error.ends_with(CANONICAL_HINT),
            "the refusal carries the hint: {error}"
        );
        // And the redefinition that makes the point: the name was suspicious when it was first
        // defined, and the definition that replaces it is harmless, so the mark must go with it.
        let harmless = format!(
            "#include <stdint.h>\n#define SPARK_M4_RETURN return\n#define SPARK_M4_RETURN \
             uint32_t\nSPARK_M4_RETURN spark_wait_for_flag(const volatile uint32_t *, uint64_t);\n\
             {observing}"
        );
        assert!(
            observing_paths_call_the_acquiring_poll(&[("k.cu", harmless.as_str())]).is_ok(),
            "a redefinition with a harmless body replaces the mark the first one left"
        );
    }

    /// The shipped helper body, the anchor every measured text is built from.
    const SHIPPED_HELPER_BODY: &str = "    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    \
                                        __threadfence();\n    return observed;\n";

    /// The header with the shipped helper body replaced by `replacement`. Every measured text is this
    /// one substitution, so each one is a mutation of the shipped text and nothing else.
    fn with_the_shipped_body(replacement: &str) -> String {
        assert_eq!(
            COMMON_HEADER.matches(SHIPPED_HELPER_BODY).count(),
            1,
            "the shipped helper body is the one this builder replaces"
        );
        COMMON_HEADER.replace(SHIPPED_HELPER_BODY, replacement)
    }

    /// The grid's and the positions table's own fixture: the poll's declaration and the helper, with no
    /// header around them. Both checks read the helper's body and the poll's declaration, so this asks
    /// the same positional question the shipped header asks at a fraction of the cost - which is what
    /// keeps 240 grid cells plus the extra rows from costing the crate suite its ten seconds.
    ///
    /// It is NOT a translation unit and this file claims no cubin for it: with no header around it, nvcc
    /// rejects it for the missing `uint32_t` and `uint64_t`, which the module documentation says in the
    /// WHICH BODIES ARE TRANSLATION UNITS paragraph as well. What the container compiles is the MATRIX
    /// rows, which carry the header, and the measured table, which is built from the shipped header by
    /// `with_the_shipped_body` - the substitution whose cubins this documentation quotes.
    fn grid_source(macro_body: &str, body: &str) -> String {
        format!(
            "{macro_body}\nuint32_t spark_wait_for_flag(const volatile uint32_t *flag, uint64_t deadline);\nstatic uint32_t spark_wait_for_flag_acquire(const volatile uint32_t *flag, uint64_t deadline) {{\n{body}}}\n"
        )
    }

    /// What both checks must answer about a measured text.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum Verdict {
        Accepted,
        Refused,
    }

    /// THE single source of the measured texts: the name the text is stored under, the text itself, the
    /// verdict BOTH checks must give it, the cubin it compiles to in the CUDA 13.0.3 container (sm_121a,
    /// the mega_mhc kernel, exactly as the committed baseline is), and the phrase its refusal has to
    /// carry (empty for an accepted text).
    ///
    /// The table is the authoritative record: every CURRENT cubin quoted in this module's documentation
    /// is this table's entry for its text (prose numbers marked pre-change are historical), and every
    /// text in this table is stored under /home/user/wt/targets/kd17_texts/<name>.cuh, where
    /// `the_stored_measured_texts_are_the_texts_the_tests_build` checks it byte for byte.
    fn measured_texts() -> Vec<(&'static str, String, Verdict, &'static str, &'static str)> {
        let mut texts: Vec<(&'static str, String, Verdict, &'static str, &'static str)> = vec![(
            "shipped",
            COMMON_HEADER.to_string(),
            Verdict::Accepted,
            "70432 / 4 / 605321041ac5ec1e",
            "",
        )];
        // The shapes an earlier rule refused as "not the canonical observation" and that are CALLS:
        // every one of them executes the poll, and each cubin is the one its text measures.
        for (name, replacement, cubin) in [
            (
                "bare_call",
                "    spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return 0u;\n",
                "70960 / 3 / bbafa1b88b3c3be7",
            ),
            (
                "unary",
                "    const bool observed = !spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed ? 1u : 0u;\n",
                "70432 / 4 / f715d21b4aef0271",
            ),
            (
                "eq",
                "    const bool observed = 0u == spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed ? 1u : 0u;\n",
                "70432 / 4 / f715d21b4aef0271",
            ),
            (
                "gt",
                "    const bool observed = 0u > spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed ? 1u : 0u;\n",
                "70960 / 3 / bbafa1b88b3c3be7",
            ),
            (
                "lt",
                "    const bool observed = spark_wait_for_flag(flag, deadline) < deadline;\n    __threadfence();\n    return observed ? 1u : 0u;\n",
                "70432 / 3 / 439d36ffff9ee9d5",
            ),
            (
                "chained",
                "    const bool observed = 0u < spark_wait_for_flag(flag, deadline) < deadline;\n    __threadfence();\n    return observed ? 1u : 0u;\n",
                "70432 / 3 / 439d36ffff9ee9d5",
            ),
            (
                "cast_seed",
                "    const uint32_t observed = (uint32_t) spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                "70432 / 4 / 605321041ac5ec1e",
            ),
            (
                "void_cast",
                "    (void) spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return 0u;\n",
                "70960 / 3 / bbafa1b88b3c3be7",
            ),
            (
                "attribute_decl",
                "    [[maybe_unused]] const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                "70432 / 4 / 605321041ac5ec1e",
            ),
            (
                "attribute_stmt",
                "    [[maybe_unused]] spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return 0u;\n",
                "70960 / 3 / bbafa1b88b3c3be7",
            ),
            (
                "spaced",
                "    const uint32_t observed = spark_wait_for_flag ( flag , deadline );\n    __threadfence();\n    return observed;\n",
                "70432 / 4 / 605321041ac5ec1e",
            ),
            (
                "paren_args",
                "    const uint32_t observed = spark_wait_for_flag((flag), (deadline));\n    __threadfence();\n    return observed;\n",
                "70432 / 4 / 605321041ac5ec1e",
            ),
            (
                "cast_arg",
                "    const uint32_t observed = spark_wait_for_flag((const uint32_t *)flag, deadline);\n    __threadfence();\n    return observed;\n",
                "70432 / 4 / 605321041ac5ec1e",
            ),
            // Round-18 A1: the designator itself is parenthesised, dereferenced or addressed, which
            // `call_argument_list` unwraps and the old rule - a literal `(` after the name - did not.
            (
                "paren_designator",
                "    const uint32_t observed = (spark_wait_for_flag)(flag, deadline);\n    __threadfence();\n    return observed;\n",
                "70432 / 4 / 605321041ac5ec1e",
            ),
            (
                "deref_designator",
                "    const uint32_t observed = (*spark_wait_for_flag)(flag, deadline);\n    __threadfence();\n    return observed;\n",
                "70432 / 4 / 605321041ac5ec1e",
            ),
            (
                "addr_designator",
                "    const uint32_t observed = (&spark_wait_for_flag)(flag, deadline);\n    __threadfence();\n    return observed;\n",
                "70432 / 4 / 605321041ac5ec1e",
            ),
            // Round-18 A2/A3: the poll is the condition of a control statement, or an operand of the
            // conditional operator, and the fence after that whole statement runs unconditionally.
            (
                "ternary",
                "    const uint32_t observed = true ? spark_wait_for_flag(flag, deadline) : 0u;\n    __threadfence();\n    return observed;\n",
                "70432 / 4 / 605321041ac5ec1e",
            ),
            (
                "if_condition",
                "    if (spark_wait_for_flag(flag, deadline)) {}\n    __threadfence();\n    return 0u;\n",
                "70960 / 3 / bbafa1b88b3c3be7",
            ),
            (
                "while_condition",
                "    while (spark_wait_for_flag(flag, deadline)) {}\n    __threadfence();\n    return 0u;\n",
                "71088 / 3 / a9ef975cf8356c47",
            ),
            // Round-18 A4: the earlier `<` of a comparison in the poll's own statement, inside balanced
            // parentheses and bare.
            (
                "paren_less_than",
                "    const bool observed = (0u < 1u) > spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                "70432 / 4 / f715d21b4aef0271",
            ),
            (
                "bare_less_than",
                "    const bool observed = 0u < 1u > spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                "70432 / 4 / f715d21b4aef0271",
            ),
            // The parenthesised comparison whose left operand is an IDENTIFIER: the template-name rule
            // cannot rescue this one, only skipping the balanced group can, so it is what makes the
            // group skip load-bearing (round-18 A4).
            (
                "paren_identifier_less_than",
                "    const bool observed = (deadline < 1u) > spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                "70472 / 3 / 03d8a68f78d8b1ce",
            ),
            // Round-19 A1: a control OPERATOR in an earlier, COMPLETED statement. Its semicolon stands
            // between it and the poll, so it never bounds the poll's own statement; all four compile to
            // the baseline cubin, which is what makes refusing them a false rejection.
            (
                "earlier_and",
                "    const bool ran = (deadline > 0u) && true;\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                "70432 / 4 / 605321041ac5ec1e",
            ),
            (
                "earlier_or",
                "    const bool ran = (deadline > 0u) || true;\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                "70432 / 4 / 605321041ac5ec1e",
            ),
            (
                "earlier_ampersand",
                "    const bool took = (deadline > 0u) & true;\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                "70432 / 4 / 605321041ac5ec1e",
            ),
            (
                "earlier_ternary",
                "    const uint32_t earlier = true ? 1u : 0u;\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                "70432 / 4 / 605321041ac5ec1e",
            ),
        ] {
            texts.push((
                name,
                with_the_shipped_body(replacement),
                Verdict::Accepted,
                cubin,
                "",
            ));
        }
        // Round-30 A1/A2: the two spellings round 29 refused conservatively, and the two lambda
        // spellings round 29's `]]` test false-refused. nvcc compiles all four to the BASELINE,
        // which is why they are rows here and not refusals - the exit belongs to the lambda.
        for (name, replacement, cubin) in [
            (
                "accepted_template_parameter_list_lambda_with_a_return",
                "    auto f = [&]<typename T>() { return; };\n    (void)f;\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                "70432 / 4 / 605321041ac5ec1e",
            ),
            (
                "accepted_decltype_auto_lambda_with_a_return",
                "    auto f = [&]() -> decltype(auto) { return; };\n    (void)f;\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                "70432 / 4 / 605321041ac5ec1e",
            ),
            (
                "accepted_init_capture_ending_in_a_subscript",
                "    auto f = [x = (&deadline)[0]] { return; };\n    (void)f;\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                "70432 / 4 / 605321041ac5ec1e",
            ),
            (
                "accepted_capture_list_with_a_trailing_attribute",
                "    auto f = [&] [[maybe_unused]] { return; };\n    (void)f;\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                "70432 / 4 / 605321041ac5ec1e",
            ),
            (
                "accepted_empty_capture_list_with_a_return",
                "    auto f = [] { return; };\n    (void)f;\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                "70432 / 4 / 605321041ac5ec1e",
            ),
            (
                "accepted_enable_if_less_than_in_a_trailing_return",
                "    auto f = [&]() -> std::enable_if_t<(1 < 2), int> { return 0; };\n    (void)f;\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                "70432 / 4 / 605321041ac5ec1e",
            ),
            (
                "accepted_enable_if_greater_than_in_a_trailing_return",
                "    auto f = [&]() -> std::enable_if_t<(2 > 1), int> { return 0; };\n    (void)f;\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                "70432 / 4 / 605321041ac5ec1e",
            ),
            (
                "accepted_decltype_shift_in_a_trailing_return",
                "    auto f = [&]() -> decltype(1 << 2, int{}) { return 0; };\n    (void)f;\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                "70432 / 4 / 605321041ac5ec1e",
            ),
            (
                "accepted_decltype_less_than_in_a_trailing_return",
                "    auto f = [&]() -> decltype((1 < 2), 0) { return 0; };\n    (void)f;\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                "70432 / 4 / 605321041ac5ec1e",
            ),
            (
                "accepted_decltype_less_equal_in_a_trailing_return",
                "    auto f = [&]() -> decltype((1 <= 2), int{}) { return 0; };\n    (void)f;\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                "70432 / 4 / 605321041ac5ec1e",
            ),
            (
                "accepted_decltype_greater_than_in_a_trailing_return",
                "    auto f = [&]() -> decltype((1 > 2), int{}) { return 0; };\n    (void)f;\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                "70432 / 4 / 605321041ac5ec1e",
            ),
            (
                "accepted_decltype_greater_equal_in_a_trailing_return",
                "    auto f = [&]() -> decltype((1 >= 2), int{}) { return 0; };\n    (void)f;\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                "70432 / 4 / 605321041ac5ec1e",
            ),
            (
                "accepted_enable_if_less_than_in_a_template_parameter_list",
                "    auto f = [&]<typename T = std::enable_if_t<(1 < 2), int>>() { return ; };\n    (void)f;\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                "70432 / 4 / 605321041ac5ec1e",
            ),
            (
                "accepted_enable_if_shift_in_a_template_parameter_list",
                "    auto f = [&]<typename T = std::enable_if_t<(1 << 2) != 0, int>>() { return ; };\n    (void)f;\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                "70432 / 4 / 605321041ac5ec1e",
            ),
            (
                "accepted_non_type_parameter_less_than_in_a_template_parameter_list",
                "    auto f = [&]<auto N = (1 < 2)>() { return ; };\n    (void)f;\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                "70432 / 4 / 605321041ac5ec1e",
            ),
            (
                "accepted_non_type_parameter_shift_in_a_template_parameter_list",
                "    auto f = [&]<auto N = (1 << 2)>() { return ; };\n    (void)f;\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                "70432 / 4 / 605321041ac5ec1e",
            ),
            (
                "accepted_return_in_a_local_constructor",
                "    struct S { S() { return; } };\n    (void)sizeof(S);\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                "70432 / 4 / 605321041ac5ec1e",
            ),
            (
                "accepted_return_in_a_local_destructor",
                "    struct S { ~S() { return; } };\n    (void)sizeof(S);\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                "70432 / 4 / 605321041ac5ec1e",
            ),
            (
                "accepted_return_in_a_local_member_function",
                "    struct S { void run() { return; } };\n    (void)sizeof(S);\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                "70432 / 4 / 605321041ac5ec1e",
            ),
            (
                "accepted_trailing_return_lambda",
                "    const uint32_t observed = spark_wait_for_flag(([&]() -> const volatile uint32_t * { return flag; }()), deadline);\n    __threadfence();\n    return observed;\n",
                "70432 / 4 / 605321041ac5ec1e",
            ),
            (
                "accepted_generic_trailing_return_lambda",
                "    const uint32_t observed = spark_wait_for_flag(([&](auto x) -> const volatile uint32_t * { return x; }(flag)), deadline);\n    __threadfence();\n    return observed;\n",
                "70432 / 4 / 605321041ac5ec1e",
            ),
            (
                "accepted_disabled_return_in_the_arguments",
                "    const uint32_t observed = spark_wait_for_flag(({\n#if 0\n        return 0u;\n#endif\n        flag;\n    }), deadline);\n    __threadfence();\n    return observed;\n",
                "70432 / 4 / 605321041ac5ec1e",
            ),
            (
                "accepted_disabled_macro_in_the_arguments",
                "    #define BAIL return 0u\n    const uint32_t observed = spark_wait_for_flag(({\n#if 0\n        BAIL;\n#endif\n        flag;\n    }), deadline);\n    __threadfence();\n    return observed;\n",
                "70432 / 4 / 605321041ac5ec1e",
            ),
            (
                "accepted_lambda_capture_before_the_poll",
                "    const uint32_t capture = (([&]{ flag; }()), 0u), observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed + capture;\n",
                "70432 / 4 / 605321041ac5ec1e",
            ),
            // Round-27 A2: the lambda whose body is only `return flag;`, in an earlier COMPLETED
            // statement before the poll. The return exits the LAMBDA, so the helper reaches the
            // acquire: nvcc compiles the text to the BASELINE, which is why a refusal that claimed
            // otherwise asserted a false fact about the compiled code.
            // Round-28 A1/A2: an exit that belongs to a LAMBDA, at every nesting the review named,
            // before and after the poll. The exit exits the lambda, the acquire still runs, and every
            // one of these is the BASELINE 70432 / 4 / 605321041ac5ec1e - measured, not quoted.
            (
                "accepted_nonreturning_call_in_a_lambda_before_the_poll",
                "    auto f = [&] { __trap(); };\n    (void)f;\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                "70432 / 4 / 605321041ac5ec1e",
            ),
            (
                "accepted_return_in_a_nested_block_inside_a_lambda",
                "    auto f = [&] { { return; } };\n    (void)f;\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                "70432 / 4 / 605321041ac5ec1e",
            ),
            (
                "accepted_return_in_an_if_inside_a_lambda",
                "    auto f = [&] { if (flag != nullptr) { return; } };\n    (void)f;\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                "70432 / 4 / 605321041ac5ec1e",
            ),
            (
                "accepted_return_in_a_loop_inside_a_lambda",
                "    auto f = [&] { for (;;) { return; } };\n    (void)f;\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                "70432 / 4 / 605321041ac5ec1e",
            ),
            (
                "accepted_trap_in_a_nested_block_inside_a_lambda",
                "    auto f = [&] { { __trap(); } };\n    (void)f;\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                "70432 / 4 / 605321041ac5ec1e",
            ),
            (
                "accepted_macro_return_in_a_lambda_before_the_poll",
                "    #define BAIL return\n    auto f = [&] { BAIL; };\n    (void)f;\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                "70432 / 4 / 605321041ac5ec1e",
            ),
            (
                "accepted_return_in_a_nested_block_inside_a_lambda_after_the_poll",
                "    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    auto f = [&] { { return; } };\n    (void)f;\n    __threadfence();\n    return observed;\n",
                "70432 / 4 / 605321041ac5ec1e",
            ),
            (
                "accepted_return_in_an_if_inside_a_lambda_after_the_poll",
                "    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    auto f = [&] { if (observed != 0u) { return; } };\n    (void)f;\n    __threadfence();\n    return observed;\n",
                "70432 / 4 / 605321041ac5ec1e",
            ),
            (
                "accepted_trap_in_a_nested_block_inside_a_lambda_after_the_poll",
                "    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    auto f = [&] { { __trap(); } };\n    (void)f;\n    __threadfence();\n    return observed;\n",
                "70432 / 4 / 605321041ac5ec1e",
            ),
            (
                "accepted_lambda_return_before_the_poll",
                "    auto f = [&] { return flag; };\n    (void)f;\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                "70432 / 4 / 605321041ac5ec1e",
            ),
            // Round-32 A1/A2/A4, every one of them measured at the BASELINE 70432 / 4 / 605321041ac5ec1e
            // (38640 / 4 / e9cc6998 pre-change). A1:
            // the comparison written BARE in a template argument or parameter list - a `<` is a
            // delimiter only when a template NAME stands in front of it, and the `>` of a `>=` is not a
            // close. A2: the class-head spellings whose comma, `]` or `)` stopped the type-body walk
            // (two base clauses, `[[maybe_unused]]`, a GNU `__attribute__` and `alignas(8)`). A4: the two
            // specifier spellings both checks already accepted and nothing pinned until now.
            (
                "accepted_bare_less_than_in_a_trailing_return",
                "    auto f = [&]() -> std::enable_if_t<1 < 2, int> { return 0; };\n    (void)f;\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                "70432 / 4 / 605321041ac5ec1e",
            ),
            (
                "accepted_bare_less_equal_in_a_trailing_return",
                "    auto f = [&]() -> std::enable_if_t<1 <= 2, int> { return 0; };\n    (void)f;\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                "70432 / 4 / 605321041ac5ec1e",
            ),
            (
                "accepted_bare_less_than_in_a_template_parameter_list",
                "    auto f = [&]<int N = 1 < 2>() { return ; };\n    (void)f;\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                "70432 / 4 / 605321041ac5ec1e",
            ),
            (
                "accepted_bare_less_equal_in_a_template_parameter_list",
                "    auto f = [&]<auto N = 1 <= 2>() { return ; };\n    (void)f;\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                "70432 / 4 / 605321041ac5ec1e",
            ),
            (
                "accepted_bare_greater_equal_in_a_template_parameter_list",
                "    auto f = [&]<auto N = 1 >= 2>() { return ; };\n    (void)f;\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                "70432 / 4 / 605321041ac5ec1e",
            ),
            (
                "accepted_return_in_a_local_type_with_two_bases",
                "    struct A {}; struct B {}; struct S : A, B { void run() { return; } };\n    (void)sizeof(S);\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                "70432 / 4 / 605321041ac5ec1e",
            ),
            (
                "accepted_return_in_a_local_type_with_public_bases",
                "    struct A {}; struct B {}; class S : public A, public B { public: void run() { return; } };\n    (void)sizeof(S);\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                "70432 / 4 / 605321041ac5ec1e",
            ),
            (
                "accepted_return_in_a_local_type_with_an_attribute",
                "    struct [[maybe_unused]] S { void run() { return; } };\n    (void)sizeof(S);\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                "70432 / 4 / 605321041ac5ec1e",
            ),
            (
                "accepted_return_in_a_local_type_with_a_gnu_attribute",
                "    struct __attribute__((packed)) S { void run() { return; } };\n    (void)sizeof(S);\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                "70432 / 4 / 605321041ac5ec1e",
            ),
            (
                "accepted_return_in_a_local_type_with_alignas",
                "    struct alignas(8) S { void run() { return; } };\n    (void)sizeof(S);\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                "70432 / 4 / 605321041ac5ec1e",
            ),
            (
                "accepted_noexcept_lambda_with_a_return",
                "    auto f = [&]() noexcept { return ; };\n    (void)f;\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                "70432 / 4 / 605321041ac5ec1e",
            ),
            (
                "accepted_mutable_lambda_with_a_return",
                "    auto f = [&]() mutable { return ; };\n    (void)f;\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                "70432 / 4 / 605321041ac5ec1e",
            ),
        ] {
            texts.push((
                name,
                with_the_shipped_body(replacement),
                Verdict::Accepted,
                cubin,
                "",
            ));
        }
        // Round-17 A2/A4: the arguments are not read, so a helper whose own parameters are named `f` and
        // `d` is not asked to write `flag` and `deadline`; and a macro whose replacement list is exactly
        // the designator IS the designator, one level deep.
        let signature_head = "spark_wait_for_flag_acquire(const volatile uint32_t *flag,";
        assert_eq!(
            COMMON_HEADER.matches(signature_head).count(),
            1,
            "the helper is the one function with this signature"
        );
        let head_at = COMMON_HEADER
            .find(signature_head)
            .expect("the helper's signature");
        let after_head = head_at + signature_head.len();
        let parameter_end = "uint64_t deadline) {";
        let tail_at = after_head
            + COMMON_HEADER[after_head..]
                .find(parameter_end)
                .expect("the helper's second parameter");
        let signature = &COMMON_HEADER[head_at..tail_at + parameter_end.len()];
        texts.push((
            "renamed",
            COMMON_HEADER
                .replace(
                    signature,
                    &signature
                        .replace("*flag,", "*f,")
                        .replace(parameter_end, "uint64_t d) {"),
                )
                .replace(
                    SHIPPED_HELPER_BODY,
                    "    const uint32_t observed = spark_wait_for_flag(f, d);\n    __threadfence();\n    return observed;\n",
                ),
            Verdict::Accepted,
            "70432 / 4 / 605321041ac5ec1e",
            "",
        ));
        // Round-19 A5: a parenthesised macro replacement is the same alias, and the designator inside
        // the parentheses may be dereferenced or addressed. All three compile to the baseline cubin.
        for (name, define) in [
            ("macro_alias_paren", "#define P (spark_wait_for_flag)"),
            ("macro_alias_deref", "#define P (*spark_wait_for_flag)"),
            ("macro_alias_addr", "#define P (&spark_wait_for_flag)"),
        ] {
            texts.push((
                name,
                format!("{define}\n{COMMON_HEADER}").replace(
                    SHIPPED_HELPER_BODY,
                    "    const uint32_t observed = P(flag, deadline);\n    __threadfence();\n    return observed;\n",
                ),
                Verdict::Accepted,
                "70432 / 4 / 605321041ac5ec1e",
                "",
            ));
        }
        // Round-20 A3: three more spellings, all measured at the baseline cubin. A doubly parenthesised
        // replacement is the same alias; a replacement that is the WHOLE call makes the use itself the
        // observation, so no argument list follows it; and a macro whose body carries only an operator
        // obeys the completed-statement scoping of the literal operator.
        for (name, define, body) in [
            (
                "macro_operator_before_the_poll",
                "#define PRE (deadline > 0u) && true",
                "    const bool ran = PRE;\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
            ),
            (
                "macro_double_paren_alias",
                "#define P ((spark_wait_for_flag))",
                "    const uint32_t observed = P(flag, deadline);\n    __threadfence();\n    return observed;\n",
            ),
            (
                "macro_whole_call_alias",
                "#define P spark_wait_for_flag(flag, deadline)",
                "    const uint32_t observed = P;\n    __threadfence();\n    return observed;\n",
            ),
        ] {
            texts.push((
                name,
                format!("{define}\n{COMMON_HEADER}").replace(SHIPPED_HELPER_BODY, body),
                Verdict::Accepted,
                "70432 / 4 / 605321041ac5ec1e",
                "",
            ));
        }
        // Round-21 A3: five more spellings, all accepted by both checks. Any number of balanced pairs
        // around the designator is still the designator; a function-like macro whose replacement list is
        // the whole call with the macro's own parameters in the arguments is still the whole call (the
        // arguments are not read, here as at a use); a replacement list that is the whole call WRAPPED in
        // parentheses was refused until round 21 although it compiles to the baseline, and the
        // parenthesised use of that alias is accepted with it. The one number that is not the baseline is
        // the discarded-value statement form, which measures what every discarded-value call in the table
        // measures.
        // Round-22 A1: a fence BEFORE the poll with the acquire still after it. This is the shape the
        // round-21 review measured - the extra fence before the poll is the only delta from the baseline
        // - and the structural check refused it until round 22 because it took the first fence ANYWHERE
        // as the acquire. There is no macro in it, so it carries no define.
        texts.push((
            "accepted_fence_before_the_poll",
            COMMON_HEADER.replace(
                SHIPPED_HELPER_BODY,
                &format!("    __threadfence();\n{SHIPPED_HELPER_BODY}"),
            ),
            Verdict::Accepted,
            "70448 / 5 / 488031bfabdebb09",
            "",
        ));
        // Round-23 A2 (HIGH): a fence in the poll's ARGUMENTS plus a real bare acquire after it. The
        // structural check used to take the in-arguments fence as the acquire and ignore the real one,
        // and refused the body while the lint accepted it; the fence search now starts after the call's
        // argument list, exactly where the lint's walk starts, so both accept it. Its cubin is the cubin
        // of the accepted fence-before-and-after shape above: the acquire after the poll is the same one.
        texts.push((
            "accepted_fence_in_the_arguments_and_after",
            with_the_shipped_body(
                "    const uint32_t observed = spark_wait_for_flag((__threadfence(), flag), deadline);\n    __threadfence();\n    return observed;\n",
            ),
            Verdict::Accepted,
            "70448 / 5 / 488031bfabdebb09",
            "",
        ));
        for (name, define, body, cubin) in [
            (
                "macro_triple_paren_alias",
                "#define P (((spark_wait_for_flag)))",
                "    const uint32_t observed = P(flag, deadline);\n    __threadfence();\n    return observed;\n",
                "70432 / 4 / 605321041ac5ec1e",
            ),
            (
                "macro_whole_call_with_parameters",
                "#define P(a, b) spark_wait_for_flag(a, b)",
                "    const uint32_t observed = P(flag, deadline);\n    __threadfence();\n    return observed;\n",
                "70432 / 4 / 605321041ac5ec1e",
            ),
            (
                "macro_parenthesised_whole_call",
                "#define P (spark_wait_for_flag(flag, deadline))",
                "    const uint32_t observed = P;\n    __threadfence();\n    return observed;\n",
                "70432 / 4 / 605321041ac5ec1e",
            ),
            (
                "macro_parenthesised_whole_call_use",
                "#define P (spark_wait_for_flag(flag, deadline))",
                "    const uint32_t observed = (P);\n    __threadfence();\n    return observed;\n",
                "70432 / 4 / 605321041ac5ec1e",
            ),
            (
                "macro_parenthesised_whole_call_statement",
                "#define P (spark_wait_for_flag(flag, deadline))",
                "    (P);\n    __threadfence();\n    return 0u;\n",
                "70960 / 3 / bbafa1b88b3c3be7",
            ),
            // Round-22 A2: the same call with its DESIGNATOR parenthesised. Neither the designator rule
            // nor the whole-call rule read it, so both checks refused a spelling that compiles to the
            // baseline cubin; the mention rule now skips the designator's own parentheses.
            (
                "macro_parenthesised_designator_whole_call",
                "#define P (spark_wait_for_flag)(flag, deadline)",
                "    const uint32_t observed = P;\n    __threadfence();\n    return observed;\n",
                "70432 / 4 / 605321041ac5ec1e",
            ),
        ] {
            texts.push((
                name,
                format!("{define}\n{COMMON_HEADER}").replace(SHIPPED_HELPER_BODY, body),
                Verdict::Accepted,
                cubin,
                "",
            ));
        }
        texts.push((
            "alias",
            format!("#define POLL_ALIAS spark_wait_for_flag\n{COMMON_HEADER}").replace(
                SHIPPED_HELPER_BODY,
                "    const uint32_t observed = POLL_ALIAS(flag, deadline);\n    __threadfence();\n    return observed;\n",
            ),
            Verdict::Accepted,
            "70432 / 4 / 605321041ac5ec1e",
            "",
        ));
        // The REFUSED list: a mention that is not a call. Each of these bodies only DECLARES the poll,
        // so the code it compiles to is not the baseline (70600 / 3 / 1f2503fd7ecaf189, measured). The bare
        // block is refused for the other reason (round-18 A8): the poll and the fence sit inside a
        // block neither check can relate to the helper's straight-line path, so both refuse it even
        // though its cubin IS the baseline.
        for (name, source, needle, cubin) in [
            (
                "refused_function_pointer_declarator",
                with_the_shipped_body(
                    "    uint32_t (*spark_wait_for_flag)(const volatile uint32_t *, uint64_t);\n    __threadfence();\n    return 0u;\n",
                ),
                "the shared header writes the call as",
                "70600 / 3 / 1f2503fd7ecaf189",
            ),
            (
                "refused_attribute_between_type_and_name",
                with_the_shipped_body(
                    "    uint32_t [[maybe_unused]] spark_wait_for_flag(const volatile uint32_t *, uint64_t);\n    __threadfence();\n    return 0u;\n",
                ),
                "the shared header writes the call as",
                "70600 / 3 / 1f2503fd7ecaf189",
            ),
            (
                "refused_macro_return",
                format!(
                    "#define SPARK_M4_RETURN decltype(true ? 0u : 1u)\n{}",
                    with_the_shipped_body(
                        "    SPARK_M4_RETURN spark_wait_for_flag(const volatile uint32_t *, uint64_t);\n    __threadfence();\n    return 0u;\n",
                    )
                ),
                "the shared header writes the call as",
                "70600 / 3 / 1f2503fd7ecaf189",
            ),
            // Round-28 A2: the same body with the lambda removed. The `return` is the HELPER's, the
            // fence after it never runs, and it compiles to 70464 / 2 / 2b56c510b5f4a945 - so this refusal is
            // not collateral damage of the lambda rule.
            (
                "refused_bare_block_before_the_poll",
                with_the_shipped_body(
                    "    { return 0u; }\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                ),
                "stands before the poll",
                "70464 / 2 / 2b56c510b5f4a945",
            ),
            // Round-29 A1: an ATTRIBUTE before a compound statement. Round 28's outward walk read the
            // closing `]]` of an attribute-specifier as a lambda capture list, so both checks ACCEPTED
            // these although the exit in each is the HELPER's. Every cubin here is measured, and the
            // pre-poll ones are the 70464 / 2 / 2b56c510b5f4a945 cubin the module documentation assigns to a
            // helper-level exit before the poll, while the after-poll ones are 70264 / 2 / 42454ad1740a315c.
            (
                "refused_attribute_before_a_block_before_the_poll",
                with_the_shipped_body(
                    "    [[maybe_unused]] { { return 0u; } }\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                ),
                "stands before the poll",
                "70464 / 2 / 2b56c510b5f4a945",
            ),
            (
                "refused_attribute_before_an_if_before_the_poll",
                with_the_shipped_body(
                    "    [[maybe_unused]] { if (flag != nullptr) { return 0u; } }\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                ),
                "stands before the poll",
                "70464 / 2 / 2b56c510b5f4a945",
            ),
            (
                "refused_attribute_before_a_while_before_the_poll",
                with_the_shipped_body(
                    "    [[maybe_unused]] { while (flag != nullptr) { return 0u; } }\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                ),
                "stands before the poll",
                "70464 / 2 / 2b56c510b5f4a945",
            ),
            (
                "refused_attribute_before_a_bare_return_before_the_poll",
                with_the_shipped_body(
                    "    [[maybe_unused]] { return 0u; }\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                ),
                "stands before the poll",
                "70464 / 2 / 2b56c510b5f4a945",
            ),
            (
                "refused_attribute_before_a_block_after_the_poll",
                with_the_shipped_body(
                    "    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    [[maybe_unused]] { { return observed; } }\n    __threadfence();\n    return observed;\n",
                ),
                "stands between the poll and the acquire",
                "70264 / 2 / 42454ad1740a315c",
            ),
            (
                "refused_attribute_before_a_bare_return_after_the_poll",
                with_the_shipped_body(
                    "    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    [[maybe_unused]] { return observed; }\n    __threadfence();\n    return observed;\n",
                ),
                "stands between the poll and the acquire",
                "70264 / 2 / 42454ad1740a315c",
            ),
            // The CONTROL: the same nested block with no attribute. It was refused before round 28 and
            // is still refused - the fix must not turn the attribute case into a general excuse.
            (
                "refused_nested_block_before_the_poll",
                with_the_shipped_body(
                    "    { { return 0u; } }\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                ),
                "stands before the poll",
                "70464 / 2 / 2b56c510b5f4a945",
            ),
            // Round-30 A1: the two spellings round 29 refused here conservatively are ACCEPTED rows
            // now - the walk-back reads the balanced template parameter list and the balanced
            // `decltype(...)` - and the comment that stood here said the scan could not see the
            // lambda, which is exactly what round 30 fixed. See the accepted loop below.
            (
                "refused_poll_in_a_bare_block",
                with_the_shipped_body(
                    "    {\n        const uint32_t observed = spark_wait_for_flag(flag, deadline);\n        __threadfence();\n        return observed;\n    }\n",
                ),
                "not a top-level statement",
                "70432 / 4 / 605321041ac5ec1e",
            ),
            // Round-18 A2/A3, corrected in round 19: the shapes that STAY refused. Two of them are
            // calls whose EVALUATION is conditional; the third, `return poll(...)`, is an
            // UNCONDITIONAL call whose fence is unreachable because the return leaves the helper first,
            // which is why its needle is not the conditional one. Each cubin is measured and each shape
            // has a needle, so its reason is asserted rather than only written.
            (
                "refused_if_branch",
                with_the_shipped_body(
                    "    if (deadline > 0u) spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return 0u;\n",
                ),
                "\"if\" stands before the poll",
                "71096 / 3 / 7ccedc01e1b172e5",
            ),
            (
                "refused_short_circuit",
                with_the_shipped_body(
                    "    const bool ran = (deadline > 0u) && spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return ran ? 1u : 0u;\n",
                ),
                "\"&&\" stands before the poll",
                "70432 / 4 / 28939293aa1b8fc2",
            ),
            (
                "refused_return_call",
                with_the_shipped_body(
                    "    return spark_wait_for_flag(flag, deadline);\n    __threadfence();\n",
                ),
                // Round-20 A6: this shape's reason is that the fence is UNREACHABLE, not that the call
                // is conditional, so the needle carries that word and nothing else does: a mutation that
                // reports it as the conditional case fails this row.
                "the acquire after it is unreachable",
                "70264 / 2 / 42454ad1740a315c",
            ),
            // Round-20 A1: the SAME kind of macro as the accepted macro_operator_before_the_poll, used as
            // the poll's own ARGUMENT. No `;` stands between the use and the observation, so the
            // completed-statement scoping does not excuse it and both checks refuse the text; its cubin is
            // 70432 / 4 / 28939293aa1b8fc2, the same code as the literal short circuit it expands to.
            (
                "refused_operator_macro_wrapping_the_poll",
                format!(
                    "#define MAYBE(x) ((deadline > 0u) && (x))\n{}",
                    with_the_shipped_body(
                        "    const uint32_t observed = MAYBE(spark_wait_for_flag(flag, deadline));\n    __threadfence();\n    return observed;\n",
                    )
                ),
                "\"MAYBE\" stands before the poll",
                "70432 / 4 / 28939293aa1b8fc2",
            ),
            // Round-20 A1/A3: the two texts where the macro reading and the call reading collide, both
            // refused by BOTH checks. `BAIL;` before the poll is a use of a macro that can bound the
            // path and its cubin is NOT the baseline (68576 / 2 / bfd9e8e177f4f76d), so the refusal is required;
            // the parameter-swapping alias IS the baseline cubin, and the scan refuses it anyway because
            // it reads two mentions and will not guess which is the observation (documented as a
            // conservative refusal, not as a defect).
            (
                "refused_suspicious_macro_before_poll",
                format!(
                    "#define BAIL return\n{}",
                    with_the_shipped_body(
                        "    BAIL;\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                    )
                ),
                "is a macro whose body contains a control keyword",
                "68576 / 2 / bfd9e8e177f4f76d",
            ),
            (
                "refused_macro_parameter_alias",
                format!(
                    "#define P(spark_wait_for_flag) spark_wait_for_flag\n{}",
                    with_the_shipped_body(
                        "    const uint32_t observed = P(spark_wait_for_flag)(flag, deadline);\n    __threadfence();\n    return observed;\n",
                    )
                ),
                "2 times",
                "70432 / 4 / 605321041ac5ec1e",
            ),
            // Round-21 A1: the SAME macro as the text above, used AFTER the poll in a later, completed
            // statement. The lint refused it and the structural check accepted it (the mirror image of
            // round 20) although its cubin is NOT the baseline - 69072 / 2 / 18f93bdc42897348, measured, because
            // the fence after the macro is unreachable - so the lint was right. The scope is now the
            // lint's own, every use before the fence, and the shared reason is the lint's forward-walk
            // sentence, which this row's needle names.
            (
                "refused_suspicious_macro_after_the_poll",
                format!(
                    "#define BAIL return\n{}",
                    with_the_shipped_body(
                        "    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    BAIL;\n    __threadfence();\n    return observed;\n",
                    )
                ),
                "the fence runs after it",
                "69072 / 2 / 18f93bdc42897348",
            ),
            // Round-19 A7: the two-poll body as a TRANSLATION UNIT. The round-13 fixture duplicated one
            // const and nvcc refuses that, so the number the documentation quoted sat with a text that
            // does not compile; this one does, and 70728 / 3 / d5a856a8953895d6 is its own measurement.
            (
                "refused_two_polls",
                with_the_shipped_body(
                    "    const uint32_t first = spark_wait_for_flag(flag, deadline);\n    const uint32_t second = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return first + second;\n",
                ),
                "2 times",
                "70728 / 3 / d5a856a8953895d6",
            ),
            // Round-23 A1 (HIGH): the fence sits in the poll's ARGUMENTS, where it runs BEFORE the call,
            // and there is no acquire after the call. The structural check read the in-arguments fence as
            // the acquire of the next mention and ACCEPTED the body while the lint refused it - and the
            // body compiles to 70280 / 3 / 18b66cc5f312046c, the cubin of the plain fence-before-the-poll body
            // both checks refuse, so the verdict was not a function of the compiled shape. Both refuse it
            // now, and in the same words.
            (
                "refused_fence_in_the_poll_arguments",
                with_the_shipped_body(
                    "    const uint32_t observed = spark_wait_for_flag(({ __threadfence(); flag; }), deadline);\n    return observed;\n",
                ),
                "no reachable __threadfence()",
                "70280 / 3 / 18b66cc5f312046c",
            ),
            // Round-23 A3 (HIGH): a poll inside a conditional the guard cannot resolve, plus a second,
            // unconditional poll and the fence. Under -DM4_FLAG=1 the text compiles to TWO real polls
            // (39072 / 95e0cf41 (pre-change), measured with that define) whose single acquire pairs with the second
            // only, so the first observation is read with no acquire after it; without the define it is
            // the baseline below. The structural check refused the body and the lint accepted it; both
            // refuse it now, and with the same sentence.
            (
                "refused_conditional_poll_with_a_second_poll",
                with_the_shipped_body(
                    "    #if M4_FLAG\n    spark_wait_for_flag(flag, deadline);\n    #endif\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                ),
                "sits inside a preprocessor conditional",
                "70432 / 4 / 605321041ac5ec1e",
            ),
            // Round-24 A1 (HIGH): control flow inside the poll's OWN argument list, which the round-23
            // span skip stopped the scan from reading, so BOTH checks accepted it although the compiled
            // acquire is absent. `(__trap(), flag)` as the first argument is 69536 / 4 / 5826994128b34ed0 -
            // byte-identical to the trap-before-the-poll statement body both checks refuse - and
            // `({ return 0u; flag; })` there is 70464 / 2 / 2b56c510b5f4a945.
            (
                "refused_trap_in_the_poll_arguments",
                with_the_shipped_body(
                    "    const uint32_t observed = spark_wait_for_flag((__trap(), flag), deadline);\n    __threadfence();\n    return observed;\n",
                ),
                "does not return and stands between the poll and the acquire",
                "69536 / 4 / 5826994128b34ed0",
            ),
            (
                "refused_return_in_the_poll_arguments",
                with_the_shipped_body(
                    "    const uint32_t observed = spark_wait_for_flag(({ return 0u; flag; }), deadline);\n    __threadfence();\n    return observed;\n",
                ),
                "\"return\" stands between the poll and the acquire",
                "70464 / 2 / 2b56c510b5f4a945",
            ),
            // The cubin the documentation already carries for a return before the poll: the arguments
            // are not a different path, which is exactly why reading them matters.
            (
                "refused_unreachable_in_the_poll_arguments",
                with_the_shipped_body(
                    "    const uint32_t observed = spark_wait_for_flag((__builtin_unreachable(), flag), deadline);\n    __threadfence();\n    return observed;\n",
                ),
                "does not return and stands between the poll and the acquire",
                "68576 / 2 / bfd9e8e177f4f76d",
            ),
            // Round-26 A3: the documented CONSERVATIVE refusal. The text compiles to the BASELINE - the
            // macro's replacement is a lambda holding a `return` - and the refusal is only that the
            // replacement list is read textually with no lambda skip. Its own cubin is recorded with it
            // so the conservatism is visible in the store rather than argued.
            (
                "refused_macro_lambda_in_the_arguments",
                with_the_shipped_body(
                    "    #define LL ([&]{ return flag; })\n    const uint32_t observed = spark_wait_for_flag((LL)(), deadline);\n    __threadfence();\n    return observed;\n",
                ),
                "a macro whose body contains a control keyword",
                "70432 / 4 / 605321041ac5ec1e",
            ),
            // Round-32 A3: the FALSE ACCEPT the second closing review found. The lambda is CALLED
            // where it stands, so the trap inside it ends the helper and nvcc compiles this text to
            // 69536 / 4 / 5826994128b34ed0 (37872 / 4 / 54cda5bc pre-change) - the cubin of
            // `__trap();` before the poll, which both checks
            // already refuse, and the refusal here is about the compiled code, so the CALLED
            // lambda body is not a body the helper does not own. The never-called twin
            // (`accepted_nonreturning_call_in_a_lambda_before_the_poll`) stays accepted: that body
            // never runs, which is what its baseline cubin measures.
            (
                "refused_trap_in_a_called_lambda_before_the_poll",
                with_the_shipped_body(
                    "    (void)([&]{ __trap(); return 0u; })();\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
                ),
                "does not return",
                "69536 / 4 / 5826994128b34ed0",
            ),
        ] {
            texts.push((name, source, Verdict::Refused, cubin, needle));
        }
        texts
    }
    /// The ACCEPTED list: every shape in which the helper's body CALLS the raw poll, each one a
    /// translation unit whose cubin was measured in the CUDA 13.0.3 container (sm_121a, the mega_mhc
    /// kernel, exactly as the committed baseline is). The scan certifies that the designator is called:
    /// it does not read the arguments, it does not read the expression the call stands in, and it does
    /// not care how the designator itself is spelled - so these shapes differ in what they COST, not in
    /// their verdict. Refusing one is a false rejection.
    ///
    /// The table lives in \`measured_texts\`; the names below are its names, and the numbers are its
    /// cubins (full shas are in the module documentation):
    ///
    ///   * \`shipped\`, the shared header unchanged - 70432 / 4 / 605321041ac5ec1e, the baseline;
    ///   * \`bare_call\` - 70960 / 3 / bbafa1b88b3c3be7; \`void_cast\` - 70960 / 3 / bbafa1b88b3c3be7; \`gt\` - 70960 / 3 /
    ///     bbafa1b8; \`attribute_stmt\` - 70960 / 3 / bbafa1b88b3c3be7;
    ///   * \`unary\`, \`eq\` - both 70432 / 4 / f715d21b4aef0271;
    ///   * \`lt\`, \`chained\` - both 70432 / 3 / 439d36ffff9ee9d5;
    ///   * \`cast_seed\`, \`attribute_decl\`, \`spaced\`, \`paren_args\`, \`cast_arg\`, \`renamed\`, \`alias\` -
    ///     all 70432 / 4 / 605321041ac5ec1e;
    ///   * round-18 A1: \`paren_designator\` (\`(poll)(flag, deadline)\`), \`deref_designator\`,
    ///     \`addr_designator\` - all 70432 / 4 / 605321041ac5ec1e, i.e. the designator spelling is not part of the
    ///     rule either, because the call is the same call;
    ///   * round-18 A2/A3: \`ternary\` (\`true ? poll(...) : 0u\`) - 70432 / 4 / 605321041ac5ec1e;
    ///     \`if_condition\` - 70960 / 3 / bbafa1b88b3c3be7; \`while_condition\` - 71088 / 3 / a9ef975cf8356c47;
    ///   * round-18 A4: \`paren_less_than\` and \`bare_less_than\` - both 70432 / 4 / f715d21b4aef0271.
    ///
    /// The renamed-parameter call and the macro alias of the designator also have their own focused test,
    /// \`a_call_under_renamed_parameters_or_through_an_alias_is_a_call\`.
    #[test]
    fn the_helper_body_accepts_every_call_of_the_poll() {
        let mut measured = 0;
        for (name, source, verdict, cubin, _) in measured_texts() {
            if verdict != Verdict::Accepted {
                continue;
            }
            measured += 1;
            let structural = acquire_helper_fences_unconditionally(&source);
            assert!(
                structural.is_ok(),
                "the structural check reads {name} as a call of the poll (measured {cubin}): \
                 {structural:?}"
            );
            assert_eq!(
                acquire_fence_after_poll(&source, HELPER_POLL),
                Ok(()),
                "the lint reads {name} as a call of the poll (measured {cubin})"
            );
        }
        assert_eq!(measured, 89, "every accepted shape is exercised");
    }

    /// Round-17 A2/A4, both accepted and both measured at the baseline (70432 / 4 / 605321041ac5ec1e):
    /// the scan does not read the arguments, so a helper whose parameters are named `f` and `d` is not
    /// asked to write `flag` and `deadline` (the shipped spelling in the messages is an illustration,
    /// not a substitution - the hint never tells a writer to use a name that does not exist), and a
    /// macro whose replacement list is exactly the designator IS the designator, one level deep.
    #[test]
    fn a_call_under_renamed_parameters_or_through_an_alias_is_a_call() {
        for name in ["renamed", "alias"] {
            let (_, source, verdict, cubin, _) = measured_texts()
                .into_iter()
                .find(|(entry, _, _, _, _)| *entry == name)
                .expect("the text is measured");
            assert_ne!(
                source, COMMON_HEADER,
                "{name} must differ from the shipped header"
            );
            assert_eq!(verdict, Verdict::Accepted, "{name} is accepted");
            let structural = acquire_helper_fences_unconditionally(&source);
            assert!(
                structural.is_ok(),
                "the structural check reads {name} as a call of the poll (measured {cubin}): \
                 {structural:?}"
            );
            assert_eq!(
                acquire_fence_after_poll(&source, HELPER_POLL),
                Ok(()),
                "the lint reads {name} as a call of the poll (measured {cubin})"
            );
        }
    }

    /// Round-17 A3: a comparison in front of the poll name is a comparison, not a template close, and
    /// the verdict may not depend on what stands earlier in the file. Both forms below are accepted by
    /// both checks: the shipped header (whose own `#include <...>` lines put an earlier `<` and an
    /// earlier `>` in the file) with the observation written `0u > poll(...)`, and the same body with a
    /// comparison added in front of it. The measured cubin of the header form is 70960 / 3 /
    /// bbafa1b88b3c3be7; the form without includes is pinned as SCAN TEXT only, because that text cannot
    /// be a translation unit (it cannot declare the types the header needs without including them, and an
    /// include is itself an earlier `<`) - which is exactly why the bound is on the statement.
    #[test]
    fn a_comparison_stays_a_call_whatever_came_before_it() {
        let body = "    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    \
                    __threadfence();\n    return observed;\n";
        let comparison = "    const bool observed = 0u > spark_wait_for_flag(flag, deadline);\n    \
                          __threadfence();\n    return observed ? 1u : 0u;\n";
        let in_the_header = COMMON_HEADER.replace(body, comparison);
        assert_ne!(in_the_header, COMMON_HEADER, "the comparison must apply");
        assert!(
            acquire_helper_fences_unconditionally(&in_the_header).is_ok(),
            "the header's own includes put earlier '<' in the file, and the '>' is still a comparison"
        );
        assert_eq!(
            acquire_fence_after_poll(&in_the_header, HELPER_POLL),
            Ok(()),
            "the lint agrees (measured 70960 / 3 / bbafa1b88b3c3be7)"
        );
        let with_an_earlier_comparison =
            format!("__device__ bool round17_prelude() {{ return 1 < 2; }}\n{in_the_header}");
        assert!(
            acquire_helper_fences_unconditionally(&with_an_earlier_comparison).is_ok(),
            "an earlier comparison in the same file does not change the verdict"
        );
        assert_eq!(
            acquire_fence_after_poll(&with_an_earlier_comparison, HELPER_POLL),
            Ok(()),
            "the two forms must agree"
        );
    }

    /// The REFUSED list (round-16 A1/A2/A3): a mention of the poll inside the helper's body that is not
    /// a call of it - a declaration in any return-type spelling, a function-pointer declarator, an
    /// attribute between the type and the name. Each is refused by both checks with its own message and
    /// the hint, and the cubin each one compiles to is what says the refusal is right rather than
    /// pedantic: a helper that only DECLARES the poll observes nothing, and the code it compiles to is
    /// not the baseline.
    ///
    ///   * A1, a function-pointer declarator - `uint32_t (*spark_wait_for_flag)(const volatile
    ///     uint32_t *, uint64_t);` in the body: nvcc compiles this text to 70600 bytes / 3 PTB membar
    ///     lines / sha 1f2503fd7ecaf189, against the 70432 / 4 / 605321041ac5ec1e baseline (measured);
    ///   * A2, an attribute between the type and the name - `uint32_t [[maybe_unused]]
    ///     spark_wait_for_flag(const volatile uint32_t *, uint64_t);`: same 70600 / 3 /
    ///     1f2503fd7ecaf189 (measured);
    ///   * A3, a macro-defined return type - `#define SPARK_M4_RETURN decltype(true ? 0u : 1u)` and
    ///     the same declaration through it: same 70600 / 3 / 1f2503fd7ecaf189 (measured).
    ///
    /// The fourth shape round 16 refused, a real call wrapped in a chained comparison, is a CALL and is
    /// accepted now (round-17 A1): it is pinned with its measured cubin in
    /// `the_helper_body_accepts_every_call_of_the_poll`, because refusing a call is a false rejection.
    ///
    /// The three numbers below are byte for byte the ones the round-16 review reported, which is what
    /// identifies those shapes.
    #[test]
    fn a_declaration_in_the_helper_body_is_not_an_observation() {
        let mut measured = 0;
        for (name, source, verdict, cubin, needle) in measured_texts() {
            if verdict != Verdict::Refused {
                continue;
            }
            measured += 1;
            for (check, error) in [
                (
                    "structural",
                    acquire_helper_fences_unconditionally(&source)
                        .expect_err("the shape is refused as one the scan will not certify"),
                ),
                (
                    "lint",
                    acquire_fence_after_poll(&source, HELPER_POLL)
                        .expect_err("the shape is refused as one the scan will not certify"),
                ),
            ] {
                assert!(
                    !needle.is_empty(),
                    "every refused text carries the needle that names its own reason: {name}"
                );
                assert!(
                    error.contains(needle) && error.ends_with(CANONICAL_HINT),
                    "the {check} check refuses {name} (measured {cubin}), says why and ends with the \
                     hint: {error}"
                );
            }
        }
        assert_eq!(measured, 27, "every refused text is exercised");
    }

    /// Round-33 A1: the round-32 rule counted a non-returning call only at the called lambda's own
    /// level, so a trap one brace deeper - inside a BARE BLOCK, which runs where it stands - was still
    /// accepted. The text below is the same program as the refused unbraced spelling, so a guard that
    /// accepts it is certifying two verdicts for one cubin. None of these texts is measured in the
    /// container and no cubin is claimed for any of them: what is pinned is the verdict, and the
    /// measured text that carries the cubin is `refused_trap_in_a_called_lambda_before_the_poll`.
    #[test]
    fn a_trap_in_a_bare_block_inside_a_called_lambda_is_refused() {
        let refused = [
            (
                "one block deep",
                "    (void)([&]{ { __trap(); } return 0u; })();\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
            ),
            (
                "two blocks deep",
                "    (void)([&]{ { { __trap(); } } return 0u; })();\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
            ),
            (
                "with an attribute on the bare block itself",
                "    (void)([&]{ [[likely]] { __trap(); } return 0u; })();
    const uint32_t observed = spark_wait_for_flag(flag, deadline);
    __threadfence();
    return observed;
",
            ),
            (
                "after another statement",
                "    (void)([&]{ flag; { __trap(); } return 0u; })();\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
            ),
        ];
        for (shape, body) in refused {
            let source = with_the_shipped_body(body);
            for (check, result) in [
                ("structural", acquire_helper_fences_unconditionally(&source)),
                ("lint", acquire_fence_after_poll(&source, HELPER_POLL)),
            ] {
                let error = result.expect_err(&format!(
                    "the {check} check must refuse a trap in a bare block inside a called lambda ({shape})"
                ));
                assert!(
                    error.contains("does not return") && error.ends_with(CANONICAL_HINT),
                    "the {check} check refuses the {shape} shape, says why and ends with the hint: {error}"
                );
            }
        }
        // The direction that must not change, which is the worse error of the two: a trap the
        // statement makes conditional, and a trap in a lambda nobody calls, both stay accepted.
        let accepted = [
            (
                "behind a condition",
                "    (void)([&]{ if (deadline) { __trap(); } return 0u; })();\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
            ),
            (
                "in a lambda nobody calls",
                "    auto f = [&]{ { __trap(); } };\n    (void)f;\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
            ),
            // A control statement's body is not descended into even when the statement's own
            // condition is a constant, because the rule reads braces and not values. A `switch` whose
            // only label is a `default` that falls into the trap is the sharpest case: it always runs
            // the trap, and this scan accepts it. That is the stated limit of a text-level rule, and
            // it is the direction this project chooses - folding `switch` or a constant `if` needs the
            // condition's value, and a wrong fold would refuse a text nvcc compiles to the baseline.
            // A control statement's body stays a control statement's body when an ATTRIBUTE stands
            // between the condition and the brace, which C++ allows on a selection or iteration
            // statement (round-34 A1: an attribute is not a bare block, and reading it as one refused
            // a text nvcc compiles to the baseline).
            (
                "behind a condition carrying an attribute",
                "    (void)([&]{ if (deadline) [[likely]] { __trap(); } return 0u; })();
    const uint32_t observed = spark_wait_for_flag(flag, deadline);
    __threadfence();
    return observed;
",
            ),
            (
                "behind an else carrying an attribute",
                "    (void)([&]{ if (deadline) { flag; } else [[likely]] { __trap(); } return 0u; })();
    const uint32_t observed = spark_wait_for_flag(flag, deadline);
    __threadfence();
    return observed;
",
            ),
            (
                "in a for body carrying an attribute",
                "    (void)([&]{ for (;;) [[unlikely]] { __trap(); } return 0u; })();
    const uint32_t observed = spark_wait_for_flag(flag, deadline);
    __threadfence();
    return observed;
",
            ),
            (
                "behind a switch carrying an attribute",
                "    (void)([&]{ switch (flag) [[likely]] { default: __trap(); } return 0u; })();
    const uint32_t observed = spark_wait_for_flag(flag, deadline);
    __threadfence();
    return observed;
",
            ),
            // Two attributes in front of the same body: the loop steps over EVERY attribute, and a
            // version that steps over only one false-refuses this text, which g++ and clang both
            // accept at -std=c++17 and c++20 (round-35 A2).
            (
                "behind a condition carrying two attributes",
                "    (void)([&]{ if (deadline) [[likely]] [[clang::likely]] { __trap(); } return 0u; })();
    const uint32_t observed = spark_wait_for_flag(flag, deadline);
    __threadfence();
    return observed;
",
            ),
            (
                "behind a constant switch",
                "    (void)([&]{ { switch (flag) { default: __trap(); } } return 0u; })();\n    const uint32_t observed = spark_wait_for_flag(flag, deadline);\n    __threadfence();\n    return observed;\n",
            ),
        ];
        for (shape, body) in accepted {
            let source = with_the_shipped_body(body);
            assert!(
                acquire_helper_fences_unconditionally(&source).is_ok(),
                "the structural check still accepts a trap {shape}"
            );
            assert!(
                acquire_fence_after_poll(&source, HELPER_POLL).is_ok(),
                "the lint still accepts a trap {shape}"
            );
        }
    }

    /// Round-35 A3: an attribute in front of the acquire block does not stop the block being bare.
    /// An attribute is only a hint, so the attributed block runs its fence exactly where the plain
    /// block does; the structural scan once took the attribute's closing bracket for a lambda capture
    /// list and refused it, which is the compile-to-the-baseline direction this project treats as the
    /// worse error. Both checks are pinned here, for one, two and no attributes.
    #[test]
    fn an_attribute_before_the_fence_block_does_not_make_it_conditional() {
        for (shape, block) in [
            ("without an attribute", "{ __threadfence(); }"),
            ("with an attribute", "[[likely]] { __threadfence(); }"),
            (
                "with two attributes",
                "[[likely]] [[clang::likely]] { __threadfence(); }",
            ),
        ] {
            let source = with_the_shipped_body(&format!(
                "    const uint32_t observed = spark_wait_for_flag(flag, deadline);
    {block}
    return observed;
"
            ));
            assert!(
                acquire_helper_fences_unconditionally(&source).is_ok(),
                "the structural check accepts the fence block {shape}"
            );
            assert!(
                acquire_fence_after_poll(&source, HELPER_POLL).is_ok(),
                "the lint accepts the fence block {shape}"
            );
        }
    }

    /// This file, as the text the documentation check below reads: the module's own documentation is
    /// part of the crate, so the shape clippy refuses can be checked here.
    const THIS_SOURCE: &str = include_str!("m4_abi.rs");

    /// The directory the measured texts are stored in. It lives outside the tree - the texts are
    /// evidence rather than fixtures - so the byte-identity check below is HOST-ONLY: it runs where the
    /// store is, which is the developer tree, and in CI the test says so and moves on (round-20 test
    /// quality: the store cannot be in CI, so the documentation says which half CI checks). What CI
    /// covers instead is every check that does not need the store: the crate and workspace suites, the
    /// feature sweep with its own tests, and the six-cubin baseline gate of the `cuda-abi` job, each of
    /// which rebuilds these texts from this module's builders or from the committed baseline. Only the
    /// comparison against the STORED bytes is host-only.
    const EVIDENCE_TEXTS: &str = "/home/user/wt/targets/kd17_texts";

    /// The directory the SAME texts were MEASURED in, written by the evidence generator
    /// (`kd18_scripts/gen.py`) rather than by this builder. Round-19 A8: the two stores are checked
    /// against each other, so a text that was edited in one place and not the other cannot sit beside a
    /// cubin that was measured on different bytes.
    const MEASURED_TEXTS: &str = "/home/user/wt/targets/kd18_texts";

    /// Round-18 C, widened in round 19: every text this module quotes a cubin for is stored, and each
    /// file in EACH store is byte-identical to the text the tests build. A number can therefore always
    /// be traced to a text: the name in the documentation is the file name, the file is the builder's
    /// own bytes, and the cubin beside it is the one measured on those bytes in the CUDA 13.0.3
    /// container. The generator's store is the one the container read; the builder's store is the one
    /// the tests build. Checking both is what keeps them from drifting apart.
    #[test]
    fn the_stored_measured_texts_are_the_texts_the_tests_build() {
        let dir =
            std::env::var("SPARKFLASH_KERNEL_TEXTS").unwrap_or_else(|_| EVIDENCE_TEXTS.to_string());
        if !std::path::Path::new(&dir).is_dir() {
            eprintln!("no evidence directory at {dir}: the stored-text check is skipped");
            return;
        }
        let measured = measured_texts();
        let mut built: Vec<String> = measured
            .iter()
            .map(|(name, _, _, _, _)| format!("{name}.cuh"))
            .collect();
        built.sort();
        let mut directories = vec![dir.clone()];
        if std::path::Path::new(MEASURED_TEXTS).is_dir() {
            directories.push(MEASURED_TEXTS.to_string());
        } else {
            eprintln!(
                "no generator store at {MEASURED_TEXTS}: it is checked only when it is there"
            );
        }
        for directory in &directories {
            for (name, source, _, cubin, _) in &measured {
                let path = format!("{directory}/{name}.cuh");
                let stored = std::fs::read_to_string(&path).unwrap_or_else(|error| {
                    panic!("{path} must exist (measured {cubin}): {error}")
                });
                assert_eq!(
                    stored, *source,
                    "{name} in {directory} must be the text the tests build (measured {cubin})"
                );
            }
            // And nothing else is stored: a text no test builds could not be traced to anything.
            let mut stored: Vec<String> = std::fs::read_dir(directory)
                .unwrap_or_else(|error| panic!("{directory} must be readable: {error}"))
                .map(|entry| {
                    entry
                        .expect("a directory entry")
                        .file_name()
                        .to_string_lossy()
                        .into_owned()
                })
                .filter(|name| name.ends_with(".cuh"))
                .collect();
            stored.sort();
            assert_eq!(
                stored, built,
                "{directory} holds exactly the measured texts"
            );
        }
    }

    /// Writes the measured texts out again from the builder, so the stored evidence can be regenerated
    /// from the tree instead of being kept in step by hand. Run it with SPARKFLASH_KERNEL_TEXTS_DUMP
    /// set to the directory; without that variable it does nothing.
    #[test]
    fn dump_measured_texts() {
        let Ok(dir) = std::env::var("SPARKFLASH_KERNEL_TEXTS_DUMP") else {
            return;
        };
        std::fs::create_dir_all(&dir).expect("the dump directory");
        for (name, source, _, _, _) in measured_texts() {
            std::fs::write(format!("{dir}/{name}.cuh"), source).expect("the dump");
        }
    }

    /// Round-18 A7: the shape clippy refuses under -D warnings (clippy::doc_lazy_continuation - a list
    /// item followed by a non-blank, unindented doc line is a lazy continuation of that item). The CI
    /// failure this comes from was fixed by hand and "checked" by a scan that lived outside the tree;
    /// this is that scan, in the tree, so the shape fails the crate's own tests instead of CI.
    #[test]
    fn the_documentation_blocks_have_no_lazy_continuation() {
        let mut in_a_list = false;
        for (number, line) in THIS_SOURCE.lines().enumerate() {
            let trimmed = line.trim_start();
            let Some(rest) = trimmed
                .strip_prefix("///")
                .or_else(|| trimmed.strip_prefix("//!"))
            else {
                in_a_list = false;
                continue;
            };
            let body = rest.strip_prefix(' ').unwrap_or(rest);
            if body.trim().is_empty() {
                in_a_list = false;
            } else if body.trim_start().starts_with("* ") || body.trim_start().starts_with("- ") {
                // A list item. The bullet is indented in this crate's documentation, so the leading
                // whitespace has to be skipped before the bullet is looked for - a scanner that only
                // matched a bullet in column zero was vacuous for every list in this file. Round-20
                // quality: a DASH bullet is a list item to clippy too, and reading only "* " missed the
                // one line in this file that clippy refused for exactly this shape at 3502. Round-21
                // quality: that line is reworded rather than repunctuated, so the dash branch would be
                // dead code against this file if nothing used a dash - and the round-21 review said so.
                // The shape list in `the_two_checks_agree_on_every_shape_of_macro_use` is written with
                // dash bullets, which is where the branch is live and where a lazy continuation after a
                // dash bullet would be caught.
                in_a_list = true;
            } else if body.starts_with("    ") {
                // The list's own continuation indent.
            } else if in_a_list {
                panic!(
                    "line {}: a list item is followed by an unindented doc line, which clippy reads \
                     as a lazy continuation of that item: {line}",
                    number + 1
                );
            }
        }
    }

    /// Round 13, A-3: the three using-declaration spellings that the classifier used to lump in with
    /// the bare form, each refused with the reason that matches its own text and its own nvcc
    /// diagnostic rather than the sentence about the bare spelling.
    #[test]
    fn the_using_spellings_are_refused_with_their_own_reason() {
        for (label, replacement, needle) in [
            (
                "template-qualified",
                "    using X::template __threadfence;\n    __threadfence();\n    return observed;",
                "names a function template",
            ),
            (
                "template-id",
                "    using X::__threadfence<int>;\n    __threadfence();\n    return observed;",
                "is a template-id",
            ),
            (
                "alias-declaration",
                "    using __threadfence = Y;\n    __threadfence();\n    return observed;",
                "alias-declaration that re-binds",
            ),
        ] {
            let source = COMMON_HEADER.replace(HELPER_ANCHOR, replacement);
            assert_ne!(source, COMMON_HEADER, "{label} must apply to the header");
            for (check, error) in [
                (
                    "structural",
                    acquire_helper_fences_unconditionally(&source)
                        .expect_err("A3: the spelling is refused"),
                ),
                (
                    "lint",
                    acquire_fence_after_poll(&source, HELPER_POLL)
                        .expect_err("A3: the spelling is refused"),
                ),
            ] {
                assert!(
                    error.contains(needle) && error.ends_with(CANONICAL_HINT),
                    "A3 {label} {check}: the reason names the spelling and ends with the hint: {error}"
                );
            }
        }
    }
}
