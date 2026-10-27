// Copyright (c) 2026 Hunter L. Allen
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
    // The hand-rolled acquire is gone; the file carries scalar and K8 paths, so check each call.
    let mut calls = MEGA_MHC.match_indices(&call).peekable();
    assert!(calls.peek().is_some(), "the call exists");
    for (observing, _) in calls {
        let read = MEGA_MHC[observing..]
            .find("*hc_rms_out")
            .map(|offset| observing + offset);
        let read = read.expect("the observing path reads hc_rms_out after the poll");
        assert!(observing < read, "the helper call must precede the read");
        assert!(
            !MEGA_MHC[observing..read].contains("__threadfence"),
            "the acquire must not be duplicated at the call site"
        );
    }
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
    let error =
        acquire_fence_after_poll(&spliced_return, POLL).expect_err("a spliced return is a return");
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
    let raw_fence =
        "spark_wait_for_flag(&flag, deadline);\nconst char *s = R\"x(;\" ;__threadfence(); )x\";\n";
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
    let error = observing_paths_call_the_acquiring_poll(&kernels_as_refs(&kernel_units(&extended)))
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
    assert!(acquire_helper_fences_unconditionally(&insert_after_poll("    return 0u;")).is_err());
    // A bare block that holds ONLY the fence has no blocker in it, which is why (A3) below can
    // accept it without weakening this rule.
    assert_eq!(
        acquire_helper_fences_unconditionally(&insert_after_poll("    { __threadfence(); }")),
        Ok(())
    );

    // (A2) the dereferenced and address-of designators are the same call.
    for spelling in ["(*__threadfence)();", "(&__threadfence)();"] {
        let mutation = COMMON_HEADER.replace("    __threadfence();", &format!("    {spelling}"));
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
                && error.contains(
                    "a branch, a loop, a try block, a requires-expression, a lambda or a type body"
                ),
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
        let mutation = COMMON_HEADER.replace("    __threadfence();", &format!("    {spelling}"));
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
    let error = the_acquiring_poll_is_defined_exactly_once(&[("mega_mhc_sm121.cu", &declaration)])
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
        let error = observing_paths_call_the_acquiring_poll(&[("mega_mhc_sm121.cu", &mutation)])
            .expect_err("(4) the parenthesised or dereferenced designator is a raw poll call");
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
    let mutation = COMMON_HEADER.replace(poll_statement, &format!("{poll_statement}\n{spliced}"));
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
        observing_paths_call_the_acquiring_poll(&[("aliased_sm121.cu", body_with_poll)]).is_err(),
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
    let spliced = format!("// filler \\\n more filler\n// third\n#define {RAW_POLL}(f, d) (0u)\n");
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
    let error =
        acquire_helper_fences_unconditionally(&second_poll).expect_err("two polls are rejected");
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
        let source = COMMON_HEADER.replace(HELPER_ANCHOR, &format!("{body}\n    return observed;"));
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
        let source = COMMON_HEADER.replace(HELPER_ANCHOR, &format!("{body}\n    return observed;"));
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
        let source = COMMON_HEADER.replace(HELPER_ANCHOR, &format!("{body}\n    return observed;"));
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
            format!("uint32_t f() {{\n    __threadfence();\n    {poll}\n    return observed;\n}}"),
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
                structural.contains(CANONICAL_POLL_CALL) && structural.ends_with(CANONICAL_HINT),
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
    let source = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/m4_abi/scanner.rs"
    ))
    .expect("the scanner's own source is readable");
    // Round-15 A-5: the region split is a text search, and this file is checked out on platforms
    // whose line endings are CRLF, so the newlines are normalised first.
    let source = source.replace("\r\n", "\n");
    // The guard's own code: from the hint constant that every rejection ends with to the end of the
    // scanner module. The scanner module holds the guard's code and nothing else after the hint.
    let code = source
        .split_once("const CANONICAL_HINT: &str")
        .expect("the guard defines the canonical hint")
        .1;
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
        structural.contains("not a top-level statement") && structural.ends_with(CANONICAL_HINT),
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
        structural.contains("no reachable __threadfence() follows the poll of spark_wait_for_flag")
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
            acquire_helper_fences_unconditionally(&poll_inside_a_returning_macro)
                .expect_err("a poll expanded inside a returning macro is not an observing path"),
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
    let conditional_fence =
        format!("{poll_line}\n#if M4_FLAG\n    __threadfence();\n#endif\n    return observed;\n");
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
    let source = THIS_SOURCE;
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
                &format!("{poll_line}\n    __threadfence();\n    BAIL;\n    return observed;\n"),
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
                &format!("{poll_line}\n    OUTER;\n    __threadfence();\n    return observed;\n"),
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
    for (index, (label, source, refuses_with_the_macro_rule)) in shapes.into_iter().enumerate() {
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
            let shared = if structural.contains("a macro whose body contains a control keyword") {
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

/// The whole `m4_abi` module family - this module, its scanner and the root file - as the text
/// the documentation checks below read: the module's own documentation is part of the crate, so
/// the shape clippy refuses can be checked here.
const THIS_SOURCE: &str = concat!(
    include_str!("../m4_abi.rs"),
    include_str!("scanner.rs"),
    include_str!("acquire_guard.rs"),
);

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
        eprintln!("no generator store at {MEASURED_TEXTS}: it is checked only when it is there");
    }
    for directory in &directories {
        for (name, source, _, cubin, _) in &measured {
            let path = format!("{directory}/{name}.cuh");
            let stored = std::fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("{path} must exist (measured {cubin}): {error}"));
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
