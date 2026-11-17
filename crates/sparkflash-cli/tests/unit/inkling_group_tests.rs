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

//! The device-arm group's flags, parsed through the real [ServeOptions].
//!
//! This lives outside `src` because the CLI's source-line ratchet counts every file under it
//! (`sparkflash-ops/tests/one_stack.rs`); the module is still a child of `serve`, so it sees the
//! private parser and the constants the flags are checked against.

use super::*;

fn checkpoint() -> tempfile::TempDir {
    let directory = tempfile::tempdir().expect("a temporary checkpoint");
    std::fs::write(
        directory.path().join(CHECKPOINT_CONFIG_NAME),
        sparkflash_model_inkling2::carried_config(),
    )
    .expect("the released config is written");
    std::fs::write(
        directory.path().join("hf_quant_config.json"),
        sparkflash_model_inkling2::carried_quant(),
    )
    .expect("the quant groups are written");
    directory
}

fn group_argv(checkpoint: &str, extra: &[&str]) -> Vec<String> {
    let mut argv = vec![
        "--model".to_owned(),
        "inkling2".to_owned(),
        "--checkpoint".to_owned(),
        checkpoint.to_owned(),
        "--rank".to_owned(),
        "0".to_owned(),
        "--peer".to_owned(),
        "127.0.0.1:29600,127.0.0.1:29601".to_owned(),
        "--converted".to_owned(),
        checkpoint.to_owned(),
    ];
    argv.extend(extra.iter().map(|flag| (*flag).to_owned()));
    argv
}

#[test]
fn the_group_flags_parse_and_refuse_the_knobs_they_cannot_honour() {
    let directory = checkpoint();
    let checkpoint = directory.path().display().to_string();
    let options = ServeOptions::parse(group_argv(&checkpoint, &[])).expect("the group parses");
    let group = options.group.expect("a group");
    assert_eq!(group.rank, 0);
    assert_eq!(group.world, 2);
    assert_eq!(group.control.port(), GROUP_DEFAULT_CONTROL_PORT);

    assert_eq!(
        group.lanes, 1,
        "a group without --fuse-batch is the serial mirror"
    );

    let error = ServeOptions::parse(group_argv(&checkpoint, &["--concurrency", "2"]))
        .expect_err("a group cannot run two interleaved sequences");
    assert!(error.contains("--concurrency"), "{error}");

    // Fused lanes: one a concurrent request, stepped together on every rank.
    let fused = ServeOptions::parse(group_argv(
        &checkpoint,
        &["--concurrency", "4", "--fuse-batch"],
    ))
    .expect("a group fuses its lanes");
    assert_eq!(fused.group.expect("a group").lanes, 4);

    let error = ServeOptions::parse(group_argv(&checkpoint, &["--fuse-batch"]))
        .expect_err("one lane is not a fused batch");
    assert!(error.contains("--concurrency of at least 2"), "{error}");

    let error = ServeOptions::parse(group_argv(&checkpoint, &["--continuous-batching"]))
        .expect_err("a group cannot batch a second request into its lockstep");
    assert!(error.contains("--continuous-batching"), "{error}");

    let error = ServeOptions::parse(group_argv(&checkpoint, &["--synthetic"]))
        .expect_err("a group has no synthetic plan to serve");
    assert!(error.contains("--synthetic"), "{error}");

    let other = ServeOptions::parse(vec![
        "--model".to_owned(),
        sparkflash_model::descriptor::QWEN38_FLASH_NEXT_WEIGHTS_ID.to_owned(),
        "--checkpoint".to_owned(),
        checkpoint.clone(),
        "--rank".to_owned(),
        "0".to_owned(),
        "--peer".to_owned(),
        "127.0.0.1:29600,127.0.0.1:29601".to_owned(),
        "--converted".to_owned(),
        checkpoint,
    ])
    .expect_err("--rank is the Inkling device arm's group");
    assert!(other.contains("Inkling-2-Small"), "{other}");
}

#[test]
fn a_group_reads_one_rank_file_per_rank() {
    let directory = tempfile::tempdir().expect("a temporary converted directory");
    std::fs::write(directory.path().join("rank-0-of-2.bin"), b"").expect("a rank file");
    let peers = "127.0.0.1:29600,127.0.0.1:29601";

    let group = group_from_cli(
        Some(0),
        None,
        Some(peers),
        None,
        Some(directory.path().to_path_buf()),
    )
    .expect("the peer list names the world")
    .expect("a group");
    assert_eq!(group.world, 2);
    assert_eq!(
        group.control,
        SocketAddr::new(
            "127.0.0.1".parse().expect("an address"),
            GROUP_DEFAULT_CONTROL_PORT
        )
    );
    assert_eq!(
        group
            .rank_file()
            .expect("a rank file")
            .file_name()
            .and_then(std::ffi::OsStr::to_str),
        Some("rank-0-of-2.bin")
    );

    let error = group_from_cli(
        Some(4),
        Some(4),
        Some("127.0.0.1:29600,127.0.0.1:29601,127.0.0.1:29602,127.0.0.1:29603"),
        None,
        Some(directory.path().to_path_buf()),
    )
    .expect_err("rank 4 is outside the world");
    assert!(error.contains("outside a world of 4"), "{error}");

    let error = group_from_cli(
        Some(0),
        Some(4),
        Some(peers),
        None,
        Some(directory.path().to_path_buf()),
    )
    .expect_err("two addresses for a world of four");
    assert!(
        error.contains("2 address(es)") && error.contains("world is 4"),
        "{error}"
    );
}
