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

//! The recurrence flags parse and refuse the shapes the group cannot serve.

use sparkflash_cli::serve::{
    RECURRENCE_DEFAULT_PORT, RecurrenceNet, RecurrenceServeOptions, ServeOptions,
};

const PEERS: &str = "192.0.2.10,192.0.2.11,192.0.2.12,192.0.2.13";

fn dory_checkpoint() -> tempfile::TempDir {
    const CONFIG: &str = include_str!(
        "../../sparkflash-model-dory/config/nvidia_nemotron_labs_dory1_0_4b_bf16.hf.json"
    );
    let directory = tempfile::tempdir().expect("a temporary checkpoint");
    std::fs::write(directory.path().join("config.json"), CONFIG)
        .expect("the Dory config is written");
    directory
}

fn parse(directory: &tempfile::TempDir, extra: &[&str]) -> Result<ServeOptions, String> {
    let mut args = vec![
        "--checkpoint".to_owned(),
        directory.path().display().to_string(),
    ];
    args.extend(extra.iter().map(|arg| (*arg).to_owned()));
    ServeOptions::parse(args)
}

#[test]
fn no_flag_is_one_process() {
    assert_eq!(
        RecurrenceServeOptions::from_cli(None, None, None, None, None),
        Ok(None)
    );
}

#[test]
fn a_dory_checkpoint_takes_a_recurrence_rank() {
    let directory = dory_checkpoint();
    let options = parse(
        &directory,
        &["--recurrence-rank", "2", "--recurrence-peers", PEERS],
    )
    .expect("a Dory rank parses");
    let group = options.recurrence.expect("the group is parsed");
    assert_eq!(group.rank, 2);
    assert_eq!(group.world(), 4);
    assert_eq!(group.port, RECURRENCE_DEFAULT_PORT);
    assert_eq!(
        group.net,
        RecurrenceNet::Rdma,
        "the ring is RDMA unless asked"
    );
    assert_eq!(
        group.draft_lanes, 16,
        "a four-rank group drafts up to two passes unless asked"
    );
    let options = parse(
        &directory,
        &[
            "--recurrence-rank",
            "0",
            "--recurrence-peers",
            PEERS,
            "--recurrence-port",
            "30000",
            "--recurrence-net",
            "any",
        ],
    )
    .expect("a rank 0 with a port and any network parses");
    let group = options.recurrence.expect("the group is parsed");
    assert_eq!((group.port, group.net), (30000, RecurrenceNet::Any));
}

#[test]
fn a_malformed_group_is_refused_by_name() {
    let refused = |rank, peers, port| {
        RecurrenceServeOptions::from_cli(rank, peers, port, None, None)
            .expect_err("the group is refused")
    };
    assert!(refused(None, Some(PEERS), None).contains("--recurrence-rank is required"));
    assert!(refused(Some(0), None, None).contains("--recurrence-peers is required"));
    assert!(refused(Some(0), Some("192.0.2.10"), None).contains("at least two ranks"));
    assert!(refused(Some(4), Some(PEERS), None).contains("outside"));
    assert!(refused(Some(0), Some("192.0.2.10,,192.0.2.11"), None).contains("empty address"));
    assert!(refused(Some(0), Some("192.0.2.10,node1"), None).contains("not an IP address"));
    assert!(refused(Some(0), Some(PEERS), Some(u16::MAX - 1)).contains("leaves no port"));
    let two = RecurrenceServeOptions::from_cli(
        Some(0),
        Some("192.0.2.10,192.0.2.11"),
        None,
        None,
        Some(1),
    );
    assert!(
        two.expect_err("rank 1 is the last")
            .contains("--recurrence-draft-lanes")
    );
}

#[test]
fn the_group_refuses_what_it_would_not_apply() {
    let directory = dory_checkpoint();
    let group = ["--recurrence-rank", "0", "--recurrence-peers", PEERS];
    // The group holds a sequence lane a concurrent request; one Dory process stays serial.
    let options = parse(&directory, &[&group[..], &["--concurrency", "4"]].concat())
        .expect("the group serves concurrent requests");
    assert_eq!(options.concurrency, 4);
    let error = parse(&directory, &["--concurrency", "2"])
        .expect_err("one Dory process holds one sequence");
    assert!(error.contains("--concurrency"), "{error}");
    let error = parse(
        &directory,
        &[&group[..], &["--tp4-rank", "0", "--tp4-peers", PEERS]].concat(),
    )
    .expect_err("two groups at once");
    assert!(error.contains("different groups"), "{error}");
    let error = ServeOptions::parse(
        [
            "--synthetic",
            "--recurrence-rank",
            "0",
            "--recurrence-peers",
            PEERS,
        ]
        .map(str::to_owned),
    )
    .expect_err("the synthetic plan has no recurrence");
    assert!(error.contains("--synthetic"), "{error}");
}
