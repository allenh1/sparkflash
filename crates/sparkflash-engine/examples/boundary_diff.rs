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

//! Phase 1 boundary diff: compare two version 1 dumps (host reference against a device run).
//!
//! Usage: boundary_diff HOST.manifest HOST.blob DEVICE.manifest DEVICE.blob
//!
//! Prints the step identity and the backbone count each half declares, then one line per differing
//! layer and tensor. A pair whose backbone lines disagree is refused. Exit status is 0 when every
//! tensor is bit-identical and 1 when anything differs, so a script can gate on it.
use sparkflash_engine::forward::boundary::{BoundaryTable, diff_v1, read_v1_with_layout};
use std::fs;

fn read_pair(manifest_path: &str, blob_path: &str) -> (String, Vec<u8>, BoundaryTable, usize) {
    let manifest = fs::read_to_string(manifest_path).unwrap_or_else(|error| {
        panic!("read {manifest_path}: {error}");
    });
    let blob = fs::read(blob_path).unwrap_or_else(|error| {
        panic!("read {blob_path}: {error}");
    });
    let (table, backbone) = read_v1_with_layout(&manifest, &blob).unwrap_or_else(|error| {
        panic!("{manifest_path} + {blob_path} are not a v1 pair: {error}");
    });
    (manifest, blob, table, backbone)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut take = || {
        args.next()
            .expect("usage: boundary_diff HOST.manifest HOST.blob DEVICE.manifest DEVICE.blob")
    };
    let (host_manifest, host_blob, host, host_backbone) = read_pair(&take(), &take());
    let (device_manifest, device_blob, device, device_backbone) = read_pair(&take(), &take());
    println!(
        "host:   rows={} start={} anchor={:?} backbone={host_backbone}",
        host.len(),
        host.start_position(),
        host.mtp_anchor()
    );
    println!(
        "device: rows={} start={} anchor={:?} backbone={device_backbone}",
        device.len(),
        device.start_position(),
        device.mtp_anchor()
    );
    let diff = diff_v1(&host_manifest, &host_blob, &device_manifest, &device_blob)
        .unwrap_or_else(|error| panic!("the two dumps are not the same step and layout: {error}"));
    if diff.is_identical() {
        println!("diff: all {} rows bit-identical", diff.rows().len());
        return;
    }
    println!("diff: differs in {} rows", diff.differing_layers().len());
    for row in diff.rows() {
        if row.is_identical() {
            continue;
        }
        println!(
            "row {}: {} of {} tensors differ",
            row.layer(),
            row.differing(),
            row.tensors().len()
        );
        for tensor in row.tensors() {
            if tensor.differing() == 0 {
                continue;
            }
            println!(
                "  {}: {} differ, max abs delta {}",
                tensor.name(),
                tensor.differing(),
                tensor.max_abs_delta()
            );
        }
    }
    std::process::exit(1);
}
