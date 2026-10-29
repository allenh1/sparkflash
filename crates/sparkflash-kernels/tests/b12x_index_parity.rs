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

//! Value-for-value check of the B12x index helpers in kernels/m4_common_sm121.cuh against their
//! Rust twins in src/mxfp8_ref.rs.
//!
//! **Attribution:** The `b12x` naming and SM120/SM121 block-scaled MXFP8/MXFP4 micro-tile
//! indexing concepts originate from [`b12x`](https://github.com/local-inference-lab/b12x) (now in
//! FlashInfer under `flashinfer/experimental/b12x`, Apache-2.0; Luke Alonso `@lukealonso`,
//! `@original-el8`, and contributors) and
//! [NVIDIA Flare](https://github.com/NVIDIA/flare-local-inference) (Apache-2.0). See the root
//! `NOTICE` file for full provenance details.
//!
//! This exists because a comment in the wo_a kernel claimed this test did, and it did not: the two
//! implementations were only compared with themselves, which is how a group stride taken from the
//! weight byte size could pass every test while reading 2 MiB past the end of the scale buffer.
//!
//! The check compiles the helper block OUT OF THE HEADER with a C compiler - not a copy of it - and
//! runs it over the same shapes and indices as the Rust functions, comparing every value. The
//! helpers are plain integer arithmetic with no CUDA builtins, so a host C compiler executes them
//! meaningfully; that is the only reason this is possible without a GPU. The three layout #defines
//! are also taken from the header text, so the C side cannot silently use stale constants, and the
//! Rust constants are compared against the same text.

use std::fmt::Write as _;
use std::path::PathBuf;
use std::process::Command;

use sparkflash_kernels::mxfp8_ref::{
    B12X_KTILE, B12X_NTILE, B12X_TILE_BYTES, b12x_buffer_bytes, b12x_padded_rows,
    b12x_row_scale_bytes, b12x_row_scale_index, b12x_scale_bytes, b12x_scale_index,
    b12x_weight_index,
};

const BEGIN_MARKER: &str = "SPARK_B12X_INDEX_HELPERS_BEGIN";
const END_MARKER: &str = "SPARK_B12X_INDEX_HELPERS_END";

fn header_source() -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../kernels/m4_common_sm121.cuh");
    std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()))
}

/// The helper block, extracted verbatim from the header.
fn helper_block(header: &str) -> String {
    // The markers sit inside line comments, so the block runs from the end of the begin marker's
    // line to the start of the end marker's line: a marker's own comment text must not leak into
    // the extracted C.
    let begin_at = header
        .find(BEGIN_MARKER)
        .unwrap_or_else(|| panic!("the header must mark the helper block with {BEGIN_MARKER}"));
    let end_at = header
        .find(END_MARKER)
        .unwrap_or_else(|| panic!("the header must end the helper block with {END_MARKER}"));
    assert!(begin_at < end_at, "the helper markers are out of order");
    let start = header[begin_at..]
        .find('\n')
        .map(|offset| begin_at + offset + 1)
        .expect("the begin marker is on its own line");
    let end = header[..end_at]
        .rfind('\n')
        .map(|offset| offset + 1)
        .expect("the end marker is on its own line");
    assert!(start < end, "the extracted block is empty");
    header[start..end].to_string()
}

/// The right-hand side of a layout #define in the header.
fn header_define(header: &str, name: &str) -> String {
    let prefix = format!("#define {name} ");
    header
        .lines()
        .find(|line| line.starts_with(&prefix))
        .unwrap_or_else(|| panic!("{name} is not defined in the header"))
        .trim_start_matches(&prefix)
        .trim()
        .trim_end_matches('u')
        .to_string()
}

/// Shapes whose shapes and sizes are compared, and the shapes enumerated index by index.
const SIZE_SHAPES: &[(usize, usize)] = &[
    (1, 32),
    (17, 64),
    (31, 32),
    (32, 96),
    (33, 128),
    (5, 32),
    (16, 4096),
    (1024, 4096),
];
const INDEX_SHAPES: &[(usize, usize)] = &[(1, 32), (17, 64), (33, 96), (64, 128)];

#[test]
fn the_cuda_b12x_index_helpers_agree_with_the_rust_ones_value_for_value() {
    let header = header_source();
    let block = helper_block(&header).replace("__device__ __forceinline__", "");
    assert!(
        block.contains("spark_b12x_weight_index") && block.contains("spark_b12x_scale_index"),
        "the extracted block must contain the index helpers"
    );
    // The Rust constants must be the header's constants.
    assert_eq!(
        B12X_NTILE,
        header_define(&header, "SPARK_B12X_NTILE")
            .parse::<usize>()
            .unwrap()
    );
    assert_eq!(
        B12X_KTILE,
        header_define(&header, "SPARK_B12X_KTILE")
            .parse::<usize>()
            .unwrap()
    );
    // The tile-byte define is an expression rather than a literal, so it is pinned as text; with
    // the two tile dimensions pinned above, the product follows.
    assert_eq!(
        header_define(&header, "SPARK_B12X_TILE_BYTES"),
        "(SPARK_B12X_NTILE * SPARK_B12X_KTILE)"
    );
    assert_eq!(B12X_TILE_BYTES, B12X_NTILE * B12X_KTILE);

    let mut source = String::new();
    source.push_str("#include <stdint.h>\n#include <stdio.h>\n");
    writeln!(
        source,
        "#define SPARK_B12X_NTILE {}u",
        header_define(&header, "SPARK_B12X_NTILE")
    )
    .unwrap();
    writeln!(
        source,
        "#define SPARK_B12X_KTILE {}u",
        header_define(&header, "SPARK_B12X_KTILE")
    )
    .unwrap();
    writeln!(
        source,
        "#define SPARK_B12X_TILE_BYTES {}",
        header_define(&header, "SPARK_B12X_TILE_BYTES")
    )
    .unwrap();
    source.push_str(&block);
    source.push_str("int main(void) {\n");
    for (rows, inner) in SIZE_SHAPES {
        writeln!(
            source,
            "  printf(\"size {rows} {inner} %llu %llu %llu %llu\\n\", (unsigned long long)spark_b12x_buffer_bytes({rows}u, {inner}u), (unsigned long long)spark_b12x_scale_bytes({rows}u, {inner}u), (unsigned long long)spark_b12x_row_scale_bytes({rows}u, {inner}u), (unsigned long long)spark_b12x_padded_rows({rows}u));"
        )
        .unwrap();
    }
    for (rows, inner) in INDEX_SHAPES {
        for n in 0..*rows {
            for k in 0..*inner {
                writeln!(
                    source,
                    "  printf(\"index {rows} {inner} {n} {k} %llu %llu %llu\\n\", (unsigned long long)spark_b12x_weight_index({n}u, {k}u, {inner}u), (unsigned long long)spark_b12x_scale_index({n}u, {k}u, {inner}u), (unsigned long long)spark_b12x_row_scale_index({n}u, {k}u, {inner}u));"
                )
                .unwrap();
            }
        }
    }
    source.push_str("  return 0;\n}\n");

    let dir = std::env::temp_dir().join(format!(
        "spark-b12x-parity-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let c_path = dir.join("parity.c");
    let exe_path = dir.join("parity");
    std::fs::write(&c_path, &source).expect("write the C source");

    let compile = Command::new("cc")
        .arg("-O2")
        .arg("-o")
        .arg(&exe_path)
        .arg(&c_path)
        .output()
        .unwrap_or_else(|error| {
            panic!(
                "a C compiler is required for this check and could not be run: {error}. The helpers \
                 are the layout contract, so this test is not optional."
            )
        });
    assert!(
        compile.status.success(),
        "the extracted helper block did not compile as C, which already means the header and the \
         Rust twin cannot be the same code:\n{}",
        String::from_utf8_lossy(&compile.stderr)
    );

    let run = Command::new(&exe_path)
        .output()
        .expect("run the C helper probe");
    assert!(run.status.success(), "the C helper probe failed");
    let output = String::from_utf8(run.stdout).expect("the probe prints ASCII");

    let mut checked_sizes = 0usize;
    let mut checked_indices = 0usize;
    for line in output.lines() {
        let fields: Vec<&str> = line.split_whitespace().collect();
        match fields.first() {
            Some(&"size") => {
                let rows: usize = fields[1].parse().unwrap();
                let inner: usize = fields[2].parse().unwrap();
                let buffer: usize = fields[3].parse().unwrap();
                let scales: usize = fields[4].parse().unwrap();
                let row_scales: usize = fields[5].parse().unwrap();
                let padded: usize = fields[6].parse().unwrap();
                assert_eq!(buffer, b12x_buffer_bytes(rows, inner), "buffer bytes");
                assert_eq!(scales, b12x_scale_bytes(rows, inner), "scale bytes");
                assert_eq!(
                    row_scales,
                    b12x_row_scale_bytes(rows, inner),
                    "row scale bytes"
                );
                assert_eq!(padded, b12x_padded_rows(rows), "padded rows");
                checked_sizes += 1;
            }
            Some(&"index") => {
                let rows: usize = fields[1].parse().unwrap();
                let inner: usize = fields[2].parse().unwrap();
                let n: usize = fields[3].parse().unwrap();
                let k: usize = fields[4].parse().unwrap();
                let weight: usize = fields[5].parse().unwrap();
                let scale: usize = fields[6].parse().unwrap();
                let row_scale: usize = fields[7].parse().unwrap();
                assert_eq!(
                    weight,
                    b12x_weight_index(n, k, inner),
                    "weight index for ({n}, {k}) in ({rows}, {inner})"
                );
                assert_eq!(
                    scale,
                    b12x_scale_index(n, k, inner),
                    "scale index for ({n}, {k}) in ({rows}, {inner})"
                );
                assert_eq!(
                    row_scale,
                    b12x_row_scale_index(n, k, inner),
                    "row scale index for ({n}, {k}) in ({rows}, {inner})"
                );
                checked_indices += 1;
            }
            other => panic!("unexpected probe line {other:?}"),
        }
    }
    assert_eq!(checked_sizes, SIZE_SHAPES.len());
    assert_eq!(
        checked_indices,
        INDEX_SHAPES
            .iter()
            .map(|(rows, inner)| rows * inner)
            .sum::<usize>()
    );
    // Non-vacuity: the helpers really are being exercised over more than one tile.
    assert!(checked_indices > 3_000);
    let _ = std::fs::remove_dir_all(&dir);
}
