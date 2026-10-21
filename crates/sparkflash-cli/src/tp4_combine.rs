//! The routed-down combine layout selector for the four-rank arm.
//!
//! Two row layouts exist for the MoE down projection (`w2`) on this plan:
//!
//! * **stored 576-K** - what the rank arena actually holds. The checkpoint keeps TWO SEPARATE
//!   tensors: `w2.weight` [experts][5120][288] packed E2M1 codes (576 values per output row) and
//!   `w2.scale` [experts][5120][18] E8M0 blocks. 306 is the cost of one output row across both
//!   tensors, NOT an interleaved row stride: the launch passes the two base pointers separately
//!   (kernel-lead's ABI note, 2026-10-20; staged 306-byte interleaved rows are rejected by
//!   `mxfp4_ref::check_stored_moe_layout`). The resident plan keeps both tensors verbatim and the
//!   bind checked every byte.
//! * **A8 pre-padded 640-K** - what `b12x_mxfp4_moe_sm121` reads: 320 packed bytes + 20 scale bytes =
//!   340 bytes per row, because the single-node loader pads each row to 640 values before launch.
//!
//! The four-rank arm never performs that padding (a boot repack would cost +25.2 GiB/rank on a
//! 121 GiB node already holding a 72.3 GiB arena), so the only valid launch on arena rows is the
//! stored-layout kernel variant from task-6 (`b12x_mxfp4_moe_stored_sm121`). This module is the one
//! decision point for that launch:
//!
//! * it derives and validates the row layout from the bind's byte geometry;
//! * it refuses, by name, when the stored ABI/cubin is not staged - it never falls back to the
//!   pre-padded kernel on stored rows, because that launch would misindex every row after the
//!   first and produce plausible-looking wrong numbers;
//! * it reports the choice as an evidence row for `sparkflash-tp4-token/1`'s `launch_census`.
//!
//! [`STORED_COMBINE_ABI_READY`] is the compile-time gate that flips when kernel-lead's task-6
//! module lands in `sparkflash-kernels` (loader + params). Until then the selector always refuses,
//! which is the honest state for the next window's preflight.

use std::fmt;
use std::path::{Path, PathBuf};

use sparkflash_kernels::cuda_driver::Deviceptr;

/// The stored-layout combine kernel variant (task-6). Its loader lands in `sparkflash-kernels`.
pub const STORED_COMBINE_KERNEL: &str = "b12x_mxfp4_moe_stored_sm121";

/// The pre-padded kernel the single-node path uses; NOT launchable on the four-rank arena rows.
pub const PRE_ADDED_COMBINE_KERNEL: &str = "b12x_mxfp4_moe_sm121";

/// Values per output row in the stored plan layout (the checkpoint's `w2` K extent per rank).
pub const STORED_ELEMENTS_PER_ROW: usize = 576;
/// Packed E2M1 bytes per stored row (two values per byte).
pub const STORED_PACKED_BYTES_PER_ROW: usize = STORED_ELEMENTS_PER_ROW / 2;
/// E8M0 scale bytes per stored row (one block scale per 32 values).
pub const STORED_SCALE_BYTES_PER_ROW: usize = STORED_ELEMENTS_PER_ROW / 32;
/// Total stored bytes per output row: packed payload + its scale row.
pub const STORED_ROW_BYTES: usize = STORED_PACKED_BYTES_PER_ROW + STORED_SCALE_BYTES_PER_ROW;

/// Values per output row in the A8 pre-padded layout (the single-node loader's padding target).
pub const PRE_ADDED_ELEMENTS_PER_ROW: usize = 640;
/// Packed E2M1 bytes per pre-padded row.
pub const PRE_ADDED_PACKED_BYTES_PER_ROW: usize = PRE_ADDED_ELEMENTS_PER_ROW / 2;
/// E8M0 scale bytes per pre-padded row.
pub const PRE_ADDED_SCALE_BYTES_PER_ROW: usize = PRE_ADDED_ELEMENTS_PER_ROW / 32;
/// Total pre-padded bytes per output row.
pub const PRE_ADDED_ROW_BYTES: usize =
    PRE_ADDED_PACKED_BYTES_PER_ROW + PRE_ADDED_SCALE_BYTES_PER_ROW;

/// The stored kernel's pinned cubin size (kernel-lead's twenty-entry baseline measurement).
pub const STORED_CUBIN_BYTES: usize = sparkflash_kernels::mxfp4_moe_stored_abi::CUBIN_BYTES;
/// The stored kernel's pinned cubin sha256, from the same measurement.
pub const STORED_CUBIN_SHA256: &str = sparkflash_kernels::mxfp4_moe_stored_abi::CUBIN_SHA256;
/// The stored kernel's pinned cubin symbol, which is also its cubin stem.
pub const STORED_CUBIN_SYMBOL: &str = sparkflash_kernels::mxfp4_moe_stored_abi::KERNEL;

/// Verifies staged cubin bytes against the pinned size and sha256 before a driver ever sees them.
///
/// # Errors
/// A size that differs from the pin, or a sha256 that differs from it, naming the expected and the
/// measured value. Both are named refusals: a cubin that is not the measured artefact must not be
/// launched, because the stored row contract is only verified for that exact build.
pub fn verify_stored_cubin(bytes: &[u8]) -> Result<(), String> {
    use sha2::{Digest, Sha256};
    if bytes.len() != STORED_CUBIN_BYTES {
        return Err(format!(
            "stored combine cubin pin: {STORED_CUBIN_SYMBOL} is {} bytes but the pinned cubin is {STORED_CUBIN_BYTES}; refusing to load a different artefact",
            bytes.len()
        ));
    }
    let digest = Sha256::digest(bytes);
    let measured = digest
        .iter()
        .fold(String::with_capacity(64), |mut text, byte| {
            use std::fmt::Write as _;
            let _ = write!(text, "{byte:02x}");
            text
        });
    if measured != STORED_CUBIN_SHA256 {
        return Err(format!(
            "stored combine cubin pin: {STORED_CUBIN_SYMBOL} sha256 is {measured} but the pinned cubin is {STORED_CUBIN_SHA256}; refusing to load a different artefact"
        ));
    }
    Ok(())
}

/// One stored-combine launch's ten arguments, in kernel-lead's pinned order, plus its grid.
///
/// The stored kernel is a drop-in for the incumbent combine call site: the argument array type is
/// shared (sparkflash_kernels::mxfp4_moe_launch::Mxfp4MoeParams), so the order cannot drift between
/// the two kernels. What this builder adds is the one thing the ABI note corrected: the two weight
/// operands are SEPARATE base pointers - packed FP4 codes and their ue8m0 scale grid - never one
/// interleaved row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StoredCombineArgs {
    /// A8 activation rows: 640 bytes per routed row, unchanged from the incumbent chain.
    pub act_fp8: Deviceptr,
    /// The A8 activation's ue8m0 grid: 20 bytes per routed row.
    pub act_scales: Deviceptr,
    /// The stored weight's packed E2M1 codes: 288 bytes per output row.
    pub w_fp4: Deviceptr,
    /// The stored weight's E8M0 scale blocks: 18 bytes per output row.
    pub w_scales: Deviceptr,
    /// The routed expert ids for this token (the kernel walks TOP_K rows).
    pub expert_ids: Deviceptr,
    /// The f32 slot accumulator: the kernel ADDS each rank's partials, the caller zeroes once.
    pub slot_acc: Deviceptr,
    /// Tokens in this launch (grid.x).
    pub tokens: u32,
    /// The trace ring's device address, or 0 for the trace-disabled cubin.
    pub trace_ring: u64,
    /// The layer id the trace event records.
    pub layer_id: u32,
    /// The step sequence the trace event records.
    pub step_seq: u64,
}

impl StoredCombineArgs {
    /// The shared argument array, in the pinned kernel declaration's order.
    #[must_use]
    pub fn params(&self) -> sparkflash_kernels::mxfp4_moe_stored_launch::Mxfp4MoeStoredParams {
        sparkflash_kernels::mxfp4_moe_stored_launch::Mxfp4MoeStoredParams::new(
            self.act_fp8,
            self.act_scales,
            self.w_fp4,
            self.w_scales,
            self.expert_ids,
            self.slot_acc,
            self.tokens,
            self.trace_ring,
            self.layer_id,
            self.step_seq,
        )
    }

    /// The ten words a launch passes, in order.
    #[must_use]
    pub fn words(&self) -> [u64; STORED_ARG_WORDS] {
        let params = self.params();
        let mut words = [0u64; STORED_ARG_WORDS];
        words.copy_from_slice(params.as_slice());
        words
    }

    /// The launch grid: one block row per token, 160 hidden tiles across.
    ///
    /// # Errors
    /// A zero token count is refused rather than launched as a zero-sized grid.
    pub fn grid(&self) -> Result<(u32, u32, u32), String> {
        sparkflash_kernels::mxfp4_moe_stored_launch::grid(self.tokens)
    }
}

/// The stored combine's argument-word count, from the pinned ABI module.
pub const STORED_ARG_WORDS: usize = sparkflash_kernels::mxfp4_moe_stored_launch::ARG_WORDS;

/// The launch evidence row: geometry and operands named, so a window log shows what was launched.
#[must_use]
pub fn stored_combine_launch_evidence(args: &StoredCombineArgs) -> serde_json::Value {
    let grid = args.grid().map_or_else(
        |error| serde_json::json!({"error": error}),
        |(x, y, z)| serde_json::json!({"x": x, "y": y, "z": z}),
    );
    serde_json::json!({
        "kernel": STORED_COMBINE_KERNEL,
        "symbol": STORED_CUBIN_SYMBOL,
        "cubin_bytes": STORED_CUBIN_BYTES,
        "cubin_sha256": STORED_CUBIN_SHA256,
        "arg_words": STORED_ARG_WORDS,
        "args": [
            "act_fp8", "act_scales", "w_fp4", "w_scales", "expert_ids",
            "slot_acc", "tokens", "trace_ring", "layer_id", "step_seq",
        ],
        "grid": grid,
        "block": {"x": 256, "y": 1, "z": 1},
        "row_layout": "two separate tensors: 288-byte FP4 code rows and 18-byte E8M0 scale rows, 306 bytes per output row across both",
        "act_layout": "A8 chain layout: 640-byte rows and a 20-byte ue8m0 grid",
        "slot_acc": "f32 in/out; the kernel adds each rank's partials, the caller zeroes once",
    })
}

/// Loads the pinned stored cubin and launches it. The pin is verified before the driver sees the
/// bytes, so a cubin that is not kernel-lead's measured artefact never becomes a launch.
#[cfg(feature = "cuda")]
pub struct StoredCombine {
    /// The cubin path the launch was loaded from (recorded in evidence).
    pub cubin: PathBuf,
    /// The selected row layout (stored 576-K; the pre-added layout has no stored kernel).
    pub layout: CombineLayout,
    module: sparkflash_kernels::cuda_driver::Module,
    kernel: sparkflash_kernels::cuda_driver::Kernel,
}

#[cfg(feature = "cuda")]
impl StoredCombine {
    /// Reads, pins and loads the stored kernel named by a launchable selection.
    ///
    /// # Errors
    /// A non-launchable selection, a read failure, a cubin that differs from the pinned size or
    /// sha256, or the driver's load error (named by the symbol).
    pub fn open(
        cuda: &sparkflash_kernels::cuda_driver::Cuda,
        selection: &CombineSelection,
    ) -> Result<Self, String> {
        let CombineSelection::LaunchStored(kernel) = selection else {
            return Err(format!(
                "stored combine: the selector did not select a launchable stored kernel ({}); refusing to load anything else",
                selection
                    .refusal()
                    .map_or_else(|| "no refusal recorded".to_string(), ToString::to_string)
            ));
        };
        let bytes = std::fs::read(&kernel.cubin)
            .map_err(|error| format!("stored combine cubin {}: {error}", kernel.cubin.display()))?;
        verify_stored_cubin(&bytes)?;
        let (module, handle) = sparkflash_kernels::mxfp4_moe_stored_launch::load(cuda, &bytes)?;
        Ok(Self {
            cubin: kernel.cubin.clone(),
            layout: kernel.layout,
            module,
            kernel: handle,
        })
    }

    /// Launches the stored combine once. The slot accumulator is added into, never overwritten.
    ///
    /// # Errors
    /// A zero token count, or the driver's launch error.
    pub fn launch(
        &self,
        cuda: &sparkflash_kernels::cuda_driver::Cuda,
        args: &StoredCombineArgs,
    ) -> Result<(), String> {
        sparkflash_kernels::mxfp4_moe_stored_launch::launch(cuda, self.kernel, &args.params())
    }

    /// Unloads the loaded module. Device buffers are the caller's (the arena's).
    pub fn close(self, cuda: &sparkflash_kernels::cuda_driver::Cuda) {
        cuda.free(self.module, &[]);
    }
}

/// Whether task-6's stored-layout kernel (loader + params) is part of this build.
///
/// `true` since kernel-lead's pinned module landed in `sparkflash-kernels`
/// (`mxfp4_moe_stored_abi` + `mxfp4_moe_stored_launch`, twenty-entry baseline, pinned cubin
/// c452a9aa...), so the selector now hands the forward loop a launchable stored kernel instead of a
/// named refusal. The pin is still verified before the driver loads the bytes: a cubin whose size or
/// sha256 differs from the measurement is refused by name, never launched.
pub const STORED_COMBINE_ABI_READY: bool = true;

/// The two row layouts the selector can name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CombineLayout {
    /// The rank arena's checkpoint rows: 576 values = 288 packed + 18 scale bytes.
    Stored576,
    /// The A8 pre-padded rows the single-node loader builds: 640 values = 320 + 20 bytes.
    PreAdded640,
}

impl CombineLayout {
    /// A one-word name for evidence files.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Stored576 => "stored_576k",
            Self::PreAdded640 => "pre_added_640k",
        }
    }

    /// Values per output row.
    #[must_use]
    pub fn elements_per_row(self) -> usize {
        match self {
            Self::Stored576 => STORED_ELEMENTS_PER_ROW,
            Self::PreAdded640 => PRE_ADDED_ELEMENTS_PER_ROW,
        }
    }

    /// Packed payload bytes per output row.
    #[must_use]
    pub fn packed_bytes_per_row(self) -> usize {
        match self {
            Self::Stored576 => STORED_PACKED_BYTES_PER_ROW,
            Self::PreAdded640 => PRE_ADDED_PACKED_BYTES_PER_ROW,
        }
    }

    /// Scale bytes per output row.
    #[must_use]
    pub fn scale_bytes_per_row(self) -> usize {
        match self {
            Self::Stored576 => STORED_SCALE_BYTES_PER_ROW,
            Self::PreAdded640 => PRE_ADDED_SCALE_BYTES_PER_ROW,
        }
    }

    /// Total bytes per output row.
    #[must_use]
    pub fn row_bytes(self) -> usize {
        self.packed_bytes_per_row() + self.scale_bytes_per_row()
    }

    /// The kernel the layout needs.
    #[must_use]
    pub fn kernel(self) -> &'static str {
        match self {
            Self::Stored576 => STORED_COMBINE_KERNEL,
            Self::PreAdded640 => PRE_ADDED_COMBINE_KERNEL,
        }
    }
}

/// One rank's down-projection row geometry, derived from the bind's byte counts and validated
/// against the two layouts. `rows` is the rank's output-row count (5120 for the released plan).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CombineGeometry {
    rows: usize,
    layout: CombineLayout,
}

impl CombineGeometry {
    /// Builds a geometry from the bind's row count and the two arena slice byte counts.
    ///
    /// The inference is arithmetic, not a guess: exactly one layout must account for both slices,
    /// or the caller is told which numbers disagree with which layout.
    ///
    /// # Errors
    /// Zero rows, a slice that does not divide by the row count, or byte counts matching neither
    /// layout (a bug in the plan or the bind, reported with both candidates).
    pub fn from_shard_bytes(
        rows: usize,
        weight_bytes: usize,
        scale_bytes: usize,
    ) -> Result<Self, String> {
        if rows == 0 {
            return Err("combine geometry: zero output rows".to_string());
        }
        if !weight_bytes.is_multiple_of(rows) || !scale_bytes.is_multiple_of(rows) {
            return Err(format!(
                "combine geometry: {rows} rows do not divide the bound slices ({weight_bytes} weight bytes, {scale_bytes} scale bytes)"
            ));
        }
        let packed = weight_bytes / rows;
        let scales = scale_bytes / rows;
        let layout = [CombineLayout::Stored576, CombineLayout::PreAdded640]
            .into_iter()
            .find(|layout| {
                layout.packed_bytes_per_row() == packed
                    && layout.scale_bytes_per_row() == scales
            })
            .ok_or_else(|| {
                format!(
                    "combine geometry: bound slices are {packed} packed + {scales} scale bytes per row, but the stored layout is {STORED_PACKED_BYTES_PER_ROW}+{STORED_SCALE_BYTES_PER_ROW} and the pre-added layout is {PRE_ADDED_PACKED_BYTES_PER_ROW}+{PRE_ADDED_SCALE_BYTES_PER_ROW}; the bind and the selector disagree"
                )
            })?;
        Ok(Self { rows, layout })
    }

    /// The stored 576-K geometry for a row count (the four-rank arm's expected layout).
    #[must_use]
    pub fn stored(rows: usize) -> Self {
        Self {
            rows,
            layout: CombineLayout::Stored576,
        }
    }

    /// The pre-padded 640-K geometry for a row count (only a boot repack produces this).
    #[must_use]
    pub fn pre_added(rows: usize) -> Self {
        Self {
            rows,
            layout: CombineLayout::PreAdded640,
        }
    }

    /// Output rows this rank's combine writes.
    #[must_use]
    pub fn rows(self) -> usize {
        self.rows
    }

    /// The row layout.
    #[must_use]
    pub fn layout(self) -> CombineLayout {
        self.layout
    }

    /// Number of values per output row.
    #[must_use]
    pub fn elements_per_row(self) -> usize {
        self.layout.elements_per_row()
    }

    /// The bytes the payload slice must hold.
    #[must_use]
    pub fn weight_bytes(self) -> usize {
        self.rows * self.layout.packed_bytes_per_row()
    }

    /// The bytes the scale slice must hold.
    #[must_use]
    pub fn scale_bytes(self) -> usize {
        self.rows * self.layout.scale_bytes_per_row()
    }

    /// The combined payload+scale bytes (the arena's per-rank `w2` footprint).
    #[must_use]
    pub fn total_bytes(self) -> usize {
        self.rows * self.layout.row_bytes()
    }

    /// The kernel this geometry needs.
    #[must_use]
    pub fn kernel(self) -> &'static str {
        self.layout.kernel()
    }
}

/// Cubins found in the staged kernel directory, keyed by the kernel they carry.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CombineKernelSet {
    dir: Option<PathBuf>,
    stored: Option<PathBuf>,
    pre_added: Option<PathBuf>,
    /// Distinguishes "scanned a directory and the file was absent" from "no directory staged".
    scanned: bool,
}

impl CombineKernelSet {
    /// An empty set: nothing was staged.
    #[must_use]
    pub fn absent() -> Self {
        Self::default()
    }

    /// Scans a cubin directory for the two combine kernels by their conventional filenames.
    ///
    /// Presence is a filename check only; the driver load happens at launch time, because a cubin
    /// cannot be validated without a driver and this selector must work on a host with none.
    #[must_use]
    pub fn scan(dir: &Path) -> Self {
        let cubin = |kernel: &str| -> Option<PathBuf> {
            let path = dir.join(format!("{kernel}.cubin"));
            path.is_file().then_some(path)
        };
        Self {
            dir: Some(dir.to_path_buf()),
            stored: cubin(STORED_COMBINE_KERNEL),
            pre_added: cubin(PRE_ADDED_COMBINE_KERNEL),
            scanned: true,
        }
    }

    /// The directory this set was scanned from, if any.
    #[must_use]
    pub fn dir(&self) -> Option<&Path> {
        self.dir.as_deref()
    }

    /// The stored-layout cubin, if present.
    #[must_use]
    pub fn stored_cubin(&self) -> Option<&Path> {
        self.stored.as_deref()
    }

    /// The pre-added-layout cubin, if present.
    #[must_use]
    pub fn pre_added_cubin(&self) -> Option<&Path> {
        self.pre_added.as_deref()
    }

    /// Whether a directory was scanned at all.
    #[must_use]
    pub fn was_scanned(&self) -> bool {
        self.scanned
    }
}

/// Why the selector refused a combine launch. Every variant names the artefact or the arithmetic
/// that is missing; none of them can be mistaken for a bind failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CombineRefusal {
    /// The stored-layout kernel cubin is not staged in the kernel directory.
    MissingStoredCubin {
        /// The directory that was scanned, or "no kernel directory staged".
        dir: String,
        /// The filename the selector looked for.
        expected: String,
    },
    /// The stored cubin is staged but the loader/params that read it are not in this build.
    MissingStoredAbi {
        /// The kernel symbol whose loader is missing.
        kernel: String,
    },
    /// The arena rows are the stored layout, so the pre-padded kernel cannot be launched.
    PreAddedKernelOnStoredRows {
        /// The kernel that was present but unusable.
        kernel: String,
    },
    /// The arena holds pre-padded rows, which only a boot repack produces.
    ArenaIsNotStoredLayout {
        /// The layout found in the arena.
        layout: String,
    },
}

impl CombineRefusal {
    /// The refusal's one-line code, for evidence files and log greps.
    #[must_use]
    pub fn code(&self) -> &'static str {
        match self {
            Self::MissingStoredCubin { .. } => "stored-layout combine kernel unavailable",
            Self::MissingStoredAbi { .. } => "stored-layout combine ABI not wired",
            Self::PreAddedKernelOnStoredRows { .. } => {
                "pre-added combine kernel unusable on stored rows"
            }
            Self::ArenaIsNotStoredLayout { .. } => "arena rows are not the stored layout",
        }
    }
}

impl fmt::Display for CombineRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingStoredCubin { dir, expected } => write!(
                f,
                "stored-layout combine kernel unavailable: {expected} is not staged in {dir} (task-6 kernel). Refusing the four-rank combine rather than launching {PRE_ADDED_COMBINE_KERNEL} on stored 576-K rows, which would misindex every row after the first."
            ),
            Self::MissingStoredAbi { kernel } => write!(
                f,
                "stored-layout combine ABI not wired: {kernel} has no loader/params in sparkflash-kernels yet (task-6 pending). Refusing the four-rank combine rather than fabricating a launch."
            ),
            Self::PreAddedKernelOnStoredRows { kernel } => write!(
                f,
                "pre-added combine kernel unusable on stored rows: the arena holds {STORED_PACKED_BYTES_PER_ROW}+{STORED_SCALE_BYTES_PER_ROW}-byte rows, so {kernel} must not launch (its {PRE_ADDED_PACKED_BYTES_PER_ROW}+{PRE_ADDED_SCALE_BYTES_PER_ROW}-byte row read would misindex)."
            ),
            Self::ArenaIsNotStoredLayout { layout } => write!(
                f,
                "arena rows are not the stored layout: found {layout}. The four-rank arm does not perform the boot repack that produces the pre-added layout, so the combine is refused rather than pointed at rows that do not exist."
            ),
        }
    }
}

/// The launchable stored kernel: the cubin and the pinned symbol/row layout it must be called with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredCombineKernel {
    /// The staged cubin (filename convention `{symbol}.cubin`).
    pub cubin: PathBuf,
    /// The kernel symbol the loader must resolve.
    pub symbol: &'static str,
    /// The row layout the kernel reads (it is pinned to the stored 576-K layout).
    pub layout: CombineLayout,
}

/// The selector's answer: a launchable kernel, or a named refusal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CombineSelection {
    /// Launch the stored-layout kernel with [`StoredCombineKernel`].
    LaunchStored(StoredCombineKernel),
    /// Do not launch; the refusal names the missing artefact or the wrong layout.
    Refused(CombineRefusal),
}

impl CombineSelection {
    /// Whether a launch was selected.
    #[must_use]
    pub fn is_launchable(&self) -> bool {
        matches!(self, Self::LaunchStored(_))
    }

    /// The refusal, if this selection refuses.
    #[must_use]
    pub fn refusal(&self) -> Option<&CombineRefusal> {
        match self {
            Self::LaunchStored(_) => None,
            Self::Refused(refusal) => Some(refusal),
        }
    }

    /// The evidence row for `sparkflash-tp4-token/1`'s `launch_census`.
    #[must_use]
    pub fn evidence(&self, geometry: CombineGeometry) -> serde_json::Value {
        match self {
            Self::LaunchStored(kernel) => serde_json::json!({
                "stage": "moe_combine",
                "layout": geometry.layout().as_str(),
                "kernel": kernel.symbol,
                "cubin": kernel.cubin.display().to_string(),
                "rows": geometry.rows(),
                "row_bytes": geometry.layout().row_bytes(),
                "launch_compatible": true,
                "abi_ready": STORED_COMBINE_ABI_READY,
                "cubin_pin": {"bytes": STORED_CUBIN_BYTES, "sha256": STORED_CUBIN_SHA256},
                "row_layout": "two separate per-rank tensors: w2.weight [experts][5120][288] packed E2M1 codes and w2.scale [experts][5120][18] E8M0; 306 bytes per output row across both, not a stride",
                "act_layout": "A8 chain layout: 640-byte activation rows and a 20-byte ue8m0 grid, unchanged from the incumbent combine",
                "kernel_args": "10 words, same order as the incumbent combine: act_fp8, act_scales, w_fp4 (stored codes), w_scales (stored scales), expert_ids, slot_acc, tokens, trace_ring, layer_id, step_seq",
            }),
            Self::Refused(refusal) => serde_json::json!({
                "stage": "moe_combine",
                "layout": geometry.layout().as_str(),
                "kernel": geometry.kernel(),
                "rows": geometry.rows(),
                "row_bytes": geometry.layout().row_bytes(),
                "launch_compatible": false,
                "refusal": refusal.code(),
                "error": refusal.to_string(),
            }),
        }
    }
}

/// Chooses the combine kernel for one rank's geometry.
///
/// The stored layout is the only layout this arm can produce, and the stored kernel is the only
/// kernel that can read it. The pre-added kernel is never substituted: if the stored cubin or ABI
/// is missing, the selection refuses.
#[must_use]
pub fn select_combine(geometry: CombineGeometry, kernels: &CombineKernelSet) -> CombineSelection {
    if geometry.layout() != CombineLayout::Stored576 {
        return CombineSelection::Refused(CombineRefusal::ArenaIsNotStoredLayout {
            layout: geometry.layout().as_str().to_string(),
        });
    }
    let Some(cubin) = kernels.stored_cubin() else {
        // The pre-added cubin being present changes nothing: it cannot read these rows, and
        // offering it would be exactly the misindexing fallback this selector forbids.
        let _ = kernels.pre_added_cubin();
        return CombineSelection::Refused(CombineRefusal::MissingStoredCubin {
            dir: kernels.dir().map_or_else(
                || "no kernel directory staged".to_string(),
                |dir| dir.display().to_string(),
            ),
            expected: format!("{STORED_COMBINE_KERNEL}.cubin"),
        });
    };
    if !STORED_COMBINE_ABI_READY {
        return CombineSelection::Refused(CombineRefusal::MissingStoredAbi {
            kernel: STORED_COMBINE_KERNEL.to_string(),
        });
    }
    CombineSelection::LaunchStored(StoredCombineKernel {
        cubin: cubin.to_path_buf(),
        symbol: STORED_COMBINE_KERNEL,
        layout: geometry.layout(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const ROWS: usize = 5120; // released plan's w2 output rows per rank

    fn stored_bytes() -> (usize, usize) {
        (
            ROWS * STORED_PACKED_BYTES_PER_ROW,
            ROWS * STORED_SCALE_BYTES_PER_ROW,
        )
    }

    fn written_stored_quad() -> (tempfile::TempDir, CombineKernelSet) {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join(format!("{STORED_COMBINE_KERNEL}.cubin"));
        std::fs::write(&path, b"not a real cubin").expect("write");
        let set = CombineKernelSet::scan(dir.path());
        (dir, set)
    }

    #[test]
    fn the_pinned_row_constants_are_the_two_layouts_the_arm_can_meet() {
        assert_eq!(STORED_ELEMENTS_PER_ROW, 576);
        assert_eq!(STORED_PACKED_BYTES_PER_ROW, 288);
        assert_eq!(STORED_SCALE_BYTES_PER_ROW, 18);
        assert_eq!(STORED_ROW_BYTES, 306);
        assert_eq!(PRE_ADDED_ELEMENTS_PER_ROW, 640);
        assert_eq!(PRE_ADDED_PACKED_BYTES_PER_ROW, 320);
        assert_eq!(PRE_ADDED_SCALE_BYTES_PER_ROW, 20);
        assert_eq!(PRE_ADDED_ROW_BYTES, 340);
    }

    #[test]
    fn the_arena_bytes_infer_the_stored_layout_and_reject_a_third_shape() {
        let (weight, scale) = stored_bytes();
        let geometry = CombineGeometry::from_shard_bytes(ROWS, weight, scale).expect("stored");
        assert_eq!(geometry.layout(), CombineLayout::Stored576);
        assert_eq!(geometry.weight_bytes(), weight);
        assert_eq!(geometry.scale_bytes(), scale);
        assert_eq!(geometry.total_bytes(), ROWS * 306);

        // Pre-padded bytes are recognised too (the selector then refuses them; recognition is not
        // acceptance).
        let pre = CombineGeometry::from_shard_bytes(
            ROWS,
            ROWS * PRE_ADDED_PACKED_BYTES_PER_ROW,
            ROWS * PRE_ADDED_SCALE_BYTES_PER_ROW,
        )
        .expect("pre-added");
        assert_eq!(pre.layout(), CombineLayout::PreAdded640);

        let error = CombineGeometry::from_shard_bytes(ROWS, ROWS * 289, ROWS * 18)
            .expect_err("289-byte rows belong to neither layout");
        assert!(
            error.contains("the bind and the selector disagree"),
            "{error}"
        );
        assert!(error.contains("288+18"), "{error}");

        let error = CombineGeometry::from_shard_bytes(0, 0, 0).expect_err("zero rows");
        assert!(error.contains("zero output rows"), "{error}");
    }

    #[test]
    fn a_stored_geometry_without_the_stored_cubin_refuses_by_name_and_never_falls_back() {
        let geometry = CombineGeometry::stored(ROWS);
        let refusal = select_combine(geometry, &CombineKernelSet::absent());
        let text = refusal.refusal().expect("refusal").to_string();
        assert!(
            text.contains("stored-layout combine kernel unavailable"),
            "{text}"
        );
        assert!(text.contains(STORED_COMBINE_KERNEL), "{text}");
        assert!(text.contains("Refusing the four-rank combine"), "{text}");
        assert!(text.contains("misindex"), "{text}");
        assert!(!refusal.is_launchable());
    }

    #[test]
    fn a_present_pre_added_cubin_is_never_substituted_for_the_missing_stored_one() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join(format!("{PRE_ADDED_COMBINE_KERNEL}.cubin")),
            b"pre-added",
        )
        .expect("write");
        let kernels = CombineKernelSet::scan(dir.path());
        assert!(kernels.pre_added_cubin().is_some());
        assert!(kernels.stored_cubin().is_none());

        let selection = select_combine(CombineGeometry::stored(ROWS), &kernels);
        assert!(!selection.is_launchable());
        let refusal = selection.refusal().expect("refusal");
        assert!(matches!(refusal, CombineRefusal::MissingStoredCubin { .. }));
        let evidence = selection.evidence(CombineGeometry::stored(ROWS));
        assert_eq!(evidence["launch_compatible"], false);
        assert_eq!(evidence["layout"], "stored_576k");
    }

    #[test]
    fn a_pre_added_geometry_is_refused_even_when_both_cubins_are_present() {
        let dir = tempfile::tempdir().expect("tempdir");
        for kernel in [STORED_COMBINE_KERNEL, PRE_ADDED_COMBINE_KERNEL] {
            std::fs::write(dir.path().join(format!("{kernel}.cubin")), b"x").expect("write");
        }
        let kernels = CombineKernelSet::scan(dir.path());
        let selection = select_combine(CombineGeometry::pre_added(ROWS), &kernels);
        let refusal = selection.refusal().expect("refusal");
        match refusal {
            CombineRefusal::ArenaIsNotStoredLayout { layout } => {
                assert_eq!(layout, "pre_added_640k");
            }
            other => panic!("wrong refusal: {other}"),
        }
        assert!(refusal.to_string().contains("boot repack"), "{refusal}");
    }

    #[test]
    fn the_launch_path_accepts_the_stored_layout_once_the_abi_module_is_wired() {
        // kernel-lead's module is in this build, so a staged stored cubin is launchable: this is the
        // launch-path test the selector flip requires (the evidence row below reports abi_ready).
        let (_dir, kernels) = written_stored_quad();
        let selection = select_combine(CombineGeometry::stored(ROWS), &kernels);
        let CombineSelection::LaunchStored(ref kernel) = selection else {
            panic!("ABI_READY is true but the selector refused");
        };
        assert_eq!(kernel.symbol, STORED_COMBINE_KERNEL);
        assert_eq!(kernel.symbol, STORED_CUBIN_SYMBOL);
        assert_eq!(kernel.layout, CombineLayout::Stored576);
        assert!(
            kernel
                .cubin
                .ends_with(format!("{STORED_COMBINE_KERNEL}.cubin"))
        );
        let evidence = selection.evidence(CombineGeometry::stored(ROWS));
        assert_eq!(evidence["launch_compatible"], true);
        assert_eq!(evidence["abi_ready"], true);
        assert_eq!(evidence["row_bytes"], 306);
        assert_eq!(evidence["cubin_pin"]["bytes"], 30_760);
        assert_eq!(
            evidence["cubin_pin"]["sha256"],
            "1dbfd764f11e7378ecc1790b8f5bca2e97871c6a6ad3092cd16c6a5df767ba08"
        );
        assert!(
            evidence["row_layout"]
                .as_str()
                .expect("row layout")
                .contains("not a stride")
        );
    }

    #[test]
    fn the_stored_combine_argument_words_are_the_pinned_ten_word_order() {
        let args = StoredCombineArgs {
            act_fp8: Deviceptr(0x1000),
            act_scales: Deviceptr(0x2000),
            w_fp4: Deviceptr(0x3000),
            w_scales: Deviceptr(0x4000),
            expert_ids: Deviceptr(0x5000),
            slot_acc: Deviceptr(0x6000),
            tokens: 1,
            trace_ring: 0x7000,
            layer_id: 7,
            step_seq: 11,
        };
        assert_eq!(
            args.words(),
            [
                0x1000, 0x2000, 0x3000, 0x4000, 0x5000, 0x6000, 1, 0x7000, 7, 11
            ]
        );
        assert_eq!(args.params().tokens(), 1);
        assert_eq!(STORED_ARG_WORDS, 10);
        // The ABI correction made explicit: the code bytes and their scale grid are two separate
        // pointers, so a stored launch can never read the two tensors as one interleaved row.
        assert_ne!(args.words()[2], args.words()[3]);
        assert_eq!(args.grid(), Ok((1, 160, 1)));

        let empty = StoredCombineArgs { tokens: 0, ..args };
        assert!(empty.grid().is_err(), "a zero-token grid is refused");
    }

    #[test]
    fn the_stored_combine_launch_evidence_names_the_tiling_and_the_two_tensor_rows() {
        let args = StoredCombineArgs {
            act_fp8: Deviceptr(0x1000),
            act_scales: Deviceptr(0x2000),
            w_fp4: Deviceptr(0x3000),
            w_scales: Deviceptr(0x4000),
            expert_ids: Deviceptr(0x5000),
            slot_acc: Deviceptr(0x6000),
            tokens: 6,
            trace_ring: 0,
            layer_id: 3,
            step_seq: 9,
        };
        let evidence = stored_combine_launch_evidence(&args);
        assert_eq!(evidence["kernel"], STORED_COMBINE_KERNEL);
        assert_eq!(evidence["arg_words"], 10);
        assert_eq!(evidence["grid"]["x"], 6);
        assert_eq!(evidence["grid"]["y"], 160);
        assert_eq!(evidence["block"]["x"], 256);
        assert_eq!(evidence["args"][2], "w_fp4");
        assert_eq!(evidence["args"][3], "w_scales");
        assert!(
            evidence["row_layout"]
                .as_str()
                .expect("row layout")
                .contains("two separate tensors")
        );
        assert_eq!(evidence["cubin_sha256"], STORED_CUBIN_SHA256);
    }

    #[test]
    fn a_cubin_that_is_not_the_pinned_artefact_is_refused_before_the_driver_loads_it() {
        // Short cubin: the size check fires first and names both sizes.
        let error = verify_stored_cubin(b"not a cubin").expect_err("short cubin");
        assert!(error.contains("stored combine cubin pin"), "{error}");
        assert!(error.contains(&STORED_CUBIN_BYTES.to_string()), "{error}");

        // Right size, wrong content: the sha256 check names both digests.
        let bytes = vec![0u8; STORED_CUBIN_BYTES];
        let error = verify_stored_cubin(&bytes).expect_err("wrong cubin");
        assert!(error.contains("sha256 is"), "{error}");
        assert!(error.contains(STORED_CUBIN_SHA256), "{error}");
        assert!(
            error.contains("refusing to load a different artefact"),
            "{error}"
        );

        // The pin is the kernels module's measurement, not a second transcription.
        assert_eq!(
            STORED_CUBIN_BYTES,
            sparkflash_kernels::mxfp4_moe_stored_abi::CUBIN_BYTES
        );
        assert_eq!(
            STORED_CUBIN_SHA256,
            sparkflash_kernels::mxfp4_moe_stored_abi::CUBIN_SHA256
        );
    }

    #[test]
    fn the_kernel_set_scanner_reports_what_it_found_and_what_it_scanned() {
        let dir = tempfile::tempdir().expect("tempdir");
        let empty = CombineKernelSet::scan(dir.path());
        assert!(empty.was_scanned());
        assert!(empty.dir().is_some());
        assert!(empty.stored_cubin().is_none());

        std::fs::write(
            dir.path().join(format!("{STORED_COMBINE_KERNEL}.cubin")),
            b"stored",
        )
        .expect("write");
        let scanned = CombineKernelSet::scan(dir.path());
        assert!(scanned.stored_cubin().is_some());
        assert!(scanned.was_scanned());
        assert_eq!(
            scanned
                .stored_cubin()
                .expect("stored")
                .file_name()
                .expect("name"),
            format!("{STORED_COMBINE_KERNEL}.cubin").as_str()
        );
    }

    #[test]
    fn the_evidence_row_names_the_layout_the_kernel_and_the_refusal() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join(format!("{STORED_COMBINE_KERNEL}.cubin")),
            b"stored",
        )
        .expect("write");
        let kernels = CombineKernelSet::scan(dir.path());
        let selection = select_combine(CombineGeometry::stored(ROWS), &kernels);
        let evidence = selection.evidence(CombineGeometry::stored(ROWS));
        assert_eq!(evidence["stage"], "moe_combine");
        assert_eq!(evidence["layout"], "stored_576k");
        assert_eq!(evidence["rows"], ROWS);
        assert_eq!(evidence["row_bytes"], 306);
        assert_eq!(evidence["kernel"], STORED_COMBINE_KERNEL);
        if STORED_COMBINE_ABI_READY {
            assert_eq!(evidence["launch_compatible"], true);
        } else {
            assert_eq!(evidence["launch_compatible"], false);
            assert!(
                evidence["error"]
                    .as_str()
                    .expect("error")
                    .contains("task-6")
            );
        }
    }
}
