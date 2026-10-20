//! Device-resident weight arenas for the S5.2 device chain.
//!
//! The device chain reads a layer's released weights on the host and copies them to the device on
//! every position (see attention_device::upload_weight and the per-expert moe_device copies). That
//! is the per-token H2D the audit measured. This module is the other shape: ResidentArena takes the
//! per-rank resident plan (ResidentPlan) and the bytes ShardPlan::read_all already reads in place,
//! allocates one slab per ResidentGroup once, and uploads each tensor once.
//!
//! The arena keeps a per-tensor device pointer and a digest of the uploaded bytes. A launch binds
//! the pointer directly; ResidentArena::verify_slice lets the wiring assert that the bytes a kernel
//! is about to read are the bytes the plan named, so a wrong pointer cannot compute silently.
//!
//! What this module does not do: it launches no kernel and produces no token. It is the load-time
//! half. ArenaEvidence records the category bytes, the allocation and upload time and the arena
//! digest, and the window measurement on a node fills in the resident device bytes per rank. The
//! host copy is kept for verification in this first step; the caller drops it once the upload has
//! been verified against the checkpoint.
//!
//! Allocation mode: AllocationMode::Device is plain cuMemAlloc, which every current launch accepts.
//! AllocationMode::Unified is the UMA-coherent slab (managed or host-mapped device memory) the
//! design wants; it is refused here with a named error until the driver allocation method lands, so
//! a caller cannot believe it got coherent memory when it did not.
//!
//! Weight layout: the B12x kernels read a micro-tiled weight order, not the checkpoint's row-major
//! order. b12x_layout decides which resident tensors need the transform, tile_b12x applies it once
//! here, and the scale grids are already in B12x order and are never transformed. The kernels do
//! not tile internally, so a tensor must be tiled exactly once - in this arena, never again.

use std::collections::BTreeMap;
use std::time::Instant;

use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr};
use sparkflash_kernels::mxfp8_ref::b12x_buffer_bytes;

use crate::resident::{Placement, ResidentGroup, ResidentPlan, ResidentShard};
use crate::shard::StoredShard;

/// Every tensor pointer in an arena is aligned to this many bytes.
///
/// The FP4 and FP8 kernels load the weight and scale pointers as vectors; 256 bytes is the device
/// allocation granularity anyway, so a per-tensor alignment inside the slab costs nothing and keeps
/// a scale grid that starts mid-row from being under-aligned.
pub const ARENA_ALIGNMENT: usize = 256;

/// The FNV-1a 64 offset basis, matching the plan and read digests.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
/// The FNV-1a 64 prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// FNV-1a 64 over a byte slice.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// How the arena's device slabs are allocated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationMode {
    /// Plain device memory (cuMemAlloc): the default and what every current launch accepts.
    Device,
    /// UMA-coherent device memory (managed or host-mapped), which the design wants for the
    /// resident set. Refused until the driver method is wired, rather than silently falling back.
    Unified,
    /// UMA-coherent memory reserved as ONE contiguous slab for the whole rank plan, instead of one
    /// slab per resident group. One reservation up front is what the four-rank arm asks for: the
    /// plan's resident bytes are known before any read, so the arena is allocated once and every
    /// tensor is streamed into its offset. Same device memory kind as Unified.
    UnifiedSingle,
}

impl AllocationMode {
    /// The one-word name the evidence records.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Device => "device",
            Self::Unified => "unified",
            Self::UnifiedSingle => "unified-single-slab",
        }
    }
}

/// Which weight layout a resident tensor is stored in on the device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorLayout {
    /// The checkpoint's stored order, uploaded unchanged.
    Stored,
    /// The B12x micro-tile order the dense FP8 kernels walk, applied once at load.
    B12x,
}

impl TensorLayout {
    /// The one-word name the evidence records.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Stored => "stored",
            Self::B12x => "b12x",
        }
    }
}

/// The allocator an arena is built through, so the load path is testable without a driver.
///
/// The real implementation is CudaArenaAllocator; a test supplies a fake that records the
/// allocations and uploads. Nothing in ResidentArena::build touches the driver directly.
pub trait ArenaAllocator {
    /// Allocates one slab of bytes and returns its device pointer.
    ///
    /// # Errors
    /// The driver's refusal, or an allocator that does not support the requested mode.
    fn allocate(&mut self, bytes: usize) -> Result<Deviceptr, String>;

    /// Copies host bytes to a slab.
    ///
    /// # Errors
    /// The driver's copy error.
    fn upload(&mut self, destination: &Deviceptr, bytes: &[u8]) -> Result<(), String>;
}

/// One tensor as the arena builder takes it: its name, its design group, its bytes and the layout
/// the device kernels require.
#[derive(Debug, Clone, Copy)]
pub struct ArenaTensor<'a> {
    /// The checkpoint name.
    pub name: &'a str,
    /// The design group the plan assigned.
    pub group: ResidentGroup,
    /// This rank's slice bytes, in stored row-major order.
    pub bytes: &'a [u8],
    /// The layout to upload.
    pub layout: TensorLayout,
    /// The (cols, inner) geometry the B12x transform needs; Some exactly when layout is B12x.
    pub b12x: Option<(usize, usize)>,
}

/// One resident tensor inside the arena.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArenaSlice {
    /// The checkpoint name.
    pub name: String,
    /// The design group.
    pub group: ResidentGroup,
    /// The device pointer at which the slice starts.
    pub ptr: Deviceptr,
    /// The byte offset inside the group slab.
    pub offset: usize,
    /// The slice length in bytes (tiled and padded when the layout is B12x).
    pub bytes: usize,
    /// The layout the device copy is in.
    pub layout: TensorLayout,
    /// FNV-1a 64 of the uploaded bytes, for the first-use assertion.
    pub digest: u64,
}

/// One group slab.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GroupArena {
    /// The design group.
    pub group: ResidentGroup,
    /// The allocated bytes (tensors plus alignment padding).
    pub bytes: usize,
    /// How many tensors the group holds.
    pub tensors: usize,
    /// How many of them were tiled to B12x.
    pub tiled: usize,
    /// FNV-1a 64 over the group's tensor digests in name order.
    pub digest: u64,
}

/// The load-time evidence of one arena build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArenaEvidence {
    /// How the slabs were allocated.
    pub mode: AllocationMode,
    /// Resident tensors uploaded.
    pub tensors: usize,
    /// How many tensors were tiled to B12x.
    pub tiled_tensors: usize,
    /// The stored bytes of the planned slices, before any tiling padding.
    pub stored_bytes: u64,
    /// Uploaded payload bytes (tiled lengths included).
    pub bytes: u64,
    /// Allocated bytes, including alignment padding.
    pub allocated_bytes: u64,
    /// Time spent allocating, milliseconds.
    pub allocate_ms: u64,
    /// Time spent uploading, milliseconds.
    pub upload_ms: u64,
    /// FNV-1a 64 of every uploaded slice digest in name order.
    pub digest: u64,
}

/// The resident weight arena for one rank.
#[derive(Debug)]
pub struct ResidentArena {
    slices: BTreeMap<String, ArenaSlice>,
    groups: Vec<GroupArena>,
    allocated_bytes: u64,
    evidence: ArenaEvidence,
}

/// Supplies one planned tensor's stored bytes on demand.
///
/// The streaming builder calls this once per resident tensor and drops the returned buffer before
/// reading the next, so peak host memory is the arena plus one tensor rather than the whole set.
pub trait ShardReader {
    /// Reads exactly the bytes the plan marks resident for this tensor.
    fn read_shard(&mut self, planned: &ResidentShard) -> Result<Vec<u8>, String>;
}

impl ResidentArena {
    /// Allocates one slab per group and uploads every tensor once.
    ///
    /// Ordering is deterministic: groups in ResidentGroup order, tensors in the order the caller
    /// supplied them (the plan's name order). Each tensor starts at an ARENA_ALIGNMENT-byte
    /// boundary inside its group slab.
    ///
    /// # Errors
    /// A duplicate name, a zero-length tensor, a B12x geometry that does not match the bytes, an
    /// allocator or copy failure.
    pub fn build(
        allocator: &mut dyn ArenaAllocator,
        tensors: &[ArenaTensor<'_>],
        mode: AllocationMode,
    ) -> Result<Self, String> {
        let allocate_started = Instant::now();
        let mut by_group: BTreeMap<ResidentGroup, Vec<usize>> = BTreeMap::new();
        let mut payloads: Vec<Vec<u8>> = Vec::with_capacity(tensors.len());
        let mut names: BTreeMap<String, ()> = BTreeMap::new();
        for tensor in tensors {
            if tensor.bytes.is_empty() {
                return Err(format!("arena tensor {} has no bytes", tensor.name));
            }
            if names.insert(tensor.name.to_owned(), ()).is_some() {
                return Err(format!("arena tensor {} appears twice", tensor.name));
            }
            let payload = match (tensor.layout, tensor.b12x) {
                (TensorLayout::Stored, None) => tensor.bytes.to_vec(),
                (TensorLayout::B12x, Some((cols, inner))) => {
                    tile_b12x(tensor.bytes, cols, inner)
                        .map_err(|error| format!("arena tensor {}: {error}", tensor.name))?
                }
                (TensorLayout::B12x, None) => {
                    return Err(format!(
                        "arena tensor {} is B12x but has no (cols, inner) geometry",
                        tensor.name
                    ));
                }
                (TensorLayout::Stored, Some(_)) => {
                    return Err(format!(
                        "arena tensor {} is stored but has B12x geometry",
                        tensor.name
                    ));
                }
            };
            payloads.push(payload);
            by_group
                .entry(tensor.group)
                .or_default()
                .push(payloads.len() - 1);
        }
        let mut groups = Vec::with_capacity(by_group.len());
        let mut slabs: BTreeMap<ResidentGroup, (Deviceptr, Vec<usize>, usize)> = BTreeMap::new();
        let mut allocated_bytes = 0u64;
        for (group, entries) in &by_group {
            let mut offset = 0usize;
            let mut offsets = Vec::with_capacity(entries.len());
            let mut tiled = 0usize;
            for index in entries {
                offsets.push(offset);
                offset = align_up(offset + payloads[*index].len());
                if tensors[*index].layout == TensorLayout::B12x {
                    tiled += 1;
                }
            }
            let bytes = offset;
            let base = allocator
                .allocate(bytes)
                .map_err(|error| format!("arena group {}: {error}", group.as_str()))?;
            slabs.insert(*group, (base, offsets, bytes));
            allocated_bytes = allocated_bytes.saturating_add(bytes as u64);
            groups.push(GroupArena {
                group: *group,
                bytes,
                tensors: entries.len(),
                tiled,
                digest: FNV_OFFSET,
            });
        }
        let allocate_ms = elapsed_ms(allocate_started);
        let upload_started = Instant::now();
        let mut slices: BTreeMap<String, ArenaSlice> = BTreeMap::new();
        let mut payload_bytes = 0u64;
        let mut stored_bytes = 0u64;
        let mut tiled_tensors = 0usize;
        let mut digest = FNV_OFFSET;
        for (group, entries) in &by_group {
            let (base, offsets, _) = slabs
                .get(group)
                .ok_or_else(|| format!("arena group {} was not allocated", group.as_str()))?;
            let group_index = groups
                .iter()
                .position(|arena| arena.group == *group)
                .ok_or_else(|| {
                    format!(
                        "arena group {} is missing from the evidence",
                        group.as_str()
                    )
                })?;
            for (slot, index) in entries.iter().enumerate() {
                let tensor = &tensors[*index];
                let payload = &payloads[*index];
                let offset = offsets[slot];
                let ptr = base.offset(offset);
                allocator
                    .upload(&ptr, payload)
                    .map_err(|error| format!("arena upload {}: {error}", tensor.name))?;
                let slice_digest = fnv1a(payload);
                stored_bytes = stored_bytes.saturating_add(tensor.bytes.len() as u64);
                payload_bytes = payload_bytes.saturating_add(payload.len() as u64);
                if tensor.layout == TensorLayout::B12x {
                    tiled_tensors += 1;
                }
                digest ^= slice_digest;
                digest = digest.wrapping_mul(FNV_PRIME);
                let group_arena = groups.get_mut(group_index).ok_or_else(|| {
                    format!(
                        "arena group {} is missing from the evidence",
                        group.as_str()
                    )
                })?;
                group_arena.digest ^= slice_digest;
                group_arena.digest = group_arena.digest.wrapping_mul(FNV_PRIME);
                slices.insert(
                    tensor.name.to_owned(),
                    ArenaSlice {
                        name: tensor.name.to_owned(),
                        group: *group,
                        ptr,
                        offset,
                        bytes: payload.len(),
                        layout: tensor.layout,
                        digest: slice_digest,
                    },
                );
            }
        }
        let upload_ms = elapsed_ms(upload_started);
        Ok(Self {
            slices,
            groups,
            allocated_bytes,
            evidence: ArenaEvidence {
                mode,
                tensors: tensors.len(),
                tiled_tensors,
                stored_bytes,
                bytes: payload_bytes,
                allocated_bytes,
                allocate_ms,
                upload_ms,
                digest,
            },
        })
    }

    /// Builds the arena for a rank's resident plan from the slices ShardPlan::read_all read.
    ///
    /// # Errors
    /// A resident planned tensor with no read slice, a read slice the plan does not mark resident,
    /// a byte length that disagrees with the plan, or a build error.
    pub fn from_plan(
        allocator: &mut dyn ArenaAllocator,
        plan: &ResidentPlan,
        slices: &[StoredShard],
        mode: AllocationMode,
    ) -> Result<Self, String> {
        let by_name: BTreeMap<&str, &StoredShard> = slices
            .iter()
            .map(|slice| (slice.name.as_str(), slice))
            .collect();
        let mut tensors: Vec<ArenaTensor<'_>> = Vec::new();
        for planned in plan.shards() {
            if planned.placement != Placement::Resident {
                continue;
            }
            let slice = by_name.get(planned.name.as_str()).ok_or_else(|| {
                format!(
                    "the plan holds {} resident but the read has no slice for it",
                    planned.name
                )
            })?;
            if slice.bytes.len() as u64 != planned.shard_bytes {
                return Err(format!(
                    "{}: the plan expects {} bytes, the read has {}",
                    planned.name,
                    planned.shard_bytes,
                    slice.bytes.len()
                ));
            }
            let (layout, b12x) = match b12x_layout(planned) {
                Some((cols, inner)) => (TensorLayout::B12x, Some((cols, inner))),
                None => (TensorLayout::Stored, None),
            };
            tensors.push(ArenaTensor {
                name: planned.name.as_str(),
                group: planned.group,
                bytes: &slice.bytes,
                layout,
                b12x,
            });
        }
        Self::build(allocator, &tensors, mode)
    }

    /// Builds the arena by streaming: layout and allocate from the plan first (no bytes), then read,
    /// tile, upload and verify one tensors at a time, dropping each buffer before the next read.
    ///
    /// The watch callback runs before each read with the tensor name, so the caller can enforce a
    /// host-memory floor and abort cleanly instead of being OOM-killed.
    ///
    /// # Errors
    /// A plan/read disagreement, an allocator/copy failure, a digest mismatch, or a watch failure.
    pub fn from_plan_streaming(
        allocator: &mut dyn ArenaAllocator,
        plan: &ResidentPlan,
        reader: &mut dyn ShardReader,
        mode: AllocationMode,
        mut watch: impl FnMut(&str) -> Result<(), String>,
    ) -> Result<Self, String> {
        use std::collections::BTreeMap;
        struct Slot {
            planned: usize,
            name: String,
            group: ResidentGroup,
            stored: u64,
            layout: TensorLayout,
            b12x: Option<(usize, usize)>,
            payload: usize,
        }
        let planned_shards = plan.shards();
        let mut slots: Vec<Slot> = Vec::new();
        for (index, planned) in planned_shards.iter().enumerate() {
            if planned.placement != Placement::Resident {
                continue;
            }
            let (layout, b12x) = match b12x_layout(planned) {
                Some((cols, inner)) => (TensorLayout::B12x, Some((cols, inner))),
                None => (TensorLayout::Stored, None),
            };
            let payload = match b12x {
                Some((cols, inner)) => b12x_buffer_bytes(cols, inner),
                None => usize::try_from(planned.shard_bytes)
                    .map_err(|_| format!("{}: shard bytes do not fit usize", planned.name))?,
            };
            slots.push(Slot {
                planned: index,
                name: planned.name.clone(),
                group: planned.group,
                stored: planned.shard_bytes,
                layout,
                b12x,
                payload,
            });
        }
        let mut by_group: BTreeMap<ResidentGroup, Vec<usize>> = BTreeMap::new();
        for (index, slot) in slots.iter().enumerate() {
            by_group.entry(slot.group).or_default().push(index);
        }
        let allocate_started = Instant::now();
        let mut groups: Vec<GroupArena> = Vec::with_capacity(by_group.len());
        let mut slabs: BTreeMap<ResidentGroup, (Deviceptr, Vec<usize>, usize)> = BTreeMap::new();
        let mut allocated_bytes = 0u64;
        // Layout every group first: offsets inside the group, and the group's size.
        let mut layout: Vec<(ResidentGroup, Vec<usize>, usize, usize)> =
            Vec::with_capacity(by_group.len());
        for (group, entries) in &by_group {
            let mut offset = 0usize;
            let mut offsets = Vec::with_capacity(entries.len());
            let mut tiled = 0usize;
            for index in entries {
                offsets.push(offset);
                offset = align_up(offset + slots[*index].payload);
                if slots[*index].layout == TensorLayout::B12x {
                    tiled += 1;
                }
            }
            layout.push((*group, offsets, offset, tiled));
        }
        if mode == AllocationMode::UnifiedSingle {
            // One reservation for the whole plan: the resident bytes are known before any read, so
            // the whole arena is allocated up front and every group gets a base inside it. This is
            // the four-rank arm's request: one chunk for the model, not one allocation per read.
            let total: usize = layout.iter().map(|(_, _, size, _)| align_up(*size)).sum();
            let base = allocator
                .allocate(total)
                .map_err(|error| format!("arena single slab ({total} bytes): {error}"))?;
            let mut cursor = 0usize;
            for (group, offsets, size, tiled) in &layout {
                let group_base = base.offset(cursor);
                slabs.insert(*group, (group_base, offsets.clone(), *size));
                groups.push(GroupArena {
                    group: *group,
                    bytes: *size,
                    tensors: offsets.len(),
                    tiled: *tiled,
                    digest: FNV_OFFSET,
                });
                cursor = align_up(cursor + *size);
            }
            allocated_bytes = total as u64;
        } else {
            for (group, offsets, size, tiled) in &layout {
                let base = allocator
                    .allocate(*size)
                    .map_err(|error| format!("arena group {}: {error}", group.as_str()))?;
                slabs.insert(*group, (base, offsets.clone(), *size));
                allocated_bytes = allocated_bytes.saturating_add(*size as u64);
                groups.push(GroupArena {
                    group: *group,
                    bytes: *size,
                    tensors: offsets.len(),
                    tiled: *tiled,
                    digest: FNV_OFFSET,
                });
            }
        }
        let allocate_ms = elapsed_ms(allocate_started);
        let upload_started = Instant::now();
        let mut slices: BTreeMap<String, ArenaSlice> = BTreeMap::new();
        let mut payload_bytes = 0u64;
        let mut stored_bytes = 0u64;
        let mut tiled_tensors = 0usize;
        let mut digest = FNV_OFFSET;
        for (group, entries) in &by_group {
            let (base, offsets, _) = slabs
                .get(group)
                .ok_or_else(|| format!("arena group {} was not allocated", group.as_str()))?;
            let group_index = groups
                .iter()
                .position(|arena| arena.group == *group)
                .ok_or_else(|| {
                    format!(
                        "arena group {} is missing from the evidence",
                        group.as_str()
                    )
                })?;
            for (slot_index, index) in entries.iter().enumerate() {
                let slot = &slots[*index];
                watch(&slot.name)?;
                let planned = planned_shards
                    .get(slot.planned)
                    .ok_or_else(|| format!("{}: planned shard vanished", slot.name))?;
                let raw = reader.read_shard(planned)?;
                if raw.len() as u64 != slot.stored {
                    return Err(format!(
                        "{}: the plan expects {} stored bytes, the reader returned {}",
                        slot.name,
                        slot.stored,
                        raw.len()
                    ));
                }
                let payload = match slot.b12x {
                    Some((cols, inner)) => tile_b12x(&raw, cols, inner)
                        .map_err(|error| format!("{}: {error}", slot.name))?,
                    None => raw,
                };
                if payload.len() != slot.payload {
                    return Err(format!(
                        "{}: payload is {} bytes, the layout reserved {}",
                        slot.name,
                        payload.len(),
                        slot.payload
                    ));
                }
                let offset = offsets[slot_index];
                let ptr = base.offset(offset);
                allocator
                    .upload(&ptr, &payload)
                    .map_err(|error| format!("arena upload {}: {error}", slot.name))?;
                let slice_digest = fnv1a(&payload);
                stored_bytes = stored_bytes.saturating_add(slot.stored);
                payload_bytes = payload_bytes.saturating_add(payload.len() as u64);
                if slot.layout == TensorLayout::B12x {
                    tiled_tensors += 1;
                }
                digest ^= slice_digest;
                digest = digest.wrapping_mul(FNV_PRIME);
                if let Some(group_arena) = groups.get_mut(group_index) {
                    group_arena.digest ^= slice_digest;
                    group_arena.digest = group_arena.digest.wrapping_mul(FNV_PRIME);
                }
                slices.insert(
                    slot.name.clone(),
                    ArenaSlice {
                        name: slot.name.clone(),
                        group: slot.group,
                        ptr,
                        offset,
                        bytes: payload.len(),
                        layout: slot.layout,
                        digest: slice_digest,
                    },
                );
            }
        }
        let upload_ms = elapsed_ms(upload_started);
        Ok(Self {
            slices,
            groups,
            allocated_bytes,
            evidence: ArenaEvidence {
                mode,
                tensors: slots.len(),
                tiled_tensors,
                stored_bytes,
                bytes: payload_bytes,
                allocated_bytes,
                allocate_ms,
                upload_ms,
                digest,
            },
        })
    }

    /// One tensor's device slice.
    #[must_use]
    pub fn slice(&self, name: &str) -> Option<&ArenaSlice> {
        self.slices.get(name)
    }

    /// One tensor's device pointer.
    #[must_use]
    pub fn device_ptr(&self, name: &str) -> Option<Deviceptr> {
        self.slices.get(name).map(|slice| slice.ptr)
    }

    /// The load-time evidence.
    #[must_use]
    pub fn evidence(&self) -> ArenaEvidence {
        self.evidence
    }

    /// The group slabs.
    #[must_use]
    pub fn groups(&self) -> &[GroupArena] {
        &self.groups
    }

    /// The allocated bytes, including padding.
    #[must_use]
    pub fn allocated_bytes(&self) -> u64 {
        self.allocated_bytes
    }

    /// How many tensors the arena holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.slices.len()
    }

    /// Whether the arena holds no tensor.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.slices.is_empty()
    }

    /// Asserts that the bytes a launch is about to bind are the bytes the arena uploaded.
    ///
    /// The wiring calls this once per tensor before the first launch. A resident slice whose
    /// digest is not the uploaded digest means a wrong pointer or a wrong slice, which must fail
    /// loudly rather than multiply a weight that belongs to another tensor.
    ///
    /// # Errors
    /// The name is not in the arena, or the bytes differ from the uploaded slice.
    pub fn verify_slice(&self, name: &str, bytes: &[u8]) -> Result<(), String> {
        let slice = self
            .slices
            .get(name)
            .ok_or_else(|| format!("the arena has no tensor {name}"))?;
        if slice.bytes != bytes.len() {
            return Err(format!(
                "{name}: the arena holds {} bytes, the launch expects {}",
                slice.bytes,
                bytes.len()
            ));
        }
        let digest = fnv1a(bytes);
        if digest != slice.digest {
            return Err(format!(
                "{name}: the bytes differ from the uploaded slice ({digest:#018x} vs {:#018x})",
                slice.digest
            ));
        }
        Ok(())
    }
}

/// Whether a resident tensor is consumed in B12x order, and the (cols, inner) geometry to tile.
///
/// The B12x kernels walk dense FP8 weights as a micro-tile over the output rows (cols) and the
/// reduction length (inner). The attention projections and the shared expert are the resident
/// groups whose device kernels are B12x today (the routed experts are FP4 and the norms, sinks and
/// router are not B12x). The scales are already in B12x order and keep their stored bytes.
///
/// The choice keys off the name and the dtype, not only the group, so a future FP8 tensor in a B12x
/// group cannot be tiled blindly: its name must be a .weight and its dtype must be F8_E4M3.
#[must_use]
pub fn b12x_layout(planned: &ResidentShard) -> Option<(usize, usize)> {
    if planned.dtype != "F8_E4M3" || !planned.name.ends_with(".weight") {
        return None;
    }
    let tiled_group = matches!(
        planned.group,
        ResidentGroup::AttentionReplicated
            | ResidentGroup::AttentionColumn
            | ResidentGroup::AttentionRow
            | ResidentGroup::SharedColumn
            | ResidentGroup::SharedRow
    );
    if !tiled_group {
        return None;
    }
    match planned.shard_shape.as_slice() {
        [cols, inner] if *cols > 0 && *inner > 0 => Some((*cols, *inner)),
        _ => None,
    }
}

/// Repacks one stored FP8 weight into the B12x micro-tile order the dense kernel walks.
///
/// Permutation only: every input byte appears exactly once in the padded output. The padding is
/// zero, matching b12x_buffer_bytes, and the scales are NOT part of this transform.
///
/// # Errors
/// The bytes are not exactly cols * inner.
pub fn tile_b12x(weights: &[u8], cols: usize, inner: usize) -> Result<Vec<u8>, String> {
    let expected = cols.saturating_mul(inner);
    if weights.len() != expected {
        return Err(format!(
            "expected {expected} E4M3 bytes for {cols} x {inner}, got {}",
            weights.len()
        ));
    }
    // One tiling implementation: the kernel crate owns the index map (kernel-dev 4759287), so
    // this wrapper only enforces the geometry before delegating.
    let tiled = sparkflash_kernels::mxfp8_ref::b12x_tile_weights(weights, cols, inner);
    if tiled.len() != b12x_buffer_bytes(cols, inner) {
        return Err(format!(
            "tile_b12x({cols}, {inner}) returned {} bytes, expected {}",
            tiled.len(),
            b12x_buffer_bytes(cols, inner)
        ));
    }
    Ok(tiled)
}

/// The real allocator: device memory through Cuda, freed on drop.
pub struct CudaArenaAllocator<'a> {
    cuda: &'a Cuda,
    mode: AllocationMode,
    pointers: Vec<Deviceptr>,
}

impl<'a> CudaArenaAllocator<'a> {
    /// A plain-device allocator.
    #[must_use]
    pub fn new(cuda: &'a Cuda) -> Self {
        Self {
            cuda,
            mode: AllocationMode::Device,
            pointers: Vec::new(),
        }
    }

    /// An allocator for one mode. AllocationMode::Unified allocates UVA-coherent managed memory
    /// (cuMemAllocManaged, CU_MEM_ATTACH_GLOBAL) through the kernel driver wrapper.
    /// managed-memory driver method lands.
    #[must_use]
    pub fn with_mode(cuda: &'a Cuda, mode: AllocationMode) -> Self {
        Self {
            cuda,
            mode,
            pointers: Vec::new(),
        }
    }

    /// The mode this allocator serves.
    #[must_use]
    pub fn mode(&self) -> AllocationMode {
        self.mode
    }
}

impl ArenaAllocator for CudaArenaAllocator<'_> {
    fn allocate(&mut self, bytes: usize) -> Result<Deviceptr, String> {
        let mut total = 0usize;
        let pointer = match self.mode {
            AllocationMode::Device => self.cuda.alloc(bytes, &mut total)?,
            AllocationMode::Unified | AllocationMode::UnifiedSingle => {
                self.cuda.alloc_managed(bytes, &mut total)?
            }
        };
        self.pointers.push(pointer);
        Ok(pointer)
    }

    fn upload(&mut self, destination: &Deviceptr, bytes: &[u8]) -> Result<(), String> {
        self.cuda.copy_htod(destination, bytes)
    }
}

impl Drop for CudaArenaAllocator<'_> {
    fn drop(&mut self) {
        if self.pointers.is_empty() {
            return;
        }
        match self.mode {
            AllocationMode::Device => {
                let references: Vec<&Deviceptr> = self.pointers.iter().collect();
                self.cuda.free_buffers(&references);
            }
            AllocationMode::Unified | AllocationMode::UnifiedSingle => {
                for pointer in &self.pointers {
                    self.cuda.free_managed(pointer);
                }
            }
        }
    }
}

/// Rounds a byte count up to ARENA_ALIGNMENT.
fn align_up(bytes: usize) -> usize {
    match bytes % ARENA_ALIGNMENT {
        0 => bytes,
        remainder => bytes + (ARENA_ALIGNMENT - remainder),
    }
}

/// Whole milliseconds of an elapsed instant.
fn elapsed_ms(started: Instant) -> u64 {
    started.elapsed().as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Records every allocation and upload so a host test can check the layout without a driver.
    #[derive(Default)]
    struct FakeAllocator {
        next: u64,
        allocations: Vec<usize>,
        uploaded: Vec<(u64, Vec<u8>)>,
        fail_upload: Option<usize>,
    }

    impl FakeAllocator {
        fn new() -> Self {
            Self {
                next: 0x1_0000,
                ..Self::default()
            }
        }
    }

    impl ArenaAllocator for FakeAllocator {
        fn allocate(&mut self, bytes: usize) -> Result<Deviceptr, String> {
            let pointer = Deviceptr(self.next);
            self.next += bytes as u64;
            self.allocations.push(bytes);
            Ok(pointer)
        }

        fn upload(&mut self, destination: &Deviceptr, bytes: &[u8]) -> Result<(), String> {
            if self.fail_upload == Some(self.uploaded.len()) {
                return Err("injected upload failure".to_string());
            }
            self.uploaded.push((destination.0, bytes.to_vec()));
            Ok(())
        }
    }

    fn tensor<'a>(
        name: &'a str,
        group: ResidentGroup,
        bytes: &'a [u8],
        layout: TensorLayout,
        b12x: Option<(usize, usize)>,
    ) -> ArenaTensor<'a> {
        ArenaTensor {
            name,
            group,
            bytes,
            layout,
            b12x,
        }
    }

    #[test]
    fn one_slab_per_group_with_aligned_offsets() {
        let mut allocator = FakeAllocator::new();
        let a = vec![1u8; 300];
        let b = vec![2u8; 10];
        let c = vec![3u8; 300];
        let arena = ResidentArena::build(
            &mut allocator,
            &[
                tensor(
                    "layers.0.attn.wq_a.weight",
                    ResidentGroup::AttentionReplicated,
                    &a,
                    TensorLayout::Stored,
                    None,
                ),
                tensor(
                    "layers.0.attn.wkv.weight",
                    ResidentGroup::AttentionReplicated,
                    &b,
                    TensorLayout::Stored,
                    None,
                ),
                tensor(
                    "layers.0.ffn.gate.weight",
                    ResidentGroup::Router,
                    &c,
                    TensorLayout::Stored,
                    None,
                ),
            ],
            AllocationMode::Device,
        )
        .expect("arena");
        assert_eq!(allocator.allocations.len(), 2, "one slab per group");
        assert_eq!(
            allocator.allocations[0], 768,
            "300 rounded to 512, + 10 more, rounded to 768"
        );
        assert_eq!(allocator.allocations[1], 512, "300 rounded up to 256");
        assert_eq!(arena.len(), 3);
        let first = arena.slice("layers.0.attn.wq_a.weight").expect("first");
        let second = arena.slice("layers.0.attn.wkv.weight").expect("second");
        assert_eq!(first.offset, 0);
        assert_eq!(
            second.offset, 512,
            "the second tensor starts on an alignment boundary"
        );
        assert_eq!(second.ptr, first.ptr.offset(512));
        assert_eq!(arena.evidence().stored_bytes, 610);
        assert_eq!(arena.evidence().bytes, 610);
        assert_eq!(arena.evidence().allocated_bytes, 1280);
        assert_eq!(arena.evidence().mode, AllocationMode::Device);
        assert_eq!(allocator.uploaded.len(), 3);
    }

    #[test]
    fn the_uploaded_bytes_are_the_tensor_bytes_and_verify() {
        let mut allocator = FakeAllocator::new();
        let weight = vec![7u8; 128];
        let arena = ResidentArena::build(
            &mut allocator,
            &[tensor(
                "w",
                ResidentGroup::Norms,
                &weight,
                TensorLayout::Stored,
                None,
            )],
            AllocationMode::Device,
        )
        .expect("arena");
        assert_eq!(allocator.uploaded[0].1, weight);
        arena
            .verify_slice("w", &weight)
            .expect("the same bytes verify");
        let mut other = weight.clone();
        other[0] ^= 0xff;
        assert!(
            arena.verify_slice("w", &other).is_err(),
            "a different weight must not verify"
        );
        assert!(arena.verify_slice("missing", &weight).is_err());
    }

    #[test]
    fn a_duplicate_name_is_refused() {
        let mut allocator = FakeAllocator::new();
        let bytes = vec![1u8; 4];
        let duplicate = ResidentArena::build(
            &mut allocator,
            &[
                tensor(
                    "same",
                    ResidentGroup::Norms,
                    &bytes,
                    TensorLayout::Stored,
                    None,
                ),
                tensor(
                    "same",
                    ResidentGroup::Norms,
                    &bytes,
                    TensorLayout::Stored,
                    None,
                ),
            ],
            AllocationMode::Device,
        );
        assert!(
            duplicate.is_err(),
            "a name that appears twice must be refused"
        );
        let empty = ResidentArena::build(
            &mut allocator,
            &[tensor(
                "empty",
                ResidentGroup::Norms,
                &[],
                TensorLayout::Stored,
                None,
            )],
            AllocationMode::Device,
        );
        assert!(empty.is_err());
    }

    #[test]
    fn b12x_tiling_is_a_bijection_and_is_uploaded_padded() {
        // 64 x 96 E4M3 bytes: both dimensions are multiples of 32, so no padding is needed and
        // every byte must appear once.
        let cols = 64usize;
        let inner = 96usize;
        let raw: Vec<u8> = (0..cols * inner).map(|index| (index % 251) as u8).collect();
        let tiled = tile_b12x(&raw, cols, inner).expect("tile");
        assert_eq!(tiled.len(), b12x_buffer_bytes(cols, inner));
        let mut sorted_raw = raw.clone();
        sorted_raw.sort_unstable();
        let mut sorted_tiled = tiled.clone();
        sorted_tiled.sort_unstable();
        assert_eq!(sorted_raw, sorted_tiled, "permutation only");
        let mut allocator = FakeAllocator::new();
        let arena = ResidentArena::build(
            &mut allocator,
            &[tensor(
                "layers.0.attn.wq_b.weight",
                ResidentGroup::AttentionColumn,
                &raw,
                TensorLayout::B12x,
                Some((cols, inner)),
            )],
            AllocationMode::Device,
        )
        .expect("arena");
        assert_eq!(arena.evidence().tiled_tensors, 1);
        assert_eq!(allocator.uploaded[0].1.len(), tiled.len());
        let slice = arena.slice("layers.0.attn.wq_b.weight").expect("slice");
        assert_eq!(slice.layout, TensorLayout::B12x);
        assert_eq!(slice.bytes, tiled.len());
    }

    #[test]
    fn a_b12x_geometry_mismatch_is_refused() {
        let mut allocator = FakeAllocator::new();
        let raw = vec![0u8; 100];
        let mismatch = ResidentArena::build(
            &mut allocator,
            &[tensor(
                "layers.0.attn.wq_b.weight",
                ResidentGroup::AttentionColumn,
                &raw,
                TensorLayout::B12x,
                Some((16, 8)),
            )],
            AllocationMode::Device,
        );
        assert!(mismatch.is_err());
        let no_geometry = ResidentArena::build(
            &mut allocator,
            &[tensor(
                "layers.0.attn.wq_b.weight",
                ResidentGroup::AttentionColumn,
                &raw,
                TensorLayout::B12x,
                None,
            )],
            AllocationMode::Device,
        );
        assert!(no_geometry.is_err());
    }

    #[test]
    fn unified_mode_is_recorded_by_the_builder() {
        let mut allocator = FakeAllocator::new();
        let bytes = vec![0u8; 8];
        let arena = ResidentArena::build(
            &mut allocator,
            &[tensor(
                "x",
                ResidentGroup::Embedding,
                &bytes,
                TensorLayout::Stored,
                None,
            )],
            AllocationMode::Unified,
        )
        .expect("the fake allocator does not enforce the mode");
        assert_eq!(arena.evidence().mode, AllocationMode::Unified);
    }

    #[test]
    fn an_upload_failure_names_the_tensor() {
        let mut allocator = FakeAllocator {
            fail_upload: Some(0),
            ..FakeAllocator::new()
        };
        let bytes = vec![0u8; 8];
        let error = ResidentArena::build(
            &mut allocator,
            &[tensor(
                "layers.0.attn.wq_a.weight",
                ResidentGroup::AttentionReplicated,
                &bytes,
                TensorLayout::Stored,
                None,
            )],
            AllocationMode::Device,
        )
        .expect_err("the upload must fail");
        assert!(error.contains("layers.0.attn.wq_a.weight"), "{error}");
    }

    #[test]
    fn the_b12x_layout_policy_tiles_only_dense_fp8_weights() {
        let planned =
            |name: &str, dtype: &str, group: ResidentGroup, shape: &[usize]| ResidentShard {
                name: name.to_string(),
                group,
                placement: Placement::Resident,
                rule: Some(crate::shard::ShardRule::Column),
                dtype: dtype.to_string(),
                stored_shape: shape.to_vec(),
                stored_bytes: 0,
                shard_bytes: 0,
                shard_shape: shape.to_vec(),
                ranges: 0,
            };
        assert_eq!(
            b12x_layout(&planned(
                "layers.0.attn.wq_b.weight",
                "F8_E4M3",
                ResidentGroup::AttentionColumn,
                &[64, 32]
            )),
            Some((64, 32))
        );
        assert_eq!(
            b12x_layout(&planned(
                "layers.0.ffn.shared_experts.w1.weight",
                "F8_E4M3",
                ResidentGroup::SharedColumn,
                &[32, 64]
            )),
            Some((32, 64))
        );
        assert_eq!(
            b12x_layout(&planned(
                "layers.0.attn.wq_b.scale",
                "F8_E8M0",
                ResidentGroup::AttentionColumn,
                &[2, 1]
            )),
            None,
            "scales are already in B12x order and are never transformed"
        );
        assert_eq!(
            b12x_layout(&planned(
                "layers.0.ffn.experts.w1.weight",
                "I8",
                ResidentGroup::ExpertsColumn,
                &[32, 64]
            )),
            None,
            "the routed experts are FP4, not B12x"
        );
    }
}
