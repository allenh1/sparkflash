//! placeholder - implemented in Milestone 3
/// Largest payload the RoCEnante one-shot path carries: one 2 MiB RDMA arena slot per rank.
/// A collective larger than this falls back to NCCL.
pub const ROCENANTE_MAX_BYTES: usize = 2 * 1024 * 1024;
