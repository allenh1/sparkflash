//! sm_121a CUDA kernels and their CPU reference implementations.
//!
//! Milestone 4 fills in the remaining kernels themselves (Mega-mHC, MXFP8 dense, MXFP4 MoE,
//! FlashMLA CSA2 and the certified LM head). What exists today is the shared trace ABI, the
//! RoCEnante one-shot All-Reduce slot ABI and kernel from Milestone 3, and the CPU reference
//! for that collective's numerics, plus the checks that keep the Rust and CUDA halves of every
//! ABI in lockstep.

pub mod abi;
pub mod roce_abi;
pub mod roce_reduce;
