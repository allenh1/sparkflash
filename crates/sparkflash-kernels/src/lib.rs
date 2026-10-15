//! sm_121a CUDA kernels and their CPU reference implementations.
//!
//! Milestone 4 fills in the kernels themselves (Mega-mHC, MXFP8 dense, MXFP4 MoE, FlashMLA
//! CSA2 and the certified LM head). What exists from Milestone 1 is the shared trace ABI and
//! the checks that keep the Rust and CUDA halves of it in lockstep.

pub mod abi;
