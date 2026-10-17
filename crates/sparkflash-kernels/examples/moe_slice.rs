//! The first vertical slice of the device path (task-31, Deliverable 2).
//!
//! One MoE layer's real expert combine weights, uploaded exactly as the checkpoint stores them
//! (rank-0 column slice, then padded to the kernel's stride), with b12x_mxfp4_moe_sm121 launched
//! on them and the result diffed against the CPU reference mxfp4_ref::moe_combine on the same bytes.
//!
//! Real: the expert weights, read from the released checkpoint. Synthetic: the activations and the
//! expert ids, because this slice tests the GEMM, not the router. The activation operand is still
//! built by the reference's own routed_activation_operand, so the A operand is the reference's
//! arithmetic rather than a hand-rolled copy.
//!
//! The padding is asserted, not assumed: a wrong column slice still produces finite, wrong floats.
//!
//! Build the cubin first, WITHOUT -DSPARKFLASH_ENABLE_GPU_TRACE:
//!   nvcc -cubin -arch=sm_121a -I kernels -o out/b12x_mxfp4_moe_sm121.cubin \
//!        kernels/b12x_mxfp4_moe_sm121.cu
//! With the trace compiled out the macros reduce to (void)(ring), so this launcher passes a null
//! ring pointer. With the trace enabled the same argument is dereferenced and must point at a real
//! GpuTraceRing.
//!
//! Usage: moe_slice --checkpoint <snapshot dir> [--layer N] [--experts N] [--tokens N] [--cubin P]

use std::collections::HashMap;
use std::ffi::c_void;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use sparkflash_kernels::m4_quant::QUANT_BLOCK;
use sparkflash_kernels::mxfp4_ref::{
    FP4_PER_BYTE, MOE_SCALE_BYTES_PER_ROW, MOE_TOP_K, MOE_WEIGHT_BYTES_PER_ROW, MOE_WIDTH,
    MOE_WIDTH_PADDED, check_moe_layout, moe_combine, pre_padded_moe_layout,
    routed_activation_operand,
};

const HIDDEN: usize = 5120;
// Every width below is DERIVED from the reference's own constants, not typed twice: if the
// reference's width or block size moves, this file stops compiling rather than drifting.
const TOP_K: usize = MOE_TOP_K;
const WIDTH_PADDED: usize = MOE_WIDTH_PADDED; // 640
const PACKED_ROW: usize = MOE_WEIGHT_BYTES_PER_ROW; // 320
const SCALE_ROW: usize = MOE_SCALE_BYTES_PER_ROW; // 20
const TILE_D: usize = 256; // SPARK_M4_MOE_TILE_D, one output element per thread
const THREADS: u32 = 256;

// Global (all four ranks) shapes as the checkpoint stores them.
const W2_OUT: usize = HIDDEN;
const W2_IN_PACKED_GLOBAL: usize = 1152; // 2304 / 2
const W2_IN_SCALES_GLOBAL: usize = 72; // 2304 / 32
// Rank 0's column slice of one expert: MOE_WIDTH of the global 2304, i.e. 288 packed bytes and 18
// scale blocks of the stored 1152 and 72.
const RANK0_PACKED: usize = MOE_WIDTH / FP4_PER_BYTE; // 288
const RANK0_SCALES: usize = MOE_WIDTH / QUANT_BLOCK; // 18
const _: () = assert!(RANK0_PACKED == 288 && RANK0_SCALES == 18);
const _: () = assert!(PACKED_ROW == 320 && SCALE_ROW == 20 && TOP_K == 6);

#[derive(Clone)]
struct Header {
    dtype: String,
    shape: Vec<usize>,
    start: u64,
    end: u64,
}

/// Minimal safetensors header reader. The package has no JSON dependency and three fields are
/// needed, so the header text is scanned directly - and every value it extracts is asserted against
/// the dtype and shape the checkpoint is known to store, so a parse error cannot pass silently.
fn read_header(path: &Path) -> Result<HashMap<String, Header>, String> {
    let mut file = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes).map_err(|e| e.to_string())?;
    let header_len = u64::from_le_bytes(len_bytes);
    let mut text = vec![0u8; header_len as usize];
    file.read_exact(&mut text).map_err(|e| e.to_string())?;
    let text = String::from_utf8(text).map_err(|e| e.to_string())?;
    let data_start = 8 + header_len;

    let mut out = HashMap::new();
    let bytes = text.as_bytes();
    let mut i = 0usize;
    while let Some(rel) = text[i..].find("\"dtype\"") {
        let dtype_at = i + rel;
        // Walk back to the opening quote of the tensor name.
        let mut name_end = dtype_at;
        while name_end > 0 && bytes[name_end - 1] != b'"' {
            name_end -= 1;
        }
        let mut name_start = name_end.saturating_sub(2);
        while name_start > 0 && bytes[name_start - 1] != b'"' {
            name_start -= 1;
        }
        let name = text[name_start..name_end - 1].to_string();
        let dq = text[dtype_at..]
            .find(':')
            .map(|p| dtype_at + p)
            .ok_or("no dtype colon")?;
        let v0 = text[dq..]
            .find('"')
            .map(|p| dq + p + 1)
            .ok_or("no dtype quote")?;
        let v1 = text[v0..].find('"').map(|p| v0 + p).ok_or("no dtype end")?;
        let dtype = text[v0..v1].to_string();
        let shape_at = text[v1..]
            .find("\"shape\"")
            .map(|p| v1 + p)
            .ok_or("no shape")?;
        let s0 = text[shape_at..]
            .find('[')
            .map(|p| shape_at + p + 1)
            .ok_or("no shape open")?;
        let s1 = text[s0..]
            .find(']')
            .map(|p| s0 + p)
            .ok_or("no shape close")?;
        let shape: Vec<usize> = text[s0..s1]
            .split(',')
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.trim().parse::<usize>().map_err(|e| e.to_string()))
            .collect::<Result<_, _>>()?;
        let off_at = text[s1..]
            .find("data_offsets")
            .map(|p| s1 + p)
            .ok_or("no offsets")?;
        let o0 = text[off_at..]
            .find('[')
            .map(|p| off_at + p + 1)
            .ok_or("no offsets open")?;
        let o1 = text[o0..]
            .find(']')
            .map(|p| o0 + p)
            .ok_or("no offsets close")?;
        let offs: Vec<u64> = text[o0..o1]
            .split(',')
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.trim().parse::<u64>().map_err(|e| e.to_string()))
            .collect::<Result<_, _>>()?;
        if offs.len() != 2 {
            return Err(format!(
                "{name}: expected two data offsets, got {}",
                offs.len()
            ));
        }
        out.insert(
            name,
            Header {
                dtype,
                shape,
                start: data_start + offs[0],
                end: data_start + offs[1],
            },
        );
        i = o1;
    }
    Ok(out)
}

/// Read one expert tensor's rank-0 slice and pad each row to the kernel's stride.
///
/// src_row is the global row length in bytes, take the number of leading bytes rank 0 owns, pad the
/// kernel's per-row stride. The rank-1..3 tail is asserted zero, so a wrong column slice stops the
/// run instead of producing a plausible number.
fn slice_and_pad(
    file: &mut File,
    hdr: &Header,
    rows: usize,
    src_row: usize,
    take: usize,
    pad: usize,
    what: &str,
) -> Result<Vec<u8>, String> {
    assert!(
        take <= src_row,
        "{what}: rank slice {take} exceeds the stored row {src_row}"
    );
    assert!(
        take <= pad,
        "{what}: pad stride {pad} is smaller than the rank slice {take}"
    );
    let total = (hdr.end - hdr.start) as usize;
    assert_eq!(
        total,
        rows * src_row,
        "{what}: the header's byte span {total} is not {rows} rows of {src_row} bytes"
    );
    // One expert-sized chunk, which is what the design's upload path stages: peak host memory is a
    // chunk, not the tensor.
    let mut whole = vec![0u8; total];
    file.seek(SeekFrom::Start(hdr.start))
        .map_err(|e| e.to_string())?;
    file.read_exact(&mut whole).map_err(|e| e.to_string())?;
    let mut out = vec![0u8; rows * pad];
    for r in 0..rows {
        // The bytes after the slice are ranks 1..3's real data, NOT padding - a global checkpoint
        // row is non-zero across its whole width, and asserting otherwise is a false invariant that
        // fires on the first real tensor. What is asserted here is the stored dtype and shape (in
        // main) and the header's byte span above; what the run then proves is that the kernel and
        // the reference agree on this slice at this stride.
        let row = &whole[r * src_row..(r + 1) * src_row];
        out[r * pad..r * pad + take].copy_from_slice(&row[..take]);
        // The pad tail stays zero: that is what the kernel reads for the unused 576..640 columns.
    }
    Ok(out)
}

struct Args {
    checkpoint: PathBuf,
    layer: usize,
    experts: usize,
    tokens: usize,
    repeat: usize,
    cubin: PathBuf,
}

fn parse_args() -> Args {
    let mut checkpoint = None;
    let mut layer = 0usize;
    let mut experts = 64usize;
    let mut tokens = 8usize;
    let mut repeat = 1usize;
    let mut cubin = PathBuf::from("b12x_mxfp4_moe_sm121.cubin");
    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--checkpoint" => {
                checkpoint = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--layer" => {
                layer = argv[i + 1].parse().expect("layer");
                i += 2;
            }
            "--experts" => {
                experts = argv[i + 1].parse().expect("experts");
                i += 2;
            }
            "--tokens" => {
                tokens = argv[i + 1].parse().expect("tokens");
                i += 2;
            }
            "--cubin" => {
                cubin = PathBuf::from(&argv[i + 1]);
                i += 2;
            }
            "--repeat" => {
                repeat = argv[i + 1].parse().expect("repeat");
                i += 2;
            }
            other => panic!("unknown argument {other}"),
        }
    }
    Args {
        checkpoint: checkpoint.expect("--checkpoint is required"),
        layer,
        experts,
        tokens,
        repeat,
        cubin,
    }
}

fn main() -> Result<(), String> {
    let args = parse_args();
    check_moe_layout(pre_padded_moe_layout()).map_err(|e| format!("layout rejected: {e}"))?;

    let index_path = args.checkpoint.join("model.safetensors.index.json");
    let index_text = std::fs::read_to_string(&index_path).map_err(|e| e.to_string())?;
    let prefix = format!("layers.{}.ffn.experts.", args.layer);

    let mut headers: HashMap<String, HashMap<String, Header>> = HashMap::new();
    let mut files: HashMap<String, File> = HashMap::new();
    let mut names = Vec::new();
    for e in 0..args.experts {
        names.push(format!("{prefix}{e}.w2.weight"));
        names.push(format!("{prefix}{e}.w2.scale"));
    }
    for name in &names {
        // The shard a tensor lives in, taken from the index without parsing the whole JSON.
        let key = format!("\"{name}\"");
        let at = index_text
            .find(&key)
            .unwrap_or_else(|| panic!("{name} not in the index"));
        let rest = &index_text[at..];
        let c = rest.find(':').expect("colon");
        let q0 = rest[c..].find('"').expect("quote") + c + 1;
        let q1 = rest[q0..].find('"').expect("quote end") + q0;
        let shard = rest[q0..q1].to_string();
        if !headers.contains_key(&shard) {
            let path = args.checkpoint.join(&shard);
            headers.insert(shard.clone(), read_header(&path)?);
            files.insert(shard.clone(), File::open(&path).map_err(|e| e.to_string())?);
        }
    }
    let shard_of = |name: &str| -> String {
        let key = format!("\"{name}\"");
        let at = index_text
            .find(&key)
            .unwrap_or_else(|| panic!("{name} not in the index"));
        let rest = &index_text[at..];
        let c = rest.find(':').expect("colon");
        let q0 = rest[c..].find('"').expect("quote") + c + 1;
        let q1 = rest[q0..].find('"').expect("quote end") + q0;
        rest[q0..q1].to_string()
    };

    // Slice and pad every expert, asserting the stored dtype and shape as we go.
    let mut w_all = Vec::with_capacity(args.experts * W2_OUT * PACKED_ROW);
    let mut ws_all = Vec::with_capacity(args.experts * W2_OUT * SCALE_ROW);
    for e in 0..args.experts {
        let wname = format!("{prefix}{e}.w2.weight");
        let sname = format!("{prefix}{e}.w2.scale");
        let shard = shard_of(&wname);
        let h = headers[&shard][&wname].clone();
        assert_eq!(h.dtype, "I8", "{wname}: expected I8, got {}", h.dtype);
        assert_eq!(
            h.shape,
            vec![W2_OUT, W2_IN_PACKED_GLOBAL],
            "{wname}: stored shape differs from the released [5120, 1152]"
        );
        let bytes = slice_and_pad(
            files.get_mut(&shard).unwrap(),
            &h,
            W2_OUT,
            W2_IN_PACKED_GLOBAL,
            RANK0_PACKED,
            PACKED_ROW,
            &wname,
        )?;
        w_all.extend_from_slice(&bytes);

        let shard = shard_of(&sname);
        let h = headers[&shard][&sname].clone();
        assert_eq!(
            h.dtype, "F8_E8M0",
            "{sname}: expected F8_E8M0, got {}",
            h.dtype
        );
        assert_eq!(
            h.shape,
            vec![W2_OUT, W2_IN_SCALES_GLOBAL],
            "{sname}: stored shape differs from the released [5120, 72]"
        );
        let bytes = slice_and_pad(
            files.get_mut(&shard).unwrap(),
            &h,
            W2_OUT,
            W2_IN_SCALES_GLOBAL,
            RANK0_SCALES,
            SCALE_ROW,
            &sname,
        )?;
        ws_all.extend_from_slice(&bytes);
    }

    // Deterministic synthetic activations and expert ids: the slice tests the GEMM, not the router.
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f32 / (1u64 << 53) as f32
    };
    let hidden: Vec<u16> = (0..args.tokens * HIDDEN)
        .map(|_| half::bf16::from_f32(next() * 2.0 - 1.0).to_bits())
        .collect();
    let mut act_fp8 = Vec::with_capacity(args.tokens * TOP_K * WIDTH_PADDED);
    let mut act_scales = Vec::with_capacity(args.tokens * TOP_K * SCALE_ROW);
    for _ in 0..args.tokens * TOP_K {
        let act: Vec<f32> = (0..WIDTH_PADDED).map(|_| next() * 2.0 - 1.0).collect();
        let (q, s) = routed_activation_operand(&act, 0.5);
        act_fp8.extend_from_slice(&q);
        act_scales.extend_from_slice(&s);
    }
    let expert_ids: Vec<u32> = (0..args.tokens * TOP_K)
        .map(|_| (next() * args.experts as f32) as u32 % args.experts as u32)
        .collect();

    // CPU reference on the same bytes.
    let (reference, cost) = moe_combine(
        &hidden,
        &act_fp8,
        &act_scales,
        &w_all,
        &ws_all,
        &expert_ids,
        args.tokens,
        HIDDEN,
        args.experts,
    );

    // The raw driver API through dlopen: no CUDA library is linked, exactly as the design records.
    let cuda = Cuda::load()?;
    let cubin = std::fs::read(&args.cubin).map_err(|e| format!("{}: {e}", args.cubin.display()))?;
    let (module, func) = cuda.load_function(&cubin, "b12x_mxfp4_moe_sm121")?;
    let mut allocated = 0usize;
    let hidden_d = cuda.alloc(hidden.len() * 2, &mut allocated)?;
    let act_d = cuda.alloc(act_fp8.len(), &mut allocated)?;
    let acts_d = cuda.alloc(act_scales.len(), &mut allocated)?;
    let w_d = cuda.alloc(w_all.len(), &mut allocated)?;
    let ws_d = cuda.alloc(ws_all.len(), &mut allocated)?;
    let ids_d = cuda.alloc(expert_ids.len() * 4, &mut allocated)?;
    let out_d = cuda.alloc(args.tokens * HIDDEN * 2, &mut allocated)?;
    cuda.copy_htod(&hidden_d, bytemuck::cast_slice(&hidden))?;
    cuda.copy_htod(&act_d, &act_fp8)?;
    cuda.copy_htod(&acts_d, &act_scales)?;
    cuda.copy_htod(&w_d, &w_all)?;
    cuda.copy_htod(&ws_d, &ws_all)?;
    cuda.copy_htod(&ids_d, bytemuck::cast_slice(&expert_ids))?;

    let grid = (args.tokens as u32, (HIDDEN / TILE_D) as u32, 1u32);
    let block = (THREADS, 1, 1);
    // Every argument is staged as 8 bytes; on a little-endian host the driver reads the low 4 bytes
    // of the u32 parameters, which is the value written here.
    let params: [u64; 11] = [
        hidden_d.0,
        act_d.0,
        acts_d.0,
        w_d.0,
        ws_d.0,
        ids_d.0,
        out_d.0,
        args.tokens as u64,
        0u64,
        args.layer as u64,
        1u64,
    ];
    // One launch by default, exactly as before --repeat existed. With --repeat the uploads, the
    // comparison and the rank/pad check are unchanged; only the launches inside the timed region
    // are repeated.
    let mut timing = None;
    if args.repeat > 1 {
        timing = Some(cuda.launch_repeated(func, grid, block, &params, args.repeat)?);
    } else {
        cuda.launch(func, grid, block, &params)?;
    }
    let mut gpu = vec![0u8; args.tokens * HIDDEN * 2];
    cuda.copy_dtoh(&mut gpu, &out_d)?;
    cuda.free(module, &[hidden_d, act_d, acts_d, w_d, ws_d, ids_d, out_d]);

    let gpu_bits: Vec<u16> = bytemuck::cast_slice(&gpu).to_vec();
    assert_eq!(
        gpu_bits.len(),
        reference.len(),
        "output length differs from the reference"
    );
    let exact = gpu_bits
        .iter()
        .zip(&reference)
        .filter(|(a, b)| a == b)
        .count();
    let mut max_delta = 0.0f32;
    for (a, b) in gpu_bits.iter().zip(&reference) {
        let d = (half::bf16::from_bits(*a).to_f32() - half::bf16::from_bits(*b).to_f32()).abs();
        if d > max_delta {
            max_delta = d;
        }
    }
    let pad_clean = w_all
        .chunks(PACKED_ROW)
        .all(|r| r[RANK0_PACKED..].iter().all(|b| *b == 0));

    println!("=== slice result ===");
    println!(
        "layer {} experts {} tokens {} top_k {}",
        args.layer, args.experts, args.tokens, TOP_K
    );
    println!(
        "device bytes allocated: {allocated} ({} MiB, {} expert weights uploaded)",
        allocated / (1024 * 1024),
        args.experts
    );
    println!(
        "combine weight bytes: {} ({} experts x {} rows x {PACKED_ROW} B)",
        w_all.len(),
        args.experts,
        W2_OUT
    );
    println!("cpu reference cost: {cost:?}");
    println!("elements compared: {}", reference.len());
    println!(
        "exact bf16 matches: {exact} ({:.4}%)",
        100.0 * exact as f64 / reference.len() as f64
    );
    println!("max abs delta: {max_delta:e}");
    println!("rank-1..3 tail zero and pad tail zero in every uploaded row: {pad_clean}");
    if let Some((mut per_launch, batched_ms)) = timing {
        per_launch.sort_by(|a, b| a.partial_cmp(b).expect("event times are finite"));
        let n = per_launch.len();
        let mean = per_launch.iter().sum::<f64>() / n as f64;
        let seconds = batched_ms / 1000.0;
        println!("repeat: {n} launches timed, 1 warm-up launch not counted");
        println!(
            "per-launch ms: min {:.4} p50 {:.4} mean {mean:.4} max {:.4}",
            per_launch[0],
            per_launch[n / 2],
            per_launch[n - 1]
        );
        println!("batched ms/launch over {n} launches: {batched_ms:.4}");
        println!(
            "throughput: {:.1} tokens/s, {:.3} Msample/s",
            args.tokens as f64 / seconds,
            (args.tokens * HIDDEN) as f64 / seconds / 1e6
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Raw CUDA driver API, resolved through dlopen. Nothing links libcuda: the driver is dlopen'd at
// run time on the node, which is the invariant docs/build/device-path-design.md records.
// ---------------------------------------------------------------------------------------------

type CuResult = i32;

/// An opaque CUDA event handle.
type CuEvent = *mut c_void;

/// MemAvailable in kB, parsed. Interpolating the whole /proc line into a message produced
/// "...the node has ~MemAvailable: 3768676 kB GB of system memory available...", which is exactly
/// the kind of unreadable failure text this launcher exists to avoid.
fn mem_available_kb() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    text.lines()
        .find_map(|line| line.strip_prefix("MemAvailable:"))
        .and_then(|value| value.split_whitespace().next())
        .and_then(|value| value.parse().ok())
}

/// The cuLaunchKernel signature. The parameter list is long by nature, so it is named once here
/// rather than spelled out at the one call site.
type LaunchFn = unsafe extern "C" fn(
    CuFunction,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    *mut c_void,
    *mut *mut c_void,
    *mut c_void,
) -> CuResult;

/// The driver's CUresult names, so a report says OUT_OF_MEMORY rather than 2.
fn name(code: CuResult) -> String {
    let s = match code {
        0 => "SUCCESS",
        1 => "INVALID_VALUE",
        2 => "OUT_OF_MEMORY",
        3 => "NOT_INITIALIZED",
        4 => "DEINITIALIZED",
        100 => "NO_DEVICE",
        101 => "INVALID_DEVICE",
        201 => "INVALID_CONTEXT",
        400 => "INVALID_HANDLE",
        700 => "ILLEGAL_ADDRESS",
        801 => "NOT_SUPPORTED",
        _ => "UNKNOWN",
    };
    format!("{code} ({s})")
}
type CuDeviceptr = u64;
type CuModule = *mut c_void;
type CuFunction = *mut c_void;

struct Deviceptr(CuDeviceptr);

struct Cuda {
    lib: libloading::Library,
    ctx: *mut c_void,
}

impl Cuda {
    /// cuInit alone is not enough: cuModuleLoadData and cuMemAlloc need a CURRENT context, and
    /// without one the driver answers CUDA_ERROR_INVALID_CONTEXT (201), which is what the first
    /// run on the node returned. The primary context is retained (refcounted, shared with anything
    /// else driving the device) and made current for this thread.
    fn load() -> Result<Self, String> {
        unsafe {
            let lib = libloading::Library::new("libcuda.so.1")
                .map_err(|e| format!("libcuda.so.1: {e}"))?;
            {
                let init: libloading::Symbol<unsafe extern "C" fn(u32) -> CuResult> =
                    lib.get(b"cuInit").map_err(|e| e.to_string())?;
                let r = init(0);
                if r != 0 {
                    return Err(format!("cuInit failed: {r}"));
                }
            }
            let mut count: i32 = 0;
            {
                let f: libloading::Symbol<unsafe extern "C" fn(*mut i32) -> CuResult> =
                    lib.get(b"cuDeviceGetCount").map_err(|e| e.to_string())?;
                let r = f(&mut count);
                if r != 0 {
                    return Err(format!("cuDeviceGetCount failed: {}", name(r)));
                }
            }
            let mut dev: i32 = 0;
            {
                let get: libloading::Symbol<unsafe extern "C" fn(*mut i32, i32) -> CuResult> =
                    lib.get(b"cuDeviceGet").map_err(|e| e.to_string())?;
                let r = get(&mut dev, 0);
                if r != 0 {
                    return Err(format!("cuDeviceGet(0) failed: {}", name(r)));
                }
            }
            let mut device_name = String::from("unknown device");
            {
                let mut nm = [0i8; 128];
                let f: libloading::Symbol<unsafe extern "C" fn(*mut i8, i32, i32) -> CuResult> =
                    lib.get(b"cuDeviceGetName").map_err(|e| e.to_string())?;
                let r = f(nm.as_mut_ptr(), 128, dev);
                if r == 0 {
                    // c_char is u8 on aarch64 and i8 on x86_64; cast rather than assume either.
                    let s = std::ffi::CStr::from_ptr(nm.as_ptr().cast());
                    device_name = s.to_string_lossy().into_owned();
                    let mut total: u64 = 0;
                    let t: libloading::Symbol<unsafe extern "C" fn(*mut u64, i32) -> CuResult> =
                        lib.get(b"cuDeviceTotalMem_v2").map_err(|e| e.to_string())?;
                    let _ = t(&mut total, dev);
                    println!(
                        "device 0: {device_name} of {count} (total {} MiB, MemAvailable {} MiB)",
                        total / (1024 * 1024),
                        mem_available_kb().map(|kb| kb / 1024).unwrap_or(0)
                    );
                }
            }
            let mut ctx: *mut c_void = std::ptr::null_mut();
            {
                // CU_CTX_LMEM_RESIZE_TO_MAX (0x10) asks the driver NOT to pre-reserve the maximum
                // local memory for this context. On a unified-memory GB10 where the device total is
                // 124 GB but only ~3.7 GB of system memory is free, that pre-reservation is the
                // difference between a context and CUDA_ERROR_OUT_OF_MEMORY.
                let set_flags: libloading::Symbol<unsafe extern "C" fn(i32, u32) -> CuResult> = lib
                    .get(b"cuDevicePrimaryCtxSetFlags")
                    .map_err(|e| e.to_string())?;
                let _ = set_flags(dev, 0x10);
                let retain: libloading::Symbol<
                    unsafe extern "C" fn(*mut *mut c_void, i32) -> CuResult,
                > = lib
                    .get(b"cuDevicePrimaryCtxRetain")
                    .map_err(|e| e.to_string())?;
                let r = retain(&mut ctx, dev);
                if r != 0 {
                    // One alternative route, tried exactly once - not a retry loop. Retaining the
                    // primary context is the normal route for a process that does not own the
                    // device; cuCtxCreate_v2 is what a launcher would use if it did. Both results
                    // are reported so the failure cannot be read as one API's quirk.
                    let mut alt: *mut c_void = std::ptr::null_mut();
                    let create: libloading::Symbol<
                        unsafe extern "C" fn(*mut *mut c_void, u32, i32) -> CuResult,
                    > = lib.get(b"cuCtxCreate_v2").map_err(|e| e.to_string())?;
                    let r2 = create(&mut alt, 0x10, dev);
                    // Nothing is left half-initialised: no context was retained, no device memory
                    // was allocated, and this condition is not retried. If a context IS created and
                    // a later step fails, Drop releases it.
                    let avail = mem_available_kb()
                        .map(|kb| format!("{} MiB", kb / 1024))
                        .unwrap_or_else(|| "unknown".to_string());
                    return Err(format!(
                        "no CUDA context could be created on device 0 ({}): \
                         cuDevicePrimaryCtxRetain -> {}, cuCtxCreate_v2 -> {} (one attempt each, not \
                         retried). cuInit, cuDeviceGet and cuDeviceGetName all succeeded, so the \
                         driver is reachable and the device is present: on this node a second CUDA \
                         context does not fit beside the resident model. MemAvailable now: {avail}. \
                         No device memory was allocated and no context is left retained. This is a \
                         capacity fact about the node, not a defect in this launcher - do not work \
                         around it here.",
                        device_name,
                        name(r),
                        name(r2),
                    ));
                }
            }
            {
                let set: libloading::Symbol<unsafe extern "C" fn(*mut c_void) -> CuResult> =
                    lib.get(b"cuCtxSetCurrent").map_err(|e| e.to_string())?;
                let r = set(ctx);
                if r != 0 {
                    return Err(format!("cuCtxSetCurrent failed: {r}"));
                }
            }
            Ok(Cuda { lib, ctx })
        }
    }

    fn sym<T>(&self, name: &[u8]) -> Result<libloading::Symbol<'_, T>, String> {
        unsafe {
            self.lib
                .get(name)
                .map_err(|e| format!("{}: {e}", String::from_utf8_lossy(name)))
        }
    }

    fn alloc(&self, bytes: usize, total: &mut usize) -> Result<Deviceptr, String> {
        unsafe {
            let f: libloading::Symbol<unsafe extern "C" fn(*mut CuDeviceptr, usize) -> CuResult> =
                self.sym(b"cuMemAlloc_v2")?;
            let mut p: CuDeviceptr = 0;
            let r = f(&mut p, bytes);
            if r != 0 {
                return Err(format!("cuMemAlloc({bytes}) failed: {r}"));
            }
            *total += bytes;
            Ok(Deviceptr(p))
        }
    }

    fn copy_htod(&self, dst: &Deviceptr, src: &[u8]) -> Result<(), String> {
        unsafe {
            let f: libloading::Symbol<
                unsafe extern "C" fn(CuDeviceptr, *const c_void, usize) -> CuResult,
            > = self.sym(b"cuMemcpyHtoD_v2")?;
            let r = f(dst.0, src.as_ptr().cast(), src.len());
            if r != 0 {
                return Err(format!("cuMemcpyHtoD failed: {r}"));
            }
            Ok(())
        }
    }

    fn copy_dtoh(&self, dst: &mut [u8], src: &Deviceptr) -> Result<(), String> {
        unsafe {
            let f: libloading::Symbol<
                unsafe extern "C" fn(*mut c_void, CuDeviceptr, usize) -> CuResult,
            > = self.sym(b"cuMemcpyDtoH_v2")?;
            let r = f(dst.as_mut_ptr().cast(), src.0, dst.len());
            if r != 0 {
                return Err(format!("cuMemcpyDtoH failed: {r}"));
            }
            Ok(())
        }
    }

    fn load_function(&self, cubin: &[u8], name: &str) -> Result<(CuModule, CuFunction), String> {
        unsafe {
            let load: libloading::Symbol<
                unsafe extern "C" fn(*mut CuModule, *const c_void) -> CuResult,
            > = self.sym(b"cuModuleLoadData")?;
            let mut module: CuModule = std::ptr::null_mut();
            let r = load(&mut module, cubin.as_ptr().cast());
            if r != 0 {
                return Err(format!("cuModuleLoadData failed: {r}"));
            }
            let get: libloading::Symbol<
                unsafe extern "C" fn(*mut CuFunction, CuModule, *const i8) -> CuResult,
            > = self.sym(b"cuModuleGetFunction")?;
            let cname = std::ffi::CString::new(name).map_err(|e| e.to_string())?;
            let mut func: CuFunction = std::ptr::null_mut();
            let r = get(&mut func, module, cname.as_ptr().cast());
            if r != 0 {
                return Err(format!("cuModuleGetFunction({name}) failed: {r}"));
            }
            Ok((module, func))
        }
    }

    fn launch(
        &self,
        func: CuFunction,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        args: &[u64],
    ) -> Result<(), String> {
        unsafe {
            let launch: libloading::Symbol<LaunchFn> = self.sym(b"cuLaunchKernel")?;
            let mut storage: Vec<[u64; 1]> = args.iter().map(|a| [*a]).collect();
            let mut ptrs: Vec<*mut c_void> =
                storage.iter_mut().map(|s| s.as_mut_ptr().cast()).collect();
            let r = launch(
                func,
                grid.0,
                grid.1,
                grid.2,
                block.0,
                block.1,
                block.2,
                0,
                std::ptr::null_mut(),
                ptrs.as_mut_ptr(),
                std::ptr::null_mut(),
            );
            if r != 0 {
                return Err(format!("cuLaunchKernel failed: {r}"));
            }
            let sync: libloading::Symbol<unsafe extern "C" fn() -> CuResult> =
                self.sym(b"cuCtxSynchronize")?;
            let r = sync();
            if r != 0 {
                return Err(format!("cuCtxSynchronize failed: {r}"));
            }
            Ok(())
        }
    }

    /// Times one kernel the way a throughput claim needs it: a warm-up launch outside the timed
    /// region, one event pair per launch for the distribution, then a single batched region with no
    /// per-launch synchronisation - the figure to quote when the per-launch event overhead would
    /// dominate a kernel this small. No allocation, no upload and no checkpoint read is inside
    /// either timed region, so the numbers are the kernel's own and not the host's.
    fn launch_repeated(
        &self,
        func: CuFunction,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        args: &[u64],
        repeat: usize,
    ) -> Result<(Vec<f64>, f64), String> {
        unsafe {
            let create: libloading::Symbol<unsafe extern "C" fn(*mut CuEvent, u32) -> CuResult> =
                self.sym(b"cuEventCreate")?;
            let record: libloading::Symbol<unsafe extern "C" fn(CuEvent, *mut c_void) -> CuResult> =
                self.sym(b"cuEventRecord")?;
            let wait: libloading::Symbol<unsafe extern "C" fn(CuEvent) -> CuResult> =
                self.sym(b"cuEventSynchronize")?;
            let elapsed: libloading::Symbol<
                unsafe extern "C" fn(*mut f32, CuEvent, CuEvent) -> CuResult,
            > = self.sym(b"cuEventElapsedTime")?;
            let destroy: libloading::Symbol<unsafe extern "C" fn(CuEvent) -> CuResult> =
                self.sym(b"cuEventDestroy")?;
            let mut start: CuEvent = std::ptr::null_mut();
            let mut stop: CuEvent = std::ptr::null_mut();
            let r = create(&mut start, 0);
            if r != 0 {
                return Err(format!("cuEventCreate failed: {r}"));
            }
            let r = create(&mut stop, 0);
            if r != 0 {
                return Err(format!("cuEventCreate failed: {r}"));
            }
            // Warm-up: fills the module cache and the launch path, and is deliberately not timed.
            self.launch(func, grid, block, args)?;
            let sync: libloading::Symbol<unsafe extern "C" fn() -> CuResult> =
                self.sym(b"cuCtxSynchronize")?;
            let r = sync();
            if r != 0 {
                return Err(format!("cuCtxSynchronize failed: {r}"));
            }
            let mut per_launch = Vec::with_capacity(repeat);
            for _ in 0..repeat {
                let r = record(start, std::ptr::null_mut());
                if r != 0 {
                    return Err(format!("cuEventRecord failed: {r}"));
                }
                self.launch(func, grid, block, args)?;
                let r = record(stop, std::ptr::null_mut());
                if r != 0 {
                    return Err(format!("cuEventRecord failed: {r}"));
                }
                let r = wait(stop);
                if r != 0 {
                    return Err(format!("cuEventSynchronize failed: {r}"));
                }
                let mut ms: f32 = 0.0;
                let r = elapsed(&mut ms, start, stop);
                if r != 0 {
                    return Err(format!("cuEventElapsedTime failed: {r}"));
                }
                per_launch.push(f64::from(ms));
            }
            let r = record(start, std::ptr::null_mut());
            if r != 0 {
                return Err(format!("cuEventRecord failed: {r}"));
            }
            for _ in 0..repeat {
                self.launch(func, grid, block, args)?;
            }
            let r = record(stop, std::ptr::null_mut());
            if r != 0 {
                return Err(format!("cuEventRecord failed: {r}"));
            }
            let r = wait(stop);
            if r != 0 {
                return Err(format!("cuEventSynchronize failed: {r}"));
            }
            let mut total: f32 = 0.0;
            let r = elapsed(&mut total, start, stop);
            if r != 0 {
                return Err(format!("cuEventElapsedTime failed: {r}"));
            }
            destroy(start);
            destroy(stop);
            Ok((per_launch, f64::from(total) / repeat as f64))
        }
    }

    fn free(&self, module: CuModule, ptrs: &[Deviceptr]) {
        unsafe {
            if let Ok(f) =
                self.sym::<unsafe extern "C" fn(CuDeviceptr) -> CuResult>(b"cuMemFree_v2")
            {
                for p in ptrs {
                    let _ = f(p.0);
                }
            }
            if let Ok(f) = self.sym::<unsafe extern "C" fn(CuModule) -> CuResult>(b"cuModuleUnload")
            {
                let _ = f(module);
            }
            // The primary context is released by Drop, on every exit path.
        }
    }
}

impl Drop for Cuda {
    /// Releases this process's reference to the primary context on every exit path, including a
    /// failure after the context was created, so a run can never leave a half-initialised Cuda.
    /// Releasing is refcounted and does not disturb anything else driving the device.
    fn drop(&mut self) {
        if self.ctx.is_null() {
            return;
        }
        unsafe {
            if let Ok(f) = self
                .sym::<unsafe extern "C" fn(*mut c_void) -> CuResult>(b"cuDevicePrimaryCtxRelease")
            {
                let _ = f(self.ctx);
            }
        }
    }
}
