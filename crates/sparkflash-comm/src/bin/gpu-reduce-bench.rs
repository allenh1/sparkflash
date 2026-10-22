//! Four-rank fabric comparison of the layer's all-reduce arms, measured in isolation.
//!
//! Two arms, one process per rank, one payload per rank per call, apples to apples on the same
//! nodes, the same payloads and the same exchange order:
//!
//! * host - today's machinery: TcpTransport's spawn/join reduce, one blocking
//!   all_reduce_bf16 per call. This is what the forward loop's moe_reduce and
//!   attention_output blocks pay.
//! * nccl - the device arm: a vendored libnccl.so.2 loaded at run time (sparkflash_comm::nccl)
//!   with one communicator created at setup, and one AllReduce(Sum) enqueued per call on the
//!   caller's stream. The timed halves are reported separately: enqueue is what the layer's host
//!   thread pays, total is enqueue plus a cold stream synchronise, and batch is the amortised
//!   cost of enqueuing N reduces back to back with one synchronise at the end - the shape a layer
//!   port uses when it does not read the result back.
//!
//! The binary never reads the result back to the host in the timed section. The one readback is
//! the correctness check: both arms are compared against the canonical rank-order f32 reduce with
//! one bf16 round (sparkflash_comm::reduce), so a tolerance arm can quote its exact deviation.
//!
//! The layer's decode payload for both cross-rank collectives is the same 5,120 bf16 elements =
//! 10,240 B per rank: the attention-output delta (hidden row) and the MoE routed-row delta.
//! --bytes defaults to that payload plus the two-token shape as a scaling row.
//!
//! Usage (four nodes; the runner is scripts/gpu-reduce-bench.sh):
//!   gpu-reduce-bench --rank 0 --world 4 --addrs 192.0.2.80:19701,... \
//!       --control 192.0.2.80:19755 --bytes 10240,20480 --samples 200 --warmup 20 --json
//!
//! Exit codes: 0 measured, 2 usage, 3 a transport or library is unavailable, 4 an exchange failed,
//! 5 a payload mismatched the canonical reference badly (more than 2 ULP or non-finite).

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::process::ExitCode;
use std::time::{Duration, Instant};

use clap::Parser;
use sparkflash_comm::device_reduce::{self, RunReduce};
use sparkflash_comm::layout::{CollectiveLayout, FabricSchedule};
use sparkflash_comm::nccl::{NcclApi, NcclUniqueId};
use sparkflash_comm::reduce::canonical_reduce_element;
use sparkflash_comm::tcp::{TcpConfig, TcpTransport};
use sparkflash_comm::transport::Transport;

/// The served decode payload, 5,120 bf16 elements, and the two-token shape as a scaling row.
const LAYER_BYTES: [usize; 2] = [10_240, 20_480];

/// Loads libcuda.so.1 at run time and binds the handful of driver entry points this bench needs.
/// No build-time CUDA dependency: the release binary is the same one that runs without a GPU.
mod cuda {
    use std::ffi::{CStr, c_int, c_void};
    use std::ptr;

    pub type Result<T> = std::result::Result<T, String>;
    type CuResult = c_int;
    type CuDevice = c_int;
    type CuContext = *mut c_void;
    type CuStream = *mut c_void;

    pub const CUDA_SUCCESS: CuResult = 0;

    struct Api {
        _library: *mut c_void,
        cu_init: unsafe extern "C" fn(c_int) -> CuResult,
        cu_device_get: unsafe extern "C" fn(*mut CuDevice, c_int) -> CuResult,
        cu_device_primary_ctx_retain: unsafe extern "C" fn(*mut CuContext, CuDevice) -> CuResult,
        cu_ctx_set_current: unsafe extern "C" fn(CuContext) -> CuResult,
        cu_mem_alloc: unsafe extern "C" fn(*mut u64, usize) -> CuResult,
        cu_mem_free: unsafe extern "C" fn(u64) -> CuResult,
        cu_memcpy_htod: unsafe extern "C" fn(u64, *const c_void, usize) -> CuResult,
        cu_memcpy_dtoh: unsafe extern "C" fn(*mut c_void, u64, usize) -> CuResult,
        cu_stream_create: unsafe extern "C" fn(*mut CuStream, u32) -> CuResult,
        cu_stream_synchronize: unsafe extern "C" fn(CuStream) -> CuResult,
        cu_stream_destroy: unsafe extern "C" fn(CuStream) -> CuResult,
    }

    /// A loaded driver and the primary context this process runs kernels on.
    pub struct Cuda {
        api: Api,
        _context: CuContext,
    }

    fn symbol(handle: *mut c_void, name: &str) -> Result<*mut c_void> {
        let cname = std::ffi::CString::new(name).map_err(|error| error.to_string())?;
        // SAFETY: handle is a live dlopen handle and the name is a nul-terminated C string.
        let pointer = unsafe { libc::dlsym(handle, cname.as_ptr()) };
        if pointer.is_null() {
            return Err(format!("libcuda.so.1 has no {name}"));
        }
        Ok(pointer)
    }

    macro_rules! bind {
        ($handle:expr, $name:literal, $ty:ty) => {{
            let pointer = symbol($handle, $name)?;
            // SAFETY: the symbol was resolved by name from the driver library; every CUDA driver
            // entry point in this list has the signature the type states.
            unsafe { std::mem::transmute::<*mut c_void, $ty>(pointer) }
        }};
    }

    impl Cuda {
        /// Opens the driver and makes device 0's primary context current on this thread.
        pub fn open() -> Result<Self> {
            let name = std::ffi::CString::new("libcuda.so.1").map_err(|e| e.to_string())?;
            // SAFETY: dlopen with RTLD_NOW on a soname; the handle is owned by Api.
            let library = unsafe { libc::dlopen(name.as_ptr(), libc::RTLD_NOW) };
            if library.is_null() {
                let detail = unsafe { CStr::from_ptr(libc::dlerror()) }
                    .to_string_lossy()
                    .into_owned();
                return Err(format!("dlopen libcuda.so.1: {detail}"));
            }
            let api = Api {
                _library: library,
                cu_init: bind!(library, "cuInit", unsafe extern "C" fn(c_int) -> CuResult),
                cu_device_get: bind!(
                    library,
                    "cuDeviceGet",
                    unsafe extern "C" fn(*mut CuDevice, c_int) -> CuResult
                ),
                cu_device_primary_ctx_retain: bind!(
                    library,
                    "cuDevicePrimaryCtxRetain",
                    unsafe extern "C" fn(*mut CuContext, CuDevice) -> CuResult
                ),
                cu_ctx_set_current: bind!(
                    library,
                    "cuCtxSetCurrent",
                    unsafe extern "C" fn(CuContext) -> CuResult
                ),
                cu_mem_alloc: bind!(
                    library,
                    "cuMemAlloc_v2",
                    unsafe extern "C" fn(*mut u64, usize) -> CuResult
                ),
                cu_mem_free: bind!(
                    library,
                    "cuMemFree_v2",
                    unsafe extern "C" fn(u64) -> CuResult
                ),
                cu_memcpy_htod: bind!(
                    library,
                    "cuMemcpyHtoD_v2",
                    unsafe extern "C" fn(u64, *const c_void, usize) -> CuResult
                ),
                cu_memcpy_dtoh: bind!(
                    library,
                    "cuMemcpyDtoH_v2",
                    unsafe extern "C" fn(*mut c_void, u64, usize) -> CuResult
                ),
                cu_stream_create: bind!(
                    library,
                    "cuStreamCreate",
                    unsafe extern "C" fn(*mut CuStream, u32) -> CuResult
                ),
                cu_stream_synchronize: bind!(
                    library,
                    "cuStreamSynchronize",
                    unsafe extern "C" fn(CuStream) -> CuResult
                ),
                cu_stream_destroy: bind!(
                    library,
                    "cuStreamDestroy_v2",
                    unsafe extern "C" fn(CuStream) -> CuResult
                ),
            };
            // SAFETY: every call below passes the out-parameters the driver documents.
            unsafe {
                check((api.cu_init)(0), "cuInit")?;
                let mut device: CuDevice = 0;
                check((api.cu_device_get)(&mut device, 0), "cuDeviceGet")?;
                let mut context: CuContext = ptr::null_mut();
                check(
                    (api.cu_device_primary_ctx_retain)(&mut context, device),
                    "cuDevicePrimaryCtxRetain",
                )?;
                check((api.cu_ctx_set_current)(context), "cuCtxSetCurrent")?;
                Ok(Self {
                    api,
                    _context: context,
                })
            }
        }

        pub fn malloc(&self, bytes: usize) -> Result<u64> {
            let mut pointer: u64 = 0;
            // SAFETY: out-parameter for cuMemAlloc.
            unsafe { check((self.api.cu_mem_alloc)(&mut pointer, bytes), "cuMemAlloc")? };
            Ok(pointer)
        }

        pub fn free(&self, pointer: u64) {
            // SAFETY: pointer came from cuMemAlloc above and is freed once.
            unsafe {
                let _ = (self.api.cu_mem_free)(pointer);
            }
        }

        pub fn upload(&self, device: u64, bytes: &[u8]) -> Result<()> {
            // SAFETY: device points to at least bytes.len() bytes; bytes is a valid host slice.
            unsafe {
                check(
                    (self.api.cu_memcpy_htod)(device, bytes.as_ptr().cast(), bytes.len()),
                    "cuMemcpyHtoD",
                )
            }
        }

        pub fn download(&self, device: u64, bytes: &mut [u8]) -> Result<()> {
            // SAFETY: device points to at least bytes.len() bytes; bytes is a valid host slice.
            unsafe {
                check(
                    (self.api.cu_memcpy_dtoh)(bytes.as_mut_ptr().cast(), device, bytes.len()),
                    "cuMemcpyDtoH",
                )
            }
        }

        pub fn stream(&self) -> Result<CuStream> {
            let mut stream: CuStream = ptr::null_mut();
            // SAFETY: out-parameter for cuStreamCreate; 0 is the default flags value.
            unsafe {
                check(
                    (self.api.cu_stream_create)(&mut stream, 0),
                    "cuStreamCreate",
                )?
            };
            Ok(stream)
        }

        pub fn stream_sync(&self, stream: CuStream) -> Result<()> {
            // SAFETY: stream came from stream() above and is live.
            unsafe {
                check(
                    (self.api.cu_stream_synchronize)(stream),
                    "cuStreamSynchronize",
                )
            }
        }

        pub fn stream_destroy(&self, stream: CuStream) {
            // SAFETY: stream came from stream() above and is destroyed once.
            unsafe {
                let _ = (self.api.cu_stream_destroy)(stream);
            }
        }
    }

    /// Wraps a driver result code: non-zero becomes a message naming the call and the code.
    fn check(result: CuResult, name: &str) -> Result<()> {
        if result == CUDA_SUCCESS {
            Ok(())
        } else {
            Err(format!("{name} failed with CUresult {result}"))
        }
    }
}

#[derive(Debug, Parser)]
#[command(
    name = "gpu-reduce-bench",
    version,
    about = "Four-rank fabric comparison of the layer's all-reduce arms at the layer's real message sizes; no number exists until it runs on the four nodes"
)]
struct Args {
    /// This rank's index in the collective.
    #[arg(long, default_value_t = 0)]
    rank: usize,
    /// Ranks in the collective.
    #[arg(long, default_value_t = 4)]
    world: usize,
    /// Control-plane address per rank in rank order, for the host transport.
    #[arg(long = "addrs", value_delimiter = ',', value_name = "IP:PORT")]
    addrs: Vec<String>,
    /// Rank-0 address the NCCL unique id is exchanged over: 128 opaque bytes, one socket each.
    #[arg(long = "control", value_name = "IP:PORT")]
    control: Option<String>,
    /// Bytes per rank per call, repeatable or comma separated.
    #[arg(long = "bytes", value_delimiter = ',', value_name = "N")]
    bytes: Vec<usize>,
    /// Timed samples per payload per arm.
    #[arg(long, default_value_t = 200)]
    samples: usize,
    /// Exchanges discarded before the timed samples.
    #[arg(long, default_value_t = 20)]
    warmup: usize,
    /// sync: enqueue then synchronise once per call; batch: enqueue --batch calls, then one
    /// synchronise, reporting the amortised per-call cost.
    #[arg(long, default_value = "batch")]
    mode: String,
    /// Calls per batch in batch mode (one per decoder layer is 40).
    #[arg(long, default_value_t = 40)]
    batch: usize,
    /// Print exactly one JSON object on stdout.
    #[arg(long)]
    json: bool,
    /// Load libnccl.so.2, report its version and exit (no GPU, no communicator).
    #[arg(long = "check-nccl")]
    check_nccl: bool,
}

/// Deterministic per-rank contribution: exact bf16 rounds of small f32s, so the canonical
/// reference compares against the same bits the wire carries.
fn contribution(rank: usize, elements: usize) -> Vec<u16> {
    (0..elements)
        .map(|index| {
            let base = 0.001_953_125_f32;
            let pattern = (((index * 7 + rank * 13) % 19) as f32) - 9.0;
            let value = (rank as f32 + 1.0) * base * pattern;
            sparkflash_comm::f32_to_bf16_rn(value)
        })
        .collect()
}

fn expected(world: usize, elements: usize) -> Vec<u16> {
    let payloads: Vec<Vec<u16>> = (0..world)
        .map(|rank| contribution(rank, elements))
        .collect();
    let mut out = vec![0u16; elements];
    for (slot, value) in out.iter_mut().enumerate() {
        let column: Vec<u16> = payloads.iter().map(|payload| payload[slot]).collect();
        *value = canonical_reduce_element(&column);
    }
    out
}

fn percentile(sorted: &[Duration], percent: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let index = ((percent / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[index.min(sorted.len() - 1)].as_secs_f64() * 1_000_000.0
}

/// Sorts in place and returns the JSON fields every arm reports.
fn summary(durations: &mut [Duration]) -> String {
    durations.sort_unstable();
    let count = durations.len().max(1);
    let mean = durations.iter().sum::<Duration>().as_secs_f64() * 1e6 / count as f64;
    format!(
        "\"p50_us\":{:.2},\"p90_us\":{:.2},\"p99_us\":{:.2},\"mean_us\":{:.2},\"max_us\":{:.2}",
        percentile(durations, 50.0),
        percentile(durations, 90.0),
        percentile(durations, 99.0),
        mean,
        durations.last().map_or(0.0, |d| d.as_secs_f64() * 1e6),
    )
}

/// Exchanges the 128-byte NCCL unique id over the control address: rank 0 hands it to every other
/// rank on its own socket. This is the caller's bootstrap channel, not an NCCL improvised one.
fn exchange_unique_id(
    rank: usize,
    world: usize,
    control: SocketAddr,
    id: Option<NcclUniqueId>,
) -> Result<NcclUniqueId, String> {
    if rank == 0 {
        let id = id.ok_or_else(|| "rank 0 has no unique id".to_owned())?;
        let hex = id.to_hex();
        let listener =
            TcpListener::bind(control).map_err(|error| format!("bind {control}: {error}"))?;
        let mut served = 1;
        while served < world {
            let (mut socket, peer) = listener
                .accept()
                .map_err(|error| format!("accept: {error}"))?;
            let mut who = [0u8; 1];
            socket
                .read_exact(&mut who)
                .map_err(|error| format!("control read from {peer}: {error}"))?;
            socket
                .write_all(hex.as_bytes())
                .map_err(|error| format!("control write to {peer}: {error}"))?;
            served += 1;
        }
        Ok(id)
    } else {
        let mut socket = TcpStream::connect(control)
            .map_err(|error| format!("connect control {control}: {error}"))?;
        socket
            .write_all(&[rank as u8])
            .map_err(|error| format!("control write: {error}"))?;
        let mut text = vec![0u8; sparkflash_comm::nccl::NCCL_UNIQUE_ID_BYTES * 2];
        socket
            .read_exact(&mut text)
            .map_err(|error| format!("control read: {error}"))?;
        let text = String::from_utf8(text).map_err(|error| format!("control hex: {error}"))?;
        NcclUniqueId::from_hex(&text).map_err(|error| format!("{error}"))
    }
}

/// The comparison an arm's payload makes against the canonical reduce: how many elements differ,
/// by how many bf16 ULPs, and whether every value is finite.
fn compare(want: &[u16], got: &[u16]) -> (usize, u32, bool) {
    let mut differing = 0usize;
    let mut max_ulp = 0u32;
    let mut all_finite = true;
    for (want, got) in want.iter().zip(got) {
        if want == got {
            continue;
        }
        differing += 1;
        let got_value = sparkflash_comm::bf16_to_f32(*got);
        if !got_value.is_finite() {
            all_finite = false;
            continue;
        }
        let ulps = (*want as i32 - *got as i32).unsigned_abs();
        max_ulp = max_ulp.max(ulps);
    }
    (differing, max_ulp, all_finite)
}

fn main() -> ExitCode {
    let args = Args::parse();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("gpu-reduce-bench: rank {}: {error}", args.rank);
            ExitCode::from(3)
        }
    }
}

fn run(args: &Args) -> Result<(), String> {
    if args.check_nccl {
        let api = NcclApi::load().map_err(|error| format!("{error}"))?;
        println!(
            "nccl_library={} nccl_version={}",
            api.library_name(),
            api.version_string().unwrap_or_else(|| "unknown".to_owned())
        );
        return Ok(());
    }
    if args.addrs.is_empty() || args.addrs.len() != args.world {
        return Err("--addrs needs one IP:PORT per rank".to_owned());
    }
    let addrs: Vec<SocketAddr> = args
        .addrs
        .iter()
        .map(|entry| {
            entry
                .parse()
                .map_err(|error| format!("bad address {entry}: {error}"))
        })
        .collect::<Result<_, _>>()?;
    let control: SocketAddr = args
        .control
        .as_deref()
        .unwrap_or("127.0.0.1:19755")
        .parse()
        .map_err(|error| format!("bad --control: {error}"))?;
    let bytes = if args.bytes.is_empty() {
        LAYER_BYTES.to_vec()
    } else {
        args.bytes.clone()
    };
    for size in &bytes {
        if size % 2 != 0 {
            return Err(format!(
                "payload {size} is not a whole number of bf16 values"
            ));
        }
    }

    // The two libraries are loaded before the communicator: a missing libnccl is a named refusal,
    // not a silent fallback.
    let cuda = cuda::Cuda::open()?;
    let api = NcclApi::load().map_err(|error| format!("{error}"))?;
    let version = api.version_string().unwrap_or_else(|| "unknown".to_owned());
    let unique_id = if args.rank == 0 {
        Some(api.unique_id().map_err(|error| format!("{error}"))?)
    } else {
        None
    };
    let unique_id = exchange_unique_id(args.rank, args.world, control, unique_id)?;
    // The device arm goes through the port's exact call shape: one run-owned opaque handle, one
    // enqueue per call, no synchronise or readback inside the wrapper.
    let handle = RunReduce::open(args.world, args.rank, unique_id)
        .map_err(|error| format!("RunReduce::open: {error}"))?;
    let stream = cuda.stream()?;

    // Host arm first: it is the blocking machinery the loop runs today.
    let config = TcpConfig::with_addresses(args.rank, addrs.clone())
        .map_err(|error| format!("address table: {error}"))?
        .with_timeout(Duration::from_secs(20))
        .with_connect_timeout(Duration::from_secs(20));
    let connected = Instant::now();
    let transport = TcpTransport::connect(config)
        .map_err(|error| format!("host transport connect: {error}"))?;
    let connect_ms = connected.elapsed().as_secs_f64() * 1000.0;

    let mut arms_json: Vec<String> = Vec::new();
    for size in &bytes {
        let elements = size / 2;
        let mut payload = contribution(args.rank, elements);
        let want = expected(args.world, elements);
        let layout = CollectiveLayout::new(args.world, elements, FabricSchedule::OneShotAllToAll)
            .map_err(|error| format!("layout: {error}"))?;
        let mut seq = 0u64;
        let mut host_us: Vec<Duration> = Vec::with_capacity(args.samples);
        for _ in 0..args.warmup {
            seq += 1;
            payload.copy_from_slice(&contribution(args.rank, elements));
            transport
                .all_reduce_bf16(&mut payload, seq, &layout)
                .map_err(|error| format!("host warmup: {error}"))?;
        }
        for _ in 0..args.samples {
            seq += 1;
            payload.copy_from_slice(&contribution(args.rank, elements));
            let started = Instant::now();
            transport
                .all_reduce_bf16(&mut payload, seq, &layout)
                .map_err(|error| format!("host sample: {error}"))?;
            host_us.push(started.elapsed());
        }
        seq += 1;
        payload.copy_from_slice(&contribution(args.rank, elements));
        transport
            .all_reduce_bf16(&mut payload, seq, &layout)
            .map_err(|error| format!("host barrier: {error}"))?;
        let (differing, max_ulp, finite) = compare(&want, &payload);
        arms_json.push(format!(
            "{{\"arm\":\"host\",\"bytes\":{size},\"elements\":{elements},\"mode\":\"sync\",{},\"check\":{{\"differing\":{differing},\"max_ulp\":{max_ulp},\"finite\":{finite}}}}}",
            summary(&mut host_us)
        ));

        // Device arm. One buffer pair per size; the timed section never touches the host.
        let send = cuda.malloc(*size)?;
        let recv = cuda.malloc(*size)?;
        // The warm-up reduces IN PLACE, so it gets its own buffer: handing it send would leave the
        // reduce input holding an already-reduced row, and the correctness check below would then
        // compare the canonical reference against a re-reduction of a reduced buffer.
        let scratch = cuda.malloc(*size)?;
        let source: Vec<u8> = contribution(args.rank, elements)
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect();
        cuda.upload(send, &source)?;
        cuda.upload(recv, &vec![0u8; *size])?;
        let reduce = |count: usize| -> Result<(), String> {
            // SAFETY: send/recv are device allocations of size bytes, the stream is live, and the
            // reduce is enqueued on the same device this process made current.
            unsafe { device_reduce::reduce(handle, stream as u64, send, recv, count) }
                .map_err(|error| format!("reduce via RunReduce: {error}"))
        };
        // The port's setup warm-up (WARM_UP_COLLECTIVES in-place enqueues), then the bench's own
        // per-size warm-up. The syncs are the bench's, not the wrapper's.
        // SAFETY: scratch is a device allocation of size bytes and the stream is live; it is only
        // ever the in-place warm-up target.
        unsafe { device_reduce::warm_up(handle, stream as u64, scratch, elements) }
            .map_err(|error| format!("warm-up: {error}"))?;
        cuda.stream_sync(stream)?;
        for _ in 0..args.warmup {
            reduce(elements)?;
            cuda.stream_sync(stream)?;
        }
        let batch = args.batch.max(1);
        let mut enqueue_us: Vec<Duration> = Vec::with_capacity(args.samples);
        let mut total_us: Vec<Duration> = Vec::with_capacity(args.samples);
        for _ in 0..args.samples {
            let started = Instant::now();
            let mut enqueued = Duration::ZERO;
            if args.mode == "batch" {
                for _ in 0..batch {
                    let one = Instant::now();
                    reduce(elements)?;
                    enqueued += one.elapsed();
                }
            } else {
                let one = Instant::now();
                reduce(elements)?;
                enqueued = one.elapsed();
            }
            cuda.stream_sync(stream)?;
            let total = started.elapsed();
            enqueue_us.push(enqueued);
            total_us.push(if args.mode == "batch" {
                total / batch as u32
            } else {
                total
            });
        }
        // One correctness readback, after the timed section.
        reduce(elements)?;
        cuda.stream_sync(stream)?;
        let mut got = vec![0u8; *size];
        cuda.download(recv, &mut got)?;
        let got: Vec<u16> = got
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| u16::from_le_bytes(*pair))
            .collect();
        let (differing, max_ulp, finite) = compare(&want, &got);
        // The run record's counters, exactly as the port reads them at the boundary.
        // SAFETY: the handle stays live until close at the end of run.
        let stats = unsafe { device_reduce::stats(handle) }
            .map_err(|error| format!("reduce stats: {error}"))?;
        let stats_json = format!(
            ",\"stats\":{{\"collectives\":{},\"elements\":{},\"bytes\":{},\"arena_bytes\":{}}}",
            stats.collectives(),
            stats.elements(),
            stats.bytes(),
            stats.arena_bytes()
        );
        arms_json.push(format!(
            "{{\"arm\":\"nccl\",\"bytes\":{size},\"elements\":{elements},\"mode\":\"{}\",\"batch\":{batch},{}, \"enqueue\":{{{}}},\"check\":{{\"differing\":{differing},\"max_ulp\":{max_ulp},\"finite\":{finite}}}{stats_json}}}",
            args.mode,
            summary(&mut total_us),
            summary(&mut enqueue_us)
        ));
        cuda.free(send);
        cuda.free(recv);
        cuda.free(scratch);
    }
    // SAFETY: the handle came from RunReduce::open and is closed exactly once, before the stream.
    unsafe { device_reduce::close(handle) };
    cuda.stream_destroy(stream);
    let host = std::env::var("HOSTNAME").unwrap_or_else(|_| "unknown".to_owned());
    let document = format!(
        "{{\"host\":\"{host}\",\"rank\":{},\"world\":{},\"connect_ms\":{connect_ms:.1},\"nccl\":{{\"library\":\"{}\",\"version\":\"{version}\"}},\"arms\":[{}]}}",
        args.rank,
        args.world,
        api.library_name(),
        arms_json.join(",")
    );
    if args.json {
        println!("{document}");
    } else {
        println!("gpu-reduce-bench rank {}: {}", args.rank, document);
    }
    Ok(())
}
