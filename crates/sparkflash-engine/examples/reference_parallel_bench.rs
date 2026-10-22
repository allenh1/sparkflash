//! Measurement harness for `docs/archive/build/reference-parallelism.md`.
//!
//! Builds the released configuration with only the first `layers` backbone layers (no MTP stages, no
//! Engram tables, no KV-source layers), then times a 130-token prefill and `decode_steps` decode
//! steps with a generated weight source. The source returns the plan's exact dtypes and shapes, so
//! every read, decode, scale pass and matmul has the released cost; only the values are synthetic,
//! which the timing does not depend on. A warm run fills the source's cache first, so the two timed
//! runs time the reference pass, not the generator.
//!
//! Usage:
//!   cargo run --release -p sparkflash-engine --example reference_parallel_bench -- [layers] [decode_steps]
//!
//! This is a measurement tool for the parallel-reference change under
//! `docs/archive/build/reference-parallelism.md`. It is not a reference run and never generates a fixture.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use sparkflash_engine::config::ModelConfig;
use sparkflash_engine::dtype::DType;
use sparkflash_engine::forward::{InferenceEngine, SequenceState};
use sparkflash_engine::parallel::with_reference_threads;
use sparkflash_engine::tensor::{TensorPlan, TensorSpec};
use sparkflash_engine::weights::{WeightError, WeightSource};

struct RandSource {
    plan: TensorPlan,
    cache: Mutex<HashMap<String, Arc<Vec<u8>>>>,
}

struct Shared(Arc<RandSource>);

impl WeightSource for Shared {
    fn names(&self) -> Vec<String> {
        self.0.names()
    }
    fn spec(&self, name: &str) -> Result<TensorSpec, WeightError> {
        self.0.spec(name)
    }
    fn read(&self, name: &str) -> Result<Vec<u8>, WeightError> {
        self.0.read(name)
    }
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn byte(&mut self) -> u8 {
        (self.next() >> 24) as u8
    }
}

impl RandSource {
    fn new(plan: TensorPlan) -> Self {
        Self {
            plan,
            cache: Mutex::new(HashMap::new()),
        }
    }

    fn generate(spec: &TensorSpec) -> Vec<u8> {
        let bytes = spec.storage_bytes() as usize;
        let mut out = vec![0u8; bytes];
        let mut rng = Rng(0x5EED_1234_ABCD_0001 ^ bytes as u64);
        let scale = spec.name.ends_with(".scale");
        match spec.dtype {
            _ if scale => out.fill(127),
            DType::BF16 => {
                for pair in out.as_chunks_mut::<2>().0 {
                    let exponent = 120 + (rng.next() % 16) as u16;
                    let mantissa = (rng.next() % 128) as u16;
                    let sign = (rng.next() % 2) as u16;
                    let bits = (sign << 15) | (exponent << 7) | mantissa;
                    pair[0] = (bits & 0xFF) as u8;
                    pair[1] = (bits >> 8) as u8;
                }
            }
            DType::F32 => {
                for word in out.as_chunks_mut::<4>().0 {
                    let exponent = 118 + (rng.next() % 20) as u32;
                    let mantissa = (rng.next() & 0x007F_FFFF) as u32;
                    let sign = (rng.next() % 2) as u32;
                    let bits = (sign << 31) | (exponent << 23) | mantissa;
                    word.copy_from_slice(&bits.to_le_bytes());
                }
            }
            DType::F8E4M3 => {
                for byte in &mut out {
                    let mut code = rng.byte();
                    if code & 0x7F == 0x7F {
                        code &= 0xFE;
                    }
                    *byte = code;
                }
            }
            DType::F8E8M0 => out.fill(127),
            DType::F4E2M1 => {
                for byte in &mut out {
                    *byte = rng.byte();
                }
            }
            DType::F16 => panic!("the released checkpoint has no f16 tensors"),
        }
        out
    }
}

impl WeightSource for RandSource {
    fn names(&self) -> Vec<String> {
        self.plan.names().map(str::to_owned).collect()
    }

    fn spec(&self, name: &str) -> Result<TensorSpec, WeightError> {
        self.plan
            .get(name)
            .cloned()
            .ok_or_else(|| WeightError::Missing {
                name: name.to_owned(),
            })
    }

    fn read(&self, name: &str) -> Result<Vec<u8>, WeightError> {
        if let Some(cached) = self.cache.lock().expect("cache").get(name) {
            return Ok((**cached).clone());
        }
        let spec = self.spec(name)?;
        let bytes = RandSource::generate(&spec);
        let arc = Arc::new(bytes);
        self.cache
            .lock()
            .expect("cache")
            .insert(name.to_owned(), Arc::clone(&arc));
        Ok((*arc).clone())
    }
}

fn config_with(layers: usize) -> ModelConfig {
    let mut config = ModelConfig::v41_flash().expect("released runtime config");
    config.n_layers = layers;
    config.n_mtp_layers = 0;
    config.compress_ratios.truncate(layers);
    config.kv_source_layers.retain(|&layer| layer < layers);
    config.index_source_layers.retain(|&layer| layer < layers);
    config.candidate_source_layer = 0;
    config.dspark_target_layer_ids.clear();
    config.engram_layer_ids.clear();
    config.engram_num_embeddings.clear();
    config
}

fn run(
    source: &Arc<RandSource>,
    config: &ModelConfig,
    threads: usize,
    decode_steps: usize,
) -> (f64, f64) {
    let engine = InferenceEngine::new(
        config.clone(),
        Box::new(Shared(Arc::clone(source))),
        1 << 30,
    )
    .expect("engine");
    let tokens: Vec<u32> = (0..130u32)
        .map(|index| (index * 7 + 3) % config.vocab_size as u32)
        .collect();
    let mut state = SequenceState::new(config);
    let start = Instant::now();
    with_reference_threads(threads, || {
        engine.prefill(&mut state, &tokens).expect("prefill");
    });
    let prefill = start.elapsed().as_secs_f64();
    let start = Instant::now();
    with_reference_threads(threads, || {
        for step in 0..decode_steps {
            let token = state.tokens().last().copied().unwrap_or(0) ^ (step as u32 + 1);
            engine.decode(&mut state, token).expect("decode");
        }
    });
    let decode = start.elapsed().as_secs_f64();
    (prefill, decode)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let layers: usize = args
        .first()
        .map(|text| text.parse().expect("layers"))
        .unwrap_or(2);
    let decode_steps: usize = args
        .get(1)
        .map(|text| text.parse().expect("steps"))
        .unwrap_or(2);
    let config = config_with(layers);
    config.validate().expect("the sliced config must be valid");
    let plan = TensorPlan::language_model(&config);
    let source = Arc::new(RandSource::new(plan));
    // Warm run: the synthetic source caches every tensor the 130-token prefill and the decodes
    // read, so both measured runs time the reference pass alone, not the generator.
    let (warm_prefill, _) = run(&source, &config, 20, decode_steps);
    let (parallel_prefill, parallel_decode) = run(&source, &config, 20, decode_steps);
    let (serial_prefill, serial_decode) = run(&source, &config, 1, decode_steps);
    println!(
        "layers={layers} decode_steps={decode_steps} warm_prefill_s={warm_prefill:.2} parallel_prefill_s={parallel_prefill:.2} serial_prefill_s={serial_prefill:.2} parallel_decode_s={parallel_decode:.2} serial_decode_s={serial_decode:.2} prefill_speedup={:.2} decode_speedup={:.2}",
        serial_prefill / parallel_prefill,
        serial_decode / parallel_decode
    );
}
