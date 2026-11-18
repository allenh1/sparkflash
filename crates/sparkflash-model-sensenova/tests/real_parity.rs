// Real-checkpoint parity, run explicitly with SENSENOVA_FP8_DIR set.

use std::path::PathBuf;

use sparkflash_model::ModelDescriptor;
use sparkflash_weights::safetensors::open_checkpoint;

use sparkflash_model_sensenova::SenseNovaConfig;
use sparkflash_model_sensenova::forward::SenseNovaWeights;

#[test]
#[ignore = "loads the 11 GB release; run with SENSENOVA_FP8_DIR=<dir> --ignored"]
fn the_device_and_host_agree_on_the_first_token() {
    let directory = PathBuf::from(std::env::var("SENSENOVA_FP8_DIR").expect("SENSENOVA_FP8_DIR"));
    let config = SenseNovaConfig::from_checkpoint_dir(&directory).expect("config");
    let source = open_checkpoint(&directory).expect("checkpoint");
    let host_weights = SenseNovaWeights::load(&config, &source).expect("host weights");
    let host = host_weights.forward_prompt(&[1]).expect("host token");
    let backend = sparkflash_model_sensenova::SenseNovaDeviceBackend::open(
        &directory,
        ModelDescriptor::sensenova_loop_9b(),
        16,
    )
    .expect("device backend");
    use sparkflash_model::backend::ModelBackend;
    let mut session = backend.open_session().expect("session");
    let step = session.prefill(&[1]).expect("device token");
    let worst = step
        .logits
        .iter()
        .zip(&host)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let host_finite = host.iter().filter(|value| value.is_finite()).count();
    let device_finite = step.logits.iter().filter(|value| value.is_finite()).count();
    let host_arg = host
        .iter()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(index, _)| index);
    let device_arg = step
        .logits
        .iter()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(index, _)| index);
    eprintln!(
        "host argmax {host_arg:?} ({host_finite} finite), device argmax {device_arg:?} ({device_finite} finite), worst {worst}; host[0..4]={:?} device[0..4]={:?}",
        &host[..4.min(host.len())],
        &step.logits[..4.min(step.logits.len())],
    );
    assert_eq!(host_arg, device_arg);
    // The device keeps the 8-bit K/V cache the 160K default needs; its E4M3 round is the
    // only difference from the f32 host reference at this position.
    assert!(worst < 0.1, "worst {worst}");
}

/// A prompt and a few greedy steps: the chunk path first, then the decode path, compared on both
/// arms step by step. A single token cannot tell a broken chunk from a broken cache, and the text
/// is the only other witness.
#[test]
#[ignore = "loads the 11 GB release; run with SENSENOVA_FP8_DIR=<dir> --ignored"]
fn the_device_tracks_the_host_over_a_prompt_and_a_few_steps() {
    use sparkflash_model::backend::ModelBackend;
    let directory = PathBuf::from(std::env::var("SENSENOVA_FP8_DIR").expect("SENSENOVA_FP8_DIR"));
    let config = SenseNovaConfig::from_checkpoint_dir(&directory).expect("config");
    let source = open_checkpoint(&directory).expect("checkpoint");
    let descriptor = ModelDescriptor::sensenova_loop_9b();
    let host = sparkflash_model_sensenova::SenseNovaHostBackend::open(
        descriptor.clone(),
        config.clone(),
        &source,
        64,
    )
    .expect("host backend");
    let device = sparkflash_model_sensenova::SenseNovaDeviceBackend::open(&directory, descriptor, 4096)
        .expect("device backend");
    let prompt = [1u32, 2, 3, 4, 5];
    let mut host_session = host.open_session().expect("host session");
    let mut device_session = device.open_session().expect("device session");
    let argmax = |values: &[f32]| {
        values
            .iter()
            .enumerate()
            .filter(|(_, value)| value.is_finite())
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(index, _)| index)
            .unwrap_or(0)
    };
    let mut host_logits = host_session.prefill(&prompt).expect("host prefill").logits;
    let mut device_logits = device_session.prefill(&prompt).expect("device prefill").logits;
    for step in 0..4 {
        let worst = host_logits
            .iter()
            .zip(&device_logits)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!(
            "step {step}: host argmax {} device argmax {}, worst {worst}",
            argmax(&host_logits),
            argmax(&device_logits)
        );
        let next = argmax(&host_logits) as u32;
        host_logits = host_session.decode(next).expect("host decode").logits;
        device_logits = device_session.decode(next).expect("device decode").logits;
    }
    let worst = host_logits
        .iter()
        .zip(&device_logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!(
        "step 4: host argmax {} device argmax {}, worst {worst}",
        argmax(&host_logits),
        argmax(&device_logits)
    );
}
