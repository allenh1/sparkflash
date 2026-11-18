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
