//! DSpark acceptance replay: the host twin's draft head over a device run's taps.
//!
//! A device run with `SPARKFLASH_GENERATE=1 SPARKFLASH_TAP_DUMP=<file>` writes, for every position,
//! the input token, the head's argmax and the three DSpark taps (the hc-mean of the inputs of layers
//! 37-39, model.py:1262-1266). This tool seeds the MTP rings with every position and drafts at every
//! generated position with the reference arithmetic (`forward::mtp`), then replays greedy
//! verification over the device's own greedy sequence: a verify step from anchor p accepts the
//! longest draft prefix equal to the tokens the backbone actually produced, plus the bonus token.
//!
//! Subcommands:
//!   encode-chat <tokenizer.json> <text-file>     prints the chat-encoded prompt ids, comma separated
//!   decode <tokenizer.json> <dump>               prints the dumped sequence as text
//!   replay <checkpoint-dir> <dump> <out.jsonl>   drafts every generated position and reports AL

use std::io::{Read, Write};
use std::path::Path;

use sparkflash_engine::config::ModelConfig;
use sparkflash_engine::forward::attention::AttnState;
use sparkflash_engine::forward::mtp;
use sparkflash_engine::linalg::CHECKPOINT_TENSOR_LIMIT;
use sparkflash_engine::safetensors::SafeTensorsSource;

const MAGIC: &[u8; 4] = b"DSPT";

struct Dump {
    prompt_len: usize,
    tokens: Vec<u32>,
    next: Vec<u32>,
    taps: Vec<Vec<f32>>,
}

fn read_u32(reader: &mut impl Read) -> std::io::Result<u32> {
    let mut word = [0u8; 4];
    reader.read_exact(&mut word)?;
    Ok(u32::from_le_bytes(word))
}

fn read_dump(path: &Path) -> Result<Dump, String> {
    let file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut reader = std::io::BufReader::new(file);
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic).map_err(|e| e.to_string())?;
    if &magic != MAGIC {
        return Err(format!("{} is not a tap dump", path.display()));
    }
    let width = read_u32(&mut reader).map_err(|e| e.to_string())? as usize;
    let prompt_len = read_u32(&mut reader).map_err(|e| e.to_string())? as usize;
    let mut dump = Dump {
        prompt_len,
        tokens: Vec::new(),
        next: Vec::new(),
        taps: Vec::new(),
    };
    loop {
        let position = match read_u32(&mut reader) {
            Ok(position) => position as usize,
            Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(error) => return Err(error.to_string()),
        };
        if position != dump.tokens.len() {
            return Err(format!(
                "record for position {position} after {} records",
                dump.tokens.len()
            ));
        }
        dump.tokens
            .push(read_u32(&mut reader).map_err(|e| e.to_string())?);
        dump.next
            .push(read_u32(&mut reader).map_err(|e| e.to_string())?);
        let mut bytes = vec![0u8; width * 4];
        reader.read_exact(&mut bytes).map_err(|e| e.to_string())?;
        dump.taps.push(
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|w| f32::from_le_bytes(*w))
                .collect(),
        );
    }
    Ok(dump)
}

/// The full sequence the run consumed and produced: every input, then the last head's argmax.
fn sequence(dump: &Dump) -> Vec<u32> {
    let mut ids = dump.tokens.clone();
    if let Some(&last) = dump.next.last() {
        ids.push(last);
    }
    ids
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("encode-chat") if args.len() == 4 => {
            let seam = sparkflash_model_deepseek_v41::encoding::TokenizerSeam::from_path(Path::new(&args[2]))
                .map_err(|e| e.to_string())?;
            let text = std::fs::read_to_string(&args[3]).map_err(|e| e.to_string())?;
            let request = serde_json::json!({"messages": [{"role": "user", "content": text.trim()}]});
            let ids = seam.encode_chat(&request).map_err(|e| format!("{e:?}"))?;
            let joined: Vec<String> = ids.iter().map(u32::to_string).collect();
            println!("{}", joined.join(","));
            Ok(())
        }
        Some("decode") if args.len() == 4 => {
            let seam = sparkflash_model_deepseek_v41::encoding::TokenizerSeam::from_path(Path::new(&args[2]))
                .map_err(|e| e.to_string())?;
            let dump = read_dump(Path::new(&args[3]))?;
            let ids = sequence(&dump);
            let (prompt, generated) = ids.split_at(dump.prompt_len.min(ids.len()));
            println!("--- prompt ({} ids)\n{}", prompt.len(), seam.decode(prompt).unwrap_or_default());
            println!(
                "--- generated ({} ids)\n{}",
                generated.len(),
                seam.decode(generated).unwrap_or_default()
            );
            Ok(())
        }
        Some("replay") if args.len() == 5 => replay(&args[2], &args[3], &args[4]),
        _ => Err(
            "usage: dspark_replay encode-chat <tokenizer.json> <text> | decode <tokenizer.json> <dump> | replay <checkpoint> <dump> <out.jsonl>"
                .to_owned(),
        ),
    }
}

fn replay(checkpoint: &str, dump_path: &str, out_path: &str) -> Result<(), String> {
    let config = ModelConfig::v41_flash_cached();
    let dir = Path::new(checkpoint);
    let src = SafeTensorsSource::open(dir, &dir.join("model.safetensors.index.json"))
        .map_err(|e| e.to_string())?;
    let dump = read_dump(Path::new(dump_path))?;
    let ids = sequence(&dump);
    let block = config.dspark_block_size;
    let mut stages: Vec<AttnState> = (0..config.n_mtp_layers)
        .map(|_| AttnState::for_config(config))
        .collect();
    let first_anchor = dump.prompt_len.saturating_sub(1);
    // drafts[p] = the block drafted at anchor p, for every generated position with a full block of
    // the true sequence after it.
    let mut drafts: Vec<Option<Vec<u32>>> = vec![None; dump.tokens.len()];
    let mut out =
        std::io::BufWriter::new(std::fs::File::create(out_path).map_err(|e| e.to_string())?);
    let started = std::time::Instant::now();
    for position in 0..dump.tokens.len() {
        let taps = &dump.taps[position];
        let wanted = position >= first_anchor && position + 1 + block < ids.len() && position > 0;
        if wanted {
            let output = mtp::dspark_forward(
                config,
                &dump.tokens[position..=position],
                position,
                taps,
                dump.next[position],
                &mut stages,
                &src,
                CHECKPOINT_TENSOR_LIMIT,
            )
            .map_err(|e| format!("position {position}: {e}"))?;
            let block_ids = output.draft[1..].to_vec();
            let truth = &ids[position + 2..position + 2 + block];
            let matched = block_ids
                .iter()
                .zip(truth)
                .take_while(|(draft, real)| draft == real)
                .count();
            writeln!(
                out,
                "{}",
                serde_json::json!({
                    "position": position,
                    "anchor": dump.next[position],
                    "draft": block_ids,
                    "truth": truth,
                    "matched": matched,
                    "confidence": output.confidence,
                })
            )
            .map_err(|e| e.to_string())?;
            out.flush().map_err(|e| e.to_string())?;
            eprintln!(
                "position {position}: matched {matched}/{block} ({:.1}s)",
                started.elapsed().as_secs_f64()
            );
            drafts[position] = Some(block_ids);
        } else {
            mtp::dspark_seed(
                config,
                position,
                taps,
                &mut stages,
                &src,
                CHECKPOINT_TENSOR_LIMIT,
            )
            .map_err(|e| format!("seed position {position}: {e}"))?;
        }
    }
    // Greedy verification replay for each draft length k: from anchor p, accept the matching prefix
    // of the first k drafts, emit it plus the bonus token, and continue from the last accepted row.
    for k in 1..=block {
        let (mut steps, mut emitted) = (0usize, 0usize);
        let mut anchor = first_anchor;
        while let Some(Some(block_ids)) = drafts.get(anchor) {
            let truth = &ids[anchor + 2..anchor + 2 + block];
            let matched = block_ids[..k]
                .iter()
                .zip(truth)
                .take_while(|(draft, real)| draft == real)
                .count();
            steps += 1;
            emitted += matched + 1;
            anchor += matched + 1;
        }
        let al = emitted as f64 / steps.max(1) as f64;
        println!("k={k}: {steps} verify steps, {emitted} tokens, AL {al:.3}");
    }
    let per_position: Vec<usize> = drafts
        .iter()
        .flatten()
        .zip(
            dump.tokens
                .iter()
                .enumerate()
                .filter_map(|(p, _)| drafts[p].as_ref().map(|_| p)),
        )
        .map(|(block_ids, p)| {
            block_ids
                .iter()
                .zip(&ids[p + 2..p + 2 + block])
                .take_while(|(d, r)| d == r)
                .count()
        })
        .collect();
    let mean = per_position.iter().sum::<usize>() as f64 / per_position.len().max(1) as f64;
    println!(
        "every-position mean matched prefix {mean:.3} over {} positions",
        per_position.len()
    );
    Ok(())
}
