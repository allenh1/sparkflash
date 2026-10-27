//! `sparkflash models`: what this binary can serve.
//!
//! The command reads the compile-time registry rather than a second table, so the list is the
//! same fact the web control plane and the HTTP surface use.

use std::process::ExitCode;

/// Render the model and engine catalogue.
#[must_use]
pub fn render() -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "{:<38} {:<24} {:>10} {:>10} {:>10}  capabilities\n",
        "id", "weights", "context", "max out", "vocab"
    ));
    for model in sparkflash_model::MODELS.iter() {
        let capabilities = format!(
            "{}{}{}{}",
            if model.capabilities.tools {
                "tools "
            } else {
                ""
            },
            if model.capabilities.thinking {
                "thinking "
            } else {
                ""
            },
            if model.capabilities.reasoning_effort {
                "reasoning "
            } else {
                ""
            },
            if model.capabilities.vision {
                "vision"
            } else {
                ""
            }
        );
        out.push_str(&format!(
            "{:<38} {:<24} {:>10} {:>10} {:>10}  {}\n",
            model.id,
            model.weights_id,
            model.context_tokens,
            model.max_output_tokens,
            model.vocab_size,
            capabilities.trim_end()
        ));
    }
    out.push_str(&format!(
        "engines: {}\n",
        sparkflash_model::ENGINE_NAMES.join(", ")
    ));
    out
}

/// Entry point called by `main.rs`.
pub fn run() -> ExitCode {
    print!("{}", render());
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalogue_names_every_registered_model_and_engine() {
        let text = render();
        for model in sparkflash_model::MODELS.iter() {
            assert!(text.contains(&model.id), "missing {}", model.id);
            assert!(
                text.contains(&model.weights_id),
                "missing {}",
                model.weights_id
            );
        }
        for engine in sparkflash_model::ENGINE_NAMES {
            assert!(text.contains(engine), "missing engine {engine}");
        }
    }
}
