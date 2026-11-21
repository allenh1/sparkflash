// Copyright (c) 2026 Hunter L. Allen
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! `sparkflash arms --list|--ledger|--crate ARM`: the serving arms, the strategies each states,
//! and the crate an arm's code lives in.
//!
//! `--list` reads `sparkflash_model::ARMS`, the one model table: the console id, the registry
//! weight selectors, the default context, the tokenizer and chat template, the conversion
//! defaults and the ledger pointer. `--ledger` reads `sparkflash_model::ARM_LEDGERS`, the same
//! declaration an arm's ledger returns at boot, so the gap table is a preview of the
//! boot line and not a second hand-kept list. No device is opened: an arm with no checkpoint and
//! no cubin still prints.

use std::process::ExitCode;

use sparkflash_model::{ARM_LEDGERS, ARMS, ArmLedger, Strategy};

/// Render the arm table as Markdown: one row an entry in `sparkflash_model::ARMS`.
#[must_use]
pub fn render_arms() -> String {
    let mut out = String::from(
        "| arm | model id | weights | context | tokenizer | chat template | conversion | ledger |\n",
    );
    out.push_str("| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |\n");
    for arm in ARMS {
        let weights = if arm.weights().is_empty() {
            "the checkpoint".to_owned()
        } else {
            arm.weights().join(", ")
        };
        let conversion = arm.conversion();
        let released = if conversion.released() {
            "released"
        } else {
            "dynamic"
        };
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | TP{} {} | {} |\n",
            arm.id(),
            arm.model_id(),
            weights.replace('|', "\\|"),
            if arm.default_context() == 0 {
                "the checkpoint's".to_owned()
            } else {
                arm.default_context().to_string()
            },
            arm.tokenizer().tokenizer_id,
            arm.tokenizer().chat_template(),
            conversion.world_size,
            released,
            arm.ledger().arm,
        ));
    }
    out
}

/// Render the gap table as Markdown: one row an arm, one column a strategy.
#[must_use]
pub fn render() -> String {
    let mut out = String::new();
    out.push_str("| arm | numerics |");
    for strategy in Strategy::ALL {
        out.push_str(&format!(" {} |", strategy.name()));
    }
    out.push('\n');
    out.push_str("| :-- | :-- |");
    for _ in Strategy::ALL {
        out.push_str(" :-- |");
    }
    out.push('\n');
    for declaration in ARM_LEDGERS {
        out.push_str(&format!(
            "| {} | {} |",
            declaration.arm,
            numerics(declaration)
        ));
        for strategy in Strategy::ALL {
            let state = declaration
                .state(strategy)
                .map_or_else(|| "unstated".to_owned(), |state| state.to_string());
            out.push_str(&format!(" {} |", state.replace('|', "\\|")));
        }
        out.push('\n');
    }
    out
}

/// What the numerics registry holds for an arm's weights ([sparkflash_model::numerics_status]).
fn numerics(declaration: &ArmLedger) -> &'static str {
    sparkflash_model::weights_numerics_status(&declaration.weights())
}

/// Entry point called by `main.rs` with the arguments after `arms`.
pub fn run(mut args: impl Iterator<Item = String>) -> ExitCode {
    const USAGE: &str = "usage: sparkflash arms [--list] [--ledger] | --crate ARM";
    let mut list = false;
    let mut ledger = false;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--list" => list = true,
            "--ledger" => ledger = true,
            "--crate" => {
                let Some(name) = args.next() else {
                    eprintln!("sparkflash arms: --crate needs an arm; {USAGE}");
                    return ExitCode::FAILURE;
                };
                if list || ledger || args.next().is_some() {
                    eprintln!("sparkflash arms: --crate takes one arm and nothing else; {USAGE}");
                    return ExitCode::FAILURE;
                }
                let Some(krate) = sparkflash_model::crate_for(&name) else {
                    let known: Vec<&str> = ARMS.iter().map(|arm| arm.id()).collect();
                    eprintln!("sparkflash arms: no arm {name:?}; the arms are {known:?}");
                    return ExitCode::FAILURE;
                };
                println!("{krate}");
                return ExitCode::SUCCESS;
            }
            other => {
                eprintln!("sparkflash arms: unknown argument {other:?}; {USAGE}");
                return ExitCode::FAILURE;
            }
        }
    }
    if !list && !ledger {
        eprintln!("sparkflash arms: {USAGE}");
        return ExitCode::FAILURE;
    }
    if list {
        print!("{}", render_arms());
    }
    if ledger {
        print!("{}", render());
    }
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_arm_table_names_every_entry_and_its_items() {
        let text = render_arms();
        for arm in ARMS {
            assert!(text.contains(arm.id()), "missing {}", arm.id());
            assert!(text.contains(arm.model_id()), "missing {}", arm.model_id());
            assert!(
                text.contains(arm.tokenizer().chat_template()),
                "missing {}'s chat template",
                arm.id()
            );
            for weights in arm.weights() {
                assert!(text.contains(weights), "missing {weights}");
            }
            assert!(
                text.contains(arm.ledger().arm),
                "missing {}'s ledger",
                arm.id()
            );
        }
        assert!(text.starts_with("| arm | model id | weights |"));
    }

    #[test]
    fn the_table_names_every_arm_and_every_strategy() {
        let text = render();
        for declaration in ARM_LEDGERS {
            assert!(
                text.contains(declaration.arm),
                "missing {}",
                declaration.arm
            );
        }
        for strategy in Strategy::ALL {
            assert!(
                text.contains(strategy.name()),
                "missing {}",
                strategy.name()
            );
        }
        assert!(!text.contains("unstated"), "a declared arm left a row out");
    }

    #[test]
    fn an_arm_with_no_capture_has_a_registered_uncaptured_entry() {
        let registry = sparkflash_model::numerics::Registry::embedded()
            .expect("the embedded numerics registry parses");
        for declaration in ARM_LEDGERS {
            let weights = declaration.weights();
            if weights.is_empty() {
                // The dense arm serves no registry model, so the registry cannot name it.
                continue;
            }
            if !numerics(declaration).starts_with("none") {
                continue;
            }
            let uncaptured = registry.config.iter().any(|entry| {
                !entry.is_captured()
                    && sparkflash_model::find_model(&entry.model)
                        .is_some_and(|model| weights.contains(&model.weights_id.as_str()))
            });
            assert!(
                uncaptured,
                "{} shows numerics: none but has no uncaptured entry",
                declaration.arm
            );
        }
    }
}
