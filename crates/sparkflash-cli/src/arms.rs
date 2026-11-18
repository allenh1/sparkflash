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

//! `sparkflash arms --ledger`: what each serving arm states of the runtime's strategies.
//!
//! The command reads `sparkflash_model::ARM_LEDGERS`, the same declaration an arm's
//! `FusedArm::ledger` returns at boot, so the gap table is a preview of the boot line and not a
//! second hand-kept list. No device is opened: an arm with no checkpoint and no cubin still
//! prints.

use std::process::ExitCode;

use sparkflash_model::{ARM_LEDGERS, Strategy};

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

/// What the numerics registry holds for an arm: a capture, a registered but uncaptured entry, or
/// no entry at all.
fn numerics(declaration: &sparkflash_model::ArmLedger) -> String {
    let registry = sparkflash_model::numerics::Registry::embedded()
        .expect("the embedded numerics registry parses");
    let mut registered = false;
    let mut captured = false;
    for entry in &registry.config {
        let Some(model) = sparkflash_model::find_model(&entry.model) else {
            continue;
        };
        if declaration.weights_ids.contains(&model.weights_id.as_str()) {
            registered = true;
            captured |= entry.is_captured();
        }
    }
    match (registered, captured) {
        (_, true) => "captured".to_owned(),
        (true, false) => "none (registered, uncaptured)".to_owned(),
        (false, false) => "none (no entry)".to_owned(),
    }
}

/// Entry point called by `main.rs` with the arguments after `arms`.
pub fn run(mut args: impl Iterator<Item = String>) -> ExitCode {
    let Some(flag) = args.next() else {
        eprintln!("sparkflash arms: usage: sparkflash arms --ledger");
        return ExitCode::FAILURE;
    };
    if flag != "--ledger" {
        eprintln!("sparkflash arms: unknown argument {flag:?}; usage: sparkflash arms --ledger");
        return ExitCode::FAILURE;
    }
    if let Some(extra) = args.next() {
        eprintln!(
            "sparkflash arms: unexpected argument {extra:?}; usage: sparkflash arms --ledger"
        );
        return ExitCode::FAILURE;
    }
    print!("{}", render());
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

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
            if declaration.weights_ids.is_empty() {
                // The dense arm serves no registry model, so the registry cannot name it.
                continue;
            }
            if !numerics(declaration).starts_with("none") {
                continue;
            }
            let uncaptured = registry.config.iter().any(|entry| {
                !entry.is_captured()
                    && sparkflash_model::find_model(&entry.model).is_some_and(|model| {
                        declaration.weights_ids.contains(&model.weights_id.as_str())
                    })
            });
            assert!(
                uncaptured,
                "{} shows numerics: none but has no uncaptured entry",
                declaration.arm
            );
        }
    }
}
