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

//! What a serving arm writes the same way in every pack.
//!
//! A pack owns its arithmetic, its tensors and its kernels. Three things it does *not* own are
//! the identity the control plane reports, the KV arithmetic an operator's flags are checked
//! against, and the rule that one session's caches fit beside its siblings. Each was written out
//! per pack, which is how a pack ends up advertising a window it cannot allocate or accepting a
//! `--concurrency` it cannot hold. This module is the one copy.
//!
//! * [PackIdentity] turns a pack's four constants plus the checkpoint's own
//!   [Architecture](crate::architecture::Architecture) into a [ModelDescriptor]. The window and
//!   the vocabulary come from the released document, so a pack cannot report a shape the
//!   checkpoint does not have ([D69](../../../docs/DECISIONS.md), col. "Decision").
//! * [KvPlan] is a session's cache claim: bytes a context position adds, and bytes that do not
//!   depend on the window at all. A pack derives it from its own geometry (never from a constant
//!   an operator is expected to keep in step), and [KvPlan::admit] answers whether a
//!   `--concurrency` at a `--context` fits a budget.
//!
//! Nothing here chooses arithmetic, reads a checkpoint or touches a device.

use crate::architecture::Architecture;
use crate::capabilities::Capabilities;
use crate::descriptor::ModelDescriptor;

/// A pack's fixed identity: what `/v1/models` reports and the hub cache resolves.
///
/// The window and the vocabulary are deliberately absent. They belong to the checkpoint the pack
/// parsed, so a build that serves a shorter-window release of the same family reports that
/// release's window instead of a constant that drifted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackIdentity {
    /// The id `/v1/models` reports and the hub cache resolves.
    pub model_id: &'static str,
    /// The family the descriptor reports and the control plane groups by.
    pub family: &'static str,
    /// The released `config.json` `model_type` the checkpoint carries.
    pub model_type: &'static str,
    /// The short family label a control surface shows.
    pub label: &'static str,
    /// The control-plane weight selector `--model` also takes.
    pub weights_id: &'static str,
    /// The protocol ceiling on one request's generated tokens.
    pub max_output_tokens: usize,
}

impl PackIdentity {
    /// What this arm serves at `context` positions of `architecture`.
    ///
    /// `context` is the window the arm actually allocated, capped by the checkpoint's own: a
    /// server asked for more than the release supports reports the release's window rather than
    /// a promise it cannot allocate.
    #[must_use]
    pub fn descriptor(
        &self,
        architecture: &Architecture,
        context: usize,
        capabilities: Capabilities,
    ) -> ModelDescriptor {
        ModelDescriptor {
            id: self.model_id.to_owned(),
            family: self.family.to_owned(),
            model_type: self.model_type.to_owned(),
            label: self.label.to_owned(),
            weights_id: self.weights_id.to_owned(),
            context_tokens: context.min(architecture.max_position_embeddings),
            max_output_tokens: self.max_output_tokens,
            vocab_size: architecture.vocab_size,
            capabilities,
        }
    }
}

/// What one session's caches claim, and the admission rule over it.
///
/// The two terms are the ones a serving flag can move. `per_token` is everything that grows
/// with the context (key and value caches, an indexer's keys, a compressed latent); `fixed` is
/// everything a session allocates regardless of the window (a recurrent state, its verify spare,
/// a drafter's row scratch that is keyed by rows rather than positions). A pack derives both from
/// its own geometry, so `--context 65536` costs what the geometry says rather than what a
/// comment says.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvPlan {
    /// Bytes one context position adds to a session.
    per_token: u64,
    /// Bytes a session allocates whatever its window.
    fixed: u64,
}

/// The cache claim one `--concurrency` at one `--context` makes, and the budget it was checked
/// against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Admission {
    /// Bytes one session claims at the context.
    pub per_session: u64,
    /// Bytes every session claims together.
    pub total: u64,
    /// The budget the total was checked against.
    pub budget: u64,
}

/// Why a concurrency and a context do not fit the budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdmissionRefusal {
    /// The sessions asked for.
    pub concurrency: usize,
    /// The window asked for.
    pub context: usize,
    /// The claim the pair makes.
    pub claimed: Admission,
}

impl KvPlan {
    /// A plan of `per_token` bytes a position and `fixed` bytes a session.
    #[must_use]
    pub const fn new(per_token: u64, fixed: u64) -> Self {
        Self { per_token, fixed }
    }

    /// Bytes one context position adds to a session.
    #[must_use]
    pub const fn per_token(&self) -> u64 {
        self.per_token
    }

    /// Bytes a session allocates whatever its window.
    #[must_use]
    pub const fn fixed(&self) -> u64 {
        self.fixed
    }

    /// Bytes one session claims at `context` positions.
    #[must_use]
    pub const fn sequence_bytes(&self, context: usize) -> u64 {
        context as u64 * self.per_token + self.fixed
    }

    /// Bytes `concurrency` sessions claim at `context` positions.
    #[must_use]
    pub const fn concurrency_bytes(&self, context: usize, concurrency: usize) -> u64 {
        self.sequence_bytes(context) * concurrency as u64
    }

    /// Whether `concurrency` sessions at `context` fit `budget`.
    ///
    /// # Errors
    /// The refusal carries the pair and the claim, so the caller's message names the same
    /// numbers the check used.
    pub fn admit(
        &self,
        context: usize,
        concurrency: usize,
        budget: u64,
    ) -> Result<Admission, AdmissionRefusal> {
        let claim = Admission {
            per_session: self.sequence_bytes(context),
            total: self.concurrency_bytes(context, concurrency),
            budget,
        };
        if claim.total > budget {
            return Err(AdmissionRefusal {
                concurrency,
                context,
                claimed: claim,
            });
        }
        Ok(claim)
    }
}

impl AdmissionRefusal {
    /// The operator-facing explanation, in the words every arm now uses.
    #[must_use]
    pub fn message(&self) -> String {
        format!(
            "--concurrency {} at --context {} needs {} of KV caches, over the {} budget; lower \
             --concurrency or --context, or raise --kv-budget-gib on a node with more room",
            self.concurrency,
            self.context,
            gib(self.claimed.total),
            gib(self.claimed.budget),
        )
    }
}

/// The bytes a budget of `gib` gibibytes allows.
#[must_use]
pub const fn kv_budget_bytes(gib: usize) -> u64 {
    (gib as u64) << 30
}

/// The per-node unified-memory ceiling of one DGX Spark: 121.7 GiB.
///
/// This is what the GB10 reports as its total after the firmware's reservation, and the figure
/// every per-node budget in the workspace is taken against: the KV layout's pools, the Engram
/// placement decision and the web console's fit estimate all read it from here.
pub const NODE_MEMORY_BUDGET_BYTES: u64 = 1217 * (1 << 30) / 10;

/// Guard band a node must leave unallocated: 8 GiB.
///
/// The specification's own budget table (spec line 180) books a 9.12 GiB "MemAvailable floor
/// guard" and the preflight gate wants MemAvailable >= 114 GiB before weights load; the
/// milestone task states the floor as 8 GiB. Both are exposed (the 9.12 GiB figure is
/// `sparkflash_kv::layout::SPEC_SAFETY_HEADROOM_BYTES`) so a caller can pick, and the
/// discrepancy is reported rather than silently reconciled.
pub const MEMORY_FLOOR_BYTES: u64 = 8 * (1 << 30);

/// A byte count as the operator-facing gibibytes, to two places.
#[must_use]
pub fn gib(bytes: u64) -> String {
    format!("{:.1} GiB", bytes as f64 / (1u64 << 30) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A released document, so the fixture exercises the parser the arms actually read.
    const DOCUMENT: &str =
        include_str!("../../sparkflash-model-glm5-next/config/glm53_flash.hf.json");

    fn architecture() -> Architecture {
        Architecture::from_json(DOCUMENT).expect("the released document parses")
    }

    const IDENTITY: PackIdentity = PackIdentity {
        model_id: "example/toy",
        family: "toy",
        model_type: "toy",
        label: "Toy",
        weights_id: "toy",
        max_output_tokens: 128,
    };

    #[test]
    fn a_descriptor_takes_its_window_and_vocabulary_from_the_checkpoint() {
        let architecture = architecture();
        let descriptor = IDENTITY.descriptor(&architecture, 2048, Capabilities::ALL);
        assert_eq!(descriptor.id, "example/toy");
        assert_eq!(descriptor.context_tokens, 2048);
        assert_eq!(descriptor.vocab_size, architecture.vocab_size);
        assert_eq!(descriptor.max_output_tokens, 128);
        assert_eq!(descriptor.capabilities, Capabilities::ALL);
    }

    #[test]
    fn a_descriptor_never_reports_a_window_past_the_checkpoint() {
        let architecture = architecture();
        let past = architecture.max_position_embeddings + 1;
        let descriptor = IDENTITY.descriptor(&architecture, past, Capabilities::ALL);
        assert_eq!(
            descriptor.context_tokens,
            architecture.max_position_embeddings
        );
    }

    #[test]
    fn a_plan_counts_the_window_and_the_fixed_claim() {
        let plan = KvPlan::new(1000, 500);
        assert_eq!(plan.sequence_bytes(10), 10_500);
        assert_eq!(plan.concurrency_bytes(10, 3), 31_500);
        assert_eq!(plan.per_token(), 1000);
        assert_eq!(plan.fixed(), 500);
    }

    #[test]
    fn admission_reports_the_claim_it_made_and_refuses_one_over_budget() {
        let plan = KvPlan::new(100_000, 0);
        let admitted = plan
            .admit(1000, 2, 300_000_000)
            .expect("200,000,000 fits 300,000,000");
        assert_eq!(admitted.total, 200_000_000);
        let refusal = plan
            .admit(1000, 4, 300_000_000)
            .expect_err("400,000,000 is over 300,000,000");
        assert_eq!(refusal.concurrency, 4);
        assert_eq!(refusal.context, 1000);
        assert_eq!(refusal.claimed.total, 400_000_000);
        assert_eq!(refusal.claimed.budget, 300_000_000);
        assert!(refusal.message().contains("--concurrency 4"));
        assert!(refusal.message().contains("0.4 GiB"));
    }
}
