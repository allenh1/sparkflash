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

//! What `sparkflash onboard ORG/REPO --emit DIR` writes: the crate a family supplies, in the shape
//! of the worked example (`tests/onboard_example.rs`), or nothing when the catalogue alone runs it.
//!
//! [survey] sorts every ordered step of the checkpoint's layers ([Architecture::walk_roles]):
//!
//! * a token mixer or feed-forward whose kind [Dense::catalogue] registers is run by the shared
//!   dense host with no code;
//! * a token mixer or feed-forward it does not register is a part the family writes, a
//!   [sparkflash_runtime::PartExec] over the dense host; [emit] writes a stub whose bodies refuse
//!   by name;
//! * a residual mixer, a side lookup, a sparse indexer or a compression step is a step
//!   [sparkflash_runtime::PlanWalk] does not run yet ([sparkflash_runtime::plan_layer] refuses
//!   it), and a draft head or an image tower is a capability the walk does not host yet. Both are
//!   named, not stubbed: no part a family writes today can carry them.
//!
//! [emit] writes only inside its directory, and refuses one that exists.

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::path::Path;

use sparkflash_model::architecture::Switch;
use sparkflash_model::compose::StepRole;
use sparkflash_model::{Architecture, Part};

use crate::Dense;

/// The license header every emitted source carries: the repository's own.
const HEADER: &str = "// Licensed under the Apache License, Version 2.0 (the \"License\");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an \"AS IS\" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
";

/// How the shared path meets a checkpoint, step by step.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Survey {
    /// Token mixers and feed-forwards the catalogue registry runs.
    pub catalogue: BTreeSet<Part>,
    /// Token mixers and feed-forwards the family writes as parts.
    pub own: BTreeSet<Part>,
    /// Steps the shared walk does not run yet, by role.
    pub unhosted: BTreeSet<(StepRole, Part)>,
    /// Capabilities the shared walk does not host yet (a draft head, an image tower).
    pub capabilities: Vec<&'static str>,
    /// Why the shared dense host refuses the checkpoint once the family registers its own
    /// parts, if it still does: what the parts alone do not settle.
    pub refusal: Option<String>,
    /// Declarations the shared description does not model ([Architecture::unmodelled_features]).
    pub unmodelled: Vec<String>,
    /// Typed switches the checkpoint turns on ([Architecture::switches]), which no shared body
    /// runs yet.
    pub switches: Vec<Switch>,
}

impl Survey {
    /// Whether the catalogue alone serves the checkpoint: the dense host takes it and nothing
    /// in it is unmodelled, own, unhosted or a missing capability.
    #[must_use]
    pub fn catalogue_only(&self) -> bool {
        self.refusal.is_none()
            && self.unmodelled.is_empty()
            && self.switches.is_empty()
            && self.own.is_empty()
            && self.unhosted.is_empty()
            && self.capabilities.is_empty()
    }
}

/// How the shared path meets `architecture`, with `unmodelled` the declarations its parse
/// dropped.
#[must_use]
pub fn survey(architecture: &Architecture, unmodelled: Vec<String>) -> Survey {
    let registry = Dense::catalogue();
    let mut survey = Survey {
        unmodelled,
        switches: architecture.switches(),
        ..Survey::default()
    };
    architecture.walk_roles(|_, role, part| match role {
        StepRole::Attention | StepRole::Ffn if registry.has(part) => {
            survey.catalogue.insert(part);
        }
        StepRole::Attention | StepRole::Ffn => {
            survey.own.insert(part);
        }
        StepRole::Compression
        | StepRole::Indexer
        | StepRole::Residual
        | StepRole::Side
        | StepRole::Message => {
            survey.unhosted.insert((role, part));
        }
    });
    let own: Vec<Part> = survey.own.iter().copied().collect();
    survey.refusal = sparkflash_ops::parts::dense_device_refusal_hosting(architecture, &own);
    if architecture.mtp.is_some() {
        survey
            .capabilities
            .push("a draft head (the walk's drafter is absent)");
    }
    if architecture.vision.is_some() {
        survey
            .capabilities
            .push("an image tower (the walk takes no images)");
    }
    survey
}

/// The crate's name for `family`: `sparkflash-model-` and the family, hyphenated.
#[must_use]
pub fn crate_name(family: &str) -> String {
    format!(
        "sparkflash-model-{}",
        family.replace('_', "-").to_ascii_lowercase()
    )
}

/// `family` as a type-name prefix: `hy_v4_flash` is `HyV4Flash`.
fn camel(name: &str) -> String {
    name.split(['_', '-'])
        .filter(|word| !word.is_empty())
        .map(|word| {
            let mut letters = word.chars();
            letters.next().map_or_else(String::new, |first| {
                first.to_ascii_uppercase().to_string() + letters.as_str()
            })
        })
        .collect()
}

/// The edits outside the crate `family` still makes, as the review's recount lists them.
#[must_use]
pub fn outside_edits(family: &str) -> Vec<String> {
    let krate = crate_name(family);
    let library = krate.replace('-', "_");
    vec![
        format!(
            "Cargo.toml: \"crates/{krate}\" in [workspace] members (cargo refuses a crate under \
             the workspace that is not one)"
        ),
        "the two CI files name the crate (crates/sparkflash-model/tests/arms.rs, \
         every_model_crate_is_named_in_both_ci_files)"
            .to_owned(),
        "to serve under a registry id: a MODELS descriptor (crates/sparkflash-model/src/\
         descriptor.rs) and an ARMS entry (crates/sparkflash-model/src/arms.rs); the tests there \
         fail on a descriptor with no arm, an arm with no ledger, or a selector with no descriptor"
            .to_owned(),
        format!(
            "crates/sparkflash-cli: {krate} as a dependency, and one DENSE_FAMILIES row in \
             src/serve/dynamic.rs, (\"{family}\", {library}::parts), so the dynamic route opens \
             the dense arm with the family's parts"
        ),
    ]
}

/// What [emit] says when the catalogue alone serves the checkpoint.
#[must_use]
pub fn serve_command(id: &str) -> String {
    format!(
        "no crate is needed: the shared dense arm serves it from the catalogue.\n  \
         sparkflash serve --checkpoint <the {id} snapshot directory>\n"
    )
}

/// Write the family's crate for `architecture` into `dir`, with `config` its `config.json` as the
/// checkpoint ships it. Returns what was written and what is left, for the operator.
///
/// # Errors
/// A `dir` that exists, a catalogue-only checkpoint (nothing to write; the message is the serve
/// command), or a write that fails.
pub fn emit(
    id: &str,
    config: &str,
    architecture: &Architecture,
    survey: &Survey,
    dir: &Path,
) -> Result<String, String> {
    if survey.catalogue_only() {
        return Err(serve_command(id));
    }
    if dir.exists() {
        return Err(format!(
            "{} exists; --emit writes a new directory and touches nothing else",
            dir.display()
        ));
    }
    let family = architecture.family.as_str();
    let krate = crate_name(family);
    let files = [
        ("Cargo.toml".to_owned(), cargo_toml(&krate)),
        (format!("config/{family}.json"), config.to_owned()),
        ("src/lib.rs".to_owned(), lib_rs(id, family, survey)),
        ("tests/conformance.rs".to_owned(), conformance_rs(family)),
    ];
    for (path, text) in &files {
        let path = dir.join(path);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|error| format!("{}: {error}", parent.display()))?;
        }
        std::fs::write(&path, text).map_err(|error| format!("{}: {error}", path.display()))?;
    }
    let mut report = format!("wrote {krate} into {}:\n", dir.display());
    for (path, _) in &files {
        let _ = writeln!(report, "  {path}");
    }
    let _ = writeln!(report, "left to write in the crate:");
    for part in &survey.own {
        let _ = writeln!(
            report,
            "  the {} part's bodies ({}Part)",
            part.name(),
            camel(part.name())
        );
    }
    for (role, part) in &survey.unhosted {
        let _ = writeln!(
            report,
            "  {} ({role:?}): the shared walk does not run it yet",
            part.name()
        );
    }
    for capability in &survey.capabilities {
        let _ = writeln!(report, "  {capability}");
    }
    for switch in &survey.switches {
        let _ = writeln!(
            report,
            "  {switch} ({:?}): typed, the shared walk does not run it yet",
            switch.role()
        );
    }
    for declaration in &survey.unmodelled {
        let _ = writeln!(report, "  describe first: {declaration}");
    }
    if let Some(refusal) = &survey.refusal {
        let _ = writeln!(
            report,
            "  with its parts registered, the dense host still refuses it: {refusal}"
        );
    }
    let _ = writeln!(report, "the edits outside the crate:");
    for edit in outside_edits(family) {
        let _ = writeln!(report, "  {edit}");
    }
    Ok(report)
}

fn cargo_toml(krate: &str) -> String {
    let header = HEADER.replace("// ", "# ").replace("//", "#");
    format!(
        "{header}
[package]
name = \"{krate}\"
version.workspace = true
edition.workspace = true
license.workspace = true
authors.workspace = true
rust-version.workspace = true

[dependencies]
sparkflash-model = {{ workspace = true }}
sparkflash-model-dense = {{ workspace = true }}
sparkflash-runtime = {{ workspace = true }}
sparkflash-weights = {{ workspace = true }}
"
    )
}

fn lib_rs(id: &str, family: &str, survey: &Survey) -> String {
    let mut out = String::from(HEADER);
    let _ = writeln!(
        out,
        "\n//! The {id} family (`{family}`), as `sparkflash onboard --emit` wrote it."
    );
    let _ = writeln!(out, "//!");
    let _ = writeln!(
        out,
        "//! The shared dense host runs the catalogue parts: {}.",
        names(survey.catalogue.iter())
    );
    if !survey.own.is_empty() {
        let _ = writeln!(
            out,
            "//! The family writes the bodies of: {}. Each stub below refuses by name until then.",
            names(survey.own.iter())
        );
    }
    for (role, part) in &survey.unhosted {
        let _ = writeln!(
            out,
            "//! Not run by the shared walk yet: {} ({role:?} step).",
            part.name()
        );
    }
    for capability in &survey.capabilities {
        let _ = writeln!(out, "//! Not hosted by the shared walk yet: {capability}.");
    }
    for switch in &survey.switches {
        let _ = writeln!(
            out,
            "//! Typed, not run by the shared walk yet: {switch} ({:?} step).",
            switch.role()
        );
    }
    for declaration in &survey.unmodelled {
        let _ = writeln!(
            out,
            "//! Not described by the shared parser: {declaration}."
        );
    }
    if let Some(refusal) = &survey.refusal {
        let _ = writeln!(
            out,
            "//! With these parts registered, the dense host still refuses the checkpoint: {refusal}."
        );
    }
    let _ = write!(
        out,
        "
use sparkflash_model::backend::BackendError;
use sparkflash_model::{{Architecture, Part, TpPlan}};
use sparkflash_model_dense::{{Dense, DensePart, DenseParts, OpenCx, PassCx}};
use sparkflash_runtime::{{PartExec, RankSplit, Region}};

/// The checkpoint's `config.json`, as its repository ships it.
pub const CONFIG: &str = include_str!(\"../config/{family}.json\");

/// The checkpoint's shape, read from [CONFIG] with no code of the family's.
///
/// # Errors
/// As [Architecture::from_json].
pub fn architecture() -> Result<Architecture, sparkflash_model::ConfigError> {{
    Architecture::from_json(CONFIG)
}}

/// The executors the family's arm opens: the catalogue, with the family's own parts.
#[must_use]
pub fn parts() -> DenseParts {{
    #[allow(unused_mut)]
    let mut parts = Dense::catalogue();
"
    );
    for part in &survey.own {
        let _ = writeln!(
            out,
            "    parts\n        .register(Part::{part:?}, {}Part::open)\n        .expect(\"the catalogue has no {} executor\");",
            camel(part.name()),
            part.name()
        );
    }
    let _ = writeln!(out, "    parts\n}}");
    for part in &survey.own {
        let name = camel(part.name());
        let _ = write!(
            out,
            "
/// The family's {part_name} part. Its bodies refuse by name until they are written.
pub struct {name}Part;

impl {name}Part {{
    /// Opens the part for one layer: load its tensors through `open` here.
    ///
    /// # Errors
    /// None yet.
    pub fn open(_open: &mut OpenCx<'_>) -> Result<DensePart, String> {{
        Ok(Box::new(Self))
    }}

    fn unwritten(&self) -> Result<(), BackendError> {{
        Err(BackendError::new(
            \"{family}\",
            \"the {part_name} part has no body yet: write {name}Part's rows and token\",
        ))
    }}
}}

impl PartExec<sparkflash_model_dense::Dense> for {name}Part {{
    fn state(&self) -> &[Region] {{
        &[]
    }}

    fn split(&self, _plan: &TpPlan) -> RankSplit {{
        RankSplit::Replicated
    }}

    fn rows(&self, _pass: &mut PassCx<'_>) -> Result<(), BackendError> {{
        self.unwritten()
    }}

    fn token(&self, _pass: &mut PassCx<'_>) -> Result<(), BackendError> {{
        self.unwritten()
    }}
}}
",
            part_name = part.name()
        );
    }
    out
}

fn conformance_rs(family: &str) -> String {
    let fixture = format!("{}Fixture", camel(family));
    let krate = crate_name(family).replace('-', "_");
    format!(
        "{HEADER}
//! The generated checks over the family's arm. The fixture refuses until it writes a tiny
//! checkpoint from the config, as `crates/sparkflash-model-dense/tests/onboard_example.rs`
//! (`write_fixture` and `open`) does, and the profile states what the arm declares.

use sparkflash_model::backend::BackendError;
use sparkflash_model_dense::DenseWalk;
use sparkflash_runtime::testing::ArmProfile;
use sparkflash_runtime::{{ArmFixture, Fixture, Limits, Region, StateLayout}};

/// The fixture the generated checks boot.
struct {fixture};

impl ArmFixture for {fixture} {{
    type Walk = DenseWalk;

    fn fixture(_world: usize) -> Result<Fixture<DenseWalk>, BackendError> {{
        let _parts = {krate}::parts();
        Err(BackendError::new(
            \"fixture\",
            \"write the tiny checkpoint and open the arm with {krate}::parts()\",
        ))
    }}

    fn tick_engine() -> bool {{
        true
    }}
}}

/// What the arm states: fill in its regions, limits and prompt chunks.
fn stated() -> ArmProfile {{
    let limits = Limits::new(256, 32_768);
    ArmProfile::new(
        \"dense\",
        StateLayout::new(vec![Region::Positional {{
            name: \"kv\",
            bytes_per_row: 4,
        }}]),
        limits,
    )
    .with_code(\"dense_device\")
    .prompt(limits, 600, &[256, 256, 88])
}}

sparkflash_runtime::conformance!({fixture}, profile: stated());
"
    )
}

fn names<'a>(parts: impl Iterator<Item = &'a Part>) -> String {
    let names: Vec<&str> = parts.map(|part| part.name()).collect();
    if names.is_empty() {
        "none".to_owned()
    } else {
        names.join(", ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LLAMA: &str = r#"{
        "model_type": "made_up_llama", "architectures": ["MadeUpForCausalLM"],
        "hidden_size": 256, "intermediate_size": 512, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 64,
        "vocab_size": 512, "max_position_embeddings": 512, "torch_dtype": "bfloat16"
    }"#;

    fn surveyed(text: &str) -> (Architecture, Survey) {
        let architecture = Architecture::from_json(text).expect("the config parses");
        let root = serde_json::from_str(text).expect("json");
        let survey = survey(&architecture, Architecture::unmodelled_features(&root));
        (architecture, survey)
    }

    /// A Llama-shaped checkpoint is the catalogue's: nothing is written, and the answer is the
    /// serve command.
    #[test]
    fn a_catalogue_checkpoint_needs_no_crate() {
        let (architecture, survey) = surveyed(LLAMA);
        assert!(survey.catalogue_only(), "{survey:?}");
        let dir = std::env::temp_dir().join("sparkflash-onboard-never-written");
        let answer = emit("made/up", LLAMA, &architecture, &survey, &dir).expect_err("no crate");
        assert!(answer.contains("no crate is needed"), "{answer}");
        assert!(!dir.exists());
    }

    /// An MLA checkpoint gets a stub for its token mixer and refuses an existing directory.
    #[test]
    fn an_mla_checkpoint_gets_a_stub_and_its_crate() {
        let text = LLAMA
            .replace("\"head_dim\": 64,", "\"kv_lora_rank\": 64, \"qk_nope_head_dim\": 32, \"qk_rope_head_dim\": 32, \"v_head_dim\": 64,");
        let (architecture, survey) = surveyed(&text);
        assert!(survey.own.contains(&Part::MlaAttention), "{survey:?}");
        assert_eq!(
            survey.refusal, None,
            "the family's MLA part settles the dense gate"
        );
        let dir = std::env::temp_dir().join(format!("sparkflash-onboard-{}", std::process::id()));
        let report = emit("made/up", &text, &architecture, &survey, &dir).expect("written");
        assert!(report.contains("MlaAttentionPart"), "{report}");
        let lib = std::fs::read_to_string(dir.join("src/lib.rs")).expect("lib.rs");
        assert!(
            lib.contains("Part::MlaAttention, MlaAttentionPart::open"),
            "{lib}"
        );
        assert!(dir.join("config/made_up_llama.json").exists());
        let again = emit("made/up", &text, &architecture, &survey, &dir).expect_err("exists");
        assert!(again.contains("exists"), "{again}");
        std::fs::remove_dir_all(&dir).expect("cleanup");
    }

    /// A checkpoint shaped like tencent/Hy4-flash-FP8 (a small config of the same structure:
    /// latent attention with a sparse indexer, a sliding layer, four hyper-connection streams,
    /// hashed Engram tables a head, routed experts, and the release's switches): its MLA is the
    /// family's part, which the dense host now takes, its switches are typed rather than
    /// unmodelled, and what is left is the three roles the shared walk does not run.
    #[test]
    fn a_hyper_connected_engram_checkpoint_names_the_roles_left() {
        let text = r#"{
            "model_type": "acme_hyper", "hidden_size": 256, "intermediate_size": 512,
            "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 4,
            "head_dim": 64, "kv_lora_rank": 64, "q_lora_rank": 128, "qk_nope_head_dim": 32,
            "qk_rope_head_dim": 32, "v_head_dim": 64,
            "layer_types": ["deepseek_sparse_attention", "sliding_attention"],
            "sliding_window": 128, "indexer_types": ["full", "full"], "index_n_heads": 4,
            "index_head_dim": 32, "index_topk": 64, "mlp_layer_types": ["dense", "sparse"],
            "first_k_dense_replace": 1, "n_routed_experts": 8, "num_experts_per_tok": 2,
            "moe_intermediate_size": 64, "n_shared_experts": 1, "hc_mult": 4,
            "engram_layer_ids": [1], "engram_vocab_size": [1000, 1000],
            "engram_layer_head_vocab_sizes": {"1": [[11, 13], [17, 19]]},
            "engram_max_ngram_size": 3, "engram_n_embed_per_ngram": 64,
            "engram_n_head_per_ngram": 2, "engram_pad_id": 0, "vocab_size": 512,
            "max_position_embeddings": 512, "rms_norm_eps": 1e-6, "hidden_act": "silu",
            "torch_dtype": "bfloat16", "swa_num_attention_heads": 8, "sandwich_norm": true,
            "learnable_sink": true, "gated_mla": true, "gating_type": "elementwise",
            "qk_norm": true, "enable_ihc": true, "hc_magnitude": 2.0,
            "use_compressed_indexer": true, "compressed_indexer_num_attention_heads": 2,
            "compressed_indexer_head_dim": 32, "compressed_indexer_compress_ratio": 4,
            "compressed_indexer_key_pooling": "gated", "compressed_indexer_candidate_topk": 64,
            "compressed_indexer_group_topk": 16, "compressed_indexer_skip_fine_select": true
        }"#;
        let (_, survey) = surveyed(text);
        assert_eq!(survey.unmodelled, Vec::<String>::new(), "{survey:?}");
        assert_eq!(
            survey.switches,
            vec![
                Switch::Sinks,
                Switch::MlaOutputGate,
                Switch::MlaQkNorm,
                Switch::SlidingHeads,
                Switch::PostNorm,
                Switch::Ihc,
                Switch::HcMagnitude,
                Switch::CompressedIndexer,
                Switch::EngramHeadTables,
            ],
            "{survey:?}"
        );
        assert!(!survey.catalogue_only());
        assert_eq!(
            survey.own,
            BTreeSet::from([Part::MlaAttention]),
            "{survey:?}"
        );
        let left: BTreeSet<StepRole> = survey.unhosted.iter().map(|(role, _)| *role).collect();
        assert_eq!(
            left,
            BTreeSet::from([StepRole::Indexer, StepRole::Residual, StepRole::Side]),
            "{survey:?}"
        );
        let refusal = survey
            .refusal
            .as_deref()
            .expect("the roles left are refused");
        for part in ["sparse_indexer", "hyper_connection", "engram"] {
            assert!(refusal.contains(part), "{refusal}");
        }
        assert!(!refusal.contains("mla_attention"), "{refusal}");
        let (architecture, _) = surveyed(text);
        let dir =
            std::env::temp_dir().join(format!("sparkflash-onboard-hyper-{}", std::process::id()));
        let report = emit("acme/hyper", text, &architecture, &survey, &dir).expect("written");
        let lib = std::fs::read_to_string(dir.join("src/lib.rs")).expect("lib.rs");
        std::fs::remove_dir_all(&dir).expect("cleanup");
        assert!(!report.contains("describe first"), "{report}");
        for switch in &survey.switches {
            let typed = format!("{switch} ({:?}): typed", switch.role());
            assert!(report.contains(&typed), "{typed} is not listed: {report}");
            assert!(
                lib.contains(&switch.to_string()),
                "{switch} is not listed: {lib}"
            );
        }
    }
}
