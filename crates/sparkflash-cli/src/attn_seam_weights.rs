//! The ratio>0 attention seam's **host-seam** weights, resolved by name.
//!
//! Layers 2..39 carry the compressor and the indexer, whose weights the plan classifies as
//! `PlanClass::AttentionHostSeam` (`resident_weights::is_host_seam_weight`) and therefore excludes
//! from the arena bind. The device arm cannot launch a compressor GEMV against a null pointer, so
//! those nineteen tensors are read from the checkpoint source at open and uploaded once into
//! run-scope device buffers - the same treatment the Engram arena and matrices already get, and the
//! design recorded in `docs/build/s1-result.md` §7.1: no plan-class change, so the arena bind record
//! stays comparable across windows and no B12x staging rule has to be invented for tensors the
//! release stores as bf16/f32.
//!
//! This module is the **name half** of that upload: it turns the plan's own host-seam names into a
//! deterministic, ordered tensor list, so the uploader cannot silently miss a layer or invent one.
//! The list is authoritative because it comes from the plan (not from a hand-written layer table), and
//! the count rule is pinned by a test: 4 `compressor.wkv`, 3 `compressor.wgate`, 4 `indexer.wk` and
//! 8 `indexer.weights_proj` = the 19 the plan classifies today.
//!
//! **Not wired yet**: the upload itself and the stage wiring are the next Part B steps
//! (`docs/build/s1-result.md` §7.1 (b)-(e)); nothing here is measured or counted.
#![allow(dead_code)]

/// One host-seam matrix the ratio>0 attention seam reads on the device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SeamMatrix {
    /// `attn.compressor.wkv`: the compressor's KV latent projection (the ratio's kv source).
    CompressorWkv,
    /// `attn.compressor.wgate`: the compressor's element-wise pooling gate (ratio 2 and 4 only).
    CompressorWgate,
    /// `attn.indexer.wk`: the indexer's key projection (the candidate source).
    IndexerWk,
    /// `attn.indexer.weights_proj`: the indexer's per-query weight projection.
    IndexerWeightsProj,
}

impl SeamMatrix {
    /// Every matrix kind, in the order a report lists them.
    pub const ALL: [Self; 4] = [
        Self::CompressorWkv,
        Self::CompressorWgate,
        Self::IndexerWk,
        Self::IndexerWeightsProj,
    ];

    /// The plan name's leaf suffix this kind matches.
    #[must_use]
    pub const fn suffix(self) -> &'static str {
        match self {
            Self::CompressorWkv => ".compressor.wkv.weight",
            Self::CompressorWgate => ".compressor.wgate.weight",
            Self::IndexerWk => ".indexer.wk.weight",
            Self::IndexerWeightsProj => ".indexer.weights_proj.weight",
        }
    }

    /// A short label for refusals and records.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::CompressorWkv => "compressor.wkv",
            Self::CompressorWgate => "compressor.wgate",
            Self::IndexerWk => "indexer.wk",
            Self::IndexerWeightsProj => "indexer.weights_proj",
        }
    }
}

/// One tensor: the layer that owns it and which seam matrix it is.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[allow(missing_docs)] // field names are the whole contract here, by the owner's cleanup
pub struct SeamTensor {
    pub layer: usize,
    pub matrix: SeamMatrix,
    pub name: String,
}

/// Parses one plan name as a host-seam tensor.
///
/// Accepts exactly `layers.{L}.attn.{suffix}` for the four suffixes; every other name (the MTP lanes,
/// the scales, the optional tensors) is `None`, so a caller can filter a whole plan without a second
/// classification rule drifting from `is_host_seam_weight`.
#[must_use]
pub fn parse(name: &str) -> Option<SeamTensor> {
    let rest = name.strip_prefix("layers.")?;
    let (layer, tail) = rest.split_once('.')?;
    let layer: usize = layer.parse().ok()?;
    if !tail.starts_with("attn.") {
        return None;
    }
    let matrix = SeamMatrix::ALL
        .into_iter()
        .find(|matrix| tail.ends_with(matrix.suffix()))?;
    Some(SeamTensor {
        layer,
        matrix,
        name: name.to_owned(),
    })
}

/// The resolved list for one plan: every host-seam tensor the device seam needs, ordered by layer
/// then matrix, with a refusal on a duplicate name (a plan that lists one name twice would upload
/// one matrix into two slots).
///
/// # Errors
/// A duplicate `(layer, matrix)` pair.
pub fn from_plan_names<'a>(
    names: impl IntoIterator<Item = &'a str>,
) -> Result<Vec<SeamTensor>, String> {
    let mut tensors: Vec<SeamTensor> = names.into_iter().filter_map(parse).collect();
    tensors.sort();
    for pair in tensors.windows(2) {
        if pair[0].layer == pair[1].layer && pair[0].matrix == pair[1].matrix {
            return Err(format!(
                "attention seam weights: {} is listed twice for layer {}",
                pair[0].matrix.label(),
                pair[0].layer
            ));
        }
    }
    Ok(tensors)
}

/// How many tensors of each kind the list holds, in [`SeamMatrix::ALL`] order.
#[must_use]
pub fn counts(tensors: &[SeamTensor]) -> [(SeamMatrix, usize); 4] {
    core::array::from_fn(|index| {
        let matrix = SeamMatrix::ALL[index];
        (
            matrix,
            tensors
                .iter()
                .filter(|tensor| tensor.matrix == matrix)
                .count(),
        )
    })
}

/// Checks a resolved list against the released plan's shape: nineteen tensors, and every `wgate` layer
/// also carrying `wkv` (the gate pools the projection it weights). The indexer's `weights_proj` owners
/// are *not* a subset of its `wk` owners in the released plan (eight versus four layers), so that
/// relation is deliberately not asserted.
///
/// # Errors
/// A count that is not the released 19, or a matrix whose owners are not a subset of the compressor
/// or indexer owners' layers - both are facts about the plan, not about a layer table, so a mismatch
/// names the two kind counts.
pub fn check_released_shape(tensors: &[SeamTensor]) -> Result<(), String> {
    const RELEASED: usize = 19;
    if tensors.len() != RELEASED {
        return Err(format!(
            "attention seam weights: the plan lists {} host-seam tensors, the released plan has {RELEASED}",
            tensors.len()
        ));
    }
    let layers_of = |matrix: SeamMatrix| -> Vec<usize> {
        tensors
            .iter()
            .filter(|tensor| tensor.matrix == matrix)
            .map(|tensor| tensor.layer)
            .collect()
    };
    let wkv = layers_of(SeamMatrix::CompressorWkv);
    let wgate = layers_of(SeamMatrix::CompressorWgate);
    for (owner, base, owner_label, base_label) in [(&wgate, &wkv, "wgate", "wkv")] {
        for layer in owner {
            if !base.contains(layer) {
                return Err(format!(
                    "attention seam weights: layer {layer} has {owner_label} but not {base_label}; the released plan's {owner_label} layers are a subset of its {base_label} layers"
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn released_names() -> Vec<String> {
        let mut names = Vec::new();
        for &layer in &[2usize, 8, 14, 20] {
            names.push(format!("layers.{layer}.attn.compressor.wkv.weight"));
            names.push(format!("layers.{layer}.attn.indexer.wk.weight"));
        }
        for &layer in &[2usize, 8, 14] {
            names.push(format!("layers.{layer}.attn.compressor.wgate.weight"));
        }
        for &layer in &[2usize, 8, 14, 20, 24, 28, 32, 36] {
            names.push(format!("layers.{layer}.attn.indexer.weights_proj.weight"));
        }
        // Names a plan also carries that the seam must not pick up.
        names.push("layers.2.attn.compressor.wkv.scale".to_owned());
        names.push("layers.2.attn.indexer.weights_proj.scale".to_owned());
        names.push("layers.2.attn.wq_b.weight".to_owned());
        names.push("mtp.0.attn.indexer.wk.weight".to_owned());
        names
    }

    #[test]
    fn the_four_kinds_parse_and_everything_else_is_ignored() {
        let tensors = from_plan_names(released_names().iter().map(String::as_str))
            .expect("the released list resolves");
        assert_eq!(tensors.len(), 19);
        assert_eq!(
            counts(&tensors),
            [
                (SeamMatrix::CompressorWkv, 4),
                (SeamMatrix::CompressorWgate, 3),
                (SeamMatrix::IndexerWk, 4),
                (SeamMatrix::IndexerWeightsProj, 8),
            ]
        );
        check_released_shape(&tensors).expect("the released shape");
        // The order is (layer, matrix): layer 2's wkv comes first, its wgate second.
        assert_eq!(tensors[0].layer, 2);
        assert_eq!(tensors[0].matrix, SeamMatrix::CompressorWkv);
        assert_eq!(tensors[1].matrix, SeamMatrix::CompressorWgate);
        assert!(
            tensors
                .iter()
                .all(|tensor| tensor.name.starts_with("layers."))
        );
    }

    #[test]
    fn a_duplicate_pair_is_refused_by_name() {
        let names = [
            "layers.2.attn.indexer.wk.weight",
            "layers.2.attn.indexer.wk.weight",
        ];
        let error = from_plan_names(names).expect_err("duplicate");
        assert!(error.contains("indexer.wk"), "{error}");
        assert!(error.contains('2'), "{error}");
    }

    #[test]
    fn a_missing_wk_drops_the_count_and_is_refused() {
        let mut names = released_names();
        names.retain(|name| name != "layers.2.attn.indexer.wk.weight");
        let tensors = from_plan_names(names.iter().map(String::as_str)).expect("resolves");
        let error = check_released_shape(&tensors).expect_err("not the released shape");
        assert!(error.contains("18 host-seam tensors"), "{error}");
    }

    #[test]
    fn a_missing_owner_is_named_by_its_two_kind_counts() {
        let names: Vec<String> = [
            "layers.2.attn.compressor.wgate.weight",
            "layers.2.attn.indexer.wk.weight",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect();
        let tensors = from_plan_names(names.iter().map(String::as_str)).expect("resolves");
        let error = check_released_shape(&tensors).expect_err("wrong count");
        assert!(error.contains("2 host-seam tensors"), "{error}");
        assert!(error.contains("19"), "{error}");
    }
}
