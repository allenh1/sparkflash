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

//! One part of a layer walk, as a plug-in.
//!
//! A walk over [`sparkflash_model::compose`]'s steps runs each step's part. The parts the
//! catalogue has are written once, in the arm that walks the steps; a family with one part of
//! its own writes that part against [`PartExec`] and registers its constructor under the part's
//! kind ([`PartRegistry`]). It writes no walk, no sequence and nothing about serving.
//!
//! The trait lives here, not in the arm that hosts the parts, for two reasons. What it must
//! mention is the runtime's and the model crate's: the [`Region`]s a part adds to the layout the
//! runtime rewinds, and the [`TpPlan`] a group divides it by. And the walk that will run parts
//! for every family (`PlanWalk`, docs/engine/04-onboarding.md mechanism 2) is the runtime's. What
//! the trait cannot mention is a device: this crate has none. So the pass and the open context
//! are the host's types ([`PartHost`]), and a part is written for the host it plugs into.

use std::collections::BTreeMap;

use sparkflash_model::TpPlan;
use sparkflash_model::backend::BackendError;
use sparkflash_model::parts::Part as PartKind;

use crate::layout::Region;
use crate::walk::PassKind;

/// The arm a part plugs into: what its walk hands a part for one pass, what its load path hands
/// a part's constructor, and how it holds the part it built.
pub trait PartHost: 'static {
    /// One pass over one part: the rows in, the rows out, each segment's share of the part's
    /// state, the scratch and the device.
    type Pass<'a>;

    /// One part being opened: the checkpoint, the rank's share and the uploader.
    type Open<'a>;

    /// A built part as the host keeps it: a boxed [`PartExec`] with the thread bounds the host's
    /// engine asks for. The trait itself carries none, as [`crate::Walk`] carries none.
    type Handle;
}

/// How a part's output rows divide over a tensor-parallel group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RankSplit {
    /// Every rank holds the whole part and leaves the same rows: nothing crosses the group.
    Replicated,
    /// Each rank holds its share of the part (heads, a width) and leaves a partial of every
    /// output row. The host sums the partials in rank order before the residual takes them, so
    /// the residual is the same bits on every rank. One rank's sum is its own rows.
    Summed,
}

/// One part of a layer, executed.
///
/// Every method is required. A part whose token body is its prompt body says so in one visible
/// line, by calling [`Self::rows`] from [`Self::token`]; a part with no state returns an empty
/// slice, and a part every rank holds whole says [`RankSplit::Replicated`].
///
/// A part holds its own weights: its constructor uploaded this rank's share of them through the
/// host's open context, typed by the host's uploader.
pub trait PartExec<H: PartHost> {
    /// What the part adds to a sequence's layout, in the order its bodies index them. A
    /// positional region is truncated on a rewind and a recurrent one is checkpointed; the part
    /// writes neither.
    fn state(&self) -> &[Region];

    /// How the part's output divides over the ranks of `plan`.
    fn split(&self, plan: &TpPlan) -> RankSplit;

    /// The prompt body: a chunk's rows.
    ///
    /// # Errors
    /// A device failure, or a pass the part cannot run.
    fn rows(&self, pass: &mut H::Pass<'_>) -> Result<(), BackendError>;

    /// The token body: a decode step's rows, or a verify pass's.
    ///
    /// # Errors
    /// A device failure, or a pass the part cannot run.
    fn token(&self, pass: &mut H::Pass<'_>) -> Result<(), BackendError>;
}

/// Which of a part's two bodies a pass runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartBody {
    /// [`PartExec::rows`].
    Rows,
    /// [`PartExec::token`].
    Token,
}

impl PartBody {
    /// The body a pass of `kind` runs: a prompt's chunk (alone or paired with others) is the
    /// prompt body, a decode step and a verify pass are the token body.
    #[must_use]
    pub const fn of(kind: PassKind) -> Self {
        match kind {
            PassKind::Bulk { .. } | PassKind::Paired { .. } => Self::Rows,
            PassKind::TokenRows | PassKind::Verify { .. } => Self::Token,
        }
    }

    /// Runs this body of `part` over `pass`.
    ///
    /// # Errors
    /// The body's.
    pub fn run<H: PartHost, P: PartExec<H> + ?Sized>(
        self,
        part: &P,
        pass: &mut H::Pass<'_>,
    ) -> Result<(), BackendError> {
        match self {
            Self::Rows => part.rows(pass),
            Self::Token => part.token(pass),
        }
    }
}

/// A part's constructor: it reads what it needs from the host's open context and returns the
/// part as the host holds it.
pub type PartMaker<H> = Box<
    dyn for<'a> Fn(&mut <H as PartHost>::Open<'a>) -> Result<<H as PartHost>::Handle, String>
        + Send
        + Sync,
>;

/// The executors a host builds its walk from: one constructor a part kind.
///
/// A host states its catalogue parts in one registry, and a family adds its own part with one
/// [`Self::register`] line. A step whose part has no constructor is refused by name
/// ([`Self::require`], [`Self::build`]); it is never skipped.
pub struct PartRegistry<H: PartHost> {
    makers: BTreeMap<PartKind, PartMaker<H>>,
}

impl<H: PartHost> Default for PartRegistry<H> {
    fn default() -> Self {
        Self::new()
    }
}

impl<H: PartHost> PartRegistry<H> {
    /// A registry with no executor.
    #[must_use]
    pub fn new() -> Self {
        Self {
            makers: BTreeMap::new(),
        }
    }

    /// Adds `make` as the constructor of `kind`.
    ///
    /// # Errors
    /// A kind that already has one: say [`Self::replace`] to take a part over.
    pub fn register(
        &mut self,
        kind: PartKind,
        make: impl for<'a> Fn(&mut H::Open<'a>) -> Result<H::Handle, String> + Send + Sync + 'static,
    ) -> Result<(), String> {
        if self.makers.contains_key(&kind) {
            return Err(format!(
                "part {} already has an executor; replace it to take the part over",
                kind.name()
            ));
        }
        self.makers.insert(kind, Box::new(make));
        Ok(())
    }

    /// Makes `make` the constructor of `kind`, in place of the one it has.
    ///
    /// # Errors
    /// A kind with no executor to replace: say [`Self::register`] to add one.
    pub fn replace(
        &mut self,
        kind: PartKind,
        make: impl for<'a> Fn(&mut H::Open<'a>) -> Result<H::Handle, String> + Send + Sync + 'static,
    ) -> Result<(), String> {
        if !self.makers.contains_key(&kind) {
            return Err(format!(
                "part {} has no executor to replace; register one",
                kind.name()
            ));
        }
        self.makers.insert(kind, Box::new(make));
        Ok(())
    }

    /// Takes `kind`'s constructor out, so a step that names the part is refused.
    ///
    /// # Errors
    /// A kind with no executor.
    pub fn remove(&mut self, kind: PartKind) -> Result<(), String> {
        match self.makers.remove(&kind) {
            Some(_) => Ok(()),
            None => Err(format!("part {} has no executor to remove", kind.name())),
        }
    }

    /// Whether `kind` has a constructor.
    #[must_use]
    pub fn has(&self, kind: PartKind) -> bool {
        self.makers.contains_key(&kind)
    }

    /// The kinds with a constructor, in the catalogue's order.
    pub fn kinds(&self) -> impl Iterator<Item = PartKind> + '_ {
        self.makers.keys().copied()
    }

    /// Refuses the first of `kinds` with no constructor, by name, before anything is built.
    ///
    /// # Errors
    /// A kind with no executor.
    pub fn require(&self, kinds: impl IntoIterator<Item = PartKind>) -> Result<(), String> {
        match kinds.into_iter().find(|kind| !self.has(*kind)) {
            None => Ok(()),
            Some(kind) => Err(Self::missing(kind)),
        }
    }

    /// Builds `kind`'s part from `open`.
    ///
    /// # Errors
    /// A kind with no executor, or the constructor's own refusal.
    pub fn build(&self, kind: PartKind, open: &mut H::Open<'_>) -> Result<H::Handle, String> {
        match self.makers.get(&kind) {
            Some(make) => make(open),
            None => Err(Self::missing(kind)),
        }
    }

    fn missing(kind: PartKind) -> String {
        format!(
            "no executor is registered for part {}; the walk runs a part only through its \
             registered executor",
            kind.name()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A host with no device: a pass is the rows a part appends to, and opening hands a part
    /// its layer.
    struct Rows;

    impl PartHost for Rows {
        type Pass<'a> = Vec<&'static str>;
        type Open<'a> = usize;
        type Handle = Box<dyn PartExec<Self>>;
    }

    /// A part with one positional region, a share a rank and two bodies.
    struct Mixer {
        state: [Region; 1],
    }

    impl PartExec<Rows> for Mixer {
        fn state(&self) -> &[Region] {
            &self.state
        }

        fn split(&self, _plan: &TpPlan) -> RankSplit {
            RankSplit::Summed
        }

        fn rows(&self, pass: &mut Vec<&'static str>) -> Result<(), BackendError> {
            pass.push("mixer rows");
            Ok(())
        }

        fn token(&self, pass: &mut Vec<&'static str>) -> Result<(), BackendError> {
            pass.push("mixer token");
            Ok(())
        }
    }

    /// A part with no state, held whole, whose token body is its prompt body.
    struct Mlp;

    impl PartExec<Rows> for Mlp {
        fn state(&self) -> &[Region] {
            &[]
        }

        fn split(&self, _plan: &TpPlan) -> RankSplit {
            RankSplit::Replicated
        }

        fn rows(&self, pass: &mut Vec<&'static str>) -> Result<(), BackendError> {
            pass.push("mlp");
            Ok(())
        }

        // The token body is the prompt body.
        fn token(&self, pass: &mut Vec<&'static str>) -> Result<(), BackendError> {
            self.rows(pass)
        }
    }

    fn registry() -> PartRegistry<Rows> {
        let mut parts = PartRegistry::new();
        parts
            .register(PartKind::GqaAttention, |layer: &mut usize| {
                Ok(Box::new(Mixer {
                    state: [Region::Positional {
                        name: "kv",
                        bytes_per_row: 8 * (*layer + 1),
                    }],
                }) as Box<dyn PartExec<Rows>>)
            })
            .expect("the first mixer");
        parts
            .register(PartKind::DenseMlp, |_: &mut usize| {
                Ok(Box::new(Mlp) as Box<dyn PartExec<Rows>>)
            })
            .expect("the first MLP");
        parts
    }

    #[test]
    fn a_pass_kind_picks_the_body() {
        assert_eq!(PartBody::of(PassKind::Bulk { rows: 9 }), PartBody::Rows);
        assert_eq!(PartBody::of(PassKind::Paired { chunks: 2 }), PartBody::Rows);
        assert_eq!(PartBody::of(PassKind::TokenRows), PartBody::Token);
        assert_eq!(PartBody::of(PassKind::Verify { rows: 4 }), PartBody::Token);
    }

    #[test]
    fn a_registered_part_is_built_and_runs_the_body_asked() {
        let parts = registry();
        let plan = TpPlan::new(4, 1).expect("a rank of four");
        let mut layer = 1usize;
        let mixer = parts
            .build(PartKind::GqaAttention, &mut layer)
            .expect("the mixer builds");
        let mlp = parts
            .build(PartKind::DenseMlp, &mut layer)
            .expect("the MLP builds");
        assert_eq!(
            mixer.state(),
            [Region::Positional {
                name: "kv",
                bytes_per_row: 16
            }]
        );
        assert!(mlp.state().is_empty());
        assert_eq!(mixer.split(&plan), RankSplit::Summed);
        assert_eq!(mlp.split(&plan), RankSplit::Replicated);
        let mut pass = Vec::new();
        for body in [PartBody::Rows, PartBody::Token] {
            body.run::<Rows, _>(&*mixer, &mut pass)
                .expect("the mixer runs");
            body.run::<Rows, _>(&*mlp, &mut pass).expect("the MLP runs");
        }
        assert_eq!(pass, ["mixer rows", "mlp", "mixer token", "mlp"]);
    }

    #[test]
    fn a_part_with_no_executor_is_refused_by_name() {
        let parts = registry();
        let mut layer = 0usize;
        let refusal = parts
            .build(PartKind::KdaScan, &mut layer)
            .err()
            .expect("a part with no executor does not build");
        assert!(refusal.contains(PartKind::KdaScan.name()), "{refusal}");
        let refusal = parts
            .require([PartKind::GqaAttention, PartKind::Moe, PartKind::DenseMlp])
            .expect_err("a step's part with no executor is refused");
        assert!(refusal.contains(PartKind::Moe.name()), "{refusal}");
        assert!(!refusal.contains(PartKind::DenseMlp.name()), "{refusal}");
        parts
            .require([PartKind::GqaAttention, PartKind::DenseMlp])
            .expect("every part named has an executor");
    }

    #[test]
    fn a_second_registration_is_refused_and_a_replacement_is_said() {
        let mut parts = registry();
        let again = parts
            .register(PartKind::DenseMlp, |_: &mut usize| {
                Ok(Box::new(Mlp) as Box<dyn PartExec<Rows>>)
            })
            .expect_err("a kind takes one executor");
        assert!(again.contains(PartKind::DenseMlp.name()), "{again}");
        let absent = parts
            .replace(PartKind::Moe, |_: &mut usize| {
                Ok(Box::new(Mlp) as Box<dyn PartExec<Rows>>)
            })
            .expect_err("nothing to replace");
        assert!(absent.contains(PartKind::Moe.name()), "{absent}");
        // The MLP's kind now builds the mixer: the replacement is the executor.
        parts
            .replace(PartKind::DenseMlp, |_: &mut usize| {
                Ok(Box::new(Mixer {
                    state: [Region::Recurrent {
                        name: "state",
                        bytes: 64,
                    }],
                }) as Box<dyn PartExec<Rows>>)
            })
            .expect("the replacement");
        let mut layer = 0usize;
        let part = parts
            .build(PartKind::DenseMlp, &mut layer)
            .expect("the replaced part builds");
        assert_eq!(part.state().len(), 1);
        parts.remove(PartKind::DenseMlp).expect("it had one");
        assert!(!parts.has(PartKind::DenseMlp));
        assert!(parts.remove(PartKind::DenseMlp).is_err());
        assert_eq!(parts.kinds().collect::<Vec<_>>(), [PartKind::GqaAttention]);
    }
}
