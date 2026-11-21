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

//! One serve entry for any arm that supplies a [Walk].
//!
//! An arm's backend used to repeat the same glue around the engine: start it alone, start it as
//! rank 0 of a group, mirror rank 0 on a worker, read the lane width, and an enum naming which
//! engine its sessions open on. A backend now states three things ([WalkArm]: its walk, its
//! group wire, and whether it was opened as a group's rank) and the glue is written here once:
//!
//! * [serve]: the tick engine over the arm's walk, in this process alone;
//! * [serve_group]: rank 0 of a tensor-parallel group, every decision broadcast before it runs;
//! * [mirror]: a worker of that group, replaying rank 0 until the group ends;
//! * [lane_limit]: the lanes a group opens when its operator named none.
//!
//! A group is the tick group unless the arm says its group still needs the older fused engine
//! ([GroupEngine::Fused]): the dense arm's lane frames cannot carry a chunk's rows yet, and
//! Inkling's group keeps it while its rank 0 encodes images. Both stay reachable here, unchanged,
//! until that engine is retired.

use std::convert::Infallible;
use std::sync::Arc;

use sparkflash_comm::control::{Tp4ControlHub, Tp4ControlWorker};
use sparkflash_model::backend::{BackendError, ModelSession};

use crate::engine::TickServe;
use crate::fused::{FusedArm, FusedEngine};
use crate::group::{
    GroupLanes, GroupWire, mirror_ticks_until_the_group_ends, mirror_until_the_group_ends,
    watched_hub,
};
use crate::runtime::Runtime;
use crate::walk::Walk;

/// How a backend is asked to serve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Place {
    /// In this process alone ([serve]).
    Alone,
    /// As one rank of a tensor-parallel group ([serve_group], [mirror]).
    Group,
}

/// The engine a group's ranks agree on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupEngine {
    /// The tick group: rank 0 decides the tick, the frames carry its rows and step shape.
    Tick,
    /// The older fused engine over [GroupLanes], for a group whose frames the tick group cannot
    /// carry yet. Every rank must give the same answer.
    Fused,
}

/// What a walk arm's backend states so the runtime can serve it.
///
/// Every item is required. The walk is returned by value because the engine's thread owns it;
/// the backend keeps what the walk shares (its weights, its collective).
pub trait WalkArm {
    /// The walk over this process's rank, for `capacity` lanes.
    ///
    /// `capacity` is the lanes the caller opens and may be wider than one pass holds: the arm
    /// caps its own batch width. [lane_limit] asks with `usize::MAX` and reads the width back.
    ///
    /// # Errors
    /// A configuration the walk cannot honour, or scratch the device refuses.
    fn walk(&self, capacity: usize) -> Result<impl Walk<Seq: Send> + Send + Sync, BackendError>;

    /// What the group's frames carry and what rank 0 refuses before it broadcasts.
    fn wire(&self) -> impl GroupWire;

    /// Says whether this process is a group's rank, by refusing the place it was not opened
    /// for: a group's rank asked to serve alone, or a lone backend asked to serve as a rank. A
    /// backend that is the same object either way refuses neither.
    ///
    /// # Errors
    /// The refusal, in the arm's words.
    fn admit(&self, place: Place) -> Result<(), BackendError>;

    /// The engine this arm's group runs on.
    fn group_engine(&self) -> GroupEngine;
}

/// The engine a served backend's sessions are handles on.
///
/// The walk's type stays inside: a backend holds one of these whatever its walk is, and opens a
/// session on it for each request.
pub struct Serving(Box<dyn Sessions>);

impl Serving {
    /// A session on the engine: one lane for one request.
    ///
    /// # Errors
    /// A poisoned handle list.
    pub fn open(&self) -> Result<Box<dyn ModelSession>, BackendError> {
        self.0.open()
    }
}

impl std::fmt::Debug for Serving {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("Serving").finish_non_exhaustive()
    }
}

/// An engine that hands out sessions.
trait Sessions: Send + Sync {
    fn open(&self) -> Result<Box<dyn ModelSession>, BackendError>;
}

impl<W> Sessions for Arc<TickServe<W>>
where
    W: Walk + Send + Sync,
    W::Seq: Send,
{
    fn open(&self) -> Result<Box<dyn ModelSession>, BackendError> {
        Ok(Box::new(TickServe::open(self)?))
    }
}

impl<A: FusedArm> Sessions for Arc<FusedEngine<A>> {
    fn open(&self) -> Result<Box<dyn ModelSession>, BackendError> {
        Ok(Box::new(FusedEngine::open(self)?))
    }
}

/// Serves `arm` in this process alone: `capacity` lanes on the tick engine over its walk.
///
/// # Errors
/// A backend that is a group's rank ([WalkArm::admit]), a walk the arm cannot build, or as
/// [TickServe::start].
pub fn serve(arm: &impl WalkArm, capacity: usize) -> Result<Serving, BackendError> {
    arm.admit(Place::Alone)?;
    let walk = arm.walk(capacity)?;
    Ok(Serving(Box::new(TickServe::start(walk, capacity)?)))
}

/// Serves `arm` as rank 0 of a group: `capacity` lanes whose every prompt, chunk, step,
/// give-back and release is broadcast through `hub` before it runs here, refused first as the
/// arm's wire refuses it. Each worker replays them ([mirror]) over `capacity` lanes of its own.
///
/// A worker whose socket closes ends this process, and so does a failure here after a frame went
/// out ([watched_hub], [crate::end_group]).
///
/// # Errors
/// A backend that is no group's rank ([WalkArm::admit]), a walk the arm cannot build, a watch
/// that cannot start, or as the engine's own start.
pub fn serve_group(
    arm: &impl WalkArm,
    capacity: usize,
    hub: Tp4ControlHub,
) -> Result<Serving, BackendError> {
    arm.admit(Place::Group)?;
    let walk = arm.walk(capacity)?;
    Ok(match arm.group_engine() {
        GroupEngine::Tick => Serving(Box::new(TickServe::start_group(
            walk,
            capacity,
            hub,
            arm.wire(),
        )?)),
        GroupEngine::Fused => {
            let lanes = GroupLanes::new(Runtime::new(walk, 0)?, watched_hub(hub)?, arm.wire());
            Serving(Box::new(FusedEngine::start(Arc::new(lanes), capacity)?))
        }
    })
}

/// A worker of a group: replays rank 0's frames over `capacity` lanes of this rank's own until
/// the group can no longer serve, then ends the process ([mirror_ticks_until_the_group_ends], or
/// [mirror_until_the_group_ends] on the fused engine).
///
/// # Errors
/// A backend that is no group's rank, or a walk or runtime the arm cannot build, before any
/// frame.
pub fn mirror(
    arm: &impl WalkArm,
    control: &mut Tp4ControlWorker,
    capacity: usize,
) -> Result<Infallible, String> {
    arm.admit(Place::Group).map_err(|error| error.to_string())?;
    let runtime = arm
        .walk(capacity)
        .and_then(|walk| Runtime::new(walk, 0))
        .map_err(|error| error.to_string())?;
    let wire = arm.wire();
    match arm.group_engine() {
        GroupEngine::Tick => mirror_ticks_until_the_group_ends(control, &runtime, &wire, capacity),
        GroupEngine::Fused => mirror_until_the_group_ends(control, &runtime, &wire, capacity),
    }
}

/// The most lanes `arm`'s batch takes: the member width its walk's [crate::Limits] states.
///
/// A group whose operator gave no `--concurrency` opens this many lanes, so the count comes from
/// the arm and not from a constant in the caller.
///
/// # Errors
/// A walk the arm cannot build, or one the runtime refuses.
pub fn lane_limit(arm: &impl WalkArm) -> Result<usize, BackendError> {
    Ok(Runtime::new(arm.walk(usize::MAX)?, 0)?.limits().members)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::group::TextWire;
    use crate::layout::{Limits, Region, StateLayout};
    use crate::testing::{ShapeWalk, pick};
    use sparkflash_model::backend::Sampling;

    /// A backend over the CPU shape walk: its batch holds four members, and `rank` says whether
    /// it was opened as a group's rank.
    struct Shape {
        rank: bool,
    }

    impl WalkArm for Shape {
        fn walk(
            &self,
            capacity: usize,
        ) -> Result<impl Walk<Seq: Send> + Send + Sync, BackendError> {
            Ok(ShapeWalk::new(
                StateLayout::new(vec![Region::Positional {
                    name: "shape",
                    bytes_per_row: 4,
                }]),
                Limits::new(64, 4_096).with_members(capacity.clamp(1, 4)),
                0,
            ))
        }

        fn wire(&self) -> impl GroupWire {
            TextWire::new("shape", "shape", 4_096)
        }

        fn admit(&self, place: Place) -> Result<(), BackendError> {
            match (self.rank, place) {
                (true, Place::Alone) => Err(BackendError::new("shape", "a group's rank")),
                (false, Place::Group) => Err(BackendError::new("shape", "no group's rank")),
                _ => Ok(()),
            }
        }

        fn group_engine(&self) -> GroupEngine {
            GroupEngine::Tick
        }
    }

    #[test]
    fn a_lone_backend_serves_a_request_on_the_tick_engine() {
        let serving = serve(&Shape { rank: false }, 2).expect("the engine starts");
        let mut session = serving.open().expect("a lane is free");
        session.set_sampling(Sampling::default());
        let first = session.prefill(&[1, 2, 3]).expect("the prompt runs");
        let token = first.next_token.expect("the prompt's last row is headed");
        assert_eq!(
            token,
            pick(3),
            "the shape walk picks the token after the last"
        );
        let next = session.decode(token).expect("the step runs");
        assert_eq!(next.next_token, Some(pick(token)));
    }

    #[test]
    fn the_lane_limit_is_the_walks_own_width() {
        assert_eq!(lane_limit(&Shape { rank: true }), Ok(4));
    }

    #[test]
    fn a_backend_is_refused_the_place_it_was_not_opened_for() {
        let refused = serve(&Shape { rank: true }, 1).expect_err("a rank does not serve alone");
        assert!(refused.to_string().contains("a group's rank"), "{refused}");
    }
}
