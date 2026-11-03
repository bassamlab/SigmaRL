# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Drive the agents of a road-traffic environment from recorded tracks (log replay).

The environment has a fixed number of agents ("slots"). A `LogReplay` attached to the scenario
(`scenario.log_replay`) gives each slot one of three roles at each time step:

- parked: off the map, standing still, hidden from observations and CBFs (`scenario.get_on_hold_agents_mask`);
- replayed: follows a recorded track (state overwritten after each world step) and ignores its action; the
  CBF-QP treats it as a non-reactive obstacle (`scenario.get_uncontrolled_agents_mask`);
- controlled: driven by the policy or controller under test. It is spawned at the recorded state of a
  human-driven vehicle (optionally with a corrected start pose) when that vehicle appears and its entry check
  passes (otherwise the entry is delayed and retried at every step), and parked when it reaches the point where
  the human left the recording (its goal) or after a timeout. It has the nominal vehicle parameters of the
  agent profile, or its own (`ControlledVehicle.params`, e.g., the recorded footprint), which its slot keeps
  until it is parked.

The scenario calls `on_reset` at the end of a full reset and `on_post_step` after each world step, before
the shared state of the transition (distances, collisions) is computed. The environment never resets
agents while a log replay is attached.
"""

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch

from sigmarl.constants import AGENTS
from sigmarl.round_tracks import RoutePolyline, Track, steering_from_yaw_rate
from sigmarl.vehicle_params import VehicleParams

# Parked agents wait in a row far away from the map
PARK_ORIGIN = np.array([-2000.0, -2000.0])
PARK_SPACING = 100.0


@dataclass
class ReplayedVehicle:
    """A recorded road user, or one car-sized box of a long one, that is replayed."""

    track: Track
    offset: float = (
        0.0  # Longitudinal offset of the box center from the track center [m]
    )
    path_id: int = (
        0  # Reference path assigned to the slot (used by the observations of the slot)
    )
    slot: Optional[int] = None


@dataclass
class ControlledVehicle:
    """A human-driven vehicle whose start and goal are given to the controller under test."""

    track: Track
    path_id: int
    route: RoutePolyline
    s_goal: float  # Arc length along the route where the human left the recording [m]
    max_duration: float  # The vehicle is parked (timeout) after this time since `t_begin` [s]
    k_start: int = 0  # Recorded sample at which control starts (> 0: already on the road when the episode starts)
    start_xy: Optional[
        np.ndarray
    ] = None  # Start position overriding the recorded one (e.g., repaired) [m]
    start_yaw: Optional[float] = None  # Start yaw overriding the recorded one [rad]
    start_repair: Optional[
        object
    ] = None  # Outcome of the start-pose repair that set start_xy and start_yaw, if any
    # Vehicle parameters (footprint, axle distances, acceleration limits); None: the nominal ones
    params: Optional[VehicleParams] = None
    slot: Optional[int] = None
    t_spawn: Optional[
        float
    ] = None  # Recording time at which it was spawned (may be later than recorded)
    # "pending", "active", "goal", or "timeout" (also if never spawned)
    status: str = "pending"
    t_finish: Optional[float] = None
    n_entry_rejections: int = 0  # Steps at which the entry check delayed the spawn
    is_entry_repair_tried: bool = (
        False  # The entry repair searched a certified start pose
    )
    # Certified entry check: (lane, pair) values [m] at the spawn and at the first rejection
    entry_values: Optional[tuple] = None
    first_rejection_values: Optional[tuple] = None
    s: float = 0.0
    seg: int = 0
    log: dict = field(default_factory=lambda: {k: [] for k in LOG_KEYS})

    @property
    def t_begin(self) -> float:
        """Recording time from which the vehicle is controlled [s]."""
        return float(self.track.t[self.k_start])

    @property
    def wheelbase(self) -> float:
        """Wheelbase of the simulated vehicle [m]."""
        return AGENTS["l_wb"] if self.params is None else self.params.l_wb


LOG_KEYS = ("t", "x", "y", "yaw", "speed", "steering", "acc", "steering_rate", "s")


class LogReplay:
    """Spawn, replay, and park the agents of one environment according to recorded tracks.

    Args:
        t0: Recording time [s] that corresponds to simulation step 0.
        dt: Simulation step [s].
        n_slots: Number of agents of the environment.
        controlled: Vehicles driven by the controller under test.
        replayed: Recorded road users that are replayed.
        entry_check: Certified entry check of a controlled vehicle, or None for the distance gate (see
            `spawn_gap_time`). It is called after the vehicle is placed at its recorded state in a free slot,
            as `entry_check(scenario, slot, others, held)` with the slots of the other vehicles on the map
            and the replayed ones among them, and returns (is certified, lane value, pair value). If the
            vehicle is not certified, it is parked again and its spawn is delayed (retried at every step,
            at the same recorded state). All vehicles are checked, also those that start mid-track. A
            recorded speed above the vehicle's maximum speed enters at the maximum speed: the filter's
            backup prediction, and thus the certificate, holds only within the speed limits.
        entry_repair: Certified start-pose repair (with `entry_check`), or None. When the entry check of a
            vehicle first fails because of its own lane constraint (negative lane value), it is called as
            `entry_repair(scenario, slot, vehicle, speed, steering)` and returns a start pose (xy, yaw, shift,
            heading change) whose braking backup stays in the lane, or None. The vehicle then uses that start
            pose from this step on; its pair constraints are checked as before.
        spawn_gap_time: Distance gate (`entry_check` None): a controlled vehicle is spawned only if no other
            active controlled vehicle is closer than `vehicle length + spawn_gap_time * speed` to its spawn
            position; otherwise its spawn is delayed. Replayed vehicles are not checked: they are where they
            were when the human appeared. Vehicles that start mid-track (`k_start > 0`, already on the road
            at the episode start) are not checked either: they are placed where the humans were.
        is_record_snapshots: Store the states of all active vehicles at each step (for evaluation).
        is_overlay_humans: Let the scenario outline the recorded humans of the controlled vehicles when it
            renders (see `human_overlay`).
        is_humans_only: Do not simulate the controlled vehicles, only outline their humans (human video); the
            vehicles keep the slots of a previous run, which give the colors of their humans.
    """

    def __init__(
        self,
        t0: float,
        dt: float,
        n_slots: int,
        controlled: list,
        replayed: list,
        device="cpu",
        entry_check: Optional[Callable] = None,
        entry_repair: Optional[Callable] = None,
        spawn_gap_time: float = 1.0,
        is_record_snapshots: bool = True,
        is_overlay_humans: bool = False,
        is_humans_only: bool = False,
    ):
        self.t0 = t0
        self.dt = dt
        self.n_slots = n_slots
        self.controlled = controlled
        self.replayed_vehicles = replayed
        self.entry_check = entry_check
        self.entry_repair = entry_repair
        self.spawn_gap_time = spawn_gap_time
        self.is_record_snapshots = is_record_snapshots
        self.is_overlay_humans = is_overlay_humans or is_humans_only
        self.is_humans_only = is_humans_only
        self.step = 0
        self.parked = torch.ones((1, n_slots), dtype=torch.bool, device=device)
        self.replayed = torch.zeros((1, n_slots), dtype=torch.bool, device=device)
        self.slot_owner = [None] * n_slots
        self.max_slots_used = 0
        self.n_replay_steps_dropped = (
            0  # Steps in which a replayed vehicle found no free agent
        )
        self.snapshots = []  # Per step: (t, list of (kind, index, x, y, yaw, speed))
        self._wheelbase = AGENTS["l_wb"]
        self._max_steering = AGENTS["max_steering"]
        self._max_speed = AGENTS["max_speed"]

    @property
    def t(self) -> float:
        """Current recording time [s]."""
        return self.t0 + self.step * self.dt

    @property
    def is_finished(self) -> bool:
        return all(v.status in ("goal", "timeout") for v in self.controlled)

    # ------------------------------------------------------------------ scenario hooks

    def on_reset(self, scenario, env_index: int):
        assert env_index == 0, "LogReplay supports a single environment."
        self.step = 0
        for slot in range(self.n_slots):
            self._park(scenario, slot)
        self._update(scenario)

    def on_post_step(self, scenario):
        self.step += 1
        self._update(scenario)

    def human_overlay(self) -> list:
        """Recorded footprints of the humans that the controlled vehicles replace, at the current time.

        Returns:
            Per human present in the recording: its corners [4, 2] (recorded length and width) and the slot
            of its controlled vehicle, or None if that vehicle is not active (not yet spawned or finished).
            With `is_humans_only`, the slot the vehicle had in the previous run (None if it never spawned).
        """
        eps = 1e-6
        overlay = []
        for vehicle in self.controlled:
            track = vehicle.track
            if not track.t_start - eps <= self.t <= track.t_end + eps:
                continue
            x, y, yaw, _ = track.state_at(self.t)
            lon = 0.5 * track.length * np.array([math.cos(yaw), math.sin(yaw)])
            lat = 0.5 * track.width * np.array([-math.sin(yaw), math.cos(yaw)])
            center = np.array([x, y])
            corners = np.stack(
                [
                    center + lon + lat,
                    center - lon + lat,
                    center - lon - lat,
                    center + lon - lat,
                ]
            )
            is_colored = self.is_humans_only or vehicle.status == "active"
            slot = vehicle.slot if is_colored else None
            overlay.append((corners, slot))
        return overlay

    # ------------------------------------------------------------------ internals

    def _free_slot(self) -> Optional[int]:
        """A parked agent, or None if all agents are in use."""
        for slot, owner in enumerate(self.slot_owner):
            if owner is None:
                return slot
        return None

    def _park(self, scenario, slot: int):
        pos = PARK_ORIGIN + np.array([PARK_SPACING * slot, 0.0])
        scenario.set_vehicle_params(slot, None)
        scenario.place_agent(0, slot, pos, rot=0.0, speed=0.0, steering=0.0)
        scenario.sync_agent_history(0, slot)
        self.parked[0, slot] = True
        self.replayed[0, slot] = False
        self.slot_owner[slot] = None

    def _occupy(self, slot: int, owner, is_replayed: bool):
        self.slot_owner[slot] = owner
        self.parked[0, slot] = False
        self.replayed[0, slot] = is_replayed
        self.max_slots_used = max(
            self.max_slots_used, sum(o is not None for o in self.slot_owner)
        )

    def _replay_pose(self, vehicle: ReplayedVehicle, t: float):
        x, y, yaw, speed = vehicle.track.state_at(t)
        x += vehicle.offset * math.cos(yaw)
        y += vehicle.offset * math.sin(yaw)
        k = int(
            np.clip(np.searchsorted(vehicle.track.t, t), 0, len(vehicle.track.t) - 1)
        )
        steering = steering_from_yaw_rate(
            vehicle.track, k, self._wheelbase, self._max_steering
        )
        return np.array([x, y]), float(yaw), float(speed), steering

    def _controlled_positions(self, scenario):
        return [
            scenario.world.agents[v.slot].state.pos[0].cpu().numpy()
            for v in self.controlled
            if v.status == "active"
        ]

    def _log_controlled(self, scenario, vehicle: ControlledVehicle):
        state = scenario.world.agents[vehicle.slot].state
        xy = state.pos[0].cpu().numpy().astype(float)
        lo, hi = max(0, vehicle.seg - 3), min(
            len(vehicle.route.seg_len), vehicle.seg + 12
        )
        s, _, seg, _ = vehicle.route.project(xy[None], lo, hi)
        vehicle.s, vehicle.seg = float(s[0]), int(seg[0])
        values = (
            self.t,
            xy[0],
            xy[1],
            float(state.rot[0, 0]),
            float(state.speed[0, 0]),
            float(state.steering[0, 0]),
            float(state.acceleration[0, 0]) if state.acceleration is not None else 0.0,
            float(state.steering_rate[0, 0])
            if state.steering_rate is not None
            else 0.0,
            vehicle.s,
        )
        for key, value in zip(LOG_KEYS, values):
            vehicle.log[key].append(value)

    def _update(self, scenario):
        t = self.t
        eps = 1e-6

        # Controlled vehicles: log, then park those that reached their goals or timed out
        for vehicle in self.controlled:
            if self.is_humans_only:
                continue
            if (
                vehicle.status == "pending"
                and t - vehicle.t_begin > vehicle.max_duration
            ):
                # Never spawned (entry delayed until the timeout)
                vehicle.status = "timeout"
                vehicle.t_finish = t
            if vehicle.status != "active":
                continue
            self._log_controlled(scenario, vehicle)
            s_target = min(vehicle.s_goal, vehicle.route.length - 1.0)
            if vehicle.s >= s_target:
                vehicle.status = "goal"
            elif t - vehicle.t_begin > vehicle.max_duration:
                vehicle.status = "timeout"
            if vehicle.status != "active":
                vehicle.t_finish = t
                self._park(scenario, vehicle.slot)

        # Replayed vehicles: follow the recording, appear and disappear with it
        for vehicle in self.replayed_vehicles:
            is_present = vehicle.track.t_start - eps <= t <= vehicle.track.t_end + eps
            if vehicle.slot is not None and not is_present:
                self._park(scenario, vehicle.slot)
                vehicle.slot = None
            elif is_present:
                pos, yaw, speed, steering = self._replay_pose(vehicle, t)
                if vehicle.slot is None:
                    vehicle.slot = self._free_slot()
                    if vehicle.slot is None:
                        self.n_replay_steps_dropped += (
                            1  # Not enough agents; counted for the evaluation
                        )
                        continue
                    self._occupy(vehicle.slot, vehicle, is_replayed=True)
                    scenario.place_agent(
                        0,
                        vehicle.slot,
                        pos,
                        yaw,
                        speed,
                        steering,
                        path_id=vehicle.path_id,
                    )
                    scenario.sync_agent_history(0, vehicle.slot)
                else:
                    scenario.place_agent(0, vehicle.slot, pos, yaw, speed, steering)

        # Controlled vehicles: spawn at the recorded initial state once the human appeared and the entry
        # check passes
        for vehicle in self.controlled:
            if (
                self.is_humans_only
                or vehicle.status != "pending"
                or t < vehicle.t_begin - eps
            ):
                continue
            k0 = vehicle.k_start
            pos = vehicle.track.xy[k0] if vehicle.start_xy is None else vehicle.start_xy
            speed = float(vehicle.track.speed[k0])
            if self.entry_check is not None:
                speed = min(speed, self._max_speed)
            if self.entry_check is None:
                clearance = AGENTS["length"] + self.spawn_gap_time * speed
                if k0 == 0 and any(
                    np.linalg.norm(p - pos) < clearance
                    for p in self._controlled_positions(scenario)
                ):
                    vehicle.n_entry_rejections += 1
                    continue
            steering = steering_from_yaw_rate(
                vehicle.track, k0, vehicle.wheelbase, self._max_steering
            )
            yaw = (
                vehicle.track.heading[k0]
                if vehicle.start_yaw is None
                else vehicle.start_yaw
            )
            yaw = float((yaw + math.pi) % (2 * math.pi) - math.pi)
            slot = self._free_slot()
            if slot is None:
                continue  # All agents in use: the spawn is delayed
            scenario.set_vehicle_params(slot, vehicle.params)
            scenario.place_agent(
                0, slot, pos, yaw, speed, steering, path_id=vehicle.path_id
            )
            if self.entry_check is not None:
                others = [
                    s for s, owner in enumerate(self.slot_owner) if owner is not None
                ]
                held = [s for s in others if bool(self.replayed[0, s])]
                is_certified, lane_value, pair_value = self.entry_check(
                    scenario, slot, others, held
                )
                if (
                    not is_certified
                    and lane_value < 0.0
                    and self.entry_repair is not None
                    and not vehicle.is_entry_repair_tried
                ):
                    vehicle.is_entry_repair_tried = True
                    repaired = self.entry_repair(
                        scenario, slot, vehicle, speed, steering
                    )
                    if repaired is not None:
                        pos, yaw, shift, dyaw = repaired
                        vehicle.start_xy, vehicle.start_yaw = pos, yaw
                        vehicle.start_repair = _backup_repair(
                            vehicle.start_repair, pos, yaw, shift, dyaw
                        )
                    scenario.place_agent(
                        0, slot, pos, yaw, speed, steering, path_id=vehicle.path_id
                    )
                    is_certified, lane_value, pair_value = self.entry_check(
                        scenario, slot, others, held
                    )
                if not is_certified:
                    self._park(scenario, slot)
                    vehicle.n_entry_rejections += 1
                    if vehicle.first_rejection_values is None:
                        vehicle.first_rejection_values = (lane_value, pair_value)
                    continue
                vehicle.entry_values = (lane_value, pair_value)
            vehicle.slot = slot
            self._occupy(vehicle.slot, vehicle, is_replayed=False)
            scenario.sync_agent_history(0, vehicle.slot)
            vehicle.status = "active"
            vehicle.t_spawn = t
            s, _, seg, _ = vehicle.route.project(pos[None])
            vehicle.s, vehicle.seg = float(s[0]), int(seg[0])
            self._log_controlled(scenario, vehicle)

        if self.is_record_snapshots:
            self._record_snapshot(scenario)

    def _record_snapshot(self, scenario):
        """States of all active vehicles: controlled ones from the simulator, replayed ones from the record."""
        entries = []
        for idx, vehicle in enumerate(self.controlled):
            if vehicle.status == "active":
                state = scenario.world.agents[vehicle.slot].state
                entries.append(
                    (
                        "c",
                        idx,
                        float(state.pos[0, 0]),
                        float(state.pos[0, 1]),
                        float(state.rot[0, 0]),
                        float(state.speed[0, 0]),
                    )
                )
        seen = set()
        for vehicle in self.replayed_vehicles:
            key = (vehicle.track.recording_id, vehicle.track.track_id)
            if vehicle.slot is None or key in seen:
                continue
            seen.add(key)  # One entry per recorded road user, not per box
            x, y, yaw, speed = vehicle.track.state_at(self.t)
            entries.append(("r", key[1], x, y, yaw, speed))
        self.snapshots.append((self.t, entries))


def split_into_boxes(track: Track, box_length: float) -> list:
    """Longitudinal offsets of car-sized boxes that together cover a long vehicle."""
    n_boxes = max(1, math.ceil(track.length / box_length - 0.15))
    if n_boxes == 1:
        return [0.0]
    half_span = (track.length - box_length) / 2
    return list(np.linspace(-half_span, half_span, n_boxes))


def _backup_repair(outcome, xy, yaw: float, shift: float, dyaw: float):
    """Start-pose repair outcome (`start_repair.StartRepair`) of a start pose that the entry repair moved
    so that its braking backup stays in the lane."""
    from sigmarl.evaluation_human_comparison.start_repair import StartRepair

    barrier_recorded = float("nan") if outcome is None else outcome.barrier_recorded
    return StartRepair(
        xy=np.asarray(xy, dtype=np.float64),
        yaw=float(yaw),
        barrier_recorded=barrier_recorded,
        barrier=float("nan"),
        shift=float(shift),
        dyaw=float(dyaw),
        status="repaired",
        is_backup_repair=True,
    )
