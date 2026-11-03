# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""ScenarioRoadTrafficLab: ScenarioRoadTraffic with the physics layer swapped.

Only two things differ from the simulation scenario:

1. ``_init_world`` builds a ``WorldLab`` backed by a bridge (real lab via ROS
   or the in-process mock) instead of ``WorldCustom``.
2. ``reset_world_at`` puts the fleet into the reset pose. There are two modes:

   * **respawn** (default) — mirror the freshly *sampled* spawn poses into the
     lab via the respawn service and pull the measured states back. Works for
     the digital twin (simulated vehicles in the ControlCenter) and the mock.
   * **manual** (``Parameters.lab_manual_reset``) — for REAL hardware, which
     cannot be teleported. Stop the fleet, let the operator place the vehicles
     by hand, then read the measured poses and derive the reference paths from
     *them*. No pose is ever sampled or commanded.

Everything else — observations, rewards, collision detection, done logic,
state buffers — is inherited unchanged and operates on the measured states
that ``WorldLab.step`` writes into ``entity.state``.
"""

import torch
from termcolor import cprint

from sigmarl.scenarios.road_traffic import ScenarioRoadTraffic
from sigmarl.lab.bridge_base import LabStopped
from sigmarl.lab.world_lab import WorldLab
from sigmarl.lab.world_state_lab import WorldStateLab


def make_bridge(scenario, device):
    """Build the bridge configured by scenario.parameters.physics_backend."""
    p = scenario.parameters
    vehicle_ids = getattr(p, "lab_vehicle_ids", None) or list(
        range(1, scenario.n_agents + 1)
    )
    assert len(vehicle_ids) == scenario.n_agents, (
        f"lab_vehicle_ids ({vehicle_ids}) must have one entry per agent "
        f"(n_agents={scenario.n_agents})"
    )
    period_ms = getattr(p, "lab_period_ms", 100)

    if p.physics_backend == "mock":
        from sigmarl.lab.mock_bridge import MockBridge

        return MockBridge(
            vehicle_ids,
            period_ms,
            scenario.world_x_dim,
            scenario.world_y_dim,
            device=device,
            max_speed=scenario.max_speed,
            min_speed=scenario.min_speed,
        )

    # rclpy and cpm_lab_lab_msgs are imported only here, so pure-VMAS and
    # mock users never need a sourced ROS environment.
    from sigmarl.lab.lab_bridge import LabBridge

    return LabBridge(
        vehicle_ids,
        period_ms,
        scenario.world_x_dim,
        scenario.world_y_dim,
        device=device,
        state_timeout_s=getattr(p, "lab_state_timeout", 30.0),
    )


class ScenarioRoadTrafficLab(ScenarioRoadTraffic):

    # --- mode ---------------------------------------------------------------

    @property
    def is_manual_reset(self) -> bool:
        """True when resets are human-in-the-loop (real hardware)."""
        return bool(getattr(self.parameters, "lab_manual_reset", False))

    # --- world-state seam ---------------------------------------------------

    def _make_world_state(self, params):
        if not self.is_manual_reset:
            return super()._make_world_state(params)
        # Manual mode: assign reference paths to measured poses instead of
        # sampling poses. See sigmarl/lab/world_state_lab.py.
        return WorldStateLab(params, self.map)

    def _init_params(self, batch_dim, device, **kwargs):
        super()._init_params(batch_dim, device, **kwargs)
        if not self.is_manual_reset:
            return

        p = self.parameters
        # These two branches of ScenarioRoadTraffic.reset_world_at *impose* a
        # pose on the world and would silently overwrite the hand placement.
        # They must be guarded here rather than only inside
        # WorldStateLab._reset_init_state, because WorldStateRTSimulation.reset
        # applies a predefined pose without ever calling that method.
        assert p.predefined_ref_path_idx is None, (
            "lab_manual_reset=True forbids predefined_ref_path_idx: reference "
            "paths must be derived from the measured poses."
        )
        assert not p.is_challenging_initial_state_buffer, (
            "lab_manual_reset=True forbids is_challenging_initial_state_buffer: "
            "it replays recorded poses on top of the measured ones."
        )
        assert p.scenario_type != "cpm_mixed", (
            "lab_manual_reset=True requires a scenario whose reference paths "
            "cover the whole map (e.g. 'cpm_entire'); 'cpm_mixed' restricts the "
            "candidate set per env/agent, so a hand-placed vehicle may match no "
            "path."
        )
        if getattr(p, "reset_agent_fixed_duration", 0):
            cprint(
                "[lab] reset_agent_fixed_duration="
                f"{p.reset_agent_fixed_duration}s will prompt the operator on a "
                "timer. Set it to 0 unless you want that.",
                "yellow",
            )

        # NOTE: self.bridge does not exist yet -- make_world calls
        # _init_params before _init_world. The world state's vehicle_ids are
        # wired up in _init_world instead.
        self._manual_reset_count = 0
        self._step_count_at_last_prompt = None

    def _init_world(self, batch_dim: int, device: torch.device):
        p = self.parameters
        assert batch_dim == 1, (
            "The lab/mock physics backend drives one physical fleet; "
            "num_vmas_envs must be 1."
        )
        period_ms = getattr(p, "lab_period_ms", 100)
        assert abs(p.dt - period_ms / 1000.0) < 1e-9, (
            f"parameters.dt ({p.dt}) must equal lab_period_ms/1000 "
            f"({period_ms / 1000.0}): one world step is one bridge period."
        )

        self.bridge = make_bridge(self, device)
        if self.is_manual_reset:
            # Let the world state name physical vehicles in its diagnostics.
            self.world_state.vehicle_ids = list(self.bridge.vehicle_ids)
        if p.physics_backend == "lab":
            self.bridge.wait_for_first_state()

        return WorldLab(
            batch_dim,
            device,
            bridge=self.bridge,
            x_semidim=torch.tensor(
                self.world_x_dim, device=device, dtype=torch.float32
            ),
            y_semidim=torch.tensor(
                self.world_y_dim, device=device, dtype=torch.float32
            ),
            dt=p.dt,
            scenario=self,
            parameters=p,
        )

    def set_policy(self, policy):
        """Set the MARL policy on the scenario and WorldLab for virtual trajectory rollout."""
        self.policy = policy
        if hasattr(self, "world") and hasattr(self.world, "set_policy"):
            self.world.set_policy(policy)

    def reset_world_at(self, env_index: int = None, agent_index: int = None):
        if self.is_manual_reset:
            return self._reset_world_at_manually(env_index, agent_index)
        return self._reset_world_at_by_respawn(env_index, agent_index)

    # --- manual reset (real hardware) ---------------------------------------

    def _reset_world_at_manually(self, env_index=None, agent_index=None):
        """Stop, ask the operator, measure, then let the base class do the rest.

        Nothing from ``super().reset_world_at`` is duplicated here. Because the
        measured poses are written into ``entity.state`` *first*, and because
        ``WorldStateLab._reset_init_state`` derives the reference path from
        ``entity.state`` without ever writing it, the timer reset, path assignment, short-term-window
        seeding, mutual distances, collision reset, deadlock reset and state
        buffer write are all correct as they stand.
        """
        # Read the cause before super() calls reset_collisions and erases it.
        reason = self._describe_reset_reason(env_index, agent_index)

        states = self._acquire_manual_placement(reason)

        # (1) entity.state <- ground truth, before any path assignment.
        self.world.write_measured_states(states)

        # (2) The unchanged base reset, now operating on measured poses.
        super().reset_world_at(env_index, agent_index)

        # (3) super() only re-anchors the agent(s) it considers reset, but the
        #     operator may have moved any vehicle at this prompt.
        self._reanchor_unreset_agents(env_index, agent_index)

    def _acquire_manual_placement(self, reason: str, max_attempts: int = 10):
        """Prompt the operator, validate the placement, re-prompt if invalid."""
        vehicle_ids = list(self.bridge.vehicle_ids)
        self._manual_reset_count += 1

        step_count = getattr(self.world, "step_count", None)
        if step_count is not None and step_count == self._step_count_at_last_prompt:
            reason += (
                "  [NO PHYSICS HAS ADVANCED SINCE THE LAST PROMPT -- nothing "
                "has moved; just press ENTER]"
            )

        for attempt in range(max_attempts):
            header = f"#{self._manual_reset_count}" + (
                f" (retry {attempt})" if attempt else ""
            )
            states = self.bridge.manual_reset(vehicle_ids, f"{header}  {reason}")

            problems = self._check_placement(states)
            if not problems:
                self._step_count_at_last_prompt = getattr(
                    self.world, "step_count", None
                )
                return states
            for problem in problems:
                cprint(f"[lab] invalid placement: {problem}", "red")
            reason = "fix the placement problems listed above"

        raise LabStopped(
            f"Manual placement still invalid after {max_attempts} attempts."
        )

    def _check_placement(self, states) -> list:
        """Reject placements that would make the first step meaningless.

        Two vehicles closer than the reset minimum distance collide on step 0,
        which in manual mode means an immediate re-prompt (and in training mode
        trips the step-0 asserts in ScenarioRoadTraffic.done).
        """
        problems = []
        pos = states["pos"]  # [n_agents, 2], SigmaRL frame

        distances = torch.cdist(pos, pos)
        distances.fill_diagonal_(float("inf"))
        limit = float(self.constants.reset_agent_min_distance)
        min_distance = float(distances.min())
        if min_distance < limit:
            i, j = (int(v) for v in torch.nonzero(distances == distances.min())[0])
            problems.append(
                f"vehicles {self.bridge.vehicle_ids[i]} and "
                f"{self.bridge.vehicle_ids[j]} are {min_distance:.2f} m apart "
                f"(minimum {limit:.2f} m)"
            )

        for k in range(pos.shape[0]):
            x, y = float(pos[k, 0]), float(pos[k, 1])
            if not (0.0 <= x <= self.world_x_dim and 0.0 <= y <= self.world_y_dim):
                problems.append(
                    f"vehicle {self.bridge.vehicle_ids[k]} at ({x:.2f}, {y:.2f}) "
                    f"is outside the {self.world_x_dim:.2f} x "
                    f"{self.world_y_dim:.2f} m map"
                )

        return problems

    def _reanchor_unreset_agents(self, env_index, agent_index):
        """Re-derive paths for agents super() skipped, and refresh the whole
        state-buffer row.

        Both are load-bearing in manual mode, because the operator can move any
        vehicle at any prompt while super() only re-anchors ``agent_index``:

        * A stale reference path leaves the old left/right boundaries attached
          to a vehicle that is now somewhere else, so ``update_collisions``
          reports a boundary collision immediately -> another reset -> an
          endless prompt loop.
        * A stale ``state_buffer`` row makes the swept-boundary check draw a
          metres-long segment from the vehicle's old pose to its new one (a
          guaranteed spurious lanelet collision), and makes the forward-progress
          reward read that same bogus displacement on the first step.
        """
        if agent_index is None:
            return  # super() already covered every agent in this env

        env_i = 0 if env_index is None else int(env_index)
        reset_i = int(agent_index)
        world_state = self.world_state
        agent_states = world_state.get_agent_state_list()

        for i_agent in range(self.n_agents):
            if i_agent == reset_i:
                continue
            (
                ref_paths_scenario,
                extended_points,
            ) = world_state._reset_scenario_related_ref_paths(
                env_index=env_i, agent_index=i_agent
            )
            ref_path, path_id = world_state._reset_init_state(
                agent_states, ref_paths_scenario, agent_index=i_agent, env_index=env_i
            )
            world_state._reset_agent_related_ref_path(
                env_i, i_agent, ref_path, path_id, extended_points
            )
            world_state.reset_init_distances_and_short_term_ref_path(
                agent_states, env_i, i_agent
            )

        world_state.update_mutual_distances(agent_states, env_index=env_i)
        world_state.reset_collisions(env_i)
        # Whole env row, not just the reset agent (see docstring).
        self._sync_state_buffer_after_reset(
            state_add=self._get_current_state_tensor(),
            env_index=env_i,
            agent_index=None,
        )

    def _describe_reset_reason(self, env_index, agent_index) -> str:
        """Name the cause of this reset for the operator prompt.

        Must run *before* ``super().reset_world_at``, which calls
        ``reset_collisions`` and clears the evidence.
        """
        env_i = 0 if env_index is None else int(env_index)
        collisions = self.world_state.collisions

        causes = []
        if bool(collisions.with_agents[env_i].any()):
            causes.append("collision between vehicles")
        if bool(collisions.with_lanelets[env_i].any()):
            causes.append("collision with a lane boundary")
        if bool(collisions.with_entry_segments[env_i].any()):
            causes.append("left the entry segment")
        if bool(collisions.with_exit_segments[env_i].any()):
            causes.append("left the exit segment")
        deadlocked = getattr(self, "deadlock_reset_agents", None)
        if deadlocked is not None and bool(deadlocked[env_i].any()):
            agents = torch.where(deadlocked[env_i])[0].tolist()
            causes.append(f"deadlock (agents {agents})")
        if int(self.timer.step[env_i]) >= self.parameters.max_steps - 1:
            causes.append("max_steps reached (episode end)")
        if not causes:
            causes.append("episode start / startup staging")

        if agent_index is None:
            scope = "whole fleet"
        else:
            i_agent = int(agent_index)
            scope = f"agent {i_agent} (vehicle {self.bridge.vehicle_ids[i_agent]})"
        return f"{', '.join(causes)}  ->  reset {scope}"

    # --- respawn reset (digital twin / mock) --------------------------------

    def _reset_world_at_by_respawn(
        self, env_index: int = None, agent_index: int = None
    ):
        # Note: env/collector construction calls a few full resets back to
        # back, each of which is a real respawn (~2-3 s against the lab,
        # free against the mock). This is deliberately not optimized away so
        # reset semantics stay exactly those of the simulation scenario.
        super().reset_world_at(env_index, agent_index)

        reset_idx = (
            list(range(self.n_agents)) if agent_index is None else [int(agent_index)]
        )

        # Mirror the sampled spawn states (written into entity.state by
        # world_state.reset inside super()) to the lab, then pull the
        # measured result back. Backends apply what they physically can (the
        # mock applies pose + speed + steering; the real lab only places the
        # pose), so the pull-back is what keeps the twin truthful. Non-reset
        # vehicles are re-measured too, absorbing any drift accumulated
        # while the respawn was in flight.
        spawn_states = torch.stack(
            [
                torch.cat(
                    [
                        self.world.agents[i].state.pos[0],
                        self.world.agents[i].state.rot[0],
                        self.world.agents[i].state.speed[0],
                        self.world.agents[i].state.steering[0],
                    ]
                )
                for i in reset_idx
            ],
            dim=0,
        )
        vehicle_ids = [self.bridge.vehicle_ids[i] for i in reset_idx]
        self.bridge.respawn(vehicle_ids, spawn_states)
        states = self.bridge.wait_for_state_list()
        self.world.write_measured_states(states)

        # Re-run the pose-dependent tail of super().reset_world_at against
        # the measured poses (path-window seeding, mutual distances, state
        # buffer). Collision/deadlock resets are pose-independent and were
        # already done by super().
        env_j = slice(None) if env_index is None else env_index
        for i_agent in reset_idx:
            self.world_state.reset_init_distances_and_short_term_ref_path(
                self.world_state.get_agent_state_list(), env_j, i_agent
            )
        self.world_state.update_mutual_distances(
            self.world_state.get_agent_state_list(), env_index=env_j
        )
        state_add = self._get_current_state_tensor()
        if env_index is None and agent_index is None:
            self.state_buffer.reset()
            self.state_buffer.add(state_add)
        else:
            self._sync_state_buffer_after_reset(
                state_add=state_add,
                env_index=env_j,
                agent_index=agent_index,
            )
