# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""WorldLab: a VMAS world whose physics is the CPM lab.

``WorldCustom.step`` converts each agent's RL action into a clamped
(speed, steering) target plus low-level control and integrates the kinematic
bicycle model. ``WorldLab.step`` keeps the conversion half bit-identical
(including the ``action.u`` / ``acceleration`` / ``steering_rate`` bookkeeping
that rewards and infos read) but delegates the integration to a bridge: it
sends the targets to the lab and writes the *measured* next states back into
``entity.state``. Everything downstream (observations, rewards, collisions,
done) reads ``entity.state`` and is none the wiser.
"""

import math
import time

from termcolor import cprint
from tensordict import TensorDict
import torch

from sigmarl.helper_common import compute_applied_action_control
from sigmarl.helper_training import (
    WorldCustom,
)
from sigmarl.lab.bridge_base import BridgeBase


class WorldLab(WorldCustom):
    def __init__(
        self,
        batch_dim: int,
        device,
        bridge: BridgeBase,
        check_stale_states: bool = False,
        stale_state_atol: float = 1e-5,
        scenario=None,
        parameters=None,
        **kwargs,
    ):
        assert batch_dim == 1, (
            "The lab/mock physics backend drives one physical fleet; "
            "num_vmas_envs must be 1."
        )
        super().__init__(batch_dim, device, **kwargs)
        self.bridge = bridge
        self.scenario = scenario
        self.parameters = parameters
        self.policy = None
        self.last_trajectories = None
        self.last_marl_horizon = None

        # Diagnostic: flag when a step's measured snapshot is identical to the
        # previous step's. A duplicate means the world advanced one step but
        # the lab reported no new physics — typically because the freshness
        # gate (command_valid_after_s) is too small and we accepted a stale /
        # re-published bridge tick. Watch stale_state_count / step_count while
        # tuning command_valid_after_s.
        self.check_stale_states = check_stale_states
        self.stale_state_atol = stale_state_atol
        self._prev_state_signature = None
        self.step_count = 0
        self.stale_state_count = 0

    def set_policy(self, policy) -> None:
        """Attach policy for virtual trajectory rollouts."""
        self.policy = policy

    def step(self):
        lab_mode = (
            getattr(self.parameters, "lab_mode", "speed_curvature")
            if self.parameters is not None
            else "speed_curvature"
        )
        if lab_mode == "trajectory":
            return self._step_trajectory()
        return self._step_speed_curvature()

    def _step_speed_curvature(self):
        targets = []
        for entity in self.entities:
            target_action, u = compute_applied_action_control(
                entity.action.u,
                entity.state.speed.squeeze(-1),
                entity.state.steering.squeeze(-1),
                self.dt,
                dynamics=entity.dynamics,
            )
            entity.action.u[:, 0] = target_action[:, 0]
            entity.action.u[:, 1] = target_action[:, 1]
            entity.state.acceleration = u[:, 0].unsqueeze(-1)
            entity.state.steering_rate = u[:, 1].unsqueeze(-1)
            targets.append(target_action[0])  # batch dim is 1

        self.bridge.apply_actions(torch.stack(targets, dim=0))
        states = self.bridge.wait_for_state_list()
        self.write_measured_states(states)
        if self.check_stale_states:
            self._check_stale_state(states)

    def _snapshot_observation_state(self):
        """Capture observation circular buffers, caches, and world_state for non-destructive virtual rollouts."""
        if self.scenario is None:
            return None
        obs_prov = self.scenario.observation_provider
        buf_snapshot = {}
        for attr_name in [
            "past_pos",
            "past_rot",
            "past_steering",
            "past_vertices",
            "past_vel",
            "past_short_term_ref_points",
            "past_action_vel",
            "past_action_steering",
            "past_distance_to_ref_path",
            "past_distance_to_boundaries",
            "past_distance_to_left_boundary",
            "past_distance_to_right_boundary",
            "past_distance_to_agents",
            "past_lengths",
            "past_widths",
            "past_left_boundary",
            "past_right_boundary",
            "past_id",
        ]:
            buf = getattr(obs_prov.observations, attr_name, None)
            if buf is not None and hasattr(buf, "buffer") and hasattr(buf, "pointer"):
                buf_snapshot[attr_name] = (
                    buf.pointer,
                    buf.valid_size,
                    buf.buffer.clone(),
                )

        caches = {
            "_latest_observations": list(getattr(obs_prov, "_latest_observations", [])),
            "_latest_positions_global": getattr(
                obs_prov, "_latest_positions_global", None
            ),
            "_latest_velocities_global": getattr(
                obs_prov, "_latest_velocities_global", None
            ),
            "stored_observations": list(
                getattr(self.scenario, "stored_observations", [])
            ),
        }

        ws = self.scenario.world_state
        ws_snapshot = {
            "route_progress": ws._route_progress.clone(),
            "route_progress_total_lengths": ws._route_progress_total_lengths.clone(),
            "route_progress_cache_valid": ws._route_progress_cache_valid.clone(),
            "route_progress_previous": ws._route_progress_previous.clone(),
            "vertices": ws.vertices.clone(),
            "dist_agents": ws.distances.agents.clone(),
            "dist_ref_paths": ws.distances.ref_paths.clone(),
            "dist_left_boundaries": ws.distances.left_boundaries.clone(),
            "dist_right_boundaries": ws.distances.right_boundaries.clone(),
            "dist_boundaries": ws.distances.boundaries.clone(),
            "short_term": ws.ref_paths_agent_related.short_term.clone(),
            "nearing_left": ws.ref_paths_agent_related.nearing_points_left_boundary.clone(),
            "nearing_right": ws.ref_paths_agent_related.nearing_points_right_boundary.clone(),
        }

        return buf_snapshot, caches, ws_snapshot

    def _restore_observation_state(self, snapshot):
        """Restore observation circular buffers, caches, and world_state to pre-rollout state."""
        if snapshot is None or self.scenario is None:
            return
        obs_prov = self.scenario.observation_provider
        buf_snapshot, caches, ws_snapshot = snapshot
        for attr_name, (ptr, vsz, buf_data) in buf_snapshot.items():
            buf = getattr(obs_prov.observations, attr_name, None)
            if buf is not None:
                buf.pointer = ptr
                buf.valid_size = vsz
                buf.buffer.copy_(buf_data)

        obs_prov._latest_observations = caches["_latest_observations"]
        obs_prov._latest_positions_global = caches["_latest_positions_global"]
        obs_prov._latest_velocities_global = caches["_latest_velocities_global"]
        if hasattr(self.scenario, "stored_observations"):
            self.scenario.stored_observations = caches["stored_observations"]

        ws = self.scenario.world_state
        ws._route_progress.copy_(ws_snapshot["route_progress"])
        ws._route_progress_total_lengths.copy_(
            ws_snapshot["route_progress_total_lengths"]
        )
        ws._route_progress_cache_valid.copy_(ws_snapshot["route_progress_cache_valid"])
        ws._route_progress_previous.copy_(ws_snapshot["route_progress_previous"])
        ws.vertices.copy_(ws_snapshot["vertices"])
        ws.distances.agents.copy_(ws_snapshot["dist_agents"])
        ws.distances.ref_paths.copy_(ws_snapshot["dist_ref_paths"])
        ws.distances.left_boundaries.copy_(ws_snapshot["dist_left_boundaries"])
        ws.distances.right_boundaries.copy_(ws_snapshot["dist_right_boundaries"])
        ws.distances.boundaries.copy_(ws_snapshot["dist_boundaries"])
        ws.ref_paths_agent_related.short_term.copy_(ws_snapshot["short_term"])
        ws.ref_paths_agent_related.nearing_points_left_boundary.copy_(
            ws_snapshot["nearing_left"]
        )
        ws.ref_paths_agent_related.nearing_points_right_boundary.copy_(
            ws_snapshot["nearing_right"]
        )

    def _step_trajectory(self):
        horizon = (
            getattr(self.parameters, "lab_traj_horizon", 3)
            if self.parameters is not None
            else 3
        )
        if horizon > 1 and self.policy is None:
            raise RuntimeError(
                "WorldLab policy is not set for lab_mode='trajectory'. "
                "Call env.scenario.set_policy(policy) before rolling out."
            )

        # 1. Step 0 action and control from the environment's current entity.action.u
        targets = []
        controls = []
        for entity in self.entities:
            target_action, u = compute_applied_action_control(
                entity.action.u,
                entity.state.speed.squeeze(-1),
                entity.state.steering.squeeze(-1),
                self.dt,
                dynamics=entity.dynamics,
            )
            entity.action.u[:, 0] = target_action[:, 0]
            entity.action.u[:, 1] = target_action[:, 1]
            entity.state.acceleration = u[:, 0].unsqueeze(-1)
            entity.state.steering_rate = u[:, 1].unsqueeze(-1)
            targets.append(target_action[0])
            controls.append(u[0])

        initial_targets = torch.stack(targets, dim=0)  # [n_agents, 2]
        initial_controls = torch.stack(controls, dim=0)  # [n_agents, 2]

        # Waypoint 0: current measured state [n_agents, 5] = (x, y, yaw, speed, steering)
        x_0 = torch.stack(
            [
                torch.cat(
                    [
                        e.state.pos[0],
                        e.state.rot[0],
                        e.state.speed[0],
                        e.state.steering[0],
                    ],
                    dim=-1,
                )
                for e in self.entities
            ],
            dim=0,
        )
        traj_states = [x_0]

        # Waypoint 1: integrate forward under initial action
        dynamics = self.entities[0].dynamics
        x_next, _, _ = dynamics.step(x_0, initial_controls, self.dt, tick_per_step=1)
        traj_states.append(x_next)

        # 2. Virtual steps 1 .. horizon - 1 using policy + bicycle model
        if horizon > 1:
            saved_entity_states = [
                (
                    e.state.pos.clone(),
                    e.state.rot.clone(),
                    e.state.speed.clone(),
                    e.state.steering.clone(),
                    e.state.vel.clone(),
                    e.state.sideslip_angle.clone(),
                    e.action.u.clone(),
                )
                for e in self.entities
            ]
            obs_snapshot = self._snapshot_observation_state()

            try:
                for _ in range(1, horizon):
                    x_cur = traj_states[-1]
                    # Update entities to virtual state x_cur
                    for i, e in enumerate(self.entities):
                        e.state.pos = x_cur[i, :2].unsqueeze(0)
                        e.state.rot = x_cur[i, 2:3].unsqueeze(0)
                        e.state.speed = x_cur[i, 3:4].unsqueeze(0)
                        e.state.steering = x_cur[i, 4:5].unsqueeze(0)
                        beta = torch.atan(
                            e.dynamics.l_r / e.dynamics.l_wb * torch.tan(x_cur[i, 4])
                        )
                        course = x_cur[i, 2] + beta
                        vel_vec = x_cur[i, 3] * torch.stack(
                            [torch.cos(course), torch.sin(course)], dim=-1
                        )
                        e.state.vel = vel_vec.unsqueeze(0)
                        e.state.sideslip_angle = beta.view(1, 1)

                    # Update world_state observation features for the virtual state
                    ws = self.scenario.world_state
                    agent_states = ws.get_agent_state_list()
                    if (
                        not ws.params.is_fast_inference_mode
                        or ws.params.is_observe_vertices
                    ):
                        ws.update_vertices(agent_states)
                    ws.update_mutual_distances(agent_states)
                    if (
                        ws.params.is_smooth_sampling
                        and ws.params.is_use_simple_road_distance
                    ):
                        ws._update_smooth_sampled_ref_paths_all_agents(agent_states)
                        ws._update_simple_road_distances_all_agents(agent_states)
                    else:
                        for agent_index in range(ws.n_agents):
                            ws.update_distances(agent_states, agent_index)

                    # Compute observation for virtual state
                    obs_list = [
                        self.scenario.observation(agent) for agent in self.agents
                    ]
                    obs = torch.stack(obs_list, dim=1)  # [1, n_agents, obs_dim]

                    with torch.inference_mode():
                        td = TensorDict(
                            {("agents", "observation"): obs},
                            batch_size=[1],
                            device=self.device,
                        )
                        td = self.policy(td)
                        virtual_raw = td.get(("agents", "action"))

                    # Compute applied control from virtual action
                    u_v_list = []
                    for i, e in enumerate(self.entities):
                        target_act, u_i = compute_applied_action_control(
                            virtual_raw[:, i],
                            e.state.speed.squeeze(-1),
                            e.state.steering.squeeze(-1),
                            self.dt,
                            dynamics=e.dynamics,
                        )
                        e.action.u[:, 0] = target_act[:, 0]
                        e.action.u[:, 1] = target_act[:, 1]
                        u_v_list.append(u_i[0])
                    u_v = torch.stack(u_v_list, dim=0)
                    x_next, _, _ = dynamics.step(x_cur, u_v, self.dt, tick_per_step=1)
                    traj_states.append(x_next)
            finally:
                # Restore original real state and observation history
                for i, e in enumerate(self.entities):
                    (
                        e.state.pos,
                        e.state.rot,
                        e.state.speed,
                        e.state.steering,
                        e.state.vel,
                        e.state.sideslip_angle,
                        e.action.u,
                    ) = saved_entity_states[i]
                self._restore_observation_state(obs_snapshot)

        # 3. Deceleration padding to meet lab MPC lead time requirement (>= 0.80 s)
        min_lead_time = 0.80
        current_span = horizon * self.dt
        if current_span < min_lead_time:
            n_pad = math.ceil((min_lead_time - current_span) / self.dt)
            for p_i in range(n_pad):
                t_rem = max(self.dt, (n_pad - p_i) * self.dt)
                x_last = traj_states[-1]
                v_cur = x_last[:, 3]
                steer_cur = x_last[:, 4]
                u_acc = torch.clamp(
                    -v_cur / t_rem,
                    dynamics.min_acc,
                    dynamics.max_acc,
                )
                u_steer = torch.clamp(
                    -steer_cur / t_rem,
                    dynamics.min_steering_rate,
                    dynamics.max_steering_rate,
                )
                u_pad = torch.stack([u_acc, u_steer], dim=-1)
                x_pad_next, _, _ = dynamics.step(
                    x_last, u_pad, self.dt, tick_per_step=1
                )
                traj_states.append(x_pad_next)

        trajectories = torch.stack(traj_states, dim=1)  # [n_agents, T_total, 5]
        self.last_trajectories = trajectories
        self.last_marl_horizon = horizon
        self.bridge.apply_trajectories(
            trajectories, self.dt, initial_target_actions=initial_targets
        )
        states = self.bridge.wait_for_state_list()
        self.write_measured_states(states)
        if self.check_stale_states:
            self._check_stale_state(states)

    def _check_stale_state(self, states: dict) -> bool:
        """Warn when the measured snapshot equals the previous step's.

        Returns True on a duplicate. Compares only consecutive ``step()``
        calls — a respawn changes the pose enough that it never registers as
        a false duplicate.
        """
        signature = torch.cat(
            [
                states["pos"].reshape(-1),
                states["rot"].reshape(-1),
                states["speed"].reshape(-1),
            ]
        )
        self.step_count += 1
        is_stale = False
        if self._prev_state_signature is not None:
            max_delta = (signature - self._prev_state_signature).abs().max().item()
            is_stale = max_delta <= self.stale_state_atol
            if is_stale:
                self.stale_state_count += 1
                cprint(
                    "[WorldLab] stale state: measured snapshot identical to "
                    f"the previous step (max_delta={max_delta:.2e}); "
                    f"{self.stale_state_count}/{self.step_count} steps stale. "
                    f"[world_lab] stale percentage: {self.stale_state_count / self.step_count * 100:.2f}% "
                    f"max_delta={max_delta:.2e}  stale atol={self.stale_state_atol} ",
                    "red",
                )

        self._prev_state_signature = signature
        return is_stale

    def write_measured_states(self, states: dict, agent_indices=None) -> None:
        """Write a bridge state dict into entity.state (the pull sync)."""
        indices = range(len(self.entities)) if agent_indices is None else agent_indices
        for i in indices:
            entity = self.entities[i]
            entity.state.pos = states["pos"][i].unsqueeze(0)
            entity.state.rot = states["rot"][i].unsqueeze(0)
            entity.state.speed = states["speed"][i].unsqueeze(0)
            entity.state.steering = states["steering"][i].unsqueeze(0)
            entity.state.vel = states["vel"][i].unsqueeze(0)
            entity.state.sideslip_angle = states["sideslip"][i].unsqueeze(0)
