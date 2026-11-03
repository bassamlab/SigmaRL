from dataclasses import dataclass
from typing import List

import torch

from sigmarl.constants import THRESHOLD
from sigmarl.helper_scenario import (
    Thresholds,
    Constants,
    Normalizers,
    Observations,
    CircularBuffer,
    angle_eliminate_two_pi,
)
from sigmarl.helper_training import WorldCustom
from sigmarl.map_manager import MapManager
from sigmarl.scenarios.observations.observation_provider import (
    ObservationProviderParameters,
    ObservationProvider,
    AgentState,
)
from sigmarl.scenarios.world_state.world_state_rt.world_state_rt import WorldStateRT
from sigmarl.scenarios.world_state.world_state_rt.world_state_rt_sim import (
    WorldStateRTSimulation,
)


@dataclass
class ObservationProviderParametersRT(ObservationProviderParameters):
    n_observed_steps: int
    n_points_nearing_boundary: int
    n_nearing_agents_observed: int
    n_points_short_term: int
    is_observe_distance_to_boundaries: bool
    is_observe_boundary_points: bool
    is_ego_view: bool
    is_using_opponent_modeling: bool
    is_apply_distance_mask: bool
    is_apply_velocity_mask: bool
    rl_obs_range: float
    visible_agent_dropout_probability: float
    is_testing_mode: bool
    is_partial_observation: bool
    is_obs_steering: bool
    is_observe_vertices: bool
    is_observe_distance_to_agents: bool
    is_observe_ref_path_other_agents: bool
    is_observe_distance_to_center_line: bool
    velocity_mask_margin: float = 0.5
    is_fast_inference_mode: bool = False  # Whether fast inference mode is active (e.g. skip render diagnostic dicts)


def neighbor_observation_layout(params) -> tuple[int, bool]:
    """Return the per-neighbor feature count and whether a valid flag is included.

    With ego view and partial observation, the observation ends with
    ``n_nearing_agents_observed`` neighbor blocks of this size, ordered as in
    ``_observe_other_agents``; the valid flag, if present, is the last entry
    of each block.

    Args:
        params: Object with the observation flags of ``Parameters``.
    """
    block_dim = 8 if params.is_observe_vertices else 5  # Vertices or pose + size
    block_dim += 2  # Velocity
    if params.is_obs_steering:
        block_dim += 1
    if params.is_observe_distance_to_agents:
        block_dim += 1
    if params.is_observe_ref_path_other_agents:
        block_dim += 2 * params.n_points_short_term
    has_valid_flag = params.is_partial_observation and (
        params.is_apply_distance_mask or params.is_apply_velocity_mask
    )
    if has_valid_flag:
        block_dim += 1
    return block_dim, has_valid_flag


def _transform_from_global_to_all_local_coordinates(
    ego_positions: torch.Tensor,
    ego_rotations: torch.Tensor,
    target_positions: torch.Tensor,
) -> torch.Tensor:
    """Transform all agent-aligned target points into every ego coordinate frame."""
    position_vectors = target_positions.unsqueeze(1) - ego_positions.unsqueeze(
        2
    ).unsqueeze(3)
    position_magnitudes = torch.norm(position_vectors, dim=-1)
    relative_angles = torch.atan2(
        position_vectors[..., 1], position_vectors[..., 0]
    ) - ego_rotations.unsqueeze(-1).unsqueeze(-1)
    return torch.stack(
        (
            torch.cos(relative_angles) * position_magnitudes,
            torch.sin(relative_angles) * position_magnitudes,
        ),
        dim=-1,
    )


class ObservationProviderRT(ObservationProvider):
    def __init__(
        self,
        params: ObservationProviderParametersRT,
        constants: Constants,
        normalizers: Normalizers,
        thresholds: Thresholds,
        map: MapManager,
        world_state: WorldStateRT,
    ):

        super().__init__(params, constants, normalizers)

        self.params = params

        self.thresholds = thresholds
        self.map = map
        self.world_state = world_state

        self.observations = Observations(
            n_nearing_agents=torch.tensor(
                params.n_nearing_agents_observed,
                device=self.device,
                dtype=torch.int32,
            ),
            obs_noise_level=torch.tensor(
                params.obs_noise_level, device=self.device, dtype=torch.float32
            ),
            n_stored_steps=torch.tensor(
                params.n_stored_steps, device=self.device, dtype=torch.int32
            ),
            n_observed_steps=torch.tensor(
                params.n_observed_steps, device=self.device, dtype=torch.int32
            ),
            nearing_agents_indices=torch.zeros(
                (self.batch_dim, self.n_agents, params.n_nearing_agents_observed),
                device=self.device,
                dtype=torch.int32,
            ),
        )
        self._latest_positions_global = None
        self._latest_velocities_global = None
        self._latest_observed_agent_render_diagnostics = {}
        self._latest_observations = [None] * self.n_agents
        self._on_hold_agents_mask = None
        n_neighbors = (
            params.n_nearing_agents_observed
            if params.is_partial_observation
            else self.n_agents
        )
        self._latest_neighbor_valid = torch.ones(
            (self.batch_dim, self.n_agents, n_neighbors),
            device=self.device,
            dtype=torch.bool,
        )

        # initialize observations
        assert (
            self.observations.n_stored_steps >= 1
        ), "The number of stored steps should be at least 1."
        assert (
            self.observations.n_observed_steps >= 1
        ), "The number of observed steps should be at least 1."
        assert (
            self.observations.n_stored_steps >= self.observations.n_observed_steps
        ), "The number of stored steps should be greater or equal than the number of observed steps."

        if self.params.is_ego_view:
            self.observations.past_pos = CircularBuffer(
                torch.zeros(
                    (
                        self.params.n_stored_steps,
                        self.batch_dim,
                        self.n_agents,
                        self.n_agents,
                        2,
                    ),
                    device=self.device,
                    dtype=torch.float32,
                )
            )
            self.observations.past_rot = CircularBuffer(
                torch.zeros(
                    (
                        self.params.n_stored_steps,
                        self.batch_dim,
                        self.n_agents,
                        self.n_agents,
                    ),
                    device=self.device,
                    dtype=torch.float32,
                )
            )
            self.observations.past_vertices = CircularBuffer(
                torch.zeros(
                    (
                        self.params.n_stored_steps,
                        self.batch_dim,
                        self.n_agents,
                        self.n_agents,
                        4,
                        2,
                    ),
                    device=self.device,
                    dtype=torch.float32,
                )
            )
            self.observations.past_vel = CircularBuffer(
                torch.zeros(
                    (
                        self.params.n_stored_steps,
                        self.batch_dim,
                        self.n_agents,
                        self.n_agents,
                        2,
                    ),
                    device=self.device,
                    dtype=torch.float32,
                )
            )
            self.observations.past_short_term_ref_points = CircularBuffer(
                torch.zeros(
                    (
                        self.params.n_stored_steps,
                        self.batch_dim,
                        self.n_agents,
                        self.n_agents,
                        self.params.n_points_short_term,
                        2,
                    ),
                    device=self.device,
                    dtype=torch.float32,
                )
            )
            self.observations.past_left_boundary = CircularBuffer(
                torch.zeros(
                    (
                        self.params.n_stored_steps,
                        self.batch_dim,
                        self.n_agents,
                        self.n_agents,
                        self.params.n_points_nearing_boundary,
                        2,
                    ),
                    device=self.device,
                    dtype=torch.float32,
                )
            )
            self.observations.past_right_boundary = CircularBuffer(
                torch.zeros(
                    (
                        self.params.n_stored_steps,
                        self.batch_dim,
                        self.n_agents,
                        self.n_agents,
                        self.params.n_points_nearing_boundary,
                        2,
                    ),
                    device=self.device,
                    dtype=torch.float32,
                )
            )
        else:
            # Bird view
            self.observations.past_pos = CircularBuffer(
                torch.zeros(
                    (self.params.n_stored_steps, self.batch_dim, self.n_agents, 2),
                    device=self.device,
                    dtype=torch.float32,
                )
            )
            self.observations.past_rot = CircularBuffer(
                torch.zeros(
                    (self.params.n_stored_steps, self.batch_dim, self.n_agents),
                    device=self.device,
                    dtype=torch.float32,
                )
            )
            self.observations.past_vertices = CircularBuffer(
                torch.zeros(
                    (self.params.n_stored_steps, self.batch_dim, self.n_agents, 4, 2),
                    device=self.device,
                    dtype=torch.float32,
                )
            )
            self.observations.past_vel = CircularBuffer(
                torch.zeros(
                    (self.params.n_stored_steps, self.batch_dim, self.n_agents, 2),
                    device=self.device,
                    dtype=torch.float32,
                )
            )
            self.observations.past_short_term_ref_points = CircularBuffer(
                torch.zeros(
                    (
                        self.params.n_stored_steps,
                        self.batch_dim,
                        self.n_agents,
                        self.params.n_points_short_term,
                        2,
                    ),
                    device=self.device,
                    dtype=torch.float32,
                )
            )
            self.observations.past_left_boundary = CircularBuffer(
                torch.zeros(
                    (
                        self.params.n_stored_steps,
                        self.batch_dim,
                        self.n_agents,
                        self.params.n_points_nearing_boundary,
                        2,
                    ),
                    device=self.device,
                    dtype=torch.float32,
                )
            )
            self.observations.past_right_boundary = CircularBuffer(
                torch.zeros(
                    (
                        self.params.n_stored_steps,
                        self.batch_dim,
                        self.n_agents,
                        self.params.n_points_nearing_boundary,
                        2,
                    ),
                    device=self.device,
                    dtype=torch.float32,
                )
            )

        self.observations.past_action_vel = CircularBuffer(
            torch.zeros(
                (self.params.n_stored_steps, self.batch_dim, self.n_agents),
                device=self.device,
                dtype=torch.float32,
            )
        )
        self.observations.past_action_steering = CircularBuffer(
            torch.zeros(
                (self.params.n_stored_steps, self.batch_dim, self.n_agents),
                device=self.device,
                dtype=torch.float32,
            )
        )
        self.observations.past_distance_to_ref_path = CircularBuffer(
            torch.zeros(
                (self.params.n_stored_steps, self.batch_dim, self.n_agents),
                device=self.device,
                dtype=torch.float32,
            )
        )
        self.observations.past_distance_to_boundaries = CircularBuffer(
            torch.zeros(
                (self.params.n_stored_steps, self.batch_dim, self.n_agents),
                device=self.device,
                dtype=torch.float32,
            )
        )
        self.observations.past_distance_to_left_boundary = CircularBuffer(
            torch.zeros(
                (self.params.n_stored_steps, self.batch_dim, self.n_agents),
                device=self.device,
                dtype=torch.float32,
            )
        )
        self.observations.past_distance_to_right_boundary = CircularBuffer(
            torch.zeros(
                (self.params.n_stored_steps, self.batch_dim, self.n_agents),
                device=self.device,
                dtype=torch.float32,
            )
        )
        self.observations.past_distance_to_agents = CircularBuffer(
            torch.zeros(
                (
                    self.params.n_stored_steps,
                    self.batch_dim,
                    self.n_agents,
                    self.n_agents,
                ),
                device=self.device,
                dtype=torch.float32,
            )
        )
        self.observations.past_lengths = CircularBuffer(
            torch.zeros(
                (self.params.n_stored_steps, self.batch_dim, self.n_agents),
                device=self.device,
                dtype=torch.float32,
            )
        )
        self.observations.past_widths = CircularBuffer(
            torch.zeros(
                (self.params.n_stored_steps, self.batch_dim, self.n_agents),
                device=self.device,
                dtype=torch.float32,
            )
        )
        self.observations.past_steering = CircularBuffer(
            torch.zeros(
                (self.params.n_stored_steps, self.batch_dim, self.n_agents),
                device=self.device,
                dtype=torch.float32,
            )
        )

    """
    ------------------------------- STATE MANAGEMENT -------------------------------
    """

    def update_state(self, agent_states: List[AgentState]):
        positions_global = torch.stack([s.pos for s in agent_states], dim=0).transpose(
            0, 1
        )
        velocities_global = torch.stack([s.vel for s in agent_states], dim=0).transpose(
            0, 1
        )
        self._latest_positions_global = positions_global
        self._latest_velocities_global = velocities_global
        self._latest_observed_agent_render_diagnostics.clear()
        self._latest_observations = [None] * self.n_agents
        rotations_global = (
            torch.stack([s.rot for s in agent_states], dim=0)
            .transpose(0, 1)
            .squeeze(-1)
        )
        steering_global = angle_eliminate_two_pi(
            torch.stack([s.steering for s in agent_states], dim=0)
            .transpose(0, 1)
            .squeeze(-1)
        )

        lengths_global = torch.tensor(
            self.world_state.agent_lengths[: len(agent_states)],
            device=self.device,
            dtype=torch.float32,
        ).repeat(self.batch_dim, 1)

        widths_global = torch.tensor(
            self.world_state.agent_widths[: len(agent_states)],
            device=self.device,
            dtype=torch.float32,
        ).repeat(self.batch_dim, 1)

        # Add new observation & normalize
        self.observations.past_distance_to_agents.add(
            self.world_state.distances.agents / self.normalizers.distance_lanelet
        )
        self.observations.past_distance_to_ref_path.add(
            self.world_state.distances.ref_paths / self.normalizers.distance_lanelet
        )
        self.observations.past_distance_to_left_boundary.add(
            torch.min(self.world_state.distances.left_boundaries, dim=-1)[0]
            / self.normalizers.distance_lanelet
        )
        self.observations.past_distance_to_right_boundary.add(
            torch.min(self.world_state.distances.right_boundaries, dim=-1)[0]
            / self.normalizers.distance_lanelet
        )
        self.observations.past_distance_to_boundaries.add(
            self.world_state.distances.boundaries / self.normalizers.distance_lanelet
        )
        self.observations.past_lengths.add(
            lengths_global / self.normalizers.distance_agent
        )  # Use distance to agents as the normalizer
        self.observations.past_widths.add(
            widths_global / self.normalizers.distance_agent
        )
        self.observations.past_steering.add(steering_global / self.normalizers.rot)

        if self.params.is_ego_view:
            pos_i_others = _transform_from_global_to_all_local_coordinates(
                ego_positions=positions_global,
                ego_rotations=rotations_global,
                target_positions=positions_global.unsqueeze(-2),
            ).squeeze(-2)
            rot_i_others = angle_eliminate_two_pi(
                rotations_global.unsqueeze(1) - rotations_global.unsqueeze(2)
            )
            velocity_magnitudes = torch.norm(velocities_global, dim=-1).unsqueeze(1)
            vel_i_others = torch.stack(
                (
                    velocity_magnitudes * torch.cos(rot_i_others),
                    velocity_magnitudes * torch.sin(rot_i_others),
                ),
                dim=-1,
            )
            ref_i_others = _transform_from_global_to_all_local_coordinates(
                ego_positions=positions_global,
                ego_rotations=rotations_global,
                target_positions=self.world_state.ref_paths_agent_related.short_term,
            )
            if self.params.is_observe_boundary_points:
                l_b_i_others = _transform_from_global_to_all_local_coordinates(
                    ego_positions=positions_global,
                    ego_rotations=rotations_global,
                    target_positions=self.world_state.ref_paths_agent_related.nearing_points_left_boundary,
                )
                r_b_i_others = _transform_from_global_to_all_local_coordinates(
                    ego_positions=positions_global,
                    ego_rotations=rotations_global,
                    target_positions=self.world_state.ref_paths_agent_related.nearing_points_right_boundary,
                )
            ver_i_others = _transform_from_global_to_all_local_coordinates(
                ego_positions=positions_global,
                ego_rotations=rotations_global,
                target_positions=self.world_state.vertices[:, :, 0:4, :],
            )

            # Add new observations & normalize
            self.observations.past_pos.add(
                pos_i_others
                / (
                    self.normalizers.pos
                    if self.params.is_ego_view
                    else self.normalizers.pos_world
                )
            )
            self.observations.past_rot.add(rot_i_others / self.normalizers.rot)
            self.observations.past_vel.add(vel_i_others / self.normalizers.v)
            self.observations.past_short_term_ref_points.add(
                ref_i_others
                / (
                    self.normalizers.pos
                    if self.params.is_ego_view
                    else self.normalizers.pos_world
                )
            )
            if self.params.is_observe_boundary_points:
                self.observations.past_left_boundary.add(
                    l_b_i_others
                    / (
                        self.normalizers.pos
                        if self.params.is_ego_view
                        else self.normalizers.pos_world
                    )
                )
                self.observations.past_right_boundary.add(
                    r_b_i_others
                    / (
                        self.normalizers.pos
                        if self.params.is_ego_view
                        else self.normalizers.pos_world
                    )
                )
            self.observations.past_vertices.add(
                ver_i_others
                / (
                    self.normalizers.pos
                    if self.params.is_ego_view
                    else self.normalizers.pos_world
                )
            )

        else:  # Global coordinate system
            # Store new observations
            self.observations.past_pos.add(
                positions_global
                / (
                    self.normalizers.pos
                    if self.params.is_ego_view
                    else self.normalizers.pos_world
                )
            )
            self.observations.past_vel.add(velocities_global / self.normalizers.v)
            self.observations.past_rot.add(
                angle_eliminate_two_pi(rotations_global[:]) / self.normalizers.rot
            )
            self.observations.past_vertices.add(
                self.world_state.vertices[:, :, 0:4, :]
                / (
                    self.normalizers.pos
                    if self.params.is_ego_view
                    else self.normalizers.pos_world
                )
            )
            self.observations.past_short_term_ref_points.add(
                self.world_state.ref_paths_agent_related.short_term[:]
                / (
                    self.normalizers.pos
                    if self.params.is_ego_view
                    else self.normalizers.pos_world
                )
            )
            if self.params.is_observe_boundary_points:
                self.observations.past_left_boundary.add(
                    self.world_state.ref_paths_agent_related.nearing_points_left_boundary
                    / (
                        self.normalizers.pos
                        if self.params.is_ego_view
                        else self.normalizers.pos_world
                    )
                )
                self.observations.past_right_boundary.add(
                    self.world_state.ref_paths_agent_related.nearing_points_right_boundary
                    / (
                        self.normalizers.pos
                        if self.params.is_ego_view
                        else self.normalizers.pos_world
                    )
                )

        self._precompute_all_observations_if_possible()

    def _precompute_all_observations_if_possible(self):
        """Build deterministic ego-view observations for all agents in one batch.

        The regular per-agent path remains the fallback for noisy, bird-view, or
        dropout-enabled observations.  The batched path mirrors the partial
        ego-view feature ordering while avoiding one nearest-neighbor/masking
        pass per ego agent.
        """
        if not self._is_observation_cache_enabled():
            return
        if not self.params.is_ego_view or not self.params.is_partial_observation:
            return

        batch_dim = self.batch_dim
        n_agents = self.n_agents
        n_neighbor_slots = int(self.params.n_nearing_agents_observed)
        n_real_neighbors = min(n_neighbor_slots, n_agents - 1)
        distances = self.world_state.distances.agents

        nearing_agents_distances = torch.full(
            (batch_dim, n_agents, n_neighbor_slots),
            float("inf"),
            device=self.device,
            dtype=torch.float32,
        )
        neighbor_indices = torch.zeros(
            (batch_dim, n_agents, n_neighbor_slots),
            device=self.device,
            dtype=torch.int32,
        )
        if n_real_neighbors > 0:
            real_distances, real_indices = torch.topk(
                distances,
                k=n_real_neighbors,
                dim=-1,
                largest=False,
            )
            nearing_agents_distances[:, :, :n_real_neighbors] = real_distances
            neighbor_indices[:, :, :n_real_neighbors] = real_indices.to(
                dtype=neighbor_indices.dtype
            )
        self.observations.nearing_agents_indices[:] = neighbor_indices

        slot_indices = torch.arange(
            n_neighbor_slots, device=self.device, dtype=torch.long
        ).view(1, 1, -1)
        padded_agents_mask = (slot_indices >= n_real_neighbors).expand(
            batch_dim, n_agents, n_neighbor_slots
        )
        distance_mask = torch.zeros_like(padded_agents_mask)
        velocity_mask = torch.zeros_like(padded_agents_mask)
        dropout_mask = torch.zeros_like(padded_agents_mask)
        masked_agents = padded_agents_mask.clone()
        neighbor_indices_long = neighbor_indices.to(dtype=torch.long)

        if self._on_hold_agents_mask is None:
            on_hold_mask = torch.zeros_like(masked_agents)
        else:
            on_hold_source = self._on_hold_agents_mask.unsqueeze(1).expand(
                batch_dim, n_agents, n_agents
            )
            on_hold_mask = torch.gather(
                on_hold_source, dim=2, index=neighbor_indices_long
            )
            ego_indices = torch.arange(
                n_agents, device=self.device, dtype=neighbor_indices.dtype
            ).view(1, n_agents, 1)
            on_hold_mask &= neighbor_indices != ego_indices
        masked_agents |= on_hold_mask

        if self.params.is_apply_distance_mask:
            distance_mask = nearing_agents_distances >= self.thresholds.rl_obs_range
            masked_agents |= distance_mask

        if self.params.is_apply_velocity_mask:
            positions = self._latest_positions_global
            velocities = self._latest_velocities_global
            target_positions = positions.unsqueeze(1).expand(
                batch_dim, n_agents, n_agents, 2
            )
            target_velocities = velocities.unsqueeze(1).expand(
                batch_dim, n_agents, n_agents, 2
            )
            neighbor_positions = torch.gather(
                target_positions,
                dim=2,
                index=neighbor_indices_long.unsqueeze(-1).expand(
                    batch_dim, n_agents, n_neighbor_slots, 2
                ),
            )
            neighbor_velocities = torch.gather(
                target_velocities,
                dim=2,
                index=neighbor_indices_long.unsqueeze(-1).expand(
                    batch_dim, n_agents, n_neighbor_slots, 2
                ),
            )
            velocity_mask = self._compute_velocity_based_mask(
                ego_positions=positions.reshape(batch_dim * n_agents, 2),
                neighbor_positions=neighbor_positions.reshape(
                    batch_dim * n_agents, n_neighbor_slots, 2
                ),
                ego_velocities=velocities.reshape(batch_dim * n_agents, 2),
                neighbor_velocities=neighbor_velocities.reshape(
                    batch_dim * n_agents, n_neighbor_slots, 2
                ),
                velocity_mask_margin=self.params.velocity_mask_margin,
            ).reshape(batch_dim, n_agents, n_neighbor_slots)
            masked_agents |= velocity_mask

        self._latest_neighbor_valid = ~masked_agents

        # Skip storing render diagnostic dicts if fast inference is on (only used by GUI visualization).
        if not self.params.is_fast_inference_mode:
            for agent_index in range(n_agents):
                self._latest_observed_agent_render_diagnostics[agent_index] = {
                    "neighbor_indices": neighbor_indices[:, agent_index]
                    .detach()
                    .clone(),
                    "padded_mask": padded_agents_mask[:, agent_index].detach().clone(),
                    "distance_mask": distance_mask[:, agent_index].detach().clone(),
                    "velocity_mask": velocity_mask[:, agent_index].detach().clone(),
                    "dropout_mask": dropout_mask[:, agent_index].detach().clone(),
                    "final_mask": masked_agents[:, agent_index].detach().clone(),
                }

        def gather_agent_axis(values):
            index = neighbor_indices_long
            for _ in range(values.ndim - 3):
                index = index.unsqueeze(-1)
            return torch.gather(
                values,
                dim=2,
                index=index.expand(
                    batch_dim,
                    n_agents,
                    n_neighbor_slots,
                    *values.shape[3:],
                ),
            )

        latest_pos = self.observations.past_pos.get_latest()
        latest_rot = self.observations.past_rot.get_latest()
        latest_vel = self.observations.past_vel.get_latest()
        latest_ref = self.observations.past_short_term_ref_points.get_latest()
        latest_vertices = self.observations.past_vertices.get_latest()
        latest_distances = self.observations.past_distance_to_agents.get_latest()
        latest_lengths = self.observations.past_lengths.get_latest()
        latest_widths = self.observations.past_widths.get_latest()
        latest_steering = self.observations.past_steering.get_latest()

        obs_pos = gather_agent_axis(latest_pos)
        obs_rot = gather_agent_axis(latest_rot)
        obs_vel = gather_agent_axis(latest_vel)
        obs_ref = gather_agent_axis(latest_ref)
        obs_vertices = gather_agent_axis(latest_vertices)
        obs_distances = gather_agent_axis(latest_distances)
        obs_lengths = torch.gather(
            latest_lengths.unsqueeze(1).expand(batch_dim, n_agents, n_agents),
            dim=2,
            index=neighbor_indices_long,
        )
        obs_widths = torch.gather(
            latest_widths.unsqueeze(1).expand(batch_dim, n_agents, n_agents),
            dim=2,
            index=neighbor_indices_long,
        )
        obs_steering = torch.gather(
            latest_steering.unsqueeze(1).expand(batch_dim, n_agents, n_agents),
            dim=2,
            index=neighbor_indices_long,
        )

        obs_pos[masked_agents] = self.constants.mask_zero
        obs_rot[masked_agents] = self.constants.mask_zero
        obs_vel[masked_agents] = self.constants.mask_zero
        obs_ref[masked_agents] = self.constants.mask_zero
        obs_vertices[masked_agents] = self.constants.mask_zero
        obs_distances[masked_agents] = self.constants.mask_zero
        obs_lengths[masked_agents] = self.constants.mask_zero
        obs_widths[masked_agents] = self.constants.mask_zero
        obs_steering[masked_agents] = self.constants.mask_zero

        obs_pos_flat = obs_pos.reshape(batch_dim, n_agents, n_neighbor_slots, -1)
        obs_rot_flat = obs_rot.reshape(batch_dim, n_agents, n_neighbor_slots, -1)
        obs_vel_flat = obs_vel.reshape(batch_dim, n_agents, n_neighbor_slots, -1)
        obs_ref_flat = obs_ref.reshape(batch_dim, n_agents, n_neighbor_slots, -1)
        obs_vertices_flat = obs_vertices.reshape(
            batch_dim, n_agents, n_neighbor_slots, -1
        )
        obs_distances_flat = obs_distances.reshape(
            batch_dim, n_agents, n_neighbor_slots, -1
        )
        obs_lengths_flat = obs_lengths.unsqueeze(-1)
        obs_widths_flat = obs_widths.unsqueeze(-1)
        obs_steering_flat = obs_steering.unsqueeze(-1)

        neighbor_valid = (~masked_agents).to(dtype=torch.float32)
        neighbor_valid_flat = (
            neighbor_valid.unsqueeze(-1)
            if self.params.is_apply_distance_mask or self.params.is_apply_velocity_mask
            else None
        )
        obs_others_list = [
            (
                obs_vertices_flat
                if self.params.is_observe_vertices
                else torch.cat(
                    [
                        obs_pos_flat,
                        obs_rot_flat,
                        obs_lengths_flat,
                        obs_widths_flat,
                    ],
                    dim=-1,
                )
            ),
            obs_vel_flat,
            obs_steering_flat if self.params.is_obs_steering else None,
            obs_distances_flat if self.params.is_observe_distance_to_agents else None,
            obs_ref_flat if self.params.is_observe_ref_path_other_agents else None,
            neighbor_valid_flat,
        ]
        obs_others_list = [o for o in obs_others_list if o is not None]
        obs_other_agents = torch.cat(obs_others_list, dim=-1).reshape(
            batch_dim, n_agents, -1
        )

        agent_indices = torch.arange(n_agents, device=self.device)
        self_parts = [
            latest_vel[:, agent_indices, agent_indices, 0].unsqueeze(-1),
        ]
        if self.params.is_obs_steering:
            self_parts.append(latest_steering.unsqueeze(-1))
        self_parts.append(
            latest_ref[:, agent_indices, agent_indices].reshape(batch_dim, n_agents, -1)
        )
        if self.params.is_observe_distance_to_center_line:
            self_parts.append(
                self.observations.past_distance_to_ref_path.get_latest().unsqueeze(-1)
            )
        if self.params.is_observe_boundary_points:
            self_parts.append(
                self.observations.past_left_boundary.get_latest()[
                    :, agent_indices, agent_indices
                ].reshape(batch_dim, n_agents, -1)
            )
        if self.params.is_observe_distance_to_boundaries:
            self_parts.append(
                self.observations.past_distance_to_left_boundary.get_latest().unsqueeze(
                    -1
                )
            )
        if self.params.is_observe_boundary_points:
            self_parts.append(
                self.observations.past_right_boundary.get_latest()[
                    :, agent_indices, agent_indices
                ].reshape(batch_dim, n_agents, -1)
            )
        if self.params.is_observe_distance_to_boundaries:
            self_parts.append(
                self.observations.past_distance_to_right_boundary.get_latest().unsqueeze(
                    -1
                )
            )
        obs_self = torch.cat(self_parts, dim=-1)
        all_observations = torch.cat((obs_self, obs_other_agents), dim=-1)
        self._latest_observations = [
            all_observations[:, agent_index] for agent_index in range(n_agents)
        ]
        self._latest_other_agents_noise_mask = None

    """
    ------------------------------- OBSERVATIONS -------------------------------
    """

    def _is_observation_cache_enabled(self) -> bool:
        """Return whether observation reuse is deterministic for this rollout."""
        if self.params.is_obs_noise:
            return False
        if self.params.is_testing_mode:
            return True
        return self.params.visible_agent_dropout_probability <= 0.0

    def get_observation(self, agent_index):
        if (
            self._is_observation_cache_enabled()
            and self._latest_observations[agent_index] is not None
        ):
            return self._latest_observations[agent_index]

        # Observation of other agents
        obs_other_agents = self._observe_other_agents(agent_index)

        obs_self = self._observe_self(agent_index)

        obs_self.append(obs_other_agents)  # Append the observations of other agents

        obs_all = [o for o in obs_self if o is not None]  # Filter out None values

        obs = torch.hstack(obs_all)  # Convert from list to tensor

        if self.params.is_obs_noise:
            obs_noise_mask = torch.ones_like(
                obs, device=self.device, dtype=torch.float32
            )
            obs_other_agents_noise_mask = getattr(
                self, "_latest_other_agents_noise_mask", None
            )
            if obs_other_agents_noise_mask is not None:
                other_obs_start = obs.shape[1] - obs_other_agents_noise_mask.shape[1]
                other_obs_end = other_obs_start + obs_other_agents_noise_mask.shape[1]
                obs_noise_mask[
                    :, other_obs_start:other_obs_end
                ] = obs_other_agents_noise_mask

            # Add sensor noise if required
            obs = obs + obs_noise_mask * (
                self.observations.obs_noise_level
                * torch.rand_like(obs, device=self.device, dtype=torch.float32)
            )

        if self._is_observation_cache_enabled():
            self._latest_observations[agent_index] = obs
        return obs

    @staticmethod
    def _compute_velocity_based_mask(
        ego_positions: torch.Tensor,
        neighbor_positions: torch.Tensor,
        ego_velocities: torch.Tensor,
        neighbor_velocities: torch.Tensor,
        velocity_mask_margin: float = 0.5,
    ) -> torch.Tensor:
        velocity_mask_horizon = 2.0
        T = velocity_mask_horizon
        eps = 1e-9

        r = neighbor_positions - ego_positions.unsqueeze(1)
        u = neighbor_velocities - ego_velocities.unsqueeze(1)
        ru = torch.sum(r * u, dim=-1)
        uu = torch.sum(u * u, dim=-1)
        uu_safe = torch.clamp(uu, min=eps)
        t_star = torch.clamp(-ru / uu_safe, min=0.0, max=T)
        closest = r + u * t_star.unsqueeze(-1)
        d_min_sq = torch.sum(closest * closest, dim=-1)

        disk_radius = torch.as_tensor(
            THRESHOLD["disk_radius"],
            device=neighbor_positions.device,
            dtype=neighbor_positions.dtype,
        )
        margin = velocity_mask_margin * disk_radius
        rho = disk_radius + disk_radius + margin

        return d_min_sq > rho**2

    def set_on_hold_agents_mask(self, on_hold_agents_mask):
        """Set the batched agents that should be hidden from observations."""
        current_mask = getattr(self, "_on_hold_agents_mask", None)
        if on_hold_agents_mask is None:
            if current_mask is not None:
                self._latest_observations = [None] * self.n_agents
                if hasattr(self, "_latest_observed_agent_render_diagnostics"):
                    self._latest_observed_agent_render_diagnostics.clear()
            self._on_hold_agents_mask = None
            return

        expected_shape = (self.batch_dim, self.n_agents)
        if tuple(on_hold_agents_mask.shape) != expected_shape:
            raise ValueError(
                "On-hold agent mask must have shape "
                f"{expected_shape}, got {tuple(on_hold_agents_mask.shape)}."
            )
        new_mask = (
            on_hold_agents_mask.to(device=self.device, dtype=torch.bool)
            .detach()
            .clone()
        )
        if current_mask is None or not torch.equal(current_mask, new_mask):
            self._latest_observations = [None] * self.n_agents
            if hasattr(self, "_latest_observed_agent_render_diagnostics"):
                self._latest_observed_agent_render_diagnostics.clear()
        self._on_hold_agents_mask = new_mask

    def get_observed_agent_render_diagnostics(self, agent_index: int):
        return self._latest_observed_agent_render_diagnostics.get(agent_index)

    def get_neighbor_valid(self):
        return self._latest_neighbor_valid

    def _observe_other_agents(self, agent_index):
        """Observe surrounding agents."""
        ##################################################
        ## Observation of other agents
        ##################################################
        if self.params.is_partial_observation:
            n_neighbor_slots = self.params.n_nearing_agents_observed
            n_real_neighbors = min(n_neighbor_slots, self.n_agents - 1)
            nearing_agents_distances = torch.full(
                (self.batch_dim, n_neighbor_slots),
                float("inf"),
                device=self.device,
                dtype=torch.float32,
            )
            self.observations.nearing_agents_indices[:, agent_index].zero_()

            if n_real_neighbors > 0:
                # Fill available neighbor slots with real nearest agents and leave the
                # remaining fixed-capacity slots padded as invalid.
                (
                    real_nearing_agents_distances,
                    real_nearing_agents_indices,
                ) = torch.topk(
                    self.world_state.distances.agents[:, agent_index],
                    k=n_real_neighbors,
                    largest=False,
                )
                nearing_agents_distances[
                    :, :n_real_neighbors
                ] = real_nearing_agents_distances
                self.observations.nearing_agents_indices[
                    :, agent_index, :n_real_neighbors
                ] = real_nearing_agents_indices

            padded_agents = (
                torch.arange(n_neighbor_slots, device=self.device).unsqueeze(0)
                >= n_real_neighbors
            )
            padded_agents_mask = padded_agents.expand(self.batch_dim, n_neighbor_slots)
            distance_mask = torch.zeros_like(padded_agents_mask)
            velocity_mask = torch.zeros_like(padded_agents_mask)
            dropout_mask = torch.zeros_like(padded_agents_mask)

            masked_agents = padded_agents_mask.clone()
            if self._on_hold_agents_mask is None:
                on_hold_mask = torch.zeros_like(masked_agents)
            else:
                neighbor_indices = self.observations.nearing_agents_indices[
                    :, agent_index
                ].long()
                on_hold_mask = self._on_hold_agents_mask.gather(
                    dim=1, index=neighbor_indices
                )
                on_hold_mask &= neighbor_indices != agent_index
            masked_agents = masked_agents | on_hold_mask

            if self.params.is_apply_distance_mask:
                distance_mask = nearing_agents_distances >= self.thresholds.rl_obs_range
                masked_agents = masked_agents | distance_mask

                dropout_probability = (
                    0.0
                    if self.params.is_testing_mode
                    else self.params.visible_agent_dropout_probability
                )
                if dropout_probability > 0:
                    dropout_probability = min(max(dropout_probability, 0.0), 1.0)
                    dropout_agents = (
                        torch.rand(
                            masked_agents.shape,
                            device=self.device,
                            dtype=torch.float32,
                        )
                        < dropout_probability
                    )
                    dropout_mask = dropout_agents & ~masked_agents
                    masked_agents = masked_agents | dropout_mask

            if self.params.is_apply_velocity_mask:
                ego_positions = self._latest_positions_global[:, agent_index]
                ego_velocities = self._latest_velocities_global[:, agent_index]
                neighbor_indices = self.observations.nearing_agents_indices[
                    :, agent_index
                ]
                neighbor_positions = self._latest_positions_global[
                    self.constants.env_idx_broadcasting, neighbor_indices
                ]
                neighbor_velocities = self._latest_velocities_global[
                    self.constants.env_idx_broadcasting, neighbor_indices
                ]
                velocity_mask = self._compute_velocity_based_mask(
                    ego_positions=ego_positions,
                    neighbor_positions=neighbor_positions,
                    ego_velocities=ego_velocities,
                    neighbor_velocities=neighbor_velocities,
                    velocity_mask_margin=self.params.velocity_mask_margin,
                )
                masked_agents = masked_agents | velocity_mask

            self._latest_neighbor_valid[:, agent_index] = ~masked_agents

            # Skip storing render diagnostic dicts if fast inference is on (only used by GUI visualization).
            if not self.params.is_fast_inference_mode:
                self._latest_observed_agent_render_diagnostics[agent_index] = {
                    "neighbor_indices": self.observations.nearing_agents_indices[
                        :, agent_index
                    ]
                    .detach()
                    .clone(),
                    "padded_mask": padded_agents_mask.detach().clone(),
                    "distance_mask": distance_mask.detach().clone(),
                    "velocity_mask": velocity_mask.detach().clone(),
                    "dropout_mask": dropout_mask.detach().clone(),
                    "final_mask": masked_agents.detach().clone(),
                }
            if agent_index == 0:
                env_index_debug = 0
                neighbor_indices_debug = self.observations.nearing_agents_indices[
                    env_index_debug, agent_index
                ]
                observed_agents = (
                    neighbor_indices_debug[~masked_agents[env_index_debug]]
                    .detach()
                    .cpu()
                    .tolist()
                )
                velocity_masked_agents = (
                    neighbor_indices_debug[
                        (~padded_agents_mask[env_index_debug])
                        & (~distance_mask[env_index_debug])
                        & velocity_mask[env_index_debug]
                    ]
                    .detach()
                    .cpu()
                    .tolist()
                )
                # print(
                #     "[OBS MASK] env 0 agent 0 observed agents "
                #     f"{observed_agents}; distance-visible velocity-masked agents "
                #     f"{velocity_masked_agents}"
                # )

            indexing_tuple_1 = (
                (self.constants.env_idx_broadcasting,)
                + ((agent_index,) if self.params.is_ego_view else ())
                + (self.observations.nearing_agents_indices[:, agent_index],)
            )

            # Positions of nearing agents
            obs_pos_other_agents = self.observations.past_pos.get_latest()[
                indexing_tuple_1
            ]  # [batch_size, n_nearing_agents, 2]
            obs_pos_other_agents[
                masked_agents
            ] = self.constants.mask_zero  # Position mask

            # Rotations of nearing agents
            obs_rot_other_agents = self.observations.past_rot.get_latest()[
                indexing_tuple_1
            ]  # [batch_size, n_nearing_agents]
            obs_rot_other_agents[
                masked_agents
            ] = self.constants.mask_zero  # Rotation mask

            # Lengths and widths of nearing agents
            obs_lengths_other_agents = self.observations.past_lengths.get_latest()[
                self.constants.env_idx_broadcasting,
                self.observations.nearing_agents_indices[:, agent_index],
            ]
            obs_widths_other_agents = self.observations.past_widths.get_latest()[
                self.constants.env_idx_broadcasting,
                self.observations.nearing_agents_indices[:, agent_index],
            ]
            obs_lengths_other_agents[masked_agents] = self.constants.mask_zero
            obs_widths_other_agents[masked_agents] = self.constants.mask_zero
            obs_steering_other_agents = self.observations.past_steering.get_latest()[
                self.constants.env_idx_broadcasting,
                self.observations.nearing_agents_indices[:, agent_index],
            ]
            obs_steering_other_agents[
                masked_agents
            ] = self.constants.mask_zero  # Steering mask

            # Velocities of nearing agents
            obs_vel_other_agents = self.observations.past_vel.get_latest()[
                indexing_tuple_1
            ]  # [batch_size, n_nearing_agents]
            obs_vel_other_agents[
                masked_agents
            ] = self.constants.mask_zero  # Velocity mask

            # Reference paths of nearing agents
            obs_ref_path_other_agents = (
                self.observations.past_short_term_ref_points.get_latest()[
                    indexing_tuple_1
                ]
            )  # [batch_size, n_nearing_agents, n_points_short_term, 2]
            obs_ref_path_other_agents[
                masked_agents
            ] = self.constants.mask_zero  # Reference-path mask

            # vertices of nearing agents
            obs_vertices_other_agents = self.observations.past_vertices.get_latest()[
                indexing_tuple_1
            ]  # [batch_size, n_nearing_agents, 4, 2]
            obs_vertices_other_agents[
                masked_agents
            ] = self.constants.mask_zero  # Vertices mask

            # Distances to nearing agents
            obs_distance_other_agents = (
                self.observations.past_distance_to_agents.get_latest()[
                    self.constants.env_idx_broadcasting,
                    agent_index,
                    self.observations.nearing_agents_indices[:, agent_index],
                ]
            )  # [batch_size, n_nearing_agents]
            obs_distance_other_agents[
                masked_agents
            ] = self.constants.mask_zero  # Distance mask

            neighbor_valid = (~masked_agents).to(dtype=torch.float32)

        else:
            indexing_tuple_2 = (self.constants.env_idx_broadcasting.squeeze(-1),) + (
                (agent_index,) if self.params.is_ego_view else ()
            )
            masked_agents = torch.zeros(
                (self.batch_dim, self.n_agents),
                device=self.device,
                dtype=torch.bool,
            )
            if self._on_hold_agents_mask is not None:
                masked_agents = self._on_hold_agents_mask.clone()
                masked_agents[:, agent_index] = False
            neighbor_valid = None

            obs_pos_other_agents = self.observations.past_pos.get_latest()[
                indexing_tuple_2
            ]  # [batch_size, n_agents, 2]
            obs_rot_other_agents = self.observations.past_rot.get_latest()[
                indexing_tuple_2
            ]  # [batch_size, n_agents, (n_agents)]
            obs_vel_other_agents = self.observations.past_vel.get_latest()[
                indexing_tuple_2
            ]  # [batch_size, n_agents, 2]
            obs_ref_path_other_agents = (
                self.observations.past_short_term_ref_points.get_latest()[
                    indexing_tuple_2
                ]
            )  # [batch_size, n_agents, n_points_short_term, 2]
            obs_vertices_other_agents = self.observations.past_vertices.get_latest()[
                indexing_tuple_2
            ]  # [batch_size, n_agents, 4, 2]
            obs_distance_other_agents = (
                self.observations.past_distance_to_agents.get_latest()[indexing_tuple_2]
            )  # [batch_size, n_agents]
            obs_distance_other_agents[
                indexing_tuple_2
            ] = 0  # Reset self-self distance to zero
            obs_lengths_other_agents = self.observations.past_lengths.get_latest()[
                indexing_tuple_2
            ]
            obs_widths_other_agents = self.observations.past_widths.get_latest()[
                indexing_tuple_2
            ]
            obs_steering_other_agents = self.observations.past_steering.get_latest()[
                indexing_tuple_2
            ]

            obs_pos_other_agents[masked_agents] = self.constants.mask_zero
            obs_rot_other_agents[masked_agents] = self.constants.mask_zero
            obs_vel_other_agents[masked_agents] = self.constants.mask_zero
            obs_ref_path_other_agents[masked_agents] = self.constants.mask_zero
            obs_vertices_other_agents[masked_agents] = self.constants.mask_zero
            obs_distance_other_agents[masked_agents] = self.constants.mask_zero
            obs_lengths_other_agents[masked_agents] = self.constants.mask_zero
            obs_widths_other_agents[masked_agents] = self.constants.mask_zero
            obs_steering_other_agents[masked_agents] = self.constants.mask_zero

        # Flatten the last dimensions to combine all features into a single dimension
        obs_pos_other_agents_flat = obs_pos_other_agents.reshape(
            self.batch_dim, self.observations.n_nearing_agents, -1
        )
        obs_rot_other_agents_flat = obs_rot_other_agents.reshape(
            self.batch_dim, self.observations.n_nearing_agents, -1
        )
        obs_vel_other_agents_flat = obs_vel_other_agents.reshape(
            self.batch_dim, self.observations.n_nearing_agents, -1
        )
        obs_ref_path_other_agents_flat = obs_ref_path_other_agents.reshape(
            self.batch_dim, self.observations.n_nearing_agents, -1
        )
        obs_vertices_other_agents_flat = obs_vertices_other_agents.reshape(
            self.batch_dim, self.observations.n_nearing_agents, -1
        )
        obs_distance_other_agents_flat = obs_distance_other_agents.reshape(
            self.batch_dim, self.observations.n_nearing_agents, -1
        )
        obs_lengths_other_agents_flat = obs_lengths_other_agents.reshape(
            self.batch_dim, self.observations.n_nearing_agents, -1
        )
        obs_widths_other_agents_flat = obs_widths_other_agents.reshape(
            self.batch_dim, self.observations.n_nearing_agents, -1
        )
        obs_steering_other_agents_flat = obs_steering_other_agents.reshape(
            self.batch_dim, self.observations.n_nearing_agents, -1
        )
        neighbor_valid_flat = (
            neighbor_valid.reshape(
                self.batch_dim, self.observations.n_nearing_agents, -1
            )
            if self.params.is_partial_observation
            and (
                self.params.is_apply_distance_mask or self.params.is_apply_velocity_mask
            )
            else None
        )
        # Observation of other agents
        obs_others_list = [
            (
                obs_vertices_other_agents_flat
                if self.params.is_observe_vertices
                else torch.cat(  # [other] vertices
                    [
                        obs_pos_other_agents_flat,  # [others] positions
                        obs_rot_other_agents_flat,  # [others] rotations
                        obs_lengths_other_agents_flat,  # [others] lengths
                        obs_widths_other_agents_flat,  # [others] widths
                    ],
                    dim=-1,
                )
            ),
            obs_vel_other_agents_flat,  # [others] velocities
            obs_steering_other_agents_flat
            if self.params.is_obs_steering
            else None,  # [others] steering angles
            (
                obs_distance_other_agents_flat
                if self.params.is_observe_distance_to_agents
                else None
            ),  # [others] mutual distances
            (
                obs_ref_path_other_agents_flat
                if self.params.is_observe_ref_path_other_agents
                else None
            ),  # [others] reference paths
            neighbor_valid_flat,  # [others] hard-mask validity
        ]
        obs_others_list = [
            o for o in obs_others_list if o is not None
        ]  # Filter out None values
        obs_others_noise_mask_list = []
        for o in obs_others_list:
            noise_mask = torch.ones_like(o, device=self.device, dtype=torch.float32)
            if o is not neighbor_valid_flat:
                noise_mask[masked_agents] = 0.0
            if o is neighbor_valid_flat:
                noise_mask.zero_()
            obs_others_noise_mask_list.append(noise_mask)
        obs_other_agents = torch.cat(obs_others_list, dim=-1).reshape(
            self.batch_dim, -1
        )  # [batch_size, -1]
        self._latest_other_agents_noise_mask = torch.cat(
            obs_others_noise_mask_list, dim=-1
        ).reshape(self.batch_dim, -1)

        return obs_other_agents

    def _observe_self(self, agent_index):
        """Observe the given agent itself."""
        indexing_tuple_3 = (
            (self.constants.env_idx_broadcasting,)
            + (agent_index,)
            + ((agent_index,) if self.params.is_ego_view else ())
        )
        indexing_tuple_vel = (
            (self.constants.env_idx_broadcasting,)
            + (agent_index,)
            + ((agent_index, 0) if self.params.is_ego_view else ())
        )  # In local coordinate system, only the first component is interesting, as the second is always 0
        # All observations
        obs_self = [
            (
                None
                if self.params.is_ego_view
                else self.observations.past_pos.get_latest()[indexing_tuple_3].reshape(
                    self.batch_dim, -1
                )
            ),  # [own] position,
            (
                None
                if self.params.is_ego_view
                else self.observations.past_rot.get_latest()[indexing_tuple_3].reshape(
                    self.batch_dim, -1
                )
            ),  # [own] rotation,
            self.observations.past_vel.get_latest()[indexing_tuple_vel].reshape(
                self.batch_dim, -1
            ),  # [own] velocity
            self.observations.past_steering.get_latest()[:, agent_index].reshape(
                self.batch_dim, -1
            )
            if self.params.is_obs_steering
            else None,  # [own] steering angle
            self.observations.past_short_term_ref_points.get_latest()[
                indexing_tuple_3
            ].reshape(
                self.batch_dim, -1
            ),  # [own] short-term reference path
            (
                self.observations.past_distance_to_ref_path.get_latest()[
                    :, agent_index
                ].reshape(self.batch_dim, -1)
                if self.params.is_observe_distance_to_center_line
                else None
            ),  # [own] distances to reference paths
            self.observations.past_left_boundary.get_latest()[indexing_tuple_3].reshape(
                self.batch_dim, -1
            )
            if self.params.is_observe_boundary_points
            else None,  # [own] left boundary points
            (
                self.observations.past_distance_to_left_boundary.get_latest()[
                    :, agent_index
                ].reshape(self.batch_dim, -1)
                if self.params.is_observe_distance_to_boundaries
                else None
            ),  # [own] distance to left boundary
            self.observations.past_right_boundary.get_latest()[
                indexing_tuple_3
            ].reshape(self.batch_dim, -1)
            if self.params.is_observe_boundary_points
            else None,  # [own] right boundary points
            (
                self.observations.past_distance_to_right_boundary.get_latest()[
                    :, agent_index
                ].reshape(self.batch_dim, -1)
                if self.params.is_observe_distance_to_boundaries
                else None
            ),  # [own] distance to right boundary
        ]

        return obs_self


# todo maybe put into own file
class RoadTrafficObservationProviderSimulation(ObservationProviderRT):
    def __init__(
        self,
        params: ObservationProviderParametersRT,
        constants: Constants,
        normalizers: Normalizers,
        thresholds: Thresholds,
        map: MapManager,
        world_state: WorldStateRTSimulation,
    ):
        super().__init__(params, constants, normalizers, thresholds, map, world_state)

        self.world_state = world_state

    def update_state(self, world: WorldCustom):

        super().update_state(self.world_state.get_agent_state_list())

        # Add new observation - actions & normalize
        if world.agents[0].action.u is None:
            self.observations.past_action_vel.add(self.constants.empty_action_vel)
            self.observations.past_action_steering.add(
                self.constants.empty_action_steering
            )
        else:
            self.observations.past_action_vel.add(
                torch.stack([a.action.u[:, 0] for a in world.agents], dim=1)
                / self.normalizers.v
            )
            self.observations.past_action_steering.add(
                torch.stack([a.action.u[:, 1] for a in world.agents], dim=1)
                / self.normalizers.steering
            )


# todo maybe put into own file
class RoadTrafficObservationProviderReal(ObservationProviderRT):
    def update_state(self, agent_states: List[AgentState]):
        # update world state first
        self.world_state.update_vertices(agent_states)
        for a_i in range(len(agent_states)):
            self.world_state.update_distances(agent_states, a_i)
            self.world_state.update_ref_paths_agent_related(a_i)

        # then execute observation specific updates
        super().update_state(agent_states)
