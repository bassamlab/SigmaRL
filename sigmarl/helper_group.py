# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class InteractionPair:
    i: int
    j: int
    risk: float
    distance: float
    closing_speed: float


def _state_xy(agent_state) -> np.ndarray:
    return np.array(
        [float(getattr(agent_state, "x")), float(getattr(agent_state, "y"))],
        dtype=np.float64,
    )


def _state_velocity(agent_state) -> np.ndarray:
    speed = float(getattr(agent_state, "v", 0.0))
    heading = float(getattr(agent_state, "theta", getattr(agent_state, "heading", 0.0)))
    return np.array(
        [speed * math.cos(heading), speed * math.sin(heading)], dtype=np.float64
    )


def _previous_group_lookup(
    previous_groups: Optional[Sequence[Sequence[int]]],
    n_agents: int,
) -> List[int]:
    lookup = [-1] * n_agents
    if previous_groups is None:
        return lookup
    for g_idx, group in enumerate(previous_groups):
        for idx in group:
            if 0 <= int(idx) < n_agents:
                lookup[int(idx)] = g_idx
    return lookup


def interaction_pair_metrics(
    agent_states: Sequence,
    i: int,
    j: int,
    interaction_range: float,
    closing_speed_scale: float = 0.5,
) -> Tuple[float, float, float]:
    """
    Return (risk, distance, closing_speed) for a pair.

    The risk is intentionally cheap to compute: proximity dominates, while
    positive closing speed raises the priority of agents that are moving toward
    each other. The output is a relative ordering score, not a physical margin.
    """

    pi = _state_xy(agent_states[i])
    pj = _state_xy(agent_states[j])
    rel_pos = pj - pi
    distance = float(np.linalg.norm(rel_pos))
    safe_distance = max(distance, 1e-9)

    vi = _state_velocity(agent_states[i])
    vj = _state_velocity(agent_states[j])
    rel_vel = vj - vi
    closing_speed = float(-np.dot(rel_pos, rel_vel) / safe_distance)
    closing_speed = max(0.0, closing_speed)

    interaction_range = max(float(interaction_range), 1e-9)
    distance_score = max(0.0, (interaction_range - distance) / interaction_range)
    closing_score = min(1.0, closing_speed / max(float(closing_speed_scale), 1e-9))
    risk = distance_score * (1.0 + 0.5 * closing_score)

    return risk, distance, closing_speed


def _pairwise_interaction_scores(
    agent_states: Sequence,
    interaction_range: float,
    previous_groups: Optional[Sequence[Sequence[int]]] = None,
    hysteresis_bonus: float = 0.25,
    hysteresis_keep_range_factor: float = 1.25,
    closing_speed_scale: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, List[InteractionPair]]:
    n_agents = len(agent_states)
    scores = np.zeros((n_agents, n_agents), dtype=np.float64)
    distances = np.full((n_agents, n_agents), np.inf, dtype=np.float64)
    previous_lookup = _previous_group_lookup(previous_groups, n_agents)
    pairs: List[InteractionPair] = []

    keep_range = max(float(interaction_range), 1e-9) * float(
        hysteresis_keep_range_factor
    )
    for i in range(n_agents - 1):
        for j in range(i + 1, n_agents):
            risk, distance, closing_speed = interaction_pair_metrics(
                agent_states,
                i,
                j,
                interaction_range,
                closing_speed_scale=closing_speed_scale,
            )
            if (
                previous_lookup[i] >= 0
                and previous_lookup[i] == previous_lookup[j]
                and distance <= keep_range
            ):
                risk += float(hysteresis_bonus)

            scores[i, j] = risk
            scores[j, i] = risk
            distances[i, j] = distance
            distances[j, i] = distance
            pairs.append(
                InteractionPair(
                    i=i,
                    j=j,
                    risk=float(risk),
                    distance=float(distance),
                    closing_speed=float(closing_speed),
                )
            )

    return scores, distances, pairs


def _best_seed_pair(
    unassigned: set[int],
    pairs: Sequence[InteractionPair],
    risk_threshold: float,
) -> Optional[Tuple[int, int]]:
    for pair in pairs:
        if pair.risk <= risk_threshold:
            break
        if pair.i in unassigned and pair.j in unassigned:
            return pair.i, pair.j
    return None


def _best_single_seed(
    unassigned: set[int],
    scores: np.ndarray,
    distances: np.ndarray,
) -> int:
    def key(idx: int) -> Tuple[float, float, int]:
        others = [j for j in unassigned if j != idx]
        max_score = max((float(scores[idx, j]) for j in others), default=0.0)
        min_dist = min((float(distances[idx, j]) for j in others), default=0.0)
        return max_score, -min_dist, -idx

    return max(unassigned, key=key)


def _best_group_candidate(
    group: Sequence[int],
    unassigned: Iterable[int],
    scores: np.ndarray,
    distances: np.ndarray,
) -> Tuple[int, float]:
    best_idx = -1
    best_score = -np.inf
    best_distance = np.inf
    for idx in unassigned:
        affinity = float(sum(scores[idx, member] for member in group))
        nearest_distance = min(float(distances[idx, member]) for member in group)
        if (
            affinity > best_score
            or (math.isclose(affinity, best_score) and nearest_distance < best_distance)
            or (
                math.isclose(affinity, best_score)
                and math.isclose(nearest_distance, best_distance)
                and idx < best_idx
            )
        ):
            best_idx = int(idx)
            best_score = affinity
            best_distance = nearest_distance
    return best_idx, float(best_score)


def group_agents_by_interaction_risk(
    agent_states: Sequence,
    max_group_size: int,
    interaction_range: float,
    previous_groups: Optional[Sequence[Sequence[int]]] = None,
    hysteresis_bonus: float = 0.25,
    hysteresis_keep_range_factor: float = 1.25,
    risk_threshold: float = 0.0,
    closing_speed_scale: float = 0.5,
) -> List[List[int]]:
    """
    Build bounded-size groups that prioritize high-risk pairwise interactions.

    The function keeps exactly ceil(N / max_group_size) groups so it can be used
    as a drop-in replacement for the existing fixed-slot grouped QP cache.
    """

    if max_group_size <= 0:
        raise ValueError("max_group_size must be positive")

    n_agents = len(agent_states)
    if n_agents == 0:
        return []

    group_count = int(math.ceil(n_agents / max_group_size))
    scores, distances, pairs = _pairwise_interaction_scores(
        agent_states,
        interaction_range,
        previous_groups=previous_groups,
        hysteresis_bonus=hysteresis_bonus,
        hysteresis_keep_range_factor=hysteresis_keep_range_factor,
        closing_speed_scale=closing_speed_scale,
    )
    pairs = sorted(pairs, key=lambda p: (-p.risk, p.distance, p.i, p.j))

    unassigned = set(range(n_agents))
    groups: List[List[int]] = []

    for group_idx in range(group_count):
        if not unassigned:
            break

        remaining_group_slots = group_count - group_idx - 1
        min_take = max(1, len(unassigned) - remaining_group_slots * max_group_size)
        capacity = min(max_group_size, len(unassigned))

        seed_pair = _best_seed_pair(unassigned, pairs, risk_threshold)
        if seed_pair is not None and capacity >= 2 and min_take >= 2:
            group = [seed_pair[0], seed_pair[1]]
        elif seed_pair is not None and capacity >= 2:
            group = [seed_pair[0], seed_pair[1]]
        else:
            group = [_best_single_seed(unassigned, scores, distances)]

        for idx in group:
            unassigned.remove(idx)

        while len(group) < capacity and unassigned:
            candidate, candidate_score = _best_group_candidate(
                group,
                unassigned,
                scores,
                distances,
            )
            must_fill = len(group) < min_take
            if candidate < 0 or (not must_fill and candidate_score <= risk_threshold):
                break
            group.append(candidate)
            unassigned.remove(candidate)

        groups.append(sorted(group))

    if unassigned:
        for idx in sorted(unassigned):
            target = min(
                (g for g in groups if len(g) < max_group_size),
                key=lambda g: len(g),
            )
            target.append(idx)
            target.sort()

    while len(groups) < group_count:
        groups.append([])

    assert len(groups) == group_count
    assert all(len(g) <= max_group_size for g in groups)
    assert sorted(i for group in groups for i in group) == list(range(n_agents))

    return groups


def sort_neighbors_by_interaction_risk(
    agent_states: Sequence,
    agent_idx: int,
    candidate_indices: Sequence[int],
    interaction_range: float,
    closing_speed_scale: float = 0.5,
) -> List[int]:
    ranked = []
    for candidate_idx in candidate_indices:
        risk, distance, closing_speed = interaction_pair_metrics(
            agent_states,
            int(agent_idx),
            int(candidate_idx),
            interaction_range,
            closing_speed_scale=closing_speed_scale,
        )
        ranked.append(
            (int(candidate_idx), float(risk), float(distance), float(closing_speed))
        )
    ranked.sort(key=lambda item: (-item[1], item[2], item[0]))
    return [idx for idx, _, _, _ in ranked]
