# Copyright (c) 2024, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Scenario-independent VMAS environment construction."""

import importlib

from termcolor import cprint
from torchrl.envs import RewardSum
from torchrl.envs.libs.vmas import VmasEnv

from sigmarl.constants import NON_ROAD_SCENARIOS, ROAD_TRAFFIC_SCENARIOS, SCENARIOS
from sigmarl.helper_common import Parameters
from sigmarl.helper_training import TransformedEnvCustom
from sigmarl.render_backend import install_render_backend

_SCENARIO_MODULES = {
    name: (f"sigmarl.scenarios.{name}", "Scenario") for name in NON_ROAD_SCENARIOS
}
_SCENARIO_MODULES["goal_reaching_1"] = (
    "sigmarl.scenarios.goal_reaching",
    "GoalReaching",
)
_ROAD_SCENARIO = ("sigmarl.scenarios.road_traffic", "ScenarioRoadTraffic")
# The lab/mock backends swap only the physics layer (WorldLab + a bridge in
# place of WorldCustom), so they reuse the road-traffic scenario. The module is
# imported lazily, which keeps rclpy/cpm_lab_lab_msgs optional for pure-VMAS
# users ("mock" itself is ROS-free).
_LAB_ROAD_SCENARIO = (
    "sigmarl.lab.scenario_road_traffic_lab",
    "ScenarioRoadTrafficLab",
)
_SCENARIO_ALIASES = {name.lower(): name for name in SCENARIOS}

# Physics backends accepted by `Parameters.physics_backend`.
PHYSICS_BACKENDS = ("vmas", "lab", "mock")
# Backends whose physics is delegated outside the kinematic in-process world:
# "lab" drives the real fleet over ROS 2, "mock" an in-process loopback.
LAB_PHYSICS_BACKENDS = ("lab", "mock")


def canonical_scenario_name(name: str) -> str:
    """Return the registry spelling for a case-insensitive scenario name."""

    if not isinstance(name, str):
        raise ValueError("`scenario_type` must be a string.")
    try:
        return _SCENARIO_ALIASES[name.lower()]
    except KeyError as error:
        raise ValueError(
            f"Unknown scenario_type '{name}'. Available scenarios: {sorted(SCENARIOS)}"
        ) from error


def resolve_physics_backend(parameters: Parameters, override: str = None) -> str:
    """Return the physics backend to build for, `override` taking precedence.

    Callers that must never touch the lab (e.g. loading a policy for inference)
    pass `override="vmas"` so a lab-flavoured checkpoint cannot drag a bridge in.
    """

    backend = (
        override
        if override is not None
        else getattr(parameters, "physics_backend", "vmas")
    )
    if backend not in PHYSICS_BACKENDS:
        raise ValueError(
            f"Unknown physics_backend '{backend}'. "
            f"Available backends: {sorted(PHYSICS_BACKENDS)}"
        )
    return backend


def validate_lab_backend(parameters: Parameters, backend: str):
    """Fail fast on settings the lab/mock physics backends cannot honour."""

    assert parameters.num_vmas_envs == 1, (
        "physics_backend='lab'/'mock' drives one physical fleet; "
        "set num_vmas_envs = 1."
    )
    assert (
        abs(parameters.dt - parameters.lab_period_ms / 1000.0) < 1e-9
    ), "physics_backend='lab'/'mock' requires dt == lab_period_ms / 1000."
    assert (
        parameters.device == "cpu"
    ), "The lab bridge exchanges CPU tensors; set device = 'cpu'."
    if backend == "lab":
        assert (
            parameters.lab_vehicle_ids
            and len(parameters.lab_vehicle_ids) == parameters.n_agents
        ), (
            "physics_backend='lab' requires lab_vehicle_ids with one "
            "vehicle ID per agent."
        )
    if getattr(parameters, "lab_manual_reset", False):
        # Fail fast here as well as in the scenario: these settings impose
        # poses on the world and would overwrite the operator's hand placement.
        assert (
            parameters.predefined_ref_path_idx is None
        ), "lab_manual_reset=True is incompatible with predefined_ref_path_idx."
        assert not parameters.is_challenging_initial_state_buffer, (
            "lab_manual_reset=True is incompatible with "
            "is_challenging_initial_state_buffer."
        )
        cprint(
            "[INFO] Manual lab reset is ON. EVERY reset stops the fleet "
            "and waits for you to place the vehicles by hand and press "
            "ENTER. Startup performs a few bookkeeping resets back to "
            "back before collection begins; nothing has moved for "
            "those, so just press ENTER (they are flagged as such).",
            "blue",
        )


def build_scenario(parameters: Parameters, physics_backend: str = None):
    """Instantiate the scenario for `scenario_type` and the physics backend."""

    backend = resolve_physics_backend(parameters, physics_backend)
    scenario_type = canonical_scenario_name(parameters.scenario_type)
    parameters.scenario_type = scenario_type

    if backend in LAB_PHYSICS_BACKENDS:
        if scenario_type not in ROAD_TRAFFIC_SCENARIOS:
            raise ValueError(
                f"physics_backend='{backend}' only supports road-traffic "
                f"scenarios; got scenario_type '{scenario_type}'."
            )
        validate_lab_backend(parameters, backend)
        module_name, class_name = _LAB_ROAD_SCENARIO
    elif scenario_type in ROAD_TRAFFIC_SCENARIOS:
        module_name, class_name = _ROAD_SCENARIO
    else:
        module_name, class_name = _SCENARIO_MODULES[scenario_type]

    scenario_class = getattr(importlib.import_module(module_name), class_name)
    scenario = scenario_class()
    # Pass the parameters to the scenario so that it can use them to set up the
    # environment (e.g., the number of agents, their dynamics, etc.).
    scenario.parameters = parameters
    return scenario


def build_vmas_env(
    parameters: Parameters,
    physics_backend: str = None,
    n_agents: int = None,
    num_envs: int = None,
    **kwargs,
) -> TransformedEnvCustom:
    """Build a transformed VMAS environment for any registered scenario.

    `physics_backend`, `n_agents` and `num_envs` override the corresponding
    `parameters` fields for callers that construct an env for something other
    than the run the parameters describe (e.g. inference on a fleet whose size
    is set by ROS launch arguments).
    """

    scenario = build_scenario(parameters, physics_backend)
    env = VmasEnv(
        scenario=scenario,
        num_envs=parameters.num_vmas_envs if num_envs is None else num_envs,
        continuous_actions=True,
        max_steps=parameters.max_steps,
        device=parameters.device,
        n_agents=parameters.n_agents if n_agents is None else n_agents,
        **kwargs,
    )
    install_render_backend(env._env)
    return TransformedEnvCustom(
        env,
        RewardSum(
            in_keys=[env.reward_key],
            out_keys=[("agents", "episode_reward")],
        ),
    )
