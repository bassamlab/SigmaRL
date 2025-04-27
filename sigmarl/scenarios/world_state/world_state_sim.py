from abc import abstractmethod

from sigmarl.helper_training import WorldCustom
from sigmarl.scenarios.observations.observation_provider import AgentState
from sigmarl.scenarios.world_state.world_state import WorldState, WorldStateParameters

"""
this class contains some useful methods and parameters which are shared 
by world states of simulated environments.
"""

class WorldStateSim(WorldState):

    def __init__(self, params: WorldStateParameters):
        super().__init__(params)

        self.world = None

    """
    Usually our world state is defined in the _init_params method of our scenarios.
    When _init_params is called there exists no world, therefore we add that attribute to
    this class after the world was created in the scenario class
    """
    def init_world(self, world: WorldCustom):
        self.world = world

    """
    This method is used to create the intermediate format for observation extraction 
    from simulated agents used by the observation providers
    """
    def get_agent_state_list(self):

        return [
            AgentState(agent.state.pos,
                       agent.state.rot,
                       agent.state.steering,
                       agent.state.vel)
            for agent in self.world.agents
        ]

    """
    This method is called in simulated environments before we calculate the rewards.
    """
    @abstractmethod
    def update_state_before_rewarding(self, agent_index: int):
        pass