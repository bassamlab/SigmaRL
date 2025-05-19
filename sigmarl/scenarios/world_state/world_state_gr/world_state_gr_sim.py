from sigmarl.scenarios.world_state.world_state_gr.world_state_gr import WorldStateGR
from sigmarl.scenarios.world_state.world_state_sim import WorldStateSim


class WorldStateGRSimulation(WorldStateGR, WorldStateSim):
    def update_state_before_rewarding(self, agent_index: int):
        """
        Update some states (such as vertices of the agent) that will be used before rewarding agents.
        """

        agent_states = self.get_agent_state_list()

        self.update_vertices(agent_states)

        self.update_distances(agent_states, agent_index)
