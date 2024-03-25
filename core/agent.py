from core.beliefs import Beliefs
from core.frame import ReferenceFrame
from core.observations import ObjectSensor
from core.states import BeliefState
from utils.logger import Logger

class PerceptionSpace:
    def __init__(
        self,
        id,
        reference_frame: ReferenceFrame,
        world: ObjectSensor,
        initial_beliefs: Beliefs
    ):
        self.id = id
        self.reference_frame = reference_frame
        self.world = world
        self.beliefs = initial_beliefs

    def observe(self):
        return self.world.observe_position()

class Agent:
    def __init__(self, perception_spaces: list[PerceptionSpace], policy):
        self.spaces = perception_spaces
        self.policy = policy

    def step(self, time):
        observations = [s.observe() for s in self.spaces]
        frame_transformations = [s.reference_frame.transformation for s in self.spaces]
        beliefs = [s.beliefs for s in self.spaces]

        # best moves and new beliefs are indexed by space
        policy_state, best_moves, new_beliefs = self.policy.select(
            frame_transformations, beliefs, observations
        )

        # Debug info
        Logger.debug(f"Step t = {time}")
        for space, best_move, observation, beliefs in zip(self.spaces, best_moves, observations, new_beliefs):        
            space.reference_frame.update(best_move.phi_rm)
            space.beliefs = beliefs
            local_observation = space.reference_frame.world_to_local(observation)
            space.beliefs.update(local_observation)
            Logger.debug(f"Space {space.id}")
            Logger.debug(
                "mean = ({:0.3f} {:0.3f})".format(
                    space.beliefs.qx.mean[0], space.beliefs.qx.mean[1]
                )
            )
            Logger.debug(
                f"frame translation: {space.reference_frame.transformation.translation}"
            )

        return policy_state

    # for data tracking
    def get_belief_states(self):
        belief_space_states = []
        for space in self.spaces:
            rotation = space.reference_frame.transformation.linear_map
            translation = space.reference_frame.transformation.translation
            beliefs = space.beliefs
            obj_position = space.observe()
            belief_space_states.append(BeliefState(rotation, translation, beliefs, obj_position))
        return belief_space_states
