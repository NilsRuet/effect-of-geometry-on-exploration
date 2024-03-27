"""
These classes define an agent and how it perceives the world
"""

import numpy as np
from core.beliefs import Beliefs
from core.frame import ReferenceFrame
from core.observations import ObjectSensor
from core.states import BeliefState, PolicyState, ActionState
from utils.logger import Logger
from utils.geometryutils import GeometryUtils


class PerceptionSpace:
    def __init__(
        self,
        id,
        reference_frame: ReferenceFrame,
        world: ObjectSensor,
        initial_beliefs: Beliefs,
        kernel_generator
    ):
        self.id = id
        self.reference_frame = reference_frame
        self.world = world
        self.beliefs = initial_beliefs
        self.kernel_generator = kernel_generator

    def observe(self):
        return self.world.observe_position()


class Agent:
    def __init__(
        self,
        perception_spaces: list[PerceptionSpace],
        action_space,
        policy
    ):
        self.spaces = perception_spaces
        self.action_space = action_space
        self.policy = policy

    def step(self, time):
        observations = [s.observe() for s in self.spaces]
        beliefs = [s.beliefs for s in self.spaces]
        frames = self.get_current_frames()

        # Sample actions and compute their outcomes
        actions_per_space, idle_action_index, world_translations, valid_actions = (
            self.action_space.sample(frames, observations)
        )

        future_beliefs_per_space = self.plan_actions(
            beliefs,
            observations,
            actions_per_space,
            world_translations,
            world_translations[idle_action_index],
        )

        # Apply policy based on the planned outcomes
        best_action_index, losses, loss_per_space = self.policy.select(
            future_beliefs_per_space, valid_actions, idle_action_index
        )
        best_moves = actions_per_space[:, best_action_index]
        new_beliefs = future_beliefs_per_space[:, best_action_index]

        # Update beliefs based on action and observations
        self.update_beliefs(observations, best_moves, new_beliefs)

        # For data tracking
        action_state = ActionState(
            best_action_index, world_translations[best_action_index]
        )
        policy_state = PolicyState(action_state, losses, loss_per_space)

        # Logging
        Logger.debug(f"Step t = {time}")
        for space in self.spaces:
            Logger.debug(
                f"frame translation: {space.reference_frame.transformation.translation}"
            )

        return policy_state

    def update_beliefs(self, observations, moves, new_beliefs):
        for space, best_move, observation, beliefs in zip(
            self.spaces, moves, observations, new_beliefs
        ):
            # Update reference frame and beliefs
            space.reference_frame.update(best_move.phi_rm)
            space.beliefs = beliefs

            # Update prior based on observation
            local_observation = space.reference_frame.world_to_local(observation)
            space.beliefs.update(local_observation)

    def plan_actions(
        self,
        beliefs: list[Beliefs],
        observations,
        actions_per_space,
        world_translations,
        current_world_position,
    ):
        # Compute beliefs and loss for each perception space
        future_beliefs_per_space = []  # indexed by space, then action

        space_count = len(actions_per_space)
        # For each space, plan the effect of actions
        for space_i, space, b, actions, obs in zip(
            range(space_count), self.spaces, beliefs, actions_per_space, observations
        ):
            observation_kernels = []
            # compute distance and eccentricty for each action
            for action_i in range(len(actions)):
                next_world_position = -world_translations[action_i]
                eccentricity = abs(
                    GeometryUtils.get_angle(
                        next_world_position, current_world_position, obs
                    )
                )
                distance = np.linalg.norm(obs - next_world_position)
                kernel = space.kernel_generator(eccentricity, distance)
                observation_kernels.append(kernel)

            future_beliefs = b.propagate_actions(
                actions, observation_kernels, f"space {space_i+1}/{space_count}: "
            )
            future_beliefs_per_space.append(future_beliefs)

        return np.array(future_beliefs_per_space)

    def get_current_frames(self):
        return [s.reference_frame.transformation for s in self.spaces]

    # for data tracking
    def get_belief_states(self):
        belief_space_states = []
        for space in self.spaces:
            rotation = space.reference_frame.transformation.linear_map
            translation = space.reference_frame.transformation.translation
            beliefs = space.beliefs
            obj_position = space.observe()
            belief_space_states.append(
                BeliefState(rotation, translation, beliefs, obj_position)
            )
        return belief_space_states
