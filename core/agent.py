"""
These classes define an agent and how it perceives the world
"""

import numpy as np
from core.policy import PolicyItem
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
        initial_beliefs: Beliefs
    ):
        self.id = id
        self.reference_frame = reference_frame
        self.world = world
        self.beliefs = initial_beliefs

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

        # Sample action that transform the belief spaces and compute their outcomes
        actions_per_space, idle_action_index, world_translations, valid_actions = (
            self.action_space.sample(frames, observations)
        )

        future_beliefs_per_space = self._plan_frame_transformations(
            beliefs,
            actions_per_space
        )

        # plan which observation will be available per translation
        fov_targets = observations
        world_positions = -world_translations
        # indexed by translation, then by target
        visibility_per_translation = self._plan_fov(world_positions, fov_targets)

   
        # List all possible outcomes, and compute the loss
        all_policy_items = []
        default_actions = [] # rotations are somewhat irrelevant, so they are several default actions
        translation_indices = []

        # This generates an item for each translation and each rotation
        for translation_i in range(actions_per_space.shape[1]):
            action_visibilities = visibility_per_translation[translation_i]
            for target, visibility in zip(fov_targets, action_visibilities):
                item = PolicyItem(
                    target,
                    actions_per_space[:, translation_i],
                    future_beliefs_per_space[:, translation_i],
                    visibility,
                    valid_actions[translation_i]
                )
                all_policy_items.append(item)
                # keep track of idle action
                translation_indices.append(translation_i)
                if(translation_i == idle_action_index):
                    default_actions.append(len(all_policy_items) - 1)

        # Apply policy based on the planned outcomes
        best_action_index, losses, loss_per_space = self.policy.select(
            all_policy_items, default_actions
        )

        # Get best action results
        best_action = all_policy_items[best_action_index]
        best_moves = best_action.actions_per_space
        new_beliefs = best_action.beliefs
        visibility = best_action.observation_available

        # Update beliefs based on action and observations
        self.update_beliefs(observations, best_moves, new_beliefs, visibility)

        # For data tracking
        best_world_translation = world_translations[translation_indices[best_action_index]]
        action_state = ActionState(
            best_action_index, best_action.target, best_world_translation 
        )
        policy_state = PolicyState(action_state, losses, loss_per_space)

        # Logging
        Logger.debug(f"Step t = {time}")
        for space in self.spaces:
            Logger.debug(
                f"frame translation: {space.reference_frame.transformation.translation}"
            )

        return policy_state

    def update_beliefs(self, observations, moves, new_beliefs, visible_observations):
        for space, best_move, observation, beliefs, is_visible in zip(
            self.spaces, moves, observations, new_beliefs, visible_observations
        ):
            # Update reference frame and beliefs
            space.reference_frame.update(best_move.phi_rm)
            space.beliefs = beliefs

            # Update prior based on observation
            if(is_visible):
                # TODO : 
                # Remove "is visible"
                # Observations become gaussians
                # Update using integration
                local_observation = space.reference_frame.world_to_local(observation)
                space.beliefs.update(local_observation)


    def get_current_frames(self):
        return [s.reference_frame.transformation for s in self.spaces]

    # for data tracking
    def get_belief_states(self):
        belief_space_states = []
        for space in self.spaces:
            rotation = space.reference_frame.transformation.linear_map
            translation = space.reference_frame.transformation.translation
            beliefs = space.beliefs
            observation_kernel_epsilon = space.beliefs.observation_kernel.epsilon
            obj_position = space.observe()
            belief_space_states.append(
                BeliefState(rotation, translation, beliefs, obj_position, observation_kernel_epsilon)
            )
        return belief_space_states

    def _plan_frame_transformations(
        self,
        beliefs: list[Beliefs],
        actions_per_space
    ):
        # Compute beliefs and loss for each perception space
        future_beliefs_per_space = []  # indexed by space, then action

        space_count = len(actions_per_space)
        # For each space, plan the effect of actions
        for space_i, b, actions in zip(
            range(space_count), beliefs, actions_per_space
        ):
            future_beliefs = b.propagate_actions(
                actions, b.observation_kernel, f"space {space_i+1}/{space_count}: "
            )
            future_beliefs_per_space.append(future_beliefs)
        return np.array(future_beliefs_per_space)
    
    def _plan_fov(self, world_positions, fov_targets):
        # TODO : real field of view, for now only one target will be visible at a given time

        target_count = len(fov_targets)
        # The visiblity matrix is indexed by fov target, and each column holds wether the associated objet is visible
        # this short line means a target will be visible only when the agent focuses on it
        visibilities = np.identity(target_count) > 0

        return np.array([visibilities for _ in world_positions])
        
