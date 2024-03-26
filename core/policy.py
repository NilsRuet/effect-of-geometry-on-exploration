"""
This class defines the policy used by the agent
"""

import numpy as np
from core.actions import Translation2DActionSpace, ProjectiveTransformation
from core.beliefs import Beliefs
from core.loss import EpistemicLoss
from core.states import PolicyState, ActionState
from utils.logger import Logger


# Policy that iterates over actions and select the minimal cost
# if it's different enough from the cost of a default action
class ArgminWithEpsilonPolicy:
    def __init__(
        self,
        action_space: Translation2DActionSpace,
        loss: EpistemicLoss,
        loss_epsilon: float,
    ):
        self.action_space = action_space
        self.loss = loss
        self.loss_epsilon = loss_epsilon

    def select(
        self,
        current_transformations: list[ProjectiveTransformation],
        beliefs: list[Beliefs],
        observations: list[np.ndarray],
    ):
        
        actions_per_space, default_action_index, world_translations, valid_actions = self.action_space.sample(
            current_transformations, observations
        )
        
        # Compute beliefs and loss for each perception space
        future_beliefs_per_space = [] # indexed by space, then action
        loss_per_space = []
        space_count = len(actions_per_space)
        for i in range(space_count):
            b = beliefs[i]
            actions = actions_per_space[i]
            future_beliefs = b.propagate_actions(actions, f"space {i+1}/{space_count}: ")
            
            future_beliefs_per_space.append(future_beliefs)
            loss_per_space.append(self.loss(future_beliefs))

        # convert to numpy arrays
        future_beliefs_per_space = np.array(future_beliefs_per_space)

        # combine losses by averaging
        losses = np.sum(loss_per_space, axis=0)

        # Find the best legal action
        original_indices = np.where(valid_actions)[0] # Keep an array that maps indices of valid_actions to the original indices (including invalid ones)
        valid_action_index = np.argmin(losses[valid_actions], axis=0) # Find the minimal loss among valid actions
        best_action_index = original_indices[valid_action_index]  # Retrieve the original index

        # If the best loss is not at least a quantity epsilon away from the default action's loss, the default action is selected
        if best_action_index != default_action_index:
            default_loss = losses[default_action_index]
            if np.abs(default_loss - losses[best_action_index]) < self.loss_epsilon:
                best_action_index = default_action_index
        
        action_state = ActionState(best_action_index, world_translations[best_action_index])
        policy_state = PolicyState(action_state, losses)

        return (
            policy_state,
            actions_per_space[:,best_action_index],
            future_beliefs_per_space[:,best_action_index],
        )
