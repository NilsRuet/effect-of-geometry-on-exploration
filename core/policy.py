"""
This class defines the policy used by the agent
"""

import numpy as np
from core.beliefs import Beliefs
from core.loss import EpistemicLoss


# Policy that iterates over actions and select the minimal cost
# if it's different enough from the cost of a default action
class ArgminWithEpsilonPolicy:
    def __init__(
        self,
        loss: EpistemicLoss,
        loss_epsilon: float,
        default_on_illegal: bool,
        merge_by_min: bool,
    ):
        self.loss = loss
        self.loss_epsilon = loss_epsilon
        self.default_on_illegal = default_on_illegal
        self.merge_by_min = merge_by_min

    def select(
        self, beliefs_per_space: list[Beliefs], valid_actions, default_action_index=0
    ):
        # Compute beliefs and loss for each perception space
        loss_per_space = []

        for beliefs in beliefs_per_space:
            loss_per_space.append(self.loss(beliefs))

        # combine losses
        if self.merge_by_min:
            losses = np.min(loss_per_space, axis=0)
        else:
            losses = np.sum(loss_per_space, axis=0)

        # Find the best action
        best_action_index = np.argmin(losses, axis=0)

        if not valid_actions[best_action_index]:
            if self.default_on_illegal:
                # Set the best action to the default action
                best_action_index = default_action_index
            else:
                # Find the best legal action
                original_indices = np.where(valid_actions)[
                    0
                ]  # Keep an array that maps indices of valid_actions to the original indices (including invalid ones)
                valid_action_index = np.argmin(
                    losses[valid_actions], axis=0
                )  # Find the minimal loss among valid actions
                best_action_index = original_indices[
                    valid_action_index
                ]  # Retrieve the original index

        # If the best loss is not at least a quantity epsilon away from the default action's loss, the default action is selected
        if best_action_index != default_action_index:
            default_loss = losses[default_action_index]
            if np.abs(default_loss - losses[best_action_index]) < self.loss_epsilon:
                best_action_index = default_action_index

        return (best_action_index, losses, loss_per_space)
