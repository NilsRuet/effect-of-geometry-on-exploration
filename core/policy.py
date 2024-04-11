"""
This class defines the policy used by the agent
"""

import numpy as np
from core.beliefs import Beliefs
from core.loss import EpistemicLoss


class PolicyItem:
    def __init__(self, target, actions_per_space, beliefs_per_space: list[Beliefs], observation_available: list[bool], is_legal: bool):
        self.target = target
        self.actions_per_space = actions_per_space
        self.beliefs = beliefs_per_space
        self.observation_available = observation_available
        self.is_legal = is_legal

# Policy that iterates over actions and select the minimal cost
# if it's different enough from the cost of a default action
class ArgminWithEpsilonPolicy:
    def __init__(
        self,
        loss: EpistemicLoss,
        loss_epsilon: float,
        default_on_illegal: bool
    ):
        self.loss = loss
        self.loss_epsilon = loss_epsilon
        self.default_on_illegal = default_on_illegal

    def select(
        self, actions: list[PolicyItem], default_action_indices
    ):
        # Compute beliefs and loss for each perception space
        losses = []
        loss_per_space = [] # datatracking variable

        for action in actions:
            action_loss = self.loss(action.beliefs)
            for space_i in range(len(action.beliefs)):
                # Track loss per space
                if(len(loss_per_space) <= space_i):
                    loss_per_space.append([])
                loss_per_space[space_i].append(action_loss[space_i])

            losses.append(np.sum(action_loss[action.observation_available]))

        losses = np.array(losses)
        loss_per_space = np.array(loss_per_space)
        
        # Find the best action
        if(self.default_on_illegal):
            best_action_index = np.argmin(losses, axis=0)
            if not actions[best_action_index].is_legal:
                best_action_index = self.select_best_among_default(losses, default_action_indices)
        else:
            legal_actions = [a.is_legal for a in actions]
            valid_mask = np.where(legal_actions, False, True)
            masked_losses = np.ma.masked_array(losses, valid_mask)
            best_action_index = np.argmin(masked_losses, axis=0)
    

        # If the best loss is not at least a quantity epsilon away from one of the default action's loss, the default action is selected
        if not best_action_index in default_action_indices:
            default_losses = [losses[i] for i in default_action_indices]
            under_epsilon = [np.abs(default_loss - losses[best_action_index]) < self.loss_epsilon for default_loss in default_losses]
            if True in under_epsilon:
                best_action_index = self.select_best_among_default(losses, default_action_indices)

        return (best_action_index, losses, loss_per_space)

    def select_best_among_default(self, losses, default_action_indices):
        # TODO : check that there are no case where the default actions result in different losses
        # Set the best available default action
        best_default_action = np.argmin(losses[default_action_indices], axis=0)
        return default_action_indices[best_default_action]