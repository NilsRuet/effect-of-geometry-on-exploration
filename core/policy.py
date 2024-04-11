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
        self, actions: list[PolicyItem], default_action_index=0
    ):
        # Compute beliefs and loss for each perception space
        losses = []
        loss_per_space = [] # datatracking variable

        for action in actions:
            current_action_losses = []
            for space_i in range(len(action.beliefs)):
                if(action.observation_available[space_i]):
                    space_loss = self.loss([action.beliefs[space_i]])[0]
                else:
                    space_loss = 0.0

                current_action_losses.append(space_loss)

                # Track loss per space
                if(len(loss_per_space) <= space_i):
                    loss_per_space.append([])
                loss_per_space[space_i].append(space_loss)

            losses.append(np.sum(current_action_losses))

        losses = np.array(losses)
        loss_per_space = np.array(loss_per_space)
        
        # Find the best action
        best_action_index = np.argmin(losses, axis=0)

        if not actions[best_action_index].is_legal:
            if self.default_on_illegal or True: # TODO: remove or True, implement the other case
                # Set the best action to the default action
                best_action_index = default_action_index

        # If the best loss is not at least a quantity epsilon away from the default action's loss, the default action is selected
        if best_action_index != default_action_index:
            default_loss = losses[default_action_index]
            if np.abs(default_loss - losses[best_action_index]) < self.loss_epsilon:
                best_action_index = default_action_index

        return (best_action_index, losses, loss_per_space)
