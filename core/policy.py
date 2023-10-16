"""
This class defines the policy used by the agent
"""

import numpy as np
from core.actions import Translation2DActionSpace, ProjectiveTransformation
from core.beliefs import Beliefs
from core.loss import EpistemicLoss
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
        current_transformation: ProjectiveTransformation,
        time: float,
        beliefs: Beliefs,
        world_object_position: np.ndarray,
    ):
        random_actions, default_action_index = self.action_space.sample(
            current_transformation, world_object_position, time
        )

        predicted_future_beliefs = beliefs.propagate_actions(random_actions)
        losses = self.loss(predicted_future_beliefs)

        Logger.debug(f"losses : {losses}")
        best_action_index = np.argmin(losses, axis=0)

        # If the best loss is not at least a quantity epsilon away from the default action's loss, the default action is selected
        if best_action_index != default_action_index:
            default_loss = losses[default_action_index]
            if np.abs(default_loss - losses[best_action_index]) < self.loss_epsilon:
                best_action_index = default_action_index

        return (
            losses,
            best_action_index,
            random_actions[best_action_index],
            predicted_future_beliefs[best_action_index],
        )
