"""
Some classes to bundle variables together when tracking the results
"""

class ActionState:
    def __init__(self, id, target, translation):
        self.id = id
        self.target = target
        self.translation = translation


class PolicyState:
    def __init__(self, chosen_action: ActionState, losses, loss_per_space):
        self.chosen_action = chosen_action
        self.losses = losses
        self.loss_per_space = loss_per_space


class BeliefState:
    def __init__(self, rotation, translation, beliefs, obj_position, kernel_epsilon):
        self.rotation = rotation
        self.translation = translation
        self.beliefs = beliefs
        self.obj_position = obj_position
        self.kernel_epsilon = kernel_epsilon
