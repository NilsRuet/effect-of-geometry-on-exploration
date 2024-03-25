class ActionState:
    def __init__(self, id, translation):
        self.id = id
        self.translation = translation

class PolicyState:
    def __init__(self, chosen_action: ActionState, losses):
        self.chosen_action = chosen_action
        self.losses = losses

class BeliefState:
    def __init__(self, rotation, translation, beliefs, obj_position):
        self.rotation = rotation
        self.translation = translation
        self.beliefs = beliefs
        self.obj_position = obj_position