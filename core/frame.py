"""
This class defines the reference frame of an agent
"""

import numpy as np
from core.actions import ProjectiveTransformation


class ReferenceFrame:
    def __init__(self, transformation):
        self.transformation: ProjectiveTransformation = transformation

    def update(self, transformation: ProjectiveTransformation):
        self.transformation = transformation

    def world_to_local(self, observation):
        return self.transformation.transform(observation)

    def __repr__(self) -> str:
        return f"{self.transformation}"
