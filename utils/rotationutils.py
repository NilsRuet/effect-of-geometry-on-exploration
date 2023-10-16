"""
This script defines common utility functions about 2D rotations
"""

import numpy as np


class RotationUtils:
    def generate_rotation_matrix(angle):
        return np.array(
            [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
        )
