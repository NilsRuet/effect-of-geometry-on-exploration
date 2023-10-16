"""
This class defines the loss used by the agent
"""

import numpy as np
from typing import List
from core.beliefs import Beliefs


class EpistemicLoss:
    def __call__(self, future_beliefs: List[Beliefs]):
        losses = []
        for i_belief, beliefs in enumerate(future_beliefs):
            Kx = np.linalg.det(beliefs.qx.cov)
            Ky = np.linalg.det(beliefs.py.cov)
            Kxy = np.linalg.det(beliefs.pxy.cov)
            mutual_information = np.log(Kx * Ky / Kxy) / 2
            losses.append(mutual_information)
        # This is a loss that will be minimized, therefore it is made negative
        return -np.array(losses)
