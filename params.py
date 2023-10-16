"""
This file defines classes used to specify the parameters of the simulations.
"""

import numpy as np


class SimParams:
    def __init__(
        self,
        gamma=1,
        object_position_in_world=np.array((0, 1)),
        initial_beliefs_covariance=0.1,
        markov_kernel_epsilon=0.5,
        deltatime=1,
        max_steps=10,
        norm_of_translations=0.1,
        direction_count=8,
        loss_epsilon=1e-4,
    ):
        self.gamma = gamma  # gamma in the projective transformation
        self.object_position_in_world = object_position_in_world
        self.initial_beliefs_covariance = initial_beliefs_covariance  # initial covariance is the identity matrix * this variable
        self.markov_kernel_epsilon = markov_kernel_epsilon
        self.deltatime = deltatime  # time between each step of the algorithm
        self.max_steps = max_steps  # How many steps before the simulation stops

        # Params for translation 2D action space
        self.norm_of_translations = norm_of_translations
        self.translation_direction_count = direction_count

        # how different from the default action the loss of an action has to be to be selected
        self.loss_epsilon = loss_epsilon


# Parameters of the simulation
class FixedParams:
    integration_std_count = 4  # range of the integrals in standard deviations
    integration_epsabs = 1.49e-2  # 1.49e-8
    integration_epsrel = 1.49e-2  # 1.49e-8
