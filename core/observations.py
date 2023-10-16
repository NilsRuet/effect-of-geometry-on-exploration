"""
This file contains classes that define the world and its observations
"""


# Used to make observations stochastic
class MarkovKernel:
    def __init__(self, epsilon):
        self.epsilon = epsilon


# Holds information about the world
class World:
    def __init__(self, object_position, noisy_sensor_kernel):
        self.markov_kernel = noisy_sensor_kernel
        self.object_position = object_position

    def observe_position(self):
        return self.object_position

    def get_sensor_kernel(self):
        return self.markov_kernel
