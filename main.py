"""
This script is used to manage, run and collect data about a set of simulations with different params
"""

import numpy as np
import time
from params import SimParams, BeliefSpaceParams
from core.simulation import Simulation
from utils.logger import Logger
from utils.datamanager import dataManager


def generate_exploration_params():
    steps = 10
    all_params = []
    base_norm = 0.1

    distance = (
        base_norm * steps
    )  # Ensure the agent can't go beyond the goal

    # Not equidistant
    pos1 = np.array((0, 1)) * distance
    pos2 = np.array((0.25, 1)) * distance

    belief_space_params = [BeliefSpaceParams(target=pos1), BeliefSpaceParams(target=pos2)]

    all_params.append(
        SimParams(
            gamma = 1,
            belief_spaces=belief_space_params,
            max_steps=steps,
            norm_of_translations=base_norm
        )
    )
    all_params.append(
        SimParams(
            gamma = 0,
            belief_spaces=belief_space_params,
            max_steps=steps,
            norm_of_translations=base_norm
        )
    )
    return all_params

# Generate params for sims with a single object located at evenly distributed angles
def generate_params_with_angles():
    steps = 10
    all_params = []
    point_count = 6
    angle_delta = np.pi * 2 / point_count
    base_norm = 0.1
    distance_variation_count = 1
    for distance_multiplier in range(1, distance_variation_count + 1):
        norm = base_norm
        distance = (
            distance_multiplier * base_norm * steps
        )  # Ensure the agent can't go beyond the goal

        for i in range(point_count):
            angle = i * angle_delta + np.pi / 2
            pos = np.array((np.cos(angle), np.sin(angle))) * distance

            all_params.append(
                SimParams(
                    gamma = 1,
                    belief_spaces=[BeliefSpaceParams(target=pos)],
                    max_steps=steps,
                    norm_of_translations=norm
                )
            )
            all_params.append(
                SimParams(
                    gamma = 0,
                    belief_spaces=[BeliefSpaceParams(target=pos)],
                    max_steps=steps,
                    norm_of_translations=norm
                )
            )
    return all_params


# Generate parameters of a sim for each position of a single object on a grid
def generate_grid_params():
    all_params = []
    movement_norm = 0.1
    half_diameter = 10

    # Creating a grid of positions
    x_range = np.linspace(
        -half_diameter * movement_norm,
        half_diameter * movement_norm,
        half_diameter * 2 + 1,
    )
    y_range = np.linspace(
        -half_diameter * movement_norm,
        half_diameter * movement_norm,
        half_diameter * 2 + 1,
    )
    pos_x, pos_y = np.meshgrid(x_range, y_range)
    positions = np.stack((pos_x, pos_y), axis=-1)

    for i in range(positions.shape[0]):
        for j in range(positions.shape[1]):
            object_pos = positions[i, j]
            if not np.array_equal(
                object_pos, (0, 0)
            ):  # Exclude case where the object is at the same position as the agent
                # euclidean
                all_params.append(
                    SimParams(
                        gamma = 1,
                        belief_spaces=[BeliefSpaceParams(target=object_pos)],
                        norm_of_translations=movement_norm,
                        max_steps=1,
                    )
                )
                # projective
                all_params.append(
                    SimParams(
                        gamma = 0,
                        belief_spaces=[BeliefSpaceParams(target=object_pos)],
                        norm_of_translations=movement_norm,
                        max_steps=1,
                    )
                )

    return all_params


def main():
    all_params = generate_exploration_params()
    sim = Simulation()
    for i, params in enumerate(all_params):
        Logger.debug(f"### Sim {i+1} ###")
        # dataManager records and write simulation data
        dataManager.notify_new_sim(params)
        t0 = time.time()
        sim.run(params)
        duration = time.time() - t0
        dataManager.notify_sim_end(duration)
        Logger.debug(f"total duration : {int(duration * 1000)}ms")


if __name__ == "__main__":
    main()
