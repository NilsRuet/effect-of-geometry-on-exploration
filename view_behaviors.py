import os
import jsonpickle
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.interpolate import griddata
import numpy as np


class BehaviorData:
    def __init__(self, gamma, epsilon, min_distances, focus_switch_count):
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_distances = min_distances
        self.focus_switch_count = focus_switch_count


def main():
    folder = "./sims/"
    files = os.listdir(folder)
    data_grid = []
    for simfile in files:
        path = os.path.join(folder, simfile)
        with open(path, "r") as file:
            content = file.read()
            deserialized = jsonpickle.decode(content)
            bhv_data = get_behavior(deserialized)
            data_grid.append(bhv_data)

        # ax2 = fig.add_subplot(122)
        # plot_traj(deserialized, ax1)
        # plot_loss(deserialized, ax2)

    fig = plt.figure(figsize=(11, 5))
    ax1 = fig.add_subplot(121)
    plot_switch_grid(data_grid, ax1)
    plt.show()


def plot_switch_grid(data_grid: list[BehaviorData], ax):
    gamma_values = [d.gamma for d in data_grid]
    epsilon_values = [d.epsilon for d in data_grid]
    switch_values = [d.focus_switch_count for d in data_grid]

    # Define grid for interpolation
    gamma_value_count = 10
    epsilon_value_count = 9
    epsilon_step = 0.1
    gamma_step = 0.1
    gamma_min_plot = min(gamma_values) - (gamma_step / 2)
    gamma_max_plot = max(gamma_values) + (gamma_step / 2)
    epsilon_min_plot = min(epsilon_values) - (epsilon_step / 2)
    epsilon_max_plot = max(epsilon_values) + (epsilon_step / 2)

    xi = np.linspace(gamma_min_plot, gamma_max_plot, gamma_value_count + 1)
    yi = np.linspace(epsilon_min_plot, epsilon_max_plot, epsilon_value_count + 1)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate k values
    zi = griddata(
        (gamma_values, epsilon_values), switch_values, (xi, yi), method="nearest"
    )

    # Create the colormap plot
    cmap = plt.get_cmap('Greys')
    norm = Normalize(vmin=min(switch_values), vmax=max(switch_values))
    im = ax.imshow(
        zi,
        extent=(gamma_min_plot, gamma_max_plot, epsilon_min_plot, epsilon_max_plot),
        origin="lower",
        cmap=cmap,
        norm=norm,
    )
    ax.set_xlabel("gamma")
    ax.set_ylabel("epsilon")
    ax.set_title("Attention swaps for the first 40 steps")
    plt.colorbar(im, ax=ax, label="k")  # Add a colorbar with label
    ax.set_aspect("equal")


def get_behavior(simulation):
    steps = simulation["steps"]
    gamma = simulation["params"]["gamma"]
    translation_norm = simulation["params"]["norm_of_translations"]
    epsilon = simulation["params"]["beliefs_spaces"][0][
        "initial_kernel_epsilon"
    ]  # both have the same epsilon

    # targets
    belief_spaces = simulation["params"]["beliefs_spaces"]
    targets = np.array([space["target"]["values"] for space in belief_spaces])

    # Compute when the agent translates toward a different object
    # We do not use gaze targets as gaze targets may change while the agent stays idle
    positions = get_positions(simulation, steps)
    positions_reshaped = positions[
        :, np.newaxis, :
    ]  # reshape so that each position may be subtracted to several points
    targets_reshaped = targets[
        np.newaxis, :, :
    ]  # reshape so that targets are compatible to the positions_reshaped
    difference = targets_reshaped - positions_reshaped  # subtract via broadcasting
    distances = np.linalg.norm(difference, axis=-1)  # compute norm for each difference

    target_indices = []
    for i in range(len(distances) - 1):
        dist_before = distances[i]
        dist_after = distances[i + 1]
        target_dist_diff = dist_after - dist_before

        if np.any(
            abs(target_dist_diff) > 1e-5
        ):  # 1e-5 is an epsilon to account for numerical errors
            # Check which target the action shortened the distance to more
            target_index = np.argmin(target_dist_diff)
            target_indices.append(target_index)

    switch_count = count_changes(target_indices)
    min_distances = np.min(distances, axis=0)
    return BehaviorData(gamma, epsilon, min_distances, switch_count)


def count_changes(sequence):
    count = 0
    for i in range(len(sequence) - 1):
        if sequence[i] != sequence[i + 1]:
            count += 1
    return count


def get_positions(simulation, steps):
    translations = [s["states"][0]["frame_translation"] for s in steps]
    rotations = [s["states"][0]["frame_rotation"] for s in steps]
    final_translation = simulation["final_state"][0]["frame_translation"]
    final_rotation = simulation["final_state"][0]["frame_rotation"]
    translations.append(final_translation)
    rotations.append(final_rotation)

    translations = np.array(translations)
    rotations = np.array(rotations)

    # Compute world position based on rotation and translation
    positions = []
    for translation, rotation in zip(translations, rotations):
        invert_rotation = np.linalg.inv(rotation)
        translation_world = np.matmul(invert_rotation, translation)
        positions.append(-translation_world)
    return np.array(positions)


if __name__ == "__main__":
    main()
