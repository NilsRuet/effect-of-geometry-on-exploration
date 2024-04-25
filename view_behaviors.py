import os
import jsonpickle
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize, BoundaryNorm
from scipy.interpolate import griddata
import numpy as np


class BehaviorData:
    def __init__(
        self,
        gamma,
        epsilon,
        min_distances,
        min_distance_indices,
        focus_switch_count,
        attention_switch_count,
        cumulated_epistemic_value,
        reached_targets
    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_distances = min_distances
        self.min_distance_indices = min_distance_indices
        self.focus_switch_count = focus_switch_count
        self.attention_switch_count = attention_switch_count
        self.cumulated_epistemic_value = cumulated_epistemic_value
        self.targets_reached = reached_targets


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

    fig = plt.figure(figsize=(11, 5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    # fig.tight_layout()
    plot_behavior_grid(
        data_grid,
        ax1,
        lambda d: d.attention_switch_count,
        "Swaps of observed object",
        "Count",
    )
    plot_behavior_grid(
        data_grid,
        ax2,
        lambda d: d.targets_reached,
        "Objects reached",
        "Count",
        discrete=True
    )

    plt.show()


def plot_behavior_grid(
    data_grid: list[BehaviorData], ax, value_getter, title, value_label, discrete = False
):
    filtered_data_grid = [d for d in data_grid if d.gamma >= 0]

    gamma_values = [d.gamma for d in filtered_data_grid]
    epsilon_values = [d.epsilon for d in filtered_data_grid]
    values = [value_getter(d) for d in filtered_data_grid]

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
    zi = griddata((gamma_values, epsilon_values), values, (xi, yi), method="nearest")

    # Create the colormap plot
    cmap = plt.get_cmap("Greys_r")

    # Value bounds (discrete case)
    max_value = max(values)
    min_value = min(values)
    number_of_values = max_value - min_value + 1
    
    if not discrete:
        norm = Normalize(vmin=min(values), vmax=max(values))
    else:
        norm = BoundaryNorm(np.linspace(min_value-0.5, max_value+0.5,  number_of_values + 1), cmap.N)

    im = ax.imshow(
        zi,
        extent=(gamma_min_plot, gamma_max_plot, epsilon_min_plot, epsilon_max_plot),
        origin="lower",
        cmap=cmap,
        norm=norm,
    )
    ax.set_xlabel("γ", fontsize=16)
    ax.set_ylabel("ε",  fontsize=16)
    if not discrete:
        plt.colorbar(im, ax=ax, label=value_label)
    else:
        plt.colorbar(im, ax=ax, label=value_label, ticks=np.arange(min_value, max_value+1))

    ax.set_aspect("equal")
    ax.set_title(title)


def get_behavior(simulation):
    steps = simulation["steps"]
    gamma = simulation["params"]["gamma"]
    translation_norm = simulation["params"]["norm_of_translations"]
    illegal_radius = simulation["params"]["distance_filter"]
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

    # target indices
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

    # Rotation targets
    rotation_targets = np.array(
        [step["policy"]["chosen_action"]["target"] for step in steps]
    )
    rotation_switch_count = count_changes(
        rotation_targets, lambda p, p2: (p == p2).all()
    )

    # Classify wether the agent is near a target
    near_target_radius = illegal_radius + translation_norm
    close_targets_per_step = distances < near_target_radius
    reached_count = 0
    for i in range(len(targets)):
        # Count how many times each target became close
        reached_count += count_set_to_true(close_targets_per_step[:, i])

    # Cumulated epistemic value
    cumulative_epistemic_value = np.zeros(len(belief_spaces))
    for step in steps:
        chosen_action = step["policy"]["chosen_action"]["id"]
        cumulative_epistemic_value -= np.array(step["policy"]["loss_per_space"])[:,chosen_action]
    switch_count = count_changes(target_indices)
    min_distances = np.min(distances, axis=0)
    min_distance_indices = np.argmin(distances, axis=0)

    threshold = 0.25
    return BehaviorData(
        gamma,
        epsilon,
        min_distances,
        min_distance_indices,
        switch_count,
        rotation_switch_count,
        sum(cumulative_epistemic_value),
        reached_count
    )


def count_changes(sequence, equal_function=None):
    if equal_function == None:
        equal_function = lambda x, y: x == y

    count = 0
    for i in range(len(sequence) - 1):
        if not equal_function(sequence[i], sequence[i + 1]):
            count += 1
    return count

def count_set_to_true(sequence: list[bool]):
    count = 0
    if(sequence[0] == True):
        count += 1

    for i in range(len(sequence) - 1):
        if sequence[i] == False and sequence[i+1] == True :
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
