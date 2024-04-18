"""
This script is used to visualize the trajectories of the agent for each sim.
"""

import os
import jsonpickle
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


def main():
    folder = "./sims/"
    files = os.listdir(folder)
    for simfile in files:
        path = os.path.join(folder, simfile)
        with open(path, "r") as file:
            content = file.read()
            deserialized = jsonpickle.decode(content)

        # Plot
        gamma =deserialized["params"]["gamma"]
        epsilon = deserialized["params"]["beliefs_spaces"][0]["initial_kernel_epsilon"]
 
        fig = plt.figure(figsize=(11, 5))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        plot_traj(deserialized, ax1)
        plot_loss(deserialized, ax2)
        # plot_priors(deserialized, ax3)
        plt.show()


def plot_traj(simulation_dictionary, ax):
    # targets
    belief_spaces = simulation_dictionary["params"]["beliefs_spaces"]
    targets = [space["target"]["values"] for space in belief_spaces]

    steps = simulation_dictionary["steps"]
    # Agent positions
    # Any one belief state space is ok for this, so we just use the first one (0)
    translations = [s["states"][0]["frame_translation"] for s in steps]
    rotations = [s["states"][0]["frame_rotation"] for s in steps]
    gaze_targets = [s["policy"]["chosen_action"]["target"] for s in steps]

    final_translation = simulation_dictionary["final_state"][0]["frame_translation"]
    final_rotation = simulation_dictionary["final_state"][0]["frame_rotation"]
    translations.append(final_translation)
    rotations.append(final_rotation)

    translations = np.array(translations)
    rotations = np.array(rotations)

    is_euclidean = simulation_dictionary["params"]["gamma"] == 0

    # Compute world position based on rotation and translation
    positions = []
    for translation, rotation in zip(translations, rotations):
        invert_rotation = np.linalg.inv(rotation)
        translation_world = np.matmul(invert_rotation, translation)
        positions.append(-translation_world)
    positions = np.array(positions)

    _plot_traj(is_euclidean, targets, positions, gaze_targets, ax)

def plot_priors(simulation_dictionary, ax):
    # targets
    belief_spaces = simulation_dictionary["params"]["beliefs_spaces"]
    space_count = len(belief_spaces)

    # data tracking, indexed by timestep
    steps = simulation_dictionary["steps"]
    is_euclidean = simulation_dictionary["params"]["gamma"] == 0

    priors_cov = []
    for i in range(space_count):
        priors_cov.append([])

    # Select loss evolution
    for step in steps:
        states = step["states"]
        for i in range(space_count):
            cov_matrix = np.array(states[i]["beliefs_cov"])
            # volume = 2 * np.pi * np.sqrt(abs(cov_matrix[0][0])) * np.sqrt(abs(cov_matrix[1][1]))
            volume = cov_matrix[0][0] * cov_matrix[1][1] - cov_matrix[0][1] * cov_matrix[1][0]
            priors_cov[i].append(volume)

    if is_euclidean:
        ax.set_title("Volume of priors (euclidean)")
    else:
        ax.set_title("Volume of priors (projective)")

    # plot each loss
    for i, priors in enumerate(priors_cov):
        ax.plot(priors, label=f"Volume in space {i+1}")

    # Add labels and legend
    ax.set_xlabel("Time")
    ax.set_ylabel("Volume of the priors")
    ax.legend()


def plot_loss(simulation_dictionary, ax):
    # targets
    belief_spaces = simulation_dictionary["params"]["beliefs_spaces"]
    space_count = len(belief_spaces)

    # data tracking, indexed by timestep
    loss_evolution = []
    for i in range(space_count):
        loss_evolution.append([])

    steps = simulation_dictionary["steps"]
    gamma = simulation_dictionary["params"]["gamma"]
    epsilon = simulation_dictionary["params"]["beliefs_spaces"][0]["initial_kernel_epsilon"]

    # Select loss evolution
    for step in steps:
        chosen_action = step["policy"]["chosen_action"]["id"]
        loss_per_space = np.array(step["policy"]["loss_per_space"])

        for i in range(space_count):
            loss_evolution[i].append(-loss_per_space[i, chosen_action])

    ax.set_title(f"Epistemic value gamma={gamma} epsilon={epsilon}")

    # plot each loss
    for i, loss_history in enumerate(loss_evolution):
        ax.plot(loss_history, label=f"target {i+1} value")

    # Add labels and legend
    ax.set_xlabel("Time")
    ax.set_ylabel("Epistemic value")
    ax.legend()


def _plot_traj(is_euclidean, targets, positions, gaze_targets, ax):
    if is_euclidean:
        ax.set_title("Agent movement in the Euclidean case")
    else:
        ax.set_title("Agent movement in the projective case")

    ax.tick_params(bottom=False, left=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_aspect("equal", adjustable="datalim")

    ax.grid(color="gray", linestyle="dashed", linewidth=0.5)
    ax.tick_params(color="gray", labelcolor="gray")
    for spine in ax.spines.values():
        spine.set_edgecolor("gray")

    ax.set_axisbelow(True)

    text_delta = np.array((0.05, 0))

    # targets
    for i_target, target in enumerate(targets):
        ax.scatter(
            target[0],
            target[1],
            facecolors="black",
            edgecolors="black",
            linewidths=1,
        )
        ax.annotate(f"object {i_target+1}", target + text_delta)

    # Translations
    arrows = [
        (positions[i], positions[i + 1] - positions[i])
        for i in range(len(positions) - 1)
    ]

    cmap = lambda x: cm.viridis(x*0.5 + 0.5)  # You can choose any colormap

    # Generate a gradient of colors
    colors = [cmap(i/len(arrows)) for i in range(len(arrows))]

    for i, arrow in enumerate(arrows):
        # don't draw arrows that are too short
        vect = np.array(arrow[1])
        if np.linalg.norm(vect) < 0.01:
            continue

        # Arrow for the translation
        ax.arrow(
            *arrow[0],
            *(arrow[1] * 0.8),
            head_width=0.02,
            head_length=0.01,
            width=0.005,
            length_includes_head=True,
            color=colors[i],
        )
        # ax.text(*arrow[0], f"{i+1}", fontsize=7, color="red")

    # Rotations
    normalize = lambda v: 0.05 * v / np.linalg.norm(v)
    rotation_arrows = [
        (positions[i], normalize(np.array(gaze_targets[i] - positions[i])))
        for i in range(len(gaze_targets))
    ]
    for i, arrow in enumerate(rotation_arrows):
        # Arrow for the rotation
        ax.arrow(
            *arrow[0],
            *(arrow[1] * 0.8),
            head_width=0.01,
            head_length=0.01,
            width=0.004,
            length_includes_head=True,
            color="black",
        )

    # Agent
    ax.scatter(
        positions[0][0],
        positions[0][1],
        facecolors="black",
        edgecolors="black",
        linewidths=1,
        marker="s",
    )
    ax.annotate("agent", positions[0] + text_delta)



if __name__ == "__main__":
    main()
