"""
This script is used to visualize the trajectories of the agent for each sim.
"""

import os
import jsonpickle
import matplotlib.pyplot as plt
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
        fig = plt.figure(figsize=(11, 5))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        plot_traj(deserialized, ax1)
        plot_loss(deserialized, ax2)
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

    _plot_traj(is_euclidean, targets, positions, ax)

def plot_loss(simulation_dictionary, ax):
    # targets
    belief_spaces = simulation_dictionary["params"]["beliefs_spaces"]
    space_count = len(belief_spaces)

    # data tracking, indexed by timestep
    loss_evolution = []
    for i in range(space_count):
        loss_evolution.append([])

    steps = simulation_dictionary["steps"]
    is_euclidean = simulation_dictionary["params"]["gamma"] == 0

    # Select loss evolution
    for step in steps:
        chosen_action = step["policy"]["chosen_action"]["id"]
        loss_per_space = np.array(step["policy"]["loss_per_space"])
        for i in range(space_count):
            loss_evolution[i].append(loss_per_space[i, chosen_action])

    if is_euclidean:
        ax.set_title("Loss evolution (euclidean)")
    else:
        ax.set_title("Loss evolution (projective)")

    # plot each loss
    for i, loss_history in enumerate(loss_evolution):
        ax.plot(loss_history, label=f"target {i+1} loss")

    # Add labels and legend
    ax.set_xlabel('Time')
    ax.set_ylabel('Loss')
    ax.legend()


def _plot_traj(is_euclidean, targets, positions, ax):
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

    for arrow in arrows:
        # don't draw arrows that are too short
        vect = np.array(arrow[1])
        if(np.linalg.norm(vect) < 0.01):
            continue

        # Arrow for the translation 
        ax.arrow(
            *arrow[0],
            *(arrow[1] * 0.8),
            head_width=0.04,
            head_length=0.03,
            width=0.008,
            length_includes_head=True,
            color="gray"
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
