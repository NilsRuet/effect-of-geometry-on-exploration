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

        view_sim(deserialized)


def view_sim(simulation_dictionary):
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

    # Plot
    traj(is_euclidean, targets, positions)

    plt.show()


def traj(is_euclidean, targets, positions):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    if is_euclidean:
        ax1.set_title("Agent movement in the Euclidean case")
    else:
        ax1.set_title("Agent movement in the projective case")

    ax1.tick_params(bottom=False, left=False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)
    ax1.spines["left"].set_visible(False)
    ax1.set_aspect("equal", adjustable="datalim")

    ax1.grid(color="gray", linestyle="dashed", linewidth=0.5)
    ax1.tick_params(color="gray", labelcolor="gray")
    for spine in ax1.spines.values():
        spine.set_edgecolor("gray")

    ax1.set_axisbelow(True)

    text_delta = np.array((0.05, 0))
    
    # targets
    for i_target, target in enumerate(targets):
        ax1.scatter(
            target[0],
            target[1],
            facecolors="black",
            edgecolors="black",
            linewidths=1,
        )
        ax1.annotate(f"object {i_target+1}", target + text_delta)

    # Translations
    arrows = [
        (positions[i], positions[i + 1] - positions[i])
        for i in range(len(positions) - 1)
    ]
    for arrow in arrows:
        ax1.arrow(
            *arrow[0],
            *(arrow[1] * 0.8),
            head_width=0.04,
            head_length=0.03,
            width=0.008,
            length_includes_head=True,
            color="gray"
        )

    # Agent
    ax1.scatter(
        positions[0][0],
        positions[0][1],
        facecolors="black",
        edgecolors="black",
        linewidths=1,
        marker="s",
    )
    ax1.annotate("agent", positions[0] + text_delta)


if __name__ == "__main__":
    main()
