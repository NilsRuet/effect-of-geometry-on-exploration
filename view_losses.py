"""
This script is used to compute and plot a figure about the losses of all simulations,
grouped by gamma value (coef used in projective transformations)
"""


import os
import jsonpickle
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 12})


class LossStats:
    def __init__(self):
        self.all_losses = {}

    def add_losses(self, gamma, losses):
        current_array = self.all_losses.get(gamma, [])
        current_array.append(losses)
        self.all_losses[gamma] = current_array

    def plot(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_box_aspect(1)
        ax1.set_xlabel("Translation direction", fontweight="bold")
        ax1.set_ylabel("Epistemic value", fontweight="bold")
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        # hardcoded for 8 directions
        idle_rewards = []

        ticks = [(i * np.pi / 4) for i in range(8)]

        tick_labels = ["-3π/4", "-π/2", "-π/4", "0", "π/4", "π/2", "3π/4", "π"]
        ax1.set_xticks(ticks)
        ax1.set_xticklabels(tick_labels)

        color_dic = {0: "gray", 0.5: "black", 1: "black"}
        default_color = "red"

        for gamma in self.all_losses.keys():
            losses = np.array(self.all_losses[gamma])
            # reshape in a total_steps x action_count array
            # negative to convert losses to rewards
            rewards = -losses.reshape((-1, losses.shape[-1]))
            idle_rewards.append(np.average(rewards[:, 0]))
            rewards = np.delete(rewards, 0, axis=1)
            rewards = np.roll(rewards, shift=1, axis=1)

            sample_count = rewards.shape[0]
            direction_count = rewards.shape[1]

            avg_rewards = np.average(rewards, axis=0)
            std_rewards = np.std(rewards, axis=0)
            std_errors = (1 / np.sqrt(sample_count)) * std_rewards
            bar_reduction_factor = 4
            bar_width = np.pi * 2 / (direction_count * bar_reduction_factor)
            bar_linewidth = 2

            for i, x in zip(range(direction_count), ticks):
                avg = avg_rewards[i]
                std_error = std_errors[i]
                left, right = (x - bar_width / 2, x + bar_width / 2)
                bot, top = (avg - std_error, avg + std_error)

                ax1.plot(
                    [x, x],
                    [bot, top],
                    c=color_dic.get(gamma, default_color),
                    zorder=0,
                    linewidth=bar_linewidth,
                )
                ax1.plot(
                    [left, right],
                    [top, top],
                    c=color_dic.get(gamma, default_color),
                    zorder=0,
                    linewidth=bar_linewidth,
                )
                ax1.plot(
                    [left, right],
                    [bot, bot],
                    c=color_dic.get(gamma, default_color),
                    zorder=0,
                    linewidth=bar_linewidth,
                )

            # plot line linking averages
            ax1.plot(
                ticks,
                avg_rewards,
                linewidth=1,
                c=color_dic.get(gamma, default_color),
                linestyle="dashed",
                marker="o",
                zorder=1,
            )
            ax1.scatter(
                ticks, avg_rewards, s=8, c=color_dic.get(gamma, default_color), zorder=1
            )

        plt.show()


def main():
    stats = LossStats()
    folder = "./sims/"
    files = os.listdir(folder)
    for i, simfile in enumerate(files):
        print(f"Reading file {i+1}/{len(files)}")
        path = os.path.join(folder, simfile)
        with open(path, "r") as file:
            content = file.read()
            current_sim = jsonpickle.decode(content)

        # Get and compute the relevant stats about the current sim
        gamma = current_sim["params"]["gamma"]
        object_pos = current_sim["params"]["object_position_in_world"]["values"]
        distance = np.sqrt(
            object_pos[0] * object_pos[0] + object_pos[1] * object_pos[1]
        )
        # Plotting is disabled for positions where the agent might have planned an action that makes the object go behind it
        if distance > 0.11:
            steps = current_sim["steps"]

            # Agent positions
            losses = np.array([np.array(s["losses"]) for s in steps])
            stats.add_losses(gamma, losses)

    stats.plot()


if __name__ == "__main__":
    main()
