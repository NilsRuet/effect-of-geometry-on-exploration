"""
This script is used to view the average, max, and min loss range within a single step per gamma
"""

import os
import numpy as np
import jsonpickle


class LossRangeStats:
    def __init__(self):
        self.delta_loss_sum = 0
        self.count = 0
        self.min_delta_loss = 9999999
        self.max_delta_loss = 0

    def add_stats(self, avg_delta_loss, min_delta_loss, max_delta_loss):
        self.delta_loss_sum += avg_delta_loss
        self.min_delta_loss = min(self.min_delta_loss, min_delta_loss)
        self.max_delta_loss = max(self.max_delta_loss, max_delta_loss)
        self.count += 1

    def compute(self):
        return (
            self.delta_loss_sum / self.count,
            self.min_delta_loss,
            self.max_delta_loss,
        )


def main():
    delta_loss_dic = {}

    folder = "./sims/"
    files = os.listdir(folder)
    for simfile in files:
        path = os.path.join(folder, simfile)
        with open(path, "r") as file:
            content = file.read()
            deserialized = jsonpickle.decode(content)

            gamma, avg_delta_loss, min_delta_loss, max_delta_loss = compute_stats(
                deserialized
            )

            stats = delta_loss_dic.get(gamma, LossRangeStats())
            stats.add_stats(avg_delta_loss, min_delta_loss, max_delta_loss)
            delta_loss_dic[gamma] = stats

    global_stats = [(k, delta_loss_dic[k].compute()) for k in delta_loss_dic.keys()]
    for gamma, (avg_delta_loss, min_delta_loss, max_delta_loss) in global_stats:
        print(
            f"gamma = {gamma} -> average {avg_delta_loss:.2E} ({min_delta_loss:.2E} -> {max_delta_loss:.2E})"
        )


def compute_stats(simulation_dictionary):
    object_position = simulation_dictionary["params"]["object_position_in_world"][
        "values"
    ]
    gamma = simulation_dictionary["params"]["gamma"]
    last_step = simulation_dictionary["steps"][-1]
    last_rotation = np.array(last_step["agent_frame_rotation"])
    end_translation = last_step["agent_frame_translation"]
    absolute_pos = -(np.matmul(np.linalg.inv(last_rotation), end_translation))
    distance = np.sqrt(np.sum(np.square(object_position - absolute_pos)))
    total = simulation_dictionary["duration"]
    average_duration = simulation_dictionary["duration"] / len(
        simulation_dictionary["steps"]
    )

    steps = simulation_dictionary["steps"]
    losses = np.array([s["losses"] for s in steps])
    delta_loss = np.max(losses, axis=1) - np.min(losses, axis=1)
    print("--")
    print(
        f"avg step duration : {int(average_duration * 1000)}ms, total duration : {int(total)}s, gamma {gamma:.2f}, final distance {distance:.2f}"
    )
    print(
        f"loss delta avg : {np.average(delta_loss):.2E}, min : {np.min(delta_loss):.2E}, max : {np.max(delta_loss):.2E}"
    )
    return gamma, np.average(delta_loss), np.min(delta_loss), np.max(delta_loss)


if __name__ == "__main__":
    main()
