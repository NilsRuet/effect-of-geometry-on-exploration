"""
This file contains classes used to save data from the simulations.
It also inits what should be the single instance of the data manager.
"""

import time
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_np
import os
from params import SimParams
from utils.logger import Logger

jsonpickle_np.register_handlers()


class _SimStepData:
    def __init__(
        self,
        time,
        rotation,
        translation,
        beliefs,
        object_position,
        chosen_action,
        losses,
        duration,
    ):
        self.t = time
        self.agent_frame_rotation = rotation.tolist()
        self.agent_frame_translation = translation.tolist()
        self.agent_beliefs_mean = beliefs.qx.mean.tolist()
        self.agent_beliefs_cov = beliefs.qx.cov.tolist()
        self.object_pos = object_position.tolist()
        self.losses = losses.tolist()
        self.selected_action = chosen_action.item()
        self.real_time_duration = duration


class _SimData:
    def __init__(self, params):
        self.duration: float = None
        self.params: SimParams = params
        self.steps = []

    def add_step(self, data: _SimStepData):
        self.steps.append(data)

    def set_duration(self, duration):
        self.duration = duration


# Records and writes data to storage after being notified of certain events
class SimDataManager:
    def __init__(self):
        self.sim_folder = "./sims/"
        self.count = 0

    def set_sim_folder(self, path):
        self.sim_folder = path

    def notify_new_sim(self, params: SimParams):
        self.count += 1
        self.current_sim_data = _SimData(params)

    def notify_new_step(
        self,
        time,
        rotation,
        translation,
        beliefs,
        object_position,
        chosen_action,
        losses,
        duration,
    ):
        self.current_step = _SimStepData(
            time,
            rotation,
            translation,
            beliefs,
            object_position,
            chosen_action,
            losses,
            duration,
        )
        self.current_sim_data.add_step(self.current_step)

    def notify_sim_end(self, duration):
        self.current_sim_data.set_duration(duration)
        self._write_current_sim(self.count)

    def _write_current_sim(self, number):
        t0 = time.time()
        filename = self._get_name(self.current_sim_data.params, number)
        path = os.path.join(self.sim_folder, filename)
        jsonpickle.set_encoder_options("json", indent=4)
        content = jsonpickle.encode(self.current_sim_data, unpicklable=False)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w+") as file:
            file.write(content)

        Logger.debug(f"File writing : {int((time.time() - t0) * 1000)}ms")

    def _get_name(self, params: SimParams, sim_number):
        name = "sim{:02d}_gamma{:0.2f}_eps{:0.2f}_obj{:0.2f},{:0.2f}_norm{:0.2f}.json"
        return name.format(
            sim_number,
            params.gamma,
            params.markov_kernel_epsilon,
            params.object_position_in_world[0],
            params.object_position_in_world[1],
            params.norm_of_translations,
        )


# This can be imported and be used as a singleton instance
dataManager = SimDataManager()
