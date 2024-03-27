"""
This file contains classes used to save data from the simulations.
It also inits what should be the single instance of the data manager.
"""

import time
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_np
import os
from core.states import BeliefState, PolicyState, ActionState
from params import SimParams
from utils.logger import Logger

jsonpickle_np.register_handlers()


class _SimBeliefData:
    def __init__(self, belief_state: BeliefState):
        self.frame_rotation = belief_state.rotation.tolist()
        self.frame_translation = belief_state.translation.tolist()
        self.beliefs_mean = belief_state.beliefs.qx.mean.tolist()
        self.beliefs_cov = belief_state.beliefs.qx.cov.tolist()
        self.object_pos = belief_state.obj_position.tolist()


class _SimActionData:
    def __init__(self, action: ActionState):
        self.id = int(action.id)
        self.translation = action.translation.tolist()


class _SimPolicyData:
    def __init__(self, policy_state: PolicyState):
        self.losses = policy_state.losses.tolist()
        self.loss_per_space = [loss.tolist() for loss in policy_state.loss_per_space]
        self.chosen_action = _SimActionData(policy_state.chosen_action)


class _SimStepData:
    def __init__(
        self,
        time,
        belief_space_states,
        policy_state,
        duration,
    ):
        self.t = time
        self.states = [_SimBeliefData(state) for state in belief_space_states]
        self.policy = _SimPolicyData(policy_state)
        self.real_time_duration = duration


class _SimData:
    def __init__(self, params):
        self.duration: float = None
        self.params: SimParams = params
        self.steps = []
        self.final_state = []

    def add_step(self, data: _SimStepData):
        self.steps.append(data)

    def set_duration(self, duration):
        self.duration = duration

    def add_final_state(self, state: list[_SimBeliefData]):
        self.final_state = state


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

    def notify_last_step(self, belief_space_states: list[BeliefState]):
        state = [_SimBeliefData(s) for s in belief_space_states]
        self.current_sim_data.add_final_state(state)

    def notify_new_step(
        self,
        time,
        belief_space_states,
        policy_state,
        duration,
    ):
        self.current_step = _SimStepData(
            time,
            belief_space_states,
            policy_state,
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
        name = "sim{:02d}_gamma{:0.2f}_norm{:0.2f}.json"
        return name.format(
            sim_number,
            params.gamma,
            params.norm_of_translations,
        )


# This can be imported and be used as a singleton instance
dataManager = SimDataManager()
