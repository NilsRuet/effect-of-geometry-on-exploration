"""
This file defines a simulation and its agent's behaviour
"""

import numpy as np
import time
from core.actions import ProjectiveTransformationFactory, Translation2DActionSpace
from core.agent import Agent, PerceptionSpace
from core.beliefs import Beliefs
from core.frame import ReferenceFrame
from core.loss import EpistemicLoss
from core.observations import ObjectSensor, MarkovKernel
from core.policy import ArgminWithEpsilonPolicy
from params import SimParams, BeliefSpaceParams
from utils.datamanager import dataManager
from utils.geometryutils import GeometryUtils
from utils.logger import Logger
from utils.rotationutils import RotationUtils


class Simulation:
    def _init_belief_space(self, id, factory: ProjectiveTransformationFactory, initial_translation, params: BeliefSpaceParams):
        angle = GeometryUtils.get_new_frame_rotation_angle(
            initial_translation, initial_translation, params.target
        )

        initial_rotation = RotationUtils.generate_rotation_matrix(angle)
        initial_reference_transformation = factory.createTransformation(
            initial_rotation, initial_translation
        )

        # Init sensor and frame
        frame = ReferenceFrame(initial_reference_transformation)
        noise_kernel = MarkovKernel(params.markov_kernel_epsilon)
        world = ObjectSensor(params.target, noise_kernel)

         # Beliefs are initialized with a mean at the "true" position in the internal world
        initial_object_position_internal = initial_reference_transformation.transform(
            params.target
        )

        initial_beliefs = Beliefs(
            initial_object_position_internal,
            params.initial_beliefs_covariance * np.identity(2),
            noise_kernel,
        )
        
        return PerceptionSpace(id, frame, world, initial_beliefs)

    def generate_distance_filter(self, radius):
        def filter_too_close(idle_translation, world_translations, observations):
            filtered = []
            radius_sqr = radius * radius
            for translation in world_translations:
                vecs = [obs - idle_translation for obs in observations]
                norms = [v[0] * v[0] + v[1] * v[1] for v in vecs]
                if(min(norms) > radius_sqr):
                    filtered.append(translation)
            return filtered

        return filter_too_close

    def _init_agent(self, params: SimParams):
        # Start with no translation
        initial_translation = np.array((0, 0))
        factory = ProjectiveTransformationFactory(gamma=params.gamma)

        # Create belief spaces
        belief_spaces = []
        for i, belief_space_param in enumerate(params.beliefs_spaces):
            belief_space = self._init_belief_space(i, factory, initial_translation, belief_space_param)
            belief_spaces.append(belief_space)

        # Action space, sampled for each target
        filter = self.generate_distance_filter(params.distance_filter)
        action_space = Translation2DActionSpace(
            factory,
            translation_norm=params.norm_of_translations,
            direction_count=params.translation_direction_count,
            agent_starting_position=-initial_translation,
            filter=filter
        )

        # Create loss and policy
        loss = EpistemicLoss()
        policy = ArgminWithEpsilonPolicy(action_space, loss, params.loss_epsilon)

        # Create and run agent
        return Agent(belief_spaces, policy)

    def run(self, params: SimParams):
        agent = self._init_agent(params)
        iteration = 0
        while iteration < params.max_steps:
            # Data tracking
            belief_space_states = agent.get_belief_states()

            t0 = time.time()
            agent_t = iteration * params.deltatime
            # Step
            policy_state = agent.step(agent_t)
            duration = time.time() - t0
            Logger.debug(f"execution: ~{int(duration * 1000)}ms")
            Logger.debug("-")

            # Notify data for the current step
            dataManager.notify_new_step(
                agent_t,
                belief_space_states,
                policy_state,
                duration
            )
            iteration += 1

        dataManager.notify_last_step(
            agent.get_belief_states()
        )
