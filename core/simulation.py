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
from params import SimParams
from utils.datamanager import dataManager
from utils.geometryutils import GeometryUtils
from utils.logger import Logger
from utils.rotationutils import RotationUtils


class Simulation:
    def _init_agent(self, params):
        factory = ProjectiveTransformationFactory(gamma=params.gamma)

        # Start rotated toward the object and with no translation
        initial_translation = np.array((0, 0))
        angle = GeometryUtils.get_new_frame_rotation_angle(
            initial_translation, initial_translation, params.object_position_in_world
        )
        initial_rotation = RotationUtils.generate_rotation_matrix(angle)
        initial_reference_transformation = factory.createTransformation(
            initial_rotation, initial_translation
        )

        # action_space = Rotation2DActionSpace(factory, params.min_angle, params.max_angle, params.angle_count)
        action_space = Translation2DActionSpace(
            factory,
            translation_norm=params.norm_of_translations,
            direction_count=params.translation_direction_count,
            agent_starting_position=-initial_translation,
        )

        # Init world and agent frames
        agent_frame = ReferenceFrame(initial_reference_transformation)
        noise_kernel = MarkovKernel(params.markov_kernel_epsilon)
        world = ObjectSensor(params.object_position_in_world, noise_kernel)

        # Beliefs are initialized with a mean at the "true" position in the internal world
        initial_object_position_internal = initial_reference_transformation.transform(
            params.object_position_in_world
        )
        initial_beliefs = Beliefs(
            initial_object_position_internal,
            params.initial_beliefs_covariance * np.identity(2),
            noise_kernel,
        )

        # Create loss and policy
        # loss = SquaredComponentLoss(component_index = 1) # The loss of a vector (x0, x1) is x1 squared
        loss = EpistemicLoss()
        policy = ArgminWithEpsilonPolicy(action_space, loss, params.loss_epsilon)

        # Create and run agent
        # TODO : multiple objects
        space = PerceptionSpace(1, agent_frame, world, initial_beliefs)
        return Agent([space], policy)

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
