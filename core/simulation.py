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
                agent_t, belief_space_states, policy_state, duration
            )
            iteration += 1

        dataManager.notify_last_step(agent.get_belief_states())

    def _init_belief_space(
        self,
        id,
        factory: ProjectiveTransformationFactory,
        initial_translation,
        initial_eccentricity,
        initial_distance,
        params: BeliefSpaceParams,
    ):
        angle = GeometryUtils.get_new_frame_rotation_angle(
            initial_translation, initial_translation, params.target
        )

        initial_rotation = RotationUtils.generate_rotation_matrix(angle)
        initial_reference_transformation = factory.createTransformation(
            initial_rotation, initial_translation
        )

        # Init sensor and frame
        frame = ReferenceFrame(initial_reference_transformation)
        world = ObjectSensor(params.target)

        # Beliefs are initialized with a mean at the "true" position in the internal world
        initial_object_position_internal = initial_reference_transformation.transform(
            params.target
        )

        kernel_generator = self.get_kernel_generator(
            params.initial_kernel_epsilon, params.acuity_coef, params.distance_coef
        )

        initial_beliefs = Beliefs(
            initial_object_position_internal,
            params.initial_beliefs_covariance * np.identity(2),
            kernel_generator(initial_eccentricity, initial_distance),
        )

        return PerceptionSpace(id, frame, world, initial_beliefs, kernel_generator)

    def _init_agent(self, params: SimParams):
        # Start with no translation
        initial_translation = np.array((0, 0))
        factory = ProjectiveTransformationFactory(gamma=params.gamma)

        # The agent starts facing an arbitrary direction (it doesn't matter as the initial beliefs are not updated using an observation)
        initial_position = -initial_translation
        initial_forward = initial_position+np.array((0,1))
        eccentricities = [abs(GeometryUtils.get_angle(initial_forward, initial_position, space.target)) for space in params.beliefs_spaces]
        distances = [np.linalg.norm(space.target - initial_position) for space in params.beliefs_spaces]

        # Create belief spaces
        belief_spaces = []
        for i, belief_space_param, initial_eccentricity, initial_distance in zip(range(len(params.beliefs_spaces)), params.beliefs_spaces, eccentricities, distances):
            belief_space = self._init_belief_space(
                i,
                factory,
                initial_translation,
                initial_eccentricity,
                initial_distance,
                belief_space_param,
            )
            belief_spaces.append(belief_space)

        # Action space, sampled for each target
        filter = self.generate_distance_filter(params.distance_filter)
        action_space = Translation2DActionSpace(
            factory,
            translation_norm=params.norm_of_translations,
            direction_count=params.translation_direction_count,
            agent_starting_position=-initial_translation,
            filter=filter,
        )

        # Create loss and policy
        loss = EpistemicLoss()
        policy = ArgminWithEpsilonPolicy(
            loss,
            params.loss_epsilon,
            params.default_on_illegal,
            params.merge_loss_by_min,
        )

        # Create and run agent
        return Agent(belief_spaces, action_space, policy)

    def generate_distance_filter(self, radius):
        def filter_too_close(world_positions, observations):
            valid = []
            radius_sqr = radius * radius
            for position in world_positions:
                vecs = [obs - position for obs in observations]
                norms = [v[0] * v[0] + v[1] * v[1] for v in vecs]
                valid.append(min(norms) > radius_sqr)
            return valid

        return filter_too_close

    def get_kernel_generator(
        self, initial_markov_epsilon, acuity_coef, distance_coef, min_variance=0.05
    ):
        def kernel_generator(eccentricity, distance):
            acuity = np.exp(-acuity_coef * eccentricity)
            certainty = distance_coef * acuity / (distance)
            return MarkovKernel(initial_markov_epsilon * np.maximum(1 - certainty, min_variance))

        return kernel_generator
