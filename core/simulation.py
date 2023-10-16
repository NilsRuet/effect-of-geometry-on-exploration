"""
This file defines a simulation and its agent's behaviour
"""

import numpy as np
import time
from core.actions import ProjectiveTransformationFactory, Translation2DActionSpace
from core.beliefs import Beliefs
from core.frame import ReferenceFrame
from core.loss import EpistemicLoss
from core.observations import World, MarkovKernel
from params import SimParams
from core.policy import ArgminWithEpsilonPolicy
from utils.datamanager import dataManager
from utils.geometryutils import GeometryUtils
from utils.logger import Logger
from utils.rotationutils import RotationUtils


# Exploration algorithm
class Agent:
    def __init__(
        self,
        reference_frame: ReferenceFrame,
        world: World,
        initial_beliefs: Beliefs,
        policy: ArgminWithEpsilonPolicy,
    ):
        self.reference_frame = reference_frame
        self.world = world
        self.beliefs = initial_beliefs
        self.policy = policy

    def step(self, time):
        object_position = self.world.observe_position()

        losses, action_i, best_move, best_move_beliefs = self.policy.select(
            self.reference_frame.transformation, time, self.beliefs, object_position
        )

        self.reference_frame.update(best_move.phi_rm)
        self.beliefs: Beliefs = best_move_beliefs
        observation = self.reference_frame.world_to_local(self.world.observe_position())
        self.beliefs.update(observation)

        # Debug info
        Logger.debug(f"Step t = {time}")
        Logger.debug(
            "mean = ({:0.3f} {:0.3f})".format(
                self.beliefs.qx.mean[0], self.beliefs.qx.mean[1]
            )
        )
        Logger.debug(
            f"frame translation: {self.reference_frame.transformation.translation}"
        )
        return action_i, losses


class Simulation:
    def run(self, params: SimParams):
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
        world = World(params.object_position_in_world, noise_kernel)

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
        agent = Agent(agent_frame, world, initial_beliefs, policy)
        iteration = 0
        while iteration < params.max_steps:
            # Data tracking
            t = iteration * params.deltatime
            rotation = agent.reference_frame.transformation.linear_map
            translation = agent.reference_frame.transformation.translation
            beliefs = agent.beliefs
            obj_position = world.observe_position()

            t0 = time.time()
            # Step
            action_i, losses = agent.step(t)
            duration = time.time() - t0
            Logger.debug(f"execution: ~{int(duration * 1000)}ms")
            Logger.debug("-")

            # Notify data
            dataManager.notify_new_step(
                t,
                rotation,
                translation,
                beliefs,
                obj_position,
                action_i,
                losses,
                duration,
            )
            iteration += 1
