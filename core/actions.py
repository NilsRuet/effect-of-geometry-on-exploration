"""
These classes specify how projective actions are defined and computed.
"""

import numpy as np
import sympy
import sympy.abc as symbols
from sympy import Matrix
from utils.geometryutils import GeometryUtils
from utils.rotationutils import RotationUtils


# Factory for projective transformations
# Precomputes expressions common to all projective transformations
class ProjectiveTransformationFactory:
    def __init__(self, gamma):
        self.gamma = gamma
        self._init_functions()

    # Defines and lambdifies the expressions to compute a projective functions and its jacobian
    def _init_functions(self):
        # Be aware that the name of the symbols impacts the order of parameters of the python functions
        # See below
        vector = Matrix([[symbols.A], [symbols.B]])
        linear_map = Matrix([[symbols.a, symbols.b], [symbols.c, symbols.d]])
        translation = Matrix([[symbols.e], [symbols.f]])

        # Args for the lambdified python functions
        # This sort guarantees the order of parameters to be alphabetical.
        # Choosing properly the symbols of the sympy expression then gives a meaningful order of arguments
        args = [
            *vector.free_symbols,
            *linear_map.free_symbols,
            *translation.free_symbols,
        ]
        args.sort(key=lambda symbol: str(symbol))

        # Compute transformation and jacobian
        vec_after_affine = linear_map.multiply(vector) + translation
        phi: Matrix = vec_after_affine / (self.gamma * vec_after_affine[-1] + 1)
        jac: Matrix = phi.jacobian(vector)

        # Convert sympy expressions to python lambda functions
        self.phi = sympy.lambdify(args, phi)
        self.det = sympy.lambdify(args, jac.det())

    # Used to generate a specific transformation
    def createTransformation(
        self, linear_map: np.ndarray, translation=np.array((0, 0))
    ):
        return ProjectiveTransformation(
            linear_map, translation, self.gamma, self.phi, self.det
        )


# Projective transformation that maps from world to internal representation
class ProjectiveTransformation:
    def __init__(
        self,
        linear_map: np.ndarray,
        translation: np.ndarray,
        gamma,
        phi_function,
        det_function,
    ):
        self.linear_map = linear_map
        self.translation = translation
        self.inverse_linear_map = np.linalg.inv(self.linear_map)
        self.gamma = gamma
        self.phi_function = phi_function
        self.det_function = det_function

    def transform(self, vector: np.ndarray):
        # Call the lambdified sympy expression
        transformed = self.phi_function(
            *vector.reshape((-1)),
            *self.linear_map.reshape((-1)),
            *self.translation.reshape((-1)),
        )
        # the sympy lambda returns a nx1 matrix, which we convert to a numpy array
        return np.array(transformed).reshape((-1))

    def inverse_transform(self, vector: np.ndarray):
        # TODO : should this be a sympy expression too ?
        divider = vector[-1] / (1 - vector[-1] * self.gamma)
        unprojected = vector * (self.gamma * divider + 1)
        untranslated = unprojected - self.translation
        res = np.matmul(self.inverse_linear_map, untranslated)
        return res

    def jacobian_det(self, vector: np.ndarray):
        # Call the lambdified sympy expression
        return self.det_function(
            *vector.reshape((-1)),
            *self.linear_map.reshape((-1)),
            *self.translation.reshape((-1)),
        )

    def __repr__(self):
        return f"({self.linear_map},\n{self.translation})"


# Composition of two projective transformations (a change of frame)
class ProjectiveAction:
    # phi_r : the projective transformation that maps the world to the current frame
    # phi_rm : the projective transformation that maps the world to a new reference frame
    def __init__(
        self, phi_r: ProjectiveTransformation, phi_rm: ProjectiveTransformation
    ):
        self.phi_r = phi_r
        self.phi_rm = phi_rm

    def transform(self, vector):
        world_vec = self.phi_r.inverse_transform(vector)
        return self.phi_rm.transform(world_vec)

    def inverse_transform(self, vector):
        world_vec = self.phi_rm.inverse_transform(vector)
        res = self.phi_r.transform(world_vec)
        return res

    def jacobian_det(self, vector):
        vec_in_world = self.phi_r.inverse_transform(vector)
        det_phi_r = 1 / self.phi_r.jacobian_det(vec_in_world)
        det_phi_rm = self.phi_rm.jacobian_det(vec_in_world)
        return det_phi_r * det_phi_rm


# Used to generate agent moves based on the agent reference frame
class Translation2DActionSpace:
    def __init__(
        self,
        factory: ProjectiveTransformationFactory,
        translation_norm,
        direction_count,
        agent_starting_position,
    ):
        self.factory = factory
        self.translation_norm = translation_norm
        self.direction_count = direction_count
        self.agent_starting_position = agent_starting_position

    def sample(
        self,
        current_frame_transformation: ProjectiveTransformation,
        object_position: np.ndarray,
        time: float,
    ):
        angle_delta = 2 * np.pi / self.direction_count
        potential_directions = [(i * angle_delta) for i in range(self.direction_count)]

        # Current translation (in the current reference frame)
        current_frame_translation = current_frame_transformation.translation

        # Translations can be in any direction
        translations = [
            self.translation_norm * -np.array((np.cos(angle), np.sin(angle)))
            + current_frame_translation
            for angle in potential_directions
        ]

        # Add the identity translation at the beginning
        translations.insert(0, current_frame_translation)

        # Convert translations into world translations
        translations = [
            np.matmul(current_frame_transformation.inverse_linear_map, t)
            for t in translations
        ]

        # Generate new rotations for each new translation
        angles = [
            GeometryUtils.get_new_frame_rotation_angle(
                self.agent_starting_position, translation, object_position
            )
            for translation in translations
        ]
        rotations = [RotationUtils.generate_rotation_matrix(a) for a in angles]

        # Change the frame of each translation
        translations = [
            np.matmul(rotation, translation)
            for rotation, translation in zip(rotations, translations)
        ]

        # Create the new projections
        new_projections = np.array(
            [
                self.factory.createTransformation(rotation, translation)
                for rotation, translation in zip(rotations, translations)
            ]
        )

        actions = np.array(
            [
                ProjectiveAction(current_frame_transformation, projection)
                for projection in new_projections
            ]
        )
        return actions, np.int32(0)
