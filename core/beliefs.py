"""
This file defines an agent beliefs about the object in its internal world,
as well as how they are affected by actions/observations
"""

import numpy as np
import scipy
import scipy.integrate
import scipy.stats
from typing import List
from core.actions import ProjectiveAction
from core.observations import MarkovKernel
from params import FixedParams
from utils.geometryutils import GeometryUtils
from utils.logger import Logger


# Beliefs about the position of the object in the internal representation
# Also holds information about the observations
class Beliefs:
    zero_covariance_threshold = 1e-8

    def __init__(self, mean, covariance, observation_markov_kernel: MarkovKernel):
        self.observation_kernel = observation_markov_kernel
        self._set_mean_and_cov(mean, covariance)

    def _set_mean_and_cov(self, mean, covariance):
        # Beliefs
        self.qx = scipy.stats.multivariate_normal(mean, covariance)

        # Observations
        cov_yy = (
            self.observation_kernel.epsilon
            * self.observation_kernel.epsilon
            * np.identity(2)
            + covariance
        )
        self.py = scipy.stats.multivariate_normal(mean, cov_yy)

        # Joint distribution of observations and beliefs
        mean_xy = (*mean, *mean)
        left = np.vstack((covariance, covariance))
        right = np.vstack((covariance, cov_yy))
        sigma_xy = np.hstack((left, right))
        self.pxy = scipy.stats.multivariate_normal(mean_xy, sigma_xy)

    # Computes new beliefs resulting from the given actions
    def propagate_actions(self, actions: List[ProjectiveAction]):
        predicted_beliefs = []
        # Integrate the transformed distribution and approximate it by a gaussian distribution
        for i_action, action in enumerate(actions):
            Logger.progress(f"Action {i_action+1}/{len(actions)}")
            # Skip the integration if the distribution is basically a single point
            if (
                self.qx.cov[0, 0] <= Beliefs.zero_covariance_threshold
                and self.qx.cov[1, 1] <= Beliefs.zero_covariance_threshold
            ):
                mean = action.transform(self.qx.mean)
                covariance = Beliefs.zero_covariance_threshold * np.identity(2)
                Logger.warning("covariance rounded to 0")
            else:
                bounds = self._get_integration_bounds(action)
                mean = self._integrate_mean(action, *bounds)
                covariance = self._integrate_covariance(action, mean, *bounds)

            predicted_beliefs.append(Beliefs(mean, covariance, self.observation_kernel))

        return np.array(predicted_beliefs)

    # Updates beliefs based on an observation
    # see "conditional distributions" https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    def update(self, observation: np.ndarray):
        # These values come from the way sigma_xy is defined
        sigma_xy_11 = self.qx.cov
        sigma_xy_12 = self.qx.cov
        sigma_xy_21 = self.qx.cov
        sigma_xy_22 = self.py.cov

        inverted_22 = np.linalg.inv(sigma_xy_22)
        regression_coefs = np.matmul(sigma_xy_12, inverted_22)
        # mean of both X and Y is mu_x
        new_mean = self.qx.mean + np.matmul(
            regression_coefs, (observation - self.qx.mean)
        )
        new_covariance = sigma_xy_11 - np.matmul(regression_coefs, sigma_xy_21)
        self._set_mean_and_cov(new_mean, new_covariance)

    # Computes the new mean (x, y) after an action
    def _integrate_mean(self, action, x1, x2, lambda_y1, lambda_y2):
        # scipy only allows to integrate functions with a single output, therefore x and y are integrated separately
        x_estimate, _ = scipy.integrate.dblquad(
            lambda y, x: self._mean_function(action, np.array((x, y)), axis=0),
            x1,
            x2,
            lambda_y1,
            lambda_y2,
            epsabs=FixedParams.integration_epsabs,
            epsrel=FixedParams.integration_epsrel,
        )

        y_estimate, _ = scipy.integrate.dblquad(
            lambda y, x: self._mean_function(action, np.array((x, y)), axis=1),
            x1,
            x2,
            lambda_y1,
            lambda_y2,
            epsabs=FixedParams.integration_epsabs,
            epsrel=FixedParams.integration_epsrel,
        )

        return np.array((x_estimate, y_estimate))

    # Computes the new covariance after an action
    def _integrate_covariance(self, action, mean, x1, x2, lambda_y1, lambda_y2):
        dims = 2
        cov = np.identity(dims)

        for i in range(dims):
            for j in range(i + 1):
                cov_ij, _ = scipy.integrate.dblquad(
                    lambda y, x: self._covariance_function(
                        action, np.array((x, y)), i, j, mean[i], mean[j]
                    ),
                    x1,
                    x2,
                    lambda_y1,
                    lambda_y2,
                    epsabs=FixedParams.integration_epsabs,
                    epsrel=FixedParams.integration_epsrel,
                )
                cov[i, j] = cov_ij
                cov[j, i] = cov_ij

        return cov

    # Computes a quad on which the new mean/covariance will be integrated after an action
    def _get_integration_bounds(
        self,
        action: ProjectiveAction,
        std_deviation_count=FixedParams.integration_std_count,
    ):
        # Select a rectangle around the mean based on covariance
        x_range = np.sqrt(self.qx.cov[0, 0]) * std_deviation_count
        y_range = np.sqrt(self.qx.cov[1, 1]) * std_deviation_count
        mean_x = self.qx.mean[0]
        mean_y = self.qx.mean[1]
        x1, y1, x2, y2 = (
            mean_x - x_range,
            mean_y - y_range,
            mean_x + x_range,
            mean_y + y_range,
        )
        bounds_points = np.array(((x1, y1), (x1, y2), (x2, y2), (x2, y1)))

        # Transform the rectangle based on the action
        bounds_points = np.apply_along_axis(
            lambda p: action.transform(p), arr=bounds_points, axis=1
        )

        # Generate the bounds for scipy
        x1 = min(bounds_points, key=lambda p: p[0])[0]
        x2 = max(bounds_points, key=lambda p: p[0])[0]
        lambda_y1 = lambda x: GeometryUtils.find_intersections(bounds_points, x)[0]
        lambda_y2 = lambda x: GeometryUtils.find_intersections(bounds_points, x)[1]
        return x1, x2, lambda_y1, lambda_y2

    # Function that is integrated to compute the mean
    def _mean_function(self, action: ProjectiveAction, vec, axis=0):
        vec_before_transform = action.inverse_transform(vec)
        jacobian_det_inverse = 1 / action.jacobian_det(vec_before_transform)

        # Axis is a trick to be able to integrate x and y of a vector (x, y) separately
        res = (
            self.qx.pdf(vec_before_transform) * vec[axis] * np.abs(jacobian_det_inverse)
        )
        return res

    # Function that is integrated to compute the covariance between two variables
    def _covariance_function(
        self, action: ProjectiveAction, vec, var_i, var_j, mean_i, mean_j
    ):
        vec_before_transform = action.inverse_transform(vec)
        jacobian_det = 1 / action.jacobian_det(vec_before_transform)
        res = (
            self.qx.pdf(vec_before_transform)
            * np.abs(jacobian_det)
            * (vec[var_i] - mean_i)
            * (vec[var_j] - mean_j)
        )
        return res

    def __repr__(self):
        return f"({self.qx.mean},\n{self.qx.cov})"
