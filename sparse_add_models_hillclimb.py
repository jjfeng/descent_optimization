import sys
import scipy as sp
from scipy.sparse.linalg import spsolve, lsqr
from fitted_model import Fitted_Model
import numpy as np
import cvxpy

from common import *
from convexopt_solvers import SparseAdditiveModelProblemWrapper
from gradient_descent_algo import Gradient_Descent_Algo

class Sparse_Add_Model_Hillclimb(Gradient_Descent_Algo):
    method_label = "Sparse_Add_Model_Hillclimb"

    def _create_descent_settings(self):
        self.num_iters = 40
        self.step_size_init = 1
        self.step_size_min = 1e-5
        self.shrink_factor = 0.1
        self.decr_enough_threshold = 1e-2
        self.use_boundary = False
        self.boundary_factor = 0.999

    def _create_lambda_configs(self):
        self.lambda_mins = [1e-6] * (self.data.num_features + 1)

    def _create_problem_wrapper(self):
        self.problem_wrapper = SparseAdditiveModelProblemWrapper(
            self.data.X_full,
            self.data.train_idx,
            self.data.y_train
        )
        self.I_tiled = np.matrix(
            np.tile(np.eye(self.data.num_samples), (self.data.num_features, 1))
        )
        self.II = self.I_tiled * self.I_tiled.T

    def get_validate_cost(self, model_params):
        return testerror_sparse_add_smooth(
            self.data.y_validate,
            self.data.validate_idx,
            model_params
        )

    def _get_lambda_derivatives(self):
        nonzero_thetas_idx = self._get_nonzero_theta_indices(self.fmodel.current_model_params)

        dtheta_dlambda = self._get_dtheta_dlambda(self.fmodel.current_model_params[:, nonzero_thetas_idx])
        fitted_y_validate = np.sum(self.fmodel.current_model_params[self.data.validate_idx], axis=1)
        print dtheta_dlambda.shape
        dloss_dlambda = -1.0/self.data.num_validate * (
            dtheta_dlambda[self.data.validate_idx, :].T * (self.data.y_validate - fitted_y_validate)
        )
        return dloss_dlambda.A1 # flatten the matrix

    def _get_dtheta_dlambda(self, nonzero_thetas):
        def create_b_diag_elem(i):
            theta = nonzero_thetas[i,:]
            theta_norm = np.linalg.norm(theta, 2)
            b_diag_elem = 1/theta_norm * (np.eye(self.data.num_samples) - theta * theta.T/(theta_norm**2))
            return b_diag_elem

        def normalize_theta(i):
            theta = nonzero_thetas[:,i]
            theta_norm = np.linalg.norm(theta, 2)
            return theta/theta_norm

        def make_diag_dtheta(i):
            dd_theta = self.problem_wrapper.diff_matrices[i].T * np.sign(
                self.problem_wrapper.diff_matrices[i] * nonzero_thetas[:,i]
            )
            return dd_theta

        num_nonzero_thetas = nonzero_thetas.shape[1]
        b_diag_elems = map(create_b_diag_elem, range(num_nonzero_thetas))
        b_diag = sp.linalg.block_diag(*b_diag_elems)

        normalized_thetas = map(normalize_theta, range(num_nonzero_thetas))
        normalized_thetas = np.vstack(normalized_thetas)

        diag_dthetas = map(make_diag_dtheta, range(num_nonzero_thetas))
        diag_dthetas = sp.linalg.block_diag(*diag_dthetas)

        rhs_matrix = np.hstack((normalized_thetas, diag_dthetas))

        lambda0 = self.fmodel.current_lambdas[0]
        dtheta_dlambda, _, _, _ = np.linalg.lstsq(self.II + lambda0 * b_diag, -1 * rhs_matrix)
        return dtheta_dlambda

    def _double_check_derivative(self, calculated_derivative):
        return True

    @staticmethod
    def _get_nonzero_theta_indices(thetas, threshold=CLOSE_TO_ZERO_THRESHOLD):
        return map(lambda i: np.linalg.norm(thetas[:,i]) > threshold, range(thetas.shape[1]))
