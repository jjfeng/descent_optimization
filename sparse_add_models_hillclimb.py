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
        self.use_boundary = True
        self.boundary_factor = 0.999

    def _create_lambda_configs(self):
        self.lambda_mins = [1e-6] * self.data.num_features

    def _create_problem_wrapper(self):
        self.problem_wrapper = SparseAdditiveModelProblemWrapper(
            self.data.X_full,
            self.data.train_idx,
            self.data.y_train
        )
        # self.II = np.eye(self.data.num_features) * 

    def get_validate_cost(self, model_params):
        return testerror_sparse_add_smooth(
            self.data.y_validate,
            self.data.validate_idx,
            model_params
        )

    def _get_lambda_derivatives(self):
        return 0

    def _double_check_derivative(self, calculated_derivative):
        return True
