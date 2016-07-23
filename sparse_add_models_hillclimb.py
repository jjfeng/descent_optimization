import sys
import scipy as sp
from scipy.sparse.linalg import spsolve, lsqr
from fitted_model import Fitted_Model
import numpy as np
import cvxpy

from common import *
from convexopt_solvers import SparseAdditiveModelProblemWrapper
from gradient_descent_algo import Gradient_Descent_Algo

class BetaForm:
    eps = 1e-7

    def __init__(self, idx, theta, diff_matrix):
        self.idx = idx
        self.theta = theta
        self.theta_norm = np.linalg.norm(theta, 2)
        self.diff_matrix = diff_matrix

        print "diff_matrix * theta", diff_matrix * theta

        # Find the null space for the subsetted diff matrix
        zero_theta_idx = self._get_zero_theta_indices(diff_matrix * theta)
        print "zero_theta_idx", zero_theta_idx
        inflatedD = np.zeros(diff_matrix.shape)
        inflatedD[:np.sum(zero_theta_idx),:] = diff_matrix[zero_theta_idx,:]
        u, s, v = sp.linalg.svd(inflatedD)
        null_mask = s <= self.eps
        null_space = sp.compress(null_mask, v, axis=0)
        null_matrix = sp.transpose(null_space)
        beta, res, _, _ = np.linalg.lstsq(null_matrix, theta)

        self.beta = np.matrix(beta)
        self.u = np.matrix(null_matrix)

        # Check that we reformulated theta but it is still very close to the original theta
        assert(res.size == 0 or res < self.eps)
        print "reformualtion difference", np.linalg.norm(self.u * self.beta - self.theta, ord=2)
        assert(np.linalg.norm(self.u * self.beta - self.theta, ord=2) < self.eps)

    def __str__(self):
        return "beta %s, theta %s" % (self.beta, self.theta)

    @staticmethod
    def _get_zero_theta_indices(theta, threshold=1e-10):
        return np.less(np.abs(theta), threshold).A1

class Sparse_Add_Model_Hillclimb(Gradient_Descent_Algo):
    method_label = "Sparse_Add_Model_Hillclimb"

    def _create_descent_settings(self):
        self.num_iters = 40
        self.step_size_init = 1
        self.step_size_min = 1e-5
        self.shrink_factor = 0.1
        self.decr_enough_threshold = 1e-5
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

    def get_validate_cost(self, model_params):
        return testerror_sparse_add_smooth(
            self.data.y_validate,
            self.data.validate_idx,
            model_params
        )

    def _get_lambda_derivatives(self):
        # First filter out the thetas that are completely zero
        nonzero_thetas_idx = self._get_nonzero_theta_vectors(self.fmodel.current_model_params)

        # Now reformulate the remaining thetas using the differentiable space
        nonzeros_idx = np.where(nonzero_thetas_idx)[0]
        beta_u_forms = map(
            lambda i: BetaForm(i, self.fmodel.current_model_params[:,i], self.problem_wrapper.diff_matrices[i]),
            nonzeros_idx
        )
        sum_dtheta_dlambda = self._get_sum_dtheta_dlambda(beta_u_forms, nonzero_thetas_idx)
        print "sum_dtheta_dlambda", sum_dtheta_dlambda.shape
        fitted_y_validate = np.sum(self.fmodel.current_model_params[self.data.validate_idx, :], axis=1)
        print "fitted_y_validate", fitted_y_validate.shape
        dloss_dlambda = -1 * sum_dtheta_dlambda[self.data.validate_idx, :].T * (self.data.y_validate - fitted_y_validate)
        print "dloss_dlambda", dloss_dlambda
        return dloss_dlambda.A1 # flatten the matrix

    def _get_sum_dtheta_dlambda(self, beta_u_forms, nonzero_thetas_idx):
        def create_b_diag_elem(i):
            u = beta_u_forms[i].u
            beta = beta_u_forms[i].beta
            theta_norm = beta_u_forms[i].theta_norm
            b_diag_elem = 1.0/theta_norm * u.T * u * (
                np.eye(beta.size) - beta * beta.T * u.T * u/(theta_norm**2)
            )
            return b_diag_elem

        def make_rhs_col1(i):
            u = beta_u_forms[i].u
            beta = beta_u_forms[i].beta
            theta_norm = beta_u_forms[i].theta_norm
            return u.T * u * beta/theta_norm

        def make_diag_rhs(i):
            u = beta_u_forms[i].u
            beta = beta_u_forms[i].beta
            diff_matrix = beta_u_forms[i].diff_matrix
            theta_norm = beta_u_forms[i].theta_norm
            return u.T * diff_matrix.T * np.sign(diff_matrix * u * beta)

        num_nonzero_thetas = len(beta_u_forms)
        b_diag_elems = map(create_b_diag_elem, range(num_nonzero_thetas))
        print "b_diag_elems", b_diag_elems
        b_diag = sp.linalg.block_diag(*b_diag_elems)

        rhs_col1 = map(make_rhs_col1, range(num_nonzero_thetas))
        rhs_col1 = np.vstack(rhs_col1)
        print "rhs_col1", rhs_col1

        rhs_diag = map(make_diag_rhs, range(num_nonzero_thetas))
        rhs_diag = sp.linalg.block_diag(*rhs_diag)
        print "rhs_diag BEFORE", rhs_diag
        print "nonzero_thetas_idx", nonzero_thetas_idx
        print "num_features range", np.arange(self.data.num_features)
        print "num_features subset?", np.arange(self.data.num_features)[~nonzero_thetas_idx]
        # insert zero columns that corresponded to the zero thetas
        rhs_diag = np.insert(rhs_diag, np.arange(self.data.num_features)[~nonzero_thetas_idx], np.zeros((rhs_diag.shape[0], 1)), axis=1)
        print "rhs_diag INSERT", rhs_diag

        rhs_matrix = np.hstack((rhs_col1, rhs_diag))
        print "rhs_matrix", rhs_matrix

        lambda0 = self.fmodel.current_lambdas[0]
        u_matrices = map(lambda i: beta_u_forms[i].u, range(num_nonzero_thetas))
        print "u_matrices", u_matrices
        u_matrices = np.hstack(u_matrices)
        uu = u_matrices.T * u_matrices
        assert(uu.shape == b_diag.shape)
        dbeta_dlambda, _, _, _ = np.linalg.lstsq(uu + lambda0 * b_diag, -1 * rhs_matrix)
        sum_dtheta_dlambda = u_matrices * dbeta_dlambda
        print "sum_dtheta_dlambda", sum_dtheta_dlambda
        return sum_dtheta_dlambda

    def _double_check_derivative(self, calculated_derivative, accept_diff=1e-3, epsilon=1e-4):
        print "double_check_derivative"
        deriv = []
        num_lambdas = len(self.fmodel.current_lambdas)
        for i in range(num_lambdas):
            print "===========CHECK I= %d ===============" % i
            reg1 = [r for r in self.fmodel.current_lambdas]
            reg1[i] += epsilon
            thetas1 = self.problem_wrapper.solve(reg1)
            error1 = self.get_validate_cost(thetas1)

            reg2 = [r for r in self.fmodel.current_lambdas]
            reg2[i] -= epsilon
            thetas2 = self.problem_wrapper.solve(reg2)
            error2 = self.get_validate_cost(thetas2)
            i_deriv = (error1 - error2)/(epsilon * 2)
            # print "thetas1 norm", np.linalg.norm(thetas1[:,0]), thetas1[:,0]
            # print "thetas2 norm", np.linalg.norm(thetas2[:,0]), thetas2[:,0]
            # print "(thetas1 - thetas2)/(epsilon * 2)", (thetas1 - thetas2)/(epsilon * 2)
            print "numerical sum_dthetas_dlambda", np.sum((thetas1 - thetas2)/(epsilon * 2), axis=1)
            print "calculated_derivative[i]", calculated_derivative[i]
            print "numerical deriv", i_deriv
            deriv.append(i_deriv)
            assert(np.abs(calculated_derivative[i] - i_deriv) < accept_diff)

        return np.hstack(deriv)

    @staticmethod
    def _get_nonzero_theta_vectors(thetas, threshold=1e-10):
        for i in range(thetas.shape[1]):
            print "thetas", np.linalg.norm(thetas[:,i])
        nonzero_thetas_idx = map(lambda i: np.linalg.norm(thetas[:,i]) > threshold, range(thetas.shape[1]))
        return np.array(nonzero_thetas_idx)

    @staticmethod
    def _zero_theta_indices(theta, threshold=1e-10):
        return np.multiply(theta, np.greater(np.abs(theta), threshold))
