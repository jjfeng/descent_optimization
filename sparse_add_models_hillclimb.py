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

    def get_validate_cost(self, model_params):
        return testerror_sparse_add_smooth(
            self.data.y_validate,
            self.data.validate_idx,
            model_params
        )

    def _get_lambda_derivatives(self):
        nonzero_thetas_idx = self._get_nonzero_theta_vectors(self.fmodel.current_model_params)
        nonzero_thetas_idx = np.array(nonzero_thetas_idx)
        print "nonzero_thetas_idx", nonzero_thetas_idx

        dtheta_dlambda = self._get_dtheta_dlambda(nonzero_thetas_idx)
        fitted_y_validate = np.sum(self.fmodel.current_model_params[self.data.validate_idx, :], axis=1)
        dloss_dlambda = -1 * (
            np.sum(dtheta_dlambda[self.data.validate_idx, :, :], axis=1).T * (self.data.y_validate - fitted_y_validate)
        )
        print "dloss_dlambda", dloss_dlambda
        return dloss_dlambda.A1 # flatten the matrix

    def _get_dtheta_dlambda(self, nonzero_thetas_idx):
        nonzero_thetas = self.fmodel.current_model_params[:, nonzero_thetas_idx]

        def create_b_diag_elem(i):
            # theta = self.fmodel.current_model_params[:,i]
            theta = nonzero_thetas[:,i]
            assert(theta.shape[0] == self.data.num_samples)
            theta_norm = np.linalg.norm(theta, 2)
            b_diag_elem = 1.0/theta_norm * (np.eye(self.data.num_samples) - theta * theta.T/(theta_norm**2))
            return b_diag_elem

        def normalize_theta(i):
            # theta = self.fmodel.current_model_params[:,i]
            theta = nonzero_thetas[:,i]
            print "theta", theta
            theta_norm = np.linalg.norm(theta, 2)
            return theta/theta_norm

        def make_diag_dtheta(i):
            # theta = self.fmodel.current_model_params[:,i]
            theta = nonzero_thetas[:,i]
            zeroed_thetas = self._zero_theta_indices(self.problem_wrapper.diff_matrices[i] * theta)
            print "self.problem_wrapper.diff_matrices[i] * theta", self.problem_wrapper.diff_matrices[i] * theta
            print "zeroed_thetas", zeroed_thetas
            dd_theta = self.problem_wrapper.diff_matrices[i].T * np.sign(zeroed_thetas)
            return dd_theta

        num_nonzero_thetas = nonzero_thetas.shape[1]
        b_diag_elems = map(create_b_diag_elem, range(num_nonzero_thetas))
        print "b_diag_elems", b_diag_elems
        b_diag = sp.linalg.block_diag(*b_diag_elems)

        # normalized_thetas = map(normalize_theta, range(self.data.num_features))
        normalized_thetas = map(normalize_theta, range(num_nonzero_thetas))
        normalized_thetas = np.vstack(normalized_thetas)
        print "normalized_thetas", normalized_thetas

        diag_dthetas = map(make_diag_dtheta, range(num_nonzero_thetas))
        diag_dthetas = sp.linalg.block_diag(*diag_dthetas)
        print "np.logical_not(nonzero_thetas_idx)", np.where(np.logical_not(nonzero_thetas_idx))
        print "diag_dthetas BEFORE", diag_dthetas
        print "INSERT", np.insert(diag_dthetas, np.where(np.logical_not(nonzero_thetas_idx))[0], np.zeros((diag_dthetas.shape[0], 1)), axis=1)
        diag_dthetas = np.insert(diag_dthetas, np.where(np.logical_not(nonzero_thetas_idx))[0], np.zeros((diag_dthetas.shape[0], 1)), axis=1)
        print "diag_dthetas", diag_dthetas

        rhs_matrix = np.hstack((normalized_thetas, diag_dthetas))
        print "rhs_matrix", rhs_matrix

        lambda0 = self.fmodel.current_lambdas[0]
        II = np.matrix(
            np.tile(np.eye(self.data.num_samples), (num_nonzero_thetas, num_nonzero_thetas))
        )
        assert(II.shape == b_diag.shape)
        dtheta_dlambda, _, _, _ = np.linalg.lstsq(II + lambda0 * b_diag, -1 * rhs_matrix)
        print "dtheta_dlambda", dtheta_dlambda
        # dtheta_dlambda = np.asarray(dtheta_dlambda).reshape(self.data.num_samples, self.data.num_features, self.fmodel.num_lambdas, order="F")
        dtheta_dlambda = np.asarray(dtheta_dlambda).reshape(self.data.num_samples, num_nonzero_thetas, self.fmodel.num_lambdas, order="F")
        print "dtheta_dlambda", dtheta_dlambda[:,:,0]
        print "dtheta_dlambda", dtheta_dlambda[:,:,1]
        return dtheta_dlambda

    def _double_check_derivative(self, calculated_derivative, accept_diff=1e-3, epsilon=1e-6):
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
            deriv.append(i_deriv)
            print "thetas1 norm", np.linalg.norm(thetas1[:,0]), thetas1[:,0]
            print "thetas2 norm", np.linalg.norm(thetas2[:,0]), thetas2[:,0]
            print "(thetas1 - thetas2)/(epsilon * 2)", (thetas1 - thetas2)/(epsilon * 2)
            print "calculated_derivative[i]", calculated_derivative[i]
            print "numerical deriv", i_deriv
            assert(np.abs(calculated_derivative[i] - i_deriv) < accept_diff)

        return deriv

    @staticmethod
    def _get_nonzero_theta_vectors(thetas, threshold=1e-10):
        for i in range(thetas.shape[1]):
            print "thetas", np.linalg.norm(thetas[:,i])
        nonzero_thetas_idx = map(lambda i: np.linalg.norm(thetas[:,i]) > threshold, range(thetas.shape[1]))
        return np.array(nonzero_thetas_idx)

    @staticmethod
    def _zero_theta_indices(theta, threshold=1e-10):
        return np.multiply(theta, np.greater(np.abs(theta), threshold))

    @staticmethod
    def _get_zero_theta_indices(theta, threshold=1e-10):
        return np.less(np.abs(theta), threshold).A1

    @staticmethod
    def _reformulate_theta_to_beta(theta, diff_matrix, eps=1e-10):
        zero_theta_idx = self._get_zero_theta_indices(diff_matrix * theta)
        inflatedD = np.zeros(diff_matrix.shape)
        inflatedD[:np.sum(zero_theta_idx),:] = diff_matrix[zero_theta_idx,:]
        u, s, v = sp.linalg.svd(inflatedD)
        null_mask = (s <= eps)
        null_space = sp.compress(null_mask, v, axis=0)
        null_matrix = sp.transpose(null_space)
        beta, res, _, _ = np.linalg.lstsq(null_matrix, theta)
        # Check that we reformulated theta but it is still very close to the original theta
        assert(res < eps)
        return beta, null_matrix
