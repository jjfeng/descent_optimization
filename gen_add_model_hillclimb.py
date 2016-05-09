import scipy as sp
from scipy.sparse.linalg import spsolve
import numpy as np
import cvxpy

from common import *
from convexopt_solvers import GenAddModelProblemWrapper

class GenAddModelHillclimb:
    NUMBER_OF_ITERATIONS = 20 #60
    BOUNDARY_FACTOR = 0.975
    STEP_SIZE = 1
    LAMBDA_MIN = 1e-6
    SHRINK_MIN = 1e-2
    SHRINK_SHRINK_FACTOR = 0.1
    SHRINK_FACTOR_INIT = 1
    DECREASING_ENOUGH_THRESHOLD = 1e-2 * 5
    METHOD_LABEL = "HC_Generalized_additive_model"
    USE_BOUNDARY = True

    def __init__(self, X_train, y_train, X_validate, y_validate, X_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_validate = X_validate
        self.y_validate = y_validate
        self.X_test = X_test
        self.X_full, self.train_idx, self.validate_idx, self.test_idx = self.stack((X_train, X_validate, X_test))

        self.problem_wrapper = GenAddModelProblemWrapper(self.X_full, self.train_idx, self.y_train)
        self.num_samples = self.problem_wrapper.num_samples
        self.num_features = self.problem_wrapper.num_features

        self.train_indices = self.problem_wrapper.train_indices
        self.M = self.problem_wrapper.train_identifier
        num_validate = len(self.validate_idx)
        num_train = len(self.train_idx)
        self.validate_M = np.matrix(np.zeros((num_validate, self.num_samples)))
        self.validate_M[np.arange(num_validate), self.validate_idx] = 1

        self.MM = self.M.T * self.M/num_train
        self.DD = []
        for feat_idx in range(self.num_features):
            D = self.problem_wrapper.diff_matrices[feat_idx]
            self.DD.append(D.T * D/self.num_samples)

    def run(self, initial_lambdas, debug=True):
        curr_regularization = initial_lambdas

        thetas = self.problem_wrapper.solve(curr_regularization)
        assert(thetas is not None)

        current_cost = testerror_multi_smooth(self.y_validate, self.validate_idx, thetas)
        print self.METHOD_LABEL, "first_regularization", curr_regularization, "first cost", current_cost

        # track progression
        cost_path = [current_cost]

        method_step_size = self.STEP_SIZE
        shrink_factor = self.SHRINK_FACTOR_INIT
        potential_thetas = None
        for i in range(0, self.NUMBER_OF_ITERATIONS):
            print "ITER", i
            lambda_derivatives = self._get_lambda_derivatives(curr_regularization, thetas)
            if debug and np.min(curr_regularization) > 0.01:
                print "get_cost_components", self.problem_wrapper.get_cost_components()
                check_derivs = self._double_check_derivative(curr_regularization)
                print "lambda_derivatives", lambda_derivatives
                print "numeric derivs", check_derivs
                for j, check_d in enumerate(check_derivs):
                    print "lambda_derivatives", lambda_derivatives[j]
                    print "numeric_deriv", check_d
                    print "np.abs(check_d - lambda_derivatives[j])/np.abs(check_d)", np.abs(check_d - lambda_derivatives[j])/np.abs(check_d)
                    assert(np.abs(check_d - lambda_derivatives[j]) < 0.1 or np.abs(check_d - lambda_derivatives[j])/np.abs(check_d) < 0.1)
            assert(not np.any(np.isnan(lambda_derivatives)))

            potential_new_regularization = self._get_updated_lambdas(
                curr_regularization,
                shrink_factor * method_step_size,
                lambda_derivatives
            )
            try:
                potential_thetas = self.problem_wrapper.solve(potential_new_regularization)
            except cvxpy.error.SolverError:
                potential_thetas = None

            if potential_thetas is None:
                print "cvxpy could not find a soln"
                potential_cost = current_cost * 100
            else:
                potential_cost = testerror_multi_smooth(self.y_validate, self.validate_idx, potential_thetas)

            while potential_cost >= current_cost and shrink_factor > self.SHRINK_MIN:
                if potential_cost > 2 * current_cost:
                    shrink_factor *= self.SHRINK_SHRINK_FACTOR * 0.01
                else:
                    shrink_factor *= self.SHRINK_SHRINK_FACTOR

                potential_new_regularization = self._get_updated_lambdas(curr_regularization, shrink_factor * method_step_size, lambda_derivatives)
                print "potential_new_regularization", potential_new_regularization
                try:
                    potential_thetas = self.problem_wrapper.solve(potential_new_regularization)
                except cvxpy.error.SolverError as e:
                    print "cvxpy could not find a soln", e
                    potential_thetas = None

                if potential_thetas is None:
                    potential_cost = current_cost * 100
                    print "try shrink", shrink_factor, "no soln. oops!"
                else:
                    potential_cost = testerror_multi_smooth(self.y_validate, self.validate_idx, potential_thetas)
                    print "try shrink", shrink_factor, "potential_cost", potential_cost

            # track progression
            if cost_path[-1] < potential_cost:
                print "COST IS INCREASING!"
                break
            else:
                curr_regularization = potential_new_regularization
                current_cost = potential_cost
                thetas = potential_thetas
                cost_path.append(current_cost)

                print self.METHOD_LABEL, "iter:", i, "current_cost:", current_cost, "lambdas:", curr_regularization, "shrink_factor", shrink_factor

                if cost_path[-2] - cost_path[-1] < self.DECREASING_ENOUGH_THRESHOLD:
                    print "progress too slow", cost_path[-2] - cost_path[-1]
                    break

            if shrink_factor < self.SHRINK_MIN:
                print self.METHOD_LABEL, "SHRINK SIZE TOO SMALL", "shrink_factor", shrink_factor
                break

        print self.METHOD_LABEL, "current cost", current_cost, "curr_regularization", curr_regularization, "total iters:", i
        print self.METHOD_LABEL, "curr lambdas:",  "first cost", cost_path[0], "initial_lambdas", initial_lambdas

        return thetas, cost_path, curr_regularization

    def _get_lambda_derivatives(self, curr_lambdas, curr_thetas):
        print "_get_lambda_derivatives, curr_lambdas", curr_lambdas
        H = np.tile(self.MM, (self.num_features, self.num_features))
        num_feat_sam = self.num_features * self.num_samples
        # print "self.problem_wrapper.tiny_e/num_feat_sam", self.problem_wrapper.tiny_e/num_feat_sam
        # H += self.problem_wrapper.tiny_e/num_feat_sam * np.eye(num_feat_sam)
        # print "H", H
        # print "curr_lambdas[i] * self.DD[i]", curr_lambdas[0] * self.DD[0]
        # print "sp.linalg.block_diag", sp.linalg.block_diag(*[curr_lambdas[i] * self.DD[i] for i in range(self.num_features)])
        H += sp.linalg.block_diag(*[
            curr_lambdas[i] * self.DD[i] for i in range(self.num_features)
        ])
        num_train = self.y_train.size
        # print "num_train", num_train
        # print "np.sum(curr_thetas, axis=1)", np.sum(curr_thetas, axis=1)
        # print "curr_lambdas[0] * self.DD[0] * curr_thetas[:,0]", curr_lambdas[0] * self.DD[0] * curr_thetas[:,0]
        # print "-self.M.T/num_train * (self.y_train - self.M * np.sum(curr_thetas, axis=1))", -self.M.T/num_train * (self.y_train - self.M * np.sum(curr_thetas, axis=1))
        # print "-self.M.T/num_train * (self.y_train - np.sum(curr_thetas[self.train_indices,:], axis=1))", -self.M.T/num_train * (self.y_train - np.sum(curr_thetas[self.train_indices,:], axis=1))
        true_grads = []
        for i in range(self.num_features):
            true_g = -self.M.T/num_train * (self.y_train - np.sum(curr_thetas[self.train_indices,:], axis=1)) + curr_lambdas[0] * self.DD[0] * curr_thetas[:,0] + self.problem_wrapper.tiny_e/num_feat_sam * curr_thetas[:,i]
            true_grads.append(true_g)
            print "zero?", np.max(np.abs(true_g)), np.min(np.abs(true_g)), np.linalg.norm(true_g, ord=2)

        # print "H", H
        # print "curr_lambdas", curr_lambdas
        # print "self.DD[0]", self.DD[0]
        # print "0: curr_lambdas * DD", curr_lambdas[0] * self.DD[0]
        # print "1: curr_lambdas * DD", curr_lambdas[1] * self.DD[1]
        # print sp.linalg.block_diag(*[
        #     curr_lambdas[i] * self.DD[i] for i in range(self.num_features)
        # ])
        H = sp.sparse.csr_matrix(H)

        sum_thetas = np.matrix(np.sum(curr_thetas, axis=1))
        dloss_dlambdas = []
        num_validate = self.y_validate.size
        for i in range(self.num_features):
            # print "=========I======", i
            b = np.zeros((num_feat_sam, 1))
            b[i * self.num_samples:(i + 1) * self.num_samples, :] = -self.DD[i] * curr_thetas[:,i]
            # print "-self.DD[i] * curr_thetas[:,i]", -self.DD[i] * curr_thetas[:,i]
            # print "b", b
            # dtheta_dlambdai = sp.linalg.solve(H, b, sym_pos=True)
            # dtheta_dlambdai = np.linalg.solve(H, b)
            # dtheta_dlambdai, _, _, _ = np.linalg.lstsq(H, b)
            dtheta_dlambdai = spsolve(H, b)
            # print "dtheta_dlambdai", dtheta_dlambdai
            dtheta_dlambdai = dtheta_dlambdai.reshape((self.num_features, self.num_samples)).T
            # print "dtheta_dlambdai reshaped", dtheta_dlambdai
            sum_dtheta_dlambdai = np.matrix(np.sum(dtheta_dlambdai, axis=1)).T
            # print "sum_dtheta_dlambdai", sum_dtheta_dlambdai
            # print "self.y_validate", self.y_validate
            # print "sum_thetas for validate", sum_thetas[self.validate_idx,:]
            # print "self.y_validate", self.y_validate
            # dloss_dlambdai = -1.0/num_validate * (self.validate_M * sum_dtheta_dlambdai).T * (self.y_validate - self.validate_M * sum_thetas)
            dloss_dlambdai = -1.0/num_validate * sum_dtheta_dlambdai[self.validate_idx].T * (self.y_validate - sum_thetas[self.validate_idx])
            # print "dloss_dlambdai", dloss_dlambdai
            dloss_dlambdas.append(dloss_dlambdai[0,0])

        print "dloss_dlambdas", dloss_dlambdas
        return np.array(dloss_dlambdas)

    def _get_updated_lambdas(self, lambdas, method_step_size, lambda_derivatives):
        new_step_size = method_step_size
        if self.USE_BOUNDARY:
            print "use_boundary!"
            potential_lambdas = lambdas - method_step_size * lambda_derivatives

            for idx in range(0, lambdas.size):
                if lambdas[idx] > self.LAMBDA_MIN and potential_lambdas[idx] < (1 - self.BOUNDARY_FACTOR) * lambdas[idx]:
                    smaller_step_size = self.BOUNDARY_FACTOR * lambdas[idx] / lambda_derivatives[idx]
                    new_step_size = min(new_step_size, smaller_step_size)

        return np.maximum(lambdas - new_step_size * lambda_derivatives, self.LAMBDA_MIN)

    def _double_check_derivative(self, regularization, epsilon=1e-7):
        print "double_check_derivative"
        deriv = []
        for i in range(len(regularization)):
            print "===========CHECK I= %d ===============" % i
            reg1 = [r for r in regularization]
            reg1[i] += epsilon
            thetas1 = self.problem_wrapper.solve(reg1)
            error1 = testerror_multi_smooth(self.y_validate, self.validate_idx, thetas1)

            reg2 = [r for r in regularization]
            reg2[i] -= epsilon
            thetas2 = self.problem_wrapper.solve(reg2)
            error2 = testerror_multi_smooth(self.y_validate, self.validate_idx, thetas2)
            print "(error1 - error2)/(epsilon * 2)", (error1 - error2)/(epsilon * 2)
            deriv.append(
                (error1 - error2)/(epsilon * 2)
            )
            print "(thetas1 - thetas2)/(epsilon * 2)", (thetas1 - thetas2)/(epsilon * 2)
        return deriv


    @staticmethod
    def stack(data_tuple):
        stacked_data = np.vstack(data_tuple)
        res = [stacked_data]
        start_idx = 0
        for d in data_tuple:
            res.append(np.arange(start_idx, start_idx + d.shape[0]))
            start_idx += d.shape[0]
        return res
