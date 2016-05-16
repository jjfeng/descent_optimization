import sys
import scipy as sp
from scipy.sparse.linalg import spsolve
import numpy as np
import cvxpy

from common import *
from convexopt_solvers import GenAddModelProblemWrapper

class GenAddModelHillclimb:
    NUMBER_OF_ITERATIONS = 50
    BOUNDARY_FACTOR = 0.95
    STEP_SIZE = 2
    LAMBDA_MIN = 1e-6
    SHRINK_MIN = 1e-2
    SHRINK_SHRINK_FACTOR = 0.1
    SHRINK_FACTOR_INIT = 1
    DECREASING_ENOUGH_THRESHOLD = 1e-4
    USE_BOUNDARY = False #True

    def __init__(self, X_train, y_train, X_validate, y_validate, X_test, nesterov=False):
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

        self.cost_fcn = testerror_multi_smooth
        self.nesterov = nesterov
        if nesterov:
            self.method_label = "hc_nesterov_gam"
        else:
            self.method_label = "hc_gam"

    def run(self, *args, **kargs):
        if self.nesterov:
            return self.run_nesterov(*args, **kargs)
        else:
            return self.run_regular(*args, **kargs)

    def run_regular(self, initial_lambdas, debug=True):
        curr_regularization = initial_lambdas

        thetas = self.problem_wrapper.solve(curr_regularization)
        assert(thetas is not None)

        current_cost = self.cost_fcn(self.y_validate, self.validate_idx, thetas)
        print self.method_label, "first_regularization", curr_regularization, "first cost", current_cost

        # track progression
        cost_path = [current_cost]

        method_step_size = self.STEP_SIZE
        shrink_factor = self.SHRINK_FACTOR_INIT
        potential_thetas = None
        for i in range(0, self.NUMBER_OF_ITERATIONS):
            print "ITER", i
            lambda_derivatives = self._get_lambda_derivatives(curr_regularization, thetas)
            assert(not np.any(np.isnan(lambda_derivatives)))
            if debug and np.min(curr_regularization) > 0.01:
                numeric_derivs = self._double_check_derivative(curr_regularization)
                for j, check_d in enumerate(numeric_derivs):
                    assert(np.abs(check_d - lambda_derivatives[j]) < 0.1 or np.abs(check_d - lambda_derivatives[j])/np.abs(check_d) < 0.1)

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
                potential_cost = self.cost_fcn(self.y_validate, self.validate_idx, potential_thetas)

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
                    potential_cost = self.cost_fcn(self.y_validate, self.validate_idx, potential_thetas)
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

                print self.method_label, "iter:", i, "current_cost:", current_cost, "lambdas:", curr_regularization, "shrink_factor", shrink_factor
                print "decrease amount", cost_path[-2] - cost_path[-1]
                if cost_path[-2] - cost_path[-1] < self.DECREASING_ENOUGH_THRESHOLD:
                    print "progress too slow", cost_path[-2] - cost_path[-1]
                    break

            if shrink_factor < self.SHRINK_MIN:
                print self.method_label, "SHRINK SIZE TOO SMALL", "shrink_factor", shrink_factor
                break
            sys.stdout.flush()

        print self.method_label, "curr lambdas:", curr_regularization
        return thetas, cost_path, curr_regularization

    def run_nesterov(self, initial_lambdas, debug=True):
        def _get_accelerated_lambdas(curr_lambdas, prev_lambdas, iter_num):
            print "orig", curr_lambdas
            return np.maximum(
                curr_lambdas + (iter_num - 2) / (iter_num + 1.0) * (curr_lambdas - prev_lambdas),
                np.minimum(curr_lambdas, self.LAMBDA_MIN)
            )

        prev_regularizations = initial_lambdas
        acc_regularizations = initial_lambdas
        best_reg = initial_lambdas
        thetas = self.problem_wrapper.solve(acc_regularizations)
        best_cost = self.cost_fcn(self.y_validate, self.validate_idx, thetas)
        print self.method_label, "init_cost", best_cost

        # track progression
        cost_path = [best_cost]

        # Perform Nesterov with adaptive restarts
        method_step_size = self.STEP_SIZE
        shrink_factor = self.SHRINK_FACTOR_INIT
        i_max = 3
        total_iters = 1
        while i_max > 2 and total_iters < self.NUMBER_OF_ITERATIONS:
            print "restart! with i_max", i_max
            for i in range(2, self.NUMBER_OF_ITERATIONS + 1):
                total_iters += 1
                i_max = i
                lambda_derivatives = self._get_lambda_derivatives(acc_regularizations, thetas)
                if np.array_equal(lambda_derivatives, np.array([0] * lambda_derivatives.size)):
                    print self.method_label, "derivatives zero. break."
                    break

                regular_step_regs = self._get_updated_lambdas(
                    acc_regularizations,
                    shrink_factor * method_step_size,
                    lambda_derivatives
                )
                acc_regularizations = _get_accelerated_lambdas(regular_step_regs, prev_regularizations, i)
                print "acc_regularizations", acc_regularizations
                prev_regularizations = regular_step_regs

                potential_thetas = self.problem_wrapper.solve(acc_regularizations)
                current_cost = self.cost_fcn(self.y_validate, self.validate_idx, potential_thetas)
                print self.method_label, "current_cost", current_cost
                is_decreasing_significantly = best_cost - current_cost > self.DECREASING_ENOUGH_THRESHOLD
                if current_cost < best_cost:
                    best_cost = current_cost
                    cost_path.append(current_cost)
                    thetas = potential_thetas
                    best_reg = acc_regularizations

                if not is_decreasing_significantly:
                    print self.method_label, "DECREASING TOO SLOW"
                    break

                print self.method_label, "iter", i - 1, "current cost", current_cost, "best cost", best_cost, "lambdas:", best_reg
                sys.stdout.flush()

        print self.method_label, "best cost", best_cost, "best lambdas:", best_reg

        return thetas, cost_path, best_reg

    def _get_lambda_derivatives(self, curr_lambdas, curr_thetas):
        print "_get_lambda_derivatives, curr_lambdas", curr_lambdas

        self._double_check_train_loss_grad(curr_thetas, curr_lambdas)

        H = np.tile(self.MM, (self.num_features, self.num_features))
        num_feat_sam = self.num_features * self.num_samples
        # print "self.problem_wrapper.tiny_e/num_feat_sam", self.problem_wrapper.tiny_e/num_feat_sam
        # H += self.problem_wrapper.tiny_e/num_feat_sam * np.eye(num_feat_sam)

        H += sp.linalg.block_diag(*[
            curr_lambdas[i] * self.DD[i] for i in range(self.num_features)
        ])
        H = sp.sparse.csr_matrix(H)

        sum_thetas = np.matrix(np.sum(curr_thetas, axis=1))
        dloss_dlambdas = []
        num_validate = self.y_validate.size
        for i in range(self.num_features):
            b = np.zeros((num_feat_sam, 1))
            b[i * self.num_samples:(i + 1) * self.num_samples, :] = -self.DD[i] * curr_thetas[:,i]
            dtheta_dlambdai = spsolve(H, b)
            dtheta_dlambdai = dtheta_dlambdai.reshape((self.num_features, self.num_samples)).T
            # print "dtheta_dlambdai reshaped", dtheta_dlambdai
            sum_dtheta_dlambdai = np.matrix(np.sum(dtheta_dlambdai, axis=1)).T
            # print "sum_dtheta_dlambdai", sum_dtheta_dlambdai
            dloss_dlambdai = -1.0/num_validate * sum_dtheta_dlambdai[self.validate_idx].T * (self.y_validate - sum_thetas[self.validate_idx])
            dloss_dlambdas.append(dloss_dlambdai[0,0])

        print "dloss_dlambdas", dloss_dlambdas
        return np.array(dloss_dlambdas)

    def _get_updated_lambdas(self, lambdas, method_step_size, lambda_derivatives):
        new_step_size = method_step_size
        if self.USE_BOUNDARY:
            potential_lambdas = lambdas - method_step_size * lambda_derivatives

            for idx in range(0, lambdas.size):
                if lambdas[idx] > self.LAMBDA_MIN and potential_lambdas[idx] < (1 - self.BOUNDARY_FACTOR) * lambdas[idx]:
                    smaller_step_size = self.BOUNDARY_FACTOR * lambdas[idx] / lambda_derivatives[idx]
                    new_step_size = min(new_step_size, smaller_step_size)
                    print "use_boundary!", new_step_size

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

    def _double_check_train_loss_grad(self, curr_thetas, curr_lambdas):
        # Check if grad of training loss is zero
        num_train = self.y_train.size
        true_grads = []
        for i in range(self.num_features):
            true_g = -self.M.T/num_train * (self.y_train - np.sum(curr_thetas[self.train_indices,:], axis=1)) + curr_lambdas[i] * self.DD[i] * curr_thetas[:,i]
            # true_g += self.problem_wrapper.tiny_e/num_feat_sam * curr_thetas[:,i]
            true_grads.append(true_g)
            print "zero?", np.max(np.abs(true_g)), np.min(np.abs(true_g)), np.linalg.norm(true_g, ord=2)


    @staticmethod
    def stack(data_tuple):
        stacked_data = np.vstack(data_tuple)
        res = [stacked_data]
        start_idx = 0
        for d in data_tuple:
            res.append(np.arange(start_idx, start_idx + d.shape[0]))
            start_idx += d.shape[0]
        return res
