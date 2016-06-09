import numpy as np
import scipy as sp
from common import *
from convexopt_solvers import Lambda12ProblemWrapper

NUMBER_OF_ITERATIONS = 60
STEP_SIZE = 0.4
DIMINISHING_STEP_ALPHA = 1.0
LAMBDA_MIN = 1e-5

# used to determine if regularization param is too close to the boundary
BOUNDARY_FACTOR = 0.7

MIN_SHRINK = 1e-8
SHRINK_SHRINK = 0.05
DECREASING_ENOUGH_THRESHOLD = 1e-4
BACKTRACK_ALPHA = 0.01

def run(X_train, y_train, X_validate, y_validate, diminishing_step_size=False, initial_lambda1=1, initial_lambda2=1):
    shrink_factor = 1
    def _get_step_size(iter_num):
        if diminishing_step_size:
            print "diminishing_step_size"
            return DIMINISHING_STEP_ALPHA / iter_num
        else:
            return STEP_SIZE

    method_label = HC_LAMBDA12_DIM_LABEL if diminishing_step_size else HC_LAMBDA12_LABEL

    # our guesses for the params
    curr_lambdas = [initial_lambda1, initial_lambda2]

    problem_wrapper = Lambda12ProblemWrapper(X_train, y_train)

    beta_guess = problem_wrapper.solve(curr_lambdas[0], curr_lambdas[1])

    current_cost = testerror(X_validate, y_validate, beta_guess)
    print method_label, "first regularization", curr_lambdas , "first_cost", current_cost

    # track progression
    cost_path = [current_cost]
    is_decreasing_significantly = True
    for i in range(0, NUMBER_OF_ITERATIONS):
        method_step_size = _get_step_size(i)

        derivative_lambdas = _get_derivative_lambda12(X_train, y_train, X_validate, y_validate, beta_guess, curr_lambdas[1])

        # do the gradient descent!
        prev_lambdas = curr_lambdas
        potential_lambdas = _get_updated_lambdas(curr_lambdas, method_step_size * shrink_factor, derivative_lambdas, use_boundary=False)

        # get corresponding beta
        potential_beta_guess = problem_wrapper.solve(potential_lambdas[0], potential_lambdas[1])
        potential_cost = testerror(X_validate, y_validate, potential_beta_guess)

        # while potential_cost > current_cost and shrink_factor > MIN_SHRINK:
        while potential_cost > current_cost - BACKTRACK_ALPHA * shrink_factor * method_step_size * np.linalg.norm(derivative_lambdas)**2 and shrink_factor > MIN_SHRINK:
            shrink_factor *= SHRINK_SHRINK
            potential_lambdas = _get_updated_lambdas(curr_lambdas, method_step_size * shrink_factor, derivative_lambdas)
            potential_beta_guess = problem_wrapper.solve(potential_lambdas[0], potential_lambdas[1])
            potential_cost = testerror(X_validate, y_validate, potential_beta_guess)
            print "try shrink", "shrink_factor", shrink_factor, "potential cost", potential_cost

        delta_change = current_cost - potential_cost
        if potential_cost < current_cost:
            current_cost = potential_cost
            beta_guess = potential_beta_guess
            curr_lambdas = potential_lambdas

        if shrink_factor <= MIN_SHRINK:
            print method_label, "shrink factor too small"
            break

        if curr_lambdas[0] == prev_lambdas[0] and curr_lambdas[1] == prev_lambdas[1]:
            print method_label, "both lambdas are stuck"
            break

        if delta_change <= DECREASING_ENOUGH_THRESHOLD:
            print method_label, "not decreasing fast enuf"
            break

        # track progression
        cost_path.append(current_cost)

        print method_label, "iter", i, "current_cost:", current_cost, "lambdas:", curr_lambdas

    print method_label, "best cost", current_cost, "best lambdas:", curr_lambdas

    return beta_guess, cost_path


def run_nesterov(X_train, y_train, X_validate, y_validate, initial_lambda1=1, initial_lambda2=1):
    def _get_accelerated_lambda(curr_lambda, prev_lambda, iter_num):
        return max(
            curr_lambda + (iter_num - 2) / (iter_num + 1.0) * (curr_lambda - prev_lambda),
            min(curr_lambda, LAMBDA_MIN)
        )

    # our guesses for the params
    curr_lambdas = [initial_lambda1, initial_lambda2]

    problem_wrapper = Lambda12ProblemWrapper(X_train, y_train)

    beta_guess = problem_wrapper.solve(curr_lambdas[0], curr_lambdas[1])

    current_cost = testerror(X_validate, y_validate, beta_guess)
    print HC_LAMBDA12_NESTEROV_LABEL, "first cost", current_cost, "first_lambdas", curr_lambdas

    # track progression
    cost_path = [current_cost]

    potential_cost = current_cost + 1

    method_step_size = STEP_SIZE
    derivative_lambdas = _get_derivative_lambda12(X_train, y_train, X_validate, y_validate, beta_guess, curr_lambdas[1])
    print "first derivative_lambdas", derivative_lambdas
    while potential_cost > current_cost:
        potential_lambdas = _get_updated_lambdas(curr_lambdas, method_step_size, derivative_lambdas)
        potential_acc_lambdas = [_get_accelerated_lambda(potential_lambdas[j], curr_lambdas[j], 2) for j in range(0, 2)]
        potential_beta_guess = problem_wrapper.solve(potential_acc_lambdas[0], potential_acc_lambdas[1])
        potential_cost = testerror(X_validate, y_validate, potential_beta_guess)
        if potential_cost > current_cost:
            method_step_size *= SHRINK_SHRINK

    print HC_LAMBDA12_NESTEROV_LABEL, "GOOD METHOD STEP SIZE", method_step_size

    # Perform Nesterov with adaptive restarts
    i_max = 3
    i_total = 0
    potential_cost = 0
    accelerated_lambdas = curr_lambdas
    while i_max > 2 and i_total < NUMBER_OF_ITERATIONS:
        print "restart! with i_max", i_max, "i_total", i_total
        curr_lambdas = accelerated_lambdas
        prev_lambdas = curr_lambdas

        for i in range(2, NUMBER_OF_ITERATIONS + 1):
            i_max = i
            i_total += 1

            derivative_lambdas = _get_derivative_lambda12(X_train, y_train, X_validate, y_validate, beta_guess, accelerated_lambdas[1])

            curr_lambdas = _get_updated_lambdas(accelerated_lambdas, method_step_size, derivative_lambdas)
            potential_accelerated_lambdas = [_get_accelerated_lambda(curr_lambdas[j], prev_lambdas[j], i) for j in range(0, 2)]

            prev_lambdas = curr_lambdas

            potential_beta_guess = problem_wrapper.solve(potential_accelerated_lambdas[0], potential_accelerated_lambdas[1])

            potential_cost = testerror(X_validate, y_validate, potential_beta_guess)

            if (current_cost - potential_cost) <= DECREASING_ENOUGH_THRESHOLD:
                print "cost is going back up", current_cost, potential_cost
                break
            else:
                beta_guess = potential_beta_guess
                accelerated_lambdas = potential_accelerated_lambdas
                current_cost = potential_cost

            # track progression
            cost_path.append(current_cost)

            print HC_LAMBDA12_NESTEROV_LABEL, "iter", i, "current_cost:", current_cost, "lambdas:", accelerated_lambdas

    print HC_LAMBDA12_NESTEROV_LABEL, "best cost", current_cost, "best lambdas:", curr_lambdas

    return beta_guess, cost_path


def _get_updated_lambdas(current_lambdas, step_size, derivative_lambdas, use_boundary=False):
    derivatives = np.array(derivative_lambdas)
    lambdas = np.array(current_lambdas)
    potential_lambdas = lambdas - step_size * derivatives

    new_step_size = step_size
    if use_boundary:
        print "use_boundary", BOUNDARY_FACTOR
        for idx in range(0, len(lambdas)):
            if potential_lambdas[idx] < (1 - BOUNDARY_FACTOR) * lambdas[idx]:
                smaller_step_size = BOUNDARY_FACTOR * lambdas[idx] / derivatives[idx]
                new_step_size = min(new_step_size, smaller_step_size)

        new_lambdas = lambdas - new_step_size * derivatives
        return new_lambdas
    else:
        return np.maximum(1e-10, lambdas - new_step_size * derivatives)

def _get_derivative_lambda12(X_train, y_train, X_validate, y_validate, beta_guess, lambda2):
    nonzero_indices = get_nonzero_indices(beta_guess)

    # If everything is zero, gradient is zero
    if np.sum(nonzero_indices) == 0:
        return [0, 0]

    X_train_mini = X_train[:, nonzero_indices]
    X_validate_mini = X_validate[:, nonzero_indices]
    beta_guess_mini = beta_guess[nonzero_indices]

    eye_matrix = np.matrix(np.identity(beta_guess_mini.size))
    to_invert_matrix = X_train_mini.T * X_train_mini + lambda2 * eye_matrix

    dbeta_dlambda1, _, _, _ = np.linalg.lstsq(to_invert_matrix, -1 * np.sign(beta_guess_mini))
    dbeta_dlambda2, _, _, _ = np.linalg.lstsq(to_invert_matrix, -1 * beta_guess_mini)

    err_vector = y_validate - X_validate_mini * beta_guess_mini
    gradient_lambda1 = -1 * (X_validate_mini * dbeta_dlambda1).T * err_vector
    gradient_lambda2 = -1 * (X_validate_mini * dbeta_dlambda2).T * err_vector

    return [gradient_lambda1[0,0], gradient_lambda2[0,0]]
