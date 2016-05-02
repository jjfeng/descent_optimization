import numpy as np
from common import *
from convexopt_solvers import LambdaAlphaProblemWrapper

NUMBER_OF_ITERATIONS = 100
STEP_SIZE = 1
DIMINISHING_STEP_FACTOR = 2.0
REGULARIZATION_MIN = 1e-5

# used to determine if regularization param is too close to the boundary
BOUNDARY_FACTOR = 0.8

MIN_SHRINK = 1e-4
SHRINK_SHRINK = 0.7

def run(X_train, y_train, X_validate, y_validate, diminishing_step_size=False, do_shrink=False):
    def _get_step_size(iter_num):
        if diminishing_step_size:
            return DIMINISHING_STEP_FACTOR / iter_num
        else:
            return STEP_SIZE

    method_label = HC_LAMBDA_ALPHA_DIM_LABEL if diminishing_step_size else HC_LAMBDA_ALPHA_LABEL

    # our guesses for the params
    curr_regularization = [2, 0.5]

    problem_wrapper = LambdaAlphaProblemWrapper(X_train, y_train)

    beta_guess = problem_wrapper.solve(curr_regularization[0], curr_regularization[1])

    current_cost = testerror(X_validate, y_validate, beta_guess)

    # track the best
    best_cost = current_cost
    best_beta = beta_guess
    best_regularization = curr_regularization

    # track progression
    cost_path = [current_cost]

    step_size_iter_num = 0
    shrink_factor = 1
    for i in range(1, NUMBER_OF_ITERATIONS):
        step_size_iter_num += 1
        method_step_size = _get_step_size(step_size_iter_num)

        derivative_wrt_regularizations = _get_derivatives(X_train, y_train, X_validate, y_validate, beta_guess, curr_regularization)

        # do the gradient descent!
        prev_regularization = curr_regularization
        potential_regularization = _get_updated_regularizations(curr_regularization, method_step_size * shrink_factor, derivative_wrt_regularizations)

        # get corresponding beta
        potential_beta_guess = problem_wrapper.solve(curr_regularization[0], curr_regularization[1])
        potential_cost = testerror(X_validate, y_validate, beta_guess)

        while do_shrink and potential_cost > best_cost and shrink_factor > MIN_SHRINK:
            print "try shrink"
            shrink_factor *= SHRINK_SHRINK
            potential_regularization = _get_updated_regularizations(curr_regularization, method_step_size * shrink_factor, derivative_wrt_regularizations)
            potential_beta_guess = problem_wrapper.solve(potential_regularization[0], potential_regularization[1])
            potential_cost = testerror(X_validate, y_validate, potential_beta_guess)

        if shrink_factor <= MIN_SHRINK:
            print method_label, "shrink factor too small"
            break
        else:
            current_cost = potential_cost
            beta_guess = potential_beta_guess
            curr_regularization = potential_regularization

        if curr_regularization[0] == prev_regularization[0] and curr_regularization[1] == prev_regularization[1]:
            print method_label, "lambda/mu stuck"
            break

        # track the best
        if best_cost > current_cost:
            best_cost = current_cost
            best_beta = beta_guess
            best_regularization = curr_regularization

        # track progression
        cost_path.append(current_cost)

        print method_label, "current_cost:", current_cost, "best cost", best_cost, "regularization:", curr_regularization

    print method_label, "best cost", best_cost, "best regularization:", best_regularization

    return best_beta, cost_path

def run_nesterov(X_train, y_train, X_validate, y_validate):
    method_label = HC_LAMBDA_ALPHA_NESTEROV_LABEL

    def _get_accelerated(curr, prev, i):
        return curr + (i - 2.0) / (i + 1.0) * (curr - prev)

    def _get_accelerated_regularizations(curr_regularizations, prev_regularizations, iter_num):
        lambda_min = min(curr_regularizations[0], REGULARIZATION_MIN)
        new_regularizations = [_get_accelerated(curr_regularizations[i], prev_regularizations[i], iter_num) for i in range(0, len(curr_regularizations))]
        # lambda cannot be negative
        if new_regularizations[0] < 0:
            new_regularizations[0] = min(curr_regularizations[0], REGULARIZATION_MIN)

        return new_regularizations

    # our guesses for the params
    curr_regularization = [2, 0]
    prev_regularization = curr_regularization
    accelerated_regularization = prev_regularization

    problem_wrapper = LambdaAlphaProblemWrapper(X_train, y_train)
    beta_guess = problem_wrapper.solve(accelerated_regularization[0], accelerated_regularization[1])
    current_cost = testerror(X_validate, y_validate, beta_guess)

    # track the best
    best_cost = current_cost
    best_beta = beta_guess
    best_regularization = accelerated_regularization

    # track progression
    cost_path = [current_cost]

    for i in range(2, NUMBER_OF_ITERATIONS + 1):
        derivative_wrt_regularizations = _get_derivatives(X_train, y_train, X_validate, y_validate, beta_guess, accelerated_regularization)

        # do the gradient descent!
        prev_regularization = curr_regularization
        curr_regularization = _get_updated_regularizations(accelerated_regularization, STEP_SIZE, derivative_wrt_regularizations)

        accelerated_regularization = _get_accelerated_regularizations(curr_regularization, prev_regularization, i)

        if curr_regularization[0] == prev_regularization[0] and curr_regularization[1] == prev_regularization[1]:
            print method_label, "lambda/mu stuck"
            break

        prev_regularization = curr_regularization

        beta_guess = problem_wrapper.solve(accelerated_regularization[0], accelerated_regularization[1])

        current_cost = testerror(X_validate, y_validate, beta_guess)

        # track the best
        if best_cost > current_cost:
            best_cost = current_cost
            best_beta = beta_guess
            best_regularization = accelerated_regularization

        # track progression
        cost_path.append(current_cost)

        print method_label, "current_cost:", current_cost, "best cost", best_cost, "lambdas:", accelerated_regularization

    print method_label, "best cost", best_cost, "best regularization:", best_regularization

    return best_beta, cost_path


def _get_updated_regularizations(curr_regularization, step_size, derivatives):
    curr_regularization = np.array(curr_regularization)
    derivatives = np.array(derivatives)
    potential_regularization = curr_regularization - step_size * derivatives

    new_step_size = step_size

    # lambda cannot be negative or too close to zero
    if potential_regularization[0] < (1 - BOUNDARY_FACTOR) * curr_regularization[0]:
        smaller_step_size = BOUNDARY_FACTOR * curr_regularization[0] / derivatives[0]
        new_step_size = min(new_step_size, smaller_step_size)

    return curr_regularization - new_step_size * derivatives

def _get_derivatives(X_train, y_train, X_validate, y_validate, beta_guess, regularizations):
    lambda_guess = regularizations[0]
    mu_guess = regularizations[1]

    nonzero_indices = get_nonzero_indices(beta_guess)

    # If everything is zero, gradient is zero
    if np.sum(nonzero_indices) == 0:
        return [0, 0]

    X_train_mini = X_train[:, nonzero_indices]
    X_validate_mini = X_validate[:, nonzero_indices]
    beta_guess_mini = beta_guess[nonzero_indices]
    eye_matrix = np.matrix(np.identity(beta_guess_mini.size))

    to_invert_matrix = X_train_mini.T * X_train_mini + lambda_guess / (1 + np.exp(mu_guess)) * eye_matrix

    dbeta_dlambda, _, _, _ = np.linalg.lstsq(to_invert_matrix, -1 * np.exp(mu_guess) / (1 + np.exp(mu_guess)) * np.sign(beta_guess_mini) - 1 / (1 + np.exp(mu_guess)) * beta_guess_mini)
    dbeta_dalpha, _, _, _ = np.linalg.lstsq(to_invert_matrix, -1 * lambda_guess * np.exp(mu_guess) / np.power((1 + np.exp(mu_guess)), 2) * (np.sign(beta_guess_mini) - beta_guess_mini))

    err_vector = y_validate - X_validate_mini * beta_guess_mini
    gradient_lambda = -1 * (X_validate_mini * dbeta_dlambda).T * err_vector
    gradient_mu = -1 * (X_validate_mini * dbeta_dalpha).T * err_vector

    gradient_lambda = gradient_lambda[0,0]
    gradient_mu = gradient_mu[0,0]

    return [gradient_lambda, gradient_mu]
