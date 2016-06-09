import scipy as sp
import numpy as np
from common import *
from convexopt_solvers import GroupedLassoProblemWrapper

NUMBER_OF_ITERATIONS = 40
STEP_SIZE = 1
BOUNDARY_FACTOR = 0.5
DECREASING_ENOUGH_THRESHOLD = 1e-4
SHRINK_SHRINK = 0.05
MIN_SHRINK = 1e-10 #2
MIN_LAMBDA = 1e-6 #8
DEFAULT_LAMBDA = 1e-4
BACKTRACK_ALPHA = 0 #0.001 - works well for the 300 feat setting

def run(X_train, y_train, X_validate, y_validate, group_feature_sizes, initial_lambda1=DEFAULT_LAMBDA):
    print "BACKTRACK_ALPHA", BACKTRACK_ALPHA
    method_step_size = STEP_SIZE
    method_label = HC_GROUPED_LASSO_LABEL

    curr_regularizations = np.concatenate((
        np.ones(len(group_feature_sizes)) * initial_lambda1,
        [initial_lambda1]
    ))
    problem_wrapper = GroupedLassoProblemWrapper(X_train, y_train, group_feature_sizes)
    betas = problem_wrapper.solve(curr_regularizations)
    best_beta = betas
    best_cost = testerror_grouped(X_validate, y_validate, betas)

    # track progression
    cost_path = [best_cost]

    shrink_factor = 1
    for i in range(1, NUMBER_OF_ITERATIONS):
        lambda_derivatives = _get_lambda_derivatives(X_train, y_train, X_validate, y_validate, betas, curr_regularizations)
        # numeric_derivs = _check_lambda_derivatives(problem_wrapper, X_validate, y_validate, curr_regularizations)
        # _compare_numeric_calculated_derivs(lambda_derivatives, numeric_derivs)

        if np.any(np.isnan(lambda_derivatives)):
            print "some value in df_dlambda is nan"
            break

        # do the gradient descent!
        pot_lambdas = _get_updated_lambdas(curr_regularizations, shrink_factor * method_step_size, lambda_derivatives)

        # get corresponding beta
        pot_betas = problem_wrapper.solve(pot_lambdas)

        try:
            pot_cost = testerror_grouped(X_validate, y_validate, pot_betas)
        except ValueError as e:
            print "value error", e
            pot_cost = 1e10

        backtrack_check = best_cost - BACKTRACK_ALPHA * shrink_factor * method_step_size * np.linalg.norm(lambda_derivatives)**2
        backtrack_check = best_cost if backtrack_check < 0 else backtrack_check
        while pot_cost > backtrack_check and shrink_factor > MIN_SHRINK:
            print "bactrack?", pot_cost, backtrack_check
            shrink_factor *= SHRINK_SHRINK
            backtrack_check = best_cost - BACKTRACK_ALPHA * shrink_factor * method_step_size * np.linalg.norm(lambda_derivatives)**2
            print "backtrack_check?", backtrack_check
            backtrack_check = best_cost if backtrack_check < 0 else backtrack_check
            print "shrink!", shrink_factor
            pot_lambdas = _get_updated_lambdas(curr_regularizations, shrink_factor * method_step_size, lambda_derivatives)
            pot_betas = problem_wrapper.solve(pot_lambdas)
            try:
                pot_cost = testerror_grouped(X_validate, y_validate, pot_betas)
            except ValueError as e:
                print "value error", e
                pot_cost = 1e10
            print "pot_cost", pot_cost

        is_decreasing_signficantly = best_cost - pot_cost > DECREASING_ENOUGH_THRESHOLD

        betas = pot_betas
        curr_regularizations = pot_lambdas
        if pot_cost < best_cost:
            best_cost = pot_cost
            best_beta = betas

        if not is_decreasing_signficantly:
            print "is_decreasing_signficantly NO!"
            break

        if shrink_factor <= MIN_SHRINK:
            print method_label, "shrink factor too small"
            break

        # track progression
        cost_path.append(pot_cost)

        print method_label, "iter", i, "best cost", best_cost #, "lambdas:", curr_regularizations

    print method_label, "best cost", best_cost, "lambdas:", curr_regularizations
    return best_beta, cost_path


def run_nesterov(X_train, y_train, X_validate, y_validate, group_feature_sizes, initial_lambda1=DEFAULT_LAMBDA):
    def _get_accelerated_lambdas(curr_lambdas, prev_lambdas, iter_num):
        return np.maximum(
            curr_lambdas + (iter_num - 2) / (iter_num + 1.0) * (curr_lambdas - prev_lambdas),
            np.minimum(curr_lambdas, MIN_LAMBDA)
        )

    method_label = "HillclimbGroupedLasso Nesterov"

    curr_regularizations = np.concatenate((
        np.ones(len(group_feature_sizes)) * initial_lambda1,  # lambda1s
        [initial_lambda1]
    ))
    prev_regularizations = curr_regularizations
    acc_regularizations = curr_regularizations
    problem_wrapper = GroupedLassoProblemWrapper(X_train, y_train, group_feature_sizes)
    betas = problem_wrapper.solve(acc_regularizations)
    best_cost = testerror_grouped(X_validate, y_validate, betas)
    best_betas = betas

    # track progression
    cost_path = [best_cost]

    # Perform Nesterov with adaptive restarts
    i_max = 3
    while i_max > 2:
        print "restart! with i_max", i_max
        for i in range(2, NUMBER_OF_ITERATIONS + 1):
            i_max = i
            lambda_derivatives = _get_lambda_derivatives(X_train, y_train, X_validate, y_validate, betas, acc_regularizations)
            if np.array_equal(lambda_derivatives, np.array([0] * lambda_derivatives.size)):
                print method_label, "derivatives zero. break."
                break

            curr_regularizations = _get_updated_lambdas(acc_regularizations, STEP_SIZE, lambda_derivatives, use_boundary=True)
            acc_regularizations = _get_accelerated_lambdas(curr_regularizations, prev_regularizations, i)
            prev_regularizations = curr_regularizations

            potential_betas = problem_wrapper.solve(acc_regularizations)
            current_cost = testerror_grouped(X_validate, y_validate, potential_betas)
            is_decreasing_significantly = best_cost - current_cost > DECREASING_ENOUGH_THRESHOLD
            if current_cost < best_cost:
                best_cost = current_cost
                best_betas = potential_betas
                cost_path.append(current_cost)
                betas = potential_betas

            if not is_decreasing_significantly:
                print method_label, "DECREASING TOO SLOW"
                break

            print method_label, "iter", i, "current cost", current_cost, "best cost", best_cost, "lambdas:", acc_regularizations

    print method_label, "best cost", best_cost, "best lambdas:", acc_regularizations

    return best_betas, cost_path

def _get_updated_lambdas(lambdas, method_step_size, lambda_derivatives, use_boundary=False):
    new_step_size = method_step_size
    if use_boundary:
        potential_lambdas = lambdas - method_step_size * lambda_derivatives

        for idx in range(0, lambdas.size):
            if lambdas[idx] > MIN_LAMBDA and potential_lambdas[idx] < (1 - BOUNDARY_FACTOR) * lambdas[idx]:
                smaller_step_size = BOUNDARY_FACTOR * lambdas[idx] / lambda_derivatives[idx]
                new_step_size = min(new_step_size, smaller_step_size)

    return np.maximum(lambdas - new_step_size * lambda_derivatives, MIN_LAMBDA)


def _get_lambda_derivatives(X_train, y_train, X_validate, y_validate, betas, curr_regularizations):
    # first minify the data
    beta_minis = []
    beta_nonzeros = []
    for beta in betas:
        beta_nonzero = get_nonzero_indices(beta)
        beta_nonzeros.append(beta_nonzero)
        beta_minis.append(beta[beta_nonzero])

    complete_beta_nonzero = np.concatenate(beta_nonzeros)
    X_train_mini = X_train[:, complete_beta_nonzero]
    X_validate_mini = X_validate[:, complete_beta_nonzero]

    if X_train_mini.size == 0:
        return np.array([0] * len(curr_regularizations))

    return _get_lambda_derivatives_mini(X_train_mini, y_train, X_validate_mini, y_validate, beta_minis, curr_regularizations)

def _get_lambda_derivatives_mini(X_train, y_train, X_validate, y_validate, betas, curr_regularizations):
    lambda1s = curr_regularizations[0:-1]

    def _get_block_diag_component(idx):
        beta = betas[idx]
        if beta.size == 0:
            return np.matrix(np.zeros((0,0))).T

        repeat_hstacked_beta = np.tile(beta, (1, beta.size)).T
        block_diag_component = -1 * lambda1s[idx] / get_norm2(beta, power=3) * np.diagflat(beta) * repeat_hstacked_beta
        return block_diag_component

    def _get_diagmatrix_component(idx):
        beta = betas[idx]
        if beta.size == 0:
            return np.matrix(np.zeros((0,0))).T
        return lambda1s[idx] / get_norm2(beta) * np.identity(beta.size)

    def _get_dbeta_dlambda1(beta, matrix_to_invert, num_features_before):
        if beta.size == 0:
            return np.matrix(np.zeros((matrix_to_invert.shape[0], 1)))
        else:
            normed_beta = beta / get_norm2(beta)
            zero_normed_beta = np.concatenate([
                np.matrix(np.zeros(num_features_before)).T,
                normed_beta,
                np.matrix(np.zeros(total_features - normed_beta.size - num_features_before)).T
            ])

            dbeta_dlambda1, _, _, _ = np.linalg.lstsq(matrix_to_invert, -1 * zero_normed_beta)
            return dbeta_dlambda1

    num_feature_groups = len(betas)
    total_features = X_train.shape[1]
    complete_beta = np.concatenate(betas)

    XX = X_train.T * X_train

    block_diag_components = [_get_block_diag_component(idx) for idx in range(0, num_feature_groups)]
    diagonal_components = [_get_diagmatrix_component(idx) for idx in range(0, num_feature_groups)]
    dgrouplasso_dlambda = sp.linalg.block_diag(*block_diag_components) + sp.linalg.block_diag(*diagonal_components)

    matrix_to_invert = 1.0 / y_train.size * XX + dgrouplasso_dlambda

    dbeta_dlambda1s = np.matrix(np.zeros((0,0))).T
    num_features_before = 0
    for beta in betas:
        dbeta_dlambda1 = _get_dbeta_dlambda1(beta, matrix_to_invert, num_features_before)
        num_features_before += beta.size

        if dbeta_dlambda1s.size == 0:  # not initialized yet
            dbeta_dlambda1s = dbeta_dlambda1
        else:
            dbeta_dlambda1s = np.hstack([dbeta_dlambda1s, dbeta_dlambda1])

    dbeta_dlambda2, _, _, _ = np.linalg.lstsq(matrix_to_invert, -1 * np.sign(complete_beta))

    err_vector = y_validate - X_validate * complete_beta
    df_dlambda1s = -1.0 / y_validate.size * (X_validate * dbeta_dlambda1s).T * err_vector
    df_dlambda1s = np.reshape(np.array(df_dlambda1s), df_dlambda1s.size)
    df_dlambda2 = -1.0 / y_validate.size * (X_validate * dbeta_dlambda2).T * err_vector
    return np.concatenate((df_dlambda1s, [df_dlambda2[0,0]]))

def _check_lambda_derivatives(problem_wrapper, X_validate, y_validate, regularization, eps=1e-5):
    print "double_check_derivative"
    deriv = []
    for i in range(min(5, len(regularization))):
        print "===========CHECK I= %d ===============" % i
        reg1 = np.copy(regularization)
        reg1[i] = reg1[i] + eps
        betas1 = problem_wrapper.solve(reg1)
        error1 = testerror_grouped(X_validate, y_validate, betas1)

        reg2 = np.copy(regularization)
        reg2[i] -= eps
        betas2 = problem_wrapper.solve(reg2)
        error2 = testerror_grouped(X_validate, y_validate, betas2)
        print "(error1 - error2)/(epsilon * 2)", (error1 - error2)/(eps * 2)
        deriv.append(
            (error1 - error2)/(eps * 2)
        )
        betas1 = np.array(betas1)
        betas2 = np.array(betas2)
        print "(betas1 - betas2)/(epsilon * 2)", (betas1 - betas2)/(eps * 2)
    return deriv

def _compare_numeric_calculated_derivs(calculated_derivatives, numeric_derivs):
    for i, n_deriv in enumerate(numeric_derivs):
        print "calc", calculated_derivatives[i], "num", n_deriv
        print "diff", np.abs(calculated_derivatives[i] - n_deriv)
