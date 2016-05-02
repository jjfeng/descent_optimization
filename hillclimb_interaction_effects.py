from common import *
from convexopt_solvers import EffectsInteractionProblemWrapper

NUMBER_OF_ITERATIONS = 100
STEP_SIZE = 1
LAMBDA_MIN = 1e-5
BOUNDARY_FACTOR = 0.8

def run(X_train, W_train, y_train, X_validate, W_validate, y_validate):
    method_label = "INTERACTIONS"
    lambda_guesses = np.array([1 for i in range(0, 4)])

    problem_wrapper = EffectsInteractionProblemWrapper(X_train, W_train, y_train)

    beta_guess, theta_guess = problem_wrapper.solve(lambda_guesses)

    current_cost = testerror_interactions(X_validate, W_validate, y_validate, beta_guess, theta_guess)
    # track the best
    best_cost = current_cost
    best_beta = beta_guess
    best_theta = theta_guess
    best_regularization = lambda_guesses

    # track progression
    cost_path = [current_cost]

    for i in range(0, NUMBER_OF_ITERATIONS):
        method_step_size = STEP_SIZE

        df_dlambdas = _get_lambda_derivatives(X_train, W_train, y_train, X_validate, W_validate, y_validate, beta_guess, theta_guess, lambda_guesses)

        prev_lambdas = lambda_guesses
        lambda_guesses = _update_lambdas(lambda_guesses, method_step_size, df_dlambdas)

        if np.array_equal(prev_lambdas, lambda_guesses):
            print method_label, "lambdas are stuck"
            break

        beta_guess, theta_guess = problem_wrapper.solve(lambda_guesses)

        current_cost = testerror_interactions(X_validate, W_validate, y_validate, beta_guess, theta_guess)

        # track the best
        if best_cost > current_cost:
            best_cost = current_cost
            best_beta = beta_guess
            best_theta = theta_guess
            best_regularization = lambda_guesses

        # track progression
        cost_path.append(current_cost)

        print method_label, "current_cost:", current_cost, "best cost", best_cost, "lambdas:", lambda_guesses

    print method_label, "best cost", best_cost, "best lambdas:", best_regularization

    return best_beta, best_theta, cost_path


def _update_lambdas(lambdas, method_step_size, df_dlambdas):
    potential_lambdas = lambdas - method_step_size * df_dlambdas

    new_step_size = method_step_size
    for idx in range(0, len(lambdas)):
        if potential_lambdas[idx] < (1 - BOUNDARY_FACTOR) * lambdas[idx]:
            smaller_step_size = BOUNDARY_FACTOR * lambdas[idx] / df_dlambdas[idx]
            new_step_size = min(new_step_size, smaller_step_size)

    return lambdas - new_step_size * df_dlambdas

def _get_lambda_derivatives(X_train, W_train, y_train, X_validate, W_validate, y_validate, beta_guess, theta_guess, lambda_guesses):
    beta_nonzero = get_nonzero_indices(beta_guess)
    theta_nonzero = get_nonzero_indices(theta_guess)

    beta_mini = beta_guess[beta_nonzero]
    theta_mini = theta_guess[theta_nonzero]
    X_train_mini = X_train[:, beta_nonzero]
    X_validate_mini = X_validate[:, beta_nonzero]
    W_train_mini = W_train[:, theta_nonzero]
    W_validate_mini = W_validate[:, theta_nonzero]

    if beta_mini.size == 0 and theta_mini.size == 0:
        return np.array([0,0,0,0])

    validate_errs = y_validate - X_validate * beta_guess - W_validate * theta_guess

    if beta_mini.size == 0:
        dbeta_dlambdas, dtheta_dlambdas = _get_dbeta_theta_dlambdas(X_train, W_train_mini, y_train, beta_guess, theta_mini, lambda_guesses)
        def _get_df_dlambda(lambda_idx):
            return (-1 * (W_validate_mini * dtheta_dlambdas[lambda_idx]).T * validate_errs)[0, 0]
        return np.array([0, _get_df_dlambda(1), 0, _get_df_dlambda(3)])
    elif theta_mini.size == 0:
        dbeta_dlambdas, dtheta_dlambdas = _get_dbeta_theta_dlambdas(X_train_mini, W_train, y_train, beta_mini, theta_guess, lambda_guesses)
        def _get_df_dlambda(lambda_idx):
            return (-1 * (X_validate_mini * dbeta_dlambdas[lambda_idx]).T * validate_errs)[0, 0]
        return np.array([_get_df_dlambda(0), 0, _get_df_dlambda(2), 0])
    else:
        dbeta_dlambdas, dtheta_dlambdas = _get_dbeta_theta_dlambdas(X_train_mini, W_train_mini, y_train, beta_mini, theta_mini, lambda_guesses)
        def _get_df_dlambda(lambda_idx):
            return (-1 * (X_validate_mini * dbeta_dlambdas[lambda_idx] + W_validate_mini * dtheta_dlambdas[lambda_idx]).T * validate_errs)[0, 0]

        # make sure the gradient is not more than length 1
        return np.array([_get_df_dlambda(i) for i in range(0, 4)])

def _get_dbeta_theta_dlambdas(X_train, W_train, y_train, beta, theta, lambda_guesses):
    XX = X_train.T * X_train
    WW = W_train.T * W_train
    XW = X_train.T * W_train
    WX = XW.T

    WW_lambda4_inv = np.linalg.pinv(WW + lambda_guesses[3] * np.matrix(np.identity(WW.shape[0])))
    WW_lambda4_inv_WX = WW_lambda4_inv * WX

    X_eye_matrix = np.matrix(np.identity(beta.size))

    to_invert_matrix_beta = XX + lambda_guesses[2] * X_eye_matrix - XW * WW_lambda4_inv_WX

    dbeta_dlambda1, _, _, _ = np.linalg.lstsq(to_invert_matrix_beta, -1 * np.sign(beta))
    dtheta_dlambda1 = -1 * WW_lambda4_inv_WX * dbeta_dlambda1

    dbeta_dlambda3, _, _, _ = np.linalg.lstsq(to_invert_matrix_beta, -1 * beta)
    dtheta_dlambda3 = -1 * WW_lambda4_inv_WX * dbeta_dlambda3

    XX_lambda3_inv = np.linalg.pinv(XX + lambda_guesses[2] * np.matrix(np.identity(XX.shape[0])))
    XX_lambda3_inv_XW = XX_lambda3_inv * XW

    W_eye_matrix = np.matrix(np.identity(theta.size))

    to_invert_matrix_theta = WW + lambda_guesses[3] * W_eye_matrix - WX * XX_lambda3_inv_XW

    dtheta_dlambda2, _, _, _ = np.linalg.lstsq(to_invert_matrix_theta, -1 * np.sign(theta))
    dbeta_dlambda2 = -1 * XX_lambda3_inv_XW * dtheta_dlambda2

    dtheta_dlambda4, _, _, _ = np.linalg.lstsq(to_invert_matrix_theta, -1 * theta)
    dbeta_dlambda4 = -1 * XX_lambda3_inv_XW * dtheta_dlambda4

    return [dbeta_dlambda1, dbeta_dlambda2, dbeta_dlambda3, dbeta_dlambda4], [dtheta_dlambda1, dtheta_dlambda2, dtheta_dlambda3, dtheta_dlambda4]
