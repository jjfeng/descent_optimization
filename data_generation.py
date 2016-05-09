import math
import numpy as np

from common import *

#######
# y = X * beta + epsilon
# X = iid standard normal
# beta = [nonzero_features, zeros] - nonzero from std normal
# epsilon = iid standard normal
########
def simple(train_size, num_features, num_nonzero_features, signal_noise_ratio=1):
    print "DATA: simple"
    validate_size = _get_validate_size(train_size)
    total_samples = train_size + validate_size + TEST_SIZE
    X = np.matrix(np.random.randn(total_samples, num_features))

    # beta real is a vector iid std normal values and zeros
    beta_real = np.matrix(
        np.concatenate((
            np.random.randn(num_nonzero_features, 1),
            np.zeros((num_features - num_nonzero_features, 1))
        ))
    )

    # Get the appropriate signal noise ratio
    beta_real = _get_rescaled_beta(signal_noise_ratio, X, beta_real)

    # add gaussian noise
    epsilon = np.matrix(np.random.randn(total_samples, 1))

    y = X * beta_real + epsilon

    return _return_split_dataset(X, y, train_size, validate_size, beta_real)

#####
# y = X * beta + epsilon
# X = corr(i, j) = 0.5 ^ abs(i - j)
# beta_real = (nonzero features followed by zero features); nonzero features are drawn uniformly from [-1, 1]
# epsilon = iid standard normal
########
def correlated(train_size, num_features, num_nonzero_features, signal_noise_ratio=1):
    print "DATA: correlated"
    validate_size = _get_validate_size(train_size)
    total_samples = train_size + validate_size + TEST_SIZE

    # Multiplying by the cholesky decomposition of the covariance matrix should suffice: http://www.sitmo.com/article/generating-correlated-random-numbers/
    correlation_matrix = np.matrix([[math.pow(0.5, abs(i - j)) for i in range(0, num_features)] for j in range(0, num_features)])
    X = np.matrix(np.random.randn(total_samples, num_features)) * np.matrix(np.linalg.cholesky(correlation_matrix)).T

    # beta real is a shuffled array of zeros and iid std normal values
    beta_real = np.matrix(
        np.concatenate((
            np.ones((num_nonzero_features, 1)), # np.random.randn(num_nonzero_features, 1),
            np.zeros((num_features - num_nonzero_features, 1))
        ))
    )

    np.random.shuffle(beta_real)

    # add gaussian noise
    epsilon = np.matrix(np.random.randn(total_samples, 1))

    # Get the appropriate signal noise ratio
    SNR_factor = signal_noise_ratio / np.linalg.norm(X * beta_real) * np.linalg.norm(epsilon)
    y = X * beta_real + (1.0 / SNR_factor) * epsilon

    # beta_real = _get_rescaled_beta(signal_noise_ratio, X, beta_real, epsilon)
    # y = X * beta_real + epsilon

    return _return_split_dataset(X, y, train_size, validate_size, beta_real)

#######
# Basically, we have three groups of very correlated variables. We've also added some noise.
# y = X * beta_real + epsilon
# X = [X1, X2, X3, X4] where
# X1 = [z1 + rand(), z1 + rand(), ...., z1 + rand()]
# X2 = [z2 + rand(), z2 + rand(), ...., z2 + rand()]
# X3 = [z3 + rand(), z3 + rand(), ...., z3 + rand()]
# X4 = just iid stanard normal
# beta_real = (3, ..., 3, 3, 0, 0, ...., 0)
# epsilon = iid standard normal
########
def three_groups(train_size, num_features, num_nonzero_features, signal_noise_ratio=1):
    print "DATA: three groups"
    NOISE_STD = 0.1
    COEFF = 3

    validate_size = _get_validate_size(train_size)
    total_samples = train_size + validate_size + TEST_SIZE

    # X1, X2, X3 are the groups of features that are *highly* correlated
    size = int(num_nonzero_features/3)
    X1 = _get_tiled_matrix(total_samples, size) + np.random.randn(total_samples, size) * NOISE_STD
    X2 = _get_tiled_matrix(total_samples, size) + np.random.randn(total_samples, size) * NOISE_STD

    remaining_nonzero_features = num_nonzero_features - 2 * size
    X3 = _get_tiled_matrix(total_samples, remaining_nonzero_features) + np.random.randn(total_samples, remaining_nonzero_features) * NOISE_STD
    X4 = np.random.randn(total_samples, num_features - num_nonzero_features)
    X = np.matrix(np.hstack((X1, X2, X3, X4)))

    beta_real = np.matrix(
        np.concatenate((
            COEFF * np.ones((num_nonzero_features, 1)),
            np.zeros((num_features - num_nonzero_features, 1))
        ))
    )

    # Get the appropriate signal noise ratio
    beta_real = _get_rescaled_beta(signal_noise_ratio, X, beta_real)

    # add gaussian noise
    epsilon = np.matrix(np.random.randn(total_samples, 1))

    y = X * beta_real + epsilon

    return _return_split_dataset(X, y, train_size, validate_size, beta_real)

# num_effects is the number of main effects
# the number of interactions will be (num_effects choose 2)
# y = X * beta + W * theta, where W are interaction terms
def effects_and_interactions(train_size, num_effects, num_nonzero_effects, num_nonzero_interactions, effects_to_interactions_ratio=EFFECTS_TO_INTERACTION_RATIO, desired_signal_noise_ratio=2):
    print "DATA: effects and interactions"
    validate_size = _get_validate_size(train_size)
    total_samples = train_size +  validate_size + TEST_SIZE
    num_interactions = num_effects * (num_effects - 1) / 2
    num_features = num_effects + num_interactions

    # make the X correlated
    # correlation_matrix = np.matrix([[math.pow(0.3, abs(i - j)) for i in range(0, num_effects)] for j in range(0, num_effects)])
    # X = np.matrix(np.random.randn(total_samples, num_effects)) * np.matrix(np.linalg.cholesky(correlation_matrix)).T
    # X = np.array(X)

    X = np.random.randn(total_samples, num_effects)

    W = np.matrix(np.zeros((total_samples, num_interactions)))
    for i in range(0, total_samples):
        sample = X[i, :]
        all_feature_combos = [sample[feature_idx] * sample[feature_idx + 1:] for feature_idx in range(0, num_effects - 1)]
        W[i, :] = np.concatenate(all_feature_combos)

    X = np.matrix(X)

    # beta real is a vector iid std normal values and zeros
    beta_real = np.matrix(
        np.concatenate((
            np.random.randn(num_nonzero_effects, 1),
            np.zeros((num_effects - num_nonzero_effects, 1))
        ))
    )

    theta_real = np.matrix(
        np.concatenate((
            np.random.randn(num_nonzero_interactions, 1),
            np.zeros((num_interactions - num_nonzero_interactions, 1))
        ))
    )
    np.random.shuffle(theta_real)

    beta_real *= effects_to_interactions_ratio / np.linalg.norm(X * beta_real) * np.linalg.norm(W * theta_real)

    SNR_factor = math.sqrt(desired_signal_noise_ratio) / math.sqrt(np.var(X * beta_real + W * theta_real))
    beta_real *= SNR_factor
    theta_real *= SNR_factor

    # add gaussian noise
    epsilon = np.matrix(np.random.randn(total_samples, 1))

    y = X * beta_real + W * theta_real + epsilon

    return _return_split_effects_interactions_dataset(X, W, y, train_size, validate_size, beta_real, theta_real)


def effects_and_other_effects(train_size, num_effects, num_nonzero_effects, num_other_effects, num_nonzero_other, effects_to_other_ratio=EFFECTS_TO_OTHER_RATIO, desired_signal_noise_ratio=2):
    print "DATA: effects and other effects"
    np.random.seed(1)
    validate_size = _get_validate_size(train_size)
    total_samples = train_size +  validate_size + TEST_SIZE

    X = _get_correlated_matrix(X_CORR, total_samples, num_effects)
    W = _get_correlated_matrix(W_CORR, total_samples, num_other_effects)

    # beta real is a vector iid std normal values and zeros
    beta_real = np.matrix(
        np.concatenate((
            np.random.randn(num_nonzero_effects, 1),
            np.zeros((num_effects - num_nonzero_effects, 1))
        ))
    )

    theta_real = np.matrix(
        np.concatenate((
            np.random.randn(num_nonzero_other, 1),
            np.zeros((num_other_effects - num_nonzero_other, 1))
        ))
    )

    np.random.shuffle(theta_real)

    beta_real *= effects_to_other_ratio / np.linalg.norm(X * beta_real) * np.linalg.norm(W * theta_real)

    SNR_factor = math.sqrt(desired_signal_noise_ratio) / math.sqrt(np.var(X * beta_real + W * theta_real))
    beta_real *= SNR_factor
    theta_real *= SNR_factor

    # add gaussian noise
    epsilon = np.matrix(np.random.randn(total_samples, 1))

    y = X * beta_real + W * theta_real + epsilon

    return _return_split_effects_interactions_dataset(X, W, y, train_size, validate_size, beta_real, theta_real)

# y = sum(X_l * beta_l) + epsilon
# X_trains, X_validates, X_tests are returned as arrays of each of the feature groups
def sparse_groups(train_size, group_feature_sizes, desired_signal_noise_ratio=2):
    print "DATA: sparse group lasso data"

    # np.random.seed(9)
    BASE_NONZERO_COEFF = [1, 2, 3, 4, 5]
    NONZERO_FEATURES = len(BASE_NONZERO_COEFF)

    validate_size = _get_validate_size(train_size)
    total_samples = train_size + validate_size + TEST_SIZE

    X = np.matrix(np.random.randn(total_samples, np.sum(group_feature_sizes)))
    betas = [np.matrix(np.concatenate((BASE_NONZERO_COEFF, [0 for i in range(0, num_features - NONZERO_FEATURES)]))).T for num_features in group_feature_sizes]
    beta = np.matrix(np.concatenate(betas))

    epsilon = np.matrix(np.random.randn(total_samples, 1))

    SNR_factor = 1.0 / desired_signal_noise_ratio / np.linalg.norm(epsilon) * np.linalg.norm(X * beta)
    y = X * beta + SNR_factor * epsilon

    return _return_split_dataset(X, y, train_size, validate_size, betas)


# beta real: ones followed by zeros
def smooth_plus_linear(train_size, num_features, num_nonzero_features, data_type=0, linear_to_smooth_ratio=1, desired_signal_noise_ratio=4):
    print "DATA TYPE", data_type
    NOISE_STD = 1.0/16
    validate_size = _get_validate_size(train_size)
    total_samples = train_size + validate_size + TEST_SIZE
    MIN_X_SMOOTH = 0.0
    MAX_X_SMOOTH = 1.0

    NUM_NONZERO_FEATURE_GROUPS = 2
    nonzero_feature_group_size = num_nonzero_features / NUM_NONZERO_FEATURE_GROUPS
    X1 = _get_tiled_matrix(total_samples, nonzero_feature_group_size) + np.random.randn(total_samples, nonzero_feature_group_size) * NOISE_STD
    X1 += np.random.randn(total_samples, nonzero_feature_group_size) * NOISE_STD

    X2 = _get_tiled_matrix(total_samples, nonzero_feature_group_size) + np.random.randn(total_samples, nonzero_feature_group_size) * NOISE_STD
    X2 += np.random.randn(total_samples, nonzero_feature_group_size) * NOISE_STD

    X3 = np.random.randn(total_samples, num_features - nonzero_feature_group_size * NUM_NONZERO_FEATURE_GROUPS)
    X_linear = np.matrix(np.hstack((X1, X2, X3)))

    beta_real = np.matrix(
        np.concatenate((
            np.ones((num_nonzero_features, 1)) * 0.2,
            np.zeros((num_features - num_nonzero_features, 1))
        ))
    )

    X_smooth = np.matrix(np.random.uniform(MIN_X_SMOOTH, MAX_X_SMOOTH, total_samples)).T
    if data_type == 0:
        # y_smooth = 5 * np.sin(X_smooth * 24)
        y_smooth = np.sin(X_smooth * 5) + np.sin(15 * (X_smooth - 3))
    elif data_type == 1:
        y_smooth = np.matrix(np.multiply(MAX_X_SMOOTH + 1 - X_smooth, np.sin(20 * np.power(X_smooth, 4))))
    elif data_type == 2:
        y_smooth = 4 * np.power(X_smooth, 3) - 4 * np.power(X_smooth, 2) + X_smooth
    elif data_type == 3:
        y_smooth = np.matrix(np.multiply(np.power(X_smooth - 1, 2), np.sin(10 * np.power(X_smooth - 1, 2))))
    else:
        y_smooth = np.abs(X_smooth - 0.5)
        # y_smooth = 10 * np.power(X_smooth, 3) + 2 * np.power(X_smooth, 2) - 3 * X_smooth

    y_smooth *= 1.0 / linear_to_smooth_ratio * np.linalg.norm(X_linear * beta_real) / np.linalg.norm(y_smooth)
    epsilon = np.matrix(np.random.randn(total_samples, 1))

    SNR_factor = desired_signal_noise_ratio / np.linalg.norm(X_linear * beta_real + y_smooth) * np.linalg.norm(epsilon)
    y = X_linear * beta_real + y_smooth + 1.0 / SNR_factor * epsilon

    Xl_train, Xl_validate, Xl_test = _split_data(X_linear, train_size, validate_size)
    Xs_train, Xs_validate, Xs_test = _split_data(X_smooth, train_size, validate_size)
    y_train, y_validate, y_test = _split_y_vector(y, train_size, validate_size)
    y_smooth_train, y_smooth_validate, y_smooth_test = _split_y_vector(y_smooth, train_size, validate_size)

    return beta_real, Xl_train, Xs_train, y_train, Xl_validate, Xs_validate, y_validate, Xl_test, Xs_test, y_test, y_smooth_train, y_smooth_validate, y_smooth_test

def multi_smooth_features(train_size, smooth_fcn_list, desired_snr=2, feat_range=[0,1], train_to_validate_ratio=15, test_size=20):
    validate_size = np.floor(train_size/train_to_validate_ratio)
    total_samples = train_size + validate_size + test_size
    num_features = len(smooth_fcn_list)

    X_smooth = np.random.uniform(feat_range[0], feat_range[1], (total_samples, num_features))
    y_smooth = 0
    for idx, fcn in enumerate(smooth_fcn_list):
        y_smooth += fcn(X_smooth[:, idx]).reshape(total_samples, 1)

    epsilon = np.matrix(np.random.randn(total_samples, 1))
    SNR_factor = desired_snr / np.linalg.norm(y_smooth) * np.linalg.norm(epsilon)
    y = y_smooth + 1.0 / SNR_factor * epsilon

    X_train, X_validate, X_test = _split_data(X_smooth, train_size, validate_size)
    y_train, y_validate, y_test = _split_y_vector(y, train_size, validate_size)

    return X_train, y_train, X_validate, y_validate, X_test, y_test

def _return_split_dataset(X, y, train_size, validate_size, beta_real):
    X_train, X_validate, X_test = _split_data(X, train_size, validate_size)
    y_train, y_validate, y_test = _split_y_vector(y, train_size, validate_size)

    return beta_real, X_train, y_train, X_validate, y_validate, X_test, y_test


def _return_split_effects_interactions_dataset(X, W, y, train_size, validate_size, beta_real, theta_real):
    X_train, X_validate, X_test = _split_data(X, train_size, validate_size)
    W_train, W_validate, W_test = _split_data(W, train_size, validate_size)
    y_train, y_validate, y_test = _split_y_vector(y, train_size, validate_size)

    return beta_real, theta_real, X_train, W_train, y_train, X_validate, W_validate, y_validate, X_test, W_test, y_test

def _split_y_vector(y, train_size, validate_size):
    return y[0:train_size], y[train_size:train_size + validate_size], y[train_size + validate_size:]

def _split_data(X, train_size, validate_size):
    return X[0:train_size, :], X[train_size:train_size + validate_size, :], X[train_size + validate_size:, :]

# Returns a new beta that is rescaled to have the correct signal noise ratio
# Signal noise ratio is the variance of the signal over the variance of the noise
# Here, we assume the variance of the noise is 1
def _get_rescaled_beta(desired_signal_noise_ratio, X, beta, epsilon):
    return beta * desired_signal_noise_ratio / np.linalg.norm(X * beta) * np.linalg.norm(epsilon)

def _get_validate_size(train_size):
    return int(train_size / TRAIN_TO_VALIDATE_RATIO)

def _get_correlated_matrix(corr, total_samples, num_features):
    if corr == 0:
        return np.matrix(np.random.randn(total_samples, num_features))
    else:
        correlation_matrix = np.matrix([[math.pow(corr, abs(i - j)) for i in range(0, num_features)] for j in range(0, num_features)])
        return np.matrix(np.random.randn(total_samples, num_features)) * np.matrix(np.linalg.cholesky(correlation_matrix)).T

def _get_tiled_matrix(total_samples, length):
    return np.tile(np.random.randn(total_samples, 1), (1, length))
