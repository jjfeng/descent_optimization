import datetime
import matplotlib.pyplot as plt

from common import *
import data_generation
import hillclimb_elasticnet_lambda12
import hillclimb_elasticnet_lambda_alpha
import gridsearch_elasticnet_lambda12

SIMPLE = "SIMPLE"
CORRELATED = "CORRELATED"
THREE_GROUPS = "THREE_GROUPS"

def make_plot(data_type, train_size=0, num_features=0, num_nonzero_features=0, signal_noise_ratio=1):
    plt.clf()

    def _get_data():
        if data_type == SIMPLE:
            return data_generation.simple(
                train_size, num_features, num_nonzero_features, signal_noise_ratio=signal_noise_ratio)
        elif data_type == CORRELATED:
            return data_generation.correlated(
                train_size, num_features, num_nonzero_features, signal_noise_ratio=signal_noise_ratio)
        elif data_type == THREE_GROUPS:
            return data_generation.three_groups(
                train_size, num_features, num_nonzero_features, signal_noise_ratio=signal_noise_ratio)
        else:
            raise Exception("bad data")

    beta_real, X_train, y_train, X_validate, y_validate, X_test, y_test = _get_data()

    hc_lambda12_beta_guess, hc_lambda12_cp = hillclimb_elasticnet_lambda12.run(X_train, y_train, X_validate, y_validate, diminishing_step_size=False)
    plt.plot(hc_lambda12_cp, label=HC_LAMBDA12_LABEL, color=HC_LAMBDA12_COLOR)

    hc_lambda12_beta_guess, hc_lambda12_cp = hillclimb_elasticnet_lambda12.run(X_train, y_train, X_validate, y_validate, diminishing_step_size=False, do_shrink=True)
    plt.plot(hc_lambda12_cp, label=HC_LAMBDA12_LABEL + "_SHRINK", color="purple")

    # hc_lambda12_dim_beta_guess, hc_lambda12_dim_cp = hillclimb_elasticnet_lambda12.run(X_train, y_train, X_validate, y_validate, diminishing_step_size=True)
    # plt.plot(hc_lambda12_dim_cp, label=HC_LAMBDA12_DIM_LABEL, color=HC_LAMBDA12_DIM_COLOR)

    # hc_lambda12_nesterov_beta_guess, hc_lambda12_nesterov_cp = hillclimb_elasticnet_lambda12.run_nesterov(X_train, y_train, X_validate, y_validate)
    # plt.plot(hc_lambda12_nesterov_cp, label=HC_LAMBDA12_NESTEROV_LABEL, color=HC_LAMBDA12_NESTEROV_COLOR)
    #
    hc_lambda_alpha_beta_guess, hc_lambda_alpha_cp = hillclimb_elasticnet_lambda_alpha.run(X_train, y_train, X_validate, y_validate, diminishing_step_size=False)
    plt.plot(hc_lambda_alpha_cp, label=HC_LAMBDA_ALPHA_LABEL, color=HC_LAMBDA_ALPHA_COLOR)

    hc_lambda_alpha_beta_guess, hc_lambda_alpha_cp = hillclimb_elasticnet_lambda_alpha.run(X_train, y_train, X_validate, y_validate, diminishing_step_size=False, do_shrink=True)
    plt.plot(hc_lambda_alpha_cp, label=HC_LAMBDA_ALPHA_LABEL+"_SHRINK", color="orange")

    #
    # hc_lambda_alpha_dim_beta_guess, hc_lambda_alpha_dim_cp = hillclimb_elasticnet_lambda_alpha.run(X_train, y_train, X_validate, y_validate, diminishing_step_size=True)
    # plt.plot(hc_lambda_alpha_dim_cp, label=HC_LAMBDA_ALPHA_DIM_LABEL, color=HC_LAMBDA_ALPHA_DIM_COLOR)
    #
    # hc_lambda_alpha_nesterov_beta_guess, hc_lambda_alpha_nesterov_cp = hillclimb_elasticnet_lambda_alpha.run_nesterov(X_train, y_train, X_validate, y_validate)
    # plt.plot(hc_lambda_alpha_nesterov_cp, label=HC_LAMBDA_ALPHA_NESTEROV_LABEL, color=HC_LAMBDA_ALPHA_NESTEROV_COLOR)

    gs_beta_guess = gridsearch_elasticnet_lambda12.run(X_train, y_train, X_validate, y_validate)
    gs_best_test_err = testerror(X_validate, y_validate, gs_beta_guess)
    plt.axhline(gs_best_test_err, label=GS_LAMBDA12_LABEL, color=GS_COLOR)

    plt.legend(fontsize="x-small")
    plt.title("Train=%d nonzero=%d feature=%d SNR=%.1f" % (train_size, num_nonzero_features, num_features, signal_noise_ratio))
    plt.xlabel("Number of iterations")
    plt.ylabel("Validation test error")

    # plt.savefig("figures/elasticnet_%s_%d_%d_%d_%.1f_%s.png" % (data_type, train_size, num_features, num_nonzero_features, signal_noise_ratio, datetime.datetime.now().isoformat()))
    plt.savefig("figures/test.png")

    def _get_test_beta_err(beta_guess):
        test_err = testerror(X_test, y_test, beta_guess)
        beta_err = betaerror(beta_real, beta_guess)
        return (test_err, beta_err)

    print HC_LAMBDA12_LABEL, _get_test_beta_err(hc_lambda12_beta_guess)
    print HC_LAMBDA12_DIM_LABEL, _get_test_beta_err(hc_lambda12_dim_beta_guess)
    print HC_LAMBDA12_NESTEROV_LABEL, _get_test_beta_err(hc_lambda12_nesterov_beta_guess)
    print HC_LAMBDA_ALPHA_LABEL, _get_test_beta_err(hc_lambda_alpha_beta_guess)
    print HC_LAMBDA_ALPHA_DIM_LABEL, _get_test_beta_err(hc_lambda_alpha_dim_beta_guess)
    print HC_LAMBDA_ALPHA_NESTEROV_LABEL, _get_test_beta_err(hc_lambda_alpha_nesterov_beta_guess)
    print GS_LAMBDA12_LABEL, _get_test_beta_err(gs_beta_guess)

    return beta_real, X_train, y_train, X_validate, y_validate, X_test, y_test

NUM_FEATURES = 200
NUM_NONZERO_FEATURES = 10
TRAIN_SIZE = 50

# TRAIN_SIZE = 40
# NUM_FEATURES = 150
# NUM_NONZERO_FEATURES = 10


SNR = [2] # [1, 2, 5]
for snr in SNR:
    # make_plot(THREE_GROUPS, train_size=TRAIN_SIZE, num_features=NUM_FEATURES, num_nonzero_features=NUM_NONZERO_FEATURES, signal_noise_ratio=snr)
    make_plot(CORRELATED, train_size=TRAIN_SIZE, num_features=NUM_FEATURES, num_nonzero_features=NUM_NONZERO_FEATURES, signal_noise_ratio=snr)
