import time
import numpy as np
import matplotlib.pyplot as plt

from convexopt_solvers import GenAddModelProblemWrapper
from data_generation import multi_smooth_features

from gen_add_model_hillclimb import GenAddModelHillclimb
from method_results import MethodResults
from method_results import MethodResult
import gen_add_model_gridsearch as gs

from common import *

np.set_printoptions(edgeitems=3, infstr='inf', linewidth=200, nanstr='nan', precision=8, suppress=False, threshold=1000, formatter=None)

##########
np.random.seed(10)
FEATURE_RANGE = [-10.0, 10.0]

NUM_RUNS = 10
NUM_FUNCS = 3
TRAIN_SIZE = 150
SNR = 2
VALIDATE_RATIO = 3
NUM_TEST = 60
NUM_GS_LAMBDAS = 4
MAX_LAMBDA = 50
TEST_HC_LAMBDAS = [5]

# NUM_RUNS = 10
# NUM_FUNCS = 2
# TRAIN_SIZE = 120
# SNR = 2
# VALIDATE_RATIO = 3
# NUM_TEST = 35
# NUM_GS_LAMBDAS = 4
# MAX_LAMBDA = 50
# TEST_HC_LAMBDAS = [1]

# NUM_RUNS = 3
# NUM_FUNCS = 4
# TRAIN_SIZE = 120
# SNR = 1
# VALIDATE_RATIO = 4
# NUM_TEST = 40
# NUM_GS_LAMBDAS = 4
# MAX_LAMBDA = 30
# TEST_HC_LAMBDAS = [0.1, 1, 10]


DEBUG = False
PLOT_RUNS = True

def identity_fcn(x):
    return x.reshape(x.size, 1)

def big_sin(x):
    return identity_fcn(5 * np.sin(x*3))

def crazy_down_sin(x):
    return identity_fcn(x * np.sin(x) - x)

def exp_small(x):
    return identity_fcn(np.exp(x * 0.6) - 8)

def log_func(x):
    return identity_fcn(np.log(x) * 5 + 10)

def _hillclimb_coarse_grid_search(X_train, y_train, X_validate, y_validate, X_test, smooth_fcn_list):
    start_time = time.time()
    best_cost = np.inf
    best_thetas = []
    best_cost_path = []
    best_regularization = []
    best_start_lambda = []
    for lam in TEST_HC_LAMBDAS:
        hc = GenAddModelHillclimb(X_train, y_train, X_validate, y_validate, X_test)
        init_lambdas = np.array([lam for i in range(len(smooth_fcn_list))])
        thetas, cost_path, curr_regularization = hc.run(init_lambdas, debug=DEBUG)

        if thetas is not None and best_cost > cost_path[-1]:
            best_cost = cost_path[-1]
            best_thetas = thetas
            best_cost_path = cost_path
            best_start_lambda = lam
            best_regularization = curr_regularization
            print "better cost", best_cost, "better regularization", best_regularization

    print "HC: best_cost", best_cost, "best_regularization", best_regularization, "best start lambda: ", best_start_lambda
    end_time = time.time()
    print "runtime", end_time - start_time
    return best_thetas, best_cost_path, end_time - start_time

def plot_res(fitted_thetas, fcn_list, X, y, outfile="figures/threegam/out.png", out_y_file="figures/threegam/out_y.png"):
    colors = ["green", "blue", "red", "purple", "orange", "black", "brown"]
    print "outfile", outfile
    num_features = fitted_thetas.shape[1]
    plt.clf()
    for i in range(num_features):
        x_features = X[:,i]
        fcn = fcn_list[i]
        order_x = np.argsort(x_features)
        plt.plot(
            x_features[order_x],
            fcn(x_features[order_x]), # true values
            '-',
            x_features[order_x],
            fitted_thetas[order_x,i], # fitted values
            '--',
            label="feat %d" % i,
            color=colors[i],
        )
    plt.savefig(outfile)

    # plt.clf()
    # fitted_y = np.sum(fitted_thetas, axis=1)
    # plt.plot(y, fitted_y, '.')
    # all_ys = np.vstack((y, np.sum(fitted_thetas, axis=1)))
    # feat_range = [np.min(all_ys) - 1, np.max(all_ys) + 1]
    # plt.plot(feat_range, feat_range, 'k-')
    # plt.plot()
    # plt.savefig(out_y_file)

    plt.clf()
    fitted_y = np.sum(fitted_thetas, axis=1)
    true_y = 0
    for i in range(num_features):
        fcn = fcn_list[i]
        print fcn(X[:,i]).shape
        true_y += fcn(X[:,i])

    for i in range(num_features):
        x_features = X[:,i]
        fcn = fcn_list[i]
        order_x = np.argsort(x_features)
        plt.plot(
            x_features[order_x],
            true_y[order_x], # true values
            '-',
            x_features[order_x], # observed x
            fitted_y[order_x], # fitted y
            '--',
            x_features[order_x], # observed x
            y[order_x], # observed values
            '-.',
            color=colors[i],
        )
    plt.savefig(out_y_file)


def main():
    SMOOTH_FCNS = [big_sin, identity_fcn, crazy_down_sin, exp_small, log_func]
    smooth_fcn_list = SMOOTH_FCNS[:NUM_FUNCS]

    hc_results = MethodResults("Hillclimb")
    gs_results = MethodResults("Gridsearch")

    for i in range(0, NUM_RUNS):
        X_train, y_train, X_validate, y_validate, X_test, y_test = multi_smooth_features(
            TRAIN_SIZE,
            smooth_fcn_list,
            desired_snr=SNR,
            feat_range=FEATURE_RANGE,
            train_to_validate_ratio=VALIDATE_RATIO,
            test_size=NUM_TEST
        )

        def _create_method_result(best_thetas, runtime):
            test_err = testerror_multi_smooth(y_test, test_idx, best_thetas)
            validate_err = testerror_multi_smooth(y_validate, validate_idx, best_thetas)
            return MethodResult(test_err=test_err, validation_err=validate_err, runtime=runtime)

        X_full, train_idx, validate_idx, test_idx = GenAddModelHillclimb.stack((X_train, X_validate, X_test))

        hc_thetas, hc_cost_path, hc_time = _hillclimb_coarse_grid_search(X_train, y_train, X_validate, y_validate, X_test, smooth_fcn_list)
        hc_results.append(_create_method_result(hc_thetas, hc_time))
        if PLOT_RUNS:
            plot_res(
                hc_thetas[test_idx], smooth_fcn_list, X_test, y_test,
                outfile="figures/threegam/out_hc_f%d.png" % NUM_FUNCS,
                out_y_file="figures/threegam/out_y_hc_f%d.png" % NUM_FUNCS,
            )
            plot_res(
                hc_thetas[validate_idx], smooth_fcn_list, X_validate, y_validate,
                outfile="figures/threegam/out_hcv_f%d.png" % NUM_FUNCS,
                out_y_file="figures/threegam/out_yv_hc_f%d.png" % NUM_FUNCS,
            )
            plot_res(
                hc_thetas[train_idx], smooth_fcn_list, X_train, y_train,
                outfile="figures/threegam/out_hct_f%d.png" % NUM_FUNCS,
                out_y_file="figures/threegam/out_yt_hc_f%d.png" % NUM_FUNCS,
            )

        print "=================================================="

        start_time = time.time()
        gs_thetas, best_lambdas = gs.run(
            y_train,
            y_validate,
            X_full,
            train_idx,
            validate_idx,
            num_lambdas=NUM_GS_LAMBDAS,
            max_lambda=MAX_LAMBDA
        )
        gs_runtime = time.time() - start_time
        gs_results.append(_create_method_result(gs_thetas, gs_runtime))

        gs_test_error = testerror_multi_smooth(y_test, test_idx, gs_thetas)
        if PLOT_RUNS:
            plot_res(
                gs_thetas[test_idx], smooth_fcn_list, X_test, y_test,
                outfile="figures/threegam/out_gs_f%d.png" % NUM_FUNCS,
                out_y_file="figures/threegam/out_y_gs_f%d.png" % NUM_FUNCS,
            )

        hc_results.print_results()
        gs_results.print_results()

if __name__ == "__main__":
    main()
