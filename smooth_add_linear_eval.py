import sys
import getopt
import matplotlib.pyplot as plt
import time
import cvxpy
from common import *
from method_results import MethodResult
from method_results import MethodResults
from data_generation import smooth_plus_linear
import hillclimb_smooth_add_linear as hc
import gridsearch_smooth_add_linear as gs

GENERATE_PLOT = False #True
NUM_RUNS = 30

NUM_TRAIN = 100
NUM_FEATURES = 20
NUM_NONZERO_FEATURES = 6

# NUM_TRAIN = 68
# NUM_FEATURES = 12
# NUM_NONZERO_FEATURES = 4

SIGNAL_TO_NOISE = 2
LINEAR_TO_SMOOTH_RATIO = 2
MAX_BEST_COST = 1e10

COARSE_LAMBDAS = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10]

RUN_GRIDSEARCH_FULL = False
RUN_NESTEROV = False
RUN_HC = True
RUN_GRIDSEARCH_2PENALTIES = True

def _hillclimb_coarse_grid_search(optimization_func, *args, **kwargs):
    start_time = time.time()
    best_cost = MAX_BEST_COST
    best_beta = []
    best_thetas = []
    best_cost_path = []
    best_regularization = []
    best_start_lambda = []
    for l1 in COARSE_LAMBDAS:
        for l2 in [None]: # COARSE_LAMBDAS:
            for l3 in [None]: # COARSE_LAMBDAS:
                kwargs["initial_lambda1"] = l1
                kwargs["initial_lambda2"] = l1
                kwargs["initial_lambda3"] = l1

                try:
                    beta, thetas, cost_path, curr_regularization = optimization_func(*args, **kwargs)
                except cvxpy.error.SolverError as e:
                    print "HC ERROR: CANT SOLVE THIS REGULARIZATION PARAM.", e
                    continue

                if beta is not None and thetas is not None and best_cost > cost_path[-1]:
                    best_cost = cost_path[-1]
                    best_beta = beta
                    best_thetas = thetas
                    best_cost_path = cost_path
                    best_start_lambda = [l1, l2, l3]
                    best_regularization = curr_regularization
                    print "better cost", best_cost, "better regularization", best_regularization

    print "HC: best_cost", best_cost, "best_regularization", best_regularization, "best start lambda: ", best_start_lambda
    end_time = time.time()
    print "runtime", end_time - start_time
    return best_beta, best_thetas, best_cost_path, end_time - start_time

def _get_ordered_Xl_y_data(Xl, Xs, y):
    indices = np.argsort(np.reshape(np.array(Xs), Xs.size), axis=0)
    return Xl[indices, :], y[indices]

def main(argv):
    try:
        opts, args = getopt.getopt(argv,"d:p")
    except getopt.GetoptError:
        print "BAD REQUEST"
        print "accepts a folder name. reads the XML files inside"
        sys.exit(2)

    RUN_HC_POOLED = False
    for opt, arg in opts:
        if opt == '-d':
            data_type = int(arg)
            assert(data_type in [0, 1, 2])

    seed = int(np.random.rand() * 1e5)
    print "numpy rand seed", seed
    np.random.seed(seed)

    hc_results = MethodResults("Hillclimb")
    hc_nesterov_results = MethodResults("Hillclimb_NESTEROV")
    gs_results = MethodResults("Gridsearch")
    gs_l2_results = MethodResults("Gridsearch_l2")

    for i in range(0, NUM_RUNS):
        beta_real, Xl_train, Xs_train, y_train, Xl_validate, Xs_validate, y_validate, Xl_test, Xs_test, y_test, y_smooth_train, y_smooth_validate, y_smooth_test = smooth_plus_linear(
            NUM_TRAIN,
            NUM_FEATURES,
            NUM_NONZERO_FEATURES,
            data_type=data_type,
            linear_to_smooth_ratio=LINEAR_TO_SMOOTH_RATIO,
            desired_signal_noise_ratio=SIGNAL_TO_NOISE)

        tot_size = Xs_train.size + Xs_validate.size + Xs_test.size
        Xs_combined = np.reshape(np.array(np.vstack((Xs_train, Xs_validate, Xs_test))), tot_size)
        order_indices = np.argsort(Xs_combined, axis=0)

        test_indices = np.greater_equal(order_indices, Xs_train.size + Xs_validate.size)
        validate_indices = np.logical_and(np.greater_equal(order_indices, Xs_train.size), np.logical_not(test_indices))
        train_validate_indices = np.logical_not(test_indices)

        Xs_ordered = Xs_combined[order_indices]
        true_y_smooth = np.matrix(np.vstack((y_smooth_train, y_smooth_validate, y_smooth_test))[order_indices])

        Xl_test_ordered, y_test_ordered = _get_ordered_Xl_y_data(Xl_test, Xs_test, y_test)
        Xl_validate_ordered, y_validate_ordered = _get_ordered_Xl_y_data(Xl_validate, Xs_validate, y_validate)

        def _create_method_result(best_beta, best_thetas, runtime):
            beta_err = betaerror(beta_real, best_beta)
            theta_err = betaerror(true_y_smooth, best_thetas)
            test_err = testerror_smooth_and_linear(Xl_test_ordered, y_test_ordered, best_beta, best_thetas[test_indices])
            validation_err = testerror_smooth_and_linear(Xl_validate_ordered, y_validate_ordered, best_beta, best_thetas[validate_indices])
            print "(test_err, beta_err, theta_err)", (test_err, beta_err, theta_err)
            return MethodResult(test_err=test_err, beta_err=beta_err, theta_err=theta_err, validation_err=validation_err, runtime=runtime)

        if RUN_HC:
            hc_beta, hc_thetas, hc_cost_path, runtime = _hillclimb_coarse_grid_search(hc.run, Xl_train, Xs_train, y_train, Xl_validate, Xs_validate, y_validate, Xs_test)
            if hc_beta is None or hc_thetas is None:
                print "THERE WERE SOME SERIOUS CONVERGENCE ISSUES"
                continue
            hc_results.append(_create_method_result(hc_beta, hc_thetas, runtime))

        if RUN_NESTEROV:
            hc_nesterov_beta, hc_nesterov_thetas, hc_nesterov_cost_path, runtime = _hillclimb_coarse_grid_search(hc.run_nesterov, Xl_train, Xs_train, y_train, Xl_validate, Xs_validate, y_validate, Xs_test)
            hc_nesterov_results.append(_create_method_result(hc_nesterov_beta, hc_nesterov_thetas, runtime))

        # if RUN_GRIDSEARCH_FULL:
        #     start_time = time.time()
        #     gs_beta, gs_thetas, gs_best_cost = gs.run(Xl_train, Xs_train, y_train, Xl_validate, Xs_validate, y_validate, Xs_test)
        #     runtime = time.time() - start_time
        #     gs_results.append(_create_method_result(gs_beta, gs_thetas, runtime))

        if RUN_GRIDSEARCH_2PENALTIES:
            start_time = time.time()
            gs_l2_beta, gs_l2_thetas, gs_l2_best_cost = gs.run_simple(Xl_train, Xs_train, y_train, Xl_validate, Xs_validate, y_validate, Xs_test, use_l1=False)
            gs_l2_results.append_runtime(time.time() - start_time)
            gs_l2_results.append(_create_method_result(gs_l2_beta, gs_l2_thetas, runtime))

        print "DATA_TYPE", data_type, "NUM RUN", hc_results.get_num_runs()
        print "NUM_TRAIN", NUM_TRAIN
        print "NUM_FEATURES", NUM_FEATURES
        print "NUM_NONZERO_FEATURES", NUM_NONZERO_FEATURES
        print "SIGNAL_TO_NOISE", SIGNAL_TO_NOISE
        print "LINEAR_TO_SMOOTH_RATIO", LINEAR_TO_SMOOTH_RATIO

        if RUN_GRIDSEARCH_FULL:
            gs_results.print_results()
        if RUN_GRIDSEARCH_2PENALTIES:
            gs_l2_results.print_results()
        if RUN_HC:
            hc_results.print_results()
        if RUN_NESTEROV:
            hc_nesterov_results.print_results()

        sys.stdout.flush()

        if GENERATE_PLOT and i == 0:
            plt.clf()
            plt.plot(Xs_ordered, np.reshape(np.array(true_y_smooth), tot_size), label="Real", color="green")
            if RUN_GRIDSEARCH_2PENALTIES:
                plt.plot(Xs_ordered, np.reshape(np.array(gs_l2_thetas), tot_size), label="Gridsearch", color="brown")
            if RUN_GRIDSEARCH_FULL:
                plt.plot(Xs_ordered, np.reshape(np.array(gs_thetas), tot_size), label="Gridsearch", color="brown")
            if RUN_HC:
                plt.plot(Xs_ordered, np.reshape(np.array(hc_thetas), tot_size), label="Gradient Descent", color="red")
            if RUN_NESTEROV:
                plt.plot(Xs_ordered, np.reshape(np.array(hc_nesterov_thetas), tot_size), label="Nesterov's Gradient Descent", color="blue")
            plt.legend(fontsize="x-small")
            plt.xlabel("X smooth")
            plt.ylabel("Y smooth")
            # plt.title("Theta comparison, data %d" % data_type)
            plt.savefig("figures/smooth_linear_thetas_%d_%d_%d_%d.png" % (data_type, NUM_TRAIN, NUM_FEATURES, NUM_NONZERO_FEATURES))

            plt.clf()
            plt.plot(np.array(beta_real), '.', label="Real", color="green")
            if RUN_GRIDSEARCH_2PENALTIES:
                plt.plot(np.array(gs_l2_beta), '.', label="Gridsearch", color="brown")
            if RUN_GRIDSEARCH_FULL:
                plt.plot(np.array(gs_beta), '.', label="Gridsearch", color="brown")
            if RUN_HC:
                plt.plot(np.array(hc_beta), '.', label="Gradient Descent", color="red")
            if RUN_NESTEROV:
                plt.plot(np.array(hc_nesterov_beta), '.', label="Hillclimb Nesterov", color="orange")
            # plt.title("Beta comparison, data %d" % data_type)
            plt.xlabel("Beta index")
            plt.ylabel("Beta value")
            plt.legend(fontsize="x-small")
            plt.xlim([-1, NUM_FEATURES + 1])
            plt.ylim([(np.min(beta_real) - 0.01) * 1.5, (np.max(beta_real) + 0.01) * 1.5])
            plt.savefig("figures/smooth_linear_beta_%d_%d_%d_%d.png" % (data_type, NUM_TRAIN, NUM_FEATURES, NUM_NONZERO_FEATURES))

            if RUN_HC:
                plt.clf()
                if RUN_HC:
                    plt.plot(hc_cost_path, label="Hillclimb", color="red")
                if RUN_NESTEROV:
                    plt.plot(hc_nesterov_cost_path, label="Hillclimb Nesterov", color="orange")
                if RUN_GRIDSEARCH_FULL:
                    plt.axhline(gs_best_cost, label="Gridsearch", color="brown")
                plt.title("Validation Cost, data %d" % data_type)
                plt.xlabel("Iteration")
                plt.ylabel("Validation Cost")
                plt.legend(fontsize="x-small")
                plt.savefig("figures/smooth_linear_cost_path_%d_%d_%d_%d.png" % (data_type, NUM_TRAIN, NUM_FEATURES, NUM_NONZERO_FEATURES))

if __name__ == "__main__":
    main(sys.argv[1:])
