import time
import matplotlib.pyplot as plt
from common import *
import data_generation
import hillclimb_elasticnet_lambda12 as hc
import hillclimb_elasticnet_lambda_alpha
import gridsearch_elasticnet_lambda12
import neldermead_elasticnet as nm
from method_results import MethodResult
from method_results import MethodResults

GENERATE_PLOT = False #True
NUM_RUNS = 1 # 30

SIGNAL_NOISE_RATIO = 2

NUM_FEATURES = 250
NUM_NONZERO_FEATURES = 15
TRAIN_SIZE = 80

# TRAIN_SIZE = 40
# NUM_FEATURES = 100
# NUM_NONZERO_FEATURES = 10

COARSE_LAMBDA_GRID = [1e-2, 1e1]
NUM_RANDOM_LAMBDAS = 3

seed = int(np.random.rand() * 1e5)
seed = 10
np.random.seed(seed)
print "SEED", seed

def _hillclimb_coarse_grid_search(optimization_func, *args, **kwargs):
    start_time = time.time()
    best_cost = 1e10
    best_beta = []
    best_start_lambdas = []
    for init_lambda1 in COARSE_LAMBDA_GRID:
        kwargs["initial_lambda1"] = init_lambda1
        kwargs["initial_lambda2"] = init_lambda1
        beta_guess, cost_path = optimization_func(*args, **kwargs)
        validation_cost = testerror(X_validate, y_validate, beta_guess)
        if best_cost > validation_cost:
            best_start_lambdas = [kwargs["initial_lambda1"], kwargs["initial_lambda2"]]
            best_cost = validation_cost
            best_beta = beta_guess
            best_cost_path = cost_path

    end_time = time.time()
    print "HC: BEST best_cost", best_cost, "best_start_lambdas", best_start_lambdas
    return beta_guess, cost_path, end_time - start_time

hc_results = MethodResults(HC_LAMBDA12_LABEL)
hc_results1 = MethodResults(HC_LAMBDA12_LABEL + "_SHRINK")
hc_dim_results = MethodResults(HC_LAMBDA12_DIM_LABEL)
hc_nesterov_results = MethodResults(HC_LAMBDA12_NESTEROV_LABEL)
hc_lambda_alpha_results = MethodResults(HC_LAMBDA_ALPHA_LABEL)
hc_lambda_alpha_results1 = MethodResults(HC_LAMBDA_ALPHA_LABEL + "_SHRINK")
hc_lambda_alpha_dim_results = MethodResults(HC_LAMBDA_ALPHA_DIM_LABEL)
hc_lambda_alpha_nesterov_results = MethodResults(HC_LAMBDA_ALPHA_NESTEROV_LABEL)
nm_results = MethodResults("NELDER-MEAD")
gs_results = MethodResults(GS_LAMBDA12_LABEL)
for i in range(0, NUM_RUNS):
    beta_real, X_train, y_train, X_validate, y_validate, X_test, y_test = data_generation.correlated(
        TRAIN_SIZE, NUM_FEATURES, NUM_NONZERO_FEATURES, signal_noise_ratio=SIGNAL_NOISE_RATIO)

    def _create_method_result(beta_guess, runtime):
        test_err = testerror(X_test, y_test, beta_guess)
        validation_err = testerror(X_validate, y_validate, beta_guess)
        beta_err = betaerror(beta_real, beta_guess)
        print "FINAL validation_err", validation_err
        return MethodResult(test_err=test_err, beta_err=beta_err, validation_err=validation_err, runtime=runtime)

    hc_beta_guess, hc_cost_path, runtime = _hillclimb_coarse_grid_search(hc.run, X_train, y_train, X_validate, y_validate, diminishing_step_size=False)
    hc_results.append(_create_method_result(hc_beta_guess, runtime))

    # hc_nesterov_beta_guess, hc_nesterov_cost_path, runtime = _hillclimb_coarse_grid_search(hc.run_nesterov, X_train, y_train, X_validate, y_validate)
    # hc_nesterov_results.append(_create_method_result(hc_nesterov_beta_guess, runtime))

    # hc_lambda_alpha_beta_guess, _ = hillclimb_elasticnet_lambda_alpha.run(X_train, y_train, X_validate, y_validate, diminishing_step_size=False, do_shrink=False)
    # hc_lambda_alpha_results.append_test_beta_err(_get_test_beta_err(hc_lambda_alpha_beta_guess))
    #
    # hc_lambda_alpha_beta_guess1, _ = hillclimb_elasticnet_lambda_alpha.run(X_train, y_train, X_validate, y_validate, diminishing_step_size=False, do_shrink=True)
    # hc_lambda_alpha_results1.append_test_beta_err(_get_test_beta_err(hc_lambda_alpha_beta_guess1))
    #
    # hc_lambda_alpha_dim_beta_guess, _ = hillclimb_elasticnet_lambda_alpha.run(X_train, y_train, X_validate, y_validate, diminishing_step_size=True)
    # hc_lambda_alpha_dim_results.append_test_beta_err(_get_test_beta_err(hc_lambda_alpha_dim_beta_guess))
    #
    # hc_lambda_alpha_nesterov_beta_guess, _ = hillclimb_elasticnet_lambda_alpha.run_nesterov(X_train, y_train, X_validate, y_validate)
    # hc_lambda_alpha_nesterov_results.append_test_beta_err(_get_test_beta_err(hc_lambda_alpha_nesterov_beta_guess))

    nm_beta_guess, runtime = nm.run(X_train, y_train, X_validate, y_validate)
    nm_method_result = _create_method_result(nm_beta_guess, runtime)
    nm_results.append(nm_method_result)

    # start = time.time()
    # gs_beta_guess = gridsearch_elasticnet_lambda12.run(X_train, y_train, X_validate, y_validate)
    # runtime = time.time() - start
    # gs_method_result = _create_method_result(gs_beta_guess, runtime)
    # gs_results.append(gs_method_result)

    print "NUM RUN", i
    print "NUM FEATURES", NUM_FEATURES
    print "NUM NONZERO FEATURES", NUM_NONZERO_FEATURES
    print "TRAIN SIZE", TRAIN_SIZE

    hc_results.print_results()
    # hc_nesterov_results.print_results()
    # hc_dim_results.print_results()
    # hc_lambda_alpha_results.print_results()
    # hc_lambda_alpha_results1.print_results()
    # hc_lambda_alpha_dim_results.print_results()
    # hc_lambda_alpha_nesterov_results.print_results()
    nm_results.print_results()
    gs_results.print_results()

    if GENERATE_PLOT and i == 0:
        plt.clf()
        plt.plot(np.array(beta_real), '.', label="Real", color="green")
        plt.plot(np.array(gs_beta_guess), '.', label="Gridsearch", color="brown")
        plt.plot(np.array(hc_beta_guess), '.', label="Gradient Descent", color="red")
        plt.plot(np.array(hc_nesterov_beta_guess), '.', label="Nesterov", color="blue")
        plt.title("Beta comparison")
        plt.xlabel("Beta index")
        plt.ylabel("Beta value")
        plt.legend(fontsize="x-small")
        plt.xlim([-1, NUM_FEATURES + 1])
        plt.ylim([(np.min(beta_real) - 0.01) * 2, (np.max(beta_real) + 0.01) * 2])
        plt.savefig("figures/elasticnet_betas_%d_%d_%d.png" % (TRAIN_SIZE, NUM_FEATURES, NUM_NONZERO_FEATURES))

        plt.clf()
        plt.plot(hc_cost_path, label="Gradient Descent", color="red")
        plt.plot(hc_nesterov_cost_path, label="Nesterov's Gradient Descent", color="blue")
        plt.axhline(gs_method_result.validation_err * y_validate.size, label="Grid Search", color="brown")
        plt.legend(fontsize="x-small")
        # plt.title("Train=%d, p=%d, nonzero=%d" % (TRAIN_SIZE, NUM_FEATURES, NUM_NONZERO_FEATURES))
        plt.xlabel("Number of iterations")
        plt.ylabel("Validation error")
        plt.xticks(np.arange(0, max(len(hc_cost_path), len(hc_nesterov_cost_path)), 1.0))
        plt.savefig("figures/elasticnet_costpath_%d_%d_%d.png" % (TRAIN_SIZE, NUM_FEATURES, NUM_NONZERO_FEATURES))
