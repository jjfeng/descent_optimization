import getopt
import time
import sys
import traceback
import numpy as np
from multiprocessing import Pool

from iteration_models import Simulation_Settings, Iteration_Data

from elasticnet_hillclimb import Elastic_Net_Hillclimb
from elasticnet_neldermead import Elastic_Net_Nelder_Mead
from elasticnet_grid_search import Elastic_Net_Grid_Search
from elasticnet_spearmint import Elastic_Net_Spearmint

from data_generator import DataGenerator
from method_results import MethodResults
from method_results import MethodResult

from common import *

class Elastic_Net_Settings(Simulation_Settings):
    results_folder = "results/elastic_net"
    gs_eigen_factor = 4
    num_features = 250
    num_nonzero_features = 15
    train_size = 80
    validate_size = 40
    method = "HC"

    def print_settings(self):
        print "SETTINGS"
        obj_str = "method %s\n" % self.method
        obj_str += "t/v/t size %d/%d/%d\n" % (self.train_size, self.validate_size, self.test_size)
        obj_str += "snr %f\n" % self.snr
        obj_str += "sp runs %d\n" % self.spearmint_numruns
        obj_str += "nm_iters %d\n" % self.nm_iters
        print obj_str

def main(argv):
    seed = int(np.random.rand() * 1e5)
    seed = 10
    np.random.seed(seed)
    print "SEED", seed

    num_threads = 1
    num_runs = 30

    try:
        opts, args = getopt.getopt(argv,"f:z:a:b:c:s:m:r:t:")
    except getopt.GetoptError:
        print "Bad Arguments to python script"
        sys.exit(2)

    settings = Elastic_Net_Settings()
    for opt, arg in opts:
        if opt == '-f':
            settings.num_features = int(arg)
        elif opt == '-z':
            settings.num_nonzero_features = int(arg)
        elif opt == '-a':
            settings.train_size = int(arg)
        elif opt == '-b':
            settings.validate_size = int(arg)
        elif opt == '-c':
            settings.test_size = int(arg)
        elif opt == "-s":
            settings.snr = float(arg)
        elif opt == "-m":
            assert(arg in METHODS)
            settings.method = arg
        elif opt == "-t":
            num_threads = int(arg)
        elif opt == "-r":
            num_runs = int(arg)

    settings.print_settings()
    sys.stdout.flush()
    data_gen = DataGenerator(settings)

    run_data = []
    for i in range(num_runs):
        observed_data = data_gen.make_correlated(settings.num_features, settings.num_nonzero_features)
        run_data.append(Iteration_Data(i, observed_data, settings))

    if settings.method != "SP" and num_threads > 1:
        print "Do multiprocessing"
        pool = Pool(num_threads)
        results = pool.map(fit_data_for_iter_safe, run_data)
    else:
        print "Avoiding multiprocessing"
        results = map(fit_data_for_iter_safe, run_data)

    method_results = MethodResults(settings.method)
    num_crashes = 0
    for r in results:
        if r is not None:
            method_results.append(r)
        else:
            num_crashes += 1
    print "==========TOTAL RUNS %d============" % method_results.get_num_runs()
    method_results.print_results()
    print "num crashes %d" % num_crashes

#########
# FUNCTIONS FOR CHILD THREADS
#########
def fit_data_for_iter_safe(iter_data):
    result = None
    try:
        result = fit_data_for_iter(iter_data)
    except Exception as e:
        print "Exception caught in iter %d: %s" % (iter_data.i, e)
        traceback.print_exc()
    return result

def fit_data_for_iter(iter_data):
    settings = iter_data.settings
    one_vec = np.ones(2)
    initial_lambdas_set = [one_vec * 1e-2, one_vec * 1e1]
    method = iter_data.settings.method

    str_identifer = "%d_%d_%d_%d_%d_%d_%s_%d" % (
        settings.num_features,
        settings.num_nonzero_features,
        settings.train_size,
        settings.validate_size,
        settings.test_size,
        settings.snr,
        method,
        iter_data.i
    )
    log_file_name = "%s/tmp/log_%s.txt" % (settings.results_folder, str_identifer)
    print "log_file_name", log_file_name
    with open(log_file_name, "w") as f:
        if method == "NM":
            algo = Elastic_Net_Nelder_Mead(iter_data.data)
            algo.run(initial_lambdas_set, num_iters=settings.nm_iters, log_file=f)
        elif method == "GS":
            largest_eigenvalue = max(np.linalg.eigvalsh(self.data.X_train.T * self.data.X_train))
            max_power = np.log(largest_eigenvalue * settings.gs_eigen_factor)
            min_power = np.log(1e-5)
            gs_lambdas = np.power(np.e, np.arange(min_power, max_power, (max_power - min_power - 0.01) / settings.gs_num_lambdas))

            algo = Elastic_Net_Grid_Search(iter_data.data)
            algo.run(lambdas1=gs_lambdas, lambdas2=gs_lambdas, log_file=f)
        elif method == "HC":
            algo = Elastic_Net_Hillclimb(iter_data.data)
            algo.run(initial_lambdas_set, debug=False, log_file=f)
        elif method == "SP":
            algo = Elastic_Net_Spearmint(iter_data.data, str_identifer)
            algo.run(settings.spearmint_numruns, log_file=f)
        sys.stdout.flush()
        method_res = create_method_result(iter_data.data, algo.fmodel)

        f.write("SUMMARY\n%s" % method_res)
    return method_res

def create_method_result(data, algo):
    test_err = testerror_elastic_net(
        data.X_test,
        data.y_test,
        algo.best_model_params
    )
    print "validation cost", algo.best_cost, "test_err", test_err
    return MethodResult(
        test_err=test_err,
        validation_err=algo.best_cost,
        beta_err=betaerror(algo.current_model_params, data.beta_real),
        runtime=algo.runtime,
        lambdas=algo.current_lambdas
    )

if __name__ == "__main__":
    main(sys.argv[1:])
