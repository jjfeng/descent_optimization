import numpy as np
import matplotlib.pyplot as plt

from convexopt_solvers import GenAddModelProblemWrapper
from data_generation import multi_smooth_features

from gen_add_model_hillclimb import GenAddModelHillclimb
import gen_add_model_gridsearch as gs

from common import *

np.set_printoptions(edgeitems=3,infstr='inf', linewidth=200, nanstr='nan', precision=8, suppress=False, threshold=1000, formatter=None)

np.random.seed(1)
TRAIN_SIZE = 10

def identity_fcn(x):
    return x.reshape(x.size, 1)

def crazy_down_sin(x):
    return np.sin(np.power(x/2, 2)) - x

def plot_res(fitted_thetas, fcn_list, X, outfile="figures/threegam/out.png"):
    num_features = fitted_thetas.shape[1]
    plt.clf()
    for i in range(num_features):
        x_features = X[:,i]
        fcn = fcn_list[i]
        order_x = np.argsort(x_features)
        plt.plot(
            x_features[order_x],
            fcn(x_features[order_x]), # true values
            'o',
            x_features[order_x],
            fitted_thetas[order_x,i], # fitted values
            '.',
            label="feat %d" % i
        )
    plt.savefig(outfile)


#### Note: there seems to be an identifiability issue

smooth_fcn_list = [crazy_down_sin, identity_fcn]
# smooth_fcn_list = [identity_fcn]

X_train, y_train, X_validate, y_validate, X_test, y_test = multi_smooth_features(
    TRAIN_SIZE,
    smooth_fcn_list,
    desired_snr=1,
    feat_range=[0,10]
)
X_full, train_idx, validate_idx, test_idx = GenAddModelHillclimb.stack((X_train, X_validate, X_test))

print "start"
hc = GenAddModelHillclimb(X_train, y_train, X_validate, y_validate, X_test)
print "inited!"
init_lambdas = np.array([0.25,0.25])
hc_thetas, cost_path, curr_regularization = hc.run(init_lambdas, debug=True)
hc_test_error = testerror_multi_smooth(y_test, test_idx, hc_thetas)
plot_res(hc_thetas[test_idx], smooth_fcn_list, X_test, outfile="figures/threegam/out_hc.png")

gs_thetas, best_lambdas = gs.run(
    y_train,
    y_validate,
    X_full,
    train_idx,
    validate_idx,
    num_lambdas=3,
    max_lambda=10
)

gs_test_error = testerror_multi_smooth(y_test, test_idx, gs_thetas)
plot_res(gs_thetas[test_idx], smooth_fcn_list, X_test, outfile="figures/threegam/out_gs.png")

print "gs_test_error", gs_test_error
print "hc_test_error", hc_test_error
