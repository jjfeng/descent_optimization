import numpy as np
import matplotlib.pyplot as plt

from convexopt_solvers import GenAddModelProblemWrapper
from data_generation import multi_smooth_features

import gen_add_model_gridsearch

np.random.seed(1)
TRAIN_SIZE = 10

def identity_fcn(x):
    return x.reshape(x.size, 1)

def plot_res(fitted_thetas, fcn_list, X):
    num_features = fitted_thetas.shape[1]
    plt.clf()
    for i in range(num_features):
        x_features = X[:,i]
        fcn = fcn_list[i]
        order_x = np.argsort(x_features)
        plt.plot(fitted_thetas[order_x,i], '.', label="fitted %d" % i, color="red")
        plt.plot(fcn(x_features[order_x]), '.', label="real %d" % i, color="green")
    plt.show()


smooth_fcn_list = [identity_fcn, np.sin]

X_train, y_train, X_validate, y_validate, X_test, y_test = multi_smooth_features(
    TRAIN_SIZE,
    smooth_fcn_list,
    desired_snr=10
)
# print X_train
# print y_train
# print X_train.shape
print y_train
print y_validate

best_thetas, best_lambdas = gen_add_model_gridsearch.run(X_train, y_train, X_validate, y_validate, num_lambdas=2)
X_full = np.vstack((X_train, X_validate))
plot_res(best_thetas, smooth_fcn_list, X_full)
