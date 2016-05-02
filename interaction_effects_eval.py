import matplotlib.pyplot as plt

from common import *
from method_results import MethodResults
from data_generation import effects_and_interactions
import hillclimb_interaction_effects
import hillclimb_mu
import gridsearch_interaction_effects

GENERATE_PLOT = False
NUM_RUNS = 15

# TRAIN_SIZE = 70
# NUM_EFFECTS = 40
# NUM_NONZERO_EFFECTS = 3
# NUM_NONZERO_INTERACTIONS = 300

TRAIN_SIZE = 40
NUM_EFFECTS = 20
NUM_NONZERO_EFFECTS = 3
NUM_NONZERO_INTERACTIONS = 40

hc_results = MethodResults("Hillclimb")
mu_results = MethodResults("MU")
gs_results = MethodResults("Gridsearch")

for i in range(0, NUM_RUNS):
    beta_real, theta_real, X_train, W_train, y_train, X_validate, W_validate, y_validate, X_test, W_test, y_test = \
        effects_and_interactions(TRAIN_SIZE, NUM_EFFECTS, NUM_NONZERO_EFFECTS, NUM_NONZERO_INTERACTIONS)

    def _get_test_beta_theta_err(beta_guess, theta_guess):
        test_err = testerror_interactions(X_test, W_test, y_test, beta_guess, theta_guess) / y_test.size * 2
        beta_err = betaerror(beta_real, beta_guess)
        theta_err = betaerror(theta_guess, theta_real)
        return (test_err, beta_err, theta_err)

    hc_beta_guess, hc_theta_guess, hc_costpath = hillclimb_interaction_effects.run(X_train, W_train, y_train, X_validate, W_validate, y_validate)
    hc_results.append_test_beta_theta_err(_get_test_beta_theta_err(hc_beta_guess, hc_theta_guess))

    mu_beta_guess, mu_theta_guess, mu_costpath = hillclimb_mu.run(X_train, W_train, y_train, X_validate, W_validate, y_validate)
    mu_results.append_test_beta_theta_err(_get_test_beta_theta_err(mu_beta_guess, mu_theta_guess))

    gs_beta_guess, gs_theta_guess, gs_lowest_cost = gridsearch_interaction_effects.run(X_train, W_train, y_train, X_validate, W_validate, y_validate)
    gs_results.append_test_beta_theta_err(_get_test_beta_theta_err(gs_beta_guess, gs_theta_guess))

    print "NUM RUN", i
    print "NUM EFFECTS", NUM_EFFECTS
    print "NUM NONZERO EFFECTS", NUM_NONZERO_EFFECTS
    print "NUM NONZERO INTERACTIONS", NUM_NONZERO_INTERACTIONS
    print "TRAIN SIZE", TRAIN_SIZE

    hc_results.print_results()
    mu_results.print_results()
    gs_results.print_results()

    if GENERATE_PLOT and i == 0:
        plt.clf()
        plt.plot(hc_costpath, label="Hillclimb", color="blue")
        plt.plot(mu_costpath, label="MU", color="green")
        plt.axhline(gs_lowest_cost, label="Gridsearch", color="brown")
        plt.legend(fontsize="x-small")
        plt.title("Train=%d effects=%d nonzero=%d,%d effect2interaction=%d" % (TRAIN_SIZE, NUM_EFFECTS, NUM_NONZERO_EFFECTS, NUM_NONZERO_INTERACTIONS, EFFECTS_TO_INTERACTION_RATIO))
        plt.xlabel("Number of iterations")
        plt.ylabel("Validation test error")
        # plt.savefig("figures/effects_interactions_test.png")
        plt.savefig("figures/interaction_effects_signal2_%d_%d_%d_%d_%d.png" % (TRAIN_SIZE, NUM_EFFECTS, NUM_NONZERO_EFFECTS, NUM_NONZERO_INTERACTIONS, EFFECTS_TO_INTERACTION_RATIO))
