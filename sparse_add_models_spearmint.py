import os
from spearmint_algo import Spearmint_Algo
from common import testerror_sparse_add_smooth
from convexopt_solvers import SparseAdditiveModelProblemWrapper

class Sparse_Add_Model_Spearmint(Spearmint_Algo):
    result_folder = "spearmint_descent/sparse_add_model"

    def _create_problem_wrapper(self):
        self.problem_wrapper = SparseAdditiveModelProblemWrapper(
            self.data.X_full,
            self.data.train_idx,
            self.data.y_train
        )
        self.num_lambdas = self.data.X_full.shape[1] + 1

    def _solve_problem(self, lambdas):
        return self.problem_wrapper.solve(lambdas, high_accur=False, quick_run=True)

    def _check_make_configs(self, folder_suffix):
        self.result_folder = "spearmint_descent/sparse_add_model%s" % folder_suffix
        if not os.path.exists(self.result_folder):
            os.makedirs(self.result_folder)
        config_file_name = "%s/config.json" % self.result_folder
        if not os.path.exists(config_file_name):
            with open(config_file_name, 'w') as config_file:
                print "_create_config_string", self._create_config_string(self.num_lambdas)
                config_file.write(self._create_config_string(self.num_lambdas))

    def get_validation_cost(self, thetas):
        validation_cost = testerror_sparse_add_smooth(
            self.data.y_validate,
            self.data.validate_idx,
            thetas
        )
        return validation_cost
