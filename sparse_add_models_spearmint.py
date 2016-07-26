import os
from spearmint_algo import Spearmint_Algo
from common import testerror_sparse_add_smooth
from convexopt_solvers import SparseAdditiveModelProblemWrapper

class Sparse_Add_Model_Spearmint(Spearmint_Algo):
    method_label = "Sparse_Add_Model_Spearmint"
    result_folder_prefix = "spearmint_descent/sparse_add_model"

    def _create_problem_wrapper(self):
        self.problem_wrapper = SparseAdditiveModelProblemWrapper(
            self.data.X_full,
            self.data.train_idx,
            self.data.y_train
        )
        self.num_lambdas = self.data.X_full.shape[1] + 1

    def _solve_problem(self, lambdas):
        return self.problem_wrapper.solve(lambdas, high_accur=False, quick_run=True)

    def get_validation_cost(self, thetas):
        validation_cost = testerror_sparse_add_smooth(
            self.data.y_validate,
            self.data.validate_idx,
            thetas
        )
        return validation_cost
