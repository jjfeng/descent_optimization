from common import testerror_sparse_add_smooth
from neldermead import Nelder_Mead_Algo
from convexopt_solvers import SparseAdditiveModelProblemWrapperSimple

class Sparse_Add_Model_Nelder_Mead(Nelder_Mead_Algo):
    method_label = "Sparse_Add_Model_Nelder_Mead"
    MAX_COST = 100000

    def _create_problem_wrapper(self):
        self.problem_wrapper = SparseAdditiveModelProblemWrapperSimple(
            self.data.X_full,
            self.data.train_idx,
            self.data.y_train
        )

    def get_validation_cost(self, lambdas):
        if lambdas[0] <= 0 or lambdas[1] <= 0:
            return self.MAX_COST
        thetas = self.problem_wrapper.solve(lambdas)
        validation_cost = testerror_sparse_add_smooth(
            self.data.y_validate,
            self.data.validate_idx,
            thetas
        )
        return validation_cost
