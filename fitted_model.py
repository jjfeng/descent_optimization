class Fitted_Model:
    # @param: init_lambdas, a np matrix
    # @param: init_model_params, a tuple of model params
    def __init__(self, lambdas, model_params, cost):
        self.num_lambdas = lambdas.size
        self.init_lambdas = lambdas
        self.lambda_history = [lambdas]
        self.model_param_history = [model_params]
        self.cost_history = [cost]
        self._update_current_vals()

    def update(self, new_lambdas, new_model_params, cost):
        self.lambda_history.append(new_lambdas)
        self.model_param_history.append(new_model_params)
        self.cost_history.append(cost)
        self._update_current_vals()

    def _update_current_vals(self):
        self.current_model_params = self.model_param_history[-1]
        self.current_cost = self.cost_history[-1]
        self.current_lambdas = self.lambda_history[-1]

    def get_cost_diff(self):
        return self.cost_history[-2] - self.cost_history[-1]

    def __str__(self):
        return "cost %f, current_lambdas %s" % (self.current_cost, self.current_lambdas)
