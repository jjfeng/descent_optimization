class Fitted_Model:
    def __init__(self, num_lambdas):
        self.num_lambdas = num_lambdas
        self.lambda_history = []
        self.model_param_history = []
        self.cost_history = []

        self.current_cost = None
        self.current_lambdas = None

    def update(self, new_lambdas, new_model_params, cost):
        self.lambda_history.append(new_lambdas)
        self.model_param_history.append(new_model_params)
        self.cost_history.append(cost)

        self.current_model_params = self.model_param_history[-1]
        self.current_cost = self.cost_history[-1]
        self.current_lambdas = self.lambda_history[-1]

    def set_runtime(self, runtime):
        self.runtime = runtime

    def get_cost_diff(self):
        return self.cost_history[-2] - self.cost_history[-1]

    def __str__(self):
        return "cost %f, current_lambdas %s" % (self.current_cost, self.current_lambdas)
