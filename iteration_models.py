class Simulation_Settings:
    train_size = 100
    validate_size = 50
    test_size = 50
    snr = 2
    gs_num_lambdas = 10
    spearmint_numruns = 50
    nm_iters = 80
    feat_range = [-5,5]
    method = "HC"

class Iteration_Data:
    def __init__(self, i, data, settings):
        self.data = data
        self.settings = settings
        self.i = i