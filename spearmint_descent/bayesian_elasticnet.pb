language: PYTHON
name:     "bayesian_elasticnet"

variable {
 name: "lambda1"
 type: FLOAT
 size: 1
 min:  0.1
 max:  100
}

variable {
 name: "lambda2"
 type: FLOAT
 size: 1
 min:  0.1
 max:  100
}
