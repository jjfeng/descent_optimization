from cvxpy import *
import numpy

# Problem data.
def solve_problem():
    m = 200
    n = 100
    A = numpy.random.randn(m, n)
    b = numpy.random.randn(m)

    # Construct the problem.
    x = Variable(n)
    objective = Minimize(sum_squares(A*x - b) + numpy.random.rand() * sum_entries(abs(b)) + numpy.random.rand() * sum_squares(b))
    constraints = [0 <= x, x <= 1]
    prob = Problem(objective, constraints)

    # The optimal objective is returned by prob.solve().
    result = prob.solve()
    # The optimal value for x is stored in x.value.
    # print(x.value)
    # The optimal Lagrange multiplier for a constraint
    # is stored in constraint.dual_value.
    # print(constraints[0].dual_value)
    print "problem",  result

# Problem data.
def solve_problem1():
    m = 250
    n = 100
    A = numpy.random.randn(m, n)
    b = numpy.random.randn(m)

    # Construct the problem.
    x = Variable(n)
    objective = Minimize(sum_squares(A*x - b) + numpy.random.rand() * sum_entries(abs(b)) + numpy.random.rand() * sum_squares(b))
    constraints = [0 <= x, x <= 1]
    prob = Problem(objective, constraints)

    # The optimal objective is returned by prob.solve().
    result = prob.solve()
    # The optimal value for x is stored in x.value.
    # print(x.value)
    # The optimal Lagrange multiplier for a constraint
    # is stored in constraint.dual_value.
    # print(constraints[0].dual_value)
    print "problem1", result

# Problem data.
def solve_problem3():
    m = 250
    n = 100
    A = numpy.random.randn(m, n)
    b = numpy.random.randn(m)

    # Construct the problem.
    x = Variable(n)
    a = numpy.random.rand()
    objective = Minimize(sum_squares(A*x - b) + numpy.random.rand() * a * sum_entries(abs(b)) + numpy.random.rand() * (1-a)* sum_squares(b))
    constraints = [0 <= x, x <= 1]
    prob = Problem(objective, constraints)

    # The optimal objective is returned by prob.solve().
    result = prob.solve()
    # The optimal value for x is stored in x.value.
    # print(x.value)
    # The optimal Lagrange multiplier for a constraint
    # is stored in constraint.dual_value.
    # print(constraints[0].dual_value)
    print "problem3", result


for j in range(1, 10):
    for i in range(1, 100):
        solve_problem()
    for k in range(1, 100):
        solve_problem1()
    for l in range(1, 100):
        solve_problem3()
