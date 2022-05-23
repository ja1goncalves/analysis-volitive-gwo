import math
import numpy as np


# This code was based on in the following references:
# [1] "Defining a Standard for Particle Swarm Optimization" published in 2007 by Bratton and Kennedy


class ObjectiveFunction(object):
    def __init__(self, name, dim, minf, maxf):
        self.function_name = name
        self.dim = dim
        self.minf = minf
        self.maxf = maxf



    def evaluate(self, x):
        pass


class SphereFunction(ObjectiveFunction):
    def __init__(self, dim):
        super(SphereFunction, self).__init__('Sphere', dim, -100.0, 100.0)

    def evaluate(self, x):
        return np.sum(x ** 2)


class RosenbrockFunction(ObjectiveFunction):
    def __init__(self, dim):
        super(RosenbrockFunction, self).__init__('Rosenbrock', dim, -30.0, 30.0)

    def evaluate(self, x):
        sum_ = 0.0
        for i in range(1, len(x) - 1):
            sum_ += 100 * (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2
        return sum_


class RastriginFunction(ObjectiveFunction):
    def __init__(self, dim):
        super(RastriginFunction, self).__init__('Rastrigin', dim, -5.12, 5.12)

    def evaluate(self, x):
        f_x = [xi ** 2 - 10 * math.cos(2 * math.pi * xi) + 10 for xi in x]
        return sum(f_x)


class SchwefelFunction(ObjectiveFunction):
    def __init__(self, dim):
        super(SchwefelFunction, self).__init__('Schwefel', dim, -30.0, 30.0)

    def evaluate(self, x):
        sum_ = 0.0
        for i in range(0, len(x)):
            in_sum = 0.0
            for j in range(i):
                in_sum += x[j] ** 2
            sum_ += in_sum
        return sum_


class GeneralizedShwefelFunction(ObjectiveFunction):
    def __init__(self, dim):
        super(GeneralizedShwefelFunction, self).__init__('GeneralizedShwefel', dim, -30.0, 30.0)

    def evaluate(self, x):
        f_x = [xi * np.sin(np.sqrt(np.absolute(xi))) for xi in x]
        return -sum(f_x)


class GriewankFunction(ObjectiveFunction):
    def __init__(self, dim):
        super(GriewankFunction, self).__init__('Griewank', dim, -600.0, 600.0)

    def evaluate(self, x):
        fi = (1.0 / 4000) * np.sum(x ** 2)
        fii = 1.0
        for i in range(len(x)):
            fii *= np.cos(x[i] / np.sqrt(i + 1))
        return fi + fii + 1


class AckleyFunction(ObjectiveFunction):
    def __init__(self, dim):
        super(AckleyFunction, self).__init__('Ackley', dim, -32.0, 32.0)

    def evaluate(self, x):
        exp_1 = -0.2 * np.sqrt((1.0 / len(x)) * np.sum(x ** 2))
        exp_2 = (1.0 / len(x)) * np.sum(np.cos(2 * math.pi * x))
        return -20 * np.exp(exp_1) - np.exp(exp_2) + 20 + math.e
