import math

import numpy as np


class Fitness_Function(object):
    def __init__(self, dimensions, max_environment, min_environment, rotation, translation):
        self.dimensions = dimensions
        self.max_environment = max_environment
        self.min_environment = min_environment
        self.rotation = rotation
        self.translation = translation
        self.optimal_fitness = 0.0

    def evaluate(self, position):
        if len(position) != self.dimensions:
            raise ValueError("number of dimensions does not equal len of position")


class Rastrigin(Fitness_Function):
    max_velocity = 5.12/2
    def __init__(self, dimensions=2, max_environment=5.12, min_environment=-5.12, rotation=None, translation=None):
    # def __init__(self, dimensions=2, max_environment=5.12, min_environment=-5.12, rotation=None, translation=None):
        self.name = "Rastrigin"
        super(Rastrigin, self).__init__(dimensions, max_environment, min_environment, rotation, translation)

    def evaluate(self, position):
        super(Rastrigin, self).evaluate(position)
        return (10 * self.dimensions) + sum([((x_i ** 2) - (10 * np.cos(2 * math.pi * x_i))) for x_i in position])


class Rosenbrock(Fitness_Function):
    max_velocity=10/2
    # def __init__(self, dimensions=2, max_environment=2.048, min_environment=-2.048, rotation=None, translation=None):
    def __init__(self, dimensions=2, max_environment=10.0, min_environment=-5.00, rotation=None, translation=None):
        self.name = "Rosenbrock"
        super(Rosenbrock, self).__init__(dimensions, max_environment, min_environment, rotation, translation)

    def evaluate(self, position):
        super(Rosenbrock, self).evaluate(position)
        return abs(sum([(((1 - position[d]) ** 2) + (100 * ((position[d + 1] - (position[d] ** 2) ** 2)))) for d in
                        range(self.dimensions - 1)]))


class Sphere(Fitness_Function):
    max_velocity=5.12/2
    # def __init__(self, dimensions=2, max_environment=100, min_environment=-100, rotation=None, translation=None):
    def __init__(self, dimensions=2, max_environment=5.12, min_environment=-5.12, rotation=None, translation=None):
        self.name = "Sphere"
        super(Sphere, self).__init__(dimensions, max_environment, min_environment, rotation, translation)

    def evaluate(self, position):
        super(Sphere, self).evaluate(position)
        return sum([(x_i ** 2) for x_i in position])


class Ackley(Fitness_Function):
    max_velocity = 32.0/2
    def __init__(self, dimensions=2, max_environment=32.0, min_environment=-32.0, rotation=None, translation=None):
        self.name = "Ackley"
        super(Ackley, self).__init__(dimensions, max_environment, min_environment, rotation, translation)

    def evaluate(self, position):
        super(Ackley, self).evaluate(position)
        c = 2 * math.pi
        b = 0.2
        sum_all = sum([(x_i ** 2) for x_i in position])
        sum_cos = sum([math.cos(c * x_i) for x_i in position])
        a = 20
        return abs(-a * math.exp(-b * (math.sqrt(((1.0 / self.dimensions) * (sum_all)))))) - (
            math.exp((1.0 / self.dimensions) * (sum_cos))) + a + math.exp(1)


class Schwefel(Fitness_Function):
    max_velocity = 500.00/2
    def __init__(self, dimensions=2, max_environment=500.00, min_environment=-500.0, rotation=None, translation=None):
        self.name = "Schwefel"
        super(Schwefel, self).__init__(dimensions, max_environment, min_environment, rotation, translation)

    def evaluate(self, position):
        super(Schwefel, self).evaluate(position)
        return abs(418.9829 * self.dimensions - sum(
            [(position[d] * math.sin(math.sqrt(math.fabs(position[d])))) for d in range(self.dimensions)]))


class Griewank(Fitness_Function):
    max_velocity = 600.00/2
    def __init__(self, dimensions=2, max_environment=600.00, min_environment=-600.00, rotation=None, translation=None):
        self.name = "Griewank"
        super(Griewank, self).__init__(dimensions, max_environment, min_environment, rotation, translation)

    def evaluate(self, position):
        super(Griewank, self).evaluate(position)
        prod = 1.0
        for d in range(1, self.dimensions + 1):
            prod *= (math.cos(position[d - 1] / math.sqrt(d)) + 1)
        return abs(sum([(position[d - 1] ** 2) / 4000.00 for d in range(1, self.dimensions + 1)]) - prod)
