import numpy as np


class Initialization_Function(object):
    def __init__(self, name, dimensions, max_environment, min_environment):
        self.name = name
        self.dimensions = dimensions
        self.max_environment = max_environment
        self.min_environment = min_environment

    def initialize(self):
        pass


class Uniform(Initialization_Function):
    def __init__(self, dimensions=2, max_environment=1.0, min_environment=0.0):
        super(Uniform, self).__init__("Uniform", dimensions, max_environment, min_environment)

    def initialize(self):
        return np.random.uniform(self.min_environment, self.max_environment, self.dimensions)


class OneQuarter(Initialization_Function):
    def __init__(self, dimensions=2, max_environment=1.0, min_environment=0.0):
        super(OneQuarter, self).__init__("OneQuarter", dimensions, max_environment, min_environment)

    def initialize(self):
        one_quarter = self.max_environment - ((1.0 / 4.0) * (self.max_environment - self.min_environment))
        return np.random.uniform(one_quarter, self.max_environment, self.dimensions)
