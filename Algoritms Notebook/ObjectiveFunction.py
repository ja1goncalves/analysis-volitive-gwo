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

  def prod(self, it):
    p = 1
    for n in it:
      p *= n
    return p

  def u_fun(self, x, a, k, m):
    y = k * ((x - a) ** m) * (x > a) + k * ((-x - a) ** m) * (x < (-a))
    return y



class SphereFunction(ObjectiveFunction): # unimodal function F1
  def __init__(self, dim):
    super(SphereFunction, self).__init__('Sphere', dim, -100.0, 100.0)

  def evaluate(self, x):
    return np.sum(x ** 2)


class RotatedHyperEllipsoidFunction(ObjectiveFunction): # unimodal function
  def __init__(self, dim):
    super(RotatedHyperEllipsoidFunction, self).__init__('RotatedHyperEllipsoid', dim, -65.536, 65.536)

  def evaluate(self, x):
    return sum([np.sum(x[0:i]**2) for i in range(len(x))])
    # sum_i = 0.0
    # for i in range(len(x)):
    #   sum_j = 0.0
    #   for j in range(i):
    #     sum_j += x[j] ** 2
    #   sum_i += sum_j
    # return sum_i


class F2(ObjectiveFunction): # unimodal function
  def __init__(self, dim):
    super(F2, self).__init__('F2', dim, -10.0, 10.0)

  def evaluate(self, x):
    return sum(abs(x)) + self.prod(abs(x))


class F3(ObjectiveFunction): # unimodal function
  def __init__(self, dim):
    super(F3, self).__init__('F3', dim, -100.0, 100.0)

  def evaluate(self, x):
    dim = len(x) + 1
    o = 0
    for i in range(1, dim):
        o = o + (np.sum(x[0:i])) ** 2
    return o


class F4(ObjectiveFunction): # unimodal function
  def __init__(self, dim):
    super(F4, self).__init__('F4', dim, -100.0, 100.0)

  def evaluate(self, x):
    return max(abs(x))


class RosenbrockFunction(ObjectiveFunction): # unimodal function F5
  def __init__(self, dim):
    super(RosenbrockFunction, self).__init__('Rosenbrock', dim, -30.0, 30.0)

  def evaluate(self, x):
    dim = len(x)
    o = np.sum(100 * (x[1:dim] - (x[0 : dim - 1] ** 2)) ** 2 + (x[0 : dim - 1] - 1) ** 2)
    return o


class DixonPriceFunction(ObjectiveFunction): # unimodal function
  def __init__(self, dim):
    super(DixonPriceFunction, self).__init__('Dixon-Price', dim, -10.0, 10.0)

  def evaluate(self, x):
    sum_ = 0.0
    for i in range(1, len(x)):
      sum_ += (i+1) * (((2 * x[i]) ** 2 - x[i-1] ** 2) ** 2)
    return ((x[0]-1)**2) + sum_


class PermFunction(ObjectiveFunction): # unimodal function
  def __init__(self, dim):
    super(PermFunction, self).__init__('Perm', dim, -dim, dim)

  def evaluate(self, x):
    b = 0.5
    sum_i = 0.0
    for i in range(1, len(x)+1):
      sum_j = 0.0
      for j in range(1, len(x)+1):
        sum_j += ((j + b) * ((x[j-1]**i) - 1/(j**i))) ** 2
      sum_i += sum_j
    return sum_i


class QuarticNoiseFunction(ObjectiveFunction): # unimodal function F7
  def __init__(self, dim):
    super(QuarticNoiseFunction, self).__init__('Quartic-Noise', dim, -1.28, 1.28)

  def evaluate(self, x):
    sum_ = 0.0
    for i in range(len(x)):
        sum_ += (i+1) * ((x[i])**4)
    return sum_ + np.random.uniform(0, 1)


class GeneralizedShwefelFunction(ObjectiveFunction): # 2.26 multimodal function F8
  def __init__(self, dim):
    super(GeneralizedShwefelFunction, self).__init__('Generalized-Shwefel', dim, -500.0, 500.0)

  def evaluate(self, x):
    f_x = np.sum(x * (np.sin(np.sqrt(abs(x)))))
    return 418.9829*len(x)-f_x


class RastriginFunction(ObjectiveFunction): # multimodal function F9
  def __init__(self, dim):
    super(RastriginFunction, self).__init__('Rastrigin', dim, -5.12, 5.12)

  def evaluate(self, x):
    f_x = [xi ** 2 - 10 * math.cos(2 * math.pi * xi) for xi in x]
    return sum(f_x) + 10*len(x)


class AckleyFunction(ObjectiveFunction): # multimodal function F10
  def __init__(self, dim):
    super(AckleyFunction, self).__init__('Ackley', dim, -32.768, 32.768)

  def evaluate(self, x):
    exp_1 = -0.2 * np.sqrt((1.0 / len(x)) * np.sum(x ** 2))
    exp_2 = (1.0 / len(x)) * np.sum(np.cos(2 * math.pi * x))
    return -20 * np.exp(exp_1) - np.exp(exp_2) + 20 + math.e


class GriewankFunction(ObjectiveFunction): # multimodal function F11
  def __init__(self, dim):
    super(GriewankFunction, self).__init__('Griewank', dim, -600.0, 600.0)

  def evaluate(self, x):
    dim = len(x)
    w = [i+1 for i in range(len(x))]
    # w = [i + 1 for i in w]
    o = np.sum(x ** 2) / 4000 - self.prod(np.cos(x / np.sqrt(w))) + 1
    return o


class LeviFunction(ObjectiveFunction): # multimodal function F12
  def __init__(self, dim):
    super(LeviFunction, self).__init__('Levi', dim, -50, 50)

  def evaluate(self, x):
    dim = len(x)
    o = (math.pi / dim) * (
        10 * ((np.sin(math.pi * (1 + (x[0] + 1) / 4))) ** 2)
        + np.sum(
            (((x[: dim - 1] + 1) / 4) ** 2)
            * (1 + 10 * ((np.sin(math.pi * (1 + (x[1 :] + 1) / 4)))) ** 2)
        )
        + ((x[dim - 1] + 1) / 4) ** 2
    ) + np.sum(self.u_fun(x, 10, 100, 4))
    return o


class Levi13Function(ObjectiveFunction): # multimodal function F13
  def __init__(self, dim):
    super(Levi13Function, self).__init__('Levi-13', dim, -50, 50)

  def evaluate(self, x):
    if x.ndim==1:
        x = x.reshape(1,-1)

    o = 0.1 * (
        (np.sin(3 * np.pi * x[:,0])) ** 2
        + np.sum(
            (x[:,:-1] - 1) ** 2
            * (1 + (np.sin(3 * np.pi * x[:,1:])) ** 2), axis=1
        )
        + ((x[:,-1] - 1) ** 2) * (1 + (np.sin(2 * np.pi * x[:,-1])) ** 2)
    ) + np.sum(self.u_fun(x, 5, 100, 4))
    return o[0]


class MichalewiczFunction(ObjectiveFunction): # unimodal function F14
  def __init__(self, dim):
    super(MichalewiczFunction, self).__init__('Michalewicz', dim, 0, math.pi)

  def evaluate(self, x):
    m = 10
    sum_ = 0.0
    for i in range(0, len(x)):
      sum_ += math.sin(x[i]) * math.sin(((i* (x[i] ** 2)) / math.pi) ** (2*m))
    return -sum_


class VicentFunction(ObjectiveFunction): # multimodal function
  def __init__(self, dim):
    super(VicentFunction, self).__init__('Vicent', dim, 0.25, 10)

  def evaluate(self, x):
    dim = len(x)
    f_x = [math.sin(10 * math.log(xi)) for xi in x]
    return sum(f_x)/dim


class ModifiedRastriginFunction(ObjectiveFunction): # multimodal function
  def __init__(self, dim):
    super(ModifiedRastriginFunction, self).__init__('Modified Rastrigin', dim, -5.12, 5.12)

  def evaluate(self, x):
    f_x = [10 + (9 * math.cos(2 * math.pi * (1 if i in [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 15] else (2 if i in [4, 8] else (3 if i == 12 else (4 if i == 16 else 1))) ) * xi)) for i, xi in enumerate(x)]
    return -sum(f_x)


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