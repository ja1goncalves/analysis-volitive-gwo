import numpy as np
import copy

class Wolf(object):
  def __init__(self, dim):
    self.pos = np.zeros(dim)
    self.score = np.inf
    self.aromatic_intensity = 0
    self.delta_pos = np.zeros(dim)
    self.delta_score = 0
    self.best_score = 0
    self.best_pos = np.zeros(dim)
    self.last_pos = np.zeros(dim)

  def update_bests(self, score=None, pos=None):
    score = self.score if score is None else score
    pos = self.pos if pos is None else pos
    if score < self.best_score or self.best_score == np.inf:
      self.best_pos = pos
      self.best_score = score
  
  def is_wolf(self, w: 'Wolf'):
    check_pos = w.pos == self.pos
    if type(check_pos) is np.ndarray:
      check_pos = check_pos.sum() == len(self.pos)

    check_last_pos = w.last_pos == self.last_pos
    if type(check_last_pos) is np.ndarray:
      check_last_pos = check_last_pos.sum() == len(self.pos)
    
    check_pos = check_last_pos and check_pos
    return check_pos and w.score == self.score and w.aromatic_intensity == self.aromatic_intensity
  
  def dist_between(self, w:'Wolf'):
    return math.sqrt(sum((px - qx) ** 2.0 for px, qx in zip(self.pos, w.pos)))


class Alpha(Wolf):
  def __init__(self, dim, score, pos, last_pos, ai):
    self.score = score
    self.pos = pos
    self.last_pos = last_pos
    self.aromatic_intensity = ai


class Beta(Wolf):
  def __init__(self, dim, score, pos, last_pos, ai):
    self.score = score
    self.pos = pos
    self.last_pos = last_pos
    self.aromatic_intensity = ai


class Delta(Wolf):
  def __init__(self, dim, score, pos, last_pos, ai):
    self.score = score
    self.pos = pos
    self.last_pos = last_pos
    self.aromatic_intensity = ai


class Pack(object):
  def __init__(self, objective_function, space_initializer, n_iter, pack_size, analytic_in=False):
    self.objective_function = objective_function # função de avalição de custo
    self.space_initializer = space_initializer # posições iniciais dos peixes

    self.dim = objective_function.dim
    self.minf = objective_function.minf # limite minimo da função
    self.maxf = objective_function.maxf # limite máximo da função
    self.n_iter = n_iter

    self.pack_size = pack_size  # quantidade de peixes

    self.a = 2
    self.r1 = np.random.uniform(size=self.dim)
    self.r2 = np.random.uniform(size=self.dim)
    self.a1 = 2 * self.a * self.r1 - self.a
    self.c1 = 2 * self.r2
    self.best = None
    self.alpha = None
    self.beta = None
    self.delta = None
    
    self.analytic_in = analytic_in
    self.i_net = []
    
    self.optimum_score_tracking_iter = []
    self.optimum_score_tracking_eval = []
  
  def __init_score_tracking(self):
    self.optimum_score_tracking_iter = []
    self.optimum_score_tracking_eval = []

  def __init_wolf(self, pos):
    wolf = Wolf(self.dim)
    wolf.pos = pos
    wolf.score = self.objective_function.evaluate(wolf.pos)
    self.optimum_score_tracking_eval.append(self.alpha.score)
    return wolf

  def __init_pack(self):
    self.best = Wolf(self.dim)
    self.alpha = Wolf(self.dim)
    self.beta = Wolf(self.dim)
    self.delta = Wolf(self.dim)
    self.pack = []
    
    positions = self.space_initializer.sample(self.objective_function, self.pack_size)

    for idx in range(self.pack_size):
      wolf = self.__init_wolf(positions[idx])
      self.pack.append(wolf)
    self.update_hierarchy()
    self.optimum_score_tracking_iter.append(self.alpha.score)
  
  def update_a(self, curr_iter):
    self.a = 2 - curr_iter * (2 / self.n_iter)

  def update_variables(self):
    self.r1 = np.random.uniform(size=self.dim)
    self.r2 = np.random.uniform(size=self.dim)
    self.a1 = 2 * self.a * self.r1 - self.a
    self.c1 = 2 * self.r2

  def update_hierarchy(self):
    for wolf in self.pack:
      if wolf.score < self.best.score:
        self.best = copy.deepcopy(wolf)
      if wolf.score < self.alpha.score:
        self.delta = copy.deepcopy(self.beta)
        self.beta = copy.deepcopy(self.alpha)
        self.alpha = copy.deepcopy(wolf)
      elif wolf.score < self.beta.score:
        self.delta = copy.deepcopy(self.beta)
        self.beta = copy.deepcopy(wolf)
      elif wolf.score < self.delta.score:
        self.delta = copy.deepcopy(wolf)

  def collective_movement(self):
    for wolf in self.pack:
      new_pos = np.zeros((self.dim,), dtype=float)
      self.update_variables()
      d_alpha = abs(self.c1 * self.alpha.pos - wolf.pos)
      x1 = self.alpha.pos - self.a1 * d_alpha
      
      self.update_variables()
      d_beta = abs(self.c1 * self.beta.pos - wolf.pos)
      x2 = self.beta.pos - self.a1 * d_beta

      self.update_variables()
      d_delta = abs(self.c1 * self.delta.pos - wolf.pos)
      x3 = self.delta.pos - self.a1 * d_delta

      new_pos = (x1 + x2 + x3) / 3
      new_pos[new_pos < self.minf] = self.minf
      new_pos[new_pos > self.maxf] = self.maxf
      
      score = self.objective_function.evaluate(new_pos)
      self.optimum_score_tracking_eval.append(self.alpha.score)
      wolf.last_pos = copy.copy(wolf.pos)
      wolf.pos = new_pos
      wolf.score = score
  
  def get_analytic_in(self):
    interaction_in = []
    for w_i in self.pack:
      interaction_w = []
      is_leader = self.alpha.is_wolf(w_i) or self.beta.is_wolf(w_i) or self.delta.is_wolf(w_i)
      for w_j in self.pack:
        euclidean_distance = w_i.dist_between(w_j) if is_leader else 0
        i_net = euclidean_distance
        interaction_w.append(i_net)
      interaction_in.append(interaction_w)
    self.i_net.append(interaction_in)

  def optimize(self):
    self.__init_score_tracking()
    self.__init_pack()

    for i in range(self.n_iter):
      self.update_a(i)
      self.collective_movement()
      self.update_hierarchy()
      self.optimum_score_tracking_iter.append(self.best.score)
      if self.analytic_in:
        self.get_analytic_in()