import copy 
from ObjectiveFunction import *

class Fish(object):
  """
  Classe que define um peixe que contém sua posição (em diversas dimensões),
  seu peso e custo de desempenho. Para auxilio também memoriza o quanto andou
  e quanto de custo mudou.
  """
  def __init__(self, dim):
    nan = float('nan')
    self.pos = [nan for _ in range(dim)]
    self.last_pos = [nan for _ in range(dim)]
    self.delta_pos = np.nan
    self.delta_cost = np.nan
    self.weight = np.nan
    self.cost = np.nan
    # self.has_improved = False

class FSS(object):
  """
  Classe que representa a escola de peixes (classe acima) que irá se locomover
  dentro de uma região e serão avaliados de acordo com uma função objetivo
  """
  def __init__(self, objective_function, space_initializer, n_iter, max_evaluations, school_size,
               step_ind_init, step_ind_final, step_vol_init, step_vol_final,
               min_weight, weight_scale, analytic_in=False):
    self.objective_function = objective_function # função de avalição de custo
    self.space_initializer = space_initializer # posições iniciais dos peixes

    self.dim = objective_function.dim
    self.minf = objective_function.minf # limite minimo da função
    self.maxf = objective_function.maxf # limite máximo da função
    self.n_iter = n_iter
    self.barycenter = np.zeros(self.dim)
    self.max_evaluations = max_evaluations
    
    self.school_size = school_size  # quantidade de peixes
    self.step_ind_init = step_ind_init
    self.step_ind_final = step_ind_final
    self.step_vol_init = step_vol_init
    self.step_vol_final = step_vol_final

    self.curr_step_ind = self.step_ind_init * (self.maxf - self.minf)
    self.curr_step_vol = self.step_vol_init * (self.maxf - self.minf)
    self.min_w = min_weight
    self.w_scale = weight_scale
    self.prev_weight_school = 0.0
    self.curr_weight_school = 0.0
    self.best_fish = None
    
    self.analytic_in = analytic_in
    self.i_net = []
    
    self.optimum_cost_tracking_iter = []
    self.optimum_cost_tracking_eval = []

  def __gen_weight(self):
    return self.w_scale / 2.0

  def __init_fss(self):
    self.optimum_cost_tracking_iter = []
    self.optimum_cost_tracking_eval = []

  def __init_fish(self, pos):
    fish = Fish(self.dim)
    fish.pos = pos
    fish.weight = self.__gen_weight()
    fish.cost = self.objective_function.evaluate(fish.pos)
    self.optimum_cost_tracking_eval.append(self.best_fish.cost)
    return fish

  def __init_school(self):
    self.best_fish = Fish(self.dim)
    self.best_fish.cost = np.inf
    self.curr_weight_school = 0.0
    self.prev_weight_school = 0.0
    self.school = []

    positions = self.space_initializer.sample(self.objective_function, self.school_size)

    for idx in range(self.school_size):
        fish = self.__init_fish(positions[idx])
        self.school.append(fish)
        self.curr_weight_school += fish.weight
    self.prev_weight_school = self.curr_weight_school
    self.update_best_fish()
    self.optimum_cost_tracking_iter.append(self.best_fish.cost)
  
  def update_best_fish(self):
    for fish in self.school:
      if self.best_fish.cost > fish.cost:
        self.best_fish = copy.copy(fish)

  def update_steps(self, curr_iter):
    self.curr_step_ind = self.step_ind_init - curr_iter * float(self.step_ind_init - self.step_ind_final) / self.n_iter
    self.curr_step_vol = self.step_vol_init - curr_iter * float(self.step_vol_init - self.step_vol_final) / self.n_iter

  def max_delta_cost(self):
    max_ = 0
    for fish in self.school:
      if max_ < fish.delta_cost:
        max_ = fish.delta_cost
    return max_
    
  def total_school_weight(self):
    self.prev_weight_school = self.curr_weight_school
    self.curr_weight_school = 0.0
    for fish in self.school:
      self.curr_weight_school += fish.weight
      
  def calculate_barycenter(self):
    barycenter = np.zeros((self.dim,), dtype=float)
    density = 0.0

    for fish in self.school:
      density += fish.weight
      for dim in range(self.dim):
        barycenter[dim] += (fish.pos[dim] * fish.weight)
    for dim in range(self.dim):
      barycenter[dim] = barycenter[dim] / density

    return barycenter

  def feeding(self):
    for fish in self.school:
      if self.max_delta_cost():
        fish.weight = fish.weight + (fish.delta_cost / self.max_delta_cost())
      if fish.weight > self.w_scale:
        fish.weight = self.w_scale
      elif fish.weight < self.min_w:
        fish.weight = self.min_w
  
  def individual_movement(self, n_iter):
    for fish in self.school:
      new_pos = np.zeros((self.dim,), dtype=float)
      for dim in range(self.dim):
        new_pos[dim] = fish.pos[dim] + (self.curr_step_ind * np.random.uniform(-1, 1))
        if new_pos[dim] < self.minf:
            new_pos[dim] = self.minf
        elif new_pos[dim] > self.maxf:
            new_pos[dim] = self.maxf
      
      if n_iter == 0:
        cost = self.objective_function.evaluate(new_pos)
        self.optimum_cost_tracking_eval.append(self.best_fish.cost)
        if cost < fish.cost:
          fish.delta_cost = abs(cost - fish.cost)
          fish.cost = cost
          delta_pos = np.zeros((self.dim,), dtype=float)
          for idx in range(self.dim):
            delta_pos[idx] = new_pos[idx] - fish.pos[idx]
          fish.last_pos = copy.copy(fish.pos)
          fish.delta_pos = delta_pos
          fish.pos = new_pos
        else:
          fish.delta_pos = np.zeros((self.dim,), dtype=float)
          fish.delta_cost = 0
      else:
        fish.pos = new_pos
          
  def collective_instinctive_movement(self):
    cost_eval_enhanced = np.zeros((self.dim,), dtype=float)
    density = 0.0
    for fish in self.school:
      density += fish.delta_cost
      for dim in range(self.dim):
        cost_eval_enhanced[dim] += (fish.delta_pos[dim] * fish.delta_cost)
    for dim in range(self.dim):
      if density != 0:
        cost_eval_enhanced[dim] = cost_eval_enhanced[dim] / density
    for fish in self.school:
      new_pos = np.zeros((self.dim,), dtype=float)
      for dim in range(self.dim):
        new_pos[dim] = fish.pos[dim] + cost_eval_enhanced[dim]
        if new_pos[dim] < self.minf:
          new_pos[dim] = self.minf
        elif new_pos[dim] > self.maxf:
          new_pos[dim] = self.maxf

        fish.pos = new_pos
  
  def collective_volitive_movement(self, n_iter):
    self.total_school_weight()
    barycenter = self.calculate_barycenter()
    self.barycenter = barycenter
    for fish in self.school:
      new_pos = np.zeros((self.dim,), dtype=float)
      for dim in range(self.dim):
        if self.curr_weight_school > self.prev_weight_school:
          new_pos[dim] = fish.pos[dim] - ((fish.pos[dim] - barycenter[dim]) * self.curr_step_vol * np.random.uniform(0, 1))
        else:
          new_pos[dim] = fish.pos[dim] + ((fish.pos[dim] - barycenter[dim]) * self.curr_step_vol * np.random.uniform(0, 1))
        if new_pos[dim] < self.minf:
          new_pos[dim] = self.minf
        elif new_pos[dim] > self.maxf:
          new_pos[dim] = self.maxf
      
      if n_iter > 0:
        cost = self.objective_function.evaluate(new_pos)
        self.optimum_cost_tracking_eval.append(self.best_fish.cost)
        if cost < fish.cost:
          fish.delta_cost = abs(cost - fish.cost)
          fish.cost = cost
          delta_pos = np.zeros((self.dim,), dtype=float)
          for idx in range(self.dim):
            delta_pos[idx] = new_pos[idx] - fish.pos[idx]
          fish.last_pos = copy.copy(fish.pos)
          fish.delta_pos = delta_pos
          fish.pos = new_pos
        else:
          fish.delta_pos = np.zeros((self.dim,), dtype=float)
          fish.delta_cost = 0
      else:
        fish.pos = new_pos


      # cost = self.objective_function.evaluate(new_pos)
      # self.optimum_cost_tracking_eval.append(self.best_fish.cost)
      # fish.cost = cost
      # fish.pos = new_pos
      
  def optimize(self):
    self.__init_fss()
    self.__init_school()
    i = 0
    while self.objective_function.evaluations < self.max_evaluations:
      self.individual_movement(i)
      self.update_best_fish()
      self.collective_instinctive_movement()
      self.collective_volitive_movement(i)
      self.feeding()
      self.update_steps(i)
      self.update_best_fish()
      self.optimum_cost_tracking_iter.append(self.best_fish.cost)
      i+=1
      #print "Iteration: ", i, " Cost: ", self.best_fish.cost
