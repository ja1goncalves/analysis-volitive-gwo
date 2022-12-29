import random
import copy
import numpy as np
import matplotlib.pyplot as plt
from VGWO_WOLF import *

plt.rcParams["figure.figsize"] = (13,8)

class VolitivePack(object):
  def __init__(self, objective_function, space_initializer, n_iter, pack_size,
               vol_init=0.01, vol_final=0.001, min_ai=0.4, analytic_in=False):
    self.objective_function = objective_function # função de avalição de custo
    self.space_initializer = space_initializer # posições iniciais dos peixes

    self.dim = objective_function.dim
    self.minf = objective_function.minf # limite minimo da função
    self.maxf = objective_function.maxf # limite máximo da função
    self.n_iter = n_iter

    self.pack_size = pack_size  # quantidade de peixes
    self.min_ai = min_ai
    self.prev_ai_pack = 0.0
    self.curr_ai_pack = 0.0
    self.barycenter = np.zeros(self.dim)

    self.a = 2
    self.r1 = random.random()
    self.r2 = random.random()
    self.a1 = 2 * self.a * self.r1 - self.a
    self.c1 = 2 * self.r2
    self.step_vol_init = vol_init
    self.step_vol_final = vol_final
    self.curr_step_vol = self.step_vol_init * (self.maxf - self.minf)
    self.curr_mult_vol = 0
    self.best_wolf_ever = None
    self.alpha = None
    self.beta = None
    self.delta = None
    
    self.analytic_in = analytic_in
    self.i_net = np.zeros((n_iter, pack_size, pack_size))

    self.max_delta_fitness = 0
    self.optimum_fitness_tracking_iter = []
    self.optimum_posit_tracking_iter = []
    self.optimum_fitness_tracking_eval = []
  
  def __init_fitness_tracking(self):
    self.optimum_fitness_tracking_iter = []
    self.optimum_posit_tracking_iter = []
    self.optimum_fitness_tracking_eval = []

  def __gen_aromatic_intensity(self, fitness):
    return self.n_iter / fitness

  def __init_wolf(self, pos):
    wolf = Wolf(self.dim)
    wolf.pos = pos
    wolf.fitness = self.objective_function.evaluate(wolf.pos)
    wolf.aromatic_intensity = self.__gen_aromatic_intensity(wolf.fitness)
    self.optimum_fitness_tracking_eval.append(self.best_wolf_ever.fitness)
    return wolf

  def __init_pack(self):
    self.best_wolf_ever = Wolf(self.dim)
    self.alpha = Wolf(self.dim)
    self.beta = Wolf(self.dim)
    self.delta = Wolf(self.dim)
    self.pack = []
    self.curr_ai_pack = 0.0
    self.prev_ai_pack = 0.0
    
    positions = self.space_initializer.sample(self.objective_function, self.pack_size)

    for idx in range(self.pack_size):
      wolf = self.__init_wolf(positions[idx])
      self.pack.append(wolf)
      self.curr_ai_pack += wolf.aromatic_intensity
    self.prev_ai_pack = self.curr_ai_pack
    self.sniffing()
    self.update_hierarchy()

  def update_steps(self, curr_iter):
    self.a = 2 - curr_iter * (2 / self.n_iter)
    self.curr_step_vol = self.step_vol_init - curr_iter * float(self.step_vol_init - self.step_vol_final) / self.n_iter

  def update_variables(self):
    self.r1 = random.random()
    self.r2 = random.random()
    self.a1 = 2 * self.a * self.r1 - self.a
    self.c1 = 2 * self.r2
    
  def total_pack_ai(self):
    self.prev_ai_pack = self.curr_ai_pack
    self.curr_ai_pack = 0.0
    for wolf in self.pack:
      self.curr_ai_pack += wolf.aromatic_intensity

  def calculate_barycenter(self):
    barycenter = np.zeros((self.dim,), dtype=float)
    density = 0.0

    for wolf in self.pack:
      density += wolf.aromatic_intensity
      for dim in range(self.dim):
        barycenter[dim] += (wolf.pos[dim] * wolf.aromatic_intensity)
    for dim in range(self.dim):
      barycenter[dim] = barycenter[dim] / density

    return barycenter

  def update_hierarchy(self, remake=False, alpha=True, beta=True, delta=True):
    """
    This function is a copy of the original in github, but I don't if this is
    correct since here we have a volitive moviment and the alpha/beta/delta
    can be not exist more. 'remake' do the find the alpha/beta/delta again, and
    when False, the wolfes remeber whats the leaders even don't exist more
    """
    self.max_delta_fitness = 0
    if remake:
      self.alpha = Wolf(self.dim) if alpha else self.alpha
      self.beta  = Wolf(self.dim) if beta else self.beta
      self.delta = Wolf(self.dim) if delta else self.delta
    
    for wolf in self.pack:
      # wolf.update_bests()
      if wolf.fitness < self.best_wolf_ever.fitness:
        self.best_wolf_ever = copy.deepcopy(wolf)

      if wolf.fitness < self.alpha.fitness:
        self.delta = copy.deepcopy(self.beta)
        self.beta = copy.deepcopy(self.alpha)
        self.alpha = copy.deepcopy(wolf)
      elif wolf.fitness < self.beta.fitness:
        self.delta = copy.deepcopy(self.beta)
        self.beta = copy.deepcopy(wolf)
      elif wolf.fitness < self.delta.fitness:
        self.delta = copy.deepcopy(wolf)

  def sniffing(self):
    self.prev_ai_pack = self.curr_ai_pack
    self.curr_ai_pack = 0.0
    for wolf in self.pack:
      if self.max_delta_fitness:
        wolf.aromatic_intensity = wolf.aromatic_intensity + (wolf.delta_fitness / self.max_delta_fitness)
      if wolf.aromatic_intensity < self.min_ai:
        wolf.aromatic_intensity = self.min_ai
      
      self.curr_ai_pack += wolf.aromatic_intensity

  def collective_movement(self): # GWO Movement
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
      
      fitness = self.objective_function.evaluate(new_pos)
      # self.optimum_fitness_tracking_eval.append(self.alpha.fitness)
      if fitness < wolf.fitness:
        wolf.delta_fitness = abs(fitness - wolf.fitness)
        wolf.fitness = fitness
        #delta_pos = np.zeros((self.dim,), dtype=float)
        #for idx in range(self.dim):
        #  delta_pos[idx] = new_pos[idx] - wolf.pos[idx]
        delta_pos = new_pos - wolf.pos
        wolf.last_pos = copy.copy(wolf.pos)
        wolf.fitness = fitness
        wolf.delta_pos = delta_pos
        wolf.pos = new_pos
      else:
        wolf.delta_pos = np.zeros((self.dim,), dtype=float)
        wolf.delta_fitness = 0

      if wolf.delta_fitness > self.max_delta_fitness:
        self.max_delta_fitness = wolf.delta_fitness

  def collective_volitive_movement(self):
    # self.total_pack_ai()
    barycenter = self.calculate_barycenter()
    self.barycenter = barycenter
    for wolf in self.pack:
      new_pos = np.zeros((self.dim,), dtype=float)
      if self.curr_ai_pack > self.prev_ai_pack:
        self.curr_mult_vol = 1
        jump = self.curr_mult_vol * self.a
        new_pos = wolf.pos - ((wolf.pos - barycenter) * (self.curr_step_vol * jump) * np.random.uniform(0, 1, size=self.dim))
      else:
        self.curr_mult_vol += 10
        jump = self.curr_mult_vol * self.a
        new_pos = wolf.pos + ((wolf.pos - barycenter) * (self.curr_step_vol * jump) * np.random.uniform(0, 1, size=self.dim))
      new_pos[new_pos < self.minf] = self.minf
      new_pos[new_pos > self.maxf] = self.maxf

      fitness = self.objective_function.evaluate(new_pos)
      self.optimum_fitness_tracking_eval.append(self.alpha.fitness)
      if self.analytic_in:
        wolf.last_pos = copy.copy(wolf.pos)
      wolf.fitness = fitness
      wolf.pos = new_pos

  def get_analytic_in(self, n_iter, euclidean=True, volitive=True):
    interaction_in = []
    for w_i in self.pack:
      interaction_w = []
      is_leader = self.alpha.is_wolf(w_i) or self.beta.is_wolf(w_i) or self.delta.is_wolf(w_i)
      for w_j in self.pack:
        euclidean_distance = w_i.dist_between(w_j) if is_leader and euclidean else 0
        interaction_vol = 0
        if volitive:
          for d in range(self.dim):
            # contribuition or interaction can be negative?????
            contribuition = (w_i.aromatic_intensity * w_i.last_pos[d])/self.barycenter[d]
            interaction_vol += (w_j.delta_pos[d] * contribuition)
        i_net = euclidean_distance + interaction_vol
        interaction_w.append(i_net)
      interaction_in.append(interaction_w)
    self.i_net[n_iter] += interaction_in
    # print(n_iter, self.i_net[n_iter].sum())

  def optimize(self):
    self.__init_fitness_tracking()
    self.__init_pack()

    for i in range(self.n_iter):
      self.update_steps(i)
      self.collective_movement()
      self.sniffing()
      self.update_hierarchy(True) # It's can be use lonely without the second (remake True is a little bit bad to Sphere and Schwefel, but good to Multimodal)
      #if self.analytic_in:
      #  self.get_analytic_in(i, euclidean=True, volitive=False)
      self.collective_volitive_movement()
      self.sniffing()
      self.update_hierarchy() # remake True is good only to Multimodal
      #if self.analytic_in:
      #  self.get_analytic_in(i, euclidean=False, volitive=True)
      self.optimum_fitness_tracking_iter.append(self.alpha.fitness)
      self.optimum_posit_tracking_iter.append(self.alpha.pos.tolist())