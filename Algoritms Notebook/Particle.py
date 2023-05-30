import numpy as np
import random 

class Particle(object):
    """
    Classe que define um particula que contém sua posição (em diversas dimensões)
    com o custo de desempenho. Para auxilio também memoriza o quanto andou
    e quanto de custo mudou.
    """
    def __init__(self, dim):
        self.pos = np.array([np.nan for _ in range(dim)])
        self.fitness = np.inf
        self.vel = np.array([random.uniform(-1,1) for _ in range(dim)])
        self.best_fitness = np.inf
        self.best_pos = np.array([np.nan for _ in range(dim)])
  
    def update_bests(self, fitness=None, pos=None):
        fitness = self.fitness if fitness is None else fitness
        pos = self.pos if pos is None else pos
        if fitness < self.best_fitness or self.best_fitness == np.inf:
            self.best_pos = pos
            self.best_fitness = fitness
