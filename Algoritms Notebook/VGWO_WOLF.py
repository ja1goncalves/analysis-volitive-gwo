import math 
import numpy as np

class Wolf(object):
  def __init__(self, dim):
    self.pos = np.zeros(dim)
    self.fitness = np.inf
    self.aromatic_intensity = 0
    self.delta_pos = np.zeros(dim)
    self.delta_score = 0
    self.best_score = 0
    self.best_pos = np.zeros(dim)
    self.last_pos = np.zeros(dim)

  def update_bests(self, score=None, pos=None):
    score = self.fitness if score is None else score
    pos = self.pos if pos is None else pos
    if score < self.best_score or self.best_score == np.inf:
      self.best_pos = pos
      self.best_score = score
  
  def is_wolf(self, w: 'Wolf'):
    check_pos = w.pos == self.pos
    if type(check_pos) is np.ndarray:
      check_pos = check_pos.sum() == len(self.pos)

    # check_last_pos = w.last_pos == self.last_pos
    # if type(check_last_pos) is np.ndarray:
    #   check_last_pos = check_last_pos.sum() == len(self.pos)
    
    # check_pos = check_last_pos and check_pos
    # print(check_pos, f"{w.score}-{self.score}", f"{w.aromatic_intensity}-{self.aromatic_intensity}")
    return check_pos and w.fitness == self.fitness and w.aromatic_intensity == self.aromatic_intensity
  
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

