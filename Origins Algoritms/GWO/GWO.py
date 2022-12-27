import numpy as np
from copy import deepcopy
from swarm.GWO.GWO_IN import GWO_IN
import pandas as pd
from random import shuffle
from time import time 
from scipy.spatial.distance import seuclidean, euclidean

def calc_stdeuclidean(u, v):
    variance = np.var(np.vstack((u, v)), axis=0)
    if np.all((variance == 0)):
        #size = u.shape[0]
        distance = euclidean(u, v)
    else:
        distance = seuclidean(u, v, variance)
    return distance
    
class Wolf():
    def __init__(self, gid, position, in_type):
        self.id = gid
        self.position = position
        self.fitness = np.inf
        self.in_type = in_type

    def move(self, a, alpha, beta, delta):
        dim = len(self.position)

        # alpha
        r1 = np.random.uniform(size=dim)
        r2 = np.random.uniform(size=dim)
        A1 = 2 * a * r1 - a 
        C1 = 2 * r2
        D_alpha = abs(C1 * alpha.position - self.position)
        X1 = alpha.position - A1 * D_alpha

        #beta
        r3 = np.random.uniform(size=dim)
        r4 = np.random.uniform(size=dim)
        A2 = 2 * a * r3 - a 
        C2 = 2 * r4
        D_beta = abs(C2 * beta.position - self.position)
        X2 = beta.position - A2 * D_beta

        #delta
        r5 = np.random.uniform(size=dim)
        r6 = np.random.uniform(size=dim)
        A3 = 2 * a * r5 - a 
        C3 = 2 * r6
        D_delta = abs(C3 * delta.position - self.position)
        X3 = delta.position - A3 * D_delta

        # TODO IN EQUATION
        ## Calculating IN
        if self.in_type == "default":
            influence_alpha = 1
            influence_beta = 1
            influence_delta = 1
        elif self.in_type == "euclidian":
            influence_alpha = np.linalg.norm(X1-self.position)
            influence_beta = np.linalg.norm(X2-self.position)
            influence_delta = np.linalg.norm(X3-self.position)

            #influence_alpha = calc_stdeuclidean(X1, self.position)            
            #influence_beta  = calc_stdeuclidean(X2, self.position)            
            #influence_delta = calc_stdeuclidean(X3, self.position)
        elif self.in_type == "influence":
            influence_alpha = np.sum((X1 - self.position)**2)
            influence_beta  = np.sum((X2 - self.position)**2)
            influence_delta = np.sum((X3 - self.position)**2)
        else:
            raise ValueError("UNKNOWN IN TYPE")

        self.position = (X1 + X2 + X3)/3

        return influence_alpha, influence_beta, influence_delta

    def deepcopy(self):
        return deepcopy(self)
    
class GWO():
    def __init__(self, nagents, max_iter, dimensions, fitness_function, type_initialization_environment, 
                path_save_results, maximum_evaluations, simulation, in_option_improved, in_type):
        np.random.seed(simulation+int(time())) # more generic seed

        self.in_option_improved = in_option_improved
        self.in_type =  in_type
        self.dimensions = dimensions
        self.fitness_function = fitness_function(dimensions=self.dimensions)
        self.type_initialization_environment = type_initialization_environment(dimensions,
                                                                               self.fitness_function.max_environment,
                                                                               self.fitness_function.min_environment)
        self.nagents = nagents
        self.max_iter = max_iter
        self.pop = []
        self.alpha = None
        self.beta = None
        self.delta = None
        self.number_evaluations = maximum_evaluations
        self.simulation = simulation
        self.path_save_results = path_save_results
        self.final_interaction_graph = np.zeros((self.nagents, self.nagents))
        self.name_files = 'GWO_%s_intype_%s_improved_%s_init_%s_it_%d_dim_%d_swarm_%d_eval_%d_sim_%.2d' % (
            self.fitness_function.name, self.in_type, self.in_option_improved, self.type_initialization_environment.name, self.max_iter,
            self.dimensions,
            self.nagents, self.number_evaluations, self.simulation)
        #print(self.name_files)
        self.interaction_network_class = GWO_IN(self.nagents, self.name_files, self.path_save_results)        
        self.interaction_graph = np.zeros((self.nagents, self.nagents))
        self.best_fitness_through_iterations = []
        self.best_fit = np.inf
        self.best_position = None
        
        
    def initialize(self):
        self.pop.clear()
        self.alpha = Wolf(-1, [], self.in_type) #dumb wolves
        self.beta = Wolf(-2, [], self.in_type)
        self.delta = Wolf(-3, [], self.in_type)
        for i in range(self.nagents):
            position = self.type_initialization_environment.initialize()
            wolf = Wolf(i, position, self.in_type)
            wolf.fitness = self.fitness_function.evaluate(wolf.position)
            #self.update_leaders(wolf)
            self.pop.append(wolf)
        self.update_leaders()

    def update_leaders(self):
        cloned_pop = self.pop[:]
        shuffle(cloned_pop)
        ranked = sorted(cloned_pop, key = lambda wolf: wolf.fitness) #Rever escolha quando for igual
        self.alpha = ranked[0].deepcopy()
        self.beta = ranked[1] .deepcopy()
        self.delta = ranked[2].deepcopy()
#
    #def update_leaders(self, wolf):
    #    if wolf.fitness < self.alpha.fitness:
    #        self.delta = self.beta
    #        self.beta = self.alpha
    #        self.alpha = wolf
    #        #self.delta = self.beta.deepcopy()
    #        #self.beta = self.alpha.deepcopy()
    #        #self.alpha = wolf.deepcopy()
    #    elif wolf.fitness < self.beta.fitness:
    #        #self.delta = self.beta.deepcopy()
    #        #self.beta = wolf.deepcopy()
    #        self.delta = self.beta
    #        self.beta = wolf
    #    elif wolf.fitness < self.delta.fitness:
    #        #self.delta = wolf.deepcopy()
    #        self.delta = wolf

    def run(self, debug=False):
        i = 1
        self.initialize()
        self.best_fitness_through_iterations = []
        alphas = []
        betas = []
        deltas = []
        while i <= self.max_iter:
            a = 2 - i * (2 / self.max_iter)
            
            alphas.append(self.alpha.id)
            betas.append(self.beta.id)
            deltas.append(self.delta.id)
            #print(f"{self.alpha.id}")
            #print(f"beta: {self.beta.id}")
            #print(f"delta: {self.delta.id}")
            
            for wolf in self.pop:
                influence_x1, influence_x2, influence_x3 = wolf.move(a, self.alpha, self.beta, self.delta)

                new_fit = self.fitness_function.evaluate(wolf.position)

                if self.in_option_improved:
                    influence = influence_x1 + influence_x2 + influence_x3 
                    if new_fit < wolf.fitness:
                        #print("Melhorou")
                        self.interaction_graph[self.alpha.id][wolf.id] += influence
                        self.interaction_graph[self.beta.id][wolf.id] += influence
                        self.interaction_graph[self.delta.id][wolf.id] += influence
                else:
#                    self.interaction_graph[self.alpha.id][wolf.id] += influence
#                    self.interaction_graph[self.beta.id][wolf.id] += influence
#                    self.interaction_graph[self.delta.id][wolf.id] += influence
                    self.interaction_graph[self.alpha.id][wolf.id] += influence_x1
                    self.interaction_graph[self.beta.id][wolf.id] += influence_x2
                    self.interaction_graph[self.delta.id][wolf.id] += influence_x3

                wolf.fitness = new_fit
                if new_fit < self.best_fit:
                    self.best_fit = new_fit
                    self.best_position = wolf.position
                                
            #for wolf in self.pop:
            self.update_leaders()

            if debug and i % 100 == 0:
                print("Simu: %d  -   Iteration: %d    -    Best Fitness: %f   -   Gbest id %d" % (self.simulation, i, self.alpha.fitness, self.alpha.id))
            
            self.best_fitness_through_iterations.append(self.best_fit)
        
            self.interaction_network_class.update_interaction_network(self.interaction_graph)
            self.interaction_network_class.save_iteration()
            self.final_interaction_graph += self.interaction_graph
            self.interaction_graph = np.zeros((self.nagents, self.nagents))
            
            i+=1
        
        #saving wolves (alpha, beta and delta)
        df = pd.DataFrame({"alpha": alphas, "beta": betas, "delta": deltas})
        df.to_csv(self.path_save_results + self.name_files + '_wolves.csv')

        np.savetxt(
            self.path_save_results + self.name_files + '_interaction_graph.txt',
            self.final_interaction_graph,
            fmt='%.4e')

        np.savetxt(
            self.path_save_results + self.name_files + "_alpha_position.txt",
            self.alpha.position, 
            fmt='%.4e')
        
        np.savetxt(self.path_save_results + self.name_files + '_best_fitness_through_iterations.txt',
                   self.best_fitness_through_iterations, fmt='%.4e')

        return self.alpha.position, self.best_fitness_through_iterations

if __name__ == '__main__':
    pass