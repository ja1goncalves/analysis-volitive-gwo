import sys

import matplotlib.pyplot as plt

from swarm.GWO.GWO import GWO
from swarm.Utils.Fitness_Function import *
from swarm.Utils.Initialization_Function import Uniform
import pandas as pd
sys.setrecursionlimit(10000)
import os
import math

os.chdir('../..')
dimensions = 100
path_save_results = f'analysis_interaction_graph/simu-journal/{dimensions}d/gwo_results/'

if not os.path.isdir(path_save_results):
    os.makedirs(path_save_results)

stop_criterion = 'iteration'
number_agents = 30
num_evaluations = 1000 * number_agents
max_iterations = num_evaluations - number_agents
max_iterations = math.floor(num_evaluations / (number_agents * 1))
print(max_iterations)
type_initialization_environment = Uniform
rotation = None
translation = None
number_simulations = 10
in_option_improved = False


intypes = ["euclidian"]

funcs = []
if len(sys.argv) <= 1: 
    funcs = [Sphere]
    #funcs = [Griewank]
    #funcs = [Rastrigin, Griewank, Sphere, Schwefel, Rosenbrock, Ackley]
    print('Function was not defined. Starting simulation using:')
    #sys.exit()
elif sys.argv[1].lower() == 'griewank':
    funcs = [Griewank]
elif sys.argv[1].lower() == 'sphere':
    funcs = [Sphere]
elif sys.argv[1].lower() == 'schwefel':
    funcs = [Schwefel]
elif sys.argv[1].lower() == 'rastrigin':
    funcs = [Rastrigin]
elif sys.argv[1].lower() == 'rosenbrock':
    funcs = [Rosenbrock]
elif sys.argv[1].lower() == 'ackley':
    funcs = [Ackley]
else:
    print ("Usage: python RunGWO.py griewank")
    print (f"Invalid function \'{sys.argv[1].lower()}\'. Only Griewank, Sphere, Schwefel, Rastrigin, Rosenbrock or Ackley are valid functions")
    sys.exit(0)

print(funcs)
    
for fitness_function in funcs:
    for in_type in intypes:
        name_file = 'GWO_%s_intype_%s_improved_%s_init_%s_it_%d_dim_%d_swarm_%d_eval_%d' % (
                    fitness_function.__name__, in_type, in_option_improved, type_initialization_environment.__name__, max_iterations,
                    dimensions,number_agents, num_evaluations)

        sim = []
        for simulation in range(number_simulations):
            gwo = GWO(nagents = number_agents, max_iter = max_iterations, dimensions = dimensions, 
                    fitness_function = fitness_function, type_initialization_environment = type_initialization_environment, 
                    path_save_results=path_save_results, maximum_evaluations = num_evaluations, simulation = simulation, 
                    in_type = in_type, in_option_improved = in_option_improved)
                    
            alpha, simulation = gwo.run(debug=False)
            #print(len(simulation))
            print(alpha)
            sim.append(simulation[-1])

        #saving simulations results
        df = pd.DataFrame({"simulations": sim})
        df.to_csv(path_save_results + f"{name_file}_results.csv")
        print (df)
