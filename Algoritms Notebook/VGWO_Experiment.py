import os

from swarmview.view import SwarmView
from ObjectiveFunction import *
from SearchSpaceInitializer import UniformSSInitializer, OneQuarterDimWiseSSInitializer
import time 
import csv
from VGWO import VolitivePack

def create_dir(path):
  directory = os.path.dirname(path)
  try:
    os.stat(directory)
  except:
    os.mkdir(directory)

def main():

  v = 0.01

  #step_volitive_init =  0.1
  #step_volitive_final = 0.0001

  step_volitive_init =  v * 10
  step_volitive_final = v
  

  print (f"starting VGWO ({step_volitive_init}-{step_volitive_final} step volitive)")

  search_space_initializer = UniformSSInitializer()
  result_path = os.path.dirname(os.path.abspath('Algorithms')) + os.sep + "Results" + os.sep + "VGWO-vol" + os.sep
  num_exec = 5
  pack_size = 30
  num_iterations = 500
  dimension = 30
  min_ai = 0.1
  
  unimodal_funcs = [SphereFunction, RotatedHyperEllipsoidFunction, RosenbrockFunction, DixonPriceFunction, QuarticNoiseFunction]
  multimodal_funcs =  [GeneralizedShwefelFunction, RastriginFunction, AckleyFunction, GriewankFunction, LeviFunction]
  regular_functions = unimodal_funcs + multimodal_funcs

  regular_functions = multimodal_funcs + unimodal_funcs
  regular_functions = [SphereFunction]
  cec_functions = []

  create_dir(result_path)
  f_handle_csv = open(result_path + f"/{step_volitive_init}-vol.csv", 'w+')
  writer_csv = csv.writer(f_handle_csv, delimiter=",")
  header = ['step_vol', 'func', 'exec_time'] + [f"run{str(i+1)}" for i in range(num_iterations)]
  writer_csv.writerow(header)

  for benchmark_func in regular_functions:
    simulations = []
    for simulation_id in range(num_exec):
      func = benchmark_func(dimension)
      start = time.time()
      
      swarm_view = SwarmView(simuid = f"{benchmark_func.__name__}_{simulation_id}", xmin = func.minf, xmax = func.maxf, is_3d = False, 
                    function = lambda x, y: x ** 2 - 10 * np.cos(2 * math.pi * x) + y ** 2 - 10 * np.cos(2 * math.pi * y), 
                    enable = False )
      
      #lambda x , y: x**2 + y**2

      bests_iter, bests_eval = run_experiments(num_iterations, pack_size, func,
                                              search_space_initializer,
                                              step_volitive_init,
                                              step_volitive_final, min_ai, swarm_view)
      end = time.time()
      row_csv = [step_volitive_init, func.function_name, (end - start)] + [b for b in bests_iter[:num_iterations]]
      writer_csv.writerow(row_csv)
      best_fit = min(bests_iter)
      simulations.append(best_fit)

      swarm_view.create_gif()

      print(f"{func.function_name}\t fit={best_fit}\t t={end - start}")    
    print(f'\t\tmean={np.mean(simulations)}\t std={np.std(simulations)}\n')  

  f_handle_csv.close()

def run_experiments(n_iter, pack_size, objective_function,
                    search_space_initializer, step_volitive_init,
                    step_individual_final, min_ai, swarm_view):

  opt1 = VolitivePack(objective_function=objective_function,
                      space_initializer=search_space_initializer,
                      n_iter=n_iter, pack_size=pack_size,
                      vol_final=step_individual_final, vol_init=step_volitive_init,
                      min_ai=min_ai)
  opt1.optimize(swarm_view)
  
  return opt1.optimum_fitness_tracking_iter, opt1.optimum_fitness_tracking_eval

main()