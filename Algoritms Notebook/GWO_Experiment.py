from GWO import Pack
from ObjectiveFunction import *
from SearchSpaceInitializer import UniformSSInitializer, OneQuarterDimWiseSSInitializer
import os
import csv
import time
import pandas as pd 

def create_dir(path):
  directory = os.path.dirname(path)
  try:
    os.stat(directory)
  except:
    os.mkdir(directory)

def main():
  dimension = 30
  print (f"starting GWO ({dimension})")
  search_space_initializer = UniformSSInitializer()
  result_path = os.path.dirname(os.path.abspath('Algoritms')) + os.sep + "Results" + os.sep + f"{dimension}d" + os.sep
  num_exec = 20
  pack_size = 30
  num_iterations = 1000

  unimodal_funcs = [SphereFunction, RotatedHyperEllipsoidFunction, RosenbrockFunction, DixonPriceFunction, QuarticNoiseFunction]
  multimodal_funcs =  [GeneralizedShwefelFunction, RastriginFunction, AckleyFunction, GriewankFunction, LeviFunction]
  regular_functions = multimodal_funcs + unimodal_funcs 
  # regular_functions = [SphereFunction]
  max_evaluations = (pack_size * num_iterations) + pack_size

  cec_functions = []

  create_dir(result_path)
  f_handle_csv = open(result_path + "/GWO_iter.csv", 'w+')
  writer_csv = csv.writer(f_handle_csv, delimiter=",")
  header = ['opt', 'func', 'exec_time'] + [f"run{str(i+1)}" for i in range(num_iterations)]
  writer_csv.writerow(header)
  name_file = f'GWO_dim_{dimension}_agents_{pack_size}_iter_{num_iterations}_eval_{max_evaluations}'

  for benchmark_func in regular_functions:
    simulations = []
    bf_dict = {}
    for simulation_id in range(num_exec):
        func = benchmark_func(dimension)
        start = time.time()
        fit_convergence, bests_eval = run_experiments(num_iterations, max_evaluations, pack_size, func, search_space_initializer)
        end = time.time()
        row_csv = ['GWO', func.function_name, (end - start)] + [b for b in fit_convergence[:num_iterations]]
        writer_csv.writerow(row_csv)
        best_fit = min(fit_convergence)
        simulations.append(best_fit)
        bf_dict[f'convergence_simulation_{simulation_id}'] = fit_convergence
        print(f"{func.function_name}\t t={end - start}\t min={best_fit}")    
    print(f'\t\tmean={np.mean(simulations)}\t std={np.std(simulations)}\n')  
    df_bf = pd.DataFrame(bf_dict)
    df_bf.to_csv(f"results/{name_file}_{func.function_name}_convergence.csv")
  f_handle_csv.close()

def run_experiments(n_iter, max_evaluations,  pack_size, objective_function, search_space_initializer):
  opt1 = Pack(objective_function=objective_function, space_initializer=search_space_initializer,
              n_iter=n_iter, max_evaluations= max_evaluations, pack_size=pack_size)
  opt1.optimize()
  return opt1.optimum_fitness_tracking_iter, opt1.optimum_fitness_tracking_eval

main()