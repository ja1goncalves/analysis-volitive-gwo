from GWO import Pack
from ObjectiveFunction import *
from SearchSpaceInitializer import UniformSSInitializer, OneQuarterDimWiseSSInitializer
import os
import csv
import time

def create_dir(path):
  directory = os.path.dirname(path)
  try:
    os.stat(directory)
  except:
    os.mkdir(directory)

def main():
  for dimension in [15]:#, 30, 50, 100]:
    print (f"starting GWO ({dimension})")
    search_space_initializer = UniformSSInitializer()
    result_path = os.path.dirname(os.path.abspath('Algoritms')) + os.sep + "Results" + os.sep + f"{dimension}d" + os.sep
    num_exec = 5
    pack_size = 30
    num_iterations = 1000

    unimodal_funcs = [SphereFunction, RotatedHyperEllipsoidFunction, RosenbrockFunction, DixonPriceFunction, QuarticNoiseFunction]
    multimodal_funcs =  [GeneralizedShwefelFunction, RastriginFunction, AckleyFunction, GriewankFunction, LeviFunction]
    regular_functions = unimodal_funcs + multimodal_funcs

    cec_functions = []

    create_dir(result_path)
    f_handle_csv = open(result_path + "/GWO_iter.csv", 'w+')
    writer_csv = csv.writer(f_handle_csv, delimiter=",")
    header = ['opt', 'func', 'exec_time'] + [f"run{str(i+1)}" for i in range(num_iterations)]
    writer_csv.writerow(header)

    for benchmark_func in regular_functions:
        simulations = []
        for simulation_id in range(num_exec):
            func = benchmark_func(dimension)
            start = time.time()
            bests_iter, bests_eval = run_experiments(num_iterations, pack_size, func, search_space_initializer)
            end = time.time()
            row_csv = ['GWO', func.function_name, (end - start)] + [b for b in bests_iter[:num_iterations]]
            writer_csv.writerow(row_csv)
            best_fit = min(bests_iter)
            simulations.append(best_fit)

            print(f"{func.function_name}\t t={end - start}\t min={best_fit}")    
        print(f'mean={np.mean(simulations)}\t std={np.std(simulations)}\n')  

    f_handle_csv.close()

def run_experiments(n_iter, pack_size, objective_function, search_space_initializer):
  opt1 = Pack(objective_function=objective_function, space_initializer=search_space_initializer,
              n_iter=n_iter, pack_size=pack_size)
  opt1.optimize()
  return opt1.optimum_score_tracking_iter, opt1.optimum_score_tracking_eval

main()