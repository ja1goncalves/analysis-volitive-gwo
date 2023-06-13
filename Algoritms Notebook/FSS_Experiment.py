import time
import os
import numpy as np
from FSS import FSS
from ObjectiveFunction import *
from SearchSpaceInitializer import UniformSSInitializer, OneQuarterDimWiseSSInitializer
import csv
import pandas as pd

def create_dir(path):
   directory = os.path.dirname(path)
   try:
      os.stat(directory)
   except:
      os.mkdir(directory)

def run_experiments(n_iter, max_evaluations, school_size, num_runs, objective_function,
                    search_space_initializer, step_individual_init,
                    step_individual_final, step_volitive_init,
                    step_volitive_final, min_w, w_scale):
    alg_name = "FSS"
    console_out = "Algorithm: {} Function: {} Best Cost: {}"

    opt1 = FSS(objective_function=objective_function, space_initializer=search_space_initializer,
                n_iter=n_iter, max_evaluations = max_evaluations, school_size=school_size, step_ind_init=step_individual_init,
                step_ind_final=step_individual_final, step_vol_init=step_volitive_init,
                step_vol_final=step_volitive_final, min_weight=min_w, weight_scale=w_scale)

    opt1.optimize()
    print(console_out.format(alg_name, objective_function.function_name, opt1.best_fish.cost))

    return opt1.optimum_cost_tracking_iter, opt1.optimum_cost_tracking_eval

def main():
    dimension = 15
    print (f"starting FSS ({dimension})")
    search_space_initializer = UniformSSInitializer()
    file_path = os.path.dirname(os.path.abspath('Algoritms')) + os.sep + "Executions" + os.sep + f"{dimension}d" + os.sep
    result_path = os.path.dirname(os.path.abspath('Algoritms')) + os.sep + "Results" + os.sep + f"{dimension}d" + os.sep
    num_exec = 30
    school_size = 30
    num_iterations = 1000
    step_individual_init = 0.1
    step_individual_final = 0.0001
    step_volitive_init = 0.1
    step_volitive_final = 0.01
    min_w = 1
    w_scale = num_iterations / 2.0
    max_evaluations = (school_size * num_iterations) + school_size

    unimodal_funcs = [SphereFunction, RotatedHyperEllipsoidFunction, RosenbrockFunction, DixonPriceFunction, QuarticNoiseFunction]
    multimodal_funcs = [GeneralizedShwefelFunction, RastriginFunction, AckleyFunction, GriewankFunction, LeviFunction]
    regular_functions = unimodal_funcs + multimodal_funcs

    # regular_functions = [RosenbrockFunction, RastriginFunction]

    # Notice that for CEC Functions only the following dimensions are available:
    # 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
    cec_functions = []

    name_file = f'FSS_dim_{dimension}_agents_{school_size}_iter_{num_iterations}_eval_{max_evaluations}'

    create_dir(result_path)
    f_handle_csv = open(result_path + "/FSS_exec.csv", 'w+')
    writer_csv = csv.writer(f_handle_csv, delimiter=",")
    header = ['opt', 'func', 'exec_time'] + [f"run{str(i+1)}" for i in range(num_exec)]
    writer_csv.writerow(header)

    for benchmark_func in regular_functions:
        simulations = []
        bf_iter = {}
        bf_eval = {}
        for simulation_id in range(num_exec):
            func = benchmark_func(dimension)
            start = time.time()
            fit_convergence, bests_eval = run_experiments(num_iterations, max_evaluations, school_size, num_exec, func,
                                    search_space_initializer, step_individual_init,
                                    step_individual_final, step_volitive_init,
                                    step_volitive_final, min_w, w_scale)

            end = time.time()
            row_csv = ['FSS', func.function_name, (end - start)] + [r for r in fit_convergence[:num_iterations]]
            writer_csv.writerow(row_csv)
            best_fit = min(fit_convergence)
            simulations.append(best_fit)
            bf_iter[f'convergence_simulation_{simulation_id}'] = fit_convergence
            bf_eval[f'eval_simulation_{simulation_id}'] = func.best_evaluation_history
            print(f"{func.function_name}\t t={end - start}\t min={best_fit}")
        print(f'\t\tmean={np.mean(simulations)}\t std={np.std(simulations)}\n')
        df_iter = pd.DataFrame(bf_iter)
        df_iter.to_csv(f"results/{name_file}_{func.function_name}_convergence.csv")
        df_be = pd.DataFrame(bf_eval)
        df_be.to_csv(f"results/{name_file}_{func.function_name}_evaluations.csv")   
    f_handle_csv.close()

if __name__ == '__main__':
    main()  