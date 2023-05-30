import os 
import csv 
import time 
from PSO import PSO
from SearchSpaceInitializer import UniformSSInitializer
from ObjectiveFunction import *
import pandas as pd 

def main(topology):
    dimension = 30
    print (f"starting PSO ({dimension}) - {topology}")
    search_space_initializer = UniformSSInitializer()
    result_path = os.path.dirname(os.path.abspath('Algoritms')) + os.sep + "Results" + os.sep + f"{dimension}d" + os.sep
    num_exec = 20
    part_size = 30
    num_iterations = 1000
    inertia = 0
    cognitive = 2
    max_evaluations = (part_size * num_iterations) + part_size

    dim = dimension

    unimodal_funcs = [SphereFunction, RotatedHyperEllipsoidFunction, RosenbrockFunction, DixonPriceFunction, QuarticNoiseFunction]
    multimodal_funcs = [GeneralizedShwefelFunction, RastriginFunction, AckleyFunction, GriewankFunction, LeviFunction]
    regular_functions = unimodal_funcs + multimodal_funcs

    cec_functions = []
    
    if topology == "GBEST":
        algo_name = "GPSO"
    elif topology == "LBEST":
        algo_name = "LPSO"
    else:
        raise ValueError("Topology does not exist")

    name_file = f'{algo_name}_dim_{dimension}_agents_{part_size}_iter_{num_iterations}_eval_{max_evaluations}'

    f_handle_csv = open(result_path + "/PSO_iter.csv", 'w+')
    writer_csv = csv.writer(f_handle_csv, delimiter=",")
    header = ['opt', 'func', 'exec_time'] + [f"run{str(i+1)}" for i in range(num_iterations)]
    writer_csv.writerow(header)

    for benchmark_func in regular_functions:
        simulations = []
        bf_iter = {}
        bf_eval = {}
        for simulation_id in range(num_exec):
            func = benchmark_func(dim)
            start = time.time()

            bests_iter, bests_eval = run_experiments(num_iterations, max_evaluations, part_size, func, search_space_initializer, topology=topology)
            end = time.time()
            best_fit = min(bests_iter)
            
            row_csv = ['PSO', func.function_name, (end - start)] + [(b if b != np.inf else 9.9e+999) for b in bests_iter[:num_iterations]]
            writer_csv.writerow(row_csv)
            bf_iter[f'convergence_simulation_{simulation_id}'] = bests_iter
            bf_eval[f'eval_simulation_{simulation_id}'] = func.best_evaluation_history
            simulations.append(best_fit)
            print(f"{func.function_name}\t t={end - start}\t min={best_fit}")    
        
        print(f'\t\tmean={np.mean(simulations)}\t std={np.std(simulations)}\n')              
        df_iter = pd.DataFrame(bf_iter)
        df_iter.to_csv(f"results/{name_file}_{func.function_name}_convergence.csv")

        df_be = pd.DataFrame(bf_eval)
        df_be.to_csv(f"results/{name_file}_{func.function_name}_evaluations.csv")
        
    f_handle_csv.close()

 
def run_experiments(n_iter, max_evaluations, part_size,  objective_function, search_space_initializer, topology):
    opt1 = PSO(objective_function=objective_function, space_initializer=search_space_initializer,
                n_iter=n_iter, max_evaluations = max_evaluations, part_size=part_size, inertia=0.73, cognitive=2.05,
                social=2.05, vel_max=10, clerck=True, topology = topology)
    opt1.optimize()
    return opt1.optimum_fitness_tracking_iter, opt1.optimum_fitness_tracking_eval

if __name__ == '__main__':
    main(topology='GBEST')
    main(topology='LBEST')