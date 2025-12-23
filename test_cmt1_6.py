"""
Test RL-CMTEA-DaS trên CMT1-CMT6 với paper budget (200K FES).
"""

import numpy as np
import time
import sys
sys.path.insert(0, '/Users/nguyenhien/Prj3-ver2')

from RL_CMTEA_SMP_DaS import RL_CMTEA_DaS

# Import all CMT benchmarks
from Problems_py.Multi_task.Constrained_CMT.CMT1 import CMT1
from Problems_py.Multi_task.Constrained_CMT.CMT2 import CMT2
from Problems_py.Multi_task.Constrained_CMT.CMT3 import CMT3
from Problems_py.Multi_task.Constrained_CMT.CMT4 import CMT4
from Problems_py.Multi_task.Constrained_CMT.CMT5 import CMT5
from Problems_py.Multi_task.Constrained_CMT.CMT6 import CMT6


class TaskWrapper:
    def __init__(self, task_dict):
        self.dim = task_dict['dims']
        self.dims = task_dict['dims']
        self.fnc = task_dict['fnc']
        self.Lb = task_dict['Lb']
        self.Ub = task_dict['Ub']
    
    def evaluate(self, x):
        x_real = self.Lb + x[:self.dim] * (self.Ub - self.Lb)
        cost, cv = self.fnc(x_real)
        return float(cost), float(cv)


def run_benchmark(problem_class, problem_name, num_runs=10, num_gens=500, sub_pop=100):
    """Run a single benchmark."""
    problem = problem_class()
    task_dicts = problem.get_tasks()
    Tasks = [TaskWrapper(td) for td in task_dicts]
    
    sub_eva = num_gens * 4 * sub_pop  # 200K FES
    
    print(f"\n{'='*60}")
    print(f"Benchmark: {problem_name}")
    print(f"Tasks: {len(Tasks)}, FES/task: {sub_eva}")
    print("="*60)
    
    all_results = {t: {'obj': [], 'cv': []} for t in range(len(Tasks))}
    
    for run in range(num_runs):
        algo = RL_CMTEA_DaS(rng=run, das_eta=0.05, das_warmup=10)
        start = time.time()
        result = algo.run(Tasks, [sub_pop, sub_eva])
        elapsed = time.time() - start
        
        for t in range(len(Tasks)):
            obj = result['convergence'][t, -1]
            cv = result['convergence_cv'][t, -1]
            all_results[t]['obj'].append(obj)
            all_results[t]['cv'].append(cv)
        
        objs = [f"T{t+1}={result['convergence'][t, -1]:.2e}" for t in range(len(Tasks))]
        print(f"Run {run+1:2d}: {', '.join(objs)} ({elapsed:.1f}s)")
    
    # Summary
    print(f"\n--- {problem_name} Summary ---")
    for t in range(len(Tasks)):
        objs = all_results[t]['obj']
        cvs = all_results[t]['cv']
        feasible = sum(1 for cv in cvs if cv <= 1e-6)
        print(f"T{t+1}: Mean={np.mean(objs):.4e} ± {np.std(objs):.2e}, "
              f"Best={np.min(objs):.4e}, Feasible={feasible}/{num_runs}")
    
    return all_results


def main():
    print("="*60)
    print("CMT1-CMT6 Experiments with DaS KT")
    print("Paper Budget: 200K FES, 10 runs each")
    print("="*60)
    
    benchmarks = [
        (CMT1, "CMT1"),
        (CMT2, "CMT2"),
        (CMT3, "CMT3"),
        (CMT4, "CMT4"),
        (CMT5, "CMT5"),
        (CMT6, "CMT6"),
    ]
    
    all_benchmark_results = {}
    
    for problem_class, name in benchmarks:
        results = run_benchmark(problem_class, name, num_runs=10, num_gens=500, sub_pop=100)
        all_benchmark_results[name] = results
    
    # Final summary table
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"{'Problem':<10} {'T1 Mean':<14} {'T1 Best':<14} {'T2 Mean':<14} {'T2 Best':<14}")
    print("-"*60)
    
    for name, results in all_benchmark_results.items():
        t1_mean = np.mean(results[0]['obj'])
        t1_best = np.min(results[0]['obj'])
        t2_mean = np.mean(results[1]['obj'])
        t2_best = np.min(results[1]['obj'])
        print(f"{name:<10} {t1_mean:<14.4e} {t1_best:<14.4e} {t2_mean:<14.4e} {t2_best:<14.4e}")
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)


if __name__ == "__main__":
    main()
