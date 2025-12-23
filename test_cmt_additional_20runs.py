"""
Additional 20 runs (runs 11-30) for CMT1-CMT9 with DaS.
Results will be saved to combine with first 10 runs.
"""

import numpy as np
import time
import sys
import json
sys.path.insert(0, '/Users/nguyenhien/Prj3-ver2')

from RL_CMTEA_DaS_v2 import RL_CMTEA

from Problems_py.Multi_task.Constrained_CMT.CMT1 import CMT1
from Problems_py.Multi_task.Constrained_CMT.CMT2 import CMT2
from Problems_py.Multi_task.Constrained_CMT.CMT3 import CMT3
from Problems_py.Multi_task.Constrained_CMT.CMT4 import CMT4
from Problems_py.Multi_task.Constrained_CMT.CMT5 import CMT5
from Problems_py.Multi_task.Constrained_CMT.CMT6 import CMT6
from Problems_py.Multi_task.Constrained_CMT.CMT7 import CMT7
from Problems_py.Multi_task.Constrained_CMT.CMT8 import CMT8
from Problems_py.Multi_task.Constrained_CMT.CMT9 import CMT9

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


def run_benchmark(problem_class, name, start_run=10, num_runs=20):
    """Run additional runs with seeds 10-29."""
    problem = problem_class()
    task_dicts = problem.get_tasks()
    Tasks = [TaskWrapper(td) for td in task_dicts]
    
    sub_pop = 100
    num_gens = 500
    sub_eva = num_gens * 4 * sub_pop
    
    print(f"\n{'='*70}")
    print(f"{name}: Running {num_runs} additional runs (seeds {start_run}-{start_run+num_runs-1})...")
    print("="*70)
    
    results = {'t1': [], 't2': [], 'cv1': [], 'cv2': []}
    
    for i in range(num_runs):
        seed = start_run + i
        algo = RL_CMTEA(rng=seed, use_das=True, das_eta=0.05, das_warmup=10)
        start = time.time()
        result = algo.run(Tasks, [sub_pop, sub_eva])
        elapsed = time.time() - start
        
        t1 = result['convergence'][0, -1]
        t2 = result['convergence'][1, -1]
        cv1 = result['convergence_cv'][0, -1]
        cv2 = result['convergence_cv'][1, -1]
        
        results['t1'].append(float(t1))
        results['t2'].append(float(t2))
        results['cv1'].append(float(cv1))
        results['cv2'].append(float(cv2))
        
        print(f"Run {seed+1:2d}: T1={t1:.4e}, T2={t2:.4e} ({elapsed:.1f}s)")
    
    return results


def main():
    print("="*70)
    print("CMT1-CMT9: Additional 20 runs (11-30) with DaS")
    print("Results will be saved to cmt_results_runs_11_30.json")
    print("="*70)
    
    benchmarks = [
        (CMT1, "CMT1"), (CMT2, "CMT2"), (CMT3, "CMT3"),
        (CMT4, "CMT4"), (CMT5, "CMT5"), (CMT6, "CMT6"),
        (CMT7, "CMT7"), (CMT8, "CMT8"), (CMT9, "CMT9"),
    ]
    
    all_results = {}
    
    for problem_class, name in benchmarks:
        results = run_benchmark(problem_class, name, start_run=10, num_runs=20)
        all_results[name] = results
    
    # Save results to JSON
    with open('/Users/nguyenhien/Prj3-ver2/cmt_results_runs_11_30.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print("Results saved to cmt_results_runs_11_30.json")
    print("="*70)


if __name__ == "__main__":
    main()
