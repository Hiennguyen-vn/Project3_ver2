"""
Comprehensive test of RL-CMTEA with DaS on CMT1-CMT9.
10 runs per benchmark, 200K FES budget.
Compare with paper results.
"""

import numpy as np
import time
import sys
sys.path.insert(0, '/Users/nguyenhien/Prj3-ver2')

from RL_CMTEA_DaS_v2 import RL_CMTEA

# Import all CMT benchmarks
from Problems_py.Multi_task.Constrained_CMT.CMT1 import CMT1
from Problems_py.Multi_task.Constrained_CMT.CMT2 import CMT2
from Problems_py.Multi_task.Constrained_CMT.CMT3 import CMT3
from Problems_py.Multi_task.Constrained_CMT.CMT4 import CMT4
from Problems_py.Multi_task.Constrained_CMT.CMT5 import CMT5
from Problems_py.Multi_task.Constrained_CMT.CMT6 import CMT6
from Problems_py.Multi_task.Constrained_CMT.CMT7 import CMT7
from Problems_py.Multi_task.Constrained_CMT.CMT8 import CMT8
from Problems_py.Multi_task.Constrained_CMT.CMT9 import CMT9

# Paper results (from Table 3)
PAPER_RESULTS = {
    'CMT1': {'T1': 4.81e-17, 'T2': 7.98e-14},
    'CMT2': {'T1': 2.19e-09, 'T2': 5.92e-17},
    'CMT3': {'T1': 2.28e-04, 'T2': 1.30e-03},
    'CMT4': {'T1': 8.79e+01, 'T2': 8.15e+02},
    'CMT5': {'T1': 4.29e-12, 'T2': 9.74e+01},
    'CMT6': {'T1': 1.79e-08, 'T2': 6.60e-05},
    'CMT7': {'T1': 1.13e+04, 'T2': 1.29e+02},
    'CMT8': {'T1': 1.61e+01, 'T2': 9.19e+01},
    'CMT9': {'T1': 1.94e+01, 'T2': 3.32e+04},
}

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


def run_benchmark(problem_class, name, num_runs=10):
    """Run a single benchmark with DaS."""
    problem = problem_class()
    task_dicts = problem.get_tasks()
    Tasks = [TaskWrapper(td) for td in task_dicts]
    
    # Settings
    sub_pop = 100
    num_gens = 500  # 200K FES
    sub_eva = num_gens * 4 * sub_pop
    
    print(f"\n{'='*70}")
    print(f"{name}: Running {num_runs} runs with DaS...")
    print("="*70)
    
    results = {'t1': [], 't2': [], 'cv1': [], 'cv2': []}
    
    for run in range(num_runs):
        algo = RL_CMTEA(rng=run, use_das=True, das_eta=0.05, das_warmup=10)
        start = time.time()
        result = algo.run(Tasks, [sub_pop, sub_eva])
        elapsed = time.time() - start
        
        t1 = result['convergence'][0, -1]
        t2 = result['convergence'][1, -1]
        cv1 = result['convergence_cv'][0, -1]
        cv2 = result['convergence_cv'][1, -1]
        
        results['t1'].append(t1)
        results['t2'].append(t2)
        results['cv1'].append(cv1)
        results['cv2'].append(cv2)
        
        print(f"Run {run+1:2d}: T1={t1:.4e}, T2={t2:.4e} ({elapsed:.1f}s)")
    
    # Statistics
    t1_mean = np.mean(results['t1'])
    t1_best = np.min(results['t1'])
    t2_mean = np.mean(results['t2'])
    t2_best = np.min(results['t2'])
    feas1 = sum(1 for cv in results['cv1'] if cv <= 1e-6)
    feas2 = sum(1 for cv in results['cv2'] if cv <= 1e-6)
    
    print(f"\n--- {name} Summary ---")
    print(f"T1: Mean={t1_mean:.4e}, Best={t1_best:.4e}, Feasible={feas1}/{num_runs}")
    print(f"T2: Mean={t2_mean:.4e}, Best={t2_best:.4e}, Feasible={feas2}/{num_runs}")
    
    return {
        't1_mean': t1_mean, 't1_best': t1_best,
        't2_mean': t2_mean, 't2_best': t2_best,
        'feas1': feas1, 'feas2': feas2
    }


def main():
    print("="*70)
    print("CMT1-CMT9: RL-CMTEA with DaS Integration")
    print("10 runs per benchmark, 200K FES")
    print("="*70)
    
    benchmarks = [
        (CMT1, "CMT1"), (CMT2, "CMT2"), (CMT3, "CMT3"),
        (CMT4, "CMT4"), (CMT5, "CMT5"), (CMT6, "CMT6"),
        (CMT7, "CMT7"), (CMT8, "CMT8"), (CMT9, "CMT9"),
    ]
    
    all_results = {}
    
    for problem_class, name in benchmarks:
        results = run_benchmark(problem_class, name, num_runs=10)
        all_results[name] = results
    
    # Final comparison table
    print(f"\n{'='*70}")
    print("FINAL COMPARISON WITH PAPER")
    print("="*70)
    print(f"\n{'Problem':<10} {'Task':<6} {'Paper':<15} {'DaS (Mean)':<15} {'DaS (Best)':<15} {'Winner':<10}")
    print("-"*70)
    
    for name in ['CMT1', 'CMT2', 'CMT3', 'CMT4', 'CMT5', 'CMT6', 'CMT7', 'CMT8', 'CMT9']:
        paper = PAPER_RESULTS[name]
        das = all_results[name]
        
        # T1
        t1_winner = 'DaS' if das['t1_mean'] < paper['T1'] else 'Paper'
        print(f"{name:<10} {'T1':<6} {paper['T1']:<15.4e} {das['t1_mean']:<15.4e} {das['t1_best']:<15.4e} {t1_winner:<10}")
        
        # T2
        t2_winner = 'DaS' if das['t2_mean'] < paper['T2'] else 'Paper'
        print(f"{'':<10} {'T2':<6} {paper['T2']:<15.4e} {das['t2_mean']:<15.4e} {das['t2_best']:<15.4e} {t2_winner:<10}")
    
    # Count wins
    das_wins_t1 = sum(1 for name in all_results if all_results[name]['t1_mean'] < PAPER_RESULTS[name]['T1'])
    das_wins_t2 = sum(1 for name in all_results if all_results[name]['t2_mean'] < PAPER_RESULTS[name]['T2'])
    
    print(f"\n{'='*70}")
    print(f"DaS Wins: T1={das_wins_t1}/9, T2={das_wins_t2}/9")
    print("="*70)


if __name__ == "__main__":
    main()
