"""
Test RL-CMTEA with DaS Integration on CMT5: 20 runs comparison.
Standard budget: 200K FES (500 generations).
"""

import numpy as np
import time
import sys
sys.path.insert(0, '/Users/nguyenhien/Prj3-ver2')

from RL_CMTEA_DaS_v2 import RL_CMTEA
from Problems_py.Multi_task.Constrained_CMT.CMT5 import CMT5

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


print("=" * 70)
print("CMT5: RL-CMTEA with DaS vs Baseline (20 runs, 200K FES)")
print("=" * 70)

# Setup problem
problem = CMT5()
task_dicts = problem.get_tasks()
Tasks = [TaskWrapper(td) for td in task_dicts]

# Settings
num_runs = 20
sub_pop = 100
num_gens = 500  # 200K FES
sub_eva = num_gens * 4 * sub_pop

# Results storage
results_with_das = {'t1': [], 't2': [], 'cv1': [], 'cv2': []}
results_baseline = {'t1': [], 't2': [], 'cv1': [], 'cv2': []}

# Test WITH DaS
print(f"\n{'='*70}")
print("1. WITH DaS (use_das=True)")
print("="*70)
for run in range(num_runs):
    algo = RL_CMTEA(rng=run, use_das=True, das_eta=0.05, das_warmup=10)
    start = time.time()
    result = algo.run(Tasks, [sub_pop, sub_eva])
    elapsed = time.time() - start
    
    t1 = result['convergence'][0, -1]
    t2 = result['convergence'][1, -1]
    cv1 = result['convergence_cv'][0, -1]
    cv2 = result['convergence_cv'][1, -1]
    
    results_with_das['t1'].append(t1)
    results_with_das['t2'].append(t2)
    results_with_das['cv1'].append(cv1)
    results_with_das['cv2'].append(cv2)
    
    print(f"Run {run+1:2d}: T1={t1:.4e}, T2={t2:.4e}, CV1={cv1:.2e}, CV2={cv2:.2e} ({elapsed:.1f}s)")

# Test WITHOUT DaS (Baseline)
print(f"\n{'='*70}")
print("2. WITHOUT DaS (Baseline, use_das=False)")
print("="*70)
for run in range(num_runs):
    algo = RL_CMTEA(rng=run, use_das=False)
    start = time.time()
    result = algo.run(Tasks, [sub_pop, sub_eva])
    elapsed = time.time() - start
    
    t1 = result['convergence'][0, -1]
    t2 = result['convergence'][1, -1]
    cv1 = result['convergence_cv'][0, -1]
    cv2 = result['convergence_cv'][1, -1]
    
    results_baseline['t1'].append(t1)
    results_baseline['t2'].append(t2)
    results_baseline['cv1'].append(cv1)
    results_baseline['cv2'].append(cv2)
    
    print(f"Run {run+1:2d}: T1={t1:.4e}, T2={t2:.4e}, CV1={cv1:.2e}, CV2={cv2:.2e} ({elapsed:.1f}s)")

# Summary Statistics
print(f"\n{'='*70}")
print("SUMMARY STATISTICS (20 runs)")
print("="*70)

print("\n--- Task 1 ---")
print(f"{'Metric':<20} {'With DaS':<20} {'Baseline':<20} {'Winner':<10}")
print("-" * 70)
t1_das_mean = np.mean(results_with_das['t1'])
t1_das_std = np.std(results_with_das['t1'])
t1_das_best = np.min(results_with_das['t1'])
t1_base_mean = np.mean(results_baseline['t1'])
t1_base_std = np.std(results_baseline['t1'])
t1_base_best = np.min(results_baseline['t1'])

print(f"{'Mean':<20} {t1_das_mean:<20.4e} {t1_base_mean:<20.4e} {'DaS' if t1_das_mean < t1_base_mean else 'Baseline':<10}")
print(f"{'Std':<20} {t1_das_std:<20.4e} {t1_base_std:<20.4e}")
print(f"{'Best':<20} {t1_das_best:<20.4e} {t1_base_best:<20.4e} {'DaS' if t1_das_best < t1_base_best else 'Baseline':<10}")
feas1_das = sum(1 for cv in results_with_das['cv1'] if cv <= 1e-6)
feas1_base = sum(1 for cv in results_baseline['cv1'] if cv <= 1e-6)
print(f"{'Feasible':<20} {feas1_das}/20{'':<15} {feas1_base}/20")

print("\n--- Task 2 ---")
print(f"{'Metric':<20} {'With DaS':<20} {'Baseline':<20} {'Winner':<10}")
print("-" * 70)
t2_das_mean = np.mean(results_with_das['t2'])
t2_das_std = np.std(results_with_das['t2'])
t2_das_best = np.min(results_with_das['t2'])
t2_base_mean = np.mean(results_baseline['t2'])
t2_base_std = np.std(results_baseline['t2'])
t2_base_best = np.min(results_baseline['t2'])

print(f"{'Mean':<20} {t2_das_mean:<20.4e} {t2_base_mean:<20.4e} {'DaS' if t2_das_mean < t2_base_mean else 'Baseline':<10}")
print(f"{'Std':<20} {t2_das_std:<20.4e} {t2_base_std:<20.4e}")
print(f"{'Best':<20} {t2_das_best:<20.4e} {t2_base_best:<20.4e} {'DaS' if t2_das_best < t2_base_best else 'Baseline':<10}")
feas2_das = sum(1 for cv in results_with_das['cv2'] if cv <= 1e-6)
feas2_base = sum(1 for cv in results_baseline['cv2'] if cv <= 1e-6)
print(f"{'Feasible':<20} {feas2_das}/20{'':<15} {feas2_base}/20")

print("\n" + "=" * 70)
print("COMPLETE!")
print("=" * 70)
