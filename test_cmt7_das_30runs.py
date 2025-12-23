"""
Test RL-CMTEA with DaS on CMT7: 30 runs with paper budget (200K FES).
Compare with paper results: T1=1.13e+04, T2=1.29e+02
"""

import numpy as np
import time
import sys
sys.path.insert(0, '/Users/nguyenhien/Prj3-ver2')

from RL_CMTEA_DaS_v2 import RL_CMTEA
from Problems_py.Multi_task.Constrained_CMT.CMT7 import CMT7

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
print("CMT7: RL-CMTEA with DaS (30 runs, 200K FES)")
print("Paper Results: T1=1.13e+04, T2=1.29e+02")
print("=" * 70)

# Setup problem
problem = CMT7()
task_dicts = problem.get_tasks()
Tasks = [TaskWrapper(td) for td in task_dicts]

# Settings
num_runs = 30
sub_pop = 100
num_gens = 500  # 200K FES
sub_eva = num_gens * 4 * sub_pop

# Results storage
results = {'t1': [], 't2': [], 'cv1': [], 'cv2': []}

print(f"\nRunning {num_runs} independent runs...")
print("=" * 70)

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
    
    print(f"Run {run+1:2d}: T1={t1:.4e}, T2={t2:.4e}, CV1={cv1:.2e}, CV2={cv2:.2e} ({elapsed:.1f}s)")

# Summary Statistics
print(f"\n{'='*70}")
print("SUMMARY STATISTICS (30 runs)")
print("="*70)

t1_mean = np.mean(results['t1'])
t1_std = np.std(results['t1'])
t1_best = np.min(results['t1'])
t1_worst = np.max(results['t1'])

t2_mean = np.mean(results['t2'])
t2_std = np.std(results['t2'])
t2_best = np.min(results['t2'])
t2_worst = np.max(results['t2'])

feas1 = sum(1 for cv in results['cv1'] if cv <= 1e-6)
feas2 = sum(1 for cv in results['cv2'] if cv <= 1e-6)

print("\n--- Task 1 ---")
print(f"Mean:      {t1_mean:.4e}")
print(f"Std:       {t1_std:.4e}")
print(f"Best:      {t1_best:.4e}")
print(f"Worst:     {t1_worst:.4e}")
print(f"Feasible:  {feas1}/30")

print("\n--- Task 2 ---")
print(f"Mean:      {t2_mean:.4e}")
print(f"Std:       {t2_std:.4e}")
print(f"Best:      {t2_best:.4e}")
print(f"Worst:     {t2_worst:.4e}")
print(f"Feasible:  {feas2}/30")

# Comparison with Paper
print(f"\n{'='*70}")
print("COMPARISON WITH PAPER")
print("="*70)

paper_t1 = 1.13e+04
paper_t2 = 1.29e+02

t1_improvement = ((paper_t1 - t1_mean) / paper_t1) * 100
t2_improvement = ((paper_t2 - t2_mean) / paper_t2) * 100

print(f"\n{'Metric':<20} {'Paper':<20} {'DaS (Mean)':<20} {'Improvement':<15}")
print("-" * 70)
print(f"{'T1 Objective':<20} {paper_t1:<20.4e} {t1_mean:<20.4e} {t1_improvement:>+.1f}%")
print(f"{'T2 Objective':<20} {paper_t2:<20.4e} {t2_mean:<20.4e} {t2_improvement:>+.1f}%")

print(f"\n{'Metric':<20} {'Paper':<20} {'DaS (Best)':<20} {'Improvement':<15}")
print("-" * 70)
print(f"{'T1 Objective':<20} {paper_t1:<20.4e} {t1_best:<20.4e} {((paper_t1 - t1_best) / paper_t1) * 100:>+.1f}%")
print(f"{'T2 Objective':<20} {paper_t2:<20.4e} {t2_best:<20.4e} {((paper_t2 - t2_best) / paper_t2) * 100:>+.1f}%")

print("\n" + "=" * 70)
if t1_improvement > 0:
    print("✅ DaS BETTER than Paper on T1!")
else:
    print("❌ Paper better than DaS on T1")
    
if t2_improvement > 0:
    print("✅ DaS BETTER than Paper on T2!")
else:
    print("❌ Paper better than DaS on T2")
print("=" * 70)
