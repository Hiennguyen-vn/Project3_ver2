"""
Test RL-CMTEA with DaS integration on CMT5 (single run for verification).
"""

import numpy as np
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


print("=" * 60)
print("Testing RL-CMTEA with DaS Integration")
print("=" * 60)

# Setup problem
problem = CMT5()
task_dicts = problem.get_tasks()
Tasks = [TaskWrapper(td) for td in task_dicts]

# Test with DaS
print("\n1. Testing WITH DaS (use_das=True):")
algo = RL_CMTEA(rng=0, use_das=True, das_eta=0.05, das_warmup=10)
result = algo.run(Tasks, [100, 50000])  # 125 generations
print(f"   T1 Final: {result['convergence'][0, -1]:.4e}")
print(f"   T2 Final: {result['convergence'][1, -1]:.4e}")

# Test without DaS (baseline)
print("\n2. Testing WITHOUT DaS (use_das=False):")
algo = RL_CMTEA(rng=0, use_das=False)
result = algo.run(Tasks, [100, 50000])  # 125 generations  
print(f"   T1 Final: {result['convergence'][0, -1]:.4e}")
print(f"   T2 Final: {result['convergence'][1, -1]:.4e}")

print("\n" + "=" * 60)
print("Test Complete!")
print("=" * 60)
