
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
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

benchmarks = [
    (CMT1, "CMT1"), (CMT2, "CMT2"), (CMT3, "CMT3"),
    (CMT4, "CMT4"), (CMT5, "CMT5"), (CMT6, "CMT6"),
    (CMT7, "CMT7"), (CMT8, "CMT8"), (CMT9, "CMT9"),
]

ARTIFACTS_DIR = "/Users/nguyenhien/.gemini/antigravity/brain/6f508cb4-b32d-41d2-97fe-d203670fcc79"

for problem_class, name in benchmarks:
    print(f"Generating plot for {name}...")
    problem = problem_class()
    task_dicts = problem.get_tasks()
    Tasks = [TaskWrapper(td) for td in task_dicts]
    
    algo = RL_CMTEA(rng=0, use_das=True)
    result = algo.run(Tasks, [100, 200000 / 2]) # Run enough generations (FES=200K / 2 tasks = 100K per task)
    
    conv = result['convergence']
    gens = np.arange(conv.shape[1])
    
    plt.figure(figsize=(10, 6))
    plt.plot(gens, conv[0, :], label='Task 1', linewidth=2)
    plt.plot(gens, conv[1, :], label='Task 2', linewidth=2)
    plt.yscale('log')
    plt.title(f'{name} Convergence (RL-CMTEA + DaS)', fontsize=14)
    plt.xlabel('Generations', fontsize=12)
    plt.ylabel('Objective Value (log scale)', fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    save_path = os.path.join(ARTIFACTS_DIR, f"convergence_{name}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved to {save_path}")
