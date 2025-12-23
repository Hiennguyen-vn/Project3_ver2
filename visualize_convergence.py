import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Ensure the path is correct
sys.path.insert(0, '/Users/nguyenhien/Prj3-ver2')

from RL_CMTEA_SMP_DaS import RL_CMTEA_DaS

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

def run_and_plot(problem_class, name):
    print(f"Running {name}...")
    problem = problem_class()
    task_dicts = problem.get_tasks()
    Tasks = [TaskWrapper(td) for td in task_dicts]
    
    # Settings
    sub_pop = 100
    num_gens = 500 # 200K FES
    sub_eva = num_gens * 4 * sub_pop
    
    # Run algo (using seed 0 for consistency)
    algo = RL_CMTEA_DaS(rng=0, das_eta=0.05, das_warmup=10)
    result = algo.run(Tasks, [sub_pop, sub_eva])
    
    convergence = result['convergence'] # shape [2, num_gens]
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{name} Convergence (DaS KT)', fontsize=16)
    
    # Task 1
    axes[0].plot(convergence[0], label='Task 1', color='#1f77b4', linewidth=2)
    axes[0].set_title('Task 1 Convergence')
    axes[0].set_xlabel('Generation')
    axes[0].set_ylabel('Objective (Log Scale)')
    axes[0].set_yscale('log')
    axes[0].grid(True, which="both", ls="-", alpha=0.2)
    axes[0].legend()
    
    # Task 2
    axes[1].plot(convergence[1], label='Task 2', color='#ff7f0e', linewidth=2)
    axes[1].set_title('Task 2 Convergence')
    axes[1].set_xlabel('Generation')
    axes[1].set_ylabel('Objective (Log Scale)')
    axes[1].set_yscale('log')
    axes[1].grid(True, which="both", ls="-", alpha=0.2)
    axes[1].legend()
    
    filename = f'convergence_{name}.png'
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved {filename}")

def main():
    benchmarks = [
        (CMT1, "CMT1"),
        (CMT2, "CMT2"),
        (CMT3, "CMT3"),
        (CMT4, "CMT4"),
        (CMT5, "CMT5"),
        (CMT6, "CMT6"),
        (CMT7, "CMT7"),
        (CMT8, "CMT8"),
        (CMT9, "CMT9"),
    ]
    
    for problem_class, name in benchmarks:
        run_and_plot(problem_class, name)

if __name__ == "__main__":
    main()
