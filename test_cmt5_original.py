"""
Test RL-CMTEA gốc (paper) trên CMT5.
So sánh với DaS version.
"""

import numpy as np
import time
import sys
sys.path.insert(0, '/Users/nguyenhien/Prj3-ver2')

from RL_CMTEA import RL_CMTEA  # Paper code
from Problems_py.Multi_task.Constrained_CMT.CMT5 import CMT5

# ====== Wrapper Task để tương thích với RL_CMTEA ======
class TaskWrapper:
    """Wrapper để chuyển đổi task dict sang object có .dim và .evaluate()"""
    def __init__(self, task_dict):
        self.dim = task_dict['dims']
        self.dims = task_dict['dims']
        self.fnc = task_dict['fnc']
        self.Lb = task_dict['Lb']
        self.Ub = task_dict['Ub']
    
    def evaluate(self, x: np.ndarray):
        """
        x: vector trong [0, 1]^D (unified search space)
        Trả về: (cost, constraint_violation)
        """
        x_real = self.Lb + x[:self.dim] * (self.Ub - self.Lb)
        cost, cv = self.fnc(x_real)
        return float(cost), float(cv)


def run_original_paper_experiment(num_generations=125, sub_pop=100, num_runs=30):
    """
    Chạy thực nghiệm với code paper gốc (không có DaS).
    """
    print("=" * 60)
    print("TEST RL-CMTEA GỐC (PAPER) TRÊN CMT5")
    print("=" * 60)
    
    # Tạo bài toán CMT5
    problem = CMT5()
    task_dicts = problem.get_tasks()
    
    # Wrap tasks
    Tasks = [TaskWrapper(td) for td in task_dicts]
    
    print(f"\nBài toán: CMT5")
    print(f"Số tasks: {len(Tasks)}")
    for i, t in enumerate(Tasks):
        print(f"  Task {i+1}: dim={t.dim}, Lb=[{t.Lb[0]:.0f}], Ub=[{t.Ub[0]:.0f}]")
    
    evals_per_gen = 4 * sub_pop
    sub_eva = num_generations * evals_per_gen
    
    print(f"\nCấu hình (PAPER GỐC):")
    print(f"  Population size: {sub_pop}")
    print(f"  Số thế hệ mục tiêu: {num_generations}")
    print(f"  Evaluations/task: {sub_eva}")
    print(f"  Số runs: {num_runs}")
    
    all_results = []
    
    for run in range(num_runs):
        print(f"\n{'='*60}")
        print(f"RUN {run+1}/{num_runs}")
        print("=" * 60)
        
        # Dùng RL_CMTEA gốc (không có DaS)
        algo = RL_CMTEA(
            GA_MuC=2.0,
            GA_MuM=5.0,
            DE_F=0.5,
            DE_CR=0.5,
            rng=run
        )
        
        RunPara = [sub_pop, sub_eva]
        
        start_time = time.time()
        result = algo.run(Tasks, RunPara)
        elapsed = time.time() - start_time
        
        all_results.append(result)
        
        convergence = result["convergence"]
        convergence_cv = result["convergence_cv"]
        bestX = result["bestX"]
        
        print(f"\nKết quả Run {run+1}:")
        print("-" * 40)
        
        for t in range(len(Tasks)):
            print(f"\n  Task {t+1}:")
            print(f"    Best Objective: {convergence[t, -1]:.6e}")
            print(f"    Best CV:        {convergence_cv[t, -1]:.6e}")
            print(f"    Feasible:       {'Yes' if convergence_cv[t, -1] <= 1e-6 else 'No'}")
        
        print(f"\n  Số thế hệ thực tế: {convergence.shape[1]}")
        print(f"  Thời gian: {elapsed:.2f} giây")
    
    # Tổng kết
    if num_runs > 1:
        print("\n" + "=" * 60)
        print("TỔNG KẾT QUA CÁC RUNS")
        print("=" * 60)
        
        for t in range(len(Tasks)):
            final_objs = [r["convergence"][t, -1] for r in all_results]
            final_cvs = [r["convergence_cv"][t, -1] for r in all_results]
            
            print(f"\nTask {t+1}:")
            print(f"  Best Obj:  Mean={np.mean(final_objs):.4e}, Std={np.std(final_objs):.4e}")
            print(f"  Best CV:   Mean={np.mean(final_cvs):.4e}, Std={np.std(final_cvs):.4e}")
            print(f"  Feasible:  {sum(1 for cv in final_cvs if cv <= 1e-6)}/{num_runs}")
    
    print("\n" + "=" * 60)
    print("HOÀN THÀNH!")
    print("=" * 60)
    
    return all_results


if __name__ == "__main__":
    # Paper budget: 1000*D = 50000 evals per task (D=50)
    results = run_original_paper_experiment(num_generations=125, sub_pop=100, num_runs=30)
