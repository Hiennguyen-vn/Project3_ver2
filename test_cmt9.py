"""
Test RL-CMTEA-SMP-DaS trên CMT9 với 20 thế hệ.
"""

import numpy as np
import time
import sys
sys.path.insert(0, '/Users/nguyenhien/Prj3-ver2')

from RL_CMTEA_SMP_DaS import RL_CMTEA_DaS
from Problems_py.Multi_task.Constrained_CMT.CMT9 import CMT9

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
        # Chuyển từ [0,1] sang không gian thực [Lb, Ub]
        x_real = self.Lb + x[:self.dim] * (self.Ub - self.Lb)
        cost, cv = self.fnc(x_real)
        return float(cost), float(cv)


def run_cmt9_experiment(num_generations=20, sub_pop=50, num_runs=1):
    """
    Chạy thực nghiệm CMT9 với số thế hệ cố định.
    """
    print("=" * 60)
    print("TEST RL-CMTEA-SMP-DaS TRÊN CMT9")
    print("=" * 60)
    
    # Tạo bài toán CMT9
    problem = CMT9()
    task_dicts = problem.get_tasks()
    
    # Wrap tasks
    Tasks = [TaskWrapper(td) for td in task_dicts]
    
    print(f"\nBài toán: CMT9")
    print(f"Số tasks: {len(Tasks)}")
    for i, t in enumerate(Tasks):
        print(f"  Task {i+1}: dim={t.dim}, Lb=[{t.Lb[0]:.0f}], Ub=[{t.Ub[0]:.0f}]")
    
    # Tính số evaluations cho ~20 thế hệ
    # Mỗi thế hệ: 2 * sub_pop evaluations (main + aux) * 2 (off1 + off2) per task
    # Ước tính: ~4 * sub_pop * num_tasks per generation
    evals_per_gen = 4 * sub_pop  # per task
    sub_eva = num_generations * evals_per_gen
    
    print(f"\nCấu hình:")
    print(f"  Population size: {sub_pop}")
    print(f"  Số thế hệ mục tiêu: {num_generations}")
    print(f"  Evaluations/task: {sub_eva}")
    print(f"  Số runs: {num_runs}")
    
    all_results = []
    
    for run in range(num_runs):
        print(f"\n{'='*60}")
        print(f"RUN {run+1}/{num_runs}")
        print("=" * 60)
        
        # Khởi tạo thuật toán
        algo = RL_CMTEA_DaS(
            GA_MuC=2.0,
            GA_MuM=5.0,
            DE_F=0.5,
            DE_CR=0.5,
            rng=run,
            das_eta=0.05,
            das_warmup=10,
            adaptive_params=True,
            early_stopping=False
        )
        
        RunPara = [sub_pop, sub_eva]
        
        start_time = time.time()
        result = algo.run(Tasks, RunPara)
        elapsed = time.time() - start_time
        
        all_results.append(result)
        
        # In kết quả
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
            
            # Verify bằng cách evaluate lại
            final_cost, final_cv = Tasks[t].evaluate(bestX[t])
            print(f"    [Verify] Cost={final_cost:.6e}, CV={final_cv:.6e}")
        
        print(f"\n  Số thế hệ thực tế: {convergence.shape[1]}")
        print(f"  Thời gian: {elapsed:.2f} giây")
        
        # In lịch sử hội tụ (một số điểm)
        gens = convergence.shape[1]
        sample_points = [0, gens//4, gens//2, 3*gens//4, gens-1]
        sample_points = sorted(set([min(p, gens-1) for p in sample_points]))
        
        print(f"\n  Lịch sử hội tụ (Best Objective):")
        print(f"  {'Gen':<8}", end="")
        for t in range(len(Tasks)):
            print(f"{'Task'+str(t+1):<15}", end="")
        print()
        
        for g in sample_points:
            print(f"  {g+1:<8}", end="")
            for t in range(len(Tasks)):
                print(f"{convergence[t, g]:<15.4e}", end="")
            print()
    
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
    # Chạy CMT9 với 20 thế hệ, 3 runs
    results = run_cmt9_experiment(num_generations=100, sub_pop=100, num_runs=30)
