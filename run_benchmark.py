"""
Thực nghiệm đầy đủ RL-CMTEA-SMP-DaS trên bộ benchmark CMT1-CMT9.
"""

import numpy as np
import time
import sys
from datetime import datetime

sys.path.insert(0, '/Users/nguyenhien/Prj3-ver2')

from RL_CMTEA import RL_CMTEA
from Problems_py.Multi_task.Constrained_CMT.CMT1 import CMT1
from Problems_py.Multi_task.Constrained_CMT.CMT2 import CMT2
from Problems_py.Multi_task.Constrained_CMT.CMT3 import CMT3
from Problems_py.Multi_task.Constrained_CMT.CMT4 import CMT4
from Problems_py.Multi_task.Constrained_CMT.CMT5 import CMT5
from Problems_py.Multi_task.Constrained_CMT.CMT6 import CMT6
from Problems_py.Multi_task.Constrained_CMT.CMT7 import CMT7
from Problems_py.Multi_task.Constrained_CMT.CMT8 import CMT8
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


def run_experiment(problem_class, sub_pop=100, sub_eva=50000, num_runs=5, seed_base=0):
    """
    Chạy thực nghiệm cho một bài toán.
    
    Args:
        problem_class: Class của bài toán (CMT1, CMT2, ...)
        sub_pop: Kích thước quần thể mỗi task
        sub_eva: Số evaluations mỗi task
        num_runs: Số lần chạy độc lập
        seed_base: Seed cơ sở
    
    Returns:
        dict chứa kết quả thống kê
    """
    # Tạo bài toán
    problem = problem_class()
    problem_name = problem.get_name()
    task_dicts = problem.get_tasks()
    
    # Wrap tasks
    Tasks = [TaskWrapper(td) for td in task_dicts]
    num_tasks = len(Tasks)
    
    print(f"\n{'='*70}")
    print(f"BÀI TOÁN: {problem_name}")
    print(f"{'='*70}")
    print(f"Số tasks: {num_tasks}")
    for i, t in enumerate(Tasks):
        print(f"  Task {i+1}: dim={t.dim}")
    print(f"Cấu hình: pop={sub_pop}, eva/task={sub_eva}, runs={num_runs}")
    
    # Lưu kết quả
    all_best_obj = [[] for _ in range(num_tasks)]
    all_best_cv = [[] for _ in range(num_tasks)]
    all_times = []
    all_gens = []
    
    for run in range(num_runs):
        print(f"\n  Run {run+1}/{num_runs}...", end=" ", flush=True)
        
        # Khởi tạo thuật toán
        algo = RL_CMTEA(
            GA_MuC=2.0,
            GA_MuM=5.0,
            DE_F=0.5,
            DE_CR=0.5,
            rng=seed_base + run,
            smp_mu=0.1,
            smp_lr=0.3,
            das_eta=1.0
        )
        
        RunPara = [sub_pop, sub_eva]
        
        start_time = time.time()
        result = algo.run(Tasks, RunPara)
        elapsed = time.time() - start_time
        
        all_times.append(elapsed)
        all_gens.append(result["convergence"].shape[1])
        
        for t in range(num_tasks):
            all_best_obj[t].append(result["convergence"][t, -1])
            all_best_cv[t].append(result["convergence_cv"][t, -1])
        
        # In kết quả ngắn gọn
        obj_str = ", ".join([f"T{t+1}:{result['convergence'][t,-1]:.2e}" for t in range(num_tasks)])
        cv_str = ", ".join([f"{result['convergence_cv'][t,-1]:.2e}" for t in range(num_tasks)])
        print(f"Obj=[{obj_str}], CV=[{cv_str}], Time={elapsed:.1f}s")
    
    # Tính thống kê
    stats = {
        'problem': problem_name,
        'num_tasks': num_tasks,
        'dims': [Tasks[t].dim for t in range(num_tasks)],
        'num_runs': num_runs,
        'mean_time': np.mean(all_times),
        'mean_gens': np.mean(all_gens),
    }
    
    for t in range(num_tasks):
        stats[f'task{t+1}_obj_mean'] = np.mean(all_best_obj[t])
        stats[f'task{t+1}_obj_std'] = np.std(all_best_obj[t])
        stats[f'task{t+1}_obj_best'] = np.min(all_best_obj[t])
        stats[f'task{t+1}_cv_mean'] = np.mean(all_best_cv[t])
        stats[f'task{t+1}_cv_std'] = np.std(all_best_cv[t])
        stats[f'task{t+1}_feasible'] = sum(1 for cv in all_best_cv[t] if cv <= 1e-6)
    
    # In tổng kết
    print(f"\n  Tổng kết {problem_name}:")
    print(f"  {'-'*50}")
    for t in range(num_tasks):
        feas = stats[f'task{t+1}_feasible']
        print(f"  Task {t+1}: Obj = {stats[f'task{t+1}_obj_mean']:.4e} ± {stats[f'task{t+1}_obj_std']:.4e}")
        print(f"           CV  = {stats[f'task{t+1}_cv_mean']:.4e} ± {stats[f'task{t+1}_cv_std']:.4e}")
        print(f"           Feasible: {feas}/{num_runs} ({100*feas/num_runs:.0f}%)")
    print(f"  Avg time: {stats['mean_time']:.2f}s, Avg gens: {stats['mean_gens']:.0f}")
    
    return stats


def print_summary_table(all_stats):
    """In bảng tổng kết."""
    print("\n")
    print("=" * 100)
    print("BẢNG TỔNG KẾT KẾT QUẢ THỰC NGHIỆM RL-CMTEA-SMP-DaS")
    print("=" * 100)
    
    # Header
    print(f"\n{'Problem':<10} {'Task':<6} {'Dim':<5} {'Obj Mean':<14} {'Obj Std':<14} {'CV Mean':<14} {'Feas':<8}")
    print("-" * 85)
    
    for stats in all_stats:
        problem = stats['problem']
        num_tasks = stats['num_tasks']
        num_runs = stats['num_runs']
        
        for t in range(num_tasks):
            dim = stats['dims'][t]
            obj_mean = stats[f'task{t+1}_obj_mean']
            obj_std = stats[f'task{t+1}_obj_std']
            cv_mean = stats[f'task{t+1}_cv_mean']
            feas = stats[f'task{t+1}_feasible']
            
            if t == 0:
                print(f"{problem:<10} Task {t+1:<2} {dim:<5} {obj_mean:<14.4e} {obj_std:<14.4e} {cv_mean:<14.4e} {feas}/{num_runs}")
            else:
                print(f"{'':<10} Task {t+1:<2} {dim:<5} {obj_mean:<14.4e} {obj_std:<14.4e} {cv_mean:<14.4e} {feas}/{num_runs}")
        print("-" * 85)
    
    print("\n")


def main():
    print("=" * 70)
    print("THỰC NGHIỆM RL-CMTEA-SMP-DaS TRÊN BỘ BENCHMARK CMT1-CMT9")
    print(f"Thời gian bắt đầu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Cấu hình thực nghiệm
    SUB_POP = 100        # Kích thước quần thể
    SUB_EVA = 50000      # Số evaluations mỗi task (50 * 1000 như default)
    NUM_RUNS = 5         # Số lần chạy độc lập
    
    # Danh sách các bài toán
    problems = [
        CMT1, CMT2, CMT3, CMT4, CMT5, CMT6, CMT7, CMT8, CMT9
    ]
    
    all_stats = []
    total_start = time.time()
    
    for i, problem_class in enumerate(problems):
        seed_base = i * 100
        stats = run_experiment(
            problem_class,
            sub_pop=SUB_POP,
            sub_eva=SUB_EVA,
            num_runs=NUM_RUNS,
            seed_base=seed_base
        )
        all_stats.append(stats)
    
    total_time = time.time() - total_start
    
    # In bảng tổng kết
    print_summary_table(all_stats)
    
    print(f"Tổng thời gian chạy: {total_time/60:.2f} phút")
    print(f"Thời gian kết thúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print("HOÀN THÀNH TẤT CẢ THỰC NGHIỆM!")
    print("=" * 70)
    
    return all_stats


if __name__ == "__main__":
    results = main()
