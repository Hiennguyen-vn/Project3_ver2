"""
Thực nghiệm RL-CMTEA-SMP-DaS trên các bài toán benchmark đơn giản.
"""

import numpy as np
import time
from dataclasses import dataclass
from typing import Tuple

# Import thuật toán
from RL_CMTEA import RL_CMTEA

# ====== Định nghĩa các Task Benchmark ======

@dataclass
class Task:
    """Base Task interface."""
    dim: int
    name: str = "Task"
    
    def evaluate(self, x: np.ndarray) -> Tuple[float, float]:
        """Trả về (cost, constraint_violation)"""
        raise NotImplementedError

class SphereTask(Task):
    """Sphere function: f(x) = sum(x^2), không có ràng buộc."""
    def __init__(self, dim: int, shift: float = 0.0):
        self.dim = dim
        self.shift = shift
        self.name = f"Sphere_{dim}D"
    
    def evaluate(self, x: np.ndarray) -> Tuple[float, float]:
        # Chuyển từ [0,1] sang [-5, 5]
        x_real = x[:self.dim] * 10 - 5 + self.shift
        cost = float(np.sum(x_real**2))
        cv = 0.0  # Không có ràng buộc
        return cost, cv

class RosenbrockTask(Task):
    """Rosenbrock function với ràng buộc."""
    def __init__(self, dim: int, shift: float = 0.0):
        self.dim = dim
        self.shift = shift
        self.name = f"Rosenbrock_{dim}D"
    
    def evaluate(self, x: np.ndarray) -> Tuple[float, float]:
        # Chuyển từ [0,1] sang [-2, 2]
        x_real = x[:self.dim] * 4 - 2 + self.shift
        
        # Rosenbrock
        cost = 0.0
        for i in range(self.dim - 1):
            cost += 100 * (x_real[i+1] - x_real[i]**2)**2 + (1 - x_real[i])**2
        
        # Ràng buộc: sum(x) <= dim/2
        cv = max(0.0, float(np.sum(x_real) - self.dim/2))
        return float(cost), cv

class RastriginTask(Task):
    """Rastrigin function với ràng buộc."""
    def __init__(self, dim: int, shift: float = 0.0):
        self.dim = dim
        self.shift = shift
        self.name = f"Rastrigin_{dim}D"
    
    def evaluate(self, x: np.ndarray) -> Tuple[float, float]:
        # Chuyển từ [0,1] sang [-5.12, 5.12]
        x_real = x[:self.dim] * 10.24 - 5.12 + self.shift
        
        # Rastrigin
        A = 10
        cost = A * self.dim + np.sum(x_real**2 - A * np.cos(2 * np.pi * x_real))
        
        # Ràng buộc: ||x||_inf <= 4
        cv = max(0.0, float(np.max(np.abs(x_real)) - 4))
        return float(cost), cv

class ConstrainedSphereTask(Task):
    """Sphere với nhiều ràng buộc."""
    def __init__(self, dim: int, shift: float = 0.0):
        self.dim = dim
        self.shift = shift
        self.name = f"ConstrainedSphere_{dim}D"
    
    def evaluate(self, x: np.ndarray) -> Tuple[float, float]:
        x_real = x[:self.dim] * 10 - 5 + self.shift
        
        # Objective
        cost = float(np.sum(x_real**2))
        
        # Ràng buộc: g1: sum(x) >= -dim, g2: sum(x) <= dim/2
        g1 = max(0.0, -self.dim - np.sum(x_real))  # sum(x) >= -dim
        g2 = max(0.0, np.sum(x_real) - self.dim/2)  # sum(x) <= dim/2
        cv = g1 + g2
        
        return cost, float(cv)


def run_single_experiment(Tasks, sub_pop, sub_eva, seed=None):
    """Chạy một lần thực nghiệm."""
    algo = RL_CMTEA(
        GA_MuC=2.0, 
        GA_MuM=5.0, 
        DE_F=0.5, 
        DE_CR=0.5,
        rng=seed,
        smp_mu=0.1,
        smp_lr=0.3,
        das_eta=1.0
    )
    
    RunPara = [sub_pop, sub_eva]
    
    start_time = time.time()
    result = algo.run(Tasks, RunPara)
    elapsed = time.time() - start_time
    
    return result, elapsed


def print_results(result, Tasks, elapsed):
    """In kết quả."""
    print("\n" + "="*60)
    print("KẾT QUẢ THỰC NGHIỆM")
    print("="*60)
    
    convergence = result["convergence"]
    convergence_cv = result["convergence_cv"]
    bestX = result["bestX"]
    
    for t in range(len(Tasks)):
        print(f"\n--- Task {t+1}: {Tasks[t].name} ---")
        print(f"  Best Objective: {convergence[t, -1]:.6e}")
        print(f"  Best CV:        {convergence_cv[t, -1]:.6e}")
        print(f"  Best X (5 đầu): {bestX[t][:min(5, len(bestX[t]))]}")
        print(f"  Số generations: {convergence.shape[1]}")
    
    print(f"\nThời gian chạy: {elapsed:.2f} giây")
    print("="*60)


def main():
    print("="*60)
    print("THỰC NGHIỆM RL-CMTEA-SMP-DaS")
    print("="*60)
    
    # ====== Cấu hình thực nghiệm ======
    dim = 10           # Số chiều
    sub_pop = 50       # Kích thước quần thể mỗi task
    sub_eva = 5000     # Số lượng đánh giá mỗi task
    num_runs = 3       # Số lần chạy
    
    # ====== Tạo các Tasks ======
    # Bài toán 1: 2 tasks tương tự (dễ transfer)
    print("\n[Bài toán 1: 2 Sphere Tasks - Tương tự, dễ transfer]")
    Tasks1 = [
        SphereTask(dim=dim, shift=0.0),
        SphereTask(dim=dim, shift=0.5),
    ]
    
    # Chạy thực nghiệm
    for run in range(num_runs):
        print(f"\n>>> Run {run+1}/{num_runs}")
        result, elapsed = run_single_experiment(Tasks1, sub_pop, sub_eva, seed=run)
        print_results(result, Tasks1, elapsed)
    
    # Bài toán 2: 2 tasks khác nhau (khó transfer hơn)
    print("\n" + "="*60)
    print("[Bài toán 2: Sphere + Rosenbrock - Khác nhau]")
    Tasks2 = [
        SphereTask(dim=dim, shift=0.0),
        RosenbrockTask(dim=dim, shift=0.0),
    ]
    
    for run in range(num_runs):
        print(f"\n>>> Run {run+1}/{num_runs}")
        result, elapsed = run_single_experiment(Tasks2, sub_pop, sub_eva, seed=run+100)
        print_results(result, Tasks2, elapsed)
    
    # Bài toán 3: 3 tasks với ràng buộc
    print("\n" + "="*60)
    print("[Bài toán 3: 3 Tasks với ràng buộc]")
    Tasks3 = [
        ConstrainedSphereTask(dim=dim, shift=0.0),
        RastriginTask(dim=dim, shift=0.0),
        RosenbrockTask(dim=dim, shift=0.0),
    ]
    
    for run in range(num_runs):
        print(f"\n>>> Run {run+1}/{num_runs}")
        result, elapsed = run_single_experiment(Tasks3, sub_pop, sub_eva, seed=run+200)
        print_results(result, Tasks3, elapsed)
    
    print("\n" + "="*60)
    print("HOÀN THÀNH TẤT CẢ THỰC NGHIỆM!")
    print("="*60)


if __name__ == "__main__":
    main()
