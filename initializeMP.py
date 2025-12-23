import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import math
import copy

# =========================
# Cấu trúc dữ liệu cơ bản
# =========================
@dataclass
class Individual:
    rnvec: np.ndarray
    factorial_costs: float = None
    constraint_violation: float = None

@dataclass
class Task:
    """Ví dụ interface: bạn có thể thay bằng task thật của bạn."""
    dim: int
    # hàm mục tiêu và vi phạm ràng buộc (trả về: (cost, cv))
    def evaluate(self, x: np.ndarray) -> Tuple[float, float]:
        # Ví dụ: Sphere + ràng buộc sum(x) - 0.5 <= 0
        cost = float(np.sum(x**2))
        cv = max(0.0, float(np.sum(x) - 0.5))  # CV >= 0; 0 là khả thi
        return cost, cv

# =========================================
# Các bộ xếp hạng ràng buộc (return indices)
# =========================================
def sort_FP(costs: np.ndarray, cvs: np.ndarray) -> np.ndarray:
    """
    Feasible Priority:
      - Ưu tiên cá thể khả thi (cv == 0).
      - Trong nhóm khả thi: sort theo cost tăng dần.
      - Trong nhóm vi phạm: sort theo cv tăng dần.
    """
    idx = np.arange(len(costs))
    feasible = cvs <= 0
    feas_idx = idx[feasible]
    infea_idx = idx[~feasible]

    feas_idx = feas_idx[np.argsort(costs[feasible], kind="mergesort")]
    infea_idx = infea_idx[np.argsort(cvs[~feasible], kind="mergesort")]

    return np.concatenate([feas_idx, infea_idx])

def sort_SR(costs: np.ndarray, cvs: np.ndarray, Pf: float = 0.45, iters: int = None,
            rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Stochastic Ranking (Runarsson & Yao, 2000/2005):
      - Bubble-sort kiểu xác suất:
        + Với xác suất Pf hoặc khi cả hai đều khả thi -> so sánh theo cost.
        + Ngược lại -> so sánh theo CV.
      - iters: số lượt quét; mặc định 2N (thực tế hay dùng N hoặc 2N).
      - rng: dùng Generator để tái lập (không trộn random.random).
    """
    N = len(costs)
    idx = list(range(N))
    if iters is None:
        iters = max(1, 2*N)
    if rng is None:
        rng = np.random.default_rng()

    for _ in range(iters):
        swapped = False
        for j in range(N - 1):
            a, b = idx[j], idx[j+1]
            both_feasible = (cvs[a] <= 0) and (cvs[b] <= 0)
            use_objective = both_feasible or (rng.random() < Pf)
            # tiêu chí so sánh
            if use_objective:
                better = costs[a] <= costs[b]
            else:
                better = cvs[a] <= cvs[b]
            if not better:
                idx[j], idx[j+1] = idx[j+1], idx[j]
                swapped = True
        if not swapped:
            break
    return np.array(idx, dtype=int)

def sort_EC(costs: np.ndarray, cvs: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """
    Epsilon-Constraint:
      - Xem cá thể nào có cv <= eps là 'khả thi'.
      - Nhóm khả thi: sort theo cost.
      - Nhóm còn lại: sort theo cv.
    """
    idx = np.arange(len(costs))
    feasible = cvs <= eps
    feas_idx = idx[feasible]
    infea_idx = idx[~feasible]

    feas_idx = feas_idx[np.argsort(costs[feasible], kind="mergesort")]
    infea_idx = infea_idx[np.argsort(cvs[~feasible], kind="mergesort")]

    return np.concatenate([feas_idx, infea_idx])

# =================================
# evaluate() cho một quần thể task
# =================================
def evaluate_population(pop: List[Individual], task: Task) -> Tuple[List[Individual], int]:
    """Đánh giá toàn bộ quần thể trên một task; trả về (pop, calls)."""
    calls = 0
    for ind in pop:
        cost, cv = task.evaluate(ind.rnvec)
        ind.factorial_costs = float(cost)
        ind.constraint_violation = float(cv)
        calls += 1
    return pop, calls

# =========================
# initializeMP (Python)
# =========================
def initialize_mp(Individual_class,
                  pop_size: int,
                  Tasks: List[Task],
                  dims,  # int hoặc List[int]; sẽ ưu tiên task.dim nếu có
                  init_type: str = 'Feasible_Priority',
                  var=None,
                  rng: Optional[np.random.Generator] = None):
    """
    Tương đương MATLAB:
      [population, fnceval_calls, bestobj, bestCV, bestX] = initializeMP(...)

    - Individual_class: callable tạo Individual (vd: lambda: Individual(np.zeros(d)))
    - pop_size: kích thước quần thể cho mỗi task
    - Tasks: danh sách Task
    - dims: int hoặc List[int] (chuẩn hoá [0,1]); nếu Task có .dim, sẽ dùng task.dim
    - init_type: 'Feasible_Priority' | 'Stochastic_Ranking' | 'Epsilon_Constraint'
    - var: tham số phụ:
        * SR: var[t] = Pf (xác suất so sánh theo objective)
        * EC: var[t] = eps (ngưỡng epsilon)
    - rng: np.random.Generator để tái lập
    """
    rng = rng if rng is not None else np.random.default_rng()

    # Chuẩn hoá dims per-task
    if isinstance(dims, int):
        dims_per_task = [dims for _ in Tasks]
    else:
        assert len(dims) == len(Tasks), "dims phải là int hoặc list cùng độ dài với Tasks"
        dims_per_task = list(dims)

    fnceval_calls = 0
    population = []   # dạng: list[ list[Individual] ] cho từng task
    bestobj = []
    bestCV = []
    bestX = []

    # Khởi tạo & đánh giá từng task
    for t, task in enumerate(Tasks):
        d_t = getattr(task, "dim", dims_per_task[t])

        # 1) Khởi tạo ngẫu nhiên rnvec trong [0,1]^d_t
        pop_t: List[Individual] = []
        for _ in range(pop_size):
            ind = Individual_class()
            ind.rnvec = rng.random(d_t)  # luôn [0,1]
            pop_t.append(ind)

        # 2) Đánh giá
        pop_t, calls = evaluate_population(pop_t, task)
        fnceval_calls += calls

        # 3) Xếp hạng theo init_type
        costs = np.array([ind.factorial_costs for ind in pop_t], dtype=float)
        cvs   = np.array([ind.constraint_violation for ind in pop_t], dtype=float)

        if init_type == 'Feasible_Priority':
            rank = sort_FP(costs, cvs)
        elif init_type == 'Stochastic_Ranking':
            Pf = 0.45 if (var is None or var[t] is None) else float(var[t])
            rank = sort_SR(costs, cvs, Pf=Pf, rng=rng)
        elif init_type == 'Epsilon_Constraint':
            eps = 1e-4 if (var is None or var[t] is None) else float(var[t])
            rank = sort_EC(costs, cvs, eps=eps)
        else:
            raise ValueError(f"Unknown init_type: {init_type}")

        # 4) Reorder: best lên index 0 (bất biến cho các DE_best_* )
        pop_t = [pop_t[i] for i in rank]

        # 5) Lưu best (đã ở vị trí 0)
        bestobj.append(pop_t[0].factorial_costs)
        bestCV.append(pop_t[0].constraint_violation)
        bestX.append(pop_t[0].rnvec.copy())

        # Gói vào population (dạng list cho từng task)
        population.append(pop_t)

    return population, fnceval_calls, bestobj, bestCV, bestX

# =========================
# Ví dụ chạy nhanh
# =========================
if __name__ == "__main__":
    # RNG cố định để tái lập
    rng = np.random.default_rng(0)

    # Tạo 2 task mẫu (khác dim)
    Tasks = [Task(dim=10), Task(dim=12)]
    pop_size = 6
    dims = [10, 12]  # hoặc có thể truyền int nếu mọi task cùng dim

    # Factory tạo Individual rỗng
    Individual_class = lambda: Individual(rnvec=None)

    # Feasible Priority
    pop, calls, bestobj, bestCV, bestX = initialize_mp(
        Individual_class, pop_size, Tasks, dims,
        init_type='Feasible_Priority', rng=rng
    )
    print("FP  -> calls:", calls, "| best cost:", np.round(bestobj, 4), "| best CV:", np.round(bestCV, 4))

    # Stochastic Ranking (Pf cho mỗi task)
    var_sr = [0.45, 0.45]
    pop, calls, bestobj, bestCV, bestX = initialize_mp(
        Individual_class, pop_size, Tasks, dims,
        init_type='Stochastic_Ranking', var=var_sr, rng=rng
    )
    print("SR  -> calls:", calls, "| best cost:", np.round(bestobj, 4), "| best CV:", np.round(bestCV, 4))

    # Epsilon-Constraint (eps cho mỗi task)
    var_ec = [1e-3, 1e-3]
    pop, calls, bestobj, bestCV, bestX = initialize_mp(
        Individual_class, pop_size, Tasks, dims,
        init_type='Epsilon_Constraint', var=var_ec, rng=rng
    )
    print("EC  -> calls:", calls, "| best cost:", np.round(bestobj, 4), "| best CV:", np.round(bestCV, 4))