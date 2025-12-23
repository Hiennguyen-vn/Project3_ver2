import numpy as np
from typing import List, Tuple

# ---- nếu đã có rồi thì bỏ đoạn này ----
class Individual:
    def __init__(self, rnvec, factorial_costs=None, constraint_violation=None):
        self.rnvec = rnvec
        self.factorial_costs = factorial_costs
        self.constraint_violation = constraint_violation

def sort_EC(costs: np.ndarray, cvs: np.ndarray, eps: float) -> np.ndarray:
    """Epsilon-Constraint ranking: feasible (cv<=eps) by cost, others by cv."""
    idx = np.arange(len(costs))
    feasible = cvs <= eps
    feas_idx = idx[feasible][np.argsort(costs[feasible], kind="mergesort")]
    infea_idx = idx[~feasible][np.argsort(cvs[~feasible],   kind="mergesort")]
    return np.concatenate([feas_idx, infea_idx])
# ---------------------------------------

def select_mp(population: List[Individual],
              offspring: List[Individual],
              bestobj: float, bestCV: float, bestX: np.ndarray,
              ep: float
              ) -> Tuple[List[Individual], np.ndarray, float, float, np.ndarray, bool]:
    """
      [population, rank, bestobj, bestCV, bestX, Flag] = selectMP(...)
    - population, offspring: danh sách Individual (kích thước bằng nhau)
    - ep: epsilon cho Epsilon-Constraint
    """
    # 1) Ghép cha + con
    pop_temp = population + offspring

    # 2) Lấy mảng obj & cv
    obj = np.array([ind.factorial_costs for ind in pop_temp], dtype=float)
    cv  = np.array([ind.constraint_violation for ind in pop_temp], dtype=float)

    # 3) Xếp hạng bằng Epsilon-Constraint
    rank = sort_EC(obj, cv, ep)

    # 4) Chọn elite: giữ lại top N và GIỮ THỨ TỰ THEO RANK
    N = len(population)
    population_new = [pop_temp[i] for i in rank[:N]]

    # 5) Cập nhật best toàn cục
    Flag = False
    best_idx_now = int(rank[0])
    bestobj_now = pop_temp[best_idx_now].factorial_costs
    bestCV_now  = pop_temp[best_idx_now].constraint_violation
    bestX_now   = pop_temp[best_idx_now].rnvec

    if (bestCV_now < bestCV) or (bestCV_now == bestCV and bestobj_now <= bestobj):
        bestobj, bestCV, bestX = bestobj_now, bestCV_now, bestX_now.copy(), 
        Flag = True

    # Quan trọng: sau bước này, population_new[0] là cá thể tốt nhất hiện tại
    return population_new, rank, bestobj, bestCV, bestX, Flag
