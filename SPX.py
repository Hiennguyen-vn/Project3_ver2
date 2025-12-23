import numpy as np
from dataclasses import dataclass
import copy
import math

# ==========================
# Cấu trúc dữ liệu
# ==========================
@dataclass
class Individual:
    rnvec: np.ndarray              # vector quyết định, chuẩn hoá [0,1]
    factorial_costs: float = None  # (tùy chọn) dùng ngoài SBX
    constraint_violation: float = None

@dataclass
class AlgoParams:
    GA_MuC: float = 20.0   # eta_c: chỉ số phân bố của SBX
    GA_MuM: float = 20.0   # eta_m: chỉ số phân bố của Polynomial Mutation

# ==========================
# SBX (GA_Crossover)
# ==========================
def GA_Crossover(x1: np.ndarray, x2: np.ndarray, eta_c: float, rng=None):
    """
    Simulated Binary Crossover (SBX)
    Trả về 2 con (c1, c2) có cùng kích thước với x1, x2.
    """
    rng = np.random.default_rng(rng)
    D = x1.size
    c1 = x1.copy()
    c2 = x2.copy()

    # SBX áp dụng theo từng gene
    u = rng.random(D)
    # mask chọn gene tham gia SBX; thông thường áp dụng cho tất cả gene
    # (nếu muốn có xác suất lai <1, có thể thêm mask khác)
    beta_q = np.empty(D, dtype=float)

    # Công thức SBX chuẩn (Deb, 2001)
    # beta = 2u  khi u <= 0.5
    # beta = 1/(2(1-u)) khi u > 0.5
    # beta_q = (beta)^(1/(eta_c+1))
    beta = np.where(u <= 0.5, 2*u, 1.0/(2.0*(1.0 - u)))
    beta_q = beta ** (1.0/(eta_c + 1.0))

    # Tạo hai con đối xứng quanh trung điểm
    c1 = 0.5 * ((x1 + x2) - beta_q * np.abs(x2 - x1))
    c2 = 0.5 * ((x1 + x2) + beta_q * np.abs(x2 - x1))

    # Cắt về [0,1]
    c1 = np.clip(c1, 0.0, 1.0)
    c2 = np.clip(c2, 0.0, 1.0)
    return c1, c2

# ==========================
# Polynomial Mutation (GA_Mutation)
# ==========================
def GA_Mutation(x: np.ndarray, eta_m: float, pm: float = None, rng=None):
    """
    Polynomial Mutation.
    - pm: xác suất đột biến mỗi gene; mặc định 1/D (chuẩn trong GA).
    """
    rng = np.random.default_rng(rng)
    D = x.size
    if pm is None:
        pm = 1.0 / D

    y = x.copy()
    # mask gene đc đột biến
    m = rng.random(D) < pm
    if not np.any(m):
        return y

    u = rng.random(D)
    # Công thức chuẩn (Deb, 2001)
    delta = np.empty(D)
    idx = u < 0.5
    delta[idx] = (2*u[idx]) ** (1.0/(eta_m + 1.0)) - 1.0
    delta[~idx] = 1.0 - (2*(1.0 - u[~idx])) ** (1.0/(eta_m + 1.0))

    y[m] = y[m] + delta[m]
    y = np.clip(y, 0.0, 1.0)
    return y

# ==========================
# SBX operator cho cả quần thể
# ==========================
def SBX(params: AlgoParams, population):
    """
    Tương đương MATLAB:
      - xáo trộn chỉ số,
      - ghép cặp p1 với p2 bằng offset floor(n/2),
      - SBX -> 2 con,
      - Polynomial Mutation trên từng con,
      - cắt về [0,1].
    Trả về danh sách offspring có cùng kích thước quần thể (nếu n lẻ, vẫn tạo đủ 2 con/cặp; cặp cuối cùng khớp như MATLAB).
    """
    n = len(population)
    perm = np.random.permutation(n)
    half = n // 2
    num_pairs = math.ceil(n / 2)

    offspring = []
    for i in range(num_pairs):
        p1 = perm[i]
        p2 = perm[i + half] if (i + half) < n else perm[i + half - n]  # bù vòng giống MATLAB khi cần

        child1 = copy.deepcopy(population[p1])
        child2 = copy.deepcopy(population[p2])

        # SBX
        c1_vec, c2_vec = GA_Crossover(population[p1].rnvec, population[p2].rnvec, params.GA_MuC)

        # Mutation (pm = 1/D)
        c1_vec = GA_Mutation(c1_vec, params.GA_MuM)
        c2_vec = GA_Mutation(c2_vec, params.GA_MuM)

        # clip [0,1] (thực ra 2 hàm trên đã clip, nhưng giữ đúng tinh thần MATLAB)
        c1_vec = np.clip(c1_vec, 0.0, 1.0)
        c2_vec = np.clip(c2_vec, 0.0, 1.0)

        child1.rnvec = c1_vec
        child2.rnvec = c2_vec

        offspring.append(child1)
        offspring.append(child2)

    # Nếu n lẻ, có thể sinh dư 1 cá thể; cắt về đúng kích thước n
    return offspring[:n]

# ==========================
# Ví dụ chạy nhanh
# ==========================
if __name__ == "__main__":
    np.random.seed(0)
    D = 7
    pop = [Individual(rnvec=np.random.rand(D)) for _ in range(5)]  # n lẻ để test case
    params = AlgoParams(GA_MuC=20.0, GA_MuM=20.0)

    off = SBX(params, pop)
    for i, ind in enumerate(off):
        print(f"offspring[{i}]: {np.round(ind.rnvec, 4)}")
