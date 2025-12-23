import numpy as np
from dataclasses import dataclass
import copy

@dataclass
class AlgoParams:
    DE_F: float = 0.5    # mutation factor
    DE_CR: float = 0.9   # crossover rate

@dataclass
class Individual:
    rnvec: np.ndarray    # decision vector (normalized in [0,1])

def DE_Crossover(v, x, CR):
    """Binomial crossover: combine mutant v with parent x"""
    D = len(v)
    jrand = np.random.randint(0, D)  # ensure at least one gene from v
    mask = np.random.rand(D) < CR
    mask[jrand] = True
    u = np.where(mask, v, x)
    return u

def DE_best_1(obj: AlgoParams, population):
    """DE/best/1 operator (Python version)"""
    pop_size = len(population)
    offspring = copy.deepcopy(population)

    for i in range(pop_size):
        # 1️⃣ Chọn ngẫu nhiên hai cá thể khác i
        idxs = np.arange(pop_size)
        idxs = idxs[idxs != i]
        np.random.shuffle(idxs)
        x1, x2 = population[idxs[0]], population[idxs[1]]

        # 2️⃣ Đột biến: best + F * (x1 - x2)
        best = population[0]  # cá thể tốt nhất (được sắp trước)
        mutant = best.rnvec + obj.DE_F * (x1.rnvec - x2.rnvec)

        # 3️⃣ Lai ghép (binomial crossover)
        trial = DE_Crossover(mutant, population[i].rnvec, obj.DE_CR)

        # 4️⃣ Giới hạn trong [0,1]
        trial = np.clip(trial, 0.0, 1.0)

        # 5️⃣ Cập nhật offspring
        offspring[i].rnvec = trial

    return offspring
