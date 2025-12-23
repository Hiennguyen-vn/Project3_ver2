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
    return np.where(mask, v, x)

def DE_rand_2(obj: AlgoParams, population):
    """DE/rand/2 operator (Python version)"""
    pop_size = len(population)
    offspring = copy.deepcopy(population)

    for i in range(pop_size):
        # 1️⃣ chọn 4 cá thể ngẫu nhiên khác i
        idxs = np.arange(pop_size)
        idxs = idxs[idxs != i]
        np.random.shuffle(idxs)
        x1, x2, x3, x4 = population[idxs[0]], population[idxs[1]], population[idxs[2]], population[idxs[3]]

        # 2️⃣ đột biến: v = x_i + F*(x1 - x2) + F*(x3 - x4)
        v = (population[i].rnvec
             + obj.DE_F * (x1.rnvec - x2.rnvec)
             + obj.DE_F * (x3.rnvec - x4.rnvec))

        # 3️⃣ lai ghép với parent
        trial = DE_Crossover(v, population[i].rnvec, obj.DE_CR)

        # 4️⃣ giới hạn biên trong [0,1]
        trial = np.clip(trial, 0.0, 1.0)

        # 5️⃣ cập nhật offspring
        offspring[i].rnvec = trial

    return offspring
