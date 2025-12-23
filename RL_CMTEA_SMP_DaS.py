"""
RL_CMTEA_DaS - Paper-like Version with DaS KT
==============================================
Giữ nguyên paper gốc, chỉ thay đổi Knowledge Transfer sử dụng DaS.

Changes from paper:
- KT sử dụng Domain-Adaptive Selection (DaS) với PCD matrix

Giống paper:
- Q-Learning alpha = 0.01 (cố định)
- UCB C = 2.0 (cố định)  
- Epsilon schedule giống paper (EC_TOP=0.2, EC_ALPHA=0.8, EC_CP=2.0, EC_TC=0.8)
- KHÔNG có Local Search
- KHÔNG có OBL
- KHÔNG có SHADE
"""

import numpy as np
import math
import copy
from dataclasses import dataclass
from typing import List, Tuple, Optional

from SPX import SBX

# tolerant imports
try:
    from RL_MFEA.DE_rand_1 import DE_rand_1
except Exception:
    from DE_rand_1 import DE_rand_1

try:
    from RL_MFEA.DE_rand_2 import DE_rand_2
except Exception:
    from DE_rand_2 import DE_rand_2

try:
    from RL_MFEA.DE_best_1 import DE_best_1
except Exception:
    from DE_best_1 import DE_best_1

try:
    from RL_MFEA.KT import KT
except Exception:
    from KT import KT

try:
    from RL_MFEA.initializeMP import initialize_mp
except Exception:
    from initializeMP import initialize_mp

try:
    from RL_MFEA.selectMP import select_mp
except Exception:
    from selectMP import select_mp


# ====== Constants (GIỐNG PAPER GỐC) ======
EPS = np.finfo(float).eps

# PCD parameters for DaS
DEFAULT_WARMUP_PCD = 0.7
PCD_MIN, PCD_MAX = 0.6, 0.95

# Epsilon-Constraint parameters (GIỐNG PAPER)
EC_TOP, EC_ALPHA, EC_CP, EC_TC = 0.2, 0.8, 2.0, 0.8

# Q-Learning parameters (GIỐNG PAPER)
QL_ALPHA = 0.01      # Cố định như paper
QL_GAMMA = 0.9

# UCB parameters (GIỐNG PAPER - sqrt(2))
UCB_C = math.sqrt(2.0)  # ~1.414, như code gốc
UCB_EPSILON = 1e-6

# DaS parameters
DAS_ETA_DEFAULT = 0.05
DAS_WARMUP = 10


# ====== Basic structures ======
@dataclass
class Individual:
    rnvec: np.ndarray
    factorial_costs: float = None
    constraint_violation: float = None


@dataclass
class AlgoParams:
    GA_MuC: float = 2.0
    GA_MuM: float = 5.0
    DE_F: float = 0.5
    DE_CR: float = 0.5


# ====== Utility functions ======
def evaluate_population(pop: List[Individual], task) -> Tuple[List[Individual], int]:
    calls = 0
    for ind in pop:
        cost, cv = task.evaluate(ind.rnvec)
        ind.factorial_costs = cost
        ind.constraint_violation = cv
        calls += 1
    return pop, calls


def gen2eva(conv_mat: np.ndarray) -> np.ndarray:
    return conv_mat


def uni2real(bestX_list: List[np.ndarray], Tasks) -> List[np.ndarray]:
    return bestX_list


# ====== DaS Functions (PHẦN MỚI) ======
def compute_kl_all_pairs(means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    """Vectorized KL divergence between all population pairs."""
    K, dims = means.shape
    stds = np.maximum(stds, EPS)
    mean_diff = means[:, np.newaxis, :] - means[np.newaxis, :, :]
    std_ratio = stds[np.newaxis, :, :] / stds[:, np.newaxis, :]
    var_src = stds[:, np.newaxis, :] ** 2
    var_tgt = stds[np.newaxis, :, :] ** 2
    kl = np.log(std_ratio) + (var_src + mean_diff**2) / (2.0 * var_tgt) - 0.5
    return np.maximum(kl, 0.0)


def update_pcd(populations: List[List[Individual]], K: int, dims: int,
               eta: float, gen: int, warmup_gens: int = DAS_WARMUP) -> np.ndarray:
    """Update Probability for Cultural Distance (PCD) matrix."""
    PCD = np.ones((K, K, dims), dtype=float)

    if gen <= warmup_gens:
        PCD[:, :, :] = DEFAULT_WARMUP_PCD
        return PCD

    means = np.zeros((K, dims), dtype=float)
    stds = np.zeros((K, dims), dtype=float)

    for k in range(K):
        if len(populations[k]) > 0:
            pop_matrix = np.array([ind.rnvec for ind in populations[k]], dtype=float)
            means[k, :] = np.mean(pop_matrix, axis=0)
            stds[k, :] = np.std(pop_matrix, axis=0) + EPS

    kl_all = compute_kl_all_pairs(means, stds)
    PCD = np.exp(-eta * kl_all)
    
    for k in range(K):
        PCD[k, k, :] = 1.0

    return np.clip(PCD, PCD_MIN, PCD_MAX)


def das_apply(child_vec: np.ndarray, parent_vec: np.ndarray,
              pcd_vec: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply DaS: selectively transfer dimensions based on PCD."""
    dims = len(child_vec)
    keep = rng.random(dims) < pcd_vec
    if not np.any(keep):
        keep[int(np.argmax(pcd_vec))] = True
    out = parent_vec.copy()
    out[keep] = child_vec[keep]
    return out


def apply_das_to_offspring(off1: List[List[Individual]],
                           target_pops: List[List[Individual]],
                           PCD: np.ndarray,
                           rng: np.random.Generator) -> List[List[Individual]]:
    """Apply DaS to KT offspring."""
    K = len(off1)
    for t in range(K):
        if len(off1[t]) == 0 or len(target_pops[t]) == 0:
            continue
        pcd_t = PCD[t, :, :].copy()
        pcd_t[t, :] = 0.0
        pcd_vec = np.max(pcd_t, axis=0)
        pop_t = target_pops[t]
        for i, child in enumerate(off1[t]):
            parent = pop_t[i % len(pop_t)]
            child.rnvec = das_apply(child.rnvec, parent.rnvec, pcd_vec, rng=rng)
            child.rnvec = np.clip(child.rnvec, 0.0, 1.0)
    return off1


def update_divd_divk(succ_flag_vec, divD: int, divK: int,
                     maxD: int, minK: int, maxK: int,
                     rng: np.random.Generator) -> Tuple[int, int]:
    succ_flag_vec = np.asarray(succ_flag_vec).astype(bool)
    if np.all(~succ_flag_vec):
        divD = rng.integers(1, maxD + 1)
        divK = rng.integers(minK, maxK + 1)
    elif np.any(~succ_flag_vec):
        divD = int(np.clip(rng.integers(divD - 1, divD + 2), 1, maxD))
        divK = int(np.clip(rng.integers(divK - 1, divK + 2), minK, maxK))
    return divD, divK


def select_operator(action_idx: int, params: AlgoParams, population: List[Individual]) -> List[Individual]:
    if action_idx == 0:
        return SBX(params, population)
    elif action_idx == 1:
        return DE_rand_1(params, population)
    elif action_idx == 2:
        return DE_rand_2(params, population)
    else:
        return DE_best_1(params, population)


# ====== Main Algorithm Class ======
class RL_CMTEA_DaS:
    """
    RL_CMTEA with Domain-Adaptive Selection (DaS) for Knowledge Transfer.
    
    Paper-like version:
    - Q-Learning alpha = 0.01 (fixed)
    - UCB C = 2.0 (fixed)
    - Epsilon schedule same as paper
    - NO Local Search, NO OBL, NO SHADE
    
    Only difference: KT uses DaS with PCD matrix.
    """
    
    def __init__(self,
                 GA_MuC: float = 2.0,
                 GA_MuM: float = 5.0,
                 DE_F: float = 0.5,
                 DE_CR: float = 0.5,
                 rng: Optional[int] = None,
                 das_eta: float = DAS_ETA_DEFAULT,
                 das_warmup: int = DAS_WARMUP):
        self.params = AlgoParams(GA_MuC=GA_MuC, GA_MuM=GA_MuM, DE_F=DE_F, DE_CR=DE_CR)
        self.rng = np.random.default_rng(rng)
        self.das_eta = float(das_eta)
        self.das_warmup = int(das_warmup)

    def run(self, Tasks, RunPara) -> dict:
        sub_pop, sub_eva = int(RunPara[0]), int(RunPara[1])
        eva_num = sub_eva * len(Tasks)
        K = len(Tasks)

        dims = max([getattr(t, "dim", getattr(t, "dims", None)) for t in Tasks])
        if dims is None:
            raise ValueError("Task needs .dim or .dims attribute")

        # Initialize populations
        Individual_factory = lambda: Individual(rnvec=None)
        population, fnceval_calls, bestobj, bestCV, bestX = initialize_mp(
            Individual_factory, sub_pop, Tasks, dims, init_type="Feasible_Priority"
        )

        convergence = np.zeros((K, 1))
        convergence_cv = np.zeros((K, 1))
        convergence[:, 0] = np.array(bestobj)
        convergence_cv[:, 0] = np.array(bestCV)
        data = {"bestX": copy.deepcopy(bestX)}

        # KT params
        maxD = int(np.min([max([getattr(t, "dim", getattr(t, "dims", 0)) for t in Tasks])]))
        main_divD = self.rng.integers(1, maxD + 1)
        aux_divD = self.rng.integers(1, maxD + 1)
        minK = 2
        maxK = max(2, sub_pop // 2)
        main_divK = int(self.rng.integers(minK, maxK + 1))
        aux_divK = int(self.rng.integers(minK, maxK + 1))

        # Q-learning + UCB (FIXED values like paper)
        alpha_ql = QL_ALPHA  # 0.01 fixed
        c_ucb = UCB_C        # 2.0 fixed
        
        num_pop_each_task, num_operator = 2, 4
        num_pop = num_pop_each_task * K
        Q_Table = np.zeros((num_pop, num_operator), dtype=float)
        action_counts = np.zeros((num_pop, num_operator), dtype=float)
        UCB_values = np.zeros((num_pop, num_operator), dtype=float)
        UCB_T = int(math.ceil(eva_num / (4 * sub_pop)))

        Ep = [0.0] * K
        main_pop = [None] * K
        aux_pop = [None] * K

        for t in range(K):
            n = int(math.ceil(EC_TOP * len(population[t])))
            cv_temp = np.array([ind.constraint_violation for ind in population[t]])
            idx_sort = np.argsort(cv_temp)
            Ep[t] = float(cv_temp[idx_sort[n - 1]]) if n > 0 else float(np.max(cv_temp))

            sub_pop1 = population[t][: sub_pop // 2]
            sub_pop2 = population[t][sub_pop // 2: sub_pop]

            main_pop[t], _, bestobj[t], bestCV[t], bestX[t], _ = select_mp(
                sub_pop1, sub_pop2, bestobj[t], bestCV[t], bestX[t], ep=0.0
            )
            aux_pop[t], _, _, _, _, _ = select_mp(
                sub_pop1, sub_pop2, bestobj[t], bestCV[t], bestX[t], ep=Ep[t]
            )

        gen = 1

        # =================== Main Evolution Loop ===================
        while fnceval_calls < eva_num:
            progress = fnceval_calls / float(eva_num)
            
            # ===== DaS: Compute PCD matrix (NEW) =====
            merged = [main_pop[t] + aux_pop[t] for t in range(K)]
            PCD = update_pcd(merged, K, dims, self.das_eta, gen, self.das_warmup)

            # Knowledge Transfer with DaS
            dims_list = [getattr(tt, "dim", getattr(tt, "dims", None)) for tt in Tasks]
            main_off1 = KT(self.params, dims_list, main_pop, divK=main_divK, divD=main_divD)
            aux_off1 = KT(self.params, dims_list, aux_pop, divK=aux_divK, divD=aux_divD)

            # ===== Apply DaS to KT offspring (NEW) =====
            main_off1 = apply_das_to_offspring(main_off1, main_pop, PCD, rng=self.rng)
            aux_off1 = apply_das_to_offspring(aux_off1, aux_pop, PCD, rng=self.rng)

            # Update epsilon (SAME as paper)
            for t in range(K):
                cv_arr = np.array([ind.constraint_violation for ind in aux_pop[t]])
                fes = cv_arr <= 0
                fea_percent = float(np.sum(fes)) / len(aux_pop[t])

                if fea_percent < 1:
                    Ep[t] = float(np.max(cv_arr))

                if progress < EC_TC:
                    if fea_percent < EC_ALPHA:
                        Ep[t] = float(Ep[t] * (1.0 - progress / EC_TC) ** EC_CP)
                    else:
                        Ep[t] = float(1.1 * np.max(cv_arr))
                else:
                    Ep[t] = 0.0

            main_flag = [False] * K
            aux_flag = [False] * K

            for t in range(K):
                t_1 = num_pop_each_task * (t + 1) - 2
                t_2 = num_pop_each_task * (t + 1) - 1

                # UCB calculation (FIXED c_ucb = 2.0)
                UCB_values[t_1, :] = Q_Table[t_1, :] + c_ucb * np.sqrt(
                    np.log(max(1, UCB_T)) / (action_counts[t_1, :] + UCB_EPSILON)
                )
                UCB_values[t_2, :] = Q_Table[t_2, :] + c_ucb * np.sqrt(
                    np.log(max(1, UCB_T)) / (action_counts[t_2, :] + UCB_EPSILON)
                )

                a1 = int(np.argmax(UCB_values[t_1, :]))
                a2 = int(np.argmax(UCB_values[t_2, :]))

                action_counts[t_1, a1] += 1.0
                action_counts[t_2, a2] += 1.0

                main_off2 = select_operator(a1, self.params, main_pop[t])
                aux_off2 = select_operator(a2, self.params, aux_pop[t])

                main_off = main_off1[t] + main_off2
                aux_off = aux_off1[t] + aux_off2

                main_off, calls = evaluate_population(main_off, Tasks[t])
                fnceval_calls += calls
                aux_off, calls = evaluate_population(aux_off, Tasks[t])
                fnceval_calls += calls

                main_pop[t], main_rank, bestobj[t], bestCV[t], bestX[t], main_flag[t] = select_mp(
                    main_pop[t], main_off, bestobj[t], bestCV[t], bestX[t], ep=0.0
                )
                main_pop[t], _, bestobj[t], bestCV[t], bestX[t], _ = select_mp(
                    main_pop[t], (main_off + aux_off), bestobj[t], bestCV[t], bestX[t], ep=0.0
                )
                aux_pop[t], aux_rank, _, _, _, aux_flag[t] = select_mp(
                    aux_pop[t], aux_off, bestobj[t], bestCV[t], bestX[t], ep=Ep[t]
                )

                # Success rate for RL
                main_next = np.zeros(len(main_rank), dtype=bool)
                aux_next = np.zeros(len(aux_rank), dtype=bool)
                main_next[main_rank[:len(main_pop[t])]] = True
                aux_next[aux_rank[:len(aux_pop[t])]] = True

                main_tail = main_next[len(main_pop[t]) + len(main_off1[t]):]
                aux_tail = aux_next[len(aux_pop[t]) + len(aux_off1[t]):]
                denom_main = max(1, len(main_pop[t]) + len(main_off2))
                denom_aux = max(1, len(aux_pop[t]) + len(aux_off2))

                main_succ_rate = float(np.sum(main_tail)) / float(denom_main)
                aux_succ_rate = float(np.sum(aux_tail)) / float(denom_aux)

                # Q-learning update (FIXED alpha = 0.01)
                Q_Table[t_1, a1] = Q_Table[t_1, a1] + alpha_ql * (
                    main_succ_rate + QL_GAMMA * np.max(Q_Table[t_1, :]) - Q_Table[t_1, a1]
                )
                Q_Table[t_2, a2] = Q_Table[t_2, a2] + alpha_ql * (
                    aux_succ_rate + QL_GAMMA * np.max(Q_Table[t_2, :]) - Q_Table[t_2, a2]
                )

            # Update divD/divK
            main_divD, main_divK = update_divd_divk(main_flag, main_divD, main_divK, maxD, minK, maxK, rng=self.rng)
            aux_divD, aux_divK = update_divd_divk(aux_flag, aux_divD, aux_divK, maxD, minK, maxK, rng=self.rng)

            gen += 1
            convergence = np.column_stack([convergence, np.array(bestobj, dtype=float)])
            convergence_cv = np.column_stack([convergence_cv, np.array(bestCV, dtype=float)])

        data["convergence"] = gen2eva(convergence)
        data["convergence_cv"] = gen2eva(convergence_cv)
        data["bestX"] = uni2real(bestX, Tasks)
        return data
