import numpy as np
import math
import copy
from dataclasses import dataclass
from typing import List, Tuple

from SPX import SBX

# tolerant imports: modules live either at top-level (old layout) or under RL_MFEA package
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


# ====== evaluate utilities ======
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


# ====== DaS: KL + PCD ======
def compute_kl_divergence_per_dim(mean1: np.ndarray, std1: np.ndarray,
                                  mean2: np.ndarray, std2: np.ndarray) -> np.ndarray:
    eps = 1e-10
    std1 = np.maximum(std1, eps)
    std2 = np.maximum(std2, eps)
    kl = np.log(std2 / std1) + (std1**2 + (mean1 - mean2)**2) / (2.0 * std2**2) - 0.5
    return np.maximum(kl, 0.0)


def update_pcd(populations: List[List[Individual]], K: int, dims: int,
               eta: float, gen: int, warmup_gens: int = 10) -> np.ndarray:
    PCD = np.ones((K, K, dims), dtype=float)

    if gen <= warmup_gens:
        PCD[:, :, :] = 0.5
        return PCD

    means = np.zeros((K, dims), dtype=float)
    stds = np.zeros((K, dims), dtype=float)

    for k in range(K):
        if len(populations[k]) > 0:
            pop_matrix = np.array([ind.rnvec for ind in populations[k]], dtype=float)
            means[k, :] = np.mean(pop_matrix, axis=0)
            stds[k, :] = np.std(pop_matrix, axis=0) + 1e-10

    for k in range(K):
        for j in range(K):
            if k != j:
                kl_div = compute_kl_divergence_per_dim(means[j], stds[j], means[k], stds[k])
                PCD[k, j, :] = np.exp(-eta * kl_div)

    return np.clip(PCD, 0.05, 0.95)


def das_apply(child_vec: np.ndarray, parent_vec: np.ndarray,
              pcd_vec: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    dims = len(child_vec)
    keep = rng.random(dims) < pcd_vec

    if not np.any(keep):
        keep[int(np.argmax(pcd_vec))] = True

    out = parent_vec.copy()
    out[keep] = child_vec[keep]
    return out


def apply_das_to_off1(off1: List[List[Individual]],
                      target_pops: List[List[Individual]],
                      PCD: np.ndarray,
                      rng: np.random.Generator) -> List[List[Individual]]:
    """
    off1[t] comes from KT() but KT does not provide src-task per dimension.
    Safe heuristic: per target task t, use pcd_vec = max_j!=t PCD[t, j, :]
    Then keep only transferable dims from KT-child, otherwise keep target parent dim.
    """
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


def update_divd_divk(succ_flag_vec, divD, divK, maxD, minK, maxK, rng=None):
    rng = np.random.default_rng(rng)
    succ_flag_vec = np.asarray(succ_flag_vec).astype(bool)
    if np.all(~succ_flag_vec):
        divD = rng.integers(1, maxD + 1)
        divK = rng.integers(minK, maxK + 1)
    elif np.any(~succ_flag_vec):
        divD = int(np.clip(rng.integers(divD - 1, divD + 2), 1, maxD))
        divK = int(np.clip(rng.integers(divK - 1, divK + 2), minK, maxK))
    return divD, divK


# ====== RL_CMTEA + DaS-on-KT(off1) ======
class RL_CMTEA_DaS:
    def __init__(self,
                 GA_MuC=2.0, GA_MuM=5.0, DE_F=0.5, DE_CR=0.5,
                 rng=None,
                 das_eta=0.05, das_warmup=10):
        self.params = AlgoParams(GA_MuC=GA_MuC, GA_MuM=GA_MuM, DE_F=DE_F, DE_CR=DE_CR)
        self.rng = np.random.default_rng(rng)
        self.das_eta = float(das_eta)
        self.das_warmup = int(das_warmup)

    def run(self, Tasks, RunPara):
        sub_pop, sub_eva = int(RunPara[0]), int(RunPara[1])
        eva_num = sub_eva * len(Tasks)
        K = len(Tasks)

        dims = max([getattr(t, "dim", getattr(t, "dims", None)) for t in Tasks])
        if dims is None:
            raise ValueError("Task needs .dim or .dims attribute")

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

        # epsilon schedule
        EC_Top, EC_Alpha, EC_Cp, EC_Tc = 0.2, 0.8, 2.0, 0.8

        # Q-learning + UCB for off2 only (same as original)
        alpha_ql, gamma_ql = 0.01, 0.9
        num_pop_each_task, num_operator = 2, 4
        num_pop = num_pop_each_task * K
        Q_Table = np.zeros((num_pop, num_operator), dtype=float)
        action_counts = np.zeros((num_pop, num_operator), dtype=float)
        varepsilon_ucb = 1e-6
        UCB_values = np.zeros((num_pop, num_operator), dtype=float)
        UCB_T = int(math.ceil(eva_num / (4 * sub_pop)))

        Ep = [0.0] * K
        main_pop = [None] * K
        aux_pop = [None] * K

        for t in range(K):
            n = int(math.ceil(EC_Top * len(population[t])))
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

        while fnceval_calls < eva_num:
            # --- compute PCD from merged populations
            merged = [main_pop[t] + aux_pop[t] for t in range(K)]
            PCD = update_pcd(merged, K, dims, self.das_eta, gen, self.das_warmup)

            # --- off1 = KT then apply DaS to improve transfer
            dims_list = [getattr(tt, "dim", getattr(tt, "dims", None)) for tt in Tasks]
            main_off1 = KT(self.params, dims_list, main_pop, divK=main_divK, divD=main_divD)
            aux_off1 = KT(self.params, dims_list, aux_pop, divK=aux_divK, divD=aux_divD)

            main_off1 = apply_das_to_off1(main_off1, main_pop, PCD, rng=self.rng)
            aux_off1 = apply_das_to_off1(aux_off1, aux_pop, PCD, rng=self.rng)

            # --- update epsilon
            progress = fnceval_calls / float(eva_num)
            for t in range(K):
                fes = np.array([ind.constraint_violation for ind in aux_pop[t]]) <= 0
                fea_percent = float(np.sum(fes)) / len(aux_pop[t])

                if fea_percent < 1:
                    Ep[t] = float(np.max([ind.constraint_violation for ind in aux_pop[t]]))

                if progress < EC_Tc:
                    if fea_percent < EC_Alpha:
                        Ep[t] = float(Ep[t] * (1.0 - progress / EC_Tc) ** EC_Cp)
                    else:
                        Ep[t] = float(1.1 * np.max([ind.constraint_violation for ind in aux_pop[t]]))
                else:
                    Ep[t] = 0.0

            main_flag = [False] * K
            aux_flag = [False] * K

            # --- per task
            for t in range(K):
                t_1 = num_pop_each_task * (t + 1) - 2  # main row
                t_2 = num_pop_each_task * (t + 1) - 1  # aux row

                UCB_values[t_1, :] = Q_Table[t_1, :] + np.sqrt(
                    2.0 * np.log(max(1, UCB_T)) / (action_counts[t_1, :] + varepsilon_ucb)
                )
                UCB_values[t_2, :] = Q_Table[t_2, :] + np.sqrt(
                    2.0 * np.log(max(1, UCB_T)) / (action_counts[t_2, :] + varepsilon_ucb)
                )

                a1 = int(np.argmax(UCB_values[t_1, :]))
                a2 = int(np.argmax(UCB_values[t_2, :]))

                action_counts[t_1, a1] += 1.0
                action_counts[t_2, a2] += 1.0

                # off2 by RL chosen operator
                if a1 == 0:
                    main_off2 = SBX(self.params, main_pop[t])
                elif a1 == 1:
                    main_off2 = DE_rand_1(self.params, main_pop[t])
                elif a1 == 2:
                    main_off2 = DE_rand_2(self.params, main_pop[t])
                else:
                    main_off2 = DE_best_1(self.params, main_pop[t])

                if a2 == 0:
                    aux_off2 = SBX(self.params, aux_pop[t])
                elif a2 == 1:
                    aux_off2 = DE_rand_1(self.params, aux_pop[t])
                elif a2 == 2:
                    aux_off2 = DE_rand_2(self.params, aux_pop[t])
                else:
                    aux_off2 = DE_best_1(self.params, aux_pop[t])

                main_off = main_off1[t] + main_off2
                aux_off = aux_off1[t] + aux_off2

                main_off, calls = evaluate_population(main_off, Tasks[t])
                fnceval_calls += calls
                aux_off, calls = evaluate_population(aux_off, Tasks[t])
                fnceval_calls += calls

                # selection
                main_pop[t], main_rank, bestobj[t], bestCV[t], bestX[t], main_flag[t] = select_mp(
                    main_pop[t], main_off, bestobj[t], bestCV[t], bestX[t], ep=0.0
                )
                main_pop[t], _, bestobj[t], bestCV[t], bestX[t], _ = select_mp(
                    main_pop[t], (main_off + aux_off), bestobj[t], bestCV[t], bestX[t], ep=0.0
                )
                aux_pop[t], aux_rank, _, _, _, aux_flag[t] = select_mp(
                    aux_pop[t], aux_off, bestobj[t], bestCV[t], bestX[t], ep=Ep[t]
                )

                # success rate for RL update, count how many off2 survived
                main_selected = {id(ind) for ind in main_pop[t]}
                aux_selected = {id(ind) for ind in aux_pop[t]}
                main_succ_rate = float(sum(id(ind) in main_selected for ind in main_off2)) / max(1, len(main_off2))
                aux_succ_rate = float(sum(id(ind) in aux_selected for ind in aux_off2)) / max(1, len(aux_off2))

                Q_Table[t_1, a1] = Q_Table[t_1, a1] + alpha_ql * (
                    main_succ_rate + gamma_ql * np.max(Q_Table[t_1, :]) - Q_Table[t_1, a1]
                )
                Q_Table[t_2, a2] = Q_Table[t_2, a2] + alpha_ql * (
                    aux_succ_rate + gamma_ql * np.max(Q_Table[t_2, :]) - Q_Table[t_2, a2]
                )

            # heuristic update divD/divK (same as original)
            main_divD, main_divK = update_divd_divk(main_flag, main_divD, main_divK, maxD, minK, maxK, rng=self.rng)
            aux_divD, aux_divK = update_divd_divk(aux_flag, aux_divD, aux_divK, maxD, minK, maxK, rng=self.rng)

            gen += 1
            convergence = np.column_stack([convergence, np.array(bestobj, dtype=float)])
            convergence_cv = np.column_stack([convergence_cv, np.array(bestCV, dtype=float)])

        data["convergence"] = gen2eva(convergence)
        data["convergence_cv"] = gen2eva(convergence_cv)
        data["bestX"] = uni2real(bestX, Tasks)
        return data