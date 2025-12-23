import numpy as np
import math
import copy
from dataclasses import dataclass
from typing import List, Tuple

from SPX import SBX, GA_Crossover, GA_Mutation

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
    from RL_MFEA.initializeMP import initialize_mp
except Exception:
    from initializeMP import initialize_mp

try:
    from RL_MFEA.selectMP import select_mp
except Exception:
    from selectMP import select_mp


@dataclass
class Individual:
    rnvec: np.ndarray
    factorial_costs: float = None
    constraint_violation: float = None
    src_task: int = -1
    op_id: int = -1
    parent_idx: int = -1


@dataclass
class AlgoParams:
    GA_MuC: float = 2.0
    GA_MuM: float = 5.0
    DE_F: float = 0.5
    DE_CR: float = 0.5


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


def roulette_choice(probs: np.ndarray, rng: np.random.Generator) -> int:
    p = np.asarray(probs, dtype=float)
    p = np.maximum(p, 0.0)
    s = float(np.sum(p))
    if not np.isfinite(s) or s <= 0.0:
        return int(rng.integers(0, len(p)))
    p /= s
    u = float(rng.random())
    c = np.cumsum(p)
    return int(np.searchsorted(c, u, side="right"))


def update_smp(smp: np.ndarray, Delta: np.ndarray, Count: np.ndarray,
               mu: float, lr: float) -> np.ndarray:
    K = smp.shape[0]
    num_sources = smp.shape[1]

    for k in range(K):
        avg_improve = np.zeros(num_sources, dtype=float)
        for j in range(num_sources):
            if Count[k, j] > 0:
                avg_improve[j] = Delta[k, j] / Count[k, j]
            else:
                avg_improve[j] = 0.0

        total = float(np.sum(avg_improve))
        if total > 1e-12:
            new_prob = avg_improve / total
        else:
            new_prob = np.ones(num_sources, dtype=float) / num_sources

        smp[k, :] = (1.0 - lr) * smp[k, :] + lr * new_prob
        lower_bound = mu / num_sources
        smp[k, :] = np.maximum(smp[k, :], lower_bound)
        smp[k, :] = smp[k, :] / float(np.sum(smp[k, :]))

    return smp


def compute_kl_divergence_per_dim(mean1: np.ndarray, std1: np.ndarray,
                                  mean2: np.ndarray, std2: np.ndarray) -> np.ndarray:
    eps = 1e-10
    std1 = np.maximum(std1, eps)
    std2 = np.maximum(std2, eps)
    kl = np.log(std2 / std1) + (std1 ** 2 + (mean1 - mean2) ** 2) / (2.0 * std2 ** 2) - 0.5
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
            pop_matrix = np.array([ind.rnvec for ind in populations[k]])
            means[k, :] = np.mean(pop_matrix, axis=0)
            stds[k, :] = np.std(pop_matrix, axis=0) + 1e-10

    for k in range(K):
        for j in range(K):
            if k != j:
                kl_div = compute_kl_divergence_per_dim(means[j], stds[j], means[k], stds[k])
                PCD[k, j, :] = np.exp(-eta * kl_div)

    PCD = np.clip(PCD, 0.05, 0.95)
    return PCD


def das_apply(offspring_rnvec: np.ndarray, parent_rnvec: np.ndarray,
              pcd_vector: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    dims = len(offspring_rnvec)
    rand_vals = rng.random(dims)
    keep_mask = rand_vals < pcd_vector
    if not np.any(keep_mask):
        keep_mask[int(np.argmax(pcd_vector))] = True
    result = parent_rnvec.copy()
    result[keep_mask] = offspring_rnvec[keep_mask]
    return result


def compute_constrained_spi(parent_cost: float, parent_cv: float,
                            child_cost: float, child_cv: float) -> float:
    parent_feasible = parent_cv <= 0
    child_feasible = child_cv <= 0

    if parent_feasible and child_feasible:
        denom = max(abs(parent_cost), 1e-12)
        imp = (parent_cost - child_cost) / denom
    elif child_feasible and not parent_feasible:
        denom = max(abs(parent_cv), 1e-12)
        imp = (parent_cv - child_cv) / denom
    elif parent_feasible and not child_feasible:
        imp = 0.0
    else:
        denom = max(abs(parent_cv), 1e-12)
        imp = (parent_cv - child_cv) / denom

    imp = float(max(imp, 0.0))
    imp = min(imp, 1.0)
    return imp ** 2


def compute_feasibility_rate(pop: List[Individual]) -> float:
    if len(pop) == 0:
        return 0.0
    return sum(1 for ind in pop if ind.constraint_violation <= 0) / len(pop)


def sample_smp_source_restricted(k: int, K: int, smp: np.ndarray,
                                 fea_rate: float, fea_threshold: float,
                                 rng: np.random.Generator) -> int:
    if fea_rate < fea_threshold:
        return k if rng.random() < 0.7 else K
    return roulette_choice(smp[k, :], rng=rng)


class RL_CMTEA:
    def __init__(self, GA_MuC=2.0, GA_MuM=5.0, DE_F=0.5, DE_CR=0.5, rng=None,
                 # SMP stable defaults (fix)
                 smp_mu=0.15, smp_lr=0.15,
                 # DaS stable defaults
                 das_eta=0.1, das_warmup=10,
                 # split feasibility thresholds (fix)
                 fea_threshold_main=0.4, fea_threshold_aux=0.2,
                 # adaptive transfer schedule (fix)
                 transfer_low=0.2, transfer_mid=0.5, transfer_high=0.8):
        self.params = AlgoParams(GA_MuC=GA_MuC, GA_MuM=GA_MuM, DE_F=DE_F, DE_CR=DE_CR)
        self.rng = np.random.default_rng(rng)

        self.smp_mu = float(smp_mu)
        self.smp_lr = float(smp_lr)

        self.das_eta = float(das_eta)
        self.das_warmup = int(das_warmup)

        self.fea_threshold_main = float(fea_threshold_main)
        self.fea_threshold_aux = float(fea_threshold_aux)

        self.transfer_low = float(transfer_low)
        self.transfer_mid = float(transfer_mid)
        self.transfer_high = float(transfer_high)

    def _transfer_ratio(self, fea_rate: float, progress: float) -> float:
        # fix: adaptive schedule: cautious early + infeasible, aggressive when stable
        # progress in [0,1]
        if fea_rate < 0.2:
            base = self.transfer_low
        elif fea_rate < 0.8:
            base = self.transfer_mid
        else:
            base = self.transfer_high
        # early stage dampening, later ramp up
        ramp = 0.5 + 0.5 * min(max(progress, 0.0), 1.0)  # 0.5 -> 1.0
        r = base * ramp
        return float(np.clip(r, 0.1, 0.9))

    def _generate_smp_offspring(self, params, k, populations, smp, PCD, dims,
                               fea_rate, fea_threshold, transfer_ratio, rng: np.random.Generator):
        K = len(populations)
        pop_k = populations[k]
        pop_size = len(pop_k)
        offspring_list = []

        n_transfer = int(max(1, round(pop_size * transfer_ratio)))

        for _ in range(n_transfer):
            i = int(rng.integers(0, pop_size))
            parent = pop_k[i]

            src_task = sample_smp_source_restricted(k, K, smp, fea_rate, fea_threshold, rng)

            if src_task == k:
                j = int(rng.integers(0, pop_size))
                while j == i and pop_size > 1:
                    j = int(rng.integers(0, pop_size))
                c1_vec, _ = GA_Crossover(parent.rnvec, pop_k[j].rnvec, params.GA_MuC, rng=rng)
                c1_vec = GA_Mutation(c1_vec, params.GA_MuM, rng=rng)
                c1_vec = np.clip(c1_vec, 0.0, 1.0)
                child = Individual(rnvec=c1_vec, src_task=k, op_id=-1, parent_idx=i)
                offspring_list.append((child, i, k))

            elif src_task == K:
                c_vec = GA_Mutation(parent.rnvec.copy(), params.GA_MuM, pm=1.0 / dims, rng=rng)
                c_vec = np.clip(c_vec, 0.0, 1.0)
                child = Individual(rnvec=c_vec, src_task=K, op_id=-1, parent_idx=i)
                offspring_list.append((child, i, K))

            else:
                pop_src = populations[src_task]
                if len(pop_src) == 0:
                    j = int(rng.integers(0, pop_size))
                    while j == i and pop_size > 1:
                        j = int(rng.integers(0, pop_size))
                    c1_vec, _ = GA_Crossover(parent.rnvec, pop_k[j].rnvec, params.GA_MuC, rng=rng)
                    c1_vec = GA_Mutation(c1_vec, params.GA_MuM, rng=rng)
                    c1_vec = np.clip(c1_vec, 0.0, 1.0)
                    child = Individual(rnvec=c1_vec, src_task=k, op_id=-1, parent_idx=i)
                    offspring_list.append((child, i, k))
                else:
                    j_src = int(rng.integers(0, len(pop_src)))
                    c1_vec, _ = GA_Crossover(parent.rnvec, pop_src[j_src].rnvec, params.GA_MuC, rng=rng)
                    c1_vec = GA_Mutation(c1_vec, params.GA_MuM, rng=rng)
                    c1_vec = np.clip(c1_vec, 0.0, 1.0)
                    c1_vec = das_apply(c1_vec, parent.rnvec, PCD[k, src_task, :], rng=rng)
                    child = Individual(rnvec=c1_vec, src_task=src_task, op_id=-1, parent_idx=i)
                    offspring_list.append((child, i, src_task))

        return offspring_list

    def _generate_rl_offspring(self, params, pop: List[Individual], op_id: int):
        if op_id == 0:
            offspring = SBX(params, pop)
        elif op_id == 1:
            offspring = DE_rand_1(params, pop)
        elif op_id == 2:
            offspring = DE_rand_2(params, pop)
        else:
            offspring = DE_best_1(params, pop)

        for idx, ind in enumerate(offspring):
            ind.op_id = op_id
            ind.parent_idx = idx % max(1, len(pop))
            ind.src_task = -1
        return offspring

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

        num_sources = K + 1
        main_smp = np.ones((K, num_sources), dtype=float) / num_sources
        aux_smp = np.ones((K, num_sources), dtype=float) / num_sources

        EC_Top, EC_Alpha, EC_Cp, EC_Tc = 0.2, 0.8, 2.0, 0.8

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
            progress = fnceval_calls / float(eva_num)

            merged_pop = [main_pop[t] + aux_pop[t] for t in range(K)]
            main_PCD = update_pcd(merged_pop, K, dims, self.das_eta, gen, self.das_warmup)
            aux_PCD = main_PCD  # reuse

            fea_rate_main = [compute_feasibility_rate(main_pop[t]) for t in range(K)]
            fea_rate_aux = [compute_feasibility_rate(aux_pop[t]) for t in range(K)]

            main_Delta = np.zeros((K, num_sources), dtype=float)
            main_Count = np.zeros((K, num_sources), dtype=float)
            aux_Delta = np.zeros((K, num_sources), dtype=float)
            aux_Count = np.zeros((K, num_sources), dtype=float)

            main_off1_data = {}
            aux_off1_data = {}
            main_off1 = {}
            aux_off1 = {}

            for t in range(K):
                tr_main = self._transfer_ratio(fea_rate_main[t], progress)
                tr_aux = self._transfer_ratio(fea_rate_aux[t], progress)

                main_off1_data[t] = self._generate_smp_offspring(
                    self.params, t, main_pop, main_smp, main_PCD, dims,
                    fea_rate_main[t], self.fea_threshold_main, tr_main, self.rng
                )
                aux_off1_data[t] = self._generate_smp_offspring(
                    self.params, t, aux_pop, aux_smp, aux_PCD, dims,
                    fea_rate_aux[t], self.fea_threshold_aux, tr_aux, self.rng
                )
                main_off1[t] = [item[0] for item in main_off1_data[t]]
                aux_off1[t] = [item[0] for item in aux_off1_data[t]]

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

            for t in range(K):
                t_1 = num_pop_each_task * (t + 1) - 2
                t_2 = num_pop_each_task * (t + 1) - 1

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

                main_off2 = self._generate_rl_offspring(self.params, main_pop[t], a1)
                aux_off2 = self._generate_rl_offspring(self.params, aux_pop[t], a2)

                main_off = main_off1[t] + main_off2
                aux_off = aux_off1[t] + aux_off2

                main_parent_info = [(main_pop[t][i].factorial_costs, main_pop[t][i].constraint_violation)
                                    for i in range(len(main_pop[t]))]
                aux_parent_info = [(aux_pop[t][i].factorial_costs, aux_pop[t][i].constraint_violation)
                                   for i in range(len(aux_pop[t]))]

                main_off, calls = evaluate_population(main_off, Tasks[t])
                fnceval_calls += calls
                aux_off, calls = evaluate_population(aux_off, Tasks[t])
                fnceval_calls += calls

                for idx, (_child, parent_idx, src_task) in enumerate(main_off1_data[t]):
                    if 0 <= src_task <= K and parent_idx < len(main_parent_info):
                        parent_cost, parent_cv = main_parent_info[parent_idx]
                        child_cost = main_off[idx].factorial_costs
                        child_cv = main_off[idx].constraint_violation
                        spi = compute_constrained_spi(parent_cost, parent_cv, child_cost, child_cv)
                        main_Delta[t, src_task] += spi
                        main_Count[t, src_task] += 1.0

                for idx, (_child, parent_idx, src_task) in enumerate(aux_off1_data[t]):
                    if 0 <= src_task <= K and parent_idx < len(aux_parent_info):
                        parent_cost, parent_cv = aux_parent_info[parent_idx]
                        child_cost = aux_off[idx].factorial_costs
                        child_cv = aux_off[idx].constraint_violation
                        spi = compute_constrained_spi(parent_cost, parent_cv, child_cost, child_cv)
                        aux_Delta[t, src_task] += spi
                        aux_Count[t, src_task] += 1.0

                old_main_pop_size = len(main_pop[t])
                old_aux_pop_size = len(aux_pop[t])

                main_pop_after_1, _, bestobj[t], bestCV[t], bestX[t], _ = select_mp(
                    main_pop[t], main_off, bestobj[t], bestCV[t], bestX[t], ep=0.0
                )
                main_pop[t], _, bestobj[t], bestCV[t], bestX[t], _ = select_mp(
                    main_pop_after_1, (main_off + aux_off), bestobj[t], bestCV[t], bestX[t], ep=0.0
                )
                aux_pop[t], _, _, _, _, _ = select_mp(
                    aux_pop[t], aux_off, bestobj[t], bestCV[t], bestX[t], ep=Ep[t]
                )

                main_selected_ids = {id(ind) for ind in main_pop[t]}
                aux_selected_ids = {id(ind) for ind in aux_pop[t]}
                main_succ_rate = float(sum((id(ind) in main_selected_ids) for ind in main_off2)) / max(1, len(main_off2))
                aux_succ_rate = float(sum((id(ind) in aux_selected_ids) for ind in aux_off2)) / max(1, len(aux_off2))

                Q_Table[t_1, a1] = Q_Table[t_1, a1] + alpha_ql * (
                    main_succ_rate + gamma_ql * np.max(Q_Table[t_1, :]) - Q_Table[t_1, a1]
                )
                Q_Table[t_2, a2] = Q_Table[t_2, a2] + alpha_ql * (
                    aux_succ_rate + gamma_ql * np.max(Q_Table[t_2, :]) - Q_Table[t_2, a2]
                )

                if len(main_pop[t]) != old_main_pop_size:
                    main_pop[t] = main_pop[t][:old_main_pop_size]
                if len(aux_pop[t]) != old_aux_pop_size:
                    aux_pop[t] = aux_pop[t][:old_aux_pop_size]

            main_smp = update_smp(main_smp, main_Delta, main_Count, self.smp_mu, self.smp_lr)
            aux_smp = update_smp(aux_smp, aux_Delta, aux_Count, self.smp_mu, self.smp_lr)

            gen += 1
            bestobj_arr = np.array(bestobj, dtype=float)
            bestcv_arr = np.array(bestCV, dtype=float)
            convergence = np.column_stack([convergence, bestobj_arr])
            convergence_cv = np.column_stack([convergence_cv, bestcv_arr])

        data["convergence"] = gen2eva(convergence)
        data["convergence_cv"] = gen2eva(convergence_cv)
        data["bestX"] = uni2real(bestX, Tasks)
        return data