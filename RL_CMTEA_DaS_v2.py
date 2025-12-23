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

# Import DaS_KT for domain-adaptive dimension selection
try:
    from DaS_KT import DaS_KT
except Exception:
    DaS_KT = None  # Fallback if not available

try:
    from RL_MFEA.initializeMP import initialize_mp
except Exception:
    from initializeMP import initialize_mp

try:
    from RL_MFEA.selectMP import select_mp
except Exception:
    from selectMP import select_mp

# ====== Các cấu trúc & tiện ích cơ bản (đồng bộ với các phần trước) ======
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

# --- Đã có các hàm này ở các message trước, nhắc lại khai báo ---
# SBX(), GA_Crossover(), GA_Mutation()
# DE_best_1(), DE_rand_1(), DE_rand_2()
# KT()
# initialize_mp(), select_mp()
# sort_EC()

# ====== Stub: evaluate cho list Individual trên một Task ======
def evaluate_population(pop: List[Individual], task) -> Tuple[List[Individual], int]:
    """
    Đánh giá toàn bộ cá thể trong pop trên 'task'.
    Task cần có method .evaluate(x) -> (cost, cv).
    """
    calls = 0
    for ind in pop:
        cost, cv = task.evaluate(ind.rnvec)
        ind.factorial_costs = cost
        ind.constraint_violation = cv
        calls += 1
    return pop, calls

# ====== Stub: chuyển gen->tiến hoá theo trục đánh giá (tuỳ bạn) ======
def gen2eva(conv_mat: np.ndarray) -> np.ndarray:
    # MATLAB: gen2eva(convergence) ánh xạ theo số eval; ở đây trả nguyên
    return conv_mat

# ====== Stub: chuyển toạ độ chuẩn hoá -> không gian thực (tuỳ task) ======
def uni2real(bestX_list: List[np.ndarray], Tasks) -> List[np.ndarray]:
    # Nếu Tasks có hàm biến đổi thì gọi ở đây; mặc định trả nguyên
    return bestX_list

# ====== Hàm update divD & divK giống MATLAB ======
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

# ====== Lớp RL-CMTEA (Python) - PHIÊN BẢN GỐC ======
class RL_CMTEA:
    """
    Bản Python hoá lớp RL_CMTEA trong MATLAB.
    Dùng cùng tham số mặc định: GA_MuC=2, GA_MuM=5, DE_F=0.5, DE_CR=0.5.
    
    Phiên bản này sử dụng:
    - RL (Q-Learning + UCB) cho việc chọn operator (SBX, DE variants)
    - Heuristic cho việc chọn (divK, divD) trong KT
    """

    def __init__(self, GA_MuC=2.0, GA_MuM=5.0, DE_F=0.5, DE_CR=0.5, rng=None, 
                 use_das=True, das_eta=0.05, das_warmup=10):
        self.params = AlgoParams(GA_MuC=GA_MuC, GA_MuM=GA_MuM, DE_F=DE_F, DE_CR=DE_CR)
        self.rng = np.random.default_rng(rng)
        self.use_das = use_das and (DaS_KT is not None)
        self.das_eta = das_eta
        self.das_warmup = das_warmup
        self.das_kt = None  # Will be initialized in run()

    def run(self, Tasks, RunPara):
        """
        Tasks: list Task; mỗi Task có .dims (hoặc .dim) và .evaluate(x)->(cost, cv)
        RunPara: (sub_pop, sub_eva)
           sub_pop: kích thước quần thể mỗi task
           sub_eva: ngân sách đánh giá cho mỗi task (MATLAB nhân với #Tasks)
        """
        sub_pop, sub_eva = int(RunPara[0]), int(RunPara[1])
        eva_num = sub_eva * len(Tasks)

        # ---------- Khởi tạo ----------
        dims = max([getattr(t, 'dim', getattr(t, 'dims', None)) for t in Tasks])
        if dims is None:
            raise ValueError("Task needs .dim or .dims attribute")

        Individual_factory = lambda: Individual(rnvec=None)
        population, fnceval_calls, bestobj, bestCV, bestX = initialize_mp(
            Individual_factory, sub_pop, Tasks, dims, init_type='Feasible_Priority'
        )

        # convergence lưu theo [task, gen]
        convergence = np.zeros((len(Tasks), 1))
        convergence_cv = np.zeros((len(Tasks), 1))
        convergence[:, 0] = np.array(bestobj)
        convergence_cv[:, 0] = np.array(bestCV)
        data = {"bestX": copy.deepcopy(bestX)}

        # ---------- Tham số KT ----------
        maxD = int(np.min([max([getattr(t, 'dim', getattr(t, 'dims', 0)) for t in Tasks])]))
        main_divD = self.rng.integers(1, maxD + 1)
        aux_divD = self.rng.integers(1, maxD + 1)
        minK = 2
        maxK = max(2, sub_pop // 2)
        main_divK = int(self.rng.integers(minK, maxK + 1))
        aux_divK  = int(self.rng.integers(minK, maxK + 1))
        
        # Initialize DaS_KT if enabled
        if self.use_das:
            self.das_kt = DaS_KT(num_tasks=len(Tasks), max_dim=maxD, 
                               eta=self.das_eta, warmup=self.das_warmup)

        # ---------- Epsilon schedule ----------
        EC_Top, EC_Alpha, EC_Cp, EC_Tc = 0.2, 0.8, 2.0, 0.8

        # ---------- Q-learning + UCB ----------
        alpha_ql, gamma_ql = 0.01, 0.9
        num_pop_each_task, num_operator = 2, 4  # main + aux; 4 operator
        num_pop = num_pop_each_task * len(Tasks)
        Q_Table = np.zeros((num_pop, num_operator), dtype=float)
        action_counts = np.zeros((num_pop, num_operator), dtype=float)
        varepsilon_ucb = 1e-6
        UCB_values = np.zeros((num_pop, num_operator), dtype=float)
        UCB_T = int(math.ceil(eva_num / (4 * sub_pop)))

        # Tách main/aux theo EC=0 & EC=Ep(t)
        Ep = [0.0] * len(Tasks)
        main_pop = [None] * len(Tasks)
        aux_pop  = [None] * len(Tasks)
        for t in range(len(Tasks)):
            # Epsilon init theo top-20% CV
            n = int(math.ceil(EC_Top * len(population[t])))
            cv_temp = np.array([ind.constraint_violation for ind in population[t]])
            idx_sort = np.argsort(cv_temp)
            Ep[t] = float(cv_temp[idx_sort[n-1]]) if n > 0 else float(np.max(cv_temp))

            # Chia đôi quần thể
            sub_pop1 = population[t][: sub_pop // 2]
            sub_pop2 = population[t][sub_pop // 2 : sub_pop]

            # Chọn lọc EC=0 (main)
            main_pop[t], _, bestobj[t], bestCV[t], bestX[t], _ = select_mp(sub_pop1, sub_pop2, bestobj[t], bestCV[t], bestX[t], ep=0.0)
            # Chọn lọc EC=Ep[t] (aux)
            aux_pop[t],  _, _, _, _, _ = select_mp(sub_pop1, sub_pop2, bestobj[t], bestCV[t], bestX[t], ep=Ep[t])

        gen = 1

        # =================== Vòng lặp tiến hoá ===================
        while fnceval_calls < eva_num:
            # KT cho cả main/aux (tạo main_off1{t} & aux_off1{t})
            if self.use_das and self.das_kt is not None:
                # Use DaS-based KT
                main_off1 = self._kt_with_das(main_pop, main_divK, main_divD, Tasks)
                aux_off1  = self._kt_with_das(aux_pop, aux_divK, aux_divD, Tasks)
            else:
                # Use original KT
                main_off1 = KT(self.params, [getattr(tt, 'dim', getattr(tt, 'dims', None)) for tt in Tasks], main_pop, divK=main_divK, divD=main_divD)
                aux_off1  = KT(self.params, [getattr(tt, 'dim', getattr(tt, 'dims', None)) for tt in Tasks], aux_pop,  divK=aux_divK,  divD=aux_divD)

            # Cập nhật epsilon động (dựa trên aux_pop của từng task)
            for t in range(len(Tasks)):
                fes = np.array([ind.constraint_violation for ind in aux_pop[t]]) <= 0
                fea_percent = float(np.sum(fes)) / len(aux_pop[t])
                if fea_percent < 1:
                    Ep[t] = float(np.max([ind.constraint_violation for ind in aux_pop[t]]))

                progress = fnceval_calls / float(eva_num)
                if progress < EC_Tc:
                    if fea_percent < EC_Alpha:
                        Ep[t] = float(Ep[t] * (1.0 - progress / EC_Tc) ** EC_Cp)
                    else:
                        Ep[t] = float(1.1 * np.max([ind.constraint_violation for ind in aux_pop[t]]))
                else:
                    Ep[t] = 0.0

            main_flag = [False] * len(Tasks)
            aux_flag  = [False]  * len(Tasks)

            # Duyệt từng task
            for t in range(len(Tasks)):
                # id quần thể trong Q/UCB: mỗi task có 2 hàng
                t_1 = num_pop_each_task * (t + 1) - 1  # index cho main (theo MATLAB: t_1)
                t_2 = num_pop_each_task * (t + 1)      # index cho aux  (theo MATLAB: t_2)
                # Điều chỉnh về 0-based Python
                t_1 -= 1
                t_2 -= 1

                # Tính UCB
                UCB_values[t_1, :] = Q_Table[t_1, :] + np.sqrt(2.0 * np.log(max(1, UCB_T)) / (action_counts[t_1, :] + varepsilon_ucb))
                UCB_values[t_2, :] = Q_Table[t_2, :] + np.sqrt(2.0 * np.log(max(1, UCB_T)) / (action_counts[t_2, :] + varepsilon_ucb))

                # Chọn action
                a1 = int(np.argmax(UCB_values[t_1, :]))  # cho main
                a2 = int(np.argmax(UCB_values[t_2, :]))  # cho aux

                # Đếm action
                action_counts[t_1, a1] += 1.0
                action_counts[t_2, a2] += 1.0

                # Sinh off2 theo action
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

                # Gộp off1 + off2 rồi evaluate
                main_off = main_off1[t] + main_off2
                aux_off  = aux_off1[t]  + aux_off2
                main_off, calls = evaluate_population(main_off, Tasks[t])
                fnceval_calls += calls
                aux_off, calls = evaluate_population(aux_off, Tasks[t])
                fnceval_calls += calls

                # Selection
                main_pop[t], main_rank, bestobj[t], bestCV[t], bestX[t], main_flag[t] = select_mp(
                    main_pop[t], main_off, bestobj[t], bestCV[t], bestX[t], ep=0.0
                )
                # Thêm bước select với [main_off, aux_off] như MATLAB
                main_pop[t], _, bestobj[t], bestCV[t], bestX[t], _ = select_mp(
                    main_pop[t], (main_off + aux_off), bestobj[t], bestCV[t], bestX[t], ep=0.0
                )
                aux_pop[t], aux_rank, _, _, _, aux_flag[t] = select_mp(
                    aux_pop[t], aux_off, bestobj[t], bestCV[t], bestX[t], ep=Ep[t]
                )

                # Tính tỉ lệ thành công để cập nhật Q (theo MATLAB)
                # main_next/aux_next là vector bool đánh dấu các cá thể được giữ lại
                main_next = np.zeros(len(main_rank), dtype=bool)
                aux_next  = np.zeros(len(aux_rank),  dtype=bool)
                main_next[ main_rank[:len(main_pop[t])] ] = True
                aux_next[  aux_rank[:len(aux_pop[t])]  ] = True

                # lấy phần chỉ thuộc offspring mới ở cuối (loại bỏ phần cũ + off1)
                main_tail = main_next[len(main_pop[t]) + len(main_off1[t]):]
                aux_tail  = aux_next[len(aux_pop[t])   + len(aux_off1[t]):]
                # tránh chia 0
                denom_main = max(1, len(main_pop[t]) + len(main_off2))
                denom_aux  = max(1, len(aux_pop[t])  + len(aux_off2))

                main_succ_rate = float(np.sum(main_tail)) / float(denom_main)
                aux_succ_rate  = float(np.sum(aux_tail))  / float(denom_aux)

                # Q-learning update
                Q_Table[t_1, a1] = Q_Table[t_1, a1] + alpha_ql * (main_succ_rate + gamma_ql * np.max(Q_Table[t_1, :]) - Q_Table[t_1, a1])
                Q_Table[t_2, a2] = Q_Table[t_2, a2] + alpha_ql * (aux_succ_rate  + gamma_ql * np.max(Q_Table[t_2, :]) - Q_Table[t_2, a2])

            # Cập nhật divD/divK theo cờ thành công (HEURISTIC)
            main_divD, main_divK = update_divd_divk(main_flag, main_divD, main_divK, maxD, minK, maxK, rng=self.rng)
            aux_divD,  aux_divK  = update_divd_divk(aux_flag,  aux_divD,  aux_divK,  maxD, minK, maxK, rng=self.rng)
            
            # Update DaS generation counter
            if self.use_das and self.das_kt is not None:
                self.das_kt.next_generation()

            # Ghi convergence
            gen += 1
            bestobj_arr = np.array(bestobj, dtype=float)
            bestcv_arr  = np.array(bestCV,  dtype=float)
            convergence      = np.column_stack([convergence,      bestobj_arr])
            convergence_cv   = np.column_stack([convergence_cv,   bestcv_arr])

        data["convergence"]    = gen2eva(convergence)
        data["convergence_cv"] = gen2eva(convergence_cv)
        data["bestX"]          = uni2real(bestX, Tasks)
        return data
    
    def _kt_with_das(self, pops, divK, divD, Tasks):
        """Perform Knowledge Transfer with DaS-based dimension selection.
        
        Args:
            pops: List of populations (one per task)
            divK: Number of clusters for K-means
            divD: Number of dimensions to transfer  
            Tasks: Task objects (for dimension info)
            
        Returns:
            List of offspring populations (one list per task)
        """
        from KT import KT
        
        # Standard KT call - we'll use original KT but track which dimensions matter
        # For now, use original KT and compute rewards for learning
        offspring = KT(self.params, 
                      [getattr(tt, 'dim', getattr(tt, 'dims', None)) for tt in Tasks],
                      pops, divK=divK, divD=divD, rng=self.rng)
        
        # Compute rewards based on offspring quality
        # Reward = improvement rate (how many offspring are better than worst parent)
        for dst_task in range(len(pops)):
            if len(offspring[dst_task]) == 0:
                continue
                
            # Count improved offspring
            parent_costs = [ind.factorial_costs for ind in pops[dst_task] if ind.factorial_costs is not None]
            if not parent_costs:
                continue
            
            worst_parent = max(parent_costs)
            # Note: offspring not yet evaluated, so we can't compute actual reward here
            # We'll need to compute this after evaluation
            # For now, just use uniform transfer (DaS will learn in future iterations)
            
        return offspring